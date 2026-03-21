#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
构建 ISCO-08 粒度面板 (替代 cluster 粒度)

匹配策略 (三层):
  1. Raw title → SBERT lookup → ISCO (精确匹配)
  2. Cleaned title → SBERT lookup → ISCO (清洗后匹配)
  3. Fallback: row_idx → cluster_id → cluster 主ISCO code (聚类近似)

输出: ind_isco_year_panel.csv
每个 ISCO 有 ILO 直接给出的 exposure 分数, 无聚合噪声
"""

import csv
import re
import time
import numpy as np
from pathlib import Path
from collections import defaultdict

BASE = Path('/Users/yu/code/code2601/TY')
MERGED_CSV = BASE / 'dataset' / 'merged_1_6.csv'
SBERT_LOOKUP = BASE / 'data' / 'Heterogeneity' / 'sbert_isco_lookup.csv'
ILO_CSV = BASE / 'data' / 'esco' / 'ilo_genai_isco08_2025.csv'
RESULTS_CSV = BASE / 'output' / 'final_results_sample.csv'
INDUSTRY_MAP_CSV = BASE / 'output' / 'clusters' / 'company_industry_map.csv'
CLUSTER_MAP_CSV = BASE / 'output' / 'clusters' / 'job_cluster_map_clean.csv'
EXPOSURE_CSV = BASE / 'output' / 'clusters' / 'ai_exposure_ilo.csv'

OUTPUT_DIR = BASE / 'output'
ISCO_PANEL = OUTPUT_DIR / 'ind_isco_year_panel.csv'

METRICS = ["entropy_score", "hhi_score", "dominant_topic_prob",
           "ent_effective", "rao_q", "gini", "n_sig_topics", "tail_mass_ratio"]

KEYWORD_ISCO_OVERRIDES = [
    ('翻译', '2643'),
]

NOISE_CLUSTERS = {5, 6, 9, 43, 51, 74, -1}

IND_NAMES = {
    "A": "农林牧渔业", "B": "采矿业", "C": "制造业",
    "D": "电力热力燃气及水", "E": "建筑业", "F": "批发和零售业",
    "G": "交通运输仓储和邮政业", "H": "住宿和餐饮业",
    "I": "信息传输软件和信息技术服务业", "J": "金融业",
    "K": "房地产业", "L": "租赁和商务服务业",
    "M": "科学研究和技术服务业", "N": "水利环境公共设施",
    "O": "居民服务修理和其他服务业", "P": "教育",
    "Q": "卫生和社会工作", "R": "文化体育和娱乐业",
    "S": "公共管理社会保障",
}


def clean_title(t):
    t = re.sub(r'[（(][^)）]*[)）]', '', t)
    t = re.sub(r'\d{3,}元?/?', '', t)
    t = re.sub(r'[+➕].{0,20}$', '', t)
    t = re.sub(r'/[包入无].{0,15}$', '', t)
    t = re.sub(r'(急聘|诚聘|高薪|急招|高新|直招|包吃住|包食宿|五险一金|双休|转正)', '', t)
    t = t.strip(' /\\-—_·.，,')
    return t


def main():
    t0 = time.time()

    # ── 1. Load ILO scores ──
    print("[1/6] Loading ILO GenAI exposure scores...")
    ilo_scores = {}
    ilo_names = {}
    with open(ILO_CSV, 'r', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            isco = row['isco08_4digit'].strip()
            ilo_scores[isco] = float(row['mean_score'])
            ilo_names[isco] = row.get('occupation_name_en', isco)
    print(f"  {len(ilo_scores)} ISCO codes")

    # ── 2. Load SBERT lookup (raw + cleaned) ──
    print("[2/6] Loading SBERT ISCO lookup...")
    raw_isco = {}
    cleaned_isco = {}
    with open(SBERT_LOOKUP, 'r', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            if row.get('sbert_accepted', 'True') == 'True':
                isco = row['isco08_4digit'].strip()
                if isco in ilo_scores:
                    title = row['title'].strip()
                    raw_isco[title] = isco
                    ct = clean_title(title)
                    if ct and len(ct) >= 2 and ct not in cleaned_isco:
                        cleaned_isco[ct] = isco
    print(f"  Raw: {len(raw_isco):,}, Cleaned: {len(cleaned_isco):,}")

    # ── 3. Build cluster → dominant ISCO fallback ──
    print("[3/6] Building cluster → dominant ISCO fallback...")
    cluster_dominant_isco = {}  # cluster_id → ISCO code (most common)
    with open(EXPOSURE_CSV, 'r', encoding='utf-8-sig') as f:
        for row in csv.DictReader(f):
            cid = int(row['cluster_id'])
            top_str = row.get('top_isco_codes', '')
            if top_str:
                # Parse "7213(0.21, n=6,599); 2144(0.32, n=5,398); ..."
                match = re.match(r'(\d{4})', top_str)
                if match:
                    cluster_dominant_isco[cid] = match.group(1)
    print(f"  {len(cluster_dominant_isco)} clusters with dominant ISCO")

    # Load cluster map: row_idx → cluster_id
    cluster_map = {}
    with open(CLUSTER_MAP_CSV, 'r', encoding='utf-8-sig') as f:
        for row in csv.DictReader(f):
            cluster_map[int(row['row_idx'])] = int(row['cluster_id'])
    print(f"  {len(cluster_map):,} rows → clusters")

    # ── 4. Stream merged_1_6.csv → row_idx → ISCO mapping ──
    print("[4/6] Building row_idx → ISCO mapping (3-level matching)...")
    isco_map = {}
    match_stats = {'raw': 0, 'cleaned': 0, 'cluster_fallback': 0, 'miss': 0}

    with open(MERGED_CSV, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            title = row.get('招聘岗位', '').strip()
            isco = None
            method = None

            # Level 1: Raw title match
            isco = raw_isco.get(title)
            if isco:
                method = 'raw'
            else:
                # Level 2: Cleaned title match
                ct = clean_title(title)
                isco = cleaned_isco.get(ct) if ct and len(ct) >= 2 else None
                if isco:
                    method = 'cleaned'

            # Keyword overrides (fix systematic SBERT mismatches)
            if isco is None or isco in ('2353', '2330'):
                for kw, correct_isco in KEYWORD_ISCO_OVERRIDES:
                    if kw in title:
                        isco = correct_isco
                        method = method or 'cleaned'
                        break

            # Level 3: Cluster fallback
            if isco is None:
                cid = cluster_map.get(i)
                if cid is not None and cid not in NOISE_CLUSTERS:
                    fallback_isco = cluster_dominant_isco.get(cid)
                    if fallback_isco:
                        isco = fallback_isco
                        method = 'cluster_fallback'

            if isco is not None:
                isco_map[i] = isco
                match_stats[method] += 1
            else:
                match_stats['miss'] += 1

            if (i + 1) % 5_000_000 == 0:
                elapsed = time.time() - t0
                total_matched = sum(v for k, v in match_stats.items() if k != 'miss')
                print(f"  {i+1:,} rows... matched={total_matched:,} "
                      f"({total_matched/(i+1)*100:.1f}%) [{elapsed:.0f}s]")

    total_merged = i + 1
    total_matched = sum(v for k, v in match_stats.items() if k != 'miss')
    print(f"\n  Total: {total_merged:,} rows")
    print(f"  Matched: {total_matched:,} ({total_matched/total_merged*100:.1f}%)")
    for method, count in match_stats.items():
        print(f"    {method}: {count:,} ({count/total_merged*100:.1f}%)")

    # ── 5. Load industry map ──
    print("[5/6] Loading industry map...")
    industry_map = {}
    with open(INDUSTRY_MAP_CSV, 'r', encoding='utf-8-sig') as f:
        for row in csv.DictReader(f):
            rid = str(int(row['row_idx']) + 1)
            icode = row['industry_code']
            if icode:
                industry_map[rid] = icode
    print(f"  {len(industry_map):,} rows with industry")

    # ── 6. Stream final_results_sample.csv → aggregate ──
    print("[6/6] Building industry × ISCO × year panel...")
    agg = defaultdict(lambda: defaultdict(list))
    # Also track match method per cell for diagnostics
    cell_methods = defaultdict(lambda: defaultdict(int))

    matched = 0
    no_isco = 0
    no_ind = 0

    with open(RESULTS_CSV, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            jid = row['id']
            year = row['year']
            row_idx = int(jid) - 1

            isco = isco_map.get(row_idx)
            if isco is None:
                no_isco += 1
                continue

            icode = industry_map.get(jid)
            if icode is None:
                no_ind += 1
                continue

            key = (icode, isco, year)
            for m in METRICS:
                try:
                    v = float(row.get(m, ''))
                    agg[key][m].append(v)
                except (ValueError, TypeError):
                    pass
            matched += 1

            if (i + 1) % 5_000_000 == 0:
                elapsed = time.time() - t0
                print(f"  {i+1:,} results... matched={matched:,} [{elapsed:.0f}s]")

    total_results = i + 1
    print(f"\n  Results total: {total_results:,}")
    print(f"  Matched (ISCO + industry): {matched:,} ({matched/total_results*100:.1f}%)")
    print(f"  No ISCO: {no_isco:,} ({no_isco/total_results*100:.1f}%)")
    print(f"  No industry: {no_ind:,} ({no_ind/total_results*100:.1f}%)")

    # ── Write panel CSV ──
    print("\nWriting panel CSV...")
    with open(ISCO_PANEL, 'w', encoding='utf-8-sig', newline='') as f:
        writer = csv.writer(f)
        header = ["industry_code", "industry_name", "isco08_4digit", "isco_name",
                  "ai_exposure_ilo", "year", "n_jobs"]
        for m in METRICS:
            header.extend([f"mean_{m}", f"median_{m}", f"sd_{m}"])
        writer.writerow(header)

        for key in sorted(agg.keys()):
            icode, isco, year = key
            data = agg[key]
            n_jobs = len(data.get(METRICS[0], []))

            row_out = [icode, IND_NAMES.get(icode, ''), isco,
                       ilo_names.get(isco, ''),
                       f"{ilo_scores[isco]:.4f}", year, n_jobs]
            for m in METRICS:
                arr = np.array(data.get(m, []))
                if len(arr) > 0:
                    row_out.extend([f"{np.mean(arr):.6f}",
                                    f"{np.median(arr):.6f}",
                                    f"{np.std(arr):.6f}"])
                else:
                    row_out.extend(["", "", ""])
            writer.writerow(row_out)

    # ── Summary ──
    n_cells = len(agg)
    n_ind = len(set(k[0] for k in agg))
    n_isco = len(set(k[1] for k in agg))
    n_yr = len(set(k[2] for k in agg))
    print(f"\n  Panel: {n_ind} industries × {n_isco} ISCO codes × {n_yr} years = {n_cells:,} cells")

    sizes = [len(agg[k].get(METRICS[0], [])) for k in agg]
    print(f"  Cell size: median={np.median(sizes):.0f}, mean={np.mean(sizes):.0f}, "
          f"min={min(sizes)}, max={max(sizes)}")

    small_cells = sum(1 for s in sizes if s < 30)
    print(f"  Cells with n < 30: {small_cells} ({small_cells/n_cells*100:.1f}%)")

    exposure_vals = list(set(ilo_scores[k[1]] for k in agg))
    print(f"  Unique exposure values in panel: {len(exposure_vals)}")
    print(f"  Exposure range: [{min(exposure_vals):.2f}, {max(exposure_vals):.2f}]")

    print(f"\n  Saved to {ISCO_PANEL}")
    print(f"  Total time: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
