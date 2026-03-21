#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
构建回归面板: 合并 entropy指标 + 职业聚类 + 行业分类
输出三个文件:
1. job_panel.csv  — 岗位级面板 (id, year, cluster_id, ..., industry_code, ..., metrics...)
2. occ_year_panel.csv — 职业×年份 聚合面板
3. ind_occ_year_panel.csv — 行业×职业×年份 聚合面板
"""

import csv
import json
import time
import numpy as np
from pathlib import Path
from collections import defaultdict

OUTPUT_DIR = Path("/Users/yu/code/code2601/TY/output")
RESULTS_CSV = OUTPUT_DIR / "final_results_sample.csv"
CLUSTER_MAP = OUTPUT_DIR / "clusters" / "job_cluster_map_clean.csv"
CLUSTER_NAMES = OUTPUT_DIR / "clusters" / "cluster_names.json"
INDUSTRY_MAP = OUTPUT_DIR / "clusters" / "company_industry_map.csv"

JOB_PANEL = OUTPUT_DIR / "job_panel.csv"
OCC_YEAR_PANEL = OUTPUT_DIR / "occ_year_panel.csv"
IND_OCC_YEAR_PANEL = OUTPUT_DIR / "ind_occ_year_panel.csv"

METRICS = ["entropy_score", "hhi_score", "dominant_topic_prob",
           "ent_effective", "rao_q", "gini", "n_sig_topics", "tail_mass_ratio"]


def write_agg_panel(filepath, agg, key_cols, key_labels_fn, description):
    """Write an aggregated panel CSV from an agg dict."""
    print(f"  Aggregating {description}...")
    with open(filepath, 'w', encoding='utf-8-sig', newline='') as f:
        writer = csv.writer(f)
        header = key_cols + ["n_jobs"]
        for m in METRICS:
            header.extend([f"mean_{m}", f"median_{m}", f"sd_{m}"])
        writer.writerow(header)

        for key in sorted(agg.keys()):
            data = agg[key]
            labels = key_labels_fn(key)
            n_jobs = len(data[METRICS[0]]) if METRICS[0] in data else 0

            row_out = list(labels) + [n_jobs]
            for m in METRICS:
                arr = np.array(data.get(m, []))
                if len(arr) > 0:
                    row_out.extend([f"{np.mean(arr):.6f}", f"{np.median(arr):.6f}", f"{np.std(arr):.6f}"])
                else:
                    row_out.extend(["", "", ""])
            writer.writerow(row_out)

    n_cells = len(agg)
    print(f"  {description}: {n_cells:,} cells -> {filepath}")


def main():
    t0 = time.time()

    # 1. Load cluster names
    with open(CLUSTER_NAMES) as f:
        cdata = json.load(f)
    noise_ids = {int(k) for k, v in cdata['clusters'].items() if v.get('noise')}
    noise_ids.add(-1)
    occ_names = {}
    for k, v in cdata['clusters'].items():
        occ_names[int(k)] = v

    # 2. Load cluster map: row_idx -> cluster info
    print("[1/5] Loading cluster map...")
    cluster_map = {}  # id (str) -> (cluster_id, cluster_name, major_class)
    with open(CLUSTER_MAP, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rid = str(int(row['row_idx']) + 1)  # row_idx+1 = id
            cluster_map[rid] = (
                int(row['cluster_id']),
                row['cluster_name'],
                row['major_class']
            )
    print(f"  Loaded {len(cluster_map):,} cluster mappings")

    # 3. Load industry map: row_idx -> (industry_code, industry_name)
    print("[2/5] Loading industry map...")
    industry_map = {}  # id (str) -> (industry_code, industry_name)
    with open(INDUSTRY_MAP, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rid = str(int(row['row_idx']) + 1)
            icode = row['industry_code']
            iname = row['industry_name']
            if icode:  # skip unmatched
                industry_map[rid] = (icode, iname)
    print(f"  Loaded {len(industry_map):,} industry mappings")

    # 4. Stream results + join -> job panel + accumulate for agg panels
    print("[3/5] Joining results with clusters & industries...")
    agg_oy = defaultdict(lambda: defaultdict(list))   # (cluster_id, year)
    agg_ioy = defaultdict(lambda: defaultdict(list))  # (ind_code, cluster_id, year)

    matched = 0
    unmatched_occ = 0
    unmatched_ind = 0
    both_matched = 0

    with open(RESULTS_CSV, 'r', encoding='utf-8-sig') as fin, \
         open(JOB_PANEL, 'w', encoding='utf-8-sig', newline='') as fout:
        reader = csv.DictReader(fin)
        writer = csv.writer(fout)
        writer.writerow([
            "id", "year",
            "cluster_id", "cluster_name", "major_class",
            "industry_code", "industry_name",
            *METRICS
        ])

        for i, row in enumerate(reader):
            jid = row['id']
            year = row['year']

            occ_info = cluster_map.get(jid)
            if occ_info is None:
                unmatched_occ += 1
                continue

            cid, cname, mclass = occ_info
            matched += 1

            ind_info = industry_map.get(jid)
            icode = ind_info[0] if ind_info else ""
            iname = ind_info[1] if ind_info else ""
            if ind_info:
                both_matched += 1
            else:
                unmatched_ind += 1

            vals = [row.get(m, '') for m in METRICS]
            writer.writerow([jid, year, cid, cname, mclass, icode, iname, *vals])

            # Accumulate for occ×year
            key_oy = (cid, year)
            for mi, m in enumerate(METRICS):
                try:
                    v = float(vals[mi])
                    agg_oy[key_oy][m].append(v)
                    # Accumulate for ind×occ×year (only if industry matched)
                    if icode:
                        agg_ioy[(icode, cid, year)][m].append(v)
                except (ValueError, IndexError):
                    pass

            if (i + 1) % 5_000_000 == 0:
                print(f"  {i+1:,} rows processed...", flush=True)

    print(f"  Occ matched: {matched:,}, Occ unmatched: {unmatched_occ:,}")
    print(f"  Both occ+ind matched: {both_matched:,}, Ind unmatched: {unmatched_ind:,}")
    print(f"  Job panel -> {JOB_PANEL}")

    # 5. Write aggregated panels
    print("[4/5] Writing occ × year panel...")
    write_agg_panel(
        OCC_YEAR_PANEL, agg_oy,
        key_cols=["cluster_id", "cluster_name", "major_class", "year"],
        key_labels_fn=lambda k: (k[0], occ_names.get(k[0], {}).get('name', ''),
                                  occ_names.get(k[0], {}).get('大类', ''), k[1]),
        description="occ × year panel"
    )

    # Industry name lookup
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

    print("[5/5] Writing industry × occ × year panel...")
    write_agg_panel(
        IND_OCC_YEAR_PANEL, agg_ioy,
        key_cols=["industry_code", "industry_name", "cluster_id", "cluster_name", "major_class", "year"],
        key_labels_fn=lambda k: (k[0], IND_NAMES.get(k[0], ''),
                                  k[1], occ_names.get(k[1], {}).get('name', ''),
                                  occ_names.get(k[1], {}).get('大类', ''), k[2]),
        description="industry × occ × year panel"
    )

    # Summary
    n_ind = len(set(k[0] for k in agg_ioy.keys()))
    n_occ = len(set(k[1] for k in agg_ioy.keys()))
    n_yr = len(set(k[2] for k in agg_ioy.keys()))
    print(f"\n  Industry × Occ × Year: {n_ind} industries × {n_occ} occupations × {n_yr} years = {len(agg_ioy):,} cells")

    elapsed = time.time() - t0
    print(f"\nDone! Total time: {elapsed:.0f}s")


if __name__ == "__main__":
    main()
