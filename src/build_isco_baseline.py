#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
构建 ISCO 级别的基线职业特征 (用于 conditional parallel trends)

从 merged_1_6.csv 提取:
  1. 平均薪资 (mean_salary) — (最低+最高)/2
  2. 教育水平 (edu_level) — 学历要求编码为 1-7 的数值
  3. 经验年限 (exp_years) — 要求经验编码为年数

按 ISCO × year 聚合, 同时计算 baseline (2016-2018均值) 作为时不变特征。
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
CLUSTER_MAP_CSV = BASE / 'output' / 'clusters' / 'job_cluster_map_clean.csv'
EXPOSURE_CSV = BASE / 'output' / 'clusters' / 'ai_exposure_ilo.csv'

OUTPUT = BASE / 'output' / 'isco_baseline_features.csv'

NOISE_CLUSTERS = {5, 6, 9, 43, 51, 74, -1}

# 学历编码 (数值越大越高)
EDU_MAP = {
    '初中': 1, '初中及以下': 1,
    '中专': 2, '中技': 2, '中专/中技': 2, '中专,技校': 2, '技校': 2, '高中': 2, '高中/中专/中技': 2,
    '大专': 3, '大专以上': 3, '大专及以上': 3,
    '本科': 4, '本科以上': 4, '本科及以上': 4, '统招本科': 4, '全日制本科': 4,
    '硕士': 5, '硕士以上': 5, '硕士及以上': 5, '研究生': 5,
    '博士': 6, '博士以上': 6, '博士及以上': 6,
    'MBA': 5, 'EMBA': 5,
}

def parse_exp_years(s):
    """将经验要求转为年数"""
    s = s.strip()
    if not s or s in ('不限', '经验不限', '无要求'):
        return None
    if s in ('应届毕业生', '无工作经验', '无经验', '在校生'):
        return 0.0
    # "3-5年" → 4, "1-3年" → 2
    m = re.match(r'(\d+)-(\d+)年', s)
    if m:
        return (int(m.group(1)) + int(m.group(2))) / 2
    # "3年以上" → 3, "5年以上" → 5
    m = re.match(r'(\d+)年以上', s)
    if m:
        return float(m.group(1))
    # "1年以下" → 0.5
    m = re.match(r'(\d+)年以下', s)
    if m:
        return float(m.group(1)) / 2
    # "3年" → 3
    m = re.match(r'(\d+)年', s)
    if m:
        return float(m.group(1))
    return None


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

    # Load ILO scores
    ilo_scores = set()
    with open(ILO_CSV, 'r', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            ilo_scores.add(row['isco08_4digit'].strip())

    # Load SBERT lookup
    print("[1/4] Loading SBERT ISCO lookup...")
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

    # Load cluster → dominant ISCO fallback
    print("[2/4] Loading cluster fallback...")
    cluster_dominant_isco = {}
    with open(EXPOSURE_CSV, 'r', encoding='utf-8-sig') as f:
        for row in csv.DictReader(f):
            cid = int(row['cluster_id'])
            top_str = row.get('top_isco_codes', '')
            if top_str:
                m = re.match(r'(\d{4})', top_str)
                if m:
                    cluster_dominant_isco[cid] = m.group(1)

    cluster_map = {}
    with open(CLUSTER_MAP_CSV, 'r', encoding='utf-8-sig') as f:
        for row in csv.DictReader(f):
            cluster_map[int(row['row_idx'])] = int(row['cluster_id'])

    # Stream merged_1_6.csv, collect features by ISCO × year
    print("[3/4] Streaming merged_1_6.csv...")
    # (isco, year) → {'salary': [], 'edu': [], 'exp': []}
    agg = defaultdict(lambda: {'salary': [], 'edu': [], 'exp': []})

    matched = 0
    with open(MERGED_CSV, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            year = row.get('招聘发布年份', '').strip()
            if not year:
                continue
            try:
                year = int(float(year))
            except:
                continue

            title = row.get('招聘岗位', '').strip()

            # 3-level ISCO matching
            isco = raw_isco.get(title)
            if not isco:
                ct = clean_title(title)
                isco = cleaned_isco.get(ct) if ct and len(ct) >= 2 else None
            if not isco:
                # Keyword override
                if '翻译' in title:
                    isco = '2643'
            if not isco:
                cid = cluster_map.get(i)
                if cid is not None and cid not in NOISE_CLUSTERS:
                    isco = cluster_dominant_isco.get(cid)

            if not isco:
                continue

            key = (isco, year)
            matched += 1

            # Salary
            try:
                lo = float(row.get('最低月薪', ''))
                hi = float(row.get('最高月薪', ''))
                if 500 <= lo <= 200000 and 500 <= hi <= 200000:
                    agg[key]['salary'].append((lo + hi) / 2)
            except:
                pass

            # Education
            edu_str = row.get('学历要求', '').strip()
            edu_val = EDU_MAP.get(edu_str)
            if edu_val is not None:
                agg[key]['edu'].append(edu_val)

            # Experience
            exp_str = row.get('要求经验', '').strip()
            exp_val = parse_exp_years(exp_str)
            if exp_val is not None:
                agg[key]['exp'].append(exp_val)

            if (i + 1) % 5_000_000 == 0:
                elapsed = time.time() - t0
                print(f"  {i+1:,} rows... matched={matched:,} [{elapsed:.0f}s]")

    print(f"  Total matched: {matched:,}")
    print(f"  ISCO × year cells: {len(agg):,}")

    # Write output
    print("[4/4] Writing baseline features...")
    with open(OUTPUT, 'w', encoding='utf-8-sig', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['isco08_4digit', 'year', 'mean_salary', 'mean_edu', 'mean_exp',
                          'n_salary', 'n_edu', 'n_exp'])
        for (isco, year) in sorted(agg.keys()):
            d = agg[(isco, year)]
            row_out = [isco, year]
            for feat in ['salary', 'edu', 'exp']:
                arr = d[feat]
                if arr:
                    row_out.extend([f"{np.mean(arr):.2f}", len(arr)])
                else:
                    row_out.extend(["", 0])
            writer.writerow(row_out)

    print(f"  Saved to {OUTPUT}")

    # Quick stats
    all_sal = [np.mean(v['salary']) for v in agg.values() if v['salary']]
    all_edu = [np.mean(v['edu']) for v in agg.values() if v['edu']]
    all_exp = [np.mean(v['exp']) for v in agg.values() if v['exp']]
    print(f"\n  Salary: mean={np.mean(all_sal):.0f}, range=[{np.min(all_sal):.0f}, {np.max(all_sal):.0f}]")
    print(f"  Education: mean={np.mean(all_edu):.2f}, range=[{np.min(all_edu):.2f}, {np.max(all_edu):.2f}]")
    print(f"  Experience: mean={np.mean(all_exp):.2f}, range=[{np.min(all_exp):.2f}, {np.max(all_exp):.2f}]")
    print(f"\n  Total time: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
