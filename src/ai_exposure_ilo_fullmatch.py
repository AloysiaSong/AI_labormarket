#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AI Exposure — 全量交叉匹配版

直接读 merged_1_6.csv (27M行)，每行同时查：
1. 原始 招聘岗位 → sbert_isco_lookup (raw title → ISCO) → ILO score
2. row_idx → job_cluster_map_clean (row_idx → cluster_id)

两边都是原始格式，不存在清洗不匹配的问题。
"""

import csv
import json
import time
import numpy as np
from pathlib import Path
from collections import defaultdict

BASE = Path('/Users/yu/code/code2601/TY')
MERGED_CSV = BASE / 'dataset' / 'merged_1_6.csv'
SBERT_LOOKUP = BASE / 'data' / 'Heterogeneity' / 'sbert_isco_lookup.csv'
ILO_CSV = BASE / 'data' / 'esco' / 'ilo_genai_isco08_2025.csv'
CLUSTER_MAP = BASE / 'output' / 'clusters' / 'job_cluster_map_clean.csv'
CLUSTER_NAMES = BASE / 'output' / 'clusters' / 'cluster_names.json'

EXPOSURE_CSV = BASE / 'output' / 'clusters' / 'ai_exposure_ilo.csv'
NOISE_CLUSTERS = {5, 6, 9, 43, 51, 74}

# 手动覆盖：SBERT 系统性匹配错误的关键词 → 正确 ISCO
# 翻译岗被 SBERT 匹配到语言教师(2353)而非翻译员(2643)
KEYWORD_ISCO_OVERRIDES = [
    ('翻译', '2643'),  # Translators, Interpreters → ILO 0.59
]


def main():
    t0 = time.time()

    # 1. Load ILO: ISCO → score
    print("[1/4] Loading ILO GenAI exposure data...")
    ilo_scores = {}
    with open(ILO_CSV, 'r', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            ilo_scores[row['isco08_4digit'].strip()] = float(row['mean_score'])
    print(f"  {len(ilo_scores)} ISCO codes")

    # 2. Load SBERT lookup: raw title → ISCO
    print("[2/4] Loading SBERT ISCO lookup (raw titles)...")
    title_isco = {}
    with open(SBERT_LOOKUP, 'r', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            if row.get('sbert_accepted', 'True') == 'True':
                isco = row['isco08_4digit'].strip()
                if isco in ilo_scores:
                    title_isco[row['title'].strip()] = isco
    print(f"  {len(title_isco):,} raw titles with valid ISCO→ILO mapping")

    # 3. Load cluster map: row_idx → cluster_id
    print("[3/4] Loading cluster map...")
    cluster_map = {}
    with open(CLUSTER_MAP, 'r', encoding='utf-8-sig') as f:
        for row in csv.DictReader(f):
            cluster_map[int(row['row_idx'])] = int(row['cluster_id'])
    print(f"  {len(cluster_map):,} rows → clusters")

    # 4. Stream merged_1_6.csv, cross-match
    print("[4/4] Streaming merged_1_6.csv for cross-matching...")
    cluster_scores = defaultdict(list)  # cid → [score, score, ...]
    cluster_isco_dist = defaultdict(lambda: defaultdict(int))

    matched_both = 0
    has_cluster_no_isco = 0
    has_isco_no_cluster = 0
    neither = 0

    with open(MERGED_CSV, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            title = row.get('招聘岗位', '').strip()
            cid = cluster_map.get(i)
            isco = title_isco.get(title)

            # 关键词覆盖：修正 SBERT 系统性误匹配
            if isco is None or isco in ('2353', '2330'):  # 语言教师/教授
                for kw, correct_isco in KEYWORD_ISCO_OVERRIDES:
                    if kw in title:
                        isco = correct_isco
                        break

            if cid is not None and cid not in NOISE_CLUSTERS and isco is not None:
                score = ilo_scores[isco]
                cluster_scores[cid].append(score)
                cluster_isco_dist[cid][isco] += 1
                matched_both += 1
            elif cid is not None and isco is None:
                has_cluster_no_isco += 1
            elif cid is None and isco is not None:
                has_isco_no_cluster += 1
            else:
                neither += 1

            if (i + 1) % 5_000_000 == 0:
                elapsed = time.time() - t0
                print(f"  {i+1:,} rows... ({elapsed:.0f}s) "
                      f"matched={matched_both:,}, cluster_only={has_cluster_no_isco:,}")

    total = matched_both + has_cluster_no_isco + has_isco_no_cluster + neither
    print(f"\n  Total rows: {total:,}")
    print(f"  Both matched (cluster + ISCO): {matched_both:,} ({matched_both/total*100:.1f}%)")
    print(f"  Cluster only (no ISCO): {has_cluster_no_isco:,}")
    print(f"  ISCO only (no cluster): {has_isco_no_cluster:,}")
    print(f"  Neither: {neither:,}")
    print(f"  Clusters with scores: {len(cluster_scores)}")

    # Load cluster names
    with open(CLUSTER_NAMES) as f:
        cdata = json.load(f)

    # Compute per-cluster averages
    results = []
    for cid_str, info in sorted(cdata['clusters'].items(), key=lambda x: int(x[0])):
        cid = int(cid_str)
        if cid in NOISE_CLUSTERS:
            continue

        scores = cluster_scores.get(cid, [])
        if not scores:
            print(f"  WARNING: C{cid:02d} ({info['name']}) has no ILO scores!")
            results.append((cid, info['name'], info['大类'], np.nan, 0, 0, ""))
            continue

        arr = np.array(scores)
        mean_score = arr.mean()
        sd_score = arr.std()
        n = len(scores)

        # Top ISCO contributions
        isco_dist = cluster_isco_dist[cid]
        top_iscos = sorted(isco_dist.items(), key=lambda x: -x[1])[:3]
        top_str = "; ".join(f"{isco}({ilo_scores[isco]:.2f}, n={cnt:,})" for isco, cnt in top_iscos)

        results.append((cid, info['name'], info['大类'], mean_score, sd_score, n, top_str))

    # Write CSV
    with open(EXPOSURE_CSV, 'w', encoding='utf-8-sig', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["cluster_id", "cluster_name", "major_class",
                         "ai_exposure_ilo", "ai_exposure_sd", "n_matched_rows", "top_isco_codes"])
        for r in results:
            cid, name, mc, score, sd, n, top = r
            writer.writerow([cid, name, mc,
                             f"{score:.4f}" if not np.isnan(score) else "",
                             f"{sd:.4f}" if not np.isnan(score) else "",
                             n, top])

    print(f"\nWritten to {EXPOSURE_CSV}")

    # Summary
    valid = [r[3] for r in results if not np.isnan(r[3])]
    arr = np.array(valid)
    print(f"\nILO AI Exposure across {len(valid)} clusters:")
    print(f"  Mean={arr.mean():.4f}, Median={np.median(arr):.4f}, "
          f"Min={arr.min():.4f}, Max={arr.max():.4f}")
    print(f"  Q25={np.percentile(arr, 25):.4f}, Q75={np.percentile(arr, 75):.4f}")

    ranked = [(r[0], r[1], r[3], r[5]) for r in results if not np.isnan(r[3])]
    ranked.sort(key=lambda x: -x[2])
    print(f"\nTop 10:")
    for cid, name, score, n in ranked[:10]:
        print(f"  C{cid:02d} {name}: {score:.4f} (n={n:,})")
    print(f"\nBottom 10:")
    for cid, name, score, n in ranked[-10:]:
        print(f"  C{cid:02d} {name}: {score:.4f} (n={n:,})")

    print(f"\nTotal time: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
