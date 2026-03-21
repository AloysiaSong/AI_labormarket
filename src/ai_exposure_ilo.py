#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AI Exposure — 基于 ILO WP140 (Gmyrek et al., 2025) 真实数据

方法：
1. 加载 title2cluster.pkl（50K 岗位名 → cluster_id）
2. 加载 sbert_isco_lookup.csv（岗位名 → ISCO-08 → ILO ai_mean_score）
3. 加载 ILO 原始数据（ISCO-08 → mean_score）
4. 交叉匹配：title → cluster_id + ISCO → ILO score
5. 按 cluster 加权平均（权重 = title 频次）

输出：output/clusters/ai_exposure_ilo.csv
"""

import csv
import json
import pickle
import re
import numpy as np
from pathlib import Path
from collections import defaultdict

BASE = Path('/Users/yu/code/code2601/TY')
OUTPUT_DIR = BASE / 'output'

# Input files
TITLE2CLUSTER = OUTPUT_DIR / 'clusters' / 'title2cluster.pkl'
TITLE_FREQ = OUTPUT_DIR / 'clusters' / 'title_freq.pkl'
SBERT_LOOKUP = BASE / 'data' / 'Heterogeneity' / 'sbert_isco_lookup.csv'
ILO_CSV = BASE / 'data' / 'esco' / 'ilo_genai_isco08_2025.csv'
CLUSTER_NAMES = OUTPUT_DIR / 'clusters' / 'cluster_names.json'

# Output
EXPOSURE_CSV = OUTPUT_DIR / 'clusters' / 'ai_exposure_ilo.csv'

NOISE_CLUSTERS = {5, 6, 9, 43, 51, 74}


def clean_title(t):
    """Same cleaning as cluster_occupations.py"""
    t = re.sub(r'[（(][^)）]*[)）]', '', t)
    t = re.sub(r'\d{3,}元?/?', '', t)
    t = re.sub(r'[+➕].{0,20}$', '', t)
    t = re.sub(r'/[包入无].{0,15}$', '', t)
    t = re.sub(r'(急聘|诚聘|高薪|急招|高新|直招|包吃住|包食宿|五险一金|双休|转正)', '', t)
    t = t.strip(' /\\-—_·.，,')
    return t


def main():
    # 1. Load ILO data: ISCO-08 4digit → mean_score
    print("[1/5] Loading ILO GenAI exposure data...")
    ilo_scores = {}
    with open(ILO_CSV, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            isco = row['isco08_4digit'].strip()
            score = float(row['mean_score'])
            ilo_scores[isco] = score
    print(f"  {len(ilo_scores)} ISCO-08 occupations with ILO scores")
    print(f"  Score range: [{min(ilo_scores.values()):.2f}, {max(ilo_scores.values()):.2f}]")

    # 2. Load SBERT ISCO lookup: title → ISCO (both raw and cleaned versions)
    print("[2/5] Loading SBERT ISCO lookup...")
    title_isco = {}       # raw title → ISCO
    cleaned_isco = {}     # cleaned title → ISCO (for matching with title2cluster)
    with open(SBERT_LOOKUP, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            title = row['title'].strip()
            isco = row['isco08_4digit'].strip()
            accepted = row.get('sbert_accepted', 'True')
            if accepted == 'True' and isco in ilo_scores:
                title_isco[title] = isco
                ct = clean_title(title)
                if ct and len(ct) >= 2:
                    # Keep first match (higher similarity score)
                    if ct not in cleaned_isco:
                        cleaned_isco[ct] = isco
    print(f"  {len(title_isco):,} raw titles, {len(cleaned_isco):,} cleaned titles with valid ISCO")

    # 3. Load title → cluster mapping
    print("[3/5] Loading title → cluster mapping...")
    with open(TITLE2CLUSTER, 'rb') as f:
        title2cluster = pickle.load(f)
    print(f"  {len(title2cluster):,} titles → clusters")

    # 4. Load title frequency for weighting
    print("[4/5] Loading title frequencies...")
    with open(TITLE_FREQ, 'rb') as f:
        freq_table = pickle.load(f)
    print(f"  {len(freq_table):,} unique titles with frequencies")

    # 5. Cross-match: accumulate ILO scores by cluster
    print("[5/5] Computing cluster-level AI exposure...")
    cluster_scores = defaultdict(list)       # cluster_id → list of (score, weight)
    cluster_isco_dist = defaultdict(lambda: defaultdict(float))  # cluster_id → ISCO → weighted count

    matched = 0
    unmatched_isco = 0
    unmatched_cluster = 0

    # For each title in title2cluster, look up its ISCO and ILO score
    for title, cid in title2cluster.items():
        if cid in NOISE_CLUSTERS:
            continue
        freq = freq_table.get(title, 1)

        # Try exact match first (cleaned title), then raw title
        isco = cleaned_isco.get(title) or title_isco.get(title)
        if isco:
            score = ilo_scores[isco]
            cluster_scores[cid].append((score, freq))
            cluster_isco_dist[cid][isco] += freq
            matched += 1
        else:
            unmatched_isco += 1

    print(f"  Matched: {matched:,}, Unmatched (no ISCO): {unmatched_isco:,}")
    print(f"  Match rate: {matched / (matched + unmatched_isco) * 100:.1f}%")

    # Load cluster names
    with open(CLUSTER_NAMES) as f:
        cdata = json.load(f)

    # Compute weighted averages
    results = []
    for cid_str, info in sorted(cdata['clusters'].items(), key=lambda x: int(x[0])):
        cid = int(cid_str)
        if cid in NOISE_CLUSTERS:
            continue

        entries = cluster_scores.get(cid, [])
        if not entries:
            print(f"  WARNING: cluster {cid} ({info['name']}) has no ILO scores!")
            results.append((cid, info['name'], info['大类'], np.nan, 0, 0, ""))
            continue

        scores = np.array([s for s, _ in entries])
        weights = np.array([w for _, w in entries])
        total_weight = weights.sum()

        wmean = np.average(scores, weights=weights)
        wsd = np.sqrt(np.average((scores - wmean) ** 2, weights=weights))
        median = np.median(scores)
        n_titles = len(entries)

        # Top ISCO contributions
        isco_dist = cluster_isco_dist[cid]
        top_iscos = sorted(isco_dist.items(), key=lambda x: -x[1])[:3]
        top_isco_str = "; ".join(f"{isco}({ilo_scores[isco]:.2f})" for isco, _ in top_iscos)

        results.append((cid, info['name'], info['大类'], wmean, wsd, n_titles, top_isco_str))

    # Write output
    with open(EXPOSURE_CSV, 'w', encoding='utf-8-sig', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["cluster_id", "cluster_name", "major_class",
                         "ai_exposure_ilo", "ai_exposure_sd", "n_matched_titles", "top_isco_codes"])
        for r in results:
            cid, name, mc, score, sd, n, top = r
            writer.writerow([cid, name, mc,
                             f"{score:.4f}" if not np.isnan(score) else "",
                             f"{sd:.4f}" if not np.isnan(score) else "",
                             n, top])

    print(f"\nWritten to {EXPOSURE_CSV}")

    # Summary
    valid_scores = [r[3] for r in results if not np.isnan(r[3])]
    arr = np.array(valid_scores)
    print(f"\nILO-based AI Exposure distribution across {len(valid_scores)} clusters:")
    print(f"  Mean: {arr.mean():.4f}, Median: {np.median(arr):.4f}")
    print(f"  Min: {arr.min():.4f}, Max: {arr.max():.4f}")
    print(f"  Q25: {np.percentile(arr, 25):.4f}, Q75: {np.percentile(arr, 75):.4f}")

    # Top 10 and Bottom 10
    ranked = [(r[0], r[1], r[3]) for r in results if not np.isnan(r[3])]
    ranked.sort(key=lambda x: -x[2])
    print(f"\nTop 10 AI-exposed clusters (ILO):")
    for cid, name, score in ranked[:10]:
        print(f"  C{cid:02d} {name}: {score:.4f}")
    print(f"\nBottom 10 AI-exposed clusters (ILO):")
    for cid, name, score in ranked[-10:]:
        print(f"  C{cid:02d} {name}: {score:.4f}")


if __name__ == "__main__":
    main()
