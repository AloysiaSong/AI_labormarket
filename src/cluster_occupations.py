#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
岗位名称 Embedding 聚类 → 职业大类
Step 1: 提取高频岗位名 + embedding
Step 2: K-means 聚类
Step 3: 生成映射表 (cleaned_title → cluster_id)
Step 4: 全量数据回标
"""

import csv
import re
import json
import time
import pickle
import numpy as np
from pathlib import Path
from collections import Counter

# ========================
# Config
# ========================
MERGED_CSV = Path("/Users/yu/code/code2601/TY/dataset/merged_1_6.csv")
OUTPUT_DIR = Path("/Users/yu/code/code2601/TY/output/clusters")

TOP_N = 50000          # embed top 5万高频岗位名
N_CLUSTERS = 80        # 聚类数
BATCH_SIZE = 512       # embedding batch size
MODEL_NAME = "shibing624/text2vec-base-chinese"  # 中文embedding模型

# ========================
# Title cleaning (same as analysis)
# ========================
def clean_title(t):
    t = re.sub(r'[（(][^)）]*[)）]', '', t)
    t = re.sub(r'\d{3,}元?/?', '', t)
    t = re.sub(r'[+➕].{0,20}$', '', t)
    t = re.sub(r'/[包入无].{0,15}$', '', t)
    t = re.sub(r'(急聘|诚聘|高薪|急招|高新|直招|包吃住|包食宿|五险一金|双休|转正)', '', t)
    t = t.strip(' /\\-—_·.，,')
    return t


def step1_extract_titles():
    """扫描全量数据，统计清洗后的岗位名频次"""
    print("[Step 1] 扫描岗位名称...")
    counter = Counter()
    with open(MERGED_CSV, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            t = row.get('招聘岗位', '').strip()
            if t:
                ct = clean_title(t)
                if ct and len(ct) >= 2:
                    counter[ct] += 1
            if i % 5_000_000 == 0:
                print(f"  {i:,} rows scanned, {len(counter):,} unique...", flush=True)

    print(f"  Total unique cleaned titles: {len(counter):,}")
    top_titles = counter.most_common(TOP_N)
    coverage = sum(v for _, v in top_titles) / sum(counter.values()) * 100
    print(f"  Top {TOP_N:,} cover {coverage:.1f}% of rows")

    # 保存完整频次表（用于后续回标低频词）
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    freq_path = OUTPUT_DIR / "title_freq.pkl"
    with open(freq_path, 'wb') as f:
        pickle.dump(counter, f)
    print(f"  Saved freq table → {freq_path}")

    return [t for t, _ in top_titles], counter


def step2_embed(titles):
    """对高频岗位名生成 embedding"""
    print(f"[Step 2] Embedding {len(titles):,} titles with {MODEL_NAME}...")
    from sentence_transformers import SentenceTransformer
    import torch

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"  Device: {device}")

    model = SentenceTransformer(MODEL_NAME, device=device)
    embeddings = model.encode(
        titles,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        normalize_embeddings=True,
    )
    emb_path = OUTPUT_DIR / "title_embeddings.npy"
    np.save(emb_path, embeddings)
    print(f"  Embeddings shape: {embeddings.shape} → {emb_path}")
    return embeddings


def step3_cluster(titles, embeddings):
    """K-means 聚类"""
    print(f"[Step 3] Clustering into {N_CLUSTERS} groups...")
    from sklearn.cluster import MiniBatchKMeans

    kmeans = MiniBatchKMeans(
        n_clusters=N_CLUSTERS,
        batch_size=4096,
        n_init=5,
        random_state=42,
        verbose=0,
    )
    labels = kmeans.fit_predict(embeddings)
    centroids = kmeans.cluster_centers_

    # 保存模型（用于低频词匹配）
    model_path = OUTPUT_DIR / "kmeans_model.pkl"
    with open(model_path, 'wb') as f:
        pickle.dump(kmeans, f)

    # 统计每个 cluster 的 top titles
    from collections import defaultdict
    cluster_titles = defaultdict(list)
    for title, label in zip(titles, labels):
        cluster_titles[int(label)].append(title)

    # 保存 cluster 详情
    cluster_info = {}
    for cid in range(N_CLUSTERS):
        members = cluster_titles[cid]
        cluster_info[cid] = {
            "size": len(members),
            "top_titles": members[:20],
        }

    info_path = OUTPUT_DIR / "cluster_info.json"
    with open(info_path, 'w', encoding='utf-8') as f:
        json.dump(cluster_info, f, ensure_ascii=False, indent=2)
    print(f"  Cluster info → {info_path}")

    # 打印概览
    print(f"\n  {'Cluster':>8} {'Size':>6}  Top Titles")
    print(f"  {'-------':>8} {'----':>6}  ----------")
    for cid in sorted(cluster_info.keys(), key=lambda x: -cluster_info[x]['size']):
        info = cluster_info[cid]
        top3 = ', '.join(info['top_titles'][:5])
        print(f"  {cid:>8} {info['size']:>6}  {top3}")

    # 保存 title → cluster 映射
    title2cluster = {t: int(l) for t, l in zip(titles, labels)}
    map_path = OUTPUT_DIR / "title2cluster.pkl"
    with open(map_path, 'wb') as f:
        pickle.dump(title2cluster, f)
    print(f"\n  Title→Cluster map → {map_path}")

    return kmeans, title2cluster


def step4_assign_all(kmeans, title2cluster, freq_table):
    """全量数据回标: 高频精确匹配 + 低频最近邻"""
    print("[Step 4] 全量回标...")
    from sentence_transformers import SentenceTransformer
    import torch

    # 收集未匹配的低频 title
    unmatched = set()
    total_rows = 0
    matched_rows = 0

    with open(MERGED_CSV, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            total_rows += 1
            t = row.get('招聘岗位', '').strip()
            if t:
                ct = clean_title(t)
                if ct and len(ct) >= 2:
                    if ct in title2cluster:
                        matched_rows += 1
                    else:
                        unmatched.add(ct)
            if i % 5_000_000 == 0:
                print(f"  {i:,} rows scanned...", flush=True)

    print(f"  Total rows: {total_rows:,}")
    print(f"  Matched by top-{TOP_N}: {matched_rows:,} ({matched_rows/total_rows*100:.1f}%)")
    print(f"  Unmatched unique titles: {len(unmatched):,}")

    # 对未匹配的做 embedding + nearest centroid
    if unmatched:
        unmatched_list = list(unmatched)
        print(f"  Embedding {len(unmatched_list):,} unmatched titles...")
        device = "mps" if torch.backends.mps.is_available() else "cpu"
        model = SentenceTransformer(MODEL_NAME, device=device)

        # 分批 embed + predict
        centroids = kmeans.cluster_centers_  # (N_CLUSTERS, dim)
        batch_count = 0
        for start in range(0, len(unmatched_list), 10000):
            batch = unmatched_list[start:start+10000]
            embs = model.encode(batch, batch_size=BATCH_SIZE, show_progress_bar=False, normalize_embeddings=True)
            # cosine similarity with centroids (embeddings are normalized)
            sims = embs @ centroids.T  # (batch, N_CLUSTERS)
            labels = sims.argmax(axis=1)
            for title, label in zip(batch, labels):
                title2cluster[title] = int(label)
            batch_count += 1
            if batch_count % 50 == 0:
                print(f"    {start+len(batch):,} / {len(unmatched_list):,} embedded...", flush=True)

    # 保存完整映射
    full_map_path = OUTPUT_DIR / "title2cluster_full.pkl"
    with open(full_map_path, 'wb') as f:
        pickle.dump(title2cluster, f)
    print(f"  Full map ({len(title2cluster):,} entries) → {full_map_path}")

    # 生成最终 CSV: id → cluster_id
    print("  Writing id → cluster_id mapping CSV...")
    out_csv = OUTPUT_DIR / "job_cluster_map.csv"
    unmapped_count = 0
    with open(MERGED_CSV, 'r', encoding='utf-8-sig') as fin, \
         open(out_csv, 'w', encoding='utf-8-sig', newline='') as fout:
        reader = csv.DictReader(fin)
        writer = csv.writer(fout)
        writer.writerow(["row_idx", "cluster_id"])
        for i, row in enumerate(reader):
            t = row.get('招聘岗位', '').strip()
            ct = clean_title(t) if t else ''
            cid = title2cluster.get(ct, -1) if ct and len(ct) >= 2 else -1
            writer.writerow([i, cid])
            if cid == -1:
                unmapped_count += 1
            if i % 5_000_000 == 0:
                print(f"    {i:,} rows written...", flush=True)

    print(f"  Done! Unmapped rows: {unmapped_count:,} → {out_csv}")
    return out_csv


def main():
    t0 = time.time()

    # Step 1: Extract titles
    top_titles, freq_table = step1_extract_titles()

    # Step 2: Embed
    embeddings = step2_embed(top_titles)

    # Step 3: Cluster
    kmeans, title2cluster = step3_cluster(top_titles, embeddings)

    # Step 4: Full assignment
    step4_assign_all(kmeans, title2cluster, freq_table)

    elapsed = (time.time() - t0) / 60
    print(f"\nTotal time: {elapsed:.1f} min")


if __name__ == "__main__":
    main()
