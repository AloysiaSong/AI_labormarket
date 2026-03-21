#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Step 2 Phase 3: 二级聚类
对技能类一级 cluster 做内部子聚类（K-means on UMAP 50d vectors）
"""

from __future__ import annotations
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_score
from gensim.models import KeyedVectors

# =========================
# Config
# =========================
KV_PATH = Path("/Users/yu/code/code2601/TY/output/w2v/word2vec.kv")
UMAP_VECTORS_PATH = Path("/Users/yu/code/code2601/TY/output/w2v/umap_50d_vectors.npy")
CLUSTERS_PATH = Path("/Users/yu/code/code2601/TY/output/w2v/word_clusters.csv")
OUTPUT_DIR = Path("/Users/yu/code/code2601/TY/output/w2v")

SEED = 42

# 需要做二级聚类的一级 cluster 及其子类数
SUBCLUSTER_CONFIG = {
    4:  16,  # 行政/合规/流程 (41,298 词)
    5:  6,   # 教育/内容/新媒体 (24,112 词)
    7:  10,  # 办公/设计软件 (12,626 词)
    8:  6,   # 基本要求/软技能 (27,179 词)
    9:  6,   # 管理层/金融/组织 (31,079 词)
    11: 20,  # IT/编程/系统 (30,779 词)
    15: 8,   # 工程/研发/质量 (29,926 词)
    16: 10,  # 核心业务/销售/运营 (44,113 词)
    19: 16,  # 职业发展/培训 (18,314 词)
}

# =========================
# Logging
# =========================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("Subcluster")


def main():
    t0 = time.time()

    # Load data
    logger.info("加载数据...")
    kv = KeyedVectors.load(str(KV_PATH))
    words = list(kv.index_to_key)
    word_freq_rank = {w: i for i, w in enumerate(words)}

    X_umap = np.load(str(UMAP_VECTORS_PATH)).astype(np.float32)
    df = pd.read_csv(CLUSTERS_PATH)
    logger.info(f"词表: {len(words):,} | UMAP: {X_umap.shape}")

    # Add sub_cluster column (default = -1 for non-skill clusters)
    df["sub_cluster"] = -1
    # Composite label: "{level1}_{level2}"
    df["cluster_label"] = ""

    all_sub_results = []

    for l1_id, sub_k in SUBCLUSTER_CONFIG.items():
        mask = df["km_cluster"].values == l1_id
        indices = np.where(mask)[0]
        X_sub = X_umap[indices]
        n_words = len(indices)

        logger.info(f"\n=== 一级 Cluster {l1_id} ({n_words:,} 词) → 子聚类 K={sub_k} ===")

        km = MiniBatchKMeans(
            n_clusters=sub_k,
            random_state=SEED,
            batch_size=min(5000, n_words),
            n_init=5,
            max_iter=500,
        )
        sub_labels = km.fit_predict(X_sub)

        # Silhouette
        if n_words > sub_k:
            sil_sample = min(10000, n_words)
            rng = np.random.RandomState(SEED)
            sil_idx = rng.choice(n_words, size=sil_sample, replace=False)
            sil = silhouette_score(X_sub[sil_idx], sub_labels[sil_idx])
            logger.info(f"  Silhouette: {sil:.4f}")

        # Assign back
        df.loc[mask, "sub_cluster"] = sub_labels
        df.loc[mask, "cluster_label"] = [f"{l1_id}_{sl}" for sl in sub_labels]

        # Print top words per sub-cluster
        for sc in range(sub_k):
            sc_mask = sub_labels == sc
            sc_indices = indices[sc_mask]
            sc_words = [words[i] for i in sc_indices]
            sc_words_sorted = sorted(sc_words, key=lambda w: word_freq_rank.get(w, 999999))
            top20 = sc_words_sorted[:20]
            logger.info(f"  {l1_id}_{sc} ({sc_mask.sum():>5,} 词): {', '.join(top20)}")

            all_sub_results.append({
                "l1_cluster": l1_id,
                "sub_cluster": sc,
                "label": f"{l1_id}_{sc}",
                "n_words": int(sc_mask.sum()),
                "top20": ", ".join(top20),
            })

    # Label non-skill clusters
    non_skill_clusters = set(range(20)) - set(SUBCLUSTER_CONFIG.keys())
    for l1_id in non_skill_clusters:
        mask = df["km_cluster"].values == l1_id
        df.loc[mask, "cluster_label"] = str(l1_id)

    # Save
    out_csv = OUTPUT_DIR / "word_clusters_hierarchical.csv"
    df[["word", "km_cluster", "sub_cluster", "cluster_label"]].to_csv(out_csv, index=False)
    logger.info(f"\n层次聚类结果已保存: {out_csv} ({len(df):,} 行)")

    # Save sub-cluster summary
    summary_df = pd.DataFrame(all_sub_results)
    summary_csv = OUTPUT_DIR / "subcluster_summary.csv"
    summary_df.to_csv(summary_csv, index=False)
    logger.info(f"子聚类汇总: {summary_csv}")

    # Count total clusters
    unique_labels = df["cluster_label"].nunique()
    logger.info(f"总 cluster 数: {unique_labels} (一级非技能 {len(non_skill_clusters)} + 二级 {sum(SUBCLUSTER_CONFIG.values())})")

    logger.info(f"总耗时: {(time.time() - t0) / 60:.1f} 分钟")


if __name__ == "__main__":
    main()
