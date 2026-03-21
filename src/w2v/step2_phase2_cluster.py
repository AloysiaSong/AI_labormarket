#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Step 2 Phase 2: 全量聚类
配置：UMAP 50维 + K=20
三种聚类方法：K-means（硬）、GMM（软）、距离加权（从K-means质心）
"""

from __future__ import annotations
import logging
import time
import warnings
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from gensim.models import KeyedVectors
from sklearn.cluster import MiniBatchKMeans
from sklearn.mixture import GaussianMixture
import umap

warnings.filterwarnings("ignore", category=FutureWarning)

# =========================
# Config
# =========================
KV_PATH = Path("/Users/yu/code/code2601/TY/output/w2v/word2vec.kv")
OUTPUT_DIR = Path("/Users/yu/code/code2601/TY/output/w2v")

UMAP_DIM = 50
K = 20
SEED = 42

# UMAP params (same as Phase 1)
UMAP_N_NEIGHBORS = 15
UMAP_MIN_DIST = 0.1
UMAP_METRIC = "cosine"

# GMM
GMM_COV_TYPE = "diag"  # diag for stability in 50-dim
GMM_N_INIT = 3
GMM_MAX_ITER = 200

# =========================
# Logging
# =========================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("W2V-Cluster")


def distance_weights(X: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    """
    从 K-means 质心计算距离加权的软分配。
    weight_k = 1 / dist(x, centroid_k)^2, 然后归一化。
    """
    # X: (N, D), centroids: (K, D)
    # dists: (N, K)
    dists = np.zeros((X.shape[0], centroids.shape[0]), dtype=np.float32)
    for k in range(centroids.shape[0]):
        diff = X - centroids[k]
        dists[:, k] = np.sum(diff ** 2, axis=1)

    # Inverse square distance, with small epsilon to avoid division by zero
    inv_dists = 1.0 / (dists + 1e-10)
    # Normalize to probabilities
    row_sums = inv_dists.sum(axis=1, keepdims=True)
    weights = inv_dists / row_sums
    return weights


def main():
    t0 = time.time()

    # Load word vectors
    logger.info("加载词向量...")
    kv = KeyedVectors.load(str(KV_PATH))
    all_vectors = kv.vectors.astype(np.float32)
    words = list(kv.index_to_key)
    vocab_size = len(words)
    logger.info(f"词表: {vocab_size:,} 个词, {all_vectors.shape[1]} 维")

    # =====================
    # UMAP 降维
    # =====================
    logger.info(f"UMAP 降维: {all_vectors.shape[1]}维 → {UMAP_DIM}维 (全量 {vocab_size:,} 个词)...")
    t1 = time.time()
    reducer = umap.UMAP(
        n_components=UMAP_DIM,
        n_neighbors=UMAP_N_NEIGHBORS,
        min_dist=UMAP_MIN_DIST,
        metric=UMAP_METRIC,
        random_state=SEED,
        verbose=True,
    )
    X_umap = reducer.fit_transform(all_vectors)
    logger.info(f"UMAP 完成 | shape={X_umap.shape} | 耗时: {(time.time() - t1) / 60:.1f} 分钟")

    # Save UMAP reducer and reduced vectors
    umap_path = OUTPUT_DIR / "umap_50d_vectors.npy"
    np.save(str(umap_path), X_umap)
    logger.info(f"UMAP 向量已保存: {umap_path}")

    reducer_path = OUTPUT_DIR / "umap_reducer.pkl"
    with open(reducer_path, "wb") as f:
        pickle.dump(reducer, f)
    logger.info(f"UMAP reducer 已保存: {reducer_path}")

    # =====================
    # 1. K-means (硬聚类)
    # =====================
    logger.info(f"K-means 聚类: K={K}...")
    t1 = time.time()
    km = MiniBatchKMeans(
        n_clusters=K,
        random_state=SEED,
        batch_size=10_000,
        n_init=5,
        max_iter=500,
    )
    km_labels = km.fit_predict(X_umap)
    km_centroids = km.cluster_centers_  # (K, 50)
    logger.info(f"K-means 完成 | inertia={km.inertia_:.0f} | 耗时: {time.time() - t1:.1f}s")

    # Cluster size distribution
    unique, counts = np.unique(km_labels, return_counts=True)
    for cid, cnt in sorted(zip(unique, counts), key=lambda x: -x[1]):
        logger.info(f"  Cluster {cid:2d}: {cnt:>7,} 词 ({cnt/vocab_size*100:.1f}%)")

    # Save K-means model
    km_path = OUTPUT_DIR / "kmeans_model.pkl"
    with open(km_path, "wb") as f:
        pickle.dump(km, f)

    # =====================
    # 2. GMM (软聚类)
    # =====================
    logger.info(f"GMM 聚类: K={K}, cov_type={GMM_COV_TYPE}...")
    t1 = time.time()
    gmm = GaussianMixture(
        n_components=K,
        covariance_type=GMM_COV_TYPE,
        n_init=GMM_N_INIT,
        max_iter=GMM_MAX_ITER,
        random_state=SEED,
        verbose=1,
    )
    gmm.fit(X_umap)
    gmm_probs = gmm.predict_proba(X_umap)  # (N, K)
    gmm_labels = gmm.predict(X_umap)
    logger.info(f"GMM 完成 | BIC={gmm.bic(X_umap):.0f} | AIC={gmm.aic(X_umap):.0f} | 耗时: {(time.time() - t1) / 60:.1f} 分钟")

    # Save GMM model
    gmm_path = OUTPUT_DIR / "gmm_model.pkl"
    with open(gmm_path, "wb") as f:
        pickle.dump(gmm, f)

    # =====================
    # 3. 距离加权 (从 K-means 质心)
    # =====================
    logger.info("计算距离加权软分配...")
    t1 = time.time()
    dist_weights = distance_weights(X_umap, km_centroids)  # (N, K)
    logger.info(f"距离加权完成 | 耗时: {time.time() - t1:.1f}s")

    # =====================
    # 保存结果
    # =====================
    logger.info("保存聚类结果...")

    # Main output: word -> cluster mappings
    result_data = {
        "word": words,
        "km_cluster": km_labels,
        "gmm_cluster": gmm_labels,
    }
    # Add GMM probs as separate columns
    for k in range(K):
        result_data[f"gmm_prob_{k}"] = gmm_probs[:, k]
    # Add distance weights as separate columns
    for k in range(K):
        result_data[f"dist_weight_{k}"] = dist_weights[:, k]

    df = pd.DataFrame(result_data)
    parquet_path = OUTPUT_DIR / "word_clusters.parquet"
    df.to_parquet(parquet_path, index=False)
    logger.info(f"聚类结果已保存: {parquet_path} ({len(df):,} 行)")

    # Also save a lightweight CSV with just hard clusters for quick inspection
    df_light = df[["word", "km_cluster", "gmm_cluster"]].copy()
    csv_path = OUTPUT_DIR / "word_clusters_light.csv"
    df_light.to_csv(csv_path, index=False)
    logger.info(f"轻量版已保存: {csv_path}")

    # =====================
    # 质检：每个 cluster 的 top 词
    # =====================
    logger.info("=== K-means 聚类质检：每个 cluster 的 top-10 高频词 ===")
    # Use word frequency from the KeyedVectors (index order = frequency order in gensim)
    word_freq_rank = {w: i for i, w in enumerate(words)}  # lower index = higher freq

    for cid in range(K):
        mask = km_labels == cid
        cluster_words = [words[i] for i in range(vocab_size) if mask[i]]
        # Sort by frequency rank (lower = more frequent)
        cluster_words_sorted = sorted(cluster_words, key=lambda w: word_freq_rank.get(w, vocab_size))
        top10 = cluster_words_sorted[:10]
        logger.info(f"  Cluster {cid:2d} ({mask.sum():>6,} 词): {', '.join(top10)}")

    # =====================
    # 可选：2D UMAP 可视化
    # =====================
    logger.info("UMAP 降到 2维 (可视化用)...")
    t1 = time.time()
    reducer_2d = umap.UMAP(
        n_components=2,
        n_neighbors=UMAP_N_NEIGHBORS,
        min_dist=UMAP_MIN_DIST,
        metric=UMAP_METRIC,
        random_state=SEED,
        verbose=False,
    )
    X_2d = reducer_2d.fit_transform(X_umap)  # from 50d, not raw 200d
    logger.info(f"2D UMAP 完成 | 耗时: {(time.time() - t1) / 60:.1f} 分钟")

    np.save(str(OUTPUT_DIR / "umap_2d_vectors.npy"), X_2d)
    logger.info("2D 向量已保存")

    total = (time.time() - t0) / 60
    logger.info(f"=== Phase 2 总耗时: {total:.1f} 分钟 ===")


if __name__ == "__main__":
    main()
