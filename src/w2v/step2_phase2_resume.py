#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Step 2 Phase 2 (恢复): 从已保存的 UMAP 向量和 K-means 结果继续
修复 GMM: reg_covar=1e-4, float64
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
from sklearn.mixture import GaussianMixture
import umap

warnings.filterwarnings("ignore", category=FutureWarning)

# =========================
# Config
# =========================
KV_PATH = Path("/Users/yu/code/code2601/TY/output/w2v/word2vec.kv")
OUTPUT_DIR = Path("/Users/yu/code/code2601/TY/output/w2v")
UMAP_VECTORS_PATH = OUTPUT_DIR / "umap_50d_vectors.npy"
KMEANS_PATH = OUTPUT_DIR / "kmeans_model.pkl"

K = 20
SEED = 42

# GMM (fixed)
GMM_COV_TYPE = "diag"
GMM_N_INIT = 3
GMM_MAX_ITER = 300
GMM_REG_COVAR = 1e-4  # regularization to prevent singular covariance

# UMAP 2D params
UMAP_N_NEIGHBORS = 15
UMAP_MIN_DIST = 0.1
UMAP_METRIC = "cosine"

# =========================
# Logging
# =========================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("W2V-Cluster-Resume")


def distance_weights(X: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    """从 K-means 质心计算距离加权的软分配"""
    dists = np.zeros((X.shape[0], centroids.shape[0]), dtype=np.float64)
    for k in range(centroids.shape[0]):
        diff = X - centroids[k]
        dists[:, k] = np.sum(diff ** 2, axis=1)
    inv_dists = 1.0 / (dists + 1e-10)
    row_sums = inv_dists.sum(axis=1, keepdims=True)
    return inv_dists / row_sums


def main():
    t0 = time.time()

    # Load saved data
    logger.info("加载已保存的数据...")
    kv = KeyedVectors.load(str(KV_PATH))
    words = list(kv.index_to_key)
    vocab_size = len(words)

    X_umap = np.load(str(UMAP_VECTORS_PATH))
    logger.info(f"UMAP 向量: {X_umap.shape}")

    with open(KMEANS_PATH, "rb") as f:
        km = pickle.load(f)
    km_labels = km.labels_
    km_centroids = km.cluster_centers_
    logger.info(f"K-means 已加载: K={K}, inertia={km.inertia_:.0f}")

    # =====================
    # K-means cluster sizes
    # =====================
    unique, counts = np.unique(km_labels, return_counts=True)
    for cid, cnt in sorted(zip(unique, counts), key=lambda x: -x[1]):
        logger.info(f"  Cluster {cid:2d}: {cnt:>7,} 词 ({cnt/vocab_size*100:.1f}%)")

    # =====================
    # GMM (fixed: float64 + reg_covar)
    # =====================
    X_f64 = X_umap.astype(np.float64)
    logger.info(f"GMM 聚类: K={K}, cov_type={GMM_COV_TYPE}, reg_covar={GMM_REG_COVAR}...")
    t1 = time.time()
    gmm = GaussianMixture(
        n_components=K,
        covariance_type=GMM_COV_TYPE,
        n_init=GMM_N_INIT,
        max_iter=GMM_MAX_ITER,
        reg_covar=GMM_REG_COVAR,
        random_state=SEED,
        verbose=1,
    )
    gmm.fit(X_f64)
    gmm_probs = gmm.predict_proba(X_f64)  # (N, K)
    gmm_labels = gmm.predict(X_f64)
    logger.info(f"GMM 完成 | BIC={gmm.bic(X_f64):.0f} | AIC={gmm.aic(X_f64):.0f} | 耗时: {(time.time() - t1) / 60:.1f} 分钟")

    gmm_path = OUTPUT_DIR / "gmm_model.pkl"
    with open(gmm_path, "wb") as f:
        pickle.dump(gmm, f)

    # =====================
    # 距离加权
    # =====================
    logger.info("计算距离加权软分配...")
    t1 = time.time()
    dist_w = distance_weights(X_f64, km_centroids.astype(np.float64))
    logger.info(f"距离加权完成 | 耗时: {time.time() - t1:.1f}s")

    # =====================
    # 保存结果
    # =====================
    logger.info("保存聚类结果...")
    result_data = {
        "word": words,
        "km_cluster": km_labels,
        "gmm_cluster": gmm_labels,
    }
    for k in range(K):
        result_data[f"gmm_prob_{k}"] = gmm_probs[:, k].astype(np.float32)
    for k in range(K):
        result_data[f"dist_weight_{k}"] = dist_w[:, k].astype(np.float32)

    df = pd.DataFrame(result_data)
    # Save as compressed npz (parquet needs pyarrow which is not installed)
    # Save soft assignments as numpy arrays for efficiency
    npz_path = OUTPUT_DIR / "word_clusters_soft.npz"
    np.savez_compressed(
        str(npz_path),
        gmm_probs=gmm_probs.astype(np.float32),
        dist_weights=dist_w.astype(np.float32),
    )
    logger.info(f"软聚类矩阵已保存: {npz_path}")

    # Save hard clusters as CSV
    df_full = df[["word", "km_cluster", "gmm_cluster"]]
    full_csv_path = OUTPUT_DIR / "word_clusters.csv"
    df_full.to_csv(full_csv_path, index=False)
    logger.info(f"聚类结果已保存: {full_csv_path} ({len(df):,} 行)")

    df_light = df[["word", "km_cluster", "gmm_cluster"]].copy()
    csv_path = OUTPUT_DIR / "word_clusters_light.csv"
    df_light.to_csv(csv_path, index=False)
    logger.info(f"轻量版已保存: {csv_path}")

    # =====================
    # 质检：每个 cluster 的 top 词
    # =====================
    logger.info("=== K-means 聚类质检：每个 cluster 的 top-10 高频词 ===")
    word_freq_rank = {w: i for i, w in enumerate(words)}
    for cid in range(K):
        mask = km_labels == cid
        cluster_words = [words[i] for i in range(vocab_size) if mask[i]]
        cluster_words_sorted = sorted(cluster_words, key=lambda w: word_freq_rank.get(w, vocab_size))
        top10 = cluster_words_sorted[:10]
        logger.info(f"  Cluster {cid:2d} ({mask.sum():>6,} 词): {', '.join(top10)}")

    logger.info("=== GMM 聚类质检：每个 cluster 的 top-10 高频词 ===")
    for cid in range(K):
        mask = gmm_labels == cid
        cluster_words = [words[i] for i in range(vocab_size) if mask[i]]
        cluster_words_sorted = sorted(cluster_words, key=lambda w: word_freq_rank.get(w, vocab_size))
        top10 = cluster_words_sorted[:10]
        logger.info(f"  Cluster {cid:2d} ({mask.sum():>6,} 词): {', '.join(top10)}")

    # =====================
    # 2D UMAP 可视化
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
    X_2d = reducer_2d.fit_transform(X_umap)
    logger.info(f"2D UMAP 完成 | 耗时: {(time.time() - t1) / 60:.1f} 分钟")
    np.save(str(OUTPUT_DIR / "umap_2d_vectors.npy"), X_2d)
    logger.info("2D 向量已保存")

    total = (time.time() - t0) / 60
    logger.info(f"=== Phase 2 (恢复) 总耗时: {total:.1f} 分钟 ===")


if __name__ == "__main__":
    main()
