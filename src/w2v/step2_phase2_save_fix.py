#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Quick fix: Load already-completed GMM + K-means, save results, run 2D UMAP.
"""

from __future__ import annotations
import logging
import time
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from gensim.models import KeyedVectors
import umap

# =========================
# Config
# =========================
KV_PATH = Path("/Users/yu/code/code2601/TY/output/w2v/word2vec.kv")
OUTPUT_DIR = Path("/Users/yu/code/code2601/TY/output/w2v")
UMAP_VECTORS_PATH = OUTPUT_DIR / "umap_50d_vectors.npy"
KMEANS_PATH = OUTPUT_DIR / "kmeans_model.pkl"
GMM_PATH = OUTPUT_DIR / "gmm_model.pkl"

K = 20
SEED = 42

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("SaveFix")


def distance_weights(X, centroids):
    dists = np.zeros((X.shape[0], centroids.shape[0]), dtype=np.float64)
    for k in range(centroids.shape[0]):
        diff = X - centroids[k]
        dists[:, k] = np.sum(diff ** 2, axis=1)
    inv_dists = 1.0 / (dists + 1e-10)
    return inv_dists / inv_dists.sum(axis=1, keepdims=True)


def main():
    t0 = time.time()

    # Load everything
    logger.info("加载数据...")
    kv = KeyedVectors.load(str(KV_PATH))
    words = list(kv.index_to_key)
    vocab_size = len(words)

    X_umap = np.load(str(UMAP_VECTORS_PATH))
    X_f64 = X_umap.astype(np.float64)

    with open(KMEANS_PATH, "rb") as f:
        km = pickle.load(f)
    km_labels = km.labels_
    km_centroids = km.cluster_centers_

    # Check if GMM model exists
    if GMM_PATH.exists():
        logger.info("GMM 模型已存在，加载...")
        with open(GMM_PATH, "rb") as f:
            gmm = pickle.load(f)
        gmm_probs = gmm.predict_proba(X_f64)
        gmm_labels = gmm.predict(X_f64)
        logger.info(f"GMM BIC={gmm.bic(X_f64):.0f} | AIC={gmm.aic(X_f64):.0f}")
    else:
        logger.info("GMM 模型不存在，重新训练...")
        from sklearn.mixture import GaussianMixture
        gmm = GaussianMixture(
            n_components=K, covariance_type="diag",
            n_init=3, max_iter=300, reg_covar=1e-4,
            random_state=SEED, verbose=1,
        )
        gmm.fit(X_f64)
        gmm_probs = gmm.predict_proba(X_f64)
        gmm_labels = gmm.predict(X_f64)
        with open(GMM_PATH, "wb") as f:
            pickle.dump(gmm, f)

    # Distance weights
    logger.info("计算距离加权...")
    dist_w = distance_weights(X_f64, km_centroids.astype(np.float64))

    # Save soft assignments as npz
    npz_path = OUTPUT_DIR / "word_clusters_soft.npz"
    np.savez_compressed(str(npz_path),
        gmm_probs=gmm_probs.astype(np.float32),
        dist_weights=dist_w.astype(np.float32),
    )
    logger.info(f"软聚类矩阵已保存: {npz_path}")

    # Save hard clusters as CSV
    df = pd.DataFrame({"word": words, "km_cluster": km_labels, "gmm_cluster": gmm_labels})
    csv_path = OUTPUT_DIR / "word_clusters.csv"
    df.to_csv(csv_path, index=False)
    logger.info(f"硬聚类已保存: {csv_path} ({len(df):,} 行)")

    # Quality check: top-10 words per K-means cluster
    logger.info("=== K-means 聚类质检：top-10 高频词 ===")
    word_freq_rank = {w: i for i, w in enumerate(words)}
    for cid in range(K):
        mask = km_labels == cid
        cw = sorted([words[i] for i in range(vocab_size) if mask[i]],
                     key=lambda w: word_freq_rank.get(w, vocab_size))
        logger.info(f"  Cluster {cid:2d} ({mask.sum():>6,} 词): {', '.join(cw[:10])}")

    logger.info("=== GMM 聚类质检：top-10 高频词 ===")
    for cid in range(K):
        mask = gmm_labels == cid
        cw = sorted([words[i] for i in range(vocab_size) if mask[i]],
                     key=lambda w: word_freq_rank.get(w, vocab_size))
        logger.info(f"  Cluster {cid:2d} ({mask.sum():>6,} 词): {', '.join(cw[:10])}")

    # 2D UMAP
    logger.info("UMAP 降到 2维...")
    t1 = time.time()
    reducer_2d = umap.UMAP(
        n_components=2, n_neighbors=15, min_dist=0.1,
        metric="cosine", random_state=SEED, verbose=False,
    )
    X_2d = reducer_2d.fit_transform(X_umap)
    np.save(str(OUTPUT_DIR / "umap_2d_vectors.npy"), X_2d)
    logger.info(f"2D UMAP 完成 | 耗时: {(time.time() - t1) / 60:.1f} 分钟")

    # Cluster size summary
    logger.info("=== Cluster 大小对比 (K-means vs GMM) ===")
    for cid in range(K):
        km_n = (km_labels == cid).sum()
        gmm_n = (gmm_labels == cid).sum()
        logger.info(f"  Cluster {cid:2d}: K-means={km_n:>6,} | GMM={gmm_n:>6,}")

    logger.info(f"=== 总耗时: {(time.time() - t0) / 60:.1f} 分钟 ===")


if __name__ == "__main__":
    main()
