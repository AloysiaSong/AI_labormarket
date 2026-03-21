#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Step 2 Phase 1: UMAP维度 × K聚类数 网格实验
子采样 100,000 个词向量，快速筛选最佳 (dim, K) 配置
"""

from __future__ import annotations
import logging
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from gensim.models import KeyedVectors
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score
import umap

warnings.filterwarnings("ignore", category=FutureWarning)

# =========================
# Config
# =========================
KV_PATH = Path("/Users/yu/code/code2601/TY/output/w2v/word2vec.kv")
OUTPUT_DIR = Path("/Users/yu/code/code2601/TY/output/w2v")

SUBSAMPLE_N = 100_000
SEED = 42

# Grid
UMAP_DIMS = [30, 50, 80]       # + raw 200
K_VALUES = [10, 20, 30, 50, 80, 100]

# UMAP params
UMAP_N_NEIGHBORS = 15
UMAP_MIN_DIST = 0.1
UMAP_METRIC = "cosine"

# Silhouette subsample (even 100k is slow for full silhouette)
SILHOUETTE_SAMPLE = 20_000

# =========================
# Logging
# =========================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("W2V-Experiment")


def main():
    t0 = time.time()

    # Load word vectors
    logger.info("加载词向量...")
    kv = KeyedVectors.load(str(KV_PATH))
    all_vectors = kv.vectors  # (596880, 200)
    vocab_size, vec_dim = all_vectors.shape
    logger.info(f"词表: {vocab_size:,} 个词, {vec_dim} 维")

    # Subsample
    rng = np.random.RandomState(SEED)
    sample_idx = rng.choice(vocab_size, size=min(SUBSAMPLE_N, vocab_size), replace=False)
    X_sample = all_vectors[sample_idx].astype(np.float32)
    logger.info(f"子采样: {X_sample.shape[0]:,} 个词向量")

    # Prepare dimension settings: UMAP reductions + raw
    dim_settings = {}

    # UMAP reductions
    for n_comp in UMAP_DIMS:
        logger.info(f"UMAP 降维到 {n_comp} 维...")
        t1 = time.time()
        reducer = umap.UMAP(
            n_components=n_comp,
            n_neighbors=UMAP_N_NEIGHBORS,
            min_dist=UMAP_MIN_DIST,
            metric=UMAP_METRIC,
            random_state=SEED,
            verbose=False,
        )
        X_reduced = reducer.fit_transform(X_sample)
        elapsed = time.time() - t1
        logger.info(f"  UMAP {n_comp}维 完成 | shape={X_reduced.shape} | 耗时: {elapsed:.1f}s")
        dim_settings[f"umap_{n_comp}"] = X_reduced

    # Raw 200-dim
    dim_settings["raw_200"] = X_sample
    logger.info("添加原始200维 (无降维)")

    # Grid experiment
    results = []
    total_combos = len(dim_settings) * len(K_VALUES)
    combo_i = 0

    for dim_name, X in dim_settings.items():
        for K in K_VALUES:
            combo_i += 1
            logger.info(f"[{combo_i}/{total_combos}] {dim_name} × K={K}")
            t1 = time.time()

            # K-means (MiniBatch for speed)
            km = MiniBatchKMeans(
                n_clusters=K,
                random_state=SEED,
                batch_size=10_000,
                n_init=3,
                max_iter=300,
            )
            labels = km.fit_predict(X)
            inertia = km.inertia_

            # Silhouette (subsample for speed)
            sil_idx = rng.choice(len(X), size=min(SILHOUETTE_SAMPLE, len(X)), replace=False)
            sil = silhouette_score(X[sil_idx], labels[sil_idx])

            # Calinski-Harabasz
            ch = calinski_harabasz_score(X, labels)

            elapsed = time.time() - t1
            logger.info(f"  inertia={inertia:.0f} | silhouette={sil:.4f} | CH={ch:.0f} | {elapsed:.1f}s")

            results.append({
                "dim_setting": dim_name,
                "n_dims": int(dim_name.split("_")[1]) if "umap" in dim_name else 200,
                "K": K,
                "inertia": inertia,
                "silhouette": sil,
                "calinski_harabasz": ch,
                "time_sec": elapsed,
            })

    # Save results
    df = pd.DataFrame(results)
    out_csv = OUTPUT_DIR / "dim_k_experiment.csv"
    df.to_csv(out_csv, index=False)
    logger.info(f"实验结果已保存: {out_csv}")

    # Print summary: best by each metric
    logger.info("=== 最佳配置 ===")
    best_sil = df.loc[df["silhouette"].idxmax()]
    logger.info(f"  Best Silhouette: {best_sil['dim_setting']} K={best_sil['K']:.0f} → {best_sil['silhouette']:.4f}")
    best_ch = df.loc[df["calinski_harabasz"].idxmax()]
    logger.info(f"  Best CH Index:   {best_ch['dim_setting']} K={best_ch['K']:.0f} → {best_ch['calinski_harabasz']:.0f}")

    # Print full table grouped by dim
    logger.info("=== 完整结果 ===")
    for dim_name in dim_settings:
        sub = df[df["dim_setting"] == dim_name].sort_values("K")
        logger.info(f"\n  {dim_name}:")
        for _, row in sub.iterrows():
            logger.info(f"    K={row['K']:3.0f} | sil={row['silhouette']:.4f} | CH={row['calinski_harabasz']:>10.0f} | inertia={row['inertia']:>12.0f}")

    total = (time.time() - t0) / 60
    logger.info(f"总耗时: {total:.1f} 分钟")


if __name__ == "__main__":
    main()
