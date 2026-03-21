#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ISCO Classification from Topic Profiles — 方案B
==================================================

检验职业分类可预测性是否在AI冲击后下降。

Pipeline:
  Stage 1 (本地): 推断topic向量 + ISCO映射 → 存为npz
  Stage 2 (远程): 训练分类器, 比较pre/post准确率

Stage 1 输出: output/topic_vectors_with_isco.npz
  - X: (N, 50) topic probability vectors (float16 to save space)
  - y_isco: (N,) ISCO-08 4-digit codes (encoded as int)
  - year: (N,) year
  - isco_labels: mapping of encoded int → ISCO code string
"""

import os
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"

import csv
import json
import math
import re
import sys
import time
import numpy as np
from pathlib import Path

BASE = Path("/Users/yu/code/code2601/TY")
CORPUS_JSONL = BASE / "output" / "processed_corpus.jsonl"
MODEL_PATH = BASE / "output" / "global_lda.bin"
SBERT_LOOKUP = BASE / "data" / "Heterogeneity" / "sbert_isco_lookup.csv"
ILO_CSV = BASE / "data" / "esco" / "ilo_genai_isco08_2025.csv"
CLUSTER_MAP_CSV = BASE / "output" / "clusters" / "job_cluster_map_clean.csv"
EXPOSURE_CSV = BASE / "output" / "clusters" / "ai_exposure_ilo.csv"
MERGED_CSV = BASE / "dataset" / "merged_1_6.csv"

OUTPUT_NPZ = BASE / "output" / "topic_vectors_with_isco.npz"

K = 50
NOISE_TOPICS = np.array([21, 25, 28, 39, 46, 49])
MIN_TOKENS = 5


def clean_title(t):
    t = re.sub(r'[（(][^)）]*[)）]', '', t)
    t = re.sub(r'\d{3,}元?/?', '', t)
    t = re.sub(r'[+➕].{0,20}$', '', t)
    t = re.sub(r'/[包入无].{0,15}$', '', t)
    t = re.sub(r'(急聘|诚聘|高薪|急招|高新|直招|包吃住|包食宿|五险一金|双休|转正)', '', t)
    t = t.strip(' /\\-—_·.，,')
    return t


def build_isco_map():
    """Build row_idx → ISCO mapping (same 3-level strategy as build_isco_panel.py)."""
    print("[1/4] Building ISCO mapping...")

    # Load ILO scores (valid ISCO codes)
    ilo_scores = set()
    with open(ILO_CSV, 'r', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            ilo_scores.add(row['isco08_4digit'].strip())
    print(f"  {len(ilo_scores)} valid ISCO codes")

    # Load SBERT lookup
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
    print(f"  SBERT: raw={len(raw_isco):,}, cleaned={len(cleaned_isco):,}")

    # Cluster fallback
    NOISE_CLUSTERS = {5, 6, 9, 43, 51, 74, -1}
    cluster_dominant_isco = {}
    with open(EXPOSURE_CSV, 'r', encoding='utf-8-sig') as f:
        for row in csv.DictReader(f):
            cid = int(row['cluster_id'])
            top_str = row.get('top_isco_codes', '')
            if top_str:
                match = re.match(r'(\d{4})', top_str)
                if match:
                    cluster_dominant_isco[cid] = match.group(1)

    cluster_map = {}
    with open(CLUSTER_MAP_CSV, 'r', encoding='utf-8-sig') as f:
        for row in csv.DictReader(f):
            cluster_map[int(row['row_idx'])] = int(row['cluster_id'])
    print(f"  Cluster fallback: {len(cluster_map):,} rows")

    # Stream merged_1_6.csv to get titles → ISCO
    print("  Streaming merged_1_6.csv for title → ISCO mapping...")
    isco_by_row = {}
    KEYWORD_OVERRIDES = [('翻译', '2643')]
    t0 = time.time()

    with open(MERGED_CSV, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            title = row.get('招聘岗位', '').strip()
            isco = raw_isco.get(title)
            if not isco:
                ct = clean_title(title)
                isco = cleaned_isco.get(ct) if ct and len(ct) >= 2 else None
            if isco is None or isco in ('2353', '2330'):
                for kw, correct_isco in KEYWORD_OVERRIDES:
                    if kw in title:
                        isco = correct_isco
                        break
            if isco is None:
                cid = cluster_map.get(i)
                if cid is not None and cid not in NOISE_CLUSTERS:
                    isco = cluster_dominant_isco.get(cid)

            if isco is not None:
                isco_by_row[i] = isco

            if (i + 1) % 5_000_000 == 0:
                print(f"    {i+1:,} rows... matched={len(isco_by_row):,} [{time.time()-t0:.0f}s]")

    print(f"  Total matched: {len(isco_by_row):,} / {i+1:,} ({len(isco_by_row)/(i+1)*100:.1f}%)")
    return isco_by_row


def infer_topic_vectors(isco_by_row):
    """Stream corpus, infer topic vectors, attach ISCO codes."""
    import tomotopy as tp

    print("\n[2/4] Loading LDA model...")
    mdl = tp.LDAModel.load(str(MODEL_PATH))
    vocabs = list(mdl.used_vocabs)
    word2idx = {w: i for i, w in enumerate(vocabs)}

    # Extract log P(word|topic) matrix
    tw = np.zeros((K, len(vocabs)), dtype=np.float32)
    for k in range(K):
        tw[k] = mdl.get_topic_word_dist(k)
    log_tw = np.log(tw + 1e-12)
    log_alpha = np.log(np.array(mdl.alpha, dtype=np.float64) + 1e-12)
    del mdl
    print(f"  Model loaded: {K} topics, {len(vocabs):,} vocab")

    print("\n[3/4] Inferring topic vectors...")
    t0 = time.time()

    # Pre-allocate in chunks
    CHUNK = 5_000_000
    X_chunks = []
    y_chunks = []
    yr_chunks = []
    n_kept = 0
    n_total = 0

    with open(CORPUS_JSONL, 'r', encoding='utf-8') as f:
        X_buf = np.zeros((CHUNK, K), dtype=np.float16)
        y_buf = []
        yr_buf = []
        buf_idx = 0

        for line_idx, line in enumerate(f):
            n_total += 1

            # Only process rows with valid ISCO mapping
            isco = isco_by_row.get(line_idx)
            if isco is None:
                continue

            obj = json.loads(line)
            tokens = obj.get('tokens', [])
            year = int(obj.get('year', 0))
            if len(tokens) < MIN_TOKENS or year < 2016:
                continue

            # Infer topic distribution
            indices = [word2idx[t] for t in tokens if t in word2idx]
            if len(indices) < MIN_TOKENS:
                continue

            log_scores = log_tw[:, indices].mean(axis=1) + log_alpha
            log_scores -= log_scores.max()
            probs = np.exp(log_scores)
            s = probs.sum()
            if s < 1e-30:
                continue
            probs /= s

            # Zero out noise topics, renormalize
            probs[NOISE_TOPICS] = 0.0
            s2 = probs.sum()
            if s2 < 1e-30:
                continue
            probs /= s2

            X_buf[buf_idx] = probs.astype(np.float16)
            y_buf.append(isco)
            yr_buf.append(year)
            buf_idx += 1
            n_kept += 1

            if buf_idx >= CHUNK:
                X_chunks.append(X_buf[:buf_idx].copy())
                y_chunks.extend(y_buf)
                yr_chunks.extend(yr_buf)
                X_buf = np.zeros((CHUNK, K), dtype=np.float16)
                y_buf = []
                yr_buf = []
                buf_idx = 0
                elapsed = time.time() - t0
                print(f"    {n_total:,} lines, {n_kept:,} kept [{elapsed:.0f}s]")

        # Flush remaining
        if buf_idx > 0:
            X_chunks.append(X_buf[:buf_idx].copy())
            y_chunks.extend(y_buf)
            yr_chunks.extend(yr_buf)

    elapsed = time.time() - t0
    print(f"  Done: {n_kept:,} jobs with topic vectors [{elapsed:.0f}s]")

    # Concatenate
    X = np.concatenate(X_chunks, axis=0)
    years = np.array(yr_chunks, dtype=np.int16)

    # Encode ISCO codes as integers
    unique_iscos = sorted(set(y_chunks))
    isco2int = {c: i for i, c in enumerate(unique_iscos)}
    y = np.array([isco2int[c] for c in y_chunks], dtype=np.int16)

    return X, y, years, unique_iscos


def main():
    t_start = time.time()

    isco_by_row = build_isco_map()
    X, y, years, isco_labels = infer_topic_vectors(isco_by_row)

    print(f"\n[4/4] Saving to {OUTPUT_NPZ}...")
    print(f"  X: {X.shape} ({X.nbytes / 1e9:.2f} GB)")
    print(f"  Classes: {len(isco_labels)} ISCO codes")
    print(f"  Year range: {years.min()}-{years.max()}")

    np.savez_compressed(
        OUTPUT_NPZ,
        X=X,
        y=y,
        years=years,
        isco_labels=np.array(isco_labels),
    )

    fsize = OUTPUT_NPZ.stat().st_size / 1e9
    print(f"  File size: {fsize:.2f} GB")
    print(f"  Total time: {time.time() - t_start:.0f}s")

    # Quick summary
    pre = years < 2022
    post = years >= 2022
    print(f"\n  Pre-period (2016-2021): {pre.sum():,} jobs")
    print(f"  Post-period (2022-2025): {post.sum():,} jobs")


if __name__ == "__main__":
    main()
