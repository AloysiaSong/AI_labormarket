#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
compute_jd_ai_score.py

为每条招聘广告计算JD级AI暴露分数。

方法：
  1. 提取所有唯一岗位名称（招聘岗位）
  2. 用SBERT (bge-large-zh-v1.5) 编码岗位名称和303个ISCO职业描述
  3. 对每个岗位名称，计算与303个ISCO职业的余弦相似度
  4. 取top-5最相似的ISCO职业，以余弦相似度为权重，加权平均其ai_mean_score
  5. 得到每个岗位名称的jd_ai_score

关键设计（规避内生性）：
  - ai_score仅从岗位名称（短文本标题）计算
  - entropy从岗位描述正文（LDA主题分布）计算
  - 两个变量来自不同文本字段，切断机械性相关

输入:
  data/Heterogeneity/master_with_ai_exposure_v2.csv
  data/esco/ilo_genai_isco08_2025.csv
  data/esco/isco08_zh_names.csv

输出:
  data/Heterogeneity/jd_ai_score_lookup.csv   (title → jd_ai_score)
  data/Heterogeneity/master_with_jd_ai_score.csv (合并后主数据)
"""

import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

# ── 路径配置 ──────────────────────────────────────────────────────────
BASE = Path('/Users/yu/code/code2601/TY')
ILO_CSV = BASE / 'data/esco/ilo_genai_isco08_2025.csv'
ZH_NAMES_CSV = BASE / 'data/esco/isco08_zh_names.csv'
MASTER_CSV = BASE / 'data/Heterogeneity/master_with_ai_exposure_v2.csv'
LOOKUP_OUT = BASE / 'data/Heterogeneity/jd_ai_score_lookup.csv'
MASTER_OUT = BASE / 'data/Heterogeneity/master_with_jd_ai_score.csv'
CKPT_DIR = BASE / 'data/Heterogeneity/jd_ai_score_checkpoints'

MODEL_NAME = 'BAAI/bge-large-zh-v1.5'
BATCH_SIZE = 512   # benchmark: 512最优 (278/sec vs 256→189/sec)
CKPT_EVERY = 100_000
PRINT_EVERY = 10_000
TOP_K = 5


def load_isco_data():
    """加载ISCO职业名称和ILO AI暴露分数。"""
    ilo = pd.read_csv(ILO_CSV)
    zh = pd.read_csv(ZH_NAMES_CSV)
    merged = zh.merge(ilo[['isco08_4digit', 'mean_score']],
                      on='isco08_4digit', how='inner')
    print(f'[ISCO] {len(merged)} occupations with ai_score')

    # 构建查询字符串：中文主名 + 别名
    queries = []
    codes = []
    scores = []
    for _, row in merged.iterrows():
        q = str(row['occupation_name_zh'])
        aliases = str(row.get('aliases_zh', '') or '').strip()
        if aliases and aliases != 'nan':
            q = q + ' ' + aliases
        queries.append(q)
        codes.append(int(row['isco08_4digit']))
        scores.append(float(row['mean_score']))

    return queries, np.array(codes), np.array(scores)


def load_unique_titles():
    """从master CSV提取全部唯一岗位名称，按长度排序减少padding浪费。"""
    print('[1] Extracting unique job titles ...')
    t0 = time.time()
    titles = set()
    for chunk in pd.read_csv(MASTER_CSV, usecols=['招聘岗位'],
                              chunksize=500_000):
        titles.update(chunk['招聘岗位'].dropna().unique())
    # 按字符长度排序：同一batch内标题长度相近，SBERT padding最小化
    titles = sorted(titles, key=len)
    print(f'    {len(titles):,} unique titles in {time.time()-t0:.1f}s')
    lens = [len(t) for t in titles]
    print(f'    Length range: {min(lens)}–{max(lens)} chars, '
          f'median={lens[len(lens)//2]}', flush=True)
    return titles


def load_checkpoint():
    """加载最近的checkpoint。"""
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    ckpts = sorted(CKPT_DIR.glob('ckpt_*.pkl'))
    if not ckpts:
        return 0, {}
    latest = ckpts[-1]
    with open(latest, 'rb') as f:
        data = pickle.load(f)
    print(f'    Resumed from checkpoint: {latest.name} '
          f'({data["n_done"]:,} titles done)')
    return data['n_done'], data['results']


def save_checkpoint(n_done, results):
    """保存checkpoint。"""
    path = CKPT_DIR / f'ckpt_{n_done:010d}.pkl'
    with open(path, 'wb') as f:
        pickle.dump({'n_done': n_done, 'results': results}, f)
    print(f'    Checkpoint saved: {path.name}')


def compute_jd_ai_scores():
    """主流程：编码、计算相似度、top-5加权平均。"""

    # 1. 加载ISCO数据
    isco_queries, isco_codes, isco_scores = load_isco_data()

    # 2. 加载模型
    print('[2] Loading SBERT model ...')
    model = SentenceTransformer(MODEL_NAME)
    print(f'    Model loaded: {MODEL_NAME}')

    # 3. 编码ISCO职业（303个，秒级）
    print('[3] Encoding ISCO occupations ...')
    isco_embs = model.encode(isco_queries, batch_size=64,
                              normalize_embeddings=True,
                              show_progress_bar=False)
    isco_embs = isco_embs.astype(np.float32)
    print(f'    ISCO embeddings: {isco_embs.shape}')

    # 4. 加载唯一岗位名称
    titles = load_unique_titles()

    # 5. 检查checkpoint
    n_done, results = load_checkpoint()

    # 6. 批量编码 + top-5计算（向量化）
    total = len(titles)
    remaining = titles[n_done:]
    print(f'[4] Encoding {len(remaining):,} remaining titles '
          f'(of {total:,} total) ...', flush=True)
    t0 = time.time()
    next_ckpt = n_done + CKPT_EVERY
    next_print = n_done + PRINT_EVERY
    processed = 0

    # 按batch处理
    for batch_start in range(0, len(remaining), BATCH_SIZE):
        batch_end = min(batch_start + BATCH_SIZE, len(remaining))
        batch_titles = remaining[batch_start:batch_end]

        # 编码当前batch
        embs = model.encode(batch_titles, batch_size=BATCH_SIZE,
                             normalize_embeddings=True,
                             show_progress_bar=False)
        embs = np.asarray(embs, dtype=np.float32)

        # 计算与所有ISCO的余弦相似度 (已L2归一化 → dot product)
        sims = embs @ isco_embs.T  # (batch, 303)

        # 向量化top-5: 消除Python内循环
        top_idx = np.argpartition(sims, -TOP_K, axis=1)[:, -TOP_K:]  # (batch, 5)
        top_sims = np.take_along_axis(sims, top_idx, axis=1)          # (batch, 5)
        weights = top_sims / top_sims.sum(axis=1, keepdims=True)       # (batch, 5)
        top_scores = isco_scores[top_idx]                              # (batch, 5)
        jd_scores = (weights * top_scores).sum(axis=1)                 # (batch,)

        # top-1信息
        best_in_top = np.argmax(top_sims, axis=1)                     # (batch,)
        best_idx = top_idx[np.arange(len(batch_titles)), best_in_top]
        best_sims = sims[np.arange(len(batch_titles)), best_idx]

        # 存入results字典
        for j, title in enumerate(batch_titles):
            results[title] = {
                'jd_ai_score': round(float(jd_scores[j]), 6),
                'top1_isco': int(isco_codes[best_idx[j]]),
                'top1_sim': round(float(best_sims[j]), 4),
            }

        processed += len(batch_titles)
        current = n_done + processed

        # 进度打印 (用>=避免batch_size不整除的问题)
        if current >= next_print or batch_end == len(remaining):
            elapsed = time.time() - t0
            rate = processed / elapsed if elapsed > 0 else 0
            eta = (len(remaining) - processed) / rate if rate > 0 else 0
            print(f'    {current:,}/{total:,} '
                  f'({current/total*100:.1f}%) '
                  f'{rate:.0f} titles/sec '
                  f'ETA {eta/60:.0f}min', flush=True)
            next_print += PRINT_EVERY

        # Checkpoint
        if current >= next_ckpt:
            save_checkpoint(current, results)
            next_ckpt += CKPT_EVERY

    # 最终checkpoint
    save_checkpoint(total, results)
    print(f'[5] Encoding complete: {len(results):,} titles processed')
    return results


def save_lookup(results):
    """保存lookup table。"""
    print('[6] Saving lookup table ...')
    rows = []
    for title, info in results.items():
        rows.append({
            '招聘岗位': title,
            'jd_ai_score': info['jd_ai_score'],
            'top1_isco': info['top1_isco'],
            'top1_sim': info['top1_sim'],
        })
    df = pd.DataFrame(rows)
    df.to_csv(LOOKUP_OUT, index=False)
    print(f'    Saved: {LOOKUP_OUT} ({len(df):,} rows)')

    # 描述性统计
    print(f'\n    jd_ai_score statistics:')
    print(f'      mean  = {df.jd_ai_score.mean():.4f}')
    print(f'      std   = {df.jd_ai_score.std():.4f}')
    print(f'      min   = {df.jd_ai_score.min():.4f}')
    print(f'      p25   = {df.jd_ai_score.quantile(0.25):.4f}')
    print(f'      p50   = {df.jd_ai_score.quantile(0.50):.4f}')
    print(f'      p75   = {df.jd_ai_score.quantile(0.75):.4f}')
    print(f'      max   = {df.jd_ai_score.max():.4f}')
    print(f'      unique= {df.jd_ai_score.nunique():,}')
    return df


def merge_to_master(lookup_df):
    """将jd_ai_score合并回master CSV。"""
    print('[7] Merging jd_ai_score to master CSV ...')
    t0 = time.time()

    # 建立 title → score 映射
    score_map = dict(zip(lookup_df['招聘岗位'], lookup_df['jd_ai_score']))

    # 分块读写
    first = True
    n_matched = 0
    n_total = 0
    for chunk in pd.read_csv(MASTER_CSV, chunksize=500_000):
        chunk['jd_ai_score'] = chunk['招聘岗位'].map(score_map)
        n_matched += chunk['jd_ai_score'].notna().sum()
        n_total += len(chunk)

        if first:
            chunk.to_csv(MASTER_OUT, index=False, mode='w')
            first = False
        else:
            chunk.to_csv(MASTER_OUT, index=False, mode='a', header=False)

    print(f'    Merged: {n_matched:,}/{n_total:,} '
          f'({n_matched/n_total*100:.1f}%) matched')
    print(f'    Saved: {MASTER_OUT}')
    print(f'    Time: {time.time()-t0:.1f}s')


if __name__ == '__main__':
    results = compute_jd_ai_scores()
    lookup_df = save_lookup(results)
    merge_to_master(lookup_df)
    print('\nDone.')
