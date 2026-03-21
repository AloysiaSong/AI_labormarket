#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
k_sensitivity.py — K值敏感性趋势分析

对 output/k_robustness/lda_k{K}.bin 中的每个模型，
用解析法 folding-in 推断全量语料，计算每年归一化 entropy 均值。

输出:
  output/k_robustness/yearly_entropy_by_k.csv   — 年度entropy对比
  output/k_robustness/k_sensitivity_trend.png    — 多K趋势叠加图
  output/k_robustness/k_sensitivity_slope.png    — 趋势斜率对K的敏感性
"""

import json
import math
import time
import csv
from pathlib import Path
from collections import defaultdict

import numpy as np
from tqdm import tqdm

# ── 路径 ────────────────────────────────────────────────────────────
BASE = Path('/Users/yu/code/code2601/TY')
INPUT_JSONL = BASE / 'output/processed_corpus.jsonl'
MODEL_DIR = BASE / 'output/k_robustness'
OUT_DIR = MODEL_DIR

K_VALUES = [20, 30, 40, 50, 60, 70, 80, 100]
MIN_TOKENS = 5
EPS = 1e-12
EXCL_YEARS = {2019, 2020, 2025}  # 2025 partial year (too few obs)


def load_model_matrices(model_path):
    """加载 tomotopy 模型，提取推断所需矩阵，释放模型。"""
    import tomotopy as tp
    mdl = tp.LDAModel.load(str(model_path))
    vocabs = list(mdl.used_vocabs)
    word2idx = {w: i for i, w in enumerate(vocabs)}

    K = mdl.k
    V = len(vocabs)
    tw = np.zeros((K, V), dtype=np.float32)
    for k in range(K):
        tw[k] = mdl.get_topic_word_dist(k)
    log_tw = np.log(tw + EPS)

    alpha = np.array(mdl.alpha, dtype=np.float64)
    log_alpha = np.log(alpha + EPS)

    del mdl
    return word2idx, log_tw, log_alpha, K


def infer_corpus(word2idx, log_tw, log_alpha, K):
    """对全量语料做 folding-in 推断，返回 {year: [entropy_list]}。"""
    log_k = math.log(K)
    year_entropies = defaultdict(list)

    with INPUT_JSONL.open('r', encoding='utf-8') as f:
        for line in tqdm(f, desc=f'    K={K} infer', leave=False):
            try:
                obj = json.loads(line)
                year = int(float(obj.get('year', 0)))
                if year in EXCL_YEARS:
                    continue
                tokens = obj.get('tokens', [])
                if len(tokens) < MIN_TOKENS:
                    continue

                indices = [word2idx[t] for t in tokens if t in word2idx]
                if len(indices) < MIN_TOKENS:
                    continue

                # Geometric mean folding-in
                log_scores = log_tw[:, indices].mean(axis=1) + log_alpha
                log_scores -= log_scores.max()
                probs = np.exp(log_scores)
                s = probs.sum()
                if s < 1e-30:
                    continue
                probs /= s

                # Normalized entropy
                entropy = float(-np.sum(probs * np.log(probs + EPS)) / log_k)
                year_entropies[year].append(entropy)

            except Exception:
                continue

    return year_entropies


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print('=' * 60)
    print('K-SENSITIVITY: Entropy trends across K values')
    print('=' * 60)

    # 检查哪些模型存在
    available_ks = []
    for k in K_VALUES:
        p = MODEL_DIR / f'lda_k{k}.bin'
        if p.exists():
            available_ks.append(k)
        else:
            print(f'  WARNING: model for K={k} not found, skipping')

    if not available_ks:
        print('ERROR: No models found. Run k_selection.py first.')
        return

    # 对每个K做推断
    all_results = {}  # {K: {year: (mean, std, n)}}

    for k in available_ks:
        print(f'\n── K={k} ──')
        model_path = MODEL_DIR / f'lda_k{k}.bin'
        t0 = time.time()

        word2idx, log_tw, log_alpha, K_actual = load_model_matrices(model_path)
        assert K_actual == k, f'Model K={K_actual} != expected {k}'

        year_entropies = infer_corpus(word2idx, log_tw, log_alpha, k)

        yearly = {}
        for year in sorted(year_entropies.keys()):
            arr = np.array(year_entropies[year])
            yearly[year] = (float(arr.mean()), float(arr.std()), len(arr))
            print(f'    {year}: mean={arr.mean():.4f}, n={len(arr):,}')

        all_results[k] = yearly
        elapsed = (time.time() - t0) / 60
        print(f'    Time: {elapsed:.1f} min')

        # 释放
        del word2idx, log_tw, log_alpha

    # 保存 CSV
    years = sorted(set(y for yearly in all_results.values() for y in yearly.keys()))

    csv_path = OUT_DIR / 'yearly_entropy_by_k.csv'
    with csv_path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        header = ['year'] + [f'K={k}_mean' for k in available_ks] + [f'K={k}_n' for k in available_ks]
        writer.writerow(header)
        for year in years:
            row = [year]
            for k in available_ks:
                stats = all_results[k].get(year, (np.nan, np.nan, 0))
                row.append(f'{stats[0]:.6f}')
            for k in available_ks:
                stats = all_results[k].get(year, (np.nan, np.nan, 0))
                row.append(stats[2])
            writer.writerow(row)
    print(f'\n✓ Saved: {csv_path}')

    # 画图
    plot_trends(all_results, available_ks, years)
    plot_slopes(all_results, available_ks, years)

    print('\nDone.')


def plot_trends(all_results, ks, years):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print('  (matplotlib not available)')
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(ks)))

    for k, color in zip(ks, colors):
        yearly = all_results[k]
        ys = [y for y in years if y in yearly]
        means = [yearly[y][0] for y in ys]
        ax.plot(ys, means, 'o-', color=color, linewidth=2, markersize=5, label=f'K={k}')

    # 标注2019-2020数据断层
    ax.axvspan(2018.5, 2020.5, alpha=0.1, color='gray')
    ax.text(2019.5, ax.get_ylim()[0] + 0.001, 'gap', ha='center', fontsize=8, color='gray')

    ax.set_xlabel('Year', fontsize=12)
    ax.set_ylabel('Normalized Entropy (SCI)', fontsize=12)
    ax.set_title('Skill Comprehensiveness Trend Across Different K Values\n'
                 '(Normalized by log K, range [0,1])', fontsize=13)
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))

    plt.tight_layout()
    out = OUT_DIR / 'k_sensitivity_trend.png'
    plt.savefig(out, dpi=200)
    print(f'  Trend plot saved: {out}')
    plt.close()


def plot_slopes(all_results, ks, years):
    """计算每个K下的 entropy 年趋势斜率 (2016-2024 OLS)，画 bar chart。"""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        return

    slopes = []
    for k in ks:
        yearly = all_results[k]
        valid_years = [y for y in years if y in yearly and y <= 2024]
        if len(valid_years) < 3:
            slopes.append(0)
            continue
        x = np.array(valid_years, dtype=float)
        y = np.array([yearly[yr][0] for yr in valid_years])
        # Simple OLS: slope = cov(x,y) / var(x)
        slope = np.polyfit(x, y, 1)[0]
        slopes.append(slope)

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ['steelblue' if k != 50 else 'firebrick' for k in ks]
    bars = ax.bar([str(k) for k in ks], [s * 1000 for s in slopes], color=colors, edgecolor='white')

    # 标注K=50
    for i, k in enumerate(ks):
        if k == 50:
            bars[i].set_edgecolor('black')
            bars[i].set_linewidth(2)

    ax.set_xlabel('Number of Topics (K)', fontsize=12)
    ax.set_ylabel('Entropy Trend Slope (×10³ per year)', fontsize=12)
    ax.set_title('Sensitivity of Entropy Trend to K\n'
                 '(OLS slope of normalized entropy on year)', fontsize=13)
    ax.grid(True, alpha=0.3, axis='y')

    # 标注数值
    for bar, s in zip(bars, slopes):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{s*1000:.2f}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    out = OUT_DIR / 'k_sensitivity_slope.png'
    plt.savefig(out, dpi=200)
    print(f'  Slope plot saved: {out}')
    plt.close()


if __name__ == '__main__':
    main()
