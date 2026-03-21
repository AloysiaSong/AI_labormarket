#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
k_selection.py — K值科学选择

对 K ∈ {20, 30, 40, 50, 60, 70, 80, 100} 分别训练 tomotopy LDA 模型，
计算 log-likelihood per word、C_V coherence、U_Mass coherence。

输出:
  output/k_robustness/lda_k{K}.bin     — 各K值模型
  output/k_robustness/k_metrics.csv    — K选择指标汇总
  output/k_robustness/k_coherence.png  — Coherence vs K 曲线

注意: K=50 如果 global_lda_fast.bin 已存在则直接复制，不重新训练。
"""

import json
import random
import shutil
import time
import csv
from pathlib import Path

import numpy as np
import tomotopy as tp
from tqdm import tqdm

# ── 路径 ────────────────────────────────────────────────────────────
BASE = Path('/Users/yu/code/code2601/TY')
INPUT_JSONL = BASE / 'output/processed_corpus.jsonl'
EXISTING_K50 = BASE / 'output/global_lda_fast.bin'
OUT_DIR = BASE / 'output/k_robustness'

# ── 参数 ────────────────────────────────────────────────────────────
K_VALUES = [20, 30, 40, 50, 60, 70, 80, 100]
MIN_CF = 10
RM_TOP = 15
ITERATIONS = 1000
MIN_TOKENS = 5
SAMPLE_RATE = 0.1
RANDOM_SEED = 42
COHERENCE_TOP_N = 20       # 每个主题取top-N词算coherence
COHERENCE_SAMPLE = 100_000  # coherence参考文本采样数


def load_sampled_corpus(rng):
    """流式采样10%语料，返回 list of list of str。"""
    docs = []
    with INPUT_JSONL.open('r', encoding='utf-8') as f:
        for line in f:
            if rng.random() > SAMPLE_RATE:
                continue
            try:
                obj = json.loads(line)
                tokens = obj.get('tokens', [])
                if len(tokens) >= MIN_TOKENS:
                    docs.append(tokens)
            except Exception:
                continue
    return docs


def train_model(docs, k):
    """训练 tomotopy LDA 并返回模型。"""
    model = tp.LDAModel(k=k, min_cf=MIN_CF, rm_top=RM_TOP, seed=RANDOM_SEED)
    for tokens in docs:
        model.add_doc(tokens)
    print(f'    Docs loaded: {len(model.docs):,}, Vocab: {len(model.used_vocabs):,}')

    for i in tqdm(range(0, ITERATIONS, 10), desc=f'    K={k} training', leave=False):
        step = min(10, ITERATIONS - i)
        model.train(step, workers=0)

    return model


def compute_coherence(model, docs_sample, metric='c_v'):
    """
    从 tomotopy 模型提取 top words，用 gensim CoherenceModel 算 coherence。
    """
    from gensim.corpora import Dictionary
    from gensim.models.coherencemodel import CoherenceModel

    # 提取每个主题的 top-N 词
    topics = []
    for k in range(model.k):
        top_words = [w for w, _ in model.get_topic_words(k, top_n=COHERENCE_TOP_N)]
        topics.append(top_words)

    # 构建 gensim dictionary
    dictionary = Dictionary(docs_sample)

    cm = CoherenceModel(
        topics=topics,
        texts=docs_sample,
        dictionary=dictionary,
        coherence=metric,
    )
    return cm.get_coherence()


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print('=' * 60)
    print('K-SELECTION: Train LDA for K ∈', K_VALUES)
    print('=' * 60)

    # 加载采样语料（所有K共用同一份）
    print('\n[1] Loading sampled corpus ...')
    rng = random.Random(RANDOM_SEED)
    docs = load_sampled_corpus(rng)
    print(f'  Sampled docs: {len(docs):,}')

    # Coherence参考文本（再采样一个子集，避免太慢）
    rng2 = random.Random(RANDOM_SEED + 1)
    if len(docs) > COHERENCE_SAMPLE:
        coh_docs = rng2.sample(docs, COHERENCE_SAMPLE)
    else:
        coh_docs = docs
    print(f'  Coherence reference docs: {len(coh_docs):,}')

    metrics = []

    for k in K_VALUES:
        print(f'\n{"="*60}')
        print(f'  K = {k}')
        print(f'{"="*60}')
        t0 = time.time()

        model_path = OUT_DIR / f'lda_k{k}.bin'

        # K=50 复用已有模型
        if k == 50 and EXISTING_K50.exists() and not model_path.exists():
            print(f'  Reusing existing K=50 model: {EXISTING_K50.name}')
            shutil.copy2(EXISTING_K50, model_path)
            model = tp.LDAModel.load(str(model_path))
        elif model_path.exists():
            print(f'  Loading existing model: {model_path.name}')
            model = tp.LDAModel.load(str(model_path))
        else:
            print(f'  Training new model ...')
            model = train_model(docs, k)
            model.save(str(model_path))
            print(f'  Saved: {model_path.name}')

        # Log-likelihood per word
        ll_pw = model.ll_per_word
        print(f'  LL/word: {ll_pw:.4f}')

        # Coherence C_V
        print(f'  Computing C_V coherence ...')
        cv = compute_coherence(model, coh_docs, metric='c_v')
        print(f'  C_V: {cv:.4f}')

        # Coherence U_Mass
        print(f'  Computing U_Mass coherence ...')
        umass = compute_coherence(model, coh_docs, metric='u_mass')
        print(f'  U_Mass: {umass:.4f}')

        elapsed = (time.time() - t0) / 60
        print(f'  Time: {elapsed:.1f} min')

        metrics.append({
            'K': k,
            'll_per_word': round(ll_pw, 6),
            'coherence_cv': round(cv, 4),
            'coherence_umass': round(umass, 4),
            'time_min': round(elapsed, 1),
        })

        del model  # 释放内存

    # 保存指标
    metrics_path = OUT_DIR / 'k_metrics.csv'
    with metrics_path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['K', 'll_per_word', 'coherence_cv', 'coherence_umass', 'time_min'])
        writer.writeheader()
        writer.writerows(metrics)
    print(f'\n✓ Metrics saved: {metrics_path}')

    # 画 coherence 曲线
    plot_coherence(metrics)

    print('\nDone.')


def plot_coherence(metrics):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print('  (matplotlib not available — skipping plot)')
        return

    ks = [m['K'] for m in metrics]
    cvs = [m['coherence_cv'] for m in metrics]
    umass = [m['coherence_umass'] for m in metrics]
    ll = [m['ll_per_word'] for m in metrics]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    # C_V
    ax = axes[0]
    ax.plot(ks, cvs, 'o-', color='steelblue', linewidth=2, markersize=8)
    best_cv_k = ks[np.argmax(cvs)]
    ax.axvline(best_cv_k, color='red', linestyle='--', alpha=0.5, label=f'Best K={best_cv_k}')
    ax.set_xlabel('Number of Topics (K)')
    ax.set_ylabel('Coherence (C_V)')
    ax.set_title('Topic Coherence (C_V)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # U_Mass
    ax = axes[1]
    ax.plot(ks, umass, 's-', color='darkorange', linewidth=2, markersize=8)
    best_um_k = ks[np.argmax(umass)]  # U_Mass: less negative = better
    ax.axvline(best_um_k, color='red', linestyle='--', alpha=0.5, label=f'Best K={best_um_k}')
    ax.set_xlabel('Number of Topics (K)')
    ax.set_ylabel('Coherence (U_Mass)')
    ax.set_title('Topic Coherence (U_Mass)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # LL per word
    ax = axes[2]
    ax.plot(ks, ll, 'D-', color='seagreen', linewidth=2, markersize=8)
    ax.set_xlabel('Number of Topics (K)')
    ax.set_ylabel('Log-likelihood per word')
    ax.set_title('Model Fit (LL/word)')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = OUT_DIR / 'k_coherence.png'
    plt.savefig(out_path, dpi=200)
    print(f'  Plot saved: {out_path}')
    plt.close()


if __name__ == '__main__':
    main()
