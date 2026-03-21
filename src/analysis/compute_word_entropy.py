#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
compute_word_entropy.py

LDA 熵的稳健性检验：基于词汇多样性的替代指标。

对每条岗位描述，直接在词频分布上计算 Shannon 熵，
完全不依赖 LDA 主题模型，作为模型无关的稳健性检验：

  H_word = -Σ p(w) * log(p(w)) / log(V)   (归一化到 [0,1])

  其中 p(w) = 词w出现次数 / 总词数，V = 词类型数

还计算：
  unique_tokens = 不重复词数（词汇宽度）
  token_count   = 总词数（与 create_sorted_results.py 一致）

ID 分配方案与 create_sorted_results.py 完全一致：
  - 按字母序读 window 文件
  - year * 10_000_000 + year_seq[year]
  - 仅对 len(tokens) >= MIN_TOKENS 的行分配 ID
  - 注意：本脚本不做 word2idx 过滤（无需 LDA 模型）
    因此 ID 与最终 entropy 数据集不完全一致
    → 通过合并 final_results_sample_sorted.csv 按 ID 取交集

输出：
  data/Heterogeneity/word_entropy_by_id.csv
    列：id, year, word_entropy, unique_tokens, token_count
  data/Heterogeneity/yearly_word_entropy_stats.csv
    列：year, n, mean_word_entropy, mean_unique_tokens
"""

import csv
import math
import re
from collections import defaultdict
from pathlib import Path

BASE = Path('/Users/yu/code/code2601/TY')
WINDOWS_DIR = BASE / 'data/processed/windows'
CORPUS_JSONL = BASE / 'output/processed_corpus.jsonl'
MODEL_PATH  = BASE / 'output/global_lda_fast.bin'
OUT_BY_ID = BASE / 'data/Heterogeneity/word_entropy_by_id.csv'
OUT_YEARLY = BASE / 'data/Heterogeneity/yearly_word_entropy_stats.csv'

MIN_TOKENS = 5  # same as create_sorted_results.py

REPLACEMENTS = {"c++": "cpp", "c#": "csharp", ".net": "dotnet"}
ENGLISH_ALLOW = {"python", "java", "sql", "c", "r", "go", "cpp", "csharp", "dotnet"}
STOPWORDS = set([
    "的","了","在","是","有","和","与","或","等","能","会","及","对","可","为","被","把","让","给","向","从","到","以","于",
    "个","这","那","一","不","也","要","就","都","而","但","如","所","他","她","它","们","我","你","上","下","中","内","外",
    "职责","任职","要求","优先","具备","良好","负责","具有","熟悉","掌握","了解","相关","以上","以下","经验","工作","能力",
    "岗位","公司","企业","团队","一定","优秀","专业","学历","年","及以上","左右","不限","若干","可以","需要","进行","开展",
    "完成","参与","支持","提供","协助","配合","执行","根据","按照","通过","利用","使用","包括","主要","其他","相应","有效","合理","准确","及时",
])


def clean_text(text):
    if not text: return ''
    t = str(text).lower()
    for k, v in REPLACEMENTS.items():
        t = t.replace(k, v)
    t = re.sub(r'[^a-z\u4e00-\u9fa5]+', ' ', t)
    return re.sub(r'\s+', ' ', t).strip()


def tokenize(text):
    import jieba
    t = clean_text(text)
    if not t: return []
    words = jieba.lcut(t)
    tokens = []
    for w in words:
        if w in STOPWORDS or w.isdigit(): continue
        if re.search(r'[\u4e00-\u9fa5]', w):
            if len(w) >= 2: tokens.append(w)
        elif w in ENGLISH_ALLOW:
            tokens.append(w)
    return tokens


def word_entropy(tokens: list) -> tuple:
    """Compute normalized Shannon entropy of token frequency distribution."""
    if not tokens:
        return 0.0, 0, 0
    freq = defaultdict(int)
    for t in tokens:
        freq[t] += 1
    n = len(tokens)
    v = len(freq)
    if v <= 1:
        return 0.0, v, n
    h = -sum((c / n) * math.log(c / n) for c in freq.values())
    h_norm = h / math.log(v)  # normalize by log(vocab_size)
    return h_norm, v, n


def load_bigrams() -> set:
    import json
    print('  Loading bigrams from processed_corpus.jsonl...')
    bigrams = set()
    with CORPUS_JSONL.open(encoding='utf-8') as f:
        for line in f:
            for t in json.loads(line).get('tokens', []):
                if '_' in t:
                    bigrams.add(t)
    print(f'  {len(bigrams):,} bigrams loaded')
    return bigrams


def load_vocab() -> set:
    """Load LDA vocab set (does NOT run inference — fast)."""
    import tomotopy as tp
    print('  Loading LDA model vocab...')
    mdl = tp.LDAModel.load(str(MODEL_PATH))
    vocab = set(mdl.used_vocabs)
    del mdl
    print(f'  {len(vocab):,} vocab terms')
    return vocab


def apply_bigrams(tokens: list, bigrams: set) -> list:
    merged, i = [], 0
    while i < len(tokens) - 1:
        pair = tokens[i] + '_' + tokens[i + 1]
        if pair in bigrams:
            merged.append(pair)
            i += 2
        else:
            merged.append(tokens[i])
            i += 1
    if i == len(tokens) - 1:
        merged.append(tokens[-1])
    return merged


def window_files():
    files = [p for p in WINDOWS_DIR.glob('window_*.csv')
             if re.match(r'window_\d{4}_\d{4}\.csv', p.name)]
    return sorted(files)


def parse_year(v):
    try:
        return int(float(str(v).strip()))
    except Exception:
        return None


def main():
    print('[0/2] Loading model resources...')
    bigrams = load_bigrams()
    vocab   = load_vocab()

    print('[1/2] Processing window files...')
    OUT_BY_ID.parent.mkdir(parents=True, exist_ok=True)

    year_seq = defaultdict(int)
    yearly_stats = defaultdict(list)
    total = 0
    kept = 0

    with OUT_BY_ID.open('w', encoding='utf-8-sig', newline='') as fout:
        writer = csv.writer(fout)
        writer.writerow(['id', 'year', 'word_entropy', 'unique_tokens', 'token_count'])

        for fp in window_files():
            print(f'  {fp.name}')
            with fp.open('r', encoding='utf-8-sig', errors='replace', newline='') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    total += 1
                    year = parse_year(row.get('招聘发布年份'))
                    if year is None:
                        continue
                    req = row.get('cleaned_requirements', '')
                    tokens = tokenize(req)
                    tokens = apply_bigrams(tokens, bigrams)
                    if len(tokens) < MIN_TOKENS:
                        continue
                    # Apply same word2idx vocab filter as create_sorted_results.py
                    in_vocab = [t for t in tokens if t in vocab]
                    if len(in_vocab) < MIN_TOKENS:
                        continue

                    year_seq[year] += 1
                    jid = year * 10_000_000 + year_seq[year]
                    # Compute word entropy on ALL tokens (not just in-vocab)
                    h, unique, n = word_entropy(tokens)
                    writer.writerow([jid, year, f'{h:.6f}', unique, n])
                    kept += 1
                    yearly_stats[year].append(h)

                    if total % 1_000_000 == 0:
                        print(f'    scanned={total:,}, kept={kept:,}')

    print(f'\n完成: 扫描={total:,}, 输出={kept:,}')

    print('[2/2] Summarising yearly stats...')
    with OUT_YEARLY.open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['year', 'n', 'mean_word_entropy'])
        for year in sorted(yearly_stats.keys()):
            vals = yearly_stats[year]
            n = len(vals)
            mean_v = sum(vals) / n
            writer.writerow([year, n, f'{mean_v:.6f}'])

    print(f'输出: {OUT_BY_ID}')
    print(f'输出: {OUT_YEARLY}')

    # Print yearly summary
    print('\n── 年度词汇熵趋势 ──')
    with OUT_YEARLY.open(encoding='utf-8-sig') as f:
        for row in csv.DictReader(f):
            print(f"  {row['year']}: n={int(row['n']):>8,}  mean_word_entropy={float(row['mean_word_entropy']):.4f}")


if __name__ == '__main__':
    main()
