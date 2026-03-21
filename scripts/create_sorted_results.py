#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
创建 final_results_sample_sorted.csv
从 window_*.csv 文件按字母序读取，分配与 prepare_joint_mapping_industry20.py
reconstruct_company_job_by_id 一致的整数 ID，并用 global_lda_fast.bin 推断主题分布。
"""
import csv
import json
import math
import re
import sys
from pathlib import Path
from collections import defaultdict
import numpy as np

sys.path.insert(0, '/Users/yu/code/code2601/TY')

WINDOWS_DIR = Path('/Users/yu/code/code2601/TY/data/processed/windows')
PROCESSED_CORPUS = Path('/Users/yu/code/code2601/TY/output/processed_corpus.jsonl')
MODEL_PATH = Path('/Users/yu/code/code2601/TY/output/global_lda_fast.bin')
OUT_SORTED = Path('/Users/yu/code/code2601/TY/data/Heterogeneity/final_results_sample_sorted.csv')

MIN_TOKENS = 5
EPS = 1e-12
K = 50  # topics (must match model)

REPLACEMENTS = {"c++": "cpp", "c#": "csharp", ".net": "dotnet"}
ENGLISH_ALLOW = {"python", "java", "sql", "c", "r", "go", "cpp", "csharp", "dotnet"}
DEFAULT_STOPWORDS = set([
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


def is_chinese_token(token):
    return bool(re.search(r'[\u4e00-\u9fa5]', token))


def tokenize_jieba(text, stopwords, bigrams):
    import jieba
    t = clean_text(text)
    if not t: return []
    words = jieba.lcut(t)
    tokens = []
    for w in words:
        if w in stopwords or w.isdigit(): continue
        if is_chinese_token(w):
            if len(w) >= 2: tokens.append(w)
        else:
            if w in ENGLISH_ALLOW: tokens.append(w)
    # apply bigrams
    merged = []
    i = 0
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
    try: return int(float(str(v).strip()))
    except: return None


def load_bigrams():
    print('[1/3] 加载 bigrams ...')
    bigrams = set()
    with PROCESSED_CORPUS.open('r', encoding='utf-8') as f:
        for i, line in enumerate(f, 1):
            obj = json.loads(line)
            for t in obj.get('tokens', []):
                if '_' in t:
                    bigrams.add(t)
            if i % 2000000 == 0:
                print(f'  scanned {i:,}, bigrams={len(bigrams):,}')
    print(f'  total bigrams: {len(bigrams):,}')
    return bigrams


def load_model():
    import tomotopy as tp
    print('[2/3] 加载 LDA 模型 ...')
    mdl = tp.LDAModel.load(str(MODEL_PATH))
    vocabs = list(mdl.used_vocabs)
    word2idx = {w: i for i, w in enumerate(vocabs)}
    K_model = mdl.k
    V = len(vocabs)
    tw = np.zeros((K_model, V), dtype=np.float32)
    for k in range(K_model):
        tw[k] = mdl.get_topic_word_dist(k)
    log_tw = np.log(tw + 1e-12)
    alpha = np.array(mdl.alpha, dtype=np.float64)
    log_alpha = np.log(alpha + 1e-12)
    K_out = K_model
    del mdl
    print(f'  K={K_out}, V={V}')
    return word2idx, log_tw, log_alpha, K_out


def infer_doc(tokens, word2idx, log_tw, log_alpha, K_out):
    indices = [word2idx[t] for t in tokens if t in word2idx]
    if len(indices) < MIN_TOKENS:
        return None
    log_scores = log_tw[:, indices].mean(axis=1) + log_alpha
    log_scores -= log_scores.max()
    probs = np.exp(log_scores)
    s = probs.sum()
    if s < 1e-30: return None
    probs /= s
    log_k = math.log(K_out)
    entropy = float(-np.sum(probs * np.log(probs + EPS)) / log_k)
    hhi = float(np.dot(probs, probs))
    dom_id = int(np.argmax(probs))
    dom_prob = float(probs[dom_id])
    return entropy, hhi, dom_id, dom_prob


def main():
    bigrams = load_bigrams()
    word2idx, log_tw, log_alpha, K_out = load_model()
    stopwords = set(DEFAULT_STOPWORDS)

    print('[3/3] 处理 window 文件并推断 ...')
    OUT_SORTED.parent.mkdir(parents=True, exist_ok=True)
    
    year_seq = defaultdict(int)
    total = 0
    kept = 0
    year_stats = defaultdict(int)

    with OUT_SORTED.open('w', encoding='utf-8-sig', newline='') as out:
        w = csv.writer(out)
        w.writerow(['id', 'year', 'entropy_score', 'hhi_score', 'dominant_topic_id', 'dominant_topic_prob', 'token_count'])

        for fp in window_files():
            print(f'  处理: {fp.name}')
            with fp.open('r', encoding='utf-8-sig', errors='replace', newline='') as f:
                rd = csv.DictReader(f)
                for row in rd:
                    total += 1
                    year = parse_year(row.get('招聘发布年份'))
                    if year is None: continue
                    req = row.get('cleaned_requirements', '')
                    tokens = tokenize_jieba(req, stopwords, bigrams)
                    if len(tokens) < MIN_TOKENS: continue
                    result = infer_doc(tokens, word2idx, log_tw, log_alpha, K_out)
                    if result is None: continue
                    year_seq[year] += 1
                    jid = year * 10_000_000 + year_seq[year]
                    entropy, hhi, dom_id, dom_prob = result
                    w.writerow([jid, year, f'{entropy:.6f}', f'{hhi:.6f}', dom_id, f'{dom_prob:.6f}', len(tokens)])
                    kept += 1
                    year_stats[year] += 1
                    if total % 1_000_000 == 0:
                        print(f'    scanned={total:,}, kept={kept:,}')

    print(f'\n完成: 总扫描={total:,}, 有效输出={kept:,}')
    print('\n年份分布:')
    for y in sorted(year_stats.keys()):
        print(f'  {y}: {year_stats[y]:,}')
    print(f'\n输出: {OUT_SORTED}')


if __name__ == '__main__':
    main()
