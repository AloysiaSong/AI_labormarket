#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Step 3: LDA Topic Inference (Analytical / Folding-in)
用贝叶斯解析法代替Gibbs sampling，速度快100倍+。
P(topic_k | doc) ∝ alpha_k * [Π P(w_i | topic_k)]^(1/n)

用法：
  python step3_infer.py          # 默认 K=100
  python step3_infer.py --k 50   # 指定 K

输入：output/processed_corpus.jsonl + output/global_lda_k{K}.bin
输出：output/topic_results_k{K}.csv
"""

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import argparse
import json
import csv
import math
import time
import multiprocessing
from pathlib import Path
from tqdm import tqdm
import numpy as np
import tomotopy as tp

# =========================
# 路径配置
# =========================
PROJECT_ROOT = Path(__file__).resolve().parents[2]  # -> TY/

MIN_TOKENS = 5
EPS = 1e-12
CHUNK_SIZE = 10000

# 噪声主题（K=100模型人工审查后确定）
# 49: 爬虫残留("由马整理")  75: 公众号模板("公众"0.146)  97: 公司地址/招聘信息
# 50: 职业路径(培训/晋升)  92: 驾驶/出差  95: 犯罪记录/背景审查
NOISE_TOPICS = np.array([49, 50, 75, 92, 95, 97])

# =========================
# Worker: 解析法推断（无Gibbs sampling）
# =========================
_word2idx = None
_log_tw = None      # shape (K, V), log P(word | topic)
_log_alpha = None   # shape (K,),   log alpha_k
_jsd_mat = None     # shape (K, K), 主题间 Jensen-Shannon 距离矩阵
_K_CLEAN = None     # 有效主题数（排除噪声后）
_MODEL_PATH = None  # worker进程需要知道模型路径

def _compute_jsd_matrix(tw):
    """计算主题间 Jensen-Shannon 散度矩阵（对称）"""
    K_model = tw.shape[0]
    jsd = np.zeros((K_model, K_model), dtype=np.float64)
    for i in range(K_model):
        for j in range(i + 1, K_model):
            m = 0.5 * (tw[i] + tw[j])
            kl_im = np.sum(tw[i] * np.log(tw[i] / (m + EPS) + EPS))
            kl_jm = np.sum(tw[j] * np.log(tw[j] / (m + EPS) + EPS))
            d = 0.5 * (kl_im + kl_jm)
            jsd[i, j] = d
            jsd[j, i] = d
    return jsd

def _worker_init(model_path):
    """每个worker加载模型，提取词-主题矩阵，然后释放模型"""
    global _word2idx, _log_tw, _log_alpha, _jsd_mat, _K_CLEAN, _MODEL_PATH
    _MODEL_PATH = model_path

    mdl = tp.LDAModel.load(str(_MODEL_PATH))

    # 动态读取K
    K_model = mdl.k
    _K_CLEAN = K_model - len(NOISE_TOPICS)

    # 构建词 -> 索引映射
    vocabs = list(mdl.used_vocabs)
    _word2idx = {w: i for i, w in enumerate(vocabs)}

    # 提取 K x V 的 log P(word|topic) 矩阵
    V = len(vocabs)
    tw = np.zeros((K_model, V), dtype=np.float32)
    for k in range(K_model):
        tw[k] = mdl.get_topic_word_dist(k)
    _log_tw = np.log(tw + 1e-12)

    # 主题间 JSD 距离矩阵（用于 Rao's Q）
    _jsd_mat = _compute_jsd_matrix(tw)

    # 主题先验 alpha
    alpha = np.array(mdl.alpha, dtype=np.float64)
    _log_alpha = np.log(alpha + 1e-12)

    # 模型本身不再需要，释放内存
    del mdl


def worker_task(lines_batch):
    """解析法计算主题分布 + 多维指标"""
    global _word2idx, _log_tw, _log_alpha, _jsd_mat, _K_CLEAN

    results = []
    error_sample = None
    log_k = math.log(_K_CLEAN)
    tau = 1.0 / _K_CLEAN  # 显著主题阈值

    for line in lines_batch:
        try:
            obj = json.loads(line)
            jid = obj.get("id") or obj.get("job_id") or "unknown"
            year = obj.get("year", "")
            tokens = obj.get("tokens", [])

            if len(tokens) < MIN_TOKENS:
                continue

            # 查词表，获取已知词的索引
            indices = [_word2idx[t] for t in tokens if t in _word2idx]
            if len(indices) < MIN_TOKENS:
                continue

            # P(topic_k | doc) ∝ alpha_k * [Π P(w_i | topic_k)]^(1/n)
            log_scores = _log_tw[:, indices].mean(axis=1) + _log_alpha

            # log-sum-exp 归一化
            log_scores -= log_scores.max()
            probs = np.exp(log_scores)
            s = probs.sum()
            if s < 1e-30:
                continue
            probs /= s

            # 排除噪声主题，重新归一化
            if len(NOISE_TOPICS) > 0:
                probs[NOISE_TOPICS] = 0.0
                s2 = probs.sum()
                if s2 < 1e-30:
                    continue
                probs /= s2

            # === 指标计算 ===
            # 1. Shannon Entropy (归一化到 [0,1])
            entropy = float(-np.sum(probs * np.log(probs + EPS)) / log_k)

            # 2. HHI
            hhi = float(np.dot(probs, probs))

            # 3. Dominant topic
            dom_id = int(np.argmax(probs))
            dom_prob = float(probs[dom_id])

            # 4. Effective Number of Topics: exp(H)
            H_raw = float(-np.sum(probs * np.log(probs + EPS)))
            ent = float(np.exp(H_raw))

            # 5. Rao's Quadratic Entropy: Q = Σ_ij d_ij * p_i * p_j
            rao_q = float(probs @ _jsd_mat @ probs)

            # 6. Gini coefficient
            p_sorted = np.sort(probs[probs > 0])
            n = len(p_sorted)
            if n > 1:
                idx = np.arange(1, n + 1)
                gini = float((2 * np.sum(idx * p_sorted) - (n + 1) * np.sum(p_sorted)) / (n * np.sum(p_sorted)))
            else:
                gini = 1.0

            # 7. Significant Topic Count: N_tau (p_k > 1/K_clean)
            n_tau = int(np.sum(probs > tau))

            # 8. Tail Mass Ratio: 1 - sum(top-3)
            top3 = float(np.sum(np.sort(probs)[-3:]))
            tmr = float(1.0 - top3)

            results.append([
                str(jid), str(year),
                f"{entropy:.6f}", f"{hhi:.6f}",
                str(dom_id), f"{dom_prob:.6f}",
                f"{ent:.4f}", f"{rao_q:.6f}",
                f"{gini:.6f}", str(n_tau), f"{tmr:.6f}"
            ])
        except Exception as e:
            if error_sample is None:
                error_sample = f"ROW_ERROR: {e} | DATA: {line[:80]}..."
            continue

    if error_sample:
        results.append(["__DEBUG_ERROR__", error_sample])

    return results

# =========================
# Main
# =========================
def _iter_chunks(jsonl_path, chunk_size):
    """流式读取 JSONL，每次 yield 一个 chunk（list of lines）"""
    buf = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            buf.append(line)
            if len(buf) >= chunk_size:
                yield buf
                buf = []
    if buf:
        yield buf


def main():
    global _MODEL_PATH

    parser = argparse.ArgumentParser(description="LDA Topic Inference")
    parser.add_argument("--k", type=int, default=100, help="主题数K (default: 100)")
    args = parser.parse_args()
    K = args.k

    INPUT_JSONL = PROJECT_ROOT / "output/processed_corpus.jsonl"
    _MODEL_PATH = PROJECT_ROOT / f"output/global_lda_k{K}.bin"
    OUTPUT_CSV = PROJECT_ROOT / f"output/topic_results_k{K}.csv"

    t0 = time.time()

    if not _MODEL_PATH.exists():
        print(f"找不到模型文件: {_MODEL_PATH}")
        return
    if not INPUT_JSONL.exists():
        print(f"找不到输入文件: {INPUT_JSONL}")
        return

    # 统计总行数（流式，不占内存）
    print("[1/3] 统计文档数...")
    total_docs = 0
    with open(INPUT_JSONL, 'r', encoding='utf-8') as f:
        for _ in f:
            total_docs += 1
    num_chunks = (total_docs + CHUNK_SIZE - 1) // CHUNK_SIZE
    print(f"  共 {total_docs} 条, {num_chunks} 个块")

    # 减少 worker 数量避免 OOM（每个 worker 加载模型占 ~200MB）
    num_cores = min(4, multiprocessing.cpu_count())
    print(f"[2/3] 启动并行: {num_cores} 核 | K={K} | 解析法(无Gibbs)")

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    ctx = multiprocessing.get_context('spawn')

    print("[3/3] 开始推断...")
    error_print_count = 0
    total_written = 0

    with open(OUTPUT_CSV, 'w', encoding='utf-8-sig', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            "id", "year", "entropy_score", "hhi_score",
            "dominant_topic_id", "dominant_topic_prob",
            "ent_effective", "rao_q", "gini", "n_sig_topics", "tail_mass_ratio"
        ])

        with ctx.Pool(processes=num_cores, initializer=_worker_init, initargs=(_MODEL_PATH,)) as pool:
            for batch_res in tqdm(pool.imap(worker_task, _iter_chunks(INPUT_JSONL, CHUNK_SIZE)), total=num_chunks, desc="Inferring"):
                if not batch_res:
                    continue

                valid_rows = []
                for row in batch_res:
                    if row[0] == "__DEBUG_ERROR__":
                        if error_print_count < 5:
                            print(f"\n  Warning: {row[1]}")
                            error_print_count += 1
                    else:
                        valid_rows.append(row)

                if valid_rows:
                    writer.writerows(valid_rows)
                    f.flush()
                    total_written += len(valid_rows)

    elapsed = (time.time() - t0) / 60
    print(f"Done! {total_written} rows -> {OUTPUT_CSV}")
    print(f"Time: {elapsed:.1f} min")


if __name__ == "__main__":
    main()
