#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Step 2: Train Global LDA (tomotopy)
读取 processed_corpus.jsonl -> 训练 Global LDA -> 保存模型

用法：
  python step2_train.py          # 默认 K=100
  python step2_train.py --k 50   # 指定 K
  python step2_train.py --k 150

输出：output/global_lda_k{K}.bin
"""

from __future__ import annotations
from typing import Generator, List
from pathlib import Path
import argparse
import json
import logging
import random
import time

import tomotopy as tp
from tqdm import tqdm


# =========================
# 路径配置
# =========================
PROJECT_ROOT = Path(__file__).resolve().parents[2]  # -> TY/

INPUT_JSONL = PROJECT_ROOT / "output/processed_corpus.jsonl"

MIN_CF = 50               # 从10提升到50，过滤极低频噪声词
RM_TOP = 50               # 从15提升到50，移除更多全局高频泛化词
ITERATIONS = 500
LOG_EVERY = 10
MIN_TOKENS = 5
SAMPLE_SIZE = 2_000_000   # 随机抽样训练，全量用step3 folding-in推断
RANDOM_SEED = 42


# =========================
# Logging
# =========================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("Step2-Train")


# =========================
# Streaming JSONL
# =========================
def iter_docs(jsonl_path: Path) -> Generator[List[str], None, None]:
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                obj = json.loads(line)
                tokens = obj.get("tokens", [])
                if len(tokens) >= MIN_TOKENS:
                    yield tokens
            except Exception:
                logger.warning("跳过损坏JSON行")
                continue


# =========================
# Main
# =========================
def main() -> None:
    parser = argparse.ArgumentParser(description="Train Global LDA")
    parser.add_argument("--k", type=int, default=100, help="主题数K (default: 100)")
    args = parser.parse_args()
    K = args.k

    OUTPUT_MODEL = PROJECT_ROOT / f"output/global_lda_k{K}.bin"

    t0 = time.time()
    if not INPUT_JSONL.exists():
        logger.error(f"找不到输入文件: {INPUT_JSONL}")
        return

    # Reservoir sampling: 流式读取，内存只保留 SAMPLE_SIZE 篇
    logger.info(f"Reservoir sampling {SAMPLE_SIZE} 篇文档 (K={K})...")
    random.seed(RANDOM_SEED)
    sampled: List[List[str]] = []
    total = 0
    for tokens in tqdm(iter_docs(INPUT_JSONL), desc="Sampling"):
        total += 1
        if total <= SAMPLE_SIZE:
            sampled.append(tokens)
        else:
            j = random.randint(0, total - 1)
            if j < SAMPLE_SIZE:
                sampled[j] = tokens
    logger.info(f"从 {total} 篇中抽样 {len(sampled)} 篇用于训练")

    model = tp.LDAModel(k=K, min_cf=MIN_CF, rm_top=RM_TOP)

    logger.info("添加文档到模型...")
    for tokens in tqdm(sampled, desc="Add Docs"):
        model.add_doc(tokens)
    del sampled

    logger.info(f"开始训练 LDA (K={K}, iterations={ITERATIONS}) ...")
    with tqdm(total=ITERATIONS, desc="LDA Training") as pbar:
        for i in range(0, ITERATIONS, LOG_EVERY):
            step = min(LOG_EVERY, ITERATIONS - i)
            model.train(step, workers=0)
            pbar.update(step)
            pbar.set_postfix({"ll_per_word": f"{model.ll_per_word:.4f}"})

    OUTPUT_MODEL.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(OUTPUT_MODEL))
    logger.info(f"模型已保存: {OUTPUT_MODEL}")

    # 打印每个主题的 top-20 关键词，方便快速审查
    logger.info(f"\n{'='*60}")
    logger.info(f"主题关键词 (K={K}, top-20)")
    logger.info(f"{'='*60}")
    for k in range(K):
        words = model.get_topic_words(k, top_n=20)
        word_str = " | ".join([f"{w}({p:.3f})" for w, p in words])
        print(f"  Topic {k:>3d}: {word_str}")

    logger.info(f"总耗时: {(time.time() - t0) / 60:.1f} 分钟")


if __name__ == "__main__":
    main()
