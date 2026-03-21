#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Step 2: Train Global LDA (tomotopy)
读取 processed_corpus.jsonl -> 训练 Global LDA -> 保存模型

输出：global_lda.bin
"""

from __future__ import annotations
from typing import Generator, List
from pathlib import Path
import json
import logging
import random
import time

import tomotopy as tp
from tqdm import tqdm


# =========================
# Config
# =========================
INPUT_JSONL = Path("/Users/yu/code/code2601/TY/output/processed_corpus.jsonl")
OUTPUT_MODEL = Path("/Users/yu/code/code2601/TY/output/global_lda.bin")

K = 50
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
    t0 = time.time()
    if not INPUT_JSONL.exists():
        logger.error(f"找不到输入文件: {INPUT_JSONL}")
        return

    # Reservoir sampling: 流式读取，内存只保留 SAMPLE_SIZE 篇
    logger.info(f"Reservoir sampling {SAMPLE_SIZE} 篇文档...")
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

    logger.info("开始训练 LDA ...")
    with tqdm(total=ITERATIONS, desc="LDA Training") as pbar:
        for i in range(0, ITERATIONS, LOG_EVERY):
            step = min(LOG_EVERY, ITERATIONS - i)
            model.train(step, workers=0)
            pbar.update(step)
            pbar.set_postfix({"ll_per_word": f"{model.ll_per_word:.4f}"})

    OUTPUT_MODEL.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(OUTPUT_MODEL))
    logger.info(f"模型已保存: {OUTPUT_MODEL}")
    logger.info(f"总耗时: {(time.time() - t0) / 60:.1f} 分钟")


if __name__ == "__main__":
    main()
