#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Step 2: Train Global LDA (tomotopy) - 极速采样版
仅抽样部分语料以加速训练，适合快速探索主题结构。
"""

from __future__ import annotations
from typing import Generator, List
from pathlib import Path
import json
import logging
import time
import random

import tomotopy as tp
from tqdm import tqdm

# =========================
# Config (参数大修)
# =========================
INPUT_JSONL = Path("/Users/yu/code/code2601/TY/output/processed_corpus.jsonl")
OUTPUT_MODEL = Path("/Users/yu/code/code2601/TY/output/global_lda_fast.bin")

K = 50
MIN_CF = 10
RM_TOP = 15
ITERATIONS = 1000
LOG_EVERY = 1       # 每一轮都汇报进度
MIN_TOKENS = 5
SAMPLE_RATE = 0.1   # 仅采样 10% 语料
RANDOM_SEED = 42

# =========================
# Logging
# =========================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("Step2-Train-Fast")


# =========================
# Streaming JSONL
# =========================
def iter_docs(jsonl_path: Path, rng: random.Random) -> Generator[List[str], None, None]:
    """流式读取 + 随机采样"""
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                if rng.random() > SAMPLE_RATE:
                    continue
                obj = json.loads(line)
                tokens = obj.get("tokens", [])
                if len(tokens) >= MIN_TOKENS:
                    yield tokens
            except Exception:
                continue


# =========================
# Main
# =========================
def main() -> None:
    t0 = time.time()

    logger.info(f"极速模式启动：采样率 = {SAMPLE_RATE*100:.1f}%")

    if not INPUT_JSONL.exists():
        logger.error(f"找不到输入文件: {INPUT_JSONL}")
        return

    rng = random.Random(RANDOM_SEED)

    # 初始化模型
    model = tp.LDAModel(k=K, min_cf=MIN_CF, rm_top=RM_TOP, seed=RANDOM_SEED)

    logger.info("加载并采样语料...")
    for tokens in tqdm(iter_docs(INPUT_JSONL, rng), desc="Add Docs"):
        model.add_doc(tokens)

    logger.info(f"最终用于训练的文档数: {len(model.docs):,}")
    logger.info("开始训练 LDA ...")

    with tqdm(total=ITERATIONS, desc="LDA Training") as pbar:
        for i in range(0, ITERATIONS, LOG_EVERY):
            step = min(LOG_EVERY, ITERATIONS - i)
            model.train(step, workers=0)
            pbar.update(step)
            pbar.set_postfix({"ll": f"{model.ll_per_word:.2f}"})

    OUTPUT_MODEL.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(OUTPUT_MODEL))
    logger.info(f"模型已保存: {OUTPUT_MODEL}")
    logger.info(f"总耗时: {(time.time() - t0) / 60:.1f} 分钟")


if __name__ == "__main__":
    main()
