#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Step 1: Train Word2Vec on processed_corpus.jsonl
全量训练，CBOW，window=10，200维
"""

from __future__ import annotations
import json
import logging
import time
from pathlib import Path

from gensim.models import Word2Vec

# =========================
# Config
# =========================
INPUT_JSONL = Path("/Users/yu/code/code2601/TY/output/processed_corpus.jsonl")
OUTPUT_DIR = Path("/Users/yu/code/code2601/TY/output/w2v")
OUTPUT_MODEL = OUTPUT_DIR / "word2vec.model"

VECTOR_SIZE = 200       # 和 Tong et al. 一致
WINDOW = 10             # JD词序意义不大，用大窗口
SG = 0                  # CBOW
MIN_COUNT = 10          # 和 LDA min_cf=10 对齐
WORKERS = 4             # 控制CPU使用
EPOCHS = 5              # gensim默认
SEED = 42
MIN_TOKENS = 5          # 和 LDA 一致，跳过极短文档

# =========================
# Logging
# =========================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("W2V-Train")


# =========================
# Streaming corpus iterator
# =========================
class CorpusIterator:
    """流式读取 JSONL，支持多轮迭代（gensim 需要两轮：build_vocab + train）"""

    def __init__(self, jsonl_path: Path, min_tokens: int = 5):
        self.jsonl_path = jsonl_path
        self.min_tokens = min_tokens
        self.doc_count = 0

    def __iter__(self):
        self.doc_count = 0
        with self.jsonl_path.open("r", encoding="utf-8") as f:
            for line in f:
                try:
                    obj = json.loads(line)
                    tokens = obj.get("tokens", [])
                    if len(tokens) >= self.min_tokens:
                        self.doc_count += 1
                        yield tokens
                except Exception:
                    continue


# =========================
# Main
# =========================
def main() -> None:
    t0 = time.time()

    if not INPUT_JSONL.exists():
        logger.error(f"找不到输入文件: {INPUT_JSONL}")
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    corpus = CorpusIterator(INPUT_JSONL, min_tokens=MIN_TOKENS)

    logger.info("=== Word2Vec 训练配置 ===")
    logger.info(f"  vector_size={VECTOR_SIZE}, window={WINDOW}, sg={SG}(CBOW)")
    logger.info(f"  min_count={MIN_COUNT}, workers={WORKERS}, epochs={EPOCHS}")
    logger.info(f"  输入: {INPUT_JSONL}")

    # 第一轮：build vocab
    logger.info("第一轮扫描：构建词表...")
    t1 = time.time()
    model = Word2Vec(
        vector_size=VECTOR_SIZE,
        window=WINDOW,
        sg=SG,
        min_count=MIN_COUNT,
        workers=WORKERS,
        seed=SEED,
        epochs=EPOCHS,
    )
    model.build_vocab(corpus, progress_per=1_000_000)
    vocab_time = (time.time() - t1) / 60
    vocab_size = len(model.wv)
    logger.info(f"词表构建完成: {vocab_size:,} 个词 | 文档数: {corpus.doc_count:,} | 耗时: {vocab_time:.1f} 分钟")

    # 第二轮：train
    logger.info("第二轮扫描：训练词向量...")
    t2 = time.time()
    model.train(corpus, total_examples=corpus.doc_count, epochs=EPOCHS)
    train_time = (time.time() - t2) / 60
    logger.info(f"训练完成 | 耗时: {train_time:.1f} 分钟")

    # 保存
    model.save(str(OUTPUT_MODEL))
    logger.info(f"模型已保存: {OUTPUT_MODEL}")

    # 也保存纯词向量（更小，加载更快）
    kv_path = OUTPUT_DIR / "word2vec.kv"
    model.wv.save(str(kv_path))
    logger.info(f"词向量已保存: {kv_path}")

    # 汇总
    total_time = (time.time() - t0) / 60
    logger.info("=== 训练汇总 ===")
    logger.info(f"  词表大小: {vocab_size:,}")
    logger.info(f"  文档数: {corpus.doc_count:,}")
    logger.info(f"  总耗时: {total_time:.1f} 分钟")

    # 快速质检：打印几个词的最近邻
    test_words = ["python", "沟通", "管理", "数据分析", "机器学习", "财务", "销售"]
    logger.info("=== 快速质检：最近邻词 ===")
    for w in test_words:
        if w in model.wv:
            neighbors = model.wv.most_similar(w, topn=5)
            neighbor_str = ", ".join(f"{n[0]}({n[1]:.2f})" for n in neighbors)
            logger.info(f"  {w} → {neighbor_str}")
        else:
            logger.info(f"  {w} → (不在词表中)")


if __name__ == "__main__":
    main()
