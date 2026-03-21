#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Global LDA on massive Chinese JDs (streaming, tomotopy, single-node)
Pipeline:
1) Stream CSVs recursively (row-by-row), skip bad lines
2) On-the-fly tokenization (jieba) + regex cleaning + stopwords
3) Optional bigram detection (two-pass, memory-capped)
4) Train global LDA with tomotopy (K=50, min_cf=10, rm_top=15)
5) Re-stream data to infer topic distributions and compute:
   - Normalized Shannon Entropy
   - HHI
   - Dominant topic
6) Stream results to output CSV (buffered writes)
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Generator, Iterable, List, Optional, Tuple, Dict
from pathlib import Path
import csv
import re
import math
import logging
import time
from collections import Counter
from tqdm import tqdm
import tomotopy as tp
import jieba


# =========================
# Config (Edit as needed)
# =========================
INPUT_DIR = Path("/Users/yu/code/code2601/TY/data/processed/windows")  # CSV根目录（只会用 window_2016_2017.csv ... window_2024_2025.csv 这5个窗口文件）
OUTPUT_MODEL = Path("/Users/yu/code/code2601/TY/output/global_lda_model.bin")
OUTPUT_RESULT = Path("/Users/yu/code/code2601/TY/output/jd_topic_metrics.csv")

# Column candidates (按优先顺序)
ID_COLS = ["id", "ID", "jd_id", "职位ID", "岗位ID"]
YEAR_COLS = ["year", "年份", "招聘发布年份", "发布年份"]
TEXT_COLS = ["cleaned_requirements", "职位描述", "岗位描述", "要求", "职责"]

# LDA hyperparams
K = 50
MIN_CF = 10
RM_TOP = 15
ITERATIONS = 2000
LOG_EVERY = 100

# Inference
MIN_TOKENS = 5
EPS = 1e-12

# Output buffering
BUFFER_SIZE = 10000

# Bigram (optional)
ENABLE_BIGRAMS = True
BIGRAM_MIN_COUNT = 20
BIGRAM_MAX_PAIRS = 200000  # 受控内存上限

# Word cleaning
ENGLISH_ALLOW = {"python", "java", "sql", "c", "r", "go", "cpp", "csharp", "dotnet"}
REPLACEMENTS = {
    "c++": "cpp",
    "c#": "csharp",
    ".net": "dotnet",
}

# Stopwords
DEFAULT_STOPWORDS = set([
    # 通用停用词
    "的","了","在","是","有","和","与","或","等","能","会","及","对","可","为","被","把","让","给","向","从","到","以","于",
    "个","这","那","一","不","也","要","就","都","而","但","如","所","他","她","它","们","我","你","上","下","中","内","外",
    # HR常见泛化词
    "职责","任职","要求","优先","具备","良好","负责","具有","熟悉","掌握","了解","相关","以上","以下","经验","工作","能力",
    "岗位","公司","企业","团队","一定","优秀","专业","学历","年","及以上","左右","不限","若干","可以","需要","进行","开展",
    "完成","参与","支持","提供","协助","配合","执行","根据","按照","通过","利用","使用","包括","主要","其他","相应","有效","合理","准确","及时",
])

# =========================
# Logging
# =========================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("GlobalLDA")


# =========================
# Utilities
# =========================
def find_first_existing(fields: List[str], header: List[str]) -> Optional[str]:
    header_set = set(header)
    for f in fields:
        if f in header_set:
            return f
    return None

def clean_text(text: str) -> str:
    if not text:
        return ""
    t = text.lower()
    for k, v in REPLACEMENTS.items():
        t = t.replace(k, v)
    # 保留中文和英文字符，移除数字/符号
    t = re.sub(r"[^a-z\u4e00-\u9fa5]+", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t

def tokenize(text: str, stopwords: set) -> List[str]:
    t = clean_text(text)
    if not t:
        return []
    words = jieba.lcut(t)
    tokens = []
    for w in words:
        if len(w) < 2 and w not in ENGLISH_ALLOW:
            continue
        if w in stopwords:
            continue
        if w.isdigit():
            continue
        tokens.append(w)
    return tokens

def apply_bigrams(tokens: List[str], bigrams: set) -> List[str]:
    if not tokens:
        return tokens
    merged = []
    i = 0
    while i < len(tokens) - 1:
        pair = tokens[i] + "_" + tokens[i+1]
        if pair in bigrams:
            merged.append(pair)
            i += 2
        else:
            merged.append(tokens[i])
            i += 1
    if i == len(tokens) - 1:
        merged.append(tokens[-1])
    return merged


# =========================
# Streaming Data Reader
# =========================
@dataclass
class JDRow:
    jid: str
    year: str
    text: str

class DataStreamer:
    def __init__(self, root_dir: Path):
        self.root_dir = root_dir

    def iter_rows(self) -> Generator[JDRow, None, None]:
        pattern = re.compile(r"^window_\d{4}_\d{4}\.csv$")
        files = [f for f in self.root_dir.rglob("*.csv") if pattern.match(f.name)]
        logger.info(f"发现CSV文件: {len(files)} 个")
        for f in files:
            try:
                with f.open("r", encoding="utf-8-sig", errors="replace", newline="") as fh:
                    reader = csv.DictReader(fh)
                    if reader.fieldnames is None:
                        logger.warning(f"跳过空文件: {f}")
                        continue
                    id_col = find_first_existing(ID_COLS, reader.fieldnames)
                    year_col = find_first_existing(YEAR_COLS, reader.fieldnames)
                    text_col = find_first_existing(TEXT_COLS, reader.fieldnames)

                    if not year_col or not text_col:
                        logger.warning(f"缺少关键列，跳过: {f.name}")
                        continue

                    for row in reader:
                        try:
                            jid = row.get(id_col, "") if id_col else ""
                            year = row.get(year_col, "")
                            text = row.get(text_col, "")
                            if not text or not year:
                                continue
                            yield JDRow(jid=jid, year=str(year), text=text)
                        except Exception:
                            logger.warning(f"跳过损坏行: {f.name}")
                            continue
            except Exception as e:
                logger.warning(f"读取失败: {f} - {e}")


# =========================
# Bigram Detector (Optional)
# =========================
class BigramDetector:
    def __init__(self, min_count: int, max_pairs: int):
        self.min_count = min_count
        self.max_pairs = max_pairs
        self.counter = Counter()

    def add_doc(self, tokens: List[str]):
        for i in range(len(tokens) - 1):
            pair = tokens[i] + "_" + tokens[i+1]
            self.counter[pair] += 1
        # 定期压缩
        if len(self.counter) > self.max_pairs:
            self._prune()

    def _prune(self):
        # 保留计数最高的一半
        most_common = self.counter.most_common(self.max_pairs // 2)
        self.counter = Counter(dict(most_common))

    def finalize(self) -> set:
        return {p for p, c in self.counter.items() if c >= self.min_count}


# =========================
# Model Trainer
# =========================
class ModelTrainer:
    def __init__(self, stopwords: set):
        self.stopwords = stopwords

    def build_lda_model(self, docs: Iterable[List[str]]) -> tp.LDAModel:
        model = tp.LDAModel(k=K, min_cf=MIN_CF, rm_top=RM_TOP, workers=0)
        for doc in docs:
            if len(doc) >= MIN_TOKENS:
                model.add_doc(doc)
        return model

    def train(self, model: tp.LDAModel):
        logger.info("开始训练 LDA ...")
        with tqdm(total=ITERATIONS, desc="LDA Training") as pbar:
            for i in range(0, ITERATIONS, LOG_EVERY):
                step = min(LOG_EVERY, ITERATIONS - i)
                model.train(step)
                ll = model.ll_per_word
                pbar.update(step)
                pbar.set_postfix({"ll_per_word": f"{ll:.4f}"})
        logger.info("训练完成")


# =========================
# Inference & Metrics
# =========================
class InferenceEngine:
    def __init__(self, model: tp.LDAModel, stopwords: set, bigrams: Optional[set]):
        self.model = model
        self.stopwords = stopwords
        self.bigrams = bigrams

    def infer(self, tokens: List[str]) -> Optional[Tuple[float, float, int, float]]:
        if self.bigrams:
            tokens = apply_bigrams(tokens, self.bigrams)
        if len(tokens) < MIN_TOKENS:
            return None
        doc = self.model.make_doc(tokens)
        theta, _ = self.model.infer(doc)
        # Normalize
        s = sum(theta) + EPS
        probs = [p / s for p in theta]
        # Entropy
        entropy = -sum(p * math.log(p + EPS) for p in probs) / math.log(K)
        # HHI
        hhi = sum(p * p for p in probs)
        # Dominant
        dom_id = int(max(range(len(probs)), key=lambda i: probs[i]))
        dom_prob = float(probs[dom_id])
        return entropy, hhi, dom_id, dom_prob


# =========================
# Main Pipeline
# =========================
def main():
    t0 = time.time()
    streamer = DataStreamer(INPUT_DIR)

    # Stopwords
    stopwords = set(DEFAULT_STOPWORDS)

    # Pass 1: (Optional) Build bigrams
    bigrams = None
    if ENABLE_BIGRAMS:
        logger.info("Pass 1: 构建全局短语(二元组) ...")
        detector = BigramDetector(BIGRAM_MIN_COUNT, BIGRAM_MAX_PAIRS)
        for row in tqdm(streamer.iter_rows(), desc="Bigram Pass"):
            tokens = tokenize(row.text, stopwords)
            if len(tokens) >= MIN_TOKENS:
                detector.add_doc(tokens)
        bigrams = detector.finalize()
        logger.info(f"短语数量: {len(bigrams)}")

    # Pass 2: Build LDA docs
    logger.info("Pass 2: 构建 LDA 文档 ...")
    def doc_generator():
        for row in streamer.iter_rows():
            tokens = tokenize(row.text, stopwords)
            if bigrams:
                tokens = apply_bigrams(tokens, bigrams)
            if len(tokens) >= MIN_TOKENS:
                yield tokens

    trainer = ModelTrainer(stopwords)
    model = trainer.build_lda_model(doc_generator())

    # Train LDA
    trainer.train(model)

    # Save model
    OUTPUT_MODEL.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(OUTPUT_MODEL))
    logger.info(f"模型已保存: {OUTPUT_MODEL}")

    # Pass 3: Inference and metrics
    logger.info("Pass 3: 重新流式推断并计算指标 ...")
    model = tp.LDAModel.load(str(OUTPUT_MODEL))
    infer_engine = InferenceEngine(model, stopwords, bigrams)

    OUTPUT_RESULT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_RESULT.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "year", "entropy_score", "hhi_score", "dominant_topic_id", "dominant_topic_prob"])

        buffer = []
        for row in tqdm(streamer.iter_rows(), desc="Infer Pass"):
            tokens = tokenize(row.text, stopwords)
            metrics = infer_engine.infer(tokens)
            if metrics is None:
                continue
            entropy, hhi, dom_id, dom_prob = metrics
            buffer.append([row.jid, row.year, f"{entropy:.6f}", f"{hhi:.6f}", dom_id, f"{dom_prob:.6f}"])
            if len(buffer) >= BUFFER_SIZE:
                writer.writerows(buffer)
                buffer.clear()
        if buffer:
            writer.writerows(buffer)

    logger.info(f"结果已写入: {OUTPUT_RESULT}")
    logger.info(f"总耗时: {(time.time() - t0)/60:.1f} 分钟")


if __name__ == "__main__":
    main()
