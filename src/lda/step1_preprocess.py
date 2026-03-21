#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Step 1: Preprocess -> JSONL
读取 skill_filtered_corpus.csv（step0产出） -> 清洗 -> jieba 分词 -> 短语识别（可选） -> 输出 JSONL

输入：output/skill_filtered_corpus.csv（has_skill_text=1 的行）
输出：output/processed_corpus.jsonl（每行包含 id, year, tokens）
"""

from __future__ import annotations
from typing import Generator, List
from pathlib import Path
import csv
import json
import logging
import re
import time
from collections import Counter

import jieba
from tqdm import tqdm

# =========================
# 路径配置（相对路径，本地/服务器通用）
# =========================
PROJECT_ROOT = Path(__file__).resolve().parents[2]  # -> TY/

ESCO_DICT_PATH = PROJECT_ROOT / "data/esco/jieba_dict/esco_jieba_dict.txt"
INPUT_CSV = PROJECT_ROOT / "output/skill_filtered_corpus.csv"
OUTPUT_JSONL = PROJECT_ROOT / "output/processed_corpus.jsonl"

# 加载ESCO领域词典（招聘/技能/职业术语）
if ESCO_DICT_PATH.exists():
    jieba.load_userdict(str(ESCO_DICT_PATH))
    print(f"已加载ESCO自定义词典: {ESCO_DICT_PATH}")
else:
    print(f"ESCO词典不存在: {ESCO_DICT_PATH}")


# =========================
# Config
# =========================
MIN_TOKENS = 5

# Bigram (optional)
ENABLE_BIGRAMS = True
BIGRAM_MIN_COUNT = 20
BIGRAM_MAX_PAIRS = 200000

# 输出缓冲
BUFFER_SIZE = 10000

# 英文词黑名单：HTML残留标签 + 英文虚词/停用词
# 策略：保留所有 >= 2字符的英文词，仅过滤以下噪声词
ENGLISH_BLACKLIST = {
    # HTML标签残留（来自原始网页爬取，clean_text去除符号后残留的标签名）
    "br", "div", "span", "li", "ul", "ol", "td", "tr", "th", "img", "href",
    "src", "alt", "font", "style", "color", "margin", "padding", "nbsp",
    "px", "pt", "em", "strong", "http", "https", "www", "com", "html", "css",
    # 英文虚词/停用词（对主题建模无区分度）
    "and", "the", "in", "to", "of", "or", "with", "for", "is", "are",
    "be", "an", "at", "by", "on", "as", "it", "we", "do", "if", "no",
    "not", "but", "all", "can", "has", "was", "been", "will", "from",
    "that", "this", "have", "they", "had", "its", "our", "you", "your",
}
# 单字母中仅保留有意义的（c语言、R语言）
ENGLISH_SINGLE_ALLOW = {"c", "r"}
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
logger = logging.getLogger("Step1-Preprocess")


# =========================
# Utilities
# =========================
def clean_text(text: str) -> str:
    if not text:
        return ""
    t = text.lower()
    for k, v in REPLACEMENTS.items():
        t = t.replace(k, v)
    # 只保留中文与英文字母，去除数字与符号
    t = re.sub(r"[^a-z\u4e00-\u9fa5]+", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def is_chinese_token(token: str) -> bool:
    return bool(re.search(r"[\u4e00-\u9fa5]", token))


def tokenize(text: str, stopwords: set) -> List[str]:
    t = clean_text(text)
    if not t:
        return []
    words = jieba.lcut(t)
    tokens: List[str] = []
    for w in words:
        if w in stopwords:
            continue
        if w.isdigit():
            continue
        if is_chinese_token(w):
            if len(w) >= 2:
                tokens.append(w)
        else:
            # 英文词：黑名单过滤
            if w in ENGLISH_BLACKLIST:
                continue
            if len(w) == 1:
                # 单字母仅保留 c, r（编程语言）
                if w in ENGLISH_SINGLE_ALLOW:
                    tokens.append(w)
            else:
                # >= 2字符的英文词全部保留
                tokens.append(w)
    return tokens


def apply_bigrams(tokens: List[str], bigrams: set) -> List[str]:
    if not tokens:
        return tokens
    merged: List[str] = []
    i = 0
    while i < len(tokens) - 1:
        pair = tokens[i] + "_" + tokens[i + 1]
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
# Streaming: 读取 skill_filtered_corpus.csv
# =========================
class JDRow:
    __slots__ = ("jid", "year", "text")
    def __init__(self, jid: str, year: str, text: str):
        self.jid = jid
        self.year = year
        self.text = text


def iter_rows(csv_path: Path) -> Generator[JDRow, None, None]:
    """流式读取 skill_filtered_corpus.csv，只返回 has_skill_text=1 的行"""
    with csv_path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            if row.get("has_skill_text") != "1":
                continue
            text = row.get("skill_text", "")
            if not text:
                continue
            yield JDRow(
                jid=row.get("job_id", ""),
                year=row.get("year", ""),
                text=text,
            )


# =========================
# Bigram Detector
# =========================
class BigramDetector:
    def __init__(self, min_count: int, max_pairs: int):
        self.min_count = min_count
        self.max_pairs = max_pairs
        self.counter = Counter()

    def add_doc(self, tokens: List[str]) -> None:
        for i in range(len(tokens) - 1):
            pair = tokens[i] + "_" + tokens[i + 1]
            self.counter[pair] += 1
        if len(self.counter) > self.max_pairs:
            self._prune()

    def _prune(self) -> None:
        most_common = self.counter.most_common(self.max_pairs // 2)
        self.counter = Counter(dict(most_common))

    def finalize(self) -> set:
        return {p for p, c in self.counter.items() if c >= self.min_count}


# =========================
# Main
# =========================
def main() -> None:
    t0 = time.time()
    stopwords = set(DEFAULT_STOPWORDS)

    if not INPUT_CSV.exists():
        logger.error(f"找不到输入文件: {INPUT_CSV}")
        return

    bigrams = None
    if ENABLE_BIGRAMS:
        logger.info("Pass 1: 构建短语（二元组）...")
        detector = BigramDetector(BIGRAM_MIN_COUNT, BIGRAM_MAX_PAIRS)
        for row in tqdm(iter_rows(INPUT_CSV), desc="Bigram Pass"):
            tokens = tokenize(row.text, stopwords)
            if len(tokens) >= MIN_TOKENS:
                detector.add_doc(tokens)
        bigrams = detector.finalize()
        logger.info(f"短语数量: {len(bigrams)}")

    logger.info("Pass 2: 输出 JSONL ...")
    OUTPUT_JSONL.parent.mkdir(parents=True, exist_ok=True)
    total_written = 0
    with OUTPUT_JSONL.open("w", encoding="utf-8") as f:
        buffer: List[str] = []
        for row in tqdm(iter_rows(INPUT_CSV), desc="Preprocess Pass"):
            tokens = tokenize(row.text, stopwords)
            if bigrams:
                tokens = apply_bigrams(tokens, bigrams)
            if len(tokens) < MIN_TOKENS:
                continue
            obj = {"id": row.jid, "year": row.year, "tokens": tokens}
            buffer.append(json.dumps(obj, ensure_ascii=False))
            total_written += 1
            if len(buffer) >= BUFFER_SIZE:
                f.write("\n".join(buffer) + "\n")
                buffer.clear()
        if buffer:
            f.write("\n".join(buffer) + "\n")

    logger.info(f"输出完成: {OUTPUT_JSONL}")
    logger.info(f"有效文档数: {total_written:,}")
    logger.info(f"总耗时: {(time.time() - t0) / 60:.1f} 分钟")


if __name__ == "__main__":
    main()
