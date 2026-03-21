#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
统计原始语料中所有英文词的词频（采样10%以加速）。
输出：英文词频表，用于决定黑名单/白名单策略。
"""

import csv
import re
import random
from collections import Counter
from pathlib import Path

import jieba

# ── 路径 ──
WINDOWS_DIR = Path("/Users/yu/code/code2601/TY/data/processed/windows")
OUTPUT_CSV = Path("/Users/yu/code/code2601/TY/output/english_token_freq.csv")

# ── 与 step1_preprocess.py 一致的清洗逻辑 ──
REPLACEMENTS = {"c++": "cpp", "c#": "csharp", ".net": "dotnet"}

def clean_text(text: str) -> str:
    if not text:
        return ""
    t = text.lower()
    for k, v in REPLACEMENTS.items():
        t = t.replace(k, v)
    t = re.sub(r"[^a-z\u4e00-\u9fa5]+", " ", t)
    return re.sub(r"\s+", " ", t).strip()

def is_english(w: str) -> bool:
    return bool(re.fullmatch(r"[a-z]+", w))

# ── Main ──
SAMPLE_RATE = 0.1
SEED = 42

def main():
    rng = random.Random(SEED)
    counter = Counter()
    total_docs = 0
    sampled = 0

    pattern = re.compile(r"^window_\d{4}_\d{4}\.csv$")
    files = sorted(f for f in WINDOWS_DIR.glob("*.csv") if pattern.match(f.name))
    print(f"Found {len(files)} window files")

    for fp in files:
        print(f"  Scanning {fp.name}...")
        with fp.open("r", encoding="utf-8-sig", errors="replace", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                total_docs += 1
                if rng.random() > SAMPLE_RATE:
                    continue
                sampled += 1

                text = row.get("cleaned_requirements", "")
                if not text:
                    continue

                cleaned = clean_text(text)
                words = jieba.lcut(cleaned)

                for w in words:
                    if is_english(w) and not w.isdigit():
                        counter[w] += 1

                if sampled % 200_000 == 0:
                    print(f"    sampled={sampled:,}, unique_en={len(counter):,}")

    print(f"\nDone: total={total_docs:,}, sampled={sampled:,}")
    print(f"Unique English tokens: {len(counter):,}")

    # 输出
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_CSV.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["token", "count"])
        for tok, cnt in counter.most_common():
            writer.writerow([tok, cnt])

    print(f"Saved to {OUTPUT_CSV}")

    # 打印 top 100
    print("\n── Top 100 English tokens ──")
    for tok, cnt in counter.most_common(100):
        print(f"  {tok:30s} {cnt:>10,}")

    # 统计长度分布
    print("\n── 按长度分布 ──")
    len_dist = Counter()
    for tok, cnt in counter.items():
        len_dist[len(tok)] += cnt
    for l in sorted(len_dist):
        print(f"  len={l}: {len_dist[l]:>12,} occurrences")


if __name__ == "__main__":
    main()
