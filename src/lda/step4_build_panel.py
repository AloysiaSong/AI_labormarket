#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Step 4: Build Job-Level Panel for DID Regression

从 skill_filtered_corpus.csv（全量骨架）出发，
合并窗口CSV元数据、ISCO匹配、AI暴露度、行业分类、主题指标，
输出 job_panel_k{K}.csv。

内存策略：skill_filtered_corpus.csv / processed_corpus.jsonl / topic_results
三个辅助文件均与窗口CSV同序（step0 → step1 → step3 保序），
使用流式 sorted merge join，不加载到内存。

行业分类：直接 import classify_industry.classify()，
与 company_industry_map.csv 同源同规则，无口径分叉。

用法：
  python step4_build_panel.py          # 默认 K=100
  python step4_build_panel.py --k 50   # 指定 K

输入：
  - output/skill_filtered_corpus.csv           (全量骨架，860万行)
  - data/processed/windows/window_*.csv        (元数据)
  - output/processed_corpus.jsonl              (token counts)
  - data/Heterogeneity/sbert_isco_lookup.csv   (ISCO映射)
  - data/esco/ilo_genai_isco08_2025.csv        (AI暴露度)
  - output/topic_results_k{K}.csv              (主题指标，左连接)

输出：
  - output/job_panel_k{K}.csv
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import re
import sys
import time
from pathlib import Path
from collections import defaultdict

from tqdm import tqdm

# =========================
# 路径配置
# =========================
PROJECT_ROOT = Path(__file__).resolve().parents[2]  # -> TY/

WINDOWS_DIR = PROJECT_ROOT / "data/processed/windows"
SKILL_FILTERED_CSV = PROJECT_ROOT / "output/skill_filtered_corpus.csv"
PROCESSED_JSONL = PROJECT_ROOT / "output/processed_corpus.jsonl"
SBERT_LOOKUP_CSV = PROJECT_ROOT / "data/Heterogeneity/sbert_isco_lookup.csv"
ILO_CSV = PROJECT_ROOT / "data/esco/ilo_genai_isco08_2025.csv"

# =========================
# 行业分类：import classify_industry.classify()
# 与 company_industry_map.csv 同源同规则
# =========================
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # add src/ to path
from classify_industry import classify as _classify_raw

def classify_industry(company_name: str):
    """返回 (industry_code,) 或 ("",)，封装 classify_industry.classify()"""
    code, _ = _classify_raw(company_name)
    return code if code else ""


# =========================
# 岗位标题清洗（用于ISCO匹配fallback，与build_isco_panel.py一致）
# =========================
def clean_title(t: str) -> str:
    t = re.sub(r'[（(][^)）]*[)）]', '', t)
    t = re.sub(r'\d{3,}元?/?', '', t)
    t = re.sub(r'[+➕].{0,20}$', '', t)
    t = re.sub(r'/[包入无].{0,15}$', '', t)
    t = re.sub(r'(急聘|诚聘|高薪|急招|高新|直招|包吃住|包食宿|五险一金|双休|转正)', '', t)
    t = t.strip(' /\\-—_·.，,')
    return t


# =========================
# 窗口标签（与step0一致）
# =========================
def window_tag(filename: str) -> str:
    m = re.search(r"window_(\d{4})_\d{4}", filename)
    return f"w{m.group(1)}" if m else "wunk"


# =========================
# 流式辅助迭代器（sorted merge join 用）
# =========================
class SkillFilteredIter:
    """流式读取 skill_filtered_corpus.csv，按 job_id 推进"""
    def __init__(self, path: Path):
        self._fh = open(path, 'r', encoding='utf-8')
        self._reader = csv.DictReader(self._fh)
        self._current = None
        self._advance()

    def _advance(self):
        try:
            row = next(self._reader)
            self._current = row
        except StopIteration:
            self._current = None

    def get(self, job_id: str):
        """如果当前行匹配 job_id，返回数据并推进；否则返回 None"""
        if self._current and self._current.get("job_id") == job_id:
            data = {
                "year": self._current.get("year", ""),
                "has_skill_text": self._current.get("has_skill_text", ""),
                "skill_char_count": self._current.get("skill_char_count", ""),
                "original_char_count": self._current.get("original_char_count", ""),
            }
            self._advance()
            return data
        return None

    def close(self):
        self._fh.close()


class JsonlTokenIter:
    """流式读取 processed_corpus.jsonl，按 id 推进"""
    def __init__(self, path: Path):
        if path.exists():
            self._fh = open(path, 'r', encoding='utf-8')
            self._current_id = None
            self._current_count = 0
            self._advance()
        else:
            self._fh = None
            self._current_id = None

    def _advance(self):
        while True:
            line = self._fh.readline()
            if not line:
                self._current_id = None
                return
            try:
                obj = json.loads(line)
                self._current_id = obj.get("id", "")
                self._current_count = len(obj.get("tokens", []))
                return
            except Exception:
                continue

    def get(self, job_id: str):
        """如果当前行匹配 job_id，返回 token count 并推进；否则返回 None"""
        if self._current_id and self._current_id == job_id:
            count = self._current_count
            self._advance()
            return count
        return None

    def close(self):
        if self._fh:
            self._fh.close()


class TopicResultIter:
    """流式读取 topic_results_k{K}.csv，按 id 推进"""
    COLS = [
        "entropy_score", "hhi_score", "dominant_topic_id", "dominant_topic_prob",
        "ent_effective", "rao_q", "gini", "n_sig_topics", "tail_mass_ratio"
    ]

    def __init__(self, path: Path):
        if path.exists():
            self._fh = open(path, 'r', encoding='utf-8-sig')
            self._reader = csv.DictReader(self._fh)
            self._current = None
            self._advance()
        else:
            self._fh = None
            self._current = None

    def _advance(self):
        try:
            self._current = next(self._reader)
        except StopIteration:
            self._current = None

    def get(self, job_id: str):
        """如果当前行匹配 job_id，返回指标 dict 并推进；否则返回 None"""
        if self._current and self._current.get("id") == job_id:
            data = {col: self._current.get(col, "") for col in self.COLS}
            self._advance()
            return data
        return None

    def close(self):
        if self._fh:
            self._fh.close()


# =========================
# Main
# =========================
def main():
    parser = argparse.ArgumentParser(description="Build Job-Level Panel")
    parser.add_argument("--k", type=int, default=100, help="主题数K (default: 100)")
    args = parser.parse_args()
    K = args.k

    TOPIC_RESULTS_CSV = PROJECT_ROOT / f"output/topic_results_k{K}.csv"
    OUTPUT_CSV = PROJECT_ROOT / f"output/job_panel_k{K}.csv"

    t0 = time.time()

    # ── 1. Load ILO AI exposure scores (tiny, ~400 entries) ──
    print("[1/4] Loading ILO GenAI exposure scores...")
    ilo_scores = {}
    with open(ILO_CSV, 'r', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            isco = row['isco08_4digit'].strip()
            ilo_scores[isco] = float(row['mean_score'])
    print(f"  {len(ilo_scores)} ISCO codes with AI exposure scores")

    # ── 2. Load SBERT ISCO lookup (raw + cleaned fallback, ~1M entries) ──
    print("[2/4] Loading SBERT ISCO lookup...")
    raw_isco = {}
    cleaned_isco = {}
    with open(SBERT_LOOKUP_CSV, 'r', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            if row.get('sbert_accepted', 'True') != 'True':
                continue
            isco = row['isco08_4digit'].strip()
            if isco not in ilo_scores:
                continue
            title = row['title'].strip()
            raw_isco[title] = isco
            ct = clean_title(title)
            if ct and len(ct) >= 2 and ct not in cleaned_isco:
                cleaned_isco[ct] = isco
    print(f"  Raw: {len(raw_isco):,}, Cleaned: {len(cleaned_isco):,}")

    # ── 3. Open streaming iterators for auxiliary files ──
    print("[3/4] Opening streaming iterators...")
    skill_iter = SkillFilteredIter(SKILL_FILTERED_CSV)
    corpus_iter = JsonlTokenIter(PROCESSED_JSONL)

    if not TOPIC_RESULTS_CSV.exists():
        print(f"  WARNING: {TOPIC_RESULTS_CSV} not found. Topic columns will be empty.")
        print(f"  请先从服务器下载 topic_results_k{K}.csv 到 output/ 目录")
    topic_iter = TopicResultIter(TOPIC_RESULTS_CSV)

    # ── 4. Process window CSVs: streaming merge join ──
    print("[4/4] Processing window CSVs (streaming merge join)...")

    csv_files = sorted(glob.glob(str(WINDOWS_DIR / "window_*.csv")))
    csv_files = [f for f in csv_files if "stats" not in f]
    print(f"  Found {len(csv_files)} window files")

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    OUTPUT_HEADER = [
        # 标识与分组
        "job_id", "year", "job_title", "isco08_4digit", "industry_code", "platform", "city",
        # 处理变量
        "ai_exposure", "post",
        # 样本结构变量
        "has_skill_text", "original_char_count", "skill_char_count", "skill_token_count",
        # 主题指标
        "entropy_score", "hhi_score", "dominant_topic_id", "dominant_topic_prob",
        "ent_effective", "rao_q", "gini", "n_sig_topics", "tail_mass_ratio",
        # 控制变量
        "salary_min", "salary_max", "edu_level", "exp_years",
    ]

    total_written = 0
    isco_match_stats = defaultdict(int)
    skill_joined = 0
    corpus_joined = 0
    topic_joined = 0
    skill_mismatch = 0

    with open(OUTPUT_CSV, 'w', encoding='utf-8-sig', newline='') as fout:
        writer = csv.writer(fout)
        writer.writerow(OUTPUT_HEADER)

        for csv_path in csv_files:
            fname = os.path.basename(csv_path)
            tag = window_tag(fname)
            print(f"\n  Processing: {fname} (tag={tag})")

            local_idx = 0
            buf = []
            FLUSH_SIZE = 50000

            with open(csv_path, 'r', encoding='utf-8', newline='') as fin:
                reader = csv.DictReader(fin)
                for row in tqdm(reader, desc=f"  {tag}"):
                    job_id = f"{tag}_{local_idx}"
                    local_idx += 1

                    # --- 元数据（来自窗口CSV）---
                    job_title = row.get("招聘岗位", "").strip()
                    company = row.get("企业名称", "").strip()
                    city = row.get("工作城市", "").strip()
                    salary_min = row.get("最低月薪", "").strip()
                    salary_max = row.get("最高月薪", "").strip()
                    edu_level = row.get("学历要求", "").strip()
                    exp_years = row.get("要求经验", "").strip()
                    platform = row.get("来源平台", "").strip()

                    # --- 行业分类（与 company_industry_map.csv 同源） ---
                    ind_code = classify_industry(company)

                    # --- ISCO匹配（2级fallback）---
                    isco = ""
                    if job_title in raw_isco:
                        isco = raw_isco[job_title]
                        isco_match_stats["raw"] += 1
                    else:
                        ct = clean_title(job_title)
                        if ct and ct in cleaned_isco:
                            isco = cleaned_isco[ct]
                            isco_match_stats["cleaned"] += 1
                        else:
                            isco_match_stats["miss"] += 1

                    # --- AI暴露度 ---
                    ai_exposure = ""
                    if isco and isco in ilo_scores:
                        ai_exposure = f"{ilo_scores[isco]:.4f}"

                    # --- 流式 merge: skill_filtered_corpus ---
                    sk = skill_iter.get(job_id)
                    if sk:
                        skill_joined += 1
                        year = sk["year"]
                        has_skill = sk["has_skill_text"]
                        skill_cc = sk["skill_char_count"]
                        orig_cc = sk["original_char_count"]
                    else:
                        skill_mismatch += 1
                        year = ""
                        has_skill = ""
                        skill_cc = ""
                        orig_cc = ""

                    # --- 流式 merge: processed_corpus.jsonl (token count) ---
                    tc = corpus_iter.get(job_id)
                    stc = str(tc) if tc is not None else ""
                    if tc is not None:
                        corpus_joined += 1

                    # --- 流式 merge: topic_results ---
                    td = topic_iter.get(job_id)
                    if td is not None:
                        topic_joined += 1

                    # --- post变量 ---
                    post = ""
                    if year:
                        try:
                            post = "1" if int(year) >= 2022 else "0"
                        except ValueError:
                            pass

                    # --- 组装行 ---
                    _td = td or {}
                    out_row = [
                        job_id, year, job_title, isco, ind_code, platform, city,
                        ai_exposure, post,
                        has_skill, orig_cc, skill_cc, stc,
                        _td.get("entropy_score", ""),
                        _td.get("hhi_score", ""),
                        _td.get("dominant_topic_id", ""),
                        _td.get("dominant_topic_prob", ""),
                        _td.get("ent_effective", ""),
                        _td.get("rao_q", ""),
                        _td.get("gini", ""),
                        _td.get("n_sig_topics", ""),
                        _td.get("tail_mass_ratio", ""),
                        salary_min, salary_max, edu_level, exp_years,
                    ]
                    buf.append(out_row)

                    if len(buf) >= FLUSH_SIZE:
                        writer.writerows(buf)
                        fout.flush()
                        total_written += len(buf)
                        buf.clear()

            if buf:
                writer.writerows(buf)
                fout.flush()
                total_written += len(buf)
                buf.clear()

            print(f"    {local_idx:,} rows processed")

    # 关闭流式迭代器
    skill_iter.close()
    corpus_iter.close()
    topic_iter.close()

    # ── 报告 ──
    elapsed = (time.time() - t0) / 60
    print(f"\n{'='*60}")
    print(f"Panel built: {total_written:,} rows -> {OUTPUT_CSV}")
    print(f"Time: {elapsed:.1f} min")

    print(f"\nISCO matching:")
    print(f"  Raw match:     {isco_match_stats['raw']:,}")
    print(f"  Cleaned match: {isco_match_stats['cleaned']:,}")
    print(f"  Miss:          {isco_match_stats['miss']:,}")
    total_isco = isco_match_stats['raw'] + isco_match_stats['cleaned']
    total_all = total_isco + isco_match_stats['miss']
    if total_all > 0:
        print(f"  Match rate:    {total_isco/total_all*100:.1f}%")

    print(f"\nStreaming merge join stats:")
    print(f"  skill_filtered joined: {skill_joined:,} (mismatch: {skill_mismatch:,})")
    print(f"  token_counts joined:   {corpus_joined:,}")
    print(f"  topic_results joined:  {topic_joined:,}")
    print(f"\n  skill_token_count coverage note:")
    print(f"    has_skill_text=1 的岗位中，仅 processed_corpus.jsonl 中的子集")
    print(f"    (step1 tokenization ≥ {5} tokens) 有 skill_token_count 值。")
    print(f"    intensive margin 样本要求 topic outcome 非空，该子集总有 token count。")


if __name__ == "__main__":
    main()
