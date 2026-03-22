#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Step 4: Build Job-Level Panel for DID Regression

ISCO 匹配策略（4级，按优先级）：
  1. raw:      原始标题精确匹配 sbert_isco_lookup.csv
  2. cleaned:  清洗标题匹配 sbert_isco_lookup.csv
  3. alias:    人工高频标题 alias 表 (title_isco_alias.csv)
  4. cluster:  title → cluster → dominant ISCO (纯度 ≥ 阈值 + 最小支撑量)

主规格用 --cluster-purity 0.4（默认），稳健性用 0.3。

用法：
  python step4_build_panel.py                         # 默认 K=100, purity>=0.4
  python step4_build_panel.py --cluster-purity 0.3    # 稳健性规格
  python step4_build_panel.py --cluster-purity 0      # 无 cluster fallback

输出列包含 match_method / fallback_cluster_id / fallback_top1_share。
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import pickle
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
ALIAS_CSV = PROJECT_ROOT / "data/Heterogeneity/title_isco_alias.csv"
ILO_CSV = PROJECT_ROOT / "data/esco/ilo_genai_isco08_2025.csv"
TITLE2CLUSTER_PKL = PROJECT_ROOT / "output/clusters/title2cluster_full.pkl"
CLUSTER_EXPOSURE_CSV = PROJECT_ROOT / "output/clusters/ai_exposure_ilo.csv"

NOISE_CLUSTERS = {5, 6, 9, 43, 51, 74, -1}
MIN_CLUSTER_MATCHED = 100  # 最小支撑量：cluster 内已匹配样本 >= 100

# =========================
# 行业分类
# =========================
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from classify_industry import classify as _classify_raw

def classify_industry(company_name: str) -> str:
    code, _ = _classify_raw(company_name)
    return code if code else ""


# =========================
# 岗位标题清洗（与 build_isco_panel.py / cluster_occupations.py 一致）
# =========================
def clean_title(t: str) -> str:
    t = re.sub(r'[（(][^)）]*[)）]', '', t)
    t = re.sub(r'\d{3,}元?/?', '', t)
    t = re.sub(r'[+➕].{0,20}$', '', t)
    t = re.sub(r'/[包入无].{0,15}$', '', t)
    t = re.sub(r'(急聘|诚聘|高薪|急招|高新|直招|包吃住|包食宿|五险一金|双休|转正)', '', t)
    t = t.strip(' /\\-—_·.，,')
    return t


def window_tag(filename: str) -> str:
    m = re.search(r"window_(\d{4})_\d{4}", filename)
    return f"w{m.group(1)}" if m else "wunk"


# =========================
# 流式辅助迭代器
# =========================
class SkillFilteredIter:
    def __init__(self, path: Path):
        self._fh = open(path, 'r', encoding='utf-8')
        self._reader = csv.DictReader(self._fh)
        self._current = None
        self._advance()

    def _advance(self):
        try:
            self._current = next(self._reader)
        except StopIteration:
            self._current = None

    def get(self, job_id: str):
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
        if self._current_id and self._current_id == job_id:
            count = self._current_count
            self._advance()
            return count
        return None

    def close(self):
        if self._fh:
            self._fh.close()


class TopicResultIter:
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
    parser.add_argument("--cluster-purity", type=float, default=0.4,
                        help="Cluster fallback 最低纯度阈值 (default: 0.4, robustness: 0.3)")
    args = parser.parse_args()
    K = args.k
    PURITY_THRESHOLD = args.cluster_purity

    TOPIC_RESULTS_CSV = PROJECT_ROOT / f"output/topic_results_k{K}.csv"
    OUTPUT_CSV = PROJECT_ROOT / f"output/job_panel_k{K}.csv"

    t0 = time.time()

    # ── 1. Load ILO AI exposure scores ──
    print("[1/6] Loading ILO GenAI exposure scores...")
    ilo_scores = {}
    with open(ILO_CSV, 'r', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            isco = row['isco08_4digit'].strip()
            ilo_scores[isco] = float(row['mean_score'])
    print(f"  {len(ilo_scores)} ISCO codes")

    # ── 2. Load SBERT ISCO lookup (Level 1 + 2) ──
    print("[2/6] Loading SBERT ISCO lookup...")
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

    # ── 3. Load alias table (Level 3) ──
    print("[3/6] Loading title alias table...")
    alias_isco = {}
    if ALIAS_CSV.exists():
        with open(ALIAS_CSV, 'r', encoding='utf-8') as f:
            for row in csv.DictReader(f):
                title = row['title'].strip()
                isco = row['isco08_4digit'].strip()
                if isco in ilo_scores:
                    alias_isco[title] = isco
        print(f"  {len(alias_isco)} alias entries (filtered to ILO-available)")
    else:
        print(f"  WARNING: {ALIAS_CSV} not found. Alias layer disabled.")

    # ── 4. Load cluster fallback (Level 4) ──
    print(f"[4/6] Loading cluster fallback (purity >= {PURITY_THRESHOLD}, min_support >= {MIN_CLUSTER_MATCHED})...")
    title2cluster = {}
    cluster_isco = {}    # cid -> isco
    cluster_purity = {}  # cid -> float

    if TITLE2CLUSTER_PKL.exists() and CLUSTER_EXPOSURE_CSV.exists():
        with open(TITLE2CLUSTER_PKL, 'rb') as f:
            title2cluster = pickle.load(f)
        print(f"  title2cluster: {len(title2cluster):,} unique titles")

        with open(CLUSTER_EXPOSURE_CSV, 'r', encoding='utf-8-sig') as f:
            for row in csv.DictReader(f):
                cid = int(row['cluster_id'])
                if cid in NOISE_CLUSTERS:
                    continue
                top_str = row.get('top_isco_codes', '')
                n_matched = int(row.get('n_matched_rows', 0))
                entries = re.findall(r'(\d{4})\(([\d.]+),\s*n=([\d,]+)\)', top_str)
                if not entries or n_matched < MIN_CLUSTER_MATCHED:
                    continue
                top1_isco = entries[0][0]
                top1_n = int(entries[0][2].replace(',', ''))
                pur = top1_n / n_matched
                if pur >= PURITY_THRESHOLD and top1_isco in ilo_scores:
                    cluster_isco[cid] = top1_isco
                    cluster_purity[cid] = pur

        print(f"  {len(cluster_isco)} clusters pass threshold (purity >= {PURITY_THRESHOLD}, support >= {MIN_CLUSTER_MATCHED})")
    else:
        print(f"  WARNING: Cluster files not found. Cluster fallback disabled.")

    # ── 5. Open streaming iterators ──
    print("[5/6] Opening streaming iterators...")
    skill_iter = SkillFilteredIter(SKILL_FILTERED_CSV)
    corpus_iter = JsonlTokenIter(PROCESSED_JSONL)
    if not TOPIC_RESULTS_CSV.exists():
        print(f"  WARNING: {TOPIC_RESULTS_CSV} not found. Topic columns will be empty.")
    topic_iter = TopicResultIter(TOPIC_RESULTS_CSV)

    # ── 6. Process window CSVs ──
    print("[6/6] Processing window CSVs (streaming merge join)...")

    csv_files = sorted(glob.glob(str(WINDOWS_DIR / "window_*.csv")))
    csv_files = [f for f in csv_files if "stats" not in f]
    print(f"  Found {len(csv_files)} window files")

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    OUTPUT_HEADER = [
        "job_id", "year", "job_title", "isco08_4digit", "industry_code", "platform", "city",
        "ai_exposure", "post",
        "has_skill_text", "original_char_count", "skill_char_count", "skill_token_count",
        "entropy_score", "hhi_score", "dominant_topic_id", "dominant_topic_prob",
        "ent_effective", "rao_q", "gini", "n_sig_topics", "tail_mass_ratio",
        "salary_min", "salary_max", "edu_level", "exp_years",
        "match_method", "fallback_cluster_id", "fallback_top1_share",
    ]

    total_written = 0
    match_stats = defaultdict(int)
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

                    job_title = row.get("招聘岗位", "").strip()
                    company = row.get("企业名称", "").strip()
                    city = row.get("工作城市", "").strip()
                    salary_min = row.get("最低月薪", "").strip()
                    salary_max = row.get("最高月薪", "").strip()
                    edu_level = row.get("学历要求", "").strip()
                    exp_years = row.get("要求经验", "").strip()
                    platform = row.get("来源平台", "").strip()

                    ind_code = classify_industry(company)

                    # --- 4级 ISCO 匹配 ---
                    isco = ""
                    match_method = "miss"
                    fb_cid = ""
                    fb_share = ""

                    # Level 1: raw
                    if job_title in raw_isco:
                        isco = raw_isco[job_title]
                        match_method = "raw"
                    else:
                        ct = clean_title(job_title)
                        # Level 2: cleaned
                        if ct and ct in cleaned_isco:
                            isco = cleaned_isco[ct]
                            match_method = "cleaned"
                        # Level 3: alias
                        elif job_title in alias_isco:
                            isco = alias_isco[job_title]
                            match_method = "alias"
                        elif ct and ct in alias_isco:
                            isco = alias_isco[ct]
                            match_method = "alias"
                        # Level 4: cluster fallback
                        elif ct and ct in title2cluster:
                            cid = title2cluster[ct]
                            if cid in cluster_isco:
                                isco = cluster_isco[cid]
                                match_method = "cluster"
                                fb_cid = str(cid)
                                fb_share = f"{cluster_purity[cid]:.3f}"

                    match_stats[match_method] += 1

                    ai_exposure = ""
                    if isco and isco in ilo_scores:
                        ai_exposure = f"{ilo_scores[isco]:.4f}"

                    # --- 流式 merge ---
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

                    tc = corpus_iter.get(job_id)
                    stc = str(tc) if tc is not None else ""
                    if tc is not None:
                        corpus_joined += 1

                    td = topic_iter.get(job_id)
                    if td is not None:
                        topic_joined += 1

                    post = ""
                    if year:
                        try:
                            post = "1" if int(year) >= 2022 else "0"
                        except ValueError:
                            pass

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
                        match_method, fb_cid, fb_share,
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

    skill_iter.close()
    corpus_iter.close()
    topic_iter.close()

    # ── 报告 ──
    elapsed = (time.time() - t0) / 60
    total_matched = sum(v for k, v in match_stats.items() if k != "miss")
    total_all = sum(match_stats.values())

    print(f"\n{'='*60}")
    print(f"Panel built: {total_written:,} rows -> {OUTPUT_CSV}")
    print(f"Time: {elapsed:.1f} min")
    print(f"Cluster purity threshold: {PURITY_THRESHOLD}")

    print(f"\nISCO matching (4-level):")
    for method in ["raw", "cleaned", "alias", "cluster", "miss"]:
        n = match_stats.get(method, 0)
        print(f"  {method:10s}: {n:>10,} ({n/total_all*100:5.1f}%)")
    print(f"  {'TOTAL':10s}: {total_matched:>10,} ({total_matched/total_all*100:5.1f}%)")

    print(f"\nStreaming merge join:")
    print(f"  skill_filtered: {skill_joined:,} (mismatch: {skill_mismatch:,})")
    print(f"  token_counts:   {corpus_joined:,}")
    print(f"  topic_results:  {topic_joined:,}")


if __name__ == "__main__":
    main()
