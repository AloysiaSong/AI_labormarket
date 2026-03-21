#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Industry Heterogeneity Analysis for TY project.

Pipeline:
1) Build company -> industry lookup from cleaned metadata.
2) Reconstruct id-aligned metadata by replaying step1 tokenization + bigram merge.
3) Merge reconstructed metadata with final_results_sample.csv.
4) Build yearly industry-level metrics and 2016 vs 2024 growth table/plots.

Notes:
- `all_in_one2_dedup.csv` has no industry column.
- This script uses company-name lookup as primary source, with company-name keyword
  fallback when lookup is missing.
"""

from __future__ import annotations

import csv
import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

import pandas as pd


# =========================
# Paths
# =========================
PROJECT_ROOT = Path("/Users/yu/code/code2601/TY")
WINDOWS_DIR = PROJECT_ROOT / "data/processed/windows"
CLEANED_METADATA = PROJECT_ROOT / "data/processed/cleaned/all_in_one1.csv"
PROCESSED_CORPUS = PROJECT_ROOT / "output/processed_corpus.jsonl"
FINAL_RESULTS = PROJECT_ROOT / "output/final_results_sample.csv"

HET_DIR = PROJECT_ROOT / "data/Heterogeneity"
OUT_DIR = PROJECT_ROOT / "output/heterogeneity"

COMPANY_LOOKUP_CSV = HET_DIR / "company_industry_lookup.csv"
RECONSTRUCTED_META_CSV = HET_DIR / "reconstructed_id_metadata.csv"
RECONSTRUCTED_META_SORTED_CSV = HET_DIR / "reconstructed_id_metadata_sorted.csv"
FINAL_RESULTS_SORTED_CSV = HET_DIR / "final_results_sample_sorted.csv"
MASTER_CSV = HET_DIR / "master_industry_analysis.csv"
YEARLY_METRICS_CSV = HET_DIR / "yearly_industry_metrics.csv"
YEARLY_METRICS_VALID_CSV = HET_DIR / "yearly_industry_metrics_valid500.csv"
GROWTH_2016_2024_CSV = HET_DIR / "industry_growth_2016_2024.csv"
DIAGNOSTICS_CSV = HET_DIR / "reconstruction_diagnostics.csv"

TREND_FIG = OUT_DIR / "industry_entropy_trend_top12.png"
SCATTER_FIG = OUT_DIR / "industry_entropy_growth_scatter_2016_2024.png"


# =========================
# Tokenization (aligned with step1_preprocess)
# =========================
REPLACEMENTS = {
    "c++": "cpp",
    "c#": "csharp",
    ".net": "dotnet",
}

ENGLISH_ALLOW = {"python", "java", "sql", "c", "r", "go", "cpp", "csharp", "dotnet"}

DEFAULT_STOPWORDS = set([
    "的","了","在","是","有","和","与","或","等","能","会","及","对","可","为","被","把","让","给","向","从","到","以","于",
    "个","这","那","一","不","也","要","就","都","而","但","如","所","他","她","它","们","我","你","上","下","中","内","外",
    "职责","任职","要求","优先","具备","良好","负责","具有","熟悉","掌握","了解","相关","以上","以下","经验","工作","能力",
    "岗位","公司","企业","团队","一定","优秀","专业","学历","年","及以上","左右","不限","若干","可以","需要","进行","开展",
    "完成","参与","支持","提供","协助","配合","执行","根据","按照","通过","利用","使用","包括","主要","其他","相应","有效","合理","准确","及时",
])

MIN_TOKENS = 5


def clean_text(text: str) -> str:
    if not text:
        return ""
    t = str(text).lower()
    for src, dst in REPLACEMENTS.items():
        t = t.replace(src, dst)
    t = re.sub(r"[^a-z\u4e00-\u9fa5]+", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def is_chinese_token(token: str) -> bool:
    return bool(re.search(r"[\u4e00-\u9fa5]", token))


def tokenize(text: str, stopwords: Set[str]) -> List[str]:
    import jieba

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
            if w in ENGLISH_ALLOW:
                tokens.append(w)
    return tokens


def apply_bigrams(tokens: List[str], bigrams: Set[str]) -> List[str]:
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
# Industry mapping
# =========================
STD_INDUSTRY = {
    "A": "A 农、林、牧、渔业",
    "B": "B 采矿业",
    "C": "C 制造业",
    "D": "D 电力、热力、燃气及水生产和供应业",
    "E": "E 建筑业",
    "F": "F 批发和零售业",
    "G": "G 交通运输、仓储和邮政业",
    "H": "H 住宿和餐饮业",
    "I": "I 信息传输、软件和信息技术服务业",
    "J": "J 金融业",
    "K": "K 房地产业",
    "L": "L 租赁和商务服务业",
    "M": "M 科学研究和技术服务业",
    "N": "N 水利、环境和公共设施管理业",
    "O": "O 居民服务、修理和其他服务业",
    "P": "P 教育",
    "Q": "Q 卫生和社会工作",
    "R": "R 文化、体育和娱乐业",
    "S": "S 公共管理、社会保障和社会组织",
    "T": "T 国际组织",
    "U": "U 未识别/其他",
}


def _norm_text(s: str) -> str:
    s = str(s or "").strip().lower()
    s = s.replace("/", "_").replace("|", "_").replace(" ", "")
    return s


def map_to_standard_industry(raw: str) -> str:
    x = _norm_text(raw)
    if not x:
        return STD_INDUSTRY["U"]

    # specific first
    if any(k in x for k in ["国际组织", "联合国", "使馆", "领事馆"]):
        return STD_INDUSTRY["T"]
    if any(k in x for k in ["政府", "公共管理", "社会保障", "社会组织", "事业单位", "机关"]):
        return STD_INDUSTRY["S"]

    if any(k in x for k in ["农", "林", "牧", "渔", "养殖", "种植", "农副"]):
        return STD_INDUSTRY["A"]
    if any(k in x for k in ["采矿", "矿产", "煤炭", "矿业", "石油开采", "天然气开采"]):
        return STD_INDUSTRY["B"]

    if any(k in x for k in ["电力", "热力", "燃气", "自来水", "供水", "供电"]):
        return STD_INDUSTRY["D"]

    if any(k in x for k in ["房地产", "房产", "物业", "地产", "房屋中介"]):
        return STD_INDUSTRY["K"]

    if any(k in x for k in ["建筑", "施工", "土木", "装修", "建材", "工程施工", "建筑设备安装"]):
        return STD_INDUSTRY["E"]

    if any(k in x for k in ["互联网", "it", "信息技术", "软件", "通信", "网络", "云计算", "大数据", "人工智能", "计算机"]):
        return STD_INDUSTRY["I"]

    if any(k in x for k in ["银行", "保险", "证券", "期货", "基金", "投资", "融资", "信托", "金融"]):
        return STD_INDUSTRY["J"]

    if any(k in x for k in ["学校", "教育", "培训", "辅导", "学历教育"]):
        return STD_INDUSTRY["P"]

    if any(k in x for k in ["医院", "卫生", "医疗服务", "护理", "康复", "社会工作", "养老服务"]):
        return STD_INDUSTRY["Q"]

    if any(k in x for k in ["文化", "体育", "娱乐", "影视", "传媒", "出版", "新媒体", "广告"]):
        return STD_INDUSTRY["R"]

    if any(k in x for k in ["交通", "运输", "物流", "仓储", "邮政", "客运", "货运", "快递"]):
        return STD_INDUSTRY["G"]

    if any(k in x for k in ["餐饮", "酒店", "住宿", "民宿"]):
        return STD_INDUSTRY["H"]

    if any(k in x for k in ["批发", "零售", "商贸", "贸易", "电子商务"]):
        return STD_INDUSTRY["F"]

    if any(k in x for k in ["科研", "研究", "检测", "认证", "工程设计", "专业技术", "专利", "技术服务"]):
        return STD_INDUSTRY["M"]

    if any(k in x for k in ["水利", "环保", "环境", "公共设施", "园林", "环卫"]):
        return STD_INDUSTRY["N"]

    if any(k in x for k in ["租赁", "人力资源", "企业服务", "咨询", "法律", "翻译", "商务服务", "代理"]):
        return STD_INDUSTRY["L"]

    if any(k in x for k in ["居民服务", "维修", "修理", "家政", "美容", "美发", "保健", "洗浴"]):
        return STD_INDUSTRY["O"]

    if any(k in x for k in [
        "制造", "加工", "电子设备", "机械", "半导体", "汽车制造", "医药制造", "化工", "金属制品", "纺织", "服装", "家具", "印刷", "包装", "仪器仪表", "食品", "饮料", "工业"
    ]):
        return STD_INDUSTRY["C"]

    # generic service fallback
    if "服务" in x:
        return STD_INDUSTRY["O"]

    return STD_INDUSTRY["U"]


def infer_from_company_name(company: str) -> str:
    x = _norm_text(company)
    if not x:
        return STD_INDUSTRY["U"]
    return map_to_standard_industry(x)


def extract_raw_industry(row: Dict[str, str]) -> str:
    c1 = str(row.get("初级分类", "") or "").strip()
    if c1 and c1.lower() != "nan":
        return c1

    c2 = str(row.get("招聘类别", "") or "").strip()
    if "<" in c2:
        return c2.split("<")[-1].strip()

    if c2 and c2 not in {"全职", "兼职", "兼职/临时", "校园", "实习"}:
        return c2

    return ""


@dataclass
class CompanyIndustry:
    raw_industry: str
    standard_industry: str
    confidence: float


def window_files() -> List[Path]:
    files = [
        p for p in WINDOWS_DIR.glob("window_*.csv")
        if re.match(r"window_\d{4}_\d{4}\.csv", p.name)
    ]
    return sorted(files)


def parse_year(v: str) -> Optional[int]:
    if v is None:
        return None
    s = str(v).strip()
    if not s:
        return None
    try:
        return int(float(s))
    except Exception:
        return None


def load_bigram_set_and_expected_counts() -> Tuple[Set[str], Dict[int, int]]:
    print("[1/7] Loading bigram set + expected counts from processed_corpus.jsonl ...")
    bigrams: Set[str] = set()
    expected = defaultdict(int)

    with PROCESSED_CORPUS.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            obj = json.loads(line)
            y = parse_year(obj.get("year"))
            if y is not None:
                expected[y] += 1
            for t in obj.get("tokens", []):
                if "_" in t:
                    bigrams.add(t)
            if i % 1_500_000 == 0:
                print(f"  scanned {i:,} lines; bigrams={len(bigrams):,}")

    print(f"  bigrams: {len(bigrams):,}")
    print(f"  expected years: {dict(sorted(expected.items()))}")
    return bigrams, dict(expected)


def collect_companies_from_windows() -> Set[str]:
    print("[2/7] Collecting company names from windows ...")
    comps: Set[str] = set()
    n = 0
    for fp in window_files():
        with fp.open("r", encoding="utf-8-sig", errors="replace", newline="") as f:
            rd = csv.DictReader(f)
            for row in rd:
                c = str(row.get("企业名称", "") or "").strip()
                if c:
                    comps.add(c)
                n += 1
        print(f"  {fp.name}: rows scanned cumulative={n:,}, unique companies={len(comps):,}")
    return comps


def build_company_lookup(target_companies: Set[str]) -> Dict[str, CompanyIndustry]:
    print("[3/7] Building company -> industry lookup from cleaned metadata ...")

    # (company, raw_industry) -> count
    pair_counts: Dict[Tuple[str, str], int] = defaultdict(int)
    n_total = 0
    n_hit_company = 0
    n_labeled = 0

    usecols = ["企业名称", "招聘类别", "初级分类"]
    for i, chunk in enumerate(pd.read_csv(CLEANED_METADATA, usecols=usecols, chunksize=300_000, low_memory=False), 1):
        n_total += len(chunk)

        # fast pre-filter by company
        chunk["企业名称"] = chunk["企业名称"].fillna("").astype(str).str.strip()
        chunk = chunk[chunk["企业名称"].isin(target_companies)]
        n_hit_company += len(chunk)

        if not chunk.empty:
            raw_list = []
            for _, r in chunk.iterrows():
                raw = extract_raw_industry(r)
                raw_list.append(raw)
            chunk = chunk.copy()
            chunk["_raw"] = raw_list
            chunk = chunk[chunk["_raw"] != ""]
            n_labeled += len(chunk)

            if not chunk.empty:
                grp = chunk.groupby(["企业名称", "_raw"], sort=False).size()
                for (comp, raw), cnt in grp.items():
                    pair_counts[(str(comp), str(raw))] += int(cnt)

        if i % 5 == 0:
            print(
                f"  chunks={i}, scanned={n_total:,}, matched_comp_rows={n_hit_company:,}, "
                f"labeled_rows={n_labeled:,}, pair_keys={len(pair_counts):,}"
            )

    total_by_company: Dict[str, int] = defaultdict(int)
    best_raw: Dict[str, Tuple[str, int]] = {}

    for (comp, raw), cnt in pair_counts.items():
        total_by_company[comp] += cnt
        if comp not in best_raw or cnt > best_raw[comp][1]:
            best_raw[comp] = (raw, cnt)

    lookup: Dict[str, CompanyIndustry] = {}
    for comp, (raw, best_cnt) in best_raw.items():
        total = max(1, total_by_company.get(comp, best_cnt))
        conf = best_cnt / total
        std = map_to_standard_industry(raw)
        lookup[comp] = CompanyIndustry(raw_industry=raw, standard_industry=std, confidence=conf)

    print(f"  company lookup size: {len(lookup):,}")

    # save lookup file
    HET_DIR.mkdir(parents=True, exist_ok=True)
    with COMPANY_LOOKUP_CSV.open("w", encoding="utf-8-sig", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["企业名称", "raw_industry", "standard_industry", "confidence"])
        for comp, info in lookup.items():
            wr.writerow([comp, info.raw_industry, info.standard_industry, f"{info.confidence:.4f}"])

    print(f"  saved: {COMPANY_LOOKUP_CSV}")
    return lookup


def resolve_industry(company: str, lookup: Dict[str, CompanyIndustry]) -> Tuple[str, str, float, str]:
    if company in lookup:
        info = lookup[company]
        if info.standard_industry != STD_INDUSTRY["U"]:
            return info.raw_industry, info.standard_industry, info.confidence, "company_lookup"

    # fallback: company-name keyword inference
    std = infer_from_company_name(company)
    if std != STD_INDUSTRY["U"]:
        return "", std, 0.0, "company_name_keyword"

    return "", STD_INDUSTRY["U"], 0.0, "unknown"


def reconstruct_id_metadata(bigrams: Set[str], expected_counts: Dict[int, int], company_lookup: Dict[str, CompanyIndustry]) -> Dict[int, int]:
    print("[4/7] Reconstructing id-aligned metadata ...")

    stopwords = set(DEFAULT_STOPWORDS)
    year_seq = defaultdict(int)

    HET_DIR.mkdir(parents=True, exist_ok=True)
    with RECONSTRUCTED_META_CSV.open("w", encoding="utf-8-sig", newline="") as out:
        wr = csv.writer(out)
        wr.writerow([
            "id", "year", "企业名称", "raw_industry", "standard_industry",
            "industry_source", "industry_confidence"
        ])

        scanned = 0
        kept = 0

        for fp in window_files():
            print(f"  processing {fp.name} ...")
            with fp.open("r", encoding="utf-8-sig", errors="replace", newline="") as f:
                rd = csv.DictReader(f)
                for row in rd:
                    scanned += 1
                    year = parse_year(row.get("招聘发布年份"))
                    if year is None:
                        continue

                    text = row.get("cleaned_requirements", "")
                    tokens = tokenize(text, stopwords)
                    tokens = apply_bigrams(tokens, bigrams)
                    if len(tokens) < MIN_TOKENS:
                        continue

                    year_seq[year] += 1
                    seq = year_seq[year]
                    job_id = year * 10_000_000 + seq

                    company = str(row.get("企业名称", "") or "").strip()
                    raw_ind, std_ind, conf, source = resolve_industry(company, company_lookup)

                    wr.writerow([
                        job_id,
                        year,
                        company,
                        raw_ind,
                        std_ind,
                        source,
                        f"{conf:.4f}",
                    ])
                    kept += 1

                    if scanned % 800_000 == 0:
                        print(f"    scanned={scanned:,}, kept={kept:,}")

    print(f"  saved: {RECONSTRUCTED_META_CSV}")

    # diagnostics
    diag_rows = []
    all_years = sorted(set(expected_counts) | set(year_seq))
    for y in all_years:
        exp = expected_counts.get(y, 0)
        got = year_seq.get(y, 0)
        diag_rows.append({
            "year": y,
            "expected_from_processed_corpus": exp,
            "reconstructed_count": got,
            "diff": got - exp,
        })
    pd.DataFrame(diag_rows).to_csv(DIAGNOSTICS_CSV, index=False, encoding="utf-8-sig")
    print(f"  diagnostics saved: {DIAGNOSTICS_CSV}")

    return dict(year_seq)


def sort_csv_by_id(input_csv: Path, output_csv: Path) -> None:
    print(f"  sorting by id: {input_csv.name} -> {output_csv.name}")
    df = pd.read_csv(input_csv, low_memory=False)
    df["id"] = pd.to_numeric(df["id"], errors="coerce")
    df = df.dropna(subset=["id"]).copy()
    df["id"] = df["id"].astype("int64")
    df = df.sort_values("id")
    df.to_csv(output_csv, index=False, encoding="utf-8-sig")


def stream_merge_master(final_csv: Path = FINAL_RESULTS, meta_csv: Path = RECONSTRUCTED_META_CSV) -> None:
    print("[5/7] Merging reconstructed metadata with final_results_sample.csv ...")

    matched = 0
    unmatched = 0
    meta_advance = 0

    with (
        final_csv.open("r", encoding="utf-8-sig", newline="") as fr,
        meta_csv.open("r", encoding="utf-8-sig", newline="") as mr,
        MASTER_CSV.open("w", encoding="utf-8-sig", newline="") as out,
    ):

        f_reader = csv.DictReader(fr)
        m_reader = csv.DictReader(mr)

        fieldnames = [
            "id", "year", "entropy_score", "hhi_score", "dominant_topic_id", "dominant_topic_prob",
            "企业名称", "raw_industry", "standard_industry", "industry_source", "industry_confidence"
        ]
        wr = csv.DictWriter(out, fieldnames=fieldnames)
        wr.writeheader()

        try:
            m_row = next(m_reader)
        except StopIteration:
            m_row = None

        for i, f_row in enumerate(f_reader, 1):
            fid = int(f_row["id"])

            while m_row is not None and int(m_row["id"]) < fid:
                meta_advance += 1
                try:
                    m_row = next(m_reader)
                except StopIteration:
                    m_row = None
                    break

            if m_row is not None and int(m_row["id"]) == fid:
                row = {
                    "id": f_row["id"],
                    "year": f_row["year"],
                    "entropy_score": f_row["entropy_score"],
                    "hhi_score": f_row["hhi_score"],
                    "dominant_topic_id": f_row["dominant_topic_id"],
                    "dominant_topic_prob": f_row["dominant_topic_prob"],
                    "企业名称": m_row.get("企业名称", ""),
                    "raw_industry": m_row.get("raw_industry", ""),
                    "standard_industry": m_row.get("standard_industry", STD_INDUSTRY["U"]),
                    "industry_source": m_row.get("industry_source", "unknown"),
                    "industry_confidence": m_row.get("industry_confidence", "0.0"),
                }
                wr.writerow(row)
                matched += 1
            else:
                row = {
                    "id": f_row["id"],
                    "year": f_row["year"],
                    "entropy_score": f_row["entropy_score"],
                    "hhi_score": f_row["hhi_score"],
                    "dominant_topic_id": f_row["dominant_topic_id"],
                    "dominant_topic_prob": f_row["dominant_topic_prob"],
                    "企业名称": "",
                    "raw_industry": "",
                    "standard_industry": STD_INDUSTRY["U"],
                    "industry_source": "unmatched_id",
                    "industry_confidence": "0.0",
                }
                wr.writerow(row)
                unmatched += 1

            if i % 1_500_000 == 0:
                print(f"  merged rows={i:,}, matched={matched:,}, unmatched={unmatched:,}")

    print(f"  saved: {MASTER_CSV}")
    print(f"  merge stats: matched={matched:,}, unmatched={unmatched:,}, meta_advance={meta_advance:,}")


def aggregate_and_plot(min_industry_total: int = 10_000, min_year_cell: int = 500) -> None:
    print("[6/7] Aggregating yearly industry metrics ...")

    df = pd.read_csv(MASTER_CSV, low_memory=False)
    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
    df["entropy_score"] = pd.to_numeric(df["entropy_score"], errors="coerce")
    df["hhi_score"] = pd.to_numeric(df["hhi_score"], errors="coerce")
    df = df.dropna(subset=["year", "entropy_score", "hhi_score"])
    df["year"] = df["year"].astype(int)

    # exclude unknown for primary heterogeneity table
    df_known = df[df["standard_industry"] != STD_INDUSTRY["U"]].copy()

    yearly = (
        df_known
        .groupby(["year", "standard_industry"], as_index=False)
        .agg(
            sample_n=("id", "size"),
            mean_entropy=("entropy_score", "mean"),
            mean_hhi=("hhi_score", "mean"),
        )
    )
    yearly["valid_sample"] = yearly["sample_n"] >= min_year_cell
    yearly.to_csv(YEARLY_METRICS_CSV, index=False, encoding="utf-8-sig")

    yearly_valid = yearly[yearly["valid_sample"]].copy()
    yearly_valid.to_csv(YEARLY_METRICS_VALID_CSV, index=False, encoding="utf-8-sig")

    # filter low-total industries
    total_by_ind = yearly.groupby("standard_industry", as_index=False)["sample_n"].sum()
    keep_inds = set(total_by_ind[total_by_ind["sample_n"] >= min_industry_total]["standard_industry"])
    yearly_keep = yearly_valid[yearly_valid["standard_industry"].isin(keep_inds)].copy()

    # 2016 vs 2024 growth
    y16 = yearly_keep[yearly_keep["year"] == 2016][["standard_industry", "sample_n", "mean_entropy"]].rename(
        columns={"sample_n": "n_2016", "mean_entropy": "entropy_2016"}
    )
    y24 = yearly_keep[yearly_keep["year"] == 2024][["standard_industry", "sample_n", "mean_entropy"]].rename(
        columns={"sample_n": "n_2024", "mean_entropy": "entropy_2024"}
    )
    growth = y16.merge(y24, on="standard_industry", how="inner")
    growth["delta_entropy"] = growth["entropy_2024"] - growth["entropy_2016"]
    growth["growth_rate_pct"] = growth["delta_entropy"] / growth["entropy_2016"] * 100
    growth = growth.sort_values("growth_rate_pct", ascending=False)
    growth.to_csv(GROWTH_2016_2024_CSV, index=False, encoding="utf-8-sig")

    print(f"  saved: {YEARLY_METRICS_CSV}")
    print(f"  saved: {YEARLY_METRICS_VALID_CSV}")
    print(f"  saved: {GROWTH_2016_2024_CSV}")

    print("[7/7] Plotting figures ...")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    import matplotlib.pyplot as plt

    # Trend plot: top 12 industries by total sample in kept set
    top_ind = (
        yearly_keep.groupby("standard_industry")["sample_n"].sum()
        .sort_values(ascending=False)
        .head(12)
        .index
        .tolist()
    )
    p1 = yearly_keep[yearly_keep["standard_industry"].isin(top_ind)]

    plt.figure(figsize=(14, 8))
    for ind in top_ind:
        g = p1[p1["standard_industry"] == ind].sort_values("year")
        plt.plot(g["year"], g["mean_entropy"], marker="o", linewidth=1.8, label=ind)
    plt.xlabel("Year")
    plt.ylabel("Mean Entropy")
    plt.title("Industry Heterogeneity: Mean Entropy Trend (Top 12 by sample)")
    plt.grid(alpha=0.25)
    plt.legend(fontsize=8, ncol=2)
    plt.tight_layout()
    plt.savefig(TREND_FIG, dpi=180)
    plt.close()

    # Scatter: 2016 baseline vs growth rate
    plt.figure(figsize=(12, 7))
    if not growth.empty:
        plt.scatter(growth["entropy_2016"], growth["growth_rate_pct"], alpha=0.75)
        for _, r in growth.iterrows():
            plt.text(r["entropy_2016"], r["growth_rate_pct"], r["standard_industry"].split(" ", 1)[0], fontsize=8)
    plt.axhline(0, color="gray", linewidth=1)
    plt.xlabel("2016 Mean Entropy")
    plt.ylabel("Entropy Growth Rate (2016 -> 2024, %) ")
    plt.title("Industry Catch-up Pattern: Baseline vs Growth")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(SCATTER_FIG, dpi=180)
    plt.close()

    print(f"  saved: {TREND_FIG}")
    print(f"  saved: {SCATTER_FIG}")


def main() -> None:
    HET_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    bigrams, expected_counts = load_bigram_set_and_expected_counts()
    companies = collect_companies_from_windows()
    company_lookup = build_company_lookup(companies)
    reconstructed_counts = reconstruct_id_metadata(bigrams, expected_counts, company_lookup)

    # hard check for id reconstruction quality
    mismatches = {
        y: reconstructed_counts.get(y, 0) - expected_counts.get(y, 0)
        for y in sorted(set(expected_counts) | set(reconstructed_counts))
    }
    bad = {y: d for y, d in mismatches.items() if d != 0}
    if bad:
        print("[WARN] reconstruction count mismatch detected:")
        print(bad)

    print("[5/7-prep] Sorting inputs for deterministic id-based stream merge ...")
    sort_csv_by_id(RECONSTRUCTED_META_CSV, RECONSTRUCTED_META_SORTED_CSV)
    sort_csv_by_id(FINAL_RESULTS, FINAL_RESULTS_SORTED_CSV)

    stream_merge_master(final_csv=FINAL_RESULTS_SORTED_CSV, meta_csv=RECONSTRUCTED_META_SORTED_CSV)
    aggregate_and_plot()

    print("Done.")


if __name__ == "__main__":
    main()
