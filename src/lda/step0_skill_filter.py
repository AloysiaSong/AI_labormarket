#!/usr/bin/env python3
"""
Step 0: 从招聘广告中抽取技能/任务表达
输入: window CSV files (cleaned_requirements列)
输出: skill_filtered_corpus.csv (过滤后的技能/任务文本)
"""

import re
import os
import sys
import glob
import pandas as pd
import jieba
from pathlib import Path
from collections import defaultdict

# ── 路径配置 ──────────────────────────────────────────────
PROJECT = Path("/Users/yu/code/code2601/TY")
WINDOWS_DIR = PROJECT / "data/processed/windows"
ESCO_DICT = PROJECT / "data/esco/jieba_dict/esco_jieba_dict.txt"
OUTPUT_CSV = PROJECT / "output/skill_filtered_corpus.csv"
CHUNK_SIZE = 50_000  # 每次读取行数

# ── ESCO 词典加载 ─────────────────────────────────────────
def load_esco_terms(path):
    """加载ESCO技能词典，返回中文技能词set"""
    terms = set()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            word = parts[0]
            # 只保留含中文的词条（跳过纯英文ESCO原文）
            if re.search(r"[\u4e00-\u9fa5]", word) and len(word) >= 2:
                terms.add(word)
    return terms


# ── 切句 ──────────────────────────────────────────────────
# 编号模式：1、 2. 2． 3） （1） (1) 一、 二、
_NUM_PATTERN = re.compile(
    r"(?:[\d]+[、.．)]|[（(][\d一二三四五六七八九十]+[)）]|[一二三四五六七八九十]+[、.])"
)

def split_sentences(text):
    """将JD文本切成句子列表"""
    if not text or not isinstance(text, str):
        return []

    # Step 1: 按句号、分号、换行切
    parts = re.split(r"[。；;\n]", text)

    # Step 2: 按多空格切（≥2个空格）
    expanded = []
    for p in parts:
        expanded.extend(re.split(r"\s{2,}", p))

    # Step 3: 按编号模式切
    result = []
    for seg in expanded:
        sub_parts = _NUM_PATTERN.split(seg)
        result.extend(sub_parts)

    # Step 4: 按中文逗号做子句切分（解决"技能描述，经验要求"混写问题）
    # 只在子句足够长时才切，避免把短句碎片化
    final = []
    for s in result:
        s = s.strip()
        if not s or len(s) <= 3:
            continue
        # 如果含逗号且总长>15字符，尝试按逗号切
        if "，" in s and len(s) > 15:
            sub = [x.strip() for x in s.split("，") if x.strip() and len(x.strip()) > 3]
            if sub:
                final.extend(sub)
            else:
                final.append(s)
        else:
            final.append(s)
    return final


# ── 剔除规则 ──────────────────────────────────────────────
_REMOVE_RULES = [
    ("学历", re.compile(
        r"本科|大专|硕士|博士|学历|毕业|全日制|统招|中专|高中学历"
    )),
    ("经验年限", re.compile(
        # "X年以上"系列（含中文数字）
        r"[一二三四五六七八九十\d]+\s*年以上"
        r"|[一二三四五六七八九十\d]+\s*年及以上"
        r"|[一二三四五六七八九十\d]+\s*年相关"
        r"|[一二三四五六七八九十\d]+\s*年左右"
        # "工作经验"仅匹配前面有量词/年限修饰的（如"3年工作经验"、"相关工作经验"）
        # 不再裸匹配"经验"，避免误杀"EMC整改经验"等技能表述
        r"|工作经验|从业经验|工作经历"
        # 应届
        r"|应届毕业生"
    )),
    ("年龄外貌", re.compile(
        r"[\d\-\—]+岁|年龄|身高|形象|气质"
    )),
    ("福利", re.compile(
        r"五险|社保|公积金|年终奖|带薪|底薪|提成|补贴|住宿|食宿|工作餐"
        r"|工作时间|上班时间|周末双休|法定节假日|年假"
    )),
    ("营销", re.compile(
        r"直招|非中介|详见官网|投递|面试|简历|联系电话|联系方式|乘车路线"
    )),
    ("身份", re.compile(
        r"党员|政治面貌|户籍|男女不限|性别"
    )),
]

# ── 保留规则 ──────────────────────────────────────────────
_KEEP_PATTERNS = [
    # 技能动词
    re.compile(r"熟悉|掌握|精通|熟练|使用|运用|操作"),
    # 任务动词
    re.compile(
        r"负责|开发|设计|编写|分析|维护|搭建|策划|撰写|制定"
        r"|推动|跟进|对接|测试|调试|部署|实施|监控|优化|编程"
        r"|配置|搭建|组织|协调|编制|审核|核算|统计|整理|规划"
    ),
    # 证书/资质
    re.compile(
        r"资格证|资质证|从业资格|执业资格|职称|持有.*证书"
        r"|注册.*师|CPA|CFA|PMP|CIPS|CPPM|ISO|六西格玛"
    ),
]

# ── 软性人格词 ────────────────────────────────────────────
_SOFT_PATTERN = re.compile(
    r"吃苦耐劳|责任心|抗压|积极主动|热爱|认真负责|团队精神"
    r"|为人正直|服从安排|沟通能力|学习能力|团队合作|善于沟通"
    r"|工作态度|有耐心|性格开朗|亲和力|执行力强|上进心"
)


def classify_sentence(sent, esco_tokens):
    """
    三步判定：
    1. 命中剔除规则 → REMOVE
    2. 命中保留规则 → KEEP
    3. 含软性人格词或无信号 → DISCARD

    esco_tokens: 该句分词后的token集合（预先分好传入）
    返回: ("KEEP",) / ("REMOVE", label) / ("DISCARD",)
    """
    # Step 1: 剔除规则
    for label, pat in _REMOVE_RULES:
        if pat.search(sent):
            return ("REMOVE", label)

    # Step 2: 保留规则 - 关键词
    for pat in _KEEP_PATTERNS:
        if pat.search(sent):
            return ("KEEP",)

    # Step 2b: 保留规则 - ESCO词典命中
    if esco_tokens:  # 如果有任何ESCO技能词命中
        return ("KEEP",)

    # Step 3: 软性人格词 → 剔除; 其余 → 丢弃
    if _SOFT_PATTERN.search(sent):
        return ("REMOVE", "软性人格")

    return ("DISCARD",)


def esco_match(sent, esco_set):
    """检查句子中是否有ESCO技能词（用jieba分词后匹配）"""
    tokens = jieba.lcut(sent)
    matched = set()
    for t in tokens:
        if t in esco_set:
            matched.add(t)
    return matched


# ── 单条JD处理 ────────────────────────────────────────────
def process_one_jd(text, esco_set):
    """
    处理一条JD文本，返回:
      skill_text, n_total, n_kept
    """
    sentences = split_sentences(text)
    n_total = len(sentences)
    kept = []

    for sent in sentences:
        esco_tokens = esco_match(sent, esco_set)
        result = classify_sentence(sent, esco_tokens)
        if result[0] == "KEEP":
            kept.append(sent)

    skill_text = "；".join(kept) if kept else ""
    return skill_text, n_total, len(kept)


# ── 窗口标签 ──────────────────────────────────────────────
def window_tag(filename):
    """从文件名提取窗口标签，如 window_2022_2023.csv -> w2022"""
    m = re.search(r"window_(\d{4})_\d{4}", filename)
    return f"w{m.group(1)}" if m else "wunk"


# ── 主流程 ────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("Step 0: 技能/任务句抽取")
    print("=" * 60)

    # 加载ESCO词典
    print(f"\n加载ESCO词典: {ESCO_DICT}")
    esco_set = load_esco_terms(ESCO_DICT)
    print(f"  中文技能词条数: {len(esco_set):,}")

    # 加载jieba ESCO词典（提高分词准确度）
    jieba.load_userdict(str(ESCO_DICT))

    # 找到所有窗口文件
    csv_files = sorted(glob.glob(str(WINDOWS_DIR / "window_*.csv")))
    csv_files = [f for f in csv_files if "stats" not in f]
    print(f"\n找到 {len(csv_files)} 个窗口文件")

    # 统计容器
    stats_by_year = defaultdict(lambda: {
        "total": 0, "has_skill": 0,
        "char_counts": [], "kept_ratios": []
    })

    # 输出目录
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    # 写CSV header
    first_write = True
    total_processed = 0

    for csv_path in csv_files:
        fname = os.path.basename(csv_path)
        tag = window_tag(fname)
        print(f"\n处理: {fname} (tag={tag})")

        row_offset = 0
        for chunk in pd.read_csv(csv_path, chunksize=CHUNK_SIZE,
                                  dtype={"cleaned_requirements": str}):
            rows = []
            for i, row in chunk.iterrows():
                local_idx = row_offset + (i - chunk.index[0])
                job_id = f"{tag}_{local_idx}"
                year = int(row["招聘发布年份"]) if pd.notna(row["招聘发布年份"]) else 0
                text = row.get("cleaned_requirements", "")
                if not isinstance(text, str) or not text.strip():
                    text = ""

                original_char_count = len(text)

                if text:
                    skill_text, n_total, n_kept = process_one_jd(text, esco_set)
                else:
                    skill_text, n_total, n_kept = "", 0, 0

                skill_char_count = len(skill_text)
                has_skill = 1 if skill_char_count >= 10 else 0

                rows.append({
                    "job_id": job_id,
                    "year": year,
                    "skill_text": skill_text,
                    "has_skill_text": has_skill,
                    "skill_char_count": skill_char_count,
                    "original_char_count": original_char_count,
                    "n_sentences_total": n_total,
                    "n_sentences_kept": n_kept,
                })

                # 收集统计
                s = stats_by_year[year]
                s["total"] += 1
                s["has_skill"] += has_skill
                s["char_counts"].append(skill_char_count)
                if n_total > 0:
                    s["kept_ratios"].append(n_kept / n_total)

            # 写出
            df_out = pd.DataFrame(rows)
            df_out.to_csv(OUTPUT_CSV, mode="a", index=False,
                          header=first_write, encoding="utf-8")
            first_write = False

            row_offset += len(chunk)
            total_processed += len(chunk)

            if total_processed % 200_000 < CHUNK_SIZE:
                print(f"  已处理: {total_processed:,} 行", end="\r")

        print(f"  完成: {row_offset:,} 行")

    # ── 输出统计摘要 ────────────────────────────────────
    print("\n" + "=" * 60)
    print("统计摘要")
    print("=" * 60)
    print(f"\n总JD数: {total_processed:,}")

    total_has = sum(s["has_skill"] for s in stats_by_year.values())
    print(f"has_skill_text=1: {total_has:,} ({total_has/total_processed*100:.1f}%)")

    import numpy as np
    all_chars = []
    for s in stats_by_year.values():
        all_chars.extend(s["char_counts"])
    all_chars = np.array(all_chars)
    print(f"\nskill_char_count 分布:")
    print(f"  均值: {all_chars.mean():.1f}")
    print(f"  中位数: {np.median(all_chars):.1f}")
    print(f"  P25: {np.percentile(all_chars, 25):.1f}")
    print(f"  P75: {np.percentile(all_chars, 75):.1f}")

    all_ratios = []
    for s in stats_by_year.values():
        all_ratios.extend(s["kept_ratios"])
    all_ratios = np.array(all_ratios)
    print(f"\nn_kept / n_total 分布:")
    print(f"  均值: {all_ratios.mean():.3f}")
    print(f"  中位数: {np.median(all_ratios):.3f}")

    print(f"\n按年分组:")
    print(f"{'year':>6}  {'总数':>10}  {'有技能%':>8}  {'字符均值':>8}  {'保留比均值':>10}")
    for year in sorted(stats_by_year.keys()):
        s = stats_by_year[year]
        if s["total"] == 0:
            continue
        chars = np.array(s["char_counts"])
        ratios = np.array(s["kept_ratios"]) if s["kept_ratios"] else np.array([0])
        print(f"{year:>6}  {s['total']:>10,}  "
              f"{s['has_skill']/s['total']*100:>7.1f}%  "
              f"{chars.mean():>8.1f}  "
              f"{ratios.mean():>10.3f}")

    print(f"\n输出文件: {OUTPUT_CSV}")
    print(f"文件大小: {OUTPUT_CSV.stat().st_size / 1e9:.2f} GB")


if __name__ == "__main__":
    main()
