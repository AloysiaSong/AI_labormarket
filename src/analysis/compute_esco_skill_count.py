#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
compute_esco_skill_count.py
从 window_*.csv 的 cleaned_requirements 字段中，
统计每条岗位描述包含的 ESCO 技能词条数量（中文）。

这是 LDA entropy 指标的稳健性检验补充：
  - LDA entropy 衡量技能综合性（话题分布的熵）
  - ESCO skill count 衡量技能宽度（匹配到的技能词条数）
  - 两者若趋势一致，增强结论可信度

输出：
  data/Heterogeneity/esco_skill_count_by_id.csv
    列：id, year, skill_count, unique_skill_count
  data/Heterogeneity/yearly_esco_skill_stats.csv
    列：year, n, mean_skill_count, mean_unique_skill_count

ESCO 技能词来自：
  data/esco/jieba_dict/skills_ch.csv  (53,474 条中文技能词)
  选取 preferredLabel（首选标签），单字词过滤
"""

import csv
import re
from pathlib import Path
from collections import defaultdict

BASE = Path('/Users/yu/code/code2601/TY')
SKILLS_CH_CSV = BASE / 'data/esco/jieba_dict/skills_ch.csv'
WINDOWS_DIR = BASE / 'data/processed/windows'
OUT_BY_ID = BASE / 'data/Heterogeneity/esco_skill_count_by_id.csv'
OUT_YEARLY = BASE / 'data/Heterogeneity/yearly_esco_skill_stats.csv'

# 与 create_sorted_results.py 一致的 ID 分配方案
# 读 window 文件顺序必须与 create_sorted_results.py 相同（字母排序）

MIN_SKILL_LEN = 2   # 过滤单字技能词（噪声太多）
MIN_TOKENS = 5       # 同 create_sorted_results.py


def load_esco_skills():
    """
    加载 ESCO 中文技能词，返回按长度降序排列的词列表（长词优先匹配）。
    只取 preferredLabel 列（首选标签），过滤单字。
    """
    skills = set()
    with SKILLS_CH_CSV.open(encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            label = row.get('preferredLabel', '').strip()
            if len(label) >= MIN_SKILL_LEN:
                skills.add(label)
    # 按长度降序，长词优先（避免短词把长词拆断）
    return sorted(skills, key=len, reverse=True)


def count_skills(text: str, skills: list):
    """
    返回 (total_count, unique_count)：
      total_count   = 所有技能词命中次数之和
      unique_count  = 命中的不重复技能词数
    """
    if not text:
        return 0, 0
    t = text.lower()
    matched = set()
    for skill in skills:
        if skill in t:
            matched.add(skill)
    return len(matched), len(matched)  # 本实现 total=unique（每词计1次）


def window_files():
    files = [p for p in WINDOWS_DIR.glob('window_*.csv')
             if re.match(r'window_\d{4}_\d{4}\.csv', p.name)]
    return sorted(files)


def parse_year(v):
    try:
        return int(float(str(v).strip()))
    except Exception:
        return None


def main():
    print('[1/3] 加载 ESCO 中文技能词...')
    skills = load_esco_skills()
    print(f'  技能词条: {len(skills):,}（已按长度降序）')

    print('[2/3] 处理 window 文件...')
    OUT_BY_ID.parent.mkdir(parents=True, exist_ok=True)

    year_seq = defaultdict(int)
    yearly_stats = defaultdict(list)
    total = 0
    kept = 0

    with OUT_BY_ID.open('w', encoding='utf-8-sig', newline='') as fout:
        writer = csv.writer(fout)
        writer.writerow(['id', 'year', 'skill_count', 'unique_skill_count'])

        for fp in window_files():
            print(f'  处理: {fp.name}')
            with fp.open('r', encoding='utf-8-sig', errors='replace', newline='') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    total += 1
                    year = parse_year(row.get('招聘发布年份'))
                    if year is None:
                        continue
                    req = row.get('cleaned_requirements', '')
                    if not req or len(req.strip()) < 10:
                        continue

                    year_seq[year] += 1
                    jid = year * 10_000_000 + year_seq[year]

                    skill_count, unique_count = count_skills(req, skills)
                    writer.writerow([jid, year, skill_count, unique_count])
                    kept += 1

                    if skill_count > 0:
                        yearly_stats[year].append(skill_count)

                    if total % 1_000_000 == 0:
                        print(f'    scanned={total:,}, kept={kept:,}')

    print(f'\n完成: 扫描={total:,}, 输出={kept:,}')

    print('[3/3] 汇总年度统计...')
    with OUT_YEARLY.open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['year', 'n_with_skills', 'mean_skill_count', 'median_skill_count'])
        for year in sorted(yearly_stats.keys()):
            vals = yearly_stats[year]
            n = len(vals)
            mean_v = sum(vals) / n
            sorted_v = sorted(vals)
            median_v = sorted_v[n // 2]
            writer.writerow([year, n, f'{mean_v:.3f}', median_v])

    print(f'\n输出:')
    print(f'  {OUT_BY_ID}')
    print(f'  {OUT_YEARLY}')


if __name__ == '__main__':
    main()
