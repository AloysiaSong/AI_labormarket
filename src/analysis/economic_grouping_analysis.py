#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
economic_grouping_analysis.py
在国标行业20类分类基础上，叠加两套经济理论分组，
计算各分组的年度熵值趋势。

分组方案：
  A. 技术类型分组（用户建议）
     - 技术驱动型：软件/半导体/互联网（I/M/J 大类）
     - 服务驱动型：咨询/金融/人力资源（K/J/L 大类的服务子集）
     - 制造型：C/D/E/B 大类
     - 传统型：A/F/H/O/N 大类

  B. 新古典 K/L 分组（Solow 1956, Krusell et al. 2000）
     - 资本密集型：D/B/E/G/K/I
     - 劳动密集型：A/C/F/H/O/N/Q

  C. 内生增长分组（Romer 1990, Lucas 1988, Acemoglu 2002）
     - 思想/知识前沿部门：I/M/J/R
     - 人力资本密集：P/Q/L
     - 物质资本密集：D/E/B/G/K
     - 劳动密集型：C/A/H/F/O/N

输入：data/Heterogeneity/yearly_industry20_entropy_metrics.csv
输出：data/Heterogeneity/yearly_economic_grouping_entropy.csv
"""

import csv
from collections import defaultdict
from pathlib import Path

BASE = Path('/Users/yu/code/code2601/TY')
YEARLY_CSV = BASE / 'data/Heterogeneity/yearly_industry20_entropy_metrics.csv'
OUT_CSV = BASE / 'data/Heterogeneity/yearly_economic_grouping_entropy.csv'

# ─────────────────────────────────────────────
# 国标行业20类 → 各分组映射
# ─────────────────────────────────────────────

# A. 技术类型分组（用户建议：技术驱动 vs 服务驱动 etc.）
TECH_TYPE_GROUP = {
    'I': 'tech_driven',    # 信息传输、软件和信息技术服务业
    'M': 'tech_driven',    # 科学研究和技术服务业
    'J': 'tech_driven',    # 金融业（互联网金融/科技金融含在内）
    'K': 'service_driven', # 房地产业（咨询/中介）
    'L': 'service_driven', # 租赁和商务服务业（含咨询/HR/广告）
    'N': 'service_driven', # 水利、环境和公共设施管理业
    'O': 'service_driven', # 居民服务、修理和其他服务业
    'P': 'service_driven', # 教育
    'Q': 'service_driven', # 卫生和社会工作
    'R': 'service_driven', # 文化、体育和娱乐业
    'S': 'service_driven', # 公共管理、社会保障和社会组织
    'G': 'service_driven', # 批发和零售业
    'H': 'service_driven', # 交通运输、仓储和邮政业
    'T': 'service_driven', # 国际组织
    'C': 'manufacturing',  # 制造业
    'D': 'manufacturing',  # 电力、热力、燃气及水生产和供应业
    'E': 'manufacturing',  # 建筑业
    'B': 'manufacturing',  # 采矿业
    'F': 'manufacturing',  # 建筑业（注：有时F=建筑，E=水利）
    'A': 'traditional',    # 农、林、牧、渔业
    'U': 'traditional',    # 未分类/其他
}

# B. 新古典 K/L 分组
NEOCLASSICAL_GROUP = {
    'D': 'capital_intensive',
    'B': 'capital_intensive',
    'E': 'capital_intensive',
    'G': 'capital_intensive',
    'K': 'capital_intensive',
    'I': 'capital_intensive',
    'J': 'capital_intensive',
    'A': 'labor_intensive',
    'C': 'labor_intensive',
    'F': 'labor_intensive',
    'H': 'labor_intensive',
    'O': 'labor_intensive',
    'N': 'labor_intensive',
    'L': 'labor_intensive',
    'M': 'capital_intensive',
    'P': 'labor_intensive',
    'Q': 'labor_intensive',
    'R': 'labor_intensive',
    'S': 'labor_intensive',
    'T': 'labor_intensive',
    'U': 'unclassified',
}

# C. 内生增长分组
ENDOGENOUS_GROUP = {
    'I': 'ideas_frontier',
    'M': 'ideas_frontier',
    'J': 'ideas_frontier',
    'R': 'ideas_frontier',
    'P': 'human_capital',
    'Q': 'human_capital',
    'L': 'human_capital',
    'D': 'physical_capital',
    'E': 'physical_capital',
    'B': 'physical_capital',
    'G': 'physical_capital',
    'K': 'physical_capital',
    'C': 'labor_intensive',
    'A': 'labor_intensive',
    'H': 'labor_intensive',
    'F': 'labor_intensive',
    'O': 'labor_intensive',
    'N': 'labor_intensive',
    'S': 'labor_intensive',
    'T': 'labor_intensive',
    'U': 'unclassified',
}

GROUP_SCHEMES = {
    'tech_type': TECH_TYPE_GROUP,
    'neoclassical': NEOCLASSICAL_GROUP,
    'endogenous_growth': ENDOGENOUS_GROUP,
}


def load_yearly_data():
    """读取 yearly_industry20_entropy_metrics.csv，
    返回 {(year, industry20_code): (mean_entropy, n)} 字典。"""
    data = {}
    with YEARLY_CSV.open(encoding='utf-8-sig') as f:
        for row in csv.DictReader(f):
            try:
                year = int(row['year'])
                code = row['industry20_code'].strip()
                mean_e = float(row['mean_entropy'])
                n = int(row['sample_n'])
                data[(year, code)] = (mean_e, n)
            except (ValueError, KeyError):
                continue
    return data


def compute_group_entropy(data: dict, scheme: dict):
    """
    按分组方案聚合，计算加权平均熵值。
    返回 {(year, group): (weighted_mean_entropy, total_n)} 字典。
    """
    groups = defaultdict(lambda: [0.0, 0])  # {(year, group): [sum_weighted_e, total_n]}
    for (year, code), (mean_e, n) in data.items():
        group = scheme.get(code, 'unclassified')
        groups[(year, group)][0] += mean_e * n
        groups[(year, group)][1] += n
    result = {}
    for (year, group), (sum_we, total_n) in groups.items():
        if total_n > 0:
            result[(year, group)] = (sum_we / total_n, total_n)
    return result


def main():
    print('加载年度行业熵值数据...')
    data = load_yearly_data()
    print(f'  读取 {len(data):,} 条 (year, industry) 记录')

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    rows_out = []

    for scheme_name, scheme in GROUP_SCHEMES.items():
        grouped = compute_group_entropy(data, scheme)
        for (year, group), (mean_e, n) in grouped.items():
            rows_out.append({
                'scheme': scheme_name,
                'year': year,
                'group': group,
                'n': n,
                'mean_entropy': round(mean_e, 6),
            })

    # 排序输出
    rows_out.sort(key=lambda r: (r['scheme'], r['group'], r['year']))

    with OUT_CSV.open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['scheme', 'year', 'group', 'n', 'mean_entropy'])
        writer.writeheader()
        writer.writerows(rows_out)

    print(f'输出: {OUT_CSV}')

    # ── 打印摘要：各分组 2016 → 2024 变化 ──
    print('\n── 分组熵值趋势摘要 ──')
    for scheme_name in GROUP_SCHEMES:
        print(f'\n[{scheme_name}]')
        groups_in_scheme = sorted(set(r['group'] for r in rows_out if r['scheme'] == scheme_name))
        for group in groups_in_scheme:
            entries = [(r['year'], r['mean_entropy']) for r in rows_out
                       if r['scheme'] == scheme_name and r['group'] == group]
            entries.sort()
            years = [e[0] for e in entries]
            if 2016 in years and 2024 in years:
                e2016 = next(e[1] for e in entries if e[0] == 2016)
                e2024 = next(e[1] for e in entries if e[0] == 2024)
                chg = (e2024 - e2016) / e2016 * 100
                print(f'  {group:25s}: 2016={e2016:.4f} → 2024={e2024:.4f}  Δ={chg:+.1f}%')


if __name__ == '__main__':
    main()
