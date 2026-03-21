#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
experiment_task_measures.py

Atalay-style task decomposition: extract verb-noun task keywords from
Chinese job descriptions, classify into Spitz-Oener categories, and test
whether AI exposure predicts differential trends in specific task types.

Key innovations vs prior experiments:
1. Outcome: task-level intensity (keyword counts per category) instead of
   aggregate entropy — directly interpretable
2. Exposure: decompose AI score into substitution vs augmentation exposure
   based on actual task content of each occupation

Pipeline:
  Step 1: Build job-title → ISCO lookup from master CSV
  Step 2: Scan window CSVs for task keywords, link to ISCO
  Step 3: Aggregate to ISCO×year cells
  Step 4: WLS regressions on task intensity outcomes
  Step 5: Substitution vs augmentation exposure decomposition

Input:
  data/Heterogeneity/master_with_ai_exposure_v2.csv
  data/processed/windows/window_*.csv

Output:
  output/task_measures/
"""

import re
import time
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'Heiti SC']
rcParams['axes.unicode_minus'] = False

# ── Paths ──────────────────────────────────────────────────────────
BASE = Path('/Users/yu/code/code2601/TY')
MASTER = BASE / 'data/Heterogeneity/master_with_ai_exposure_v2.csv'
WINDOWS_DIR = BASE / 'data/processed/windows'
OUT = BASE / 'output/task_measures'
OUT.mkdir(parents=True, exist_ok=True)

CHUNK = 200_000
REF_YEAR = 2016

# ── Task keyword dictionaries (Chinese verb-noun pairs) ──────────
# Following Spitz-Oener (2006) / Atalay et al. (2024) task categories

# RC: Routine Cognitive (AI替代型 — standardized information processing)
RC_KEYWORDS = [
    # Compound verb-noun phrases
    '数据录入', '信息录入', '文件归档', '资料整理', '报表制作',
    '数据统计', '单据处理', '票据整理', '台账登记', '凭证录入',
    '档案管理', '文档处理', '表格填写', '数据核对', '资料归档',
    '报表汇总', '数据汇总', '账务处理', '开具发票', '收发文件',
    '合同归档', '文书处理', '信息登记', '数据整理', '报表统计',
    # Short task-indicative words
    '录入', '归档', '核对', '台账', '发票', '单据', '凭证',
    '誊写', '抄录', '开票', '记账', '对账', '核账', '结算',
    '报表', '装订', '收发',
]

# NRA: Non-routine Analytical (AI增强型 — creative/analytical tasks)
NRA_KEYWORDS = [
    # Compound verb-noun phrases
    '数据分析', '系统开发', '产品研发', '方案设计', '算法开发',
    '架构设计', '技术研发', '需求分析', '性能优化', '程序编写',
    '代码开发', '系统设计', '技术方案', '产品设计', '结构设计',
    '模型设计', '策略分析', '竞品分析', '市场分析', '创新设计',
    '技术攻关', '方案编制', '程序开发', '技术开发', '独立研发',
    # Short task-indicative words
    '编程', '调试', '算法', '架构', '建模', '仿真',
    '研发', '创新', '专利', '论文', '实验',
]

# NRI: Non-routine Interactive (人际互动型)
NRI_KEYWORDS = [
    # Compound verb-noun phrases
    '客户沟通', '团队管理', '商务谈判', '培训指导', '项目协调',
    '市场推广', '销售拓展', '客户维护', '需求对接', '跨部门协调',
    '人员管理', '团队建设', '业务拓展', '客户开发', '渠道管理',
    '品牌推广', '客户关系', '组织协调', '带领团队', '指导下属',
    '商务洽谈', '战略规划', '人才培养', '绩效管理', '项目管理',
    # Short task-indicative words
    '谈判', '培训', '带教', '辅导', '演讲', '汇报',
    '公关', '拓展', '策划', '招商',
]

# RM: Routine Manual (例行体力型)
RM_KEYWORDS = [
    # Compound verb-noun phrases
    '设备操作', '产品组装', '货物搬运', '设备维修', '质量检测',
    '焊接加工', '机器操作', '产品包装', '物料搬运', '产线操作',
    '设备巡检', '零件加工', '装配调试', '设备安装', '工件加工',
    '设备保养', '模具维修', '线路安装', '管道安装', '车床操作',
    '叉车操作', '起重操作', '电气维修', '机械加工', '产品检验',
    # Short task-indicative words
    '焊接', '加工', '组装', '搬运', '装配', '巡检',
    '叉车', '打磨', '切割', '冲压', '车床', '铣床',
    '钻孔', '抛光', '涂装',
]

# Compile regex for each category (match any keyword)
def _compile_pattern(keywords):
    # Sort by length descending to match longer phrases first
    kws = sorted(set(keywords), key=len, reverse=True)
    return re.compile('|'.join(re.escape(kw) for kw in kws))

PAT_RC = _compile_pattern(RC_KEYWORDS)
PAT_NRA = _compile_pattern(NRA_KEYWORDS)
PAT_NRI = _compile_pattern(NRI_KEYWORDS)
PAT_RM = _compile_pattern(RM_KEYWORDS)

CATEGORIES = ['RC', 'NRA', 'NRI', 'RM']
PATTERNS = {'RC': PAT_RC, 'NRA': PAT_NRA, 'NRI': PAT_NRI, 'RM': PAT_RM}


# ══════════════════════════════════════════════════════════════════
# Step 1: Build title → ISCO lookup from master CSV
# ══════════════════════════════════════════════════════════════════

def build_isco_lookup():
    """Build job title → (isco, ai_score, ai_gradient) lookup from master."""
    print('[Step 1] Building title → ISCO lookup from master CSV ...')
    t0 = time.time()

    title_isco = {}  # title → {isco: count}
    title_ai = {}    # title → ai_score (constant per ISCO)

    cols = ['招聘岗位', 'isco08_4digit', 'ai_mean_score']
    for i, chunk in enumerate(pd.read_csv(MASTER, usecols=cols, chunksize=500_000)):
        chunk = chunk.dropna(subset=['招聘岗位', 'isco08_4digit'])
        for title, isco, ai in zip(chunk['招聘岗位'],
                                    chunk['isco08_4digit'].astype(int),
                                    chunk['ai_mean_score']):
            if title not in title_isco:
                title_isco[title] = Counter()
                title_ai[title] = {}
            title_isco[title][isco] += 1
            title_ai[title][isco] = ai

        if (i + 1) % 4 == 0:
            print(f'  Chunk {i+1}, {len(title_isco):,} unique titles')

    # Take mode ISCO for each title
    lookup = {}
    for title, counts in title_isco.items():
        best_isco = counts.most_common(1)[0][0]
        lookup[title] = {
            'isco': best_isco,
            'ai_score': title_ai[title].get(best_isco, np.nan),
        }

    print(f'  Built lookup: {len(lookup):,} unique titles')
    print(f'  Time: {time.time()-t0:.1f}s')
    return lookup


# ══════════════════════════════════════════════════════════════════
# Step 2: Scan window CSVs for task keywords
# ══════════════════════════════════════════════════════════════════

def count_task_keywords(text):
    """Count task keywords in each category for a given text."""
    counts = {}
    for cat, pat in PATTERNS.items():
        matches = pat.findall(text)
        counts[cat] = len(matches)
    counts['total'] = sum(counts.values())
    return counts


def scan_windows(lookup):
    """Scan window CSVs, extract task keywords, link to ISCO.
    Uses vectorized pandas str operations for speed."""
    print('\n[Step 2] Scanning window CSVs for task keywords ...')
    t0 = time.time()

    window_files = sorted(f for f in WINDOWS_DIR.glob('window_2*.csv'))
    print(f'  Found {len(window_files)} window files')

    # Build fast lookup: title_str → (isco, ai_score)
    title_to_isco = {t: v['isco'] for t, v in lookup.items()}
    title_to_ai = {t: v['ai_score'] for t, v in lookup.items()}

    cells = {}  # (isco, year) → aggregated stats
    total_rows = 0
    matched_rows = 0
    keyword_hits = Counter()

    for wf in window_files:
        print(f'\n  Reading {wf.name} ...')
        for chunk in pd.read_csv(wf, chunksize=CHUNK,
                                  dtype={'招聘发布年份': str}):
            total_rows += len(chunk)

            # Parse year (vectorized)
            chunk['yr'] = pd.to_numeric(chunk['招聘发布年份'], errors='coerce')
            chunk = chunk.dropna(subset=['yr'])
            chunk['yr'] = chunk['yr'].astype(int)
            chunk = chunk[(chunk['yr'] >= 2016) & (chunk['yr'] <= 2025)]

            if len(chunk) == 0:
                continue

            # Look up ISCO (vectorized via map)
            chunk['isco'] = chunk['招聘岗位'].map(title_to_isco)
            chunk['ai_score'] = chunk['招聘岗位'].map(title_to_ai)
            chunk = chunk.dropna(subset=['isco'])
            chunk['isco'] = chunk['isco'].astype(int)
            matched_rows += len(chunk)

            if len(chunk) == 0:
                continue

            # Count task keywords (vectorized str.count)
            text = chunk['cleaned_requirements'].fillna('').astype(str)
            chunk['textlen'] = text.str.len()

            for cat, pat in PATTERNS.items():
                # str.count counts non-overlapping matches
                chunk[f'cnt_{cat}'] = text.str.count(pat)
                keyword_hits[cat] += chunk[f'cnt_{cat}'].sum()

            chunk['cnt_total'] = sum(chunk[f'cnt_{cat}'] for cat in CATEGORIES)

            # Aggregate to ISCO×year cells within this chunk
            for (isco, yr), grp in chunk.groupby(['isco', 'yr']):
                key = (int(isco), int(yr))
                if key not in cells:
                    cells[key] = {
                        'n': 0,
                        'ai_score': grp['ai_score'].iloc[0],
                        'sum_textlen': 0,
                        **{f'sum_{cat}': 0 for cat in CATEGORIES},
                        'sum_total_kw': 0,
                        **{f'has_{cat}': 0 for cat in CATEGORIES},
                    }
                c = cells[key]
                c['n'] += len(grp)
                c['sum_textlen'] += grp['textlen'].sum()
                c['sum_total_kw'] += grp['cnt_total'].sum()
                for cat in CATEGORIES:
                    c[f'sum_{cat}'] += grp[f'cnt_{cat}'].sum()
                    c[f'has_{cat}'] += (grp[f'cnt_{cat}'] > 0).sum()

            if total_rows % 1_000_000 < CHUNK:
                print(f'    {total_rows:>10,} rows processed, '
                      f'{matched_rows:,} matched')

    print(f'\n  Total: {total_rows:,} rows, {matched_rows:,} matched '
          f'({matched_rows/total_rows*100:.1f}%)')
    print(f'  Cells: {len(cells):,}')
    print(f'  Keyword hits: {dict(keyword_hits)}')
    print(f'  Time: {time.time()-t0:.1f}s')

    # Build DataFrame
    rows = []
    for (isco, yr), c in cells.items():
        n = c['n']
        row = {
            'isco': isco,
            'year': yr,
            'n': n,
            'ai_score': c['ai_score'],
            'mean_textlen': c['sum_textlen'] / n,
        }
        for cat in CATEGORIES:
            row[f'mean_{cat}_count'] = c[f'sum_{cat}'] / n
            row[f'share_has_{cat}'] = c[f'has_{cat}'] / n
        total_kw = c['sum_total_kw']
        for cat in CATEGORIES:
            row[f'share_{cat}_kw'] = c[f'sum_{cat}'] / total_kw if total_kw > 0 else 0
        rows.append(row)

    df = pd.DataFrame(rows).sort_values(['isco', 'year']).reset_index(drop=True)
    df['log_textlen'] = np.log(df['mean_textlen'].clip(lower=1))
    return df


# ══════════════════════════════════════════════════════════════════
# Step 3: Descriptive statistics
# ══════════════════════════════════════════════════════════════════

def descriptive_stats(df):
    """Report task keyword frequencies and distributions."""
    print('\n[Step 3] Descriptive Statistics')
    print('=' * 70)

    lines = []
    lines.append('Task Keyword Descriptive Statistics')
    lines.append('=' * 70)

    # Overall means (weighted by cell size)
    lines.append('\nMean task keyword counts per job posting:')
    for cat in CATEGORIES:
        wmean = np.average(df[f'mean_{cat}_count'], weights=df['n'])
        wshare = np.average(df[f'share_has_{cat}'], weights=df['n'])
        lines.append(f'  {cat:>5}: mean count = {wmean:.4f}, '
                     f'share with ≥1 keyword = {wshare:.1%}')

    # By year
    lines.append('\nMean task keyword count by year (weighted):')
    lines.append(f'{"Year":<6} {"RC":>8} {"NRA":>8} {"NRI":>8} {"RM":>8} {"N":>12}')
    lines.append('-' * 56)
    for yr in sorted(df.year.unique()):
        sub = df[df.year == yr]
        n_total = sub.n.sum()
        vals = []
        for cat in CATEGORIES:
            vals.append(np.average(sub[f'mean_{cat}_count'], weights=sub['n']))
        lines.append(f'{yr:<6} {vals[0]:>8.4f} {vals[1]:>8.4f} '
                     f'{vals[2]:>8.4f} {vals[3]:>8.4f} {n_total:>12,}')

    # By AI exposure quartile
    lines.append('\nMean task keyword count by AI score quartile (weighted):')
    df['ai_q'] = pd.qcut(df['ai_score'], 4, labels=['Q1 (Low)', 'Q2', 'Q3', 'Q4 (High)'],
                          duplicates='drop')
    lines.append(f'{"Quartile":<12} {"RC":>8} {"NRA":>8} {"NRI":>8} {"RM":>8}')
    lines.append('-' * 48)
    for q in df['ai_q'].cat.categories:
        sub = df[df.ai_q == q]
        if len(sub) == 0:
            continue
        vals = []
        for cat in CATEGORIES:
            vals.append(np.average(sub[f'mean_{cat}_count'], weights=sub['n']))
        lines.append(f'{q:<12} {vals[0]:>8.4f} {vals[1]:>8.4f} '
                     f'{vals[2]:>8.4f} {vals[3]:>8.4f}')

    result_text = '\n'.join(lines)
    print(result_text)
    with open(OUT / 'descriptive_stats.txt', 'w') as f:
        f.write(result_text)

    return df


# ══════════════════════════════════════════════════════════════════
# Step 4: Task intensity regressions
# ══════════════════════════════════════════════════════════════════

def run_task_regressions(df):
    """WLS regressions: task_intensity ~ C(year) × ai_score + log_textlen."""
    print('\n[Step 4] Task Intensity Regressions')
    print('=' * 70)

    results = {}
    lines = []
    lines.append('Task Intensity WLS Regressions')
    lines.append('=' * 70)
    lines.append('DV: mean keyword count per category')
    lines.append('Model: DV ~ C(year) × ai_score + log_textlen')
    lines.append('WLS weights = cell size, HC3 standard errors')

    for cat in CATEGORIES:
        dv = f'mean_{cat}_count'
        lines.append(f'\n{"=" * 60}')
        lines.append(f'{cat} (mean keyword count)')
        lines.append(f'{"=" * 60}')

        # Model without text length control
        try:
            m1 = smf.wls(f'{dv} ~ C(year) * ai_score',
                          data=df, weights=df['n']).fit(cov_type='HC3')
        except Exception as e:
            lines.append(f'  Model 1 failed: {e}')
            continue

        int_p1 = {k: v for k, v in m1.pvalues.items() if ':ai_score' in k}
        if int_p1:
            r1 = np.zeros((len(int_p1), len(m1.params)))
            for i, name in enumerate(sorted(int_p1.keys())):
                r1[i, list(m1.params.index).index(name)] = 1
            f1 = m1.f_test(r1)
            f_p1 = float(np.asarray(f1.pvalue).flat[0])
        else:
            f_p1 = 1.0

        # Model with text length control
        try:
            m2 = smf.wls(f'{dv} ~ C(year) * ai_score + log_textlen',
                          data=df, weights=df['n']).fit(cov_type='HC3')
        except Exception as e:
            lines.append(f'  Model 2 failed: {e}')
            continue

        int_p2 = {k: v for k, v in m2.pvalues.items() if ':ai_score' in k}
        if int_p2:
            r2 = np.zeros((len(int_p2), len(m2.params)))
            for i, name in enumerate(sorted(int_p2.keys())):
                r2[i, list(m2.params.index).index(name)] = 1
            f2 = m2.f_test(r2)
            f_p2 = float(np.asarray(f2.pvalue).flat[0])
        else:
            f_p2 = 1.0

        lines.append(f'  Without text length control: Joint F p = {f_p1:.4f}, R² = {m1.rsquared:.4f}')
        lines.append(f'  With text length control:    Joint F p = {f_p2:.4f}, R² = {m2.rsquared:.4f}')

        # Post-2023 interaction coefficients (from model with controls)
        lines.append(f'  Post-2023 interactions (with log_textlen control):')
        for k in sorted(int_p2.keys()):
            if any(str(y) in k for y in [2023, 2024, 2025]):
                yr = k.split('[T.')[1].split(']')[0]
                lines.append(f'    {yr}: coef={m2.params[k]:+.6f}, '
                             f'SE={m2.bse[k]:.6f}, p={m2.pvalues[k]:.4f}')

        # ai_score main effect
        if 'ai_score' in m2.params:
            lines.append(f'  ai_score main: coef={m2.params["ai_score"]:+.6f}, '
                         f'p={m2.pvalues["ai_score"]:.4f}')
        if 'log_textlen' in m2.params:
            lines.append(f'  log_textlen:   coef={m2.params["log_textlen"]:+.6f}, '
                         f'p={m2.pvalues["log_textlen"]:.4f}')

        print(f'  {cat}: no ctrl p={f_p1:.4f}, +textlen p={f_p2:.4f}')

        results[cat] = {
            'no_ctrl_p': f_p1,
            'with_ctrl_p': f_p2,
            'model': m2,
        }

    # Also run on share_has (extensive margin: does the posting mention ANY
    # keyword in this category?)
    lines.append(f'\n\n{"=" * 70}')
    lines.append('Extensive Margin: share of postings with ≥1 keyword')
    lines.append('=' * 70)

    for cat in CATEGORIES:
        dv = f'share_has_{cat}'
        try:
            m = smf.wls(f'{dv} ~ C(year) * ai_score + log_textlen',
                        data=df, weights=df['n']).fit(cov_type='HC3')
        except Exception as e:
            lines.append(f'  {cat}: failed — {e}')
            continue

        int_p = {k: v for k, v in m.pvalues.items() if ':ai_score' in k}
        if int_p:
            r = np.zeros((len(int_p), len(m.params)))
            for i, name in enumerate(sorted(int_p.keys())):
                r[i, list(m.params.index).index(name)] = 1
            ft = m.f_test(r)
            f_p = float(np.asarray(ft.pvalue).flat[0])
        else:
            f_p = 1.0

        lines.append(f'  {cat}: Joint F p = {f_p:.4f}')

        # Post-2023
        for k in sorted(int_p.keys()):
            if any(str(y) in k for y in [2023, 2024, 2025]):
                yr = k.split('[T.')[1].split(']')[0]
                lines.append(f'    {yr}: coef={m.params[k]:+.6f}, p={m.pvalues[k]:.4f}')

        print(f'  {cat} (extensive): p={f_p:.4f}')
        results[f'{cat}_ext'] = f_p

    result_text = '\n'.join(lines)
    with open(OUT / 'task_regressions.txt', 'w') as f:
        f.write(result_text)

    return results


# ══════════════════════════════════════════════════════════════════
# Step 5: Substitution vs Augmentation exposure decomposition
# ══════════════════════════════════════════════════════════════════

def exposure_decomposition(df):
    """
    Decompose AI exposure into substitution and augmentation components
    based on actual task content of each occupation.

    substitution_exposure = ai_score × RC_task_share (of the occupation)
    augmentation_exposure = ai_score × NRA_task_share (of the occupation)

    These are occupation-level, time-invariant measures based on baseline
    (pre-2023) task profiles.
    """
    print('\n[Step 5] Substitution vs Augmentation Exposure Decomposition')
    print('=' * 70)

    # Compute baseline (pre-2023) task profile per ISCO
    pre = df[(df.year >= 2016) & (df.year <= 2022)].copy()
    # Remove small/anomalous years
    pre = pre[~pre.year.isin([2019, 2020])]

    baseline = pre.groupby('isco').apply(
        lambda g: pd.Series({
            'base_RC': np.average(g['mean_RC_count'], weights=g['n']),
            'base_NRA': np.average(g['mean_NRA_count'], weights=g['n']),
            'base_NRI': np.average(g['mean_NRI_count'], weights=g['n']),
            'base_RM': np.average(g['mean_RM_count'], weights=g['n']),
            'base_n': g['n'].sum(),
        })
    ).reset_index()

    # Normalize to shares
    for cat in CATEGORIES:
        baseline[f'base_{cat}_share'] = baseline[f'base_{cat}']
    total = baseline[['base_RC', 'base_NRA', 'base_NRI', 'base_RM']].sum(axis=1)
    for cat in CATEGORIES:
        baseline[f'base_{cat}_share'] = np.where(
            total > 0,
            baseline[f'base_{cat}'] / total,
            0.25  # uniform if no keywords found
        )

    # Merge baseline task profile into full panel
    df = df.merge(baseline[['isco'] + [f'base_{cat}_share' for cat in CATEGORIES]],
                   on='isco', how='left')

    # Construct decomposed exposure measures
    df['subst_exposure'] = df['ai_score'] * df['base_RC_share']
    df['aug_exposure'] = df['ai_score'] * df['base_NRA_share']

    lines = []
    lines.append('Substitution vs Augmentation Exposure Decomposition')
    lines.append('=' * 70)
    lines.append(f'subst_exposure = ai_score × baseline_RC_share')
    lines.append(f'aug_exposure   = ai_score × baseline_NRA_share')
    lines.append(f'\nCorrelation matrix:')
    corr_cols = ['ai_score', 'subst_exposure', 'aug_exposure']
    corr_mat = df[corr_cols].corr()
    for c1 in corr_cols:
        vals = [f'{corr_mat.loc[c1, c2]:.3f}' for c2 in corr_cols]
        lines.append(f'  {c1:<18} ' + '  '.join(f'{v:>8}' for v in vals))

    print(f'  Correlation(ai_score, subst): {corr_mat.loc["ai_score","subst_exposure"]:.3f}')
    print(f'  Correlation(ai_score, aug):   {corr_mat.loc["ai_score","aug_exposure"]:.3f}')
    print(f'  Correlation(subst, aug):      {corr_mat.loc["subst_exposure","aug_exposure"]:.3f}')

    # Run regressions with decomposed exposure
    lines.append(f'\n\nRegressions with decomposed exposure:')
    lines.append('DV ~ C(year) × subst_exposure + C(year) × aug_exposure + log_textlen')
    lines.append('=' * 70)

    results = {}
    for cat in CATEGORIES:
        dv = f'mean_{cat}_count'
        lines.append(f'\n--- {cat} ---')

        try:
            m = smf.wls(f'{dv} ~ C(year) * subst_exposure + C(year) * aug_exposure + log_textlen',
                        data=df, weights=df['n']).fit(cov_type='HC3')
        except Exception as e:
            lines.append(f'  Failed: {e}')
            continue

        # Joint F-test on substitution × year interactions
        subst_int = {k: v for k, v in m.pvalues.items() if ':subst_exposure' in k}
        if subst_int:
            r_s = np.zeros((len(subst_int), len(m.params)))
            for i, name in enumerate(sorted(subst_int.keys())):
                r_s[i, list(m.params.index).index(name)] = 1
            fs = m.f_test(r_s)
            f_ps = float(np.asarray(fs.pvalue).flat[0])
        else:
            f_ps = 1.0

        # Joint F-test on augmentation × year interactions
        aug_int = {k: v for k, v in m.pvalues.items() if ':aug_exposure' in k}
        if aug_int:
            r_a = np.zeros((len(aug_int), len(m.params)))
            for i, name in enumerate(sorted(aug_int.keys())):
                r_a[i, list(m.params.index).index(name)] = 1
            fa = m.f_test(r_a)
            f_pa = float(np.asarray(fa.pvalue).flat[0])
        else:
            f_pa = 1.0

        lines.append(f'  Substitution × year: Joint F p = {f_ps:.4f}')
        lines.append(f'  Augmentation × year: Joint F p = {f_pa:.4f}')

        # Post-2023 coefficients
        for prefix, label in [(':subst_exposure', 'SUBST'), (':aug_exposure', 'AUG')]:
            for k in sorted(m.pvalues.keys()):
                if prefix in k and any(str(y) in k for y in [2023, 2024, 2025]):
                    yr = k.split('[T.')[1].split(']')[0]
                    lines.append(f'    {label} {yr}: coef={m.params[k]:+.6f}, p={m.pvalues[k]:.4f}')

        print(f'  {cat}: subst p={f_ps:.4f}, aug p={f_pa:.4f}')
        results[cat] = {'subst_p': f_ps, 'aug_p': f_pa}

    result_text = '\n'.join(lines)
    with open(OUT / 'exposure_decomposition.txt', 'w') as f:
        f.write(result_text)

    return df, results


# ══════════════════════════════════════════════════════════════════
# Step 6: Visualization
# ══════════════════════════════════════════════════════════════════

def visualize(df):
    """Generate diagnostic and result plots."""
    print('\n[Step 6] Generating visualizations ...')

    # Plot 1: Task keyword trends by year
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    colors = {'RC': '#e74c3c', 'NRA': '#2563EB', 'NRI': '#2ecc71', 'RM': '#e67e22'}
    labels = {'RC': 'RC (Routine Cognitive)',
              'NRA': 'NRA (Non-routine Analytical)',
              'NRI': 'NRI (Non-routine Interactive)',
              'RM': 'RM (Routine Manual)'}

    for ax, cat in zip(axes.flat, CATEGORIES):
        yearly = df.groupby('year').apply(
            lambda g: pd.Series({
                'mean': np.average(g[f'mean_{cat}_count'], weights=g['n']),
                'n': g['n'].sum(),
            })
        )
        ax.bar(yearly.index, yearly['mean'], color=colors[cat], alpha=0.8)
        ax.set_title(labels[cat])
        ax.set_xlabel('Year')
        ax.set_ylabel('Mean keyword count')

    plt.suptitle('Task Keyword Intensity by Year', fontsize=14)
    plt.tight_layout()
    plt.savefig(OUT / 'task_trends_by_year.png', dpi=150)
    plt.close()
    print('  Saved task_trends_by_year.png')

    # Plot 2: Task intensity by AI score quartile × year (line plot)
    if 'ai_q' in df.columns:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        for ax, cat in zip(axes.flat, CATEGORIES):
            for q in df['ai_q'].cat.categories:
                sub = df[df.ai_q == q]
                yearly = sub.groupby('year').apply(
                    lambda g: np.average(g[f'mean_{cat}_count'], weights=g['n'])
                )
                ax.plot(yearly.index, yearly.values, 'o-', label=str(q),
                        markersize=4)
            ax.set_title(labels[cat])
            ax.set_xlabel('Year')
            ax.set_ylabel('Mean keyword count')
            ax.legend(fontsize=8)

        plt.suptitle('Task Intensity by AI Exposure Quartile × Year', fontsize=14)
        plt.tight_layout()
        plt.savefig(OUT / 'task_by_ai_quartile.png', dpi=150)
        plt.close()
        print('  Saved task_by_ai_quartile.png')

    # Plot 3: Task profile by AI exposure level (radar/bar chart)
    fig, ax = plt.subplots(figsize=(10, 6))
    if 'ai_q' in df.columns:
        q_data = []
        for q in df['ai_q'].cat.categories:
            sub = df[df.ai_q == q]
            vals = [np.average(sub[f'mean_{cat}_count'], weights=sub['n'])
                    for cat in CATEGORIES]
            q_data.append(vals)

        x = np.arange(len(CATEGORIES))
        width = 0.2
        for i, (q, vals) in enumerate(zip(df['ai_q'].cat.categories, q_data)):
            ax.bar(x + i * width, vals, width, label=str(q), alpha=0.8)

        ax.set_xticks(x + 1.5 * width)
        ax.set_xticklabels(CATEGORIES)
        ax.set_ylabel('Mean keyword count')
        ax.set_title('Task Profile by AI Exposure Quartile')
        ax.legend()
        plt.tight_layout()
        plt.savefig(OUT / 'task_profile_by_ai.png', dpi=150)
        plt.close()
        print('  Saved task_profile_by_ai.png')

    # Plot 4: Pre/Post comparison for high vs low AI
    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    df_plot = df[~df.year.isin([2019, 2020, 2025])].copy()
    df_plot['period'] = np.where(df_plot['year'] >= 2023, 'Post-2023', 'Pre-2023')
    median_ai = df_plot['ai_score'].median()
    df_plot['ai_group'] = np.where(df_plot['ai_score'] >= median_ai, 'High AI', 'Low AI')

    for ax, cat in zip(axes, CATEGORIES):
        for ai_grp, color in [('Low AI', '#3498db'), ('High AI', '#e74c3c')]:
            for period, ls in [('Pre-2023', '--'), ('Post-2023', '-')]:
                sub = df_plot[(df_plot.ai_group == ai_grp) & (df_plot.period == period)]
                if len(sub) == 0:
                    continue
                val = np.average(sub[f'mean_{cat}_count'], weights=sub['n'])
                label = f'{ai_grp} {period}'
                ax.barh(label, val, color=color,
                        alpha=0.6 if period == 'Pre-2023' else 0.9)

        ax.set_title(cat)
        ax.set_xlabel('Mean keyword count')

    plt.suptitle('Task Intensity: Pre vs Post-2023, High vs Low AI Exposure', fontsize=13)
    plt.tight_layout()
    plt.savefig(OUT / 'task_pre_post_comparison.png', dpi=150)
    plt.close()
    print('  Saved task_pre_post_comparison.png')


# ══════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    t_start = time.time()

    print('=' * 70)
    print('EXPERIMENT: Atalay-style Task Keyword Extraction')
    print('  Outcome: task-level intensity (keyword counts)')
    print('  Exposure: ai_score + substitution/augmentation decomposition')
    print('=' * 70)

    # Step 1
    lookup = build_isco_lookup()

    # Step 2
    df = scan_windows(lookup)

    # Save cell-level data
    df.to_csv(OUT / 'isco_year_task_cells.csv', index=False)
    print(f'\n  Saved {len(df):,} cells to isco_year_task_cells.csv')

    # Step 3
    df = descriptive_stats(df)

    # Step 4
    task_results = run_task_regressions(df)

    # Step 5
    df, decomp_results = exposure_decomposition(df)

    # Step 6
    visualize(df)

    elapsed = time.time() - t_start
    print(f'\n{"=" * 70}')
    print(f'DONE in {elapsed/60:.1f} min')
    print(f'\nAll results saved to {OUT}')
    for f in sorted(OUT.glob('*')):
        print(f'  {f.name}')
