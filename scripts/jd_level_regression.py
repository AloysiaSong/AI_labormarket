#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
jd_level_regression.py

使用JD级ai_score进行个体层面回归分析。

核心创新：
  ai_score不再从ISCO码直接继承（仅304个唯一值），
  而是通过SBERT将每条JD的岗位名称与303个ISCO职业描述做余弦相似度，
  取top-5加权平均，得到JD级连续ai_score（数百万唯一值）。

方法论优势：
  - ai_score来自岗位名称（短文本），entropy来自JD正文（LDA），
    不同文本字段 → 切断机械性相关
  - JD级变异 → 有效样本量从304个ISCO码跃升至数百万个JD
  - 同一ISCO内不同JD可有不同ai_score（"Java实习生"≠"高级Java架构师"）

规格：
  Model 1: entropy ~ year_FE + jd_ai_score + year×jd_ai_score
           HC1 robust SEs
  Model 2: entropy ~ year_FE + jd_ai_score + year×jd_ai_score
           Cluster at ISCO 4-digit level
  Model 3: entropy ~ ISCO_FE + year_FE + year×jd_ai_score
           Cluster at ISCO 4-digit level (FWL demeaning)
  Model 4: entropy ~ ISCO_FE + year_FE + year×jd_ai_score
           + industry×year interactions (industry moderation)

输入:  data/Heterogeneity/master_with_jd_ai_score.csv
输出:  output/jd_regression/
"""

import time
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

BASE = Path('/Users/yu/code/code2601/TY')
MASTER = BASE / 'data/Heterogeneity/master_with_jd_ai_score.csv'
OUT = BASE / 'output/jd_regression'
OUT.mkdir(parents=True, exist_ok=True)

REF_YEAR = 2016


def load_data():
    """读取含jd_ai_score的master数据。"""
    print('[1] Loading master_with_jd_ai_score.csv ...')
    t0 = time.time()
    cols = ['year', 'entropy_score', 'isco08_4digit', 'ai_mean_score',
            'jd_ai_score', 'industry20_code', 'ai_exposure_gradient']
    chunks = []
    for chunk in pd.read_csv(MASTER, usecols=cols, chunksize=500_000):
        chunk = chunk.dropna(subset=['year', 'entropy_score', 'jd_ai_score',
                                      'isco08_4digit'])
        chunk['year'] = chunk['year'].astype(int)
        chunk['isco08_4digit'] = chunk['isco08_4digit'].astype(int)
        chunks.append(chunk)
    df = pd.concat(chunks, ignore_index=True)
    print(f'    Loaded {len(df):,} rows in {time.time()-t0:.1f}s')
    print(f'    Years: {sorted(df.year.unique())}')
    print(f'    ISCO codes: {df.isco08_4digit.nunique()}')
    print(f'    jd_ai_score unique values: {df.jd_ai_score.nunique():,}')
    print(f'    jd_ai_score range: [{df.jd_ai_score.min():.4f}, {df.jd_ai_score.max():.4f}]')
    print(f'    jd_ai_score mean={df.jd_ai_score.mean():.4f}, std={df.jd_ai_score.std():.4f}')

    # 对比isco级 vs jd级
    print(f'\n    Comparison: ISCO-level ai_mean_score vs JD-level jd_ai_score')
    print(f'    ai_mean_score unique: {df.ai_mean_score.nunique()}')
    print(f'    jd_ai_score unique:   {df.jd_ai_score.nunique():,}')
    corr = df[['ai_mean_score', 'jd_ai_score']].corr().iloc[0, 1]
    print(f'    Correlation: {corr:.4f}')
    return df


def model1_hc1(df):
    """Model 1: OLS with HC1 robust SEs (no clustering)."""
    print('\n' + '=' * 70)
    print('Model 1: entropy ~ year_FE + jd_ai_score + year×jd_ai_score')
    print('HC1 robust SEs (no clustering)')
    print('=' * 70)

    years = sorted(df.year.unique())
    years_excl = [y for y in years if y != REF_YEAR]

    # Create variables
    X_parts = []
    col_names = []
    for y in years_excl:
        X_parts.append((df['year'] == y).astype(float).values)
        col_names.append(f'yr_{y}')
    for y in years_excl:
        X_parts.append(((df['year'] == y).astype(float) * df['jd_ai_score']).values)
        col_names.append(f'yr_{y}_x_ai')
    X_parts.append(df['jd_ai_score'].values)
    col_names.append('jd_ai_score')

    X = np.column_stack(X_parts)
    X = sm.add_constant(X)
    col_names = ['const'] + col_names
    y = df['entropy_score'].values

    print(f'    X shape: {X.shape}')
    t0 = time.time()
    model = sm.OLS(y, X)
    res = model.fit(cov_type='HC1')
    print(f'    Done in {time.time()-t0:.1f}s')

    return _report(res, col_names, df, 'model1_hc1.txt',
                   'Model 1: HC1 robust SEs', years_excl)


def model2_isco_cluster(df):
    """Model 2: OLS with ISCO-clustered SEs."""
    print('\n' + '=' * 70)
    print('Model 2: entropy ~ year_FE + jd_ai_score + year×jd_ai_score')
    print('Cluster-robust SEs at ISCO 4-digit level')
    print('=' * 70)

    years = sorted(df.year.unique())
    years_excl = [y for y in years if y != REF_YEAR]

    X_parts = []
    col_names = []
    for y in years_excl:
        X_parts.append((df['year'] == y).astype(float).values)
        col_names.append(f'yr_{y}')
    for y in years_excl:
        X_parts.append(((df['year'] == y).astype(float) * df['jd_ai_score']).values)
        col_names.append(f'yr_{y}_x_ai')
    X_parts.append(df['jd_ai_score'].values)
    col_names.append('jd_ai_score')

    X = np.column_stack(X_parts)
    X = sm.add_constant(X)
    col_names = ['const'] + col_names
    y = df['entropy_score'].values

    t0 = time.time()
    model = sm.OLS(y, X)
    res = model.fit(cov_type='cluster',
                    cov_kwds={'groups': df['isco08_4digit'].values})
    print(f'    Done in {time.time()-t0:.1f}s')

    return _report(res, col_names, df, 'model2_isco_cluster.txt',
                   'Model 2: ISCO-clustered SEs', years_excl)


def model3_isco_fe(df):
    """Model 3: ISCO FE (FWL demeaning) + ISCO-clustered SEs."""
    print('\n' + '=' * 70)
    print('Model 3: entropy ~ ISCO_FE + year_FE + year×jd_ai_score')
    print('FWL demeaning for ISCO FE, cluster at ISCO level')
    print('=' * 70)

    years = sorted(df.year.unique())
    years_excl = [y for y in years if y != REF_YEAR]

    # Build raw columns
    raw_cols = {}
    raw_cols['entropy_score'] = df['entropy_score'].values.copy()
    for y in years_excl:
        raw_cols[f'yr_{y}'] = (df['year'] == y).astype(float).values
        raw_cols[f'yr_{y}_x_ai'] = raw_cols[f'yr_{y}'] * df['jd_ai_score'].values
    raw_cols['jd_ai_score'] = df['jd_ai_score'].values.copy()

    # FWL: demean by ISCO
    print('    Demeaning by ISCO ...')
    t0 = time.time()
    isco = df['isco08_4digit'].values
    unique_isco = np.unique(isco)
    demeaned = {k: v.copy() for k, v in raw_cols.items()}

    for code in unique_isco:
        mask = isco == code
        for k in demeaned:
            demeaned[k][mask] -= demeaned[k][mask].mean()
    print(f'    Demeaned in {time.time()-t0:.1f}s')

    yr_cols = [f'yr_{y}' for y in years_excl]
    int_cols = [f'yr_{y}_x_ai' for y in years_excl]
    all_x_cols = yr_cols + int_cols + ['jd_ai_score']
    col_names = ['const'] + all_x_cols

    X = np.column_stack([demeaned[k] for k in all_x_cols])
    X = sm.add_constant(X)
    y = demeaned['entropy_score']

    t0 = time.time()
    model = sm.OLS(y, X)
    res = model.fit(cov_type='cluster',
                    cov_kwds={'groups': isco})
    print(f'    Done in {time.time()-t0:.1f}s')

    return _report(res, col_names, df, 'model3_isco_fe.txt',
                   'Model 3: ISCO FE + ISCO-clustered SEs', years_excl)


def _report(res, col_names, df, filename, title, years_excl):
    """格式化回归结果并保存。"""
    lines = []
    lines.append(title)
    lines.append('=' * 70)
    lines.append(f'N observations = {int(res.nobs):,}')
    lines.append(f'N ISCO clusters = {df.isco08_4digit.nunique()}')
    lines.append(f'R² = {res.rsquared:.6f}')
    lines.append(f'Adj R² = {res.rsquared_adj:.6f}')
    lines.append('')
    lines.append(f'{"Variable":<25} {"Coef":>12} {"SE":>12} {"p":>8}')
    lines.append('-' * 60)
    for i, name in enumerate(col_names):
        lines.append(f'{name:<25} {res.params[i]:>12.6f} '
                      f'{res.bse[i]:>12.6f} {res.pvalues[i]:>8.4f}')

    # Joint F-test on interaction terms
    int_cols = [f'yr_{y}_x_ai' for y in years_excl]
    n_int = len(int_cols)
    int_indices = [col_names.index(c) for c in int_cols]

    R = np.zeros((n_int, len(col_names)))
    for k, idx in enumerate(int_indices):
        R[k, idx] = 1
    f_test = res.f_test(R)
    lines.append('')
    lines.append(f'Joint F-test (all year×jd_ai_score = 0):')
    lines.append(f'  F({n_int}, {int(f_test.df_denom)}) = '
                 f'{float(f_test.fvalue):.4f}, p = {float(f_test.pvalue):.4f}')

    # Post-2023 interactions
    lines.append('')
    lines.append('Post-2023 interaction coefficients:')
    for y in years_excl:
        if y >= 2023:
            idx = col_names.index(f'yr_{y}_x_ai')
            lines.append(f'  {y}: coef={res.params[idx]:.6f}, '
                          f'SE={res.bse[idx]:.6f}, p={res.pvalues[idx]:.4f}')

    result_text = '\n'.join(lines)
    print(result_text)

    with open(OUT / filename, 'w') as f:
        f.write(result_text)

    return res


def comparison_table(df):
    """对比ISCO级 vs JD级ai_score的描述统计。"""
    print('\n' + '=' * 70)
    print('Comparison: ISCO-level vs JD-level ai_score')
    print('=' * 70)

    lines = []
    lines.append(f'{"Metric":<30} {"ISCO-level":>15} {"JD-level":>15}')
    lines.append('-' * 62)
    lines.append(f'{"Unique values":<30} {df.ai_mean_score.nunique():>15,} {df.jd_ai_score.nunique():>15,}')
    lines.append(f'{"Mean":<30} {df.ai_mean_score.mean():>15.4f} {df.jd_ai_score.mean():>15.4f}')
    lines.append(f'{"Std":<30} {df.ai_mean_score.std():>15.4f} {df.jd_ai_score.std():>15.4f}')
    lines.append(f'{"Min":<30} {df.ai_mean_score.min():>15.4f} {df.jd_ai_score.min():>15.4f}')
    lines.append(f'{"P25":<30} {df.ai_mean_score.quantile(.25):>15.4f} {df.jd_ai_score.quantile(.25):>15.4f}')
    lines.append(f'{"P50":<30} {df.ai_mean_score.quantile(.50):>15.4f} {df.jd_ai_score.quantile(.50):>15.4f}')
    lines.append(f'{"P75":<30} {df.ai_mean_score.quantile(.75):>15.4f} {df.jd_ai_score.quantile(.75):>15.4f}')
    lines.append(f'{"Max":<30} {df.ai_mean_score.max():>15.4f} {df.jd_ai_score.max():>15.4f}')
    lines.append(f'{"Correlation":<30} {df[["ai_mean_score","jd_ai_score"]].corr().iloc[0,1]:>15.4f}')

    # Within-ISCO variation
    within_std = df.groupby('isco08_4digit')['jd_ai_score'].std().mean()
    between_std = df.groupby('isco08_4digit')['jd_ai_score'].mean().std()
    lines.append('')
    lines.append(f'JD-level ai_score decomposition:')
    lines.append(f'  Between-ISCO std:  {between_std:.4f}')
    lines.append(f'  Within-ISCO std:   {within_std:.4f}')
    lines.append(f'  Within/Total ratio: {within_std/(within_std+between_std):.2%}')

    result_text = '\n'.join(lines)
    print(result_text)

    with open(OUT / 'comparison_isco_vs_jd.txt', 'w') as f:
        f.write(result_text)

    # Scatter: jd_ai_score vs ai_mean_score
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left: histogram of jd_ai_score
    ax = axes[0]
    ax.hist(df['jd_ai_score'].values, bins=100, alpha=0.7, color='#2563EB',
            density=True, label='JD-level')
    ax.hist(df['ai_mean_score'].dropna().values, bins=50, alpha=0.5,
            color='#DC2626', density=True, label='ISCO-level')
    ax.set_xlabel('AI Exposure Score')
    ax.set_ylabel('Density')
    ax.set_title('Distribution: ISCO-level vs JD-level ai_score')
    ax.legend()

    # Right: jd_ai_score vs ai_mean_score scatter (sample)
    ax = axes[1]
    sample = df.sample(min(50000, len(df)), random_state=42)
    ax.scatter(sample['ai_mean_score'], sample['jd_ai_score'],
               alpha=0.05, s=1, color='#2563EB')
    ax.plot([0.15, 0.75], [0.15, 0.75], 'r--', lw=1, label='45° line')
    ax.set_xlabel('ISCO-level ai_mean_score')
    ax.set_ylabel('JD-level jd_ai_score')
    ax.set_title(f'JD vs ISCO ai_score (r={df[["ai_mean_score","jd_ai_score"]].corr().iloc[0,1]:.3f})')
    ax.legend()

    plt.tight_layout()
    plt.savefig(OUT / 'isco_vs_jd_ai_score.png', dpi=150)
    plt.close()
    print(f'    Saved comparison plot')


if __name__ == '__main__':
    df = load_data()
    comparison_table(df)
    res1 = model1_hc1(df)
    res2 = model2_isco_cluster(df)
    res3 = model3_isco_fe(df)

    print('\n' + '=' * 70)
    print('ALL RESULTS SAVED TO:', OUT)
    for f in sorted(OUT.glob('*')):
        print(f'  {f.name}')
