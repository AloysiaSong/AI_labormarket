#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
individual_twfe.py

Individual-level TWFE regression on full ~7.8M records.
Uses Frisch-Waugh-Lovell theorem to absorb ISCO 4-digit FE via demeaning,
then runs OLS with cluster-robust SEs at ISCO level (304 clusters).

Specifications:
  Model A (continuous): entropy ~ ISCO_FE + year_FE + year×ai_score
  Model B (binary DID):  entropy ~ ISCO_FE + year_FE + Treated×Post

Input:  data/Heterogeneity/master_with_ai_exposure_v2.csv
Output: output/individual_twfe/
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

BASE = Path('/Users/yu/code/code2601/TY')
MASTER = BASE / 'data/Heterogeneity/master_with_ai_exposure_v2.csv'
OUT = BASE / 'output/individual_twfe'
OUT.mkdir(parents=True, exist_ok=True)

POST_YEAR = 2023
REF_YEAR = 2016  # omitted year dummy


def load_data():
    """Read full master CSV, keep only needed columns."""
    print('[1] Loading full dataset ...')
    t0 = time.time()
    cols = ['year', 'entropy_score', 'isco08_4digit', 'ai_mean_score',
            'ai_exposure_gradient']
    chunks = []
    for chunk in pd.read_csv(MASTER, usecols=cols, chunksize=500_000):
        chunk = chunk.dropna(subset=['year', 'entropy_score', 'isco08_4digit',
                                      'ai_mean_score'])
        chunk['year'] = chunk['year'].astype(int)
        chunk['isco08_4digit'] = chunk['isco08_4digit'].astype(int)
        chunks.append(chunk)
    df = pd.concat(chunks, ignore_index=True)
    print(f'    Loaded {len(df):,} rows in {time.time()-t0:.1f}s')
    print(f'    Years: {sorted(df.year.unique())}')
    print(f'    ISCO codes: {df.isco08_4digit.nunique()}')
    return df


def demean_by_isco(df, cols):
    """FWL: subtract ISCO-group means to absorb ISCO fixed effects."""
    print('[2] Demeaning by ISCO (absorbing ISCO FE) ...')
    t0 = time.time()
    group_means = df.groupby('isco08_4digit')[cols].transform('mean')
    demeaned = df[cols] - group_means
    print(f'    Done in {time.time()-t0:.1f}s')
    return demeaned


def model_a_continuous(df):
    """
    Model A: entropy ~ ISCO_FE + year_FE + year×ai_score (continuous)
    After FWL demeaning, regress demeaned entropy on demeaned year dummies
    and demeaned (year_dummy × ai_score) interactions.
    Cluster SEs at ISCO 4-digit level.
    """
    print('\n' + '='*70)
    print('Model A: Continuous ai_score × year interactions')
    print('entropy ~ ISCO_FE + year_FE + year×ai_score')
    print('Cluster-robust SEs at ISCO 4-digit level')
    print('='*70)

    years = sorted(df.year.unique())
    years_excl_ref = [y for y in years if y != REF_YEAR]

    # Create year dummies and interactions
    for y in years_excl_ref:
        df[f'yr_{y}'] = (df['year'] == y).astype(float)
        df[f'yr_{y}_x_ai'] = df[f'yr_{y}'] * df['ai_mean_score']

    yr_cols = [f'yr_{y}' for y in years_excl_ref]
    int_cols = [f'yr_{y}_x_ai' for y in years_excl_ref]
    all_cols = yr_cols + int_cols + ['entropy_score']

    # FWL: demean by ISCO
    demeaned = demean_by_isco(df, all_cols)

    y = demeaned['entropy_score'].values
    X = sm.add_constant(demeaned[yr_cols + int_cols].values)
    col_names = ['const'] + yr_cols + int_cols

    print('[3] Running OLS with ISCO-clustered SEs ...')
    t0 = time.time()
    model = sm.OLS(y, X)
    # Cluster at ISCO level
    res = model.fit(cov_type='cluster',
                    cov_kwds={'groups': df['isco08_4digit'].values})
    print(f'    Done in {time.time()-t0:.1f}s')

    # Pretty output
    lines = []
    lines.append(f'N observations = {len(df):,}')
    lines.append(f'N ISCO clusters = {df.isco08_4digit.nunique()}')
    lines.append(f'R² (within) = {res.rsquared:.6f}')
    lines.append(f'Adj R² (within) = {res.rsquared_adj:.6f}')
    lines.append('')
    lines.append(f'{"Variable":<25} {"Coef":>10} {"SE":>10} {"p":>8}')
    lines.append('-' * 55)
    for i, name in enumerate(col_names):
        lines.append(f'{name:<25} {res.params[i]:>10.6f} {res.bse[i]:>10.6f} {res.pvalues[i]:>8.4f}')

    # Joint F-test on all interaction terms
    n_int = len(int_cols)
    n_yr = len(yr_cols)
    # Interaction terms are the last n_int columns (indices n_yr+1 to n_yr+n_int, +1 for const)
    R = np.zeros((n_int, len(col_names)))
    for k in range(n_int):
        R[k, 1 + n_yr + k] = 1  # +1 for const, +n_yr to skip year dummies
    f_test = res.f_test(R)
    lines.append('')
    lines.append(f'Joint F-test (all year×ai_score = 0):')
    lines.append(f'  F({n_int}, {int(f_test.df_denom)}) = {float(f_test.fvalue):.4f}, p = {float(f_test.pvalue):.4f}')

    result_text = '\n'.join(lines)
    print(result_text)

    with open(OUT / 'model_a_continuous.txt', 'w') as f:
        f.write('Model A: entropy ~ ISCO_FE + year_FE + year×ai_score\n')
        f.write('Individual-level, ISCO-clustered SEs\n')
        f.write('='*70 + '\n')
        f.write(result_text)

    # Cleanup temp columns
    for y_ in years_excl_ref:
        df.drop(columns=[f'yr_{y_}', f'yr_{y_}_x_ai'], inplace=True)

    return res


def model_b_binary_did(df):
    """
    Model B: Binary DID — entropy ~ ISCO_FE + year_FE + Treated×Post
    Treated = G3/G4, Control = Minimal/NotExposed (drops G1/G2)
    Post = year >= 2023
    Cluster SEs at ISCO 4-digit level.
    """
    print('\n' + '='*70)
    print('Model B: Binary DID (Treated×Post)')
    print('Treated = G3+G4, Control = Minimal+NotExposed')
    print(f'Post = year >= {POST_YEAR}')
    print('Cluster-robust SEs at ISCO 4-digit level')
    print('='*70)

    HIGH = {'Gradient 3', 'Gradient 4'}
    LOW = {'Minimal Exposure', 'Not Exposed'}

    mask = df['ai_exposure_gradient'].isin(HIGH | LOW)
    dfsub = df[mask].copy()
    print(f'    Subsample: {len(dfsub):,} rows ({mask.sum():,} treated+control)')

    dfsub['treated'] = dfsub['ai_exposure_gradient'].isin(HIGH).astype(float)
    dfsub['post'] = (dfsub['year'] >= POST_YEAR).astype(float)
    dfsub['did'] = dfsub['treated'] * dfsub['post']

    years = sorted(dfsub.year.unique())
    years_excl_ref = [y for y in years if y != REF_YEAR]
    for y in years_excl_ref:
        dfsub[f'yr_{y}'] = (dfsub['year'] == y).astype(float)

    yr_cols = [f'yr_{y}' for y in years_excl_ref]
    all_cols = yr_cols + ['treated', 'did', 'entropy_score']

    # FWL: demean by ISCO
    demeaned = demean_by_isco(dfsub, all_cols)

    y = demeaned['entropy_score'].values
    X_cols = yr_cols + ['treated', 'did']
    X = sm.add_constant(demeaned[X_cols].values)
    col_names = ['const'] + X_cols

    print('[3] Running OLS with ISCO-clustered SEs ...')
    t0 = time.time()
    model = sm.OLS(y, X)
    res = model.fit(cov_type='cluster',
                    cov_kwds={'groups': dfsub['isco08_4digit'].values})
    print(f'    Done in {time.time()-t0:.1f}s')

    lines = []
    lines.append(f'N observations = {len(dfsub):,}')
    lines.append(f'N ISCO clusters = {dfsub.isco08_4digit.nunique()}')
    lines.append(f'R² (within) = {res.rsquared:.6f}')
    lines.append('')
    lines.append(f'{"Variable":<25} {"Coef":>10} {"SE":>10} {"p":>8}')
    lines.append('-' * 55)
    for i, name in enumerate(col_names):
        lines.append(f'{name:<25} {res.params[i]:>10.6f} {res.bse[i]:>10.6f} {res.pvalues[i]:>8.4f}')

    # The DID coefficient
    did_idx = col_names.index('did')
    lines.append('')
    lines.append(f'>>> DID coefficient = {res.params[did_idx]:.6f}')
    lines.append(f'    SE = {res.bse[did_idx]:.6f}')
    lines.append(f'    p  = {res.pvalues[did_idx]:.4f}')
    lines.append(f'    95% CI = [{res.conf_int()[did_idx][0]:.6f}, {res.conf_int()[did_idx][1]:.6f}]')

    result_text = '\n'.join(lines)
    print(result_text)

    with open(OUT / 'model_b_binary_did.txt', 'w') as f:
        f.write('Model B: Binary DID — entropy ~ ISCO_FE + year_FE + Treated×Post\n')
        f.write(f'Treated=G3+G4, Control=Minimal+NotExposed, Post>=2023\n')
        f.write('Individual-level, ISCO-clustered SEs\n')
        f.write('='*70 + '\n')
        f.write(result_text)

    return res


def model_c_event_study(df):
    """
    Model C: Event study with continuous ai_score.
    entropy ~ ISCO_FE + year_FE + year_t × ai_score (same as A, but formatted
    as event study for visual comparison with the binary version)
    Also: binary event study for comparison.
    """
    print('\n' + '='*70)
    print('Model C: Binary Event Study')
    print('entropy ~ ISCO_FE + year_t × Treated')
    print('Cluster-robust SEs at ISCO 4-digit level')
    print('='*70)

    HIGH = {'Gradient 3', 'Gradient 4'}
    LOW = {'Minimal Exposure', 'Not Exposed'}

    mask = df['ai_exposure_gradient'].isin(HIGH | LOW)
    dfsub = df[mask].copy()

    dfsub['treated'] = dfsub['ai_exposure_gradient'].isin(HIGH).astype(float)

    years = sorted(dfsub.year.unique())
    ref = 2022  # last pre-treatment year
    years_excl_ref = [y for y in years if y != ref]

    for y in years_excl_ref:
        dfsub[f'yr_{y}'] = (dfsub['year'] == y).astype(float)
        dfsub[f'yr_{y}_x_tr'] = dfsub[f'yr_{y}'] * dfsub['treated']

    yr_cols = [f'yr_{y}' for y in years_excl_ref]
    int_cols = [f'yr_{y}_x_tr' for y in years_excl_ref]
    all_cols = yr_cols + ['treated'] + int_cols + ['entropy_score']

    demeaned = demean_by_isco(dfsub, all_cols)

    y_var = demeaned['entropy_score'].values
    X_cols = yr_cols + ['treated'] + int_cols
    X = sm.add_constant(demeaned[X_cols].values)
    col_names = ['const'] + X_cols

    print('[3] Running OLS with ISCO-clustered SEs ...')
    t0 = time.time()
    model = sm.OLS(y_var, X)
    res = model.fit(cov_type='cluster',
                    cov_kwds={'groups': dfsub['isco08_4digit'].values})
    print(f'    Done in {time.time()-t0:.1f}s')

    lines = []
    lines.append(f'N observations = {len(dfsub):,}')
    lines.append(f'N ISCO clusters = {dfsub.isco08_4digit.nunique()}')
    lines.append(f'Reference year = {ref}')
    lines.append('')
    lines.append('Event study coefficients (year × Treated):')
    lines.append(f'{"Year":<10} {"Coef":>10} {"SE":>10} {"p":>8} {"95% CI"}')
    lines.append('-' * 60)

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    es_years = []
    es_coefs = []
    es_ci_lo = []
    es_ci_hi = []

    for y in years_excl_ref:
        idx = col_names.index(f'yr_{y}_x_tr')
        coef = res.params[idx]
        se = res.bse[idx]
        p = res.pvalues[idx]
        ci = res.conf_int()[idx]
        lines.append(f'{y:<10} {coef:>10.6f} {se:>10.6f} {p:>8.4f} [{ci[0]:.6f}, {ci[1]:.6f}]')
        es_years.append(y)
        es_coefs.append(coef)
        es_ci_lo.append(ci[0])
        es_ci_hi.append(ci[1])

    # Add reference year (zero by construction)
    es_years.append(ref)
    es_coefs.append(0.0)
    es_ci_lo.append(0.0)
    es_ci_hi.append(0.0)

    # Sort by year
    order = np.argsort(es_years)
    es_years = [es_years[i] for i in order]
    es_coefs = [es_coefs[i] for i in order]
    es_ci_lo = [es_ci_lo[i] for i in order]
    es_ci_hi = [es_ci_hi[i] for i in order]

    result_text = '\n'.join(lines)
    print(result_text)

    with open(OUT / 'model_c_event_study.txt', 'w') as f:
        f.write('Model C: Binary Event Study\n')
        f.write(f'entropy ~ ISCO_FE + year_FE + year_t × Treated\n')
        f.write(f'Reference year = {ref}\n')
        f.write('Individual-level, ISCO-clustered SEs\n')
        f.write('='*70 + '\n')
        f.write(result_text)

    # Plot event study
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.errorbar(es_years, es_coefs,
                yerr=[np.array(es_coefs) - np.array(es_ci_lo),
                      np.array(es_ci_hi) - np.array(es_coefs)],
                fmt='o-', capsize=4, color='#2563EB', markersize=6)
    ax.axhline(0, color='grey', ls='--', lw=0.8)
    ax.axvline(2022.5, color='red', ls=':', lw=1, label='Post-treatment (2023+)')
    ax.set_xlabel('Year')
    ax.set_ylabel('Treated × Year coefficient')
    ax.set_title('Event Study: High vs Low AI Exposure (ISCO-clustered SEs)')
    ax.legend()
    plt.tight_layout()
    plt.savefig(OUT / 'event_study_isco_cluster.png', dpi=150)
    plt.close()
    print(f'    Saved event study plot')

    return res


if __name__ == '__main__':
    df = load_data()

    res_a = model_a_continuous(df)
    res_b = model_b_binary_did(df)
    res_c = model_c_event_study(df)

    print('\n' + '='*70)
    print('SUMMARY')
    print('='*70)
    print('All results saved to:', OUT)
    print('Files:')
    for f in sorted(OUT.glob('*')):
        print(f'  {f.name}')
