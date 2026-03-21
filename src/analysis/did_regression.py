#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
did_regression.py

Difference-in-Differences analysis of AI-driven skill comprehensiveness:
  Outcome:   LDA entropy (skill comprehensiveness index, 0–1)
  Treatment: High AI exposure occupation (Gradient 3 or 4 per ILO WP140)
  Control:   Low AI exposure (Minimal Exposure or Not Exposed)
  Post:      year >= 2023 (Chinese LLM shock — ChatGPT + Kimi/Baidu ERNIE)

Two specifications:
  1. Cell-level aggregate DID (WLS on year × gradient-group means) — fast
  2. Individual-level TWFE OLS (year FE + exposure-group FE via linearmodels) — correct SEs

Event study:
  Interaction of year dummies with Treated dummy, reference year = 2022.
  Checks parallel pre-trends and visualises post-2023 divergence.

Input:  data/Heterogeneity/master_with_ai_exposure.csv
Output: data/Heterogeneity/did_results/
  did_cell_ols.txt          — aggregate WLS results
  did_indiv_twfe.txt        — individual TWFE results (main spec)
  event_study_coefs.csv     — year-specific β_t coefficients
  event_study_plot.png      — event study plot

Note: 2019 and 2020 have small/anomalous sample sizes (data coverage issues)
and are excluded from the main specification. They are noted in the text.
"""

import csv
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path('/Users/yu/code/code2601/TY')
MASTER_CSV = BASE / 'data/Heterogeneity/master_with_ai_exposure.csv'
WORD_ENT_CSV = BASE / 'data/Heterogeneity/word_entropy_by_id.csv'
OUT_DIR = BASE / 'data/Heterogeneity/did_results'

# ── Design choices ────────────────────────────────────────────────────────────
POST_YEAR = 2023          # treatment begins: Chinese LLMs widely deployed
EXCL_YEARS = {2019, 2020} # anomalous coverage; sensitivity check includes them
HIGH_EXPOSURE = {'Gradient 3', 'Gradient 4'}
LOW_EXPOSURE  = {'Minimal Exposure', 'Not Exposed'}
REF_YEAR = 2022           # event-study reference (last pre-treatment year)
# ─────────────────────────────────────────────────────────────────────────────


def load_data() -> pd.DataFrame:
    print('[1] Loading master_with_ai_exposure.csv ...')
    rows = []
    skipped = 0
    with MASTER_CSV.open(encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for row in reader:
            g = row.get('ai_exposure_gradient', '').strip()
            if g not in HIGH_EXPOSURE and g not in LOW_EXPOSURE:
                skipped += 1
                continue
            try:
                year = int(row['year'])
                entropy = float(row['entropy_score'])
            except (ValueError, KeyError):
                skipped += 1
                continue
            if year in EXCL_YEARS:
                skipped += 1
                continue

            tc = row.get('token_count', '')
            r = {
                'id':        int(row['id']),
                'year':      year,
                'gradient':  g,
                'entropy':   entropy,
                'hhi':       float(row.get('hhi_score', 0) or 0),
                'post':      int(year >= POST_YEAR),
                'treated':   int(g in HIGH_EXPOSURE),
                'industry':  row.get('industry20_code', '').strip() or 'unknown',
                'log_tokens': np.log(int(tc)) if tc.isdigit() and int(tc) > 0 else np.nan,
            }
            rows.append(r)

    df = pd.DataFrame(rows)
    print(f'  Kept:    {len(df):>10,}  (dropped {skipped:,})')
    print(f'  Years:   {sorted(df.year.unique())}')
    print(f'  Treated: {df.treated.mean():.3f}')

    # Optionally merge word entropy (unique_tokens) if available
    if WORD_ENT_CSV.exists():
        print('  Merging word_entropy_by_id.csv ...')
        we = pd.read_csv(WORD_ENT_CSV, usecols=['id', 'unique_tokens', 'token_count'],
                         dtype={'id': int})
        we['log_unique_tokens'] = np.log(we['unique_tokens'].clip(lower=1))
        df = df.merge(we[['id', 'unique_tokens', 'log_unique_tokens']], on='id', how='left')
        print(f'  Word entropy merge: {df["unique_tokens"].notna().sum():,} matched')
    else:
        df['unique_tokens'] = np.nan
        df['log_unique_tokens'] = np.nan

    return df


def aggregate_cells(df: pd.DataFrame, outcome: str = 'entropy') -> pd.DataFrame:
    """Collapse to (year, gradient) cells for WLS."""
    agg_dict = {'n': (outcome, 'size'), f'mean_{outcome}': (outcome, 'mean')}
    # include mean_log_tokens if available
    if 'log_tokens' in df.columns and df['log_tokens'].notna().any():
        agg_dict['mean_log_tokens'] = ('log_tokens', 'mean')
    agg = (df.groupby(['year', 'gradient', 'post', 'treated'])
             .agg(**agg_dict)
             .reset_index())
    agg = agg.rename(columns={f'mean_{outcome}': 'mean_outcome'})
    return agg


# ── Spec 1: Cell-level aggregate WLS ─────────────────────────────────────────
def run_cell_did(cells: pd.DataFrame, label: str = 'entropy',
                 add_token_ctrl: bool = False) -> str:
    """WLS on cell means: outcome ~ did [+ mean_log_tokens] + year_FE + group_FE, wt=n.

    Note: post and treated are collinear with year/gradient FEs respectively,
    so we include only did = post*treated as the identifying interaction.
    """
    import statsmodels.formula.api as smf

    data = cells.copy()
    data['did'] = data['post'] * data['treated']
    rhs = 'did + C(year) + C(gradient)'
    if add_token_ctrl and 'mean_log_tokens' in data.columns and data['mean_log_tokens'].notna().all():
        rhs += ' + mean_log_tokens'
    formula = f'mean_outcome ~ {rhs}'
    model = smf.wls(formula, data=data, weights=data['n'])
    res = model.fit(cov_type='HC3')

    ctrl_note = ' + log(token_count)' if add_token_ctrl else ''
    lines = [
        f'=== Cell-level Aggregate DID (WLS) — outcome: {label}{ctrl_note} ===',
        f'Spec: {label} ~ did{ctrl_note} + year_FE + gradient_FE, weighted by cell size',
        f'N cells: {len(cells)}',
        f'Total obs (sum of weights): {cells.n.sum():,}',
        '',
        res.summary().as_text(),
    ]
    return '\n'.join(lines)


# ── Spec 2: Individual-level TWFE ────────────────────────────────────────────
def run_individual_twfe(df: pd.DataFrame, sample_n: int = 0) -> str:
    """
    Two-way FE regression (year FE + gradient FE) on individual postings.
    Uses linearmodels.iv.absorbing.AbsorbingLS for large N.
    If sample_n > 0, runs on a random sample to reduce memory.
    """
    try:
        from linearmodels.iv.absorbing import AbsorbingLS
    except ImportError:
        return '(linearmodels not available — skipping individual TWFE)'

    work = df.copy()
    if sample_n > 0 and len(work) > sample_n:
        work = work.sample(sample_n, random_state=42)
        print(f'  [TWFE] Using random sample of {sample_n:,} obs')

    work['did'] = work['post'] * work['treated']
    # Absorb year FE and gradient group FE
    # Note: only include 'did' in exog; post and treated are collinear with the FEs
    work['year_cat'] = pd.Categorical(work['year'])
    work['grad_cat'] = pd.Categorical(work['gradient'])

    endog  = work[['entropy']]
    exog   = work[['did']]           # only the DID interaction term
    absorb = work[['year_cat', 'grad_cat']]

    mod = AbsorbingLS(endog, exog, absorb=absorb)
    # Cluster by year (8 clusters); gradient has too few (4) for clustering
    res = mod.fit(cov_type='clustered', clusters=work['year'].values)

    lines = [
        '=== Individual-level TWFE (AbsorbingLS, clustered by gradient group) ===',
        f'N obs: {len(work):,}',
        '',
        str(res.summary),
    ]
    return '\n'.join(lines)


# ── Event study ───────────────────────────────────────────────────────────────
def run_event_study(cells: pd.DataFrame) -> pd.DataFrame:
    """
    Interacts year dummies with Treated, reference = REF_YEAR.
    Returns DataFrame of (year, coef, ci_low, ci_high).
    Uses cell-level WLS for speed.
    """
    import statsmodels.formula.api as smf

    years = sorted(cells['year'].unique())
    # Create year-relative dummies × treated (excluding reference year)
    data = cells.copy()
    for y in years:
        if y == REF_YEAR:
            continue
        col = f'yr{y}'
        data[col] = (data['year'] == y).astype(int) * data['treated']

    yr_terms = ' + '.join(f'yr{y}' for y in years if y != REF_YEAR)
    formula = f'mean_outcome ~ {yr_terms} + C(year) + C(gradient)'
    model = smf.wls(formula, data=data, weights=data['n'])
    res = model.fit(cov_type='HC3')

    rows = []
    for y in years:
        if y == REF_YEAR:
            rows.append({'year': y, 'coef': 0.0, 'ci_low': 0.0, 'ci_high': 0.0,
                         'se': 0.0, 'pvalue': np.nan})
            continue
        col = f'yr{y}'
        coef = res.params.get(col, np.nan)
        se   = res.bse.get(col, np.nan)
        rows.append({
            'year':    y,
            'coef':    coef,
            'ci_low':  coef - 1.96 * se,
            'ci_high': coef + 1.96 * se,
            'se':      se,
            'pvalue':  res.pvalues.get(col, np.nan),
        })

    return pd.DataFrame(rows)


# ── Plot event study ──────────────────────────────────────────────────────────
def plot_event_study(es: pd.DataFrame, out_path: Path):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print('  (matplotlib not available — skipping plot)')
        return

    fig, ax = plt.subplots(figsize=(9, 5))
    pre  = es[es.year < POST_YEAR]
    post = es[es.year >= POST_YEAR]

    # Pre-treatment markers (blue circles)
    ax.errorbar(pre.year, pre.coef,
                yerr=[pre.coef - pre.ci_low, pre.ci_high - pre.coef],
                fmt='o-', color='steelblue', capsize=4, label='Pre-treatment')
    # Post-treatment markers (red squares)
    ax.errorbar(post.year, post.coef,
                yerr=[post.coef - post.ci_low, post.ci_high - post.coef],
                fmt='s-', color='firebrick', capsize=4, label='Post-treatment')

    ax.axhline(0, color='black', linewidth=0.8, linestyle='--')
    ax.axvline(REF_YEAR + 0.5, color='gray', linewidth=1, linestyle=':', label=f'Treatment ({POST_YEAR}+)')
    ax.set_xlabel('Year')
    ax.set_ylabel('Entropy difference (Treated − Control)')
    ax.set_title('Event Study: AI Exposure and Skill Comprehensiveness\n'
                 f'(Gradient 3/4 vs Minimal/Not Exposed, ref={REF_YEAR})')
    ax.legend()
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f'  Plot saved: {out_path}')
    plt.close()


# ── Summary table ─────────────────────────────────────────────────────────────
def print_cell_summary(cells: pd.DataFrame):
    print('\n── Cell means by year and treatment status ──')
    print(f'{"year":>5}  {"treated":>8}  {"control":>8}  {"diff":>8}  {"n_treat":>9}  {"n_ctrl":>9}')
    by_year = cells.groupby(['year', 'treated']).apply(
        lambda g: pd.Series({'mean': np.average(g.mean_outcome, weights=g.n),
                             'n': g.n.sum()}),
        include_groups=False
    ).reset_index()

    for year in sorted(cells.year.unique()):
        t = by_year[(by_year.year == year) & (by_year.treated == 1)]
        c = by_year[(by_year.year == year) & (by_year.treated == 0)]
        if t.empty or c.empty:
            continue
        tm, cm = t['mean'].values[0], c['mean'].values[0]
        tn, cn = int(t['n'].values[0]), int(c['n'].values[0])
        mark = ' *' if year >= POST_YEAR else ''
        print(f'{year:>5}  {tm:>8.6f}  {cm:>8.6f}  {tm-cm:>+8.6f}  {tn:>9,}  {cn:>9,}{mark}')
    print(f'  (* = post-treatment years, ref = {REF_YEAR})')


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df = load_data()
    cells_entropy = aggregate_cells(df, outcome='entropy')
    print_cell_summary(cells_entropy)

    # ── Spec 1: Baseline DID (entropy, no controls) ───────────────────────────
    print('\n[2] Spec 1: Baseline cell-level WLS (entropy) ...')
    res1 = run_cell_did(cells_entropy, label='entropy')
    (OUT_DIR / 'did_cell_baseline.txt').write_text(res1, encoding='utf-8')
    # Show DID coefficient only
    _print_did_coef(res1, 'Baseline')

    # ── Spec 2: Add log(token_count) control ─────────────────────────────────
    has_tokens = df['log_tokens'].notna().mean() > 0.5
    if has_tokens:
        print('\n[3] Spec 2: With log(token_count) control ...')
        res2 = run_cell_did(cells_entropy, label='entropy', add_token_ctrl=True)
        (OUT_DIR / 'did_cell_token_ctrl.txt').write_text(res2, encoding='utf-8')
        _print_did_coef(res2, '+log_tokens')
    else:
        print('\n[3] Skipping token_count control (not available in master)')

    # ── Spec 3: Robustness — HHI as outcome ──────────────────────────────────
    if 'hhi' in df.columns:
        print('\n[4] Spec 3: Robustness — HHI as outcome ...')
        df['neg_hhi'] = -df['hhi']  # negate so "higher = more comprehensive"
        cells_hhi = aggregate_cells(df, outcome='neg_hhi')
        res3 = run_cell_did(cells_hhi, label='-HHI (more diverse → higher)')
        (OUT_DIR / 'did_cell_hhi.txt').write_text(res3, encoding='utf-8')
        _print_did_coef(res3, '-HHI')

    # ── Spec 4: Robustness — unique_tokens as outcome ────────────────────────
    has_unique = df['unique_tokens'].notna().mean() > 0.5
    if has_unique:
        print('\n[5] Spec 4: Robustness — unique_tokens (vocab breadth) as outcome ...')
        df['log_unique'] = np.log(df['unique_tokens'].clip(lower=1))
        cells_unique = aggregate_cells(df, outcome='log_unique')
        res4 = run_cell_did(cells_unique, label='log(unique_tokens)')
        (OUT_DIR / 'did_cell_unique_tokens.txt').write_text(res4, encoding='utf-8')
        _print_did_coef(res4, 'log_unique')
    else:
        print('\n[5] Skipping unique_tokens robustness (word_entropy not merged)')

    # ── Individual TWFE (main spec, sample if needed) ────────────────────────
    print('\n[6] Individual TWFE (main spec, entropy) ...')
    sample = 500_000 if len(df) > 500_000 else 0
    twfe_res = run_individual_twfe(df, sample_n=sample)
    (OUT_DIR / 'did_indiv_twfe.txt').write_text(twfe_res, encoding='utf-8')
    print(twfe_res[:2000])

    # ── Event study ───────────────────────────────────────────────────────────
    print('\n[7] Event study ...')
    es = run_event_study(cells_entropy)
    es.to_csv(OUT_DIR / 'event_study_coefs.csv', index=False, float_format='%.6f')
    print(es.to_string(index=False))
    plot_event_study(es, OUT_DIR / 'event_study_plot.png')

    print(f'\n✓ Results saved to {OUT_DIR}/')


def _print_did_coef(result_text: str, label: str):
    """Extract and print just the DID coefficient line."""
    for line in result_text.split('\n'):
        if line.strip().startswith('did '):
            print(f'  [{label}]  {line.strip()}')
            return
    print(f'  [{label}]  (DID coef not found in output)')


if __name__ == '__main__':
    main()
