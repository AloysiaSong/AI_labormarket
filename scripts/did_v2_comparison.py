#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
did_v2_comparison.py

Compare DID results between v1 (keyword-only) and v2 (keyword + SBERT) master data.
Addresses fautes_1.txt criticism IV-2: "30% systematic missing data".

Steps:
  1. Coverage diagnostics: unknown rate v1 vs v2
  2. Composition check: are the newly-matched records systematically different?
  3. Re-run all 5 DID specs on v2 data
  4. Print side-by-side comparison table
"""

import csv
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

BASE = Path('/Users/yu/code/code2601/TY')
V1_CSV = BASE / 'data/Heterogeneity/master_with_ai_exposure.csv'
V2_CSV = BASE / 'data/Heterogeneity/master_with_ai_exposure_v2.csv'
WORD_ENT_CSV = BASE / 'data/Heterogeneity/word_entropy_by_id.csv'
OUT_DIR = BASE / 'data/Heterogeneity/did_results_v2'

POST_YEAR = 2023
EXCL_YEARS = {2019, 2020}
HIGH_EXPOSURE = {'Gradient 3', 'Gradient 4'}
LOW_EXPOSURE = {'Minimal Exposure', 'Not Exposed'}
REF_YEAR = 2022


# ═══════════════════════════════════════════════════════════════════════
# Part 1: Coverage diagnostics
# ═══════════════════════════════════════════════════════════════════════

def coverage_diagnostics():
    """Compare match coverage between v1 and v2."""
    print('=' * 70)
    print('PART 1: COVERAGE DIAGNOSTICS — v1 vs v2')
    print('=' * 70)

    for label, path in [('v1 (keyword only)', V1_CSV), ('v2 (keyword + SBERT)', V2_CSV)]:
        print(f'\n── {label}: {path.name} ──')
        # Stream to avoid loading 1.6GB into memory for just counts
        method_counts = {}
        gradient_counts = {}
        year_unknown = {}
        year_total = {}
        total = 0

        with path.open(encoding='utf-8-sig') as f:
            reader = csv.DictReader(f)
            for row in reader:
                total += 1
                method = row.get('isco_match_method', 'unknown').strip()

                # Bucket sbert_0.XXX variants into 'sbert'
                if method.startswith('sbert'):
                    method_key = 'sbert'
                else:
                    method_key = method
                method_counts[method_key] = method_counts.get(method_key, 0) + 1

                g = row.get('ai_exposure_gradient', '').strip()
                gradient_counts[g] = gradient_counts.get(g, 0) + 1

                try:
                    year = int(row['year'])
                except (ValueError, KeyError):
                    continue
                if year in EXCL_YEARS:
                    continue
                year_total[year] = year_total.get(year, 0) + 1
                if method_key == 'unknown':
                    year_unknown[year] = year_unknown.get(year, 0) + 1

        print(f'  Total records: {total:,}')

        print(f'\n  Match method distribution:')
        for m, c in sorted(method_counts.items(), key=lambda x: -x[1]):
            print(f'    {m:<20s}  {c:>10,}  ({c/total*100:5.1f}%)')

        unknown = method_counts.get('unknown', 0)
        print(f'\n  ★ Unknown (unmatched) rate: {unknown:,} / {total:,} = {unknown/total*100:.1f}%')

        print(f'\n  AI exposure gradient distribution:')
        for g, c in sorted(gradient_counts.items(), key=lambda x: -x[1]):
            print(f'    {g:<25s}  {c:>10,}  ({c/total*100:5.1f}%)')

        # DID-eligible (high + low exposure, excl 2019/2020)
        did_eligible = sum(c for g, c in gradient_counts.items()
                          if g in HIGH_EXPOSURE | LOW_EXPOSURE)
        print(f'\n  DID-eligible (High+Low exposure): {did_eligible:,}')

        print(f'\n  Unknown rate by year (excl 2019-2020):')
        for y in sorted(year_total.keys()):
            unk = year_unknown.get(y, 0)
            tot = year_total[y]
            print(f'    {y}:  {unk:>8,} / {tot:>10,}  ({unk/tot*100:5.1f}%)')


# ═══════════════════════════════════════════════════════════════════════
# Part 2: Load data for DID
# ═══════════════════════════════════════════════════════════════════════

def load_did_data(csv_path: Path, label: str) -> pd.DataFrame:
    """Load and prepare data for DID regression."""
    print(f'\n[Loading {label}] {csv_path.name} ...')
    rows = []
    skipped = 0
    with csv_path.open(encoding='utf-8-sig') as f:
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
                'match_method': row.get('isco_match_method', 'unknown').strip(),
            }
            rows.append(r)

    df = pd.DataFrame(rows)
    print(f'  Kept: {len(df):>10,}  (dropped {skipped:,})')
    print(f'  Years: {sorted(df.year.unique())}')
    print(f'  Treated: {df.treated.sum():,} ({df.treated.mean():.3f})')
    print(f'  Control: {(1-df.treated).sum():,.0f} ({1-df.treated.mean():.3f})')

    # Merge word entropy if available
    if WORD_ENT_CSV.exists():
        we = pd.read_csv(WORD_ENT_CSV, usecols=['id', 'unique_tokens', 'token_count'],
                         dtype={'id': int})
        we['log_unique_tokens'] = np.log(we['unique_tokens'].clip(lower=1))
        df = df.merge(we[['id', 'unique_tokens', 'log_unique_tokens']], on='id', how='left')
    else:
        df['unique_tokens'] = np.nan
        df['log_unique_tokens'] = np.nan

    return df


# ═══════════════════════════════════════════════════════════════════════
# Part 3: DID regressions
# ═══════════════════════════════════════════════════════════════════════

def aggregate_cells(df, outcome='entropy'):
    agg_dict = {'n': (outcome, 'size'), f'mean_{outcome}': (outcome, 'mean')}
    if 'log_tokens' in df.columns and df['log_tokens'].notna().any():
        agg_dict['mean_log_tokens'] = ('log_tokens', 'mean')
    agg = (df.groupby(['year', 'gradient', 'post', 'treated'])
             .agg(**agg_dict)
             .reset_index())
    agg = agg.rename(columns={f'mean_{outcome}': 'mean_outcome'})
    return agg


def run_cell_did(cells, label='entropy', add_token_ctrl=False):
    import statsmodels.formula.api as smf
    data = cells.copy()
    data['did'] = data['post'] * data['treated']
    rhs = 'did + C(year) + C(gradient)'
    if add_token_ctrl and 'mean_log_tokens' in data.columns and data['mean_log_tokens'].notna().all():
        rhs += ' + mean_log_tokens'
    formula = f'mean_outcome ~ {rhs}'
    model = smf.wls(formula, data=data, weights=data['n'])
    res = model.fit(cov_type='HC3')

    did_coef = res.params.get('did', np.nan)
    did_se = res.bse.get('did', np.nan)
    did_p = res.pvalues.get('did', np.nan)
    n_obs = int(cells.n.sum())

    return {
        'label': label,
        'token_ctrl': add_token_ctrl,
        'did_coef': did_coef,
        'did_se': did_se,
        'did_p': did_p,
        'n_cells': len(cells),
        'n_obs': n_obs,
        'full_result': res,
    }


def run_individual_twfe(df, sample_n=500_000):
    try:
        from linearmodels.iv.absorbing import AbsorbingLS
    except ImportError:
        return None

    work = df.copy()
    if sample_n > 0 and len(work) > sample_n:
        work = work.sample(sample_n, random_state=42)

    work['did'] = work['post'] * work['treated']
    work['year_cat'] = pd.Categorical(work['year'])
    work['grad_cat'] = pd.Categorical(work['gradient'])

    endog = work[['entropy']]
    exog = work[['did']]
    absorb = work[['year_cat', 'grad_cat']]

    mod = AbsorbingLS(endog, exog, absorb=absorb)
    res = mod.fit(cov_type='clustered', clusters=work['year'].values)

    return {
        'label': 'TWFE (500k sample)',
        'did_coef': res.params['did'],
        'did_se': res.std_errors['did'],
        'did_p': res.pvalues['did'],
        'n_obs': len(work),
        'full_result': res,
    }


def run_event_study(cells):
    import statsmodels.formula.api as smf
    years = sorted(cells['year'].unique())
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
            rows.append({'year': y, 'coef': 0.0, 'se': 0.0, 'pvalue': np.nan})
            continue
        col = f'yr{y}'
        coef = res.params.get(col, np.nan)
        se = res.bse.get(col, np.nan)
        rows.append({
            'year': y,
            'coef': coef,
            'se': se,
            'pvalue': res.pvalues.get(col, np.nan),
        })
    return pd.DataFrame(rows)


def run_all_specs(df, label_prefix):
    """Run all 5 specs and return list of result dicts."""
    results = []

    # Spec 1: Baseline WLS
    cells = aggregate_cells(df, 'entropy')
    r1 = run_cell_did(cells, label='entropy')
    r1['spec'] = f'{label_prefix} Spec1: WLS baseline'
    results.append(r1)

    # Spec 2: + log(token_count)
    r2 = run_cell_did(cells, label='entropy', add_token_ctrl=True)
    r2['spec'] = f'{label_prefix} Spec2: +log_tokens'
    results.append(r2)

    # Spec 3: -HHI
    df_tmp = df.copy()
    df_tmp['neg_hhi'] = -df_tmp['hhi']
    cells_hhi = aggregate_cells(df_tmp, 'neg_hhi')
    r3 = run_cell_did(cells_hhi, label='-HHI')
    r3['spec'] = f'{label_prefix} Spec3: -HHI'
    results.append(r3)

    # Spec 4: unique_tokens
    has_unique = df['unique_tokens'].notna().mean() > 0.5
    if has_unique:
        df_tmp2 = df.copy()
        df_tmp2['log_unique'] = np.log(df_tmp2['unique_tokens'].clip(lower=1))
        cells_uniq = aggregate_cells(df_tmp2, 'log_unique')
        r4 = run_cell_did(cells_uniq, label='log(unique_tokens)')
        r4['spec'] = f'{label_prefix} Spec4: unique_tokens'
        results.append(r4)

    # Spec 5: TWFE
    r5 = run_individual_twfe(df)
    if r5:
        r5['spec'] = f'{label_prefix} Spec5: TWFE 500k'
        results.append(r5)

    return results


# ═══════════════════════════════════════════════════════════════════════
# Part 4: Composition analysis of newly-matched records
# ═══════════════════════════════════════════════════════════════════════

def composition_check(df_v2):
    """Compare entropy/treatment distribution of keyword vs sbert-matched records."""
    print('\n' + '=' * 70)
    print('PART 2: COMPOSITION CHECK — Keyword vs SBERT-matched records')
    print('=' * 70)

    is_sbert = df_v2['match_method'].str.startswith('sbert')
    is_keyword = df_v2['match_method'].str.startswith('keyword')

    for name, mask in [('keyword', is_keyword), ('sbert', is_sbert)]:
        sub = df_v2[mask]
        if len(sub) == 0:
            print(f'\n  {name}: 0 records')
            continue
        print(f'\n  ── {name} ({len(sub):,} records) ──')
        print(f'    Treated rate:    {sub.treated.mean():.3f}')
        print(f'    Mean entropy:    {sub.entropy.mean():.4f}')
        print(f'    Mean log_tokens: {sub.log_tokens.mean():.3f}')
        print(f'    Year distribution:')
        yd = sub.groupby('year').size()
        for y, n in yd.items():
            print(f'      {y}: {n:>10,}')
        print(f'    Gradient distribution:')
        gd = sub.groupby('gradient').size()
        for g, n in gd.items():
            print(f'      {g:<25s} {n:>10,}  ({n/len(sub)*100:.1f}%)')


# ═══════════════════════════════════════════════════════════════════════
# Part 5: Side-by-side comparison
# ═══════════════════════════════════════════════════════════════════════

def comparison_table(v1_results, v2_results):
    print('\n' + '=' * 70)
    print('PART 4: SIDE-BY-SIDE COMPARISON — v1 vs v2')
    print('=' * 70)

    header = f'{"Spec":<30s}  {"β(v1)":>10s}  {"p(v1)":>8s}  {"N(v1)":>12s}  │  {"β(v2)":>10s}  {"p(v2)":>8s}  {"N(v2)":>12s}  {"Δβ":>8s}'
    print(f'\n{header}')
    print('─' * len(header))

    for r1, r2 in zip(v1_results, v2_results):
        spec_name = r1['spec'].split(': ', 1)[1] if ': ' in r1['spec'] else r1['spec']
        b1 = r1['did_coef']
        b2 = r2['did_coef']
        delta = b2 - b1
        print(f'{spec_name:<30s}  {b1:>10.4f}  {r1["did_p"]:>8.3f}  {r1["n_obs"]:>12,}  │  '
              f'{b2:>10.4f}  {r2["did_p"]:>8.3f}  {r2["n_obs"]:>12,}  {delta:>+8.4f}')


def event_study_comparison(es_v1, es_v2):
    print('\n── Event Study Comparison ──')
    header = f'{"Year":>6}  {"β(v1)":>10s}  {"p(v1)":>8s}  │  {"β(v2)":>10s}  {"p(v2)":>8s}  {"Δβ":>10s}'
    print(header)
    print('─' * len(header))
    merged = es_v1.merge(es_v2, on='year', suffixes=('_v1', '_v2'))
    for _, row in merged.iterrows():
        y = int(row['year'])
        marker = ' ←ref' if y == REF_YEAR else (' *' if y >= POST_YEAR else '')
        delta = row['coef_v2'] - row['coef_v1']
        p1 = f'{row["pvalue_v1"]:.3f}' if not np.isnan(row['pvalue_v1']) else '  ref'
        p2 = f'{row["pvalue_v2"]:.3f}' if not np.isnan(row['pvalue_v2']) else '  ref'
        print(f'{y:>6}  {row["coef_v1"]:>+10.4f}  {p1:>8s}  │  '
              f'{row["coef_v2"]:>+10.4f}  {p2:>8s}  {delta:>+10.4f}{marker}')


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Part 1: Coverage ──
    coverage_diagnostics()

    # ── Part 2: Load data ──
    df_v1 = load_did_data(V1_CSV, 'v1')
    df_v2 = load_did_data(V2_CSV, 'v2')

    # ── Part 2b: Composition ──
    composition_check(df_v2)

    # ── Part 3: Cell-level summary ──
    print('\n' + '=' * 70)
    print('PART 3: DID REGRESSIONS')
    print('=' * 70)

    print('\n── v1: Running all specs ──')
    v1_results = run_all_specs(df_v1, 'v1')
    print('\n── v2: Running all specs ──')
    v2_results = run_all_specs(df_v2, 'v2')

    # ── Part 4: Compare ──
    comparison_table(v1_results, v2_results)

    # ── Event study ──
    cells_v1 = aggregate_cells(df_v1, 'entropy')
    cells_v2 = aggregate_cells(df_v2, 'entropy')
    es_v1 = run_event_study(cells_v1)
    es_v2 = run_event_study(cells_v2)
    event_study_comparison(es_v1, es_v2)

    # ── Save detailed results ──
    es_v2.to_csv(OUT_DIR / 'event_study_coefs_v2.csv', index=False, float_format='%.6f')

    # Save comparison summary
    summary_lines = []
    summary_lines.append('DID v1 vs v2 Comparison Summary')
    summary_lines.append('=' * 60)
    summary_lines.append(f'v1 DID-eligible obs: {len(df_v1):,}')
    summary_lines.append(f'v2 DID-eligible obs: {len(df_v2):,}')
    summary_lines.append(f'New records added:   {len(df_v2)-len(df_v1):,}  ({(len(df_v2)-len(df_v1))/len(df_v1)*100:+.1f}%)')
    summary_lines.append('')
    summary_lines.append(f'{"Spec":<30s}  {"β_v1":>8s} {"p_v1":>6s}  {"β_v2":>8s} {"p_v2":>6s}  Δβ')
    for r1, r2 in zip(v1_results, v2_results):
        spec_name = r1['spec'].split(': ', 1)[1]
        summary_lines.append(
            f'{spec_name:<30s}  {r1["did_coef"]:>8.4f} {r1["did_p"]:>6.3f}  '
            f'{r2["did_coef"]:>8.4f} {r2["did_p"]:>6.3f}  {r2["did_coef"]-r1["did_coef"]:>+8.4f}'
        )
    (OUT_DIR / 'comparison_summary.txt').write_text('\n'.join(summary_lines), encoding='utf-8')

    # Save full Spec1 and TWFE results for v2
    for r in v2_results:
        if hasattr(r.get('full_result'), 'summary'):
            fname = r['spec'].replace(' ', '_').replace(':', '').replace('/', '_')
            try:
                (OUT_DIR / f'{fname}.txt').write_text(
                    str(r['full_result'].summary()), encoding='utf-8')
            except Exception:
                pass

    print(f'\n✓ All results saved to {OUT_DIR}/')
    print('Done.')


if __name__ == '__main__':
    main()
