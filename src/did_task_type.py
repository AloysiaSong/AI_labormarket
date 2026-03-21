#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ALM Task-type DID: For each task category (NRA, NRI, RC, RM, M),
test whether high-AI-exposure occupations see a change in task-type share.

Uses dominant_topic_id -> ALM category mapping, aggregated to
industry × ISCO × year cells.
"""

import time
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from pathlib import Path
from collections import defaultdict, Counter

BASE = Path("/Users/yu/code/code2601/TY")
OUTPUT_DIR = BASE / "output"
RESULTS_DIR = OUTPUT_DIR / "regression"

ISCO_PANEL = OUTPUT_DIR / "ind_isco_year_panel.csv"
FINAL_RESULTS = OUTPUT_DIR / "final_results_sample.csv"
JOB_PANEL = OUTPUT_DIR / "job_panel.csv"
TOPIC_CLASS = OUTPUT_DIR / "topic_decomposition" / "topic_classification.csv"
ISCO_BASELINE = OUTPUT_DIR / "isco_baseline_features.csv"

MIN_CELL_SIZE = 30
CHUNK = 2_000_000

CATEGORY_LABELS = {
    'NRA': 'Non-routine Analytical',
    'NRI': 'Non-routine Interactive',
    'RC': 'Routine Cognitive',
    'RM': 'Routine Manual',
    'M': 'Meta/Generic',
}


def build_task_share_panel():
    """Build ind × ISCO × year panel with 5 ALM task-type shares."""
    print("[1] Building task-type share panel...")

    # Topic -> ALM category mapping
    tc = pd.read_csv(TOPIC_CLASS)
    topic_to_cat = dict(zip(tc['topic_id'], tc['category']))
    cats = sorted(CATEGORY_LABELS.keys())

    # Load job_panel: id -> (isco, industry)
    print("  Loading job_panel mappings...")
    job_info = {}
    for chunk in pd.read_csv(JOB_PANEL, chunksize=CHUNK, encoding='utf-8-sig',
                             usecols=['id', 'cluster_id', 'industry_code']):
        for _, row in chunk.iterrows():
            job_info[str(row['id'])] = (str(row['cluster_id']), str(row['industry_code']))
    print(f"  Job mappings: {len(job_info):,}")

    # Stream final_results: accumulate task-type counts per cell
    cell_cat_counts = defaultdict(Counter)
    cell_total = defaultdict(int)
    matched = 0

    for chunk in pd.read_csv(FINAL_RESULTS, chunksize=CHUNK, encoding='utf-8-sig',
                             usecols=['id', 'year', 'dominant_topic_id']):
        for _, row in chunk.iterrows():
            info = job_info.get(str(row['id']))
            if info is None:
                continue
            isco, ind = info
            if not ind or ind == 'nan':
                continue
            cat = topic_to_cat.get(int(row['dominant_topic_id']), 'M')
            key = (isco, ind, int(row['year']))
            cell_cat_counts[key][cat] += 1
            cell_total[key] += 1
            matched += 1
        print(f"    {matched:,} matched...", flush=True)

    print(f"  Total matched: {matched:,}, cells: {len(cell_total):,}")

    # Build DataFrame
    rows = []
    for key, total in cell_total.items():
        if total < MIN_CELL_SIZE:
            continue
        isco, ind, year = key
        row = {'isco08_4digit': isco, 'industry_code': ind, 'year': year, 'n_jobs': total}
        counts = cell_cat_counts[key]
        for cat in cats:
            row[f'share_{cat}'] = counts.get(cat, 0) / total
        rows.append(row)

    df = pd.DataFrame(rows)
    print(f"  Panel: {len(df):,} cells (min_jobs={MIN_CELL_SIZE})")
    return df


def add_controls(df):
    """Add treatment variables, FEs, and conditional PT controls."""
    # AI exposure: map from cluster_id (= isco08_4digit in this panel)
    # The task share panel uses cluster_id from job_panel, not ISCO-08 codes.
    # Load cluster-level exposure from ai_exposure_ilo.csv
    ae = pd.read_csv(OUTPUT_DIR / "clusters" / "ai_exposure_ilo.csv", encoding='utf-8-sig')
    ae['cluster_id'] = ae['cluster_id'].astype(str)
    expo = dict(zip(ae['cluster_id'], ae['ai_exposure_ilo']))
    df['isco08_4digit'] = df['isco08_4digit'].astype(str)
    df['ai_exposure_ilo'] = df['isco08_4digit'].map(expo)
    df = df.dropna(subset=['ai_exposure_ilo']).copy()

    df['year'] = df['year'].astype(int)
    df['post'] = (df['year'] >= 2022).astype(int)
    df['treat_x_post'] = df['ai_exposure_ilo'] * df['post']
    df['occ_trend'] = df['ai_exposure_ilo'] * df['year']

    df['occ_fe'] = pd.Categorical(df['isco08_4digit'])
    df['year_fe'] = pd.Categorical(df['year'])
    df['ind_fe'] = pd.Categorical(df['industry_code'])
    df['cluster_id'] = df['isco08_4digit']

    # Baseline features: ISCO baseline uses isco08_4digit, but our panel uses cluster_id.
    # Cluster_id maps to multiple ISCO codes. Use cluster-level baseline from ind_occ_year_panel.
    # Compute baseline from pre-period of our own panel.
    pre = df[df['year'].isin([2016, 2017, 2018])].copy()
    if len(pre) > 0:
        # Use share_NRA as a proxy for cognitive intensity (baseline occupation characteristic)
        baseline = pre.groupby('isco08_4digit').agg(
            base_nra=('share_NRA', 'mean'),
            base_rc=('share_RC', 'mean'),
        ).reset_index()
        df = df.merge(baseline, on='isco08_4digit', how='left')
        for col in ['base_nra', 'base_rc']:
            m, s = df[col].mean(), df[col].std()
            df[col] = (df[col] - m) / s if s > 0 else 0
        df['nra_trend'] = df['base_nra'] * df['year']
        df['rc_trend'] = df['base_rc'] * df['year']
    else:
        df['nra_trend'] = 0
        df['rc_trend'] = 0

    return df


def run_task_did(df, cat, label, cond_pt=True):
    """Run DID for one task category share."""
    dv = f'share_{cat}'
    cpt = " + nra_trend + rc_trend" if cond_pt else ""
    formula = f"{dv} ~ treat_x_post + C(occ_fe) + C(year_fe) + C(ind_fe){cpt}"

    model = smf.wls(formula, data=df, weights=df['n_jobs']).fit(
        cov_type='cluster', cov_kwds={'groups': df['cluster_id']}
    )
    b = model.params['treat_x_post']
    se = model.bse['treat_x_post']
    p = model.pvalues['treat_x_post']
    ci_lo, ci_hi = model.conf_int().loc['treat_x_post']
    stars = '***' if p < 0.01 else '**' if p < 0.05 else '*' if p < 0.1 else ''
    mean_share = df[dv].mean()

    print(f"  {label:30s}  β={b:+.6f}{stars:3s}  SE={se:.6f}  p={p:.4f}  "
          f"mean_share={mean_share:.4f}  CI=[{ci_lo:.6f}, {ci_hi:.6f}]  "
          f"N={model.nobs:.0f}  R²={model.rsquared:.4f}")

    return {
        'category': cat, 'label': label, 'beta': b, 'se': se, 'pval': p,
        'ci_lo': ci_lo, 'ci_hi': ci_hi, 'mean_share': mean_share,
        'stars': stars, 'n': model.nobs, 'r2': model.rsquared
    }


def main():
    t0 = time.time()

    df = build_task_share_panel()
    df = add_controls(df)
    print(f"  Final panel: {len(df):,} cells, {df['isco08_4digit'].nunique()} ISCO codes\n")

    cats = ['NRA', 'NRI', 'RC', 'RM', 'M']

    # Descriptive: mean shares
    print("[2] Baseline task-type shares (pre-2022):")
    pre = df[df['year'] < 2022]
    for cat in cats:
        m = np.average(pre[f'share_{cat}'], weights=pre['n_jobs'])
        print(f"  {CATEGORY_LABELS[cat]:30s}: {m:.4f}")

    # Main DID with cond PT
    print("\n[3] Task-type DID (cond PT):")
    print("=" * 120)
    results = []
    for cat in cats:
        r = run_task_did(df, cat, CATEGORY_LABELS[cat])
        results.append(r)

    # Also without cond PT for comparison
    print("\n[3b] Task-type DID (baseline, no cond PT):")
    print("=" * 120)
    results_base = []
    for cat in cats:
        r = run_task_did(df, cat, f"{CATEGORY_LABELS[cat]} (base)", cond_pt=False)
        results_base.append(r)

    # Save
    pd.DataFrame(results).to_csv(RESULTS_DIR / "did_task_type.csv", index=False)
    pd.DataFrame(results_base).to_csv(RESULTS_DIR / "did_task_type_base.csv", index=False)

    # Sum check: do betas sum to ~0? (shares must sum to 1)
    total_b = sum(r['beta'] for r in results)
    print(f"\n  Sum of β across 5 categories: {total_b:+.8f} (should ≈ 0)")

    elapsed = time.time() - t0
    print(f"\nDone! Total time: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
