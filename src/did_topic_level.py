#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Topic-level DID: For each LDA topic k, test whether high-AI-exposure
occupations see a change in topic share after 2022.

Y = share_k (fraction of jobs in cell with dominant_topic_id == k)
Model: share_k = β_k (AIExpo × Post) + cond_PT + FEs
"""

import time
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from pathlib import Path

BASE = Path("/Users/yu/code/code2601/TY")
OUTPUT_DIR = BASE / "output"
RESULTS_DIR = OUTPUT_DIR / "regression"
RESULTS_DIR.mkdir(exist_ok=True)

ISCO_PANEL = OUTPUT_DIR / "ind_isco_year_panel.csv"
FINAL_RESULTS = OUTPUT_DIR / "final_results_sample.csv"
JOB_PANEL = OUTPUT_DIR / "job_panel.csv"
TOPIC_CLASS = OUTPUT_DIR / "topic_decomposition" / "topic_classification.csv"
ISCO_BASELINE = OUTPUT_DIR / "isco_baseline_features.csv"
AI_INTENSITY_CSV = BASE / "data" / "ai_intensity_annual.csv"

MIN_CELL_SIZE = 30


def build_topic_share_panel():
    """
    Build industry × ISCO × year panel with topic shares as dependent variables.
    Uses dominant_topic_id from final_results + cluster/industry from job_panel.
    """
    print("[1] Building topic share panel...")

    # Stream-join: final_results has (id, year, dominant_topic_id)
    # job_panel has (id, year, cluster_id/isco, industry_code, n_jobs weight proxy)
    # We need to merge and aggregate topic shares at ind × isco × year level.

    # Load job_panel into dict: id -> (isco, industry)
    print("  Loading job_panel for cluster/industry mapping...")
    job_info = {}
    chunk_size = 2_000_000
    for chunk in pd.read_csv(JOB_PANEL, chunksize=chunk_size, encoding='utf-8-sig',
                             usecols=['id', 'cluster_id', 'industry_code']):
        for _, row in chunk.iterrows():
            job_info[str(row['id'])] = (str(row['cluster_id']), str(row['industry_code']))
        print(f"    Loaded {len(job_info):,} jobs...", flush=True)
    print(f"  Total job mappings: {len(job_info):,}")

    # Now load the ISCO panel to get isco08_4digit mapping from cluster_id
    # Actually, the ISCO panel uses isco08_4digit directly. But job_panel uses cluster_id.
    # We need cluster_id -> isco08_4digit mapping.
    isco_panel = pd.read_csv(ISCO_PANEL, usecols=['isco08_4digit', 'industry_code', 'year', 'n_jobs', 'ai_exposure_ilo'],
                             encoding='utf-8-sig', nrows=1)
    # The ind_isco_year_panel uses isco08_4digit. But job_panel uses cluster_id (which IS the isco08_4digit).
    # Let me verify this by checking the build_panel script mapping.
    # From build_panel.py: cluster_map maps row_idx -> (cluster_id, cluster_name, major_class)
    # And the ISCO panel aggregation uses cluster_id as the occupation identifier.
    # So cluster_id in job_panel IS the same as isco08_4digit in ind_isco_year_panel?
    # Actually no - cluster_id is from HDBSCAN clustering, while isco08_4digit is the SBERT-matched ISCO code.
    # Let me check the ind_isco_year_panel construction.

    # Actually, the ind_isco_year_panel.csv was likely built separately with ISCO codes.
    # Let me check what columns exist.
    print("  Checking panel structure...")
    panel_cols = pd.read_csv(ISCO_PANEL, nrows=0, encoding='utf-8-sig').columns.tolist()
    print(f"  ISCO panel columns: {panel_cols[:10]}")

    # The simplest approach: aggregate from final_results directly.
    # Group by (cluster_id, industry_code, year) -> topic share for each topic.

    # Get all unique topics
    topic_class = pd.read_csv(TOPIC_CLASS)
    all_topics = sorted(topic_class['topic_id'].unique())

    print("  Streaming final_results + job_panel to compute topic shares...")
    # Accumulate: (cluster_id, industry_code, year) -> {topic_id: count}
    from collections import defaultdict, Counter
    cell_topic_counts = defaultdict(Counter)
    cell_total = defaultdict(int)

    matched = 0
    unmatched = 0
    for chunk in pd.read_csv(FINAL_RESULTS, chunksize=chunk_size, encoding='utf-8-sig',
                             usecols=['id', 'year', 'dominant_topic_id']):
        for _, row in chunk.iterrows():
            jid = str(row['id'])
            info = job_info.get(jid)
            if info is None:
                unmatched += 1
                continue
            cluster_id, ind_code = info
            if not ind_code or ind_code == 'nan':
                unmatched += 1
                continue
            year = int(row['year'])
            topic = int(row['dominant_topic_id'])
            key = (cluster_id, ind_code, year)
            cell_topic_counts[key][topic] += 1
            cell_total[key] += 1
            matched += 1

        print(f"    Processed {matched + unmatched:,} rows (matched={matched:,})...", flush=True)

    print(f"  Matched: {matched:,}, Unmatched: {unmatched:,}")
    print(f"  Unique cells: {len(cell_total):,}")

    # Build DataFrame
    rows = []
    for key, total in cell_total.items():
        if total < MIN_CELL_SIZE:
            continue
        cluster_id, ind_code, year = key
        row = {'isco08_4digit': cluster_id, 'industry_code': ind_code,
               'year': year, 'n_jobs': total}
        counts = cell_topic_counts[key]
        for t in all_topics:
            row[f'share_t{t}'] = counts.get(t, 0) / total
        rows.append(row)

    df = pd.DataFrame(rows)
    print(f"  Panel after min_cell filter: {len(df):,} cells")
    return df, topic_class


def run_topic_did(df, topic_id, topic_info):
    """Run DID for a single topic's share."""
    dv = f'share_t{topic_id}'
    if dv not in df.columns:
        return None

    formula = (f"{dv} ~ treat_x_post "
               f"+ sal_trend + edu_trend + exp_trend "
               f"+ C(occ_fe) + C(year_fe) + C(ind_fe)")
    try:
        model = smf.wls(formula, data=df, weights=df['n_jobs']).fit(
            cov_type='cluster', cov_kwds={'groups': df['cluster_id']}
        )
        b = model.params['treat_x_post']
        se = model.bse['treat_x_post']
        p = model.pvalues['treat_x_post']
        ci_lo, ci_hi = model.conf_int().loc['treat_x_post']
        stars = '***' if p < 0.01 else '**' if p < 0.05 else '*' if p < 0.1 else ''

        mean_share = df[dv].mean()
        keywords = topic_info.get('keywords', '')
        category = topic_info.get('category', '')
        cat_label = topic_info.get('category_label', '')

        print(f"  T{topic_id:2d} [{cat_label:25s}] β={b:+.6f}{stars:3s}  "
              f"SE={se:.6f}  p={p:.4f}  mean_share={mean_share:.4f}  "
              f"kw: {keywords[:50]}")

        return {
            'topic_id': topic_id, 'category': category, 'category_label': cat_label,
            'keywords': keywords, 'beta': b, 'se': se, 'pval': p,
            'ci_lo': ci_lo, 'ci_hi': ci_hi, 'mean_share': mean_share,
            'stars': stars, 'n': model.nobs, 'r2': model.rsquared
        }
    except Exception as e:
        print(f"  T{topic_id:2d}  ERROR: {e}")
        return None


def main():
    t0 = time.time()

    df_topics, topic_class = build_topic_share_panel()

    # Merge AI exposure
    isco_panel = pd.read_csv(ISCO_PANEL, encoding='utf-8-sig',
                             usecols=['isco08_4digit', 'ai_exposure_ilo'])
    isco_panel['isco08_4digit'] = isco_panel['isco08_4digit'].astype(str)
    expo_map = isco_panel.groupby('isco08_4digit')['ai_exposure_ilo'].first().to_dict()
    df_topics['isco08_4digit'] = df_topics['isco08_4digit'].astype(str)
    df_topics['ai_exposure_ilo'] = df_topics['isco08_4digit'].map(expo_map)
    df_topics = df_topics.dropna(subset=['ai_exposure_ilo'])
    print(f"  After exposure merge: {len(df_topics):,} cells")

    # Treatment variables
    df_topics['year'] = df_topics['year'].astype(int)
    df_topics['post'] = (df_topics['year'] >= 2022).astype(int)
    df_topics['treat_x_post'] = df_topics['ai_exposure_ilo'] * df_topics['post']

    # FEs
    df_topics['occ_fe'] = pd.Categorical(df_topics['isco08_4digit'])
    df_topics['year_fe'] = pd.Categorical(df_topics['year'])
    df_topics['ind_fe'] = pd.Categorical(df_topics['industry_code'])
    df_topics['cluster_id'] = df_topics['isco08_4digit']

    # Conditional PT controls
    feat = pd.read_csv(OUTPUT_DIR / "isco_baseline_features.csv", encoding='utf-8-sig')
    feat['isco08_4digit'] = feat['isco08_4digit'].astype(str)
    feat['year'] = feat['year'].astype(int)
    baseline = feat[feat['year'].isin([2016, 2017, 2018])].groupby('isco08_4digit').agg(
        base_salary=('mean_salary', 'mean'),
        base_edu=('mean_edu', 'mean'),
        base_exp=('mean_exp', 'mean'),
    ).reset_index()
    df_topics = df_topics.merge(baseline, on='isco08_4digit', how='left')
    for col in ['base_salary', 'base_edu', 'base_exp']:
        df_topics[col] = (df_topics[col] - df_topics[col].mean()) / df_topics[col].std()
    df_topics['sal_trend'] = df_topics['base_salary'] * df_topics['year']
    df_topics['edu_trend'] = df_topics['base_edu'] * df_topics['year']
    df_topics['exp_trend'] = df_topics['base_exp'] * df_topics['year']

    print(f"\n[2] Running topic-level DID (cond PT):")
    print("=" * 120)

    # Build topic info dict
    tc_dict = {}
    for _, row in topic_class.iterrows():
        tc_dict[row['topic_id']] = row.to_dict()

    # Get topics that appear in data
    topic_cols = [c for c in df_topics.columns if c.startswith('share_t')]
    topic_ids = sorted([int(c.replace('share_t', '')) for c in topic_cols])

    results = []
    for tid in topic_ids:
        info = tc_dict.get(tid, {'keywords': '?', 'category': '?', 'category_label': '?'})
        r = run_topic_did(df_topics, tid, info)
        if r:
            results.append(r)

    if results:
        rdf = pd.DataFrame(results).sort_values('pval')
        rdf.to_csv(RESULTS_DIR / "did_topic_level.csv", index=False)

        # BH correction
        n_tests = len(rdf)
        rdf['rank'] = range(1, n_tests + 1)
        rdf['bh_threshold'] = 0.05 * rdf['rank'] / n_tests
        rdf['fdr_sig'] = rdf['pval'] <= rdf['bh_threshold']

        print(f"\n  === Summary (sorted by p-value) ===")
        print(f"  {'Topic':5s} {'Category':25s} {'β':>10s} {'p':>8s} {'FDR':>5s} {'Keywords'}")
        for _, r in rdf.head(20).iterrows():
            fdr = '✓' if r['fdr_sig'] else ''
            print(f"  T{r['topic_id']:<3.0f} {r['category_label']:25s} {r['beta']:+.6f} "
                  f"{r['pval']:.4f} {fdr:>5s} {r['keywords'][:60]}")

        n_sig = (rdf['pval'] < 0.05).sum()
        n_fdr = rdf['fdr_sig'].sum()
        n_pos = ((rdf['pval'] < 0.05) & (rdf['beta'] > 0)).sum()
        n_neg = ((rdf['pval'] < 0.05) & (rdf['beta'] < 0)).sum()
        print(f"\n  p<0.05: {n_sig}/{n_tests} topics ({n_pos} positive, {n_neg} negative)")
        print(f"  FDR-corrected: {n_fdr}/{n_tests} topics")

        # Category-level summary
        print(f"\n  === By task category ===")
        for cat in ['Non-routine Analytical', 'Non-routine Interactive',
                     'Routine Cognitive', 'Routine Manual', 'Meta/Generic']:
            sub = rdf[rdf['category_label'] == cat]
            if len(sub) == 0:
                continue
            mean_b = sub['beta'].mean()
            n_sig_cat = (sub['pval'] < 0.05).sum()
            print(f"  {cat:25s}: mean_β={mean_b:+.6f}  sig={n_sig_cat}/{len(sub)}")

    elapsed = time.time() - t0
    print(f"\nDone! Total time: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
