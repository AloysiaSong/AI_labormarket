#!/usr/bin/env python3
"""
Robustness checks for post-period timing sensitivity.

Tests:
1. Alternative post-period definitions (Post≥2022, Post≥2023, Post≥2024)
2. Randomization inference: placebo cutpoints in pre-period
3. Post-period dose-response with time: AIExpo × (t - cutpoint) within post
"""

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from pathlib import Path

BASE = Path("/Users/yu/code/code2601/TY")
OUTPUT_DIR = BASE / "output"
RESULTS_DIR = OUTPUT_DIR / "regression"

# Reuse load_data from did_regression but we need to override post definition
ISCO_PANEL = OUTPUT_DIR / "ind_isco_year_panel.csv"
ISCO_BASELINE = OUTPUT_DIR / "isco_baseline_features.csv"
AI_INTENSITY_CSV = BASE / "data" / "ai_intensity_annual.csv"
MIN_CELL_SIZE = 30


def load_base_data():
    """Load panel without creating treatment variables (we'll define our own)."""
    df = pd.read_csv(ISCO_PANEL, encoding='utf-8-sig')
    df = df[df['n_jobs'] >= MIN_CELL_SIZE].copy()
    df['ai_exposure_ilo'] = df['ai_exposure_ilo'].astype(float)
    df['year'] = df['year'].astype(int)

    # Baseline features for cond PT (same logic as did_regression.py)
    feat = pd.read_csv(ISCO_BASELINE, encoding='utf-8-sig')
    feat['isco08_4digit'] = feat['isco08_4digit'].astype(str)
    feat['year'] = feat['year'].astype(int)
    baseline = feat[feat['year'].isin([2016, 2017, 2018])].groupby('isco08_4digit').agg(
        base_salary=('mean_salary', 'mean'),
        base_edu=('mean_edu', 'mean'),
        base_exp=('mean_exp', 'mean'),
    ).reset_index()
    df['isco08_4digit'] = df['isco08_4digit'].astype(str)
    df = df.merge(baseline, on='isco08_4digit', how='left')
    for col in ['base_salary', 'base_edu', 'base_exp']:
        df[col] = (df[col] - df[col].mean()) / df[col].std()
    df['sal_trend'] = df['base_salary'] * df['year']
    df['edu_trend'] = df['base_edu'] * df['year']
    df['exp_trend'] = df['base_exp'] * df['year']
    # Keep baseline_ aliases for cross-section test
    df['baseline_salary'] = df['base_salary']
    df['baseline_edu'] = df['base_edu']
    df['baseline_exp'] = df['base_exp']

    # FEs
    df['occ_fe'] = pd.Categorical(df['isco08_4digit'])
    df['year_fe'] = pd.Categorical(df['year'])
    df['ind_fe'] = pd.Categorical(df['industry_code'])
    df['cluster_id'] = df['isco08_4digit'].astype(str)

    return df


def run_did_with_cutpoint(df, dep_var, cutpoint, cond_pt=True):
    """Run DID with a specific post-period cutpoint."""
    df = df.copy()
    df['post'] = (df['year'] >= cutpoint).astype(int)
    df['treat_x_post'] = df['ai_exposure_ilo'] * df['post']

    controls = "+ sal_trend + edu_trend + exp_trend" if cond_pt else ""
    formula = f"mean_{dep_var} ~ treat_x_post {controls} + C(occ_fe) + C(year_fe) + C(ind_fe)"

    model = smf.wls(formula, data=df, weights=df['n_jobs']).fit(
        cov_type='cluster', cov_kwds={'groups': df['cluster_id']}
    )
    return {
        'cutpoint': cutpoint,
        'beta': model.params['treat_x_post'],
        'se': model.bse['treat_x_post'],
        'pval': model.pvalues['treat_x_post'],
        'n': model.nobs,
        'n_post': df[df['post'] == 1].shape[0],
    }


def test1_alternative_post_definitions(df):
    """Test 1: Vary the post-period cutpoint (2022, 2023, 2024)."""
    print("\n" + "="*70)
    print("TEST 1: Alternative Post-Period Definitions (Cond PT)")
    print("="*70)

    results = []
    for cutpoint in [2022, 2023, 2024]:
        r = run_did_with_cutpoint(df, 'entropy_score', cutpoint, cond_pt=True)
        stars = '***' if r['pval'] < 0.01 else '**' if r['pval'] < 0.05 else '*' if r['pval'] < 0.1 else ''
        print(f"  Post >= {cutpoint}: β={r['beta']:+.4f}{stars:3s}  SE={r['se']:.4f}  p={r['pval']:.4f}  N_post={r['n_post']}")
        results.append(r)

    # Also do effective topics
    print("\n  --- Effective # Topics ---")
    for cutpoint in [2022, 2023, 2024]:
        r = run_did_with_cutpoint(df, 'ent_effective', cutpoint, cond_pt=True)
        stars = '***' if r['pval'] < 0.01 else '**' if r['pval'] < 0.05 else '*' if r['pval'] < 0.1 else ''
        print(f"  Post >= {cutpoint}: β={r['beta']:+.4f}{stars:3s}  SE={r['se']:.4f}  p={r['pval']:.4f}")
        r['dep_var'] = 'ent_effective'
        results.append(r)

    return results


def test2_randomization_inference(df, n_perms=1000):
    """
    Test 2: Randomization inference using placebo cutpoints.
    Randomly reassign the post indicator across years within the pre-period,
    compute the DID coefficient, and build a null distribution.
    """
    print("\n" + "="*70)
    print("TEST 2: Randomization Inference (placebo cutpoints in pre-period)")
    print("="*70)

    # Actual effect with Post >= 2022
    actual = run_did_with_cutpoint(df, 'entropy_score', 2022, cond_pt=True)
    print(f"  Actual β (Post>=2022, cond PT): {actual['beta']:+.6f}  p={actual['pval']:.4f}")

    # Pre-period only: 2016-2021
    df_pre = df[df['year'] <= 2021].copy()
    pre_years = sorted(df_pre['year'].unique())  # [2016,2017,2018,2019,2020,2021]
    print(f"  Pre-period years: {pre_years}")

    # Placebo cutpoints: every possible year in pre-period (excluding first and last)
    placebo_cutpoints = [2017, 2018, 2019, 2020, 2021]
    placebo_betas = []

    for cp in placebo_cutpoints:
        r = run_did_with_cutpoint(df_pre, 'entropy_score', cp, cond_pt=True)
        placebo_betas.append(r['beta'])
        stars = '***' if r['pval'] < 0.01 else '**' if r['pval'] < 0.05 else '*' if r['pval'] < 0.1 else ''
        print(f"  Placebo cutpoint {cp}: β={r['beta']:+.6f}{stars:3s}  p={r['pval']:.4f}")

    # Also do permutation-based RI: randomly shuffle the year labels
    print(f"\n  Running {n_perms} permutations (shuffling year labels within occupation)...")
    np.random.seed(42)
    perm_betas = []

    for i in range(n_perms):
        df_perm = df.copy()
        # Shuffle years within each occupation-industry cell
        df_perm['year_shuffled'] = df_perm.groupby(['isco08_4digit', 'industry_code'])['year'].transform(
            lambda x: np.random.permutation(x.values)
        )
        df_perm['post_perm'] = (df_perm['year_shuffled'] >= 2022).astype(int)
        df_perm['treat_x_post'] = df_perm['ai_exposure_ilo'] * df_perm['post_perm']
        # Also shuffle the baseline trends to match
        df_perm['sal_trend'] = df_perm['baseline_salary'] * df_perm['year_shuffled']
        df_perm['edu_trend'] = df_perm['baseline_edu'] * df_perm['year_shuffled']
        df_perm['exp_trend'] = df_perm['baseline_exp'] * df_perm['year_shuffled']

        formula = ("mean_entropy_score ~ treat_x_post + sal_trend + edu_trend + exp_trend "
                   "+ C(occ_fe) + C(year_fe) + C(ind_fe)")
        try:
            model = smf.wls(formula, data=df_perm, weights=df_perm['n_jobs']).fit(
                cov_type='cluster', cov_kwds={'groups': df_perm['cluster_id']}
            )
            perm_betas.append(model.params['treat_x_post'])
        except Exception:
            pass

        if (i + 1) % 100 == 0:
            print(f"    ... {i+1}/{n_perms} permutations done")

    perm_betas = np.array(perm_betas)
    ri_pval = np.mean(np.abs(perm_betas) >= np.abs(actual['beta']))
    print(f"\n  Permutation distribution: mean={perm_betas.mean():.6f}, sd={perm_betas.std():.6f}")
    print(f"  Actual β = {actual['beta']:.6f}")
    print(f"  RI p-value (two-sided): {ri_pval:.4f}")
    print(f"  % of permutations with |β| >= |actual|: {ri_pval*100:.1f}%")

    return {
        'actual_beta': actual['beta'],
        'placebo_betas': placebo_betas,
        'perm_betas': perm_betas.tolist(),
        'ri_pval': ri_pval,
    }


def test3_post_dose_response_time(df):
    """
    Test 3: Within post-period, does the effect grow with time?
    AIExpo × (year - 2021) for post-period years only.
    """
    print("\n" + "="*70)
    print("TEST 3: Post-period dose-response with time")
    print("="*70)

    # Year-by-year effects in post period (already in event study, but let's
    # also do a continuous time interaction)
    df = df.copy()
    df['post'] = (df['year'] >= 2022).astype(int)
    df['years_since_2021'] = np.maximum(0, df['year'] - 2021)
    df['treat_x_time'] = df['ai_exposure_ilo'] * df['years_since_2021']

    # Model: entropy ~ AIExpo×Post + AIExpo×years_since_2021 + controls
    # This tests if the effect grows linearly within the post period
    formula = ("mean_entropy_score ~ treat_x_time "
               "+ sal_trend + edu_trend + exp_trend "
               "+ C(occ_fe) + C(year_fe) + C(ind_fe)")

    model = smf.wls(formula, data=df, weights=df['n_jobs']).fit(
        cov_type='cluster', cov_kwds={'groups': df['cluster_id']}
    )
    b = model.params['treat_x_time']
    se = model.bse['treat_x_time']
    p = model.pvalues['treat_x_time']
    stars = '***' if p < 0.01 else '**' if p < 0.05 else '*' if p < 0.1 else ''
    print(f"  AIExpo × years_since_2021: β={b:+.6f}{stars:3s}  SE={se:.6f}  p={p:.4f}")
    print(f"  Interpretation: each additional year post-2021, the effect grows by {b:.4f} per unit AIExpo")

    # Also try with both trend + acceleration
    df['occ_trend'] = df['ai_exposure_ilo'] * df['year']
    formula2 = ("mean_entropy_score ~ occ_trend + treat_x_time "
                "+ sal_trend + edu_trend + exp_trend "
                "+ C(occ_fe) + C(year_fe) + C(ind_fe)")
    model2 = smf.wls(formula2, data=df, weights=df['n_jobs']).fit(
        cov_type='cluster', cov_kwds={'groups': df['cluster_id']}
    )
    b2 = model2.params['treat_x_time']
    se2 = model2.bse['treat_x_time']
    p2 = model2.pvalues['treat_x_time']
    bt = model2.params['occ_trend']
    pt = model2.pvalues['occ_trend']
    stars2 = '***' if p2 < 0.01 else '**' if p2 < 0.05 else '*' if p2 < 0.1 else ''
    starst = '***' if pt < 0.01 else '**' if pt < 0.05 else '*' if pt < 0.1 else ''
    print(f"\n  With pre-trend control:")
    print(f"    occ_trend (linear):   β={bt:+.6f}{starst:3s}  p={pt:.4f}")
    print(f"    treat_x_time (accel): β={b2:+.6f}{stars2:3s}  p={p2:.4f}")

    return {
        'beta_time': b, 'se_time': se, 'pval_time': p,
        'beta_time_ctrl': b2, 'se_time_ctrl': se2, 'pval_time_ctrl': p2,
        'beta_trend': bt, 'pval_trend': pt,
    }


def test4_cross_section_2024(df):
    """
    Test 4: Pure cross-sectional regression using only 2024 data.
    entropy ~ AIExpo + industry FE + baseline controls
    """
    print("\n" + "="*70)
    print("TEST 4: Cross-sectional analysis (2024 only)")
    print("="*70)

    df24 = df[df['year'] == 2024].copy()
    print(f"  2024 observations: {len(df24)}")

    # Simple cross-section
    formula = ("mean_entropy_score ~ ai_exposure_ilo "
               "+ baseline_salary + baseline_edu + baseline_exp "
               "+ C(ind_fe)")
    model = smf.wls(formula, data=df24, weights=df24['n_jobs']).fit(
        cov_type='cluster', cov_kwds={'groups': df24['cluster_id']}
    )
    b = model.params['ai_exposure_ilo']
    se = model.bse['ai_exposure_ilo']
    p = model.pvalues['ai_exposure_ilo']
    stars = '***' if p < 0.01 else '**' if p < 0.05 else '*' if p < 0.1 else ''
    print(f"  AIExpo (levels): β={b:+.4f}{stars:3s}  SE={se:.4f}  p={p:.4f}")
    print(f"  N={model.nobs:.0f}  R²={model.rsquared:.4f}")

    # Also for 2021 (last pre-period year) as comparison
    df21 = df[df['year'] == 2021].copy()
    model21 = smf.wls(formula, data=df21, weights=df21['n_jobs']).fit(
        cov_type='cluster', cov_kwds={'groups': df21['cluster_id']}
    )
    b21 = model21.params['ai_exposure_ilo']
    p21 = model21.pvalues['ai_exposure_ilo']
    stars21 = '***' if p21 < 0.01 else '**' if p21 < 0.05 else '*' if p21 < 0.1 else ''
    print(f"  AIExpo (2021 comparison): β={b21:+.4f}{stars21:3s}  p={p21:.4f}")
    print(f"  Difference 2024-2021: Δβ = {b - b21:+.4f}")

    # Effective topics too
    formula_eff = ("mean_ent_effective ~ ai_exposure_ilo "
                   "+ baseline_salary + baseline_edu + baseline_exp "
                   "+ C(ind_fe)")
    model_eff = smf.wls(formula_eff, data=df24, weights=df24['n_jobs']).fit(
        cov_type='cluster', cov_kwds={'groups': df24['cluster_id']}
    )
    be = model_eff.params['ai_exposure_ilo']
    pe = model_eff.pvalues['ai_exposure_ilo']
    starse = '***' if pe < 0.01 else '**' if pe < 0.05 else '*' if pe < 0.1 else ''
    print(f"\n  Effective topics (2024): β={be:+.4f}{starse:3s}  p={pe:.4f}")

    return {
        'beta_2024': b, 'se_2024': se, 'pval_2024': p,
        'beta_2021': b21, 'pval_2021': p21,
        'beta_eff_2024': be, 'pval_eff_2024': pe,
    }


if __name__ == '__main__':
    print("Loading data...")
    df = load_base_data()
    print(f"  Panel: {len(df)} cells, years {df['year'].min()}-{df['year'].max()}")

    # Test 1: Alternative post definitions
    res1 = test1_alternative_post_definitions(df)

    # Test 2: Randomization inference (use fewer perms for speed)
    res2 = test2_randomization_inference(df, n_perms=500)

    # Test 3: Post-period time dose-response
    res3 = test3_post_dose_response_time(df)

    # Test 4: Cross-section 2024
    res4 = test4_cross_section_2024(df)

    # Save results
    pd.DataFrame(res1).to_csv(RESULTS_DIR / "robustness_alt_post.csv", index=False)

    import json
    with open(RESULTS_DIR / "robustness_ri.json", 'w') as f:
        json.dump({
            'actual_beta': res2['actual_beta'],
            'ri_pval': res2['ri_pval'],
            'perm_mean': float(np.mean(res2['perm_betas'])),
            'perm_sd': float(np.std(res2['perm_betas'])),
            'placebo_betas': res2['placebo_betas'],
        }, f, indent=2)

    print("\n" + "="*70)
    print("ALL TESTS COMPLETE")
    print("="*70)
