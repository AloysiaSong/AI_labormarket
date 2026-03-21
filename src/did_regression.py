#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DID Regression: AI Exposure × Post-2022 Treatment on Skill Comprehensiveness

Model:
  Y_{o,i,t} = β (AIExposure_o × Post_t) + γ_o + δ_t + θ_i + ε_{o,i,t}

Where:
  Y = entropy_score (and robustness: ent_effective, rao_q, gini, n_sig_topics, etc.)
  AIExposure_o = ILO GenAI exposure score for ISCO-08 occupation o ∈ [0.08, 0.70]
  Post_t = 1 if year ≥ 2022
  γ_o = ISCO-08 occupation fixed effects
  δ_t = year fixed effects
  θ_i = industry fixed effects

Data: ind_isco_year_panel.csv (industry × ISCO-08 × year)
"""

import time
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from pathlib import Path

BASE = Path("/Users/yu/code/code2601/TY")
OUTPUT_DIR = BASE / "output"
ISCO_PANEL = OUTPUT_DIR / "ind_isco_year_panel.csv"
ISCO_BASELINE = OUTPUT_DIR / "isco_baseline_features.csv"
AI_INTENSITY_CSV = BASE / "data" / "ai_intensity_annual.csv"

RESULTS_DIR = OUTPUT_DIR / "regression"
RESULTS_DIR.mkdir(exist_ok=True)

MIN_CELL_SIZE = 30  # Minimum n_jobs per cell for reliable estimation


def load_data():
    """Load ISCO-08 level panel data (exposure already embedded)."""
    print("[1/4] Loading data...")

    # Load industry × ISCO × year panel (ai_exposure_ilo already included)
    df = pd.read_csv(ISCO_PANEL, encoding='utf-8-sig')
    print(f"  ind_isco_year panel: {len(df):,} cells")

    # Filter out small cells
    df = df[df['n_jobs'] >= MIN_CELL_SIZE].copy()
    print(f"  After n_jobs >= {MIN_CELL_SIZE} filter: {len(df):,} cells")

    df['ai_exposure_ilo'] = df['ai_exposure_ilo'].astype(float)

    # Create treatment variables
    df['year'] = df['year'].astype(int)
    df['post'] = (df['year'] >= 2022).astype(int)
    df['treat_x_post'] = df['ai_exposure_ilo'] * df['post']

    # For high/low binary treatment (robustness)
    median_exp = df['ai_exposure_ilo'].median()
    df['high_exposure'] = (df['ai_exposure_ilo'] >= median_exp).astype(int)
    df['high_x_post'] = df['high_exposure'] * df['post']

    # Occupation-specific linear time trend: AIExposure_o × t
    df['occ_trend'] = df['ai_exposure_ilo'] * df['year']

    # Time-varying treatment: AIExposure_o × AIIntensity_t (normalized 0-1)
    ai_int = pd.read_csv(AI_INTENSITY_CSV)
    ai_int['ai_intensity'] = ai_int['llm_models_cumulative'] / ai_int['llm_models_cumulative'].max()
    df = df.merge(ai_int[['year', 'ai_intensity']], on='year', how='left')
    df['treat_tv'] = df['ai_exposure_ilo'] * df['ai_intensity']
    print(f"  AI intensity (cumul. LLM models): {df['ai_intensity'].min():.3f} - {df['ai_intensity'].max():.3f}")

    # Spline: slope break at 2021 (post-2021 acceleration)
    # Spline_t = max(0, year - 2021): 0 for 2016-2021, then 1,2,3,4 for 2022-2025
    df['spline_2021'] = np.maximum(0, df['year'] - 2021)
    df['expo_spline_2021'] = df['ai_exposure_ilo'] * df['spline_2021']
    # Placebo splines at 2018 and 2019 (pre-LLM commercialization)
    df['spline_2018'] = np.maximum(0, df['year'] - 2018)
    df['expo_spline_2018'] = df['ai_exposure_ilo'] * df['spline_2018']
    df['spline_2019'] = np.maximum(0, df['year'] - 2019)
    df['expo_spline_2019'] = df['ai_exposure_ilo'] * df['spline_2019']

    # Quartile dummies for dose-response (Q1 = reference)
    quartile_cuts = df.groupby('isco08_4digit')['ai_exposure_ilo'].first()  # one value per occ
    q_labels = pd.qcut(quartile_cuts, 4, labels=['Q1', 'Q2', 'Q3', 'Q4'])
    occ_quartile = q_labels.to_dict()
    df['expo_quartile'] = df['isco08_4digit'].map(occ_quartile)
    for q in ['Q2', 'Q3', 'Q4']:
        df[f'{q}_x_post'] = ((df['expo_quartile'] == q) * df['post']).astype(int)
    print(f"  Exposure quartiles: {df['expo_quartile'].value_counts().to_dict()}")

    # Tercile dummies for sensitivity check (T1 = reference)
    t_labels = pd.qcut(quartile_cuts, 3, labels=['T1', 'T2', 'T3'])
    occ_tercile = t_labels.to_dict()
    df['expo_tercile'] = df['isco08_4digit'].map(occ_tercile)
    for t in ['T2', 'T3']:
        df[f'{t}_x_post'] = ((df['expo_tercile'] == t) * df['post']).astype(int)
    print(f"  Exposure terciles: {df['expo_tercile'].value_counts().to_dict()}")

    # Industry groups for heterogeneity
    # 国家统计局《数字经济及其核心产业统计分类（2021）》+ OECD知识密集型服务业
    DIGITAL_TECH = {'I'}                          # 数字技术产业（信息传输、软件和信息技术服务业）
    KNOWLEDGE_SVC = {'J', 'K', 'L', 'M', 'N', 'P', 'Q'}  # 知识密集型服务业（金融、房地产、租赁商务、科研、水利环保、教育、卫生）
    SECONDARY = {'B', 'C', 'D', 'E'}             # 第二产业（采矿、制造、电热燃气、水利环境）
    TRAD_SERVICES = {'F', 'G', 'H', 'O', 'R', 'S'}  # 传统服务业（建筑、批零、交运、居民服务、文体娱乐、其他）
    def _ind_group(code):
        c = str(code).strip()[0] if pd.notna(code) and str(code).strip() else ''
        if c in DIGITAL_TECH:
            return 'Digital_tech'
        elif c in KNOWLEDGE_SVC:
            return 'Knowledge_svc'
        elif c in SECONDARY:
            return 'Secondary'
        elif c in TRAD_SERVICES:
            return 'Trad_services'
        else:
            return 'Other'
    df['ind_group'] = df['industry_code'].apply(_ind_group)
    print(f"  Industry groups: {df['ind_group'].value_counts().to_dict()}")

    # Categorical FEs — use ISCO-08 code as occupation identifier
    df['occ_fe'] = pd.Categorical(df['isco08_4digit'])
    df['year_fe'] = pd.Categorical(df['year'])
    df['ind_fe'] = pd.Categorical(df['industry_code'])

    # Cluster SE by ISCO code (occupation-level clustering)
    df['cluster_id'] = df['isco08_4digit']

    # ── Baseline occupation features (for conditional parallel trends) ──
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
    print(f"  Baseline features merged (salary, edu, exp)")

    print(f"  Years: {sorted(df['year'].unique())}")
    print(f"  Industries: {df['industry_code'].nunique()}")
    print(f"  ISCO occupations: {df['isco08_4digit'].nunique()}")
    print(f"  Post=1 rows: {df['post'].sum():,} / {len(df):,}")
    print(f"  Median AI exposure: {median_exp:.2f}")
    print(f"  Exposure range: [{df['ai_exposure_ilo'].min():.2f}, {df['ai_exposure_ilo'].max():.2f}]")

    return df


def run_did_continuous(df, dep_var, label, occ_trend=False, cond_pt=False):
    """Run DID with continuous AI exposure treatment."""
    trend = " + occ_trend" if occ_trend else ""
    cpt = " + sal_trend + edu_trend + exp_trend" if cond_pt else ""
    formula = f"mean_{dep_var} ~ treat_x_post + C(occ_fe) + C(year_fe) + C(ind_fe){trend}{cpt}"
    try:
        model = smf.wls(formula, data=df, weights=df['n_jobs']).fit(
            cov_type='cluster', cov_kwds={'groups': df['cluster_id']}
        )
        beta = model.params['treat_x_post']
        se = model.bse['treat_x_post']
        pval = model.pvalues['treat_x_post']
        ci_lo, ci_hi = model.conf_int().loc['treat_x_post']
        n = model.nobs
        r2 = model.rsquared

        stars = '***' if pval < 0.01 else '**' if pval < 0.05 else '*' if pval < 0.1 else ''
        print(f"  {label:30s}  β={beta:+.6f}{stars:3s}  SE={se:.6f}  p={pval:.4f}  "
              f"CI=[{ci_lo:.6f}, {ci_hi:.6f}]  N={n:.0f}  R²={r2:.4f}")
        return {
            'dep_var': dep_var, 'label': label, 'beta': beta, 'se': se,
            'pval': pval, 'ci_lo': ci_lo, 'ci_hi': ci_hi, 'n': n, 'r2': r2, 'stars': stars
        }
    except Exception as e:
        print(f"  {label:30s}  ERROR: {e}")
        return None


def run_did_binary(df, dep_var, label):
    """Run DID with binary high/low AI exposure treatment."""
    formula = f"mean_{dep_var} ~ high_x_post + C(occ_fe) + C(year_fe) + C(ind_fe)"
    try:
        model = smf.wls(formula, data=df, weights=df['n_jobs']).fit(
            cov_type='cluster', cov_kwds={'groups': df['cluster_id']}
        )
        beta = model.params['high_x_post']
        se = model.bse['high_x_post']
        pval = model.pvalues['high_x_post']
        ci_lo, ci_hi = model.conf_int().loc['high_x_post']
        n = model.nobs
        r2 = model.rsquared

        stars = '***' if pval < 0.01 else '**' if pval < 0.05 else '*' if pval < 0.1 else ''
        print(f"  {label:30s}  β={beta:+.6f}{stars:3s}  SE={se:.6f}  p={pval:.4f}  "
              f"CI=[{ci_lo:.6f}, {ci_hi:.6f}]  N={n:.0f}  R²={r2:.4f}")
        return {
            'dep_var': dep_var, 'label': label, 'beta': beta, 'se': se,
            'pval': pval, 'ci_lo': ci_lo, 'ci_hi': ci_hi, 'n': n, 'r2': r2, 'stars': stars
        }
    except Exception as e:
        print(f"  {label:30s}  ERROR: {e}")
        return None


def run_event_study(df, dep_var, ind_trend=False, cond_pt=False):
    """Event study: interact AI exposure with year dummies (base = 2021)."""
    suffix = " (+ ind trend)" if ind_trend else (" (+ cond PT)" if cond_pt else "")
    print(f"\n  Event study for {dep_var}{suffix}:")
    df = df.copy()
    years = sorted(df['year'].unique())
    base_year = 2021

    # Create year × exposure interactions (omitting base year)
    for y in years:
        if y == base_year:
            continue
        df[f'exp_x_{y}'] = df['ai_exposure_ilo'] * (df['year'] == y).astype(int)

    interaction_terms = " + ".join(f'exp_x_{y}' for y in years if y != base_year)
    extra = " + C(ind_fe):year" if ind_trend else ""
    cpt = " + sal_trend + edu_trend + exp_trend" if cond_pt else ""
    formula = f"mean_{dep_var} ~ {interaction_terms} + C(occ_fe) + C(year_fe) + C(ind_fe){extra}{cpt}"

    try:
        model = smf.wls(formula, data=df, weights=df['n_jobs']).fit(
            cov_type='cluster', cov_kwds={'groups': df['cluster_id']}
        )

        results = []
        for y in years:
            if y == base_year:
                results.append({'year': y, 'beta': 0.0, 'se': 0.0, 'pval': 1.0,
                                'ci_lo': 0.0, 'ci_hi': 0.0})
                continue
            var = f'exp_x_{y}'
            beta = model.params[var]
            se = model.bse[var]
            pval = model.pvalues[var]
            ci_lo, ci_hi = model.conf_int().loc[var]
            stars = '***' if pval < 0.01 else '**' if pval < 0.05 else '*' if pval < 0.1 else ''
            print(f"    {y}: β={beta:+.6f}{stars:3s}  SE={se:.6f}  p={pval:.4f}")
            results.append({'year': y, 'beta': beta, 'se': se, 'pval': pval,
                            'ci_lo': ci_lo, 'ci_hi': ci_hi})

        return pd.DataFrame(results)
    except Exception as e:
        print(f"    ERROR: {e}")
        return None


def run_did_timevarying(df, dep_var, label, occ_trend=False, ind_trend=False):
    """Run DID with time-varying treatment: AIExposure_o × AIIntensity_t."""
    extra = ""
    if occ_trend:
        extra += " + occ_trend"
    if ind_trend:
        extra += " + C(ind_fe):year"
    formula = f"mean_{dep_var} ~ treat_tv + C(occ_fe) + C(year_fe) + C(ind_fe){extra}"
    try:
        model = smf.wls(formula, data=df, weights=df['n_jobs']).fit(
            cov_type='cluster', cov_kwds={'groups': df['cluster_id']}
        )
        beta = model.params['treat_tv']
        se = model.bse['treat_tv']
        pval = model.pvalues['treat_tv']
        ci_lo, ci_hi = model.conf_int().loc['treat_tv']
        n = model.nobs
        r2 = model.rsquared

        stars = '***' if pval < 0.01 else '**' if pval < 0.05 else '*' if pval < 0.1 else ''
        print(f"  {label:30s}  β={beta:+.6f}{stars:3s}  SE={se:.6f}  p={pval:.4f}  "
              f"CI=[{ci_lo:.6f}, {ci_hi:.6f}]  N={n:.0f}  R²={r2:.4f}")
        return {
            'dep_var': dep_var, 'label': label, 'beta': beta, 'se': se,
            'pval': pval, 'ci_lo': ci_lo, 'ci_hi': ci_hi, 'n': n, 'r2': r2, 'stars': stars
        }
    except Exception as e:
        print(f"  {label:30s}  ERROR: {e}")
        return None


def run_detrended_did(df, dep_var, label):
    """
    Pre-trend detrending: estimate AIExposure×t trend from pre-period only,
    then test whether post-period deviates from the extrapolated trend.

    Step 1: On pre-period (year < 2022), regress Y on occ_trend + FEs → get trend coef
    Step 2: On full sample, subtract predicted trend: Y_detrended = Y - β_trend * occ_trend
    Step 3: Run standard DID on Y_detrended
    """
    pre = df[df['year'] < 2022].copy()
    formula_pre = f"mean_{dep_var} ~ occ_trend + C(occ_fe) + C(year_fe) + C(ind_fe)"
    try:
        model_pre = smf.wls(formula_pre, data=pre, weights=pre['n_jobs']).fit()
        trend_coef = model_pre.params['occ_trend']

        # Detrend full sample
        df_dt = df.copy()
        df_dt[f'mean_{dep_var}'] = df_dt[f'mean_{dep_var}'] - trend_coef * df_dt['occ_trend']

        # Run DID on detrended Y
        formula_did = f"mean_{dep_var} ~ treat_x_post + C(occ_fe) + C(year_fe) + C(ind_fe)"
        model = smf.wls(formula_did, data=df_dt, weights=df_dt['n_jobs']).fit(
            cov_type='cluster', cov_kwds={'groups': df_dt['cluster_id']}
        )
        beta = model.params['treat_x_post']
        se = model.bse['treat_x_post']
        pval = model.pvalues['treat_x_post']
        ci_lo, ci_hi = model.conf_int().loc['treat_x_post']
        n = model.nobs
        r2 = model.rsquared

        stars = '***' if pval < 0.01 else '**' if pval < 0.05 else '*' if pval < 0.1 else ''
        print(f"  {label:30s}  β={beta:+.6f}{stars:3s}  SE={se:.6f}  p={pval:.4f}  "
              f"trend_coef={trend_coef:+.6f}  N={n:.0f}  R²={r2:.4f}")
        return {
            'dep_var': dep_var, 'label': label, 'beta': beta, 'se': se,
            'pval': pval, 'ci_lo': ci_lo, 'ci_hi': ci_hi, 'n': n, 'r2': r2,
            'stars': stars, 'trend_coef': trend_coef
        }
    except Exception as e:
        print(f"  {label:30s}  ERROR: {e}")
        return None


def run_did_quartile(df, dep_var, label):
    """
    Quartile dose-response: Q1 as reference, estimate β_Q2, β_Q3, β_Q4.
    Model: Y = Σ_{q=2}^{4} β_q (Quartile_q × Post) + cond_PT + FEs
    """
    formula = (f"mean_{dep_var} ~ Q2_x_post + Q3_x_post + Q4_x_post "
               f"+ sal_trend + edu_trend + exp_trend "
               f"+ C(occ_fe) + C(year_fe) + C(ind_fe)")
    try:
        model = smf.wls(formula, data=df, weights=df['n_jobs']).fit(
            cov_type='cluster', cov_kwds={'groups': df['cluster_id']}
        )
        result = {'dep_var': dep_var, 'label': label}
        print(f"  {label}:")
        for q in ['Q2', 'Q3', 'Q4']:
            var = f'{q}_x_post'
            b = model.params[var]
            se = model.bse[var]
            p = model.pvalues[var]
            ci_lo, ci_hi = model.conf_int().loc[var]
            stars = '***' if p < 0.01 else '**' if p < 0.05 else '*' if p < 0.1 else ''
            print(f"    {q}: β={b:+.6f}{stars:3s}  SE={se:.6f}  p={p:.4f}  CI=[{ci_lo:.6f}, {ci_hi:.6f}]")
            result[f'beta_{q}'] = b
            result[f'se_{q}'] = se
            result[f'pval_{q}'] = p
            result[f'ci_lo_{q}'] = ci_lo
            result[f'ci_hi_{q}'] = ci_hi
        result['n'] = model.nobs
        result['r2'] = model.rsquared
        print(f"    N={model.nobs:.0f}  R²={model.rsquared:.4f}")
        return result
    except Exception as e:
        print(f"  {label}  ERROR: {e}")
        return None


def run_did_tercile(df, dep_var, label):
    """
    Tercile dose-response: T1 as reference, estimate β_T2, β_T3.
    Model: Y = Σ_{t=2}^{3} β_t (Tercile_t × Post) + cond_PT + FEs
    """
    formula = (f"mean_{dep_var} ~ T2_x_post + T3_x_post "
               f"+ sal_trend + edu_trend + exp_trend "
               f"+ C(occ_fe) + C(year_fe) + C(ind_fe)")
    try:
        model = smf.wls(formula, data=df, weights=df['n_jobs']).fit(
            cov_type='cluster', cov_kwds={'groups': df['cluster_id']}
        )
        result = {'dep_var': dep_var, 'label': label}
        print(f"  {label}:")
        for t in ['T2', 'T3']:
            var = f'{t}_x_post'
            b = model.params[var]
            se = model.bse[var]
            p = model.pvalues[var]
            ci_lo, ci_hi = model.conf_int().loc[var]
            stars = '***' if p < 0.01 else '**' if p < 0.05 else '*' if p < 0.1 else ''
            print(f"    {t}: β={b:+.6f}{stars:3s}  SE={se:.6f}  p={p:.4f}  CI=[{ci_lo:.6f}, {ci_hi:.6f}]")
            result[f'beta_{t}'] = b
            result[f'se_{t}'] = se
            result[f'pval_{t}'] = p
            result[f'ci_lo_{t}'] = ci_lo
            result[f'ci_hi_{t}'] = ci_hi
        result['n'] = model.nobs
        result['r2'] = model.rsquared
        print(f"    N={model.nobs:.0f}  R²={model.rsquared:.4f}")
        return result
    except Exception as e:
        print(f"  {label}  ERROR: {e}")
        return None


def print_baseline_by_quartile(df):
    """Print pre-period (2016-2021) mean entropy by exposure quartile."""
    pre = df[df['year'] < 2022]
    print("\n  Baseline (pre-2022) mean entropy by exposure quartile:")
    for q in ['Q1', 'Q2', 'Q3', 'Q4']:
        sub = pre[pre['expo_quartile'] == q]
        m = np.average(sub['mean_entropy_score'], weights=sub['n_jobs'])
        m_eff = np.average(sub['mean_ent_effective'], weights=sub['n_jobs'])
        n_occ = sub['isco08_4digit'].nunique()
        print(f"    {q}: entropy={m:.4f}  eff_topics={m_eff:.2f}  n_occ={n_occ}  n_cells={len(sub)}")
    # Also by tercile
    print("  Baseline (pre-2022) mean entropy by exposure tercile:")
    for t in ['T1', 'T2', 'T3']:
        sub = pre[pre['expo_tercile'] == t]
        m = np.average(sub['mean_entropy_score'], weights=sub['n_jobs'])
        m_eff = np.average(sub['mean_ent_effective'], weights=sub['n_jobs'])
        n_occ = sub['isco08_4digit'].nunique()
        print(f"    {t}: entropy={m:.4f}  eff_topics={m_eff:.2f}  n_occ={n_occ}  n_cells={len(sub)}")


def run_did_industry_group(df, dep_var, label):
    """
    Industry group heterogeneity: triple interaction AIExpo × Post × IndustryGroup.
    Groups: Digital_tech (I), Knowledge_svc (J/K/L/M/N/P/Q), Secondary (B/C/D/E),
            Trad_services (F/G/H/O/R/S), Other (everything else).
    Reference: Other. With cond PT controls.
    """
    formula = (f"mean_{dep_var} ~ treat_x_post "
               f"+ treat_x_post:C(ind_group, Treatment(reference='Other')) "
               f"+ sal_trend + edu_trend + exp_trend "
               f"+ C(occ_fe) + C(year_fe) + C(ind_fe)")
    try:
        model = smf.wls(formula, data=df, weights=df['n_jobs']).fit(
            cov_type='cluster', cov_kwds={'groups': df['cluster_id']}
        )
        result = {'dep_var': dep_var, 'label': label}
        # Base effect (Other group)
        b0 = model.params['treat_x_post']
        se0 = model.bse['treat_x_post']
        p0 = model.pvalues['treat_x_post']
        stars0 = '***' if p0 < 0.01 else '**' if p0 < 0.05 else '*' if p0 < 0.1 else ''
        print(f"  {label}:")
        print(f"    Base (Other):     β={b0:+.6f}{stars0:3s}  SE={se0:.6f}  p={p0:.4f}")
        result['beta_base'] = b0
        result['se_base'] = se0
        result['pval_base'] = p0

        # Differential effects for each group
        for grp in ['Digital_tech', 'Knowledge_svc', 'Secondary', 'Trad_services']:
            # statsmodels names: treat_x_post:C(ind_group, Treatment(reference='Other'))[T.grp]
            var = f"treat_x_post:C(ind_group, Treatment(reference='Other'))[T.{grp}]"
            if var in model.params:
                b = model.params[var]
                se = model.bse[var]
                p = model.pvalues[var]
                stars = '***' if p < 0.01 else '**' if p < 0.05 else '*' if p < 0.1 else ''
                total = b0 + b
                print(f"    Δ {grp:20s}: β={b:+.6f}{stars:3s}  SE={se:.6f}  p={p:.4f}  (total={total:+.6f})")
                result[f'beta_delta_{grp}'] = b
                result[f'se_delta_{grp}'] = se
                result[f'pval_delta_{grp}'] = p
                result[f'beta_total_{grp}'] = total
            else:
                print(f"    Δ {grp:20s}: NOT FOUND in model")
        result['n'] = model.nobs
        result['r2'] = model.rsquared
        print(f"    N={model.nobs:.0f}  R²={model.rsquared:.4f}")
        return result
    except Exception as e:
        print(f"  {label}  ERROR: {e}")
        import traceback; traceback.print_exc()
        return None


def run_did_spline(df, dep_var, breakyear, label):
    """
    Spline DID: test whether AI-exposed occupations show a slope break at breakyear.

    Model: Y = β₁(AIExpo × t) + β₂(AIExpo × Spline_t) + γ_o + δ_t + θ_i + ε
      - β₁ absorbs differential linear pre-trends
      - β₂ captures post-breakyear slope acceleration
      - Spline_t = max(0, year - breakyear)
    """
    spline_var = f'expo_spline_{breakyear}'
    formula = (f"mean_{dep_var} ~ occ_trend + {spline_var} "
               f"+ C(occ_fe) + C(year_fe) + C(ind_fe)")
    try:
        model = smf.wls(formula, data=df, weights=df['n_jobs']).fit(
            cov_type='cluster', cov_kwds={'groups': df['cluster_id']}
        )
        # β₁: linear trend
        b1 = model.params['occ_trend']
        se1 = model.bse['occ_trend']
        p1 = model.pvalues['occ_trend']
        # β₂: slope break (the key coefficient)
        b2 = model.params[spline_var]
        se2 = model.bse[spline_var]
        p2 = model.pvalues[spline_var]
        ci_lo, ci_hi = model.conf_int().loc[spline_var]
        n = model.nobs
        r2 = model.rsquared

        stars1 = '***' if p1 < 0.01 else '**' if p1 < 0.05 else '*' if p1 < 0.1 else ''
        stars2 = '***' if p2 < 0.01 else '**' if p2 < 0.05 else '*' if p2 < 0.1 else ''
        print(f"  {label:35s}  β₁(trend)={b1:+.6f}{stars1:3s}  "
              f"β₂(spline)={b2:+.6f}{stars2:3s}  SE₂={se2:.6f}  p₂={p2:.4f}  "
              f"CI₂=[{ci_lo:.6f}, {ci_hi:.6f}]  N={n:.0f}  R²={r2:.4f}")
        return {
            'dep_var': dep_var, 'label': label, 'breakyear': breakyear,
            'beta_trend': b1, 'se_trend': se1, 'pval_trend': p1,
            'beta_spline': b2, 'se_spline': se2, 'pval_spline': p2,
            'ci_lo': ci_lo, 'ci_hi': ci_hi, 'n': n, 'r2': r2,
            'stars_trend': stars1, 'stars_spline': stars2
        }
    except Exception as e:
        print(f"  {label:35s}  ERROR: {e}")
        return None


def main():
    t0 = time.time()
    df = load_data()

    # ===== Main DID: Continuous Treatment =====
    print("\n[2/4] Main DID results (continuous AI exposure × Post):")
    print("=" * 110)
    dep_vars = [
        ("entropy_score",       "Shannon Entropy"),
        ("ent_effective",       "Effective # Topics"),
        ("rao_q",               "Rao Quadratic Entropy"),
        ("gini",                "Gini Index (↑=concentrated)"),
        ("n_sig_topics",        "# Significant Topics"),
        ("tail_mass_ratio",     "Tail Mass Ratio"),
        ("hhi_score",           "HHI (↑=concentrated)"),
        ("dominant_topic_prob", "Dominant Topic Prob"),
    ]

    results_cont = []
    for dv, label in dep_vars:
        r = run_did_continuous(df, dv, label)
        if r:
            results_cont.append(r)

    # ===== Robustness: Occupation-specific linear time trend =====
    print("\n[2b/4] DID with occupation-specific linear time trend:")
    print("=" * 110)
    results_trend = []
    for dv, label in dep_vars:
        r = run_did_continuous(df, dv, f"{label} (+trend)", occ_trend=True)
        if r:
            results_trend.append(r)

    # ===== Robustness: Binary Treatment =====
    print("\n[3/4] Robustness: Binary high/low AI exposure × Post:")
    print("=" * 110)
    results_bin = []
    for dv, label in dep_vars:
        r = run_did_binary(df, dv, f"{label} (binary)")
        if r:
            results_bin.append(r)

    # ===== Time-varying treatment: AIExposure × AIIntensity =====
    print("\n[3c/4] Time-varying treatment (AIExposure_o × Cumulative LLM Models_t):")
    print("=" * 110)
    results_tv = []
    for dv, label in dep_vars:
        r = run_did_timevarying(df, dv, f"{label} (time-var)")
        if r:
            results_tv.append(r)

    # ===== Time-varying + linear trend control =====
    print("\n[3d/4] Time-varying treatment + occupation-specific linear trend:")
    print("=" * 110)
    results_tv_trend = []
    for dv, label in dep_vars:
        r = run_did_timevarying(df, dv, f"{label} (tv+trend)", occ_trend=True)
        if r:
            results_tv_trend.append(r)

    # ===== Time-varying + industry-specific linear trend =====
    print("\n[3e/4] Time-varying treatment + industry-specific linear trend:")
    print("=" * 110)
    results_tv_ind_trend = []
    for dv, label in dep_vars:
        r = run_did_timevarying(df, dv, f"{label} (tv+ind_trend)", ind_trend=True)
        if r:
            results_tv_ind_trend.append(r)

    # ===== Robustness: Pre-trend detrending =====
    print("\n[3b/4] DID with pre-period detrending (trend estimated from 2016-2021 only):")
    print("=" * 110)
    results_detrend = []
    for dv, label in dep_vars:
        r = run_detrended_did(df, dv, f"{label} (detrend)")
        if r:
            results_detrend.append(r)

    # ===== Conditional Parallel Trends: baseline features × year =====
    print("\n[3f/4] Conditional Parallel Trends (+ salary×year + edu×year + exp×year):")
    print("=" * 110)
    results_cpt = []
    for dv, label in dep_vars:
        r = run_did_continuous(df, dv, f"{label} (cond PT)", cond_pt=True)
        if r:
            results_cpt.append(r)

    # ===== Spline DID: slope break at 2021 (main test) =====
    print("\n[3g/4] Spline DID: slope break at 2021 (AIExpo×t + AIExpo×Spline₂₀₂₁):")
    print("=" * 130)
    results_spline_2021 = []
    for dv, label in dep_vars:
        r = run_did_spline(df, dv, 2021, f"{label} (spline@2021)")
        if r:
            results_spline_2021.append(r)

    # ===== Placebo spline: slope break at 2018 =====
    print("\n[3h/4] Placebo spline: slope break at 2018 (pre-LLM):")
    print("=" * 130)
    results_spline_2018 = []
    for dv, label in dep_vars:
        r = run_did_spline(df, dv, 2018, f"{label} (spline@2018)")
        if r:
            results_spline_2018.append(r)

    # ===== Placebo spline: slope break at 2019 =====
    print("\n[3i/4] Placebo spline: slope break at 2019 (pre-LLM):")
    print("=" * 130)
    results_spline_2019 = []
    for dv, label in dep_vars:
        r = run_did_spline(df, dv, 2019, f"{label} (spline@2019)")
        if r:
            results_spline_2019.append(r)

    # ===== Baseline diagnostic: entropy by quartile/tercile =====
    print("\n[5] Heterogeneity analysis")
    print_baseline_by_quartile(df)

    # ===== Heterogeneity: Quartile dose-response (with cond PT) =====
    print("\n[5a] Quartile dose-response (Q1=ref, with cond PT controls):")
    print("=" * 110)
    results_quartile = []
    for dv, label in dep_vars:
        r = run_did_quartile(df, dv, label)
        if r:
            results_quartile.append(r)

    # ===== Heterogeneity: Tercile dose-response sensitivity (with cond PT) =====
    print("\n[5a-sens] Tercile dose-response sensitivity (T1=ref, with cond PT):")
    print("=" * 110)
    results_tercile = []
    for dv, label in dep_vars:
        r = run_did_tercile(df, dv, label)
        if r:
            results_tercile.append(r)

    # ===== Heterogeneity: Industry group triple interaction (with cond PT) =====
    print("\n[5b] Industry group heterogeneity (AIExpo×Post×IndGroup, with cond PT):")
    print("=" * 110)
    results_ind_group = []
    for dv, label in dep_vars:
        r = run_did_industry_group(df, dv, label)
        if r:
            results_ind_group.append(r)

    # ===== Event Study =====
    print("\n[4/4] Event study (parallel trends check):")
    print("=" * 90)
    es_entropy = run_event_study(df, "entropy_score")
    es_effective = run_event_study(df, "ent_effective")

    # ===== Event Study with industry trend control =====
    print("\n[4b/4] Event study + industry linear trend:")
    print("=" * 90)
    es_entropy_ind = run_event_study(df, "entropy_score", ind_trend=True)
    es_effective_ind = run_event_study(df, "ent_effective", ind_trend=True)

    # ===== Event Study with conditional PT =====
    print("\n[4c/4] Event study + conditional parallel trends controls:")
    print("=" * 90)
    es_entropy_cpt = run_event_study(df, "entropy_score", cond_pt=True)
    es_effective_cpt = run_event_study(df, "ent_effective", cond_pt=True)

    # ===== Save results =====
    if results_cont:
        pd.DataFrame(results_cont).to_csv(RESULTS_DIR / "did_continuous.csv", index=False)
    if results_bin:
        pd.DataFrame(results_bin).to_csv(RESULTS_DIR / "did_binary.csv", index=False)
    if results_trend:
        pd.DataFrame(results_trend).to_csv(RESULTS_DIR / "did_trend.csv", index=False)
    if results_detrend:
        pd.DataFrame(results_detrend).to_csv(RESULTS_DIR / "did_detrend.csv", index=False)
    if results_tv:
        pd.DataFrame(results_tv).to_csv(RESULTS_DIR / "did_timevarying.csv", index=False)
    if results_tv_trend:
        pd.DataFrame(results_tv_trend).to_csv(RESULTS_DIR / "did_timevarying_trend.csv", index=False)
    if results_tv_ind_trend:
        pd.DataFrame(results_tv_ind_trend).to_csv(RESULTS_DIR / "did_timevarying_ind_trend.csv", index=False)
    if es_entropy is not None:
        es_entropy.to_csv(RESULTS_DIR / "event_study_entropy.csv", index=False)
    if es_entropy_ind is not None:
        es_entropy_ind.to_csv(RESULTS_DIR / "event_study_entropy_ind_trend.csv", index=False)
    if es_effective_ind is not None:
        es_effective_ind.to_csv(RESULTS_DIR / "event_study_effective_ind_trend.csv", index=False)
    if es_effective is not None:
        es_effective.to_csv(RESULTS_DIR / "event_study_effective.csv", index=False)
    if results_cpt:
        pd.DataFrame(results_cpt).to_csv(RESULTS_DIR / "did_cond_pt.csv", index=False)
    if es_entropy_cpt is not None:
        es_entropy_cpt.to_csv(RESULTS_DIR / "event_study_entropy_cond_pt.csv", index=False)
    if es_effective_cpt is not None:
        es_effective_cpt.to_csv(RESULTS_DIR / "event_study_effective_cond_pt.csv", index=False)
    if results_spline_2021:
        pd.DataFrame(results_spline_2021).to_csv(RESULTS_DIR / "did_spline_2021.csv", index=False)
    if results_spline_2018:
        pd.DataFrame(results_spline_2018).to_csv(RESULTS_DIR / "did_spline_2018.csv", index=False)
    if results_spline_2019:
        pd.DataFrame(results_spline_2019).to_csv(RESULTS_DIR / "did_spline_2019.csv", index=False)
    if results_quartile:
        pd.DataFrame(results_quartile).to_csv(RESULTS_DIR / "did_quartile_dose_response.csv", index=False)
    if results_tercile:
        pd.DataFrame(results_tercile).to_csv(RESULTS_DIR / "did_tercile_dose_response.csv", index=False)
    if results_ind_group:
        pd.DataFrame(results_ind_group).to_csv(RESULTS_DIR / "did_industry_group.csv", index=False)

    elapsed = time.time() - t0
    print(f"\nAll results saved to {RESULTS_DIR}/")
    print(f"Total time: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
