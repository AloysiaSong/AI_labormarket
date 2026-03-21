#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Within-Occupation Heterogeneity Analysis
=========================================

检验职业内部岗位技能需求的异质性是否在AI冲击后上升。

思路：
  - 对每个 ISCO × 年份，计算岗位间技能 profile 的离散程度
  - 用现有 ind_isco_year_panel.csv 的 cell-level 统计量（mean, sd, n）
    通过 pooled variance 公式聚合到 ISCO × year 层面
  - 以职业内异质性作为因变量，跑与主分析相同的 DID

因变量：
  - sd_entropy:  ISCO×year 内岗位 Shannon 熵的标准差
  - sd_effective: ISCO×year 内岗位有效主题数的标准差
  - cv_entropy:  变异系数 (sd / mean)，消除水平效应

Model:
  Heterogeneity_{o,t} = β (AIExposure_o × Post_t) + γ_o + δ_t + controls + ε_{o,t}

Output: output/regression/within_occ_heterogeneity.csv
"""

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from pathlib import Path

BASE = Path("/Users/yu/code/code2601/TY")
OUTPUT_DIR = BASE / "output"
ISCO_PANEL = OUTPUT_DIR / "ind_isco_year_panel.csv"
ISCO_BASELINE = OUTPUT_DIR / "isco_baseline_features.csv"
RESULTS_DIR = OUTPUT_DIR / "regression"
RESULTS_DIR.mkdir(exist_ok=True)

MIN_CELL_SIZE = 30
MIN_OCC_JOBS = 50  # 聚合到ISCO×year后的最低岗位数


def build_occ_year_panel():
    """
    从 industry × ISCO × year 面板聚合到 ISCO × year 面板，
    计算职业内异质性指标。

    Pooled variance 公式:
      pooled_var = [Σ n_i (sd_i² + mean_i²)] / N  -  grand_mean²
    其中 i 为同一ISCO×year内的不同行业cell。
    """
    print("[1/3] Building ISCO × year panel with within-occupation heterogeneity...")

    df = pd.read_csv(ISCO_PANEL, encoding='utf-8-sig')
    df = df[df['n_jobs'] >= MIN_CELL_SIZE].copy()
    df['isco08_4digit'] = df['isco08_4digit'].astype(str)
    df['year'] = df['year'].astype(int)
    print(f"  Cell-level panel: {len(df):,} cells")

    # 对每个 ISCO × year 聚合
    metrics = ['entropy_score', 'ent_effective']
    records = []

    for (isco, year), grp in df.groupby(['isco08_4digit', 'year']):
        n_total = grp['n_jobs'].sum()
        if n_total < MIN_OCC_JOBS:
            continue

        rec = {
            'isco08_4digit': isco,
            'year': year,
            'ai_exposure_ilo': grp['ai_exposure_ilo'].iloc[0],
            'n_jobs': n_total,
            'n_cells': len(grp),
        }

        for m in metrics:
            mean_col = f'mean_{m}'
            sd_col = f'sd_{m}'

            ns = grp['n_jobs'].values
            means = grp[mean_col].values
            sds = grp[sd_col].values

            # Grand mean (weighted)
            grand_mean = np.average(means, weights=ns)

            # Pooled variance: within-cell + between-cell components
            # Var_total = E[Var_within] + Var[means]
            # = Σ(n_i * sd_i²) / N + Σ(n_i * (mean_i - grand_mean)²) / N
            var_within = np.sum(ns * sds**2) / n_total
            var_between = np.sum(ns * (means - grand_mean)**2) / n_total
            pooled_sd = np.sqrt(var_within + var_between)

            rec[f'mean_{m}'] = grand_mean
            rec[f'sd_{m}'] = pooled_sd
            rec[f'cv_{m}'] = pooled_sd / grand_mean if grand_mean > 0 else np.nan

        records.append(rec)

    panel = pd.DataFrame(records)
    print(f"  ISCO × year panel: {len(panel):,} obs "
          f"({panel['isco08_4digit'].nunique()} occupations, "
          f"{sorted(panel['year'].unique())})")

    # 创建分析变量
    panel['post'] = (panel['year'] >= 2022).astype(int)
    panel['treat_x_post'] = panel['ai_exposure_ilo'] * panel['post']
    panel['occ_fe'] = pd.Categorical(panel['isco08_4digit'])
    panel['year_fe'] = pd.Categorical(panel['year'])
    panel['cluster_id'] = panel['isco08_4digit']

    # 合并基线特征 (conditional parallel trends)
    feat = pd.read_csv(ISCO_BASELINE, encoding='utf-8-sig')
    feat['isco08_4digit'] = feat['isco08_4digit'].astype(str)
    feat['year'] = feat['year'].astype(int)
    baseline = feat[feat['year'].isin([2016, 2017, 2018])].groupby('isco08_4digit').agg(
        base_salary=('mean_salary', 'mean'),
        base_edu=('mean_edu', 'mean'),
        base_exp=('mean_exp', 'mean'),
    ).reset_index()
    panel = panel.merge(baseline, on='isco08_4digit', how='left')
    for col in ['base_salary', 'base_edu', 'base_exp']:
        panel[col] = (panel[col] - panel[col].mean()) / panel[col].std()
    panel['sal_trend'] = panel['base_salary'] * panel['year']
    panel['edu_trend'] = panel['base_edu'] * panel['year']
    panel['exp_trend'] = panel['base_exp'] * panel['year']
    print(f"  Baseline features merged")

    return panel


def run_did(panel, dep_var, label, cond_pt=False):
    """Run DID regression on heterogeneity outcome."""
    cpt = " + sal_trend + edu_trend + exp_trend" if cond_pt else ""
    formula = f"{dep_var} ~ treat_x_post + C(occ_fe) + C(year_fe){cpt}"

    model = smf.wls(formula, data=panel, weights=panel['n_jobs'])
    res = model.fit(cov_type='cluster', cov_kwds={'groups': panel['cluster_id']})

    beta = res.params['treat_x_post']
    se = res.bse['treat_x_post']
    pval = res.pvalues['treat_x_post']
    ci = res.conf_int().loc['treat_x_post']
    stars = '***' if pval < 0.01 else '**' if pval < 0.05 else '*' if pval < 0.1 else ''

    return {
        'dep_var': dep_var,
        'label': label,
        'cond_pt': cond_pt,
        'beta': beta,
        'se': se,
        'pval': pval,
        'ci_lo': ci[0],
        'ci_hi': ci[1],
        'n': len(panel),
        'n_occ': panel['isco08_4digit'].nunique(),
        'r2': res.rsquared,
        'stars': stars,
    }


def run_event_study(panel, dep_var, label, cond_pt=False):
    """Event study: year × exposure interactions (ref year = 2021)."""
    years = sorted(panel['year'].unique())
    ref_year = 2021
    for y in years:
        if y != ref_year:
            panel[f'expo_x_{y}'] = panel['ai_exposure_ilo'] * (panel['year'] == y).astype(int)

    year_vars = " + ".join(f'expo_x_{y}' for y in years if y != ref_year)
    cpt = " + sal_trend + edu_trend + exp_trend" if cond_pt else ""
    formula = f"{dep_var} ~ {year_vars} + C(occ_fe) + C(year_fe){cpt}"

    model = smf.wls(formula, data=panel, weights=panel['n_jobs'])
    res = model.fit(cov_type='cluster', cov_kwds={'groups': panel['cluster_id']})

    rows = []
    for y in years:
        if y == ref_year:
            rows.append({'year': y, 'beta': 0, 'se': 0, 'pval': 1, 'ci_lo': 0, 'ci_hi': 0})
        else:
            v = f'expo_x_{y}'
            rows.append({
                'year': y,
                'beta': res.params[v],
                'se': res.bse[v],
                'pval': res.pvalues[v],
                'ci_lo': res.conf_int().loc[v, 0],
                'ci_hi': res.conf_int().loc[v, 1],
            })
    return pd.DataFrame(rows)


def main():
    panel = build_occ_year_panel()

    # ── Descriptive stats ──
    print("\n[2/3] Descriptive statistics...")
    for period, label in [(0, 'Pre (2016-2021)'), (1, 'Post (2022-2025)')]:
        sub = panel[panel['post'] == period]
        print(f"\n  {label}:")
        for m in ['sd_entropy_score', 'sd_ent_effective', 'cv_entropy_score', 'cv_ent_effective']:
            vals = sub[m].dropna()
            print(f"    {m:25s}: mean={vals.mean():.4f}  sd={vals.std():.4f}  median={vals.median():.4f}")

    # 按AI暴露度三分位比较
    expo_med = panel.groupby('isco08_4digit')['ai_exposure_ilo'].first()
    tercile_map = pd.qcut(expo_med, 3, labels=['T1_low', 'T2_mid', 'T3_high']).to_dict()
    panel['expo_tercile'] = panel['isco08_4digit'].map(tercile_map)

    print("\n  Within-occ SD of entropy by exposure tercile × period:")
    summary = panel.groupby(['expo_tercile', 'post']).agg(
        sd_entropy_mean=('sd_entropy_score', 'mean'),
        sd_effective_mean=('sd_ent_effective', 'mean'),
        n=('n_jobs', 'sum'),
    ).round(4)
    print(summary.to_string())

    # ── DID regressions ──
    print("\n[3/3] DID regressions on within-occupation heterogeneity...")
    outcomes = [
        ('sd_entropy_score', 'SD of Shannon Entropy'),
        ('sd_ent_effective', 'SD of Effective Topics'),
        ('cv_entropy_score', 'CV of Shannon Entropy'),
        ('cv_ent_effective', 'CV of Effective Topics'),
    ]

    results = []
    for dep_var, label in outcomes:
        for cond_pt in [False, True]:
            spec = "Cond. PT" if cond_pt else "Base"
            r = run_did(panel, dep_var, label, cond_pt=cond_pt)
            results.append(r)
            print(f"  [{spec:8s}] {label:30s}: β={r['beta']:+.5f} (SE={r['se']:.5f}) p={r['pval']:.4f} {r['stars']}")

    results_df = pd.DataFrame(results)
    out_path = RESULTS_DIR / "within_occ_heterogeneity.csv"
    results_df.to_csv(out_path, index=False)
    print(f"\n  Saved: {out_path}")

    # ── Event study for main outcome ──
    print("\n  Event study: SD of entropy (cond PT)...")
    es = run_event_study(panel, 'sd_entropy_score', 'SD of Shannon Entropy', cond_pt=True)
    es_path = RESULTS_DIR / "event_study_within_occ_heterogeneity.csv"
    es.to_csv(es_path, index=False)
    print(es.to_string(index=False))
    print(f"  Saved: {es_path}")

    # ── Summary ──
    print("\n" + "=" * 60)
    print("  Within-Occupation Heterogeneity Analysis Complete")
    print("=" * 60)
    main_r = [r for r in results if r['dep_var'] == 'sd_entropy_score' and r['cond_pt']]
    if main_r:
        r = main_r[0]
        if r['pval'] < 0.05:
            print(f"  结论: 职业内异质性显著上升 (β={r['beta']:+.5f}, p={r['pval']:.4f})")
            print(f"  → 综合化部分发生在职业内部，同一职业内岗位要求在分化")
        else:
            print(f"  结论: 职业内异质性变化不显著 (β={r['beta']:+.5f}, p={r['pval']:.4f})")
            print(f"  → 综合化主要是职业层面的整体平移，职业分类的内部同质性未被动摇")
            print(f"  → 用职业级AI暴露度作为处理变量是合理的")


if __name__ == "__main__":
    main()
