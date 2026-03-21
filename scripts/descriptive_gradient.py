#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
descriptive_gradient.py

Descriptive analysis of skill comprehensiveness trends by AI exposure gradient.
Replaces the DID framework with a continuous gradient approach.

Core insight: the advisor asked "which industries/occupations see faster skill
comprehensiveness rises" — a descriptive question, not a causal one.

Uses pre-aggregated data (yearly_ai_exposure_entropy.csv, 60 rows)
so NO need to read the 8M-row master CSV.

Models:
  A. mean_entropy ~ year_FE + ai_score  (cross-sectional association)
  B. mean_entropy ~ year_FE + ai_score + year_FE × ai_score  (differential trends)
  C. mean_entropy ~ year_FE + gradient_group + year_FE × gradient_group (categorical)

Output:
  output/gradient_analysis/gradient_trend.png
  output/gradient_analysis/gradient_slopes.png
  output/gradient_analysis/interaction_coefs.png
  output/gradient_analysis/regression_results.txt
  output/gradient_analysis/gradient_summary.csv
"""

import csv
import math
from pathlib import Path
from collections import defaultdict

import numpy as np

BASE = Path('/Users/yu/code/code2601/TY')
AGG_CSV = BASE / 'data/Heterogeneity/yearly_ai_exposure_entropy.csv'
OUT_DIR = BASE / 'output/gradient_analysis'

EXCL_YEARS = {2019, 2020, 2025}

# Gradient display order (low → high exposure)
GRADIENT_ORDER = ['NotExposed', 'Minimal', 'Gradient1', 'Gradient2', 'Gradient3', 'Gradient4']
GRADIENT_LABELS = {
    'NotExposed': 'Not Exposed',
    'Minimal':    'Minimal',
    'Gradient1':  'G1 (Low)',
    'Gradient2':  'G2 (Moderate)',
    'Gradient3':  'G3 (Significant)',
    'Gradient4':  'G4 (High)',
}


def load_aggregated():
    """Load yearly_ai_exposure_entropy.csv, exclude anomalous years."""
    rows = []
    with AGG_CSV.open(encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for r in reader:
            year = int(r['year'])
            if year in EXCL_YEARS:
                continue
            rows.append({
                'year': year,
                'group': r['gradient_group'],
                'n': int(r['n']),
                'mean_entropy': float(r['mean_entropy']),
                'mean_ai_score': float(r['mean_ai_score']),
            })
    print(f'Loaded {len(rows)} cells (excluded years {EXCL_YEARS})')
    return rows


# ── Part 1: Descriptive Trend Plots ──────────────────────────────────────────

def plot_trends(rows):
    """Overlay entropy trends for all 6 gradient groups."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    groups_data = defaultdict(list)
    for r in rows:
        groups_data[r['group']].append(r)

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = {
        'NotExposed': '#2c3e50', 'Minimal': '#7f8c8d',
        'Gradient1': '#3498db', 'Gradient2': '#2ecc71',
        'Gradient3': '#f39c12', 'Gradient4': '#e74c3c',
    }
    markers = {
        'NotExposed': 'v', 'Minimal': '<',
        'Gradient1': 'D', 'Gradient2': 'o',
        'Gradient3': 's', 'Gradient4': '^',
    }

    for g in GRADIENT_ORDER:
        if g not in groups_data:
            continue
        data = sorted(groups_data[g], key=lambda x: x['year'])
        years = [d['year'] for d in data]
        ents = [d['mean_entropy'] for d in data]
        ax.plot(years, ents, f'{markers[g]}-', color=colors[g],
                linewidth=2, markersize=6, label=GRADIENT_LABELS[g])

    ax.axvspan(2018.5, 2020.5, alpha=0.08, color='gray')
    ax.set_xlabel('Year', fontsize=12)
    ax.set_ylabel('Mean Entropy (Skill Comprehensiveness)', fontsize=12)
    ax.set_title('Skill Comprehensiveness by AI Exposure Gradient\n'
                 '(All groups show parallel upward trends)', fontsize=13)
    ax.legend(title='AI Exposure', loc='upper left', fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    plt.tight_layout()
    out = OUT_DIR / 'gradient_trend.png'
    plt.savefig(out, dpi=200)
    print(f'  Trend plot: {out}')
    plt.close()


def plot_slopes(rows):
    """Bar chart of OLS trend slope for each gradient group."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    groups_data = defaultdict(list)
    for r in rows:
        groups_data[r['group']].append(r)

    slopes = {}
    for g in GRADIENT_ORDER:
        if g not in groups_data:
            continue
        data = sorted(groups_data[g], key=lambda x: x['year'])
        x = np.array([d['year'] for d in data], dtype=float)
        y = np.array([d['mean_entropy'] for d in data])
        w = np.array([d['n'] for d in data], dtype=float)
        # Weighted OLS slope
        xm = np.average(x, weights=w)
        ym = np.average(y, weights=w)
        slope = np.sum(w * (x - xm) * (y - ym)) / np.sum(w * (x - xm)**2)
        slopes[g] = slope

    fig, ax = plt.subplots(figsize=(8, 5))
    groups = [g for g in GRADIENT_ORDER if g in slopes]
    labels = [GRADIENT_LABELS[g] for g in groups]
    vals = [slopes[g] * 1000 for g in groups]

    bars = ax.bar(labels, vals, color=['#2c3e50', '#7f8c8d', '#3498db',
                                        '#2ecc71', '#f39c12', '#e74c3c'][:len(groups)])
    ax.set_xlabel('AI Exposure Gradient', fontsize=12)
    ax.set_ylabel('Entropy Trend Slope (×10³ per year)', fontsize=12)
    ax.set_title('Trend Slope by AI Exposure Group\n'
                 '(All positive — universal upward trend)', fontsize=13)
    ax.grid(True, alpha=0.3, axis='y')

    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.03,
                f'{v:.2f}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    out = OUT_DIR / 'gradient_slopes.png'
    plt.savefig(out, dpi=200)
    print(f'  Slope plot: {out}')
    plt.close()

    return slopes


# ── Part 2: Cell-level WLS Regressions ───────────────────────────────────────

def run_regressions(rows):
    """Run cell-level WLS regressions with continuous ai_score interactions."""
    import statsmodels.formula.api as smf
    import pandas as pd

    df = pd.DataFrame(rows)
    results_text = []

    # ── Model A: Baseline (year FE + continuous ai_score) ────────────────
    print('\n── Model A: mean_entropy ~ year_FE + ai_score ──')
    res_a = smf.wls('mean_entropy ~ C(year) + mean_ai_score',
                     data=df, weights=df['n']).fit(cov_type='HC3')
    coef_ai = res_a.params.get('mean_ai_score', float('nan'))
    p_ai = res_a.pvalues.get('mean_ai_score', float('nan'))
    print(f'  ai_score coef: {coef_ai:.6f}  (p={p_ai:.4f})')
    results_text.append('=' * 70)
    results_text.append('Model A: mean_entropy ~ C(year) + mean_ai_score')
    results_text.append('Cross-sectional association of AI exposure with entropy')
    results_text.append('=' * 70)
    results_text.append(res_a.summary().as_text())

    # ── Model B: Continuous interaction (year FE × ai_score) ─────────────
    print('\n── Model B: mean_entropy ~ year_FE * ai_score ──')
    res_b = smf.wls('mean_entropy ~ C(year) * mean_ai_score',
                     data=df, weights=df['n']).fit(cov_type='HC3')
    results_text.append('\n\n' + '=' * 70)
    results_text.append('Model B: mean_entropy ~ C(year) * mean_ai_score')
    results_text.append('Tests whether entropy trends differ by AI exposure level')
    results_text.append('=' * 70)
    results_text.append(res_b.summary().as_text())

    # Extract interaction coefficients
    interaction_coefs = {}
    for param, val in res_b.params.items():
        if ':mean_ai_score' in param:
            # Extract year from param name like "C(year)[T.2017]:mean_ai_score"
            year_str = param.split('[T.')[1].split(']')[0]
            interaction_coefs[int(year_str)] = {
                'coef': val,
                'se': res_b.bse[param],
                'p': res_b.pvalues[param],
            }
    print('  Year × ai_score interaction coefficients:')
    for y in sorted(interaction_coefs):
        ic = interaction_coefs[y]
        sig = '*' if ic['p'] < 0.05 else ''
        print(f'    {y}: {ic["coef"]:+.6f} (p={ic["p"]:.3f}){sig}')

    # ── Model C: Categorical (group FE + year FE, no interaction) ──────
    # Note: full interaction C(year)*C(group) is saturated (42 params, 42 obs).
    # Instead, use additive model to test group-level entropy differences.
    print('\n── Model C: mean_entropy ~ year_FE + gradient_group (additive) ──')
    df['group_cat'] = pd.Categorical(df['group'], categories=GRADIENT_ORDER, ordered=True)
    res_c = smf.wls('mean_entropy ~ C(year) + C(group_cat)',
                     data=df, weights=df['n']).fit(cov_type='HC3')
    results_text.append('\n\n' + '=' * 70)
    results_text.append('Model C: mean_entropy ~ C(year) + C(gradient_group)')
    results_text.append('Additive model: group differences after controlling for year')
    results_text.append('(Full interaction omitted: saturated with 42 cells)')
    results_text.append('=' * 70)
    results_text.append(res_c.summary().as_text())
    # Print group coefficients
    print('  Group coefficients (ref=NotExposed):')
    for param, val in res_c.params.items():
        if 'group_cat' in param:
            p = res_c.pvalues[param]
            print(f'    {param}: {val:+.6f} (p={p:.3f})')

    # Save all results
    results_path = OUT_DIR / 'regression_results.txt'
    results_path.write_text('\n'.join(results_text), encoding='utf-8')
    print(f'\n  Regression results: {results_path}')

    return interaction_coefs, res_a, res_b


def plot_interactions(interaction_coefs, ref_year=2016):
    """Plot year × ai_score interaction coefficients."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    years = sorted(interaction_coefs.keys())
    coefs = [interaction_coefs[y]['coef'] for y in years]
    ses = [interaction_coefs[y]['se'] for y in years]
    ci_lo = [c - 1.96 * s for c, s in zip(coefs, ses)]
    ci_hi = [c + 1.96 * s for c, s in zip(coefs, ses)]

    # Add reference year (zero by construction)
    all_years = [ref_year] + years
    all_coefs = [0.0] + coefs
    all_ci_lo = [0.0] + ci_lo
    all_ci_hi = [0.0] + ci_hi

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.fill_between(all_years, all_ci_lo, all_ci_hi, alpha=0.2, color='steelblue')
    ax.plot(all_years, all_coefs, 'o-', color='steelblue', linewidth=2, markersize=7)
    ax.axhline(0, color='black', linewidth=0.8, linestyle='--')
    ax.axvspan(2018.5, 2020.5, alpha=0.08, color='gray')

    ax.set_xlabel('Year', fontsize=12)
    ax.set_ylabel('Interaction Coefficient (year × ai_score)', fontsize=12)
    ax.set_title('Differential Entropy Trend by AI Exposure\n'
                 f'(Year × ai_score interaction, ref={ref_year})', fontsize=13)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))

    # Annotate: if all near zero, the trend is universal
    max_abs = max(abs(c) for c in all_coefs)
    if max_abs < 0.01:
        ax.text(0.5, 0.95, 'All interactions ≈ 0: trend is universal across exposure levels',
                transform=ax.transAxes, ha='center', va='top',
                fontsize=10, color='green', fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()
    out = OUT_DIR / 'interaction_coefs.png'
    plt.savefig(out, dpi=200)
    print(f'  Interaction plot: {out}')
    plt.close()


# ── Part 3: Summary Statistics ───────────────────────────────────────────────

def write_summary(rows, slopes):
    """Write key summary statistics to CSV."""
    groups_data = defaultdict(list)
    for r in rows:
        groups_data[r['group']].append(r)

    summary_rows = []
    for g in GRADIENT_ORDER:
        if g not in groups_data:
            continue
        data = groups_data[g]
        total_n = sum(d['n'] for d in data)
        mean_score = np.average([d['mean_ai_score'] for d in data],
                                weights=[d['n'] for d in data])
        ent_2016 = next((d['mean_entropy'] for d in data if d['year'] == 2016), None)
        ent_2024 = next((d['mean_entropy'] for d in data if d['year'] == 2024), None)
        change = (ent_2024 - ent_2016) if ent_2016 and ent_2024 else None

        summary_rows.append({
            'gradient_group': g,
            'label': GRADIENT_LABELS[g],
            'total_n': total_n,
            'mean_ai_score': f'{mean_score:.3f}',
            'entropy_2016': f'{ent_2016:.4f}' if ent_2016 else '',
            'entropy_2024': f'{ent_2024:.4f}' if ent_2024 else '',
            'entropy_change': f'{change:+.4f}' if change else '',
            'trend_slope_x1000': f'{slopes.get(g, 0)*1000:.2f}',
        })

    out = OUT_DIR / 'gradient_summary.csv'
    with out.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=summary_rows[0].keys())
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f'  Summary: {out}')

    # Print to console
    print('\n── Summary by Gradient Group ──')
    print(f'{"Group":<16} {"N":>10} {"AI score":>9} {"Ent 2016":>9} '
          f'{"Ent 2024":>9} {"Change":>8} {"Slope×1e3":>10}')
    for r in summary_rows:
        print(f'{r["label"]:<16} {r["total_n"]:>10,} {r["mean_ai_score"]:>9} '
              f'{r["entropy_2016"]:>9} {r["entropy_2024"]:>9} '
              f'{r["entropy_change"]:>8} {r["trend_slope_x1000"]:>10}')


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print('=' * 60)
    print('DESCRIPTIVE GRADIENT ANALYSIS')
    print('AI exposure as continuous variable, no DID')
    print('=' * 60)

    rows = load_aggregated()

    # Part 1: Descriptive plots
    print('\n[1] Trend plots ...')
    plot_trends(rows)
    slopes = plot_slopes(rows)

    # Part 2: Regressions
    print('\n[2] Cell-level WLS regressions ...')
    interaction_coefs, res_a, res_b = run_regressions(rows)

    # Part 3: Interaction plot
    print('\n[3] Interaction coefficient plot ...')
    plot_interactions(interaction_coefs)

    # Part 4: Summary
    print('\n[4] Summary statistics ...')
    write_summary(rows, slopes)

    print(f'\n✓ All outputs saved to {OUT_DIR}/')


if __name__ == '__main__':
    main()
