#!/usr/bin/env python3
"""
ISCO 4-digit × Year level regression with continuous AI exposure score.

Methodological improvements over 6-group aggregation:
1. Uses each ISCO occupation's own ai_mean_score (~300 unique continuous values)
2. Aggregates to ISCO × year cells (~2000-3000 observations)
3. Provides sufficient degrees of freedom for interaction tests
4. Includes ALL years (no exclusions)
"""

import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import statsmodels.api as sm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
MASTER = ROOT / 'data' / 'Heterogeneity' / 'master_with_ai_exposure_v2.csv'
OUT = ROOT / 'output' / 'isco_regression'
OUT.mkdir(parents=True, exist_ok=True)

CHUNK = 500_000
COLS = ['year', 'isco08_4digit', 'entropy_score', 'ai_mean_score', 'token_count']

# ── Phase 1: Chunked aggregation ───────────────────────────
print("=" * 60)
print("Phase 1: Chunked aggregation to ISCO × year cells")
print("=" * 60)

# Accumulate: (isco, year) -> [sum_entropy, sum_log_tc, n, ai_score]
cells = {}
total_rows = 0
skipped = 0

for i, chunk in enumerate(pd.read_csv(MASTER, chunksize=CHUNK,
                                       usecols=COLS, encoding='utf-8-sig')):
    total_rows += len(chunk)

    # Drop records without ISCO code or ai_score
    valid = chunk.dropna(subset=['isco08_4digit', 'ai_mean_score', 'entropy_score'])
    skipped += len(chunk) - len(valid)

    valid = valid.copy()
    valid['isco'] = valid['isco08_4digit'].apply(lambda x: int(float(x)))
    valid['yr'] = valid['year'].apply(lambda x: int(float(x)))
    valid['log_tc'] = np.log(valid['token_count'].clip(lower=1))

    for (isco, yr), grp in valid.groupby(['isco', 'yr']):
        key = (isco, yr)
        s_ent = grp['entropy_score'].sum()
        s_ltc = grp['log_tc'].sum()
        n = len(grp)
        ai = grp['ai_mean_score'].iloc[0]  # constant within ISCO

        if key in cells:
            old = cells[key]
            cells[key] = [old[0] + s_ent, old[1] + s_ltc, old[2] + n, ai]
        else:
            cells[key] = [s_ent, s_ltc, n, ai]

    print(f"  Chunk {i+1}: {total_rows:>10,} rows processed, "
          f"{len(cells):,} cells so far")

print(f"\nTotal: {total_rows:,} rows, {skipped:,} skipped (no ISCO/ai_score), "
      f"{len(cells):,} ISCO×year cells")

# Build DataFrame
rows = []
for (isco, yr), (s_ent, s_ltc, n, ai) in cells.items():
    rows.append({
        'isco': isco, 'year': yr,
        'mean_entropy': s_ent / n,
        'mean_log_tc': s_ltc / n,
        'n': n,
        'ai_score': ai,
    })

df = pd.DataFrame(rows).sort_values(['isco', 'year']).reset_index(drop=True)

print(f"\n{'Cell-level summary':=^60}")
print(f"  Cells:              {len(df):,}")
print(f"  Unique ISCO codes:  {df['isco'].nunique()}")
print(f"  Years:              {sorted(df['year'].unique())}")
print(f"  ai_score range:     [{df['ai_score'].min():.3f}, {df['ai_score'].max():.3f}]")
print(f"  ai_score unique:    {df['ai_score'].nunique()}")
print(f"  Total observations: {df['n'].sum():,}")

df.to_csv(OUT / 'isco_year_cells.csv', index=False)

# ── Phase 2: WLS Regressions ──────────────────────────────
print(f"\n{'Phase 2: WLS Regressions':=^60}")

results_text = []

# ─── Model 1: year FE + ai_score ───
print("\n--- Model 1: C(year) + ai_score ---")
m1 = smf.wls('mean_entropy ~ C(year) + ai_score',
             data=df, weights=df['n']).fit(cov_type='HC3')

results_text.append("=" * 70)
results_text.append("Model 1: mean_entropy ~ C(year) + ai_score")
results_text.append("Cross-sectional association at ISCO occupation level")
results_text.append(f"N = {len(df)} cells, {df['isco'].nunique()} ISCO codes, "
                    f"{df['year'].nunique()} years")
results_text.append("=" * 70)
results_text.append(str(m1.summary()))

ai_coef1 = m1.params['ai_score']
ai_p1 = m1.pvalues['ai_score']
print(f"  ai_score: coef={ai_coef1:.6f}, p={ai_p1:.4f}")

# ─── Model 2: year FE × ai_score (full interaction) ───
print("\n--- Model 2: C(year) * ai_score ---")
m2 = smf.wls('mean_entropy ~ C(year) * ai_score',
             data=df, weights=df['n']).fit(cov_type='HC3')

results_text.append("\n\n" + "=" * 70)
results_text.append("Model 2: mean_entropy ~ C(year) * ai_score")
results_text.append("Tests whether entropy trends differ by AI exposure level")
results_text.append(f"N = {len(df)} cells")
results_text.append("=" * 70)
results_text.append(str(m2.summary()))

# Extract interaction terms
int_params = {k: v for k, v in m2.params.items() if ':ai_score' in k}
int_pvals = {k: v for k, v in m2.pvalues.items() if ':ai_score' in k}
print("  Interaction coefficients:")
for k in sorted(int_params.keys()):
    yr = k.split('[T.')[1].split(']')[0]
    print(f"    {yr} × ai_score: {int_params[k]:+.6f}  p={int_pvals[k]:.4f}")

# ─── Model 3: year FE + ISCO FE + year×ai_score ───
print("\n--- Model 3: C(year) + C(isco) + year×ai_score ---")
print("  (ISCO FE absorbs ai_score main effect)")

# Build design matrix manually for efficiency
ref_year = df['year'].min()
year_vals = sorted(df['year'].unique())
non_ref_years = [y for y in year_vals if y != ref_year]

# Year dummies
yr_dummies = pd.DataFrame({
    f'yr_{y}': (df['year'] == y).astype(float) for y in non_ref_years
})

# ISCO dummies
ref_isco = df['isco'].mode().iloc[0]  # most common as reference
isco_vals = sorted(df['isco'].unique())
non_ref_iscos = [o for o in isco_vals if o != ref_isco]
isco_dummies = pd.DataFrame({
    f'occ_{o}': (df['isco'] == o).astype(float) for o in non_ref_iscos
})

# Interaction: year_dummy × ai_score
int_df = pd.DataFrame({
    f'yr_{y}_x_ai': yr_dummies[f'yr_{y}'] * df['ai_score'].values
    for y in non_ref_years
})
int_cols = list(int_df.columns)

X = pd.concat([yr_dummies, isco_dummies, int_df], axis=1)
X = sm.add_constant(X)
y = df['mean_entropy']
w = df['n']

print(f"  Design matrix: {X.shape[0]} obs × {X.shape[1]} params "
      f"({len(non_ref_iscos)} ISCO FEs)")

m3 = sm.WLS(y, X, weights=w).fit(cov_type='HC3')

results_text.append("\n\n" + "=" * 70)
results_text.append("Model 3: mean_entropy ~ C(year) + C(isco) + year × ai_score")
results_text.append("Within-occupation specification")
results_text.append(f"ISCO FE absorbs ai_score main effect")
results_text.append(f"N = {len(df)}, {len(non_ref_iscos)+1} ISCO codes, "
                    f"{len(year_vals)} years")
results_text.append("=" * 70)
results_text.append(f"\nR²     = {m3.rsquared:.6f}")
results_text.append(f"Adj R² = {m3.rsquared_adj:.6f}")
results_text.append(f"Params = {X.shape[1]}")
results_text.append(f"\nInteraction coefficients (year × ai_score):")
results_text.append(f"{'Variable':<20} {'Coef':>12} {'SE':>12} {'p':>8}")
results_text.append("-" * 55)

for col in sorted(int_cols):
    yr = col.split('_')[1]
    results_text.append(
        f"{'yr_'+yr+' × ai':.<20} {m3.params[col]:>12.6f} "
        f"{m3.bse[col]:>12.6f} {m3.pvalues[col]:>8.4f}"
    )
    print(f"  {yr} × ai_score: {m3.params[col]:+.6f}  p={m3.pvalues[col]:.4f}")

# Joint F-test
r_matrix = np.zeros((len(int_cols), X.shape[1]))
for i, col in enumerate(sorted(int_cols)):
    r_matrix[i, list(X.columns).index(col)] = 1
f_test = m3.f_test(r_matrix)

f_val = float(np.asarray(f_test.fvalue).flat[0])
f_p = float(np.asarray(f_test.pvalue).flat[0])

results_text.append(f"\nJoint F-test (all year×ai_score interactions = 0):")
results_text.append(f"  F({len(int_cols)}, {int(m3.df_resid)}) = {f_val:.4f},  "
                    f"p = {f_p:.4f}")
print(f"\n  Joint F-test: F={f_val:.4f}, p={f_p:.4f}")

# ─── Model 4: continuous year × ai_score (parsimonious) ───
print("\n--- Model 4: year_continuous × ai_score + ISCO FE ---")

X4 = pd.DataFrame({
    'year_c': df['year'] - df['year'].median(),  # centered
    'ai_score': df['ai_score'],
})
X4['year_x_ai'] = X4['year_c'] * X4['ai_score']
X4 = pd.concat([X4, isco_dummies], axis=1)
X4 = sm.add_constant(X4)

m4 = sm.WLS(y, X4, weights=w).fit(cov_type='HC3')

results_text.append("\n\n" + "=" * 70)
results_text.append("Model 4: mean_entropy ~ year_continuous + ai_score "
                    "+ year×ai_score + ISCO FE")
results_text.append("Parsimonious: single interaction term (linear trend × ai_score)")
results_text.append("=" * 70)
results_text.append(f"\nR²     = {m4.rsquared:.6f}")
results_text.append(f"Adj R² = {m4.rsquared_adj:.6f}")
results_text.append(f"\n{'Variable':<20} {'Coef':>12} {'SE':>12} {'p':>8}")
results_text.append("-" * 55)
for v in ['year_c', 'ai_score', 'year_x_ai']:
    results_text.append(
        f"{v:<20} {m4.params[v]:>12.6f} {m4.bse[v]:>12.6f} "
        f"{m4.pvalues[v]:>8.4f}"
    )
    print(f"  {v}: coef={m4.params[v]:+.6f}, p={m4.pvalues[v]:.4f}")

# Save all results
with open(OUT / 'regression_results.txt', 'w') as f:
    f.write('\n'.join(results_text))
print(f"\nResults saved to {OUT / 'regression_results.txt'}")

# ── Phase 3: Visualization ────────────────────────────────
print(f"\n{'Phase 3: Visualization':=^60}")

# ─── Plot 1: Model 2 interaction coefficients ───
fig, ax = plt.subplots(figsize=(10, 6))
years_m2 = []
coefs_m2 = []
ses_m2 = []

for k in sorted(int_params.keys()):
    yr = int(k.split('[T.')[1].split(']')[0])
    years_m2.append(yr)
    coefs_m2.append(m2.params[k])
    ses_m2.append(m2.bse[k])

colors = ['steelblue'] * len(years_m2)
ax.bar(range(len(years_m2)), coefs_m2,
       yerr=[1.96 * s for s in ses_m2],
       capsize=4, alpha=0.7, color=colors, edgecolor='navy', linewidth=0.5)
ax.axhline(0, color='black', linewidth=0.8)
ax.set_xticks(range(len(years_m2)))
ax.set_xticklabels(years_m2, rotation=45)
ax.set_xlabel('Year')
ax.set_ylabel('Interaction coefficient (year × ai_score)')
ax.set_title(f'Model 2: Year × AI Exposure Interaction (no ISCO FE)\n'
             f'(N = {len(df):,} ISCO×year cells, '
             f'{df["ai_score"].nunique()} unique ai_scores)')
plt.tight_layout()
plt.savefig(OUT / 'isco_interaction_m2.png', dpi=150)
plt.close()
print("  Saved isco_interaction_m2.png")

# ─── Plot 2: Model 3 interaction coefficients (with ISCO FE) ───
fig, ax = plt.subplots(figsize=(10, 6))
years_m3 = []
coefs_m3 = []
ses_m3 = []

for col in sorted(int_cols):
    yr = int(col.split('_')[1])
    years_m3.append(yr)
    coefs_m3.append(m3.params[col])
    ses_m3.append(m3.bse[col])

ax.bar(range(len(years_m3)), coefs_m3,
       yerr=[1.96 * s for s in ses_m3],
       capsize=4, alpha=0.7, color='darkorange', edgecolor='saddlebrown',
       linewidth=0.5)
ax.axhline(0, color='black', linewidth=0.8)
ax.set_xticks(range(len(years_m3)))
ax.set_xticklabels(years_m3, rotation=45)
ax.set_xlabel('Year')
ax.set_ylabel('Interaction coefficient (year × ai_score)')
ax.set_title(f'Model 3: Year × AI Exposure Interaction (with ISCO FE)\n'
             f'(N = {len(df):,} cells, {len(non_ref_iscos)+1} occupation FEs, '
             f'Joint F p = {f_p:.4f})')
plt.tight_layout()
plt.savefig(OUT / 'isco_interaction_m3.png', dpi=150)
plt.close()
print("  Saved isco_interaction_m3.png")

# ─── Plot 3: Per-ISCO entropy slope vs ai_score scatter ───
print("  Computing per-ISCO trend slopes...")
slope_rows = []
for isco, grp in df.groupby('isco'):
    if len(grp) >= 3:
        b = np.polyfit(grp['year'], grp['mean_entropy'], 1,
                       w=np.sqrt(grp['n']))
        slope_rows.append({
            'isco': isco,
            'slope': b[0],
            'ai_score': grp['ai_score'].iloc[0],
            'total_n': grp['n'].sum()
        })

slope_df = pd.DataFrame(slope_rows)

fig, ax = plt.subplots(figsize=(10, 7))
sizes = np.sqrt(slope_df['total_n']) / 10
sizes = sizes.clip(upper=200)  # cap for readability
ax.scatter(slope_df['ai_score'], slope_df['slope'] * 1000,
           s=sizes, alpha=0.35, c='steelblue', edgecolors='navy', linewidth=0.3)

# WLS trend line
b_line = np.polyfit(slope_df['ai_score'], slope_df['slope'] * 1000,
                    1, w=np.sqrt(slope_df['total_n']))
x_line = np.linspace(slope_df['ai_score'].min(), slope_df['ai_score'].max(), 100)
ax.plot(x_line, np.polyval(b_line, x_line), 'r-', linewidth=2,
        label=f'WLS fit: slope = {b_line[0]:.2f}')

# Weighted correlation
from scipy.stats import pearsonr
r_val, r_p = pearsonr(slope_df['ai_score'], slope_df['slope'])
ax.text(0.02, 0.95, f'r = {r_val:.3f}, p = {r_p:.3f}\nN = {len(slope_df)} occupations',
        transform=ax.transAxes, fontsize=11, verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

ax.set_xlabel('AI Exposure Score (ai_mean_score)', fontsize=12)
ax.set_ylabel('Entropy Trend Slope (×10⁻³/year)', fontsize=12)
ax.set_title(f'Per-ISCO Occupation: Entropy Trend Slope vs AI Exposure Score\n'
             f'(bubble size ∝ √sample, {len(slope_df)} occupations)')
ax.legend(fontsize=11)
plt.tight_layout()
plt.savefig(OUT / 'isco_slope_vs_ai.png', dpi=150)
plt.close()
print("  Saved isco_slope_vs_ai.png")

# ─── Summary printout ───
print(f"\n{'SUMMARY':=^60}")
print(f"ISCO × Year cells: {len(df):,}")
print(f"Unique ISCO codes: {df['isco'].nunique()}")
print(f"Unique ai_scores:  {df['ai_score'].nunique()}")
print(f"Years:             {sorted(df['year'].unique())}")
print(f"\nModel 1 (no FE): ai_score coef = {ai_coef1:.6f}, p = {ai_p1:.4f}")
print(f"Model 2 (no FE): interaction terms — see above")
print(f"Model 3 (ISCO FE): Joint F = {f_val:.4f}, p = {f_p:.4f}")
print(f"Model 4 (linear): year×ai_score = {m4.params['year_x_ai']:.6f}, "
      f"p = {m4.pvalues['year_x_ai']:.4f}")
print(f"\nAll outputs saved to {OUT}")
