#!/usr/bin/env python3
"""
xgboost_ai_exposure.py

用XGBoost检验AI暴露度对entropy的预测贡献。

核心思路：
  如果AI exposure对entropy有预测力（线性或非线性），
  XGBoost应能捕捉到。通过以下方法量化其贡献：

  1. Feature importance: jd_ai_score在所有特征中的排名
  2. Ablation: 去掉AI特征后R²下降多少
  3. SHAP values: AI exposure的边际贡献分布
  4. Partial dependence: AI exposure与entropy的非线性关系

特征集：
  - year (时间趋势)
  - jd_ai_score (JD级AI暴露)
  - ai_mean_score (ISCO级AI暴露)
  - token_count (文档长度)
  - industry20_code (行业，20个类别)
  - isco08_4digit (职业，304个码)
  - dominant_topic_prob (主题集中度)

输出: output/xgboost_analysis/
"""

import time
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE = Path('/Users/yu/code/code2601/TY')
MASTER = BASE / 'data/Heterogeneity/master_with_jd_ai_score.csv'
OUT = BASE / 'output/xgboost_analysis'
OUT.mkdir(parents=True, exist_ok=True)

SEED = 42
SAMPLE_N = 500_000  # 50万条样本（全量太慢）


def load_data():
    """加载并预处理数据。"""
    print('[1] Loading data ...')
    t0 = time.time()
    cols = ['year', 'entropy_score', 'jd_ai_score', 'ai_mean_score',
            'token_count', 'industry20_code', 'isco08_4digit',
            'dominant_topic_prob', 'ai_exposure_gradient']
    chunks = []
    for chunk in pd.read_csv(MASTER, usecols=cols, chunksize=500_000):
        chunk = chunk.dropna(subset=['entropy_score', 'jd_ai_score',
                                      'isco08_4digit', 'industry20_code'])
        chunks.append(chunk)
    df = pd.concat(chunks, ignore_index=True)
    print(f'    Loaded {len(df):,} rows in {time.time()-t0:.1f}s')

    # 采样
    if len(df) > SAMPLE_N:
        df = df.sample(SAMPLE_N, random_state=SEED).reset_index(drop=True)
        print(f'    Sampled to {SAMPLE_N:,}')

    # 编码分类变量
    df['industry_cat'] = df['industry20_code'].astype('category').cat.codes
    df['isco_cat'] = df['isco08_4digit'].astype(int)
    df['year'] = df['year'].astype(int)
    df['log_token_count'] = np.log(df['token_count'].clip(lower=1))

    print(f'    Years: {sorted(df.year.unique())}')
    print(f'    Industries: {df.industry20_code.nunique()}')
    print(f'    ISCO codes: {df.isco_cat.nunique()}')
    return df


def build_features(df, include_ai=True, include_isco=True):
    """构建特征矩阵。
    注意：排除dominant_topic_prob，因为它与entropy_score机械相关
    （两者均来自同一LDA主题分布）。
    """
    feat_cols = ['year', 'log_token_count', 'industry_cat']
    if include_isco:
        feat_cols.append('isco_cat')
    if include_ai:
        feat_cols.extend(['jd_ai_score', 'ai_mean_score'])
    X = df[feat_cols].copy()
    y = df['entropy_score'].values
    return X, y, feat_cols


def train_xgb(X_train, y_train, X_test, y_test, label=''):
    """训练XGBoost并返回模型和指标。"""
    model = xgb.XGBRegressor(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=50,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=SEED,
        n_jobs=-1,
        enable_categorical=True,
    )
    t0 = time.time()
    model.fit(X_train, y_train,
              eval_set=[(X_test, y_test)],
              verbose=False)
    elapsed = time.time() - t0

    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    r2_train = r2_score(y_train, y_pred_train)
    r2_test = r2_score(y_test, y_pred_test)
    rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))

    print(f'    [{label}] R²_train={r2_train:.4f}  R²_test={r2_test:.4f}  '
          f'RMSE={rmse_test:.4f}  ({elapsed:.1f}s)')
    return model, r2_test, rmse_test


def main():
    df = load_data()

    # Train/test split
    X_full, y_full, cols_full = build_features(df, include_ai=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X_full, y_full, test_size=0.2, random_state=SEED)
    idx_train, idx_test = X_train.index, X_test.index

    results = []
    lines = []

    # ────────────────────────────────────────────────────────
    # Experiment 1: Full model (all features)
    # ────────────────────────────────────────────────────────
    print('\n[2] Training XGBoost models ...')
    print('=' * 60)

    model_full, r2_full, rmse_full = train_xgb(
        X_train, y_train, X_test, y_test, 'Full model')
    results.append(('Full (all features)', r2_full, rmse_full))

    # ────────────────────────────────────────────────────────
    # Experiment 2: Without AI features
    # ────────────────────────────────────────────────────────
    X_noai, y_noai, cols_noai = build_features(df, include_ai=False)
    _, r2_noai, rmse_noai = train_xgb(
        X_noai.loc[idx_train], y_train,
        X_noai.loc[idx_test], y_test, 'No AI features')
    results.append(('Without AI features', r2_noai, rmse_noai))

    # ────────────────────────────────────────────────────────
    # Experiment 3: Without ISCO (keep AI)
    # ────────────────────────────────────────────────────────
    X_noisco, y_noisco, cols_noisco = build_features(
        df, include_ai=True, include_isco=False)
    _, r2_noisco, rmse_noisco = train_xgb(
        X_noisco.loc[idx_train], y_train,
        X_noisco.loc[idx_test], y_test, 'No ISCO code')
    results.append(('Without ISCO code', r2_noisco, rmse_noisco))

    # ────────────────────────────────────────────────────────
    # Experiment 4: Only AI + year
    # ────────────────────────────────────────────────────────
    X_aionly = df[['year', 'jd_ai_score', 'ai_mean_score']].copy()
    _, r2_aionly, rmse_aionly = train_xgb(
        X_aionly.loc[idx_train], y_train,
        X_aionly.loc[idx_test], y_test, 'Only AI + year')
    results.append(('Only AI + year', r2_aionly, rmse_aionly))

    # ────────────────────────────────────────────────────────
    # Experiment 5: Only year (baseline)
    # ────────────────────────────────────────────────────────
    X_yearonly = df[['year']].copy()
    _, r2_yearonly, rmse_yearonly = train_xgb(
        X_yearonly.loc[idx_train], y_train,
        X_yearonly.loc[idx_test], y_test, 'Only year')
    results.append(('Only year (baseline)', r2_yearonly, rmse_yearonly))

    # ────────────────────────────────────────────────────────
    # Ablation summary
    # ────────────────────────────────────────────────────────
    print('\n' + '=' * 60)
    print('ABLATION SUMMARY')
    print('=' * 60)
    lines.append('ABLATION SUMMARY')
    lines.append('=' * 60)
    lines.append(f'{"Model":<30} {"R²_test":>10} {"RMSE":>10} {"ΔR² vs Full":>12}')
    lines.append('-' * 65)
    for name, r2, rmse in results:
        delta = r2 - r2_full
        line = f'{name:<30} {r2:>10.4f} {rmse:>10.4f} {delta:>+12.4f}'
        print(f'  {line}')
        lines.append(line)

    ai_contribution = r2_full - r2_noai
    lines.append('')
    lines.append(f'AI features contribution to R²: {ai_contribution:+.4f}')
    lines.append(f'  ({ai_contribution/r2_full*100:.1f}% of total R²)')
    print(f'\n  AI features contribution: ΔR² = {ai_contribution:+.4f} '
          f'({ai_contribution/r2_full*100:.1f}% of total)')

    # ────────────────────────────────────────────────────────
    # Feature importance (full model)
    # ────────────────────────────────────────────────────────
    print('\n[3] Feature importance ...')
    importance = model_full.feature_importances_
    feat_names = list(X_full.columns)
    imp_df = pd.DataFrame({
        'feature': feat_names,
        'importance': importance
    }).sort_values('importance', ascending=False)

    lines.append('')
    lines.append('FEATURE IMPORTANCE (gain)')
    lines.append('=' * 40)
    for _, row in imp_df.iterrows():
        bar = '█' * int(row['importance'] / imp_df['importance'].max() * 30)
        line = f"  {row['feature']:<22} {row['importance']:.4f}  {bar}"
        print(line)
        lines.append(line)

    # Plot feature importance
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ['#e74c3c' if 'ai' in f.lower() else '#3498db'
              for f in imp_df['feature']]
    ax.barh(range(len(imp_df)), imp_df['importance'].values,
            color=colors, edgecolor='white', linewidth=0.5)
    ax.set_yticks(range(len(imp_df)))
    ax.set_yticklabels(imp_df['feature'].values)
    ax.invert_yaxis()
    ax.set_xlabel('Feature Importance (gain)')
    ax.set_title(f'XGBoost Feature Importance\n'
                 f'(Red = AI exposure features, '
                 f'Full model R² = {r2_full:.4f})')
    plt.tight_layout()
    plt.savefig(OUT / 'feature_importance.png', dpi=150)
    plt.close()
    print(f'    Saved feature_importance.png')

    # ────────────────────────────────────────────────────────
    # SHAP values (may fail due to numpy/numba version)
    # ────────────────────────────────────────────────────────
    print('\n[4] SHAP analysis ...')
    try:
        import shap

        shap_sample = X_test.sample(min(5000, len(X_test)), random_state=SEED)
        explainer = shap.TreeExplainer(model_full)
        shap_values = explainer.shap_values(shap_sample)

        fig, ax = plt.subplots(figsize=(10, 7))
        shap.summary_plot(shap_values, shap_sample,
                          show=False, max_display=len(feat_names))
        plt.title('SHAP Value Distribution by Feature')
        plt.tight_layout()
        plt.savefig(OUT / 'shap_summary.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f'    Saved shap_summary.png')

        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        shap_df = pd.DataFrame({
            'feature': feat_names,
            'mean_abs_shap': mean_abs_shap
        }).sort_values('mean_abs_shap', ascending=False)

        lines.append('')
        lines.append('SHAP ANALYSIS (mean |SHAP|)')
        lines.append('=' * 40)
        for _, row in shap_df.iterrows():
            bar = '█' * int(row['mean_abs_shap'] / shap_df['mean_abs_shap'].max() * 30)
            line = f"  {row['feature']:<22} {row['mean_abs_shap']:.6f}  {bar}"
            print(f'  {line}')
            lines.append(line)

        ai_idx = feat_names.index('jd_ai_score')
        ai_shap = shap_values[:, ai_idx]
        lines.append('')
        lines.append(f'jd_ai_score SHAP detail:')
        lines.append(f'  mean |SHAP|  = {np.abs(ai_shap).mean():.6f}')
        lines.append(f'  max  |SHAP|  = {np.abs(ai_shap).max():.6f}')
        lines.append(f'  std  SHAP    = {ai_shap.std():.6f}')
        lines.append(f'  % samples with |SHAP| > 0.001: '
                     f'{(np.abs(ai_shap) > 0.001).mean():.1%}')
    except (ImportError, Exception) as e:
        print(f'    SHAP skipped: {e}')
        lines.append('')
        lines.append(f'SHAP ANALYSIS: skipped ({e})')

    # ────────────────────────────────────────────────────────
    # Partial dependence: jd_ai_score
    # ────────────────────────────────────────────────────────
    print('\n[5] Partial dependence plot ...')
    from sklearn.inspection import partial_dependence

    pd_result = partial_dependence(
        model_full, X_train.sample(min(10000, len(X_train)), random_state=SEED),
        features=['jd_ai_score'],
        kind='average', grid_resolution=50)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: PDP for jd_ai_score
    ax = axes[0]
    ax.plot(pd_result['grid_values'][0], pd_result['average'][0],
            'b-', linewidth=2)
    ax.set_xlabel('jd_ai_score')
    ax.set_ylabel('Partial Dependence (entropy)')
    ax.set_title('Partial Dependence: jd_ai_score → entropy')
    ax.grid(True, alpha=0.3)

    # Right: PDP for year
    pd_year = partial_dependence(
        model_full, X_train.sample(min(10000, len(X_train)), random_state=SEED),
        features=['year'],
        kind='average', grid_resolution=10)
    ax = axes[1]
    ax.plot(pd_year['grid_values'][0], pd_year['average'][0],
            'r-o', linewidth=2, markersize=6)
    ax.set_xlabel('Year')
    ax.set_ylabel('Partial Dependence (entropy)')
    ax.set_title('Partial Dependence: Year → entropy')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUT / 'partial_dependence.png', dpi=150)
    plt.close()
    print(f'    Saved partial_dependence.png')

    # PDP range comparison
    pdp_ai_range = pd_result['average'][0].max() - pd_result['average'][0].min()
    pdp_year_range = pd_year['average'][0].max() - pd_year['average'][0].min()
    lines.append('')
    lines.append('PARTIAL DEPENDENCE RANGE')
    lines.append(f'  jd_ai_score PDP range: {pdp_ai_range:.4f}')
    lines.append(f'  year PDP range:        {pdp_year_range:.4f}')
    lines.append(f'  ratio (year/ai):       {pdp_year_range/pdp_ai_range:.1f}x')

    # ────────────────────────────────────────────────────────
    # Year × AI interaction: train separate models per period
    # ────────────────────────────────────────────────────────
    print('\n[6] Temporal analysis: AI importance by period ...')
    lines.append('')
    lines.append('AI IMPORTANCE BY PERIOD')
    lines.append('=' * 50)

    for period_name, year_range in [('Pre-2023 (2016-2022)',
                                      range(2016, 2023)),
                                     ('Post-2023 (2023-2025)',
                                      range(2023, 2026))]:
        mask = df['year'].isin(year_range)
        df_period = df[mask]
        if len(df_period) < 1000:
            continue
        X_p, y_p, _ = build_features(df_period, include_ai=True)
        X_ptr, X_pte, y_ptr, y_pte = train_test_split(
            X_p, y_p, test_size=0.2, random_state=SEED)

        m_with, r2_with, _ = train_xgb(X_ptr, y_ptr, X_pte, y_pte,
                                         f'{period_name} +AI')
        X_p_noai, _, _ = build_features(df_period, include_ai=False)
        _, r2_without, _ = train_xgb(
            X_p_noai.loc[X_ptr.index], y_ptr,
            X_p_noai.loc[X_pte.index], y_pte,
            f'{period_name} -AI')

        delta = r2_with - r2_without
        line = f'  {period_name}: ΔR² = {delta:+.4f} (with={r2_with:.4f}, without={r2_without:.4f})'
        print(f'  {line}')
        lines.append(line)

        # Feature importance for this period
        imp_p = dict(zip(X_p.columns, m_with.feature_importances_))
        ai_imp = imp_p.get('jd_ai_score', 0)
        total_imp = sum(imp_p.values())
        lines.append(f'    jd_ai_score importance: {ai_imp:.4f} '
                     f'({ai_imp/total_imp*100:.1f}%)')

    # ────────────────────────────────────────────────────────
    # Save results
    # ────────────────────────────────────────────────────────
    result_text = '\n'.join(lines)
    with open(OUT / 'xgboost_results.txt', 'w') as f:
        f.write(result_text)

    print(f'\n{"="*60}')
    print(f'All results saved to {OUT}/')
    for f in sorted(OUT.glob('*')):
        print(f'  {f.name}')
    print('Done.')


if __name__ == '__main__':
    main()
