#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
experiment_ai_adoption.py

Two improvements:
1. Direct AI adoption measure: scan job titles & descriptions for AI keywords,
   creating a time-varying exposure measure (vs static ILO theoretical score).
2. Token-count control: add log_token_count as control in topic decomposition
   regressions, since XGBoost showed 70% of entropy driven by text length.

Input:
  data/Heterogeneity/master_with_ai_exposure_v2.csv (job titles + metadata)
  output/processed_corpus.jsonl (tokenized job descriptions, for full-text scan)

Output:
  output/ai_adoption/
"""

import json
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── Paths ──────────────────────────────────────────────────────────
BASE = Path('/Users/yu/code/code2601/TY')
MASTER = BASE / 'data/Heterogeneity/master_with_ai_exposure_v2.csv'
CORPUS = BASE / 'output/processed_corpus.jsonl'
OUT = BASE / 'output/ai_adoption'
OUT.mkdir(parents=True, exist_ok=True)

CHUNK = 500_000
REF_YEAR = 2016

# ── AI Keywords ────────────────────────────────────────────────────
# Conservative list: terms that unambiguously indicate AI involvement
AI_KEYWORDS_TITLE = [
    '人工智能', 'AI', 'ai', '机器学习', '深度学习', '大模型', 'LLM', 'llm',
    'ChatGPT', 'chatgpt', 'AIGC', 'aigc', 'GPT', 'gpt',
    '自然语言处理', 'NLP', 'nlp',
    '计算机视觉', '图像识别', '语音识别',
    '神经网络', '算法工程', '数据标注',
    '智能客服', '智能推荐', '智能驾驶',
]

# For tokenized text: keywords as they'd appear after jieba segmentation
AI_KEYWORDS_TEXT = set([
    '人工智能', '机器学习', '深度学习', '大模型', '大语言模型',
    '自然语言处理', '计算机视觉', '图像识别', '语音识别',
    '神经网络', '卷积神经网络', '循环神经网络',
    'ai', 'AI', 'nlp', 'NLP', 'llm', 'LLM',
    'chatgpt', 'ChatGPT', 'gpt', 'GPT',
    'aigc', 'AIGC',
    'tensorflow', 'TensorFlow', 'pytorch', 'PyTorch',
    '数据标注', '模型训练', '模型部署',
    '智能客服', '智能推荐', '智能驾驶', '智能制造',
    '算法工程师', '算法',
    'bert', 'BERT', 'transformer', 'Transformer',
])

# Compile regex for title scanning
AI_PATTERN = re.compile('|'.join(re.escape(kw) for kw in AI_KEYWORDS_TITLE),
                        re.IGNORECASE)


# ── Topic classification (reuse from experiment_topic_decomposition.py) ──
TOPIC_TO_CATEGORY = {
    7: 'NRA', 8: 'NRA', 14: 'NRA', 15: 'NRA', 26: 'NRA', 29: 'NRA',
    30: 'NRA', 31: 'NRA', 34: 'NRA', 45: 'NRA', 48: 'NRA', 54: 'NRA',
    56: 'NRA', 57: 'NRA',
    0: 'NRI', 2: 'NRI', 9: 'NRI', 11: 'NRI', 12: 'NRI', 13: 'NRI',
    17: 'NRI', 20: 'NRI', 24: 'NRI', 33: 'NRI', 38: 'NRI', 40: 'NRI',
    41: 'NRI', 44: 'NRI', 46: 'NRI', 49: 'NRI', 50: 'NRI',
    1: 'RC', 10: 'RC', 23: 'RC', 25: 'RC', 37: 'RC', 42: 'RC',
    43: 'RC', 53: 'RC', 58: 'RC',
    16: 'RM', 18: 'RM', 22: 'RM', 51: 'RM', 52: 'RM',
    3: 'M', 4: 'M', 5: 'M', 6: 'M', 19: 'M', 21: 'M', 27: 'M',
    28: 'M', 32: 'M', 35: 'M', 36: 'M', 39: 'M', 47: 'M', 55: 'M', 59: 'M',
}
SUBSTANTIVE_CATS = ['NRA', 'NRI', 'RC', 'RM']


def scan_titles_for_ai():
    """Scan job titles in master data for AI keywords."""
    print('\n[1] Scanning job titles for AI keywords ...')
    t0 = time.time()

    cols = ['year', 'isco08_4digit', 'ai_mean_score', '招聘岗位',
            'entropy_score', 'token_count', 'dominant_topic_id']
    cells = {}
    total = 0
    ai_count = 0

    for i, chunk in enumerate(pd.read_csv(MASTER, usecols=cols,
                                           chunksize=CHUNK)):
        total += len(chunk)
        valid = chunk.dropna(subset=['isco08_4digit', 'year', 'entropy_score',
                                      'dominant_topic_id'])
        valid = valid.copy()
        valid['isco'] = valid['isco08_4digit'].astype(int)
        valid['yr'] = valid['year'].astype(int)
        valid['topic'] = valid['dominant_topic_id'].astype(int)
        valid['log_tc'] = np.log(valid['token_count'].clip(lower=1))

        # Scan job titles
        titles = valid['招聘岗位'].fillna('').astype(str)
        valid['has_ai'] = titles.apply(lambda t: 1 if AI_PATTERN.search(t) else 0)
        ai_count += valid['has_ai'].sum()

        # Aggregate to ISCO×year cells
        for (isco, yr), grp in valid.groupby(['isco', 'yr']):
            key = (isco, yr)
            if key not in cells:
                cells[key] = {
                    'n': 0, 'sum_ent': 0.0, 'sum_log_tc': 0.0,
                    'ai': float(grp['ai_mean_score'].iloc[0]),
                    'ai_mentions': 0, 'topic_counts': {},
                }
            c = cells[key]
            c['n'] += len(grp)
            c['sum_ent'] += grp['entropy_score'].sum()
            c['sum_log_tc'] += grp['log_tc'].sum()
            c['ai_mentions'] += grp['has_ai'].sum()
            for t, cnt in grp['topic'].value_counts().items():
                c['topic_counts'][t] = c['topic_counts'].get(t, 0) + cnt

        if (i + 1) % 4 == 0:
            print(f'  Chunk {i+1}: {total:>10,} rows, AI mentions: {ai_count:,}')

    print(f'  Total: {total:,} rows, AI title mentions: {ai_count:,} '
          f'({ai_count/total*100:.2f}%)')
    print(f'  Cells: {len(cells):,}')
    print(f'  Time: {time.time()-t0:.1f}s')

    # Build DataFrame
    all_topics = set()
    for c in cells.values():
        all_topics.update(c['topic_counts'].keys())
    all_topics = sorted(all_topics)

    rows = []
    for (isco, yr), c in cells.items():
        row = {
            'isco': isco, 'year': yr, 'n': c['n'],
            'mean_entropy': c['sum_ent'] / c['n'],
            'mean_log_tc': c['sum_log_tc'] / c['n'],
            'ai_score': c['ai'],
            'ai_adoption_rate': c['ai_mentions'] / c['n'],
            'ai_mentions': c['ai_mentions'],
        }
        # Category shares
        cat_counts = {}
        for t, cnt in c['topic_counts'].items():
            cat = TOPIC_TO_CATEGORY.get(t, 'M')
            cat_counts[cat] = cat_counts.get(cat, 0) + cnt
        for cat in SUBSTANTIVE_CATS + ['M']:
            row[f'share_{cat}'] = cat_counts.get(cat, 0) / c['n']
        rows.append(row)

    df = pd.DataFrame(rows).sort_values(['isco', 'year']).reset_index(drop=True)
    return df


def scan_full_text_for_ai():
    """Scan processed_corpus.jsonl for AI keywords in tokenized text."""
    if not CORPUS.exists():
        print('[!] processed_corpus.jsonl not found, skipping full-text scan')
        return None

    print('\n[1b] Scanning full text (processed_corpus.jsonl) for AI keywords ...')
    t0 = time.time()

    ai_by_id = {}  # id -> has_ai
    total = 0
    ai_count = 0

    with open(CORPUS, 'r') as f:
        for line in f:
            total += 1
            doc = json.loads(line)
            doc_id = doc.get('id')
            tokens = doc.get('tokens', [])
            has_ai = 1 if any(t in AI_KEYWORDS_TEXT for t in tokens) else 0
            ai_by_id[doc_id] = has_ai
            ai_count += has_ai

            if total % 1_000_000 == 0:
                print(f'  {total/1e6:.0f}M docs, AI mentions: {ai_count:,} '
                      f'({ai_count/total*100:.2f}%)')

    print(f'  Total: {total:,} docs, AI mentions: {ai_count:,} '
          f'({ai_count/total*100:.2f}%)')
    print(f'  Time: {time.time()-t0:.1f}s')

    return ai_by_id


def merge_fulltext_ai(df_cells, ai_by_id):
    """Merge full-text AI adoption rates into cell-level data."""
    if ai_by_id is None:
        return df_cells

    print('\n[1c] Merging full-text AI adoption into cells ...')
    # Need to re-aggregate with full-text AI indicator
    # Since we already have cell-level data, we need individual-level to merge
    # This would require re-reading master data with id column
    # For efficiency, let's use the title-based measure and report full-text
    # stats separately
    total_ai = sum(ai_by_id.values())
    total_docs = len(ai_by_id)
    print(f'  Full-text AI adoption rate: {total_ai/total_docs*100:.2f}%')
    print(f'  (Using title-based measure for regressions, full-text for descriptive stats)')

    # Save full-text stats by year
    year_stats = {}
    for doc_id, has_ai in ai_by_id.items():
        yr = int(str(doc_id)[:4]) if doc_id else 0
        if yr not in year_stats:
            year_stats[yr] = {'n': 0, 'ai': 0}
        year_stats[yr]['n'] += 1
        year_stats[yr]['ai'] += has_ai

    lines = ['AI Keyword Mentions in Full Text by Year',
             '=' * 50,
             f'{"Year":<8} {"Total":>10} {"AI":>10} {"Rate":>8}',
             '-' * 40]
    for yr in sorted(year_stats.keys()):
        s = year_stats[yr]
        rate = s['ai'] / s['n'] * 100 if s['n'] > 0 else 0
        lines.append(f'{yr:<8} {s["n"]:>10,} {s["ai"]:>10,} {rate:>7.2f}%')
    with open(OUT / 'fulltext_ai_by_year.txt', 'w') as f:
        f.write('\n'.join(lines))
    print('\n'.join(lines))

    return df_cells


def descriptive_stats(df):
    """Report descriptive statistics of AI adoption measure."""
    print('\n[2] AI Adoption Descriptive Statistics ...')

    # By year
    lines = []
    lines.append('AI Title Mention Rate by Year')
    lines.append('=' * 60)
    lines.append(f'{"Year":<8} {"Cells":>6} {"Total N":>12} {"AI Mentions":>12} {"Rate":>8}')
    lines.append('-' * 50)

    for yr in sorted(df.year.unique()):
        sub = df[df.year == yr]
        total_n = sub.n.sum()
        total_ai = sub.ai_mentions.sum()
        rate = total_ai / total_n * 100 if total_n > 0 else 0
        lines.append(f'{yr:<8} {len(sub):>6} {total_n:>12,} {total_ai:>12,} {rate:>7.3f}%')
        print(f'  {yr}: {total_ai:,}/{total_n:,} = {rate:.3f}%')

    # By AI exposure gradient
    lines.append('')
    lines.append('AI Title Mention Rate by ILO Exposure Level')
    lines.append('=' * 60)

    # Add exposure level
    ilo = pd.read_csv(BASE / 'data/esco/ilo_genai_isco08_2025.csv')
    ilo_lookup = dict(zip(ilo['isco08_4digit'], ilo['exposure_gradient']))
    df['gradient'] = df['isco'].map(ilo_lookup)

    for grad in ['Gradient 4', 'Gradient 3', 'Gradient 2', 'Gradient 1',
                 'Minimal Exposure', 'Not Exposed']:
        sub = df[df.gradient == grad]
        if len(sub) == 0:
            continue
        total_n = sub.n.sum()
        total_ai = sub.ai_mentions.sum()
        rate = total_ai / total_n * 100 if total_n > 0 else 0
        lines.append(f'  {grad:<20} {rate:>7.3f}% ({total_ai:,}/{total_n:,})')

    # Correlation between ILO score and adoption rate
    weighted_corr = np.average(
        (df['ai_adoption_rate'] - np.average(df['ai_adoption_rate'], weights=df['n']))
        * (df['ai_score'] - np.average(df['ai_score'], weights=df['n'])),
        weights=df['n']
    ) / (
        np.sqrt(np.average((df['ai_adoption_rate'] - np.average(df['ai_adoption_rate'], weights=df['n']))**2, weights=df['n']))
        * np.sqrt(np.average((df['ai_score'] - np.average(df['ai_score'], weights=df['n']))**2, weights=df['n']))
    )
    lines.append(f'\nWeighted correlation(ILO score, adoption rate): {weighted_corr:.4f}')

    result_text = '\n'.join(lines)
    print(result_text)
    with open(OUT / 'descriptive_stats.txt', 'w') as f:
        f.write(result_text)

    return df


def regression_entropy_with_controls(df):
    """Regression on entropy with token_count control + AI adoption."""
    print('\n[3] Entropy Regressions with Controls')
    print('=' * 70)

    lines = []

    # Model A: Original (no controls) - baseline
    print('\n--- Model A: Baseline (ILO score, no controls) ---')
    mA = smf.wls('mean_entropy ~ C(year) * ai_score',
                  data=df, weights=df['n']).fit(cov_type='HC3')
    int_pA = {k: v for k, v in mA.pvalues.items() if ':ai_score' in k}
    r_A = np.zeros((len(int_pA), len(mA.params)))
    for i, name in enumerate(sorted(int_pA.keys())):
        r_A[i, list(mA.params.index).index(name)] = 1
    fA = mA.f_test(r_A)
    f_pA = float(np.asarray(fA.pvalue).flat[0])
    lines.append(f'Model A (baseline): Joint F p = {f_pA:.4f}, R² = {mA.rsquared:.6f}')
    print(f'  Joint F p = {f_pA:.4f}')

    # Model B: + log_token_count control
    print('\n--- Model B: + log_token_count control ---')
    mB = smf.wls('mean_entropy ~ C(year) * ai_score + mean_log_tc',
                  data=df, weights=df['n']).fit(cov_type='HC3')
    int_pB = {k: v for k, v in mB.pvalues.items() if ':ai_score' in k}
    r_B = np.zeros((len(int_pB), len(mB.params)))
    for i, name in enumerate(sorted(int_pB.keys())):
        r_B[i, list(mB.params.index).index(name)] = 1
    fB = mB.f_test(r_B)
    f_pB = float(np.asarray(fB.pvalue).flat[0])
    tc_coef = mB.params['mean_log_tc']
    tc_p = mB.pvalues['mean_log_tc']
    lines.append(f'Model B (+log_tc): Joint F p = {f_pB:.4f}, R² = {mB.rsquared:.6f}')
    lines.append(f'  log_token_count: coef={tc_coef:.6f}, p={tc_p:.4f}')
    print(f'  Joint F p = {f_pB:.4f}')
    print(f'  log_tc: coef={tc_coef:.6f}, p={tc_p:.4f}')

    # Model C: AI adoption rate (title-based) instead of ILO score
    print('\n--- Model C: AI adoption rate (title keyword) ---')
    mC = smf.wls('mean_entropy ~ C(year) * ai_adoption_rate',
                  data=df, weights=df['n']).fit(cov_type='HC3')
    int_pC = {k: v for k, v in mC.pvalues.items() if ':ai_adoption_rate' in k}
    if int_pC:
        r_C = np.zeros((len(int_pC), len(mC.params)))
        for i, name in enumerate(sorted(int_pC.keys())):
            r_C[i, list(mC.params.index).index(name)] = 1
        fC = mC.f_test(r_C)
        f_pC = float(np.asarray(fC.pvalue).flat[0])
    else:
        f_pC = 1.0
    lines.append(f'Model C (AI adoption): Joint F p = {f_pC:.4f}, R² = {mC.rsquared:.6f}')
    print(f'  Joint F p = {f_pC:.4f}')

    # Print post-2023 interactions for Model C
    for k in sorted(int_pC.keys()):
        if any(str(y) in k for y in [2023, 2024, 2025]):
            yr = k.split('[T.')[1].split(']')[0]
            print(f'    {yr}: coef={mC.params[k]:+.6f}, p={mC.pvalues[k]:.4f}')
            lines.append(f'    {yr}: coef={mC.params[k]:+.6f}, p={mC.pvalues[k]:.4f}')

    # Model D: AI adoption rate + log_tc control
    print('\n--- Model D: AI adoption + log_tc control ---')
    mD = smf.wls('mean_entropy ~ C(year) * ai_adoption_rate + mean_log_tc',
                  data=df, weights=df['n']).fit(cov_type='HC3')
    int_pD = {k: v for k, v in mD.pvalues.items() if ':ai_adoption_rate' in k}
    if int_pD:
        r_D = np.zeros((len(int_pD), len(mD.params)))
        for i, name in enumerate(sorted(int_pD.keys())):
            r_D[i, list(mD.params.index).index(name)] = 1
        fD = mD.f_test(r_D)
        f_pD = float(np.asarray(fD.pvalue).flat[0])
    else:
        f_pD = 1.0
    lines.append(f'Model D (adoption+tc): Joint F p = {f_pD:.4f}, R² = {mD.rsquared:.6f}')
    print(f'  Joint F p = {f_pD:.4f}')

    for k in sorted(int_pD.keys()):
        if any(str(y) in k for y in [2023, 2024, 2025]):
            yr = k.split('[T.')[1].split(']')[0]
            print(f'    {yr}: coef={mD.params[k]:+.6f}, p={mD.pvalues[k]:.4f}')
            lines.append(f'    {yr}: coef={mD.params[k]:+.6f}, p={mD.pvalues[k]:.4f}')

    with open(OUT / 'entropy_regressions.txt', 'w') as f:
        f.write('Entropy Regressions: ILO Score vs AI Adoption, ± Token Count Control\n')
        f.write('=' * 70 + '\n')
        f.write('\n'.join(lines))

    return {'A': f_pA, 'B': f_pB, 'C': f_pC, 'D': f_pD}


def regression_categories_with_controls(df):
    """Category share regressions with AI adoption measure + log_tc control."""
    print('\n[4] Category Share Regressions with AI Adoption')
    print('=' * 70)

    results = {}
    lines = []

    for cat in SUBSTANTIVE_CATS:
        dv = f'share_{cat}'
        lines.append(f'\n{"=" * 60}')
        lines.append(f'{cat}')
        lines.append(f'{"=" * 60}')

        # Model with ILO score + log_tc control
        m1 = smf.wls(f'{dv} ~ C(year) * ai_score + mean_log_tc',
                      data=df, weights=df['n']).fit(cov_type='HC3')
        int_p1 = {k: v for k, v in m1.pvalues.items() if ':ai_score' in k}
        r1 = np.zeros((len(int_p1), len(m1.params)))
        for i, name in enumerate(sorted(int_p1.keys())):
            r1[i, list(m1.params.index).index(name)] = 1
        f1 = m1.f_test(r1)
        f_p1 = float(np.asarray(f1.pvalue).flat[0])

        # Model with AI adoption rate + log_tc control
        m2 = smf.wls(f'{dv} ~ C(year) * ai_adoption_rate + mean_log_tc',
                      data=df, weights=df['n']).fit(cov_type='HC3')
        int_p2 = {k: v for k, v in m2.pvalues.items() if ':ai_adoption_rate' in k}
        if int_p2:
            r2 = np.zeros((len(int_p2), len(m2.params)))
            for i, name in enumerate(sorted(int_p2.keys())):
                r2[i, list(m2.params.index).index(name)] = 1
            f2 = m2.f_test(r2)
            f_p2 = float(np.asarray(f2.pvalue).flat[0])
        else:
            f_p2 = 1.0

        lines.append(f'  ILO score + log_tc: Joint F p = {f_p1:.4f}')
        lines.append(f'  AI adoption + log_tc: Joint F p = {f_p2:.4f}')

        # Post-2023 for AI adoption model
        for k in sorted(int_p2.keys()):
            if any(str(y) in k for y in [2023, 2024, 2025]):
                yr = k.split('[T.')[1].split(']')[0]
                lines.append(f'    {yr}: coef={m2.params[k]:+.6f}, p={m2.pvalues[k]:.4f}')

        print(f'  {cat}: ILO+tc p={f_p1:.4f}, Adoption+tc p={f_p2:.4f}')

        results[cat] = {'ilo_tc_p': f_p1, 'adopt_tc_p': f_p2}

    with open(OUT / 'category_regressions.txt', 'w') as f:
        f.write('Category Share Regressions: ILO vs AI Adoption, with log_tc\n')
        f.write('=' * 70 + '\n')
        f.write('\n'.join(lines))

    return results


def visualize(df):
    """Generate plots."""
    print('\n[5] Generating visualizations ...')

    # Plot 1: AI adoption rate over time
    fig, ax = plt.subplots(figsize=(10, 6))
    yearly = df.groupby('year').apply(
        lambda g: pd.Series({
            'rate': g['ai_mentions'].sum() / g['n'].sum() * 100,
            'n': g['n'].sum(),
        })
    )
    ax.bar(yearly.index, yearly['rate'], color='#2563EB', alpha=0.8,
           edgecolor='grey', lw=0.5)
    ax.set_xlabel('Year')
    ax.set_ylabel('AI Keyword Mention Rate (%)')
    ax.set_title('AI Adoption in Job Titles Over Time\n'
                 '(% of postings mentioning AI/ML/DL/LLM keywords)')
    for i, (yr, row) in enumerate(yearly.iterrows()):
        ax.text(yr, row['rate'] + 0.02, f'{row["rate"]:.2f}%',
                ha='center', fontsize=9)
    plt.tight_layout()
    plt.savefig(OUT / 'ai_adoption_trend.png', dpi=150)
    plt.close()
    print('  Saved ai_adoption_trend.png')

    # Plot 2: AI adoption by ILO exposure gradient
    if 'gradient' in df.columns:
        fig, ax = plt.subplots(figsize=(10, 6))
        grad_order = ['Gradient 4', 'Gradient 3', 'Gradient 2', 'Gradient 1',
                      'Minimal Exposure', 'Not Exposed']
        rates = []
        labels = []
        for grad in grad_order:
            sub = df[df.gradient == grad]
            if len(sub) == 0:
                continue
            rate = sub.ai_mentions.sum() / sub.n.sum() * 100
            rates.append(rate)
            labels.append(grad)

        colors = ['#e74c3c', '#e67e22', '#f1c40f', '#2ecc71', '#3498db', '#95a5a6']
        ax.barh(range(len(labels)), rates, color=colors[:len(labels)],
                alpha=0.8, edgecolor='grey', lw=0.5)
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels)
        ax.set_xlabel('AI Keyword Mention Rate (%)')
        ax.set_title('AI Adoption Rate by ILO Exposure Level\n'
                     '(Validation: do high-exposure occupations actually adopt AI?)')
        ax.invert_yaxis()
        plt.tight_layout()
        plt.savefig(OUT / 'adoption_by_gradient.png', dpi=150)
        plt.close()
        print('  Saved adoption_by_gradient.png')

    # Plot 3: AI adoption × entropy scatter by year
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Pre-2023
    pre = df[(df.year >= 2016) & (df.year <= 2022) & (df.year != 2019) & (df.year != 2020)]
    ax = axes[0]
    ax.scatter(pre['ai_adoption_rate'] * 100, pre['mean_entropy'],
               s=np.sqrt(pre['n']) / 5, alpha=0.3, c='#3498db')
    ax.set_xlabel('AI Adoption Rate (%)')
    ax.set_ylabel('Mean Entropy')
    ax.set_title('Pre-2023 (2016-2022)')

    # Post-2023
    post = df[df.year >= 2023]
    ax = axes[1]
    ax.scatter(post['ai_adoption_rate'] * 100, post['mean_entropy'],
               s=np.sqrt(post['n']) / 5, alpha=0.3, c='#e74c3c')
    ax.set_xlabel('AI Adoption Rate (%)')
    ax.set_ylabel('Mean Entropy')
    ax.set_title('Post-2023 (2023-2025)')

    fig.suptitle('AI Adoption Rate vs Entropy by Period', fontsize=13)
    plt.tight_layout()
    plt.savefig(OUT / 'adoption_vs_entropy.png', dpi=150)
    plt.close()
    print('  Saved adoption_vs_entropy.png')


# ── Main ────────────────────────────────────────────────────────────
if __name__ == '__main__':
    t_start = time.time()

    print('=' * 70)
    print('EXPERIMENT: AI Adoption Measure + Token Count Control')
    print('=' * 70)

    # Step 1: Scan job titles
    df = scan_titles_for_ai()

    # Step 1b: Full text scan (if available)
    ai_by_id = scan_full_text_for_ai()
    df = merge_fulltext_ai(df, ai_by_id)

    # Step 2: Descriptive stats
    df = descriptive_stats(df)

    # Step 3: Entropy regressions
    entropy_results = regression_entropy_with_controls(df)

    # Step 4: Category share regressions
    cat_results = regression_categories_with_controls(df)

    # Step 5: Visualization
    visualize(df)

    elapsed = time.time() - t_start
    print(f'\n{"=" * 70}')
    print(f'DONE in {elapsed:.1f}s')
    print(f'\nSUMMARY:')
    print(f'  Entropy regressions (Joint F p-values):')
    for k, v in entropy_results.items():
        print(f'    Model {k}: p = {v:.4f}')
    print(f'  Category regressions (Joint F p-values):')
    for cat, r in cat_results.items():
        print(f'    {cat}: ILO+tc p={r["ilo_tc_p"]:.4f}, Adopt+tc p={r["adopt_tc_p"]:.4f}')
    print(f'\nAll results saved to {OUT}')
