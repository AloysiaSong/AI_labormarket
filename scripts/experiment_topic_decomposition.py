#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
experiment_topic_decomposition.py

Experiment 2: Decomposed outcome variable.
Instead of aggregate entropy, decompose into Spitz-Oener task categories
and individual topic shares, then test whether AI exposure predicts
differential changes in specific skill components.

Part A: Classify LDA topics into Spitz-Oener categories (NRA/NRI/RC/RM),
        run WLS regressions on category shares × AI exposure × year.
Part B: Run 60 separate WLS regressions on individual topic shares,
        apply Benjamini-Hochberg FDR correction.

Input:  data/Heterogeneity/master_with_ai_exposure_v2.csv
Output: output/topic_decomposition/
"""

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
TOPIC_CSV = BASE / 'output/lda/alignment/topic_labels_template.csv'
OUT = BASE / 'output/topic_decomposition'
OUT.mkdir(parents=True, exist_ok=True)

CHUNK = 500_000
REF_YEAR = 2016

# ── Topic Classification ──────────────────────────────────────────
# Spitz-Oener (2006) task categories:
#   NRA = Non-routine analytical (tech, development, analysis, design, R&D)
#   NRI = Non-routine interactive (communication, management, marketing, sales, education)
#   RC  = Routine cognitive (accounting, admin, filing, procurement, logistics)
#   RM  = Routine manual (manufacturing, equipment, construction, driving)
#   M   = Meta/Generic (soft skills, benefits, education requirements, boilerplate)

TOPIC_TO_CATEGORY = {
    # NRA: Non-routine analytical
    7: 'NRA',   # 精通, 技术, 熟练, 开发, 移动 → Tech/mobile development
    8: 'NRA',   # 检测, 化工, 环境, 化学, 分析 → Chemical analysis/testing
    14: 'NRA',  # 设计, 电子, 熟练, 硬件, 开发 → Hardware/electronics design
    15: 'NRA',  # 计算机, 电子, 本科, 自动化, 大专 → Computer/automation engineering
    26: 'NRA',  # 测试, 方法, 工具, 理论, 流程 → Testing methodology
    29: 'NRA',  # 产品, 研发, 技术, 设计, 产品设计 → Product R&D
    30: 'NRA',  # 网络, 系统, 维护, 计算机相关, 技术 → Network/systems tech
    31: 'NRA',  # 分析, 信息, 方案, 需求, 收集 → Information analysis
    34: 'NRA',  # 设计, 软件, 平面设计, 熟练, 制作 → Graphic/software design
    45: 'NRA',  # 数据, 数据分析, 统计, 已婚, 分析 → Data analysis
    48: 'NRA',  # 设计, 软件, 熟练, CAD, 室内设计 → CAD/interior design
    54: 'NRA',  # 开发, 精通, 熟练, 数据库, 技术 → Database/software dev
    56: 'NRA',  # 互联网, 行业, 游戏, 理解, 思维 → Internet/gaming tech
    57: 'NRA',  # 开发, 精通, 编程, Android, C++ → Programming

    # NRI: Non-routine interactive
    0: 'NRI',   # 沟通, 协调, 市场营销, 组织, 大专 → Marketing coordination
    2: 'NRI',   # 擅长, 文字_功底, 文字, 编辑, 写作能力 → Writing/editing
    9: 'NRI',   # 运营, 推广, 网站, 电子商务, 平台 → E-commerce operations
    11: 'NRI',  # 金融, 从业, 行业, 投资, 本科 → Finance/investment
    12: 'NRI',  # 大型, 招商, 商业, 酒店, 零售 → Commercial/retail
    13: 'NRI',  # 招聘, 人力资源, 人力资源管理, 大专, 流程 → HR/recruitment
    17: 'NRI',  # 策划, 广告, 媒体, 品牌, 活动 → Advertising/media
    20: 'NRI',  # 市场, 渠道, 资源, 营销, 拓展 → Marketing/channels
    24: 'NRI',  # 培训, 经理, 主管, 学习, 员工 → Training/management
    33: 'NRI',  # 项目, 项目管理, 实施, 行业, 政府 → Project management
    38: 'NRI',  # 安全, 顾客, 商品, 店铺, 门店 → Retail store management
    40: 'NRI',  # 教育, 教学, 沟通, 压力, 热爱 → Education/teaching
    41: 'NRI',  # 房地产, 医疗, 行业, 医药, 医院 → Real estate/healthcare
    44: 'NRI',  # 计划, 部门, 组织, 制定, 管理 → Planning/org management
    46: 'NRI',  # 沟通, 亲和力, 形象_气质佳, 服务_意识 → Customer service
    49: 'NRI',  # 客户, 销售, 需求, 服务, 沟通 → Sales/customer service
    50: 'NRI',  # 销售, 行业, 沟通, 大专, 市场营销 → Sales/marketing

    # RC: Routine cognitive
    1: 'RC',    # 物流, 业务流程, 基本知识, 行业 → Logistics/business process
    10: 'RC',   # 财务, 会计, 大专, 财务管理, 财务软件 → Accounting/finance
    23: 'RC',   # 管理, 行政, 大专, 管理工作, 本科 → Administrative mgmt
    25: 'RC',   # 仓库, ERP, 管理流程, 适当_放宽 → Warehouse/ERP
    37: 'RC',   # 采购, 流程, 供应商, 物料, 材料 → Procurement/supply chain
    42: 'RC',   # 法律, 食品, 口头_表达能力, 流程 → Legal/office procedures
    43: 'RC',   # 做好, 日常, 资料, 整理, 文件 → Filing/documentation
    53: 'RC',   # 熟练, 办公软件, 操作, 大专, 沟通 → Office software ops
    58: 'RC',   # 业务, 汽车, 外贸, 操作, 流程 → Business ops/trade

    # RM: Routine manual
    16: 'RM',   # 生产, 问题, 质量, 解决, 工艺 → Manufacturing/QC
    18: 'RM',   # 机械, 电气, 自动化, 设备, 大专 → Mechanical/electrical
    22: 'RM',   # 设备, 维修, 系统, 维护, 安装 → Equipment maintenance
    51: 'RM',   # 工程, 施工, 建筑, 管理, 现场 → Construction
    52: 'RM',   # 驾驶, 驾照, 车辆, 责任心, 年龄 → Driving/transport

    # Meta/Generic (excluded from main analysis)
    3: 'M',     # 学习, 沟通, 热爱, 责任心, 吃苦耐劳 → Generic soft skills
    4: 'M',     # 必须, 时间, 面试, 简历, 应聘 → Application reqs
    5: 'M',     # 应届_毕业生, 本科学历, 全日制 → Education reqs
    6: 'M',     # 福利, 旅游, 员工, 免费, 享受 → Benefits/perks
    19: 'M',    # 能够, 接受, 适应, 出差 → Travel/adaptability
    21: 'M',    # 证书, 地址, 持有, 联系人 → Certificates/contact
    27: 'M',    # 知识, 基本, 客服, 服务, 技能 → Basic knowledge (generic)
    28: 'M',    # 沟通, 责任心, 团队合作_精神, 学习 → Teamwork/soft skills
    32: 'M',    # 薪资, 待遇, 工资, 提成, 学习 → Salary/compensation
    35: 'M',    # 英语, 英文, 水平, 本科 → English language reqs
    36: 'M',    # 我们, 自己, 一个, 只要, 如果 → Informal/casual text
    39: 'M',    # 行业, 学习, 大专, 应届生, 热爱 → Industry/entry-level
    47: 'M',    # 职位, 条件, 职位_描述, 地点 → Job desc boilerplate
    55: 'M',    # 年龄, 吃苦耐劳, 男女, 高中 → Physical/age reqs
    59: 'M',    # 药学, 适应_加班, 农业, 制药 → Mixed/pharmaceutical
}

CATEGORY_LABELS = {
    'NRA': 'Non-routine Analytical',
    'NRI': 'Non-routine Interactive',
    'RC':  'Routine Cognitive',
    'RM':  'Routine Manual',
    'M':   'Meta/Generic',
}

SUBSTANTIVE_CATS = ['NRA', 'NRI', 'RC', 'RM']  # exclude Meta


def save_topic_classification():
    """Save topic classification to CSV for manual review."""
    try:
        tpl = pd.read_csv(TOPIC_CSV)
    except Exception:
        tpl = pd.DataFrame({'Base_Topic_ID': range(60),
                            'Top_Keywords': [''] * 60})

    rows = []
    for _, row in tpl.iterrows():
        tid = int(row['Base_Topic_ID'])
        cat = TOPIC_TO_CATEGORY.get(tid, 'M')
        rows.append({
            'topic_id': tid,
            'keywords': row.get('Top_Keywords', ''),
            'category': cat,
            'category_label': CATEGORY_LABELS.get(cat, ''),
        })
    out_df = pd.DataFrame(rows)
    out_df.to_csv(OUT / 'topic_classification.csv', index=False)
    print(f'  Topic classification saved ({len(out_df)} topics)')

    for cat in ['NRA', 'NRI', 'RC', 'RM', 'M']:
        n = sum(1 for v in TOPIC_TO_CATEGORY.values() if v == cat)
        print(f'    {cat} ({CATEGORY_LABELS[cat]}): {n} topics')
    return out_df


def load_and_aggregate():
    """Load master data, compute ISCO×year topic/category shares."""
    print('\n[1] Loading and aggregating data ...')
    t0 = time.time()

    cols = ['year', 'isco08_4digit', 'ai_mean_score', 'dominant_topic_id',
            'entropy_score']
    cells = {}   # (isco, year) -> {topic_counts, total_n, ai_score, sum_entropy}
    total = 0
    skipped = 0

    for i, chunk in enumerate(pd.read_csv(MASTER, usecols=cols,
                                           chunksize=CHUNK)):
        total += len(chunk)
        valid = chunk.dropna(subset=['isco08_4digit', 'ai_mean_score',
                                      'dominant_topic_id', 'year',
                                      'entropy_score'])
        skipped += len(chunk) - len(valid)
        valid = valid.copy()
        valid['isco'] = valid['isco08_4digit'].astype(int)
        valid['yr'] = valid['year'].astype(int)
        valid['topic'] = valid['dominant_topic_id'].astype(int)

        for (isco, yr), grp in valid.groupby(['isco', 'yr']):
            key = (isco, yr)
            topic_counts = grp['topic'].value_counts().to_dict()
            n = len(grp)
            ai = float(grp['ai_mean_score'].iloc[0])
            s_ent = float(grp['entropy_score'].sum())

            if key not in cells:
                cells[key] = {'counts': {}, 'n': 0, 'ai': ai, 'sum_ent': 0.0}
            c = cells[key]
            for t, cnt in topic_counts.items():
                c['counts'][t] = c['counts'].get(t, 0) + cnt
            c['n'] += n
            c['sum_ent'] += s_ent

        if (i + 1) % 4 == 0:
            print(f'  Chunk {i+1}: {total:>10,} rows, {len(cells):,} cells')

    print(f'  Total: {total:,} rows, {skipped:,} skipped, '
          f'{len(cells):,} ISCO×year cells')
    print(f'  Aggregation time: {time.time()-t0:.1f}s')

    # Discover all topic IDs present in data
    all_topics = set()
    for c in cells.values():
        all_topics.update(c['counts'].keys())
    all_topics = sorted(all_topics)
    n_topics = len(all_topics)
    print(f'  Topics found in data: {n_topics} (range {min(all_topics)}-{max(all_topics)})')

    # Build cell-level DataFrame
    rows = []
    for (isco, yr), c in cells.items():
        row = {
            'isco': isco, 'year': yr,
            'n': c['n'], 'ai_score': c['ai'],
            'mean_entropy': c['sum_ent'] / c['n'],
        }
        # Topic shares
        for t in all_topics:
            row[f'share_t{t}'] = c['counts'].get(t, 0) / c['n']
        # Category shares (excluding Meta from denominator is optional;
        # we use total n as denominator for transparency)
        cat_counts = {}
        for t, cnt in c['counts'].items():
            cat = TOPIC_TO_CATEGORY.get(t, 'M')
            cat_counts[cat] = cat_counts.get(cat, 0) + cnt
        for cat in SUBSTANTIVE_CATS + ['M']:
            row[f'share_{cat}'] = cat_counts.get(cat, 0) / c['n']
        rows.append(row)

    df = pd.DataFrame(rows).sort_values(['isco', 'year']).reset_index(drop=True)
    print(f'\n  Cell DataFrame: {len(df)} cells × {len(df.columns)} columns')
    print(f'  Category share means:')
    for cat in SUBSTANTIVE_CATS + ['M']:
        wmean = np.average(df[f'share_{cat}'], weights=df['n'])
        print(f'    {cat} ({CATEGORY_LABELS[cat]}): {wmean:.4f}')

    return df, all_topics


def run_category_regressions(df):
    """Part A: WLS regressions with Spitz-Oener category shares as DV."""
    print('\n' + '=' * 70)
    print('PART A: Spitz-Oener Category Share Regressions')
    print('=' * 70)

    results = {}
    lines = []

    for cat in SUBSTANTIVE_CATS:
        label = CATEGORY_LABELS[cat]
        dv = f'share_{cat}'

        print(f'\n--- {cat}: {label} ---')

        # Model: share_k ~ C(year) * ai_score, WLS with HC3
        formula = f'{dv} ~ C(year) * ai_score'
        m = smf.wls(formula, data=df, weights=df['n']).fit(cov_type='HC3')

        # Extract interaction terms
        int_params = {k: v for k, v in m.params.items() if ':ai_score' in k}
        int_pvals = {k: v for k, v in m.pvalues.items() if ':ai_score' in k}

        # Joint F-test on interactions
        r_matrix = np.zeros((len(int_params), len(m.params)))
        int_names = sorted(int_params.keys())
        for i, name in enumerate(int_names):
            r_matrix[i, list(m.params.index).index(name)] = 1
        f_test = m.f_test(r_matrix)
        f_val = float(np.asarray(f_test.fvalue).flat[0])
        f_p = float(np.asarray(f_test.pvalue).flat[0])

        # Post-2023 interactions
        post_ints = {k: v for k, v in int_params.items()
                     if any(str(y) in k for y in [2023, 2024, 2025])}

        lines.append('=' * 70)
        lines.append(f'{cat}: {label}')
        lines.append(f'DV = {dv}')
        lines.append(f'Model: {formula}')
        lines.append(f'N = {len(df)} cells, R² = {m.rsquared:.6f}')
        lines.append('=' * 70)
        lines.append(f'\n{"Year×ai_score interactions":}')
        lines.append(f'{"Term":<35} {"Coef":>10} {"SE":>10} {"p":>8}')
        lines.append('-' * 65)

        for k in sorted(int_params.keys()):
            yr = k.split('[T.')[1].split(']')[0]
            lines.append(f'  {yr} × ai_score'
                         f'  {int_params[k]:>10.6f}'
                         f'  {m.bse[k]:>10.6f}'
                         f'  {int_pvals[k]:>8.4f}')
            print(f'  {yr} × ai_score: {int_params[k]:+.6f}  p={int_pvals[k]:.4f}')

        lines.append(f'\nJoint F-test (all year×ai_score = 0):')
        lines.append(f'  F = {f_val:.4f}, p = {f_p:.4f}')
        print(f'  Joint F: {f_val:.4f}, p = {f_p:.4f}')

        if post_ints:
            lines.append(f'\nPost-2023 interactions:')
            for k in sorted(post_ints.keys()):
                yr = k.split('[T.')[1].split(']')[0]
                lines.append(f'  {yr}: coef={post_ints[k]:+.6f}, p={int_pvals[k]:.4f}')

        lines.append('\n')

        results[cat] = {
            'model': m,
            'f_val': f_val,
            'f_p': f_p,
            'int_params': int_params,
            'int_pvals': int_pvals,
        }

    # Save
    with open(OUT / 'part_a_category_regressions.txt', 'w') as f:
        f.write('PART A: Spitz-Oener Category Share × AI Exposure\n')
        f.write(f'DV = share of jobs in each task category (based on dominant LDA topic)\n')
        f.write(f'Model: share_k ~ C(year) * ai_score, WLS weighted by cell n, HC3 SEs\n')
        f.write(f'N = {len(df)} ISCO×year cells\n\n')
        f.write('\n'.join(lines))

    # Summary table
    lines_summary = []
    lines_summary.append('SUMMARY: Joint F-tests for year × ai_score interactions')
    lines_summary.append('=' * 60)
    lines_summary.append(f'{"Category":<30} {"F":>8} {"p":>8} {"Sig":>5}')
    lines_summary.append('-' * 60)
    for cat in SUBSTANTIVE_CATS:
        r = results[cat]
        sig = '***' if r['f_p'] < 0.01 else '**' if r['f_p'] < 0.05 else '*' if r['f_p'] < 0.1 else ''
        lines_summary.append(f'{cat} ({CATEGORY_LABELS[cat]})'.ljust(30)
                             + f'{r["f_val"]:>8.4f}'
                             + f'{r["f_p"]:>8.4f}'
                             + f'{sig:>5}')
    lines_summary.append('')
    print('\n' + '\n'.join(lines_summary))

    return results


def run_topic_regressions(df, all_topics):
    """Part B: Individual topic share regressions with FDR correction."""
    print('\n' + '=' * 70)
    print('PART B: Individual Topic Share Regressions (with FDR)')
    print('=' * 70)

    topic_results = []

    for t in all_topics:
        dv = f'share_t{t}'
        if dv not in df.columns or df[dv].std() < 1e-10:
            continue

        try:
            formula = f'{dv} ~ C(year) * ai_score'
            m = smf.wls(formula, data=df, weights=df['n']).fit(cov_type='HC3')

            # Joint F-test on interactions
            int_names = [k for k in m.params.index if ':ai_score' in k]
            if not int_names:
                continue
            r_matrix = np.zeros((len(int_names), len(m.params)))
            for i, name in enumerate(int_names):
                r_matrix[i, list(m.params.index).index(name)] = 1
            f_test = m.f_test(r_matrix)
            f_val = float(np.asarray(f_test.fvalue).flat[0])
            f_p = float(np.asarray(f_test.pvalue).flat[0])

            # Post-2023 interaction average
            post_coefs = []
            for k in int_names:
                if any(str(y) in k for y in [2023, 2024, 2025]):
                    post_coefs.append(m.params[k])
            avg_post = np.mean(post_coefs) if post_coefs else 0.0

            cat = TOPIC_TO_CATEGORY.get(t, 'M')
            wmean = np.average(df[dv], weights=df['n'])

            topic_results.append({
                'topic_id': t,
                'category': cat,
                'mean_share': wmean,
                'f_val': f_val,
                'f_p': f_p,
                'avg_post_coef': avg_post,
                'r2': m.rsquared,
            })
        except Exception as e:
            print(f'  Topic {t}: regression failed ({e})')

    results_df = pd.DataFrame(topic_results).sort_values('f_p')

    # Benjamini-Hochberg FDR correction
    n_tests = len(results_df)
    results_df = results_df.sort_values('f_p').reset_index(drop=True)
    results_df['rank'] = range(1, n_tests + 1)
    results_df['bh_threshold'] = results_df['rank'] / n_tests * 0.10  # q = 0.10
    results_df['fdr_significant'] = results_df['f_p'] <= results_df['bh_threshold']

    # How many survive FDR?
    n_sig_raw = (results_df['f_p'] < 0.05).sum()
    n_sig_fdr = results_df['fdr_significant'].sum()
    print(f'\n  Topics tested: {n_tests}')
    print(f'  Significant at p<0.05 (raw): {n_sig_raw}')
    print(f'  Significant after BH FDR (q=0.10): {n_sig_fdr}')

    # Save detailed results
    results_df.to_csv(OUT / 'part_b_topic_regressions.csv', index=False)

    lines = []
    lines.append('PART B: Individual Topic Share × AI Exposure')
    lines.append(f'Model: share_topic_k ~ C(year) * ai_score, WLS, HC3')
    lines.append(f'N = {len(df)} ISCO×year cells, {n_tests} topics tested')
    lines.append(f'Benjamini-Hochberg FDR correction at q = 0.10')
    lines.append('=' * 70)
    lines.append(f'\nRaw p < 0.05: {n_sig_raw} topics')
    lines.append(f'FDR significant: {n_sig_fdr} topics')
    lines.append(f'\nTop 20 topics by Joint F p-value:')
    lines.append(f'{"Topic":>6} {"Cat":>4} {"Mean%":>7} {"F":>8} {"p":>8} '
                 f'{"FDR":>4} {"AvgPostCoef":>12}')
    lines.append('-' * 55)

    for _, row in results_df.head(20).iterrows():
        fdr_mark = '*' if row['fdr_significant'] else ''
        lines.append(f'{int(row["topic_id"]):>6} {row["category"]:>4} '
                     f'{row["mean_share"]*100:>6.2f}% '
                     f'{row["f_val"]:>8.4f} {row["f_p"]:>8.4f} '
                     f'{fdr_mark:>4} {row["avg_post_coef"]:>+12.6f}')

    with open(OUT / 'part_b_topic_regressions.txt', 'w') as f:
        f.write('\n'.join(lines))

    print('\n' + '\n'.join(lines))
    return results_df


def visualize_results(cat_results, topic_df, df_cells):
    """Generate plots for both parts."""
    print('\n[4] Generating visualizations ...')

    # ── Plot 1: Category share interaction coefficients (post-2023) ──
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('AI Exposure × Year Interactions on Task Category Shares\n'
                 '(WLS, HC3 SEs, ISCO×year cells)', fontsize=13)

    for idx, cat in enumerate(SUBSTANTIVE_CATS):
        ax = axes[idx // 2][idx % 2]
        r = cat_results[cat]
        int_p = r['int_params']

        years = []
        coefs = []
        ses = []
        for k in sorted(int_p.keys()):
            yr = int(k.split('[T.')[1].split(']')[0])
            years.append(yr)
            coefs.append(int_p[k])
            ses.append(r['model'].bse[k])

        colors = ['#e74c3c' if y >= 2023 else '#3498db' for y in years]
        ax.bar(range(len(years)), coefs,
               yerr=[1.96 * s for s in ses],
               capsize=3, alpha=0.75, color=colors, edgecolor='grey', lw=0.5)
        ax.axhline(0, color='black', lw=0.8)
        ax.set_xticks(range(len(years)))
        ax.set_xticklabels(years, rotation=45, fontsize=9)
        ax.set_title(f'{cat}: {CATEGORY_LABELS[cat]}\n'
                     f'Joint F p = {r["f_p"]:.4f}', fontsize=11)
        ax.set_ylabel('Year × AI score coef')

    plt.tight_layout()
    plt.savefig(OUT / 'category_interactions.png', dpi=150)
    plt.close()
    print('  Saved category_interactions.png')

    # ── Plot 2: Topic-level heatmap ──
    if len(topic_df) > 0:
        # Sort by category then by average post coefficient
        topic_df = topic_df.copy()
        cat_order = {'NRA': 0, 'NRI': 1, 'RC': 2, 'RM': 3, 'M': 4}
        topic_df['cat_rank'] = topic_df['category'].map(cat_order)
        topic_df = topic_df.sort_values(['cat_rank', 'avg_post_coef'],
                                         ascending=[True, False])

        # Filter to substantive topics only
        sub = topic_df[topic_df['category'] != 'M'].copy()

        if len(sub) > 0:
            fig, ax = plt.subplots(figsize=(10, max(8, len(sub) * 0.25)))

            colors = []
            for _, row in sub.iterrows():
                if row['f_p'] < 0.05:
                    colors.append('#e74c3c' if row['avg_post_coef'] > 0 else '#2980b9')
                else:
                    colors.append('#95a5a6')

            y_pos = range(len(sub))
            ax.barh(y_pos, sub['avg_post_coef'].values * 100,
                    color=colors, alpha=0.8, edgecolor='grey', lw=0.3)
            ax.axvline(0, color='black', lw=0.8)

            labels = [f"T{int(r['topic_id'])} ({r['category']})"
                      for _, r in sub.iterrows()]
            ax.set_yticks(y_pos)
            ax.set_yticklabels(labels, fontsize=8)
            ax.set_xlabel('Avg Post-2023 Interaction Coef (×100)')
            ax.set_title('Topic-Level: AI Exposure × Post-2023 Interaction\n'
                         '(red/blue = p<0.05, grey = n.s.)')
            ax.invert_yaxis()
            plt.tight_layout()
            plt.savefig(OUT / 'topic_interactions_heatmap.png', dpi=150)
            plt.close()
            print('  Saved topic_interactions_heatmap.png')

    # ── Plot 3: Category shares over time by AI exposure level ──
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Task Category Shares Over Time by AI Exposure Level\n'
                 '(Weighted by cell size)', fontsize=13)

    # Split cells into high/low AI exposure
    median_ai = np.average(df_cells['ai_score'], weights=df_cells['n'])

    for idx, cat in enumerate(SUBSTANTIVE_CATS):
        ax = axes[idx // 2][idx % 2]
        dv = f'share_{cat}'

        for label, mask, color in [
            ('High AI', df_cells['ai_score'] >= median_ai, '#e74c3c'),
            ('Low AI', df_cells['ai_score'] < median_ai, '#3498db'),
        ]:
            sub = df_cells[mask]
            yearly = sub.groupby('year').apply(
                lambda g: np.average(g[dv], weights=g['n'])
            ).reset_index()
            yearly.columns = ['year', 'share']
            ax.plot(yearly['year'], yearly['share'] * 100, 'o-',
                    color=color, label=label, markersize=5)

        ax.set_title(f'{cat}: {CATEGORY_LABELS[cat]}')
        ax.set_ylabel('Share (%)')
        ax.set_xlabel('Year')
        ax.legend(fontsize=9)

    plt.tight_layout()
    plt.savefig(OUT / 'category_trends_by_ai.png', dpi=150)
    plt.close()
    print('  Saved category_trends_by_ai.png')


def run_individual_lpm(cat_results):
    """If category WLS shows interesting results, run individual-level LPM.
    Only for categories where Joint F p < 0.10."""
    interesting = [cat for cat in SUBSTANTIVE_CATS
                   if cat_results[cat]['f_p'] < 0.10]
    if not interesting:
        print('\n[5] No category has Joint F p < 0.10, skipping individual LPM')
        return

    print(f'\n[5] Running individual-level LPM for: {interesting}')
    print('    Loading full dataset ...')
    t0 = time.time()

    cols = ['year', 'isco08_4digit', 'ai_mean_score', 'dominant_topic_id']
    chunks = []
    for chunk in pd.read_csv(MASTER, usecols=cols, chunksize=CHUNK):
        chunk = chunk.dropna()
        chunk['year'] = chunk['year'].astype(int)
        chunk['isco08_4digit'] = chunk['isco08_4digit'].astype(int)
        chunk['topic'] = chunk['dominant_topic_id'].astype(int)
        chunk['category'] = chunk['topic'].map(TOPIC_TO_CATEGORY).fillna('M')
        chunks.append(chunk)

    df = pd.concat(chunks, ignore_index=True)
    print(f'    Loaded {len(df):,} rows in {time.time()-t0:.1f}s')

    years = sorted(df.year.unique())
    years_excl_ref = [y for y in years if y != REF_YEAR]

    for cat in interesting:
        print(f'\n  --- LPM for {cat} ({CATEGORY_LABELS[cat]}) ---')

        df[f'is_{cat}'] = (df['category'] == cat).astype(float)

        # Year dummies and interactions
        for y in years_excl_ref:
            df[f'yr_{y}'] = (df['year'] == y).astype(float)
            df[f'yr_{y}_x_ai'] = df[f'yr_{y}'] * df['ai_mean_score']

        yr_cols = [f'yr_{y}' for y in years_excl_ref]
        int_cols = [f'yr_{y}_x_ai' for y in years_excl_ref]
        all_cols = yr_cols + int_cols + [f'is_{cat}']

        # FWL demean by ISCO
        group_means = df.groupby('isco08_4digit')[all_cols].transform('mean')
        demeaned = df[all_cols] - group_means

        y_var = demeaned[f'is_{cat}'].values
        X = sm.add_constant(demeaned[yr_cols + int_cols].values)
        col_names = ['const'] + yr_cols + int_cols

        model = sm.OLS(y_var, X)
        res = model.fit(cov_type='cluster',
                        cov_kwds={'groups': df['isco08_4digit'].values})

        # Joint F-test on interactions
        n_int = len(int_cols)
        n_yr = len(yr_cols)
        R = np.zeros((n_int, len(col_names)))
        for k in range(n_int):
            R[k, 1 + n_yr + k] = 1
        f_test = res.f_test(R)
        f_val = float(np.asarray(f_test.fvalue).flat[0])
        f_p = float(np.asarray(f_test.pvalue).flat[0])

        lines = []
        lines.append(f'Individual LPM: P({cat}) ~ ISCO_FE + year_FE + year × ai_score')
        lines.append(f'N = {len(df):,}, ISCO clusters = {df.isco08_4digit.nunique()}')
        lines.append(f'R² (within) = {res.rsquared:.6f}')
        lines.append('')
        lines.append(f'Year × ai_score interactions:')
        lines.append(f'{"Year":<10} {"Coef":>12} {"SE":>12} {"p":>8}')
        lines.append('-' * 45)
        for col in sorted(int_cols):
            yr = col.split('_')[1]
            idx = col_names.index(col)
            lines.append(f'{yr:<10} {res.params[idx]:>12.6f} '
                         f'{res.bse[idx]:>12.6f} {res.pvalues[idx]:>8.4f}')
        lines.append(f'\nJoint F = {f_val:.4f}, p = {f_p:.4f}')

        result_text = '\n'.join(lines)
        print(result_text)

        with open(OUT / f'individual_lpm_{cat}.txt', 'w') as f:
            f.write(result_text)

        # Cleanup
        df.drop(columns=[f'is_{cat}'] + [f'yr_{y}' for y in years_excl_ref]
                + [f'yr_{y}_x_ai' for y in years_excl_ref],
                inplace=True)


# ── Main ────────────────────────────────────────────────────────────
if __name__ == '__main__':
    t_start = time.time()

    print('=' * 70)
    print('EXPERIMENT 2: Decomposed Outcome Variable')
    print('=' * 70)

    # Step 0: Save classification
    print('\n[0] Topic classification')
    save_topic_classification()

    # Step 1: Aggregate
    df_cells, all_topics = load_and_aggregate()

    # Step 2: Part A — Category regressions
    cat_results = run_category_regressions(df_cells)

    # Step 3: Part B — Topic-level regressions
    topic_df = run_topic_regressions(df_cells, all_topics)

    # Step 4: Visualization
    visualize_results(cat_results, topic_df, df_cells)

    # Step 5: Individual LPM (conditional)
    run_individual_lpm(cat_results)

    elapsed = time.time() - t_start
    print(f'\n{"=" * 70}')
    print(f'DONE in {elapsed:.1f}s')
    print(f'All results saved to {OUT}')
    for f in sorted(OUT.glob('*')):
        print(f'  {f.name}')
