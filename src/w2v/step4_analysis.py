"""
Step 4: 聚合分析与可视化
按 step4_plan.md 的 7 个分析一比一实现。
"""

import csv
import re
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from scipy import sparse
from scipy.spatial.distance import cosine as cosine_dist
from pathlib import Path
from collections import defaultdict

# ── 路径 ──
BASE = Path(__file__).resolve().parent.parent.parent
W2V_DIR = BASE / "output" / "w2v"
FIG_DIR = W2V_DIR / "figures"
FIG_DIR.mkdir(exist_ok=True)

METADATA_PATH = W2V_DIR / "jd_metadata.csv"
PROFILES_PATH = W2V_DIR / "jd_skill_profiles.npz"
COLUMNS_PATH  = W2V_DIR / "skill_cluster_columns.csv"

MERGED_CSV = BASE / "dataset" / "merged_1_6.csv"
COMPANY_LOOKUP = BASE / "data" / "Heterogeneity" / "company_industry_lookup.csv"
JD_INDUSTRY_PATH = W2V_DIR / "jd_industry.csv"

# ── 中文字体 ──
plt.rcParams['font.sans-serif'] = ['STHeiti', 'SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# ── 行业分类 (GB/T 4754 20大类) ──
INDUSTRY20 = {
    'A': 'A 农林牧渔', 'B': 'B 采矿', 'C': 'C 制造',
    'D': 'D 电力燃气', 'E': 'E 建筑', 'F': 'F 批发零售',
    'G': 'G 交通仓储', 'H': 'H 住宿餐饮', 'I': 'I 信息技术',
    'J': 'J 金融', 'K': 'K 房地产', 'L': 'L 商务服务',
    'M': 'M 科研技术', 'N': 'N 环境管理', 'O': 'O 居民服务',
    'P': 'P 教育', 'Q': 'Q 卫生社工', 'R': 'R 文体娱乐',
    'S': 'S 公共管理', 'T': 'T 国际组织', 'U': 'U 未识别',
}

KEYWORDS_INDUSTRY = {
    'T': ['国际组织', '联合国', '使馆', '领事馆'],
    'S': ['政府', '公共管理', '社会保障', '社会组织', '事业单位', '机关'],
    'A': ['农', '林', '牧', '渔', '养殖', '种植', '农副'],
    'B': ['采矿', '矿产', '煤炭', '矿业', '石油开采', '天然气开采'],
    'D': ['电力', '热力', '燃气', '自来水', '供水', '供电'],
    'K': ['房地产', '房产', '物业', '地产', '房屋中介'],
    'E': ['建筑', '施工', '土木', '装修', '建材', '工程施工'],
    'I': ['互联网', 'it', '信息技术', '软件', '通信', '网络', '云计算', '大数据', '人工智能', '计算机'],
    'J': ['银行', '保险', '证券', '期货', '基金', '投资', '融资', '信托', '金融'],
    'P': ['学校', '教育', '培训', '辅导', '学历教育'],
    'Q': ['医院', '卫生', '医疗服务', '护理', '康复', '社会工作', '养老服务'],
    'R': ['文化', '体育', '娱乐', '影视', '传媒', '出版', '新媒体', '广告'],
    'G': ['交通', '运输', '物流', '仓储', '邮政', '客运', '货运', '快递'],
    'H': ['餐饮', '酒店', '住宿', '民宿'],
    'F': ['批发', '零售', '商贸', '贸易', '电子商务'],
    'M': ['科研', '研究', '检测', '认证', '工程设计', '专业技术', '专利', '技术服务'],
    'N': ['水利', '环保', '环境', '公共设施', '园林', '环卫'],
    'L': ['租赁', '人力资源', '企业服务', '咨询', '法律', '翻译', '商务服务', '代理'],
    'O': ['居民服务', '维修', '修理', '家政', '美容', '美发', '保健', '洗浴'],
    'C': ['制造', '加工', '电子设备', '机械', '半导体', '汽车制造', '医药制造', '化工',
          '金属制品', '纺织', '服装', '家具', '印刷', '包装', '仪器仪表', '食品', '饮料', '工业'],
}


def keyword_industry(text):
    """用关键词匹配行业代码"""
    x = str(text or '').strip().lower()
    if not x:
        return 'U'
    for code, kws in KEYWORDS_INDUSTRY.items():
        if any(k in x for k in kws):
            return code
    if '服务' in x:
        return 'O'
    return 'U'


def code_from_label(label):
    s = str(label or '').strip()
    if s and s[0].isalpha():
        return s[0]
    return 'U'


# ============================================================
# 前置: 生成 jd_industry.csv
# ============================================================
def build_jd_industry():
    """通过 merged_1_6.csv 的企业名称查 industry lookup + 关键词兜底"""
    if JD_INDUSTRY_PATH.exists():
        print(f"  jd_industry.csv already exists, skipping build.")
        return

    print("  Loading company industry lookup...")
    company2code = {}
    with open(COMPANY_LOOKUP, 'r', encoding='utf-8-sig') as f:
        for row in csv.DictReader(f):
            name = row['企业名称'].strip()
            label = row.get('standard_industry', '')
            code = code_from_label(label)
            if code != 'U' and name:
                company2code[name] = code
    print(f"  {len(company2code):,} companies with industry labels")

    print(f"  Streaming merged_1_6.csv → jd_industry.csv ...")
    t0 = time.time()
    # jd_metadata 里的 id 集合 (只处理存在的 id)
    valid_ids = set()
    with open(METADATA_PATH, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            valid_ids.add(int(row['id']))
    print(f"  {len(valid_ids):,} valid JD ids loaded")

    with open(MERGED_CSV, 'r', encoding='utf-8-sig') as fin, \
         open(JD_INDUSTRY_PATH, 'w', encoding='utf-8', newline='') as fout:
        reader = csv.DictReader(fin)
        writer = csv.writer(fout)
        writer.writerow(['id', 'industry_code'])
        row_id = 0
        written = 0
        for row in reader:
            row_id += 1
            if row_id not in valid_ids:
                continue
            company = row.get('企业名称', '').strip()
            code = company2code.get(company)
            if code is None:
                # 关键词兜底: 用企业名 + 招聘岗位 + 招聘类别
                fallback_text = company + ' ' + row.get('招聘岗位', '') + ' ' + row.get('招聘类别', '')
                code = keyword_industry(fallback_text)
            writer.writerow([row_id, code])
            written += 1
            if row_id % 5_000_000 == 0:
                print(f"    row {row_id:>12,} | written {written:,} | {time.time()-t0:.0f}s")

    print(f"  Done: {written:,} rows, {time.time()-t0:.0f}s")


# ============================================================
# 加载数据
# ============================================================
def load_data():
    print("\n[Load] Reading metadata...")
    dtype = {
        'id': 'int32', 'year': 'str', 'is_fresh_grad': 'int8',
        'data_source': 'str', 'skill_token_count': 'int32',
        'n_skill_clusters': 'int8', 'top_cluster': 'str', 'ai_binary': 'int8',
    }
    meta = pd.read_csv(METADATA_PATH, dtype=dtype,
                       usecols=['id', 'year', 'is_fresh_grad', 'data_source',
                                'skill_token_count', 'n_skill_clusters',
                                'ai_intensity', 'ai_binary'])
    meta['year'] = meta['year'].astype(int)
    # 过滤有效年份 (2016-2025)
    meta = meta[(meta['year'] >= 2016) & (meta['year'] <= 2025)].copy()
    print(f"  {len(meta):,} rows after year filter")

    print("[Load] Reading skill cluster columns...")
    cols_df = pd.read_csv(COLUMNS_PATH)
    col_names = cols_df['cluster_label'].tolist()
    col_cn = cols_df['name_cn'].tolist()

    print("[Load] Reading sparse matrix...")
    mat = sparse.load_npz(PROFILES_PATH)
    print(f"  Shape: {mat.shape}")

    # 加载行业
    industry = None
    if JD_INDUSTRY_PATH.exists():
        print("[Load] Reading industry mapping...")
        ind = pd.read_csv(JD_INDUSTRY_PATH, dtype={'id': 'int32', 'industry_code': 'str'})
        meta = meta.merge(ind, on='id', how='left')
        meta['industry_code'] = meta['industry_code'].fillna('U')
        print(f"  Industry coverage: {(meta['industry_code'] != 'U').sum():,} / {len(meta):,} "
              f"({(meta['industry_code'] != 'U').mean()*100:.1f}%)")

    return meta, mat, col_names, col_cn


# ── 数据源列表 ──
SOURCES = ['上市公司', '应届生', '综合']
SRC_COLORS = {'上市公司': '#2196F3', '应届生': '#4CAF50', '综合': '#FF9800'}


def _src_subplots(n_src=3, figw=6, figh=5):
    """创建 1 行 n_src 列子图"""
    fig, axes = plt.subplots(1, n_src, figsize=(figw * n_src, figh), sharey=True)
    if n_src == 1:
        axes = [axes]
    return fig, axes


# ============================================================
# 分析 1: AI 渗透率年度趋势 (广度+深度) — 分数据源
# ============================================================
def analysis_1(meta):
    print("\n[Analysis 1] AI penetration trend (by data_source)...")
    all_rows = []
    fig, axes = _src_subplots(len(SOURCES), figw=7, figh=5)

    for ax, src in zip(axes, SOURCES):
        sub = meta[meta['data_source'] == src]
        yearly = sub.groupby('year').agg(
            n=('ai_binary', 'size'),
            ai_breadth=('ai_binary', 'mean'),
        ).reset_index()
        ai_only = sub[sub['ai_binary'] == 1].groupby('year')['ai_intensity'].mean().reset_index()
        ai_only.columns = ['year', 'ai_depth']
        yearly = yearly.merge(ai_only, on='year', how='left')
        yearly['data_source'] = src
        all_rows.append(yearly)

        ax2 = ax.twinx()
        ax.plot(yearly['year'], yearly['ai_breadth'] * 100, 'o-', color='#2196F3',
                linewidth=2, markersize=5, label='广度')
        ax2.plot(yearly['year'], yearly['ai_depth'] * 100, 's--', color='#FF5722',
                 linewidth=2, markersize=5, label='深度')
        ax.set_title(f'{src} (n={len(sub):,})', fontsize=12)
        ax.set_xlabel('年份')
        if ax == axes[0]:
            ax.set_ylabel('渗透广度 (%)', color='#2196F3')
        if ax == axes[-1]:
            ax2.set_ylabel('渗透深度 (%)', color='#FF5722')
        ax.tick_params(axis='x', rotation=45)
        ax.grid(axis='y', alpha=0.3)
        # 只在年份有数据时显示
        valid_years = yearly[yearly['n'] >= 100]['year']
        ax.set_xlim(valid_years.min() - 0.5, valid_years.max() + 0.5)
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc='upper left')

    fig.suptitle('AI渗透率年度趋势 — 分数据源', fontsize=14)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig1_ai_penetration_trend.pdf", dpi=150)
    plt.close(fig)

    summary = pd.concat(all_rows, ignore_index=True)
    summary.to_csv(W2V_DIR / "yearly_summary.csv", index=False)
    print(summary.to_string(index=False))
    print("  Saved fig1")
    return summary


# ============================================================
# 分析 2: AI岗 vs 非AI岗 技能差异 — 分数据源
# ============================================================
def analysis_2(meta, mat, col_names, col_cn):
    print("\n[Analysis 2] AI vs non-AI skill profile diff (by data_source)...")
    all_diffs = []
    fig, axes = _src_subplots(len(SOURCES), figw=6, figh=10)

    for ax, src in zip(axes, SOURCES):
        sub = meta[meta['data_source'] == src]
        ai_idx = sub.index[sub['ai_binary'] == 1].values
        non_ai_idx = sub.index[sub['ai_binary'] == 0].values

        if len(ai_idx) < 100:
            ax.set_title(f'{src} (insufficient AI JDs)')
            continue

        ai_mean = np.asarray(mat[ai_idx].mean(axis=0)).flatten()
        non_ai_mean = np.asarray(mat[non_ai_idx].mean(axis=0)).flatten()
        diff = ai_mean - non_ai_mean

        diff_df = pd.DataFrame({
            'cluster': col_names, 'name_cn': col_cn,
            'ai_mean': ai_mean, 'nonai_mean': non_ai_mean, 'diff': diff,
            'data_source': src,
        }).sort_values('diff', ascending=False)
        all_diffs.append(diff_df)

        # Top10 + Bottom10
        plot_df = pd.concat([diff_df.head(10), diff_df.tail(10)])
        labels = [f"{r['cluster']} {r['name_cn']}" for _, r in plot_df.iterrows()]
        colors = ['#2196F3' if d > 0 else '#FF5722' for d in plot_df['diff']]
        ax.barh(range(len(plot_df)), plot_df['diff'].values, color=colors)
        ax.set_yticks(range(len(plot_df)))
        ax.set_yticklabels(labels, fontsize=7)
        ax.axvline(0, color='black', linewidth=0.5)
        ax.set_title(f'{src}', fontsize=11)
        ax.invert_yaxis()

    axes[0].set_ylabel('')
    fig.suptitle('AI岗 vs 非AI岗 技能差异 — 分数据源', fontsize=13)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig2_skill_diff_ai_vs_nonai.pdf", dpi=150)
    plt.close(fig)

    if all_diffs:
        pd.concat(all_diffs).to_csv(W2V_DIR / "ai_vs_nonai_skill_diff.csv", index=False)
    print("  Saved fig2")


# ============================================================
# 分析 3: 技能广度年度变化 — 分数据源
# ============================================================
def analysis_3(meta):
    print("\n[Analysis 3] Skill breadth trend (by data_source)...")
    fig, axes = _src_subplots(len(SOURCES), figw=7, figh=5)

    for ax, src in zip(axes, SOURCES):
        sub = meta[meta['data_source'] == src]
        groups = {
            '全量': sub,
            'AI岗': sub[sub['ai_binary'] == 1],
            '非AI岗': sub[sub['ai_binary'] == 0],
        }
        styles = ['o-', 's--', '^:']
        for (name, df), style in zip(groups.items(), styles):
            yearly = df.groupby('year')['n_skill_clusters'].mean()
            ax.plot(yearly.index, yearly.values, style, label=name, linewidth=1.5, markersize=4)
        ax.set_title(f'{src}', fontsize=11)
        ax.set_xlabel('年份')
        ax.legend(fontsize=9)
        ax.grid(axis='y', alpha=0.3)
        ax.tick_params(axis='x', rotation=45)

    axes[0].set_ylabel('平均技能域数量')
    fig.suptitle('技能广度年度变化 — 分数据源', fontsize=13)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig3_skill_breadth_trend.pdf", dpi=150)
    plt.close(fig)
    print("  Saved fig3")


# ============================================================
# 分析 4: 应届 vs 社招 AI 渗透 — 分数据源
# ============================================================
def analysis_4(meta):
    print("\n[Analysis 4] Fresh grad vs experienced (by data_source)...")
    fig, axes = _src_subplots(len(SOURCES), figw=7, figh=5)

    for ax, src in zip(axes, SOURCES):
        sub = meta[meta['data_source'] == src]
        pivot = sub.groupby(['year', 'is_fresh_grad'])['ai_binary'].mean().unstack()
        if 0 in pivot.columns:
            ax.plot(pivot.index, pivot[0] * 100, 's--', label='社招', linewidth=2, markersize=5)
        if 1 in pivot.columns:
            ax.plot(pivot.index, pivot[1] * 100, 'o-', label='应届生', linewidth=2, markersize=5)
        ax.set_title(f'{src}', fontsize=11)
        ax.set_xlabel('年份')
        ax.legend(fontsize=10)
        ax.grid(axis='y', alpha=0.3)
        ax.tick_params(axis='x', rotation=45)

    axes[0].set_ylabel('AI渗透率 (%)')
    fig.suptitle('应届生 vs 社招 AI渗透率 — 分数据源', fontsize=13)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig4_freshgrad_vs_experienced.pdf", dpi=150)
    plt.close(fig)
    print("  Saved fig4")


# ============================================================
# 分析 5: 87 个技能域年度消长 — 分数据源
# ============================================================
def analysis_5(meta, mat, col_names, col_cn):
    print("\n[Analysis 5] Skill domain yearly changes (by data_source)...")
    all_change_dfs = []

    for src in SOURCES:
        sub = meta[meta['data_source'] == src]
        years = sorted(sub['year'].unique())
        if len(years) < 2:
            continue
        n_cols = len(col_names)

        yearly_means = np.zeros((len(years), n_cols))
        for i, yr in enumerate(years):
            idx = sub.index[sub['year'] == yr].values
            yearly_means[i] = np.asarray(mat[idx].mean(axis=0)).flatten()

        # 保存每个源的 yearly CSV
        yearly_df = pd.DataFrame(yearly_means, index=years, columns=col_names)
        yearly_df.index.name = 'year'
        yearly_df['data_source'] = src
        yearly_df.to_csv(W2V_DIR / f"skill_domain_yearly_{src}.csv")

        # 变化率
        first_yr, last_yr = years[0], years[-1]
        v_first = yearly_means[0]
        v_last = yearly_means[-1]
        change = np.where(v_first > 1e-6, (v_last - v_first) / v_first * 100, 0)

        change_df = pd.DataFrame({
            'cluster': col_names, 'name_cn': col_cn,
            f'mean_{first_yr}': v_first, f'mean_{last_yr}': v_last,
            'change_pct': change, 'data_source': src,
        }).sort_values('change_pct', ascending=False)
        all_change_dfs.append(change_df)

        print(f"\n  [{src}] Top 5 增长 ({first_yr}→{last_yr}):")
        print(change_df.head(5)[['cluster', 'name_cn', 'change_pct']].to_string(index=False))
        print(f"  [{src}] Top 5 萎缩:")
        print(change_df.tail(5)[['cluster', 'name_cn', 'change_pct']].to_string(index=False))

    # 热力图: 每个数据源一个子图
    fig, axes = _src_subplots(len(SOURCES), figw=5, figh=22)
    for ax, src in zip(axes, SOURCES):
        sub = meta[meta['data_source'] == src]
        years = sorted(sub['year'].unique())
        n_cols = len(col_names)
        yearly_means = np.zeros((len(years), n_cols))
        for i, yr in enumerate(years):
            idx = sub.index[sub['year'] == yr].values
            yearly_means[i] = np.asarray(mat[idx].mean(axis=0)).flatten()

        v_first = yearly_means[0]
        v_last = yearly_means[-1]
        change = np.where(v_first > 1e-6, (v_last - v_first) / v_first * 100, 0)
        sorted_idx = np.argsort(change)[::-1]
        sorted_data = yearly_means[:, sorted_idx].T

        im = ax.imshow(sorted_data, aspect='auto', cmap='YlOrRd', interpolation='nearest')
        ax.set_xticks(range(len(years)))
        ax.set_xticklabels(years, fontsize=7, rotation=45)
        if ax == axes[0]:
            sorted_labels = [f"{col_names[j]} {col_cn[j]}" for j in sorted_idx]
            ax.set_yticks(range(n_cols))
            ax.set_yticklabels(sorted_labels, fontsize=6)
        else:
            ax.set_yticks([])
        ax.set_title(src, fontsize=10)

    fig.suptitle('87技能域年度占比热力图 — 分数据源', fontsize=13)
    fig.colorbar(im, ax=axes, shrink=0.3, label='平均占比')
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig5_skill_domain_changes.pdf", dpi=150)
    plt.close(fig)
    print("  Saved fig5")

    if all_change_dfs:
        pd.concat(all_change_dfs).to_csv(W2V_DIR / "skill_domain_changes_by_source.csv", index=False)


# ============================================================
# 分析 6: (已合并到各分析中, 保留原始汇总图)
# ============================================================
def analysis_6(meta):
    print("\n[Analysis 6] Data source robustness (summary overlay)...")
    pivot = meta.groupby(['year', 'data_source'])['ai_binary'].mean().unstack()

    fig, ax = plt.subplots(figsize=(10, 6))
    for col in SOURCES:
        if col in pivot.columns:
            vals = pivot[col].dropna()
            ax.plot(vals.index, vals.values * 100, 'o-', label=col,
                    color=SRC_COLORS[col], linewidth=2, markersize=5)
    ax.set_xlabel('年份', fontsize=13)
    ax.set_ylabel('AI渗透率 (%)', fontsize=13)
    ax.set_title('三个数据源的AI渗透率对比', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig6_datasource_robustness.pdf", dpi=150)
    plt.close(fig)
    print("  Saved fig6")


# ============================================================
# 分析 7: 分行业技能结构偏移 — 分数据源
# ============================================================
def analysis_7(meta, mat, col_names, col_cn):
    if 'industry_code' not in meta.columns:
        print("\n[Analysis 7] SKIPPED - no industry data")
        return

    print("\n[Analysis 7] Industry skill structure shift (by data_source)...")

    for src in SOURCES:
        sub_src = meta[meta['data_source'] == src]
        years = sorted(sub_src['year'].unique())
        if len(years) < 2:
            print(f"  [{src}] skipped (< 2 years)")
            continue
        first_yr, last_yr = years[0], years[-1]

        ind_counts = sub_src['industry_code'].value_counts()
        valid_codes = ind_counts[ind_counts >= 1000].index.tolist()
        if 'U' in valid_codes:
            valid_codes.remove('U')
        valid_codes.sort()
        print(f"  [{src}] {len(valid_codes)} industries with >= 1K JDs ({first_yr}-{last_yr})")

        if not valid_codes:
            continue

        # (a) 行业 AI 渗透率趋势
        fig, ax = plt.subplots(figsize=(12, 7))
        colors = plt.cm.tab20(np.linspace(0, 1, max(len(valid_codes), 1)))
        for code, color in zip(valid_codes, colors):
            sub = sub_src[sub_src['industry_code'] == code]
            yearly = sub.groupby('year')['ai_binary'].mean() * 100
            ax.plot(yearly.index, yearly.values, 'o-', label=INDUSTRY20.get(code, code),
                    color=color, linewidth=1.5, markersize=4)
        ax.set_xlabel('年份', fontsize=13)
        ax.set_ylabel('AI渗透率 (%)', fontsize=13)
        ax.set_title(f'分行业AI渗透率 — {src}', fontsize=14)
        ax.legend(fontsize=7, ncol=2, loc='upper left')
        ax.grid(axis='y', alpha=0.3)
        fig.tight_layout()
        safe_name = src.replace('/', '_')
        fig.savefig(FIG_DIR / f"fig7_industry_ai_{safe_name}.pdf", dpi=150)
        plt.close(fig)

        # (b) 余弦偏移表
        shift_rows = []
        for code in valid_codes:
            sub = sub_src[sub_src['industry_code'] == code]
            idx_first = sub.index[sub['year'] == first_yr].values
            idx_last = sub.index[sub['year'] == last_yr].values
            if len(idx_first) < 50 or len(idx_last) < 50:
                continue
            v_first = np.asarray(mat[idx_first].mean(axis=0)).flatten()
            v_last = np.asarray(mat[idx_last].mean(axis=0)).flatten()
            cos_sim = 1 - cosine_dist(v_first, v_last)
            ai_first = sub[sub['year'] == first_yr]['ai_binary'].mean()
            ai_last = sub[sub['year'] == last_yr]['ai_binary'].mean()
            shift_rows.append({
                'data_source': src, 'industry_code': code,
                'industry_name': INDUSTRY20.get(code, code),
                'n_jds': len(sub),
                f'cos_sim_{first_yr}_{last_yr}': round(cos_sim, 4),
                f'ai_{first_yr}': round(ai_first, 4),
                f'ai_{last_yr}': round(ai_last, 4),
                'ai_change_pp': round((ai_last - ai_first) * 100, 2),
            })
        if shift_rows:
            shift_df = pd.DataFrame(shift_rows)
            shift_df.to_csv(W2V_DIR / f"industry_skill_shift_{safe_name}.csv", index=False)
            print(shift_df.to_string(index=False))

    print("  Saved fig7 (per source)")

    # (c) 选 Top 4 行业, 分数据源对比技能画像
    top_industries = ['I', 'C', 'J', 'P']
    fig, axes = plt.subplots(len(SOURCES), len(top_industries),
                             figsize=(5 * len(top_industries), 5 * len(SOURCES)),
                             sharey='row')
    for row, src in enumerate(SOURCES):
        sub_src = meta[meta['data_source'] == src]
        years = sorted(sub_src['year'].unique())
        if len(years) < 2:
            continue
        first_yr, last_yr = years[0], years[-1]

        for col_i, code in enumerate(top_industries):
            ax = axes[row][col_i]
            sub = sub_src[sub_src['industry_code'] == code]
            idx_f = sub.index[sub['year'] == first_yr].values
            idx_l = sub.index[sub['year'] == last_yr].values
            if len(idx_f) < 50 or len(idx_l) < 50:
                ax.set_title(f'{INDUSTRY20.get(code,code)}\n(insufficient data)')
                continue
            v_f = np.asarray(mat[idx_f].mean(axis=0)).flatten()
            v_l = np.asarray(mat[idx_l].mean(axis=0)).flatten()
            top_k = 12
            max_val = np.maximum(v_f, v_l)
            top_idx = np.argsort(max_val)[-top_k:][::-1]
            y_pos = range(top_k)
            ax.barh([p - 0.15 for p in y_pos], v_f[top_idx], height=0.3,
                    label=str(first_yr), color='#90CAF9')
            ax.barh([p + 0.15 for p in y_pos], v_l[top_idx], height=0.3,
                    label=str(last_yr), color='#FF8A65')
            labels = [col_names[j] for j in top_idx]
            ax.set_yticks(list(y_pos))
            ax.set_yticklabels(labels, fontsize=7)
            title = INDUSTRY20.get(code, code) if row == 0 else ''
            ax.set_title(title, fontsize=10)
            if col_i == 0:
                ax.set_ylabel(f'{src}\n({first_yr}-{last_yr})', fontsize=10)
            ax.legend(fontsize=7)
            ax.invert_yaxis()

    fig.suptitle('代表行业技能画像变化 — 分数据源', fontsize=14)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig7b_industry_skill_shift.pdf", dpi=150)
    plt.close(fig)
    print("  Saved fig7b")


# ============================================================
# Main
# ============================================================
def main():
    print("=" * 60)
    print("Step 4: Aggregate Analysis & Visualization")
    print("=" * 60)

    # 前置: 行业映射
    print("\n[0/7] Building industry mapping...")
    build_jd_industry()

    # 加载数据
    meta, mat, col_names, col_cn = load_data()

    # 对齐 meta 和 mat: meta 经过年份过滤后 index 可能不连续
    # 重置 index 前记录原始位置用于 mat 索引
    # meta 读入时 index = 0..26.8M-1, 对应 mat 的行号
    # 年份过滤后 index 不连续但仍指向 mat 的正确行
    # pandas 切片 mat[meta.index] 即可

    analysis_1(meta)
    analysis_2(meta, mat, col_names, col_cn)
    analysis_3(meta)
    analysis_4(meta)
    analysis_5(meta, mat, col_names, col_cn)
    analysis_6(meta)
    analysis_7(meta, mat, col_names, col_cn)

    print("\n" + "=" * 60)
    print("All analyses complete!")
    print(f"Figures saved to: {FIG_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
