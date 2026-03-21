"""
Step 4 Deep-dive: 岗位综合化趋势分析
核心问题：AI 是否让岗位职责变得更综合化？

AI 暴露度定义：
  不用 JD 级别的 ai_binary（太粗糙，混淆了"做AI"和"被AI影响"），
  而用 **行业-年份层面的 AI 渗透率** 作为连续型暴露变量。
  一个行业的 AI 渗透率越高 → 该行业受 AI 影响越大 → 所有岗位都可能被重塑。

子分析：
  3A  n_skill_clusters 年度趋势（整体描述性基线）
  3B  Shannon 熵年度趋势（技能分布均匀度 — 另一个综合化度量）
  3C  n_skill_clusters 分布密度图（首尾年份对比，看分布偏移而非仅均值）
  3D  行业-年份面板：AI暴露度 × 综合化（within-industry 分析）
  3E  行业综合化热力图（行业 × 年份）
  3F  重点行业小多图：AI渗透率与综合化的共变趋势
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from scipy import sparse
from pathlib import Path

# ── 路径 ──
BASE = Path(__file__).resolve().parent.parent.parent
W2V_DIR = BASE / "output" / "w2v"
FIG_DIR = W2V_DIR / "figures"
FIG_DIR.mkdir(exist_ok=True)

METADATA_PATH = W2V_DIR / "jd_metadata.csv"
PROFILES_PATH = W2V_DIR / "jd_skill_profiles.npz"
COLUMNS_PATH  = W2V_DIR / "skill_cluster_columns.csv"
JD_INDUSTRY_PATH = W2V_DIR / "jd_industry.csv"

# ── 中文字体 ──
plt.rcParams['font.sans-serif'] = ['STHeiti', 'SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# ── 配色 ──
C = {
    'dark':    '#00102c',
    'red':     '#002566',
    'navy':    '#003591',
    'orange':  '#19499c',
    'blue':    '#335da7',
    'green':   '#809ac8',
    'gray':    '#99aed3',
    'lred':    '#ccd7e9',
    'lblue':   '#e6ebf4',
}

PALETTE = [C['dark'], C['navy'], C['blue'], C['green'], C['gray']]

INDUSTRY20 = {
    'A': 'A 农林牧渔', 'B': 'B 采矿', 'C': 'C 制造',
    'D': 'D 电力燃气', 'E': 'E 建筑', 'F': 'F 批发零售',
    'G': 'G 交通仓储', 'H': 'H 住宿餐饮', 'I': 'I 信息技术',
    'J': 'J 金融', 'K': 'K 房地产', 'L': 'L 商务服务',
    'M': 'M 科研技术', 'N': 'N 环境管理', 'O': 'O 居民服务',
    'P': 'P 教育', 'Q': 'Q 卫生社工', 'R': 'R 文体娱乐',
    'S': 'S 公共管理',
}

SOURCES = ['上市公司', '应届生']

# ── 全局样式 ──
def _style_ax(ax, xlabel='', ylabel='', title=''):
    ax.set_xlabel(xlabel, fontsize=11, color=C['dark'])
    ax.set_ylabel(ylabel, fontsize=11, color=C['dark'])
    if title:
        ax.set_title(title, fontsize=12, fontweight='bold', color=C['dark'])
    ax.tick_params(colors=C['dark'], labelsize=9)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color(C['gray'])
    ax.spines['bottom'].set_color(C['gray'])
    ax.grid(axis='y', alpha=0.2, color=C['gray'])


# ──────────────────────────────────────────────
# 数据加载
# ──────────────────────────────────────────────
def load_data():
    print("[Load] Reading metadata...")
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
    meta = meta[(meta['year'] >= 2016) & (meta['year'] <= 2025)].copy()
    print(f"  {len(meta):,} rows after year filter")

    print("[Load] Reading skill cluster columns...")
    cols_df = pd.read_csv(COLUMNS_PATH)
    col_names = cols_df['cluster_label'].tolist()
    col_cn = cols_df['name_cn'].tolist()

    print("[Load] Reading sparse matrix...")
    mat = sparse.load_npz(PROFILES_PATH)
    print(f"  Shape: {mat.shape}")

    if JD_INDUSTRY_PATH.exists():
        print("[Load] Reading industry mapping...")
        ind = pd.read_csv(JD_INDUSTRY_PATH, dtype={'id': 'int32', 'industry_code': 'str'})
        meta = meta.merge(ind, on='id', how='left')
        has_ind = meta['industry_code'].notna().sum()
        print(f"  Industry coverage: {has_ind:,} / {len(meta):,} ({has_ind/len(meta)*100:.1f}%)")

    return meta, mat, col_names, col_cn


def compute_entropy_batch(mat, indices, batch_size=500_000):
    """分批计算 Shannon 熵"""
    results = np.zeros(len(indices))
    for start in range(0, len(indices), batch_size):
        end = min(start + batch_size, len(indices))
        batch_idx = indices[start:end]
        dense = np.asarray(mat[batch_idx].todense())
        row_sums = dense.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums > 0, row_sums, 1)
        probs = dense / row_sums
        for i in range(len(batch_idx)):
            p = probs[i]
            p = p[p > 0]
            results[start + i] = -np.sum(p * np.log2(p))
    return results


def build_industry_year_panel(meta, src):
    """构建行业-年份面板数据"""
    sub = meta[(meta['data_source'] == src) &
               (meta['industry_code'].notna()) &
               (meta['industry_code'] != 'U')]
    years = sorted(sub['year'].unique())

    rows = []
    for code in sorted(sub['industry_code'].unique()):
        ind_data = sub[sub['industry_code'] == code]
        for yr in years:
            yr_data = ind_data[ind_data['year'] == yr]
            if len(yr_data) < 100:
                continue
            rows.append({
                'industry_code': code,
                'industry_name': INDUSTRY20.get(code, code),
                'year': yr,
                'n_jds': len(yr_data),
                'ai_exposure': yr_data['ai_binary'].mean(),
                'mean_ai_intensity': yr_data['ai_intensity'].mean(),
                'mean_breadth': yr_data['n_skill_clusters'].mean(),
                'median_breadth': yr_data['n_skill_clusters'].median(),
            })
            if 'entropy' in yr_data.columns:
                rows[-1]['mean_entropy'] = yr_data['entropy'].mean()

    panel = pd.DataFrame(rows)
    return panel


# ──────────────────────────────────────────────
# 3A: 综合化基线趋势
# ──────────────────────────────────────────────
def analysis_3a(meta):
    print("\n[3A] Overall comprehensiveness trend...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), sharey=True)

    for ax, src in zip(axes, SOURCES):
        sub = meta[meta['data_source'] == src]

        # 全量趋势
        yearly = sub.groupby('year')['n_skill_clusters'].mean()
        ax.plot(yearly.index, yearly.values, 'o-',
                color=C['navy'], linewidth=2.5, markersize=6, label='全量均值', zorder=3)

        # 分位数带
        q25 = sub.groupby('year')['n_skill_clusters'].quantile(0.25)
        q75 = sub.groupby('year')['n_skill_clusters'].quantile(0.75)
        ax.fill_between(q25.index, q25.values, q75.values,
                        color=C['lblue'], alpha=0.6, label='25-75 分位')

        _style_ax(ax, xlabel='年份', title=src)
        ax.legend(fontsize=10, framealpha=0.9)
        ax.tick_params(axis='x', rotation=45)

    axes[0].set_ylabel('技能域数量', fontsize=12)
    fig.suptitle('岗位综合化趋势', fontsize=14,
                 fontweight='bold', color=C['dark'], y=1.02)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig3a_breadth_trend.pdf", dpi=200, bbox_inches='tight')
    plt.close(fig)
    print("  Saved fig3a")


# ──────────────────────────────────────────────
# 3B: Shannon 熵趋势
# ──────────────────────────────────────────────
def analysis_3b(meta, mat):
    print("\n[3B] Shannon entropy trend...")

    print("  Computing entropy for all rows...")
    all_idx = meta.index.values
    ent = compute_entropy_batch(mat, all_idx)
    meta = meta.copy()
    meta['entropy'] = ent
    print(f"  Mean entropy: {ent.mean():.3f}, max possible: {np.log2(87):.3f}")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), sharey=True)

    for ax, src in zip(axes, SOURCES):
        sub = meta[meta['data_source'] == src]
        yearly = sub.groupby('year')['entropy'].mean()
        ax.plot(yearly.index, yearly.values, 'o-',
                color=C['navy'], linewidth=2.5, markersize=6)

        q25 = sub.groupby('year')['entropy'].quantile(0.25)
        q75 = sub.groupby('year')['entropy'].quantile(0.75)
        ax.fill_between(q25.index, q25.values, q75.values,
                        color=C['lblue'], alpha=0.6, label='25-75 分位')

        _style_ax(ax, xlabel='年份', title=src)
        ax.legend(fontsize=10, framealpha=0.9)
        ax.tick_params(axis='x', rotation=45)

    axes[0].set_ylabel('Shannon 熵 (bits)', fontsize=12)
    fig.suptitle('技能分布均匀度变化', fontsize=14,
                 fontweight='bold', color=C['dark'], y=1.02)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig3b_entropy_trend.pdf", dpi=200, bbox_inches='tight')
    plt.close(fig)
    print("  Saved fig3b")

    return meta


# ──────────────────────────────────────────────
# 3C: 分布偏移
# ──────────────────────────────────────────────
def analysis_3c(meta):
    print("\n[3C] Distribution shift (first vs last year)...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), sharey=True)

    for ax, src in zip(axes, SOURCES):
        sub = meta[meta['data_source'] == src]
        years = sorted(sub['year'].unique())
        yr_first, yr_last = years[0], years[-1]

        for yr, color, alpha in [(yr_first, C['green'], 0.6), (yr_last, C['navy'], 0.7)]:
            data = sub[sub['year'] == yr]['n_skill_clusters']
            bins = np.arange(0, data.max() + 2) - 0.5
            counts, edges = np.histogram(data, bins=bins, density=True)
            centers = (edges[:-1] + edges[1:]) / 2
            ax.bar(centers, counts * 100, width=0.8, alpha=alpha, color=color,
                   label=f'{yr} (M={data.mean():.1f})', edgecolor='white', linewidth=0.3)

        _style_ax(ax, xlabel='技能域数量', title=src)
        ax.legend(fontsize=11, framealpha=0.9)
        ax.set_xlim(-0.5, 45)

    axes[0].set_ylabel('频率 (%)', fontsize=12)
    fig.suptitle('岗位技能域数量分布偏移', fontsize=14,
                 fontweight='bold', color=C['dark'], y=1.02)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig3c_breadth_distribution.pdf", dpi=200, bbox_inches='tight')
    plt.close(fig)
    print("  Saved fig3c")


# ──────────────────────────────────────────────
# 3D: 行业面板分析 — AI暴露度 × 综合化
#     行业层面 AI exposure = 该行业该年所有 JD 的 ai_binary 均值
#     这不是"这个岗位是不是AI岗"，而是"这个行业受AI影响多深"
# ──────────────────────────────────────────────
def analysis_3d(meta):
    if 'industry_code' not in meta.columns:
        print("\n[3D] SKIPPED - no industry data")
        return
    print("\n[3D] Industry panel: AI exposure × comprehensiveness...")

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    for ax_i, (ax, src) in enumerate(zip(axes, SOURCES)):
        panel = build_industry_year_panel(meta, src)
        if panel.empty:
            continue

        # --- 一阶差分（within-industry change）---
        # 对每个行业，计算首尾年份的变化
        changes = []
        for code in panel['industry_code'].unique():
            ind_p = panel[panel['industry_code'] == code].sort_values('year')
            if len(ind_p) < 2:
                continue
            first, last = ind_p.iloc[0], ind_p.iloc[-1]
            changes.append({
                'code': code,
                'name': INDUSTRY20.get(code, code),
                'delta_ai': (last['ai_exposure'] - first['ai_exposure']) * 100,  # pp
                'delta_breadth': last['mean_breadth'] - first['mean_breadth'],
                'n_total': ind_p['n_jds'].sum(),
                'yr_range': f"{int(first['year'])}-{int(last['year'])}",
            })

        if not changes:
            continue

        ch = pd.DataFrame(changes)

        # 气泡大小
        sizes = np.clip(ch['n_total'] / ch['n_total'].max() * 400, 40, 400)
        ax.scatter(ch['delta_ai'], ch['delta_breadth'], s=sizes,
                   c=C['navy'], alpha=0.55, edgecolors=C['dark'], linewidths=0.5)

        # 标注行业代码
        for _, row in ch.iterrows():
            ax.annotate(row['code'], (row['delta_ai'], row['delta_breadth']),
                        fontsize=9, fontweight='bold', ha='center', va='bottom',
                        xytext=(0, 6), textcoords='offset points', color=C['dark'])

        # 趋势线 + 相关系数
        if len(ch) >= 5:
            z = np.polyfit(ch['delta_ai'], ch['delta_breadth'], 1)
            x_line = np.linspace(ch['delta_ai'].min() - 0.5, ch['delta_ai'].max() + 0.5, 50)
            ax.plot(x_line, np.polyval(z, x_line), '--', color=C['green'],
                    linewidth=2, alpha=0.8)
            corr = ch[['delta_ai', 'delta_breadth']].corr().iloc[0, 1]
            slope = z[0]
            ax.text(0.05, 0.95,
                    f'r = {corr:.3f}\nslope = {slope:.2f}',
                    transform=ax.transAxes, fontsize=10, va='top', color=C['dark'],
                    bbox=dict(boxstyle='round,pad=0.4', facecolor=C['lblue'], alpha=0.9))

        ax.axhline(0, color=C['gray'], linewidth=0.8, linestyle=':')
        ax.axvline(0, color=C['gray'], linewidth=0.8, linestyle=':')

        yr_range = ch['yr_range'].iloc[0] if len(ch) > 0 else ''
        _style_ax(ax, xlabel='ΔAI渗透率 (pp)',
                  ylabel='Δ平均技能域数量' if ax_i == 0 else '',
                  title=f'{src} ({yr_range})')

        # 打印结果
        print(f"\n  [{src}] Industry first-difference (corr = {corr:.3f}):")
        ch_sorted = ch.sort_values('delta_ai', ascending=False)
        for _, r in ch_sorted.iterrows():
            print(f"    {r['name']:16s}  ΔAI={r['delta_ai']:+5.2f}pp  Δbreadth={r['delta_breadth']:+5.2f}")

    fig.suptitle('行业面板：AI渗透率变化 × 综合化变化（一阶差分）',
                 fontsize=14, fontweight='bold', color=C['dark'], y=1.02)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig3d_panel_first_diff.pdf", dpi=200, bbox_inches='tight')
    plt.close(fig)
    print("  Saved fig3d")


# ──────────────────────────────────────────────
# 3E: 行业综合化热力图（行业 × 年份）
# ──────────────────────────────────────────────
def analysis_3e(meta):
    if 'industry_code' not in meta.columns:
        print("\n[3E] SKIPPED - no industry data")
        return
    print("\n[3E] Industry × year comprehensiveness heatmap...")

    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    for ax, src in zip(axes, SOURCES):
        panel = build_industry_year_panel(meta, src)
        if panel.empty:
            continue

        # 只保留跨度 >= 3 年的行业
        valid_codes = []
        for code in sorted(panel['industry_code'].unique()):
            if len(panel[panel['industry_code'] == code]) >= 3:
                valid_codes.append(code)

        if not valid_codes:
            continue

        # 构建矩阵
        years = sorted(panel['year'].unique())
        mat_data = np.full((len(valid_codes), len(years)), np.nan)
        for i, code in enumerate(valid_codes):
            ind_p = panel[panel['industry_code'] == code]
            for _, row in ind_p.iterrows():
                j = years.index(int(row['year']))
                mat_data[i, j] = row['mean_breadth']

        # 按最后一年的综合化程度排序
        last_vals = []
        for i in range(len(valid_codes)):
            non_nan = mat_data[i][~np.isnan(mat_data[i])]
            last_vals.append(non_nan[-1] if len(non_nan) > 0 else 0)
        sort_idx = np.argsort(last_vals)[::-1]

        sorted_mat = mat_data[sort_idx]
        sorted_labels = [INDUSTRY20.get(valid_codes[i], valid_codes[i]) for i in sort_idx]

        # 用 navy 色系的 colormap
        from matplotlib.colors import LinearSegmentedColormap
        navy_cmap = LinearSegmentedColormap.from_list(
            'navy', [C['lblue'], C['lred'], C['green'], C['blue'], C['navy'], C['dark']])

        im = ax.imshow(sorted_mat, aspect='auto', cmap=navy_cmap,
                       interpolation='nearest', vmin=8, vmax=22)
        ax.set_xticks(range(len(years)))
        ax.set_xticklabels(years, fontsize=8, rotation=45)
        ax.set_yticks(range(len(sorted_labels)))
        ax.set_yticklabels(sorted_labels, fontsize=9)

        # 在单元格中标注数值
        for i in range(sorted_mat.shape[0]):
            for j in range(sorted_mat.shape[1]):
                val = sorted_mat[i, j]
                if not np.isnan(val):
                    text_color = 'white' if val > 16 else C['dark']
                    ax.text(j, i, f'{val:.1f}', ha='center', va='center',
                            fontsize=6.5, color=text_color)

        _style_ax(ax, title=src)
        ax.spines['left'].set_visible(True)
        ax.spines['bottom'].set_visible(True)

    fig.colorbar(im, ax=axes, shrink=0.6, label='平均技能域数量', pad=0.02)
    fig.suptitle('分行业岗位综合化程度（行业 × 年份）', fontsize=14,
                 fontweight='bold', color=C['dark'], y=1.02)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig3e_industry_heatmap.pdf", dpi=200, bbox_inches='tight')
    plt.close(fig)
    print("  Saved fig3e")


# ──────────────────────────────────────────────
# 3F: 重点行业双轴图 — AI渗透率 & 综合化共变
# ──────────────────────────────────────────────
def analysis_3f(meta):
    if 'industry_code' not in meta.columns:
        print("\n[3F] SKIPPED - no industry data")
        return
    print("\n[3F] Top industry dual-axis: AI exposure & breadth co-movement...")

    # 选 8 个代表性行业
    top_codes = ['C', 'I', 'J', 'K', 'F', 'L', 'P', 'M']

    for src in SOURCES:
        panel = build_industry_year_panel(meta, src)
        if panel.empty:
            continue

        fig, axes = plt.subplots(2, 4, figsize=(20, 9))
        axes_flat = axes.flatten()

        for idx, code in enumerate(top_codes):
            ax = axes_flat[idx]
            ind_p = panel[panel['industry_code'] == code].sort_values('year')

            if len(ind_p) < 2:
                ax.set_title(f'{INDUSTRY20.get(code, code)}\n(数据不足)', fontsize=10)
                continue

            # 左轴：综合化
            ln1 = ax.plot(ind_p['year'], ind_p['mean_breadth'], 'o-',
                          color=C['navy'], linewidth=2, markersize=5, label='技能域数量')
            ax.set_ylabel('技能域数量', fontsize=9, color=C['navy'])
            ax.tick_params(axis='y', labelcolor=C['navy'], labelsize=8)

            # 右轴：AI渗透率
            ax2 = ax.twinx()
            ln2 = ax2.plot(ind_p['year'], ind_p['ai_exposure'] * 100, 's--',
                           color=C['green'], linewidth=2, markersize=5, label='AI渗透率')
            ax2.set_ylabel('AI渗透率 (%)', fontsize=9, color=C['green'])
            ax2.tick_params(axis='y', labelcolor=C['green'], labelsize=8)

            # 计算相关系数
            if len(ind_p) >= 3:
                corr = ind_p[['ai_exposure', 'mean_breadth']].corr().iloc[0, 1]
                ax.text(0.05, 0.95, f'r={corr:.2f}', transform=ax.transAxes,
                        fontsize=9, va='top', color=C['dark'],
                        bbox=dict(boxstyle='round,pad=0.3', facecolor=C['lblue'], alpha=0.9))

            ax.set_title(INDUSTRY20.get(code, code), fontsize=10,
                         fontweight='bold', color=C['dark'])
            ax.tick_params(axis='x', rotation=45, labelsize=8)
            ax.spines['top'].set_visible(False)

            # 合并图例
            lns = ln1 + ln2
            labs = [l.get_label() for l in lns]
            if idx == 0:
                ax.legend(lns, labs, fontsize=7, loc='lower right', framealpha=0.9)

        fig.suptitle(f'重点行业 AI渗透率与综合化共变趋势 — {src}', fontsize=14,
                     fontweight='bold', color=C['dark'], y=1.02)
        fig.tight_layout()
        fig.savefig(FIG_DIR / f"fig3f_comovement_{src}.pdf", dpi=200, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved fig3f ({src})")


# ──────────────────────────────────────────────
# 汇总表 + 面板数据输出
# ──────────────────────────────────────────────
def summary_table(meta):
    print("\n[Summary] Building summary tables...")

    # 1) 整体趋势汇总
    rows = []
    for src in SOURCES:
        sub = meta[meta['data_source'] == src]
        for yr in sorted(sub['year'].unique()):
            yr_data = sub[sub['year'] == yr]
            row = {
                'data_source': src,
                'year': yr,
                'n_jds': len(yr_data),
                'mean_breadth': yr_data['n_skill_clusters'].mean(),
                'median_breadth': yr_data['n_skill_clusters'].median(),
                'p25_breadth': yr_data['n_skill_clusters'].quantile(0.25),
                'p75_breadth': yr_data['n_skill_clusters'].quantile(0.75),
                'ai_exposure': yr_data['ai_binary'].mean(),
            }
            if 'entropy' in yr_data.columns:
                row['mean_entropy'] = yr_data['entropy'].mean()
            rows.append(row)

    summary = pd.DataFrame(rows)
    summary.to_csv(W2V_DIR / "comprehensiveness_summary.csv", index=False, float_format='%.4f')
    print("  Saved comprehensiveness_summary.csv")

    # 2) 行业面板数据
    for src in SOURCES:
        panel = build_industry_year_panel(meta, src)
        if not panel.empty:
            panel.to_csv(W2V_DIR / f"industry_panel_{src}.csv", index=False, float_format='%.4f')
            print(f"  Saved industry_panel_{src}.csv")

    # 3) 打印关键数字
    for src in SOURCES:
        s = summary[summary['data_source'] == src]
        first, last = s.iloc[0], s.iloc[-1]
        print(f"\n  [{src}] {int(first['year'])}→{int(last['year'])}:")
        print(f"    综合化: {first['mean_breadth']:.2f} → {last['mean_breadth']:.2f} ({last['mean_breadth'] - first['mean_breadth']:+.2f})")
        print(f"    AI渗透率: {first['ai_exposure']*100:.2f}% → {last['ai_exposure']*100:.2f}% ({(last['ai_exposure'] - first['ai_exposure'])*100:+.2f}pp)")
        if 'mean_entropy' in s.columns:
            print(f"    Shannon 熵: {first['mean_entropy']:.3f} → {last['mean_entropy']:.3f}")


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────
if __name__ == '__main__':
    print("=" * 60)
    print("Deep-dive: 岗位综合化趋势分析")
    print("=" * 60)

    meta, mat, col_names, col_cn = load_data()

    analysis_3a(meta)
    meta = analysis_3b(meta, mat)
    analysis_3c(meta)
    analysis_3d(meta)
    analysis_3e(meta)
    analysis_3f(meta)
    summary_table(meta)

    print("\n" + "=" * 60)
    print("All comprehensiveness analyses complete!")
    print(f"Figures saved to: {FIG_DIR}")
    print("=" * 60)
