#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
可视化分析结果

生成图表：
1. 综合化指数时序图（LDA vs SBERT对比）
2. 技能演化热力图
3. 主题对齐质量评估
4. 相关性散点图（稳健性检验）
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys

# 配置中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 路径配置
PROJECT_ROOT = Path(__file__).parent.parent
LDA_OUTPUT = PROJECT_ROOT / "output" / "lda"
SBERT_OUTPUT = PROJECT_ROOT / "output" / "sbert"
FIGURE_OUTPUT = PROJECT_ROOT / "output" / "reports" / "figures"
FIGURE_OUTPUT.mkdir(parents=True, exist_ok=True)

sns.set_style("whitegrid")
sns.set_context("talk")


def plot_comprehensiveness_trend():
    """绘制综合化指数时序图"""
    print("\n📊 绘制综合化指数趋势...")
    
    # 加载LDA结果
    lda_path = LDA_OUTPUT / "analysis" / "comprehensiveness_time_series.csv"
    if not lda_path.exists():
        print(f"  ⚠️ LDA结果不存在: {lda_path}")
        return
    
    lda_df = pd.read_csv(lda_path, index_col='Window')
    
    # 提取年份
    def extract_year(window_name):
        parts = window_name.split('_')
        try:
            return int(parts[1])
        except:
            return None
    
    lda_df['year'] = lda_df.index.map(extract_year)
    lda_df = lda_df.dropna(subset=['year']).sort_values('year')
    
    # 加载SBERT结果（如果存在）
    sbert_path = SBERT_OUTPUT / "fingerprint_evolution_TY.csv"
    has_sbert = sbert_path.exists()
    
    if has_sbert:
        sbert_df = pd.read_csv(sbert_path, index_col='Window')
        # 计算SBERT熵
        sbert_entropy = []
        for idx, row in sbert_df.iterrows():
            skill_vec = row.values
            prob = skill_vec / skill_vec.sum()
            entropy = -np.sum(prob * np.log(prob + 1e-10))
            normalized = entropy / np.log(len(skill_vec))
            sbert_entropy.append(normalized)
        
        sbert_df['comprehensiveness'] = sbert_entropy
        sbert_df['year'] = sbert_df.index.map(extract_year)
        sbert_df = sbert_df.dropna(subset=['year']).sort_values('year')
    
    # 绘图
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # LDA曲线
    ax.plot(lda_df['year'], lda_df['lda_mean'], 
            marker='o', linewidth=2.5, markersize=8,
            label='LDA Method', color='#2E86AB')
    
    # 误差带
    ax.fill_between(lda_df['year'], 
                     lda_df['lda_mean'] - lda_df['lda_std'],
                     lda_df['lda_mean'] + lda_df['lda_std'],
                     alpha=0.2, color='#2E86AB')
    
    # SBERT曲线（如果存在）
    if has_sbert:
        ax.plot(sbert_df['year'], sbert_df['comprehensiveness'],
                marker='s', linewidth=2.5, markersize=8,
                label='SBERT Method', color='#A23B72')
    
    ax.set_xlabel('Year', fontsize=14, fontweight='bold')
    ax.set_ylabel('Job Comprehensiveness Index', fontsize=14, fontweight='bold')
    ax.set_title('Evolution of Job Comprehensiveness (2015-2025)', 
                 fontsize=16, fontweight='bold', pad=20)
    ax.legend(loc='best', fontsize=12, frameon=True)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = FIGURE_OUTPUT / "comprehensiveness_trend.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  ✅ 保存至: {output_path}")
    plt.close()


def plot_skill_heatmap():
    """绘制技能演化热力图"""
    print("\n🔥 绘制技能演化热力图...")
    
    sbert_path = SBERT_OUTPUT / "fingerprint_evolution_TY.csv"
    if not sbert_path.exists():
        print(f"  ⚠️ SBERT结果不存在")
        return
    
    df = pd.read_csv(sbert_path, index_col='Window')
    
    # 提取年份作为行标签
    def extract_year(window_name):
        parts = window_name.split('_')
        try:
            return f"{parts[1]}-{parts[2]}"
        except:
            return window_name
    
    df.index = df.index.map(extract_year)
    
    # 选择前10个技能（避免过于密集）
    top_skills = df.iloc[:, :10]
    
    # 绘制热力图
    fig, ax = plt.subplots(figsize=(14, 8))
    sns.heatmap(top_skills.T, annot=True, fmt='.3f', 
                cmap='YlOrRd', cbar_kws={'label': 'Similarity Score'},
                linewidths=0.5, ax=ax)
    
    ax.set_xlabel('Time Window', fontsize=12, fontweight='bold')
    ax.set_ylabel('Skill Dimension', fontsize=12, fontweight='bold')
    ax.set_title('Skill Profile Evolution Heatmap', 
                 fontsize=14, fontweight='bold', pad=15)
    
    plt.tight_layout()
    output_path = FIGURE_OUTPUT / "skill_heatmap.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  ✅ 保存至: {output_path}")
    plt.close()


def plot_alignment_quality():
    """绘制主题对齐质量"""
    print("\n📐 绘制主题对齐质量...")
    
    quality_path = LDA_OUTPUT / "alignment" / "alignment_quality.csv"
    if not quality_path.exists():
        print(f"  ⚠️ 对齐质量数据不存在")
        return
    
    df = pd.read_csv(quality_path, index_col=0)
    
    # 提取年份
    def extract_year(window_name):
        parts = window_name.split('_')
        try:
            return int(parts[1])
        except:
            return None
    
    df['year'] = df.index.map(extract_year)
    df = df.dropna(subset=['year']).sort_values('year')
    
    # 绘制柱状图
    fig, ax = plt.subplots(figsize=(10, 6))
    
    bars = ax.bar(df['year'], df['Alignment_Quality'], 
                   color=['#2E86AB' if x > 0.7 else '#F77F00' if x > 0.5 else '#D62828' 
                          for x in df['Alignment_Quality']],
                   edgecolor='black', linewidth=1.2)
    
    # 添加阈值线
    ax.axhline(y=0.7, color='green', linestyle='--', linewidth=2, 
               label='Excellent (>0.7)', alpha=0.7)
    ax.axhline(y=0.5, color='orange', linestyle='--', linewidth=2,
               label='Acceptable (>0.5)', alpha=0.7)
    
    ax.set_xlabel('Year', fontsize=14, fontweight='bold')
    ax.set_ylabel('Topic Alignment Quality', fontsize=14, fontweight='bold')
    ax.set_title('LDA Topic Alignment Quality Across Windows',
                 fontsize=16, fontweight='bold', pad=20)
    ax.legend(loc='best', fontsize=11)
    ax.set_ylim(0, 1)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    output_path = FIGURE_OUTPUT / "alignment_quality.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  ✅ 保存至: {output_path}")
    plt.close()


def plot_robustness_check():
    """绘制稳健性检验散点图"""
    print("\n🔬 绘制稳健性检验散点图...")
    
    robustness_path = LDA_OUTPUT / "analysis" / "robustness_check.csv"
    if not robustness_path.exists():
        print(f"  ⚠️ 稳健性检验结果不存在")
        print(f"  💡 提示：运行 build_comprehensiveness_index.py 中的 validate_with_sbert()")
        return
    
    df = pd.read_csv(robustness_path)
    
    # 提取相关系数
    pearson_r = df['pearson_r'].iloc[0]
    
    # 绘制散点图
    fig, ax = plt.subplots(figsize=(8, 8))
    
    ax.scatter(df['lda_mean'], df['comprehensiveness_sbert'],
               s=150, alpha=0.7, c=df['year'], cmap='viridis',
               edgecolors='black', linewidth=1.5)
    
    # 拟合线
    z = np.polyfit(df['lda_mean'], df['comprehensiveness_sbert'], 1)
    p = np.poly1d(z)
    x_line = np.linspace(df['lda_mean'].min(), df['lda_mean'].max(), 100)
    ax.plot(x_line, p(x_line), "r--", linewidth=2, label=f'Fit Line')
    
    # 添加文本
    ax.text(0.05, 0.95, f'Pearson r = {pearson_r:.3f}',
            transform=ax.transAxes, fontsize=14, fontweight='bold',
            verticalalignment='top', bbox=dict(boxstyle='round', 
                                                facecolor='wheat', 
                                                alpha=0.5))
    
    ax.set_xlabel('LDA Comprehensiveness Index', fontsize=14, fontweight='bold')
    ax.set_ylabel('SBERT Comprehensiveness Index', fontsize=14, fontweight='bold')
    ax.set_title('Robustness Check: LDA vs SBERT Methods',
                 fontsize=16, fontweight='bold', pad=20)
    ax.legend(loc='best', fontsize=11)
    ax.grid(True, alpha=0.3)
    
    # 添加颜色条
    cbar = plt.colorbar(ax.collections[0], ax=ax)
    cbar.set_label('Year', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    output_path = FIGURE_OUTPUT / "robustness_scatter.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  ✅ 保存至: {output_path}")
    plt.close()


def main():
    print("="*70)
    print("📊 开始生成可视化报告")
    print("="*70)
    
    # 1. 综合化指数趋势
    plot_comprehensiveness_trend()
    
    # 2. 技能热力图
    plot_skill_heatmap()
    
    # 3. 对齐质量
    plot_alignment_quality()
    
    # 4. 稳健性检验
    plot_robustness_check()
    
    print("\n" + "="*70)
    print(f"✅ 所有图表已生成！")
    print(f"📁 输出目录: {FIGURE_OUTPUT}")
    print("="*70)


if __name__ == "__main__":
    main()
