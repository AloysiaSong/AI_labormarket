#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
生成ESCO技能评分统计报告
"""

import pandas as pd
import numpy as np

def generate_report(csv_file: str):
    """生成详细的统计报告"""
    
    print("="*80)
    print("ESCO技能评分系统 - 数据处理报告 (方案C)")
    print("="*80)
    
    # 读取样本数据
    print("\n正在读取数据...")
    df = pd.read_csv(csv_file, nrows=50000)
    
    print(f"✓ 成功读取 {len(df)} 行样本数据")
    
    # 识别技能列
    skill_columns = [col for col in df.columns if col.startswith(('1.', '2.', '3.', '4.', '5.', '6.', '7.', '8.'))]
    
    print(f"\n共识别到 {len(skill_columns)} 个ESCO能力维度")
    
    # 1. 总体统计
    print("\n" + "-"*80)
    print("1. 总体统计")
    print("-"*80)
    
    for col in skill_columns:
        avg_score = df[col].mean()
        zero_ratio = (df[col] == 0).sum() / len(df) * 100
        nonzero_avg = df[df[col] > 0][col].mean() if (df[col] > 0).any() else 0
        
        print(f"{col:30s} | 平均分: {avg_score:.2f} | 0分占比: {zero_ratio:5.1f}% | 非0平均: {nonzero_avg:.2f}")
    
    # 2. 分数分布
    print("\n" + "-"*80)
    print("2. 分数分布统计 (所有能力维度合计)")
    print("-"*80)
    
    all_scores = []
    for col in skill_columns:
        all_scores.extend(df[col].tolist())
    
    score_dist = pd.Series(all_scores).value_counts().sort_index()
    total_scores = len(all_scores)
    
    for score in range(6):
        count = score_dist.get(score, 0)
        ratio = count / total_scores * 100
        print(f"  {score}分: {count:8d} ({ratio:5.2f}%)")
    
    # 3. 按岗位类型分析（如果有招聘类别字段）
    if '招聘岗位' in df.columns:
        print("\n" + "-"*80)
        print("3. 热门岗位能力要求分析 (Top 10)")
        print("-"*80)
        
        top_jobs = df['招聘岗位'].value_counts().head(10)
        
        for job_title in top_jobs.index[:5]:  # 分析前5个
            job_df = df[df['招聘岗位'] == job_title]
            print(f"\n{job_title} (样本数: {len(job_df)})")
            
            # 找出该岗位要求最高的3个能力
            avg_scores = {}
            for col in skill_columns:
                avg = job_df[col].mean()
                if avg > 0:
                    avg_scores[col] = avg
            
            sorted_skills = sorted(avg_scores.items(), key=lambda x: x[1], reverse=True)[:3]
            
            for skill, score in sorted_skills:
                print(f"  {skill}: {score:.2f}")
    
    # 4. 学历与能力的关系
    if '学历要求' in df.columns:
        print("\n" + "-"*80)
        print("4. 学历与能力要求的关系")
        print("-"*80)
        
        edu_levels = ['中专', '大专', '本科', '硕士']
        
        for edu in edu_levels:
            edu_df = df[df['学历要求'] == edu]
            if len(edu_df) > 0:
                avg_skill = edu_df[skill_columns].mean().mean()
                print(f"  {edu:6s}: 平均能力得分 {avg_skill:.2f} (样本数: {len(edu_df)})")
    
    # 5. 关键发现
    print("\n" + "-"*80)
    print("5. 关键发现")
    print("-"*80)
    
    # 找出最常要求的能力
    avg_scores_all = df[skill_columns].mean().sort_values(ascending=False)
    print("\n✓ 最常要求的5个能力维度:")
    for i, (skill, score) in enumerate(avg_scores_all.head(5).items(), 1):
        print(f"  {i}. {skill}: {score:.2f}")
    
    # 找出最少要求的能力
    print("\n✓ 最少要求的5个能力维度:")
    for i, (skill, score) in enumerate(avg_scores_all.tail(5).items(), 1):
        print(f"  {i}. {skill}: {score:.2f}")
    
    # 零分比例分析
    zero_ratios = {}
    for col in skill_columns:
        zero_ratio = (df[col] == 0).sum() / len(df) * 100
        zero_ratios[col] = zero_ratio
    
    print("\n✓ 涉及率最高的5个能力 (0分占比最低):")
    sorted_zero = sorted(zero_ratios.items(), key=lambda x: x[1])[:5]
    for skill, ratio in sorted_zero:
        print(f"  {skill}: {100-ratio:.1f}% 的岗位涉及")
    
    print("\n" + "="*80)
    print("报告生成完成")
    print("="*80)


if __name__ == "__main__":
    # 使用集中路径配置
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from config.paths import ESCO_OUTPUT_DIR

    csv_file = str(ESCO_OUTPUT_DIR / '2016_esco_A.csv')
    generate_report(csv_file)
