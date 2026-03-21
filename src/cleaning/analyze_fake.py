#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
分析虚假数据的原因
"""

import pandas as pd
import numpy as np
from collections import Counter

def analyze_fake_data(file_path: str):
    """分析虚假数据"""
    print("="*80)
    print("虚假数据分析报告")
    print("="*80)
    
    # 读取数据
    print(f"\n正在读取文件: {file_path}")
    df = pd.read_csv(file_path, encoding='utf-8-sig', low_memory=False)
    
    total = len(df)
    print(f"总计虚假数据: {total:,} 条")
    
    # 分析原因
    reasons = {
        '企业名称缺失': 0,
        '招聘岗位缺失': 0,
        '工作城市缺失': 0,
        '职位描述缺失': 0,
        '职位描述过短(<10字符)': 0,
        '最低月薪异常(<500)': 0,
        '最低月薪异常(>100万)': 0,
        '最高月薪异常(<500)': 0,
        '最高月薪异常(>100万)': 0,
        '薪资逻辑错误(最低>最高)': 0,
        '发布年份异常(<2010)': 0,
        '发布年份异常(>2026)': 0,
    }
    
    print("\n"+ "="*80)
    print("逐条检查虚假原因...")
    print("="*80)
    
    # 逐行分析
    for idx, row in df.iterrows():
        row_reasons = []
        
        # 1. 企业名称
        if pd.isna(row['企业名称']) or str(row['企业名称']).strip() == '':
            reasons['企业名称缺失'] += 1
            row_reasons.append('企业名称缺失')
        
        # 2. 招聘岗位
        if pd.isna(row['招聘岗位']) or str(row['招聘岗位']).strip() == '':
            reasons['招聘岗位缺失'] += 1
            row_reasons.append('招聘岗位缺失')
        
        # 3. 工作城市
        if pd.isna(row['工作城市']) or str(row['工作城市']).strip() == '':
            reasons['工作城市缺失'] += 1
            row_reasons.append('工作城市缺失')
        
        # 4. 职位描述
        if pd.isna(row['职位描述']):
            reasons['职位描述缺失'] += 1
            row_reasons.append('职位描述缺失')
        else:
            desc = str(row['职位描述']).strip()
            if len(desc) < 10:
                reasons['职位描述过短(<10字符)'] += 1
                row_reasons.append(f'职位描述过短(仅{len(desc)}字符)')
        
        # 5. 薪资检查
        try:
            min_salary = float(row['最低月薪']) if pd.notna(row['最低月薪']) else 0
            max_salary = float(row['最高月薪']) if pd.notna(row['最高月薪']) else 0
            
            if min_salary > 0:
                if min_salary < 500:
                    reasons['最低月薪异常(<500)'] += 1
                    row_reasons.append(f'最低月薪过低({min_salary})')
                elif min_salary > 1000000:
                    reasons['最低月薪异常(>100万)'] += 1
                    row_reasons.append(f'最低月薪过高({min_salary})')
            
            if max_salary > 0:
                if max_salary < 500:
                    reasons['最高月薪异常(<500)'] += 1
                    row_reasons.append(f'最高月薪过低({max_salary})')
                elif max_salary > 1000000:
                    reasons['最高月薪异常(>100万)'] += 1
                    row_reasons.append(f'最高月薪过高({max_salary})')
            
            if min_salary > 0 and max_salary > 0 and min_salary > max_salary:
                reasons['薪资逻辑错误(最低>最高)'] += 1
                row_reasons.append(f'薪资逻辑错误({min_salary}>{max_salary})')
        except:
            pass
        
        # 6. 年份检查
        try:
            year = int(row['招聘发布年份']) if pd.notna(row['招聘发布年份']) else 0
            if year > 0:
                if year < 2010:
                    reasons['发布年份异常(<2010)'] += 1
                    row_reasons.append(f'年份过早({year})')
                elif year > 2026:
                    reasons['发布年份异常(>2026)'] += 1
                    row_reasons.append(f'年份过晚({year})')
        except:
            pass
        
        # 显示前10个样例
        if idx < 10:
            print(f"\n样例 {idx+1}:")
            print(f"  企业: {row['企业名称']}")
            print(f"  岗位: {row['招聘岗位']}")
            print(f"  城市: {row['工作城市']}")
            desc = str(row['职位描述'])[:50] if pd.notna(row['职位描述']) else 'N/A'
            print(f"  描述: {desc}...")
            print(f"  薪资: {row['最低月薪']} - {row['最高月薪']}")
            print(f"  年份: {row['招聘发布年份']}")
            print(f"  ❌ 虚假原因: {', '.join(row_reasons)}")
    
    # 统计汇总
    print("\n" + "="*80)
    print("虚假原因统计")
    print("="*80)
    
    sorted_reasons = sorted(reasons.items(), key=lambda x: x[1], reverse=True)
    for reason, count in sorted_reasons:
        if count > 0:
            percentage = count / total * 100
            print(f"{reason:30s}: {count:8,} 条 ({percentage:5.1f}%)")
    
    # 组合原因分析
    print("\n" + "="*80)
    print("数据质量问题分布")
    print("="*80)
    
    # 关键字段缺失率
    missing_stats = {
        '企业名称': df['企业名称'].isna().sum(),
        '招聘岗位': df['招聘岗位'].isna().sum(),
        '工作城市': df['工作城市'].isna().sum(),
        '职位描述': df['职位描述'].isna().sum(),
        '最低月薪': df['最低月薪'].isna().sum(),
        '最高月薪': df['最高月薪'].isna().sum(),
    }
    
    print("\n字段缺失率:")
    for field, count in missing_stats.items():
        percentage = count / total * 100
        print(f"  {field:15s}: {count:8,} 条缺失 ({percentage:5.1f}%)")
    
    # 职位描述长度分布
    print("\n职位描述长度分布:")
    desc_lengths = df['职位描述'].apply(lambda x: len(str(x)) if pd.notna(x) else 0)
    print(f"  平均长度: {desc_lengths.mean():.1f} 字符")
    print(f"  中位数: {desc_lengths.median():.0f} 字符")
    print(f"  最小值: {desc_lengths.min():.0f} 字符")
    print(f"  最大值: {desc_lengths.max():.0f} 字符")
    print(f"  <10字符: {(desc_lengths < 10).sum():,} 条 ({(desc_lengths < 10).sum()/total*100:.1f}%)")
    
    # 薪资分布
    print("\n薪资异常分布:")
    valid_min = df['最低月薪'].dropna()
    valid_max = df['最高月薪'].dropna()
    
    if len(valid_min) > 0:
        print(f"  最低月薪:")
        print(f"    <500元: {(valid_min < 500).sum():,} 条")
        print(f"    >100万: {(valid_min > 1000000).sum():,} 条")
        print(f"    正常范围(500-100万): {((valid_min >= 500) & (valid_min <= 1000000)).sum():,} 条")
    
    if len(valid_max) > 0:
        print(f"  最高月薪:")
        print(f"    <500元: {(valid_max < 500).sum():,} 条")
        print(f"    >100万: {(valid_max > 1000000).sum():,} 条")
        print(f"    正常范围(500-100万): {((valid_max >= 500) & (valid_max <= 1000000)).sum():,} 条")
    
    # 年份分布
    print("\n年份分布:")
    valid_years = df['招聘发布年份'].dropna()
    if len(valid_years) > 0:
        year_counts = valid_years.value_counts().sort_index()
        print(f"  年份范围: {valid_years.min():.0f} - {valid_years.max():.0f}")
        print(f"  异常年份(<2010 或 >2026): {((valid_years < 2010) | (valid_years > 2026)).sum():,} 条")
    
    print("\n" + "="*80)
    print("分析完成")
    print("="*80)


if __name__ == "__main__":
    file_path = '/Users/yu/code/code2601/TY/data_cleaning/removed_fake.csv'
    analyze_fake_data(file_path)
