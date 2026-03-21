#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
重构ESCO技能列：删除旧列，添加基于ESCO官方框架的新列
"""

import pandas as pd
import numpy as np

# 定义新的ESCO列
NEW_ESCO_COLUMNS = [
    # 1. Communication, collaboration and creativity
    '1.Communication_collaboration_creativity',
    '1.1_沟通技能',
    '1.2_团队协作',
    '1.3_创造性思维',
    # 2. Information skills
    '2.Information_skills',
    '2.1_信息处理',
    '2.2_数字素养',
    '2.3_数据分析',
    # 3. Language skills
    '3.Language_skills',
    '3.1_母语能力',
    '3.2_外语能力',
    # 4. Management skills
    '4.Management_skills',
    '4.1_项目管理',
    '4.2_人员管理',
    '4.3_资源管理',
    # 5. Physical abilities
    '5.Physical_abilities',
    '5.1_体力要求',
    '5.2_精细动作技能',
    # 6. Learning to learn
    '6.Learning_to_learn',
    '6.1_自主学习',
    '6.2_适应能力',
    # 7. Problem solving
    '7.Problem_solving',
    '7.1_分析思维',
    '7.2_决策能力',
    # 8. Digital competences
    '8.Digital_competences',
    '8.1_数字技术使用',
    '8.2_软件应用'
]

# 要删除的旧列
COLUMNS_TO_REMOVE = [
    'ESCO技能类别',
    '主要技能',
    '技能水平标注',
    '整体技能要求等级'
]


def restructure_data(input_file: str, output_file: str, chunk_size: int = 10000):
    """
    重构数据：删除旧列，添加新的ESCO列
    
    Args:
        input_file: 输入CSV文件路径
        output_file: 输出CSV文件路径
        chunk_size: 每次处理的行数
    """
    print(f"开始重构文件: {input_file}")
    print(f"删除列: {COLUMNS_TO_REMOVE}")
    print(f"添加 {len(NEW_ESCO_COLUMNS)} 个新列")
    
    first_chunk = True
    total_processed = 0
    
    # 分块读取处理
    for chunk_id, chunk in enumerate(pd.read_csv(input_file, chunksize=chunk_size, low_memory=False)):
        print(f"\r处理第 {chunk_id + 1} 块数据，共 {len(chunk)} 行...", end='')
        
        # 删除旧列（如果存在）
        for col in COLUMNS_TO_REMOVE:
            if col in chunk.columns:
                chunk = chunk.drop(columns=[col])
        
        # 添加新列（初始化为空字符串）
        for col in NEW_ESCO_COLUMNS:
            chunk[col] = ''
        
        # 保存结果
        if first_chunk:
            chunk.to_csv(output_file, index=False, encoding='utf-8-sig')
            first_chunk = False
        else:
            chunk.to_csv(output_file, index=False, mode='a', header=False, encoding='utf-8-sig')
        
        total_processed += len(chunk)
    
    print(f"\n\n✓ 重构完成！")
    print(f"  总处理行数: {total_processed:,}")
    print(f"  结果已保存至: {output_file}")


def verify_structure(file_path: str):
    """验证文件结构"""
    print("\n" + "="*60)
    print("验证文件结构...")
    
    # 只读取第一行来查看列名
    df_sample = pd.read_csv(file_path, nrows=5)
    
    print(f"\n文件列数: {len(df_sample.columns)}")
    print("\n所有列名:")
    for i, col in enumerate(df_sample.columns, 1):
        print(f"  {i}. {col}")
    
    print("\n新增的ESCO列:")
    for col in NEW_ESCO_COLUMNS:
        if col in df_sample.columns:
            print(f"  ✓ {col}")
        else:
            print(f"  ✗ {col} (未找到)")
    
    print("\n已删除的列:")
    for col in COLUMNS_TO_REMOVE:
        if col not in df_sample.columns:
            print(f"  ✓ {col} (已删除)")
        else:
            print(f"  ✗ {col} (仍存在)")
    
    print("\n前3行数据预览:")
    print(df_sample[['招聘岗位', '学历要求', '要求经验']].head(3))
    
    print("="*60)


if __name__ == "__main__":
    # 使用集中路径配置
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from config.paths import ESCO_OUTPUT_DIR

    input_file = str(ESCO_OUTPUT_DIR / '2016_esco.csv')
    output_file = str(ESCO_OUTPUT_DIR / '2016_esco_restructured.csv')
    
    # 重构数据
    restructure_data(input_file, output_file, chunk_size=10000)
    
    # 验证结果
    verify_structure(output_file)
    
    print("\n准备替换原文件...")
    import os
    import shutil
    
    # 备份原文件
    backup_file = input_file + '.backup'
    print(f"备份原文件到: {backup_file}")
    shutil.move(input_file, backup_file)
    
    # 用新文件替换
    print(f"用重构后的文件替换原文件")
    shutil.move(output_file, input_file)
    
    print("\n✓ 全部完成！")
    print(f"  原文件已备份至: {backup_file}")
    print(f"  更新后的文件: {input_file}")
