#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
使用AI助手批量翻译ESCO技能库
适合小批量翻译或演示用途
"""

import pandas as pd
import json
import os
from typing import List, Dict

# AI助手提供的高质量翻译映射
# 这些是常见技能的专业翻译
SKILL_TRANSLATIONS = {
    "manage musical staff": "管理音乐工作人员",
    "manage customer service": "管理客户服务",
    "manage staff": "管理员工",
    "manage contracts": "管理合同",
    "manage budget": "管理预算",
    "program software": "编程软件",
    "develop software": "开发软件",
    "design software": "设计软件",
    "test software": "测试软件",
    "maintain software": "维护软件",
    "communicate with customers": "与客户沟通",
    "provide customer service": "提供客户服务",
    "handle customer complaints": "处理客户投诉",
    "process data": "处理数据",
    "analyze data": "分析数据",
    "manage database": "管理数据库",
    "design database": "设计数据库",
    "administer database": "管理数据库系统",
    "write documentation": "编写文档",
    "prepare technical documentation": "准备技术文档",
    "create project documentation": "创建项目文档",
    "manage projects": "管理项目",
    "coordinate projects": "协调项目",
    "plan projects": "规划项目",
    "lead teams": "领导团队",
    "supervise staff": "监督员工",
    "train employees": "培训员工",
    "recruit employees": "招聘员工",
    "evaluate employee performance": "评估员工绩效",
    "develop business relationships": "发展业务关系",
    "maintain business relationships": "维护业务关系",
    "negotiate contracts": "谈判合同",
    "prepare reports": "准备报告",
    "write reports": "撰写报告",
    "present reports": "展示报告",
    "conduct research": "开展研究",
    "perform research": "执行研究",
    "analyze research data": "分析研究数据",
    "organize meetings": "组织会议",
    "schedule meetings": "安排会议",
    "facilitate meetings": "主持会议",
    "use computer": "使用计算机",
    "use spreadsheet software": "使用电子表格软件",
    "use word processing software": "使用文字处理软件",
    "use presentation software": "使用演示文稿软件",
    "speak different languages": "说不同的语言",
    "speak English": "说英语",
    "write in English": "用英语写作",
    "solve problems": "解决问题",
    "think critically": "批判性思维",
    "make decisions": "做决策",
    "work independently": "独立工作",
    "work in teams": "团队协作",
    "adapt to change": "适应变化",
    "handle stress": "处理压力",
    "manage time": "管理时间",
    "prioritize tasks": "优先处理任务",
    "meet deadlines": "按时完成任务",
}


def load_skills_csv(csv_path: str) -> pd.DataFrame:
    """加载技能CSV文件"""
    print(f"加载文件: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"✓ 加载了 {len(df)} 条技能")
    return df


def apply_translations(df: pd.DataFrame, translations: Dict[str, str]) -> pd.DataFrame:
    """应用预定义的翻译"""
    if 'description_cn' not in df.columns:
        df['description_cn'] = ''
    
    count = 0
    for idx, row in df.iterrows():
        en_text = row['description'].lower().strip()
        
        # 精确匹配
        if en_text in translations:
            df.at[idx, 'description_cn'] = translations[en_text]
            count += 1
        # 部分匹配（包含关键词）
        elif not df.at[idx, 'description_cn']:
            for en_key, cn_val in translations.items():
                if en_key in en_text:
                    df.at[idx, 'description_cn'] = cn_val
                    count += 1
                    break
    
    print(f"✓ 应用了 {count} 条预定义翻译")
    return df


def generate_translation_sample(df: pd.DataFrame, output_file: str, sample_size: int = 100):
    """生成翻译样本供AI助手翻译"""
    # 选择未翻译的样本
    untranslated = df[df['description_cn'].isna() | (df['description_cn'] == '')]
    
    if len(untranslated) == 0:
        print("✓ 所有技能已翻译完成！")
        return
    
    sample = untranslated.head(sample_size)
    
    # 生成JSON格式便于AI处理
    sample_data = []
    for idx, row in sample.iterrows():
        sample_data.append({
            'id': row['id'],
            'english': row['description'],
            'chinese': ''  # 待填充
        })
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(sample_data, f, ensure_ascii=False, indent=2)
    
    print(f"✓ 已生成 {len(sample)} 条待翻译样本: {output_file}")
    print(f"剩余未翻译: {len(untranslated)} 条")


def main():
    """主函数"""
    print("="*80)
    print("AI助手翻译系统")
    print("="*80)
    
    input_csv = '/Users/yu/code/miniconda3/lib/python3.13/site-packages/esco_skill_extractor/data/skills.csv'
    output_csv = '/Users/yu/code/code2601/TY/Test_ESCO/skills_chinese.csv'
    
    # 加载或创建数据
    if os.path.exists(output_csv):
        print(f"\n发现已有文件，继续翻译...")
        df = pd.read_csv(output_csv)
    else:
        print(f"\n创建新的翻译文件...")
        df = load_skills_csv(input_csv)
        if 'description_cn' not in df.columns:
            df['description_cn'] = ''
    
    print(f"\n当前状态:")
    total = len(df)
    translated = df['description_cn'].notna().sum() - (df['description_cn'] == '').sum()
    print(f"  总计: {total} 条")
    print(f"  已翻译: {translated} 条")
    print(f"  未翻译: {total - translated} 条")
    
    # 应用预定义翻译
    print(f"\n应用预定义翻译...")
    df = apply_translations(df, SKILL_TRANSLATIONS)
    
    # 保存
    df.to_csv(output_csv, index=False, encoding='utf-8-sig')
    print(f"✓ 已保存至: {output_csv}")
    
    # 生成样本
    sample_file = '/Users/yu/code/code2601/TY/Test_ESCO/translation_sample.json'
    generate_translation_sample(df, sample_file, sample_size=100)
    
    print("\n" + "="*80)
    print("建议：使用免费的百度翻译API完成剩余翻译")
    print("="*80)
    print("1. 百度翻译API完全免费（200万字符/月）")
    print("2. 运行: python translate_with_baidu.py")
    print("3. 只需要注册百度账号即可")


if __name__ == "__main__":
    main()
