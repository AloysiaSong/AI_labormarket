#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ESCO技能框架职位映射脚本
基于ESCO (European Skills, Competences, Qualifications and Occupations) 框架
对智联招聘职位描述进行技能类别和要求水平的映射
"""

import pandas as pd
import re
from typing import List, Dict, Tuple
import numpy as np

# ESCO技能分类框架（简化版，基于主要技能类别）
ESCO_SKILL_CATEGORIES = {
    'communication': {
        'keywords': ['沟通', '交流', '表达', '演讲', '汇报', '客户', '接待', '协调', '公关'],
        'level_indicators': {
            '初级': ['协助', '配合', '辅助'],
            '中级': ['负责', '独立', '主导'],
            '高级': ['管理', '领导', '统筹', '决策']
        }
    },
    'digital_technology': {
        'keywords': ['Office', 'Excel', 'Word', 'PPT', 'PS', 'CAD', '3DMAX', '软件', '电脑', '计算机', '网络', '编程', 'Python', 'Java', 'C++', '数据库', 'SQL', 'ERP', 'CRM'],
        'level_indicators': {
            '初级': ['熟悉', '了解', '会使用', '基本'],
            '中级': ['熟练', '精通', '掌握'],
            '高级': ['专家', '架构', '开发', '设计']
        }
    },
    'language': {
        'keywords': ['英语', '日语', '韩语', '法语', '德语', '西班牙语', '外语', '翻译', '口译'],
        'level_indicators': {
            '初级': ['一般', '基础', '简单'],
            '中级': ['流利', '熟练', '良好'],
            '高级': ['精通', '母语', '同声传译']
        }
    },
    'management': {
        'keywords': ['管理', '团队', '领导', '计划', '组织', '协调', '监督', '培训', '指导', '绩效'],
        'level_indicators': {
            '初级': ['协助', '参与', '支持'],
            '中级': ['负责', '管理', '带领'],
            '高级': ['总监', '战略', '高级', '全面']
        }
    },
    'analytical': {
        'keywords': ['分析', '数据', '统计', '研究', '评估', '预测', '建模', '报表'],
        'level_indicators': {
            '初级': ['收集', '整理', '记录'],
            '中级': ['分析', '评估', '汇总'],
            '高级': ['决策', '战略', '规划']
        }
    },
    'sales_marketing': {
        'keywords': ['销售', '营销', '推广', '客户开发', '市场', '商务', '业绩', '业务'],
        'level_indicators': {
            '初级': ['协助', '跟进', '维护'],
            '中级': ['独立', '负责', '开发'],
            '高级': ['策略', '管理', '总监']
        }
    },
    'design_creative': {
        'keywords': ['设计', '创意', '美术', '平面', '产品设计', 'UI', 'UX', '广告', '策划'],
        'level_indicators': {
            '初级': ['助理', '初级', '辅助'],
            '中级': ['独立', '主导', '设计师'],
            '高级': ['总监', '创意总监', '首席']
        }
    },
    'financial': {
        'keywords': ['会计', '财务', '审计', '税务', '成本', '预算', '账务', '报税', '结算'],
        'level_indicators': {
            '初级': ['出纳', '助理', '文员'],
            '中级': ['会计', '主管', '负责'],
            '高级': ['经理', '总监', '财务总监']
        }
    },
    'technical_specialist': {
        'keywords': ['工程', '技术', '维修', '维护', '测试', '检测', '质量', '生产', '研发'],
        'level_indicators': {
            '初级': ['技工', '操作', '辅助'],
            '中级': ['工程师', '技术员', '负责'],
            '高级': ['专家', '总工', '高级工程师']
        }
    },
    'administrative': {
        'keywords': ['行政', '文员', '前台', '助理', '秘书', '人事', 'HR', '办公'],
        'level_indicators': {
            '初级': ['文员', '助理', '前台'],
            '中级': ['主管', '专员', '负责'],
            '高级': ['经理', '总监', '人力资源总监']
        }
    }
}

# 教育水平映射
EDUCATION_LEVEL_MAP = {
    '不限': 0,
    '中专': 1,
    '高中': 1,
    '大专': 2,
    '本科': 3,
    '硕士': 4,
    '博士': 5,
    'MBA': 4
}

# 经验水平映射
EXPERIENCE_LEVEL_MAP = {
    '不限': 0,
    '应届毕业生': 0,
    '1年以下': 1,
    '1-3年': 2,
    '3-5年': 3,
    '5-10年': 4,
    '10年以上': 5
}


def extract_skills_from_description(description: str) -> List[Dict[str, any]]:
    """
    从职位描述中提取技能类别和要求水平
    
    Args:
        description: 职位描述文本
        
    Returns:
        技能列表，包含类别和水平
    """
    if pd.isna(description) or not isinstance(description, str):
        return []
    
    description = description.lower()
    skills_found = []
    
    for category, info in ESCO_SKILL_CATEGORIES.items():
        # 检查是否包含该类别的关键词
        keywords_found = [kw for kw in info['keywords'] if kw.lower() in description]
        
        if keywords_found:
            # 确定技能水平
            level = '未指定'
            for lvl, indicators in info['level_indicators'].items():
                if any(ind.lower() in description for ind in indicators):
                    level = lvl
                    break
            
            skills_found.append({
                'category': category,
                'level': level,
                'keywords_matched': keywords_found
            })
    
    return skills_found


def determine_overall_skill_level(skills: List[Dict], education: str, experience: str) -> str:
    """
    综合确定职位的整体技能要求水平
    
    Args:
        skills: 提取的技能列表
        education: 学历要求
        experience: 经验要求
        
    Returns:
        整体技能水平：初级/中级/高级/专家
    """
    # 计算得分
    edu_score = EDUCATION_LEVEL_MAP.get(education, 0)
    exp_score = EXPERIENCE_LEVEL_MAP.get(experience, 0)
    
    skill_score = 0
    if skills:
        level_scores = {'初级': 1, '中级': 2, '高级': 3, '未指定': 0}
        skill_score = sum(level_scores.get(s['level'], 0) for s in skills) / len(skills)
    
    # 综合评分
    total_score = edu_score * 0.3 + exp_score * 0.3 + skill_score * 0.4
    
    if total_score < 1.5:
        return '初级'
    elif total_score < 2.5:
        return '中级'
    elif total_score < 3.5:
        return '高级'
    else:
        return '专家'


def process_job_data(input_file: str, output_file: str, chunk_size: int = 10000):
    """
    处理招聘数据，添加ESCO技能映射
    
    Args:
        input_file: 输入CSV文件路径
        output_file: 输出CSV文件路径
        chunk_size: 每次处理的行数
    """
    print(f"开始处理文件: {input_file}")
    print(f"分块大小: {chunk_size} 行")
    
    first_chunk = True
    total_processed = 0
    
    # 分块读取处理大文件
    for chunk_id, chunk in enumerate(pd.read_csv(input_file, chunksize=chunk_size, low_memory=False)):
        print(f"\n处理第 {chunk_id + 1} 块数据，共 {len(chunk)} 行...")
        
        # 添加新列
        chunk['ESCO技能类别'] = ''
        chunk['主要技能'] = ''
        chunk['技能水平标注'] = ''
        chunk['整体技能要求等级'] = ''
        
        # 逐行处理
        for idx, row in chunk.iterrows():
            description = row.get('职位描述', '')
            education = row.get('学历要求', '不限')
            experience = row.get('要求经验', '不限')
            
            # 提取技能
            skills = extract_skills_from_description(description)
            
            if skills:
                # 技能类别（所有匹配的类别）
                categories = list(set([s['category'] for s in skills]))
                chunk.at[idx, 'ESCO技能类别'] = ', '.join(categories)
                
                # 主要技能（取前3个最相关的）
                main_skills = []
                for skill in skills[:3]:
                    main_skills.extend(skill['keywords_matched'][:2])
                chunk.at[idx, '主要技能'] = ', '.join(main_skills[:5])
                
                # 技能水平标注
                level_info = [f"{s['category']}({s['level']})" for s in skills]
                chunk.at[idx, '技能水平标注'] = '; '.join(level_info)
            
            # 整体技能要求等级
            overall_level = determine_overall_skill_level(skills, education, experience)
            chunk.at[idx, '整体技能要求等级'] = overall_level
        
        # 保存结果（第一块写入头，后续追加）
        if first_chunk:
            chunk.to_csv(output_file, index=False, encoding='utf-8-sig')
            first_chunk = False
        else:
            chunk.to_csv(output_file, index=False, mode='a', header=False, encoding='utf-8-sig')
        
        total_processed += len(chunk)
        print(f"已处理 {total_processed} 行")
    
    print(f"\n处理完成！共处理 {total_processed} 行数据")
    print(f"结果已保存至: {output_file}")


def analyze_results(output_file: str):
    """
    分析处理结果的统计信息
    
    Args:
        output_file: 输出CSV文件路径
    """
    print("\n" + "="*60)
    print("分析处理结果...")
    
    # 读取少量数据进行统计
    sample_data = pd.read_csv(output_file, nrows=50000)
    
    print(f"\n样本数据量: {len(sample_data)} 行")
    
    # 技能类别统计
    print("\n技能类别分布 (Top 10):")
    all_categories = []
    for cats in sample_data['ESCO技能类别'].dropna():
        if cats:
            all_categories.extend([c.strip() for c in cats.split(',')])
    
    if all_categories:
        from collections import Counter
        cat_counts = Counter(all_categories)
        for cat, count in cat_counts.most_common(10):
            print(f"  {cat}: {count}")
    
    # 整体技能水平分布
    print("\n整体技能要求等级分布:")
    level_dist = sample_data['整体技能要求等级'].value_counts()
    for level, count in level_dist.items():
        print(f"  {level}: {count} ({count/len(sample_data)*100:.1f}%)")
    
    print("\n前5行示例:")
    print(sample_data[['招聘岗位', 'ESCO技能类别', '整体技能要求等级']].head())
    print("="*60)


if __name__ == "__main__":
    # 使用集中路径配置
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from config.paths import RAW_YEARLY_DIR, ESCO_OUTPUT_DIR

    # 确保输出目录存在
    ESCO_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 文件路径
    input_file = str(RAW_YEARLY_DIR / '智联招聘数据库2016.csv')
    output_file = str(ESCO_OUTPUT_DIR / '2016_esco.csv')
    
    # 处理数据
    process_job_data(input_file, output_file, chunk_size=10000)
    
    # 分析结果
    analyze_results(output_file)
    
    print("\n✓ 全部完成！")
