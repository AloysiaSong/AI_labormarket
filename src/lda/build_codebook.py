#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
技能组合Codebook构建工具
基于LDA结果进行人工标注
"""

import pandas as pd
import json


class SkillCodebookBuilder:
    """技能编码手册构建器"""
    
    def __init__(self, skill_keywords_csv: str, output_dir: str):
        """
        初始化
        
        Args:
            skill_keywords_csv: LDA输出的技能关键词CSV
            output_dir: 输出目录
        """
        self.skill_df = pd.read_csv(skill_keywords_csv, encoding='utf-8-sig')
        self.output_dir = output_dir
        
        print("="*80)
        print("技能组合Codebook构建工具")
        print("="*80)
        print(f"加载了 {len(self.skill_df)} 个技能组合")
    
    def create_excel_template(self):
        """创建Excel编码模板"""
        template_data = []
        
        for idx, row in self.skill_df.iterrows():
            template_data.append({
                'Skill_ID': int(row['skill_id']),
                'LDA关键词_Top10': row['top_10'],
                'LDA关键词_Full30': row['keywords'],
                
                # 人工标注字段
                '技能大类': '',  # 如：技术、营销、管理、财务等
                '技能子类': '',  # 如：编程、数据分析、客户服务等
                '具体技能': '',  # 如：Python开发、SQL分析、客户关系管理等
                
                '技能描述': '',  # 详细描述
                '典型职位': '',  # 典型职位名称
                '行业倾向': '',  # 主要出现的行业
                
                '标注质量': '',  # 高/中/低（关键词清晰度）
                '备注': ''
            })
        
        df = pd.DataFrame(template_data)
        
        output_file = f"{self.output_dir}/skill_codebook_template.xlsx"
        df.to_excel(output_file, index=False, engine='openpyxl')
        
        print(f"✓ Excel模板已创建: {output_file}")
        
        return output_file
    
    def create_json_template(self):
        """创建JSON编码模板"""
        codebook = {
            "metadata": {
                "version": "1.0",
                "date": "2026-01-23",
                "framework": "Occupations as bundles of skills",
                "description": "基于LDA的技能组合分类编码手册",
                "num_skills": len(self.skill_df)
            },
            "skill_bundles": []
        }
        
        for idx, row in self.skill_df.iterrows():
            skill = {
                "skill_id": int(row['skill_id']),
                "lda_keywords": row['keywords'].split(', '),
                "top_10_keywords": row['top_10'].split(', '),
                
                # 人工标注
                "skill_category_l1": "",
                "skill_category_l2": "",
                "skill_category_l3": "",
                "description": "",
                "typical_occupations": [],
                "industry_focus": [],
                "annotation_quality": "",
                "notes": ""
            }
            
            codebook["skill_bundles"].append(skill)
        
        output_file = f"{self.output_dir}/skill_codebook_template.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(codebook, f, ensure_ascii=False, indent=2)
        
        print(f"✓ JSON模板已创建: {output_file}")
        
        return output_file
    
    def create_annotation_guide(self):
        """创建标注指南"""
        guide = """
================================================================================
技能组合人工标注指南
================================================================================

理论基础: Occupations as bundles of skills (Alabdulkareem et al., 2024)

核心概念:
- 每个LDA主题 = 一个潜在的技能组合 (skill bundle)
- 职位描述 = 多个技能组合的混合
- 技能组合可以跨越传统职业边界

================================================================================
标注任务
================================================================================

对每个技能组合（Skill ID），基于LDA关键词，标注以下字段：

1. 技能大类 (Level 1)
   分类体系：
   - 技术研发 (Technical Development)
   - 市场营销 (Marketing & Sales)
   - 运营管理 (Operations Management)
   - 财务会计 (Finance & Accounting)
   - 人力资源 (Human Resources)
   - 行政支持 (Administrative Support)
   - 客户服务 (Customer Service)
   - 生产制造 (Manufacturing)
   - 供应链物流 (Supply Chain & Logistics)
   - 其他 (Other)

2. 技能子类 (Level 2)
   示例：
   - 技术研发 → 软件开发、硬件工程、数据分析、产品设计
   - 市场营销 → 品牌推广、销售管理、市场调研、数字营销
   - 运营管理 → 项目管理、流程优化、质量控制

3. 具体技能 (Level 3)
   示例：
   - 软件开发 → Python开发、Java开发、前端开发、移动开发
   - 数据分析 → SQL分析、数据可视化、统计建模、机器学习

4. 技能描述
   用1-2句话描述该技能组合的核心内容

5. 典型职位
   列举3-5个典型使用该技能组合的职位名称

6. 行业倾向
   该技能组合主要出现在哪些行业

7. 标注质量
   - 高: 关键词清晰，技能组合明确
   - 中: 关键词有一定模糊，但可解释
   - 低: 关键词混杂，难以定义

================================================================================
标注原则
================================================================================

1. 基于关键词，而非主观经验
   - 紧扣LDA提取的关键词
   - 如关键词显示"python, 开发, 代码"，归入编程技能

2. 技能优先，而非职位
   - 关注"做什么"（技能），而非"叫什么"（职位名称）
   - 一个技能组合可对应多个职位

3. 允许跨界组合
   - 技能组合可能跨越传统职业边界
   - 如"数据分析+营销"是合理的技能组合

4. 不确定时建立新类目
   - 如果不能归入已有类目，建立新类目
   - 记录在"备注"中说明原因

================================================================================
质量检查
================================================================================

完成标注后，检查：
1. 是否所有技能组合都已标注？
2. 技能大类分布是否合理？（避免过度集中）
3. 关键词与标注的技能类别是否匹配？
4. 同一技能子类的不同技能组合，区别是否明确？

================================================================================
示例
================================================================================

Skill ID: 5
LDA关键词: python, 开发, 代码, 数据, 分析, 算法, 编程, 软件, java, sql

标注:
- 技能大类: 技术研发
- 技能子类: 软件开发
- 具体技能: 后端开发（Python/Java）
- 技能描述: 使用Python、Java等语言进行后端软件开发，具备数据处理和算法实现能力
- 典型职位: Python开发工程师, Java工程师, 后端开发, 软件工程师
- 行业倾向: 互联网、软件、金融科技
- 标注质量: 高
- 备注: 同时涉及数据处理技能

================================================================================
"""
        
        output_file = f"{self.output_dir}/annotation_guide.txt"
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(guide)
        
        print(f"✓ 标注指南已创建: {output_file}")
        
        return output_file


def main():
    """主函数"""
    print("="*80)
    print("技能组合Codebook构建")
    print("="*80)

    # 使用集中路径配置
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from config.paths import LDA_OUTPUT_DIR

    skill_keywords = str(LDA_OUTPUT_DIR / 'skill_keywords.csv')
    output_dir = str(LDA_OUTPUT_DIR)
    
    builder = SkillCodebookBuilder(skill_keywords, output_dir)
    
    # 创建模板
    print("\n创建标注模板...")
    builder.create_excel_template()
    builder.create_json_template()
    
    # 创建指南
    print("\n创建标注指南...")
    builder.create_annotation_guide()
    
    print("\n" + "="*80)
    print("Codebook模板创建完成！")
    print("="*80)
    print("\n下一步:")
    print("  1. 查看 annotation_guide.txt 了解标注规则")
    print("  2. 打开 skill_codebook_template.xlsx 进行标注")
    print("  3. 完成后运行面板数据分析脚本")


if __name__ == "__main__":
    main()
