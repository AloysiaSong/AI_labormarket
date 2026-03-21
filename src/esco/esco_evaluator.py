#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ESCO技能评分系统 - 方案C（纯中文关键词匹配）
由于esco-skill-extractor不支持中文，改用中文关键词匹配方案
"""

import pandas as pd
import numpy as np
from typing import Dict, List

# 22个ESCO能力维度的中文关键词映射
SKILL_MAPPING = {
    # 1. Communication, collaboration and creativity
    '1.1_沟通技能': {
        'keywords': ['沟通', '交流', '表达', '汇报', '演讲', '对接', '协调', '联络', '洽谈', '接待', '客户', '谈判', '说服', '沟通能力'],
        'levels': {
            1: ['协助沟通', '配合', '转达', '记录', '基本沟通'],
            2: ['接待', '接听', '回复', '日常沟通', '电话沟通'],
            3: ['独立沟通', '负责对接', '协调沟通', '需求分析', '跨部门'],
            4: ['谈判', '商务洽谈', '关系维护', '影响力', '客户关系'],
            5: ['高层对接', '战略合作', '重大项目谈判', '危机公关', '战略沟通']
        }
    },
    '1.2_团队协作': {
        'keywords': ['团队', '协作', '配合', '合作', '团队合作', '协同', '集体', '团队精神'],
        'levels': {
            1: ['配合', '辅助', '服从', '协助团队'],
            2: ['团队成员', '参与', '协助团队', '团队工作'],
            3: ['团队协作', '协同工作', '跨部门', '团队配合'],
            4: ['带领团队', '小组长', '团队管理', '团队带头'],
            5: ['团队建设', '团队领导', '多团队协调', '团队战略']
        }
    },
    '1.3_创造性思维': {
        'keywords': ['创新', '创意', '设计', '策划', '方案', '研发', '开拓', '改进', '优化', '创造'],
        'levels': {
            1: ['参与', '建议', '改进意见', '参与创新'],
            2: ['独立设计', '方案优化', '创意执行', '设计方案'],
            3: ['创意策划', '产品设计', '流程改进', '创新方案'],
            4: ['创新项目', '战略规划', '研发创新', '创新管理'],
            5: ['行业领先', '原创', '颠覆性', '首创', '引领创新']
        }
    },
    
    # 2. Information skills
    '2.1_信息处理': {
        'keywords': ['信息', '资料', '数据', '文档', '档案', '资料整理', '信息收集', '信息管理'],
        'levels': {
            1: ['记录', '登记', '归档', '整理', '录入'],
            2: ['收集', '汇总', '统计', '整理资料'],
            3: ['筛选', '分类', '管理', '维护', '信息管理'],
            4: ['分析', '挖掘', '信息系统', '信息分析'],
            5: ['信息战略', '大数据', '信息架构', '数据治理']
        }
    },
    '2.2_数字素养': {
        'keywords': ['数字化', '在线', '互联网', '网络', '电子', '数字工具', '线上', '数字平台'],
        'levels': {
            1: ['使用电脑', '上网', '邮件', '基本'],
            2: ['在线办公', '网络平台', '电子商务', '线上'],
            3: ['数字化流程', '线上协作', '数字营销', '数字工具'],
            4: ['数字化转型', '数字化管理', '智能化', '数字创新'],
            5: ['数字战略', '数字化创新', '数字生态', '数字领导']
        }
    },
    '2.3_数据分析': {
        'keywords': ['数据', '分析', '统计', '报表', '指标', '数据分析', '数据挖掘', 'Excel'],
        'levels': {
            1: ['数据录入', '记录', '简单统计', '录入'],
            2: ['报表', '汇总', '图表', '数据汇总'],
            3: ['数据分析', '业务分析', '趋势分析', '分析能力'],
            4: ['建模', '预测', '算法', '高级分析'],
            5: ['数据战略', '大数据', '数据科学', '数据专家']
        }
    },
    
    # 3. Language skills
    '3.1_母语能力': {
        'keywords': ['中文', '普通话', '口齿清晰', '表达清晰', '文字', '写作', '表达能力'],
        'levels': {
            1: ['基本表达', '简单沟通', '基本'],
            2: ['口齿清晰', '表达清晰', '普通话流利', '良好表达'],
            3: ['文字功底', '写作能力', '专业表达', '文案'],
            4: ['公文写作', '演讲', '专业文案', '高级写作'],
            5: ['编辑', '文字工作者', '作家', '写作专家']
        }
    },
    '3.2_外语能力': {
        'keywords': ['英语', '日语', '韩语', '法语', '德语', '外语', '翻译', '口译', '英文'],
        'levels': {
            1: ['英语基础', '简单英语', '外语一般', '基础'],
            2: ['读写', '邮件', '外语良好', '英语读写'],
            3: ['流利', '听说读写', '业务交流', '英语流利'],
            4: ['翻译', '商务谈判', '专业外语', '英语精通'],
            5: ['精通', '母语', '同声传译', '外语专家']
        }
    },
    
    # 4. Management skills
    '4.1_项目管理': {
        'keywords': ['项目', '项目管理', '项目执行', '项目跟进', '项目协调', '项目实施'],
        'levels': {
            1: ['协助项目', '项目成员', '参与项目', '项目协助'],
            2: ['项目跟进', '执行项目', '项目对接', '项目执行'],
            3: ['项目管理', '项目负责人', '独立项目', '项目管理'],
            4: ['项目经理', '多项目', '项目统筹', '项目管理经验'],
            5: ['项目总监', '重大项目', '战略项目', '项目管理专家']
        }
    },
    '4.2_人员管理': {
        'keywords': ['管理', '团队管理', '下属', '培训', '指导', '考核', '招聘', 'HR', '人员'],
        'levels': {
            1: ['辅助管理', '指导新人', '协助管理'],
            2: ['小组长', '团队管理', '下属', '小组管理'],
            3: ['部门管理', '团队建设', '人员培训', '管理经验'],
            4: ['经理', '多部门', '高级管理', '管理团队'],
            5: ['总监', '高层', '战略管理', '高级管理']
        }
    },
    '4.3_资源管理': {
        'keywords': ['资源', '预算', '成本', '采购', '供应链', '库存', '资源调配', '资源配置'],
        'levels': {
            1: ['物资', '用品', '设备使用', '物资管理'],
            2: ['采购', '协调资源', '库存管理', '资源协调'],
            3: ['资源配置', '预算', '成本控制', '预算管理'],
            4: ['资源优化', '供应链', '成本优化', '供应链管理'],
            5: ['资源战略', '战略配置', '资源整合', '战略资源']
        }
    },
    
    # 5. Physical abilities
    '5.1_体力要求': {
        'keywords': ['体力', '搬运', '装卸', '站立', '行走', '驾驶', '外勤', '出差'],
        'levels': {
            1: ['基本活动', '偶尔站立', '基本'],
            2: ['站立工作', '经常走动', '驾驶', '站立'],
            3: ['搬运', '体力工作', '外勤', '中等体力'],
            4: ['重体力', '装卸', '体力劳动', '搬运重物'],
            5: ['高强度', '特殊体能', '极限体力', '极高体力']
        }
    },
    '5.2_精细动作技能': {
        'keywords': ['操作', '手工', '精细', '装配', '维修', '制作', '手工制作', '精密'],
        'levels': {
            1: ['基本操作', '简单制作', '简单操作'],
            2: ['熟练操作', '手工制作', '操作熟练'],
            3: ['精细操作', '精密装配', '精细加工'],
            4: ['高精度', '专业操作', '精密制作', '精密操作'],
            5: ['工匠', '极致', '大师级', '工匠级']
        }
    },
    
    # 6. Learning to learn
    '6.1_自主学习': {
        'keywords': ['学习', '学习能力', '自学', '培训', '进修', '提升', '成长', '学习力'],
        'levels': {
            1: ['接受培训', '参加学习', '培训'],
            2: ['学习能力', '主动学习', '快速学习', '学习能力强'],
            3: ['自学', '持续学习', '学习能力强', '自主学习'],
            4: ['深度学习', '专业提升', '研究学习', '深入学习'],
            5: ['终身学习', '学习专家', '培训师', '学习大师']
        }
    },
    '6.2_适应能力': {
        'keywords': ['适应', '灵活', '应变', '变化', '压力', '多任务', '调整', '抗压'],
        'levels': {
            1: ['基本适应', '简单调整', '适应'],
            2: ['快速适应', '灵活', '应变', '灵活应变'],
            3: ['多任务', '抗压', '承压能力', '压力适应'],
            4: ['复杂环境', '快速转换', '高压', '高压适应'],
            5: ['危机应对', '极限挑战', '高度灵活', '极限适应']
        }
    },
    
    # 7. Problem solving
    '7.1_分析思维': {
        'keywords': ['分析', '思维', '逻辑', '判断', '推理', '思考', '洞察', '分析能力'],
        'levels': {
            1: ['基本判断', '简单分析', '基本'],
            2: ['逻辑思维', '分析能力', '思维清晰', '逻辑'],
            3: ['系统分析', '深度分析', '洞察力', '系统思维'],
            4: ['战略分析', '复杂分析', '高级分析', '战略思维'],
            5: ['专家级', '行业洞察', '战略思维', '分析专家']
        }
    },
    '7.2_决策能力': {
        'keywords': ['决策', '判断', '决定', '选择', '方案', '解决问题', '问题解决'],
        'levels': {
            1: ['执行决策', '参与', '参与决策'],
            2: ['解决问题', '日常决策', '判断', '问题解决'],
            3: ['独立决策', '方案选择', '决定', '独立判断'],
            4: ['重要决策', '关键决策', '战术', '战术决策'],
            5: ['战略决策', '高层决策', '重大决策', '战略判断']
        }
    },
    
    # 8. Digital competences
    '8.1_数字技术使用': {
        'keywords': ['软件', '系统', '技术', '工具', '平台', '应用', 'IT', '信息技术'],
        'levels': {
            1: ['基本软件', '简单操作', '基本'],
            2: ['熟练软件', '常用工具', '软件熟练'],
            3: ['专业软件', '多种工具', '技术应用', '技术熟练'],
            4: ['开发', '系统管理', '技术实施', '技术开发'],
            5: ['技术专家', '架构', '技术领导', '技术架构']
        }
    },
    '8.2_软件应用': {
        'keywords': ['Office', 'Excel', 'Word', 'PPT', 'PS', 'CAD', '3D', '软件', '编程', 'Python', 'Java'],
        'levels': {
            1: ['会使用Office', '基本操作', '会使用'],
            2: ['熟练Office', 'Excel', 'Word', 'PPT', '熟练'],
            3: ['PS', 'CAD', '专业软件', '熟练掌握', '精通Office'],
            4: ['编程', '高级应用', '多软件精通', '开发'],
            5: ['软件开发', '系统设计', '架构', '软件专家']
        }
    }
}

# 学历调整因子
EDUCATION_FACTORS = {
    '不限': 0,
    '中专': -0.3,
    '高中': -0.3,
    '大专': 0,
    '本科': 0.3,
    '硕士': 0.6,
    '博士': 0.8,
    'MBA': 0.6
}


class ChineseESCOEvaluator:
    """基于中文关键词的ESCO技能评估器"""
    
    def __init__(self):
        self.skill_mapping = SKILL_MAPPING
        self.education_factors = EDUCATION_FACTORS
    
    def evaluate_job(self, description: str, education: str) -> Dict[str, int]:
        """
        评估单个职位的所有技能维度
        
        Args:
            description: 职位描述
            education: 学历要求
            
        Returns:
            22个能力维度的评分字典 (0-5分)
        """
        if pd.isna(description):
            description = ""
        if pd.isna(education):
            education = "不限"
        
        description = str(description).lower()
        results = {}
        
        for skill_name, skill_info in self.skill_mapping.items():
            score = self._evaluate_single_skill(description, education, skill_info)
            results[skill_name] = score
        
        return results
    
    def _evaluate_single_skill(self, desc: str, edu: str, skill_info: dict) -> int:
        """评估单个技能维度"""
        
        # 步骤1: 判断是否涉及该能力
        keywords = skill_info['keywords']
        if not any(kw in desc for kw in keywords):
            return 0  # 不涉及
        
        # 步骤2: 判定初始等级（从高到低匹配）
        initial_level = 0
        for level in [5, 4, 3, 2, 1]:
            level_keywords = skill_info['levels'].get(level, [])
            if any(kw in desc for kw in level_keywords):
                initial_level = level
                break
        
        # 如果有关键词但没有匹配到等级关键词，默认为2级
        if initial_level == 0:
            initial_level = 2
        
        # 步骤3: 学历调整
        edu_factor = self.education_factors.get(edu, 0)
        final_level = initial_level + edu_factor
        
        # 步骤4: 限制在1-5范围内
        final_level = max(1, min(5, round(final_level)))
        
        return int(final_level)


def process_job_data(input_file: str, output_file: str, chunk_size: int = 5000):
    """
    处理招聘数据，添加22个ESCO技能维度评分
    
    Args:
        input_file: 输入CSV文件路径
        output_file: 输出CSV文件路径
        chunk_size: 每次处理的行数
    """
    print(f"开始处理文件: {input_file}")
    print(f"使用方案C: 纯中文关键词匹配")
    print(f"分块大小: {chunk_size} 行")
    print("="*60)
    
    evaluator = ChineseESCOEvaluator()
    
    first_chunk = True
    total_processed = 0
    
    # 分块读取处理大文件
    for chunk_id, chunk in enumerate(pd.read_csv(input_file, chunksize=chunk_size, low_memory=False)):
        print(f"\n处理第 {chunk_id + 1} 块数据，共 {len(chunk)} 行...")
        
        # 初始化22个能力维度列
        skill_columns = list(SKILL_MAPPING.keys())
        for col in skill_columns:
            chunk[col] = 0
        
        # 逐行评估
        for idx, row in chunk.iterrows():
            description = row.get('职位描述', '')
            education = row.get('学历要求', '不限')
            
            # 评估所有能力维度
            scores = evaluator.evaluate_job(description, education)
            
            # 写入评分
            for skill_name, score in scores.items():
                chunk.at[idx, skill_name] = score
        
        # 保存结果
        if first_chunk:
            chunk.to_csv(output_file, index=False, encoding='utf-8-sig')
            first_chunk = False
        else:
            chunk.to_csv(output_file, index=False, mode='a', header=False, encoding='utf-8-sig')
        
        total_processed += len(chunk)
        print(f"已处理 {total_processed} 行")
        
        # 显示当前块的统计
        if chunk_id == 0:
            print("\n前10行评分示例:")
            sample_cols = ['招聘岗位'] + skill_columns[:5]  # 显示前5个能力维度
            if '招聘岗位' in chunk.columns:
                print(chunk[sample_cols].head(10).to_string())
    
    print(f"\n" + "="*60)
    print(f"✓ 处理完成！共处理 {total_processed} 行数据")
    print(f"✓ 结果已保存至: {output_file}")
    
    # 简单统计
    print("\n读取结果进行统计分析...")
    result_sample = pd.read_csv(output_file, nrows=10000)
    
    print(f"\n样本统计 (前10000行):")
    print(f"  总行数: {len(result_sample)}")
    
    skill_columns = [col for col in result_sample.columns if col.startswith(('1.', '2.', '3.', '4.', '5.', '6.', '7.', '8.'))]
    
    print(f"\n  各能力维度平均分:")
    for col in skill_columns[:10]:  # 显示前10个维度
        avg_score = result_sample[col].mean()
        zero_ratio = (result_sample[col] == 0).sum() / len(result_sample) * 100
        print(f"    {col}: {avg_score:.2f} (0分占比: {zero_ratio:.1f}%)")
    
    print("\n" + "="*60)


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
    output_file = str(ESCO_OUTPUT_DIR / '2016_esco_A.csv')
    
    # 处理数据
    process_job_data(input_file, output_file, chunk_size=5000)
    
    print("\n✓ 全部完成！")
