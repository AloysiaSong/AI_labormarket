#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TY项目：使用SBERT计算技能相似度矩阵

基于 ICM_F 项目的 compute_skill_matrix.py 修改
适配 TY 项目的数据结构和时间窗口

输入: data/processed/tokenized/window_YYYY_YYYY_tokenized.csv
输出: output/sbert/fingerprint_evolution_TY.csv
"""

import pandas as pd
import numpy as np
import os
from pathlib import Path
import torch
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import sys

# 路径配置
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config.paths import TOKENIZED_DIR

# 输出目录
OUTPUT_DIR = Path(__file__).parent.parent.parent / "output" / "sbert"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# =================核心技能本体（20维）=================
keywords_dict = {
    '1.1_沟通技能': '沟通 表达 汇报 演讲 谈判 Communication Presentation Negotiation Public Speaking Interpersonal Skills',
    '1.2_团队协作': '团队协作 合作 配合 团队精神 协调 互助 协同 团队意识 共事 Teamwork Collaboration Cooperation Team Player',
    '1.3_创造性思维': '创新 创意 脑洞 设计思维 独创 从0到1 革新 想象力 发散思维 Creativity Innovation Brainstorming Design Thinking Out of the box',
    '2.1_信息处理': '信息检索 资料收集 调研 归纳整理 查阅 搜索 文献 筛选 记录 Information Retrieval Research Data Gathering Documentation',
    '2.2_数字素养': '办公软件 Office Word PPT PowerPoint 计算机基础 互联网 邮件 打印 信息化 Computer Literacy Microsoft Office Email Digital Skills',
    '2.3_数据分析': '数据分析 统计 报表 可视化 数据挖掘 SQL SPSS Excel函数 透视表 BI Tableau Matlab Data Analysis Statistics Data Visualization Reporting Metrics',
    '3.1_母语能力': '文案 撰写 编辑 校对 公文 语文 写作 稿件 文字功底 笔杆子 Writing Editing Copywriting Drafting',
    '3.2_外语能力': '英语 外语 翻译 口语 CET IELTS TOEFL 托福 雅思 专八 听说读写 Japanese French English Translation Foreign Language Bilingual',
    '4.1_项目管理': '项目管理 进度控制 统筹 PMP 敏捷 规划 排期 里程碑 甘特图 Scrum Project Management Agile Scheduling Planning Milestones',
    '4.2_人员管理': '领导力 团队管理 带人 培训 招聘 考核 绩效 激励 督导 人才培养 Leadership Team Management Mentoring Recruitment Performance Management',
    '4.3_资源管理': '成本控制 预算 采购 资产管理 物料 库存 供应链 供应商 费用 Resource Management Budgeting Cost Control Inventory Procurement',
    '5.1_体力要求': '体力 搬运 负重 久站 户外 身体健康 吃苦耐劳 夜班 三班倒 Physical Strength Stamina Manual Labor Lifting Outdoor Work',
    '5.2_精细动作技能': '精细 手工 绘图 仪器操作 维修 焊接 装配 驾驶 制图 实操 Fine Motor Skills Dexterity Precision Assembly Operation',
    '6.1_自主学习': '自驱力 学习能力 好奇心 主动 钻研 自我提升 快速学习 求知欲 Self-learning Proactive Fast Learner Curiosity Self-motivated',
    '6.2_适应能力': '适应能力 抗压 灵活性 应变 接受出差 拥抱变化 多任务 快节奏 弹性 Adaptability Flexibility Resilience Stress Management Change Management',
    '7.1_分析思维': '逻辑 推理 问题解决 批判性思维 拆解 溯源 条理 分析 因果 Analytical Thinking Logical Problem Solving Critical Thinking Reasoning',
    '7.2_决策能力': '决策 判断 战略 执行力 拍板 大局观 策略 规划 权衡 Decision Making Strategic Planning Judgment Execution',
    '8.1_数字技术使用': '编程 开发 代码 架构 运维 测试 Java C++ Python Go Linux Cloud AWS Docker Kubernetes 算法 后端 前端 Programming Coding Software Development Technical Skills Engineering',
    '8.2_软件应用': '熟练使用 PS Photoshop AI Illustrator CAD AutoCAD 3DMax Revit SAP ERP CRM Salesforce Premiere Figma Sketch Software Proficiency Adobe Creative Suite Design Tools',
    '9.1_GenAI': '生成式 大模型 LLM AIGC ChatGPT GPT 文心一言 提示词 Prompt Copilot Midjourney Stable Diffusion Transformer RAG LangChain HuggingFace 预训练 Fine-tuning 微调 AI绘画 AI写作 Claude Gemini Generative AI Large Language Model Prompt Engineering'
}

GENAI_NAME = '9.1_GenAI'


def detect_hardware():
    """自动检测硬件加速"""
    if torch.backends.mps.is_available():
        device = "mps"
        print("🚀 检测到 Apple Silicon (M1/M2/M3)，已开启 MPS 硬件加速！")
    elif torch.cuda.is_available():
        device = "cuda"
        print("🚀 检测到 NVIDIA GPU，已开启 CUDA 加速！")
    else:
        device = "cpu"
        print("🐢 未检测到 GPU，正在使用 CPU (速度较慢)...")
    return device


def load_window_data(window_name: str):
    """加载指定窗口的数据"""
    csv_file = TOKENIZED_DIR / f"{window_name}_tokenized.csv"
    
    if not csv_file.exists():
        print(f"⚠️ 文件不存在: {csv_file}")
        return None
    
    print(f"  加载: {csv_file}")
    df = pd.read_csv(csv_file)
    
    # 尝试找到文本列（根据TY项目的实际列名调整）
    text_col = None
    
    # 优先查找这些可能的列名（TY项目实际列名）
    possible_cols = [
        'tokenized',              # TY项目分词后的列 ⭐
        'cleaned_requirements',   # TY项目清洗后的列
        'requirement', 
        'requirements', 
        '任职要求', 
        'jd_text', 
        'description', 
        'content', 
        'text'
    ]
    
    for col in possible_cols:
        if col in df.columns:
            text_col = col
            print(f"    ✅ 找到文本列: {text_col}")
            break
    
    if text_col is None:
        # 检查所有列名，排除明显不是JD文本的列
        exclude_cols = ['企业名称', '招聘岗位', '公司', '薪资', '地址', 'id', 'ID', '日期', 'date', 
                        '工作城市', '工作区域', '学历要求', '要求经验', '来源平台']
        object_cols = df.select_dtypes(include=['object']).columns
        
        for col in object_cols:
            if not any(excl in str(col) for excl in exclude_cols):
                text_col = col
                print(f"    ⚠️ 自动选择列: {text_col}")
                break
        
        if text_col is None:
            print(f"    ❌ 错误：无法找到合适的文本列！")
            print(f"    可用列: {list(df.columns)}")
            return None
    
    # 清洗空值
    df[text_col] = df[text_col].fillna("")
    texts = df[text_col].tolist()
    
    print(f"    ✅ 加载 {len(texts)} 条记录")
    return texts


def compute_window_fingerprint(model, skill_embeddings, skill_names, window_name: str):
    """计算单个窗口的技能指纹"""
    print(f"\n{'='*70}")
    print(f"处理窗口: {window_name}")
    print(f"{'='*70}")
    
    # 1. 加载数据
    texts = load_window_data(window_name)
    if texts is None or len(texts) == 0:
        return None
    
    # 2. 批量编码
    print(f"  🔄 编码 {len(texts)} 条文本...")
    text_embeddings = model.encode(
        texts,
        batch_size=128,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True
    )
    
    # 3. 计算相似度矩阵 (N_docs × 20_skills)
    print(f"  🧮 计算相似度矩阵...")
    similarity_matrix = cosine_similarity(text_embeddings, skill_embeddings)
    
    # 4. GenAI 关键词门控
    genai_idx = skill_names.index(GENAI_NAME)
    genai_keywords = [k.lower() for k in keywords_dict[GENAI_NAME].split()]
    
    lower_texts = [t.lower() for t in texts]
    genai_counts = [
        sum(1 for kw in genai_keywords if kw in text)
        for text in lower_texts
    ]
    genai_weights = np.array(
        [0.0 if c == 0 else 0.3 if c == 1 else 0.6 if c == 2 else 1.0 for c in genai_counts],
        dtype=np.float32
    )
    similarity_matrix[:, genai_idx] *= genai_weights
    
    # 5. 聚合为窗口指纹（平均值）
    window_fingerprint = similarity_matrix.mean(axis=0)
    
    print(f"  ✅ 指纹计算完成")
    print(f"      前5个技能得分: {window_fingerprint[:5]}")
    
    # 可选：保存详细矩阵（用于方差分析）
    detail_path = OUTPUT_DIR / f"detail_{window_name}.npy"
    np.save(detail_path, similarity_matrix)
    print(f"  💾 详细矩阵保存至: {detail_path}")
    
    return window_fingerprint


def main():
    print("="*70)
    print("🚀 TY项目 - SBERT技能指纹分析")
    print("="*70)
    
    # 1. 检测硬件
    device = detect_hardware()
    
    # 2. 加载模型
    print("\n📦 加载NLP模型...")
    model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2', device=device)
    
    # 3. 编码技能本体（只做一次）
    print("\n🧬 编码技能本体（20维）...")
    skill_names = list(keywords_dict.keys())
    skill_texts = list(keywords_dict.values())
    skill_embeddings = model.encode(
        skill_texts, 
        convert_to_numpy=True, 
        normalize_embeddings=True
    )
    print(f"  ✅ 技能向量形状: {skill_embeddings.shape}")
    
    # 4. 查找所有时间窗口
    all_windows = sorted([
        f.stem.replace("_tokenized", "") 
        for f in TOKENIZED_DIR.glob("window_*_tokenized.csv")
    ])
    
    print(f"\n📁 发现 {len(all_windows)} 个时间窗口:")
    for w in all_windows:
        print(f"    - {w}")
    
    # 5. 逐个窗口计算
    fingerprints = {}
    
    for window in all_windows:
        fp = compute_window_fingerprint(model, skill_embeddings, skill_names, window)
        if fp is not None:
            fingerprints[window] = fp
    
    # 6. 汇总为DataFrame
    df_fingerprints = pd.DataFrame.from_dict(
        fingerprints, 
        orient='index', 
        columns=skill_names
    )
    df_fingerprints.index.name = 'Window'
    
    # 7. 保存结果
    output_path = OUTPUT_DIR / "fingerprint_evolution_TY.csv"
    df_fingerprints.to_csv(output_path)
    
    print("\n" + "="*70)
    print(f"✅ 所有窗口处理完成！")
    print(f"💾 结果保存至: {output_path}")
    print("="*70)
    
    print("\n📊 指纹矩阵预览:")
    print(df_fingerprints)
    
    print("\n🔍 技能演化趋势（前5个技能）:")
    print(df_fingerprints.iloc[:, :5])


if __name__ == "__main__":
    main()
