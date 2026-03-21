#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
构建岗位综合化指标 (Job Comprehensiveness Index)

两种方法：
1. LDA方法：基于主题熵（Topic Entropy）
2. SBERT方法：基于技能熵（Skill Entropy）

理论基础：
- Shannon Entropy: 衡量概率分布的不确定性
- HHI (Herfindahl-Hirschman Index): 经济学中的集中度指标

输出：
- comprehensiveness_scores.csv: 每年的综合化指数
- correlation_analysis.csv: 双方法相关性验证
"""

import numpy as np
import pandas as pd
from pathlib import Path
from gensim import corpora
from gensim.models import LdaMulticore
import pickle
import sys
import logging
from scipy.stats import pearsonr, spearmanr

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config.paths import LDA_OUTPUT_DIR, TOKENIZED_DIR

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

MODEL_DIR = LDA_OUTPUT_DIR / "models"
OUTPUT_DIR = LDA_OUTPUT_DIR / "analysis"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

NUM_TOPICS = 60


# =============================================================================
# 方法1: LDA 综合化指数
# =============================================================================

def calculate_topic_entropy(doc_topic_distribution):
    """
    计算文档的主题熵（标准化到[0,1]）
    
    参数:
        doc_topic_distribution: (N_docs, N_topics) 数组
    
    返回:
        entropies: (N_docs,) 数组，每个文档的标准化主题熵
        
    解释:
        - 熵 = 0: 完全集中在一个主题（专业化）
        - 熵 = 1: 均匀分布在所有主题（综合化）
    """
    # 避免 log(0)
    prob = doc_topic_distribution + 1e-10
    
    # Shannon熵
    entropy = -np.sum(prob * np.log(prob), axis=1)
    
    # 标准化：除以最大熵 log(N_topics)
    max_entropy = np.log(NUM_TOPICS)
    normalized_entropy = entropy / max_entropy
    
    return normalized_entropy


def calculate_hhi_index(doc_topic_distribution):
    """
    计算 Herfindahl-Hirschman Index (HHI) - 经济学经典指标
    
    返回: 1 - HHI，使得值越大代表越分散（综合化）
    """
    # HHI = sum(s_i^2) where s_i is the share of topic i
    hhi = np.sum(doc_topic_distribution ** 2, axis=1)
    
    # 转换为分散度指标（越高越综合）
    diversification = 1 - hhi
    
    # 标准化到[0,1]
    max_div = 1 - (1 / NUM_TOPICS)  # 完全均匀分布时的HHI
    normalized_div = diversification / max_div
    
    return normalized_div


def compute_lda_comprehensiveness(window_name: str):
    """
    计算指定窗口的LDA综合化指数
    
    返回: DataFrame with columns ['doc_id', 'entropy', 'hhi', 'avg_score']
    """
    logger.info(f"\n📊 计算 {window_name} 的LDA综合化指数...")
    
    # 加载模型和语料
    model_path = MODEL_DIR / f"{window_name}_lda.model"
    corpus_file = LDA_OUTPUT_DIR / "dictionaries" / f"{window_name}.mm"
    
    if not model_path.exists() or not corpus_file.exists():
        logger.warning(f"  ⚠️ 缺少必要文件，跳过 {window_name}")
        return None
    
    model = LdaMulticore.load(str(model_path))
    corpus = corpora.MmCorpus(str(corpus_file))
    
    # 推断文档-主题分布
    logger.info(f"  推断文档-主题分布 (共 {len(corpus)} 个文档)...")
    doc_topics = []
    
    for doc_bow in corpus:
        # 获取主题分布 [(topic_id, prob), ...]
        topic_dist = model.get_document_topics(doc_bow, minimum_probability=0)
        
        # 转换为密集向量
        prob_vec = np.zeros(NUM_TOPICS)
        for topic_id, prob in topic_dist:
            prob_vec[topic_id] = prob
        
        doc_topics.append(prob_vec)
    
    doc_topics = np.array(doc_topics)
    
    # 计算两种指标
    logger.info("  计算主题熵...")
    entropy_scores = calculate_topic_entropy(doc_topics)
    
    logger.info("  计算HHI指数...")
    hhi_scores = calculate_hhi_index(doc_topics)
    
    # 综合分数（两种指标的平均）
    avg_scores = (entropy_scores + hhi_scores) / 2
    
    # 汇总结果
    results = pd.DataFrame({
        'doc_id': range(len(corpus)),
        'entropy': entropy_scores,
        'hhi': hhi_scores,
        'comprehensiveness_lda': avg_scores
    })
    
    logger.info(f"  ✅ 平均综合化指数: {avg_scores.mean():.4f} (std: {avg_scores.std():.4f})")
    
    return results


# =============================================================================
# 方法2: SBERT 综合化指数（与compute_skill_matrix.py配合）
# =============================================================================

def calculate_skill_entropy(skill_similarity_matrix):
    """
    计算文档的技能熵（基于SBERT相似度矩阵）
    
    参数:
        skill_similarity_matrix: (N_docs, N_skills) 数组
    
    返回:
        normalized_entropy: (N_docs,) 数组
    """
    # 归一化为概率分布（行和为1）
    row_sums = skill_similarity_matrix.sum(axis=1, keepdims=True)
    prob = skill_similarity_matrix / (row_sums + 1e-10)
    
    # Shannon熵
    entropy = -np.sum(prob * np.log(prob + 1e-10), axis=1)
    
    # 标准化
    n_skills = skill_similarity_matrix.shape[1]
    max_entropy = np.log(n_skills)
    normalized_entropy = entropy / max_entropy
    
    return normalized_entropy


def compute_sbert_comprehensiveness(sbert_matrix_path: Path):
    """
    从SBERT相似度矩阵计算综合化指数
    
    参数:
        sbert_matrix_path: numpy文件路径 (.npy)，形状 (N_docs, 20)
    
    返回: DataFrame with ['doc_id', 'skill_entropy', 'comprehensiveness_sbert']
    """
    logger.info(f"\n📊 计算SBERT综合化指数...")
    logger.info(f"  加载: {sbert_matrix_path}")
    
    if not sbert_matrix_path.exists():
        logger.warning(f"  ⚠️ 文件不存在")
        return None
    
    # 加载相似度矩阵
    similarity_matrix = np.load(sbert_matrix_path)
    logger.info(f"  矩阵形状: {similarity_matrix.shape}")
    
    # 计算技能熵
    entropy_scores = calculate_skill_entropy(similarity_matrix)
    
    results = pd.DataFrame({
        'doc_id': range(len(similarity_matrix)),
        'skill_entropy': entropy_scores,
        'comprehensiveness_sbert': entropy_scores  # 直接使用熵作为综合化指标
    })
    
    logger.info(f"  ✅ 平均综合化指数: {entropy_scores.mean():.4f} (std: {entropy_scores.std():.4f})")
    
    return results


# =============================================================================
# 主流程：计算所有窗口并进行相关性验证
# =============================================================================

def compute_all_windows_comprehensiveness():
    """计算所有时间窗口的综合化指数"""
    logger.info("="*70)
    logger.info("📊 开始计算综合化指数")
    logger.info("="*70)
    
    # 查找所有已训练的窗口
    all_windows = sorted([
        f.stem.replace("_lda.model", "") 
        for f in MODEL_DIR.glob("*_lda.model")
    ])
    
    logger.info(f"\n发现 {len(all_windows)} 个训练好的窗口:")
    for w in all_windows:
        logger.info(f"  - {w}")
    
    # 逐个窗口计算
    annual_results = {}
    
    for window in all_windows:
        # LDA方法
        lda_results = compute_lda_comprehensiveness(window)
        
        if lda_results is not None:
            # 按年度聚合（取平均值）
            annual_mean = lda_results['comprehensiveness_lda'].mean()
            annual_std = lda_results['comprehensiveness_lda'].std()
            
            annual_results[window] = {
                'lda_mean': annual_mean,
                'lda_std': annual_std,
                'n_docs': len(lda_results)
            }
            
            # 保存详细结果（可选，数据量大时考虑抽样）
            detail_path = OUTPUT_DIR / f"{window}_lda_scores.csv"
            lda_results.to_csv(detail_path, index=False)
            logger.info(f"  💾 详细结果保存至: {detail_path}")
    
    # 汇总为时序DataFrame
    summary_df = pd.DataFrame.from_dict(annual_results, orient='index')
    summary_df.index.name = 'Window'
    
    output_path = OUTPUT_DIR / "comprehensiveness_time_series.csv"
    summary_df.to_csv(output_path)
    
    logger.info("\n" + "="*70)
    logger.info(f"✅ 汇总结果保存至: {output_path}")
    logger.info("="*70)
    
    return summary_df


def validate_with_sbert(lda_summary_path: Path, sbert_fingerprint_path: Path):
    """
    稳健性检验：对比LDA和SBERT方法的相关性
    
    参数:
        lda_summary_path: LDA方法的年度汇总CSV
        sbert_fingerprint_path: SBERT的指纹演化CSV (如ICM_F项目的输出)
    """
    logger.info("\n" + "="*70)
    logger.info("📊 稳健性检验：LDA vs SBERT 相关性分析")
    logger.info("="*70)
    
    # 1. 加载LDA结果
    lda_df = pd.read_csv(lda_summary_path, index_col='Window')
    logger.info(f"\n加载LDA结果: {len(lda_df)} 个窗口")
    
    # 2. 加载SBERT结果（需要先计算SBERT的熵）
    if not sbert_fingerprint_path.exists():
        logger.warning(f"⚠️ SBERT结果不存在: {sbert_fingerprint_path}")
        logger.info("👉 请先运行 TY 版本的 compute_skill_matrix.py")
        return None
    
    sbert_df = pd.read_csv(sbert_fingerprint_path, index_col=0)
    logger.info(f"加载SBERT结果: {len(sbert_df)} 年")
    
    # 计算SBERT的技能熵
    sbert_entropy = []
    for idx, row in sbert_df.iterrows():
        skill_vec = row.values
        prob = skill_vec / skill_vec.sum()
        entropy = -np.sum(prob * np.log(prob + 1e-10))
        normalized = entropy / np.log(len(skill_vec))
        sbert_entropy.append(normalized)
    
    sbert_df['comprehensiveness_sbert'] = sbert_entropy
    
    # 3. 对齐数据（需要将窗口映射到年份）
    # 假设窗口命名为 window_2016_2018 等，提取起始年份
    def extract_year(window_name):
        """从窗口名称提取起始年份"""
        parts = window_name.split('_')
        try:
            return int(parts[1])  # 假设格式为 window_YYYY_YYYY
        except:
            return None
    
    lda_df['year'] = lda_df.index.map(extract_year)
    lda_df = lda_df.dropna(subset=['year'])
    lda_df['year'] = lda_df['year'].astype(int)
    
    # 合并
    merged = lda_df.merge(
        sbert_df[['comprehensiveness_sbert']], 
        left_on='year', 
        right_index=True,
        how='inner'
    )
    
    if len(merged) == 0:
        logger.warning("⚠️ 没有匹配的年份数据")
        return None
    
    logger.info(f"\n✅ 成功对齐 {len(merged)} 个时间点")
    
    # 4. 相关性分析
    lda_scores = merged['lda_mean'].values
    sbert_scores = merged['comprehensiveness_sbert'].values
    
    pearson_r, pearson_p = pearsonr(lda_scores, sbert_scores)
    spearman_r, spearman_p = spearmanr(lda_scores, sbert_scores)
    
    logger.info("\n" + "="*70)
    logger.info("📈 相关性结果:")
    logger.info(f"  Pearson相关系数:  r = {pearson_r:.4f}, p = {pearson_p:.4f}")
    logger.info(f"  Spearman相关系数: ρ = {spearman_r:.4f}, p = {spearman_p:.4f}")
    
    if pearson_r > 0.7:
        logger.info("  ✅ 强相关！两种方法高度一致，可作为稳健性检验。")
    elif pearson_r > 0.5:
        logger.info("  ⚠️ 中等相关，建议检查数据质量或方法参数。")
    else:
        logger.info("  ❌ 相关性较弱，需要进一步分析原因。")
    
    logger.info("="*70)
    
    # 5. 保存对比结果
    comparison_df = merged[['year', 'lda_mean', 'lda_std', 'comprehensiveness_sbert']].copy()
    comparison_df['pearson_r'] = pearson_r
    comparison_df['spearman_r'] = spearman_r
    
    output_path = OUTPUT_DIR / "robustness_check.csv"
    comparison_df.to_csv(output_path, index=False)
    logger.info(f"\n💾 对比结果保存至: {output_path}")
    
    return comparison_df


if __name__ == "__main__":
    # Step 1: 计算所有窗口的LDA综合化指数
    summary_df = compute_all_windows_comprehensiveness()
    
    print("\n" + "="*70)
    print("📊 年度综合化指数汇总:")
    print(summary_df)
    
    # Step 2: 稳健性检验（需要先运行SBERT分析）
    print("\n" + "="*70)
    print("📝 提示：要进行稳健性检验，请：")
    print("  1. 运行 TY 版本的 compute_skill_matrix.py")
    print("  2. 取消下面的注释，运行 validate_with_sbert()")
    print("="*70)
    
    # 示例调用（需要实际路径）
    # lda_path = OUTPUT_DIR / "comprehensiveness_time_series.csv"
    # sbert_path = Path("/path/to/fingerprint_evolution_TY.csv")
    # validate_with_sbert(lda_path, sbert_path)
