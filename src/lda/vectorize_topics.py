#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Task 3.1: 主题向量化

功能：将LDA主题转换为稠密向量用于主题对齐
方法：V_topic = Σ P(word|topic) × V_word

流程：
1. 训练Word2Vec模型（使用所有时间窗口的分词语料）
2. 将每个LDA主题转换为加权平均的词向量
3. 保存主题向量矩阵供后续对齐使用

输入：
- data/processed/tokenized/window_*_corpus.pkl (分词语料)
- output/lda/models/window_*_lda.model (LDA模型)

输出：
- output/lda/word2vec.model (Word2Vec模型)
- output/lda/topic_vectors/window_*_topic_vectors.npy (主题向量)
"""

import numpy as np
from gensim.models import LdaMulticore, Word2Vec
import pickle
import sys
import logging
from pathlib import Path
from tqdm import tqdm

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config.paths import TOKENIZED_DIR, LDA_OUTPUT_DIR

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 输出目录
MODEL_DIR = LDA_OUTPUT_DIR / "models"
VECTOR_OUTPUT_DIR = LDA_OUTPUT_DIR / "topic_vectors"
VECTOR_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Word2Vec 超参数
W2V_CONFIG = {
    'vector_size': 200,      # 向量维度
    'window': 5,             # 上下文窗口
    'min_count': 50,         # 最小词频
    'workers': 4,            # 并行数
    'epochs': 10,            # 迭代次数
    'sg': 0,                 # 0=CBOW, 1=Skip-gram
    'negative': 5,           # 负采样数
    'alpha': 0.025,          # 初始学习率
}


def train_word2vec(all_corpus: list) -> Word2Vec:
    """
    训练Word2Vec模型
    
    Args:
        all_corpus: 所有时间窗口的分词语料
    
    Returns:
        训练好的Word2Vec模型
    """
    logger.info("  开始训练Word2Vec...")
    logger.info(f"  参数: vector_size={W2V_CONFIG['vector_size']}, "
                f"window={W2V_CONFIG['window']}, "
                f"min_count={W2V_CONFIG['min_count']}")
    
    model = Word2Vec(
        sentences=all_corpus,
        **W2V_CONFIG
    )
    
    logger.info(f"  ✅ Word2Vec训练完成")
    logger.info(f"     词汇量: {len(model.wv):,}")
    logger.info(f"     向量维度: {model.vector_size}")
    
    return model


def vectorize_topic(lda_model: LdaMulticore, 
                    topic_id: int,
                    w2v_model: Word2Vec,
                    topn: int = 50) -> np.ndarray:
    """
    将单个LDA主题转换为稠密向量
    
    方法：V_topic = Σ P(word|topic) × V_word
    
    Args:
        lda_model: LDA模型
        topic_id: 主题ID
        w2v_model: Word2Vec模型
        topn: 使用前N个主题词
    
    Returns:
        主题向量 (shape: [vector_size])
    """
    # 获取主题词分布
    topic_words = lda_model.show_topic(topic_id, topn=topn)
    
    # 初始化向量
    vector = np.zeros(w2v_model.vector_size)
    total_prob = 0
    missing_words = 0
    
    # 加权求和
    for word, prob in topic_words:
        if word in w2v_model.wv:
            vector += prob * w2v_model.wv[word]
            total_prob += prob
        else:
            missing_words += 1
    
    # 归一化（如果有有效词向量）
    if total_prob > 0:
        vector = vector / total_prob
    
    # L2归一化（便于后续余弦相似度计算）
    norm = np.linalg.norm(vector)
    if norm > 0:
        vector = vector / norm
    
    # 警告：如果太多词缺失
    if missing_words > topn * 0.3:
        logger.warning(f"  主题 {topic_id}: {missing_words}/{topn} 词缺失Word2Vec向量")
    
    return vector


def vectorize_all_topics(lda_model: LdaMulticore, 
                         w2v_model: Word2Vec) -> np.ndarray:
    """
    向量化所有主题
    
    Args:
        lda_model: LDA模型
        w2v_model: Word2Vec模型
    
    Returns:
        主题向量矩阵 (shape: [num_topics, vector_size])
    """
    vectors = []
    for topic_id in range(lda_model.num_topics):
        vec = vectorize_topic(lda_model, topic_id, w2v_model)
        vectors.append(vec)
    
    return np.array(vectors)


def load_all_corpus() -> list:
    """
    加载所有时间窗口的分词语料
    
    Returns:
        合并后的语料列表
    """
    logger.info("📚 加载所有时间窗口的语料...")
    
    corpus_files = sorted(TOKENIZED_DIR.glob("window_*_corpus.pkl"))
    
    if not corpus_files:
        raise FileNotFoundError(f"未找到语料文件: {TOKENIZED_DIR}/window_*_corpus.pkl")
    
    all_corpus = []
    for corpus_file in corpus_files:
        window_name = corpus_file.stem.replace('_corpus', '')
        logger.info(f"  加载 {window_name}...")
        
        with open(corpus_file, 'rb') as f:
            corpus = pickle.load(f)
        
        logger.info(f"    ✅ {len(corpus):,} 个文档")
        all_corpus.extend(corpus)
    
    logger.info(f"  总计: {len(all_corpus):,} 个文档")
    
    return all_corpus


def process_window(window_name: str, w2v_model: Word2Vec):
    """
    处理单个时间窗口：向量化主题并保存
    
    Args:
        window_name: 窗口名称
        w2v_model: Word2Vec模型
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"向量化 {window_name}")
    logger.info(f"{'='*60}")
    
    # 加载LDA模型
    model_file = MODEL_DIR / f"{window_name}_lda.model"
    
    if not model_file.exists():
        logger.warning(f"  ⚠️ 模型不存在: {model_file}")
        return None
    
    logger.info(f"  加载LDA模型...")
    lda_model = LdaMulticore.load(str(model_file))
    logger.info(f"    主题数: {lda_model.num_topics}")
    
    # 向量化所有主题
    logger.info(f"  向量化主题...")
    vectors = vectorize_all_topics(lda_model, w2v_model)
    logger.info(f"    ✅ 向量矩阵形状: {vectors.shape}")
    
    # 保存
    output_file = VECTOR_OUTPUT_DIR / f"{window_name}_topic_vectors.npy"
    np.save(str(output_file), vectors)
    logger.info(f"  💾 保存至: {output_file.name}")
    
    return {
        'window': window_name,
        'num_topics': lda_model.num_topics,
        'vector_shape': vectors.shape,
        'vector_dim': w2v_model.vector_size
    }


def main():
    """主流程"""
    logger.info("="*70)
    logger.info("Task 3.1: 主题向量化")
    logger.info("="*70)
    
    # Step 1: 加载所有语料
    logger.info("\n[Step 1/3] 加载分词语料")
    try:
        all_corpus = load_all_corpus()
    except FileNotFoundError as e:
        logger.error(f"❌ {e}")
        logger.error("   请先运行 Task 1.3 (tokenize_with_esco.py)")
        return
    
    # Step 2: 训练Word2Vec
    logger.info("\n[Step 2/3] 训练Word2Vec模型")
    w2v_model = train_word2vec(all_corpus)
    
    # 保存Word2Vec模型
    w2v_path = LDA_OUTPUT_DIR / "word2vec.model"
    w2v_model.save(str(w2v_path))
    logger.info(f"  💾 Word2Vec模型保存至: {w2v_path.name}")
    
    # Step 3: 向量化每个窗口的主题
    logger.info("\n[Step 3/3] 向量化各窗口主题")
    
    # 自动检测所有已训练的LDA模型
    model_files = sorted(MODEL_DIR.glob("window_*_lda.model"))
    
    if not model_files:
        logger.error("❌ 未找到LDA模型")
        logger.error(f"   请检查目录: {MODEL_DIR}")
        logger.error("   请先运行 Task 2.2 (train_lda.py)")
        return
    
    windows = [f.stem.replace('_lda', '') for f in model_files]
    
    logger.info(f"\n📁 发现 {len(windows)} 个时间窗口:")
    for w in windows:
        logger.info(f"  - {w}")
    
    # 处理每个窗口
    results = []
    for window in windows:
        result = process_window(window, w2v_model)
        if result:
            results.append(result)
    
    # 汇总结果
    logger.info(f"\n{'='*70}")
    logger.info("向量化完成汇总")
    logger.info(f"{'='*70}")
    
    logger.info(f"\n{'窗口':<25} {'主题数':>8} {'向量维度':>10} {'矩阵形状':>15}")
    logger.info("-" * 62)
    for r in results:
        logger.info(f"{r['window']:<25} {r['num_topics']:>8} "
                   f"{r['vector_dim']:>10} {str(r['vector_shape']):>15}")
    
    logger.info(f"\n输出目录: {VECTOR_OUTPUT_DIR}")
    logger.info(f"Word2Vec模型: {w2v_path}")
    
    # 保存元数据
    metadata = {
        'w2v_config': W2V_CONFIG,
        'w2v_vocab_size': len(w2v_model.wv),
        'windows': results
    }
    
    metadata_file = VECTOR_OUTPUT_DIR / "vectorization_metadata.pkl"
    with open(metadata_file, 'wb') as f:
        pickle.dump(metadata, f)
    
    logger.info(f"元数据: {metadata_file.name}")
    
    logger.info("\n✅ Task 3.1 完成！")
    logger.info("下一步: Task 3.2 混合对齐算法 (align_topics.py)")


if __name__ == "__main__":
    main()
