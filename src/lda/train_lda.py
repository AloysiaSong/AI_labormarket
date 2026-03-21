#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Task 2.2: 为每个时间窗口训练LDA模型

训练参数说明:
- num_topics=60: 每个窗口60个主题，平衡粒度与对齐复杂度
- passes=15: 迭代次数，确保收敛
- alpha='auto': 自动学习文档-主题分布的稀疏性
- eta='auto': 自动学习主题-词分布的稀疏性
- chunksize=10000: 批量大小，适配大数据

输出:
- {window_name}_lda.model: LDA模型
- {window_name}_topics.txt: 主题词列表
"""
from gensim import corpora
from gensim.models import LdaMulticore
from gensim.models.coherencemodel import CoherenceModel
import pickle
import sys
import logging
from pathlib import Path

# 使用集中路径配置
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config.paths import LDA_OUTPUT_DIR, TOKENIZED_DIR

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# 词典目录
DICT_DIR = LDA_OUTPUT_DIR / "dictionaries"

# 模型输出目录
MODEL_OUTPUT_DIR = LDA_OUTPUT_DIR / "models"
MODEL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# LDA超参数
# 注意：LdaMulticore不支持alpha='auto'，使用'symmetric'代替
LDA_CONFIG = {
    'num_topics': 60,
    'passes': 15,
    'chunksize': 10000,
    'alpha': 'symmetric',  # LdaMulticore不支持'auto'
    'eta': 'auto',
    'iterations': 400,
    'random_state': 42,
    'workers': 4,  # 并行数，根据CPU核心调整
}


def train_window(window_name: str):
    """训练单个时间窗口的LDA模型"""
    print(f"\n{'='*60}")
    print(f"训练 {window_name}")
    print(f"{'='*60}")

    # 检查输入文件
    dict_file = DICT_DIR / f"{window_name}.dict"
    corpus_file = DICT_DIR / f"{window_name}.mm"
    raw_corpus_file = TOKENIZED_DIR / f"{window_name}_corpus.pkl"

    if not dict_file.exists():
        print(f"错误: 找不到词典 {dict_file}")
        return None
    if not corpus_file.exists():
        print(f"错误: 找不到BOW语料 {corpus_file}")
        return None
    if not raw_corpus_file.exists():
        print(f"错误: 找不到原始语料 {raw_corpus_file}")
        return None

    # 加载词典和语料
    print("\n加载数据...")
    dictionary = corpora.Dictionary.load(str(dict_file))
    corpus = corpora.MmCorpus(str(corpus_file))

    print(f"  词汇量: {len(dictionary):,}")
    print(f"  文档数: {len(corpus):,}")

    # 加载原始分词文本（用于Coherence计算）
    print("  加载原始语料（用于Coherence）...")
    with open(raw_corpus_file, 'rb') as f:
        raw_texts = pickle.load(f)

    # 训练LDA
    print(f"\n开始训练LDA...")
    print(f"  参数: num_topics={LDA_CONFIG['num_topics']}, passes={LDA_CONFIG['passes']}")
    print(f"  workers={LDA_CONFIG['workers']}, chunksize={LDA_CONFIG['chunksize']}")

    model = LdaMulticore(
        corpus=list(corpus),
        id2word=dictionary,
        **LDA_CONFIG
    )

    # 计算Coherence
    print("\n计算Topic Coherence...")
    # 使用部分样本加速（100k文档）
    sample_size = min(100000, len(raw_texts))
    sample_texts = raw_texts[:sample_size]

    coherence_model = CoherenceModel(
        model=model,
        texts=sample_texts,
        dictionary=dictionary,
        coherence='c_v'
    )
    coherence = coherence_model.get_coherence()
    print(f"  Coherence (Cv): {coherence:.4f}")

    # 评估结果
    status = "OK" if coherence >= 0.40 else "WARN (< 0.40)"
    print(f"  评估: {status}")

    # 保存模型
    print("\n保存模型...")
    model_file = MODEL_OUTPUT_DIR / f"{window_name}_lda.model"
    model.save(str(model_file))
    print(f"  模型: {model_file}")

    # 保存Top Words
    topics_file = MODEL_OUTPUT_DIR / f"{window_name}_topics.txt"
    with open(topics_file, 'w', encoding='utf-8') as f:
        f.write(f"Window: {window_name}\n")
        f.write(f"Coherence (Cv): {coherence:.4f}\n")
        f.write(f"num_topics: {model.num_topics}\n")
        f.write(f"{'='*60}\n\n")

        for idx in range(model.num_topics):
            top_words = model.show_topic(idx, topn=20)
            words_str = ', '.join([f"{w}({p:.3f})" for w, p in top_words])
            f.write(f"Topic {idx}: {words_str}\n\n")

    print(f"  主题词: {topics_file}")

    return {
        'window': window_name,
        'num_topics': model.num_topics,
        'vocab_size': len(dictionary),
        'num_docs': len(corpus),
        'coherence': coherence
    }


def run():
    print("=" * 60)
    print("Task 2.2: LDA模型训练")
    print("=" * 60)

    print(f"\n输入目录: {DICT_DIR}")
    print(f"输出目录: {MODEL_OUTPUT_DIR}")

    print(f"\nLDA参数:")
    for k, v in LDA_CONFIG.items():
        print(f"  {k}: {v}")

    # 自动检测所有已构建词典的窗口
    dict_files = sorted(DICT_DIR.glob("window_*.dict"))
    
    if not dict_files:
        print("\n❌ 错误: 未找到词典文件")
        print(f"   请检查目录: {DICT_DIR}")
        print("   请先运行 Task 2.1 (build_dictionary.py)")
        return
    
    windows = [f.stem for f in dict_files]
    
    print(f"\n📁 发现 {len(windows)} 个时间窗口:")
    for w in windows:
        print(f"  - {w}")

    # 训练每个窗口
    results = []
    for window in windows:
        result = train_window(window)
        if result:
            results.append(result)

    # 汇总统计
    print("\n" + "=" * 60)
    print("训练结果汇总")
    print("=" * 60)

    print(f"\n{'窗口':<25} {'主题数':>8} {'词汇量':>10} {'文档数':>12} {'Coherence':>12}")
    print("-" * 70)
    for r in results:
        status = "OK" if r['coherence'] >= 0.40 else "WARN"
        print(f"{r['window']:<25} {r['num_topics']:>8} {r['vocab_size']:>10,} {r['num_docs']:>12,} {r['coherence']:>10.4f} [{status}]")

    print(f"\n输出文件位置: {MODEL_OUTPUT_DIR}")

    # 保存汇总统计
    stats_file = MODEL_OUTPUT_DIR / "training_stats.pkl"
    with open(stats_file, 'wb') as f:
        pickle.dump(results, f)
    print(f"统计信息: {stats_file}")


if __name__ == '__main__':
    run()
