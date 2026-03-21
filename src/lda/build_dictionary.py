#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Task 2.1: 构建gensim词典与BOW语料

对每个时间窗口的分词语料:
1. 构建gensim Dictionary
2. 应用词频过滤（去除极端词频）
3. 生成BOW (Bag of Words) 语料
4. 过滤学历/经验等门槛词（在词典阶段）

输出:
- {window_name}.dict: gensim词典
- {window_name}.mm: BOW语料（Matrix Market格式）
"""
from gensim import corpora
import pickle
import sys
from pathlib import Path
from tqdm import tqdm
import re

# 使用集中路径配置
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config.paths import TOKENIZED_DIR, LDA_OUTPUT_DIR

# 输出目录
DICT_OUTPUT_DIR = LDA_OUTPUT_DIR / "dictionaries"
DICT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 过滤参数（针对百万级数据调优）
MIN_DF = 100      # 最小文档频率（过滤拼写错误/极罕见词）
MAX_DF = 0.4      # 最大文档频率比例（过滤通用词）
KEEP_N = 30000    # 保留词汇上限

# 学历/经验等门槛词过滤（词典阶段）
GATEKEEPING_TERMS = set([
    # 学历/学位
    "学历", "学位", "学士", "硕士", "博士", "博士后", "研究生", "本科", "大专", "中专", "高中", "初中", "小学",
    "专科", "本科及以上", "大专及以上", "硕士及以上", "博士及以上", "应届", "应届生",
    "MBA", "EMBA",
    # 经验/年限
    "经验", "工作经验", "行业经验", "相关经验", "年限", "工作年限",
])

GATEKEEPING_PATTERNS = [
    re.compile(r'^\d+年$'),
    re.compile(r'^\d+年经验$'),
    re.compile(r'^\d+年左右$'),
    re.compile(r'^\d+年以上$'),
    re.compile(r'^\d+年以下$'),
    re.compile(r'^\d+[-~]\d+年$'),
]


def is_gatekeeping_token(token: str) -> bool:
    t = token.strip()
    if not t:
        return False
    if t in GATEKEEPING_TERMS:
        return True
    return any(p.match(t) for p in GATEKEEPING_PATTERNS)


def filter_gatekeeping(corpus):
    for doc in corpus:
        yield [t for t in doc if not is_gatekeeping_token(t)]


def process_window(window_name: str):
    """处理单个时间窗口"""
    print(f"\n{'='*60}")
    print(f"处理 {window_name}")
    print(f"{'='*60}")

    # 加载分词后语料
    corpus_file = TOKENIZED_DIR / f"{window_name}_corpus.pkl"
    if not corpus_file.exists():
        print(f"错误: 找不到语料文件 {corpus_file}")
        return None

    print(f"加载语料: {corpus_file}")
    with open(corpus_file, 'rb') as f:
        corpus = pickle.load(f)

    print(f"文档数: {len(corpus):,}")

    # Step 1: 构建词典（先过滤门槛词）
    print("\nStep 1: 构建词典...")
    dictionary = corpora.Dictionary(filter_gatekeeping(corpus))
    original_vocab = len(dictionary)
    print(f"  原始词汇量: {original_vocab:,}")

    # Step 2: 过滤极端词频
    print(f"\nStep 2: 应用词频过滤...")
    print(f"  参数: min_df={MIN_DF}, max_df={MAX_DF}, keep_n={KEEP_N}")
    dictionary.filter_extremes(
        no_below=MIN_DF,
        no_above=MAX_DF,
        keep_n=KEEP_N
    )
    filtered_vocab = len(dictionary)
    print(f"  过滤后词汇量: {filtered_vocab:,} (保留 {filtered_vocab/original_vocab*100:.1f}%)")

    if filtered_vocab == 0:
        print("❌ 过滤后词典为空，请降低 MIN_DF 或 MAX_DF")
        return None

    # 压缩词典ID（去除间隙）
    dictionary.compactify()

    # Step 3: 构建BOW语料
    print("\nStep 3: 构建BOW语料...")
    stats_acc = {"total_tokens": 0, "non_empty": 0}

    def bow_generator():
        for doc in tqdm(corpus, desc="生成BOW"):
            bow = dictionary.doc2bow(doc)
            if bow:
                stats_acc["non_empty"] += 1
                stats_acc["total_tokens"] += sum(cnt for _, cnt in bow)
            yield bow

    # Step 4: 保存
    print("\nStep 4: 保存...")
    dict_file = DICT_OUTPUT_DIR / f"{window_name}.dict"
    dictionary.save(str(dict_file))
    print(f"  词典: {dict_file}")

    corpus_file = DICT_OUTPUT_DIR / f"{window_name}.mm"
    corpora.MmCorpus.serialize(str(corpus_file), bow_generator())
    print(f"  BOW语料: {corpus_file}")

    total_tokens = stats_acc["total_tokens"]
    non_empty = stats_acc["non_empty"]
    avg_tokens = total_tokens / non_empty if non_empty > 0 else 0

    print(f"\n统计:")
    print(f"  非空文档: {non_empty:,} / {len(corpus):,}")
    print(f"  总词数: {total_tokens:,}")
    print(f"  平均词数/文档: {avg_tokens:.1f}")

    # 保存词典统计信息
    stats = {
        'window': window_name,
        'original_vocab': original_vocab,
        'filtered_vocab': filtered_vocab,
        'num_docs': len(corpus),
        'non_empty_docs': non_empty,
        'total_tokens': total_tokens,
        'avg_tokens_per_doc': avg_tokens,
        'filter_params': {
            'min_df': MIN_DF,
            'max_df': MAX_DF,
            'keep_n': KEEP_N
        }
    }

    return stats


def run():
    print("=" * 60)
    print("Task 2.1: 构建词典与BOW语料")
    print("=" * 60)

    print(f"\n输入目录: {TOKENIZED_DIR}")
    print(f"输出目录: {DICT_OUTPUT_DIR}")

    # 自动检测所有已分词的窗口
    corpus_files = sorted(
        f for f in TOKENIZED_DIR.glob("window_*_corpus.pkl")
        if re.match(r'^window_\d{4}_\d{4}_corpus\.pkl$', f.name)
    )
    
    if not corpus_files:
        print("\n❌ 错误: 未找到分词语料文件")
        print(f"   请检查目录: {TOKENIZED_DIR}")
        print("   应包含 window_*_corpus.pkl 文件")
        return
    
    windows = [f.stem.replace('_corpus', '') for f in corpus_files]
    
    print(f"\n📁 发现 {len(windows)} 个已分词窗口:")
    for w in windows:
        print(f"  - {w}")

    # 处理每个窗口
    all_stats = []
    for window in windows:
        stats = process_window(window)
        if stats:
            all_stats.append(stats)

    # 汇总统计
    print("\n" + "=" * 60)
    print("词典构建完成！")
    print("=" * 60)

    print("\n汇总统计:")
    print(f"{'窗口':<25} {'原始词汇':>12} {'过滤后词汇':>12} {'文档数':>12}")
    print("-" * 65)
    for s in all_stats:
        print(f"{s['window']:<25} {s['original_vocab']:>12,} {s['filtered_vocab']:>12,} {s['num_docs']:>12,}")

    print(f"\n输出文件位置: {DICT_OUTPUT_DIR}")

    # 保存汇总统计
    stats_file = DICT_OUTPUT_DIR / "dictionary_stats.pkl"
    with open(stats_file, 'wb') as f:
        pickle.dump(all_stats, f)
    print(f"统计信息: {stats_file}")


if __name__ == '__main__':
    run()
