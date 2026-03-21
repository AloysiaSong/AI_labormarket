#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
使用ESCO词典进行分词+短语识别

对每个时间窗口的招聘数据进行：
1. 基于jieba+ESCO用户词典的中文分词
2. 停用词过滤（招聘文本专用）
3. Bigram/Trigram短语识别（如"机器_学习"、"数据_分析"）

输出：
- corpus.pkl: 分词后的语料（list of list of tokens）
- global_phraser.pkl: 全局短语模型（跨窗口一致）
- tokenized.csv: 带tokenized列的CSV（便于人工检查）
"""
import jieba
import pandas as pd
import pickle
import re
import sys
import os
import multiprocessing as mp
import random
from pathlib import Path
try:
    from tqdm import tqdm
except ImportError:
    class _DummyTqdm:
        def __init__(self, iterable=None, total=None, **kwargs):
            self.iterable = iterable
            self.total = total
        def __iter__(self):
            return iter(self.iterable) if self.iterable is not None else iter([])
        def update(self, n=1):
            return None
        def close(self):
            return None
        def __enter__(self):
            return self
        def __exit__(self, exc_type, exc, tb):
            return False
    def tqdm(iterable=None, **kwargs):
        return _DummyTqdm(iterable=iterable, **kwargs)

# 使用集中路径配置
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config.paths import ESCO_JIEBA_DICT, WINDOWS_DIR, PROCESSED_DIR

# 输出目录
TOKENIZED_DIR = PROCESSED_DIR / "tokenized"
TOKENIZED_DIR.mkdir(parents=True, exist_ok=True)

# 并行配置（M4建议8核）
TOKENIZE_WORKERS = min(8, os.cpu_count() or 1)
RANDOM_SEED = 42
# 短语识别安全阈值（文档过多时关闭，避免内存爆炸）
MAX_DOCS_FOR_PHRASES = 1000000
# 允许通过环境变量强制关闭短语识别（1=关闭）
DISABLE_PHRASES = os.environ.get("TOKENIZE_NO_PHRASES", "0") == "1"
# 是否使用全局短语模型（1=启用，0=关闭）
USE_GLOBAL_PHRASES = os.environ.get("TOKENIZE_GLOBAL_PHRASES", "1") == "1"
# 复用已有全局短语模型（1=复用；0=重新训练）
REUSE_GLOBAL_PHRASE = os.environ.get("TOKENIZE_REUSE_GLOBAL_PHRASE", "0") == "1"
# 全局短语训练时每个窗口抽样文档数
GLOBAL_PHRASE_SAMPLE_PER_WINDOW = 100000
# 文本列（只使用cleaned_requirements）
TEXT_COL = "cleaned_requirements"

# 招聘文本专用停用词
STOP_WORDS = set([
    # 通用停用词
    '的', '了', '在', '是', '有', '和', '与', '或', '等', '能', '会', '及',
    '对', '可', '为', '被', '把', '让', '给', '向', '从', '到', '以', '于',
    '个', '这', '那', '一', '不', '也', '要', '就', '都', '而', '但', '如',
    '所', '他', '她', '它', '们', '我', '你', '上', '下', '中', '内', '外',
    # 招聘高频无意义词
    '负责', '具有', '具备', '熟悉', '掌握', '了解', '相关', '以上', '以下',
    '优先', '考虑', '经验', '工作', '能力', '要求', '岗位', '公司', '企业',
    '团队', '良好', '较强', '一定', '优秀', '专业', '学历', '年以上', '年',
    '及以上', '左右', '不限', '若干', '可以', '需要', '进行', '开展',
    '完成', '参与', '支持', '提供', '协助', '配合', '执行', '根据',
    '按照', '通过', '利用', '使用', '包括', '主要', '其他', '相应',
    '有效', '合理', '准确', '及时', '积极', '认真', '仔细', '独立',
    # 薪酬与福利（防止主题被待遇信息污染）
    '薪资', '薪酬', '待遇', '福利', '工资', '底薪', '提成', '奖金', '补贴', '津贴',
    '年终奖', '绩效奖', '加班费', '加班', '餐补', '话补', '车补', '房补',
    '五险', '五险一金', '社保', '公积金', '商业险', '保险', '双休', '大小周',
    '包吃', '包住', '住宿', '食宿', '免费', '小时',
])


def _init_worker(jieba_dict_path: str):
    """子进程初始化：加载自定义词典"""
    if jieba_dict_path and os.path.exists(jieba_dict_path):
        try:
            jieba.load_userdict(jieba_dict_path)
        except Exception:
            pass


def _tokenize_worker(text: str) -> list:
    """多进程分词入口"""
    return tokenize(text)


def clean_text(text: str) -> str:
    """清理文本"""
    if pd.isna(text):
        return ""
    text = str(text)
    # 去除特殊字符，保留中英文、数字、常见编程符号
    text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9\.\+\#\-\_\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def tokenize(text: str) -> list:
    """分词并过滤"""
    text = clean_text(text)
    if not text:
        return []

    words = jieba.lcut(text)
    # 过滤条件：
    # 1. 长度>=2
    # 2. 非停用词
    # 3. 非纯数字
    # 4. 非纯数字加点号（如版本号3.0）
    words = [w for w in words
             if len(w) >= 2
             and w.lower() not in STOP_WORDS
             and w not in STOP_WORDS
             and not w.isdigit()
             and not re.match(r'^[\d\.]+$', w)]
    return words


def _reservoir_sample_texts(window_file: Path, text_col: str, sample_size: int) -> list:
    """从大文件流式抽样文本（不分词），用于全局短语训练"""
    if sample_size <= 0:
        return []
    sample = []
    seen = 0
    try:
        for chunk in pd.read_csv(window_file, usecols=[text_col], chunksize=50000, low_memory=False):
            for text in chunk[text_col]:
                if pd.isna(text):
                    continue
                seen += 1
                if len(sample) < sample_size:
                    sample.append(text)
                else:
                    j = random.randrange(seen)
                    if j < sample_size:
                        sample[j] = text
    except ValueError:
        raise ValueError(f"缺少文本列: {text_col} in {window_file}")
    return sample


def train_global_phraser(window_files: list) -> tuple:
    """训练全局短语模型（跨窗口一致）"""
    try:
        from gensim.models import Phrases
        from gensim.models.phrases import Phraser
    except ImportError:
        print("警告: gensim未安装，跳过短语识别")
        return None

    print("\nStep 0: 训练全局短语模型...")
    samples = []
    for window_file in window_files:
        print(f"  抽样: {window_file.name} (每窗 {GLOBAL_PHRASE_SAMPLE_PER_WINDOW:,})")
        texts = _reservoir_sample_texts(window_file, TEXT_COL, GLOBAL_PHRASE_SAMPLE_PER_WINDOW)
        for text in texts:
            tokens = tokenize(text)
            if tokens:
                samples.append(tokens)

    if not samples:
        print("  警告: 全局短语训练样本为空，跳过")
        return None

    print(f"  训练样本总量: {len(samples):,} 文档")
    print("  训练 Bigram...")
    bigram = Phrases(samples, min_count=100, threshold=15)
    bigram_phraser = Phraser(bigram)

    print("  训练 Trigram...")
    trigram = Phrases(bigram_phraser[samples], min_count=50, threshold=15)
    trigram_phraser = Phraser(trigram)

    phraser_file = TOKENIZED_DIR / "global_phraser.pkl"
    with open(phraser_file, 'wb') as f:
        pickle.dump((bigram_phraser, trigram_phraser), f)
    print(f"  保存全局短语模型: {phraser_file}")

    return bigram_phraser, trigram_phraser


def process_window(window_file: Path, use_phrases: bool = True, global_phraser: tuple = None):
    """
    处理单个时间窗口

    Args:
        window_file: 窗口CSV文件路径
        use_phrases: 是否使用短语识别（对于大数据集建议关闭以节省内存）
    """
    window_name = window_file.stem
    print(f"\n{'='*60}")
    print(f"处理 {window_name}")
    print(f"{'='*60}")

    # 读取数据
    print(f"\n读取数据: {window_file}")
    df = pd.read_csv(window_file, low_memory=False)
    print(f"记录数: {len(df):,}")

    # 确定文本列
    text_col = TEXT_COL
    if text_col not in df.columns:
        print(f"错误: 找不到文本列 '{TEXT_COL}'。可用列: {df.columns.tolist()}")
        return 0

    print(f"使用文本列: {text_col}")

    # Step 1: 基础分词（并行）
    print("\nStep 1: 分词...")
    corpus = []
    valid_indices = []

    texts = df[text_col].tolist()
    if TOKENIZE_WORKERS > 1:
        print(f"  使用并行分词: {TOKENIZE_WORKERS} 核")
        with mp.Pool(
            processes=TOKENIZE_WORKERS,
            initializer=_init_worker,
            initargs=(str(ESCO_JIEBA_DICT),)
        ) as pool:
            for idx, tokens in enumerate(tqdm(pool.imap(_tokenize_worker, texts, chunksize=1000), desc="分词")):
                if tokens:
                    corpus.append(tokens)
                    valid_indices.append(idx)
    else:
        for idx, text in enumerate(tqdm(texts, desc="分词")):
            tokens = tokenize(text)
            if tokens:
                corpus.append(tokens)
                valid_indices.append(idx)

    print(f"有效文档: {len(corpus):,} / {len(df):,} ({len(corpus)/len(df)*100:.1f}%)")

    # Step 2: 短语识别（可选）
    if use_phrases and len(corpus) > 0 and global_phraser is None:
        if len(corpus) > MAX_DOCS_FOR_PHRASES:
            print(f"  ⚠️ 文档数 {len(corpus):,} 超过阈值 {MAX_DOCS_FOR_PHRASES:,}，自动关闭短语识别")
            use_phrases = False
        else:
            print(f"  ✅ 短语识别启用（文档数 {len(corpus):,}）")

    if use_phrases and len(corpus) > 0:
        if global_phraser is not None:
            print("\nStep 2: 应用全局短语模型...")
            bigram_phraser, trigram_phraser = global_phraser
            corpus = [list(trigram_phraser[bigram_phraser[doc]])
                      for doc in tqdm(corpus, desc="应用短语")]
        else:
            try:
                from gensim.models import Phrases
                from gensim.models.phrases import Phraser

                print("\nStep 2: 训练短语模型...")

                # 使用随机样本训练短语模型，避免按文件顺序取样带来的偏差
                sample_size = min(500000, len(corpus))
                random.seed(RANDOM_SEED)
                sample_indices = random.sample(range(len(corpus)), sample_size)
                sample = [corpus[i] for i in sample_indices]
                print(f"  训练样本: {sample_size:,} 文档")

                # Bigram
                print("  训练 Bigram...")
                bigram = Phrases(sample, min_count=100, threshold=15)
                bigram_phraser = Phraser(bigram)

                # Trigram
                print("  训练 Trigram...")
                trigram = Phrases(bigram_phraser[sample], min_count=50, threshold=15)
                trigram_phraser = Phraser(trigram)

                # Step 3: 应用短语模型
                print("\nStep 3: 应用短语模型...")
                corpus = [list(trigram_phraser[bigram_phraser[doc]])
                          for doc in tqdm(corpus, desc="应用短语")]

                # 保存短语模型
                phraser_file = TOKENIZED_DIR / f"{window_name}_phraser.pkl"
                with open(phraser_file, 'wb') as f:
                    pickle.dump((bigram_phraser, trigram_phraser), f)
                print(f"保存短语模型: {phraser_file}")

            except ImportError:
                print("警告: gensim未安装，跳过短语识别")
                use_phrases = False

    # 保存语料
    corpus_file = TOKENIZED_DIR / f"{window_name}_corpus.pkl"
    with open(corpus_file, 'wb') as f:
        pickle.dump(corpus, f)
    print(f"保存语料: {corpus_file}")

    # 保存带tokenized列的CSV（用于后续分析和人工检查）
    df_valid = df.iloc[valid_indices].copy()
    df_valid['tokenized'] = [' '.join(doc) for doc in corpus]
    tokenized_file = TOKENIZED_DIR / f"{window_name}_tokenized.csv"
    df_valid.to_csv(tokenized_file, index=False, encoding='utf-8-sig')
    print(f"保存分词CSV: {tokenized_file}")

    # 统计信息
    total_tokens = sum(len(doc) for doc in corpus)
    avg_tokens = total_tokens / len(corpus) if corpus else 0
    vocab = set(token for doc in corpus for token in doc)

    print(f"\n统计:")
    print(f"  文档数: {len(corpus):,}")
    print(f"  总词数: {total_tokens:,}")
    print(f"  平均词数/文档: {avg_tokens:.1f}")
    print(f"  词表大小: {len(vocab):,}")

    return len(corpus)


def run():
    print("=" * 60)
    print("ESCO词典分词与短语识别")
    print("=" * 60)

    # 加载ESCO词典
    print(f"\n加载ESCO词典: {ESCO_JIEBA_DICT}")
    if ESCO_JIEBA_DICT.exists():
        jieba.load_userdict(str(ESCO_JIEBA_DICT))
        with open(ESCO_JIEBA_DICT, 'r', encoding='utf-8') as f:
            dict_size = sum(1 for _ in f)
        print(f"词典大小: {dict_size:,} 词")
    else:
        print(f"警告: ESCO词典不存在: {ESCO_JIEBA_DICT}")
        print("将使用jieba默认词典")

    print(f"\n输入目录: {WINDOWS_DIR}")
    print(f"输出目录: {TOKENIZED_DIR}")

    # 处理所有窗口
    total = 0
    window_files = sorted(
        f for f in WINDOWS_DIR.glob('window_*.csv')
        if re.match(r'^window_\d{4}_\d{4}\.csv$', f.name)
    )

    if not window_files:
        print(f"错误: 在 {WINDOWS_DIR} 中未找到窗口文件")
        return

    print(f"\n找到 {len(window_files)} 个窗口文件:")
    for f in window_files:
        print(f"  - {f.name}")

    global_phraser = None
    if not DISABLE_PHRASES and USE_GLOBAL_PHRASES:
        global_phraser_path = TOKENIZED_DIR / "global_phraser.pkl"
        if REUSE_GLOBAL_PHRASE and global_phraser_path.exists():
            print("\n复用已有全局短语模型...")
            with open(global_phraser_path, 'rb') as f:
                global_phraser = pickle.load(f)
        else:
            global_phraser = train_global_phraser(window_files)

    for window_file in window_files:
        if USE_GLOBAL_PHRASES:
            count = process_window(window_file, use_phrases=not DISABLE_PHRASES, global_phraser=global_phraser)
        else:
            count = process_window(window_file, use_phrases=not DISABLE_PHRASES, global_phraser=None)
        total += count

    print("\n" + "=" * 60)
    print(f"全部完成！总文档数: {total:,}")
    print("=" * 60)
    print(f"\n输出文件位置: {TOKENIZED_DIR}")


if __name__ == '__main__':
    run()
