#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step 2: 基于 MinHash + LSH 去除重复发布的JD
功能：
    1. 使用 MinHash 计算文本指纹
    2. 基于 Jaccard 相似度的 LSH 近似去重
    3. Blocking 降低规模与内存占用
    
输入: /Users/yu/code/code2601/TY/Test_LDA/task1_datacleaning/all_in_one2.csv
输出: /Users/yu/code/code2601/TY/Test_LDA/task1_datacleaning/all_in_one2_dedup.csv

注意：SimHash 适合大规模文本去重，计算效率高
"""

import pandas as pd
import re
import hashlib
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
import logging
from datasketch import MinHash, MinHashLSH, LeanMinHash

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============== 路径配置 ==============
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config.paths import EXTRACTED_DATA, EXTRACTED_DIR, DEDUPED_DATA, DEDUPED_DIR

# 确保输出目录存在
DEDUPED_DIR.mkdir(parents=True, exist_ok=True)

INPUT_FILE = str(EXTRACTED_DATA)
OUTPUT_FILE = str(DEDUPED_DATA)
DUPLICATES_FILE = str(DEDUPED_DIR / 'removed_duplicates.csv')
PARTS_DIR = EXTRACTED_DIR / "parts"
MERGED_FILE = EXTRACTED_DIR / "all_in_one2_merged.csv"
MERGED_DONE_FILE = EXTRACTED_DIR / "all_in_one2_merged.done"

# MinHash 参数
MINHASH_NUM_PERM = 64          # 置换数（越大越准，越慢/占内存）
MINHASH_THRESHOLD = 0.90       # Jaccard相似度阈值
SHINGLE_SIZE = 3               # 字符n-gram大小
BLOCK_COLS = ['招聘发布年份', '工作城市']  # Blocking字段，降低规模


def _normalize_text(text: str) -> str:
    """清理文本，保留中文/英文/数字"""
    if not text:
        return ""
    return re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9]', '', str(text))


def _shingles(text: str, size: int = SHINGLE_SIZE):
    """生成字符n-gram"""
    if len(text) < size:
        return []
    return [text[i:i+size] for i in range(len(text) - size + 1)]


def _build_minhash(text: str) -> MinHash:
    """构建MinHash"""
    text = _normalize_text(text)
    if not text:
        return None
    mh = MinHash(num_perm=MINHASH_NUM_PERM)
    for token in _shingles(text, SHINGLE_SIZE):
        mh.update(token.encode('utf-8'))
    return mh


def merge_parts_if_needed() -> str:
    """
    若存在分段输出，则合并为单一CSV供去重使用。
    返回实际输入文件路径。
    """
    input_path = Path(INPUT_FILE)
    part_files = sorted(PARTS_DIR.glob("part_*.csv")) if PARTS_DIR.exists() else []
    has_parts = len(part_files) > 0
    has_input = input_path.exists()

    if not has_parts and not has_input:
        raise FileNotFoundError(f"未找到输入文件或分段目录: {INPUT_FILE} / {PARTS_DIR}")

    if has_input and not has_parts:
        logger.info(f"使用已存在的输入文件: {INPUT_FILE}")
        return INPUT_FILE

    if not has_parts:
        raise FileNotFoundError(f"分段目录为空: {PARTS_DIR}")

    # 若 parts 比 all_in_one2 新，则优先使用 parts（确保使用最新抽取结果）
    parts_mtime = max(p.stat().st_mtime for p in part_files)
    if has_input and input_path.stat().st_mtime >= parts_mtime:
        logger.info(f"使用已存在输入文件（不早于 parts）: {INPUT_FILE}")
        return INPUT_FILE

    # 仅当存在完成标记时，才认为合并文件可复用
    if MERGED_FILE.exists() and MERGED_DONE_FILE.exists():
        merged_mtime = MERGED_FILE.stat().st_mtime
        if merged_mtime >= parts_mtime:
            logger.info(f"使用已合并文件: {MERGED_FILE}")
            return str(MERGED_FILE)
    elif MERGED_FILE.exists() and not MERGED_DONE_FILE.exists():
        logger.warning(f"检测到未完成的合并文件，将重建: {MERGED_FILE}")
        MERGED_FILE.unlink(missing_ok=True)

    logger.info(f"合并分段文件到: {MERGED_FILE}")
    MERGED_FILE.parent.mkdir(parents=True, exist_ok=True)

    first = True
    for part in part_files:
        logger.info(f"  合并: {part.name}")
        for chunk in pd.read_csv(part, chunksize=50000):
            chunk.to_csv(
                MERGED_FILE,
                mode='w' if first else 'a',
                index=False,
                header=first,
                encoding='utf-8'
            )
            first = False

    # 写入完成标记，避免下次误用中断产生的半成品
    MERGED_DONE_FILE.write_text("ok", encoding="utf-8")

    return str(MERGED_FILE)


def deduplicate_minhash(df: pd.DataFrame,
                        text_column: str = 'cleaned_requirements',
                        block_cols: list = None) -> tuple:
    """
    使用 MinHash + LSH 去重（Blocking分组）
    返回: (保留索引列表, 删除索引列表)
    """
    logger.info("开始 MinHash 去重...")
    block_cols = block_cols or []

    keep_indices = []
    duplicate_indices = []

    if block_cols:
        grouped = df.groupby(block_cols, dropna=False)
    else:
        grouped = [(None, df)]

    for block_key, block_df in grouped:
        logger.info(f"处理分组: {block_key} | 规模: {len(block_df):,}")

        lsh = MinHashLSH(threshold=MINHASH_THRESHOLD, num_perm=MINHASH_NUM_PERM)
        stored = {}

        for idx, text in tqdm(block_df[text_column].fillna('').items(), desc="MinHash去重"):
            mh = _build_minhash(text)
            if mh is None:
                duplicate_indices.append(idx)
                continue

            candidates = lsh.query(mh)
            is_dup = False
            for cand in candidates:
                if stored[cand].jaccard(mh) >= MINHASH_THRESHOLD:
                    is_dup = True
                    break

            if is_dup:
                duplicate_indices.append(idx)
            else:
                key = str(idx)
                lsh.insert(key, mh)
                stored[key] = LeanMinHash(mh)
                keep_indices.append(idx)

    logger.info(f"去重完成: 保留 {len(keep_indices):,}, 删除 {len(duplicate_indices):,}")
    return keep_indices, duplicate_indices


def exact_hash_dedup(df: pd.DataFrame, 
                     text_column: str = 'cleaned_requirements') -> tuple:
    """
    精确去重：完全相同的文本
    作为 SimHash 的补充
    """
    logger.info("Step 0: 精确哈希去重...")
    
    seen_hashes = {}
    keep_indices = []
    duplicate_indices = []
    
    for idx, text in enumerate(tqdm(df[text_column].fillna(''), desc="精确去重")):
        text_hash = hashlib.md5(text.encode('utf-8')).hexdigest()
        
        if text_hash in seen_hashes:
            duplicate_indices.append(idx)
        else:
            seen_hashes[text_hash] = idx
            keep_indices.append(idx)
    
    logger.info(f"精确去重完成: 保留 {len(keep_indices):,}, 删除 {len(duplicate_indices):,}")
    
    return keep_indices, duplicate_indices


def main():
    """主函数"""
    logger.info(f"开始去重处理...")
    input_file = merge_parts_if_needed()
    logger.info(f"输入文件: {input_file}")
    logger.info(f"输出文件: {OUTPUT_FILE}")
    
    # 读取数据
    logger.info("读取数据...")
    df = pd.read_csv(input_file)
    original_count = len(df)
    logger.info(f"原始记录数: {original_count:,}")
    
    # Step 0: 精确去重
    exact_keep, exact_dup = exact_hash_dedup(df)
    df_step0 = df.iloc[exact_keep].reset_index(drop=True)
    logger.info(f"精确去重后: {len(df_step0):,} 条")
    
    # Step 1: MinHash 近似去重（Blocking）
    minhash_keep, minhash_dup = deduplicate_minhash(df_step0, block_cols=BLOCK_COLS)
    
    # 获取最终保留的数据
    df_final = df_step0.loc[minhash_keep].reset_index(drop=True)
    
    # 获取被删除的数据（用于审查）
    df_exact_dup = df.iloc[exact_dup]
    df_simhash_dup = df_step0.loc[minhash_dup]
    df_all_dup = pd.concat([df_exact_dup, df_simhash_dup], ignore_index=True)
    
    # 保存结果
    logger.info("保存结果...")
    df_final.to_csv(OUTPUT_FILE, index=False, encoding='utf-8')
    
    # 保存被删除的重复项（可选，用于审查）
    if len(df_all_dup) > 0:
        # 只保存一部分样本用于审查
        sample_size = min(10000, len(df_all_dup))
        df_all_dup.sample(sample_size).to_csv(DUPLICATES_FILE, index=False, encoding='utf-8')
        logger.info(f"已保存 {sample_size} 条重复样本到: {DUPLICATES_FILE}")
    
    # 统计
    final_count = len(df_final)
    logger.info(f"\n{'='*50}")
    logger.info(f"去重统计:")
    logger.info(f"  原始记录数: {original_count:,}")
    logger.info(f"  精确去重删除: {len(exact_dup):,}")
    logger.info(f"  MinHash去重删除: {len(minhash_dup):,}")
    logger.info(f"  最终保留: {final_count:,}")
    logger.info(f"  去重比例: {(original_count-final_count)/original_count*100:.2f}%")
    logger.info(f"{'='*50}")
    logger.info(f"输出文件: {OUTPUT_FILE}")


if __name__ == '__main__':
    main()
