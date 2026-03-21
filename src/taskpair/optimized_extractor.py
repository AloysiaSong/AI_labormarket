#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
任务对提取优化版本
使用优化的多进程处理和更好的内存管理
"""

import pandas as pd
import jieba
import jieba.posseg as pseg
import re
from collections import defaultdict, Counter
from pathlib import Path
import pickle
import logging
import time
import gc
from typing import Dict, List, Tuple
import os

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/Users/yu/code/code2601/TY/output/taskpair/taskpair_extraction_optimized.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class OptimizedTaskPairExtractor:
    """优化的任务对提取器"""

    def __init__(self, data_path: str, output_dir: str):
        self.data_path = data_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 动词和名词词性标签
        self.verb_tags = {'v', 'vn', 'vd', 'vg'}
        self.noun_tags = {'n', 'nr', 'ns', 'nt', 'nz', 'nl', 'ng'}

        # 停用词
        self.stop_words = self._load_stop_words()

        # 任务对统计
        self.task_pairs = defaultdict(int)
        self.verb_freq = Counter()
        self.noun_freq = Counter()

        logger.info("OptimizedTaskPairExtractor initialized")

    def _load_stop_words(self) -> set:
        return {
            '负责', '进行', '完成', '参与', '协助', '配合', '做好', '确保',
            '具备', '具有', '熟悉', '掌握', '了解', '能够', '可以', '需要',
            '要求', '包括', '涉及', '相关', '以及', '和', '与', '或', '等',
            '的', '了', '是', '在', '有', '为', '将', '对', '上', '下',
            '中', '大', '小', '多', '少', '好', '差', '高', '低', '长', '短'
        }

    def clean_text(self, text: str) -> str:
        """清洗职位描述文本"""
        if pd.isna(text):
            return ""

        text = str(text)

        # 移除HTML标签
        text = re.sub(r'<[^>]+>', '', text)

        # 移除特殊字符和数字
        text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z\s]', ' ', text)

        # 移除多余空格
        text = re.sub(r'\s+', ' ', text).strip()

        return text

    def extract_task_pairs(self, text: str) -> List[Tuple[str, str]]:
        """从文本中提取动词-名词对"""
        if not text or len(text) < 10:
            return []

        pairs = []

        try:
            # 分词并标注词性
            words = pseg.cut(text)

            # 转换为列表
            word_list = [(word, flag) for word, flag in words if word not in self.stop_words]

            # 滑动窗口提取相邻的动词-名词对
            for i in range(len(word_list) - 1):
                current_word, current_flag = word_list[i]
                next_word, next_flag = word_list[i + 1]

                # 动词-名词组合
                if (current_flag in self.verb_tags and next_flag in self.noun_tags):
                    if len(current_word) >= 2 and len(next_word) >= 2:
                        pairs.append((current_word, next_word))

                # 名词-动词组合（调换顺序）
                elif (current_flag in self.noun_tags and next_flag in self.verb_tags):
                    if len(current_word) >= 2 and len(next_word) >= 2:
                        pairs.append((next_word, current_word))

        except Exception as e:
            logger.warning(f"Error processing text: {e}")

        return pairs

    def process_chunk_single_thread(self, chunk: pd.DataFrame) -> Dict:
        """单线程处理数据块"""
        local_pairs = defaultdict(int)
        local_verbs = Counter()
        local_nouns = Counter()
        processed_count = 0

        for _, row in chunk.iterrows():
            # 尝试使用不同列名
            text = None
            if 'cleaned_requirements' in row.index and pd.notna(row['cleaned_requirements']):
                text = row['cleaned_requirements']
            elif '职位描述' in row.index and pd.notna(row['职位描述']):
                text = row['职位描述']

            if text:
                cleaned_text = self.clean_text(text)
                pairs = self.extract_task_pairs(cleaned_text)

                # 统计
                for verb, noun in pairs:
                    local_pairs[(verb, noun)] += 1
                    local_verbs[verb] += 1
                    local_nouns[noun] += 1

                processed_count += 1

        return {
            'pairs': dict(local_pairs),
            'verbs': dict(local_verbs),
            'nouns': dict(local_nouns),
            'processed_count': processed_count
        }

    def merge_results(self, results: List[Dict]) -> None:
        """合并结果"""
        for result in results:
            for pair, count in result['pairs'].items():
                self.task_pairs[pair] += count

            for verb, count in result['verbs'].items():
                self.verb_freq[verb] += count

            for noun, count in result['nouns'].items():
                self.noun_freq[noun] += count

    def extract_from_file_optimized(self, chunk_size: int = 10000, max_chunks: int = None) -> None:
        """
        优化的文件处理方式
        使用单线程分批处理，避免多进程开销
        """
        logger.info(f"Starting optimized task pair extraction")
        logger.info(f"Data file: {self.data_path}")
        logger.info(f"Chunk size: {chunk_size}")

        start_time = time.time()
        total_processed = 0
        chunk_count = 0

        try:
            # 分块读取和处理
            for chunk in pd.read_csv(self.data_path, chunksize=chunk_size):
                chunk_start = time.time()

                # 处理当前块
                result = self.process_chunk_single_thread(chunk)
                self.merge_results([result])

                total_processed += result['processed_count']
                chunk_count += 1

                chunk_time = time.time() - chunk_start
                elapsed = time.time() - start_time

                logger.info(f"Processed chunk {chunk_count}: {result['processed_count']} texts "
                           ".2f"
                           ".1f"
                           ".2f")

                # 每处理5个chunk保存一次中间结果
                if chunk_count % 5 == 0:
                    self.save_results(intermediate=True)
                    logger.info(f"Intermediate results saved at chunk {chunk_count}")

                # 内存清理
                del chunk
                gc.collect()

                # 可选的限制处理数量（用于测试）
                if max_chunks and chunk_count >= max_chunks:
                    logger.info(f"Reached max_chunks limit: {max_chunks}")
                    break

        except Exception as e:
            logger.error(f"Error during processing: {e}")
            # 保存当前结果
            self.save_results(intermediate=True)
            logger.info("Intermediate results saved due to error")
            raise

        # 最终统计
        end_time = time.time()
        logger.info(f"Task pair extraction completed in {end_time - start_time:.2f} seconds")
        logger.info(f"Total chunks processed: {chunk_count}")
        logger.info(f"Total texts processed: {total_processed}")
        logger.info(f"Total unique task pairs found: {len(self.task_pairs)}")
        logger.info(f"Total unique verbs: {len(self.verb_freq)}")
        logger.info(f"Total unique nouns: {len(self.noun_freq)}")

    def filter_and_rank_pairs(self, min_freq: int = 5) -> Dict:
        """过滤和排序任务对"""
        filtered_pairs = {
            pair: count for pair, count in self.task_pairs.items()
            if count >= min_freq
        }

        sorted_pairs = dict(sorted(filtered_pairs.items(), key=lambda x: x[1], reverse=True))

        logger.info(f"Filtered task pairs: {len(sorted_pairs)} (min_freq={min_freq})")

        return sorted_pairs

    def save_results(self, intermediate: bool = False) -> None:
        """保存结果"""
        suffix = "_intermediate" if intermediate else ""

        filtered_pairs = self.filter_and_rank_pairs()

        result_dict = {
            'task_pairs': filtered_pairs,
            'verb_freq': dict(self.verb_freq),
            'noun_freq': dict(self.noun_freq),
            'metadata': {
                'total_pairs': len(self.task_pairs),
                'filtered_pairs': len(filtered_pairs),
                'total_verbs': len(self.verb_freq),
                'total_nouns': len(self.noun_freq),
                'extraction_time': time.time()
            }
        }

        output_file = self.output_dir / f"task_pairs_dict{suffix}.pkl"
        with open(output_file, 'wb') as f:
            pickle.dump(result_dict, f)

        logger.info(f"Task pairs dictionary saved to: {output_file}")

        if not intermediate:
            self.save_statistics_report()

    def save_statistics_report(self) -> None:
        """保存统计报告"""
        report_file = self.output_dir / "task_pairs_statistics.txt"

        filtered_pairs = self.filter_and_rank_pairs()

        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("=== 任务对提取统计报告 ===\n\n")

            f.write(f"总任务对数量: {len(self.task_pairs)}\n")
            f.write(f"过滤后任务对数量: {len(filtered_pairs)}\n")
            f.write(f"唯一动词数量: {len(self.verb_freq)}\n")
            f.write(f"唯一名词数量: {len(self.noun_freq)}\n\n")

            f.write("=== Top 20 任务对 ===\n")
            for i, ((verb, noun), count) in enumerate(list(filtered_pairs.items())[:20]):
                f.write("2d")

            f.write("\n=== Top 20 动词 ===\n")
            for i, (verb, count) in enumerate(self.verb_freq.most_common(20)):
                f.write("2d")

            f.write("\n=== Top 20 名词 ===\n")
            for i, (noun, count) in enumerate(self.noun_freq.most_common(20)):
                f.write("2d")

        logger.info(f"Statistics report saved to: {report_file}")


def main():
    """主函数"""
    data_path = "/Users/yu/code/code2601/TY/data/processed/cleaned/all_in_one1.csv"
    output_dir = "/Users/yu/code/code2601/TY/output/taskpair"

    # 创建提取器
    extractor = OptimizedTaskPairExtractor(data_path, output_dir)

    try:
        # 先处理少量数据进行测试
        extractor.extract_from_file_optimized(
            chunk_size=5000,   # 每块5000行（更小的块以便更快保存）
            max_chunks=20      # 先处理20个chunk（10万行）进行测试
        )

        # 保存最终结果
        extractor.save_results(intermediate=False)

        logger.info("✅ Task pair extraction test completed successfully!")
        logger.info("💡 To process full dataset, set max_chunks=None in the script")

    except Exception as e:
        logger.error(f"❌ Task pair extraction failed: {e}")
        raise


if __name__ == "__main__":
    main()