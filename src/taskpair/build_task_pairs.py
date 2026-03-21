#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
任务对提取模块 - Task Pair Extraction
从职位描述中提取动词-名词对，用于分析技能组合模式

作者: AI Assistant
日期: 2026-02-04
"""

import pandas as pd
import jieba
import jieba.posseg as pseg
import re
import pickle
from collections import defaultdict, Counter
from pathlib import Path
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
from typing import Dict, List, Tuple, Set
import gc
import time

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/Users/yu/code/code2601/TY/output/taskpair/taskpair_extraction.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class TaskPairExtractor:
    """任务对提取器"""

    def __init__(self, data_path: str, output_dir: str):
        """
        初始化任务对提取器

        Args:
            data_path: 输入数据文件路径
            output_dir: 输出目录
        """
        self.data_path = data_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 动词和名词词性标签
        self.verb_tags = {'v', 'vn', 'vd', 'vg'}  # 动词相关
        self.noun_tags = {'n', 'nr', 'ns', 'nt', 'nz', 'nl', 'ng'}  # 名词相关

        # 停用词
        self.stop_words = self._load_stop_words()

        # 任务对统计
        self.task_pairs = defaultdict(int)  # (verb, noun) -> count
        self.verb_freq = Counter()  # 动词频率
        self.noun_freq = Counter()  # 名词频率

        logger.info(f"TaskPairExtractor initialized with data: {data_path}")

    def _load_stop_words(self) -> Set[str]:
        """加载停用词"""
        stop_words = {
            '负责', '进行', '完成', '参与', '协助', '配合', '做好', '确保',
            '具备', '具有', '熟悉', '掌握', '了解', '能够', '可以', '需要',
            '要求', '包括', '涉及', '相关', '以及', '和', '与', '或', '等',
            '的', '了', '是', '在', '有', '为', '将', '对', '上', '下',
            '中', '大', '小', '多', '少', '好', '差', '高', '低', '长', '短'
        }
        return stop_words

    def clean_text(self, text: str) -> str:
        """清洗职位描述文本"""
        if pd.isna(text):
            return ""

        # 移除HTML标签
        text = re.sub(r'<[^>]+>', '', str(text))

        # 移除特殊字符和数字
        text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z\s]', ' ', text)

        # 移除多余空格
        text = re.sub(r'\s+', ' ', text).strip()

        return text

    def extract_task_pairs(self, text: str) -> List[Tuple[str, str]]:
        """
        从文本中提取动词-名词对

        Args:
            text: 输入文本

        Returns:
            动词-名词对列表 [(verb, noun), ...]
        """
        if not text or len(text) < 10:
            return []

        pairs = []

        try:
            # 分词并标注词性
            words = pseg.cut(text)

            # 转换为列表以便多次遍历
            word_list = [(word, flag) for word, flag in words if word not in self.stop_words]

            # 滑动窗口提取相邻的动词-名词对
            for i in range(len(word_list) - 1):
                current_word, current_flag = word_list[i]
                next_word, next_flag = word_list[i + 1]

                # 检查是否是动词-名词组合
                if (current_flag in self.verb_tags and next_flag in self.noun_tags):
                    # 过滤太短的词
                    if len(current_word) >= 2 and len(next_word) >= 2:
                        pairs.append((current_word, next_word))

                # 也检查名词-动词组合（被动结构）
                elif (current_flag in self.noun_tags and next_flag in self.verb_tags):
                    if len(current_word) >= 2 and len(next_word) >= 2:
                        pairs.append((next_word, current_word))  # 调换顺序，使动词在前

        except Exception as e:
            logger.warning(f"Error processing text: {e}")

        return pairs

    def process_chunk(self, chunk: pd.DataFrame) -> Dict:
        """
        处理数据块

        Args:
            chunk: 数据块

        Returns:
            处理结果字典
        """
        local_pairs = defaultdict(int)
        local_verbs = Counter()
        local_nouns = Counter()
        processed_count = 0

        for _, row in chunk.iterrows():
            # 尝试使用 cleaned_requirements 列，如果没有则使用职位描述
            text_col = 'cleaned_requirements' if 'cleaned_requirements' in row.index else '职位描述'

            if text_col in row.index:
                text = row[text_col]
                cleaned_text = self.clean_text(text)
                pairs = self.extract_task_pairs(cleaned_text)

                # 统计任务对
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
        """合并多进程结果"""
        for result in results:
            # 合并任务对
            for pair, count in result['pairs'].items():
                self.task_pairs[pair] += count

            # 合并动词频率
            for verb, count in result['verbs'].items():
                self.verb_freq[verb] += count

            # 合并名词频率
            for noun, count in result['nouns'].items():
                self.noun_freq[noun] += count

    def extract_from_file(self, chunk_size: int = 50000, max_workers: int = None) -> None:
        """
        从文件中提取任务对

        Args:
            chunk_size: 每次处理的数据块大小
            max_workers: 最大并行进程数
        """
        if max_workers is None:
            max_workers = min(mp.cpu_count(), 8)  # 最多使用8个进程

        logger.info(f"Starting task pair extraction with {max_workers} workers")
        logger.info(f"Data file: {self.data_path}")
        logger.info(f"Chunk size: {chunk_size}")

        start_time = time.time()
        total_processed = 0

        # 分块读取和处理
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = []

            try:
                # 读取数据文件
                for chunk in pd.read_csv(self.data_path, chunksize=chunk_size):
                    # 提交处理任务
                    future = executor.submit(self.process_chunk, chunk)
                    futures.append(future)

                    # 控制内存使用，每100个chunk处理一次结果
                    if len(futures) >= 100:
                        self._process_completed_futures(futures, start_time, total_processed)
                        total_processed += sum(f.result()['processed_count'] for f in futures if f.done())
                        futures = []
                        gc.collect()  # 垃圾回收

                # 处理剩余的任务
                if futures:
                    self._process_completed_futures(futures, start_time, total_processed)

            except Exception as e:
                logger.error(f"Error during processing: {e}")
                raise

        # 最终统计
        end_time = time.time()
        logger.info(f"Task pair extraction completed in {end_time - start_time:.2f} seconds")
        logger.info(f"Total task pairs found: {len(self.task_pairs)}")
        logger.info(f"Total unique verbs: {len(self.verb_freq)}")
        logger.info(f"Total unique nouns: {len(self.noun_freq)}")

    def _process_completed_futures(self, futures, start_time, total_processed):
        """处理已完成的任务"""
        results = []
        for future in as_completed(futures):
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                logger.error(f"Task failed: {e}")

        self.merge_results(results)

        # 定期保存中间结果
        elapsed = time.time() - start_time
        if elapsed > 300:  # 每5分钟保存一次
            self.save_results(intermediate=True)

    def filter_and_rank_pairs(self, min_freq: int = 5) -> Dict:
        """
        过滤和排序任务对

        Args:
            min_freq: 最小出现频率

        Returns:
            过滤后的任务对字典
        """
        # 过滤低频任务对
        filtered_pairs = {
            pair: count for pair, count in self.task_pairs.items()
            if count >= min_freq
        }

        # 按频率排序
        sorted_pairs = dict(sorted(filtered_pairs.items(), key=lambda x: x[1], reverse=True))

        logger.info(f"Filtered task pairs: {len(sorted_pairs)} (min_freq={min_freq})")

        return sorted_pairs

    def save_results(self, intermediate: bool = False) -> None:
        """
        保存结果

        Args:
            intermediate: 是否为中间结果
        """
        suffix = "_intermediate" if intermediate else ""

        # 保存任务对词典
        pairs_file = self.output_dir / f"task_pairs_dict{suffix}.pkl"
        filtered_pairs = self.filter_and_rank_pairs()

        with open(pairs_file, 'wb') as f:
            pickle.dump({
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
            }, f)

        logger.info(f"Task pairs dictionary saved to: {pairs_file}")

        # 保存统计报告
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
    # 配置路径
    data_path = "/Users/yu/code/code2601/TY/data/processed/cleaned/all_in_one1.csv"
    output_dir = "/Users/yu/code/code2601/TY/output/taskpair"

    # 创建输出目录
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # 初始化提取器
    extractor = TaskPairExtractor(data_path, output_dir)

    # 执行提取
    try:
        extractor.extract_from_file(
            chunk_size=50000,  # 每块处理5万行
            max_workers=6      # 使用6个并行进程（适合M4芯片）
        )

        # 保存最终结果
        extractor.save_results(intermediate=False)

        logger.info("Task pair extraction completed successfully!")

    except Exception as e:
        logger.error(f"Task pair extraction failed: {e}")
        raise


if __name__ == "__main__":
    main()