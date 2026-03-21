#!/usr/bin/env python3
"""
生产版本的任务对提取脚本 - 处理完整数据集
"""

import pandas as pd
import jieba
import jieba.posseg as pseg
from collections import defaultdict, Counter
import pickle
import logging
import time
import gc
import os
from pathlib import Path

class ProductionTaskPairExtractor:
    def __init__(self, data_file, output_dir, chunk_size=10000, min_freq=5):
        self.data_file = data_file
        self.output_dir = Path(output_dir)
        self.chunk_size = chunk_size
        self.min_freq = min_freq

        # 创建输出目录
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 设置日志
        self.setup_logging()

        # 初始化数据结构
        self.task_pairs = defaultdict(int)
        self.verb_freq = Counter()
        self.noun_freq = Counter()
        self.total_texts = 0

        logging.info(f"ProductionTaskPairExtractor initialized")
        logging.info(f"Data file: {data_file}")
        logging.info(f"Chunk size: {chunk_size}")

    def setup_logging(self):
        log_file = self.output_dir / "taskpair_extraction_production.log"
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file, encoding='utf-8'),
                logging.StreamHandler()
            ]
        )

    def extract_task_pairs_from_text(self, text):
        """从单个文本中提取任务对"""
        if not isinstance(text, str) or not text.strip():
            return []

        pairs = []
        try:
            # 分词并标注词性
            words = pseg.cut(text)

            # 提取动词和名词序列
            verbs = []
            nouns = []

            for word, flag in words:
                if flag.startswith('v'):  # 动词
                    verbs.append(word)
                elif flag.startswith('n'):  # 名词
                    nouns.append(word)

            # 生成动词-名词对
            for verb in verbs:
                for noun in nouns:
                    pairs.append((verb, noun))

        except Exception as e:
            logging.warning(f"Error processing text: {e}")

        return pairs

    def process_chunk(self, chunk, chunk_num):
        """处理一个数据块"""
        start_time = time.time()
        chunk_pairs = defaultdict(int)
        chunk_verbs = Counter()
        chunk_nouns = Counter()

        for text in chunk:
            pairs = self.extract_task_pairs_from_text(text)
            for verb, noun in pairs:
                chunk_pairs[(verb, noun)] += 1
                chunk_verbs[verb] += 1
                chunk_nouns[noun] += 1

        # 合并到全局计数器
        for pair, count in chunk_pairs.items():
            self.task_pairs[pair] += count
        self.verb_freq.update(chunk_verbs)
        self.noun_freq.update(chunk_nouns)

        processing_time = time.time() - start_time
        texts_count = len(chunk)
        self.total_texts += texts_count

        logging.info(f"Processed chunk {chunk_num}: {texts_count} texts in {processing_time:.2f}s")

        # 每5个块保存一次中间结果
        if chunk_num % 5 == 0:
            self.save_intermediate_results(chunk_num)

        # 垃圾回收
        del chunk_pairs, chunk_verbs, chunk_nouns
        gc.collect()

    def save_intermediate_results(self, chunk_num):
        """保存中间结果"""
        # 过滤低频任务对
        filtered_pairs = {pair: count for pair, count in self.task_pairs.items() if count >= self.min_freq}

        data = {
            'task_pairs': filtered_pairs,
            'verb_freq': dict(self.verb_freq),
            'noun_freq': dict(self.noun_freq),
            'metadata': {
                'total_pairs': len(self.task_pairs),
                'filtered_pairs': len(filtered_pairs),
                'total_verbs': len(self.verb_freq),
                'total_nouns': len(self.noun_freq),
                'total_texts': self.total_texts,
                'chunk_num': chunk_num,
                'min_freq': self.min_freq
            }
        }

        output_file = self.output_dir / f"task_pairs_dict_intermediate_chunk_{chunk_num}.pkl"
        with open(output_file, 'wb') as f:
            pickle.dump(data, f)

        logging.info(f"Intermediate results saved at chunk {chunk_num}: {len(filtered_pairs)} filtered pairs")

    def save_final_results(self):
        """保存最终结果"""
        # 过滤低频任务对
        filtered_pairs = {pair: count for pair, count in self.task_pairs.items() if count >= self.min_freq}

        data = {
            'task_pairs': filtered_pairs,
            'verb_freq': dict(self.verb_freq),
            'noun_freq': dict(self.noun_freq),
            'metadata': {
                'total_pairs': len(self.task_pairs),
                'filtered_pairs': len(filtered_pairs),
                'total_verbs': len(self.verb_freq),
                'total_nouns': len(self.noun_freq),
                'total_texts': self.total_texts,
                'min_freq': self.min_freq
            }
        }

        # 保存最终结果
        final_file = self.output_dir / "task_pairs_dict_final.pkl"
        with open(final_file, 'wb') as f:
            pickle.dump(data, f)

        # 生成统计报告
        self.generate_statistics_report(data)

        logging.info(f"Final results saved: {len(filtered_pairs)} filtered pairs")

    def generate_statistics_report(self, data):
        """生成统计报告"""
        report_file = self.output_dir / "task_pairs_statistics_final.txt"

        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("=== 任务对提取统计报告 (完整数据集) ===\n\n")
            f.write(f"总任务对数量: {data['metadata']['total_pairs']}\n")
            f.write(f"过滤后任务对数量: {data['metadata']['filtered_pairs']}\n")
            f.write(f"唯一动词数量: {data['metadata']['total_verbs']}\n")
            f.write(f"唯一名词数量: {data['metadata']['total_nouns']}\n")
            f.write(f"处理文本总数: {data['metadata']['total_texts']}\n")
            f.write(f"最小频率阈值: {data['metadata']['min_freq']}\n\n")

            # Top 20 任务对
            f.write("=== Top 20 任务对 ===\n")
            sorted_pairs = sorted(data['task_pairs'].items(), key=lambda x: x[1], reverse=True)
            for i, ((verb, noun), count) in enumerate(sorted_pairs[:20]):
                f.write(f"{i+1}. {verb}-{noun}: {count}\n")

            # Top 20 动词
            f.write("\n=== Top 20 动词 ===\n")
            sorted_verbs = sorted(data['verb_freq'].items(), key=lambda x: x[1], reverse=True)
            for i, (verb, count) in enumerate(sorted_verbs[:20]):
                f.write(f"{i+1}. {verb}: {count}\n")

            # Top 20 名词
            f.write("\n=== Top 20 名词 ===\n")
            sorted_nouns = sorted(data['noun_freq'].items(), key=lambda x: x[1], reverse=True)
            for i, (noun, count) in enumerate(sorted_nouns[:20]):
                f.write(f"{i+1}. {noun}: {count}\n")

    def run(self):
        """运行完整的提取过程"""
        start_time = time.time()
        logging.info("Starting production task pair extraction")

        try:
            # 读取数据文件
            df_iter = pd.read_csv(self.data_file, chunksize=self.chunk_size, encoding='utf-8')

            chunk_num = 0
            for chunk in df_iter:
                chunk_num += 1

                # 假设文本在第一列，或者你需要指定列名
                if len(chunk.columns) > 0:
                    text_column = chunk.columns[0]  # 使用第一列
                    texts = chunk[text_column].fillna('').astype(str).tolist()
                else:
                    logging.error("No columns found in CSV file")
                    break

                self.process_chunk(texts, chunk_num)

                # 检查是否需要停止（可以设置最大块数限制）
                # 如果需要处理完整数据集，移除这个检查
                # if chunk_num >= 100:  # 示例：限制为100个块
                #     logging.info(f"Reached chunk limit: {chunk_num}")
                #     break

            # 保存最终结果
            self.save_final_results()

            total_time = time.time() - start_time
            logging.info(f"Task pair extraction completed in {total_time:.2f} seconds")
            logging.info(f"Total chunks processed: {chunk_num}")
            logging.info(f"Total texts processed: {self.total_texts}")
            logging.info(f"Total unique task pairs found: {len(self.task_pairs)}")
            logging.info(f"Total unique verbs: {len(self.verb_freq)}")
            logging.info(f"Total unique nouns: {len(self.noun_freq)}")

            # 过滤统计
            filtered_count = sum(1 for count in self.task_pairs.values() if count >= self.min_freq)
            logging.info(f"Filtered task pairs (min_freq={self.min_freq}): {filtered_count}")

            print("✅ Production task pair extraction completed successfully!")
            print(f"📊 Processed {self.total_texts} texts in {total_time:.2f} seconds")
            print(f"🔍 Found {len(self.task_pairs)} unique task pairs, {filtered_count} after filtering")

        except Exception as e:
            logging.error(f"Error during extraction: {e}")
            raise

def main():
    # 配置路径
    data_file = "/Users/yu/code/code2601/TY/data/processed/cleaned/all_in_one1.csv"
    output_dir = "/Users/yu/code/code2601/TY/output/taskpair"

    # 检查数据文件是否存在
    if not os.path.exists(data_file):
        print(f"❌ Data file not found: {data_file}")
        return

    # 创建提取器并运行
    extractor = ProductionTaskPairExtractor(
        data_file=data_file,
        output_dir=output_dir,
        chunk_size=10000,  # 每块处理10000条记录
        min_freq=5  # 最小频率阈值
    )

    extractor.run()

if __name__ == "__main__":
    main()