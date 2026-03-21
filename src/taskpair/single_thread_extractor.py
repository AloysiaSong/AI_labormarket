#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
任务对提取单线程版本
用于处理小批量数据测试功能
"""

import pandas as pd
import jieba
import jieba.posseg as pseg
import re
from collections import defaultdict, Counter
from pathlib import Path
import pickle
import time

class SingleThreadTaskPairExtractor:
    """单线程任务对提取器"""

    def __init__(self, output_dir: str):
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

    def _load_stop_words(self) -> set:
        """加载停用词"""
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

        # 移除HTML标签
        text = re.sub(r'<[^>]+>', '', str(text))

        # 移除特殊字符和数字
        text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z\s]', ' ', text)

        # 移除多余空格
        text = re.sub(r'\s+', ' ', text).strip()

        return text

    def extract_task_pairs(self, text: str) -> list:
        """
        从文本中提取动词-名词对
        """
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
            print(f"Error processing text: {e}")

        return pairs

    def process_sample_data(self, data_path: str, sample_size: int = 1000):
        """
        处理样本数据
        """
        print(f"Loading sample data from {data_path}...")

        # 读取样本数据
        df = pd.read_csv(data_path, nrows=sample_size)
        print(f"Loaded {len(df)} rows of sample data")

        processed_count = 0
        start_time = time.time()

        for idx, row in df.iterrows():
            # 尝试使用不同列名
            text_col = None
            if 'cleaned_requirements' in row.index and pd.notna(row['cleaned_requirements']):
                text_col = 'cleaned_requirements'
            elif '职位描述' in row.index and pd.notna(row['职位描述']):
                text_col = '职位描述'

            if text_col:
                text = row[text_col]
                cleaned_text = self.clean_text(text)
                pairs = self.extract_task_pairs(cleaned_text)

                # 统计
                for verb, noun in pairs:
                    self.task_pairs[(verb, noun)] += 1
                    self.verb_freq[verb] += 1
                    self.noun_freq[noun] += 1

                processed_count += 1

                # 进度显示
                if processed_count % 100 == 0:
                    elapsed = time.time() - start_time
                    print(f"Processed {processed_count} texts, {processed_count/elapsed:.1f} texts/sec")
        elapsed = time.time() - start_time
        print(f"\nProcessing completed in {elapsed:.2f} seconds")
        print(f"Processed {processed_count} texts")
        print(f"Found {len(self.task_pairs)} unique task pairs")

    def save_results(self):
        """保存结果"""
        # 过滤低频任务对
        filtered_pairs = {pair: count for pair, count in self.task_pairs.items() if count >= 2}

        # 保存任务对词典
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

        output_file = self.output_dir / "task_pairs_dict_sample.pkl"
        with open(output_file, 'wb') as f:
            pickle.dump(result_dict, f)

        print(f"Results saved to: {output_file}")

        # 保存统计报告
        self.save_statistics_report()

    def save_statistics_report(self):
        """保存统计报告"""
        report_file = self.output_dir / "task_pairs_statistics_sample.txt"

        filtered_pairs = {pair: count for pair, count in self.task_pairs.items() if count >= 2}

        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("=== 任务对提取统计报告 (样本数据) ===\n\n")

            f.write(f"总任务对数量: {len(self.task_pairs)}\n")
            f.write(f"过滤后任务对数量: {len(filtered_pairs)}\n")
            f.write(f"唯一动词数量: {len(self.verb_freq)}\n")
            f.write(f"唯一名词数量: {len(self.noun_freq)}\n\n")

            f.write("=== Top 20 任务对 ===\n")
            for i, ((verb, noun), count) in enumerate(sorted(filtered_pairs.items(), key=lambda x: x[1], reverse=True)[:20]):
                f.write("2d")

            f.write("\n=== Top 20 动词 ===\n")
            for i, (verb, count) in enumerate(self.verb_freq.most_common(20)):
                f.write("2d")

            f.write("\n=== Top 20 名词 ===\n")
            for i, (noun, count) in enumerate(self.noun_freq.most_common(20)):
                f.write("2d")

        print(f"Statistics report saved to: {report_file}")

def main():
    """主函数"""
    data_path = "/Users/yu/code/code2601/TY/data/processed/cleaned/all_in_one1.csv"
    output_dir = "/Users/yu/code/code2601/TY/output/taskpair"

    # 创建提取器
    extractor = SingleThreadTaskPairExtractor(output_dir)

    # 处理样本数据
    extractor.process_sample_data(data_path, sample_size=5000)

    # 保存结果
    extractor.save_results()

    print("\n✅ Sample task pair extraction completed!")

if __name__ == "__main__":
    main()