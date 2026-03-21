# 文件: task3_modeling/t3_3_hyprid_alignment/align_topics.py

"""
功能：跨时间窗口的主题对齐
算法：
  1. 匈牙利算法 -> 1-to-1 Survival
  2. 贪婪阈值法 -> Split/Merge检测
输出：演化事件表
"""

import os
import sys
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.metrics.pairwise import cosine_similarity
import pickle
from pathlib import Path
import logging

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TopicAligner:
    def __init__(self, base_path="../../"):
        self.base_path = base_path
        self.windows = ['window_2016_2017', 'window_2018_2019', 'window_2020_2021', 'window_2022_2023', 'window_2024_2025']
        self.survival_threshold = 0.7  # 生存主题相似度阈值
        self.split_merge_threshold = 0.5  # 分裂/合并相似度阈值

    def load_merged_vectors(self, window):
        """加载合并后的主题向量"""
        vector_path = f"{self.base_path}output/lda/merged_topic_vectors/{window}_merged_vectors.npy"
        return np.load(vector_path)

    def load_cluster_info(self, window):
        """加载聚类信息"""
        info_path = f"{self.base_path}output/lda/merged_topic_vectors/{window}_cluster_info.pkl"
        with open(info_path, 'rb') as f:
            return pickle.load(f)

    def hungarian_alignment(self, vectors_t1, vectors_t2):
        """
        使用匈牙利算法进行1-to-1主题对齐
        返回：对齐映射和相似度矩阵
        """

        # 计算相似度矩阵
        similarity_matrix = cosine_similarity(vectors_t1, vectors_t2)

        # 使用匈牙利算法找到最优匹配
        row_ind, col_ind = linear_sum_assignment(-similarity_matrix)  # 最大化相似度

        # 构建对齐映射
        alignment = {}
        for i, j in zip(row_ind, col_ind):
            similarity = similarity_matrix[i, j]
            if similarity >= self.survival_threshold:
                alignment[i] = {
                    'target_topic': j,
                    'similarity': similarity,
                    'event_type': 'survival'
                }

        return alignment, similarity_matrix

    def detect_split_merge_events(self, similarity_matrix, used_topics_t1, used_topics_t2):
        """
        使用贪婪阈值法检测分裂和合并事件
        """

        events = []

        # 找到未匹配的主题
        available_topics_t1 = [i for i in range(similarity_matrix.shape[0]) if i not in used_topics_t1]
        available_topics_t2 = [j for j in range(similarity_matrix.shape[1]) if j not in used_topics_t2]

        # 检测合并事件 (多个t1主题 -> 一个t2主题)
        for j in available_topics_t2:
            candidates = []
            for i in available_topics_t1:
                sim = similarity_matrix[i, j]
                if sim >= self.split_merge_threshold:
                    candidates.append((i, sim))

            if len(candidates) >= 2:  # 至少2个主题合并
                # 按相似度排序，取前几个
                candidates.sort(key=lambda x: x[1], reverse=True)
                merge_topics = candidates[:min(len(candidates), 3)]  # 最多3个主题合并

                events.append({
                    'event_type': 'merge',
                    'source_topics': [topic for topic, _ in merge_topics],
                    'target_topic': j,
                    'avg_similarity': np.mean([sim for _, sim in merge_topics]),
                    'topic_count': len(merge_topics)
                })

                # 标记已使用的主题
                for topic, _ in merge_topics:
                    used_topics_t1.add(topic)
                used_topics_t2.add(j)

        # 检测分裂事件 (一个t1主题 -> 多个t2主题)
        for i in available_topics_t1:
            candidates = []
            for j in available_topics_t2:
                sim = similarity_matrix[i, j]
                if sim >= self.split_merge_threshold:
                    candidates.append((j, sim))

            if len(candidates) >= 2:  # 至少分裂为2个主题
                # 按相似度排序，取前几个
                candidates.sort(key=lambda x: x[1], reverse=True)
                split_topics = candidates[:min(len(candidates), 3)]  # 最多分裂为3个主题

                events.append({
                    'event_type': 'split',
                    'source_topic': i,
                    'target_topics': [topic for topic, _ in split_topics],
                    'avg_similarity': np.mean([sim for _, sim in split_topics]),
                    'topic_count': len(split_topics)
                })

                # 标记已使用的主题
                used_topics_t1.add(i)
                for topic, _ in split_topics:
                    used_topics_t2.add(topic)

        return events

    def align_window_pair(self, window1, window2):
        """
        对齐两个相邻时间窗口的主题
        """

        logger.info(f"对齐窗口: {window1} -> {window2}")

        # 加载合并后的主题向量
        vectors1 = self.load_merged_vectors(window1)
        vectors2 = self.load_merged_vectors(window2)

        # 加载聚类信息
        cluster_info1 = self.load_cluster_info(window1)
        cluster_info2 = self.load_cluster_info(window2)

        logger.info(f"{window1}: {vectors1.shape[0]} 主题, {window2}: {vectors2.shape[0]} 主题")

        # 1. 匈牙利算法进行1-to-1对齐
        survival_alignment, similarity_matrix = self.hungarian_alignment(vectors1, vectors2)

        # 2. 检测分裂/合并事件
        used_topics_t1 = set(survival_alignment.keys())
        used_topics_t2 = set([info['target_topic'] for info in survival_alignment.values()])

        split_merge_events = self.detect_split_merge_events(
            similarity_matrix, used_topics_t1, used_topics_t2
        )

        # 统计结果
        result = {
            'window_pair': f"{window1}_{window2}",
            'survival_topics': len(survival_alignment),
            'split_events': len([e for e in split_merge_events if e['event_type'] == 'split']),
            'merge_events': len([e for e in split_merge_events if e['event_type'] == 'merge']),
            'survival_alignment': survival_alignment,
            'split_merge_events': split_merge_events,
            'total_events': len(survival_alignment) + len(split_merge_events)
        }

        logger.info(f"  生存主题: {result['survival_topics']}")
        logger.info(f"  分裂事件: {result['split_events']}")
        logger.info(f"  合并事件: {result['merge_events']}")

        return result

    def create_evolution_table(self, alignment_results):
        """
        创建主题演化事件表
        """

        evolution_events = []

        for result in alignment_results:
            window_pair = result['window_pair']
            # 正确解析窗口对，如 "window_2016_2017_window_2018_2019"
            parts = window_pair.split('_')
            window_from = f"{parts[0]}_{parts[1]}_{parts[2]}"
            window_to = f"{parts[3]}_{parts[4]}_{parts[5]}"

            # 添加生存事件
            for source_topic, alignment_info in result['survival_alignment'].items():
                evolution_events.append({
                    'window_from': window_from,
                    'window_to': window_to,
                    'event_type': 'survival',
                    'source_topics': [source_topic],
                    'target_topics': [alignment_info['target_topic']],
                    'similarity': alignment_info['similarity'],
                    'topic_count': 1
                })

            # 添加分裂/合并事件
            for event in result['split_merge_events']:
                if event['event_type'] == 'split':
                    evolution_events.append({
                        'window_from': window_from,
                        'window_to': window_to,
                        'event_type': 'split',
                        'source_topics': [event['source_topic']],
                        'target_topics': event['target_topics'],
                        'similarity': event['avg_similarity'],
                        'topic_count': event['topic_count']
                    })
                elif event['event_type'] == 'merge':
                    evolution_events.append({
                        'window_from': window_from,
                        'window_to': window_to,
                        'event_type': 'merge',
                        'source_topics': event['source_topics'],
                        'target_topics': [event['target_topic']],
                        'similarity': event['avg_similarity'],
                        'topic_count': event['topic_count']
                    })

        return pd.DataFrame(evolution_events)

    def analyze_evolution_patterns(self, evolution_df):
        """
        分析主题演化模式
        """

        # 统计不同类型事件的频率
        event_stats = evolution_df['event_type'].value_counts()

        # 计算生存率
        total_transitions = len(evolution_df)
        survival_count = event_stats.get('survival', 0)
        survival_rate = survival_count / total_transitions if total_transitions > 0 else 0

        # 分析时间趋势
        window_pairs = evolution_df['window_from'] + '_' + evolution_df['window_to']
        time_trends = {}

        for pair in window_pairs.unique():
            pair_data = evolution_df[window_pairs == pair]
            time_trends[pair] = {
                'total_events': len(pair_data),
                'survival_rate': len(pair_data[pair_data['event_type'] == 'survival']) / len(pair_data),
                'split_rate': len(pair_data[pair_data['event_type'] == 'split']) / len(pair_data),
                'merge_rate': len(pair_data[pair_data['event_type'] == 'merge']) / len(pair_data)
            }

        return {
            'event_statistics': event_stats.to_dict(),
            'overall_survival_rate': survival_rate,
            'time_trends': time_trends
        }

    def run(self):
        """主执行函数"""

        logger.info("🎯 开始主题对齐处理")

        # 对齐所有相邻窗口
        alignment_results = []
        for i in range(len(self.windows) - 1):
            result = self.align_window_pair(self.windows[i], self.windows[i+1])
            alignment_results.append(result)

        # 创建演化事件表
        evolution_df = self.create_evolution_table(alignment_results)

        # 分析演化模式
        evolution_analysis = self.analyze_evolution_patterns(evolution_df)

        # 保存结果
        output_dir = Path('./')
        output_dir.mkdir(exist_ok=True)

        # 保存演化事件表
        evolution_df.to_csv('topic_evolution_events.csv', index=False)

        # 保存对齐结果
        with open('alignment_results.pkl', 'wb') as f:
            pickle.dump(alignment_results, f)

        # 保存演化分析
        with open('evolution_analysis.pkl', 'wb') as f:
            pickle.dump(evolution_analysis, f)

        # 输出总结报告
        logger.info("\\n=== 主题演化分析报告 ===")
        logger.info(f"总演化事件数: {len(evolution_df)}")
        logger.info(f"生存事件: {evolution_analysis['event_statistics'].get('survival', 0)}")
        logger.info(f"分裂事件: {evolution_analysis['event_statistics'].get('split', 0)}")
        logger.info(f"合并事件: {evolution_analysis['event_statistics'].get('merge', 0)}")
        logger.info(f"总体生存率: {evolution_analysis['overall_survival_rate']:.3f}")
        # 时间趋势分析
        logger.info("\\n时间趋势:")
        for pair, stats in evolution_analysis['time_trends'].items():
            logger.info(f"  {pair}: 生存率 {stats['survival_rate']:.3f}")

        logger.info("\\n✅ 主题对齐处理完成")
        logger.info("📁 输出文件:")
        logger.info("  - topic_evolution_events.csv")
        logger.info("  - alignment_results.pkl")
        logger.info("  - evolution_analysis.pkl")

if __name__ == "__main__":
    aligner = TopicAligner()
    aligner.run()