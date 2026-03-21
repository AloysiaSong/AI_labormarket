#!/usr/bin/env python3
"""
智能任务对过滤器 - 使用语义分析过滤固定搭配
"""

import pickle
import re
import jieba
import jieba.posseg as pseg
from pathlib import Path
from collections import defaultdict, Counter

class SmartTaskPairFilter:
    def __init__(self):
        # 加载语料库来分析词语共现模式
        self.load_corpus_patterns()

        # 语义相似度阈值
        self.similarity_threshold = 0.3

        # 上下文窗口大小
        self.context_window = 3

    def load_corpus_patterns(self):
        """从职位描述中学习词语共现模式"""
        try:
            # 尝试加载之前保存的模式数据
            with open('/Users/yu/code/code2601/TY/output/taskpair/corpus_patterns.pkl', 'rb') as f:
                self.corpus_patterns = pickle.load(f)
        except:
            print("未找到预计算的语料库模式，将使用规则-based过滤")
            self.corpus_patterns = None

    def analyze_pair_coherence(self, verb, noun, context_data=None):
        """
        分析动词-名词对的语义连贯性

        Args:
            verb: 动词
            noun: 名词
            context_data: 上下文数据（可选）

        Returns:
            float: 连贯性分数 (0-1, 1表示高度连贯)
        """
        # 1. 词长检查 - 太短的词通常不是有意义的搭配
        if len(verb) < 2 or len(noun) < 2:
            return 0.1

        # 2. 词性组合合理性检查
        verb_pos_patterns = ['v', 'vn', 'vg']  # 动词相关词性
        noun_pos_patterns = ['n', 'nr', 'ns', 'nt', 'nz']  # 名词相关词性

        # 使用jieba分析词性
        verb_words = pseg.cut(verb)
        noun_words = pseg.cut(noun)

        verb_pos = None
        noun_pos = None

        for word, pos in verb_words:
            if pos.startswith('v'):
                verb_pos = pos
                break

        for word, pos in noun_words:
            if pos.startswith('n'):
                noun_pos = pos
                break

        # 如果无法确定词性，给较低分数
        if not verb_pos or not noun_pos:
            return 0.3

        # 3. 语义关联度分析
        # 检查是否是常见的管理/工作动词搭配常见对象
        management_verbs = {'管理', '负责', '领导', '主管', '协调', '组织'}
        work_verbs = {'工作', '从事', '担任', '负责', '参与'}
        common_nouns = {'团队', '项目', '客户', '公司', '企业', '部门', '员工'}

        if verb in management_verbs and noun in common_nouns:
            return 0.9  # 高连贯性

        if verb in work_verbs and noun in {'经验', '能力', '技能'}:
            return 0.8

        # 4. 固定搭配检测
        fixed_pairs = {
            ('工作', '经验'): 0.1,
            ('沟通', '能力'): 0.1,
            ('管理', '经验'): 0.1,
            ('销售', '经验'): 0.1,
            ('项目', '经验'): 0.1,
            ('任职', '资格'): 0.1,
            ('工作', '时间'): 0.1,
            ('工作', '地点'): 0.1,
        }

        if (verb, noun) in fixed_pairs:
            return fixed_pairs[(verb, noun)]

        # 5. 默认中等连贯性
        return 0.5

    def filter_by_semantic_coherence(self, task_pairs_dict, coherence_threshold=0.4):
        """
        基于语义连贯性过滤任务对

        Args:
            task_pairs_dict: 任务对词典
            coherence_threshold: 连贯性阈值

        Returns:
            dict: 过滤后的任务对词典
        """
        filtered_pairs = {}
        removed_pairs = {}

        print(f"基于语义连贯性过滤 (阈值: {coherence_threshold})...")

        for i, ((verb, noun), count) in enumerate(task_pairs_dict.items()):
            if i % 10000 == 0:
                print(f"处理进度: {i}/{len(task_pairs_dict)}")

            coherence_score = self.analyze_pair_coherence(verb, noun)

            if coherence_score >= coherence_threshold:
                filtered_pairs[(verb, noun)] = count
            else:
                removed_pairs[(verb, noun)] = (count, coherence_score)

        print(f"语义过滤完成: 保留 {len(filtered_pairs)}, 移除 {len(removed_pairs)}")
        return filtered_pairs, removed_pairs

    def filter_by_frequency_distribution(self, task_pairs_dict, percentile_threshold=95):
        """
        基于频率分布过滤异常值

        Args:
            task_pairs_dict: 任务对词典
            percentile_threshold: 百分位数阈值

        Returns:
            dict: 过滤后的任务对词典
        """
        import numpy as np

        frequencies = list(task_pairs_dict.values())
        threshold = np.percentile(frequencies, percentile_threshold)

        print(f"频率分布过滤 (阈值: {percentile_threshold}百分位, 值: {threshold})...")

        filtered_pairs = {
            pair: count for pair, count in task_pairs_dict.items()
            if count <= threshold
        }

        removed_count = len(task_pairs_dict) - len(filtered_pairs)
        print(f"频率过滤完成: 保留 {len(filtered_pairs)}, 移除 {removed_count}")

        return filtered_pairs

    def comprehensive_filter(self, task_pairs_dict, coherence_threshold=0.4, frequency_percentile=99):
        """
        综合过滤：结合多种策略

        Args:
            task_pairs_dict: 原始任务对词典
            coherence_threshold: 语义连贯性阈值
            frequency_percentile: 频率百分位数阈值

        Returns:
            dict: 过滤后的任务对词典
        """
        print("开始综合过滤...")

        # 第一步：语义连贯性过滤
        semantic_filtered, semantic_removed = self.filter_by_semantic_coherence(
            task_pairs_dict, coherence_threshold
        )

        # 第二步：频率分布过滤
        final_filtered = self.filter_by_frequency_distribution(
            semantic_filtered, frequency_percentile
        )

        # 统计信息
        stats = {
            'original_count': len(task_pairs_dict),
            'after_semantic_filter': len(semantic_filtered),
            'final_count': len(final_filtered),
            'semantic_removed': len(semantic_removed),
            'frequency_removed': len(semantic_filtered) - len(final_filtered)
        }

        print("\n综合过滤统计:")
        print(f"原始任务对: {stats['original_count']}")
        print(f"语义过滤后: {stats['after_semantic_filter']} (移除 {stats['semantic_removed']})")
        print(f"频率过滤后: {stats['final_count']} (移除 {stats['frequency_removed']})")
        total_removed = stats['semantic_removed'] + stats['frequency_removed']
        print(f"总移除比例: {total_removed/stats['original_count']*100:.1f}%")
        return final_filtered, stats

def main():
    """主函数：智能过滤任务对"""
    # 加载数据
    data_file = '/Users/yu/code/code2601/TY/output/taskpair/task_pairs_dict_filtered_final.pkl'
    output_file = '/Users/yu/code/code2601/TY/output/taskpair/task_pairs_dict_smart_filtered.pkl'

    print("加载过滤后的任务对数据...")
    with open(data_file, 'rb') as f:
        data = pickle.load(f)

    print(f"加载数据: {len(data['task_pairs'])} 个任务对")

    # 创建智能过滤器
    smart_filter = SmartTaskPairFilter()

    # 综合过滤
    filtered_pairs, stats = smart_filter.comprehensive_filter(
        data['task_pairs'],
        coherence_threshold=0.4,  # 语义连贯性阈值
        frequency_percentile=99   # 频率百分位数
    )

    # 重新计算动词和名词频率
    verb_freq = Counter()
    noun_freq = Counter()

    for (verb, noun), count in filtered_pairs.items():
        verb_freq[verb] += count
        noun_freq[noun] += count

    # 保存结果
    result_data = {
        'task_pairs': filtered_pairs,
        'verb_freq': dict(verb_freq),
        'noun_freq': dict(noun_freq),
        'metadata': {
            'original_total_pairs': len(data['task_pairs']),
            'filtered_total_pairs': len(filtered_pairs),
            'total_verbs': len(verb_freq),
            'total_nouns': len(noun_freq),
            'filter_stats': stats,
            'filter_type': 'smart_semantic_filter'
        }
    }

    print(f"保存智能过滤结果到: {output_file}")
    with open(output_file, 'wb') as f:
        pickle.dump(result_data, f)

    # 显示Top 10结果
    print("\n=== 智能过滤后的Top 10任务对 ===")
    sorted_pairs = sorted(filtered_pairs.items(), key=lambda x: x[1], reverse=True)
    for i, ((verb, noun), count) in enumerate(sorted_pairs[:10]):
        print(f"{i+1}. {verb}-{noun}: {count}")

if __name__ == "__main__":
    main()