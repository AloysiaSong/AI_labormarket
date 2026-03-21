#!/usr/bin/env python3
"""
任务对过滤器 - 过滤固定搭配和不合理的动词-名词组合
"""

import pickle
import re
from pathlib import Path

class TaskPairFilter:
    def __init__(self):
        # 常见的固定搭配词典（扩展版）
        self.fixed_collocations = {
            # 工作相关
            ('工作', '经验'): '工作经验',
            ('工作', '时间'): '工作时间',
            ('工作', '内容'): '工作内容',
            ('工作', '地点'): '工作地点',
            ('工作', '职责'): '工作职责',
            ('工作', '要求'): '工作要求',
            ('工作', '性质'): '工作性质',
            ('工作', '环境'): '工作环境',
            ('工作', '条件'): '工作条件',
            ('工作', '压力'): '工作压力',
            ('工作', '岗位'): '工作岗位',

            # 能力相关
            ('沟通', '能力'): '沟通能力',
            ('协调', '能力'): '协调能力',
            ('学习', '能力'): '学习能力',
            ('分析', '能力'): '分析能力',
            ('解决', '能力'): '解决能力',
            ('创新', '能力'): '创新能力',
            ('适应', '能力'): '适应能力',
            ('执行', '能力'): '执行能力',
            ('管理', '能力'): '管理能力',

            # 经验相关
            ('管理', '经验'): '管理经验',
            ('销售', '经验'): '销售经验',
            ('项目', '经验'): '项目经验',
            ('技术', '经验'): '技术经验',
            ('行业', '经验'): '行业经验',

            # 资格相关
            ('任职', '资格'): '任职资格',
            ('专业', '资格'): '专业资格',
            ('资格', '证书'): '资格证书',

            # 团队相关
            ('合作', '团队'): '合作团队',
            ('协作', '团队'): '协作团队',
            ('团队', '合作'): '团队合作',
            ('团队', '精神'): '团队精神',
            ('合作', '精神'): '合作精神',
            ('协作', '精神'): '协作精神',
            ('管理', '团队'): '管理团队',

            # 意识相关
            ('服务', '意识'): '服务意识',
            ('质量', '意识'): '质量意识',
            ('安全', '意识'): '安全意识',
            ('责任', '意识'): '责任意识',

            # 毕业生相关
            ('应届', '毕业生'): '应届毕业生',

            # 五险一金相关
            ('缴纳', '五险'): '缴纳五险',

            # 销售相关
            ('销售', '目标'): '销售目标',

            # 职位相关
            ('描述', '职位'): '职位描述',

            # 领导相关
            ('交办', '领导'): '领导交办',

            # 公司类型相关（这些通常是固定搭配）
            ('管理', '有限公司'): '管理有限公司',
            ('服务', '有限公司'): '服务有限公司',
            ('咨询', '有限公司'): '咨询有限公司',
            ('管理', '企业'): '管理企业',
            ('教育', '有限公司'): '教育有限公司',
            ('发展', '有限公司'): '发展有限公司',
            ('投资', '有限公司'): '投资有限公司',
            ('销售', '有限公司'): '销售有限公司',
            ('服务', '人力资源'): '人力资源服务',
            ('咨询', '企业'): '企业咨询',

            # 其他高频但可能是固定搭配的组合
            ('技术', '有限公司'): '技术有限公司',
            ('贸易', '有限公司'): '贸易有限公司',
            ('建设', '有限公司'): '建设有限公司',
            ('房地产', '有限公司'): '房地产有限公司',
            ('建筑', '有限公司'): '建筑有限公司',
            ('工程', '有限公司'): '工程有限公司',
            ('制造', '有限公司'): '制造有限公司',
            ('科技', '有限公司'): '科技有限公司',
            ('信息', '有限公司'): '信息有限公司',
            ('网络', '有限公司'): '网络有限公司',
        }

        # 反向映射（处理顺序颠倒的情况）
        self.reverse_collocations = {v: k for k, v in self.fixed_collocations.items()}

        # 不合理的组合模式
        self.invalid_patterns = [
            # 动词+时间词
            (r'^v.*', r'^t.*$'),  # 动词+时间词
            # 动词+数词
            (r'^v.*', r'^m.*$'),  # 动词+数词
            # 动词+量词
            (r'^v.*', r'^q.*$'),  # 动词+量词
            # 代词+名词
            (r'^r.*', r'^n.*$'),  # 代词+名词
        ]

    def is_fixed_collocation(self, verb, noun):
        """检查是否是固定搭配"""
        return (verb, noun) in self.fixed_collocations

    def is_invalid_pattern(self, verb_pos, noun_pos):
        """检查是否是不合理的词性组合"""
        for verb_pattern, noun_pattern in self.invalid_patterns:
            if re.match(verb_pattern, verb_pos) and re.match(noun_pattern, noun_pos):
                return True
        return False

    def should_filter_pair(self, verb, noun, verb_pos=None, noun_pos=None, min_freq=2):
        """
        判断是否应该过滤这个任务对

        Args:
            verb: 动词
            noun: 名词
            verb_pos: 动词词性标注（可选）
            noun_pos: 名词词性标注（可选）
            min_freq: 最小频率阈值

        Returns:
            bool: True表示应该过滤掉
        """
        # 1. 检查固定搭配
        if self.is_fixed_collocation(verb, noun):
            return True

        # 2. 检查词性模式（如果提供了词性信息）
        if verb_pos and noun_pos:
            if self.is_invalid_pattern(verb_pos, noun_pos):
                return True

        # 3. 检查语义合理性（简单的启发式规则）
        # 太短的词组合
        if len(verb) < 2 or len(noun) < 2:
            return True

        # 包含数字的组合
        if any(char.isdigit() for char in verb + noun):
            return True

        # 包含特殊字符的组合
        if any(char in '()[]{}【】《》' for char in verb + noun):
            return True

        return False

    def filter_task_pairs(self, task_pairs_dict, verb_pos_dict=None, noun_pos_dict=None):
        """
        过滤任务对词典

        Args:
            task_pairs_dict: 任务对词典 {('verb', 'noun'): count}
            verb_pos_dict: 动词词性词典（可选）
            noun_pos_dict: 名词词性词典（可选）

        Returns:
            dict: 过滤后的任务对词典
        """
        filtered_pairs = {}
        filtered_count = 0

        for (verb, noun), count in task_pairs_dict.items():
            # 获取词性信息
            verb_pos = verb_pos_dict.get(verb, 'v') if verb_pos_dict else 'v'
            noun_pos = noun_pos_dict.get(noun, 'n') if noun_pos_dict else 'n'

            # 检查是否应该过滤
            if not self.should_filter_pair(verb, noun, verb_pos, noun_pos):
                filtered_pairs[(verb, noun)] = count
            else:
                filtered_count += 1

        print(f"过滤了 {filtered_count} 个固定搭配或不合理组合")
        print(f"保留了 {len(filtered_pairs)} 个有效任务对")

        return filtered_pairs

    def get_filtered_statistics(self, original_data, filtered_pairs):
        """生成过滤后的统计信息"""
        # 重新计算动词和名词频率
        verb_freq = {}
        noun_freq = {}

        for (verb, noun), count in filtered_pairs.items():
            verb_freq[verb] = verb_freq.get(verb, 0) + count
            noun_freq[noun] = noun_freq.get(noun, 0) + count

        # 生成统计报告
        stats = {
            'original_total_pairs': len(original_data['task_pairs']),
            'filtered_total_pairs': len(filtered_pairs),
            'removed_pairs': len(original_data['task_pairs']) - len(filtered_pairs),
            'original_verbs': len(original_data['verb_freq']),
            'filtered_verbs': len(verb_freq),
            'original_nouns': len(original_data['noun_freq']),
            'filtered_nouns': len(noun_freq),
            'filtered_pairs': filtered_pairs,
            'filtered_verb_freq': verb_freq,
            'filtered_noun_freq': noun_freq
        }

        return stats

def main():
    """主函数：过滤任务对数据"""
    # 加载完整数据集（final文件包含所有数据）
    data_file = '/Users/yu/code/code2601/TY/output/taskpair/task_pairs_dict_final.pkl'
    output_file = '/Users/yu/code/code2601/TY/output/taskpair/task_pairs_dict_filtered_final.pkl'

    print("加载完整任务对数据...")
    with open(data_file, 'rb') as f:
        original_data = pickle.load(f)

    print(f"完整数据: {len(original_data['task_pairs'])} 个任务对")
    print(f"总文本数: {original_data['metadata']['total_texts']}")

    # 创建过滤器
    filter = TaskPairFilter()

    # 过滤任务对
    print("开始过滤固定搭配...")
    filtered_pairs = filter.filter_task_pairs(original_data['task_pairs'])

    # 生成统计信息
    stats = filter.get_filtered_statistics(original_data, filtered_pairs)

    # 保存过滤后的数据
    filtered_data = {
        'task_pairs': filtered_pairs,
        'verb_freq': stats['filtered_verb_freq'],
        'noun_freq': stats['filtered_noun_freq'],
        'metadata': {
            'original_total_pairs': stats['original_total_pairs'],
            'filtered_total_pairs': stats['filtered_total_pairs'],
            'removed_pairs': stats['removed_pairs'],
            'original_verbs': stats['original_verbs'],
            'filtered_verbs': stats['filtered_verbs'],
            'original_nouns': stats['original_nouns'],
            'filtered_nouns': stats['filtered_nouns'],
            'filter_type': 'fixed_collocations_filter'
        }
    }

    print(f"保存过滤后的数据到: {output_file}")
    with open(output_file, 'wb') as f:
        pickle.dump(filtered_data, f)

    # 显示Top 10过滤后的任务对
    print("\n=== 过滤后的Top 10任务对 ===")
    sorted_pairs = sorted(filtered_pairs.items(), key=lambda x: x[1], reverse=True)
    for i, ((verb, noun), count) in enumerate(sorted_pairs[:10]):
        print(f"{i+1}. {verb}-{noun}: {count}")

    print("\n过滤统计:")
    print(f"原始任务对: {stats['original_total_pairs']}")
    print(f"过滤后任务对: {stats['filtered_total_pairs']}")
    print(f"移除数量: {stats['removed_pairs']}")
    print(f"移除比例: {stats['removed_pairs']/stats['original_total_pairs']*100:.1f}%")

if __name__ == "__main__":
    main()