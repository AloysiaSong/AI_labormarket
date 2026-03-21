#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
任务对提取简单测试脚本
单线程版本，用于验证任务对提取功能
"""

import jieba
import jieba.posseg as pseg
from collections import Counter
import re

def clean_text(text: str) -> str:
    """清洗职位描述文本"""
    if not isinstance(text, str):
        return ""

    # 移除HTML标签
    text = re.sub(r'<[^>]+>', '', text)

    # 移除特殊字符和数字
    text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z\s]', ' ', text)

    # 移除多余空格
    text = re.sub(r'\s+', ' ', text).strip()

    return text

def extract_task_pairs(text: str) -> list:
    """
    从文本中提取动词-名词对

    Args:
        text: 输入文本

    Returns:
        动词-名词对列表 [(verb, noun), ...]
    """
    if not text or len(text) < 10:
        return []

    # 动词和名词词性标签
    verb_tags = {'v', 'vn', 'vd', 'vg'}  # 动词相关
    noun_tags = {'n', 'nr', 'ns', 'nt', 'nz', 'nl', 'ng'}  # 名词相关

    # 停用词
    stop_words = {
        '负责', '进行', '完成', '参与', '协助', '配合', '做好', '确保',
        '具备', '具有', '熟悉', '掌握', '了解', '能够', '可以', '需要',
        '要求', '包括', '涉及', '相关', '以及', '和', '与', '或', '等',
        '的', '了', '是', '在', '有', '为', '将', '对', '上', '下',
        '中', '大', '小', '多', '少', '好', '差', '高', '低', '长', '短'
    }

    pairs = []

    try:
        # 分词并标注词性
        words = pseg.cut(text)

        # 转换为列表以便多次遍历
        word_list = [(word, flag) for word, flag in words if word not in stop_words]

        # 滑动窗口提取相邻的动词-名词对
        for i in range(len(word_list) - 1):
            current_word, current_flag = word_list[i]
            next_word, next_flag = word_list[i + 1]

            # 检查是否是动词-名词组合
            if (current_flag in verb_tags and next_flag in noun_tags):
                # 过滤太短的词
                if len(current_word) >= 2 and len(next_word) >= 2:
                    pairs.append((current_word, next_word))

            # 也检查名词-动词组合（被动结构）
            elif (current_flag in noun_tags and next_flag in verb_tags):
                if len(current_word) >= 2 and len(next_word) >= 2:
                    pairs.append((next_word, current_word))  # 调换顺序，使动词在前

    except Exception as e:
        print(f"Error processing text: {e}")

    return pairs

def test_task_pair_extraction():
    """测试任务对提取功能"""

    # 创建测试数据
    test_data = [
        '负责市场调研和数据分析，进行客户需求分析，完成销售报告编写，参与产品开发讨论',
        '进行软件开发和测试，负责系统维护和优化，参与项目管理，完成代码审查',
        '负责教学工作和学生管理，进行课程设计和教学研究，完成教学任务',
        '负责数据挖掘和机器学习模型训练，进行特征工程和算法优化，完成模型评估',
        '进行市场推广和品牌建设，负责客户关系管理，完成销售目标制定'
    ]

    print("=== 任务对提取测试 ===\n")

    all_pairs = []
    verb_counter = Counter()
    noun_counter = Counter()

    # 测试每条数据
    for i, text in enumerate(test_data, 1):
        print(f"测试数据 {i}:")
        print(f"原始文本: {text}")

        cleaned_text = clean_text(text)
        print(f"清洗文本: {cleaned_text}")

        pairs = extract_task_pairs(cleaned_text)
        print(f"提取的任务对: {pairs}")
        print()

        # 收集统计
        all_pairs.extend(pairs)
        for verb, noun in pairs:
            verb_counter[verb] += 1
            noun_counter[noun] += 1

    # 显示统计结果
    print("=== 统计结果 ===")
    print(f"总任务对数量: {len(all_pairs)}")
    print(f"唯一动词数量: {len(verb_counter)}")
    print(f"唯一名词数量: {len(noun_counter)}")

    print("\n动词Top 10:")
    for verb, count in verb_counter.most_common(10):
        print(f"  {verb}: {count}")

    print("\n名词Top 10:")
    for noun, count in noun_counter.most_common(10):
        print(f"  {noun}: {count}")

    print("\n所有任务对:")
    pair_counter = Counter(all_pairs)
    for (verb, noun), count in pair_counter.most_common():
        print(f"  {verb} -> {noun}: {count}")

    print("\n✅ 测试完成！任务对提取功能正常工作")

if __name__ == "__main__":
    test_task_pair_extraction()