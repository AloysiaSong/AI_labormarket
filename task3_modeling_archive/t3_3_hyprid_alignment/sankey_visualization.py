# 文件: task3_modeling/t3_3_hyprid_alignment/sankey_visualization.py

"""
功能：绘制技能演化桑基图
输入：evolution_events.csv + LDA模型(获取Topic标签)
输出：交互式HTML桑基图
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import pickle
from gensim.models import LdaModel

def load_lda_model(window_name):
    """加载指定窗口的LDA模型"""
    model_path = f"../../output/lda/models/{window_name}_lda.model"
    model = LdaModel.load(model_path)
    return model

def get_topic_labels(model, num_words=2, max_topics=None):
    """从LDA模型获取主题标签（改进版）"""
    # 计算主题的"独特性"分数（基于与其他主题的差异）
    if max_topics is None:
        max_topics = model.num_topics

    # 如果主题太多，只选择最具代表性的
    if model.num_topics > max_topics:
        # 计算主题权重（基于主题中词的概率分布）
        topic_weights = []
        for topic_id in range(model.num_topics):
            topic_terms = model.get_topic_terms(topic_id, topn=10)
            weight = sum(prob for _, prob in topic_terms)
            topic_weights.append((topic_id, weight))

        # 选择权重最高的主题
        topic_weights.sort(key=lambda x: x[1], reverse=True)
        selected_topics = [tid for tid, _ in topic_weights[:max_topics]]
    else:
        selected_topics = list(range(model.num_topics))

    topic_labels = {}
    for topic_id in range(model.num_topics):
        if topic_id in selected_topics:
            top_words = model.show_topic(topic_id, topn=num_words)
            # 过滤掉太短的词和常见词
            filtered_words = []
            for word, prob in top_words:
                if len(word) > 1 and prob > 0.01:  # 过滤条件
                    filtered_words.append(word)
                    if len(filtered_words) >= num_words:
                        break

            if filtered_words:
                label = " ".join(filtered_words)
                topic_labels[topic_id] = f"T{topic_id}: {label}"
            else:
                topic_labels[topic_id] = f"Topic {topic_id}"
        else:
            topic_labels[topic_id] = f"Topic {topic_id}"

    return topic_labels

def create_sankey_data(evolution_df, topic_labels_dict):
    """创建桑基图数据"""
    # 收集所有节点
    nodes = []
    node_indices = {}
    node_counter = 0

    # 时间窗口顺序
    windows = ['window_2016_2017', 'window_2018_2019', 'window_2020_2021', 'window_2022_2023', 'window_2024_2025']

    # 为每个窗口的每个主题创建节点
    for window in windows:
        if window in topic_labels_dict:
            labels = topic_labels_dict[window]
            for topic_id in range(60):  # 假设每个窗口都有60个主题
                if topic_id in labels:
                    label = labels[topic_id]
                else:
                    label = f"Topic {topic_id}"
                node_name = f"{window.split('_')[1]}-{window.split('_')[2]}<br>{label}"
                nodes.append(node_name)
                node_indices[(window, topic_id)] = node_counter
                node_counter += 1

    # 收集链接
    sources = []
    targets = []
    values = []
    link_labels = []
    colors = []

    # 处理Survival匹配
    survival_df = pd.read_csv("../../output/lda/alignment/alignment_matrix.csv")
    for _, row in survival_df.iterrows():
        base_window = row['base_window']
        target_window = row['target_window']
        base_topic = row['base_topic']
        target_topic = row['target_topic']
        similarity = row['similarity']

        if (base_window, base_topic) in node_indices and (target_window, target_topic) in node_indices:
            sources.append(node_indices[(base_window, base_topic)])
            targets.append(node_indices[(target_window, target_topic)])
            values.append(similarity * 100)  # 放大显示
            link_labels.append(f"Survival<br>相似度: {similarity:.3f}")
            colors.append("rgba(0, 128, 0, 0.6)")  # 绿色

    # 处理演化事件
    event_colors = {
        'split': 'rgba(255, 165, 0, 0.6)',  # 橙色
        'merge': 'rgba(0, 0, 255, 0.6)',   # 蓝色
        'birth': 'rgba(128, 0, 128, 0.6)', # 紫色
        'death': 'rgba(255, 0, 0, 0.6)'    # 红色
    }

    for _, row in evolution_df.iterrows():
        base_window = row['base_window']
        target_window = row['target_window']
        event_type = row['event_type']

        if event_type == 'split':
            base_topics = eval(row['base_topics'])
            target_topics = eval(row['target_topics'])
            similarities = eval(row['similarities'])

            for i, base_topic in enumerate(base_topics):
                for j, target_topic in enumerate(target_topics):
                    if (base_window, base_topic) in node_indices and (target_window, target_topic) in node_indices:
                        sources.append(node_indices[(base_window, base_topic)])
                        targets.append(node_indices[(target_window, target_topic)])
                        values.append(similarities[j] * 100)
                        link_labels.append(f"Split<br>相似度: {similarities[j]:.3f}")
                        colors.append(event_colors['split'])

        elif event_type == 'merge':
            base_topics = eval(row['base_topics'])
            target_topics = eval(row['target_topics'])
            similarities = eval(row['similarities'])

            for i, target_topic in enumerate(target_topics):
                for j, base_topic in enumerate(base_topics):
                    if (base_window, base_topic) in node_indices and (target_window, target_topic) in node_indices:
                        sources.append(node_indices[(base_window, base_topic)])
                        targets.append(node_indices[(target_window, target_topic)])
                        values.append(similarities[j] * 100)
                        link_labels.append(f"Merge<br>相似度: {similarities[j]:.3f}")
                        colors.append(event_colors['merge'])

        elif event_type == 'birth':
            target_topics = eval(row['target_topics'])
            for target_topic in target_topics:
                if (target_window, target_topic) in node_indices:
                    # Birth事件：从一个虚拟源节点连接
                    sources.append(len(nodes))  # 虚拟节点
                    targets.append(node_indices[(target_window, target_topic)])
                    values.append(50)  # 固定值
                    link_labels.append("Birth<br>新主题")
                    colors.append(event_colors['birth'])

        elif event_type == 'death':
            base_topics = eval(row['base_topics'])
            for base_topic in base_topics:
                if (base_window, base_topic) in node_indices:
                    # Death事件：连接到一个虚拟目标节点
                    sources.append(node_indices[(base_window, base_topic)])
                    targets.append(len(nodes) + 1)  # 虚拟节点
                    values.append(50)  # 固定值
                    link_labels.append("Death<br>消亡主题")
                    colors.append(event_colors['death'])

    # 添加虚拟节点用于Birth和Death
    nodes.append("新主题源")
    nodes.append("消亡汇")

    return nodes, sources, targets, values, link_labels, colors

def create_sankey_diagram():
    """创建桑基图"""
    print("🔄 加载演化事件数据...")
    evolution_df = pd.read_csv("../../output/lda/alignment/evolution_events.csv")

    print("🔄 加载LDA模型获取主题标签...")
    windows = ['window_2016_2017', 'window_2018_2019', 'window_2020_2021', 'window_2022_2023', 'window_2024_2025']
    topic_labels_dict = {}

    for window in windows:
        try:
            model = load_lda_model(window)
            # 只显示最重要的30个主题，避免图表过于拥挤
            labels = get_topic_labels(model, num_words=2, max_topics=30)
            topic_labels_dict[window] = labels
            detailed_labels = len([l for l in labels.values() if ':' in l])
            print(f"  ✅ 加载 {window}: {detailed_labels}/60 个详细标签 (其余为简短标签)")
        except Exception as e:
            print(f"  ❌ 加载 {window} 失败: {e}")
            # 使用默认标签
            topic_labels_dict[window] = {i: f"Topic {i}" for i in range(60)}

    print("🔄 创建桑基图数据...")
    nodes, sources, targets, values, link_labels, colors = create_sankey_data(evolution_df, topic_labels_dict)

    print("🔄 生成桑基图...")

    # 创建桑基图
    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=15,
            thickness=20,
            line=dict(color="black", width=0.5),
            label=nodes,
            color="lightblue"
        ),
        link=dict(
            source=sources,
            target=targets,
            value=values,
            label=link_labels,
            color=colors
        )
    )])

    fig.update_layout(
        title_text="工作主题演化桑基图 (2016-2025)<br>基于混合对齐算法的结果",
        font_size=10,
        width=1200,
        height=800
    )

    # 保存为HTML
    output_path = "../../output/lda/alignment/sankey_diagram.html"
    fig.write_html(output_path)
    print(f"💾 桑基图已保存至: {output_path}")

    # 显示图表
    fig.show()

if __name__ == "__main__":
    print("🎯 Task 3.3: 桑基图可视化")
    print("=" * 50)

    create_sankey_diagram()

    print("\n✅ Task 3.3 完成！")
    print("📊 桑基图已生成，可在浏览器中查看交互式可视化")