# 文件: task3_modeling/t3_3_hyprid_alignment/job_integration_analysis.py

"""
功能：量化岗位综合化趋势分析
输入：演化事件数据
输出：岗位综合化指标和趋势分析
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pickle

def calculate_integration_index(evolution_events: pd.DataFrame) -> dict:
    """
    计算岗位综合化指数
    综合化指数 = (合并事件数 - 分裂事件数) / 总事件数
    正值表示综合化趋势增强，负值表示细分化趋势增强
    """

    # 按时间窗口分组统计
    window_stats = {}

    # 获取所有唯一的窗口对
    window_pairs = []
    for _, row in evolution_events.iterrows():
        pair = f"{row['window_from']}_{row['window_to']}"
        if pair not in window_pairs:
            window_pairs.append(pair)

    for window_pair in window_pairs:
        window_from, window_to = window_pair.split('_')[0] + '_' + window_pair.split('_')[1], \
                               window_pair.split('_')[2] + '_' + window_pair.split('_')[3]

        window_data = evolution_events[
            (evolution_events['window_from'] == window_from) &
            (evolution_events['window_to'] == window_to)
        ]

        merge_count = len(window_data[window_data['event_type'] == 'merge'])
        split_count = len(window_data[window_data['event_type'] == 'split'])
        total_events = len(window_data)

        # 综合化指数
        integration_index = (merge_count - split_count) / total_events if total_events > 0 else 0

        window_stats[window_pair] = {
            'merge_events': merge_count,
            'split_events': split_count,
            'total_events': total_events,
            'integration_index': integration_index,
            'survival_rate': len(window_data[window_data['event_type'] == 'survival']) / total_events if total_events > 0 else 0
        }

    return window_stats

def analyze_ai_impact_trends(integration_stats: dict) -> dict:
    """
    分析AI影响下的综合化趋势
    """

    # 时间序列分析
    windows = list(integration_stats.keys())
    integration_indices = [stats['integration_index'] for stats in integration_stats.values()]

    # 计算趋势
    if len(integration_indices) >= 2:
        trend_slope = np.polyfit(range(len(integration_indices)), integration_indices, 1)[0]
    else:
        trend_slope = 0

    # 识别关键转折点
    turning_points = []
    for i in range(1, len(integration_indices)):
        if abs(integration_indices[i] - integration_indices[i-1]) > 0.1:  # 显著变化阈值
            turning_points.append({
                'window': windows[i],
                'change': integration_indices[i] - integration_indices[i-1],
                'direction': '更综合化' if integration_indices[i] > integration_indices[i-1] else '更细分化'
            })

    return {
        'overall_trend': '综合化增强' if trend_slope > 0 else '细分化增强',
        'trend_slope': trend_slope,
        'turning_points': turning_points,
        'ai_era_integration': integration_indices[-1] if integration_indices else 0
    }

def create_integration_visualization(integration_stats: dict,
                                   trend_analysis: dict,
                                   output_file: str = 'job_integration_trends.html'):
    """
    创建岗位综合化趋势可视化
    """

    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    windows = list(integration_stats.keys())
    indices = [stats['integration_index'] for stats in integration_stats.values()]

    # 创建子图
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=['岗位综合化指数趋势', '事件类型分布'],
        vertical_spacing=0.1
    )

    # 综合化指数趋势
    fig.add_trace(
        go.Scatter(
            x=windows,
            y=indices,
            mode='lines+markers',
            name='综合化指数',
            line=dict(color='red', width=3),
            marker=dict(size=8)
        ),
        row=1, col=1
    )

    # 添加趋势线
    if len(indices) >= 2:
        z = np.polyfit(range(len(indices)), indices, 1)
        p = np.poly1d(z)
        trend_line = p(range(len(indices)))

        fig.add_trace(
            go.Scatter(
                x=windows,
                y=trend_line,
                mode='lines',
                name='趋势线',
                line=dict(color='blue', dash='dash')
            ),
            row=1, col=1
        )

    # 事件类型分布
    merge_counts = [stats['merge_events'] for stats in integration_stats.values()]
    split_counts = [stats['split_events'] for stats in integration_stats.values()]

    fig.add_trace(
        go.Bar(
            x=windows,
            y=merge_counts,
            name='合并事件',
            marker_color='green'
        ),
        row=2, col=1
    )

    fig.add_trace(
        go.Bar(
            x=windows,
            y=split_counts,
            name='分裂事件',
            marker_color='orange'
        ),
        row=2, col=1
    )

    # 更新布局
    fig.update_layout(
        title="AI时代岗位综合化趋势分析",
        height=800,
        showlegend=True
    )

    # 添加趋势分析注释
    trend_text = f"""
    <b>趋势分析结果：</b><br>
    • 整体趋势：{trend_analysis['overall_trend']}<br>
    • 趋势斜率：{trend_analysis['trend_slope']:.4f}<br>
    • AI时代综合化水平：{trend_analysis['ai_era_integration']:.3f}<br>
    """

    if trend_analysis['turning_points']:
        trend_text += "<b>关键转折点：</b><br>"
        for tp in trend_analysis['turning_points']:
            trend_text += f"• {tp['window']}: {tp['direction']} ({tp['change']:.3f})<br>"

    fig.add_annotation(
        text=trend_text,
        xref="paper", yref="paper",
        x=0.02, y=0.98,
        showarrow=False,
        align="left",
        bgcolor="rgba(255,255,255,0.8)",
        bordercolor="black",
        borderwidth=1
    )

    fig.write_html(output_file)
    print(f"岗位综合化趋势图已保存: {output_file}")

def main():
    """主函数"""

    # 读取演化事件数据
    evolution_file = 'topic_evolution_events.csv'
    if not Path(evolution_file).exists():
        print(f"错误：找不到文件 {evolution_file}")
        return

    evolution_events = pd.read_csv(evolution_file)
    print(f"加载演化事件数据：{len(evolution_events)} 条记录")

    # 计算综合化指数
    integration_stats = calculate_integration_index(evolution_events)
    print("综合化指数计算完成：")
    for window, stats in integration_stats.items():
        print(f"  {window}: 综合化指数 = {stats['integration_index']:.3f}")

    # 分析AI影响趋势
    trend_analysis = analyze_ai_impact_trends(integration_stats)
    print("\\nAI影响趋势分析：")
    print(f"  整体趋势：{trend_analysis['overall_trend']}")
    print(f"  趋势斜率：{trend_analysis['trend_slope']:.4f}")

    # 创建可视化
    create_integration_visualization(integration_stats, trend_analysis)

    # 保存分析结果
    results = {
        'integration_stats': integration_stats,
        'trend_analysis': trend_analysis,
        'evolution_events_count': len(evolution_events)
    }

    with open('job_integration_analysis.pkl', 'wb') as f:
        pickle.dump(results, f)

    print("\\n岗位综合化分析完成！")
    print("📁 输出文件：")
    print("  - job_integration_trends.html")
    print("  - job_integration_analysis.pkl")

if __name__ == "__main__":
    main()