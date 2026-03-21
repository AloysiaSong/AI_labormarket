import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pickle

def create_comprehensive_integration_analysis():
    """创建综合的岗位综合化趋势分析图"""

    # 读取数据
    df = pd.read_csv('topic_evolution_events.csv')

    # 定义时间窗口
    windows = [
        ('window_2016_2017', 'window_2018_2019'),
        ('window_2018_2019', 'window_2020_2021'),
        ('window_2020_2021', 'window_2022_2023'),
        ('window_2022_2023', 'window_2024_2025')
    ]

    # 收集数据
    window_labels = ['2016-2017→2018-2019', '2018-2019→2020-2021',
                     '2020-2021→2022-2023', '2022-2023→2024-2025']
    integration_indices = []
    merge_counts = []
    split_counts = []
    survival_rates = []

    for w_from, w_to in windows:
        subset = df[(df['window_from'] == w_from) & (df['window_to'] == w_to)]

        merge_count = len(subset[subset['event_type'] == 'merge'])
        split_count = len(subset[subset['event_type'] == 'split'])
        survival_count = len(subset[subset['event_type'] == 'survival'])
        total = len(subset)

        integration_index = (merge_count - split_count) / total if total > 0 else 0
        survival_rate = survival_count / total if total > 0 else 0

        integration_indices.append(integration_index)
        merge_counts.append(merge_count)
        split_counts.append(split_count)
        survival_rates.append(survival_rate)

    # 创建综合图表
    fig = make_subplots(
        rows=3, cols=1,
        subplot_titles=['岗位综合化指数趋势', '事件类型分布', '生存率趋势'],
        vertical_spacing=0.1
    )

    # 1. 综合化指数趋势
    fig.add_trace(
        go.Scatter(
            x=window_labels,
            y=integration_indices,
            mode='lines+markers',
            name='综合化指数',
            line=dict(color='red', width=4),
            marker=dict(size=10, symbol='diamond')
        ),
        row=1, col=1
    )

    # 添加趋势线
    z = np.polyfit(range(len(integration_indices)), integration_indices, 1)
    p = np.poly1d(z)
    trend_line = p(range(len(integration_indices)))

    fig.add_trace(
        go.Scatter(
            x=window_labels,
            y=trend_line,
            mode='lines',
            name='趋势线',
            line=dict(color='blue', dash='dash', width=2)
        ),
        row=1, col=1
    )

    # 2. 事件类型分布
    fig.add_trace(
        go.Bar(
            x=window_labels,
            y=merge_counts,
            name='合并事件',
            marker_color='green',
            offsetgroup=0
        ),
        row=2, col=1
    )

    fig.add_trace(
        go.Bar(
            x=window_labels,
            y=split_counts,
            name='分裂事件',
            marker_color='orange',
            offsetgroup=0
        ),
        row=2, col=1
    )

    # 3. 生存率趋势
    fig.add_trace(
        go.Scatter(
            x=window_labels,
            y=survival_rates,
            mode='lines+markers',
            name='生存率',
            line=dict(color='purple', width=3),
            marker=dict(size=8)
        ),
        row=3, col=1
    )

    # 更新布局
    fig.update_layout(
        title="AI时代岗位综合化趋势综合分析",
        height=900,
        showlegend=True
    )

    # 添加分析注释
    analysis_text = """
    <b>📊 关键发现：</b><br>
    • 整体趋势：岗位综合化程度随时间增强 (斜率=0.0194)<br>
    • 疫情前：细分化趋势明显<br>
    • 疫情后：综合化趋势加速<br>
    • 稳定性：84.5%的生存事件确保演化连续性<br><br>
    <b>🔍 AI影响机制：</b><br>
    • 技术替代：AI接管重复性工作<br>
    • 跨界融合：打破行业壁垒<br>
    • 效率优化：人类负责综合判断<br>
    • 疫情加速：数字化转型推动
    """

    fig.add_annotation(
        text=analysis_text,
        xref="paper", yref="paper",
        x=0.02, y=0.98,
        showarrow=False,
        align="left",
        bgcolor="rgba(255,255,255,0.9)",
        bordercolor="black",
        borderwidth=1,
        font=dict(size=10)
    )

    # 保存主图表
    fig.write_html('comprehensive_integration_analysis.html')
    print("综合分析图表已保存: comprehensive_integration_analysis.html")

    # 创建单独的雷达图
    fig_radar = go.Figure()

    categories = ['综合化指数', '生存率', '合并率', '分裂率']

    stage_features = {
        '细分化期 (2016-2019)': [-0.049, 0.854, 0.049, 0.098],
        '平衡期 (2018-2021)': [0.000, 0.762, 0.119, 0.119],
        '综合化期 (2020-2023)': [0.048, 0.857, 0.095, 0.048],
        '稳定期 (2022-2025)': [0.000, 0.907, 0.047, 0.047]
    }

    colors = ['red', 'orange', 'green', 'blue']
    for i, (stage, values) in enumerate(stage_features.items()):
        fig_radar.add_trace(
            go.Scatterpolar(
                r=values,
                theta=categories,
                fill='toself',
                name=stage,
                line_color=colors[i]
            )
        )

    fig_radar.update_layout(
        title="岗位综合化各阶段特征雷达图",
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[-0.1, 1.0]
            )
        )
    )

    fig_radar.write_html('integration_stages_radar.html')
    print("雷达图已保存: integration_stages_radar.html")

    # 保存分析结果
    results = {
        'window_labels': window_labels,
        'integration_indices': integration_indices,
        'merge_counts': merge_counts,
        'split_counts': split_counts,
        'survival_rates': survival_rates,
        'trend_slope': float(z[0]),
        'overall_trend': '综合化增强' if z[0] > 0 else '细分化增强'
    }

    with open('comprehensive_integration_analysis.pkl', 'wb') as f:
        pickle.dump(results, f)

    print("分析结果已保存: comprehensive_integration_analysis.pkl")

if __name__ == '__main__':
    create_comprehensive_integration_analysis()

if __name__ == '__main__':
    create_comprehensive_integration_analysis()