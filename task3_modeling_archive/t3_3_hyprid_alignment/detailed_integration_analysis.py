import pandas as pd
import numpy as np

# 读取数据
df = pd.read_csv('topic_evolution_events.csv')

# 分析每个时间窗口的综合化趋势
windows = [
    ('window_2016_2017', 'window_2018_2019'),
    ('window_2018_2019', 'window_2020_2021'),
    ('window_2020_2021', 'window_2022_2023'),
    ('window_2022_2023', 'window_2024_2025')
]

print('=== 岗位综合化趋势详细分析 ===')
print()

for i, (w_from, w_to) in enumerate(windows):
    subset = df[(df['window_from'] == w_from) & (df['window_to'] == w_to)]

    merge_count = len(subset[subset['event_type'] == 'merge'])
    split_count = len(subset[subset['event_type'] == 'split'])
    survival_count = len(subset[subset['event_type'] == 'survival'])
    total = len(subset)

    # 计算综合化指数
    integration_index = (merge_count - split_count) / total if total > 0 else 0

    print(f'{w_from} -> {w_to}:')
    print(f'  生存事件: {survival_count} ({survival_count/total*100:.1f}%)')
    print(f'  合并事件: {merge_count} ({merge_count/total*100:.1f}%)')
    print(f'  分裂事件: {split_count} ({split_count/total*100:.1f}%)')
    print(f'  综合化指数: {integration_index:.3f}')

    # 解释趋势
    if merge_count > split_count:
        print('  趋势: 综合化增强 (合并多于分裂)')
    elif split_count > merge_count:
        print('  趋势: 细分化增强 (分裂多于合并)')
    else:
        print('  趋势: 平衡状态 (合并等于分裂)')
    print()

print('=== 总体趋势分析 ===')
print(f'总事件数: {len(df)}')
print(f'生存事件: {len(df[df["event_type"] == "survival"])} ({len(df[df["event_type"] == "survival"])/len(df)*100:.1f}%)')
print(f'合并事件: {len(df[df["event_type"] == "merge"])} ({len(df[df["event_type"] == "merge"])/len(df)*100:.1f}%)')
print(f'分裂事件: {len(df[df["event_type"] == "split"])} ({len(df[df["event_type"] == "split"])/len(df)*100:.1f}%)')

# 计算时间趋势
indices = []
for w_from, w_to in windows:
    subset = df[(df['window_from'] == w_from) & (df['window_to'] == w_to)]
    merge_count = len(subset[subset['event_type'] == 'merge'])
    split_count = len(subset[subset['event_type'] == 'split'])
    total = len(subset)
    index = (merge_count - split_count) / total if total > 0 else 0
    indices.append(index)

if len(indices) >= 2:
    trend = np.polyfit(range(len(indices)), indices, 1)[0]
    print(f'时间趋势斜率: {trend:.4f}')
    if trend > 0:
        print('整体趋势: 岗位综合化程度随时间增强')
    elif trend < 0:
        print('整体趋势: 岗位细分化程度随时间增强')
    else:
        print('整体趋势: 岗位综合化程度保持稳定')