#!/usr/bin/env python3

import pickle

# 加载数据
with open('/Users/yu/code/code2601/TY/output/taskpair/task_pairs_dict.pkl', 'rb') as f:
    data = pickle.load(f)

print('=== 任务对提取结果 ===')
print(f'总任务对数量: {data["metadata"]["total_pairs"]}')
print(f'过滤后任务对数量: {data["metadata"]["filtered_pairs"]}')
print(f'唯一动词数量: {data["metadata"]["total_verbs"]}')
print(f'唯一名词数量: {data["metadata"]["total_nouns"]}')

print('\n=== Top 10 任务对 ===')
sorted_pairs = sorted(data['task_pairs'].items(), key=lambda x: x[1], reverse=True)
for i, ((verb, noun), count) in enumerate(sorted_pairs[:10]):
    print(f"{i+1}. {verb}-{noun}: {count}")

print('\n=== Top 10 动词 ===')
for i, (verb, count) in enumerate(sorted(data['verb_freq'].items(), key=lambda x: x[1], reverse=True)[:10]):
    print(f"{i+1}. {verb}: {count}")

print('\n=== Top 10 名词 ===')
for i, (noun, count) in enumerate(sorted(data['noun_freq'].items(), key=lambda x: x[1], reverse=True)[:10]):
    print(f"{i+1}. {noun}: {count}")