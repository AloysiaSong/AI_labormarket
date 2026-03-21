"""查看词典过滤效果"""
from gensim import corpora
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from config.paths import LDA_OUTPUT_DIR

# 加载2022-2023的词典（最大窗口）
dict_path = LDA_OUTPUT_DIR / "dictionaries" / "window_2022_2023.dict"
dictionary = corpora.Dictionary.load(str(dict_path))

print(f"\n📊 词典大小: {len(dictionary):,} 词\n")

# 查看词频分布
word_freqs = [(dictionary[id], dictionary.dfs[id]) for id in dictionary.keys()]
word_freqs.sort(key=lambda x: x[1], reverse=True)

print("=== 保留的高频词 (Top 30) ===")
print(f"{'词':20s} {'文档频率':>15s}")
print("-" * 38)
for word, freq in word_freqs[:30]:
    print(f"{word:20s} {freq:>15,}")

print("\n=== 保留的中频词 (示例) ===")
mid_start = len(word_freqs) // 2
for word, freq in word_freqs[mid_start:mid_start+15]:
    print(f"{word:20s} {freq:>15,}")

print("\n=== 保留的低频词 (Bottom 20) ===")
for word, freq in word_freqs[-20:]:
    print(f"{word:20s} {freq:>15,}")

# 统计
total_docs = 4188301
min_threshold = 100
max_threshold = int(total_docs * 0.4)
print(f"\n{'='*50}")
print(f"过滤阈值说明:")
print(f"  最小文档频率: {min_threshold:,} (删除低频罕见词)")
print(f"  最大文档频率: {max_threshold:,} (删除高频通用词)")
print(f"  最大保留数: 30,000 词")
print(f"{'='*50}")
