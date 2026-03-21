"""
Step 3: 计算每个 JD 的技能域分布 (86维向量)

输入:
  - output/processed_corpus.jsonl          (26.8M JD, 每行有 id/year/tokens/...)
  - output/w2v/word_clusters_hierarchical.csv (596K 词 → cluster_label)
  - output/w2v/cluster_names.csv             (109 cluster → type: skill/non_skill/noise)

输出:
  - output/w2v/jd_skill_profiles.npz       稀疏矩阵 N x 86
  - output/w2v/jd_metadata.csv             元数据 + 附带指标
  - output/w2v/skill_cluster_columns.csv   列号 → cluster_label 映射
"""

import json
import csv
import time
import numpy as np
from scipy import sparse
from pathlib import Path
from collections import defaultdict

# ── 路径 ──
BASE = Path(__file__).resolve().parent.parent.parent
CORPUS_PATH = BASE / "output" / "processed_corpus.jsonl"
WORD_CLUSTERS_PATH = BASE / "output" / "w2v" / "word_clusters_hierarchical.csv"
CLUSTER_NAMES_PATH = BASE / "output" / "w2v" / "cluster_names.csv"
OUT_DIR = BASE / "output" / "w2v"

OUT_PROFILES = OUT_DIR / "jd_skill_profiles.npz"
OUT_METADATA = OUT_DIR / "jd_metadata.csv"
OUT_COLUMNS  = OUT_DIR / "skill_cluster_columns.csv"

# ── 参数 ──
MIN_SKILL_TOKENS = 5  # skill token 数低于此值的 JD 分布置零，避免噪声


def load_skill_clusters():
    """加载 skill cluster 列表，返回排序后的 list 和 label→col_idx 映射"""
    skill_labels = []
    with open(CLUSTER_NAMES_PATH, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["type"] == "skill":
                skill_labels.append(row["cluster_label"])
    skill_labels.sort(key=lambda x: (x.split("_")[0].zfill(3), x))
    label2col = {label: i for i, label in enumerate(skill_labels)}
    return skill_labels, label2col


def load_word2cluster(label2col):
    """加载词→cluster映射，只保留属于 skill cluster 的词"""
    word2col = {}
    with open(WORD_CLUSTERS_PATH, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            label = row["cluster_label"]
            if label in label2col:
                word2col[row["word"]] = label2col[label]
    return word2col


def process_corpus(word2col, n_cols, ai_col, col2label):
    """流式处理 JSONL，输出稀疏矩阵和元数据"""
    # 稀疏矩阵用 COO 格式积累
    all_rows = []
    all_cols = []
    all_vals = []

    metadata_fields = ["id", "year", "is_fresh_grad", "data_source", "token_count",
                       "skill_token_count", "skill_coverage", "n_skill_clusters",
                       "top_cluster", "ai_intensity", "ai_binary"]

    meta_f = open(OUT_METADATA, "w", encoding="utf-8", newline="")
    meta_writer = csv.writer(meta_f)
    meta_writer.writerow(metadata_fields)

    row_idx = 0
    t0 = time.time()

    with open(CORPUS_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            doc = json.loads(line)
            tokens = doc.get("tokens", [])

            # 统计每个 skill column 的词频
            col_counts = defaultdict(int)
            for tok in tokens:
                col = word2col.get(tok)
                if col is not None:
                    col_counts[col] += 1

            skill_token_count = sum(col_counts.values())
            token_count = doc.get("token_count", len(tokens))

            # 归一化 → 分布向量 (L1)
            # skill token 太少的 JD 不写入稀疏矩阵（分布噪声大）
            if skill_token_count >= MIN_SKILL_TOKENS:
                for col, cnt in col_counts.items():
                    all_rows.append(row_idx)
                    all_cols.append(col)
                    all_vals.append(cnt / skill_token_count)

            # 计算附带指标
            skill_coverage = skill_token_count / token_count if token_count > 0 else 0.0
            n_skill_clusters = len(col_counts)

            if col_counts:
                top_col = max(col_counts, key=col_counts.get)
                top_label = col2label.get(top_col, "")
            else:
                top_label = ""

            # ai_intensity: cluster 11_0 的占比 (连续值)
            ai_count = col_counts.get(ai_col, 0)
            ai_val = ai_count / skill_token_count if skill_token_count > 0 else 0.0
            # ai_binary: 是否包含至少 1 个 AI cluster 词 (0/1)
            ai_bin = 1 if ai_count > 0 else 0

            meta_writer.writerow([
                doc.get("id", row_idx),
                doc.get("year", ""),
                doc.get("is_fresh_grad", ""),
                doc.get("data_source", ""),
                token_count,
                skill_token_count,
                f"{skill_coverage:.4f}",
                n_skill_clusters,
                top_label,
                f"{ai_val:.6f}",
                ai_bin,
            ])

            row_idx += 1
            if row_idx % 1_000_000 == 0:
                elapsed = time.time() - t0
                speed = row_idx / elapsed
                print(f"  {row_idx:>12,} docs  |  {elapsed:.0f}s  |  {speed:,.0f} docs/s")

    meta_f.close()

    print(f"\nTotal: {row_idx:,} docs, building sparse matrix...")

    # 构建稀疏矩阵
    mat = sparse.csr_matrix(
        (np.array(all_vals, dtype=np.float32),
         (np.array(all_rows, dtype=np.int64),
          np.array(all_cols, dtype=np.int32))),
        shape=(row_idx, n_cols)
    )
    return mat


def main():
    print("=" * 60)
    print("Step 3: JD Skill Profile Computation")
    print("=" * 60)

    # 1. 加载 skill cluster 列表
    print("\n[1/4] Loading skill clusters...")
    skill_labels, label2col = load_skill_clusters()
    n_cols = len(skill_labels)
    print(f"  {n_cols} skill clusters")

    # 保存列号映射
    with open(OUT_COLUMNS, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["col_idx", "cluster_label", "name_cn"])
        # 读 cluster_names 获取中文名
        cn_map = {}
        with open(CLUSTER_NAMES_PATH, "r", encoding="utf-8") as cf:
            for row in csv.DictReader(cf):
                cn_map[row["cluster_label"]] = row["name_cn"]
        for i, label in enumerate(skill_labels):
            writer.writerow([i, label, cn_map.get(label, "")])
    print(f"  Saved {OUT_COLUMNS.name}")

    # 2. 加载词→列号映射
    print("\n[2/4] Loading word → cluster mapping...")
    word2col = load_word2cluster(label2col)
    print(f"  {len(word2col):,} words mapped to skill clusters")

    # 找到 AI cluster (11_0) 的列号
    ai_col = label2col.get("11_0")
    print(f"  AI/ML cluster (11_0) → column {ai_col}")

    # 列号→label 反查表
    col2label = {i: label for label, i in label2col.items()}

    # 3. 流式处理
    print(f"\n[3/4] Processing corpus: {CORPUS_PATH}")
    mat = process_corpus(word2col, n_cols, ai_col, col2label)

    # 4. 保存稀疏矩阵
    print(f"\n[4/4] Saving sparse matrix...")
    sparse.save_npz(OUT_PROFILES, mat)
    nnz = mat.nnz
    density = nnz / (mat.shape[0] * mat.shape[1]) * 100
    size_mb = OUT_PROFILES.stat().st_size / 1024 / 1024
    print(f"  Shape: {mat.shape[0]:,} x {mat.shape[1]}")
    print(f"  Non-zero: {nnz:,} ({density:.2f}%)")
    print(f"  File size: {size_mb:.1f} MB")
    print(f"  Saved: {OUT_PROFILES.name}")

    # 快速统计
    print("\n── Quick Stats ──")
    ai_vals = []
    ai_bins = []
    skill_coverages = []
    n_clusters_list = []
    skill_counts = []
    with open(OUT_METADATA, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            ai_vals.append(float(row["ai_intensity"]))
            ai_bins.append(int(row["ai_binary"]))
            skill_coverages.append(float(row["skill_coverage"]))
            n_clusters_list.append(int(row["n_skill_clusters"]))
            skill_counts.append(int(row["skill_token_count"]))
            if i >= 100_000:
                break
    n = len(ai_vals)
    n_valid = sum(1 for s in skill_counts if s >= MIN_SKILL_TOKENS)
    print(f"  (Based on first {n:,} docs)")
    print(f"  Mean skill_coverage: {np.mean(skill_coverages):.3f}")
    print(f"  Mean n_skill_clusters: {np.mean(n_clusters_list):.1f}")
    print(f"  Docs with skill_tokens >= {MIN_SKILL_TOKENS}: {n_valid:,} ({n_valid/n*100:.1f}%)")
    print(f"  Mean ai_intensity: {np.mean(ai_vals):.6f}")
    print(f"  ai_binary=1 (has AI skill): {sum(ai_bins):,} ({sum(ai_bins)/n*100:.2f}%)")

    print("\nDone!")


if __name__ == "__main__":
    main()
