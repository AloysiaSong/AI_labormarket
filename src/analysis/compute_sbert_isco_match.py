#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
compute_sbert_isco_match.py

用 SBERT 语义相似度补充 compute_ai_exposure.py 中未能匹配的岗位名称。

策略：
  1. 从 master_with_ai_exposure.csv 中提取所有 isco_match_method=='unknown' 的唯一岗位名
  2. 对 303 个 ILO ISCO-08 职业，用中文主名 + 别名构建查询向量（取均值或拼接）
  3. 对未匹配岗位名批量编码，与 ISCO 向量做余弦相似度，取最近邻
  4. 相似度超过阈值则接受匹配；否则保留 unknown
  5. 输出：
       data/Heterogeneity/sbert_isco_lookup.csv   （唯一岗位名 → isco结果）
       data/Heterogeneity/master_with_ai_exposure_v2.csv （补充匹配后的主数据）

模型：BAAI/bge-large-zh-v1.5
  - 中文专用SBERT，显著优于多语言模型的中文匹配质量
  - 用中文职业名 + 别名作为 ISCO 侧查询字符串
  - 已在 ~/.cache/huggingface/hub/ 缓存

阈值说明：
  SIM_THRESHOLD = 0.55  → 接受匹配（bge-large-zh 余弦值通常偏高，适当提高阈值）
  低于阈值的保留 unknown，不强行赋值

中文ISCO名称文件：
  data/esco/isco08_zh_names.csv
  列：isco08_4digit, occupation_name_en, occupation_name_zh, aliases_zh
"""

import time
from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

# ── 路径配置 ──────────────────────────────────────────────────────────
BASE        = Path('/Users/yu/code/code2601/TY')
ILO_CSV     = BASE / 'data/esco/ilo_genai_isco08_2025.csv'
ZH_NAMES_CSV = BASE / 'data/esco/isco08_zh_names.csv'
MASTER_IN   = BASE / 'data/Heterogeneity/master_with_ai_exposure.csv'
LOOKUP_OUT  = BASE / 'data/Heterogeneity/sbert_isco_lookup.csv'
MASTER_OUT  = BASE / 'data/Heterogeneity/master_with_ai_exposure_v2.csv'
CKPT_DIR    = BASE / 'data/Heterogeneity/sbert_checkpoints'

MODEL_NAME   = 'BAAI/bge-large-zh-v1.5'
SIM_THRESHOLD = 0.55   # bge-large-zh 余弦值偏高，阈值相应调高
BATCH_SIZE    = 256    # bge-large 模型较大，batch 适当缩小
CKPT_EVERY    = 50_000  # 每 N 条保存一次进度


def load_ilo(ilo_path: Path, zh_path: Path) -> pd.DataFrame:
    """
    合并 ILO ISCO-08 职业表与中文名称表。
    返回含 isco08_4digit, occupation_name_en, occupation_name_zh,
           aliases_zh, exposure_gradient 的 DataFrame。
    """
    ilo = pd.read_csv(ilo_path)
    ilo['isco08_4digit'] = ilo['isco08_4digit'].astype(int)
    zh  = pd.read_csv(zh_path)
    zh['isco08_4digit'] = zh['isco08_4digit'].astype(int)
    merged = ilo.merge(zh[['isco08_4digit', 'occupation_name_zh', 'aliases_zh']],
                       on='isco08_4digit', how='left')
    missing_zh = merged['occupation_name_zh'].isna().sum()
    if missing_zh > 0:
        print(f"  WARNING: {missing_zh} ISCO codes have no Chinese name, "
              "falling back to English.")
        merged['occupation_name_zh'].fillna(merged['occupation_name_en'], inplace=True)
        merged['aliases_zh'].fillna('', inplace=True)
    return merged


def build_isco_query_strings(ilo: pd.DataFrame) -> list[str]:
    """
    构造用于 bge-large-zh 编码的 ISCO 查询字符串。
    格式："{中文主名} {别名1} {别名2} ..."
    把主名和别名拼成一段话，让模型编码时覆盖更多同义词。
    """
    queries = []
    for _, row in ilo.iterrows():
        aliases = str(row.get('aliases_zh', '') or '').strip()
        q = row['occupation_name_zh']
        if aliases:
            q = q + ' ' + aliases
        queries.append(q)
    return queries


def encode_isco(model: SentenceTransformer, ilo: pd.DataFrame) -> np.ndarray:
    """对 303 个 ISCO 职业（中文名+别名）编码，返回 (303, dim) numpy 数组"""
    queries = build_isco_query_strings(ilo)
    print(f"  Encoding {len(queries)} ISCO occupations (Chinese names) ...")
    embs = model.encode(queries, batch_size=BATCH_SIZE,
                        normalize_embeddings=True, show_progress_bar=False)
    print(f"  ISCO embedding shape: {embs.shape}")
    return embs


def cosine_topk(query_embs: np.ndarray,
                key_embs: np.ndarray,
                k: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """
    批量余弦相似度最近邻（embeddings 已 L2 归一化 → 点积即余弦）
    返回 (scores, indices)，shape 均为 (N, k)
    """
    # query_embs: (N, d)  key_embs: (M, d)
    sims = query_embs @ key_embs.T          # (N, M)
    idx  = np.argpartition(sims, -k, axis=1)[:, -k:]   # top-k 索引（未排序）
    scores = np.take_along_axis(sims, idx, axis=1)
    # 对 k=1 直接 squeeze
    return scores[:, 0], idx[:, 0]


def match_titles(titles: list[str],
                 model: SentenceTransformer,
                 isco_embs: np.ndarray,
                 ilo: pd.DataFrame,
                 threshold: float = SIM_THRESHOLD,
                 ckpt_dir: Path | None = None) -> pd.DataFrame:
    """
    对唯一岗位名列表进行 SBERT 最近邻匹配。
    支持断点续跑（保存 / 恢复 checkpoint）。
    返回 DataFrame: title, isco08_4digit, sim_score, isco_name_en, gradient
    """
    n = len(titles)
    print(f"\nTotal unique titles to match: {n:,}")

    # ── 断点恢复 ──
    results: list[dict] = []
    start_idx = 0
    if ckpt_dir:
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        ckpt_files = sorted(ckpt_dir.glob('ckpt_*.pkl'))
        if ckpt_files:
            last = ckpt_files[-1]
            prev = pd.read_pickle(last)
            results = prev.to_dict('records')
            start_idx = len(results)
            print(f"  Resumed from checkpoint: {start_idx:,} already done")

    # ── 批量编码 & 匹配 ──
    t0 = time.time()
    for i in range(start_idx, n, BATCH_SIZE):
        batch = titles[i: i + BATCH_SIZE]
        embs  = model.encode(batch, batch_size=BATCH_SIZE,
                             normalize_embeddings=True, show_progress_bar=False)
        scores, idxs = cosine_topk(embs, isco_embs)

        for title, score, isco_idx in zip(batch, scores, idxs):
            row_ilo = ilo.iloc[isco_idx]
            accepted = bool(score >= threshold)
            results.append({
                'title': title,
                'isco08_4digit':    int(row_ilo['isco08_4digit']) if accepted else -1,
                'sim_score':        float(score),
                'isco_name_en':     row_ilo['occupation_name_en'] if accepted else '',
                'ai_exposure_gradient': row_ilo['exposure_gradient'] if accepted else 'unknown',
                'sbert_accepted':   accepted,
            })

        done = min(i + BATCH_SIZE, n)
        elapsed = time.time() - t0
        speed = (done - start_idx) / elapsed if elapsed > 0 else 0
        eta   = (n - done) / speed if speed > 0 else 0
        print(f"  [{done:>7,}/{n:,}]  {speed:.0f} sent/s  ETA {eta/60:.1f} min", end='\r')

        # 断点保存
        if ckpt_dir and (done % CKPT_EVERY == 0 or done == n):
            ckpt_path = ckpt_dir / f'ckpt_{done:09d}.pkl'
            pd.DataFrame(results).to_pickle(ckpt_path)

    print()  # newline after \r
    return pd.DataFrame(results)


def main():
    print("=" * 60)
    print("SBERT ISCO Matching")
    print("=" * 60)

    # 1. 加载 ILO 数据（合并中文名称）
    print("\n[1] Loading ILO ISCO-08 data + Chinese names ...")
    ilo = load_ilo(ILO_CSV, ZH_NAMES_CSV)
    print(f"  ISCO occupations: {len(ilo)}")
    print(f"  Gradient distribution:\n{ilo['exposure_gradient'].value_counts().to_string()}")

    # 2. 加载主数据，提取唯一未匹配岗位名
    print("\n[2] Loading master data and extracting unknown titles ...")
    master = pd.read_csv(MASTER_IN)
    print(f"  Total rows: {len(master):,}")

    unknown_mask = master['isco_match_method'] == 'unknown'
    print(f"  Unknown rows: {unknown_mask.sum():,}  ({unknown_mask.mean()*100:.1f}%)")

    unique_titles = master.loc[unknown_mask, '招聘岗位'].unique().tolist()
    print(f"  Unique unknown titles: {len(unique_titles):,}")

    # 3. 加载 SBERT 模型
    print(f"\n[3] Loading SBERT model: {MODEL_NAME} ...")
    model = SentenceTransformer(MODEL_NAME)

    # 4. 编码 ISCO 职业名
    print("\n[4] Encoding ISCO occupations ...")
    isco_embs = encode_isco(model, ilo)

    # 5. 批量匹配未知岗位名
    print(f"\n[5] Matching unknown titles (threshold={SIM_THRESHOLD}) ...")
    lookup = match_titles(
        titles    = unique_titles,
        model     = model,
        isco_embs = isco_embs,
        ilo       = ilo,
        threshold = SIM_THRESHOLD,
        ckpt_dir  = CKPT_DIR,
    )

    # 6. 统计 & 保存 lookup
    accepted = lookup['sbert_accepted'].sum()
    print(f"\n[6] SBERT match results:")
    print(f"  Accepted (sim >= {SIM_THRESHOLD}): {accepted:,}  ({accepted/len(lookup)*100:.1f}%)")
    print(f"  Rejected:                          {len(lookup)-accepted:,}")
    print(f"\n  Score distribution (accepted only):")
    acc = lookup[lookup['sbert_accepted']]
    if len(acc) > 0:
        print(f"    mean={acc['sim_score'].mean():.3f}  "
              f"p25={acc['sim_score'].quantile(.25):.3f}  "
              f"p50={acc['sim_score'].quantile(.5):.3f}  "
              f"p75={acc['sim_score'].quantile(.75):.3f}")
        print(f"\n  Top accepted gradient distribution:")
        print(acc['ai_exposure_gradient'].value_counts().to_string())

    print(f"\n  Sample high-confidence matches:")
    sample = acc.nlargest(15, 'sim_score')[['title','isco_name_en','sim_score','ai_exposure_gradient']]
    print(sample.to_string(index=False))

    print(f"\n  Sample low-confidence (near threshold):")
    near = acc[(acc['sim_score'] >= SIM_THRESHOLD) &
               (acc['sim_score'] < SIM_THRESHOLD + 0.05)].head(15)
    print(near[['title','isco_name_en','sim_score']].to_string(index=False))

    lookup.to_csv(LOOKUP_OUT, index=False)
    print(f"\n  Saved: {LOOKUP_OUT}")

    # 7. 更新 master 数据
    print("\n[7] Updating master data ...")
    title_to_row = lookup.set_index('title')

    def fill_row(row):
        if row['isco_match_method'] != 'unknown':
            return row
        lup = title_to_row.loc[row['招聘岗位']] if row['招聘岗位'] in title_to_row.index else None
        if lup is None or not lup['sbert_accepted']:
            return row
        row = row.copy()
        row['isco08_4digit']        = lup['isco08_4digit']
        row['ai_exposure_gradient'] = lup['ai_exposure_gradient']
        # Look up ai_mean_score and ai_sd_score from ILO table
        ilo_row = ilo[ilo['isco08_4digit'] == lup['isco08_4digit']]
        if len(ilo_row) > 0:
            row['ai_mean_score'] = ilo_row.iloc[0]['mean_score']
            row['ai_sd_score']   = ilo_row.iloc[0]['sd_score']
        row['isco_match_method'] = f'sbert_{lup["sim_score"]:.3f}'
        return row

    # Vectorised update via merge (much faster than apply)
    accepted_lookup = lookup[lookup['sbert_accepted']][
        ['title', 'isco08_4digit', 'ai_exposure_gradient', 'sim_score', 'isco_name_en']
    ].rename(columns={'title': '招聘岗位'})

    # Merge on 招聘岗位 for unknown rows only
    master_unk = master[unknown_mask].merge(
        accepted_lookup, on='招聘岗位', how='left', suffixes=('', '_sbert')
    )
    filled_mask = master_unk['isco08_4digit_sbert'].notna()
    master_unk.loc[filled_mask, 'isco08_4digit']        = master_unk.loc[filled_mask, 'isco08_4digit_sbert'].astype(int)
    master_unk.loc[filled_mask, 'ai_exposure_gradient'] = master_unk.loc[filled_mask, 'ai_exposure_gradient_sbert']
    master_unk.loc[filled_mask, 'isco_match_method']    = 'sbert_' + master_unk.loc[filled_mask, 'sim_score'].round(3).astype(str)
    # Update mean/sd from ILO
    ilo_score_lookup = ilo.set_index('isco08_4digit')[['mean_score', 'sd_score']]
    valid_isco = master_unk.loc[filled_mask, 'isco08_4digit'].astype(int)
    master_unk.loc[filled_mask, 'ai_mean_score'] = valid_isco.map(ilo_score_lookup['mean_score']).values
    master_unk.loc[filled_mask, 'ai_sd_score']   = valid_isco.map(ilo_score_lookup['sd_score']).values
    # Drop the extra sbert columns
    master_unk.drop(columns=['isco08_4digit_sbert', 'ai_exposure_gradient_sbert',
                              'sim_score', 'isco_name_en'], inplace=True)

    # Reconstruct full master
    master_v2 = pd.concat([master[~unknown_mask], master_unk], ignore_index=True)
    master_v2.sort_index(inplace=True)

    # Summary
    new_unknown = (master_v2['isco_match_method'] == 'unknown').sum()
    sbert_filled = filled_mask.sum()
    print(f"  SBERT newly matched rows: {sbert_filled:,}")
    print(f"  Remaining unknown:        {new_unknown:,}  ({new_unknown/len(master_v2)*100:.1f}%)")
    print(f"  New match_method breakdown:")
    print(master_v2['isco_match_method'].value_counts().head(5).to_string())

    master_v2.to_csv(MASTER_OUT, index=False)
    print(f"\n  Saved: {MASTER_OUT}")
    print(f"\n{'='*60}")
    print("Done.")


if __name__ == '__main__':
    main()
