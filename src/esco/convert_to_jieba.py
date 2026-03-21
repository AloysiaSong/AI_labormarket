#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ESCO词典转换为jieba用户词典格式（仅使用 preferredLabel）
"""
import pandas as pd
import re
import sys
from pathlib import Path

# 使用集中路径配置
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config.paths import ESCO_ORIGINAL_DIR, ESCO_JIEBA_DICT, SKILL_METADATA_CSV, JIEBA_DICT_DIR

# 确保输出目录存在
JIEBA_DICT_DIR.mkdir(parents=True, exist_ok=True)

# 配置输入输出路径
INPUT_FILE = str(ESCO_ORIGINAL_DIR / 'ESCO dataset - v1.2.1 - classification - en - csv' / 'skills_en.csv')
OUTPUT_DICT = str(ESCO_JIEBA_DICT)
OUTPUT_META = str(SKILL_METADATA_CSV)
OUTPUT_CLEAN_CSV = str(JIEBA_DICT_DIR / 'skills_ch.csv')


def clean_term(text: str) -> str:
    """清洗：去括号、去多余空格"""
    if not isinstance(text, str):
        return ""
    text = re.sub(r'\(.*?\)', '', text)
    text = re.sub(r'（.*?）', '', text)
    return text.strip()


def split_alt_labels(text: str) -> list:
    """ESCO 标准格式：altLabels 以换行符分隔"""
    if not isinstance(text, str) or not text.strip():
        return []
    return [clean_term(t) for t in text.split('\n') if clean_term(t)]

def main():
    print(f"Reading {INPUT_FILE}...")
    try:
        df = pd.read_csv(INPUT_FILE)
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return

    cols = df.columns.tolist()
    print(f"\n[DEBUG] CSV Columns found: {cols}")

    if 'skillType' not in cols:
        print("\n[WARNING] ⚠️ Could not find column: skillType")
    if 'reuseLevel' not in cols:
        print("\n[WARNING] ⚠️ Could not find column: reuseLevel")

    jieba_terms = set()
    metadata_rows = []

    print("Processing terms (Preferred Labels Only)...")

    count_hard = 0
    count_soft = 0
    count_knowledge = 0

    for _, row in df.iterrows():
        raw_pref = row.get('preferredLabel', '')
        if pd.isna(raw_pref) or not raw_pref:
            continue

        term = clean_term(str(raw_pref))
        if len(term) < 2 or len(term) > 30:
            continue

        jieba_terms.add(term)

        s_type = 'unknown'
        skill_type_val = str(row.get('skillType', '')).lower()
        reuse_level_val = str(row.get('reuseLevel', '')).lower()
        if 'knowledge' in skill_type_val:
            s_type = 'knowledge'
            count_knowledge += 1
        elif 'skill' in skill_type_val or 'competence' in skill_type_val:
            s_type = 'skill'
            count_hard += 1
        elif 'transversal' in reuse_level_val or 'soft' in skill_type_val:
            s_type = 'soft'
            count_soft += 1

        metadata_rows.append({
            'term': term,
            'origin_label': raw_pref,
            'type': s_type,
            'skillType': row.get('skillType', ''),
            'reuseLevel': row.get('reuseLevel', '')
        })

        alt_terms = split_alt_labels(row.get('altLabels', ''))
        for alt in alt_terms:
            if len(alt) < 2 or len(alt) > 30:
                continue
            jieba_terms.add(alt)
            metadata_rows.append({
                'term': alt,
                'origin_label': raw_pref,
                'type': s_type,
                'skillType': row.get('skillType', ''),
                'reuseLevel': row.get('reuseLevel', '')
            })

    print(f"\n[SUMMARY] Terms extracted: {len(jieba_terms)}")
    print(f" - Knowledge/Hard: {count_knowledge}")
    print(f" - Skill/Hard: {count_hard}")
    print(f" - Soft/Transversal: {count_soft}")
    if count_soft == 0 and count_knowledge == 0:
        print("⚠️ WARNING: No distinction found. Check your CSV columns!")

    Path(OUTPUT_DICT).parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_DICT, 'w', encoding='utf-8') as f:
        for term in sorted(jieba_terms):
            f.write(f"{term} 2000 nz\n")
    print(f"Saved {OUTPUT_DICT}")

    meta_df = pd.DataFrame(metadata_rows)
    meta_df.drop_duplicates(subset=['term'], keep='first', inplace=True)
    Path(OUTPUT_META).parent.mkdir(parents=True, exist_ok=True)
    meta_df.to_csv(OUTPUT_META, index=False)
    print(f"Saved {OUTPUT_META}")

    cleaned_df = meta_df[['term', 'origin_label', 'skillType', 'reuseLevel']].copy()
    cleaned_df.rename(columns={'term': 'preferredLabel'}, inplace=True)
    Path(OUTPUT_CLEAN_CSV).parent.mkdir(parents=True, exist_ok=True)
    cleaned_df.to_csv(OUTPUT_CLEAN_CSV, index=False)
    print(f"Saved {OUTPUT_CLEAN_CSV}")

if __name__ == '__main__':
    main()
