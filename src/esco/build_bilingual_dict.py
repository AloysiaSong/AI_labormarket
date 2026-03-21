#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ESCO双语词典构建器
- 保留 Hard Skills 英文原名
- 自动将 Soft Skills 翻译成中文
- 生成 skill_metadata.csv 和 esco_jieba_dict.txt

支持的翻译后端:
- google: Google翻译 (免费，但国内不稳定)
- baidu: 百度翻译 (需要API key，国内稳定)
"""
import pandas as pd
import time
import os
import random
import sys
from pathlib import Path

# ============== 路径配置 ==============
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config.paths import (
    ESCO_ORIGINAL_DIR,
    JIEBA_DICT_DIR,
    ESCO_JIEBA_DICT,
    SKILL_METADATA_CSV
)

# 确保输出目录存在
JIEBA_DICT_DIR.mkdir(parents=True, exist_ok=True)

# 输入：ESCO官方英文技能表
INPUT_FILE = ESCO_ORIGINAL_DIR / 'ESCO dataset - v1.2.1 - classification - en - csv' / 'skills_en.csv'

# 输出
OUTPUT_META = SKILL_METADATA_CSV  # data/esco/jieba_dict/skill_metadata.csv
OUTPUT_DICT = ESCO_JIEBA_DICT     # data/esco/jieba_dict/esco_jieba_dict.txt
TEMP_FILE = JIEBA_DICT_DIR / 'temp_translation_progress.csv'  # 断点续传临时文件

# ============== 翻译配置 ==============
MAX_RETRIES = 3           # 最大重试次数
RETRY_DELAY = 5           # 重试等待秒数
TIMEOUT_DELAY = 10        # 超时后等待秒数
BATCH_SIZE = 50           # 每批保存数量
MIN_DELAY = 0.8           # 最小请求间隔
MAX_DELAY = 2.0           # 最大请求间隔


def translate_with_retry(translator, text, max_retries=MAX_RETRIES):
    """带重试机制的翻译"""
    for attempt in range(max_retries):
        try:
            result = translator.translate(text)
            return result
        except Exception as e:
            error_msg = str(e).lower()
            if 'timeout' in error_msg or 'connection' in error_msg:
                wait_time = TIMEOUT_DELAY * (attempt + 1)
                print(f"  [Retry {attempt+1}/{max_retries}] 网络超时，等待 {wait_time}s...")
                time.sleep(wait_time)
            else:
                print(f"  [Retry {attempt+1}/{max_retries}] 错误: {e}")
                time.sleep(RETRY_DELAY)
    return None  # 所有重试都失败


def run():
    print("="*60)
    print("ESCO双语词典构建器")
    print("="*60)
    print(f"\n输入文件: {INPUT_FILE}")
    print(f"输出元数据: {OUTPUT_META}")
    print(f"输出词典: {OUTPUT_DICT}")

    print(f"\nReading ESCO data: {INPUT_FILE}...")
    try:
        df = pd.read_csv(INPUT_FILE)
    except Exception as e:
        print(f"读取失败: {e}")
        return

    # 1. 准备数据
    tasks = df[['conceptUri', 'preferredLabel', 'skillType', 'reuseLevel']].copy()
    tasks['term'] = tasks['preferredLabel'].astype(str).str.strip()
    total_tasks = len(tasks)

    print(f"Total terms to process: {total_tasks}")

    # 2. 检查是否有断点续传文件
    if os.path.exists(TEMP_FILE):
        print("检测到临时文件，正在恢复进度...")
        finished_df = pd.read_csv(TEMP_FILE)
        finished_ids = set(finished_df['conceptUri'])
        print(f"已完成: {len(finished_ids)} 条")
    else:
        print("开始新任务...")
        finished_df = pd.DataFrame(columns=['conceptUri', 'term', 'lang', 'skillType', 'reuseLevel'])
        finished_ids = set()

    # 3. 初始化翻译器
    try:
        from deep_translator import GoogleTranslator
        translator = GoogleTranslator(source='en', target='zh-CN')
        print("使用 Google 翻译")
    except ImportError:
        print("请先安装: pip install deep-translator")
        return

    # 4. 开始循环 (只处理未完成的)
    new_rows = []
    error_count = 0
    consecutive_errors = 0

    # 筛选出未完成的任务
    pending_tasks = tasks[~tasks['conceptUri'].isin(finished_ids)]
    print(f"剩余任务: {len(pending_tasks)} 条")

    if len(pending_tasks) == 0:
        print("所有任务已完成，直接生成最终文件...")
    else:
        counter = 0
        for idx, row in pending_tasks.iterrows():
            original_term = row['term']
            uri = row['conceptUri']
            s_type = row['skillType']
            reuse = row['reuseLevel']

            # A. 保留英文原词 (English)
            new_rows.append({
                'conceptUri': uri,
                'term': original_term,
                'lang': 'en',
                'skillType': s_type,
                'reuseLevel': reuse
            })

            # B. 翻译中文词 (Chinese)
            if len(original_term) > 1 and not original_term.isdigit():
                # 使用带重试的翻译
                cn_term = translate_with_retry(translator, original_term)

                if cn_term and cn_term.lower() != original_term.lower():
                    new_rows.append({
                        'conceptUri': uri,
                        'term': cn_term,
                        'lang': 'zh',
                        'skillType': s_type,
                        'reuseLevel': reuse
                    })
                    consecutive_errors = 0  # 重置连续错误计数
                elif cn_term is None:
                    error_count += 1
                    consecutive_errors += 1
                    # 连续多次错误，可能网络问题，长等待
                    if consecutive_errors >= 3:
                        print(f"  连续 {consecutive_errors} 次失败，等待 30s...")
                        time.sleep(30)
                        consecutive_errors = 0

            counter += 1

            # C. 批次保存
            if counter % BATCH_SIZE == 0:
                batch_df = pd.DataFrame(new_rows)
                if not os.path.exists(TEMP_FILE):
                    batch_df.to_csv(TEMP_FILE, index=False)
                else:
                    batch_df.to_csv(TEMP_FILE, mode='a', header=False, index=False)

                pct = counter / len(pending_tasks) * 100
                print(f"Progress: {counter}/{len(pending_tasks)} ({pct:.1f}%) | Errors: {error_count}")
                new_rows = []

                # 随机休眠
                sleep_time = random.uniform(MIN_DELAY, MAX_DELAY)
                time.sleep(sleep_time)

        # 5. 循环结束，处理剩余缓存
        if new_rows:
            batch_df = pd.DataFrame(new_rows)
            if not os.path.exists(TEMP_FILE):
                batch_df.to_csv(TEMP_FILE, index=False)
            else:
                batch_df.to_csv(TEMP_FILE, mode='a', header=False, index=False)

        print(f"\n翻译完成! 总错误数: {error_count}")

    print("\n所有翻译任务结束！正在生成最终文件...")

    # 6. 生成最终产物
    if os.path.exists(TEMP_FILE):
        final_full_df = pd.read_csv(TEMP_FILE)
    else:
        print("没有找到翻译进度文件，退出。")
        return

    # 去重 (term 必须唯一)
    final_full_df.drop_duplicates(subset=['term'], inplace=True)

    # 添加 'category' 列 (Soft/Hard)
    final_full_df['category'] = final_full_df['reuseLevel'].apply(
        lambda x: 'soft' if str(x).strip().lower() == 'transversal' else 'hard'
    )

    # 输出 Metadata
    final_full_df.to_csv(OUTPUT_META, index=False)
    print(f"Metadata Saved: {OUTPUT_META} ({len(final_full_df)} terms)")

    # 输出 Jieba Dict
    with open(OUTPUT_DICT, 'w', encoding='utf-8') as f:
        for term in final_full_df['term']:
            term = str(term).strip()
            if len(term) > 1:
                f.write(f"{term} 20000 nz\n")

    print(f"Jieba Dict Saved: {OUTPUT_DICT}")
    print("\nDone! You have the ultimate bilingual dictionary.")


if __name__ == '__main__':
    run()
