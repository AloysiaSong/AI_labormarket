#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NLP 预处理管线：merged_1_6.csv → processed_corpus.jsonl

两遍扫描：
  Pass 1: CSV → 正则 → jieba分词 → 停词/英文过滤 → 临时JSONL + bigram统计
  Pass 2: 临时JSONL → bigram合并 → MIN_TOKENS过滤 → 最终JSONL + 诊断统计
"""

import csv
import json
import re
import math
import time
from collections import Counter
from pathlib import Path

# ============================================================
# 路径配置
# ============================================================
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "dataset"
OUTPUT_DIR = BASE_DIR / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

INPUT_CSV = DATA_DIR / "merged_1_6.csv"
HR_DICT = BASE_DIR / "data" / "esco" / "jieba_dict" / "hr_jieba_dict.txt"
STOPWORDS_FILE = BASE_DIR / "data" / "stopwords.txt"

TEMP_JSONL = OUTPUT_DIR / "temp_pass1.jsonl"
FINAL_JSONL = OUTPUT_DIR / "processed_corpus.jsonl"
STATS_FILE = OUTPUT_DIR / "preprocessing_stats.json"

# ============================================================
# 参数
# ============================================================
MIN_TOKENS = 5
BIGRAM_MIN_COUNT = 100           # 从20提升到100，过滤低频噪声bigram
BIGRAM_MIN_PMI = 3.0             # PMI阈值从0提升到3，只保留强共现
BIGRAM_MAX_PAIRS = 200_000
BIGRAM_PRUNE_EVERY = 1_000_000   # 每处理N条文档后剪枝低频词对
BIGRAM_PRUNE_THRESHOLD = 20      # 剪枝阈值（与MIN_COUNT对齐）

# Bigram 黑名单：即使频次和PMI达标也不保留的噪声bigram
# 这些bigram反映的是福利待遇、数据源水印、HR模板，而非技能/任务内容
BIGRAM_BLACKLIST_PATTERNS = {
    # 福利/薪资
    '薪资_待遇', '法定_节假日', '节日_福利', '底薪_提成', '综合_工资',
    '福利_五险一金', '综合_薪资', '责任_底薪', '生日_福利', '国家_法定',
    '国家_规定',
    # 数据源水印
    '来源_百度', '百度_搜索', '数据_由马', '详见_官网', '关注_公众',
    '数据_搜索', '分享_微信', '微信_分享', '数据_详见',
    # HR模板/招聘流程
    '投递_简历', '加入_我们', '一经_录用', '晋升_机会', '晋升_机制',
    '接受_应届毕业生', '欢迎_应届毕业生', '应届毕业生_亦可',
    # 条件/要求
    '年龄_周岁', '形象_气质佳', '承受_压力', '承受_较大', '较大_压力',
    '思维_敏捷', '统招_本科', '全日制_本科', '中专_以上学历',
    '高中_以上学历',
    # 工作时间/地点
    '上午_下午', '周一_周五', '每天_小时', '上班_地点',
    # 泛化搭配
    '上级_交办', '上级领导_交办', '完善_培训', '扁平_管理',
}

# ============================================================
# 英文黑名单
# ============================================================
EN_BLACKLIST = {
    # HTML/网页残留
    'br', 'lt', 'gt', 'nbsp', 'amp', 'div', 'span', 'font', 'img',
    'href', 'style', 'color', 'margin', 'padding', 'px', 'display',
    'border', 'width', 'height', 'background', 'text', 'align',
    'www', 'http', 'https', 'com', 'cn', 'org', 'net',
    'macrodatas', 'xmrc', 'pgt',
    # 英文虚词
    'and', 'the', 'in', 'to', 'of', 'or', 'with', 'for',
    'is', 'are', 'be', 'an', 'at', 'by', 'on', 'as',
    'we', 'do', 'if', 'no', 'not', 'can', 'will',
    'this', 'that', 'from', 'but', 'all', 'so', 'our', 'your',
    # JD模板英文词（非技能）
    'experience', 'management', 'project', 'work', 'team',
    'position', 'job', 'company', 'please', 'good', 'well',
    # 度量/无意义
    'cm', 'mm', 'kg', 'ee',
}

# 单字母仅保留 c 和 r（C语言、R语言）
KEEP_SINGLE_LETTERS = {'c', 'r'}

# ============================================================
# 加载停词表
# ============================================================
def load_stopwords(path):
    stopwords = set()
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                stopwords.add(line)
    return stopwords

# ============================================================
# 正则预编译
# ============================================================
RE_KEEP = re.compile(r'[^\u4e00-\u9fa5a-z]')

# ============================================================
# Token 过滤函数
# ============================================================
def filter_tokens(raw_tokens, cn_stopwords):
    """对 jieba 输出的 token 列表做停词+英文黑名单过滤"""
    filtered = []
    for t in raw_tokens:
        t = t.strip()
        if not t:
            continue

        if t.isascii():
            # 英文 token
            if not t.isalpha():
                continue
            if len(t) == 1:
                if t in KEEP_SINGLE_LETTERS:
                    filtered.append(t)
                continue
            if t in EN_BLACKLIST:
                continue
            filtered.append(t)
        else:
            # 中文 token
            if len(t) < 2:
                continue
            if t in cn_stopwords:
                continue
            filtered.append(t)

    return filtered

# ============================================================
# Bigram 合并
# ============================================================
def apply_bigrams(tokens, bigram_set):
    """贪心从左到右合并 bigram"""
    if not bigram_set:
        return tokens
    merged = []
    i = 0
    while i < len(tokens):
        if i < len(tokens) - 1 and (tokens[i], tokens[i + 1]) in bigram_set:
            merged.append(tokens[i] + '_' + tokens[i + 1])
            i += 2
        else:
            merged.append(tokens[i])
            i += 1
    return merged

# ============================================================
# Pass 1
# ============================================================
def pass1(jieba_mod, cn_stopwords):
    """
    读 CSV → 正则 → jieba → 过滤 → 写临时 JSONL
    同时统计 bigram 和单词频次
    """
    print("=" * 60)
    print("Pass 1: 分词 + 过滤 + bigram 统计")
    print("=" * 60)

    word_count = Counter()
    pair_count = Counter()
    total_pairs = 0
    row_id = 0
    written = 0
    t0 = time.time()

    with open(INPUT_CSV, 'r', encoding='utf-8-sig') as fin, \
         open(TEMP_JSONL, 'w', encoding='utf-8') as fout:

        reader = csv.DictReader(fin)

        for row in reader:
            row_id += 1

            # --- 正则过滤 ---
            desc = row.get('职位描述', '')
            text = RE_KEEP.sub(' ', desc)

            # --- jieba 分词 ---
            raw_tokens = jieba_mod.lcut(text)

            # --- 停词 + 英文过滤 ---
            tokens = filter_tokens(raw_tokens, cn_stopwords)

            # --- 写入临时 JSONL（含所有文档，Pass 2 再按 MIN_TOKENS 过滤）---
            obj = {
                "id": row_id,
                "year": row.get('招聘发布年份', ''),
                "tokens": tokens,
                "is_fresh_grad": int(row.get('is_fresh_grad', 0)),
                "platform": row.get('来源', ''),
                "data_source": row.get('data_source', ''),
            }
            fout.write(json.dumps(obj, ensure_ascii=False) + '\n')
            written += 1

            # --- Bigram 统计 ---
            for t in tokens:
                word_count[t] += 1
            for i in range(len(tokens) - 1):
                pair_count[(tokens[i], tokens[i + 1])] += 1
                total_pairs += 1

            # --- 定期剪枝 + 进度报告 ---
            if row_id % BIGRAM_PRUNE_EVERY == 0:
                elapsed = time.time() - t0
                speed = row_id / elapsed
                print(f"  {row_id:>12,} rows | {elapsed:.0f}s | {speed:.0f} rows/s | "
                      f"pairs: {len(pair_count):,}", flush=True)
                pair_count = Counter({k: v for k, v in pair_count.items()
                                      if v >= BIGRAM_PRUNE_THRESHOLD})

    elapsed = time.time() - t0
    print(f"\nPass 1 完成: {row_id:,} rows, {elapsed:.0f}s")
    print(f"  单词种类: {len(word_count):,}")
    print(f"  词对种类(剪枝后): {len(pair_count):,}")
    print(f"  总词对数: {total_pairs:,}")

    return word_count, pair_count, total_pairs, row_id

# ============================================================
# Bigram 筛选（频次 + PMI）
# ============================================================
def select_bigrams(word_count, pair_count, total_pairs):
    """用频次 >= BIGRAM_MIN_COUNT AND PMI >= BIGRAM_MIN_PMI 筛选 bigram，排除黑名单"""
    print(f"\n筛选 bigram (freq >= {BIGRAM_MIN_COUNT}, PMI >= {BIGRAM_MIN_PMI})...")

    candidates = []
    blacklisted = 0
    for (a, b), cnt in pair_count.items():
        if cnt < BIGRAM_MIN_COUNT:
            continue
        # 黑名单过滤
        bigram_str = f"{a}_{b}"
        if bigram_str in BIGRAM_BLACKLIST_PATTERNS:
            blacklisted += 1
            continue
        ca = word_count.get(a, 1)
        cb = word_count.get(b, 1)
        pmi = math.log(cnt * total_pairs / (ca * cb))
        if pmi >= BIGRAM_MIN_PMI:
            candidates.append(((a, b), cnt, pmi))

    # 按频次排序，取 top BIGRAM_MAX_PAIRS
    candidates.sort(key=lambda x: -x[1])
    selected = candidates[:BIGRAM_MAX_PAIRS]
    bigram_set = {pair for pair, cnt, pmi in selected}

    print(f"  黑名单排除: {blacklisted:,}")
    print(f"  候选(freq>={BIGRAM_MIN_COUNT} & PMI>={BIGRAM_MIN_PMI}): {len(candidates):,}")
    print(f"  保留(top {BIGRAM_MAX_PAIRS:,}): {len(bigram_set):,}")
    if selected:
        print(f"\n  Top 30 bigrams:")
        for (a, b), cnt, pmi in selected[:30]:
            print(f"    {a}_{b:20s}  freq={cnt:>8,}  PMI={pmi:.2f}")

    return bigram_set, selected

# ============================================================
# Pass 2
# ============================================================
def pass2(bigram_set):
    """
    读临时 JSONL → 合并 bigram → 过滤 MIN_TOKENS → 写最终 JSONL
    """
    print("\n" + "=" * 60)
    print("Pass 2: bigram 合并 + 最终过滤")
    print("=" * 60)

    kept = 0
    dropped = 0
    year_counts = Counter()
    token_counts = []
    t0 = time.time()

    with open(TEMP_JSONL, 'r', encoding='utf-8') as fin, \
         open(FINAL_JSONL, 'w', encoding='utf-8') as fout:

        for line_no, line in enumerate(fin, 1):
            obj = json.loads(line)
            tokens = obj['tokens']

            # --- Bigram 合并 ---
            tokens = apply_bigrams(tokens, bigram_set)
            token_count = len(tokens)

            # --- MIN_TOKENS 过滤 ---
            if token_count < MIN_TOKENS:
                dropped += 1
                continue

            # --- 写入最终 JSONL ---
            obj['tokens'] = tokens
            obj['token_count'] = token_count
            fout.write(json.dumps(obj, ensure_ascii=False) + '\n')
            kept += 1
            year_counts[obj['year']] += 1
            token_counts.append(token_count)

            if line_no % 5_000_000 == 0:
                elapsed = time.time() - t0
                print(f"  {line_no:>12,} lines | kept {kept:,} | "
                      f"dropped {dropped:,} | {elapsed:.0f}s", flush=True)

    elapsed = time.time() - t0
    print(f"\nPass 2 完成: kept {kept:,}, dropped {dropped:,}, {elapsed:.0f}s")

    return kept, dropped, year_counts, token_counts

# ============================================================
# 诊断统计
# ============================================================
def write_stats(total_input, kept, dropped, year_counts, token_counts,
                bigram_selected, bigram_set_size):
    """输出 preprocessing_stats.json"""
    tc = sorted(token_counts)
    n = len(tc)

    stats = {
        "total_input": total_input,
        "total_output": kept,
        "dropped_short": dropped,
        "drop_rate_pct": round(dropped / total_input * 100, 2),
        "token_count_stats": {
            "mean": round(sum(tc) / n, 1),
            "median": tc[n // 2],
            "p5": tc[int(n * 0.05)],
            "p25": tc[int(n * 0.25)],
            "p75": tc[int(n * 0.75)],
            "p95": tc[int(n * 0.95)],
            "p99": tc[int(n * 0.99)],
            "max": tc[-1],
        },
        "bigram_count": bigram_set_size,
        "top_50_bigrams": [
            {"bigram": f"{a}_{b}", "freq": cnt, "pmi": round(pmi, 2)}
            for (a, b), cnt, pmi in bigram_selected[:50]
        ],
        "per_year_counts": {k: v for k, v in sorted(year_counts.items())},
    }

    with open(STATS_FILE, 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print(f"\n诊断统计已保存: {STATS_FILE}")
    print(f"  输入: {total_input:,}")
    print(f"  输出: {kept:,}")
    print(f"  丢弃 (token<{MIN_TOKENS}): {dropped:,} ({stats['drop_rate_pct']}%)")
    print(f"  token_count: mean={stats['token_count_stats']['mean']}, "
          f"median={stats['token_count_stats']['median']}, "
          f"p95={stats['token_count_stats']['p95']}, "
          f"max={stats['token_count_stats']['max']}")
    print(f"  bigram 数量: {bigram_set_size:,}")
    print(f"\n  每年文档数:")
    for y, c in sorted(year_counts.items()):
        print(f"    {y}: {c:,}")

# ============================================================
# Main
# ============================================================
def main():
    print(f"输入: {INPUT_CSV}")
    print(f"输出: {FINAL_JSONL}")
    print(f"HR词典: {HR_DICT}")
    print(f"停词表: {STOPWORDS_FILE}")
    print(f"参数: MIN_TOKENS={MIN_TOKENS}, BIGRAM_MIN_COUNT={BIGRAM_MIN_COUNT}, "
          f"BIGRAM_MAX_PAIRS={BIGRAM_MAX_PAIRS}")
    print()

    # 加载 jieba + 词典
    import jieba
    print("加载 jieba + HR 词典...", flush=True)
    jieba.load_userdict(str(HR_DICT))
    jieba.enable_parallel(8)          # 并行分词，使用 8 个进程
    jieba.lcut("预热测试")
    print("jieba 就绪 (parallel=8)\n")

    # 加载停词
    cn_stopwords = load_stopwords(STOPWORDS_FILE)
    print(f"停词表: {len(cn_stopwords)} 个\n")

    # Pass 1
    word_count, pair_count, total_pairs, total_input = pass1(jieba, cn_stopwords)

    # 筛选 bigram
    bigram_set, bigram_selected = select_bigrams(word_count, pair_count, total_pairs)

    # 释放 Pass 1 大对象
    del word_count, pair_count

    # Pass 2
    kept, dropped, year_counts, token_counts = pass2(bigram_set)

    # 诊断统计
    write_stats(total_input, kept, dropped, year_counts, token_counts,
                bigram_selected, len(bigram_set))

    # 临时文件提示
    if TEMP_JSONL.exists():
        size_gb = TEMP_JSONL.stat().st_size / (1024 ** 3)
        print(f"\n临时文件 {TEMP_JSONL} ({size_gb:.1f}GB) 保留供检查。")
        print("确认无误后可手动删除。")

    print("\n全部完成!")


if __name__ == '__main__':
    main()
