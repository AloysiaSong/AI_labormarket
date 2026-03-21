#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
experiment_auto_aug.py

Experiment 1: Automation vs Augmentation Decomposition
Inspired by Hampole, Rao & Tan (NBER WP 2025).

Decomposes aggregate AI exposure into automation-dominant and augmentation-dominant
components, then tests whether they have differential (opposing) effects on entropy.

Method:
  1. Use SBERT to classify 303 ISCO occupations as more "automation-like" or
     "augmentation-like" by computing cosine similarity to concept reference texts
  2. Decompose mean_score into automation_score + augmentation_score
  3. Run regressions with both components as separate regressors
  4. Robustness: binary classification based on ISCO major groups

Input:
  data/esco/ilo_genai_isco08_2025.csv
  data/esco/isco08_zh_names.csv
  data/Heterogeneity/master_with_ai_exposure_v2.csv

Output:
  output/auto_aug/
"""

import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── Paths ──────────────────────────────────────────────────────────
BASE = Path('/Users/yu/code/code2601/TY')
ILO_CSV = BASE / 'data/esco/ilo_genai_isco08_2025.csv'
ZH_NAMES_CSV = BASE / 'data/esco/isco08_zh_names.csv'
MASTER = BASE / 'data/Heterogeneity/master_with_ai_exposure_v2.csv'
OUT = BASE / 'output/auto_aug'
OUT.mkdir(parents=True, exist_ok=True)

CHUNK = 500_000
REF_YEAR = 2016
POST_YEAR = 2023
MODEL_NAME = 'BAAI/bge-large-zh-v1.5'

# ── Concept reference texts ────────────────────────────────────────
# Automation: routine, standardized, repetitive cognitive/manual tasks
AUTOMATION_CONCEPTS = [
    "数据录入和文书处理等重复性认知工作",
    "信息分类、翻译和格式转换等标准化任务",
    "表格填写、数据核对和记录归档",
    "按既定规则处理客户咨询和订单",
    "重复性计算、统计和报告生成",
    "抄写、打字和文档排版",
    "标准化流程中的信息录入与核实",
]

# Augmentation: complex, creative, interpersonal, analytical tasks
AUGMENTATION_CONCEPTS = [
    "专业分析、创新设计和战略决策",
    "复杂问题诊断和创造性解决方案",
    "人际沟通、团队领导和谈判协调",
    "研究开发、实验设计和技术创新",
    "教学指导、心理咨询和个性化服务",
    "艺术创作、产品设计和品牌策划",
    "跨部门协调、组织管理和政策制定",
]


def compute_sbert_scores():
    """Use SBERT to decompose each ISCO occupation into auto/aug scores."""
    from sentence_transformers import SentenceTransformer

    print('\n[1] Computing SBERT-based automation/augmentation scores ...')
    t0 = time.time()

    # Load data
    ilo = pd.read_csv(ILO_CSV)
    zh = pd.read_csv(ZH_NAMES_CSV)
    merged = zh.merge(ilo[['isco08_4digit', 'mean_score', 'sd_score',
                            'exposure_gradient']],
                      on='isco08_4digit', how='inner')
    print(f'  {len(merged)} ISCO occupations loaded')

    # Build query strings (Chinese names + aliases, same as compute_jd_ai_score.py)
    queries = []
    for _, row in merged.iterrows():
        q = str(row['occupation_name_zh'])
        aliases = str(row.get('aliases_zh', '') or '').strip()
        if aliases and aliases != 'nan':
            q = q + ' ' + aliases
        queries.append(q)

    # Load SBERT model
    print(f'  Loading SBERT model: {MODEL_NAME} ...')
    model = SentenceTransformer(MODEL_NAME)

    # Encode everything
    print(f'  Encoding {len(queries)} ISCO occupations ...')
    occ_embs = model.encode(queries, batch_size=64, normalize_embeddings=True,
                            show_progress_bar=False)

    print(f'  Encoding {len(AUTOMATION_CONCEPTS)} automation concepts ...')
    auto_embs = model.encode(AUTOMATION_CONCEPTS, normalize_embeddings=True)

    print(f'  Encoding {len(AUGMENTATION_CONCEPTS)} augmentation concepts ...')
    aug_embs = model.encode(AUGMENTATION_CONCEPTS, normalize_embeddings=True)

    # Compute mean concept embeddings
    auto_center = auto_embs.mean(axis=0)
    auto_center = auto_center / np.linalg.norm(auto_center)
    aug_center = aug_embs.mean(axis=0)
    aug_center = aug_center / np.linalg.norm(aug_center)

    # Compute similarities
    sim_auto = occ_embs @ auto_center
    sim_aug = occ_embs @ aug_center

    # Decompose: share_auto + share_aug = 1
    total_sim = sim_auto + sim_aug
    share_auto = sim_auto / total_sim
    share_aug = sim_aug / total_sim

    # Weight by mean_score
    merged['sim_auto'] = sim_auto
    merged['sim_aug'] = sim_aug
    merged['share_auto'] = share_auto
    merged['share_aug'] = share_aug
    merged['automation_score'] = share_auto * merged['mean_score']
    merged['augmentation_score'] = share_aug * merged['mean_score']

    print(f'  Done in {time.time()-t0:.1f}s')
    print(f'\n  Similarity stats:')
    print(f'    sim_auto:  mean={sim_auto.mean():.4f}, std={sim_auto.std():.4f}')
    print(f'    sim_aug:   mean={sim_aug.mean():.4f}, std={sim_aug.std():.4f}')
    print(f'    share_auto: mean={share_auto.mean():.4f}, range=[{share_auto.min():.4f}, {share_auto.max():.4f}]')
    print(f'  Correlation(sim_auto, sim_aug) = {np.corrcoef(sim_auto, sim_aug)[0,1]:.4f}')
    print(f'  Correlation(automation_score, augmentation_score) = '
          f'{np.corrcoef(merged["automation_score"], merged["augmentation_score"])[0,1]:.4f}')

    # ISCO major group classification (robustness)
    merged['isco_major'] = merged['isco08_4digit'] // 1000
    group_map = {
        1: 'aug',   # Managers → augmentation
        2: 'aug',   # Professionals → augmentation
        3: 'mixed', # Technicians → mixed
        4: 'auto',  # Clerical → automation
        5: 'mixed', # Service/Sales → mixed
        6: 'auto',  # Agricultural → automation (less GenAI relevant)
        7: 'auto',  # Craft → automation
        8: 'auto',  # Machine operators → automation
        9: 'auto',  # Elementary → automation
    }
    merged['isco_type'] = merged['isco_major'].map(group_map)

    # Save decomposition
    out_cols = ['isco08_4digit', 'occupation_name_en', 'occupation_name_zh',
                'mean_score', 'sd_score', 'exposure_gradient',
                'sim_auto', 'sim_aug', 'share_auto', 'share_aug',
                'automation_score', 'augmentation_score',
                'isco_major', 'isco_type']
    merged[out_cols].to_csv(OUT / 'isco_decomposition.csv', index=False)
    print(f'  Saved isco_decomposition.csv')

    # Top-10 most automation-exposed
    print(f'\n  Top-10 automation-dominated (highest share_auto):')
    for _, r in merged.nlargest(10, 'share_auto').iterrows():
        print(f'    {r.occupation_name_zh} ({r.occupation_name_en[:35]}): '
              f'auto={r.share_auto:.3f}, ai={r.mean_score:.2f}')

    print(f'\n  Top-10 augmentation-dominated (highest share_aug):')
    for _, r in merged.nlargest(10, 'share_aug').iterrows():
        print(f'    {r.occupation_name_zh} ({r.occupation_name_en[:35]}): '
              f'aug={r.share_aug:.3f}, ai={r.mean_score:.2f}')

    return merged


def compute_gemini_scores():
    """Use Gemini LLM to score each ISCO occupation's automation vs augmentation
    potential, following Hampole, Rao & Tan (NBER WP 2025).

    For each of ~303 ISCO occupations, asks Gemini to estimate what share of the
    occupation's tasks are automation-prone vs augmentation-prone by GenAI.
    """
    import google.generativeai as genai

    print('\n[1] Computing Gemini-based automation/augmentation scores ...')
    t0 = time.time()

    api_key = os.environ.get('GEMINI_API_KEY')
    if not api_key:
        raise RuntimeError('GEMINI_API_KEY environment variable not set')
    genai.configure(api_key=api_key)

    # Load data
    ilo = pd.read_csv(ILO_CSV)
    zh = pd.read_csv(ZH_NAMES_CSV)
    merged = zh.merge(ilo[['isco08_4digit', 'mean_score', 'sd_score',
                            'exposure_gradient']],
                      on='isco08_4digit', how='inner')
    print(f'  {len(merged)} ISCO occupations loaded')

    # Check cache
    cache_path = OUT / 'gemini_scores_cache.json'
    cache = {}
    if cache_path.exists():
        with open(cache_path) as f:
            cache = json.load(f)
        print(f'  Loaded {len(cache)} cached scores')

    # Prepare Gemini model
    model = genai.GenerativeModel('gemini-2.0-flash')

    # System prompt following Hampole et al. (2025) methodology
    system_prompt = """你是一位劳动经济学专家。请评估以下职业中的工作任务被生成式AI（如ChatGPT、Copilot等大语言模型）影响的方式。

对于每个职业，请评估：
1. **自动化比例 (automation_share)**：该职业中有多大比例的核心任务可以被AI直接替代或自动完成？这些通常是重复性的、规则明确的、标准化的认知任务（如数据录入、格式转换、信息分类、模板化写作）。
2. **增强比例 (augmentation_share)**：该职业中有多大比例的核心任务会因AI而被增强（人类仍然主导，但AI提高效率和质量）？这些通常是需要判断力、创造力、人际交往、复杂分析的任务。

要求：
- automation_share + augmentation_share = 1.0
- 两个值都在 [0.0, 1.0] 范围内
- 请仅基于职业的典型工作内容判断，不要考虑该职业是否实际已采用AI
- 请严格按照JSON格式返回，不要包含任何其他文字"""

    # Process in batches of 10
    BATCH_SIZE = 10
    occupations = merged.to_dict('records')
    uncached = [(i, r) for i, r in enumerate(occupations)
                if str(r['isco08_4digit']) not in cache]
    print(f'  Need to score {len(uncached)} occupations (batch size={BATCH_SIZE})')

    for batch_start in range(0, len(uncached), BATCH_SIZE):
        batch = uncached[batch_start:batch_start + BATCH_SIZE]

        # Build prompt for batch
        occ_list = []
        for _, r in batch:
            occ_list.append(
                f"- ISCO {r['isco08_4digit']}: {r['occupation_name_zh']} "
                f"({r['occupation_name_en']})"
            )
        occ_text = '\n'.join(occ_list)

        prompt = f"""{system_prompt}

请为以下职业评分：
{occ_text}

请返回一个JSON数组，每个元素包含 isco08_4digit, automation_share, augmentation_share。
示例格式：
[{{"isco08_4digit": 1234, "automation_share": 0.3, "augmentation_share": 0.7}}]"""

        # Call Gemini with retry
        for attempt in range(3):
            try:
                response = model.generate_content(prompt)
                text = response.text.strip()
                # Extract JSON from response (handle markdown code blocks)
                if '```json' in text:
                    text = text.split('```json')[1].split('```')[0].strip()
                elif '```' in text:
                    text = text.split('```')[1].split('```')[0].strip()
                scores = json.loads(text)

                # Validate and cache
                for s in scores:
                    isco = str(s['isco08_4digit'])
                    auto = float(s['automation_share'])
                    aug = float(s['augmentation_share'])
                    # Normalize to sum to 1
                    total = auto + aug
                    if total > 0:
                        auto, aug = auto / total, aug / total
                    cache[isco] = {'automation_share': auto, 'augmentation_share': aug}

                # Save cache after each batch
                with open(cache_path, 'w') as f:
                    json.dump(cache, f, indent=2)

                n_done = min(batch_start + BATCH_SIZE, len(uncached))
                print(f'  Batch {batch_start//BATCH_SIZE + 1}: scored {len(scores)} '
                      f'occupations ({n_done}/{len(uncached)} total)')
                break

            except Exception as e:
                if attempt < 2:
                    wait = 10 * (attempt + 1)
                    print(f'  Retry {attempt+1} (wait {wait}s): {e}')
                    time.sleep(wait)
                else:
                    print(f'  FAILED batch starting at {batch_start}: {e}')
                    # Score individually as fallback
                    for _, r in batch:
                        isco = str(r['isco08_4digit'])
                        if isco not in cache:
                            try:
                                single_prompt = (
                                    f"{system_prompt}\n\n"
                                    f"请为以下职业评分：\n"
                                    f"- ISCO {r['isco08_4digit']}: {r['occupation_name_zh']} "
                                    f"({r['occupation_name_en']})\n\n"
                                    f"返回JSON: "
                                    f'[{{"isco08_4digit": {r["isco08_4digit"]}, '
                                    f'"automation_share": 0.5, "augmentation_share": 0.5}}]'
                                )
                                resp = model.generate_content(single_prompt)
                                txt = resp.text.strip()
                                if '```json' in txt:
                                    txt = txt.split('```json')[1].split('```')[0].strip()
                                elif '```' in txt:
                                    txt = txt.split('```')[1].split('```')[0].strip()
                                s = json.loads(txt)[0]
                                auto = float(s['automation_share'])
                                aug = float(s['augmentation_share'])
                                total = auto + aug
                                if total > 0:
                                    auto, aug = auto / total, aug / total
                                cache[isco] = {'automation_share': auto,
                                               'augmentation_share': aug}
                                time.sleep(5)
                            except Exception as e2:
                                print(f'    Individual fallback failed for {isco}: {e2}')
                                cache[isco] = {'automation_share': 0.5,
                                               'augmentation_share': 0.5}

                    with open(cache_path, 'w') as f:
                        json.dump(cache, f, indent=2)

        # Rate limiting — Gemini free tier: 15 RPM
        time.sleep(5)

    print(f'  Gemini scoring done in {time.time()-t0:.1f}s')
    print(f'  {len(cache)} occupations scored')

    # Apply scores to merged dataframe
    share_autos = []
    share_augs = []
    for _, r in merged.iterrows():
        isco = str(r['isco08_4digit'])
        if isco in cache:
            share_autos.append(cache[isco]['automation_share'])
            share_augs.append(cache[isco]['augmentation_share'])
        else:
            share_autos.append(0.5)
            share_augs.append(0.5)

    merged['share_auto'] = share_autos
    merged['share_aug'] = share_augs
    merged['sim_auto'] = merged['share_auto']  # compatibility with vis
    merged['sim_aug'] = merged['share_aug']
    merged['automation_score'] = merged['share_auto'] * merged['mean_score']
    merged['augmentation_score'] = merged['share_aug'] * merged['mean_score']

    # Diagnostics
    sa = np.array(share_autos)
    print(f'\n  Score stats:')
    print(f'    share_auto:  mean={sa.mean():.4f}, std={sa.std():.4f}, '
          f'range=[{sa.min():.4f}, {sa.max():.4f}]')
    print(f'    share_aug:   mean={(1-sa).mean():.4f}, std={(1-sa).std():.4f}')
    print(f'  Correlation(auto_score, aug_score) = '
          f'{np.corrcoef(merged["automation_score"], merged["augmentation_score"])[0,1]:.4f}')

    # ISCO major group classification (robustness)
    merged['isco_major'] = merged['isco08_4digit'] // 1000
    group_map = {
        1: 'aug', 2: 'aug', 3: 'mixed', 4: 'auto',
        5: 'mixed', 6: 'auto', 7: 'auto', 8: 'auto', 9: 'auto',
    }
    merged['isco_type'] = merged['isco_major'].map(group_map)

    # Save decomposition
    out_cols = ['isco08_4digit', 'occupation_name_en', 'occupation_name_zh',
                'mean_score', 'sd_score', 'exposure_gradient',
                'sim_auto', 'sim_aug', 'share_auto', 'share_aug',
                'automation_score', 'augmentation_score',
                'isco_major', 'isco_type']
    merged[out_cols].to_csv(OUT / 'isco_decomposition_gemini.csv', index=False)
    print(f'  Saved isco_decomposition_gemini.csv')

    # Top-10 most automation-exposed
    print(f'\n  Top-10 automation-dominated (highest share_auto):')
    for _, r in merged.nlargest(10, 'share_auto').iterrows():
        print(f'    {r.occupation_name_zh} ({r.occupation_name_en[:35]}): '
              f'auto={r.share_auto:.3f}, ai={r.mean_score:.2f}')

    print(f'\n  Top-10 augmentation-dominated (highest share_aug):')
    for _, r in merged.nlargest(10, 'share_aug').iterrows():
        print(f'    {r.occupation_name_zh} ({r.occupation_name_en[:35]}): '
              f'aug={r.share_aug:.3f}, ai={r.mean_score:.2f}')

    return merged


def aggregate_cells(decomp):
    """Load master data and aggregate to ISCO×year cells with auto/aug scores."""
    print('\n[2] Aggregating to ISCO × year cells ...')
    t0 = time.time()

    # Build lookup: isco -> (automation_score, augmentation_score, isco_type)
    lookup = {}
    for _, r in decomp.iterrows():
        lookup[int(r['isco08_4digit'])] = {
            'auto_score': float(r['automation_score']),
            'aug_score': float(r['augmentation_score']),
            'share_auto': float(r['share_auto']),
            'isco_type': r['isco_type'],
        }

    cols = ['year', 'isco08_4digit', 'entropy_score', 'ai_mean_score']
    cells = {}
    total = 0
    skipped = 0

    for i, chunk in enumerate(pd.read_csv(MASTER, usecols=cols,
                                           chunksize=CHUNK)):
        total += len(chunk)
        valid = chunk.dropna()
        skipped += len(chunk) - len(valid)
        valid = valid.copy()
        valid['isco'] = valid['isco08_4digit'].astype(int)
        valid['yr'] = valid['year'].astype(int)

        for (isco, yr), grp in valid.groupby(['isco', 'yr']):
            key = (isco, yr)
            n = len(grp)
            s_ent = grp['entropy_score'].sum()
            ai = float(grp['ai_mean_score'].iloc[0])

            if key not in cells:
                cells[key] = {'n': 0, 'sum_ent': 0.0, 'ai': ai}
            cells[key]['n'] += n
            cells[key]['sum_ent'] += s_ent

        if (i + 1) % 4 == 0:
            print(f'  Chunk {i+1}: {total:>10,} rows, {len(cells):,} cells')

    print(f'  Total: {total:,} rows, {skipped:,} skipped, {len(cells):,} cells')

    rows = []
    for (isco, yr), c in cells.items():
        info = lookup.get(isco, None)
        if info is None:
            continue
        rows.append({
            'isco': isco, 'year': yr,
            'n': c['n'],
            'mean_entropy': c['sum_ent'] / c['n'],
            'ai_score': c['ai'],
            'auto_score': info['auto_score'],
            'aug_score': info['aug_score'],
            'share_auto': info['share_auto'],
            'isco_type': info['isco_type'],
        })

    df = pd.DataFrame(rows).sort_values(['isco', 'year']).reset_index(drop=True)
    print(f'  Cell DataFrame: {len(df)} cells')
    print(f'  auto_score: mean={df.auto_score.mean():.4f}, '
          f'range=[{df.auto_score.min():.4f}, {df.auto_score.max():.4f}]')
    print(f'  aug_score:  mean={df.aug_score.mean():.4f}, '
          f'range=[{df.aug_score.min():.4f}, {df.aug_score.max():.4f}]')
    print(f'  Aggregation time: {time.time()-t0:.1f}s')

    return df


def model_1_isco_wls(df):
    """Model 1: ISCO-level WLS with auto/aug decomposition."""
    print('\n' + '=' * 70)
    print('Model 1: ISCO-level WLS with auto/aug decomposition')
    print('mean_entropy ~ C(year) * auto_score + C(year) * aug_score')
    print('=' * 70)

    formula = 'mean_entropy ~ C(year) * auto_score + C(year) * aug_score'
    m = smf.wls(formula, data=df, weights=df['n']).fit(cov_type='HC3')

    lines = []
    lines.append('Model 1: mean_entropy ~ C(year) * auto_score + C(year) * aug_score')
    lines.append(f'N = {len(df)} ISCO×year cells, WLS with HC3 SEs')
    lines.append(f'R² = {m.rsquared:.6f}, Adj R² = {m.rsquared_adj:.6f}')
    lines.append('=' * 70)

    # Automation interactions
    auto_ints = {k: v for k, v in m.params.items() if ':auto_score' in k}
    auto_pvals = {k: v for k, v in m.pvalues.items() if ':auto_score' in k}

    lines.append('\nAutomation × Year interactions:')
    lines.append(f'{"Term":<35} {"Coef":>10} {"SE":>10} {"p":>8}')
    lines.append('-' * 65)
    for k in sorted(auto_ints.keys()):
        yr = k.split('[T.')[1].split(']')[0]
        lines.append(f'  {yr} × auto_score'
                     f'  {auto_ints[k]:>10.6f}'
                     f'  {m.bse[k]:>10.6f}'
                     f'  {auto_pvals[k]:>8.4f}')
        print(f'  {yr} × auto: {auto_ints[k]:+.6f}  p={auto_pvals[k]:.4f}')

    # Joint F for automation
    auto_names = sorted(auto_ints.keys())
    r_auto = np.zeros((len(auto_names), len(m.params)))
    for i, name in enumerate(auto_names):
        r_auto[i, list(m.params.index).index(name)] = 1
    f_auto = m.f_test(r_auto)
    f_auto_val = float(np.asarray(f_auto.fvalue).flat[0])
    f_auto_p = float(np.asarray(f_auto.pvalue).flat[0])
    lines.append(f'\nJoint F (auto × year = 0): F = {f_auto_val:.4f}, p = {f_auto_p:.4f}')
    print(f'  Joint F (auto): {f_auto_val:.4f}, p = {f_auto_p:.4f}')

    # Augmentation interactions
    aug_ints = {k: v for k, v in m.params.items() if ':aug_score' in k}
    aug_pvals = {k: v for k, v in m.pvalues.items() if ':aug_score' in k}

    lines.append('\nAugmentation × Year interactions:')
    lines.append(f'{"Term":<35} {"Coef":>10} {"SE":>10} {"p":>8}')
    lines.append('-' * 65)
    for k in sorted(aug_ints.keys()):
        yr = k.split('[T.')[1].split(']')[0]
        lines.append(f'  {yr} × aug_score'
                     f'  {aug_ints[k]:>10.6f}'
                     f'  {m.bse[k]:>10.6f}'
                     f'  {aug_pvals[k]:>8.4f}')
        print(f'  {yr} × aug: {aug_ints[k]:+.6f}  p={aug_pvals[k]:.4f}')

    # Joint F for augmentation
    aug_names = sorted(aug_ints.keys())
    r_aug = np.zeros((len(aug_names), len(m.params)))
    for i, name in enumerate(aug_names):
        r_aug[i, list(m.params.index).index(name)] = 1
    f_aug = m.f_test(r_aug)
    f_aug_val = float(np.asarray(f_aug.fvalue).flat[0])
    f_aug_p = float(np.asarray(f_aug.pvalue).flat[0])
    lines.append(f'\nJoint F (aug × year = 0): F = {f_aug_val:.4f}, p = {f_aug_p:.4f}')
    print(f'  Joint F (aug): {f_aug_val:.4f}, p = {f_aug_p:.4f}')

    # Main effects
    lines.append('\nMain effects:')
    for v in ['auto_score', 'aug_score']:
        if v in m.params.index:
            lines.append(f'  {v}: coef={m.params[v]:.6f}, p={m.pvalues[v]:.4f}')

    with open(OUT / 'model_1_isco_wls.txt', 'w') as f:
        f.write('\n'.join(lines))

    return m, {
        'auto_f': f_auto_val, 'auto_p': f_auto_p,
        'aug_f': f_aug_val, 'aug_p': f_aug_p,
        'auto_ints': auto_ints, 'aug_ints': aug_ints,
        'auto_pvals': auto_pvals, 'aug_pvals': aug_pvals,
    }


def model_2_individual_twfe(decomp):
    """Model 2: Individual-level TWFE with auto/aug decomposition."""
    print('\n' + '=' * 70)
    print('Model 2: Individual TWFE with auto/aug decomposition')
    print('entropy ~ ISCO_FE + year_FE + year×auto_score + year×aug_score')
    print('ISCO-clustered SEs')
    print('=' * 70)

    # Build lookup
    lookup = {}
    for _, r in decomp.iterrows():
        lookup[int(r['isco08_4digit'])] = (
            float(r['automation_score']),
            float(r['augmentation_score']),
        )

    print('  Loading individual data ...')
    t0 = time.time()
    cols = ['year', 'entropy_score', 'isco08_4digit', 'ai_mean_score']
    chunks = []
    for chunk in pd.read_csv(MASTER, usecols=cols, chunksize=CHUNK):
        chunk = chunk.dropna()
        chunk['year'] = chunk['year'].astype(int)
        chunk['isco08_4digit'] = chunk['isco08_4digit'].astype(int)
        chunks.append(chunk)
    df = pd.concat(chunks, ignore_index=True)
    print(f'  Loaded {len(df):,} rows in {time.time()-t0:.1f}s')

    # Merge auto/aug scores
    df['auto_score'] = df['isco08_4digit'].map(lambda x: lookup.get(x, (np.nan, np.nan))[0])
    df['aug_score'] = df['isco08_4digit'].map(lambda x: lookup.get(x, (np.nan, np.nan))[1])
    df = df.dropna(subset=['auto_score', 'aug_score'])
    print(f'  After merging: {len(df):,} rows')

    years = sorted(df.year.unique())
    years_excl_ref = [y for y in years if y != REF_YEAR]

    # Create year dummies and interactions
    for y in years_excl_ref:
        df[f'yr_{y}'] = (df['year'] == y).astype(float)
        df[f'yr_{y}_x_auto'] = df[f'yr_{y}'] * df['auto_score']
        df[f'yr_{y}_x_aug'] = df[f'yr_{y}'] * df['aug_score']

    yr_cols = [f'yr_{y}' for y in years_excl_ref]
    auto_cols = [f'yr_{y}_x_auto' for y in years_excl_ref]
    aug_cols = [f'yr_{y}_x_aug' for y in years_excl_ref]
    all_cols = yr_cols + auto_cols + aug_cols + ['entropy_score']

    # FWL: demean by ISCO
    print('  Demeaning by ISCO (FWL) ...')
    t0 = time.time()
    group_means = df.groupby('isco08_4digit')[all_cols].transform('mean')
    demeaned = df[all_cols] - group_means
    print(f'  Done in {time.time()-t0:.1f}s')

    y = demeaned['entropy_score'].values
    X = sm.add_constant(demeaned[yr_cols + auto_cols + aug_cols].values)
    col_names = ['const'] + yr_cols + auto_cols + aug_cols

    print('  Running OLS with ISCO-clustered SEs ...')
    t0 = time.time()
    model = sm.OLS(y, X)
    res = model.fit(cov_type='cluster',
                    cov_kwds={'groups': df['isco08_4digit'].values})
    print(f'  Done in {time.time()-t0:.1f}s')

    lines = []
    lines.append(f'N = {len(df):,}')
    lines.append(f'ISCO clusters = {df.isco08_4digit.nunique()}')
    lines.append(f'R² (within) = {res.rsquared:.6f}')

    # Automation interactions
    lines.append('\nAutomation × Year:')
    lines.append(f'{"Year":<10} {"Coef":>12} {"SE":>12} {"p":>8}')
    lines.append('-' * 45)
    for col in sorted(auto_cols):
        yr = col.split('_')[1]
        idx = col_names.index(col)
        lines.append(f'{yr:<10} {res.params[idx]:>12.6f} '
                     f'{res.bse[idx]:>12.6f} {res.pvalues[idx]:>8.4f}')
        print(f'  {yr} × auto: {res.params[idx]:+.6f}  p={res.pvalues[idx]:.4f}')

    # Joint F for automation
    n_yr = len(yr_cols)
    n_auto = len(auto_cols)
    R_auto = np.zeros((n_auto, len(col_names)))
    for k in range(n_auto):
        R_auto[k, 1 + n_yr + k] = 1
    f_auto = res.f_test(R_auto)
    f_auto_val = float(np.asarray(f_auto.fvalue).flat[0])
    f_auto_p = float(np.asarray(f_auto.pvalue).flat[0])
    lines.append(f'\nJoint F (auto × year = 0): F = {f_auto_val:.4f}, p = {f_auto_p:.4f}')
    print(f'  Joint F (auto): {f_auto_val:.4f}, p = {f_auto_p:.4f}')

    # Augmentation interactions
    lines.append('\nAugmentation × Year:')
    lines.append(f'{"Year":<10} {"Coef":>12} {"SE":>12} {"p":>8}')
    lines.append('-' * 45)
    for col in sorted(aug_cols):
        yr = col.split('_')[1]
        idx = col_names.index(col)
        lines.append(f'{yr:<10} {res.params[idx]:>12.6f} '
                     f'{res.bse[idx]:>12.6f} {res.pvalues[idx]:>8.4f}')
        print(f'  {yr} × aug: {res.params[idx]:+.6f}  p={res.pvalues[idx]:.4f}')

    # Joint F for augmentation
    n_aug = len(aug_cols)
    R_aug = np.zeros((n_aug, len(col_names)))
    for k in range(n_aug):
        R_aug[k, 1 + n_yr + n_auto + k] = 1
    f_aug = res.f_test(R_aug)
    f_aug_val = float(np.asarray(f_aug.fvalue).flat[0])
    f_aug_p = float(np.asarray(f_aug.pvalue).flat[0])
    lines.append(f'\nJoint F (aug × year = 0): F = {f_aug_val:.4f}, p = {f_aug_p:.4f}')
    print(f'  Joint F (aug): {f_aug_val:.4f}, p = {f_aug_p:.4f}')

    # Post-2023 specific F tests
    post_years = [y for y in years_excl_ref if y >= 2023]
    lines.append(f'\n--- Post-2023 specific tests (years: {post_years}) ---')
    print(f'\n  Post-2023 specific F tests (years: {post_years}):')

    # Post-2023 automation interactions
    post_auto_indices = [col_names.index(f'yr_{y}_x_auto') for y in post_years]
    R_post_auto = np.zeros((len(post_auto_indices), len(col_names)))
    for i, idx_ in enumerate(post_auto_indices):
        R_post_auto[i, idx_] = 1
    f_post_auto = res.f_test(R_post_auto)
    f_post_auto_val = float(np.asarray(f_post_auto.fvalue).flat[0])
    f_post_auto_p = float(np.asarray(f_post_auto.pvalue).flat[0])
    lines.append(f'Joint F (post-2023 auto × year = 0): F = {f_post_auto_val:.4f}, '
                 f'p = {f_post_auto_p:.4f}')
    print(f'  Post-2023 Joint F (auto): {f_post_auto_val:.4f}, p = {f_post_auto_p:.4f}')

    # Post-2023 augmentation interactions
    post_aug_indices = [col_names.index(f'yr_{y}_x_aug') for y in post_years]
    R_post_aug = np.zeros((len(post_aug_indices), len(col_names)))
    for i, idx_ in enumerate(post_aug_indices):
        R_post_aug[i, idx_] = 1
    f_post_aug = res.f_test(R_post_aug)
    f_post_aug_val = float(np.asarray(f_post_aug.fvalue).flat[0])
    f_post_aug_p = float(np.asarray(f_post_aug.pvalue).flat[0])
    lines.append(f'Joint F (post-2023 aug × year = 0): F = {f_post_aug_val:.4f}, '
                 f'p = {f_post_aug_p:.4f}')
    print(f'  Post-2023 Joint F (aug): {f_post_aug_val:.4f}, p = {f_post_aug_p:.4f}')

    result_text = '\n'.join(lines)
    with open(OUT / 'model_2_individual_twfe.txt', 'w') as f:
        f.write('Model 2: Individual TWFE\n')
        f.write('entropy ~ ISCO_FE + year_FE + year×auto + year×aug\n')
        f.write('ISCO-clustered SEs\n')
        f.write('=' * 70 + '\n')
        f.write(result_text)

    return {
        'auto_f': f_auto_val, 'auto_p': f_auto_p,
        'aug_f': f_aug_val, 'aug_p': f_aug_p,
        'post_auto_p': f_post_auto_p, 'post_aug_p': f_post_aug_p,
    }


def model_3_binary_isco(decomp):
    """Model 3: Robustness — binary ISCO major group classification."""
    print('\n' + '=' * 70)
    print('Model 3: Binary DID with ISCO major group classification')
    print('entropy ~ ISCO_FE + year_FE + Post×Treated_auto + Post×Treated_aug')
    print('=' * 70)

    # Build lookup
    lookup = {}
    for _, r in decomp.iterrows():
        lookup[int(r['isco08_4digit'])] = r['isco_type']

    print('  Loading individual data ...')
    t0 = time.time()
    cols = ['year', 'entropy_score', 'isco08_4digit', 'ai_mean_score']
    chunks = []
    for chunk in pd.read_csv(MASTER, usecols=cols, chunksize=CHUNK):
        chunk = chunk.dropna()
        chunk['year'] = chunk['year'].astype(int)
        chunk['isco08_4digit'] = chunk['isco08_4digit'].astype(int)
        chunks.append(chunk)
    df = pd.concat(chunks, ignore_index=True)

    df['isco_type'] = df['isco08_4digit'].map(lookup)
    df = df.dropna(subset=['isco_type'])
    print(f'  Loaded {len(df):,} rows in {time.time()-t0:.1f}s')
    print(f'  Type distribution:')
    for t, n in df['isco_type'].value_counts().items():
        print(f'    {t}: {n:,} ({n/len(df)*100:.1f}%)')

    # Create treatment indicators
    df['treated_auto'] = (df['isco_type'] == 'auto').astype(float)
    df['treated_aug'] = (df['isco_type'] == 'aug').astype(float)
    df['post'] = (df['year'] >= POST_YEAR).astype(float)
    df['did_auto'] = df['treated_auto'] * df['post']
    df['did_aug'] = df['treated_aug'] * df['post']

    years = sorted(df.year.unique())
    years_excl_ref = [y for y in years if y != REF_YEAR]
    for y in years_excl_ref:
        df[f'yr_{y}'] = (df['year'] == y).astype(float)

    yr_cols = [f'yr_{y}' for y in years_excl_ref]
    treat_cols = ['treated_auto', 'treated_aug', 'did_auto', 'did_aug']
    all_cols = yr_cols + treat_cols + ['entropy_score']

    # FWL demean
    print('  Demeaning by ISCO ...')
    group_means = df.groupby('isco08_4digit')[all_cols].transform('mean')
    demeaned = df[all_cols] - group_means

    y = demeaned['entropy_score'].values
    X_cols = yr_cols + treat_cols
    X = sm.add_constant(demeaned[X_cols].values)
    col_names = ['const'] + X_cols

    model = sm.OLS(y, X)
    res = model.fit(cov_type='cluster',
                    cov_kwds={'groups': df['isco08_4digit'].values})

    lines = []
    lines.append(f'N = {len(df):,}')
    lines.append(f'ISCO clusters = {df.isco08_4digit.nunique()}')
    lines.append(f'R² (within) = {res.rsquared:.6f}')
    lines.append('')
    lines.append(f'{"Variable":<20} {"Coef":>12} {"SE":>12} {"p":>8}')
    lines.append('-' * 55)
    for name in treat_cols:
        idx = col_names.index(name)
        lines.append(f'{name:<20} {res.params[idx]:>12.6f} '
                     f'{res.bse[idx]:>12.6f} {res.pvalues[idx]:>8.4f}')
        print(f'  {name}: {res.params[idx]:+.6f}  p={res.pvalues[idx]:.4f}')

    # Key results
    did_auto_idx = col_names.index('did_auto')
    did_aug_idx = col_names.index('did_aug')
    lines.append(f'\n>>> DID auto: coef={res.params[did_auto_idx]:.6f}, '
                 f'p={res.pvalues[did_auto_idx]:.4f}, '
                 f'95% CI=[{res.conf_int()[did_auto_idx][0]:.6f}, '
                 f'{res.conf_int()[did_auto_idx][1]:.6f}]')
    lines.append(f'>>> DID aug:  coef={res.params[did_aug_idx]:.6f}, '
                 f'p={res.pvalues[did_aug_idx]:.4f}, '
                 f'95% CI=[{res.conf_int()[did_aug_idx][0]:.6f}, '
                 f'{res.conf_int()[did_aug_idx][1]:.6f}]')

    result_text = '\n'.join(lines)
    print(result_text)

    with open(OUT / 'model_3_binary_did.txt', 'w') as f:
        f.write('Model 3: Binary DID (ISCO major group classification)\n')
        f.write('auto = ISCO 4/6/7/8/9, aug = ISCO 1/2, mixed = ISCO 3/5\n')
        f.write('entropy ~ ISCO_FE + year_FE + treated_auto + treated_aug '
                '+ did_auto + did_aug\n')
        f.write('ISCO-clustered SEs\n')
        f.write('=' * 70 + '\n')
        f.write(result_text)

    return res


def visualize(decomp, df_cells, m1_info):
    """Generate plots."""
    print('\n[6] Generating visualizations ...')

    # ── Plot 1: Automation vs Augmentation score scatter ──
    fig, ax = plt.subplots(figsize=(10, 8))
    colors = {'Gradient 4': '#e74c3c', 'Gradient 3': '#e67e22',
              'Gradient 2': '#f1c40f', 'Gradient 1': '#2ecc71',
              'Minimal Exposure': '#3498db', 'Not Exposed': '#95a5a6'}
    for grad in ['Gradient 4', 'Gradient 3', 'Gradient 2', 'Gradient 1',
                 'Minimal Exposure', 'Not Exposed']:
        sub = decomp[decomp['exposure_gradient'] == grad]
        ax.scatter(sub['automation_score'], sub['augmentation_score'],
                   c=colors.get(grad, 'grey'), label=grad, s=30, alpha=0.7)
    ax.plot([0, 0.5], [0, 0.5], 'k--', lw=0.8, alpha=0.5, label='45° line')
    ax.set_xlabel('Automation Score')
    ax.set_ylabel('Augmentation Score')
    ax.set_title('ISCO Occupation Decomposition:\nAutomation vs Augmentation AI Exposure')
    ax.legend(fontsize=9, loc='upper left')
    plt.tight_layout()
    plt.savefig(OUT / 'auto_vs_aug_scatter.png', dpi=150)
    plt.close()
    print('  Saved auto_vs_aug_scatter.png')

    # ── Plot 2: Interaction coefficients comparison ──
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('AI Exposure × Year Interactions: Automation vs Augmentation\n'
                 '(ISCO-level WLS, HC3 SEs)', fontsize=13)

    for idx, (label, ints, pvals, color) in enumerate([
        ('Automation', m1_info['auto_ints'], m1_info['auto_pvals'], '#e74c3c'),
        ('Augmentation', m1_info['aug_ints'], m1_info['aug_pvals'], '#2980b9'),
    ]):
        ax = axes[idx]
        years = []
        coefs = []
        ses = []
        for k in sorted(ints.keys()):
            yr = int(k.split('[T.')[1].split(']')[0])
            years.append(yr)
            coefs.append(ints[k])
            ses.append(m1_info.get('model', None))

        # Get SEs from model
        m = m1_info.get('model_obj', None)
        if m is None:
            # Reconstruct from info
            ses_vals = []
            for k in sorted(ints.keys()):
                ses_vals.append(0.01)  # placeholder
        else:
            ses_vals = [m.bse[k] for k in sorted(ints.keys())]

        bar_colors = ['#c0392b' if y >= 2023 else color for y in years]
        ax.bar(range(len(years)), coefs,
               yerr=[1.96 * s for s in ses_vals],
               capsize=3, alpha=0.75, color=bar_colors, edgecolor='grey', lw=0.5)
        ax.axhline(0, color='black', lw=0.8)
        ax.set_xticks(range(len(years)))
        ax.set_xticklabels(years, rotation=45, fontsize=9)
        key = 'auto_p' if idx == 0 else 'aug_p'
        ax.set_title(f'{label}\nJoint F p = {m1_info[key]:.4f}')
        ax.set_ylabel('Year × score coefficient')

    plt.tight_layout()
    plt.savefig(OUT / 'auto_aug_interactions.png', dpi=150)
    plt.close()
    print('  Saved auto_aug_interactions.png')

    # ── Plot 3: Entropy trends by auto/aug type ──
    fig, ax = plt.subplots(figsize=(10, 7))

    for label, mask, color in [
        ('Automation-type', df_cells['isco_type'] == 'auto', '#e74c3c'),
        ('Augmentation-type', df_cells['isco_type'] == 'aug', '#2980b9'),
        ('Mixed-type', df_cells['isco_type'] == 'mixed', '#27ae60'),
    ]:
        sub = df_cells[mask]
        yearly = sub.groupby('year').apply(
            lambda g: np.average(g['mean_entropy'], weights=g['n'])
        )
        ax.plot(yearly.index, yearly.values, 'o-', color=color,
                label=label, markersize=5)

    ax.axvline(2022.5, color='red', ls=':', lw=1, alpha=0.7)
    ax.set_xlabel('Year')
    ax.set_ylabel('Weighted Mean Entropy')
    ax.set_title('Entropy Trends by ISCO Type\n'
                 '(auto=ISCO 4/6/7/8/9, aug=ISCO 1/2, mixed=ISCO 3/5)')
    ax.legend()
    plt.tight_layout()
    plt.savefig(OUT / 'entropy_by_isco_type.png', dpi=150)
    plt.close()
    print('  Saved entropy_by_isco_type.png')


# ── Main ────────────────────────────────────────────────────────────
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', choices=['sbert', 'gemini'], default='gemini',
                        help='Scoring method: sbert (concept similarity) or gemini (LLM)')
    args = parser.parse_args()

    t_start = time.time()

    print('=' * 70)
    print(f'EXPERIMENT 1: Automation vs Augmentation Decomposition (method={args.method})')
    print('=' * 70)

    # Step 1: Score occupations
    if args.method == 'gemini':
        decomp = compute_gemini_scores()
    else:
        decomp = compute_sbert_scores()

    # Step 2: Aggregate cells
    df_cells = aggregate_cells(decomp)

    # Step 3: Model 1 — ISCO-level WLS
    m1, m1_info = model_1_isco_wls(df_cells)
    m1_info['model_obj'] = m1

    # Step 4: Model 2 — Individual TWFE
    m2_info = model_2_individual_twfe(decomp)

    # Step 5: Model 3 — Binary DID (robustness)
    m3 = model_3_binary_isco(decomp)

    # Step 6: Visualization
    visualize(decomp, df_cells, m1_info)

    elapsed = time.time() - t_start
    print(f'\n{"=" * 70}')
    print(f'DONE in {elapsed:.1f}s')
    print(f'\nSUMMARY:')
    print(f'  Model 1 (ISCO WLS):')
    print(f'    Auto Joint F p = {m1_info["auto_p"]:.4f}')
    print(f'    Aug  Joint F p = {m1_info["aug_p"]:.4f}')
    print(f'  Model 2 (Individual TWFE):')
    print(f'    Auto Joint F p = {m2_info["auto_p"]:.4f}')
    print(f'    Aug  Joint F p = {m2_info["aug_p"]:.4f}')
    print(f'\nAll results saved to {OUT}')
    for f in sorted(OUT.glob('*')):
        print(f'  {f.name}')
