#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path('/Users/yu/code/code2601/TY')
CLEANED = PROJECT_ROOT / 'data/processed/cleaned/all_in_one1.csv'
HET_DIR = PROJECT_ROOT / 'data/Heterogeneity'
MASTER = HET_DIR / 'master_industry_analysis.csv'
LOOKUP_OUT = HET_DIR / 'company_primary_category_lookup.csv'
YEARLY_PRIMARY_OUT = HET_DIR / 'yearly_primary_category_metrics.csv'
YEARLY_PRIMARY_VALID_OUT = HET_DIR / 'yearly_primary_category_metrics_valid500.csv'
GROWTH_PRIMARY_OUT = HET_DIR / 'primary_category_growth_2016_2025.csv'
GROWTH_INDUSTRY_2025_OUT = HET_DIR / 'industry_growth_2016_2025.csv'


def collect_master_companies() -> set[str]:
    companies = set()
    for ch in pd.read_csv(MASTER, usecols=['企业名称'], chunksize=300000, low_memory=False):
        s = ch['企业名称'].fillna('').astype(str).str.strip()
        companies.update(x for x in s.tolist() if x)
    return companies


def build_company_primary_lookup(companies: set[str]) -> dict[str, tuple[str, float]]:
    pair_counts: dict[tuple[str, str], int] = defaultdict(int)
    total_rows = 0
    valid_rows = 0

    usecols = ['企业名称', '初级分类']
    for i, ch in enumerate(pd.read_csv(CLEANED, usecols=usecols, chunksize=300000, low_memory=False), 1):
        total_rows += len(ch)
        ch['企业名称'] = ch['企业名称'].fillna('').astype(str).str.strip()
        ch['初级分类'] = ch['初级分类'].fillna('').astype(str).str.strip()

        ch = ch[
            (ch['企业名称'].isin(companies))
            & (ch['初级分类'] != '')
            & (ch['初级分类'].str.lower() != 'nan')
        ]
        valid_rows += len(ch)

        if not ch.empty:
            grp = ch.groupby(['企业名称', '初级分类']).size()
            for (comp, cate), cnt in grp.items():
                pair_counts[(str(comp), str(cate))] += int(cnt)

        if i % 5 == 0:
            print(f'[lookup] chunks={i}, scanned={total_rows:,}, valid_rows={valid_rows:,}, pair_keys={len(pair_counts):,}')

    total_by_company: dict[str, int] = defaultdict(int)
    best_by_company: dict[str, tuple[str, int]] = {}
    for (comp, cate), cnt in pair_counts.items():
        total_by_company[comp] += cnt
        if comp not in best_by_company or cnt > best_by_company[comp][1]:
            best_by_company[comp] = (cate, cnt)

    lookup: dict[str, tuple[str, float]] = {}
    with LOOKUP_OUT.open('w', encoding='utf-8-sig', newline='') as f:
        wr = csv.writer(f)
        wr.writerow(['企业名称', '初级分类', 'confidence'])
        for comp, (cate, cnt) in best_by_company.items():
            conf = cnt / max(1, total_by_company[comp])
            lookup[comp] = (cate, conf)
            wr.writerow([comp, cate, f'{conf:.4f}'])

    print(f'[lookup] saved: {LOOKUP_OUT}')
    print(f'[lookup] size: {len(lookup):,}')
    return lookup


def enrich_master(lookup: dict[str, tuple[str, float]]) -> None:
    tmp = MASTER.with_suffix('.tmp.csv')
    total = 0
    hit = 0
    first = True

    for i, ch in enumerate(pd.read_csv(MASTER, chunksize=300000, low_memory=False), 1):
        ch['企业名称'] = ch['企业名称'].fillna('').astype(str).str.strip()

        cat = []
        conf = []
        src = []
        for comp in ch['企业名称'].tolist():
            if comp in lookup:
                c, cf = lookup[comp]
                cat.append(c)
                conf.append(cf)
                src.append('company_primary_category_lookup')
                hit += 1
            else:
                cat.append('')
                conf.append(0.0)
                src.append('missing')

        ch['初级分类'] = cat
        ch['初级分类_confidence'] = conf
        ch['初级分类_source'] = src

        ch.to_csv(tmp, mode='w' if first else 'a', header=first, index=False, encoding='utf-8-sig')
        first = False
        total += len(ch)

        if i % 5 == 0:
            print(f'[master] chunks={i}, rows={total:,}, hit={hit:,} ({hit/max(1,total):.2%})')

    tmp.replace(MASTER)
    print(f'[master] updated: {MASTER}')
    print(f'[master] final hit ratio: {hit/max(1,total):.2%}')


def aggregate_primary_category() -> None:
    stats: dict[tuple[int, str], list[float]] = defaultdict(lambda: [0, 0.0, 0.0])

    usecols = ['year', 'entropy_score', 'hhi_score', '初级分类']
    for ch in pd.read_csv(MASTER, usecols=usecols, chunksize=300000, low_memory=False):
        ch['year'] = pd.to_numeric(ch['year'], errors='coerce')
        ch['entropy_score'] = pd.to_numeric(ch['entropy_score'], errors='coerce')
        ch['hhi_score'] = pd.to_numeric(ch['hhi_score'], errors='coerce')
        ch['初级分类'] = ch['初级分类'].fillna('').astype(str).str.strip()

        ch = ch.dropna(subset=['year', 'entropy_score', 'hhi_score'])
        ch = ch[ch['初级分类'] != '']
        if ch.empty:
            continue

        ch['year'] = ch['year'].astype(int)
        grp = ch.groupby(['year', '初级分类'], as_index=False).agg(
            sample_n=('entropy_score', 'size'),
            sum_entropy=('entropy_score', 'sum'),
            sum_hhi=('hhi_score', 'sum'),
        )
        for _, r in grp.iterrows():
            k = (int(r['year']), str(r['初级分类']))
            stats[k][0] += int(r['sample_n'])
            stats[k][1] += float(r['sum_entropy'])
            stats[k][2] += float(r['sum_hhi'])

    rows = []
    for (y, c), (n, se, sh) in stats.items():
        rows.append({
            'year': y,
            '初级分类': c,
            'sample_n': n,
            'mean_entropy': se / n,
            'mean_hhi': sh / n,
        })

    yearly = pd.DataFrame(rows)
    yearly['valid_sample'] = yearly['sample_n'] >= 500
    yearly = yearly.sort_values(['year', 'sample_n'], ascending=[True, False])
    yearly.to_csv(YEARLY_PRIMARY_OUT, index=False, encoding='utf-8-sig')
    yearly_valid = yearly[yearly['valid_sample']].copy()
    yearly_valid.to_csv(YEARLY_PRIMARY_VALID_OUT, index=False, encoding='utf-8-sig')

    # growth 2016 -> 2025
    keep = set(yearly.groupby('初级分类')['sample_n'].sum().loc[lambda s: s >= 10000].index)
    yearly_keep = yearly_valid[yearly_valid['初级分类'].isin(keep)].copy()
    y16 = yearly_keep[yearly_keep['year'] == 2016][['初级分类', 'sample_n', 'mean_entropy']].rename(
        columns={'sample_n': 'n_2016', 'mean_entropy': 'entropy_2016'}
    )
    y25 = yearly_keep[yearly_keep['year'] == 2025][['初级分类', 'sample_n', 'mean_entropy']].rename(
        columns={'sample_n': 'n_2025', 'mean_entropy': 'entropy_2025'}
    )
    growth = y16.merge(y25, on='初级分类', how='inner')
    growth['delta_entropy'] = growth['entropy_2025'] - growth['entropy_2016']
    growth['growth_rate_pct'] = growth['delta_entropy'] / growth['entropy_2016'] * 100
    growth = growth.sort_values('growth_rate_pct', ascending=False)
    growth.to_csv(GROWTH_PRIMARY_OUT, index=False, encoding='utf-8-sig')

    print(f'[agg] saved: {YEARLY_PRIMARY_OUT}')
    print(f'[agg] saved: {YEARLY_PRIMARY_VALID_OUT}')
    print(f'[agg] saved: {GROWTH_PRIMARY_OUT}')


def build_industry_growth_2016_2025() -> None:
    src = HET_DIR / 'yearly_industry_metrics.csv'
    if not src.exists():
        return
    df = pd.read_csv(src)
    y16 = df[df['year'] == 2016][['standard_industry', 'sample_n', 'mean_entropy']].rename(
        columns={'sample_n': 'n_2016', 'mean_entropy': 'entropy_2016'}
    )
    y25 = df[df['year'] == 2025][['standard_industry', 'sample_n', 'mean_entropy']].rename(
        columns={'sample_n': 'n_2025', 'mean_entropy': 'entropy_2025'}
    )
    g = y16.merge(y25, on='standard_industry', how='inner')
    g['delta_entropy'] = g['entropy_2025'] - g['entropy_2016']
    g['growth_rate_pct'] = g['delta_entropy'] / g['entropy_2016'] * 100
    g = g.sort_values('growth_rate_pct', ascending=False)
    g.to_csv(GROWTH_INDUSTRY_2025_OUT, index=False, encoding='utf-8-sig')
    print(f'[agg] saved: {GROWTH_INDUSTRY_2025_OUT}')


def main():
    print('Step 1/4 collect master companies...')
    companies = collect_master_companies()
    print(f'companies: {len(companies):,}')

    print('Step 2/4 build company->初级分类 lookup...')
    lookup = build_company_primary_lookup(companies)

    print('Step 3/4 enrich master with 初级分类...')
    enrich_master(lookup)

    print('Step 4/4 aggregate by 初级分类 and 2016-2025 growth...')
    aggregate_primary_category()
    build_industry_growth_2016_2025()
    print('Done')


if __name__ == '__main__':
    main()
