#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import csv
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import pandas as pd

PROJECT_ROOT = Path('/Users/yu/code/code2601/TY')
WINDOWS_DIR = PROJECT_ROOT / 'data/processed/windows'
PROCESSED_CORPUS = PROJECT_ROOT / 'output/processed_corpus.jsonl'
FINAL_RESULTS_SORTED = PROJECT_ROOT / 'data/Heterogeneity/final_results_sample_sorted.csv'

HET_DIR = PROJECT_ROOT / 'data/Heterogeneity'
COMPANY_INDUSTRY_LOOKUP = HET_DIR / 'company_industry_lookup.csv'
COMPANY_PRIMARY_LOOKUP = HET_DIR / 'company_primary_category_lookup.csv'
EVENTS_CSV = PROJECT_ROOT / 'output/lda/alignment/topic_evolution_events.csv'

TMP_YEAR_DIR = HET_DIR / 'tmp_joint_year_parts'
RECON_COMPANY_JOB_SORTED = HET_DIR / 'reconstructed_id_company_job_sorted.csv'
MASTER_JOINT = HET_DIR / 'master_joint_industry20_analysis.csv'
YEARLY_ENTROPY = HET_DIR / 'yearly_industry20_entropy_metrics.csv'
ALIGN_SURVIVAL = HET_DIR / 'industry20_alignment_survival_metrics.csv'
DIAG = HET_DIR / 'joint_mapping_diagnostics.csv'
VALID_UNIT_THRESHOLD = 50

REPLACEMENTS = {"c++": "cpp", "c#": "csharp", ".net": "dotnet"}
ENGLISH_ALLOW = {"python", "java", "sql", "c", "r", "go", "cpp", "csharp", "dotnet"}
DEFAULT_STOPWORDS = set([
    "的","了","在","是","有","和","与","或","等","能","会","及","对","可","为","被","把","让","给","向","从","到","以","于",
    "个","这","那","一","不","也","要","就","都","而","但","如","所","他","她","它","们","我","你","上","下","中","内","外",
    "职责","任职","要求","优先","具备","良好","负责","具有","熟悉","掌握","了解","相关","以上","以下","经验","工作","能力",
    "岗位","公司","企业","团队","一定","优秀","专业","学历","年","及以上","左右","不限","若干","可以","需要","进行","开展",
    "完成","参与","支持","提供","协助","配合","执行","根据","按照","通过","利用","使用","包括","主要","其他","相应","有效","合理","准确","及时",
])
MIN_TOKENS = 5

INDUSTRY20 = {
    'A': 'A 农、林、牧、渔业',
    'B': 'B 采矿业',
    'C': 'C 制造业',
    'D': 'D 电力、热力、燃气及水生产和供应业',
    'E': 'E 建筑业',
    'F': 'F 批发和零售业',
    'G': 'G 交通运输、仓储和邮政业',
    'H': 'H 住宿和餐饮业',
    'I': 'I 信息传输、软件和信息技术服务业',
    'J': 'J 金融业',
    'K': 'K 房地产业',
    'L': 'L 租赁和商务服务业',
    'M': 'M 科学研究和技术服务业',
    'N': 'N 水利、环境和公共设施管理业',
    'O': 'O 居民服务、修理和其他服务业',
    'P': 'P 教育',
    'Q': 'Q 卫生和社会工作',
    'R': 'R 文化、体育和娱乐业',
    'S': 'S 公共管理、社会保障和社会组织',
    'T': 'T 国际组织',
    'U': 'U 未识别/其他',
}

KEYWORDS = {
    'A': ['农业', '农林', '农牧', '渔业', '养殖', '种植', '畜牧', '农机', '农场'],
    'B': ['采矿', '矿业', '煤矿', '矿山', '石油开采', '天然气开采'],
    'C': ['制造', '生产', '工厂', '车间', '加工', '机械', '电子厂', '化工', '装配', '质检'],
    'D': ['电力', '热力', '燃气', '供水', '水务', '电网', '供电'],
    'E': ['建筑', '施工', '土木', '工程', '装修', '建造', '监理'],
    'F': ['零售', '批发', '门店', '商超', '商场', '导购', '店长'],
    'G': ['物流', '仓储', '运输', '快递', '配送', '货运', '司机', '邮政', '供应链'],
    'H': ['酒店', '餐饮', '民宿', '厨师', '服务员', '前厅'],
    'I': ['互联网', '软件', '开发', '程序员', '前端', '后端', '算法', '数据', '运维', '测试', '网络', '信息技术'],
    'J': ['金融', '银行', '证券', '保险', '基金', '信托', '投资', '风控', '审计'],
    'K': ['房地产', '物业', '置业', '楼盘', '房产中介', '售楼'],
    'L': ['人力资源', '招聘', '外包', '商务服务', '企业服务', '猎头', '咨询顾问'],
    'M': ['科研', '研发', '检测', '认证', '实验室', '设计', '技术服务', '专利'],
    'N': ['环保', '环境', '水利', '环卫', '园林', '公共设施'],
    'O': ['家政', '维修', '修理', '美容', '美发', '保健', '洗浴', '家电维修'],
    'P': ['教育', '培训', '学校', '教师', '讲师', '教研', '辅导'],
    'Q': ['医疗', '医院', '护理', '医生', '药师', '康复', '卫生', '社工'],
    'R': ['传媒', '影视', '娱乐', '体育', '文化', '演艺', '出版', '新媒体'],
    'S': ['政府', '公务员', '事业单位', '社会保障', '社会组织', '公共管理'],
    'T': ['国际组织', '联合国', '使馆', '领事馆', '外交'],
}

WINDOW_YEAR = {
    'window_2016_2017': [2016, 2017],
    'window_2018_2019': [2018, 2019],
    'window_2020_2021': [2020, 2021],
    'window_2022_2023': [2022, 2023],
    'window_2024_2025': [2024, 2025],
}

YEAR_TO_SOURCE_PAIR = {
    2016: ('window_2016_2017', 'window_2018_2019'),
    2017: ('window_2016_2017', 'window_2018_2019'),
    2018: ('window_2018_2019', 'window_2020_2021'),
    2019: ('window_2018_2019', 'window_2020_2021'),
    2020: ('window_2020_2021', 'window_2022_2023'),
    2021: ('window_2020_2021', 'window_2022_2023'),
    2022: ('window_2022_2023', 'window_2024_2025'),
    2023: ('window_2022_2023', 'window_2024_2025'),
}

LEGAL_SUFFIXES = [
    '集团股份有限公司',
    '股份有限公司',
    '有限责任公司',
    '集团有限公司',
    '控股有限公司',
    '有限公司',
    '集团',
]

BRANCH_SUFFIXES = [
    '分公司',
    '子公司',
    '办事处',
    '代表处',
]

REGION_PREFIXES = [
    '北京', '上海', '广州', '深圳', '杭州', '南京', '苏州', '天津', '重庆', '武汉', '成都', '西安', '郑州',
    '青岛', '厦门', '宁波', '长沙', '沈阳', '大连', '福州', '济南', '哈尔滨', '长春', '合肥', '南昌',
    '石家庄', '太原', '昆明', '贵阳', '南宁', '海口', '呼和浩特', '兰州', '银川', '乌鲁木齐', '拉萨',
    '内蒙古', '广西', '西藏', '宁夏', '新疆', '香港', '澳门',
    '河北', '山西', '辽宁', '吉林', '黑龙江', '江苏', '浙江', '安徽', '福建', '江西', '山东', '河南',
    '湖北', '湖南', '广东', '海南', '四川', '贵州', '云南', '陕西', '甘肃', '青海', '台湾',
]
REGION_PATTERN = '|'.join(sorted({re.escape(x) for x in REGION_PREFIXES}, key=len, reverse=True))
BRANCH_WITH_REGION_RE = re.compile(
    rf'(?:{REGION_PATTERN})(?:市|省|区|县)?(?:分公司|子公司|办事处|代表处)$'
)


def parse_year(v: str) -> Optional[int]:
    if v is None:
        return None
    s = str(v).strip()
    if not s:
        return None
    try:
        return int(float(s))
    except Exception:
        return None


def code_from_std_label(label: str) -> str:
    s = str(label or '').strip()
    if not s:
        return 'U'
    m = re.match(r'^([A-U])\b', s)
    return m.group(1) if m else 'U'


def _norm_text(s: str) -> str:
    s = str(s or '').strip().lower()
    s = s.replace('/', '_').replace('|', '_').replace(' ', '')
    return s


def canonical_company_name(name: str) -> str:
    """
    Normalize company names for robust prior matching:
    1) remove common punctuation/symbols
    2) remove (中国)/(china) markers
    3) strip legal/branch suffixes
    4) strip common region prefixes
    """
    s = str(name or '').strip().lower()
    if not s:
        return ''

    s = (
        s.replace('（', '(').replace('）', ')')
         .replace('【', '[').replace('】', ']')
         .replace('“', '"').replace('”', '"')
         .replace('‘', "'").replace('’', "'")
    )
    s = re.sub(r'[\s·\.,，。/\\|_\-;；:：!！?？"\'\[\]{}<>《》]+', '', s)
    s = s.replace('(中国)', '').replace('(china)', '')
    s = s.replace('（中国）', '').replace('（china）', '')
    s = s.replace('()', '')

    changed = True
    while changed and s:
        changed = False
        # remove trailing "地名+分支后缀" (e.g., 武汉分公司, 深圳办事处)
        s2 = BRANCH_WITH_REGION_RE.sub('', s)
        if s2 != s and len(s2) >= 2:
            s = s2
            changed = True

        for suf in LEGAL_SUFFIXES + BRANCH_SUFFIXES:
            if s.endswith(suf) and len(s) > len(suf) + 1:
                s = s[: -len(suf)]
                changed = True

    changed = True
    while changed and s:
        changed = False
        for pref in REGION_PREFIXES:
            pref_hit = None
            if s.startswith(pref + '市'):
                pref_hit = pref + '市'
            elif s.startswith(pref + '省'):
                pref_hit = pref + '省'
            elif s.startswith(pref):
                pref_hit = pref

            if pref_hit and len(s) > len(pref_hit) + 1:
                s = s[len(pref_hit):]
                changed = True

    # clean leftover leading administrative marker
    s = re.sub(r'^[市省]+', '', s)

    return s


def map_to_standard_industry(raw: str) -> str:
    x = _norm_text(raw)
    if not x:
        return INDUSTRY20['U']

    if any(k in x for k in ['国际组织', '联合国', '使馆', '领事馆']):
        return INDUSTRY20['T']
    if any(k in x for k in ['政府', '公共管理', '社会保障', '社会组织', '事业单位', '机关']):
        return INDUSTRY20['S']
    if any(k in x for k in ['农', '林', '牧', '渔', '养殖', '种植', '农副']):
        return INDUSTRY20['A']
    if any(k in x for k in ['采矿', '矿产', '煤炭', '矿业', '石油开采', '天然气开采']):
        return INDUSTRY20['B']
    if any(k in x for k in ['电力', '热力', '燃气', '自来水', '供水', '供电']):
        return INDUSTRY20['D']
    if any(k in x for k in ['房地产', '房产', '物业', '地产', '房屋中介']):
        return INDUSTRY20['K']
    if any(k in x for k in ['建筑', '施工', '土木', '装修', '建材', '工程施工', '建筑设备安装']):
        return INDUSTRY20['E']
    if any(k in x for k in ['互联网', 'it', '信息技术', '软件', '通信', '网络', '云计算', '大数据', '人工智能', '计算机']):
        return INDUSTRY20['I']
    if any(k in x for k in ['银行', '保险', '证券', '期货', '基金', '投资', '融资', '信托', '金融']):
        return INDUSTRY20['J']
    if any(k in x for k in ['学校', '教育', '培训', '辅导', '学历教育']):
        return INDUSTRY20['P']
    if any(k in x for k in ['医院', '卫生', '医疗服务', '护理', '康复', '社会工作', '养老服务']):
        return INDUSTRY20['Q']
    if any(k in x for k in ['文化', '体育', '娱乐', '影视', '传媒', '出版', '新媒体', '广告']):
        return INDUSTRY20['R']
    if any(k in x for k in ['交通', '运输', '物流', '仓储', '邮政', '客运', '货运', '快递']):
        return INDUSTRY20['G']
    if any(k in x for k in ['餐饮', '酒店', '住宿', '民宿']):
        return INDUSTRY20['H']
    if any(k in x for k in ['批发', '零售', '商贸', '贸易', '电子商务']):
        return INDUSTRY20['F']
    if any(k in x for k in ['科研', '研究', '检测', '认证', '工程设计', '专业技术', '专利', '技术服务']):
        return INDUSTRY20['M']
    if any(k in x for k in ['水利', '环保', '环境', '公共设施', '园林', '环卫']):
        return INDUSTRY20['N']
    if any(k in x for k in ['租赁', '人力资源', '企业服务', '咨询', '法律', '翻译', '商务服务', '代理']):
        return INDUSTRY20['L']
    if any(k in x for k in ['居民服务', '维修', '修理', '家政', '美容', '美发', '保健', '洗浴']):
        return INDUSTRY20['O']
    if any(k in x for k in ['制造', '加工', '电子设备', '机械', '半导体', '汽车制造', '医药制造', '化工', '金属制品', '纺织', '服装', '家具', '印刷', '包装', '仪器仪表', '食品', '饮料', '工业']):
        return INDUSTRY20['C']
    if '服务' in x:
        return INDUSTRY20['O']
    return INDUSTRY20['U']


def clean_text(text: str) -> str:
    if not text:
        return ''
    t = str(text).lower()
    for src, dst in REPLACEMENTS.items():
        t = t.replace(src, dst)
    t = re.sub(r'[^a-z\u4e00-\u9fa5]+', ' ', t)
    t = re.sub(r'\s+', ' ', t).strip()
    return t


def is_chinese_token(token: str) -> bool:
    return bool(re.search(r'[\u4e00-\u9fa5]', token))


def tokenize(text: str, stopwords: Set[str]) -> List[str]:
    import jieba
    t = clean_text(text)
    if not t:
        return []
    words = jieba.lcut(t)
    tokens: List[str] = []
    for w in words:
        if w in stopwords or w.isdigit():
            continue
        if is_chinese_token(w):
            if len(w) >= 2:
                tokens.append(w)
        else:
            if w in ENGLISH_ALLOW:
                tokens.append(w)
    return tokens


def apply_bigrams(tokens: List[str], bigrams: Set[str]) -> List[str]:
    if not tokens:
        return tokens
    merged: List[str] = []
    i = 0
    while i < len(tokens) - 1:
        pair = tokens[i] + '_' + tokens[i + 1]
        if pair in bigrams:
            merged.append(pair)
            i += 2
        else:
            merged.append(tokens[i])
            i += 1
    if i == len(tokens) - 1:
        merged.append(tokens[-1])
    return merged


def window_files() -> List[Path]:
    files = [
        p for p in WINDOWS_DIR.glob('window_*.csv')
        if re.match(r'window_\d{4}_\d{4}\.csv', p.name)
    ]
    return sorted(files)


def load_bigrams() -> Set[str]:
    print('[1/6] loading bigrams from processed_corpus ...')
    bigrams: Set[str] = set()
    with PROCESSED_CORPUS.open('r', encoding='utf-8') as f:
        for i, line in enumerate(f, 1):
            obj = json.loads(line)
            for t in obj.get('tokens', []):
                if '_' in t:
                    bigrams.add(t)
            if i % 1_500_000 == 0:
                print(f'  scanned {i:,}, bigrams={len(bigrams):,}')
    print(f'  total bigrams: {len(bigrams):,}')
    return bigrams


def reconstruct_company_job_by_id(bigrams: Set[str]) -> None:
    print('[2/6] reconstructing id->(company,job_title) ...')
    HET_DIR.mkdir(parents=True, exist_ok=True)
    TMP_YEAR_DIR.mkdir(parents=True, exist_ok=True)

    writers: Dict[int, csv.writer] = {}
    files: Dict[int, object] = {}

    def get_writer(year: int) -> csv.writer:
        if year not in writers:
            fp = TMP_YEAR_DIR / f'year_{year}.csv'
            fo = fp.open('w', encoding='utf-8-sig', newline='')
            wr = csv.writer(fo)
            wr.writerow(['id', 'year', '企业名称', '招聘岗位'])
            writers[year] = wr
            files[year] = fo
        return writers[year]

    stopwords = set(DEFAULT_STOPWORDS)
    year_seq: Dict[int, int] = defaultdict(int)
    scanned = 0
    kept = 0

    for fp in window_files():
        print(f'  processing {fp.name} ...')
        with fp.open('r', encoding='utf-8-sig', errors='replace', newline='') as f:
            rd = csv.DictReader(f)
            for row in rd:
                scanned += 1
                year = parse_year(row.get('招聘发布年份'))
                if year is None:
                    continue
                toks = tokenize(row.get('cleaned_requirements', ''), stopwords)
                toks = apply_bigrams(toks, bigrams)
                if len(toks) < MIN_TOKENS:
                    continue

                year_seq[year] += 1
                jid = year * 10_000_000 + year_seq[year]
                comp = str(row.get('企业名称', '') or '').strip()
                title = str(row.get('招聘岗位', '') or '').strip()
                get_writer(year).writerow([jid, year, comp, title])
                kept += 1

                if scanned % 800_000 == 0:
                    print(f'    scanned={scanned:,}, kept={kept:,}')

    for fo in files.values():
        fo.close()

    # concat year files in ascending order to sorted output
    with RECON_COMPANY_JOB_SORTED.open('w', encoding='utf-8-sig', newline='') as g:
        w = None
        for yp in sorted(TMP_YEAR_DIR.glob('year_*.csv'), key=lambda p: int(p.stem.split('_')[1])):
            with yp.open('r', encoding='utf-8-sig', newline='') as f:
                rd = csv.DictReader(f)
                if w is None:
                    w = csv.DictWriter(g, fieldnames=rd.fieldnames)
                    w.writeheader()
                for row in rd:
                    w.writerow(row)

    print(f'  saved: {RECON_COMPANY_JOB_SORTED}')


def load_company_lookups() -> Tuple[Dict[str, Tuple[str, float]], Dict[str, Tuple[str, float]]]:
    print('[3/6] loading company lookup priors ...')
    company_std: Dict[str, Tuple[str, float]] = {}
    company_primary: Dict[str, Tuple[str, float]] = {}

    def upsert_best(d: Dict[str, Tuple[str, float]], key: str, code: str, conf: float) -> None:
        if not key:
            return
        old = d.get(key)
        if old is None or conf > old[1]:
            d[key] = (code, conf)

    if COMPANY_INDUSTRY_LOOKUP.exists():
        df = pd.read_csv(COMPANY_INDUSTRY_LOOKUP, low_memory=False)
        for _, r in df.iterrows():
            comp = str(r.get('企业名称', '') or '').strip()
            std = str(r.get('standard_industry', '') or '').strip()
            conf = float(r.get('confidence', 0) or 0)
            code = code_from_std_label(std)
            if comp:
                ckey = canonical_company_name(comp)
                # canonical first for robust matching; keep raw fallback key
                upsert_best(company_std, ckey, code, conf)
                upsert_best(company_std, comp, code, conf)

    if COMPANY_PRIMARY_LOOKUP.exists():
        df = pd.read_csv(COMPANY_PRIMARY_LOOKUP, low_memory=False)
        for _, r in df.iterrows():
            comp = str(r.get('企业名称', '') or '').strip()
            cate = str(r.get('初级分类', '') or '').strip()
            conf = float(r.get('confidence', 0) or 0)
            std = map_to_standard_industry(cate)
            code = code_from_std_label(std)
            if comp:
                ckey = canonical_company_name(comp)
                upsert_best(company_primary, ckey, code, conf)
                upsert_best(company_primary, comp, code, conf)

    print(f'  company_std size: {len(company_std):,}')
    print(f'  company_primary size: {len(company_primary):,}')
    return company_std, company_primary


def classify_industry20(company: str, job_title: str,
                        company_std: Dict[str, Tuple[str, float]],
                        company_primary: Dict[str, Tuple[str, float]]) -> Tuple[str, str, float, str]:
    comp = str(company or '')
    title = str(job_title or '')
    scores: Dict[str, float] = defaultdict(float)

    comp_key = canonical_company_name(comp)

    prior_std_code, prior_std_conf = company_std.get(comp_key, ('U', 0.0))
    if prior_std_code == 'U':
        prior_std_code, prior_std_conf = company_std.get(comp, ('U', 0.0))

    prior_primary_code, prior_primary_conf = company_primary.get(comp_key, ('U', 0.0))
    if prior_primary_code == 'U':
        prior_primary_code, prior_primary_conf = company_primary.get(comp, ('U', 0.0))

    if prior_std_code in INDUSTRY20 and prior_std_code != 'U':
        scores[prior_std_code] += 2.0 * max(0.1, prior_std_conf)

    if prior_primary_code in INDUSTRY20 and prior_primary_code != 'U':
        scores[prior_primary_code] += 1.2 * max(0.1, prior_primary_conf)

    comp_l = comp.lower()
    title_l = title.lower()

    for code, kws in KEYWORDS.items():
        hc = sum(1 for kw in kws if kw in comp_l)
        ht = sum(1 for kw in kws if kw in title_l)
        if hc > 0:
            scores[code] += min(1.0, 0.35 * hc)
        if ht > 0:
            scores[code] += min(1.0, 0.30 * ht)

    if not scores:
        return 'U', INDUSTRY20['U'], 0.0, 'unmapped'

    best_code = max(scores, key=scores.get)
    total = sum(scores.values())
    conf = float(scores[best_code] / total) if total > 0 else 0.0

    src = []
    if best_code == prior_std_code and prior_std_code != 'U':
        src.append('company_std_prior')
    if best_code == prior_primary_code and prior_primary_code != 'U':
        src.append('company_primary_prior')

    if any(kw in comp_l for kw in KEYWORDS.get(best_code, [])):
        src.append('company_name_keyword')
    if any(kw in title_l for kw in KEYWORDS.get(best_code, [])):
        src.append('job_title_keyword')

    if not src:
        src.append('score_fallback')

    return best_code, INDUSTRY20[best_code], conf, '|'.join(src)


def parse_survival_topic_sets() -> Dict[Tuple[str, str], Set[int]]:
    print('[4/6] parsing alignment survival topic sets ...')
    ev = pd.read_csv(EVENTS_CSV, low_memory=False)
    ev = ev[ev['event_type'].str.lower() == 'survival'].copy()

    def parse_topics(s: str) -> List[int]:
        if pd.isna(s):
            return []
        nums = re.findall(r'-?\d+', str(s))
        vals = [int(x) for x in nums if 0 <= int(x) <= 500]
        vals = [v for v in vals if v < 100]
        if not vals:
            return []
        if len(vals) >= 2 and vals[0] == 64:
            return [vals[-1]]
        out = []
        for v in vals:
            if v not in out:
                out.append(v)
        return out

    surv: Dict[Tuple[str, str], Set[int]] = defaultdict(set)
    for _, r in ev.iterrows():
        wf = str(r['window_from'])
        wt = str(r['window_to'])
        for t in parse_topics(r['source_topics']):
            surv[(wf, wt)].add(int(t))

    for k, v in surv.items():
        print(f'  {k[0]} -> {k[1]}: {len(v)} survival topics')
    return surv


def merge_and_prepare(company_std, company_primary, survival_sets) -> None:
    print('[5/6] merging + joint mapping + aggregations ...')
    entropy_stats: Dict[Tuple[int, str, str], List[float]] = defaultdict(lambda: [0, 0.0, 0.0])
    survival_stats: Dict[Tuple[str, str, str, str], List[int]] = defaultdict(lambda: [0, 0])
    source_counter: Dict[str, int] = defaultdict(int)

    with (
        FINAL_RESULTS_SORTED.open('r', encoding='utf-8-sig', newline='') as fr,
        RECON_COMPANY_JOB_SORTED.open('r', encoding='utf-8-sig', newline='') as rr,
        MASTER_JOINT.open('w', encoding='utf-8-sig', newline='') as out,
    ):
        f_reader = csv.DictReader(fr)
        r_reader = csv.DictReader(rr)

        fields = [
            'id', 'year', 'entropy_score', 'hhi_score', 'dominant_topic_id', 'dominant_topic_prob',
            'token_count',
            '企业名称', '招聘岗位',
            'industry20_code', 'industry20_label', 'mapping_confidence', 'mapping_source',
            'company_std_prior_code', 'company_primary_prior_code'
        ]
        w = csv.DictWriter(out, fieldnames=fields)
        w.writeheader()

        try:
            r_row = next(r_reader)
        except StopIteration:
            r_row = None

        matched = 0
        unmatched = 0

        for i, f_row in enumerate(f_reader, 1):
            fid = int(f_row['id'])
            while r_row is not None and int(r_row['id']) < fid:
                try:
                    r_row = next(r_reader)
                except StopIteration:
                    r_row = None
                    break

            if r_row is not None and int(r_row['id']) == fid:
                comp = str(r_row.get('企业名称', '') or '').strip()
                title = str(r_row.get('招聘岗位', '') or '').strip()
                matched += 1
            else:
                comp = ''
                title = ''
                unmatched += 1

            cstd = company_std.get(comp, ('U', 0.0))[0]
            cprim = company_primary.get(comp, ('U', 0.0))[0]
            code, label, conf, src = classify_industry20(comp, title, company_std, company_primary)
            source_counter[src] += 1

            row = {
                'id': f_row['id'],
                'year': f_row['year'],
                'entropy_score': f_row['entropy_score'],
                'hhi_score': f_row['hhi_score'],
                'dominant_topic_id': f_row['dominant_topic_id'],
                'dominant_topic_prob': f_row['dominant_topic_prob'],
                'token_count': f_row.get('token_count', ''),
                '企业名称': comp,
                '招聘岗位': title,
                'industry20_code': code,
                'industry20_label': label,
                'mapping_confidence': f'{conf:.4f}',
                'mapping_source': src,
                'company_std_prior_code': cstd,
                'company_primary_prior_code': cprim,
            }
            w.writerow(row)

            # entropy stats
            try:
                y = int(float(f_row['year']))
                ent = float(f_row['entropy_score'])
                hhi = float(f_row['hhi_score'])
                dt = int(float(f_row['dominant_topic_id']))
            except Exception:
                y = None

            if y is not None:
                k = (y, code, label)
                entropy_stats[k][0] += 1
                entropy_stats[k][1] += ent
                entropy_stats[k][2] += hhi

                pair = YEAR_TO_SOURCE_PAIR.get(y)
                if pair and pair in survival_sets:
                    sk = (pair[0], pair[1], code, label)
                    survival_stats[sk][0] += 1
                    if dt in survival_sets[pair]:
                        survival_stats[sk][1] += 1

            if i % 1_500_000 == 0:
                print(f'  merged={i:,}, matched={matched:,}, unmatched={unmatched:,}')

    # write yearly entropy table
    rows = []
    for (y, code, label), (n, se, sh) in entropy_stats.items():
        rows.append({
            'year': y,
            'industry20_code': code,
            'industry20_label': label,
            'sample_n': n,
            'mean_entropy': se / n,
            'mean_hhi': sh / n,
            'valid_sample': n >= VALID_UNIT_THRESHOLD,
        })
    pd.DataFrame(rows).sort_values(['year', 'sample_n'], ascending=[True, False]).to_csv(
        YEARLY_ENTROPY, index=False, encoding='utf-8-sig'
    )

    # write alignment survival table
    srows = []
    for (wf, wt, code, label), (nsrc, nsurv) in survival_stats.items():
        rate = nsurv / nsrc if nsrc > 0 else 0.0
        srows.append({
            'window_from': wf,
            'window_to': wt,
            'industry20_code': code,
            'industry20_label': label,
            'n_source': nsrc,
            'n_survive': nsurv,
            'survival_rate': rate,
            'reorganization_rate': 1 - rate,
            'valid_sample': nsrc >= VALID_UNIT_THRESHOLD,
        })
    pd.DataFrame(srows).sort_values(['window_from', 'n_source'], ascending=[True, False]).to_csv(
        ALIGN_SURVIVAL, index=False, encoding='utf-8-sig'
    )

    # diagnostics
    drows = [
        {'metric': 'master_rows', 'value': sum(v[0] for v in entropy_stats.values())},
        {'metric': 'matched_rows', 'value': int(pd.read_csv(MASTER_JOINT, usecols=['企业名称'], low_memory=False)['企业名称'].fillna('').astype(str).str.strip().ne('').sum())},
    ]
    for src, cnt in sorted(source_counter.items(), key=lambda x: x[1], reverse=True):
        drows.append({'metric': f'mapping_source::{src}', 'value': cnt})
    pd.DataFrame(drows).to_csv(DIAG, index=False, encoding='utf-8-sig')

    print(f'  saved: {MASTER_JOINT}')
    print(f'  saved: {YEARLY_ENTROPY}')
    print(f'  saved: {ALIGN_SURVIVAL}')
    print(f'  saved: {DIAG}')


def cleanup_tmp() -> None:
    if TMP_YEAR_DIR.exists():
        for p in TMP_YEAR_DIR.glob('year_*.csv'):
            p.unlink(missing_ok=True)
        TMP_YEAR_DIR.rmdir()


def main() -> None:
    HET_DIR.mkdir(parents=True, exist_ok=True)

    bigrams = load_bigrams()
    reconstruct_company_job_by_id(bigrams)
    company_std, company_primary = load_company_lookups()
    survival_sets = parse_survival_topic_sets()
    merge_and_prepare(company_std, company_primary, survival_sets)
    cleanup_tmp()
    print('[6/6] done.')


if __name__ == '__main__':
    main()
