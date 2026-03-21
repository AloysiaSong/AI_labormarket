#!/usr/bin/env python3
import argparse
import csv
import json
import os
import re
from typing import List, Optional, Tuple


CITY_LIST = [
    "北京", "上海", "广州", "深圳", "天津", "重庆", "杭州", "南京", "苏州", "成都", "武汉", "西安",
    "长沙", "郑州", "青岛", "宁波", "厦门", "福州", "济南", "合肥", "昆明", "无锡", "佛山", "东莞",
    "南昌", "沈阳", "大连", "哈尔滨", "长春", "石家庄", "太原", "南宁", "贵阳", "兰州", "海口",
    "呼和浩特", "乌鲁木齐", "银川", "西宁", "珠海", "中山", "惠州", "嘉兴", "绍兴", "金华", "台州",
    "温州", "徐州", "常州", "南通", "扬州", "镇江", "盐城", "洛阳", "泉州", "漳州", "烟台", "潍坊",
    "济宁", "淄博", "唐山", "保定", "廊坊", "邯郸", "秦皇岛", "丹东", "鞍山", "营口", "铁岭", "大庆",
    "柳州", "绵阳", "湖州", "义乌", "九江", "赣州", "咸阳", "襄阳",
]

COMPANY_HINT = re.compile(
    r"(公司|集团|科技|网络|信息|数据|软件|教育|医院|学校|银行|证券|保险|物流|贸易|咨询|管理|工作室|事务所|中心|研究院|传媒|有限|股份)"
)
JOB_HINT = re.compile(
    r"(工程师|开发|产品|运营|经理|专员|顾问|设计|销售|会计|行政|人事|教师|老师|讲师|编辑|客服|法务|总监|主管|助理|分析师|架构师|测试|前端|后端|Java|Python|C\\+\\+|Go|算法|数据)"
)
TITLE_SUFFIX_HINT = re.compile(
    r"(师|员|经理|主管|总监|助理|顾问|讲师|分析师|架构师|实习生|管培生|店长|主播|编辑|会计|法务|前台|文员)$"
)
NON_TITLE_CUE = re.compile(
    r"(负责|熟悉|能力|经验|待遇|福利|涨薪|自由度|团队|公司|沟通|学习|配合|完成|组织|参与|制定|执行|分析|提升|挖掘|协助|审查|落实|处理|安排|我们|你将|你需要|工作时间|岗位职责|任职要求|招聘要求)"
)
NON_JOB_PREFIX = (
    "职位描述", "岗位职责", "任职要求", "任职资格", "岗位要求", "岗位职责", "公司介绍", "福利待遇",
    "岗位责任", "职位要求", "招聘要求", "薪资待遇", "工作内容", "我们", "你将", "你需要", "简历", "联系方式",
    "企业介绍", "职责描述", "职位亮点", "岗位说明", "职位信息"
)
ENUM_VERB_CUE = re.compile(
    r"^(负责|参与|协助|编写|核查|跟踪|配合|完成|处理|审查|组织|制定|执行|提升|挖掘|分析|对接|保障|优化|维护|管理|安排)"
)


def clean_text(s: str) -> str:
    return re.sub(r"[ \t\r]+", " ", (s or "")).strip()


def clean_candidate(s: str) -> str:
    s = clean_text(s)
    s = re.sub(r"[，,。；;：:]+$", "", s).strip()
    s = re.sub(r"^(岗位|职位)\s*[:：]\s*", "", s).strip()
    return s


def is_likely_job_title(s: str) -> bool:
    s = clean_candidate(s)
    if not s:
        return False
    if len(s) < 2 or len(s) > 24:
        return False
    if any(s.startswith(p) for p in NON_JOB_PREFIX):
        return False
    if re.search(r"[。；;！!？?]", s):
        return False
    if "http" in s.lower() or "@" in s:
        return False
    if NON_TITLE_CUE.search(s):
        return False
    if JOB_HINT.search(s):
        return True
    if re.search(r"(实习生|储备干部|管培生|管理培训生|店员|普工|技工|司机|文员|前台)", s):
        return True
    if TITLE_SUFFIX_HINT.search(s):
        return True
    return False


def is_strong_enum_title(s: str) -> bool:
    s = clean_candidate(s)
    if not is_likely_job_title(s):
        return False
    if len(s) > 18:
        return False
    if ENUM_VERB_CUE.search(s):
        return False
    if re.search(r"[，,；;。:：]", s):
        return False
    return True


def split_title_outside_parentheses(s: str) -> List[str]:
    s = clean_candidate(s)
    if not s:
        return []
    parts = []
    buf = []
    depth = 0
    n = len(s)
    for i, ch in enumerate(s):
        if ch in "（(":
            depth += 1
            buf.append(ch)
            continue
        if ch in "）)" and depth > 0:
            depth -= 1
            buf.append(ch)
            continue
        if depth == 0 and ch == "/":
            left = s[i - 1] if i > 0 else ""
            right = s[i + 1] if i + 1 < n else ""
            if re.match(r"[A-Za-z0-9+#]", left) and re.match(r"[A-Za-z0-9+#]", right):
                buf.append(ch)
                continue
        if depth == 0 and ch in "、,，;；/|｜":
            t = clean_candidate("".join(buf))
            buf = []
            if t:
                parts.append(t)
        else:
            buf.append(ch)
    t = clean_candidate("".join(buf))
    if t:
        parts.append(t)
    out = []
    seen = set()
    for p in parts:
        if not is_likely_job_title(p):
            continue
        k = p.lower()
        if k in seen:
            continue
        seen.add(k)
        out.append(p)
    return out


def extract_company(text: str) -> str:
    patterns = [
        r"(?:公司名称|企业名称|单位名称|招聘单位|公司名)\s*[:：]\s*([^\n]{2,60})",
        r"(?:公司|企业)\s*[:：]\s*([^\n]{2,60})",
        r"公司介绍\s*\n\s*([^\n]{2,60})",
    ]
    for p in patterns:
        m = re.search(p, text, flags=re.IGNORECASE)
        if m:
            cand = clean_candidate(m.group(1))
            if COMPANY_HINT.search(cand) and len(cand) <= 40 and cand.count(" ") <= 3:
                return cand[:80]
    return ""


def extract_location_raw(text: str) -> str:
    patterns = [
        r"(?:工作地点|工作地址|上班地点|办公地点|工作地|工作城市|工作区域)\s*[:：]\s*([^\n]{2,80})",
        r"(?:地点)\s*[:：]\s*([^\n]{2,80})",
    ]
    for p in patterns:
        m = re.search(p, text, flags=re.IGNORECASE)
        if m:
            return clean_candidate(m.group(1))[:80]
    return ""


def parse_city_region(location_raw: str, text: str) -> Tuple[str, str]:
    src = f"{location_raw}\n{text}" if location_raw else text
    city = ""
    for c in CITY_LIST:
        if c in src:
            city = c
            break

    region = ""
    if location_raw:
        m = re.search(r"([^\s,，;；]{1,12}(?:区|县|市))", location_raw)
        if m:
            region = clean_candidate(m.group(1))
    if not region and city:
        m2 = re.search(re.escape(city) + r"([^\s,，;；]{1,12}(?:区|县|市))", src)
        if m2:
            region = clean_candidate(m2.group(1))
    return city, region


def extract_job_titles(text: str, row_job: str = "") -> List[str]:
    titles = []
    lines = [clean_candidate(x) for x in (text or "").splitlines()]
    lines = [x for x in lines if x]

    # 0) 先用原始岗位字段（如果有）
    row_job = clean_candidate(row_job or "")
    if row_job:
        parts = split_title_outside_parentheses(row_job)
        if parts:
            titles.extend(parts)
        elif is_likely_job_title(row_job):
            titles.append(row_job)

    # 识别“职位描述”区块，优先从区块前若干行抽标题，避免抓到福利/职责编号项
    desc_idx = -1
    for i, line in enumerate(lines):
        if line.startswith("职位描述"):
            desc_idx = i
            break

    # 1) 编号列表（仅在明显“多岗位标题列表”时采用）
    enum_source = lines
    if desc_idx >= 0:
        enum_source = lines[desc_idx + 1 : desc_idx + 10]
    enum_candidates = []
    for line in enum_source:
        m = re.match(r"^\s*\d{1,2}[\.、]\s*(.+)$", line)
        if m:
            cand = clean_candidate(m.group(1))
            if is_strong_enum_title(cand):
                enum_candidates.append(cand)
    if len(enum_candidates) >= 2:
        titles.extend(enum_candidates)

    # 2) 显式键值
    for p in [
        r"(?:招聘岗位|岗位名称|职位名称|招聘职位)\s*[:：]\s*([^\n]{2,40})",
    ]:
        for m in re.finditer(p, text, flags=re.IGNORECASE):
            cand = clean_candidate(m.group(1))
            if is_likely_job_title(cand):
                titles.append(cand)

    # 3) “职位描述”后的第一短行（仅在未识别到列表标题时兜底）
    if not titles:
        for i, line in enumerate(lines):
            if line.startswith("职位描述") and i + 1 < len(lines):
                cand = clean_candidate(lines[i + 1])
                cand = re.sub(r"^\d{1,2}[\.、]\s*", "", cand)
                if is_likely_job_title(cand):
                    titles.append(cand)
                break

    # 去重保持顺序
    dedup = []
    seen = set()
    for t in titles:
        key = t.lower()
        if key not in seen:
            seen.add(key)
            dedup.append(t)
    return dedup


def has_explicit_job_marker(text: str) -> bool:
    return bool(
        re.search(
            r"(招聘岗位|岗位名称|职位名称|招聘职位|岗位\s*[:：]|职位\s*[:：])",
            text or "",
            flags=re.IGNORECASE,
        )
    )


def main():
    parser = argparse.ArgumentParser(
        description="Split multi-job records and enrich fields from description."
    )
    parser.add_argument(
        "--input-csv",
        default="data/archive_recovery/unified/wayback_2019_allinone_schema.csv",
        help="Input CSV in all_in_one schema.",
    )
    parser.add_argument(
        "--output-csv",
        default="data/archive_recovery/unified/wayback_2019_allinone_schema_split_enriched.csv",
        help="Output CSV after split+enrich.",
    )
    parser.add_argument(
        "--summary-json",
        default="data/archive_recovery/unified/wayback_2019_allinone_schema_split_enriched_summary.json",
        help="Summary JSON path.",
    )
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
    os.makedirs(os.path.dirname(args.summary_json), exist_ok=True)

    in_rows = 0
    out_rows = 0
    split_rows = 0
    extracted_company = 0
    extracted_job = 0
    extracted_city = 0
    extracted_region = 0

    with open(args.input_csv, "r", encoding="utf-8-sig", newline="") as fin:
        reader = csv.DictReader(fin)
        fields = reader.fieldnames or []
        with open(args.output_csv, "w", encoding="utf-8-sig", newline="") as fout:
            writer = csv.DictWriter(fout, fieldnames=fields)
            writer.writeheader()

            for row in reader:
                in_rows += 1
                text = row.get("职位描述", "") or ""

                company = row.get("企业名称", "") or extract_company(text)
                location_raw = extract_location_raw(text)
                city, region = parse_city_region(location_raw, text)
                titles = extract_job_titles(text, row.get("招聘岗位", "") or "")

                base = dict(row)
                if company and not row.get("企业名称"):
                    base["企业名称"] = company
                    extracted_company += 1
                if location_raw and not row.get("工作地点"):
                    base["工作地点"] = location_raw
                if city and not row.get("工作城市"):
                    base["工作城市"] = city
                    extracted_city += 1
                if region and not row.get("工作区域"):
                    base["工作区域"] = region
                    extracted_region += 1
                if location_raw and not row.get("公司地点"):
                    base["公司地点"] = location_raw

                if titles:
                    if len(titles) > 1:
                        split_rows += 1
                        for t in titles:
                            out = dict(base)
                            out["招聘岗位"] = t
                            writer.writerow(out)
                            out_rows += 1
                            extracted_job += 1
                    elif len(titles) == 1 and (
                        has_explicit_job_marker(text) or not (row.get("招聘岗位") or "").strip()
                    ):
                        out = dict(base)
                        out["招聘岗位"] = titles[0]
                        writer.writerow(out)
                        out_rows += 1
                        extracted_job += 1
                    else:
                        writer.writerow(base)
                        out_rows += 1
                else:
                    writer.writerow(base)
                    out_rows += 1

    summary = {
        "input_csv": args.input_csv,
        "output_csv": args.output_csv,
        "rows_in": in_rows,
        "rows_out": out_rows,
        "rows_expanded_net": out_rows - in_rows,
        "rows_with_split": split_rows,
        "extracted_company_rows": extracted_company,
        "extracted_job_rows": extracted_job,
        "extracted_city_rows": extracted_city,
        "extracted_region_rows": extracted_region,
    }
    with open(args.summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(
        "done\n"
        f"output: {args.output_csv}\n"
        f"summary: {args.summary_json}\n"
        f"rows_in: {in_rows}\n"
        f"rows_out: {out_rows}"
    )


if __name__ == "__main__":
    main()
