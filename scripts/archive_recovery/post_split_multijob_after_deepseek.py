#!/usr/bin/env python3
import argparse
import csv
import json
import os
import re


TITLE_HINT = re.compile(
    r"(工程师|开发|产品|运营|经理|专员|顾问|设计|销售|会计|行政|人事|教师|老师|讲师|编辑|客服|法务|总监|主管|助理|分析师|架构师|测试|前端|后端|程序员|研究员|文员|店长|导购|经纪人|总裁|CEO|普工|电焊工|钳工|录入员)"
)
BAD_CUE = re.compile(
    r"(负责|参与|协助|执行|制定|完成|处理|安排|能力|经验|福利|待遇|薪资|沟通|学习|岗位职责|任职要求|招聘要求|工作内容)"
)


def clean(s: str) -> str:
    s = (s or "").strip()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"^[：:·•\\-\\s]+", "", s)
    s = re.sub(r"[，,。；;：:]+$", "", s)
    return s.strip()


def is_title_like(s: str) -> bool:
    s = clean(s)
    if not s:
        return False
    if len(s) < 2 or len(s) > 30:
        return False
    if re.search(r"[。；;!?！？]", s):
        return False
    if BAD_CUE.search(s):
        return False
    if s.startswith(("岗位职责", "任职要求", "职位描述", "公司介绍", "福利")):
        return False
    if TITLE_HINT.search(s):
        return True
    if re.search(r"(实习生|管培生|储备干部|营业员|店员)$", s):
        return True
    return False


def dedup_keep_order(items):
    out, seen = [], set()
    for x in items:
        k = x.lower()
        if k in seen:
            continue
        seen.add(k)
        out.append(x)
    return out


def split_job_field(job: str):
    job = clean(job)
    if not job:
        return []

    protected = {
        "C/C++": "C_CPLUSPLUS",
        "C/C#": "C_CSHARP",
        "UI/UX": "UI_UX",
        "B/S": "B_S",
        "C/S": "C_S",
    }
    tmp = job
    for k, v in protected.items():
        tmp = tmp.replace(k, v)

    parts = [clean(x) for x in re.split(r"[、,，;；/|｜]", tmp) if clean(x)]
    restored = []
    for p in parts:
        for k, v in protected.items():
            p = p.replace(v, k)
        restored.append(clean(p))

    restored = dedup_keep_order([x for x in restored if is_title_like(x)])
    if len(restored) >= 2:
        return restored
    return []


def extract_from_desc(desc: str):
    if not desc:
        return []
    lines = [clean(x) for x in desc.splitlines()]
    lines = [x for x in lines if x]
    if not lines:
        return []

    # 聚焦描述开头区域
    start = 0
    for i, ln in enumerate(lines[:12]):
        if ln.startswith("职位描述"):
            start = i + 1
            break
    window = lines[start : start + 12]

    cands = []
    for ln in window:
        m = re.match(r"^\d{1,2}[\\.、]\\s*(.+)$", ln)
        if not m:
            continue
        t = clean(m.group(1))
        if is_title_like(t):
            cands.append(t)

    cands = dedup_keep_order(cands)
    if len(cands) >= 2:
        return cands
    return []


def row_key(row: dict, fieldnames):
    return tuple((row.get(k) or "") for k in fieldnames)


def main():
    parser = argparse.ArgumentParser(description="Post-split multi-job rows after deepseek enrichment.")
    parser.add_argument(
        "--input-csv",
        default="data/archive_recovery/unified/wayback_2019_allinone_schema_deepseek_enriched_full.csv",
    )
    parser.add_argument(
        "--output-csv",
        default="data/archive_recovery/unified/wayback_2019_allinone_schema_deepseek_enriched_full_splitfix.csv",
    )
    parser.add_argument(
        "--summary-json",
        default="data/archive_recovery/unified/wayback_2019_allinone_schema_deepseek_enriched_full_splitfix_summary.json",
    )
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
    os.makedirs(os.path.dirname(args.summary_json), exist_ok=True)

    rows_in = 0
    rows_out = 0
    split_by_job_field = 0
    split_by_desc = 0
    split_rows = 0
    dup_drop = 0

    with open(args.input_csv, "r", encoding="utf-8-sig", newline="") as fin:
        reader = csv.DictReader(fin)
        fieldnames = reader.fieldnames or []

        seen = set()
        with open(args.output_csv, "w", encoding="utf-8-sig", newline="") as fout:
            writer = csv.DictWriter(fout, fieldnames=fieldnames)
            writer.writeheader()

            for row in reader:
                rows_in += 1
                job = row.get("招聘岗位", "") or ""
                desc = row.get("职位描述", "") or ""

                titles = split_job_field(job)
                source = ""
                if titles:
                    source = "job_field"
                    split_by_job_field += 1
                else:
                    if not job.strip():
                        titles = extract_from_desc(desc)
                        if titles:
                            source = "desc"
                            split_by_desc += 1

                if len(titles) >= 2:
                    split_rows += 1
                    for t in titles:
                        out = dict(row)
                        out["招聘岗位"] = t
                        k = row_key(out, fieldnames)
                        if k in seen:
                            dup_drop += 1
                            continue
                        seen.add(k)
                        writer.writerow(out)
                        rows_out += 1
                else:
                    k = row_key(row, fieldnames)
                    if k in seen:
                        dup_drop += 1
                        continue
                    seen.add(k)
                    writer.writerow(row)
                    rows_out += 1

    summary = {
        "input_csv": args.input_csv,
        "output_csv": args.output_csv,
        "rows_in": rows_in,
        "rows_out": rows_out,
        "rows_expanded_net": rows_out - rows_in,
        "split_rows": split_rows,
        "split_by_job_field": split_by_job_field,
        "split_by_desc": split_by_desc,
        "dup_drop": dup_drop,
    }
    with open(args.summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(
        "done\n"
        f"output: {args.output_csv}\n"
        f"summary: {args.summary_json}\n"
        f"rows_in: {rows_in}\n"
        f"rows_out: {rows_out}"
    )


if __name__ == "__main__":
    main()
