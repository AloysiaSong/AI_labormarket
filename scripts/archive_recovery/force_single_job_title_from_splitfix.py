#!/usr/bin/env python3
import argparse
import csv
import json
import os
import re


TITLE_HINT = re.compile(
    r"(工程师|开发|产品|运营|经理|专员|顾问|设计|销售|会计|行政|人事|教师|老师|讲师|编辑|客服|法务|总监|主管|助理|分析师|架构师|测试|前端|后端|程序员|研究员|文员|店长|导购|经纪人|普工|操作工|出纳|收银|项目经理)"
)
VERB_CUE = re.compile(r"(负责|参与|执行|协助|制定|完成|处理|安排|能力|经验|福利|待遇|岗位职责|任职要求)")
MODIFIER = {"高级", "中级", "初级", "资深", "副", "助理"}


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
    if VERB_CUE.search(s):
        return False
    if TITLE_HINT.search(s):
        return True
    if re.search(r"(实习生|管培生|储备干部|营业员|店员)$", s):
        return True
    return False


def split_title_outside_parentheses(s: str):
    s = clean(s)
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

        if depth == 0 and ch in "/":
            # keep slash for token like C/C++ or UI/UX
            left = s[i - 1] if i > 0 else ""
            right = s[i + 1] if i + 1 < n else ""
            if re.match(r"[A-Za-z0-9+#]", left) and re.match(r"[A-Za-z0-9+#]", right):
                buf.append(ch)
                continue

        if depth == 0 and ch in "、,，;；/|｜":
            t = clean("".join(buf))
            buf = []
            if t:
                parts.append(t)
        else:
            buf.append(ch)

    t = clean("".join(buf))
    if t:
        parts.append(t)
    return parts


def extract_titles_from_desc(desc: str):
    lines = [clean(x) for x in (desc or "").splitlines()]
    lines = [x for x in lines if x]
    if not lines:
        return []
    start = 0
    for i, ln in enumerate(lines[:12]):
        if ln.startswith("职位描述"):
            start = i + 1
            break
    window = lines[start : start + 12]
    out = []
    for ln in window:
        m = re.match(r"^\d{1,2}[\\.、]\\s*(.+)$", ln)
        if not m:
            continue
        t = clean(m.group(1))
        if is_title_like(t):
            out.append(t)
    dedup = []
    seen = set()
    for x in out:
        k = x.lower()
        if k in seen:
            continue
        seen.add(k)
        dedup.append(x)
    return dedup


def row_key(row: dict, fields):
    return tuple((row.get(k) or "") for k in fields)


def main():
    parser = argparse.ArgumentParser(description="Force single job title per row by splitting residual multi-title rows.")
    parser.add_argument(
        "--input-csv",
        default="data/archive_recovery/unified/wayback_2019_allinone_schema_deepseek_enriched_full_splitfix.csv",
    )
    parser.add_argument(
        "--output-csv",
        default="data/archive_recovery/unified/wayback_2019_allinone_schema_deepseek_enriched_full_splitfix_v2.csv",
    )
    parser.add_argument(
        "--summary-json",
        default="data/archive_recovery/unified/wayback_2019_allinone_schema_deepseek_enriched_full_splitfix_v2_summary.json",
    )
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
    os.makedirs(os.path.dirname(args.summary_json), exist_ok=True)

    rows_in = 0
    rows_out = 0
    split_rows = 0
    normalized_single = 0
    by_desc_split = 0
    dup_drop = 0

    with open(args.input_csv, "r", encoding="utf-8-sig", newline="") as fin:
        reader = csv.DictReader(fin)
        fields = reader.fieldnames or []
        seen = set()
        with open(args.output_csv, "w", encoding="utf-8-sig", newline="") as fout:
            writer = csv.DictWriter(fout, fieldnames=fields)
            writer.writeheader()

            for row in reader:
                rows_in += 1
                jt = clean(row.get("招聘岗位", "") or "")
                desc = row.get("职位描述", "") or ""

                titles = []
                if jt:
                    raw_parts = split_title_outside_parentheses(jt)
                    title_parts = [p for p in raw_parts if is_title_like(p)]
                    if len(title_parts) >= 2:
                        titles = title_parts
                        split_rows += 1
                    elif len(title_parts) == 1 and len(raw_parts) >= 2:
                        # keep one clean title if others are non-title noise/cities
                        t = title_parts[0]
                        # prefix modifier like "高级/人力行政主管"
                        if len(raw_parts) == 2 and raw_parts[0] in MODIFIER:
                            t = clean(raw_parts[0] + raw_parts[1])
                        out = dict(row)
                        out["招聘岗位"] = t
                        k = row_key(out, fields)
                        if k in seen:
                            dup_drop += 1
                        else:
                            seen.add(k)
                            writer.writerow(out)
                            rows_out += 1
                            normalized_single += 1
                        continue
                else:
                    desc_titles = extract_titles_from_desc(desc)
                    if len(desc_titles) >= 2:
                        titles = desc_titles
                        by_desc_split += 1
                        split_rows += 1

                if titles:
                    dedup = []
                    sset = set()
                    for t in titles:
                        t = clean(t)
                        if not t:
                            continue
                        k2 = t.lower()
                        if k2 in sset:
                            continue
                        sset.add(k2)
                        dedup.append(t)
                    for t in dedup:
                        out = dict(row)
                        out["招聘岗位"] = t
                        k = row_key(out, fields)
                        if k in seen:
                            dup_drop += 1
                            continue
                        seen.add(k)
                        writer.writerow(out)
                        rows_out += 1
                else:
                    k = row_key(row, fields)
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
        "by_desc_split": by_desc_split,
        "normalized_single": normalized_single,
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
