#!/usr/bin/env python3
import argparse
import csv
import json
import os
import re
import time
import urllib.error
import urllib.request


TARGET_KEYS = ["企业名称", "招聘岗位", "工作城市", "工作区域", "工作地点"]
TITLE_HINT = re.compile(
    r"(工程师|开发|产品|运营|经理|专员|顾问|设计|销售|会计|行政|人事|教师|老师|讲师|编辑|客服|法务|总监|主管|助理|分析师|架构师|测试|前端|后端|程序员|研究员|文员|店长|导购|经纪人|普工|操作工|出纳|收银|代表|主任)"
)
BAD_CUE = re.compile(
    r"(负责|参与|协助|执行|制定|完成|处理|安排|能力|经验|福利|待遇|薪资|岗位职责|任职要求|招聘要求|工作内容)"
)


def clean_val(v: str, max_len: int = 160) -> str:
    v = (v or "").strip()
    v = re.sub(r"\s+", " ", v).strip()
    if len(v) > max_len:
        v = v[:max_len]
    return v


def extract_json_array(text: str):
    text = (text or "").strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    s = text.find("[")
    e = text.rfind("]")
    if s == -1 or e == -1 or e <= s:
        return []
    payload = text[s : e + 1]
    try:
        obj = json.loads(payload)
    except json.JSONDecodeError:
        return []
    if not isinstance(obj, list):
        return []
    return [x for x in obj if isinstance(x, dict)]


def normalize_items(items):
    out = []
    seen = set()
    for x in items:
        company = clean_val(str(x.get("company_name", "") or x.get("企业名称", "")), 120)
        job = clean_val(
            str(
                x.get("job_title", "")
                or x.get("岗位", "")
                or x.get("招聘岗位", "")
                or x.get("title", "")
            ),
            80,
        )
        city = clean_val(str(x.get("city", "") or x.get("工作城市", "")), 40)
        district = clean_val(str(x.get("district", "") or x.get("region", "") or x.get("工作区域", "")), 40)
        location = clean_val(str(x.get("work_location", "") or x.get("工作地点", "") or x.get("location", "")), 160)

        if not any([company, job, city, district, location]):
            continue
        k = (company.lower(), job.lower(), city.lower(), district.lower(), location.lower())
        if k in seen:
            continue
        seen.add(k)
        out.append(
            {
                "企业名称": company,
                "招聘岗位": job,
                "工作城市": city,
                "工作区域": district,
                "工作地点": location,
            }
        )
    return out


def clean_title(s: str) -> str:
    s = clean_val(s, 80)
    s = re.sub(r"^(岗位|职位)\s*[:：]\s*", "", s)
    s = re.sub(r"[。；;！!？?]+$", "", s)
    return s.strip()


def is_title_like(s: str) -> bool:
    s = clean_title(s)
    if not s:
        return False
    if len(s) < 2 or len(s) > 30:
        return False
    if BAD_CUE.search(s):
        return False
    if re.search(r"[。；;！!？?]", s):
        return False
    if TITLE_HINT.search(s):
        return True
    if re.search(r"(实习生|管培生|储备干部|营业员|店员)$", s):
        return True
    return False


def split_title_outside_parentheses(s: str):
    s = clean_title(s)
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
            t = clean_title("".join(buf))
            buf = []
            if t:
                parts.append(t)
        else:
            buf.append(ch)
    t = clean_title("".join(buf))
    if t:
        parts.append(t)
    return [x for x in parts if is_title_like(x)]


def dedup_titles(titles):
    out = []
    seen = set()
    for t in titles:
        t = clean_title(t)
        if not t:
            continue
        k = t.lower()
        if k in seen:
            continue
        seen.add(k)
        out.append(t)
    return out


def infer_titles_from_row(row: dict):
    titles = []
    job_field = clean_title(row.get("招聘岗位", "") or "")
    if job_field:
        titles.extend(split_title_outside_parentheses(job_field) or [job_field] if is_title_like(job_field) else [])

    text = row.get("职位描述", "") or ""
    # 显式键值
    for p in [
        r"(?:招聘岗位|岗位名称|职位名称|招聘职位|岗位|职位)\s*[:：]\s*([^\n]{2,60})",
    ]:
        for m in re.finditer(p, text, flags=re.IGNORECASE):
            cand = clean_title(m.group(1))
            titles.extend(split_title_outside_parentheses(cand) or ([cand] if is_title_like(cand) else []))

    lines = [clean_title(x) for x in text.splitlines()]
    lines = [x for x in lines if x]
    if lines:
        start = 0
        for i, ln in enumerate(lines[:15]):
            if ln.startswith("职位描述"):
                start = i + 1
                break
        window = lines[start : start + 12]

        # 编号列表
        enum_titles = []
        for ln in window:
            m = re.match(r"^\d{1,2}[\\.、]\s*(.+)$", ln)
            if not m:
                continue
            cand = clean_title(m.group(1))
            if is_title_like(cand):
                enum_titles.extend(split_title_outside_parentheses(cand) or [cand])
        if len(enum_titles) >= 2:
            titles.extend(enum_titles)

        # 兜底：首个短标题行
        if not titles:
            for ln in window[:4]:
                if is_title_like(ln):
                    titles.extend(split_title_outside_parentheses(ln) or [ln])
                    break

    return dedup_titles(titles)


def enrich_items_with_inferred_titles(row: dict, items):
    inferred = infer_titles_from_row(row)
    if not items:
        if not inferred:
            return []
        return [
            {
                "企业名称": "",
                "招聘岗位": t,
                "工作城市": "",
                "工作区域": "",
                "工作地点": "",
            }
            for t in inferred
        ]

    # 如果仅1条且无岗位，但推断出了多岗位，则展开
    if len(items) == 1 and not (items[0].get("招聘岗位") or "").strip() and len(inferred) >= 2:
        base = dict(items[0])
        out = []
        for t in inferred:
            x = dict(base)
            x["招聘岗位"] = t
            out.append(x)
        items = out
    else:
        for i, it in enumerate(items):
            if not (it.get("招聘岗位") or "").strip():
                if len(inferred) == 1:
                    it["招聘岗位"] = inferred[0]
                elif len(inferred) > 1 and i < len(inferred):
                    it["招聘岗位"] = inferred[i]

    # 再次按岗位字段拆分（处理 API 返回中的多岗位串）
    expanded = []
    for it in items:
        jt = clean_title(it.get("招聘岗位", "") or "")
        parts = split_title_outside_parentheses(jt)
        if len(parts) >= 2:
            for p in parts:
                x = dict(it)
                x["招聘岗位"] = p
                expanded.append(x)
        else:
            if jt:
                it["招聘岗位"] = jt
            expanded.append(it)

    # 去重
    dedup = []
    seen = set()
    for it in expanded:
        key = (
            (it.get("企业名称") or "").strip().lower(),
            (it.get("招聘岗位") or "").strip().lower(),
            (it.get("工作城市") or "").strip().lower(),
            (it.get("工作区域") or "").strip().lower(),
            (it.get("工作地点") or "").strip().lower(),
        )
        if key in seen:
            continue
        seen.add(key)
        dedup.append(it)
    return dedup


def call_deepseek(api_key, api_base, model, row, timeout):
    system_prompt = (
        "你是招聘信息结构化抽取器。"
        "目标：对单条职位文本做高精度拆分。"
        "若文本明确包含多个不同岗位（例如 1.岗位A 2.岗位B，或 岗位A/岗位B），返回多条。"
        "若只是同义称呼/级别（如 研究员/分析员，初级/高级）或岗位职责条目，不要拆成多个岗位。"
        "禁止把职责句、技能句、福利句当岗位名。"
        "输出必须是 JSON 数组，元素字段固定为：company_name, job_title, city, district, work_location。"
        "未知填空字符串。不要输出解释。"
    )

    payload = {
        "task": "逐条抽取并必要时拆分岗位",
        "platform_hint": row.get("来源平台", ""),
        "source_url": row.get("来源", ""),
        "publish_date": row.get("招聘发布日期", ""),
        "existing_fields": {
            "企业名称": row.get("企业名称", ""),
            "招聘岗位": row.get("招聘岗位", ""),
            "工作城市": row.get("工作城市", ""),
            "工作区域": row.get("工作区域", ""),
            "工作地点": row.get("工作地点", ""),
        },
        "text": (row.get("职位描述", "") or "")[:10000],
    }
    body = {
        "model": model,
        "temperature": 0,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ],
    }
    req = urllib.request.Request(
        api_base.rstrip("/") + "/chat/completions",
        data=json.dumps(body).encode("utf-8"),
        method="POST",
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        result = json.loads(resp.read().decode("utf-8", errors="ignore"))
    content = result["choices"][0]["message"]["content"]
    return normalize_items(extract_json_array(content))


def load_done_rows(cache_path: str):
    done = set()
    if not os.path.exists(cache_path):
        return done
    with open(cache_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if bool(obj.get("ok", False)):
                    done.add(int(obj["row_id"]))
            except Exception:
                continue
    return done


def append_cache(cache_path: str, row_id: int, items, ok: bool):
    with open(cache_path, "a", encoding="utf-8") as f:
        f.write(
            json.dumps(
                {"row_id": row_id, "ok": ok, "items": items},
                ensure_ascii=False,
            )
            + "\n"
        )


def write_summary(summary_json, summary_obj):
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summary_obj, f, ensure_ascii=False, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Strict row-by-row DeepSeek enrichment and split.")
    parser.add_argument(
        "--input-csv",
        default="data/archive_recovery/unified/wayback_2019_allinone_schema.csv",
    )
    parser.add_argument(
        "--output-csv",
        default="data/archive_recovery/unified/wayback_2019_allinone_schema_deepseek_rowwise_full.csv",
    )
    parser.add_argument(
        "--summary-json",
        default="data/archive_recovery/unified/wayback_2019_allinone_schema_deepseek_rowwise_full_summary.json",
    )
    parser.add_argument(
        "--cache-jsonl",
        default="data/archive_recovery/unified/wayback_2019_deepseek_rowwise_cache.jsonl",
    )
    parser.add_argument("--api-base", default="https://api.deepseek.com")
    parser.add_argument("--model", default="deepseek-chat")
    parser.add_argument("--api-key-env", default="DEEPSEEK_API_KEY")
    parser.add_argument("--request-interval", type=float, default=0.8)
    parser.add_argument("--max-attempts", type=int, default=4)
    parser.add_argument("--retry-wait", type=float, default=1.2)
    parser.add_argument("--timeout", type=float, default=50.0)
    parser.add_argument("--max-rows", type=int, default=0)
    parser.add_argument("--progress-every", type=int, default=500)
    parser.add_argument("--start-row", type=int, default=1, help="1-based")
    args = parser.parse_args()

    api_key = os.environ.get(args.api_key_env, "").strip()
    if not api_key:
        raise SystemExit(f"Missing API key env: {args.api_key_env}")

    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
    os.makedirs(os.path.dirname(args.summary_json), exist_ok=True)
    os.makedirs(os.path.dirname(args.cache_jsonl), exist_ok=True)

    rows = []
    with open(args.input_csv, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames or []
        for i, row in enumerate(reader, start=1):
            if i < args.start_row:
                continue
            rows.append((i, row))
            if args.max_rows and len(rows) >= args.max_rows:
                break

    done = load_done_rows(args.cache_jsonl)
    cache_map = {}
    if os.path.exists(args.cache_jsonl):
        with open(args.cache_jsonl, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    cache_map[int(obj["row_id"])] = obj
                except Exception:
                    continue

    total = len(rows)
    api_new = 0
    api_ok = 0
    api_fail = 0
    for idx, (row_id, row) in enumerate(rows, start=1):
        if row_id in done:
            continue

        ok = False
        items = []
        for attempt in range(1, max(1, args.max_attempts) + 1):
            try:
                items = call_deepseek(
                    api_key=api_key,
                    api_base=args.api_base,
                    model=args.model,
                    row=row,
                    timeout=args.timeout,
                )
                ok = True
                break
            except Exception:
                if attempt < max(1, args.max_attempts):
                    time.sleep(args.retry_wait)
        append_cache(args.cache_jsonl, row_id, items, ok)
        cache_map[row_id] = {"row_id": row_id, "ok": ok, "items": items}
        done.add(row_id)
        api_new += 1
        if ok:
            api_ok += 1
        else:
            api_fail += 1

        if args.progress_every > 0 and api_new % args.progress_every == 0:
            print(f"progress rowwise_new_calls: {api_new} ok={api_ok} fail={api_fail} total_target={total}", flush=True)
        if args.request_interval > 0:
            time.sleep(args.request_interval)

    out_rows = 0
    expanded_net = 0
    with open(args.output_csv, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row_id, row in rows:
            obj = cache_map.get(row_id, {"ok": False, "items": []})
            items = obj.get("items") or []
            items = enrich_items_with_inferred_titles(row, items)
            if items:
                wrote = 0
                for it in items:
                    out = dict(row)
                    for k in TARGET_KEYS:
                        if not (out.get(k) or "").strip():
                            out[k] = it.get(k, "")
                    if (out.get("工作地点") or "").strip() and not (out.get("公司地点") or "").strip():
                        out["公司地点"] = out["工作地点"]
                    writer.writerow(out)
                    out_rows += 1
                    wrote += 1
                if wrote > 1:
                    expanded_net += (wrote - 1)
            else:
                writer.writerow(row)
                out_rows += 1

    summary = {
        "input_csv": args.input_csv,
        "output_csv": args.output_csv,
        "cache_jsonl": args.cache_jsonl,
        "rows_target": total,
        "rows_out": out_rows,
        "rows_expanded_net": expanded_net,
        "api_new_calls": api_new,
        "api_ok": api_ok,
        "api_fail": api_fail,
        "params": {
            "model": args.model,
            "request_interval": args.request_interval,
            "max_attempts": args.max_attempts,
            "retry_wait": args.retry_wait,
            "timeout": args.timeout,
            "start_row": args.start_row,
            "max_rows": args.max_rows,
        },
    }
    write_summary(args.summary_json, summary)
    print(
        "done\n"
        f"output: {args.output_csv}\n"
        f"summary: {args.summary_json}\n"
        f"rows_target: {total}\n"
        f"rows_out: {out_rows}\n"
        f"api_new_calls: {api_new} ok={api_ok} fail={api_fail}"
    )


if __name__ == "__main__":
    main()
