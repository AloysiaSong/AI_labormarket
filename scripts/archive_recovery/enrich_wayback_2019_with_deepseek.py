#!/usr/bin/env python3
import argparse
import concurrent.futures
import csv
import hashlib
import json
import os
import re
import time
import urllib.error
import urllib.request
from typing import Dict, List


TARGET_KEYS = ["企业名称", "招聘岗位", "工作城市", "工作区域", "工作地点"]


def sha1_text(s: str) -> str:
    return hashlib.sha1((s or "").encode("utf-8", errors="ignore")).hexdigest()


def clean_val(v: str, max_len: int = 120) -> str:
    v = (v or "").strip()
    v = re.sub(r"\s+", " ", v).strip()
    if len(v) > max_len:
        v = v[:max_len]
    return v


def load_cache(path: str) -> Dict[str, List[dict]]:
    cache = {}
    if not os.path.exists(path):
        return cache
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            cache[obj["text_hash"]] = obj["items"]
    return cache


def append_cache(path: str, text_hash: str, items: List[dict]) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps({"text_hash": text_hash, "items": items}, ensure_ascii=False) + "\n")


def extract_json_array(text: str) -> List[dict]:
    text = (text or "").strip()
    # strip markdown fences if present
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    start = text.find("[")
    end = text.rfind("]")
    if start == -1 or end == -1 or end <= start:
        return []
    payload = text[start : end + 1]
    try:
        obj = json.loads(payload)
        if isinstance(obj, list):
            out = []
            for x in obj:
                if isinstance(x, dict):
                    out.append(x)
            return out
    except json.JSONDecodeError:
        return []
    return []


def normalize_items(items: List[dict]) -> List[dict]:
    norm = []
    seen = set()
    for x in items:
        company = clean_val(str(x.get("company_name", "") or x.get("company", "") or x.get("企业名称", "")), 80)
        job = clean_val(str(x.get("job_title", "") or x.get("岗位", "") or x.get("招聘岗位", "")), 80)
        city = clean_val(str(x.get("city", "") or x.get("工作城市", "")), 40)
        district = clean_val(str(x.get("district", "") or x.get("region", "") or x.get("工作区域", "")), 40)
        work_loc = clean_val(str(x.get("work_location", "") or x.get("location", "") or x.get("工作地点", "")), 120)

        if not any([company, job, city, district, work_loc]):
            continue
        key = (company.lower(), job.lower(), city.lower(), district.lower(), work_loc.lower())
        if key in seen:
            continue
        seen.add(key)
        norm.append(
            {
                "企业名称": company,
                "招聘岗位": job,
                "工作城市": city,
                "工作区域": district,
                "工作地点": work_loc,
            }
        )
    return norm


def call_deepseek(
    api_key: str,
    api_base: str,
    model: str,
    text: str,
    platform: str,
    source_url: str,
    timeout: float,
) -> List[dict]:
    system_prompt = (
        "You are an information extraction engine. "
        "Extract job posting fields from Chinese recruitment page text. "
        "Do not fabricate."
    )
    user_prompt = {
        "task": (
            "Return a JSON array. Each element is one job posting with keys: "
            "company_name, job_title, city, district, work_location. "
            "If a field is unknown, use empty string. "
            "If no job posting can be identified, return []. "
            "No explanation, JSON array only."
        ),
        "platform_hint": platform,
        "source_url": source_url,
        "text": text[:8000],
    }
    body = {
        "model": model,
        "temperature": 0,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": json.dumps(user_prompt, ensure_ascii=False)},
        ],
    }
    data = json.dumps(body).encode("utf-8")
    url = api_base.rstrip("/") + "/chat/completions"
    req = urllib.request.Request(
        url,
        data=data,
        method="POST",
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        payload = json.loads(resp.read().decode("utf-8", errors="ignore"))
    content = payload["choices"][0]["message"]["content"]
    return normalize_items(extract_json_array(content))


def main():
    parser = argparse.ArgumentParser(
        description="Enrich/split wayback 2019 all_in_one schema via DeepSeek."
    )
    parser.add_argument(
        "--input-csv",
        default="data/archive_recovery/unified/wayback_2019_allinone_schema.csv",
    )
    parser.add_argument(
        "--output-csv",
        default="data/archive_recovery/unified/wayback_2019_allinone_schema_deepseek_enriched.csv",
    )
    parser.add_argument(
        "--summary-json",
        default="data/archive_recovery/unified/wayback_2019_allinone_schema_deepseek_enriched_summary.json",
    )
    parser.add_argument(
        "--cache-jsonl",
        default="data/archive_recovery/unified/wayback_2019_deepseek_cache.jsonl",
    )
    parser.add_argument("--api-base", default="https://api.deepseek.com")
    parser.add_argument("--model", default="deepseek-chat")
    parser.add_argument("--api-key-env", default="DEEPSEEK_API_KEY")
    parser.add_argument("--request-interval", type=float, default=0.25)
    parser.add_argument("--timeout", type=float, default=45.0)
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--retry-wait", type=float, default=1.0)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--max-new-calls", type=int, default=0, help="0 means no limit.")
    parser.add_argument("--min-text-len", type=int, default=120)
    parser.add_argument("--max-rows", type=int, default=0, help="0 means all rows.")
    args = parser.parse_args()

    api_key = os.environ.get(args.api_key_env, "").strip()
    if not api_key:
        raise SystemExit(f"Missing API key in env: {args.api_key_env}")

    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
    os.makedirs(os.path.dirname(args.summary_json), exist_ok=True)
    os.makedirs(os.path.dirname(args.cache_jsonl), exist_ok=True)

    rows = []
    with open(args.input_csv, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames or []
        for i, row in enumerate(reader, start=1):
            if args.max_rows and i > args.max_rows:
                break
            rows.append(row)

    cache = load_cache(args.cache_jsonl)
    text_map = {}
    text_meta = {}
    for row in rows:
        text = (row.get("职位描述") or "").strip()
        if len(text) < args.min_text_len:
            continue
        h = sha1_text(text)
        text_map[h] = text
        if h not in text_meta:
            text_meta[h] = {
                "平台": row.get("来源平台", ""),
                "来源": row.get("来源", ""),
            }

    pending = [h for h in text_map.keys() if h not in cache]
    if args.max_new_calls > 0:
        pending = pending[: args.max_new_calls]

    api_ok = 0
    api_fail = 0
    failed_hashes = []

    def process_one(h: str):
        meta = text_meta[h]
        last_err = ""
        for attempt in range(1, max(1, args.max_attempts) + 1):
            try:
                items = call_deepseek(
                    api_key=api_key,
                    api_base=args.api_base,
                    model=args.model,
                    text=text_map[h],
                    platform=meta["平台"],
                    source_url=meta["来源"],
                    timeout=args.timeout,
                )
                return h, True, items, ""
            except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, KeyError, json.JSONDecodeError) as e:
                last_err = str(e)
                if attempt < max(1, args.max_attempts):
                    time.sleep(args.retry_wait)
                continue

        return h, False, [], last_err

    completed = 0
    workers = max(1, args.workers)
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
        fut_map = {ex.submit(process_one, h): h for h in pending}
        for fut in concurrent.futures.as_completed(fut_map):
            h, ok, items, _err = fut.result()
            completed += 1
            if ok:
                cache[h] = items
                append_cache(args.cache_jsonl, h, items)
                api_ok += 1
            else:
                api_fail += 1
                failed_hashes.append(h)

            if completed % 20 == 0:
                print(f"progress calls: {completed}/{len(pending)} ok={api_ok} fail={api_fail}", flush=True)

            if args.request_interval > 0:
                time.sleep(args.request_interval)

    if pending and completed < len(pending):
        # defensive, should not happen
        for h in pending[completed:]:
            api_fail += 1
            failed_hashes.append(h)

    out_rows = 0
    expanded_rows = 0
    with open(args.output_csv, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            text = (row.get("职位描述") or "").strip()
            if len(text) < args.min_text_len:
                writer.writerow(row)
                out_rows += 1
                continue
            h = sha1_text(text)
            items = cache.get(h, [])
            if items:
                wrote = 0
                for item in items:
                    out = dict(row)
                    for k in TARGET_KEYS:
                        if not (out.get(k) or "").strip():
                            out[k] = item.get(k, "")
                    if (out.get("工作地点") or "").strip() and not (out.get("公司地点") or "").strip():
                        out["公司地点"] = out["工作地点"]
                    writer.writerow(out)
                    wrote += 1
                    out_rows += 1
                if wrote > 1:
                    expanded_rows += (wrote - 1)
            else:
                writer.writerow(row)
                out_rows += 1

    summary = {
        "input_csv": args.input_csv,
        "output_csv": args.output_csv,
        "cache_jsonl": args.cache_jsonl,
        "rows_in": len(rows),
        "rows_out": out_rows,
        "rows_expanded_net": expanded_rows,
        "unique_texts": len(text_map),
        "cache_size": len(cache),
        "api_new_calls": len(pending),
        "api_ok": api_ok,
        "api_fail": api_fail,
        "failed_hashes": failed_hashes,
        "params": {
            "model": args.model,
            "api_base": args.api_base,
            "request_interval": args.request_interval,
            "workers": workers,
            "max_attempts": args.max_attempts,
            "retry_wait": args.retry_wait,
            "min_text_len": args.min_text_len,
            "max_rows": args.max_rows,
            "max_new_calls": args.max_new_calls,
        },
    }
    with open(args.summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(
        "done\n"
        f"output: {args.output_csv}\n"
        f"summary: {args.summary_json}\n"
        f"api_new_calls: {len(pending)} ok={api_ok} fail={api_fail}\n"
        f"rows_in: {len(rows)} rows_out: {out_rows}"
    )


if __name__ == "__main__":
    main()
