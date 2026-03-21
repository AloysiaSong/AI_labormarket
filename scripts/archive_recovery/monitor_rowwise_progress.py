#!/usr/bin/env python3
import argparse
import datetime as dt
import json
import subprocess
import time
from pathlib import Path


def count_cache(cache_jsonl: Path):
    ok = 0
    fail = 0
    total = 0
    if not cache_jsonl.exists():
        return ok, fail, total
    with cache_jsonl.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            total += 1
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if obj.get("ok") is True:
                ok += 1
            else:
                fail += 1
    return ok, fail, total


def list_target_pids(keyword: str):
    try:
        out = subprocess.check_output(
            ["ps", "aux"],
            text=True,
        )
    except Exception:
        return []
    pids = []
    for line in out.splitlines():
        line = line.strip()
        if not line:
            continue
        if keyword not in line:
            continue
        parts = line.split()
        if len(parts) < 2:
            continue
        try:
            pids.append(int(parts[1]))
        except Exception:
            continue
    return pids


def now_iso():
    return dt.datetime.now().isoformat(timespec="seconds")


def main():
    parser = argparse.ArgumentParser(description="Monitor DeepSeek rowwise process and cache progress.")
    parser.add_argument(
        "--cache-jsonl",
        default="data/archive_recovery/unified/wayback_2019_deepseek_rowwise_cache_full.jsonl",
    )
    parser.add_argument(
        "--keyword",
        default="enrich_wayback_2019_deepseek_rowwise.py",
    )
    parser.add_argument("--interval", type=float, default=30.0)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()

    cache_jsonl = Path(args.cache_jsonl)
    last_ok = None
    last_fail = None
    print(f"[{now_iso()}] monitor_start keyword={args.keyword} cache={cache_jsonl}", flush=True)

    while True:
        pids = list_target_pids(args.keyword)
        ok, fail, total = count_cache(cache_jsonl)
        delta_ok = 0 if last_ok is None else ok - last_ok
        delta_fail = 0 if last_fail is None else fail - last_fail
        last_ok = ok
        last_fail = fail
        print(
            f"[{now_iso()}] proc_count={len(pids)} pids={pids} ok={ok} fail={fail} total={total} "
            f"delta_ok={delta_ok} delta_fail={delta_fail}",
            flush=True,
        )
        if args.once:
            return
        time.sleep(max(1.0, args.interval))


if __name__ == "__main__":
    main()
