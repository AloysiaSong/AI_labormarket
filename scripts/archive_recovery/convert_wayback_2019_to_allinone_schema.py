#!/usr/bin/env python3
import argparse
import csv
import glob
import json
import os
from datetime import datetime


PLATFORM_MAP = {
    "bosszhipin": "BOSS直聘",
    "zhaopin": "智联招聘",
    "智联招聘": "智联招聘",
    "51job": "前程无忧",
    "liepin": "猎聘",
    "lagou": "拉勾招聘",
}


def normalize_platform(seed_platform: str, seed_domain: str) -> str:
    key = (seed_platform or "").strip()
    if key in PLATFORM_MAP:
        return PLATFORM_MAP[key]
    if key:
        return key
    return (seed_domain or "").strip()


def parse_publish_date(ts: str):
    ts = (ts or "").strip()
    if len(ts) < 8 or not ts[:8].isdigit():
        return "", ""
    try:
        dt = datetime.strptime(ts[:8], "%Y%m%d")
        return dt.strftime("%Y-%m-%d"), str(dt.year)
    except ValueError:
        return "", ""


def dedup_key(row: dict):
    snapshot_url = (row.get("snapshot_url") or "").strip()
    if snapshot_url:
        return ("snapshot_url", snapshot_url)
    return (
        "original_ts",
        (row.get("original") or "").strip(),
        (row.get("timestamp") or "").strip(),
    )


def main():
    parser = argparse.ArgumentParser(
        description="Convert merged Wayback 2019 text CSVs to all_in_one schema."
    )
    parser.add_argument(
        "--input-glob",
        default="data/archive_recovery/wayback/wayback_text_2019*_text.csv",
        help="Glob for 2019 wayback text CSV files.",
    )
    parser.add_argument(
        "--schema-csv",
        default="dataset/all_in_one.csv",
        help="CSV whose header defines target schema.",
    )
    parser.add_argument(
        "--output-csv",
        default="data/archive_recovery/unified/wayback_2019_allinone_schema.csv",
        help="Output CSV path.",
    )
    parser.add_argument(
        "--summary-json",
        default="data/archive_recovery/unified/wayback_2019_allinone_schema_summary.json",
        help="Output summary JSON path.",
    )
    args = parser.parse_args()

    files = sorted(glob.glob(args.input_glob))
    if not files:
        raise SystemExit(f"No files matched: {args.input_glob}")

    with open(args.schema_csv, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)
        target_fields = next(reader)

    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
    os.makedirs(os.path.dirname(args.summary_json), exist_ok=True)

    rows_in = 0
    rows_out = 0
    dup_rows = 0
    platform_counts = {}
    seen = set()

    with open(args.output_csv, "w", encoding="utf-8-sig", newline="") as fout:
        writer = csv.DictWriter(fout, fieldnames=target_fields, extrasaction="ignore")
        writer.writeheader()

        for path in files:
            with open(path, "r", encoding="utf-8-sig", newline="") as fin:
                reader = csv.DictReader(fin)
                for row in reader:
                    rows_in += 1
                    key = dedup_key(row)
                    if key in seen:
                        dup_rows += 1
                        continue
                    seen.add(key)

                    publish_date, publish_year = parse_publish_date(row.get("timestamp", ""))
                    platform = normalize_platform(
                        row.get("seed_platform", ""),
                        row.get("seed_domain", ""),
                    )
                    platform_counts[platform] = platform_counts.get(platform, 0) + 1

                    out = {k: "" for k in target_fields}
                    out["职位描述"] = (row.get("text") or "").strip()
                    out["来源平台"] = platform
                    out["招聘发布日期"] = publish_date
                    out["招聘发布年份"] = publish_year
                    out["来源"] = (row.get("snapshot_url") or row.get("original") or "").strip()
                    writer.writerow(out)
                    rows_out += 1

    summary = {
        "input_glob": args.input_glob,
        "files": files,
        "rows_in": rows_in,
        "rows_out": rows_out,
        "dup_rows": dup_rows,
        "output_csv": args.output_csv,
        "platform_counts": dict(sorted(platform_counts.items(), key=lambda x: x[1], reverse=True)),
    }

    with open(args.summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"done\noutput: {args.output_csv}\nsummary: {args.summary_json}\nrows_out: {rows_out}")


if __name__ == "__main__":
    main()
