#!/usr/bin/env python3
"""Merge multiple wayback text CSV batches into one file."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Merge wayback text batch CSV files.")
    p.add_argument(
        "--glob",
        required=True,
        help="Glob pattern for input CSV files, e.g. data/.../wayback_text_2019_*.csv",
    )
    p.add_argument("--out-csv", required=True, help="Merged output CSV path.")
    p.add_argument(
        "--dedupe-key",
        default="",
        help="Optional dedupe key column, e.g. original or snapshot_url.",
    )
    p.add_argument(
        "--sort-by",
        default="timestamp",
        help="Optional sort key column (default: timestamp).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    files = sorted(Path().glob(args.glob))
    if not files:
        raise SystemExit(f"No files matched: {args.glob}")

    rows: List[Dict[str, str]] = []
    fields: List[str] = []
    for f in files:
        with f.open("r", encoding="utf-8", newline="") as fp:
            r = csv.DictReader(fp)
            if not fields:
                fields = list(r.fieldnames or [])
            for row in r:
                rows.append(row)

    if args.dedupe_key:
        key = args.dedupe_key
        uniq: List[Dict[str, str]] = []
        seen = set()
        for row in rows:
            v = row.get(key, "")
            if v in seen:
                continue
            seen.add(v)
            uniq.append(row)
        rows = uniq

    sort_key = args.sort_by.strip()
    if sort_key:
        rows.sort(key=lambda x: (x.get(sort_key, ""), x.get("original", "")))

    out = Path(args.out_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="") as fp:
        w = csv.DictWriter(fp, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    print(f"files: {len(files)}")
    print(f"rows:  {len(rows)}")
    print(f"out:   {out}")


if __name__ == "__main__":
    main()

