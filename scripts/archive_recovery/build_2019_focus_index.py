#!/usr/bin/env python3
"""Build a 2019 high-intent CDX index from a raw CDX CSV."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List
from urllib.parse import urlsplit


INCLUDE_PATH = (
    "/job_detail/",
    "/job/",
    "/jobs/",
    "/position/",
    "/schooljobshow",
    "/social_show",
    "/xzxq",
    "/xiangqing",
    "/jobad/info",
)
INCLUDE_QUERY = ("jobid=", "jobnumber=", "adid=")
EXCLUDE_ANY = (
    "/wapi/zpantispam",
    "security-check",
    "/captcha/",
    "/portal/account/login",
    "/portal/account/register",
    "/search",
    "/alljob",
    "/campus",
    "/newslist",
    "/newsdetail",
    "/intern",
    "/recruit process",
    "/school trip",
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Filter raw CDX rows into 2019 high-intent job URLs."
    )
    p.add_argument("--in-cdx-csv", required=True, help="Input raw CDX CSV.")
    p.add_argument("--out-cdx-csv", required=True, help="Output filtered CDX CSV.")
    p.add_argument("--year", default="2019", help="Target year (default: 2019).")
    p.add_argument(
        "--dedupe-original",
        action="store_true",
        help="Dedupe by original URL (keep earliest timestamp).",
    )
    return p.parse_args()


def keep_row(row: Dict[str, str], year: str) -> bool:
    ts = (row.get("timestamp") or "").strip()
    if not ts.startswith(year):
        return False
    original = (row.get("original") or "").strip()
    if not original:
        return False

    ul = original.lower()
    if any(x in ul for x in EXCLUDE_ANY):
        return False

    s = urlsplit(original)
    path = (s.path or "").lower()
    query = (s.query or "").lower()

    if any(x in path for x in INCLUDE_PATH):
        return True
    if any(x in query for x in INCLUDE_QUERY):
        return True
    return False


def main() -> None:
    args = parse_args()
    inp = Path(args.in_cdx_csv)
    out = Path(args.out_cdx_csv)
    out.parent.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, str]] = []
    with inp.open("r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        fields = r.fieldnames or []
        for row in r:
            if keep_row(row, year=args.year):
                rows.append(row)

    if args.dedupe_original:
        rows.sort(key=lambda x: ((x.get("original") or ""), (x.get("timestamp") or "")))
        uniq: List[Dict[str, str]] = []
        seen = set()
        for row in rows:
            o = row.get("original") or ""
            if o in seen:
                continue
            seen.add(o)
            uniq.append(row)
        rows = uniq

    rows.sort(key=lambda x: ((x.get("timestamp") or ""), (x.get("original") or "")))

    with out.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    print(f"input:  {inp}")
    print(f"output: {out}")
    print(f"rows:   {len(rows)}")


if __name__ == "__main__":
    main()
