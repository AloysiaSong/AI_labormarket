#!/usr/bin/env python3
"""Run Wayback CDX in serial batches with retry and rerun list output."""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
import time
from pathlib import Path
from typing import Dict, List, Tuple
from urllib.parse import urlsplit

import requests


RETRYABLE_HTTP = {429, 500, 502, 503, 504}
DEFAULT_PLATFORM_ALLOWLIST = "zhaopin,51job,bosszhipin,liepin,lagou,chinahr"
IPV4_RE = re.compile(r"^(?:\d{1,3}\.){3}\d{1,3}$")
ROOT_PATHS = {
    "",
    "/",
    "/index",
    "/index.html",
    "/index.htm",
    "/default",
    "/default.html",
    "/home",
    "/homepage",
}
JOB_URL_KEYWORDS = (
    "job",
    "jobs",
    "job-detail",
    "jobdetail",
    "joblist",
    "position",
    "vacancy",
    "recruit",
    "career",
    "search",
    "keyword",
    "jobid",
    "apply",
    "campus",
    "school",
    "zhaopin",
    "xiangqing",
    "xq",
    "qiuzhi",
    "gongzuo",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch-serial Wayback CDX runner with failure rerun list."
    )
    parser.add_argument(
        "--seed-csv",
        default="data/archive_recovery/seeds/seed_urls_highvalue.csv",
        help="Seed CSV with columns: domain,url,platform",
    )
    parser.add_argument(
        "--out-dir",
        default="data/archive_recovery/wayback",
        help="Output directory for CDX files",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=50,
        help="Number of seeds per serial batch",
    )
    parser.add_argument(
        "--request-interval",
        type=float,
        default=1.2,
        help="Sleep seconds between requests",
    )
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=3,
        help="Max attempts per seed",
    )
    parser.add_argument(
        "--connect-timeout",
        type=float,
        default=5.0,
        help="Connect timeout seconds",
    )
    parser.add_argument(
        "--read-timeout",
        type=float,
        default=18.0,
        help="Read timeout seconds",
    )
    parser.add_argument(
        "--batch-cooldown",
        type=float,
        default=8.0,
        help="Sleep seconds between batches",
    )
    parser.add_argument(
        "--from-date",
        default="201401",
        help="CDX from date (YYYYMM)",
    )
    parser.add_argument(
        "--to-date",
        default="202012",
        help="CDX to date (YYYYMM)",
    )
    parser.add_argument(
        "--start-idx",
        type=int,
        default=1,
        help="1-based start seed index for partial rerun",
    )
    parser.add_argument(
        "--end-idx",
        type=int,
        default=0,
        help="1-based end seed index (0 means all)",
    )
    parser.add_argument(
        "--prefix",
        default="cdx_highvalue_batch_serial",
        help="Output file prefix",
    )
    parser.add_argument(
        "--small-clean-mode",
        action="store_true",
        help=(
            "Enable small-and-clean seed mode: keep mainstream platform seeds only, "
            "drop noisy URLs, cap per domain, and apply seed limit."
        ),
    )
    parser.add_argument(
        "--seed-limit",
        type=int,
        default=0,
        help="Limit number of seeds after filtering (0 means no limit).",
    )
    parser.add_argument(
        "--platform-allowlist",
        default=DEFAULT_PLATFORM_ALLOWLIST,
        help="Comma-separated mainstream platforms for clean mode.",
    )
    parser.add_argument(
        "--per-domain-cap",
        type=int,
        default=5,
        help="Max seeds per domain in small clean mode.",
    )
    parser.add_argument(
        "--strict-original-filter",
        action="store_true",
        help=(
            "Filter CDX originals to keep likely job/listing URLs only and "
            "drop homepage-like originals."
        ),
    )
    parser.add_argument(
        "--collapse",
        default="digest",
        help=(
            "CDX collapse field. Use 'digest' (default) for de-dup; "
            "use 'none' to disable collapse and keep all captures."
        ),
    )
    return parser.parse_args()


def platform_domain_hints(platform: str) -> List[str]:
    mapping = {
        "zhaopin": ["zhaopin.com", "zhiye.com"],
        "51job": ["51job.com"],
        "bosszhipin": ["bosszhipin.com", "zhipin.com"],
        "liepin": ["liepin.com"],
        "lagou": ["lagou.com"],
        "chinahr": ["chinahr.com"],
    }
    return mapping.get(platform, [])


def is_clean_seed(
    seed: Dict[str, str],
    allow_platforms: List[str],
    allow_domain_hints: List[str],
) -> bool:
    platform = seed["platform"]
    domain = seed["domain"]
    url = seed["url"]

    # Keep mainstream platforms/domains only.
    if not (
        platform in allow_platforms
        or any(h in domain for h in allow_domain_hints)
    ):
        return False

    # Drop noisy URLs that commonly pollute CDX results.
    if "@" in url:
        return False
    if IPV4_RE.match(domain):
        return False
    if len(url) > 400:
        return False
    return True


def split_url(url: str) -> Tuple[str, str, str]:
    try:
        u = urlsplit(url)
    except Exception:
        return "", "/", ""
    host = (u.netloc or "").lower()
    path = (u.path or "/").strip()
    query = (u.query or "").strip()
    return host, path, query


def norm_path(path: str) -> str:
    p = (path or "/").strip().lower()
    if not p.startswith("/"):
        p = "/" + p
    p = p.rstrip("/")
    return p if p else "/"


def is_root_like_path(path: str) -> bool:
    return norm_path(path) in ROOT_PATHS


def is_likely_job_url(url: str) -> bool:
    host, path, query = split_url(url)
    hay = f"{host}{norm_path(path)}?{query}".lower()
    return any(k in hay for k in JOB_URL_KEYWORDS)


def is_high_value_original(url: str) -> bool:
    if not url or "@" in url:
        return False
    host, path, _ = split_url(url)
    if not host:
        return False
    if is_root_like_path(path):
        return False
    return is_likely_job_url(url)


def load_seeds(
    seed_csv: Path,
    start_idx: int,
    end_idx: int,
    small_clean_mode: bool,
    seed_limit: int,
    platform_allowlist: str,
    per_domain_cap: int,
) -> List[Dict[str, str]]:
    seeds: List[Dict[str, str]] = []
    with seed_csv.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader, 1):
            if i < start_idx:
                continue
            if end_idx and i > end_idx:
                break
            url = (row.get("url") or "").strip()
            domain = (row.get("domain") or "").strip().lower()
            platform = (row.get("platform") or "other").strip().lower()
            if not url or not domain:
                continue
            seeds.append(
                {
                    "idx": i,
                    "url": url,
                    "domain": domain,
                    "platform": platform if platform else "other",
                }
            )

    # De-duplicate by (domain, url) while preserving order.
    seen = set()
    uniq = []
    for s in seeds:
        key = (s["domain"], s["url"])
        if key in seen:
            continue
        seen.add(key)
        uniq.append(s)
    seeds = uniq

    if small_clean_mode:
        allow_platforms = [
            p.strip().lower() for p in platform_allowlist.split(",") if p.strip()
        ]
        allow_domain_hints = []
        for p in allow_platforms:
            allow_domain_hints.extend(platform_domain_hints(p))

        seeds = [
            s
            for s in seeds
            if is_clean_seed(s, allow_platforms, allow_domain_hints)
        ]

        # Prefer non-homepage, job-like URLs in clean mode.
        seeds.sort(
            key=lambda s: (
                1 if is_root_like_path(split_url(s["url"])[1]) else 0,
                0 if is_likely_job_url(s["url"]) else 1,
                0 if "?" in s["url"] else 1,
                len(s["url"]),
                s["domain"],
                s["url"],
            )
        )

        # Cap per domain to avoid one subdomain dominating.
        domain_count: Dict[str, int] = {}
        capped = []
        for s in seeds:
            c = domain_count.get(s["domain"], 0)
            if c >= per_domain_cap:
                continue
            domain_count[s["domain"]] = c + 1
            capped.append(s)
        seeds = capped

    if seed_limit > 0:
        seeds = seeds[:seed_limit]
    return seeds


def request_cdx(
    session: requests.Session,
    seed: Dict[str, str],
    from_date: str,
    to_date: str,
    strict_original_filter: bool,
    collapse: str,
    max_attempts: int,
    timeout: Tuple[float, float],
) -> Tuple[str, List[Dict[str, str]], Dict[str, str]]:
    """Return (kind, hit_rows, meta).

    kind: hit|nohit|error
    """
    params = {
        "url": seed["url"],
        "from": from_date,
        "to": to_date,
        "output": "json",
        "fl": "timestamp,original,mimetype,statuscode,digest,length",
        "filter": ["statuscode:200", "mimetype:text/html"],
    }
    if collapse and collapse.strip().lower() != "none":
        params["collapse"] = collapse.strip()
    last_error = ""

    for attempt in range(1, max_attempts + 1):
        try:
            resp = session.get(
                "https://web.archive.org/cdx/search/cdx",
                params=params,
                timeout=timeout,
            )
            status = resp.status_code

            if status in RETRYABLE_HTTP and attempt < max_attempts:
                time.sleep((1.0 * attempt) + random.uniform(0.1, 0.4))
                continue

            if status != 200:
                return (
                    "error",
                    [],
                    {
                        "error_type": f"HTTP_{status}",
                        "detail": (resp.text or "")[:200],
                        "attempts": str(attempt),
                    },
                )

            text = resp.text or ""
            try:
                data = json.loads(text)
            except Exception:
                t = text.strip().lower()
                if t in ("[]", "[ ]") or t.startswith("no captures"):
                    return (
                        "nohit",
                        [],
                        {
                            "reason": "no_capture_text",
                            "attempts": str(attempt),
                        },
                    )
                return (
                    "error",
                    [],
                    {
                        "error_type": "JSONDecodeError",
                        "detail": text[:200],
                        "attempts": str(attempt),
                    },
                )

            if not isinstance(data, list):
                return (
                    "error",
                    [],
                    {
                        "error_type": "UnexpectedJSONType",
                        "detail": str(type(data)),
                        "attempts": str(attempt),
                    },
                )

            if len(data) <= 1:
                return (
                    "nohit",
                    [],
                    {
                        "reason": "json_empty_or_header_only",
                        "attempts": str(attempt),
                    },
                )

            header = data[0]
            hit_rows: List[Dict[str, str]] = []
            for item in data[1:]:
                rec = dict(zip(header, item))
                mimetype = (rec.get("mimetype") or "").strip().lower()
                status = (rec.get("statuscode") or "").strip()
                original = (rec.get("original") or "").strip()

                # Strong post-filter in case upstream filtering is bypassed.
                if status != "200":
                    continue
                if mimetype != "text/html":
                    continue
                if "warc/revisit" in mimetype:
                    continue
                if "@" in original:
                    continue
                if strict_original_filter and not is_high_value_original(original):
                    continue

                rec["seed_idx"] = str(seed["idx"])
                rec["seed_domain"] = seed["domain"]
                rec["seed_platform"] = seed["platform"]
                rec["seed_url"] = seed["url"]
                hit_rows.append(rec)

            if not hit_rows:
                return (
                    "nohit",
                    [],
                    {
                        "reason": "filtered_out_by_quality_rules",
                        "attempts": str(attempt),
                    },
                )
            return ("hit", hit_rows, {"attempts": str(attempt)})

        except requests.exceptions.ReadTimeout:
            last_error = "ReadTimeout"
        except requests.exceptions.ConnectTimeout:
            last_error = "ConnectTimeout"
        except requests.exceptions.ConnectionError as e:
            last_error = f"ConnectionError:{str(e)[:120]}"
        except Exception as e:
            last_error = f"Exception:{str(e)[:120]}"

        if attempt < max_attempts:
            time.sleep((1.0 * attempt) + random.uniform(0.1, 0.4))

    return (
        "error",
        [],
        {"error_type": last_error or "UnknownError", "detail": "", "attempts": str(max_attempts)},
    )


def write_csv(path: Path, rows: List[Dict[str, str]], fieldnames: List[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    seed_csv = Path(args.seed_csv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    seeds = load_seeds(
        seed_csv=seed_csv,
        start_idx=args.start_idx,
        end_idx=args.end_idx,
        small_clean_mode=args.small_clean_mode,
        seed_limit=args.seed_limit,
        platform_allowlist=args.platform_allowlist,
        per_domain_cap=args.per_domain_cap,
    )
    if not seeds:
        raise SystemExit("No seeds loaded from seed CSV and index range.")

    hit_rows: List[Dict[str, str]] = []
    error_rows: List[Dict[str, str]] = []
    nohit_rows: List[Dict[str, str]] = []

    session = requests.Session()
    total = len(seeds)
    batches = (total + args.batch_size - 1) // args.batch_size

    hit_csv = out_dir / f"{args.prefix}_index.csv"
    err_csv = out_dir / f"{args.prefix}_errors.csv"
    nohit_csv = out_dir / f"{args.prefix}_nohits.csv"
    rerun_csv = out_dir / f"{args.prefix}_failed_seeds_rerun.csv"
    summary_json = out_dir / f"{args.prefix}_summary.json"

    for b in range(batches):
        start = b * args.batch_size
        end = min((b + 1) * args.batch_size, total)
        batch = seeds[start:end]
        print(f"[batch {b+1}/{batches}] seeds {start+1}-{end}")

        for s in batch:
            kind, rows, meta = request_cdx(
                session=session,
                seed=s,
                from_date=args.from_date,
                to_date=args.to_date,
                strict_original_filter=(
                    args.strict_original_filter or args.small_clean_mode
                ),
                collapse=args.collapse,
                max_attempts=args.max_attempts,
                timeout=(args.connect_timeout, args.read_timeout),
            )
            if kind == "hit":
                hit_rows.extend(rows)
            elif kind == "nohit":
                nohit_rows.append(
                    {
                        "idx": s["idx"],
                        "seed_url": s["url"],
                        "seed_domain": s["domain"],
                        "seed_platform": s["platform"],
                        "reason": meta.get("reason", ""),
                        "attempts": meta.get("attempts", ""),
                    }
                )
            else:
                error_rows.append(
                    {
                        "idx": s["idx"],
                        "seed_url": s["url"],
                        "seed_domain": s["domain"],
                        "seed_platform": s["platform"],
                        "error_type": meta.get("error_type", ""),
                        "detail": meta.get("detail", ""),
                        "attempts": meta.get("attempts", ""),
                    }
                )
            time.sleep(args.request_interval)

        # checkpoint outputs after each batch
        hit_fields = [
            "timestamp",
            "original",
            "mimetype",
            "statuscode",
            "digest",
            "length",
            "seed_idx",
            "seed_domain",
            "seed_platform",
            "seed_url",
        ]
        err_fields = [
            "idx",
            "seed_url",
            "seed_domain",
            "seed_platform",
            "error_type",
            "detail",
            "attempts",
        ]
        nohit_fields = [
            "idx",
            "seed_url",
            "seed_domain",
            "seed_platform",
            "reason",
            "attempts",
        ]

        write_csv(hit_csv, hit_rows, hit_fields)
        write_csv(err_csv, error_rows, err_fields)
        write_csv(nohit_csv, nohit_rows, nohit_fields)
        write_csv(
            rerun_csv,
            [
                {
                    "idx": r["idx"],
                    "seed_url": r["seed_url"],
                    "seed_domain": r["seed_domain"],
                    "seed_platform": r["seed_platform"],
                }
                for r in error_rows
            ],
            ["idx", "seed_url", "seed_domain", "seed_platform"],
        )

        year_counts: Dict[str, int] = {}
        for r in hit_rows:
            ts = (r.get("timestamp") or "").strip()
            y = ts[:4] if len(ts) >= 4 and ts[:4].isdigit() else "UNKNOWN"
            year_counts[y] = year_counts.get(y, 0) + 1

        summary = {
            "seed_total": total,
            "seed_processed": end,
            "seed_hit": len({r["seed_url"] for r in hit_rows}),
            "seed_nohit": len(nohit_rows),
            "seed_error": len(error_rows),
            "hit_rows": len(hit_rows),
            "nohit_rows": len(nohit_rows),
            "error_rows": len(error_rows),
            "year_counts": dict(sorted(year_counts.items())),
            "params": {
                "batch_size": args.batch_size,
                "request_interval": args.request_interval,
                "max_attempts": args.max_attempts,
                "connect_timeout": args.connect_timeout,
                "read_timeout": args.read_timeout,
                "batch_cooldown": args.batch_cooldown,
                "from_date": args.from_date,
                "to_date": args.to_date,
                "start_idx": args.start_idx,
                "end_idx": args.end_idx,
                "small_clean_mode": args.small_clean_mode,
                "seed_limit": args.seed_limit,
                "platform_allowlist": args.platform_allowlist,
                "per_domain_cap": args.per_domain_cap,
                "strict_original_filter": (
                    args.strict_original_filter or args.small_clean_mode
                ),
                "collapse": args.collapse,
            },
        }
        with summary_json.open("w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

        print(
            f"  checkpoint: hit_rows={len(hit_rows)} nohit={len(nohit_rows)} "
            f"error={len(error_rows)}"
        )

        if end < total and args.batch_cooldown > 0:
            time.sleep(args.batch_cooldown)

    print("done")
    print(f"summary: {summary_json}")
    print(f"index:   {hit_csv}")
    print(f"nohits:  {nohit_csv}")
    print(f"errors:  {err_csv}")
    print(f"rerun:   {rerun_csv}")


if __name__ == "__main__":
    main()
