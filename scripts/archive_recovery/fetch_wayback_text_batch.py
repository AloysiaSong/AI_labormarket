#!/usr/bin/env python3
"""Fetch Wayback snapshot pages from CDX rows and extract clean text."""

from __future__ import annotations

import argparse
import csv
import json
import re
import time
from pathlib import Path
from typing import Dict, List, Tuple
from urllib.parse import urlsplit

import requests
import trafilatura

CHARSET_PATTERN = re.compile(
    r"charset\s*=\s*['\"]?\s*([a-zA-Z0-9._-]+)", re.IGNORECASE
)
ENCODING_ALIASES = {
    "utf8": "utf-8",
    "gb2312": "gb18030",
    "gb_2312": "gb18030",
    "gb_2312-80": "gb18030",
    "gbk": "gb18030",
    "x-gbk": "gb18030",
}
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
LOW_VALUE_PATTERNS = (
    "客户服务热线",
    "未经",
    "版权所有",
    "按拼音选择",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch Wayback snapshot text from a CDX CSV in small batches."
    )
    parser.add_argument(
        "--cdx-csv",
        default="data/archive_recovery/wayback/cdx_small_clean_50_v1_index.csv",
        help="Input CDX CSV path.",
    )
    parser.add_argument(
        "--out-dir",
        default="data/archive_recovery/wayback",
        help="Output directory.",
    )
    parser.add_argument(
        "--start-row",
        type=int,
        default=1,
        help="1-based row index in CDX data (excluding header).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=200,
        help="Maximum number of CDX rows to process.",
    )
    parser.add_argument(
        "--request-interval",
        type=float,
        default=1.2,
        help="Seconds to sleep between requests.",
    )
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=2,
        help="Max HTTP attempts per row.",
    )
    parser.add_argument(
        "--connect-timeout",
        type=float,
        default=3.0,
        help="Connect timeout seconds.",
    )
    parser.add_argument(
        "--read-timeout",
        type=float,
        default=10.0,
        help="Read timeout seconds.",
    )
    parser.add_argument(
        "--prefix",
        default="wayback_text_batch1",
        help="Output file prefix.",
    )
    parser.add_argument(
        "--quality-mode",
        choices=["off", "strict"],
        default="off",
        help="strict: prefilter non-job/homepage URLs, dedupe, and drop low-value text.",
    )
    parser.add_argument(
        "--min-text-length",
        type=int,
        default=220,
        help="Minimum extracted text length in strict quality mode.",
    )
    return parser.parse_args()


def load_cdx_rows(cdx_csv: Path, start_row: int, limit: int) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    with cdx_csv.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader, 1):
            if i < start_row:
                continue
            if len(rows) >= limit:
                break
            rows.append({"cdx_row_idx": str(i), **row})
    return rows


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


def select_rows_for_quality(
    rows: List[Dict[str, str]], quality_mode: str
) -> Tuple[List[Dict[str, str]], Dict[str, int]]:
    if quality_mode != "strict":
        return rows, {"input_rows": len(rows), "selected_rows": len(rows)}

    out: List[Dict[str, str]] = []
    seen_orig = set()
    seen_digest = set()
    stats = {
        "input_rows": len(rows),
        "drop_lowvalue_original": 0,
        "drop_dup_original": 0,
        "drop_dup_digest": 0,
        "selected_rows": 0,
    }
    for row in rows:
        original = (row.get("original") or "").strip()
        digest = (row.get("digest") or "").strip()
        if not is_high_value_original(original):
            stats["drop_lowvalue_original"] += 1
            continue
        if original and original in seen_orig:
            stats["drop_dup_original"] += 1
            continue
        if digest and digest in seen_digest:
            stats["drop_dup_digest"] += 1
            continue
        if original:
            seen_orig.add(original)
        if digest:
            seen_digest.add(digest)
        out.append(row)
    stats["selected_rows"] = len(out)
    return out, stats


def build_snapshot_url(timestamp: str, original: str) -> str:
    return f"https://web.archive.org/web/{timestamp}id_/{original}"


def extract_text_from_html(html: str) -> str:
    text = trafilatura.extract(
        html,
        include_comments=False,
        include_tables=False,
        no_fallback=False,
        output_format="txt",
    )
    return (text or "").strip()


def normalize_encoding(enc: str) -> str:
    e = (enc or "").strip().lower()
    return ENCODING_ALIASES.get(e, e)


def collect_declared_encodings(resp: requests.Response, raw: bytes) -> List[str]:
    candidates: List[str] = []

    # 1) HTTP header charset
    content_type = resp.headers.get("Content-Type", "")
    m = CHARSET_PATTERN.search(content_type)
    if m:
        candidates.append(normalize_encoding(m.group(1)))

    # 2) requests-detected encoding (can be useful when header missing)
    if resp.encoding:
        candidates.append(normalize_encoding(resp.encoding))
    if getattr(resp, "apparent_encoding", None):
        candidates.append(normalize_encoding(resp.apparent_encoding))

    # 3) meta charset in HTML head
    head_text = raw[:8192].decode("latin-1", errors="ignore")
    m2 = CHARSET_PATTERN.search(head_text)
    if m2:
        candidates.append(normalize_encoding(m2.group(1)))

    # 4) fallback list (include Chinese encodings explicitly)
    candidates.extend(["utf-8", "gb18030", "gbk", "gb2312", "big5", "latin-1"])

    # Dedupe while preserving order.
    out: List[str] = []
    seen = set()
    for c in candidates:
        if not c or c in seen:
            continue
        seen.add(c)
        out.append(c)
    return out


def decode_quality_score(text: str) -> int:
    """Lower score is better."""
    sample = text[:12000]
    replacement = sample.count("\ufffd")
    cjk = sum(1 for ch in sample if "\u4e00" <= ch <= "\u9fff")
    high_ascii = sum(1 for ch in sample if 128 <= ord(ch) <= 255)
    return replacement * 50 + high_ascii - cjk * 2


def decode_html_with_fallback(resp: requests.Response) -> tuple[str, str]:
    raw = resp.content or b""
    if not raw:
        return "", "empty"

    candidates = collect_declared_encodings(resp, raw)
    best_text = ""
    best_enc = ""
    best_score = None

    # First pass: strict decoding only.
    for enc in candidates:
        try:
            text = raw.decode(enc, errors="strict")
        except Exception:
            continue
        score = decode_quality_score(text)
        if best_score is None or score < best_score:
            best_score = score
            best_text = text
            best_enc = enc

    # Second pass: allow replacement if strict pass found nothing.
    if best_score is None:
        for enc in candidates:
            try:
                text = raw.decode(enc, errors="replace")
            except Exception:
                continue
            score = decode_quality_score(text)
            if best_score is None or score < best_score:
                best_score = score
                best_text = text
                best_enc = enc

    if best_score is None:
        return "", "decode_failed"
    return best_text, best_enc


def low_value_text_reason(text: str, min_text_length: int) -> str:
    n = len(text or "")
    if n < min_text_length:
        return "too_short"
    compact = re.sub(r"\s+", " ", text or "").strip()
    hit = sum(1 for p in LOW_VALUE_PATTERNS if p in compact)
    if hit >= 2 and len(compact) < max(min_text_length * 2, 800):
        return "boilerplate"
    return ""


def fetch_snapshot_text(
    session: requests.Session,
    row: Dict[str, str],
    max_attempts: int,
    timeout: tuple[float, float],
    quality_mode: str,
    min_text_length: int,
) -> tuple[bool, Dict[str, str]]:
    ts = (row.get("timestamp") or "").strip()
    original = (row.get("original") or "").strip()
    snapshot_url = build_snapshot_url(ts, original)

    last_error = ""
    for attempt in range(1, max_attempts + 1):
        try:
            resp = session.get(snapshot_url, timeout=timeout)
            status = resp.status_code
            if status != 200:
                last_error = f"HTTP_{status}"
                if attempt < max_attempts:
                    continue
                return False, {
                    "cdx_row_idx": row["cdx_row_idx"],
                    "timestamp": ts,
                    "original": original,
                    "snapshot_url": snapshot_url,
                    "seed_domain": row.get("seed_domain", ""),
                    "seed_platform": row.get("seed_platform", ""),
                    "seed_url": row.get("seed_url", ""),
                    "error_type": last_error,
                    "attempts": str(attempt),
                }

            html, decoded_encoding = decode_html_with_fallback(resp)
            text = extract_text_from_html(html)
            if not text:
                return False, {
                    "cdx_row_idx": row["cdx_row_idx"],
                    "timestamp": ts,
                    "original": original,
                    "snapshot_url": snapshot_url,
                    "seed_domain": row.get("seed_domain", ""),
                    "seed_platform": row.get("seed_platform", ""),
                    "seed_url": row.get("seed_url", ""),
                    "error_type": "EmptyExtractedText",
                    "decoded_encoding": decoded_encoding,
                    "attempts": str(attempt),
                }

            if quality_mode == "strict":
                reason = low_value_text_reason(text, min_text_length=min_text_length)
                if reason:
                    return False, {
                        "cdx_row_idx": row["cdx_row_idx"],
                        "timestamp": ts,
                        "original": original,
                        "snapshot_url": snapshot_url,
                        "seed_domain": row.get("seed_domain", ""),
                        "seed_platform": row.get("seed_platform", ""),
                        "seed_url": row.get("seed_url", ""),
                        "error_type": f"LowValueText:{reason}",
                        "decoded_encoding": decoded_encoding,
                        "attempts": str(attempt),
                    }

            return True, {
                "cdx_row_idx": row["cdx_row_idx"],
                "timestamp": ts,
                "original": original,
                "snapshot_url": snapshot_url,
                "seed_domain": row.get("seed_domain", ""),
                "seed_platform": row.get("seed_platform", ""),
                "seed_url": row.get("seed_url", ""),
                "digest": row.get("digest", ""),
                "decoded_encoding": decoded_encoding,
                "text_length": str(len(text)),
                "text": text,
            }
        except requests.exceptions.ReadTimeout:
            last_error = "ReadTimeout"
        except requests.exceptions.ConnectTimeout:
            last_error = "ConnectTimeout"
        except requests.exceptions.ConnectionError as e:
            last_error = f"ConnectionError:{str(e)[:120]}"
        except Exception as e:  # pragma: no cover
            last_error = f"Exception:{str(e)[:120]}"

    return False, {
        "cdx_row_idx": row["cdx_row_idx"],
        "timestamp": ts,
        "original": original,
        "snapshot_url": snapshot_url,
        "seed_domain": row.get("seed_domain", ""),
        "seed_platform": row.get("seed_platform", ""),
        "seed_url": row.get("seed_url", ""),
        "error_type": last_error or "UnknownError",
        "decoded_encoding": "",
        "attempts": str(max_attempts),
    }


def write_csv(path: Path, rows: List[Dict[str, str]], fieldnames: List[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    cdx_csv = Path(args.cdx_csv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = load_cdx_rows(cdx_csv, start_row=args.start_row, limit=args.limit)
    if not rows:
        raise SystemExit("No CDX rows selected. Check --start-row and --limit.")
    selected_rows, selection_stats = select_rows_for_quality(
        rows, quality_mode=args.quality_mode
    )
    if not selected_rows:
        raise SystemExit(
            "No rows left after quality prefilter. Try --quality-mode off "
            "or different CDX input."
        )

    ok_rows: List[Dict[str, str]] = []
    err_rows: List[Dict[str, str]] = []

    session = requests.Session()
    total = len(selected_rows)
    for i, row in enumerate(selected_rows, 1):
        ok, payload = fetch_snapshot_text(
            session=session,
            row=row,
            max_attempts=args.max_attempts,
            timeout=(args.connect_timeout, args.read_timeout),
            quality_mode=args.quality_mode,
            min_text_length=args.min_text_length,
        )
        if ok:
            ok_rows.append(payload)
        else:
            err_rows.append(payload)

        if i % 25 == 0 or i == total:
            print(f"progress {i}/{total} success={len(ok_rows)} errors={len(err_rows)}")
        time.sleep(args.request_interval)

    text_csv = out_dir / f"{args.prefix}_text.csv"
    err_csv = out_dir / f"{args.prefix}_errors.csv"
    summary_json = out_dir / f"{args.prefix}_summary.json"

    write_csv(
        text_csv,
        ok_rows,
        [
            "cdx_row_idx",
            "timestamp",
            "original",
            "snapshot_url",
            "seed_domain",
            "seed_platform",
            "seed_url",
            "digest",
            "decoded_encoding",
            "text_length",
            "text",
        ],
    )
    write_csv(
        err_csv,
        err_rows,
        [
            "cdx_row_idx",
            "timestamp",
            "original",
            "snapshot_url",
            "seed_domain",
            "seed_platform",
            "seed_url",
            "error_type",
            "decoded_encoding",
            "attempts",
        ],
    )

    summary = {
        "cdx_csv": str(cdx_csv),
        "start_row": args.start_row,
        "limit": args.limit,
        "processed": total,
        "success_rows": len(ok_rows),
        "error_rows": len(err_rows),
        "quality_mode": args.quality_mode,
        "min_text_length": args.min_text_length,
        "selection_stats": selection_stats,
        "params": {
            "request_interval": args.request_interval,
            "max_attempts": args.max_attempts,
            "connect_timeout": args.connect_timeout,
            "read_timeout": args.read_timeout,
        },
    }
    with summary_json.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("done")
    print(f"text_csv: {text_csv}")
    print(f"err_csv:  {err_csv}")
    print(f"summary:  {summary_json}")


if __name__ == "__main__":
    main()
