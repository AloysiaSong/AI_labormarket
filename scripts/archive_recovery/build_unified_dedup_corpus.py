#!/usr/bin/env python3
"""Build a unified, text-deduplicated job corpus from local sources.

Sources currently supported:
1) Historical CSV (all_in_one.csv)
2) Wayback extracted text CSVs (*_text.csv)

Dedup key is sha1(normalized_text).
"""

from __future__ import annotations

import argparse
import csv
import glob
import hashlib
import html
import json
import re
import sqlite3
import time
from pathlib import Path
from typing import Dict, Iterable, List, Tuple
from urllib.parse import urlsplit


TAG_RE = re.compile(r"<[^>]+>")
WS_RE = re.compile(r"\s+")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build unified deduplicated job corpus.")
    p.add_argument(
        "--db-path",
        default="data/archive_recovery/unified/job_corpus_dedup.sqlite",
        help="SQLite DB output path.",
    )
    p.add_argument(
        "--summary-json",
        default="data/archive_recovery/unified/job_corpus_dedup_summary.json",
        help="Summary JSON path.",
    )
    p.add_argument(
        "--all-in-one-csv",
        default="dataset/all_in_one.csv",
        help="Historical all-in-one CSV path.",
    )
    p.add_argument(
        "--all-in-one-start-row",
        type=int,
        default=1,
        help="1-based start row for all_in_one ingestion.",
    )
    p.add_argument(
        "--all-in-one-max-rows",
        type=int,
        default=0,
        help="Max rows to read from all_in_one after start row (0 means no limit).",
    )
    p.add_argument(
        "--wayback-glob",
        default="data/archive_recovery/wayback/*_text.csv",
        help="Glob for wayback text CSV files.",
    )
    p.add_argument(
        "--min-text-len",
        type=int,
        default=120,
        help="Minimum cleaned text length to keep.",
    )
    p.add_argument(
        "--commit-every",
        type=int,
        default=5000,
        help="SQLite commit interval.",
    )
    p.add_argument(
        "--skip-all-in-one",
        action="store_true",
        help="Skip all_in_one source.",
    )
    p.add_argument(
        "--skip-wayback",
        action="store_true",
        help="Skip wayback source.",
    )
    return p.parse_args()


def clean_text(text: str) -> str:
    t = html.unescape(text or "")
    t = TAG_RE.sub(" ", t)
    t = WS_RE.sub(" ", t).strip()
    return t


def text_hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8", errors="ignore")).hexdigest()


def get_platform_from_host(host: str) -> str:
    h = (host or "").lower()
    if "zhipin.com" in h or "bosszhipin.com" in h:
        return "bosszhipin"
    if "zhaopin.com" in h or "zhiye.com" in h:
        return "zhaopin"
    if "51job.com" in h:
        return "51job"
    if "liepin.com" in h:
        return "liepin"
    if "lagou.com" in h:
        return "lagou"
    if "chinahr.com" in h:
        return "chinahr"
    return "other"


def init_db(conn: sqlite3.Connection) -> None:
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    conn.execute("PRAGMA temp_store=MEMORY;")
    conn.execute("PRAGMA cache_size=200000;")
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS dedup_corpus (
            text_hash TEXT PRIMARY KEY,
            text TEXT NOT NULL,
            text_len INTEGER NOT NULL,
            title TEXT,
            company TEXT,
            platform TEXT,
            year TEXT,
            publish_date TEXT,
            source TEXT,
            source_file TEXT,
            original_url TEXT,
            snapshot_url TEXT,
            inserted_at TEXT DEFAULT (datetime('now'))
        )
        """
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_dedup_platform_year ON dedup_corpus(platform, year)"
    )
    conn.commit()


def iter_all_in_one_rows(
    csv_path: Path, start_row: int, max_rows: int
) -> Iterable[Tuple[int, Dict[str, str]]]:
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader, 1):
            if i < start_row:
                continue
            if max_rows > 0 and i >= start_row + max_rows:
                break
            yield i, row


def ingest_all_in_one(
    conn: sqlite3.Connection,
    csv_path: Path,
    start_row: int,
    max_rows: int,
    min_text_len: int,
    commit_every: int,
) -> Dict[str, int]:
    stats = {
        "read_rows": 0,
        "kept_rows": 0,
        "inserted_rows": 0,
        "dup_rows": 0,
        "short_rows": 0,
        "empty_rows": 0,
        "last_row_idx": start_row - 1,
    }
    ins_sql = """
        INSERT OR IGNORE INTO dedup_corpus (
            text_hash, text, text_len, title, company, platform, year,
            publish_date, source, source_file
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """

    t0 = time.time()
    for i, row in iter_all_in_one_rows(csv_path, start_row, max_rows):
        stats["read_rows"] += 1
        stats["last_row_idx"] = i

        raw_text = row.get("职位描述", "")
        text = clean_text(raw_text)
        if not text:
            stats["empty_rows"] += 1
            continue
        if len(text) < min_text_len:
            stats["short_rows"] += 1
            continue
        stats["kept_rows"] += 1

        h = text_hash(text)
        # Prefer the concrete source (e.g., 智联招聘) over data vendor label.
        platform = (row.get("来源") or row.get("来源平台") or "").strip() or "unknown"
        year = (row.get("招聘发布年份") or "").strip()[:4]
        publish_date = (row.get("招聘发布日期") or "").strip()

        before = conn.total_changes
        conn.execute(
            ins_sql,
            (
                h,
                text,
                len(text),
                (row.get("招聘岗位") or "").strip(),
                (row.get("企业名称") or row.get("\ufeff企业名称") or "").strip(),
                platform,
                year,
                publish_date,
                "history_all_in_one",
                str(csv_path),
            ),
        )
        if conn.total_changes > before:
            stats["inserted_rows"] += 1
        else:
            stats["dup_rows"] += 1

        if stats["read_rows"] % commit_every == 0:
            conn.commit()
            dt = time.time() - t0
            print(
                f"[all_in_one] read={stats['read_rows']} kept={stats['kept_rows']} "
                f"inserted={stats['inserted_rows']} dup={stats['dup_rows']} "
                f"row_idx={stats['last_row_idx']} elapsed={dt:.1f}s"
            )

    conn.commit()
    return stats


def ingest_wayback(
    conn: sqlite3.Connection,
    files: List[Path],
    min_text_len: int,
    commit_every: int,
) -> Dict[str, int]:
    stats = {
        "files": len(files),
        "read_rows": 0,
        "kept_rows": 0,
        "inserted_rows": 0,
        "dup_rows": 0,
        "short_rows": 0,
        "empty_rows": 0,
    }
    ins_sql = """
        INSERT OR IGNORE INTO dedup_corpus (
            text_hash, text, text_len, title, company, platform, year,
            publish_date, source, source_file, original_url, snapshot_url
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """

    t0 = time.time()
    for f in files:
        with f.open("r", encoding="utf-8", newline="") as fp:
            reader = csv.DictReader(fp)
            for row in reader:
                stats["read_rows"] += 1
                raw_text = row.get("text", "")
                text = clean_text(raw_text)
                if not text:
                    stats["empty_rows"] += 1
                    continue
                if len(text) < min_text_len:
                    stats["short_rows"] += 1
                    continue
                stats["kept_rows"] += 1

                h = text_hash(text)
                ts = (row.get("timestamp") or "").strip()
                year = ts[:4] if len(ts) >= 4 and ts[:4].isdigit() else ""
                original = (row.get("original") or "").strip()
                snapshot = (row.get("snapshot_url") or "").strip()

                platform = (row.get("seed_platform") or "").strip()
                if not platform:
                    platform = get_platform_from_host(urlsplit(original).netloc)

                before = conn.total_changes
                conn.execute(
                    ins_sql,
                    (
                        h,
                        text,
                        len(text),
                        "",  # wayback rows often don't have explicit title/company
                        "",
                        platform,
                        year,
                        ts,
                        "wayback",
                        str(f),
                        original,
                        snapshot,
                    ),
                )
                if conn.total_changes > before:
                    stats["inserted_rows"] += 1
                else:
                    stats["dup_rows"] += 1

                if stats["read_rows"] % commit_every == 0:
                    conn.commit()
                    dt = time.time() - t0
                    print(
                        f"[wayback] read={stats['read_rows']} kept={stats['kept_rows']} "
                        f"inserted={stats['inserted_rows']} dup={stats['dup_rows']} "
                        f"elapsed={dt:.1f}s"
                    )
    conn.commit()
    return stats


def query_counts(conn: sqlite3.Connection) -> Dict[str, object]:
    out: Dict[str, object] = {}
    out["unique_texts"] = conn.execute("SELECT COUNT(*) FROM dedup_corpus").fetchone()[0]
    out["platform_counts"] = dict(
        conn.execute(
            "SELECT COALESCE(platform,''), COUNT(*) FROM dedup_corpus GROUP BY platform ORDER BY COUNT(*) DESC"
        ).fetchall()
    )
    out["year_counts_top"] = conn.execute(
        "SELECT COALESCE(year,''), COUNT(*) FROM dedup_corpus GROUP BY year ORDER BY COUNT(*) DESC LIMIT 20"
    ).fetchall()
    out["source_counts"] = dict(
        conn.execute(
            "SELECT COALESCE(source,''), COUNT(*) FROM dedup_corpus GROUP BY source ORDER BY COUNT(*) DESC"
        ).fetchall()
    )
    return out


def main() -> None:
    args = parse_args()
    db_path = Path(args.db_path)
    summary_path = Path(args.summary_json)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(str(db_path))
    init_db(conn)

    run_summary: Dict[str, object] = {
        "db_path": str(db_path),
        "start_time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "params": {
            "all_in_one_csv": args.all_in_one_csv,
            "all_in_one_start_row": args.all_in_one_start_row,
            "all_in_one_max_rows": args.all_in_one_max_rows,
            "wayback_glob": args.wayback_glob,
            "min_text_len": args.min_text_len,
            "commit_every": args.commit_every,
            "skip_all_in_one": args.skip_all_in_one,
            "skip_wayback": args.skip_wayback,
        },
    }

    if not args.skip_all_in_one:
        csv_path = Path(args.all_in_one_csv)
        if not csv_path.exists():
            raise SystemExit(f"all_in_one CSV not found: {csv_path}")
        print(f"[all_in_one] ingest start: {csv_path}")
        run_summary["all_in_one"] = ingest_all_in_one(
            conn=conn,
            csv_path=csv_path,
            start_row=args.all_in_one_start_row,
            max_rows=args.all_in_one_max_rows,
            min_text_len=args.min_text_len,
            commit_every=args.commit_every,
        )

    if not args.skip_wayback:
        files = [Path(p) for p in sorted(glob.glob(args.wayback_glob))]
        print(f"[wayback] files matched: {len(files)}")
        run_summary["wayback"] = ingest_wayback(
            conn=conn,
            files=files,
            min_text_len=args.min_text_len,
            commit_every=args.commit_every,
        )

    counts = query_counts(conn)
    run_summary["counts"] = counts
    run_summary["end_time"] = time.strftime("%Y-%m-%d %H:%M:%S")
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(run_summary, f, ensure_ascii=False, indent=2)

    print("done")
    print(f"summary: {summary_path}")
    print(f"unique_texts: {counts['unique_texts']}")
    conn.close()


if __name__ == "__main__":
    main()
