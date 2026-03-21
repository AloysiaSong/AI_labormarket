#!/usr/bin/env python3
import argparse
import csv
import importlib.util
import json
from pathlib import Path


def load_module(mod_path: Path, mod_name: str):
    spec = importlib.util.spec_from_file_location(mod_name, str(mod_path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load module: {mod_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_latest_cache(cache_jsonl: Path):
    latest = {}
    if not cache_jsonl.exists():
        return latest
    with cache_jsonl.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            try:
                rid = int(obj.get("row_id", 0) or 0)
            except Exception:
                continue
            if rid <= 0:
                continue
            latest[rid] = obj
    return latest


def main():
    parser = argparse.ArgumentParser(
        description="Build hybrid enriched output: use successful DeepSeek cache first, then local rules fallback."
    )
    parser.add_argument(
        "--input-csv",
        default="data/archive_recovery/unified/wayback_2019_allinone_schema.csv",
    )
    parser.add_argument(
        "--cache-jsonl",
        default="data/archive_recovery/unified/wayback_2019_deepseek_rowwise_cache_full.jsonl",
    )
    parser.add_argument(
        "--output-csv",
        default="data/archive_recovery/unified/wayback_2019_allinone_schema_hybrid_cache_local.csv",
    )
    parser.add_argument(
        "--summary-json",
        default="data/archive_recovery/unified/wayback_2019_allinone_schema_hybrid_cache_local_summary.json",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    rowwise_mod = load_module(root / "enrich_wayback_2019_deepseek_rowwise.py", "rowwise_mod")
    local_mod = load_module(root / "enrich_split_wayback_2019_allinone.py", "local_mod")

    input_csv = Path(args.input_csv)
    cache_jsonl = Path(args.cache_jsonl)
    output_csv = Path(args.output_csv)
    summary_json = Path(args.summary_json)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    summary_json.parent.mkdir(parents=True, exist_ok=True)

    cache_map = load_latest_cache(cache_jsonl)

    rows_in = 0
    rows_out = 0
    rows_from_cache_ok = 0
    rows_from_cache_ok_items = 0
    rows_from_cache_ok_empty = 0
    rows_from_cache_fail_or_miss = 0
    rows_with_local_title = 0
    rows_expanded_net = 0
    extracted_company = 0
    extracted_city = 0
    extracted_region = 0
    extracted_location = 0

    with input_csv.open("r", encoding="utf-8-sig", newline="") as fin:
        reader = csv.DictReader(fin)
        fields = reader.fieldnames or []
        with output_csv.open("w", encoding="utf-8-sig", newline="") as fout:
            writer = csv.DictWriter(fout, fieldnames=fields)
            writer.writeheader()

            for row_id, row in enumerate(reader, start=1):
                rows_in += 1
                text = row.get("职位描述", "") or ""

                base = dict(row)
                company = row.get("企业名称", "") or local_mod.extract_company(text)
                location_raw = local_mod.extract_location_raw(text)
                city, region = local_mod.parse_city_region(location_raw, text)
                if company and not (base.get("企业名称") or "").strip():
                    base["企业名称"] = company
                    extracted_company += 1
                if location_raw and not (base.get("工作地点") or "").strip():
                    base["工作地点"] = location_raw
                    extracted_location += 1
                if city and not (base.get("工作城市") or "").strip():
                    base["工作城市"] = city
                    extracted_city += 1
                if region and not (base.get("工作区域") or "").strip():
                    base["工作区域"] = region
                    extracted_region += 1
                if location_raw and not (base.get("公司地点") or "").strip():
                    base["公司地点"] = location_raw

                cache_obj = cache_map.get(row_id)
                items = []
                if cache_obj and bool(cache_obj.get("ok")):
                    rows_from_cache_ok += 1
                    raw_items = cache_obj.get("items") or []
                    if raw_items:
                        rows_from_cache_ok_items += 1
                    else:
                        rows_from_cache_ok_empty += 1
                    items = rowwise_mod.enrich_items_with_inferred_titles(base, raw_items)
                else:
                    rows_from_cache_fail_or_miss += 1
                    titles = local_mod.extract_job_titles(text, row.get("招聘岗位", "") or "")
                    if titles:
                        rows_with_local_title += 1
                        items = [
                            {
                                "企业名称": "",
                                "招聘岗位": t,
                                "工作城市": "",
                                "工作区域": "",
                                "工作地点": "",
                            }
                            for t in titles
                        ]

                if items:
                    wrote = 0
                    for it in items:
                        out = dict(base)
                        for k in rowwise_mod.TARGET_KEYS:
                            if not (out.get(k) or "").strip():
                                out[k] = it.get(k, "")
                        if (out.get("工作地点") or "").strip() and not (out.get("公司地点") or "").strip():
                            out["公司地点"] = out["工作地点"]
                        writer.writerow(out)
                        rows_out += 1
                        wrote += 1
                    if wrote > 1:
                        rows_expanded_net += (wrote - 1)
                else:
                    writer.writerow(base)
                    rows_out += 1

    summary = {
        "input_csv": str(input_csv),
        "cache_jsonl": str(cache_jsonl),
        "output_csv": str(output_csv),
        "rows_in": rows_in,
        "rows_out": rows_out,
        "rows_expanded_net": rows_expanded_net,
        "rows_from_cache_ok": rows_from_cache_ok,
        "rows_from_cache_ok_items": rows_from_cache_ok_items,
        "rows_from_cache_ok_empty": rows_from_cache_ok_empty,
        "rows_from_cache_fail_or_miss": rows_from_cache_fail_or_miss,
        "rows_with_local_title": rows_with_local_title,
        "extracted_company_rows": extracted_company,
        "extracted_city_rows": extracted_city,
        "extracted_region_rows": extracted_region,
        "extracted_location_rows": extracted_location,
    }
    with summary_json.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("done")
    print(f"output: {output_csv}")
    print(f"summary: {summary_json}")
    print(f"rows_in: {rows_in}")
    print(f"rows_out: {rows_out}")


if __name__ == "__main__":
    main()

