#!/bin/zsh
set -euo pipefail

CACHE_PATH="${1:-data/archive_recovery/unified/wayback_2019_deepseek_rowwise_cache_full.jsonl}"
KEYWORD="${2:-enrich_wayback_2019_deepseek_rowwise.py}"
INTERVAL="${3:-30}"

last_ok=-1
last_fail=-1

echo "[$(date '+%Y-%m-%d %H:%M:%S %Z')] monitor_start keyword=${KEYWORD} cache=${CACHE_PATH} interval=${INTERVAL}s"

while true; do
  proc_lines="$(ps aux | rg "${KEYWORD}" | rg -v 'rg ' || true)"
  if [[ -n "${proc_lines}" ]]; then
    proc_count="$(printf '%s\n' "${proc_lines}" | sed '/^$/d' | wc -l | tr -d ' ')"
    pids="$(printf '%s\n' "${proc_lines}" | awk '{print $2}' | paste -sd, -)"
  else
    proc_count="0"
    pids=""
  fi

  stats="$(python3 - <<'PY' "${CACHE_PATH}"
import json, sys
from pathlib import Path
p = Path(sys.argv[1])
ok = fail = total = 0
if p.exists():
    with p.open('r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            total += 1
            try:
                o = json.loads(line)
            except Exception:
                continue
            if o.get('ok') is True:
                ok += 1
            else:
                fail += 1
print(f"{ok} {fail} {total}")
PY
)"

  ok="$(echo "${stats}" | awk '{print $1}')"
  fail="$(echo "${stats}" | awk '{print $2}')"
  total="$(echo "${stats}" | awk '{print $3}')"

  if [[ "${last_ok}" -lt 0 ]]; then
    delta_ok=0
    delta_fail=0
  else
    delta_ok=$((ok - last_ok))
    delta_fail=$((fail - last_fail))
  fi
  last_ok="${ok}"
  last_fail="${fail}"

  echo "[$(date '+%Y-%m-%d %H:%M:%S %Z')] proc_count=${proc_count} pids=${pids} ok=${ok} fail=${fail} total=${total} delta_ok=${delta_ok} delta_fail=${delta_fail}"
  sleep "${INTERVAL}"
done

