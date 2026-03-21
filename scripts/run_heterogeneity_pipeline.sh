#!/bin/bash
# run_heterogeneity_pipeline.sh
# Runs the full heterogeneity analysis pipeline after create_sorted_results.py finishes.
# Steps:
#   1. Wait for create_sorted_results.py output to stabilise (polls final_results_sample_sorted.csv)
#   2. prepare_joint_mapping_industry20.py  → master_joint_industry20_analysis.csv
#   3. compute_ai_exposure.py              → master_with_ai_exposure.csv
#   4. economic_grouping_analysis.py       → yearly_economic_grouping_entropy.csv
#   5. did_regression.py                   → did_results/ (with token_count, once available)
# Usage:
#   bash scripts/run_heterogeneity_pipeline.sh [--skip-wait]

set -e

PYTHON_MCM=/Users/yu/code/miniconda3/envs/mcm2026/bin/python
PYTHON_ZHI=/Users/yu/code/miniconda3/envs/zhilian/bin/python
BASE=/Users/yu/code/code2601/TY
FINAL_SORTED="$BASE/data/Heterogeneity/final_results_sample_sorted.csv"
LOG="$BASE/data/Heterogeneity/pipeline_run.log"

echo "[$(date)] Pipeline started" | tee -a "$LOG"

# ── Step 1: Wait for create_sorted_results.py to finish ──────────────────────
if [[ "$1" != "--skip-wait" ]]; then
    echo "[$(date)] Waiting for create_sorted_results.py to finish..." | tee -a "$LOG"
    prev=0
    while true; do
        # If no python process running create_sorted_results.py, assume done
        if ! pgrep -f "create_sorted_results.py" > /dev/null; then
            echo "[$(date)] create_sorted_results.py process not found — assuming done" | tee -a "$LOG"
            break
        fi
        curr=$(wc -l < "$FINAL_SORTED" 2>/dev/null || echo 0)
        echo "[$(date)]   final_results rows: $curr" | tee -a "$LOG"
        if [[ "$curr" == "$prev" && "$curr" -gt 1000000 ]]; then
            echo "[$(date)] Row count stable — file complete" | tee -a "$LOG"
            break
        fi
        prev=$curr
        sleep 60
    done
fi

# Verify the new file has token_count column
HEADER=$(head -1 "$FINAL_SORTED")
if [[ "$HEADER" != *"token_count"* ]]; then
    echo "[$(date)] ERROR: final_results_sample_sorted.csv missing token_count column" | tee -a "$LOG"
    echo "Header: $HEADER"
    exit 1
fi
echo "[$(date)] Verified token_count in final_results_sample_sorted.csv" | tee -a "$LOG"

# ── Step 2: prepare_joint_mapping_industry20.py ───────────────────────────────
echo "[$(date)] Step 2: Running prepare_joint_mapping_industry20.py..." | tee -a "$LOG"
cd "$BASE"
$PYTHON_MCM src/analysis/prepare_joint_mapping_industry20.py 2>&1 | tee -a "$LOG"
echo "[$(date)] Step 2 done." | tee -a "$LOG"

# ── Step 3: compute_ai_exposure.py ────────────────────────────────────────────
echo "[$(date)] Step 3: Running compute_ai_exposure.py..." | tee -a "$LOG"
$PYTHON_ZHI src/analysis/compute_ai_exposure.py 2>&1 | tee -a "$LOG"
echo "[$(date)] Step 3 done." | tee -a "$LOG"

# ── Step 4: economic_grouping_analysis.py ────────────────────────────────────
echo "[$(date)] Step 4: Running economic_grouping_analysis.py..." | tee -a "$LOG"
$PYTHON_ZHI src/analysis/economic_grouping_analysis.py 2>&1 | tee -a "$LOG"
echo "[$(date)] Step 4 done." | tee -a "$LOG"

# ── Step 5: did_regression.py ─────────────────────────────────────────────────
echo "[$(date)] Step 5: Running did_regression.py..." | tee -a "$LOG"
$PYTHON_MCM src/analysis/did_regression.py 2>&1 | tee -a "$LOG"
echo "[$(date)] Step 5 done." | tee -a "$LOG"

echo "[$(date)] Pipeline complete." | tee -a "$LOG"
