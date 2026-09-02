#!/usr/bin/env bash
# ============================================================================
# run_all_models.sh - runs the full experiment ladder for every target model,
# one command per model, then generates the combined report at the end.
#
# Every model is listed explicitly below (not discovered via a file glob) so
# the complete set being run is visible directly in this one file.
#
# Defaults to the FULL train and test datasets (a real, reportable run).
# Pass numeric TRAIN_N/TEST_N arguments only for a quick smoke test:
#   bash run_all_models.sh 60 500
#
# Run this INSIDE tmux (see README.md's Local (NVIDIA GPU) section) so a
# dropped SSH connection doesn't kill a multi-hour, multi-model run.
# ============================================================================
set -euo pipefail
cd "$(dirname "$0")"   # repository root

TRAIN_N="${1:-full}"
TEST_N="${2:-full}"

bash run_full_experiment.sh "Qwen/Qwen2.5-Math-1.5B"             "$TRAIN_N" "$TEST_N"
bash run_full_experiment.sh "Qwen/Qwen2.5-Math-1.5B-Instruct"    "$TRAIN_N" "$TEST_N"
bash run_full_experiment.sh "Qwen/Qwen2.5-Math-7B"               "$TRAIN_N" "$TEST_N"
bash run_full_experiment.sh "Qwen/Qwen2.5-Math-7B-Instruct"      "$TRAIN_N" "$TEST_N"
bash run_full_experiment.sh "deepseek-ai/deepseek-math-7b-base"  "$TRAIN_N" "$TEST_N"
bash run_full_experiment.sh "deepseek-ai/deepseek-math-7b-rl"    "$TRAIN_N" "$TEST_N"
bash run_full_experiment.sh "AI-MO/NuminaMath-7B-CoT"            "$TRAIN_N" "$TEST_N"

echo ""
echo "=== All models complete. Generating combined report. ==="
python3 experiment_ledger.py --print --ledger outputs/experiment_ledger.jsonl
python3 generate_all_reports.py --ledger outputs/experiment_ledger.jsonl --out_dir outputs/report

echo ""
echo "Done. See outputs/report/ for tables, figures, and diagrams, and"
echo "outputs/experiment_ledger.jsonl for the full machine-readable record."
