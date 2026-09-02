#!/usr/bin/env bash
# Runs the full SMART experiment ladder (baseline -> SFT -> 3 augmentation
# variants -> 4 GRPO variants) for: AI-MO/NuminaMath-7B-CoT
#
# Defaults to the FULL train and test datasets (a real, reportable run).
# Pass numeric TRAIN_N/TEST_N arguments only for a quick smoke test:
#   bash run_07_AI-MO_NuminaMath-7B-CoT.sh 60 500
#
# Run from the repository root:
#   bash run_07_AI-MO_NuminaMath-7B-CoT.sh
set -euo pipefail
cd "$(dirname "$0")"   # repository root
bash run_full_experiment.sh "AI-MO/NuminaMath-7B-CoT" "${1:-full}" "${2:-full}"
