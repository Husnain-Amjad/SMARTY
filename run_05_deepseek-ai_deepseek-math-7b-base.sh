#!/usr/bin/env bash
# Runs the full SMART experiment ladder (baseline -> SFT -> 3 augmentation
# variants -> 4 GRPO variants) for: deepseek-ai/deepseek-math-7b-base
#
# Defaults to the FULL train and test datasets (a real, reportable run).
# Pass numeric TRAIN_N/TEST_N arguments only for a quick smoke test:
#   bash run_05_deepseek-ai_deepseek-math-7b-base.sh 60 500
#
# Run from the repository root:
#   bash run_05_deepseek-ai_deepseek-math-7b-base.sh
set -euo pipefail
cd "$(dirname "$0")"   # repository root
bash run_full_experiment.sh "deepseek-ai/deepseek-math-7b-base" "${1:-full}" "${2:-full}"
