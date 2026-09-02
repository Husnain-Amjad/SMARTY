#!/usr/bin/env bash
# Runs the full SMART experiment ladder (baseline -> SFT -> 3 augmentation
# variants -> 4 GRPO variants) for: Qwen/Qwen2.5-Math-1.5B
#
# Defaults to the FULL train and test datasets (a real, reportable run).
# Pass numeric TRAIN_N/TEST_N arguments only for a quick smoke test:
#   bash run_01_Qwen_Qwen2_5-Math-1_5B.sh 60 500
#
# Run from the repository root:
#   bash run_01_Qwen_Qwen2_5-Math-1_5B.sh
set -euo pipefail
cd "$(dirname "$0")"   # repository root
bash run_full_experiment.sh "Qwen/Qwen2.5-Math-1.5B" "${1:-full}" "${2:-full}"
