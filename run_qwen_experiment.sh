#!/usr/bin/env bash
# ============================================================================
# run_qwen_experiment.sh - SMART ladder for Qwen2.5-Math-7B-Instruct
#
#   bash run_qwen_experiment.sh                 # full data
#   bash run_qwen_experiment.sh 200 500         # TRAIN_N=200, TEST_N=500
#
# Re-running continues from where it stopped. To redo one step:
#   FORCE_RERUN=sft_skill bash run_qwen_experiment.sh
#
# Requires: export HF_TOKEN=hf_xxxxxxxx
# ============================================================================
set -Eeuo pipefail

export MODEL="Qwen/Qwen2.5-Math-7B-Instruct"
export SMART_HF_SFT_REPO="HusnainAmjad/Qwen_2.5_Math_7b_SFT"
export SMART_HF_REPLAY_REPO="HusnainAmjad/Qwen_2.5_Math_7b_Replay_SFT"
export SMART_HF_GRPO_REPO="HusnainAmjad/Qwen_2.5_Math_7b_GRPO"
export SMART_HF_OUTPUTS_REPO="HusnainAmjad/Qwen_2.5_Math_7b_Outputs"

export TRAIN_N="${1:-full}"
export TEST_N="${2:-full}"

source "$(dirname "$0")/run_experiment_core.sh"
