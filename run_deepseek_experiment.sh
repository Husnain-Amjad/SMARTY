#!/usr/bin/env bash
# ============================================================================
# run_deepseek_experiment.sh - SMART ladder for deepseek-math-7b-rl
#
#   bash run_deepseek_experiment.sh             # full data
#   bash run_deepseek_experiment.sh 200 500     # TRAIN_N=200, TEST_N=500
#
# Re-running continues from where it stopped. To redo one step:
#   FORCE_RERUN=sft_skill bash run_deepseek_experiment.sh
#
# Requires: export HF_TOKEN=hf_xxxxxxxx
#
# Identical protocol to run_qwen_experiment.sh - both source the same
# run_experiment_core.sh, so the two models cannot drift apart.
# ============================================================================
set -Eeuo pipefail

export MODEL="deepseek-ai/deepseek-math-7b-rl"
export SMART_HF_SFT_REPO="HusnainAmjad/Deepseek_Math_7b_r1_SFT"
export SMART_HF_REPLAY_REPO="HusnainAmjad/Deepseek_Math_7b_r1_Replay_SFT"
export SMART_HF_GRPO_REPO="HusnainAmjad/Deepseek_Math_7b_r1_GRPO"
export SMART_HF_OUTPUTS_REPO="HusnainAmjad/Deepseek_Math_7b_r1_Outputs"

export TRAIN_N="${1:-full}"
export TEST_N="${2:-full}"

source "$(dirname "$0")/run_experiment_core.sh"
