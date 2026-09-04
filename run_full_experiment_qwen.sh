#!/usr/bin/env bash

# ============================================================
# SMARTY - Qwen 2.5-Math-7B-Instruct Full Experiment Pipeline
#
# Model:
#   Qwen/Qwen2.5-Math-7B-Instruct
#
# Pipeline:
#   Stage 0: Baseline
#   Stage 1: Skill-aware SFT
#   Stage 2: Diagnosis + Augmentation
#   Stage 3: Progressive Replay SFT + GRPO
#
# Progressive storage strategy:
#   SFT -> Eval -> GRPO -> Eval -> HF Push -> Cleanup
#
# This avoids keeping all SFT + GRPO checkpoints locally.
#
# Usage:
#   bash run_full_experiment_qwen.sh
#
# Optional:
#   bash run_full_experiment_qwen.sh 20 20
#
# Environment:
#   HF_TOKEN must be set unless SMART_HF_DISABLE=1
#
# ============================================================

set -Eeuo pipefail

trap '
    echo ""
    echo "============================================================"
    echo "[ERROR] Pipeline failed at line $LINENO"
    echo "[ERROR] Command: $BASH_COMMAND"
    echo "============================================================"
    exit 1
' ERR


# ============================================================
# CONFIGURATION
# ============================================================

MODEL="${1:-Qwen/Qwen2.5-Math-7B-Instruct}"
TRAIN_N="${2:-full}"
TEST_N="${3:-full}"

MODEL_SLUG="Qwen_2.5_Math_7b"

OUT="outputs/${MODEL_SLUG}"
CKPT="ckpts/${MODEL_SLUG}"

LEDGER="outputs/experiment_ledger.jsonl"


# ------------------------------------------------------------
# Hugging Face repositories
# ------------------------------------------------------------

SMART_HF_SFT_REPO="${SMART_HF_SFT_REPO:-HusnainAmjad/Qwen_2.5_Math_7b_SFT}"
SMART_HF_REPLAY_REPO="${SMART_HF_REPLAY_REPO:-HusnainAmjad/Qwen_2.5_Math_7b_Replay_SFT}"
SMART_HF_GRPO_REPO="${SMART_HF_GRPO_REPO:-HusnainAmjad/Qwen_2.5_Math_7b_GRPO}"
SMART_HF_OUTPUTS_REPO="${SMART_HF_OUTPUTS_REPO:-HusnainAmjad/Qwen_2.5_Math_7b_Outputs}"


# ------------------------------------------------------------
# HF cleanup behaviour
#
# Normal:
#   HF_TOKEN must exist.
#   Successful uploads permit local cleanup.
#
# Disable:
#   SMART_HF_DISABLE=1
#   Nothing is deleted because HF is not being used as archive.
# ------------------------------------------------------------

SMART_HF_DISABLE="${SMART_HF_DISABLE:-0}"


# ------------------------------------------------------------
# Training configuration
# ------------------------------------------------------------

LORA_R="${LORA_R:-16}"
LORA_ALPHA="${LORA_ALPHA:-32}"

EPOCHS="${EPOCHS:-4}"

SAVE_EVERY_EPOCHS="${SAVE_EVERY_EPOCHS:-1}"

PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE:-32}"
GRAD_ACCUM="${GRAD_ACCUM:-1}"

REPLAY_RATIO="${REPLAY_RATIO:-0.7}"


# ------------------------------------------------------------
# GRPO configuration
# ------------------------------------------------------------

GRPO_W_CORRECTNESS="${GRPO_W_CORRECTNESS:-1.0}"
GRPO_W_FORMAT="${GRPO_W_FORMAT:-0.2}"
GRPO_W_PERSISTENCE="${GRPO_W_PERSISTENCE:-0.15}"
GRPO_W_CHAIN_STABILITY="${GRPO_W_CHAIN_STABILITY:-0.25}"


# ------------------------------------------------------------
# Evaluation
# ------------------------------------------------------------

MAX_TOKENS="${MAX_TOKENS:-1024}"

# vLLM is used by evaluation.
USE_VLLM="${USE_VLLM:-1}"


# ============================================================
# VALIDATION
# ============================================================

echo ""
echo "============================================================"
echo " SMARTY - QWEN FULL EXPERIMENT"
echo "============================================================"
echo "Model              : ${MODEL}"
echo "Train size         : ${TRAIN_N}"
echo "Test size          : ${TEST_N}"
echo "Output directory   : ${OUT}"
echo "Checkpoint dir     : ${CKPT}"
echo ""
echo "SFT HF repo        : ${SMART_HF_SFT_REPO}"
echo "Replay HF repo     : ${SMART_HF_REPLAY_REPO}"
echo "GRPO HF repo       : ${SMART_HF_GRPO_REPO}"
echo "Outputs HF repo    : ${SMART_HF_OUTPUTS_REPO}"
echo "============================================================"
echo ""


if [ "${SMART_HF_DISABLE}" != "1" ]; then

    if [ -z "${HF_TOKEN:-}" ]; then
        echo "[HF] ERROR: HF_TOKEN is not set."
        echo ""
        echo "HF upload is required for progressive cleanup."
        echo ""
        echo "Set it before running:"
        echo '  export HF_TOKEN="hf_..."'
        echo ""
        echo "Or explicitly disable HF cleanup:"
        echo '  export SMART_HF_DISABLE=1'
        echo ""
        exit 1
    fi

    echo "[HF] Token: set"

else

    echo "[HF] Upload/cleanup: DISABLED"
    echo "[HF] Local checkpoints will NOT be deleted."

fi


# ============================================================
# DIRECTORY SETUP
# ============================================================

mkdir -p "${OUT}"
mkdir -p "${CKPT}"
mkdir -p "$(dirname "${LEDGER}")"


# ============================================================
# DISK MONITOR
# ============================================================

check_disk() {

    echo ""
    echo "------------------------------------------------------------"
    echo "[DISK] Current usage"
    echo "------------------------------------------------------------"

    df -h .

    echo ""
    echo "[DISK] Checkpoints:"
    du -sh "${CKPT}" 2>/dev/null || true

    echo ""
    echo "[DISK] Outputs:"
    du -sh "${OUT}" 2>/dev/null || true

    echo ""
}


# ============================================================
# HF UPLOAD STATUS
# ============================================================

declare -A PUSH_OK


# ============================================================
# HF MODEL PUSH
# ============================================================

push_model_hf() {

    local MODEL_PATH="$1"
    local VARIANT="$2"

    echo ""
    echo "============================================================"
    echo "[HF] Uploading model"
    echo "============================================================"
    echo "Variant : ${VARIANT}"
    echo "Path    : ${MODEL_PATH}"
    echo "============================================================"

    if [ "${SMART_HF_DISABLE}" = "1" ]; then

        echo "[HF] Disabled."
        PUSH_OK["${VARIANT}"]="no"

        return 0
    fi


    if python hf_sync.py \
        --push-model "${MODEL_PATH}" \
        --variant "${VARIANT}"
    then

        echo "[HF] Upload SUCCESS: ${VARIANT}"

        PUSH_OK["${VARIANT}"]="yes"

    else

        echo "[HF] Upload FAILED: ${VARIANT}"

        PUSH_OK["${VARIANT}"]="no"

        return 1
    fi
}


# ============================================================
# HF OUTPUT PUSH
# ============================================================

push_outputs_hf() {

    if [ "${SMART_HF_DISABLE}" = "1" ]; then
        echo "[HF] Output upload disabled."
        return 0
    fi

    echo ""
    echo "============================================================"
    echo "[HF] Uploading experiment outputs"
    echo "============================================================"

    python hf_sync.py \
        --push-outputs "${OUT}"

    echo "[HF] Output upload completed."
}


# ============================================================
# PROGRESSIVE CLEANUP
# ============================================================

cleanup_variant_pair() {

    local BASE_NAME="$1"
    local GRPO_VARIANT="$2"

    local SFT_DIR="${CKPT}/${BASE_NAME}"
    local SFT_MERGED="${CKPT}/${BASE_NAME}_merged"

    local GRPO_DIR="${CKPT}/grpo_${GRPO_VARIANT}"
    local GRPO_MERGED="${GRPO_DIR}_merged"


    echo ""
    echo "============================================================"
    echo "[CLEANUP] Checking ${BASE_NAME}"
    echo "============================================================"


    if [ "${SMART_HF_DISABLE}" = "1" ]; then

        echo "[CLEANUP] HF disabled."
        echo "[CLEANUP] Keeping local files."

        return 0
    fi


    # --------------------------------------------------------
    # IMPORTANT:
    # Delete only if BOTH SFT and GRPO were uploaded.
    # --------------------------------------------------------

    if [ "${PUSH_OK[${BASE_NAME}]:-no}" = "yes" ] && \
       [ "${PUSH_OK[${GRPO_VARIANT}]:-no}" = "yes" ]; then

        echo "[CLEANUP] Both uploads confirmed."
        echo "[CLEANUP] Removing local SFT + GRPO checkpoints."


        if [ -d "${SFT_DIR}" ]; then
            echo "[CLEANUP] Removing ${SFT_DIR}"
            rm -rf "${SFT_DIR}"
        fi


        if [ -d "${SFT_MERGED}" ]; then
            echo "[CLEANUP] Removing ${SFT_MERGED}"
            rm -rf "${SFT_MERGED}"
        fi


        if [ -d "${GRPO_DIR}" ]; then
            echo "[CLEANUP] Removing ${GRPO_DIR}"
            rm -rf "${GRPO_DIR}"
        fi


        if [ -d "${GRPO_MERGED}" ]; then
            echo "[CLEANUP] Removing ${GRPO_MERGED}"
            rm -rf "${GRPO_MERGED}"
        fi


        echo "[CLEANUP] Completed."

    else

        echo "[CLEANUP] WARNING:"
        echo "[CLEANUP] One or both HF uploads were NOT confirmed."
        echo "[CLEANUP] Keeping local checkpoints for safety."

    fi


    check_disk
}


# ============================================================
# VLLM EVALUATION
# ============================================================

run_vllm_eval() {

    local MODEL_PATH="$1"
    local SPLIT="$2"
    local OUT_FILE="$3"

    echo ""
    echo "============================================================"
    echo "[EVAL] vLLM evaluation"
    echo "============================================================"
    echo "Model : ${MODEL_PATH}"
    echo "Split : ${SPLIT}"
    echo "Output: ${OUT_FILE}"
    echo "============================================================"


    if [ "${USE_VLLM}" = "1" ]; then

        python run_eval.py \
            --model "${MODEL_PATH}" \
            --split "${SPLIT}" \
            --out "${OUT_FILE}" \
            --use_vllm \
            --max_tokens "${MAX_TOKENS}"

    else

        python run_eval.py \
            --model "${MODEL_PATH}" \
            --split "${SPLIT}" \
            --out "${OUT_FILE}" \
            --max_tokens "${MAX_TOKENS}"

    fi
}


# ============================================================
# SCORE EVALUATION
# ============================================================

run_score() {

    local PREDICTIONS="$1"

    echo ""
    echo "============================================================"
    echo "[SCORE]"
    echo "============================================================"

    python evaluator.py \
        --predictions "${PREDICTIONS}"

}


# ============================================================
# LEDGER LOG
# ============================================================

run_log() {

    local RUN_ID="$1"
    local MODEL_NAME="$2"
    local STAGE="$3"
    local VARIANT="$4"
    local PREDICTIONS="$5"

    echo ""
    echo "[LEDGER] Logging ${RUN_ID}"

    python experiment_ledger.py \
        --run_id "${RUN_ID}" \
        --model "${MODEL_NAME}" \
        --stage "${STAGE}" \
        --variant "${VARIANT}" \
        --predictions "${PREDICTIONS}"

}


# ============================================================
# SFT TRAINING
# ============================================================

run_sft_variant() {

    local VARIANT="$1"

    local DATA_ARG=""
    local EXTRA_ARG=""

    case "${VARIANT}" in

        sft_skill)

            DATA_ARG="${OUT}/sft_data.jsonl"

            ;;

        simple_replay)

            DATA_ARG="${OUT}/sft_data.jsonl"
            EXTRA_ARG="--extra_data ${OUT}/sft_data.jsonl"

            ;;

        sem_replay)

            DATA_ARG="${OUT}/sft_data.jsonl"
            EXTRA_ARG="--extra_data ${OUT}/semantic_aug.jsonl"

            ;;

        num_replay)

            DATA_ARG="${OUT}/sft_data.jsonl"
            EXTRA_ARG="--extra_data ${OUT}/numeric_aug.jsonl"

            ;;

        both_replay)

            DATA_ARG="${OUT}/sft_data.jsonl"
            EXTRA_ARG="--extra_data ${OUT}/semantic_aug.jsonl ${OUT}/numeric_aug.jsonl"

            ;;

        *)

            echo "[SFT] ERROR: Unknown variant ${VARIANT}"
            exit 1

            ;;

    esac


    echo ""
    echo "============================================================"
    echo "[SFT] ${VARIANT}"
    echo "============================================================"


    python sft_train.py \
        --model "${MODEL}" \
        --train_file "${DATA_ARG}" \
        --output_dir "${CKPT}/${VARIANT}" \
        --lora_r "${LORA_R}" \
        --lora_alpha "${LORA_ALPHA}" \
        --num_train_epochs "${EPOCHS}" \
        --save_every_epochs "${SAVE_EVERY_EPOCHS}" \
        --per_device_train_batch_size "${PER_DEVICE_BATCH_SIZE}" \
        --gradient_accumulation_steps "${GRAD_ACCUM}" \
        --attn_implementation flash_attention_2 \
        ${EXTRA_ARG}


    echo ""
    echo "[SFT] Training completed: ${VARIANT}"


    echo ""
    echo "[SFT] Merging LoRA adapter..."

    python sft_train.py \
        --merge_only \
        --adapter_path "${CKPT}/${VARIANT}" \
        --output_dir "${CKPT}/${VARIANT}_merged"


    echo ""
    echo "[SFT] Merge completed: ${VARIANT}_merged"


    check_disk
}


# ============================================================
# STAGE 0 - BASELINE
# ============================================================

echo ""
echo "============================================================"
echo " STAGE 0 - BASELINE"
echo "============================================================"

BASELINE_PRED="${OUT}/baseline_predictions.jsonl"

run_vllm_eval \
    "${MODEL}" \
    "test" \
    "${BASELINE_PRED}"

run_score "${BASELINE_PRED}"

run_log \
    "qwen_baseline" \
    "${MODEL}" \
    "baseline" \
    "baseline" \
    "${BASELINE_PRED}"


# ============================================================
# STAGE 1 - SKILL-AWARE SFT
# ============================================================

echo ""
echo "============================================================"
echo " STAGE 1 - SKILL-AWARE SFT"
echo "============================================================"


# ------------------------------------------------------------
# Train Skill SFT
# ------------------------------------------------------------

run_sft_variant "sft_skill"


# ------------------------------------------------------------
# Evaluate Skill SFT
# ------------------------------------------------------------

SFT_SKILL_PRED="${OUT}/sft_skill_predictions.jsonl"

run_vllm_eval \
    "${CKPT}/sft_skill_merged" \
    "test" \
    "${SFT_SKILL_PRED}"

run_score "${SFT_SKILL_PRED}"

run_log \
    "qwen_sft_skill" \
    "${MODEL}" \
    "sft" \
    "sft_skill" \
    "${SFT_SKILL_PRED}"


# ============================================================
# STAGE 2 - DIAGNOSIS + AUGMENTATION
# ============================================================

echo ""
echo "============================================================"
echo " STAGE 2 - DIAGNOSIS + AUGMENTATION"
echo "============================================================"


# ------------------------------------------------------------
# Diagnosis
# ------------------------------------------------------------

echo ""
echo "[DIAGNOSIS] Running skill diagnosis..."

python run_augmentation.py \
    --mode diagnose \
    --model "${CKPT}/sft_skill_merged" \
    --train_file "${OUT}/sft_data.jsonl" \
    --test_file "${OUT}/sft_skill_predictions.jsonl" \
    --out_dir "${OUT}"


# ------------------------------------------------------------
# Semantic augmentation
# ------------------------------------------------------------

echo ""
echo "[AUGMENTATION] Semantic..."

python run_augmentation.py \
    --mode semantic \
    --model "${CKPT}/sft_skill_merged" \
    --out_dir "${OUT}" \
    --output "${OUT}/semantic_aug.jsonl"


# ------------------------------------------------------------
# Numeric augmentation
# ------------------------------------------------------------

echo ""
echo "[AUGMENTATION] Numeric..."

python run_augmentation.py \
    --mode numeric \
    --model "${CKPT}/sft_skill_merged" \
    --out_dir "${OUT}" \
    --output "${OUT}/numeric_aug.jsonl"


check_disk


# ============================================================
# STAGE 3 - PROGRESSIVE REPLAY + GRPO
# ============================================================

echo ""
echo "============================================================"
echo " STAGE 3 - PROGRESSIVE REPLAY + GRPO"
echo "============================================================"


# ============================================================
# VARIANT 1 - SIMPLE SKILL
# ============================================================

echo ""
echo "############################################################"
echo "# VARIANT: SIMPLE SKILL"
echo "############################################################"


# ------------------------------------------------------------
# GRPO directly on Skill SFT
# ------------------------------------------------------------

GRPO_VARIANT="sft_skill"
BASE_VARIANT="sft_skill"

GRPO_DIR="${CKPT}/grpo_${GRPO_VARIANT}"

echo ""
echo "[GRPO] Training ${GRPO_VARIANT}"


python grpo_train.py \
    --model "${CKPT}/${BASE_VARIANT}_merged" \
    --output_dir "${GRPO_DIR}" \
    --w_correctness "${GRPO_W_CORRECTNESS}" \
    --w_format "${GRPO_W_FORMAT}" \
    --w_persistence "${GRPO_W_PERSISTENCE}" \
    --w_chain_stability "${GRPO_W_CHAIN_STABILITY}" \
    --attn_implementation flash_attention_2


echo ""
echo "[GRPO] Merging ${GRPO_VARIANT}"


python grpo_train.py \
    --merge_only \
    --adapter_path "${GRPO_DIR}" \
    --output_dir "${GRPO_DIR}_merged"


# ------------------------------------------------------------
# Evaluate GRPO
# ------------------------------------------------------------

GRPO_PRED="${OUT}/grpo_${GRPO_VARIANT}_predictions.jsonl"

run_vllm_eval \
    "${GRPO_DIR}_merged" \
    "test" \
    "${GRPO_PRED}"

run_score "${GRPO_PRED}"

run_log \
    "qwen_grpo_${GRPO_VARIANT}" \
    "${MODEL}" \
    "grpo" \
    "${GRPO_VARIANT}" \
    "${GRPO_PRED}"


# ------------------------------------------------------------
# Upload SFT + GRPO
# ------------------------------------------------------------

push_model_hf \
    "${CKPT}/${BASE_VARIANT}_merged" \
    "${BASE_VARIANT}"

push_model_hf \
    "${GRPO_DIR}_merged" \
    "${GRPO_VARIANT}"


# ------------------------------------------------------------
# IMPORTANT:
# Do NOT delete sft_skill yet until augmentation is complete.
# Now augmentation IS complete, so cleanup is safe.
# ------------------------------------------------------------

cleanup_variant_pair \
    "${BASE_VARIANT}" \
    "${GRPO_VARIANT}"


# ============================================================
# VARIANT 2 - SIMPLE REPLAY
# ============================================================

echo ""
echo "############################################################"
echo "# VARIANT: SIMPLE REPLAY"
echo "############################################################"


BASE_VARIANT="simple_replay"
GRPO_VARIANT="simple_replay"


run_sft_variant "${BASE_VARIANT}"


# ------------------------------------------------------------
# SFT evaluation
# ------------------------------------------------------------

SFT_PRED="${OUT}/${BASE_VARIANT}_predictions.jsonl"

run_vllm_eval \
    "${CKPT}/${BASE_VARIANT}_merged" \
    "test" \
    "${SFT_PRED}"

run_score "${SFT_PRED}"

run_log \
    "qwen_${BASE_VARIANT}" \
    "${MODEL}" \
    "replay_sft" \
    "${BASE_VARIANT}" \
    "${SFT_PRED}"


# ------------------------------------------------------------
# GRPO
# ------------------------------------------------------------

GRPO_DIR="${CKPT}/grpo_${GRPO_VARIANT}"

python grpo_train.py \
    --model "${CKPT}/${BASE_VARIANT}_merged" \
    --output_dir "${GRPO_DIR}" \
    --w_correctness "${GRPO_W_CORRECTNESS}" \
    --w_format "${GRPO_W_FORMAT}" \
    --w_persistence "${GRPO_W_PERSISTENCE}" \
    --w_chain_stability "${GRPO_W_CHAIN_STABILITY}" \
    --attn_implementation flash_attention_2


python grpo_train.py \
    --merge_only \
    --adapter_path "${GRPO_DIR}" \
    --output_dir "${GRPO_DIR}_merged"


# ------------------------------------------------------------
# GRPO evaluation
# ------------------------------------------------------------

GRPO_PRED="${OUT}/grpo_${GRPO_VARIANT}_predictions.jsonl"

run_vllm_eval \
    "${GRPO_DIR}_merged" \
    "test" \
    "${GRPO_PRED}"

run_score "${GRPO_PRED}"

run_log \
    "qwen_grpo_${GRPO_VARIANT}" \
    "${MODEL}" \
    "grpo" \
    "${GRPO_VARIANT}" \
    "${GRPO_PRED}"


# ------------------------------------------------------------
# Upload + cleanup
# ------------------------------------------------------------

push_model_hf \
    "${CKPT}/${BASE_VARIANT}_merged" \
    "${BASE_VARIANT}"

push_model_hf \
    "${GRPO_DIR}_merged" \
    "${GRPO_VARIANT}"

cleanup_variant_pair \
    "${BASE_VARIANT}" \
    "${GRPO_VARIANT}"


# ============================================================
# VARIANT 3 - SEMANTIC REPLAY
# ============================================================

echo ""
echo "############################################################"
echo "# VARIANT: SEMANTIC REPLAY"
echo "############################################################"


BASE_VARIANT="sem_replay"
GRPO_VARIANT="sem_replay"


run_sft_variant "${BASE_VARIANT}"


SFT_PRED="${OUT}/${BASE_VARIANT}_predictions.jsonl"

run_vllm_eval \
    "${CKPT}/${BASE_VARIANT}_merged" \
    "test" \
    "${SFT_PRED}"

run_score "${SFT_PRED}"

run_log \
    "qwen_${BASE_VARIANT}" \
    "${MODEL}" \
    "replay_sft" \
    "${BASE_VARIANT}" \
    "${SFT_PRED}"


# ------------------------------------------------------------
# GRPO
# ------------------------------------------------------------

GRPO_DIR="${CKPT}/grpo_${GRPO_VARIANT}"

python grpo_train.py \
    --model "${CKPT}/${BASE_VARIANT}_merged" \
    --output_dir "${GRPO_DIR}" \
    --w_correctness "${GRPO_W_CORRECTNESS}" \
    --w_format "${GRPO_W_FORMAT}" \
    --w_persistence "${GRPO_W_PERSISTENCE}" \
    --w_chain_stability "${GRPO_W_CHAIN_STABILITY}" \
    --attn_implementation flash_attention_2


python grpo_train.py \
    --merge_only \
    --adapter_path "${GRPO_DIR}" \
    --output_dir "${GRPO_DIR}_merged"


# ------------------------------------------------------------
# GRPO evaluation
# ------------------------------------------------------------

GRPO_PRED="${OUT}/grpo_${GRPO_VARIANT}_predictions.jsonl"

run_vllm_eval \
    "${GRPO_DIR}_merged" \
    "test" \
    "${GRPO_PRED}"

run_score "${GRPO_PRED}"

run_log \
    "qwen_grpo_${GRPO_VARIANT}" \
    "${MODEL}" \
    "grpo" \
    "${GRPO_VARIANT}" \
    "${GRPO_PRED}"


# ------------------------------------------------------------
# Upload + cleanup
# ------------------------------------------------------------

push_model_hf \
    "${CKPT}/${BASE_VARIANT}_merged" \
    "${BASE_VARIANT}"

push_model_hf \
    "${GRPO_DIR}_merged" \
    "${GRPO_VARIANT}"

cleanup_variant_pair \
    "${BASE_VARIANT}" \
    "${GRPO_VARIANT}"


# ============================================================
# VARIANT 4 - NUMERIC REPLAY
# ============================================================

echo ""
echo "############################################################"
echo "# VARIANT: NUMERIC REPLAY"
echo "############################################################"


BASE_VARIANT="num_replay"
GRPO_VARIANT="num_replay"


run_sft_variant "${BASE_VARIANT}"


SFT_PRED="${OUT}/${BASE_VARIANT}_predictions.jsonl"

run_vllm_eval \
    "${CKPT}/${BASE_VARIANT}_merged" \
    "test" \
    "${SFT_PRED}"

run_score "${SFT_PRED}"

run_log \
    "qwen_${BASE_VARIANT}" \
    "${MODEL}" \
    "replay_sft" \
    "${BASE_VARIANT}" \
    "${SFT_PRED}"


# ------------------------------------------------------------
# GRPO
# ------------------------------------------------------------

GRPO_DIR="${CKPT}/grpo_${GRPO_VARIANT}"

python grpo_train.py \
    --model "${CKPT}/${BASE_VARIANT}_merged" \
    --output_dir "${GRPO_DIR}" \
    --w_correctness "${GRPO_W_CORRECTNESS}" \
    --w_format "${GRPO_W_FORMAT}" \
    --w_persistence "${GRPO_W_PERSISTENCE}" \
    --w_chain_stability "${GRPO_W_CHAIN_STABILITY}" \
    --attn_implementation flash_attention_2


python grpo_train.py \
    --merge_only \
    --adapter_path "${GRPO_DIR}" \
    --output_dir "${GRPO_DIR}_merged"


# ------------------------------------------------------------
# GRPO evaluation
# ------------------------------------------------------------

GRPO_PRED="${OUT}/grpo_${GRPO_VARIANT}_predictions.jsonl"

run_vllm_eval \
    "${GRPO_DIR}_merged" \
    "test" \
    "${GRPO_PRED}"

run_score "${GRPO_PRED}"

run_log \
    "qwen_grpo_${GRPO_VARIANT}" \
    "${MODEL}" \
    "grpo" \
    "${GRPO_VARIANT}" \
    "${GRPO_PRED}"


# ------------------------------------------------------------
# Upload + cleanup
# ------------------------------------------------------------

push_model_hf \
    "${CKPT}/${BASE_VARIANT}_merged" \
    "${BASE_VARIANT}"

push_model_hf \
    "${GRPO_DIR}_merged" \
    "${GRPO_VARIANT}"

cleanup_variant_pair \
    "${BASE_VARIANT}" \
    "${GRPO_VARIANT}"


# ============================================================
# VARIANT 5 - BOTH REPLAY
# ============================================================

echo ""
echo "############################################################"
echo "# VARIANT: BOTH REPLAY"
echo "############################################################"


BASE_VARIANT="both_replay"
GRPO_VARIANT="both_replay"


run_sft_variant "${BASE_VARIANT}"


SFT_PRED="${OUT}/${BASE_VARIANT}_predictions.jsonl"

run_vllm_eval \
    "${CKPT}/${BASE_VARIANT}_merged" \
    "test" \
    "${SFT_PRED}"

run_score "${SFT_PRED}"

run_log \
    "qwen_${BASE_VARIANT}" \
    "${MODEL}" \
    "replay_sft" \
    "${BASE_VARIANT}" \
    "${SFT_PRED}"


# ------------------------------------------------------------
# GRPO
# ------------------------------------------------------------

GRPO_DIR="${CKPT}/grpo_${GRPO_VARIANT}"

python grpo_train.py \
    --model "${CKPT}/${BASE_VARIANT}_merged" \
    --output_dir "${GRPO_DIR}" \
    --w_correctness "${GRPO_W_CORRECTNESS}" \
    --w_format "${GRPO_W_FORMAT}" \
    --w_persistence "${GRPO_W_PERSISTENCE}" \
    --w_chain_stability "${GRPO_W_CHAIN_STABILITY}" \
    --attn_implementation flash_attention_2


python grpo_train.py \
    --merge_only \
    --adapter_path "${GRPO_DIR}" \
    --output_dir "${GRPO_DIR}_merged"


# ------------------------------------------------------------
# GRPO evaluation
# ------------------------------------------------------------

GRPO_PRED="${OUT}/grpo_${GRPO_VARIANT}_predictions.jsonl"

run_vllm_eval \
    "${GRPO_DIR}_merged" \
    "test" \
    "${GRPO_PRED}"

run_score "${GRPO_PRED}"

run_log \
    "qwen_grpo_${GRPO_VARIANT}" \
    "${MODEL}" \
    "grpo" \
    "${GRPO_VARIANT}" \
    "${GRPO_PRED}"


# ------------------------------------------------------------
# Upload + cleanup
# ------------------------------------------------------------

push_model_hf \
    "${CKPT}/${BASE_VARIANT}_merged" \
    "${BASE_VARIANT}"

push_model_hf \
    "${GRPO_DIR}_merged" \
    "${GRPO_VARIANT}"

cleanup_variant_pair \
    "${BASE_VARIANT}" \
    "${GRPO_VARIANT}"


# ============================================================
# FINAL CLEANUP
# ============================================================

echo ""
echo "============================================================"
echo " FINAL CLEANUP"
echo "============================================================"


if [ "${SMART_HF_DISABLE}" = "1" ]; then

    echo "[FINAL CLEANUP] HF disabled."
    echo "[FINAL CLEANUP] Keeping shared experiment data."

else

    echo "[FINAL CLEANUP] Removing temporary augmentation datasets."

    rm -f "${OUT}/sft_data.jsonl" || true
    rm -f "${OUT}/semantic_aug.jsonl" || true
    rm -f "${OUT}/numeric_aug.jsonl" || true

fi


# ============================================================
# UPLOAD EXPERIMENT OUTPUTS
# ============================================================

push_outputs_hf


# ============================================================
# FINAL DISK STATUS
# ============================================================

check_disk


# ============================================================
# COMPLETION
# ============================================================

echo ""
echo "============================================================"
echo " SMARTY - QWEN EXPERIMENT COMPLETE"
echo "============================================================"
echo ""
echo "Model:"
echo "  ${MODEL}"
echo ""
echo "Completed:"
echo "  [x] Baseline"
echo "  [x] Skill SFT"
echo "  [x] Diagnosis"
echo "  [x] Semantic augmentation"
echo "  [x] Numeric augmentation"
echo "  [x] Simple Replay SFT + GRPO"
echo "  [x] Semantic Replay SFT + GRPO"
echo "  [x] Numeric Replay SFT + GRPO"
echo "  [x] Both Replay SFT + GRPO"
echo ""
echo "Outputs:"
echo "  ${OUT}"
echo ""
echo "Ledger:"
echo "  ${LEDGER}"
echo ""
echo "============================================================"
