
#!/usr/bin/env bash
# ============================================================================
# run_full_experiment.sh - SMART experiment ladder
#
#   Stage 0 : Baseline (untrained)
#   Stage 1 : SFT on Skill_MATH
#   Stage 2 : Diagnosis + augmentation generation (semantic + numeric)
#   Stage 3 : SFT + simple replay          (Skill_MATH)
#             SFT + semantic replay        (Skill_MATH + semantic aug)
#             SFT + numeric replay         (Skill_MATH + numeric aug)
#             SFT + both replay            (Skill_MATH + semantic + numeric aug)
#   Stage 4 : GRPO on each of the five Skill_MATH SFT models
#
#   Total: 10 evaluation runs
#     1 baseline
#     1 Skill_MATH SFT
#     4 replay SFT
#     5 GRPO
#
# NOTES
#   - Training (SFT/GRPO) always uses flash-attention-2. No SDPA fallback.
#   - All inference/evaluation uses vLLM via --use_vllm.
#   - Only END evaluation per stage. Per-checkpoint evaluation is NOT part of
#     this script.
#   - The TEST split is the ORIGINAL Hendrycks MATH test set.
#   - The test set carries no skill annotations, so skill-prediction/skill-usage
#     report N/A there; final-answer accuracy, format compliance and arithmetic
#     consistency are computed normally.
#
# HUGGING FACE
#   Token must come from the environment:
#       export HF_TOKEN=hf_xxxxxxxx
#   Set SMART_HF_DISABLE=1 to skip all uploads.
#
# Usage:
#   ./run_full_experiment.sh <MODEL> [TRAIN_N] [TEST_N]
#
# Examples:
#   ./run_full_experiment.sh Qwen/Qwen2.5-Math-7B-Instruct
#   ./run_full_experiment.sh Qwen/Qwen2.5-Math-7B-Instruct 60 500
# ============================================================================

set -Eeuo pipefail

trap 'echo ""; echo "[ERROR] Pipeline failed at line $LINENO"; echo "[ERROR] Command: $BASH_COMMAND"; exit 1' ERR

# ============================================================================
# ARGUMENTS
# ============================================================================

MODEL="${1:?Usage: ./run_full_experiment.sh <MODEL> [TRAIN_N] [TEST_N]}"
TRAIN_N="${2:-full}"
TEST_N="${3:-full}"

MODEL_SLUG="$(echo "$MODEL" | tr '/' '_')"
OUT="outputs/${MODEL_SLUG}"
CKPT="ckpts/${MODEL_SLUG}"
LEDGER="outputs/experiment_ledger.jsonl"

mkdir -p "$OUT" "$CKPT"

# ============================================================================
# HUGGING FACE REPOSITORIES
# Token is read from HF_TOKEN in the environment - never hardcode it here.
# ============================================================================

export SMART_HF_SFT_REPO="${SMART_HF_SFT_REPO:-HusnainAmjad/Deepseek_Math_7b_r1_SFT}"
export SMART_HF_REPLAY_REPO="${SMART_HF_REPLAY_REPO:-HusnainAmjad/Deepseek_Math_7b_r1_Replay_SFT}"
export SMART_HF_GRPO_REPO="${SMART_HF_GRPO_REPO:-HusnainAmjad/Deepseek_Math_7b_r1_GRPO}"
export SMART_HF_OUTPUTS_REPO="${SMART_HF_OUTPUTS_REPO:-HusnainAmjad/Deepseek_Math_7b_r1_Outputs}"

if [ -z "${HF_TOKEN:-}" ] && [ -z "${SMART_HF_DISABLE:-}" ]; then
    echo "[HF] WARNING: HF_TOKEN is not set. Uploads will fail."
    echo "     The experiment itself will continue and artifacts remain local."
    echo "     Either:"
    echo "       export HF_TOKEN=hf_xxxxxxxxxxxxxxxx"
    echo "       export SMART_HF_DISABLE=1"
fi

# ============================================================================
# TRAINING SETTINGS
# ============================================================================

EPOCHS="${EPOCHS:-1}"
SAVE_EVERY_EPOCHS="${SAVE_EVERY_EPOCHS:-1}"
PER_DEVICE_BATCH="${PER_DEVICE_BATCH:-4}"
GRAD_ACCUM="${GRAD_ACCUM:-2}"

LORA_R="${LORA_R:-16}"
LORA_ALPHA="${LORA_ALPHA:-32}"

REPLAY_RATIO="${REPLAY_RATIO:-0.7}"
REPLAY_MODE="${REPLAY_MODE:-additive}"

GRPO_EPOCHS="${GRPO_EPOCHS:-1}"
GRPO_GENERATIONS="${GRPO_GENERATIONS:-4}"

# ============================================================================
# vLLM SETTINGS
# ============================================================================

VLLM_GPU_MEMORY="${VLLM_GPU_MEMORY:-0.90}"
VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-4096}"
VLLM_TENSOR_PARALLEL="${VLLM_TENSOR_PARALLEL:-1}"
VLLM_MAX_TOKENS="${VLLM_MAX_TOKENS:-2048}"

# ============================================================================
# LIMITS
# ============================================================================

if [ "$TEST_N" = "full" ]; then
    LIMIT_ARG=()
else
    LIMIT_ARG=(--limit "$TEST_N")
fi

if [ "$TRAIN_N" = "full" ]; then
    AUG_SOURCE_ARG=()
else
    AUG_SOURCE_ARG=(--limit_source "$TRAIN_N")
fi

# ============================================================================
# HEADER
# ============================================================================

echo "============================================================"
echo "SMART EXPERIMENT LADDER"
echo "============================================================"
echo "MODEL:                 $MODEL"
echo "MODEL_SLUG:            $MODEL_SLUG"
echo "TRAIN_N:               $TRAIN_N"
echo "TEST_N:                $TEST_N"
echo "OUTPUTS:               $OUT"
echo "CHECKPOINTS:           $CKPT"
echo "LEDGER:                $LEDGER"
echo "ATTENTION:             flash_attention_2 (forced, no sdpa)"
echo "INFERENCE:             vLLM"
echo "CHECKPOINT EVAL:       none (end evaluation only)"
echo "SFT EPOCHS:            $EPOCHS"
echo "GRPO EPOCHS:           $GRPO_EPOCHS"
echo "REPLAY MODE:           $REPLAY_MODE (ratio $REPLAY_RATIO)"
echo "vLLM GPU UTILIZATION:  $VLLM_GPU_MEMORY"
echo "vLLM MAX MODEL LEN:    $VLLM_MAX_MODEL_LEN"
echo "vLLM TP:               $VLLM_TENSOR_PARALLEL"
echo "vLLM MAX TOKENS:       $VLLM_MAX_TOKENS"
echo "------------------------------------------------------------"
echo "HF SFT REPO:           $SMART_HF_SFT_REPO"
echo "HF REPLAY REPO:        $SMART_HF_REPLAY_REPO"
echo "HF GRPO REPO:          $SMART_HF_GRPO_REPO"
echo "HF OUTPUTS DATASET:    $SMART_HF_OUTPUTS_REPO"
echo "HF TOKEN:              ${HF_TOKEN:+set}${HF_TOKEN:-NOT SET}"
echo "HF UPLOADS:            ${SMART_HF_DISABLE:+DISABLED}${SMART_HF_DISABLE:-enabled}"
echo "============================================================"

# ============================================================================
# HELPERS
# ============================================================================

run_vllm_eval() {
    local MODEL_PATH="$1"
    local OUTPUT_PATH="$2"

    echo ""
    echo "----------------------------------------------------------------------"
    echo "vLLM EVALUATION  |  model: $MODEL_PATH"
    echo "----------------------------------------------------------------------"

    python run_eval.py \
        --model "$MODEL_PATH" \
        --split test \
        "${LIMIT_ARG[@]}" \
        --use_vllm \
        --gpu_memory_utilization "$VLLM_GPU_MEMORY" \
        --max_model_len "$VLLM_MAX_MODEL_LEN" \
        --tensor_parallel_size "$VLLM_TENSOR_PARALLEL" \
        --max_tokens "$VLLM_MAX_TOKENS" \
        --out "$OUTPUT_PATH"

    echo "[OK] vLLM evaluation completed."
}

run_score() {
    local PREDICTIONS="$1"
    local DETAILED="$2"
    local SUMMARY="$3"

    echo ""
    echo "----------------------------------------------------------------------"
    echo "SCORING  |  $PREDICTIONS"
    echo "----------------------------------------------------------------------"

    python evaluator.py \
        --score \
        --predictions "$PREDICTIONS" \
        --split test \
        --out-detailed "$DETAILED" \
        --out-summary "$SUMMARY"

    echo "[OK] Scoring completed."
}

run_log() {
    local RUN_ID="$1"
    local CONFIG="$2"
    local SUMMARY="$3"
    local BASELINE="$4"
    local NOTES="$5"

    echo ""
    echo "----------------------------------------------------------------------"
    echo "LEDGER  |  $RUN_ID"
    echo "----------------------------------------------------------------------"

    if [ -n "$BASELINE" ]; then
        python experiment_ledger.py --log \
            --run_id "$RUN_ID" \
            --training_config "$CONFIG" \
            --eval_summary "$SUMMARY" \
            --ledger "$LEDGER" \
            --baseline_run_id "$BASELINE" \
            --notes "$NOTES"
    else
        python experiment_ledger.py --log \
            --run_id "$RUN_ID" \
            --training_config "$CONFIG" \
            --eval_summary "$SUMMARY" \
            --ledger "$LEDGER" \
            --notes "$NOTES"
    fi

    echo "[OK] Ledger entry completed."
}

# Push a merged checkpoint.
# A failed upload must not destroy a completed training run.

push_model_hf() {
    local LOCAL_DIR="$1"
    local VARIANT="$2"

    if [ -n "${SMART_HF_DISABLE:-}" ]; then
        return 0
    fi

    echo ""
    echo "[HF] pushing $VARIANT -> Hub"

    python hf_sync.py --push-model \
        --local "$LOCAL_DIR" \
        --model-slug "$MODEL_SLUG" \
        --variant "$VARIANT" \
        || echo "[HF] WARNING: push of $VARIANT failed - local checkpoint intact, continuing"
}

# ============================================================================
# DATA PREPARATION
#   sft_data.jsonl - Skill_MATH (skill-labeled)
# ============================================================================

echo ""
echo "======================================================================"
echo "PREPARING DATA"
echo "======================================================================"

if [ ! -f outputs/sft_data_full.jsonl ]; then
    echo "[DATA] Building Skill_MATH SFT dataset..."

    python data_pipeline.py \
        --build-sft \
        --split train \
        --out outputs/sft_data_full.jsonl
else
    echo "[DATA] Skill_MATH SFT dataset already exists."
fi

if [ "$TRAIN_N" = "full" ]; then
    cp outputs/sft_data_full.jsonl "$OUT/sft_data.jsonl"
else
    head -n "$TRAIN_N" outputs/sft_data_full.jsonl > "$OUT/sft_data.jsonl"
fi

echo "[DATA] Skill_MATH training examples:"
wc -l "$OUT/sft_data.jsonl"

# ============================================================================
# STAGE 0 - BASELINE
# ============================================================================

echo ""
echo "======================================================================"
echo "STAGE 0: BASELINE"
echo "======================================================================"

run_vllm_eval \
    "$MODEL" \
    "$OUT/predictions_baseline.jsonl"

run_score \
    "$OUT/predictions_baseline.jsonl" \
    "$OUT/eval_baseline_detailed.jsonl" \
    "$OUT/eval_baseline_summary.json"

python3 -c "
import json
json.dump(
    {'model': '${MODEL}', 'mode': 'baseline'},
    open('${OUT}/baseline_config.json', 'w'),
    indent=2
)
"

run_log \
    "${MODEL_SLUG}__baseline" \
    "$OUT/baseline_config.json" \
    "$OUT/eval_baseline_summary.json" \
    "" \
    "vanilla, untrained, n=$TEST_N"

# ============================================================================
# SFT VARIANT FUNCTION
#
#   $1 variant name
#   $2 training data file
#   $3 extra augmented data, "" for none
#   $4 use replay? yes/no
#   $5 ledger baseline run_id
#   $6 notes
# ============================================================================

run_sft_variant() {
    local VARIANT_NAME="$1"
    local DATA_FILE="$2"
    local EXTRA_DATA="${3:-}"
    local USE_REPLAY="${4:-no}"
    local BASELINE_ID="$5"
    local NOTES="$6"

    local OUT_DIR="${CKPT}/${VARIANT_NAME}"

    echo ""
    echo "======================================================================"
    echo "SFT VARIANT: $VARIANT_NAME"
    echo "======================================================================"
    echo "[SFT] data:       $DATA_FILE"
    echo "[SFT] extra data: ${EXTRA_DATA:-<none>}"
    echo "[SFT] replay:     $USE_REPLAY"

    local REPLAY_ARGS=()

    if [ "$USE_REPLAY" = "yes" ]; then
        REPLAY_ARGS=(
            --replay_strategy skill
            --replay_ratio "$REPLAY_RATIO"
            --replay_mode "$REPLAY_MODE"
        )
    fi

    local EXTRA_ARGS=()

    if [ -n "$EXTRA_DATA" ]; then
        EXTRA_ARGS=(--extra_data $EXTRA_DATA)
    fi

    python sft_train.py \
        --model "$MODEL" \
        --data "$DATA_FILE" \
        "${EXTRA_ARGS[@]}" \
        "${REPLAY_ARGS[@]}" \
        --mode lora \
        --lora_r "$LORA_R" \
        --lora_alpha "$LORA_ALPHA" \
        --output_dir "$OUT_DIR" \
        --epochs "$EPOCHS" \
        --save_every_epochs "$SAVE_EVERY_EPOCHS" \
        --per_device_batch_size "$PER_DEVICE_BATCH" \
        --grad_accum "$GRAD_ACCUM"

    echo ""
    echo "[SFT] Final merged model: ${OUT_DIR}_merged"

    run_vllm_eval \
        "${OUT_DIR}_merged" \
        "$OUT/predictions_${VARIANT_NAME}.jsonl"

    run_score \
        "$OUT/predictions_${VARIANT_NAME}.jsonl" \
        "$OUT/eval_${VARIANT_NAME}_detailed.jsonl" \
        "$OUT/eval_${VARIANT_NAME}_summary.json"

    run_log \
        "${MODEL_SLUG}__${VARIANT_NAME}" \
        "${OUT_DIR}_merged/training_config.json" \
        "$OUT/eval_${VARIANT_NAME}_summary.json" \
        "$BASELINE_ID" \
        "$NOTES"

    push_model_hf \
        "${OUT_DIR}_merged" \
        "$VARIANT_NAME"
}

# ============================================================================
# STAGE 1 - SFT ON SKILL_MATH
# ============================================================================

run_sft_variant \
    "sft_skill" \
    "$OUT/sft_data.jsonl" \
    "" \
    "no" \
    "${MODEL_SLUG}__baseline" \
    "SFT on Skill_MATH (skill-labeled), n=$TEST_N"

# ============================================================================
# STAGE 2 - DIAGNOSIS
# ============================================================================

echo ""
echo "======================================================================"
echo "STAGE 2: DIAGNOSIS"
echo "======================================================================"

python data_pipeline.py \
    --diagnose \
    --predictions "$OUT/predictions_sft_skill.jsonl" \
    --weak-report "$OUT/weak_clusters.json"

cat "$OUT/weak_clusters.json"

echo "[OK] Diagnosis completed."

# ============================================================================
# STAGE 2 - AUGMENTATION GENERATION
# ============================================================================

echo ""
echo "======================================================================"
echo "STAGE 2: AUGMENTATION GENERATION"
echo "======================================================================"

echo "[AUG] semantic..."

python run_augmentation.py \
    --stage semantic \
    --model "${CKPT}/sft_skill_merged" \
    --weak_report "$OUT/weak_clusters.json" \
    --out "$OUT/semantic_aug.jsonl" \
    "${AUG_SOURCE_ARG[@]}" \
    --batch_size 64

echo "[OK] Semantic augmentation completed."

echo "[AUG] numeric..."

python run_augmentation.py \
    --stage numeric \
    --model "${CKPT}/sft_skill_merged" \
    --weak_report "$OUT/weak_clusters.json" \
    --out "$OUT/numeric_aug.jsonl" \
    --n_per_problem 1 \
    --votes 3 \
    "${AUG_SOURCE_ARG[@]}" \
    --batch_size 64

echo "[OK] Numeric augmentation completed."

# ============================================================================
# STAGE 3 - REPLAY VARIANTS
#
#   1. Simple Replay
#   2. Semantic Replay
#   3. Numeric Replay
#   4. Semantic + Numeric Replay
# ============================================================================

echo ""
echo "======================================================================"
echo "STAGE 3: REPLAY VARIANTS"
echo "======================================================================"

run_sft_variant \
    "simple_replay" \
    "$OUT/sft_data.jsonl" \
    "" \
    "yes" \
    "${MODEL_SLUG}__sft_skill" \
    "Skill_MATH + simple skill replay, n=$TEST_N"

run_sft_variant \
    "sem_replay" \
    "$OUT/sft_data.jsonl" \
    "$OUT/semantic_aug.jsonl" \
    "yes" \
    "${MODEL_SLUG}__sft_skill" \
    "Skill_MATH + semantic augmentation + replay, n=$TEST_N"

run_sft_variant \
    "num_replay" \
    "$OUT/sft_data.jsonl" \
    "$OUT/numeric_aug.jsonl" \
    "yes" \
    "${MODEL_SLUG}__sft_skill" \
    "Skill_MATH + numeric augmentation + replay, n=$TEST_N"

run_sft_variant \
    "both_replay" \
    "$OUT/sft_data.jsonl" \
    "$OUT/semantic_aug.jsonl $OUT/numeric_aug.jsonl" \
    "yes" \
    "${MODEL_SLUG}__sft_skill" \
    "Skill_MATH + semantic & numeric augmentation + replay, n=$TEST_N"

# ============================================================================
# STAGE 4 - GRPO FUNCTION
# ============================================================================

run_stage4_grpo() {
    local BASE_NAME="$1"
    local BASE_MODEL_DIR="$2"

    local GRPO_DIR="${CKPT}/grpo_${BASE_NAME}"

    echo ""
    echo "======================================================================"
    echo "STAGE 4: GRPO ON ${BASE_NAME}"
    echo "======================================================================"

    python grpo_train.py \
        --model "$BASE_MODEL_DIR" \
        --data "$OUT/sft_data.jsonl" \
        --output_dir "$GRPO_DIR" \
        --num_generations "$GRPO_GENERATIONS" \
        --per_device_batch_size 4 \
        --grad_accum 1 \
        --num_train_epochs "$GRPO_EPOCHS" \
        --w_correctness 1.0 \
        --w_format 0.2 \
        --w_persistence 0.15 \
        --w_chain_stability 0.25

    echo ""
    echo "[GRPO] Final merged model: ${GRPO_DIR}_merged"

    run_vllm_eval \
        "${GRPO_DIR}_merged" \
        "$OUT/predictions_grpo_${BASE_NAME}.jsonl"

    run_score \
        "$OUT/predictions_grpo_${BASE_NAME}.jsonl" \
        "$OUT/eval_grpo_${BASE_NAME}_detailed.jsonl" \
        "$OUT/eval_grpo_${BASE_NAME}_summary.json"

    run_log \
        "${MODEL_SLUG}__grpo_${BASE_NAME}" \
        "${GRPO_DIR}_merged/training_config.json" \
        "$OUT/eval_grpo_${BASE_NAME}_summary.json" \
        "${MODEL_SLUG}__${BASE_NAME}" \
        "GRPO on ${BASE_NAME}, n=$TEST_N"

    push_model_hf \
        "${GRPO_DIR}_merged" \
        "grpo_${BASE_NAME}"
}

# ============================================================================
# STAGE 4 - GRPO ON THE FIVE SKILL_MATH SFT MODELS
#
#   1. SFT Skill_MATH
#   2. Simple Replay
#   3. Semantic Replay
#   4. Numeric Replay
#   5. Semantic + Numeric Replay
# ============================================================================

run_stage4_grpo \
    "sft_skill" \
    "${CKPT}/sft_skill_merged"

run_stage4_grpo \
    "simple_replay" \
    "${CKPT}/simple_replay_merged"

run_stage4_grpo \
    "sem_replay" \
    "${CKPT}/sem_replay_merged"

run_stage4_grpo \
    "num_replay" \
    "${CKPT}/num_replay_merged"

run_stage4_grpo \
    "both_replay" \
    "${CKPT}/both_replay_merged"

# ============================================================================
# REPORTS
# ============================================================================

echo ""
echo "======================================================================"
echo "GENERATING REPORTS"
echo "======================================================================"

python experiment_ledger.py \
    --print \
    --ledger "$LEDGER"

python generate_all_reports.py \
    --ledger "$LEDGER" \
    --out_dir outputs/report

# ============================================================================
# HUGGING FACE - PUSH ALL ARTIFACTS
# ============================================================================

if [ -z "${SMART_HF_DISABLE:-}" ]; then

    echo ""
    echo "======================================================================"
    echo "PUSHING ARTIFACTS TO $SMART_HF_OUTPUTS_REPO"
    echo "======================================================================"

    python hf_sync.py \
        --push-outputs \
        --model-slug "$MODEL_SLUG" \
        || echo "[HF] WARNING: outputs push failed - local artifacts are intact"

fi

# ============================================================================
# DONE
# ============================================================================

echo ""
echo "======================================================================"
echo "SMART EXPERIMENT LADDER COMPLETE"
echo "======================================================================"

echo "MODEL: $MODEL"
echo ""

echo "Stage 0:"
echo "  ${MODEL_SLUG}__baseline"

echo ""
echo "Stage 1:"
echo "  ${MODEL_SLUG}__sft_skill"

echo ""
echo "Stage 3:"
echo "  ${MODEL_SLUG}__simple_replay"
echo "  ${MODEL_SLUG}__sem_replay"
echo "  ${MODEL_SLUG}__num_replay"
echo "  ${MODEL_SLUG}__both_replay"

echo ""
echo "Stage 4:"
echo "  ${MODEL_SLUG}__grpo_sft_skill"
echo "  ${MODEL_SLUG}__grpo_simple_replay"
echo "  ${MODEL_SLUG}__grpo_sem_replay"
echo "  ${MODEL_SLUG}__grpo_num_replay"
echo "  ${MODEL_SLUG}__grpo_both_replay"

echo ""
echo "Total evaluation runs: 10"
echo "  1 baseline"
echo "  1 Skill_MATH SFT"
echo "  4 replay SFT"
echo "  5 GRPO"

echo ""
echo "Checkpoint evaluations: none (end evaluation only)"
echo ""
echo "Ledger:      $LEDGER"
echo "Report:      outputs/report/"
echo "HF SFT:      https://huggingface.co/$SMART_HF_SFT_REPO"
echo "HF Replay:   https://huggingface.co/$SMART_HF_REPLAY_REPO"
echo "HF GRPO:     https://huggingface.co/$SMART_HF_GRPO_REPO"
echo "HF Outputs:  https://huggingface.co/datasets/$SMART_HF_OUTPUTS_REPO"

echo "======================================================================"
