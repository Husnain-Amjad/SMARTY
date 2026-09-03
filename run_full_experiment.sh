#!/usr/bin/env bash
# ============================================================================
# run_full_experiment.sh - SMART experiment ladder
#
#   Stage 0 : Baseline (untrained)
#
#   Stage 1a: SFT on ORIGINAL Hendrycks MATH train   <- control arm
#   Stage 1b: SFT on Skill_MATH (skill-labeled)
#
#   Stage 2 : Simple Replay              (Skill_MATH)
#   Stage 2 : Semantic Augmentation      (Skill_MATH + semantic)
#   Stage 2 : Numeric Augmentation       (Skill_MATH + numeric)
#   Stage 2 : Semantic + Numeric         (Skill_MATH + both)
#
#   Stage 3 : GRPO on each of the six SFT variants above
#
# NOTES
#   - Training (SFT/GRPO) always uses flash-attention-2. No sdpa fallback.
#   - All inference/evaluation uses vLLM via --use_vllm.
#   - The TEST split is the ORIGINAL Hendrycks MATH test set. It has no skill
#     annotations, so skill-prediction/skill-usage metrics report N/A there;
#     final-answer accuracy, format compliance and arithmetic consistency are
#     all computed normally.
#   - Per-checkpoint evaluation is written but COMMENTED OUT (see the
#     "OPTIONAL" blocks). Uncomment if you want retention curves.
#
# HUGGING FACE (optional). Set these to auto-upload; unset = no uploads:
#   export HF_TOKEN=hf_xxx
#   export SMART_HF_SFT_REPO="you/sft_model_name"
#   export SMART_HF_REPLAY_REPO="you/replay_augm_sft_model_name"
#   export SMART_HF_GRPO_REPO="you/GRPO_model_name"
#   export SMART_HF_OUTPUTS_REPO="you/model_name_outputs_results"
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
# TRAINING SETTINGS
# ============================================================================
EPOCHS="${EPOCHS:-5}"
SAVE_EVERY_EPOCHS="${SAVE_EVERY_EPOCHS:-1}"
PER_DEVICE_BATCH="${PER_DEVICE_BATCH:-4}"
GRAD_ACCUM="${GRAD_ACCUM:-2}"
LORA_R="${LORA_R:-16}"
LORA_ALPHA="${LORA_ALPHA:-32}"
REPLAY_RATIO="${REPLAY_RATIO:-0.7}"
REPLAY_MODE="${REPLAY_MODE:-additive}"      # additive = augmented data GROWS the set
GRPO_EPOCHS="${GRPO_EPOCHS:-5}"
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

# Augmentation draws from the full split unless capped. Without this a small
# smoke test still generates thousands of augmented examples.
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
echo "EPOCHS (SFT/GRPO):     $EPOCHS / $GRPO_EPOCHS"
echo "REPLAY MODE:           $REPLAY_MODE (ratio $REPLAY_RATIO)"
echo "vLLM GPU UTILIZATION:  $VLLM_GPU_MEMORY"
echo "vLLM MAX MODEL LEN:    $VLLM_MAX_MODEL_LEN"
echo "vLLM TP:               $VLLM_TENSOR_PARALLEL"
echo "vLLM MAX TOKENS:       $VLLM_MAX_TOKENS"
echo "CHECKPOINT EVAL:       DISABLED (commented out)"
echo "HF MODEL REPOS:        sft=${SMART_HF_SFT_REPO:-<unset>}"
echo "                       replay=${SMART_HF_REPLAY_REPO:-<unset>}"
echo "                       grpo=${SMART_HF_GRPO_REPO:-<unset>}"
echo "HF OUTPUTS REPO:       ${SMART_HF_OUTPUTS_REPO:-<unset>}"
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
    local PREDICTIONS="$1"; local DETAILED="$2"; local SUMMARY="$3"
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
    local RUN_ID="$1"; local CONFIG="$2"; local SUMMARY="$3"
    local BASELINE="$4"; local NOTES="$5"
    echo ""
    echo "----------------------------------------------------------------------"
    echo "LEDGER  |  $RUN_ID"
    echo "----------------------------------------------------------------------"
    if [ -n "$BASELINE" ]; then
        python experiment_ledger.py --log --run_id "$RUN_ID" \
            --training_config "$CONFIG" --eval_summary "$SUMMARY" \
            --ledger "$LEDGER" --baseline_run_id "$BASELINE" --notes "$NOTES"
    else
        python experiment_ledger.py --log --run_id "$RUN_ID" \
            --training_config "$CONFIG" --eval_summary "$SUMMARY" \
            --ledger "$LEDGER" --notes "$NOTES"
    fi
    echo "[OK] Ledger entry completed."
}

# Push one merged checkpoint. Family (sft/replay/grpo) is auto-routed from the
# variant name by hf_sync.py. No-op when the repo env vars are unset.
push_model_hf() {
    local LOCAL_DIR="$1"; local VARIANT="$2"
    if [ -z "${SMART_HF_SFT_REPO:-}${SMART_HF_REPLAY_REPO:-}${SMART_HF_GRPO_REPO:-}" ]; then
        return 0
    fi
    echo ""
    echo "[HF] pushing $VARIANT -> Hub"
    python hf_sync.py --push-model \
        --local "$LOCAL_DIR" \
        --model-slug "$MODEL_SLUG" \
        --variant "$VARIANT" \
        || echo "[HF] WARNING: push of $VARIANT failed - continuing"
}

# ============================================================================
# SHARED DATA
#   sft_data_original.jsonl  - ORIGINAL Hendrycks MATH   (control arm)
#   sft_data.jsonl           - Skill_MATH (skill-labeled)
# ============================================================================
echo ""
echo "======================================================================"
echo "PREPARING DATA"
echo "======================================================================"

if [ ! -f outputs/sft_data_original_full.jsonl ]; then
    echo "[DATA] Building ORIGINAL Hendrycks MATH SFT dataset (no skill labels)..."
    python data_pipeline.py --build-sft-original --split train \
        --out outputs/sft_data_original_full.jsonl
else
    echo "[DATA] Original-MATH SFT dataset already exists."
fi

if [ ! -f outputs/sft_data_full.jsonl ]; then
    echo "[DATA] Building Skill_MATH SFT dataset..."
    python data_pipeline.py --build-sft --split train \
        --out outputs/sft_data_full.jsonl
else
    echo "[DATA] Skill_MATH SFT dataset already exists."
fi

if [ "$TRAIN_N" = "full" ]; then
    cp outputs/sft_data_original_full.jsonl "$OUT/sft_data_original.jsonl"
    cp outputs/sft_data_full.jsonl          "$OUT/sft_data.jsonl"
else
    head -n "$TRAIN_N" outputs/sft_data_original_full.jsonl > "$OUT/sft_data_original.jsonl"
    head -n "$TRAIN_N" outputs/sft_data_full.jsonl          > "$OUT/sft_data.jsonl"
fi

echo "[DATA] Original-MATH training examples:"; wc -l "$OUT/sft_data_original.jsonl"
echo "[DATA] Skill_MATH training examples:";    wc -l "$OUT/sft_data.jsonl"

# ============================================================================
# STAGE 0 - BASELINE
# ============================================================================
echo ""
echo "======================================================================"
echo "STAGE 0: BASELINE"
echo "======================================================================"

run_vllm_eval "$MODEL" "$OUT/predictions_baseline.jsonl"
run_score "$OUT/predictions_baseline.jsonl" \
          "$OUT/eval_baseline_detailed.jsonl" \
          "$OUT/eval_baseline_summary.json"

python3 -c "
import json
json.dump({'model': '${MODEL}', 'mode': 'baseline'},
          open('${OUT}/baseline_config.json', 'w'), indent=2)
"

run_log "${MODEL_SLUG}__baseline" \
        "$OUT/baseline_config.json" \
        "$OUT/eval_baseline_summary.json" \
        "" \
        "vanilla, untrained, n=$TEST_N"

# ============================================================================
# SFT VARIANT FUNCTION
#   $1 variant name
#   $2 training data file
#   $3 extra (augmented) data, "" for none
#   $4 use replay strategy? "yes"/"no"
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
    echo "[SFT] data:        $DATA_FILE"
    echo "[SFT] extra data:  ${EXTRA_DATA:-<none>}"
    echo "[SFT] replay:      $USE_REPLAY"

    local REPLAY_ARGS=()
    if [ "$USE_REPLAY" = "yes" ]; then
        REPLAY_ARGS=(--replay_strategy skill --replay_ratio "$REPLAY_RATIO"
                     --replay_mode "$REPLAY_MODE")
    fi

    local EXTRA_ARGS=()
    if [ -n "$EXTRA_DATA" ]; then
        # unquoted on purpose: may be two space-separated paths
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

    run_vllm_eval "${OUT_DIR}_merged" "$OUT/predictions_${VARIANT_NAME}.jsonl"
    run_score "$OUT/predictions_${VARIANT_NAME}.jsonl" \
              "$OUT/eval_${VARIANT_NAME}_detailed.jsonl" \
              "$OUT/eval_${VARIANT_NAME}_summary.json"
    run_log "${MODEL_SLUG}__${VARIANT_NAME}" \
            "${OUT_DIR}_merged/training_config.json" \
            "$OUT/eval_${VARIANT_NAME}_summary.json" \
            "$BASELINE_ID" \
            "$NOTES"

    push_model_hf "${OUT_DIR}_merged" "$VARIANT_NAME"

    # ------------------------------------------------------------------------
    # OPTIONAL: per-checkpoint evaluation (retention curves).
    # Uncomment to enable. Costs a full vLLM engine startup per checkpoint.
    # ------------------------------------------------------------------------
    # python run_multi_checkpoint_eval.py \
    #     --checkpoints_dir "$OUT_DIR" \
    #     --base_model "$MODEL" \
    #     --mode lora \
    #     --split test \
    #     "${LIMIT_ARG[@]}" \
    #     --use_vllm \
    #     --checkpoint_stride 1 \
    #     --out-dir "$OUT/checkpoint_eval_${VARIANT_NAME}" \
    #     --ledger "$LEDGER" \
    #     --run_id_prefix "${MODEL_SLUG}__${VARIANT_NAME}_ckpts" \
    #     --baseline_run_id "$BASELINE_ID" \
    #     || echo "[WARN] checkpoint eval for $VARIANT_NAME did not complete - continuing"
}

# ============================================================================
# STAGE 1a - SFT ON ORIGINAL HENDRYCKS MATH  (control arm)
# ============================================================================
run_sft_variant "sft_original" \
    "$OUT/sft_data_original.jsonl" "" "no" \
    "${MODEL_SLUG}__baseline" \
    "SFT on ORIGINAL Hendrycks MATH (no skill labels), n=$TEST_N"

# ============================================================================
# STAGE 1b - SFT ON SKILL_MATH
# ============================================================================
run_sft_variant "sft_skill" \
    "$OUT/sft_data.jsonl" "" "no" \
    "${MODEL_SLUG}__sft_original" \
    "SFT on Skill_MATH (skill-labeled), n=$TEST_N"

# ============================================================================
# STAGE 2 - DIAGNOSIS
# ============================================================================
echo ""
echo "======================================================================"
echo "STAGE 2: DIAGNOSIS"
echo "======================================================================"

python data_pipeline.py --diagnose \
    --predictions "$OUT/predictions_sft_skill.jsonl" \
    --weak-report "$OUT/weak_clusters.json"
cat "$OUT/weak_clusters.json"
echo "[OK] Diagnosis completed."

# ============================================================================
# STAGE 2 - AUGMENTATION
# ============================================================================
echo ""
echo "======================================================================"
echo "STAGE 2: AUGMENTATION"
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
# STAGE 2 VARIANTS
# ============================================================================
run_sft_variant "simple_replay" \
    "$OUT/sft_data.jsonl" "" "yes" \
    "${MODEL_SLUG}__sft_skill" \
    "Skill_MATH + simple skill replay, n=$TEST_N"

run_sft_variant "sem_replay" \
    "$OUT/sft_data.jsonl" "$OUT/semantic_aug.jsonl" "yes" \
    "${MODEL_SLUG}__sft_skill" \
    "Skill_MATH + semantic augmentation + replay, n=$TEST_N"

run_sft_variant "num_replay" \
    "$OUT/sft_data.jsonl" "$OUT/numeric_aug.jsonl" "yes" \
    "${MODEL_SLUG}__sft_skill" \
    "Skill_MATH + numeric augmentation + replay, n=$TEST_N"

run_sft_variant "both_replay" \
    "$OUT/sft_data.jsonl" "$OUT/semantic_aug.jsonl $OUT/numeric_aug.jsonl" "yes" \
    "${MODEL_SLUG}__sft_skill" \
    "Skill_MATH + semantic & numeric augmentation + replay, n=$TEST_N"

# ============================================================================
# STAGE 3 - GRPO FUNCTION
# ============================================================================
run_stage3_grpo() {
    local BASE_NAME="$1"
    local BASE_MODEL_DIR="$2"
    local GRPO_DIR="${CKPT}/grpo_${BASE_NAME}"

    echo ""
    echo "======================================================================"
    echo "STAGE 3: GRPO ON ${BASE_NAME}"
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

    run_vllm_eval "${GRPO_DIR}_merged" "$OUT/predictions_grpo_${BASE_NAME}.jsonl"
    run_score "$OUT/predictions_grpo_${BASE_NAME}.jsonl" \
              "$OUT/eval_grpo_${BASE_NAME}_detailed.jsonl" \
              "$OUT/eval_grpo_${BASE_NAME}_summary.json"
    run_log "${MODEL_SLUG}__grpo_${BASE_NAME}" \
            "${GRPO_DIR}_merged/training_config.json" \
            "$OUT/eval_grpo_${BASE_NAME}_summary.json" \
            "${MODEL_SLUG}__${BASE_NAME}" \
            "GRPO on ${BASE_NAME}, n=$TEST_N"

    push_model_hf "${GRPO_DIR}_merged" "grpo_${BASE_NAME}"

    # ------------------------------------------------------------------------
    # OPTIONAL: per-checkpoint GRPO evaluation. Uncomment to enable.
    # ------------------------------------------------------------------------
    # python run_multi_checkpoint_eval.py \
    #     --checkpoints_dir "$GRPO_DIR" \
    #     --base_model "$BASE_MODEL_DIR" \
    #     --mode lora \
    #     --split test \
    #     "${LIMIT_ARG[@]}" \
    #     --use_vllm \
    #     --checkpoint_stride 1 \
    #     --out-dir "$OUT/checkpoint_eval_grpo_${BASE_NAME}" \
    #     --ledger "$LEDGER" \
    #     --run_id_prefix "${MODEL_SLUG}__grpo_${BASE_NAME}_ckpts" \
    #     --baseline_run_id "${MODEL_SLUG}__baseline" \
    #     || echo "[WARN] checkpoint eval for grpo_${BASE_NAME} did not complete - continuing"
}

# ============================================================================
# STAGE 3 - GRPO ON ALL SIX SFT VARIANTS
# ============================================================================
run_stage3_grpo "sft_original"  "${CKPT}/sft_original_merged"
run_stage3_grpo "sft_skill"     "${CKPT}/sft_skill_merged"
run_stage3_grpo "simple_replay" "${CKPT}/simple_replay_merged"
run_stage3_grpo "sem_replay"    "${CKPT}/sem_replay_merged"
run_stage3_grpo "num_replay"    "${CKPT}/num_replay_merged"
run_stage3_grpo "both_replay"   "${CKPT}/both_replay_merged"

# ============================================================================
# REPORTS
# ============================================================================
echo ""
echo "======================================================================"
echo "GENERATING REPORTS"
echo "======================================================================"
python experiment_ledger.py --print --ledger "$LEDGER"
python generate_all_reports.py --ledger "$LEDGER" --out_dir outputs/report

# ============================================================================
# HUGGING FACE - PUSH ALL ARTIFACTS
# ============================================================================
if [ -n "${SMART_HF_OUTPUTS_REPO:-}" ]; then
    echo ""
    echo "======================================================================"
    echo "PUSHING ARTIFACTS TO HUGGING FACE"
    echo "======================================================================"
    python hf_sync.py --push-outputs --model-slug "$MODEL_SLUG" \
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
echo "Stage 0:  ${MODEL_SLUG}__baseline"
echo ""
echo "Stage 1:  ${MODEL_SLUG}__sft_original      (control: original MATH)"
echo "          ${MODEL_SLUG}__sft_skill         (Skill_MATH)"
echo ""
echo "Stage 2:  ${MODEL_SLUG}__simple_replay"
echo "          ${MODEL_SLUG}__sem_replay"
echo "          ${MODEL_SLUG}__num_replay"
echo "          ${MODEL_SLUG}__both_replay"
echo ""
echo "Stage 3:  ${MODEL_SLUG}__grpo_sft_original"
echo "          ${MODEL_SLUG}__grpo_sft_skill"
echo "          ${MODEL_SLUG}__grpo_simple_replay"
echo "          ${MODEL_SLUG}__grpo_sem_replay"
echo "          ${MODEL_SLUG}__grpo_num_replay"
echo "          ${MODEL_SLUG}__grpo_both_replay"
echo ""
echo "Total evaluation runs: 13  (1 baseline + 6 SFT + 6 GRPO)"
echo "Checkpoint evaluations: DISABLED (commented out)"
echo ""
echo "Ledger:  $LEDGER"
echo "Report:  outputs/report/"
echo "======================================================================"
