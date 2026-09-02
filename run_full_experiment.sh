```bash
#!/usr/bin/env bash
# ============================================================================
# run_full_experiment.sh
#
# SMART experiment ladder
#
#   Stage 0 : Baseline
#   Stage 1 : Simple SFT
#
#   Stage 2 : Simple Replay
#   Stage 2 : Semantic Replay
#   Stage 2 : Numeric Replay
#   Stage 2 : Semantic + Numeric Replay
#
#   Stage 3 : GRPO on Stage 1
#   Stage 3 : GRPO on Simple Replay
#   Stage 3 : GRPO on Semantic Replay
#   Stage 3 : GRPO on Numeric Replay
#   Stage 3 : GRPO on Semantic + Numeric Replay
#
# IMPORTANT:
#   - NO checkpoint evaluations are performed.
#   - All final evaluations use vLLM.
#
# Usage:
#   ./run_full_experiment.sh <MODEL> [TRAIN_N] [TEST_N]
#
# Examples:
#   ./run_full_experiment.sh Qwen/Qwen2.5-Math-7B-Instruct
#   ./run_full_experiment.sh Qwen/Qwen2.5-Math-7B-Instruct 60 500
# ============================================================================

set -Eeuo pipefail


# ============================================================================
# ERROR HANDLING
# ============================================================================

trap 'echo ""; echo "[ERROR] Pipeline failed at line $LINENO"; echo "[ERROR] Command: $BASH_COMMAND"; exit 1' ERR


# ============================================================================
# ARGUMENTS
# ============================================================================

MODEL="${1:?Usage: ./run_full_experiment.sh <MODEL> [TRAIN_N] [TEST_N]}"
TRAIN_N="${2:-full}"
TEST_N="${3:-full}"


# ============================================================================
# MODEL / OUTPUT PATHS
# ============================================================================

MODEL_SLUG="$(echo "$MODEL" | tr '/' '_')"

OUT="outputs/${MODEL_SLUG}"
CKPT="ckpts/${MODEL_SLUG}"
LEDGER="outputs/experiment_ledger.jsonl"

mkdir -p "$OUT" "$CKPT"


# ============================================================================
# vLLM SETTINGS
# ============================================================================

VLLM_GPU_MEMORY="${VLLM_GPU_MEMORY:-0.90}"
VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-4096}"
VLLM_TENSOR_PARALLEL="${VLLM_TENSOR_PARALLEL:-1}"
VLLM_MAX_TOKENS="${VLLM_MAX_TOKENS:-2048}"


# ============================================================================
# TEST LIMIT
# ============================================================================

if [ "$TEST_N" = "full" ]; then
    LIMIT_ARG=()
else
    LIMIT_ARG=(--limit "$TEST_N")
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
echo "vLLM GPU UTILIZATION:  $VLLM_GPU_MEMORY"
echo "vLLM MAX MODEL LEN:    $VLLM_MAX_MODEL_LEN"
echo "vLLM TP:               $VLLM_TENSOR_PARALLEL"
echo "vLLM MAX TOKENS:       $VLLM_MAX_TOKENS"
echo "CHECKPOINT EVALUATION: DISABLED"
echo "============================================================"


# ============================================================================
# SHARED SFT DATA
# ============================================================================

echo ""
echo "======================================================================"
echo "PREPARING SFT DATA"
echo "======================================================================"

if [ ! -f outputs/sft_data_full.jsonl ]; then

    echo "[DATA] Building shared SFT dataset..."

    python data_pipeline.py \
        --build-sft \
        --split train \
        --out outputs/sft_data_full.jsonl

else

    echo "[DATA] Shared SFT dataset already exists."

fi


if [ "$TRAIN_N" = "full" ]; then

    cp outputs/sft_data_full.jsonl "$OUT/sft_data.jsonl"

else

    head -n "$TRAIN_N" \
        outputs/sft_data_full.jsonl \
        > "$OUT/sft_data.jsonl"

fi

echo "[DATA] Training examples:"
wc -l "$OUT/sft_data.jsonl"


# ============================================================================
# HELPER: RUN vLLM EVALUATION
# ============================================================================

run_vllm_eval() {

    local MODEL_PATH="$1"
    local OUTPUT_PATH="$2"

    echo ""
    echo "----------------------------------------------------------------------"
    echo "vLLM EVALUATION"
    echo "Model:  $MODEL_PATH"
    echo "Output: $OUTPUT_PATH"
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


# ============================================================================
# HELPER: SCORE
# ============================================================================

run_score() {

    local PREDICTIONS="$1"
    local DETAILED="$2"
    local SUMMARY="$3"

    echo ""
    echo "----------------------------------------------------------------------"
    echo "SCORING"
    echo "Predictions: $PREDICTIONS"
    echo "----------------------------------------------------------------------"

    python evaluator.py \
        --score \
        --predictions "$PREDICTIONS" \
        --split test \
        --out-detailed "$DETAILED" \
        --out-summary "$SUMMARY"

    echo "[OK] Scoring completed."
}


# ============================================================================
# HELPER: LOG
# ============================================================================

run_log() {

    local RUN_ID="$1"
    local CONFIG="$2"
    local SUMMARY="$3"
    local BASELINE="$4"
    local NOTES="$5"

    echo ""
    echo "----------------------------------------------------------------------"
    echo "LEDGER"
    echo "Run: $RUN_ID"
    echo "----------------------------------------------------------------------"

    if [ -n "$BASELINE" ]; then

        python experiment_ledger.py \
            --log \
            --run_id "$RUN_ID" \
            --training_config "$CONFIG" \
            --eval_summary "$SUMMARY" \
            --ledger "$LEDGER" \
            --baseline_run_id "$BASELINE" \
            --notes "$NOTES"

    else

        python experiment_ledger.py \
            --log \
            --run_id "$RUN_ID" \
            --training_config "$CONFIG" \
            --eval_summary "$SUMMARY" \
            --ledger "$LEDGER" \
            --notes "$NOTES"

    fi

    echo "[OK] Ledger entry completed."
}


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
    {
        'model': '${MODEL}',
        'mode': 'baseline'
    },
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
# STAGE 1 - SIMPLE SFT
# ============================================================================

echo ""
echo "======================================================================"
echo "STAGE 1: SIMPLE SFT"
echo "======================================================================"

python sft_train.py \
    --model "$MODEL" \
    --data "$OUT/sft_data.jsonl" \
    --mode lora \
    --lora_r 16 \
    --lora_alpha 32 \
    --output_dir "$CKPT/stage1" \
    --epochs 5 \
    --save_every_epochs 1 \
    --per_device_batch_size 4 \
    --grad_accum 2


echo ""
echo "[STAGE 1] Final merged model:"
echo "           ${CKPT}/stage1_merged"


# ============================================================================
# STAGE 1 - FINAL EVALUATION
# ============================================================================

run_vllm_eval \
    "${CKPT}/stage1_merged" \
    "$OUT/predictions_stage1.jsonl"

run_score \
    "$OUT/predictions_stage1.jsonl" \
    "$OUT/eval_stage1_detailed.jsonl" \
    "$OUT/eval_stage1_summary.json"

run_log \
    "${MODEL_SLUG}__stage1_sft" \
    "${CKPT}/stage1_merged/training_config.json" \
    "$OUT/eval_stage1_summary.json" \
    "${MODEL_SLUG}__baseline" \
    "Simple SFT, LoRA, n=$TEST_N"


# ============================================================================
# STAGE 2 - DIAGNOSIS
# ============================================================================

echo ""
echo "======================================================================"
echo "STAGE 2: DIAGNOSIS"
echo "======================================================================"

python data_pipeline.py \
    --diagnose \
    --predictions "$OUT/predictions_stage1.jsonl" \
    --weak-report "$OUT/weak_clusters.json"

cat "$OUT/weak_clusters.json"

echo "[OK] Diagnosis completed."


# ============================================================================
# STAGE 2 - AUGMENTATION
# ============================================================================

# echo ""
# echo "======================================================================"
# echo "STAGE 2: AUGMENTATION"
# echo "======================================================================"

# echo ""
# echo "----------------------------------------------------------------------"
# echo "Generating semantic augmentation"
# echo "----------------------------------------------------------------------"

# python run_augmentation.py \
#     --stage semantic \
#     --model "${CKPT}/stage1_merged" \
#     --weak_report "$OUT/weak_clusters.json" \
#     --out "$OUT/semantic_aug.jsonl" \
#     --batch_size 64

# echo "[OK] Semantic augmentation completed."


# echo ""
# echo "----------------------------------------------------------------------"
# echo "Generating numeric augmentation"
# echo "----------------------------------------------------------------------"

# python run_augmentation.py \
#     --stage numeric \
#     --model "${CKPT}/stage1_merged" \
#     --weak_report "$OUT/weak_clusters.json" \
#     --out "$OUT/numeric_aug.jsonl" \
#     --n_per_problem 1 \
#     --votes 3 \
#     --batch_size 64

# echo "[OK] Numeric augmentation completed."


# ============================================================================
# STAGE 2 VARIANT FUNCTION
# ============================================================================

run_stage2_variant() {

    local VARIANT_NAME="$1"
    local EXTRA_DATA="${2:-}"

    local OUT_DIR="${CKPT}/${VARIANT_NAME}"

    echo ""
    echo "======================================================================"
    echo "STAGE 2 VARIANT: $VARIANT_NAME"
    echo "======================================================================"

    # ------------------------------------------------------------------------
    # TRAINING
    # ------------------------------------------------------------------------

    if [ -z "$EXTRA_DATA" ]; then

        echo "[STAGE 2] No augmentation."
        echo "[STAGE 2] Using simple skill-based replay only."

        python sft_train.py \
            --model "$MODEL" \
            --data "$OUT/sft_data.jsonl" \
            --replay_strategy skill \
            --replay_ratio 0.7 \
            --mode lora \
            --lora_r 16 \
            --lora_alpha 32 \
            --output_dir "$OUT_DIR" \
            --epochs 5 \
            --save_every_epochs 1 \
            --per_device_batch_size 4 \
            --grad_accum 2

    else

        echo "[STAGE 2] Extra augmentation data:"
        echo "          $EXTRA_DATA"

        python sft_train.py \
            --model "$MODEL" \
            --data "$OUT/sft_data.jsonl" \
            --extra_data $EXTRA_DATA \
            --replay_strategy skill \
            --replay_ratio 0.7 \
            --mode lora \
            --lora_r 16 \
            --lora_alpha 32 \
            --output_dir "$OUT_DIR" \
            --epochs 5 \
            --save_every_epochs 1 \
            --per_device_batch_size 4 \
            --grad_accum 2

    fi


    echo ""
    echo "[STAGE 2] Final merged model:"
    echo "           ${OUT_DIR}_merged"


    # ------------------------------------------------------------------------
    # FINAL EVALUATION
    # ------------------------------------------------------------------------

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
        "${MODEL_SLUG}__stage1_sft" \
        "SFT + ${VARIANT_NAME}, n=$TEST_N"

}


# ============================================================================
# STAGE 2 VARIANT 1 - SIMPLE REPLAY
# ============================================================================

run_stage2_variant \
    "simple_replay"


# ============================================================================
# STAGE 2 VARIANT 2 - SEMANTIC REPLAY
# ============================================================================

# run_stage2_variant \
#     "sem_replay" \
#     "$OUT/semantic_aug.jsonl"


# ============================================================================
# STAGE 2 VARIANT 3 - NUMERIC REPLAY
# ============================================================================

# run_stage2_variant \
#     "num_replay" \
#     "$OUT/numeric_aug.jsonl"


# ============================================================================
# STAGE 2 VARIANT 4 - SEMANTIC + NUMERIC REPLAY
# ============================================================================

# run_stage2_variant \
#     "both_replay" \
#     "$OUT/semantic_aug.jsonl $OUT/numeric_aug.jsonl"


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
        --num_generations 4 \
        --per_device_batch_size 4 \
        --grad_accum 1 \
        --num_train_epochs 5 \
        --w_correctness 1.0 \
        --w_format 0.2 \
        --w_persistence 0.15 \
        --w_chain_stability 0.25


    echo ""
    echo "[STAGE 3] Final merged model:"
    echo "           ${GRPO_DIR}_merged"


    # ------------------------------------------------------------------------
    # FINAL GRPO EVALUATION
    # ------------------------------------------------------------------------

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
        "${MODEL_SLUG}__baseline" \
        "SFT (${BASE_NAME}) + GRPO, n=$TEST_N"

}


# ============================================================================
# STAGE 3 - GRPO ON STAGE 1
# ============================================================================

run_stage3_grpo \
    "stage1" \
    "${CKPT}/stage1_merged"


# ============================================================================
# STAGE 3 - GRPO ON SIMPLE REPLAY
# ============================================================================

run_stage3_grpo \
    "simple_replay" \
    "${CKPT}/simple_replay_merged"


# # ============================================================================
# # STAGE 3 - GRPO ON SEMANTIC REPLAY
# # ============================================================================

# run_stage3_grpo \
#     "sem_replay" \
#     "${CKPT}/sem_replay_merged"


# # ============================================================================
# # STAGE 3 - GRPO ON NUMERIC REPLAY
# # ============================================================================

# run_stage3_grpo \
#     "num_replay" \
#     "${CKPT}/num_replay_merged"


# # ============================================================================
# # STAGE 3 - GRPO ON BOTH REPLAY
# # ============================================================================

# run_stage3_grpo \
#     "both_replay" \
#     "${CKPT}/both_replay_merged"


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
echo "  ${MODEL_SLUG}__stage1_sft"

echo ""
echo "Stage 2:"
echo "  ${MODEL_SLUG}__simple_replay"
echo "  ${MODEL_SLUG}__sem_replay"
echo "  ${MODEL_SLUG}__num_replay"
echo "  ${MODEL_SLUG}__both_replay"

echo ""
echo "Stage 3:"
echo "  ${MODEL_SLUG}__grpo_stage1"
echo "  ${MODEL_SLUG}__grpo_simple_replay"
echo "  ${MODEL_SLUG}__grpo_sem_replay"
echo "  ${MODEL_SLUG}__grpo_num_replay"
echo "  ${MODEL_SLUG}__grpo_both_replay"

echo ""
echo "Total final evaluation runs: 10"
echo "  1 Baseline"
echo "  1 Stage 1 SFT"
echo "  4 Stage 2 variants"
echo "  5 Stage 3 GRPO variants"

echo ""
echo "Checkpoint evaluations: NONE"

echo ""
echo "Ledger:"
echo "  $LEDGER"

echo "======================================================================"
```
