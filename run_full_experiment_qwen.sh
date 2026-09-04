#!/usr/bin/env bash
# ============================================================================
# run_full_experiment.sh - SMART experiment ladder, one variant at a time
#
#   For each variant, in this exact order, completing fully before the next
#   variant starts:
#
#       Train SFT -> Evaluate SFT -> Train GRPO -> Evaluate GRPO ->
#       Upload SFT + GRPO to HF -> Delete BOTH locally -> next variant
#
#   Variants, in order:
#       1. sft_skill      (SFT on Skill_MATH)
#       2. simple_replay
#       3. sem_replay      (semantic augmentation + replay)
#       4. num_replay       (numeric augmentation + replay)
#       5. both_replay      (semantic + numeric augmentation + replay)
#
#   Total: 10 evaluation runs (1 baseline + 5 SFT + 5 GRPO... wait, actually
#   1 baseline + 1 sft_skill + 4 replay SFT + 5 GRPO = 11? No: baseline(1) +
#   sft_skill(1) + simple/sem/num/both replay(4) + grpo on each of those
#   5(5) = 11 total. See the completion banner at the end for the exact list.
#
# WHY THIS ORDER (as opposed to "all SFT, then all GRPO")
#   Running every SFT variant before any GRPO run means up to four merged 7B
#   checkpoints exist on disk simultaneously before ANY of them are cleaned
#   up. Completing one variant's full lifecycle (SFT -> GRPO -> upload ->
#   delete) before starting the next caps peak local disk usage to roughly
#   ONE variant's worth of checkpoints at any given time.
#
#   ONE EXCEPTION: augmentation generation (Stage 2, semantic + numeric) needs
#   sft_skill's merged model as the generator, so it runs between sft_skill's
#   own SFT-eval and its own GRPO step, WHILE that checkpoint still exists -
#   not after, when it would already be deleted. This is the only reason
#   sft_skill's lifecycle looks slightly different from the other four.
#
# DISK-SPACE MANAGEMENT / SAFETY GUARANTEE
#   A checkpoint is deleted ONLY if its Hub push actually succeeded, tracked
#   via a real per-variant PUSH_OK status (not swallowed by `|| echo`). If a
#   push fails, or SMART_HF_DISABLE=1, that pair's local checkpoints are kept.
#   Deleting the only copy of a model that never reached the Hub would be
#   real data loss, not a disk-space optimisation.
#
#   Predictions, eval summaries, and the ledger are SMALL and are kept
#   locally for the entire run (needed for the final report) - only the large
#   model checkpoints and shared training-data files are cleaned up. Outputs
#   (predictions/ledger/report) are pushed to the HF dataset repo after every
#   variant AND at the very end, so a crash mid-run still leaves results-so-
#   far safely on the Hub, not just on local disk.
#
# NOTES
#   - Training (SFT/GRPO) always uses flash-attention-2. No SDPA fallback.
#   - All inference/evaluation uses vLLM via --use_vllm.
#   - End evaluation only - no per-checkpoint evaluation in this script.
#   - The TEST split is the ORIGINAL Hendrycks MATH test set (no skill
#     annotations - skill-prediction/skill-usage report N/A there;
#     final-answer accuracy, format compliance, arithmetic consistency all
#     compute normally).
#
# HUGGING FACE
#   export HF_TOKEN=hf_xxxxxxxx      # required for uploads AND for cleanup
#                                    # to trigger (see safety guarantee above)
#   export SMART_HF_DISABLE=1        # skip all uploads AND all cleanup
#
# Usage:
#   ./run_full_experiment.sh <MODEL> [TRAIN_N] [TEST_N]
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
export SMART_HF_SFT_REPO="${SMART_HF_SFT_REPO:-HusnainAmjad/Qwen_2.5_Math_7b_SFT}"
export SMART_HF_REPLAY_REPO="${SMART_HF_REPLAY_REPO:-HusnainAmjad/Qwen_2.5_Math_7b_Replay_SFT}"
export SMART_HF_GRPO_REPO="${SMART_HF_GRPO_REPO:-HusnainAmjad/Qwen_2.5_Math_7b_GRPO}"
export SMART_HF_OUTPUTS_REPO="${SMART_HF_OUTPUTS_REPO:-HusnainAmjad/Qwen_2.5_Math_7b_Outputs}"

if [ -z "${HF_TOKEN:-}" ] && [ -z "${SMART_HF_DISABLE:-}" ]; then
    echo "[HF] WARNING: HF_TOKEN is not set. Uploads will fail, which means disk"
    echo "     cleanup will ALSO be skipped for every variant (by design). Either:"
    echo "       export HF_TOKEN=hf_xxxxxxxxxxxxxxxx"
    echo "       export SMART_HF_DISABLE=1"
fi

# ============================================================================
# TRAINING SETTINGS
# ============================================================================
EPOCHS="${EPOCHS:-4}"
SAVE_EVERY_EPOCHS="${SAVE_EVERY_EPOCHS:-1}"
PER_DEVICE_BATCH="${PER_DEVICE_BATCH:-4}"
GRAD_ACCUM="${GRAD_ACCUM:-2}"
LORA_R="${LORA_R:-16}"
LORA_ALPHA="${LORA_ALPHA:-32}"
REPLAY_RATIO="${REPLAY_RATIO:-0.7}"
REPLAY_MODE="${REPLAY_MODE:-additive}"
GRPO_EPOCHS="${GRPO_EPOCHS:-4}"
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
echo "SMART EXPERIMENT LADDER - one variant fully completes before the next starts"
echo "============================================================"
echo "MODEL:                 $MODEL"
echo "MODEL_SLUG:            $MODEL_SLUG"
echo "TRAIN_N / TEST_N:      $TRAIN_N / $TEST_N"
echo "ATTENTION:             flash_attention_2 (forced, no sdpa)"
echo "INFERENCE:             vLLM"
echo "CHECKPOINT EVAL:       none (end evaluation only)"
echo "SFT / GRPO EPOCHS:     $EPOCHS / $GRPO_EPOCHS"
echo "REPLAY MODE:           $REPLAY_MODE (ratio $REPLAY_RATIO)"
echo "------------------------------------------------------------"
echo "HF SFT REPO:           $SMART_HF_SFT_REPO"
echo "HF REPLAY REPO:        $SMART_HF_REPLAY_REPO"
echo "HF GRPO REPO:          $SMART_HF_GRPO_REPO"
echo "HF OUTPUTS DATASET:    $SMART_HF_OUTPUTS_REPO"
echo "HF TOKEN:              ${HF_TOKEN:+set}${HF_TOKEN:-NOT SET}"
echo "HF UPLOADS:            ${SMART_HF_DISABLE:+DISABLED}${SMART_HF_DISABLE:-enabled}"
echo "DISK CLEANUP:          ${SMART_HF_DISABLE:+DISABLED (uploads off)}${SMART_HF_DISABLE:-enabled, gated on successful push}"
echo "============================================================"

# ============================================================================
# HELPERS
# ============================================================================
run_vllm_eval() {
    local MODEL_PATH="$1"; local OUTPUT_PATH="$2"
    echo ""
    echo "----------------------------------------------------------------------"
    echo "vLLM EVALUATION  |  model: $MODEL_PATH"
    echo "----------------------------------------------------------------------"
    python run_eval.py \
        --model "$MODEL_PATH" --split test "${LIMIT_ARG[@]}" --use_vllm \
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
    python evaluator.py --score --predictions "$PREDICTIONS" --split test \
        --out-detailed "$DETAILED" --out-summary "$SUMMARY"
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

# --- Push tracking: real per-variant status, populated ONLY here ----------
declare -A PUSH_OK

push_model_hf() {
    local LOCAL_DIR="$1"; local VARIANT="$2"
    if [ -n "${SMART_HF_DISABLE:-}" ]; then
        echo "[HF] uploads disabled - skipping push of $VARIANT (local copy will be KEPT)"
        PUSH_OK["$VARIANT"]="no"
        return 0
    fi
    echo ""
    echo "[HF] pushing $VARIANT -> Hub"
    if python hf_sync.py --push-model --local "$LOCAL_DIR" \
            --model-slug "$MODEL_SLUG" --variant "$VARIANT"; then
        echo "[HF] push of $VARIANT: SUCCESS"
        PUSH_OK["$VARIANT"]="yes"
    else
        echo "[HF] push of $VARIANT: FAILED - local checkpoint kept, NOT deleting"
        PUSH_OK["$VARIANT"]="no"
    fi
}

# Pushes predictions/eval-summaries/ledger (report/tables/figures too, once
# they exist) to the outputs dataset repo. Safe to call repeatedly - patterns
# that don't exist yet (e.g. the report, before it's generated) are skipped
# with a printed note, not an error. Called after every variant so a crash
# mid-run still leaves results-so-far on the Hub, not just on local disk.
push_outputs_progress() {
    if [ -n "${SMART_HF_DISABLE:-}" ]; then
        return 0
    fi
    echo ""
    echo "[HF] pushing predictions/ledger progress so far -> $SMART_HF_OUTPUTS_REPO"
    python hf_sync.py --push-outputs --model-slug "$MODEL_SLUG" \
        || echo "[HF] WARNING: progress push failed - local artifacts are intact, will retry at the end"
}

# Deletes an SFT variant's checkpoints AND its GRPO variant's checkpoints from
# LOCAL disk (adapter dir + merged dir, both) - ONLY if both pushes for this
# pair are confirmed successful. Never touches the Hugging Face copy.
cleanup_variant_pair() {
    local SFT_VARIANT="$1"; local GRPO_VARIANT="$2"
    if [ -n "${SMART_HF_DISABLE:-}" ]; then
        echo "[CLEANUP] skipped for $SFT_VARIANT/$GRPO_VARIANT - HF uploads are disabled,"
        echo "          so nothing has a remote copy yet. Keeping local checkpoints."
        return 0
    fi
    local sft_status="${PUSH_OK[$SFT_VARIANT]:-no}"
    local grpo_status="${PUSH_OK[$GRPO_VARIANT]:-no}"
    if [ "$sft_status" != "yes" ] || [ "$grpo_status" != "yes" ]; then
        echo ""
        echo "[CLEANUP] SKIPPED for $SFT_VARIANT / $GRPO_VARIANT."
        echo "          $SFT_VARIANT push: $sft_status   $GRPO_VARIANT push: $grpo_status"
        echo "          At least one push did not succeed - keeping BOTH local checkpoints."
        echo "          Re-run: python hf_sync.py --push-model ... manually once fixed."
        return 0
    fi
    echo ""
    echo "[CLEANUP] $SFT_VARIANT and $GRPO_VARIANT are both confirmed on the Hub - freeing local disk"
    du -sh "${CKPT}/${SFT_VARIANT}" "${CKPT}/${SFT_VARIANT}_merged" \
          "${CKPT}/${GRPO_VARIANT}" "${CKPT}/${GRPO_VARIANT}_merged" 2>/dev/null || true
    rm -rf "${CKPT}/${SFT_VARIANT}" "${CKPT}/${SFT_VARIANT}_merged" \
           "${CKPT}/${GRPO_VARIANT}" "${CKPT}/${GRPO_VARIANT}_merged"
    echo "[CLEANUP] done - $SFT_VARIANT and $GRPO_VARIANT removed from local disk."
    df -h . 2>/dev/null | tail -1 || true
}

# ============================================================================
# SFT STEP (train -> eval -> log -> push). Does NOT clean up - that happens
# after this variant's paired GRPO step completes (see run_variant below).
# ============================================================================
run_sft_step() {
    local VARIANT_NAME="$1"; local DATA_FILE="$2"; local EXTRA_DATA="${3:-}"
    local USE_REPLAY="${4:-no}"; local BASELINE_ID="$5"; local NOTES="$6"
    local OUT_DIR="${CKPT}/${VARIANT_NAME}"

    echo ""
    echo "======================================================================"
    echo "TRAIN SFT: $VARIANT_NAME"
    echo "======================================================================"
    echo "[SFT] data: $DATA_FILE   extra: ${EXTRA_DATA:-<none>}   replay: $USE_REPLAY"

    local REPLAY_ARGS=()
    if [ "$USE_REPLAY" = "yes" ]; then
        REPLAY_ARGS=(--replay_strategy skill --replay_ratio "$REPLAY_RATIO" --replay_mode "$REPLAY_MODE")
    fi
    local EXTRA_ARGS=()
    if [ -n "$EXTRA_DATA" ]; then
        EXTRA_ARGS=(--extra_data $EXTRA_DATA)
    fi

    python sft_train.py \
        --model "$MODEL" --data "$DATA_FILE" \
        "${EXTRA_ARGS[@]}" "${REPLAY_ARGS[@]}" \
        --mode lora --lora_r "$LORA_R" --lora_alpha "$LORA_ALPHA" \
        --output_dir "$OUT_DIR" --epochs "$EPOCHS" \
        --save_every_epochs "$SAVE_EVERY_EPOCHS" \
        --per_device_batch_size "$PER_DEVICE_BATCH" --grad_accum "$GRAD_ACCUM"

    echo ""
    echo "EVALUATE SFT: $VARIANT_NAME"
    run_vllm_eval "${OUT_DIR}_merged" "$OUT/predictions_${VARIANT_NAME}.jsonl"
    run_score "$OUT/predictions_${VARIANT_NAME}.jsonl" \
              "$OUT/eval_${VARIANT_NAME}_detailed.jsonl" "$OUT/eval_${VARIANT_NAME}_summary.json"
    run_log "${MODEL_SLUG}__${VARIANT_NAME}" \
            "${OUT_DIR}_merged/training_config.json" \
            "$OUT/eval_${VARIANT_NAME}_summary.json" "$BASELINE_ID" "$NOTES"

    push_model_hf "${OUT_DIR}_merged" "$VARIANT_NAME"
}

# ============================================================================
# GRPO STEP (train -> eval -> log -> push). Does NOT clean up on its own -
# caller (run_variant) does that once both this and the paired SFT push are
# confirmed.
# ============================================================================
run_grpo_step() {
    local BASE_NAME="$1"; local BASE_MODEL_DIR="$2"
    local GRPO_DIR="${CKPT}/grpo_${BASE_NAME}"
    local GRPO_VARIANT="grpo_${BASE_NAME}"

    echo ""
    echo "======================================================================"
    echo "TRAIN GRPO: on $BASE_NAME"
    echo "======================================================================"
    python grpo_train.py \
        --model "$BASE_MODEL_DIR" --data "$OUT/sft_data.jsonl" \
        --output_dir "$GRPO_DIR" \
        --num_generations "$GRPO_GENERATIONS" \
        --per_device_batch_size 4 --grad_accum 1 \
        --num_train_epochs "$GRPO_EPOCHS" \
        --w_correctness 1.0 --w_format 0.2 --w_persistence 0.15 --w_chain_stability 0.25

    echo ""
    echo "EVALUATE GRPO: on $BASE_NAME"
    run_vllm_eval "${GRPO_DIR}_merged" "$OUT/predictions_grpo_${BASE_NAME}.jsonl"
    run_score "$OUT/predictions_grpo_${BASE_NAME}.jsonl" \
              "$OUT/eval_grpo_${BASE_NAME}_detailed.jsonl" \
              "$OUT/eval_grpo_${BASE_NAME}_summary.json"
    run_log "${MODEL_SLUG}__grpo_${BASE_NAME}" \
            "${GRPO_DIR}_merged/training_config.json" \
            "$OUT/eval_grpo_${BASE_NAME}_summary.json" \
            "${MODEL_SLUG}__${BASE_NAME}" "GRPO on ${BASE_NAME}, n=$TEST_N"

    push_model_hf "${GRPO_DIR}_merged" "$GRPO_VARIANT"
}

# ============================================================================
# FULL VARIANT LIFECYCLE:
#   Train SFT -> Evaluate SFT -> [optional: diagnose+augment, sft_skill only]
#   -> Train GRPO -> Evaluate GRPO -> Upload SFT+GRPO -> Delete both
#   -> push progress -> next variant
# ============================================================================
run_variant() {
    local VARIANT_NAME="$1"; local DATA_FILE="$2"; local EXTRA_DATA="${3:-}"
    local USE_REPLAY="${4:-no}"; local BASELINE_ID="$5"; local NOTES="$6"

    echo ""
    echo "############################################################"
    echo "# VARIANT: $VARIANT_NAME"
    echo "############################################################"

    run_sft_step "$VARIANT_NAME" "$DATA_FILE" "$EXTRA_DATA" "$USE_REPLAY" "$BASELINE_ID" "$NOTES"

    # ------------------------------------------------------------------------
    # sft_skill ONLY: generate diagnosis + augmentation now, while this
    # variant's merged checkpoint still exists on disk - it is the generator
    # model for both. This MUST happen before this variant is cleaned up
    # below, since cleanup deletes exactly the checkpoint augmentation needs.
    # ------------------------------------------------------------------------
    if [ "$VARIANT_NAME" = "sft_skill" ]; then
        echo ""
        echo "======================================================================"
        echo "DIAGNOSIS + AUGMENTATION GENERATION (using sft_skill's merged model"
        echo "while it still exists locally - before this variant's own cleanup)"
        echo "======================================================================"
        python data_pipeline.py --diagnose \
            --predictions "$OUT/predictions_sft_skill.jsonl" \
            --weak-report "$OUT/weak_clusters.json"
        cat "$OUT/weak_clusters.json"

        echo "[AUG] semantic..."
        python run_augmentation.py --stage semantic --model "${CKPT}/sft_skill_merged" \
            --weak_report "$OUT/weak_clusters.json" --out "$OUT/semantic_aug.jsonl" \
            "${AUG_SOURCE_ARG[@]}" --batch_size 64
        echo "[OK] Semantic augmentation completed."

        echo "[AUG] numeric..."
        python run_augmentation.py --stage numeric --model "${CKPT}/sft_skill_merged" \
            --weak_report "$OUT/weak_clusters.json" --out "$OUT/numeric_aug.jsonl" \
            --n_per_problem 1 --votes 3 "${AUG_SOURCE_ARG[@]}" --batch_size 64
        echo "[OK] Numeric augmentation completed."
    fi

    run_grpo_step "$VARIANT_NAME" "${CKPT}/${VARIANT_NAME}_merged"

    cleanup_variant_pair "$VARIANT_NAME" "grpo_${VARIANT_NAME}"
    push_outputs_progress
}

# ============================================================================
# DATA PREPARATION
# ============================================================================
echo ""
echo "======================================================================"
echo "PREPARING DATA"
echo "======================================================================"
if [ ! -f outputs/sft_data_full.jsonl ]; then
    echo "[DATA] Building Skill_MATH SFT dataset..."
    python data_pipeline.py --build-sft --split train --out outputs/sft_data_full.jsonl
else
    echo "[DATA] Skill_MATH SFT dataset already exists."
fi
if [ "$TRAIN_N" = "full" ]; then
    cp outputs/sft_data_full.jsonl "$OUT/sft_data.jsonl"
else
    head -n "$TRAIN_N" outputs/sft_data_full.jsonl > "$OUT/sft_data.jsonl"
fi
echo "[DATA] Skill_MATH training examples:"; wc -l "$OUT/sft_data.jsonl"

# ============================================================================
# STAGE 0 - BASELINE  (no local checkpoint - nothing to clean up)
# ============================================================================
echo ""
echo "======================================================================"
echo "STAGE 0: BASELINE"
echo "======================================================================"
run_vllm_eval "$MODEL" "$OUT/predictions_baseline.jsonl"
run_score "$OUT/predictions_baseline.jsonl" \
          "$OUT/eval_baseline_detailed.jsonl" "$OUT/eval_baseline_summary.json"
python3 -c "
import json
json.dump({'model': '${MODEL}', 'mode': 'baseline'}, open('${OUT}/baseline_config.json', 'w'), indent=2)
"
run_log "${MODEL_SLUG}__baseline" \
        "$OUT/baseline_config.json" "$OUT/eval_baseline_summary.json" \
        "" "vanilla, untrained, n=$TEST_N"
push_outputs_progress

# ============================================================================
# VARIANTS - each one fully completes (SFT -> GRPO -> upload -> delete)
# before the next one starts.
# ============================================================================
run_variant "sft_skill" \
    "$OUT/sft_data.jsonl" "" "no" \
    "${MODEL_SLUG}__baseline" "SFT on Skill_MATH (skill-labeled), n=$TEST_N"

run_variant "simple_replay" \
    "$OUT/sft_data.jsonl" "" "yes" \
    "${MODEL_SLUG}__sft_skill" "Skill_MATH + simple skill replay, n=$TEST_N"

run_variant "sem_replay" \
    "$OUT/sft_data.jsonl" "$OUT/semantic_aug.jsonl" "yes" \
    "${MODEL_SLUG}__sft_skill" "Skill_MATH + semantic augmentation + replay, n=$TEST_N"

run_variant "num_replay" \
    "$OUT/sft_data.jsonl" "$OUT/numeric_aug.jsonl" "yes" \
    "${MODEL_SLUG}__sft_skill" "Skill_MATH + numeric augmentation + replay, n=$TEST_N"

run_variant "both_replay" \
    "$OUT/sft_data.jsonl" "$OUT/semantic_aug.jsonl $OUT/numeric_aug.jsonl" "yes" \
    "${MODEL_SLUG}__sft_skill" "Skill_MATH + semantic & numeric augmentation + replay, n=$TEST_N"

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
# HUGGING FACE - FINAL PUSH OF ALL ARTIFACTS (now including the report)
# ============================================================================
if [ -z "${SMART_HF_DISABLE:-}" ]; then
    echo ""
    echo "======================================================================"
    echo "FINAL PUSH TO $SMART_HF_OUTPUTS_REPO"
    echo "======================================================================"
    if python hf_sync.py --push-outputs --model-slug "$MODEL_SLUG"; then
        echo "[HF] final outputs push: SUCCESS"
        echo ""
        echo "[CLEANUP] final sweep - all models and results confirmed on the Hub"
        du -sh "$OUT" 2>/dev/null || true
        rm -f "$OUT/sft_data.jsonl" "$OUT/semantic_aug.jsonl" "$OUT/numeric_aug.jsonl"
        echo "[CLEANUP] large shared data files removed. Predictions, eval summaries,"
        echo "          and the ledger are kept locally as your reference copies."
    else
        echo "[HF] WARNING: final outputs push failed - local artifacts are intact"
    fi
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
echo "Baseline:  ${MODEL_SLUG}__baseline"
echo ""
echo "Each variant (SFT -> GRPO -> upload -> delete, in this order):"
echo "  ${MODEL_SLUG}__sft_skill        + ${MODEL_SLUG}__grpo_sft_skill"
echo "  ${MODEL_SLUG}__simple_replay    + ${MODEL_SLUG}__grpo_simple_replay"
echo "  ${MODEL_SLUG}__sem_replay       + ${MODEL_SLUG}__grpo_sem_replay"
echo "  ${MODEL_SLUG}__num_replay       + ${MODEL_SLUG}__grpo_num_replay"
echo "  ${MODEL_SLUG}__both_replay      + ${MODEL_SLUG}__grpo_both_replay"
echo ""
echo "Total evaluation runs: 11  (1 baseline + 5 SFT + 5 GRPO)"
echo "Checkpoint evaluations: none (end evaluation only)"
echo "Local disk: cleaned after EACH variant's full lifecycle, gated on"
echo "            confirmed successful Hub upload."
echo ""
echo "Ledger:      $LEDGER"
echo "Report:      outputs/report/"
echo "HF SFT:      https://huggingface.co/$SMART_HF_SFT_REPO"
echo "HF Replay:   https://huggingface.co/$SMART_HF_REPLAY_REPO"
echo "HF GRPO:     https://huggingface.co/$SMART_HF_GRPO_REPO"
echo "HF Outputs:  https://huggingface.co/datasets/$SMART_HF_OUTPUTS_REPO"
echo "======================================================================"
