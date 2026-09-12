#!/usr/bin/env bash
# ============================================================================
# run_experiment_core.sh - SMART experiment ladder (shared core)
#
# Do not call this directly. Call one of the model wrappers, which set the
# model and HF repos and then source this file:
#     ./run_qwen_experiment.sh   [TRAIN_N] [TEST_N]
#     ./run_deepseek_experiment.sh [TRAIN_N] [TEST_N]
#
# Keeping the logic in ONE file is what guarantees the two models run an
# identical protocol - duplicating ~500 lines twice guarantees they drift.
#
# ORDER
#   1. Baseline inference + evaluation
#   2. ALL SFT variants first:
#        sft_skill        - SFT on Skill_MATH
#        simple_replay    - + skill replay
#        sem_replay       - + semantic augmentation + replay
#        num_replay       - + numeric augmentation + replay
#        both_replay      - + semantic & numeric augmentation + replay
#   3. THEN all GRPO runs, one per SFT variant (GRPO_EPOCHS, default 2)
#
#   Augmentation is generated right after sft_skill's evaluation, since
#   sft_skill's merged model is the generator for it.
#
# RESUME
#   Every step is skipped if it already completed. A step counts as complete
#   only when ALL of these exist:
#       - its run_id in outputs/experiment_ledger.jsonl
#       - its predictions_<name>.jsonl
#       - its eval_<name>_summary.json
#   A half-finished step (e.g. training done, evaluation crashed) is NOT
#   counted as complete and is re-run from its training stage - which is the
#   only safe interpretation, since a partial checkpoint cannot be trusted.
#   Re-running the same script therefore continues from where it stopped
#   instead of starting over.
#
#   Force a specific step to re-run:   FORCE_RERUN=sft_skill ./run_..._experiment.sh
#   Force everything:                  FORCE_RERUN=all       ./run_..._experiment.sh
#
# CHECKPOINTS
#   Adapter-only, one per epoch (--save_every_epochs 1), for both SFT and
#   GRPO. Adapter checkpoints cannot be loaded standalone by vLLM - merge one
#   on demand when you want to evaluate it:
#       python merge_lora.py --base_model <base> --adapter_path <ckpt>/checkpoint-N --out /tmp/m
#       python run_eval.py --model /tmp/m --split test --use_vllm --out preds.jsonl
#   or evaluate every checkpoint from a run in one pass:
#       python run_multi_checkpoint_eval.py --checkpoints_dir <ckpt> --base_model <base> --mode lora ...
#
# ATTENTION / INFERENCE
#   Training uses flash-attention (Hub kernel by default, never sdpa - see
#   hardware_utils.select_attn_implementation). All evaluation uses vLLM.
#
# HUGGING FACE
#   export HF_TOKEN=hf_xxxxxxxx
#   Uploads happen after EVERY SFT variant and EVERY GRPO run: model to its
#   family repo, plus predictions/logs/ledger to the outputs dataset repo.
#   NOTHING is deleted from local disk by this script.
#   export SMART_HF_DISABLE=1 to skip uploads entirely.
# ============================================================================
set -Eeuo pipefail
trap 'echo ""; echo "[ERROR] Pipeline failed at line $LINENO"; echo "[ERROR] Command: $BASH_COMMAND"; exit 1' ERR

: "${MODEL:?run_experiment_core.sh must be sourced from a wrapper that sets MODEL}"
: "${SMART_HF_SFT_REPO:?wrapper must set SMART_HF_SFT_REPO}"
: "${SMART_HF_REPLAY_REPO:?wrapper must set SMART_HF_REPLAY_REPO}"
: "${SMART_HF_GRPO_REPO:?wrapper must set SMART_HF_GRPO_REPO}"
: "${SMART_HF_OUTPUTS_REPO:?wrapper must set SMART_HF_OUTPUTS_REPO}"
export SMART_HF_SFT_REPO SMART_HF_REPLAY_REPO SMART_HF_GRPO_REPO SMART_HF_OUTPUTS_REPO

# Run from the repository root regardless of where the wrapper was invoked.
cd "$(dirname "${BASH_SOURCE[0]}")"

TRAIN_N="${TRAIN_N:-full}"
TEST_N="${TEST_N:-full}"
FORCE_RERUN="${FORCE_RERUN:-}"

MODEL_SLUG="$(echo "$MODEL" | tr '/' '_')"
OUT="outputs/${MODEL_SLUG}"
CKPT="ckpts/${MODEL_SLUG}"
LEDGER="outputs/experiment_ledger.jsonl"
mkdir -p "$OUT" "$CKPT"

# ============================================================================
# SETTINGS (all overridable from the environment)
# ============================================================================
EPOCHS="${EPOCHS:-4}"
SAVE_EVERY_EPOCHS="${SAVE_EVERY_EPOCHS:-1}"
PER_DEVICE_BATCH="${PER_DEVICE_BATCH:-4}"
GRAD_ACCUM="${GRAD_ACCUM:-2}"
LORA_R="${LORA_R:-16}"
LORA_ALPHA="${LORA_ALPHA:-32}"
REPLAY_RATIO="${REPLAY_RATIO:-0.7}"
REPLAY_MODE="${REPLAY_MODE:-additive}"

GRPO_EPOCHS="${GRPO_EPOCHS:-2}"          # capped at 2 per your spec
GRPO_GENERATIONS="${GRPO_GENERATIONS:-4}"
GRPO_DATA_LIMIT="${GRPO_DATA_LIMIT:-1000}"   # stratified by (subject, level)

VLLM_GPU_MEMORY="${VLLM_GPU_MEMORY:-0.90}"
VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-4096}"
VLLM_TENSOR_PARALLEL="${VLLM_TENSOR_PARALLEL:-1}"
VLLM_MAX_TOKENS="${VLLM_MAX_TOKENS:-2048}"

if [ "$TEST_N" = "full" ]; then LIMIT_ARG=(); else LIMIT_ARG=(--limit "$TEST_N"); fi
if [ "$TRAIN_N" = "full" ]; then AUG_SOURCE_ARG=(); else AUG_SOURCE_ARG=(--limit_source "$TRAIN_N"); fi
if [ -n "$GRPO_DATA_LIMIT" ]; then GRPO_LIMIT_ARG=(--data_limit "$GRPO_DATA_LIMIT"); else GRPO_LIMIT_ARG=(); fi

echo "============================================================"
echo "SMART EXPERIMENT LADDER"
echo "============================================================"
echo "MODEL:                 $MODEL"
echo "MODEL_SLUG:            $MODEL_SLUG"
echo "TRAIN_N / TEST_N:      $TRAIN_N / $TEST_N"
echo "ORDER:                 all SFT variants first, then all GRPO"
echo "ATTENTION:             ${SMART_ATTN_IMPL:-kernels-community/flash-attn2@v2} (never sdpa)"
echo "INFERENCE:             vLLM"
echo "SFT EPOCHS:            $EPOCHS  (checkpoint every $SAVE_EVERY_EPOCHS epoch, adapter-only)"
echo "GRPO EPOCHS:           $GRPO_EPOCHS (max)"
echo "GRPO DATA LIMIT:       ${GRPO_DATA_LIMIT:-<all>} (stratified by subject x level)"
echo "REPLAY MODE:           $REPLAY_MODE (ratio $REPLAY_RATIO)"
echo "RESUME:                enabled (completed steps are skipped)"
echo "FORCE_RERUN:           ${FORCE_RERUN:-<none>}"
echo "LOCAL DELETION:        disabled - nothing is removed from disk"
echo "------------------------------------------------------------"
echo "HF SFT REPO:           $SMART_HF_SFT_REPO"
echo "HF REPLAY REPO:        $SMART_HF_REPLAY_REPO"
echo "HF GRPO REPO:          $SMART_HF_GRPO_REPO"
echo "HF OUTPUTS DATASET:    $SMART_HF_OUTPUTS_REPO"
echo "HF TOKEN:              ${HF_TOKEN:+set}${HF_TOKEN:-NOT SET}"
echo "HF UPLOADS:            ${SMART_HF_DISABLE:+DISABLED}${SMART_HF_DISABLE:-enabled}"
echo "============================================================"

if [ -z "${HF_TOKEN:-}" ] && [ -z "${SMART_HF_DISABLE:-}" ]; then
    echo "[HF] WARNING: HF_TOKEN is not set - uploads will fail (the run itself continues,"
    echo "     artifacts stay on local disk). export HF_TOKEN=... or SMART_HF_DISABLE=1"
fi

# ============================================================================
# RESUME LOGIC
# ============================================================================
# A step is complete only if its ledger entry AND both eval artifacts exist.
step_is_complete() {
    local NAME="$1"
    local RUN_ID="${MODEL_SLUG}__${NAME}"

    if [ "$FORCE_RERUN" = "all" ] || [ "$FORCE_RERUN" = "$NAME" ]; then
        echo "[RESUME] FORCE_RERUN matches '$NAME' - re-running it"
        return 1
    fi

    local preds="$OUT/predictions_${NAME}.jsonl"
    local summary="$OUT/eval_${NAME}_summary.json"
    [ -s "$preds" ]   || return 1
    [ -s "$summary" ] || return 1
    [ -f "$LEDGER" ]  || return 1
    grep -q "\"run_id\": \"${RUN_ID}\"" "$LEDGER" || return 1
    return 0
}

skip_if_complete() {
    local NAME="$1"; local LABEL="$2"
    if step_is_complete "$NAME"; then
        echo ""
        echo "[RESUME] SKIP $LABEL - already complete"
        echo "         (ledger entry + predictions + eval summary all present)"
        return 0
    fi
    return 1
}

# ============================================================================
# HELPERS
# ============================================================================
run_vllm_eval() {
    local MODEL_PATH="$1"; local OUTPUT_PATH="$2"
    echo ""
    echo "----------------------------------------------------------------------"
    echo "vLLM EVALUATION  |  $MODEL_PATH"
    echo "----------------------------------------------------------------------"
    python run_eval.py --model "$MODEL_PATH" --split test "${LIMIT_ARG[@]}" --use_vllm \
        --gpu_memory_utilization "$VLLM_GPU_MEMORY" \
        --max_model_len "$VLLM_MAX_MODEL_LEN" \
        --tensor_parallel_size "$VLLM_TENSOR_PARALLEL" \
        --max_tokens "$VLLM_MAX_TOKENS" --out "$OUTPUT_PATH"
}

run_score() {
    python evaluator.py --score --predictions "$1" --split test \
        --out-detailed "$2" --out-summary "$3"
}

run_log() {
    local RUN_ID="$1"; local CONFIG="$2"; local SUMMARY="$3"
    local BASELINE="$4"; local NOTES="$5"
    if [ -n "$BASELINE" ]; then
        python experiment_ledger.py --log --run_id "$RUN_ID" \
            --training_config "$CONFIG" --eval_summary "$SUMMARY" \
            --ledger "$LEDGER" --baseline_run_id "$BASELINE" --notes "$NOTES"
    else
        python experiment_ledger.py --log --run_id "$RUN_ID" \
            --training_config "$CONFIG" --eval_summary "$SUMMARY" \
            --ledger "$LEDGER" --notes "$NOTES"
    fi
}

push_model_hf() {
    local LOCAL_DIR="$1"; local VARIANT="$2"
    if [ -n "${SMART_HF_DISABLE:-}" ]; then return 0; fi
    echo ""
    echo "[HF] pushing model $VARIANT -> Hub"
    python hf_sync.py --push-model --local "$LOCAL_DIR" \
        --model-slug "$MODEL_SLUG" --variant "$VARIANT" \
        || echo "[HF] WARNING: model push of $VARIANT failed - local copy intact, continuing"
}

push_outputs_hf() {
    if [ -n "${SMART_HF_DISABLE:-}" ]; then return 0; fi
    echo ""
    echo "[HF] pushing predictions / logs / ledger -> $SMART_HF_OUTPUTS_REPO"
    python hf_sync.py --push-outputs --model-slug "$MODEL_SLUG" \
        || echo "[HF] WARNING: outputs push failed - local artifacts intact, will retry next step"
}

# ============================================================================
# SFT STEP
#   $1 variant  $2 data file  $3 extra data ("" = none)
#   $4 replay yes/no  $5 ledger baseline run_id  $6 notes
# ============================================================================
run_sft_variant() {
    local NAME="$1"; local DATA_FILE="$2"; local EXTRA_DATA="${3:-}"
    local USE_REPLAY="${4:-no}"; local BASELINE_ID="$5"; local NOTES="$6"
    local OUT_DIR="${CKPT}/${NAME}"

    if skip_if_complete "$NAME" "SFT $NAME"; then return 0; fi

    echo ""
    echo "######################################################################"
    echo "# SFT: $NAME"
    echo "######################################################################"
    echo "[SFT] data: $DATA_FILE   extra: ${EXTRA_DATA:-<none>}   replay: $USE_REPLAY"

    local REPLAY_ARGS=()
    if [ "$USE_REPLAY" = "yes" ]; then
        REPLAY_ARGS=(--replay_strategy skill --replay_ratio "$REPLAY_RATIO" --replay_mode "$REPLAY_MODE")
    fi
    local EXTRA_ARGS=()
    if [ -n "$EXTRA_DATA" ]; then EXTRA_ARGS=(--extra_data $EXTRA_DATA); fi

    python sft_train.py \
        --model "$MODEL" --data "$DATA_FILE" \
        "${EXTRA_ARGS[@]}" "${REPLAY_ARGS[@]}" \
        --mode lora --lora_r "$LORA_R" --lora_alpha "$LORA_ALPHA" \
        --output_dir "$OUT_DIR" --epochs "$EPOCHS" \
        --save_every_epochs "$SAVE_EVERY_EPOCHS" \
        --per_device_batch_size "$PER_DEVICE_BATCH" --grad_accum "$GRAD_ACCUM"

    run_vllm_eval "${OUT_DIR}_merged" "$OUT/predictions_${NAME}.jsonl"
    run_score "$OUT/predictions_${NAME}.jsonl" \
              "$OUT/eval_${NAME}_detailed.jsonl" "$OUT/eval_${NAME}_summary.json"
    run_log "${MODEL_SLUG}__${NAME}" "${OUT_DIR}_merged/training_config.json" \
            "$OUT/eval_${NAME}_summary.json" "$BASELINE_ID" "$NOTES"

    push_model_hf "${OUT_DIR}_merged" "$NAME"
    push_outputs_hf
}

# ============================================================================
# GRPO STEP
# ============================================================================
run_grpo_variant() {
    local BASE_NAME="$1"
    local BASE_MODEL_DIR="${CKPT}/${BASE_NAME}_merged"
    local GRPO_DIR="${CKPT}/grpo_${BASE_NAME}"
    local NAME="grpo_${BASE_NAME}"

    if skip_if_complete "$NAME" "GRPO on $BASE_NAME"; then return 0; fi

    if [ ! -d "$BASE_MODEL_DIR" ]; then
        echo "[GRPO] SKIP $NAME - base model $BASE_MODEL_DIR not found on disk."
        echo "       Its SFT step must complete first (or be restored from the Hub:"
        echo "       python hf_sync.py --pull-model --model-slug $MODEL_SLUG --variant $BASE_NAME --dest ckpts)"
        return 0
    fi

    echo ""
    echo "######################################################################"
    echo "# GRPO on: $BASE_NAME  (${GRPO_EPOCHS} epochs)"
    echo "######################################################################"

    python grpo_train.py \
        --model "$BASE_MODEL_DIR" --data "$OUT/sft_data.jsonl" \
        --output_dir "$GRPO_DIR" \
        --num_generations "$GRPO_GENERATIONS" \
        "${GRPO_LIMIT_ARG[@]}" \
        --per_device_batch_size 4 --grad_accum 1 \
        --num_train_epochs "$GRPO_EPOCHS" \
        --w_correctness 1.0 --w_format 0.2 --w_persistence 0.15 --w_chain_stability 0.25

    run_vllm_eval "${GRPO_DIR}_merged" "$OUT/predictions_${NAME}.jsonl"
    run_score "$OUT/predictions_${NAME}.jsonl" \
              "$OUT/eval_${NAME}_detailed.jsonl" "$OUT/eval_${NAME}_summary.json"
    run_log "${MODEL_SLUG}__${NAME}" "${GRPO_DIR}_merged/training_config.json" \
            "$OUT/eval_${NAME}_summary.json" "${MODEL_SLUG}__${BASE_NAME}" \
            "GRPO on ${BASE_NAME}, ${GRPO_EPOCHS} epochs, n=$TEST_N"

    push_model_hf "${GRPO_DIR}_merged" "$NAME"
    push_outputs_hf
}

# ============================================================================
# DATA
# ============================================================================
echo ""
echo "======================================================================"
echo "PREPARING DATA"
echo "======================================================================"
if [ ! -s outputs/sft_data_full.jsonl ]; then
    python data_pipeline.py --build-sft --split train --out outputs/sft_data_full.jsonl
else
    echo "[DATA] Skill_MATH SFT dataset already exists - reusing."
fi
if [ "$TRAIN_N" = "full" ]; then
    cp outputs/sft_data_full.jsonl "$OUT/sft_data.jsonl"
else
    head -n "$TRAIN_N" outputs/sft_data_full.jsonl > "$OUT/sft_data.jsonl"
fi
wc -l "$OUT/sft_data.jsonl"

# ============================================================================
# 1. BASELINE
# ============================================================================
if ! skip_if_complete "baseline" "BASELINE"; then
    echo ""
    echo "######################################################################"
    echo "# BASELINE"
    echo "######################################################################"
    run_vllm_eval "$MODEL" "$OUT/predictions_baseline.jsonl"
    run_score "$OUT/predictions_baseline.jsonl" \
              "$OUT/eval_baseline_detailed.jsonl" "$OUT/eval_baseline_summary.json"
    python3 -c "
import json
json.dump({'model': '${MODEL}', 'mode': 'baseline'}, open('${OUT}/baseline_config.json','w'), indent=2)
"
    run_log "${MODEL_SLUG}__baseline" "$OUT/baseline_config.json" \
            "$OUT/eval_baseline_summary.json" "" "vanilla, untrained, n=$TEST_N"
    push_outputs_hf
fi

# ============================================================================
# 2. ALL SFT VARIANTS
# ============================================================================
run_sft_variant "sft_skill" "$OUT/sft_data.jsonl" "" "no" \
    "${MODEL_SLUG}__baseline" "SFT on Skill_MATH, n=$TEST_N"

# Augmentation needs sft_skill's merged model as the generator.
if [ ! -s "$OUT/semantic_aug.jsonl" ] || [ ! -s "$OUT/numeric_aug.jsonl" ]; then
    echo ""
    echo "######################################################################"
    echo "# DIAGNOSIS + AUGMENTATION GENERATION"
    echo "######################################################################"
    if [ ! -s "$OUT/weak_clusters.json" ]; then
        python data_pipeline.py --diagnose \
            --predictions "$OUT/predictions_sft_skill.jsonl" \
            --weak-report "$OUT/weak_clusters.json"
    else
        echo "[AUG] weak_clusters.json exists - reusing."
    fi
    cat "$OUT/weak_clusters.json"

    if [ ! -s "$OUT/semantic_aug.jsonl" ]; then
        python run_augmentation.py --stage semantic --model "${CKPT}/sft_skill_merged" \
            --weak_report "$OUT/weak_clusters.json" --out "$OUT/semantic_aug.jsonl" \
            "${AUG_SOURCE_ARG[@]}" --batch_size 64
    else
        echo "[AUG] semantic_aug.jsonl exists - reusing."
    fi

    if [ ! -s "$OUT/numeric_aug.jsonl" ]; then
        python run_augmentation.py --stage numeric --model "${CKPT}/sft_skill_merged" \
            --weak_report "$OUT/weak_clusters.json" --out "$OUT/numeric_aug.jsonl" \
            --n_per_problem 1 --votes 3 "${AUG_SOURCE_ARG[@]}" --batch_size 64
    else
        echo "[AUG] numeric_aug.jsonl exists - reusing."
    fi
    push_outputs_hf
else
    echo ""
    echo "[RESUME] SKIP augmentation generation - both augmented files already exist"
fi

run_sft_variant "simple_replay" "$OUT/sft_data.jsonl" "" "yes" \
    "${MODEL_SLUG}__sft_skill" "Skill_MATH + simple skill replay, n=$TEST_N"

run_sft_variant "sem_replay" "$OUT/sft_data.jsonl" "$OUT/semantic_aug.jsonl" "yes" \
    "${MODEL_SLUG}__sft_skill" "Skill_MATH + semantic augmentation + replay, n=$TEST_N"

run_sft_variant "num_replay" "$OUT/sft_data.jsonl" "$OUT/numeric_aug.jsonl" "yes" \
    "${MODEL_SLUG}__sft_skill" "Skill_MATH + numeric augmentation + replay, n=$TEST_N"

run_sft_variant "both_replay" "$OUT/sft_data.jsonl" \
    "$OUT/semantic_aug.jsonl $OUT/numeric_aug.jsonl" "yes" \
    "${MODEL_SLUG}__sft_skill" "Skill_MATH + semantic & numeric augmentation + replay, n=$TEST_N"

# ============================================================================
# 3. ALL GRPO RUNS
# ============================================================================
echo ""
echo "======================================================================"
echo "GRPO PHASE (${GRPO_EPOCHS} epochs each)"
echo "======================================================================"
run_grpo_variant "sft_skill"
run_grpo_variant "simple_replay"
run_grpo_variant "sem_replay"
run_grpo_variant "num_replay"
run_grpo_variant "both_replay"

# ============================================================================
# REPORTS + FINAL UPLOAD
# ============================================================================
echo ""
echo "======================================================================"
echo "REPORTS"
echo "======================================================================"
python experiment_ledger.py --print --ledger "$LEDGER"
python generate_all_reports.py --ledger "$LEDGER" --out_dir outputs/report
push_outputs_hf

echo ""
echo "======================================================================"
echo "COMPLETE: $MODEL"
echo "======================================================================"
echo "Runs: ${MODEL_SLUG}__{baseline,sft_skill,simple_replay,sem_replay,num_replay,both_replay}"
echo "      ${MODEL_SLUG}__grpo_{sft_skill,simple_replay,sem_replay,num_replay,both_replay}"
echo "Total: 11 evaluation runs (1 baseline + 5 SFT + 5 GRPO)"
echo ""
echo "Per-epoch adapter checkpoints kept under $CKPT/<variant>/checkpoint-*"
echo "Nothing was deleted from local disk."
echo ""
echo "Ledger:  $LEDGER"
echo "Report:  outputs/report/"
echo "HF:      https://huggingface.co/$SMART_HF_SFT_REPO"
echo "         https://huggingface.co/$SMART_HF_REPLAY_REPO"
echo "         https://huggingface.co/$SMART_HF_GRPO_REPO"
echo "         https://huggingface.co/datasets/$SMART_HF_OUTPUTS_REPO"
echo "======================================================================"
