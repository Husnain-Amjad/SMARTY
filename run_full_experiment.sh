#!/usr/bin/env bash
# ============================================================================
# run_full_experiment.sh - the full SMART experiment ladder for ONE model:
#   Baseline -> Stage 1 (SFT) -> Stage 2 (3 augmentation variants) ->
#   Stage 3 (GRPO on each of the 4 resulting checkpoints)
#
# Every output path and every ledger run_id is namespaced by the model name,
# so running this once per model (see the per-model wrapper scripts in this
# directory) never overwrites another model's results - all 7 models' runs
# land in the SAME shared outputs/experiment_ledger.jsonl, which is what
# makes a single cross-model comparison table/figure possible afterward.
#
# DATA SIZE: defaults to the FULL train and test datasets - this is what an
# actual submitted/reported run should use. TRAIN_N/TEST_N are opt-in numeric
# caps for a quick smoke test ONLY (confirming the pipeline runs end to end
# before committing to a full run) - pass explicit numbers to use them,
# otherwise every problem in both splits is used.
#
# Usage:
#   ./run_full_experiment.sh <MODEL> [TRAIN_N] [TEST_N]
#
# Examples:
#   ./run_full_experiment.sh Qwen/Qwen2.5-Math-1.5B-Instruct                # full data (real run)
#   ./run_full_experiment.sh Qwen/Qwen2.5-Math-1.5B-Instruct 60 500         # capped (smoke test only)
#
# Run from the repository root (where data_pipeline.py etc. live).
# ============================================================================
set -euo pipefail

MODEL="${1:?Usage: run_full_experiment.sh <MODEL> [TRAIN_N] [TEST_N]}"
TRAIN_N="${2:-full}"
TEST_N="${3:-full}"
CHECKPOINT_STRIDE="${4:-1}"

# Sanitize the model name into a filesystem-safe, ledger-safe slug (e.g.
# "Qwen/Qwen2.5-Math-1.5B-Instruct" -> "Qwen_Qwen2.5-Math-1.5B-Instruct")
MODEL_SLUG="$(echo "$MODEL" | tr '/' '_')"

OUT="outputs/${MODEL_SLUG}"
CKPT="ckpts/${MODEL_SLUG}"
LEDGER="outputs/experiment_ledger.jsonl"   # shared across ALL models - this is
                                            # what enables a single cross-model
                                            # comparison table/figure afterward

mkdir -p "$OUT" "$CKPT"

echo "============================================================"
echo "MODEL:      $MODEL"
echo "MODEL_SLUG: $MODEL_SLUG"
echo "TRAIN_N:    $TRAIN_N"
echo "TEST_N:     $TEST_N"
echo "OUTPUTS:    $OUT"
echo "CHECKPOINTS: $CKPT"
echo "LEDGER:     $LEDGER"
echo "============================================================"

# ----------------------------------------------------------------------------
# Shared dataset (model-agnostic - built once, reused across every model this
# script is run for; safe to skip rebuilding if it already exists)
# ----------------------------------------------------------------------------
if [ ! -f outputs/sft_data_full.jsonl ]; then
    python data_pipeline.py --build-sft --split train --out outputs/sft_data_full.jsonl
fi
if [ "$TRAIN_N" = "full" ]; then
    cp outputs/sft_data_full.jsonl "$OUT/sft_data.jsonl"
else
    head -n "$TRAIN_N" outputs/sft_data_full.jsonl > "$OUT/sft_data.jsonl"
fi
wc -l "$OUT/sft_data.jsonl"

# Built once and reused in every run_eval.py call below: expands to nothing
# (full test split) when TEST_N=full, or to `--limit N` otherwise. Left
# UNQUOTED at each call site deliberately, so an empty value disappears
# entirely rather than being passed as a literal empty string argument.
if [ "$TEST_N" = "full" ]; then
    LIMIT_ARG=""
else
    LIMIT_ARG="--limit $TEST_N"
fi

# Auto-detected once: if a test-set skill-label file exists (produced by
# create_test_skill_labels.sh), every evaluator.py call below
# automatically uses it - without this, skill-prediction/skill-usage metrics
# show 0% coverage on the test split (only final-answer accuracy and
# arithmetic-consistency work without it). Left UNQUOTED at each call site
# for the same reason as LIMIT_ARG above.
TEST_SKILL_LABELS_FILE="data/test_skill_labels.jsonl"
if [ -f "$TEST_SKILL_LABELS_FILE" ]; then
    SKILL_LABELS_ARG="--skill-labels-file $TEST_SKILL_LABELS_FILE"
    echo "[run_full_experiment] using test-set skill labels: $TEST_SKILL_LABELS_FILE"
else
    SKILL_LABELS_ARG=""
    echo "[run_full_experiment] no test-set skill labels found at $TEST_SKILL_LABELS_FILE - "
    echo "  skill-prediction/skill-usage metrics will show 0% coverage this run. Run "
    echo "  create_test_skill_labels.sh first if you need them."
fi

# ----------------------------------------------------------------------------
# STAGE 0 - Baseline (vanilla, no training)
# ----------------------------------------------------------------------------
echo ""; echo "=== STAGE 0: Baseline ==="
python run_eval.py --model "$MODEL" --split test $LIMIT_ARG \
    --out "$OUT/predictions_baseline.jsonl"
python evaluator.py --score --predictions "$OUT/predictions_baseline.jsonl" --split test $SKILL_LABELS_ARG \
    --out-detailed "$OUT/eval_baseline_detailed.jsonl" --out-summary "$OUT/eval_baseline_summary.json"

python3 -c "
import json
json.dump({'model': '${MODEL}', 'mode': 'baseline'}, open('${OUT}/baseline_config.json', 'w'), indent=2)
"
python experiment_ledger.py --log --run_id "${MODEL_SLUG}__baseline" \
    --training_config "$OUT/baseline_config.json" \
    --eval_summary "$OUT/eval_baseline_summary.json" --ledger "$LEDGER" \
    --notes "vanilla, untrained, n=$TEST_N"

# ----------------------------------------------------------------------------
# STAGE 1 - Supervised Fine-Tuning
# ----------------------------------------------------------------------------
echo ""; echo "=== STAGE 1: SFT ==="
python sft_train.py --model "$MODEL" --data "$OUT/sft_data.jsonl" --mode lora \
    --lora_r 16 --lora_alpha 32 \
    --output_dir "$CKPT/stage1" --epochs 1 --save_every_epochs 0.5 \
    --per_device_batch_size 4 --grad_accum 2
# NOTE: sft_train.py auto-merges LoRA runs to <output_dir>_merged by default -
# no separate merge_lora.py call needed here (a redundant one just re-does the
# same merge for no benefit, and adds up across 7 models x every stage).

python run_eval.py --model "${CKPT}/stage1_merged" --split test $LIMIT_ARG \
    --out "$OUT/predictions_stage1.jsonl"
python evaluator.py --score --predictions "$OUT/predictions_stage1.jsonl" --split test $SKILL_LABELS_ARG \
    --out-detailed "$OUT/eval_stage1_detailed.jsonl" --out-summary "$OUT/eval_stage1_summary.json"
python experiment_ledger.py --log --run_id "${MODEL_SLUG}__stage1_sft" \
    --training_config "${CKPT}/stage1_merged/training_config.json" \
    --eval_summary "$OUT/eval_stage1_summary.json" --ledger "$LEDGER" \
    --baseline_run_id "${MODEL_SLUG}__baseline" --notes "SFT, LoRA, n=$TEST_N"

# Also evaluate and log EVERY intermediate checkpoint (--save_every_epochs 0.5
# above means checkpoint-N exists at 0.5 and 1.0 epochs) - a distinct
# "_ckpts" run_id_prefix keeps these separate from the single "official"
# stage1_sft run_id above, so nothing downstream that references that bare
# run_id needs to change; this supplementary pass is purely what enables an
# accuracy-vs-epoch retention curve afterward (generate_figures.py).
# Guarded with `|| echo ...`: this supplementary per-checkpoint pass must never
# abort the pipeline under `set -e`. The primary final-checkpoint evaluation
# above has already been logged.
python run_multi_checkpoint_eval.py --checkpoints_dir "${CKPT}/stage1" \
    --base_model "$MODEL" --mode lora --split test $LIMIT_ARG \
    --checkpoint_stride "$CHECKPOINT_STRIDE" \
    --out-dir "$OUT/checkpoint_eval_stage1" \
    --ledger "$LEDGER" --run_id_prefix "${MODEL_SLUG}__stage1_sft_ckpts" \
    --baseline_run_id "${MODEL_SLUG}__baseline" \
    || echo "[run_full_experiment] WARNING: checkpoint eval for stage1 did not complete - continuing"

# ----------------------------------------------------------------------------
# STAGE 2 - Diagnosis + 3 augmentation variants (semantic / numeric / both)
# ----------------------------------------------------------------------------
echo ""; echo "=== STAGE 2: Diagnosis + Augmentation ==="
python data_pipeline.py --diagnose --predictions "$OUT/predictions_stage1.jsonl" \
    --weak-report "$OUT/weak_clusters.json"
cat "$OUT/weak_clusters.json"

python run_augmentation.py --stage semantic --model "${CKPT}/stage1_merged" \
    --weak_report "$OUT/weak_clusters.json" --out "$OUT/semantic_aug.jsonl" --batch_size 64
python run_augmentation.py --stage numeric --model "${CKPT}/stage1_merged" \
    --weak_report "$OUT/weak_clusters.json" --out "$OUT/numeric_aug.jsonl" \
    --n_per_problem 1 --votes 3 --batch_size 64

run_stage2_variant () {
    local variant_name="$1"; shift
    local extra_data_files="$1"; shift
    local out_dir="${CKPT}/${variant_name}"

    echo ""; echo "--- Stage 2 variant: $variant_name ---"
    python sft_train.py --model "$MODEL" --data "$OUT/sft_data.jsonl" \
        --extra_data $extra_data_files \
        --replay_strategy skill --replay_ratio 0.7 --mode lora \
        --lora_r 16 --lora_alpha 32 \
        --output_dir "$out_dir" --epochs 1 --save_every_epochs 0.5 \
        --per_device_batch_size 4 --grad_accum 2

    python run_eval.py --model "${out_dir}_merged" --split test $LIMIT_ARG \
        --out "$OUT/predictions_${variant_name}.jsonl"
    python evaluator.py --score --predictions "$OUT/predictions_${variant_name}.jsonl" --split test $SKILL_LABELS_ARG \
        --out-detailed "$OUT/eval_${variant_name}_detailed.jsonl" \
        --out-summary "$OUT/eval_${variant_name}_summary.json"
    python experiment_ledger.py --log --run_id "${MODEL_SLUG}__${variant_name}" \
        --training_config "${out_dir}_merged/training_config.json" \
        --eval_summary "$OUT/eval_${variant_name}_summary.json" --ledger "$LEDGER" \
        --baseline_run_id "${MODEL_SLUG}__stage1_sft" --notes "SFT + $variant_name, n=$TEST_N"

    # Supplementary per-checkpoint pass for this variant too (see the Stage 1
    # comment above for why this uses a distinct "_ckpts" run_id_prefix).
    # Guarded with `|| echo ...`: this supplementary per-checkpoint pass must never
# abort the pipeline under `set -e`. The primary final-checkpoint evaluation
# above has already been logged.
python run_multi_checkpoint_eval.py --checkpoints_dir "$out_dir" \
        --base_model "$MODEL" --mode lora --split test $LIMIT_ARG \
    --checkpoint_stride "$CHECKPOINT_STRIDE" \
        --out-dir "$OUT/checkpoint_eval_${variant_name}" \
        --ledger "$LEDGER" --run_id_prefix "${MODEL_SLUG}__${variant_name}_ckpts" \
        --baseline_run_id "${MODEL_SLUG}__stage1_sft" \
        || echo "[run_full_experiment] WARNING: checkpoint eval for $variant_name did not complete - continuing"
}

run_stage2_variant "sem_replay"  "$OUT/semantic_aug.jsonl"
run_stage2_variant "num_replay"  "$OUT/numeric_aug.jsonl"
run_stage2_variant "both_replay" "$OUT/semantic_aug.jsonl $OUT/numeric_aug.jsonl"

# ----------------------------------------------------------------------------
# STAGE 3 - GRPO on top of Stage 1 and each Stage 2 variant
# ----------------------------------------------------------------------------
echo ""; echo "=== STAGE 3: GRPO ==="

run_stage3_grpo () {
    local base_name="$1"; shift          # e.g. "stage1", "sem_replay"
    local base_model_dir="$1"; shift     # e.g. ckpts/<slug>/stage1_merged
    local grpo_dir="${CKPT}/grpo_${base_name}"

    echo ""; echo "--- Stage 3 GRPO on: $base_name ---"
    python grpo_train.py --model "$base_model_dir" --data "$OUT/sft_data.jsonl" \
        --output_dir "$grpo_dir" \
        --num_generations 4 --per_device_batch_size 2 --grad_accum 2 --num_train_epochs 1 \
        --w_correctness 1.0 --w_format 0.2 --w_persistence 0.15 --w_chain_stability 0.25
    # grpo_train.py auto-merges to <output_dir>_merged by default too - no
    # separate merge_lora.py call needed here either.

    python run_eval.py --model "${grpo_dir}_merged" --split test $LIMIT_ARG \
        --out "$OUT/predictions_grpo_${base_name}.jsonl"
    python evaluator.py --score --predictions "$OUT/predictions_grpo_${base_name}.jsonl" --split test $SKILL_LABELS_ARG \
        --out-detailed "$OUT/eval_grpo_${base_name}_detailed.jsonl" \
        --out-summary "$OUT/eval_grpo_${base_name}_summary.json"
    python experiment_ledger.py --log --run_id "${MODEL_SLUG}__grpo_${base_name}" \
        --training_config "${grpo_dir}_merged/training_config.json" \
        --eval_summary "$OUT/eval_grpo_${base_name}_summary.json" --ledger "$LEDGER" \
        --baseline_run_id "${MODEL_SLUG}__baseline" --notes "SFT ($base_name) + GRPO, n=$TEST_N"

    # Supplementary per-checkpoint pass (grpo_train.py checkpoints every 50
    # steps, not epoch-linked, but the same retention-curve logic applies).
    # Guarded with `|| echo ...`: this supplementary per-checkpoint pass must never
# abort the pipeline under `set -e`. The primary final-checkpoint evaluation
# above has already been logged.
python run_multi_checkpoint_eval.py --checkpoints_dir "$grpo_dir" \
        --base_model "$base_model_dir" --mode lora --split test $LIMIT_ARG \
        --checkpoint_stride "$CHECKPOINT_STRIDE" \
        --out-dir "$OUT/checkpoint_eval_grpo_${base_name}" \
        --ledger "$LEDGER" --run_id_prefix "${MODEL_SLUG}__grpo_${base_name}_ckpts" \
        --baseline_run_id "${MODEL_SLUG}__baseline" \
        || echo "[run_full_experiment] WARNING: checkpoint eval for grpo_$base_name did not complete - continuing"
}

run_stage3_grpo "stage1"      "${CKPT}/stage1_merged"
run_stage3_grpo "sem_replay"  "${CKPT}/sem_replay_merged"
run_stage3_grpo "num_replay"  "${CKPT}/num_replay_merged"
run_stage3_grpo "both_replay" "${CKPT}/both_replay_merged"

echo ""
echo "============================================================"
echo "DONE: $MODEL"
echo "9 runs logged to $LEDGER, all prefixed ${MODEL_SLUG}__"
echo "============================================================"
