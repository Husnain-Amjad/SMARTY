#!/usr/bin/env bash
# ============================================================================
# create_test_skill_labels.sh - generates test-split skill labels matching
# the exact training-data format, using a powerful labeler model, a judge
# model to verify format/skill correctness, and a fixer model to correct
# anything the judge rejects. See label_test_set_v2.py's own docstring for
# the full design rationale.
#
# Output is saved to data/test_skill_labels.jsonl by default - this is the
# well-known path run_full_experiment.sh automatically looks for and passes
# to every evaluator.py call via --skill-labels-file, so skill-prediction/
# skill-usage metrics populate on every stage's evaluation once this exists.
#
# Usage:
#   bash create_test_skill_labels.sh [LABELER_MODEL] [JUDGE_MODEL] [LIMIT]
#
# Examples:
#   bash create_test_skill_labels.sh                                    # defaults, full test set
#   bash create_test_skill_labels.sh openai/gpt-oss-120b                 # specific labeler, same model judges
#   bash create_test_skill_labels.sh openai/gpt-oss-120b openai/gpt-oss-20b 50   # tiny sample test
# ============================================================================
set -euo pipefail
cd "$(dirname "$0")"   # repository root

LABELER_MODEL="${1:-openai/gpt-oss-120b}"
JUDGE_MODEL="${2:-$LABELER_MODEL}"
LIMIT="${3:-}"

mkdir -p data

LIMIT_ARG=""
if [ -n "$LIMIT" ]; then
    LIMIT_ARG="--limit $LIMIT"
fi

echo "=== Inspecting canonical vocabulary (no model needed) ==="
python label_test_set_v2.py --inspect-vocab

echo ""
echo "=== Labeling test set: labeler=$LABELER_MODEL judge/fixer=$JUDGE_MODEL ==="
python label_test_set_v2.py --label \
    --labeler-model "$LABELER_MODEL" \
    --judge-model "$JUDGE_MODEL" \
    --out data/test_skill_labels.jsonl \
    $LIMIT_ARG

echo ""
echo "Done. Saved to data/test_skill_labels.jsonl"
echo "run_full_experiment.sh will automatically use this file for every"
echo "evaluation stage's skill-prediction/skill-usage metrics if present."
