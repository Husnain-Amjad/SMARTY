"""
Professional-grade test-set skill labeling using a large, INDEPENDENT judge
model (default: openai/gpt-oss-120b, quantized) for both labeling and
verification - distinct from label_test_create.py's simpler two-pass
self-labeling pipeline (both now share the same grounded-annotation Pass 1,
see below).

Three passes, not two:
  Pass 1 (Annotate): the judge model is given the PROBLEM *and* its correct
    reference solution, and asked to annotate that existing solution's steps
    with skill tags - NOT to independently re-solve the problem. This is
    deliberate: if the model were asked to solve freely, an incorrect or
    differently-derived solution that still happens to land on the right
    final answer would produce skill labels reflecting that flawed/different
    path rather than the actual reference derivation. Grounding Pass 1 in the
    given solution removes that failure mode entirely.
  Pass 2 (Extract): given the annotated reference trace + correct answer,
    extract the minimum skill set actually necessary to reach the answer.
  Pass 3 (Verify) - THE NEW PART: for each (step, claimed skill) pair
    surviving the canonical-vocabulary hard filter, ask the SAME judge model
    a narrowly-scoped yes/no question - was this specific skill correctly
    and actually applied in THIS specific step of the reference solution?
    A skill is kept in the final label only if at least one step where it
    was claimed passes verification.

Why this is a separate pass from the vocabulary filter: constraining to a
canonical vocabulary stops HALLUCINATED skill names (a skill that doesn't
exist in the taxonomy at all), but does nothing to stop a genuinely real
skill being mis-attributed to the wrong step (e.g. tagging "Quadratic
Equations" on a step that's actually just arithmetic). Pass 3 targets that
second, distinct failure mode.

Model choice: openai/gpt-oss-120b (a large, independent MoE model, not the
model under evaluation) is used deliberately so the reference labels aren't
graded by the same model being tested - a materially different reliability
bar than self-labeling. gpt-oss-120b ships with native MXFP4-quantized
weights on Hugging Face; vLLM auto-detects this from the model's own config
in most versions, so no explicit --quantization flag is needed for the
default case (still exposed as a flag for other models/overrides). The
quantized checkpoint is roughly 65GB - this needs the large majority of a
single 80GB card (or --tensor_parallel_size across multiple smaller GPUs).
Run this as a standalone job, not concurrently with training.

Usage:
  # Step 1: inspect the canonical vocabulary first, no model/GPU needed
  python label_test_set.py --inspect-vocab \
      --train-skill-labels-file data/skill_labels.jsonl.example

  # Step 2: label + verify
  python label_test_set.py --label \
      --train-skill-labels-file data/skill_labels.jsonl.example \
      --model openai/gpt-oss-120b \
      --out outputs/test_skill_labels_verified.jsonl
"""

import argparse
import json
import re
from collections import Counter, defaultdict

from label_test_create import (
    extract_canonical_vocabulary, print_vocab_report, hard_filter_skills,
    build_annotate_prompt, build_extract_prompt, parse_skills_from_trace, parse_minimum_skills,
    normalize_trace_to_bracket_tags, normalize_subject_key,
)
from data_pipeline import load_hendrycks_math, _hash_problem
from reward_fn import extract_boxed
from storage_utils import ensure_output_path, add_destination_args, dispatch_destination
from determinism import set_all_seeds

# ---------------------------------------------------------------------------
# Pass 3: verification prompt + parser
# ---------------------------------------------------------------------------

VERIFY_SYSTEM = """You are a strict verifier auditing mathematical solutions.
You will be given a problem, ONE specific solution step from its worked solution, \
and a claimed skill that step is supposed to demonstrate.
Judge ONLY whether the claimed skill accurately describes the mathematical operation \
actually performed in THIS specific step - NOT whether the final numeric answer is \
correct, and NOT whether some other skill might also plausibly apply.
Respond with EXACTLY one line, nothing else:
VERDICT: YES
or
VERDICT: NO
"""


def build_verify_prompt(problem: str, step_text: str, claimed_skill: str) -> str:
    return (f"{VERIFY_SYSTEM}\n\nProblem:\n{problem}\n\nSolution step:\n{step_text}\n\n"
            f"Claimed skill: {claimed_skill}\n\nVerdict?")


def parse_verdict(output: str) -> bool:
    """Fails CLOSED: an unparseable or missing verdict does not count as verified -
    this errs toward under-crediting a skill rather than accepting an ambiguous judgment."""
    m = re.search(r"VERDICT:\s*(YES|NO)", output or "", re.IGNORECASE)
    if not m:
        return False
    return m.group(1).upper() == "YES"


def verify_skill_usage(rows_with_tagged_steps: list, batch_generate_fn, batch_size: int = 32):
    """
    rows_with_tagged_steps: list of {'problem': str, 'tagged_steps': [(skill, step_text), ...]}
        (only steps that actually have a claimed skill tag - untagged steps aren't verified)
    batch_generate_fn: callable(prompts: list[str]) -> list[str], one verdict text per prompt

    Returns (verified_per_row, failed_per_row, n_calls): two lists of sets, aligned to
    rows_with_tagged_steps, plus the total number of verification calls made (for reporting).
    """
    jobs = []  # (row_idx, skill, prompt)
    for i, row in enumerate(rows_with_tagged_steps):
        for skill, step_text in row["tagged_steps"]:
            jobs.append((i, skill, build_verify_prompt(row["problem"], step_text, skill)))

    print(f"[label_test_set] verifying {len(jobs)} (step, claimed skill) pairs "
          f"in batches of {batch_size} ({(len(jobs) + batch_size - 1) // batch_size} calls)")

    outputs = []
    for start in range(0, len(jobs), batch_size):
        batch_prompts = [p for _, _, p in jobs[start:start + batch_size]]
        outputs.extend(batch_generate_fn(batch_prompts))
        print(f"[label_test_set] verification {min(start + batch_size, len(jobs))}/{len(jobs)}")

    verified_per_row = [set() for _ in rows_with_tagged_steps]
    failed_per_row = [set() for _ in rows_with_tagged_steps]
    for (row_idx, skill, _), output in zip(jobs, outputs):
        if parse_verdict(output):
            verified_per_row[row_idx].add(skill)
        else:
            failed_per_row[row_idx].add(skill)

    return verified_per_row, failed_per_row, len(jobs)


# ---------------------------------------------------------------------------
# Full three-pass pipeline
# ---------------------------------------------------------------------------

def label_and_verify_test_split(flat_vocab: dict, batch_generate_fn, out_path: str,
                                 split: str = "test", batch_size: int = 32,
                                 fuzzy_threshold: float = 0.75, limit: int = None,
                                 verify_batch_size: int = 32):
    math_rows = load_hendrycks_math(split)
    if limit:
        math_rows = math_rows[:limit]
    print(f"[label_test_set] Pass 1+2: labeling {len(math_rows)} problems from split='{split}'")

    # --- Pass 1: annotate the REFERENCE solution with skill tags (not an independent re-derivation) ---
    solve_prompts = [build_annotate_prompt(r["problem"], r["solution"], r["subject"], flat_vocab) for r in math_rows]
    traces = []
    for i in range(0, len(solve_prompts), batch_size):
        batch = solve_prompts[i:i + batch_size]
        traces.extend(batch_generate_fn(batch))
        print(f"[label_test_set] pass 1 (annotate reference solution): {min(i + batch_size, len(solve_prompts))}/{len(solve_prompts)}")
    # Real model output frequently ignores the requested [SKILL: X] bracket
    # format (observed directly: Qwen2.5-Math models commonly use **Skill: X**
    # markdown instead) - normalize immediately so every downstream step
    # (vocabulary filtering, Pass 3 step-splitting) works against one
    # consistent format regardless of what style the model actually used.
    traces = [normalize_trace_to_bracket_tags(t) for t in traces]

    # --- Pass 2: extract minimum skill set ---
    extract_prompts = []
    for row, trace in zip(math_rows, traces):
        gold = extract_boxed(row["solution"]) or row["solution"][:80]
        extract_prompts.append(build_extract_prompt(row["problem"], trace, gold))
    extraction_raw_list = []
    for i in range(0, len(extract_prompts), batch_size):
        batch = extract_prompts[i:i + batch_size]
        extraction_raw_list.extend(batch_generate_fn(batch))
        print(f"[label_test_set] pass 2 (extract): {min(i + batch_size, len(extract_prompts))}/{len(extract_prompts)}")

    # --- Hard canonical-vocabulary filter (stops hallucinated skill names) ---
    per_problem_used_filtered = []
    per_problem_min_filtered = []
    per_problem_tagged_steps = []  # for Pass 3: [(skill, step_text), ...] per problem
    total_vocab_rejected = 0

    _step_re = re.compile(r"\[SKILL:\s*([^\]]+)\]")
    for row, trace, extraction_raw in zip(math_rows, traces, extraction_raw_list):
        canonical_for_subject = flat_vocab.get(normalize_subject_key(row["subject"]), [])
        used_in_steps = parse_skills_from_trace(trace)
        minimum_skills_raw = parse_minimum_skills(extraction_raw)

        used_filtered, n_rej1, _ = hard_filter_skills(used_in_steps, canonical_for_subject, fuzzy_threshold)
        min_filtered, n_rej2, _ = hard_filter_skills(minimum_skills_raw, canonical_for_subject, fuzzy_threshold)
        total_vocab_rejected += n_rej1 + n_rej2

        per_problem_used_filtered.append(used_filtered)
        per_problem_min_filtered.append(min_filtered)

        # Split the trace into (skill, step_text) pairs, keeping only steps whose
        # tag survived the vocabulary filter above - these are what Pass 3 audits.
        parts = _step_re.split(trace)
        tagged = []
        for j in range(1, len(parts), 2):
            skill = parts[j].strip()
            step_text = parts[j + 1].strip() if j + 1 < len(parts) else ""
            if skill in used_filtered and step_text:
                tagged.append((skill, step_text))
        per_problem_tagged_steps.append(tagged)

    print(f"[label_test_set] vocabulary filter rejected {total_vocab_rejected} raw skill "
          f"mentions before verification even started")

    # --- Pass 3: verify each surviving (step, skill) claim ---
    rows_for_verification = [
        {"problem": row["problem"], "tagged_steps": tagged}
        for row, tagged in zip(math_rows, per_problem_tagged_steps)
    ]
    verified_per_row, failed_per_row, n_verify_calls = verify_skill_usage(
        rows_for_verification, batch_generate_fn, batch_size=verify_batch_size,
    )

    # --- Assemble final results: a skill survives only if it's in the Pass-2
    # minimum set AND verified correct in at least one step where it was claimed ---
    results = []
    n_skill_verification_failures = 0
    for idx, (row, trace, extraction_raw) in enumerate(zip(math_rows, traces, extraction_raw_list)):
        min_filtered = per_problem_min_filtered[idx]
        used_filtered = per_problem_used_filtered[idx]
        verified = verified_per_row[idx]
        failed = failed_per_row[idx]

        final_skills = [s for s in min_filtered if s in verified]
        n_dropped_by_verification = len([s for s in min_filtered if s in failed and s not in verified])
        n_skill_verification_failures += n_dropped_by_verification

        if not final_skills:
            # Nothing survived verification - fall back to the vocabulary-filtered
            # (but unverified) set rather than emitting an empty label, with a flag
            # so this is visible in the output rather than silently degraded.
            final_skills = min_filtered or used_filtered
            verification_status = "fallback_unverified" if final_skills else "empty"
        else:
            verification_status = "verified"

        results.append({
            "subject": row["subject"],
            "level": row.get("level", "unknown"),
            "problem": row["problem"],
            "original_solution": row["solution"],
            "model_answer": extract_boxed(trace) or "",
            "reasoning_trace": trace,
            "skills_used_in_steps": " | ".join(used_filtered),
            "minimum_skills": " | ".join(final_skills),
            "n_skills": len(final_skills),
            "verification_status": verification_status,
            "verification_verified_skills": " | ".join(sorted(verified)),
            "verification_failed_skills": " | ".join(sorted(failed)),
            "_extraction_raw": extraction_raw,
        })

    ensure_output_path(out_path)
    with open(out_path, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    n_verified = sum(1 for r in results if r["verification_status"] == "verified")
    n_fallback = sum(1 for r in results if r["verification_status"] == "fallback_unverified")
    n_empty = sum(1 for r in results if r["verification_status"] == "empty")
    print(f"\n[label_test_set] wrote {len(results)} labeled+verified test examples -> {out_path}")
    print(f"[label_test_set] verification summary: {n_verified} fully verified, "
          f"{n_fallback} fell back to unverified vocabulary-filtered skills (verification "
          f"rejected everything claimed), {n_empty} empty (no skills survived any stage)")
    print(f"[label_test_set] {n_verify_calls} total verification calls made, "
          f"{n_skill_verification_failures} individual skill claims rejected by Pass 3")

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--inspect-vocab", action="store_true",
                     help="print the canonical vocabulary extracted from train labels and exit - "
                          "no model/GPU needed. Run this first.")
    ap.add_argument("--label", action="store_true",
                     help="run the full 3-pass label + verify pipeline")

    ap.add_argument("--train-split", default="train")
    ap.add_argument("--skill-repo", default=None)
    ap.add_argument("--train-skill-labels-file", default=None)

    ap.add_argument("--test-split", default="test")
    ap.add_argument("--model", default="openai/gpt-oss-120b",
                     help="the independent judge/labeler model. Defaults to gpt-oss-120b "
                          "(native MXFP4-quantized weights, ~65GB - needs most of an 80GB "
                          "card, or --tensor_parallel_size across multiple smaller GPUs).")
    ap.add_argument("--quantization", default=None,
                     help="override vLLM's auto-detected quantization method. Leave unset "
                          "for gpt-oss-120b - its MXFP4 quantization is auto-detected from "
                          "the model's own config in most vLLM versions.")
    ap.add_argument("--use_vllm", action="store_true", default=True)
    ap.add_argument("--gpu_memory_utilization", type=float, default=0.92)
    ap.add_argument("--max_model_len", type=int, default=4096)
    ap.add_argument("--tensor_parallel_size", type=int, default=1,
                     help="shard the model across this many GPUs - likely needed for a "
                          "120B model unless your single GPU has ample headroom above the "
                          "~65GB quantized weight size.")
    ap.add_argument("--batch_size", type=int, default=16,
                     help="lower default than the augmentation scripts - a 120B model at "
                          "high memory utilization has less headroom for large batches.")
    ap.add_argument("--verify_batch_size", type=int, default=32,
                     help="verification prompts are short (one step + one skill), so this "
                          "can usually run larger than the solve/extract batch size.")
    ap.add_argument("--fuzzy_threshold", type=float, default=0.75)
    ap.add_argument("--limit", type=int, default=None, help="cap problems for a quick test")
    ap.add_argument("--out", default="outputs/test_skill_labels_verified.jsonl")
    ap.add_argument("--seed", type=int, default=42)
    add_destination_args(ap, default_repo_type="dataset")
    args = ap.parse_args()

    set_all_seeds(args.seed)

    per_subject, flat_vocab = extract_canonical_vocabulary(
        args.train_split, skill_repo=args.skill_repo, skill_labels_file=args.train_skill_labels_file,
    )

    if args.inspect_vocab:
        print_vocab_report(per_subject)

    elif args.label:
        from run_augmentation import build_backend
        from templates import render_prompt_only

        backend = build_backend(
            args.model, args.use_vllm, seed=args.seed,
            gpu_memory_utilization=args.gpu_memory_utilization, max_model_len=args.max_model_len,
            tensor_parallel_size=args.tensor_parallel_size, quantization=args.quantization,
        )

        def batch_generate_fn(prompts):
            # Solve/extract prompts already contain their own full system+user
            # instructions (see build_solve_prompt/build_extract_prompt in
            # label_test_create.py) - wrap with this model's own turn tokens
            # via templates.py rather than assuming a fixed chat format.
            formatted = [render_prompt_only(backend.tokenizer, p, system_prompt="") for p in prompts]
            results = backend.generate(formatted, n=1, temperature=0.2, max_tokens=1536)
            return [r[0] for r in results]

        results = label_and_verify_test_split(
            flat_vocab, batch_generate_fn, args.out, split=args.test_split,
            batch_size=args.batch_size, fuzzy_threshold=args.fuzzy_threshold, limit=args.limit,
            verify_batch_size=args.verify_batch_size,
        )
        dispatch_destination(args.out, args)

    else:
        print("Pass --inspect-vocab (no model needed) or --label (requires --model). "
              "Run --inspect-vocab first to sanity-check the vocabulary before generating.")
