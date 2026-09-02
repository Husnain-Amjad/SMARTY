"""
Generates test-split skill labels matching the EXACT format and column
conventions of the real training data (HusnainAmjad/Skill_MATH), using:
  - A one-shot worked example (a real training-set instance) embedded
    directly in every prompt, so the labeler model sees exactly the target
    format rather than inferring it from a text description alone.
  - A powerful labeler model to annotate the reference solution and propose
    skills (same "annotate the given solution, don't re-derive" design as
    label_test_create.py - see that module for why).
  - Semantic canonicalization: raw skill guesses that don't match the
    canonical vocabulary by fuzzy string similarity get a SECOND model call
    asking specifically "which canonical skill is this semantically the
    same as?" - this catches cases fuzzy string matching can't (e.g. "sequence
    and series skills" -> "Arithmetic and Geometric Sequences" are unrelated
    as strings but the same skill in meaning).
  - LLM-as-judge verification: a (possibly different, more capable) judge
    model checks whether the format is correct (inline [SKILL: X] tags,
    trailing "ANSWER: \\boxed{...}") and whether the skills used are
    genuinely canonical - not just "did parsing succeed."
  - LLM-as-fixer: rows the judge rejects are sent to a second model call
    (same or different model) with the judge's specific feedback and the
    one-shot example again, to regenerate a corrected version - not just
    discarded.

Output schema matches the real training data EXACTLY - the same 9 columns,
no extra metadata columns:
  subject, level, problem, original_solution, model_answer, reasoning_trace,
  skills_used_in_steps, minimum_skills, n_skills, _extraction_raw

Matching two real training-data conventions that differ from earlier scripts
in this repo:
  - minimum_skills is stored WITH the "MINIMUM_SKILLS: " prefix retained
    (data_pipeline.clean_skill_list() already strips it on read either way).
  - _extraction_raw holds the PRE-canonicalization raw skill guess (which may
    use non-canonical wording), not a cleaned-up version.

Source: EleutherAI/hendrycks_math test split (via data_pipeline.load_hendrycks_math).

Usage:
  python label_test_set_v2.py --inspect-vocab

  python label_test_set_v2.py --label \
      --labeler-model openai/gpt-oss-120b \
      --judge-model openai/gpt-oss-120b \
      --out outputs/test_skill_labels_v2.jsonl
"""

import argparse
import json
import re

from label_test_create import (
    extract_canonical_vocabulary, print_vocab_report, hard_filter_skills,
    parse_skills_from_trace, normalize_trace_to_bracket_tags, normalize_subject_key,
    _similarity,
)
from data_pipeline import load_hendrycks_math, clean_skill_list
from reward_fn import extract_boxed
from storage_utils import ensure_output_path, add_destination_args, dispatch_destination
from determinism import set_all_seeds

# ---------------------------------------------------------------------------
# One-shot example - a real training-set instance, used verbatim in every
# prompt so the model sees the exact target format rather than inferring it
# from a text description alone.
# ---------------------------------------------------------------------------

ONE_SHOT_EXAMPLE = {
    "subject": "algebra",
    "level": "Level 2",
    "problem": r"Find the sum: $1+2+3+4+\dots +48+49$",
    "original_solution": r"For all $n$, $1 + 2 + \dots + n = n(n + 1)/2$, so $1 + 2 + \dots + 49 = 49 \cdot 50/2 = \boxed{1225}$.",
    "model_answer": "1225",
    "reasoning_trace": (
        "[SKILL: Basic Arithmetic Operations]\n"
        "We recognize that the sum $1 + 2 + 3 + \\dots + 49$ is an arithmetic series starting at 1 and ending at 49.\n"
        "[SKILL: Arithmetic and Geometric Sequences]\n"
        "The formula for the sum of the first $n$ positive integers is:\n"
        "$$\n\\sum_{k=1}^{n} k = \\frac{n(n+1)}{2}\n$$\n"
        "Here, $n = 49$.\n"
        "[SKILL: Basic Arithmetic Operations]\n"
        "Substitute $n = 49$ into the formula:\n"
        "$$\n\\frac{49 \\cdot 50}{2}\n$$\n"
        "[SKILL: Basic Arithmetic Operations]\n"
        "Calculate the numerator:\n"
        "$$\n49 \\cdot 50 = 2450\n$$\n"
        "[SKILL: Basic Arithmetic Operations]\n"
        "Divide by 2:\n"
        "$$\n\\frac{2450}{2} = 1225\n$$\n"
        "ANSWER: \\boxed{1225}"
    ),
    "skills_used_in_steps": "Basic Arithmetic Operations | Arithmetic and Geometric Sequences",
    "minimum_skills": "MINIMUM_SKILLS: Arithmetic and Geometric Sequences | Basic Arithmetic Operations",
    "n_skills": 2,
    "_extraction_raw": "MINIMUM_SKILLS: sequence and series skills | arithmetic skills",
}


def _format_one_shot_block() -> str:
    ex = ONE_SHOT_EXAMPLE
    return (
        "--- WORKED EXAMPLE (follow this exact format) ---\n"
        f"Problem: {ex['problem']}\n"
        f"Reference solution: {ex['original_solution']}\n\n"
        f"Correctly annotated reasoning_trace:\n{ex['reasoning_trace']}\n\n"
        f"Correctly extracted skills_used_in_steps: {ex['skills_used_in_steps']}\n"
        f"Correctly extracted minimum_skills: {ex['minimum_skills']}\n"
        "--- END WORKED EXAMPLE ---\n"
    )


# ---------------------------------------------------------------------------
# Pass 1: annotate, with the one-shot example embedded
# ---------------------------------------------------------------------------

ANNOTATE_SYSTEM_V2 = """You are a precise math solution annotator.
You will be shown one worked example of the exact format required, then a new
problem and its correct reference solution for you to annotate the same way.

Your task is NOT to solve the problem yourself or produce a different
derivation - it is to annotate the GIVEN reference solution, breaking it into
its logical steps and labeling EACH step with the skill it demonstrates,
using EXACTLY the format shown in the worked example: [SKILL: <skill_name>]
on its own line immediately before the step it applies to.

Each skill MUST come from this allowed list EXACTLY as written - do not
invent new skill names:
{allowed_skills}

End with a line reading exactly: ANSWER: \\boxed{{value}}
(matching the reference solution's own final answer - you are reporting it,
not computing a new one)
"""


def build_annotate_prompt_v2(problem: str, reference_solution: str, subject: str, flat_vocab: dict) -> str:
    allowed = " | ".join(flat_vocab.get(normalize_subject_key(subject),
                                        ["(no canonical skills recorded for this subject)"]))
    system = ANNOTATE_SYSTEM_V2.format(allowed_skills=allowed)
    return (f"{system}\n\n{_format_one_shot_block()}\n"
            f"Now annotate this one:\n\nProblem: {problem}\n\n"
            f"Reference solution (annotate this exactly - do not re-derive):\n{reference_solution}\n\n"
            f"Annotated solution:")


# ---------------------------------------------------------------------------
# Pass 2: extract the raw (possibly non-canonical) minimum skill guess
# ---------------------------------------------------------------------------

EXTRACT_SYSTEM_V2 = """You are a math solution analyst identifying the minimum skills needed.
You will be shown one worked example, then a new annotated solution to analyze
the same way.
List ONLY the skills DIRECTLY needed to reach the final answer, removing any
skill that appeared in a redundant step. Output format (one line):
  MINIMUM_SKILLS: skill1 | skill2 | skill3
Do NOT output anything else - no explanation, no numbering.
"""


def build_extract_prompt_v2(problem: str, annotated_trace: str, correct_answer: str) -> str:
    return (f"{EXTRACT_SYSTEM_V2}\n\n{_format_one_shot_block()}\n"
            f"Now analyze this one:\n\nProblem: {problem}\n\n"
            f"Annotated solution:\n{annotated_trace}\n\nCorrect answer: {correct_answer}\n\n"
            f"Minimum skills needed:")


def parse_minimum_skills_v2(extraction_output: str) -> list:
    """Same tolerant parsing as label_test_create.parse_minimum_skills, returning
    just the list of raw skill strings (canonicalization happens separately here,
    since v2 needs to retain BOTH the raw guess (for _extraction_raw) and the
    canonicalized result (for minimum_skills) as two distinct output fields."""
    from label_test_create import parse_minimum_skills
    return parse_minimum_skills(extraction_output)


# ---------------------------------------------------------------------------
# Semantic canonicalization: for raw skills that fail fuzzy string matching,
# ask the model directly which canonical skill it semantically means.
# ---------------------------------------------------------------------------

CANONICALIZE_SYSTEM = """You are mapping an informally-worded math skill description to the
closest matching entry in a fixed, canonical skill vocabulary.
Respond with EXACTLY one line:
  MATCH: <exact canonical skill name>
or, if genuinely nothing in the list is a reasonable match:
  MATCH: NONE
"""


def build_canonicalize_prompt(raw_skill: str, canonical_skills: list) -> str:
    allowed = " | ".join(canonical_skills)
    return (f"{CANONICALIZE_SYSTEM}\n\nInformal skill description: \"{raw_skill}\"\n\n"
            f"Canonical vocabulary:\n{allowed}\n\nMATCH:")


def parse_canonicalize_match(output: str, canonical_skills: list) -> str:
    m = re.search(r"MATCH:\s*(.+)", output or "", re.IGNORECASE)
    if not m:
        return None
    candidate = m.group(1).strip()
    if candidate.upper() == "NONE":
        return None
    # exact (case-insensitive) match against the canonical list only - never
    # trust the model's own spelling verbatim, even here
    for c in canonical_skills:
        if c.lower() == candidate.lower():
            return c
    return None


def canonicalize_with_model(raw_skills: list, canonical_skills: list, batch_generate_fn,
                             fuzzy_threshold: float = 0.75):
    """
    First tries fuzzy string matching (cheap, no model call). Only for skills
    that fail that check does this fall back to a model call asking
    specifically for the semantic match - this is what catches cases like
    "sequence and series skills" -> "Arithmetic and Geometric Sequences",
    which fuzzy string similarity alone cannot.
    """
    canonical_lower = {c.lower(): c for c in canonical_skills}
    resolved, unresolved = [], []
    for raw in raw_skills:
        if raw.strip().lower() in canonical_lower:
            resolved.append(canonical_lower[raw.strip().lower()])
            continue
        best_match, best_score = None, 0.0
        for canon in canonical_skills:
            score = _similarity(raw, canon)
            if score > best_score:
                best_match, best_score = canon, score
        if best_score >= fuzzy_threshold:
            resolved.append(best_match)
        else:
            unresolved.append(raw)

    if unresolved and canonical_skills:
        prompts = [build_canonicalize_prompt(raw, canonical_skills) for raw in unresolved]
        outputs = batch_generate_fn(prompts)
        for raw, output in zip(unresolved, outputs):
            match = parse_canonicalize_match(output, canonical_skills)
            if match:
                resolved.append(match)
            # else: genuinely no match found even semantically - dropped, not kept

    seen, deduped = set(), []
    for s in resolved:
        if s not in seen:
            seen.add(s)
            deduped.append(s)
    return deduped


# ---------------------------------------------------------------------------
# LLM-as-judge: verifies format AND skill correctness together
# ---------------------------------------------------------------------------

JUDGE_SYSTEM = """You are a strict quality auditor for math-solution skill annotations.
You will be shown the required format (via a worked example), then a
candidate annotation to audit.
Check BOTH of the following:
  1. FORMAT: does the reasoning_trace use [SKILL: <name>] on its own line
     before each step, and end with a line "ANSWER: \\boxed{...}"?
  2. SKILLS: are the listed skills genuinely canonical (drawn from the
     allowed vocabulary) and genuinely relevant to the steps they're
     attached to (not hallucinated, not misattributed)?
Respond with EXACTLY one line:
  VERDICT: PASS
or
  VERDICT: FAIL <one short sentence saying what's wrong>
"""


def build_judge_prompt(problem: str, reasoning_trace: str, minimum_skills: str,
                        canonical_skills: list) -> str:
    allowed = " | ".join(canonical_skills)
    return (f"{JUDGE_SYSTEM}\n\n{_format_one_shot_block()}\n"
            f"Now audit this candidate:\n\nProblem: {problem}\n\n"
            f"Canonical vocabulary for this subject:\n{allowed}\n\n"
            f"Candidate reasoning_trace:\n{reasoning_trace}\n\n"
            f"Candidate minimum_skills: {minimum_skills}\n\nVERDICT:")


def parse_judge_verdict(output: str):
    m = re.search(r"VERDICT:\s*(PASS|FAIL)\s*(.*)", output or "", re.IGNORECASE | re.DOTALL)
    if not m:
        return False, "unparseable judge output"
    passed = m.group(1).upper() == "PASS"
    reason = m.group(2).strip()
    return passed, reason


# ---------------------------------------------------------------------------
# LLM-as-fixer: regenerates a corrected row given the judge's specific feedback
# ---------------------------------------------------------------------------

FIXER_SYSTEM = """You are correcting a math-solution skill annotation that failed quality review.
You will be shown the required format (via a worked example), the original
problem and reference solution, the candidate that failed, and specifically
why it failed. Produce a corrected version following the worked example's
format exactly, drawing skills ONLY from the allowed vocabulary, fixing the
specific problem described.
End with a line reading exactly: ANSWER: \\boxed{value}
"""


def build_fix_prompt(problem: str, reference_solution: str, subject: str, flat_vocab: dict,
                      failed_trace: str, judge_reason: str) -> str:
    allowed = " | ".join(flat_vocab.get(normalize_subject_key(subject), []))
    return (f"{FIXER_SYSTEM}\n\n{_format_one_shot_block()}\n"
            f"Problem: {problem}\n\nReference solution:\n{reference_solution}\n\n"
            f"Allowed skill vocabulary:\n{allowed}\n\n"
            f"Failed candidate:\n{failed_trace}\n\n"
            f"Why it failed: {judge_reason}\n\nCorrected annotated solution:")


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

def label_test_split_v2(flat_vocab: dict, labeler_fn, judge_fn, fixer_fn, out_path: str,
                         split: str = "test", batch_size: int = 16, fuzzy_threshold: float = 0.75,
                         limit: int = None, max_fix_attempts: int = 1):
    math_rows = load_hendrycks_math(split)
    if limit:
        math_rows = math_rows[:limit]
    print(f"[label_test_set_v2] labeling {len(math_rows)} problems from split='{split}'")

    # --- Pass 1: annotate ---
    annotate_prompts = [build_annotate_prompt_v2(r["problem"], r["solution"], r["subject"], flat_vocab)
                        for r in math_rows]
    traces = []
    for i in range(0, len(annotate_prompts), batch_size):
        traces.extend(labeler_fn(annotate_prompts[i:i + batch_size]))
        print(f"[label_test_set_v2] pass 1 (annotate): {min(i + batch_size, len(annotate_prompts))}/{len(annotate_prompts)}")
    traces = [normalize_trace_to_bracket_tags(t) for t in traces]

    # --- Pass 2: extract raw minimum skill guess ---
    extract_prompts = []
    for row, trace in zip(math_rows, traces):
        gold = extract_boxed(row["solution"]) or row["solution"][:80]
        extract_prompts.append(build_extract_prompt_v2(row["problem"], trace, gold))
    extraction_raw_list = []
    for i in range(0, len(extract_prompts), batch_size):
        extraction_raw_list.extend(labeler_fn(extract_prompts[i:i + batch_size]))
        print(f"[label_test_set_v2] pass 2 (extract): {min(i + batch_size, len(extract_prompts))}/{len(extract_prompts)}")

    # --- Canonicalize (fuzzy first, then semantic model fallback) ---
    results = []
    for row, trace, extraction_raw in zip(math_rows, traces, extraction_raw_list):
        canonical = flat_vocab.get(normalize_subject_key(row["subject"]), [])
        used_raw = parse_skills_from_trace(trace)
        min_raw = parse_minimum_skills_v2(extraction_raw)

        used_canon = canonicalize_with_model(used_raw, canonical, judge_fn, fuzzy_threshold)
        min_canon = canonicalize_with_model(min_raw, canonical, judge_fn, fuzzy_threshold)
        final_skills = [s for s in min_canon if s in used_canon] or min_canon or used_canon

        results.append({
            "subject": row["subject"], "level": row.get("level", "unknown"),
            "problem": row["problem"], "original_solution": row["solution"],
            "model_answer": extract_boxed(trace) or "",
            "reasoning_trace": trace,
            "skills_used_in_steps": " | ".join(used_canon),
            "minimum_skills": f"MINIMUM_SKILLS: {' | '.join(final_skills)}" if final_skills else "",
            "n_skills": len(final_skills),
            "_extraction_raw": extraction_raw,
        })

    # --- LLM-as-judge, then LLM-as-fixer for anything that fails ---
    n_passed_first_try, n_fixed, n_still_failing = 0, 0, 0
    for attempt in range(max_fix_attempts + 1):
        judge_prompts, judge_indices = [], []
        for idx, (row, res) in enumerate(zip(math_rows, results)):
            canonical = flat_vocab.get(normalize_subject_key(row["subject"]), [])
            judge_prompts.append(build_judge_prompt(row["problem"], res["reasoning_trace"],
                                                     res["minimum_skills"], canonical))
            judge_indices.append(idx)

        verdicts = []
        for i in range(0, len(judge_prompts), batch_size):
            verdicts.extend(judge_fn(judge_prompts[i:i + batch_size]))
            print(f"[label_test_set_v2] judging (attempt {attempt}): "
                  f"{min(i + batch_size, len(judge_prompts))}/{len(judge_prompts)}")

        fix_targets = []
        for idx, verdict_output in zip(judge_indices, verdicts):
            passed, reason = parse_judge_verdict(verdict_output)
            if passed:
                if attempt == 0:
                    n_passed_first_try += 1
            else:
                fix_targets.append((idx, reason))

        if not fix_targets:
            break
        if attempt == max_fix_attempts:
            n_still_failing = len(fix_targets)
            print(f"[label_test_set_v2] {n_still_failing} rows still failing after "
                  f"{max_fix_attempts} fix attempt(s) - kept as-is (best available), not discarded.")
            break

        fix_prompts = []
        for idx, reason in fix_targets:
            row = math_rows[idx]
            fix_prompts.append(build_fix_prompt(row["problem"], row["solution"], row["subject"],
                                                 flat_vocab, results[idx]["reasoning_trace"], reason))
        fixed_traces = []
        for i in range(0, len(fix_prompts), batch_size):
            fixed_traces.extend(fixer_fn(fix_prompts[i:i + batch_size]))
            print(f"[label_test_set_v2] fixing: {min(i + batch_size, len(fix_prompts))}/{len(fix_prompts)}")

        for (idx, _), fixed_trace in zip(fix_targets, fixed_traces):
            fixed_trace = normalize_trace_to_bracket_tags(fixed_trace)
            row = math_rows[idx]
            canonical = flat_vocab.get(normalize_subject_key(row["subject"]), [])
            used_raw = parse_skills_from_trace(fixed_trace)
            used_canon, _, _ = hard_filter_skills(used_raw, canonical, fuzzy_threshold)
            results[idx]["reasoning_trace"] = fixed_trace
            results[idx]["skills_used_in_steps"] = " | ".join(used_canon)
            results[idx]["minimum_skills"] = f"MINIMUM_SKILLS: {' | '.join(used_canon)}" if used_canon else ""
            results[idx]["n_skills"] = len(used_canon)
            results[idx]["model_answer"] = extract_boxed(fixed_trace) or results[idx]["model_answer"]
            n_fixed += 1

    ensure_output_path(out_path)
    with open(out_path, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    print(f"\n[label_test_set_v2] wrote {len(results)} labeled test examples -> {out_path}")
    print(f"[label_test_set_v2] judge: {n_passed_first_try} passed first try, "
          f"{n_fixed} corrected by the fixer, {n_still_failing} kept as best-available after fix attempts")
    return results


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--inspect-vocab", action="store_true")
    ap.add_argument("--label", action="store_true")

    ap.add_argument("--train-split", default="train")
    ap.add_argument("--skill-repo", default=None)
    ap.add_argument("--train-skill-labels-file", default=None)

    ap.add_argument("--test-split", default="test")
    ap.add_argument("--labeler-model", default=None, required=False,
                     help="the primary model that annotates solutions and extracts skills")
    ap.add_argument("--judge-model", default=None,
                     help="model used for verification AND semantic canonicalization - "
                          "defaults to the same model as --labeler-model if not set")
    ap.add_argument("--fixer-model", default=None,
                     help="model used to correct rows the judge rejects - defaults to "
                          "--judge-model if not set, or --labeler-model if neither is set")
    ap.add_argument("--use_vllm", action="store_true", default=True)
    ap.add_argument("--quantization", default=None)
    ap.add_argument("--gpu_memory_utilization", type=float, default=0.9)
    ap.add_argument("--max_model_len", type=int, default=4096)
    ap.add_argument("--tensor_parallel_size", type=int, default=1)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--fuzzy_threshold", type=float, default=0.75)
    ap.add_argument("--max_fix_attempts", type=int, default=1)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", default="outputs/test_skill_labels_v2.jsonl")
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
        if not args.labeler_model:
            raise SystemExit("[label_test_set_v2] --label requires --labeler-model")
        judge_model = args.judge_model or args.labeler_model
        fixer_model = args.fixer_model or judge_model

        from run_augmentation import build_backend
        from templates import render_prompt_only

        labeler_backend = build_backend(
            args.labeler_model, args.use_vllm, seed=args.seed,
            gpu_memory_utilization=args.gpu_memory_utilization, max_model_len=args.max_model_len,
            tensor_parallel_size=args.tensor_parallel_size, quantization=args.quantization,
        )
        judge_backend = labeler_backend if judge_model == args.labeler_model else build_backend(
            judge_model, args.use_vllm, seed=args.seed,
            gpu_memory_utilization=args.gpu_memory_utilization, max_model_len=args.max_model_len,
            tensor_parallel_size=args.tensor_parallel_size,
        )
        fixer_backend = judge_backend if fixer_model == judge_model else build_backend(
            fixer_model, args.use_vllm, seed=args.seed,
            gpu_memory_utilization=args.gpu_memory_utilization, max_model_len=args.max_model_len,
            tensor_parallel_size=args.tensor_parallel_size,
        )

        def _make_batch_fn(backend, max_tokens):
            def _fn(prompts):
                formatted = [render_prompt_only(backend.tokenizer, p, system_prompt="") for p in prompts]
                results = backend.generate(formatted, n=1, temperature=0.2, max_tokens=max_tokens)
                return [r[0] for r in results]
            return _fn

        labeler_fn = _make_batch_fn(labeler_backend, 1536)
        judge_fn = _make_batch_fn(judge_backend, 128)
        fixer_fn = _make_batch_fn(fixer_backend, 1536)

        results = label_test_split_v2(
            flat_vocab, labeler_fn, judge_fn, fixer_fn, args.out, split=args.test_split,
            batch_size=args.batch_size, fuzzy_threshold=args.fuzzy_threshold, limit=args.limit,
            max_fix_attempts=args.max_fix_attempts,
        )
        dispatch_destination(args.out, args)

    else:
        print("Pass --inspect-vocab (no model needed) or --label (requires --labeler-model).")
