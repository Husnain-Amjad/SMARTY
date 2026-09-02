"""
Repairs and reformats an already-generated test_skill_labels.jsonl (from
label_test_create.py / label_test_set.py) to match the training skill-label
format exactly.

Why this is needed: real model output (observed directly from
Qwen2.5-Math-7B-Instruct) frequently ignores the exact "[SKILL: X]" bracket
format the prompt asks for, using its own markdown style instead
(**Skill: X**), and ignores the "MINIMUM_SKILLS: A | B" prefix format for
Pass 2's output, answering in \\boxed{...} instead (a near-universal habit
for this model family). The original strict regexes matched literally nothing
in either case, silently producing 0 skills for every single row - not a
generation failure, a PARSING failure. label_test_create.py's parser
functions have since been fixed to tolerate these real formats; this script
re-parses your ALREADY-GENERATED file with the fixed parser instead of
requiring you to regenerate everything (Pass 1 and Pass 2's raw model output,
stored in reasoning_trace/_extraction_raw, was correct all along).

What this fixes fully offline (no model needed):
  - Re-extracts skills_used_in_steps / minimum_skills / n_skills correctly.
  - Rewrites reasoning_trace so every detected skill mention uses the
    canonical [SKILL: X] bracket format, matching the training data's own
    convention exactly, regardless of what style the model originally used.
  - Appends a trailing "ANSWER: \\boxed{value}" line if missing, matching the
    training data's reasoning_trace convention (consumed by
    data_pipeline.strip_trailing_answer_line() when building <think> sections).

What this does NOT fix offline (needs a live model - see --reverify):
  - Pass 3 verification (label_test_set.py only) was run against an EMPTY
    skill list for every row (since Pass 1's skills all silently parsed to
    zero), so verification_status/verification_verified_skills/
    verification_failed_skills reflect that failure, not a real judgment.
    Repaired rows are marked "not_reverified" rather than reusing those
    stale values. Use --reverify (with --model) to re-run ONLY the
    verification pass against the now-correctly-parsed skills - this reuses
    the existing Pass 1/2 output rather than regenerating from scratch.

Usage:
  # offline repair only
  python format_test_labels.py --in outputs/test_skill_labels.jsonl \
      --out outputs/test_skill_labels_fixed.jsonl

  # offline repair + preview the actual <think>/<solution> XML form for a
  # few rows, to visually compare against training examples
  python format_test_labels.py --in outputs/test_skill_labels.jsonl \
      --out outputs/test_skill_labels_fixed.jsonl --preview-xml 3

  # offline repair + re-run Pass 3 verification against the fixed skill list
  python format_test_labels.py --in outputs/test_skill_labels.jsonl \
      --out outputs/test_skill_labels_fixed.jsonl --reverify --model ckpts/stage1_merged
"""

import argparse
import json
import re

from label_test_create import (
    parse_skills_from_trace, parse_minimum_skills, hard_filter_skills, normalize_subject_key,
)
from data_pipeline import build_think_section, strip_trailing_answer_line
from reward_fn import extract_boxed


# Single combined alternation, substituted in ONE pass - this is what avoids
# the double-substitution risk of running each pattern as a separate sequential
# re.sub() call (a later pattern's case-insensitive "Skill:" match could
# otherwise re-match text an earlier pass already normalized into [SKILL: X]).
_COMBINED_SKILL_SUB_RE = re.compile(
    r"\[SKILL:\s*(?P<b1>[^\]\n]+)\]"
    r"|\*\*\s*Skill:\s*(?P<b2>[^*\n]+?)\s*\*\*"
    r"|(?:^|\n)(?P<prefix>[ \t]*[-*]?\s*)Skill:\s*(?P<b3>[^\n*]+)",
    re.IGNORECASE,
)


def normalize_trace_to_bracket_tags(trace: str) -> str:
    """Rewrites every detected skill mention (bracket, bold-markdown, or
    plain line-start) into the canonical [SKILL: X] bracket format."""
    def _sub(m):
        name = (m.group("b1") or m.group("b2") or m.group("b3") or "").strip().rstrip("*").strip()
        if not name:
            return m.group(0)
        prefix = m.group("prefix")
        return f"{prefix}[SKILL: {name}]" if prefix is not None else f"[SKILL: {name}]"
    return _COMBINED_SKILL_SUB_RE.sub(_sub, trace)


def ensure_trailing_answer_line(trace: str, model_answer: str) -> str:
    """Appends 'ANSWER: \\boxed{value}' if the trace doesn't already end with
    one, matching the training data's reasoning_trace convention."""
    if re.search(r"ANSWER:\s*\\boxed\{.*?\}\s*$", trace, re.IGNORECASE | re.DOTALL):
        return trace
    if not model_answer:
        return trace
    return f"{trace.rstrip()}\nANSWER: \\boxed{{{model_answer}}}"


def repair_row(row: dict, flat_vocab: dict, fuzzy_threshold: float = 0.75) -> dict:
    trace = row.get("reasoning_trace", "")
    extraction_raw = row.get("_extraction_raw", "")
    subject = row.get("subject", "unknown")
    canonical = flat_vocab.get(normalize_subject_key(subject), [])

    used_raw = parse_skills_from_trace(trace)
    min_raw = parse_minimum_skills(extraction_raw)

    used_filtered, _, _ = hard_filter_skills(used_raw, canonical, fuzzy_threshold)
    min_filtered, _, _ = hard_filter_skills(min_raw, canonical, fuzzy_threshold)
    final_skills = [s for s in min_filtered if s in used_filtered] or min_filtered or used_filtered

    normalized_trace = normalize_trace_to_bracket_tags(trace)
    normalized_trace = ensure_trailing_answer_line(normalized_trace, row.get("model_answer", ""))

    repaired = dict(row)
    repaired["reasoning_trace"] = normalized_trace
    repaired["skills_used_in_steps"] = " | ".join(used_filtered)
    repaired["minimum_skills"] = " | ".join(final_skills)
    repaired["n_skills"] = len(final_skills)
    if "verification_status" in row:
        repaired["verification_status"] = "not_reverified"
        repaired["verification_verified_skills"] = ""
        repaired["verification_failed_skills"] = ""
    return repaired


def repair_file(in_path: str, out_path: str, flat_vocab: dict, fuzzy_threshold: float = 0.75):
    rows = []
    with open(in_path) as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))

    # Loud, early warning for a real, previously-silent failure mode: if a
    # subject actually present in this file has NO canonical vocabulary at
    # all, every skill for that subject will be rejected regardless of how
    # correct they are - this is virtually always a subject-key mismatch
    # (e.g. "algebra" vs "Algebra" between this file and the train-labeled
    # data), not a legitimate "this subject has no skills" case.
    subjects_in_file = {r.get("subject", "unknown") for r in rows}
    for subject in subjects_in_file:
        if not flat_vocab.get(normalize_subject_key(subject)):
            print(f"[format_test_labels] WARNING: no canonical vocabulary found for "
                  f"subject '{subject}' - every skill for this subject's rows will be "
                  f"rejected as out-of-vocabulary, likely INCORRECTLY. This usually means "
                  f"a subject-name mismatch between this file and the train-labeled data "
                  f"used to build the vocabulary (--skill-repo/--train-skill-labels-file) - "
                  f"run 'python label_test_create.py --inspect-vocab' with the same "
                  f"train-data arguments to see the actual subject keys available.")

    n_before_empty = sum(1 for r in rows if not r.get("minimum_skills"))
    repaired = [repair_row(r, flat_vocab, fuzzy_threshold) for r in rows]
    n_after_empty = sum(1 for r in repaired if not r.get("minimum_skills"))

    with open(out_path, "w") as f:
        for r in repaired:
            f.write(json.dumps(r) + "\n")

    print(f"[format_test_labels] repaired {len(rows)} rows -> {out_path}")
    print(f"[format_test_labels] rows with empty minimum_skills: {n_before_empty} before -> "
          f"{n_after_empty} after")
    if "verification_status" in rows[0] if rows else False:
        print(f"[format_test_labels] verification_status reset to 'not_reverified' on repaired "
              f"rows - the original verification ran against an empty skill list and isn't "
              f"meaningful. Use --reverify to re-run Pass 3 against the corrected skills.")
    return repaired


def preview_xml(repaired_rows: list, n: int = 3):
    print("\n" + "=" * 70)
    print(f"PREVIEW: rendered <think>/<solution> form for {min(n, len(repaired_rows))} rows")
    print("(this is what data_pipeline.build_think_section() + <solution> produces at "
          "SFT/eval render time - compare this shape directly against training examples)")
    print("=" * 70)
    for row in repaired_rows[:n]:
        think = build_think_section(row)
        solution = row.get("original_solution", "")
        print(f"\n--- {row.get('subject')} / {row.get('level')} ---")
        print(f"<think>\n{think}\n</think>")
        print(f"<solution>\n{solution}\n</solution>")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path", required=True)
    ap.add_argument("--out", dest="out_path", required=True)
    ap.add_argument("--train-split", default="train")
    ap.add_argument("--skill-repo", default=None)
    ap.add_argument("--train-skill-labels-file", default=None)
    ap.add_argument("--fuzzy_threshold", type=float, default=0.75)
    ap.add_argument("--preview-xml", type=int, default=0, metavar="N",
                     help="print the rendered <think>/<solution> form for N rows after repair")
    ap.add_argument("--reverify", action="store_true",
                     help="re-run Pass 3 verification against the corrected skills (needs --model)")
    ap.add_argument("--model", default=None, help="required if --reverify is set")
    ap.add_argument("--verify_batch_size", type=int, default=32)
    ap.add_argument("--use_vllm", action="store_true", default=True)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    from label_test_create import extract_canonical_vocabulary
    _, flat_vocab = extract_canonical_vocabulary(
        args.train_split, skill_repo=args.skill_repo, skill_labels_file=args.train_skill_labels_file,
    )

    repaired = repair_file(args.in_path, args.out_path, flat_vocab, args.fuzzy_threshold)

    if args.preview_xml:
        preview_xml(repaired, args.preview_xml)

    if args.reverify:
        if not args.model:
            raise SystemExit("[format_test_labels] --reverify requires --model")
        from run_augmentation import build_backend
        from templates import render_prompt_only
        from label_test_set import verify_skill_usage
        import re as _re

        backend = build_backend(args.model, args.use_vllm, seed=args.seed)

        def batch_generate_fn(prompts):
            formatted = [render_prompt_only(backend.tokenizer, p, system_prompt="") for p in prompts]
            results = backend.generate(formatted, n=1, temperature=0.2, max_tokens=256)
            return [r[0] for r in results]

        step_re = _re.compile(r"\[SKILL:\s*([^\]]+)\]")
        rows_for_verification = []
        for row in repaired:
            parts = step_re.split(row["reasoning_trace"])
            tagged = []
            used = set(row.get("skills_used_in_steps", "").split(" | "))
            for j in range(1, len(parts), 2):
                skill = parts[j].strip()
                step_text = parts[j + 1].strip() if j + 1 < len(parts) else ""
                if skill in used and step_text:
                    tagged.append((skill, step_text))
            rows_for_verification.append({"problem": row["problem"], "tagged_steps": tagged})

        verified_per_row, failed_per_row, n_calls = verify_skill_usage(
            rows_for_verification, batch_generate_fn, batch_size=args.verify_batch_size,
        )
        print(f"[format_test_labels] re-verification made {n_calls} model calls")

        for row, verified, failed in zip(repaired, verified_per_row, failed_per_row):
            min_skills = [s for s in row["minimum_skills"].split(" | ") if s]
            final = [s for s in min_skills if s in verified] or min_skills
            row["minimum_skills"] = " | ".join(final)
            row["n_skills"] = len(final)
            row["verification_status"] = "verified" if any(s in verified for s in min_skills) else "fallback_unverified"
            row["verification_verified_skills"] = " | ".join(sorted(verified))
            row["verification_failed_skills"] = " | ".join(sorted(failed))

        with open(args.out_path, "w") as f:
            for r in repaired:
                f.write(json.dumps(r) + "\n")
        print(f"[format_test_labels] re-verification complete, updated -> {args.out_path}")
