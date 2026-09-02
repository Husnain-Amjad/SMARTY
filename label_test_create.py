"""
Creates skill labels for the MATH TEST split, constrained to the canonical
skill vocabulary that was actually realized in the TRAIN split's skill labels
- so evaluation on the test set never introduces a skill the model was never
trained to use, and skill-prediction/skill-usage metrics on test genuinely
measure retention/generalization of a fixed, closed vocabulary rather than
partial credit for a vocabulary mismatch.

Two things this script does that the original labeling pipeline didn't:
  1. Extracts the allowed-skill list PER SUBJECT directly from the train
     labels' `minimum_skills` column (not a static hand-written list) - this
     is the vocabulary that was actually learnable, not aspirational.
  2. Applies a HARD post-hoc filter after generation: any skill the model
     produces that isn't in that subject's canonical set is either fuzzy-
     remapped to the nearest canonical skill (difflib similarity, matching
     evaluator.py's own normalize_skill/skill_set_metrics logic - no extra
     embedding-model dependency needed) or dropped and counted as rejected.
     A prompt-level constraint alone does not prevent this leakage - see
     the hallucination discussion this design followed from.

Output schema matches data_pipeline.py's expected skill-label columns exactly
(subject, level, problem, original_solution, model_answer, reasoning_trace,
skills_used_in_steps, minimum_skills, n_skills, _extraction_raw) so the result
loads directly via data_pipeline.load_skill_labels(split="test", ...) with no
further conversion.

Usage:
  # Step 1: see what the canonical vocabulary actually is, before generating anything
  python label_test_create.py --inspect-vocab \
      --train-skill-labels-file data/skill_labels.jsonl.example

  # Step 2: generate test-set labels constrained to that vocabulary
  python label_test_create.py --label \
      --train-skill-labels-file data/skill_labels.jsonl.example \
      --model ckpts/qwen7b_stage1_merged \
      --out outputs/test_skill_labels.jsonl
"""

import argparse
import json
import re
from collections import Counter, defaultdict

from data_pipeline import (
    SUBJECTS, load_hendrycks_math, load_skill_labels, clean_skill_list,
    strip_trailing_answer_line, _hash_problem,
)
from reward_fn import extract_boxed
from storage_utils import ensure_output_path, add_destination_args, dispatch_destination

# Real model output frequently ignores the exact "[SKILL: X]" format we ask for
# in the prompt, despite explicit instructions - Qwen2.5-Math models in
# particular tend to use their own natural markdown style instead. Matching
# only the literal bracket format silently produces zero skills for every
# example (a systemic failure discovered by inspecting real generated output,
# not a hypothetical). These three patterns cover every real variant observed:
_SKILL_TAG_PATTERNS = [
    re.compile(r"\[SKILL:\s*([^\]\n]+)\]"),                          # [SKILL: X]  (the requested format)
    re.compile(r"\*\*\s*Skill:\s*([^*\n]+?)\s*\*\*", re.IGNORECASE),  # **Skill: X**  (bold markdown - the common actual output)
    re.compile(r"(?:^|\n)[ \t]*[-*]?\s*Skill:\s*([^\n*]+)", re.IGNORECASE),  # Skill: X / - Skill: X (plain, line-start)
]


# ---------------------------------------------------------------------------
# Step 1: canonical vocabulary extraction from train labels
# ---------------------------------------------------------------------------

def normalize_subject_key(subject: str) -> str:
    """
    Normalizes a subject string for use as a canonical-vocabulary dict key -
    lowercase and whitespace-stripped. Without this, a real, observed failure
    mode is silent: if the train-labeled dataset's own "subject" column uses
    different casing/formatting than a test file's "subject" field (e.g.
    "Algebra" vs "algebra"), flat_vocab.get(subject, []) returns an empty
    list for a perfectly real subject, which then rejects EVERY skill as
    "out of vocabulary" - not because the skills are wrong, but because the
    lookup key itself never matched. Applying this same normalization both
    when building the vocabulary and when looking it up closes that gap.
    """
    return (subject or "unknown").strip().lower()


def extract_canonical_vocabulary(train_split: str = "train", skill_repo: str = None,
                                  skill_labels_file: str = None):
    """
    Returns:
      per_subject: {normalized_subject: Counter({skill_name: frequency})}
      flat_vocab:  {normalized_subject: sorted list of skill names actually
                    used in train, sorted by descending frequency}
    Subject keys are normalized (see normalize_subject_key) - always look
    them up the same way, not with a raw/un-normalized subject string.
    Reads the SAME minimum_skills column build_think_section() already parses,
    so this is exactly "what the model was actually trained to produce," not
    a separately-maintained list that can drift out of sync.
    """
    kwargs = {}
    if skill_repo:
        kwargs["repo_id"] = skill_repo
    labels = load_skill_labels(train_split, local_path=skill_labels_file, **kwargs) if skill_labels_file \
        else load_skill_labels(train_split, **kwargs)

    per_subject = defaultdict(Counter)
    for row in labels.values():
        subject = normalize_subject_key(row.get("subject", "unknown"))
        skills_raw = clean_skill_list(row.get("minimum_skills", "")) or row.get("skills_used_in_steps", "")
        for skill in skills_raw.split("|"):
            skill = skill.strip()
            if skill:
                per_subject[subject][skill] += 1

    flat_vocab = {
        subj: [s for s, _ in counter.most_common()]
        for subj, counter in per_subject.items()
    }
    return per_subject, flat_vocab


def print_vocab_report(per_subject):
    print("=" * 70)
    print("CANONICAL SKILL VOCABULARY (extracted from TRAIN labels)")
    print("=" * 70)
    total_skills = 0
    for subject in sorted(per_subject.keys()):
        counter = per_subject[subject]
        print(f"\n{subject} ({len(counter)} distinct skills):")
        for skill, freq in counter.most_common():
            print(f"  {freq:4d}  {skill}")
        total_skills += len(counter)
    print(f"\n[label_test_create] {total_skills} distinct (subject, skill) pairs total "
          f"across {len(per_subject)} subjects")


# ---------------------------------------------------------------------------
# Step 2: two-pass generation, constrained to the canonical vocabulary
# ---------------------------------------------------------------------------

ANNOTATE_SYSTEM_TEMPLATE = """You are a precise math solution annotator.
You will be given a math problem AND its correct, complete reference solution.
Your task is NOT to solve the problem yourself or produce a different derivation -
it is to ANNOTATE the reference solution exactly as given, breaking it into its
logical steps and labeling EACH step with the skill it demonstrates.

Do not change, correct, shorten, extend, or re-derive the mathematical content of
the reference solution. Reproduce its actual reasoning faithfully, broken into
steps, each preceded by:
[SKILL: <skill_name>]

Each skill MUST come from this allowed list EXACTLY as written - do not invent new
skill names:
{allowed_skills}
A skill may only appear ONCE across all steps unless used in a genuinely new sub-problem.

End with:
ANSWER: \\boxed{{value}}
(this must match the reference solution's own final answer exactly - you are not
computing a new answer, you are reporting the reference solution's answer)
"""

EXTRACT_SYSTEM = """You are a strict skill auditor for math solutions.
You will be given a math problem, a step-by-step ANNOTATED reference solution with
[SKILL: ...] labels, and the correct answer.
List ONLY the skills that were DIRECTLY needed to reach the final answer, removing
any skill that appeared in an unnecessary or redundant step.
Output format (one line, pipe-separated, nothing else):
  MINIMUM_SKILLS: skill1 | skill2 | skill3
Do NOT output anything else. No explanation. No numbering.
"""


def build_annotate_prompt(problem: str, reference_solution: str, subject: str, flat_vocab: dict) -> str:
    """
    Grounds skill labeling in the ACTUAL reference solution, rather than asking
    the model to independently re-derive the problem (see module-level design note
    above) - this is what prevents a model that solves incorrectly, or via a
    different path than the reference, from producing skill labels that don't
    reflect the true required derivation.
    """
    allowed = " | ".join(flat_vocab.get(normalize_subject_key(subject), ["(no canonical skills recorded for this subject)"]))
    system = ANNOTATE_SYSTEM_TEMPLATE.format(allowed_skills=allowed)
    return (f"{system}\n\nProblem:\n{problem}\n\n"
            f"Reference solution (annotate this exactly - do not re-derive or solve "
            f"independently):\n{reference_solution}\n\nAnnotated solution:")


# Alias under the old name for anything that still imports it - note the SIGNATURE
# changed (reference_solution is now a required argument) since the old 3-arg
# version's behavior (independent re-derivation) was the actual bug being fixed here,
# not something worth preserving under either name.
def build_solve_prompt(problem: str, reference_solution: str, subject: str, flat_vocab: dict) -> str:
    return build_annotate_prompt(problem, reference_solution, subject, flat_vocab)


def build_extract_prompt(problem: str, trace: str, correct_answer: str) -> str:
    return (f"{EXTRACT_SYSTEM}\n\nProblem:\n{problem}\n\nAnnotated reference solution:\n{trace}\n\n"
            f"Correct answer: {correct_answer}\n\nList the minimum skills needed.")


def parse_skills_from_trace(trace: str) -> list:
    found = []
    for pattern in _SKILL_TAG_PATTERNS:
        found.extend(pattern.findall(trace))
    seen, unique = set(), []
    for s in found:
        s = s.strip().rstrip("*").strip()  # strip stray trailing markdown asterisks
        key = s.lower()
        if s and key not in seen:
            seen.add(key)
            unique.append(s)
    return unique


def parse_minimum_skills(extraction_output: str) -> list:
    m = re.search(r"MINIMUM_SKILLS:\s*(.+)", extraction_output, re.IGNORECASE)
    if m:
        return [s.strip() for s in m.group(1).split("|") if s.strip()]

    # Fallback: Qwen-Math-family models almost always answer in \boxed{...}
    # regardless of what format the prompt actually asked for (a strong,
    # near-universal habit observed in real output) - extract that boxed
    # content with the same brace-balanced parser used for final answers
    # elsewhere in this pipeline (handles nested \text{} braces correctly,
    # unlike a naive regex), then split on comma (the model's natural
    # list-separator inside a boxed answer, not the pipe we originally asked
    # for), stripping any \text{...} LaTeX wrapping it commonly adds per item.
    boxed = extract_boxed(extraction_output)
    if not boxed:
        return []
    boxed = boxed.strip()

    # Two real observed \text{} wrapping styles need different handling:
    #   (a) ONE \text{} wraps the whole comma-separated list:
    #       \boxed{\text{Skill A, Skill B, Skill C}}
    #   (b) EACH item is wrapped separately: \boxed{\text{A}, \text{B}, \text{C}}
    # A single greedy regex can't tell these apart (multiple \text{} groups
    # look like one big wrapper to a greedy `.*`) - count occurrences instead.
    n_text_wraps = boxed.count("\\text{")
    if n_text_wraps == 1 and boxed.startswith("\\text{") and boxed.endswith("}"):
        boxed = boxed[len("\\text{"):-1]
        return [x.strip() for x in boxed.split(",") if x.strip()]

    items = [x.strip() for x in boxed.split(",") if x.strip()]
    cleaned = []
    for item in items:
        per_item = re.match(r"^\\text\{([^{}]+)\}$", item)
        cleaned.append(per_item.group(1).strip() if per_item else item)
    return cleaned


# Single combined alternation, substituted in ONE pass - this is what avoids the
# double-substitution risk of running each tag-style pattern as a separate
# sequential re.sub() call (a later pattern's case-insensitive "Skill:" match
# could otherwise re-match text an earlier pass already normalized).
_COMBINED_SKILL_SUB_RE = re.compile(
    r"\[SKILL:\s*(?P<b1>[^\]\n]+)\]"
    r"|\*\*\s*Skill:\s*(?P<b2>[^*\n]+?)\s*\*\*"
    r"|(?:^|\n)(?P<prefix>[ \t]*[-*]?\s*)Skill:\s*(?P<b3>[^\n*]+)",
    re.IGNORECASE,
)


def normalize_trace_to_bracket_tags(trace: str) -> str:
    """
    Rewrites every detected skill mention (bracket, bold-markdown, or plain
    line-start - see _SKILL_TAG_PATTERNS) into the canonical [SKILL: X]
    bracket format. Applied right after Pass 1 generation in both
    label_test_split() below and label_and_verify_test_split() in
    label_test_set.py, so every downstream step (vocabulary filtering, Pass 3
    step-splitting for verification) can rely on one consistent tag format
    regardless of which style the model actually used.
    """
    def _sub(m):
        name = (m.group("b1") or m.group("b2") or m.group("b3") or "").strip().rstrip("*").strip()
        if not name:
            return m.group(0)
        prefix = m.group("prefix")
        return f"{prefix}[SKILL: {name}]" if prefix is not None else f"[SKILL: {name}]"
    return _COMBINED_SKILL_SUB_RE.sub(_sub, trace)


# ---------------------------------------------------------------------------
# Hard post-hoc filter: this is what actually prevents vocabulary leakage,
# not the prompt constraint above (models leak past closed-list instructions
# regardless of how the prompt is worded - see the design discussion this
# script implements).
# ---------------------------------------------------------------------------

def _similarity(a: str, b: str) -> float:
    import difflib
    return difflib.SequenceMatcher(None, a.lower().strip(), b.lower().strip()).ratio()


def hard_filter_skills(raw_skills: list, canonical_skills: list, fuzzy_threshold: float = 0.75):
    """
    For each raw skill: exact match (case-insensitive) -> keep as-is.
    Else fuzzy match >= threshold against the canonical list -> remap to the
    canonical spelling. Else -> reject (dropped, counted, not silently kept).
    Returns (filtered_skills, n_rejected, rejected_examples).
    """
    canonical_lower = {c.lower(): c for c in canonical_skills}
    filtered, rejected = [], []

    for raw in raw_skills:
        raw_norm = raw.strip()
        if raw_norm.lower() in canonical_lower:
            filtered.append(canonical_lower[raw_norm.lower()])
            continue
        best_match, best_score = None, 0.0
        for canon in canonical_skills:
            score = _similarity(raw_norm, canon)
            if score > best_score:
                best_match, best_score = canon, score
        if best_score >= fuzzy_threshold:
            filtered.append(best_match)
        else:
            rejected.append(raw_norm)

    # de-dupe while preserving order
    seen, deduped = set(), []
    for s in filtered:
        if s not in seen:
            seen.add(s)
            deduped.append(s)
    return deduped, len(rejected), rejected


# ---------------------------------------------------------------------------
# Batched generation + assembly (mirrors run_augmentation.py's batching pattern)
# ---------------------------------------------------------------------------

def label_test_split(flat_vocab: dict, batch_generate_fn, out_path: str,
                      split: str = "test", batch_size: int = 32, fuzzy_threshold: float = 0.75,
                      limit: int = None):
    """
    batch_generate_fn: callable(prompts: list[str]) -> list[str]
        A single batched model call, same contract as data_pipeline.py's
        augmentation functions - plug in run_augmentation.py's backend for
        the actual model (see __main__ below).
    """
    math_rows = load_hendrycks_math(split)
    if limit:
        math_rows = math_rows[:limit]
    print(f"[label_test_create] labeling {len(math_rows)} problems from split='{split}'")

    # --- Pass 1: annotate the REFERENCE solution with skill tags (not an independent re-derivation) ---
    solve_prompts = [build_annotate_prompt(r["problem"], r["solution"], r["subject"], flat_vocab) for r in math_rows]
    traces = []
    for i in range(0, len(solve_prompts), batch_size):
        batch = solve_prompts[i:i + batch_size]
        traces.extend(batch_generate_fn(batch))
        print(f"[label_test_create] pass 1 (annotate reference solution): {min(i + batch_size, len(solve_prompts))}/{len(solve_prompts)}")
    # Real model output frequently ignores the requested [SKILL: X] bracket
    # format (observed directly: Qwen2.5-Math models commonly use **Skill: X**
    # markdown instead) - normalize immediately so every downstream step
    # (vocabulary filtering, step-splitting) works against one consistent format.
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
        print(f"[label_test_create] pass 2 (extract): {min(i + batch_size, len(extract_prompts))}/{len(extract_prompts)}")

    # --- Assemble + hard-filter ---
    results = []
    total_rejected = 0
    rejected_samples = []
    for row, trace, extraction_raw in zip(math_rows, traces, extraction_raw_list):
        used_in_steps = parse_skills_from_trace(trace)
        minimum_skills_raw = parse_minimum_skills(extraction_raw)
        canonical_for_subject = flat_vocab.get(normalize_subject_key(row["subject"]), [])

        used_filtered, n_rej1, rej1 = hard_filter_skills(used_in_steps, canonical_for_subject, fuzzy_threshold)
        min_filtered, n_rej2, rej2 = hard_filter_skills(minimum_skills_raw, canonical_for_subject, fuzzy_threshold)
        total_rejected += n_rej1 + n_rej2
        rejected_samples.extend(rej1 + rej2)

        final_skills = [s for s in min_filtered if s in used_filtered] or min_filtered or used_filtered

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
            "_extraction_raw": extraction_raw,
        })

    ensure_output_path(out_path)
    with open(out_path, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    print(f"\n[label_test_create] wrote {len(results)} labeled test examples -> {out_path}")
    print(f"[label_test_create] {total_rejected} raw skill mentions rejected as out-of-vocabulary "
          f"(fuzzy_threshold={fuzzy_threshold})")
    if rejected_samples:
        sample_counts = Counter(rejected_samples).most_common(10)
        print("[label_test_create] most common rejected (hallucinated/off-vocabulary) skill names:")
        for s, c in sample_counts:
            print(f"    {c:3d}  {s}")
    return results


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--inspect-vocab", action="store_true",
                     help="print the canonical vocabulary extracted from train labels and exit - "
                          "run this FIRST to sanity-check the vocabulary before spending model "
                          "compute on generation.")
    ap.add_argument("--label", action="store_true",
                     help="generate and hard-filter test-set skill labels")

    ap.add_argument("--train-split", default="train")
    ap.add_argument("--skill-repo", default=None,
                     help="HF dataset repo for train labels (default: data_pipeline.DEFAULT_SKILL_REPO)")
    ap.add_argument("--train-skill-labels-file", default=None,
                     help="local jsonl/csv override instead of --skill-repo")

    ap.add_argument("--test-split", default="test")
    ap.add_argument("--model", default=None, help="required for --label")
    ap.add_argument("--use_vllm", action="store_true", default=True)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--fuzzy_threshold", type=float, default=0.75)
    ap.add_argument("--limit", type=int, default=None, help="cap problems for a quick test")
    ap.add_argument("--out", default="outputs/test_skill_labels.jsonl")
    ap.add_argument("--seed", type=int, default=42)
    add_destination_args(ap, default_repo_type="dataset")
    args = ap.parse_args()

    per_subject, flat_vocab = extract_canonical_vocabulary(
        args.train_split, skill_repo=args.skill_repo, skill_labels_file=args.train_skill_labels_file,
    )

    if args.inspect_vocab:
        print_vocab_report(per_subject)

    elif args.label:
        if not args.model:
            raise SystemExit("[label_test_create] --label requires --model")
        from run_augmentation import build_backend

        backend = build_backend(args.model, args.use_vllm, seed=args.seed)

        def batch_generate_fn(prompts):
            # prompts here are already full system+user text (see build_solve_prompt /
            # build_extract_prompt) - wrap minimally with this model's own turn tokens
            # via templates.py rather than assuming a specific chat format.
            from templates import render_prompt_only
            formatted = [render_prompt_only(backend.tokenizer, p, system_prompt="") for p in prompts]
            results = backend.generate(formatted, n=1, temperature=0.2, max_tokens=1536)
            return [r[0] for r in results]

        results = label_test_split(
            flat_vocab, batch_generate_fn, args.out, split=args.test_split,
            batch_size=args.batch_size, fuzzy_threshold=args.fuzzy_threshold, limit=args.limit,
        )
        dispatch_destination(args.out, args)

    else:
        print("Pass --inspect-vocab (no model needed) or --label (requires --model). "
              "Run --inspect-vocab first to sanity-check the vocabulary before generating.")
