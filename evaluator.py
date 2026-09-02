"""
Evaluation module: scores model predictions along four dimensions, then
supports comparing multiple runs (ablations) against a baseline with
statistical-significance-aware charts and tables.

Metric families:
  1. final_correct          - boxed-answer match against gold (always available)
  2. intermediate reasoning  - rule-based arithmetic-consistency proxy on each
     correctness                step's extractable "A = B" assertions (sympy-verified),
                                plus an OPTIONAL LLM-judge hook for a stronger check
                                (see judge_steps_batch / run_judge_eval.py)
  3. skill prediction         - does the model's own predicted skill set (parsed from
     correctness                its <think> section) match the reference minimum_skills
                                for that problem (exact + fuzzy set metrics)
  4. correct skill usage      - for each step where the model tagged a skill, is that
                                skill plausible for this problem's known skill vocabulary
                                (fuzzy-matched against the reference) - a cheaper proxy
                                for "did it use a sensible skill tag" distinct from #3's
                                problem-level set match. A full per-step-position check
                                needs the LLM-judge hook.

IMPORTANT CAVEAT: skill labels (from load_skill_labels) currently only cover the MATH
TRAIN split (per the labeling work described by the user). Metrics #3 and #4 are only
meaningful for predictions on problems that ARE in the labeled set - for predictions on
the test split (which has no skill labels), those metrics come back as None per-example
and are excluded from aggregates, with a coverage percentage printed so this is visible
rather than silently producing a misleading number. final_correct (#1) and the rule-based
arithmetic-consistency proxy (#2) do not depend on the skill-label split at all.

CLI:
  python evaluator.py --score --predictions outputs/predictions_run1.jsonl \
      --split train --out-detailed outputs/eval_run1_detailed.jsonl \
      --out-summary outputs/eval_run1_summary.json

  python evaluator.py --compare --run baseline=outputs/eval_baseline_summary.json \
      --run lora_run1=outputs/eval_run1_summary.json --baseline baseline \
      --out-dir outputs/comparison
"""

import argparse
import json
import re
import math
import difflib
from collections import defaultdict
from pathlib import Path

from reward_fn import extract_boxed, answers_match
from data_pipeline import load_skill_labels, clean_skill_list, DEFAULT_SKILL_REPO
from storage_utils import ensure_output_path, ensure_dir, require_input_path, add_destination_args, dispatch_destination

_SKILL_TAG_RE = re.compile(r"\[SKILL:\s*([^\]]+)\]")
_RELEVANT_SKILLS_HEADER_RE = re.compile(r"Relevant skills:\s*(.+)", re.IGNORECASE)
_THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL | re.IGNORECASE)
_SOLUTION_RE = re.compile(r"<solution>(.*?)</solution>", re.DOTALL | re.IGNORECASE)


# ---------------------------------------------------------------------------
# Parsing model output
# ---------------------------------------------------------------------------

def split_think_solution(text: str):
    m_think = _THINK_RE.search(text)
    m_sol = _SOLUTION_RE.search(text)
    think = m_think.group(1).strip() if m_think else None
    solution = m_sol.group(1).strip() if m_sol else None
    return think, solution


def extract_predicted_skills(text: str) -> set:
    skills = set()
    header = _RELEVANT_SKILLS_HEADER_RE.search(text)
    if header:
        for s in header.group(1).split("|"):
            s = s.strip().rstrip(".")
            if s:
                skills.add(s)
    for m in _SKILL_TAG_RE.finditer(text):
        s = m.group(1).strip()
        if s:
            skills.add(s)
    return skills


def split_into_tagged_steps(text: str):
    """Splits on [SKILL: ...] markers -> list of (skill_or_None, step_text)."""
    parts = _SKILL_TAG_RE.split(text)
    if len(parts) == 1:
        return [(None, text.strip())] if text.strip() else []
    steps = []
    pre = parts[0].strip()
    if pre:
        steps.append((None, pre))
    for i in range(1, len(parts), 2):
        skill = parts[i].strip()
        step_text = parts[i + 1].strip() if i + 1 < len(parts) else ""
        if step_text:
            steps.append((skill, step_text))
    return steps


def parse_model_output(text: str) -> dict:
    think, solution = split_think_solution(text)
    source_for_skills = think if think is not None else text
    predicted_skills = extract_predicted_skills(source_for_skills)
    steps = split_into_tagged_steps(source_for_skills)
    final_answer = extract_boxed(solution) if solution is not None else extract_boxed(text)
    return {
        "think": think, "solution": solution,
        "predicted_skills": predicted_skills, "steps": steps,
        "final_answer": final_answer,
        "has_think": think is not None, "has_boxed": final_answer is not None,
    }


# ---------------------------------------------------------------------------
# Skill-set metrics (metric family #3)
# ---------------------------------------------------------------------------

def normalize_skill(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"[^a-z0-9 ]", "", s)
    s = re.sub(r"\s+", " ", s)
    return s


def _f1(p, r):
    return 0.0 if (p + r) == 0 else 2 * p * r / (p + r)


def skill_set_metrics(predicted: set, reference: set, fuzzy_threshold: float = 0.8) -> dict:
    pred_norm = {normalize_skill(s) for s in predicted if s}
    ref_norm = {normalize_skill(s) for s in reference if s}

    exact_overlap = pred_norm & ref_norm
    exact_precision = (len(exact_overlap) / len(pred_norm)) if pred_norm else (1.0 if not ref_norm else 0.0)
    exact_recall = (len(exact_overlap) / len(ref_norm)) if ref_norm else (1.0 if not pred_norm else 0.0)

    fuzzy_matched_ref, fuzzy_matched_pred = set(), set()
    for r in ref_norm:
        for p in pred_norm:
            if difflib.SequenceMatcher(None, r, p).ratio() >= fuzzy_threshold:
                fuzzy_matched_ref.add(r)
                fuzzy_matched_pred.add(p)
    fuzzy_precision = (len(fuzzy_matched_pred) / len(pred_norm)) if pred_norm else (1.0 if not ref_norm else 0.0)
    fuzzy_recall = (len(fuzzy_matched_ref) / len(ref_norm)) if ref_norm else (1.0 if not pred_norm else 0.0)

    return {
        "exact_precision": exact_precision, "exact_recall": exact_recall,
        "exact_f1": _f1(exact_precision, exact_recall),
        "fuzzy_precision": fuzzy_precision, "fuzzy_recall": fuzzy_recall,
        "fuzzy_f1": _f1(fuzzy_precision, fuzzy_recall),
        "exact_match": pred_norm == ref_norm,
        "n_predicted": len(pred_norm), "n_reference": len(ref_norm),
    }


# ---------------------------------------------------------------------------
# Rule-based intermediate-reasoning-correctness proxy (metric family #2)
# ---------------------------------------------------------------------------

def _clean_expr_for_sympy(x: str) -> str:
    x = x.strip()
    x = x.replace("$", "").replace("\\left", "").replace("\\right", "")
    x = x.replace("\\cdot", "*").replace("\\times", "*")
    x = re.sub(r"\\frac\{([^{}]+)\}\{([^{}]+)\}", r"(\1)/(\2)", x)
    x = re.sub(r"\\sqrt\{([^{}]+)\}", r"sqrt(\1)", x)
    x = x.replace("\\pi", "pi")
    x = x.replace("^", "**")
    x = x.replace("\\", "")
    return x


def arithmetic_consistency_score(step_text: str):
    """
    Crude, rule-based proxy: finds single 'A = B' assertions per line and checks
    numeric/symbolic equality via sympy. Returns (n_checkable, n_correct) - most
    step text isn't a clean checkable equality (verbal reasoning, multi-equals
    chains, etc.), so this is a lower-bound signal, not a full step-correctness
    judge. Use judge_steps_batch for a stronger LLM-based check.
    """
    from sympy import sympify, simplify
    import warnings
    n_checkable, n_correct = 0, 0
    for line in step_text.splitlines():
        line = line.strip()
        if line.count("=") != 1:
            continue
        lhs, rhs = line.split("=")
        lhs, rhs = _clean_expr_for_sympy(lhs), _clean_expr_for_sympy(rhs)
        if not lhs or not rhs:
            continue
        try:
            with warnings.catch_warnings():
                # sympify() parses via an internal eval() on the cleaned string -
                # malformed model output (e.g. a stray "{...}(" that looks like
                # calling a set literal) can make Python's own parser emit a
                # SyntaxWarning before the resulting TypeError is raised and
                # caught below. The warning is harmless noise, not a real issue -
                # this line is already being discarded as non-checkable either way.
                warnings.simplefilter("ignore", SyntaxWarning)
                vl, vr = sympify(lhs), sympify(rhs)
            n_checkable += 1
            if simplify(vl - vr) == 0:
                n_correct += 1
        except Exception:
            continue
    return n_checkable, n_correct


# ---------------------------------------------------------------------------
# Skill-usage validity proxy (metric family #4)
# ---------------------------------------------------------------------------

def skill_usage_validity(steps, reference_skills: set, fuzzy_threshold: float = 0.8):
    """
    Proxy: for each step where the model tagged a skill, checks whether that
    skill plausibly belongs to this problem's known skill vocabulary (fuzzy
    match against reference minimum_skills/skills_used_in_steps). Catches the
    cheap, common failure mode of a fabricated/off-vocabulary tag anywhere in
    the trace - it does NOT verify the tag is right for that step's specific
    position in the derivation (that needs judge_steps_batch).
    Returns None if the model tagged no steps at all (nothing to evaluate).
    """
    tagged = [(skill, text) for skill, text in steps if skill]
    if not tagged:
        return None
    ref_norm = {normalize_skill(s) for s in reference_skills if s}
    if not ref_norm:
        return None
    n_valid = 0
    for skill, _ in tagged:
        s_norm = normalize_skill(skill)
        if s_norm in ref_norm or any(
            difflib.SequenceMatcher(None, s_norm, r).ratio() >= fuzzy_threshold for r in ref_norm
        ):
            n_valid += 1
    return n_valid / len(tagged)


# ---------------------------------------------------------------------------
# Optional LLM-judge hook for stronger step-level correctness (metric family #2, strong version)
# ---------------------------------------------------------------------------

JUDGE_PROMPT_TEMPLATE = (
    "You are grading ONE reasoning step from a student's solution to a math problem.\n"
    "Problem: {problem}\n"
    "Reference correct solution (for your context only): {reference_solution}\n"
    "Step to grade: \"{step_text}\"\n"
    "Is this step mathematically valid and consistent with a correct solution path? "
    "Answer with exactly one word: CORRECT or INCORRECT."
)


def judge_steps_batch(examples_with_steps: list, batch_judge_fn, batch_size: int = 32):
    """
    examples_with_steps: list of {'problem', 'reference_solution', 'steps': [str, ...]}
    batch_judge_fn: callable(prompts: list[str]) -> list[str] (batched model call -
        reuse run_augmentation.build_backend to supply this; see run_judge_eval.py)
    Returns a list of (n_steps_judged, n_judged_correct) aligned to examples_with_steps.
    """
    jobs = []
    for ex_idx, ex in enumerate(examples_with_steps):
        for step_text in ex["steps"]:
            jobs.append((ex_idx, JUDGE_PROMPT_TEMPLATE.format(
                problem=ex["problem"], reference_solution=ex["reference_solution"],
                step_text=step_text,
            )))

    tallies = defaultdict(lambda: [0, 0])  # ex_idx -> [n_total, n_correct]
    for i in range(0, len(jobs), batch_size):
        batch = jobs[i:i + batch_size]
        outs = batch_judge_fn([p for _, p in batch])
        for (ex_idx, _), out in zip(batch, outs):
            tallies[ex_idx][0] += 1
            out_up = (out or "").upper()
            if "CORRECT" in out_up and "INCORRECT" not in out_up:
                tallies[ex_idx][1] += 1

    return [tuple(tallies.get(i, [0, 0])) for i in range(len(examples_with_steps))]


# ---------------------------------------------------------------------------
# Per-run evaluation
# ---------------------------------------------------------------------------

def _mean(vals):
    vals = [v for v in vals if v is not None]
    return sum(vals) / len(vals) if vals else None


def evaluate_run(predictions_path: str, out_detailed_path: str, out_summary_path: str,
                  base_split: str = "train", skill_repo: str = DEFAULT_SKILL_REPO,
                  skill_labels_file: str = None, fuzzy_threshold: float = 0.8):
    predictions_path = str(require_input_path(predictions_path))
    base_labels = load_skill_labels(base_split, repo_id=skill_repo, local_path=skill_labels_file)

    detailed = []
    n_with_ref = 0
    with open(predictions_path) as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            pid = row["problem_id"]
            ref = base_labels.get(pid)

            parsed = parse_model_output(row["prediction"])
            final_correct = parsed["final_answer"] is not None and answers_match(
                parsed["final_answer"], row["gold_boxed"]
            )

            n_checkable, n_correct = 0, 0
            for _, step_text in parsed["steps"]:
                nc, ncorr = arithmetic_consistency_score(step_text)
                n_checkable += nc
                n_correct += ncorr

            m = {
                "problem_id": pid, "subject": row["subject"], "level": str(row["level"]),
                "final_correct": final_correct,
                "has_think": parsed["has_think"], "has_boxed": parsed["has_boxed"],
                "n_predicted_skills": len(parsed["predicted_skills"]),
                "n_steps": len(parsed["steps"]),
                "step_checkable": n_checkable, "step_arith_correct": n_correct,
                "step_arith_consistency": (n_correct / n_checkable) if n_checkable else None,
                "has_reference": ref is not None,
            }

            if ref is not None:
                n_with_ref += 1
                ref_skills_raw = clean_skill_list(ref.get("minimum_skills", "")) or ref.get("skills_used_in_steps", "")
                ref_skills = {s.strip() for s in ref_skills_raw.split("|") if s.strip()}
                sk = skill_set_metrics(parsed["predicted_skills"], ref_skills, fuzzy_threshold)
                for k, v in sk.items():
                    m[f"skill_{k}"] = v
                m["skill_usage_validity"] = skill_usage_validity(parsed["steps"], ref_skills, fuzzy_threshold)
                m["reference_skills"] = sorted(ref_skills)
            else:
                for k in ["exact_precision", "exact_recall", "exact_f1", "fuzzy_precision",
                          "fuzzy_recall", "fuzzy_f1", "exact_match", "n_predicted", "n_reference"]:
                    m[f"skill_{k}"] = None
                m["skill_usage_validity"] = None
                m["reference_skills"] = []

            detailed.append(m)

    coverage = (n_with_ref / len(detailed) * 100) if detailed else 0.0
    print(f"[evaluate_run] {len(detailed)} examples scored, {n_with_ref} had skill ground truth "
          f"({coverage:.1f}% coverage - skill metrics only meaningful for these)")
    if n_with_ref == 0:
        print("[evaluate_run] WARNING: 0 examples had skill ground truth. Skill labels currently "
              "only cover the MATH train split - evaluate on (a subset of) train if you need the "
              "skill-prediction / skill-usage metrics; final_correct and arithmetic-consistency "
              "are still valid on any split.")

    ensure_output_path(out_detailed_path)
    with open(out_detailed_path, "w") as f:
        for m in detailed:
            f.write(json.dumps(m) + "\n")

    summary = aggregate_summary(detailed)
    summary["skills"] = skill_wise_accuracy(detailed)
    ensure_output_path(out_summary_path)
    with open(out_summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"[evaluate_run] wrote detailed -> {out_detailed_path}, summary -> {out_summary_path}")
    return summary


def skill_wise_accuracy(detailed: list) -> dict:
    """
    Per-skill final-answer accuracy: each example counts toward EVERY skill in
    its reference_skills list (a problem often requires multiple skills, so
    this is a many-to-many attribution, not a strict partition - an example
    needing 3 skills contributes to all 3 skills' n and correct counts).
    Returns {skill_name: {"n": int, "final_accuracy": float}}, sorted by n descending.
    """
    buckets = defaultdict(lambda: [0, 0])  # skill -> [n_correct, n_total]
    for r in detailed:
        for skill in r.get("reference_skills", []):
            buckets[skill][1] += 1
            if r["final_correct"]:
                buckets[skill][0] += 1

    out = {
        skill: {"n": total, "final_accuracy": round(correct / total, 4) if total else None}
        for skill, (correct, total) in buckets.items()
    }
    return dict(sorted(out.items(), key=lambda kv: kv[1]["n"], reverse=True))


def aggregate_summary(detailed: list) -> dict:
    def agg(rows):
        ref_rows = [r for r in rows if r["has_reference"]]
        arith_vals = [r["step_arith_consistency"] for r in rows if r["step_arith_consistency"] is not None]
        usage_vals = [r["skill_usage_validity"] for r in ref_rows if r["skill_usage_validity"] is not None]
        return {
            "n": len(rows),
            "final_accuracy": _mean([r["final_correct"] for r in rows]),
            "format_compliance": _mean([r["has_think"] and r["has_boxed"] for r in rows]),
            "arith_consistency": _mean(arith_vals),
            "arith_coverage": (len(arith_vals) / len(rows)) if rows else 0.0,
            "n_with_skill_ref": len(ref_rows),
            "skill_exact_f1": _mean([r["skill_exact_f1"] for r in ref_rows]) if ref_rows else None,
            "skill_fuzzy_f1": _mean([r["skill_fuzzy_f1"] for r in ref_rows]) if ref_rows else None,
            "skill_exact_match_rate": _mean([r["skill_exact_match"] for r in ref_rows]) if ref_rows else None,
            "skill_usage_validity": _mean(usage_vals) if usage_vals else None,
        }

    by_cluster = defaultdict(list)
    for r in detailed:
        by_cluster[(r["subject"], r["level"])].append(r)

    clusters = {f"{subj}|{lvl}": agg(rows) for (subj, lvl), rows in sorted(by_cluster.items())}
    return {"overall": agg(detailed), "clusters": clusters}


# ---------------------------------------------------------------------------
# Cross-run comparison / ablation studies
# ---------------------------------------------------------------------------

def retention_analysis(baseline_detailed: list, current_detailed: list):
    """
    Catastrophic-forgetting measurement by direct per-example comparison against
    a baseline, matched on problem_id.

    Aggregate accuracy alone cannot distinguish "learned 50 new, forgot 0" from
    "learned 80 new, forgot 30" - both show the same net change. This decomposes
    the change into its two opposing components, which is what a forgetting
    claim actually requires.

    Partitions the matched examples into a 2x2 contingency:

                          current WRONG   current CORRECT
        baseline WRONG      both_wrong        learned
        baseline CORRECT    forgotten         retained

    Returns:
      retained    - baseline correct AND current correct
      forgotten   - baseline correct BUT current wrong   <- catastrophic forgetting
      learned     - baseline wrong BUT current correct   <- new capability
      both_wrong  - neither correct
      retention_rate - retained / (retained + forgotten): of what the baseline
                       could already do, the fraction still done correctly.
                       1.0 means nothing was forgotten.
      forgetting_rate - forgotten / (retained + forgotten) = 1 - retention_rate
      learning_rate  - learned / (learned + both_wrong): of what the baseline
                       could NOT do, the fraction now done correctly
      net_change     - learned - forgotten (matches the aggregate accuracy delta)

    Also reports the same decomposition per (subject, level) cluster, since
    forgetting is frequently concentrated in clusters not targeted by
    augmentation rather than spread evenly.
    """
    base_by_id = {r["problem_id"]: r for r in baseline_detailed}
    matched = [(base_by_id[r["problem_id"]], r) for r in current_detailed
               if r["problem_id"] in base_by_id]
    if not matched:
        return None

    def _decompose(pairs):
        retained = forgotten = learned = both_wrong = 0
        for b, c in pairs:
            bc, cc = bool(b["final_correct"]), bool(c["final_correct"])
            if bc and cc:
                retained += 1
            elif bc and not cc:
                forgotten += 1
            elif not bc and cc:
                learned += 1
            else:
                both_wrong += 1
        base_correct = retained + forgotten
        base_wrong = learned + both_wrong
        return {
            "n": len(pairs),
            "retained": retained, "forgotten": forgotten,
            "learned": learned, "both_wrong": both_wrong,
            "retention_rate": round(retained / base_correct, 4) if base_correct else None,
            "forgetting_rate": round(forgotten / base_correct, 4) if base_correct else None,
            "learning_rate": round(learned / base_wrong, 4) if base_wrong else None,
            "net_change": learned - forgotten,
        }

    by_cluster = defaultdict(list)
    for b, c in matched:
        by_cluster[(c["subject"], c["level"])].append((b, c))

    return {
        "overall": _decompose(matched),
        "clusters": {f"{s}|{l}": _decompose(p) for (s, l), p in sorted(by_cluster.items())},
    }


def standard_error(p, n):
    if n is None or n == 0 or p is None:
        return None
    if p <= 0 or p >= 1:
        return 0.0001
    return math.sqrt(p * (1 - p) / n)


def mcnemar_test(detailed_a: list, detailed_b: list):
    """
    McNemar's test - the statistically correct test for comparing two
    classifiers (here: two model checkpoints) evaluated on the SAME set of
    items, which is exactly this pipeline's situation: every run scores the
    same test problems. Using an unpaired two-proportion test (as
    standard_error() above effectively does) throws away the pairing
    information and is both less powerful and technically the wrong test for
    this design - McNemar's test only looks at the DISCORDANT pairs (items
    where the two runs disagree), which is what actually carries information
    about a real difference between them.

    detailed_a, detailed_b: lists of per-example dicts as written by
    evaluate_run()'s --out-detailed output, each with "problem_id" and
    "final_correct".

    Returns a dict with the discordant pair counts, exact two-sided p-value,
    and a boolean significance flag at p < 0.05. Returns None if the two
    detailed files don't share any problem_ids (can't be paired at all).
    """
    b_by_id = {r["problem_id"]: r["final_correct"] for r in detailed_b}
    n01, n10, n_matched = 0, 0, 0  # n01: A wrong/B correct, n10: A correct/B wrong
    for row in detailed_a:
        pid = row["problem_id"]
        if pid not in b_by_id:
            continue
        n_matched += 1
        a_correct, b_correct = bool(row["final_correct"]), bool(b_by_id[pid])
        if a_correct and not b_correct:
            n10 += 1
        elif not a_correct and b_correct:
            n01 += 1

    if n_matched == 0:
        return None

    n_discordant = n01 + n10
    if n_discordant == 0:
        # The two runs agree on every single matched item - no evidence of any
        # difference at all, not even noise. p=1.0 is the correct result here,
        # not an error case.
        p_value = 1.0
    else:
        k = min(n01, n10)
        # Exact two-sided binomial test (McNemar's exact form) - computed
        # directly rather than via a chi-squared approximation, which is only
        # reliable for larger discordant-pair counts (typically n_discordant >= 25).
        # Exact computation is correct at any sample size and our test sets
        # (at most a few thousand items) keep this fast.
        def binom_cdf(k, n, p=0.5):
            return sum(math.comb(n, i) * (p ** i) * ((1 - p) ** (n - i)) for i in range(0, k + 1))
        p_value = min(1.0, 2 * binom_cdf(k, n_discordant))

    return {
        "n_matched": n_matched, "n01_a_wrong_b_correct": n01, "n10_a_correct_b_wrong": n10,
        "n_discordant": n_discordant, "p_value": round(p_value, 6),
        "significant_at_0.05": p_value < 0.05,
    }


OVERALL_METRIC_KEYS = [
    "final_accuracy", "format_compliance", "arith_consistency",
    "skill_exact_f1", "skill_fuzzy_f1", "skill_usage_validity",
]


def compare_runs(run_summaries: dict, out_dir: str, baseline: str = None, run_detailed: dict = None):
    """run_summaries: {run_name: summary_dict} as produced by evaluate_run.
    run_detailed: optional {run_name: [per-example dicts]} as written to
    --out-detailed - when provided, this is what enables the correct paired
    McNemar's test (see mcnemar_test() above) rather than relying solely on
    the unpaired standard-error approximation for the cluster-level table."""
    ensure_dir(out_dir)
    names = list(run_summaries.keys())
    baseline = baseline or names[0]
    if baseline not in run_summaries:
        raise SystemExit(f"[compare_runs] baseline '{baseline}' not found among runs: {names}")

    overall_table = []
    for name in names:
        row = {"run": name, "n": run_summaries[name]["overall"]["n"]}
        row.update({k: run_summaries[name]["overall"].get(k) for k in OVERALL_METRIC_KEYS})
        overall_table.append(row)
    with open(Path(out_dir) / "overall_comparison.json", "w") as f:
        json.dump(overall_table, f, indent=2)

    # --- Primary significance check: paired McNemar's test at the OVERALL
    # level (correct for this pipeline's design - every run scores the same
    # test problems) - this is what should actually be cited as "statistical
    # significance testing" in a paper/thesis, not the per-cluster SE flags
    # below, which remain only as a secondary, coarser diagnostic. ---
    significance_results = []
    if run_detailed and baseline in run_detailed:
        for name in names:
            if name == baseline or name not in run_detailed:
                continue
            result = mcnemar_test(run_detailed[baseline], run_detailed[name])
            if result:
                result["run"] = name
                result["baseline"] = baseline
                significance_results.append(result)
        if significance_results:
            with open(Path(out_dir) / "mcnemar_significance.json", "w") as f:
                json.dump(significance_results, f, indent=2)

        # Retention / catastrophic-forgetting decomposition vs the same baseline.
        retention_results = {}
        for name in names:
            if name == baseline or name not in run_detailed:
                continue
            r = retention_analysis(run_detailed[baseline], run_detailed[name])
            if r:
                retention_results[name] = r
                o = r["overall"]
                print(f"[compare_runs] retention '{name}' vs '{baseline}': "
                      f"retained={o['retained']} forgotten={o['forgotten']} "
                      f"learned={o['learned']} | retention_rate={o['retention_rate']} "
                      f"net={o['net_change']:+d}")
        if retention_results:
            with open(Path(out_dir) / "retention_analysis.json", "w") as f:
                json.dump({"baseline": baseline, "runs": retention_results}, f, indent=2)
            n_sig = sum(1 for r in significance_results if r["significant_at_0.05"])
            print(f"[compare_runs] McNemar's test (paired, exact): {n_sig}/{len(significance_results)} "
                  f"runs differ significantly from baseline '{baseline}' at p<0.05")

    cluster_keys = set()
    for name in names:
        cluster_keys |= set(run_summaries[name]["clusters"].keys())

    ablation_rows = []
    for ck in sorted(cluster_keys):
        subj, lvl = ck.split("|")
        base_cluster = run_summaries[baseline]["clusters"].get(ck)
        if base_cluster is None or base_cluster["final_accuracy"] is None:
            continue
        for name in names:
            if name == baseline:
                continue
            cur_cluster = run_summaries[name]["clusters"].get(ck)
            if cur_cluster is None or cur_cluster["final_accuracy"] is None:
                continue
            base_acc, cur_acc = base_cluster["final_accuracy"], cur_cluster["final_accuracy"]
            base_n, cur_n = base_cluster["n"], cur_cluster["n"]
            p_avg, n_avg = (base_acc + cur_acc) / 2, (base_n + cur_n) / 2
            se = standard_error(p_avg, n_avg)
            diff = cur_acc - base_acc
            se_units = (abs(diff) / se) if se else None
            ablation_rows.append({
                "subject": subj, "level": lvl, "run": name, "baseline": baseline,
                "baseline_acc": round(base_acc, 4), "run_acc": round(cur_acc, 4),
                "diff_pp": round(diff * 100, 2),
                "n_baseline": base_n, "n_run": cur_n,
                "se_units": round(se_units, 2) if se_units is not None else None,
                "likely_real_difference": bool(se_units is not None and se_units > 2),
            })
    with open(Path(out_dir) / "ablation_deltas.json", "w") as f:
        json.dump(ablation_rows, f, indent=2)

    n_flagged = sum(1 for r in ablation_rows if r["likely_real_difference"])
    print(f"[compare_runs] (secondary, coarser diagnostic) {n_flagged}/{len(ablation_rows)} "
          f"cluster-level diffs vs baseline '{baseline}' exceed 2 standard errors - this unpaired "
          f"approximation is only a quick per-cluster signal; cite mcnemar_significance.json's "
          f"exact paired test above as the actual significance result.")

    _plot_accuracy_by_level(run_summaries, out_dir)
    _plot_overall_metrics(run_summaries, out_dir)

    print(f"[compare_runs] wrote overall_comparison.json, ablation_deltas.json, and charts -> {out_dir}")
    return overall_table, ablation_rows, significance_results


def _plot_accuracy_by_level(run_summaries: dict, out_dir: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    subjects = sorted({ck.split("|")[0] for s in run_summaries.values() for ck in s["clusters"]})
    for subj in subjects:
        fig, ax = plt.subplots(figsize=(7, 4))
        plotted_any = False
        for name, summary in run_summaries.items():
            xs, ys = [], []
            for lvl_num in range(1, 6):
                for lvl_key in (f"Level {lvl_num}", str(lvl_num)):
                    c = summary["clusters"].get(f"{subj}|{lvl_key}")
                    if c is not None and c["final_accuracy"] is not None:
                        xs.append(f"L{lvl_num}")
                        ys.append(c["final_accuracy"])
                        break
            if xs:
                ax.plot(xs, ys, marker="o", label=name)
                plotted_any = True
        if not plotted_any:
            plt.close(fig)
            continue
        ax.set_title(f"{subj}: accuracy by level")
        ax.set_ylabel("accuracy")
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(Path(out_dir) / f"accuracy_by_level_{subj}.png", dpi=150)
        plt.close(fig)


def _plot_overall_metrics(run_summaries: dict, out_dir: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    names = list(run_summaries.keys())
    x = np.arange(len(OVERALL_METRIC_KEYS))
    width = 0.8 / max(len(names), 1)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for i, name in enumerate(names):
        vals = [run_summaries[name]["overall"].get(k) or 0 for k in OVERALL_METRIC_KEYS]
        ax.bar(x + i * width, vals, width=width, label=name)
    ax.set_xticks(x + width * (len(names) - 1) / 2)
    ax.set_xticklabels(OVERALL_METRIC_KEYS, rotation=20, ha="right", fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8)
    ax.set_title("Overall metric comparison across runs")
    fig.tight_layout()
    fig.savefig(Path(out_dir) / "overall_metrics_comparison.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_run_arg(run_args):
    runs = {}
    for item in run_args or []:
        if "=" not in item:
            raise SystemExit(f"[evaluator] --run expects name=path, got: {item}")
        name, path = item.split("=", 1)
        runs[name] = path
    return runs


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--score", action="store_true")
    ap.add_argument("--compare", action="store_true")

    # --score args
    ap.add_argument("--predictions", type=str, default=None)
    ap.add_argument("--split", type=str, default="train",
                     help="split to load skill labels from - use 'train' if your "
                          "predictions are also on the train split (required for skill "
                          "metrics, since labels only cover train currently)")
    ap.add_argument("--skill-repo", type=str, default=DEFAULT_SKILL_REPO)
    ap.add_argument("--skill-labels-file", type=str, default=None)
    ap.add_argument("--out-detailed", type=str, default="outputs/eval_detailed.jsonl")
    ap.add_argument("--out-summary", type=str, default="outputs/eval_summary.json")
    ap.add_argument("--fuzzy-threshold", type=float, default=0.8)

    # --compare args
    ap.add_argument("--run", action="append", default=[],
                     help="repeatable: --run name=path/to/summary.json")
    ap.add_argument("--run-detailed", action="append", default=[],
                     help="repeatable, optional: --run-detailed name=path/to/detailed.jsonl - "
                          "enables the exact paired McNemar's test (the correct significance "
                          "test for this pipeline's design) rather than only the coarser "
                          "unpaired standard-error approximation")
    ap.add_argument("--baseline", type=str, default=None)
    ap.add_argument("--out-dir", type=str, default="outputs/comparison")

    add_destination_args(ap, default_repo_type="dataset")
    args = ap.parse_args()

    if args.score:
        if not args.predictions:
            raise SystemExit("[evaluator] --score requires --predictions")
        evaluate_run(
            args.predictions, args.out_detailed, args.out_summary,
            base_split=args.split, skill_repo=args.skill_repo,
            skill_labels_file=args.skill_labels_file, fuzzy_threshold=args.fuzzy_threshold,
        )
        dispatch_destination(args.out_summary, args)

    elif args.compare:
        run_paths = _parse_run_arg(args.run)
        if not run_paths:
            raise SystemExit("[evaluator] --compare requires at least one --run name=path")
        run_summaries = {}
        for name, path in run_paths.items():
            with open(str(require_input_path(path))) as f:
                run_summaries[name] = json.load(f)

        run_detailed = None
        if args.run_detailed:
            detailed_paths = _parse_run_arg(args.run_detailed)
            run_detailed = {}
            for name, path in detailed_paths.items():
                rows = []
                with open(str(require_input_path(path))) as f:
                    for line in f:
                        if line.strip():
                            rows.append(json.loads(line))
                run_detailed[name] = rows

        compare_runs(run_summaries, args.out_dir, baseline=args.baseline, run_detailed=run_detailed)
        dispatch_destination(args.out_dir, args)

    else:
        print("Pass --score (single-run evaluation) or --compare (multi-run ablation "
              "comparison with charts). See module docstring for examples.")
