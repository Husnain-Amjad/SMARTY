"""
Generates data-driven figures directly from experiment_ledger.py's ledger:
retention/forgetting curves (accuracy vs. checkpoint, with vs. without replay)
and skill-wise performance bar charts. These are the DATA figures from the
paper outline (Figures 5 and 6) - the CONCEPTUAL/architecture diagrams
(framework overview, skill taxonomy, replay mechanism, training pipeline;
Figures 1-4) are illustrative, not data-driven, and are handled separately by
generate_diagrams.py.

Usage:
  # retention curve: accuracy vs checkpoint step, one line per named run
  python generate_figures.py --figure retention_curve \
      --ledger outputs/experiment_ledger.jsonl \
      --run_ids qwen7b_e05 qwen7b_e1 qwen7b_e15 qwen7b_e2 \
      --x_values 0.5 1.0 1.5 2.0 --x_label "Epoch" \
      --out outputs/figures/retention_curve.png

  # skill-wise performance bar chart for one run
  python generate_figures.py --figure skill_wise \
      --ledger outputs/experiment_ledger.jsonl --run_id qwen7b_skill_lora_e05 \
      --top_n 15 --out outputs/figures/skill_wise.png
"""

import argparse

from experiment_ledger import load_ledger, find_record
from storage_utils import ensure_output_path


def plot_retention_curve(ledger_path, run_ids, x_values, x_label, out_path,
                          baseline_run_id=None, metric="final_accuracy"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if len(run_ids) != len(x_values):
        raise SystemExit(f"[generate_figures] --run_ids ({len(run_ids)}) and --x_values "
                          f"({len(x_values)}) must be the same length - one x-position per run.")

    ys = []
    for rid in run_ids:
        rec = find_record(ledger_path, rid)
        if rec is None:
            raise SystemExit(f"[generate_figures] run_id '{rid}' not found in {ledger_path}")
        ys.append(rec["overall"].get(metric))

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(x_values, ys, marker="o", label="with this run's config", color="#2F5496")

    if baseline_run_id:
        base_rec = find_record(ledger_path, baseline_run_id)
        if base_rec is not None:
            base_val = base_rec["overall"].get(metric)
            if base_val is not None:
                ax.axhline(base_val, linestyle="--", color="gray", label=f"baseline ({baseline_run_id})")

    ax.set_xlabel(x_label)
    ax.set_ylabel(metric.replace("_", " "))
    ax.set_title(f"Retention curve: {metric.replace('_', ' ')} vs. {x_label.lower()}")
    ax.legend(fontsize=8)
    ax.set_ylim(0, 1.05)
    fig.tight_layout()
    ensure_output_path(out_path)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[generate_figures] wrote {out_path}")


def plot_retention_comparison(ledger_path, run_id_groups, x_values, x_label, out_path, metric="final_accuracy"):
    """
    run_id_groups: {label: [run_id_at_x0, run_id_at_x1, ...]} - e.g.
      {"with replay": [...4 checkpoints...], "without replay": [...4 checkpoints...]}
    Plots one line per group, for direct with/without (or strategy A/B/C/D) comparison.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for label, run_ids in run_id_groups.items():
        if len(run_ids) != len(x_values):
            raise SystemExit(f"[generate_figures] group '{label}' has {len(run_ids)} run_ids "
                              f"but {len(x_values)} x_values were given - must match.")
        ys = []
        for rid in run_ids:
            rec = find_record(ledger_path, rid)
            if rec is None:
                raise SystemExit(f"[generate_figures] run_id '{rid}' (group '{label}') not found")
            ys.append(rec["overall"].get(metric))
        ax.plot(x_values, ys, marker="o", label=label)

    ax.set_xlabel(x_label)
    ax.set_ylabel(metric.replace("_", " "))
    ax.set_title(f"Retention comparison: {metric.replace('_', ' ')} vs. {x_label.lower()}")
    ax.legend(fontsize=8)
    ax.set_ylim(0, 1.05)
    fig.tight_layout()
    ensure_output_path(out_path)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[generate_figures] wrote {out_path}")


def plot_skill_wise(ledger_path, run_id, out_path, top_n=15):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rec = find_record(ledger_path, run_id)
    if rec is None:
        raise SystemExit(f"[generate_figures] run_id '{run_id}' not found in {ledger_path}")
    skills = rec.get("skills") or {}
    if not skills:
        raise SystemExit(f"[generate_figures] run '{run_id}' has no skill-wise data logged - "
                          f"see evaluator.py's skill_wise_accuracy().")

    items = list(skills.items())[:top_n]
    names = [k for k, _ in items]
    accs = [v.get("final_accuracy") or 0 for _, v in items]
    ns = [v.get("n") or 0 for _, v in items]

    fig, ax = plt.subplots(figsize=(9, max(4, 0.35 * len(names))))
    y_pos = range(len(names))
    bars = ax.barh(y_pos, accs, color="#2F5496")
    ax.set_yticks(y_pos)
    ax.set_yticklabels([f"{n} (n={c})" for n, c in zip(names, ns)], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("accuracy")
    ax.set_xlim(0, 1.05)
    ax.set_title(f"Skill-wise accuracy (top {top_n} by frequency) - {run_id}")
    fig.tight_layout()
    ensure_output_path(out_path)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[generate_figures] wrote {out_path}")


def plot_all_methods_separately(ledger_path, model_slug, out_dir, metric="final_accuracy"):
    """
    Produces ONE SEPARATE retention-curve figure per training method, rather
    than combining everything into one chart - covers every method
    run_full_experiment.sh actually produces for one model: the 4 SFT-stage
    variants (fine-tuning only, +semantic replay, +numeric replay, +both) and
    the 4 GRPO variants (GRPO on top of each of those). Each figure plots
    only that method's own per-checkpoint entries (run_ids matching
    '{model_slug}__{method}_ckpts__...'), auto-discovered from the ledger -
    you don't need to list checkpoint run_ids by hand.

    Methods with no logged checkpoints yet (e.g. GRPO stages that haven't run)
    are skipped with a printed note, not an error - this is expected mid-run.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import re

    METHOD_LABELS = {
        "stage1_sft": "Fine-Tuning Only (SFT)",
        "sem_replay": "SFT + Semantic Augmentation Replay",
        "num_replay": "SFT + Numeric Augmentation Replay",
        "both_replay": "SFT + Semantic & Numeric Augmentation Replay",
        "grpo_stage1": "GRPO on Fine-Tuning Only",
        "grpo_sem_replay": "GRPO on Semantic Replay",
        "grpo_num_replay": "GRPO on Numeric Replay",
        "grpo_both_replay": "GRPO on Semantic & Numeric Replay",
    }

    records = load_ledger(ledger_path)
    written = []

    for method, label in METHOD_LABELS.items():
        prefix = f"{model_slug}__{method}_ckpts__"
        matches = [r for r in records if r["run_id"].startswith(prefix)]
        if not matches:
            print(f"[generate_figures] no checkpoints logged yet for '{method}' - skipping "
                  f"(expected if this stage hasn't run, not an error)")
            continue

        # Sort by the numeric step/epoch embedded in the run_id suffix
        def _sort_key(r):
            m = re.search(r"(step|epoch)_([0-9.]+)$", r["run_id"])
            return float(m.group(2)) if m else 0.0
        matches.sort(key=_sort_key)

        x_values = [_sort_key(r) for r in matches]
        y_values = [r["overall"].get(metric) for r in matches]

        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.plot(x_values, y_values, marker="o", color="#2F5496")
        ax.set_xlabel("Checkpoint (step/epoch)")
        ax.set_ylabel(metric.replace("_", " "))
        ax.set_title(f"{label}\n{metric.replace('_', ' ')} across checkpoints")
        ax.set_ylim(0, 1.05)
        fig.tight_layout()

        out_path = f"{out_dir.rstrip('/')}/retention_{model_slug}_{method}.png"
        ensure_output_path(out_path)
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"[generate_figures] wrote {out_path} ({len(matches)} checkpoints)")
        written.append(out_path)

    return written


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--figure", required=True,
                     choices=["retention_curve", "skill_wise", "all_methods"])
    ap.add_argument("--ledger", default="outputs/experiment_ledger.jsonl")
    ap.add_argument("--run_ids", nargs="*", default=None)
    ap.add_argument("--run_id", default=None)
    ap.add_argument("--x_values", nargs="*", type=float, default=None)
    ap.add_argument("--x_label", default="Epoch")
    ap.add_argument("--baseline_run_id", default=None)
    ap.add_argument("--metric", default="final_accuracy")
    ap.add_argument("--top_n", type=int, default=15)
    ap.add_argument("--out", default=None)
    ap.add_argument("--model_slug", default=None,
                     help="required for --figure all_methods - the model_slug prefix used in "
                          "run_full_experiment.sh's logged run_ids, e.g. Qwen_Qwen2.5-Math-7B-Instruct")
    ap.add_argument("--out_dir", default=None,
                     help="required for --figure all_methods - directory each method's separate "
                          "figure gets written into")
    args = ap.parse_args()

    if args.figure == "retention_curve":
        if not (args.run_ids and args.x_values and args.out):
            raise SystemExit("--figure retention_curve requires --run_ids, --x_values, and --out")
        plot_retention_curve(args.ledger, args.run_ids, args.x_values, args.x_label, args.out,
                              baseline_run_id=args.baseline_run_id, metric=args.metric)
    elif args.figure == "skill_wise":
        if not (args.run_id and args.out):
            raise SystemExit("--figure skill_wise requires --run_id and --out")
        plot_skill_wise(args.ledger, args.run_id, args.out, top_n=args.top_n)
    elif args.figure == "all_methods":
        if not (args.model_slug and args.out_dir):
            raise SystemExit("--figure all_methods requires --model_slug and --out_dir")
        plot_all_methods_separately(args.ledger, args.model_slug, args.out_dir, metric=args.metric)
