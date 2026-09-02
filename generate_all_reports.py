"""
Generates EVERYTHING in one command: architecture diagrams (always, no ledger
needed) plus every standard table and figure the experiment ledger currently
supports (if a ledger exists). This exists specifically to fix an
inconsistency in the earlier design - treating diagrams (generate_diagrams.py)
as a separate, standalone thing from tables (generate_tables.py) and figures
(generate_figures.py) implied they were less important or less connected,
when all three are equally important experiment-reporting outputs and belong
in one place, run together.

Usage:
  python generate_all_reports.py --ledger outputs/experiment_ledger.jsonl --out_dir outputs/report
  python generate_all_reports.py --out_dir outputs/report   # diagrams only, no ledger yet
"""

import argparse
import os

from experiment_ledger import load_ledger
import generate_tables as gt
import generate_figures as gf
import generate_diagrams as gd


def generate_all(ledger_path: str, out_dir: str):
    diagrams_dir = os.path.join(out_dir, "diagrams")
    tables_dir = os.path.join(out_dir, "tables")
    figures_dir = os.path.join(out_dir, "figures")
    for d in (diagrams_dir, tables_dir, figures_dir):
        os.makedirs(d, exist_ok=True)

    print("=" * 60)
    print("DIAGRAMS (architecture/conceptual - no ledger needed)")
    print("=" * 60)
    for name in gd.DIAGRAMS:
        gd.render(name, os.path.join(diagrams_dir, name))

    if not ledger_path or not os.path.exists(ledger_path):
        print(f"\n[generate_all_reports] no ledger found at '{ledger_path}' - skipping tables "
              f"and figures, which are data-driven and need at least one logged run "
              f"(experiment_ledger.py --log). Diagrams above don't depend on this.")
        return

    records = load_ledger(ledger_path)
    if not records:
        print(f"\n[generate_all_reports] ledger at '{ledger_path}' is empty - skipping "
              f"tables/figures for the same reason.")
        return

    # Unique run_ids, keeping the LATEST position of each (matches how the
    # ledger's own dedup-to-latest behavior works elsewhere in this pipeline).
    unique_run_ids = list(dict.fromkeys(r["run_id"] for r in records))
    latest_run_id = unique_run_ids[-1]

    print()
    print("=" * 60)
    print("TABLES")
    print("=" * 60)
    gt.table_main_results(ledger_path, os.path.join(tables_dir, "main_results"))

    if len(unique_run_ids) >= 2:
        gt.table_compare(ledger_path, unique_run_ids, os.path.join(tables_dir, "compare"))
    else:
        print("[generate_all_reports] only one unique run_id logged - skipping the "
              "cross-run compare table (needs at least 2).")

    try:
        gt.table_subject_level(ledger_path, latest_run_id,
                                os.path.join(tables_dir, f"subject_level_{latest_run_id}"))
    except SystemExit as e:
        print(f"[generate_all_reports] subject_level table skipped: {e}")

    try:
        gt.table_skill_wise(ledger_path, latest_run_id,
                             os.path.join(tables_dir, f"skill_wise_{latest_run_id}"))
    except SystemExit as e:
        print(f"[generate_all_reports] skill_wise table skipped: {e}")

    print()
    print("=" * 60)
    print("FIGURES")
    print("=" * 60)
    if len(unique_run_ids) >= 2:
        x_values = list(range(len(unique_run_ids)))
        gf.plot_retention_curve(
            ledger_path, unique_run_ids, x_values, "Stage",
            os.path.join(figures_dir, "retention_curve.png"),
            baseline_run_id=unique_run_ids[0],
        )
    else:
        print("[generate_all_reports] only one unique run_id logged - skipping the "
              "retention curve (needs at least 2 points to plot a curve).")

    try:
        gf.plot_skill_wise(ledger_path, latest_run_id,
                            os.path.join(figures_dir, f"skill_wise_{latest_run_id}.png"))
    except SystemExit as e:
        print(f"[generate_all_reports] skill_wise figure skipped: {e}")

    # Separate figure per training method (fine-tuning-only, +semantic replay,
    # +numeric replay, +both, and GRPO on each) rather than one combined
    # chart - auto-discovers every distinct model_slug present in the ledger
    # (run_full_experiment.sh's run_id convention is "{model_slug}__{stage}...",
    # so this works correctly even when the ledger holds multiple models'
    # results, as it does after run_all_models.sh).
    model_slugs = sorted(set(rid.split("__")[0] for rid in unique_run_ids if "__" in rid))
    for slug in model_slugs:
        gf.plot_all_methods_separately(ledger_path, slug, figures_dir)

    print()
    print(f"[generate_all_reports] done -> {out_dir}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ledger", default="outputs/experiment_ledger.jsonl")
    ap.add_argument("--out_dir", default="outputs/report")
    args = ap.parse_args()
    generate_all(args.ledger, args.out_dir)
