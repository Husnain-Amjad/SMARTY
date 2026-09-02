"""
Replaces "hit and trial" hyperparameter selection with a documented, small
grid search on a held-out VALIDATION subset - carved from the TRAIN split,
never touching the test split, so the final reported test-set numbers are
never used, even indirectly, to select hyperparameters. This is standard,
defensible ML methodology and the right way to answer "how were
hyperparameters chosen?" in a paper/thesis - not "by hand," but "by a
documented search on held-out validation data."

Design (deliberately a SMALL, bounded search, not an exhaustive one - this
is meant to be affordable on a resource-constrained thesis budget, not a
full hyperparameter sweep):
  1. Split the built SFT dataset into an HP-search TRAIN subset and a
     held-out VALIDATION subset (fixed seed, default 80/20) - the validation
     subset is used ONLY to score candidate configurations, never for
     final training, and is never the same data as the test split used for
     final reported results.
  2. For each candidate configuration in a small, EXPLICITLY-DEFINED grid
     (you specify it directly - see --grid; typically 4-8 total
     configurations from varying 1-2 dimensions, not a large Cartesian
     product), run a short training run and evaluate final-answer accuracy
     on the validation subset via a lightweight, self-contained eval loop
     (reuses run_augmentation.py's backend + reward_fn.answers_match,
     doesn't require a separate run_eval.py invocation per config).
  3. Report every config's validation accuracy, ranked, and select the best.
     THIS selected configuration - not a hand-picked one - is what should be
     used for the full/final experiments and cited in the paper's
     Implementation Details section, along with a one-line note: "learning
     rate and LoRA rank were selected via a grid search over {grid} on a
     held-out validation subset (n={val_size}, distinct from both the
     training data used for final models and the test set used for
     reporting), selecting by validation accuracy."

Usage:
  python hyperparameter_search.py \
      --model Qwen/Qwen2.5-Math-1.5B-Instruct \
      --sft-data outputs/sft_data_full.jsonl \
      --grid lr=1e-5,2e-5 lora_r=8,16,32 \
      --train-size 200 --val-size 100 --epochs 1 \
      --out outputs/hp_search_results.json
"""

import argparse
import itertools
import json
import os
import random
import subprocess
import sys
import tempfile

from reward_fn import answers_match, extract_boxed


def parse_grid(grid_args: list) -> dict:
    """Parses ['lr=1e-5,2e-5', 'lora_r=8,16,32'] into
    {'lr': [1e-5, 2e-5], 'lora_r': [8, 16, 32]} - values are parsed as float
    if they contain '.'/'e', else int, else left as a string."""
    def _parse_value(v):
        try:
            if "." in v or "e" in v.lower():
                return float(v)
            return int(v)
        except ValueError:
            return v

    grid = {}
    for arg in grid_args:
        if "=" not in arg:
            raise SystemExit(f"[hyperparameter_search] --grid entries must be name=v1,v2,... "
                              f"(got '{arg}')")
        name, values_str = arg.split("=", 1)
        grid[name] = [_parse_value(v) for v in values_str.split(",")]
    return grid


def expand_grid(grid: dict) -> list:
    """Cartesian product of the grid, as a list of {name: value} dicts.
    Deliberately NOT recommended for more than 2-3 dimensions at once - the
    whole point of this script is a small, affordable search, not an
    exhaustive sweep. A 3-dimension grid of size (2,3,2) is already 12 full
    training+eval runs; keep dimensions and per-dimension value counts small."""
    if not grid:
        return [{}]
    names = list(grid.keys())
    combos = list(itertools.product(*[grid[n] for n in names]))
    return [dict(zip(names, combo)) for combo in combos]


def split_train_val(sft_data_path: str, train_size: int, val_size: int, seed: int = 42):
    """Deterministic, seeded split of the SFT dataset into an HP-search
    training subset and a held-out validation subset - disjoint from each
    other, and this whole split is itself disjoint from the test split used
    for final reporting (this function only ever touches train-derived data)."""
    with open(sft_data_path) as f:
        rows = [json.loads(line) for line in f if line.strip()]

    rng = random.Random(seed)
    indices = list(range(len(rows)))
    rng.shuffle(indices)

    if train_size + val_size > len(rows):
        raise SystemExit(f"[hyperparameter_search] requested train_size={train_size} + "
                          f"val_size={val_size} = {train_size + val_size}, but sft_data only "
                          f"has {len(rows)} rows. Reduce the sizes or build a larger SFT dataset.")

    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size + val_size]
    train_rows = [rows[i] for i in train_indices]
    val_rows = [rows[i] for i in val_indices]
    return train_rows, val_rows


def evaluate_checkpoint_on_validation(model_path: str, val_rows: list, use_vllm: bool = True,
                                       seed: int = 42, batch_size: int = 32) -> float:
    """Lightweight, self-contained validation accuracy check - reuses
    run_augmentation.py's backend rather than shelling out to run_eval.py,
    since this needs to run once per candidate configuration and only needs
    final-answer accuracy, not the full evaluator.py metric suite."""
    from run_augmentation import build_backend
    from templates import render_prompt_only

    backend = build_backend(model_path, use_vllm, seed=seed)
    prompts = [render_prompt_only(backend.tokenizer, row["problem"]) for row in val_rows]

    n_correct = 0
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i:i + batch_size]
        results = backend.generate(batch, n=1, temperature=0.0, max_tokens=1024)
        for row, result in zip(val_rows[i:i + batch_size], results):
            prediction = result[0]
            pred_boxed = extract_boxed(prediction) or ""
            gold = row.get("gold_boxed", "")
            if gold and answers_match(pred_boxed, gold):
                n_correct += 1
    return n_correct / len(val_rows) if val_rows else 0.0


def run_search(model: str, sft_data_path: str, grid: dict, train_size: int, val_size: int,
                epochs: float, seed: int, out_path: str, use_vllm: bool = True):
    configs = expand_grid(grid)
    print(f"[hyperparameter_search] {len(configs)} configuration(s) to evaluate: {configs}")
    if len(configs) > 12:
        print(f"[hyperparameter_search] WARNING: {len(configs)} configurations is a large search "
              f"for this script's intended scope (a small, affordable, documented search, not an "
              f"exhaustive sweep) - consider narrowing --grid.")

    train_rows, val_rows = split_train_val(sft_data_path, train_size, val_size, seed)
    print(f"[hyperparameter_search] split: {len(train_rows)} HP-search train rows, "
          f"{len(val_rows)} held-out validation rows (both from train data only - "
          f"the test split is never touched during this search)")

    results = []
    with tempfile.TemporaryDirectory() as tmp_dir:
        train_subset_path = os.path.join(tmp_dir, "hp_search_train.jsonl")
        with open(train_subset_path, "w") as f:
            for row in train_rows:
                f.write(json.dumps(row) + "\n")

        for i, config in enumerate(configs):
            print(f"\n[hyperparameter_search] --- config {i+1}/{len(configs)}: {config} ---")
            output_dir = os.path.join(tmp_dir, f"config_{i}")

            cmd = [sys.executable, "sft_train.py", "--model", model,
                   "--data", train_subset_path, "--mode", "lora",
                   "--output_dir", output_dir, "--epochs", str(epochs),
                   "--seed", str(seed)]
            for name, value in config.items():
                cmd.extend([f"--{name}", str(value)])

            try:
                subprocess.run(cmd, check=True)
            except subprocess.CalledProcessError as e:
                print(f"[hyperparameter_search] config {config} FAILED to train "
                      f"(exit code {e.returncode}) - skipping, not counted in results")
                results.append({"config": config, "val_accuracy": None, "status": "train_failed"})
                continue

            merged_dir = output_dir.rstrip("/") + "_merged"
            model_dir = merged_dir if os.path.isdir(merged_dir) else output_dir
            val_acc = evaluate_checkpoint_on_validation(model_dir, val_rows, use_vllm=use_vllm, seed=seed)
            print(f"[hyperparameter_search] config {config} -> validation accuracy: {val_acc:.4f}")
            results.append({"config": config, "val_accuracy": val_acc, "status": "ok"})

    results_sorted = sorted([r for r in results if r["val_accuracy"] is not None],
                             key=lambda r: r["val_accuracy"], reverse=True)
    failed = [r for r in results if r["val_accuracy"] is None]

    with open(out_path, "w") as f:
        json.dump({"model": model, "val_size": len(val_rows), "train_size": len(train_rows),
                   "seed": seed, "results_ranked": results_sorted, "failed": failed}, f, indent=2)

    print(f"\n[hyperparameter_search] wrote ranked results -> {out_path}")
    if results_sorted:
        best = results_sorted[0]
        print(f"[hyperparameter_search] BEST configuration: {best['config']} "
              f"(validation accuracy: {best['val_accuracy']:.4f})")
        print(f"[hyperparameter_search] Suggested methodology sentence for the paper: "
              f"\"Hyperparameters were selected via a grid search over {list(grid.keys())} "
              f"on a held-out validation subset (n={len(val_rows)}, disjoint from both the "
              f"final training data and the test set used for reporting), selecting the "
              f"configuration with highest validation accuracy: {best['config']}.\"")
    return results_sorted


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--sft-data", required=True, dest="sft_data_path")
    ap.add_argument("--grid", nargs="+", required=True,
                     help="one or more name=v1,v2,... entries, e.g. --grid lr=1e-5,2e-5 lora_r=8,16,32")
    ap.add_argument("--train-size", type=int, default=200,
                     help="rows used for each candidate's short training run")
    ap.add_argument("--val-size", type=int, default=100,
                     help="held-out rows used to score each candidate - never used for training")
    ap.add_argument("--epochs", type=float, default=1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--use_vllm", action="store_true", default=True)
    ap.add_argument("--out", default="outputs/hp_search_results.json")
    args = ap.parse_args()

    grid = parse_grid(args.grid)
    run_search(args.model, args.sft_data_path, grid, args.train_size, args.val_size,
               args.epochs, args.seed, args.out, use_vllm=args.use_vllm)
