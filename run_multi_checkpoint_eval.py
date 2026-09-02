"""
Evaluates every checkpoint produced by ONE continuous sft_train.py run (e.g.
--save_every_epochs 0.5 over --epochs 2 gives checkpoint-N at 0.5, 1.0, 1.5, 2.0
epochs) and produces a single accuracy-vs-epoch comparison via evaluator.py -
this is what replaces launching training separately for each eval point.

For LoRA runs, each checkpoint is merged into a throwaway temp directory before
evaluation (never touching the original adapter checkpoint), since vLLM/HF
generation needs a standalone model, not an adapter-only directory.

Usage:
  python run_multi_checkpoint_eval.py --checkpoints_dir ckpts/qwen7b_run \
      --base_model Qwen/Qwen2.5-Math-7B --mode lora \
      --split test --out-dir outputs/checkpoint_eval

  # if some checkpoints were only ever pushed to HF and the local disk was wiped:
  python run_multi_checkpoint_eval.py --checkpoints_dir ckpts/qwen7b_run \
      --hf_repo_id me/my-model --base_model Qwen/Qwen2.5-Math-7B --mode lora \
      --split test --out-dir outputs/checkpoint_eval
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile

from storage_utils import ensure_dir, require_input_path


def find_all_checkpoints(checkpoints_dir):
    """Returns [(step, path), ...] sorted by step, for every checkpoint-N under checkpoints_dir."""
    if not os.path.isdir(checkpoints_dir):
        return []
    out = []
    for d in os.listdir(checkpoints_dir):
        m = re.match(r"^checkpoint-(\d+)$", d)
        if m:
            out.append((int(m.group(1)), os.path.join(checkpoints_dir, d)))
    return sorted(out)


def apply_checkpoint_stride(checkpoints: list, stride: int = 1) -> list:
    """
    Subsamples the checkpoint list to reduce per-checkpoint evaluation cost -
    each checkpoint pays a full subprocess + vLLM engine startup (model load,
    torch.compile warmup, CUDA graph capture), commonly 40-90+ seconds BEFORE
    any actual generation happens, so evaluating every single checkpoint on a
    long run with many save points can dominate total runtime. stride=1
    (default) evaluates every checkpoint, unchanged. stride=2 evaluates every
    other one, stride=3 every third, etc. - the first and last checkpoints
    are ALWAYS kept regardless of stride, since those are the two points a
    retention curve needs most (start and end of training).
    """
    if stride <= 1 or len(checkpoints) <= 2:
        return checkpoints
    strided = checkpoints[::stride]
    if checkpoints[-1] not in strided:
        strided.append(checkpoints[-1])
    return strided


def download_checkpoint_from_hf(hf_repo_id, checkpoint_label, dest_dir):
    """Pulls one checkpoint-N subfolder back down from HF (for when local disk was wiped)."""
    from huggingface_hub import snapshot_download
    path = snapshot_download(repo_id=hf_repo_id, allow_patterns=[f"{checkpoint_label}/*"])
    src = os.path.join(path, checkpoint_label)
    shutil.copytree(src, dest_dir, dirs_exist_ok=True)
    return dest_dir


def merge_checkpoint(base_model, adapter_path, out_dir):
    """Calls merge_lora.py as a subprocess into a throwaway directory - never touches
    the original adapter checkpoint, so this is safe to run on every checkpoint in a loop."""
    result = subprocess.run(
        [sys.executable, "merge_lora.py", "--base_model", base_model,
         "--adapter_path", adapter_path, "--out", out_dir],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"merge_lora.py failed for {adapter_path}:\n{result.stderr}")
    return out_dir


def _echo_backend_lines(captured_stdout: str, label: str):
    """
    run_eval.py's own '[backend] ...' / '[hardware_utils] ...' lines report
    which inference backend (vLLM vs HF-generate) and which attention
    implementation actually got used. Because run_eval.py is invoked here as
    a subprocess with capture_output=True, those lines are swallowed and never
    reach the user's terminal - which makes it genuinely impossible to tell
    whether vLLM engaged or silently fell back to the much slower HF-generate
    path. This re-prints just those diagnostic lines so that's visible.
    """
    for line in (captured_stdout or "").splitlines():
        if line.startswith("[backend]") or line.startswith("[hardware_utils]"):
            print(f"  [{label}] {line}")


def run_eval_and_score(model_path, split, run_name, out_dir, seed, limit=None):
    """Calls run_eval.py then evaluator.py --score as subprocesses, returns the summary path."""
    predictions_path = os.path.join(out_dir, f"predictions_{run_name}.jsonl")
    eval_cmd = [sys.executable, "run_eval.py", "--model", model_path, "--split", split,
                "--out", predictions_path, "--seed", str(seed)]
    if limit:
        eval_cmd += ["--limit", str(limit)]
    result = subprocess.run(eval_cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"run_eval.py failed for {run_name}:\n{result.stderr}")
    _echo_backend_lines(result.stdout, run_name)

    detailed_path = os.path.join(out_dir, f"eval_{run_name}_detailed.jsonl")
    summary_path = os.path.join(out_dir, f"eval_{run_name}_summary.json")
    score_cmd = [sys.executable, "evaluator.py", "--score", "--predictions", predictions_path,
                 "--split", split, "--out-detailed", detailed_path, "--out-summary", summary_path]
    result = subprocess.run(score_cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"evaluator.py --score failed for {run_name}:\n{result.stderr}")
    return summary_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoints_dir", required=True,
                     help="the --output_dir a single sft_train.py run wrote checkpoint-N/ into")
    ap.add_argument("--checkpoint_stride", type=int, default=1,
                     help="evaluate every Nth checkpoint instead of all of them (default 1 = "
                          "all). Each checkpoint pays a full subprocess + vLLM engine startup "
                          "(commonly 40-90+ seconds before generation even begins), so this "
                          "trades retention-curve granularity for speed on runs with many save "
                          "points. The first and last checkpoints are always kept regardless.")
    ap.add_argument("--base_model", required=True, help="needed to merge LoRA checkpoints")
    ap.add_argument("--mode", choices=["full", "lora"], required=True)
    ap.add_argument("--hf_repo_id", default=None,
                     help="if a checkpoint isn't found locally, try downloading it from "
                          "this repo (see PushCheckpointCallback in sft_train.py)")
    ap.add_argument("--split", default="test")
    ap.add_argument("--out-dir", default="outputs/checkpoint_eval")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit", type=int, default=None, help="cap problems per checkpoint for a quick pass")
    ap.add_argument("--steps_per_epoch", type=float, default=None,
                     help="if known, converts step numbers to epoch fractions in the "
                          "final report/chart labels (purely cosmetic - the underlying "
                          "comparison works either way, labeled by step if omitted)")
    ap.add_argument("--ledger", default=None,
                     help="if given, also logs EVERY checkpoint's result to this experiment "
                          "ledger file (not just the final checkpoint), one run_id per "
                          "checkpoint - this is what makes an accuracy-vs-epoch retention "
                          "curve possible via generate_figures.py afterward.")
    ap.add_argument("--run_id_prefix", default=None,
                     help="required if --ledger is set - each checkpoint is logged as "
                          "'<run_id_prefix>__<label>' (e.g. 'qwen7b_stage1__epoch_0.50')")
    ap.add_argument("--baseline_run_id", default=None,
                     help="optional - if set, every logged checkpoint's ledger entry also "
                          "records its accuracy increment over this baseline run_id")
    args = ap.parse_args()

    if args.ledger and not args.run_id_prefix:
        raise SystemExit("[run_multi_checkpoint_eval] --ledger requires --run_id_prefix "
                          "(checked before any evaluation work starts, not after)")

    ensure_dir(args.out_dir)
    checkpoints = find_all_checkpoints(args.checkpoints_dir)
    if args.checkpoint_stride > 1:
        n_before = len(checkpoints)
        checkpoints = apply_checkpoint_stride(checkpoints, args.checkpoint_stride)
        print(f"[run_multi_checkpoint_eval] --checkpoint_stride {args.checkpoint_stride}: "
              f"evaluating {len(checkpoints)}/{n_before} checkpoints (first and last always kept)")

    if not checkpoints and args.hf_repo_id:
        print(f"[run_multi_checkpoint_eval] no local checkpoints under {args.checkpoints_dir} - "
              f"nothing to enumerate locally. Pass explicit checkpoint labels via a future run, "
              f"or ensure at least one checkpoint-N directory exists locally to discover the set.")
    if not checkpoints:
        # Deliberately a CLEAN exit (code 0), not SystemExit with a message.
        # "no intermediate checkpoints" is a legitimate state - e.g. a run with
        # --epochs 1 --save_every_epochs 1 may only produce the final save.
        # run_full_experiment.sh uses `set -euo pipefail`, so a non-zero exit
        # here aborts the ENTIRE multi-hour pipeline (skipping Stage 2, Stage 3
        # and all remaining evaluation) over a condition that should merely be
        # skipped and reported.
        print(f"[run_multi_checkpoint_eval] no checkpoint-N directories found under "
              f"{args.checkpoints_dir} - nothing to evaluate at this stage, skipping. "
              f"(This is not an error: the final merged checkpoint is evaluated "
              f"separately by run_full_experiment.sh. If you expected intermediate "
              f"checkpoints, check that --save_every_epochs was set on the "
              f"corresponding sft_train.py/grpo_train.py call.)")
        return

    print(f"[run_multi_checkpoint_eval] found {len(checkpoints)} checkpoints: "
          f"{[s for s, _ in checkpoints]}")

    run_summaries = {}
    for step, ckpt_path in checkpoints:
        label = f"epoch_{step / args.steps_per_epoch:.2f}" if args.steps_per_epoch else f"step_{step}"
        print(f"\n[run_multi_checkpoint_eval] === {label} (checkpoint-{step}) ===")

        eval_model_path = ckpt_path
        if args.mode == "lora":
            merged_dir = os.path.join(tempfile.mkdtemp(), f"merged_{label}")
            print(f"[run_multi_checkpoint_eval] merging into throwaway dir {merged_dir}")
            merge_checkpoint(args.base_model, ckpt_path, merged_dir)
            eval_model_path = merged_dir

        summary_path = run_eval_and_score(eval_model_path, args.split, label, args.out_dir,
                                           args.seed, limit=args.limit)
        with open(summary_path) as f:
            run_summaries[label] = json.load(f)

        if args.ledger:
            run_id = f"{args.run_id_prefix}__{label}"
            config_path = os.path.join(args.out_dir, f"config_{label}.json")
            with open(config_path, "w") as f:
                json.dump({
                    "model": args.base_model, "mode": args.mode,
                    "checkpoint_step": step, "checkpoint_label": label,
                    "checkpoints_dir": args.checkpoints_dir,
                }, f, indent=2)
            from experiment_ledger import append_record
            append_record(args.ledger, run_id, config_path, summary_path,
                          baseline_run_id=args.baseline_run_id,
                          notes=f"checkpoint {label} from {args.checkpoints_dir}")

        if args.mode == "lora":
            shutil.rmtree(os.path.dirname(eval_model_path), ignore_errors=True)

    combined_path = os.path.join(args.out_dir, "all_checkpoint_summaries.json")
    with open(combined_path, "w") as f:
        json.dump(run_summaries, f, indent=2)
    print(f"\n[run_multi_checkpoint_eval] wrote combined summaries -> {combined_path}")

    import evaluator as ev
    baseline = next(iter(run_summaries))  # earliest checkpoint as baseline by default
    ev.compare_runs(run_summaries, os.path.join(args.out_dir, "comparison"), baseline=baseline)
    print(f"[run_multi_checkpoint_eval] wrote ablation/comparison + charts -> "
          f"{os.path.join(args.out_dir, 'comparison')}")


if __name__ == "__main__":
    main()
