"""
Stage 2 (part 1): generate predictions from a trained checkpoint on the MATH
test split, producing predictions.jsonl in the shape data_pipeline.diagnose()
expects: {problem_id, subject, level, prediction, gold_boxed}.

Usage:
  python run_eval.py --model ckpts/qwen7b_stage1 --split test \
      --out outputs/predictions.jsonl
"""

import argparse
import json

import data_pipeline as dp
from run_augmentation import build_backend
from reward_fn import extract_boxed
from determinism import set_all_seeds
from storage_utils import ensure_output_path, add_destination_args, dispatch_destination
from templates import render_prompt_only


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--out", default="outputs/predictions.jsonl")
    ap.add_argument("--use_vllm", action="store_true", default=True)
    ap.add_argument("--max_tokens", type=int, default=1024)
    ap.add_argument("--limit", type=int, default=None,
                     help="cap number of problems for a quick smoke test")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--gpu_memory_utilization", type=float, default=0.9,
                     help="fraction of GPU memory vLLM pre-allocates for KV cache. "
                          "A100 80GB can usually run 0.90-0.95 safely; lower it if you "
                          "hit OOM or are sharing the GPU with anything else.")
    ap.add_argument("--max_model_len", type=int, default=4096,
                     help="vLLM max sequence length (prompt+completion). Raise this if "
                          "your problems+solutions are longer than this, lower it to fit "
                          "more KV cache / larger batches on a smaller GPU.")
    ap.add_argument("--tensor_parallel_size", type=int, default=1,
                     help="shard the model across this many GPUs for vLLM inference "
                          "(cluster/multi-GPU setups). 1 = single GPU, the default. "
                          "See CLUSTER_TUTORIAL.md.")
    add_destination_args(ap, default_repo_type="dataset")
    args = ap.parse_args()

    set_all_seeds(args.seed)

    math_rows = dp.load_hendrycks_math(args.split)
    if args.limit:
        math_rows = math_rows[:args.limit]
    print(f"[run_eval] evaluating on {len(math_rows)} problems ({args.split} split)")

    backend = build_backend(args.model, args.use_vllm, seed=args.seed,
                             gpu_memory_utilization=args.gpu_memory_utilization,
                             max_model_len=args.max_model_len,
                             tensor_parallel_size=args.tensor_parallel_size)
    prompts = [render_prompt_only(backend.tokenizer, r["problem"]) for r in math_rows]

    # batch through vLLM in one call if available; HF backend loops internally
    completions = backend.generate(prompts, n=1, temperature=0.0, max_tokens=args.max_tokens)

    ensure_output_path(args.out)
    with open(args.out, "w") as f:
        for row, comp in zip(math_rows, completions):
            gold = extract_boxed(row["solution"])
            if gold is None:
                continue
            f.write(json.dumps({
                "problem_id": row["problem_id"],
                "subject": row["subject"],
                "level": row.get("level", "unknown"),
                "prediction": comp[0],
                "gold_boxed": gold,
            }) + "\n")

    print(f"[run_eval] wrote predictions -> {args.out}")
    dispatch_destination(args.out, args)
    print("Next: python data_pipeline.py --diagnose --predictions "
          f"{args.out} --weak-report outputs/weak_clusters.json")


if __name__ == "__main__":
    main()
