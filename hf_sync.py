"""
Hugging Face Hub synchronisation for the SMART pipeline.

Pushes trained checkpoints to MODEL repos and all experiment artifacts to a
single DATASET repo, and pulls either back for later evaluation. This makes a
finished run reproducible from the Hub alone, without the original machine.

Layout
------
Checkpoints are split across three model repos by training family, because a
single repo holding ten 7B checkpoints becomes unwieldy and the families are
what you actually compare against each other:

    <sft repo>       stage1
    <replay repo>    simple_replay, sem_replay, num_replay, both_replay
    <grpo repo>      grpo_stage1, grpo_simple_replay, grpo_sem_replay,
                     grpo_num_replay, grpo_both_replay

Within each repo, every checkpoint lands under its own subfolder
`<model_slug>/<variant>/`, so one repo can safely hold several variants AND
several base models without overwriting anything.

All non-model artifacts go to ONE dataset repo. Hugging Face dataset repos
accept arbitrary file types, so predictions, the ledger, LaTeX/Markdown
tables, and PNG figures/diagrams can all live together:

    predictions/<model_slug>/predictions_<run>.jsonl
    evals/<model_slug>/eval_<run>_{detailed.jsonl,summary.json}
    training_logs/<model_slug>/<variant>_training_log.jsonl
    ledger/experiment_ledger.jsonl
    report/tables/...        report/figures/...        report/diagrams/...

Usage
-----
  # push one checkpoint (family auto-routed from the variant name)
  python hf_sync.py --push-model \
      --local ckpts/Qwen_Qwen2.5-Math-7B-Instruct/stage1_merged \
      --model-slug Qwen_Qwen2.5-Math-7B-Instruct --variant stage1 \
      --sft-repo you/sft_qwen7b --replay-repo you/replay_qwen7b --grpo-repo you/grpo_qwen7b

  # push every artifact produced so far
  python hf_sync.py --push-outputs \
      --model-slug Qwen_Qwen2.5-Math-7B-Instruct \
      --outputs-repo you/qwen7b_outputs_results

  # pull a checkpoint back later to evaluate it
  python hf_sync.py --pull-model --repo you/grpo_qwen7b \
      --model-slug Qwen_Qwen2.5-Math-7B-Instruct --variant grpo_both_replay \
      --dest ckpts/restored

  # pull the ledger + predictions back
  python hf_sync.py --pull-outputs --outputs-repo you/qwen7b_outputs_results --dest .

Authentication: `export HF_TOKEN=hf_...` or pass --hf_token.
"""

import argparse
import glob
import os

from storage_utils import push_to_hf, require_input_path


# ---------------------------------------------------------------------------
# Family routing
# ---------------------------------------------------------------------------

def variant_family(variant: str) -> str:
    """
    Maps a variant name to its model-repo family. Order matters: the GRPO
    check must come first, because 'grpo_sem_replay' contains 'replay' and
    would otherwise be misrouted to the replay repo.
    """
    v = variant.strip().lower()
    if v.startswith("grpo"):
        return "grpo"
    if "replay" in v:
        return "replay"
    return "sft"


def resolve_model_repo(variant: str, sft_repo: str, replay_repo: str, grpo_repo: str) -> str:
    family = variant_family(variant)
    repo = {"sft": sft_repo, "replay": replay_repo, "grpo": grpo_repo}[family]
    if not repo:
        raise SystemExit(
            f"[hf_sync] variant '{variant}' routes to the '{family}' family, but no "
            f"--{family}-repo was provided. Pass it, or set SMART_HF_{family.upper()}_REPO."
        )
    return repo


# ---------------------------------------------------------------------------
# Push
# ---------------------------------------------------------------------------

def push_model(local: str, model_slug: str, variant: str, sft_repo: str,
               replay_repo: str, grpo_repo: str, private: bool = True, token: str = None):
    local = str(require_input_path(local))
    repo = resolve_model_repo(variant, sft_repo, replay_repo, grpo_repo)
    path_in_repo = f"{model_slug}/{variant}"
    print(f"[hf_sync] {variant} -> family '{variant_family(variant)}' -> {repo}/{path_in_repo}")
    push_to_hf(local, repo_id=repo, repo_type="model", private=private,
               path_in_repo=path_in_repo, token=token,
               commit_message=f"SMART {model_slug} / {variant}")


# Each entry: (glob pattern, destination prefix inside the dataset repo).
# Missing paths are skipped with a note rather than failing, so this is safe to
# call mid-run when only some stages have completed.
OUTPUT_SPEC = [
    ("outputs/{slug}/predictions_*.jsonl",              "predictions/{slug}"),
    ("outputs/{slug}/eval_*_detailed.jsonl",            "evals/{slug}"),
    ("outputs/{slug}/eval_*_summary.json",              "evals/{slug}"),
    ("outputs/{slug}/weak_clusters.json",               "evals/{slug}"),
    ("outputs/{slug}/semantic_aug.jsonl",               "augmented/{slug}"),
    ("outputs/{slug}/numeric_aug.jsonl",                "augmented/{slug}"),
    ("outputs/{slug}/sft_data.jsonl",                   "data/{slug}"),
    ("ckpts/{slug}/*/training_log.jsonl",               "training_logs/{slug}"),
    ("ckpts/{slug}/*/training_config.json",             "training_configs/{slug}"),
    ("outputs/experiment_ledger.jsonl",                 "ledger"),
    ("outputs/report/tables/*",                         "report/tables"),
    ("outputs/report/figures/*",                        "report/figures"),
    ("outputs/report/diagrams/*",                       "report/diagrams"),
]


def push_outputs(model_slug: str, outputs_repo: str, private: bool = True, token: str = None):
    if not outputs_repo:
        raise SystemExit("[hf_sync] --push-outputs requires --outputs-repo")

    n_pushed, n_skipped = 0, 0
    for pattern, dest_prefix in OUTPUT_SPEC:
        pat = pattern.format(slug=model_slug)
        dest = dest_prefix.format(slug=model_slug)
        matches = sorted(glob.glob(pat))
        if not matches:
            print(f"[hf_sync] skip (no match): {pat}")
            n_skipped += 1
            continue
        for m in matches:
            # For ckpts/<slug>/<variant>/training_log.jsonl, prefix the filename
            # with the variant so files from different variants don't collide
            # in one flat destination folder.
            fname = os.path.basename(m)
            parent = os.path.basename(os.path.dirname(m))
            if fname in ("training_log.jsonl", "training_config.json"):
                fname = f"{parent}_{fname}"
            push_to_hf(m, repo_id=outputs_repo, repo_type="dataset", private=private,
                       path_in_repo=f"{dest}/{fname}", token=token,
                       commit_message=f"SMART {model_slug}: {fname}")
            n_pushed += 1
    print(f"[hf_sync] pushed {n_pushed} artifact(s) to hf://{outputs_repo} "
          f"({n_skipped} pattern(s) had nothing to push yet)")


# ---------------------------------------------------------------------------
# Pull
# ---------------------------------------------------------------------------

def pull_model(repo: str, model_slug: str, variant: str, dest: str, token: str = None):
    """Downloads one checkpoint subfolder back, returning the local path to
    point --model at."""
    from huggingface_hub import snapshot_download
    subfolder = f"{model_slug}/{variant}"
    print(f"[hf_sync] pulling hf://{repo}/{subfolder} -> {dest}")
    path = snapshot_download(repo_id=repo, repo_type="model", token=token,
                              allow_patterns=[f"{subfolder}/*"], local_dir=dest)
    resolved = os.path.join(path, model_slug, variant)
    print(f"[hf_sync] checkpoint ready at: {resolved}")
    print(f"[hf_sync] evaluate it with:  python run_eval.py --model {resolved} "
          f"--split test --use_vllm --out outputs/predictions_restored.jsonl")
    return resolved


def pull_outputs(outputs_repo: str, dest: str, patterns: list = None, token: str = None):
    """Downloads artifacts back. By default pulls the ledger and predictions,
    which is what's needed to regenerate tables/figures without re-running."""
    from huggingface_hub import snapshot_download
    patterns = patterns or ["ledger/*", "predictions/*", "evals/*"]
    print(f"[hf_sync] pulling {patterns} from hf://{outputs_repo} -> {dest}")
    path = snapshot_download(repo_id=outputs_repo, repo_type="dataset", token=token,
                              allow_patterns=patterns, local_dir=dest)
    print(f"[hf_sync] artifacts restored under: {path}")
    print(f"[hf_sync] NOTE: the ledger lands at {os.path.join(dest,'ledger','experiment_ledger.jsonl')}. "
          f"Copy it to outputs/experiment_ledger.jsonl before running "
          f"generate_all_reports.py, which reads that default path.")
    return path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--push-model", action="store_true")
    ap.add_argument("--push-outputs", action="store_true")
    ap.add_argument("--pull-model", action="store_true")
    ap.add_argument("--pull-outputs", action="store_true")

    ap.add_argument("--local", default=None, help="local checkpoint dir (for --push-model)")
    ap.add_argument("--model-slug", default=None,
                     help="filesystem-safe model name, e.g. Qwen_Qwen2.5-Math-7B-Instruct")
    ap.add_argument("--variant", default=None,
                     help="stage1 | simple_replay | sem_replay | num_replay | both_replay | grpo_*")

    ap.add_argument("--sft-repo", default=os.environ.get("SMART_HF_SFT_REPO"))
    ap.add_argument("--replay-repo", default=os.environ.get("SMART_HF_REPLAY_REPO"))
    ap.add_argument("--grpo-repo", default=os.environ.get("SMART_HF_GRPO_REPO"))
    ap.add_argument("--outputs-repo", default=os.environ.get("SMART_HF_OUTPUTS_REPO"))

    ap.add_argument("--repo", default=None, help="explicit repo for --pull-model")
    ap.add_argument("--dest", default="restored")
    ap.add_argument("--patterns", nargs="*", default=None, help="allow_patterns for --pull-outputs")
    ap.add_argument("--public", action="store_true", default=False,
                     help="create repos as public. Default is PRIVATE, appropriate for "
                          "unpublished thesis work.")
    ap.add_argument("--hf_token", default=os.environ.get("HF_TOKEN"))
    args = ap.parse_args()

    private = not args.public

    if args.push_model:
        if not (args.local and args.model_slug and args.variant):
            raise SystemExit("[hf_sync] --push-model requires --local, --model-slug, --variant")
        push_model(args.local, args.model_slug, args.variant, args.sft_repo,
                   args.replay_repo, args.grpo_repo, private=private, token=args.hf_token)

    elif args.push_outputs:
        if not args.model_slug:
            raise SystemExit("[hf_sync] --push-outputs requires --model-slug")
        push_outputs(args.model_slug, args.outputs_repo, private=private, token=args.hf_token)

    elif args.pull_model:
        repo = args.repo or (args.variant and resolve_model_repo(
            args.variant, args.sft_repo, args.replay_repo, args.grpo_repo))
        if not (repo and args.model_slug and args.variant):
            raise SystemExit("[hf_sync] --pull-model requires --model-slug, --variant, and "
                              "either --repo or the matching family repo argument")
        pull_model(repo, args.model_slug, args.variant, args.dest, token=args.hf_token)

    elif args.pull_outputs:
        if not args.outputs_repo:
            raise SystemExit("[hf_sync] --pull-outputs requires --outputs-repo")
        pull_outputs(args.outputs_repo, args.dest, patterns=args.patterns, token=args.hf_token)

    else:
        print("Pass one of --push-model / --push-outputs / --pull-model / --pull-outputs. "
              "See the module docstring for examples.")
