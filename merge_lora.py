"""
Merges a LoRA adapter checkpoint (produced by sft_train.py --mode lora, or a
Trainer auto-checkpoint like 'checkpoint-113' under such a run) into the base
model, producing a standalone directory that vLLM / transformers can load
directly with --model <merged_dir> - no --enable-lora / base-model plumbing
needed at eval time.

Usage:
  python merge_lora.py --base_model Qwen/Qwen2.5-Math-7B \
      --adapter_path ckpts/qwen7b_stage1/checkpoint-113 \
      --out ckpts/qwen7b_stage1/checkpoint-113_merged
"""

import argparse
import os
import shutil

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model", required=True,
                     help="the original base model the adapter was trained from, "
                          "e.g. Qwen/Qwen2.5-Math-7B")
    ap.add_argument("--adapter_path", required=True,
                     help="directory containing adapter_config.json + adapter weights")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    print(f"[merge_lora] loading base model {args.base_model}")
    base = AutoModelForCausalLM.from_pretrained(
        args.base_model, torch_dtype=torch.bfloat16, attn_implementation="sdpa",
    )
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)

    print(f"[merge_lora] applying adapter from {args.adapter_path}")
    merged = PeftModel.from_pretrained(base, args.adapter_path)
    merged = merged.merge_and_unload()  # folds LoRA deltas into base weights

    print(f"[merge_lora] saving standalone merged model -> {args.out}")
    merged.save_pretrained(args.out)
    tokenizer.save_pretrained(args.out)

    src_config = os.path.join(args.adapter_path, "training_config.json")
    if os.path.exists(src_config):
        shutil.copy2(src_config, os.path.join(args.out, "training_config.json"))
        print(f"[merge_lora] copied training_config.json -> {args.out}/training_config.json "
              f"(so experiment_ledger.py can read run metadata directly from the merged dir)")
    else:
        print(f"[merge_lora] no training_config.json found at {args.adapter_path} - "
              f"if you're logging this run with experiment_ledger.py, point --training_config "
              f"at wherever the original run's config actually lives instead.")

    print("[merge_lora] done. Point --model at this directory for eval/vLLM/GRPO.")


if __name__ == "__main__":
    main()
