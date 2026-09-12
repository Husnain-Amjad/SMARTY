"""
Hardware detection and platform-aware configuration for A100, H100, ROCm
(MI-series), and multi-GPU cluster setups. Every training/inference script
imports detect_platform() and uses its recommendations rather than hardcoding
one hardware target - this is what "optimized for A100/H100/ROCm/cluster" means
concretely: the same script picks different defaults depending on what it
actually finds, instead of you hand-editing flags per machine.

This module never hard-fails if it can't detect something (e.g. running on a
machine with no GPU at all, for a CPU smoke test) - it degrades to conservative
defaults and tells you why via the returned report's 'notes' field.
"""

import os
import subprocess
from dataclasses import dataclass, field


@dataclass
class PlatformInfo:
    backend: str                     # "cuda", "rocm", "cpu"
    gpu_name: str = "unknown"
    gpu_count: int = 0
    gpu_memory_gb: float = 0.0
    is_a100: bool = False
    is_h100: bool = False
    is_mi_series: bool = False
    recommended_attn_impl: str = "sdpa"
    recommended_gpu_memory_utilization: float = 0.85
    recommended_max_model_len: int = 4096
    recommended_tf32: bool = True
    supports_bnb: bool = False
    supports_vllm_native_install: bool = True
    multi_gpu: bool = False
    recommended_dist_strategy: str = "none"   # "none", "ddp", "fsdp"
    notes: list = field(default_factory=list)


def recommended_batch_size(gpu_memory_gb: float, backend: str = "vllm") -> int:
    """
    Heuristic starting point for how many prompts to send to a single
    generate() call, scaled by available GPU memory and which serving
    backend is in play. vLLM manages KV cache continuously and handles much
    larger concurrent request batches than a padded HF-generate call, which
    allocates for the longest sequence in the batch up front - so the two
    backends get different scaling.

    This is a STARTING POINT, not a hard guarantee - actual memory use also
    depends on model size, sequence length, and how many other things are
    resident on the GPU. Pair with an OOM-safe retry (see safe_generate in
    run_augmentation.py) rather than treating this number as exact.
    """
    if gpu_memory_gb <= 0:
        return 8  # CPU or undetected - conservative default
    if backend == "vllm":
        if gpu_memory_gb >= 75:
            return 256
        elif gpu_memory_gb >= 38:
            return 128
        elif gpu_memory_gb >= 20:
            return 64
        else:
            return 32
    else:  # HF-generate fallback - padded batches, more memory-hungry per item
        if gpu_memory_gb >= 75:
            return 64
        elif gpu_memory_gb >= 38:
            return 32
        elif gpu_memory_gb >= 20:
            return 16
        else:
            return 8


def select_attn_implementation(prefer_flash: bool = True) -> str:
    """
    Returns the attention implementation to use for training. Never sdpa.

    Resolution order:
      1. SMART_ATTN_IMPL environment variable, if set (full manual override).
      2. "kernels-community/flash-attn2@v2" - the default.

    Why the Hub-kernel string rather than plain "flash_attention_2":

      transformers>=5 resolves "flash_attention_2" to the compiled flash-attn
      pip package if present, and otherwise FALLS BACK to the
      kernels-community/flash-attn2 Hub kernel. Both paths work, but the
      fallback is implicit - nothing in the logs or training_config.json
      records which one was actually used, and HF documents that the default
      Hub branch a bare name resolves to "can change from one Transformers
      release to the next". Naming the repo and pinning the branch makes the
      kernel an explicit, recorded, reproducible choice instead of an
      environment-dependent one.

      The @v2 pin also avoids the known v3 issue where CUDA 12.8 builds fail
      in the backward pass for grouped-query-attention models - which both
      Qwen2.5-Math and deepseek-math are. (See
      huggingface/kernels-community#1085.)

    To use the compiled pip package instead, or a different build:
        export SMART_ATTN_IMPL=flash_attention_2
        export SMART_ATTN_IMPL=kernels-community/flash-attn2@v3
        export SMART_ATTN_IMPL=kernels-community/vllm-flash-attn3
    """
    import os
    override = os.environ.get("SMART_ATTN_IMPL", "").strip()
    if override:
        return override
    return "kernels-community/flash-attn2@v2"


def load_model_with_best_attention(model_class, model_name_or_path, **kwargs):
    """
    Loads a model with the attention implementation from
    select_attn_implementation(). Returns (model, attn_impl_used).

    One retry is allowed, and only ever onto ANOTHER flash-attention path -
    never onto sdpa. If the pinned Hub-kernel string is rejected (e.g. an
    older transformers that predates Hub-kernel support), this retries with
    plain "flash_attention_2", which on transformers>=5 itself resolves to
    the same Hub kernel when the pip package is absent. If that also fails,
    it raises: a silent downgrade to sdpa would mean two runs of the same
    config used different kernels with nothing recording it.
    """
    attn_impl = select_attn_implementation()
    try:
        model = model_class.from_pretrained(model_name_or_path,
                                            attn_implementation=attn_impl, **kwargs)
        print(f"[hardware_utils] loaded {model_name_or_path} with attn_implementation={attn_impl}")
        return model, attn_impl
    except Exception as e:
        if attn_impl == "flash_attention_2":
            raise RuntimeError(
                f"[hardware_utils] failed to load '{model_name_or_path}' with "
                f"attn_implementation=flash_attention_2: {type(e).__name__}: {e}\n"
                f"This pipeline requires flash-attention (no sdpa fallback, by design). "
                f"Either install the compiled package:\n"
                f"    pip install flash-attn --no-build-isolation\n"
                f"or install the kernels library so transformers can pull the Hub kernel:\n"
                f"    pip install -U kernels"
            ) from e
        print(f"[hardware_utils] attn_implementation={attn_impl} was rejected "
              f"({type(e).__name__}: {e}) - retrying with plain 'flash_attention_2'. "
              f"On transformers>=5 this resolves to the same Hub kernel when the "
              f"compiled flash-attn package isn't installed.")
        try:
            model = model_class.from_pretrained(model_name_or_path,
                                                attn_implementation="flash_attention_2", **kwargs)
            print(f"[hardware_utils] loaded {model_name_or_path} with "
                  f"attn_implementation=flash_attention_2 (retry)")
            return model, "flash_attention_2"
        except Exception as e2:
            raise RuntimeError(
                f"[hardware_utils] failed to load '{model_name_or_path}' with both "
                f"'{attn_impl}' and 'flash_attention_2'. Last error: "
                f"{type(e2).__name__}: {e2}\n"
                f"No sdpa fallback by design. Install one of:\n"
                f"    pip install -U kernels                       # Hub kernel (no compilation)\n"
                f"    pip install flash-attn --no-build-isolation  # compiled package\n"
                f"Or override explicitly: export SMART_ATTN_IMPL=<impl>"
            ) from e2


def _try_run(cmd):
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        return out.stdout if out.returncode == 0 else ""
    except Exception:
        return ""


def detect_platform() -> PlatformInfo:
    info = PlatformInfo(backend="cpu")

    try:
        import torch
    except ImportError:
        info.notes.append("torch not importable - returning CPU defaults; install torch first.")
        return info

    is_rocm = bool(getattr(torch.version, "hip", None))
    cuda_available = torch.cuda.is_available()

    if not cuda_available:
        info.notes.append("torch.cuda.is_available() is False - no GPU visible to this process, "
                           "or torch was installed without GPU support. Using CPU defaults.")
        return info

    info.backend = "rocm" if is_rocm else "cuda"
    info.gpu_count = torch.cuda.device_count()
    info.multi_gpu = info.gpu_count > 1

    try:
        props = torch.cuda.get_device_properties(0)
        info.gpu_name = props.name
        info.gpu_memory_gb = round(props.total_memory / (1024 ** 3), 1)
    except Exception as e:
        info.notes.append(f"could not read device properties: {e}")

    name_lower = info.gpu_name.lower()
    info.is_a100 = "a100" in name_lower
    info.is_h100 = "h100" in name_lower
    info.is_mi_series = is_rocm and ("mi2" in name_lower or "mi3" in name_lower or "instinct" in name_lower)

    # --- attention implementation ---
    # sdpa is portable (CUDA + ROCm via AOTriton) and is the pipeline-wide default.
    # flash_attention_2 is CUDA-only and needs the separate `flash-attn` pip package
    # (fiddly to match against torch/CUDA versions) - only suggest it, never force it,
    # since sdpa already dispatches to a fused fast kernel on Ampere+/MI200+.
    info.recommended_attn_impl = "sdpa"
    if info.backend == "cuda" and (info.is_a100 or info.is_h100):
        info.notes.append("On A100/H100, 'sdpa' already dispatches to a fused fast-attention "
                           "kernel. flash_attention_2 (separate `pip install flash-attn`) may give "
                           "a further few-percent speedup but isn't required - try sdpa first.")

    # --- bitsandbytes (4-bit QLoRA) ---
    info.supports_bnb = (info.backend == "cuda")
    if info.backend == "rocm":
        info.notes.append("bitsandbytes 4-bit QLoRA is unreliable on ROCm - sft_train.py's "
                           "--use_bnb auto-disables itself here regardless of this flag.")

    # --- vLLM ---
    info.supports_vllm_native_install = (info.backend == "cuda")
    if info.backend == "rocm":
        info.notes.append("vLLM needs a ROCm-specific build (not `pip install vllm`) - see "
                           "ROCM_TUTORIAL.md Step 3. run_eval.py/run_augmentation.py fall back "
                           "to HF-generate automatically if it's not importable.")

    # --- GPU-memory-scaled defaults ---
    if info.gpu_memory_gb >= 75:      # 80GB-class card (A100-80GB, H100-80GB, MI250/300)
        info.recommended_gpu_memory_utilization = 0.92
        info.recommended_max_model_len = 8192
    elif info.gpu_memory_gb >= 38:    # 40GB-class card (A100-40GB)
        info.recommended_gpu_memory_utilization = 0.88
        info.recommended_max_model_len = 4096
    else:                              # smaller cards - be conservative
        info.recommended_gpu_memory_utilization = 0.80
        info.recommended_max_model_len = 2048
        info.notes.append(f"GPU memory ({info.gpu_memory_gb}GB) is under 40GB - consider "
                           f"--mode lora over --mode full, and/or --use_bnb (CUDA only), for "
                           f"7B+ models.")

    if info.is_h100:
        info.notes.append("H100 detected: bf16/TF32 throughput is substantially higher than "
                           "A100 at the same nominal batch size - consider raising "
                           "--per_device_batch_size and re-measuring rather than assuming the "
                           "A100-tuned batch size is optimal here.")

    # --- multi-GPU / cluster ---
    if info.multi_gpu:
        info.recommended_dist_strategy = "fsdp"  # for full fine-tuning; LoRA is fine under plain DDP
        info.notes.append(
            f"{info.gpu_count} GPUs visible - launch sft_train.py/grpo_train.py under "
            f"`accelerate launch` (see cluster_configs/ and CLUSTER_TUTORIAL.md) rather than "
            f"plain `python`, and set --tensor_parallel_size on run_eval.py/run_augmentation.py "
            f"to shard vLLM inference across all GPUs instead of using only GPU 0."
        )

    return info


def print_report(info: PlatformInfo = None):
    info = info or detect_platform()
    print("=" * 70)
    print(f"[hardware] backend={info.backend}  gpu={info.gpu_name}  "
          f"count={info.gpu_count}  memory={info.gpu_memory_gb}GB")
    print(f"[hardware] is_a100={info.is_a100}  is_h100={info.is_h100}  is_mi_series={info.is_mi_series}")
    print(f"[hardware] recommended: attn_impl={info.recommended_attn_impl}  "
          f"gpu_memory_utilization={info.recommended_gpu_memory_utilization}  "
          f"max_model_len={info.recommended_max_model_len}")
    print(f"[hardware] supports_bnb={info.supports_bnb}  "
          f"supports_vllm_native_install={info.supports_vllm_native_install}")
    print(f"[hardware] multi_gpu={info.multi_gpu}  recommended_dist_strategy={info.recommended_dist_strategy}")
    for note in info.notes:
        print(f"[hardware] NOTE: {note}")
    print("=" * 70)
    return info


if __name__ == "__main__":
    print_report()
