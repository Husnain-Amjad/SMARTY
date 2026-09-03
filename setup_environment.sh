#!/usr/bin/env bash
# ============================================================================
# setup_environment.sh - one-shot environment setup for an NVIDIA CUDA GPU
# server (local university server or a cluster node reached via SSH). No
# Colab-specific steps (no uv/nightly-torch workarounds, no ROCm).
#
# Usage (from the repository root, after cloning):
#   bash setup_environment.sh
# ============================================================================
set -euo pipefail

echo "=== GPU / driver check ==="
nvidia-smi --query-gpu=name,memory.total,driver_version,compute_cap --format=csv,noheader

echo ""
echo "=== Step 1: PyTorch (CUDA build) ==="
# Adjust the cu1XX suffix below to match the CUDA version `nvidia-smi` reports
# above if it differs from 12.1 - see https://pytorch.org/get-started/locally/
# Installed FIRST and separately from the packages below: if transformers/
# accelerate/etc. install before torch exists, pip can pull in a wrong
# (CPU-only or mismatched-CUDA) torch as a transitive dependency, which then
# has to be wastefully overridden by this step anyway.
pip install --break-system-packages torch --index-url https://download.pytorch.org/whl/cu121

echo ""
echo "=== Step 2: vLLM ==="
pip install --break-system-packages vllm

echo ""
echo "=== Step 3: Core Python packages + torchao (this also re-confirms Steps 1-2 succeeded) ==="
python3 check_environment.py --install-missing

echo ""
echo "=== Step 4: Remove hf-xet (Hugging Face's newer download backend) ==="
echo "hf-xet is a known source of hangs and hard segfaults during model downloads"
echo "(see huggingface/xet-core issues #850, #869, #483, huggingface/huggingface_hub"
echo "issue #3067/#3266) - particularly under concurrent/parallel downloads, which"
echo "this pipeline does when running multiple models at once. The environment"
echo "variable meant to disable it (HF_HUB_DISABLE_XET=1) is reported unreliable in"
echo "some huggingface_hub versions, so this removes the package entirely instead -"
echo "a package that isn't installed can't be invoked regardless. This has no"
echo "downside here: it just falls back to the classic, reliable HTTP downloader,"
echo "which is slightly slower but has none of these crash/hang issues."
pip uninstall -y hf-xet 2>/dev/null || echo "hf-xet was not installed - nothing to remove"
export HF_HUB_DISABLE_XET=1
if ! grep -q "HF_HUB_DISABLE_XET" ~/.bashrc 2>/dev/null; then
    echo 'export HF_HUB_DISABLE_XET=1' >> ~/.bashrc
    echo "Added HF_HUB_DISABLE_XET=1 to ~/.bashrc for future sessions too."
fi

echo ""
echo "=== Step 5: flash-attention-2 (REQUIRED) ==="
echo "This pipeline forces attn_implementation=flash_attention_2 for SFT and GRPO."
echo "There is NO sdpa fallback - a silent kernel downgrade would mean two runs of"
echo "the same config could differ with nothing recording it. If this install fails,"
echo "training will fail loudly rather than quietly using a different kernel."
echo ""
echo "Requires compute capability >= 8.0 (Ampere/A100 or newer). Compilation can"
echo "take a long time when no prebuilt wheel matches your exact torch/CUDA/Python"
echo "combination - that is expected, not a hang. Installing ninja first helps."
pip install --break-system-packages ninja
pip install --break-system-packages flash-attn --no-build-isolation
python3 -c "import flash_attn; print('flash-attn', flash_attn.__version__, 'OK')"

echo "=== Final verification ==="
python3 -c "import torch; print('torch', torch.__version__, 'cuda available:', torch.cuda.is_available())"
python3 -c "import vllm; print('vllm', vllm.__version__)"
python3 hardware_utils.py

echo ""
echo "Environment setup complete."
