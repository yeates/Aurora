#!/bin/bash
# Aurora environment setup (training + inference + evaluation).
#
# Tested base: Python 3.10, PyTorch 2.5.0 + CUDA 12.4, CUDA toolkit at /usr/local/cuda.
# Run this from the repository root (the directory containing pyproject.toml).
set -euo pipefail

echo "=== Aurora Environment Setup ==="

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"

# 1. PyTorch (CUDA 12.4 build, validated combination).
#    flash-attn must match this torch ABI; do NOT bump torch without re-pinning flash-attn.
echo "--- Installing PyTorch 2.5.0 (cu124) ---"
pip install torch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0 --index-url https://download.pytorch.org/whl/cu124

# 2. Core Python deps in one shot (modelscope first; it may pull the wrong transformers).
echo "--- Installing core Python packages ---"
pip install accelerate pyyaml modelscope imageio imageio-ffmpeg einops wandb \
    safetensors sentencepiece protobuf pandas peft lmdb datasets huggingface-hub==0.34

# 3. Pin transformers AFTER modelscope (overrides whatever modelscope installed).
echo "--- Pinning transformers==5.3.0 (MUST be after modelscope) ---"
pip install transformers==5.3.0

# 4. flash-attn (MUST match torch: torch 2.5.0 -> flash-attn 2.7.3).
#    If you see "undefined symbol" on import, rebuild with --no-build-isolation.
echo "--- Installing flash-attn 2.7.3 ---"
pip install flash-attn==2.7.3 --no-build-isolation

# 5. DeepSpeed (needs nvcc; uses CUDA_HOME).
echo "--- Installing DeepSpeed ---"
CUDA_HOME="${CUDA_HOME}" pip install deepspeed

# 6. Evaluation / benchmark deps.
#    setuptools<70 is required because ViCLIP code uses pkg_resources.packaging
#    (removed in setuptools>=70). openai is the Gemini OpenAI-compatible client.
#    opencv-python-headless is needed by aurora.agent (cv2 fallback for video frame sampling).
echo "--- Installing evaluation deps ---"
pip install decord openai ftfy 'setuptools<70' opencv-python-headless

# 7. Verify imports — if this fails, something went wrong above.
echo "--- Verifying imports ---"
python3 -c "
import transformers, deepspeed, peft, imageio, flash_attn, modelscope, decord, openai, cv2
assert transformers.__version__ == '5.3.0', f'transformers wrong: {transformers.__version__}'
print(f'OK: transformers={transformers.__version__} flash_attn={flash_attn.__version__} deepspeed={deepspeed.__version__}')
print(f'    decord={decord.__version__} openai={openai.__version__} cv2={cv2.__version__}')
"

# 8. Install the Aurora packages (diffsynth + aurora + evaluation) in editable mode.
#    [diffusers] extra pulls in `diffusers`, required by aurora.diffusers_pipeline.AuroraPipeline.
echo "--- Installing Aurora packages (pip install -e '.[diffusers]') ---"
cd "${SCRIPT_DIR}"
pip install -e '.[diffusers]'
python3 -c "import diffusers; print(f'OK: diffusers={diffusers.__version__}')"

echo ""
echo "=== Setup complete! ==="
echo "Download the base weights before training/inference:"
echo "  - WAN2.2-TI2V-5B DiT + Wan2.2_VAE.pth  (hf download Wan-AI/Wan2.2-TI2V-5B)"
echo "  - Qwen3.5-4B MLLM"
echo "Point MODEL_DIR at the directory that contains them, then run:"
echo "  bash ${SCRIPT_DIR}/scripts/train_stage1.sh"
echo "  bash ${SCRIPT_DIR}/scripts/train_stage2.sh"
