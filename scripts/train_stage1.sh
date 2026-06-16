#!/bin/bash
# Aurora stage1 training: frozen Qwen3.5-4B MLLM + trainable context_projector / DiT /
# ref_vae_condition, low-resolution warmup. Single-node, 8-GPU accelerate launcher.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
BASE_DIR="$(cd "${REPO_DIR}/.." && pwd)"

export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export TOKENIZERS_PARALLELISM=false
export WANDB_MODE="${WANDB_MODE:-offline}"
export PYTHONPATH="${REPO_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

# Env-var-driven paths (override as needed). Defaults are relative to the repo.
MODEL_DIR="${MODEL_DIR:-${BASE_DIR}/models}"
META_DIR="${META_DIR:-${BASE_DIR}/dataset/metadata}"
DATA_DIR="${DATA_DIR:-${BASE_DIR}/dataset/videos}"
LMDB_DIR="${LMDB_DIR:-${BASE_DIR}/dataset/lmdb}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_DIR}/output/aurora_stage1}"
LOG_PATH="${LOG_PATH:-${OUTPUT_DIR}/train_stage1.log}"
MLLM_MODEL="${MLLM_MODEL:-${MODEL_DIR}/Qwen3.5-4B}"
DS_CONFIG="${DS_CONFIG:-${SCRIPT_DIR}/ds_config.json}"

# Optional: resume from an existing Aurora checkpoint. Leave empty to start from
# the base WAN2.2-TI2V-5B + VAE weights given in --model_paths.
STAGE1_RESUME_CKPT="${STAGE1_RESUME_CKPT:-}"

PERMANENT_SAVE_STEPS="${PERMANENT_SAVE_STEPS:-2500}"
SAVE_STEPS="${SAVE_STEPS:-500}"
NUM_PROCESSES="${NUM_PROCESSES:-8}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-2}"
DEBUG_DUMP_EVERY="${DEBUG_DUMP_EVERY:-1}"
DEBUG_DUMP_LIMIT="${DEBUG_DUMP_LIMIT:-15}"
DEBUG_DUMP_NUM_RANKS="${DEBUG_DUMP_NUM_RANKS:-8}"
DATASET_MIX_PATTERN="${DATASET_MIX_PATTERN:-img,vid,vid,vid_ref}"
DATASET_MIX_MODE="${DATASET_MIX_MODE:-sequential}"
METADATA_SHUFFLE_SEED="${METADATA_SHUFFLE_SEED:-per_run}"

require_cmd() {
  command -v "$1" >/dev/null 2>&1 || {
    echo "Missing required command: $1" >&2
    exit 1
  }
}

require_path() {
  [ -e "$1" ] || {
    echo "Missing required path: $1" >&2
    exit 1
  }
}

require_cmd accelerate
require_cmd python
require_path "${DS_CONFIG}"
require_path "${META_DIR}/all_img_edit.jsonl"
require_path "${META_DIR}/all_vid_edit_hq_0418.jsonl"
require_path "${META_DIR}/all_vid_ref_with_mask_removal.jsonl"
require_path "${LMDB_DIR}/CrispEdit-2M"
require_path "${MODEL_DIR}/Wan2.2-TI2V-5B/diffusion_pytorch_model-00001-of-00003.safetensors"
require_path "${MODEL_DIR}/Wan2.2-TI2V-5B/diffusion_pytorch_model-00002-of-00003.safetensors"
require_path "${MODEL_DIR}/Wan2.2-TI2V-5B/diffusion_pytorch_model-00003-of-00003.safetensors"
require_path "${MODEL_DIR}/Wan2.2-TI2V-5B/Wan2.2_VAE.pth"
require_path "${MLLM_MODEL}"
mkdir -p "$(dirname "${LOG_PATH}")" "${OUTPUT_DIR}"

cd "${REPO_DIR}"

CKPT_ARGS=()
if [ -n "${STAGE1_RESUME_CKPT}" ]; then
  require_path "${STAGE1_RESUME_CKPT}"
  CKPT_ARGS=(--checkpoint "${STAGE1_RESUME_CKPT}")
fi

accelerate launch \
  --num_processes="${NUM_PROCESSES}" \
  --num_machines=1 \
  --machine_rank=0 \
  --mixed_precision='bf16' \
  --use_deepspeed \
  --gradient_accumulation_steps="${GRADIENT_ACCUMULATION_STEPS}" \
  --deepspeed_config_file "${DS_CONFIG}" \
  --deepspeed_multinode_launcher='standard' \
  train.py \
  --dataset_base_path "${DATA_DIR}" \
  --img_dataset_metadata_path "${META_DIR}/all_img_edit.jsonl" \
  --lmdb_roots "crispedit_2m=${LMDB_DIR}/CrispEdit-2M,textedit=${LMDB_DIR}/TextEdit,ultraedit=${LMDB_DIR}/ultraedit" \
  --vid_dataset_metadata_path "${META_DIR}/all_vid_edit_hq_0418.jsonl" \
  --vid_ref_dataset_metadata_path "${META_DIR}/all_vid_ref_with_mask_removal.jsonl" \
  --dataset_mix_pattern "${DATASET_MIX_PATTERN}" \
  --dataset_mix_mode "${DATASET_MIX_MODE}" \
  --metadata_shuffle_seed "${METADATA_SHUFFLE_SEED}" \
  --num_frames 81 \
  --dataset_repeat 1 \
  --auto_balance_dataset_repeats \
  --model_paths '["'"${MODEL_DIR}"'/Wan2.2-TI2V-5B/diffusion_pytorch_model-00001-of-00003.safetensors","'"${MODEL_DIR}"'/Wan2.2-TI2V-5B/diffusion_pytorch_model-00002-of-00003.safetensors","'"${MODEL_DIR}"'/Wan2.2-TI2V-5B/diffusion_pytorch_model-00003-of-00003.safetensors","'"${MODEL_DIR}"'/Wan2.2-TI2V-5B/Wan2.2_VAE.pth"]' \
  --mllm_model "${MLLM_MODEL}" \
  "${CKPT_ARGS[@]}" \
  --learning_rate 1e-5 \
  --num_epochs 1 \
  --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}" \
  --remove_prefix_in_ckpt "pipe." \
  --output_path "${OUTPUT_DIR}" \
  --trainable_models "mllm.context_projector,dit,ref_vae_condition" \
  --data_file_keys "src_video,tgt_video,ref_image,ref_mask" \
  --project_name "aurora" \
  --exp_name "aurora_stage1_8gpu" \
  --extra_inputs "source_input,ref_image" \
  --source_condition_mode temporal_concat \
  --max_pixels 399360 \
  --prompt_dropout_prob 0.1 \
  --visual_dropout_given_prompt_prob 0.5 \
  --neg_prompt_given_drop_prob 0.5 \
  --ref_image_max_pixels 921600 \
  --mllm_max_pixels_per_frame 147456 \
  --mllm_ref_max_pixels 147456 \
  --mllm_video_sample_fps 1 \
  --mllm_video_min_frames 2 \
  --ref_max_items 8 \
  --save_steps "${SAVE_STEPS}" \
  --permanent_save_steps "${PERMANENT_SAVE_STEPS}" \
  --debug_dump_every "${DEBUG_DUMP_EVERY}" \
  --debug_dump_limit "${DEBUG_DUMP_LIMIT}" \
  --debug_dump_num_ranks "${DEBUG_DUMP_NUM_RANKS}" \
  --find_unused_parameters \
  --dataset_num_workers 4 \
  2>&1 | tee "${LOG_PATH}"

ret=$?
echo "Training exited with code: $ret"
exit $ret
