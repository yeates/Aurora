<div align="center">

# Aurora

### Unified Video Editing with a Tool-Using Agent


[![arXiv](https://img.shields.io/badge/arXiv-2605.18748-b31b1b.svg)](https://arxiv.org/abs/2605.18748)
[![Project Page](https://img.shields.io/badge/Project-Page-1f6feb.svg)](https://yeates.github.io/Aurora-Page)
[![HF Models](https://img.shields.io/badge/🤗%20Models-Weights-FFD21E.svg)](https://huggingface.co/yeates/aurora-weights)
[![HF Dataset](https://img.shields.io/badge/🤗%20Dataset-Training-FFD21E.svg)](https://huggingface.co/datasets/yeates/aurora-training-data)
[![License](https://img.shields.io/badge/License-MIT-3da639.svg)](LICENSE)

</div>

Aurora is an agentic video-editing framework. Given a source video and a natural
language request, a tool-using vision-language agent first resolves ambiguity:
it rewrites the request into an editor-facing instruction, decides the edit type,
optionally retrieves a reference image from the web, and optionally grounds a
target object with a mask. A unified video diffusion editor then performs the
actual edit.

The editor uses a WAN2.2-TI2V-5B diffusion transformer, a WAN2.2 VAE, and a
frozen Qwen3.5-4B multimodal LLM. One Aurora checkpoint supports
source-conditioned editing (s2v), video-to-video editing (v2v), and
reference-conditioned editing (sv2v).

## Highlights

- **One editor for many edit types.** Aurora supports object insertion and
  removal, stylization, background/material/color/weather edits, reference-based
  identity insertion, and reasoning edits.
- **Agent-first editing.** A Qwen3-VL planner converts vague user requests into a
  structured plan that the editor can consume reliably.
- **Built-in ambiguity resolution.** Web image search handles external concepts
  such as IPs, celebrities, landmarks, brands, and logos; GroundingDINO + SAM
  provides masks for localized edits.
- **AgentEdit-Bench.** The release includes code for evaluating agent-enhanced
  video editing under textual and visual underspecification.
- **Optional familiar APIs.** The reference path uses `diffsynth` for the editor
  and `transformers` for the agent; optional `diffusers` and vLLM wrappers are
  also provided.

## Contents

- [News](#news)
- [TODO](#todo)
- [Installation](#installation)
- [Model Zoo](#model-zoo)
- [Quick Start](#quick-start)
- [Inference](#inference)
- [Agent Pipeline](#agent-pipeline)
- [Optional Backends](#optional-backends)
- [Training](#training)
- [Benchmarks](#benchmarks)
- [Data Attribution](#data-attribution)
- [Acknowledgements](#acknowledgements)
- [Citation](#citation)

## News

- **2026-06**: Video-editing training subsets released as
  WebDataset shards at
  [yeates/aurora-training-data](https://huggingface.co/datasets/yeates/aurora-training-data).
- **2026-06**: Day-0 vLLM support — the Qwen3-VL agent planner can be served with
  vLLM for faster planning (`pip install -e ".[vllm]"`).
- **2026-06**: Day-0 🤗 Diffusers support — `AuroraPipeline` runs the full agent
  planner + editor pipeline through the familiar `DiffusionPipeline` API
  (`pip install -e ".[diffusers]"`). See [Optional Backends](#optional-backends).
- **2026-06**: Aurora code and weight release.
- **2026-05**: Aurora paper released on arXiv.

## TODO

Release progress — checked items are available now; the rest are on the way.

- [x] Inference and training code
- [x] Pre-trained weights — editor + agent LoRA ([yeates/aurora-weights](https://huggingface.co/yeates/aurora-weights))
- [x] Training data (self-curated subsets) ([yeates/aurora-training-data](https://huggingface.co/datasets/yeates/aurora-training-data))
- [x] 🤗 Diffusers support
- [x] vLLM support
- [ ] AgentEdit-Bench data (eval code is already included)

## Installation

Aurora has been tested with **Python 3.10**, **PyTorch 2.5.0 + CUDA 12.4**,
`transformers==5.3.0`, `flash-attn==2.7.3`, and DeepSpeed.

```bash
git clone https://github.com/yeates/Aurora.git
cd Aurora

# Installs PyTorch/cu124, flash-attn, DeepSpeed, evaluation extras,
# and this repository in editable mode.
bash setup_env.sh
```

The setup script pins `transformers` after `modelscope`, builds `flash-attn`
against PyTorch 2.5.0, installs evaluation dependencies such as `decord` and
`openai`, and runs a small import check.

After installation, the main Python entry points are:

```python
from diffsynth.pipelines.wan_video import WanVideoPipeline
from evaluation.pipeline_loader import load_v2_pipeline
import aurora.agent
from aurora import editor_bridge_video
```

## Model Zoo

Aurora uses four public backbone models plus the two trained Aurora weights.
**All of them download automatically on the first inference run** into `models/`
(or `AURORA_MODEL_DIR`) — no manual step is required. The two Aurora weights live
in a single repository, `yeates/aurora-weights`.

| Component | Role | HF source |
|---|---|---|
| WAN2.2-TI2V-5B DiT | Editor backbone | [Wan-AI/Wan2.2-TI2V-5B](https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B) |
| WAN2.2 VAE (`Wan2.2_VAE.pth`) | Latent encoder/decoder | [Wan-AI/Wan2.2-TI2V-5B](https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B) |
| Qwen3.5-4B | Frozen MLLM used by the editor | [Qwen/Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B) |
| Qwen3-VL-8B-Instruct | Agent planner base model | [Qwen/Qwen3-VL-8B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct) |
| Aurora editor checkpoint | Trained `dit`, `mllm.context_projector`, and `ref_vae_condition` modules | [yeates/aurora-weights](https://huggingface.co/yeates/aurora-weights) (`aurora_editor.safetensors`) |
| Aurora agent adapter | PEFT LoRA on Qwen3-VL (`r=32`, `alpha=64`) | [yeates/aurora-weights](https://huggingface.co/yeates/aurora-weights) (`aurora_agent_vlm/`) |

If you want to pre-fetch everything ahead of time:

```bash
python -m evaluation.model_download --what all   # or: --what editor | agent
```

Expected layout (created automatically; files can also be placed here manually):

```text
models/
├── Wan2.2-TI2V-5B/
│   ├── diffusion_pytorch_model-0000{1,2,3}-of-00003.safetensors
│   └── Wan2.2_VAE.pth
├── Qwen3.5-4B/
├── Qwen3-VL-8B-Instruct/
├── aurora_editor.safetensors
└── aurora_agent_vlm/
```

`aurora_editor.safetensors` contains only the trained Aurora modules. It is
loaded on top of the frozen WAN and Qwen3.5-4B backbones by
`evaluation.pipeline_loader.load_v2_pipeline`.

## Quick Start

The editor checkpoint and all backbone models download automatically on first
use into `models/`. To fetch everything up front:
`python -m evaluation.model_download --what all`.

Run the editor directly on a benchmark-style dataset (omit `--ckpt` to use the
auto-downloaded Aurora checkpoint; pass it to point at a custom one):

```bash
python evaluation/editverse_infer.py \
  --bench_dir /path/to/editverse \
  --num_gpus 8 \
  --max_pixels 399360 \
  --num_frames 81 \
  --save_frames 64 \
  --cfg_scale 2.0 \
  --image_cfg_scale 1.0 \
  --fallback_to_two_pass_cfg
```

To use the full agentic pipeline, first let the agent produce edit records, then
render those records with the editor:

```bash
export SERPER_KEY_ID=<your-serper-key>

# Agent planner (Qwen3-VL base + aurora_agent_vlm adapter auto-download on first use)
python -m aurora.agent \
  --custom_cases_jsonl /path/to/cases.jsonl \
  --custom_only \
  --mask_backend grounded_sam \
  --out_dir /path/to/agent_out \
  --serper_api_key "$SERPER_KEY_ID"

# Render with the editor (models/aurora_editor.safetensors is auto-downloaded by
# any *_infer.py run or `python -m evaluation.model_download --what editor`)
python -m aurora.editor_bridge_video \
  --records_jsonl /path/to/agent_out/agent_pipeline_records.jsonl \
  --ckpt models/aurora_editor.safetensors \
  --out_dir /path/to/agent_video_out \
  --num_frames 81 \
  --save_frames 64
```

## Inference

All editor evaluation scripts share the same loader in
`evaluation/pipeline_loader.py` and support multi-GPU inference with one worker
per GPU.

### Guidance Defaults

Aurora uses three-pass classifier-free guidance. The editor combines an
unconditional pass, a visual-only pass, and a fully conditioned pass:

```text
result = uncond + image_cfg * (visual_neg - uncond) + text_cfg * (positive - visual_neg)
```

Use these flags to control guidance:

- `--cfg_scale`: text CFG (`text_cfg`)
- `--image_cfg_scale`: image CFG (`image_cfg`)
- `--fallback_to_two_pass_cfg`: when `image_cfg == 1.0`, skip the unconditional
  branch and use the equivalent two-pass path

The best global default is:

```bash
--cfg_scale 2.0 --image_cfg_scale 1.0 --fallback_to_two_pass_cfg
```

No single CFG setting is optimal for every edit type. Stylization, background or
object changes, object removal, and identity insertion often benefit from
`image_cfg` in `{1.5, 2.0}`. Weather, color, material, reasoning, effects, and
combined edits usually work best with `image_cfg = 1.0`.

### EditVerse

EditVerse inference generates 81 frames at `480x832`, then saves 64 frames to
match the baseline length and resolution.

```bash
python evaluation/editverse_infer.py \
  --ckpt <CKPT> \
  --bench_dir /path/to/editverse \
  --num_gpus 8 \
  --max_pixels 399360 \
  --num_frames 81 \
  --save_frames 64 \
  --cfg_scale 2.0 \
  --image_cfg_scale 1.0 \
  --fallback_to_two_pass_cfg
```

For stage-2 checkpoints, use `--max_pixels 921600`.

### OpenVE

OpenVE source frames are linspace-resampled to exactly 81 frames with
`temporal_resize`; output FPS is rescaled to match the source duration.

```bash
python evaluation/openve_infer.py \
  --ckpt <CKPT> \
  --bench_dir /path/to/openve \
  --num_gpus 8 \
  --num_frames 81 \
  --frame_sampling_mode temporal_resize \
  --cfg_scale 2.0 \
  --image_cfg_scale 1.0
```

Each benchmark has a matching `evaluation/<bench>_score.py` script that calls a
VLM judge through an OpenAI-compatible API. Configure your provider and API key
before scoring.

## Agent Pipeline

The agent pipeline has two stages:

1. `aurora.agent` reads the user request and source video, then writes structured
   records.
2. `aurora.editor_bridge_video` maps those records into editor inputs and renders
   the output video.

For each case, the agent emits a JSON record with this planning contract:

```json
{
  "refined_text_instruction": "clean editor-facing instruction",
  "subtask": "global_style | remove_object | add_object | replace_object | change_background | change_color | change_weather | add_effect | customization | combined_tasks | camera_edit",
  "image_search": "short web search query or false",
  "mask": "short mask noun phrase or false"
}
```

When `image_search` is set, Aurora calls Serper image search, downloads
candidates, and uses an image-selector pass to choose the best reference image.
When `mask` is set, the GroundingDINO + SAM backend produces an object mask.

Use `--disable_image_search` to force all web lookups off for fully local
benchmark comparisons. `aurora/editor_bridge_openve.py` provides the analogous
bridge for OpenVE-style inputs.

## Optional Backends

Aurora ships optional wrappers for users who prefer `diffusers` or vLLM. These
wrappers are additive: the reference implementation remains `diffsynth` for the
editor and `transformers` for the agent.

Install only the backend you need:

```bash
pip install -e ".[diffusers]"

# Install vLLM in a separate environment so it does not disturb the
# PyTorch 2.5.0 / flash-attn 2.7.3 reference environment.
pip install "vllm>=0.11"
```

### diffusers

`aurora/diffusers_pipeline.py` exposes `AuroraPipeline`, a
`diffusers.DiffusionPipeline` that can package both the Qwen3-VL planner and the
diffusion editor. The diffusion math is delegated to the verified
`WanVideoPipeline`.

```python
from diffsynth import VideoData, save_video
from aurora.diffusers_pipeline import AuroraPipeline

# All weights auto-download on first use; omit `ckpt` for the Aurora checkpoint
# or pass a path/HF repo id for a custom one. Drop agent_base/agent_adapter for
# an editor-only pipeline.
pipe = AuroraPipeline.from_pretrained(
    device="cuda:0",
    ref_max_items=8,
    agent_base="models/Qwen3-VL-8B-Instruct",
    agent_adapter="models/aurora_agent_vlm",
)

src = VideoData("source.mp4", length=81, max_pixels=480 * 832)

# End-to-end path: plan from a raw request, then render.
out = pipe(
    request="put the man in a snowy street",
    video=src,
    num_frames=len(src),
    height=H, width=W,
    guidance_scale=2.0,
    image_guidance_scale=1.0,
)
save_video(list(out.frames[0]), "out.mp4", fps=24, quality=5)
print(out.plan)

# Editor-only path: skip the agent and provide the editor prompt directly.
out = pipe(
    prompt="make it snow",
    video=src,
    num_frames=len(src),
    height=H, width=W,
    guidance_scale=2.0,
    image_guidance_scale=1.0,
)
save_video(list(out.frames[0]), "out.mp4", fps=24, quality=5)
```

`guidance_scale` maps to `cfg_scale`; `image_guidance_scale` maps to
`image_cfg_scale`. Omit `agent_base` and `agent_adapter` for an editor-only
pipeline. The diffusion editor is not served by vLLM; use this pipeline for a
single Python API around the editor.

### vLLM Agent

vLLM is supported for the Qwen3-VL planner only. The diffusion editor still runs
on the Aurora editor engine. Because the LoRA adapter is merged at load time,
first persist a merged agent model, then pass `--agent_backend vllm`.

```bash
# One-time: merge the LoRA into a plain model that vLLM can serve statically.
python -c "from aurora.agent_vllm import merge_agent_to_dir; \
  merge_agent_to_dir('models/Qwen3-VL-8B-Instruct','models/aurora_agent_vlm','models/aurora_agent_vlm_merged')"

# Step 1: plan with vLLM.
python -m aurora.agent \
  --agent_backend vllm \
  --agent_merged_dir models/aurora_agent_vlm_merged \
  --custom_cases_jsonl cases.jsonl \
  --custom_only \
  --out_dir agent_out

# Step 2: render with the standard editor bridge.
python -m aurora.editor_bridge_video \
  --records_jsonl agent_out/agent_pipeline_records.jsonl \
  --ckpt <CKPT> \
  --out_dir agent_video_out \
  --num_frames 81 \
  --save_frames 64
```

Greedy decoding is preserved with `temperature=0.0`, and
`agent_pipeline_records.jsonl` keeps the same contract as the default
transformers path. Use `--agent_backend hf` for the reference path. Do not use
vLLM `--enable-lora` for this model because Qwen3-VL vision adapters require the
pre-merged directory.

## Training

Training runs on multiple GPUs using `accelerate` and DeepSpeed
ZeRO-2. The DeepSpeed config lives at `scripts/ds_config.json`.

Only these modules are trained:

- `mllm.context_projector`
- `dit`
- `ref_vae_condition`

The WAN VAE and Qwen3.5-4B MLLM remain frozen.

```bash
# Stage 1: low-resolution warmup (max_pixels=399360).
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash scripts/train_stage1.sh

# Stage 2: high-resolution training (max_pixels=921600), initialized from stage 1.
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  STAGE1_CKPT=/path/to/stage1/latest.safetensors \
  bash scripts/train_stage2.sh
```

Stage 1 is a low-resolution warmup and Stage 2 trains at higher resolution
(`max_pixels=921600`), initialized from Stage 1.

> **Note.** The optimization hyperparameters in these single-node launchers
> (learning rate, `gradient_accumulation_steps`, etc.) are values for
> single-node development/debugging on one 8-GPU machine, **not** the production
> recipe. For the multi-node training settings used for the released model, refer
> to the training details in the paper appendix.

The launchers read paths from environment variables including `MODEL_DIR`,
`META_DIR`, `DATA_DIR`, `LMDB_DIR`, `OUTPUT_DIR`, `STAGE1_CKPT`, and
`MLLM_MODEL`. Defaults are defined at the top of each launcher. Run
`python train.py --help` for the full flag list.

## Benchmarks

### AgentEdit-Bench

AgentEdit-Bench is the agentic video-editing benchmark introduced with Aurora. It
stresses retrieval-conditioned edits such as IP, branded-entity, and landmark
insertion, as well as reasoning edits and object removal. Evaluation uses a
multi-axis VLM rubric.

Unlike non-agentic benchmarks, AgentEdit-Bench uses the agent path: Aurora may
perform web image search and feed the retrieved reference into the editor.
Therefore, AgentEdit-Bench scores are not directly comparable to EditVerse or
OpenVE scores.

```bash
python evaluation/agentedit_bench_infer.py \
  --ckpt <CKPT> \
  --bench_dir /path/to/AgentEdit-Bench \
  --num_gpus 8 \
  --agent_records_jsonl /path/to/agent_out/agent_pipeline_records.jsonl

python evaluation/agentedit_bench_score.py --help
```

`--agent_records_jsonl` makes the editor consume the agent records, including the
refined prompt and retrieved reference. Omit it to run on raw benchmark prompts.

### Other Benchmarks

Aurora also includes inference and scoring scripts for:

- EditVerse: `evaluation/editverse_infer.py`,
  `evaluation/editverse_score.py`
- OpenVE: `evaluation/openve_infer.py`, `evaluation/openve_score.py`

## Data Attribution

Aurora's editor is trained on a mixture of image-edit, video-edit, and
video-reference datasets. Please consult and comply with each upstream dataset
license.

The **video-editing subsets reported in the paper's Table 1** are released —
repackaged as ready-to-use WebDataset shards with the training captions and
reference images — at
[**yeates/aurora-training-data**](https://huggingface.co/datasets/yeates/aurora-training-data)
(OpenS2V, the Ditto combined-task split, ROSE, EffectErase, and SpatialVID).
This is a curated slice, not the full training mixture; all upstream sources are
listed below.

<details>
<summary>Training dataset sources</summary>

| Group | Dataset | Source |
|---|---|---|
| Image edit | CrispEdit-2M | https://huggingface.co/datasets/WeiChow/CrispEdit-2M |
| Image edit | UltraEdit | https://huggingface.co/datasets/BUAADreamer/UltraEdit |
| Image edit | TextEdit | https://huggingface.co/datasets/opencompass/TextEdit |
| Video edit | Ditto-1M | https://huggingface.co/datasets/QingyanBai/Ditto-1M |
| Video edit | EgoEdit | https://huggingface.co/datasets/liguang0115/EgoEdit |
| Video edit | OpenVE-3M | https://huggingface.co/datasets/Lewandofski/OpenVE-3M |
| Video edit | ReCo-Data | https://huggingface.co/datasets/HiDream-ai/ReCo-Data |
| Video edit | EffectErase | https://huggingface.co/datasets/FudanCVL/EffectErase |
| Video edit | ROSE | https://huggingface.co/datasets/Kunbyte/ROSE-Dataset |
| Video ref | HuMoSet | https://modelscope.cn/datasets/leoniuschen/HuMoSet |
| Video ref | OpenS2V | https://huggingface.co/datasets/BestWishYsh/OpenS2V-5M |
| Video ref | RefVie | https://huggingface.co/datasets/linyq/kiwi_edit_training_data |
| Video ref | SpatialVID | https://huggingface.co/datasets/SpatialVID/SpatialVID |

</details>

## Acknowledgements

The diffusion engine in `diffsynth/` derives from ModelScope's
[DiffSynth-Studio](https://github.com/modelscope/DiffSynth-Studio) (Apache-2.0);
see [`NOTICE`](NOTICE). Aurora builds on the WAN2.2-TI2V-5B / WAN2.2 VAE and
Qwen3.5-4B / Qwen3-VL-8B-Instruct model families.

## License

Released under the [MIT License](LICENSE). Pretrained model weights and
third-party datasets are governed by their own respective licenses.

## Citation

If you find Aurora useful, please cite:

```bibtex
@article{yu2026aurora,
  title={Aurora: Unified Video Editing with a Tool-Using Agent},
  author={Yu, Yongsheng and Zeng, Ziyun and Xiao, Zhiyuan and Zhou, Zhenghong and Hua, Hang and Xiong, Wei and Luo, Jiebo},
  journal={arXiv preprint arXiv:2605.18748},
  year={2026}
}
```
