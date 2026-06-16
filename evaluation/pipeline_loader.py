from __future__ import annotations

import glob
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import torch

from diffsynth.models.state_dict_utils import load_state_dict
from diffsynth.pipelines.wan_video import ModelConfig, WanVideoPipeline


@dataclass(frozen=True)
class ProjectPaths:
    repo_dir: Path
    base_dir: Path
    model_dir: Path
    data_dir: Path
    meta_dir: Path
    eval_output_dir: Path
    validate_output_dir: Path


def default_paths() -> ProjectPaths:
    # All roots are env-overridable with repo-relative defaults so the release is
    # self-contained: place weights under <repo>/../models (or set AURORA_MODEL_DIR)
    # and outputs land under <repo>/outputs (or set AURORA_OUTPUT_DIR).
    repo_dir = Path(__file__).resolve().parent.parent
    base_dir = repo_dir.parent
    model_dir = Path(os.environ.get("AURORA_MODEL_DIR", base_dir / "models"))
    output_dir = Path(os.environ.get("AURORA_OUTPUT_DIR", repo_dir / "outputs"))
    return ProjectPaths(
        repo_dir=repo_dir,
        base_dir=base_dir,
        model_dir=model_dir,
        data_dir=Path(os.environ.get("AURORA_DATA_DIR", base_dir / "dataset" / "videos")),
        meta_dir=Path(os.environ.get("AURORA_META_DIR", base_dir / "dataset" / "metadata")),
        eval_output_dir=output_dir / "run_eval_v2",
        validate_output_dir=output_dir / "validate_v2",
    )


def _build_base_pipeline(
    *,
    device: str,
    mllm_model: Optional[str] = None,
    ref_pad_first: bool = False,
    ref_max_items: int = 8,
    mllm_max_pixels_per_frame: int = 147456,
    mllm_ref_max_pixels: int = 147456,
    mllm_video_sample_fps: float = 1.0,
    mllm_video_min_frames: int = 2,
    paths: Optional[ProjectPaths] = None,
) -> WanVideoPipeline:
    paths = paths or default_paths()
    mllm_model = mllm_model or str(paths.model_dir / "Qwen3.5-4B")
    dit_dir = paths.model_dir / "Wan2.2-TI2V-5B"
    dit_shards = sorted(glob.glob(str(dit_dir / "diffusion_pytorch_model-*.safetensors")))
    if not dit_shards:
        raise FileNotFoundError(f"No DiT shards found under {dit_dir}")
    vae_path = dit_dir / "Wan2.2_VAE.pth"
    if not vae_path.exists():
        raise FileNotFoundError(f"Missing VAE weights: {vae_path}")

    return WanVideoPipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device="cpu",
        model_configs=[ModelConfig(path=dit_shards), ModelConfig(path=str(vae_path))],
        mllm_model=mllm_model,
        ref_pad_first=ref_pad_first,
        ref_max_items=ref_max_items,
        mllm_max_pixels_per_frame=mllm_max_pixels_per_frame,
        mllm_ref_max_pixels=mllm_ref_max_pixels,
        mllm_video_sample_fps=mllm_video_sample_fps,
        mllm_video_min_frames=mllm_video_min_frames,
    )


def _load_partial_checkpoint(pipe: WanVideoPipeline, ckpt_path: str | Path, label: str) -> None:
    ckpt_path = Path(ckpt_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Missing {label} checkpoint: {ckpt_path}")
    print(f"Loading {label} checkpoint: {ckpt_path}")
    state_dict = load_state_dict(str(ckpt_path), torch_dtype=torch.bfloat16, device="cpu")
    expected_keys = set(pipe.state_dict().keys())
    required_prefixes = (
        "dit.",
        "mllm.context_projector.",
        "ref_vae_condition.",
    )
    matched_required = {
        prefix: sum(1 for key in state_dict if key.startswith(prefix) and key in expected_keys)
        for prefix in required_prefixes
    }
    missing_required = [prefix for prefix, count in matched_required.items() if count == 0]
    if missing_required:
        sample_keys = list(state_dict.keys())[:20]
        raise ValueError(
            f"{label} checkpoint does not contain loadable keys for required module "
            f"prefixes {missing_required}. This usually means the checkpoint prefix is "
            f"incompatible with load_v2_pipeline(). sample_keys={sample_keys}"
        )
    result = pipe.load_state_dict(state_dict, strict=False)
    print(
        f"  loaded={len(state_dict)} missing={len(result.missing_keys)} "
        f"unexpected={len(result.unexpected_keys)} required_matches={matched_required}"
    )
    del state_dict


def load_v2_pipeline(
    v2_ckpt: str | Path,
    *,
    device: str = "cuda:0",
    mllm_model: Optional[str] = None,
    ref_pad_first: bool = False,
    ref_max_items: int = 8,
    mllm_video_sample_fps: float = 1.0,
    mllm_video_min_frames: int = 2,
    paths: Optional[ProjectPaths] = None,
) -> WanVideoPipeline:
    paths = paths or default_paths()

    print("Step 1: loading base models")
    pipe = _build_base_pipeline(
        device=device,
        mllm_model=mllm_model,
        ref_pad_first=ref_pad_first,
        ref_max_items=ref_max_items,
        mllm_video_sample_fps=mllm_video_sample_fps,
        mllm_video_min_frames=mllm_video_min_frames,
        paths=paths,
    )

    print("Step 2: restoring v2 checkpoint")
    _load_partial_checkpoint(pipe, v2_ckpt, "v2")

    print(f"Step 3: moving pipeline to {device}")
    pipe.mllm.eval()
    pipe.to(torch.bfloat16)
    pipe.to(device)
    return pipe


def concat_videos(*videos: Iterable) -> list:
    from PIL import Image

    videos = [list(video) for video in videos]
    if not videos:
        return []
    num_frames = min(len(video) for video in videos)
    merged = []
    for frame_id in range(num_frames):
        frames = [video[frame_id] for video in videos]
        widths = [frame.size[0] for frame in frames]
        heights = [frame.size[1] for frame in frames]
        merged_frame = Image.new("RGB", (sum(widths), max(heights)))
        x = 0
        for frame in frames:
            merged_frame.paste(frame, (x, 0))
            x += frame.size[0]
        merged.append(merged_frame)
    return merged
