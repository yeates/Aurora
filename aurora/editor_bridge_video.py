"""Run the Aurora video editor from Aurora agent pipeline records.

This is the thin bridge from the agent contract to the current editor API:
  - final_payload.refined_text_instruction -> prompt
  - final_payload.search_image or original ref_image_path -> ref_image
  - final_payload.object_mask -> mask overlay ref_image, because the current
    WanVideoPipeline has no raw mask argument.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import imageio
import numpy as np
from PIL import Image

from diffsynth import VideoData, save_video
from diffsynth.utils.mask_overlay import compose_masked_image
from evaluation.pipeline_loader import default_paths, load_v2_pipeline


REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = Path(os.environ.get("AURORA_OUTPUT_DIR", str(REPO_ROOT / "outputs")))
BENCH_DIR = Path(os.environ.get("AURORA_BENCH_DIR", str(REPO_ROOT / "benchmarks")))
MODEL_DIR = Path(os.environ.get("AURORA_MODEL_DIR", str(REPO_ROOT.parent / "models")))
DEFAULT_CKPT = MODEL_DIR / "aurora_video_editor" / "model.safetensors"
DEFAULT_OUT_DIR = OUTPUT_DIR / "video_editor_from_agent"
DEFAULT_BASELINE_DIR = Path(os.environ.get("AURORA_BASELINE_DIR", str(BENCH_DIR / "editverse" / "baselines" / "EditVerse")))

TIME_DIV_FACTOR = 4
TIME_DIV_REMAINDER = 1

# Per-bench-tag (cfg_scale, image_cfg_scale) for dynamic CFG routing on
# ``plan.subtask``. When image_cfg == 1.0 the 2-pass fallback is auto-enabled
# (image CFG is a no-op there, so 2-pass saves one forward per step).
BENCH_TAG_CFG_BUCKETS: dict[str, tuple[float, float]] = {
    "add object":        (4.0, 1.0),
    "remove object":     (3.5, 1.5),
    "change object":     (3.0, 1.0),
    "stylization":       (2.0, 1.0),
    "ID insertion":      (3.5, 2.0),
    "reasoning":         (3.0, 1.25),
    "change background": (3.5, 1.0),
    "change color":      (3.5, 1.25),
    "change material":   (3.5, 1.0),
    "add effect":        (2.5, 1.0),
    "change weather":    (1.5, 1.25),
    "combined tasks":    (3.0, 1.25),
}


def resolve_dynamic_cfg(edit_type: str | None) -> tuple[float, float, bool]:
    """Return (cfg_scale, image_cfg_scale, fallback_to_two_pass_cfg) for the
    given EditVerse bench tag. KeyError if the tag is unknown — we want the
    benchmark run to fail loudly on unrecognized cases rather than silently
    falling back to a global default."""
    tcfg, icfg = BENCH_TAG_CFG_BUCKETS[str(edit_type)]
    return tcfg, icfg, abs(icfg - 1.0) < 1e-9


def _align_frames(n: int) -> int:
    while n > 1 and n % TIME_DIV_FACTOR != TIME_DIV_REMAINDER:
        n -= 1
    return n


def _read_records(path: Path, bench_ids: set[str] | None) -> list[dict[str, Any]]:
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            if "error" in rec:
                continue
            if bench_ids and str(rec.get("bench_id")) not in bench_ids:
                continue
            records.append(rec)
    return records


def _source_fps(video_path: str) -> float:
    try:
        reader = imageio.get_reader(video_path)
        fps = float(reader.get_meta_data().get("fps", 15))
        reader.close()
        return fps
    except Exception:
        return 15.0


def _target_output_spec(rec: dict[str, Any], args: argparse.Namespace, fallback_video_path: str) -> tuple[int | None, int | None, float]:
    baseline_video = args.baseline_dir / str(rec["bench_id"]) / "video1.mp4" if args.baseline_dir else None
    if baseline_video and baseline_video.exists():
        reader = imageio.get_reader(str(baseline_video))
        frame = reader.get_data(0)
        fps = float(reader.get_meta_data().get("fps", _source_fps(fallback_video_path)))
        reader.close()
        return int(frame.shape[1]), int(frame.shape[0]), fps
    return None, None, _source_fps(fallback_video_path)


def _load_ref_images(
    rec: dict[str, Any],
    use_mask_overlay: bool = False,
    source_frame: Image.Image | None = None,
) -> tuple[list[Image.Image] | None, list[str]]:
    payload = rec.get("final_payload", {})
    paths: list[str] = []
    images: list[Image.Image] = []
    search_image = payload.get("search_image")
    if search_image and os.path.exists(search_image):
        paths.append(search_image)
        images.append(Image.open(search_image).convert("RGB"))
    elif rec.get("ref_image_path") and os.path.exists(rec["ref_image_path"]):
        paths.append(rec["ref_image_path"])
        images.append(Image.open(rec["ref_image_path"]).convert("RGB"))

    if use_mask_overlay:
        mask_path = rec.get("mask", {}).get("mask_path") or payload.get("object_mask")
        if mask_path and os.path.exists(mask_path):
            if source_frame is None:
                raise ValueError("use_mask_overlay requires source_frame=src_video[0]")
            mask_img = Image.open(mask_path)
            composite = compose_masked_image(source_frame, mask_img)
            images.append(composite)
            paths.append(f"compose_masked_image({mask_path}, src_video[0])")

    return (images or None), paths


def run_one(pipe, rec: dict[str, Any], out_dir: Path, args: argparse.Namespace) -> str | None:
    bench_id = str(rec["bench_id"])
    payload = rec.get("final_payload", {})
    prompt = payload.get("refined_text_instruction") or rec.get("prompt", "")
    video_path = rec["video_path"]
    if not os.path.exists(video_path):
        raise FileNotFoundError(video_path)

    case_dir = out_dir / bench_id
    case_dir.mkdir(parents=True, exist_ok=True)
    out_path = case_dir / "generate.mp4"
    if out_path.exists() and out_path.stat().st_size > 0 and not args.overwrite:
        return str(out_path)

    src_video = VideoData(video_path, length=args.num_frames, max_pixels=args.max_pixels)
    raw_frames = len(src_video)
    num_frames = _align_frames(raw_frames)
    if args.num_frames and args.num_frames > 0:
        num_frames = _align_frames(min(args.num_frames, raw_frames))
    if num_frames != raw_frames:
        src_video = VideoData(video_path, length=num_frames, max_pixels=args.max_pixels)

    mllm_src = src_video
    if args.mllm_src_frames > 0 and len(src_video) > args.mllm_src_frames:
        indices = np.linspace(0, len(src_video) - 1, args.mllm_src_frames, dtype=int)
        mllm_src = [src_video[i] for i in indices]

    source_frame = src_video[0] if args.use_mask_overlay else None
    ref_images, ref_paths = _load_ref_images(
        rec,
        use_mask_overlay=args.use_mask_overlay,
        source_frame=source_frame,
    )

    if args.dynamic_cfg_by_bench_tag:
        eff_cfg, eff_icfg, eff_two_pass = resolve_dynamic_cfg(rec.get("edit_type"))
    else:
        eff_cfg = args.cfg_scale
        eff_icfg = args.image_cfg_scale
        eff_two_pass = args.fallback_to_two_pass_cfg

    video = pipe(
        prompt=prompt,
        negative_prompt=args.negative_prompt,
        source_input=src_video,
        src_video=mllm_src,
        ref_image=ref_images,
        height=src_video.height,
        width=src_video.width,
        num_frames=num_frames,
        seed=args.seed,
        cfg_scale=eff_cfg,
        image_cfg_scale=eff_icfg,
        fallback_to_two_pass_cfg=eff_two_pass,
        num_inference_steps=args.num_inference_steps,
        tiled=True,
    )
    frames = list(video)
    if args.save_frames > 0:
        frames = frames[: args.save_frames]
    target_width, target_height, target_fps = _target_output_spec(rec, args, video_path)
    if target_width and target_height:
        frames = [frame.resize((target_width, target_height), Image.LANCZOS) for frame in frames]
    save_video(frames, str(out_path), fps=target_fps, quality=5)

    sidecar = {
        "bench_id": bench_id,
        "source_video": video_path,
        "raw_instruction": rec.get("prompt", ""),
        "prompt": prompt,
        "subtask": payload.get("subtask"),
        "ref_image_paths_for_current_editor_api": ref_paths,
        "agent_final_payload": payload,
        "generation_params": {
            "height": src_video.height,
            "width": src_video.width,
            "saved_height": frames[0].height if frames else None,
            "saved_width": frames[0].width if frames else None,
            "num_frames": num_frames,
            "save_frames": args.save_frames,
            "max_pixels": args.max_pixels,
            "mllm_src_frames": args.mllm_src_frames,
            "num_inference_steps": args.num_inference_steps,
            "cfg_scale": eff_cfg,
            "image_cfg_scale": eff_icfg,
            "fallback_to_two_pass_cfg": eff_two_pass,
            "dynamic_cfg_by_bench_tag": bool(args.dynamic_cfg_by_bench_tag),
            "edit_type_routed": rec.get("edit_type") if args.dynamic_cfg_by_bench_tag else None,
            "seed": args.seed,
            "output_fps": target_fps,
        },
        "output_video": str(out_path),
    }
    (case_dir / "agent_editor_inputs.json").write_text(json.dumps(sidecar, ensure_ascii=False, indent=2), encoding="utf-8")
    return str(out_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--records_jsonl", type=Path, required=True)
    parser.add_argument("--bench_ids", default="", help="Comma-separated record ids. Empty means all.")
    parser.add_argument("--ckpt", type=Path, default=DEFAULT_CKPT)
    parser.add_argument("--out_dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--baseline_dir", type=Path, default=DEFAULT_BASELINE_DIR)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num_frames", type=int, default=81)
    parser.add_argument("--save_frames", type=int, default=64)
    parser.add_argument("--max_pixels", type=int, default=480 * 832)
    parser.add_argument("--mllm_src_frames", type=int, default=0)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cfg_scale", type=float, default=1.5)
    parser.add_argument("--image_cfg_scale", type=float, default=1.0)
    parser.add_argument("--fallback_to_two_pass_cfg", action="store_true")
    parser.add_argument(
        "--dynamic_cfg_by_bench_tag",
        action="store_true",
        help="Per-case route (cfg_scale, image_cfg_scale) from "
             "BENCH_TAG_CFG_BUCKETS keyed on rec['edit_type'] (the EditVerse "
             "bench tag — NOT the agent's plan.subtask). When the routed "
             "image_cfg == 1.0, fallback_to_two_pass_cfg is auto-enabled. "
             "Overrides --cfg_scale, --image_cfg_scale, --fallback_to_two_pass_cfg.",
    )
    parser.add_argument("--negative_prompt", default="")
    parser.add_argument(
        "--use_mask_overlay",
        action="store_true",
        help="Pass the agent's object_mask overlay as an extra ref_image. "
             "Default OFF: stage1 checkpoints were not trained with masked-image "
             "guidance so adding the overlay ref pollutes conditioning. Only enable "
             "this on stage2 checkpoints that were trained with ref_mask composite.",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    wanted = {x.strip() for x in args.bench_ids.split(",") if x.strip()} or None
    records = _read_records(args.records_jsonl, wanted)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Loaded {len(records)} agent records")
    print(f"Video ckpt: {args.ckpt}")
    if args.dynamic_cfg_by_bench_tag:
        print("Dynamic CFG routing by bench tag is ON — per-case (cfg_scale, image_cfg_scale, 2-pass) from BENCH_TAG_CFG_BUCKETS:")
        for tag, (tcfg, icfg) in BENCH_TAG_CFG_BUCKETS.items():
            two_pass = abs(icfg - 1.0) < 1e-9
            print(f"  {tag:>20s}  tcfg={tcfg:.2f}  icfg={icfg:.2f}  2pass={two_pass}")
    pipe = load_v2_pipeline(args.ckpt, device=args.device, ref_max_items=8, paths=default_paths())

    outputs = []
    for idx, rec in enumerate(records, start=1):
        print(f"[{idx}/{len(records)}] {rec['bench_id']}: {rec.get('final_payload', {}).get('subtask')}")
        outputs.append(run_one(pipe, rec, args.out_dir, args))

    summary_path = args.out_dir / "video_editor_outputs.json"
    summary_path.write_text(json.dumps(outputs, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
