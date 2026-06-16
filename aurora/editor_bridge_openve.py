"""OpenVE-shaped bridge from Aurora agent records to the Aurora video editor.

Mirrors evaluation/openve_infer.py mechanics (temporal_resize 81f,
output fps rescale, flat {idx:04d}.mp4 layout, benchmark_results.csv emit)
but reads its prompts from Aurora agent records (final_payload.refined_text_instruction)
instead of the raw OpenVE CSV. Mask overlay and search image are honoured the
same way as the EditVerse bridge but default OFF for benchmarking.
"""

from __future__ import annotations

import argparse
import csv
import fcntl
import json
import os
import sys
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
DEFAULT_OUT_DIR = OUTPUT_DIR / "openve" / "results" / "agent_openve_default"
DEFAULT_BENCH_ROOT = BENCH_DIR / "openve"

TIME_DIV_FACTOR = 4
TIME_DIV_REMAINDER = 1


def _align_frames(n: int) -> int:
    while n > 1 and n % TIME_DIV_FACTOR != TIME_DIV_REMAINDER:
        n -= 1
    return n


def _read_records(paths: list[Path], shard_index: int, num_shards: int) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in paths:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                rec = json.loads(line)
                if "error" in rec:
                    continue
                records.append(rec)

    def _idx(r: dict[str, Any]) -> int:
        try:
            return int(r["bench_id"])
        except (KeyError, ValueError, TypeError):
            return -1

    records.sort(key=_idx)
    if num_shards > 1:
        records = [r for i, r in enumerate(records) if i % num_shards == shard_index]
    return records


def _source_fps(video_path: str) -> float:
    try:
        reader = imageio.get_reader(video_path)
        fps = float(reader.get_meta_data().get("fps", 24))
        reader.close()
        return fps
    except Exception:
        return 24.0


def _temporal_resample_indices(raw_frames: int, target_frames: int) -> list[int]:
    if raw_frames <= 0 or target_frames <= 0:
        return []
    if raw_frames == 1:
        return [0] * target_frames
    idx = np.linspace(0, raw_frames - 1, target_frames).round().astype(int)
    return [int(i) for i in idx]


class _SubsampledVideo:
    """VideoData-compatible wrapper that picks specific frame indices."""

    def __init__(self, base_video, indices):
        self._base = base_video
        self._indices = list(indices)
        self.height = base_video.height
        self.width = base_video.width

    def __len__(self):
        return len(self._indices)

    def __getitem__(self, i):
        return self._base[self._indices[i]]


def _load_ref_images(
    rec: dict[str, Any],
    use_mask_overlay: bool,
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


def run_one(pipe, rec: dict[str, Any], out_dir: Path, args: argparse.Namespace) -> tuple[str | None, dict[str, Any]]:
    bench_id = str(rec["bench_id"])
    try:
        idx = int(bench_id)
    except ValueError:
        idx = -1
    payload = rec.get("final_payload", {})
    prompt = payload.get("refined_text_instruction") or rec.get("prompt", "")
    raw_prompt = rec.get("prompt", "")
    edit_type = rec.get("edit_type", "")
    video_path = rec["video_path"]

    out_path = out_dir / f"{idx:04d}.mp4"
    info: dict[str, Any] = {
        "bench_id": bench_id,
        "idx": idx,
        "edited_type": edit_type,
        "raw_prompt": raw_prompt,
        "refined_prompt": prompt,
        "original_video": rec.get("original_video_rel") or os.path.relpath(video_path, args.bench_root),
        "edited_result_path": "",
        "success": False,
    }

    if out_path.exists() and out_path.stat().st_size > 0 and not args.overwrite:
        info["edited_result_path"] = str(out_path)
        info["success"] = True
        return str(out_path), info

    if not os.path.exists(video_path):
        return None, info

    src_fps = _source_fps(video_path)

    if args.frame_sampling_mode == "temporal_resize" and args.num_frames > 0:
        src_full = VideoData(video_path, length=999, max_pixels=args.max_pixels)
        height, width = src_full.height, src_full.width
        raw_frames = len(src_full)
        num_frames = args.num_frames
        indices = _temporal_resample_indices(raw_frames, num_frames)
        src_video = _SubsampledVideo(src_full, indices)
        output_fps = src_fps * num_frames / max(raw_frames, 1) if raw_frames > 0 else src_fps
    else:
        load_length = args.num_frames if args.num_frames > 0 else 999
        src_video = VideoData(video_path, length=load_length, max_pixels=args.max_pixels)
        height, width = src_video.height, src_video.width
        raw_frames = len(src_video)
        num_frames = _align_frames(raw_frames)
        if num_frames != raw_frames:
            src_video = VideoData(video_path, length=num_frames, max_pixels=args.max_pixels)
        output_fps = src_fps

    mllm_src = src_video
    if args.mllm_src_frames > 0 and len(src_video) > args.mllm_src_frames:
        sub_idx = np.linspace(0, len(src_video) - 1, args.mllm_src_frames, dtype=int)
        mllm_src = [src_video[i] for i in sub_idx]

    source_frame = src_video[0] if args.use_mask_overlay else None
    ref_images, ref_paths = _load_ref_images(
        rec,
        use_mask_overlay=args.use_mask_overlay,
        source_frame=source_frame,
    )

    print(f"  [{bench_id}] {edit_type}: {height}x{width} {raw_frames}->{num_frames}f @ src={src_fps:.1f} -> out={output_fps:.2f}fps")
    print(f"  [{bench_id}] refined: {prompt[:120]}")

    video = pipe(
        prompt=prompt,
        negative_prompt=args.negative_prompt,
        source_input=src_video,
        src_video=mllm_src,
        ref_image=ref_images,
        height=height,
        width=width,
        num_frames=num_frames,
        seed=args.seed,
        cfg_scale=args.cfg_scale,
        image_cfg_scale=args.image_cfg_scale,
        fallback_to_two_pass_cfg=args.fallback_to_two_pass_cfg,
        num_inference_steps=args.num_inference_steps,
        tiled=True,
    )
    save_video(list(video), str(out_path), fps=output_fps, quality=5)

    info["edited_result_path"] = str(out_path)
    info["success"] = True
    info["output_fps"] = float(output_fps)
    info["height"] = int(height)
    info["width"] = int(width)
    info["num_frames"] = int(num_frames)
    info["ref_image_paths"] = ref_paths
    return str(out_path), info


def _write_benchmark_results_csv(out_dir: Path) -> Path:
    """Merge completed shard summaries into the OpenVE scoring CSV contract."""
    result_csv = out_dir / "benchmark_results.csv"
    lock_path = out_dir / ".benchmark_results_csv.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("w") as lf:
        fcntl.flock(lf, fcntl.LOCK_EX)
        merged: dict[tuple[int, str], dict[str, Any]] = {}
        for shard_path in sorted(out_dir.glob(".results_shard*.json")):
            try:
                infos = json.loads(shard_path.read_text(encoding="utf-8"))
            except Exception as exc:
                print(f"WARN: could not read {shard_path}: {exc}")
                continue
            for info in infos:
                key = (int(info.get("idx", -1)), str(info.get("bench_id", "")))
                merged[key] = info

        rows = sorted(merged.values(), key=lambda r: (int(r.get("idx", -1)), str(r.get("bench_id", ""))))
        tmp_csv = out_dir / f".benchmark_results.csv.tmp.{os.getpid()}"
        with tmp_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["edited_type", "prompt", "original_video", "edited_result_path"],
            )
            writer.writeheader()
            for info in rows:
                if not info.get("success"):
                    continue
                writer.writerow({
                    "edited_type": info.get("edited_type", ""),
                    "prompt": info.get("refined_prompt") or info.get("raw_prompt", ""),
                    "original_video": info.get("original_video", ""),
                    "edited_result_path": info.get("edited_result_path", ""),
                })
        os.replace(tmp_csv, result_csv)
    return result_csv


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--records_jsonl", type=Path, action="append", required=True,
                        help="Agent records JSONL. Pass multiple times to merge shards.")
    parser.add_argument("--ckpt", type=Path, default=DEFAULT_CKPT)
    parser.add_argument("--out_dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--bench_root", default=str(DEFAULT_BENCH_ROOT))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num_frames", type=int, default=81)
    parser.add_argument("--frame_sampling_mode", default="temporal_resize",
                        choices=["temporal_resize", "truncate"])
    parser.add_argument("--max_pixels", type=int, default=921600)
    parser.add_argument("--mllm_src_frames", type=int, default=0)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cfg_scale", type=float, default=2.0)
    parser.add_argument("--image_cfg_scale", type=float, default=1.0)
    parser.add_argument("--fallback_to_two_pass_cfg", action="store_true")
    parser.add_argument("--negative_prompt", default="")
    parser.add_argument("--use_mask_overlay", action="store_true",
                        help="Pass agent's mask overlay through ref_image (mask ON setting).")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--shard_index", type=int, default=0)
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--lock_path", default="/tmp/openve_pipeline.lock",
                        help="File lock path used to serialize pipeline loading.")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    records = _read_records(args.records_jsonl, args.shard_index, args.num_shards)
    print(f"[shard {args.shard_index}/{args.num_shards}] {len(records)} records")
    print(f"Video ckpt: {args.ckpt}")

    Path(args.lock_path).parent.mkdir(parents=True, exist_ok=True)
    with open(args.lock_path, "w") as lf:
        fcntl.flock(lf, fcntl.LOCK_EX)
        pipe = load_v2_pipeline(args.ckpt, device=args.device, ref_max_items=8, paths=default_paths())

    infos: list[dict[str, Any]] = []
    for i, rec in enumerate(records, start=1):
        print(f"[{i}/{len(records)}] {rec.get('bench_id')} {rec.get('edit_type')}")
        try:
            _, info = run_one(pipe, rec, args.out_dir, args)
        except Exception as exc:
            import traceback
            print(f"  ERROR: {exc}")
            traceback.print_exc()
            info = {
                "bench_id": rec.get("bench_id"),
                "idx": int(rec.get("bench_id", -1)) if str(rec.get("bench_id", "")).isdigit() else -1,
                "edited_type": rec.get("edit_type", ""),
                "raw_prompt": rec.get("prompt", ""),
                "refined_prompt": rec.get("final_payload", {}).get("refined_text_instruction", ""),
                "original_video": rec.get("original_video_rel", ""),
                "edited_result_path": "",
                "success": False,
                "error": repr(exc),
            }
        infos.append(info)

    shard_summary = args.out_dir / f".results_shard{args.shard_index}.json"
    shard_summary.write_text(json.dumps(infos, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Shard summary: {shard_summary}")
    result_csv = _write_benchmark_results_csv(args.out_dir)
    print(f"Result CSV: {result_csv}")


if __name__ == "__main__":
    main()
