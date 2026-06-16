"""OpenVE-Bench inference: run Aurora on all 431 benchmark samples using multi-GPU."""

import argparse
import csv
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

WORKER_SCRIPT = r'''
"""Single-GPU OpenVE-Bench inference worker."""
import csv, fcntl, json, os, sys, time
from pathlib import Path

sys.path.insert(0, os.environ["AURORA_ROOT"])
import imageio
from diffsynth import VideoData, save_video
from evaluation.pipeline_loader import default_paths, load_v2_pipeline

PATHS = default_paths()


def _load_with_lock(ckpt, device, ref_max_items, lock_path):
    """Load pipeline while holding a file lock to avoid CPU contention."""
    Path(lock_path).parent.mkdir(parents=True, exist_ok=True)
    with open(lock_path, "w") as lf:
        fcntl.flock(lf, fcntl.LOCK_EX)
        pipe = load_v2_pipeline(ckpt, device=device, ref_max_items=ref_max_items, paths=PATHS)
    return pipe

# WAN model constraint: frames must be 4k+1
TIME_DIV_FACTOR = 4
TIME_DIV_REMAINDER = 1


def _align_frames(n, factor=TIME_DIV_FACTOR, remainder=TIME_DIV_REMAINDER):
    """Round down to nearest valid frame count (4k+1)."""
    while n > 1 and n % factor != remainder:
        n -= 1
    return n


def _get_source_fps(video_path):
    """Read FPS from source video metadata."""
    try:
        reader = imageio.get_reader(video_path)
        fps = reader.get_meta_data().get("fps", 15)
        reader.close()
        return float(fps)
    except Exception:
        return 15.0


class _SubsampledVideo:
    """VideoData-compatible wrapper that picks specific frame indices.

    Exposes .height / .width / __len__ / __getitem__ so it drops in for a
    VideoData object in the pipeline call.
    """

    def __init__(self, base_video, indices):
        self._base = base_video
        self._indices = list(indices)
        self.height = base_video.height
        self.width = base_video.width

    def __len__(self):
        return len(self._indices)

    def __getitem__(self, i):
        return self._base[self._indices[i]]


def _temporal_resample_indices(raw_frames, target_frames):
    """Evenly-spaced frame indices covering the full source temporal range.

    - raw_frames > target_frames: picks ~raw/target spaced indices (downsample)
    - raw_frames < target_frames: linspace produces duplicate indices, yielding
      nearest-neighbor temporal upsample (slow-motion). This is what we want
      so the DiT always sees exactly target_frames with well-defined motion.
    - raw_frames == target_frames: identity.
    """
    import numpy as np
    if raw_frames <= 0 or target_frames <= 0:
        return []
    if raw_frames == 1:
        return [0] * target_frames
    idx = np.linspace(0, raw_frames - 1, target_frames).round().astype(int)
    return [int(i) for i in idx]


def run_one(pipe, row, out_dir, *, seed=42, cfg_scale=6.0, image_cfg_scale=2.5,
            fallback_to_two_pass_cfg=False,
            max_frames=None, max_pixels=480 * 832,
            num_frames_override=0, frame_sampling_mode="temporal_resize"):
    idx = row["idx"]
    prompt = row["prompt"]
    original_video = row["original_video_abs"]

    out_path = out_dir / f"{idx:04d}.mp4"
    if out_path.exists() and out_path.stat().st_size > 0:
        print(f"  [{idx}] RESUME: already exists, skipping")
        return str(out_path)

    if not os.path.exists(original_video):
        print(f"  [{idx}] SKIP: source not found: {original_video}")
        return None

    # Read source FPS before loading through VideoData
    src_fps = _get_source_fps(original_video)

    if frame_sampling_mode == "temporal_resize" and num_frames_override > 0:
        # Load the entire source, then subsample to exactly num_frames_override
        # evenly spaced frames so DiT always sees a training-matched number.
        src_full = VideoData(original_video, length=999, max_pixels=max_pixels)
        height, width = src_full.height, src_full.width
        raw_frames = len(src_full)
        num_frames = num_frames_override  # already 4k+1-valid (e.g. 81)
        indices = _temporal_resample_indices(raw_frames, num_frames)
        src_video = _SubsampledVideo(src_full, indices)
        # Output fps keeps the source's wall-clock duration intact (option B):
        # src duration = raw_frames / src_fps; we want output duration equal,
        # so output_fps = num_frames / (raw_frames / src_fps).
        output_fps = src_fps * num_frames / max(raw_frames, 1) if raw_frames > 0 else src_fps
        mode_tag = f"temporal_resize {raw_frames}->{num_frames}"
    else:
        # Truncate mode: first N frames aligned to 4k+1 (legacy behaviour).
        if num_frames_override and num_frames_override > 0:
            load_length = num_frames_override
        else:
            load_length = max_frames if max_frames else 999
        src_video = VideoData(original_video, length=load_length, max_pixels=max_pixels)
        height, width = src_video.height, src_video.width
        raw_frames = len(src_video)
        num_frames = _align_frames(raw_frames)
        if num_frames != raw_frames:
            src_video = VideoData(original_video, length=num_frames, max_pixels=max_pixels)
        output_fps = src_fps
        mode_tag = f"truncate {raw_frames}f->{num_frames}f"

    print(f"  [{idx}] {row['edited_type']}: {height}x{width}, "
          f"{mode_tag} @ src_fps={src_fps:.1f} -> out_fps={output_fps:.2f}")
    print(f"  [{idx}] Prompt: {prompt[:100]}...")

    try:
        video = pipe(
            prompt=prompt,
            source_input=src_video,
            src_video=src_video,
            ref_image=None,
            height=height,
            width=width,
            num_frames=num_frames,
            seed=seed,
            cfg_scale=cfg_scale,
            image_cfg_scale=image_cfg_scale,
            fallback_to_two_pass_cfg=fallback_to_two_pass_cfg,
            tiled=True,
        )
        out_path = out_dir / f"{idx:04d}.mp4"
        save_video(video, str(out_path), fps=output_fps, quality=5)
        print(f"  [{idx}] Saved: {out_path} ({num_frames}f @ {output_fps:.2f}fps)")
        return str(out_path)
    except Exception as exc:
        print(f"  [{idx}] ERROR: {exc}")
        import traceback
        traceback.print_exc()
        return None


def main():
    ckpt = os.environ["OPENVE_CKPT"]
    device = os.environ["OPENVE_DEVICE"]
    out_dir = Path(os.environ["OPENVE_OUT_DIR"])
    cases = json.loads(os.environ["OPENVE_CASES"])
    seed = int(os.environ.get("OPENVE_SEED", "42"))
    cfg_scale = float(os.environ.get("OPENVE_CFG_SCALE", "6.0"))
    image_cfg_scale = float(os.environ.get("OPENVE_IMAGE_CFG_SCALE", "2.5"))
    fallback_to_two_pass_cfg = os.environ.get("OPENVE_FALLBACK_TO_TWO_PASS_CFG", "0") == "1"
    max_pixels = int(os.environ.get("OPENVE_MAX_PIXELS", str(480 * 832)))
    gpu_id = os.environ.get("OPENVE_GPU_ID", "?")
    num_frames_override = int(os.environ.get("OPENVE_NUM_FRAMES", "0"))
    frame_sampling_mode = os.environ.get("OPENVE_FRAME_SAMPLING_MODE", "temporal_resize")

    out_dir.mkdir(parents=True, exist_ok=True)

    lock_path = os.environ.get("OPENVE_LOCK", str(out_dir / ".pipeline_load.lock"))
    print(f"[GPU {gpu_id}] Loading pipeline on {device} (lock: {lock_path})...")
    pipe = _load_with_lock(ckpt, device=device, ref_max_items=8, lock_path=lock_path)
    print(f"[GPU {gpu_id}] Pipeline loaded, running {len(cases)} cases")

    results = []
    for case in cases:
        out_path = run_one(pipe, case, out_dir, seed=seed,
                           cfg_scale=cfg_scale, image_cfg_scale=image_cfg_scale,
                           fallback_to_two_pass_cfg=fallback_to_two_pass_cfg,
                           max_pixels=max_pixels,
                           num_frames_override=num_frames_override,
                           frame_sampling_mode=frame_sampling_mode)
        results.append({
            "idx": case["idx"],
            "edited_type": case["edited_type"],
            "prompt": case["prompt"],
            "original_video": case["original_video"],
            "edited_result_path": out_path or "",
            "success": out_path is not None,
        })

    result_file = out_dir / f".results_gpu{gpu_id}.json"
    with open(result_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[GPU {gpu_id}] Done: {sum(1 for r in results if r['success'])}/{len(results)} succeeded")


if __name__ == "__main__":
    main()
'''


def load_benchmark_csv(csv_path: str, bench_root: str) -> list[dict]:
    """Load benchmark_videos.csv and resolve absolute video paths."""
    rows = []
    with open(csv_path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader):
            row["idx"] = idx
            row["original_video_abs"] = os.path.join(bench_root, row["original_video"])
            rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser(description="OpenVE-Bench multi-GPU inference")
    parser.add_argument("--ckpt", default=None,
                        help="Editor checkpoint path. Omit to auto-download "
                             "yeates/aurora-weights:aurora_editor.safetensors "
                             "(public; no token required).")
    parser.add_argument("--bench_dir", default=os.environ.get(
        "AURORA_BENCH_DIR", str(REPO_ROOT / "benchmarks" / "openve")
    ), help="OpenVE-Bench data directory")
    parser.add_argument("--out_dir", default=os.path.join(
        os.environ.get("AURORA_OUTPUT_DIR", str(REPO_ROOT / "outputs")), "openve-bench", "results"
    ), help="Output directory for edited videos")
    parser.add_argument("--num_gpus", type=int, default=8)
    parser.add_argument("--max_pixels", type=int, default=480 * 832,
                        help="Max pixel budget for inference (stage1=399360, stage2=921600)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cfg_scale", type=float, default=6.0)
    parser.add_argument("--image_cfg_scale", type=float, default=2.5)
    parser.add_argument(
        "--fallback_to_two_pass_cfg",
        action="store_true",
        help="When image_cfg_scale=1.0, skip the unconditional branch and use equivalent 2-pass CFG.",
    )
    parser.add_argument("--num_frames", type=int, default=81,
                        help="DiT num_frames per case (must be 4k+1). Default 81 matches "
                             "stage1 training. Set 0 to fall back to source-aligned frame count.")
    parser.add_argument("--frame_sampling_mode", default="temporal_resize",
                        choices=["temporal_resize", "truncate"],
                        help="How to map source frames to DiT input. "
                             "'temporal_resize' (default): linspace sampling covers the full "
                             "source duration. Output fps rescaled so output duration = source "
                             "duration. For sources shorter than --num_frames, indices repeat "
                             "(nearest-neighbor upsample / slow-motion). "
                             "'truncate': take first --num_frames source frames, keep source fps "
                             "(legacy behaviour).")
    parser.add_argument("--case_indices", default=None,
                        help="Comma-separated case indices to run (default=all).")
    args = parser.parse_args()

    sys.path.insert(0, str(REPO_ROOT))
    from evaluation.model_download import resolve_editor_ckpt
    args.ckpt = resolve_editor_ckpt(args.ckpt)

    bench_dir = Path(args.bench_dir)
    csv_path = bench_dir / "benchmark_videos.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing benchmark CSV: {csv_path}")

    rows = load_benchmark_csv(str(csv_path), str(bench_dir))
    if args.case_indices:
        wanted = set(int(s.strip()) for s in args.case_indices.split(",") if s.strip())
        rows = [r for r in rows if r["idx"] in wanted]
    print(f"Loaded {len(rows)} benchmark cases")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    worker_path = out_dir / "_worker.py"
    with open(worker_path, "w") as f:
        f.write(WORKER_SCRIPT)

    gpu_cases = [[] for _ in range(args.num_gpus)]
    for i, row in enumerate(rows):
        gpu_cases[i % args.num_gpus].append(row)

    repo_root = str(REPO_ROOT)
    triton_cache = os.environ.get("TRITON_CACHE_DIR") or str(Path(tempfile.gettempdir()) / ".triton")

    procs = []
    for gpu_id in range(args.num_gpus):
        if not gpu_cases[gpu_id]:
            continue
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        env["AURORA_ROOT"] = repo_root
        env["PYTHONPATH"] = repo_root
        env["OPENVE_CKPT"] = args.ckpt
        env["OPENVE_DEVICE"] = "cuda:0"
        env["OPENVE_OUT_DIR"] = str(out_dir)
        env["OPENVE_CASES"] = json.dumps(gpu_cases[gpu_id])
        env["OPENVE_SEED"] = str(args.seed)
        env["OPENVE_CFG_SCALE"] = str(args.cfg_scale)
        env["OPENVE_IMAGE_CFG_SCALE"] = str(args.image_cfg_scale)
        env["OPENVE_FALLBACK_TO_TWO_PASS_CFG"] = "1" if args.fallback_to_two_pass_cfg else "0"
        env["OPENVE_MAX_PIXELS"] = str(args.max_pixels)
        env["OPENVE_GPU_ID"] = str(gpu_id)
        env["OPENVE_NUM_FRAMES"] = str(args.num_frames)
        env["OPENVE_FRAME_SAMPLING_MODE"] = args.frame_sampling_mode
        env["TRITON_CACHE_DIR"] = triton_cache

        log_path = out_dir / f"log_gpu{gpu_id}.txt"
        print(f"Launching GPU {gpu_id}: {len(gpu_cases[gpu_id])} cases")
        proc = subprocess.Popen(
            [sys.executable, str(worker_path)],
            env=env,
            stdout=open(log_path, "w"),
            stderr=subprocess.STDOUT,
        )
        procs.append((gpu_id, proc))

    print(f"\nAll {len(procs)} workers launched. Waiting...")
    for gpu_id, proc in procs:
        proc.wait()
        print(f"  GPU {gpu_id}: exit code {proc.returncode}")

    all_results = []
    for gpu_id in range(args.num_gpus):
        result_file = out_dir / f".results_gpu{gpu_id}.json"
        if result_file.exists():
            with open(result_file) as f:
                all_results.extend(json.load(f))

    all_results.sort(key=lambda r: r["idx"])

    result_csv = out_dir / "benchmark_results.csv"
    with open(result_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["edited_type", "prompt", "original_video", "edited_result_path"])
        writer.writeheader()
        for r in all_results:
            if r["success"]:
                writer.writerow({
                    "edited_type": r["edited_type"],
                    "prompt": r["prompt"],
                    "original_video": r["original_video"],
                    "edited_result_path": r["edited_result_path"],
                })

    success_count = sum(1 for r in all_results if r["success"])
    print(f"\nInference complete: {success_count}/{len(all_results)} succeeded")
    print(f"Result CSV: {result_csv}")


if __name__ == "__main__":
    main()
