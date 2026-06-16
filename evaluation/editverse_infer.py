"""EditVerse benchmark inference: run Aurora on all benchmark samples using multi-GPU.

Reads EditVerseBench_3.json (120 samples, 12 edit types).
Each GPU worker loads the pipeline once and processes its shard.
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

PER_TYPE_NEGATIVE_PROMPTS = {
    "add object":
        "blurry, low quality, distorted, pasted-on sticker, 2D cutout, flat lighting, no shadows, floating object, wrong scale, unnatural, artifacts, watermark, text",
    "ID insertion":
        "blurry, low quality, distorted, pasted-on sticker, 2D cutout, flat lighting, no shadows, floating object, wrong scale, wrong identity, static insert, artifacts, watermark, text",
    "add effect":
        "blurry, low quality, distorted, static, frozen, non-animated, flat overlay, uniform overlay, pasted-on, global color filter, over-saturated, artifacts, watermark, text",
    "remove object":
        "blurry, low quality, distorted, residual ghost, outline remaining, incomplete removal, incorrect inpainting, artifacts, watermark, text",
    "change background":
        "blurry, low quality, distorted, pasted-on subject, wrong lighting match, halo around subject, unchanged background, artifacts, watermark, text",
    "change weather":
        "blurry, low quality, distorted, unchanged, inconsistent lighting, global color filter, artifacts, watermark, text",
    "change material":
        "blurry, low quality, distorted, flat color, plastic-looking, unrealistic texture, painted-on, unchanged, global color shift, artifacts, watermark, text",
    "change color":
        "blurry, low quality, distorted, color leak onto background, shape changed, flat color, global color filter, artifacts, watermark, text",
    "change object":
        "blurry, low quality, distorted, pasted-on sticker, flat lighting, no shadows, wrong scale, unnatural, artifacts, watermark, text",
    "reasoning":
        "blurry, low quality, distorted, unchanged, ignored instruction, wrong target, global color filter, over-saturated, partial edit, artifacts, watermark, text",
    "combined tasks":
        "blurry, low quality, distorted, incomplete, missing step, only one edit performed, unchanged, partial edit, artifacts, watermark, text",
    "stylization":
        "blurry, low quality, distorted, unchanged, inconsistent style across frames, halftone, artifacts, watermark, text",
}


WORKER_SCRIPT = r'''
"""Single-GPU EditVerse inference worker."""
import fcntl, json, os, sys, time
from pathlib import Path
from PIL import Image

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

TIME_DIV_FACTOR = 4
TIME_DIV_REMAINDER = 1


def _align_frames(n, factor=TIME_DIV_FACTOR, remainder=TIME_DIV_REMAINDER):
    """Round down to nearest valid frame count (4k+1)."""
    while n > 1 and n % factor != remainder:
        n -= 1
    return n


def _get_source_fps(video_path):
    try:
        reader = imageio.get_reader(video_path)
        fps = reader.get_meta_data().get("fps", 15)
        reader.close()
        return float(fps)
    except Exception:
        return 15.0


def run_one(pipe, case, out_dir, *, seed=42, cfg_scale=6.0, image_cfg_scale=2.5,
            fallback_to_two_pass_cfg=False,
            max_frames=None, max_pixels=480 * 832, baseline_dir="",
            mllm_src_frames=0, num_frames_override=0, save_frames=0,
            negative_prompt="", negative_prompt_by_type=None):
    idx = case["idx"]
    bench_id = case["bench_id"]
    prompt = case["prompt"]
    video_path = case["video_path"]
    ref_image_path = case.get("ref_image_path")

    # Per-edit-type override beats the uniform negative prompt when provided.
    effective_neg = negative_prompt
    if negative_prompt_by_type:
        et = case.get("edit_type", "")
        if et in negative_prompt_by_type:
            effective_neg = negative_prompt_by_type[et]

    # Output layout matches official EditVerse: {bench_id}/generate.mp4 + video1.mp4
    case_dir = out_dir / bench_id
    case_dir.mkdir(parents=True, exist_ok=True)
    out_path = case_dir / "generate.mp4"
    if out_path.exists() and out_path.stat().st_size > 0:
        print(f"  [{bench_id}] RESUME: already exists, skipping")
        return str(out_path)

    if not os.path.exists(video_path):
        print(f"  [{bench_id}] SKIP: source not found: {video_path}")
        return None

    src_fps = _get_source_fps(video_path)

    # Read target resolution and frame count from baseline video (for fair comparison).
    baseline_video = os.path.join(baseline_dir, bench_id, "video1.mp4") if baseline_dir else ""
    if baseline_video and os.path.exists(baseline_video):
        bl_reader = imageio.get_reader(baseline_video)
        bl_frame_count = bl_reader.count_frames()
        bl_frame = bl_reader.get_data(0)
        bl_fps = bl_reader.get_meta_data().get("fps", src_fps)
        bl_reader.close()
        target_height, target_width = bl_frame.shape[0], bl_frame.shape[1]
        target_frames = _align_frames(bl_frame_count)
        target_fps = float(bl_fps)
    else:
        raw_reader = imageio.get_reader(video_path)
        raw_frame = raw_reader.get_data(0)
        raw_reader.close()
        target_height, target_width = raw_frame.shape[0], raw_frame.shape[1]
        target_frames = None
        bl_frame_count = 0
        target_fps = src_fps

    # Determine how many frames the DiT should generate. If --num_frames is
    # passed (>0), use it (typically 81 to match training distribution);
    # otherwise fall back to the baseline-derived target_frames for backwards
    # compatibility.
    if num_frames_override and num_frames_override > 0:
        load_length = num_frames_override
    else:
        load_length = target_frames if target_frames else (max_frames if max_frames else 999)

    # Load source at max_pixels training resolution for inference.
    src_video = VideoData(video_path, length=load_length, max_pixels=max_pixels)
    gen_height, gen_width = src_video.height, src_video.width
    raw_frames = len(src_video)
    num_frames = _align_frames(raw_frames)

    # Determine how many frames to actually save to disk. When save_frames>0
    # we truncate the generated video to that many frames (useful for matching
    # external baseline video lengths, e.g. EditVerse's 64-frame video1.mp4).
    if save_frames and save_frames > 0:
        save_count = min(save_frames, num_frames)
    elif bl_frame_count > 0:
        save_count = min(bl_frame_count, num_frames)
    else:
        save_count = num_frames

    print(f"  [{bench_id}] {case['edit_type']}: target {target_width}x{target_height} "
          f"{target_frames}f -> gen {gen_width}x{gen_height} {num_frames}f -> save {save_count}f "
          f"@ {target_fps:.1f}fps")
    print(f"  [{bench_id}] Prompt: {prompt[:100]}...")

    if num_frames != raw_frames:
        src_video = VideoData(video_path, length=num_frames, max_pixels=max_pixels)

    # Load reference image if present (for ID insertion tasks)
    ref_image = None
    if ref_image_path and os.path.exists(ref_image_path):
        ref_image = [Image.open(ref_image_path).convert("RGB")]
        print(f"  [{bench_id}] Ref image: {ref_image_path}")

    try:
        # Optionally subsample src_video for MLLM (ablation).
        # source_input (VAE conditioning) always uses the full video.
        mllm_src = src_video
        if mllm_src_frames > 0 and src_video is not None and len(src_video) > mllm_src_frames:
            import numpy as np
            indices = np.linspace(0, len(src_video) - 1, mllm_src_frames, dtype=int)
            mllm_src = [src_video[i] for i in indices]
            print(f"  [{bench_id}] MLLM src subsampled: {len(src_video)} -> {len(mllm_src)} frames")

        if effective_neg:
            print(f"  [{bench_id}] neg_prompt: {effective_neg[:140]}")
        video = pipe(
            prompt=prompt,
            negative_prompt=effective_neg,
            source_input=src_video,
            src_video=mllm_src,
            ref_image=ref_image,
            height=gen_height,
            width=gen_width,
            num_frames=num_frames,
            seed=seed,
            cfg_scale=cfg_scale,
            image_cfg_scale=image_cfg_scale,
            fallback_to_two_pass_cfg=fallback_to_two_pass_cfg,
            tiled=True,
        )
        # Truncate to save_count (match baseline length) before resize/save.
        video = list(video)
        if save_count < len(video):
            video = video[:save_count]
        # Resize output to baseline resolution for fair comparison.
        video = [frame.resize((target_width, target_height), Image.LANCZOS)
                 for frame in video]
        save_video(video, str(out_path), fps=target_fps, quality=5)

        # Copy baseline video1.mp4 as scoring reference (matching resolution).
        video1_path = case_dir / "video1.mp4"
        if not video1_path.exists():
            if baseline_video and os.path.exists(baseline_video):
                import shutil
                shutil.copy2(baseline_video, video1_path)
            else:
                reader = imageio.get_reader(video_path)
                src_pil = []
                for fi, frame_rgb in enumerate(reader):
                    if fi >= num_frames:
                        break
                    src_pil.append(Image.fromarray(frame_rgb))
                reader.close()
                save_video(src_pil, str(video1_path), fps=src_fps, quality=5)

        print(f"  [{bench_id}] Saved: {out_path} ({save_count}f @ {target_fps:.1f}fps, {target_width}x{target_height})")
        return str(out_path)
    except Exception as exc:
        print(f"  [{bench_id}] ERROR: {exc}")
        import traceback
        traceback.print_exc()
        return None


def main():
    ckpt = os.environ["EV_CKPT"]
    device = os.environ["EV_DEVICE"]
    out_dir = Path(os.environ["EV_OUT_DIR"])
    cases = json.loads(os.environ["EV_CASES"])
    seed = int(os.environ.get("EV_SEED", "42"))
    cfg_scale = float(os.environ.get("EV_CFG_SCALE", "6.0"))
    image_cfg_scale = float(os.environ.get("EV_IMAGE_CFG_SCALE", "2.5"))
    fallback_to_two_pass_cfg = os.environ.get("EV_FALLBACK_TO_TWO_PASS_CFG", "0") == "1"
    max_pixels = int(os.environ.get("EV_MAX_PIXELS", str(480 * 832)))
    gpu_id = os.environ.get("EV_GPU_ID", "?")
    baseline_dir = os.environ.get("EV_BASELINE_DIR", "")
    mllm_src_frames = int(os.environ.get("EV_MLLM_SRC_FRAMES", "0"))
    num_frames_override = int(os.environ.get("EV_NUM_FRAMES", "0"))
    save_frames = int(os.environ.get("EV_SAVE_FRAMES", "0"))
    negative_prompt = os.environ.get("EV_NEGATIVE_PROMPT", "")
    neg_by_type_json = os.environ.get("EV_NEG_BY_TYPE_JSON", "")
    negative_prompt_by_type = json.loads(neg_by_type_json) if neg_by_type_json else None

    out_dir.mkdir(parents=True, exist_ok=True)

    lock_path = os.environ.get("EV_LOCK", str(out_dir / ".pipeline_load.lock"))
    print(f"[GPU {gpu_id}] Loading pipeline on {device} (lock: {lock_path})...")
    pipe = _load_with_lock(ckpt, device=device, ref_max_items=8, lock_path=lock_path)
    print(f"[GPU {gpu_id}] Pipeline loaded, running {len(cases)} cases"
          + (f" (MLLM src subsampled to {mllm_src_frames}f)" if mllm_src_frames > 0 else ""))

    results = []
    for case in cases:
        out_path = run_one(pipe, case, out_dir, seed=seed,
                           cfg_scale=cfg_scale, image_cfg_scale=image_cfg_scale,
                           fallback_to_two_pass_cfg=fallback_to_two_pass_cfg,
                           max_pixels=max_pixels, baseline_dir=baseline_dir,
                           mllm_src_frames=mllm_src_frames,
                           num_frames_override=num_frames_override,
                           save_frames=save_frames,
                           negative_prompt=negative_prompt,
                           negative_prompt_by_type=negative_prompt_by_type)
        results.append({
            "bench_id": case["bench_id"],
            "edit_type": case["edit_type"],
            "prompt": case["prompt"],
            "video_path": case["video_path"],
            "edited_path": out_path or "",
            "success": out_path is not None,
        })

    result_file = out_dir / f".results_gpu{gpu_id}.json"
    with open(result_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[GPU {gpu_id}] Done: {sum(1 for r in results if r['success'])}/{len(results)} succeeded")


if __name__ == "__main__":
    main()
'''


def load_benchmark(json_path: str, bench_dir: str, source_dir: str | None = None) -> list[dict]:
    """Load EditVerseBench_3.json and resolve absolute paths.

    Args:
        source_dir: If set, use {source_dir}/{bench_id}/video1.mp4 as source
                    instead of source_videos/. This matches the preprocessed
                    input used by baselines (lower res, 64 frames, 24fps).
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    cases = []
    for bench_id, entry in sorted(data.items(), key=lambda x: int(x[0])):
        if source_dir:
            video_abs = os.path.join(source_dir, bench_id, "video1.mp4")
        else:
            video_rel = entry.get("<video1>", "")
            video_abs = os.path.join(bench_dir, "source_videos", os.path.basename(video_rel))

        ref_image_abs = None
        image_rel = entry.get("<image1>", "")
        if image_rel:
            ref_image_abs = os.path.join(bench_dir, "source_images", os.path.basename(image_rel))

        prompt = entry.get("<text>", "")
        prompt = prompt.replace("<video1>", "").replace("<image1>", "").strip()

        cases.append({
            "idx": len(cases),
            "bench_id": bench_id,
            "edit_type": entry.get("type", "unknown"),
            "prompt": prompt,
            "video_path": video_abs,
            "ref_image_path": ref_image_abs,
            "direction": entry.get("direction", "horizontal"),
            "target_prompt": entry.get("target_prompt", ""),
            "source_prompt": entry.get("source_prompt", ""),
        })
    return cases


def main():
    parser = argparse.ArgumentParser(description="EditVerse multi-GPU inference")
    parser.add_argument("--ckpt", default=None,
                        help="Editor checkpoint path. Omit to auto-download "
                             "yeates/aurora-weights:aurora_editor.safetensors "
                             "(public; no token required).")
    parser.add_argument("--bench_dir", default=os.environ.get(
        "AURORA_BENCH_DIR", str(REPO_ROOT / "benchmarks" / "editverse")
    ), help="EditVerse benchmark data directory")
    parser.add_argument("--out_dir", default=None,
                        help="Output directory (default: bench_dir/results/<ckpt_stem>)")
    parser.add_argument("--num_gpus", type=int, default=8)
    parser.add_argument("--gpu_ids", default=None,
                        help="Comma-separated physical GPU IDs (e.g. '3,4,5'). "
                             "Overrides --num_gpus. Useful for running multiple "
                             "inference jobs on disjoint GPU subsets.")
    parser.add_argument("--max_pixels", type=int, default=480 * 832,
                        help="Max pixel budget (stage1=399360, stage2=921600)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cfg_scale", type=float, default=1.5)
    parser.add_argument("--image_cfg_scale", type=float, default=1.0)
    parser.add_argument(
        "--fallback_to_two_pass_cfg",
        action="store_true",
        help="When image_cfg_scale=1.0, skip the unconditional branch and use equivalent 2-pass CFG.",
    )
    parser.add_argument("--source_dir", default=None,
                        help="Override source video dir ({source_dir}/{bench_id}/video1.mp4). "
                             "Default: use source_videos/ from bench_dir")
    parser.add_argument("--baseline_dir", default=None,
                        help="Baseline dir for target output resolution and video1.mp4 reference. "
                             "Default: bench_dir/baselines/EditVerse")
    parser.add_argument("--mllm_src_frames", type=int, default=0,
                        help="Subsample src_video to N evenly-spaced frames for MLLM only "
                             "(0=use all frames). VAE source_input is unaffected.")
    parser.add_argument("--num_frames", type=int, default=81,
                        help="Force DiT to generate this many frames (must be 4k+1). "
                             "Default 81 matches stage1 training num_frames. "
                             "Set to 0 to fall back to baseline video1.mp4-derived length.")
    parser.add_argument("--save_frames", type=int, default=64,
                        help="Truncate generated video to this many frames before saving. "
                             "Default 64 matches EditVerse baselines' video1.mp4 length "
                             "so scoring pipelines see same temporal extent. "
                             "Set to 0 to save all generated frames (or baseline length).")
    parser.add_argument("--bench_ids", default=None,
                        help="Comma-separated bench_ids to run (default=all).")
    parser.add_argument("--negative_prompt", default="",
                        help="Uniform negative prompt text. Training never saw "
                             "non-empty negatives (only empty-text dropout), so "
                             "passing non-empty text is inference-time extrapolation. "
                             "Under 2-pass CFG this directly steers the negative "
                             "branch away from the listed attributes. Ignored "
                             "when --per_type_negative_prompts is set.")
    parser.add_argument("--per_type_negative_prompts", action="store_true",
                        help="Use the built-in PER_TYPE_NEGATIVE_PROMPTS mapping "
                             "(edit_type -> negative prompt). Avoids cross-task "
                             "conflicts, e.g. keeping 'cartoonish' out of stylization.")
    parser.add_argument("--neg_by_type_json", default="",
                        help="Path to a JSON file {edit_type: negative_prompt}. "
                             "Overrides --per_type_negative_prompts when set.")
    parser.add_argument("--tag", default="",
                        help="Extra suffix appended to the auto-generated out_dir.")
    args = parser.parse_args()

    sys.path.insert(0, str(REPO_ROOT))
    from evaluation.model_download import resolve_editor_ckpt
    args.ckpt = resolve_editor_ckpt(args.ckpt)

    bench_dir = Path(args.bench_dir)
    json_path = bench_dir / "EditVerseBench_3.json"
    if not json_path.exists():
        raise FileNotFoundError(f"Missing benchmark JSON: {json_path}")

    cases = load_benchmark(str(json_path), str(bench_dir), source_dir=args.source_dir)
    if args.bench_ids:
        wanted = set(s.strip() for s in args.bench_ids.split(",") if s.strip())
        cases = [c for c in cases if c["bench_id"] in wanted]
    print(f"Loaded {len(cases)} benchmark cases")

    if args.gpu_ids:
        physical_gpus = [int(g) for g in args.gpu_ids.split(",")]
    else:
        physical_gpus = list(range(args.num_gpus))
    num_workers = len(physical_gpus)

    if args.out_dir:
        out_dir = Path(args.out_dir)
    else:
        ckpt_stem = Path(args.ckpt).stem
        cfg_tag = f"t{args.cfg_scale}_i{args.image_cfg_scale}"
        if args.fallback_to_two_pass_cfg:
            cfg_tag += "_2pass"
        if args.num_frames:
            cfg_tag += f"_nf{args.num_frames}"
        if args.save_frames:
            cfg_tag += f"_sv{args.save_frames}"
        if args.neg_by_type_json or args.per_type_negative_prompts:
            cfg_tag += "_negpt"
        elif args.negative_prompt:
            cfg_tag += "_neg"
        if args.tag:
            cfg_tag += f"_{args.tag}"
        out_dir = bench_dir / "results" / f"{ckpt_stem}_{cfg_tag}"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {out_dir}")
    print(f"Using GPUs: {physical_gpus}")

    worker_path = out_dir / "_worker.py"
    with open(worker_path, "w") as f:
        f.write(WORKER_SCRIPT)

    gpu_cases = [[] for _ in range(num_workers)]
    for i, case in enumerate(cases):
        gpu_cases[i % num_workers].append(case)

    repo_root = str(REPO_ROOT)
    triton_cache = os.environ.get("TRITON_CACHE_DIR") or str(Path(tempfile.gettempdir()) / ".triton")

    procs = []
    for worker_idx, phys_gpu in enumerate(physical_gpus):
        if not gpu_cases[worker_idx]:
            continue
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(phys_gpu)
        env["AURORA_ROOT"] = repo_root
        env["PYTHONPATH"] = repo_root
        env["EV_CKPT"] = args.ckpt
        env["EV_DEVICE"] = "cuda:0"
        env["EV_OUT_DIR"] = str(out_dir)
        env["EV_CASES"] = json.dumps(gpu_cases[worker_idx])
        env["EV_SEED"] = str(args.seed)
        env["EV_CFG_SCALE"] = str(args.cfg_scale)
        env["EV_IMAGE_CFG_SCALE"] = str(args.image_cfg_scale)
        env["EV_FALLBACK_TO_TWO_PASS_CFG"] = "1" if args.fallback_to_two_pass_cfg else "0"
        env["EV_MAX_PIXELS"] = str(args.max_pixels)
        env["EV_GPU_ID"] = str(phys_gpu)
        env["EV_BASELINE_DIR"] = args.baseline_dir or str(bench_dir / "baselines" / "EditVerse")
        env["EV_MLLM_SRC_FRAMES"] = str(args.mllm_src_frames)
        env["EV_NUM_FRAMES"] = str(args.num_frames)
        env["EV_SAVE_FRAMES"] = str(args.save_frames)
        env["EV_NEGATIVE_PROMPT"] = args.negative_prompt
        if args.neg_by_type_json:
            with open(args.neg_by_type_json, "r") as _fh:
                _neg_map = json.load(_fh)
        elif args.per_type_negative_prompts:
            _neg_map = PER_TYPE_NEGATIVE_PROMPTS
        else:
            _neg_map = None
        env["EV_NEG_BY_TYPE_JSON"] = json.dumps(_neg_map) if _neg_map else ""
        env["TRITON_CACHE_DIR"] = triton_cache

        log_path = out_dir / f"log_gpu{phys_gpu}.txt"
        print(f"Launching GPU {phys_gpu}: {len(gpu_cases[worker_idx])} cases")
        proc = subprocess.Popen(
            [sys.executable, str(worker_path)],
            env=env,
            stdout=open(log_path, "w"),
            stderr=subprocess.STDOUT,
        )
        procs.append((phys_gpu, proc))

    print(f"\nAll {len(procs)} workers launched. Waiting...")
    for phys_gpu, proc in procs:
        proc.wait()
        print(f"  GPU {phys_gpu}: exit code {proc.returncode}")

    all_results = []
    for phys_gpu in physical_gpus:
        result_file = out_dir / f".results_gpu{phys_gpu}.json"
        if result_file.exists():
            with open(result_file) as f:
                all_results.extend(json.load(f))

    all_results.sort(key=lambda r: int(r["bench_id"]))

    results_json = out_dir / "inference_results.json"
    with open(results_json, "w") as f:
        json.dump(all_results, f, indent=2)

    success_count = sum(1 for r in all_results if r["success"])
    print(f"\nInference complete: {success_count}/{len(all_results)} succeeded")
    print(f"Results: {results_json}")


if __name__ == "__main__":
    main()
