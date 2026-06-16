from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


WORKER_SCRIPT = r'''
import fcntl, json, os, sys
from pathlib import Path

sys.path.insert(0, os.environ["AURORA_ROOT"])
import imageio
from PIL import Image
from diffsynth import VideoData, save_video
from evaluation.pipeline_loader import default_paths, load_v2_pipeline

PATHS = default_paths()


def _load_with_lock(ckpt, device, ref_max_items, lock_path):
    Path(lock_path).parent.mkdir(parents=True, exist_ok=True)
    with open(lock_path, "w") as lf:
        fcntl.flock(lf, fcntl.LOCK_EX)
        pipe = load_v2_pipeline(ckpt, device=device, ref_max_items=ref_max_items, paths=PATHS)
    return pipe


TIME_DIV_FACTOR = 4
TIME_DIV_REMAINDER = 1


def _align_frames(n, factor=TIME_DIV_FACTOR, remainder=TIME_DIV_REMAINDER):
    while n > 1 and n % factor != remainder:
        n -= 1
    return n


def _get_source_fps(p):
    try:
        r = imageio.get_reader(p); fps = r.get_meta_data().get("fps", 16); r.close()
        return float(fps)
    except Exception:
        return 16.0


def _maybe_load_ref_image(case, src_video):
    """Build the ref_image list this case will pass into the editor.

    Sources, in order:
      1. case['search_image'] (Serper-retrieved IP reference, from agent records)
      2. compose_masked_image(src_video[0], case['mask_path']) -- only when
         EV_USE_MASK_OVERLAY=1 AND case['mask_path'] exists. Matches training's
         UnifiedDataset._compose_ref_mask_into_ref_image: composites onto the
         *model-input* first frame (VideoData[0], i.e. post max_pixels +
         32-aligned crop_and_resize), NOT the raw mp4 frame. Compositor knobs
         (alpha=0.55, red, 3px contour) and the no-JPEG-roundtrip path also
         match. The agent's pre-baked object_mask_overlay.jpg is intentionally
         ignored because its alpha / color / resize do not match training.
    """
    images, paths = [], []
    p = case.get("search_image")
    if isinstance(p, str) and p and os.path.exists(p):
        images.append(Image.open(p).convert("RGB"))
        paths.append(p)
    if os.environ.get("EV_USE_MASK_OVERLAY", "0") == "1":
        mp = case.get("mask_path")
        if isinstance(mp, str) and mp and os.path.exists(mp) and src_video is not None:
            from diffsynth.utils.mask_overlay import compose_masked_image
            first_frame = src_video[0]
            mask_img = Image.open(mp)
            composite = compose_masked_image(first_frame, mask_img)
            images.append(composite)
            paths.append(f"compose_masked_image(src_video[0], {mp})")
    return (images or None), paths


def run_one(pipe, case, bench_dir, out_dir, *, seed, cfg_scale, image_cfg_scale,
            fallback_to_two_pass_cfg, max_pixels, num_frames_override, save_frames):
    case_id = case["case_id"]
    prompt = case["prompt"]
    src_rel = case["src_video"]
    src_abs = os.path.join(bench_dir, src_rel)
    # Agent-record asset paths (search_image / mask_path) are stored bench-relative;
    # resolve to absolute against bench_dir, mirroring src_video.
    for _k in ("search_image", "mask_path"):
        _v = case.get(_k)
        if isinstance(_v, str) and _v and not os.path.isabs(_v):
            case[_k] = os.path.join(bench_dir, _v)

    case_dir = out_dir / case_id
    case_dir.mkdir(parents=True, exist_ok=True)
    out_path = case_dir / "generate.mp4"
    if out_path.exists() and out_path.stat().st_size > 0:
        print(f"  [{case_id}] RESUME: skipping")
        return str(out_path)

    if not os.path.exists(src_abs):
        print(f"  [{case_id}] SKIP: source not found: {src_abs}")
        return None

    src_fps = _get_source_fps(src_abs)
    num_frames = num_frames_override if num_frames_override > 0 else 81
    num_frames = _align_frames(num_frames)

    src_video = VideoData(src_abs, length=num_frames, max_pixels=max_pixels)
    raw_frames = len(src_video)
    num_frames = _align_frames(raw_frames)
    if num_frames != raw_frames:
        src_video = VideoData(src_abs, length=num_frames, max_pixels=max_pixels)

    save_count = save_frames if (save_frames and save_frames > 0) else num_frames
    save_count = min(save_count, num_frames)

    ref_images, ref_paths = _maybe_load_ref_image(case, src_video)

    print(f"  [{case_id}] {case['edit_type']}: {src_video.width}x{src_video.height} "
          f"{num_frames}f -> save {save_count}f @ {src_fps:.1f}fps  refs={len(ref_paths)}")
    print(f"  [{case_id}] Prompt: {prompt[:100]}...")
    if ref_paths:
        print(f"  [{case_id}] Ref image: {ref_paths[0]}")

    try:
        video = pipe(
            prompt=prompt,
            negative_prompt="",
            source_input=src_video,
            src_video=src_video,
            ref_image=ref_images,
            height=src_video.height,
            width=src_video.width,
            num_frames=num_frames,
            seed=seed,
            cfg_scale=cfg_scale,
            image_cfg_scale=image_cfg_scale,
            fallback_to_two_pass_cfg=fallback_to_two_pass_cfg,
            tiled=True,
        )
        video = list(video)
        if save_count < len(video):
            video = video[:save_count]
        save_video(video, str(out_path), fps=src_fps, quality=5)

        # Save source at the model-input dimensions (post max_pixels cap + VAE
        # 32-alignment), at the same length and fps as the generated output, so
        # source.mp4 and generate.mp4 are directly comparable side-by-side.
        src_copy = case_dir / "source.mp4"
        if not src_copy.exists():
            src_frames = [src_video[i] for i in range(min(save_count, len(src_video)))]
            save_video(src_frames, str(src_copy), fps=src_fps, quality=5)

        if ref_paths:
            try:
                Image.open(ref_paths[0]).convert("RGB").save(case_dir / "ref_image.jpg", "JPEG", quality=92)
            except Exception:
                pass

        sidecar = case_dir / "agent_editor_inputs.json"
        sidecar.write_text(json.dumps({
            "case_id": case_id,
            "edit_type": case.get("edit_type", ""),
            "raw_instruction": case.get("raw_instruction", ""),
            "prompt": prompt,
            "agent_subtask": case.get("agent_subtask", ""),
            "ref_image_paths_for_current_editor_api": ref_paths,
            "agent_image_search_query": case.get("agent_image_search_query", ""),
            "src_video": src_rel,
            "max_pixels": max_pixels,
            "num_frames": num_frames,
            "save_frames": save_count,
            "cfg_scale": cfg_scale,
            "image_cfg_scale": image_cfg_scale,
            "fallback_to_two_pass_cfg": bool(fallback_to_two_pass_cfg),
            "seed": seed,
        }, ensure_ascii=False, indent=2))

        print(f"  [{case_id}] Saved: {out_path}")
        return str(out_path)
    except Exception as exc:
        import traceback
        print(f"  [{case_id}] ERROR: {exc}")
        traceback.print_exc()
        return None


def main():
    ckpt = os.environ["EV_CKPT"]
    device = os.environ["EV_DEVICE"]
    bench_dir = os.environ["EV_BENCH_DIR"]
    out_dir = Path(os.environ["EV_OUT_DIR"])
    cases = json.loads(os.environ["EV_CASES"])
    seed = int(os.environ.get("EV_SEED", "42"))
    cfg_scale = float(os.environ.get("EV_CFG_SCALE", "2.0"))
    image_cfg_scale = float(os.environ.get("EV_IMAGE_CFG_SCALE", "1.0"))
    fallback_to_two_pass_cfg = os.environ.get("EV_FALLBACK_TO_TWO_PASS_CFG", "1") == "1"
    max_pixels = int(os.environ.get("EV_MAX_PIXELS", str(480 * 832)))
    num_frames_override = int(os.environ.get("EV_NUM_FRAMES", "81"))
    save_frames = int(os.environ.get("EV_SAVE_FRAMES", "0"))
    gpu_id = os.environ.get("EV_GPU_ID", "?")
    run_tag = os.environ.get("EV_RUN_TAG", "")

    out_dir.mkdir(parents=True, exist_ok=True)
    lock_suffix = f"_{run_tag}" if run_tag else ""
    lock_path = os.environ.get("EV_LOCK", str(out_dir / f".pipeline_load{lock_suffix}.lock"))
    print(f"[GPU {gpu_id}] Loading pipeline on {device} (lock: {lock_path})...")
    pipe = _load_with_lock(ckpt, device=device, ref_max_items=8, lock_path=lock_path)
    print(f"[GPU {gpu_id}] Pipeline loaded, running {len(cases)} cases")

    results = []
    for case in cases:
        out_path = run_one(pipe, case, bench_dir, out_dir, seed=seed,
                           cfg_scale=cfg_scale, image_cfg_scale=image_cfg_scale,
                           fallback_to_two_pass_cfg=fallback_to_two_pass_cfg,
                           max_pixels=max_pixels,
                           num_frames_override=num_frames_override,
                           save_frames=save_frames)
        results.append({
            "case_id": case["case_id"],
            "edit_type": case["edit_type"],
            "prompt": case["prompt"],
            "target_entity": case.get("target_entity", ""),
            "src_video": case["src_video"],
            "edited_path": out_path or "",
            "success": out_path is not None,
        })

    rt_suffix = f"_{run_tag}" if run_tag else ""
    result_file = out_dir / f".results_gpu{gpu_id}{rt_suffix}.json"
    with open(result_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[GPU {gpu_id}] Done: {sum(1 for r in results if r['success'])}/{len(results)} succeeded")


if __name__ == "__main__":
    main()
'''


def load_cases(prompts_jsonl: Path, case_ids: set | None = None) -> list[dict]:
    cases = []
    with prompts_jsonl.open() as f:
        for line in f:
            d = json.loads(line)
            if case_ids and d["case_id"] not in case_ids:
                continue
            cases.append(d)
    return cases


def load_agent_records(records_jsonl: Path) -> dict[str, dict]:
    """Load aurora-style agent_pipeline_records.jsonl, key by bench_id (== case_id).
    Skips error rows. Returns empty dict if path is empty/missing."""
    out: dict[str, dict] = {}
    if not records_jsonl or not Path(records_jsonl).exists():
        return out
    with Path(records_jsonl).open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if "error" in rec:
                continue
            cid = str(rec.get("bench_id", ""))
            if cid:
                out[cid] = rec
    return out


def apply_agent_records(cases: list[dict], records: dict[str, dict],
                        *, require_record: bool, use_search_image: bool,
                        use_mask_overlay: bool = False,
                        use_refined_prompt: bool = True) -> list[dict]:
    """For each case, optionally override prompt with refined_text_instruction,
    and attach search_image path so the worker can load it as ref_image.

    require_record: if True, drop cases without an agent record (and warn).
    use_search_image: if False, ignore search_image even if present in record
      (e.g. mask off + no image search ablation).
    use_refined_prompt: if False, keep the raw prompt even when the record has
      a refined_text_instruction (e.g. visual_refine ablation).
    """
    out = []
    missing = 0
    for c in cases:
        rec = records.get(c["case_id"])
        if rec is None:
            missing += 1
            if require_record:
                continue
            out.append(c)
            continue

        merged = dict(c)
        merged["raw_instruction"] = c["prompt"]
        payload = rec.get("final_payload", {}) or {}
        plan = rec.get("plan", {}) or {}
        refined = (payload.get("refined_text_instruction") or "").strip()
        if refined and use_refined_prompt:
            merged["prompt"] = refined
        merged["agent_subtask"] = payload.get("subtask", "")
        merged["agent_image_search_query"] = (
            (rec.get("search", {}) or {}).get("query")
            or (rec.get("search", {}) or {}).get("agent_query")
            or plan.get("image_search")
            or ""
        )
        if use_search_image:
            si = payload.get("search_image")
            if isinstance(si, str) and si and os.path.exists(si):
                merged["search_image"] = si
        if use_mask_overlay:
            mp = (rec.get("mask", {}) or {}).get("mask_path") or payload.get("object_mask")
            if isinstance(mp, str) and mp and os.path.exists(mp):
                merged["mask_path"] = mp
        out.append(merged)
    if missing:
        msg = f"agent records missing for {missing} cases"
        if require_record:
            print(f"WARNING: {msg} (dropped because --require_agent_record was on)")
        else:
            print(f"NOTE: {msg} (kept with raw prompt, no ref image)")
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", default=None,
                        help="Editor checkpoint path. Omit to auto-download "
                             "yeates/aurora-weights:aurora_editor.safetensors "
                             "(public; no token required).")
    parser.add_argument("--bench_dir", default=os.environ.get(
        "AURORA_BENCH_DIR", str(REPO_ROOT / "benchmarks" / "agent_bench")
    ))
    parser.add_argument("--out_dir", default=None)
    parser.add_argument("--num_gpus", type=int, default=8)
    parser.add_argument("--gpu_ids", default=None)
    parser.add_argument("--max_pixels", type=int, default=480 * 832)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cfg_scale", type=float, default=2.0)
    parser.add_argument("--image_cfg_scale", type=float, default=1.0)
    parser.add_argument("--fallback_to_two_pass_cfg", action="store_true", default=True)
    parser.add_argument("--num_frames", type=int, default=81)
    parser.add_argument("--save_frames", type=int, default=0)
    parser.add_argument("--case_ids", default=None)
    parser.add_argument("--tag", default="")
    parser.add_argument("--agent_records_jsonl", default=None,
                        help="Aurora agent_pipeline_records.jsonl. When set, override per-case "
                             "prompt with final_payload.refined_text_instruction and feed "
                             "final_payload.search_image as a single ref_image (mask is ignored).")
    parser.add_argument("--require_agent_record", action="store_true",
                        help="Drop cases that have no matching agent record. Default: keep with raw prompt.")
    parser.add_argument("--no_search_image", action="store_true",
                        help="Even with --agent_records_jsonl, do not feed search_image into ref_image.")
    parser.add_argument("--no_refined_prompt", action="store_true",
                        help="Even with --agent_records_jsonl, keep the raw prompt instead of "
                             "overriding it with final_payload.refined_text_instruction "
                             "(visual_refine ablation: only image search + mask, no text refinement).")
    parser.add_argument("--use_mask_overlay", action="store_true",
                        help="Compose the agent's binary mask onto src_video[0] via "
                             "diffsynth.utils.mask_overlay.compose_masked_image (alpha=0.55, "
                             "red, 3px contour) and feed as an additional ref_image. Matches "
                             "stage2 training (UnifiedDataset._compose_ref_mask_into_ref_image). "
                             "Default OFF — only applicable for cases whose agent record "
                             "produced a mask file (typically remove_object subtask).")
    parser.add_argument("--reverse", action="store_true",
                        help="Reverse case ordering before sharding across GPUs. Pair with the "
                             "default forward ordering on a second machine to bidirectionally "
                             "consume the bench: both runs auto-skip cases whose generate.mp4 "
                             "already exists, so they meet in the middle without re-doing work. "
                             "Both processes must share the same --out_dir.")
    parser.add_argument("--run_tag", default="",
                        help="Per-machine label that suffixes _worker, log_gpu*, .results_gpu*, "
                             ".pipeline_load.lock, and inference_results files so two machines "
                             "sharing --out_dir do not clobber each other's run artifacts. The "
                             "case dirs (<case_id>/generate.mp4 etc.) stay shared, which is "
                             "what enables resume + reverse cooperation. Recommended values: "
                             "fwd / rev, or hostnames.")
    args = parser.parse_args()

    sys.path.insert(0, str(REPO_ROOT))
    from evaluation.model_download import resolve_editor_ckpt
    args.ckpt = resolve_editor_ckpt(args.ckpt)

    bench_dir = Path(args.bench_dir).resolve()
    prompts_jsonl = bench_dir / "prompts.jsonl"
    if not prompts_jsonl.exists():
        raise FileNotFoundError(f"Missing {prompts_jsonl}; build it via build_bench.py")

    case_ids = set(s.strip() for s in args.case_ids.split(",")) if args.case_ids else None
    cases = load_cases(prompts_jsonl, case_ids)
    print(f"Loaded {len(cases)} cases from {prompts_jsonl}")

    if args.agent_records_jsonl:
        records = load_agent_records(Path(args.agent_records_jsonl))
        print(f"Loaded {len(records)} agent records from {args.agent_records_jsonl}")
        cases = apply_agent_records(
            cases, records,
            require_record=args.require_agent_record,
            use_search_image=not args.no_search_image,
            use_mask_overlay=args.use_mask_overlay,
            use_refined_prompt=not args.no_refined_prompt,
        )
        n_with_ref = sum(1 for c in cases if c.get("search_image"))
        n_with_mask = sum(1 for c in cases if c.get("mask_path"))
        print(f"After agent record merge: {len(cases)} cases, "
              f"{n_with_ref} with search_image, {n_with_mask} with mask_path"
              + (" (mask compositing ON)" if args.use_mask_overlay else ""))

    if args.reverse:
        cases = list(reversed(cases))
        print(f"Reversed case order ({len(cases)} cases) — pair with non-reversed run on another machine for bi-directional consumption.")

    physical_gpus = ([int(g) for g in args.gpu_ids.split(",")]
                     if args.gpu_ids else list(range(args.num_gpus)))
    num_workers = len(physical_gpus)

    if args.out_dir:
        out_dir = Path(args.out_dir)
    else:
        ckpt_stem = Path(args.ckpt).stem
        cfg_tag = f"t{args.cfg_scale}_i{args.image_cfg_scale}"
        if args.fallback_to_two_pass_cfg:
            cfg_tag += "_2pass"
        cfg_tag += f"_nf{args.num_frames}"
        if args.tag:
            cfg_tag += f"_{args.tag}"
        out_dir = bench_dir / "results" / f"{ckpt_stem}_{cfg_tag}"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {out_dir}")
    print(f"GPUs: {physical_gpus}")

    rt_suffix = f"_{args.run_tag}" if args.run_tag else ""
    worker_path = out_dir / f"_worker{rt_suffix}.py"
    worker_path.write_text(WORKER_SCRIPT)

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
        env["EV_BENCH_DIR"] = str(bench_dir)
        env["EV_OUT_DIR"] = str(out_dir)
        env["EV_CASES"] = json.dumps(gpu_cases[worker_idx])
        env["EV_SEED"] = str(args.seed)
        env["EV_CFG_SCALE"] = str(args.cfg_scale)
        env["EV_IMAGE_CFG_SCALE"] = str(args.image_cfg_scale)
        env["EV_FALLBACK_TO_TWO_PASS_CFG"] = "1" if args.fallback_to_two_pass_cfg else "0"
        env["EV_MAX_PIXELS"] = str(args.max_pixels)
        env["EV_NUM_FRAMES"] = str(args.num_frames)
        env["EV_SAVE_FRAMES"] = str(args.save_frames)
        env["EV_GPU_ID"] = str(phys_gpu)
        env["EV_RUN_TAG"] = args.run_tag
        env["EV_USE_MASK_OVERLAY"] = "1" if args.use_mask_overlay else "0"
        env["TRITON_CACHE_DIR"] = triton_cache

        log_path = out_dir / f"log_gpu{phys_gpu}{rt_suffix}.txt"
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
        rf = out_dir / f".results_gpu{phys_gpu}{rt_suffix}.json"
        if rf.exists():
            all_results.extend(json.loads(rf.read_text()))
    all_results.sort(key=lambda r: r["case_id"])

    results_json = out_dir / f"inference_results{rt_suffix}.json"
    results_json.write_text(json.dumps(all_results, indent=2))

    n_ok = sum(1 for r in all_results if r["success"])
    print(f"\nInference complete: {n_ok}/{len(all_results)} succeeded")
    print(f"Results: {results_json}")


if __name__ == "__main__":
    main()
