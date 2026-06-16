"""EditVerse benchmark scoring using an OpenAI-compatible VLM judge.

Faithful reproduction of EditVerse's editing_vlm_evaluation.py:
- Samples 3 frames from the edited video (evenly spaced)
- For each frame, sends source frame + edited frame as JPEG images to VLM
- Uses the exact same prompt as EditVerse (Prompt Following 0-3, Edit Quality 0-3,
  Background Consistency 0-3, Total 0-9)
- Second VLM call extracts the numeric total score
- Per-sample score = mean of 3 frame scores
- Overall score = mean of all per-sample scores

Configure the judge via --model / --base_url / --api_key (any OpenAI-compatible
VLM endpoint). Defaults read OPENAI_BASE_URL and OPENAI_API_KEY from the
environment.
"""

import argparse
import base64
import csv
import io
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import imageio
import numpy as np
from PIL import Image as PILImage
from openai import OpenAI


REPO_ROOT = Path(__file__).resolve().parents[1]


EVAL_PROMPT_TEMPLATE = (
    'You are a meticulous video editing quality evaluator. Your task is to provide a '
    'detailed assessment of a video edit by comparing the original image with the edited '
    'image based on a given text prompt.\n'
    'Editing Prompt:\n{editing_prompt}\n'
    'Instructions:\n'
    'Analyze the provided image (the edited video frame) and evaluate how well the '
    '"Editing Prompt" has been executed. You will evaluate the edit across three distinct '
    'criteria. For each criterion, provide a score from 0 (worst) to 3 (best) and a brief '
    'justification. Finally, provide the total score.\n'
    'Your evaluation should focus on three key aspects:\n'
    '1. Prompt Following (Score: 0-3) \n'
    'Question: Does the edit accurately and completely fulfill the instructions in the '
    '"Editing Prompt"? \n'
    'Scoring Guide:\n'
    '- 3: The prompt is perfectly and completely followed.\n'
    '- 2: The prompt is mostly followed but with minor inaccuracies or omissions.\n'
    '- 1: The prompt is poorly followed or only partially executed.\n'
    '- 0: The prompt is completely ignored or the opposite was done. \n'
    '2. Edit Quality (Score: 0-3) \n'
    'Question: How is the visual quality of the edited area itself? Is it realistic, '
    'seamless, and free of artifacts (e.g., blurriness, distortion, unnatural textures)?\n'
    'Scoring Guide:\n'
    '- 3: The edit is of high visual quality, seamless, and artifact-free.\n'
    '- 2: The edit is good but has minor, noticeable artifacts.\n'
    '- 1: The edit is of low quality with significant, distracting artifacts.\n'
    '- 0: The edited area is extremely poor, garbled, or has completely failed.\n'
    '3. Background Consistency (Score: 0-3) \n'
    'Question: Have the areas that should not have been edited remained unchanged between '
    'the "Before" and "After" images? \n'
    'Scoring Guide:\n'
    '- 3: The areas that should not have been edited are perfectly preserved and stable. \n'
    '- 2: There are minor, subtle, but noticeable changes or flickers in the areas that '
    'should not have been edited.\n'
    '- 1: There are significant and distracting changes in the areas that should not have '
    'been edited. \n'
    '- 0: The areas that should not have been edited is completely or catastrophically '
    'altered. \n'
    'Please provide your evaluation in the following format: \n'
    'Prompt Following: [Your score, 0-3] - [Brief justification for the score.]\n'
    'Edit Quality: [Your score, 0-3] - [Brief justification for the score.]\n'
    'Background Consistency: [Your score, 0-3] - [Brief justification for the score.]\n'
    'Total Score: [Sum of the three scores]\n'
)

EXTRACT_PROMPT_TEMPLATE = (
    'Please output the overall score mentioned in this sentence. '
    'Only output the overall score number. Sentence: {response}'
)

VALID_SCORES = {'0', '1', '2', '3', '4', '5', '6', '7', '8', '9'}

_DIM_PATTERNS = {
    "prompt_following": re.compile(
        r"(?im)^[\s>\-*#]*\**\s*Prompt\s*Following\b[^\n]*?[:：]\s*[\[\(`*]*\s*([0-3])(?!-)\b"
    ),
    "edit_quality": re.compile(
        r"(?im)^[\s>\-*#]*\**\s*Edit\s*Quality\b[^\n]*?[:：]\s*[\[\(`*]*\s*([0-3])(?!-)\b"
    ),
    "background_consistency": re.compile(
        r"(?im)^[\s>\-*#]*\**\s*Background\s*Consistency\b[^\n]*?[:：]\s*[\[\(`*]*\s*([0-3])(?!-)\b"
    ),
}


def _parse_dimension_scores(eval_text: str) -> dict:
    """Extract the three 0-3 dimension scores from Gemini's detailed response."""
    out = {}
    for key, pat in _DIM_PATTERNS.items():
        m = pat.search(eval_text or "")
        out[key] = int(m.group(1)) if m else None
    return out


def frame_to_base64_jpeg(frame_bgr: np.ndarray) -> str:
    """Encode a BGR frame as base64 JPEG, matching EditVerse's get_base64()."""
    frame_rgb = frame_bgr[:, :, ::-1]
    img = PILImage.fromarray(frame_rgb)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def get_video_frame_count(video_path: str) -> int:
    """Get the total number of frames without reading them all into memory."""
    try:
        reader = imageio.get_reader(video_path)
        count = reader.count_frames()
        reader.close()
        return count
    except Exception:
        return 0


def read_video_frame(video_path: str, frame_idx: int) -> np.ndarray | None:
    """Read a single frame from a video as a BGR numpy array."""
    try:
        reader = imageio.get_reader(video_path)
        frame_rgb = reader.get_data(frame_idx)
        reader.close()
        return frame_rgb[:, :, ::-1]
    except Exception:
        return None


def read_video_frames(video_path: str) -> list[np.ndarray]:
    """Read all frames from a video as BGR numpy arrays.

    Uses imageio for reading, then converts RGB -> BGR to match EditVerse's
    cv2-based get_base64().
    """
    try:
        reader = imageio.get_reader(video_path)
        frames = []
        for frame_rgb in reader:
            frame_bgr = frame_rgb[:, :, ::-1]
            frames.append(frame_bgr)
        reader.close()
        return frames
    except Exception:
        return []


def sample_frame_indices(num_frames: int, num_samples: int = 3) -> list[int]:
    """Sample evenly spaced frame indices, matching EditVerse's range(0, len, len//n)."""
    if num_frames == 0:
        return []
    step = num_frames // num_samples
    if step == 0:
        step = 1
    return list(range(0, num_frames, step))


def call_vlm_eval(client: OpenAI, model: str, before_b64: str, after_b64: str,
                   editing_prompt: str, max_retries: int = 5) -> dict | None:
    """Run the two-call EditVerse VLM evaluation for one frame pair.

    Returns a dict with the total (0-9), the three 0-3 dimension scores, and
    the raw VLM justification text; or None if all retries fail.
    """
    prompt_text = EVAL_PROMPT_TEMPLATE.format(editing_prompt=editing_prompt)

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt_text},
                        {"type": "image_url",
                         "image_url": {"url": f"data:image/jpeg;base64,{before_b64}"}},
                        {"type": "image_url",
                         "image_url": {"url": f"data:image/jpeg;base64,{after_b64}"}},
                    ],
                }],
                temperature=0,
                max_tokens=4096,
                timeout=120,
            )
            eval_text = response.choices[0].message.content
            if not eval_text:
                print(f"    Empty eval response, retrying...")
                continue

            extract_prompt = EXTRACT_PROMPT_TEMPLATE.format(response=eval_text)
            response2 = client.chat.completions.create(
                model=model,
                messages=[{
                    "role": "user",
                    "content": [{"type": "text", "text": extract_prompt}],
                }],
                temperature=0,
                max_tokens=512,
                timeout=60,
            )
            score_text = (response2.choices[0].message.content or "").strip()

            if score_text in VALID_SCORES:
                dims = _parse_dimension_scores(eval_text)
                return {
                    "total": int(score_text),
                    "prompt_following": dims["prompt_following"],
                    "edit_quality": dims["edit_quality"],
                    "background_consistency": dims["background_consistency"],
                    "justification": eval_text,
                }
            print(f"    Invalid score '{score_text}', retrying...")
            continue

        except Exception as e:
            print(f"    Retry {attempt + 1}/{max_retries}: {e}")
            if attempt < max_retries - 1:
                time.sleep(10)

    return None


def score_one_sample(client: OpenAI, model: str, source_video_path: str,
                     edited_video_path: str, editing_prompt: str,
                     bench_id: str, total: int) -> dict:
    """Score one EditVerse sample (3 frames). Thread-safe."""
    result = {
        "bench_id": bench_id,
        "editing_prompt": editing_prompt,
        "source_video": source_video_path,
        "edited_video": edited_video_path,
        "frame_scores": [],
        "frame_details": [],
        "dimension_averages": {},
        "average": None,
        "error": None,
    }

    if not os.path.exists(edited_video_path):
        result["error"] = f"edited video not found: {edited_video_path}"
        print(f"  [{bench_id}] SKIP: {result['error']}")
        return result

    if not os.path.exists(source_video_path):
        result["error"] = f"source video not found: {source_video_path}"
        print(f"  [{bench_id}] SKIP: {result['error']}")
        return result

    edited_count = get_video_frame_count(edited_video_path)
    source_count = get_video_frame_count(source_video_path)

    if edited_count == 0:
        result["error"] = "edited video has 0 frames"
        return result

    frame_indices = sample_frame_indices(edited_count, 3)
    frame_details: list[dict] = []

    for fi in frame_indices:
        src_fi = min(fi, source_count - 1) if source_count > 0 else 0
        source_frame = read_video_frame(source_video_path, src_fi)
        edited_frame = read_video_frame(edited_video_path, fi)

        if source_frame is None or edited_frame is None:
            continue

        before_b64 = frame_to_base64_jpeg(source_frame)
        after_b64 = frame_to_base64_jpeg(edited_frame)

        detail = call_vlm_eval(client, model, before_b64, after_b64, editing_prompt)
        if detail is not None:
            detail = dict(detail)
            detail["frame_index"] = int(fi)
            frame_details.append(detail)

    totals = [d["total"] for d in frame_details]
    result["frame_scores"] = totals
    result["frame_details"] = frame_details
    if totals:
        result["average"] = round(sum(totals) / len(totals), 4)

    dim_avgs: dict = {}
    for dim in ("prompt_following", "edit_quality", "background_consistency"):
        vals = [d.get(dim) for d in frame_details if d.get(dim) is not None]
        dim_avgs[dim] = round(sum(vals) / len(vals), 4) if vals else None
    result["dimension_averages"] = dim_avgs

    print(
        f"  [{bench_id}] totals={totals} avg={result['average']} "
        f"dims={dim_avgs}"
    )
    return result


def main():
    parser = argparse.ArgumentParser(description="EditVerse VLM scoring")
    parser.add_argument("--results_dir", required=True,
                        help="Directory with edited videos (bench_id.mp4 or bench_id/generate.mp4)")
    parser.add_argument("--bench_dir", default=os.environ.get(
        "AURORA_BENCH_DIR", str(REPO_ROOT / "benchmarks" / "editverse")
    ), help="EditVerse benchmark data directory")
    parser.add_argument("--bench_json", default=None,
                        help="Benchmark JSON (default: bench_dir/EditVerseBench_3.json)")
    parser.add_argument("--output", default=None,
                        help="Output JSON path (default: results_dir/scores.json)")
    parser.add_argument("--api_key", default=os.environ.get("OPENAI_API_KEY", ""))
    parser.add_argument("--base_url", default=os.environ.get("OPENAI_BASE_URL", ""))
    parser.add_argument(
        "--model",
        default="gpt-4o",
        help=(
            "VLM judge model on an OpenAI-compatible endpoint "
            "(configure --model/--base_url/--api_key). Default 'gpt-4o'."
        ),
    )
    parser.add_argument("--num_workers", type=int, default=32)
    parser.add_argument("--source_dir", default=None,
                        help="Directory with source videos in {bench_id}/video1.mp4 layout "
                             "(default: use results_dir for nested, or baselines/EditVerse for flat)")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from existing scores, only re-score errors/missing")
    args = parser.parse_args()

    if not args.api_key:
        parser.error("Provide --api_key or set OPENAI_API_KEY env var")
    if not args.base_url:
        parser.error("Provide --base_url or set OPENAI_BASE_URL env var")

    bench_dir = Path(args.bench_dir)
    bench_json = args.bench_json or str(bench_dir / "EditVerseBench_3.json")
    results_dir = Path(args.results_dir)
    output_path = Path(args.output) if args.output else results_dir / "scores.json"

    with open(bench_json, "r", encoding="utf-8") as f:
        bench_data = json.load(f)

    source_dir = Path(args.source_dir) if args.source_dir else None
    if source_dir is None:
        test_nested = results_dir / "0" / "video1.mp4"
        if test_nested.exists():
            source_dir = results_dir
        else:
            source_dir = bench_dir / "baselines" / "EditVerse"

    tasks = []
    for bench_id, entry in sorted(bench_data.items(), key=lambda x: int(x[0])):
        source_path = str(source_dir / bench_id / "video1.mp4")

        edited_flat = results_dir / f"{bench_id}.mp4"
        edited_nested = results_dir / bench_id / "generate.mp4"
        if edited_flat.exists():
            edited_path = str(edited_flat)
        elif edited_nested.exists():
            edited_path = str(edited_nested)
        else:
            edited_path = str(edited_flat)

        prompt = entry.get("<text>", "")

        tasks.append({
            "bench_id": bench_id,
            "edit_type": entry.get("type", "unknown"),
            "editing_prompt": prompt,
            "source_path": source_path,
            "edited_path": edited_path,
        })

    existing: dict[str, dict] = {}
    if args.resume and output_path.exists():
        with open(output_path, "r") as f:
            prev = json.load(f)
        for item in prev.get("samples", []):
            if item.get("average") is not None and item.get("error") is None:
                existing[item["bench_id"]] = item

    to_score = [t for t in tasks if t["bench_id"] not in existing]
    print(f"Scoring {len(to_score)}/{len(tasks)} samples with {args.num_workers} workers"
          f" ({len(existing)} cached)")

    client = OpenAI(api_key=args.api_key, base_url=args.base_url)

    results_map: dict[str, dict] = dict(existing)

    with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
        futures = {}
        for t in to_score:
            fut = executor.submit(
                score_one_sample, client, args.model,
                t["source_path"], t["edited_path"], t["editing_prompt"],
                t["bench_id"], len(tasks),
            )
            futures[fut] = t

        for fut in as_completed(futures):
            t = futures[fut]
            try:
                res = fut.result()
                res["edit_type"] = t["edit_type"]
                results_map[t["bench_id"]] = res
            except Exception as e:
                print(f"  [{t['bench_id']}] EXCEPTION: {e}")
                results_map[t["bench_id"]] = {
                    "bench_id": t["bench_id"],
                    "edit_type": t["edit_type"],
                    "editing_prompt": t["editing_prompt"],
                    "error": str(e),
                    "average": None,
                    "frame_scores": [],
                    "frame_details": [],
                    "dimension_averages": {},
                }

    samples = []
    for bench_id in sorted(results_map.keys(), key=lambda x: int(x)):
        samples.append(results_map[bench_id])

    from collections import defaultdict
    type_scores = defaultdict(list)
    all_scores = []
    dim_totals: dict[str, list[float]] = {
        "prompt_following": [], "edit_quality": [], "background_consistency": [],
    }
    type_dim_totals: dict = defaultdict(lambda: {
        "prompt_following": [], "edit_quality": [], "background_consistency": [],
    })
    for s in samples:
        if s.get("average") is not None:
            all_scores.append(s["average"])
            type_scores[s.get("edit_type", "unknown")].append(s["average"])
        dims = s.get("dimension_averages") or {}
        for dim in dim_totals:
            v = dims.get(dim)
            if v is not None:
                dim_totals[dim].append(v)
                type_dim_totals[s.get("edit_type", "unknown")][dim].append(v)

    type_averages = {
        t: round(sum(sc) / len(sc), 4) for t, sc in type_scores.items() if sc
    }
    overall = round(sum(all_scores) / len(all_scores), 4) if all_scores else None

    dimension_averages = {
        dim: (round(sum(vs) / len(vs), 4) if vs else None)
        for dim, vs in dim_totals.items()
    }
    type_dimension_averages = {
        t: {dim: (round(sum(vs) / len(vs), 4) if vs else None)
            for dim, vs in dims.items()}
        for t, dims in type_dim_totals.items()
    }

    output_data = {
        "overall_average": overall,
        "type_averages": type_averages,
        "dimension_averages": dimension_averages,
        "type_dimension_averages": type_dimension_averages,
        "num_scored": len(all_scores),
        "num_total": len(tasks),
        "model": args.model,
        "samples": samples,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    csv_path = output_path.with_suffix(".csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "editing_vlm_evaluation", "type"])
        writer.writeheader()
        for s in samples:
            writer.writerow({
                "id": s["bench_id"],
                "editing_vlm_evaluation": s.get("average", ""),
                "type": s.get("edit_type", ""),
            })
        writer.writerow({
            "id": "Average",
            "editing_vlm_evaluation": overall if overall is not None else "",
            "type": "",
        })

    print(f"\n{'=' * 60}")
    print(f"Scoring complete!")
    print(f"  Overall average: {overall}")
    print(f"  Per-type: {json.dumps(type_averages, indent=2)}")
    print(f"  Output JSON: {output_path}")
    print(f"  Output CSV:  {csv_path}")


if __name__ == "__main__":
    main()
