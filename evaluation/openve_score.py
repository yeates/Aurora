"""OpenVE-Bench scoring using an OpenAI-compatible VLM judge.

Uses the exact same evaluation prompts as the official OpenVE-Bench
gemini_benchmark.py (loaded from openve_prompts.json) to ensure scores
are comparable.

Configure the judge via --model / --base_url / --api_key (any
OpenAI-compatible VLM endpoint). Defaults read OPENAI_BASE_URL and
OPENAI_API_KEY from the environment.

Supports parallel scoring with --num_workers (default 16).
"""

import argparse
import base64
import csv
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from openai import OpenAI


REPO_ROOT = Path(__file__).resolve().parents[1]

_PROMPT_FILE = Path(__file__).resolve().parent / "openve_prompts.json"
with open(_PROMPT_FILE, "r", encoding="utf-8") as _f:
    PROMPTS: dict[str, str] = json.load(_f)


def extract_scores_and_average(entry: str) -> tuple[float | None, list[float]]:
    """Extract numeric scores from VLM response. Returns (average, [scores]).

    Matches the official OpenVE-Bench extraction logic exactly:
    grabs all numbers following a colon, averages them.
    """
    pattern = r':\s*(\d+\.?\d*)'
    matches = re.findall(pattern, entry)
    scores = []
    for match in matches:
        try:
            scores.append(float(match))
        except ValueError:
            continue
    if scores:
        return round(sum(scores) / len(scores), 2), scores
    return None, []


def encode_video_base64(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def call_vlm(client: OpenAI, original_path: str, edited_path: str,
             prompt: str, model: str = "gpt-4o",
             max_retries: int = 5) -> str:
    """Call the VLM judge with original + edited video and evaluation prompt.

    The judge model is selected via the `--model` CLI arg on an
    OpenAI-compatible endpoint (configure --model/--base_url/--api_key).
    """
    for attempt in range(max_retries):
        try:
            user_content = [{"type": "text", "text": prompt.strip()}]

            b64_orig = encode_video_base64(original_path)
            user_content.append({
                "type": "image_url",
                "image_url": {"url": f"data:video/mp4;base64,{b64_orig}"}
            })

            b64_edit = encode_video_base64(edited_path)
            user_content.append({
                "type": "image_url",
                "image_url": {"url": f"data:video/mp4;base64,{b64_edit}"}
            })

            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": user_content},
                ],
                temperature=0.7,
                max_tokens=8192,
                timeout=180,
            )
            return response.choices[0].message.content

        except Exception as e:
            print(f"  Retry {attempt + 1}/{max_retries}: {e}")
            if attempt < max_retries - 1:
                time.sleep(30)
    return "ERROR: max retries exceeded"


def score_one(client: OpenAI, row: dict, bench_dir: str, row_idx: int,
              total: int, model: str = "gpt-4o") -> dict:
    """Score a single sample. Thread-safe."""
    edited_type = row["edited_type"]
    prompt = row["prompt"]
    original_video = os.path.join(bench_dir, row["original_video"])
    edited_path = row["edited_result_path"]

    result = {
        "idx": row_idx,
        "edited_type": edited_type,
        "prompt": prompt,
        "original_video": row["original_video"],
        "edited_result_path": edited_path,
        "response": "",
        "average": None,
        "scores": [],
    }

    if not os.path.exists(edited_path):
        result["response"] = f"ERROR: edited video not found: {edited_path}"
        print(f"  [{row_idx + 1}/{total}] SKIP: {edited_path}")
        return result

    if not os.path.exists(original_video):
        result["response"] = f"ERROR: original not found: {original_video}"
        print(f"  [{row_idx + 1}/{total}] SKIP: {original_video}")
        return result

    system_prompt = PROMPTS.get(edited_type)
    if system_prompt is None:
        result["response"] = f"ERROR: unknown type {edited_type}"
        return result

    full_prompt = system_prompt.replace("<edit_prompt>", prompt)
    response = call_vlm(client, original_video, edited_path, full_prompt, model=model)
    average, scores = extract_scores_and_average(response)

    result["response"] = response
    result["average"] = average
    result["scores"] = scores

    print(f"  [{row_idx + 1}/{total}] {edited_type}: avg={average} scores={scores}")
    return result


def _row_identity(row: dict) -> tuple[str, str, str]:
    return (
        row.get("original_video", ""),
        row.get("edited_result_path", ""),
        row.get("prompt", ""),
    )


def _read_cached_score_row(scored_row: dict, idx: int) -> dict:
    return {
        "idx": idx,
        "edited_type": scored_row["edited_type"],
        "prompt": scored_row["prompt"],
        "original_video": scored_row["original_video"],
        "edited_result_path": scored_row["edited_result_path"],
        "response": scored_row["response"].replace("\\n", "\n"),
        "average": float(scored_row["average"]),
        "scores": json.loads(scored_row.get("scores", "[]")),
    }


def main():
    parser = argparse.ArgumentParser(description="OpenVE-Bench VLM scoring (parallel)")
    parser.add_argument("--input_csv", default=None,
                        help="CSV with edited_type, prompt, original_video, edited_result_path. "
                             "If omitted, must pass --results_dir (reads {dir}/benchmark_results.csv).")
    parser.add_argument("--bench_dir", default=os.environ.get(
        "AURORA_BENCH_DIR", str(REPO_ROOT / "benchmarks" / "openve")
    ))
    parser.add_argument("--output_csv", default=None,
                        help="Output CSV with scores (default: input_csv stem + _scored.csv)")
    parser.add_argument("--api_key", default=os.environ.get("OPENAI_API_KEY", ""))
    parser.add_argument("--base_url", default=os.environ.get("OPENAI_BASE_URL", ""))
    parser.add_argument("--num_workers", type=int, default=32,
                        help="Number of parallel scoring workers")
    parser.add_argument(
        "--model",
        default="gpt-4o",
        help=(
            "VLM judge model on an OpenAI-compatible endpoint "
            "(configure --model/--base_url/--api_key). Default 'gpt-4o'."
        ),
    )
    parser.add_argument("--resume", action="store_true",
                        help="Resume from existing scored CSV, only re-score ERROR entries")
    parser.add_argument("--results_dir", default=None,
                        help="Shortcut: when set, input_csv defaults to "
                             "{results_dir}/benchmark_results.csv and output_csv "
                             "defaults to {results_dir}/benchmark_results_scored.csv.")
    args = parser.parse_args()

    if args.results_dir:
        if not args.input_csv:
            args.input_csv = str(Path(args.results_dir) / "benchmark_results.csv")
        if args.output_csv is None:
            args.output_csv = str(Path(args.results_dir) / "benchmark_results_scored.csv")

    if not args.input_csv:
        raise ValueError("Provide --input_csv or --results_dir")

    if not args.api_key:
        parser.error("Provide --api_key or set OPENAI_API_KEY env var")
    if not args.base_url:
        parser.error("Provide --base_url or set OPENAI_BASE_URL env var")

    if args.output_csv is None:
        stem = Path(args.input_csv).stem
        args.output_csv = str(Path(args.input_csv).parent / f"{stem}_scored.csv")

    client = OpenAI(api_key=args.api_key, base_url=args.base_url)
    bench_dir = Path(args.bench_dir)

    with open(args.input_csv, "r", encoding="utf-8-sig") as infile:
        reader = csv.DictReader(infile)
        rows = list(reader)

    total = len(rows)

    existing_results: dict[int, dict] = {}
    if args.resume and Path(args.output_csv).exists():
        row_index_by_identity: dict[tuple[str, str, str], int] = {}
        duplicate_identities = set()
        for idx, row in enumerate(rows):
            identity = _row_identity(row)
            if identity in row_index_by_identity:
                duplicate_identities.add(identity)
            else:
                row_index_by_identity[identity] = idx
        with open(args.output_csv, "r", encoding="utf-8-sig") as f:
            scored_reader = csv.DictReader(f)
            for legacy_idx, scored_row in enumerate(scored_reader):
                avg_str = scored_row.get("average", "")
                resp = scored_row.get("response", "")
                if not avg_str or resp.startswith("ERROR"):
                    continue
                if scored_row.get("idx", "") != "":
                    scored_idx = int(scored_row["idx"])
                    if not 0 <= scored_idx < total:
                        print(f"  Ignoring cached row with out-of-range idx={scored_idx}")
                        continue
                else:
                    identity = _row_identity(scored_row)
                    if identity in duplicate_identities:
                        print(f"  Ignoring legacy cached row with duplicate identity: {identity}")
                        continue
                    scored_idx = row_index_by_identity.get(identity)
                    if scored_idx is None:
                        print(f"  Ignoring legacy cached row not present in current input: legacy_row={legacy_idx}")
                        continue
                current_identity = _row_identity(rows[scored_idx])
                if _row_identity(scored_row) != current_identity:
                    print(f"  Ignoring cached row idx={scored_idx} because input identity changed")
                    continue
                existing_results[scored_idx] = _read_cached_score_row(scored_row, scored_idx)

    to_score = [idx for idx in range(total) if idx not in existing_results]
    print(f"Scoring {len(to_score)}/{total} samples with {args.num_workers} parallel workers"
          f" ({len(existing_results)} cached from resume)...")

    results = [None] * total
    for idx, cached in existing_results.items():
        results[idx] = cached

    def _write_score_csv(output_path: Path):
        """Write current `results` list to CSV atomically."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = output_path.with_name(f"{output_path.name}.tmp.{os.getpid()}")
        header = ["idx", "edited_type", "prompt", "original_video", "edited_result_path",
                  "response", "average", "scores"]
        with open(tmp_path, "w", encoding="utf-8", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=header)
            w.writeheader()
            for r in results:
                if r is None:
                    continue
                w.writerow({
                    "idx": r["idx"],
                    "edited_type": r["edited_type"],
                    "prompt": r["prompt"],
                    "original_video": r["original_video"],
                    "edited_result_path": r["edited_result_path"],
                    "response": r["response"].replace("\n", "\\n"),
                    "average": r["average"] if r["average"] is not None else "",
                    "scores": json.dumps(r["scores"]),
                })
        os.replace(tmp_path, output_path)

    output_csv_path = Path(args.output_csv)
    ckpt_every = max(1, int(os.environ.get("OPENVE_CKPT_EVERY", "25")))

    with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
        futures = {}
        for idx in to_score:
            fut = executor.submit(score_one, client, rows[idx], str(bench_dir), idx, total, args.model)
            futures[fut] = idx

        done_count = 0
        for fut in as_completed(futures):
            idx = futures[fut]
            try:
                results[idx] = fut.result()
            except Exception as e:
                print(f"  [{idx}] EXCEPTION: {e}")
                results[idx] = {
                    "idx": idx, "edited_type": rows[idx].get("edited_type", ""),
                    "prompt": rows[idx].get("prompt", ""),
                    "original_video": rows[idx].get("original_video", ""),
                    "edited_result_path": rows[idx].get("edited_result_path", ""),
                    "response": f"ERROR: {e}", "average": None, "scores": [],
                }
            done_count += 1
            if done_count % ckpt_every == 0:
                _write_score_csv(output_csv_path)
                print(f"  Progress: {done_count}/{total} (checkpoint written)")

    _write_score_csv(output_csv_path)

    all_scores_by_type: dict[str, list[float]] = {}
    all_scores: list[float] = []
    for r in results:
        if r["average"] is not None and 1 <= r["average"] <= 5:
            all_scores.append(r["average"])
            all_scores_by_type.setdefault(r["edited_type"], []).append(r["average"])

    type_averages = {}
    for etype, scores in all_scores_by_type.items():
        type_averages[etype] = round(sum(scores) / len(scores), 2) if scores else None

    overall_average = round(sum(all_scores) / len(all_scores), 2) if all_scores else None

    stats = {
        "overall_average": overall_average,
        "type_averages": type_averages,
        "total_processed": len(all_scores),
        "total_valid_scores": len(all_scores),
        "breakdown_by_type": {},
    }
    for etype, scores in all_scores_by_type.items():
        stats["breakdown_by_type"][etype] = {
            "count": len(scores),
            "average": round(sum(scores) / len(scores), 2) if scores else None,
        }

    stats_path = Path(args.output_csv).with_suffix(".json")
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    detailed_path = Path(args.output_csv).parent / "detailed_results.json"
    with open(detailed_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\n{'=' * 60}")
    print(f"Scoring complete!")
    print(f"  Overall average: {overall_average}")
    print(f"  Per-type averages: {json.dumps(type_averages, indent=2)}")
    print(f"  Scored CSV: {args.output_csv}")
    print(f"  Stats JSON: {stats_path}")
    print(f"  Detailed JSON: {detailed_path}")


if __name__ == "__main__":
    main()
