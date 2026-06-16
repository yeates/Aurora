from __future__ import annotations

import argparse
import base64
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


RUBRIC_AXES = [
    "instruction_following",
    "edit_region_localization",
    "source_preservation",
    "visual_quality",
    "temporal_consistency",
    "ip_presence",
    "ip_identity_match",
]
IP_AXES = {"ip_presence", "ip_identity_match"}
NON_IP_EDIT_TYPES = {"reasoning", "removal"}


def relevant_axes(edit_type: str) -> list[str]:
    """Axes that are meaningful for an edit_type. removal/reasoning have no
    target IP, so the IP Presence / IP Identity Match axes are dropped."""
    if edit_type in NON_IP_EDIT_TYPES:
        return [a for a in RUBRIC_AXES if a not in IP_AXES]
    return list(RUBRIC_AXES)


def total_max(edit_type: str) -> int:
    return 3 * len(relevant_axes(edit_type))

EVAL_PROMPT_TEMPLATE = """\
You are a meticulous video-editing quality evaluator specialized in IP / branded-entity
edits. You are given two images: BEFORE (a frame from the source video) and AFTER (a frame
from the AI-edited video). The editing system was asked to perform a specific edit that
inserts, replaces, or backgrounds a named real-world entity.

Edit metadata:
  Edit type:        {edit_type}
  Edit region:      {edit_region}
  Target entity:    {target_entity}
  Search query:     {web_query}
  Full instruction: {instruction}

You will score 7 dimensions on a 0-3 scale. Be strict. The "Total" must be the
exact integer sum of the seven sub-scores.

[1] Instruction Following (0-3)
   Was the requested edit ({edit_type}) actually performed at all?
   - 3: instruction perfectly executed (right action on right region with right entity)
   - 2: action correct but with minor mismatches (e.g. partial completion, slight wrong region)
   - 1: only partial or wrong-direction execution
   - 0: instruction ignored or clearly opposite

[2] Edit Region Localization (0-3)
   Was the edit confined to the specified region "{edit_region}"?
   - 3: edit confined precisely to the specified region
   - 2: minor bleed beyond the region
   - 1: significant unintended changes outside the region
   - 0: edit happens in the wrong region or all over the frame

[3] Source Preservation (0-3)
   Are unedited parts of the scene preserved (subject identity, motion, surrounding
   objects, geometry, lighting outside the edit region)?
   - 3: perfectly preserved
   - 2: minor changes
   - 1: significant unintended changes
   - 0: catastrophically altered

[4] Visual Quality (0-3)
   Visual fidelity of the edited region: realism, seamlessness, lighting/shadow match,
   correct scale/perspective, absence of artifacts (blur, halos, pasted-sticker look,
   wrong occlusion).
   - 3: seamless and realistic
   - 2: noticeable but tolerable artifacts
   - 1: significant artifacts or scale/lighting mismatch
   - 0: garbled, distorted, or visibly fake

[5] Temporal Consistency (0-3)
   (Score this as if the AFTER frame is representative of the whole video; downgrade if
   the edited entity appears unstable, half-formed, ghosted, or inconsistent with what
   you would expect across a 81-frame clip.)
   - 3: edit is stable, well-defined, looks coherent across time
   - 2: minor temporal issues likely
   - 1: significant flickering / pop-in expected
   - 0: edit is unstable, ghosted, or barely formed

[6] IP Presence (0-3)
   Is the target entity "{target_entity}" actually present and visible in the AFTER
   frame, in roughly the right location?
   - 3: clearly visible and recognizable in the right region
   - 2: present but partially occluded, small, or off-region
   - 1: a vague hint that an edit was attempted but no clear entity
   - 0: target entity is not visible at all

[7] IP Identity Match (0-3)
   Does the visible entity actually match the *specific real-world identity* of
   "{target_entity}"? Consider distinguishing features: brand-specific shape, color,
   logo, signature design language, recognizable architecture or geography. Use your
   world knowledge of this specific IP / brand / location.
{identity_search_hint}
   - 3: closely matches canonical reference (recognizable shape, color, branding,
        logo, signature features)
   - 2: right product category but key identity features wrong (e.g., right type
        of pot but wrong color/branding/shape)
   - 1: only superficial resemblance to the named entity
   - 0: no resemblance / wrong product entirely / generic placeholder
   (Note: if [6] IP Presence == 0, set this to 0 as well.)

Output format - exactly these 8 lines, no preamble, no markdown headers:

Instruction Following: <0-3> - <one-sentence justification>
Edit Region Localization: <0-3> - <one-sentence justification>
Source Preservation: <0-3> - <one-sentence justification>
Visual Quality: <0-3> - <one-sentence justification>
Temporal Consistency: <0-3> - <one-sentence justification>
IP Presence: <0-3> - <one-sentence justification>
IP Identity Match: <0-3> - <one-sentence justification>
Total: <0-21, exact integer sum>
"""

EVAL_PROMPT_TEMPLATE_NON_IP = """\
You are a meticulous video-editing quality evaluator. You are given two images:
BEFORE (a frame from the source video) and AFTER (a frame from the AI-edited video).
The editing system was asked to perform a {edit_type} edit (no specific real-world
IP / branded entity is involved).

Edit metadata:
  Edit type:        {edit_type}
  Edit region:      {edit_region}
  Full instruction: {instruction}

You will score 5 dimensions on a 0-3 scale. Be strict. The "Total" must be the
exact integer sum of the five sub-scores.

[1] Instruction Following (0-3)
   Was the requested edit ({edit_type}) actually performed at all?
   - 3: instruction perfectly executed (right action on right region)
   - 2: action correct but with minor mismatches (e.g. partial completion, slight wrong region)
   - 1: only partial or wrong-direction execution
   - 0: instruction ignored or clearly opposite
   For removal: '3' = the target is gone AND the resulting hole is plausibly
   reconstructed; if the model REPLACED the target with a new object instead
   of removing it, give at most 1.

[2] Edit Region Localization (0-3)
   Was the edit confined to the specified region "{edit_region}"?
   - 3: edit confined precisely to the specified region
   - 2: minor bleed beyond the region
   - 1: significant unintended changes outside the region
   - 0: edit happens in the wrong region or all over the frame

[3] Source Preservation (0-3)
   Are unedited parts of the scene preserved (subject identity, motion, surrounding
   objects, geometry, lighting outside the edit region)?
   - 3: perfectly preserved
   - 2: minor changes
   - 1: significant unintended changes
   - 0: catastrophically altered

[4] Visual Quality (0-3)
   Visual fidelity of the edited region: realism, seamlessness, lighting/shadow match,
   correct scale/perspective, absence of artifacts (blur, halos, pasted-sticker look,
   wrong occlusion). For removal, this also covers the inpainted reconstruction.
   - 3: seamless and realistic
   - 2: noticeable but tolerable artifacts
   - 1: significant artifacts or scale/lighting mismatch
   - 0: garbled, distorted, or visibly fake

[5] Temporal Consistency (0-3)
   (Score this as if the AFTER frame is representative of the whole video; downgrade if
   the edited region appears unstable, half-formed, ghosted, or inconsistent with what
   you would expect across a 81-frame clip.)
   - 3: edit is stable, well-defined, looks coherent across time
   - 2: minor temporal issues likely
   - 1: significant flickering / pop-in expected
   - 0: edit is unstable, ghosted, or barely formed

Output format - exactly these 6 lines, no preamble, no markdown headers:

Instruction Following: <0-3> - <one-sentence justification>
Edit Region Localization: <0-3> - <one-sentence justification>
Source Preservation: <0-3> - <one-sentence justification>
Visual Quality: <0-3> - <one-sentence justification>
Temporal Consistency: <0-3> - <one-sentence justification>
Total: <0-15, exact integer sum>
"""


IDENTITY_SEARCH_HINT_NO_TOOL = (
    "   Use only your internal training knowledge of this entity; do not assume access "
    "to web search."
)
IDENTITY_SEARCH_HINT_WITH_TOOL = (
    "   You have access to web image search via the google_search tool. You SHOULD use it "
    "with the query \"{web_query}\" to retrieve canonical reference images of "
    "\"{target_entity}\" before scoring this dimension. Compare the visible entity in the "
    "AFTER frame to the retrieved references."
)


_DIM_LABELS = {
    "instruction_following": "Instruction Following",
    "edit_region_localization": "Edit Region Localization",
    "source_preservation": "Source Preservation",
    "visual_quality": "Visual Quality",
    "temporal_consistency": "Temporal Consistency",
    "ip_presence": "IP Presence",
    "ip_identity_match": "IP Identity Match",
}


def _make_dim_pattern(label: str) -> re.Pattern:
    esc = re.escape(label)
    return re.compile(
        rf"(?im)^[\s>\-*#\d.]*\**\s*{esc}\b[^\n]*?[:：][\s\[\(`*]*([0-3])(?!-)\b"
    )


_DIM_PATTERNS = {k: _make_dim_pattern(v) for k, v in _DIM_LABELS.items()}
_TOTAL_PATTERN = re.compile(
    r"(?im)^[\s>\-*#\d.]*\**\s*Total\b[^\n]*?[:：][\s\[\(`*]*(\d{1,2})\b"
)


def parse_scores(eval_text: str) -> dict:
    out: dict = {"raw": eval_text}
    for key, pat in _DIM_PATTERNS.items():
        m = pat.search(eval_text or "")
        out[key] = int(m.group(1)) if m else None
    m = _TOTAL_PATTERN.search(eval_text or "")
    if m:
        out["total"] = int(m.group(1))
    else:
        sub = [out.get(k) for k in RUBRIC_AXES]
        out["total"] = sum(v for v in sub if v is not None) if all(v is not None for v in sub) else None
    return out


def get_video_frame_count(p: str) -> int:
    try:
        r = imageio.get_reader(p); n = r.count_frames(); r.close(); return n
    except Exception:
        return 0


def read_frame(p: str, idx: int) -> np.ndarray | None:
    try:
        r = imageio.get_reader(p); f = r.get_data(idx); r.close(); return f
    except Exception:
        return None


def frame_to_jpeg_b64(frame_rgb: np.ndarray, quality: int = 90) -> str:
    img = PILImage.fromarray(frame_rgb)
    buf = io.BytesIO(); img.save(buf, format="JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def frame_to_jpeg_bytes(frame_rgb: np.ndarray, quality: int = 90) -> bytes:
    img = PILImage.fromarray(frame_rgb)
    buf = io.BytesIO(); img.save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


def sample_frame_indices(n: int, k: int = 3) -> list[int]:
    if n <= 0:
        return []
    if n <= k:
        return list(range(n))
    step = n // k
    return [i * step for i in range(k)]


def _build_user_content_oai(prompt_text: str, before_b64: str, after_b64: str) -> list:
    return [
        {"type": "text", "text": prompt_text},
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{before_b64}"}},
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{after_b64}"}},
    ]


def call_openai(client, model: str, prompt_text: str,
                before_b64: str, after_b64: str, max_retries: int = 5) -> str | None:
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": _build_user_content_oai(
                    prompt_text, before_b64, after_b64)}],
                temperature=0,
                max_tokens=4096,
                timeout=120,
            )
            txt = resp.choices[0].message.content
            if txt:
                return txt
            print(f"    empty VLM resp, retry {attempt+1}/{max_retries}")
        except Exception as e:
            print(f"    VLM retry {attempt+1}/{max_retries}: {e}")
            if attempt < max_retries - 1:
                time.sleep(8)
    return None


def _gemini_client(api_key: str):
    from google import genai
    return genai.Client(api_key=api_key)


def call_gemini_with_search(client, model: str, prompt_text: str,
                            before_jpeg: bytes, after_jpeg: bytes,
                            max_retries: int = 5) -> str | None:
    from google.genai import types

    contents = [
        types.Content(role="user", parts=[
            types.Part.from_text(text=prompt_text),
            types.Part.from_bytes(data=before_jpeg, mime_type="image/jpeg"),
            types.Part.from_bytes(data=after_jpeg, mime_type="image/jpeg"),
        ])
    ]
    config = types.GenerateContentConfig(
        temperature=0,
        max_output_tokens=4096,
        tools=[types.Tool(google_search=types.GoogleSearch())],
    )
    for attempt in range(max_retries):
        try:
            resp = client.models.generate_content(model=model, contents=contents, config=config)
            if resp.text:
                return resp.text
            print(f"    empty Gemini resp, retry {attempt+1}/{max_retries}")
        except Exception as e:
            print(f"    Gemini retry {attempt+1}/{max_retries}: {e}")
            if attempt < max_retries - 1:
                time.sleep(8)
    return None


def _build_prompt_text(case: dict, with_search: bool) -> str:
    edit_type = case.get("edit_type", "")
    if edit_type in NON_IP_EDIT_TYPES:
        return EVAL_PROMPT_TEMPLATE_NON_IP.format(
            edit_type=edit_type,
            edit_region=case.get("edit_region", ""),
            instruction=case.get("prompt") or case.get("instruction", ""),
        )
    target_entity = case.get("target_entity", "")
    web_query = case.get("web_query", target_entity)
    if with_search:
        hint = IDENTITY_SEARCH_HINT_WITH_TOOL.format(
            web_query=web_query, target_entity=target_entity)
    else:
        hint = IDENTITY_SEARCH_HINT_NO_TOOL
    return EVAL_PROMPT_TEMPLATE.format(
        edit_type=edit_type,
        edit_region=case.get("edit_region", ""),
        target_entity=target_entity,
        web_query=web_query,
        instruction=case.get("prompt") or case.get("instruction", ""),
        identity_search_hint=hint,
    )


def score_one_case(case: dict, *, source_path: str, edited_path: str,
                   model: str, api: str, num_frames: int,
                   openai_client=None, gemini_client=None,
                   max_retries: int = 4) -> dict:
    case_id = case["case_id"]
    out = {
        "case_id": case_id,
        "edit_type": case.get("edit_type", ""),
        "target_entity": case.get("target_entity", ""),
        "instruction": case.get("prompt", ""),
        "source_video": source_path,
        "edited_video": edited_path,
        "frame_scores": [],
        "average": None,
        "dimension_averages": {},
        "error": None,
    }

    if not os.path.exists(edited_path):
        out["error"] = f"missing edited video: {edited_path}"; return out
    if not os.path.exists(source_path):
        out["error"] = f"missing source video: {source_path}"; return out

    edited_count = get_video_frame_count(edited_path)
    source_count = get_video_frame_count(source_path)
    if edited_count == 0:
        out["error"] = "edited video has 0 frames"; return out

    indices = sample_frame_indices(edited_count, num_frames)
    prompt_text = _build_prompt_text(case, with_search=(api == "gemini"))

    frame_results = []
    for fi in indices:
        src_fi = min(fi, source_count - 1) if source_count > 0 else 0
        sf = read_frame(source_path, src_fi); ef = read_frame(edited_path, fi)
        if sf is None or ef is None:
            continue
        if api == "openai":
            before_b64 = frame_to_jpeg_b64(sf); after_b64 = frame_to_jpeg_b64(ef)
            txt = call_openai(openai_client, model, prompt_text,
                              before_b64, after_b64, max_retries=max_retries)
        else:
            before_b = frame_to_jpeg_bytes(sf); after_b = frame_to_jpeg_bytes(ef)
            txt = call_gemini_with_search(gemini_client, model, prompt_text,
                                          before_b, after_b, max_retries=max_retries)
        if txt is None:
            continue
        parsed = parse_scores(txt)
        parsed["frame_index"] = int(fi)
        frame_results.append(parsed)

    out["frame_scores"] = frame_results
    rel = relevant_axes(case.get("edit_type", ""))
    out["axes_used"] = rel
    out["total_max"] = total_max(case.get("edit_type", ""))
    if frame_results:
        per_frame_totals = []
        for d in frame_results:
            vals = [d.get(a) for a in rel]
            if all(v is not None for v in vals):
                per_frame_totals.append(sum(vals))
        if per_frame_totals:
            avg_total = sum(per_frame_totals) / len(per_frame_totals)
            out["average"] = round(avg_total, 4)
            out["fair_fraction"] = round(avg_total / out["total_max"], 4)
            out["fair_total_on_21_scale"] = round(out["fair_fraction"] * 21, 4)
        for axis in RUBRIC_AXES:
            if axis not in rel:
                out["dimension_averages"][axis] = None
                continue
            vals = [d.get(axis) for d in frame_results if d.get(axis) is not None]
            out["dimension_averages"][axis] = (
                round(sum(vals) / len(vals), 4) if vals else None)

    return out


def main():
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="agent-bench scoring via an OpenAI-compatible VLM judge "
                    "(configure --model/--base_url/--api_key); --api gemini uses "
                    "the public Google genai backend.")
    parser.add_argument("--results_dir", required=True)
    parser.add_argument("--bench_dir", default=os.environ.get(
        "AURORA_BENCH_DIR", str(repo_root / "benchmarks" / "agent_bench")
    ))
    parser.add_argument("--output", default=None)
    parser.add_argument("--api", choices=["openai", "gemini"], default="openai",
                        help="'openai': OpenAI-compatible endpoint; "
                             "'gemini': public Google genai backend.")
    parser.add_argument("--model", default=None,
                        help="VLM judge model. Defaults: 'gpt-4o' for the "
                             "OpenAI-compatible path, 'gemini-2.5-pro' for --api gemini.")
    parser.add_argument("--api_key", default=None)
    parser.add_argument("--base_url", default=os.environ.get("OPENAI_BASE_URL", ""))
    parser.add_argument("--num_workers", type=int, default=16)
    parser.add_argument("--num_frames", type=int, default=3)
    parser.add_argument("--case_ids", default=None)
    parser.add_argument("--max_retries", type=int, default=4)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    if args.model is None:
        args.model = "gemini-2.5-pro" if args.api == "gemini" else "gpt-4o"

    bench_dir = Path(args.bench_dir).resolve()
    results_dir = Path(args.results_dir).resolve()
    output = Path(args.output) if args.output else results_dir / "scores.json"

    cases_json = bench_dir / "cases.json"
    if not cases_json.exists():
        raise FileNotFoundError(f"Missing {cases_json}")
    cases_full = {c["case_id"]: c for c in json.loads(cases_json.read_text())}

    inf_json = results_dir / "inference_results.json"
    if inf_json.exists():
        inf = json.loads(inf_json.read_text())
        case_ids = [r["case_id"] for r in inf if r.get("success")]
    else:
        case_ids = sorted([p.parent.name for p in results_dir.glob("*/generate.mp4")])

    if args.case_ids:
        wanted = set(s.strip() for s in args.case_ids.split(",") if s.strip())
        case_ids = [c for c in case_ids if c in wanted]

    cases = []
    for cid in case_ids:
        if cid not in cases_full:
            print(f"WARN: {cid} not in cases.json, skipping")
            continue
        c = dict(cases_full[cid])
        aligned_source = results_dir / cid / "source.mp4"
        c["__source_path"] = str(aligned_source if aligned_source.exists() else bench_dir / c["src_video"])
        c["__edited_path"] = str(results_dir / cid / "generate.mp4")
        cases.append(c)
    print(f"will score {len(cases)} cases")

    existing: dict[str, dict] = {}
    if args.resume and output.exists():
        old = json.loads(output.read_text())
        for r in old.get("per_case", []):
            if r.get("error") is None and r.get("average") is not None:
                existing[r["case_id"]] = r
        cases = [c for c in cases if c["case_id"] not in existing]
        print(f"resume: keeping {len(existing)} existing, scoring {len(cases)} new")

    openai_client = None
    gemini_client = None
    if args.api == "openai":
        api_key = args.api_key or os.environ.get("OPENAI_API_KEY", "")
        if not api_key:
            raise SystemExit("Set OPENAI_API_KEY or pass --api_key for --api openai")
        if not args.base_url:
            raise SystemExit("Set OPENAI_BASE_URL or pass --base_url for --api openai")
        from openai import OpenAI
        openai_client = OpenAI(base_url=args.base_url, api_key=api_key)
    else:
        api_key = args.api_key or os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise SystemExit("Set GEMINI_API_KEY or pass --api_key for --api gemini.")
        gemini_client = _gemini_client(api_key)

    per_case: list[dict] = list(existing.values())
    if cases:
        with ThreadPoolExecutor(max_workers=args.num_workers) as pool:
            futures = {
                pool.submit(score_one_case, c,
                            source_path=c["__source_path"],
                            edited_path=c["__edited_path"],
                            model=args.model, api=args.api,
                            num_frames=args.num_frames,
                            openai_client=openai_client,
                            gemini_client=gemini_client,
                            max_retries=args.max_retries): c["case_id"]
                for c in cases
            }
            for fut in as_completed(futures):
                cid = futures[fut]
                try:
                    res = fut.result()
                    per_case.append(res)
                    avg = res.get("average")
                    dims = res.get("dimension_averages", {})
                    print(f"  [{cid}] avg={avg} dims={dims} err={res.get('error')}")
                except Exception as e:
                    print(f"  [{cid}] EXC {e}")
                    per_case.append({"case_id": cid, "error": str(e)})

    per_case.sort(key=lambda r: r["case_id"])

    valid = [r for r in per_case if r.get("average") is not None]
    overall_native = round(sum(r["average"] for r in valid) / max(1, len(valid)), 4)
    fair_fracs = [r.get("fair_fraction") for r in valid if r.get("fair_fraction") is not None]
    overall_pct = round((sum(fair_fracs) / len(fair_fracs)) if fair_fracs else 0.0, 4)
    overall_fair_frac = overall_pct
    overall_fair_on_21 = round(overall_pct * 21, 4)

    dim_aggs: dict = {}
    for axis in RUBRIC_AXES:
        vs = [r["dimension_averages"].get(axis) for r in valid
              if axis in r.get("axes_used", RUBRIC_AXES)
              and r.get("dimension_averages", {}).get(axis) is not None]
        dim_aggs[axis] = {
            "n": len(vs),
            "avg": round(sum(vs) / len(vs), 4) if vs else None,
        }

    by_type: dict[str, dict] = {}
    for r in valid:
        et = r.get("edit_type", "unknown")
        by_type.setdefault(et, {"recs": []})["recs"].append(r)
    type_summary = {}
    for et, d in by_type.items():
        recs = d["recs"]
        n = len(recs)
        avgs = [r["average"] for r in recs]
        ffracs = [r.get("fair_fraction") for r in recs if r.get("fair_fraction") is not None]
        rel = relevant_axes(et)
        tmax = total_max(et)
        avg_pct = round(sum(ffracs) / n, 4) if ffracs else None
        type_summary[et] = {
            "n": n,
            "axes_used": rel,
            "total_max": tmax,
            "avg_total_native": round(sum(avgs) / n, 4),
            "avg_pct_on_1": avg_pct,
            "avg_fair_fraction": avg_pct,
            "avg_fair_on_21_scale": round(avg_pct * 21, 4) if avg_pct is not None else None,
            **{f"avg_{a}": (
                round(sum(r["dimension_averages"][a] for r in recs
                          if r["dimension_averages"].get(a) is not None) /
                      max(1, sum(1 for r in recs if r["dimension_averages"].get(a) is not None)), 4)
                if a in rel and any(r["dimension_averages"].get(a) is not None for r in recs)
                else None
            ) for a in RUBRIC_AXES},
        }

    summary = {
        "n_cases_total": len(per_case),
        "n_cases_valid": len(valid),
        "overall_pct_on_1": overall_pct,
        "overall_fair_on_21_scale": overall_fair_on_21,
        "overall_avg_total_native": overall_native,
        "overall_fair_fraction": overall_pct,
        "overall_avg_total": overall_fair_on_21,
        "dimension_averages": dim_aggs,
        "by_edit_type": type_summary,
        "non_ip_edit_types": sorted(NON_IP_EDIT_TYPES),
        "ip_axes": sorted(IP_AXES),
        "config": {
            "api": args.api,
            "model": args.model,
            "num_frames_per_case": args.num_frames,
        },
    }
    output.write_text(json.dumps({"summary": summary, "per_case": per_case},
                                 indent=2, ensure_ascii=False))
    print(f"\nOverall: {summary}")
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
