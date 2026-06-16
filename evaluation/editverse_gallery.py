"""Generate an HTML gallery for EditVerse benchmark results.

Supports comparing multiple methods side-by-side. Each row shows:
- Source video + reference image (if present)
- Editing prompt + edit type
- Edited videos from each method
- VLM and auto metric scores

Videos are re-encoded to base64 for self-contained offline viewing.
"""

import argparse
import base64
import csv
import json
import os
import subprocess
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def _video_to_base64(video_path: str, target_fps: int = 24) -> str | None:
    """Re-encode video to compact mp4 and return base64 string."""
    if not os.path.exists(video_path):
        return None
    try:
        result = subprocess.run(
            ["ffmpeg", "-y", "-i", video_path,
             "-vf", f"fps={target_fps},scale='min(480,iw)':'-2'",
             "-c:v", "libx264", "-preset", "ultrafast", "-crf", "28",
             "-an", "-movflags", "+faststart", "-f", "mp4", "pipe:1"],
            capture_output=True, timeout=30,
        )
        if result.returncode == 0 and result.stdout:
            return base64.b64encode(result.stdout).decode("utf-8")
    except Exception:
        pass
    try:
        with open(video_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    except Exception:
        return None


def _image_to_base64(image_path: str, max_width: int = 320) -> str | None:
    """Encode image as base64 JPEG, optionally resizing."""
    if not os.path.exists(image_path):
        return None
    try:
        from PIL import Image
        import io
        img = Image.open(image_path).convert("RGB")
        if img.width > max_width:
            ratio = max_width / img.width
            img = img.resize((max_width, int(img.height * ratio)), Image.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=85)
        return base64.b64encode(buf.getvalue()).decode("utf-8")
    except Exception:
        return None


def load_scores(scores_json: str) -> dict[str, dict]:
    """Load scores.json and return {bench_id: sample_dict}."""
    with open(scores_json, "r") as f:
        data = json.load(f)
    return {s["bench_id"]: s for s in data.get("samples", [])}


def load_auto_metrics(csv_path: str) -> dict:
    """Load auto_metrics.csv and return {averages: {...}, per_sample: {bench_id: {...}}}."""
    if not os.path.exists(csv_path):
        return {}
    rows = []
    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        metric_cols = [c for c in reader.fieldnames if c not in ("id", "type")]
        for row in reader:
            rows.append(row)
    if not rows:
        return {}
    averages = {}
    for col in metric_cols:
        vals = [float(r[col]) for r in rows if r.get(col)]
        averages[col] = round(sum(vals) / len(vals), 4) if vals else None
    per_sample = {r["id"]: {c: float(r[c]) for c in metric_cols if r.get(c)} for r in rows}
    return {"averages": averages, "per_sample": per_sample, "metric_cols": metric_cols}


AUTO_SHORT_NAMES = {
    "clip_temporal_consistency": "CLIP TC",
    "dino_temporal_consistency": "DINO TC",
    "frame_text_alignment": "Frame-Text",
    "video_text_alignment": "Video-Text",
    "pick_score_video_quality": "PickScore",
}

HIGHER_BETTER = {"frame_text_alignment", "video_text_alignment", "pick_score_video_quality",
                 "clip_temporal_consistency", "dino_temporal_consistency"}


def _html_escape(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")


def generate_gallery(
    bench_dir: str,
    methods: dict[str, dict],
    output_path: str,
    title: str = "EditVerse Benchmark Gallery",
):
    bench_dir = Path(bench_dir)
    bench_json = bench_dir / "EditVerseBench_3.json"
    with open(bench_json, "r") as f:
        bench_data = json.load(f)

    by_type = defaultdict(list)
    for bench_id, entry in sorted(bench_data.items(), key=lambda x: int(x[0])):
        by_type[entry.get("type", "unknown")].append((bench_id, entry))

    method_names = list(methods.keys())
    per_case_method_names = [m for m in method_names if not methods[m].get("summary_only")]
    num_methods = len(per_case_method_names)

    html_parts = [f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<title>{_html_escape(title)}</title>
<style>
:root {{
  --border: #d0d7de;
  --bg-card: #ffffff;
  --bg-page: #f6f8fa;
  --bg-header: #f0f3f6;
  --text-primary: #1f2328;
  --text-secondary: #656d76;
  --accent: #0969da;
  --best-bg: #ddf4ff;
}}
* {{ box-sizing: border-box; margin: 0; padding: 0; }}
body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
       background: var(--bg-page); color: var(--text-primary); padding: 24px; }}
h1 {{ font-size: 24px; margin-bottom: 8px; }}
h1 small {{ font-size: 14px; color: var(--text-secondary); font-weight: normal; }}
h2 {{ font-size: 18px; color: var(--text-primary); margin: 32px 0 12px;
     padding-bottom: 8px; border-bottom: 2px solid var(--border); }}

/* TOC */
.toc {{ margin: 16px 0 24px; display: flex; flex-wrap: wrap; gap: 6px; }}
.toc a {{ display: inline-block; padding: 4px 12px; background: var(--bg-card);
          border: 1px solid var(--border); border-radius: 16px; text-decoration: none;
          color: var(--accent); font-size: 13px; }}
.toc a:hover {{ background: var(--best-bg); }}

/* Summary tables */
.summary {{ margin: 16px 0; overflow-x: auto; }}
.summary table {{ border-collapse: collapse; font-size: 13px; width: auto; }}
.summary th, .summary td {{ border: 1px solid var(--border); padding: 6px 14px; text-align: center; white-space: nowrap; }}
.summary th {{ background: var(--bg-header); font-weight: 600; }}
.summary td:first-child {{ text-align: left; font-weight: 500; }}
.summary .best {{ background: var(--best-bg); font-weight: 700; color: var(--accent); }}

/* Sample card — uses CSS grid for uniform column widths */
.sample {{
  display: grid;
  grid-template-columns: 280px repeat({num_methods}, 1fr);
  gap: 0;
  margin: 12px 0;
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-radius: 8px;
  overflow: hidden;
}}
.sample .cell {{
  padding: 12px;
  border-right: 1px solid var(--border);
  display: flex;
  flex-direction: column;
  align-items: center;
}}
.sample .cell:last-child {{ border-right: none; }}

/* Info cell (first column) */
.sample .info-cell {{
  padding: 12px;
  border-right: 1px solid var(--border);
  display: flex;
  flex-direction: column;
  gap: 8px;
}}
.info-cell .header {{ display: flex; align-items: center; gap: 8px; flex-wrap: wrap; }}
.info-cell .type-tag {{
  display: inline-block; background: #ddf4ff; color: #0969da;
  padding: 2px 8px; border-radius: 12px; font-size: 11px; font-weight: 600;
}}
.info-cell .bench-id {{ font-weight: 700; font-size: 15px; }}
.info-cell .prompt {{ font-size: 12px; color: var(--text-secondary); line-height: 1.5; }}
.info-cell .input-media {{ display: flex; flex-direction: column; gap: 6px; margin-top: 4px; }}
.info-cell .input-media .media-label {{ font-size: 11px; color: var(--text-secondary); font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px; }}
.info-cell video {{ width: 100%; max-width: 256px; border-radius: 4px; border: 1px solid var(--border); }}
.info-cell img {{ max-width: 180px; max-height: 120px; border-radius: 4px; border: 1px solid var(--border); object-fit: contain; }}

/* Method cell */
.cell .method-label {{ font-size: 12px; font-weight: 600; color: var(--text-secondary); margin-bottom: 6px; text-align: center; }}
.cell video {{ width: 100%; max-width: 320px; border-radius: 4px; border: 1px solid var(--border); }}
.cell .score-line {{ font-size: 12px; margin-top: 6px; color: var(--text-secondary); }}
.cell .score-line .val {{ font-weight: 600; color: var(--text-primary); }}
.cell .dim-line {{ font-size: 11px; margin-top: 2px; color: var(--text-secondary); text-align: center; }}
.cell .dim-line code {{ background: var(--bg-header); padding: 1px 5px; border-radius: 3px; color: var(--text-primary); }}
.cell .na {{ color: #cf222e; font-size: 13px; margin-top: 20px; }}
.cell details.judge {{ width: 100%; margin-top: 8px; font-size: 12px; }}
.cell details.judge > summary {{ cursor: pointer; color: var(--accent); font-size: 11px;
                                   padding: 2px 0; list-style: none; user-select: none; }}
.cell details.judge > summary::-webkit-details-marker {{ display: none; }}
.cell details.judge > summary::before {{ content: "▸ "; display: inline-block; transition: transform 0.1s; }}
.cell details.judge[open] > summary::before {{ content: "▾ "; }}
.cell details.judge .frame-block {{ background: var(--bg-header); border: 1px solid var(--border);
                                     border-radius: 4px; padding: 8px 10px; margin-top: 6px; }}
.cell details.judge .frame-block .fb-head {{ font-size: 11px; color: var(--text-secondary);
                                               font-weight: 600; margin-bottom: 4px; }}
.cell details.judge .frame-block pre {{ font-family: inherit; white-space: pre-wrap; word-wrap: break-word;
                                          font-size: 11.5px; line-height: 1.5; color: var(--text-primary);
                                          margin: 0; background: transparent; border: none; padding: 0; }}

/* Rubric legend */
.legend {{ margin: 16px 0 24px; background: var(--bg-card); border: 1px solid var(--border);
           border-radius: 8px; padding: 14px 18px; font-size: 13px; line-height: 1.55; }}
.legend h3 {{ font-size: 14px; margin-bottom: 8px; color: var(--text-primary); }}
.legend ul {{ list-style: none; padding-left: 0; }}
.legend li {{ margin-bottom: 6px; }}
.legend li strong {{ color: var(--accent); }}
.legend .note {{ color: var(--text-secondary); font-size: 12px; margin-top: 6px; }}
</style></head><body>
<h1>{_html_escape(title)} <small>{len(bench_data)} samples, {len(by_type)} edit types, {num_methods} methods</small></h1>

<div class="legend">
  <h3>VLM judge — scoring rubric</h3>
  <ul>
    <li><strong>Prompt Following (0-3)</strong>: does the edit accurately and completely fulfill the editing instruction?
        <em>3</em>=perfectly followed, <em>2</em>=mostly followed with minor issues,
        <em>1</em>=poorly / partially executed, <em>0</em>=ignored or opposite.</li>
    <li><strong>Edit Quality (0-3)</strong>: visual quality of the edited area itself — realism, seamlessness, absence of artifacts.
        <em>3</em>=seamless and artifact-free, <em>2</em>=minor artifacts,
        <em>1</em>=significant artifacts, <em>0</em>=garbled or failed.</li>
    <li><strong>Background Consistency (0-3)</strong>: are unedited regions preserved unchanged between source and edit?
        <em>3</em>=perfectly preserved, <em>2</em>=minor/subtle changes,
        <em>1</em>=significant distracting changes, <em>0</em>=catastrophic drift.</li>
    <li><strong>Total (0-9)</strong>: sum of the three dimension scores. Per-sample score = mean over 3 sampled frames.</li>
  </ul>
  <div class="note">Click &quot;Show VLM justification&quot; under each edit to expand the full untruncated judge text for each sampled frame.</div>
</div>
"""]

    html_parts.append('<div class="toc">')
    for etype in by_type:
        anchor = etype.replace(" ", "_")
        html_parts.append(f'<a href="#{anchor}">{etype} ({len(by_type[etype])})</a>')
    html_parts.append('</div>')

    summary_rows = []
    for mname in method_names:
        scores = methods[mname].get("scores_data", {})
        overall = scores.get("overall_average")
        type_avgs = scores.get("type_averages", {})
        summary_rows.append((mname, overall, type_avgs))

    has_vlm = any(o is not None for _, o, _ in summary_rows)
    all_types = list(by_type.keys())

    if has_vlm:
        best_overall = None
        best_by_type: dict[str, float] = {}
        for _, overall, type_avgs in summary_rows:
            if overall is not None:
                if best_overall is None or overall > best_overall:
                    best_overall = overall
            for t in all_types:
                v = type_avgs.get(t)
                if v is not None and (t not in best_by_type or v > best_by_type[t]):
                    best_by_type[t] = v

        judge_models = sorted({s.get("scores_data", {}).get("model", "") for s in methods.values() if s.get("scores_data")})
        judge_label = ", ".join(m for m in judge_models if m) or "unknown judge"
        html_parts.append(f'<h2>VLM Scores ({judge_label}, 0-9 scale)</h2>')
        html_parts.append('<div class="summary"><table><tr><th>Method</th><th>Overall</th>')
        for t in all_types:
            html_parts.append(f'<th>{t}</th>')
        html_parts.append('</tr>')
        for mname, overall, type_avgs in summary_rows:
            cls_o = ' class="best"' if overall is not None and overall == best_overall else ""
            html_parts.append(f'<tr><td>{mname}</td><td{cls_o}>{overall if overall is not None else "N/A"}</td>')
            for t in all_types:
                v = type_avgs.get(t)
                cls = ' class="best"' if v is not None and v == best_by_type.get(t) else ""
                html_parts.append(f'<td{cls}>{v if v is not None else "N/A"}</td>')
            html_parts.append('</tr>')
        html_parts.append('</table></div>')

    auto_methods = [(mn, methods[mn].get("auto_metrics", {})) for mn in method_names]
    has_auto = any(am for _, am in auto_methods)
    if has_auto:
        all_metric_cols = []
        for _, am in auto_methods:
            for c in am.get("metric_cols", []):
                if c not in all_metric_cols:
                    all_metric_cols.append(c)
        best_auto: dict[str, float] = {}
        for c in all_metric_cols:
            vals = []
            for _, am in auto_methods:
                v = (am.get("averages") or {}).get(c)
                if v is not None:
                    vals.append(v)
            if vals:
                best_auto[c] = max(vals) if c in HIGHER_BETTER else min(vals)

        html_parts.append('<h2>Automated Metrics (averaged over all samples)</h2>')
        html_parts.append('<div class="summary"><table><tr><th>Method</th>')
        for c in all_metric_cols:
            html_parts.append(f'<th>{AUTO_SHORT_NAMES.get(c, c)}</th>')
        html_parts.append('</tr>')
        for mname, am in auto_methods:
            avgs = am.get("averages", {})
            html_parts.append(f'<tr><td>{mname}</td>')
            for c in all_metric_cols:
                v = avgs.get(c)
                if v is None:
                    html_parts.append('<td>N/A</td>')
                else:
                    cls = ' class="best"' if v == best_auto.get(c) else ""
                    html_parts.append(f'<td{cls}>{v:.4f}</td>')
            html_parts.append('</tr>')
        html_parts.append('</table></div>')

    for etype, items in by_type.items():
        anchor = etype.replace(" ", "_")
        html_parts.append(f'<h2 id="{anchor}">{etype} ({len(items)} samples)</h2>')

        for bench_id, entry in items:
            prompt = entry.get("<text>", "").replace("<video1>", "").replace("<image1>", "").strip()
            prompt_escaped = _html_escape(prompt)

            source_path = str(bench_dir / "baselines" / "EditVerse" / bench_id / "video1.mp4")
            if not os.path.exists(source_path):
                video_rel = entry.get("<video1>", "")
                source_path = str(bench_dir / "source_videos" / os.path.basename(video_rel))

            ref_image_rel = entry.get("<image1>", "")
            ref_image_path = None
            if ref_image_rel:
                ref_image_path = str(bench_dir / "source_images" / os.path.basename(ref_image_rel))

            html_parts.append('<div class="sample">')

            html_parts.append('<div class="info-cell">')
            html_parts.append(f'<div class="header"><span class="type-tag">{etype}</span>'
                              f'<span class="bench-id">#{bench_id}</span></div>')
            html_parts.append(f'<div class="prompt">{prompt_escaped}</div>')
            html_parts.append('<div class="input-media">')

            src_b64 = _video_to_base64(source_path)
            if src_b64:
                html_parts.append(
                    f'<div><div class="media-label">Source Video</div>'
                    f'<video src="data:video/mp4;base64,{src_b64}" '
                    f'controls muted loop playsinline></video></div>')

            if ref_image_path:
                ref_b64 = _image_to_base64(ref_image_path)
                if ref_b64:
                    html_parts.append(
                        f'<div><div class="media-label">Reference Image</div>'
                        f'<img src="data:image/jpeg;base64,{ref_b64}" /></div>')

            html_parts.append('</div>')
            html_parts.append('</div>')

            for mname in per_case_method_names:
                minfo = methods[mname]
                results_dir = minfo["results_dir"]
                scores_map = minfo.get("scores_map", {})

                html_parts.append('<div class="cell">')
                html_parts.append(f'<div class="method-label">{_html_escape(mname)}</div>')

                edited_flat = os.path.join(results_dir, f"{bench_id}.mp4")
                edited_nested = os.path.join(results_dir, bench_id, "generate.mp4")
                if os.path.exists(edited_flat):
                    edited_path = edited_flat
                elif os.path.exists(edited_nested):
                    edited_path = edited_nested
                else:
                    html_parts.append('<div class="na">N/A</div></div>')
                    continue

                vid_b64 = _video_to_base64(edited_path)
                if vid_b64:
                    html_parts.append(
                        f'<video src="data:video/mp4;base64,{vid_b64}" '
                        f'controls muted loop playsinline></video>')

                score_info = scores_map.get(bench_id, {})
                avg = score_info.get("average")
                frame_scores = score_info.get("frame_scores", [])
                frame_details = score_info.get("frame_details", [])
                dim_avgs = score_info.get("dimension_averages", {}) or {}
                if avg is not None:
                    fs_str = ", ".join(str(s) for s in frame_scores)
                    html_parts.append(
                        f'<div class="score-line">VLM: <span class="val">{avg}</span>'
                        f' <small>({fs_str})</small></div>')
                    def _fmt_d(v):
                        return f"{v:.2f}" if isinstance(v, (int, float)) else "—"
                    pf = _fmt_d(dim_avgs.get("prompt_following"))
                    eq = _fmt_d(dim_avgs.get("edit_quality"))
                    bc = _fmt_d(dim_avgs.get("background_consistency"))
                    html_parts.append(
                        '<div class="dim-line">'
                        f'PF <code>{pf}</code> · EQ <code>{eq}</code> · BG <code>{bc}</code>'
                        '</div>'
                    )

                if frame_details:
                    html_parts.append('<details class="judge">')
                    html_parts.append('<summary>Show VLM justification</summary>')
                    for i, fd in enumerate(frame_details):
                        fi = fd.get("frame_index")
                        total = fd.get("total")
                        pf_i = fd.get("prompt_following")
                        eq_i = fd.get("edit_quality")
                        bc_i = fd.get("background_consistency")
                        just = fd.get("justification") or ""
                        head_bits = [f"frame {i + 1}"]
                        if fi is not None:
                            head_bits.append(f"idx {fi}")
                        if total is not None:
                            head_bits.append(f"total {total}/9")
                        parts_dim = []
                        if pf_i is not None: parts_dim.append(f"PF {pf_i}")
                        if eq_i is not None: parts_dim.append(f"EQ {eq_i}")
                        if bc_i is not None: parts_dim.append(f"BG {bc_i}")
                        if parts_dim:
                            head_bits.append(" · ".join(parts_dim))
                        head_str = _html_escape(" | ".join(head_bits))
                        html_parts.append('<div class="frame-block">')
                        html_parts.append(f'<div class="fb-head">{head_str}</div>')
                        html_parts.append(f'<pre>{_html_escape(just)}</pre>')
                        html_parts.append('</div>')
                    html_parts.append('</details>')

                html_parts.append('</div>')

            html_parts.append('</div>')

    html_parts.append('</body></html>')

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(html_parts))
    print(f"Gallery written to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="EditVerse gallery generator")
    parser.add_argument("--bench_dir", default=os.environ.get(
        "AURORA_BENCH_DIR", str(REPO_ROOT / "benchmarks" / "editverse")
    ))
    parser.add_argument("--methods", required=True, nargs="+",
                        help="Method specs: name:results_dir[:scores.json] ...")
    parser.add_argument("--output", default=os.path.join(
        os.environ.get("AURORA_OUTPUT_DIR", str(REPO_ROOT / "outputs")), "gallery", "editverse_gallery.html"
    ))
    parser.add_argument("--title", default="EditVerse Benchmark Gallery")
    args = parser.parse_args()

    methods = {}
    for spec in args.methods:
        parts = spec.split(":")
        if len(parts) < 2:
            raise ValueError(f"Method spec must be name:results_dir[:scores.json], got: {spec}")
        mname = parts[0]
        results_dir = parts[1]
        scores_json = parts[2] if len(parts) > 2 else os.path.join(results_dir, "scores.json")
        summary_only = not results_dir

        minfo = {
            "results_dir": results_dir,
            "scores_map": {},
            "scores_data": {},
            "summary_only": summary_only,
        }
        if os.path.exists(scores_json):
            with open(scores_json, "r") as f:
                scores_data = json.load(f)
            minfo["scores_data"] = scores_data
            minfo["scores_map"] = {s["bench_id"]: s for s in scores_data.get("samples", [])}
        auto_csv = os.path.join(results_dir, "auto_metrics.csv") if results_dir else ""
        minfo["auto_metrics"] = load_auto_metrics(auto_csv) if auto_csv else {}
        methods[mname] = minfo

    generate_gallery(args.bench_dir, methods, args.output, args.title)


if __name__ == "__main__":
    main()
