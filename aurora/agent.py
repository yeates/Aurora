"""Run the Aurora agent VLM in front of EditVerse-style video editing inputs.

The script is intentionally inspection-first:
  1. load EditVerse video / instruction / optional reference image,
  2. ask the trained agent VLM for the Type-1 planning JSON,
  3. optionally call image search + Type-3 image selection,
  4. optionally build a text-conditioned object mask,
  5. write JSONL plus an HTML gallery for manual inspection.

It does not run the downstream video editor by default. The emitted
``final_payload`` is the contract that should be passed to video editing.
"""

from __future__ import annotations

import argparse
import html
import json
import os
import random
import re
import subprocess
import sys
import tempfile
import traceback
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = Path(os.environ.get("AURORA_OUTPUT_DIR", str(REPO_ROOT / "outputs")))
BENCH_DIR = Path(os.environ.get("AURORA_BENCH_DIR", str(REPO_ROOT / "benchmarks")))
MODEL_DIR = Path(os.environ.get("AURORA_MODEL_DIR", str(REPO_ROOT.parent / "models")))
_PROMPT_DIR = Path(__file__).resolve().parent / "prompts"
TYPE1_SYSTEM = (_PROMPT_DIR / "type1_system.txt").read_text()
TYPE3_SYSTEM = (_PROMPT_DIR / "type3_system.txt").read_text()
DEFAULT_AGENT_BASE = MODEL_DIR / "Qwen3-VL-8B-Instruct"
DEFAULT_AGENT_ADAPTER = MODEL_DIR / "aurora_agent_vlm"
DEFAULT_BENCH_DIR = BENCH_DIR / "editverse"
DEFAULT_OUT_DIR = OUTPUT_DIR / "agent"


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _safe_name(text: str) -> str:
    out = re.sub(r"[^0-9A-Za-z_.-]+", "_", str(text).strip())
    return out[:100] or "item"


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _boolish_false(value: Any) -> bool:
    if value is False or value is None:
        return True
    if isinstance(value, str) and value.strip().lower() in {"", "false", "none", "null", "no"}:
        return True
    return False


def _as_query(value: Any) -> str | None:
    if _boolish_false(value):
        return None
    return str(value).strip()


def load_editverse_cases(bench_dir: Path, bench_ids: set[str] | None = None) -> list[dict[str, Any]]:
    data = _read_json(bench_dir / "EditVerseBench_3.json")
    cases: list[dict[str, Any]] = []
    for bench_id, entry in sorted(data.items(), key=lambda kv: int(kv[0])):
        if bench_ids and bench_id not in bench_ids:
            continue
        video_rel = entry.get("<video1>", "")
        image_rel = entry.get("<image1>", "")
        prompt = (entry.get("<text>", "") or "").replace("<video1>", "").replace("<image1>", "").strip()
        cases.append(
            {
                "bench_id": bench_id,
                "edit_type": entry.get("type", "unknown"),
                "prompt": prompt,
                "video_path": str(bench_dir / "source_videos" / Path(video_rel).name),
                "ref_image_path": str(bench_dir / "source_images" / Path(image_rel).name) if image_rel else "",
                "target_prompt": entry.get("target_prompt", ""),
                "source_prompt": entry.get("source_prompt", ""),
                "direction": entry.get("direction", ""),
            }
        )
    return cases


def load_custom_cases(path: Path) -> list[dict[str, Any]]:
    """Load JSONL cases with explicit video/prompt fields for branch probes."""
    cases: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            bench_id = str(item.get("bench_id") or item.get("id") or f"custom_{line_no}")
            prompt = str(item["prompt"]).strip()
            video_path = str(item["video_path"])
            ref_image_path = str(item.get("ref_image_path") or "")
            cases.append(
                {
                    "bench_id": bench_id,
                    "edit_type": item.get("edit_type", "custom"),
                    "prompt": prompt,
                    "video_path": video_path,
                    "ref_image_path": ref_image_path,
                    "target_prompt": item.get("target_prompt", ""),
                    "source_prompt": item.get("source_prompt", ""),
                    "direction": item.get("direction", ""),
                }
            )
    return cases


def sample_video_frames(video_path: str, max_frames: int, max_side: int) -> tuple[list[Image.Image], float]:
    frames, fps = sample_video_frames_ffmpeg(video_path, max_frames=max_frames, max_side=max_side)
    if frames:
        return frames, fps

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 24.0)
    if total <= 0:
        total = max_frames
    indices = np.linspace(0, max(total - 1, 0), num=min(max_frames, total), dtype=int)
    frames: list[Image.Image] = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if not ok:
            continue
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb)
        img.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
        frames.append(img.convert("RGB"))
    cap.release()
    if not frames:
        raise RuntimeError(f"No frames decoded from video: {video_path}")
    return frames, fps


def sample_video_frames_ffmpeg(video_path: str, max_frames: int, max_side: int) -> tuple[list[Image.Image], float]:
    try:
        probe = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=nb_frames,r_frame_rate",
                "-of",
                "json",
                video_path,
            ],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        stream = (json.loads(probe.stdout).get("streams") or [{}])[0]
        fps_expr = stream.get("r_frame_rate") or "24/1"
        num, den = fps_expr.split("/") if "/" in fps_expr else (fps_expr, "1")
        fps = float(num) / max(float(den), 1.0)
        total = int(stream.get("nb_frames") or 0)
        if total <= 0:
            total = max_frames
        indices = np.linspace(0, max(total - 1, 0), num=min(max_frames, total), dtype=int)
        select_expr = "+".join(f"eq(n\\,{int(i)})" for i in indices)
        with tempfile.TemporaryDirectory(prefix="aurora_frames_") as tmp:
            pattern = str(Path(tmp) / "frame_%03d.jpg")
            subprocess.run(
                ["ffmpeg", "-y", "-v", "error", "-i", video_path, "-vf", f"select='{select_expr}'", "-vsync", "0", pattern],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            frames = []
            for frame_path in sorted(Path(tmp).glob("frame_*.jpg")):
                img = Image.open(frame_path).convert("RGB")
                img.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
                frames.append(img)
        return frames, fps
    except Exception:
        return [], 24.0


def load_ref_image(path: str, max_side: int) -> Image.Image | None:
    if not path or not os.path.exists(path):
        return None
    img = Image.open(path).convert("RGB")
    img.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
    return img


def make_contact_sheet(images: list[Image.Image], columns: int = 4, pad: int = 6) -> Image.Image:
    if not images:
        return Image.new("RGB", (256, 144), "white")
    w = max(img.width for img in images)
    h = max(img.height for img in images)
    rows = (len(images) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * w + (columns + 1) * pad, rows * h + (rows + 1) * pad), (245, 245, 245))
    for i, img in enumerate(images):
        x = pad + (i % columns) * (w + pad)
        y = pad + (i // columns) * (h + pad)
        sheet.paste(img, (x, y))
    return sheet


def save_jpeg(image: Image.Image, path: Path, quality: int = 88) -> str:
    _ensure_dir(path.parent)
    image.convert("RGB").save(path, "JPEG", quality=quality)
    return str(path)


def save_png(image: Image.Image, path: Path) -> str:
    _ensure_dir(path.parent)
    image.save(path)
    return str(path)


def parse_json_object(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError(f"No JSON object found in model output: {text[:300]}")
    return json.loads(cleaned[start : end + 1])


def normalize_plan(raw: dict[str, Any]) -> dict[str, Any]:
    allowed = {
        "global_style",
        "remove_object",
        "add_object",
        "replace_object",
        "change_background",
        "change_color",
        "change_weather",
        "add_effect",
        "customization",
        "combined_tasks",
        "camera_edit",
    }
    plan = {
        "refined_text_instruction": str(raw.get("refined_text_instruction", "")).strip(),
        "subtask": str(raw.get("subtask", "")).strip(),
        "image_search": raw.get("image_search", False),
        "mask": raw.get("mask", False),
    }
    if plan["subtask"] not in allowed:
        plan["subtask"] = "combined_tasks"
    if _boolish_false(plan["image_search"]):
        plan["image_search"] = False
    else:
        plan["image_search"] = str(plan["image_search"]).strip()
    if _boolish_false(plan["mask"]):
        plan["mask"] = False
    else:
        plan["mask"] = str(plan["mask"]).strip()
    return plan


@dataclass
class PreparedVideo:
    frames: list[Image.Image]
    fps: float


class AgentVLM:
    def __init__(
        self,
        base_model: Path,
        adapter_path: Path,
        *,
        device: str = "cuda:0",
        dtype: torch.dtype = torch.bfloat16,
        max_new_tokens: int = 256,
    ) -> None:
        from peft import PeftModel
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

        self.device = device
        self.max_new_tokens = max_new_tokens
        self.processor = AutoProcessor.from_pretrained(str(base_model), trust_remote_code=True)
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            str(base_model),
            torch_dtype=dtype,
            device_map=device,
            trust_remote_code=True,
        )
        self.model = PeftModel.from_pretrained(model, str(adapter_path)).merge_and_unload()
        self.model.eval()

    def _generate_from_parts(
        self,
        content: list[dict[str, Any]],
        *,
        images: list[Image.Image] | None = None,
        video: PreparedVideo | None = None,
        max_new_tokens: int | None = None,
    ) -> str:
        messages = [{"role": "user", "content": content}]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        kwargs: dict[str, Any] = {
            "text": [text],
            "images": images or None,
            "padding": True,
            "return_tensors": "pt",
        }
        if video and video.frames:
            kwargs["videos"] = [video.frames]
        inputs = self.processor(**kwargs)
        self._fix_qwen3vl_video_grid(inputs)
        inputs = inputs.to(self.device)
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens or self.max_new_tokens,
                do_sample=False,
            )
        new_ids = output_ids[:, inputs["input_ids"].shape[1] :]
        return self.processor.batch_decode(new_ids, skip_special_tokens=True)[0].strip()

    @staticmethod
    def _count_mm_groups(mm_token_type_ids: torch.Tensor, value: int) -> int:
        ids = mm_token_type_ids[0].tolist()
        groups = 0
        prev = None
        for item in ids:
            if item == value and prev != value:
                groups += 1
            prev = item
        return groups

    @classmethod
    def _fix_qwen3vl_video_grid(cls, inputs: Any) -> None:
        """Align Qwen3-VL timestamp-expanded video token groups with grids.

        transformers 5.3 expands one video placeholder into one token group per
        temporal cell, separated by timestamp text, while ``video_grid_thw`` may
        remain a single ``[T,H,W]`` row. Qwen3VLModel.get_rope_index consumes one
        grid row per contiguous video token group, so we split ``[T,H,W]`` into
        T rows of ``[1,H,W]`` when needed.
        """
        if "video_grid_thw" not in inputs or "mm_token_type_ids" not in inputs:
            return
        video_grid = inputs["video_grid_thw"]
        group_count = cls._count_mm_groups(inputs["mm_token_type_ids"], 2)
        if group_count <= int(video_grid.shape[0]):
            return
        expanded = []
        for row in video_grid:
            t, h, w = [int(x) for x in row.tolist()]
            expanded.extend([[1, h, w] for _ in range(max(t, 1))])
        if len(expanded) == group_count:
            inputs["video_grid_thw"] = torch.tensor(expanded, dtype=video_grid.dtype)

    def plan(
        self,
        instruction: str,
        *,
        video: PreparedVideo | None = None,
        ref_images: list[Image.Image] | None = None,
    ) -> tuple[dict[str, Any], str]:
        content: list[dict[str, Any]] = [{"type": "text", "text": TYPE1_SYSTEM.rstrip() + "\n\n"}]
        images: list[Image.Image] = []
        for i, image in enumerate(ref_images or [], start=1):
            content.append({"type": "text", "text": f"Image{i}: "})
            content.append({"type": "image", "image": image})
            images.append(image)
            content.append({"type": "text", "text": "\n"})
        if video and video.frames:
            content.append({"type": "text", "text": "Video: "})
            content.append({"type": "video", "video": video.frames, "fps": video.fps})
            content.append({"type": "text", "text": "\n"})
        content.append({"type": "text", "text": f"Text Instruction: {instruction}"})
        raw = self._generate_from_parts(content, images=images, video=video, max_new_tokens=320)
        return normalize_plan(parse_json_object(raw)), raw

    def select_image(
        self,
        request: str,
        entity_description: str,
        candidate_images: list[Image.Image],
    ) -> tuple[int, str]:
        content: list[dict[str, Any]] = [
            {
                "type": "text",
                "text": TYPE3_SYSTEM.rstrip() + f"\n\nUser request: {request}\n\n",
            }
        ]
        for i, image in enumerate(candidate_images, start=1):
            content.append({"type": "text", "text": f"Image{i}: "})
            content.append({"type": "image", "image": image})
            content.append({"type": "text", "text": "\n"})
        content.append({"type": "text", "text": f"\nEntity description from web:\n{entity_description}"})
        raw = self._generate_from_parts(content, images=candidate_images, max_new_tokens=32)
        match = re.search(r"Image\s+(\d+)", raw, flags=re.I)
        if not match:
            return 1, raw
        idx = int(match.group(1))
        idx = max(1, min(idx, len(candidate_images)))
        return idx, raw


def setup_serper_env(api_key: str | None) -> None:
    if api_key:
        os.environ["SERPER_KEY_ID"] = api_key
    os.environ.setdefault("TEXT_SEARCH_API_BASE_URL", "https://google.serper.dev/search")
    os.environ.setdefault("IMAGE_SEARCH_API_BASE_URL", "https://google.serper.dev/images")


def run_text_search(query: str, top_k: int) -> str:
    import requests

    api_key = (os.environ.get("SERPER_KEY_ID") or "").strip()
    if not api_key:
        return "[Search] SERPER_KEY_ID is not set."
    url = (os.environ.get("TEXT_SEARCH_API_BASE_URL") or "").strip()
    if not url:
        return "[Search] TEXT_SEARCH_API_BASE_URL is not set."

    q_clean = (query or "").replace('"', "").replace("'", "").strip()
    top_k = min(10, max(1, top_k))
    headers = {"X-API-KEY": api_key, "Content-Type": "application/json"}
    payload = {"q": q_clean, "num": top_k}
    last_error = ""
    for retry in range(5):
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=30, proxies=None)
            if resp.status_code == 429 or resp.status_code >= 500:
                time.sleep(2 + random.uniform(0, 2))
                continue
            resp.raise_for_status()
            data = resp.json()
            organic = data.get("organic") or []
            snippets = []
            for page in organic[:top_k]:
                title = re.sub(r"</?b>", "", page.get("title", ""))
                link = page.get("link", "")
                snippet = re.sub(r"</?b>", "", page.get("snippet", ""))
                snippets.append(f"[{title}]({link}) {snippet}")
            body = "\n\n".join(snippets) if snippets else f"No results for '{q_clean}'."
            return f"--- search result for [{q_clean}] ---\n{body}\n--- end of search result ---"
        except Exception as exc:
            last_error = str(exc)
            time.sleep(2 + random.uniform(0, 2))
    return f"Search failed for '{q_clean}': {last_error}"


def run_image_search(query: str, out_dir: Path, sample_id: str, top_k: int) -> list[dict[str, str]]:
    from aurora.tools import search_universal_image

    save_dir = _ensure_dir(out_dir / "search_images")
    return search_universal_image(
        query=query,
        topk=top_k,
        max_retry=5,
        save_dir=str(save_dir),
        sample_id=sample_id,
    )


def refine_search_query(agent_query: str, plan: dict[str, Any], case: dict[str, Any]) -> str:
    """Keep tool queries specific when the refined edit names a visual form."""
    query = agent_query.strip()
    context = f"{plan.get('refined_text_instruction', '')} {case.get('prompt', '')}".lower()
    if "logo" in context and "logo" not in query.lower():
        query = f"{query} logo"
    return query


def load_search_candidate(path: str, max_side: int) -> Image.Image:
    img = Image.open(path).convert("RGB")
    img.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
    return img


def overlay_mask(image: Image.Image, mask: Image.Image, alpha: float = 0.45) -> Image.Image:
    base = image.convert("RGBA")
    m = mask.convert("L").resize(base.size, Image.Resampling.NEAREST)
    red = Image.new("RGBA", base.size, (255, 45, 45, int(255 * alpha)))
    out = Image.alpha_composite(base, Image.composite(red, Image.new("RGBA", base.size, (0, 0, 0, 0)), m))
    return out.convert("RGB")


class GroundedSamMasker:
    def __init__(
        self,
        *,
        device: str = "cuda:0",
        dino_model: str = "IDEA-Research/grounding-dino-base",
        sam_model: str = "facebook/sam-vit-base",
        box_threshold: float = 0.25,
        text_threshold: float = 0.25,
    ) -> None:
        self.device = device
        self.dino_model_name = dino_model
        self.sam_model_name = sam_model
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold
        self._loaded = False

    def _load(self) -> None:
        if self._loaded:
            return
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor, SamModel, SamProcessor

        self.dino_processor = AutoProcessor.from_pretrained(self.dino_model_name)
        self.dino_model = AutoModelForZeroShotObjectDetection.from_pretrained(self.dino_model_name).to(self.device)
        self.sam_processor = SamProcessor.from_pretrained(self.sam_model_name)
        self.sam_model = SamModel.from_pretrained(self.sam_model_name).to(self.device)
        self.dino_model.eval()
        self.sam_model.eval()
        self._loaded = True

    def segment(self, image: Image.Image, phrase: str) -> tuple[Image.Image, dict[str, Any]]:
        self._load()
        prompt = phrase.strip().lower().rstrip(".") + "."
        inputs = self.dino_processor(images=image, text=prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.dino_model(**inputs)
        target_sizes = torch.tensor([image.size[::-1]], device=self.device)
        results = self.dino_processor.post_process_grounded_object_detection(
            outputs,
            input_ids=inputs.get("input_ids"),
            threshold=self.box_threshold,
            text_threshold=self.text_threshold,
            target_sizes=target_sizes,
        )[0]
        boxes = results.get("boxes")
        scores = results.get("scores")
        labels = results.get("labels", [])
        if boxes is None or len(boxes) == 0:
            empty = Image.new("L", image.size, 0)
            return empty, {"backend": "grounded_sam", "found": False, "phrase": phrase, "boxes": []}

        best = int(torch.argmax(scores).item()) if scores is not None and len(scores) else 0
        box = boxes[best].detach().cpu().tolist()
        sam_inputs = self.sam_processor(image, input_boxes=[[[box]]], return_tensors="pt").to(self.device)
        with torch.no_grad():
            sam_outputs = self.sam_model(**sam_inputs)
        masks = self.sam_processor.image_processor.post_process_masks(
            sam_outputs.pred_masks.detach().cpu(),
            sam_inputs["original_sizes"].detach().cpu(),
            sam_inputs["reshaped_input_sizes"].detach().cpu(),
        )[0]
        iou = sam_outputs.iou_scores.detach().cpu()[0, 0]
        mask_idx = int(torch.argmax(iou).item())
        mask_arr = masks[0, mask_idx].numpy().astype(np.uint8) * 255
        mask = Image.fromarray(mask_arr, mode="L").resize(image.size, Image.Resampling.NEAREST)
        meta = {
            "backend": "grounded_sam",
            "found": True,
            "phrase": phrase,
            "box": [float(x) for x in box],
            "score": float(scores[best].detach().cpu().item()) if scores is not None and len(scores) else None,
            "label": labels[best] if labels else "",
            "sam_iou": float(iou[mask_idx].item()),
        }
        return mask, meta


def write_gallery(records: list[dict[str, Any]], html_path: Path) -> None:
    _ensure_dir(html_path.parent)
    cards: list[str] = []
    for rec in records:
        assets = rec.get("assets", {})
        search_html = ""
        for cand in rec.get("search", {}).get("candidates", []):
            cls = " selected" if cand.get("selected") else ""
            img_rel = html.escape(os.path.relpath(cand["local_path"], html_path.parent))
            search_html += (
                f'<figure class="candidate{cls}"><img src="{img_rel}">'
                f"<figcaption>{html.escape(cand.get('rank_label', ''))}<br>{html.escape(cand.get('title', ''))}</figcaption></figure>"
            )
        source_rel = html.escape(os.path.relpath(assets["source_sheet"], html_path.parent)) if assets.get("source_sheet") else ""
        ref_rel = html.escape(os.path.relpath(assets["ref_image"], html_path.parent)) if assets.get("ref_image") else ""
        mask_rel = html.escape(os.path.relpath(assets["mask_overlay"], html_path.parent)) if assets.get("mask_overlay") else ""
        plan = rec.get("plan", {})
        final_payload = rec.get("final_payload", {})
        cards.append(
            f"""
<section class="card">
  <header>
    <h2>{html.escape(rec['bench_id'])} · {html.escape(rec.get('edit_type', ''))}</h2>
    <span>{html.escape(plan.get('subtask', ''))}</span>
  </header>
  <div class="media">
    <figure><img src="{source_rel}"><figcaption>source keyframes</figcaption></figure>
    {f'<figure><img src="{ref_rel}"><figcaption>input ref image</figcaption></figure>' if ref_rel else ''}
    {f'<figure><img src="{mask_rel}"><figcaption>mask overlay</figcaption></figure>' if mask_rel else ''}
  </div>
  <p><b>Instruction</b>: {html.escape(rec.get('prompt', ''))}</p>
  <p><b>Refined</b>: {html.escape(plan.get('refined_text_instruction', ''))}</p>
  <div class="kv">
    <div><b>image_search</b><pre>{html.escape(json.dumps(plan.get('image_search'), ensure_ascii=False))}</pre></div>
    <div><b>mask</b><pre>{html.escape(json.dumps(plan.get('mask'), ensure_ascii=False))}</pre></div>
  </div>
  {f'<div class="candidates">{search_html}</div>' if search_html else ''}
  <details><summary>Raw agent output</summary><pre>{html.escape(rec.get('agent_raw', ''))}</pre></details>
  <details><summary>Final payload</summary><pre>{html.escape(json.dumps(final_payload, ensure_ascii=False, indent=2))}</pre></details>
</section>
"""
        )
    html_doc = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Aurora Agent EditVerse</title>
<style>
body{{font-family:Arial,sans-serif;background:#f6f6f4;color:#151515;margin:0;padding:24px}}
h1{{margin:0 0 18px;font-size:26px}} .card{{background:white;border:1px solid #d8d8d0;border-radius:6px;padding:14px;margin:0 0 18px}}
header{{display:flex;justify-content:space-between;gap:12px;align-items:center;border-bottom:1px solid #ecece6;margin-bottom:12px}}
h2{{font-size:18px;margin:0 0 10px}} header span{{font-family:monospace;background:#eef3f0;padding:4px 8px;border-radius:4px}}
.media,.candidates{{display:flex;gap:10px;flex-wrap:wrap}} figure{{margin:0 0 8px}} img{{max-width:360px;max-height:260px;border:1px solid #ddd;background:#fafafa}}
figcaption{{font-size:12px;color:#555;max-width:360px}} .candidate{{width:190px}} .candidate img{{width:188px;height:140px;object-fit:contain}}
.candidate.selected{{outline:4px solid #1b8a5a;outline-offset:2px}} .kv{{display:grid;grid-template-columns:repeat(auto-fit,minmax(240px,1fr));gap:10px}}
pre{{white-space:pre-wrap;word-break:break-word;background:#f1f1ed;padding:8px;border-radius:4px;font-size:12px}}
</style></head><body><h1>Aurora Agent EditVerse</h1>{''.join(cards)}</body></html>"""
    html_path.write_text(html_doc, encoding="utf-8")


def process_case(
    case: dict[str, Any],
    agent: AgentVLM,
    out_dir: Path,
    args: argparse.Namespace,
    masker: GroundedSamMasker | None,
) -> dict[str, Any]:
    bench_id = case["bench_id"]
    case_dir = _ensure_dir(out_dir / "cases" / _safe_name(bench_id))
    frames, fps = sample_video_frames(case["video_path"], args.agent_video_frames, args.agent_frame_max_side)
    video = PreparedVideo(frames=frames, fps=fps)
    ref = load_ref_image(case.get("ref_image_path", ""), args.agent_frame_max_side)
    plan, raw = agent.plan(case["prompt"], video=video, ref_images=[ref] if ref else None)
    if ref:
        plan["image_search"] = False
    if getattr(args, "disable_image_search", False):
        plan["image_search"] = False

    assets: dict[str, str] = {}
    assets["source_sheet"] = save_jpeg(make_contact_sheet(frames), case_dir / "source_keyframes.jpg")
    if ref:
        assets["ref_image"] = save_jpeg(ref, case_dir / "input_ref.jpg")

    search_info: dict[str, Any] = {"agent_query": None, "query": None, "selected_path": None, "selector_raw": "", "candidates": []}
    query = _as_query(plan.get("image_search"))
    if query and not ref:
        tool_query = refine_search_query(query, plan, case)
        text_desc = run_text_search(tool_query, args.text_search_top_k)
        results = run_image_search(tool_query, out_dir, bench_id, args.image_search_top_k)
        cand_images = [load_search_candidate(r["local_path"], args.search_image_max_side) for r in results]
        if cand_images:
            selected_idx, selector_raw = agent.select_image(
                plan["refined_text_instruction"] or case["prompt"],
                text_desc,
                cand_images,
            )
            selected_path = results[selected_idx - 1]["local_path"]
        else:
            selected_idx, selector_raw, selected_path = 1, "", None
        search_info.update({
            "agent_query": query,
            "query": tool_query,
            "entity_description": text_desc,
            "selector_raw": selector_raw,
            "selected_path": selected_path,
        })
        for i, r in enumerate(results, start=1):
            search_info["candidates"].append(
                {
                    "rank_label": f"Image {i}",
                    "title": r.get("title", ""),
                    "url": r.get("url", ""),
                    "page_url": r.get("page_url", ""),
                    "local_path": r.get("local_path", ""),
                    "selected": bool(selected_path and r.get("local_path") == selected_path),
                }
            )

    mask_info: dict[str, Any] = {"phrase": None, "mask_path": None, "overlay_path": None, "meta": None}
    mask_phrase = _as_query(plan.get("mask"))
    if mask_phrase and masker is not None:
        mask, meta = masker.segment(frames[0], mask_phrase)
        mask_path = save_png(mask, case_dir / "object_mask.png")
        overlay_path = save_jpeg(overlay_mask(frames[0], mask), case_dir / "object_mask_overlay.jpg")
        assets["mask_overlay"] = overlay_path
        mask_info = {"phrase": mask_phrase, "mask_path": mask_path, "overlay_path": overlay_path, "meta": meta}

    final_payload = {
        "refined_text_instruction": plan["refined_text_instruction"],
        "subtask": plan["subtask"],
        "search_image": search_info.get("selected_path") or False,
        "object_mask": mask_info.get("mask_path") or False,
    }
    return {
        **case,
        "plan": plan,
        "agent_raw": raw,
        "search": search_info,
        "mask": mask_info,
        "assets": assets,
        "final_payload": final_payload,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bench_dir", type=Path, default=DEFAULT_BENCH_DIR)
    parser.add_argument("--out_dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--agent_base", type=Path, default=DEFAULT_AGENT_BASE)
    parser.add_argument("--agent_adapter", type=Path, default=DEFAULT_AGENT_ADAPTER)
    parser.add_argument("--agent_backend", choices=["hf", "vllm"], default="hf",
                        help="Planner backend: 'hf' (transformers, reference) or 'vllm'.")
    parser.add_argument("--agent_merged_dir", type=Path, default=None,
                        help="Pre-merged Qwen3-VL dir for --agent_backend vllm "
                             "(create once via aurora.agent_vllm.merge_agent_to_dir).")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--bench_ids", default="", help="Comma-separated EditVerse ids. Empty means all.")
    parser.add_argument("--custom_cases_jsonl", type=Path, default=None)
    parser.add_argument("--max_cases", type=int, default=0)
    parser.add_argument("--serper_api_key", default=os.environ.get("SERPER_KEY_ID", ""))
    parser.add_argument("--agent_video_frames", type=int, default=8)
    parser.add_argument("--agent_frame_max_side", type=int, default=448)
    parser.add_argument("--search_image_max_side", type=int, default=512)
    parser.add_argument("--image_search_top_k", type=int, default=5)
    parser.add_argument("--text_search_top_k", type=int, default=5)
    parser.add_argument("--mask_backend", choices=["none", "grounded_sam"], default="grounded_sam")
    parser.add_argument("--dino_model", default="IDEA-Research/grounding-dino-base")
    parser.add_argument("--sam_model", default="facebook/sam-vit-base")
    parser.add_argument(
        "--disable_image_search",
        action="store_true",
        help="Force plan['image_search']=False for every case. Use for benchmarking "
             "(EditVerse / OpenVE): fair comparison requires zero external "
             "web image lookup, regardless of what the agent plan emits.",
    )
    parser.add_argument(
        "--custom_only",
        action="store_true",
        help="Skip EditVerse case loading entirely; only run cases from --custom_cases_jsonl. "
             "Use for benches that are not EditVerse, so you don't need "
             "the editverse benchmark directory to exist.",
    )
    parser.add_argument(
        "--shard_index",
        type=int,
        default=0,
        help="0-based shard index for splitting the case list across multiple GPUs/processes.",
    )
    parser.add_argument(
        "--num_shards",
        type=int,
        default=1,
        help="Total number of shards. shard_index in [0, num_shards). cases[i] goes to shard "
             "i %% num_shards.",
    )
    args = parser.parse_args()

    if args.agent_backend == "hf":
        # Auto-download the Qwen3-VL base + aurora_agent_vlm adapter on first use.
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        from evaluation.model_download import resolve_agent_paths
        args.agent_base, args.agent_adapter = resolve_agent_paths(args.agent_base, args.agent_adapter)

    setup_serper_env(args.serper_api_key)
    wanted = {x.strip() for x in args.bench_ids.split(",") if x.strip()} or None
    if args.custom_only:
        cases = []
    else:
        cases = load_editverse_cases(args.bench_dir, wanted)
    if args.custom_cases_jsonl:
        cases.extend(load_custom_cases(args.custom_cases_jsonl))
    if args.num_shards > 1:
        if not (0 <= args.shard_index < args.num_shards):
            raise ValueError(f"shard_index={args.shard_index} out of range for num_shards={args.num_shards}")
        cases = [c for i, c in enumerate(cases) if i % args.num_shards == args.shard_index]
        print(f"Sharded: shard {args.shard_index}/{args.num_shards} -> {len(cases)} cases")
    if args.max_cases:
        cases = cases[: args.max_cases]
    _ensure_dir(args.out_dir)

    print(f"Loaded {len(cases)} cases")
    print(f"Agent base: {args.agent_base}")
    print(f"Agent adapter: {args.agent_adapter}")
    if args.agent_backend == "vllm":
        from aurora.agent_vllm import AgentVLMvLLM
        agent = AgentVLMvLLM(args.agent_merged_dir or args.agent_base, device=args.device,
                             max_images=args.agent_video_frames + args.image_search_top_k + 2)
    else:
        agent = AgentVLM(args.agent_base, args.agent_adapter, device=args.device)
    masker = None
    if args.mask_backend == "grounded_sam":
        masker = GroundedSamMasker(device=args.device, dino_model=args.dino_model, sam_model=args.sam_model)

    records = []
    jsonl_path = args.out_dir / "agent_pipeline_records.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as f:
        for idx, case in enumerate(cases, start=1):
            print(f"[{idx}/{len(cases)}] {case['bench_id']} {case['edit_type']}: {case['prompt'][:90]}")
            try:
                rec = process_case(case, agent, args.out_dir, args, masker)
            except Exception as exc:
                tb = traceback.format_exc()
                rec = {**case, "error": repr(exc), "traceback": tb}
                print(f"  ERROR: {repr(exc)}")
                print(tb)
            records.append(rec)
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
    gallery_path = args.out_dir / "inspection_gallery.html"
    write_gallery([r for r in records if "error" not in r], gallery_path)
    print(f"Records: {jsonl_path}")
    print(f"Gallery: {gallery_path}")


if __name__ == "__main__":
    main()
