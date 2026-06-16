"""API (hosted-VLM) agent pipeline for AgentEdit-Bench (backward-compatible).

Mirrors aurora/agent.py but swaps the
Qwen3-VL+LoRA Aurora agent for a hosted VLM served over an API:

  - backend=openai   -> OpenAI Responses API + web_search_preview tool
                        (model default: gpt-5.5)
  - backend=google   -> google-genai SDK + google_search tool
                        (model default: gemini-3.1-pro-preview)

The script reuses helpers imported from aurora.agent so the agent code path is
shared. The output `agent_pipeline_records.jsonl`
follows the same schema, so it can be fed verbatim into
`evaluation/agentedit_bench_infer.py --agent_records_jsonl ...`.

Differences vs the Aurora script:

  - Mask backend is forced OFF (hosted API VLMs cannot produce a binary
    mask; downstream supports `mask=null` cleanly).
  - For OpenAI / Google backends, image retrieval is driven by the VLM's
    native search tool — it returns image URLs which we download into
    `<out_dir>/search_images/` ourselves. No Serper key required.
  - Per-record token usage + wall time are captured in `_token_usage` and
    `_timing`. Aggregate totals are written to `_agent_run_summary.json`.

Required env vars:
  - OPENAI_API_KEY      (backend=openai)
  - GEMINI_API_KEY      (backend=google)
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import os
import re
import sys
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import requests
from PIL import Image

REPO_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_DIR))

from aurora.agent import PreparedVideo, TYPE1_SYSTEM, TYPE3_SYSTEM, _as_query, _ensure_dir, _safe_name, load_custom_cases, load_ref_image, make_contact_sheet, normalize_plan, parse_json_object, sample_video_frames, save_jpeg, write_gallery


OUTPUT_DIR = Path(os.environ.get("AURORA_OUTPUT_DIR", str(REPO_DIR / "outputs")))
DEFAULT_OUT_DIR = OUTPUT_DIR / "agent_api"


def _encode_jpeg_b64(image: Image.Image, max_side: int = 768, quality: int = 90) -> str:
    img = image.convert("RGB")
    if max(img.size) > max_side:
        ratio = max_side / max(img.size)
        img = img.resize(
            (int(img.width * ratio), int(img.height * ratio)),
            Image.Resampling.LANCZOS,
        )
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _build_text_part(text: str) -> dict[str, Any]:
    return {"type": "input_text", "text": text}


def _build_image_part_openai(image: Image.Image) -> dict[str, Any]:
    b64 = _encode_jpeg_b64(image)
    return {
        "type": "input_image",
        "image_url": f"data:image/jpeg;base64,{b64}",
    }


_BROWSER_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/130.0.0.0 Safari/537.36"
    ),
    "Accept": "image/avif,image/webp,image/apng,image/*,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.google.com/",
}


def _looks_like_image_url(url: str) -> bool:
    """Reject obvious product-page / search-result URLs that GPT-5.x sometimes
    hallucinates into image_url slots (e.g. walmart /ip/<id>, kikkomanusa /
    homecooks/products/...). We require either an image extension OR a
    pattern that looks like a CDN asset path."""
    u = url.lower().split("?", 1)[0].split("#", 1)[0]
    if u.endswith((".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp", ".avif", ".heic")):
        return True
    cdn_markers = ("/cdn/", "/cdn-cgi/", "/images/", "/media/", "/medias/",
                   "/assets/", "/uploads/", "/photos/", "/img/", "/wikipedia/commons/")
    return any(m in u for m in cdn_markers)


def _download_url_to(image_url: str, dest_dir: Path, prefix: str) -> str | None:
    """Download URL into dest_dir keyed by md5(url). Returns local path or None."""
    if not _looks_like_image_url(image_url):
        return None
    try:
        r = requests.get(image_url, headers=_BROWSER_HEADERS, timeout=30, stream=True)
        r.raise_for_status()
        ctype = (r.headers.get("Content-Type") or "").lower()
        if not ctype.startswith("image/") and not image_url.lower().endswith(
            (".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp")
        ):
            return None
        h = hashlib.md5(image_url.encode("utf-8")).hexdigest()
        ext = ".jpg"
        if "png" in ctype:
            ext = ".png"
        elif "webp" in ctype:
            ext = ".webp"
        save_path = dest_dir / f"{prefix}-{h}{ext}"
        _ensure_dir(dest_dir)
        with save_path.open("wb") as f:
            for chunk in r.iter_content(chunk_size=65536):
                if chunk:
                    f.write(chunk)
        try:
            img = Image.open(save_path).convert("RGB")
            jpeg_path = save_path.with_suffix(".jpg")
            img.save(jpeg_path, "JPEG", quality=90)
            if jpeg_path != save_path:
                save_path.unlink(missing_ok=True)
            return str(jpeg_path)
        except Exception:
            save_path.unlink(missing_ok=True)
            return None
    except Exception as e:  # noqa: BLE001
        print(f"  download fail {image_url[:80]}: {e}", flush=True)
        return None


@dataclass
class APIResponse:
    text: str
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    raw_search_results: list[dict[str, Any]] = field(default_factory=list)
    extra: dict[str, Any] = field(default_factory=dict)


class APIAgent:
    """Abstract hosted-VLM (API) wrapper with plan() + select_image()."""

    backend: str = "abstract"
    model: str = ""

    def __init__(self, model: str, *, temperature: float = 0.4) -> None:
        self.model = model
        self.temperature = temperature
        self.totals: dict[str, int] = {"input": 0, "output": 0, "total": 0, "calls": 0}

    def _track(self, resp: APIResponse) -> None:
        self.totals["input"] += int(resp.input_tokens or 0)
        self.totals["output"] += int(resp.output_tokens or 0)
        self.totals["total"] += int(resp.total_tokens or 0)
        self.totals["calls"] += 1

    def _call_text(self, system_prompt: str, user_text: str, *,
                   images: list[Image.Image] | None = None,
                   enable_search: bool = False,
                   max_tokens: int = 512) -> APIResponse:
        raise NotImplementedError

    def plan(
        self,
        instruction: str,
        *,
        video: PreparedVideo | None,
        ref_images: list[Image.Image] | None,
    ) -> tuple[dict[str, Any], str, APIResponse]:
        images: list[Image.Image] = []
        text_parts: list[str] = []
        if video and video.frames:
            text_parts.append(f"Source video ({len(video.frames)} keyframes "
                              f"sampled at fps≈{video.fps:.1f}):")
            images.extend(video.frames)
        for i, image in enumerate(ref_images or [], start=1):
            text_parts.append(f"Reference image {i}:")
            images.append(image)
        text_parts.append(f"Text Instruction: {instruction}")
        user_text = "\n".join(text_parts)
        resp = self._call_text(
            TYPE1_SYSTEM,
            user_text,
            images=images,
            enable_search=False,
            max_tokens=512,
        )
        self._track(resp)
        plan = normalize_plan(parse_json_object(resp.text))
        return plan, resp.text, resp

    def search_and_select(
        self,
        agent_query: str,
        refined_instruction: str,
        out_dir: Path,
        bench_id: str,
        top_k: int = 5,
    ) -> tuple[str | None, list[dict[str, Any]], APIResponse]:
        """Use the model's web_search tool to find candidate images,
        download them, then pick one. Returns (selected_path, candidates_list, resp)."""
        raise NotImplementedError


class OpenAIAgent(APIAgent):
    backend = "openai"

    def __init__(self, model: str = "gpt-5.5", *, temperature: float = 0.4) -> None:
        super().__init__(model, temperature=temperature)
        self.api_key = os.environ.get("OPENAI_API_KEY") or ""
        if not self.api_key:
            raise RuntimeError("Set OPENAI_API_KEY for backend=openai")

    def _post_responses(self, payload: dict[str, Any]) -> dict[str, Any]:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        delay = 3.0
        last_err = ""
        for _ in range(4):
            try:
                r = requests.post(
                    "https://api.openai.com/v1/responses",
                    headers=headers,
                    json=payload,
                    timeout=180,
                )
                if r.status_code == 429 or r.status_code >= 500:
                    time.sleep(delay)
                    delay *= 1.8
                    continue
                r.raise_for_status()
                return r.json()
            except Exception as e:  # noqa: BLE001
                last_err = str(e)
                time.sleep(delay)
                delay *= 1.8
        raise RuntimeError(f"OpenAI Responses API failed: {last_err}")

    @staticmethod
    def _flatten_output_text(data: dict[str, Any]) -> str:
        if data.get("output_text"):
            return data["output_text"]
        chunks: list[str] = []
        for item in data.get("output", []) or []:
            for c in item.get("content", []) or []:
                if c.get("type") in ("output_text", "text"):
                    chunks.append(c.get("text", ""))
        return "".join(chunks)

    @staticmethod
    def _extract_search_results(data: dict[str, Any]) -> list[dict[str, Any]]:
        """Pull citations / web search results out of an OpenAI Responses payload."""
        results: list[dict[str, Any]] = []
        for item in data.get("output", []) or []:
            if item.get("type") == "web_search_call":
                for src in item.get("results", []) or []:
                    results.append(src)
                continue
            for c in item.get("content", []) or []:
                annotations = c.get("annotations") or []
                for ann in annotations:
                    if ann.get("type") in ("url_citation", "web_search.citation"):
                        results.append({
                            "url": ann.get("url"),
                            "title": ann.get("title"),
                            "page_url": ann.get("url"),
                        })
        return results

    def _call_text(
        self,
        system_prompt: str,
        user_text: str,
        *,
        images: list[Image.Image] | None = None,
        enable_search: bool = False,
        max_tokens: int = 512,
    ) -> APIResponse:
        content_parts: list[dict[str, Any]] = [_build_text_part(user_text)]
        for img in images or []:
            content_parts.append(_build_image_part_openai(img))

        payload: dict[str, Any] = {
            "model": self.model,
            "instructions": system_prompt,
            "input": [
                {"role": "user", "content": content_parts},
            ],
            "max_output_tokens": max_tokens,
        }
        if enable_search:
            payload["tools"] = [{"type": "web_search_preview"}]
        payload["temperature"] = self.temperature

        data = self._post_responses(payload)
        text = self._flatten_output_text(data)
        usage = data.get("usage") or {}
        return APIResponse(
            text=text,
            input_tokens=int(usage.get("input_tokens") or 0),
            output_tokens=int(usage.get("output_tokens") or 0),
            total_tokens=int(usage.get("total_tokens") or 0),
            raw_search_results=self._extract_search_results(data) if enable_search else [],
            extra={"id": data.get("id", "")},
        )

    def search_and_select(
        self,
        agent_query: str,
        refined_instruction: str,
        out_dir: Path,
        bench_id: str,
        top_k: int = 5,
    ) -> tuple[str | None, list[dict[str, Any]], APIResponse]:
        search_prompt = (
            "You are an image scout. The user wants a single clean reference "
            "image to feed a video editor. Use the web_search tool to find "
            f"candidate images for: \"{agent_query}\". "
            "Prefer official / canonical product photos, brand sites, or "
            "Wikipedia. Return ONLY a JSON object of the form:\n"
            '{"query": "<refined search query>", '
            f'"images": [{{"image_url": "https://...jpg", "page_url": "https://...", '
            f'"title": "..."}}, ...up to {top_k} items]}}'
        )
        resp = self._call_text(
            "You are a precise image-retrieval assistant. "
            "Always invoke web_search and respond with JSON only.",
            search_prompt,
            enable_search=True,
            max_tokens=2048,
        )
        self._track(resp)
        candidates: list[dict[str, Any]] = []
        try:
            parsed = parse_json_object(resp.text)
            for item in (parsed.get("images") or [])[:top_k]:
                if not isinstance(item, dict):
                    continue
                url = (item.get("image_url") or "").strip()
                if not url.lower().startswith("http"):
                    continue
                candidates.append({
                    "image_url": url,
                    "page_url": (item.get("page_url") or "").strip(),
                    "title": (item.get("title") or "").strip(),
                })
        except Exception:
            pass
        if not candidates and resp.raw_search_results:
            for src in resp.raw_search_results[:top_k]:
                u = src.get("url") or ""
                if u.lower().startswith("http"):
                    candidates.append({
                        "image_url": u, "page_url": u,
                        "title": src.get("title", ""),
                    })

        save_dir = _ensure_dir(out_dir / "search_images")
        downloaded: list[dict[str, Any]] = []
        for c in candidates:
            local = _download_url_to(c["image_url"], save_dir, _safe_name(bench_id))
            if local is None:
                continue
            downloaded.append({**c, "local_path": local})

        if not downloaded:
            return None, [], resp

        select_text = (
            f"User request: {refined_instruction}\n\n"
            f"Pick the single best reference image. "
            "Respond strictly as `Image N` (1-indexed)."
        )
        cand_images: list[Image.Image] = []
        for d in downloaded:
            try:
                cand_images.append(Image.open(d["local_path"]).convert("RGB"))
            except Exception:
                cand_images.append(Image.new("RGB", (256, 256), "white"))
        sel = self._call_text(
            TYPE3_SYSTEM,
            select_text,
            images=cand_images,
            enable_search=False,
            max_tokens=32,
        )
        self._track(sel)
        m = re.search(r"image\s+(\d+)", sel.text, flags=re.I)
        idx = int(m.group(1)) if m else 1
        idx = max(1, min(idx, len(downloaded)))
        selected_path = downloaded[idx - 1]["local_path"]
        for i, d in enumerate(downloaded):
            d["rank_label"] = f"Image {i + 1}"
            d["selected"] = (i + 1 == idx)
            d["url"] = d["image_url"]
        return selected_path, downloaded, sel


class GeminiAgent(APIAgent):
    """Gemini via direct google-genai SDK with google_search tool."""

    backend = "google"

    def __init__(self, model: str = "gemini-3.1-pro-preview", *,
                 temperature: float = 0.4) -> None:
        super().__init__(model, temperature=temperature)
        self.api_key = (
            os.environ.get("GEMINI_API_KEY")
            or os.environ.get("GOOGLE_API_KEY")
            or ""
        )
        if not self.api_key:
            raise RuntimeError("Set GEMINI_API_KEY for backend=google")
        try:
            from google import genai  # noqa: F401
            from google.genai import types  # noqa: F401
        except Exception as e:  # noqa: BLE001
            raise RuntimeError(
                "google-genai SDK missing. pip install google-genai"
            ) from e

    def _client(self):
        from google import genai
        return genai.Client(api_key=self.api_key)

    def _call_text(
        self,
        system_prompt: str,
        user_text: str,
        *,
        images: list[Image.Image] | None = None,
        enable_search: bool = False,
        max_tokens: int = 512,
    ) -> APIResponse:
        from google.genai import types

        parts: list[Any] = [user_text]
        for img in images or []:
            buf = io.BytesIO()
            img.convert("RGB").save(buf, format="JPEG", quality=90)
            parts.append(types.Part.from_bytes(
                data=buf.getvalue(), mime_type="image/jpeg",
            ))

        config_kwargs: dict[str, Any] = {
            "system_instruction": system_prompt,
            "temperature": self.temperature,
            "max_output_tokens": max_tokens,
        }
        if enable_search:
            config_kwargs["tools"] = [types.Tool(google_search=types.GoogleSearch())]

        delay = 3.0
        last_err = ""
        for _ in range(4):
            try:
                resp = self._client().models.generate_content(
                    model=self.model,
                    contents=parts,
                    config=types.GenerateContentConfig(**config_kwargs),
                )
                text = resp.text or ""
                usage = getattr(resp, "usage_metadata", None)
                ipt = int(getattr(usage, "prompt_token_count", 0) or 0) if usage else 0
                opt = int(getattr(usage, "candidates_token_count", 0) or 0) if usage else 0
                tot = int(getattr(usage, "total_token_count", 0) or 0) if usage else 0
                results: list[dict[str, Any]] = []
                for cand in getattr(resp, "candidates", []) or []:
                    gm = getattr(cand, "grounding_metadata", None)
                    if not gm:
                        continue
                    for chunk in getattr(gm, "grounding_chunks", []) or []:
                        web = getattr(chunk, "web", None)
                        if web and getattr(web, "uri", None):
                            results.append({
                                "url": web.uri,
                                "title": getattr(web, "title", ""),
                                "page_url": web.uri,
                            })
                return APIResponse(
                    text=text,
                    input_tokens=ipt,
                    output_tokens=opt,
                    total_tokens=tot or (ipt + opt),
                    raw_search_results=results,
                )
            except Exception as e:  # noqa: BLE001
                last_err = str(e)
                time.sleep(delay)
                delay *= 1.8
        raise RuntimeError(f"Gemini API failed: {last_err}")

    def search_and_select(
        self,
        agent_query: str,
        refined_instruction: str,
        out_dir: Path,
        bench_id: str,
        top_k: int = 5,
    ) -> tuple[str | None, list[dict[str, Any]], APIResponse]:
        search_prompt = (
            "Use Google Search to find canonical reference images for: "
            f"\"{agent_query}\". Return ONLY a JSON object of the form:\n"
            '{"query": "...", "images": [{"image_url": "https://...jpg", '
            f'"page_url": "https://...", "title": "..."}}, ...max {top_k}]}}'
        )
        resp = self._call_text(
            "You are a precise image-retrieval assistant. "
            "Use google_search and return JSON only.",
            search_prompt,
            enable_search=True,
            max_tokens=2048,
        )
        self._track(resp)
        candidates: list[dict[str, Any]] = []
        try:
            parsed = parse_json_object(resp.text)
            for item in (parsed.get("images") or [])[:top_k]:
                if not isinstance(item, dict):
                    continue
                u = (item.get("image_url") or "").strip()
                if u.lower().startswith("http"):
                    candidates.append({
                        "image_url": u,
                        "page_url": (item.get("page_url") or "").strip(),
                        "title": (item.get("title") or "").strip(),
                    })
        except Exception:
            pass
        if not candidates and resp.raw_search_results:
            for src in resp.raw_search_results[:top_k]:
                u = src.get("url") or ""
                if u.lower().startswith("http"):
                    candidates.append({
                        "image_url": u, "page_url": u,
                        "title": src.get("title", ""),
                    })
        save_dir = _ensure_dir(out_dir / "search_images")
        downloaded: list[dict[str, Any]] = []
        for c in candidates:
            local = _download_url_to(c["image_url"], save_dir, _safe_name(bench_id))
            if local is None:
                continue
            downloaded.append({**c, "local_path": local})
        if not downloaded:
            return None, [], resp
        select_text = (
            f"User request: {refined_instruction}\n\n"
            f"Pick the single best reference image. Respond as `Image N`."
        )
        cand_images = []
        for d in downloaded:
            try:
                cand_images.append(Image.open(d["local_path"]).convert("RGB"))
            except Exception:
                cand_images.append(Image.new("RGB", (256, 256), "white"))
        sel = self._call_text(
            TYPE3_SYSTEM,
            select_text,
            images=cand_images,
            enable_search=False,
            max_tokens=32,
        )
        self._track(sel)
        m = re.search(r"image\s+(\d+)", sel.text, flags=re.I)
        idx = int(m.group(1)) if m else 1
        idx = max(1, min(idx, len(downloaded)))
        selected_path = downloaded[idx - 1]["local_path"]
        for i, d in enumerate(downloaded):
            d["rank_label"] = f"Image {i + 1}"
            d["selected"] = (i + 1 == idx)
            d["url"] = d["image_url"]
        return selected_path, downloaded, sel


def build_agent(backend: str, model: str, *, temperature: float) -> APIAgent:
    if backend == "openai":
        return OpenAIAgent(model, temperature=temperature)
    if backend == "google":
        return GeminiAgent(model, temperature=temperature)
    raise ValueError(f"unknown backend: {backend}")


def process_case(
    case: dict[str, Any],
    agent: APIAgent,
    out_dir: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    bench_id = case["bench_id"]
    case_dir = _ensure_dir(out_dir / "cases" / _safe_name(bench_id))
    t_case_start = time.time()
    frames, fps = sample_video_frames(
        case["video_path"],
        max_frames=args.agent_video_frames,
        max_side=args.agent_frame_max_side,
    )
    video = PreparedVideo(frames=frames, fps=fps)
    ref = load_ref_image(case.get("ref_image_path", ""), args.agent_frame_max_side)

    t0 = time.time()
    plan, raw_plan, plan_resp = agent.plan(
        case["prompt"],
        video=video,
        ref_images=[ref] if ref else None,
    )
    t_plan = time.time() - t0

    if ref or args.disable_image_search:
        plan["image_search"] = False
    plan["mask"] = False

    assets: dict[str, str] = {}
    assets["source_sheet"] = save_jpeg(make_contact_sheet(frames), case_dir / "source_keyframes.jpg")
    if ref:
        assets["ref_image"] = save_jpeg(ref, case_dir / "input_ref.jpg")

    search_info: dict[str, Any] = {
        "agent_query": None,
        "query": None,
        "selected_path": None,
        "selector_raw": "",
        "candidates": [],
    }
    select_resp: APIResponse | None = None
    query = _as_query(plan.get("image_search"))
    t_search = 0.0
    if query and not ref:
        t1 = time.time()
        try:
            selected_path, candidates, select_resp = agent.search_and_select(
                agent_query=query,
                refined_instruction=plan["refined_text_instruction"] or case["prompt"],
                out_dir=out_dir,
                bench_id=bench_id,
                top_k=args.image_search_top_k,
            )
        except Exception as exc:
            tb = traceback.format_exc()
            print(f"  search_and_select failed: {exc}\n{tb}", flush=True)
            selected_path, candidates = None, []
        t_search = time.time() - t1
        search_info.update({
            "agent_query": query,
            "query": query,
            "selected_path": selected_path,
            "selector_raw": (select_resp.text if select_resp else ""),
        })
        for c in candidates:
            search_info["candidates"].append({
                "rank_label": c.get("rank_label", ""),
                "title": c.get("title", ""),
                "url": c.get("url") or c.get("image_url", ""),
                "page_url": c.get("page_url", ""),
                "local_path": c.get("local_path", ""),
                "selected": bool(c.get("selected")),
            })

    mask_info: dict[str, Any] = {
        "phrase": None, "mask_path": None, "overlay_path": None, "meta": None
    }

    final_payload = {
        "refined_text_instruction": plan["refined_text_instruction"],
        "subtask": plan["subtask"],
        "search_image": search_info.get("selected_path") or False,
        "object_mask": False,
    }

    timing = {
        "wall_total_s": time.time() - t_case_start,
        "plan_s": t_plan,
        "search_s": t_search,
    }
    token_usage = {
        "plan": {
            "input": plan_resp.input_tokens,
            "output": plan_resp.output_tokens,
            "total": plan_resp.total_tokens,
        },
        "search_or_select": {
            "input": (select_resp.input_tokens if select_resp else 0),
            "output": (select_resp.output_tokens if select_resp else 0),
            "total": (select_resp.total_tokens if select_resp else 0),
        },
    }

    return {
        **case,
        "plan": plan,
        "agent_raw": raw_plan,
        "search": search_info,
        "mask": mask_info,
        "assets": assets,
        "final_payload": final_payload,
        "_token_usage": token_usage,
        "_timing": timing,
        "_agent_backend": agent.backend,
        "_agent_model": agent.model,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["openai", "google"], required=True)
    parser.add_argument("--model", required=True,
                        help="e.g. gpt-5.5 (openai) | gemini-3.1-pro-preview (google)")
    parser.add_argument("--out_dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--custom_cases_jsonl", type=Path, required=True,
                        help="JSONL with bench_id/prompt/video_path/ref_image_path")
    parser.add_argument("--temperature", type=float, default=0.4)
    parser.add_argument("--agent_video_frames", type=int, default=8)
    parser.add_argument("--agent_frame_max_side", type=int, default=448)
    parser.add_argument("--image_search_top_k", type=int, default=5)
    parser.add_argument("--max_cases", type=int, default=0)
    parser.add_argument("--shard_index", type=int, default=0)
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--disable_image_search", action="store_true")
    args = parser.parse_args()

    cases = load_custom_cases(args.custom_cases_jsonl)
    if args.num_shards > 1:
        cases = [c for i, c in enumerate(cases) if i % args.num_shards == args.shard_index]
        print(f"Shard {args.shard_index}/{args.num_shards}: {len(cases)} cases")
    if args.max_cases:
        cases = cases[: args.max_cases]
    _ensure_dir(args.out_dir)
    print(f"backend={args.backend} model={args.model}")
    print(f"Loaded {len(cases)} cases -> {args.out_dir}")

    agent = build_agent(args.backend, args.model, temperature=args.temperature)

    records: list[dict[str, Any]] = []
    jsonl_path = args.out_dir / "agent_pipeline_records.jsonl"
    run_started = time.time()
    with jsonl_path.open("w", encoding="utf-8") as f:
        for idx, case in enumerate(cases, start=1):
            t0 = time.time()
            print(f"[{idx}/{len(cases)}] {case['bench_id']} {case['edit_type']}: "
                  f"{case['prompt'][:90]}", flush=True)
            try:
                rec = process_case(case, agent, args.out_dir, args)
            except Exception as exc:
                tb = traceback.format_exc()
                rec = {**case, "error": repr(exc), "traceback": tb}
                print(f"  ERROR: {repr(exc)}", flush=True)
                print(tb, flush=True)
            dt = time.time() - t0
            print(f"  done in {dt:.1f}s "
                  f"tot_tokens={agent.totals['total']:,} "
                  f"calls={agent.totals['calls']}", flush=True)
            records.append(rec)
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()

    run_summary = {
        "backend": args.backend,
        "model": args.model,
        "shard_index": args.shard_index,
        "num_shards": args.num_shards,
        "n_cases": len(records),
        "n_errors": sum(1 for r in records if "error" in r),
        "wall_total_s": time.time() - run_started,
        "tokens": agent.totals,
    }
    sum_path = args.out_dir / f"_agent_run_summary_shard{args.shard_index}.json"
    with sum_path.open("w", encoding="utf-8") as f:
        json.dump(run_summary, f, indent=2, ensure_ascii=False)
    print(f"Records: {jsonl_path}")
    print(f"Summary: {sum_path}")
    print(f"Tokens: {agent.totals}")

    try:
        gallery_path = args.out_dir / f"inspection_gallery_shard{args.shard_index}.html"
        write_gallery([r for r in records if "error" not in r], gallery_path)
        print(f"Gallery: {gallery_path}")
    except Exception as e:  # noqa: BLE001
        print(f"  gallery write skipped: {e}")


if __name__ == "__main__":
    main()
