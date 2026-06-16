"""vLLM backend for the Aurora Qwen3-VL agent planner.

`AgentVLMvLLM` mirrors `aurora.agent.AgentVLM` exactly: the same prompt/content
construction and the same `parse_json_object` / `normalize_plan` post-processing,
so the emitted `agent_pipeline_records.jsonl` stays byte-compatible. Only the
single generate boundary (`_generate_from_parts`) is swapped from HF
`model.generate` to a vLLM offline `LLM`. Greedy decoding is preserved
(temperature=0.0).

vLLM's dynamic `--enable-lora` is broken for Qwen3-VL vision-trained adapters, so
the LoRA adapter is merged offline once (`merge_agent_to_dir`) and the merged
directory is served as a plain Qwen3-VL model.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from PIL import Image

from aurora.agent import (
    TYPE1_SYSTEM,
    TYPE3_SYSTEM,
    PreparedVideo,
    normalize_plan,
    parse_json_object,
)


def merge_agent_to_dir(base_model, adapter, out_dir) -> str:
    """One-time: merge the PEFT LoRA into the base Qwen3-VL and save a plain model
    directory that vLLM can serve statically."""
    import torch
    from peft import PeftModel
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    model = Qwen3VLForConditionalGeneration.from_pretrained(
        str(base_model), torch_dtype=torch.bfloat16, trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(model, str(adapter)).merge_and_unload()
    model.save_pretrained(str(out_dir))
    AutoProcessor.from_pretrained(str(base_model), trust_remote_code=True).save_pretrained(str(out_dir))
    return str(out_dir)


class AgentVLMvLLM:
    def __init__(self, merged_dir, *, device: str = "cuda:0", max_new_tokens: int = 256,
                 max_images: int = 16) -> None:
        try:
            from vllm import LLM, SamplingParams
        except ImportError as exc:  # opt-in backend; default --agent_backend hf never imports this
            raise ImportError("The vLLM agent backend needs vllm: "
                              "pip install 'vllm>=0.11' (in a separate venv)") from exc
        from transformers import AutoProcessor

        self.device = device
        self.max_new_tokens = max_new_tokens
        self._SamplingParams = SamplingParams
        self.processor = AutoProcessor.from_pretrained(str(merged_dir), trust_remote_code=True)
        self.llm = LLM(
            model=str(merged_dir), trust_remote_code=True, dtype="bfloat16",
            # Each sampled video frame is fed to vLLM as a still image (see
            # _generate_from_parts), so a single prompt holds video_frames +
            # reference images (plan) or candidate images (select_image). Size the
            # image budget to cover the largest case instead of a fixed 8.
            limit_mm_per_prompt={"image": max(1, int(max_images)), "video": 1},
        )

    def _generate_from_parts(
        self,
        content: list[dict[str, Any]],
        *,
        images: list[Image.Image] | None = None,
        video: PreparedVideo | None = None,
        max_new_tokens: int | None = None,
    ) -> str:
        # vLLM's Qwen3-VL video input requires per-frame metadata and is
        # version-dependent; feed the sampled video frames as still images
        # instead (robust across vLLM versions, same as the API agent path).
        parts: list[dict[str, Any]] = []
        for p in content:
            if p.get("type") == "video":
                for frame in p.get("video", []):
                    parts.append({"type": "image", "image": frame})
            else:
                parts.append(p)
        imgs = [p["image"] for p in parts if p.get("type") == "image"]
        messages = [{"role": "user", "content": parts}]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        mm = {"image": imgs} if imgs else None
        sampling = self._SamplingParams(temperature=0.0, max_tokens=max_new_tokens or self.max_new_tokens)
        outputs = self.llm.generate([{"prompt": text, "multi_modal_data": mm}], sampling)
        return outputs[0].outputs[0].text.strip()

    # plan() and select_image() are identical to aurora.agent.AgentVLM: they build
    # the same content parts and route through _generate_from_parts, then reuse the
    # shared normalize_plan / parse_json_object / Image-N regex.
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
