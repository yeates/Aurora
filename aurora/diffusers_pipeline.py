"""diffusers-facing end-to-end Aurora pipeline.

`AuroraPipeline` is a `diffusers.DiffusionPipeline` that packages BOTH Aurora
models behind one familiar API: the tool-using Qwen3-VL **agent planner** and the
unified video **diffusion editor**. It is a thin facade — every bit of diffusion
math (AnchorRoPE, source temporal-concat conditioning, sequence-side reference
packing, 3-pass CFG, the WAN2.2 VAE) lives in the verified diffsynth
`WanVideoPipeline` held in ``self._engine`` and is invoked unchanged. The agent
plan step reuses ``aurora.agent`` verbatim. Nothing here re-implements either model.

This is the diffusers "day-0" entry point and is fully independent from the vLLM
entry point (``aurora.agent --agent_backend vllm``); the two share no runtime state.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional, Union

import torch
from PIL import Image
try:
    from diffusers import DiffusionPipeline
    from diffusers.utils import BaseOutput
except ImportError as exc:  # opt-in backend; never imported by the default paths
    raise ImportError("AuroraPipeline needs the optional 'diffusers' extra: "
                      "pip install -e '.[diffusers]'") from exc

from evaluation.pipeline_loader import default_paths, load_v2_pipeline
from evaluation.model_download import resolve_editor_ckpt


@dataclass
class AuroraPipelineOutput(BaseOutput):
    # frames[0] is a list[PIL.Image], matching the diffusers video convention
    # (export_to_video(out.frames[0])). ``plan`` is the agent plan when the agent ran.
    frames: List[List[Image.Image]]
    plan: Optional[dict] = None


class AuroraPipeline(DiffusionPipeline):
    """End-to-end Aurora editor + agent planner as a diffusers pipeline."""

    def __init__(self, engine, agent=None):
        super().__init__()
        self._engine = engine
        self._agent = agent
        try:
            # Ergonomics only (pipe.components / inspection). The engine keeps its
            # own module refs and owns device placement / offload.
            self.register_modules(
                dit=engine.dit, vae=engine.vae, mllm=engine.mllm,
                ref_vae_condition=engine.ref_vae_condition,
            )
        except Exception:
            pass

    @classmethod
    def from_pretrained(
        cls,
        ckpt: Optional[Union[str, Path]] = None,
        *,
        device: str = "cuda:0",
        ref_max_items: int = 8,
        mllm_model: Optional[str] = None,
        ref_pad_first: bool = False,
        mllm_video_sample_fps: float = 1.0,
        mllm_video_min_frames: int = 2,
        paths=None,
        agent_base: Optional[Union[str, Path]] = None,
        agent_adapter: Optional[Union[str, Path]] = None,
    ) -> "AuroraPipeline":
        """Load the editor (always) and, if ``agent_base``/``agent_adapter`` are
        given, the Qwen3-VL planner. ``ckpt`` is the trained Aurora editor
        ``.safetensors``; weight loading is delegated to the verified
        ``load_v2_pipeline`` (3 WAN shards + WAN2.2 VAE + Qwen3.5-4B + partial ckpt)."""
        paths = paths or default_paths()
        ckpt = resolve_editor_ckpt(ckpt, paths=paths)
        engine = load_v2_pipeline(
            ckpt, device=device, ref_max_items=ref_max_items, mllm_model=mllm_model,
            ref_pad_first=ref_pad_first, mllm_video_sample_fps=mllm_video_sample_fps,
            mllm_video_min_frames=mllm_video_min_frames, paths=paths,
        )
        agent = None
        if agent_base and agent_adapter:
            from evaluation.model_download import resolve_agent_paths
            agent_base, agent_adapter = resolve_agent_paths(agent_base, agent_adapter, paths=paths)
            from aurora.agent import AgentVLM
            agent = AgentVLM(Path(agent_base), Path(agent_adapter), device=device)
        return cls(engine, agent)

    @property
    def device(self):
        return self._engine.device

    def to(self, *args, **kwargs):
        # The diffsynth engine manages its own device placement / offload.
        return self

    def _search_reference(self, request, query, serper_api_key, top_k):
        """Optional convenience web-image search reusing aurora.agent tools. The
        canonical full agentic flow (search + text-to-mask + gallery) lives in the
        aurora.agent CLI; this hook only retrieves and selects a reference image."""
        from aurora.agent import run_image_search, setup_serper_env
        import tempfile
        setup_serper_env(serper_api_key)
        out_dir = Path(tempfile.mkdtemp(prefix="aurora_search_"))
        hits = run_image_search(str(query), out_dir, "diffusers", top_k)
        imgs = [Image.open(h["local_path"]).convert("RGB") for h in hits if h.get("local_path")]
        if not imgs:
            return None
        if self._agent is not None and len(imgs) > 1:
            idx, _ = self._agent.select_image(request, str(query), imgs)
            return [imgs[idx - 1]]
        return [imgs[0]]

    @torch.no_grad()
    def __call__(
        self,
        prompt: Optional[str] = None,
        *,
        request: Optional[str] = None,
        negative_prompt: str = "",
        video: Optional[Any] = None,
        # Aurora source split (the single most important correctness detail):
        #   source_input -> VAE/DiT spatially-aligned conditioning (len == num_frames)
        #   src_video    -> the Qwen MLLM "view" (defaults to source_input; may subsample)
        source_input: Optional[List[Image.Image]] = None,
        src_video: Optional[List[Image.Image]] = None,
        image: Optional[Union[Image.Image, List[Image.Image]]] = None,
        ref_image: Optional[List[Image.Image]] = None,
        fps: float = 24.0,
        enable_image_search: bool = False,
        serper_api_key: Optional[str] = None,
        image_search_top_k: int = 5,
        height: int = 480,
        width: int = 832,
        num_frames: Optional[int] = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 2.0,         # -> cfg_scale (text CFG); Aurora best-global
        image_guidance_scale: float = 1.0,   # -> image_cfg_scale (image CFG)
        fallback_to_two_pass_cfg: bool = True,
        denoising_strength: float = 1.0,
        generator: Optional[torch.Generator] = None,
        seed: Optional[int] = None,
        tiled: bool = True,
        tile_size=(30, 52),
        tile_stride=(15, 26),
        return_dict: bool = True,
        **engine_kwargs,
    ):
        if source_input is None and video is not None:
            source_input = list(video)
        if src_video is None:
            src_video = source_input
        if ref_image is None and image is not None:
            ref_image = image if isinstance(image, list) else [image]
        if num_frames is None:
            num_frames = len(source_input) if source_input is not None else 81

        plan = None
        if prompt is None:
            if request is None:
                raise ValueError("Provide either prompt= (pure edit) or request= (run the agent planner).")
            if self._agent is None:
                raise ValueError("Agent planner not loaded; pass agent_base/agent_adapter to from_pretrained, or pass prompt=.")
            from aurora.agent import PreparedVideo
            pv = PreparedVideo(frames=list(src_video), fps=fps) if src_video else None
            plan, _ = self._agent.plan(request, video=pv, ref_images=ref_image)
            prompt = plan["refined_text_instruction"]
            if enable_image_search and plan.get("image_search"):
                found = self._search_reference(request, plan["image_search"], serper_api_key, image_search_top_k)
                if found:
                    ref_image = found

        if seed is None and generator is not None:
            seed = int(generator.initial_seed())

        frames = self._engine(
            prompt=prompt, negative_prompt=negative_prompt,
            source_input=source_input, src_video=src_video, ref_image=ref_image,
            height=height, width=width, num_frames=num_frames,
            num_inference_steps=num_inference_steps,
            cfg_scale=guidance_scale, image_cfg_scale=image_guidance_scale,
            fallback_to_two_pass_cfg=fallback_to_two_pass_cfg,
            denoising_strength=denoising_strength, seed=seed,
            tiled=tiled, tile_size=tile_size, tile_stride=tile_stride,
            **engine_kwargs,
        )
        if not return_dict:
            return (frames,)
        return AuroraPipelineOutput(frames=[frames], plan=plan)
