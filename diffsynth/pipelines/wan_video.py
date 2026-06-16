import json
import math
import os
import torch
import numpy as np
from PIL import Image
from einops import repeat, rearrange
from typing import Optional, Union
from tqdm import tqdm
from diffsynth.utils import BasePipeline, ModelConfig, PipelineUnit, PipelineUnitRunner
from diffsynth.models import ModelManager, load_state_dict
from diffsynth.models.dit import WanModel, sinusoidal_embedding_1d, ConditionalEmbedder
from diffsynth.models.vae import WanVideoVAE
from diffsynth.models.mllm_encoder import VLMEncoder, DEFAULT_MLLM_REF_MAX_PIXELS
from diffsynth.schedulers import FlowMatchScheduler
from diffsynth.vram_management import AutoWrappedLinear
from diffsynth.lora import GeneralLoRALoader
try:
    import deepspeed
except (ModuleNotFoundError, ImportError):
    deepspeed = None

DEBUG = False
PIL_LANCZOS = getattr(getattr(Image, "Resampling", Image), "LANCZOS")


def _rank0():
    return not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0


def _select_debug_frames(frames, max_items):
    if not frames:
        return []
    if max_items <= 0 or len(frames) <= max_items:
        return list(frames)
    indices = np.linspace(0, len(frames) - 1, max_items).round().astype(int).tolist()
    return [frames[i] for i in indices]


def _save_debug_images(images, out_dir, prefix, max_items=0):
    os.makedirs(out_dir, exist_ok=True)
    for idx, img in enumerate(_select_debug_frames(images, max_items)):
        img.save(os.path.join(out_dir, f"{prefix}_{idx:03d}.png"))


def _tensor_stats(tensor):
    if tensor is None:
        return None
    if isinstance(tensor, list):
        return [_tensor_stats(item) for item in tensor]
    return {"shape": list(tensor.shape), "dtype": str(tensor.dtype),
            "mean": float(tensor.float().mean()), "std": float(tensor.float().std())}


def _dump_diffusion_inputs(pipe, height, width, num_frames, input_video, source_input,
                           prepared_ref_image, noise, input_latents=None,
                           vae_source_input=None, vae_ref_image=None):
    dd = getattr(pipe, "debug_dump_state", None)
    if not dd or dd.get("diffusion_dumped"):
        return
    out_dir = os.path.join(dd["step_dir"], "diffusion")
    os.makedirs(out_dir, exist_ok=True)
    max_frames = dd.get("max_frames", 4)
    if input_video is not None:
        _save_debug_images(input_video, out_dir, "tgt_frame", max_items=max_frames)
    if source_input is not None:
        src_frames = [source_input[i] for i in range(min(num_frames, len(source_input)))]
        _save_debug_images(src_frames, out_dir, "src_frame", max_items=max_frames)
    if prepared_ref_image is not None:
        _save_debug_images(prepared_ref_image, out_dir, "ref_prepared", max_items=0)
    meta = {
        "step": dd["step"], "rank": dd.get("rank"), "dataset": dd.get("dataset"),
        "height": height, "width": width, "num_frames": num_frames,
        "n_tgt_frames": len(input_video) if input_video else 0,
        "n_src_frames": len(source_input) if source_input else 0,
        "n_ref": len(prepared_ref_image) if prepared_ref_image else 0,
        "noise": _tensor_stats(noise),
        "input_latents": _tensor_stats(input_latents),
        "vae_source_input": _tensor_stats(vae_source_input),
        "vae_ref_image": _tensor_stats(vae_ref_image),
    }
    with open(os.path.join(out_dir, "metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)
    dd["diffusion_dumped"] = True


def calculate_dimensions(target_area, ratio, factor):
    width = math.sqrt(target_area * ratio)
    height = width / ratio
    width = max(factor, round(width / factor) * factor)
    height = max(factor, round(height / factor) * factor)
    return int(width), int(height)


def resize_preserving_aspect(image, target_area, factor):
    if target_area is None or target_area <= 0:
        return image.convert("RGB")
    width, height = image.size
    ratio = width / max(height, 1)
    target_width, target_height = calculate_dimensions(target_area, ratio, factor)
    return image.convert("RGB").resize((target_width, target_height), PIL_LANCZOS)


def _repeat_to_batch(tensor_or_list, target_batch):
    if tensor_or_list is None:
        return None
    if isinstance(tensor_or_list, list):
        return [_repeat_to_batch(item, target_batch) for item in tensor_or_list]
    if tensor_or_list.shape[0] == target_batch:
        return tensor_or_list
    if tensor_or_list.shape[0] != 1:
        raise ValueError(f"Cannot broadcast batch {tensor_or_list.shape[0]} to {target_batch}")
    return tensor_or_list.repeat(target_batch, *([1] * (tensor_or_list.dim() - 1)))


def _has_ref_condition(ref_items):
    return ref_items is not None and len(ref_items) > 0


def _normalize_ref_image_list(ref_image):
    if ref_image is None:
        return None
    if not isinstance(ref_image, (list, tuple)):
        ref_image = [ref_image]
    ref_image = [item for item in ref_image if item is not None]
    return ref_image or None


def _has_visual_condition(src_video=None, ref_image=None, vae_source_input=None, vae_ref_image=None):
    return (
        src_video is not None
        or bool(ref_image)
        or vae_source_input is not None
        or _has_ref_condition(vae_ref_image)
    )


def _should_fallback_to_two_pass_cfg(
    *,
    has_visual,
    cfg_scale,
    image_cfg_scale,
    negative_context,
    fallback_to_two_pass_cfg,
):
    return (
        fallback_to_two_pass_cfg
        and has_visual
        and negative_context is not None
        and cfg_scale != 1.0
        and image_cfg_scale == 1.0
    )


def _normalize_source_condition_mode(mode, *, allow_auto=False):
    if mode is None:
        mode = "auto" if allow_auto else "temporal_concat"
    mode = str(mode).strip().lower()
    aliases = {
        "concat": "temporal_concat",
        "temporal": "temporal_concat",
        "channel_concat": "temporal_concat",
        "legacy_additive": "additive",
        "sigma_weighted": "additive",
    }
    mode = aliases.get(mode, mode)
    valid_modes = {"temporal_concat", "additive"}
    if allow_auto:
        valid_modes = valid_modes | {"auto"}
    if mode not in valid_modes:
        raise ValueError(f"Unsupported source_condition_mode={mode!r}. Expected one of {sorted(valid_modes)}.")
    return mode


def _state_dict_uses_additive_source_condition(state_dict: dict) -> bool:
    for name, value in state_dict.items():
        if "ref_vae_condition" in name:
            continue
        if not name.endswith("vae_condition.patch_embedding.weight") and not name.endswith("vae_condition.patch_embedding.bias"):
            continue
        if isinstance(value, torch.Tensor) and torch.count_nonzero(value).item() > 0:
            return True
    return False


def _pad_context_batch(contexts: list[torch.Tensor]) -> list[torch.Tensor]:
    if not contexts:
        return []
    max_seq_len = max(context.shape[1] for context in contexts)
    padded = []
    for context in contexts:
        if context.shape[1] == max_seq_len:
            padded.append(context)
            continue
        pad = context.new_zeros((context.shape[0], max_seq_len - context.shape[1], context.shape[2]))
        padded.append(torch.cat([context, pad], dim=1))
    return padded


def _can_batch_contexts(contexts: list[torch.Tensor]) -> bool:
    return len({context.shape[1] for context in contexts}) == 1


def _classifier_free_guidance(positive_pred: torch.Tensor, negative_pred: torch.Tensor, guidance_scale: float) -> torch.Tensor:
    return negative_pred + guidance_scale * (positive_pred - negative_pred)


def _three_pass_guidance(
    unconditional_pred: torch.Tensor,
    visual_negative_pred: torch.Tensor,
    positive_pred: torch.Tensor,
    text_guidance_scale: float,
    image_guidance_scale: float,
) -> torch.Tensor:
    return (
        unconditional_pred
        + image_guidance_scale * (visual_negative_pred - unconditional_pred)
        + text_guidance_scale * (positive_pred - visual_negative_pred)
    )


def _select_freq_rows(freq_table, positions, device):
    positions = positions.to(device=device, dtype=torch.long)
    if positions.numel() == 0:
        return freq_table[:0].to(device)
    abs_positions = positions.abs()
    max_index = int(abs_positions.max().item())
    if max_index >= freq_table.shape[0]:
        raise ValueError(f"RoPE position {max_index} exceeds precomputed limit {freq_table.shape[0] - 1}")
    rows = freq_table.index_select(0, abs_positions.to(freq_table.device)).to(device)
    neg_mask = positions < 0
    if neg_mask.any():
        rows = rows.clone()
        rows[neg_mask] = rows[neg_mask].conj()
    return rows


def _build_spatial_positions(length, centered):
    if not centered:
        return torch.arange(length, dtype=torch.long)
    start = -(length - length // 2)
    end = length // 2
    return torch.arange(start, end, dtype=torch.long)


def _build_rotary_freqs(dit, frame_positions, height, width, device, centered_spatial):
    frame_freqs = _select_freq_rows(dit.freqs[0], frame_positions, device)
    height_positions = _build_spatial_positions(height, centered_spatial).to(device=device)
    width_positions = _build_spatial_positions(width, centered_spatial).to(device=device)
    height_freqs = _select_freq_rows(dit.freqs[1], height_positions, device)
    width_freqs = _select_freq_rows(dit.freqs[2], width_positions, device)
    frame_freqs = frame_freqs.view(len(frame_positions), 1, 1, -1).expand(len(frame_positions), height, width, -1)
    height_freqs = height_freqs.view(1, height, 1, -1).expand(len(frame_positions), height, width, -1)
    width_freqs = width_freqs.view(1, 1, width, -1).expand(len(frame_positions), height, width, -1)
    return torch.cat([frame_freqs, height_freqs, width_freqs], dim=-1).reshape(len(frame_positions) * height * width, 1, -1)

class WanVideoPipeline(BasePipeline):

    def __init__(self, device="cuda", torch_dtype=torch.bfloat16):
        super().__init__(
            device=device, torch_dtype=torch_dtype,
            height_division_factor=16, width_division_factor=16, 
            time_division_factor=4, time_division_remainder=1,
        )
        self.scheduler = FlowMatchScheduler(shift=5, sigma_min=0.0, extra_one_step=True)
        self.dit: WanModel = None
        self.vae: WanVideoVAE = None
        self.mllm: VLMEncoder = None
        self.vae_condition: ConditionalEmbedder = None
        self.ref_vae_condition: ConditionalEmbedder = None
        self.in_iteration_models = ("dit", "vae_condition", "ref_vae_condition")
        self.unit_runner = PipelineUnitRunner()
        self.units = [
            WanVideoUnit_ShapeChecker(),
            WanVideoUnit_NoiseInitializer(),
            WanVideoUnit_MLLMEmbedder(),
            WanVideoUnit_InputVideoEmbedder(),
        ]
        self.post_units = []
        self.model_fn = model_fn_wan_video
        self.debug_dump_state = None
        self.ref_image_max_pixels = 921600
        self.ref_zero_cond_t = True
        self.source_zero_cond_t = True
        self.rope_mode = "anchor"
        self.default_source_condition_mode = "temporal_concat"
        self.source_condition_mode = "auto"

    def get_source_condition_mode(self):
        mode = _normalize_source_condition_mode(self.source_condition_mode, allow_auto=True)
        return self.default_source_condition_mode if mode == "auto" else mode

    def normalize_trainable_models(self, trainable_models: Optional[str]) -> Optional[str]:
        if trainable_models is None:
            return None
        names = [name.strip() for name in trainable_models.split(",") if name.strip()]
        if self.get_source_condition_mode() != "additive":
            names = [name for name in names if name != "vae_condition"]
        seen = set()
        normalized = []
        for name in names:
            if name in seen:
                continue
            seen.add(name)
            normalized.append(name)
        return ",".join(normalized) if normalized else None

    def load_state_dict(self, state_dict, strict: bool = True, assign: bool = False):
        if _normalize_source_condition_mode(self.source_condition_mode, allow_auto=True) == "auto":
            if _state_dict_uses_additive_source_condition(state_dict):
                self.source_condition_mode = "additive"
            else:
                self.source_condition_mode = self.default_source_condition_mode
        return super().load_state_dict(state_dict, strict=strict, assign=assign)
    
    def load_lora(
        self,
        module: torch.nn.Module,
        lora_config: Union[ModelConfig, str] = None,
        alpha=1,
        hotload=False,
        state_dict=None,
    ):
        if state_dict is None:
            if isinstance(lora_config, str):
                lora = load_state_dict(lora_config, torch_dtype=self.torch_dtype, device=self.device)
            else:
                lora_config.download_if_necessary()
                lora = load_state_dict(lora_config.path, torch_dtype=self.torch_dtype, device=self.device)
        else:
            lora = state_dict
        print(lora.keys())
        if hotload:
            for name, module in module.named_modules():
                if isinstance(module, AutoWrappedLinear):
                    lora_a_name = f'{name}.lora_A.default.weight'
                    lora_b_name = f'{name}.lora_B.default.weight'
                    if lora_a_name in lora and lora_b_name in lora:
                        module.lora_A_weights.append(lora[lora_a_name] * alpha)
                        module.lora_B_weights.append(lora[lora_b_name])
        else:
            loader = GeneralLoRALoader(torch_dtype=self.torch_dtype, device=self.device)
            loader.load(module, lora, alpha=alpha)
        
    def training_loss(self, **inputs):
        max_timestep_boundary = int(inputs.get("max_timestep_boundary", 1) * self.scheduler.num_train_timesteps)
        min_timestep_boundary = int(inputs.get("min_timestep_boundary", 0) * self.scheduler.num_train_timesteps)
        timestep_id = torch.randint(min_timestep_boundary, max_timestep_boundary, (1,))
        timestep = self.scheduler.timesteps[timestep_id].to(dtype=self.torch_dtype, device=self.device)
        inputs.setdefault("rope_mode", self.rope_mode)
        inputs.setdefault("ref_zero_cond_t", self.ref_zero_cond_t)
        inputs.setdefault("source_zero_cond_t", self.source_zero_cond_t)
        inputs.setdefault("source_condition_mode", self.get_source_condition_mode())
        inputs["latents"] = self.scheduler.add_noise(inputs["input_latents"], inputs["noise"], timestep)
        training_target = self.scheduler.training_target(inputs["input_latents"], inputs["noise"], timestep)
        
        noise_pred = self.model_fn(**inputs, timestep=timestep, scheduler=self.scheduler)
        
        loss = torch.nn.functional.mse_loss(noise_pred.float(), training_target.float())
        loss = loss * self.scheduler.training_weight(timestep)
        return loss


    def encode_context(
        self,
        prompt: str,
        src_video: Optional[list[Image.Image]] = None,
        ref_image: Optional[list[Image.Image]] = None,
        debug_dump=None,
    ) -> Optional[torch.Tensor]:
        if self.mllm is None:
            if DEBUG: print("skip mllm (no mllm model)")
            return None
        prompt = "" if prompt is None else prompt
        self.load_models_to_device(["mllm"])
        if src_video is None and not ref_image:
            if DEBUG: print("running in text-only mode", prompt)
            return self.mllm(prompt, debug_dump=debug_dump)
        if src_video is not None and len(src_video) == 1:
            if DEBUG: print(len(src_video), "running in image", prompt)
            return self.mllm(prompt, src_image=src_video, ref_image=ref_image, debug_dump=debug_dump)
        if src_video is not None and len(src_video) > 1:
            if DEBUG: print(len(src_video), "running in video", prompt, "ref:", ref_image is not None)
            return self.mllm(prompt, src_video=src_video, ref_image=ref_image, debug_dump=debug_dump)
        if DEBUG: print("running in ref-only mode", prompt)
        return self.mllm(prompt, ref_image=ref_image, debug_dump=debug_dump)


    def _run_model_pass(
        self,
        models: dict,
        inputs_shared: dict,
        context: torch.Tensor,
        timestep: torch.Tensor,
        *,
        with_visual: bool,
    ) -> torch.Tensor:
        model_inputs = dict(inputs_shared)
        model_inputs["context"] = context
        if not with_visual:
            model_inputs["vae_source_input"] = None
            model_inputs["vae_ref_image"] = None
        model_inputs["cfg_batch_size"] = int(context.shape[0])
        return self.model_fn(**models, **model_inputs, timestep=timestep, scheduler=self.scheduler)


    def _predict_noise_with_cfg(
        self,
        models: dict,
        inputs_shared: dict,
        full_context: torch.Tensor,
        negative_context: Optional[torch.Tensor],
        unconditional_context: Optional[torch.Tensor],
        timestep: torch.Tensor,
    ) -> torch.Tensor:
        cfg_scale = float(inputs_shared.get("cfg_scale", 1.0))
        image_cfg_scale = float(inputs_shared.get("image_cfg_scale", 1.0))
        fallback_to_two_pass_cfg = bool(inputs_shared.get("fallback_to_two_pass_cfg", False))
        has_visual = _has_visual_condition(
            src_video=inputs_shared.get("src_video"),
            ref_image=inputs_shared.get("ref_image"),
            vae_source_input=inputs_shared.get("vae_source_input"),
            vae_ref_image=inputs_shared.get("vae_ref_image"),
        )
        do_text_cfg = cfg_scale != 1.0 and negative_context is not None
        use_two_pass_cfg_fallback = _should_fallback_to_two_pass_cfg(
            has_visual=has_visual,
            cfg_scale=cfg_scale,
            image_cfg_scale=image_cfg_scale,
            negative_context=negative_context,
            fallback_to_two_pass_cfg=fallback_to_two_pass_cfg,
        )
        # Default behavior keeps the 3-pass structure whenever text CFG is active.
        # When explicitly requested, image_cfg_scale=1.0 skips the unconditional
        # branch and falls back to the equivalent 2-pass visual CFG path.
        do_image_cfg = (
            has_visual
            and negative_context is not None
            and unconditional_context is not None
            and (image_cfg_scale != 1.0 or (do_text_cfg and not use_two_pass_cfg_fallback))
        )

        if do_image_cfg and do_text_cfg:
            if _can_batch_contexts([negative_context, full_context]):
                visual_context = torch.cat([negative_context, full_context], dim=0)
                visual_preds = self._run_model_pass(models, inputs_shared, visual_context, timestep, with_visual=True)
                negative_pred, positive_pred = visual_preds.chunk(2, dim=0)
            else:
                negative_pred = self._run_model_pass(models, inputs_shared, negative_context, timestep, with_visual=True)
                positive_pred = self._run_model_pass(models, inputs_shared, full_context, timestep, with_visual=True)
            unconditional_pred = self._run_model_pass(models, inputs_shared, unconditional_context, timestep, with_visual=False)
            return _three_pass_guidance(unconditional_pred, negative_pred, positive_pred, cfg_scale, image_cfg_scale)

        if do_image_cfg:
            negative_pred = self._run_model_pass(models, inputs_shared, negative_context, timestep, with_visual=True)
            unconditional_pred = self._run_model_pass(models, inputs_shared, unconditional_context, timestep, with_visual=False)
            return _classifier_free_guidance(negative_pred, unconditional_pred, image_cfg_scale)

        if do_text_cfg:
            if _can_batch_contexts([negative_context, full_context]):
                cfg_context = torch.cat([negative_context, full_context], dim=0)
                cfg_preds = self._run_model_pass(models, inputs_shared, cfg_context, timestep, with_visual=True)
                negative_pred, positive_pred = cfg_preds.chunk(2, dim=0)
            else:
                negative_pred = self._run_model_pass(models, inputs_shared, negative_context, timestep, with_visual=True)
                positive_pred = self._run_model_pass(models, inputs_shared, full_context, timestep, with_visual=True)
            return _classifier_free_guidance(positive_pred, negative_pred, cfg_scale)

        return self._run_model_pass(models, inputs_shared, full_context, timestep, with_visual=True)


    @staticmethod
    def from_pretrained(
        torch_dtype: torch.dtype = torch.bfloat16,
        device: Union[str, torch.device] = "cuda",
        model_configs: list[ModelConfig] = [],
        tokenizer_config: ModelConfig = ModelConfig(model_id="Wan-AI/Wan2.1-T2V-1.3B", origin_file_pattern="google/*"),
        audio_processor_config: ModelConfig = None,
        redirect_common_files: bool = False,
        use_usp=False,
        checkpoint: str = None,
        mllm_model: str = 'Qwen/Qwen3.5-4B',
        mllm_max_pixels_per_frame: int = 512*512,
        mllm_ref_max_pixels: int = DEFAULT_MLLM_REF_MAX_PIXELS,
        mllm_video_sample_fps: Optional[float] = None,
        mllm_video_min_frames: Optional[int] = None,
        mllm_gradient_checkpointing: bool = False,
        ref_pad_first: bool = False,
        ref_max_items: int = 8,
        ref_image_max_pixels: int = 921600,
        rope_mode: str = "anchor",
        ref_zero_cond_t: bool = True,
        source_zero_cond_t: bool = True,
        source_condition_mode: str = "auto",
    ):
        if redirect_common_files:
            redirect_dict = {
                "models_t5_umt5-xxl-enc-bf16.pth": "Wan-AI/Wan2.1-T2V-1.3B",
                "Wan2.1_VAE.pth": "Wan-AI/Wan2.1-T2V-1.3B",
                "models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth": "Wan-AI/Wan2.1-I2V-14B-480P",
            }
            for model_config in model_configs:
                if model_config.origin_file_pattern is None or model_config.model_id is None:
                    continue
                if model_config.origin_file_pattern in redirect_dict and model_config.model_id != redirect_dict[model_config.origin_file_pattern]:
                    print(f"To avoid repeatedly downloading model files, ({model_config.model_id}, {model_config.origin_file_pattern}) is redirected to ({redirect_dict[model_config.origin_file_pattern]}, {model_config.origin_file_pattern}). You can use `redirect_common_files=False` to disable file redirection.")
                    model_config.model_id = redirect_dict[model_config.origin_file_pattern]
        
        pipe = WanVideoPipeline(device=device, torch_dtype=torch_dtype)
        model_manager = ModelManager()
        for model_config in model_configs:
            model_config.skip_download = True
            model_config.download_if_necessary(use_usp=use_usp)
            model_manager.load_model(
                model_config.path,
                device=model_config.offload_device or device,
                torch_dtype=model_config.offload_dtype or torch_dtype
            )

        dit = model_manager.fetch_model("wan_video_dit", index=2)
        pipe.dit = dit
        pipe.mllm = VLMEncoder(
            model_path=mllm_model,
            device=device,
            dit_dim=dit.dim,
            max_pixels_per_frame=mllm_max_pixels_per_frame,
            ref_max_pixels_per_image=mllm_ref_max_pixels,
            video_sample_fps=mllm_video_sample_fps,
            video_min_frames=mllm_video_min_frames,
            gradient_checkpointing=mllm_gradient_checkpointing,
        )

        pipe.vae_condition = ConditionalEmbedder(in_dim=dit.in_dim, dim=dit.dim, patch_size=dit.patch_size, zero_init=True)
        pipe.ref_vae_condition = ConditionalEmbedder(
            in_dim=dit.in_dim,
            dim=dit.dim,
            patch_size=dit.patch_size,
            zero_init=True,
            max_items=ref_max_items,
        )
        pipe.ref_vae_condition.ref_pad_first = ref_pad_first
        pipe.ref_image_max_pixels = ref_image_max_pixels
        pipe.rope_mode = rope_mode
        pipe.ref_zero_cond_t = ref_zero_cond_t
        pipe.source_zero_cond_t = source_zero_cond_t
        pipe.source_condition_mode = _normalize_source_condition_mode(source_condition_mode, allow_auto=True)
        pipe.vae = model_manager.fetch_model("wan_video_vae")
        if pipe.vae is not None:
            pipe.height_division_factor = pipe.vae.upsampling_factor * 2
            pipe.width_division_factor = pipe.vae.upsampling_factor * 2
        
        return pipe

    @torch.no_grad()
    def __call__(
        self,
        prompt: str,
        negative_prompt: Optional[str] = "",
        input_video: Optional[list[Image.Image]] = None,
        src_video: Optional[list[Image.Image]] = None,
        source_input: Optional[list[Image.Image]] = None,
        ref_image: Optional[list[Image.Image]] = None,
        denoising_strength: Optional[float] = 1.0,
        seed: Optional[int] = None,
        rand_device: Optional[str] = "cpu",
        height: Optional[int] = 480,
        width: Optional[int] = 832,
        num_frames=81,
        # Classifier-free guidance
        cfg_scale: Optional[float] = 6.0,
        image_cfg_scale: Optional[float] = 2.5,
        fallback_to_two_pass_cfg: Optional[bool] = False,
        cfg_merge: Optional[bool] = False,
        switch_DiT_boundary: Optional[float] = 0.875,
        num_inference_steps: Optional[int] = 50,
        sigma_shift: Optional[float] = 5.0,
        tiled: Optional[bool] = True,
        tile_size: Optional[tuple[int, int]] = (30, 52),
        tile_stride: Optional[tuple[int, int]] = (15, 26),
        sliding_window_size: Optional[int] = None,
        sliding_window_stride: Optional[int] = None,
        tea_cache_l1_thresh: Optional[float] = None,
        tea_cache_model_id: Optional[str] = "",
        progress_bar_cmd=tqdm,
    ):
        self.scheduler.set_timesteps(num_inference_steps, denoising_strength=denoising_strength, shift=sigma_shift)
        
        inputs_posi = {
            "prompt": prompt,
        }
        inputs_nega = {
            "negative_prompt": negative_prompt,
        }
        inputs_shared = {
            "prompt": prompt,
            "input_video": input_video, "denoising_strength": denoising_strength,
            "src_video": src_video, "source_input": source_input, "ref_image": ref_image,
            "seed": seed, "rand_device": rand_device,
            "height": height, "width": width, "num_frames": num_frames,
            "cfg_scale": cfg_scale,
            "image_cfg_scale": image_cfg_scale,
            "fallback_to_two_pass_cfg": fallback_to_two_pass_cfg,
            "cfg_merge": cfg_merge,
            "sigma_shift": sigma_shift,
            "rope_mode": self.rope_mode,
            "ref_zero_cond_t": self.ref_zero_cond_t,
            "source_zero_cond_t": self.source_zero_cond_t,
            "source_condition_mode": self.get_source_condition_mode(),
            "tiled": tiled, "tile_size": tile_size, "tile_stride": tile_stride,
            "sliding_window_size": sliding_window_size, "sliding_window_stride": sliding_window_stride,
        }
        for unit in self.units:
            inputs_shared, inputs_posi, inputs_nega = self.unit_runner(unit, self, inputs_shared, inputs_posi, inputs_nega)
        full_context = inputs_posi.get("context", inputs_shared.get("context"))
        negative_context = inputs_nega.get("context")
        unconditional_context = None
        has_visual = _has_visual_condition(
            src_video=inputs_shared.get("src_video"),
            ref_image=inputs_shared.get("ref_image"),
            vae_source_input=inputs_shared.get("vae_source_input"),
            vae_ref_image=inputs_shared.get("vae_ref_image"),
        )
        use_two_pass_cfg_fallback = _should_fallback_to_two_pass_cfg(
            has_visual=has_visual,
            cfg_scale=float(cfg_scale),
            image_cfg_scale=float(image_cfg_scale),
            negative_context=negative_context,
            fallback_to_two_pass_cfg=bool(fallback_to_two_pass_cfg),
        )
        if has_visual and negative_context is not None and (
            image_cfg_scale != 1.0 or (cfg_scale != 1.0 and not use_two_pass_cfg_fallback)
        ):
            unconditional_context = self.encode_context(negative_prompt, debug_dump=None)
        inputs_shared.pop("prompt", None)
        self.load_models_to_device(self.in_iteration_models)
        models = {name: getattr(self, name) for name in self.in_iteration_models}
        for progress_id, timestep in enumerate(progress_bar_cmd(self.scheduler.timesteps)):
            timestep = timestep.unsqueeze(0).to(dtype=self.torch_dtype, device=self.device)
            noise_pred = self._predict_noise_with_cfg(
                models,
                inputs_shared,
                full_context,
                negative_context,
                unconditional_context,
                timestep,
            )
            inputs_shared["latents"] = self.scheduler.step(noise_pred, self.scheduler.timesteps[progress_id], inputs_shared["latents"])
 
        for unit in self.post_units:
            inputs_shared, _, _ = self.unit_runner(unit, self, inputs_shared, inputs_posi, inputs_nega)
        self.load_models_to_device(['vae'])
        video = self.vae.decode(inputs_shared["latents"], device=self.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)
        video = self.vae_output_to_video(video)
        self.load_models_to_device([])

        return video


class WanVideoUnit_ShapeChecker(PipelineUnit):
    def __init__(self):
        super().__init__(input_params=("height", "width", "num_frames"))

    def process(self, pipe: WanVideoPipeline, height, width, num_frames):
        height, width, num_frames = pipe.check_resize_height_width(height, width, num_frames)
        return {"height": height, "width": width, "num_frames": num_frames}


class WanVideoUnit_NoiseInitializer(PipelineUnit):
    def __init__(self):
        super().__init__(input_params=("height", "width", "num_frames", "seed", "rand_device", "vae"))

    def process(self, pipe: WanVideoPipeline, height, width, num_frames, seed, rand_device, vae):
        length = (num_frames - 1) // 4 + 1
        if getattr(pipe, "vae", None):
            vae = pipe.vae
        shape = (1, vae.model.z_dim, length, height // vae.upsampling_factor, width // vae.upsampling_factor)
        noise = pipe.generate_noise(shape, seed=seed, rand_device=rand_device)
        return {"noise": noise}


class WanVideoUnit_InputVideoEmbedder(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("height", "width", "input_video", "noise", "tiled", "tile_size", "tile_stride", "num_frames", "source_input", "ref_image", "vae"),
        )

    def process(self, pipe: WanVideoPipeline, height, width, input_video, noise, tiled, tile_size, tile_stride, num_frames, vae, source_input=None, ref_image=None):
        if getattr(pipe, "vae", None):
            vae = pipe.vae
        if source_input is not None:
            if DEBUG: print("add reference vae", len(source_input))
            if pipe.scheduler.training and len(source_input) != num_frames:
                raise ValueError(
                    f"source_input frame mismatch: expected {num_frames}, got {len(source_input)}"
                )
            source_frames = source_input if len(source_input) == num_frames else source_input[:num_frames]
            vae_source_input = pipe.preprocess_video(source_frames)
            if DEBUG: print(vae_source_input[0].shape, tiled, tile_size, tile_stride)
            with torch.no_grad():
                vae_source_input = vae.encode(vae_source_input, device=pipe.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride).to(dtype=pipe.torch_dtype, device=pipe.device)
            if DEBUG: print(vae_source_input.size())
        else:
            vae_source_input = None
        ref_image = _normalize_ref_image_list(ref_image)
        if ref_image:
            vae_ref_image = []
            prepared_ref_images = []
            ref_image_max_pixels = getattr(pipe, "ref_image_max_pixels", 921600)
            for item in ref_image:
                if DEBUG: print("add reference image", item.size, width, height)
                item = resize_preserving_aspect(item, ref_image_max_pixels, pipe.height_division_factor)
                prepared_ref_images.append(item)
                if DEBUG: print("Resize reference image", item.size, width, height)
                item = pipe.preprocess_video([item])
                if DEBUG: print("Reference image size", item[0].shape)
                with torch.no_grad():
                    item = vae.encode(item, device=pipe.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride).to(dtype=pipe.torch_dtype, device=pipe.device)
                if DEBUG: print("Reference vae size", item.size())
                vae_ref_image.append(item)
        else:
            vae_ref_image = None
            prepared_ref_images = None
        if input_video is None:
            _dump_diffusion_inputs(pipe, height, width, num_frames, None, source_input, prepared_ref_images, noise,
                                   vae_source_input=vae_source_input, vae_ref_image=vae_ref_image)
            return {"latents": noise, "vae_source_input": vae_source_input, "vae_ref_image": vae_ref_image}
        input_video_raw = input_video
        input_video = pipe.preprocess_video(input_video)
        with torch.no_grad():
            input_latents = vae.encode(input_video, device=pipe.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride).to(dtype=pipe.torch_dtype, device=pipe.device)
        _dump_diffusion_inputs(pipe, height, width, num_frames, input_video_raw, source_input, prepared_ref_images, noise,
                               input_latents=input_latents, vae_source_input=vae_source_input, vae_ref_image=vae_ref_image)
        if pipe.scheduler.training:
            return {"latents": noise, "input_latents": input_latents, "vae_source_input": vae_source_input, "vae_ref_image": vae_ref_image}
        else:
            latents = pipe.scheduler.add_noise(input_latents, noise, timestep=pipe.scheduler.timesteps[0])
            return {"latents": latents, "vae_source_input": vae_source_input, "vae_ref_image": vae_ref_image}


class WanVideoUnit_MLLMEmbedder(PipelineUnit):
    def __init__(self):
        super().__init__(
            seperate_cfg=True,
            input_params=("src_video", "ref_image"),
            input_params_posi={"prompt": "prompt"},
            input_params_nega={"prompt": "negative_prompt"},
            onload_model_names=("mllm")
        )

    def process(self, pipe: WanVideoPipeline, prompt, src_video, ref_image=None):
        prompt_emb = pipe.encode_context(
            prompt,
            src_video=src_video,
            ref_image=ref_image,
            debug_dump=getattr(pipe, "debug_dump_state", None),
        )
        if DEBUG: print(prompt_emb.size(), prompt, prompt_emb[0][:5,:5])
        return {"context": prompt_emb}

class WanVideoUnit_CfgMerger(PipelineUnit):
    def __init__(self):
        super().__init__(take_over=True)
        self.concat_tensor_names = ["context", "clip_feature", "y", "reference_latents"]

    def process(self, pipe: WanVideoPipeline, inputs_shared, inputs_posi, inputs_nega):
        if not inputs_shared["cfg_merge"]:
            return inputs_shared, inputs_posi, inputs_nega
        for name in self.concat_tensor_names:
            tensor_posi = inputs_posi.get(name)
            tensor_nega = inputs_nega.get(name)
            tensor_shared = inputs_shared.get(name)
            if tensor_posi is not None and tensor_nega is not None:
                inputs_shared[name] = torch.concat((tensor_posi, tensor_nega), dim=0)
            elif tensor_shared is not None:
                inputs_shared[name] = torch.concat((tensor_shared, tensor_shared), dim=0)
        inputs_posi.clear()
        inputs_nega.clear()
        return inputs_shared, inputs_posi, inputs_nega


class TemporalTiler_BCTHW:
    def __init__(self):
        pass

    def build_1d_mask(self, length, left_bound, right_bound, border_width):
        x = torch.ones((length,))
        if border_width == 0:
            return x
        
        shift = 0.5
        if not left_bound:
            x[:border_width] = (torch.arange(border_width) + shift) / border_width
        if not right_bound:
            x[-border_width:] = torch.flip((torch.arange(border_width) + shift) / border_width, dims=(0,))
        return x

    def build_mask(self, data, is_bound, border_width):
        _, _, T, _, _ = data.shape
        t = self.build_1d_mask(T, is_bound[0], is_bound[1], border_width[0])
        mask = repeat(t, "T -> 1 1 T 1 1")
        return mask
    
    def run(self, model_fn, sliding_window_size, sliding_window_stride, computation_device, computation_dtype, model_kwargs, tensor_names, batch_size=None):
        tensor_names = [tensor_name for tensor_name in tensor_names if model_kwargs.get(tensor_name) is not None]
        tensor_dict = {tensor_name: model_kwargs[tensor_name] for tensor_name in tensor_names}
        B, C, T, H, W = tensor_dict[tensor_names[0]].shape
        if batch_size is not None:
            B *= batch_size
        data_device, data_dtype = tensor_dict[tensor_names[0]].device, tensor_dict[tensor_names[0]].dtype
        value = torch.zeros((B, C, T, H, W), device=data_device, dtype=data_dtype)
        weight = torch.zeros((1, 1, T, 1, 1), device=data_device, dtype=data_dtype)
        for t in range(0, T, sliding_window_stride):
            if t - sliding_window_stride >= 0 and t - sliding_window_stride + sliding_window_size >= T:
                continue
            t_ = min(t + sliding_window_size, T)
            model_kwargs.update({
                tensor_name: tensor_dict[tensor_name][:, :, t: t_:, :].to(device=computation_device, dtype=computation_dtype) \
                    for tensor_name in tensor_names
            })
            model_output = model_fn(**model_kwargs).to(device=data_device, dtype=data_dtype)
            mask = self.build_mask(
                model_output,
                is_bound=(t == 0, t_ == T),
                border_width=(sliding_window_size - sliding_window_stride,)
            ).to(device=data_device, dtype=data_dtype)
            value[:, :, t: t_, :, :] += model_output * mask
            weight[:, :, t: t_, :, :] += mask
        value /= weight
        model_kwargs.update(tensor_dict)
        return value


def model_fn_wan_video(
    dit: WanModel,
    vae_condition: ConditionalEmbedder = None,
    ref_vae_condition: ConditionalEmbedder = None,
    vae_source_input: Optional[torch.Tensor] = None,
    vae_ref_image: Optional[torch.Tensor] = None,
    latents: torch.Tensor = None,
    timestep: torch.Tensor = None,
    context: torch.Tensor = None,
    sliding_window_size: Optional[int] = None,
    sliding_window_stride: Optional[int] = None,
    cfg_merge: bool = False,
    use_gradient_checkpointing: bool = False,
    use_gradient_checkpointing_offload: bool = False,
    control_camera_latents_input = None,
    scheduler: FlowMatchScheduler = None,
    **kwargs,
):
    if sliding_window_size is not None and sliding_window_stride is not None:
        cfg_batch_size = kwargs.get("cfg_batch_size", 2 if cfg_merge else 1)
        model_kwargs = dict(
            dit=dit,
            latents=latents,
            timestep=timestep,
            context=context,
            cfg_batch_size=cfg_batch_size,
            rope_mode=kwargs.get("rope_mode", "anchor"),
            ref_zero_cond_t=kwargs.get("ref_zero_cond_t", True),
            source_zero_cond_t=kwargs.get("source_zero_cond_t", True),
            source_condition_mode=kwargs.get("source_condition_mode", "temporal_concat"),
        )
        return TemporalTiler_BCTHW().run(
            model_fn_wan_video,
            sliding_window_size, sliding_window_stride,
            latents.device, latents.dtype,
            model_kwargs=model_kwargs,
            tensor_names=["latents"],
            batch_size=cfg_batch_size,
        )
    x = latents
    if x.shape[0] != context.shape[0]:
        x = torch.concat([x] * context.shape[0], dim=0)
    if timestep.shape[0] != context.shape[0]:
        timestep = torch.concat([timestep] * context.shape[0], dim=0)
    vae_source_input = _repeat_to_batch(vae_source_input, x.shape[0])
    vae_ref_image = _repeat_to_batch(vae_ref_image, x.shape[0])

    source_condition_mode = _normalize_source_condition_mode(kwargs.get("source_condition_mode", "temporal_concat"))
    rope_mode = kwargs.get("rope_mode", "anchor")
    if rope_mode not in {"anchor", "legacy"}:
        raise ValueError(f"Unsupported rope_mode={rope_mode!r}. Expected 'anchor' or 'legacy'.")
    # AnchorRoPE: reference tokens live on negative time anchors while spatial
    use_anchor_rope = rope_mode == "anchor"
    has_ref_image = _has_ref_condition(vae_ref_image)
    use_ref_zero_cond_t = bool(kwargs.get("ref_zero_cond_t", True)) and has_ref_image
    target_latent_frames = x.shape[2]
    source_token_start = target_latent_frames

    if vae_source_input is not None and source_condition_mode == "temporal_concat":
        if x.shape[3:] != vae_source_input.shape[3:]:
            raise ValueError(
                f"Source latent spatial shape {tuple(vae_source_input.shape[3:])} does not match target {tuple(x.shape[3:])}."
            )
        x = torch.cat([x, vae_source_input], dim=2)
    has_temporal_source_tokens = vae_source_input is not None and source_condition_mode == "temporal_concat"
    use_source_zero_cond_t = bool(kwargs.get("source_zero_cond_t", True)) and has_temporal_source_tokens
    use_zero_cond_t = use_ref_zero_cond_t or use_source_zero_cond_t

    target_t = dit.time_embedding(sinusoidal_embedding_1d(dit.freq_dim, timestep))
    mod_timestep = timestep
    if use_zero_cond_t:
        mod_timestep = torch.cat([timestep, torch.zeros_like(timestep)], dim=0)
    t_mod = dit.time_projection(
        dit.time_embedding(sinusoidal_embedding_1d(dit.freq_dim, mod_timestep))
    ).unflatten(1, (6, dit.dim))

    if vae_source_input is not None and source_condition_mode == "additive":
        if vae_condition is None:
            raise ValueError("vae_condition is required when source_condition_mode='additive'.")
        if scheduler is None:
            raise ValueError("scheduler is required when source_condition_mode='additive'.")
        if DEBUG: print("source cond before:", x.shape, vae_source_input.shape)
        vae_source_input = vae_condition(vae_source_input)
        x = dit.patchify(x, control_camera_latents_input)
        sigma = scheduler.get_sigma(timestep)
        if DEBUG: print("source cond after:", x.shape, vae_source_input.shape, sigma, timestep)
        x = x + vae_source_input * sigma
    else:
        x = dit.patchify(x, control_camera_latents_input)

    f, h, w = x.shape[2:]
    video_tokens = rearrange(x, 'b c f h w -> b (f h w) c').contiguous()
    video_freqs = _build_rotary_freqs(
        dit,
        torch.arange(f, device=x.device),
        h,
        w,
        x.device,
        centered_spatial=use_anchor_rope,
    )
    video_modulate_index = torch.zeros(
        (video_tokens.shape[0], video_tokens.shape[1]),
        device=x.device,
        dtype=torch.long,
    )
    if use_source_zero_cond_t:
        source_frame_mask = torch.arange(f, device=x.device) >= source_token_start
        source_token_mask = repeat(source_frame_mask.long(), "f -> 1 (f h w)", h=h, w=w)
        video_modulate_index = source_token_mask.expand(video_tokens.shape[0], -1).contiguous()
    sequence_sections = [video_tokens]
    freqs_sections = [video_freqs]
    modulate_sections = [video_modulate_index]
    video_token_slice = slice(0, video_tokens.shape[1])

    if has_ref_image:
        ref_sequence_sections = []
        ref_freq_sections = []
        ref_modulate_sections = []
        for ref_index, ref_latent in enumerate(vae_ref_image):
            if DEBUG: print("vae_ref_image before patch embedding:", ref_latent.shape)
            ref_hidden = ref_vae_condition(ref_latent)
            if hasattr(ref_vae_condition, "add_single_item_embedding"):
                ref_hidden = ref_vae_condition.add_single_item_embedding(ref_hidden, ref_index)
            elif hasattr(ref_vae_condition, "add_item_embeddings"):
                ref_hidden = ref_vae_condition.add_item_embeddings(ref_hidden)
            if DEBUG: print("vae_ref_image after patch embedding:", ref_hidden.shape)
            ref_f, ref_h, ref_w = ref_hidden.shape[2:]
            ref_tokens = rearrange(ref_hidden, 'b c f h w -> b (f h w) c').contiguous()
            if use_anchor_rope:
                ref_frame_positions = torch.full(
                    (ref_f,),
                    -(ref_index + 1),
                    device=x.device,
                    dtype=torch.long,
                )
            else:
                ref_frame_positions = torch.full(
                    (ref_f,),
                    ref_index,
                    device=x.device,
                    dtype=torch.long,
                )
            ref_freqs = _build_rotary_freqs(
                dit,
                ref_frame_positions,
                ref_h,
                ref_w,
                x.device,
                centered_spatial=use_anchor_rope,
            )
            ref_sequence_sections.append(ref_tokens)
            ref_freq_sections.append(ref_freqs)
            ref_modulate_sections.append(
                torch.full(
                    (ref_tokens.shape[0], ref_tokens.shape[1]),
                    1 if use_ref_zero_cond_t else 0,
                    device=x.device,
                    dtype=torch.long,
                )
            )

        if ref_vae_condition.ref_pad_first:
            ref_token_count = sum(section.shape[1] for section in ref_sequence_sections)
            sequence_sections = ref_sequence_sections + sequence_sections
            freqs_sections = ref_freq_sections + freqs_sections
            modulate_sections = ref_modulate_sections + modulate_sections
            video_token_slice = slice(ref_token_count, ref_token_count + video_tokens.shape[1])
        else:
            sequence_sections = sequence_sections + ref_sequence_sections
            freqs_sections = freqs_sections + ref_freq_sections
            modulate_sections = modulate_sections + ref_modulate_sections

    x = torch.cat(sequence_sections, dim=1)
    freqs = torch.cat(freqs_sections, dim=0).to(x.device)
    modulate_index = torch.cat(modulate_sections, dim=1) if use_zero_cond_t else None

    
    def create_custom_forward(module):
        def custom_forward(*inputs):
            return module(*inputs)
        return custom_forward

    for block_id, block in enumerate(dit.blocks):
        if use_gradient_checkpointing_offload:
            with torch.autograd.graph.save_on_cpu():
                x = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    x, context, t_mod, freqs, modulate_index,
                    use_reentrant=False,
                )
        elif use_gradient_checkpointing:
            x = deepspeed.checkpointing.checkpoint(
                create_custom_forward(block),
                x, context, t_mod, freqs, modulate_index
            )
        else:
            x = block(x, context, t_mod, freqs, modulate_index=modulate_index)

    x = x[:, video_token_slice, :]
    x = dit.head(x, target_t)
    x = dit.unpatchify(x, (f, h, w))
    if has_temporal_source_tokens:
        x = x[:, :, :source_token_start]
    return x
