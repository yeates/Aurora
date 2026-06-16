import json
import os

import torch
import torch.nn as nn
from PIL import Image
from transformers import AutoConfig, AutoModelForCausalLM, AutoProcessor
from transformers.models.qwen2_vl.image_processing_qwen2_vl import smart_resize as qwen_image_smart_resize
from transformers.video_utils import VideoMetadata


DEBUG = False
PIL_LANCZOS = getattr(getattr(Image, "Resampling", Image), "LANCZOS")


def _rank0():
    return not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0


DEFAULT_VLM_SYSTEM_PROMPT = (
    "You are a video editing assistant. "
    "You will be given source content, optional reference images, "
    "and an editing instruction. "
    "Analyze the request using the provided visuals and describe how to apply the edit."
)
DEFAULT_VLM_VIDEO_FPS = 24.0
DEFAULT_MLLM_REF_MAX_PIXELS = 147456


def _find_subsequence(sequence, subsequence):
    if not subsequence:
        return -1
    for index in range(len(sequence) - len(subsequence) + 1):
        if sequence[index: index + len(subsequence)] == subsequence:
            return index
    return -1


def resize_visual_preserving_aspect(image, target_area, factor=28):
    if target_area is None or target_area <= 0:
        return image.convert("RGB")
    width, height = image.size
    target_height, target_width = qwen_image_smart_resize(
        height,
        width,
        factor=factor,
        min_pixels=0,
        max_pixels=target_area,
    )
    return image.convert("RGB").resize((target_width, target_height), PIL_LANCZOS)


class VLMEncoder(torch.nn.Module):
    def __init__(
        self,
        model_path="Qwen/Qwen3.5-4B",
        dtype=torch.bfloat16,
        device="cuda",
        dit_dim=3072,
        max_pixels_per_frame=512 * 512,
        ref_max_pixels_per_image=DEFAULT_MLLM_REF_MAX_PIXELS,
        video_sample_fps=None,
        video_min_frames=None,
        gradient_checkpointing=False,
        system_prompt=DEFAULT_VLM_SYSTEM_PROMPT,
        drop_system_tokens=True,
    ):
        super().__init__()
        self.model_config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        self.model_type = getattr(self.model_config, "model_type", None)
        if self.model_type not in {"qwen3", "qwen3_5"}:
            raise ValueError(
                "Aurora expects a Qwen3.x multimodal backbone. "
                f"Got model_type={self.model_type!r} from {model_path}"
            )

        device_map = str(device) if isinstance(device, torch.device) else device
        if isinstance(device_map, str) and device_map not in {
            "auto",
            "balanced",
            "balanced_low_0",
            "sequential",
            "cpu",
        }:
            device_map = {"": device_map}

        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=dtype,
            device_map=device_map,
            trust_remote_code=True,
        )
        self.model.eval()
        self.model.requires_grad_(False)
        self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        self.tokenizer = getattr(self.processor, "tokenizer", None)
        if self.tokenizer is None:
            raise ValueError(f"{model_path} processor does not expose a tokenizer")
        self.video_processor = getattr(self.processor, "video_processor", None)
        if self.video_processor is None:
            raise ValueError(f"{model_path} processor does not expose a video_processor")

        hidden_size = self.model.config.hidden_size
        self.dtype = dtype
        self.max_pixels_per_frame = (
            int(max_pixels_per_frame) if max_pixels_per_frame is not None else None
        )
        self.gradient_checkpointing = gradient_checkpointing
        self.system_prompt = system_prompt
        self.drop_system_tokens = drop_system_tokens
        self.default_video_fps = float(DEFAULT_VLM_VIDEO_FPS)
        if video_sample_fps is not None:
            self.video_processor.fps = float(video_sample_fps)
        if video_min_frames is not None:
            self.video_processor.min_frames = int(video_min_frames)
        self.video_sample_fps = float(getattr(self.video_processor, "fps", 0.0))
        self.video_min_frames = int(getattr(self.video_processor, "min_frames", 0))
        self.image_resize_factor = int(
            getattr(self.processor.image_processor, "patch_size", 16)
            * getattr(self.processor.image_processor, "merge_size", 2)
        )
        self.video_resize_factor = int(
            getattr(self.video_processor, "patch_size", self.image_resize_factor)
            * getattr(self.video_processor, "merge_size", 1)
        )
        self.ref_max_pixels = (
            DEFAULT_MLLM_REF_MAX_PIXELS
            if ref_max_pixels_per_image is None
            else int(ref_max_pixels_per_image)
        )
        self.context_projector = nn.Sequential(
            nn.Linear(hidden_size, dit_dim, dtype=dtype),
            nn.GELU(approximate="tanh"),
            nn.Linear(dit_dim, dit_dim, dtype=dtype),
        ).to(device)
        nn.init.zeros_(self.context_projector[2].weight)
        nn.init.zeros_(self.context_projector[2].bias)

        self._drop_idx = self._measure_system_prefix_length() if self.drop_system_tokens else 0
        print(
            model_path,
            "model_type:", self.model_type,
            "processor:", type(self.processor).__name__,
            "manual_video_frame_sampling:", False,
            "default_video_fps:", self.default_video_fps,
            "video_sample_fps:", self.video_sample_fps,
            "video_min_frames:", self.video_min_frames,
            "max_pixels_per_frame:", self.max_pixels_per_frame,
            "ref_max_pixels:", self.ref_max_pixels,
            "image_resize_factor:", self.image_resize_factor,
            "video_resize_factor:", self.video_resize_factor,
            "Using Unified Context Projector:\n", self.context_projector,
            "\nSys Prompt:\n", self.system_prompt,
            "\nSystem Token Drop Idx:", self._drop_idx,
        )

    @property
    def model_device(self):
        return next(self.model.parameters()).device

    def _apply_chat_template(self, messages):
        return self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )

    def _measure_system_prefix_length(self):
        sentinel = "<<<__SENTINEL_USER_TEXT__>>>"
        messages = []
        if self.system_prompt is not None:
            messages.append({"role": "system", "content": [{"type": "text", "text": self.system_prompt}]})
        messages.append({"role": "user", "content": [{"type": "text", "text": sentinel}]})
        rendered = self._apply_chat_template(messages)
        full_ids = self.tokenizer(
            rendered,
            return_tensors="pt",
            padding=False,
            add_special_tokens=False,
        ).input_ids[0].tolist()
        sentinel_ids = self.tokenizer(
            sentinel,
            return_tensors="pt",
            padding=False,
            add_special_tokens=False,
        ).input_ids[0].tolist()
        start = _find_subsequence(full_ids, sentinel_ids)
        if start != -1:
            return int(start)
        prefix = rendered.split(sentinel)[0]
        return int(
            self.tokenizer(
                prefix,
                return_tensors="pt",
                padding=False,
                add_special_tokens=False,
            ).input_ids.shape[1]
        )

    @staticmethod
    def _normalize_ref_image_list(ref_image):
        if ref_image is None:
            return None
        if not isinstance(ref_image, (list, tuple)):
            ref_image = [ref_image]
        ref_image = [item for item in ref_image if item is not None]
        return ref_image or None

    def _prepare_visual(self, image, max_pixels, factor):
        return resize_visual_preserving_aspect(image, max_pixels, factor=factor)

    def _build_video_metadata(self, frames):
        if not frames:
            return None
        width, height = frames[0].size
        duration = len(frames) / self.default_video_fps if self.default_video_fps > 0 else None
        return VideoMetadata(
            total_num_frames=len(frames),
            fps=self.default_video_fps,
            width=width,
            height=height,
            duration=duration,
        )

    def _build_messages(self, instruction, src_image=None, src_video=None, ref_image=None):
        ref_image = self._normalize_ref_image_list(ref_image)
        user_content = []
        image_inputs = []
        video_inputs = []
        video_metadata = []

        if src_video is not None and len(src_video) > 1:
            source_visuals = [
                self._prepare_visual(frame, self.max_pixels_per_frame, self.video_resize_factor)
                for frame in src_video
            ]
            if source_visuals:
                user_content.append({"type": "text", "text": "Source video:"})
                user_content.append({"type": "video", "video": source_visuals, "fps": self.default_video_fps})
                video_inputs.append(source_visuals)
                video_metadata.append(self._build_video_metadata(source_visuals))
        elif src_image is not None or (src_video is not None and len(src_video) == 1):
            source_frame = src_image[0] if src_image else src_video[0]
            source_visuals = [
                self._prepare_visual(source_frame, self.max_pixels_per_frame, self.image_resize_factor)
            ]
            user_content.append({"type": "text", "text": "Source image:"})
            user_content.append({"type": "image", "image": source_visuals[0]})
            image_inputs.append(source_visuals[0])
        else:
            source_visuals = []

        prepared_refs = []
        for index, image in enumerate(ref_image or []):
            prepared = self._prepare_visual(image, self.ref_max_pixels, self.image_resize_factor)
            prepared_refs.append(prepared)
            user_content.append({"type": "text", "text": f"Reference {index + 1}:"})
            user_content.append({"type": "image", "image": prepared})
            image_inputs.append(prepared)

        user_content.append({"type": "text", "text": f"Editing instruction: {instruction}"})
        messages = []
        if self.system_prompt is not None:
            messages.append({"role": "system", "content": [{"type": "text", "text": self.system_prompt}]})
        messages.append({"role": "user", "content": user_content})
        return messages, image_inputs, video_inputs, video_metadata, source_visuals, prepared_refs

    def _extract_user_hidden_states(self, attention_mask, hidden_states):
        batch_size, _, hidden_size = hidden_states.shape
        packed_hidden_states = []
        packed_attention_mask = []
        for batch_id in range(batch_size):
            valid = (attention_mask[batch_id] == 1).nonzero(as_tuple=False).flatten()
            if valid.numel() == 0:
                packed_hidden_states.append(hidden_states.new_zeros((1, hidden_size)))
                packed_attention_mask.append(attention_mask.new_zeros((1,)))
                continue
            first_valid = int(valid.min().item())
            end_idx = int(valid.max().item()) + 1
            start_idx = first_valid + self._drop_idx if self.drop_system_tokens else first_valid
            start_idx = max(first_valid, min(start_idx, end_idx))
            packed_hidden_states.append(hidden_states[batch_id, start_idx:end_idx, :])
            packed_attention_mask.append(attention_mask[batch_id, start_idx:end_idx])

        max_seq_len = max(item.shape[0] for item in packed_hidden_states)
        hidden_states_out = hidden_states.new_zeros((batch_size, max_seq_len, hidden_size))
        attention_mask_out = attention_mask.new_zeros((batch_size, max_seq_len))
        for batch_id, (seq_hidden, seq_mask) in enumerate(zip(packed_hidden_states, packed_attention_mask)):
            seq_len = seq_hidden.shape[0]
            hidden_states_out[batch_id, :seq_len] = seq_hidden
            attention_mask_out[batch_id, :seq_len] = seq_mask
        return hidden_states_out, attention_mask_out

    @staticmethod
    def _save_images(images, out_dir, prefix):
        os.makedirs(out_dir, exist_ok=True)
        for idx, img in enumerate(images):
            img.save(os.path.join(out_dir, f"{prefix}_{idx:03d}.png"))

    @staticmethod
    def _describe_content_part(part):
        part_type = part.get("type", "?")
        if part_type == "text":
            return f"text({len(part.get('text', ''))}ch)"
        if part_type == "image":
            image = part.get("image")
            if image is not None and hasattr(image, "size"):
                width, height = image.size
                return f"image({width}x{height})"
            return "image"
        if part_type == "video":
            video = part.get("video") or []
            frame_count = len(video) if isinstance(video, (list, tuple)) else 0
            fps = part.get("fps")
            if frame_count and fps is not None:
                return f"video({frame_count}f@{float(fps):g}fps)"
            if frame_count:
                return f"video({frame_count}f)"
            return "video"
        return part_type

    @classmethod
    def _simplify_messages(cls, messages):
        simplified = []
        for msg in messages:
            role = msg.get("role", "?")
            raw_content = msg.get("content", "")
            parts = []
            if isinstance(raw_content, list):
                for part in raw_content:
                    parts.append(cls._describe_content_part(part))
            elif isinstance(raw_content, str):
                parts.append(f"text({len(raw_content)}ch)")
            else:
                parts.append(str(type(raw_content)))
            simplified.append({"role": role, "parts": parts})
        return simplified

    @staticmethod
    def _serialize_video_metadata(video_metadata):
        if not video_metadata:
            return None
        serialized = []
        for item in video_metadata:
            if item is None:
                serialized.append(None)
                continue
            serialized.append(
                {
                    "total_num_frames": int(item.total_num_frames),
                    "fps": float(item.fps) if item.fps is not None else None,
                    "width": int(item.width) if item.width is not None else None,
                    "height": int(item.height) if item.height is not None else None,
                    "duration": float(item.duration) if item.duration is not None else None,
                    "video_backend": item.video_backend,
                    "frames_indices": [int(index) for index in item.frames_indices] if item.frames_indices else None,
                }
            )
        return serialized

    @staticmethod
    def _tensor_shape(value):
        if value is None or not torch.is_tensor(value):
            return None
        return [int(dim) for dim in value.shape]

    @staticmethod
    def _extract_mm_token_stats(inputs):
        mm_token_type_ids = inputs.get("mm_token_type_ids")
        if mm_token_type_ids is None:
            return {
                "mm_token_type_counts": None,
                "text_token_count": None,
                "image_token_count": None,
                "video_token_count": None,
            }
        counts = {}
        for value in mm_token_type_ids[0].detach().cpu().tolist():
            key = str(int(value))
            counts[key] = counts.get(key, 0) + 1
        return {
            "mm_token_type_counts": counts,
            "text_token_count": counts.get("0", 0),
            "image_token_count": counts.get("1", 0),
            "video_token_count": counts.get("2", 0),
        }

    def _dump_vlm_inputs(self, debug_dump, instruction, mode, source_visuals, ref_image, video_metadata, messages, text, inputs):
        if not debug_dump or debug_dump.get("vlm_dumped"):
            return
        out_dir = os.path.join(debug_dump["step_dir"], "vlm")
        os.makedirs(out_dir, exist_ok=True)
        if source_visuals:
            prefix = "src_frame" if mode == "video" else "src_image"
            self._save_images(source_visuals, out_dir, prefix)
        if ref_image:
            self._save_images(ref_image, out_dir, "ref_image")
        mm_stats = self._extract_mm_token_stats(inputs)
        meta = {
            "step": debug_dump["step"],
            "rank": debug_dump.get("rank"),
            "dataset": debug_dump.get("dataset"),
            "mode": mode,
            "instruction": instruction,
            "n_source": len(source_visuals) if source_visuals else 0,
            "n_ref": len(ref_image) if ref_image else 0,
            "source_frame_size": (
                {"width": int(source_visuals[0].size[0]), "height": int(source_visuals[0].size[1])}
                if source_visuals
                else None
            ),
            "source_video_metadata": self._serialize_video_metadata(video_metadata),
            "processor_sampling": {
                "do_sample_frames": bool(video_metadata),
                "default_video_fps": self.default_video_fps if video_metadata else None,
                "video_sample_fps": self.video_sample_fps if video_metadata else None,
                "video_min_frames": self.video_min_frames if video_metadata else None,
            },
            "input_ids_len": int(inputs["input_ids"].shape[1]),
            "vlm_input_tokens": int(inputs["input_ids"].shape[1]),
            "attention_mask_len": int(inputs["attention_mask"].shape[1]),
            "text_token_count": mm_stats["text_token_count"],
            "image_token_count": mm_stats["image_token_count"],
            "video_token_count": mm_stats["video_token_count"],
            "mm_token_type_counts": mm_stats["mm_token_type_counts"],
            "system_drop_idx": self._drop_idx,
            "processor_keys": sorted(inputs.keys()),
            "input_ids_shape": self._tensor_shape(inputs.get("input_ids")),
            "pixel_values_shape": self._tensor_shape(inputs.get("pixel_values")),
            "pixel_values_videos_shape": self._tensor_shape(inputs.get("pixel_values_videos")),
            "video_grid_thw": (
                inputs["video_grid_thw"].detach().cpu().tolist()
                if "video_grid_thw" in inputs
                else None
            ),
            "messages": self._simplify_messages(messages),
            "chat_template_text": text,
        }
        with open(os.path.join(out_dir, "metadata.json"), "w") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)
        debug_dump["vlm_dumped"] = True

    def _tokenize_messages(self, messages, images, videos, video_metadata):
        text = self._apply_chat_template(messages)
        processor_kwargs = dict(
            text=[text],
            images=images if images else None,
            videos=videos if videos else None,
            padding=True,
            return_tensors="pt",
        )
        if videos:
            processor_kwargs["video_metadata"] = video_metadata
            processor_kwargs["do_sample_frames"] = True
        inputs = self.processor(**processor_kwargs)
        return text, inputs

    def _move_inputs_to_model(self, inputs):
        inputs = inputs.to(self.model_device)
        for name, value in list(inputs.items()):
            if torch.is_tensor(value) and value.is_floating_point():
                inputs[name] = value.to(dtype=self.dtype)
        return inputs

    def forward(
        self,
        instruction,
        src_image=None,
        src_video=None,
        ref_image=None,
        **kwargs,
    ):
        debug_dump = kwargs.pop("debug_dump", None)
        ref_image = self._normalize_ref_image_list(ref_image)
        messages, images, videos, video_metadata, source_visuals, prepared_refs = self._build_messages(
            instruction=instruction,
            src_image=src_image,
            src_video=src_video,
            ref_image=ref_image,
        )
        text, inputs = self._tokenize_messages(messages, images, videos, video_metadata)

        has_video = src_video is not None and len(src_video) > 1
        has_image = src_image is not None or (src_video is not None and len(src_video) == 1)
        if has_video:
            mode = "video"
        elif has_image:
            mode = "image"
        else:
            mode = "ref_only"
        self._dump_vlm_inputs(debug_dump, instruction, mode, source_visuals, prepared_refs, video_metadata, messages, text, inputs)

        inputs = self._move_inputs_to_model(inputs)

        with torch.no_grad():
            outputs = self.model(
                **inputs,
                output_attentions=False,
                output_hidden_states=True,
                return_dict=True,
                **kwargs,
            )
        hidden_states = outputs.hidden_states[-1].detach()
        attention_mask = inputs["attention_mask"]
        hidden_states, attention_mask = self._extract_user_hidden_states(attention_mask, hidden_states)
        hidden_states = hidden_states.to(
            device=self.context_projector[0].weight.device,
            dtype=self.context_projector[0].weight.dtype,
        )
        attention_mask = attention_mask.to(hidden_states.device)
        context = self.context_projector(hidden_states)
        context = context * attention_mask.unsqueeze(-1).to(context.dtype)
        return context
