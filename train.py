import fcntl
import hashlib
import json
import os
import time
import torch
from diffsynth.trainers.engine import DiffusionTrainingModule, MixDataloader, ModelLogger, launch_mix_training_task, wan_parser
from diffsynth.trainers.dataset import UnifiedDataset
os.environ["TOKENIZERS_PARALLELISM"] = "false"


_DATASET_SEED_OFFSETS = {
    "img": 0x13579BDF,
    "vid": 0x2468ACE0,
    "vid_ref": 0x10203040,
}

_DATASET_REPEAT_ARG_NAMES = {
    "img": "img_dataset_repeat",
    "vid": "vid_dataset_repeat",
    "vid_ref": "vid_ref_dataset_repeat",
}


def _env_global_rank():
    for key in ("RANK", "PROCESS_RANK", "LOCAL_RANK"):
        value = os.environ.get(key)
        if value is not None:
            return int(value)
    return 0


def _is_env_rank0():
    return _env_global_rank() == 0


def _env_world_size():
    value = os.environ.get("WORLD_SIZE")
    return int(value) if value is not None else 1


def _is_training_state_dir(path):
    if not path or not os.path.isdir(path):
        return False
    return os.path.isfile(os.path.join(path, "training_meta.json"))


def _resolve_resume_state_path(output_path, explicit_path):
    if explicit_path:
        if not os.path.exists(explicit_path):
            raise ValueError(
                "Expected --resume_from_training_state to point to a training-state directory, "
                f"but the path does not exist: {explicit_path}"
            )
        if not os.path.isdir(explicit_path):
            raise ValueError(
                "Expected --resume_from_training_state to point to a training-state directory, "
                f"but got a file: {explicit_path}"
            )
        if not _is_training_state_dir(explicit_path):
            raise ValueError(
                "Expected --resume_from_training_state to point to a training-state directory "
                "containing training_meta.json (for example, <output_path>/training_state_latest), "
                f"but got: {explicit_path}"
            )
        return explicit_path
    latest_state = os.path.join(output_path, "training_state_latest")
    if os.path.isdir(latest_state):
        return latest_state
    return None


def _load_training_state_meta(state_path):
    if not state_path:
        return {}
    meta_path = os.path.join(state_path, "training_meta.json")
    if not os.path.exists(meta_path):
        return {}
    with open(meta_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _active_dataset_types(args):
    active_types = []
    if args.img_dataset_metadata_path:
        active_types.append("img")
    if args.vid_dataset_metadata_path:
        active_types.append("vid")
    if args.vid_ref_dataset_metadata_path:
        active_types.append("vid_ref")
    return active_types


def _parse_metadata_shuffle_seed_arg(value):
    if value is None:
        return None
    value = str(value).strip().lower()
    if value in {"", "none", "off", "false"}:
        return None
    if value == "per_run":
        return value
    return int(value)


def _derive_metadata_shuffle_seed_map(base_seed, active_types):
    if base_seed is None:
        return {}
    return {
        dataset_type: int(base_seed) + _DATASET_SEED_OFFSETS[dataset_type]
        for dataset_type in active_types
    }


def _data_order_signature(args, dataset_mix_pattern, dataset_mix_mode):
    dataset_repeat_signature = _dataset_repeat_signature(args)
    return {
        "dataset_mix_pattern": dataset_mix_pattern,
        "dataset_mix_mode": dataset_mix_mode,
        "img_dataset_metadata_path": args.img_dataset_metadata_path,
        "vid_dataset_metadata_path": args.vid_dataset_metadata_path,
        "vid_ref_dataset_metadata_path": args.vid_ref_dataset_metadata_path,
        "vid_metadata_exclude_rules": args.vid_metadata_exclude_rules,
        **dataset_repeat_signature,
    }


def _validate_positive_repeat(name, value):
    value = int(value)
    if value < 1:
        raise ValueError(f"{name} must be >= 1, got {value}")
    return value


def _dataset_repeat_signature(args):
    signature = {
        "dataset_repeat": _validate_positive_repeat("dataset_repeat", args.dataset_repeat),
        "auto_balance_dataset_repeats": bool(args.auto_balance_dataset_repeats),
    }
    for dataset_type, arg_name in _DATASET_REPEAT_ARG_NAMES.items():
        raw_value = getattr(args, arg_name)
        signature[arg_name] = None if raw_value is None else _validate_positive_repeat(arg_name, raw_value)
    return signature


def _requested_dataset_repeats(args, active_types):
    base_repeat = _validate_positive_repeat("dataset_repeat", args.dataset_repeat)
    requested = {}
    for dataset_type in active_types:
        arg_name = _DATASET_REPEAT_ARG_NAMES[dataset_type]
        raw_value = getattr(args, arg_name)
        requested[dataset_type] = (
            base_repeat if raw_value is None else _validate_positive_repeat(arg_name, raw_value)
        )
    return requested


def _dataset_raw_length(dataset):
    if dataset is None:
        return 0
    if getattr(dataset, "load_from_cache", False):
        return len(dataset.cached_data)
    return len(dataset.data)


def _ceil_div(num, den):
    return (num + den - 1) // den


def _pattern_counts_by_type(dataset_mix_pattern):
    counts = {}
    for sample_type in MixDataloader._parse_pattern(dataset_mix_pattern):
        dataset_type = MixDataloader._TYPE_NAME_BY_ID[sample_type]
        counts[dataset_type] = counts.get(dataset_type, 0) + 1
    return counts


def _normalize_repeat_map(repeat_map, active_types, field_name):
    normalized = {}
    if repeat_map is None:
        return normalized
    for dataset_type in active_types:
        if dataset_type not in repeat_map:
            raise ValueError(f"{field_name} is missing repeat for dataset_type={dataset_type!r}")
        normalized[dataset_type] = _validate_positive_repeat(
            f"{field_name}[{dataset_type}]",
            repeat_map[dataset_type],
        )
    return normalized


def _compute_auto_balanced_dataset_repeats(raw_lengths, requested_repeats, dataset_mix_pattern, world_size):
    if world_size < 1:
        raise ValueError(f"world_size must be >= 1, got {world_size}")
    pattern_counts = _pattern_counts_by_type(dataset_mix_pattern)
    if not pattern_counts:
        raise ValueError("Cannot auto-balance dataset repeats without an active dataset mix pattern.")

    target_cycles = 0
    for dataset_type, pattern_count in pattern_counts.items():
        raw_length = raw_lengths.get(dataset_type, 0)
        requested_repeat = requested_repeats.get(dataset_type)
        if requested_repeat is None:
            raise ValueError(f"Missing requested repeat for dataset_type={dataset_type!r}")
        if raw_length <= 0:
            raise ValueError(f"Dataset {dataset_type!r} has no samples to auto-balance.")
        target_cycles = max(
            target_cycles,
            (raw_length * requested_repeat) // (world_size * pattern_count),
        )
    if target_cycles <= 0:
        raise ValueError(
            "Auto-balanced dataset repeats still yield zero per-rank cycles. "
            f"raw_lengths={raw_lengths}, requested_repeats={requested_repeats}, "
            f"dataset_mix_pattern={dataset_mix_pattern!r}, world_size={world_size}"
        )

    realized = dict(requested_repeats)
    for dataset_type, pattern_count in pattern_counts.items():
        required_total = target_cycles * world_size * pattern_count
        realized[dataset_type] = max(
            requested_repeats[dataset_type],
            _ceil_div(required_total, raw_lengths[dataset_type]),
        )
    return realized, target_cycles, pattern_counts


def _apply_dataset_repeat_policy(args, run_training_meta, datasets_by_type):
    active_types = [dataset_type for dataset_type, dataset in datasets_by_type.items() if dataset is not None]
    requested_repeats = run_training_meta.get("requested_dataset_repeats")
    if requested_repeats is None:
        requested_repeats = _requested_dataset_repeats(args, active_types)
    else:
        requested_repeats = _normalize_repeat_map(
            requested_repeats,
            active_types,
            "requested_dataset_repeats",
        )
    run_training_meta["requested_dataset_repeats"] = dict(requested_repeats)

    saved_realized_repeats = run_training_meta.get("realized_dataset_repeats")
    raw_lengths = {
        dataset_type: _dataset_raw_length(dataset)
        for dataset_type, dataset in datasets_by_type.items()
        if dataset is not None
    }

    if saved_realized_repeats is not None:
        realized_repeats = _normalize_repeat_map(
            saved_realized_repeats,
            active_types,
            "realized_dataset_repeats",
        )
        target_cycles = run_training_meta.get("auto_balance_target_cycles_per_rank")
        pattern_counts = run_training_meta.get("auto_balance_pattern_counts")
        saved_world_size = run_training_meta.get("auto_balance_world_size")
        if bool(run_training_meta.get("auto_balance_dataset_repeats")) and saved_world_size is not None:
            current_world_size = _env_world_size()
            if int(saved_world_size) != current_world_size:
                raise ValueError(
                    "Refusing to resume auto-balanced training with a different WORLD_SIZE. "
                    f"saved={saved_world_size}, current={current_world_size}."
                )
    elif args.auto_balance_dataset_repeats:
        realized_repeats, target_cycles, pattern_counts = _compute_auto_balanced_dataset_repeats(
            raw_lengths=raw_lengths,
            requested_repeats=requested_repeats,
            dataset_mix_pattern=run_training_meta["dataset_mix_pattern"],
            world_size=_env_world_size(),
        )
        run_training_meta["auto_balance_world_size"] = _env_world_size()
        run_training_meta["auto_balance_target_cycles_per_rank"] = int(target_cycles)
        run_training_meta["auto_balance_pattern_counts"] = dict(pattern_counts)
    else:
        realized_repeats = dict(requested_repeats)
        target_cycles = None
        pattern_counts = None

    for dataset_type, dataset in datasets_by_type.items():
        if dataset is not None:
            dataset.repeat = realized_repeats[dataset_type]
    run_training_meta["realized_dataset_repeats"] = {
        dataset_type: int(repeat)
        for dataset_type, repeat in realized_repeats.items()
    }

    if _is_env_rank0():
        print(f"Requested dataset repeats: {requested_repeats}")
        print(f"Realized dataset repeats: {run_training_meta['realized_dataset_repeats']}")
        print(f"Raw dataset sizes before repeats: {raw_lengths}")
        if target_cycles is not None:
            print(
                "Auto-balanced repeat policy: "
                f"world_size={_env_world_size()}, "
                f"target_cycles_per_rank={int(target_cycles)}, "
                f"pattern_counts={pattern_counts}"
            )
    return realized_repeats


def _compute_launch_token(output_path):
    launch_fields = {
        "output_path": os.path.abspath(output_path),
        "master_addr": os.environ.get("MASTER_ADDR"),
        "master_port": os.environ.get("MASTER_PORT"),
        "world_size": os.environ.get("WORLD_SIZE"),
        "local_world_size": os.environ.get("LOCAL_WORLD_SIZE"),
        "num_nodes": os.environ.get("NUM_NODES") or os.environ.get("NNODES"),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "submit_job_id": os.environ.get("SUBMIT_JOB_ID"),
        "job_id": os.environ.get("JOB_ID"),
        "torchelastic_run_id": os.environ.get("TORCHELASTIC_RUN_ID"),
    }
    payload = json.dumps(launch_fields, sort_keys=True)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]


def _atomic_write_json(path, payload):
    tmp_path = f"{path}.tmp.{os.getpid()}"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
    os.replace(tmp_path, path)


def _load_json_if_exists(path):
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _all_ranks_joined(joined_ranks, world_size):
    joined_rank_set = {int(rank) for rank in joined_ranks}
    return all(rank in joined_rank_set for rank in range(world_size))


def _resolve_per_run_training_meta(args, signature, active_types):
    run_meta_path = os.path.join(
        args.output_path,
        f"training_run_meta_{_compute_launch_token(args.output_path)}.json",
    )
    lock_path = f"{run_meta_path}.lock"
    world_size = _env_world_size()
    rank = _env_global_rank()

    with open(lock_path, "a+", encoding="utf-8") as lock_file:
        fcntl.flock(lock_file, fcntl.LOCK_EX)
        try:
            state = _load_json_if_exists(run_meta_path)
        except json.JSONDecodeError:
            state = {}

        state_world_size = int(state.get("world_size", world_size))
        joined_ranks = state.get("joined_ranks") or []
        saved_run_meta = state.get("run_meta") or {}
        signature_mismatch = any(saved_run_meta.get(key) != value for key, value in signature.items())
        needs_new_generation = (
            "run_meta" not in state
            or state_world_size != world_size
            or signature_mismatch
            or _all_ranks_joined(joined_ranks, state_world_size)
        )
        if needs_new_generation:
            fresh_seed = int.from_bytes(os.urandom(8), "big") & 0x7FFFFFFF
            state = {
                "generation": int(state.get("generation", 0)) + 1,
                "world_size": world_size,
                "joined_ranks": [],
                "run_meta": {
                    **signature,
                    "metadata_shuffle_seed": fresh_seed,
                    "metadata_shuffle_seed_map": _derive_metadata_shuffle_seed_map(fresh_seed, active_types),
                },
            }

        if rank not in state["joined_ranks"]:
            state["joined_ranks"].append(rank)
            state["joined_ranks"].sort()
            _atomic_write_json(run_meta_path, state)

    return state["run_meta"]


def _resolve_run_training_meta(args):
    active_types = _active_dataset_types(args)
    dataset_mix_pattern = MixDataloader.canonicalize_pattern(args.dataset_mix_pattern, active_types)
    dataset_mix_mode = MixDataloader.canonicalize_mix_mode(args.dataset_mix_mode)
    signature = _data_order_signature(args, dataset_mix_pattern, dataset_mix_mode)
    resume_state_path = _resolve_resume_state_path(args.output_path, args.resume_from_training_state)
    saved_meta = _load_training_state_meta(resume_state_path)
    requested_seed = _parse_metadata_shuffle_seed_arg(args.metadata_shuffle_seed)

    if resume_state_path:
        saved_pattern = saved_meta.get("dataset_mix_pattern")
        if saved_pattern is None:
            saved_pattern = MixDataloader.canonicalize_pattern(None, active_types)
        else:
            saved_pattern = MixDataloader.canonicalize_pattern(saved_pattern, active_types)
        if dataset_mix_pattern != saved_pattern:
            raise ValueError(
                "Refusing to resume with a different dataset_mix_pattern. "
                f"saved={saved_pattern!r}, requested={dataset_mix_pattern!r}. "
                "Start a fresh run or keep the original pattern."
            )
        saved_mix_mode = MixDataloader.canonicalize_mix_mode(saved_meta.get("dataset_mix_mode", "sequential"))
        if dataset_mix_mode != saved_mix_mode:
            raise ValueError(
                "Refusing to resume with a different dataset_mix_mode. "
                f"saved={saved_mix_mode!r}, requested={dataset_mix_mode!r}. "
                "Start a fresh run or keep the original mix mode."
            )
        saved_seed = saved_meta.get("metadata_shuffle_seed")
        if saved_seed is None and requested_seed not in (None,):
            raise ValueError(
                "Refusing to resume an old training state with metadata shuffling enabled, "
                "because the original shuffle seed was never saved."
            )
        if isinstance(requested_seed, int) and saved_seed is not None and int(saved_seed) != requested_seed:
            raise ValueError(
                "Refusing to resume with a different metadata shuffle seed. "
                f"saved={saved_seed}, requested={requested_seed}."
            )
        for key in (
            "img_dataset_metadata_path",
            "vid_dataset_metadata_path",
            "vid_ref_dataset_metadata_path",
            "vid_metadata_exclude_rules",
            "dataset_repeat",
            "img_dataset_repeat",
            "vid_dataset_repeat",
            "vid_ref_dataset_repeat",
            "auto_balance_dataset_repeats",
        ):
            if key not in saved_meta:
                if key in {"img_dataset_repeat", "vid_dataset_repeat", "vid_ref_dataset_repeat"} and signature[key] is None:
                    continue
                if key == "auto_balance_dataset_repeats" and not signature[key]:
                    continue
                raise ValueError(
                    f"Refusing to resume because {key} was not saved in {resume_state_path}. "
                    "Start a fresh run or keep the original repeat-policy settings."
                )
            saved_value = saved_meta[key]
            if saved_value != signature[key]:
                raise ValueError(
                    f"Refusing to resume because {key} changed: saved={saved_value!r}, current={signature[key]!r}"
                )
        signature["dataset_mix_pattern"] = saved_pattern
        signature["dataset_mix_mode"] = saved_mix_mode
        signature["metadata_shuffle_seed"] = saved_seed
        signature["metadata_shuffle_seed_map"] = saved_meta.get("metadata_shuffle_seed_map") or _derive_metadata_shuffle_seed_map(saved_seed, active_types)
        for key in (
            "requested_dataset_repeats",
            "realized_dataset_repeats",
            "auto_balance_world_size",
            "auto_balance_target_cycles_per_rank",
            "auto_balance_pattern_counts",
        ):
            if key in saved_meta:
                signature[key] = saved_meta[key]
        return signature, resume_state_path

    if requested_seed == "per_run":
        os.makedirs(args.output_path, exist_ok=True)
        return _resolve_per_run_training_meta(args, signature, active_types), None

    fixed_seed = requested_seed if isinstance(requested_seed, int) else None
    signature["metadata_shuffle_seed"] = fixed_seed
    signature["metadata_shuffle_seed_map"] = _derive_metadata_shuffle_seed_map(fixed_seed, active_types)
    return signature, None

class WanTrainingModule(DiffusionTrainingModule):
    def __init__(
        self,
        model_paths=None, model_id_with_origin_paths=None, audio_processor_config=None,
        trainable_models=None,
        use_gradient_checkpointing=True,
        use_gradient_checkpointing_offload=False,
        extra_inputs=None,
        max_timestep_boundary=1.0,
        min_timestep_boundary=0.0,
        checkpoint=None,
        mllm_model='Qwen/Qwen3.5-4B',
        mllm_max_pixels_per_frame=512*512,
        mllm_ref_max_pixels=147456,
        mllm_video_sample_fps=1.0,
        mllm_video_min_frames=2,
        mllm_gradient_checkpointing=False,
        ref_pad_first=False,
        ref_max_items=8,
        ref_image_max_pixels=921600,
        rope_mode="anchor",
        source_condition_mode="auto",
        source_zero_cond_t=True,
        debug_dump_every=10,
        debug_dump_limit=10,
        debug_dump_num_ranks=1,
        debug_dump_max_frames=4,
        debug_dump_dir=None,
    ):
        super().__init__()
        from diffsynth.pipelines import wan_video as wan_video_pipeline_module

        model_configs = self.parse_model_configs(model_paths, model_id_with_origin_paths, enable_fp8_training=False)
        print(model_configs)
        self._pipeline_debug = wan_video_pipeline_module.DEBUG
        self.pipe = wan_video_pipeline_module.WanVideoPipeline.from_pretrained(
            torch_dtype=torch.bfloat16,
            device="cpu",
            model_configs=model_configs,
            audio_processor_config=audio_processor_config,
            mllm_model=mllm_model,
            mllm_max_pixels_per_frame=mllm_max_pixels_per_frame,
            mllm_ref_max_pixels=mllm_ref_max_pixels,
            mllm_video_sample_fps=mllm_video_sample_fps,
            mllm_video_min_frames=mllm_video_min_frames,
            mllm_gradient_checkpointing=mllm_gradient_checkpointing,
            ref_pad_first=ref_pad_first,
            ref_max_items=ref_max_items,
            ref_image_max_pixels=ref_image_max_pixels,
            rope_mode=rope_mode,
            ref_zero_cond_t=True,
            source_condition_mode=source_condition_mode,
            source_zero_cond_t=source_zero_cond_t,
        )
        self.switch_pipe_to_training_mode(
            self.pipe, trainable_models,
            checkpoint=checkpoint,
        )

        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.use_gradient_checkpointing_offload = use_gradient_checkpointing_offload
        self.extra_inputs = extra_inputs.split(",") if extra_inputs is not None else []
        self.max_timestep_boundary = max_timestep_boundary
        self.min_timestep_boundary = min_timestep_boundary

        self.debug_dump_every = max(0, debug_dump_every)
        self.debug_dump_limit = debug_dump_limit
        self.debug_dump_num_ranks = max(0, int(debug_dump_num_ranks))
        self.debug_dump_max_frames = max(1, debug_dump_max_frames)
        self.debug_dump_dir = debug_dump_dir
        self._debug_dump_count = 0
        self._debug_step_offset = 0
        self._forward_calls = 0

    def _current_rank(self):
        return torch.distributed.get_rank() if torch.distributed.is_initialized() else 0

    def set_debug_dump_root(self, output_path):
        if self.debug_dump_dir is None:
            self.debug_dump_dir = os.path.join(output_path, "debug_inputs")

    def set_debug_step_offset(self, step_offset):
        self._debug_step_offset = step_offset

    def _build_debug_dump_state(self, data):
        self._forward_calls += 1
        step_id = self._debug_step_offset + self._forward_calls
        rank = self._current_rank()
        should_dump = (
            self.debug_dump_every > 0
            and self.debug_dump_num_ranks > 0
            and rank < self.debug_dump_num_ranks
            and step_id % self.debug_dump_every == 0
            and (self.debug_dump_limit <= 0 or self._debug_dump_count < self.debug_dump_limit)
            and self.debug_dump_dir is not None
        )
        if not should_dump:
            self.pipe.debug_dump_state = None
            return
        self._debug_dump_count += 1
        self.pipe.debug_dump_state = {
            "step": step_id,
            "rank": rank,
            "dataset": data.get("dataset", "unknown"),
            "step_dir": os.path.join(
                self.debug_dump_dir,
                f"rank_{rank:03d}",
                f"step_{step_id:06d}_{data.get('dataset', 'unknown')}",
            ),
            "max_frames": self.debug_dump_max_frames,
        }

    def forward_preprocess(self, data, vae=None):
        self._build_debug_dump_state(data)
        # CFG-sensitive parameters
        inputs_posi = {"prompt": data["prompt"]}
        inputs_nega = {}
        
        # CFG-unsensitive parameters
        inputs_shared = {
            "prompt": data["prompt"],
            "input_video": data["tgt_video"],
            "src_video": data.get("src_video"),
            "height": data["tgt_video"][0].size[1],
            "width": data["tgt_video"][0].size[0],
            "num_frames": len(data["tgt_video"]),
            "cfg_scale": 1,
            "tiled": False,
            "rand_device": self.pipe.device,
            "use_gradient_checkpointing": self.use_gradient_checkpointing,
            "use_gradient_checkpointing_offload": self.use_gradient_checkpointing_offload,
            "cfg_merge": False,
            "vace_scale": 1,
            "max_timestep_boundary": self.max_timestep_boundary,
            "min_timestep_boundary": self.min_timestep_boundary,
            "vae": vae
        }
        
        for extra_input in self.extra_inputs:
            if extra_input == "source_input":
                inputs_shared["source_input"] = data.get("src_video")
            elif extra_input == "ref_image":
                if "ref_image" in data:
                    inputs_shared["ref_image"] = data["ref_image"]
                else:
                    inputs_shared["ref_image"] = None
        for unit in self.pipe.units:
            inputs_shared, inputs_posi, inputs_nega = self.pipe.unit_runner(unit, self.pipe, inputs_shared, inputs_posi, inputs_nega)
        return {**inputs_shared, **inputs_posi}
    
    
    def forward(self, data, inputs=None, vae=None):
        if self._pipeline_debug: print("WanTrainingModule Raw Input", data.keys())
        if inputs is None: inputs = self.forward_preprocess(data, vae)
        models = {name: getattr(self.pipe, name) for name in self.pipe.in_iteration_models}
        loss = self.pipe.training_loss(**models, **inputs)
        self.pipe.debug_dump_state = None
        return loss


if __name__ == "__main__":
    parser = wan_parser()
    parser.add_argument("--lmdb_roots", type=str, default=None, help="LMDB root mapping, e.g. 'pico_banana_sft=/path,unicedit_10m=/path'")
    parser.add_argument("--ref_image_max_pixels", type=int, default=921600, help="Per-reference image resize budget before VAE encoding.")
    parser.add_argument("--rope_mode", type=str, default="anchor", help="RoPE variant to use for visual tokens.")
    args = parser.parse_args()
    run_training_meta, resume_state_path = _resolve_run_training_meta(args)
    args.dataset_mix_pattern = run_training_meta["dataset_mix_pattern"]
    args.dataset_mix_mode = run_training_meta["dataset_mix_mode"]
    args.resume_from_training_state = resume_state_path
    shuffle_seed_map = run_training_meta.get("metadata_shuffle_seed_map", {})
    if _is_env_rank0():
        print(f"Resolved dataset_mix_pattern: {args.dataset_mix_pattern}")
        print(f"Resolved dataset_mix_mode: {args.dataset_mix_mode}")
        if run_training_meta.get("metadata_shuffle_seed") is None:
            print("Metadata shuffle is disabled.")
        else:
            print(
                f"Resolved metadata shuffle base seed {run_training_meta['metadata_shuffle_seed']} "
                f"with per-dataset seeds {shuffle_seed_map}"
            )
        if resume_state_path:
            print(f"Resuming data-order config from {resume_state_path}")
    lmdb_roots = dict(kv.split("=", 1) for kv in args.lmdb_roots.split(",")) if args.lmdb_roots else None

    if args.tar_shard_index:
        from diffsynth.utils.tar_shard import set_active_shard_index
        set_active_shard_index(args.tar_shard_index)
        if _is_env_rank0():
            print(f"Tar shard index enabled: {args.tar_shard_index}")

    neg_prompt_pool_list = None
    if args.neg_prompt_given_drop_prob > 0.0 and args.neg_prompt_pool_file:
        import json as _json
        _pool_path = args.neg_prompt_pool_file
        if not os.path.exists(_pool_path):
            raise FileNotFoundError(f"--neg_prompt_pool_file not found: {_pool_path}")
        with open(_pool_path, "r", encoding="utf-8") as _fh:
            _raw = _fh.read()
        try:
            _parsed = _json.loads(_raw)
            if not isinstance(_parsed, list):
                raise ValueError("not a list")
            neg_prompt_pool_list = [str(x).strip() for x in _parsed if str(x).strip()]
        except (ValueError, _json.JSONDecodeError):
            neg_prompt_pool_list = [
                line.strip() for line in _raw.splitlines()
                if line.strip() and not line.strip().startswith("#")
            ]
        if not neg_prompt_pool_list:
            raise ValueError(f"--neg_prompt_pool_file has no usable entries: {_pool_path}")
        if _is_env_rank0():
            print(
                f"Negative-prompt dropout ON: prob={args.neg_prompt_given_drop_prob}, "
                f"pool_size={len(neg_prompt_pool_list)} loaded from {_pool_path}"
            )
    elif args.neg_prompt_given_drop_prob > 0.0:
        if _is_env_rank0():
            print(
                f"Negative-prompt dropout ON: prob={args.neg_prompt_given_drop_prob}, "
                "using built-in default pool"
            )

    special_operator_map = {
        "ref_image": UnifiedDataset.default_image_load_operator(
            base_path=args.dataset_base_path,
            lmdb_roots=lmdb_roots,
        ),
        "ref_mask": UnifiedDataset.default_mask_load_operator(
            base_path=args.dataset_base_path,
        ),
    }
    if args.img_dataset_metadata_path:
        dataset = UnifiedDataset(
            base_path=args.dataset_base_path,
            metadata_path=args.img_dataset_metadata_path,
            repeat=args.dataset_repeat,
            data_file_keys=args.data_file_keys.split(","),
            main_data_operator=UnifiedDataset.default_video_operator(
                base_path=args.dataset_base_path,
                max_pixels=args.max_pixels,
                height=args.height,
                width=args.width,
                height_division_factor=32,
                width_division_factor=32,
                lmdb_roots=lmdb_roots,
                num_frames=1,
                time_division_factor=4,
                time_division_remainder=1,
            ),
            paired_image_operator=UnifiedDataset.default_paired_image_operator(
                base_path=args.dataset_base_path,
                max_pixels=args.max_pixels,
                height=args.height,
                width=args.width,
                height_division_factor=32,
                width_division_factor=32,
                lmdb_roots=lmdb_roots,
            ),
            special_operator_map=special_operator_map,
            prompt_dropout_prob=args.prompt_dropout_prob,
            visual_dropout_given_prompt_prob=args.visual_dropout_given_prompt_prob,
            neg_prompt_given_drop_prob=args.neg_prompt_given_drop_prob,
            neg_prompt_pool=neg_prompt_pool_list,
            max_ref_items=args.ref_max_items,
            metadata_shuffle_seed=shuffle_seed_map.get("img"),
        )
    else:
        dataset = None
    if args.vid_dataset_metadata_path:
        vid_dataset = UnifiedDataset(
            base_path=args.dataset_base_path,
            metadata_path=args.vid_dataset_metadata_path,
            repeat=args.dataset_repeat,
            data_file_keys=args.data_file_keys.split(","),
            main_data_operator=UnifiedDataset.default_video_operator(
                base_path=args.dataset_base_path,
                max_pixels=args.max_pixels,
                height=args.height,
                width=args.width,
                height_division_factor=32,
                width_division_factor=32,
                lmdb_roots=lmdb_roots,
                num_frames=args.num_frames,
                time_division_factor=4,
                time_division_remainder=1,
            ),
            special_operator_map=special_operator_map,
            prompt_dropout_prob=args.prompt_dropout_prob,
            visual_dropout_given_prompt_prob=args.visual_dropout_given_prompt_prob,
            neg_prompt_given_drop_prob=args.neg_prompt_given_drop_prob,
            neg_prompt_pool=neg_prompt_pool_list,
            max_ref_items=args.ref_max_items,
            metadata_exclude_rules=args.vid_metadata_exclude_rules,
            metadata_shuffle_seed=shuffle_seed_map.get("vid"),
        )
    else:
        vid_dataset = None
    if args.vid_ref_dataset_metadata_path:
        vid_ref_dataset = UnifiedDataset(
            base_path=args.dataset_base_path,
            metadata_path=args.vid_ref_dataset_metadata_path,
            repeat=args.dataset_repeat,
            data_file_keys=args.data_file_keys.split(","),
            main_data_operator=UnifiedDataset.default_video_operator(
                base_path=args.dataset_base_path,
                max_pixels=args.max_pixels,
                height=args.height,
                width=args.width,
                height_division_factor=32,
                width_division_factor=32,
                lmdb_roots=lmdb_roots,
                num_frames=args.num_frames,
                time_division_factor=4,
                time_division_remainder=1,
            ),
            special_operator_map=special_operator_map,
            prompt_dropout_prob=args.prompt_dropout_prob,
            visual_dropout_given_prompt_prob=args.visual_dropout_given_prompt_prob,
            neg_prompt_given_drop_prob=args.neg_prompt_given_drop_prob,
            neg_prompt_pool=neg_prompt_pool_list,
            max_ref_items=args.ref_max_items,
            metadata_shuffle_seed=shuffle_seed_map.get("vid_ref"),
        )
    else:
        vid_ref_dataset = None
    
    if not (dataset or vid_dataset or vid_ref_dataset):
        raise ValueError("dataset, vid_dataset, or vid_ref_dataset is required.")

    _apply_dataset_repeat_policy(
        args,
        run_training_meta,
        {
            "img": dataset,
            "vid": vid_dataset,
            "vid_ref": vid_ref_dataset,
        },
    )

    model = WanTrainingModule(
        model_paths=args.model_paths,
        model_id_with_origin_paths=args.model_id_with_origin_paths,
        audio_processor_config=args.audio_processor_config,
        trainable_models=args.trainable_models,
        use_gradient_checkpointing_offload=args.use_gradient_checkpointing_offload,
        extra_inputs=args.extra_inputs,
        max_timestep_boundary=args.max_timestep_boundary,
        min_timestep_boundary=args.min_timestep_boundary,
        checkpoint=args.checkpoint,
        mllm_model=args.mllm_model,
        mllm_max_pixels_per_frame=args.mllm_max_pixels_per_frame,
        mllm_ref_max_pixels=args.mllm_ref_max_pixels,
        mllm_video_sample_fps=args.mllm_video_sample_fps,
        mllm_video_min_frames=args.mllm_video_min_frames,
        mllm_gradient_checkpointing=args.mllm_gradient_checkpointing,
        ref_pad_first=args.ref_pad_first,
        ref_max_items=args.ref_max_items,
        ref_image_max_pixels=args.ref_image_max_pixels,
        rope_mode=args.rope_mode,
        source_condition_mode=args.source_condition_mode,
        source_zero_cond_t=args.source_zero_cond_t,
        debug_dump_every=args.debug_dump_every,
        debug_dump_limit=args.debug_dump_limit,
        debug_dump_num_ranks=args.debug_dump_num_ranks,
        debug_dump_max_frames=args.debug_dump_max_frames,
        debug_dump_dir=args.debug_dump_dir,
    )
    model_logger = ModelLogger(
        args.output_path,
        remove_prefix_in_ckpt=args.remove_prefix_in_ckpt,
        permanent_save_steps=args.permanent_save_steps,
        training_meta=run_training_meta,
    )
    model.set_debug_dump_root(args.output_path)
    launch_mix_training_task(dataset, vid_dataset, vid_ref_dataset, model, model_logger, args=args)
