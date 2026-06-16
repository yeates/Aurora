import imageio, os, shutil, torch, warnings, torchvision, argparse, json
from diffsynth.utils import ModelConfig
from diffsynth.models.state_dict_utils import load_state_dict
from peft import LoraConfig, inject_adapter_in_model
from PIL import Image
import pandas as pd
from tqdm import tqdm
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs
from datetime import timedelta
from accelerate import Accelerator, InitProcessGroupKwargs
from accelerate.utils import DeepSpeedPlugin

class ImageDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        base_path=None, metadata_path=None,
        max_pixels=1920*1080, height=None, width=None,
        height_division_factor=16, width_division_factor=16,
        data_file_keys=("image",),
        image_file_extension=("jpg", "jpeg", "png", "webp"),
        repeat=1,
        args=None,
    ):
        if args is not None:
            base_path = args.dataset_base_path
            metadata_path = args.dataset_metadata_path
            height = args.height
            width = args.width
            max_pixels = args.max_pixels
            data_file_keys = args.data_file_keys.split(",")
            repeat = args.dataset_repeat
            
        self.base_path = base_path
        self.max_pixels = max_pixels
        self.height = height
        self.width = width
        self.height_division_factor = height_division_factor
        self.width_division_factor = width_division_factor
        self.data_file_keys = data_file_keys
        self.image_file_extension = image_file_extension
        self.repeat = repeat

        if height is not None and width is not None:
            print("Height and width are fixed. Setting `dynamic_resolution` to False.")
            self.dynamic_resolution = False
        elif height is None and width is None:
            print("Height and width are none. Setting `dynamic_resolution` to True.")
            self.dynamic_resolution = True
            
        if metadata_path is None:
            print("No metadata. Trying to generate it.")
            metadata = self.generate_metadata(base_path)
            print(f"{len(metadata)} lines in metadata.")
            self.data = [metadata.iloc[i].to_dict() for i in range(len(metadata))]
        elif metadata_path.endswith(".json"):
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
            self.data = metadata
        elif metadata_path.endswith(".jsonl"):
            metadata = []
            with open(metadata_path, 'r') as f:
                for line in tqdm(f):
                    metadata.append(json.loads(line.strip()))
            self.data = metadata
        else:
            metadata = pd.read_csv(metadata_path)
            self.data = [metadata.iloc[i].to_dict() for i in range(len(metadata))]


    def generate_metadata(self, folder):
        image_list, prompt_list = [], []
        file_set = set(os.listdir(folder))
        for file_name in file_set:
            if "." not in file_name:
                continue
            file_ext_name = file_name.split(".")[-1].lower()
            file_base_name = file_name[:-len(file_ext_name)-1]
            if file_ext_name not in self.image_file_extension:
                continue
            prompt_file_name = file_base_name + ".txt"
            if prompt_file_name not in file_set:
                continue
            with open(os.path.join(folder, prompt_file_name), "r", encoding="utf-8") as f:
                prompt = f.read().strip()
            image_list.append(file_name)
            prompt_list.append(prompt)
        metadata = pd.DataFrame()
        metadata["image"] = image_list
        metadata["prompt"] = prompt_list
        return metadata
    
    
    def crop_and_resize(self, image, target_height, target_width):
        width, height = image.size
        scale = max(target_width / width, target_height / height)
        image = torchvision.transforms.functional.resize(
            image,
            (round(height*scale), round(width*scale)),
            interpolation=torchvision.transforms.InterpolationMode.BILINEAR
        )
        image = torchvision.transforms.functional.center_crop(image, (target_height, target_width))
        return image
    
    
    def get_height_width(self, image):
        if self.dynamic_resolution:
            width, height = image.size
            if width * height > self.max_pixels:
                scale = (width * height / self.max_pixels) ** 0.5
                height, width = int(height / scale), int(width / scale)
            height = height // self.height_division_factor * self.height_division_factor
            width = width // self.width_division_factor * self.width_division_factor
        else:
            height, width = self.height, self.width
        return height, width
    
    
    def load_image(self, file_path):
        image = Image.open(file_path).convert("RGB")
        image = self.crop_and_resize(image, *self.get_height_width(image))
        return image
    
    
    def load_data(self, file_path):
        return self.load_image(file_path)


    def __getitem__(self, data_id):
        data = self.data[data_id % len(self.data)].copy()
        for key in self.data_file_keys:
            if key in data:
                if isinstance(data[key], list):
                    path = [os.path.join(self.base_path, p) for p in data[key]]
                    data[key] = [self.load_data(p) for p in path]
                else:
                    path = os.path.join(self.base_path, data[key])
                    data[key] = self.load_data(path)
                if data[key] is None:
                    warnings.warn(f"cannot load file {data[key]}.")
                    return None
        return data
    

    def __len__(self):
        return len(self.data) * self.repeat


class VideoDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        base_path=None, metadata_path=None,
        num_frames=81,
        time_division_factor=4, time_division_remainder=1,
        max_pixels=1920*1080, height=None, width=None,
        height_division_factor=16, width_division_factor=16,
        data_file_keys=("video",),
        image_file_extension=("jpg", "jpeg", "png", "webp"),
        video_file_extension=("mp4", "avi", "mov", "wmv", "mkv", "flv", "webm", "gif"),
        repeat=1,
        args=None,
    ):
        if args is not None:
            base_path = args.dataset_base_path
            metadata_path = args.dataset_metadata_path
            height = args.height
            width = args.width
            max_pixels = args.max_pixels
            num_frames = args.num_frames
            data_file_keys = args.data_file_keys.split(",")
            repeat = args.dataset_repeat
        
        self.base_path = base_path
        self.num_frames = num_frames
        self.time_division_factor = time_division_factor
        self.time_division_remainder = time_division_remainder
        self.max_pixels = max_pixels
        self.height = height
        self.width = width
        self.height_division_factor = height_division_factor
        self.width_division_factor = width_division_factor
        self.data_file_keys = data_file_keys
        self.image_file_extension = image_file_extension
        self.video_file_extension = video_file_extension
        self.repeat = repeat
        
        if height is not None and width is not None:
            print("Height and width are fixed. Setting `dynamic_resolution` to False.")
            self.dynamic_resolution = False
        elif height is None and width is None:
            print("Height and width are none. Setting `dynamic_resolution` to True.")
            self.dynamic_resolution = True
            
        if metadata_path is None:
            print("No metadata. Trying to generate it.")
            metadata = self.generate_metadata(base_path)
            print(f"{len(metadata)} lines in metadata.")
            self.data = [metadata.iloc[i].to_dict() for i in range(len(metadata))]
        elif metadata_path.endswith(".json"):
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
            self.data = metadata
        else:
            metadata = pd.read_csv(metadata_path)
            self.data = [metadata.iloc[i].to_dict() for i in range(len(metadata))]
            
    
    def generate_metadata(self, folder):
        video_list, prompt_list = [], []
        file_set = set(os.listdir(folder))
        for file_name in file_set:
            if "." not in file_name:
                continue
            file_ext_name = file_name.split(".")[-1].lower()
            file_base_name = file_name[:-len(file_ext_name)-1]
            if file_ext_name not in self.image_file_extension and file_ext_name not in self.video_file_extension:
                continue
            prompt_file_name = file_base_name + ".txt"
            if prompt_file_name not in file_set:
                continue
            with open(os.path.join(folder, prompt_file_name), "r", encoding="utf-8") as f:
                prompt = f.read().strip()
            video_list.append(file_name)
            prompt_list.append(prompt)
        metadata = pd.DataFrame()
        metadata["video"] = video_list
        metadata["prompt"] = prompt_list
        return metadata
        
        
    def crop_and_resize(self, image, target_height, target_width):
        width, height = image.size
        scale = max(target_width / width, target_height / height)
        image = torchvision.transforms.functional.resize(
            image,
            (round(height*scale), round(width*scale)),
            interpolation=torchvision.transforms.InterpolationMode.BILINEAR
        )
        image = torchvision.transforms.functional.center_crop(image, (target_height, target_width))
        return image
    
    
    def get_height_width(self, image):
        if self.dynamic_resolution:
            width, height = image.size
            if width * height > self.max_pixels:
                scale = (width * height / self.max_pixels) ** 0.5
                height, width = int(height / scale), int(width / scale)
            height = height // self.height_division_factor * self.height_division_factor
            width = width // self.width_division_factor * self.width_division_factor
        else:
            height, width = self.height, self.width
        return height, width
    
    
    def get_num_frames(self, reader):
        num_frames = self.num_frames
        if int(reader.count_frames()) < num_frames:
            num_frames = int(reader.count_frames())
            while num_frames > 1 and num_frames % self.time_division_factor != self.time_division_remainder:
                num_frames -= 1
        return num_frames
    
    def _load_gif(self, file_path):
        gif_img = Image.open(file_path)
        frame_count = 0
        delays, frames = [], []
        while True:
            delay = gif_img.info.get('duration', 100)
            delays.append(delay)
            rgb_frame = gif_img.convert("RGB")   
            croped_frame = self.crop_and_resize(rgb_frame, *self.get_height_width(rgb_frame))
            frames.append(croped_frame)             
            frame_count += 1
            try:
                gif_img.seek(frame_count)
            except:
                break
        if any((delays[0] != i) for i in delays):
            minimal_interval = min([i for i in delays if i > 0])
            start_end_idx_map = [((sum(delays[:i]), sum(delays[:i+1])), i) for i in range(len(delays))]
            _frames = []
            last_match = 0
            for i in range(sum(delays) // minimal_interval):
                current_time = minimal_interval * i
                for idx, ((start, end), frame_idx) in enumerate(start_end_idx_map[last_match:]):
                    if start <= current_time < end:
                        _frames.append(frames[frame_idx])
                        last_match = idx + last_match
                        break
            frames = _frames
        num_frames = len(frames)
        if num_frames > self.num_frames:
            num_frames = self.num_frames
        else:
            while num_frames > 1 and num_frames % self.time_division_factor != self.time_division_remainder:
                num_frames -= 1
        frames = frames[:num_frames]
        return frames
    
    def load_video(self, file_path):
        if file_path.lower().endswith(".gif"):
            return self._load_gif(file_path)
        reader = imageio.get_reader(file_path)
        num_frames = self.get_num_frames(reader)
        frames = []
        for frame_id in range(num_frames):
            frame = reader.get_data(frame_id)
            frame = Image.fromarray(frame)
            frame = self.crop_and_resize(frame, *self.get_height_width(frame))
            frames.append(frame)
        reader.close()
        return frames
    
    
    def load_image(self, file_path):
        image = Image.open(file_path).convert("RGB")
        image = self.crop_and_resize(image, *self.get_height_width(image))
        frames = [image]
        return frames
    
    
    def is_image(self, file_path):
        file_ext_name = file_path.split(".")[-1]
        return file_ext_name.lower() in self.image_file_extension
    
    
    def is_video(self, file_path):
        file_ext_name = file_path.split(".")[-1]
        return file_ext_name.lower() in self.video_file_extension
    
    
    def load_data(self, file_path):
        if self.is_image(file_path):
            return self.load_image(file_path)
        elif self.is_video(file_path):
            return self.load_video(file_path)
        else:
            return None


    def __getitem__(self, data_id):
        data = self.data[data_id % len(self.data)].copy()
        for key in self.data_file_keys:
            if key in data:
                path = os.path.join(self.base_path, data[key])
                data[key] = self.load_data(path)
                if data[key] is None:
                    warnings.warn(f"cannot load file {data[key]}.")
                    return None
        return data
    

    def __len__(self):
        return len(self.data) * self.repeat


def rgetattr(obj, attr):
    for part in attr.split("."):
        obj = getattr(obj, part)
    return obj

def rsetattr(obj, attr, value):
    parts = attr.split(".")
    for part in parts[:-1]:
        obj = getattr(obj, part)
    setattr(obj, parts[-1], value)

class DiffusionTrainingModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        
        
    def to(self, *args, **kwargs):
        for name, model in self.named_children():
            model.to(*args, **kwargs)
        return self
        
        
    def trainable_modules(self):
        trainable_modules = filter(lambda p: p.requires_grad, self.parameters())
        return trainable_modules
    
    
    def trainable_param_names(self):
        trainable_param_names = list(filter(lambda named_param: named_param[1].requires_grad, self.named_parameters()))
        trainable_param_names = set([named_param[0] for named_param in trainable_param_names])
        return trainable_param_names
    
    
    def add_lora_to_model(self, model, target_modules, lora_rank, lora_alpha=None, upcast_dtype=None):
        if lora_alpha is None:
            lora_alpha = lora_rank
        lora_config = LoraConfig(r=lora_rank, lora_alpha=lora_alpha, target_modules=target_modules)
        model = inject_adapter_in_model(lora_config, model)
        if upcast_dtype is not None:
            for param in model.parameters():
                if param.requires_grad:
                    param.data = param.to(upcast_dtype)
        return model


    def mapping_lora_state_dict(self, state_dict):
        new_state_dict = {}
        for key, value in state_dict.items():
            if "lora_A.weight" in key or "lora_B.weight" in key:
                new_key = key.replace("lora_A.weight", "lora_A.default.weight").replace("lora_B.weight", "lora_B.default.weight")
                new_state_dict[new_key] = value
            elif "lora_A.default.weight" in key or "lora_B.default.weight" in key:
                new_state_dict[key] = value
        return new_state_dict


    def export_trainable_state_dict(self, state_dict, remove_prefix=None):
        trainable_param_names = self.trainable_param_names()
        state_dict = {name: param for name, param in state_dict.items() if name in trainable_param_names}
        if remove_prefix is not None:
            state_dict_ = {}
            for name, param in state_dict.items():
                if name.startswith(remove_prefix):
                    name = name[len(remove_prefix):]
                state_dict_[name] = param
            state_dict = state_dict_
        return state_dict
    
    
    def transfer_data_to_device(self, data, device, torch_float_dtype=None):
        for key in data:
            if isinstance(data[key], torch.Tensor):
                data[key] = data[key].to(device)
                if torch_float_dtype is not None and data[key].dtype in [torch.float, torch.float16, torch.bfloat16]:
                    data[key] = data[key].to(torch_float_dtype)
        return data
    
    
    def parse_model_configs(self, model_paths, model_id_with_origin_paths, enable_fp8_training=False):
        offload_dtype = torch.float8_e4m3fn if enable_fp8_training else None
        model_configs = []
        if model_paths is not None:
            model_paths = json.loads(model_paths)
            from collections import defaultdict
            groups = defaultdict(list)
            standalone = []
            for path in model_paths:
                base = os.path.basename(path)
                if "-of-" in base and base.endswith(".safetensors"):
                    prefix = base.split("-0")[0]
                    dir_key = (os.path.dirname(path), prefix)
                    groups[dir_key].append(path)
                else:
                    standalone.append(path)
            for paths in groups.values():
                model_configs.append(ModelConfig(path=sorted(paths), offload_dtype=offload_dtype))
            for path in standalone:
                model_configs.append(ModelConfig(path=path, offload_dtype=offload_dtype))
        if model_id_with_origin_paths is not None:
            model_id_with_origin_paths = model_id_with_origin_paths.split(",")
            model_configs += [ModelConfig(model_id=i.split(":")[0], origin_file_pattern=i.split(":")[1], offload_dtype=offload_dtype) for i in model_id_with_origin_paths]
        return model_configs
    
    def mapping_mix_lora_state_dict(self, state_dict):
        new_state_dict = {}
        for key, value in state_dict.items():
            if "lora_A.weight" in key or "lora_B.weight" in key:
                new_key = key.replace("lora_A.weight", "lora_A.default.weight").replace("lora_B.weight", "lora_B.default.weight")
                new_state_dict[new_key] = value
            elif "lora_A.default.weight" in key or "lora_B.default.weight" in key:
                new_state_dict[key] = value
            else:
                new_state_dict[key] = value
        return new_state_dict

    @staticmethod
    def merge_lora_weights(model):
        """Merge LoRA weights into base model and remove LoRA layers."""
        from peft.tuners.lora import Linear as LoRALinear
        merged_count = 0
        for name, module in list(model.named_modules()):
            if isinstance(module, LoRALinear):
                module.merge()
                merged_count += 1
        print(f"Merged {merged_count} LoRA layers into base weights")
        return model

    def switch_pipe_to_training_mode(
        self,
        pipe,
        trainable_models,
        checkpoint: str = None,
    ):
        pipe.scheduler.set_timesteps(1000, training=True)

        if checkpoint is not None:
            state_dict = load_state_dict(checkpoint, torch_dtype=torch.bfloat16, device=pipe.mllm.model.device)
            res = pipe.load_state_dict(state_dict, strict=False)
            print(state_dict.keys())
            print(f"Checkpoint loaded: {checkpoint}, total {len(state_dict)} keys, {res}")
            del state_dict

        if hasattr(pipe, "normalize_trainable_models"):
            trainable_models = pipe.normalize_trainable_models(trainable_models)

        pipe.freeze_except([] if trainable_models is None else trainable_models.split(","))


class ModelLogger:
    def __init__(
        self,
        output_path,
        remove_prefix_in_ckpt=None,
        state_dict_converter=lambda x:x,
        permanent_save_steps=2500,
        training_meta=None,
    ):
        self.output_path = output_path
        self.remove_prefix_in_ckpt = remove_prefix_in_ckpt
        self.state_dict_converter = state_dict_converter
        self.num_steps = 0
        self.num_consumed_batches = 0
        self.permanent_save_steps = permanent_save_steps
        self.training_meta = {} if training_meta is None else dict(training_meta)


    def on_step_end(self, accelerator, model, save_steps=None):
        self.num_steps += 1
        is_latest_step = (save_steps is not None and self.num_steps % save_steps == 0)
        is_permanent = (self.permanent_save_steps and self.num_steps % self.permanent_save_steps == 0)
        if not (is_latest_step or is_permanent):
            return
        file_names = ["latest.safetensors"]
        if is_permanent:
            file_names.append(f"step-{self.num_steps}.safetensors")
        self.save_model(accelerator, model, file_names)
        self.save_training_state(accelerator, self.num_steps)


    def on_epoch_end(self, accelerator, model, epoch_id):
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            state_dict = accelerator.get_state_dict(model)
            state_dict = accelerator.unwrap_model(model).export_trainable_state_dict(state_dict, remove_prefix=self.remove_prefix_in_ckpt)
            state_dict = self.state_dict_converter(state_dict)
            os.makedirs(self.output_path, exist_ok=True)
            path = os.path.join(self.output_path, f"epoch-{epoch_id}.safetensors")
            accelerator.save(state_dict, path, safe_serialization=True)


    def on_training_end(self, accelerator, model, save_steps=None):
        if save_steps is not None and self.num_steps % save_steps != 0:
            self.save_model(accelerator, model, ["latest.safetensors"])
            self.save_training_state(accelerator, self.num_steps)
        torch.distributed.destroy_process_group()


    def save_model(self, accelerator, model, file_names):
        if isinstance(file_names, str):
            file_names = [file_names]
        accelerator.wait_for_everyone()
        full_state_dict = accelerator.get_state_dict(model)
        if accelerator.is_main_process:
            state_dict = accelerator.unwrap_model(model).export_trainable_state_dict(
                full_state_dict,
                remove_prefix=self.remove_prefix_in_ckpt
            )
            state_dict = self.state_dict_converter(state_dict)
            os.makedirs(self.output_path, exist_ok=True)
            for fn in file_names:
                path = os.path.join(self.output_path, fn)
                accelerator.save(state_dict, path, safe_serialization=True)
        accelerator.wait_for_everyone()

    def save_training_state(self, accelerator, step):
        """Save training state to training_state_latest/, overwriting previous."""
        state_dir = os.path.join(self.output_path, "training_state_latest")
        if accelerator.is_main_process:
            if os.path.exists(state_dir):
                shutil.rmtree(state_dir)
        accelerator.wait_for_everyone()
        accelerator.save_state(state_dir)
        if accelerator.is_main_process:
            meta = {
                **self.training_meta,
                "num_steps": step,
                "num_consumed_batches": int(self.num_consumed_batches),
            }
            with open(os.path.join(state_dir, "training_meta.json"), "w") as f:
                json.dump(meta, f)
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            print(f"Training state saved to {state_dir} (step {step})")


class DistributedRangedSampler(torch.utils.data.Sampler):
    """Sampler that assigns a contiguous range to each rank. O(1) resume via set_start."""

    def __init__(self, dataset, num_replicas=1, rank=0):
        self.dataset = dataset
        self.num_samples = len(dataset)
        self.num_replicas = num_replicas
        self.rank = rank
        chunk = self.num_samples // num_replicas
        remainder = self.num_samples % num_replicas
        self.worker_start = rank * chunk + min(rank, remainder)
        self.worker_end = self.worker_start + chunk + (1 if rank < remainder else 0)
        self.step_start = 0

    def set_start(self, start):
        worker_size = self.worker_end - self.worker_start
        self.step_start = max(0, min(start, worker_size))

    def __iter__(self):
        start = min(self.worker_start + self.step_start, self.worker_end)
        yield from range(start, self.worker_end)

    def __len__(self):
        return max(0, self.worker_end - self.worker_start - self.step_start)


class MixDataloader:
    _TYPE_BY_SAMPLER_KEY = {"img": 0, "vid": 1, "vid_ref": 2}
    _TYPE_NAME_BY_ID = {value: key for key, value in _TYPE_BY_SAMPLER_KEY.items()}
    _VALID_MIX_MODES = {"sequential", "rank_staggered"}

    @classmethod
    def _default_pattern(cls, loader_by_type):
        ordered_types = [sample_type for sample_type in (0, 1, 2) if loader_by_type.get(sample_type) is not None]
        if not ordered_types:
            raise ValueError("No dataloader is provided")
        return ordered_types

    @classmethod
    def _parse_pattern(cls, mix_pattern):
        if mix_pattern is None:
            return None
        if isinstance(mix_pattern, str):
            tokens = [token.strip() for token in mix_pattern.split(",") if token.strip()]
        else:
            tokens = [str(token).strip() for token in mix_pattern if str(token).strip()]
        if not tokens:
            raise ValueError("dataset_mix_pattern cannot be empty.")
        invalid = [token for token in tokens if token not in cls._TYPE_BY_SAMPLER_KEY]
        if invalid:
            raise ValueError(
                "dataset_mix_pattern only supports "
                f"{sorted(cls._TYPE_BY_SAMPLER_KEY)}, got invalid entries {invalid}."
            )
        return [cls._TYPE_BY_SAMPLER_KEY[token] for token in tokens]

    @classmethod
    def canonicalize_pattern(cls, mix_pattern, active_type_names):
        active_set = set(active_type_names)
        pattern = cls._default_pattern(
            {cls._TYPE_BY_SAMPLER_KEY[name]: object() for name in active_type_names}
        ) if mix_pattern is None else cls._parse_pattern(mix_pattern)
        missing = [cls._TYPE_NAME_BY_ID[sample_type] for sample_type in pattern if cls._TYPE_NAME_BY_ID[sample_type] not in active_set]
        if missing:
            raise ValueError(
                f"dataset_mix_pattern references unavailable dataset types: {missing}. "
                f"Available types: {sorted(active_type_names)}"
            )
        return ",".join(cls._TYPE_NAME_BY_ID[sample_type] for sample_type in pattern)

    @classmethod
    def canonicalize_mix_mode(cls, mix_mode):
        if mix_mode is None:
            return "sequential"
        mix_mode = str(mix_mode).strip().lower().replace("-", "_")
        if mix_mode in {"", "none", "legacy"}:
            mix_mode = "sequential"
        elif mix_mode == "staggered":
            mix_mode = "rank_staggered"
        if mix_mode not in cls._VALID_MIX_MODES:
            raise ValueError(
                f"dataset_mix_mode must be one of {sorted(cls._VALID_MIX_MODES)}, got {mix_mode!r}"
            )
        return mix_mode

    def __init__(
        self,
        dataloader,
        vid_dataloader,
        vid_ref_dataloader,
        num_epochs=1,
        samplers=None,
        mix_pattern=None,
        mix_mode="sequential",
        rank=0,
    ):
        self.dataloader = dataloader
        self.vid_dataloader = vid_dataloader
        self.vid_ref_dataloader = vid_ref_dataloader
        self.num_epochs = num_epochs
        self.samplers = samplers or {}
        self.mix_mode = self.canonicalize_mix_mode(mix_mode)
        self.rank = int(rank)
        self.loader_by_type = {
            0: self.dataloader,
            1: self.vid_dataloader,
            2: self.vid_ref_dataloader,
        }
        if dataloader:
            self.iter = iter(self.dataloader)
            print("Image dataloader", len(self.dataloader))
        else:
            print("No image dataloader is provided")
        if vid_dataloader:
            self.vid_iter = iter(self.vid_dataloader)
            print("Instuct Vid dataloader", len(self.vid_dataloader))
        else:
            print("No instruct vid dataloader is provided")
        if vid_ref_dataloader:
            print("Instuct Ref Vid dataloader", len(self.vid_ref_dataloader))
            self.vid_ref_iter = iter(self.vid_ref_dataloader)
        else:
            print("No instruct ref vid dataloader is provided")
        self.pattern = self._default_pattern(self.loader_by_type) if mix_pattern is None else self._parse_pattern(mix_pattern)
        unavailable = [self._TYPE_NAME_BY_ID[sample_type] for sample_type in self.pattern if self.loader_by_type.get(sample_type) is None]
        if unavailable:
            raise ValueError(
                f"dataset_mix_pattern references dataset types without dataloaders: {unavailable}"
            )
        self._pattern_counts = {sample_type: self.pattern.count(sample_type) for sample_type in set(self.pattern)}
        cycles = min(
            self._stable_loader_length(sample_type) // pattern_count
            for sample_type, pattern_count in self._pattern_counts.items()
        )
        self.length = cycles * len(self.pattern)
        if self.length <= 0:
            raise ValueError(
                "dataset_mix_pattern requires more samples per cycle than at least one dataloader can provide. "
                f"pattern={self.pattern}, per_gpu_lengths="
                f"{ {self._TYPE_NAME_BY_ID[sample_type]: self._stable_loader_length(sample_type) for sample_type, loader in self.loader_by_type.items() if loader is not None} }"
            )
        self.pattern_offset = self.rank % len(self.pattern) if self.mix_mode == "rank_staggered" else 0
        print(
            f"Mix pattern: {[self._TYPE_NAME_BY_ID[sample_type] for sample_type in self.pattern]} "
            f"mode={self.mix_mode} rank={self.rank} offset={self.pattern_offset}"
        )

    def _stable_loader_length(self, sample_type):
        sampler = self.samplers.get(self._TYPE_NAME_BY_ID[sample_type])
        if sampler is not None and getattr(sampler, "num_replicas", 0):
            return len(sampler.dataset) // int(sampler.num_replicas)
        loader = self.loader_by_type[sample_type]
        return len(loader)

    def _per_type_skip_counts(self, n):
        if not self.pattern:
            return {}
        full_cycles, remainder = divmod(n, len(self.pattern))
        counts = {sample_type: full_cycles * pattern_count for sample_type, pattern_count in self._pattern_counts.items()}
        for i in range(remainder):
            sample_type = self._sample_type_at(i)
            counts[sample_type] = counts.get(sample_type, 0) + 1
        return counts

    def _sample_type_at(self, sample_idx):
        return self.pattern[(sample_idx + self.pattern_offset) % len(self.pattern)]

    @staticmethod
    def _tag_data(data, sample_type):
        if isinstance(data, dict):
            data["_loader_type"] = MixDataloader._TYPE_NAME_BY_ID[sample_type]
        return data

    def _next_data(self, sample_type):
        if sample_type == 0:
            try:
                return self._tag_data(next(self.iter), sample_type)
            except StopIteration:
                self.iter = iter(self.dataloader)
                return self._tag_data(next(self.iter), sample_type)
        if sample_type == 1:
            try:
                return self._tag_data(next(self.vid_iter), sample_type)
            except StopIteration:
                self.vid_iter = iter(self.vid_dataloader)
                return self._tag_data(next(self.vid_iter), sample_type)
        try:
            return self._tag_data(next(self.vid_ref_iter), sample_type)
        except StopIteration:
            self.vid_ref_iter = iter(self.vid_ref_dataloader)
            return self._tag_data(next(self.vid_ref_iter), sample_type)

    def skip_first_n(self, n):
        """O(1) skip: advance each sampler by the exact number of already-consumed samples of its type."""
        total_steps = self.length * self.num_epochs
        n = max(0, min(n, total_steps))
        per_type_skip = self._per_type_skip_counts(n)
        for sampler_key, sampler in self.samplers.items():
            sample_type = self._TYPE_BY_SAMPLER_KEY.get(sampler_key)
            sampler.set_start(per_type_skip.get(sample_type, 0))
        if self.dataloader:
            self.iter = iter(self.dataloader)
        if self.vid_dataloader:
            self.vid_iter = iter(self.vid_dataloader)
        if self.vid_ref_dataloader:
            self.vid_ref_iter = iter(self.vid_ref_dataloader)
        self._skip_n = n

    def __iter__(self):
        skip_n = getattr(self, "_skip_n", 0)
        total_steps = self.length * self.num_epochs
        for sample_idx in range(skip_n, total_steps):
            yield self._next_data(self._sample_type_at(sample_idx))

    def __len__(self):
        skip_n = getattr(self, "_skip_n", 0)
        return max(0, self.length * self.num_epochs - skip_n)


def _scheduler_impl(scheduler):
    return getattr(scheduler, "scheduler", scheduler)


def _lr_to_float(lr):
    return lr.item() if isinstance(lr, torch.Tensor) else float(lr)


def _get_optimizer_lrs(optimizer):
    return [_lr_to_float(group["lr"]) for group in optimizer.param_groups]


def _set_optimizer_lrs(optimizer, lrs):
    if len(lrs) != len(optimizer.param_groups):
        raise ValueError(f"Expected {len(optimizer.param_groups)} lr values, got {len(lrs)}")
    for group, lr in zip(optimizer.param_groups, lrs):
        if isinstance(group["lr"], torch.Tensor):
            group["lr"].fill_(lr)
        else:
            group["lr"] = lr
        if "initial_lr" in group:
            group["initial_lr"] = lr


def _distributed_loader_type_stats(loader_type, loss, device):
    stats = torch.zeros((len(MixDataloader._TYPE_NAME_BY_ID), 2), device=device, dtype=torch.float32)
    sample_type = MixDataloader._TYPE_BY_SAMPLER_KEY.get(loader_type)
    if sample_type is not None:
        stats[sample_type, 0] = 1.0
        stats[sample_type, 1] = loss.detach().float()
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.all_reduce(stats, op=torch.distributed.ReduceOp.SUM)

    counts = {}
    losses = {}
    total_count = 0.0
    total_loss_sum = 0.0
    for sample_type, name in MixDataloader._TYPE_NAME_BY_ID.items():
        count = float(stats[sample_type, 0].item())
        loss_sum = float(stats[sample_type, 1].item())
        counts[name] = count
        total_count += count
        total_loss_sum += loss_sum
        if count > 0:
            losses[name] = loss_sum / count
    losses["global"] = total_loss_sum / total_count if total_count > 0 else float(loss.detach().float().item())
    return counts, losses


def _align_constant_lr_state(optimizer, scheduler, fallback_lr=None):
    """
    Keep optimizer and scheduler aligned to a true constant LR.

    Older training states used ConstantLR's default factor=1/3 warmup. When
    those states are resumed, normalize the scheduler back into a no-op
    constant schedule before re-applying the configured LR.
    """
    raw_scheduler = _scheduler_impl(scheduler)
    if isinstance(raw_scheduler, torch.optim.lr_scheduler.ConstantLR):
        raw_scheduler.factor = 1.0
        raw_scheduler.total_iters = 0
    target_lrs = getattr(raw_scheduler, "base_lrs", None)
    if target_lrs:
        target_lrs = [_lr_to_float(lr) for lr in target_lrs]
    elif fallback_lr is not None:
        target_lrs = [float(fallback_lr)] * len(optimizer.param_groups)
    else:
        target_lrs = _get_optimizer_lrs(optimizer)

    _set_optimizer_lrs(optimizer, target_lrs)
    if hasattr(raw_scheduler, "base_lrs"):
        raw_scheduler.base_lrs = list(target_lrs)
    if hasattr(raw_scheduler, "_last_lr"):
        raw_scheduler._last_lr = list(target_lrs)
    return list(target_lrs)

def launch_mix_training_task(
    dataset: torch.utils.data.Dataset,
    vid_dataset: torch.utils.data.Dataset,
    vid_ref_dataset: torch.utils.data.Dataset,
    model: DiffusionTrainingModule,
    model_logger: ModelLogger,
    learning_rate: float = 1e-5,
    weight_decay: float = 1e-2,
    num_workers: int = 8,
    save_steps: int = None,
    num_epochs: int = 1,
    gradient_accumulation_steps: int = 1,
    find_unused_parameters: bool = False,
    args = None,
):
    if args is not None:
        learning_rate = args.learning_rate
        weight_decay = args.weight_decay
        num_workers = args.dataset_num_workers
        save_steps = args.save_steps
        num_epochs = args.num_epochs
        gradient_accumulation_steps = args.gradient_accumulation_steps
        find_unused_parameters = args.find_unused_parameters
    optimizer = torch.optim.AdamW(model.trainable_modules(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1.0, total_iters=0)
    _align_constant_lr_state(optimizer, scheduler, learning_rate)

    ds_config_path = getattr(args, "deepspeed_config", None) if args else None
    if ds_config_path:
        ds_plugin = DeepSpeedPlugin(hf_ds_config=ds_config_path)
    else:
        ds_plugin = DeepSpeedPlugin()
    accelerator = Accelerator(
        gradient_accumulation_steps=gradient_accumulation_steps,
        deepspeed_plugin=ds_plugin,
        kwargs_handlers=[
            DistributedDataParallelKwargs(find_unused_parameters=find_unused_parameters),
            InitProcessGroupKwargs(timeout=timedelta(seconds=5400))],
    )
    if accelerator.state.deepspeed_plugin is not None:
        ds_cfg = accelerator.state.deepspeed_plugin.deepspeed_config
        if "train_micro_batch_size_per_gpu" not in ds_cfg or ds_cfg["train_micro_batch_size_per_gpu"] == "auto":
            ds_cfg["train_micro_batch_size_per_gpu"] = 1

    world_size = accelerator.num_processes
    rank = accelerator.process_index
    if accelerator.is_main_process:
        print(f"[dist] world_size={world_size}, rank={rank}, device={accelerator.device}")

    samplers = {}

    def _make_dataloader(ds, name, sampler_key):
        if ds is None:
            print(f"{name} dataset is None, skip training.")
            return None
        sampler = DistributedRangedSampler(ds, num_replicas=world_size, rank=rank)
        samplers[sampler_key] = sampler
        return torch.utils.data.DataLoader(
            ds, sampler=sampler, collate_fn=lambda x: x[0], num_workers=num_workers,
        )

    dataloader = _make_dataloader(dataset, "Image", "img")
    vid_dataloader = _make_dataloader(vid_dataset, "Video", "vid")
    vid_ref_dataloader = _make_dataloader(vid_ref_dataset, "Video ref", "vid_ref")

    if accelerator.is_main_process:
        total_img = len(dataset) if dataset else 0
        total_vid = len(vid_dataset) if vid_dataset else 0
        total_ref = len(vid_ref_dataset) if vid_ref_dataset else 0
        print(f"[dist] Dataset sizes — img: {total_img}, vid: {total_vid}, ref: {total_ref}, total: {total_img+total_vid+total_ref}")
        dl_img = len(dataloader) if dataloader else 0
        dl_vid = len(vid_dataloader) if vid_dataloader else 0
        dl_ref = len(vid_ref_dataloader) if vid_ref_dataloader else 0
        print(f"[dist] Per-GPU dataloader sizes (÷{world_size}) — img: {dl_img}, vid: {dl_vid}, ref: {dl_ref}")

    vae = getattr(model.pipe, "vae")
    delattr(model.pipe, "vae")
    _dummy_dl = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(torch.zeros(1)), batch_size=1)
    model, optimizer, scheduler, _dummy_dl = accelerator.prepare(model, optimizer, scheduler, _dummy_dl)
    _align_constant_lr_state(optimizer, scheduler, learning_rate)
    vae = vae.to(accelerator.device)
    vae.eval()
    vae.requires_grad_(False)

    resume_step = 0
    resume_consumed_batches = 0
    resume_state_path = getattr(args, "resume_from_training_state", None) if args else None
    if not resume_state_path:
        latest_state = os.path.join(model_logger.output_path, "training_state_latest")
        if os.path.isdir(latest_state):
            resume_state_path = latest_state
            if accelerator.is_main_process:
                print(f"Auto-resuming from {resume_state_path}")
    if resume_state_path and os.path.isdir(resume_state_path):
        accelerator.load_state(resume_state_path)
        _align_constant_lr_state(optimizer, scheduler, learning_rate)
        meta_path = os.path.join(resume_state_path, "training_meta.json")
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                meta = json.load(f)
            resume_step = meta["num_steps"]
            resume_consumed_batches = int(meta.get("num_consumed_batches", resume_step))
            model_logger.num_steps = resume_step
            model_logger.num_consumed_batches = resume_consumed_batches
        if accelerator.is_main_process:
            print(
                f"Resumed training state from {resume_state_path}, continuing from "
                f"optimizer step {resume_step}, consumed batches {resume_consumed_batches}"
            )

    if hasattr(model, 'module') and hasattr(model.module, 'set_debug_step_offset'):
        model.module.set_debug_step_offset(resume_step)
    elif hasattr(model, 'set_debug_step_offset'):
        model.set_debug_step_offset(resume_step)

    wandb_mod = None
    if accelerator.is_main_process:
        import wandb as wandb_mod
        print((sorted(model.trainable_param_names())))
        wandb_mod.init(project=args.project_name, name=args.exp_name)
    dataloader = MixDataloader(
        dataloader,
        vid_dataloader,
        vid_ref_dataloader,
        num_epochs=num_epochs,
        samplers=samplers,
        mix_pattern=getattr(args, "dataset_mix_pattern", None) if args else None,
        mix_mode=getattr(args, "dataset_mix_mode", None) if args else None,
        rank=rank,
    )
    if resume_consumed_batches > 0:
        dataloader.skip_first_n(resume_consumed_batches)
        if accelerator.is_main_process:
            print(
                f"Skipped {resume_consumed_batches} consumed batches via "
                "sampler.set_start (O(1), no data loaded)"
            )
    import time as _time
    _step_t0 = _time.time()
    optimizer.zero_grad()
    max_steps = int(getattr(args, "max_steps", 0) or 0) if args else 0
    for data in tqdm(dataloader):
        if max_steps > 0 and model_logger.num_steps >= max_steps:
            if accelerator.is_main_process:
                print(f"[max_steps] reached {max_steps}, exiting training loop.")
            break
        _t_data = _time.time()
        _dt_data = _t_data - _step_t0
        with accelerator.accumulate(model):
            step_lr = _get_optimizer_lrs(optimizer)[0]
            loss = model(data, vae=vae)
            _t_fwd = _time.time()
            _dt_fwd = _t_fwd - _t_data
            loader_type = data.get("_loader_type", "unknown") if isinstance(data, dict) else "unknown"
            loader_type_counts, loader_type_losses = _distributed_loader_type_stats(
                loader_type,
                loss,
                accelerator.device,
            )
            _t_stats = _time.time()
            _dt_stats = _t_stats - _t_fwd
            accelerator.backward(loss)
            _t_bwd = _time.time()
            _dt_bwd = _t_bwd - _t_stats
            optimizer.step()
            _t_opt = _time.time()
            _dt_opt = _t_opt - _t_bwd
            _dt_total = _t_opt - _step_t0
            scheduler.step()
            optimizer.zero_grad()
            model_logger.num_consumed_batches += 1
            did_optimizer_update = bool(accelerator.sync_gradients)
            if did_optimizer_update:
                model_logger.on_step_end(accelerator, model, save_steps)
        if accelerator.is_main_process:
            step_label = (
                f"step {model_logger.num_steps}"
                if did_optimizer_update
                else f"step {model_logger.num_steps} (accum)"
            )
            type_counts = " ".join(
                f"{name}:{int(count)}"
                for name, count in loader_type_counts.items()
                if count > 0
            )
            print(f"{step_label}  loss: {loader_type_losses['global']:.4f}  lr: {step_lr:.2e}  "
                  f"| types: {type_counts}  "
                  f"| data: {_dt_data:.1f}s  forward: {_dt_fwd:.1f}s  backward: {_dt_bwd:.1f}s  "
                  f"stats: {_dt_stats:.1f}s  optim: {_dt_opt:.1f}s  total: {_dt_total:.1f}s")
            if wandb_mod is not None:
                log_payload = {
                    "loss": loader_type_losses["global"],
                    "lr": step_lr,
                    "time/data": _dt_data,
                    "time/forward": _dt_fwd,
                    "time/backward": _dt_bwd,
                    "time/stats": _dt_stats,
                    "time/optim": _dt_opt,
                    "time/total": _dt_total,
                }
                for name, count in loader_type_counts.items():
                    log_payload[f"batch_count/{name}"] = count
                for name, value in loader_type_losses.items():
                    log_payload[f"loss/{name}"] = value
                wandb_mod.log(log_payload, step=model_logger.num_steps)
        _step_t0 = _time.time()
    if save_steps is None:
        model_logger.on_epoch_end(accelerator, model, num_epochs)
    model_logger.on_training_end(accelerator, model, save_steps)


def wan_parser():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument("--dataset_base_path", type=str, default=None, help="Base path of the dataset.")
    parser.add_argument("--tar_shard_index", type=str, default=None, help="Optional path to tar_shards/index.json. When set, dataset relative paths covered by the index are read from tar shards instead of loose files.")
    parser.add_argument("--vid_dataset_metadata_path", type=str, default=None, help="Base path of the dataset.")
    parser.add_argument("--vid_metadata_exclude_rules", type=str, default=None, help="Semicolon-separated metadata exclude rules for vid_dataset, e.g. 'dataset=example_dataset,edit_type=example_type'.")
    parser.add_argument("--vid_ref_dataset_metadata_path", type=str, default=None, help="Path to the metadata file of the dataset.")
    parser.add_argument("--img_dataset_metadata_path", type=str, default=None, help="Path to the metadata file of the dataset.")
    parser.add_argument("--max_pixels", type=int, default=1024*1024, help="Maximum number of pixels per frame, used for dynamic resolution..")
    parser.add_argument("--height", type=int, default=None, help="Height of images or videos. Leave `height` and `width` empty to enable dynamic resolution.")
    parser.add_argument("--width", type=int, default=None, help="Width of images or videos. Leave `height` and `width` empty to enable dynamic resolution.")
    parser.add_argument("--num_frames", type=int, default=81, help="Number of frames per video. Frames are sampled from the video prefix.")
    parser.add_argument("--data_file_keys", type=str, default="image,video", help="Data file keys in the metadata. Comma-separated.")
    parser.add_argument("--dataset_repeat", type=int, default=1, help="Base repeat factor applied to every active dataset before any per-dataset overrides.")
    parser.add_argument("--img_dataset_repeat", type=int, default=None, help="Override repeat factor for the image-edit dataset.")
    parser.add_argument("--vid_dataset_repeat", type=int, default=None, help="Override repeat factor for the video-edit dataset.")
    parser.add_argument("--vid_ref_dataset_repeat", type=int, default=None, help="Override repeat factor for the video-ref dataset.")
    parser.add_argument("--auto_balance_dataset_repeats", default=False, action="store_true", help="Increase shorter dataset repeats so the epoch is not capped by the shortest loader for the active mix pattern/world size.")
    parser.add_argument("--model_paths", type=str, default=None, help="Paths to load models. In JSON format.")
    parser.add_argument("--model_id_with_origin_paths", type=str, default=None, help="Model ID with origin paths, e.g., Wan-AI/Wan2.1-T2V-1.3B:diffusion_pytorch_model*.safetensors. Comma-separated.")
    parser.add_argument("--audio_processor_config", type=str, default=None, help="Model ID with origin paths to the audio processor config, e.g., Wan-AI/Wan2.2-S2V-14B:wav2vec2-large-xlsr-53-english/")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate.")
    parser.add_argument("--num_epochs", type=int, default=1, help="Number of epochs.")
    parser.add_argument("--max_steps", type=int, default=0, help="If >0, stop training after this many optimizer steps (smoke tests).")
    parser.add_argument("--output_path", type=str, default="./models", help="Output save path.")
    parser.add_argument("--remove_prefix_in_ckpt", type=str, default="pipe.dit.", help="Remove prefix in ckpt.")
    parser.add_argument("--trainable_models", type=str, default=None, help="Models to train, e.g., dit, vae, text_encoder.")
    parser.add_argument("--project_name", type=str, default='diffsynth', help="Project name.")
    parser.add_argument("--exp_name", type=str, default='run', help="Experiment name.")
    parser.add_argument("--extra_inputs", default=None, help="Additional model inputs, comma-separated.")
    parser.add_argument("--use_gradient_checkpointing_offload", default=False, action="store_true", help="Whether to offload gradient checkpointing to CPU memory.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Gradient accumulation steps.")
    parser.add_argument("--max_timestep_boundary", type=float, default=1.0, help="Max timestep boundary (for mixed models, e.g., Wan-AI/Wan2.2-I2V-A14B).")
    parser.add_argument("--min_timestep_boundary", type=float, default=0.0, help="Min timestep boundary (for mixed models, e.g., Wan-AI/Wan2.2-I2V-A14B).")
    parser.add_argument("--find_unused_parameters", default=False, action="store_true", help="Whether to find unused parameters in DDP.")
    parser.add_argument("--save_steps", type=int, default=None, help="Number of checkpoint saving invervals. If None, checkpoints will be saved every epoch.")
    parser.add_argument("--permanent_save_steps", type=int, default=2500, help="Save an additional permanent checkpoint every N training steps, independent of save_steps. 0 to disable.")
    parser.add_argument("--dataset_num_workers", type=int, default=0, help="Number of workers for data loading.")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay.")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to the checkpoint. If provided, the model will be loaded from this checkpoint.")
    parser.add_argument("--resume_from_training_state", type=str, default=None, help="Path to training state directory (e.g., ckpt/.../training_state_step_500) for full resume including optimizer, scheduler, and step counter.")
    parser.add_argument("--mllm_model", type=str, default='Qwen/Qwen3.5-4B', help="Path to the MLLM model.")
    parser.add_argument("--mllm_gradient_checkpointing", type=bool, default=False, help="Whether to use gradient checkpointing for MLLM.")
    parser.add_argument("--mllm_max_pixels_per_frame", type=int, default=None, help="Maximum number of pixels per frame for MLLM.")
    parser.add_argument("--mllm_ref_max_pixels", type=int, default=147456, help="Maximum number of pixels per reference image for MLLM.")
    parser.add_argument("--mllm_video_sample_fps", type=float, default=1.0, help="Target FPS used by the Qwen video processor when sampling source-video frames.")
    parser.add_argument("--mllm_video_min_frames", type=int, default=2, help="Minimum number of source-video frames kept by the Qwen video processor.")
    parser.add_argument("--prompt_dropout_prob", type=float, default=0.1, help="Probability of dropping the text prompt during training.")
    parser.add_argument("--visual_dropout_given_prompt_prob", type=float, default=0.0, help="Probability of dropping all visual conditioning after the text prompt has already been dropped.")
    parser.add_argument("--neg_prompt_given_drop_prob", type=float, default=0.0, help="When a sample hits prompt-dropout AND is not visual-dropped, probability of replacing the empty prompt with a sampled short negative-direction phrase. Trains the visual_neg branch to carry negative semantics, so inference-time negative prompts (blurry / distorted / pasted / ...) can actually work. Default 0 = byte-identical to prior training.")
    parser.add_argument("--neg_prompt_pool_file", type=str, default="", help="Optional path to a newline-separated text file (or JSON list) of negative phrases. Lines starting with '#' and blank lines are ignored. Ignored when neg_prompt_given_drop_prob=0. If left empty, a built-in default pool (see diffsynth/trainers/unified_dataset.py::_DEFAULT_NEG_PROMPT_POOL) is used.")
    parser.add_argument("--dataset_mix_pattern", type=str, default=None, help="Comma-separated task mix pattern, e.g. 'img,vid,vid,vid_ref'. Default preserves the old active-loader order.")
    parser.add_argument("--dataset_mix_mode", type=str, default="sequential", help="Task mix scheduling mode: 'sequential' keeps every rank on the same loader type per step; 'rank_staggered' offsets the pattern by rank so each global optimizer step mixes loader types across ranks.")
    parser.add_argument("--metadata_shuffle_seed", type=str, default="none", help="Metadata shuffle seed for all_* JSONLs. Use an integer for a fixed seed, 'per_run' for a fresh run-level seed, or 'none' to disable.")
    parser.add_argument("--ref_max_items", type=int, default=8, help="Maximum number of reference images with dedicated index embeddings.")
    parser.add_argument("--ref_pad_first", type=bool, default=False, help="Pad reference video to the first frame.")
    parser.add_argument("--source_condition_mode", type=str, default="auto", help="Source latent conditioning mode: temporal_concat or additive.")
    parser.add_argument("--source_zero_cond_t", type=bool, default=True, help="Use timestep=0 modulation for temporal-concat source tokens.")
    parser.add_argument("--deepspeed_config", type=str, default=None, help="Path to DeepSpeed config JSON (for torchrun multi-node). If provided, uses DeepSpeed via this config instead of accelerate CLI.")
    parser.add_argument("--debug_dump_every", type=int, default=10, help="Dump training inputs every N steps (0=off).")
    parser.add_argument("--debug_dump_limit", type=int, default=10, help="Max number of dumps to write (0=unlimited).")
    parser.add_argument("--debug_dump_num_ranks", type=int, default=1, help="Dump debug inputs from ranks [0, N).")
    parser.add_argument("--debug_dump_max_frames", type=int, default=4, help="Max video frames to save per dump.")
    parser.add_argument("--debug_dump_dir", type=str, default=None, help="Debug dump output directory (default: <output_path>/debug_inputs).")
    return parser
