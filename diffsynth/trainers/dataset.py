import torch, torchvision, imageio, os, io, json, pandas, tempfile
import imageio.v3 as iio
from PIL import Image
import random

from diffsynth.utils.mask_overlay import compose_masked_image
from diffsynth.utils.tar_shard import TarSlicePath, get_active_shard_index

DEBUG = False

class DataProcessingPipeline:
    def __init__(self, operators=None):
        self.operators: list[DataProcessingOperator] = [] if operators is None else operators

    def __call__(self, data):
        for operator in self.operators:
            data = operator(data)
        return data

    def __rshift__(self, pipe):
        if isinstance(pipe, DataProcessingOperator):
            pipe = DataProcessingPipeline([pipe])
        return DataProcessingPipeline(self.operators + pipe.operators)


class DataProcessingOperator:
    def __call__(self, data):
        raise NotImplementedError("DataProcessingOperator cannot be called directly.")

    def __rshift__(self, pipe):
        if isinstance(pipe, DataProcessingOperator):
            pipe = DataProcessingPipeline([pipe])
        return DataProcessingPipeline([self]).__rshift__(pipe)


class DataProcessingOperatorRaw(DataProcessingOperator):
    def __call__(self, data):
        return data


class ToInt(DataProcessingOperator):
    def __call__(self, data):
        return int(data)


class ToFloat(DataProcessingOperator):
    def __call__(self, data):
        return float(data)


class ToStr(DataProcessingOperator):
    def __init__(self, none_value=""):
        self.none_value = none_value

    def __call__(self, data):
        if data is None: data = self.none_value
        return str(data)


class LoadImage(DataProcessingOperator):
    def __init__(self, convert_RGB=True):
        self.convert_RGB = convert_RGB

    def __call__(self, data: str):
        if isinstance(data, TarSlicePath):
            image = Image.open(get_active_shard_index().read_bytesio(data))
        else:
            image = Image.open(data)
        if self.convert_RGB: image = image.convert("RGB")
        return image

class ImageCropAndResize(DataProcessingOperator):
    def __init__(self, height, width, max_pixels, height_division_factor, width_division_factor):
        self.height = height
        self.width = width
        self.max_pixels = max_pixels
        self.height_division_factor = height_division_factor
        self.width_division_factor = width_division_factor

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
        if self.height is None or self.width is None:
            width, height = image.size
            if width * height > self.max_pixels:
                scale = (width * height / self.max_pixels) ** 0.5
                height, width = int(height / scale), int(width / scale)
            height = height // self.height_division_factor * self.height_division_factor
            width = width // self.width_division_factor * self.width_division_factor
        else:
            height, width = self.height, self.width
        if DEBUG: print(self.height, self.width, "height: ", height, "width: ", width)
        return height, width


    def __call__(self, data: Image.Image):
        image = self.crop_and_resize(data, *self.get_height_width(data))
        return image


class PairedImageCropAndResize(DataProcessingOperator):
    def __init__(self, loader, crop_resize, src_key="src_video", tgt_key="tgt_video"):
        self.loader = loader
        self.crop_resize = crop_resize
        self.src_key = src_key
        self.tgt_key = tgt_key

    def __call__(self, data):
        src_image = self.loader(data[self.src_key])
        tgt_image = self.loader(data[self.tgt_key])
        target_height, target_width = self.crop_resize.get_height_width(tgt_image)
        return {
            self.src_key: [self.crop_resize.crop_and_resize(src_image, target_height, target_width)],
            self.tgt_key: [self.crop_resize.crop_and_resize(tgt_image, target_height, target_width)],
        }


class ToList(DataProcessingOperator):
    def __call__(self, data):
        return [data]


class _VideoPathWithStart(str):
    """str subclass that carries a per-sample start_frame hint for LoadVideo."""
    def __new__(cls, path, start_frame=0):
        obj = super().__new__(cls, path)
        obj._start_frame = int(start_frame)
        return obj


class LoadVideo(DataProcessingOperator):
    def __init__(self, num_frames=81, time_division_factor=4, time_division_remainder=1, frame_processor=lambda x: x):
        self.num_frames = num_frames
        self.time_division_factor = time_division_factor
        self.time_division_remainder = time_division_remainder
        self.frame_processor = frame_processor

    def get_num_frames(self, reader, start_frame=0):
        num_frames = self.num_frames
        available = max(0, int(reader.count_frames()) - start_frame)
        if available < num_frames:
            num_frames = available
            while num_frames > 1 and num_frames % self.time_division_factor != self.time_division_remainder:
                num_frames -= 1
        return num_frames

    def _read_frames(self, reader, start_frame):
        if start_frame >= int(reader.count_frames()):
            start_frame = 0
        num_frames = self.get_num_frames(reader, start_frame=start_frame)
        frames = []
        for frame_id in range(start_frame, start_frame + num_frames):
            frame = reader.get_data(frame_id)
            frame = Image.fromarray(frame)
            frame = self.frame_processor(frame)
            frames.append(frame)
        return frames

    def __call__(self, data: str):
        start_frame = getattr(data, "_start_frame", 0)
        if isinstance(data, TarSlicePath):
            buf = get_active_shard_index().read(data)
            ext = os.path.splitext(str(data))[1].lower() or ".mp4"
            with tempfile.NamedTemporaryFile(dir="/dev/shm", suffix=ext) as tf:
                tf.write(buf)
                tf.flush()
                reader = imageio.get_reader(tf.name)
                try:
                    frames = self._read_frames(reader, start_frame)
                finally:
                    reader.close()
            return frames
        reader = imageio.get_reader(data)
        try:
            frames = self._read_frames(reader, start_frame)
        finally:
            reader.close()
        return frames


class SequencialProcess(DataProcessingOperator):
    def __init__(self, operator=lambda x: x):
        self.operator = operator

    def __call__(self, data):
        return [self.operator(i) for i in data]


class LoadGIF(DataProcessingOperator):
    def __init__(self, num_frames=81, time_division_factor=4, time_division_remainder=1, frame_processor=lambda x: x):
        self.num_frames = num_frames
        self.time_division_factor = time_division_factor
        self.time_division_remainder = time_division_remainder
        self.frame_processor = frame_processor

    def _read_images(self, data):
        if isinstance(data, TarSlicePath):
            buf = get_active_shard_index().read(data)
            ext = os.path.splitext(str(data))[1].lower() or ".gif"
            return iio.imread(io.BytesIO(buf), mode="RGB", extension=ext)
        return iio.imread(data, mode="RGB")

    def get_num_frames(self, path):
        num_frames = self.num_frames
        images = self._read_images(path)
        if len(images) < num_frames:
            num_frames = len(images)
            while num_frames > 1 and num_frames % self.time_division_factor != self.time_division_remainder:
                num_frames -= 1
        return num_frames

    def __call__(self, data: str):
        num_frames = self.get_num_frames(data)
        frames = []
        images = self._read_images(data)
        for img in images:
            frame = Image.fromarray(img)
            frame = self.frame_processor(frame)
            frames.append(frame)
            if len(frames) >= num_frames:
                break
        return frames


class RouteByExtensionName(DataProcessingOperator):
    def __init__(self, operator_map):
        self.operator_map = operator_map

    def __call__(self, data: str):
        file_ext_name = data.split(".")[-1].lower()
        for ext_names, operator in self.operator_map:
            if ext_names is None or file_ext_name in ext_names:
                return operator(data)
        raise ValueError(f"Unsupported file: {data}")

class RouteByType(DataProcessingOperator):
    def __init__(self, operator_map):
        self.operator_map = operator_map

    def __call__(self, data):
        for dtype, operator in self.operator_map:
            if dtype is None or isinstance(data, dtype):
                return operator(data)
        raise ValueError(f"Unsupported data: {data}")


class LoadTorchPickle(DataProcessingOperator):
    def __init__(self, map_location="cpu"):
        self.map_location = map_location

    def __call__(self, data):
        if isinstance(data, TarSlicePath):
            return torch.load(
                get_active_shard_index().read_bytesio(data),
                map_location=self.map_location,
                weights_only=False,
            )
        return torch.load(data, map_location=self.map_location, weights_only=False)


class ToAbsolutePath(DataProcessingOperator):
    def __init__(self, base_path=""):
        self.base_path = base_path

    def __call__(self, data):
        if not isinstance(data, str):
            return data
        start_frame = getattr(data, "_start_frame", 0)
        index = get_active_shard_index()
        if index is not None:
            slice_path = index.lookup(data)
            if slice_path is not None:
                return slice_path.with_start_frame(start_frame) if start_frame else slice_path
        joined = os.path.join(self.base_path, data)
        if isinstance(data, _VideoPathWithStart):
            return _VideoPathWithStart(joined, start_frame=start_frame)
        return joined

class LoadAudio(DataProcessingOperator):
    def __init__(self, sr=16000):
        self.sr = sr
    def __call__(self, data: str):
        import librosa
        input_audio, sample_rate = librosa.load(data, sr=self.sr)
        return input_audio


class LoadImageFromLMDB(DataProcessingOperator):
    """Load image from LMDB spec dict {lmdb_dir, lmdb_file, lmdb_key}."""
    def __init__(self, lmdb_roots):
        self.lmdb_roots = lmdb_roots or {}
        self._envs = {}

    def _get_env(self, lmdb_dir, lmdb_file):
        key = (lmdb_dir, lmdb_file)
        if key not in self._envs:
            import lmdb
            root = self.lmdb_roots[lmdb_dir]
            self._envs[key] = lmdb.open(os.path.join(root, lmdb_file), readonly=True, lock=False, readahead=False)
        return self._envs[key]

    def __call__(self, data):
        env = self._get_env(data["lmdb_dir"], data["lmdb_file"])
        with env.begin(buffers=True) as txn:
            buf = txn.get(data["lmdb_key"].encode())
            if buf is None:
                raise KeyError(f"LMDB key not found: {data['lmdb_key']} in {data['lmdb_dir']}/{data['lmdb_file']}")
            return Image.open(io.BytesIO(bytes(buf))).convert("RGB")


_DEFAULT_NEG_PROMPT_POOL = (
    "low quality, blurry, distorted, artifacts, watermark, text",
    "blurry, pixelated, jpeg compression artifacts",
    "distorted, warped, melted, garbled",
    "pasted-on sticker, 2D cutout, flat lighting, no shadows",
    "cartoonish, unrealistic, fake, plastic looking",
    "floating object, wrong scale, wrong perspective",
    "static, frozen, non-animated, motion-less",
    "flickering, temporally inconsistent, jittery",
    "unchanged, identical to source, ignored instruction",
    "partial edit, incomplete edit, missing step",
    "global color filter, over-saturated, tinted, washed out",
    "wrong target, edited the wrong thing",
)


def _resolve_neg_prompt_pool(neg_prompt_pool):
    """Normalize a pool arg to a tuple of non-empty stripped strings."""
    if neg_prompt_pool is None:
        return tuple(_DEFAULT_NEG_PROMPT_POOL)
    pool = tuple(str(x).strip() for x in neg_prompt_pool if str(x).strip())
    if not pool:
        raise ValueError(
            "neg_prompt_pool must be None (use built-in default) or a non-empty iterable of strings"
        )
    return pool


class UnifiedDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        base_path=None, metadata_path=None,
        repeat=1,
        data_file_keys=tuple(),
        main_data_operator=lambda x: x,
        paired_image_operator=None,
        special_operator_map=None,
        prompt_dropout_prob=0.0,
        visual_dropout_given_prompt_prob=0.0,
        neg_prompt_given_drop_prob=0.0,
        neg_prompt_pool=None,
        max_ref_items=None,
        metadata_exclude_rules=None,
        metadata_shuffle_seed=None,
    ):
        self.base_path = base_path
        self.metadata_path = metadata_path
        self.repeat = repeat
        self.data_file_keys = data_file_keys
        self.main_data_operator = main_data_operator
        self.paired_image_operator = paired_image_operator
        self.cached_data_operator = LoadTorchPickle()
        self.special_operator_map = {} if special_operator_map is None else special_operator_map
        self.prompt_dropout_prob = self._validate_dropout_prob("prompt_dropout_prob", prompt_dropout_prob)
        self.visual_dropout_given_prompt_prob = self._validate_dropout_prob(
            "visual_dropout_given_prompt_prob",
            visual_dropout_given_prompt_prob,
        )
        self.neg_prompt_given_drop_prob = self._validate_dropout_prob(
            "neg_prompt_given_drop_prob",
            neg_prompt_given_drop_prob,
        )
        self.neg_prompt_pool = (
            _resolve_neg_prompt_pool(neg_prompt_pool)
            if self.neg_prompt_given_drop_prob > 0.0
            else ()
        )
        self.max_ref_items = None if max_ref_items is None else int(max_ref_items)
        self.metadata_exclude_rules = self._parse_metadata_exclude_rules(metadata_exclude_rules)
        self.metadata_shuffle_seed = (
            None if metadata_shuffle_seed is None else int(metadata_shuffle_seed)
        )
        self.data = []
        self.cached_data = []
        self.load_from_cache = metadata_path is None
        self.load_metadata(metadata_path)

    @staticmethod
    def _validate_dropout_prob(name, value):
        value = float(value)
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must be in [0, 1], got {value}")
        return value

    @staticmethod
    def _parse_metadata_exclude_rules(rules):
        if rules is None:
            return []
        if isinstance(rules, str):
            parsed_rules = []
            for raw_rule in rules.split(";"):
                raw_rule = raw_rule.strip()
                if not raw_rule:
                    continue
                rule = {}
                for raw_clause in raw_rule.split(","):
                    raw_clause = raw_clause.strip()
                    if not raw_clause:
                        continue
                    if "=" not in raw_clause:
                        raise ValueError(
                            "metadata_exclude_rules clauses must use key=value syntax, "
                            f"got {raw_clause!r}"
                        )
                    key, value = raw_clause.split("=", 1)
                    key = key.strip()
                    value = value.strip()
                    if not key or value == "":
                        raise ValueError(
                            "metadata_exclude_rules clauses require non-empty key/value, "
                            f"got {raw_clause!r}"
                        )
                    rule[key] = value
                if rule:
                    parsed_rules.append(rule)
            return parsed_rules
        if isinstance(rules, (list, tuple)):
            parsed_rules = []
            for rule in rules:
                if not isinstance(rule, dict):
                    raise TypeError("metadata_exclude_rules list items must be dicts")
                parsed_rules.append({str(key): str(value) for key, value in rule.items()})
            return parsed_rules
        raise TypeError("metadata_exclude_rules must be None, a string, or a list of dicts")

    @staticmethod
    def _entry_matches_rule(entry, rule):
        for key, expected in rule.items():
            if key not in entry:
                return False
            value = entry.get(key)
            if isinstance(value, (list, tuple)):
                if expected not in {str(item) for item in value}:
                    return False
            elif str(value) != expected:
                return False
        return True

    @staticmethod
    def _format_metadata_rule(rule):
        return ",".join(f"{key}={value}" for key, value in rule.items())

    @staticmethod
    def _has_visual_condition(data):
        return (
            bool(data.get("src_video"))
            or bool(data.get("ref_image"))
            or bool(data.get("source_input"))
            or bool(data.get("ref_mask"))
        )

    @staticmethod
    def _compose_ref_mask_into_ref_image(data):
        """If data has a ref_mask, composite it onto the paired ref_image[i]
        (or src_video[0] when unpaired) and write the result into
        data["ref_image"]. The ref_mask field is consumed (set to None) so
        downstream code sees a normal ref_image — VLM and VAE both receive
        the composite via the same `ref_image` channel.

        No-op when ref_mask is missing / None / all-None (e.g. after visual
        dropout, or on samples without a ref_mask field)."""
        ref_mask = data.get("ref_mask")
        if ref_mask is None:
            return data
        if not isinstance(ref_mask, (list, tuple)):
            ref_mask = [ref_mask]
        if not any(m is not None for m in ref_mask):
            data["ref_mask"] = None
            return data

        ref_image = data.get("ref_image")
        if ref_image is None:
            ref_image_list = []
        elif not isinstance(ref_image, (list, tuple)):
            ref_image_list = [ref_image]
        else:
            ref_image_list = list(ref_image)

        src_video = data.get("src_video")
        src_first_frame = src_video[0] if src_video else None

        out_refs = list(ref_image_list)
        total = max(len(ref_image_list), len(ref_mask))
        for i in range(total):
            mask_item = ref_mask[i] if i < len(ref_mask) else None
            base_item = ref_image_list[i] if i < len(ref_image_list) else None
            if mask_item is None:
                continue
            base = base_item if base_item is not None else src_first_frame
            if base is None:
                raise ValueError(
                    "ref_mask provided without a base image (no paired ref_image and no src_video)."
                )
            composite = compose_masked_image(base, mask_item)
            if i < len(out_refs):
                out_refs[i] = composite
            else:
                out_refs.append(composite)

        data["ref_image"] = out_refs if out_refs else None
        data["ref_mask"] = None
        return data

    def _sample_dropout_flags(self, data):
        drop_prompt = self.prompt_dropout_prob > 0.0 and random.random() < self.prompt_dropout_prob
        drop_visual = (
            drop_prompt
            and self.visual_dropout_given_prompt_prob > 0.0
            and self._has_visual_condition(data)
            and random.random() < self.visual_dropout_given_prompt_prob
        )
        neg_text = None
        if (
            drop_prompt
            and not drop_visual
            and self.neg_prompt_given_drop_prob > 0.0
            and self.neg_prompt_pool
            and random.random() < self.neg_prompt_given_drop_prob
        ):
            neg_text = random.choice(self.neg_prompt_pool)
        return drop_prompt, drop_visual, neg_text

    @staticmethod
    def _apply_condition_dropout(data, *, drop_prompt, drop_visual, neg_text=None):
        if drop_prompt and "prompt" in data:
            data["prompt"] = neg_text if neg_text is not None else ""
        if drop_visual:
            for key in ("src_video", "source_input", "ref_image", "ref_mask"):
                if key in data:
                    data[key] = None
        return data

    @staticmethod
    def default_image_operator(
        base_path="",
        max_pixels=1920*1080, height=None, width=None,
        height_division_factor=16, width_division_factor=16,
        lmdb_roots=None,
    ):
        crop_resize = ImageCropAndResize(height, width, max_pixels, height_division_factor, width_division_factor)
        single_img_op = ToAbsolutePath(base_path) >> LoadImage() >> crop_resize
        operators = []
        if lmdb_roots:
            operators.append((dict, LoadImageFromLMDB(lmdb_roots) >> crop_resize))
        operators.extend([
            (str, single_img_op),
            (list, SequencialProcess(RouteByType(operator_map=(
                [(dict, LoadImageFromLMDB(lmdb_roots) >> crop_resize)] if lmdb_roots else []
            ) + [(str, single_img_op)]))),
        ])
        return RouteByType(operator_map=operators)

    @staticmethod
    def default_image_load_operator(base_path="", lmdb_roots=None):
        single_img_op = ToAbsolutePath(base_path) >> LoadImage()
        operators = []
        if lmdb_roots:
            operators.append((dict, LoadImageFromLMDB(lmdb_roots)))
        operators.extend([
            (str, single_img_op),
            (list, SequencialProcess(RouteByType(operator_map=(
                [(dict, LoadImageFromLMDB(lmdb_roots))] if lmdb_roots else []
            ) + [(str, single_img_op)]))),
        ])
        return RouteByType(operator_map=operators)

    @staticmethod
    def default_mask_load_operator(base_path=""):
        single_img_op = ToAbsolutePath(base_path) >> LoadImage(convert_RGB=False)
        return RouteByType(operator_map=[
            (str, single_img_op),
            (list, SequencialProcess(RouteByType(operator_map=[(str, single_img_op)]))),
        ])

    @staticmethod
    def default_paired_image_operator(
        base_path="",
        max_pixels=1920*1080, height=None, width=None,
        height_division_factor=16, width_division_factor=16,
        lmdb_roots=None,
    ):
        loader = UnifiedDataset.default_image_load_operator(
            base_path=base_path,
            lmdb_roots=lmdb_roots,
        )
        crop_resize = ImageCropAndResize(
            height,
            width,
            max_pixels,
            height_division_factor,
            width_division_factor,
        )
        return PairedImageCropAndResize(loader, crop_resize)

    @staticmethod
    def default_video_operator(
        base_path="",
        max_pixels=1920*1080, height=None, width=None,
        height_division_factor=16, width_division_factor=16,
        num_frames=81, time_division_factor=4, time_division_remainder=1,
        lmdb_roots=None,
    ):
        crop_resize = ImageCropAndResize(height, width, max_pixels, height_division_factor, width_division_factor)
        str_op = ToAbsolutePath(base_path) >> RouteByExtensionName(operator_map=[
            (("jpg", "jpeg", "png", "webp"), LoadImage() >> crop_resize >> ToList()),
            (("gif",), LoadGIF(
                num_frames, time_division_factor, time_division_remainder,
                frame_processor=crop_resize,
            )),
            (("mp4", "avi", "mov", "wmv", "mkv", "flv", "webm"), LoadVideo(
                num_frames, time_division_factor, time_division_remainder,
                frame_processor=crop_resize,
            )),
        ])
        operators = []
        if lmdb_roots:
            operators.append((dict, LoadImageFromLMDB(lmdb_roots) >> crop_resize >> ToList()))
        operators.append((str, str_op))
        return RouteByType(operator_map=operators)

    def search_for_cached_data_files(self, path):
        for file_name in os.listdir(path):
            subpath = os.path.join(path, file_name)
            if os.path.isdir(subpath):
                self.search_for_cached_data_files(subpath)
            elif subpath.endswith(".pth"):
                self.cached_data.append(subpath)

    def _maybe_filter_metadata(self, metadata_path):
        if self.load_from_cache or not self.metadata_exclude_rules:
            return
        original_count = len(self.data)
        if original_count == 0:
            return
        removed_by_rule = [0] * len(self.metadata_exclude_rules)
        kept = []
        for entry in self.data:
            matched = False
            for idx, rule in enumerate(self.metadata_exclude_rules):
                if self._entry_matches_rule(entry, rule):
                    removed_by_rule[idx] += 1
                    matched = True
                    break
            if not matched:
                kept.append(entry)
        removed = original_count - len(kept)
        path_label = metadata_path or self.metadata_path or "<memory>"
        self.data = kept
        print(
            f"Filtered {removed}/{original_count} metadata rows from {path_label} "
            f"using {len(self.metadata_exclude_rules)} exclude rule(s)."
        )
        for idx, count in enumerate(removed_by_rule):
            if count:
                print(f"  - removed {count} rows matching {self._format_metadata_rule(self.metadata_exclude_rules[idx])}")
        if len(self.data) == 0:
            raise ValueError(f"No metadata rows remain after filtering {path_label}.")

    def _maybe_shuffle_metadata(self, metadata_path):
        if self.load_from_cache or self.metadata_shuffle_seed is None:
            return
        if len(self.data) <= 1:
            return
        path_label = metadata_path or self.metadata_path or "<memory>"
        random.Random(self.metadata_shuffle_seed).shuffle(self.data)
        print(
            f"Shuffled {len(self.data)} metadata rows from {path_label} "
            f"with local seed {self.metadata_shuffle_seed}."
        )

    def load_metadata(self, metadata_path):
        if metadata_path is None:
            print("No metadata_path. Searching for cached data files.")
            self.search_for_cached_data_files(self.base_path)
            print(f"{len(self.cached_data)} cached data files found.")
        elif metadata_path.endswith(".json"):
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
            self.data = metadata
        elif metadata_path.endswith(".jsonl"):
            metadata = []
            with open(metadata_path, 'r') as f:
                for line in f:
                    metadata.append(json.loads(line.strip()))
            self.data = metadata
        else:
            metadata = pandas.read_csv(metadata_path)
            self.data = [metadata.iloc[i].to_dict() for i in range(len(metadata))]
        self._maybe_filter_metadata(metadata_path)
        self._maybe_shuffle_metadata(metadata_path)

    def _meta_summary(self, raw):
        """One-line summary of a metadata entry for error logging."""
        ds = raw.get("dataset", "?")
        et = raw.get("edit_type", "?")
        src = raw.get("src_video", "")
        tgt = raw.get("tgt_video", "")
        ref = raw.get("ref_image", "")
        if isinstance(src, str) and len(src) > 60: src = "..." + src[-57:]
        if isinstance(tgt, str) and len(tgt) > 60: tgt = "..." + tgt[-57:]
        if isinstance(src, dict): src = f"lmdb:{src.get('lmdb_key','')[-40:]}"
        if isinstance(tgt, dict): tgt = f"lmdb:{tgt.get('lmdb_key','')[-40:]}"
        n_ref = len(ref) if isinstance(ref, list) else (1 if ref else 0)
        return f"[{ds}/{et}] src={src} tgt={tgt} refs={n_ref}"

    def _skip_ref_reason(self, raw):
        if self.max_ref_items is None:
            return None
        ref = raw.get("ref_image")
        if not isinstance(ref, list):
            return None
        if len(ref) <= self.max_ref_items:
            return None
        return f"skip sample with {len(ref)} refs > max_ref_items={self.max_ref_items}"

    def _should_use_paired_image_operator(self, raw):
        return (
            self.paired_image_operator is not None
            and raw.get("media_type") == "image"
            and "src_video" in raw
            and "tgt_video" in raw
        )

    def _maybe_mark_ditto_start(self, data, raw_meta):
        start = 13
        ds = (raw_meta or {}).get("dataset", "")
        dataset_flag_matches = isinstance(ds, str) and ds.startswith("ditto")
        for key in ("src_video", "tgt_video"):
            path = data.get(key)
            if not isinstance(path, str) or isinstance(path, _VideoPathWithStart):
                continue
            path_is_ditto = "ditto/" in path
            if dataset_flag_matches or path_is_ditto:
                data[key] = _VideoPathWithStart(path, start_frame=start)

    def __getitem__(self, data_id):
        max_retry = 30
        retry_count = 0
        while retry_count < max_retry:
            raw_meta = None
            try:
                if self.load_from_cache:
                    data = self.cached_data[data_id % len(self.cached_data)]
                    data = self.cached_data_operator(data)
                else:
                    data = self.data[data_id % len(self.data)].copy()
                    raw_meta = data.copy()
                    skip_reason = self._skip_ref_reason(raw_meta)
                    if skip_reason is not None:
                        retry_count += 1
                        data_id = random.randint(0, len(self.data) - 1) % len(self.data)
                        continue
                    paired_image_processed = False
                    if self._should_use_paired_image_operator(raw_meta):
                        data.update(self.paired_image_operator(raw_meta))
                        paired_image_processed = True
                    self._maybe_mark_ditto_start(data, raw_meta)
                    for key in self.data_file_keys:
                        if paired_image_processed and key in ("src_video", "tgt_video"):
                            continue
                        if key in data:
                            if key in self.special_operator_map:
                                data[key] = self.special_operator_map[key](data[key])
                            elif key in self.data_file_keys:
                                if isinstance(data[key], list):
                                    data[key] = [self.main_data_operator(item)[0] for item in data[key]]
                                else:
                                    data[key] = self.main_data_operator(data[key])
                    if data.get("src_video") is not None and data.get("tgt_video") is not None:
                        err_message = self.check_paired_size(data['src_video'], data['tgt_video'])
                        if err_message:
                            raise ValueError(err_message)
                    drop_prompt, drop_visual, neg_text = self._sample_dropout_flags(data)
                    data = self._apply_condition_dropout(
                        data,
                        drop_prompt=drop_prompt,
                        drop_visual=drop_visual,
                        neg_text=neg_text,
                    )
                    data = self._compose_ref_mask_into_ref_image(data)
                return data
            except Exception as e:
                summary = self._meta_summary(raw_meta) if raw_meta else f"data_id={data_id}"
                print(f"Error {retry_count}/{max_retry} {summary}: {e}")
                retry_count += 1
                data_id = random.randint(0, len(self.data) - 1) % len(self.data)
                continue
        raise ValueError(f"Failed to load data {data_id} after {max_retry} retries.")

    def __len__(self):
        if self.load_from_cache:
            return len(self.cached_data) * self.repeat
        else:
            return len(self.data) * self.repeat

    def check_data_equal(self, data1, data2):
        if len(data1) != len(data2):
            return False
        for k in data1:
            if data1[k] != data2[k]:
                return False
        return True

    def check_paired_size(self, data1, data2):
        err_message = ""
        if data1[0].size[0] != data2[0].size[0]:
            err_message += f'mismatch width size {data1[0].size[0]} {data2[0].size[0]}'
        if data1[0].size[1] != data2[0].size[1]:
            err_message += f'mismatch height size {data1[0].size[1]} {data2[0].size[1]}'
        if len(data1) != len(data2):
            err_message += f'mismatch frame length {len(data1)} {len(data2)}'
        return err_message
