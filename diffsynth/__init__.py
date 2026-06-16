from diffsynth.data import VideoData, merge_video_audio, save_frames, save_video, save_video_with_audio
from diffsynth.utils import ModelConfig

__all__ = [
    "ModelConfig",
    "VideoData",
    "WanVideoPipeline",
    "merge_video_audio",
    "save_frames",
    "save_video",
    "save_video_with_audio",
]


def __getattr__(name):
    if name == "WanVideoPipeline":
        from diffsynth.pipelines import WanVideoPipeline

        return WanVideoPipeline
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
