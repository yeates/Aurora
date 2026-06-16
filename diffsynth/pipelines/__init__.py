__all__ = ["WanVideoPipeline"]


def __getattr__(name):
    if name == "WanVideoPipeline":
        from diffsynth.pipelines.wan_video import WanVideoPipeline

        return WanVideoPipeline
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
