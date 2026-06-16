"""First-run auto-download of Aurora's base models and trained weights from
HuggingFace into the local ``models/`` layout that the loaders already expect.

This is pure path materialization that runs *before* the unchanged inference
loaders; it does not touch the diffusion/inference math. The base models
(``Wan-AI/Wan2.2-TI2V-5B``, ``Qwen/Qwen3.5-4B``, ``Qwen/Qwen3-VL-8B-Instruct``)
are public, and so are the Aurora weights in ``yeates/aurora-weights`` — the
download needs no credentials.
"""
from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Optional

AURORA_WEIGHTS_REPO = "yeates/aurora-weights"
WAN_REPO = "Wan-AI/Wan2.2-TI2V-5B"
MLLM_REPO = "Qwen/Qwen3.5-4B"
AGENT_BASE_REPO = "Qwen/Qwen3-VL-8B-Instruct"

# The editor only needs the DiT shards + VAE; skip the ~11 GB T5 text encoder.
_WAN_ALLOW = ["diffusion_pytorch_model-*.safetensors", "Wan2.2_VAE.pth", "*.json"]

_REPO_ID_RE = re.compile(r"^[A-Za-z0-9][\w.-]*/[\w.-]+$")
# strings ending in a weight-file suffix are local checkpoint paths, not repo ids
# (a HF repo id is 'org/name' with no file extension; repo *names* may contain dots)
_WEIGHT_EXTS = (".safetensors", ".pth", ".bin", ".ckpt", ".pt", ".gguf")


def _token(token: Optional[str]) -> Optional[str]:
    return token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")


def _is_repo_id(value) -> bool:
    """A HF repo id ('org/name') that is not an existing local path."""
    if value is None:
        return False
    s = str(value)
    if s.endswith(_WEIGHT_EXTS):
        return False
    return not Path(s).exists() and bool(_REPO_ID_RE.match(s))


def _log(msg: str, verbose: bool) -> None:
    if verbose:
        print(f"[model-download] {msg}", flush=True)


def _snapshot(repo: str, local_dir, allow=None, token=None, verbose=True) -> None:
    from huggingface_hub import snapshot_download

    _log(f"fetching {repo} -> {local_dir}", verbose)
    snapshot_download(
        repo_id=repo,
        local_dir=str(local_dir),
        allow_patterns=allow,
        token=_token(token),
    )


def ensure_editor_models(model_dir, token=None, verbose=True) -> None:
    """Ensure WAN2.2-TI2V-5B (DiT shards + VAE) and Qwen3.5-4B exist in model_dir."""
    model_dir = Path(model_dir)
    wan = model_dir / "Wan2.2-TI2V-5B"
    have_wan = bool(sorted(wan.glob("diffusion_pytorch_model-*.safetensors"))) and (wan / "Wan2.2_VAE.pth").exists()
    if not have_wan:
        _snapshot(WAN_REPO, wan, allow=_WAN_ALLOW, token=token, verbose=verbose)
    mllm = model_dir / "Qwen3.5-4B"
    if not (mllm / "config.json").exists():
        _snapshot(MLLM_REPO, mllm, token=token, verbose=verbose)


def ensure_editor_ckpt(model_dir, token=None, verbose=True) -> Path:
    """Ensure aurora_editor.safetensors exists in model_dir; return its path."""
    model_dir = Path(model_dir)
    ckpt = model_dir / "aurora_editor.safetensors"
    if not ckpt.exists():
        from huggingface_hub import hf_hub_download

        _log(f"fetching {AURORA_WEIGHTS_REPO}:aurora_editor.safetensors -> {model_dir}", verbose)
        hf_hub_download(
            repo_id=AURORA_WEIGHTS_REPO,
            filename="aurora_editor.safetensors",
            local_dir=str(model_dir),
            token=_token(token),
        )
    return ckpt


def ensure_agent_models(model_dir, token=None, verbose=True):
    """Ensure the Qwen3-VL base and the aurora_agent_vlm adapter exist in model_dir.

    Returns (base_dir, adapter_dir)."""
    model_dir = Path(model_dir)
    base = model_dir / "Qwen3-VL-8B-Instruct"
    if not (base / "config.json").exists():
        _snapshot(AGENT_BASE_REPO, base, token=token, verbose=verbose)
    adapter = model_dir / "aurora_agent_vlm"
    if not (adapter / "adapter_config.json").exists():
        # the adapter lives in the aurora_agent_vlm/ subfolder of the weights repo
        _snapshot(AURORA_WEIGHTS_REPO, model_dir, allow=["aurora_agent_vlm/*"], token=token, verbose=verbose)
    return base, adapter


def resolve_editor_ckpt(ckpt, paths=None, token=None, verbose=True) -> str:
    """Resolve the editor checkpoint, downloading whatever is missing.

    - ``ckpt is None``  -> auto-download from ``yeates/aurora-weights``.
    - ``ckpt`` is a HF repo id -> download ``aurora_editor.safetensors`` from it.
    - ``ckpt`` is a local path -> use as-is.

    Always ensures the frozen base models (WAN + Qwen3.5-4B) are present first.
    """
    if paths is None:
        from evaluation.pipeline_loader import default_paths
        paths = default_paths()
    ensure_editor_models(paths.model_dir, token=token, verbose=verbose)
    if ckpt is None:
        return str(ensure_editor_ckpt(paths.model_dir, token=token, verbose=verbose))
    if _is_repo_id(ckpt):
        from huggingface_hub import hf_hub_download

        _log(f"fetching {ckpt}:aurora_editor.safetensors", verbose)
        return hf_hub_download(repo_id=str(ckpt), filename="aurora_editor.safetensors", token=_token(token))
    return str(ckpt)


def resolve_agent_paths(agent_base, agent_adapter, paths=None, token=None, verbose=True):
    """Resolve the agent base model + LoRA adapter, downloading whatever is missing.

    A path that already exists locally is used unchanged; otherwise the canonical
    HF sources are fetched into the documented ``models/`` layout."""
    if paths is None:
        from evaluation.pipeline_loader import default_paths
        paths = default_paths()
    base, adapter = ensure_agent_models(paths.model_dir, token=token, verbose=verbose)
    if agent_base is not None and Path(str(agent_base)).exists():
        base = agent_base
    if agent_adapter is not None and Path(str(agent_adapter)).exists():
        adapter = agent_adapter
    return str(base), str(adapter)


def main():
    import argparse

    from evaluation.pipeline_loader import default_paths

    ap = argparse.ArgumentParser(description="Pre-download Aurora models into models/")
    ap.add_argument("--what", choices=["editor", "agent", "all"], default="all")
    ap.add_argument("--model_dir", default=None, help="Override target dir (else AURORA_MODEL_DIR / <repo>/../models)")
    args = ap.parse_args()
    model_dir = Path(args.model_dir) if args.model_dir else default_paths().model_dir
    model_dir.mkdir(parents=True, exist_ok=True)
    if args.what in ("editor", "all"):
        ensure_editor_models(model_dir)
        ensure_editor_ckpt(model_dir)
    if args.what in ("agent", "all"):
        ensure_agent_models(model_dir)
    print(f"[model-download] models ready under {model_dir}")


if __name__ == "__main__":
    main()
