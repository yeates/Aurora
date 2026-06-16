"""On-the-fly mask -> VLM-friendly masked image compositing.

Design rule: the dataset stores raw (binary) masks. Whenever a mask would
otherwise reach the VLM, route the (image, mask) pair through
``compose_masked_image`` first and feed the resulting masked image to the VLM.
Raw binary masks must not be sent to the VLM directly — they are out of
distribution for natural-image VLMs and produce near-zero semantic signal.
"""

from __future__ import annotations

from typing import Optional, Tuple, Union

import numpy as np
import torch
from PIL import Image, ImageFilter

ImageLike = Union[Image.Image, np.ndarray, torch.Tensor]
MaskLike = Union[Image.Image, np.ndarray, torch.Tensor]

_VALID_OUTPUT_FORMATS = ("pil", "numpy", "tensor")


def _tensor_to_numpy_hwc_uint8(tensor: torch.Tensor) -> np.ndarray:
    t = tensor.detach().cpu()
    if t.ndim == 4 and t.shape[0] == 1:
        t = t[0]
    if t.ndim == 3 and t.shape[0] in (1, 3, 4) and t.shape[-1] not in (1, 3, 4):
        t = t.permute(1, 2, 0).contiguous()
    if t.is_floating_point():
        t = t.float()
        lo = float(t.min().item()) if t.numel() > 0 else 0.0
        hi = float(t.max().item()) if t.numel() > 0 else 0.0
        if lo < -0.01 and hi <= 1.01:
            t = (t + 1.0) * 0.5
        elif hi <= 1.01:
            pass
        else:
            t = t / 255.0
        t = (t.clamp(0.0, 1.0) * 255.0).round()
    return t.to(torch.uint8).numpy()


def _to_hwc_uint8_image(image: ImageLike) -> np.ndarray:
    if isinstance(image, Image.Image):
        return np.asarray(image.convert("RGB"), dtype=np.uint8).copy()
    if isinstance(image, torch.Tensor):
        arr = _tensor_to_numpy_hwc_uint8(image)
    elif isinstance(image, np.ndarray):
        arr = image.copy()
    else:
        raise TypeError(f"Unsupported image type: {type(image)}")

    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    elif arr.ndim == 3:
        if arr.shape[0] in (1, 3, 4) and arr.shape[-1] not in (1, 3, 4):
            arr = np.transpose(arr, (1, 2, 0))
        if arr.shape[-1] == 1:
            arr = np.repeat(arr, 3, axis=-1)
        if arr.shape[-1] == 4:
            arr = arr[..., :3]
    else:
        raise ValueError(f"Unsupported image shape: {arr.shape}")

    if arr.dtype != np.uint8:
        a = arr.astype(np.float32)
        if a.size > 0:
            hi = float(a.max())
            lo = float(a.min())
            if lo < -0.01 and hi <= 1.01:
                a = (a + 1.0) * 0.5 * 255.0
            elif hi <= 1.01:
                a = a * 255.0
        arr = np.clip(a, 0.0, 255.0).astype(np.uint8)

    return np.ascontiguousarray(arr)


def _to_hw_float_mask(mask: MaskLike) -> np.ndarray:
    if isinstance(mask, Image.Image):
        if mask.mode in ("1", "L", "I", "F"):
            arr = np.asarray(mask)
        else:
            arr = np.asarray(mask.convert("L"))
    elif isinstance(mask, torch.Tensor):
        arr = mask.detach().cpu().numpy()
    elif isinstance(mask, np.ndarray):
        arr = mask
    else:
        raise TypeError(f"Unsupported mask type: {type(mask)}")

    arr = np.asarray(arr)
    if arr.ndim == 3:
        if arr.shape[-1] in (1, 3, 4):
            arr = arr[..., 0] if arr.shape[-1] == 1 else arr[..., :3].mean(axis=-1)
        elif arr.shape[0] in (1, 3, 4):
            arr = arr[0] if arr.shape[0] == 1 else arr[:3].mean(axis=0)
        else:
            raise ValueError(f"Unsupported mask shape: {arr.shape}")
    elif arr.ndim == 4 and arr.shape[0] == 1:
        return _to_hw_float_mask(arr[0])
    elif arr.ndim != 2:
        raise ValueError(f"Unsupported mask shape: {arr.shape}")

    arr = arr.astype(np.float32)
    if arr.size > 0:
        hi = float(arr.max())
        lo = float(arr.min())
        if lo < -0.01 and hi <= 1.01:
            arr = (arr + 1.0) * 0.5
        elif hi > 1.0 + 1e-5:
            arr = arr / 255.0
    return np.clip(arr, 0.0, 1.0)


def _resize_mask(mask: np.ndarray, target_hw: Tuple[int, int]) -> np.ndarray:
    H, W = target_hw
    if mask.shape == (H, W):
        return mask
    m_pil = Image.fromarray(np.clip(mask * 255.0, 0, 255).astype(np.uint8), mode="L")
    m_pil = m_pil.resize((W, H), Image.BILINEAR)
    return np.asarray(m_pil, dtype=np.float32) / 255.0


def _draw_contour(out: np.ndarray, mask_binary: np.ndarray, color: Tuple[int, int, int], width: int) -> np.ndarray:
    mask_u8 = (mask_binary * 255).astype(np.uint8)
    if mask_u8.max() == 0 or mask_u8.min() == 255:
        return out
    kernel = max(3, 2 * int(width) + 1)
    mask_pil = Image.fromarray(mask_u8, mode="L")
    dilated = np.asarray(mask_pil.filter(ImageFilter.MaxFilter(kernel)))
    eroded = np.asarray(mask_pil.filter(ImageFilter.MinFilter(kernel)))
    edge = (dilated > 127) & (eroded < 128)
    if edge.any():
        out[edge] = np.array(color, dtype=np.uint8)
    return out


def compose_masked_image(
    image: ImageLike,
    mask: MaskLike,
    overlay_color: Tuple[int, int, int] = (255, 0, 0),
    alpha: float = 0.55,
    mask_threshold: float = 0.5,
    draw_contour: bool = True,
    contour_width: int = 3,
    contour_color: Optional[Tuple[int, int, int]] = None,
    output_format: str = "pil",
) -> ImageLike:
    """Composite a mask onto an image to produce a VLM-friendly masked image.

    Wherever ``mask > mask_threshold`` the image is blended toward
    ``overlay_color`` at ``alpha`` opacity. When ``draw_contour`` is set, a
    crisp ``contour_width``-pixel outline (in ``contour_color``, defaults to
    ``overlay_color``) is drawn on the mask boundary.

    Accepted input types are PIL images, NumPy arrays, and PyTorch tensors.
    Images may be uint8 HxWx3 / HxWx1 / HxW, float [0,1], float [-1,1], or CHW
    layout. Masks may be HxW, HxWx1, HxWx3, or CHW, either uint8 ([0,255]) or
    float ([0,1] / [-1,1]). Masks whose spatial size differs from the image
    are bilinearly resized.

    Returns the composited image in ``output_format`` — ``"pil"`` (default),
    ``"numpy"`` (HxWx3 uint8), or ``"tensor"`` (HxWx3 uint8).
    """
    if output_format not in _VALID_OUTPUT_FORMATS:
        raise ValueError(f"output_format must be one of {_VALID_OUTPUT_FORMATS}, got {output_format!r}")
    if not 0.0 <= float(alpha) <= 1.0:
        raise ValueError(f"alpha must be in [0, 1], got {alpha}")
    if len(overlay_color) != 3:
        raise ValueError(f"overlay_color must be an RGB triple, got {overlay_color}")

    img = _to_hwc_uint8_image(image)
    mask_f = _to_hw_float_mask(mask)
    mask_f = _resize_mask(mask_f, img.shape[:2])

    if mask_threshold is not None and mask_threshold > 0:
        mask_binary = (mask_f >= float(mask_threshold)).astype(np.float32)
    else:
        mask_binary = mask_f

    blend = (mask_binary * float(alpha))[..., None]
    overlay = np.array(overlay_color, dtype=np.float32).reshape(1, 1, 3)
    out = img.astype(np.float32) * (1.0 - blend) + overlay * blend
    out = np.clip(out, 0.0, 255.0).astype(np.uint8)

    if draw_contour and contour_width > 0:
        out = _draw_contour(out, mask_binary, contour_color or overlay_color, contour_width)

    if output_format == "numpy":
        return out
    if output_format == "tensor":
        return torch.from_numpy(out)
    return Image.fromarray(out)


__all__ = ["compose_masked_image"]
