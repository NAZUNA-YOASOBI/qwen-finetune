from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class VisionResizeResult:
    resized_height: int
    resized_width: int
    grid_h: int
    grid_w: int
    num_image_tokens: int
    factor: int


def smart_resize(
    *,
    height: int,
    width: int,
    factor: int,
    min_pixels: int,
    max_pixels: int,
) -> tuple[int, int]:
    h = int(height)
    w = int(width)
    f = int(factor)
    min_p = int(min_pixels)
    max_p = int(max_pixels)

    if h <= 0 or w <= 0:
        raise ValueError(f"Invalid image size: {h}x{w}")
    if f <= 0:
        raise ValueError(f"factor must be > 0, got {f}")
    if min_p <= 0 or max_p <= 0 or min_p > max_p:
        raise ValueError(f"Invalid min/max pixels: min={min_p}, max={max_p}")

    ratio = max(h, w) / min(h, w)
    if ratio > 200:
        raise ValueError(f"absolute aspect ratio must be smaller than 200, got {ratio}")

    h_bar = max(f, round(h / f) * f)
    w_bar = max(f, round(w / f) * f)

    if h_bar * w_bar > max_p:
        beta = math.sqrt((h * w) / max_p)
        h_bar = max(f, math.floor(h / beta / f) * f)
        w_bar = max(f, math.floor(w / beta / f) * f)
    elif h_bar * w_bar < min_p:
        beta = math.sqrt(min_p / (h * w))
        h_bar = math.ceil(h * beta / f) * f
        w_bar = math.ceil(w * beta / f) * f

    return int(h_bar), int(w_bar)


def compute_vision_resize(
    *,
    height: int,
    width: int,
    patch_size: int,
    merge_size: int,
    min_pixels: int,
    max_pixels: int,
) -> VisionResizeResult:
    p = int(patch_size)
    m = int(merge_size)
    if p <= 0 or m <= 0:
        raise ValueError(f"Invalid patch/merge size: patch={p}, merge={m}")

    factor = int(p * m)
    resized_h, resized_w = smart_resize(
        height=int(height),
        width=int(width),
        factor=int(factor),
        min_pixels=int(min_pixels),
        max_pixels=int(max_pixels),
    )

    if resized_h % p != 0 or resized_w % p != 0:
        raise ValueError(
            "Resized shape must be divisible by patch_size, "
            f"got {resized_h}x{resized_w}, patch_size={p}"
        )

    grid_h = int(resized_h // p)
    grid_w = int(resized_w // p)
    if grid_h % m != 0 or grid_w % m != 0:
        raise ValueError(
            "Patch grid must be divisible by merge_size, "
            f"got grid={grid_h}x{grid_w}, merge_size={m}"
        )

    num_image_tokens = int((grid_h * grid_w) // (m**2))
    return VisionResizeResult(
        resized_height=int(resized_h),
        resized_width=int(resized_w),
        grid_h=int(grid_h),
        grid_w=int(grid_w),
        num_image_tokens=int(num_image_tokens),
        factor=int(factor),
    )
