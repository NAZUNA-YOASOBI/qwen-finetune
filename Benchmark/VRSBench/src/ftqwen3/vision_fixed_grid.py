from __future__ import annotations

from dataclasses import dataclass

from .vision_resize import smart_resize


@dataclass(frozen=True)
class FixedGridResizeResult:
    resized_height: int
    resized_width: int
    actual_grid_h: int
    actual_grid_w: int
    latent_grid_h: int
    latent_grid_w: int
    latent_patch_tokens: int
    llm_image_tokens: int
    factor: int


def compute_fixed_grid_resize(
    *,
    height: int,
    width: int,
    patch_size: int,
    latent_grid_h: int,
    latent_grid_w: int,
    merge_size: int,
    min_pixels: int,
    max_pixels: int,
) -> FixedGridResizeResult:
    p = int(patch_size)
    latent_h = int(latent_grid_h)
    latent_w = int(latent_grid_w)
    m = int(merge_size)
    if p <= 0 or latent_h <= 0 or latent_w <= 0 or m <= 0:
        raise ValueError(
            f"Invalid fixed-grid resize args: patch={p}, latent={latent_h}x{latent_w}, merge={m}"
        )

    # 为了让 patch-grid 的高宽都能整除固定 latent 网格，
    # 像素尺寸必须分别是 patch_size * latent_grid_side 的整数倍。
    factor_h = int(p * latent_h)
    factor_w = int(p * latent_w)
    if factor_h != factor_w:
        raise ValueError(
            "Current fixed-grid resize helper expects square latent grid factor, "
            f"got factor_h={factor_h}, factor_w={factor_w}"
        )
    factor = int(factor_h)

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

    actual_grid_h = int(resized_h // p)
    actual_grid_w = int(resized_w // p)
    if actual_grid_h % latent_h != 0 or actual_grid_w % latent_w != 0:
        raise ValueError(
            "Actual patch grid must be divisible by fixed latent grid, "
            f"got actual={actual_grid_h}x{actual_grid_w}, latent={latent_h}x{latent_w}"
        )
    if latent_h % m != 0 or latent_w % m != 0:
        raise ValueError(
            "Fixed latent grid must be divisible by merge_size, "
            f"got latent={latent_h}x{latent_w}, merge={m}"
        )

    latent_patch_tokens = int(latent_h * latent_w)
    llm_image_tokens = int(latent_patch_tokens // (m**2))
    return FixedGridResizeResult(
        resized_height=int(resized_h),
        resized_width=int(resized_w),
        actual_grid_h=int(actual_grid_h),
        actual_grid_w=int(actual_grid_w),
        latent_grid_h=int(latent_h),
        latent_grid_w=int(latent_w),
        latent_patch_tokens=int(latent_patch_tokens),
        llm_image_tokens=int(llm_image_tokens),
        factor=int(factor),
    )
