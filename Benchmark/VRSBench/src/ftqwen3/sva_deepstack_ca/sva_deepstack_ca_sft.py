from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torch.utils.data import Dataset

from ..shared.sft import (
    _load_image_hw_cache,
    _read_image_hw,
    _resolve_from_project,
    _strip_llava_image_placeholder,
    _write_json_atomic,
    build_sft_texts,
)
from .vision_fixed_grid import compute_fixed_grid_resize


class VrsbenchMultiTaskSVAFixedGridDataset(Dataset):
    """SVA fixed-grid 版 VRSBench 三任务联合训练数据集。"""

    def __init__(
        self,
        items: list[dict[str, Any]],
        *,
        tokenizer,
        image_processor,
        dataset_root: str | Path,
        split: str = "train",
        patch_size: int,
        merge_size: int,
        latent_grid_h: int,
        latent_grid_w: int,
        smart_resize_min_pixels: int = 256 * 256,
        smart_resize_max_pixels: int = 4096 * 4096,
        image_token: str = "<|image_pad|>",
    ) -> None:
        self.items = items
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.dataset_root = _resolve_from_project(dataset_root)
        self.split = str(split)
        self.patch_size = int(patch_size)
        self.merge_size = int(merge_size)
        self.latent_grid_h = int(latent_grid_h)
        self.latent_grid_w = int(latent_grid_w)
        self.smart_resize_min_pixels = int(smart_resize_min_pixels)
        self.smart_resize_max_pixels = int(smart_resize_max_pixels)
        self.image_token = str(image_token)

        if self.patch_size <= 0 or self.merge_size <= 0:
            raise ValueError(f"Invalid patch/merge size: patch={self.patch_size}, merge={self.merge_size}")
        if self.latent_grid_h <= 0 or self.latent_grid_w <= 0:
            raise ValueError(f"Invalid latent grid: {self.latent_grid_h}x{self.latent_grid_w}")
        if self.smart_resize_min_pixels <= 0 or self.smart_resize_max_pixels < self.smart_resize_min_pixels:
            raise ValueError(
                "Invalid smart resize range: "
                f"min={self.smart_resize_min_pixels}, max={self.smart_resize_max_pixels}"
            )

        self.images_dir = self.dataset_root / ("Images_train" if self.split == "train" else "Images_val")
        self._resize_key_cache: dict[int, tuple[int, int, int, int, int]] = {}
        self.cache_dir = self.dataset_root / "cache"
        self.image_hw_cache_path = self.cache_dir / f"vrsbench_{self.split}_image_hw.json"
        self._image_hw_cache: dict[str, tuple[int, int]] | None = None

    def _resolve_image_fs_path(self, image_name: str) -> Path:
        return (self.images_dir / str(image_name)).resolve()

    def ensure_image_hw_cache(self, *, build_if_missing: bool) -> None:
        if self._image_hw_cache is not None:
            return

        cache = _load_image_hw_cache(self.image_hw_cache_path)
        if cache is not None:
            self._image_hw_cache = cache
            return
        if not build_if_missing:
            return

        unique_names: list[str] = []
        seen: set[str] = set()
        for item in self.items:
            image_name = str(item.get("image", "")).strip()
            if not image_name:
                raise ValueError("Found empty image name while building image size cache.")
            if image_name in seen:
                continue
            seen.add(image_name)
            unique_names.append(image_name)

        images_payload: dict[str, dict[str, int]] = {}
        for image_name in unique_names:
            height, width = _read_image_hw(self._resolve_image_fs_path(image_name))
            images_payload[image_name] = {"height": int(height), "width": int(width)}

        payload = {
            "version": 1,
            "dataset": "VRSBench",
            "split": self.split,
            "images_dir": self.images_dir.name,
            "num_images": int(len(images_payload)),
            "images": images_payload,
        }
        _write_json_atomic(self.image_hw_cache_path, payload)
        self._image_hw_cache = _load_image_hw_cache(self.image_hw_cache_path)

    def _compute_resize_key_from_hw(self, *, height: int, width: int) -> tuple[int, int, int, int, int]:
        resize = compute_fixed_grid_resize(
            height=int(height),
            width=int(width),
            patch_size=int(self.patch_size),
            latent_grid_h=int(self.latent_grid_h),
            latent_grid_w=int(self.latent_grid_w),
            merge_size=int(self.merge_size),
            min_pixels=int(self.smart_resize_min_pixels),
            max_pixels=int(self.smart_resize_max_pixels),
        )
        return (
            int(resize.resized_height),
            int(resize.resized_width),
            int(resize.actual_grid_h),
            int(resize.actual_grid_w),
            int(resize.llm_image_tokens),
        )

    def get_resize_bucket_key(self, idx: int) -> tuple[int, int, int, int, int]:
        i = int(idx)
        cached = self._resize_key_cache.get(i)
        if cached is not None:
            return cached

        it = self.items[i]
        image_name = str(it.get("image", "")).strip()
        if not image_name:
            raise ValueError(f"Empty image name at idx={i}")

        if self._image_hw_cache is None:
            self._image_hw_cache = _load_image_hw_cache(self.image_hw_cache_path)

        cached_hw = None if self._image_hw_cache is None else self._image_hw_cache.get(image_name)
        if cached_hw is not None:
            key = self._compute_resize_key_from_hw(height=int(cached_hw[0]), width=int(cached_hw[1]))
        else:
            height, width = _read_image_hw(self._resolve_image_fs_path(image_name))
            key = self._compute_resize_key_from_hw(height=int(height), width=int(width))
        self._resize_key_cache[i] = key
        return key

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        it = self.items[idx]
        image_name = str(it.get("image", "")).strip()
        if not image_name:
            raise ValueError(f"Empty image name at idx={idx}")

        conv = it.get("conversations", [])
        if not isinstance(conv, list) or not conv:
            raise ValueError(f"Invalid conversations at idx={idx}")

        human_text = None
        gpt_text = None
        for message in conv:
            if not isinstance(message, dict):
                continue
            if human_text is None and str(message.get("from", "")).lower() == "human":
                human_text = str(message.get("value", ""))
                continue
            if gpt_text is None and str(message.get("from", "")).lower() == "gpt":
                gpt_text = str(message.get("value", ""))
        if human_text is None or gpt_text is None:
            raise ValueError(f"Missing human/gpt message at idx={idx}")

        prompt = _strip_llava_image_placeholder(human_text)
        answer = str(gpt_text).strip()

        image_fs_path = self._resolve_image_fs_path(image_name)
        with Image.open(str(image_fs_path)) as pil_img:
            img = pil_img.convert("RGB")

        resize = compute_fixed_grid_resize(
            height=int(img.height),
            width=int(img.width),
            patch_size=int(self.patch_size),
            latent_grid_h=int(self.latent_grid_h),
            latent_grid_w=int(self.latent_grid_w),
            merge_size=int(self.merge_size),
            min_pixels=int(self.smart_resize_min_pixels),
            max_pixels=int(self.smart_resize_max_pixels),
        )
        img = img.resize((int(resize.resized_width), int(resize.resized_height)), Image.BICUBIC)

        img_inputs = self.image_processor(
            images=img,
            return_tensors="pt",
            do_resize=False,
        )
        pixel_values = img_inputs["pixel_values"]
        if pixel_values.ndim == 3 and int(pixel_values.shape[0]) == 1:
            pixel_values = pixel_values.squeeze(0)
        if pixel_values.ndim != 2:
            raise ValueError(f"Unexpected pixel_values shape for SVA fixed-grid visual: {tuple(pixel_values.shape)}")

        actual_grid_thw = img_inputs["image_grid_thw"]
        if actual_grid_thw.ndim == 2 and int(actual_grid_thw.shape[0]) == 1:
            actual_grid_thw = actual_grid_thw.squeeze(0)
        if actual_grid_thw.ndim != 1 or int(actual_grid_thw.shape[0]) != 3:
            raise ValueError(f"Unexpected image_grid_thw shape for SVA fixed-grid visual: {tuple(actual_grid_thw.shape)}")
        actual_grid_thw = actual_grid_thw.to(dtype=torch.long)

        actual_grid_h = int(actual_grid_thw[1].item())
        actual_grid_w = int(actual_grid_thw[2].item())
        if actual_grid_h != int(resize.actual_grid_h) or actual_grid_w != int(resize.actual_grid_w):
            raise ValueError(
                "actual_image_grid_thw mismatch with computed fixed-grid resize: "
                f"processor=({actual_grid_h},{actual_grid_w}) vs resize=({resize.actual_grid_h},{resize.actual_grid_w})"
            )

        fixed_grid_thw = torch.tensor(
            [1, int(resize.latent_grid_h), int(resize.latent_grid_w)],
            dtype=torch.long,
        )
        num_image_tokens = int(resize.llm_image_tokens)
        if num_image_tokens <= 0:
            raise ValueError(f"Invalid llm_image_tokens={num_image_tokens}")

        prompt_text, full_text = build_sft_texts(
            tokenizer=self.tokenizer,
            image_path=str(image_fs_path),
            prompt=prompt,
            caption=answer,
            image_token=self.image_token,
            num_image_tokens=int(num_image_tokens),
        )

        prompt_tok = self.tokenizer(prompt_text, add_special_tokens=False, return_tensors="pt")
        full_tok = self.tokenizer(full_text, add_special_tokens=False, return_tensors="pt")

        input_ids = full_tok["input_ids"].squeeze(0)
        attention_mask = full_tok["attention_mask"].squeeze(0)

        prompt_len = int(prompt_tok["input_ids"].shape[1])
        labels = input_ids.clone()
        labels[:prompt_len] = -100

        return {
            "imgid": int(idx),
            "filename": str(image_name),
            "image_path": str(image_fs_path),
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "pixel_values": pixel_values,
            "image_grid_thw": fixed_grid_thw,
            "actual_image_grid_thw": actual_grid_thw,
        }


class QwenNativeSVAFixedGridCollator:
    def __init__(self, *, pad_token_id: int) -> None:
        self.pad_token_id = int(pad_token_id)

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, Any]:
        max_len = max(int(x["input_ids"].shape[0]) for x in batch)

        input_ids_list = []
        attention_mask_list = []
        labels_list = []
        pixel_values_list = []
        fixed_grid_list = []
        actual_grid_list = []
        meta = []

        patch_dim: int | None = None
        for ex in batch:
            ids: torch.Tensor = ex["input_ids"]
            am: torch.Tensor = ex["attention_mask"]
            labels: torch.Tensor = ex["labels"]

            pad = max_len - int(ids.shape[0])
            if pad > 0:
                ids = torch.cat([ids, torch.full((pad,), self.pad_token_id, dtype=ids.dtype)], dim=0)
                am = torch.cat([am, torch.zeros((pad,), dtype=am.dtype)], dim=0)
                labels = torch.cat([labels, torch.full((pad,), -100, dtype=labels.dtype)], dim=0)

            input_ids_list.append(ids)
            attention_mask_list.append(am)
            labels_list.append(labels)
            meta.append({"imgid": ex.get("imgid"), "filename": ex.get("filename")})

            pixel_values: torch.Tensor = ex["pixel_values"]
            if pixel_values.ndim != 2:
                raise ValueError(
                    "QwenNativeSVAFixedGridCollator expects pixel_values shape [num_patches, patch_dim], "
                    f"got {tuple(pixel_values.shape)}"
                )
            if patch_dim is None:
                patch_dim = int(pixel_values.shape[1])
            elif int(pixel_values.shape[1]) != int(patch_dim):
                raise ValueError(
                    f"Mixed patch_dim in one batch: first={patch_dim}, current={int(pixel_values.shape[1])}"
                )
            pixel_values_list.append(pixel_values)

            fixed_grid: torch.Tensor = ex["image_grid_thw"]
            actual_grid: torch.Tensor = ex["actual_image_grid_thw"]
            if fixed_grid.ndim != 1 or int(fixed_grid.shape[0]) != 3:
                raise ValueError(f"Expected fixed image_grid_thw shape [3], got {tuple(fixed_grid.shape)}")
            if actual_grid.ndim != 1 or int(actual_grid.shape[0]) != 3:
                raise ValueError(f"Expected actual_image_grid_thw shape [3], got {tuple(actual_grid.shape)}")
            fixed_grid_list.append(fixed_grid.to(dtype=torch.long))
            actual_grid_list.append(actual_grid.to(dtype=torch.long))

        return {
            "input_ids": torch.stack(input_ids_list, dim=0),
            "attention_mask": torch.stack(attention_mask_list, dim=0),
            "labels": torch.stack(labels_list, dim=0),
            "pixel_values": torch.cat(pixel_values_list, dim=0),
            "image_grid_thw": torch.stack(fixed_grid_list, dim=0),
            "actual_image_grid_thw": torch.stack(actual_grid_list, dim=0),
            "meta": meta,
        }
