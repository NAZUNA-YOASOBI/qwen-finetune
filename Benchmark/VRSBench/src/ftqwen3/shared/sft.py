from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torch.utils.data import Dataset

from .vision_resize import compute_vision_resize


def _project_root() -> Path:
    # src/ftqwen3/shared/sft.py -> parents[2] == 项目根目录
    return Path(__file__).resolve().parents[2]


def _resolve_from_project(path: str | Path) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    return (_project_root() / p).resolve()


def _read_image_hw(image_fs_path: Path) -> tuple[int, int]:
    with Image.open(str(image_fs_path)) as pil_img:
        return int(pil_img.height), int(pil_img.width)


def _compute_resize_key_from_hw(
    *,
    height: int,
    width: int,
    patch_size: int,
    merge_size: int,
    smart_resize_min_pixels: int,
    smart_resize_max_pixels: int,
) -> tuple[int, int, int, int, int]:
    resize = compute_vision_resize(
        height=int(height),
        width=int(width),
        patch_size=int(patch_size),
        merge_size=int(merge_size),
        min_pixels=int(smart_resize_min_pixels),
        max_pixels=int(smart_resize_max_pixels),
    )
    return (
        int(resize.resized_height),
        int(resize.resized_width),
        int(resize.grid_h),
        int(resize.grid_w),
        int(resize.num_image_tokens),
    )


def _load_image_hw_cache(cache_path: Path) -> dict[str, tuple[int, int]] | None:
    if not cache_path.is_file():
        return None
    try:
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None

    raw_images = payload.get("images")
    if not isinstance(raw_images, dict):
        return None

    image_hw: dict[str, tuple[int, int]] = {}
    for image_name, hw in raw_images.items():
        if not isinstance(hw, dict):
            return None
        try:
            height = int(hw.get("height", 0))
            width = int(hw.get("width", 0))
        except Exception:
            return None
        if height <= 0 or width <= 0:
            return None
        image_hw[str(image_name)] = (int(height), int(width))
    return image_hw


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    tmp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp_path.replace(path)


def expand_image_tokens(text: str, *, image_token: str, num_image_tokens: int) -> str:
    if num_image_tokens <= 0:
        raise ValueError(f"num_image_tokens must be > 0. Got {num_image_tokens}")
    if text.count(image_token) != 1:
        raise ValueError(f"Expected exactly 1 {image_token} in prompt, got {text.count(image_token)}")
    return text.replace(image_token, image_token * int(num_image_tokens), 1)


@dataclass(frozen=True)
class CaptionExample:
    imgid: int
    filename: str
    image_path: str
    caption: str


def build_chat_messages(*, image_path: str, prompt: str, caption: str | None) -> list[dict[str, Any]]:
    user = {
        "role": "user",
        "content": [
            {"type": "image", "image": str(image_path)},
            {"type": "text", "text": str(prompt)},
        ],
    }
    if caption is None:
        return [user]
    assistant = {"role": "assistant", "content": str(caption)}
    return [user, assistant]


def build_sft_texts(
    *,
    tokenizer,
    image_path: str,
    prompt: str,
    caption: str,
    image_token: str = "<|image_pad|>",
    num_image_tokens: int,
) -> tuple[str, str]:
    prompt_messages = build_chat_messages(image_path=image_path, prompt=prompt, caption=None)
    full_messages = build_chat_messages(image_path=image_path, prompt=prompt, caption=caption)

    prompt_text = tokenizer.apply_chat_template(prompt_messages, tokenize=False, add_generation_prompt=True)
    full_text = tokenizer.apply_chat_template(full_messages, tokenize=False, add_generation_prompt=False)

    prompt_text = expand_image_tokens(prompt_text, image_token=image_token, num_image_tokens=num_image_tokens)
    full_text = expand_image_tokens(full_text, image_token=image_token, num_image_tokens=num_image_tokens)
    return prompt_text, full_text


class RsicdCaptionSFTDataset(Dataset):
    def __init__(
        self,
        rows: list[dict[str, Any]],
        *,
        tokenizer,
        image_processor,
        prompt: str,
        image_size: int,
        patch_size: int,
        merge_size: int,
        smart_resize_min_pixels: int = 224 * 224,
        smart_resize_max_pixels: int = 512 * 512,
        image_token: str = "<|image_pad|>",
    ) -> None:
        self.rows = rows
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.prompt = str(prompt)
        self.image_size = int(image_size)
        self.patch_size = int(patch_size)
        self.merge_size = int(merge_size)
        self.smart_resize_min_pixels = int(smart_resize_min_pixels)
        self.smart_resize_max_pixels = int(smart_resize_max_pixels)
        self.image_token = str(image_token)

        if self.patch_size <= 0 or self.merge_size <= 0:
            raise ValueError(f"Invalid patch/merge size: patch={self.patch_size}, merge={self.merge_size}")
        if self.smart_resize_min_pixels <= 0 or self.smart_resize_max_pixels < self.smart_resize_min_pixels:
            raise ValueError(
                "Invalid smart resize range: "
                f"min={self.smart_resize_min_pixels}, max={self.smart_resize_max_pixels}"
            )
        self.resize_factor = int(self.patch_size * self.merge_size)
        self._resize_key_cache: dict[int, tuple[int, int, int, int, int]] = {}

    def _compute_resize_key_from_path(self, image_fs_path: Path) -> tuple[int, int, int, int, int]:
        with Image.open(str(image_fs_path)) as pil_img:
            h = int(pil_img.height)
            w = int(pil_img.width)
        resize = compute_vision_resize(
            height=int(h),
            width=int(w),
            patch_size=int(self.patch_size),
            merge_size=int(self.merge_size),
            min_pixels=int(self.smart_resize_min_pixels),
            max_pixels=int(self.smart_resize_max_pixels),
        )
        return (
            int(resize.resized_height),
            int(resize.resized_width),
            int(resize.grid_h),
            int(resize.grid_w),
            int(resize.num_image_tokens),
        )

    def get_resize_bucket_key(self, idx: int) -> tuple[int, int, int, int, int]:
        i = int(idx)
        cached = self._resize_key_cache.get(i)
        if cached is not None:
            return cached

        row = self.rows[i]
        image_fs_path = _resolve_from_project(str(row["image_path"]))
        key = self._compute_resize_key_from_path(image_fs_path)
        self._resize_key_cache[i] = key
        return key

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.rows[idx]
        ex = CaptionExample(
            imgid=int(row.get("imgid", -1)),
            filename=str(row.get("filename", "")),
            image_path=str(row["image_path"]),
            caption=str(row["caption"]),
        )

        # 图像：按 DINOv3 的方式做 resize/normalize
        image_fs_path = _resolve_from_project(ex.image_path)
        # 用上下文管理器读取图片，避免长时间训练时文件句柄堆积。
        with Image.open(str(image_fs_path)) as pil_img:
            img = pil_img.convert("RGB")
        resize = compute_vision_resize(
            height=int(img.height),
            width=int(img.width),
            patch_size=int(self.patch_size),
            merge_size=int(self.merge_size),
            min_pixels=int(self.smart_resize_min_pixels),
            max_pixels=int(self.smart_resize_max_pixels),
        )
        img_inputs = self.image_processor(
            images=img,
            return_tensors="pt",
            size={"height": int(resize.resized_height), "width": int(resize.resized_width)},
        )
        pixel_values = img_inputs["pixel_values"].squeeze(0)

        grid_h = int(resize.grid_h)
        grid_w = int(resize.grid_w)
        grid_thw = torch.tensor([1, grid_h, grid_w], dtype=torch.long)
        num_image_tokens = int(resize.num_image_tokens)

        prompt_text, full_text = build_sft_texts(
            tokenizer=self.tokenizer,
            image_path=ex.image_path,
            prompt=self.prompt,
            caption=ex.caption,
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
            "imgid": ex.imgid,
            "filename": ex.filename,
            "image_path": ex.image_path,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "pixel_values": pixel_values,
            "image_grid_thw": grid_thw,
        }


class CaptionSFTCollator:
    def __init__(self, *, pad_token_id: int) -> None:
        self.pad_token_id = int(pad_token_id)

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, Any]:
        max_len = max(int(x["input_ids"].shape[0]) for x in batch)

        first_hw: tuple[int, int] | None = None
        first_grid: tuple[int, int, int] | None = None
        first_name = ""

        input_ids_list = []
        attention_mask_list = []
        labels_list = []
        pixel_values_list = []
        grid_list = []
        meta = []

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
            pixel_values_list.append(ex["pixel_values"])
            grid_list.append(ex["image_grid_thw"])
            meta.append({"imgid": ex.get("imgid"), "filename": ex.get("filename")})

            cur_hw = (int(ex["pixel_values"].shape[-2]), int(ex["pixel_values"].shape[-1]))
            cur_grid_tensor: torch.Tensor = ex["image_grid_thw"]
            cur_grid = tuple(int(x) for x in cur_grid_tensor.tolist())
            cur_name = str(ex.get("filename", ""))

            if first_hw is None:
                first_hw = cur_hw
                first_grid = cur_grid
                first_name = cur_name
            else:
                if cur_hw != first_hw:
                    raise ValueError(
                        "Mixed resized image shapes within one batch are not supported: "
                        f"first={first_hw} ({first_name}), current={cur_hw} ({cur_name})."
                    )
                if first_grid is not None and cur_grid != first_grid:
                    raise ValueError(
                        "Mixed image_grid_thw within one batch are not supported: "
                        f"first={first_grid} ({first_name}), current={cur_grid} ({cur_name})."
                    )

        return {
            "input_ids": torch.stack(input_ids_list, dim=0),
            "attention_mask": torch.stack(attention_mask_list, dim=0),
            "labels": torch.stack(labels_list, dim=0),
            "pixel_values": torch.stack(pixel_values_list, dim=0),
            "image_grid_thw": torch.stack(grid_list, dim=0),
            "meta": meta,
        }


def _strip_llava_image_placeholder(text: str) -> str:
    """移除 VRSBench/LLaVA 风格 prompt 里的 `<image>` 占位符。

    - 我们在 chat template 里会以“图片 + 文本”的结构提供 image，因此这里要把文本里重复的 `<image>` 去掉。
    """
    t = str(text).replace("<image>", "").strip()
    # 常见格式是 "<image>\n...."
    if t.startswith("\n"):
        t = t.lstrip("\n").strip()
    return t


class VrsbenchMultiTaskSFTDataset(Dataset):
    """VRSBench 三任务联合训练数据集（caption + refer + vqa）。

    输入数据来自 VRSBench_train.json，样本结构：
    - image: 00002_0000.png
    - conversations: [{"from":"human","value":"<image>\\n[caption] ..."}, {"from":"gpt","value":"..."}]
    """

    def __init__(
        self,
        items: list[dict[str, Any]],
        *,
        tokenizer,
        image_processor,
        dataset_root: str | Path,
        split: str = "train",
        image_size: int,
        patch_size: int,
        merge_size: int,
        smart_resize_min_pixels: int = 224 * 224,
        smart_resize_max_pixels: int = 512 * 512,
        image_token: str = "<|image_pad|>",
    ) -> None:
        self.items = items
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.dataset_root = _resolve_from_project(dataset_root)
        self.split = str(split)
        self.image_size = int(image_size)
        self.patch_size = int(patch_size)
        self.merge_size = int(merge_size)
        self.smart_resize_min_pixels = int(smart_resize_min_pixels)
        self.smart_resize_max_pixels = int(smart_resize_max_pixels)
        self.image_token = str(image_token)

        if self.patch_size <= 0 or self.merge_size <= 0:
            raise ValueError(f"Invalid patch/merge size: patch={self.patch_size}, merge={self.merge_size}")
        if self.smart_resize_min_pixels <= 0 or self.smart_resize_max_pixels < self.smart_resize_min_pixels:
            raise ValueError(
                "Invalid smart resize range: "
                f"min={self.smart_resize_min_pixels}, max={self.smart_resize_max_pixels}"
            )
        self.resize_factor = int(self.patch_size * self.merge_size)

        # VRSBench 的训练图片目录固定为 Images_train
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

    def _compute_resize_key_from_image_name(self, image_name: str) -> tuple[int, int, int, int, int]:
        image_fs_path = self._resolve_image_fs_path(image_name)
        height, width = _read_image_hw(image_fs_path)
        return _compute_resize_key_from_hw(
            height=int(height),
            width=int(width),
            patch_size=int(self.patch_size),
            merge_size=int(self.merge_size),
            smart_resize_min_pixels=int(self.smart_resize_min_pixels),
            smart_resize_max_pixels=int(self.smart_resize_max_pixels),
        )

    def get_resize_bucket_key(self, idx: int) -> tuple[int, int, int, int, int]:
        i = int(idx)
        cached = self._resize_key_cache.get(i)
        if cached is not None:
            return cached

        it = self.items[i]
        image_name = str(it.get("image", ""))
        if not image_name:
            raise ValueError(f"Empty image name at idx={i}")

        if self._image_hw_cache is None:
            self._image_hw_cache = _load_image_hw_cache(self.image_hw_cache_path)

        cached_hw = None if self._image_hw_cache is None else self._image_hw_cache.get(image_name)
        if cached_hw is not None:
            key = _compute_resize_key_from_hw(
                height=int(cached_hw[0]),
                width=int(cached_hw[1]),
                patch_size=int(self.patch_size),
                merge_size=int(self.merge_size),
                smart_resize_min_pixels=int(self.smart_resize_min_pixels),
                smart_resize_max_pixels=int(self.smart_resize_max_pixels),
            )
        else:
            key = self._compute_resize_key_from_image_name(image_name)
        self._resize_key_cache[i] = key
        return key

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        it = self.items[idx]
        image_name = str(it.get("image", ""))
        if not image_name:
            raise ValueError(f"Empty image name at idx={idx}")

        conv = it.get("conversations", [])
        if not isinstance(conv, list) or not conv:
            raise ValueError(f"Invalid conversations at idx={idx}")

        # 默认每条样本是 1 轮对话：human -> gpt。
        human_text = None
        gpt_text = None
        for m in conv:
            if not isinstance(m, dict):
                continue
            if human_text is None and str(m.get("from", "")).lower() == "human":
                human_text = str(m.get("value", ""))
                continue
            if gpt_text is None and str(m.get("from", "")).lower() == "gpt":
                gpt_text = str(m.get("value", ""))
        if human_text is None or gpt_text is None:
            raise ValueError(f"Missing human/gpt message at idx={idx}")

        prompt = _strip_llava_image_placeholder(human_text)
        answer = str(gpt_text).strip()

        # 图像：按 DINOv3 的方式做 resize/normalize
        image_fs_path = self._resolve_image_fs_path(image_name)
        # 用上下文管理器读取图片，避免长时间训练时文件句柄堆积。
        with Image.open(str(image_fs_path)) as pil_img:
            img = pil_img.convert("RGB")
        resize = compute_vision_resize(
            height=int(img.height),
            width=int(img.width),
            patch_size=int(self.patch_size),
            merge_size=int(self.merge_size),
            min_pixels=int(self.smart_resize_min_pixels),
            max_pixels=int(self.smart_resize_max_pixels),
        )
        img_inputs = self.image_processor(
            images=img,
            return_tensors="pt",
            size={"height": int(resize.resized_height), "width": int(resize.resized_width)},
        )
        pixel_values = img_inputs["pixel_values"].squeeze(0)

        grid_h = int(resize.grid_h)
        grid_w = int(resize.grid_w)
        grid_thw = torch.tensor([1, grid_h, grid_w], dtype=torch.long)
        num_image_tokens = int(resize.num_image_tokens)

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
            "image_grid_thw": grid_thw,
        }


class VrsbenchMultiTaskQwenNativeSFTDataset(Dataset):
    """VRSBench 三任务联合训练数据集（Qwen3-VL 原生视觉分支）。"""

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
        smart_resize_min_pixels: int = 224 * 224,
        smart_resize_max_pixels: int = 512 * 512,
        image_token: str = "<|image_pad|>",
    ) -> None:
        self.items = items
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.dataset_root = _resolve_from_project(dataset_root)
        self.split = str(split)
        self.patch_size = int(patch_size)
        self.merge_size = int(merge_size)
        self.smart_resize_min_pixels = int(smart_resize_min_pixels)
        self.smart_resize_max_pixels = int(smart_resize_max_pixels)
        self.image_token = str(image_token)

        if self.patch_size <= 0 or self.merge_size <= 0:
            raise ValueError(f"Invalid patch/merge size: patch={self.patch_size}, merge={self.merge_size}")
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

    def _compute_resize_key_from_image_name(self, image_name: str) -> tuple[int, int, int, int, int]:
        image_fs_path = self._resolve_image_fs_path(image_name)
        height, width = _read_image_hw(image_fs_path)
        return _compute_resize_key_from_hw(
            height=int(height),
            width=int(width),
            patch_size=int(self.patch_size),
            merge_size=int(self.merge_size),
            smart_resize_min_pixels=int(self.smart_resize_min_pixels),
            smart_resize_max_pixels=int(self.smart_resize_max_pixels),
        )

    def get_resize_bucket_key(self, idx: int) -> tuple[int, int, int, int, int]:
        i = int(idx)
        cached = self._resize_key_cache.get(i)
        if cached is not None:
            return cached

        it = self.items[i]
        image_name = str(it.get("image", ""))
        if not image_name:
            raise ValueError(f"Empty image name at idx={i}")

        if self._image_hw_cache is None:
            self._image_hw_cache = _load_image_hw_cache(self.image_hw_cache_path)

        cached_hw = None if self._image_hw_cache is None else self._image_hw_cache.get(image_name)
        if cached_hw is not None:
            key = _compute_resize_key_from_hw(
                height=int(cached_hw[0]),
                width=int(cached_hw[1]),
                patch_size=int(self.patch_size),
                merge_size=int(self.merge_size),
                smart_resize_min_pixels=int(self.smart_resize_min_pixels),
                smart_resize_max_pixels=int(self.smart_resize_max_pixels),
            )
        else:
            key = self._compute_resize_key_from_image_name(image_name)
        self._resize_key_cache[i] = key
        return key

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        it = self.items[idx]
        image_name = str(it.get("image", ""))
        if not image_name:
            raise ValueError(f"Empty image name at idx={idx}")

        conv = it.get("conversations", [])
        if not isinstance(conv, list) or not conv:
            raise ValueError(f"Invalid conversations at idx={idx}")

        human_text = None
        gpt_text = None
        for m in conv:
            if not isinstance(m, dict):
                continue
            if human_text is None and str(m.get("from", "")).lower() == "human":
                human_text = str(m.get("value", ""))
                continue
            if gpt_text is None and str(m.get("from", "")).lower() == "gpt":
                gpt_text = str(m.get("value", ""))
        if human_text is None or gpt_text is None:
            raise ValueError(f"Missing human/gpt message at idx={idx}")

        prompt = _strip_llava_image_placeholder(human_text)
        answer = str(gpt_text).strip()

        image_fs_path = self._resolve_image_fs_path(image_name)
        with Image.open(str(image_fs_path)) as pil_img:
            img = pil_img.convert("RGB")

        resize = compute_vision_resize(
            height=int(img.height),
            width=int(img.width),
            patch_size=int(self.patch_size),
            merge_size=int(self.merge_size),
            min_pixels=int(self.smart_resize_min_pixels),
            max_pixels=int(self.smart_resize_max_pixels),
        )
        img = img.resize((int(resize.resized_width), int(resize.resized_height)), Image.BICUBIC)

        # 对 Qwen3-VL 原生视觉分支，图像预处理返回展平后的 patch 表示：
        # pixel_values 形状为 [num_patches, patch_dim]，并显式给出 image_grid_thw。
        img_inputs = self.image_processor(
            images=img,
            return_tensors="pt",
            do_resize=False,
        )
        pixel_values = img_inputs["pixel_values"]
        if pixel_values.ndim == 3 and int(pixel_values.shape[0]) == 1:
            pixel_values = pixel_values.squeeze(0)
        if pixel_values.ndim != 2:
            raise ValueError(f"Unexpected pixel_values shape for Qwen native visual: {tuple(pixel_values.shape)}")

        grid_thw = img_inputs["image_grid_thw"]
        if grid_thw.ndim == 2 and int(grid_thw.shape[0]) == 1:
            grid_thw = grid_thw.squeeze(0)
        if grid_thw.ndim != 1 or int(grid_thw.shape[0]) != 3:
            raise ValueError(f"Unexpected image_grid_thw shape for Qwen native visual: {tuple(grid_thw.shape)}")
        grid_thw = grid_thw.to(dtype=torch.long)

        grid_h = int(grid_thw[1].item())
        grid_w = int(grid_thw[2].item())
        if grid_h != int(resize.grid_h) or grid_w != int(resize.grid_w):
            raise ValueError(
                "image_grid_thw mismatch with computed resize: "
                f"processor=({grid_h},{grid_w}) vs resize=({resize.grid_h},{resize.grid_w})"
            )
        num_image_tokens = int((grid_h * grid_w) // (self.merge_size**2))
        if num_image_tokens <= 0:
            raise ValueError(
                f"Invalid num_image_tokens={num_image_tokens} from grid=({grid_h},{grid_w}) merge={self.merge_size}"
            )

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
            "image_grid_thw": grid_thw,
        }


class QwenNativeSFTCollator:
    def __init__(self, *, pad_token_id: int) -> None:
        self.pad_token_id = int(pad_token_id)

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, Any]:
        max_len = max(int(x["input_ids"].shape[0]) for x in batch)

        input_ids_list = []
        attention_mask_list = []
        labels_list = []
        pixel_values_list = []
        grid_list = []
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
                    "QwenNativeSFTCollator expects pixel_values shape [num_patches, patch_dim], "
                    f"got {tuple(pixel_values.shape)}"
                )
            if patch_dim is None:
                patch_dim = int(pixel_values.shape[1])
            elif int(pixel_values.shape[1]) != int(patch_dim):
                raise ValueError(
                    f"Mixed patch_dim in one batch: first={patch_dim}, current={int(pixel_values.shape[1])}"
                )
            pixel_values_list.append(pixel_values)

            grid_thw: torch.Tensor = ex["image_grid_thw"]
            if grid_thw.ndim != 1 or int(grid_thw.shape[0]) != 3:
                raise ValueError(f"Expected image_grid_thw shape [3], got {tuple(grid_thw.shape)}")
            grid_list.append(grid_thw.to(dtype=torch.long))

        return {
            "input_ids": torch.stack(input_ids_list, dim=0),
            "attention_mask": torch.stack(attention_mask_list, dim=0),
            "labels": torch.stack(labels_list, dim=0),
            "pixel_values": torch.cat(pixel_values_list, dim=0),
            "image_grid_thw": torch.stack(grid_list, dim=0),
            "meta": meta,
        }
