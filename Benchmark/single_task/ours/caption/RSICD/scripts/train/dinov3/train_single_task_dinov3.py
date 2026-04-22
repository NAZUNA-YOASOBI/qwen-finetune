from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.distributed as dist
from accelerate import Accelerator
from peft import LoraConfig, PeftModel, get_peft_model  # type: ignore
from PIL import Image
from torch.utils.data import BatchSampler, DataLoader, Dataset, Sampler
from transformers import AutoImageProcessor, AutoProcessor, Qwen3VLForConditionalGeneration, get_scheduler

PROJECT_ROOT = Path(__file__).resolve().parents[4]
VRSBENCH_SRC_ROOT = PROJECT_ROOT / "VRSBench" / "src"
if str(VRSBENCH_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(VRSBENCH_SRC_ROOT))

from ftqwen3.device import require_cuda
from ftqwen3.dinov3_adapter import DinoV3AdapterConfig, DinoV3VisualAdapter
from ftqwen3.qwen_dinov3 import (
    assert_dino_runtime_matches_merger,
    load_merger_safetensors,
    resolve_dino_resize_config,
    save_merger_safetensors,
    torch_dtype_from_str,
)
from ftqwen3.sft import CaptionSFTCollator, build_chat_messages, expand_image_tokens
from ftqwen3.training_losses import causal_lm_sample_average_loss
from ftqwen3.vision_resize import compute_vision_resize

DATASET_CHOICES = ("rsicd", "rsvqa_hr", "geochat_single_object")
RSVQA_HR_PROMPT = "Answer the question using a single word or phrase.\nQuestion: {question}\nAnswer:"


def _project_root() -> Path:
    return PROJECT_ROOT


def _resolve_from_project(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return (_project_root() / candidate).resolve()


def _rel_to_project(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(_project_root()))
    except Exception:
        return str(path.resolve())


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    tmp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp_path.replace(path)


def _parse_int_list(raw: str, *, sep: str) -> list[int]:
    values = [item.strip() for item in str(raw).split(sep) if item.strip()]
    if not values:
        return []
    parsed_values: list[int] = []
    for item in values:
        value = int(item)
        if value <= 0:
            raise ValueError(f"Batch values must be > 0, got {value} from '{raw}'.")
        parsed_values.append(value)
    return parsed_values


def _resolve_local_batch_sizes(*, world_size: int, base_batch: int, per_rank: str, ratio: str) -> list[int]:
    if world_size <= 0:
        raise ValueError(f"Invalid world_size={world_size}")

    if str(per_rank).strip():
        sizes = _parse_int_list(str(per_rank), sep=",")
        if len(sizes) != int(world_size):
            raise ValueError(
                f"--batch-size-per-rank expects {world_size} values, got {len(sizes)} from '{per_rank}'."
            )
        return sizes

    ratios = _parse_int_list(str(ratio), sep=":")
    if not ratios:
        ratios = [1] * int(world_size)
    if len(ratios) != int(world_size):
        raise ValueError(f"--batch-size-ratio expects {world_size} values, got {len(ratios)} from '{ratio}'.")

    if int(base_batch) <= 0:
        raise ValueError(f"--batch-size must be > 0, got {base_batch}.")

    base_ratio = int(ratios[0])
    sizes: list[int] = []
    for ratio_value in ratios:
        scaled = int(round(float(base_batch) * float(ratio_value) / float(base_ratio)))
        sizes.append(max(1, scaled))
    return sizes


def _resolve_language_lora_targets(base_model, target_leaf_names: list[str]) -> list[str]:
    leaf_set = {str(item).strip() for item in target_leaf_names if str(item).strip()}
    if not leaf_set:
        raise ValueError("Empty LoRA target module list.")

    matched: list[str] = []
    for name, _module in base_model.named_modules():
        if not name:
            continue
        leaf = name.rsplit(".", 1)[-1]
        if leaf not in leaf_set:
            continue
        if ".language_model." not in f".{name}.":
            continue
        matched.append(name)

    matched = sorted(set(matched))
    if not matched:
        wanted = ", ".join(sorted(leaf_set))
        raise RuntimeError(f"No language_model LoRA targets matched. wanted=[{wanted}]")
    return matched


def _assert_no_visual_lora_trainables(model) -> None:
    bad: list[str] = []
    for name, parameter in model.named_parameters():
        if "lora_" not in name:
            continue
        if ".visual." in f".{name}." and bool(getattr(parameter, "requires_grad", False)):
            bad.append(name)
            if len(bad) >= 8:
                break
    if bad:
        raise RuntimeError("Found trainable visual LoRA params (expected LLM-only LoRA): " + ", ".join(bad))


def _assert_expected_trainables_layout(model) -> None:
    bad: list[str] = []
    has_visual = False
    has_lora = False
    for name, parameter in model.named_parameters():
        if not bool(getattr(parameter, "requires_grad", False)):
            continue
        scoped = f".{name}."
        is_visual = (
            ".visual.merger." in scoped
            or ".visual.deepstack_merger_list." in scoped
            or ".visual.input_proj." in scoped
        )
        is_lora = "lora_" in name
        if is_visual:
            has_visual = True
        if is_lora:
            has_lora = True
        if not (is_visual or is_lora):
            bad.append(name)
            if len(bad) >= 8:
                break
    if bad:
        raise RuntimeError(
            "Found unexpected trainable params outside merger/deepstack/input_proj + LoRA: " + ", ".join(bad)
        )
    if not has_visual:
        raise RuntimeError("No trainable visual merger params found.")
    if not has_lora:
        raise RuntimeError("No trainable LoRA params found.")


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


def _read_image_hw(image_fs_path: Path) -> tuple[int, int]:
    with Image.open(str(image_fs_path)) as pil_image:
        return int(pil_image.height), int(pil_image.width)


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


def _sample_rows_without_replacement(
    rows: list[dict[str, Any]],
    *,
    sample_ratio: float,
    seed: int,
    key_field: str,
) -> list[dict[str, Any]]:
    if float(sample_ratio) <= 0 or float(sample_ratio) > 1:
        raise ValueError(f"sample_ratio must be in (0, 1], got {sample_ratio}")
    if float(sample_ratio) >= 1.0:
        return sorted(rows, key=lambda row: int(row[key_field]))

    target_count = max(1, int(math.floor(float(len(rows)) * float(sample_ratio))))
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    perm = torch.randperm(len(rows), generator=generator).tolist()
    selected = [rows[int(index)] for index in perm[:target_count]]
    return sorted(selected, key=lambda row: int(row[key_field]))


def _sample_rows_by_group_without_replacement(
    rows: list[dict[str, Any]],
    *,
    sample_ratio: float,
    seed: int,
    key_field: str,
    group_field: str,
) -> list[dict[str, Any]]:
    if float(sample_ratio) <= 0 or float(sample_ratio) > 1:
        raise ValueError(f"sample_ratio must be in (0, 1], got {sample_ratio}")
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        group_name = str(row[group_field]).strip()
        groups.setdefault(group_name, []).append(row)
    if not groups:
        return []

    selected_rows: list[dict[str, Any]] = []
    for group_index, group_name in enumerate(sorted(groups.keys())):
        group_rows = sorted(groups[group_name], key=lambda row: int(row[key_field]))
        if float(sample_ratio) >= 1.0:
            selected_group_rows = group_rows
        else:
            target_count = max(1, int(math.floor(float(len(group_rows)) * float(sample_ratio))))
            generator = torch.Generator()
            generator.manual_seed(int(seed) + int(group_index) * 1000)
            perm = torch.randperm(len(group_rows), generator=generator).tolist()
            selected_group_rows = [group_rows[int(index)] for index in perm[:target_count]]
            selected_group_rows = sorted(selected_group_rows, key=lambda row: int(row[key_field]))
        selected_rows.extend(selected_group_rows)
    return sorted(selected_rows, key=lambda row: int(row[key_field]))


def _rsicd_data_root() -> Path:
    return (
        PROJECT_ROOT
        / "VRSBench"
        / "benchmark"
        / "single_task"
        / "datasets"
        / "caption"
        / "RSICD"
        / "official"
        / "raw"
    )


def _rsvqa_hr_data_root() -> Path:
    return (
        PROJECT_ROOT
        / "VRSBench"
        / "benchmark"
        / "single_task"
        / "datasets"
        / "vqa"
        / "RSVQA-HR"
        / "official"
        / "raw"
    )


def _geochat_single_object_data_root() -> Path:
    return (
        PROJECT_ROOT
        / "VRSBench"
        / "benchmark"
        / "single_task"
        / "datasets"
        / "grounding"
        / "GeoChat_single_object"
        / "benchmark"
        / "data"
    )


def _default_output_dir(dataset_name: str) -> Path:
    return (
        PROJECT_ROOT
        / "VRSBench"
        / "checkpoints"
        / "single_task"
        / "dinov3"
        / f"{dataset_name}_merger_lora_sampleavg_wd001"
    )


def _build_rsicd_rows(*, json_path: Path, image_root: Path) -> list[dict[str, Any]]:
    payload = _read_json(json_path)
    if not isinstance(payload, list) or not payload:
        raise ValueError(f"Invalid RSICD split json: {json_path}")
    rows: list[dict[str, Any]] = []
    for item in payload:
        image_name = str(item.get("image_id", "")).strip()
        prompt = str(item.get("question", "")).strip()
        answer = str(item.get("ground_truth", "")).strip()
        sample_id = int(item.get("question_id", len(rows)))
        if not image_name or not prompt or not answer:
            continue
        rows.append(
            {
                "sample_id": int(sample_id),
                "filename": image_name,
                "image_path": str((image_root / image_name).resolve()),
                "prompt": prompt,
                "answer": answer,
                "task_name": "caption",
            }
        )
    if not rows:
        raise RuntimeError(f"No valid RSICD rows loaded from {json_path}")
    return rows


def _resolve_rsvqa_hr_image_path(*, image_root: Path, image_id: int) -> Path:
    for ext in ("tif", "png", "jpg", "jpeg"):
        candidate = image_root / f"{image_id}.{ext}"
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(f"Missing RSVQA-HR image for image_id={image_id}: {image_root}")


def _build_rsvqa_hr_rows(*, questions_json: Path, answers_json: Path, images_json: Path, image_root: Path) -> list[dict[str, Any]]:
    questions_payload = _read_json(questions_json)
    answers_payload = _read_json(answers_json)
    images_payload = _read_json(images_json)

    questions = [item for item in questions_payload.get("questions", []) if item.get("active") is True]
    answers = [item for item in answers_payload.get("answers", []) if item.get("active") is True]
    images = [item for item in images_payload.get("images", []) if item.get("active") is True]

    answers_by_qid = {int(item["question_id"]): item for item in answers}
    images_by_id = {int(item["id"]): item for item in images}

    rows: list[dict[str, Any]] = []
    for item in questions:
        question_id = int(item["id"])
        image_id = int(item["img_id"])
        answer = answers_by_qid.get(question_id)
        image_meta = images_by_id.get(image_id)
        if answer is None or image_meta is None:
            continue
        question = str(item.get("question", "")).strip()
        answer_text = str(answer.get("answer", "")).strip()
        if not question or not answer_text:
            continue
        image_path = _resolve_rsvqa_hr_image_path(image_root=image_root, image_id=image_id)
        rows.append(
            {
                "sample_id": int(question_id),
                "filename": image_path.name,
                "image_path": str(image_path),
                "prompt": RSVQA_HR_PROMPT.format(question=question),
                "answer": answer_text,
                "task_name": "vqa",
                "question_type": str(item.get("type", "")).strip(),
                "image_id": int(image_id),
                "question": question,
            }
        )
    if not rows:
        raise RuntimeError(
            "No valid RSVQA-HR rows loaded from "
            f"{questions_json}, {answers_json}, {images_json}"
        )
    return rows


def _build_geochat_single_object_rows(*, json_path: Path) -> list[dict[str, Any]]:
    payload = _read_json(json_path)
    if not isinstance(payload, list) or not payload:
        raise ValueError(f"Invalid GeoChat single-object split json: {json_path}")

    rows: list[dict[str, Any]] = []
    for item in payload:
        sample_id = int(item.get("question_id", len(rows)))
        image_name = str(item.get("image_id", "")).strip()
        image_rel_path = str(item.get("image_rel_path", "")).strip()
        prompt = str(item.get("question", "")).strip()
        answer = str(item.get("ground_truth", "")).strip()
        if not image_name or not image_rel_path or not prompt or not answer:
            continue
        rows.append(
            {
                "sample_id": int(sample_id),
                "filename": image_name,
                "image_path": str(_resolve_from_project(image_rel_path)),
                "prompt": prompt,
                "answer": answer,
                "task_name": str(item.get("task", "refer")).strip() or "refer",
                "source_dataset": str(item.get("source_dataset", "")).strip(),
            }
        )
    if not rows:
        raise RuntimeError(f"No valid GeoChat single-object rows loaded from {json_path}")
    return rows


def _split_rows_into_train_val(
    rows: list[dict[str, Any]],
    *,
    val_ratio: float,
    seed: int,
    key_field: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if float(val_ratio) <= 0 or float(val_ratio) >= 1:
        raise ValueError(f"val_ratio must be in (0, 1), got {val_ratio}")
    if len(rows) < 2:
        raise ValueError(f"Need at least 2 rows to split train/val, got {len(rows)}")

    sorted_rows = sorted(rows, key=lambda row: int(row[key_field]))
    target_val_count = max(1, int(math.floor(float(len(sorted_rows)) * float(val_ratio))))
    target_val_count = min(target_val_count, len(sorted_rows) - 1)

    generator = torch.Generator()
    generator.manual_seed(int(seed))
    perm = torch.randperm(len(sorted_rows), generator=generator).tolist()
    val_index_set = {int(index) for index in perm[:target_val_count]}

    train_rows: list[dict[str, Any]] = []
    val_rows: list[dict[str, Any]] = []
    for row_index, row in enumerate(sorted_rows):
        if int(row_index) in val_index_set:
            val_rows.append(row)
        else:
            train_rows.append(row)
    return train_rows, val_rows


def _load_dataset_splits(
    *,
    dataset_name: str,
    seed: int,
    rsvqa_train_ratio: float,
    rsvqa_val_ratio: float,
    geochat_val_ratio: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    if dataset_name == "rsicd":
        data_root = _rsicd_data_root()
        image_root = data_root / "RSICD_images"
        train_rows = _build_rsicd_rows(json_path=data_root / "dataset_train.json", image_root=image_root)
        val_rows = _build_rsicd_rows(json_path=data_root / "dataset_val.json", image_root=image_root)
        meta = {
            "dataset_name": "rsicd",
            "task_name": "caption",
            "dataset_label": "RSICD",
            "data_root": _rel_to_project(data_root),
            "image_root": _rel_to_project(image_root),
            "train_source": _rel_to_project(data_root / "dataset_train.json"),
            "val_source": _rel_to_project(data_root / "dataset_val.json"),
            "train_full_count": int(len(train_rows)),
            "val_full_count": int(len(val_rows)),
            "train_selected_count": int(len(train_rows)),
            "val_selected_count": int(len(val_rows)),
            "train_sample_ratio": 1.0,
            "val_sample_ratio": 1.0,
            "cache_dir": _rel_to_project(data_root / "cache"),
        }
        return train_rows, val_rows, meta

    if dataset_name == "rsvqa_hr":
        data_root = _rsvqa_hr_data_root()
        image_root = data_root / "Data"

        full_train_rows = _build_rsvqa_hr_rows(
            questions_json=data_root / "USGS_split_train_questions.json",
            answers_json=data_root / "USGS_split_train_answers.json",
            images_json=data_root / "USGS_split_train_images.json",
            image_root=image_root,
        )
        full_val_rows = _build_rsvqa_hr_rows(
            questions_json=data_root / "USGS_split_val_questions.json",
            answers_json=data_root / "USGS_split_val_answers.json",
            images_json=data_root / "USGS_split_val_images.json",
            image_root=image_root,
        )
        train_rows = _sample_rows_by_group_without_replacement(
            full_train_rows,
            sample_ratio=float(rsvqa_train_ratio),
            seed=int(seed) + 100,
            key_field="sample_id",
            group_field="question_type",
        )
        val_rows = _sample_rows_by_group_without_replacement(
            full_val_rows,
            sample_ratio=float(rsvqa_val_ratio),
            seed=int(seed) + 200,
            key_field="sample_id",
            group_field="question_type",
        )
        meta = {
            "dataset_name": "rsvqa_hr",
            "task_name": "vqa",
            "dataset_label": "RSVQA-HR",
            "data_root": _rel_to_project(data_root),
            "image_root": _rel_to_project(image_root),
            "train_source": _rel_to_project(data_root / "USGS_split_train_questions.json"),
            "val_source": _rel_to_project(data_root / "USGS_split_val_questions.json"),
            "train_full_count": int(len(full_train_rows)),
            "val_full_count": int(len(full_val_rows)),
            "train_selected_count": int(len(train_rows)),
            "val_selected_count": int(len(val_rows)),
            "train_sample_ratio": float(rsvqa_train_ratio),
            "val_sample_ratio": float(rsvqa_val_ratio),
            "train_type_counts_full": dict(sorted(Counter(str(row["question_type"]) for row in full_train_rows).items())),
            "val_type_counts_full": dict(sorted(Counter(str(row["question_type"]) for row in full_val_rows).items())),
            "train_type_counts_selected": dict(sorted(Counter(str(row["question_type"]) for row in train_rows).items())),
            "val_type_counts_selected": dict(sorted(Counter(str(row["question_type"]) for row in val_rows).items())),
            "cache_dir": _rel_to_project(data_root / "cache"),
        }
        return train_rows, val_rows, meta

    if dataset_name == "geochat_single_object":
        data_root = _geochat_single_object_data_root()
        full_train_rows = _build_geochat_single_object_rows(json_path=data_root / "train_refer_single_object.json")
        train_rows, val_rows = _split_rows_into_train_val(
            full_train_rows,
            val_ratio=float(geochat_val_ratio),
            seed=int(seed) + 300,
            key_field="sample_id",
        )
        meta = {
            "dataset_name": "geochat_single_object",
            "task_name": "refer",
            "dataset_label": "GeoChat Single-Object Refer",
            "data_root": _rel_to_project(data_root),
            "image_root": _rel_to_project(PROJECT_ROOT / "GeoChat-Bench" / "dataset" / "raw" / "GeoChat_Instruct" / "images"),
            "train_source": _rel_to_project(data_root / "train_refer_single_object.json"),
            "val_source": f"{_rel_to_project(data_root / 'train_refer_single_object.json')} (seed split)",
            "test_source": _rel_to_project(data_root / "test_single_object.json"),
            "train_full_count": int(len(full_train_rows)),
            "val_full_count": int(len(full_train_rows)),
            "train_selected_count": int(len(train_rows)),
            "val_selected_count": int(len(val_rows)),
            "train_sample_ratio": float(1.0 - geochat_val_ratio),
            "val_sample_ratio": float(geochat_val_ratio),
            "cache_dir": _rel_to_project(data_root / "cache"),
        }
        return train_rows, val_rows, meta

    raise ValueError(f"Unsupported dataset: {dataset_name}")


class SingleTaskDinoSFTDataset(Dataset):
    def __init__(
        self,
        rows: list[dict[str, Any]],
        *,
        tokenizer,
        image_processor,
        cache_dir: Path,
        cache_name: str,
        patch_size: int,
        merge_size: int,
        smart_resize_min_pixels: int,
        smart_resize_max_pixels: int,
        image_token: str = "<|image_pad|>",
    ) -> None:
        self.rows = rows
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.cache_dir = cache_dir
        self.cache_name = str(cache_name)
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

        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.image_hw_cache_path = self.cache_dir / f"{self.cache_name}_image_hw.json"
        self._image_hw_cache: dict[str, tuple[int, int]] | None = None
        self._resize_key_cache: dict[int, tuple[int, int, int, int, int]] = {}

    def ensure_image_hw_cache(self, *, build_if_missing: bool) -> None:
        if self._image_hw_cache is not None:
            return

        cache = _load_image_hw_cache(self.image_hw_cache_path)
        if cache is not None:
            self._image_hw_cache = cache
            return
        if not build_if_missing:
            return

        unique_paths = sorted({str(row["image_path"]) for row in self.rows})
        images_payload: dict[str, dict[str, int]] = {}
        for image_path in unique_paths:
            height, width = _read_image_hw(Path(image_path))
            images_payload[image_path] = {"height": int(height), "width": int(width)}

        payload = {
            "version": 1,
            "cache_name": self.cache_name,
            "num_images": int(len(images_payload)),
            "images": images_payload,
        }
        _write_json_atomic(self.image_hw_cache_path, payload)
        self._image_hw_cache = _load_image_hw_cache(self.image_hw_cache_path)

    def get_resize_bucket_key(self, idx: int) -> tuple[int, int, int, int, int]:
        index = int(idx)
        cached = self._resize_key_cache.get(index)
        if cached is not None:
            return cached

        image_path = str(self.rows[index]["image_path"])
        if self._image_hw_cache is None:
            self._image_hw_cache = _load_image_hw_cache(self.image_hw_cache_path)
        cached_hw = None if self._image_hw_cache is None else self._image_hw_cache.get(image_path)
        if cached_hw is None:
            height, width = _read_image_hw(Path(image_path))
        else:
            height, width = int(cached_hw[0]), int(cached_hw[1])
        key = _compute_resize_key_from_hw(
            height=int(height),
            width=int(width),
            patch_size=int(self.patch_size),
            merge_size=int(self.merge_size),
            smart_resize_min_pixels=int(self.smart_resize_min_pixels),
            smart_resize_max_pixels=int(self.smart_resize_max_pixels),
        )
        self._resize_key_cache[index] = key
        return key

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.rows[int(idx)]
        image_fs_path = Path(str(row["image_path"])).resolve()
        prompt = str(row["prompt"])
        answer = str(row["answer"])

        with Image.open(str(image_fs_path)) as pil_image:
            image = pil_image.convert("RGB")

        resize = compute_vision_resize(
            height=int(image.height),
            width=int(image.width),
            patch_size=int(self.patch_size),
            merge_size=int(self.merge_size),
            min_pixels=int(self.smart_resize_min_pixels),
            max_pixels=int(self.smart_resize_max_pixels),
        )
        image_inputs = self.image_processor(
            images=image,
            return_tensors="pt",
            size={"height": int(resize.resized_height), "width": int(resize.resized_width)},
        )
        pixel_values = image_inputs["pixel_values"].squeeze(0)
        grid_thw = torch.tensor([1, int(resize.grid_h), int(resize.grid_w)], dtype=torch.long)

        prompt_messages = build_chat_messages(image_path=str(image_fs_path), prompt=prompt, caption=None)
        full_messages = build_chat_messages(image_path=str(image_fs_path), prompt=prompt, caption=answer)
        prompt_text = self.tokenizer.apply_chat_template(prompt_messages, tokenize=False, add_generation_prompt=True)
        full_text = self.tokenizer.apply_chat_template(full_messages, tokenize=False, add_generation_prompt=False)
        prompt_text = expand_image_tokens(
            prompt_text,
            image_token=self.image_token,
            num_image_tokens=int(resize.num_image_tokens),
        )
        full_text = expand_image_tokens(
            full_text,
            image_token=self.image_token,
            num_image_tokens=int(resize.num_image_tokens),
        )

        prompt_tok = self.tokenizer(prompt_text, add_special_tokens=False, return_tensors="pt")
        full_tok = self.tokenizer(full_text, add_special_tokens=False, return_tensors="pt")
        input_ids = full_tok["input_ids"].squeeze(0)
        attention_mask = full_tok["attention_mask"].squeeze(0)
        labels = input_ids.clone()
        prompt_len = int(prompt_tok["input_ids"].shape[1])
        labels[:prompt_len] = -100

        return {
            "imgid": int(row["sample_id"]),
            "filename": str(row["filename"]),
            "image_path": str(image_fs_path),
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "pixel_values": pixel_values,
            "image_grid_thw": grid_thw,
        }


def _enable_visual_trainables(target_model: torch.nn.Module) -> None:
    visual = target_model.model.visual
    for parameter in visual.merger.parameters():
        parameter.requires_grad = True
    if getattr(visual, "deepstack_merger_list", None) is not None:
        for parameter in visual.deepstack_merger_list.parameters():
            parameter.requires_grad = True
    if getattr(visual, "input_proj", None) is not None:
        for parameter in visual.input_proj.parameters():
            parameter.requires_grad = True


def _move_batch_to_device(batch: dict[str, object], *, device: torch.device) -> dict[str, object]:
    out: dict[str, object] = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            out[key] = value.to(device=device, non_blocking=True)
        else:
            out[key] = value
    return out


def preallocate_cuda_cache(*, device: torch.device, keep_free_gb: float | None) -> dict[str, int | float | bool | None]:
    if keep_free_gb is None:
        return {
            "enabled": False,
            "keep_free_gb": None,
            "free_before_bytes": 0,
            "reserved_bytes": 0,
            "free_after_bytes": 0,
        }

    if str(getattr(device, "type", "")) != "cuda":
        return {
            "enabled": False,
            "keep_free_gb": float(keep_free_gb),
            "free_before_bytes": 0,
            "reserved_bytes": 0,
            "free_after_bytes": 0,
        }

    keep_free_bytes = max(0, int(float(keep_free_gb) * (1024**3)))
    free_before_bytes, _total_bytes = torch.cuda.mem_get_info(device=device)
    target_reserve_bytes = max(0, int(free_before_bytes) - int(keep_free_bytes))
    if target_reserve_bytes <= 0:
        free_after_bytes, _ = torch.cuda.mem_get_info(device=device)
        return {
            "enabled": False,
            "keep_free_gb": float(keep_free_gb),
            "free_before_bytes": int(free_before_bytes),
            "reserved_bytes": 0,
            "free_after_bytes": int(free_after_bytes),
        }

    blocks: list[torch.Tensor] = []
    reserved_bytes = 0
    remaining_bytes = int(target_reserve_bytes)
    chunk_bytes = 512 * 1024 * 1024
    min_chunk_bytes = 16 * 1024 * 1024

    while remaining_bytes >= min_chunk_bytes:
        next_bytes = min(int(chunk_bytes), int(remaining_bytes))
        allocated = False
        while next_bytes >= min_chunk_bytes:
            try:
                blocks.append(torch.empty(int(next_bytes), dtype=torch.uint8, device=device))
                reserved_bytes += int(next_bytes)
                remaining_bytes -= int(next_bytes)
                allocated = True
                break
            except torch.OutOfMemoryError:
                next_bytes = int(next_bytes // 2)
                next_bytes = int((next_bytes // min_chunk_bytes) * min_chunk_bytes)
        if not allocated:
            break

    del blocks
    free_after_bytes, _ = torch.cuda.mem_get_info(device=device)
    return {
        "enabled": bool(reserved_bytes > 0),
        "keep_free_gb": float(keep_free_gb),
        "free_before_bytes": int(free_before_bytes),
        "reserved_bytes": int(reserved_bytes),
        "free_after_bytes": int(free_after_bytes),
    }


class ProportionalDistributedSampler(Sampler[int]):
    def __init__(
        self,
        *,
        dataset_size: int,
        local_batch_sizes: list[int],
        rank: int,
        seed: int,
        grad_accum: int,
        sample_bucket_keys: list[tuple[int, int, int, int, int]],
    ) -> None:
        self.dataset_size = int(dataset_size)
        self.local_batch_sizes = [int(x) for x in local_batch_sizes]
        self.rank = int(rank)
        self.seed = int(seed)
        self.grad_accum = int(grad_accum)
        self.epoch = 0

        if self.dataset_size <= 0:
            raise ValueError(f"Invalid dataset_size={self.dataset_size}.")
        if self.rank < 0 or self.rank >= len(self.local_batch_sizes):
            raise ValueError(f"Invalid rank={self.rank} for local_batch_sizes={self.local_batch_sizes}.")
        if self.grad_accum <= 0:
            raise ValueError(f"Invalid grad_accum={self.grad_accum}.")
        if len(sample_bucket_keys) != self.dataset_size:
            raise ValueError(
                "sample_bucket_keys length mismatch: "
                f"expect={self.dataset_size}, got={len(sample_bucket_keys)}"
            )

        self.global_micro_batch = int(sum(self.local_batch_sizes))
        if self.global_micro_batch <= 0:
            raise ValueError(f"Invalid global_micro_batch={self.global_micro_batch}.")

        self.global_batch = int(self.global_micro_batch * self.grad_accum)
        self.local_batch_size = int(self.local_batch_sizes[self.rank])
        self.local_samples_per_update = int(self.local_batch_size * self.grad_accum)
        self.rank_offset = int(sum(self.local_batch_sizes[: self.rank]))

        bucket_to_indices: dict[tuple[int, int, int, int, int], list[int]] = {}
        for idx, key in enumerate(sample_bucket_keys):
            bucket_key = tuple(int(x) for x in key)
            bucket_to_indices.setdefault(bucket_key, []).append(int(idx))
        if not bucket_to_indices:
            raise RuntimeError("No bucket keys found for sampler.")

        self.bucket_to_indices = {key: tuple(indices) for key, indices in bucket_to_indices.items()}
        self.bucket_keys = sorted(self.bucket_to_indices.keys())
        self.bucket_sizes = {key: len(indices) for key, indices in self.bucket_to_indices.items()}
        self.bucket_used = {
            key: (len(indices) // self.global_batch) * self.global_batch
            for key, indices in self.bucket_to_indices.items()
        }
        self.bucket_dropped = {key: int(self.bucket_sizes[key] - self.bucket_used[key]) for key in self.bucket_keys}
        self.steps_per_epoch = int(sum(self.bucket_used[key] for key in self.bucket_keys) // self.global_batch)
        if self.steps_per_epoch <= 0:
            raise RuntimeError(
                "Dataset too small after per-bucket drop for one optimizer step. "
                f"dataset_size={self.dataset_size}, global_batch={self.global_batch}."
            )

        self.total_used_samples = int(self.steps_per_epoch * self.global_batch)
        self.total_dropped_samples = int(self.dataset_size - self.total_used_samples)
        self.local_num_samples = int(self.steps_per_epoch * self.local_samples_per_update)

    def __len__(self) -> int:
        return int(self.local_num_samples)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self):
        generator = torch.Generator()
        generator.manual_seed(int(self.seed) + int(self.epoch))

        global_update_groups: list[list[int]] = []
        for key in self.bucket_keys:
            src = self.bucket_to_indices[key]
            used = int(self.bucket_used[key])
            if used <= 0:
                continue
            order = torch.randperm(len(src), generator=generator).tolist()
            shuffled = [int(src[i]) for i in order]
            for start in range(0, used, self.global_batch):
                global_update_groups.append(shuffled[start : start + self.global_batch])

        if len(global_update_groups) != int(self.steps_per_epoch):
            raise RuntimeError(
                f"Update group count mismatch: expect={self.steps_per_epoch}, got={len(global_update_groups)}"
            )

        if len(global_update_groups) > 1:
            order = torch.randperm(len(global_update_groups), generator=generator).tolist()
            global_update_groups = [global_update_groups[i] for i in order]

        local_indices: list[int] = []
        for update_group in global_update_groups:
            if len(update_group) != int(self.global_batch):
                raise RuntimeError(
                    f"Global batch size mismatch: expect={self.global_batch}, got={len(update_group)}"
                )

            for micro_idx in range(int(self.grad_accum)):
                start = int(micro_idx * self.global_micro_batch)
                end = int(start + self.global_micro_batch)
                micro_group = update_group[start:end]
                if len(micro_group) != int(self.global_micro_batch):
                    raise RuntimeError(
                        f"Micro group size mismatch: expect={self.global_micro_batch}, got={len(micro_group)}"
                    )
                local_start = int(self.rank_offset)
                local_end = int(local_start + self.local_batch_size)
                local_chunk = micro_group[local_start:local_end]
                if len(local_chunk) != int(self.local_batch_size):
                    raise RuntimeError(
                        f"Local chunk size mismatch: expect={self.local_batch_size}, got={len(local_chunk)}"
                    )
                local_indices.extend(int(x) for x in local_chunk)

        if len(local_indices) != int(self.local_num_samples):
            raise RuntimeError(
                f"Local sampled size mismatch: expect={self.local_num_samples}, got={len(local_indices)}"
            )
        return iter(local_indices)


class SequentialBucketBatchSampler(BatchSampler):
    def __init__(
        self,
        *,
        local_batch_size: int,
        process_index: int,
        world_size: int,
        sample_bucket_keys: list[tuple[int, int, int, int, int]],
    ) -> None:
        self.local_batch_size = int(local_batch_size)
        self.process_index = int(process_index)
        self.world_size = int(world_size)
        if self.local_batch_size <= 0:
            raise ValueError(f"Invalid local_batch_size: {self.local_batch_size}")
        if self.world_size <= 0:
            raise ValueError(f"Invalid world_size: {self.world_size}")
        if self.process_index < 0 or self.process_index >= self.world_size:
            raise ValueError(f"Invalid process_index={self.process_index} for world_size={self.world_size}")

        bucket_to_indices: dict[tuple[int, int, int, int, int], list[int]] = {}
        for index, key in enumerate(sample_bucket_keys):
            bucket_key = tuple(int(x) for x in key)
            bucket_to_indices.setdefault(bucket_key, []).append(int(index))
        if not bucket_to_indices:
            raise RuntimeError("No bucket keys found for eval sampler.")
        self.bucket_keys = sorted(bucket_to_indices.keys())
        self.bucket_to_local_indices = {
            key: indices[self.process_index :: self.world_size]
            for key, indices in bucket_to_indices.items()
        }
        self.total_batches = int(
            sum(
                math.ceil(len(indices) / self.local_batch_size)
                for indices in self.bucket_to_local_indices.values()
                if indices
            )
        )

    def __len__(self) -> int:
        return self.total_batches

    def __iter__(self):
        for key in self.bucket_keys:
            local_indices = self.bucket_to_local_indices[key]
            for start in range(0, len(local_indices), self.local_batch_size):
                yield local_indices[start : start + self.local_batch_size]


def _dist_ready() -> bool:
    return dist.is_available() and dist.is_initialized()


class SingleTaskLossPlotter:
    def __init__(self, *, output_dir: Path, title: str):
        self.output_dir = output_dir.resolve()
        self.title = str(title)
        self.history_path = self.output_dir / "epoch_loss_history.json"
        self.overall_plot_path = self.output_dir / "epoch_loss_overall.svg"
        self.history = self._load_history()

    def _load_history(self) -> dict[str, dict[str, dict[str, float]]]:
        history = {"train": {"overall": {}}, "eval": {"overall": {}}}
        if not self.history_path.is_file():
            return history
        payload = json.loads(self.history_path.read_text(encoding="utf-8"))
        for mode in ("train", "eval"):
            source = payload.get(mode, {}).get("overall", {})
            history[mode]["overall"] = {str(k): float(v) for k, v in source.items()}
        return history

    def record_epoch(
        self,
        *,
        mode: str,
        epoch: int,
        local_loss_sum: torch.Tensor,
        local_sample_count: torch.Tensor,
    ) -> float | None:
        pair = torch.stack(
            (
                local_loss_sum.detach().to(dtype=torch.float64).reshape(()),
                local_sample_count.detach().to(dtype=torch.float64).reshape(()),
            )
        )
        if _dist_ready():
            dist.all_reduce(pair, op=dist.ReduceOp.SUM)
        total_loss_sum = float(pair[0].item())
        total_sample_count = float(pair[1].item())
        if total_sample_count <= 0:
            return None
        epoch_loss = float(total_loss_sum / total_sample_count)
        self.history[mode]["overall"][str(int(epoch))] = float(epoch_loss)
        return epoch_loss

    def write_outputs(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.history_path.write_text(json.dumps(self.history, ensure_ascii=False, indent=2), encoding="utf-8")
        self._plot_overall()

    def _epoch_values(self, mode: str) -> list[tuple[int, float]]:
        records = self.history.get(mode, {}).get("overall", {})
        return sorted((int(epoch), float(loss)) for epoch, loss in records.items())

    def _plot_overall(self) -> None:
        train_points = self._epoch_values("train")
        eval_points = self._epoch_values("eval")
        plt.figure(figsize=(8, 5))
        if train_points:
            plt.plot([x for x, _ in train_points], [y for _, y in train_points], marker="o", linewidth=2, label="train")
        if eval_points:
            plt.plot([x for x, _ in eval_points], [y for _, y in eval_points], marker="s", linewidth=2, label="val")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title(self.title)
        plt.grid(True, linestyle="--", alpha=0.35)
        handles, _labels = plt.gca().get_legend_handles_labels()
        if handles:
            plt.legend()
        plt.tight_layout()
        plt.savefig(self.overall_plot_path, dpi=200)
        plt.close()


def _save_training_checkpoint(
    *,
    accelerator,
    model,
    optimizer,
    lr_scheduler,
    out_dir: Path,
    checkpoint_name: str,
    global_step: int,
    epoch: int,
    run_meta: dict[str, Any],
    adapter_cfg: DinoV3AdapterConfig,
    dinov3_dir: Path,
    dataset_meta: dict[str, Any],
) -> None:
    if not bool(getattr(accelerator, "is_main_process", False)):
        return

    checkpoint_dir = out_dir / str(checkpoint_name)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    model_to_save = accelerator.unwrap_model(model)
    base_to_save = model_to_save.get_base_model()
    model_to_save.save_pretrained(str(checkpoint_dir / "lora"))
    torch.save(optimizer.state_dict(), str(checkpoint_dir / "optimizer.pt"))
    torch.save(lr_scheduler.state_dict(), str(checkpoint_dir / "scheduler.pt"))

    merger_extra = {
        "step": int(global_step),
        "epoch": int(epoch),
        "run": run_meta,
        "task_order": [str(dataset_meta["task_name"])],
        "dataset": dict(dataset_meta),
        "adapter": {
            "dinov3_dir": _rel_to_project(dinov3_dir),
            "image_size": int(adapter_cfg.image_size),
            "merge_size": int(adapter_cfg.merge_size),
            "deepstack_visual_indexes": list(adapter_cfg.deepstack_visual_indexes),
        },
    }
    save_merger_safetensors(
        base_to_save,
        checkpoint_dir / "merger.safetensors",
        extra=merger_extra,
    )


def _run_validation(
    *,
    accelerator,
    model,
    val_loader: DataLoader,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    model.eval()
    loss_sum = torch.zeros((), dtype=torch.float64, device=device)
    sample_count = torch.zeros((), dtype=torch.float64, device=device)
    with torch.no_grad():
        for batch in val_loader:
            batch.pop("meta", None)
            batch = _move_batch_to_device(batch, device=device)
            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"]
            pixel_values = batch["pixel_values"]
            image_grid_thw = batch["image_grid_thw"]
            labels = batch["labels"]
            with accelerator.autocast():
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    pixel_values=pixel_values,
                    image_grid_thw=image_grid_thw,
                )
            batch_size = int(labels.shape[0])
            batch_mean_loss = causal_lm_sample_average_loss(outputs.logits, labels)
            loss_sum += batch_mean_loss.detach().to(dtype=torch.float64) * float(batch_size)
            sample_count += float(batch_size)
    model.train()
    return loss_sum, sample_count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Single-task DINOv3 SFT for RSICD / RSVQA-HR with sample-average loss."
    )
    parser.add_argument("--dataset", type=str, required=True, choices=DATASET_CHOICES)
    parser.add_argument("--qwen-model-dir", type=str, default="VRSBench/models/Qwen3-VL-8B-Instruct")
    parser.add_argument(
        "--dinov3-dir",
        type=str,
        default="VRSBench/models/dinov3/dinov3-vitl16-pretrain-sat493m",
    )
    parser.add_argument("--output-dir", type=str, default="")
    parser.add_argument("--init-merger", type=str, default="", help="Merger safetensors to init from.")
    parser.add_argument("--resume-lora", type=str, default="", help="LoRA dir to resume from.")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--smart-resize-min-pixels", type=int, default=224 * 224)
    parser.add_argument("--smart-resize-max-pixels", type=int, default=512 * 512)
    parser.add_argument("--epochs", type=float, default=5.0)
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--batch-size-per-rank", type=str, default="")
    parser.add_argument("--batch-size-ratio", type=str, default="1:1")
    parser.add_argument("--grad-accum", type=int, default=2)
    parser.add_argument("--target-global-batch", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--mixed-precision", type=str, default="bf16", choices=["no", "fp16", "bf16"])
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument(
        "--cuda-reserve-free-gb",
        type=float,
        default=None,
        help="Pre-allocate CUDA cache after model setup and keep this much free memory per GPU. Omit to disable.",
    )
    parser.add_argument("--lora-r", type=int, default=64)
    parser.add_argument("--lora-alpha", type=int, default=64)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument(
        "--lora-target",
        type=str,
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
        help="Comma-separated module names.",
    )
    parser.add_argument("--rsvqa-train-ratio", type=float, default=0.2)
    parser.add_argument("--rsvqa-val-ratio", type=float, default=0.2)
    parser.add_argument("--geochat-val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if int(args.smart_resize_min_pixels) <= 0:
        raise ValueError(f"--smart-resize-min-pixels must be > 0, got {args.smart_resize_min_pixels}")
    if int(args.smart_resize_max_pixels) < int(args.smart_resize_min_pixels):
        raise ValueError(
            "--smart-resize-max-pixels must be >= --smart-resize-min-pixels, "
            f"got {args.smart_resize_max_pixels} < {args.smart_resize_min_pixels}"
        )
    if args.cuda_reserve_free_gb is not None and float(args.cuda_reserve_free_gb) < 0:
        raise ValueError(f"--cuda-reserve-free-gb must be >= 0, got {args.cuda_reserve_free_gb}")
    if float(args.rsvqa_train_ratio) <= 0 or float(args.rsvqa_train_ratio) > 1:
        raise ValueError(f"--rsvqa-train-ratio must be in (0, 1], got {args.rsvqa_train_ratio}")
    if float(args.rsvqa_val_ratio) <= 0 or float(args.rsvqa_val_ratio) > 1:
        raise ValueError(f"--rsvqa-val-ratio must be in (0, 1], got {args.rsvqa_val_ratio}")
    if float(args.geochat_val_ratio) <= 0 or float(args.geochat_val_ratio) >= 1:
        raise ValueError(f"--geochat-val-ratio must be in (0, 1), got {args.geochat_val_ratio}")

    torch.manual_seed(int(args.seed))
    require_cuda()

    accelerator = Accelerator(mixed_precision=None if args.mixed_precision == "no" else str(args.mixed_precision))
    device = accelerator.device
    if str(getattr(device, "type", "")) != "cuda":
        raise RuntimeError(f"Unexpected accelerator device: {device}. This run requires CUDA.")
    if accelerator.is_main_process:
        world_size_hint = int(getattr(accelerator, "num_processes", 1))
        visible_cuda = int(torch.cuda.device_count())
        if world_size_hint == 1 and visible_cuda > 1:
            print(
                "[WARN] Detected multiple CUDA devices but current run uses a single process. "
                "Use: accelerate launch --num_processes N ...",
                flush=True,
            )

    world_size = int(getattr(accelerator, "num_processes", 1))
    rank = int(getattr(accelerator, "process_index", 0))
    local_batch_sizes = _resolve_local_batch_sizes(
        world_size=world_size,
        base_batch=int(args.batch_size),
        per_rank=str(args.batch_size_per_rank),
        ratio=str(args.batch_size_ratio),
    )
    local_batch_size = int(local_batch_sizes[rank])
    global_micro_batch = int(sum(local_batch_sizes))
    effective_global_batch = int(global_micro_batch * int(args.grad_accum))
    if int(args.target_global_batch) > 0 and int(effective_global_batch) != int(args.target_global_batch):
        raise ValueError(
            "Global batch mismatch: "
            f"expect={int(args.target_global_batch)}, got={int(effective_global_batch)} "
            f"(batch_per_rank={local_batch_sizes}, grad_accum={int(args.grad_accum)}, world_size={world_size})."
        )
    sample_ddp_loss_scale = float(world_size) * float(local_batch_size) / float(global_micro_batch)

    qwen_model_dir = _resolve_from_project(args.qwen_model_dir)
    dinov3_dir = _resolve_from_project(args.dinov3_dir)
    out_dir = (
        _resolve_from_project(args.output_dir)
        if str(args.output_dir).strip()
        else _default_output_dir(str(args.dataset))
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    resume_lora_dir = _resolve_from_project(args.resume_lora) if str(args.resume_lora).strip() else None
    init_merger_path = _resolve_from_project(args.init_merger) if str(args.init_merger).strip() else None
    resume_step = 0
    resume_optimizer_path: Path | None = None
    resume_scheduler_path: Path | None = None

    if resume_lora_dir is not None:
        auto_merger = resume_lora_dir.parent / "merger.safetensors"
        if not auto_merger.is_file():
            raise FileNotFoundError(f"Missing sibling merger checkpoint for resume_lora: {auto_merger}")
        if init_merger_path is None:
            init_merger_path = auto_merger
        elif init_merger_path.resolve() != auto_merger.resolve():
            raise ValueError(
                "resume_lora and init_merger mismatch. "
                f"resume expects {auto_merger}, but got {init_merger_path}."
            )

        meta_path = auto_merger.with_suffix(".json")
        if not meta_path.is_file():
            raise FileNotFoundError(f"Missing merger metadata for strict resume: {meta_path}")
        try:
            resume_step = max(0, int(_read_json(meta_path).get("step", 0)))
        except Exception as exc:
            raise RuntimeError(f"Failed to read resume step from merger metadata: {meta_path}") from exc

        optimizer_path = resume_lora_dir.parent / "optimizer.pt"
        scheduler_path = resume_lora_dir.parent / "scheduler.pt"
        if not optimizer_path.is_file():
            raise FileNotFoundError(f"Missing optimizer state for strict resume: {optimizer_path}")
        if not scheduler_path.is_file():
            raise FileNotFoundError(f"Missing scheduler state for strict resume: {scheduler_path}")
        resume_optimizer_path = optimizer_path
        resume_scheduler_path = scheduler_path

    if init_merger_path is not None:
        resize_cfg = assert_dino_runtime_matches_merger(
            qwen_model_dir=qwen_model_dir,
            dinov3_dir=dinov3_dir,
            image_size=int(args.image_size),
            smart_resize_min_pixels=int(args.smart_resize_min_pixels),
            smart_resize_max_pixels=int(args.smart_resize_max_pixels),
            merger_ckpt=init_merger_path,
        )
    else:
        resize_cfg = resolve_dino_resize_config(
            image_size=int(args.image_size),
            smart_resize_min_pixels=int(args.smart_resize_min_pixels),
            smart_resize_max_pixels=int(args.smart_resize_max_pixels),
            merger_ckpt=None,
        )
    args.image_size = int(resize_cfg.image_size)
    args.smart_resize_min_pixels = int(resize_cfg.smart_resize_min_pixels)
    args.smart_resize_max_pixels = int(resize_cfg.smart_resize_max_pixels)

    train_rows, val_rows, dataset_meta = _load_dataset_splits(
        dataset_name=str(args.dataset),
        seed=int(args.seed),
        rsvqa_train_ratio=float(args.rsvqa_train_ratio),
        rsvqa_val_ratio=float(args.rsvqa_val_ratio),
        geochat_val_ratio=float(args.geochat_val_ratio),
    )
    cache_dir = _resolve_from_project(str(dataset_meta["cache_dir"]))

    processor = AutoProcessor.from_pretrained(str(qwen_model_dir))
    tokenizer = processor.tokenizer
    image_processor = AutoImageProcessor.from_pretrained(str(dinov3_dir))

    model_dtype = torch_dtype_from_str(
        "bf16"
        if args.mixed_precision == "bf16"
        else "fp16"
        if args.mixed_precision == "fp16"
        else "fp32"
    )
    base_model = Qwen3VLForConditionalGeneration.from_pretrained(
        str(qwen_model_dir),
        torch_dtype=model_dtype,
    )
    base_model.config.use_cache = False

    old_visual = base_model.model.visual
    adapter_cfg = DinoV3AdapterConfig(
        dinov3_dir=dinov3_dir,
        image_size=int(args.image_size),
        merge_size=int(old_visual.spatial_merge_size),
        deepstack_visual_indexes=tuple(int(x) for x in getattr(old_visual, "deepstack_visual_indexes", (5, 11, 17))),
        qwen_vision_depth=int(
            getattr(getattr(old_visual, "config", None), "depth", 0) or len(getattr(old_visual, "blocks", []))
        ),
    )
    adapter = DinoV3VisualAdapter(
        adapter_cfg,
        merger=old_visual.merger,
        deepstack_merger_list=getattr(old_visual, "deepstack_merger_list", None),
        torch_dtype=base_model.dtype,
    )
    base_model.model.visual = adapter

    for parameter in base_model.parameters():
        parameter.requires_grad = False
    _enable_visual_trainables(base_model)

    if init_merger_path is not None:
        load_merger_safetensors(base_model, init_merger_path)

    lora_target_leaf = [item.strip() for item in str(args.lora_target).split(",") if item.strip()]
    lora_targets = _resolve_language_lora_targets(base_model, lora_target_leaf)
    lora_cfg = LoraConfig(
        r=int(args.lora_r),
        lora_alpha=int(args.lora_alpha),
        lora_dropout=float(args.lora_dropout),
        bias="none",
        target_modules=lora_targets,
        task_type="CAUSAL_LM",
    )

    if resume_lora_dir is not None:
        model = PeftModel.from_pretrained(base_model, str(resume_lora_dir), is_trainable=True)
    else:
        model = get_peft_model(base_model, lora_cfg)

    _enable_visual_trainables(model.get_base_model())
    _assert_no_visual_lora_trainables(model)
    _assert_expected_trainables_layout(model)

    if args.gradient_checkpointing:
        try:
            base = model.get_base_model()
            base.model.language_model.gradient_checkpointing_enable()
        except Exception:
            pass

    train_dataset = SingleTaskDinoSFTDataset(
        train_rows,
        tokenizer=tokenizer,
        image_processor=image_processor,
        cache_dir=cache_dir,
        cache_name=f"{args.dataset}_train",
        patch_size=int(base_model.config.vision_config.patch_size),
        merge_size=int(base_model.config.vision_config.spatial_merge_size),
        smart_resize_min_pixels=int(args.smart_resize_min_pixels),
        smart_resize_max_pixels=int(args.smart_resize_max_pixels),
        image_token="<|image_pad|>",
    )
    val_dataset = SingleTaskDinoSFTDataset(
        val_rows,
        tokenizer=tokenizer,
        image_processor=image_processor,
        cache_dir=cache_dir,
        cache_name=f"{args.dataset}_val",
        patch_size=int(base_model.config.vision_config.patch_size),
        merge_size=int(base_model.config.vision_config.spatial_merge_size),
        smart_resize_min_pixels=int(args.smart_resize_min_pixels),
        smart_resize_max_pixels=int(args.smart_resize_max_pixels),
        image_token="<|image_pad|>",
    )
    collator = CaptionSFTCollator(pad_token_id=int(tokenizer.pad_token_id))

    train_dataset.ensure_image_hw_cache(build_if_missing=bool(accelerator.is_main_process))
    val_dataset.ensure_image_hw_cache(build_if_missing=bool(accelerator.is_main_process))
    accelerator.wait_for_everyone()
    train_dataset.ensure_image_hw_cache(build_if_missing=False)
    val_dataset.ensure_image_hw_cache(build_if_missing=False)

    train_bucket_keys = [train_dataset.get_resize_bucket_key(index) for index in range(len(train_dataset))]
    val_bucket_keys = [val_dataset.get_resize_bucket_key(index) for index in range(len(val_dataset))]

    train_sampler = ProportionalDistributedSampler(
        dataset_size=int(len(train_dataset)),
        local_batch_sizes=local_batch_sizes,
        rank=rank,
        seed=int(args.seed),
        grad_accum=int(args.grad_accum),
        sample_bucket_keys=train_bucket_keys,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(local_batch_size),
        shuffle=False,
        sampler=train_sampler,
        num_workers=int(args.num_workers),
        pin_memory=True,
        collate_fn=collator,
        drop_last=True,
    )

    val_batch_sampler = SequentialBucketBatchSampler(
        local_batch_size=int(local_batch_size),
        process_index=rank,
        world_size=world_size,
        sample_bucket_keys=val_bucket_keys,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_sampler=val_batch_sampler,
        num_workers=int(args.num_workers),
        pin_memory=True,
        collate_fn=collator,
    )

    local_loader_len = int(len(train_loader))
    local_loader_len_tensor = torch.tensor([local_loader_len], device=device, dtype=torch.long)
    all_loader_lens = accelerator.gather(local_loader_len_tensor).detach().cpu().tolist()
    min_loader_len = int(min(int(x) for x in all_loader_lens))
    max_loader_len = int(max(int(x) for x in all_loader_lens))
    if min_loader_len != max_loader_len:
        raise RuntimeError(
            "Loader length mismatch across ranks under proportional sampler. "
            f"lens={all_loader_lens}, batch_per_rank={local_batch_sizes}."
        )

    steps_per_epoch = math.floor(float(min_loader_len) / max(1, int(args.grad_accum)))
    if steps_per_epoch <= 0:
        raise RuntimeError(
            "steps_per_epoch is 0. "
            "Reduce batch_size/grad_accum or check dataset size."
        )

    train_used_items_per_epoch = int(steps_per_epoch * effective_global_batch)
    if train_used_items_per_epoch > len(train_rows):
        raise RuntimeError(
            f"Computed used_items_per_epoch exceeds train size: used={train_used_items_per_epoch}, size={len(train_rows)}."
        )
    train_dropped_items_per_epoch = int(len(train_rows) - train_used_items_per_epoch)
    samples_per_rank_per_epoch = [
        int(steps_per_epoch * int(args.grad_accum) * batch_size) for batch_size in local_batch_sizes
    ]

    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=float(args.lr), weight_decay=float(args.weight_decay))
    model, optimizer = accelerator.prepare(model, optimizer)
    accelerator.unwrap_model(model, keep_fp32_wrapper=False)

    prealloc_info = preallocate_cuda_cache(
        device=device,
        keep_free_gb=None if args.cuda_reserve_free_gb is None else float(args.cuda_reserve_free_gb),
    )
    prealloc_tensor = torch.tensor(
        [
            int(prealloc_info["free_before_bytes"]),
            int(prealloc_info["reserved_bytes"]),
            int(prealloc_info["free_after_bytes"]),
        ],
        device=device,
        dtype=torch.long,
    ).unsqueeze(0)
    gathered_prealloc = accelerator.gather(prealloc_tensor).detach().cpu().tolist()

    total_epochs = int(math.ceil(float(args.epochs)))
    total_steps = int(math.ceil(float(args.epochs) * steps_per_epoch))
    if int(args.max_steps) > 0:
        total_steps = min(total_steps, int(args.max_steps))

    warmup_steps = int(float(args.warmup_ratio) * total_steps)
    lr_scheduler = get_scheduler(
        name="cosine",
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    optimizer_state_loaded = False
    scheduler_state_loaded = False
    if resume_optimizer_path is not None:
        optimizer.load_state_dict(torch.load(str(resume_optimizer_path), map_location="cpu"))
        optimizer_state_loaded = True
    if resume_scheduler_path is not None:
        lr_scheduler.load_state_dict(torch.load(str(resume_scheduler_path), map_location="cpu"))
        scheduler_state_loaded = True

    start_epoch = 0
    if resume_step > 0:
        if resume_step > total_steps:
            raise RuntimeError(
                f"resume step ({resume_step}) exceeds total_steps ({total_steps}). Check --epochs/--max-steps."
            )
        if resume_step % steps_per_epoch != 0:
            raise RuntimeError(
                "Resume step is not aligned to epoch boundary. "
                f"resume_step={resume_step}, steps_per_epoch={steps_per_epoch}."
            )
        start_epoch = int(resume_step // steps_per_epoch)
        if not scheduler_state_loaded:
            for _ in range(int(resume_step)):
                lr_scheduler.step()

    run_meta = {
        "qwen_model_dir": _rel_to_project(qwen_model_dir),
        "dinov3_dir": _rel_to_project(dinov3_dir),
        "dataset_name": str(dataset_meta["dataset_name"]),
        "task_name": str(dataset_meta["task_name"]),
        "dataset_label": str(dataset_meta["dataset_label"]),
        "data_root": str(dataset_meta["data_root"]),
        "image_root": str(dataset_meta["image_root"]),
        "image_size": int(args.image_size),
        "smart_resize_min_pixels": int(args.smart_resize_min_pixels),
        "smart_resize_max_pixels": int(args.smart_resize_max_pixels),
    }

    plotter = SingleTaskLossPlotter(
        output_dir=out_dir,
        title=f"{dataset_meta['dataset_label']} Train / Val Loss",
    )

    if accelerator.is_main_process:
        (out_dir / "run_config.json").write_text(
            json.dumps(
                {
                    "run": run_meta,
                    "dataset": dict(dataset_meta),
                    "adapter": {
                        "dinov3_dir": _rel_to_project(dinov3_dir),
                        "image_size": int(adapter_cfg.image_size),
                        "merge_size": int(adapter_cfg.merge_size),
                        "deepstack_visual_indexes": list(adapter_cfg.deepstack_visual_indexes),
                    },
                    "train": {
                        "epochs": float(args.epochs),
                        "max_steps": int(args.max_steps),
                        "batch_size": int(args.batch_size),
                        "batch_size_per_rank": list(local_batch_sizes),
                        "batch_size_ratio": str(args.batch_size_ratio),
                        "local_batch_size": int(local_batch_size),
                        "global_micro_batch": int(global_micro_batch),
                        "global_batch": int(effective_global_batch),
                        "target_global_batch": int(args.target_global_batch),
                        "micro_batch_loss_average": "sample",
                        "ddp_loss_scale_mode": "sample_true_mean",
                        "sample_ddp_loss_scale": float(sample_ddp_loss_scale),
                        "sampling_strategy": "single_task_bucketed_shuffle_no_replacement",
                        "samples_per_rank_per_epoch": [int(x) for x in samples_per_rank_per_epoch],
                        "samples_per_epoch": int(train_used_items_per_epoch),
                        "train_used_items_per_epoch": int(train_used_items_per_epoch),
                        "train_dropped_items_per_epoch": int(train_dropped_items_per_epoch),
                        "steps_per_epoch": int(steps_per_epoch),
                        "total_steps": int(total_steps),
                        "grad_accum": int(args.grad_accum),
                        "lr": float(args.lr),
                        "weight_decay": float(args.weight_decay),
                        "warmup_ratio": float(args.warmup_ratio),
                        "mixed_precision": str(args.mixed_precision),
                        "gradient_checkpointing": bool(args.gradient_checkpointing),
                        "cuda_reserve_free_gb": None
                        if args.cuda_reserve_free_gb is None
                        else float(args.cuda_reserve_free_gb),
                        "num_workers": int(args.num_workers),
                        "save_strategy": "epoch_before_val",
                    },
                    "val": {
                        "num_rows": int(len(val_rows)),
                        "batch_size_per_rank": list(local_batch_sizes),
                        "sampler": "single_task_bucketed_sequential",
                        "history_path": "epoch_loss_history.json",
                        "plot_path": "epoch_loss_overall.svg",
                    },
                    "lora": {
                        "r": int(args.lora_r),
                        "alpha": int(args.lora_alpha),
                        "dropout": float(args.lora_dropout),
                        "target_leaf": lora_target_leaf,
                        "target_modules": lora_targets,
                    },
                    "init_merger": _rel_to_project(init_merger_path) if init_merger_path is not None else "",
                    "resume_lora": _rel_to_project(resume_lora_dir) if resume_lora_dir is not None else "",
                    "resume_optimizer": _rel_to_project(resume_optimizer_path) if resume_optimizer_path is not None else "",
                    "resume_scheduler": _rel_to_project(resume_scheduler_path) if resume_scheduler_path is not None else "",
                    "optimizer_state_loaded": bool(optimizer_state_loaded),
                    "scheduler_state_loaded": bool(scheduler_state_loaded),
                    "resume_step": int(resume_step),
                },
                ensure_ascii=False,
                indent=2,
                default=str,
            )
            + "\n",
            encoding="utf-8",
        )

        print(
            f"[INFO] dataset={dataset_meta['dataset_name']} task={dataset_meta['task_name']} "
            f"train_items={len(train_rows)} val_items={len(val_rows)} "
            f"batch_per_rank={local_batch_sizes} grad_accum={int(args.grad_accum)} "
            f"global_batch={effective_global_batch} steps/epoch={steps_per_epoch} total_steps={total_steps} "
            f"start_epoch={start_epoch} resume_step={resume_step} save=epoch_before_val",
            flush=True,
        )
        if args.cuda_reserve_free_gb is None:
            print("[INFO] cuda_prealloc disabled", flush=True)
        else:
            for prealloc_rank, values in enumerate(gathered_prealloc):
                free_before_bytes, reserved_bytes, free_after_bytes = [int(x) for x in values]
                print(
                    f"[INFO] cuda_prealloc rank={prealloc_rank} keep_free_gb={float(args.cuda_reserve_free_gb):.2f} "
                    f"free_before_gb={free_before_bytes / (1024**3):.2f} "
                    f"reserved_gb={reserved_bytes / (1024**3):.2f} "
                    f"free_after_gb={free_after_bytes / (1024**3):.2f}",
                    flush=True,
                )
        if str(args.dataset) == "rsvqa_hr":
            print(
                f"[INFO] rsvqa_train_type_counts={dataset_meta['train_type_counts_selected']} "
                f"rsvqa_val_type_counts={dataset_meta['val_type_counts_selected']}",
                flush=True,
            )

    model.train()
    optimizer.zero_grad(set_to_none=True)
    global_step = int(resume_step)

    for epoch in range(start_epoch, total_epochs):
        if global_step >= total_steps:
            break

        if hasattr(train_sampler, "set_epoch"):
            train_sampler.set_epoch(int(epoch))
        train_iterator = iter(train_loader)
        epoch_updates = 0
        train_loss_sum = torch.zeros((), dtype=torch.float64, device=device)
        train_sample_count = torch.zeros((), dtype=torch.float64, device=device)

        while epoch_updates < int(steps_per_epoch) and global_step < total_steps:
            update_loss = 0.0
            for micro_step in range(int(args.grad_accum)):
                try:
                    batch = next(train_iterator)
                except StopIteration:
                    raise RuntimeError(
                        "Single-task train loader exhausted before expected updates were consumed: "
                        f"epoch={epoch + 1}, micro_step={micro_step}, grad_accum={int(args.grad_accum)}."
                    )

                batch.pop("meta", None)
                batch = _move_batch_to_device(batch, device=device)
                input_ids = batch["input_ids"]
                attention_mask = batch["attention_mask"]
                pixel_values = batch["pixel_values"]
                image_grid_thw = batch["image_grid_thw"]
                labels = batch["labels"]
                with accelerator.autocast():
                    outputs = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        pixel_values=pixel_values,
                        image_grid_thw=image_grid_thw,
                    )
                local_sample_mean_loss = causal_lm_sample_average_loss(outputs.logits, labels)
                local_batch_count = int(labels.shape[0])
                train_loss_sum += local_sample_mean_loss.detach().to(dtype=torch.float64) * float(local_batch_count)
                train_sample_count += float(local_batch_count)
                loss = local_sample_mean_loss * float(sample_ddp_loss_scale)
                loss = loss / float(args.grad_accum)
                accelerator.backward(loss)
                update_loss += float(loss.detach().item())

            if float(args.max_grad_norm) > 0:
                accelerator.clip_grad_norm_(trainable, float(args.max_grad_norm))
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad(set_to_none=True)

            global_step += 1
            epoch_updates += 1

            if accelerator.is_main_process:
                lr = float(lr_scheduler.get_last_lr()[0])
                print(
                    f"step={global_step} epoch={epoch + 1} lr={lr:.3e} loss={update_loss:.4f}",
                    flush=True,
                )

        completed_full_epoch = int(epoch_updates) == int(steps_per_epoch)
        if completed_full_epoch:
            accelerator.wait_for_everyone()
            _save_training_checkpoint(
                accelerator=accelerator,
                model=model,
                optimizer=optimizer,
                lr_scheduler=lr_scheduler,
                out_dir=out_dir,
                checkpoint_name=f"epoch{epoch + 1}",
                global_step=int(global_step),
                epoch=int(epoch + 1),
                run_meta=run_meta,
                adapter_cfg=adapter_cfg,
                dinov3_dir=dinov3_dir,
                dataset_meta=dataset_meta,
            )
            train_epoch_loss = plotter.record_epoch(
                mode="train",
                epoch=int(epoch + 1),
                local_loss_sum=train_loss_sum,
                local_sample_count=train_sample_count,
            )
            val_loss_sum, val_sample_count = _run_validation(
                accelerator=accelerator,
                model=model,
                val_loader=val_loader,
                device=device,
            )
            val_epoch_loss = plotter.record_epoch(
                mode="eval",
                epoch=int(epoch + 1),
                local_loss_sum=val_loss_sum,
                local_sample_count=val_sample_count,
            )
            accelerator.wait_for_everyone()
            if accelerator.is_main_process:
                plotter.write_outputs()
                print(f"[OK] Saved epoch checkpoint: {out_dir / f'epoch{epoch + 1}'}", flush=True)
                if train_epoch_loss is not None:
                    print(f"[INFO] epoch={epoch + 1} train_loss={train_epoch_loss:.6f}", flush=True)
                if val_epoch_loss is not None:
                    print(f"[INFO] epoch={epoch + 1} val_loss={val_epoch_loss:.6f}", flush=True)
        elif accelerator.is_main_process and int(epoch_updates) > 0:
            print(
                f"[WARN] Epoch {epoch + 1} ended early: updates={epoch_updates}/{steps_per_epoch}.",
                flush=True,
            )

        if global_step >= total_steps:
            break

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        if int(global_step) > 0 and int(global_step) % int(steps_per_epoch) == 0:
            _save_training_checkpoint(
                accelerator=accelerator,
                model=model,
                optimizer=optimizer,
                lr_scheduler=lr_scheduler,
                out_dir=out_dir,
                checkpoint_name="final",
                global_step=int(global_step),
                epoch=int(global_step // steps_per_epoch),
                run_meta=run_meta,
                adapter_cfg=adapter_cfg,
                dinov3_dir=dinov3_dir,
                dataset_meta=dataset_meta,
            )
            print(f"[OK] Saved: {out_dir / 'final'}", flush=True)
        else:
            print(
                "[WARN] Skip saving final checkpoint because current step is not on an epoch boundary: "
                f"step={global_step}, steps_per_epoch={steps_per_epoch}.",
                flush=True,
            )

    accelerator.end_training()


if __name__ == "__main__":
    main()
