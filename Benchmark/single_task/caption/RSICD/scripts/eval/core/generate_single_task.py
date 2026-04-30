from __future__ import annotations

import argparse
import gc
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm

SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[5]
VRSBENCH_ROOT = SCRIPT_PATH.parents[4]
VRSBENCH_SRC = VRSBENCH_ROOT / "src"
if str(VRSBENCH_SRC) not in sys.path:
    sys.path.insert(0, str(VRSBENCH_SRC))
GROUNDING_SCRIPT_DIR = VRSBENCH_ROOT / "benchmark" / "single_task" / "grounding" / "scripts"
if str(GROUNDING_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(GROUNDING_SCRIPT_DIR))

from ftqwen3.dinov3_captioner import DinoV3Captioner
from ftqwen3.jsonl import append_jsonl, read_json, read_jsonl, write_json
from ftqwen3.qwen3_vl_native_captioner import Qwen3VLNativeCaptioner
from ftqwen3.qwen_dinov3 import (
    DinoResizeConfig,
    assert_dino_runtime_matches_merger,
    assert_path_metadata_matches,
    read_merger_run_meta,
    resolve_dino_resize_config,
)
from eval_grounding_single_object import GeoChatRunner, GeoGroundRunner


DEFAULT_CAPTION_PROMPT = (
    "Write one concise English caption for this remote sensing image in 8 to 15 words. "
    "Mention only the main scene and key objects."
)
DEFAULT_DATASETS = (
    "ucm_captions",
    "sydney_captions",
    "rsicd",
)


@dataclass(frozen=True)
class CaptionDatasetSpec:
    name: str
    dataset_json: Path
    image_dir: Path


@dataclass(frozen=True)
class ModelPreset:
    name: str
    model_family: str
    qwen_model_dir: Path
    dinov3_dir: Path | None
    merger_ckpt: Path | None
    lora_dir: Path | None
    image_size: int | None
    smart_resize_min_pixels: int | None
    smart_resize_max_pixels: int | None
    dtype: str


@dataclass(frozen=True)
class GenerationResult:
    text: str
    generated_token_count: int
    ended_by_eos: bool
    last_generated_token_id: int | None


class CaptionRunner:
    def __init__(self, *, preset: ModelPreset, args: argparse.Namespace, max_new_tokens: int) -> None:
        self.preset = preset
        self.model_family = str(preset.model_family)
        self.requested_batch_size = int(args.batch_size)
        self.dtype = str(args.dtype if args.dtype else preset.dtype)
        self.device_map = str(args.device_map)
        self.merge_lora = bool(args.merge_lora)
        self.runtime_resize: DinoResizeConfig | None = None
        self.impl: DinoV3Captioner | Qwen3VLNativeCaptioner | GeoChatRunner | GeoGroundRunner

        if self.model_family == "dinov3":
            if preset.dinov3_dir is None:
                raise ValueError("dinov3 preset requires dinov3_dir")
            if preset.merger_ckpt is None:
                raise ValueError("dinov3 preset requires merger_ckpt")
            image_size = int(args.image_size) if args.image_size is not None else int(preset.image_size or 512)
            min_pixels = (
                int(args.smart_resize_min_pixels)
                if args.smart_resize_min_pixels is not None
                else preset.smart_resize_min_pixels
            )
            max_pixels = (
                int(args.smart_resize_max_pixels)
                if args.smart_resize_max_pixels is not None
                else preset.smart_resize_max_pixels
            )
            resize_override = (
                args.image_size is not None
                or args.smart_resize_min_pixels is not None
                or args.smart_resize_max_pixels is not None
            )
            force_exact_image_size = (
                args.image_size is not None
                and args.smart_resize_min_pixels is not None
                and args.smart_resize_max_pixels is not None
                and int(args.smart_resize_min_pixels) == int(args.image_size) * int(args.image_size)
                and int(args.smart_resize_max_pixels) == int(args.image_size) * int(args.image_size)
            )
            if resize_override:
                run_meta = read_merger_run_meta(preset.merger_ckpt)
                assert_path_metadata_matches(
                    label="Qwen model dir",
                    expected=run_meta.get("qwen_model_dir"),
                    actual=preset.qwen_model_dir,
                )
                assert_path_metadata_matches(
                    label="DINOv3 dir",
                    expected=run_meta.get("dinov3_dir"),
                    actual=preset.dinov3_dir,
                )
                resize_cfg = resolve_dino_resize_config(
                    image_size=int(image_size),
                    smart_resize_min_pixels=min_pixels,
                    smart_resize_max_pixels=max_pixels,
                    merger_ckpt=None,
                )
            else:
                resize_cfg = assert_dino_runtime_matches_merger(
                    qwen_model_dir=preset.qwen_model_dir,
                    dinov3_dir=preset.dinov3_dir,
                    image_size=int(image_size),
                    smart_resize_min_pixels=min_pixels,
                    smart_resize_max_pixels=max_pixels,
                    merger_ckpt=preset.merger_ckpt,
                )
            self.runtime_resize = resize_cfg
            self.impl = DinoV3Captioner(
                qwen_model_dir=preset.qwen_model_dir,
                dinov3_dir=preset.dinov3_dir,
                image_size=int(resize_cfg.image_size),
                smart_resize_min_pixels=int(resize_cfg.smart_resize_min_pixels),
                smart_resize_max_pixels=int(resize_cfg.smart_resize_max_pixels),
                merger_ckpt=preset.merger_ckpt,
                lora_dir=preset.lora_dir,
                device_map=self.device_map,
                dtype=self.dtype,
                max_new_tokens=int(max_new_tokens),
                do_sample=args.do_sample,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                num_beams=args.num_beams,
                seed=args.seed,
                repetition_penalty=args.repetition_penalty,
                merge_lora=self.merge_lora,
                force_exact_image_size=bool(force_exact_image_size),
            )
            self.decode_strategy = str(self.impl.decode_strategy)
            return

        if self.model_family == "qwen_native":
            self.impl = Qwen3VLNativeCaptioner(
                preset.qwen_model_dir,
                merger_ckpt=preset.merger_ckpt,
                lora_dir=preset.lora_dir,
                merge_lora=self.merge_lora,
                device_map=self.device_map,
                dtype=self.dtype,
                max_new_tokens=int(max_new_tokens),
                do_sample=args.do_sample,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                num_beams=args.num_beams,
                repetition_penalty=args.repetition_penalty,
                seed=args.seed,
            )
            self.decode_strategy = str(self.impl.decode_strategy)
            return

        if self.model_family == "geochat":
            self.impl = GeoChatRunner(
                code_root=resolve_from_project(args.geochat_code_root),
                model_path=resolve_from_project(args.geochat_model_dir),
                device=self.device_map,
                conv_mode=str(args.conv_mode),
                max_new_tokens=int(max_new_tokens),
                temperature=float(args.temperature if args.temperature is not None else 0.2),
                top_p=None if args.top_p is None else float(args.top_p),
                num_beams=int(args.num_beams if args.num_beams is not None else 1),
            )
            self.decode_strategy = "sample" if bool(args.do_sample) else "greedy"
            return

        if self.model_family == "geoground":
            llava_code_root = (
                None if not str(args.llava_code_root).strip() else resolve_from_project(args.llava_code_root)
            )
            geoground_model_base = (
                None
                if not str(args.geoground_model_base).strip()
                else resolve_from_project(args.geoground_model_base)
            )
            self.impl = GeoGroundRunner(
                llava_code_root=llava_code_root,
                model_path=resolve_from_project(args.geoground_model_dir),
                model_base=geoground_model_base,
                device=self.device_map,
                conv_mode=str(args.conv_mode),
                max_new_tokens=int(max_new_tokens),
                temperature=float(args.temperature if args.temperature is not None else 0.2),
                top_p=None if args.top_p is None else float(args.top_p),
                num_beams=int(args.num_beams if args.num_beams is not None else 1),
            )
            self.decode_strategy = "sample" if bool(args.do_sample) else "greedy"
            return

        raise ValueError(f"unsupported model_family: {self.model_family}")

    def generate_batch_prompts(self, *, image_paths: list[Path], prompts: list[str]) -> list[GenerationResult]:
        if self.model_family in {"dinov3", "qwen_native"}:
            outputs = self.impl.caption_batch_prompts(image_paths=image_paths, prompts=prompts)
        else:
            outputs = self.impl.generate_batch(image_paths=image_paths, prompts=prompts)

        normalized: list[GenerationResult] = []
        for item in outputs:
            if hasattr(item, "text"):
                normalized.append(
                    GenerationResult(
                        text=str(item.text).strip(),
                        generated_token_count=int(getattr(item, "generated_token_count", -1)),
                        ended_by_eos=bool(getattr(item, "ended_by_eos", False)),
                        last_generated_token_id=getattr(item, "last_generated_token_id", None),
                    )
                )
            else:
                normalized.append(
                    GenerationResult(
                        text=str(item).strip(),
                        generated_token_count=-1,
                        ended_by_eos=False,
                        last_generated_token_id=None,
                    )
                )
        return normalized


def single_task_source_root() -> Path:
    return PROJECT_ROOT / "fine-tune-qwen3-vl" / "Benchmark" / "single_task"


def benchmark_root() -> Path:
    return VRSBENCH_ROOT / "benchmark" / "single_task"


def resolve_from_vrsbench(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    if candidate.parts and candidate.parts[0] == VRSBENCH_ROOT.name:
        return (PROJECT_ROOT / candidate).resolve()
    return (VRSBENCH_ROOT / candidate).resolve()


def resolve_from_project(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    project_candidate = (PROJECT_ROOT / candidate).resolve()
    if project_candidate.exists():
        return project_candidate
    return (VRSBENCH_ROOT / candidate).resolve()


def checkpoint_run_root(merger_ckpt: Path) -> Path:
    path = Path(merger_ckpt)
    if path.name == "merger.safetensors":
        return path.parents[1]
    raise ValueError(f"unexpected merger checkpoint path: {merger_ckpt}")


def load_preset_from_checkpoint(*, name: str, model_family: str, merger_ckpt: Path) -> ModelPreset:
    run_root = checkpoint_run_root(merger_ckpt)
    run_config = read_json(run_root / "run_config.json")
    run_meta = run_config.get("run", {})
    qwen_model_dir = resolve_from_vrsbench(run_meta.get("qwen_model_dir", "models/Qwen3-VL-8B-Instruct"))
    lora_dir = merger_ckpt.parent / "lora"

    if model_family == "dinov3":
        dinov3_dir = resolve_from_vrsbench(run_meta.get("dinov3_dir", "models/dinov3/dinov3-vitl16-pretrain-sat493m"))
        return ModelPreset(
            name=name,
            model_family="dinov3",
            qwen_model_dir=qwen_model_dir,
            dinov3_dir=dinov3_dir,
            merger_ckpt=merger_ckpt,
            lora_dir=lora_dir if lora_dir.is_dir() else None,
            image_size=int(run_meta.get("image_size", 512)),
            smart_resize_min_pixels=None,
            smart_resize_max_pixels=None,
            dtype="bf16",
        )

    if model_family == "qwen_native":
        return ModelPreset(
            name=name,
            model_family="qwen_native",
            qwen_model_dir=qwen_model_dir,
            dinov3_dir=None,
            merger_ckpt=merger_ckpt,
            lora_dir=lora_dir if lora_dir.is_dir() else None,
            image_size=None,
            smart_resize_min_pixels=_coerce_optional_int(run_meta.get("smart_resize_min_pixels")),
            smart_resize_max_pixels=_coerce_optional_int(run_meta.get("smart_resize_max_pixels")),
            dtype="bf16",
        )

    raise ValueError(f"unsupported model_family: {model_family}")


def build_presets() -> dict[str, ModelPreset]:
    dino_ckpt = resolve_from_vrsbench(
        "checkpoints/vrsbench_joint/"
        "merger_lora_8b_dinov3_micro8_8_ga2_effective32_taskseq_smartbucket_sampleavg_wd001_run_20260308_025747/"
        "epoch10/merger.safetensors"
    )
    qwen_ckpt = resolve_from_vrsbench(
        "checkpoints/vrsbench_joint/"
        "merger_lora_8b_qwen_native_micro8_8_ga2_effective32_wd001_taskseq_run_20260302_160151/"
        "epoch10/merger.safetensors"
    )
    qwen_model_dir = resolve_from_vrsbench("models/Qwen3-VL-8B-Instruct")
    return {
        "dinov3_epoch10": load_preset_from_checkpoint(
            name="dinov3_epoch10",
            model_family="dinov3",
            merger_ckpt=dino_ckpt,
        ),
        "qwen_native_epoch10": load_preset_from_checkpoint(
            name="qwen_native_epoch10",
            model_family="qwen_native",
            merger_ckpt=qwen_ckpt,
        ),
        "qwen3vl_base": ModelPreset(
            name="qwen3vl_base",
            model_family="qwen_native",
            qwen_model_dir=qwen_model_dir,
            dinov3_dir=None,
            merger_ckpt=None,
            lora_dir=None,
            image_size=None,
            smart_resize_min_pixels=None,
            smart_resize_max_pixels=None,
            dtype="bf16",
        ),
        "geochat": ModelPreset(
            name="geochat",
            model_family="geochat",
            qwen_model_dir=qwen_model_dir,
            dinov3_dir=None,
            merger_ckpt=None,
            lora_dir=None,
            image_size=None,
            smart_resize_min_pixels=None,
            smart_resize_max_pixels=None,
            dtype="bf16",
        ),
        "geoground": ModelPreset(
            name="geoground",
            model_family="geoground",
            qwen_model_dir=qwen_model_dir,
            dinov3_dir=None,
            merger_ckpt=None,
            lora_dir=None,
            image_size=None,
            smart_resize_min_pixels=None,
            smart_resize_max_pixels=None,
            dtype="bf16",
        ),
    }


def build_caption_specs() -> dict[str, CaptionDatasetSpec]:
    source_root = single_task_source_root()
    caption_root = source_root / "caption"
    rsicd_images = PROJECT_ROOT / "RSICD" / "datasets" / "rsicd" / "RSICD_images"
    return {
        "ucm_captions": CaptionDatasetSpec(
            name="ucm_captions",
            dataset_json=caption_root / "UCM-captions" / "data" / "dataset.json",
            image_dir=caption_root / "UCM-captions" / "data" / "imgs",
        ),
        "sydney_captions": CaptionDatasetSpec(
            name="sydney_captions",
            dataset_json=caption_root / "Sydney-captions" / "data" / "dataset.json",
            image_dir=caption_root / "Sydney-captions" / "data" / "imgs",
        ),
        "rsicd": CaptionDatasetSpec(
            name="rsicd",
            dataset_json=caption_root / "RSICD" / "data" / "dataset_rsicd.json",
            image_dir=rsicd_images,
        ),
    }


def _coerce_optional_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except Exception:
        return None


def parse_datasets(raw: str) -> list[str]:
    items = [item.strip() for item in str(raw).split(",") if item.strip()]
    if not items or items == ["all"]:
        return list(DEFAULT_DATASETS)
    return items


def load_caption_samples(spec: CaptionDatasetSpec, *, max_samples: int) -> list[dict[str, Any]]:
    payload = read_json(spec.dataset_json)
    rows: list[dict[str, Any]] = []
    for item in payload["images"]:
        if str(item.get("split", "")).strip().lower() != "test":
            continue
        filename = str(item["filename"])
        refs = [
            str(sentence.get("raw", "")).strip()
            for sentence in item.get("sentences", [])
            if str(sentence.get("raw", "")).strip()
        ]
        rows.append(
            {
                "dataset": spec.name,
                "task": "caption",
                "sample_id": str(item.get("imgid", filename)),
                "filename": filename,
                "image_path": spec.image_dir / filename,
                "refs": refs,
            }
        )
    rows = sorted(rows, key=lambda row: str(row["filename"]))
    if int(max_samples) > 0:
        rows = rows[: int(max_samples)]
    return rows


def resolve_output_dir(raw: str, *, preset_name: str) -> Path:
    if str(raw).strip():
        output_dir = Path(raw)
        if not output_dir.is_absolute():
            return resolve_from_vrsbench(output_dir)
        return output_dir
    return benchmark_root() / "outputs" / preset_name


def dataset_output_path(output_dir: Path, *, dataset_name: str, shard_suffix: str) -> Path:
    return output_dir / f"{dataset_name}{str(shard_suffix)}.jsonl"


def shard_file_suffix(*, world_size: int, rank: int) -> str:
    if int(world_size) <= 1:
        return ""
    return f".gpu{int(rank)}"


def parse_shard_weights(weights: str, *, world_size: int) -> list[int] | None:
    raw = str(weights).strip()
    if not raw:
        return None
    values = [int(item.strip()) for item in raw.split(":") if item.strip()]
    if len(values) != int(world_size):
        raise ValueError(f"shard_weights expects {world_size} values, got {len(values)}: {weights}")
    if any(value <= 0 for value in values):
        raise ValueError(f"shard_weights must be positive integers: {weights}")
    return values


def slice_indices_by_shard(indices: list[int], *, world_size: int, rank: int, weights: str) -> list[int]:
    if int(world_size) <= 0:
        raise ValueError(f"shard_world_size must be >= 1, got {world_size}")
    if int(rank) < 0 or int(rank) >= int(world_size):
        raise ValueError(f"shard_rank out of range: rank={rank}, world_size={world_size}")

    parsed = parse_shard_weights(weights, world_size=int(world_size))
    if parsed is None:
        return [item for index, item in enumerate(indices) if (index % int(world_size)) == int(rank)]

    total = len(indices)
    denom = int(sum(parsed))
    left = int(sum(parsed[: int(rank)]))
    right = int(sum(parsed[: int(rank) + 1]))
    start = (total * left) // denom
    end = (total * right) // denom
    return indices[start:end]


def select_rows_for_shard(rows: list[dict[str, Any]], *, world_size: int, rank: int, weights: str) -> list[dict[str, Any]]:
    selected_indices = slice_indices_by_shard(
        list(range(len(rows))),
        world_size=int(world_size),
        rank=int(rank),
        weights=str(weights),
    )
    return [rows[index] for index in selected_indices]


def runtime_meta(runner: CaptionRunner) -> dict[str, Any]:
    base = {
        "preset": runner.preset.name,
        "model_family": runner.model_family,
        "merger_ckpt": str(runner.preset.merger_ckpt) if runner.preset.merger_ckpt is not None else "",
        "lora_dir": str(runner.preset.lora_dir) if runner.preset.lora_dir is not None else "",
        "device_map": str(runner.device_map),
        "dtype": str(runner.dtype),
        "decode_strategy": str(runner.decode_strategy),
    }
    if runner.model_family in {"dinov3", "qwen_native"}:
        base["qwen_model_dir"] = str(runner.preset.qwen_model_dir)
    if runner.model_family == "dinov3":
        base["dinov3_dir"] = str(runner.preset.dinov3_dir)
        if runner.runtime_resize is not None:
            base["image_size"] = int(runner.runtime_resize.image_size)
            base["smart_resize_min_pixels"] = int(runner.runtime_resize.smart_resize_min_pixels)
            base["smart_resize_max_pixels"] = int(runner.runtime_resize.smart_resize_max_pixels)
            base["resize_mode"] = str(runner.runtime_resize.mode)
        base["force_exact_image_size"] = bool(getattr(runner.impl, "force_exact_image_size", False))
    if runner.model_family == "geochat":
        base["geochat_model_dir"] = str(resolve_from_project("GeoChat-Bench/model/geochat-7B"))
    if runner.model_family == "geoground":
        base["geoground_model_dir"] = str(
            resolve_from_project("VRSBench/models/GeoGround/llava-v1.5-7b-task-lora-geoground")
        )
    return base


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
            except Exception as exc:
                if not is_cuda_oom_error(exc):
                    raise
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


def resolve_runner_device(runner: Any) -> torch.device:
    impl = getattr(runner, "impl", None)
    if impl is not None:
        runner = impl

    captioner = getattr(runner, "captioner", None)
    if captioner is not None:
        model = getattr(captioner, "model", None)
        if model is not None and hasattr(model, "device"):
            return torch.device(str(model.device))

    model = getattr(runner, "model", None)
    if model is not None and hasattr(model, "device"):
        return torch.device(str(model.device))

    runner_device = getattr(runner, "device", None)
    if runner_device is not None:
        return torch.device(str(runner_device))

    return torch.device("cuda:0")


def is_cuda_oom_error(exc: BaseException) -> bool:
    torch_oom = getattr(torch, "OutOfMemoryError", None)
    if torch_oom is not None and isinstance(exc, torch_oom):
        return True
    cuda_oom = getattr(torch.cuda, "OutOfMemoryError", None)
    if cuda_oom is not None and isinstance(exc, cuda_oom):
        return True
    return isinstance(exc, RuntimeError) and "out of memory" in str(exc).lower()


def run_caption_dataset(
    *,
    runner: CaptionRunner,
    spec: CaptionDatasetSpec,
    output_dir: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    rows = load_caption_samples(spec, max_samples=int(args.caption_samples))
    rows = select_rows_for_shard(
        rows,
        world_size=int(args.shard_world_size),
        rank=int(args.shard_rank),
        weights=str(args.shard_weights),
    )
    output_path = dataset_output_path(
        output_dir,
        dataset_name=spec.name,
        shard_suffix=shard_file_suffix(world_size=int(args.shard_world_size), rank=int(args.shard_rank)),
    )
    done_ids = {str(row.get("sample_id", "")) for row in read_jsonl(output_path)}
    pending = [row for row in rows if str(row["sample_id"]) not in done_ids]
    prompt = str(args.caption_prompt)
    current_batch_size = max(1, int(args.batch_size))
    samples_preview: list[dict[str, Any]] = []
    progress = tqdm(total=len(pending), desc=f"{spec.name}:caption", leave=False)
    index = 0

    while index < len(pending):
        chunk = pending[index : index + current_batch_size]
        image_paths = [Path(item["image_path"]) for item in chunk]
        prompts = [prompt] * len(chunk)
        try:
            outputs = runner.generate_batch_prompts(image_paths=image_paths, prompts=prompts)
        except torch.cuda.OutOfMemoryError:
            gc.collect()
            if current_batch_size <= 1:
                raise
            current_batch_size = max(1, current_batch_size // 2)
            continue

        meta = runtime_meta(runner)
        for item, output in zip(chunk, outputs):
            row = {
                "dataset": spec.name,
                "task": "caption",
                "sample_id": str(item["sample_id"]),
                "filename": str(item["filename"]),
                "image_path": str(item["image_path"]),
                "prompt": prompt,
                "prediction": str(output.text),
                "refs": list(item["refs"]),
                "generated_token_count": int(output.generated_token_count),
                "generation_ended_by_eos": bool(output.ended_by_eos),
                "generation_last_token_id": output.last_generated_token_id,
                **meta,
            }
            append_jsonl(output_path, row)
            if len(samples_preview) < 5:
                samples_preview.append(
                    {
                        "filename": row["filename"],
                        "prediction": row["prediction"],
                        "first_ref": row["refs"][0] if row["refs"] else "",
                    }
                )
        progress.update(len(chunk))
        index += len(chunk)

    progress.close()
    final_rows = [row for row in read_jsonl(output_path) if str(row.get("dataset", "")) == spec.name]
    return {
        "dataset": spec.name,
        "task": "caption",
        "num_rows": len(final_rows),
        "num_pending_before_run": len(pending),
        "prompt": prompt,
        "sample_predictions": samples_preview,
        "output_path": str(output_path),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run VRSBench fine-tuned models on single-task benchmarks.")
    parser.add_argument("--preset", type=str, default="")
    parser.add_argument("--model-family", type=str, default="")
    parser.add_argument("--merger-ckpt", type=str, default="")
    parser.add_argument("--preset-name", type=str, default="")
    parser.add_argument("--datasets", type=str, default="all")
    parser.add_argument("--output-dir", type=str, default="")
    parser.add_argument("--caption-samples", type=int, default=0)
    parser.add_argument("--caption-prompt", type=str, default=DEFAULT_CAPTION_PROMPT)
    parser.add_argument("--caption-max-new-tokens", type=int, default=48)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--device-map", type=str, default="cuda:0")
    parser.add_argument("--cuda-reserve-free-gb", type=float, default=None)
    parser.add_argument("--dtype", type=str, default="")
    parser.add_argument("--shard-world-size", type=int, default=1)
    parser.add_argument("--shard-rank", type=int, default=0)
    parser.add_argument("--shard-weights", type=str, default="")
    parser.add_argument("--do-sample", dest="do_sample", action="store_true", default=None)
    parser.add_argument("--no-sample", dest="do_sample", action="store_false")
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--num-beams", type=int, default=None)
    parser.add_argument("--repetition-penalty", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--merge-lora", action="store_true")
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--smart-resize-min-pixels", type=int, default=None)
    parser.add_argument("--smart-resize-max-pixels", type=int, default=None)
    parser.add_argument("--geochat-model-dir", type=str, default="GeoChat-Bench/model/geochat-7B")
    parser.add_argument("--geochat-code-root", type=str, default="GeoChat-Bench/GeoChat")
    parser.add_argument("--geoground-model-dir", type=str, default="models/GeoGround/llava-v1.5-7b-task-lora-geoground")
    parser.add_argument("--geoground-model-base", type=str, default="")
    parser.add_argument("--llava-code-root", type=str, default="models/GeoGround/LLaVA")
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.cuda_reserve_free_gb is not None and float(args.cuda_reserve_free_gb) < 0:
        raise ValueError(f"--cuda-reserve-free-gb must be >= 0, got {args.cuda_reserve_free_gb}")
    if int(args.shard_world_size) <= 0:
        raise ValueError(f"shard_world_size must be >= 1, got {args.shard_world_size}")
    if int(args.shard_rank) < 0 or int(args.shard_rank) >= int(args.shard_world_size):
        raise ValueError(
            f"shard_rank out of range: rank={args.shard_rank}, world_size={args.shard_world_size}"
        )
    parse_shard_weights(str(args.shard_weights), world_size=int(args.shard_world_size))

    preset_name = str(args.preset).strip()
    if preset_name:
        presets = build_presets()
        if preset_name not in presets:
            raise ValueError(f"unknown preset: {preset_name}")
        preset = presets[preset_name]
    else:
        model_family = str(args.model_family).strip()
        merger_ckpt_raw = str(args.merger_ckpt).strip()
        if not model_family or not merger_ckpt_raw:
            raise ValueError("custom checkpoint mode requires both --model-family and --merger-ckpt")
        if model_family not in {"dinov3", "qwen_native"}:
            raise ValueError(f"unsupported --model-family: {model_family}")
        merger_ckpt = resolve_from_vrsbench(merger_ckpt_raw)
        if not merger_ckpt.is_file():
            raise FileNotFoundError(f"missing merger checkpoint: {merger_ckpt}")
        checkpoint_name = merger_ckpt.parent.name
        run_name = checkpoint_run_root(merger_ckpt).name
        preset_name = str(args.preset_name).strip() or f"{model_family}_{run_name}_{checkpoint_name}"
        preset = load_preset_from_checkpoint(
            name=preset_name,
            model_family=model_family,
            merger_ckpt=merger_ckpt,
        )

    output_dir = resolve_output_dir(args.output_dir, preset_name=preset_name)
    output_dir.mkdir(parents=True, exist_ok=True)

    caption_specs = build_caption_specs()
    selected = parse_datasets(args.datasets)
    unknown = [name for name in selected if name not in caption_specs]
    if unknown:
        raise ValueError(f"unknown datasets: {unknown}")

    summary: dict[str, Any] = {
        "preset": str(preset_name),
        "output_dir": str(output_dir),
        "datasets": selected,
        "caption_prompt": str(args.caption_prompt),
        "caption_max_new_tokens": int(args.caption_max_new_tokens),
        "batch_size": int(args.batch_size),
        "device_map": str(args.device_map),
        "cuda_reserve_free_gb": None if args.cuda_reserve_free_gb is None else float(args.cuda_reserve_free_gb),
        "shard_world_size": int(args.shard_world_size),
        "shard_rank": int(args.shard_rank),
        "shard_weights": str(args.shard_weights),
        "dtype": str(args.dtype if args.dtype else preset.dtype),
        "do_sample": args.do_sample,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "num_beams": args.num_beams,
        "repetition_penalty": args.repetition_penalty,
        "seed": args.seed,
        "caption": {},
    }

    caption_selected = [dataset_name for dataset_name in selected if dataset_name in caption_specs]

    if caption_selected:
        caption_runner = CaptionRunner(
            preset=preset,
            args=args,
            max_new_tokens=int(args.caption_max_new_tokens),
        )
        summary["caption_runtime"] = runtime_meta(caption_runner)
        caption_prealloc = preallocate_cuda_cache(
            device=resolve_runner_device(caption_runner),
            keep_free_gb=None if args.cuda_reserve_free_gb is None else float(args.cuda_reserve_free_gb),
        )
        summary["caption_cuda_prealloc"] = caption_prealloc
        if args.cuda_reserve_free_gb is None:
            print("[INFO] caption cuda_prealloc disabled", flush=True)
        else:
            print(
                "[INFO] caption cuda_prealloc "
                f"keep_free_gb={float(args.cuda_reserve_free_gb):.2f} "
                f"free_before_gb={int(caption_prealloc['free_before_bytes']) / (1024**3):.2f} "
                f"reserved_gb={int(caption_prealloc['reserved_bytes']) / (1024**3):.2f} "
                f"free_after_gb={int(caption_prealloc['free_after_bytes']) / (1024**3):.2f}",
                flush=True,
            )
        for dataset_name in caption_selected:
            dataset_summary = run_caption_dataset(
                runner=caption_runner,
                spec=caption_specs[dataset_name],
                output_dir=output_dir,
                args=args,
            )
            summary["caption"][dataset_name] = dataset_summary
        del caption_runner
        gc.collect()

    summary_path = output_dir / f"generation_summary{shard_file_suffix(world_size=int(args.shard_world_size), rank=int(args.shard_rank))}.json"
    write_json(summary_path, summary)
    print(f"[OK] Wrote generation summary: {summary_path}", flush=True)


if __name__ == "__main__":
    main()
