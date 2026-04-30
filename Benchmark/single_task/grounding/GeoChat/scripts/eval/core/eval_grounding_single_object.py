from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from tqdm import tqdm

SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[5]
VRSBENCH_ROOT = SCRIPT_PATH.parents[4]
VRSBENCH_SRC = VRSBENCH_ROOT / "src"
if str(VRSBENCH_SRC) not in sys.path:
    sys.path.insert(0, str(VRSBENCH_SRC))

from ftqwen3.dinov3_captioner import DinoV3Captioner
from ftqwen3.qwen3_vl_native_captioner import Qwen3VLNativeCaptioner


FAMILY_CHOICES = ("dinov3", "qwen_native", "qwen3vl_base", "geochat", "geoground")
ANGLE_BOX_PATTERN = re.compile(
    r"\{<(-?\d+(?:\.\d+)?)><(-?\d+(?:\.\d+)?)><(-?\d+(?:\.\d+)?)><(-?\d+(?:\.\d+)?)>\|<(-?\d+(?:\.\d+)?)>}"
)
FLOAT_ARRAY_PATTERN = re.compile(
    r"\[\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\]"
)
FLOAT_ARRAY5_PATTERN = re.compile(
    r"\[\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\]"
)
BOUNDING_BOX_SIZE = 100.0
REFER_JSON_STYLE_HINT = (
    " Output strict JSON only. "
    'If the query refers to one target, return {"instance": "short target description", "bbox_2d": [x0, y0, x1, y1]}. '
    'If the query refers to multiple targets, return a JSON array, and each element must use this schema: {"instance": "short target description", "bbox_2d": [x0, y0, x1, y1]}. '
    "Prefer bbox_2d in 0..1000 normalized coordinates. "
    "If uncertain, return an empty bbox array or an empty JSON array."
)
QWEN_TAGGED_POLYGON_STYLE_HINT = (
    " Output strict JSON only. "
    'If the query refers to one target, return exactly one JSON object in this format: {"instance": "short target description", "bbox_2d": [[x0, y0], [x1, y1], [x2, y2], [x3, y3]]}. '
    'If the query refers to multiple targets, return a JSON array where each element corresponds to exactly one target, and each element must use this schema: {"instance": "short target description", "bbox_2d": [[x0, y0], [x1, y1], [x2, y2], [x3, y3]]}. '
    'Example for two targets: [{"instance": "target 1", "bbox_2d": [[x0, y0], [x1, y1], [x2, y2], [x3, y3]]}, {"instance": "target 2", "bbox_2d": [[x4, y4], [x5, y5], [x6, y6], [x7, y7]]}]. '
    "Each bbox_2d must contain exactly one polygon with exactly 4 corner points ordered either clockwise or counterclockwise. "
    "Never put multiple targets or multiple polygons inside one bbox_2d field, never repeat the bbox_2d key, and never output [x0, y0, x1, y1]. "
    "Prefer bbox_2d in 0..1000 normalized coordinates. "
    "If uncertain, return an empty bbox array or an empty JSON array."
)
OBB_BOX_FORMAT_HINT = (
    " Return only one oriented bounding box in this exact format: {<x0><y0><x1><y1>|<angle>}."
    " Use integers in [0,100] for coordinates and an integer angle."
    " Do not output JSON, descriptions, or explanations."
)


class CaptionerRunner:
    def __init__(self, captioner: DinoV3Captioner | Qwen3VLNativeCaptioner) -> None:
        self.captioner = captioner

    def generate_batch(self, *, image_paths: list[Path], prompts: list[str]) -> list[str]:
        results = self.captioner.caption_batch_prompts(image_paths=image_paths, prompts=prompts)
        return [str(item.text).strip() for item in results]


class GeoChatRunner:
    def __init__(
        self,
        *,
        code_root: Path,
        model_path: Path,
        device: str,
        conv_mode: str,
        max_new_tokens: int,
        temperature: float,
        top_p: float | None,
        num_beams: int,
    ) -> None:
        code_root = Path(code_root).resolve()
        if str(code_root) not in sys.path:
            sys.path.insert(0, str(code_root))

        from geochat.constants import DEFAULT_IMAGE_TOKEN, DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN, IMAGE_TOKEN_INDEX
        from geochat.conversation import SeparatorStyle, conv_templates
        from geochat.mm_utils import get_model_name_from_path, tokenizer_image_token
        from geochat.model.builder import load_pretrained_model
        from geochat.utils import disable_torch_init

        disable_torch_init()
        model_name = get_model_name_from_path(str(model_path))
        tokenizer, model, image_processor, _context_len = load_pretrained_model(
            str(model_path),
            None,
            model_name,
            device_map=str(device),
            device=str(device).split(":", 1)[0],
        )

        self.tokenizer = tokenizer
        self.model = model.eval()
        self.image_processor = image_processor
        self.device = torch.device(str(device))
        self.conv_mode = str(conv_mode)
        self.max_new_tokens = int(max_new_tokens)
        self.temperature = float(temperature)
        self.top_p = None if top_p is None else float(top_p)
        self.num_beams = int(num_beams)
        self.default_image_token = str(DEFAULT_IMAGE_TOKEN)
        self.default_im_start_token = str(DEFAULT_IM_START_TOKEN)
        self.default_im_end_token = str(DEFAULT_IM_END_TOKEN)
        self.image_token_index = int(IMAGE_TOKEN_INDEX)
        self.separator_style_two = SeparatorStyle.TWO
        self.conv_templates = conv_templates
        self.tokenizer_image_token = tokenizer_image_token

    def generate_batch(self, *, image_paths: list[Path], prompts: list[str]) -> list[str]:
        if not image_paths:
            return []
        if len(image_paths) != len(prompts):
            raise ValueError(f"image_paths and prompts must have same length, got {len(image_paths)} vs {len(prompts)}")

        input_tensors: list[torch.Tensor] = []
        pil_images: list[Image.Image] = []
        stop_str: str | None = None
        for image_path, prompt in zip(image_paths, prompts):
            if self.model.config.mm_use_im_start_end:
                full_prompt = self.default_im_start_token + self.default_image_token + self.default_im_end_token + "\n" + str(prompt)
            else:
                full_prompt = self.default_image_token + "\n" + str(prompt)

            conv = self.conv_templates[self.conv_mode].copy()
            conv.append_message(conv.roles[0], full_prompt)
            conv.append_message(conv.roles[1], None)
            prompt_text = conv.get_prompt()
            stop_str = conv.sep if conv.sep_style != self.separator_style_two else conv.sep2

            input_ids = self.tokenizer_image_token(
                prompt_text,
                self.tokenizer,
                self.image_token_index,
                return_tensors="pt",
            ).unsqueeze(0).to(self.device)
            input_tensors.append(input_ids)

            with Image.open(str(image_path)) as image:
                pil_images.append(image.convert("RGB"))

        max_length = max(int(tensor.shape[1]) for tensor in input_tensors)
        padded_tensors = []
        for tensor in input_tensors:
            pad_len = int(max_length - tensor.shape[1])
            if pad_len > 0:
                pad = torch.zeros((1, pad_len), dtype=tensor.dtype, device=tensor.device)
                padded = torch.cat((pad, tensor), dim=1)
            else:
                padded = tensor
            padded_tensors.append(padded)
        final_input_tensors = torch.cat(padded_tensors, dim=0)
        image_tensor_batch = self.image_processor.preprocess(
            pil_images,
            crop_size={"height": 504, "width": 504},
            size={"shortest_edge": 504},
            return_tensors="pt",
        )["pixel_values"].half().to(self.device)

        with torch.inference_mode():
            output_ids = self.model.generate(
                final_input_tensors,
                images=image_tensor_batch,
                do_sample=False,
                temperature=self.temperature,
                top_p=self.top_p,
                num_beams=self.num_beams,
                max_new_tokens=self.max_new_tokens,
                length_penalty=2.0,
                use_cache=True,
            )

        input_token_len = int(final_input_tensors.shape[1])
        outputs = self.tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)
        cleaned: list[str] = []
        for output in outputs:
            text = str(output).strip()
            if stop_str and text.endswith(stop_str):
                text = text[: -len(stop_str)]
            cleaned.append(text.strip())
        return cleaned


class GeoGroundRunner:
    def __init__(
        self,
        *,
        llava_code_root: Path | None,
        model_path: Path,
        model_base: Path | None,
        device: str,
        conv_mode: str,
        max_new_tokens: int,
        temperature: float,
        top_p: float | None,
        num_beams: int,
    ) -> None:
        if llava_code_root is not None:
            llava_code_root = Path(llava_code_root).resolve()
            if str(llava_code_root) not in sys.path:
                sys.path.insert(0, str(llava_code_root))

        try:
            from llava.constants import DEFAULT_IMAGE_TOKEN, DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN, IMAGE_TOKEN_INDEX
            from llava.conversation import SeparatorStyle, conv_templates
            from llava.mm_utils import get_model_name_from_path, tokenizer_image_token
            from llava.model.builder import load_pretrained_model
            from llava.utils import disable_torch_init
        except Exception as exc:
            raise RuntimeError(
                "GeoGround requires LLaVA runtime code. Current environment cannot import `llava`. "
                "Provide --llava-code-root or install the matching LLaVA package."
            ) from exc

        disable_torch_init()
        model_name = get_model_name_from_path(str(model_path))
        base_path = str(model_base) if model_base is not None else None
        try:
            tokenizer, model, image_processor, _context_len = load_pretrained_model(
                str(model_path),
                base_path,
                model_name,
                device_map=str(device),
            )
        except TypeError:
            tokenizer, model, image_processor, _context_len = load_pretrained_model(str(model_path), base_path, model_name)

        self.tokenizer = tokenizer
        self.model = model.eval()
        self.image_processor = image_processor
        self.device = torch.device(str(device))
        self.conv_mode = str(conv_mode)
        self.max_new_tokens = int(max_new_tokens)
        self.temperature = float(temperature)
        self.top_p = None if top_p is None else float(top_p)
        self.num_beams = int(num_beams)
        self.default_image_token = str(DEFAULT_IMAGE_TOKEN)
        self.default_im_start_token = str(DEFAULT_IM_START_TOKEN)
        self.default_im_end_token = str(DEFAULT_IM_END_TOKEN)
        self.image_token_index = int(IMAGE_TOKEN_INDEX)
        self.separator_style_two = SeparatorStyle.TWO
        self.conv_templates = conv_templates
        self.tokenizer_image_token = tokenizer_image_token

    def generate_batch(self, *, image_paths: list[Path], prompts: list[str]) -> list[str]:
        if not image_paths:
            return []
        if len(image_paths) != len(prompts):
            raise ValueError(f"image_paths and prompts must have same length, got {len(image_paths)} vs {len(prompts)}")

        input_tensors: list[torch.Tensor] = []
        pil_images: list[Image.Image] = []
        stop_str: str | None = None
        for image_path, prompt in zip(image_paths, prompts):
            if self.model.config.mm_use_im_start_end:
                full_prompt = self.default_im_start_token + self.default_image_token + self.default_im_end_token + "\n" + str(prompt)
            else:
                full_prompt = self.default_image_token + "\n" + str(prompt)

            conv = self.conv_templates[self.conv_mode].copy()
            conv.append_message(conv.roles[0], full_prompt)
            conv.append_message(conv.roles[1], None)
            prompt_text = conv.get_prompt()
            stop_str = conv.sep if conv.sep_style != self.separator_style_two else conv.sep2

            input_ids = self.tokenizer_image_token(
                prompt_text,
                self.tokenizer,
                self.image_token_index,
                return_tensors="pt",
            ).unsqueeze(0).to(self.device)
            input_tensors.append(input_ids)

            with Image.open(str(image_path)) as image:
                pil_images.append(image.convert("RGB").resize((336, 336)))

        max_length = max(int(tensor.shape[1]) for tensor in input_tensors)
        padded_tensors = []
        for tensor in input_tensors:
            pad_len = int(max_length - tensor.shape[1])
            if pad_len > 0:
                pad = torch.zeros((1, pad_len), dtype=tensor.dtype, device=tensor.device)
                padded = torch.cat((pad, tensor), dim=1)
            else:
                padded = tensor
            padded_tensors.append(padded)
        final_input_tensors = torch.cat(padded_tensors, dim=0)
        image_tensor_batch = self.image_processor.preprocess(
            pil_images,
            crop_size={"height": 336, "width": 336},
            size={"shortest_edge": 336},
            return_tensors="pt",
        )["pixel_values"].half().to(self.device)

        with torch.inference_mode():
            output_ids = self.model.generate(
                final_input_tensors,
                images=image_tensor_batch,
                do_sample=False,
                temperature=self.temperature,
                top_p=self.top_p,
                num_beams=self.num_beams,
                max_new_tokens=self.max_new_tokens,
                length_penalty=2.0,
                use_cache=True,
            )

        outputs = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        cleaned: list[str] = []
        for output in outputs:
            text = str(output).strip()
            if stop_str and text.endswith(stop_str):
                text = text[: -len(stop_str)]
            cleaned.append(text.strip())
        return cleaned


def _resolve_from_project(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return (PROJECT_ROOT / candidate).resolve()


def _rel_to_project(path: Path) -> str:
    try:
        return str(Path(path).resolve().relative_to(PROJECT_ROOT))
    except Exception:
        return str(Path(path).resolve())


def _default_data_json() -> str:
    return "fine-tune-qwen3-vl/Benchmark/single_task/grounding/GeoChat/data/test.json"


def _default_qwen_model_dir() -> str:
    return "VRSBench/models/Qwen3-VL-8B-Instruct"


def _default_dinov3_dir() -> str:
    return "VRSBench/models/dinov3/dinov3-vitl16-pretrain-sat493m"


def _default_geochat_model_dir() -> str:
    return "GeoChat-Bench/model/geochat-7B"


def _default_geochat_code_root() -> str:
    return "GeoChat-Bench/GeoChat"


def _default_geoground_model_dir() -> str:
    return "VRSBench/models/GeoGround/llava-v1.5-7b-task-lora-geoground"


def _default_llava_code_root() -> str:
    return "VRSBench/models/GeoGround/LLaVA"


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            raw = line.strip()
            if not raw:
                continue
            rows.append(json.loads(raw))
    return rows


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


def parse_shard_weights(weights: str, *, world_size: int) -> list[int] | None:
    raw = str(weights).strip()
    if not raw:
        return None
    values = [int(part.strip()) for part in raw.split(":") if part.strip()]
    if len(values) != int(world_size):
        raise ValueError(f"shard_weights expects {world_size} values, got {len(values)}: {weights}")
    if any(value <= 0 for value in values):
        raise ValueError(f"shard_weights must be positive integers: {weights}")
    return values


def slice_by_shard(items: list[dict[str, Any]], *, world_size: int, rank: int, weights: str, key_name: str) -> list[dict[str, Any]]:
    if int(world_size) <= 0:
        raise ValueError(f"shard_world_size must be >= 1, got {world_size}")
    if int(rank) < 0 or int(rank) >= int(world_size):
        raise ValueError(f"shard_rank out of range: rank={rank}, world_size={world_size}")

    parsed = parse_shard_weights(weights, world_size=int(world_size))
    total = len(items)
    if parsed is None:
        shard = [item for index, item in enumerate(items) if (index % int(world_size)) == int(rank)]
    else:
        denom = int(sum(parsed))
        left = int(sum(parsed[: int(rank)]))
        right = int(sum(parsed[: int(rank) + 1]))
        start = (total * left) // denom
        end = (total * right) // denom
        shard = items[start:end]

    first_key = shard[0].get(key_name, "") if shard else ""
    last_key = shard[-1].get(key_name, "") if shard else ""
    print(
        f"[INFO] shard rank={rank}/{world_size} weights={weights or 'even'} "
        f"selected={len(shard)}/{total} first_{key_name}={first_key} last_{key_name}={last_key}",
        flush=True,
    )
    return shard


def _normalize_polygon(points: Any) -> list[list[float]] | None:
    if not isinstance(points, list) or len(points) < 3:
        return None
    output: list[list[float]] = []
    for point in points:
        if not isinstance(point, list) or len(point) != 2:
            return None
        try:
            output.append([float(point[0]), float(point[1])])
        except Exception:
            return None
    center_x = sum(point[0] for point in output) / float(len(output))
    center_y = sum(point[1] for point in output) / float(len(output))
    ordered = sorted(output, key=lambda point: math.atan2(point[1] - center_y, point[0] - center_x))
    area = 0.0
    for index, point in enumerate(ordered):
        nxt = ordered[(index + 1) % len(ordered)]
        area += (point[0] * nxt[1]) - (nxt[0] * point[1])
    if area < 0.0:
        ordered.reverse()
    return [[float(point[0]), float(point[1])] for point in ordered]


def _polygon_area(points: list[list[float]]) -> float:
    area = 0.0
    for index, point in enumerate(points):
        nxt = points[(index + 1) % len(points)]
        area += (float(point[0]) * float(nxt[1])) - (float(nxt[0]) * float(point[1]))
    return abs(float(area / 2.0))


def _cross(a: list[float], b: list[float], c: list[float]) -> float:
    return (float(b[0]) - float(a[0])) * (float(c[1]) - float(a[1])) - (float(b[1]) - float(a[1])) * (
        float(c[0]) - float(a[0])
    )


def _inside(point: list[float], edge_start: list[float], edge_end: list[float]) -> bool:
    return _cross(edge_start, edge_end, point) >= -1e-9


def _segment_intersection(start_a: list[float], end_a: list[float], start_b: list[float], end_b: list[float]) -> list[float]:
    x1, y1 = float(start_a[0]), float(start_a[1])
    x2, y2 = float(end_a[0]), float(end_a[1])
    x3, y3 = float(start_b[0]), float(start_b[1])
    x4, y4 = float(end_b[0]), float(end_b[1])
    denom = ((x1 - x2) * (y3 - y4)) - ((y1 - y2) * (x3 - x4))
    if abs(denom) < 1e-12:
        return [float(x2), float(y2)]
    det_a = (x1 * y2) - (y1 * x2)
    det_b = (x3 * y4) - (y3 * x4)
    point_x = ((det_a * (x3 - x4)) - ((x1 - x2) * det_b)) / denom
    point_y = ((det_a * (y3 - y4)) - ((y1 - y2) * det_b)) / denom
    return [float(point_x), float(point_y)]


def _polygon_clip(subject_polygon: list[list[float]], clip_polygon: list[list[float]]) -> list[list[float]]:
    output = [list(point) for point in subject_polygon]
    clip = _normalize_polygon(clip_polygon)
    if clip is None:
        return []
    for index in range(len(clip)):
        edge_start = clip[index]
        edge_end = clip[(index + 1) % len(clip)]
        input_list = output
        output = []
        if not input_list:
            break
        previous = input_list[-1]
        for current in input_list:
            if _inside(current, edge_start, edge_end):
                if not _inside(previous, edge_start, edge_end):
                    output.append(_segment_intersection(previous, current, edge_start, edge_end))
                output.append(current)
            elif _inside(previous, edge_start, edge_end):
                output.append(_segment_intersection(previous, current, edge_start, edge_end))
            previous = current
    normalized = _normalize_polygon(output)
    return normalized if normalized is not None else []


def compute_polygon_iou(poly_a: list[list[float]], poly_b: list[list[float]]) -> float:
    norm_a = _normalize_polygon(poly_a)
    norm_b = _normalize_polygon(poly_b)
    if norm_a is None or norm_b is None:
        return 0.0
    area_a = _polygon_area(norm_a)
    area_b = _polygon_area(norm_b)
    if area_a <= 0.0 or area_b <= 0.0:
        return 0.0
    inter_poly = _polygon_clip(norm_a, norm_b)
    inter_area = _polygon_area(inter_poly) if len(inter_poly) >= 3 else 0.0
    union_area = area_a + area_b - inter_area
    if union_area <= 0.0:
        return 0.0
    return float(inter_area / union_area)


def _gt_polygons(row: dict[str, Any]) -> list[list[list[float]]]:
    ground_truth = row.get("ground_truth", [])
    output: list[list[list[float]]] = []
    if not isinstance(ground_truth, list):
        return output
    for item in ground_truth:
        polygon = _normalize_polygon(item)
        if polygon is not None:
            output.append(polygon)
    return output


def _clamp_pixel_xyxy(box: list[float], width: int, height: int) -> list[float] | None:
    if len(box) != 4:
        return None
    x0, y0, x1, y1 = [float(value) for value in box]
    x0 = max(0.0, min(float(width), x0))
    y0 = max(0.0, min(float(height), y0))
    x1 = max(0.0, min(float(width), x1))
    y1 = max(0.0, min(float(height), y1))
    if x1 <= x0 or y1 <= y0:
        return None
    return [float(x0), float(y0), float(x1), float(y1)]


def _xyxy_to_polygon(box: list[float]) -> list[list[float]] | None:
    if len(box) != 4:
        return None
    x0, y0, x1, y1 = [float(value) for value in box]
    if x1 <= x0 or y1 <= y0:
        return None
    return _normalize_polygon([[x0, y0], [x0, y1], [x1, y1], [x1, y0]])


def _bbox_and_angle_to_polygon(x0: float, y0: float, x1: float, y1: float, angle: float) -> list[list[float]] | None:
    center_x = (float(x0) + float(x1)) / 2.0
    center_y = (float(y0) + float(y1)) / 2.0
    width = abs(float(x1) - float(x0))
    height = abs(float(y1) - float(y0))
    if width <= 0.0 or height <= 0.0:
        return None

    angle_rad = math.radians(float(angle))
    cos_angle = math.cos(angle_rad)
    sin_angle = math.sin(angle_rad)
    corners = [
        (-width / 2.0, -height / 2.0),
        (width / 2.0, -height / 2.0),
        (width / 2.0, height / 2.0),
        (-width / 2.0, height / 2.0),
    ]
    polygon = []
    for dx, dy in corners:
        rot_x = (cos_angle * dx) - (sin_angle * dy) + center_x
        rot_y = (sin_angle * dx) + (cos_angle * dy) + center_y
        polygon.append([float(rot_x), float(rot_y)])
    return _normalize_polygon(polygon)


def _angle_box_100_to_polygon(box: list[float], *, width: int, height: int) -> list[list[float]] | None:
    if len(box) != 5:
        return None
    x0, y0, x1, y1, angle = [float(value) for value in box]
    x0 = max(0.0, min(BOUNDING_BOX_SIZE, x0))
    y0 = max(0.0, min(BOUNDING_BOX_SIZE, y0))
    x1 = max(0.0, min(BOUNDING_BOX_SIZE, x1))
    y1 = max(0.0, min(BOUNDING_BOX_SIZE, y1))
    if x1 <= x0 or y1 <= y0:
        return None
    pixel_box = _clamp_pixel_xyxy(
        [
            x0 * float(width) / BOUNDING_BOX_SIZE,
            y0 * float(height) / BOUNDING_BOX_SIZE,
            x1 * float(width) / BOUNDING_BOX_SIZE,
            y1 * float(height) / BOUNDING_BOX_SIZE,
        ],
        width,
        height,
    )
    if pixel_box is None:
        return None
    return _bbox_and_angle_to_polygon(pixel_box[0], pixel_box[1], pixel_box[2], pixel_box[3], angle)


def _xywha_100_to_polygon(box: list[float], *, width: int, height: int) -> list[list[float]] | None:
    if len(box) != 5:
        return None
    center_x, center_y, box_width, box_height, angle = [float(value) for value in box]
    center_x = max(0.0, min(BOUNDING_BOX_SIZE, center_x))
    center_y = max(0.0, min(BOUNDING_BOX_SIZE, center_y))
    box_width = max(0.0, min(BOUNDING_BOX_SIZE, box_width))
    box_height = max(0.0, min(BOUNDING_BOX_SIZE, box_height))
    if box_width <= 0.0 or box_height <= 0.0:
        return None

    pixel_center_x = center_x * float(width) / BOUNDING_BOX_SIZE
    pixel_center_y = center_y * float(height) / BOUNDING_BOX_SIZE
    pixel_width = box_width * float(width) / BOUNDING_BOX_SIZE
    pixel_height = box_height * float(height) / BOUNDING_BOX_SIZE
    return _bbox_and_angle_to_polygon(
        pixel_center_x - (pixel_width / 2.0),
        pixel_center_y - (pixel_height / 2.0),
        pixel_center_x + (pixel_width / 2.0),
        pixel_center_y + (pixel_height / 2.0),
        angle,
    )


def _bbox2d_1000_to_polygon(box: list[float], *, width: int, height: int) -> list[list[float]] | None:
    if len(box) != 4:
        return None
    x0, y0, x1, y1 = [float(value) for value in box]
    x0 = max(0.0, min(1000.0, x0))
    y0 = max(0.0, min(1000.0, y0))
    x1 = max(0.0, min(1000.0, x1))
    y1 = max(0.0, min(1000.0, y1))
    if x1 <= x0 or y1 <= y0:
        return None
    pixel_box = _clamp_pixel_xyxy(
        [
            x0 * float(width) / 1000.0,
            y0 * float(height) / 1000.0,
            x1 * float(width) / 1000.0,
            y1 * float(height) / 1000.0,
        ],
        width,
        height,
    )
    if pixel_box is None:
        return None
    return _xyxy_to_polygon(pixel_box)


def _scaled_hbb_to_polygon(box: list[float], *, width: int, height: int, scale: float) -> list[list[float]] | None:
    if len(box) != 4 or float(scale) <= 0.0:
        return None
    x0, y0, x1, y1 = [float(value) for value in box]
    pixel_box = _clamp_pixel_xyxy(
        [
            x0 * float(width) / float(scale),
            y0 * float(height) / float(scale),
            x1 * float(width) / float(scale),
            y1 * float(height) / float(scale),
        ],
        width,
        height,
    )
    if pixel_box is None:
        return None
    return _xyxy_to_polygon(pixel_box)


def _strip_code_fence(text: str) -> str:
    raw = str(text or "").strip()
    if not raw.startswith("```"):
        return raw
    raw = re.sub(r"^```[A-Za-z0-9_-]*\s*", "", raw, count=1).strip()
    raw = re.sub(r"\s*```$", "", raw, count=1).strip()
    return raw


def _try_parse_json_value(text: str) -> Any | None:
    raw = str(text or "").strip()
    if not raw:
        return None
    try:
        return json.loads(raw)
    except Exception:
        pass
    for left, right in (("{", "}"), ("[", "]")):
        start = raw.find(left)
        end = raw.rfind(right)
        if start == -1 or end == -1 or end <= start:
            continue
        try:
            return json.loads(raw[start : end + 1])
        except Exception:
            continue
    return None


def _extract_angle_boxes(text: str) -> list[list[float]]:
    output: list[list[float]] = []
    for match in ANGLE_BOX_PATTERN.finditer(_strip_code_fence(str(text or ""))):
        try:
            output.append([float(match.group(index)) for index in range(1, 6)])
        except Exception:
            continue
    return output


def _collect_bbox2d_values(node: Any, output: list[list[float]]) -> None:
    if isinstance(node, dict):
        for key, value in node.items():
            if key == "bbox_2d" and isinstance(value, list) and len(value) == 4:
                try:
                    output.append([float(item) for item in value])
                except Exception:
                    pass
                continue
            _collect_bbox2d_values(value, output)
        return
    if isinstance(node, list):
        for item in node:
            _collect_bbox2d_values(item, output)


def _extract_array_boxes(text: str) -> list[list[float]]:
    raw = _strip_code_fence(str(text or ""))
    output: list[list[float]] = []
    parsed = _try_parse_json_value(raw)
    if parsed is not None:
        if isinstance(parsed, list) and len(parsed) == 4:
            try:
                output.append([float(value) for value in parsed])
            except Exception:
                pass
        elif isinstance(parsed, list) and parsed and all(isinstance(item, list) and len(item) == 4 for item in parsed):
            for item in parsed:
                try:
                    output.append([float(value) for value in item])
                except Exception:
                    continue
        else:
            _collect_bbox2d_values(parsed, output)
    for match in FLOAT_ARRAY_PATTERN.finditer(raw):
        try:
            output.append([float(match.group(index)) for index in range(1, 5)])
        except Exception:
            continue
    return output


def _extract_geoground_obb_boxes(text: str) -> list[list[float]]:
    raw = _strip_code_fence(str(text or ""))
    output: list[list[float]] = []
    for match in FLOAT_ARRAY5_PATTERN.finditer(raw):
        try:
            output.append([float(match.group(index)) for index in range(1, 6)])
        except Exception:
            continue
    return output


def _dedup_polygons(polygons: list[list[list[float]]]) -> list[list[list[float]]]:
    output: list[list[list[float]]] = []
    seen: set[tuple[float, ...]] = set()
    for polygon in polygons:
        normalized = _normalize_polygon(polygon)
        if normalized is None:
            continue
        key = tuple(round(coord, 4) for point in normalized for coord in point)
        if key in seen:
            continue
        seen.add(key)
        output.append(normalized)
    return output


def extract_pred_polygons(
    *,
    family: str,
    text: str,
    width: int,
    height: int,
    geoground_box_scale: float,
) -> list[list[list[float]]]:
    polygons: list[list[list[float]]] = []
    if family == "geoground":
        for box in _extract_geoground_obb_boxes(text):
            polygon = _xywha_100_to_polygon(box, width=width, height=height)
            if polygon is not None:
                polygons.append(polygon)
        if polygons:
            return _dedup_polygons(polygons)

    for box in _extract_angle_boxes(text):
        polygon = _angle_box_100_to_polygon(box, width=width, height=height)
        if polygon is not None:
            polygons.append(polygon)
    if polygons:
        return _dedup_polygons(polygons)

    for box in _extract_array_boxes(text):
        if family == "geoground":
            polygon = _scaled_hbb_to_polygon(box, width=width, height=height, scale=float(geoground_box_scale))
        else:
            polygon = _bbox2d_1000_to_polygon(box, width=width, height=height)
        if polygon is not None:
            polygons.append(polygon)
    return _dedup_polygons(polygons)


def build_prompt(*, family: str, row: dict[str, Any]) -> str:
    prompt = str(row.get("prompt", "")).strip()
    question = str(row.get("question", "")).strip()
    if family == "qwen3vl_base":
        image_name = str(row.get("image_id", row.get("source_image_id", ""))).strip()
        return (
            "You are a visual grounding assistant.\n"
            "Given an image and a referring expression, output strict JSON only.\n"
            f"Referring expression: {question}\n"
            f"Image: {image_name}\n"
            f"{REFER_JSON_STYLE_HINT}"
        )
    if family == "geoground":
        return f"[refer] output the oriented bounding box of the <ref>{question}</ref> in the image.{OBB_BOX_FORMAT_HINT}"
    if family != "geoground":
        if not prompt:
            raise ValueError(f"Missing prompt for question_id={row.get('question_id')}")
        return prompt
    raise ValueError(f"Unsupported family for prompt building: {family}")


def resolve_checkpoint_dir(checkpoint_dir: str | None) -> Path | None:
    if checkpoint_dir is None or not str(checkpoint_dir).strip():
        return None
    path = _resolve_from_project(str(checkpoint_dir))
    if not path.is_dir():
        raise FileNotFoundError(f"Missing checkpoint directory: {path}")
    if not (path / "merger.safetensors").is_file():
        raise FileNotFoundError(f"Missing merger.safetensors under checkpoint directory: {path}")
    return path


def build_runner(args: argparse.Namespace):
    family = str(args.family)
    checkpoint_dir = resolve_checkpoint_dir(args.checkpoint_dir)
    if family == "dinov3":
        if checkpoint_dir is None:
            raise ValueError("--checkpoint-dir is required for family=dinov3")
        captioner = DinoV3Captioner(
            qwen_model_dir=_resolve_from_project(args.qwen_model_dir),
            dinov3_dir=_resolve_from_project(args.dinov3_dir),
            image_size=int(args.image_size),
            smart_resize_min_pixels=int(args.smart_resize_min_pixels),
            smart_resize_max_pixels=int(args.smart_resize_max_pixels),
            merger_ckpt=checkpoint_dir / "merger.safetensors",
            lora_dir=(checkpoint_dir / "lora") if (checkpoint_dir / "lora").is_dir() else None,
            device_map=str(args.device),
            dtype=str(args.dtype),
            max_new_tokens=int(args.max_new_tokens),
            do_sample=False,
            temperature=None,
            top_p=None,
            top_k=None,
            num_beams=1,
            repetition_penalty=None,
            merge_lora=bool(args.merge_lora),
            force_exact_image_size=bool(args.force_exact_image_size),
        )
        return CaptionerRunner(captioner)

    if family == "qwen_native":
        if checkpoint_dir is None:
            raise ValueError("--checkpoint-dir is required for family=qwen_native")
        captioner = Qwen3VLNativeCaptioner(
            _resolve_from_project(args.qwen_model_dir),
            merger_ckpt=checkpoint_dir / "merger.safetensors",
            lora_dir=(checkpoint_dir / "lora") if (checkpoint_dir / "lora").is_dir() else None,
            merge_lora=bool(args.merge_lora),
            device_map=str(args.device),
            dtype=str(args.dtype),
            max_new_tokens=int(args.max_new_tokens),
            do_sample=False,
            temperature=None,
            top_p=None,
            top_k=None,
            num_beams=1,
            repetition_penalty=None,
        )
        return CaptionerRunner(captioner)

    if family == "qwen3vl_base":
        captioner = Qwen3VLNativeCaptioner(
            _resolve_from_project(args.qwen_model_dir),
            merger_ckpt=None,
            lora_dir=None,
            merge_lora=False,
            device_map=str(args.device),
            dtype=str(args.dtype),
            max_new_tokens=int(args.max_new_tokens),
            do_sample=False,
            temperature=None,
            top_p=None,
            top_k=None,
            num_beams=1,
            repetition_penalty=None,
        )
        return CaptionerRunner(captioner)

    if family == "geochat":
        return GeoChatRunner(
            code_root=_resolve_from_project(args.geochat_code_root),
            model_path=_resolve_from_project(args.geochat_model_dir),
            device=str(args.device),
            conv_mode=str(args.conv_mode),
            max_new_tokens=int(args.max_new_tokens),
            temperature=float(args.temperature),
            top_p=None if args.top_p is None else float(args.top_p),
            num_beams=int(args.num_beams),
        )

    if family == "geoground":
        llava_code_root = None if not str(args.llava_code_root).strip() else _resolve_from_project(args.llava_code_root)
        return GeoGroundRunner(
            llava_code_root=llava_code_root,
            model_path=_resolve_from_project(args.geoground_model_dir),
            model_base=_resolve_from_project(args.geoground_model_base) if str(args.geoground_model_base).strip() else None,
            device=str(args.device),
            conv_mode=str(args.conv_mode),
            max_new_tokens=int(args.max_new_tokens),
            temperature=float(args.temperature),
            top_p=None if args.top_p is None else float(args.top_p),
            num_beams=int(args.num_beams),
        )

    raise ValueError(f"Unsupported family: {family}")


def load_eval_rows(*, json_path: Path, max_samples: int) -> list[dict[str, Any]]:
    rows = read_json(json_path)
    if not isinstance(rows, list):
        raise ValueError(f"Dataset json must be a list: {json_path}")
    if max_samples > 0:
        rows = rows[: int(max_samples)]
    output: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        question_id = str(row.get("question_id", "")).strip()
        image_rel_path = str(row.get("image_rel_path", "")).strip()
        prompt = str(row.get("prompt", "")).strip()
        if not question_id or not image_rel_path or not prompt:
            continue
        output.append(row)
    if not output:
        raise ValueError(f"No valid rows loaded from dataset json: {json_path}")
    return output


def load_prediction_rows(*, predictions_jsonl: Path) -> list[dict[str, Any]]:
    rows = read_jsonl(predictions_jsonl)
    if not rows:
        raise ValueError(f"No prediction rows found in: {predictions_jsonl}")
    return rows


def auto_batch_generate(
    *,
    runner,
    family: str,
    rows: list[dict[str, Any]],
    requested_batch_size: int,
    geoground_box_scale: float,
) -> tuple[list[dict[str, Any]], int]:
    outputs: list[dict[str, Any]] = []
    index = 0
    batch_size = max(1, int(requested_batch_size))
    image_size_cache: dict[str, tuple[int, int]] = {}

    progress = tqdm(total=len(rows), desc=f"{family}_generate", leave=True)
    while index < len(rows):
        batch_rows = rows[index : index + batch_size]
        image_paths: list[Path] = []
        prompts: list[str] = []
        try:
            for row in batch_rows:
                image_path = _resolve_from_project(str(row["image_rel_path"]))
                if not image_path.is_file():
                    raise FileNotFoundError(f"Missing image file: {image_path}")
                image_paths.append(image_path)
                prompts.append(build_prompt(family=family, row=row))

            answers = runner.generate_batch(image_paths=image_paths, prompts=prompts)
            if len(answers) != len(batch_rows):
                raise RuntimeError(f"Runner returned {len(answers)} answers for {len(batch_rows)} inputs.")

            for row, image_path, prompt_used, answer_raw in zip(batch_rows, image_paths, prompts, answers):
                image_key = str(image_path)
                if image_key not in image_size_cache:
                    with Image.open(str(image_path)) as image:
                        image_size_cache[image_key] = (int(image.size[0]), int(image.size[1]))
                width, height = image_size_cache[image_key]
                gt_polygons = _gt_polygons(row)
                pred_polygons = extract_pred_polygons(
                    family=family,
                    text=str(answer_raw),
                    width=width,
                    height=height,
                    geoground_box_scale=float(geoground_box_scale),
                )
                best_iou = 0.0
                best_index = -1
                if gt_polygons:
                    for pred_index, pred_polygon in enumerate(pred_polygons):
                        for gt_polygon in gt_polygons:
                            iou = compute_polygon_iou(pred_polygon, gt_polygon)
                            if iou > best_iou:
                                best_iou = float(iou)
                                best_index = int(pred_index)
                outputs.append(
                    {
                        "family": str(family),
                        "question_id": str(row["question_id"]),
                        "task": str(row.get("task", "")),
                        "source_type": str(row.get("source_type", "")),
                        "image_id": str(row.get("image_id", "")),
                        "image_rel_path": str(row.get("image_rel_path", "")),
                        "question": str(row.get("question", "")),
                        "prompt_used": str(prompt_used),
                        "answer_raw": str(answer_raw),
                        "image_width": int(width),
                        "image_height": int(height),
                        "pred_polygon_count": int(len(pred_polygons)),
                        "pred_polygons": pred_polygons,
                        "gt_polygons": gt_polygons,
                        "best_iou": float(best_iou),
                        "best_pred_index": int(best_index),
                    }
                )

            index += len(batch_rows)
            progress.update(len(batch_rows))
        except Exception as exc:
            if not is_cuda_oom_error(exc):
                progress.close()
                raise
            if batch_size <= 1:
                progress.close()
                raise
            batch_size = max(1, int(batch_size // 2))
            continue
    progress.close()
    return outputs, int(batch_size)


def summarize_results(*, family: str, outputs: list[dict[str, Any]]) -> dict[str, Any]:
    buckets: dict[str, dict[str, int | float]] = {
        "all": {"total": 0, "hit50": 0, "hit70": 0, "sum_iou": 0.0, "parse_fail": 0},
        "refer": {"total": 0, "hit50": 0, "hit70": 0, "sum_iou": 0.0, "parse_fail": 0},
    }
    mismatch_examples: list[dict[str, Any]] = []

    for row in outputs:
        task = str(row.get("task", "")).strip().lower()
        if task != "refer":
            raise ValueError(f"Unexpected task in outputs: {task}")
        keys = ["all", "refer"]
        best_iou = float(row.get("best_iou", 0.0))
        pred_polygon_count = int(row.get("pred_polygon_count", 0))
        for key in keys:
            buckets[key]["total"] += 1
            buckets[key]["sum_iou"] += float(best_iou)
            if pred_polygon_count <= 0:
                buckets[key]["parse_fail"] += 1
            if best_iou >= 0.5:
                buckets[key]["hit50"] += 1
            if best_iou >= 0.7:
                buckets[key]["hit70"] += 1
        if best_iou < 0.7 and len(mismatch_examples) < 50:
            mismatch_examples.append(
                {
                    "question_id": str(row.get("question_id", "")),
                    "task": task,
                    "question": str(row.get("question", "")),
                    "answer_raw": str(row.get("answer_raw", "")),
                    "pred_polygon_count": int(pred_polygon_count),
                    "best_iou": float(best_iou),
                }
            )

    metrics: dict[str, dict[str, float | int]] = {}
    for key, stats in buckets.items():
        total = int(stats["total"])
        hit50 = int(stats["hit50"])
        hit70 = int(stats["hit70"])
        sum_iou = float(stats["sum_iou"])
        parse_fail = int(stats["parse_fail"])
        metrics[key] = {
            "total": total,
            "parse_fail": parse_fail,
            "acc@0.5": (float(hit50) * 100.0 / float(total)) if total > 0 else 0.0,
            "acc@0.7": (float(hit70) * 100.0 / float(total)) if total > 0 else 0.0,
            "mean_iou": (float(sum_iou) * 100.0 / float(total)) if total > 0 else 0.0,
            "acc@0.5_count": hit50,
            "acc@0.7_count": hit70,
        }

    return {
        "task": "geochat_single_object_refer",
        "family": str(family),
        "thresholds": [0.5, 0.7],
        "metrics": metrics,
        "mismatch_examples": mismatch_examples,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate single-object grounding on GeoChat-single-object benchmark.")
    parser.add_argument("--family", type=str, required=True, choices=FAMILY_CHOICES)
    parser.add_argument("--data-json", type=str, default=_default_data_json())
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--predictions-out", type=str, default="")
    parser.add_argument("--predictions-jsonl", type=str, default="")
    parser.add_argument("--summary-out", type=str, default="")
    parser.add_argument("--checkpoint-dir", type=str, default="")
    parser.add_argument("--qwen-model-dir", type=str, default=_default_qwen_model_dir())
    parser.add_argument("--dinov3-dir", type=str, default=_default_dinov3_dir())
    parser.add_argument("--geochat-model-dir", type=str, default=_default_geochat_model_dir())
    parser.add_argument("--geochat-code-root", type=str, default=_default_geochat_code_root())
    parser.add_argument("--geoground-model-dir", type=str, default=_default_geoground_model_dir())
    parser.add_argument("--geoground-model-base", type=str, default="")
    parser.add_argument("--llava-code-root", type=str, default=_default_llava_code_root())
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--dtype", type=str, default="bf16")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--num-beams", type=int, default=1)
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--smart-resize-min-pixels", type=int, default=224 * 224)
    parser.add_argument("--smart-resize-max-pixels", type=int, default=512 * 512)
    parser.add_argument("--force-exact-image-size", action="store_true")
    parser.add_argument("--merge-lora", action="store_true")
    parser.add_argument("--geoground-box-scale", type=float, default=1000.0)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--generate-only", action="store_true")
    parser.add_argument("--shard-world-size", type=int, default=1)
    parser.add_argument("--shard-rank", type=int, default=0)
    parser.add_argument("--shard-weights", type=str, default="")
    parser.add_argument("--cuda-reserve-free-gb", type=float, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this evaluation script.")
    if int(args.batch_size) <= 0:
        raise ValueError(f"--batch-size must be > 0, got {args.batch_size}")
    if int(args.max_new_tokens) <= 0:
        raise ValueError(f"--max-new-tokens must be > 0, got {args.max_new_tokens}")
    if float(args.geoground_box_scale) <= 0.0:
        raise ValueError(f"--geoground-box-scale must be > 0, got {args.geoground_box_scale}")
    if args.cuda_reserve_free_gb is not None and float(args.cuda_reserve_free_gb) < 0:
        raise ValueError(f"--cuda-reserve-free-gb must be >= 0, got {args.cuda_reserve_free_gb}")
    if int(args.shard_world_size) <= 0:
        raise ValueError(f"--shard-world-size must be >= 1, got {args.shard_world_size}")
    if int(args.shard_rank) < 0 or int(args.shard_rank) >= int(args.shard_world_size):
        raise ValueError(
            f"--shard-rank must satisfy 0 <= rank < world_size, got rank={args.shard_rank}, world_size={args.shard_world_size}"
        )

    data_json = _resolve_from_project(args.data_json)
    output_dir = _resolve_from_project(args.output_dir)
    predictions_path = (
        _resolve_from_project(args.predictions_out) if str(args.predictions_out).strip() else output_dir / "predictions.jsonl"
    )
    summary_path = _resolve_from_project(args.summary_out) if str(args.summary_out).strip() else output_dir / "evaluation_summary.json"
    config_path = output_dir / "run_config.json"
    final_batch_size: int | None = None

    if str(args.predictions_jsonl).strip():
        predictions = load_prediction_rows(predictions_jsonl=_resolve_from_project(args.predictions_jsonl))
    else:
        rows = load_eval_rows(json_path=data_json, max_samples=int(args.max_samples))
        rows = slice_by_shard(
            rows,
            world_size=int(args.shard_world_size),
            rank=int(args.shard_rank),
            weights=str(args.shard_weights),
            key_name="question_id",
        )
        runner = build_runner(args)
        prealloc_info = preallocate_cuda_cache(
            device=resolve_runner_device(runner),
            keep_free_gb=None if args.cuda_reserve_free_gb is None else float(args.cuda_reserve_free_gb),
        )
        if args.cuda_reserve_free_gb is None:
            print("[INFO] cuda_prealloc disabled", flush=True)
        else:
            print(
                "[INFO] cuda_prealloc "
                f"keep_free_gb={float(args.cuda_reserve_free_gb):.2f} "
                f"free_before_gb={int(prealloc_info['free_before_bytes']) / (1024**3):.2f} "
                f"reserved_gb={int(prealloc_info['reserved_bytes']) / (1024**3):.2f} "
                f"free_after_gb={int(prealloc_info['free_after_bytes']) / (1024**3):.2f}",
                flush=True,
            )
        predictions, final_batch_size = auto_batch_generate(
            runner=runner,
            family=str(args.family),
            rows=rows,
            requested_batch_size=int(args.batch_size),
            geoground_box_scale=float(args.geoground_box_scale),
        )
        write_jsonl(predictions_path, predictions)
        print(f"[OK] Wrote predictions: {predictions_path}", flush=True)

    if not bool(args.generate_only):
        summary = summarize_results(family=str(args.family), outputs=predictions)
        summary["data_json"] = _rel_to_project(data_json)
        summary["predictions"] = _rel_to_project(predictions_path)
        summary["requested_batch_size"] = int(args.batch_size)
        summary["final_batch_size"] = None if final_batch_size is None else int(final_batch_size)
        summary["cuda_reserve_free_gb"] = None if args.cuda_reserve_free_gb is None else float(args.cuda_reserve_free_gb)
        summary["shard_world_size"] = int(args.shard_world_size)
        summary["shard_rank"] = int(args.shard_rank)
        summary["shard_weights"] = str(args.shard_weights)
        write_json(summary_path, summary)
        print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)

    run_config = {
        "family": str(args.family),
        "data_json": _rel_to_project(data_json),
        "output_dir": _rel_to_project(output_dir),
        "predictions_out": _rel_to_project(predictions_path),
        "predictions_jsonl": _rel_to_project(_resolve_from_project(args.predictions_jsonl))
        if str(args.predictions_jsonl).strip()
        else "",
        "summary_out": _rel_to_project(summary_path),
        "checkpoint_dir": _rel_to_project(_resolve_from_project(args.checkpoint_dir))
        if str(args.checkpoint_dir).strip()
        else "",
        "qwen_model_dir": _rel_to_project(_resolve_from_project(args.qwen_model_dir)),
        "dinov3_dir": _rel_to_project(_resolve_from_project(args.dinov3_dir)),
        "geochat_model_dir": _rel_to_project(_resolve_from_project(args.geochat_model_dir)),
        "geochat_code_root": _rel_to_project(_resolve_from_project(args.geochat_code_root)),
        "geoground_model_dir": _rel_to_project(_resolve_from_project(args.geoground_model_dir)),
        "geoground_model_base": _rel_to_project(_resolve_from_project(args.geoground_model_base))
        if str(args.geoground_model_base).strip()
        else "",
        "llava_code_root": _rel_to_project(_resolve_from_project(args.llava_code_root))
        if str(args.llava_code_root).strip()
        else "",
        "device": str(args.device),
        "dtype": str(args.dtype),
        "batch_size": int(args.batch_size),
        "final_batch_size": None if final_batch_size is None else int(final_batch_size),
        "max_new_tokens": int(args.max_new_tokens),
        "temperature": float(args.temperature),
        "top_p": None if args.top_p is None else float(args.top_p),
        "num_beams": int(args.num_beams),
        "conv_mode": str(args.conv_mode),
        "image_size": int(args.image_size),
        "smart_resize_min_pixels": int(args.smart_resize_min_pixels),
        "smart_resize_max_pixels": int(args.smart_resize_max_pixels),
        "force_exact_image_size": bool(args.force_exact_image_size),
        "merge_lora": bool(args.merge_lora),
        "geoground_box_scale": float(args.geoground_box_scale),
        "max_samples": int(args.max_samples),
        "generate_only": bool(args.generate_only),
        "shard_world_size": int(args.shard_world_size),
        "shard_rank": int(args.shard_rank),
        "shard_weights": str(args.shard_weights),
        "cuda_reserve_free_gb": None if args.cuda_reserve_free_gb is None else float(args.cuda_reserve_free_gb),
    }
    write_json(config_path, run_config)


if __name__ == "__main__":
    main()
