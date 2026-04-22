from __future__ import annotations

import argparse
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

try:
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
except Exception:
    plt = None
    Rectangle = None

try:
    from sklearn.decomposition import PCA
except Exception:
    PCA = None

try:
    from tqdm import tqdm
except Exception:
    tqdm = None

try:
    from transformers import AutoImageProcessor, AutoProcessor, Qwen3VLForConditionalGeneration
except Exception:
    AutoImageProcessor = None
    AutoProcessor = None
    Qwen3VLForConditionalGeneration = None

try:
    from transformers.models.dinov3_vit.modeling_dinov3_vit import DINOv3ViTModel
except Exception:
    DINOv3ViTModel = None


PROJECT_ROOT = Path(__file__).resolve().parents[3]


def resolve_from_project(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return (PROJECT_ROOT / candidate).resolve()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Foreground PCA and full-image patch matching comparison for DINOv3 and SigLIP2."
    )
    parser.add_argument(
        "--dataset-jsonl",
        type=str,
        default="benchmark/vrsbench/data/grounding_tsne/vrsbench_referring_tsne_clean_subset.jsonl",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="benchmark/vrsbench/analysis/backbone_foreground_viz/vrsbench_subset3000_seed42",
    )
    parser.add_argument("--qwen-model-dir", type=str, default="models/Qwen3-VL-8B-Instruct")
    parser.add_argument("--dinov3-dir", type=str, default="models/dinov3/dinov3-vitl16-pretrain-sat493m")
    parser.add_argument("--qwen-device", type=str, default="cuda:0")
    parser.add_argument("--dino-device", type=str, default="cuda:1")
    parser.add_argument("--qwen-dtype", type=str, default="bf16", choices=["fp16", "bf16", "fp32"])
    parser.add_argument("--dino-dtype", type=str, default="fp32", choices=["fp16", "bf16", "fp32"])
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--top-k", type=int, default=24)
    parser.add_argument("--contact-sheet-k", type=int, default=12)
    parser.add_argument("--min-box-area-ratio", type=float, default=0.0)
    parser.add_argument("--max-box-area-ratio", type=float, default=1.0)
    parser.add_argument("--min-border-margin-px", type=int, default=0)
    parser.add_argument("--min-dino-gt-tokens", type=int, default=4)
    parser.add_argument("--min-qwen-gt-tokens", type=int, default=9)
    parser.add_argument("--score-weights", type=str, default="1.0,1.0,1.0")
    parser.add_argument("--figure-dpi", type=int, default=240)
    parser.add_argument("--max-question-chars", type=int, default=110)
    return parser


def set_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def torch_dtype_from_name(name: str) -> torch.dtype:
    norm = str(name).strip().lower()
    if norm == "fp16":
        return torch.float16
    if norm == "bf16":
        return torch.bfloat16
    return torch.float32


def parse_score_weights(text: str) -> tuple[float, float, float]:
    parts = [part.strip() for part in str(text).split(",") if part.strip()]
    if len(parts) != 3:
        raise ValueError(f"Expected three score weights, got: {text}")
    values = tuple(float(part) for part in parts)
    if not all(math.isfinite(value) for value in values):
        raise ValueError(f"Invalid score weights: {text}")
    return values


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    if not rows:
        raise ValueError(f"Empty dataset jsonl: {path}")
    return rows


def load_rgb_image(image_path: str | Path) -> Image.Image:
    return Image.open(image_path).convert("RGB")


def split_flat_visual_tokens(flat_tokens: torch.Tensor, token_counts: list[int]) -> list[torch.Tensor]:
    outputs: list[torch.Tensor] = []
    offset = 0
    for count in token_counts:
        next_offset = offset + int(count)
        outputs.append(flat_tokens[offset:next_offset])
        offset = next_offset
    if offset != int(flat_tokens.shape[0]):
        raise RuntimeError(f"Token split mismatch: consumed={offset}, total={int(flat_tokens.shape[0])}")
    return outputs


def qwen_tokens_to_feature_map(tokens: torch.Tensor, grid_thw: torch.Tensor) -> torch.Tensor:
    t_size, h_size, w_size = [int(v) for v in grid_thw.tolist()]
    feature_map = tokens.reshape(t_size, h_size, w_size, tokens.shape[-1])
    if t_size > 1:
        return feature_map.mean(dim=0)
    return feature_map[0]


def build_image_groups(rows: list[dict[str, Any]]) -> tuple[list[str], dict[str, list[dict[str, Any]]]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        row_copy = dict(row)
        row_copy["image_path"] = str(resolve_from_project(str(row["image_path"])))
        groups[str(row_copy["image_path"])].append(row_copy)
    return sorted(groups.keys()), groups


def compute_roi_bounds(gt_xyxy_100: list[int], grid_h: int, grid_w: int) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = [float(value) for value in gt_xyxy_100]
    left = int(np.floor(x0 / 100.0 * grid_w))
    top = int(np.floor(y0 / 100.0 * grid_h))
    right = int(np.ceil(x1 / 100.0 * grid_w))
    bottom = int(np.ceil(y1 / 100.0 * grid_h))

    left = max(0, min(left, grid_w - 1))
    top = max(0, min(top, grid_h - 1))
    right = max(left + 1, min(right, grid_w))
    bottom = max(top + 1, min(bottom, grid_h))
    return left, top, right, bottom


def build_roi_mask(gt_xyxy_100: list[int], grid_h: int, grid_w: int) -> np.ndarray:
    left, top, right, bottom = compute_roi_bounds(gt_xyxy_100, grid_h, grid_w)
    mask = np.zeros((grid_h, grid_w), dtype=bool)
    mask[top:bottom, left:right] = True
    return mask


def cosine_similarity_map(feature_map: torch.Tensor, gt_xyxy_100: list[int]) -> tuple[np.ndarray, np.ndarray, dict[str, float | int]]:
    if feature_map.ndim != 3:
        raise ValueError(f"Expected HWC feature map, got shape={tuple(feature_map.shape)}")

    grid_h, grid_w, channels = feature_map.shape
    mask = build_roi_mask(gt_xyxy_100, grid_h, grid_w)
    flat_mask = mask.reshape(-1)
    if int(flat_mask.sum()) <= 0:
        raise RuntimeError("Empty ROI mask while computing similarity map.")

    roi = feature_map[mask].reshape(-1, channels)
    query = roi.mean(dim=0)

    flat_features = feature_map.reshape(-1, channels).to(torch.float32)
    flat_features = torch.nn.functional.normalize(flat_features, dim=1)
    query = torch.nn.functional.normalize(query.to(torch.float32), dim=0)
    similarity = torch.matmul(flat_features, query).reshape(grid_h, grid_w)
    similarity_np = similarity.detach().cpu().numpy().astype(np.float32)

    flat_sim = similarity_np.reshape(-1)
    inside = flat_sim[flat_mask]
    outside = flat_sim[~flat_mask]
    if outside.size == 0:
        outside = np.asarray([float(np.min(flat_sim))], dtype=np.float32)

    topk = int(max(1, min(int(flat_mask.sum()), int(flat_sim.size))))
    top_indices = np.argpartition(flat_sim, -topk)[-topk:]
    top_precision = float(np.mean(flat_mask[top_indices]))

    shifted = flat_sim - float(np.min(flat_sim))
    shifted_sum = float(np.sum(shifted))
    gt_energy_ratio = 0.0 if shifted_sum <= 1e-12 else float(np.sum(shifted[flat_mask]) / shifted_sum)

    contrast = float(np.mean(inside) - np.mean(outside))
    std = float(np.std(flat_sim))
    contrast_z = float(contrast / max(std, 1e-6))
    peak_inside = float(np.max(inside))
    peak_outside = float(np.max(outside))

    metrics = {
        "grid_h": int(grid_h),
        "grid_w": int(grid_w),
        "gt_tokens": int(flat_mask.sum()),
        "inside_mean": float(np.mean(inside)),
        "outside_mean": float(np.mean(outside)),
        "contrast": contrast,
        "contrast_z": contrast_z,
        "topk_precision": top_precision,
        "gt_energy_ratio": gt_energy_ratio,
        "peak_inside": peak_inside,
        "peak_outside": peak_outside,
        "peak_gap": float(peak_inside - peak_outside),
    }
    return similarity_np, mask, metrics


def pca_project_whitened(flat_features: np.ndarray, foreground_features: np.ndarray, seed: int) -> np.ndarray:
    if int(foreground_features.shape[0]) < 3:
        raise RuntimeError("Foreground patch count is too small for PCA projection.")

    if PCA is not None:
        pca = PCA(n_components=3, whiten=True, random_state=int(seed))
        pca.fit(foreground_features)
        return pca.transform(flat_features)

    mean = np.mean(foreground_features, axis=0, keepdims=True)
    centered_fg = foreground_features - mean
    _u, singular_values, vt = np.linalg.svd(centered_fg, full_matrices=False)
    components = vt[:3]
    projected = (flat_features - mean) @ components.T
    denom_base = max(1, int(foreground_features.shape[0]) - 1)
    explained_var = (singular_values[:3] ** 2) / float(denom_base)
    projected = projected / np.sqrt(np.maximum(explained_var, 1e-12))
    return projected


def foreground_pca_rgb_map(feature_map: torch.Tensor, fg_mask: np.ndarray, seed: int) -> np.ndarray:
    if feature_map.ndim != 3:
        raise ValueError(f"Expected HWC feature map, got shape={tuple(feature_map.shape)}")

    grid_h, grid_w, channels = feature_map.shape
    flat_features = feature_map.reshape(-1, channels).detach().cpu().to(torch.float32).numpy()
    flat_mask = fg_mask.reshape(-1)
    foreground_features = flat_features[flat_mask]
    projected = pca_project_whitened(flat_features, foreground_features, seed=int(seed)).reshape(grid_h, grid_w, 3)

    projected_fg = projected.reshape(-1, 3)[flat_mask]
    rgb = np.zeros_like(projected, dtype=np.float32)
    for channel in range(3):
        values = projected_fg[:, channel]
        low = float(np.percentile(values, 1.0))
        high = float(np.percentile(values, 99.0))
        if high <= low + 1e-8:
            rgb[..., channel] = 0.5
            continue
        rgb[..., channel] = np.clip((projected[..., channel] - low) / (high - low), 0.0, 1.0)
    rgb[~fg_mask] = 0.0
    return rgb


def score_from_metrics(metrics: dict[str, float | int], weights: tuple[float, float, float]) -> float:
    w_contrast, w_precision, w_energy = weights
    return (
        float(w_contrast) * float(metrics["contrast_z"])
        + float(w_precision) * float(metrics["topk_precision"])
        + float(w_energy) * float(metrics["gt_energy_ratio"])
    )


def build_progress(total: int, desc: str):
    if tqdm is None:
        return None
    return tqdm(total=total, desc=desc)


def truncate_text(text: str, limit: int) -> str:
    value = str(text).strip().replace("\n", " ")
    if len(value) <= int(limit):
        return value
    return value[: max(0, int(limit) - 3)].rstrip() + "..."


class PreMergerHook:
    def __init__(self) -> None:
        self.tensor: torch.Tensor | None = None

    def __call__(self, _module, inputs) -> None:
        if not inputs:
            raise RuntimeError("Missing merger inputs while capturing Qwen visual tokens.")
        self.tensor = inputs[0]


def make_record(
    row: dict[str, Any],
    dino_metrics: dict[str, float | int],
    qwen_metrics: dict[str, float | int],
    dino_score: float,
    qwen_score: float,
) -> dict[str, Any]:
    return {
        "qid": int(row["qid"]),
        "image_id": str(row["image_id"]),
        "image_path": str(row["image_path"]),
        "question": str(row["question"]),
        "obj_cls": str(row.get("obj_cls", "")),
        "size_bucket": str(row.get("size_bucket", "")),
        "image_width": int(row["image_width"]),
        "image_height": int(row["image_height"]),
        "box_area_ratio": float(row["box_area_ratio"]),
        "gt_xyxy_100": [int(value) for value in row["gt_xyxy_100"]],
        "gt_xyxy_pixel": [int(value) for value in row["gt_xyxy_pixel"]],
        "dino": {key: (int(value) if isinstance(value, int) else float(value)) for key, value in dino_metrics.items()},
        "qwen": {key: (int(value) if isinstance(value, int) else float(value)) for key, value in qwen_metrics.items()},
        "dino_score": float(dino_score),
        "qwen_score": float(qwen_score),
        "delta_score": float(dino_score - qwen_score),
        "delta_contrast_z": float(dino_metrics["contrast_z"]) - float(qwen_metrics["contrast_z"]),
        "delta_topk_precision": float(dino_metrics["topk_precision"]) - float(qwen_metrics["topk_precision"]),
        "delta_gt_energy_ratio": float(dino_metrics["gt_energy_ratio"]) - float(qwen_metrics["gt_energy_ratio"]),
    }


def is_candidate(
    record: dict[str, Any],
    *,
    min_box_area_ratio: float,
    max_box_area_ratio: float,
    min_border_margin_px: int,
    min_dino_gt_tokens: int,
    min_qwen_gt_tokens: int,
) -> bool:
    ratio = float(record["box_area_ratio"])
    if ratio < float(min_box_area_ratio) or ratio > float(max_box_area_ratio):
        return False
    x0, y0, x1, y1 = [int(value) for value in record["gt_xyxy_pixel"]]
    image_width = int(record["image_width"])
    image_height = int(record["image_height"])
    if (
        x0 < int(min_border_margin_px)
        or y0 < int(min_border_margin_px)
        or (image_width - 1 - x1) < int(min_border_margin_px)
        or (image_height - 1 - y1) < int(min_border_margin_px)
    ):
        return False
    if int(record["dino"]["gt_tokens"]) < int(min_dino_gt_tokens):
        return False
    if int(record["qwen"]["gt_tokens"]) < int(min_qwen_gt_tokens):
        return False
    return True


def load_models(
    *,
    qwen_model_dir: Path,
    dinov3_dir: Path,
    qwen_device: str,
    dino_device: str,
    qwen_dtype_name: str,
    dino_dtype_name: str,
):
    if AutoProcessor is None or Qwen3VLForConditionalGeneration is None:
        raise RuntimeError("Qwen3-VL dependencies are unavailable in the current environment.")
    if AutoImageProcessor is None or DINOv3ViTModel is None:
        raise RuntimeError("DINOv3 dependencies are unavailable in the current environment.")

    qwen_processor = AutoProcessor.from_pretrained(str(qwen_model_dir))
    qwen_dtype = torch_dtype_from_name(qwen_dtype_name)
    try:
        qwen_model = Qwen3VLForConditionalGeneration.from_pretrained(
            str(qwen_model_dir),
            dtype=qwen_dtype,
            device_map=str(qwen_device),
        )
    except TypeError:
        qwen_model = Qwen3VLForConditionalGeneration.from_pretrained(
            str(qwen_model_dir),
            torch_dtype=qwen_dtype,
            device_map=str(qwen_device),
        )
    qwen_model.eval()
    qwen_visual = qwen_model.model.visual
    qwen_hook = PreMergerHook()
    qwen_handle = qwen_visual.merger.register_forward_pre_hook(qwen_hook)

    dino_processor = AutoImageProcessor.from_pretrained(str(dinov3_dir))
    dino_model = DINOv3ViTModel.from_pretrained(str(dinov3_dir))
    dino_dtype = torch_dtype_from_name(dino_dtype_name)
    dino_target_device = torch.device(str(dino_device))
    if dino_dtype == torch.float32:
        dino_model.to(device=dino_target_device)
    else:
        dino_model.to(device=dino_target_device, dtype=dino_dtype)
    dino_model.eval()
    dino_patch_size = int(dino_model.config.patch_size)
    dino_register_tokens = int(getattr(dino_model.config, "num_register_tokens", 0))
    dino_patch_start = 1 + dino_register_tokens

    runtime = {
        "qwen_processor": qwen_processor,
        "qwen_model": qwen_model,
        "qwen_visual": qwen_visual,
        "qwen_hook": qwen_hook,
        "qwen_handle": qwen_handle,
        "dino_processor": dino_processor,
        "dino_model": dino_model,
        "dino_target_device": dino_target_device,
        "dino_dtype": dino_dtype,
        "dino_patch_size": dino_patch_size,
        "dino_patch_start": dino_patch_start,
    }
    return runtime


def unload_models(runtime: dict[str, Any]) -> None:
    runtime["qwen_handle"].remove()
    del runtime["qwen_model"]
    del runtime["dino_model"]


def extract_feature_maps_for_batch(
    batch_paths: list[str],
    *,
    runtime: dict[str, Any],
) -> tuple[list[np.ndarray], list[torch.Tensor], list[torch.Tensor]]:
    images = [load_rgb_image(path) for path in batch_paths]
    image_arrays = [np.asarray(image).copy() for image in images]

    qwen_inputs = runtime["qwen_processor"].image_processor(images=images, return_tensors="pt")
    qwen_pixel_values = qwen_inputs["pixel_values"].to(runtime["qwen_model"].device)
    qwen_image_grid_thw = qwen_inputs["image_grid_thw"].to(runtime["qwen_model"].device)
    runtime["qwen_hook"].tensor = None
    with torch.no_grad():
        _ = runtime["qwen_visual"](qwen_pixel_values, grid_thw=qwen_image_grid_thw)
    if runtime["qwen_hook"].tensor is None:
        raise RuntimeError("Failed to capture Qwen pre-merger visual tokens.")
    qwen_token_counts = [int(torch.prod(item).item()) for item in qwen_image_grid_thw]
    qwen_token_groups = split_flat_visual_tokens(runtime["qwen_hook"].tensor, qwen_token_counts)
    qwen_feature_maps = [
        qwen_tokens_to_feature_map(tokens, grid_thw)
        for tokens, grid_thw in zip(qwen_token_groups, qwen_image_grid_thw)
    ]

    dino_inputs = runtime["dino_processor"](images=images, return_tensors="pt")
    dino_pixel_values = dino_inputs["pixel_values"].to(runtime["dino_target_device"], dtype=runtime["dino_dtype"])
    with torch.no_grad():
        dino_result = runtime["dino_model"](pixel_values=dino_pixel_values, return_dict=True)
    dino_spatial_tokens = dino_result.last_hidden_state[:, runtime["dino_patch_start"] :, :]
    dino_grid_h = int(dino_pixel_values.shape[-2]) // int(runtime["dino_patch_size"])
    dino_grid_w = int(dino_pixel_values.shape[-1]) // int(runtime["dino_patch_size"])
    if int(dino_spatial_tokens.shape[1]) != dino_grid_h * dino_grid_w:
        raise RuntimeError(
            f"Unexpected DINO token count: got={int(dino_spatial_tokens.shape[1])}, expected={dino_grid_h * dino_grid_w}"
        )
    dino_feature_maps = [
        dino_spatial_tokens[index].reshape(dino_grid_h, dino_grid_w, dino_spatial_tokens.shape[-1])
        for index in range(len(batch_paths))
    ]
    return image_arrays, qwen_feature_maps, dino_feature_maps


def collect_scores(
    rows: list[dict[str, Any]],
    *,
    qwen_model_dir: Path,
    dinov3_dir: Path,
    qwen_device: str,
    dino_device: str,
    qwen_dtype_name: str,
    dino_dtype_name: str,
    batch_size: int,
    score_weights: tuple[float, float, float],
) -> list[dict[str, Any]]:
    runtime = load_models(
        qwen_model_dir=qwen_model_dir,
        dinov3_dir=dinov3_dir,
        qwen_device=qwen_device,
        dino_device=dino_device,
        qwen_dtype_name=qwen_dtype_name,
        dino_dtype_name=dino_dtype_name,
    )

    unique_paths, groups = build_image_groups(rows)
    records: list[dict[str, Any]] = []
    progress = build_progress(len(unique_paths), "fgviz:score")
    try:
        for start in range(0, len(unique_paths), int(batch_size)):
            batch_paths = unique_paths[start : start + int(batch_size)]
            _image_arrays, qwen_feature_maps, dino_feature_maps = extract_feature_maps_for_batch(batch_paths, runtime=runtime)
            for image_path, qwen_feature_map, dino_feature_map in zip(batch_paths, qwen_feature_maps, dino_feature_maps):
                for row in groups[image_path]:
                    _qwen_map, _qwen_mask, qwen_metrics = cosine_similarity_map(qwen_feature_map, list(row["gt_xyxy_100"]))
                    _dino_map, _dino_mask, dino_metrics = cosine_similarity_map(dino_feature_map, list(row["gt_xyxy_100"]))
                    qwen_score = score_from_metrics(qwen_metrics, score_weights)
                    dino_score = score_from_metrics(dino_metrics, score_weights)
                    records.append(make_record(row, dino_metrics, qwen_metrics, dino_score, qwen_score))
            if progress is not None:
                progress.update(len(batch_paths))
    finally:
        if progress is not None:
            progress.close()
        unload_models(runtime)

    records.sort(key=lambda item: (float(item["delta_score"]), float(item["dino_score"])), reverse=True)
    return records


def render_candidate_figure(
    *,
    record: dict[str, Any],
    image_rgb: np.ndarray,
    dino_pca_rgb: np.ndarray,
    qwen_pca_rgb: np.ndarray,
    dino_match_map: np.ndarray,
    qwen_match_map: np.ndarray,
    output_path: Path,
    dpi: int,
    max_question_chars: int,
) -> None:
    if plt is None or Rectangle is None:
        raise RuntimeError("matplotlib is unavailable in the current environment.")

    image_h, image_w = image_rgb.shape[:2]
    x0, y0, x1, y1 = [float(value) for value in record["gt_xyxy_pixel"]]
    width = max(1.0, x1 - x0)
    height = max(1.0, y1 - y0)
    question = truncate_text(str(record["question"]), int(max_question_chars))

    combined_match = np.concatenate([dino_match_map.reshape(-1), qwen_match_map.reshape(-1)], axis=0)
    vmin = float(np.percentile(combined_match, 1.0))
    vmax = float(np.percentile(combined_match, 99.0))
    if vmax <= vmin:
        vmin = float(np.min(combined_match))
        vmax = float(np.max(combined_match) + 1e-6)

    fig = plt.figure(figsize=(21.5, 5.1), constrained_layout=False)
    grid = fig.add_gridspec(1, 6, width_ratios=[1.18, 1.0, 1.0, 1.0, 1.0, 0.06], wspace=0.10)
    axes = [fig.add_subplot(grid[0, index]) for index in range(5)]
    cbar_axis = fig.add_subplot(grid[0, 5])

    axes[0].imshow(image_rgb)
    axes[0].add_patch(Rectangle((x0, y0), width, height, fill=False, edgecolor="#00ff9c", linewidth=2.2))
    axes[0].set_title(f"Image\nqid={int(record['qid'])}, cls={record['obj_cls']}", fontsize=11, pad=8)
    axes[0].set_xticks([])
    axes[0].set_yticks([])

    pca_panels = [
        ("DINOv3 FG PCA", dino_pca_rgb, axes[1], record["dino"]),
        ("SigLIP2 FG PCA", qwen_pca_rgb, axes[2], record["qwen"]),
    ]
    for title, rgb_map, axis, metrics in pca_panels:
        axis.set_facecolor("black")
        axis.imshow(
            rgb_map,
            interpolation="bilinear",
            extent=(0, image_w, image_h, 0),
        )
        axis.add_patch(Rectangle((x0, y0), width, height, fill=False, edgecolor="#ffffff", linewidth=1.6))
        axis.set_title(f"{title}\nfg={int(metrics['gt_tokens'])}", fontsize=11, pad=8)
        axis.set_xticks([])
        axis.set_yticks([])

    match_panels = [
        ("DINOv3 Matching", dino_match_map, axes[3], record["dino"], float(record["dino_score"])),
        ("SigLIP2 Matching", qwen_match_map, axes[4], record["qwen"], float(record["qwen_score"])),
    ]
    last_im = None
    for title, heatmap, axis, metrics, score in match_panels:
        axis.imshow(image_rgb)
        last_im = axis.imshow(
            heatmap,
            cmap="turbo",
            alpha=0.56,
            interpolation="bilinear",
            extent=(0, image_w, image_h, 0),
            vmin=vmin,
            vmax=vmax,
        )
        axis.add_patch(Rectangle((x0, y0), width, height, fill=False, edgecolor="#00ff9c", linewidth=2.2))
        axis.set_title(
            (
                f"{title}\n"
                f"s={score:.2f}, cz={float(metrics['contrast_z']):.2f}, tp={float(metrics['topk_precision']):.2f}"
            ),
            fontsize=11,
            pad=8,
        )
        axis.set_xticks([])
        axis.set_yticks([])

    cbar = fig.colorbar(last_im, cax=cbar_axis)
    cbar.ax.set_ylabel("Cosine similarity", rotation=90)
    cbar.ax.tick_params(labelsize=9)
    fig.suptitle(
        question,
        fontsize=14,
        y=0.992,
    )
    fig.text(
        0.5,
        0.935,
        (
            f"Δscore={float(record['delta_score']):+.3f} | "
            f"Δcontrast_z={float(record['delta_contrast_z']):+.3f} | "
            f"Δtopk={float(record['delta_topk_precision']):+.3f} | "
            f"Δenergy={float(record['delta_gt_energy_ratio']):+.3f}"
        ),
        ha="center",
        va="center",
        fontsize=11,
    )
    fig.subplots_adjust(left=0.015, right=0.975, top=0.78, bottom=0.06)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=int(dpi), bbox_inches="tight")
    plt.close(fig)


def build_contact_sheet(image_paths: list[Path], output_path: Path) -> None:
    if plt is None:
        raise RuntimeError("matplotlib is unavailable in the current environment.")
    if not image_paths:
        return

    images = [Image.open(path).convert("RGB") for path in image_paths]
    cols = 2
    rows = int(math.ceil(len(images) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(24, max(5, rows * 5.2)), constrained_layout=False)
    if rows == 1:
        axes = np.asarray(axes).reshape(1, cols)

    for axis in axes.reshape(-1):
        axis.axis("off")

    for index, (image, path) in enumerate(zip(images, image_paths)):
        axis = axes.reshape(-1)[index]
        axis.imshow(np.asarray(image))
        axis.set_title(path.stem, fontsize=8)
        axis.axis("off")

    fig.subplots_adjust(wspace=0.03, hspace=0.16)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def export_top_figures(
    top_records: list[dict[str, Any]],
    *,
    qwen_model_dir: Path,
    dinov3_dir: Path,
    qwen_device: str,
    dino_device: str,
    qwen_dtype_name: str,
    dino_dtype_name: str,
    batch_size: int,
    figures_dir: Path,
    dpi: int,
    max_question_chars: int,
    seed: int,
) -> list[Path]:
    if not top_records:
        return []

    runtime = load_models(
        qwen_model_dir=qwen_model_dir,
        dinov3_dir=dinov3_dir,
        qwen_device=qwen_device,
        dino_device=dino_device,
        qwen_dtype_name=qwen_dtype_name,
        dino_dtype_name=dino_dtype_name,
    )

    qid_to_record = {int(record["qid"]): record for record in top_records}
    qid_to_rank = {int(record["qid"]): index for index, record in enumerate(top_records, start=1)}
    image_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in top_records:
        image_rows[str(record["image_path"])].append(record)
    unique_paths = sorted(image_rows.keys())

    figure_paths: list[Path] = []
    progress = build_progress(len(top_records), "fgviz:plot")
    try:
        for start in range(0, len(unique_paths), int(batch_size)):
            batch_paths = unique_paths[start : start + int(batch_size)]
            image_arrays, qwen_feature_maps, dino_feature_maps = extract_feature_maps_for_batch(batch_paths, runtime=runtime)
            for batch_index, image_path in enumerate(batch_paths):
                image_rgb = image_arrays[batch_index]
                qwen_feature_map = qwen_feature_maps[batch_index]
                dino_feature_map = dino_feature_maps[batch_index]
                for record in image_rows[image_path]:
                    dino_match_map, dino_fg_mask, _dino_metrics = cosine_similarity_map(dino_feature_map, list(record["gt_xyxy_100"]))
                    qwen_match_map, qwen_fg_mask, _qwen_metrics = cosine_similarity_map(qwen_feature_map, list(record["gt_xyxy_100"]))
                    dino_pca_rgb = foreground_pca_rgb_map(dino_feature_map, dino_fg_mask, seed=int(seed))
                    qwen_pca_rgb = foreground_pca_rgb_map(qwen_feature_map, qwen_fg_mask, seed=int(seed))
                    qid = int(record["qid"])
                    rank = int(qid_to_rank[qid])
                    output_path = figures_dir / f"rank{rank:03d}_qid{qid:06d}_delta{float(record['delta_score']):+.3f}.svg"
                    render_candidate_figure(
                        record=qid_to_record[qid],
                        image_rgb=image_rgb,
                        dino_pca_rgb=dino_pca_rgb,
                        qwen_pca_rgb=qwen_pca_rgb,
                        dino_match_map=dino_match_map,
                        qwen_match_map=qwen_match_map,
                        output_path=output_path,
                        dpi=dpi,
                        max_question_chars=max_question_chars,
                    )
                    figure_paths.append(output_path)
                    if progress is not None:
                        progress.update(1)
    finally:
        if progress is not None:
            progress.close()
        unload_models(runtime)

    figure_paths.sort()
    return figure_paths


def main() -> None:
    args = build_parser().parse_args()
    set_seed(int(args.seed))
    score_weights = parse_score_weights(str(args.score_weights))

    dataset_path = resolve_from_project(args.dataset_jsonl)
    output_dir = resolve_from_project(args.output_dir)
    figures_dir = output_dir / "figures"
    rows = load_jsonl(dataset_path)

    records = collect_scores(
        rows,
        qwen_model_dir=resolve_from_project(args.qwen_model_dir),
        dinov3_dir=resolve_from_project(args.dinov3_dir),
        qwen_device=str(args.qwen_device),
        dino_device=str(args.dino_device),
        qwen_dtype_name=str(args.qwen_dtype),
        dino_dtype_name=str(args.dino_dtype),
        batch_size=int(args.batch_size),
        score_weights=score_weights,
    )

    filtered = [
        record
        for record in records
        if is_candidate(
            record,
            min_box_area_ratio=float(args.min_box_area_ratio),
            max_box_area_ratio=float(args.max_box_area_ratio),
            min_border_margin_px=int(args.min_border_margin_px),
            min_dino_gt_tokens=int(args.min_dino_gt_tokens),
            min_qwen_gt_tokens=int(args.min_qwen_gt_tokens),
        )
    ]
    top_records = filtered[: int(args.top_k)]

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "all_scores.json").write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "filtered_top_scores.json").write_text(json.dumps(top_records, ensure_ascii=False, indent=2), encoding="utf-8")

    figure_paths = export_top_figures(
        top_records,
        qwen_model_dir=resolve_from_project(args.qwen_model_dir),
        dinov3_dir=resolve_from_project(args.dinov3_dir),
        qwen_device=str(args.qwen_device),
        dino_device=str(args.dino_device),
        qwen_dtype_name=str(args.qwen_dtype),
        dino_dtype_name=str(args.dino_dtype),
        batch_size=int(args.batch_size),
        figures_dir=figures_dir,
        dpi=int(args.figure_dpi),
        max_question_chars=int(args.max_question_chars),
        seed=int(args.seed),
    )

    summary = {
        "dataset_jsonl": str(dataset_path),
        "num_rows": int(len(rows)),
        "num_ranked": int(len(records)),
        "num_filtered": int(len(filtered)),
        "num_exported": int(len(figure_paths)),
        "score_weights": [float(value) for value in score_weights],
        "filter": {
            "min_box_area_ratio": float(args.min_box_area_ratio),
            "max_box_area_ratio": float(args.max_box_area_ratio),
            "min_border_margin_px": int(args.min_border_margin_px),
            "min_dino_gt_tokens": int(args.min_dino_gt_tokens),
            "min_qwen_gt_tokens": int(args.min_qwen_gt_tokens),
        },
        "top_example": top_records[0] if top_records else None,
        "qwen_model_dir": str(resolve_from_project(args.qwen_model_dir)),
        "dinov3_dir": str(resolve_from_project(args.dinov3_dir)),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
