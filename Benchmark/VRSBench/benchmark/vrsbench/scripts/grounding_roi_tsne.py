from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None

try:
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    from sklearn.metrics import davies_bouldin_score, silhouette_score
except Exception:
    PCA = None
    TSNE = None
    davies_bouldin_score = None
    silhouette_score = None

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
    parser = argparse.ArgumentParser(description="Grounding ROI t-SNE for DINOv3 and Qwen3-VL SigLIP2.")
    parser.add_argument("--mode", type=str, default="all", choices=["extract", "plot", "all"])
    parser.add_argument(
        "--dataset-jsonl",
        type=str,
        default="benchmark/vrsbench/data/grounding_tsne/vrsbench_referring_tsne_clean_subset.jsonl",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="benchmark/vrsbench/analysis/grounding_roi_tsne/vrsbench_referring_subset3000_seed42",
    )
    parser.add_argument("--dataset-name", type=str, default="VRSBench Grounding ROI")
    parser.add_argument("--file-prefix", type=str, default="vrsbench_grounding_roi")
    parser.add_argument("--qwen-model-dir", type=str, default="models/Qwen3-VL-8B-Instruct")
    parser.add_argument("--dinov3-dir", type=str, default="models/dinov3/dinov3-vitl16-pretrain-sat493m")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--qwen-device", type=str, default="cuda:0")
    parser.add_argument("--dino-device", type=str, default="cuda:0")
    parser.add_argument("--qwen-dtype", type=str, default="bf16", choices=["fp16", "bf16", "fp32"])
    parser.add_argument("--dino-dtype", type=str, default="fp32", choices=["fp16", "bf16", "fp32"])
    parser.add_argument("--qwen-batch-size", type=int, default=8)
    parser.add_argument("--dino-batch-size", type=int, default=32)
    parser.add_argument("--pca-dim", type=int, default=50)
    parser.add_argument("--size-bins", type=int, default=3)
    parser.add_argument("--tsne-perplexity", type=float, default=30.0)
    parser.add_argument("--tsne-learning-rate", type=float, default=200.0)
    parser.add_argument("--tsne-iterations", type=int, default=1000)
    parser.add_argument("--figure-dpi", type=int, default=220)
    return parser


def set_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def torch_dtype_from_name(name: str) -> torch.dtype:
    norm = str(name).lower().strip()
    if norm == "fp16":
        return torch.float16
    if norm == "bf16":
        return torch.bfloat16
    return torch.float32


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


def save_manifest(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "num_samples": int(len(rows)),
        "classes": sorted({str(row["size_bucket"]) for row in rows}),
        "rows": rows,
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def load_manifest(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("rows", [])
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"Invalid manifest rows in {path}")
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


def build_image_groups(rows: list[dict[str, Any]]) -> tuple[list[str], dict[str, list[tuple[int, dict[str, Any]]]]]:
    groups: dict[str, list[tuple[int, dict[str, Any]]]] = defaultdict(list)
    for index, row in enumerate(rows):
        image_path = str(resolve_from_project(str(row["image_path"])))
        groups[image_path].append((index, row))
    unique_paths = sorted(groups.keys())
    return unique_paths, groups


def roi_mean_pool_2d(feature_map: torch.Tensor, gt_xyxy_100: list[int]) -> np.ndarray:
    if feature_map.ndim != 3:
        raise ValueError(f"Expected HWC feature map, got shape={tuple(feature_map.shape)}")
    grid_h, grid_w, _ = feature_map.shape
    x0, y0, x1, y1 = [float(v) for v in gt_xyxy_100]

    left = int(np.floor(x0 / 100.0 * grid_w))
    top = int(np.floor(y0 / 100.0 * grid_h))
    right = int(np.ceil(x1 / 100.0 * grid_w))
    bottom = int(np.ceil(y1 / 100.0 * grid_h))

    left = max(0, min(left, grid_w - 1))
    top = max(0, min(top, grid_h - 1))
    right = max(left + 1, min(right, grid_w))
    bottom = max(top + 1, min(bottom, grid_h))

    roi = feature_map[top:bottom, left:right, :]
    pooled = roi.reshape(-1, roi.shape[-1]).mean(dim=0)
    return pooled.detach().cpu().to(torch.float32).numpy()


def qwen_tokens_to_feature_map(tokens: torch.Tensor, grid_thw: torch.Tensor) -> torch.Tensor:
    t_size, h_size, w_size = [int(v) for v in grid_thw.tolist()]
    feature_map = tokens.reshape(t_size, h_size, w_size, tokens.shape[-1])
    if t_size > 1:
        feature_map = feature_map.mean(dim=0)
    else:
        feature_map = feature_map[0]
    return feature_map


class PreMergerHook:
    def __init__(self) -> None:
        self.tensor: torch.Tensor | None = None

    def __call__(self, _module, inputs) -> None:
        if not inputs:
            raise RuntimeError("Missing merger inputs while capturing Qwen visual tokens.")
        self.tensor = inputs[0]


def extract_qwen_roi_features(
    rows: list[dict[str, Any]],
    *,
    qwen_model_dir: Path,
    device: str,
    dtype_name: str,
    batch_size: int,
) -> np.ndarray:
    if AutoProcessor is None or Qwen3VLForConditionalGeneration is None:
        raise RuntimeError("Qwen3-VL dependencies are unavailable in the current environment.")

    processor = AutoProcessor.from_pretrained(str(qwen_model_dir))
    torch_dtype = torch_dtype_from_name(dtype_name)
    try:
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            str(qwen_model_dir),
            dtype=torch_dtype,
            device_map=str(device),
        )
    except TypeError:
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            str(qwen_model_dir),
            torch_dtype=torch_dtype,
            device_map=str(device),
        )
    model.eval()
    visual = model.model.visual
    hook = PreMergerHook()
    handle = visual.merger.register_forward_pre_hook(hook)

    unique_paths, groups = build_image_groups(rows)
    outputs: list[np.ndarray | None] = [None] * len(rows)
    try:
        for start in range(0, len(unique_paths), int(batch_size)):
            batch_paths = unique_paths[start : start + int(batch_size)]
            images = [load_rgb_image(image_path) for image_path in batch_paths]
            inputs = processor.image_processor(images=images, return_tensors="pt")
            pixel_values = inputs["pixel_values"].to(model.device)
            image_grid_thw = inputs["image_grid_thw"].to(model.device)
            hook.tensor = None
            with torch.no_grad():
                _ = visual(pixel_values, grid_thw=image_grid_thw)
            if hook.tensor is None:
                raise RuntimeError("Failed to capture Qwen pre-merger visual tokens.")

            token_counts = [int(torch.prod(item).item()) for item in image_grid_thw]
            token_groups = split_flat_visual_tokens(hook.tensor, token_counts)
            if len(token_groups) != len(batch_paths):
                raise RuntimeError("Qwen feature count does not match image batch size.")

            for image_path, tokens, grid_thw in zip(batch_paths, token_groups, image_grid_thw):
                feature_map = qwen_tokens_to_feature_map(tokens, grid_thw)
                for row_index, row in groups[image_path]:
                    outputs[row_index] = roi_mean_pool_2d(feature_map, list(row["gt_xyxy_100"]))
    finally:
        handle.remove()
        del model

    if any(item is None for item in outputs):
        raise RuntimeError("Missing Qwen ROI features for some rows.")
    return np.stack([item for item in outputs if item is not None], axis=0)


def extract_dino_roi_features(
    rows: list[dict[str, Any]],
    *,
    dinov3_dir: Path,
    device: str,
    dtype_name: str,
    batch_size: int,
) -> np.ndarray:
    if AutoImageProcessor is None or DINOv3ViTModel is None:
        raise RuntimeError("DINOv3 dependencies are unavailable in the current environment.")

    processor = AutoImageProcessor.from_pretrained(str(dinov3_dir))
    model = DINOv3ViTModel.from_pretrained(str(dinov3_dir))
    torch_dtype = torch_dtype_from_name(dtype_name)
    target_device = torch.device(device)
    model.to(device=target_device, dtype=torch_dtype if torch_dtype != torch.float32 else None)
    model.eval()

    patch_size = int(model.config.patch_size)
    num_register_tokens = int(getattr(model.config, "num_register_tokens", 0))
    patch_start = 1 + num_register_tokens

    unique_paths, groups = build_image_groups(rows)
    outputs: list[np.ndarray | None] = [None] * len(rows)
    for start in range(0, len(unique_paths), int(batch_size)):
        batch_paths = unique_paths[start : start + int(batch_size)]
        images = [load_rgb_image(image_path) for image_path in batch_paths]
        inputs = processor(images=images, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(target_device, dtype=torch_dtype)
        with torch.no_grad():
            result = model(pixel_values=pixel_values, return_dict=True)
        spatial_tokens = result.last_hidden_state[:, patch_start:, :]
        grid_h = int(pixel_values.shape[-2]) // patch_size
        grid_w = int(pixel_values.shape[-1]) // patch_size
        if int(spatial_tokens.shape[1]) != grid_h * grid_w:
            raise RuntimeError(
                f"Unexpected DINO token count: got={int(spatial_tokens.shape[1])}, expected={grid_h * grid_w}"
            )

        for batch_index, image_path in enumerate(batch_paths):
            feature_map = spatial_tokens[batch_index].reshape(grid_h, grid_w, spatial_tokens.shape[-1])
            for row_index, row in groups[image_path]:
                outputs[row_index] = roi_mean_pool_2d(feature_map, list(row["gt_xyxy_100"]))

    del model
    if any(item is None for item in outputs):
        raise RuntimeError("Missing DINO ROI features for some rows.")
    return np.stack([item for item in outputs if item is not None], axis=0)


def l2_normalize(features: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    return features / norms


def run_tsne(
    features: np.ndarray,
    *,
    pca_dim: int,
    perplexity: float,
    learning_rate: float,
    iterations: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    if PCA is None or TSNE is None:
        raise RuntimeError("scikit-learn is unavailable in the current environment.")

    normalized = l2_normalize(features)
    effective_pca_dim = min(int(pca_dim), int(normalized.shape[0]), int(normalized.shape[1]))
    if effective_pca_dim < 2:
        raise ValueError(f"PCA dimension is too small after clipping: {effective_pca_dim}")
    pca = PCA(n_components=effective_pca_dim, random_state=int(seed))
    pca_features = pca.fit_transform(normalized)

    max_valid_perplexity = max(2.0, float(len(features) - 1) / 3.0)
    effective_perplexity = min(float(perplexity), max_valid_perplexity)
    try:
        tsne = TSNE(
            n_components=2,
            perplexity=effective_perplexity,
            learning_rate=float(learning_rate),
            max_iter=int(iterations),
            init="pca",
            random_state=int(seed),
        )
    except TypeError:
        tsne = TSNE(
            n_components=2,
            perplexity=effective_perplexity,
            learning_rate=float(learning_rate),
            n_iter=int(iterations),
            init="pca",
            random_state=int(seed),
        )
    coords = tsne.fit_transform(pca_features)
    return normalized, pca_features, coords, effective_perplexity


def compute_metrics(features: np.ndarray, labels: np.ndarray) -> dict[str, float | None]:
    metrics: dict[str, float | None] = {"silhouette": None, "davies_bouldin": None}
    if silhouette_score is None or davies_bouldin_score is None:
        return metrics
    if len(np.unique(labels)) < 2:
        return metrics
    metrics["silhouette"] = float(silhouette_score(features, labels))
    metrics["davies_bouldin"] = float(davies_bouldin_score(features, labels))
    return metrics


def get_size_bin_names(size_bins: int) -> list[str]:
    if int(size_bins) == 3:
        return ["small", "medium", "large"]
    if int(size_bins) == 5:
        return ["very_small", "small", "medium", "large", "very_large"]
    return [f"bin_{index + 1}" for index in range(int(size_bins))]


def assign_size_bin_labels(rows: list[dict[str, Any]], size_bins: int) -> tuple[np.ndarray, list[str], list[float], dict[str, int]]:
    if int(size_bins) < 2:
        raise ValueError(f"size_bins must be >= 2, got {size_bins}")

    ratios = sorted(float(row["box_area_ratio"]) for row in rows)
    names = get_size_bin_names(int(size_bins))
    thresholds: list[float] = []
    for bucket_index in range(1, int(size_bins)):
        boundary_index = min(len(ratios) - 1, max(0, (len(ratios) * bucket_index + int(size_bins) - 1) // int(size_bins) - 1))
        thresholds.append(float(ratios[boundary_index]))

    labels: list[int] = []
    counts = {name: 0 for name in names}
    for row in rows:
        ratio = float(row["box_area_ratio"])
        bucket_index = 0
        while bucket_index < len(thresholds) and ratio > thresholds[bucket_index]:
            bucket_index += 1
        labels.append(int(bucket_index))
        counts[names[bucket_index]] += 1

    return np.asarray(labels, dtype=np.int64), names, thresholds, counts


def build_color_map(class_names: list[str]) -> dict[str, Any]:
    if len(class_names) == 3:
        base = {
            "small": "#d84b3c",
            "medium": "#f2b134",
            "large": "#2f7ed8",
        }
        return {name: base.get(name, "#666666") for name in class_names}
    if len(class_names) == 5:
        base = {
            "very_small": "#b2182b",
            "small": "#ef8a62",
            "medium": "#fddbc7",
            "large": "#67a9cf",
            "very_large": "#2166ac",
        }
        return {name: base.get(name, "#666666") for name in class_names}
    cmap = plt.get_cmap("tab20", max(20, int(len(class_names)))) if plt is not None else None
    if cmap is None:
        return {name: "#666666" for name in class_names}
    return {name: cmap(index % cmap.N) for index, name in enumerate(class_names)}


def plot_tsne_pair(
    *,
    dataset_name: str,
    dino_coords: np.ndarray,
    qwen_coords: np.ndarray,
    labels: np.ndarray,
    class_names: list[str],
    output_png: Path,
    output_svg: Path,
    dpi: int,
) -> None:
    if plt is None:
        raise RuntimeError("matplotlib is unavailable in the current environment.")

    colors = build_color_map(class_names)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6.5), constrained_layout=False)
    panels = [("DINOv3", dino_coords, axes[0]), ("Qwen3-VL SigLIP2", qwen_coords, axes[1])]
    for title, coords, axis in panels:
        for class_index, class_name in enumerate(class_names):
            mask = labels == int(class_index)
            axis.scatter(
                coords[mask, 0],
                coords[mask, 1],
                s=18,
                alpha=0.82,
                color=colors[class_name],
                edgecolors="none",
                label=class_name,
            )
        axis.set_title(title)
        axis.set_xticks([])
        axis.set_yticks([])
    handles, names = axes[1].get_legend_handles_labels()
    fig.legend(handles, names, loc="center left", bbox_to_anchor=(0.99, 0.5), fontsize=9, frameon=False)
    fig.suptitle(f"{dataset_name} t-SNE", fontsize=14)
    fig.subplots_adjust(right=0.84, wspace=0.08)
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=int(dpi), bbox_inches="tight")
    fig.savefig(output_svg, bbox_inches="tight")
    plt.close(fig)


def run_extract(args) -> None:
    dataset_path = resolve_from_project(args.dataset_jsonl)
    output_dir = resolve_from_project(args.output_dir)
    rows = load_jsonl(dataset_path)
    save_manifest(output_dir / f"{str(args.file_prefix)}_manifest.json", rows)

    qwen_features = extract_qwen_roi_features(
        rows,
        qwen_model_dir=resolve_from_project(args.qwen_model_dir),
        device=str(args.qwen_device),
        dtype_name=str(args.qwen_dtype),
        batch_size=int(args.qwen_batch_size),
    )
    np.save(output_dir / "qwen_siglip2_roi_features.npy", qwen_features)

    dino_features = extract_dino_roi_features(
        rows,
        dinov3_dir=resolve_from_project(args.dinov3_dir),
        device=str(args.dino_device),
        dtype_name=str(args.dino_dtype),
        batch_size=int(args.dino_batch_size),
    )
    np.save(output_dir / "dinov3_roi_features.npy", dino_features)

    unique_images = len({str(row["image_path"]) for row in rows})
    summary = {
        "num_samples": int(len(rows)),
        "num_unique_images": int(unique_images),
        "size_bucket_counts": {
            "small": int(sum(1 for row in rows if str(row["size_bucket"]) == "small")),
            "medium": int(sum(1 for row in rows if str(row["size_bucket"]) == "medium")),
            "large": int(sum(1 for row in rows if str(row["size_bucket"]) == "large")),
        },
        "qwen_feature_shape": list(qwen_features.shape),
        "dino_feature_shape": list(dino_features.shape),
        "qwen_model_dir": str(resolve_from_project(args.qwen_model_dir)),
        "dinov3_dir": str(resolve_from_project(args.dinov3_dir)),
    }
    (output_dir / "extract_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


def run_plot(args) -> None:
    output_dir = resolve_from_project(args.output_dir)
    manifest_path = output_dir / f"{str(args.file_prefix)}_manifest.json"
    dino_path = output_dir / "dinov3_roi_features.npy"
    qwen_path = output_dir / "qwen_siglip2_roi_features.npy"
    rows = load_manifest(manifest_path)

    labels, label_order, thresholds, label_counts = assign_size_bin_labels(rows, int(args.size_bins))

    dino_features = np.load(dino_path)
    qwen_features = np.load(qwen_path)
    if int(dino_features.shape[0]) != len(rows) or int(qwen_features.shape[0]) != len(rows):
        raise ValueError("Feature row count does not match manifest row count.")

    _, dino_pca, dino_coords, dino_perplexity = run_tsne(
        dino_features,
        pca_dim=int(args.pca_dim),
        perplexity=float(args.tsne_perplexity),
        learning_rate=float(args.tsne_learning_rate),
        iterations=int(args.tsne_iterations),
        seed=int(args.seed),
    )
    _, qwen_pca, qwen_coords, qwen_perplexity = run_tsne(
        qwen_features,
        pca_dim=int(args.pca_dim),
        perplexity=float(args.tsne_perplexity),
        learning_rate=float(args.tsne_learning_rate),
        iterations=int(args.tsne_iterations),
        seed=int(args.seed),
    )

    np.save(output_dir / "dinov3_roi_tsne_coords.npy", dino_coords)
    np.save(output_dir / "qwen_siglip2_roi_tsne_coords.npy", qwen_coords)

    plot_tsne_pair(
        dataset_name=str(args.dataset_name),
        dino_coords=dino_coords,
        qwen_coords=qwen_coords,
        labels=labels,
        class_names=label_order,
        output_png=output_dir / f"{str(args.file_prefix)}_tsne_pair.png",
        output_svg=output_dir / f"{str(args.file_prefix)}_tsne_pair.svg",
        dpi=int(args.figure_dpi),
    )

    metrics = {
        "dino": compute_metrics(dino_pca, labels),
        "qwen": compute_metrics(qwen_pca, labels),
        "size_binning": {
            "num_bins": int(args.size_bins),
            "class_names": label_order,
            "thresholds": [float(x) for x in thresholds],
            "counts": {key: int(value) for key, value in label_counts.items()},
        },
        "tsne": {
            "pca_dim": int(min(int(args.pca_dim), dino_features.shape[0], dino_features.shape[1])),
            "dino_perplexity": float(dino_perplexity),
            "qwen_perplexity": float(qwen_perplexity),
            "learning_rate": float(args.tsne_learning_rate),
            "iterations": int(args.tsne_iterations),
        },
    }
    (output_dir / "tsne_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    set_seed(int(args.seed))

    mode = str(args.mode)
    if mode in {"extract", "all"}:
        run_extract(args)
    if mode in {"plot", "all"}:
        run_plot(args)


if __name__ == "__main__":
    main()
