from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

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


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def resolve_from_project(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return (PROJECT_ROOT / candidate).resolve()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Visual-feature t-SNE for DINOv3 and Qwen3-VL.")
    parser.add_argument("--mode", type=str, default="all", choices=["extract", "plot", "all"])
    parser.add_argument(
        "--dataset-root",
        "--aid-root",
        dest="dataset_root",
        type=str,
        default="GeoChat-Bench/dataset/raw/AID/test",
    )
    parser.add_argument("--dataset-name", type=str, default="AID")
    parser.add_argument("--file-prefix", type=str, default="aid")
    parser.add_argument("--output-dir", type=str, default="GeoChat-Bench/analysis/aid_tsne")
    parser.add_argument("--qwen-model-dir", type=str, default="VRSBench/models/Qwen3-VL-8B-Instruct")
    parser.add_argument(
        "--dinov3-dir",
        type=str,
        default="VRSBench/models/dinov3/dinov3-vitl16-pretrain-sat493m",
    )
    parser.add_argument("--per-class", type=int, default=50)
    parser.add_argument("--class-limit", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--qwen-device", type=str, default="cuda:0")
    parser.add_argument("--dino-device", type=str, default="cuda:0")
    parser.add_argument("--qwen-dtype", type=str, default="bf16", choices=["fp16", "bf16", "fp32"])
    parser.add_argument("--dino-dtype", type=str, default="fp32", choices=["fp16", "bf16", "fp32"])
    parser.add_argument("--qwen-batch-size", type=int, default=8)
    parser.add_argument("--dino-batch-size", type=int, default=32)
    parser.add_argument("--pool", type=str, default="mean", choices=["mean"])
    parser.add_argument("--pca-dim", type=int, default=50)
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


def collect_classification_samples(
    dataset_root: Path,
    *,
    per_class: int,
    class_limit: int,
    seed: int,
) -> list[dict[str, Any]]:
    if not dataset_root.is_dir():
        raise FileNotFoundError(f"Missing dataset root: {dataset_root}")

    class_dirs = [path for path in sorted(dataset_root.iterdir()) if path.is_dir()]
    if not class_dirs:
        raise ValueError(f"No class directories found under {dataset_root}")
    if int(class_limit) > 0:
        class_dirs = class_dirs[: int(class_limit)]

    rng = random.Random(int(seed))
    rows: list[dict[str, Any]] = []
    for class_index, class_dir in enumerate(class_dirs):
        image_paths = [
            path
            for path in sorted(class_dir.iterdir())
            if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}
        ]
        if not image_paths:
            continue
        if len(image_paths) < int(per_class):
            raise ValueError(
                f"Class {class_dir.name} has only {len(image_paths)} images, smaller than per-class={int(per_class)}."
            )
        selected = list(image_paths)
        rng.shuffle(selected)
        selected = selected[: int(per_class)]
        selected.sort()
        for image_path in selected:
            rows.append(
                {
                    "image_path": str(image_path),
                    "image_relpath": str(image_path.relative_to(dataset_root)),
                    "label_name": str(class_dir.name),
                    "label_index": int(class_index),
                }
            )
    if not rows:
        raise ValueError(f"No images selected from {dataset_root}")
    return rows


def save_manifest(manifest_path: Path, rows: list[dict[str, Any]]) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "num_samples": int(len(rows)),
        "classes": sorted({str(row["label_name"]) for row in rows}),
        "rows": rows,
    }
    manifest_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def load_manifest(manifest_path: Path) -> list[dict[str, Any]]:
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows = payload.get("rows", [])
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"Invalid manifest rows in {manifest_path}")
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


def mean_pool_tokens(tokens: torch.Tensor) -> np.ndarray:
    pooled = tokens.mean(dim=0)
    return pooled.detach().cpu().to(torch.float32).numpy()


class PreMergerHook:
    def __init__(self) -> None:
        self.tensor: torch.Tensor | None = None

    def __call__(self, _module, inputs) -> None:
        if not inputs:
            raise RuntimeError("Missing merger inputs while capturing Qwen visual tokens.")
        self.tensor = inputs[0]


def extract_qwen_features(
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

    outputs: list[np.ndarray] = []
    try:
        for start in range(0, len(rows), int(batch_size)):
            batch_rows = rows[start : start + int(batch_size)]
            images = [load_rgb_image(row["image_path"]) for row in batch_rows]
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
            if len(token_groups) != len(batch_rows):
                raise RuntimeError("Qwen feature count does not match batch size.")
            for tokens in token_groups:
                outputs.append(mean_pool_tokens(tokens))
    finally:
        handle.remove()
        del model

    return np.stack(outputs, axis=0)


def extract_dino_features(
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

    num_register_tokens = int(getattr(model.config, "num_register_tokens", 0))
    patch_start = 1 + num_register_tokens

    outputs: list[np.ndarray] = []
    for start in range(0, len(rows), int(batch_size)):
        batch_rows = rows[start : start + int(batch_size)]
        images = [load_rgb_image(row["image_path"]) for row in batch_rows]
        inputs = processor(images=images, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(target_device, dtype=torch_dtype)
        with torch.no_grad():
            result = model(pixel_values=pixel_values, return_dict=True)
        spatial_tokens = result.last_hidden_state[:, patch_start:, :]
        pooled = spatial_tokens.mean(dim=1).detach().cpu().to(torch.float32).numpy()
        outputs.extend(list(pooled))

    return np.stack(outputs, axis=0)


def l2_normalize(features: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    return features / norms


def run_tsne(features: np.ndarray, *, pca_dim: int, perplexity: float, learning_rate: float, iterations: int, seed: int):
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


def build_color_map(num_classes: int):
    cmap = plt.get_cmap("tab20", max(20, int(num_classes)))
    colors = [cmap(i % cmap.N) for i in range(int(num_classes))]
    if int(num_classes) > 20:
        cmap2 = plt.get_cmap("tab20b", max(20, int(num_classes)))
        colors = []
        for index in range(int(num_classes)):
            if index < 20:
                colors.append(plt.get_cmap("tab20")(index))
            else:
                colors.append(cmap2((index - 20) % cmap2.N))
    return colors


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
    colors = build_color_map(len(class_names))
    fig, axes = plt.subplots(1, 2, figsize=(16, 7), constrained_layout=False)
    panels = [
        ("DINOv3", dino_coords, axes[0]),
        ("SigLIP2", qwen_coords, axes[1]),
    ]
    for title, coords, axis in panels:
        for class_index, class_name in enumerate(class_names):
            mask = labels == int(class_index)
            axis.scatter(
                coords[mask, 0],
                coords[mask, 1],
                s=18,
                alpha=0.85,
                color=colors[class_index],
                edgecolors="none",
                label=class_name,
            )
        axis.set_title(title)
        axis.set_xticks([])
        axis.set_yticks([])
    handles, names = axes[1].get_legend_handles_labels()
    fig.legend(
        handles,
        names,
        loc="center left",
        bbox_to_anchor=(0.99, 0.5),
        fontsize=8,
        frameon=False,
        ncol=1,
    )
    fig.suptitle(f"{dataset_name} visual-feature t-SNE", fontsize=14)
    fig.subplots_adjust(right=0.82, wspace=0.08)
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=int(dpi), bbox_inches="tight")
    fig.savefig(output_svg, bbox_inches="tight")
    plt.close(fig)


def run_extract(args) -> None:
    dataset_root = resolve_from_project(args.dataset_root)
    output_dir = resolve_from_project(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = collect_classification_samples(
        dataset_root,
        per_class=int(args.per_class),
        class_limit=int(args.class_limit),
        seed=int(args.seed),
    )
    manifest_path = output_dir / f"{str(args.file_prefix)}_manifest.json"
    save_manifest(manifest_path, rows)

    qwen_features = extract_qwen_features(
        rows,
        qwen_model_dir=resolve_from_project(args.qwen_model_dir),
        device=str(args.qwen_device),
        dtype_name=str(args.qwen_dtype),
        batch_size=int(args.qwen_batch_size),
    )
    np.save(output_dir / "qwen_siglip2_features.npy", qwen_features)

    dino_features = extract_dino_features(
        rows,
        dinov3_dir=resolve_from_project(args.dinov3_dir),
        device=str(args.dino_device),
        dtype_name=str(args.dino_dtype),
        batch_size=int(args.dino_batch_size),
    )
    np.save(output_dir / "dinov3_features.npy", dino_features)

    summary = {
        "num_samples": int(len(rows)),
        "num_classes": int(len(sorted({str(row['label_name']) for row in rows}))),
        "qwen_feature_shape": list(qwen_features.shape),
        "dino_feature_shape": list(dino_features.shape),
        "qwen_model_dir": str(resolve_from_project(args.qwen_model_dir)),
        "dinov3_dir": str(resolve_from_project(args.dinov3_dir)),
    }
    (output_dir / "extract_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


def run_plot(args) -> None:
    output_dir = resolve_from_project(args.output_dir)
    manifest_path = output_dir / f"{str(args.file_prefix)}_manifest.json"
    dino_path = output_dir / "dinov3_features.npy"
    qwen_path = output_dir / "qwen_siglip2_features.npy"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing manifest: {manifest_path}")
    if not dino_path.is_file():
        raise FileNotFoundError(f"Missing DINO features: {dino_path}")
    if not qwen_path.is_file():
        raise FileNotFoundError(f"Missing Qwen features: {qwen_path}")

    rows = load_manifest(manifest_path)
    labels = np.asarray([int(row["label_index"]) for row in rows], dtype=np.int64)
    class_names = [name for _, name in sorted({(int(row["label_index"]), str(row["label_name"])) for row in rows})]

    dino_features = np.load(dino_path)
    qwen_features = np.load(qwen_path)
    if int(dino_features.shape[0]) != len(rows) or int(qwen_features.shape[0]) != len(rows):
        raise ValueError("Feature row count does not match manifest row count.")

    dino_normalized, dino_pca, dino_coords, dino_perplexity = run_tsne(
        dino_features,
        pca_dim=int(args.pca_dim),
        perplexity=float(args.tsne_perplexity),
        learning_rate=float(args.tsne_learning_rate),
        iterations=int(args.tsne_iterations),
        seed=int(args.seed),
    )
    qwen_normalized, qwen_pca, qwen_coords, qwen_perplexity = run_tsne(
        qwen_features,
        pca_dim=int(args.pca_dim),
        perplexity=float(args.tsne_perplexity),
        learning_rate=float(args.tsne_learning_rate),
        iterations=int(args.tsne_iterations),
        seed=int(args.seed),
    )

    np.save(output_dir / "dinov3_tsne_coords.npy", dino_coords)
    np.save(output_dir / "qwen_siglip2_tsne_coords.npy", qwen_coords)

    plot_tsne_pair(
        dataset_name=str(args.dataset_name),
        dino_coords=dino_coords,
        qwen_coords=qwen_coords,
        labels=labels,
        class_names=class_names,
        output_png=output_dir / f"{str(args.file_prefix)}_tsne_pair.png",
        output_svg=output_dir / f"{str(args.file_prefix)}_tsne_pair.svg",
        dpi=int(args.figure_dpi),
    )

    metrics = {
        "dino": compute_metrics(dino_pca, labels),
        "qwen": compute_metrics(qwen_pca, labels),
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
