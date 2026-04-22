from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

from PIL import Image


SIGNED_INT_RE = re.compile(r"-?\d+")


def project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def resolve_from_project(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return (project_root() / candidate).resolve()


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def parse_gt_box_100(text: str) -> list[int] | None:
    values = [int(x) for x in SIGNED_INT_RE.findall(text or "")]
    if len(values) < 4:
        return None
    return values[:4]


def clamp_int(value: float, low: int, high: int) -> int:
    return int(max(low, min(high, round(float(value)))))


def normalized_100_to_pixel_xyxy(box_100: list[int], width: int, height: int) -> list[int] | None:
    if len(box_100) != 4:
        return None
    x0, y0, x1, y1 = [int(v) for v in box_100]
    if any(v < 0 or v > 100 for v in (x0, y0, x1, y1)):
        return None
    if not (x0 < x1 and y0 < y1):
        return None
    if width <= 1 or height <= 1:
        return None

    px0 = clamp_int(x0 * (width - 1) / 100.0, 0, width - 1)
    py0 = clamp_int(y0 * (height - 1) / 100.0, 0, height - 1)
    px1 = clamp_int(x1 * (width - 1) / 100.0, 0, width - 1)
    py1 = clamp_int(y1 * (height - 1) / 100.0, 0, height - 1)

    if not (px0 < px1 and py0 < py1):
        return None
    return [px0, py0, px1, py1]


def load_items(annotation_path: Path) -> list[dict]:
    items = json.loads(annotation_path.read_text(encoding="utf-8"))
    if not isinstance(items, list):
        raise TypeError(f"Unexpected annotation format: expected list, got {type(items)}")
    return items


def attach_clean_box_fields(items: list[dict], dataset_root_arg: str, image_dir_arg: str) -> tuple[list[dict], dict]:
    clean_rows: list[dict] = []
    dropped = {
        "missing_image_id": 0,
        "missing_question": 0,
        "missing_ground_truth": 0,
        "missing_image_file": 0,
        "invalid_gt_parse": 0,
        "invalid_gt_range_or_order": 0,
        "invalid_pixel_box": 0,
    }

    for item in items:
        qid = item.get("question_id")
        image_id = str(item.get("image_id", "")).strip()
        question = str(item.get("question", "")).strip()
        ground_truth = str(item.get("ground_truth", "")).strip()

        if not isinstance(qid, int):
            raise ValueError(f"Invalid question_id: {qid}")
        if not image_id:
            dropped["missing_image_id"] += 1
            continue
        if not question:
            dropped["missing_question"] += 1
            continue
        if not ground_truth:
            dropped["missing_ground_truth"] += 1
            continue

        image_relpath = str(Path(dataset_root_arg) / str(image_dir_arg) / image_id)
        image_abspath = resolve_from_project(image_relpath)
        if not image_abspath.is_file():
            dropped["missing_image_file"] += 1
            continue

        box_100 = parse_gt_box_100(ground_truth)
        if box_100 is None:
            dropped["invalid_gt_parse"] += 1
            continue
        if any(v < 0 or v > 100 for v in box_100) or not (box_100[0] < box_100[2] and box_100[1] < box_100[3]):
            dropped["invalid_gt_range_or_order"] += 1
            continue

        with Image.open(image_abspath) as img:
            width, height = img.size

        pixel_box = normalized_100_to_pixel_xyxy(box_100, width, height)
        if pixel_box is None:
            dropped["invalid_pixel_box"] += 1
            continue

        box_area = int((pixel_box[2] - pixel_box[0]) * (pixel_box[3] - pixel_box[1]))
        image_area = int(width * height)
        if box_area <= 0 or image_area <= 0:
            dropped["invalid_pixel_box"] += 1
            continue

        clean_rows.append(
            {
                "qid": int(qid),
                "filename": image_id,
                "image_id": image_id,
                "image_path": image_relpath,
                "question": question,
                "ground_truth": ground_truth,
                "ground_truth_xyxy_100": [int(v) for v in box_100],
                "ground_truth_xyxy_pixel": [int(v) for v in pixel_box],
                "image_width": int(width),
                "image_height": int(height),
                "box_area": int(box_area),
                "image_area": int(image_area),
                "box_area_ratio": float(box_area) / float(image_area),
                "is_unique": bool(item.get("unique", False)),
                "obj_cls": str(item.get("obj_cls", "")),
                "size_group": str(item.get("size_group", "")),
            }
        )

    clean_rows = sorted(clean_rows, key=lambda row: int(row["qid"]))
    return clean_rows, dropped


def assign_size_bucket(rows: list[dict]) -> dict:
    if not rows:
        raise ValueError("No clean rows available for size bucketing.")

    ratios = sorted(float(row["box_area_ratio"]) for row in rows)
    q33_index = min(len(ratios) - 1, max(0, len(ratios) // 3 - 1))
    q67_index = min(len(ratios) - 1, max(0, (len(ratios) * 2) // 3 - 1))
    q33 = float(ratios[q33_index])
    q67 = float(ratios[q67_index])

    counts = {"small": 0, "medium": 0, "large": 0}
    for row in rows:
        ratio = float(row["box_area_ratio"])
        if ratio <= q33:
            row["size_bucket"] = "small"
            counts["small"] += 1
        elif ratio <= q67:
            row["size_bucket"] = "medium"
            counts["medium"] += 1
        else:
            row["size_bucket"] = "large"
            counts["large"] += 1
    return {"q33": float(q33), "q67": float(q67), "counts": counts}


def take_stratified_subset(rows: list[dict], subset_size: int, seed: int) -> list[dict]:
    if subset_size <= 0 or subset_size >= len(rows):
        return list(rows)

    groups = {"small": [], "medium": [], "large": []}
    for row in rows:
        bucket = str(row["size_bucket"])
        if bucket not in groups:
            raise ValueError(f"Unexpected size_bucket: {bucket}")
        groups[bucket].append(row)

    rng = random.Random(int(seed))
    for bucket_rows in groups.values():
        bucket_rows.sort(key=lambda row: int(row["qid"]))
        rng.shuffle(bucket_rows)

    bucket_names = ["small", "medium", "large"]
    base = subset_size // 3
    remainder = subset_size % 3
    target = {name: base for name in bucket_names}
    for index in range(remainder):
        target[bucket_names[index]] += 1

    subset: list[dict] = []
    deficits = 0
    for name in bucket_names:
        take = min(target[name], len(groups[name]))
        subset.extend(groups[name][:take])
        deficits += target[name] - take

    if deficits > 0:
        leftovers: list[dict] = []
        for name in bucket_names:
            leftovers.extend(groups[name][min(target[name], len(groups[name])) :])
        leftovers.sort(key=lambda row: int(row["qid"]))
        rng.shuffle(leftovers)
        subset.extend(leftovers[:deficits])

    subset = sorted(subset, key=lambda row: int(row["qid"]))
    return subset


def summarize_counts(rows: list[dict]) -> dict:
    summary = {
        "num_samples": len(rows),
        "unique_samples": 0,
        "non_unique_samples": 0,
        "size_bucket_counts": {"small": 0, "medium": 0, "large": 0},
    }
    for row in rows:
        if bool(row["is_unique"]):
            summary["unique_samples"] += 1
        else:
            summary["non_unique_samples"] += 1
        summary["size_bucket_counts"][str(row["size_bucket"])] += 1
    return summary


def xyxy_to_polygon(box: list[int]) -> list[list[int]]:
    x0, y0, x1, y1 = [int(v) for v in box]
    return [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]


def convert_to_tsne_rows(rows: list[dict]) -> list[dict]:
    tsne_rows: list[dict] = []
    for row in rows:
        gt_xyxy_100 = [int(v) for v in row["ground_truth_xyxy_100"]]
        gt_xyxy_pixel = [int(v) for v in row["ground_truth_xyxy_pixel"]]
        tsne_rows.append(
            {
                "qid": int(row["qid"]),
                "image_id": str(row["image_id"]),
                "image_path": str(row["image_path"]),
                "question": str(row["question"]),
                "is_unique": bool(row["is_unique"]),
                "obj_cls": str(row["obj_cls"]),
                "image_width": int(row["image_width"]),
                "image_height": int(row["image_height"]),
                "gt_xyxy_100": gt_xyxy_100,
                "gt_xyxy_pixel": gt_xyxy_pixel,
                "gt_polygon_pixel": xyxy_to_polygon(gt_xyxy_pixel),
                "box_area": int(row["box_area"]),
                "image_area": int(row["image_area"]),
                "box_area_ratio": float(row["box_area_ratio"]),
                "size_bucket": str(row["size_bucket"]),
            }
        )
    return tsne_rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare a clean VRSBench referring subset for grounding t-SNE.")
    parser.add_argument("--dataset-root", type=str, default="datasets/VRSBench")
    parser.add_argument("--annotation-file", type=str, default="VRSBench_EVAL_referring.json")
    parser.add_argument("--image-dir", type=str, default="Images_val")
    parser.add_argument("--output-dir", type=str, default="benchmark/vrsbench/data/grounding_tsne")
    parser.add_argument("--subset-size", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    dataset_root = resolve_from_project(args.dataset_root)
    annotation_path = dataset_root / str(args.annotation_file)
    output_dir = resolve_from_project(args.output_dir)

    if not annotation_path.is_file():
        raise FileNotFoundError(f"Missing annotation file: {annotation_path}")

    items = load_items(annotation_path)
    clean_rows, dropped = attach_clean_box_fields(items, str(args.dataset_root), str(args.image_dir))
    bucket_info = assign_size_bucket(clean_rows)
    subset_rows = take_stratified_subset(clean_rows, int(args.subset_size), int(args.seed))

    tsne_full_rows = convert_to_tsne_rows(clean_rows)
    tsne_subset_rows = convert_to_tsne_rows(subset_rows)

    write_jsonl(output_dir / "vrsbench_referring_test_clean_full.jsonl", clean_rows)
    write_jsonl(output_dir / "vrsbench_referring_test_clean_subset.jsonl", subset_rows)
    write_jsonl(output_dir / "vrsbench_referring_tsne_clean_full.jsonl", tsne_full_rows)
    write_jsonl(output_dir / "vrsbench_referring_tsne_clean_subset.jsonl", tsne_subset_rows)
    write_json(
        output_dir / "vrsbench_referring_tsne_meta.json",
        {
            "dataset_root": str(args.dataset_root),
            "annotation_file": str(Path(args.dataset_root) / str(args.annotation_file)),
            "image_dir": str(Path(args.dataset_root) / str(args.image_dir)),
            "seed": int(args.seed),
            "subset_size_requested": int(args.subset_size),
            "subset_size_actual": int(len(subset_rows)),
            "clean_full": summarize_counts(clean_rows),
            "clean_subset": summarize_counts(subset_rows),
            "size_bucket_thresholds": {
                "small_leq": float(bucket_info["q33"]),
                "medium_leq": float(bucket_info["q67"]),
            },
            "dropped_counts": dropped,
            "tsne_dataset_files": {
                "full": "benchmark/vrsbench/data/grounding_tsne/vrsbench_referring_tsne_clean_full.jsonl",
                "subset": "benchmark/vrsbench/data/grounding_tsne/vrsbench_referring_tsne_clean_subset.jsonl",
            },
        },
    )

    print(f"[OK] full_clean={len(clean_rows)} subset={len(subset_rows)} output_dir={output_dir}")


if __name__ == "__main__":
    main()
