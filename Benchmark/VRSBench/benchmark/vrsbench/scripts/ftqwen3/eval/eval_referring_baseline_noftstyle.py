from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any


def _project_root() -> Path:
    return Path(__file__).resolve().parents[5]


def _resolve_from_project(path: str | Path) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    return (_project_root() / p).resolve()


def _rel_to_project(path: Path) -> str:
    p = Path(path)
    try:
        return str(p.resolve().relative_to(_project_root()))
    except Exception:
        return str(p.resolve())


def clamp_bbox_xyxy(bbox: list[float], width: int, height: int) -> list[int] | None:
    if len(bbox) != 4:
        return None
    try:
        x0, y0, x1, y1 = [float(v) for v in bbox]
    except Exception:
        return None

    x0 = max(0.0, min(float(width), x0))
    y0 = max(0.0, min(float(height), y0))
    x1 = max(0.0, min(float(width), x1))
    y1 = max(0.0, min(float(height), y1))
    if x1 <= x0 or y1 <= y0:
        return None
    return [int(round(x0)), int(round(y0)), int(round(x1)), int(round(y1))]


def clamp_bbox_2d_1000(bbox: list[float]) -> list[float] | None:
    if len(bbox) != 4:
        return None
    try:
        x0, y0, x1, y1 = [float(v) for v in bbox]
    except Exception:
        return None

    x0 = max(0.0, min(1000.0, x0))
    y0 = max(0.0, min(1000.0, y0))
    x1 = max(0.0, min(1000.0, x1))
    y1 = max(0.0, min(1000.0, y1))
    if x1 <= x0 or y1 <= y0:
        return None
    return [x0, y0, x1, y1]


def bbox2d_to_pixels(bbox_2d: list[float], width: int, height: int) -> list[int] | None:
    bb = clamp_bbox_2d_1000(bbox_2d)
    if bb is None:
        return None
    x0, y0, x1, y1 = bb
    px = [
        x0 * width / 1000.0,
        y0 * height / 1000.0,
        x1 * width / 1000.0,
        y1 * height / 1000.0,
    ]
    return clamp_bbox_xyxy(px, width, height)


def compute_iou_xyxy(bbox1: list[int], bbox2: list[int], *, return_parts: bool = False):
    """
    与 eval_vrsbench_referring.py 保持一致，复刻官方 eval_utils.py 的 computeIoU（面积计算含 +1）。
    bbox: [x1, y1, x2, y2]
    """

    x1, y1, x2, y2 = bbox1
    x3, y3, x4, y4 = bbox2
    ix1 = max(x1, x3)
    iy1 = max(y1, y3)
    ix2 = min(x2, x4)
    iy2 = min(y2, y4)

    inter_w = max(0, ix2 - ix1 + 1)
    inter_h = max(0, iy2 - iy1 + 1)
    inter_area = inter_w * inter_h

    area1 = (x2 - x1 + 1) * (y2 - y1 + 1)
    area2 = (x4 - x3 + 1) * (y4 - y3 + 1)
    union = area1 + area2 - inter_area
    if union <= 0:
        if return_parts:
            return 0.0, int(inter_area), int(union)
        return 0.0
    iou = float(inter_area) / float(union)
    if return_parts:
        return float(iou), int(inter_area), int(union)
    return float(iou)


def parse_gt_bbox_1000(gt: str) -> list[float] | None:
    if not isinstance(gt, str):
        return None
    nums = [float(x) for x in re.findall(r"\d+(?:\.\d+)?", gt)]
    if len(nums) != 4:
        return None
    max_v = max(nums)
    if max_v <= 100.0:
        return [v * 10.0 for v in nums]
    if max_v <= 1000.0:
        return nums
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate noft-style VRSBench grounding predictions.")
    parser.add_argument(
        "--preds",
        type=str,
        required=True,
        help="Prediction jsonl from generate_referring_baseline_noftstyle.py",
    )
    parser.add_argument(
        "--meta",
        type=str,
        default="benchmark/vrsbench/data/vrsbench_referring_meta.json",
        help="Prepared meta json path (optional, used for expected sample count).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="benchmark/vrsbench/eval/baseline_referring_noftstyle_summary.json",
    )
    args = parser.parse_args()

    sys.path.insert(0, str(_project_root() / "src"))
    from ftqwen3.jsonl import read_jsonl  # type: ignore

    preds_path = _resolve_from_project(args.preds)
    rows = read_jsonl(preds_path)

    expected_num_samples = None
    meta_path = _resolve_from_project(args.meta) if str(args.meta).strip() else None
    if meta_path and meta_path.is_file():
        try:
            meta_obj = json.loads(meta_path.read_text(encoding="utf-8"))
            expected_num_samples = int(meta_obj.get("num_samples"))
        except Exception:
            expected_num_samples = None

    split_stats: dict[str, dict[str, int | float]] = {
        "unique": {"total": 0, "valid": 0, "hit50": 0, "hit70": 0, "sum_iou": 0.0, "cum_i": 0, "cum_u": 0},
        "non_unique": {"total": 0, "valid": 0, "hit50": 0, "hit70": 0, "sum_iou": 0.0, "cum_i": 0, "cum_u": 0},
        "all": {"total": 0, "valid": 0, "hit50": 0, "hit70": 0, "sum_iou": 0.0, "cum_i": 0, "cum_u": 0},
    }
    valid = 0
    sum_iou = 0.0
    cum_i = 0
    cum_u = 0

    for row in rows:
        split_key = "unique" if bool(row.get("is_unique", row.get("unique", False))) else "non_unique"
        split_stats["all"]["total"] += 1
        split_stats[split_key]["total"] += 1

        width = row.get("image_width")
        height = row.get("image_height")
        if width in (None, "") or height in (None, ""):
            continue

        gt_1000 = parse_gt_bbox_1000(str(row.get("ground_truth", "")))
        if gt_1000 is None:
            continue
        gt_xyxy = bbox2d_to_pixels(gt_1000, int(width), int(height))
        if gt_xyxy is None:
            continue

        prediction = row.get("prediction", {})
        pred_bbox2d = prediction.get("bbox_2d", []) if isinstance(prediction, dict) else []
        if not isinstance(pred_bbox2d, list) or len(pred_bbox2d) != 4:
            continue
        pred_xyxy = bbox2d_to_pixels([float(v) for v in pred_bbox2d], int(width), int(height))
        if pred_xyxy is None:
            continue

        iou, inter, union = compute_iou_xyxy(pred_xyxy, gt_xyxy, return_parts=True)
        valid += 1
        sum_iou += iou
        cum_i += int(inter)
        cum_u += int(union)
        split_stats["all"]["valid"] += 1
        split_stats[split_key]["valid"] += 1
        split_stats["all"]["sum_iou"] += float(iou)
        split_stats[split_key]["sum_iou"] += float(iou)
        split_stats["all"]["cum_i"] += int(inter)
        split_stats[split_key]["cum_i"] += int(inter)
        split_stats["all"]["cum_u"] += int(union)
        split_stats[split_key]["cum_u"] += int(union)
        if iou >= 0.5:
            split_stats["all"]["hit50"] += 1
            split_stats[split_key]["hit50"] += 1
        if iou >= 0.7:
            split_stats["all"]["hit70"] += 1
            split_stats[split_key]["hit70"] += 1

    split_metrics: dict[str, dict[str, float | int]] = {}
    for key in ("unique", "non_unique", "all"):
        total_k = split_stats[key]["total"]
        valid_k = split_stats[key]["valid"]
        hit50_k = split_stats[key]["hit50"]
        hit70_k = split_stats[key]["hit70"]
        split_metrics[key] = {
            "count_total": total_k,
            "count_valid_for_iou": valid_k,
            "acc@0.5": ((hit50_k / total_k) * 100.0) if total_k > 0 else 0.0,
            "acc@0.7": ((hit70_k / total_k) * 100.0) if total_k > 0 else 0.0,
            "meanIoU": ((float(split_stats[key]["sum_iou"]) / total_k) * 100.0) if total_k > 0 else 0.0,
            "cumIoU": ((float(split_stats[key]["cum_i"]) / float(split_stats[key]["cum_u"])) * 100.0)
            if int(split_stats[key]["cum_u"]) > 0
            else 0.0,
            "acc@0.5_count": hit50_k,
            "acc@0.7_count": hit70_k,
        }

    splits: dict[str, dict[str, float | int]] = {}
    for key in ("unique", "non_unique", "all"):
        total_k = split_stats[key]["total"]
        hit50_k = split_stats[key]["hit50"]
        hit70_k = split_stats[key]["hit70"]
        denom_k = float(max(1, total_k))
        splits[key] = {
            "total": int(total_k),
            "Acc@0.5": float(hit50_k) / denom_k * 100.0,
            "Acc@0.7": float(hit70_k) / denom_k * 100.0,
            "meanIoU": (float(split_stats[key]["sum_iou"]) / denom_k) * 100.0,
            "cumIoU": ((float(split_stats[key]["cum_i"]) / float(split_stats[key]["cum_u"])) * 100.0)
            if int(split_stats[key]["cum_u"]) > 0
            else 0.0,
        }

    config_hint: dict[str, Any] = {}
    if rows:
        first = rows[0]
        for k in [
            "model",
            "model_dir",
            "prompt_template",
            "max_new_tokens",
            "requested_batch_size",
            "dtype",
            "decode_strategy",
            "do_sample",
            "temperature",
            "top_p",
            "top_k",
            "num_beams",
            "repetition_penalty",
            "seed",
        ]:
            if k in first:
                config_hint[k] = first.get(k)

    metrics: dict[str, Any] = {
        "task": "vrsbench_referring_noftstyle",
        "thresholds": [0.5, 0.7],
        "expected_num_samples": expected_num_samples,
        "num_rows_in_file": int(len(rows)),
        "preds": _rel_to_project(preds_path),
        "meta": _rel_to_project(meta_path) if meta_path and meta_path.is_file() else "",
        "config_hint": config_hint,
        "splits": splits,
        "total_selected": len(rows),
        "processed": len(rows),
        "valid_for_iou": valid,
        "mean_iou": ((sum_iou / split_stats["all"]["total"]) * 100.0) if split_stats["all"]["total"] > 0 else 0.0,
        "cum_iou": ((float(cum_i) / float(cum_u)) * 100.0) if cum_u > 0 else 0.0,
        "acc@0.5": ((split_stats["all"]["hit50"] / split_stats["all"]["total"]) * 100.0) if split_stats["all"]["total"] > 0 else 0.0,
        "acc@0.7": ((split_stats["all"]["hit70"] / split_stats["all"]["total"]) * 100.0) if split_stats["all"]["total"] > 0 else 0.0,
        "acc@0.5_count": split_stats["all"]["hit50"],
        "acc@0.7_count": split_stats["all"]["hit70"],
        "by_split": split_metrics,
        "predictions_file": str(preds_path),
    }

    out_path = _resolve_from_project(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    if expected_num_samples is not None and int(len(rows)) != int(expected_num_samples):
        print(
            f"[WARN] num_rows_in_file({len(rows)}) != expected_num_samples({expected_num_samples}). "
            "If the run is incomplete, resume generation then re-evaluate."
        )


if __name__ == "__main__":
    main()
