from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any


FLOAT_ARRAY_PATTERN = re.compile(
    r"\[\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\]"
)


def _project_root() -> Path:
    return Path(__file__).resolve().parents[5]


def _resolve_from_project(path: str | Path) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    return (_project_root() / p).resolve()


def _polygon_to_xyxy(poly: list[list[float]]) -> list[int]:
    xs = [float(p[0]) for p in poly]
    ys = [float(p[1]) for p in poly]
    return [int(math.floor(min(xs))), int(math.floor(min(ys))), int(math.ceil(max(xs))), int(math.ceil(max(ys)))]


def _gt_boxes(row: dict[str, Any]) -> list[list[int]]:
    gts = row.get("ground_truth", [])
    out: list[list[int]] = []
    if not isinstance(gts, list):
        return out
    for item in gts:
        if isinstance(item, list) and item and isinstance(item[0], list):
            out.append(_polygon_to_xyxy(item))
    return out


def _strip_code_fence(text: str) -> str:
    raw = str(text or "").strip()
    if not raw.startswith("```"):
        return raw
    raw = re.sub(r"^```[A-Za-z0-9_-]*\s*", "", raw, count=1).strip()
    raw = re.sub(r"\s*```$", "", raw, count=1).strip()
    if raw:
        return raw
    lines = raw.splitlines()
    if not lines:
        return raw
    if lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip().startswith("```"):
        lines = lines[:-1]
    return "\n".join(lines).strip()


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


def _clamp_pixel_xyxy(box: list[float], width: int, height: int) -> list[int] | None:
    if len(box) != 4:
        return None
    x0, y0, x1, y1 = [float(v) for v in box]
    x0 = max(0.0, min(float(width), x0))
    y0 = max(0.0, min(float(height), y0))
    x1 = max(0.0, min(float(width), x1))
    y1 = max(0.0, min(float(height), y1))
    if x1 <= x0 or y1 <= y0:
        return None
    return [int(round(x0)), int(round(y0)), int(round(x1)), int(round(y1))]


def _clamp_bbox_2d_1000(box: list[float]) -> list[float] | None:
    if len(box) != 4:
        return None
    x0, y0, x1, y1 = [float(v) for v in box]
    x0 = max(0.0, min(1000.0, x0))
    y0 = max(0.0, min(1000.0, y0))
    x1 = max(0.0, min(1000.0, x1))
    y1 = max(0.0, min(1000.0, y1))
    if x1 <= x0 or y1 <= y0:
        return None
    return [x0, y0, x1, y1]


def _bbox2d_1000_to_pixels(box: list[float], *, width: int, height: int) -> list[int] | None:
    bb = _clamp_bbox_2d_1000(box)
    if bb is None:
        return None
    x0, y0, x1, y1 = bb
    px = [
        x0 * float(width) / 1000.0,
        y0 * float(height) / 1000.0,
        x1 * float(width) / 1000.0,
        y1 * float(height) / 1000.0,
    ]
    return _clamp_pixel_xyxy(px, width, height)


def _dedup_boxes(boxes: list[list[float]]) -> list[list[float]]:
    out: list[list[float]] = []
    seen: set[tuple[float, float, float, float]] = set()
    for box in boxes:
        if len(box) != 4:
            continue
        key = (float(box[0]), float(box[1]), float(box[2]), float(box[3]))
        if key in seen:
            continue
        seen.add(key)
        out.append([float(v) for v in key])
    return out


def _collect_bbox_2d_values(node: Any, out: list[list[float]]) -> None:
    if isinstance(node, dict):
        for key, value in node.items():
            if key == "bbox_2d" and isinstance(value, list) and len(value) == 4:
                try:
                    out.append([float(v) for v in value])
                except Exception:
                    pass
                continue
            _collect_bbox_2d_values(value, out)
    elif isinstance(node, list):
        for item in node:
            _collect_bbox_2d_values(item, out)


def _extract_pred_boxes(text: str) -> list[list[float]]:
    raw = _strip_code_fence(str(text or ""))
    parsed = _try_parse_json_value(raw)
    parsed_boxes: list[list[float]] = []
    if parsed is not None:
        if isinstance(parsed, list) and len(parsed) == 4:
            try:
                parsed_boxes = [[float(v) for v in parsed]]
            except Exception:
                parsed_boxes = []
        elif isinstance(parsed, list) and parsed and all(isinstance(item, list) and len(item) == 4 for item in parsed):
            try:
                parsed_boxes = [[float(v) for v in item] for item in parsed]
            except Exception:
                parsed_boxes = []
        else:
            _collect_bbox_2d_values(parsed, parsed_boxes)
    regex_boxes: list[list[float]] = []
    for match in FLOAT_ARRAY_PATTERN.finditer(raw):
        try:
            regex_boxes.append([float(match.group(i)) for i in range(1, 5)])
        except Exception:
            continue
    return _dedup_boxes(parsed_boxes + regex_boxes)


def _compute_iou_xyxy(box1: list[int], box2: list[int]) -> float:
    x1, y1, x2, y2 = box1
    x3, y3, x4, y4 = box2
    inter_x1 = max(x1, x3)
    inter_y1 = max(y1, y3)
    inter_x2 = min(x2, x4)
    inter_y2 = min(y2, y4)
    inter = max(0, inter_x2 - inter_x1 + 1) * max(0, inter_y2 - inter_y1 + 1)
    area1 = max(0, x2 - x1 + 1) * max(0, y2 - y1 + 1)
    area2 = max(0, x4 - x3 + 1) * max(0, y4 - y3 + 1)
    union = area1 + area2 - inter
    return float(inter / union) if union > 0 else 0.0


def _greedy_true_positives(iou_mat: list[list[float]], thr: float) -> int:
    if not iou_mat or not iou_mat[0]:
        return 0
    work = [row[:] for row in iou_mat]
    n_gt = len(work)
    n_pr = len(work[0])
    matched = 0

    for _ in range(min(n_gt, n_pr)):
        best_iou = -1.0
        best_gt = -1
        best_pr = -1
        for gt_idx in range(n_gt):
            for pr_idx in range(n_pr):
                value = float(work[gt_idx][pr_idx])
                if value > best_iou:
                    best_iou = value
                    best_gt = gt_idx
                    best_pr = pr_idx
        if best_iou < float(thr) or best_gt < 0 or best_pr < 0:
            break
        matched += 1
        for pr_idx in range(n_pr):
            work[best_gt][pr_idx] = 0.0
        for gt_idx in range(n_gt):
            work[gt_idx][best_pr] = 0.0
    return matched


def _sample_f1(*, tp: int, num_pred: int, num_gt: int) -> float:
    if num_pred <= 0 and num_gt <= 0:
        return 1.0
    fp = max(0, int(num_pred) - int(tp))
    fn = max(0, int(num_gt) - int(tp))
    denom = 2 * int(tp) + fp + fn
    if denom <= 0:
        return 0.0
    return float((2 * int(tp)) / denom)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate GeoChat referring predictions.")
    parser.add_argument("--preds", type=str, required=True)
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--image-root", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    import sys

    sys.path.insert(0, str(_project_root() / "src"))
    from shared.common import find_image_path, read_jsonl, write_json
    from shared.eval_utils import assert_prediction_integrity, load_benchmark_row_map

    try:
        from PIL import Image
    except Exception as e:
        raise RuntimeError("Missing dependency: pillow") from e

    pred_path = _resolve_from_project(args.preds)
    data_path = _resolve_from_project(args.data)
    image_root = _resolve_from_project(args.image_root)
    if not pred_path.is_file():
        raise FileNotFoundError(f"Missing prediction file: {pred_path}")
    if not image_root.is_dir():
        raise FileNotFoundError(f"Missing referring image root: {image_root}")
    assert_prediction_integrity(pred_path, data_path, key_field="question_id", answer_field="answer")
    data_row_map = load_benchmark_row_map(data_path, key_field="question_id")
    rows = read_jsonl(pred_path, allow_truncated_last_line=True)
    if not rows:
        raise ValueError(f"Prediction file is empty: {pred_path}")

    totals = {
        "all": {"total": 0, "correct": 0, "f1_sum": 0.0},
        "small": {"total": 0, "correct": 0, "f1_sum": 0.0},
        "medium": {"total": 0, "correct": 0, "f1_sum": 0.0},
        "large": {"total": 0, "correct": 0, "f1_sum": 0.0},
        "single_object": {"total": 0, "correct": 0, "f1_sum": 0.0},
        "multi_object": {"total": 0, "correct": 0, "f1_sum": 0.0},
        "refer": {"total": 0, "correct": 0, "f1_sum": 0.0},
        "grounding": {"total": 0, "correct": 0, "f1_sum": 0.0},
    }
    parse_fail = 0
    mismatch_examples: list[dict[str, Any]] = []
    image_size_cache: dict[str, tuple[int, int]] = {}

    for row in rows:
        qid = str(row.get("question_id", ""))
        data_row = data_row_map.get(qid)
        if data_row is None:
            raise ValueError(f"Missing benchmark row for prediction question_id={qid}")

        image_path = find_image_path(
            image_root,
            image_value=data_row.get("image", None),
            image_id=data_row.get("image_id", None),
            default_ext=".png",
        )
        image_path_key = str(image_path)
        if image_path_key not in image_size_cache:
            with Image.open(image_path) as img:
                image_size_cache[image_path_key] = (int(img.size[0]), int(img.size[1]))
        width, height = image_size_cache[image_path_key]

        gt_boxes = _gt_boxes(data_row)
        pred_boxes = []
        pred_boxes_raw = _extract_pred_boxes(str(row.get("answer", "")))
        for box in pred_boxes_raw:
            px = _bbox2d_1000_to_pixels(box, width=int(width), height=int(height))
            if px is not None:
                pred_boxes.append(px)

        iou_mat = [[_compute_iou_xyxy(gt, pred) for pred in pred_boxes] for gt in gt_boxes]
        tp = _greedy_true_positives(iou_mat, float(args.threshold))
        sample_f1 = _sample_f1(tp=tp, num_pred=len(pred_boxes), num_gt=len(gt_boxes))
        ok = bool(gt_boxes) and sample_f1 >= 1.0
        if not pred_boxes:
            parse_fail += 1

        buckets = ["all"]
        size_group = str(data_row.get("size_group", "")).strip().lower()
        if size_group in {"small", "medium", "large"}:
            buckets.append(size_group)
        if len(gt_boxes) > 1:
            buckets.append("multi_object")
        else:
            buckets.append("single_object")
        qtype = str(data_row.get("type", "")).strip().lower()
        if qtype == "ref":
            buckets.append("refer")
        elif qtype:
            buckets.append("grounding")

        for bucket in buckets:
            totals[bucket]["total"] += 1
            totals[bucket]["f1_sum"] += float(sample_f1)
            if ok:
                totals[bucket]["correct"] += 1

        if (not ok) and len(mismatch_examples) < 50:
            mismatch_examples.append(
                {
                    "question_id": qid,
                    "answer": row.get("answer", ""),
                    "pred_box_count": len(pred_boxes),
                    "gt_box_count": len(gt_boxes),
                    "tp": int(tp),
                    "f1": float(sample_f1),
                    "type": qtype,
                    "size_group": size_group,
                }
            )

    if totals["all"]["total"] <= 0:
        raise ValueError(f"No valid referring rows found in prediction file: {pred_path}")

    summary = {
        "task": "geochat_referring",
        "threshold": float(args.threshold),
        "parse_fail_count": int(parse_fail),
        "coord_mode": "qwen_native_bbox2d_1000_only",
        "match_policy": "global_greedy_iou_matching_with_strict_f1_eq_1_success",
        "metrics": {
            key: {
                "total": int(value["total"]),
                "correct": int(value["correct"]),
                "acc": float(value["correct"] / value["total"]) if value["total"] else 0.0,
                "acc_x100": float(value["correct"] * 100.0 / value["total"]) if value["total"] else 0.0,
                "mean_f1": float(value["f1_sum"] / value["total"]) if value["total"] else 0.0,
                "mean_f1_x100": float(value["f1_sum"] * 100.0 / value["total"]) if value["total"] else 0.0,
            }
            for key, value in totals.items()
        },
        "mismatch_examples": mismatch_examples,
    }
    write_json(_resolve_from_project(args.output), summary)
    print(summary)


if __name__ == "__main__":
    main()
