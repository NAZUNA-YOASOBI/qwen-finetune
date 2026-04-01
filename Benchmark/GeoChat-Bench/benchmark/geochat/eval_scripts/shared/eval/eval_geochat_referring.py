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


def _polygon_centroid(points: list[list[float]]) -> tuple[float, float]:
    cx = sum(float(point[0]) for point in points) / float(len(points))
    cy = sum(float(point[1]) for point in points) / float(len(points))
    return float(cx), float(cy)


def _polygon_signed_area(points: list[list[float]]) -> float:
    area = 0.0
    for index, point in enumerate(points):
        nxt = points[(index + 1) % len(points)]
        area += (float(point[0]) * float(nxt[1])) - (float(nxt[0]) * float(point[1]))
    return float(area / 2.0)


def _order_polygon_ccw(points: list[list[float]]) -> list[list[float]]:
    cx, cy = _polygon_centroid(points)
    ordered = sorted(points, key=lambda point: math.atan2(float(point[1]) - cy, float(point[0]) - cx))
    if _polygon_signed_area(ordered) < 0.0:
        ordered.reverse()
    return [[float(point[0]), float(point[1])] for point in ordered]


def _normalize_polygon(points: Any) -> list[list[float]] | None:
    if not isinstance(points, list) or len(points) < 3:
        return None
    out: list[list[float]] = []
    for point in points:
        if not isinstance(point, list) or len(point) != 2:
            return None
        try:
            out.append([float(point[0]), float(point[1])])
        except Exception:
            return None
    return _order_polygon_ccw(out)


def _polygon_area(points: list[list[float]]) -> float:
    return abs(_polygon_signed_area(points))


def _cross(a: list[float], b: list[float], c: list[float]) -> float:
    return (float(b[0]) - float(a[0])) * (float(c[1]) - float(a[1])) - (float(b[1]) - float(a[1])) * (float(c[0]) - float(a[0]))


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
    px = ((det_a * (x3 - x4)) - ((x1 - x2) * det_b)) / denom
    py = ((det_a * (y3 - y4)) - ((y1 - y2) * det_b)) / denom
    return [float(px), float(py)]


def _polygon_clip(subject_polygon: list[list[float]], clip_polygon: list[list[float]]) -> list[list[float]]:
    output = [list(point) for point in subject_polygon]
    clip = _order_polygon_ccw(clip_polygon)
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
    return _order_polygon_ccw(output) if len(output) >= 3 else []


def _compute_polygon_iou(poly_a: list[list[float]], poly_b: list[list[float]]) -> float:
    norm_a = _normalize_polygon(poly_a)
    norm_b = _normalize_polygon(poly_b)
    if not norm_a or not norm_b:
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
    gts = row.get("ground_truth", [])
    out: list[list[list[float]]] = []
    if not isinstance(gts, list):
        return out
    for item in gts:
        polygon = _normalize_polygon(item)
        if polygon is not None:
            out.append(polygon)
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


def _clamp_pixel_xyxy(box: list[float], width: int, height: int) -> list[float] | None:
    if len(box) != 4:
        return None
    x0, y0, x1, y1 = [float(v) for v in box]
    x0 = max(0.0, min(float(width), x0))
    y0 = max(0.0, min(float(height), y0))
    x1 = max(0.0, min(float(width), x1))
    y1 = max(0.0, min(float(height), y1))
    if x1 <= x0 or y1 <= y0:
        return None
    return [float(x0), float(y0), float(x1), float(y1)]


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


def _xyxy_to_polygon(box: list[float]) -> list[list[float]] | None:
    if len(box) != 4:
        return None
    x0, y0, x1, y1 = [float(v) for v in box]
    if x1 <= x0 or y1 <= y0:
        return None
    return [
        [float(x0), float(y0)],
        [float(x0), float(y1)],
        [float(x1), float(y1)],
        [float(x1), float(y0)],
    ]


def _bbox2d_1000_to_polygon(box: list[float], *, width: int, height: int) -> list[list[float]] | None:
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
    pixel_box = _clamp_pixel_xyxy(px, width, height)
    if pixel_box is None:
        return None
    polygon = _xyxy_to_polygon(pixel_box)
    return _normalize_polygon(polygon) if polygon is not None else None


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


def _maximum_match_counts(pred_polygons: list[list[list[float]]], gt_polygons: list[list[list[float]]], threshold: float) -> tuple[int, int, int]:
    adjacency: list[list[int]] = []
    for pred_polygon in pred_polygons:
        neighbors: list[int] = []
        for gt_index, gt_polygon in enumerate(gt_polygons):
            if _compute_polygon_iou(pred_polygon, gt_polygon) >= float(threshold):
                neighbors.append(int(gt_index))
        adjacency.append(neighbors)

    match_to_pred = [-1] * len(gt_polygons)

    def _dfs(pred_index: int, visited_gt: list[bool]) -> bool:
        for gt_index in adjacency[pred_index]:
            if visited_gt[gt_index]:
                continue
            visited_gt[gt_index] = True
            if match_to_pred[gt_index] == -1 or _dfs(match_to_pred[gt_index], visited_gt):
                match_to_pred[gt_index] = int(pred_index)
                return True
        return False

    matched = 0
    for pred_index in range(len(pred_polygons)):
        if _dfs(pred_index, [False] * len(gt_polygons)):
            matched += 1

    tp = int(matched)
    fp = int(len(pred_polygons) - matched)
    fn = int(len(gt_polygons) - matched)
    return tp, fp, fn


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

        gt_polygons = _gt_polygons(data_row)
        pred_polygons = []
        pred_boxes_raw = _extract_pred_boxes(str(row.get("answer", "")))
        for box in pred_boxes_raw:
            polygon = _bbox2d_1000_to_polygon(box, width=int(width), height=int(height))
            if polygon is not None:
                pred_polygons.append(polygon)

        tp, fp, fn = _maximum_match_counts(pred_polygons, gt_polygons, float(args.threshold))
        sample_f1 = _sample_f1(tp=tp, num_pred=len(pred_polygons), num_gt=len(gt_polygons))
        ok = bool(gt_polygons) and sample_f1 >= 1.0
        if not pred_polygons:
            parse_fail += 1

        buckets = ["all"]
        size_group = str(data_row.get("size_group", "")).strip().lower()
        if size_group in {"small", "medium", "large"}:
            buckets.append(size_group)
        if len(gt_polygons) > 1:
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
                    "pred_polygon_count": len(pred_polygons),
                    "gt_polygon_count": len(gt_polygons),
                    "tp": int(tp),
                    "fp": int(fp),
                    "fn": int(fn),
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
        "match_policy": "pred_bbox_to_rectangle_polygon_vs_gt_polygon_with_bipartite_iou_matching_and_strict_f1_eq_1_success",
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
