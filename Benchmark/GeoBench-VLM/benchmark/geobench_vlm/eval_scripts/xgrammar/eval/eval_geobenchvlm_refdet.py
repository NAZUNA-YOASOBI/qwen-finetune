from __future__ import annotations

import argparse
from collections import defaultdict
import json
from math import atan2
from pathlib import Path
import re
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[5]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from shared.common import prediction_key, read_json, read_jsonl, resolve_from_project, write_json


def infer_expected_benchmark_task(data_path: Path) -> str:
    parts = {part.lower() for part in data_path.parts}
    if "single" in parts:
        return "single"
    if "temporal" in parts:
        return "temporal"
    if "captioning" in parts:
        return "captioning"
    if "ref-det" in parts or "ref_det" in parts:
        return "ref_det"
    return ""


POLYGON_PATTERN = re.compile(
    r'"bbox_2d"\s*:\s*\[\s*'
    r'\[\s*([-+]?\d*\.?\d+)\s*,\s*([-+]?\d*\.?\d+)\s*\]\s*,\s*'
    r'\[\s*([-+]?\d*\.?\d+)\s*,\s*([-+]?\d*\.?\d+)\s*\]\s*,\s*'
    r'\[\s*([-+]?\d*\.?\d+)\s*,\s*([-+]?\d*\.?\d+)\s*\]\s*,\s*'
    r'\[\s*([-+]?\d*\.?\d+)\s*,\s*([-+]?\d*\.?\d+)\s*\]\s*'
    r'\]',
    flags=re.IGNORECASE,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate GeoBench-VLM Ref-Det predictions.")
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--predictions", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--details-output", type=str, default="")
    parser.add_argument("--expected-model-family", type=str, default="")
    parser.add_argument("--expected-model-dir", type=str, default="")
    parser.add_argument("--expected-prompt-version", type=str, default="")
    parser.add_argument("--thresholds", type=str, default="0.25,0.5")
    return parser


def strip_code_fence(text: str) -> str:
    raw = str(text or "").strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?", "", raw).strip()
        raw = re.sub(r"```$", "", raw).strip()
    return raw


def extract_json_segment(text: str) -> str:
    raw = strip_code_fence(text)
    start_candidates = [index for index in (raw.find("["), raw.find("{")) if index >= 0]
    if not start_candidates:
        return raw
    start = min(start_candidates)
    end_list = raw.rfind("]")
    end_dict = raw.rfind("}")
    end = max(end_list, end_dict)
    if end < start:
        return raw[start:]
    return raw[start : end + 1]


def _clamp_pixel_value(value: Any, image_size: float) -> float:
    number = float(value)
    if number < 0.0:
        return 0.0
    if number > float(image_size):
        return float(image_size)
    return float(number)


def _polygon_centroid(points: list[list[float]]) -> tuple[float, float]:
    xs = [float(point[0]) for point in points]
    ys = [float(point[1]) for point in points]
    return float(sum(xs) / len(xs)), float(sum(ys) / len(ys))


def _polygon_signed_area(points: list[list[float]]) -> float:
    if len(points) < 3:
        return 0.0
    total = 0.0
    for index, (x0, y0) in enumerate(points):
        x1, y1 = points[(index + 1) % len(points)]
        total += (float(x0) * float(y1)) - (float(x1) * float(y0))
    return float(total / 2.0)


def _order_polygon_ccw(points: list[list[float]]) -> list[list[float]]:
    if len(points) < 3:
        return [[float(point[0]), float(point[1])] for point in points]
    cx, cy = _polygon_centroid(points)
    ordered = sorted(
        [[float(point[0]), float(point[1])] for point in points],
        key=lambda point: atan2(float(point[1]) - cy, float(point[0]) - cx),
    )
    if _polygon_signed_area(ordered) < 0.0:
        ordered.reverse()
    return ordered


def _normalize_scaled_polygon(points: list[list[float]]) -> list[list[float]] | None:
    if not isinstance(points, list) or len(points) != 4:
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


def clamp_polygon_to_image(points: list[list[float]], image_size: float) -> list[list[float]] | None:
    if not isinstance(points, list) or len(points) != 4:
        return None
    scaled: list[list[float]] = []
    for point in points:
        if not isinstance(point, list) or len(point) != 2:
            return None
        try:
            x = _clamp_pixel_value(point[0], image_size)
            y = _clamp_pixel_value(point[1], image_size)
        except Exception:
            return None
        scaled.append([float(x), float(y)])
    return _order_polygon_ccw(scaled)


def _polygon_area(points: list[list[float]]) -> float:
    return abs(_polygon_signed_area(points))


def _cross(a: list[float], b: list[float], c: list[float]) -> float:
    return (float(b[0]) - float(a[0])) * (float(c[1]) - float(a[1])) - (float(b[1]) - float(a[1])) * (float(c[0]) - float(a[0]))


def _inside(point: list[float], edge_start: list[float], edge_end: list[float]) -> bool:
    return _cross(edge_start, edge_end, point) >= -1e-9


def _segment_intersection(
    start_a: list[float],
    end_a: list[float],
    start_b: list[float],
    end_b: list[float],
) -> list[float]:
    x1, y1 = float(start_a[0]), float(start_a[1])
    x2, y2 = float(end_a[0]), float(end_a[1])
    x3, y3 = float(start_b[0]), float(start_b[1])
    x4, y4 = float(end_b[0]), float(end_b[1])
    denom = ((x1 - x2) * (y3 - y4)) - ((y1 - y2) * (x3 - x4))
    if abs(denom) <= 1e-9:
        return [float(end_a[0]), float(end_a[1])]
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


def compute_polygon_iou(poly_a: list[list[float]], poly_b: list[list[float]]) -> float:
    norm_a = _normalize_scaled_polygon(poly_a)
    norm_b = _normalize_scaled_polygon(poly_b)
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


def parse_predicted_polygons(answer: str, image_size: float) -> tuple[list[list[list[float]]], bool]:
    if not str(answer or "").strip():
        return [], False

    raw = extract_json_segment(answer)
    parse_failed = False
    objects: list[Any] = []

    try:
        parsed = json.loads(raw)
        if isinstance(parsed, dict):
            objects = [parsed]
        elif isinstance(parsed, list):
            objects = list(parsed)
        else:
            objects = []
    except Exception:
        parse_failed = True
        objects = []

    polygons: list[list[list[float]]] = []
    for obj in objects:
        if not isinstance(obj, dict):
            continue
        bbox = obj.get("bbox_2d", None)
        polygon = clamp_polygon_to_image(bbox, image_size)
        if polygon is not None:
            polygons.append(polygon)

    if polygons:
        return polygons, parse_failed

    regex_matches = POLYGON_PATTERN.findall(raw)
    if regex_matches:
        fallback_polygons: list[list[list[float]]] = []
        for match in regex_matches:
            coords = [float(value) for value in match]
            polygon = clamp_polygon_to_image(
                [
                    [coords[0], coords[1]],
                    [coords[2], coords[3]],
                    [coords[4], coords[5]],
                    [coords[6], coords[7]],
                ],
                image_size,
            )
            if polygon is not None:
                fallback_polygons.append(polygon)
        if fallback_polygons:
            return fallback_polygons, parse_failed

    return [], parse_failed or bool(str(answer).strip())


def maximum_match_counts(pred_polygons: list[list[list[float]]], gt_polygons: list[list[list[float]]], threshold: float) -> tuple[int, int, int]:
    adjacency: list[list[int]] = []
    for pred_polygon in pred_polygons:
        neighbors: list[int] = []
        for gt_index, gt_polygon in enumerate(gt_polygons):
            if compute_polygon_iou(pred_polygon, gt_polygon) >= float(threshold):
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


def f1_from_counts(tp: int, fp: int, fn: int) -> float:
    denom = (2 * int(tp)) + int(fp) + int(fn)
    if denom <= 0:
        return 0.0
    return float((2.0 * float(tp)) / float(denom))


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)

    thresholds = [float(part.strip()) for part in str(args.thresholds).split(",") if part.strip()]
    data_path = resolve_from_project(args.data)
    pred_path = resolve_from_project(args.predictions)
    out_path = resolve_from_project(args.output)
    expected_model_family = str(args.expected_model_family).strip()
    expected_model_dir = str(resolve_from_project(args.expected_model_dir)) if str(args.expected_model_dir).strip() else ""
    expected_prompt_version = str(args.expected_prompt_version).strip()

    expected_task = infer_expected_benchmark_task(data_path)
    rows = read_json(data_path)
    preds = read_jsonl(pred_path, allow_truncated_last_line=True)

    pred_map: dict[str, dict[str, Any]] = {}
    for row in preds:
        if expected_task and str(row.get("benchmark_task", expected_task)) != expected_task:
            continue
        if expected_model_family and str(row.get("model_family", "")).strip() != expected_model_family:
            continue
        if expected_model_dir and str(row.get("model_dir", "")).strip() != expected_model_dir:
            continue
        if expected_prompt_version and str(row.get("prompt_version", "")).strip() != expected_prompt_version:
            continue
        key = prediction_key(row.get("question_id"), row.get("prompt_index", 0))
        pred_map[key] = row

    threshold_stats: dict[float, dict[str, Any]] = {
        threshold: {
            "success_sum": 0.0,
            "f1_sum": 0.0,
            "tp": 0,
            "fp": 0,
            "fn": 0,
            "sample_count": 0,
        }
        for threshold in thresholds
    }
    parse_fail_count = 0
    missing_prediction_count = 0
    sample_details: list[dict[str, Any]] = []
    task_scores: dict[float, dict[str, list[float]]] = {threshold: defaultdict(list) for threshold in thresholds}

    for row in rows:
        qid = row.get("question_id")
        pred_row = pred_map.get(prediction_key(qid, 0))
        answer = "" if pred_row is None else str(pred_row.get("answer", ""))
        if pred_row is None:
            missing_prediction_count += 1

        gt_polygons = [_normalize_scaled_polygon(polygon) for polygon in list(row.get("ground_truth", [])) if isinstance(polygon, list) and polygon]
        gt_polygons = [polygon for polygon in gt_polygons if polygon is not None]
        pred_polygons, parse_failed = parse_predicted_polygons(answer, float(row.get("image_size", 1000)))
        if parse_failed:
            parse_fail_count += 1

        per_threshold_results: dict[str, Any] = {}
        for threshold in thresholds:
            tp, fp, fn = maximum_match_counts(pred_polygons, gt_polygons, threshold)
            sample_f1 = f1_from_counts(tp, fp, fn)
            success = 1.0 if (int(fp) == 0 and int(fn) == 0 and int(len(gt_polygons)) == int(len(pred_polygons))) else 0.0
            stats = threshold_stats[threshold]
            stats["success_sum"] += success
            stats["f1_sum"] += float(sample_f1)
            stats["tp"] += int(tp)
            stats["fp"] += int(fp)
            stats["fn"] += int(fn)
            stats["sample_count"] += 1
            task_scores[threshold][str(row.get("task", ""))].append(float(success))
            per_threshold_results[f"{threshold:.2f}"] = {
                "acc": float(success),
                "sample_f1": float(sample_f1),
                "tp": int(tp),
                "fp": int(fp),
                "fn": int(fn),
            }

        sample_details.append(
            {
                "question_id": qid,
                "task": str(row.get("task", "")),
                "prompt": str((row.get("prompts") or [""])[0]),
                "gt_object_count": int(len(gt_polygons)),
                "pred_object_count": int(len(pred_polygons)),
                "parse_failed": bool(parse_failed),
                "thresholds": per_threshold_results,
                "answer": answer,
            }
        )

    threshold_summaries: dict[str, Any] = {}
    for threshold, stats in threshold_stats.items():
        sample_count = int(stats["sample_count"])
        mean_success = float(stats["success_sum"] / sample_count) if sample_count else 0.0
        mean_f1 = float(stats["f1_sum"] / sample_count) if sample_count else 0.0
        tp = int(stats["tp"])
        fp = int(stats["fp"])
        fn = int(stats["fn"])
        micro_precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        micro_recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        threshold_summaries[f"{threshold:.2f}"] = {
            "acc": float(mean_success),
            "mean_f1": float(mean_f1),
            "total_tp": tp,
            "total_fp": fp,
            "total_fn": fn,
            "micro_precision": float(micro_precision),
            "micro_recall": float(micro_recall),
            "per_task": {
                task_name: {
                    "num_questions": int(len(values)),
                    "acc": float(sum(values) / len(values)) if values else 0.0,
                }
                for task_name, values in sorted(task_scores[threshold].items())
            },
        }

    summary = {
        "metric": "strict Acc and mean F1",
        "aggregation": "GT polygons and predicted polygons use 4-point polygons; predicted bbox_2d uses direct pixel polygon corners, values outside the image range are clamped to [0, image_size], maximum bipartite matching is applied, and a sample counts as correct only when sample_F1=1",
        "num_questions": int(len(rows)),
        "missing_prediction_count": int(missing_prediction_count),
        "parse_fail_count": int(parse_fail_count),
        "thresholds": threshold_summaries,
    }

    write_json(out_path, summary)
    if str(args.details_output).strip():
        write_json(resolve_from_project(args.details_output), sample_details)
    print(f"[OK] Wrote summary: {out_path}")


if __name__ == "__main__":
    main()
