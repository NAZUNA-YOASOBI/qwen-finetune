from __future__ import annotations

import argparse
from collections import defaultdict
import json
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


BBOX_PATTERN = re.compile(
    r'"bbox_2d"\s*:\s*\[\s*'
    r'([-+]?\d*\.?\d+)\s*,\s*'
    r'([-+]?\d*\.?\d+)\s*,\s*'
    r'([-+]?\d*\.?\d+)\s*,\s*'
    r'([-+]?\d*\.?\d+)\s*'
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


def _clamp_norm_value(value: Any) -> float:
    number = float(value)
    if number < 0.0:
        return 0.0
    if number > 1000.0:
        return 1000.0
    return float(number)


def _normalize_xyxy(box: list[float]) -> list[float] | None:
    if not isinstance(box, list) or len(box) != 4:
        return None
    try:
        x0, y0, x1, y1 = [float(v) for v in box]
    except Exception:
        return None
    left = min(x0, x1)
    top = min(y0, y1)
    right = max(x0, x1)
    bottom = max(y0, y1)
    return [float(left), float(top), float(right), float(bottom)]


def scale_xyxy_from_1000(box: list[float], image_size: float) -> list[float] | None:
    norm = _normalize_xyxy(box)
    if norm is None:
        return None
    scale = float(image_size) / 1000.0
    x0, y0, x1, y1 = norm
    return [
        _clamp_norm_value(x0) * scale,
        _clamp_norm_value(y0) * scale,
        _clamp_norm_value(x1) * scale,
        _clamp_norm_value(y1) * scale,
    ]


def polygon_to_xyxy(points: list[list[float]]) -> list[float] | None:
    if not isinstance(points, list) or not points:
        return None
    try:
        xs = [float(point[0]) for point in points]
        ys = [float(point[1]) for point in points]
    except Exception:
        return None
    return [float(min(xs)), float(min(ys)), float(max(xs)), float(max(ys))]


def compute_iou_xyxy(box_a: list[float], box_b: list[float]) -> float:
    norm_a = _normalize_xyxy(box_a)
    norm_b = _normalize_xyxy(box_b)
    if not norm_a or not norm_b:
        return 0.0
    ax0, ay0, ax1, ay1 = norm_a
    bx0, by0, bx1, by1 = norm_b
    inter_x0 = max(ax0, bx0)
    inter_y0 = max(ay0, by0)
    inter_x1 = min(ax1, bx1)
    inter_y1 = min(ay1, by1)
    inter_w = max(0.0, inter_x1 - inter_x0)
    inter_h = max(0.0, inter_y1 - inter_y0)
    inter_area = inter_w * inter_h
    area_a = max(0.0, ax1 - ax0) * max(0.0, ay1 - ay0)
    area_b = max(0.0, bx1 - bx0) * max(0.0, by1 - by0)
    union_area = area_a + area_b - inter_area
    if union_area <= 0.0:
        return 0.0
    return float(inter_area / union_area)


def parse_predicted_boxes(answer: str, image_size: float) -> tuple[list[list[float]], bool]:
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

    boxes: list[list[float]] = []
    for obj in objects:
        if not isinstance(obj, dict):
            continue
        bbox = obj.get("bbox_2d", None)
        box = scale_xyxy_from_1000(bbox, image_size)
        if box is not None:
            boxes.append(box)

    if boxes:
        return boxes, parse_failed

    regex_matches = BBOX_PATTERN.findall(raw)
    if regex_matches:
        fallback_boxes: list[list[float]] = []
        for match in regex_matches:
            box = scale_xyxy_from_1000([float(value) for value in match], image_size)
            if box is not None:
                fallback_boxes.append(box)
        if fallback_boxes:
            return fallback_boxes, parse_failed

    return [], parse_failed or bool(str(answer).strip())


def greedy_match(pred_boxes: list[list[float]], gt_boxes: list[list[float]], threshold: float) -> tuple[int, int, int]:
    remaining_pred = set(range(len(pred_boxes)))
    remaining_gt = set(range(len(gt_boxes)))
    matched = 0

    while remaining_pred and remaining_gt:
        best_pair: tuple[int, int] | None = None
        best_iou = -1.0
        for pred_index in remaining_pred:
            for gt_index in remaining_gt:
                iou = compute_iou_xyxy(pred_boxes[pred_index], gt_boxes[gt_index])
                if iou > best_iou:
                    best_iou = float(iou)
                    best_pair = (pred_index, gt_index)
        if best_pair is None or best_iou < float(threshold):
            break
        pred_index, gt_index = best_pair
        remaining_pred.remove(pred_index)
        remaining_gt.remove(gt_index)
        matched += 1

    tp = int(matched)
    fp = int(len(pred_boxes) - matched)
    fn = int(len(gt_boxes) - matched)
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

        gt_boxes = [
            polygon_to_xyxy(polygon)
            for polygon in list(row.get("ground_truth", []))
            if isinstance(polygon, list) and polygon
        ]
        gt_boxes = [box for box in gt_boxes if box is not None]
        pred_boxes, parse_failed = parse_predicted_boxes(answer, float(row.get("image_size", 1000)))
        if parse_failed:
            parse_fail_count += 1

        per_threshold_results: dict[str, Any] = {}
        for threshold in thresholds:
            tp, fp, fn = greedy_match(pred_boxes, gt_boxes, threshold)
            sample_f1 = f1_from_counts(tp, fp, fn)
            success = 1.0 if (int(fp) == 0 and int(fn) == 0 and int(len(gt_boxes)) == int(len(pred_boxes))) else 0.0
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
                "gt_object_count": int(len(gt_boxes)),
                "pred_object_count": int(len(pred_boxes)),
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
        "aggregation": "GT polygons are converted to xyxy, predicted bbox_2d uses 0..1000 normalized xyxy, greedy matching is applied, and a sample counts as correct only when sample_F1=1",
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
