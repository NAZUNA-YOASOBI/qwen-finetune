from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any


def _project_root() -> Path:
    return Path(__file__).resolve().parents[5]


def _resolve_from_project(path: str | Path) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    return (_project_root() / p).resolve()


def _extract_answer(row: dict[str, Any], *, explicit_field: str) -> str | None:
    if explicit_field and explicit_field in row and row[explicit_field] not in (None, ""):
        return str(row[explicit_field])
    for key in ("answer", "ground_truth", "gt", "label", "target", "response"):
        if key in row and row[key] not in (None, ""):
            return str(row[key])
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate GeoChat VQA predictions with external ground-truth file.")
    parser.add_argument("--preds", type=str, required=True)
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--ground-truth-file", type=str, required=True)
    parser.add_argument("--gt-id-field", type=str, default="question_id")
    parser.add_argument("--gt-answer-field", type=str, default="")
    parser.add_argument("--exclude-categories", type=str, default="count")
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()

    import sys

    sys.path.insert(0, str(_project_root() / "src"))
    from shared.common import iter_rows_from_json_or_jsonl, normalize_free_text, read_jsonl, write_json
    from shared.eval_utils import assert_prediction_integrity, load_benchmark_row_map

    gt_path = _resolve_from_project(args.ground_truth_file)
    data_path = _resolve_from_project(args.data)
    if not gt_path.is_file():
        raise FileNotFoundError(f"Missing ground-truth file: {gt_path}")

    gt_map: dict[str, dict[str, Any]] = {}
    duplicate_gt_ids: list[str] = []
    for row in iter_rows_from_json_or_jsonl(gt_path):
        if "active" in row and row.get("active") is not True:
            continue
        if args.gt_id_field not in row:
            continue
        gt_id = str(row[args.gt_id_field])
        if gt_id in gt_map:
            duplicate_gt_ids.append(gt_id)
            continue
        gt_map[gt_id] = row
    if not gt_map:
        raise ValueError(f"No valid ground-truth rows loaded from: {gt_path}")
    if duplicate_gt_ids:
        raise ValueError(
            "Ground-truth file contains duplicated active ids: "
            f"{sorted(set(duplicate_gt_ids))[:20]}"
        )

    excluded = {x.strip().lower() for x in str(args.exclude_categories).split(",") if x.strip()}
    pred_path = _resolve_from_project(args.preds)
    if not pred_path.is_file():
        raise FileNotFoundError(f"Missing prediction file: {pred_path}")
    assert_prediction_integrity(pred_path, data_path, key_field="question_id", answer_field="answer")
    data_row_map = load_benchmark_row_map(data_path, key_field="question_id")
    preds = read_jsonl(pred_path, allow_truncated_last_line=True)
    if not preds:
        raise ValueError(f"Prediction file is empty: {pred_path}")

    total = 0
    correct = 0
    missing_gt = 0
    missing_answer = 0
    excluded_rows = 0
    per_category: dict[str, dict[str, int]] = {}
    mismatches: list[dict[str, Any]] = []
    missing_gt_examples: list[str] = []
    missing_answer_examples: list[str] = []

    for row in preds:
        qid = str(row.get("question_id", ""))
        data_row = data_row_map.get(qid, None)
        if data_row is None:
            missing_gt += 1
            if len(missing_gt_examples) < 20:
                missing_gt_examples.append(qid)
            continue
        category = str(data_row.get("category", "")).strip().lower()
        if category in excluded:
            excluded_rows += 1
            continue

        gt_row = gt_map.get(qid, None)
        if gt_row is None:
            missing_gt += 1
            if len(missing_gt_examples) < 20:
                missing_gt_examples.append(qid)
            continue

        gt_answer = _extract_answer(gt_row, explicit_field=str(args.gt_answer_field))
        if gt_answer is None:
            missing_answer += 1
            if len(missing_answer_examples) < 20:
                missing_answer_examples.append(qid)
            continue

        pred_norm = normalize_free_text(str(row.get("answer", "")))
        gt_norm = normalize_free_text(str(gt_answer))
        ok = pred_norm == gt_norm

        total += 1
        if ok:
            correct += 1

        stats = per_category.setdefault(category or "unknown", {"total": 0, "correct": 0})
        stats["total"] += 1
        if ok:
            stats["correct"] += 1
        elif len(mismatches) < 100:
            mismatches.append(
                {
                    "question_id": qid,
                    "category": category,
                    "answer": str(row.get("answer", "")),
                    "answer_norm": pred_norm,
                    "ground_truth": str(gt_answer),
                    "ground_truth_norm": gt_norm,
                }
            )

    if total <= 0:
        raise ValueError(
            f"No valid VQA rows were evaluated. preds={pred_path}, ground_truth={gt_path}, "
            f"missing_gt={missing_gt}, missing_answer={missing_answer}, excluded_rows={excluded_rows}"
        )
    if missing_gt > 0 or missing_answer > 0:
        raise ValueError(
            "Ground-truth file does not fully cover evaluated VQA predictions. "
            f"missing_gt={missing_gt}, missing_answer={missing_answer}, "
            f"missing_gt_examples={missing_gt_examples}, "
            f"missing_answer_examples={missing_answer_examples}"
        )

    per_category_summary = {
        key: {
            "total": int(value["total"]),
            "correct": int(value["correct"]),
            "accuracy": float(value["correct"] / value["total"]) if value["total"] else 0.0,
            "accuracy_x100": float(value["correct"] * 100.0 / value["total"]) if value["total"] else 0.0,
        }
        for key, value in sorted(per_category.items())
    }

    summary = {
        "task": "geochat_vqa",
        "total": int(total),
        "correct": int(correct),
        "accuracy": float(correct / total) if total else 0.0,
        "accuracy_x100": float(correct * 100.0 / total) if total else 0.0,
        "missing_ground_truth": int(missing_gt),
        "excluded_rows": int(excluded_rows),
        "excluded_categories": sorted(excluded),
        "per_category": per_category_summary,
        "mismatch_examples": mismatches,
    }
    write_json(_resolve_from_project(args.output), summary)
    print(summary)


if __name__ == "__main__":
    main()
