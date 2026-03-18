from __future__ import annotations

import argparse
from pathlib import Path


def _project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _resolve_from_project(path: str | Path) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    return (_project_root() / p).resolve()


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate GeoChat scene classification predictions.")
    parser.add_argument("--preds", type=str, required=True)
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()

    import sys

    sys.path.insert(0, str(_project_root() / "src"))
    from geochatbench_common import normalize_scene_label, read_jsonl, write_json
    from geochatbench_eval_utils import assert_prediction_integrity

    pred_path = _resolve_from_project(args.preds)
    data_path = _resolve_from_project(args.data)
    if not pred_path.is_file():
        raise FileNotFoundError(f"Missing prediction file: {pred_path}")
    assert_prediction_integrity(pred_path, data_path, key_field="question_id", answer_field="answer")
    rows = read_jsonl(pred_path, allow_truncated_last_line=True)
    if not rows:
        raise ValueError(f"Prediction file is empty: {pred_path}")
    correct = 0
    incorrect = 0
    mismatches: list[dict] = []
    for row in rows:
        question_id = str(row.get("question_id", ""))
        gt = question_id.split("/", 1)[0].lower()
        pred = normalize_scene_label(str(row.get("answer", "")))
        if gt == pred:
            correct += 1
        else:
            incorrect += 1
            if len(mismatches) < 50:
                mismatches.append(
                    {
                        "question_id": question_id,
                        "ground_truth": gt,
                        "answer": str(row.get("answer", "")),
                        "answer_norm": pred,
                    }
                )

    total = correct + incorrect
    if total <= 0:
        raise ValueError(f"No valid scene rows found in prediction file: {pred_path}")
    summary = {
        "task": "geochat_scene",
        "total": int(total),
        "correct": int(correct),
        "incorrect": int(incorrect),
        "accuracy": float(correct / total) if total else 0.0,
        "accuracy_x100": float(correct * 100.0 / total) if total else 0.0,
        "mismatch_examples": mismatches,
    }
    write_json(_resolve_from_project(args.output), summary)
    print(summary)


if __name__ == "__main__":
    main()
