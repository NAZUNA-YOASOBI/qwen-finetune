from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from tqdm import tqdm


SHORT_EXACT_ANSWERS = {"yes", "no", "rural", "urban", *[str(index) for index in range(100)]}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.is_file():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            raw = line.strip()
            if not raw:
                continue
            rows.append(json.loads(raw))
    return rows


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def normalize_vqa_answer(text: Any) -> str:
    return str(text).strip().lower().strip(" \t\r\n.,!?;:'\"")


def local_vqa_match(*, ground_truth: str, predicted: str) -> str:
    gt = normalize_vqa_answer(ground_truth)
    pred = normalize_vqa_answer(predicted)
    if not gt:
        return "0"
    if gt in SHORT_EXACT_ANSWERS:
        return "1" if gt == pred else "0"
    return "1" if gt in pred else "0"


def compute_vqa_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_type: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        qtype = str(row.get("question_type", ""))
        by_type.setdefault(qtype, []).append(row)

    per_type: dict[str, Any] = {}
    type_accs: list[float] = []
    total_correct = 0
    for qtype, items in by_type.items():
        correct = sum(1 for row in items if str(row.get("correct", "")) == "1")
        accuracy = float(correct) / float(len(items))
        type_accs.append(accuracy)
        total_correct += correct
        per_type[qtype] = {
            "paper_question_type": str(items[0].get("paper_question_type", qtype)),
            "num_rows": len(items),
            "accuracy": accuracy,
            "accuracy_x100": accuracy * 100.0,
        }

    overall_accuracy = float(total_correct) / float(len(rows)) if rows else 0.0
    average_accuracy = sum(type_accs) / float(len(type_accs)) if type_accs else 0.0
    return {
        "num_rows": len(rows),
        "per_type": per_type,
        "average_accuracy": average_accuracy,
        "average_accuracy_x100": average_accuracy * 100.0,
        "overall_accuracy": overall_accuracy,
        "overall_accuracy_x100": overall_accuracy * 100.0,
    }


def list_prediction_datasets(output_dir: Path) -> list[str]:
    dataset_names: list[str] = []
    for path in sorted(output_dir.glob("*.jsonl")):
        if path.name.endswith("_scored.jsonl"):
            continue
        dataset_names.append(path.stem)
    return dataset_names


def parse_prediction_datasets(raw: str, *, output_dir: Path) -> list[str]:
    items = [item.strip() for item in str(raw).split(",") if item.strip()]
    if not items or items == ["all"]:
        return list_prediction_datasets(output_dir)
    return items


def run_local_vqa_eval_for_dataset(*, pred_path: Path, eval_dir: Path) -> dict[str, Any]:
    rows = read_jsonl(pred_path)
    scored_path = eval_dir / f"{pred_path.stem}_scored.jsonl"
    done_by_qid = {int(row.get("question_id", -1)): row for row in read_jsonl(scored_path)}
    pending = [row for row in rows if int(row.get("question_id", -1)) not in done_by_qid]
    samples_preview: list[dict[str, Any]] = []
    progress = tqdm(total=len(pending), desc=f"{pred_path.stem}:eval", leave=False)

    for row in pending:
        ground_truth = normalize_vqa_answer(row.get("answer", ""))
        predicted = normalize_vqa_answer(row.get("prediction", ""))
        correct = local_vqa_match(ground_truth=ground_truth, predicted=predicted)
        scored_row = {
            **row,
            "ground_truth": ground_truth,
            "predicted": predicted,
            "correct": correct,
            "eval_source": "local_exact_match",
        }
        append_jsonl(scored_path, scored_row)
        if len(samples_preview) < 8:
            samples_preview.append(
                {
                    "question_id": scored_row["question_id"],
                    "question_type": scored_row.get("question_type", ""),
                    "predicted": scored_row["predicted"],
                    "ground_truth": scored_row["ground_truth"],
                    "correct": scored_row["correct"],
                    "eval_source": scored_row["eval_source"],
                }
            )
        progress.update(1)

    progress.close()
    scored_rows = read_jsonl(scored_path)
    summary = {
        "dataset": pred_path.stem,
        "prediction_path": str(pred_path),
        "scored_path": str(scored_path),
        "num_pending_before_run": len(pending),
        "eval_mode": "local_exact_match",
        "sample_evaluations": samples_preview,
        **compute_vqa_metrics(scored_rows),
    }
    write_json(eval_dir / f"{pred_path.stem}_summary.json", summary)
    return summary
