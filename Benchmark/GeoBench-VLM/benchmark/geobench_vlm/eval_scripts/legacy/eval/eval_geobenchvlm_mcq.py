from __future__ import annotations

import argparse
from collections import defaultdict
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


CHOICE_PATTERN = re.compile(r"(?:^|[^A-Z])([A-E])(?=$|[^A-Z])")


def extract_choice_letter(text: str) -> str:
    raw = str(text or "").strip().upper()
    if not raw:
        return ""
    for candidate in (raw, raw.lstrip("([{< \t\r\n"), raw.replace("OPTION", " "), raw.replace("ANSWER", " ")):
        match = CHOICE_PATTERN.search(candidate)
        if match:
            return str(match.group(1))
    return ""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate GeoBench-VLM MCQ predictions.")
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--predictions", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--details-output", type=str, default="")
    parser.add_argument("--expected-model-family", type=str, default="")
    parser.add_argument("--expected-model-dir", type=str, default="")
    parser.add_argument("--expected-prompt-version", type=str, default="")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)

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
        key = prediction_key(row.get("question_id"), row.get("prompt_index"))
        pred_map[key] = row

    sample_details: list[dict[str, Any]] = []
    task_scores: dict[str, list[float]] = defaultdict(list)
    total_prompt_count = 0
    missing_prompt_count = 0
    parse_fail_count = 0
    correct_prompt_total = 0.0

    for row in rows:
        qid = row.get("question_id")
        prompt_texts = list(row.get("prompts", []))
        prompt_records: list[dict[str, Any]] = []
        sample_sum = 0.0
        for prompt_index, prompt_text in enumerate(prompt_texts):
            total_prompt_count += 1
            pred_row = pred_map.get(prediction_key(qid, prompt_index))
            answer = "" if pred_row is None else str(pred_row.get("answer", ""))
            letter = extract_choice_letter(answer)
            correct = 1.0 if letter and letter == str(row.get("ground_truth_option", "")).strip().upper() else 0.0
            if pred_row is None:
                missing_prompt_count += 1
            elif not letter:
                parse_fail_count += 1
            sample_sum += correct
            correct_prompt_total += correct
            prompt_records.append(
                {
                    "prompt_index": int(prompt_index),
                    "prompt": str(prompt_text),
                    "predicted_letter": str(letter),
                    "ground_truth_option": str(row.get("ground_truth_option", "")).strip().upper(),
                    "correct": float(correct),
                    "answer": answer,
                }
            )

        prompt_count = len(prompt_texts)
        sample_score = (sample_sum / prompt_count) if prompt_count else 0.0
        task_name = str(row.get("task", ""))
        task_scores[task_name].append(float(sample_score))
        sample_details.append(
            {
                "question_id": qid,
                "task": task_name,
                "ground_truth": row.get("ground_truth"),
                "ground_truth_option": row.get("ground_truth_option"),
                "prompt_count": int(prompt_count),
                "sample_accuracy": float(sample_score),
                "prompt_results": prompt_records,
            }
        )

    overall_accuracy = (correct_prompt_total / total_prompt_count) if total_prompt_count else 0.0
    summary = {
        "metric": "accuracy",
        "aggregation": "per-prompt accuracy averaged across prompts, then averaged across samples",
        "num_questions": int(len(rows)),
        "total_prompts": int(total_prompt_count),
        "missing_prompt_count": int(missing_prompt_count),
        "parse_fail_count": int(parse_fail_count),
        "overall_accuracy": float(overall_accuracy),
        "per_task": {
            task_name: {
                "num_questions": int(len(values)),
                "accuracy": float(sum(values) / len(values)) if values else 0.0,
            }
            for task_name, values in sorted(task_scores.items())
        },
    }
    write_json(out_path, summary)
    if str(args.details_output).strip():
        write_json(resolve_from_project(args.details_output), sample_details)
    print(f"[OK] Wrote summary: {out_path}")


if __name__ == "__main__":
    main()
