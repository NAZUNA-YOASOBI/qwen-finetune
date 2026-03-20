from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate GeoBench-VLM captioning predictions with BERTScore.")
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--predictions", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--details-output", type=str, default="")
    parser.add_argument("--expected-model-family", type=str, default="")
    parser.add_argument("--expected-model-dir", type=str, default="")
    parser.add_argument("--expected-prompt-version", type=str, default="")
    parser.add_argument("--lang", type=str, default="en")
    parser.add_argument("--model-type", type=str, default="")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", type=str, default="")
    parser.add_argument("--rescale-with-baseline", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)

    try:
        from bert_score import score as bert_score  # type: ignore
    except Exception as e:
        raise RuntimeError("Missing dependency: bert_score") from e

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

    candidates: list[str] = []
    references: list[str] = []
    metas: list[tuple[Any, int]] = []
    missing_prompt_count = 0

    for row in rows:
        qid = row.get("question_id")
        ground_truth = str(row.get("ground_truth", ""))
        prompts = list(row.get("prompts", []))
        for prompt_index, _prompt in enumerate(prompts):
            pred_row = pred_map.get(prediction_key(qid, prompt_index))
            answer = "" if pred_row is None else str(pred_row.get("answer", ""))
            if pred_row is None:
                missing_prompt_count += 1
            candidates.append(answer)
            references.append(ground_truth)
            metas.append((qid, int(prompt_index)))

    score_kwargs: dict[str, Any] = {
        "cands": candidates,
        "refs": references,
        "lang": str(args.lang),
        "batch_size": int(args.batch_size),
        "rescale_with_baseline": bool(args.rescale_with_baseline),
        "verbose": False,
    }
    if str(args.model_type).strip():
        score_kwargs["model_type"] = str(args.model_type)
    if str(args.device).strip():
        score_kwargs["device"] = str(args.device)

    _precision, _recall, f1 = bert_score(**score_kwargs)
    f1_values = [float(value) for value in f1.tolist()]

    per_question_scores: dict[Any, list[float]] = defaultdict(list)
    prompt_details_map: dict[Any, list[dict[str, Any]]] = defaultdict(list)
    for (qid, prompt_index), score_value, answer in zip(metas, f1_values, candidates):
        per_question_scores[qid].append(float(score_value))
        prompt_details_map[qid].append(
            {
                "prompt_index": int(prompt_index),
                "bertscore_f1": float(score_value),
                "answer": str(answer),
            }
        )

    sample_details: list[dict[str, Any]] = []
    task_scores: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        qid = row.get("question_id")
        values = per_question_scores.get(qid, [])
        sample_score = float(sum(values) / len(values)) if values else 0.0
        task_name = str(row.get("task", ""))
        task_scores[task_name].append(float(sample_score))
        sample_details.append(
            {
                "question_id": qid,
                "task": task_name,
                "prompt_count": int(len(row.get("prompts", []))),
                "bertscore_f1_mean": float(sample_score),
                "prompt_results": sorted(prompt_details_map.get(qid, []), key=lambda item: int(item["prompt_index"])),
            }
        )

    overall_score = float(sum(f1_values) / len(f1_values)) if f1_values else 0.0
    summary = {
        "metric": "BERTScore-F1",
        "aggregation": "per-prompt BERTScore-F1 averaged across prompts, then averaged across samples",
        "num_questions": int(len(rows)),
        "total_prompts": int(len(candidates)),
        "missing_prompt_count": int(missing_prompt_count),
        "overall_bertscore_f1": float(overall_score),
        "lang": str(args.lang),
        "model_type": str(args.model_type),
        "rescale_with_baseline": bool(args.rescale_with_baseline),
        "per_task": {
            task_name: {
                "num_questions": int(len(values)),
                "bertscore_f1": float(sum(values) / len(values)) if values else 0.0,
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
