from __future__ import annotations

import argparse
from pathlib import Path


def _project_root() -> Path:
    return Path(__file__).resolve().parents[5]


def _resolve_from_project(path: str | Path) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    return (_project_root() / p).resolve()


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate GeoChat region caption predictions.")
    parser.add_argument("--preds", type=str, required=True)
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()

    import sys

    sys.path.insert(0, str(_project_root() / "src"))
    from shared.common import read_jsonl, write_json
    from shared.eval_utils import assert_prediction_integrity, load_benchmark_row_map

    try:
        from nltk.translate.meteor_score import meteor_score  # type: ignore
        from rouge_score import rouge_scorer  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "Missing dependency for region caption evaluation. "
            "Please install `nltk` and `rouge-score` in the target environment."
        ) from e

    pred_path = _resolve_from_project(args.preds)
    data_path = _resolve_from_project(args.data)
    if not pred_path.is_file():
        raise FileNotFoundError(f"Missing prediction file: {pred_path}")
    assert_prediction_integrity(pred_path, data_path, key_field="question_id", answer_field="answer")
    data_row_map = load_benchmark_row_map(data_path, key_field="question_id")
    rows = read_jsonl(pred_path, allow_truncated_last_line=True)
    if not rows:
        raise ValueError(f"Prediction file is empty: {pred_path}")
    scorer = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=False)

    rouge1_sum = 0.0
    rougel_sum = 0.0
    meteor_sum = 0.0
    total = 0
    missing_reference_ids: list[str] = []
    for row in rows:
        qid = str(row.get("question_id", ""))
        data_row = data_row_map.get(qid)
        if data_row is None:
            raise ValueError(f"Missing benchmark row for prediction question_id={qid}")
        pred = str(row.get("answer", "")).strip()
        ref = str(data_row.get("ground_truth", "")).strip()
        if not ref:
            if len(missing_reference_ids) < 20:
                missing_reference_ids.append(qid)
            continue
        scores = scorer.score(ref, pred)
        rouge1_sum += float(scores["rouge1"].fmeasure)
        rougel_sum += float(scores["rougeL"].fmeasure)
        meteor_sum += float(meteor_score([ref.split()], pred.split()))
        total += 1

    if missing_reference_ids:
        raise ValueError(
            "Benchmark data has rows without `ground_truth` for region caption evaluation. "
            f"question_ids={missing_reference_ids}"
        )
    if total <= 0:
        raise ValueError(f"No valid region-caption rows with reference text found in: {pred_path}")

    summary = {
        "task": "geochat_region_caption",
        "total": int(total),
        "metrics": {
            "ROUGE-1": float(rouge1_sum / total) if total else 0.0,
            "ROUGE-L": float(rougel_sum / total) if total else 0.0,
            "METEOR": float(meteor_sum / total) if total else 0.0,
        },
        "metrics_x100": {
            "ROUGE-1": float(rouge1_sum * 100.0 / total) if total else 0.0,
            "ROUGE-L": float(rougel_sum * 100.0 / total) if total else 0.0,
            "METEOR": float(meteor_sum * 100.0 / total) if total else 0.0,
        },
    }
    write_json(_resolve_from_project(args.output), summary)
    print(summary)


if __name__ == "__main__":
    main()
