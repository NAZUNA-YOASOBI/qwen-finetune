from __future__ import annotations

import argparse
import json
from pathlib import Path


def _project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _resolve_from_project(path: str | Path) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    return (_project_root() / p).resolve()


def _read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _normalize_text(text: str) -> str:
    return " ".join(str(text or "").strip().split())


def _strip_eval_suffix(text: str) -> str:
    suffix = "Answer the question using a single word or phrase."
    norm = _normalize_text(text)
    if norm.endswith(suffix):
        norm = norm[: -len(suffix)].strip()
    return norm


def main() -> None:
    parser = argparse.ArgumentParser(description="Build aligned HRBEN ground-truth file for GeoChat benchmark.")
    parser.add_argument("--bench-data", type=str, required=True)
    parser.add_argument("--raw-questions", type=str, required=True)
    parser.add_argument("--raw-answers", type=str, required=True)
    parser.add_argument("--raw-images", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()

    bench_path = _resolve_from_project(args.bench_data)
    raw_questions_path = _resolve_from_project(args.raw_questions)
    raw_answers_path = _resolve_from_project(args.raw_answers)
    raw_images_path = _resolve_from_project(args.raw_images)
    output_path = _resolve_from_project(args.output)

    bench_rows = []
    with bench_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            bench_rows.append(json.loads(line))
    if not bench_rows:
        raise ValueError(f"Empty bench data: {bench_path}")

    raw_questions = _read_json(raw_questions_path)["questions"]
    raw_answers = _read_json(raw_answers_path)["answers"]
    raw_images = _read_json(raw_images_path)["images"]

    q_map = {int(row["id"]): row for row in raw_questions if row.get("active") is True and "id" in row and "question" in row}
    a_map = {
        int(row["question_id"]): row
        for row in raw_answers
        if row.get("active") is True and "question_id" in row and "answer" in row
    }
    i_map = {int(row["id"]): row for row in raw_images if row.get("active") is True and "id" in row}

    out_rows: list[dict] = []
    missing_examples: list[dict] = []
    mismatch_examples: list[dict] = []

    type_to_category = {
        "presence": "presence",
        "comp": "comp",
    }

    for row in bench_rows:
        qid = int(row["question_id"])
        q_row = q_map.get(qid)
        a_row = a_map.get(qid)
        image_name = str(row.get("image", ""))

        if q_row is None or a_row is None:
            if len(missing_examples) < 20:
                missing_examples.append({"question_id": qid, "image": image_name, "has_question": q_row is not None, "has_answer": a_row is not None})
            continue

        img_id = int(q_row["img_id"])
        i_row = i_map.get(img_id)
        if i_row is None:
            if len(missing_examples) < 20:
                missing_examples.append({"question_id": qid, "image": image_name, "missing_image_meta": img_id})
            continue

        expected_question = _normalize_text(str(q_row["question"]))
        bench_question = _strip_eval_suffix(str(row.get("text", "")))
        expected_image_name = f"{img_id}.tif"
        expected_category = type_to_category.get(str(q_row.get("type", "")).strip().lower(), "")
        bench_category = str(row.get("category", "")).strip().lower()

        if (
            expected_question != bench_question
            or expected_image_name != image_name
            or expected_category != bench_category
        ):
            if len(mismatch_examples) < 20:
                mismatch_examples.append(
                    {
                        "question_id": qid,
                        "bench_image": image_name,
                        "expected_image": expected_image_name,
                        "bench_question": bench_question,
                        "expected_question": expected_question,
                        "bench_category": bench_category,
                        "expected_category": expected_category,
                    }
                )
            continue

        out_rows.append(
            {
                "question_id": qid,
                "answer": str(a_row["answer"]),
                "image": image_name,
                "image_id": img_id,
                "question": expected_question,
                "category": bench_category,
                "raw_question_type": str(q_row.get("type", "")),
                "raw_original_name": str(i_row.get("original_name", "")),
            }
        )

    if missing_examples:
        raise ValueError(f"Missing raw HRBEN rows during alignment: {missing_examples}")
    if mismatch_examples:
        raise ValueError(f"Mismatch between bench and raw HRBEN rows: {mismatch_examples}")
    if len(out_rows) != len(bench_rows):
        raise ValueError(f"Aligned row count mismatch: bench={len(bench_rows)} aligned={len(out_rows)}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for row in out_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = {
        "bench_rows": len(bench_rows),
        "aligned_rows": len(out_rows),
        "output": str(output_path),
    }
    print(summary)


if __name__ == "__main__":
    main()
