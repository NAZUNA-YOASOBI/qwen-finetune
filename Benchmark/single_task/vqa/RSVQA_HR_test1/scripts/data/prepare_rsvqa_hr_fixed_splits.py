from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter
from pathlib import Path
from typing import Any


def detect_project_root() -> Path:
    current = Path(__file__).resolve().parent
    for candidate in [current, *current.parents]:
        if (candidate / "VRSBench").is_dir() and (candidate / "fine-tune-qwen3-vl").is_dir():
            return candidate
    raise FileNotFoundError(f"Cannot locate project root from {__file__}")


PROJECT_ROOT = detect_project_root()
RAW_ROOT = (
    PROJECT_ROOT
    / "VRSBench"
    / "benchmark"
    / "single_task"
    / "datasets"
    / "vqa"
    / "RSVQA-HR"
    / "official"
    / "raw"
)
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "fine-tune-qwen3-vl"
    / "Benchmark"
    / "single_task"
    / "vqa"
    / "RSVQA_HR_test1"
    / "data"
)
TRAIN_ALLOWED_QTYPES = ("area", "comp", "count", "presence")
TEST_ALLOWED_QTYPES = ("presence", "comp")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def ratio_tag(ratio: float) -> str:
    percent = float(ratio) * 100.0
    rounded_percent = int(round(percent))
    if not math.isclose(percent, float(rounded_percent), rel_tol=0.0, abs_tol=1e-9):
        raise ValueError(f"Only integer percentage ratios are supported, got {ratio}")
    return f"{rounded_percent}pct"


def safe_list(payload: dict[str, Any], key: str) -> list[dict[str, Any]]:
    raw = payload.get(key, [])
    if not isinstance(raw, list):
        raise ValueError(f"Expected list at key='{key}'.")
    return [item for item in raw if isinstance(item, dict)]


def filter_rows(
    *,
    questions_payload: dict[str, Any],
    answers_payload: dict[str, Any],
    images_payload: dict[str, Any],
    allowed_qtypes: tuple[str, ...],
    source_name: str,
) -> tuple[list[dict[str, Any]], dict[int, dict[str, Any]], dict[int, dict[str, Any]]]:
    answer_by_qid = {
        int(item["question_id"]): item
        for item in safe_list(answers_payload, "answers")
        if item.get("active") is True and "question_id" in item
    }
    image_by_id = {
        int(item["id"]): item
        for item in safe_list(images_payload, "images")
        if item.get("active") is True and "id" in item
    }

    rows: list[dict[str, Any]] = []
    for item in safe_list(questions_payload, "questions"):
        if item.get("active") is not True:
            continue
        qtype = str(item.get("type", "")).strip()
        if qtype not in allowed_qtypes:
            continue
        question_id = int(item["id"])
        image_id = int(item["img_id"])
        answer = answer_by_qid.get(question_id)
        image = image_by_id.get(image_id)
        if answer is None or image is None:
            continue
        row = dict(item)
        row["type"] = qtype
        row["id"] = question_id
        row["img_id"] = image_id
        row["source_split"] = source_name
        rows.append(row)
    return rows, answer_by_qid, image_by_id


def sample_rows_by_type(
    rows: list[dict[str, Any]],
    *,
    qtype_key: str,
    id_key: str,
    ratio: float,
    seed: int,
) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        qtype = str(row[qtype_key]).strip()
        grouped.setdefault(qtype, []).append(row)

    selected: list[dict[str, Any]] = []
    for type_index, qtype in enumerate(sorted(grouped.keys())):
        current = sorted(grouped[qtype], key=lambda row: int(row[id_key]))
        if float(ratio) >= 1.0:
            chosen = current
        else:
            target_count = max(1, int(math.floor(len(current) * float(ratio))))
            generator = random.Random(int(seed) + int(type_index) * 1000)
            order = list(range(len(current)))
            generator.shuffle(order)
            chosen = [current[index] for index in order[:target_count]]
            chosen = sorted(chosen, key=lambda row: int(row[id_key]))
        selected.extend(chosen)
    return sorted(selected, key=lambda row: int(row[id_key]))


def strip_source_split(row: dict[str, Any]) -> dict[str, Any]:
    out = dict(row)
    out.pop("source_split", None)
    return out


def build_split(
    *,
    split_name: str,
    questions_path: Path,
    answers_path: Path,
    images_path: Path,
    allowed_qtypes: tuple[str, ...],
    ratio: float,
    seed: int,
    prefix: str,
    output_dir: Path,
) -> dict[str, Any]:
    question_rows, answer_by_qid, image_by_id = filter_rows(
        questions_payload=read_json(questions_path),
        answers_payload=read_json(answers_path),
        images_payload=read_json(images_path),
        allowed_qtypes=allowed_qtypes,
        source_name=split_name,
    )
    selected_questions = sample_rows_by_type(
        question_rows,
        qtype_key="type",
        id_key="id",
        ratio=float(ratio),
        seed=int(seed),
    )
    selected_qids = {int(row["id"]) for row in selected_questions}
    selected_image_ids = sorted({int(row["img_id"]) for row in selected_questions})
    selected_answers = [
        answer_by_qid[qid]
        for qid in sorted(selected_qids)
        if qid in answer_by_qid and answer_by_qid[qid].get("active") is True
    ]
    selected_images = [
        image_by_id[image_id]
        for image_id in selected_image_ids
        if image_id in image_by_id and image_by_id[image_id].get("active") is True
    ]

    question_path = output_dir / f"{prefix}_questions.json"
    answer_path = output_dir / f"{prefix}_answers.json"
    image_path = output_dir / f"{prefix}_images.json"
    write_json(question_path, {"questions": [strip_source_split(row) for row in selected_questions]})
    write_json(answer_path, {"answers": selected_answers})
    write_json(image_path, {"images": selected_images})

    return {
        "dataset_name": prefix,
        "seed": int(seed),
        "sample_ratio": float(ratio),
        "allowed_question_types": list(allowed_qtypes),
        "source_files": {
            "questions": str(questions_path.relative_to(PROJECT_ROOT)),
            "answers": str(answers_path.relative_to(PROJECT_ROOT)),
            "images": str(images_path.relative_to(PROJECT_ROOT)),
        },
        "counts_before_sampling": dict(sorted(Counter(str(row["type"]) for row in question_rows).items())),
        "counts_after_sampling": dict(sorted(Counter(str(row["type"]) for row in selected_questions).items())),
        "selected_question_count": int(len(selected_questions)),
        "selected_answer_count": int(len(selected_answers)),
        "selected_image_count": int(len(selected_images)),
        "output_files": {
            "questions": question_path.name,
            "answers": answer_path.name,
            "images": image_path.name,
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare fixed RSVQA-HR train/val/test1 splits for the ours VQA bundle.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.2)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()

    train_meta = build_split(
        split_name="train",
        questions_path=RAW_ROOT / "USGS_split_train_questions.json",
        answers_path=RAW_ROOT / "USGS_split_train_answers.json",
        images_path=RAW_ROOT / "USGS_split_train_images.json",
        allowed_qtypes=TRAIN_ALLOWED_QTYPES,
        ratio=float(args.train_ratio),
        seed=int(args.seed) + 100,
        prefix=f"USGS_split_train_all_types_{ratio_tag(float(args.train_ratio))}",
        output_dir=output_dir,
    )
    val_meta = build_split(
        split_name="val",
        questions_path=RAW_ROOT / "USGS_split_val_questions.json",
        answers_path=RAW_ROOT / "USGS_split_val_answers.json",
        images_path=RAW_ROOT / "USGS_split_val_images.json",
        allowed_qtypes=TRAIN_ALLOWED_QTYPES,
        ratio=float(args.val_ratio),
        seed=int(args.seed) + 200,
        prefix=f"USGS_split_val_all_types_{ratio_tag(float(args.val_ratio))}",
        output_dir=output_dir,
    )
    test1_meta = build_split(
        split_name="test1",
        questions_path=RAW_ROOT / "USGS_split_test_questions.json",
        answers_path=RAW_ROOT / "USGS_split_test_answers.json",
        images_path=RAW_ROOT / "USGS_split_test_images.json",
        allowed_qtypes=TEST_ALLOWED_QTYPES,
        ratio=float(args.test_ratio),
        seed=int(args.seed),
        prefix=f"USGS_split_test_presence_comp_{ratio_tag(float(args.test_ratio))}",
        output_dir=output_dir,
    )

    write_json(
        output_dir / "dataset_info.json",
        {
            "train": train_meta,
            "val": val_meta,
            "test1": test1_meta,
        },
    )
    print(
        json.dumps(
            {
                "train": train_meta,
                "val": val_meta,
                "test1": test1_meta,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
