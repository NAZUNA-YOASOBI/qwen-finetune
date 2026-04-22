from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def project_root() -> Path:
    return Path(__file__).resolve().parents[4]


def resolve_from_project(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return (project_root() / candidate).resolve()


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare official VRSBench VQA benchmark data.")
    parser.add_argument("--dataset-root", type=str, default="VRSBench/datasets/VRSBench")
    parser.add_argument("--annotation-file", type=str, default="VRSBench_EVAL_vqa.json")
    parser.add_argument("--image-dir", type=str, default="Images_val")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="VRSBench/benchmark/vrsbench/data",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_root = resolve_from_project(args.dataset_root)
    ann_path = dataset_root / str(args.annotation_file)
    image_dir = dataset_root / str(args.image_dir)
    output_dir = resolve_from_project(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not ann_path.is_file():
        raise FileNotFoundError(f"missing annotation file: {ann_path}")
    if not image_dir.is_dir():
        raise FileNotFoundError(f"missing image dir: {image_dir}")

    items = json.loads(ann_path.read_text(encoding="utf-8"))
    if not isinstance(items, list):
        raise TypeError(f"unexpected annotation format: expected list, got {type(items)}")

    rows: list[dict] = []
    missing_images: list[str] = []
    type_counter: Counter[str] = Counter()
    for item in items:
        question_id = item.get("question_id")
        image_id = str(item.get("image_id", "")).strip()
        question = str(item.get("question", "")).strip()
        ground_truth = str(item.get("ground_truth", "")).strip()
        question_type = str(item.get("type", "")).strip()
        dataset = str(item.get("dataset", "")).strip()

        if not isinstance(question_id, int):
            raise ValueError(f"invalid question_id: {question_id}")
        if not image_id:
            raise ValueError(f"missing image_id for question_id={question_id}")
        if not question:
            raise ValueError(f"missing question for question_id={question_id}")
        if not ground_truth:
            raise ValueError(f"missing ground_truth for question_id={question_id}")
        if not question_type:
            raise ValueError(f"missing type for question_id={question_id}")

        image_path = image_dir / image_id
        if not image_path.is_file():
            missing_images.append(str(image_path))

        rows.append(
            {
                "qid": int(question_id),
                "question_id": int(question_id),
                "image_id": image_id,
                "filename": image_id,
                "image_path": str(Path(args.dataset_root) / str(args.image_dir) / image_id),
                "question": question,
                "ground_truth": ground_truth,
                "type": question_type,
                "dataset": dataset,
            }
        )
        type_counter[question_type] += 1

    if missing_images:
        sample = "\n".join(missing_images[:20])
        raise FileNotFoundError(f"found missing images ({len(missing_images)}). sample:\n{sample}")

    rows = sorted(rows, key=lambda row: int(row["qid"]))
    output_jsonl = output_dir / "vrsbench_vqa_test.jsonl"
    output_meta = output_dir / "vrsbench_vqa_meta.json"

    write_jsonl(output_jsonl, rows)
    write_json(
        output_meta,
        {
            "dataset_root": str(Path(args.dataset_root)),
            "annotation_file": str(Path(args.dataset_root) / str(args.annotation_file)),
            "image_dir": str(Path(args.dataset_root) / str(args.image_dir)),
            "num_questions": len(rows),
            "num_images": len({row["image_id"] for row in rows}),
            "type_counts": dict(sorted(type_counter.items())),
        },
    )
    print(f"[OK] Wrote: {output_jsonl}", flush=True)
    print(f"[OK] Wrote: {output_meta}", flush=True)


if __name__ == "__main__":
    main()
