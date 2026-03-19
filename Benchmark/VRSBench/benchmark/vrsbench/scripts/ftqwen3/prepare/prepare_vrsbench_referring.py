from __future__ import annotations

import argparse
import json
from pathlib import Path


def _project_root() -> Path:
    # benchmark/vrsbench/scripts/ftqwen*/<group>/*.py -> parents[5] == 项目根目录
    return Path(__file__).resolve().parents[5]


def _resolve_from_project(path: str | Path) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    return (_project_root() / p).resolve()


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare VRSBench referring eval split (test=Images_val).")
    parser.add_argument(
        "--dataset-root",
        type=str,
        default="datasets/VRSBench",
        help="VRSBench root directory. Default is relative to project root.",
    )
    parser.add_argument(
        "--annotation-file",
        type=str,
        default="VRSBench_EVAL_referring.json",
        help="Referring eval json file name under dataset root.",
    )
    parser.add_argument(
        "--image-dir",
        type=str,
        default="Images_val",
        help="Image directory name under dataset root (referring eval uses val images).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="benchmark/vrsbench/data",
        help="Where to write prepared artifacts under the project.",
    )
    args = parser.parse_args()

    dataset_root = _resolve_from_project(args.dataset_root)
    ann_path = dataset_root / str(args.annotation_file)
    img_dir = dataset_root / str(args.image_dir)
    out_dir = _resolve_from_project(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not ann_path.is_file():
        raise FileNotFoundError(f"Missing annotation file: {ann_path}")
    if not img_dir.is_dir():
        raise FileNotFoundError(f"Missing image dir: {img_dir}")

    items = json.loads(ann_path.read_text(encoding="utf-8"))
    if not isinstance(items, list):
        raise TypeError(f"Unexpected annotation format: expected list, got {type(items)}")

    rows: list[dict] = []
    missing_images: list[str] = []
    unique_cnt = 0
    non_unique_cnt = 0

    for it in items:
        qid = it.get("question_id")
        image_id = str(it.get("image_id", "")).strip()
        question = str(it.get("question", "")).strip()
        gt = str(it.get("ground_truth", "")).strip()
        unique = bool(it.get("unique", False))
        if not isinstance(qid, int):
            raise ValueError(f"Invalid question_id: {qid}")
        if not image_id:
            raise ValueError(f"Missing image_id for question_id={qid}")
        if not question:
            raise ValueError(f"Missing question for question_id={qid}")
        if not gt:
            raise ValueError(f"Missing ground_truth for question_id={qid}")

        fs_image = img_dir / image_id
        if not fs_image.is_file():
            missing_images.append(str(fs_image))

        if unique:
            unique_cnt += 1
        else:
            non_unique_cnt += 1

        rel_image_path = str(Path(args.dataset_root) / str(args.image_dir) / image_id)
        rows.append(
            {
                "qid": int(qid),
                "filename": image_id,
                "image_id": image_id,
                "image_path": rel_image_path,
                "question": question,
                "ground_truth": gt,
                "is_unique": bool(unique),
                "obj_cls": str(it.get("obj_cls", "")),
                "size_group": str(it.get("size_group", "")),
            }
        )

    if missing_images:
        sample = "\n".join(missing_images[:20])
        raise FileNotFoundError(f"Found missing images ({len(missing_images)}). Sample:\n{sample}")

    rows = sorted(rows, key=lambda x: int(x["qid"]))

    out_jsonl = out_dir / "vrsbench_referring_test.jsonl"
    out_meta = out_dir / "vrsbench_referring_meta.json"

    write_jsonl(out_jsonl, rows)
    write_json(
        out_meta,
        {
            "dataset_root": str(Path(args.dataset_root)),
            "annotation_file": str(Path(args.dataset_root) / str(args.annotation_file)),
            "image_dir": str(Path(args.dataset_root) / str(args.image_dir)),
            "num_samples": len(rows),
            "unique_samples": int(unique_cnt),
            "non_unique_samples": int(non_unique_cnt),
        },
    )

    print(f"[OK] Wrote: {out_jsonl}")
    print(f"[OK] Wrote: {out_meta}")


if __name__ == "__main__":
    main()

