from __future__ import annotations

import argparse
import json
from pathlib import Path


def _project_root() -> Path:
    # benchmark/vrsbench/scripts/*.py -> parents[3] == 项目根目录
    return Path(__file__).resolve().parents[3]


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
    parser = argparse.ArgumentParser(description="Prepare VRSBench caption eval split (test=Images_val).")
    parser.add_argument(
        "--dataset-root",
        type=str,
        default="datasets/VRSBench",
        help="VRSBench root directory. Default is relative to project root.",
    )
    parser.add_argument(
        "--annotation-file",
        type=str,
        default="VRSBench_EVAL_Cap.json",
        help="Caption eval json file name under dataset root.",
    )
    parser.add_argument(
        "--image-dir",
        type=str,
        default="Images_val",
        help="Image directory name under dataset root (caption eval uses val images).",
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

    images_rows: list[dict] = []
    refs: dict[str, list[str]] = {}

    # VRSBench_EVAL_Cap.json: one item per image, with unique question_id (0..9349).
    missing_images: list[str] = []
    for it in items:
        qid = it.get("question_id")
        image_id = str(it.get("image_id", "")).strip()
        gt = str(it.get("ground_truth", "")).strip()
        if not isinstance(qid, int):
            raise ValueError(f"Invalid question_id: {qid}")
        if not image_id:
            raise ValueError(f"Missing image_id for question_id={qid}")
        if not gt:
            raise ValueError(f"Missing ground_truth for question_id={qid}")

        fs_image = img_dir / image_id
        if not fs_image.is_file():
            missing_images.append(str(fs_image))

        # 统一用 imgid=question_id，便于对齐 refs/preds。
        imgid = str(int(qid))

        # 尽量使用相对路径（相对于项目根目录），方便迁移。
        rel_image_path = str(Path(args.dataset_root) / str(args.image_dir) / image_id)

        images_rows.append(
            {
                "imgid": int(qid),
                "filename": image_id,
                "image_path": rel_image_path,
            }
        )
        refs[imgid] = [gt]

    if missing_images:
        # 只展示少量缺失，避免刷屏。
        sample = "\n".join(missing_images[:20])
        raise FileNotFoundError(f"Found missing images ({len(missing_images)}). Sample:\n{sample}")

    images_rows = sorted(images_rows, key=lambda x: int(x["imgid"]))

    images_out = out_dir / "vrsbench_images_test.jsonl"
    refs_out = out_dir / "vrsbench_refs_test.json"
    meta_out = out_dir / "vrsbench_meta.json"

    write_jsonl(images_out, images_rows)
    write_json(refs_out, refs)
    write_json(
        meta_out,
        {
            "dataset_root": str(Path(args.dataset_root)),
            "annotation_file": str(Path(args.dataset_root) / str(args.annotation_file)),
            "image_dir": str(Path(args.dataset_root) / str(args.image_dir)),
            "num_images": len(images_rows),
        },
    )

    print(f"[OK] Wrote: {images_out}")
    print(f"[OK] Wrote: {refs_out}")
    print(f"[OK] Wrote: {meta_out}")


if __name__ == "__main__":
    main()
