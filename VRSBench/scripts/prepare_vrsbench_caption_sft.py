from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _resolve_from_project(path: str | Path) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    return (_project_root() / p).resolve()


def _rel_to_project(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(_project_root()))
    except Exception:
        return str(path.resolve())


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare VRSBench caption data for SFT training (jsonl).")
    parser.add_argument(
        "--dataset-root",
        type=str,
        default="datasets/VRSBench",
        help="VRSBench dataset root (contains Images_train/Annotations_train).",
    )
    parser.add_argument("--split", type=str, default="train", choices=["train", "val"])
    parser.add_argument("--output-dir", type=str, default="data/vrsbench")
    args = parser.parse_args()

    dataset_root = _resolve_from_project(args.dataset_root)
    out_dir = _resolve_from_project(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ann_dir = dataset_root / ("Annotations_train" if str(args.split) == "train" else "Annotations_val")
    img_dir = dataset_root / ("Images_train" if str(args.split) == "train" else "Images_val")

    if not ann_dir.is_dir():
        raise FileNotFoundError(f"Missing annotations dir: {ann_dir}")
    if not img_dir.is_dir():
        raise FileNotFoundError(f"Missing images dir: {img_dir}")

    rows: list[dict[str, Any]] = []
    missing_images: list[str] = []

    ann_files = sorted(ann_dir.glob("*.json"))
    for p in ann_files:
        obj = json.loads(p.read_text(encoding="utf-8"))
        image_name = str(obj.get("image", "")).strip()
        caption = str(obj.get("caption", "")).strip()
        if not image_name or not caption:
            continue
        img_path = img_dir / image_name
        if not img_path.is_file():
            missing_images.append(image_name)
            continue

        # 训练代码会把 image_path 当作“相对项目根目录”的路径来解析，这里用 relpath 固定写成 ../../datasets/... 形式，避免路径漂移。
        import os

        rel_image_path_str = os.path.relpath(str(img_path.resolve()), str(_project_root()))

        rows.append(
            {
                "imgid": -1,  # 先占位，后面统一按排序重编号
                "filename": image_name,
                "image_path": rel_image_path_str,
                "caption": caption,
                "caption_idx": 0,
                "split": str(args.split),
            }
        )

    # 固定排序 + 重编号，保证可复现
    rows = sorted(rows, key=lambda x: str(x.get("filename", "")))
    for i, r in enumerate(rows):
        r["imgid"] = int(i)

    out_jsonl = out_dir / f"vrsbench_captions_{args.split}.jsonl"
    meta = {
        "dataset_root": _rel_to_project(dataset_root),
        "split": str(args.split),
        "num_rows": int(len(rows)),
        "annotations_dir": _rel_to_project(ann_dir),
        "images_dir": _rel_to_project(img_dir),
        "missing_images": missing_images[:50],
        "missing_images_count": int(len(missing_images)),
        "output": _rel_to_project(out_jsonl),
    }
    out_meta = out_dir / f"vrsbench_meta_{args.split}.json"

    write_jsonl(out_jsonl, rows)
    out_meta.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"[OK] Wrote: {out_jsonl} (rows={len(rows)})")
    if missing_images:
        print(f"[WARN] Missing images referenced by annotations: {len(missing_images)} (showing up to 10)")
        for x in missing_images[:10]:
            print(f"  - {x}")
    print(f"[OK] Wrote: {out_meta}")


if __name__ == "__main__":
    main()
