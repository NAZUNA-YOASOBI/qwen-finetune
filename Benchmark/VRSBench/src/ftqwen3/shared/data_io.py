from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


def read_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: Path, obj: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    path = Path(path)
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as file_obj:
        for line in file_obj:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file_obj:
        for row in rows:
            file_obj.write(json.dumps(row, ensure_ascii=False) + "\n")


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file_obj:
        file_obj.write(json.dumps(row, ensure_ascii=False) + "\n")


@dataclass(frozen=True)
class RsicdImageItem:
    imgid: int
    filename: str
    split: str
    captions: list[str]


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _rel_to_project(path: Path) -> str:
    path = Path(path)
    try:
        return str(path.resolve().relative_to(_project_root()))
    except Exception:
        return str(path.resolve())


def load_rsicd_items(rsicd_json: Path) -> list[RsicdImageItem]:
    rsicd_json = Path(rsicd_json)
    obj = json.loads(rsicd_json.read_text(encoding="utf-8"))
    if not isinstance(obj, dict) or "images" not in obj:
        raise ValueError(f"Unexpected RSICD format: {rsicd_json}")

    items: list[RsicdImageItem] = []
    for item in obj["images"]:
        imgid = int(item["imgid"])
        filename = str(item["filename"])
        split = str(item["split"])
        sentences = item.get("sentences", [])
        captions: list[str] = []
        for sentence in sentences:
            raw = str(sentence.get("raw", "")).strip()
            if raw:
                captions.append(raw)
        if not captions:
            continue
        items.append(RsicdImageItem(imgid=imgid, filename=filename, split=split, captions=captions))
    return items


def build_rsicd_refs(items: Iterable[RsicdImageItem]) -> dict[str, list[str]]:
    refs: dict[str, list[str]] = {}
    for item in items:
        refs[str(item.imgid)] = list(item.captions)
    return refs


def build_rsicd_caption_rows(items: Iterable[RsicdImageItem], *, image_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in items:
        for caption_idx, caption in enumerate(item.captions):
            rows.append(
                {
                    "imgid": int(item.imgid),
                    "filename": item.filename,
                    "image_path": _rel_to_project(Path(image_dir) / item.filename),
                    "caption": caption,
                    "caption_idx": int(caption_idx),
                    "split": item.split,
                }
            )
    return rows


def build_rsicd_image_rows(items: Iterable[RsicdImageItem], *, image_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in items:
        rows.append(
            {
                "imgid": int(item.imgid),
                "filename": item.filename,
                "image_path": _rel_to_project(Path(image_dir) / item.filename),
                "refs": list(item.captions),
                "split": item.split,
            }
        )
    return rows


def prepare_rsicd_dataset(*, rsicd_dir: Path, out_dir: Path) -> None:
    rsicd_dir = Path(rsicd_dir)
    out_dir = Path(out_dir)

    ann_path = rsicd_dir / "dataset_rsicd.json"
    image_dir = rsicd_dir / "RSICD_images"

    if not ann_path.is_file():
        raise FileNotFoundError(f"Missing annotation file: {ann_path}")
    if not image_dir.is_dir():
        raise FileNotFoundError(f"Missing image directory: {image_dir}")

    items = load_rsicd_items(ann_path)
    split_counter = Counter([item.split for item in items])

    missing: list[str] = []
    for item in items:
        if not (image_dir / item.filename).is_file():
            missing.append(item.filename)
    if missing:
        raise FileNotFoundError(f"Missing {len(missing)} images under {image_dir}. Sample: {missing[:5]}")

    meta = {
        "rsicd_dir": _rel_to_project(rsicd_dir),
        "annotation_file": _rel_to_project(ann_path),
        "image_dir": _rel_to_project(image_dir),
        "num_images": len(items),
        "splits": dict(split_counter),
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / "rsicd_meta.json", meta)

    for split in ["train", "val", "test"]:
        split_items = [item for item in items if item.split == split]
        write_json(out_dir / f"rsicd_refs_{split}.json", build_rsicd_refs(split_items))

    for split in ["train", "val", "test"]:
        split_items = [item for item in items if item.split == split]
        write_jsonl(out_dir / f"rsicd_images_{split}.jsonl", build_rsicd_image_rows(split_items, image_dir=image_dir))

    for split in ["train", "val", "test"]:
        split_items = [item for item in items if item.split == split]
        write_jsonl(
            out_dir / f"rsicd_captions_{split}.jsonl",
            build_rsicd_caption_rows(split_items, image_dir=image_dir),
        )
