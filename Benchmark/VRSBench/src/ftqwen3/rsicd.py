from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .jsonl import write_json, write_jsonl


@dataclass(frozen=True)
class RsicdImageItem:
    imgid: int
    filename: str
    split: str
    captions: list[str]


def _project_root() -> Path:
    # src/ftqwen/rsicd.py -> parents[2] == 项目根目录
    return Path(__file__).resolve().parents[2]


def _rel_to_project(path: Path) -> str:
    p = Path(path)
    try:
        return str(p.resolve().relative_to(_project_root()))
    except Exception:
        return str(p.resolve())


def load_rsicd_items(rsicd_json: Path) -> list[RsicdImageItem]:
    """Load RSICD from dataset_rsicd.json.

    RSICD_optimal 的标注格式是一个 dict，包含 `images` 列表。
    每个 image 里有 5 条 `sentences`，每条有 `raw` 字段。
    """
    rsicd_json = Path(rsicd_json)
    obj = json.loads(rsicd_json.read_text(encoding="utf-8"))
    if not isinstance(obj, dict) or "images" not in obj:
        raise ValueError(f"Unexpected RSICD format: {rsicd_json}")

    items: list[RsicdImageItem] = []
    for it in obj["images"]:
        imgid = int(it["imgid"])
        filename = str(it["filename"])
        split = str(it["split"])
        sentences = it.get("sentences", [])
        captions: list[str] = []
        for s in sentences:
            raw = str(s.get("raw", "")).strip()
            if raw:
                captions.append(raw)
        if not captions:
            # 极少数情况下可能存在标注异常；这里直接跳过，避免后续训练/评测报错。
            continue
        items.append(RsicdImageItem(imgid=imgid, filename=filename, split=split, captions=captions))
    return items


def build_rsicd_refs(items: Iterable[RsicdImageItem]) -> dict[str, list[str]]:
    refs: dict[str, list[str]] = {}
    for it in items:
        refs[str(it.imgid)] = list(it.captions)
    return refs


def build_rsicd_caption_rows(
    items: Iterable[RsicdImageItem], *, image_dir: Path
) -> list[dict[str, Any]]:
    """One row per (image, caption). Useful for supervised caption training."""
    rows: list[dict[str, Any]] = []
    for it in items:
        for cap_idx, cap in enumerate(it.captions):
            rows.append(
                {
                    "imgid": int(it.imgid),
                    "filename": it.filename,
                    "image_path": _rel_to_project(Path(image_dir) / it.filename),
                    "caption": cap,
                    "caption_idx": int(cap_idx),
                    "split": it.split,
                }
            )
    return rows


def build_rsicd_image_rows(
    items: Iterable[RsicdImageItem], *, image_dir: Path
) -> list[dict[str, Any]]:
    """One row per image, keep all refs. Useful for inference/eval."""
    rows: list[dict[str, Any]] = []
    for it in items:
        rows.append(
            {
                "imgid": int(it.imgid),
                "filename": it.filename,
                "image_path": _rel_to_project(Path(image_dir) / it.filename),
                "refs": list(it.captions),
                "split": it.split,
            }
        )
    return rows


def prepare_rsicd_dataset(*, rsicd_dir: Path, out_dir: Path) -> None:
    """Prepare RSICD json/jsonl artifacts for training and evaluation."""
    rsicd_dir = Path(rsicd_dir)
    out_dir = Path(out_dir)

    ann_path = rsicd_dir / "dataset_rsicd.json"
    image_dir = rsicd_dir / "RSICD_images"

    if not ann_path.is_file():
        raise FileNotFoundError(f"Missing annotation file: {ann_path}")
    if not image_dir.is_dir():
        raise FileNotFoundError(f"Missing image directory: {image_dir}")

    items = load_rsicd_items(ann_path)
    split_counter = Counter([it.split for it in items])

    # 检查文件是否存在，避免后续跑一半才报错。
    missing: list[str] = []
    for it in items:
        if not (image_dir / it.filename).is_file():
            missing.append(it.filename)
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

    # 1) refs: imgid -> [ref captions]
    for split in ["train", "val", "test"]:
        split_items = [it for it in items if it.split == split]
        write_json(out_dir / f"rsicd_refs_{split}.json", build_rsicd_refs(split_items))

    # 2) images: one line per image (keep refs)
    for split in ["train", "val", "test"]:
        split_items = [it for it in items if it.split == split]
        write_jsonl(out_dir / f"rsicd_images_{split}.jsonl", build_rsicd_image_rows(split_items, image_dir=image_dir))

    # 3) captions: one line per (image, caption)
    for split in ["train", "val", "test"]:
        split_items = [it for it in items if it.split == split]
        write_jsonl(
            out_dir / f"rsicd_captions_{split}.jsonl",
            build_rsicd_caption_rows(split_items, image_dir=image_dir),
        )
