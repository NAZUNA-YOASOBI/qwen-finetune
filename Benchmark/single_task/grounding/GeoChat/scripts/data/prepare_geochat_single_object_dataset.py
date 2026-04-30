from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[5]
TRAIN_DATA_PATH = PROJECT_ROOT / "GeoChat-Bench" / "dataset" / "clean" / "GeoChat_Instruct" / "GeoChat_Instruct_clean.json"
TEST_DATA_PATH = PROJECT_ROOT / "GeoChat-Bench" / "dataset" / "GeoChat-Bench" / "referring.jsonl"
IMAGE_ROOT = (
    PROJECT_ROOT
    / "GeoChat-Bench"
    / "dataset"
    / "raw"
    / "GeoChat_Instruct"
    / "images"
    / "share"
    / "softwares"
    / "kartik"
    / "GeoChat_finetuning"
    / "final_images_llava"
)
OUTPUT_ROOT = (
    PROJECT_ROOT
    / "VRSBench"
    / "benchmark"
    / "single_task"
    / "datasets"
    / "grounding"
    / "GeoChat"
)
OUTPUT_DATA_ROOT = OUTPUT_ROOT / "benchmark" / "data"

BOX_PATTERN = re.compile(r"\{<[^{}]+>\}")
PTAG_PATTERN = re.compile(r"<p>(.*?)</p>", flags=re.IGNORECASE | re.DOTALL)


@dataclass(frozen=True)
class ImageIndex:
    by_basename: dict[str, list[Path]]
    by_stem: dict[str, list[Path]]


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def strip_image_placeholder(text: str) -> str:
    stripped = str(text).replace("<image>", "").strip()
    if stripped.startswith("\n"):
        stripped = stripped.lstrip("\n").strip()
    return stripped


def detect_train_task(question: str) -> str | None:
    lowered = str(question).lower()
    if "[refer]" in lowered:
        return "refer"
    return None


def count_boxes(answer: str) -> int:
    return len(BOX_PATTERN.findall(str(answer)))


def normalize_rel_path(path: Path) -> str:
    return path.resolve().relative_to(PROJECT_ROOT).as_posix()


def build_image_index(image_root: Path) -> ImageIndex:
    by_basename_sets: dict[str, set[Path]] = defaultdict(set)
    by_stem_sets: dict[str, set[Path]] = defaultdict(set)
    for path in image_root.rglob("*"):
        if not path.is_file():
            continue
        resolved = path.resolve()
        by_basename_sets[path.name].add(resolved)
        by_stem_sets[path.stem].add(resolved)
    by_basename = {key: sorted(values) for key, values in by_basename_sets.items()}
    by_stem = {key: sorted(values) for key, values in by_stem_sets.items()}
    return ImageIndex(by_basename=by_basename, by_stem=by_stem)


def resolve_image_path(raw_identifier: str, image_index: ImageIndex, preferred_suffix: str | None = None) -> Path:
    raw = str(raw_identifier).strip()
    if not raw:
        raise ValueError("Empty image identifier.")

    direct_path = (IMAGE_ROOT / raw).resolve()
    if direct_path.is_file():
        return direct_path

    raw_path = Path(raw)
    if preferred_suffix and not raw_path.suffix:
        preferred_name = f"{raw}{preferred_suffix}"
        preferred_direct = (IMAGE_ROOT / preferred_name).resolve()
        if preferred_direct.is_file():
            return preferred_direct
        preferred_hits = image_index.by_basename.get(preferred_name, [])
        if len(preferred_hits) == 1:
            return preferred_hits[0]
        if len(preferred_hits) > 1:
            raise RuntimeError(f"Ambiguous preferred suffix match for image '{raw}': {preferred_hits[:4]}")

    basename = raw_path.name
    basename_hits = image_index.by_basename.get(basename, [])
    if len(basename_hits) == 1:
        return basename_hits[0]
    if len(basename_hits) > 1:
        raise RuntimeError(f"Ambiguous basename match for image '{raw}': {basename_hits[:4]}")

    stem = raw_path.stem if raw_path.suffix else raw
    stem_hits = image_index.by_stem.get(stem, [])
    if len(stem_hits) == 1:
        return stem_hits[0]
    if len(stem_hits) > 1:
        raise RuntimeError(f"Ambiguous stem match for image '{raw}': {stem_hits[:4]}")

    raise FileNotFoundError(f"Cannot resolve image '{raw}' under {IMAGE_ROOT}")


def extract_query_text(task: str, prompt: str) -> str:
    prompt_text = str(prompt).strip()
    if task == "refer":
        matched = PTAG_PATTERN.search(prompt_text)
        if matched is not None:
            return str(matched.group(1)).strip()
    return prompt_text


def build_eval_prompt(task: str, question: str) -> str:
    raw_question = str(question).strip()
    if task == "refer":
        return f"[refer] where is <p>{raw_question}</p> ?"
    raise ValueError(f"Unsupported task: {task}")


def build_train_rows(train_items: list[dict[str, Any]], image_index: ImageIndex) -> tuple[list[dict[str, Any]], Counter]:
    rows: list[dict[str, Any]] = []
    stats: Counter = Counter()
    next_question_id = 0

    for item_index, item in enumerate(train_items):
        image_path = resolve_image_path(str(item.get("image", "")), image_index)
        conversations = item.get("conversations", [])
        if not isinstance(conversations, list) or len(conversations) < 2:
            stats["skipped_invalid_conversations"] += 1
            continue

        for turn_index in range(len(conversations) - 1):
            human = conversations[turn_index]
            assistant = conversations[turn_index + 1]
            if str(human.get("from", "")).lower() != "human":
                continue
            if str(assistant.get("from", "")).lower() != "gpt":
                continue

            question = strip_image_placeholder(str(human.get("value", "")))
            answer = str(assistant.get("value", "")).strip()
            task = detect_train_task(question)
            if task is None:
                continue

            stats[f"{task}_pairs_total"] += 1
            box_count = count_boxes(answer)
            stats[f"{task}_box_count_{box_count}"] += 1

            if box_count != 1:
                stats[f"{task}_dropped_multi_or_empty"] += 1
                continue

            rows.append(
                {
                    "question_id": int(next_question_id),
                    "task": str(task),
                    "split": "train",
                    "image_id": image_path.name,
                    "image_rel_path": normalize_rel_path(image_path),
                    "question": str(question),
                    "query_text": extract_query_text(task, question),
                    "ground_truth": str(answer),
                    "source_dataset": "GeoChat_Instruct_clean",
                    "source_item_id": str(item.get("id", "")),
                    "source_image_field": str(item.get("image", "")),
                    "source_item_index": int(item_index),
                    "source_turn_index": int(turn_index),
                    "box_count": int(box_count),
                }
            )
            next_question_id += 1
            stats[f"{task}_single_object_kept"] += 1

    rows.sort(key=lambda row: int(row["question_id"]))
    return rows, stats


def build_test_rows(test_items: list[dict[str, Any]], image_index: ImageIndex) -> tuple[list[dict[str, Any]], Counter]:
    rows: list[dict[str, Any]] = []
    stats: Counter = Counter()

    for item in test_items:
        raw_type = str(item.get("type", "")).strip().lower()
        if raw_type != "ref":
            continue

        obj_ids = item.get("obj_ids") or []
        object_count = int(len(obj_ids))
        stats["refer_samples_total"] += 1
        stats[f"refer_obj_count_{object_count}"] += 1
        if object_count != 1:
            stats["refer_dropped_multi_object"] += 1
            continue

        image_path = resolve_image_path(str(item.get("image_id", "")), image_index, preferred_suffix=".png")
        question = str(item.get("question", "")).strip()
        rows.append(
            {
                "question_id": str(item.get("question_id", "")),
                "task": "refer",
                "split": "test",
                "image_id": image_path.name,
                "image_rel_path": normalize_rel_path(image_path),
                "question": str(question),
                "prompt": build_eval_prompt("refer", question),
                "ground_truth": item.get("ground_truth"),
                "source_dataset": str(item.get("dataset", "")),
                "source_type": str(raw_type),
                "source_image_id": str(item.get("image_id", "")),
                "obj_ids": list(obj_ids),
                "size_group": str(item.get("size_group", "")),
                "object_count": int(object_count),
            }
        )
        stats["refer_single_object_kept"] += 1

    rows.sort(key=lambda row: (str(row["task"]), str(row["question_id"])))
    return rows, stats


def split_rows_by_task(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {"refer": []}
    for row in rows:
        task = str(row["task"])
        if task not in grouped:
            raise ValueError(f"Unexpected task in rows: {task}")
        grouped[task].append(row)
    return grouped


def main() -> None:
    if not TRAIN_DATA_PATH.is_file():
        raise FileNotFoundError(f"Missing training source: {TRAIN_DATA_PATH}")
    if not TEST_DATA_PATH.is_file():
        raise FileNotFoundError(f"Missing test source: {TEST_DATA_PATH}")
    if not IMAGE_ROOT.is_dir():
        raise FileNotFoundError(f"Missing image root: {IMAGE_ROOT}")

    image_index = build_image_index(IMAGE_ROOT)
    train_items = read_json(TRAIN_DATA_PATH)
    test_items = read_jsonl(TEST_DATA_PATH)

    train_rows, train_stats = build_train_rows(train_items, image_index)
    test_rows, test_stats = build_test_rows(test_items, image_index)

    train_grouped = split_rows_by_task(train_rows)
    test_grouped = split_rows_by_task(test_rows)

    write_json(OUTPUT_DATA_ROOT / "train_single_object.json", train_rows)
    write_json(OUTPUT_DATA_ROOT / "train.json", train_grouped["refer"])
    write_json(OUTPUT_DATA_ROOT / "test.json", test_rows)
    write_json(OUTPUT_DATA_ROOT / "test_refer_single_object.json", test_grouped["refer"])

    summary = {
        "dataset_name": "GeoChat",
        "output_root": normalize_rel_path(OUTPUT_ROOT),
        "source": {
            "train_json": normalize_rel_path(TRAIN_DATA_PATH),
            "test_jsonl": normalize_rel_path(TEST_DATA_PATH),
            "image_root": normalize_rel_path(IMAGE_ROOT),
        },
        "notes": [
            "训练集来自 GeoChat_Instruct_clean，只抽取 [refer] 相邻 human->gpt 对。",
            "多物体清洗规则：训练集只保留答案中恰好 1 个 GeoChat box 的样本；测试集只保留 obj_ids 长度等于 1 的样本。",
            "测试集保留 benchmark 原始 polygon ground truth 和可直接推理的 prompt。",
        ],
        "counts": {
            "train_total_kept": int(len(train_rows)),
            "train_refer_kept": int(len(train_grouped["refer"])),
            "test_total_kept": int(len(test_rows)),
            "test_refer_kept": int(len(test_grouped["refer"])),
        },
        "train_stats": dict(train_stats),
        "test_stats": dict(test_stats),
    }
    write_json(OUTPUT_DATA_ROOT / "dataset_info.json", summary)

    print(json.dumps(summary["counts"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
