import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create deterministic few-shot splits from VRSBench train260 jsonl."
    )
    parser.add_argument(
        "--input_file",
        type=str,
        default="data/VRSBench/RL_VRSBench_VG_vlmr1_train260_size_new_clip.jsonl",
        help="Source JSONL with 26 classes x 10 samples.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/VRSBench",
        help="Directory to write few-shot JSONL files.",
    )
    parser.add_argument(
        "--shots",
        type=int,
        nargs="+",
        default=[1, 5, 10],
        help="Few-shot counts to generate per class.",
    )
    return parser.parse_args()


def load_records(path: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    if not records:
        raise ValueError(f"No records loaded from {path}")
    return records


def stable_record_key(record: Dict[str, Any]) -> tuple:
    conversations = record.get("conversations", [])
    human = conversations[0].get("value", "") if conversations else ""
    return (
        str(record.get("obj_cls", "")),
        str(record.get("image", "")),
        str(record.get("obj_position", "")),
        str(record.get("obj_size", "")),
        str(record.get("is_unique", "")),
        str(human),
    )


def group_by_class(records: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for record in records:
        obj_cls = record.get("obj_cls")
        if obj_cls is None:
            raise KeyError("Each record must contain 'obj_cls'")
        grouped[str(obj_cls)].append(record)

    for obj_cls, items in grouped.items():
        items.sort(key=stable_record_key)
        if len(items) < 10:
            raise ValueError(f"Class {obj_cls} has only {len(items)} items")
    return dict(sorted(grouped.items()))


def write_jsonl(path: Path, records: List[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    input_path = Path(args.input_file)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    records = load_records(input_path)
    grouped = group_by_class(records)
    num_classes = len(grouped)

    for shot in args.shots:
        if shot <= 0:
            raise ValueError(f"Invalid shot value: {shot}")

        selected: List[Dict[str, Any]] = []
        for obj_cls, items in grouped.items():
            if len(items) < shot:
                raise ValueError(
                    f"Class {obj_cls} has {len(items)} samples, cannot make {shot}-shot split"
                )
            selected.extend(items[:shot])

        selected.sort(key=stable_record_key)
        output_name = f"RL_VRSBench_VG_full_{shot}shots_vlmr1_size_new_clip.jsonl"
        output_path = output_dir / output_name
        write_jsonl(output_path, selected)
        print(
            f"wrote {output_path} | num_classes={num_classes} | "
            f"shot={shot} | num_samples={len(selected)}"
        )


if __name__ == "__main__":
    main()
