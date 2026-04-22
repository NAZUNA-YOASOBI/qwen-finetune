from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    tmp_path.replace(path)


def key_of(row: dict[str, Any], key_name: str) -> int | str:
    value = row.get(key_name)
    if value is None:
        raise KeyError(f"Missing key `{key_name}` in row: {row}")
    try:
        return int(value)
    except Exception:
        return str(value)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge sharded jsonl files and sort by key.")
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--key", type=str, required=True)
    parser.add_argument("--delete-inputs", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_paths = [Path(path).resolve() for path in args.inputs]
    output_path = Path(args.output).resolve()
    key_name = str(args.key)

    all_rows: list[dict[str, Any]] = []
    for path in input_paths:
        if not path.is_file():
            raise FileNotFoundError(f"Missing input shard: {path}")
        all_rows.extend(read_jsonl(path))

    merged: list[dict[str, Any]] = []
    seen: set[int | str] = set()
    for row in sorted(all_rows, key=lambda item: key_of(item, key_name)):
        key_value = key_of(row, key_name)
        if key_value in seen:
            raise ValueError(f"Duplicated `{key_name}`={key_value} when merging shards")
        seen.add(key_value)
        merged.append(row)

    write_jsonl(output_path, merged)
    print(f"[OK] merged_rows={len(merged)} -> {output_path}", flush=True)

    if bool(args.delete_inputs):
        for path in input_paths:
            path.unlink(missing_ok=True)
            print(f"[OK] deleted shard: {path}", flush=True)


if __name__ == "__main__":
    main()
