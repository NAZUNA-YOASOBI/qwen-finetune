from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _project_root() -> Path:
    return Path(__file__).resolve().parents[5]


def _resolve_from_project(path: str | Path) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    return (_project_root() / p).resolve()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _key_of(row: dict[str, Any], key: str) -> int | str:
    value = row.get(key)
    if value is None:
        raise KeyError(f"Missing key `{key}` in row: {row}")
    try:
        return int(value)
    except Exception:
        return str(value)


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge sharded jsonl files and sort by key.")
    parser.add_argument("--inputs", nargs="+", required=True, help="Input shard jsonl files.")
    parser.add_argument("--output", type=str, required=True, help="Merged output jsonl file.")
    parser.add_argument("--key", type=str, required=True, help="Sort / dedupe key, e.g. uid or qid.")
    parser.add_argument("--delete-inputs", action="store_true", help="Delete input shard files after merge.")
    args = parser.parse_args()

    in_paths = [_resolve_from_project(path) for path in args.inputs]
    out_path = _resolve_from_project(args.output)
    key_name = str(args.key)

    all_rows: list[dict[str, Any]] = []
    for path in in_paths:
        if not path.is_file():
            raise FileNotFoundError(f"Missing input shard: {path}")
        all_rows.extend(read_jsonl(path))

    seen: set[int | str] = set()
    merged: list[dict[str, Any]] = []
    for row in sorted(all_rows, key=lambda item: _key_of(item, key_name)):
        key_val = _key_of(row, key_name)
        if key_val in seen:
            raise ValueError(f"Duplicated `{key_name}`={key_val} when merging shards")
        seen.add(key_val)
        merged.append(row)

    write_jsonl(out_path, merged)
    print(f"[OK] merged_rows={len(merged)} -> {out_path}")

    if bool(args.delete_inputs):
        for path in in_paths:
            path.unlink(missing_ok=True)
            print(f"[OK] deleted shard: {path}")


if __name__ == "__main__":
    main()
