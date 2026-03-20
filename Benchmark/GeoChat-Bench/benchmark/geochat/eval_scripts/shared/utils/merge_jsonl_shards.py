from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any


def _project_root() -> Path:
    return Path(__file__).resolve().parents[5]


def _resolve_from_project(path: str | Path) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    return (_project_root() / p).resolve()

def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    tmp_path.replace(path)


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
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--key", type=str, required=True)
    parser.add_argument("--delete-inputs", action="store_true")
    args = parser.parse_args()

    in_paths = [_resolve_from_project(p) for p in args.inputs]
    out_path = _resolve_from_project(args.output)
    key_name = str(args.key)
    sys.path.insert(0, str(_project_root() / "src"))
    from shared.common import read_jsonl

    all_rows: list[dict[str, Any]] = []
    for path in in_paths:
        if not path.is_file():
            raise FileNotFoundError(f"Missing input shard: {path}")
        all_rows.extend(read_jsonl(path, allow_truncated_last_line=True))

    merged: list[dict[str, Any]] = []
    seen: set[int | str] = set()
    for row in sorted(all_rows, key=lambda r: _key_of(r, key_name)):
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
