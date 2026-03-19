from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _resolve_from_project(path: str | Path) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    return (_project_root() / p).resolve()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            rows.append(json.loads(ln))
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
    parser = argparse.ArgumentParser(description="Apply jsonl patch rows into a base jsonl by key.")
    parser.add_argument("--base", type=str, required=True, help="Base jsonl file.")
    parser.add_argument("--patches", nargs="+", required=True, help="Patch jsonl files.")
    parser.add_argument("--output", type=str, default="", help="Output file. Default: overwrite base.")
    parser.add_argument("--key", type=str, default="imgid", help="Join key.")
    parser.add_argument("--delete-patches", action="store_true", help="Delete patch files after applying.")
    args = parser.parse_args()

    base_path = _resolve_from_project(args.base)
    out_path = _resolve_from_project(args.output) if str(args.output).strip() else base_path
    patch_paths = [_resolve_from_project(p) for p in args.patches]
    key_name = str(args.key)

    if not base_path.is_file():
        raise FileNotFoundError(f"Missing base jsonl: {base_path}")

    base_rows = read_jsonl(base_path)
    patch_map: dict[int | str, dict[str, Any]] = {}
    for patch_path in patch_paths:
        if not patch_path.is_file():
            raise FileNotFoundError(f"Missing patch jsonl: {patch_path}")
        for row in read_jsonl(patch_path):
            key_val = _key_of(row, key_name)
            if key_val in patch_map:
                raise ValueError(f"Duplicated patch key `{key_name}`={key_val}")
            patch_map[key_val] = row

    patched = 0
    out_rows: list[dict[str, Any]] = []
    for row in base_rows:
        key_val = _key_of(row, key_name)
        patch = patch_map.get(key_val)
        if patch is not None:
            row = dict(row)
            for k, v in patch.items():
                row[k] = v
            patched += 1
        out_rows.append(row)

    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
    write_jsonl(tmp_path, out_rows)
    tmp_path.replace(out_path)
    print(f"[OK] patched_rows={patched} -> {out_path}")

    if bool(args.delete_patches):
        for patch_path in patch_paths:
            patch_path.unlink(missing_ok=True)
            print(f"[OK] deleted patch: {patch_path}")


if __name__ == "__main__":
    main()
