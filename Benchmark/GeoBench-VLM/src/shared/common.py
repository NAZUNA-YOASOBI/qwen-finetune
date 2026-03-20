from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def resolve_from_project(path: str | Path) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    if p.exists():
        return p.resolve()
    return (project_root() / p).resolve()


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: str | Path, obj: Any) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def read_jsonl(path: str | Path, *, allow_truncated_last_line: bool = False) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    p = Path(path)
    if not p.exists():
        return rows
    with p.open("r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            raw = line.strip()
            if not raw:
                continue
            try:
                row = json.loads(raw)
            except json.JSONDecodeError as e:
                tail = f.read()
                if allow_truncated_last_line and not tail.strip():
                    print(f"[WARN] Ignore truncated last JSONL line: {p}:{lineno}", file=sys.stderr, flush=True)
                    break
                raise ValueError(f"Invalid JSONL at {p}:{lineno}: {e.msg}") from e
            if not isinstance(row, dict):
                raise ValueError(f"JSONL row must be an object at {p}:{lineno}, got {type(row).__name__}")
            rows.append(row)
    return rows


def append_jsonl(path: str | Path, row: dict[str, Any]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        with out.open("rb+") as f:
            f.seek(0, 2)
            end = f.tell()
            if end > 0:
                f.seek(end - 1)
                if f.read(1) != b"\n":
                    pos = end - 1
                    while pos >= 0:
                        f.seek(pos)
                        if f.read(1) == b"\n":
                            f.truncate(pos + 1)
                            break
                        pos -= 1
                    else:
                        f.truncate(0)
    with out.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def parse_shard_weights(weights: str, *, world_size: int) -> list[int] | None:
    text = str(weights).strip()
    if not text:
        return None
    values = [int(part.strip()) for part in text.split(":") if part.strip()]
    if len(values) != int(world_size):
        raise ValueError(f"shard_weights expects {world_size} values, got {len(values)}: {weights}")
    if any(value <= 0 for value in values):
        raise ValueError(f"shard_weights must be positive integers: {weights}")
    return values


def slice_by_shard(items: list[dict[str, Any]], *, world_size: int, rank: int, weights: str, key_name: str) -> list[dict[str, Any]]:
    if int(world_size) <= 0:
        raise ValueError(f"shard_world_size must be >=1, got {world_size}")
    if int(rank) < 0 or int(rank) >= int(world_size):
        raise ValueError(f"shard_rank out of range: rank={rank}, world_size={world_size}")

    parsed = parse_shard_weights(weights, world_size=int(world_size))
    total = len(items)
    if parsed is None:
        shard = [item for index, item in enumerate(items) if (index % int(world_size)) == int(rank)]
    else:
        denom = int(sum(parsed))
        left = int(sum(parsed[: int(rank)]))
        right = int(sum(parsed[: int(rank) + 1]))
        start = (total * left) // denom
        end = (total * right) // denom
        shard = items[start:end]

    first_key = shard[0].get(key_name, "") if shard else ""
    last_key = shard[-1].get(key_name, "") if shard else ""
    print(
        f"[INFO] shard rank={rank}/{world_size} weights={weights or 'even'} "
        f"selected={len(shard)}/{total} first_{key_name}={first_key} last_{key_name}={last_key}",
        flush=True,
    )
    return shard


def prediction_key(question_id: Any, prompt_index: Any) -> str:
    return f"{question_id}::{prompt_index}"


def resolve_dataset_image_paths(data_root: str | Path, image_value: Any) -> list[Path]:
    root = Path(data_root)
    if isinstance(image_value, list):
        values = image_value
    else:
        values = [image_value]

    paths: list[Path] = []
    for value in values:
        p = Path(str(value))
        if not p.is_absolute():
            p = (root / p).resolve()
        if not p.is_file():
            raise FileNotFoundError(f"Missing image file: {p}")
        paths.append(p)
    return paths
