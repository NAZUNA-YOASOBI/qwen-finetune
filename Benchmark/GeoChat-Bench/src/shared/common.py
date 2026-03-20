from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any, Iterable


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
                    print(
                        f"[WARN] Ignore truncated last JSONL line: {p}:{lineno}",
                        file=sys.stderr,
                        flush=True,
                    )
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
    w = str(weights).strip()
    if not w:
        return None
    vals = [int(x.strip()) for x in w.split(":") if x.strip()]
    if len(vals) != int(world_size):
        raise ValueError(f"shard_weights expects {world_size} values, got {len(vals)}: {weights}")
    if any(v <= 0 for v in vals):
        raise ValueError(f"shard_weights must be positive integers: {weights}")
    return vals


def slice_by_shard(items: list[dict[str, Any]], *, world_size: int, rank: int, weights: str, key_name: str) -> list[dict[str, Any]]:
    if int(world_size) <= 0:
        raise ValueError(f"shard_world_size must be >=1, got {world_size}")
    if int(rank) < 0 or int(rank) >= int(world_size):
        raise ValueError(f"shard_rank out of range: rank={rank}, world_size={world_size}")

    parsed = parse_shard_weights(weights, world_size=int(world_size))
    total = len(items)
    if parsed is None:
        shard = [it for i, it in enumerate(items) if (i % int(world_size)) == int(rank)]
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


def load_done_keys(path: str | Path, *, key_name: str) -> set[str]:
    done: set[str] = set()
    for row in read_jsonl(path, allow_truncated_last_line=True):
        if key_name in row:
            done.add(str(row[key_name]))
    return done


def normalize_scene_label(text: str) -> str:
    return re.sub(r"[\s.]+", "", str(text or "").strip().lower())


def normalize_free_text(text: str) -> str:
    s = str(text or "").strip().lower()
    s = s.replace("\n", " ")
    s = re.sub(r"\s+", " ", s)
    s = s.strip(" \t\r\n.,;:!?\"'")
    return s


def find_image_path(
    image_root: str | Path,
    *,
    image_value: str | None = None,
    image_id: str | None = None,
    default_ext: str | None = None,
) -> Path:
    root = Path(image_root)
    candidates: list[Path] = []

    if image_value:
        rel = Path(str(image_value))
        candidates.append(root / rel)
        if rel.suffix:
            candidates.append(root / rel.with_suffix(rel.suffix.lower()))
            candidates.append(root / rel.with_suffix(rel.suffix.upper()))
            alt_exts = [".png", ".jpg", ".jpeg", ".tif", ".tiff", ".JPG", ".PNG", ".TIF", ".TIFF"]
            for ext in alt_exts:
                candidates.append(root / rel.with_suffix(ext))

    if image_id:
        raw = str(image_id)
        raw_path = Path(raw)
        if raw_path.suffix:
            candidates.append(root / raw_path)
        exts = []
        if default_ext:
            exts.append(str(default_ext))
        exts.extend([".png", ".jpg", ".jpeg", ".tif", ".tiff", ".JPG", ".PNG", ".TIF", ".TIFF"])
        seen_exts: set[str] = set()
        for ext in exts:
            if ext in seen_exts:
                continue
            seen_exts.add(ext)
            candidates.append(root / f"{raw}{ext}")

    seen: set[str] = set()
    uniq: list[Path] = []
    for cand in candidates:
        key = str(cand)
        if key in seen:
            continue
        seen.add(key)
        uniq.append(cand)

    for cand in uniq:
        if cand.is_file():
            return cand.resolve()

    tried = ", ".join(str(x) for x in uniq[:8])
    raise FileNotFoundError(f"Could not resolve image file under {root}. Tried: {tried}")


def flatten_text_lines(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "").strip())


def maybe_int(value: Any) -> int | None:
    try:
        return int(value)
    except Exception:
        return None


def read_json_or_jsonl(path: str | Path) -> Any:
    p = Path(path)
    if p.suffix.lower() == ".jsonl":
        return read_jsonl(p)
    return read_json(p)


def iter_rows_from_json_or_jsonl(path: str | Path) -> Iterable[dict[str, Any]]:
    data = read_json_or_jsonl(path)
    if isinstance(data, list):
        for item in data:
            if isinstance(item, dict):
                yield item
        return
    if isinstance(data, dict):
        for key in ("questions", "annotations", "data", "rows", "items", "answers", "images"):
            value = data.get(key, None)
            if isinstance(value, list):
                for item in value:
                    if isinstance(item, dict):
                        yield item
                return
    raise ValueError(f"Unsupported JSON/JSONL structure: {path}")
