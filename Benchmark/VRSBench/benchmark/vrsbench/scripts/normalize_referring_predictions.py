from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

_INT_RE = re.compile(r"\d+")
_ANGLE_INT_RE = re.compile(r"<\s*(\d+)\s*>")


def _project_root() -> Path:
    # benchmark/vrsbench/scripts/*.py -> parents[3] == 项目根目录
    return Path(__file__).resolve().parents[3]


def _resolve_from_project(path: str | Path) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    return (_project_root() / p).resolve()


def _extract_box(text: str) -> tuple[str | None, list[int] | None]:
    """
    从字符串中提取 bbox（4 个整数），并规范化成 `{<x1><y1><x2><y2>}`。
    - 优先匹配 `<num>` 形式
    - 否则取“所有数字的前 4 个”（用于规避前缀提示数字影响）
    """

    t = str(text or "")
    nums = _ANGLE_INT_RE.findall(t)
    if len(nums) >= 4:
        vals = [int(x) for x in nums[:4]]
        return f"{{<{vals[0]}><{vals[1]}><{vals[2]}><{vals[3]}>}}", vals

    nums2 = _INT_RE.findall(t)
    if len(nums2) >= 4:
        vals = [int(x) for x in nums2[:4]]
        return f"{{<{vals[0]}><{vals[1]}><{vals[2]}><{vals[3]}>}}", vals

    return None, None


def main() -> None:
    parser = argparse.ArgumentParser(description="Normalize VRSBench referring predictions (clean `answer`).")
    parser.add_argument("--preds", type=str, required=True, help="Input predictions jsonl.")
    parser.add_argument("--out", type=str, default="", help="Output jsonl. Default: in-place overwrite.")
    args = parser.parse_args()

    in_path = _resolve_from_project(args.preds)
    if not in_path.is_file():
        raise FileNotFoundError(f"Missing preds: {in_path}")

    out_path = _resolve_from_project(args.out) if str(args.out).strip() else in_path
    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")

    total = 0
    ok = 0
    bad = 0

    tmp_path.parent.mkdir(parents=True, exist_ok=True)
    with in_path.open("r", encoding="utf-8") as fin, tmp_path.open("w", encoding="utf-8") as fout:
        for ln in fin:
            ln = ln.strip()
            if not ln:
                continue
            total += 1
            row: dict[str, Any] = json.loads(ln)

            raw = str(row.get("answer_raw") or row.get("answer") or "").strip()
            clean_box, clean_vals = _extract_box(raw)
            row["answer_raw"] = raw
            row["answer"] = str(clean_box or "").strip()
            row["answer_parsed"] = clean_vals or []
            row["answer_parse_ok"] = bool(clean_box)

            if clean_box:
                ok += 1
            else:
                bad += 1

            fout.write(json.dumps(row, ensure_ascii=False) + "\n")

    if out_path == in_path:
        # 覆盖原文件
        tmp_path.replace(out_path)
    else:
        # 保留原文件，写到新位置
        tmp_path.replace(out_path)

    print(f"[OK] Wrote: {out_path}")
    print(f"[STAT] total={total} parse_ok={ok} parse_fail={bad}")


if __name__ == "__main__":
    main()
