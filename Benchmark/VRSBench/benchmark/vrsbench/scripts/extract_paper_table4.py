from __future__ import annotations

import argparse
import json
from pathlib import Path


def _project_root() -> Path:
    # benchmark/vrsbench/scripts/*.py -> parents[3] == 项目根目录
    return Path(__file__).resolve().parents[3]


def _resolve_from_project(path: str | Path) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    return (_project_root() / p).resolve()


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _parse_table4_from_readme(text: str) -> list[dict]:
    lines = [ln.rstrip("\n") for ln in text.splitlines()]

    # 找到 Visual Grounding 表头行
    header_idx = None
    for i, ln in enumerate(lines):
        if ln.strip().startswith("| Method") and "Acc@0.5 (Unique)" in ln and "Acc@0.7 (All)" in ln:
            header_idx = i
            break
    if header_idx is None:
        raise RuntimeError("Failed to locate Visual Grounding table header in README.")

    rows: list[dict] = []
    for ln in lines[header_idx + 1 :]:
        s = ln.strip()
        if not s.startswith("|"):
            # 表格结束
            break
        # 跳过分隔行
        if set(s.replace("|", "").strip()) <= {"-"}:
            continue

        parts = [p.strip() for p in s.strip().strip("|").split("|")]
        if len(parts) < 7:
            continue

        method = parts[0]
        try:
            nums = [float(parts[j]) for j in range(1, 7)]
        except Exception:
            continue

        rows.append(
            {
                "method": method,
                "Acc@0.5 (Unique)": nums[0],
                "Acc@0.7 (Unique)": nums[1],
                "Acc@0.5 (Non Unique)": nums[2],
                "Acc@0.7 (Non Unique)": nums[3],
                "Acc@0.5 (All)": nums[4],
                "Acc@0.7 (All)": nums[5],
            }
        )

    if not rows:
        raise RuntimeError("Parsed 0 rows for Table 4.")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract Table 4 (visual grounding) from VRSBench GitHub README.")
    parser.add_argument(
        "--readme",
        type=str,
        default="benchmark/vrsbench/paper/VRSBench_github_lx709/README.md",
        help="Local copy of the official repository README.",
    )
    parser.add_argument("--output", type=str, default="benchmark/vrsbench/paper/table4_grounding_paper.json")
    args = parser.parse_args()

    readme_path = _resolve_from_project(args.readme)
    if not readme_path.is_file():
        raise FileNotFoundError(f"Missing README: {readme_path}")

    text = readme_path.read_text(encoding="utf-8", errors="replace")
    rows = _parse_table4_from_readme(text)

    out_path = _resolve_from_project(args.output)
    write_json(
        out_path,
        {
            "source": str(Path(args.readme)),
            "table": "Visual Grounding Performance (Table 4 in the paper, mirrored in the official README)",
            "rows": rows,
        },
    )
    print(f"[OK] Wrote: {out_path}")


if __name__ == "__main__":
    main()

