from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


COLUMNS = [
    "Identity",
    "Color",
    "Orientation",
    "Shape",
    "Area",
    "Resolution",
    "Modality",
    "Location",
    "Distance",
    "Quantity",
    "Reasoning",
    "Avg",
]


def _project_root() -> Path:
    return Path(__file__).resolve().parents[5]


def _resolve_from_project(path: str | Path) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    return (_project_root() / p).resolve()


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _fmt(value: float | int | str | None) -> str:
    if value is None:
        return "-"
    if isinstance(value, (int, float)):
        return f"{float(value):.2f}"
    return str(value)


def _ours_row(method_name: str, summary: dict[str, Any]) -> dict[str, Any]:
    type_pct = dict(summary.get("metrics_by_type_pct", {}))
    row = {"Method": method_name}
    for column in COLUMNS[:-1]:
        row[column] = float(type_pct.get(column.lower(), type_pct.get(column, 0.0)))
    row["Avg"] = float(summary.get("avg_pct", 0.0))
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description="Build LHRS-Bench comparison markdown.")
    parser.add_argument("--paper", type=str, required=True)
    parser.add_argument("--baseline", type=str, required=True)
    parser.add_argument("--ours-summary", type=str, required=True)
    parser.add_argument("--ours-name", type=str, default="Ours")
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()

    paper = _load_json(_resolve_from_project(args.paper))
    baseline = _load_json(_resolve_from_project(args.baseline))
    ours_summary = _load_json(_resolve_from_project(args.ours_summary))

    rows: list[dict[str, Any]] = []
    for row in paper.get("methods", []):
        rows.append(dict(row))
    rows.append(_ours_row("Ours-baseline8b", baseline))
    rows.append(_ours_row(str(args.ours_name).strip() or "Ours", ours_summary))

    out_path = _resolve_from_project(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    lines: list[str] = []
    lines.append("# LHRS-Bench 指标对比（Paper vs Ours）")
    lines.append(f"- Paper 来源：`{paper.get('source', '')}`")
    lines.append(f"- Baseline summary：`{args.baseline}`")
    lines.append(f"- Ours summary：`{args.ours_summary}`")
    lines.append("")
    lines.append("| Method | " + " | ".join(COLUMNS) + " |")
    lines.append("|---|" + "---:|" * len(COLUMNS))

    for row in rows:
        values = [_fmt(row.get(column)) for column in COLUMNS]
        lines.append("| " + str(row.get("Method", "")) + " | " + " | ".join(values) + " |")

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[OK] wrote: {out_path}")


if __name__ == "__main__":
    main()
