from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def _project_root() -> Path:
    # benchmark/vrsbench/scripts/*.py -> parents[3] == 项目根目录
    return Path(__file__).resolve().parents[3]


def _resolve_from_project(path: str | Path) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    return (_project_root() / p).resolve()


def _rel_to_project(path: Path) -> str:
    p = Path(path)
    try:
        return str(p.resolve().relative_to(_project_root()))
    except Exception:
        return str(p.resolve())


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _fmt(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, (int, float)):
        return f"{v:.2f}"
    return str(v)


def _get_acc(summary: dict[str, Any], *, split: str, key: str) -> float | None:
    try:
        return float(summary["splits"][split][key])
    except Exception:
        return None


def _our_row(method: str, summary_path: Path) -> dict[str, Any]:
    s = read_json(summary_path)
    return {
        "method": method,
        "Acc@0.5 (Unique)": _get_acc(s, split="unique", key="Acc@0.5"),
        "Acc@0.7 (Unique)": _get_acc(s, split="unique", key="Acc@0.7"),
        "Acc@0.5 (Non Unique)": _get_acc(s, split="non_unique", key="Acc@0.5"),
        "Acc@0.7 (Non Unique)": _get_acc(s, split="non_unique", key="Acc@0.7"),
        "Acc@0.5 (All)": _get_acc(s, split="all", key="Acc@0.5"),
        "Acc@0.7 (All)": _get_acc(s, split="all", key="Acc@0.7"),
        "summary": str(summary_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a unified VRSBench grounding report table.")
    parser.add_argument(
        "--paper-table",
        type=str,
        default="benchmark/vrsbench/paper/table4_grounding_paper.json",
        help="Extracted paper Table 4 (json).",
    )
    parser.add_argument(
        "--baseline",
        type=str,
        required=True,
        help="Baseline referring eval summary json.",
    )
    parser.add_argument(
        "--merger-only",
        type=str,
        required=True,
        help="Merger-only referring eval summary json.",
    )
    parser.add_argument(
        "--merger-lora",
        type=str,
        required=True,
        help="Merger+LoRA referring eval summary json.",
    )
    parser.add_argument("--out-md", type=str, default="benchmark/vrsbench/eval/report_grounding_table.md")
    parser.add_argument("--out-csv", type=str, default="benchmark/vrsbench/eval/report_grounding_table.csv")
    parser.add_argument("--out-report", type=str, default="benchmark/vrsbench/eval/report_grounding_compare.md")
    args = parser.parse_args()

    paper = read_json(_resolve_from_project(args.paper_table))
    paper_rows = paper.get("rows", [])
    if not isinstance(paper_rows, list):
        raise TypeError("Invalid paper table json: rows must be a list")

    our_paths = {
        "Ours (baseline, SigLIP2 visual)": _resolve_from_project(args.baseline),
        "Ours (DINOv3 + merger-only)": _resolve_from_project(args.merger_only),
        "Ours (DINOv3 + merger + LoRA)": _resolve_from_project(args.merger_lora),
    }

    our_rows: list[dict[str, Any]] = []
    for name, p in our_paths.items():
        if not p.is_file():
            raise FileNotFoundError(f"Missing summary: {p}")
        our_rows.append(_our_row(name, p))

    combined: list[dict[str, Any]] = []
    combined.extend(our_rows)
    for r in paper_rows:
        combined.append(
            {
                "method": r.get("method", ""),
                "Acc@0.5 (Unique)": r.get("Acc@0.5 (Unique)"),
                "Acc@0.7 (Unique)": r.get("Acc@0.7 (Unique)"),
                "Acc@0.5 (Non Unique)": r.get("Acc@0.5 (Non Unique)"),
                "Acc@0.7 (Non Unique)": r.get("Acc@0.7 (Non Unique)"),
                "Acc@0.5 (All)": r.get("Acc@0.5 (All)"),
                "Acc@0.7 (All)": r.get("Acc@0.7 (All)"),
                "summary": "",
            }
        )

    cols = [
        "method",
        "Acc@0.5 (Unique)",
        "Acc@0.7 (Unique)",
        "Acc@0.5 (Non Unique)",
        "Acc@0.7 (Non Unique)",
        "Acc@0.5 (All)",
        "Acc@0.7 (All)",
        "summary",
    ]

    # CSV
    out_csv = _resolve_from_project(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in combined:
            w.writerow({k: r.get(k, "") for k in cols})

    # Markdown table
    out_md = _resolve_from_project(args.out_md)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    header = "| " + " | ".join(cols) + " |\n"
    sep = "| " + " | ".join(["---"] * len(cols)) + " |\n"
    lines = [header, sep]
    for r in combined:
        lines.append("| " + " | ".join(_fmt(r.get(k, "")) for k in cols) + " |\n")
    out_md.write_text("".join(lines), encoding="utf-8")

    # 额外写一份“可读性更强”的对比报告（包含少量解释 + 我们三组差值）
    report_path = _resolve_from_project(args.out_report)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    def _delta(a: float | None, b: float | None) -> str:
        if a is None or b is None:
            return ""
        return f"{(b - a):+.2f}"

    base = read_json(our_paths["Ours (baseline, SigLIP2 visual)"])
    mo = read_json(our_paths["Ours (DINOv3 + merger-only)"])
    ml = read_json(our_paths["Ours (DINOv3 + merger + LoRA)"])

    keys = [
        ("Acc@0.5 (Unique)", "unique", "Acc@0.5"),
        ("Acc@0.7 (Unique)", "unique", "Acc@0.7"),
        ("Acc@0.5 (Non Unique)", "non_unique", "Acc@0.5"),
        ("Acc@0.7 (Non Unique)", "non_unique", "Acc@0.7"),
        ("Acc@0.5 (All)", "all", "Acc@0.5"),
        ("Acc@0.7 (All)", "all", "Acc@0.7"),
    ]

    lines2: list[str] = []
    lines2.append("# VRSBench Visual Grounding（Referring）对比报告\n\n")
    lines2.append("本报告对齐 VRSBench 官方评测脚本的口径：\n")
    lines2.append("- 预测与标注都用 `{<x1><y1><x2><y2>}` 四个整数（0~100）表示框\n")
    lines2.append("- IoU 计算采用官方 `computeIoU`（面积计算含 `+1`）\n")
    lines2.append("- `Acc@t`：IoU >= t 的样本比例（按 **total** 归一化）\n\n")

    lines2.append("## 我们三组的关键结果（相对 baseline 的差值）\n\n")
    lines2.append("| metric | baseline | merger-only | delta | merger+LoRA | delta |\n")
    lines2.append("| --- | ---: | ---: | ---: | ---: | ---: |\n")
    for title, split, k in keys:
        b = _get_acc(base, split=split, key=k)
        mo_v = _get_acc(mo, split=split, key=k)
        ml_v = _get_acc(ml, split=split, key=k)
        lines2.append(
            "| "
            + title
            + " | "
            + _fmt(b)
            + " | "
            + _fmt(mo_v)
            + " | "
            + _delta(b, mo_v)
            + " | "
            + _fmt(ml_v)
            + " | "
            + _delta(b, ml_v)
            + " |\n"
        )
    lines2.append("\n")

    lines2.append("## 产物路径\n\n")
    lines2.append(f"- 表格（md）: `{_rel_to_project(out_md)}`\n")
    lines2.append(f"- 表格（csv）: `{_rel_to_project(out_csv)}`\n")
    lines2.append(f"- 本报告: `{_rel_to_project(report_path)}`\n\n")

    report_path.write_text("".join(lines2), encoding="utf-8")

    print(f"[OK] Wrote: {out_md}")
    print(f"[OK] Wrote: {out_csv}")
    print(f"[OK] Wrote: {report_path}")


if __name__ == "__main__":
    main()
