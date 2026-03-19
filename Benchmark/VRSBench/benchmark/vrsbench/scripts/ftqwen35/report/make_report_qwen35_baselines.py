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


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _fmt(v: Any) -> str:
    if v is None or v == "":
        return "-"
    if isinstance(v, (int, float)):
        return f"{float(v):.2f}"
    return str(v)


def _caption_row(method: str, summary: dict[str, Any]) -> dict[str, Any]:
    metrics = summary.get("metrics", summary)
    return {
        "Group": "Qwen3.5",
        "Method": method,
        "BLEU-1": metrics.get("BLEU-1", 0.0) * 100.0,
        "BLEU-2": metrics.get("BLEU-2", 0.0) * 100.0,
        "BLEU-3": metrics.get("BLEU-3", 0.0) * 100.0,
        "BLEU-4": metrics.get("BLEU-4", 0.0) * 100.0,
        "METEOR": metrics.get("METEOR", 0.0) * 100.0,
        "ROUGE_L": metrics.get("ROUGE_L", 0.0) * 100.0,
        "CIDEr": metrics.get("CIDEr", 0.0) * 100.0,
        "CHAIR": metrics.get("CHAIR2", None),
        "Avg_L": metrics.get("Avg_L", None),
    }


def _ground_row(method: str, summary: dict[str, Any]) -> dict[str, Any]:
    splits = summary.get("splits", {})
    return {
        "Group": "Qwen3.5",
        "Method": method,
        "Acc@0.5 (Unique)": ((splits.get("unique") or {}).get("Acc@0.5")),
        "Acc@0.7 (Unique)": ((splits.get("unique") or {}).get("Acc@0.7")),
        "Acc@0.5 (Non Unique)": ((splits.get("non_unique") or {}).get("Acc@0.5")),
        "Acc@0.7 (Non Unique)": ((splits.get("non_unique") or {}).get("Acc@0.7")),
        "Acc@0.5 (All)": ((splits.get("all") or {}).get("Acc@0.5")),
        "Acc@0.7 (All)": ((splits.get("all") or {}).get("Acc@0.7")),
    }


def _md_table(rows: list[dict[str, Any]], cols: list[str]) -> str:
    lines = []
    lines.append("| " + " | ".join(cols) + " |\n")
    lines.append("| " + " | ".join(["---"] * len(cols)) + " |\n")
    for row in rows:
        lines.append("| " + " | ".join(_fmt(row.get(col, "")) for col in cols) + " |\n")
    return "".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the Qwen3.5 VRSBench compare report.")
    parser.add_argument("--paper-caption", type=str, default="benchmark/vrsbench/paper/table3_caption_paper.json")
    parser.add_argument("--paper-grounding", type=str, default="benchmark/vrsbench/paper/table4_grounding_paper.json")
    parser.add_argument("--qwen35-4b-dir", type=str, default="benchmark/vrsbench/eval/01_qwen35_baselines_20260306/01_qwen35_4b")
    parser.add_argument("--qwen35-9b-dir", type=str, default="benchmark/vrsbench/eval/01_qwen35_baselines_20260306/02_qwen35_9b")
    parser.add_argument("--out", type=str, default="benchmark/vrsbench/eval/01_qwen35_baselines_20260306/compare_bench_vs_qwen35.md")
    args = parser.parse_args()

    paper_caption = read_json(_resolve_from_project(args.paper_caption)).get("rows", [])
    paper_grounding = read_json(_resolve_from_project(args.paper_grounding)).get("rows", [])
    model_dirs = [
        ("Qwen3.5-4B", _resolve_from_project(args.qwen35_4b_dir)),
        ("Qwen3.5-9B", _resolve_from_project(args.qwen35_9b_dir)),
    ]

    caption_rows: list[dict[str, Any]] = []
    grounding_rows: list[dict[str, Any]] = []
    for method, model_dir in model_dirs:
        cap_summary = read_json(model_dir / "caption_summary.json")
        grd_summary = read_json(model_dir / "grounding_summary.json")
        caption_rows.append(_caption_row(method, cap_summary))
        grounding_rows.append(_ground_row(method, grd_summary))

    caption_all: list[dict[str, Any]] = []
    for row in paper_caption:
        caption_all.append(
            {
                "Group": "Bench",
                "Method": row.get("method", ""),
                "BLEU-1": row.get("BLEU-1"),
                "BLEU-2": row.get("BLEU-2"),
                "BLEU-3": row.get("BLEU-3"),
                "BLEU-4": row.get("BLEU-4"),
                "METEOR": row.get("METEOR"),
                "ROUGE_L": row.get("ROUGE_L"),
                "CIDEr": row.get("CIDEr"),
                "CHAIR": row.get("CHAIR2"),
                "Avg_L": row.get("Avg_L"),
            }
        )
    caption_all.extend(caption_rows)

    grounding_all: list[dict[str, Any]] = []
    for row in paper_grounding:
        grounding_all.append(
            {
                "Group": "Bench",
                "Method": row.get("method", ""),
                "Acc@0.5 (Unique)": row.get("Acc@0.5 (Unique)"),
                "Acc@0.7 (Unique)": row.get("Acc@0.7 (Unique)"),
                "Acc@0.5 (Non Unique)": row.get("Acc@0.5 (Non Unique)"),
                "Acc@0.7 (Non Unique)": row.get("Acc@0.7 (Non Unique)"),
                "Acc@0.5 (All)": row.get("Acc@0.5 (All)"),
                "Acc@0.7 (All)": row.get("Acc@0.7 (All)"),
            }
        )
    grounding_all.extend(grounding_rows)

    caption_cols = ["Group", "Method", "BLEU-1", "BLEU-2", "BLEU-3", "BLEU-4", "METEOR", "ROUGE_L", "CIDEr", "CHAIR", "Avg_L"]
    grounding_cols = ["Group", "Method", "Acc@0.5 (Unique)", "Acc@0.7 (Unique)", "Acc@0.5 (Non Unique)", "Acc@0.7 (Non Unique)", "Acc@0.5 (All)", "Acc@0.7 (All)"]

    out_path = _resolve_from_project(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    lines.append("# VRSBench 指标对比（Bench vs Qwen3.5）\n")
    lines.append("- Caption 的 BLEU/METEOR/ROUGE_L/CIDEr 按 x100 展示。\n")
    lines.append("- Grounding 的 Acc 指标直接来自官方口径 summary（单位：百分比点）。\n")
    lines.append("- 解码按 Qwen3.5 非思考模式：`temperature=1.0, top_p=0.95, top_k=20, do_sample=True`。\n")
    lines.append("- Grounding 使用严格坐标 prompt。\n\n")
    lines.append("## Caption 对比\n")
    lines.append(_md_table(caption_all, caption_cols))
    lines.append("\n## Grounding 对比\n")
    lines.append(_md_table(grounding_all, grounding_cols))
    out_path.write_text("".join(lines), encoding="utf-8")
    print(f"[OK] Wrote: {out_path}")


if __name__ == "__main__":
    main()
