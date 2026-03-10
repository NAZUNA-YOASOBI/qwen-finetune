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


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _fmt(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, (int, float)):
        return f"{v:.3f}"
    return str(v)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a unified VRSBench caption report table.")
    parser.add_argument(
        "--paper-table",
        type=str,
        default="benchmark/vrsbench/paper/table3_caption_paper.json",
        help="Extracted paper Table 3 (json).",
    )
    parser.add_argument(
        "--baseline",
        type=str,
        default="benchmark/vrsbench/outputs/baseline_predictions_test_summary.json",
        help="Baseline summary json.",
    )
    parser.add_argument(
        "--merger-only",
        type=str,
        default="benchmark/vrsbench/outputs/dinov3_merger_only_predictions_test_summary.json",
        help="Merger-only summary json.",
    )
    parser.add_argument(
        "--merger-lora",
        type=str,
        default="benchmark/vrsbench/outputs/dinov3_merger_lora_predictions_test_summary.json",
        help="Merger+LoRA summary json.",
    )
    parser.add_argument("--out-md", type=str, default="benchmark/vrsbench/eval/report_table.md")
    parser.add_argument("--out-csv", type=str, default="benchmark/vrsbench/eval/report_table.csv")
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
        s = read_json(p)
        mx = s.get("metrics_x100", {}) or {}
        bs_f1 = None
        if isinstance(s.get("bertscore"), dict):
            # eval_vrsbench_cap.py 保存的 key 是大写 "F1"（同时兼容小写写法）
            m = s["bertscore"].get("metrics") or {}
            bs_f1 = m.get("F1", m.get("f1"))
        our_rows.append(
            {
                "method": name,
                "BLEU-1": mx.get("BLEU-1"),
                "BLEU-2": mx.get("BLEU-2"),
                "BLEU-3": mx.get("BLEU-3"),
                "BLEU-4": mx.get("BLEU-4"),
                "METEOR": mx.get("METEOR"),
                "ROUGE_L": mx.get("ROUGE_L"),
                "CIDEr": mx.get("CIDEr"),
                "CHAIR2": "",
                "Avg_L": s.get("avg_len_words"),
                "BERTScore_F1": bs_f1,
            }
        )

    combined: list[dict[str, Any]] = []
    combined.extend(our_rows)
    for r in paper_rows:
        combined.append(
            {
                "method": r.get("method", ""),
                "BLEU-1": r.get("BLEU-1"),
                "BLEU-2": r.get("BLEU-2"),
                "BLEU-3": r.get("BLEU-3"),
                "BLEU-4": r.get("BLEU-4"),
                "METEOR": r.get("METEOR"),
                "ROUGE_L": r.get("ROUGE_L"),
                "CIDEr": r.get("CIDEr"),
                "CHAIR2": r.get("CHAIR2"),
                "Avg_L": r.get("Avg_L"),
                "BERTScore_F1": "",
            }
        )

    cols = [
        "method",
        "BLEU-1",
        "BLEU-2",
        "BLEU-3",
        "BLEU-4",
        "METEOR",
        "ROUGE_L",
        "CIDEr",
        "CHAIR2",
        "Avg_L",
        "BERTScore_F1",
    ]

    # 写 CSV
    out_csv = _resolve_from_project(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in combined:
            w.writerow({k: r.get(k, "") for k in cols})

    # 写 Markdown
    out_md = _resolve_from_project(args.out_md)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    header = "| " + " | ".join(cols) + " |\n"
    sep = "| " + " | ".join(["---"] * len(cols)) + " |\n"
    lines = [header, sep]
    for r in combined:
        lines.append("| " + " | ".join(_fmt(r.get(k, "")) for k in cols) + " |\n")
    out_md.write_text("".join(lines), encoding="utf-8")

    print(f"[OK] Wrote: {out_md}")
    print(f"[OK] Wrote: {out_csv}")


if __name__ == "__main__":
    main()
