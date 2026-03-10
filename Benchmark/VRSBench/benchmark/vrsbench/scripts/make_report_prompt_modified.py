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


def _rel_to_project(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(_project_root()))
    except Exception:
        return str(path.resolve())


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_first_jsonl_row(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            return row if isinstance(row, dict) else {}
    return {}


def _load_grounding_config_hint(model_dir: Path, summary: dict[str, Any]) -> dict[str, Any]:
    for cand in sorted(model_dir.glob('grounding*.jsonl')):
        row = read_first_jsonl_row(cand)
        if row:
            return row
    cfg = summary.get('config_hint')
    return cfg if isinstance(cfg, dict) else {}


def _validate_prompt_modified_method(method: str, model_dir: Path, summary: dict[str, Any]) -> None:
    cfg = _load_grounding_config_hint(model_dir, summary)
    if not cfg:
        raise ValueError(f'Missing runtime config hint for {method}: {model_dir}')

    image_size = cfg.get('image_size', None)
    min_pixels = cfg.get('smart_resize_min_pixels', None)
    max_pixels = cfg.get('smart_resize_max_pixels', None)
    if image_size is None:
        return

    fixed_pixels = int(image_size) * int(image_size)
    if 'fixed256' in method:
        if min_pixels is None or max_pixels is None:
            raise ValueError(
                f'{method} is marked fixed256, but smart resize range is missing in runtime metadata: {model_dir}'
            )
        if int(min_pixels) != fixed_pixels or int(max_pixels) != fixed_pixels:
            raise ValueError(
                f'{method} is marked fixed256, but runtime range is ({min_pixels}, {max_pixels}) instead of '
                f'({fixed_pixels}, {fixed_pixels}): {model_dir}'
            )
    if 'smartresize' in method:
        if min_pixels is None or max_pixels is None:
            raise ValueError(
                f'{method} is marked smartresize, but smart resize range is missing in runtime metadata: {model_dir}'
            )
        if int(min_pixels) == fixed_pixels and int(max_pixels) == fixed_pixels:
            raise ValueError(
                f'{method} is marked smartresize, but runtime range is fixed ({fixed_pixels}, {fixed_pixels}): {model_dir}'
            )


def _fmt(v: Any) -> str:
    if v is None or v == "":
        return "-"
    if isinstance(v, (int, float)):
        return f"{float(v):.2f}"
    return str(v)


def _caption_row(method: str, summary: dict[str, Any]) -> dict[str, Any]:
    mx = summary.get("metrics_x100", {}) or {}
    return {
        "Group": "Ours",
        "Method": method,
        "BLEU-1": mx.get("BLEU-1"),
        "BLEU-2": mx.get("BLEU-2"),
        "BLEU-3": mx.get("BLEU-3"),
        "BLEU-4": mx.get("BLEU-4"),
        "METEOR": mx.get("METEOR"),
        "ROUGE_L": mx.get("ROUGE_L"),
        "CIDEr": mx.get("CIDEr"),
        "CHAIR": "-",
        "Avg_L": summary.get("avg_len_words"),
    }


def _ground_row(method: str, summary: dict[str, Any]) -> dict[str, Any]:
    splits = summary.get("splits", {}) or {}
    return {
        "Group": "Ours",
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
    parser = argparse.ArgumentParser(description="Build the prompt_modified VRSBench compare report.")
    parser.add_argument("--paper-caption", type=str, default="benchmark/vrsbench/paper/table3_caption_paper.json")
    parser.add_argument("--paper-grounding", type=str, default="benchmark/vrsbench/paper/table4_grounding_paper.json")
    parser.add_argument("--baseline-dir", type=str, default="benchmark/vrsbench/eval/prompt_modified/01_baseline_qwen3vl8b")
    parser.add_argument("--merger-only-dir", type=str, default="benchmark/vrsbench/eval/prompt_modified/02_merger_only_epoch10_fixed256")
    parser.add_argument("--merger-lora-dir", type=str, default="benchmark/vrsbench/eval/prompt_modified/03_merger_lora_epoch10_fixed256")
    parser.add_argument("--smartbucket-dir", type=str, default="benchmark/vrsbench/eval/prompt_modified/04_merger_lora_epoch10_smartresize512_sampleavg")
    parser.add_argument("--qwen-native-dir", type=str, default="benchmark/vrsbench/eval/prompt_modified/05_qwen_native_epoch10")
    parser.add_argument("--out", type=str, default="benchmark/vrsbench/eval/prompt_modified/compare_bench_vs_ours.md")
    args = parser.parse_args()

    paper_caption = read_json(_resolve_from_project(args.paper_caption)).get("rows", [])
    paper_grounding = read_json(_resolve_from_project(args.paper_grounding)).get("rows", [])

    model_dirs = [
        ("Ours-baseline8b", _resolve_from_project(args.baseline_dir)),
        ("Ours-merger_only-epoch10-fixed256", _resolve_from_project(args.merger_only_dir)),
        ("Ours-merger_lora-epoch10-fixed256", _resolve_from_project(args.merger_lora_dir)),
        ("Ours-merger_lora-epoch10-smartresize512-sampleavg", _resolve_from_project(args.smartbucket_dir)),
        ("Ours-qwen_native-epoch10", _resolve_from_project(args.qwen_native_dir)),
    ]

    caption_rows: list[dict[str, Any]] = []
    grounding_rows: list[dict[str, Any]] = []
    for method, model_dir in model_dirs:
        cap_summary_path = model_dir / "caption_summary.json"
        grd_summary_path = model_dir / "grounding_summary.json"
        if not cap_summary_path.is_file():
            raise FileNotFoundError(f"Missing caption summary: {cap_summary_path}")
        if not grd_summary_path.is_file():
            raise FileNotFoundError(f"Missing grounding summary: {grd_summary_path}")
        cap_summary = read_json(cap_summary_path)
        grd_summary = read_json(grd_summary_path)
        _validate_prompt_modified_method(method, model_dir, grd_summary)
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
    grounding_cols = [
        "Group",
        "Method",
        "Acc@0.5 (Unique)",
        "Acc@0.7 (Unique)",
        "Acc@0.5 (Non Unique)",
        "Acc@0.7 (Non Unique)",
        "Acc@0.5 (All)",
        "Acc@0.7 (All)",
    ]

    out_path = _resolve_from_project(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    lines.append("# VRSBench 指标对比（Bench vs Ours）\n")
    lines.append("- 运行标签：`prompt_modified`。\n")
    lines.append("- Caption 的 BLEU/METEOR/ROUGE_L/CIDEr 按 x100 展示。\n")
    lines.append("- Grounding 的 Acc 指标直接来自各自 summary（单位：百分比点）。\n")
    lines.append("- 当前 grounding 生成脚本按各自输出协议做解析与规范化。\n")
    lines.append(
        f"- Bench 数据来源：`{_rel_to_project(_resolve_from_project(args.paper_caption))}` 与 `{_rel_to_project(_resolve_from_project(args.paper_grounding))}`。\n\n"
    )
    lines.append("## Caption 对比\n")
    lines.append(_md_table(caption_all, caption_cols))
    lines.append("\n## Grounding 对比\n")
    lines.append(_md_table(grounding_all, grounding_cols))
    out_path.write_text("".join(lines), encoding="utf-8")
    print(f"[OK] Wrote: {out_path}")


if __name__ == "__main__":
    main()
