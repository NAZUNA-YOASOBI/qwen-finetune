from __future__ import annotations

import argparse
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


def read_jsonl_first(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            return json.loads(ln)
    raise ValueError(f"Empty jsonl: {path}")


def _fmt(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, (int, float)):
        return f"{v:.3f}"
    return str(v)


def _pick(d: dict[str, Any], keys: list[str]) -> dict[str, Any]:
    return {k: d.get(k) for k in keys if k in d}


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a detailed VRSBench caption comparison report (ours + paper).")
    parser.add_argument(
        "--paper-table",
        type=str,
        default="benchmark/vrsbench/paper/table3_caption_paper.json",
        help="Extracted paper Table 3 (json).",
    )

    parser.add_argument("--baseline-preds", type=str, required=True)
    parser.add_argument("--baseline-summary", type=str, required=True)
    parser.add_argument("--merger-only-preds", type=str, required=True)
    parser.add_argument("--merger-only-summary", type=str, required=True)
    parser.add_argument("--merger-lora-preds", type=str, required=True)
    parser.add_argument("--merger-lora-summary", type=str, required=True)

    parser.add_argument("--out", type=str, default="benchmark/vrsbench/eval/report_caption.md")
    args = parser.parse_args()

    paper = read_json(_resolve_from_project(args.paper_table))
    paper_rows = paper.get("rows", [])
    if not isinstance(paper_rows, list):
        raise TypeError("Invalid paper table json: rows must be a list")

    # 读取我们的运行配置（直接从预测 jsonl 第一行拿到 prompt/采样参数等，避免手填出错）
    baseline_preds_path = _resolve_from_project(args.baseline_preds)
    merger_only_preds_path = _resolve_from_project(args.merger_only_preds)
    merger_lora_preds_path = _resolve_from_project(args.merger_lora_preds)
    baseline_cfg = read_jsonl_first(baseline_preds_path)
    merger_only_cfg = read_jsonl_first(merger_only_preds_path)
    merger_lora_cfg = read_jsonl_first(merger_lora_preds_path)

    # 读取我们的分数（summary）
    baseline_sum_path = _resolve_from_project(args.baseline_summary)
    merger_only_sum_path = _resolve_from_project(args.merger_only_summary)
    merger_lora_sum_path = _resolve_from_project(args.merger_lora_summary)
    baseline_sum = read_json(baseline_sum_path)
    merger_only_sum = read_json(merger_only_sum_path)
    merger_lora_sum = read_json(merger_lora_sum_path)

    def _our_row(name: str, s: dict[str, Any]) -> dict[str, Any]:
        mx = s.get("metrics_x100", {}) or {}
        bs_f1 = None
        if isinstance(s.get("bertscore"), dict):
            m = s["bertscore"].get("metrics") or {}
            bs_f1 = m.get("F1", m.get("f1"))
        return {
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

    ours = [
        _our_row("Ours (baseline, Qwen3-VL visual)", baseline_sum),
        _our_row("Ours (DINOv3 + merger-only)", merger_only_sum),
        _our_row("Ours (DINOv3 + merger + LoRA)", merger_lora_sum),
    ]

    paper_combined: list[dict[str, Any]] = []
    for r in paper_rows:
        paper_combined.append(
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

    def _find_paper(method: str) -> dict[str, Any] | None:
        for r in paper_combined:
            if str(r.get("method")) == method:
                return r
        return None

    ref_geochat_wo = _find_paper("GeoChat w/o ft")
    ref_llava = _find_paper("LLaVA-1.5")

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

    # 拼 Markdown 表格（我们的 3 组在前，然后是论文 Table 3）
    header = "| " + " | ".join(cols) + " |\n"
    sep = "| " + " | ".join(["---"] * len(cols)) + " |\n"
    table_lines = [header, sep]
    for r in ours + paper_combined:
        table_lines.append("| " + " | ".join(_fmt(r.get(k, "")) for k in cols) + " |\n")

    # 关键对比：和 GeoChat w/o ft、LLaVA-1.5 做差值（ours - paper）
    metrics = ["BLEU-1", "BLEU-2", "BLEU-3", "BLEU-4", "METEOR", "ROUGE_L", "CIDEr", "Avg_L"]

    def _delta_block(title: str, ref: dict[str, Any] | None) -> str:
        if not ref:
            return f"## {title}\n\n(Reference row not found in paper table.)\n"
        lines = [f"## {title}\n\n"]
        lines.append(f"Reference: `{ref.get('method')}`\n\n")
        lines.append("| method | " + " | ".join(metrics) + " |\n")
        lines.append("| --- | " + " | ".join(["---"] * len(metrics)) + " |\n")
        for r in ours:
            vals = []
            for m in metrics:
                a = r.get(m)
                b = ref.get(m)
                if isinstance(a, (int, float)) and isinstance(b, (int, float)):
                    vals.append(_fmt(a - b))
                else:
                    vals.append("")
            lines.append("| " + str(r.get("method")) + " | " + " | ".join(vals) + " |\n")
        lines.append("\n")
        return "".join(lines)

    # 把我们实际运行参数写清楚（只摘取最关键字段）
    cfg_keys_common = [
        "prompt",
        "max_new_tokens",
        "do_sample",
        "temperature",
        "top_p",
        "top_k",
        "num_beams",
        "repetition_penalty",
        "seed",
        "batch_size",
        "requested_batch_size",
    ]
    cfg_keys_dino_extra = ["image_size", "qwen_model_dir", "dinov3_dir", "merger_ckpt", "lora_dir"]
    cfg_keys_base_extra = ["model", "model_dir"]

    out_path = _resolve_from_project(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    md: list[str] = []
    md.append("# VRSBench Caption 评测对比报告（我们 vs 论文 Table 3）\n\n")
    md.append(
        "说明：\n"
        "- 论文分数来自 `benchmark/vrsbench/paper/table3_caption_paper.json`（从论文 PDF 的 Table 3 抽取，数字按论文原样抄录）。\n"
        "- 我们分数来自本项目的 `*_summary.json`（使用 COCO caption 常用指标：BLEU/METEOR/ROUGE_L/CIDEr；同时额外算了 BERTScore）。\n"
        "- 论文与我们在 prompt/解码/输出长度等设置上不一定一致，所以这些数值只能做“并列对比参考”，不能当作严格公平的 SOTA 对比。\n\n"
    )

    md.append("## 我们本次评测设置（从预测文件自动读取）\n\n")
    md.append("### baseline（Qwen3-VL 原视觉）\n\n")
    md.append(f"- preds: `{_rel_to_project(baseline_preds_path)}`\n")
    md.append(f"- summary: `{_rel_to_project(baseline_sum_path)}`\n")
    md.append("```json\n")
    md.append(json.dumps(_pick(baseline_cfg, cfg_keys_base_extra + cfg_keys_common), ensure_ascii=False, indent=2))
    md.append("\n```\n\n")

    md.append("### DINOv3 + merger-only\n\n")
    md.append(f"- preds: `{_rel_to_project(merger_only_preds_path)}`\n")
    md.append(f"- summary: `{_rel_to_project(merger_only_sum_path)}`\n")
    md.append("```json\n")
    md.append(json.dumps(_pick(merger_only_cfg, cfg_keys_dino_extra + cfg_keys_common), ensure_ascii=False, indent=2))
    md.append("\n```\n\n")

    md.append("### DINOv3 + merger + LoRA\n\n")
    md.append(f"- preds: `{_rel_to_project(merger_lora_preds_path)}`\n")
    md.append(f"- summary: `{_rel_to_project(merger_lora_sum_path)}`\n")
    md.append("```json\n")
    md.append(json.dumps(_pick(merger_lora_cfg, cfg_keys_dino_extra + cfg_keys_common), ensure_ascii=False, indent=2))
    md.append("\n```\n\n")

    md.append("## 分数总表（我们的 3 组 + 论文 Table 3）\n\n")
    md.append("注：BLEU/METEOR/ROUGE_L/CIDEr 为 x100 形式；BERTScore_F1 为 0~1。\n\n")
    md.append("".join(table_lines))
    md.append("\n")

    md.append("## 关键现象（只基于表格数字的直接观察）\n\n")
    # 避免“写死结论”，这里完全从数据里推导结论，防止结论与新实验不一致。
    ours_by_name = {str(r.get("method")): r for r in ours}

    def _best(rows: list[dict[str, Any]], key: str, *, reverse: bool = True) -> tuple[str, float] | None:
        best_name: str | None = None
        best_val: float | None = None
        for r in rows:
            v = r.get(key)
            if not isinstance(v, (int, float)):
                continue
            if best_val is None:
                best_val = float(v)
                best_name = str(r.get("method"))
                continue
            if reverse and float(v) > float(best_val):
                best_val = float(v)
                best_name = str(r.get("method"))
            if (not reverse) and float(v) < float(best_val):
                best_val = float(v)
                best_name = str(r.get("method"))
        if best_name is None or best_val is None:
            return None
        return best_name, float(best_val)

    def _delta(a: Any, b: Any) -> float | None:
        if isinstance(a, (int, float)) and isinstance(b, (int, float)):
            return float(a) - float(b)
        return None

    # 1) 我们内部：谁更高 / 谁更低
    for m in ["BLEU-1", "METEOR", "ROUGE_L", "CIDEr", "BERTScore_F1"]:
        best = _best(ours, m, reverse=True)
        if best:
            md.append(f"- {m} 最高：{best[0]}（{_fmt(best[1])}）\n")
    best_len = _best(ours, "Avg_L", reverse=False)
    if best_len:
        md.append(f"- Avg_L 最短：{best_len[0]}（{_fmt(best_len[1])}）\n")

    # 2) 相对 baseline 的提升/下降
    base = ours_by_name.get("Ours (baseline, Qwen3-VL visual)")
    if isinstance(base, dict):
        for name in ["Ours (DINOv3 + merger-only)", "Ours (DINOv3 + merger + LoRA)"]:
            cur = ours_by_name.get(name)
            if not isinstance(cur, dict):
                continue
            d_bleu1 = _delta(cur.get("BLEU-1"), base.get("BLEU-1"))
            d_meteor = _delta(cur.get("METEOR"), base.get("METEOR"))
            d_rouge = _delta(cur.get("ROUGE_L"), base.get("ROUGE_L"))
            d_cider = _delta(cur.get("CIDEr"), base.get("CIDEr"))
            d_len = _delta(cur.get("Avg_L"), base.get("Avg_L"))
            d_bs = _delta(cur.get("BERTScore_F1"), base.get("BERTScore_F1"))
            md.append(
                f"- 相对 baseline，{name}："
                f"BLEU-1 {_fmt(d_bleu1)}，METEOR {_fmt(d_meteor)}，ROUGE_L {_fmt(d_rouge)}，CIDEr {_fmt(d_cider)}，"
                f"Avg_L {_fmt(d_len)}，BERTScore_F1 {_fmt(d_bs)}。\n"
            )

    # 3) 长度与论文表的差距（只陈述事实，不做“必然导致”的推断）
    paper_avg_l: list[float] = []
    for r in paper_combined:
        v = r.get("Avg_L")
        if isinstance(v, (int, float)):
            paper_avg_l.append(float(v))
    if paper_avg_l:
        md.append(
            f"- 论文 Table 3 的 Avg_L 范围：{_fmt(min(paper_avg_l))} ~ {_fmt(max(paper_avg_l))}。\n"
        )
        md.append(
            f"- 我们的 Avg_L：baseline={_fmt(base.get('Avg_L') if isinstance(base, dict) else None)}，"
            f"merger-only={_fmt(ours_by_name.get('Ours (DINOv3 + merger-only)', {}).get('Avg_L'))}，"
            f"merger+LoRA={_fmt(ours_by_name.get('Ours (DINOv3 + merger + LoRA)', {}).get('Avg_L'))}。\n"
        )
    md.append("\n")

    md.append(_delta_block("对比差值：相对 GeoChat w/o ft（ours - paper）", ref_geochat_wo))
    md.append(_delta_block("对比差值：相对 LLaVA-1.5（ours - paper）", ref_llava))

    out_path.write_text("".join(md), encoding="utf-8")
    print(f"[OK] Wrote: {out_path}")


if __name__ == "__main__":
    main()
