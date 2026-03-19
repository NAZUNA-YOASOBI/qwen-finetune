from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def _project_root() -> Path:
    # benchmark/vrsbench/scripts/ftqwen*/<group>/*.py -> parents[5] == 项目根目录
    return Path(__file__).resolve().parents[5]


def _resolve_from_project(path: str | Path) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    return (_project_root() / p).resolve()


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _parse_table3_from_text(text: str) -> list[dict]:
    # 由于 PDF 抽取文本时可能出现 “[28]13.9” 这种粘连，这里先强行断开。
    text = re.sub(r"\](?=\d)", "] ", text)

    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    # 找到表头行（包含 Method 与 BLEU-1）
    header_idx = None
    for i, ln in enumerate(lines):
        if "Method" in ln and "BLEU-1" in ln and "CIDEr" in ln:
            header_idx = i
            break
    if header_idx is None:
        raise RuntimeError("Failed to locate Table 3 header in PDF text.")

    # 表头后面连续几行是表格内容，遇到明显非表格行就停止。
    rows: list[dict] = []
    for ln in lines[header_idx + 1 :]:
        if ln.startswith("2http") or ln.isdigit():
            break
        # 每行最后 9 个数字是指标：BLEU-1/2/3/4, METEOR, ROUGE_L, CIDEr, CHAIR, Avg_L
        toks = ln.split()
        if len(toks) < 10:
            continue

        # 取末尾 9 个纯数字 token
        tail = toks[-9:]
        if not all(re.fullmatch(r"\d+(?:\.\d+)?", t) for t in tail):
            continue

        nums = [float(x) for x in tail]
        method_tokens = toks[:-9]
        # 去掉引用编号 token，例如 “[28]”
        method_tokens = [t for t in method_tokens if not re.fullmatch(r"\[\d+\]", t)]
        method = " ".join(method_tokens).strip()
        method = re.sub(r"\s+", " ", method)
        # 修正少量 PDF 抽取造成的断词（不影响数值）。
        method = method.replace("LLaV A", "LLaVA")

        rows.append(
            {
                "method": method,
                "BLEU-1": nums[0],
                "BLEU-2": nums[1],
                "BLEU-3": nums[2],
                "BLEU-4": nums[3],
                "METEOR": nums[4],
                "ROUGE_L": nums[5],
                "CIDEr": nums[6],
                "CHAIR2": nums[7],
                "Avg_L": nums[8],
            }
        )

    if not rows:
        raise RuntimeError("Parsed 0 rows for Table 3.")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract Table 3 (caption metrics) from VRSBench paper PDF.")
    parser.add_argument("--pdf", type=str, default="benchmark/vrsbench/paper/vrsbench_2406.12384.pdf")
    parser.add_argument("--output", type=str, default="benchmark/vrsbench/paper/table3_caption_paper.json")
    args = parser.parse_args()

    pdf_path = _resolve_from_project(args.pdf)
    if not pdf_path.is_file():
        raise FileNotFoundError(f"Missing PDF: {pdf_path}")

    try:
        from pypdf import PdfReader  # type: ignore
    except Exception as e:  # noqa: BLE001
        raise RuntimeError("Missing dependency: pypdf") from e

    reader = PdfReader(str(pdf_path))
    all_text = []
    for page in reader.pages:
        try:
            all_text.append(page.extract_text() or "")
        except Exception:
            all_text.append("")
    joined = "\n".join(all_text)

    rows = _parse_table3_from_text(joined)
    out_path = _resolve_from_project(args.output)
    write_json(
        out_path,
        {
            "source": str(Path(args.pdf)),
            "table": "Table 3: Detailed image caption performance on VRSBench dataset",
            "metrics_scale_note": "Numbers are copied as-is from the paper table (typically reported in x100 scale for BLEU/METEOR/ROUGE/CIDEr).",
            "rows": rows,
        },
    )
    print(f"[OK] Wrote: {out_path}")


if __name__ == "__main__":
    main()
