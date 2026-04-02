from __future__ import annotations

import argparse
import io
import json
import re
import tarfile
from pathlib import Path
from typing import Any

import requests


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

METHOD_MAP = {
    "LLaVA-1.5": "LLaVA-1.5",
    "MiniGPTv2": "MiniGPTv2",
    "InstructBLIP": "InstructBLIP",
    "mPLUG-OWL2": "mPLUG-OWL2",
    "QWen-VL-Chat": "QWen-VL-Chat",
    "InternLM-XComposer": "InternLM-XComposer",
    "GPT-4-Turbo": "GPT-4-Turbo",
    "Claude-3-Opus": "Claude-3-Opus",
    "\\MODELNAME": "LHRS-Bot",
}


def _project_root() -> Path:
    return Path(__file__).resolve().parents[5]


def _resolve_from_project(path: str | Path) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    return (_project_root() / p).resolve()


def _clean_tex_num_line(line: str) -> list[float]:
    text = str(line)
    if "&" in text:
        text = text.split("&", 1)[1]
    text = re.sub(r"\\textbf\{([^}]*)\}", r"\1", text)
    text = re.sub(r"\\underline\{([^}]*)\}", r"\1", text)
    text = text.replace("{", "").replace("}", "")
    return [float(x) for x in re.findall(r"\d+\.\d+", text)]


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract LHRS-Bench Table 5 metrics from arXiv source.")
    parser.add_argument("--arxiv-id", type=str, default="2402.02544")
    parser.add_argument("--output", type=str, default="benchmark/lhrsbench/paper/table5_lhrs_paper.json")
    args = parser.parse_args()

    url = f"https://arxiv.org/e-print/{args.arxiv_id}"
    response = requests.get(url, timeout=60)
    response.raise_for_status()

    with tarfile.open(fileobj=io.BytesIO(response.content), mode="r:gz") as tar:
        member = tar.getmember("main.tex")
        with tar.extractfile(member) as f:
            if f is None:
                raise RuntimeError("Cannot read main.tex from arXiv source")
            tex = f.read().decode("utf-8", errors="replace")

    start_match = re.search(r"\\label\{tab:bench_result\}", tex)
    if not start_match:
        raise RuntimeError("Cannot find tab:bench_result in main.tex")
    tail = tex[start_match.start() :]
    end_match = re.search(r"\\end\{table\}", tail)
    block = tail[: end_match.start()] if end_match else tail

    methods: list[dict[str, Any]] = []
    for raw_name, method_name in METHOD_MAP.items():
        line = ""
        for candidate in block.splitlines():
            if raw_name in candidate:
                line = candidate
                break
        if not line:
            raise RuntimeError(f"Cannot find row for {raw_name}")

        values = _clean_tex_num_line(line)
        if len(values) < len(COLUMNS):
            raise RuntimeError(f"Parsed values too few for {raw_name}: {values}")
        values = values[: len(COLUMNS)]

        row = {"Method": method_name}
        for column, value in zip(COLUMNS, values):
            row[column] = float(value)
        methods.append(row)

    out_obj = {
        "source": f"arXiv {args.arxiv_id} Table 5",
        "columns": ["Method", *COLUMNS],
        "methods": methods,
    }

    out_path = _resolve_from_project(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out_obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"[OK] extracted {len(methods)} methods")
    print(f"[OK] wrote: {out_path}")


if __name__ == "__main__":
    main()
