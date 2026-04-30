from __future__ import annotations

import argparse
import contextlib
import io
import json
from pathlib import Path
from typing import Any


SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[5]
VRSBENCH_ROOT = SCRIPT_PATH.parents[4]
BENCHMARK_ROOT = VRSBENCH_ROOT / "benchmark" / "single_task"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.is_file():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def resolve_from_vrsbench(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    if candidate.parts and candidate.parts[0] == VRSBENCH_ROOT.name:
        return (PROJECT_ROOT / candidate).resolve()
    return (VRSBENCH_ROOT / candidate).resolve()


def default_summary_path(output_dir: Path) -> Path:
    if output_dir.parent.name == "outputs" and output_dir.parent.parent == BENCHMARK_ROOT:
        return BENCHMARK_ROOT / "eval" / output_dir.name / "evaluation_summary.json"
    return output_dir / "evaluation_summary.json"


def to_coco_dict_raw(refs: dict[str, list[str]], preds: dict[str, str]) -> tuple[dict[str, Any], dict[str, Any]]:
    gts: dict[str, Any] = {}
    res: dict[str, Any] = {}
    for image_id, ref_list in refs.items():
        if image_id not in preds:
            continue
        gts[image_id] = [{"caption": str(ref)} for ref in ref_list]
        res[image_id] = [{"caption": str(preds[image_id])}]
    return gts, res


def tokenize(gts_raw: dict[str, Any], res_raw: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer  # type: ignore

    with contextlib.redirect_stdout(io.StringIO()):
        tokenizer = PTBTokenizer()
        gts = tokenizer.tokenize(gts_raw)
        res = tokenizer.tokenize(res_raw)
    return gts, res


def compute_caption_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    from pycocoevalcap.bleu.bleu import Bleu  # type: ignore
    from pycocoevalcap.cider.cider import Cider  # type: ignore
    from pycocoevalcap.meteor.meteor import Meteor  # type: ignore
    from pycocoevalcap.rouge.rouge import Rouge  # type: ignore

    refs = {str(row["sample_id"]): list(row.get("refs", [])) for row in rows}
    preds = {str(row["sample_id"]): str(row.get("prediction", "")) for row in rows}
    gts_raw, res_raw = to_coco_dict_raw(refs, preds)
    gts, res = tokenize(gts_raw, res_raw)

    with contextlib.redirect_stdout(io.StringIO()):
        bleu_scores, _ = Bleu(4).compute_score(gts, res)
        rouge_score, _ = Rouge().compute_score(gts, res)
        cider_score, _ = Cider().compute_score(gts, res)
        meteor = Meteor()
        meteor_score, _ = meteor.compute_score(gts, res)

    avg_len_words = sum(len(str(row.get("prediction", "")).strip().split()) for row in rows) / max(1, len(rows))
    metrics = {
        "BLEU-1": float(bleu_scores[0]),
        "BLEU-2": float(bleu_scores[1]),
        "BLEU-3": float(bleu_scores[2]),
        "BLEU-4": float(bleu_scores[3]),
        "METEOR": float(meteor_score),
        "ROUGE_L": float(rouge_score),
        "CIDEr": float(cider_score),
    }
    return {
        "num_rows": len(rows),
        "metrics": metrics,
        "metrics_x100": {key: float(value) * 100.0 for key, value in metrics.items()},
        "avg_len_words": float(avg_len_words),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate single-task benchmark predictions.")
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--summary-out", type=str, default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = resolve_from_vrsbench(args.output_dir)

    summary_path = Path(args.summary_out) if str(args.summary_out).strip() else default_summary_path(output_dir)
    if not summary_path.is_absolute():
        summary_path = resolve_from_vrsbench(summary_path)

    caption_files = ("ucm_captions.jsonl", "sydney_captions.jsonl", "rsicd.jsonl")

    result: dict[str, Any] = {
        "output_dir": str(output_dir),
        "caption": {},
    }

    for filename in caption_files:
        rows = read_jsonl(output_dir / filename)
        if rows:
            result["caption"][Path(filename).stem] = compute_caption_metrics(rows)

    generation_summary = output_dir / "generation_summary.json"
    if generation_summary.is_file():
        result["generation_summary"] = json.loads(generation_summary.read_text(encoding="utf-8"))

    write_json(summary_path, result)
    print(f"[OK] Wrote evaluation summary: {summary_path}", flush=True)


if __name__ == "__main__":
    main()
