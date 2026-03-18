from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any


CONFIG = {
    "eval_json": "VRSBench_EVAL_Cap.json",
    "preds": "caption_eval_predictions.jsonl",
    "output": "caption_eval_metrics.json",
}


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def build_refs_from_eval_json(path: Path) -> dict[str, list[str]]:
    data = read_json(path)
    if not isinstance(data, list):
        raise ValueError(f"eval_json must be a list: {path}")
    refs: dict[str, list[str]] = {}
    for item in data:
        question_id = item.get("question_id", None)
        ground_truth = str(item.get("ground_truth", "")).strip()
        if question_id is None or not ground_truth:
            continue
        refs[str(question_id)] = [ground_truth]
    return refs


def _to_coco_dict_raw(refs: dict[str, list[str]], preds: dict[str, str]) -> tuple[dict[str, Any], dict[str, Any]]:
    gts: dict[str, Any] = {}
    res: dict[str, Any] = {}
    for img_id, ref_list in refs.items():
        if img_id not in preds:
            raise KeyError(f"Missing prediction for imgid={img_id}")
        gts[img_id] = [{"caption": str(r)} for r in ref_list]
        res[img_id] = [{"caption": str(preds[img_id])}]
    return gts, res


def _tokenize(gts_raw: dict[str, Any], res_raw: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer  # type: ignore

    with contextlib.redirect_stdout(io.StringIO()):
        tokenizer = PTBTokenizer()
        gts = tokenizer.tokenize(gts_raw)
        res = tokenizer.tokenize(res_raw)
    return gts, res


def compute_caption_metrics(refs: dict[str, list[str]], preds: dict[str, str]) -> dict[str, float]:
    from pycocoevalcap.bleu.bleu import Bleu  # type: ignore
    from pycocoevalcap.cider.cider import Cider  # type: ignore
    from pycocoevalcap.meteor.meteor import Meteor  # type: ignore
    from pycocoevalcap.rouge.rouge import Rouge  # type: ignore

    gts_raw, res_raw = _to_coco_dict_raw(refs, preds)
    gts, res = _tokenize(gts_raw, res_raw)

    with contextlib.redirect_stdout(io.StringIO()):
        bleu_scores, _ = Bleu(4).compute_score(gts, res)
        rouge_score, _ = Rouge().compute_score(gts, res)
        cider_score, _ = Cider().compute_score(gts, res)
        meteor = Meteor()
        meteor_score, _ = meteor.compute_score(gts, res)

    return {
        "BLEU-1": float(bleu_scores[0]),
        "BLEU-2": float(bleu_scores[1]),
        "BLEU-3": float(bleu_scores[2]),
        "BLEU-4": float(bleu_scores[3]),
        "METEOR": float(meteor_score),
        "ROUGE_L": float(rouge_score),
        "CIDEr": float(cider_score),
    }


def compute_avg_len_words(preds: dict[str, str], imgids: list[str]) -> float:
    total = 0
    for imgid in imgids:
        total += len(str(preds[imgid]).strip().split())
    return float(total) / float(max(1, len(imgids)))


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate VRSBench caption predictions.")
    parser.add_argument("--eval-json", type=str, default=CONFIG["eval_json"])
    parser.add_argument("--preds", type=str, default=CONFIG["preds"])
    parser.add_argument("--output", type=str, default=CONFIG["output"])
    parser.add_argument("--max-images", type=int, default=0)
    args = parser.parse_args()

    if shutil.which("java") is None:
        py_bin = str(Path(sys.executable).resolve().parent)
        os.environ["PATH"] = py_bin + os.pathsep + os.environ.get("PATH", "")

    eval_json_path = Path(args.eval_json).resolve()
    preds_path = Path(args.preds).resolve()
    output_path = Path(args.output).resolve()
    if not eval_json_path.is_file():
        raise FileNotFoundError(f"Missing eval_json: {eval_json_path}")
    if not preds_path.is_file():
        raise FileNotFoundError(f"Missing preds: {preds_path}")

    refs = build_refs_from_eval_json(eval_json_path)
    pred_rows = read_jsonl(preds_path)
    preds: dict[str, str] = {}
    for row in pred_rows:
        imgid = str(row.get("imgid", "")).strip()
        pred = str(row.get("prediction", "")).strip()
        if imgid and pred:
            preds[imgid] = pred

    all_ids = sorted(refs.keys(), key=lambda x: int(x))
    if args.max_images > 0:
        all_ids = all_ids[: args.max_images]
        refs = {k: refs[k] for k in all_ids}
        preds = {k: preds[k] for k in all_ids if k in preds}

    metrics = compute_caption_metrics(refs, preds)
    avg_len = compute_avg_len_words(preds, list(refs.keys()))
    metrics_x100 = {k: float(v) * 100.0 for k, v in metrics.items()}
    summary = {
        "BLEU-1": metrics["BLEU-1"],
        "BLEU-2": metrics["BLEU-2"],
        "BLEU-3": metrics["BLEU-3"],
        "BLEU-4": metrics["BLEU-4"],
        "METEOR": metrics["METEOR"],
        "ROUGE_L": metrics["ROUGE_L"],
        "CIDEr": metrics["CIDEr"],
        "Avg_L": float(avg_len),
        "num_images": len(refs),
        "metrics_x100": metrics_x100,
        "eval_json": str(eval_json_path),
        "preds": str(preds_path),
    }
    write_json(output_path, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
