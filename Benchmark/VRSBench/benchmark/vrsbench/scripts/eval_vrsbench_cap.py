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


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            rows.append(json.loads(ln))
    return rows


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


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
    # PTBTokenizer 会调用 java（已在环境中配置）。
    from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer  # type: ignore

    import contextlib
    import io

    with contextlib.redirect_stdout(io.StringIO()):
        tok = PTBTokenizer()
        gts_tok = tok.tokenize(gts_raw)
        res_tok = tok.tokenize(res_raw)
    return gts_tok, res_tok


def compute_caption_metrics(refs: dict[str, list[str]], preds: dict[str, str]) -> dict[str, float]:
    # 统一走 COCO caption 常用的 tokenization + scorers。
    from pycocoevalcap.bleu.bleu import Bleu  # type: ignore
    from pycocoevalcap.cider.cider import Cider  # type: ignore
    from pycocoevalcap.meteor.meteor import Meteor  # type: ignore
    from pycocoevalcap.rouge.rouge import Rouge  # type: ignore

    gts_raw, res_raw = _to_coco_dict_raw(refs, preds)
    gts, res = _tokenize(gts_raw, res_raw)

    import contextlib
    import io

    # 部分 scorer 会输出日志，这里统一静音。
    with contextlib.redirect_stdout(io.StringIO()):
        bleu_scores, _ = Bleu(4).compute_score(gts, res)
        rouge_score, _ = Rouge().compute_score(gts, res)
        cider_score, _ = Cider().compute_score(gts, res)
        meteor = Meteor()
        meteor_score, _ = meteor.compute_score(gts, res)
        # pycocoevalcap 的 Meteor 会在对象析构时关闭子进程，这里不额外调用 close。

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
    # 论文里 Avg_L 是“单词数”，这里按空格切分统计。
    total = 0
    for imgid in imgids:
        total += len(str(preds[imgid]).strip().split())
    return float(total) / float(max(1, len(imgids)))


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate VRSBench caption predictions (paper-style metrics).")
    parser.add_argument("--refs", type=str, default="benchmark/vrsbench/data/vrsbench_refs_test.json")
    parser.add_argument("--preds", type=str, required=True, help="Predictions jsonl path.")
    parser.add_argument("--output", type=str, default="", help="Output summary json path.")
    parser.add_argument("--max-images", type=int, default=0)
    parser.add_argument("--bertscore", action="store_true")
    parser.add_argument("--bertscore-model", type=str, default="roberta-large")
    parser.add_argument("--bertscore-agg", type=str, default="max", choices=["max", "mean"])
    parser.add_argument("--bertscore-idf", action="store_true")
    parser.add_argument("--bertscore-batch-size", type=int, default=32)
    parser.add_argument("--bertscore-device", type=str, default="")
    args = parser.parse_args()

    # 为了复用项目里已有的 BERTScore 逻辑，这里加入 src 到 sys.path。
    import sys

    sys.path.insert(0, str(_project_root() / "src"))

    # pycocoevalcap 的 PTBTokenizer / METEOR 都会调用 `java`。
    # 如果用户没有显式激活 conda，PATH 里可能找不到 java，这里补一刀。
    import os
    import shutil

    if shutil.which("java") is None:
        py_bin = str(Path(sys.executable).resolve().parent)
        os.environ["PATH"] = py_bin + os.pathsep + os.environ.get("PATH", "")

    refs_path = _resolve_from_project(args.refs)
    preds_path = _resolve_from_project(args.preds)
    if not refs_path.is_file():
        raise FileNotFoundError(f"Missing refs: {refs_path}")
    if not preds_path.is_file():
        raise FileNotFoundError(f"Missing preds: {preds_path}")

    refs: dict[str, list[str]] = read_json(refs_path)
    preds_rows = read_jsonl(preds_path)
    preds: dict[str, str] = {}
    for row in preds_rows:
        imgid = str(row.get("imgid"))
        pred = str(row.get("prediction", "")).strip()
        if imgid:
            preds[imgid] = pred

    # 按 imgid 排序后截断，保证可复现。
    all_ids = sorted(refs.keys(), key=lambda x: int(x))
    if args.max_images and int(args.max_images) > 0:
        all_ids = all_ids[: int(args.max_images)]
        refs = {k: refs[k] for k in all_ids}
        preds = {k: preds[k] for k in all_ids if k in preds}

    metrics = compute_caption_metrics(refs, preds)
    avg_len = compute_avg_len_words(preds, list(refs.keys()))

    # 论文里这些指标通常按 x100 报告，方便直接对齐表格。
    metrics_x100 = {k: float(v) * 100.0 for k, v in metrics.items()}

    out_path = _resolve_from_project(args.output) if args.output else preds_path.with_name(preds_path.stem + "_summary.json")
    summary: dict[str, Any] = {
        "num_images": len(refs),
        "metrics": metrics,
        "metrics_x100": metrics_x100,
        "avg_len_words": float(avg_len),
        "refs": _rel_to_project(refs_path),
        "preds": _rel_to_project(preds_path),
    }

    if bool(args.bertscore):
        from ftqwen.semantic_metrics import compute_bertscore

        bert_metrics, bert_meta = compute_bertscore(
            refs,
            preds,
            model_type=str(args.bertscore_model),
            agg=str(args.bertscore_agg),  # type: ignore[arg-type]
            idf=bool(args.bertscore_idf),
            batch_size=int(args.bertscore_batch_size),
            device=str(args.bertscore_device).strip() or None,
        )
        summary["bertscore"] = {
            **bert_meta,
            "metrics": bert_metrics.to_dict(),
        }

    # 附带少量样例，便于快速 sanity check。
    sample_ids = sorted(list(refs.keys()), key=lambda x: int(x))[:5]
    summary["samples"] = [
        {
            "imgid": imgid,
            "prediction": preds.get(imgid, ""),
            "refs": refs.get(imgid, [])[:1],
        }
        for imgid in sample_ids
    ]

    write_json(out_path, summary)
    print(f"[OK] Wrote summary: {out_path}")


if __name__ == "__main__":
    main()
