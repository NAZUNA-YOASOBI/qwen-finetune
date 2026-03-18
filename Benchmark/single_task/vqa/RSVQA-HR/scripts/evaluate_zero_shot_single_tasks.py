from __future__ import annotations

import argparse
import contextlib
import io
import json
from pathlib import Path
from typing import Any


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    with path.open('r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def _to_coco_dict_raw(refs: dict[str, list[str]], preds: dict[str, str]) -> tuple[dict[str, Any], dict[str, Any]]:
    gts: dict[str, Any] = {}
    res: dict[str, Any] = {}
    for image_id, ref_list in refs.items():
        if image_id not in preds:
            continue
        gts[image_id] = [{'caption': str(ref)} for ref in ref_list]
        res[image_id] = [{'caption': str(preds[image_id])}]
    return gts, res


def _tokenize(gts_raw: dict[str, Any], res_raw: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer  # type: ignore

    with contextlib.redirect_stdout(io.StringIO()):
        tokenizer = PTBTokenizer()
        gts_tok = tokenizer.tokenize(gts_raw)
        res_tok = tokenizer.tokenize(res_raw)
    return gts_tok, res_tok


def compute_caption_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    from pycocoevalcap.bleu.bleu import Bleu  # type: ignore
    from pycocoevalcap.cider.cider import Cider  # type: ignore
    from pycocoevalcap.meteor.meteor import Meteor  # type: ignore
    from pycocoevalcap.rouge.rouge import Rouge  # type: ignore

    refs = {str(row['sample_id']): list(row.get('refs', [])) for row in rows}
    preds = {str(row['sample_id']): str(row.get('prediction', '')) for row in rows}
    gts_raw, res_raw = _to_coco_dict_raw(refs, preds)
    gts, res = _tokenize(gts_raw, res_raw)

    with contextlib.redirect_stdout(io.StringIO()):
        bleu_scores, _ = Bleu(4).compute_score(gts, res)
        rouge_score, _ = Rouge().compute_score(gts, res)
        cider_score, _ = Cider().compute_score(gts, res)
        meteor = Meteor()
        meteor_score, _ = meteor.compute_score(gts, res)

    avg_len_words = sum(len(str(row.get('prediction', '')).strip().split()) for row in rows) / max(1, len(rows))
    metrics = {
        'BLEU-1': float(bleu_scores[0]),
        'BLEU-2': float(bleu_scores[1]),
        'BLEU-3': float(bleu_scores[2]),
        'BLEU-4': float(bleu_scores[3]),
        'METEOR': float(meteor_score),
        'ROUGE_L': float(rouge_score),
        'CIDEr': float(cider_score),
    }
    return {
        'num_rows': len(rows),
        'metrics': metrics,
        'metrics_x100': {k: float(v) * 100.0 for k, v in metrics.items()},
        'avg_len_words': float(avg_len_words),
    }


def compute_vqa_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_type: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        qtype = str(row.get('question_type', ''))
        by_type.setdefault(qtype, []).append(row)

    per_type: dict[str, Any] = {}
    type_accs: list[float] = []
    total_correct = 0
    for qtype, items in by_type.items():
        correct = sum(1 for row in items if bool(row.get('correct_normalized')))
        acc = float(correct) / float(len(items))
        type_accs.append(acc)
        total_correct += correct
        per_type[qtype] = {
            'paper_question_type': str(items[0].get('paper_question_type', qtype)),
            'num_rows': len(items),
            'accuracy': acc,
            'accuracy_x100': acc * 100.0,
        }

    overall = float(total_correct) / float(len(rows)) if rows else 0.0
    average_accuracy = sum(type_accs) / float(len(type_accs)) if type_accs else 0.0
    return {
        'num_rows': len(rows),
        'per_type': per_type,
        'average_accuracy': average_accuracy,
        'average_accuracy_x100': average_accuracy * 100.0,
        'overall_accuracy': overall,
        'overall_accuracy_x100': overall * 100.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description='Evaluate zero-shot predictions on RSGPT single-task benchmarks.')
    parser.add_argument('--output-dir', type=str, required=True)
    parser.add_argument('--summary-out', type=str, default='')
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    if not out_dir.is_absolute():
        out_dir = (project_root() / out_dir).resolve()

    caption_files = ['ucm_captions.jsonl', 'sydney_captions.jsonl', 'rsicd.jsonl']
    vqa_files = ['rsvqa_hr_test1.jsonl', 'rsvqa_hr_test2.jsonl', 'rsvqa_lr_test.jsonl']

    result: dict[str, Any] = {
        'output_dir': str(out_dir),
        'caption': {},
        'vqa': {},
    }

    for name in caption_files:
        path = out_dir / name
        rows = read_jsonl(path)
        if rows:
            result['caption'][path.stem] = compute_caption_metrics(rows)

    for name in vqa_files:
        path = out_dir / name
        rows = read_jsonl(path)
        if rows:
            result['vqa'][path.stem] = compute_vqa_metrics(rows)

    out_path = Path(args.summary_out) if args.summary_out else out_dir / 'evaluation_summary.json'
    if not out_path.is_absolute():
        out_path = (project_root() / out_path).resolve()
    write_json(out_path, result)
    print(f'[OK] Wrote evaluation summary: {out_path}', flush=True)


if __name__ == '__main__':
    main()
