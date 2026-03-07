from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding='utf-8'))


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding='utf-8')


def fmt(value: Any, digits: int = 2) -> str:
    if value is None:
        return '-'
    try:
        return f'{float(value):.{digits}f}'
    except Exception:
        return str(value)


def make_caption_table(title: str, paper_rows: list[dict[str, Any]], ours: dict[str, Any] | None) -> str:
    headers = ['Method', 'BLEU-1', 'BLEU-2', 'BLEU-3', 'BLEU-4', 'METEOR', 'ROUGE_L', 'CIDEr']
    lines = [f'## {title}', '', '| ' + ' | '.join(headers) + ' |', '| ' + ' | '.join(['---'] * len(headers)) + ' |']
    for row in paper_rows:
        lines.append('| ' + ' | '.join([
            str(row.get('Method', '')),
            fmt(row.get('BLEU-1')),
            fmt(row.get('BLEU-2')),
            fmt(row.get('BLEU-3')),
            fmt(row.get('BLEU-4')),
            fmt(row.get('METEOR')),
            fmt(row.get('ROUGE_L')),
            fmt(row.get('CIDEr')),
        ]) + ' |')
    if ours is not None:
        mx = ours.get('metrics_x100', {})
        lines.append('| ' + ' | '.join([
            'Qwen3-VL-8B-Instruct (zero-shot)',
            fmt(mx.get('BLEU-1')),
            fmt(mx.get('BLEU-2')),
            fmt(mx.get('BLEU-3')),
            fmt(mx.get('BLEU-4')),
            fmt(mx.get('METEOR')),
            fmt(mx.get('ROUGE_L')),
            fmt(mx.get('CIDEr')),
        ]) + ' |')
        lines.append('')
        lines.append(f"- Avg caption length: {fmt(ours.get('avg_len_words'))} words")
    lines.append('')
    return '\n'.join(lines)


def make_vqa_table(title: str, paper_rows: list[dict[str, Any]], ours: dict[str, Any] | None, columns: list[str]) -> str:
    headers = ['Method'] + columns + ['Average Accuracy']
    lines = [f'## {title}', '', '| ' + ' | '.join(headers) + ' |', '| ' + ' | '.join(['---'] * len(headers)) + ' |']
    for row in paper_rows:
        vals = [str(row.get('Method', ''))]
        for col in columns:
            vals.append(fmt(row.get(col)))
        vals.append(fmt(row.get('Average Accuracy')))
        lines.append('| ' + ' | '.join(vals) + ' |')
    if ours is not None:
        per_type = ours.get('per_type', {})
        vals = ['Qwen3-VL-8B-Instruct (zero-shot)']
        for col in columns:
            key = col.lower().replace('/', '_')
            if key == 'comparison':
                key = 'comp'
            if key == 'rural_urban':
                key = 'rural_urban'
            if key == 'presence':
                key = 'presence'
            item = per_type.get(key, {})
            vals.append(fmt(item.get('accuracy_x100')))
        vals.append(fmt(ours.get('average_accuracy_x100')))
        lines.append('| ' + ' | '.join(vals) + ' |')
        lines.append('')
        lines.append(f"- Overall accuracy: {fmt(ours.get('overall_accuracy_x100'))}")
    lines.append('')
    return '\n'.join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description='Make comparison markdown for RSGPT single-task benchmarks.')
    parser.add_argument('--paper-json', type=str, default='RSGPT-Simbench/reference/rsgpt_paper_single_task_tables_2307_15266.json')
    parser.add_argument('--eval-json', type=str, required=True)
    parser.add_argument('--output-md', type=str, required=True)
    args = parser.parse_args()

    paper_path = Path(args.paper_json)
    eval_path = Path(args.eval_json)
    out_path = Path(args.output_md)
    if not paper_path.is_absolute():
        paper_path = (project_root() / paper_path).resolve()
    if not eval_path.is_absolute():
        eval_path = (project_root() / eval_path).resolve()
    if not out_path.is_absolute():
        out_path = (project_root() / out_path).resolve()

    paper = read_json(paper_path)
    ours = read_json(eval_path)

    md: list[str] = []
    md.append('# RSGPT Single-Task Benchmark Comparison\n')
    md.append('- Paper source: `tmp/2307.15266_layout.txt`')
    md.append(f'- Our evaluation: `{eval_path}`')
    md.append('- Note: our row is raw zero-shot `Qwen3-VL-8B-Instruct`, while the paper `RSGPT (ours)` row is fine-tuned.')
    md.append('')

    md.append(make_caption_table('UCM-captions', paper['caption']['ucm_captions']['rows'], ours.get('caption', {}).get('ucm_captions')))
    md.append(make_caption_table('Sydney-captions', paper['caption']['sydney_captions']['rows'], ours.get('caption', {}).get('sydney_captions')))
    md.append(make_caption_table('RSICD', paper['caption']['rsicd']['rows'], ours.get('caption', {}).get('rsicd')))
    md.append(make_vqa_table('RSVQA-HR Test Set 1', paper['vqa']['rsvqa_hr_test1']['rows'], ours.get('vqa', {}).get('rsvqa_hr_test1'), ['Presence', 'Comparison']))
    md.append(make_vqa_table('RSVQA-HR Test Set 2', paper['vqa']['rsvqa_hr_test2']['rows'], ours.get('vqa', {}).get('rsvqa_hr_test2'), ['Presence', 'Comparison']))
    md.append(make_vqa_table('RSVQA-LR Test Set', paper['vqa']['rsvqa_lr_test']['rows'], ours.get('vqa', {}).get('rsvqa_lr_test'), ['Presence', 'Comparison', 'Rural/Urban']))

    write_text(out_path, '\n'.join(md).rstrip() + '\n')
    print(f'[OK] Wrote markdown report: {out_path}', flush=True)


if __name__ == '__main__':
    main()
