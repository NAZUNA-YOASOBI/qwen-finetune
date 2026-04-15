from __future__ import annotations

import argparse
import gc
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any

SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from legacy.geobenchvlm_models import GenerateResult, Qwen35Runner, Qwen3VLRunner, _summarize_generated_sequences
from shared.common import append_jsonl, prediction_key, read_json, read_jsonl, resolve_dataset_image_paths, resolve_from_project, slice_by_shard


@dataclass(frozen=True)
class TaskSpec:
    name: str
    data_default: str
    prompt_version: str
    default_max_new_tokens: int


TASK_SPEC = TaskSpec(
    name='ref_det',
    data_default='../../../GeoBench-VLM/dataset/GEOBench-VLM/Ref-Det/qa.json',
    prompt_version='bbox2d1000_xyxy_json_array_v2',
    default_max_new_tokens=512,
)

DEFAULT_PROMPT_TEMPLATE = (
    'You are a visual grounding assistant.\n'
    'Given an image and a referring expression, output strict JSON only.\n'
    'Referring expression: {prompt_text}\n'
    'Always return a JSON array.\n'
    'If the expression refers to one target, return a JSON array with exactly one element:\n'
    '[{{"instance": "short target description", "bbox_2d": [x0, y0, x1, y1]}}]\n'
    'If the expression refers to multiple targets, return a JSON array, and each element must use:\n'
    '{{"instance": "short target description", "bbox_2d": [x0, y0, x1, y1]}}\n'
    'Use integer bbox_2d coordinates in 0..1000 normalized scale.\n'
    'Use only two corner points, not polygon points.\n'
    'Ensure x0 < x1 and y0 < y1.\n'
    'Do not output markdown fences or explanation.'
)


def _default_model_dir(model_family: str) -> str:
    if str(model_family) == 'qwen3vl':
        return '../../../VRSBench/models/Qwen3-VL-8B-Instruct'
    if str(model_family) == 'qwen35':
        return '../../../../fine-tune-qwen3.5/models/Qwen3.5-9B'
    raise ValueError(f'Unsupported model family: {model_family}')


def _build_runner(model_family: str, args) -> Any:
    if str(model_family) == 'qwen3vl':
        return Qwen3VLRunner(
            resolve_from_project(args.model_dir),
            device_map=str(args.device_map),
            dtype=str(args.dtype),
            max_new_tokens=int(args.max_new_tokens),
            do_sample=args.do_sample,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            num_beams=args.num_beams,
            repetition_penalty=args.repetition_penalty,
            seed=args.seed,
        )
    if str(model_family) == 'qwen35':
        return Qwen35Runner(
            resolve_from_project(args.model_dir),
            device_map=str(args.device_map),
            dtype=str(args.dtype),
            max_new_tokens=int(args.max_new_tokens),
            do_sample=args.do_sample,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            num_beams=args.num_beams,
            repetition_penalty=args.repetition_penalty,
            seed=args.seed,
            enable_thinking=False,
        )
    raise ValueError(f'Unsupported model family: {model_family}')


def _load_done_keys(out_path: Path, *, model_family: str, model_dir: Path, prompt_version: str) -> set[str]:
    if not out_path.is_file():
        return set()
    done: set[str] = set()
    for row in read_jsonl(out_path, allow_truncated_last_line=True):
        if str(row.get('benchmark_task', '')) != str(TASK_SPEC.name):
            continue
        if str(row.get('model_family', '')) != str(model_family):
            continue
        if str(row.get('model_dir', '')) != str(model_dir):
            continue
        if str(row.get('prompt_version', '')) != str(prompt_version):
            continue
        done.add(prediction_key(row.get('question_id'), row.get('prompt_index')))
    return done


def _build_prompt(prompt_text: str) -> str:
    return str(DEFAULT_PROMPT_TEMPLATE).format(prompt_text=str(prompt_text))


def _build_conversation(image_path: Path, prompt: str) -> list[dict[str, Any]]:
    return [
        {
            'role': 'user',
            'content': [
                {'type': 'image', 'image': str(image_path)},
                {'type': 'text', 'text': str(prompt)},
            ],
        }
    ]


def _prompt_lens_from_inputs(inputs: Any) -> list[int]:
    input_ids = inputs.get('input_ids', None)
    if input_ids is not None and hasattr(input_ids, 'shape'):
        return [int(input_ids.shape[1])] * int(input_ids.shape[0])
    attention_mask = inputs.get('attention_mask', None)
    if attention_mask is not None and hasattr(attention_mask, 'shape'):
        return [int(attention_mask.shape[1])] * int(attention_mask.shape[0])
    raise RuntimeError('Missing both input_ids and attention_mask while trimming generated ids.')


def _get_special_token_ids(runner: Any) -> tuple[Any, Any]:
    generation_config = getattr(runner.model, 'generation_config', None)
    eos_token_id = getattr(generation_config, 'eos_token_id', None) if generation_config is not None else None
    pad_token_id = getattr(generation_config, 'pad_token_id', None) if generation_config is not None else None
    tokenizer = getattr(runner.processor, 'tokenizer', None)
    if eos_token_id is None and tokenizer is not None:
        eos_token_id = getattr(tokenizer, 'eos_token_id', None)
    if pad_token_id is None and tokenizer is not None:
        pad_token_id = getattr(tokenizer, 'pad_token_id', None)
    return eos_token_id, pad_token_id


def _decode_results(runner: Any, summaries: list[Any]) -> list[GenerateResult]:
    texts = runner.processor.batch_decode(
        [item.token_ids for item in summaries],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    return [
        GenerateResult(
            text=str(text).strip(),
            generated_token_count=int(item.generated_token_count),
            ended_by_eos=bool(item.ended_by_eos),
            last_generated_token_id=item.last_generated_token_id,
        )
        for text, item in zip(texts, summaries)
    ]


def _generate_batch_exact_trim(runner: Any, conversations: list[list[dict[str, Any]]]) -> list[GenerateResult]:
    import torch

    if not conversations:
        return []

    if isinstance(runner, Qwen3VLRunner):
        inputs = runner.processor.apply_chat_template(
            conversations,
            tokenize=True,
            add_generation_prompt=True,
            padding=True,
            return_dict=True,
            return_tensors='pt',
        )
        inputs = inputs.to(runner.model.device)
        with torch.no_grad():
            generated_ids = runner.model.generate(**inputs, **runner.gen_kwargs)
        prompt_lens = _prompt_lens_from_inputs(inputs)
        eos_token_id, pad_token_id = _get_special_token_ids(runner)
        summaries = _summarize_generated_sequences(
            generated_ids,
            prompt_lens,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
        )
        return _decode_results(runner, summaries)

    if isinstance(runner, Qwen35Runner):
        rendered_texts: list[str] = []
        image_batches: list[list[Any]] = []
        for conversation in conversations:
            converted_messages, image_objects = runner._convert_conversation(conversation)
            rendered = runner.processor.tokenizer.apply_chat_template(
                converted_messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=bool(runner.enable_thinking),
            )
            rendered_texts.append(str(rendered))
            image_batches.append(image_objects)
        inputs = runner.processor(
            text=rendered_texts,
            images=image_batches,
            padding=True,
            return_tensors='pt',
        )
        inputs = inputs.to(runner.model.device)
        with torch.no_grad():
            generated_ids = runner.model.generate(**inputs, **runner.gen_kwargs)
        prompt_lens = _prompt_lens_from_inputs(inputs)
        eos_token_id, pad_token_id = _get_special_token_ids(runner)
        summaries = _summarize_generated_sequences(
            generated_ids,
            prompt_lens,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
        )
        return _decode_results(runner, summaries)

    return runner.generate_batch_conversations(conversations)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='GeoBench-VLM Ref-Det xyxy1000 generation runner.')
    parser.add_argument('--model-family', type=str, required=True, choices=['qwen3vl', 'qwen35'])
    parser.add_argument('--model-dir', type=str, default='')
    parser.add_argument('--data', type=str, default='')
    parser.add_argument('--data-root', type=str, default='../../../GeoBench-VLM/dataset/GEOBench-VLM')
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--max-new-tokens', type=int, default=0)
    parser.add_argument('--do-sample', dest='do_sample', action='store_true', default=None)
    parser.add_argument('--no-sample', dest='do_sample', action='store_false')
    parser.add_argument('--temperature', type=float, default=None)
    parser.add_argument('--top-p', type=float, default=None)
    parser.add_argument('--top-k', type=int, default=None)
    parser.add_argument('--num-beams', type=int, default=None)
    parser.add_argument('--repetition-penalty', type=float, default=None)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--device-map', type=str, default='cuda:0')
    parser.add_argument('--dtype', type=str, default='cuda:0', choices=['auto', 'fp16', 'bf16', 'fp32'])
    parser.add_argument('--max-rows', type=int, default=0)
    parser.add_argument('--shard-world-size', type=int, default=1)
    parser.add_argument('--shard-rank', type=int, default=0)
    parser.add_argument('--shard-weights', type=str, default='')
    return parser


def run_generation(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    if not str(args.model_dir).strip():
        args.model_dir = _default_model_dir(str(args.model_family))
    if not str(args.data).strip():
        args.data = TASK_SPEC.data_default
    if int(args.max_new_tokens) <= 0:
        args.max_new_tokens = int(TASK_SPEC.default_max_new_tokens)

    data_path = resolve_from_project(args.data)
    data_root = resolve_from_project(args.data_root)
    out_path = resolve_from_project(args.output)
    model_dir = resolve_from_project(args.model_dir)

    if not data_path.is_file():
        raise FileNotFoundError(f'Missing input json file: {data_path}')
    if not data_root.is_dir():
        raise FileNotFoundError(f'Missing dataset root: {data_root}')

    rows = read_json(data_path)
    if not isinstance(rows, list) or not rows:
        raise ValueError(f'No rows loaded from input json: {data_path}')
    if args.max_rows and int(args.max_rows) > 0:
        rows = rows[: int(args.max_rows)]
    rows = slice_by_shard(
        rows,
        world_size=int(args.shard_world_size),
        rank=int(args.shard_rank),
        weights=str(args.shard_weights),
        key_name='question_id',
    )
    if not rows:
        print(f'[INFO] No rows assigned to current shard for {data_path}')
        return

    expanded_rows: list[dict[str, Any]] = []
    for row in rows:
        prompts = list(row.get('prompts', []))
        for prompt_index, prompt_text in enumerate(prompts):
            expanded_rows.append(
                {
                    'row': row,
                    'question_id': row.get('question_id'),
                    'prompt_index': int(prompt_index),
                    'prompt_count': int(len(prompts)),
                    'prompt_text': str(prompt_text),
                }
            )

    done_keys = _load_done_keys(
        out_path,
        model_family=str(args.model_family),
        model_dir=model_dir,
        prompt_version=str(TASK_SPEC.prompt_version),
    )
    pending = [item for item in expanded_rows if prediction_key(item['question_id'], item['prompt_index']) not in done_keys]
    if not pending:
        print(f'[OK] No pending rows. Output already complete: {out_path}')
        return

    runner = _build_runner(str(args.model_family), args)

    from tqdm import tqdm  # type: ignore
    import torch

    requested_batch_size = max(1, int(args.batch_size))
    current_batch_size = int(requested_batch_size)
    index = 0
    pbar = tqdm(total=len(pending), desc=f'{TASK_SPEC.name}:{args.model_family}:xyxy1000')

    while index < len(pending):
        chunk = pending[index : index + current_batch_size]
        conversations: list[list[dict[str, Any]]] = []
        image_paths: list[Path] = []
        prompts: list[str] = []

        for item in chunk:
            row = item['row']
            resolved_paths = resolve_dataset_image_paths(data_root, row.get('image_path'))
            if len(resolved_paths) != 1:
                raise ValueError(f'Ref-Det expects exactly one image, got {len(resolved_paths)} for question_id={row.get("question_id")}')
            prompt = _build_prompt(item['prompt_text'])
            conversations.append(_build_conversation(resolved_paths[0], prompt))
            image_paths.append(resolved_paths[0])
            prompts.append(prompt)

        try:
            predictions = _generate_batch_exact_trim(runner, conversations)
        except torch.cuda.OutOfMemoryError:
            gc.collect()
            if current_batch_size <= 1:
                raise
            current_batch_size = max(1, int(current_batch_size) // 2)
            print(f'[WARN] CUDA OOM, reduce batch size to {current_batch_size}', flush=True)
            continue

        decode_strategy = str(getattr(runner, 'decode_strategy', ''))
        effective_batch_size = len(chunk)
        for item, image_path, prompt, prediction in zip(chunk, image_paths, prompts, predictions):
            row = item['row']
            payload = {
                'benchmark_task': str(TASK_SPEC.name),
                'task': str(row.get('task', '')),
                'question_id': row.get('question_id'),
                'prompt_index': int(item['prompt_index']),
                'prompt_count': int(item['prompt_count']),
                'prompt': str(prompt),
                'image_paths': [str(image_path)],
                'image_count': 1,
                'answer': str(prediction.text),
                'model_family': str(args.model_family),
                'model_dir': str(model_dir),
                'max_new_tokens': int(args.max_new_tokens),
                'device_map': str(args.device_map),
                'dtype': str(args.dtype),
                'do_sample': args.do_sample,
                'temperature': args.temperature,
                'top_p': args.top_p,
                'top_k': args.top_k,
                'num_beams': args.num_beams,
                'repetition_penalty': args.repetition_penalty,
                'seed': args.seed,
                'batch_size': int(effective_batch_size),
                'requested_batch_size': int(requested_batch_size),
                'decode_strategy': decode_strategy,
                'generated_token_count': int(prediction.generated_token_count),
                'generation_ended_by_eos': bool(prediction.ended_by_eos),
                'generation_last_token_id': prediction.last_generated_token_id,
                'shard_world_size': int(args.shard_world_size),
                'shard_rank': int(args.shard_rank),
                'shard_weights': str(args.shard_weights),
                'prompt_version': str(TASK_SPEC.prompt_version),
            }
            for key in ('ground_truth', 'image_name', 'cls_description', 'image_size', 'source', 'prompts'):
                if key in row:
                    payload[key] = row[key]
            append_jsonl(out_path, payload)

        index += len(chunk)
        pbar.update(len(chunk))
    pbar.close()
    print(f'[OK] Wrote predictions: {out_path}')


if __name__ == '__main__':
    run_generation()
