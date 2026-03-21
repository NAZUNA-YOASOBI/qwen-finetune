from __future__ import annotations

import argparse
import gc
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any, Callable

SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from shared.common import append_jsonl, prediction_key, read_json, read_jsonl, resolve_dataset_image_paths, resolve_from_project, slice_by_shard


@dataclass(frozen=True)
class TaskSpec:
    name: str
    desc: str
    data_default: str
    prompt_version: str
    default_max_new_tokens: int
    prompt_builder: Callable[[dict[str, Any], str], str]
    conversation_builder: Callable[[list[Path], str], list[dict[str, Any]]]


def _mcq_prompt(row: dict[str, Any], prompt_text: str) -> str:
    cls_description = str(row.get("cls_description", "")).strip()
    parts = [
        "For the given the Multiple Choice Question Answer below, analyze the question and answer strictly from one of the options below.",
        "Strictly answer the choice only. No additional text.",
        "Provide only the letter (A., B., C., D. or E.) corresponding to the correct answer for the multiple-choice question given.",
    ]
    if cls_description:
        parts.append(cls_description)
    parts.append(str(prompt_text))
    parts.append(f"Options: {row.get('options', '')}")
    return "\n".join(parts)


def _caption_prompt(_row: dict[str, Any], prompt_text: str) -> str:
    return "\n".join([str(prompt_text), "Write around 140 words."])


def _single_conversation(image_paths: list[Path], prompt: str) -> list[dict[str, Any]]:
    return [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": str(image_paths[0])},
                {"type": "text", "text": str(prompt)},
            ],
        }
    ]


def _temporal_conversation(image_paths: list[Path], prompt: str) -> list[dict[str, Any]]:
    content: list[dict[str, Any]] = [
        {"type": "text", "text": "The following images show the condition before and after the event."},
        {"type": "text", "text": "This is the 'pre' image:"},
        {"type": "image", "image": str(image_paths[0])},
        {"type": "text", "text": "This is the 'post' image:"},
        {"type": "image", "image": str(image_paths[-1])},
        {"type": "text", "text": str(prompt)},
    ]
    return [{"role": "user", "content": content}]


TASK_SPECS: dict[str, TaskSpec] = {
    "single": TaskSpec(
        name="single",
        desc="GeoBench-VLM single-image MCQ generation",
        data_default="../../../GeoBench-VLM/dataset/GEOBench-VLM/Single/qa.json",
        prompt_version="official_mcq_single_v1",
        default_max_new_tokens=32,
        prompt_builder=_mcq_prompt,
        conversation_builder=_single_conversation,
    ),
    "temporal": TaskSpec(
        name="temporal",
        desc="GeoBench-VLM temporal MCQ generation",
        data_default="../../../GeoBench-VLM/dataset/GEOBench-VLM/Temporal/qa.json",
        prompt_version="official_mcq_temporal_all_frames_v1",
        default_max_new_tokens=32,
        prompt_builder=_mcq_prompt,
        conversation_builder=_temporal_conversation,
    ),
    "captioning": TaskSpec(
        name="captioning",
        desc="GeoBench-VLM captioning generation",
        data_default="../../../GeoBench-VLM/dataset/GEOBench-VLM/Captioning/qa.json",
        prompt_version="dataset_caption_prompt_v1",
        default_max_new_tokens=512,
        prompt_builder=_caption_prompt,
        conversation_builder=_single_conversation,
    ),
}


def _default_model_dir(model_family: str) -> str:
    if str(model_family) == "qwen3vl":
        return "../../../VRSBench/models/Qwen3-VL-8B-Instruct"
    if str(model_family) == "qwen35":
        return "../../../../fine-tune-qwen3.5/models/Qwen3.5-9B"
    raise ValueError(f"Unsupported model family: {model_family}")


def _build_runner(model_family: str, args) -> Any:
    from legacy.geobenchvlm_models import Qwen35Runner, Qwen3VLRunner

    if str(model_family) == "qwen3vl":
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
    if str(model_family) == "qwen35":
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
    raise ValueError(f"Unsupported model family: {model_family}")


def _load_done_keys(out_path: Path, *, benchmark_task: str, model_family: str, model_dir: Path, prompt_version: str) -> set[str]:
    if not out_path.is_file():
        return set()
    done: set[str] = set()
    for row in read_jsonl(out_path, allow_truncated_last_line=True):
        if str(row.get("benchmark_task", "")) != str(benchmark_task):
            continue
        if str(row.get("model_family", "")) != str(model_family):
            continue
        if str(row.get("model_dir", "")) != str(model_dir):
            continue
        if str(row.get("prompt_version", "")) != str(prompt_version):
            continue
        done.add(prediction_key(row.get("question_id"), row.get("prompt_index")))
    return done


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="GeoBench-VLM generation runner.")
    parser.add_argument("--task", type=str, required=True, choices=sorted(TASK_SPECS.keys()))
    parser.add_argument("--model-family", type=str, required=True, choices=["qwen3vl", "qwen35"])
    parser.add_argument("--model-dir", type=str, default="")
    parser.add_argument("--data", type=str, default="")
    parser.add_argument("--data-root", type=str, default="../../../GeoBench-VLM/dataset/GEOBench-VLM")
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=0)
    parser.add_argument("--do-sample", dest="do_sample", action="store_true", default=None)
    parser.add_argument("--no-sample", dest="do_sample", action="store_false")
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--num-beams", type=int, default=None)
    parser.add_argument("--repetition-penalty", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--device-map", type=str, default="auto")
    parser.add_argument("--dtype", type=str, default="auto", choices=["auto", "fp16", "bf16", "fp32"])
    parser.add_argument("--max-rows", type=int, default=0)
    parser.add_argument("--shard-world-size", type=int, default=1)
    parser.add_argument("--shard-rank", type=int, default=0)
    parser.add_argument("--shard-weights", type=str, default="")
    return parser


def run_generation(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    spec = TASK_SPECS[str(args.task)]
    if not str(args.model_dir).strip():
        args.model_dir = _default_model_dir(str(args.model_family))
    if not str(args.data).strip():
        args.data = spec.data_default
    if int(args.max_new_tokens) <= 0:
        args.max_new_tokens = int(spec.default_max_new_tokens)

    data_path = resolve_from_project(args.data)
    data_root = resolve_from_project(args.data_root)
    out_path = resolve_from_project(args.output)
    model_dir = resolve_from_project(args.model_dir)

    if not data_path.is_file():
        raise FileNotFoundError(f"Missing input json file: {data_path}")
    if not data_root.is_dir():
        raise FileNotFoundError(f"Missing dataset root: {data_root}")

    rows = read_json(data_path)
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"No rows loaded from input json: {data_path}")
    if args.max_rows and int(args.max_rows) > 0:
        rows = rows[: int(args.max_rows)]
    rows = slice_by_shard(
        rows,
        world_size=int(args.shard_world_size),
        rank=int(args.shard_rank),
        weights=str(args.shard_weights),
        key_name="question_id",
    )
    if not rows:
        print(f"[INFO] No rows assigned to current shard for {data_path}")
        return

    expanded_rows: list[dict[str, Any]] = []
    for row in rows:
        prompts = list(row.get("prompts", []))
        for prompt_index, prompt_text in enumerate(prompts):
            expanded_rows.append(
                {
                    "row": row,
                    "question_id": row.get("question_id"),
                    "prompt_index": int(prompt_index),
                    "prompt_count": int(len(prompts)),
                    "prompt_text": str(prompt_text),
                }
            )

    done_keys = _load_done_keys(
        out_path,
        benchmark_task=str(args.task),
        model_family=str(args.model_family),
        model_dir=model_dir,
        prompt_version=str(spec.prompt_version),
    )
    pending = [item for item in expanded_rows if prediction_key(item["question_id"], item["prompt_index"]) not in done_keys]
    if not pending:
        print(f"[OK] No pending rows. Output already complete: {out_path}")
        return

    runner = _build_runner(str(args.model_family), args)

    from tqdm import tqdm  # type: ignore
    import torch

    requested_batch_size = max(1, int(args.batch_size))
    current_batch_size = int(requested_batch_size)
    index = 0
    pbar = tqdm(total=len(pending), desc=f"{args.task}:{args.model_family}")

    while index < len(pending):
        chunk = pending[index : index + current_batch_size]
        conversations: list[list[dict[str, Any]]] = []
        image_paths_list: list[list[Path]] = []
        prompts: list[str] = []

        for item in chunk:
            row = item["row"]
            image_paths = resolve_dataset_image_paths(data_root, row.get("image_path"))
            prompt = spec.prompt_builder(row, item["prompt_text"])
            conversations.append(spec.conversation_builder(image_paths, prompt))
            image_paths_list.append(image_paths)
            prompts.append(prompt)

        try:
            predictions = runner.generate_batch_conversations(conversations)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            gc.collect()
            if current_batch_size <= 1:
                raise
            current_batch_size = max(1, int(current_batch_size) // 2)
            continue

        decode_strategy = str(getattr(runner, "decode_strategy", ""))
        effective_batch_size = len(chunk)
        for item, image_paths, prompt, prediction in zip(chunk, image_paths_list, prompts, predictions):
            row = item["row"]
            payload = {
                "benchmark_task": str(args.task),
                "task": str(row.get("task", "")),
                "question_id": row.get("question_id"),
                "prompt_index": int(item["prompt_index"]),
                "prompt_count": int(item["prompt_count"]),
                "prompt": str(prompt),
                "image_paths": [str(path) for path in image_paths],
                "image_count": int(len(image_paths)),
                "answer": str(prediction.text),
                "model_family": str(args.model_family),
                "model_dir": str(model_dir),
                "max_new_tokens": int(args.max_new_tokens),
                "device_map": str(args.device_map),
                "dtype": str(args.dtype),
                "do_sample": args.do_sample,
                "temperature": args.temperature,
                "top_p": args.top_p,
                "top_k": args.top_k,
                "num_beams": args.num_beams,
                "repetition_penalty": args.repetition_penalty,
                "seed": args.seed,
                "batch_size": int(effective_batch_size),
                "requested_batch_size": int(requested_batch_size),
                "decode_strategy": decode_strategy,
                "generated_token_count": int(prediction.generated_token_count),
                "generation_ended_by_eos": bool(prediction.ended_by_eos),
                "generation_last_token_id": prediction.last_generated_token_id,
                "shard_world_size": int(args.shard_world_size),
                "shard_rank": int(args.shard_rank),
                "shard_weights": str(args.shard_weights),
                "prompt_version": str(spec.prompt_version),
            }
            for key in (
                "ground_truth",
                "ground_truth_option",
                "options",
                "options_list",
                "image_name",
                "cls_description",
                "image_size",
                "source",
            ):
                if key in row:
                    payload[key] = row[key]
            append_jsonl(out_path, payload)

        index += len(chunk)
        pbar.update(len(chunk))
    pbar.close()
    print(f"[OK] Wrote predictions: {out_path}")


if __name__ == "__main__":
    run_generation()
