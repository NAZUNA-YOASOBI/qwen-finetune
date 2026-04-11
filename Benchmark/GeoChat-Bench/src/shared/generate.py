from __future__ import annotations

import argparse
import gc
from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any, Callable

from shared.common import append_jsonl, find_image_path, read_jsonl, slice_by_shard


@dataclass(frozen=True)
class TaskSpec:
    name: str
    desc: str
    data_default: str
    image_field: str | None
    image_id_field: str | None
    default_ext: str | None
    prompt_version: str
    prompt_builder: Callable[[dict[str, Any]], str]


REGION_CAPTION_STYLE_HINT = (
    " Answer with one short phrase or one short sentence. "
    "Keep it concise, around 1 short sentence, and do not exceed 2 very short sentences."
)

REFER_JSON_STYLE_HINT = (
    " Output strict JSON only. "
    'If the query refers to one target, return {"instance": "short target description", "bbox_2d": [x0, y0, x1, y1]}. '
    'If the query refers to multiple targets, return a JSON array, and each element must use this schema: {"instance": "short target description", "bbox_2d": [x0, y0, x1, y1]}. '
    "Prefer bbox_2d in 0..1000 normalized coordinates. "
    "If uncertain, return an empty bbox array or an empty JSON array."
)

def _scene_prompt(row: dict[str, Any]) -> str:
    return str(row["text"])


def _vqa_prompt(row: dict[str, Any]) -> str:
    return str(row["text"])


def _region_caption_prompt(row: dict[str, Any]) -> str:
    return f"[identify] What is the object present at {row['question']}{REGION_CAPTION_STYLE_HINT}"


def _referring_prompt(row: dict[str, Any]) -> str:
    qtype = str(row.get("type", "")).strip().lower()
    question = str(row["question"])
    image_name = str(row.get("image_id", row.get("image", ""))).strip()
    if qtype == "ref":
        return (
            "You are a visual grounding assistant.\n"
            "Given an image and a referring expression, output strict JSON only.\n"
            f"Referring expression: {question}\n"
            f"Image: {image_name}\n"
            f"{REFER_JSON_STYLE_HINT}"
        )
    return (
        "You are a visual grounding assistant.\n"
        "Given an image and a grounding query, output strict JSON only.\n"
        f"Grounding query: {question}\n"
        f"Image: {image_name}\n"
        f"{REFER_JSON_STYLE_HINT}"
    )


def _resolve_prompt_version(*, task: str, row: dict[str, Any], default_prompt_version: str) -> str:
    if str(task) != "referring":
        return str(default_prompt_version)
    return "bbox2d1000_qwen_native_style_auto_v3"


TASK_SPECS: dict[str, TaskSpec] = {
    "scene": TaskSpec(
        name="scene",
        desc="GeoChat scene classification generation",
        data_default="dataset/GeoChat-Bench/aid.jsonl",
        image_field="image",
        image_id_field=None,
        default_ext=None,
        prompt_version="dataset_text_passthrough_v1",
        prompt_builder=_scene_prompt,
    ),
    "vqa": TaskSpec(
        name="vqa",
        desc="GeoChat VQA generation",
        data_default="dataset/GeoChat-Bench/hrben.jsonl",
        image_field="image",
        image_id_field=None,
        default_ext=None,
        prompt_version="dataset_text_passthrough_v1",
        prompt_builder=_vqa_prompt,
    ),
    "region_caption": TaskSpec(
        name="region_caption",
        desc="GeoChat region caption generation",
        data_default="dataset/GeoChat-Bench/region_captioning.jsonl",
        image_field=None,
        image_id_field="image_id",
        default_ext=".png",
        prompt_version="identify_short_phrase_or_short_sentence_v2",
        prompt_builder=_region_caption_prompt,
    ),
    "referring": TaskSpec(
        name="referring",
        desc="GeoChat referring generation",
        data_default="dataset/GeoChat-Bench/referring.jsonl",
        image_field=None,
        image_id_field="image_id",
        default_ext=".png",
        prompt_version="bbox2d1000_qwen_native_style_v1",
        prompt_builder=_referring_prompt,
    ),
}


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _resolve_from_project(path: str | Path) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    return (_project_root() / p).resolve()


def _default_model_dir(model_family: str) -> str:
    if str(model_family) == "qwen3vl":
        return "../VRSBench/models/Qwen3-VL-8B-Instruct"
    if str(model_family) == "qwen35":
        return "../../fine-tune-qwen3.5/models/Qwen3.5-9B"
    raise ValueError(f"Unsupported model family: {model_family}")


def _load_reusable_done_keys(
    out_path: Path,
    *,
    task: str,
    model_family: str,
    model_dir: Path,
    rows: list[dict[str, Any]],
    default_prompt_version: str,
) -> set[str]:
    if not out_path.is_file():
        return set()

    expected_row_map = {str(row.get("question_id")): row for row in rows if row.get("question_id") is not None}
    done_keys: set[str] = set()
    for row in read_jsonl(out_path, allow_truncated_last_line=True):
        qid = str(row.get("question_id", "")).strip()
        if not qid or qid not in expected_row_map:
            continue
        if str(row.get("task", "")) != str(task):
            continue
        if str(row.get("model_family", "")) != str(model_family):
            continue
        if str(row.get("model_dir", "")) != str(model_dir):
            continue
        expected_prompt_version = _resolve_prompt_version(
            task=str(task),
            row=expected_row_map[qid],
            default_prompt_version=str(default_prompt_version),
        )
        if str(row.get("prompt_version", "")) != str(expected_prompt_version):
            continue
        done_keys.add(qid)
    return done_keys


def _build_runner(model_family: str, args) -> Any:
    if str(model_family) == "qwen3vl":
        from qwen3vl.runner import Qwen3VLRunner

        return Qwen3VLRunner(
            _resolve_from_project(args.model_dir),
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
        from qwen35.runner import Qwen35Runner

        return Qwen35Runner(
            _resolve_from_project(args.model_dir),
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


def build_parser(*, task: str, model_family: str) -> argparse.ArgumentParser:
    spec = TASK_SPECS[str(task)]
    parser = argparse.ArgumentParser(description=f"{spec.desc} ({model_family}).")
    parser.add_argument("--model-dir", type=str, default=_default_model_dir(model_family))
    parser.add_argument("--data", type=str, default=spec.data_default)
    parser.add_argument("--image-root", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=256)
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


def run_generation(*, task: str, model_family: str, argv: list[str] | None = None) -> None:
    spec = TASK_SPECS[str(task)]
    parser = build_parser(task=task, model_family=model_family)
    args = parser.parse_args(argv)

    data_path = _resolve_from_project(args.data)
    out_path = _resolve_from_project(args.output)
    if not data_path.is_file():
        raise FileNotFoundError(f"Missing input jsonl file: {data_path}")

    rows = read_jsonl(data_path)
    if not rows:
        raise ValueError(f"No rows loaded from input jsonl: {data_path}")
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

    done_keys = _load_reusable_done_keys(
        out_path,
        task=str(task),
        model_family=str(model_family),
        model_dir=_resolve_from_project(args.model_dir),
        rows=rows,
        default_prompt_version=str(spec.prompt_version),
    )
    pending = [row for row in rows if str(row.get("question_id")) not in done_keys]
    if not pending:
        print(f"[OK] No pending rows. Output already complete: {out_path}")
        return

    runner = _build_runner(model_family, args)

    from tqdm import tqdm  # type: ignore

    import torch

    max_batch_size = max(1, int(args.batch_size))
    cur_bs = int(max_batch_size)

    pbar = tqdm(total=len(pending), desc=f"{task}:{model_family}")
    idx = 0
    while idx < len(pending):
        chunk = pending[idx : idx + cur_bs]
        image_paths = [
            find_image_path(
                _resolve_from_project(args.image_root),
                image_value=row.get(spec.image_field, None) if spec.image_field else None,
                image_id=row.get(spec.image_id_field, None) if spec.image_id_field else None,
                default_ext=spec.default_ext,
            )
            for row in chunk
        ]
        prompts = [spec.prompt_builder(row) for row in chunk]
        effective_bs = len(chunk)

        try:
            preds = runner.generate_batch(image_paths=image_paths, prompts=prompts)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            gc.collect()
            if cur_bs <= 1:
                raise
            cur_bs = max(1, int(cur_bs) // 2)
            continue

        for row, image_path, prompt, pred in zip(chunk, image_paths, prompts, preds):
            answer = re.sub(r"\s+", " ", str(pred.text).strip())
            payload = {
                "question_id": row.get("question_id"),
                "image_id": row.get("image_id", row.get("image", "")),
                "image": row.get("image", ""),
                "image_path": str(image_path),
                "answer": str(answer),
                "prompt": str(prompt),
                "task": str(task),
                "model_family": str(model_family),
                "model_dir": str(_resolve_from_project(args.model_dir)),
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
                "batch_size": int(effective_bs),
                "requested_batch_size": int(max_batch_size),
                "decode_strategy": str(getattr(runner, "decode_strategy", "")),
                "generated_token_count": int(pred.generated_token_count),
                "generation_ended_by_eos": bool(pred.ended_by_eos),
                "generation_last_token_id": pred.last_generated_token_id,
                "shard_world_size": int(args.shard_world_size),
                "shard_rank": int(args.shard_rank),
                "shard_weights": str(args.shard_weights),
                "prompt_version": _resolve_prompt_version(
                    task=str(task),
                    row=row,
                    default_prompt_version=str(spec.prompt_version),
                ),
            }
            for key in ("text", "question", "ground_truth", "category", "dataset", "type", "obj_ids", "size_group"):
                if key in row:
                    payload[key] = row[key]
            append_jsonl(out_path, payload)

        idx += len(chunk)
        pbar.update(len(chunk))
    pbar.close()
    print(f"[OK] Wrote predictions: {out_path}")
