from __future__ import annotations

import argparse
import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image


DEFAULT_PROMPT_TEMPLATE = (
    "You are a visual grounding assistant.\n"
    "Given an image and a referring expression, output strict JSON only.\n"
    "Referring expression: {ref_sentence}\n"
    "Image: {image_name}\n"
    "JSON schema:\n"
    "{{\"instance\": \"short target description\", \"bbox_2d\": [x0, y0, x1, y1]}}\n"
    "Rules:\n"
    "1) Prefer bbox_2d in 0..1000 normalized coordinates.\n"
    "2) If uncertain, return empty bbox arrays."
)


@dataclass
class GridCell:
    grid_id: str
    x0: int
    y0: int
    x1: int
    y1: int


def _project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _resolve_from_project(path: str | Path) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    return (_project_root() / p).resolve()


def _parse_shard_weights(weights: str, *, world_size: int) -> list[int] | None:
    w = str(weights).strip()
    if not w:
        return None
    vals = [int(x.strip()) for x in w.split(":") if x.strip()]
    if len(vals) != int(world_size):
        raise ValueError(f"shard_weights expects {world_size} values, got {len(vals)}: {weights}")
    if any(v <= 0 for v in vals):
        raise ValueError(f"shard_weights must be positive integers: {weights}")
    return vals


def _slice_by_shard(items: list[dict[str, Any]], *, world_size: int, rank: int, weights: str, key_name: str) -> list[dict[str, Any]]:
    if int(world_size) <= 0:
        raise ValueError(f"shard_world_size must be >=1, got {world_size}")
    if int(rank) < 0 or int(rank) >= int(world_size):
        raise ValueError(f"shard_rank out of range: rank={rank}, world_size={world_size}")

    parsed = _parse_shard_weights(weights, world_size=int(world_size))
    total = len(items)

    if parsed is None:
        shard = [it for i, it in enumerate(items) if (i % int(world_size)) == int(rank)]
    else:
        denom = int(sum(parsed))
        left = int(sum(parsed[: int(rank)]))
        right = int(sum(parsed[: int(rank) + 1]))
        start = (total * left) // denom
        end = (total * right) // denom
        shard = items[start:end]

    first_key = shard[0].get(key_name, "") if shard else ""
    last_key = shard[-1].get(key_name, "") if shard else ""
    print(
        f"[INFO] shard rank={rank}/{world_size} weights={weights or 'even'} "
        f"selected={len(shard)}/{total} first_{key_name}={first_key} last_{key_name}={last_key}",
        flush=True,
    )
    return shard


def build_grid(width: int, height: int, rows: int, cols: int) -> list[GridCell]:
    cells: list[GridCell] = []
    cell_w = math.ceil(width / cols)
    cell_h = math.ceil(height / rows)

    for r in range(rows):
        for c in range(cols):
            x0 = c * cell_w
            y0 = r * cell_h
            x1 = min((c + 1) * cell_w, width)
            y1 = min((r + 1) * cell_h, height)
            cells.append(
                GridCell(
                    grid_id=f"R{r + 1:02d}C{c + 1:02d}",
                    x0=x0,
                    y0=y0,
                    x1=x1,
                    y1=y1,
                )
            )
    return cells


def clamp_bbox_xyxy(bbox: list[float], width: int, height: int) -> list[int] | None:
    if len(bbox) != 4:
        return None

    try:
        x0, y0, x1, y1 = [float(v) for v in bbox]
    except Exception:
        return None

    x0 = max(0.0, min(float(width), x0))
    y0 = max(0.0, min(float(height), y0))
    x1 = max(0.0, min(float(width), x1))
    y1 = max(0.0, min(float(height), y1))

    if x1 <= x0 or y1 <= y0:
        return None

    return [int(round(x0)), int(round(y0)), int(round(x1)), int(round(y1))]


def clamp_bbox_2d_1000(bbox: list[float]) -> list[float] | None:
    if len(bbox) != 4:
        return None
    try:
        x0, y0, x1, y1 = [float(v) for v in bbox]
    except Exception:
        return None

    x0 = max(0.0, min(1000.0, x0))
    y0 = max(0.0, min(1000.0, y0))
    x1 = max(0.0, min(1000.0, x1))
    y1 = max(0.0, min(1000.0, y1))

    if x1 <= x0 or y1 <= y0:
        return None
    return [x0, y0, x1, y1]


def bbox2d_to_pixels(bbox_2d: list[float], width: int, height: int) -> list[int] | None:
    bb = clamp_bbox_2d_1000(bbox_2d)
    if bb is None:
        return None
    x0, y0, x1, y1 = bb
    px = [
        x0 * width / 1000.0,
        y0 * height / 1000.0,
        x1 * width / 1000.0,
        y1 * height / 1000.0,
    ]
    return clamp_bbox_xyxy(px, width, height)


def pixels_to_bbox2d_1000(bbox_xyxy: list[int], width: int, height: int) -> list[float] | None:
    if len(bbox_xyxy) != 4 or width <= 0 or height <= 0:
        return None
    x0, y0, x1, y1 = bbox_xyxy
    bb = [
        x0 * 1000.0 / float(width),
        y0 * 1000.0 / float(height),
        x1 * 1000.0 / float(width),
        y1 * 1000.0 / float(height),
    ]
    return clamp_bbox_2d_1000(bb)


def intersection_area(a: list[int], b: list[int]) -> int:
    x0 = max(a[0], b[0])
    y0 = max(a[1], b[1])
    x1 = min(a[2], b[2])
    y1 = min(a[3], b[3])
    if x1 <= x0 or y1 <= y0:
        return 0
    return (x1 - x0) * (y1 - y0)


def grid_ids_from_bbox(cells: list[GridCell], bbox: list[int]) -> list[str]:
    picked: list[str] = []
    for c in cells:
        cell_bbox = [c.x0, c.y0, c.x1, c.y1]
        if intersection_area(cell_bbox, bbox) > 0:
            picked.append(c.grid_id)
    return picked


def try_parse_json(text: str) -> dict[str, Any] | None:
    text = text.strip()
    if not text:
        return None

    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        snippet = text[start : end + 1]
        try:
            obj = json.loads(snippet)
            if isinstance(obj, dict):
                return obj
        except Exception:
            return None

    return None


def strip_thinking_and_role_markers(text: str) -> str:
    s = (text or "").strip()
    if not s:
        return s

    s = re.sub(r"^<\|im_start\|>\s*assistant\s*", "", s, flags=re.IGNORECASE)
    s = re.sub(r"^assistant\s*", "", s, flags=re.IGNORECASE)
    s = s.replace("<|im_end|>", "").strip()

    if "<think>" in s:
        if "</think>" in s:
            s = s.split("</think>", 1)[1].strip()
        else:
            s = s.replace("<think>", "").strip()
    return s


def extract_bbox_from_anywhere(parsed: dict[str, Any] | None, raw_text: str) -> tuple[str | None, list[float] | None]:
    if isinstance(parsed, dict):
        v2d = parsed.get("bbox_2d", None)
        if isinstance(v2d, list) and len(v2d) == 4:
            return "bbox_2d", [float(x) for x in v2d]

        vxy = parsed.get("bbox_xyxy", None)
        if isinstance(vxy, list) and len(vxy) == 4:
            return "bbox_xyxy", [float(x) for x in vxy]

    patterns = [
        r"\[\s*([0-9]+(?:\.[0-9]+)?)\s*,\s*([0-9]+(?:\.[0-9]+)?)\s*,\s*([0-9]+(?:\.[0-9]+)?)\s*,\s*([0-9]+(?:\.[0-9]+)?)\s*\]",
        r"\(\s*([0-9]+(?:\.[0-9]+)?)\s*,\s*([0-9]+(?:\.[0-9]+)?)\s*,\s*([0-9]+(?:\.[0-9]+)?)\s*,\s*([0-9]+(?:\.[0-9]+)?)\s*\)",
    ]
    for pat in patterns:
        m = re.search(pat, raw_text)
        if m:
            nums = [float(m.group(i)) for i in range(1, 5)]
            if any(v > 1000.0 for v in nums):
                return "regex_xyxy", nums
            return "regex_2d", nums

    return None, None


def normalize_result(
    parsed: dict[str, Any] | None,
    raw_text: str,
    cells: list[GridCell],
    width: int,
    height: int,
) -> dict[str, Any]:
    instance = ""
    if isinstance(parsed, dict):
        instance = str(parsed.get("instance", "")).strip()

    bbox_type, bbox_vals = extract_bbox_from_anywhere(parsed, raw_text)
    bbox_original: list[int] | None = None
    bbox_2d: list[float] = []
    grid_ids: list[str] = []

    if bbox_type and bbox_vals:
        if bbox_type in ("bbox_2d", "regex_2d"):
            bbox_original = bbox2d_to_pixels(bbox_vals, width, height)
        elif bbox_type in ("bbox_xyxy", "regex_xyxy"):
            max_x = max(float(bbox_vals[0]), float(bbox_vals[2]))
            max_y = max(float(bbox_vals[1]), float(bbox_vals[3]))
            if (max_x > width or max_y > height) and (max_x <= 1000.0 and max_y <= 1000.0):
                bbox_original = bbox2d_to_pixels(bbox_vals, width, height)
                bbox_type = "bbox_xyxy_as_2d"
            else:
                bbox_original = clamp_bbox_xyxy(bbox_vals, width, height)

    if bbox_original is not None:
        bb_2d = pixels_to_bbox2d_1000(bbox_original, width, height)
        if bb_2d is not None:
            bbox_2d = bb_2d
        grid_ids = grid_ids_from_bbox(cells, bbox_original)

    return {
        "instance": instance,
        "bbox_2d": bbox_2d,
        "grid_ids": grid_ids,
        "debug": {"bbox_source": bbox_type or ""},
    }


def _build_prompt(prompt_template: str, *, ref_sentence: str, image_name: str) -> str:
    return str(prompt_template).format(ref_sentence=ref_sentence, image_name=image_name)


def main() -> None:
    parser = argparse.ArgumentParser(description="VRSBench referring baseline generation with noft-style grounding format.")
    parser.add_argument("--model-dir", type=str, default="models/Qwen3-VL-8B-Instruct")
    parser.add_argument("--data", type=str, default="benchmark/vrsbench/data/vrsbench_referring_test.jsonl")
    parser.add_argument(
        "--output",
        type=str,
        default="benchmark/vrsbench/outputs/baseline_referring_noftstyle_predictions_test.jsonl",
    )
    parser.add_argument("--prompt-template", type=str, default=DEFAULT_PROMPT_TEMPLATE)
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
    parser.add_argument("--device-map", type=str, default="cuda:0")
    parser.add_argument("--dtype", type=str, default="cuda:0", choices=["auto", "fp16", "bf16", "fp32"])
    parser.add_argument("--max-items", type=int, default=0)
    parser.add_argument("--grid-rows", type=int, default=6)
    parser.add_argument("--grid-cols", type=int, default=6)
    parser.add_argument("--shard-world-size", type=int, default=1)
    parser.add_argument("--shard-rank", type=int, default=0)
    parser.add_argument("--shard-weights", type=str, default="")
    args = parser.parse_args()

    sys.path.insert(0, str(_project_root() / "src"))

    from tqdm import tqdm  # type: ignore

    from ftqwen3.device import assert_model_on_cuda, require_cuda
    from ftqwen3.jsonl import append_jsonl, read_jsonl
    from ftqwen3.qwen_dinov3 import build_generate_kwargs, maybe_set_generation_seed, torch_dtype_from_str

    data_path = _resolve_from_project(args.data)
    if not data_path.is_file():
        raise FileNotFoundError(f"Missing prepared file: {data_path}")

    rows = read_jsonl(data_path)
    rows = sorted(rows, key=lambda x: int(x["qid"]))
    if args.max_items and int(args.max_items) > 0:
        rows = rows[: int(args.max_items)]
    rows = _slice_by_shard(
        rows,
        world_size=int(args.shard_world_size),
        rank=int(args.shard_rank),
        weights=str(args.shard_weights),
        key_name="qid",
    )

    out_path = _resolve_from_project(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    done: set[int] = set()
    if out_path.exists():
        for r in read_jsonl(out_path):
            try:
                done.add(int(r.get("qid")))
            except Exception:
                continue

    pending = [r for r in rows if int(r["qid"]) not in done]
    if not pending:
        print(f"[OK] No pending samples. Output already complete: {out_path}")
        return

    require_cuda()

    import gc
    import torch
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration  # type: ignore

    model_dir = _resolve_from_project(args.model_dir)
    gen_cfg = build_generate_kwargs(
        max_new_tokens=int(args.max_new_tokens),
        do_sample=args.do_sample,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        num_beams=args.num_beams,
        repetition_penalty=args.repetition_penalty,
    )
    maybe_set_generation_seed(args.seed)
    processor = AutoProcessor.from_pretrained(str(model_dir))
    try:
        processor.tokenizer.padding_side = "left"
        if processor.tokenizer.pad_token_id is None and processor.tokenizer.eos_token_id is not None:
            processor.tokenizer.pad_token_id = processor.tokenizer.eos_token_id
    except Exception:
        pass

    dtype = torch_dtype_from_str(str(args.dtype))
    try:
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            str(model_dir),
            dtype=dtype,
            device_map=str(args.device_map),
        )
    except TypeError:
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            str(model_dir),
            torch_dtype=dtype,
            device_map=str(args.device_map),
        )
    model.eval()
    assert_model_on_cuda(model)

    max_batch_size = max(1, int(args.batch_size))
    cur_bs = int(max_batch_size)
    pbar = tqdm(total=len(pending), desc="referring:vrsbench:baseline:noftstyle")
    idx = 0

    while idx < len(pending):
        chunk = pending[idx : idx + cur_bs]
        prepared: list[dict[str, Any]] = []
        prep_errors: list[dict[str, Any]] = []
        conversations: list[list[dict[str, Any]]] = []

        for it in chunk:
            img_path = _resolve_from_project(str(it["image_path"]))
            try:
                with Image.open(img_path) as img:
                    width, height = img.size
                question = str(it.get("question", "")).strip()
                image_name = str(it.get("image_id", it.get("filename", img_path.name)))
                prompt = _build_prompt(
                    args.prompt_template,
                    ref_sentence=question,
                    image_name=image_name,
                )
                prepared.append(
                    {
                        "qid": int(it["qid"]),
                        "image_id": image_name,
                        "filename": str(it.get("filename", "")),
                        "image_path": str(Path(it.get("image_path", ""))),
                        "question": question,
                        "ground_truth": str(it.get("ground_truth", "")),
                        "is_unique": bool(it.get("is_unique", it.get("unique", False))),
                        "prompt": prompt,
                        "width": int(width),
                        "height": int(height),
                        "cells": build_grid(int(width), int(height), int(args.grid_rows), int(args.grid_cols)),
                    }
                )
                conversations.append(
                    [
                        {
                            "role": "user",
                            "content": [
                                {"type": "image", "image": str(img_path)},
                                {"type": "text", "text": prompt},
                            ],
                        }
                    ]
                )
            except Exception as e:
                prep_errors.append(
                    {
                        "qid": int(it["qid"]),
                        "image_id": str(it.get("image_id", it.get("filename", ""))),
                        "filename": str(it.get("filename", "")),
                        "image_path": str(Path(it.get("image_path", ""))),
                        "question": str(it.get("question", "")),
                        "ground_truth": str(it.get("ground_truth", "")),
                        "is_unique": bool(it.get("is_unique", it.get("unique", False))),
                        "answer_raw": "",
                        "answer_json": None,
                        "prediction": {"instance": "", "bbox_2d": [], "grid_ids": [], "debug": {"bbox_source": ""}},
                        "image_width": None,
                        "image_height": None,
                        "prompt": "",
                        "prompt_template": str(args.prompt_template),
                        "model": "qwen3-vl-baseline-noftstyle",
                        "model_dir": str(Path(args.model_dir)),
                        "max_new_tokens": int(args.max_new_tokens),
                        "do_sample": args.do_sample,
                        "temperature": args.temperature,
                        "top_p": args.top_p,
                        "top_k": args.top_k,
                        "num_beams": args.num_beams,
                        "repetition_penalty": args.repetition_penalty,
                        "seed": args.seed,
                        "batch_size": 0,
                        "requested_batch_size": int(max_batch_size),
                        "decode_strategy": str(gen_cfg.strategy),
                        "shard_world_size": int(args.shard_world_size),
                        "shard_rank": int(args.shard_rank),
                        "shard_weights": str(args.shard_weights),
                        "error": f"prepare_error: {e}",
                    }
                )

        if not prepared:
            for rec in prep_errors:
                append_jsonl(out_path, rec)
                pbar.update(1)
            idx += len(chunk)
            continue

        def _run_batch(batch_conversations: list[list[dict[str, Any]]]) -> list[str]:
            inputs = processor.apply_chat_template(
                batch_conversations,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
                padding=True,
            )
            inputs = inputs.to(model.device)
            with torch.no_grad():
                generated_ids = model.generate(**inputs, **dict(gen_cfg.gen_kwargs))

            input_ids = inputs.get("input_ids", None)
            if input_ids is not None:
                prompt_len = int(input_ids.shape[1])
                prompt_lens = [prompt_len] * int(generated_ids.shape[0])
            else:
                attn = inputs.get("attention_mask", None)
                if attn is None:
                    raise RuntimeError("Missing both input_ids and attention_mask while trimming generated ids.")
                prompt_len = int(attn.shape[1])
                prompt_lens = [prompt_len] * int(generated_ids.shape[0])

            trimmed = [out[int(pl) :] for out, pl in zip(generated_ids, prompt_lens)]
            texts = processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            return [str(x).strip() for x in texts]

        try:
            raw_outputs = _run_batch(conversations)
            batch_records: list[dict[str, Any]] = []
            for item, ans in zip(prepared, raw_outputs):
                clean_text = strip_thinking_and_role_markers(str(ans))
                parsed = try_parse_json(clean_text)
                pred = normalize_result(
                    parsed=parsed,
                    raw_text=clean_text or str(ans),
                    cells=item["cells"],
                    width=int(item["width"]),
                    height=int(item["height"]),
                )
                batch_records.append(
                    {
                        "qid": int(item["qid"]),
                        "image_id": str(item["image_id"]),
                        "filename": str(item["filename"]),
                        "image_path": str(item["image_path"]),
                        "question": str(item["question"]),
                        "ground_truth": str(item["ground_truth"]),
                        "is_unique": bool(item["is_unique"]),
                        "answer_raw": str(ans).strip(),
                        "answer_json": parsed,
                        "prediction": pred,
                        "image_width": int(item["width"]),
                        "image_height": int(item["height"]),
                        "prompt": str(item["prompt"]),
                        "prompt_template": str(args.prompt_template),
                        "model": "qwen3-vl-baseline-noftstyle",
                        "model_dir": str(Path(args.model_dir)),
                        "max_new_tokens": int(args.max_new_tokens),
                        "do_sample": args.do_sample,
                        "temperature": args.temperature,
                        "top_p": args.top_p,
                        "top_k": args.top_k,
                        "num_beams": args.num_beams,
                        "repetition_penalty": args.repetition_penalty,
                        "seed": args.seed,
                        "batch_size": int(len(prepared)),
                        "requested_batch_size": int(max_batch_size),
                        "decode_strategy": str(gen_cfg.strategy),
                        "shard_world_size": int(args.shard_world_size),
                        "shard_rank": int(args.shard_rank),
                        "shard_weights": str(args.shard_weights),
                    }
                )
        except torch.cuda.OutOfMemoryError:
            gc.collect()
            if cur_bs <= 1:
                raise
            cur_bs = max(1, int(cur_bs) // 2)
            continue
        except Exception as e:
            batch_records = []
            for item, conv in zip(prepared, conversations):
                try:
                    raw_text = _run_batch([conv])[0]
                    clean_text = strip_thinking_and_role_markers(str(raw_text))
                    parsed = try_parse_json(clean_text)
                    pred = normalize_result(
                        parsed=parsed,
                        raw_text=clean_text or str(raw_text),
                        cells=item["cells"],
                        width=int(item["width"]),
                        height=int(item["height"]),
                    )
                    batch_records.append(
                        {
                            "qid": int(item["qid"]),
                            "image_id": str(item["image_id"]),
                            "filename": str(item["filename"]),
                            "image_path": str(item["image_path"]),
                            "question": str(item["question"]),
                            "ground_truth": str(item["ground_truth"]),
                            "is_unique": bool(item["is_unique"]),
                            "answer_raw": str(raw_text).strip(),
                            "answer_json": parsed,
                            "prediction": pred,
                            "image_width": int(item["width"]),
                            "image_height": int(item["height"]),
                            "prompt": str(item["prompt"]),
                            "prompt_template": str(args.prompt_template),
                            "model": "qwen3-vl-baseline-noftstyle",
                            "model_dir": str(Path(args.model_dir)),
                            "max_new_tokens": int(args.max_new_tokens),
                            "do_sample": args.do_sample,
                            "temperature": args.temperature,
                            "top_p": args.top_p,
                            "top_k": args.top_k,
                            "num_beams": args.num_beams,
                            "repetition_penalty": args.repetition_penalty,
                            "seed": args.seed,
                            "batch_size": 1,
                            "requested_batch_size": int(max_batch_size),
                            "decode_strategy": str(gen_cfg.strategy),
                            "shard_world_size": int(args.shard_world_size),
                            "shard_rank": int(args.shard_rank),
                            "shard_weights": str(args.shard_weights),
                        }
                    )
                except Exception as e2:
                    batch_records.append(
                        {
                            "qid": int(item["qid"]),
                            "image_id": str(item["image_id"]),
                            "filename": str(item["filename"]),
                            "image_path": str(item["image_path"]),
                            "question": str(item["question"]),
                            "ground_truth": str(item["ground_truth"]),
                            "is_unique": bool(item["is_unique"]),
                            "answer_raw": "",
                            "answer_json": None,
                            "prediction": {"instance": "", "bbox_2d": [], "grid_ids": [], "debug": {"bbox_source": ""}},
                            "image_width": int(item["width"]),
                            "image_height": int(item["height"]),
                            "prompt": str(item["prompt"]),
                            "prompt_template": str(args.prompt_template),
                            "model": "qwen3-vl-baseline-noftstyle",
                            "model_dir": str(Path(args.model_dir)),
                            "max_new_tokens": int(args.max_new_tokens),
                            "do_sample": args.do_sample,
                            "temperature": args.temperature,
                            "top_p": args.top_p,
                            "top_k": args.top_k,
                            "num_beams": args.num_beams,
                            "repetition_penalty": args.repetition_penalty,
                            "seed": args.seed,
                            "batch_size": 1,
                            "requested_batch_size": int(max_batch_size),
                            "decode_strategy": str(gen_cfg.strategy),
                            "shard_world_size": int(args.shard_world_size),
                            "shard_rank": int(args.shard_rank),
                            "shard_weights": str(args.shard_weights),
                            "error": f"inference_error: {e2} (batch_error: {e})",
                        }
                    )

        for rec in prep_errors:
            append_jsonl(out_path, rec)
            pbar.update(1)
        for rec in batch_records:
            append_jsonl(out_path, rec)
            pbar.update(1)

        idx += len(chunk)

    pbar.close()
    print(f"[OK] Wrote predictions: {out_path}")


if __name__ == "__main__":
    main()
