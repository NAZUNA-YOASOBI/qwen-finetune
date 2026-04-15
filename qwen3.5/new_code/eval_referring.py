import json
import math
import os
import re
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image, ImageOps

try:
    import torch
except Exception:
    torch = None

try:
    from transformers import AutoModelForImageTextToText, AutoProcessor
except Exception:
    AutoModelForImageTextToText = None
    AutoProcessor = None


CONFIG: Dict[str, Any] = {
    # modes: "dataset_eval"
    "mode": "dataset_eval",
    "model": "Qwen/Qwen3.5-4B",
    "temperature": 0.0,
    # Qwen Non-Thinking mode (official): chat_template_kwargs.enable_thinking=False
    "enable_thinking": False,
    "max_output_tokens": 256,
    "DEBUG_RAW_OUTPUT": False,
    # dataset mode
    "eval_json": "data/VRSBench_EVAL_referring.json",
    "image_dir": "data/Images_val",
    "dataset_output": "vg_eval_predictions.jsonl",
    "metrics_output": "vg_eval_metrics.json",
    "start_idx": 0,
    "limit": 0,
    "batch_size": 64,
}


@dataclass
class GridCell:
    grid_id: str
    x0: int
    y0: int
    x1: int
    y1: int


def build_grid(width: int, height: int, rows: int, cols: int) -> List[GridCell]:
    cells: List[GridCell] = []
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


def default_prompt(query: str, image_name: str) -> str:
    return (
        "You are a visual grounding assistant.\n"
        "Given an image and a referring expression, output strict JSON only.\n"
        f"Referring expression: {query}\n"
        f"Image: {image_name}\n"
        "JSON schema:\n"
        "{"
        ' "instance": "short target description",'
        ' "bbox_2d": [x0, y0, x1, y1]'
        "}\n"
        "Rules:\n"
        "1) Use bbox_2d only, in 0..1000 normalized coordinates.\n"
        "2) Do not output bbox_xyxy, pixel coordinates, or free-text coordinates.\n"
        "3) If uncertain, return empty bbox arrays.\n"
    )


def _basename(path: str) -> str:
    return os.path.basename(path)


def clamp_bbox_xyxy(bbox: List[float], width: int, height: int) -> Optional[List[int]]:
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


def clamp_bbox_2d_1000(bbox: List[float]) -> Optional[List[float]]:
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


def bbox2d_to_pixels(bbox_2d: List[float], width: int, height: int) -> Optional[List[int]]:
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


def pixels_to_bbox2d_1000(bbox_xyxy: List[int], width: int, height: int) -> Optional[List[float]]:
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


def intersection_area(a: List[int], b: List[int]) -> int:
    x0 = max(a[0], b[0])
    y0 = max(a[1], b[1])
    x1 = min(a[2], b[2])
    y1 = min(a[3], b[3])
    if x1 <= x0 or y1 <= y0:
        return 0
    return (x1 - x0) * (y1 - y0)


def bbox_iou(a: List[int], b: List[int]) -> float:
    inter = float(intersection_area(a, b))
    if inter <= 0.0:
        return 0.0
    area_a = float(max(0, a[2] - a[0]) * max(0, a[3] - a[1]))
    area_b = float(max(0, b[2] - b[0]) * max(0, b[3] - b[1]))
    union = area_a + area_b - inter
    if union <= 0.0:
        return 0.0
    return inter / union


def grid_ids_from_bbox(cells: List[GridCell], bbox: List[int]) -> List[str]:
    picked: List[str] = []
    for c in cells:
        cell_bbox = [c.x0, c.y0, c.x1, c.y1]
        if intersection_area(cell_bbox, bbox) > 0:
            picked.append(c.grid_id)
    return picked


def try_parse_json(text: str) -> Optional[Dict[str, Any]]:
    text = text.strip()
    if not text:
        return None

    def _coerce_json_object(obj: Any) -> Optional[Dict[str, Any]]:
        if isinstance(obj, dict):
            return obj
        if isinstance(obj, list) and obj and isinstance(obj[0], dict):
            return obj[0]
        return None

    try:
        obj = json.loads(text)
        parsed = _coerce_json_object(obj)
        if parsed is not None:
            return parsed
    except Exception:
        pass

    list_start = text.find("[")
    list_end = text.rfind("]")
    if list_start != -1 and list_end != -1 and list_end > list_start:
        snippet = text[list_start : list_end + 1]
        try:
            obj = json.loads(snippet)
            parsed = _coerce_json_object(obj)
            if parsed is not None:
                return parsed
        except Exception:
            pass

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        snippet = text[start : end + 1]
        try:
            obj = json.loads(snippet)
            parsed = _coerce_json_object(obj)
            if parsed is not None:
                return parsed
        except Exception:
            return None

    return None


def apply_chat_template_with_thinking(
    processor: Any, messages: List[Dict[str, Any]], enable_thinking: bool
) -> str:
    # Pass enable_thinking as a top-level template argument.
    return processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )


def slice_generated_tokens(generated_ids: Any, inputs: Dict[str, Any]) -> List[Any]:
    # For decoder-only batched generation with left padding, generated sequences share
    # the same input width. Slicing by attention_mask.sum() leaks prompt suffix tokens.
    if "input_ids" in inputs:
        prompt_len = int(inputs["input_ids"].shape[1])
    elif "attention_mask" in inputs:
        prompt_len = int(inputs["attention_mask"].shape[1])
    else:
        raise KeyError("Expected input_ids or attention_mask in model inputs.")

    return [generated_ids[i, prompt_len:] for i in range(generated_ids.shape[0])]


def strip_output_wrappers(text: str) -> str:
    s = (text or "").strip()
    if not s:
        return s

    # Remove common chat role wrappers.
    s = re.sub(r"^<\|im_start\|>\s*assistant\s*", "", s, flags=re.IGNORECASE)
    s = re.sub(r"^assistant\s*", "", s, flags=re.IGNORECASE)
    s = s.replace("<|im_end|>", "").strip()
    s = re.sub(r"^```(?:json)?\s*", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\s*```$", "", s)
    return s


def extract_bbox_from_anywhere(
    parsed: Optional[Dict[str, Any]], raw_text: str
) -> Tuple[Optional[str], Optional[List[float]]]:
    if isinstance(parsed, dict):
        v2d = parsed.get("bbox_2d", None)
        if isinstance(v2d, list) and len(v2d) == 4:
            return "bbox_2d", v2d

    return None, None


def normalize_result(
    parsed: Optional[Dict[str, Any]],
    raw_text: str,
    cells: List[GridCell],
    width: int,
    height: int,
) -> Dict[str, Any]:
    instance = ""

    if isinstance(parsed, dict):
        instance = str(parsed.get("instance", "")).strip()

    bbox_type, bbox_vals = extract_bbox_from_anywhere(parsed, raw_text)
    bbox_original: Optional[List[int]] = None
    bbox_2d: List[float] = []
    grid_ids: List[str] = []

    if bbox_type and bbox_vals:
        if bbox_type == "bbox_2d":
            bbox_original = bbox2d_to_pixels(bbox_vals, width, height)

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


def load_vlm(model: str) -> Tuple[Any, Any]:
    if torch is None or AutoModelForImageTextToText is None or AutoProcessor is None:
        raise RuntimeError(
            "Missing dependencies. Please install: pip install torch transformers accelerate pillow"
        )

    use_cuda = torch.cuda.is_available()
    dtype = torch.bfloat16 if use_cuda else torch.float32
    if use_cuda:
        # PyTorch 2.9+ recommends the new fp32_precision API for TF32 control.
        try:
            torch.backends.cuda.matmul.fp32_precision = "tf32"
            torch.backends.cudnn.conv.fp32_precision = "tf32"
        except Exception:
            # Backward compatibility for older torch versions.
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.set_float32_matmul_precision("high")

    processor = AutoProcessor.from_pretrained(model, trust_remote_code=True)
    # Decoder-only generation expects left padding for batched inputs.
    if hasattr(processor, "tokenizer") and processor.tokenizer is not None:
        processor.tokenizer.padding_side = "left"
        if processor.tokenizer.pad_token_id is None and processor.tokenizer.eos_token_id is not None:
            processor.tokenizer.pad_token_id = processor.tokenizer.eos_token_id
    vlm = AutoModelForImageTextToText.from_pretrained(
        model,
        dtype=dtype,
        device_map="cuda:0",
        trust_remote_code=True,
    )
    return processor, vlm


def call_lvlm_batch_with_loaded(
    processor: Any,
    vlm: Any,
    prompts: List[str],
    images: List[Image.Image],
    temperature: float,
    enable_thinking: bool,
    max_output_tokens: int,
    debug_raw_output: bool = False,
) -> List[Tuple[Optional[Dict[str, Any]], str]]:
    if len(prompts) != len(images):
        raise ValueError(
            f"prompts/images length mismatch: {len(prompts)} vs {len(images)}"
        )

    messages_list: List[List[Dict[str, Any]]] = []
    for prompt, image in zip(prompts, images):
        messages_list.append(
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": prompt},
                    ],
                }
            ]
        )

    text_inputs = [
        apply_chat_template_with_thinking(
            processor=processor, messages=messages, enable_thinking=enable_thinking
        )
        for messages in messages_list
    ]
    inputs = processor(
        text=text_inputs,
        images=images,
        padding=True,
        return_tensors="pt",
    )
    inputs = {k: v.to(vlm.device) for k, v in inputs.items()}

    do_sample = temperature > 0
    generation_kwargs: Dict[str, Any] = {
        "max_new_tokens": max_output_tokens,
        "do_sample": do_sample,
    }
    if do_sample:
        generation_kwargs["temperature"] = max(temperature, 1e-5)

    with torch.inference_mode():
        generated_ids = vlm.generate(**inputs, **generation_kwargs)

    generated_only_list = slice_generated_tokens(generated_ids, inputs)
    raw_texts = processor.batch_decode(
        generated_only_list,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    raw_texts = [x.strip() for x in raw_texts]
    clean_texts = [strip_output_wrappers(x) for x in raw_texts]

    if debug_raw_output:
        for i, txt in enumerate(raw_texts):
            print(f"\n========== RAW_MODEL_OUTPUT_BEGIN[{i}] ==========")
            print(txt)
            print(f"=========== RAW_MODEL_OUTPUT_END[{i}] ===========\n")

    results: List[Tuple[Optional[Dict[str, Any]], str]] = []
    for raw_txt, clean_txt in zip(raw_texts, clean_texts):
        txt = clean_txt or raw_txt or ""
        results.append((try_parse_json(txt), txt))
    return results


def parse_gt_bbox_1000(gt: str) -> Optional[List[float]]:
    if not isinstance(gt, str):
        return None
    nums = [float(x) for x in re.findall(r"\d+(?:\.\d+)?", gt)]
    if len(nums) != 4:
        return None
    max_v = max(nums)
    if max_v <= 100.0:
        return [v * 10.0 for v in nums]
    if max_v <= 1000.0:
        return nums
    return None


def run_dataset_eval_mode(processor: Any, vlm: Any) -> None:
    eval_json = str(CONFIG["eval_json"])
    image_dir = str(CONFIG["image_dir"])
    start_idx = int(CONFIG.get("start_idx", 0))
    limit = int(CONFIG.get("limit", 0))
    rows = int(CONFIG.get("rows", 6))
    cols = int(CONFIG.get("cols", 6))
    temperature = float(CONFIG["temperature"])
    enable_thinking = bool(CONFIG.get("enable_thinking", False))
    max_output_tokens = int(CONFIG["max_output_tokens"])
    debug_raw_output = bool(CONFIG.get("DEBUG_RAW_OUTPUT", False))
    pred_path = str(CONFIG.get("dataset_output", "vg_eval_predictions.jsonl"))
    metrics_path = str(CONFIG.get("metrics_output", "vg_eval_metrics.json"))
    batch_size = int(CONFIG.get("batch_size", 1))
    if batch_size <= 0:
        batch_size = 1

    with open(eval_json, "r", encoding="utf-8") as f:
        samples = json.load(f)
    if not isinstance(samples, list):
        raise ValueError("eval_json must be a list.")

    if start_idx < 0:
        start_idx = 0
    if start_idx >= len(samples):
        raise ValueError(f"start_idx={start_idx} out of range. total={len(samples)}")

    selected = samples[start_idx:]
    if limit > 0:
        selected = selected[:limit]

    total = len(selected)
    if total == 0:
        raise ValueError("No samples selected. Check start_idx/limit.")

    processed = 0
    valid = 0
    sum_iou = 0.0
    acc50 = 0
    batch_fallback_count = 0

    split_stats: Dict[str, Dict[str, int]] = {
        "all": {"total": 0, "valid": 0, "hit50": 0, "hit70": 0},
        "unique": {"total": 0, "valid": 0, "hit50": 0, "hit70": 0},
        "non_unique": {"total": 0, "valid": 0, "hit50": 0, "hit70": 0},
    }
    t0 = time.time()

    with open(pred_path, "w", encoding="utf-8") as wf:
        for chunk_start in range(0, total, batch_size):
            chunk = selected[chunk_start : chunk_start + batch_size]
            prepared: List[Dict[str, Any]] = []

            for item in chunk:
                image_id = str(item.get("image_id", ""))
                question = str(item.get("question", ""))
                gt_raw = item.get("ground_truth", "")
                qid = item.get("question_id", None)
                is_unique = bool(item.get("unique", False))
                split_key_item = "unique" if is_unique else "non_unique"
                image_path = os.path.join(image_dir, image_id)
                if not os.path.isfile(image_path):
                    split_stats["all"]["total"] += 1
                    split_stats[split_key_item]["total"] += 1
                    rec = {
                        "question_id": qid,
                        "image_id": image_id,
                        "unique": is_unique,
                        "error": "image_not_found",
                    }
                    wf.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    processed += 1
                    continue

                with Image.open(image_path) as img_fp:
                    img = ImageOps.exif_transpose(img_fp).copy()
                width, height = img.size
                prepared.append(
                    {
                        "qid": qid,
                        "image_id": image_id,
                        "question": question,
                        "unique": is_unique,
                        "gt_raw": gt_raw,
                        "img": img,
                        "width": width,
                        "height": height,
                        "cells": build_grid(width, height, rows, cols),
                        "prompt": default_prompt(query=question, image_name=image_id),
                    }
                )

            if not prepared:
                continue

            try:
                batch_outputs = call_lvlm_batch_with_loaded(
                    processor=processor,
                    vlm=vlm,
                    prompts=[p["prompt"] for p in prepared],
                    images=[p["img"] for p in prepared],
                    temperature=temperature,
                    enable_thinking=enable_thinking,
                    max_output_tokens=max_output_tokens,
                    debug_raw_output=debug_raw_output,
                )
            except Exception as e:
                batch_fallback_count += 1
                print(
                    f"Batch inference failed for chunk starting at {chunk_start}; "
                    f"falling back to per-sample retries. error={e}"
                )
                batch_outputs = []
                for p in prepared:
                    try:
                        parsed, raw_text = call_lvlm_batch_with_loaded(
                            processor=processor,
                            vlm=vlm,
                            prompts=[p["prompt"]],
                            images=[p["img"]],
                            temperature=temperature,
                            enable_thinking=enable_thinking,
                            max_output_tokens=max_output_tokens,
                            debug_raw_output=debug_raw_output,
                        )[0]
                        batch_outputs.append((parsed, raw_text))
                    except Exception as e2:
                        rec = {
                            "question_id": p["qid"],
                            "image_id": p["image_id"],
                            "question": p["question"],
                            "unique": p["unique"],
                            "ground_truth": p["gt_raw"],
                            "error": f"inference_error: {e2} (batch_error: {e})",
                        }
                        wf.write(json.dumps(rec, ensure_ascii=False) + "\n")
                        processed += 1
                        batch_outputs.append((None, "__ERROR_ALREADY_LOGGED__"))

            for p, (parsed, raw_text) in zip(prepared, batch_outputs):
                split_key = "unique" if bool(p.get("unique", False)) else "non_unique"
                split_stats["all"]["total"] += 1
                split_stats[split_key]["total"] += 1

                if raw_text == "__ERROR_ALREADY_LOGGED__":
                    continue

                if raw_text == "" and parsed is None:
                    rec = {
                        "question_id": p["qid"],
                        "image_id": p["image_id"],
                        "question": p["question"],
                        "unique": p["unique"],
                        "ground_truth": p["gt_raw"],
                        "error": "empty_model_output",
                    }
                    wf.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    processed += 1
                    continue

                pred = normalize_result(
                    parsed=parsed,
                    raw_text=raw_text,
                    cells=p["cells"],
                    width=p["width"],
                    height=p["height"],
                )

                gt_1000 = parse_gt_bbox_1000(str(p["gt_raw"]))
                gt_xyxy = bbox2d_to_pixels(gt_1000, p["width"], p["height"]) if gt_1000 else None
                pred_bbox2d = pred.get("bbox_2d", [])
                pred_xyxy = (
                    bbox2d_to_pixels(pred_bbox2d, p["width"], p["height"])
                    if isinstance(pred_bbox2d, list) and len(pred_bbox2d) == 4
                    else None
                )

                iou = 0.0
                is_hit50 = False
                is_hit70 = False
                if pred_xyxy is not None and gt_xyxy is not None:
                    iou = bbox_iou([int(v) for v in pred_xyxy], gt_xyxy)
                    is_hit50 = iou >= 0.5
                    is_hit70 = iou >= 0.7
                    valid += 1
                    sum_iou += iou
                    split_stats["all"]["valid"] += 1
                    split_stats[split_key]["valid"] += 1
                    if is_hit50:
                        acc50 += 1
                        split_stats["all"]["hit50"] += 1
                        split_stats[split_key]["hit50"] += 1
                    if is_hit70:
                        split_stats["all"]["hit70"] += 1
                        split_stats[split_key]["hit70"] += 1

                rec = {
                    "question_id": p["qid"],
                    "image_id": p["image_id"],
                    "question": p["question"],
                    "unique": p["unique"],
                    "ground_truth": p["gt_raw"],
                    "gt_bbox_xyxy": gt_xyxy or [],
                    "pred_bbox_xyxy": pred_xyxy or [],
                    "iou": iou,
                    "hit@0.5": is_hit50,
                    "hit@0.7": is_hit70,
                    "prediction": pred,
                }
                wf.write(json.dumps(rec, ensure_ascii=False) + "\n")
                processed += 1

            if processed % 10 == 0 or processed == total:
                elapsed = time.time() - t0
                speed = processed / max(1e-6, elapsed)
                print(
                    f"[{processed}/{total}] processed, valid={valid}, "
                    f"speed={speed:.3f} samples/s, batch_size={batch_size}"
                )

    split_metrics: Dict[str, Dict[str, float]] = {}
    for key in ("unique", "non_unique", "all"):
        total_k = split_stats[key]["total"]
        valid_k = split_stats[key]["valid"]
        hit50_k = split_stats[key]["hit50"]
        hit70_k = split_stats[key]["hit70"]
        split_metrics[key] = {
            "count_total": total_k,
            "count_valid_for_iou": valid_k,
            "acc@0.5": (hit50_k / total_k) if total_k > 0 else 0.0,
            "acc@0.7": (hit70_k / total_k) if total_k > 0 else 0.0,
            "acc@0.5_count": hit50_k,
            "acc@0.7_count": hit70_k,
        }

    metrics = {
        "total_selected": total,
        "processed": processed,
        "valid_for_iou": valid,
        "mean_iou": (sum_iou / valid) if valid > 0 else 0.0,
        "acc@0.5": (split_stats["all"]["hit50"] / split_stats["all"]["total"])
        if split_stats["all"]["total"] > 0
        else 0.0,
        "acc@0.7": (split_stats["all"]["hit70"] / split_stats["all"]["total"])
        if split_stats["all"]["total"] > 0
        else 0.0,
        "acc@0.5_count": acc50,
        "acc@0.7_count": split_stats["all"]["hit70"],
        "by_split": split_metrics,
        "predictions_file": pred_path,
        "start_idx": start_idx,
        "limit": limit,
        "elapsed_seconds": time.time() - t0,
        "batch_size": batch_size,
        "batch_fallback_count": batch_fallback_count,
    }

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print(json.dumps(metrics, ensure_ascii=False, indent=2))


def main() -> None:
    mode = str(CONFIG.get("mode", "dataset_eval"))
    model = str(CONFIG["model"])
    print(f"Loading model: {model}")
    processor, vlm = load_vlm(model=model)

    if mode == "dataset_eval":
        run_dataset_eval_mode(processor, vlm)
    else:
        raise ValueError(f"Unsupported mode: {mode}")


if __name__ == "__main__":
    main()
