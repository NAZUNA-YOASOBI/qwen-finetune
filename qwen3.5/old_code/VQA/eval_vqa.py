import json
import os
import re
import time
from collections import defaultdict
from pathlib import Path
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
    "model": "Qwen/Qwen3.5-4B",
    "temperature": 0.0,
    "enable_thinking": False,
    "max_output_tokens": 64,
    "DEBUG_RAW_OUTPUT": False,
    "eval_json": "data/VRSBench_EVAL_vqa.json",
    "image_dir": "data/Images_val",
    "dataset_output": "vqa_eval_predictions.jsonl",
    "metrics_output": "vqa_eval_metrics.json",
    "start_idx": 0,
    "limit": 0,
    "batch_size": 64,
    "eval_protocol": "exact_match",
}


NUMBER_WORDS: Dict[str, str] = {
    "zero": "0",
    "one": "1",
    "two": "2",
    "three": "3",
    "four": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "nine": "9",
    "ten": "10",
    "eleven": "11",
    "twelve": "12",
}

YES_SET = {"yes", "yeah", "yep", "true", "present", "exists"}
NO_SET = {"no", "nope", "false", "absent", "none"}

PAPER_TYPE_MAP: Dict[str, str] = {
    "object category": "Category",
    "object existence": "Presence",
    "object quantity": "Quantity",
    "object color": "Color",
    "object shape": "Shape",
    "object size": "Size",
    "object position": "Position",
    "object direction": "Direction",
    "scene type": "Scene",
    "image": "Scene",
    "rural or urban": "Scene",
    "reasoning": "Reasoning",
}

PAPER_TYPE_ORDER: List[str] = [
    "Category",
    "Presence",
    "Quantity",
    "Color",
    "Shape",
    "Size",
    "Position",
    "Direction",
    "Scene",
    "Reasoning",
]


def default_prompt(question: str, image_name: str, question_type: str) -> str:
    return (
        "You are an expert in remote sensing visual question answering.\n"
        "Answer the question using only the image.\n"
        "Return only the final short answer, with no explanation.\n"
        "If the answer is yes/no, return exactly Yes or No.\n"
        "If the answer is a count, return only the number.\n"
        f"Image: {image_name}\n"
        f"Question type: {question_type}\n"
        f"Question: {question}\n"
        "Answer:"
    )


def tokenize(text: str) -> List[str]:
    text = (text or "").strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text.split() if text else []


def apply_chat_template_with_thinking(
    processor: Any, messages: List[Dict[str, Any]], enable_thinking: bool
) -> str:
    return processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )


def slice_generated_tokens(generated_ids: Any, inputs: Dict[str, Any]) -> List[Any]:
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
    s = re.sub(r"^<\|im_start\|>\s*assistant\s*", "", s, flags=re.IGNORECASE)
    s = re.sub(r"^assistant\s*", "", s, flags=re.IGNORECASE)
    s = s.replace("<|im_end|>", "").strip()
    s = re.sub(r"^```(?:text|markdown|json)?\s*", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\s*```$", "", s)
    return s.strip()


def extract_short_answer(text: str) -> str:
    s = strip_output_wrappers(text)
    if not s:
        return ""

    s = s.replace("\r", "\n").strip()
    s = re.sub(r"^\s*answer\s*:\s*", "", s, flags=re.IGNORECASE)
    s = re.sub(r"^\s*final answer\s*:\s*", "", s, flags=re.IGNORECASE)
    lines = [line.strip() for line in s.split("\n") if line.strip()]
    s = lines[0] if lines else s
    s = re.split(r"(?<=[.!?])\s+", s, maxsplit=1)[0].strip()
    s = s.strip(" \"'`.,;:()[]{}")
    return s


def normalize_free_text(text: str) -> str:
    s = (text or "").strip().lower()
    s = re.sub(r"[_/]", " ", s)
    s = s.replace("&", " and ")
    s = re.sub(r"\bthe\b|\ba\b|\ban\b", " ", s)
    s = re.sub(r"[^a-z0-9\- ]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def normalize_yes_no(text: str) -> str:
    s = normalize_free_text(text)
    if s in YES_SET:
        return "yes"
    if s in NO_SET:
        return "no"
    if s.startswith("yes "):
        return "yes"
    if s.startswith("no "):
        return "no"
    return s


def normalize_count(text: str) -> str:
    s = normalize_free_text(text)
    if s in NUMBER_WORDS:
        return NUMBER_WORDS[s]
    parts = s.split()
    if parts and parts[0] in NUMBER_WORDS:
        return NUMBER_WORDS[parts[0]]
    m = re.search(r"\d+", s)
    if m:
        return m.group(0)
    return s


def normalize_answer(text: str, question_type: str) -> str:
    short = extract_short_answer(text)
    if question_type == "object existence":
        return normalize_yes_no(short)
    if question_type == "object quantity":
        return normalize_count(short)

    s = normalize_free_text(short)
    if s in NUMBER_WORDS:
        return NUMBER_WORDS[s]
    return s


def to_paper_type(question_type: str) -> str:
    return PAPER_TYPE_MAP.get(question_type, question_type)


def load_vlm(model: str) -> Tuple[Any, Any]:
    if torch is None or AutoModelForImageTextToText is None or AutoProcessor is None:
        raise RuntimeError(
            "Missing dependencies. Please install: pip install torch transformers accelerate pillow"
        )

    use_cuda = torch.cuda.is_available()
    dtype = torch.bfloat16 if use_cuda else torch.float32
    if use_cuda:
        try:
            torch.backends.cuda.matmul.fp32_precision = "tf32"
            torch.backends.cudnn.conv.fp32_precision = "tf32"
        except Exception:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.set_float32_matmul_precision("high")

    processor = AutoProcessor.from_pretrained(model, trust_remote_code=True)
    if hasattr(processor, "tokenizer") and processor.tokenizer is not None:
        processor.tokenizer.padding_side = "left"
        if processor.tokenizer.pad_token_id is None and processor.tokenizer.eos_token_id is not None:
            processor.tokenizer.pad_token_id = processor.tokenizer.eos_token_id

    vlm = AutoModelForImageTextToText.from_pretrained(
        model,
        dtype=dtype,
        device_map="auto",
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
) -> List[str]:
    if len(prompts) != len(images):
        raise ValueError(f"prompts/images length mismatch: {len(prompts)} vs {len(images)}")

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
    inputs = processor(text=text_inputs, images=images, padding=True, return_tensors="pt")
    inputs = {k: v.to(vlm.device) for k, v in inputs.items()}

    generation_kwargs: Dict[str, Any] = {
        "max_new_tokens": max_output_tokens,
        "do_sample": temperature > 0,
    }
    if temperature > 0:
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

    if debug_raw_output:
        for i, txt in enumerate(raw_texts):
            print(f"\n========== RAW_MODEL_OUTPUT_BEGIN[{i}] ==========")
            print(txt)
            print(f"=========== RAW_MODEL_OUTPUT_END[{i}] ===========\n")

    return [strip_output_wrappers(x) for x in raw_texts]


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def run_dataset_eval_mode(processor: Any, vlm: Any) -> None:
    eval_json = str(CONFIG["eval_json"])
    image_dir = str(CONFIG["image_dir"])
    pred_path = str(CONFIG["dataset_output"])
    start_idx = max(int(CONFIG.get("start_idx", 0)), 0)
    limit = int(CONFIG.get("limit", 0))
    temperature = float(CONFIG["temperature"])
    enable_thinking = bool(CONFIG.get("enable_thinking", False))
    max_output_tokens = int(CONFIG["max_output_tokens"])
    debug_raw_output = bool(CONFIG.get("DEBUG_RAW_OUTPUT", False))
    batch_size = max(int(CONFIG.get("batch_size", 1)), 1)

    with open(eval_json, "r", encoding="utf-8") as f:
        samples = json.load(f)
    if not isinstance(samples, list):
        raise ValueError("eval_json must be a list.")

    if start_idx >= len(samples):
        raise ValueError(f"start_idx={start_idx} out of range. total={len(samples)}")

    selected = samples[start_idx:]
    if limit > 0:
        selected = selected[:limit]
    total = len(selected)
    if total == 0:
        raise ValueError("No samples selected. Check start_idx/limit.")

    processed = 0
    missing_images = 0
    empty_outputs = 0
    inference_errors = 0
    batch_fallback_count = 0
    t0 = time.time()

    with open(pred_path, "w", encoding="utf-8") as wf:
        for chunk_start in range(0, total, batch_size):
            chunk = selected[chunk_start : chunk_start + batch_size]
            prepared: List[Dict[str, Any]] = []

            for item in chunk:
                image_id = str(item.get("image_id", ""))
                question = str(item.get("question", ""))
                ground_truth = str(item.get("ground_truth", ""))
                question_id = item.get("question_id", None)
                question_type = str(item.get("type", ""))
                dataset = str(item.get("dataset", ""))
                image_path = os.path.join(image_dir, image_id)
                if not os.path.isfile(image_path):
                    rec = {
                        "imgid": str(question_id),
                        "question_id": question_id,
                        "image_id": image_id,
                        "question": question,
                        "type": question_type,
                        "dataset": dataset,
                        "ground_truth": ground_truth,
                        "error": "image_not_found",
                    }
                    wf.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    missing_images += 1
                    processed += 1
                    continue

                with Image.open(image_path) as img_fp:
                    img = ImageOps.exif_transpose(img_fp).convert("RGB")
                prepared.append(
                    {
                        "imgid": str(question_id),
                        "question_id": question_id,
                        "image_id": image_id,
                        "question": question,
                        "type": question_type,
                        "dataset": dataset,
                        "ground_truth": ground_truth,
                        "prompt": default_prompt(
                            question=question,
                            image_name=image_id,
                            question_type=question_type,
                        ),
                        "img": img.copy(),
                    }
                )
                img.close()

            if not prepared:
                continue

            try:
                outputs = call_lvlm_batch_with_loaded(
                    processor=processor,
                    vlm=vlm,
                    prompts=[p["prompt"] for p in prepared],
                    images=[p["img"] for p in prepared],
                    temperature=temperature,
                    enable_thinking=enable_thinking,
                    max_output_tokens=max_output_tokens,
                    debug_raw_output=debug_raw_output,
                )
            except Exception as exc:
                batch_fallback_count += 1
                print(
                    f"Batch inference failed for chunk starting at {chunk_start}; "
                    f"falling back to per-sample retries. error={exc}"
                )
                outputs = []
                for p in prepared:
                    try:
                        output = call_lvlm_batch_with_loaded(
                            processor=processor,
                            vlm=vlm,
                            prompts=[p["prompt"]],
                            images=[p["img"]],
                            temperature=temperature,
                            enable_thinking=enable_thinking,
                            max_output_tokens=max_output_tokens,
                            debug_raw_output=debug_raw_output,
                        )[0]
                        outputs.append(output)
                    except Exception as exc2:
                        rec = {
                            "imgid": p["imgid"],
                            "question_id": p["question_id"],
                            "image_id": p["image_id"],
                            "question": p["question"],
                            "type": p["type"],
                            "dataset": p["dataset"],
                            "ground_truth": p["ground_truth"],
                            "error": f"inference_error: {exc2} (batch_error: {exc})",
                        }
                        wf.write(json.dumps(rec, ensure_ascii=False) + "\n")
                        inference_errors += 1
                        processed += 1
                        outputs.append("__ERROR_ALREADY_LOGGED__")

            for p, prediction in zip(prepared, outputs):
                if prediction == "__ERROR_ALREADY_LOGGED__":
                    continue
                prediction = (prediction or "").strip()
                if not prediction:
                    rec = {
                        "imgid": p["imgid"],
                        "question_id": p["question_id"],
                        "image_id": p["image_id"],
                        "question": p["question"],
                        "type": p["type"],
                        "dataset": p["dataset"],
                        "ground_truth": p["ground_truth"],
                        "error": "empty_model_output",
                    }
                    wf.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    empty_outputs += 1
                    processed += 1
                    continue

                pred_answer = extract_short_answer(prediction)
                pred_norm = normalize_answer(prediction, p["type"])
                gt_norm = normalize_answer(p["ground_truth"], p["type"])
                is_correct = pred_norm == gt_norm

                rec = {
                    "imgid": p["imgid"],
                    "question_id": p["question_id"],
                    "image_id": p["image_id"],
                    "question": p["question"],
                    "type": p["type"],
                    "dataset": p["dataset"],
                    "ground_truth": p["ground_truth"],
                    "ground_truth_normalized": gt_norm,
                    "prediction": prediction,
                    "prediction_answer": pred_answer,
                    "prediction_normalized": pred_norm,
                    "correct": is_correct,
                    "prediction_token_count": len(tokenize(prediction)),
                }
                wf.write(json.dumps(rec, ensure_ascii=False) + "\n")
                processed += 1

            for p in prepared:
                p["img"].close()

            if processed % 10 == 0 or processed == total:
                elapsed = time.time() - t0
                speed = processed / max(1e-6, elapsed)
                print(
                    f"[{processed}/{total}] processed, "
                    f"speed={speed:.3f} samples/s, batch_size={batch_size}"
                )

    summary = {
        "total_selected": total,
        "processed": processed,
        "missing_images": missing_images,
        "empty_outputs": empty_outputs,
        "inference_errors": inference_errors,
        "batch_fallback_count": batch_fallback_count,
        "predictions_file": pred_path,
        "elapsed_seconds": time.time() - t0,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def run_vqa_eval_mode() -> None:
    eval_json_path = Path(str(CONFIG["eval_json"])).resolve()
    preds_path = Path(str(CONFIG["dataset_output"])).resolve()
    metrics_path = Path(str(CONFIG["metrics_output"])).resolve()
    eval_protocol = str(CONFIG.get("eval_protocol", "exact_match")).strip()

    if not eval_json_path.is_file():
        raise FileNotFoundError(f"Missing eval_json: {eval_json_path}")
    if not preds_path.is_file():
        raise FileNotFoundError(f"Missing preds: {preds_path}")

    eval_rows = json.loads(eval_json_path.read_text(encoding="utf-8"))
    pred_rows = read_jsonl(preds_path)

    gt_by_id: Dict[str, Dict[str, str]] = {}
    for row in eval_rows:
        qid = str(row.get("question_id", "")).strip()
        if not qid:
            continue
        qtype = str(row.get("type", ""))
        ground_truth = str(row.get("ground_truth", ""))
        gt_by_id[qid] = {
            "type": qtype,
            "ground_truth": ground_truth,
            "ground_truth_normalized": normalize_answer(ground_truth, qtype),
        }

    by_type: Dict[str, Dict[str, int]] = defaultdict(lambda: {"total": 0, "correct": 0})
    by_paper_type: Dict[str, Dict[str, int]] = defaultdict(lambda: {"total": 0, "correct": 0})
    overall_total = 0
    overall_correct = 0
    if eval_protocol != "exact_match":
        raise ValueError(f"Unsupported eval_protocol: {eval_protocol}")

    for row in pred_rows:
        qid = str(row.get("question_id", "")).strip()
        pred_norm = str(row.get("prediction_normalized", "")).strip()
        if not qid or qid not in gt_by_id:
            continue
        qtype = gt_by_id[qid]["type"]
        paper_type = to_paper_type(qtype)
        gt_norm = gt_by_id[qid]["ground_truth_normalized"]
        is_correct = pred_norm == gt_norm

        overall_total += 1
        overall_correct += int(is_correct)
        by_type[qtype]["total"] += 1
        by_type[qtype]["correct"] += int(is_correct)
        by_paper_type[paper_type]["total"] += 1
        by_paper_type[paper_type]["correct"] += int(is_correct)

    metrics_by_type: Dict[str, Dict[str, Any]] = {}
    for qtype in sorted(by_type.keys()):
        total = by_type[qtype]["total"]
        correct = by_type[qtype]["correct"]
        metrics_by_type[qtype] = {
            "count": total,
            "correct": correct,
            "accuracy": (correct / total) if total > 0 else 0.0,
        }

    metrics_by_paper_type: Dict[str, Dict[str, Any]] = {}
    for qtype in PAPER_TYPE_ORDER:
        total = by_paper_type[qtype]["total"]
        correct = by_paper_type[qtype]["correct"]
        metrics_by_paper_type[qtype] = {
            "count": total,
            "correct": correct,
            "accuracy": (correct / total) if total > 0 else 0.0,
        }

    extra_types = sorted(k for k in by_paper_type.keys() if k not in PAPER_TYPE_ORDER)
    for qtype in extra_types:
        total = by_paper_type[qtype]["total"]
        correct = by_paper_type[qtype]["correct"]
        metrics_by_paper_type[qtype] = {
            "count": total,
            "correct": correct,
            "accuracy": (correct / total) if total > 0 else 0.0,
        }

    summary = {
        "accuracy": (overall_correct / overall_total) if overall_total > 0 else 0.0,
        "correct": overall_correct,
        "num_questions": overall_total,
        "num_questions_in_eval_json": len(gt_by_id),
        "by_type_raw": metrics_by_type,
        "by_type_paper_table3": metrics_by_paper_type,
        "eval_json": str(eval_json_path),
        "preds": str(preds_path),
        "metric_protocol": eval_protocol,
        "paper_alignment_note": (
            "OpenAI-based GPT judge was removed. This metric is local normalized exact match only, "
            "while keeping the paper-style 10-category grouping."
        ),
    }
    write_json(metrics_path, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def main() -> None:
    model = str(CONFIG["model"])
    print(f"Loading model: {model}")
    processor, vlm = load_vlm(model=model)
    run_dataset_eval_mode(processor, vlm)
    run_vqa_eval_mode()


if __name__ == "__main__":
    main()
