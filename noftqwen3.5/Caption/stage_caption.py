import json
import os
import re
import time
from typing import Any, Dict, List, Tuple

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
    "mode": "dataset_eval",
    "model": "Qwen/Qwen3.5-4B",
    "temperature": 0.0,
    "enable_thinking": False,
    "max_output_tokens": 192,
    "DEBUG_RAW_OUTPUT": False,
    "eval_json": "VRSBench_EVAL_Cap.json",
    "image_dir": "Images_val",
    "dataset_output": "caption_eval_predictions.jsonl",
    "start_idx": 0,
    "limit": 0,
    "batch_size": 64,
}


def default_prompt(question: str, image_name: str) -> str:
    return (
        "You are an expert in remote sensing image captioning.\n"
        "Describe the image in one detailed paragraph.\n"
        "Focus on visible objects, scene layout, relative positions, counts when reliable, "
        "and major land-use or infrastructure clues.\n"
        "Do not output JSON, Markdown, or bullet points.\n"
        f"Image: {image_name}\n"
        f"Task: {question}\n"
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
                question = str(item.get("question", "Describe the image in detail"))
                ground_truth = str(item.get("ground_truth", ""))
                question_id = item.get("question_id", None)
                image_path = os.path.join(image_dir, image_id)
                if not os.path.isfile(image_path):
                    rec = {
                        "imgid": str(question_id),
                        "question_id": question_id,
                        "image_id": image_id,
                        "question": question,
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
                        "ground_truth": ground_truth,
                        "prompt": default_prompt(question=question, image_name=image_id),
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
                        "ground_truth": p["ground_truth"],
                        "error": "empty_model_output",
                    }
                    wf.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    empty_outputs += 1
                    processed += 1
                    continue

                rec = {
                    "imgid": p["imgid"],
                    "question_id": p["question_id"],
                    "image_id": p["image_id"],
                    "question": p["question"],
                    "ground_truth": p["ground_truth"],
                    "prediction": prediction,
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
