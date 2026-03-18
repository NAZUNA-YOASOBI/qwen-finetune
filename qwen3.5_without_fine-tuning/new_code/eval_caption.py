import contextlib
import io
import json
import os
import re
import shutil
import time
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


BASE_DIR = Path(__file__).resolve().parent


CONFIG: Dict[str, Any] = {
    "model": "Qwen/Qwen3.5-4B",
    "temperature": 0.0,
    "enable_thinking": False,
    "max_output_tokens": 96,
    "DEBUG_RAW_OUTPUT": False,
    "eval_json": "data/VRSBench_EVAL_Cap.json",
    "image_dir": "data/Images_val",
    "dataset_output": "caption_eval_predictions.jsonl",
    "metrics_output": "caption_eval_metrics.json",
    "start_idx": 0,
    "limit": 0,
    "batch_size": 64,
}


def resolve_config_path(path_str: str) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    return BASE_DIR / path


def default_prompt(question: str, image_name: str) -> str:
    return (
        "You are an expert in remote sensing image captioning.\n"
        "Describe the image in 3 to 5 sentences, around 50 words total.\n"
        "Focus on visible objects, scene layout, relative positions, counts when reliable, "
        "and major land-use or infrastructure clues.\n"
        "Keep the caption concise and close to the reference length.\n"
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
    eval_json = str(resolve_config_path(str(CONFIG["eval_json"])))
    image_dir = str(resolve_config_path(str(CONFIG["image_dir"])))
    pred_path = str(resolve_config_path(str(CONFIG["dataset_output"])))
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


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


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


def build_refs_from_eval_json(path: Path) -> Dict[str, List[str]]:
    data = read_json(path)
    if not isinstance(data, list):
        raise ValueError(f"eval_json must be a list: {path}")
    refs: Dict[str, List[str]] = {}
    for item in data:
        question_id = item.get("question_id", None)
        ground_truth = str(item.get("ground_truth", "")).strip()
        if question_id is None or not ground_truth:
            continue
        refs[str(question_id)] = [ground_truth]
    return refs


def _to_coco_dict_raw(refs: Dict[str, List[str]], preds: Dict[str, str]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    gts: Dict[str, Any] = {}
    res: Dict[str, Any] = {}
    for img_id, ref_list in refs.items():
        if img_id not in preds:
            raise KeyError(f"Missing prediction for imgid={img_id}")
        gts[img_id] = [{"caption": str(r)} for r in ref_list]
        res[img_id] = [{"caption": str(preds[img_id])}]
    return gts, res


def _tokenize(gts_raw: Dict[str, Any], res_raw: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer  # type: ignore

    with contextlib.redirect_stdout(io.StringIO()):
        tokenizer = PTBTokenizer()
        gts = tokenizer.tokenize(gts_raw)
        res = tokenizer.tokenize(res_raw)
    return gts, res


def compute_caption_metrics(refs: Dict[str, List[str]], preds: Dict[str, str]) -> Dict[str, Optional[float]]:
    from pycocoevalcap.bleu.bleu import Bleu  # type: ignore
    from pycocoevalcap.cider.cider import Cider  # type: ignore
    from pycocoevalcap.meteor.meteor import Meteor  # type: ignore
    from pycocoevalcap.rouge.rouge import Rouge  # type: ignore

    gts_raw, res_raw = _to_coco_dict_raw(refs, preds)
    gts, res = _tokenize(gts_raw, res_raw)

    with contextlib.redirect_stdout(io.StringIO()):
        bleu_scores, _ = Bleu(4).compute_score(gts, res)
        rouge_score, _ = Rouge().compute_score(gts, res)
        cider_score, _ = Cider().compute_score(gts, res)

    meteor_score: Optional[float] = None
    if shutil.which("java") is not None:
        with contextlib.redirect_stdout(io.StringIO()):
            meteor = Meteor()
            meteor_raw, _ = meteor.compute_score(gts, res)
        meteor_score = float(meteor_raw)

    return {
        "BLEU-1": float(bleu_scores[0]),
        "BLEU-2": float(bleu_scores[1]),
        "BLEU-3": float(bleu_scores[2]),
        "BLEU-4": float(bleu_scores[3]),
        "METEOR": meteor_score,
        "ROUGE_L": float(rouge_score),
        "CIDEr": float(cider_score),
    }


def compute_avg_len_words(preds: Dict[str, str], imgids: List[str]) -> float:
    total = 0
    for imgid in imgids:
        total += len(str(preds[imgid]).strip().split())
    return float(total) / float(max(1, len(imgids)))


def run_caption_eval_mode() -> None:
    eval_json_path = resolve_config_path(str(CONFIG["eval_json"])).resolve()
    preds_path = resolve_config_path(str(CONFIG["dataset_output"])).resolve()
    metrics_path = resolve_config_path(str(CONFIG["metrics_output"])).resolve()

    if shutil.which("java") is None:
        py_bin = str(Path(__import__("sys").executable).resolve().parent)
        os.environ["PATH"] = py_bin + os.pathsep + os.environ.get("PATH", "")

    if not eval_json_path.is_file():
        raise FileNotFoundError(f"Missing eval_json: {eval_json_path}")
    if not preds_path.is_file():
        raise FileNotFoundError(f"Missing preds: {preds_path}")

    refs = build_refs_from_eval_json(eval_json_path)
    pred_rows = read_jsonl(preds_path)
    preds: Dict[str, str] = {}
    for row in pred_rows:
        imgid = str(row.get("imgid", "")).strip()
        pred = str(row.get("prediction", "")).strip()
        if imgid and pred:
            preds[imgid] = pred

    all_ids = sorted(refs.keys(), key=lambda x: int(x))
    refs = {k: refs[k] for k in all_ids}
    preds = {k: preds[k] for k in all_ids if k in preds}

    metrics = compute_caption_metrics(refs, preds)
    avg_len = compute_avg_len_words(preds, list(refs.keys()))
    summary = {
        "BLEU-1": metrics["BLEU-1"],
        "BLEU-2": metrics["BLEU-2"],
        "BLEU-3": metrics["BLEU-3"],
        "BLEU-4": metrics["BLEU-4"],
        "METEOR": metrics["METEOR"],
        "ROUGE_L": metrics["ROUGE_L"],
        "CIDEr": metrics["CIDEr"],
        "Avg_L": float(avg_len),
        "num_images": len(refs),
        "metrics_x100": {k: (float(v) * 100.0 if v is not None else None) for k, v in metrics.items()},
        "eval_json": str(eval_json_path),
        "preds": str(preds_path),
        "warnings": [] if metrics["METEOR"] is not None else ["METEOR requires java in PATH."],
    }
    write_json(metrics_path, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def main() -> None:
    model = str(CONFIG["model"])
    print(f"Loading model: {model}")
    processor, vlm = load_vlm(model=model)
    run_dataset_eval_mode(processor, vlm)
    run_caption_eval_mode()


if __name__ == "__main__":
    main()
