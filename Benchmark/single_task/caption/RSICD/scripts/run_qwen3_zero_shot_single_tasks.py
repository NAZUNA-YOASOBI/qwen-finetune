from __future__ import annotations

import argparse
import json
import re
import tarfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_CAPTION_PROMPT = (
    "Write one concise English caption for this remote sensing image in 8 to 15 words. "
    "Mention only the main scene and key objects."
)

DEFAULT_YES_NO_PROMPT = (
    "Answer the question about this remote sensing image with only one word: yes or no.\n"
    "Question: {question}\n"
    "Answer:"
)

DEFAULT_RURAL_URBAN_PROMPT = (
    "Answer the question about this remote sensing image with only one word: rural or urban.\n"
    "Question: {question}\n"
    "Answer:"
)


@dataclass(frozen=True)
class CaptionDatasetSpec:
    name: str
    dataset_json: Path
    image_dir: Path


@dataclass(frozen=True)
class VQADatasetSpec:
    name: str
    questions_json: Path
    answers_json: Path
    images_json: Path
    image_archive: Path
    image_extract_dir: Path
    archive_type: str
    archive_prefix: str
    allowed_qtypes: tuple[str, ...]


class Qwen3VLRunner:
    """原生 Qwen3-VL 单图推理封装。"""

    def __init__(
        self,
        *,
        model_dir: Path,
        device_map: str,
        dtype: str,
        seed: int | None,
    ) -> None:
        try:
            import torch
            from transformers import AutoProcessor, Qwen3VLForConditionalGeneration  # type: ignore
        except Exception as e:  # noqa: BLE001
            raise RuntimeError("Missing dependency: transformers/torch") from e

        from ftqwen.device import assert_model_on_cuda, require_cuda
        from ftqwen.qwen_dinov3 import maybe_set_generation_seed

        require_cuda()
        maybe_set_generation_seed(seed)

        self.torch = torch
        self.model_dir = Path(model_dir)
        self.device_map = str(device_map)
        self.dtype = str(dtype)

        self.processor = AutoProcessor.from_pretrained(str(self.model_dir))
        try:
            self.processor.tokenizer.padding_side = "left"
        except Exception:
            pass

        torch_dtype = self._torch_dtype_from_str(self.dtype)
        try:
            self.model = Qwen3VLForConditionalGeneration.from_pretrained(
                str(self.model_dir),
                dtype=torch_dtype,
                device_map=self.device_map,
            )
        except TypeError:
            self.model = Qwen3VLForConditionalGeneration.from_pretrained(
                str(self.model_dir),
                torch_dtype=torch_dtype,
                device_map=self.device_map,
            )
        self.model.eval()
        assert_model_on_cuda(self.model)

    def _torch_dtype_from_str(self, dtype: str):
        dtype = str(dtype).lower().strip()
        if dtype == "auto":
            return self.torch.float16 if self.torch.cuda.is_available() else self.torch.float32
        if dtype in {"fp16", "float16"}:
            return self.torch.float16
        if dtype in {"bf16", "bfloat16"}:
            return self.torch.bfloat16
        if dtype in {"fp32", "float32"}:
            return self.torch.float32
        return self.torch.float32

    def generate_batch(
        self,
        *,
        image_paths: list[Path],
        prompts: list[str],
        max_new_tokens: int,
        do_sample: bool | None,
        temperature: float | None,
        top_p: float | None,
        top_k: int | None,
        num_beams: int | None,
        repetition_penalty: float | None,
    ) -> list[str]:
        from ftqwen.qwen_dinov3 import build_generate_kwargs

        if not image_paths:
            return []
        if len(image_paths) != len(prompts):
            raise ValueError("image_paths and prompts must have the same length")

        conversations: list[list[dict[str, Any]]] = []
        for image_path, prompt in zip(image_paths, prompts):
            conversations.append(
                [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": str(image_path)},
                            {"type": "text", "text": str(prompt)},
                        ],
                    }
                ]
            )

        inputs = self.processor.apply_chat_template(
            conversations,
            tokenize=True,
            add_generation_prompt=True,
            padding=True,
            return_dict=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.model.device)

        gen_cfg = build_generate_kwargs(
            max_new_tokens=int(max_new_tokens),
            do_sample=(bool(do_sample) if do_sample is not None else None),
            temperature=(float(temperature) if do_sample is True and temperature is not None else None),
            top_p=(float(top_p) if do_sample is True and top_p is not None else None),
            top_k=(int(top_k) if do_sample is True and top_k is not None else None),
            num_beams=(int(num_beams) if num_beams is not None and int(num_beams) > 1 else None),
            repetition_penalty=(
                float(repetition_penalty)
                if repetition_penalty is not None and float(repetition_penalty) != 1.0
                else None
            ),
        )

        with self.torch.no_grad():
            generated_ids = self.model.generate(**inputs, **gen_cfg.gen_kwargs)

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
        texts = self.processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        return [str(text).strip() for text in texts]


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def simbench_root() -> Path:
    return project_root() / "RSGPT-Simbench"


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def append_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def build_caption_specs() -> dict[str, CaptionDatasetSpec]:
    root = simbench_root()
    return {
        "ucm_captions": CaptionDatasetSpec(
            name="ucm_captions",
            dataset_json=root / "UCM-captions" / "dataset.json",
            image_dir=root / "UCM-captions" / "imgs",
        ),
        "sydney_captions": CaptionDatasetSpec(
            name="sydney_captions",
            dataset_json=root / "Sydney-captions" / "dataset.json",
            image_dir=root / "Sydney-captions" / "imgs",
        ),
        "rsicd": CaptionDatasetSpec(
            name="rsicd",
            dataset_json=root / "RSICD" / "dataset_rsicd.json",
            image_dir=root / "RSICD" / "RSICD_images",
        ),
    }


def build_vqa_specs() -> dict[str, VQADatasetSpec]:
    root = simbench_root()
    hr_raw = root / "RSVQA-HR" / "raw"
    lr_raw = root / "RSVQA-LR" / "raw"
    return {
        "rsvqa_hr_test1": VQADatasetSpec(
            name="rsvqa_hr_test1",
            questions_json=hr_raw / "USGS_split_test_questions.json",
            answers_json=hr_raw / "USGS_split_test_answers.json",
            images_json=hr_raw / "USGS_split_test_images.json",
            image_archive=hr_raw / "Images.tar",
            image_extract_dir=root / "RSVQA-HR" / "images",
            archive_type="tar",
            archive_prefix="Data",
            allowed_qtypes=("presence", "comp"),
        ),
        "rsvqa_hr_test2": VQADatasetSpec(
            name="rsvqa_hr_test2",
            questions_json=hr_raw / "USGS_split_test_phili_questions.json",
            answers_json=hr_raw / "USGS_split_test_phili_answers.json",
            images_json=hr_raw / "USGS_split_test_phili_images.json",
            image_archive=hr_raw / "Images.tar",
            image_extract_dir=root / "RSVQA-HR" / "images",
            archive_type="tar",
            archive_prefix="Data",
            allowed_qtypes=("presence", "comp"),
        ),
        "rsvqa_lr_test": VQADatasetSpec(
            name="rsvqa_lr_test",
            questions_json=lr_raw / "LR_split_test_questions.json",
            answers_json=lr_raw / "LR_split_test_answers.json",
            images_json=lr_raw / "LR_split_test_images.json",
            image_archive=lr_raw / "Images_LR.zip",
            image_extract_dir=root / "RSVQA-LR" / "images",
            archive_type="zip",
            archive_prefix="Images_LR",
            allowed_qtypes=("presence", "comp", "rural_urban"),
        ),
    }


def load_caption_samples(spec: CaptionDatasetSpec, *, max_samples: int) -> list[dict[str, Any]]:
    data = read_json(spec.dataset_json)
    rows: list[dict[str, Any]] = []
    for item in data["images"]:
        if str(item.get("split", "")).strip().lower() != "test":
            continue
        filename = str(item["filename"])
        image_path = spec.image_dir / filename
        refs = [str(sent.get("raw", "")).strip() for sent in item.get("sentences", []) if str(sent.get("raw", "")).strip()]
        rows.append(
            {
                "dataset": spec.name,
                "task": "caption",
                "sample_id": str(item.get("imgid", filename)),
                "filename": filename,
                "image_path": image_path,
                "refs": refs,
            }
        )
    rows = sorted(rows, key=lambda x: str(x["filename"]))
    if int(max_samples) > 0:
        rows = rows[: int(max_samples)]
    return rows


def load_vqa_samples(spec: VQADatasetSpec, *, max_samples_per_type: int) -> list[dict[str, Any]]:
    questions = [item for item in read_json(spec.questions_json)["questions"] if item.get("active") is True]
    answers = [item for item in read_json(spec.answers_json)["answers"] if item.get("active") is True]
    images = [item for item in read_json(spec.images_json)["images"] if item.get("active") is True]

    answers_by_qid = {int(item["question_id"]): item for item in answers}
    images_by_id = {int(item["id"]): item for item in images}

    by_type: dict[str, list[dict[str, Any]]] = {qtype: [] for qtype in spec.allowed_qtypes}
    for item in questions:
        qtype = str(item.get("type", "")).strip()
        if qtype not in spec.allowed_qtypes:
            continue
        qid = int(item["id"])
        img_id = int(item["img_id"])
        answer = answers_by_qid.get(qid)
        image = images_by_id.get(img_id)
        if answer is None or image is None:
            continue
        by_type[qtype].append(
            {
                "dataset": spec.name,
                "task": "vqa",
                "question_id": qid,
                "image_id": img_id,
                "question_type": qtype,
                "paper_question_type": paper_question_type(qtype),
                "question": str(item.get("question", "")).strip(),
                "answer": str(answer.get("answer", "")).strip(),
                "original_name": str(image.get("original_name", "")).strip(),
            }
        )

    rows: list[dict[str, Any]] = []
    for qtype in spec.allowed_qtypes:
        cur = sorted(by_type[qtype], key=lambda x: int(x["question_id"]))
        if int(max_samples_per_type) > 0:
            cur = cur[: int(max_samples_per_type)]
        rows.extend(cur)
    return rows


def paper_question_type(qtype: str) -> str:
    qtype = str(qtype).strip()
    if qtype == "comp":
        return "comparison"
    if qtype == "rural_urban":
        return "rural/urban"
    return qtype


def normalize_vqa_answer(text: str, *, qtype: str) -> str:
    value = str(text).strip().lower()
    value = re.sub(r"\s+", " ", value)
    if qtype in {"presence", "comp"}:
        match = re.search(r"\b(yes|no)\b", value)
        if match:
            return str(match.group(1))
    if qtype == "rural_urban":
        match = re.search(r"\b(rural|urban)\b", value)
        if match:
            return str(match.group(1))
    value = value.strip(" .,!?:;\"'`()[]{}")
    return value


def resolve_vqa_image_path(spec: VQADatasetSpec, *, image_id: int, extract_mode: str) -> Path:
    extract_dir = Path(spec.image_extract_dir)
    extract_dir.mkdir(parents=True, exist_ok=True)

    for ext in ("tif", "png", "jpg", "jpeg"):
        candidate = extract_dir / f"{image_id}.{ext}"
        if candidate.is_file():
            return candidate

    if str(extract_mode) == "none":
        raise FileNotFoundError(f"Missing extracted image for image_id={image_id}: {extract_dir}")

    if spec.archive_type == "tar":
        return extract_one_from_tar(spec.image_archive, extract_dir, image_id=image_id, prefix=spec.archive_prefix)
    if spec.archive_type == "zip":
        return extract_one_from_zip(spec.image_archive, extract_dir, image_id=image_id, prefix=spec.archive_prefix)
    raise ValueError(f"Unsupported archive type: {spec.archive_type}")


def extract_one_from_tar(archive_path: Path, extract_dir: Path, *, image_id: int, prefix: str) -> Path:
    preferred_names = [f"{prefix}/{image_id}.tif", f"{prefix}/{image_id}.png"]
    with tarfile.open(archive_path, "r") as tf:
        for name in preferred_names:
            try:
                member = tf.getmember(name)
            except KeyError:
                continue
            out_path = extract_dir / Path(member.name).name
            if out_path.is_file():
                return out_path
            src = tf.extractfile(member)
            if src is None:
                continue
            out_path.write_bytes(src.read())
            return out_path
    raise FileNotFoundError(f"Image id {image_id} not found in tar archive: {archive_path}")


def extract_one_from_zip(archive_path: Path, extract_dir: Path, *, image_id: int, prefix: str) -> Path:
    preferred_names = [f"{prefix}/{image_id}.tif", f"{prefix}/{image_id}.png"]
    with zipfile.ZipFile(archive_path, "r") as zf:
        for name in preferred_names:
            try:
                data = zf.read(name)
            except KeyError:
                continue
            out_path = extract_dir / Path(name).name
            if out_path.is_file():
                return out_path
            out_path.write_bytes(data)
            return out_path
    raise FileNotFoundError(f"Image id {image_id} not found in zip archive: {archive_path}")


def build_caption_prompt(args) -> str:
    return str(args.caption_prompt)


def build_vqa_prompt(args, *, qtype: str, question: str) -> str:
    template = args.vqa_yesno_prompt if qtype in {"presence", "comp"} else args.vqa_rural_urban_prompt
    return str(template).format(question=str(question).strip())


def parse_datasets(arg: str) -> list[str]:
    raw = [x.strip() for x in str(arg).split(",") if x.strip()]
    if not raw or raw == ["all"]:
        return [
            "ucm_captions",
            "sydney_captions",
            "rsicd",
            "rsvqa_hr_test1",
            "rsvqa_hr_test2",
            "rsvqa_lr_test",
        ]
    return raw


def ensure_ftqwen_on_path() -> None:
    import sys

    src = project_root() / "qwen-finetune" / "VRSBench" / "src"
    sys.path.insert(0, str(src))


def run_caption_dataset(
    *,
    runner: Qwen3VLRunner,
    spec: CaptionDatasetSpec,
    args,
    out_path: Path,
) -> dict[str, Any]:
    rows = load_caption_samples(spec, max_samples=int(args.caption_samples))
    prompt = build_caption_prompt(args)
    if not rows:
        return {"dataset": spec.name, "task": "caption", "num_rows": 0, "prompt": prompt, "sample_predictions": []}

    done_ids = {str(row.get("sample_id", "")) for row in read_jsonl(out_path)}
    pending = [row for row in rows if str(row["sample_id"]) not in done_ids]

    import gc

    batch_size = max(1, int(args.batch_size))
    cur_bs = int(batch_size)
    sample_predictions: list[dict[str, Any]] = []
    num_done = len(done_ids)
    start = 0
    while start < len(pending):
        chunk = pending[start : start + cur_bs]
        image_paths = [Path(item["image_path"]) for item in chunk]
        prompts = [prompt] * len(chunk)
        try:
            texts = runner.generate_batch(
                image_paths=image_paths,
                prompts=prompts,
                max_new_tokens=int(args.caption_max_new_tokens),
                do_sample=args.do_sample,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                num_beams=args.num_beams,
                repetition_penalty=args.repetition_penalty,
            )
        except runner.torch.cuda.OutOfMemoryError:
            runner.torch.cuda.empty_cache()
            gc.collect()
            if cur_bs <= 1:
                raise
            cur_bs = max(1, cur_bs // 2)
            print(f"[OOM] {spec.name}: reduce batch_size to {cur_bs}", flush=True)
            continue
        batch_rows: list[dict[str, Any]] = []
        for item, text in zip(chunk, texts):
            row = {
                "dataset": spec.name,
                "task": "caption",
                "sample_id": str(item["sample_id"]),
                "filename": str(item["filename"]),
                "image_path": str(item["image_path"]),
                "prompt": prompt,
                "prediction": str(text),
                "refs": list(item["refs"]),
            }
            batch_rows.append(row)
            if len(sample_predictions) < 5:
                sample_predictions.append(
                    {
                        "filename": row["filename"],
                        "prediction": row["prediction"],
                        "first_ref": row["refs"][0] if row["refs"] else "",
                    }
                )
        append_jsonl(out_path, batch_rows)
        num_done += len(batch_rows)
        start += len(batch_rows)
        print(f"[PROGRESS] {spec.name}: {num_done}/{len(rows)}", flush=True)

    final_rows = read_jsonl(out_path)
    final_rows = [row for row in final_rows if str(row.get("dataset", "")) == spec.name]
    if not sample_predictions:
        for row in final_rows[:5]:
            sample_predictions.append(
                {
                    "filename": row["filename"],
                    "prediction": row["prediction"],
                    "first_ref": row["refs"][0] if row.get("refs") else "",
                }
            )

    summary = {
        "dataset": spec.name,
        "task": "caption",
        "num_rows": len(final_rows),
        "prompt": prompt,
        "sample_predictions": sample_predictions,
        "num_pending_before_run": len(pending),
    }
    return summary


def run_vqa_dataset(
    *,
    runner: Qwen3VLRunner,
    spec: VQADatasetSpec,
    args,
    out_path: Path,
) -> dict[str, Any]:
    rows = load_vqa_samples(spec, max_samples_per_type=int(args.vqa_samples_per_type))
    if not rows:
        return {"dataset": spec.name, "task": "vqa", "num_rows": 0, "per_type": {}, "sample_predictions": []}

    done_qids = {int(row.get("question_id", -1)) for row in read_jsonl(out_path)}
    pending = [row for row in rows if int(row["question_id"]) not in done_qids]

    import gc

    batch_size = max(1, int(args.batch_size))
    cur_bs = int(batch_size)
    sample_predictions: list[dict[str, Any]] = []
    num_done = len(done_qids)
    start = 0
    while start < len(pending):
        chunk = pending[start : start + cur_bs]
        image_paths: list[Path] = []
        prompts: list[str] = []
        for item in chunk:
            image_paths.append(
                resolve_vqa_image_path(
                    spec,
                    image_id=int(item["image_id"]),
                    extract_mode=str(args.extract_vqa_images),
                )
            )
            prompts.append(build_vqa_prompt(args, qtype=str(item["question_type"]), question=str(item["question"])))

        try:
            texts = runner.generate_batch(
                image_paths=image_paths,
                prompts=prompts,
                max_new_tokens=int(args.vqa_max_new_tokens),
                do_sample=args.do_sample,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                num_beams=args.num_beams,
                repetition_penalty=args.repetition_penalty,
            )
        except runner.torch.cuda.OutOfMemoryError:
            runner.torch.cuda.empty_cache()
            gc.collect()
            if cur_bs <= 1:
                raise
            cur_bs = max(1, cur_bs // 2)
            print(f"[OOM] {spec.name}: reduce batch_size to {cur_bs}", flush=True)
            continue

        batch_rows: list[dict[str, Any]] = []
        for item, image_path, prompt, text in zip(chunk, image_paths, prompts, texts):
            pred_norm = normalize_vqa_answer(str(text), qtype=str(item["question_type"]))
            gt_norm = normalize_vqa_answer(str(item["answer"]), qtype=str(item["question_type"]))
            row = {
                "dataset": spec.name,
                "task": "vqa",
                "question_id": int(item["question_id"]),
                "image_id": int(item["image_id"]),
                "original_name": str(item["original_name"]),
                "image_path": str(image_path),
                "question_type": str(item["question_type"]),
                "paper_question_type": str(item["paper_question_type"]),
                "question": str(item["question"]),
                "prompt": str(prompt),
                "prediction": str(text),
                "prediction_normalized": pred_norm,
                "answer": str(item["answer"]),
                "answer_normalized": gt_norm,
                "correct_normalized": bool(pred_norm == gt_norm),
            }
            batch_rows.append(row)
            if len(sample_predictions) < 8:
                sample_predictions.append(
                    {
                        "question_id": row["question_id"],
                        "question_type": row["question_type"],
                        "question": row["question"],
                        "prediction": row["prediction"],
                        "prediction_normalized": row["prediction_normalized"],
                        "answer": row["answer"],
                        "correct_normalized": row["correct_normalized"],
                    }
                )
        append_jsonl(out_path, batch_rows)
        num_done += len(batch_rows)
        start += len(batch_rows)
        print(f"[PROGRESS] {spec.name}: {num_done}/{len(rows)}", flush=True)

    final_rows = read_jsonl(out_path)
    final_rows = [row for row in final_rows if str(row.get("dataset", "")) == spec.name]
    per_type: dict[str, dict[str, Any]] = {}
    for qtype in spec.allowed_qtypes:
        cur = [row for row in final_rows if str(row["question_type"]) == qtype]
        if not cur:
            continue
        correct = sum(1 for row in cur if bool(row["correct_normalized"]))
        per_type[qtype] = {
            "paper_question_type": paper_question_type(qtype),
            "num_rows": len(cur),
            "accuracy_normalized": float(correct) / float(len(cur)),
        }

    if not sample_predictions:
        for row in final_rows[:8]:
            sample_predictions.append(
                {
                    "question_id": row["question_id"],
                    "question_type": row["question_type"],
                    "question": row["question"],
                    "prediction": row["prediction"],
                    "prediction_normalized": row["prediction_normalized"],
                    "answer": row["answer"],
                    "correct_normalized": row["correct_normalized"],
                }
            )

    summary = {
        "dataset": spec.name,
        "task": "vqa",
        "num_rows": len(final_rows),
        "per_type": per_type,
        "sample_predictions": sample_predictions,
        "num_pending_before_run": len(pending),
    }
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run zero-shot Qwen3-VL-8B-Instruct on RSGPT single-task benchmarks.")
    parser.add_argument("--datasets", type=str, default="all", help="Comma-separated dataset keys, or all.")
    parser.add_argument("--model-dir", type=str, default="VRSBench/models/Qwen3-VL-8B-Instruct")
    parser.add_argument("--output-dir", type=str, default="RSGPT-Simbench/outputs/qwen3_zero_shot_stage1_smoke")
    parser.add_argument("--caption-samples", type=int, default=2, help="Per caption dataset; 0 means full test split.")
    parser.add_argument("--vqa-samples-per-type", type=int, default=2, help="Per allowed question type; 0 means full test split.")
    parser.add_argument("--extract-vqa-images", type=str, default="sample", choices=["sample", "none"])
    parser.add_argument("--caption-prompt", type=str, default=DEFAULT_CAPTION_PROMPT)
    parser.add_argument("--vqa-yesno-prompt", type=str, default=DEFAULT_YES_NO_PROMPT)
    parser.add_argument("--vqa-rural-urban-prompt", type=str, default=DEFAULT_RURAL_URBAN_PROMPT)
    parser.add_argument("--caption-max-new-tokens", type=int, default=48)
    parser.add_argument("--vqa-max-new-tokens", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--device-map", type=str, default="auto")
    parser.add_argument("--dtype", type=str, default="fp16", choices=["auto", "fp16", "bf16", "fp32"])
    parser.add_argument("--do-sample", dest="do_sample", action="store_true", default=None)
    parser.add_argument("--no-sample", dest="do_sample", action="store_false")
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--num-beams", type=int, default=None)
    parser.add_argument("--repetition-penalty", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_ftqwen_on_path()

    selected = parse_datasets(args.datasets)
    caption_specs = build_caption_specs()
    vqa_specs = build_vqa_specs()

    unknown = [name for name in selected if name not in caption_specs and name not in vqa_specs]
    if unknown:
        raise ValueError(f"Unknown datasets: {unknown}")

    runner = Qwen3VLRunner(
        model_dir=(project_root() / args.model_dir).resolve() if not Path(args.model_dir).is_absolute() else Path(args.model_dir),
        device_map=str(args.device_map),
        dtype=str(args.dtype),
        seed=(int(args.seed) if args.seed is not None and int(args.seed) > 0 else None),
    )

    out_dir = (project_root() / args.output_dir).resolve() if not Path(args.output_dir).is_absolute() else Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    combined_summary: dict[str, Any] = {
        "model_dir": str(Path(args.model_dir)),
        "datasets": selected,
        "caption_prompt": str(args.caption_prompt),
        "vqa_yesno_prompt": str(args.vqa_yesno_prompt),
        "vqa_rural_urban_prompt": str(args.vqa_rural_urban_prompt),
        "caption_samples": int(args.caption_samples),
        "vqa_samples_per_type": int(args.vqa_samples_per_type),
        "caption_max_new_tokens": int(args.caption_max_new_tokens),
        "vqa_max_new_tokens": int(args.vqa_max_new_tokens),
        "batch_size": int(args.batch_size),
        "dtype": str(args.dtype),
        "do_sample": args.do_sample,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "num_beams": args.num_beams,
        "repetition_penalty": args.repetition_penalty,
        "seed": args.seed,
        "decode_strategy": "generation_config_default" if args.do_sample is None and args.temperature is None and args.top_p is None and args.top_k is None and args.num_beams is None and args.repetition_penalty is None else "cli_override",
        "results": {},
    }

    for name in selected:
        dataset_out = out_dir / f"{name}.jsonl"
        if name in caption_specs:
            summary = run_caption_dataset(runner=runner, spec=caption_specs[name], args=args, out_path=dataset_out)
        else:
            summary = run_vqa_dataset(runner=runner, spec=vqa_specs[name], args=args, out_path=dataset_out)
        combined_summary["results"][name] = summary
        print(f"[OK] {name}: now has {summary['num_rows']} rows in {dataset_out.resolve()}", flush=True)
        write_json(out_dir / "summary.json", combined_summary)

    write_json(out_dir / "summary.json", combined_summary)
    print(f"[OK] Wrote summary: {(out_dir / 'summary.json').resolve()}", flush=True)


if __name__ == "__main__":
    main()
