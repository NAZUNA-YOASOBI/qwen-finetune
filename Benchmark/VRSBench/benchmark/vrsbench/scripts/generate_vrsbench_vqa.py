from __future__ import annotations

import argparse
import gc
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm


SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[4]
VRSBENCH_DIR = SCRIPT_PATH.parents[3]
VRSBENCH_SRC = PROJECT_ROOT / "VRSBench" / "src"
if str(VRSBENCH_SRC) not in sys.path:
    sys.path.insert(0, str(VRSBENCH_SRC))

from ftqwen3.dinov3_captioner import DinoV3Captioner
from ftqwen3.jsonl import append_jsonl, read_json, read_jsonl, write_json
from ftqwen3.qwen3_vl_native_captioner import Qwen3VLNativeCaptioner
from ftqwen3.qwen_dinov3 import (
    DinoResizeConfig,
    assert_dino_runtime_matches_merger,
    assert_path_metadata_matches,
    read_merger_run_meta,
    resolve_dino_resize_config,
)
from ftqwen35.qwen3_5_captioner import Qwen35Captioner


NUMBER_WORDS: dict[str, str] = {
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


@dataclass(frozen=True)
class ModelPreset:
    name: str
    model_family: str
    qwen_model_dir: Path
    dinov3_dir: Path | None
    merger_ckpt: Path | None
    lora_dir: Path | None
    image_size: int | None
    smart_resize_min_pixels: int | None
    smart_resize_max_pixels: int | None
    dtype: str


class CaptionRunner:
    def __init__(self, *, preset: ModelPreset, args: argparse.Namespace, max_new_tokens: int) -> None:
        self.preset = preset
        self.model_family = str(preset.model_family)
        self.dtype = str(args.dtype if args.dtype else preset.dtype)
        self.device_map = str(args.device_map)
        self.merge_lora = bool(args.merge_lora)
        self.runtime_resize: DinoResizeConfig | None = None
        self.impl: DinoV3Captioner | Qwen3VLNativeCaptioner | Qwen35Captioner

        if self.model_family == "dinov3":
            if preset.dinov3_dir is None:
                raise ValueError("dinov3 preset requires dinov3_dir")
            image_size = int(args.image_size) if args.image_size is not None else int(preset.image_size or 512)
            min_pixels = (
                int(args.smart_resize_min_pixels)
                if args.smart_resize_min_pixels is not None
                else preset.smart_resize_min_pixels
            )
            max_pixels = (
                int(args.smart_resize_max_pixels)
                if args.smart_resize_max_pixels is not None
                else preset.smart_resize_max_pixels
            )
            resize_override = (
                args.image_size is not None
                or args.smart_resize_min_pixels is not None
                or args.smart_resize_max_pixels is not None
            )
            force_exact_image_size = (
                args.image_size is not None
                and args.smart_resize_min_pixels is not None
                and args.smart_resize_max_pixels is not None
                and int(args.smart_resize_min_pixels) == int(args.image_size) * int(args.image_size)
                and int(args.smart_resize_max_pixels) == int(args.image_size) * int(args.image_size)
            )
            if resize_override:
                if preset.merger_ckpt is not None:
                    run_meta = read_merger_run_meta(preset.merger_ckpt)
                    assert_path_metadata_matches(
                        label="Qwen model dir",
                        expected=run_meta.get("qwen_model_dir"),
                        actual=preset.qwen_model_dir,
                    )
                    assert_path_metadata_matches(
                        label="DINOv3 dir",
                        expected=run_meta.get("dinov3_dir"),
                        actual=preset.dinov3_dir,
                    )
                resize_cfg = resolve_dino_resize_config(
                    image_size=int(image_size),
                    smart_resize_min_pixels=min_pixels,
                    smart_resize_max_pixels=max_pixels,
                    merger_ckpt=None,
                )
            else:
                if preset.merger_ckpt is None:
                    raise ValueError("dinov3 preset requires merger_ckpt when runtime resize is not overridden")
                resize_cfg = assert_dino_runtime_matches_merger(
                    qwen_model_dir=preset.qwen_model_dir,
                    dinov3_dir=preset.dinov3_dir,
                    image_size=int(image_size),
                    smart_resize_min_pixels=min_pixels,
                    smart_resize_max_pixels=max_pixels,
                    merger_ckpt=preset.merger_ckpt,
                )
            self.runtime_resize = resize_cfg
            self.impl = DinoV3Captioner(
                qwen_model_dir=preset.qwen_model_dir,
                dinov3_dir=preset.dinov3_dir,
                image_size=int(resize_cfg.image_size),
                smart_resize_min_pixels=int(resize_cfg.smart_resize_min_pixels),
                smart_resize_max_pixels=int(resize_cfg.smart_resize_max_pixels),
                merger_ckpt=preset.merger_ckpt,
                lora_dir=preset.lora_dir,
                device_map=self.device_map,
                dtype=self.dtype,
                max_new_tokens=int(max_new_tokens),
                do_sample=args.do_sample,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                num_beams=args.num_beams,
                seed=args.seed,
                repetition_penalty=args.repetition_penalty,
                merge_lora=self.merge_lora,
                force_exact_image_size=bool(force_exact_image_size),
            )
            self.decode_strategy = str(self.impl.decode_strategy)
            return

        if self.model_family == "qwen35":
            self.impl = Qwen35Captioner(
                preset.qwen_model_dir,
                device_map=self.device_map,
                dtype=self.dtype,
                max_new_tokens=int(max_new_tokens),
                do_sample=args.do_sample,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                num_beams=args.num_beams,
                repetition_penalty=args.repetition_penalty,
                seed=args.seed,
                enable_thinking=False,
            )
            self.decode_strategy = str(self.impl.decode_strategy)
            return

        if self.model_family != "qwen_native":
            raise ValueError(f"unsupported model_family: {self.model_family}")

        self.impl = Qwen3VLNativeCaptioner(
            preset.qwen_model_dir,
            merger_ckpt=preset.merger_ckpt,
            lora_dir=preset.lora_dir,
            merge_lora=self.merge_lora,
            device_map=self.device_map,
            dtype=self.dtype,
            max_new_tokens=int(max_new_tokens),
            do_sample=args.do_sample,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            num_beams=args.num_beams,
            repetition_penalty=args.repetition_penalty,
            seed=args.seed,
        )
        self.decode_strategy = str(self.impl.decode_strategy)

    def generate_batch_prompts(self, *, image_paths: list[Path], prompts: list[str]):
        return self.impl.caption_batch_prompts(image_paths=image_paths, prompts=prompts)


def resolve_from_project(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return (PROJECT_ROOT / candidate).resolve()


def resolve_from_vrsbench(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return (VRSBENCH_DIR / candidate).resolve()


def checkpoint_run_root(merger_ckpt: Path) -> Path:
    path = Path(merger_ckpt)
    if path.name == "merger.safetensors":
        return path.parents[1]
    raise ValueError(f"unexpected merger checkpoint path: {merger_ckpt}")


def coerce_optional_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except Exception:
        return None


def load_preset_from_checkpoint(*, name: str, model_family: str, merger_ckpt: Path) -> ModelPreset:
    run_root = checkpoint_run_root(merger_ckpt)
    run_config = read_json(run_root / "run_config.json")
    run_meta = run_config.get("run", {})
    qwen_model_dir = resolve_from_vrsbench(run_meta.get("qwen_model_dir", "models/Qwen3-VL-8B-Instruct"))
    lora_dir = merger_ckpt.parent / "lora"

    if model_family == "dinov3":
        dinov3_dir = resolve_from_vrsbench(
            run_meta.get("dinov3_dir", "models/dinov3/dinov3-vitl16-pretrain-sat493m")
        )
        return ModelPreset(
            name=name,
            model_family="dinov3",
            qwen_model_dir=qwen_model_dir,
            dinov3_dir=dinov3_dir,
            merger_ckpt=merger_ckpt,
            lora_dir=lora_dir if lora_dir.is_dir() else None,
            image_size=int(run_meta.get("image_size", 512)),
            smart_resize_min_pixels=coerce_optional_int(run_meta.get("smart_resize_min_pixels")),
            smart_resize_max_pixels=coerce_optional_int(run_meta.get("smart_resize_max_pixels")),
            dtype="bf16",
        )

    if model_family == "qwen_native":
        return ModelPreset(
            name=name,
            model_family="qwen_native",
            qwen_model_dir=qwen_model_dir,
            dinov3_dir=None,
            merger_ckpt=merger_ckpt,
            lora_dir=lora_dir if lora_dir.is_dir() else None,
            image_size=None,
            smart_resize_min_pixels=coerce_optional_int(run_meta.get("smart_resize_min_pixels")),
            smart_resize_max_pixels=coerce_optional_int(run_meta.get("smart_resize_max_pixels")),
            dtype="bf16",
        )

    raise ValueError(f"unsupported model_family: {model_family}")


def build_presets() -> dict[str, ModelPreset]:
    dino_ckpt = resolve_from_project(
        "VRSBench/checkpoints/vrsbench_joint/"
        "merger_lora_8b_dinov3_micro8_8_ga2_effective32_taskseq_smartbucket_sampleavg_wd001_run_20260308_025747/"
        "epoch10/merger.safetensors"
    )
    qwen_native_ckpt = resolve_from_project(
        "VRSBench/checkpoints/vrsbench_joint/"
        "merger_lora_8b_qwen_native_micro8_8_ga2_effective32_wd001_taskseq_run_20260302_160151/"
        "epoch10/merger.safetensors"
    )
    qwen_model_dir = resolve_from_project("VRSBench/models/Qwen3-VL-8B-Instruct")
    qwen35_model_dir = Path("/opt/yanzixi/project/fine-tune-qwen3.5/models/Qwen3.5-9B").resolve()
    return {
        "dinov3_epoch10": load_preset_from_checkpoint(
            name="dinov3_epoch10",
            model_family="dinov3",
            merger_ckpt=dino_ckpt,
        ),
        "qwen_native_epoch10": load_preset_from_checkpoint(
            name="qwen_native_epoch10",
            model_family="qwen_native",
            merger_ckpt=qwen_native_ckpt,
        ),
        "qwen3vl_base": ModelPreset(
            name="qwen3vl_base",
            model_family="qwen_native",
            qwen_model_dir=qwen_model_dir,
            dinov3_dir=None,
            merger_ckpt=None,
            lora_dir=None,
            image_size=None,
            smart_resize_min_pixels=None,
            smart_resize_max_pixels=None,
            dtype="bf16",
        ),
        "qwen35_9b_base": ModelPreset(
            name="qwen35_9b_base",
            model_family="qwen35",
            qwen_model_dir=qwen35_model_dir,
            dinov3_dir=None,
            merger_ckpt=None,
            lora_dir=None,
            image_size=None,
            smart_resize_min_pixels=None,
            smart_resize_max_pixels=None,
            dtype="bf16",
        ),
    }


def default_prompt(*, question: str, image_name: str, question_type: str) -> str:
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


def strip_output_wrappers(text: str) -> str:
    value = str(text or "").strip()
    if not value:
        return value
    value = re.sub(r"^<\|im_start\|>\s*assistant\s*", "", value, flags=re.IGNORECASE)
    value = re.sub(r"^assistant\s*", "", value, flags=re.IGNORECASE)
    value = value.replace("<|im_end|>", "").strip()
    value = re.sub(r"^```(?:text|markdown|json)?\s*", "", value, flags=re.IGNORECASE)
    value = re.sub(r"\s*```$", "", value)
    return value.strip()


def extract_short_answer(text: str) -> str:
    value = strip_output_wrappers(text)
    if not value:
        return ""
    value = value.replace("\r", "\n").strip()
    value = re.sub(r"^\s*answer\s*:\s*", "", value, flags=re.IGNORECASE)
    value = re.sub(r"^\s*final answer\s*:\s*", "", value, flags=re.IGNORECASE)
    lines = [line.strip() for line in value.split("\n") if line.strip()]
    value = lines[0] if lines else value
    value = re.split(r"(?<=[.!?])\s+", value, maxsplit=1)[0].strip()
    value = value.strip(" \"'`.,;:()[]{}")
    return value


def normalize_free_text(text: str) -> str:
    value = str(text or "").strip().lower()
    value = re.sub(r"[_/]", " ", value)
    value = value.replace("&", " and ")
    value = re.sub(r"\bthe\b|\ba\b|\ban\b", " ", value)
    value = re.sub(r"[^a-z0-9\- ]+", " ", value)
    value = re.sub(r"\s+", " ", value).strip()
    return value


def normalize_yes_no(text: str) -> str:
    value = normalize_free_text(text)
    if value in YES_SET:
        return "yes"
    if value in NO_SET:
        return "no"
    if value.startswith("yes "):
        return "yes"
    if value.startswith("no "):
        return "no"
    return value


def normalize_count(text: str) -> str:
    value = normalize_free_text(text)
    if value in NUMBER_WORDS:
        return NUMBER_WORDS[value]
    parts = value.split()
    if parts and parts[0] in NUMBER_WORDS:
        return NUMBER_WORDS[parts[0]]
    match = re.search(r"\d+", value)
    if match is not None:
        return match.group(0)
    return value


def normalize_answer(text: str, question_type: str) -> str:
    short_answer = extract_short_answer(text)
    if question_type == "object existence":
        return normalize_yes_no(short_answer)
    if question_type == "object quantity":
        return normalize_count(short_answer)
    value = normalize_free_text(short_answer)
    if value in NUMBER_WORDS:
        return NUMBER_WORDS[value]
    return value


def runtime_meta(runner: CaptionRunner) -> dict[str, Any]:
    meta = {
        "preset": runner.preset.name,
        "model_family": runner.model_family,
        "qwen_model_dir": str(runner.preset.qwen_model_dir),
        "merger_ckpt": str(runner.preset.merger_ckpt) if runner.preset.merger_ckpt is not None else "",
        "lora_dir": str(runner.preset.lora_dir) if runner.preset.lora_dir is not None else "",
        "device_map": str(runner.device_map),
        "dtype": str(runner.dtype),
        "decode_strategy": str(runner.decode_strategy),
    }
    if runner.model_family == "dinov3":
        meta["dinov3_dir"] = str(runner.preset.dinov3_dir)
        if runner.runtime_resize is not None:
            meta["image_size"] = int(runner.runtime_resize.image_size)
            meta["smart_resize_min_pixels"] = int(runner.runtime_resize.smart_resize_min_pixels)
            meta["smart_resize_max_pixels"] = int(runner.runtime_resize.smart_resize_max_pixels)
            meta["resize_mode"] = str(runner.runtime_resize.mode)
        meta["force_exact_image_size"] = bool(getattr(runner.impl, "force_exact_image_size", False))
    return meta


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate official VRSBench VQA predictions.")
    parser.add_argument(
        "--preset",
        type=str,
        required=True,
        choices=["dinov3_epoch10", "qwen_native_epoch10", "qwen3vl_base", "qwen35_9b_base"],
    )
    parser.add_argument(
        "--data",
        type=str,
        default="VRSBench/benchmark/vrsbench/data/vrsbench_vqa_test.jsonl",
    )
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--device-map", type=str, default="cuda:0")
    parser.add_argument("--dtype", type=str, default="")
    parser.add_argument("--do-sample", dest="do_sample", action="store_true", default=None)
    parser.add_argument("--no-sample", dest="do_sample", action="store_false")
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--num-beams", type=int, default=None)
    parser.add_argument("--repetition-penalty", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--merge-lora", action="store_true")
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--smart-resize-min-pixels", type=int, default=None)
    parser.add_argument("--smart-resize-max-pixels", type=int, default=None)
    parser.add_argument("--max-questions", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    num_shards = int(args.num_shards)
    shard_index = int(args.shard_index)
    if num_shards <= 0:
        raise ValueError(f"num_shards must be positive, got {num_shards}")
    if shard_index < 0 or shard_index >= num_shards:
        raise ValueError(f"shard_index must be in [0, {num_shards}), got {shard_index}")

    presets = build_presets()
    preset = presets[str(args.preset)]
    data_path = resolve_from_project(args.data)
    output_path = resolve_from_project(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not data_path.is_file():
        raise FileNotFoundError(f"missing prepared file: {data_path}")

    rows = read_jsonl(data_path)
    rows = sorted(rows, key=lambda row: int(row["qid"]))
    if int(args.max_questions) > 0:
        rows = rows[: int(args.max_questions)]
    if num_shards > 1:
        rows = [row for idx, row in enumerate(rows) if idx % num_shards == shard_index]

    done_qids: set[int] = set()
    if output_path.is_file():
        for row in read_jsonl(output_path):
            try:
                done_qids.add(int(row["qid"]))
            except Exception:
                continue
    pending = [row for row in rows if int(row["qid"]) not in done_qids]
    if not pending:
        print(f"[OK] No pending questions. Output already complete: {output_path}", flush=True)
        return

    runner = CaptionRunner(preset=preset, args=args, max_new_tokens=int(args.max_new_tokens))
    max_batch_size = max(1, int(args.batch_size))
    current_batch_size = int(max_batch_size)
    progress = tqdm(total=len(pending), desc=f"vrsbench:vqa:{args.preset}")
    index = 0

    while index < len(pending):
        chunk = pending[index : index + current_batch_size]
        image_paths = [resolve_from_project(row["image_path"]) for row in chunk]
        prompts = [
            default_prompt(
                question=str(row["question"]),
                image_name=str(row["image_id"]),
                question_type=str(row["type"]),
            )
            for row in chunk
        ]
        try:
            outputs = runner.generate_batch_prompts(image_paths=image_paths, prompts=prompts)
        except torch.cuda.OutOfMemoryError:
            gc.collect()
            if current_batch_size <= 1:
                raise
            current_batch_size = max(1, current_batch_size // 2)
            continue

        meta = runtime_meta(runner)
        for row, prompt, output in zip(chunk, prompts, outputs):
            prediction_raw = str(output.text)
            answer = extract_short_answer(prediction_raw)
            answer_normalized = normalize_answer(prediction_raw, str(row["type"]))
            append_jsonl(
                output_path,
                {
                    "qid": int(row["qid"]),
                    "question_id": int(row["question_id"]),
                    "image_id": str(row["image_id"]),
                    "filename": str(row["filename"]),
                    "image_path": str(row["image_path"]),
                    "question": str(row["question"]),
                    "type": str(row["type"]),
                    "dataset": str(row["dataset"]),
                    "ground_truth": str(row["ground_truth"]),
                    "prompt": prompt,
                    "answer": answer,
                    "answer_normalized": answer_normalized,
                    "prediction_raw": prediction_raw,
                    "generated_token_count": int(output.generated_token_count),
                    "generation_ended_by_eos": bool(output.ended_by_eos),
                    "generation_last_token_id": output.last_generated_token_id,
                    "batch_size": int(len(chunk)),
                    "requested_batch_size": int(max_batch_size),
                    **meta,
                },
            )
        progress.update(len(chunk))
        index += len(chunk)

    progress.close()
    write_json(
        output_path.with_suffix(".summary.json"),
        {
            "preset": str(args.preset),
            "data": str(data_path),
            "output": str(output_path),
            "num_questions": len(rows),
            "num_pending_before_run": len(pending),
            "num_shards": num_shards,
            "shard_index": shard_index,
            "max_new_tokens": int(args.max_new_tokens),
            "requested_batch_size": int(max_batch_size),
            "runtime": runtime_meta(runner),
        },
    )
    print(f"[OK] Wrote predictions: {output_path}", flush=True)


if __name__ == "__main__":
    main()
