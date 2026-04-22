from __future__ import annotations

import argparse
import gc
import json
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from tqdm import tqdm

SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[6]
VRSBENCH_ROOT = SCRIPT_PATH.parents[5]
GEOCHAT_CODE_ROOT = PROJECT_ROOT / "GeoChat-Bench" / "GeoChat"
if str(GEOCHAT_CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(GEOCHAT_CODE_ROOT))

from geochat.constants import (  # type: ignore
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_END_TOKEN,
    DEFAULT_IM_START_TOKEN,
    IMAGE_TOKEN_INDEX,
)
from geochat.conversation import SeparatorStyle, conv_templates  # type: ignore
from geochat.mm_utils import (  # type: ignore
    KeywordsStoppingCriteria,
    get_model_name_from_path,
    process_images,
    tokenizer_image_token,
)
from geochat.model.builder import load_pretrained_model  # type: ignore
from geochat.utils import disable_torch_init  # type: ignore


DEFAULT_CAPTION_PROMPT = (
    "Write one concise English caption for this remote sensing image in 8 to 15 words. "
    "Mention only the main scene and key objects."
)


@dataclass(frozen=True)
class CaptionResult:
    text: str
    generated_token_count: int
    ended_by_eos: bool
    last_generated_token_id: int | None


def _resolve_from_project(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return (PROJECT_ROOT / candidate).resolve()


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.is_file():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text).strip())


def _infer_conv_mode(model_name: str) -> str:
    model_name_lower = str(model_name).lower()
    if "llama-2" in model_name_lower:
        return "llava_llama_2"
    if "v1" in model_name_lower:
        return "llava_v1"
    if "mpt" in model_name_lower:
        return "mpt"
    return "llava_v0"


def _load_rsicd_test_rows(dataset_json: Path, image_root: Path) -> list[dict[str, Any]]:
    payload = _read_json(dataset_json)
    rows: list[dict[str, Any]] = []
    for item in payload["images"]:
        if str(item.get("split", "")).strip().lower() != "test":
            continue
        filename = str(item["filename"]).strip()
        refs = []
        for sentence in item.get("sentences", []):
            raw = _normalize_text(str(sentence.get("raw", "")))
            if raw:
                refs.append(raw)
        rows.append(
            {
                "dataset": "rsicd",
                "task": "caption",
                "sample_id": str(item.get("imgid", filename)),
                "filename": filename,
                "image_path": image_root / filename,
                "refs": refs,
            }
        )
    rows.sort(key=lambda row: str(row["filename"]))
    return rows


class GeoChatCaptioner:
    def __init__(
        self,
        *,
        model_dir: Path,
        device: str,
        image_aspect_ratio: str,
        max_new_tokens: int,
        do_sample: bool,
        temperature: float,
        top_p: float | None,
        num_beams: int,
        repetition_penalty: float | None,
        load_8bit: bool,
        load_4bit: bool,
    ) -> None:
        disable_torch_init()
        self.model_dir = Path(model_dir)
        self.device = str(device)
        self.image_aspect_ratio = str(image_aspect_ratio)
        self.max_new_tokens = int(max_new_tokens)
        self.do_sample = bool(do_sample)
        self.temperature = float(temperature)
        self.top_p = None if top_p is None else float(top_p)
        self.num_beams = int(num_beams)
        self.repetition_penalty = None if repetition_penalty is None else float(repetition_penalty)
        self.model_name = get_model_name_from_path(str(self.model_dir))
        self.conv_mode = _infer_conv_mode(self.model_name)
        self.decode_strategy = "sample" if self.do_sample else "greedy"

        if self.device == "cpu":
            device_map: dict[str, Any] = {"": "cpu"}
        else:
            device_index = 0
            if ":" in self.device:
                suffix = str(self.device).split(":")[-1].strip()
                if suffix.isdigit():
                    device_index = int(suffix)
            device_map = {"": device_index}

        tokenizer, model, image_processor, _ = load_pretrained_model(
            model_path=str(self.model_dir),
            model_base=None,
            model_name=str(self.model_name),
            load_8bit=bool(load_8bit),
            load_4bit=bool(load_4bit),
            device_map=device_map,
            device=str(self.device),
        )
        model.config.image_aspect_ratio = str(self.image_aspect_ratio)

        self.tokenizer = tokenizer
        self.model = model
        self.image_processor = image_processor
        self.model.eval()

    def generate_one(self, *, image_path: Path, prompt: str) -> CaptionResult:
        image = Image.open(image_path).convert("RGB")
        image_tensor = process_images([image], self.image_processor, self.model.config)
        if isinstance(image_tensor, list):
            image_inputs = [tensor.to(device=self.model.device, dtype=torch.float16) for tensor in image_tensor]
        else:
            image_inputs = image_tensor.to(device=self.model.device, dtype=torch.float16)

        conv = conv_templates[str(self.conv_mode)].copy()
        if bool(getattr(self.model.config, "mm_use_im_start_end", False)):
            user_prompt = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + "\n" + str(prompt)
        else:
            user_prompt = DEFAULT_IMAGE_TOKEN + "\n" + str(prompt)
        conv.append_message(conv.roles[0], user_prompt)
        conv.append_message(conv.roles[1], None)
        full_prompt = conv.get_prompt()

        input_ids = tokenizer_image_token(
            full_prompt,
            self.tokenizer,
            IMAGE_TOKEN_INDEX,
            return_tensors="pt",
        ).unsqueeze(0).to(device=self.model.device)
        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        stopping_criteria = KeywordsStoppingCriteria([str(stop_str)], self.tokenizer, input_ids)

        generate_args: dict[str, Any] = {
            "input_ids": input_ids,
            "images": image_inputs,
            "do_sample": bool(self.do_sample),
            "max_new_tokens": int(self.max_new_tokens),
            "use_cache": True,
            "stopping_criteria": [stopping_criteria],
        }
        if self.do_sample:
            generate_args["temperature"] = float(self.temperature)
            if self.top_p is not None:
                generate_args["top_p"] = float(self.top_p)
        else:
            generate_args["num_beams"] = int(self.num_beams)
        if self.repetition_penalty is not None:
            generate_args["repetition_penalty"] = float(self.repetition_penalty)

        with torch.inference_mode():
            output_ids = self.model.generate(**generate_args)

        generated_ids = output_ids[0, input_ids.shape[1] :]
        decoded = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        if stop_str and decoded.endswith(str(stop_str)):
            decoded = decoded[: -len(str(stop_str))].strip()
        generated_token_count = int(generated_ids.shape[0])
        last_generated_token_id = None
        ended_by_eos = False
        if generated_token_count > 0:
            last_generated_token_id = int(generated_ids[-1].item())
            ended_by_eos = last_generated_token_id == int(self.tokenizer.eos_token_id)

        return CaptionResult(
            text=_normalize_text(decoded),
            generated_token_count=generated_token_count,
            ended_by_eos=bool(ended_by_eos),
            last_generated_token_id=last_generated_token_id,
        )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate GeoChat-7B on RSICD with the VRSBench single-task pipeline.")
    parser.add_argument("--model-dir", type=str, default="GeoChat-Bench/model/geochat-7B")
    parser.add_argument(
        "--data",
        type=str,
        default="VRSBench/benchmark/single_task/datasets/caption/RSICD/benchmark/data/dataset_rsicd.json",
    )
    parser.add_argument(
        "--image-root",
        type=str,
        default="VRSBench/benchmark/single_task/datasets/caption/RSICD/official/raw/RSICD_images",
    )
    parser.add_argument("--output-dir", type=str, default="VRSBench/benchmark/single_task/outputs/geochat7b_rsicd")
    parser.add_argument("--summary-out", type=str, default="VRSBench/benchmark/single_task/eval/geochat7b_rsicd/evaluation_summary.json")
    parser.add_argument("--caption-prompt", type=str, default=DEFAULT_CAPTION_PROMPT)
    parser.add_argument("--max-new-tokens", type=int, default=48)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--image-aspect-ratio", type=str, default="pad")
    parser.add_argument("--do-sample", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--num-beams", type=int, default=1)
    parser.add_argument("--repetition-penalty", type=float, default=None)
    parser.add_argument("--load-8bit", action="store_true")
    parser.add_argument("--load-4bit", action="store_true")
    parser.add_argument("--max-rows", type=int, default=0)
    parser.add_argument("--evaluate-only", action="store_true")
    return parser


def main() -> None:
    args = _build_parser().parse_args()

    model_dir = _resolve_from_project(args.model_dir)
    data_path = _resolve_from_project(args.data)
    image_root = _resolve_from_project(args.image_root)
    output_dir = _resolve_from_project(args.output_dir)
    summary_out = _resolve_from_project(args.summary_out)
    pred_path = output_dir / "rsicd.jsonl"
    generation_summary_path = output_dir / "generation_summary.json"

    if not model_dir.is_dir():
        raise FileNotFoundError(f"Missing model dir: {model_dir}")
    if not data_path.is_file():
        raise FileNotFoundError(f"Missing RSICD json: {data_path}")
    if not image_root.is_dir():
        raise FileNotFoundError(f"Missing RSICD image root: {image_root}")

    rows = _load_rsicd_test_rows(data_path, image_root)
    if int(args.max_rows) > 0:
        rows = rows[: int(args.max_rows)]
    if not rows:
        raise ValueError(f"No RSICD test rows found in {data_path}")

    done_ids = {str(row.get("sample_id", "")).strip() for row in _read_jsonl(pred_path)}
    pending = [row for row in rows if str(row["sample_id"]) not in done_ids]

    if not args.evaluate_only and pending:
        runner = GeoChatCaptioner(
            model_dir=model_dir,
            device=str(args.device),
            image_aspect_ratio=str(args.image_aspect_ratio),
            max_new_tokens=int(args.max_new_tokens),
            do_sample=bool(args.do_sample),
            temperature=float(args.temperature),
            top_p=args.top_p,
            num_beams=int(args.num_beams),
            repetition_penalty=args.repetition_penalty,
            load_8bit=bool(args.load_8bit),
            load_4bit=bool(args.load_4bit),
        )

        sample_predictions: list[dict[str, Any]] = []
        progress = tqdm(total=len(pending), desc="rsicd:geochat7b")
        for row in pending:
            try:
                pred = runner.generate_one(
                    image_path=Path(row["image_path"]),
                    prompt=str(args.caption_prompt),
                )
            except torch.cuda.OutOfMemoryError:
                gc.collect()
                raise

            payload = {
                "dataset": "rsicd",
                "task": "caption",
                "sample_id": str(row["sample_id"]),
                "filename": str(row["filename"]),
                "image_path": str(row["image_path"]),
                "prompt": str(args.caption_prompt),
                "prediction": str(pred.text),
                "refs": list(row["refs"]),
                "model_family": "geochat7b",
                "model_dir": str(model_dir),
                "device": str(args.device),
                "image_aspect_ratio": str(args.image_aspect_ratio),
                "max_new_tokens": int(args.max_new_tokens),
                "do_sample": bool(args.do_sample),
                "temperature": float(args.temperature),
                "top_p": args.top_p,
                "num_beams": int(args.num_beams),
                "repetition_penalty": args.repetition_penalty,
                "decode_strategy": str(runner.decode_strategy),
                "generated_token_count": int(pred.generated_token_count),
                "generation_ended_by_eos": bool(pred.ended_by_eos),
                "generation_last_token_id": pred.last_generated_token_id,
            }
            _append_jsonl(pred_path, payload)
            if len(sample_predictions) < 8:
                sample_predictions.append(
                    {
                        "filename": str(row["filename"]),
                        "prediction": str(pred.text),
                        "first_ref": str(row["refs"][0]) if row["refs"] else "",
                    }
                )
            progress.update(1)
        progress.close()

        final_rows = _read_jsonl(pred_path)
        generation_summary = {
            "dataset": "rsicd",
            "task": "caption",
            "model_family": "geochat7b",
            "model_dir": str(model_dir),
            "device": str(args.device),
            "num_rows": len(final_rows),
            "num_pending_before_run": len(pending),
            "prompt": str(args.caption_prompt),
            "max_new_tokens": int(args.max_new_tokens),
            "do_sample": bool(args.do_sample),
            "temperature": float(args.temperature),
            "top_p": args.top_p,
            "num_beams": int(args.num_beams),
            "repetition_penalty": args.repetition_penalty,
            "image_aspect_ratio": str(args.image_aspect_ratio),
            "sample_predictions": sample_predictions,
        }
        _write_json(generation_summary_path, generation_summary)
        print(f"[OK] Wrote predictions: {pred_path}", flush=True)
        print(f"[OK] Wrote generation summary: {generation_summary_path}", flush=True)
    elif args.evaluate_only:
        print(f"[INFO] Skip generation because --evaluate-only is set.", flush=True)
    else:
        print(f"[OK] No pending RSICD rows. Reuse existing predictions: {pred_path}", flush=True)

    eval_script = VRSBENCH_ROOT / "benchmark" / "single_task" / "common" / "scripts" / "evaluate_single_task.py"
    command = [
        sys.executable,
        str(eval_script),
        "--output-dir",
        str(output_dir),
        "--summary-out",
        str(summary_out),
    ]
    subprocess.run(command, check=True)
    print(f"[OK] Wrote evaluation summary: {summary_out}", flush=True)


if __name__ == "__main__":
    main()
