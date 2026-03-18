from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class GenerateResult:
    text: str
    generated_token_count: int
    ended_by_eos: bool
    last_generated_token_id: int | None


def _coerce_optional_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except Exception:
        return None


def _normalize_token_ids(tokens: Any) -> list[int]:
    if tokens is None:
        return []
    if hasattr(tokens, "detach") and hasattr(tokens, "cpu"):
        values = tokens.detach().cpu().tolist()
    elif hasattr(tokens, "tolist"):
        values = tokens.tolist()
    else:
        values = list(tokens)
    return [int(x) for x in values]


def _normalize_token_id_set(token_ids: Any) -> set[int]:
    if token_ids is None:
        return set()
    if isinstance(token_ids, (list, tuple, set)):
        out: set[int] = set()
        for token_id in token_ids:
            maybe = _coerce_optional_int(token_id)
            if maybe is not None:
                out.add(int(maybe))
        return out
    maybe = _coerce_optional_int(token_ids)
    return {int(maybe)} if maybe is not None else set()


def _summarize_generated_sequences(generated_ids: Any, prompt_lens: list[int], *, eos_token_id: Any, pad_token_id: Any) -> list[GenerateResult]:
    eos_ids = _normalize_token_id_set(eos_token_id)
    pad_id = _coerce_optional_int(pad_token_id)
    summaries: list[GenerateResult] = []
    for row_ids, prompt_len in zip(generated_ids, prompt_lens):
        full_ids = _normalize_token_ids(row_ids)
        effective_ids = full_ids[int(prompt_len) :] if int(prompt_len) < len(full_ids) else []

        if pad_id is not None and (not eos_ids or int(pad_id) not in eos_ids):
            try:
                first_pad = effective_ids.index(int(pad_id))
                effective_ids = effective_ids[:first_pad]
            except ValueError:
                pass

        if eos_ids:
            first_eos: int | None = None
            for idx, token_id in enumerate(effective_ids):
                if int(token_id) in eos_ids:
                    first_eos = int(idx)
                    break
            if first_eos is not None:
                effective_ids = effective_ids[: first_eos + 1]

        ended_by_eos = bool(effective_ids) and bool(eos_ids) and int(effective_ids[-1]) in eos_ids
        summaries.append(
            GenerateResult(
                text="",
                generated_token_count=int(len(effective_ids)),
                ended_by_eos=bool(ended_by_eos),
                last_generated_token_id=int(effective_ids[-1]) if effective_ids else None,
            )
        )
    return summaries


def _build_generate_kwargs(
    *,
    max_new_tokens: int,
    do_sample: bool | None = None,
    temperature: float | None = None,
    top_p: float | None = None,
    top_k: int | None = None,
    num_beams: int | None = None,
    repetition_penalty: float | None = None,
) -> tuple[dict[str, Any], str]:
    gen_kwargs: dict[str, Any] = {"max_new_tokens": int(max_new_tokens)}
    explicit_fields: list[str] = []
    if do_sample is not None:
        gen_kwargs["do_sample"] = bool(do_sample)
        explicit_fields.append("do_sample")
    if temperature is not None:
        gen_kwargs["temperature"] = float(temperature)
        explicit_fields.append("temperature")
    if top_p is not None:
        gen_kwargs["top_p"] = float(top_p)
        explicit_fields.append("top_p")
    if top_k is not None:
        gen_kwargs["top_k"] = int(top_k)
        explicit_fields.append("top_k")
    if num_beams is not None:
        gen_kwargs["num_beams"] = int(num_beams)
        explicit_fields.append("num_beams")
    if repetition_penalty is not None:
        gen_kwargs["repetition_penalty"] = float(repetition_penalty)
        explicit_fields.append("repetition_penalty")
    strategy = "generation_config_default" if not explicit_fields else "cli_override:" + ",".join(explicit_fields)
    return gen_kwargs, strategy


def _maybe_set_generation_seed(seed: int | None) -> None:
    if seed is None:
        return
    seed_value = int(seed)
    random.seed(seed_value)
    try:
        import numpy as np

        np.random.seed(seed_value)
    except Exception:
        pass
    try:
        from transformers import set_seed  # type: ignore

        set_seed(seed_value)
    except Exception:
        pass
    try:
        import torch

        torch.manual_seed(seed_value)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed_value)
    except Exception:
        pass


def _torch_dtype_from_str(torch_mod, dtype: str):
    norm = str(dtype).lower().strip()
    if norm == "auto":
        return torch_mod.float16 if torch_mod.cuda.is_available() else torch_mod.float32
    if norm in {"fp16", "float16"}:
        return torch_mod.float16
    if norm in {"bf16", "bfloat16"}:
        return torch_mod.bfloat16
    if norm in {"fp32", "float32"}:
        return torch_mod.float32
    return torch_mod.float32


def _assert_safe_inference_device_map(torch_mod, device_map: str) -> None:
    raw = str(device_map).strip().lower()
    if raw == "cpu":
        raise ValueError("device_map=cpu is not allowed for this project.")
    if raw != "auto":
        return
    visible_cuda = int(torch_mod.cuda.device_count())
    if visible_cuda > 1:
        raise RuntimeError(
            "Unsafe inference setup: device_map=auto with multiple visible CUDA devices. "
            "Expose exactly one GPU via CUDA_VISIBLE_DEVICES, or use the project shard wrapper."
        )


class Qwen3VLRunner:
    def __init__(
        self,
        model_dir: str | Path,
        *,
        device_map: str = "auto",
        dtype: str = "auto",
        max_new_tokens: int = 256,
        do_sample: bool | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        top_k: int | None = None,
        num_beams: int | None = None,
        repetition_penalty: float | None = None,
        seed: int | None = None,
    ) -> None:
        self.model_dir = Path(model_dir)
        self.device_map = str(device_map)
        self.dtype = str(dtype)
        self.max_new_tokens = int(max_new_tokens)
        self.gen_kwargs, self.decode_strategy = _build_generate_kwargs(
            max_new_tokens=int(max_new_tokens),
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            num_beams=num_beams,
            repetition_penalty=repetition_penalty,
        )
        _maybe_set_generation_seed(seed)

        try:
            import torch
            from transformers import AutoProcessor, Qwen3VLForConditionalGeneration  # type: ignore
        except Exception as e:
            raise RuntimeError("Missing dependency: transformers/torch") from e

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for this project.")
        _assert_safe_inference_device_map(torch, self.device_map)

        torch_dtype = _torch_dtype_from_str(torch, self.dtype)
        self.processor = AutoProcessor.from_pretrained(str(self.model_dir))
        try:
            self.processor.tokenizer.padding_side = "left"
        except Exception:
            pass
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

    def generate_batch(self, *, image_paths: list[str | Path], prompts: list[str]) -> list[GenerateResult]:
        try:
            import torch
        except Exception as e:
            raise RuntimeError("Missing dependency: torch") from e

        if len(image_paths) != len(prompts):
            raise ValueError(f"image_paths and prompts must have same length, got {len(image_paths)} vs {len(prompts)}")
        if not image_paths:
            return []

        conversations: list[list[dict[str, Any]]] = []
        for image_path, prompt in zip(image_paths, prompts):
            conversations.append(
                [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": str(Path(image_path))},
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

        with torch.no_grad():
            generated_ids = self.model.generate(**inputs, **self.gen_kwargs)

        input_ids = inputs.get("input_ids", None)
        if input_ids is not None:
            prompt_lens = [int(input_ids.shape[1])] * int(generated_ids.shape[0])
        else:
            attn = inputs.get("attention_mask", None)
            if attn is None:
                raise RuntimeError("Missing both input_ids and attention_mask while trimming generated ids.")
            prompt_lens = [int(attn.shape[1])] * int(generated_ids.shape[0])

        generation_config = getattr(self.model, "generation_config", None)
        eos_token_id = getattr(generation_config, "eos_token_id", None)
        pad_token_id = getattr(generation_config, "pad_token_id", None)
        if eos_token_id is None:
            eos_token_id = getattr(self.processor.tokenizer, "eos_token_id", None)
        if pad_token_id is None:
            pad_token_id = getattr(self.processor.tokenizer, "pad_token_id", None)

        summaries = _summarize_generated_sequences(
            generated_ids,
            prompt_lens,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
        )
        texts = self.processor.batch_decode(
            [
                _normalize_token_ids(row)[int(prompt_len) :]
                for row, prompt_len in zip(generated_ids, prompt_lens)
            ],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

        out: list[GenerateResult] = []
        for text, summary in zip(texts, summaries):
            out.append(
                GenerateResult(
                    text=str(text).strip(),
                    generated_token_count=int(summary.generated_token_count),
                    ended_by_eos=bool(summary.ended_by_eos),
                    last_generated_token_id=summary.last_generated_token_id,
                )
            )
        return out
