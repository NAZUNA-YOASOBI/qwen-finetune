from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class GenerateKwargsConfig:
    gen_kwargs: dict[str, Any]
    strategy: str


@dataclass(frozen=True)
class GeneratedSequenceSummary:
    token_ids: list[int]
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
            maybe_token_id = _coerce_optional_int(token_id)
            if maybe_token_id is not None:
                out.add(int(maybe_token_id))
        return out
    maybe_token_id = _coerce_optional_int(token_ids)
    return {int(maybe_token_id)} if maybe_token_id is not None else set()


def build_generate_kwargs(
    *,
    max_new_tokens: int,
    do_sample: bool | None = None,
    temperature: float | None = None,
    top_p: float | None = None,
    top_k: int | None = None,
    num_beams: int | None = None,
    repetition_penalty: float | None = None,
) -> GenerateKwargsConfig:
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
    return GenerateKwargsConfig(gen_kwargs=gen_kwargs, strategy=str(strategy))


def summarize_generated_sequences(
    generated_ids: Any,
    prompt_lens: list[int] | tuple[int, ...],
    *,
    eos_token_id: Any,
    pad_token_id: Any,
) -> list[GeneratedSequenceSummary]:
    prompt_lens_list = [int(x) for x in prompt_lens]
    batch_size = int(generated_ids.shape[0]) if hasattr(generated_ids, "shape") else len(generated_ids)
    if int(batch_size) != int(len(prompt_lens_list)):
        raise ValueError(
            "generated_ids and prompt_lens size mismatch: "
            f"generated={batch_size}, prompt_lens={len(prompt_lens_list)}"
        )

    eos_ids = _normalize_token_id_set(eos_token_id)
    pad_id = _coerce_optional_int(pad_token_id)
    summaries: list[GeneratedSequenceSummary] = []

    for row_ids, prompt_len in zip(generated_ids, prompt_lens_list):
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
        last_generated_token_id = int(effective_ids[-1]) if effective_ids else None
        summaries.append(
            GeneratedSequenceSummary(
                token_ids=[int(x) for x in effective_ids],
                generated_token_count=int(len(effective_ids)),
                ended_by_eos=bool(ended_by_eos),
                last_generated_token_id=last_generated_token_id,
            )
        )
    return summaries


def maybe_set_generation_seed(seed: int | None) -> None:
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


def torch_dtype_from_str(torch_mod, dtype: str):
    dtype = str(dtype).lower().strip()
    if dtype == "auto":
        return torch_mod.float16 if torch_mod.cuda.is_available() else torch_mod.float32
    if dtype in {"fp16", "float16"}:
        return torch_mod.float16
    if dtype in {"bf16", "bfloat16"}:
        return torch_mod.bfloat16
    if dtype in {"fp32", "float32"}:
        return torch_mod.float32
    return torch_mod.float32
