from __future__ import annotations

import json
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


@dataclass(frozen=True)
class _SequenceSummary:
    token_ids: list[int]
    generated_token_count: int
    ended_by_eos: bool
    last_generated_token_id: int | None


MCQ_CHOICE_GRAMMAR = 'root ::= "A" | "B" | "C" | "D" | "E"'

def _build_refdet_json_schema(expected_count: int | None = None) -> str:
    schema: dict[str, Any] = {
        "type": "array",
        "minItems": 1,
        "items": {
            "type": "object",
            "additionalProperties": False,
            "required": ["bbox_2d"],
            "properties": {
                "bbox_2d": {
                    "type": "array",
                    "minItems": 4,
                    "maxItems": 4,
                    "items": {
                        "type": "array",
                        "minItems": 2,
                        "maxItems": 2,
                        "items": {
                            "type": "integer",
                            "minimum": -128,
                            "maximum": 1280,
                        },
                    },
                },
            },
        },
    }
    if expected_count is not None and int(expected_count) > 0:
        schema["minItems"] = int(expected_count)
        schema["maxItems"] = int(expected_count)
    return json.dumps(schema, ensure_ascii=False)


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


def _summarize_generated_sequences(generated_ids: Any, prompt_lens: list[int], *, eos_token_id: Any, pad_token_id: Any) -> list[_SequenceSummary]:
    eos_ids = _normalize_token_id_set(eos_token_id)
    pad_id = _coerce_optional_int(pad_token_id)
    summaries: list[_SequenceSummary] = []
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
            for index, token_id in enumerate(effective_ids):
                if int(token_id) in eos_ids:
                    first_eos = int(index)
                    break
            if first_eos is not None:
                effective_ids = effective_ids[: first_eos + 1]

        ended_by_eos = bool(effective_ids) and bool(eos_ids) and int(effective_ids[-1]) in eos_ids
        summaries.append(
            _SequenceSummary(
                token_ids=list(effective_ids),
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


def _configure_generation_special_tokens(processor: Any, model: Any, gen_kwargs: dict[str, Any]) -> None:
    generation_config = getattr(model, "generation_config", None)
    tokenizer = getattr(processor, "tokenizer", None)

    eos_ids: set[int] = set()
    if generation_config is not None:
        eos_ids.update(_normalize_token_id_set(getattr(generation_config, "eos_token_id", None)))
    if tokenizer is not None:
        eos_ids.update(_normalize_token_id_set(getattr(tokenizer, "eos_token_id", None)))
    eos_token_id = sorted(eos_ids) if len(eos_ids) > 1 else (next(iter(eos_ids)) if eos_ids else None)
    primary_eos_token_id = sorted(eos_ids)[0] if eos_ids else None
    pad_token_id = getattr(generation_config, "pad_token_id", None) if generation_config is not None else None

    if pad_token_id is None and tokenizer is not None:
        pad_token_id = getattr(tokenizer, "pad_token_id", None)
    if pad_token_id is None:
        pad_token_id = primary_eos_token_id

    if generation_config is not None:
        if getattr(generation_config, "eos_token_id", None) is None and eos_token_id is not None:
            generation_config.eos_token_id = eos_token_id
        if getattr(generation_config, "pad_token_id", None) is None and pad_token_id is not None:
            generation_config.pad_token_id = pad_token_id

    if eos_token_id is not None and "eos_token_id" not in gen_kwargs:
        if isinstance(eos_token_id, (list, tuple, set)):
            gen_kwargs["eos_token_id"] = [int(token_id) for token_id in eos_token_id]
        else:
            gen_kwargs["eos_token_id"] = int(eos_token_id)
    if pad_token_id is not None and "pad_token_id" not in gen_kwargs:
        gen_kwargs["pad_token_id"] = int(pad_token_id)


def _resolve_generation_stop_token_ids(model: Any, processor: Any, gen_kwargs: dict[str, Any]) -> list[int]:
    candidates = []
    if "eos_token_id" in gen_kwargs:
        candidates.append(gen_kwargs.get("eos_token_id"))
    generation_config = getattr(model, "generation_config", None)
    if generation_config is not None:
        candidates.append(getattr(generation_config, "eos_token_id", None))
    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is not None:
        candidates.append(getattr(tokenizer, "eos_token_id", None))
    merged: set[int] = set()
    for candidate in candidates:
        merged.update(_normalize_token_id_set(candidate))
    return sorted(int(token_id) for token_id in merged)


def _resolve_summary_special_tokens(model: Any, processor: Any, gen_kwargs: dict[str, Any]) -> tuple[Any, Any]:
    eos_token_id = _resolve_generation_stop_token_ids(model, processor, gen_kwargs)
    generation_config = getattr(model, "generation_config", None)
    tokenizer = getattr(processor, "tokenizer", None)
    pad_token_id = getattr(generation_config, "pad_token_id", None) if generation_config is not None else None
    if pad_token_id is None and tokenizer is not None:
        pad_token_id = getattr(tokenizer, "pad_token_id", None)
    return eos_token_id, pad_token_id


def _init_xgrammar(processor: Any, model_dir: Path) -> tuple[Any | None, Any | None, dict[str, Any]]:
    try:
        import xgrammar as xgr
        from transformers import AutoConfig  # type: ignore
    except Exception:
        return None, None, {}

    config = AutoConfig.from_pretrained(str(model_dir))
    vocab_candidates: list[int] = []
    for value in (
        getattr(config, "vocab_size", None),
        getattr(processor.tokenizer, "vocab_size", None),
    ):
        maybe = _coerce_optional_int(value)
        if maybe is not None:
            vocab_candidates.append(int(maybe))
    try:
        vocab_candidates.append(int(len(processor.tokenizer)))
    except Exception:
        pass
    if not vocab_candidates:
        raise RuntimeError("Unable to determine tokenizer vocab size for XGrammar.")
    vocab_size = max(vocab_candidates)

    tokenizer_info = xgr.TokenizerInfo.from_huggingface(processor.tokenizer, vocab_size=int(vocab_size))
    compiler = xgr.GrammarCompiler(tokenizer_info)
    compiled = {
        "mcq_choice": compiler.compile_grammar(MCQ_CHOICE_GRAMMAR),
    }
    return xgr, compiler, compiled


def _normalize_refdet_expected_count(value: Any) -> int | None:
    maybe = _coerce_optional_int(value)
    if maybe is None or int(maybe) <= 0:
        return None
    return int(maybe)


def _get_compiled_structured_output(
    *,
    compiler: Any,
    compiled_grammars: dict[str, Any],
    structured_output: Any,
) -> Any:
    if isinstance(structured_output, str):
        compiled = compiled_grammars.get(str(structured_output))
        if compiled is None:
            raise ValueError(f"Unsupported structured output mode: {structured_output}")
        return compiled

    if isinstance(structured_output, dict):
        mode = str(structured_output.get("type", "")).strip()
        if mode == "refdet_json_array":
            expected_count = _normalize_refdet_expected_count(structured_output.get("expected_count"))
            cache_key = f"refdet_json_array:{expected_count if expected_count is not None else 'open'}"
            compiled = compiled_grammars.get(cache_key)
            if compiled is None:
                compiled = compiler.compile_json_schema(_build_refdet_json_schema(expected_count))
                compiled_grammars[cache_key] = compiled
            return compiled

    raise ValueError(f"Unsupported structured output spec: {structured_output}")


def _build_xgrammar_logits_processor(
    xgr: Any | None,
    compiler: Any | None,
    compiled_grammars: dict[str, Any],
    structured_output: Any,
    *,
    override_stop_tokens: list[int] | None = None,
) -> Any | None:
    if structured_output is None:
        return None
    if xgr is None or compiler is None:
        raise RuntimeError(f"XGrammar is required for structured output mode: {structured_output}")

    if isinstance(structured_output, list):
        compiled_batch = [
            _get_compiled_structured_output(
                compiler=compiler,
                compiled_grammars=compiled_grammars,
                structured_output=item,
            )
            for item in structured_output
        ]
        return _XGrammarHFLogitsProcessor(
            xgr,
            compiled_batch,
            override_stop_tokens=override_stop_tokens,
        )

    compiled = _get_compiled_structured_output(
        compiler=compiler,
        compiled_grammars=compiled_grammars,
        structured_output=structured_output,
    )
    return _XGrammarHFLogitsProcessor(
        xgr,
        compiled,
        override_stop_tokens=override_stop_tokens,
    )


class _XGrammarHFLogitsProcessor:
    def __init__(
        self,
        xgr: Any,
        compiled_grammar: Any,
        *,
        override_stop_tokens: list[int] | None = None,
    ) -> None:
        self.xgr = xgr
        self.compiled_grammars = compiled_grammar if isinstance(compiled_grammar, list) else [compiled_grammar]
        self.override_stop_tokens = list(override_stop_tokens or [])
        self.matchers: list[Any] = []
        self.full_vocab_size = self.compiled_grammars[0].tokenizer_info.vocab_size
        self.token_bitmask = None
        self.prefilled = False
        self.batch_size = 0

    def __call__(self, input_ids, scores):
        if len(self.matchers) == 0:
            self.batch_size = int(input_ids.shape[0])
            self.compiled_grammars = (
                self.compiled_grammars
                if len(self.compiled_grammars) > 1
                else self.compiled_grammars * self.batch_size
            )
            assert len(self.compiled_grammars) == self.batch_size
            self.matchers = [
                self.xgr.GrammarMatcher(
                    self.compiled_grammars[i],
                    override_stop_tokens=self.override_stop_tokens or None,
                )
                for i in range(self.batch_size)
            ]
            self.token_bitmask = self.xgr.allocate_token_bitmask(self.batch_size, self.full_vocab_size)

        if int(input_ids.shape[0]) != int(self.batch_size):
            raise RuntimeError(
                "Expect input_ids.shape[0] to match XGrammar batch size. "
                f"Got {input_ids.shape[0]} vs {self.batch_size}."
            )

        if not self.prefilled:
            self.prefilled = True
        else:
            for i in range(self.batch_size):
                if not self.matchers[i].is_terminated():
                    sampled_token = int(input_ids[i][-1])
                    assert self.matchers[i].accept_token(sampled_token)

        for i in range(self.batch_size):
            if not self.matchers[i].is_terminated():
                self.matchers[i].fill_next_token_bitmask(self.token_bitmask, i)

        device_type = scores.device.type
        if device_type != "cuda":
            scores = scores.to("cpu")
        self.xgr.apply_token_bitmask_inplace(scores, self.token_bitmask.to(scores.device))
        if device_type != "cuda":
            scores = scores.to(device_type)
        return scores


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
        _configure_generation_special_tokens(self.processor, self.model, self.gen_kwargs)
        self._xgr, self._xgr_compiler, self._compiled_grammars = _init_xgrammar(self.processor, self.model_dir)
        self._xgrammar_stop_token_ids = _resolve_generation_stop_token_ids(self.model, self.processor, self.gen_kwargs)

    def generate_batch_conversations(
        self,
        conversations: list[list[dict[str, Any]]],
        *,
        structured_output: Any = None,
    ) -> list[GenerateResult]:
        try:
            import torch
        except Exception as e:
            raise RuntimeError("Missing dependency: torch") from e

        if not conversations:
            return []

        inputs = self.processor.apply_chat_template(
            conversations,
            tokenize=True,
            add_generation_prompt=True,
            padding=True,
            return_dict=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.model.device)

        generate_kwargs = dict(self.gen_kwargs)
        logits_processor = _build_xgrammar_logits_processor(
            self._xgr,
            self._xgr_compiler,
            self._compiled_grammars,
            structured_output,
            override_stop_tokens=self._xgrammar_stop_token_ids,
        )
        if logits_processor is not None:
            generate_kwargs["logits_processor"] = [logits_processor]

        with torch.no_grad():
            generated_ids = self.model.generate(**inputs, **generate_kwargs)

        input_ids = inputs.get("input_ids", None)
        if input_ids is not None:
            prompt_lens = [int(input_ids.shape[1])] * int(generated_ids.shape[0])
        else:
            attn = inputs.get("attention_mask", None)
            if attn is None:
                raise RuntimeError("Missing both input_ids and attention_mask while trimming generated ids.")
            prompt_lens = [int(attn.shape[1])] * int(generated_ids.shape[0])

        eos_token_id, pad_token_id = _resolve_summary_special_tokens(self.model, self.processor, self.gen_kwargs)

        summaries = _summarize_generated_sequences(
            generated_ids,
            prompt_lens,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
        )
        texts = self.processor.batch_decode(
            [item.token_ids for item in summaries],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

        return [
            GenerateResult(
                text=str(text).strip(),
                generated_token_count=int(item.generated_token_count),
                ended_by_eos=bool(item.ended_by_eos),
                last_generated_token_id=item.last_generated_token_id,
            )
            for text, item in zip(texts, summaries)
        ]


class Qwen35Runner:
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
        enable_thinking: bool = False,
    ) -> None:
        self.model_dir = Path(model_dir)
        self.device_map = str(device_map)
        self.dtype = str(dtype)
        self.max_new_tokens = int(max_new_tokens)
        self.enable_thinking = bool(enable_thinking)
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
            from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration  # type: ignore
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
        self.model = Qwen3_5ForConditionalGeneration.from_pretrained(
            str(self.model_dir),
            torch_dtype=torch_dtype,
            device_map=self.device_map,
        )
        self.model.eval()
        _configure_generation_special_tokens(self.processor, self.model, self.gen_kwargs)
        self._xgr, self._xgr_compiler, self._compiled_grammars = _init_xgrammar(self.processor, self.model_dir)
        self._xgrammar_stop_token_ids = _resolve_generation_stop_token_ids(self.model, self.processor, self.gen_kwargs)

    def _convert_conversation(self, conversation: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[Any]]:
        from PIL import Image

        converted_messages: list[dict[str, Any]] = []
        image_objects: list[Image.Image] = []
        for message in conversation:
            content_out: list[dict[str, Any]] = []
            for item in list(message.get("content", [])):
                item_type = str(item.get("type", "")).strip().lower()
                if item_type == "image":
                    image_value = item.get("image", "")
                    with Image.open(Path(str(image_value))) as opened:
                        rgb = opened.convert("RGB")
                    image_objects.append(rgb)
                    content_out.append({"type": "image", "image": rgb})
                else:
                    content_out.append({"type": "text", "text": str(item.get("text", ""))})
            converted_messages.append({"role": str(message.get("role", "user")), "content": content_out})
        return converted_messages, image_objects

    def _generate_one(self, conversation: list[dict[str, Any]]) -> GenerateResult:
        try:
            import torch
        except Exception as e:
            raise RuntimeError("Missing dependency: torch") from e

        converted_messages, image_objects = self._convert_conversation(conversation)
        rendered = self.processor.tokenizer.apply_chat_template(
            converted_messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=bool(self.enable_thinking),
        )
        inputs = self.processor(
            text=[str(rendered)],
            images=image_objects,
            padding=True,
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

        eos_token_id, pad_token_id = _resolve_summary_special_tokens(self.model, self.processor, self.gen_kwargs)

        summaries = _summarize_generated_sequences(
            generated_ids,
            prompt_lens,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
        )
        texts = self.processor.batch_decode(
            [item.token_ids for item in summaries],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        if not summaries:
            return GenerateResult(text="", generated_token_count=0, ended_by_eos=False, last_generated_token_id=None)
        item = summaries[0]
        text = texts[0] if texts else ""
        return GenerateResult(
            text=str(text).strip(),
            generated_token_count=int(item.generated_token_count),
            ended_by_eos=bool(item.ended_by_eos),
            last_generated_token_id=item.last_generated_token_id,
        )

    def generate_batch_conversations(
        self,
        conversations: list[list[dict[str, Any]]],
        *,
        structured_output: Any = None,
    ) -> list[GenerateResult]:
        try:
            import torch
        except Exception as e:
            raise RuntimeError("Missing dependency: torch") from e

        if not conversations:
            return []

        rendered_texts: list[str] = []
        image_batches: list[list[Any]] = []
        for conversation in conversations:
            converted_messages, image_objects = self._convert_conversation(conversation)
            rendered = self.processor.tokenizer.apply_chat_template(
                converted_messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=bool(self.enable_thinking),
            )
            rendered_texts.append(str(rendered))
            image_batches.append(image_objects)

        inputs = self.processor(
            text=rendered_texts,
            images=image_batches,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.model.device)

        generate_kwargs = dict(self.gen_kwargs)
        logits_processor = _build_xgrammar_logits_processor(
            self._xgr,
            self._xgr_compiler,
            self._compiled_grammars,
            structured_output,
            override_stop_tokens=self._xgrammar_stop_token_ids,
        )
        if logits_processor is not None:
            generate_kwargs["logits_processor"] = [logits_processor]

        with torch.no_grad():
            generated_ids = self.model.generate(**inputs, **generate_kwargs)

        input_ids = inputs.get("input_ids", None)
        if input_ids is not None:
            prompt_lens = [int(input_ids.shape[1])] * int(generated_ids.shape[0])
        else:
            attn = inputs.get("attention_mask", None)
            if attn is None:
                raise RuntimeError("Missing both input_ids and attention_mask while trimming generated ids.")
            prompt_lens = [int(attn.shape[1])] * int(generated_ids.shape[0])

        eos_token_id, pad_token_id = _resolve_summary_special_tokens(self.model, self.processor, self.gen_kwargs)

        summaries = _summarize_generated_sequences(
            generated_ids,
            prompt_lens,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
        )
        texts = self.processor.batch_decode(
            [item.token_ids for item in summaries],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        return [
            GenerateResult(
                text=str(text).strip(),
                generated_token_count=int(item.generated_token_count),
                ended_by_eos=bool(item.ended_by_eos),
                last_generated_token_id=item.last_generated_token_id,
            )
            for text, item in zip(texts, summaries)
        ]
