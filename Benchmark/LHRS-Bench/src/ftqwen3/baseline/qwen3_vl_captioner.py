from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..shared.device import assert_model_on_cuda, require_cuda
from ..shared.generation import (
    build_generate_kwargs,
    maybe_set_generation_seed,
    summarize_generated_sequences,
    torch_dtype_from_str,
)


@dataclass(frozen=True)
class CaptionResult:
    text: str
    generated_token_count: int = 0
    ended_by_eos: bool = False
    last_generated_token_id: int | None = None


class Qwen3VLCaptioner:
    def __init__(
        self,
        model_dir: Path,
        *,
        device_map: str = "auto",
        dtype: str = "auto",
        max_new_tokens: int = 64,
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
        self.generate_config = build_generate_kwargs(
            max_new_tokens=int(max_new_tokens),
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            num_beams=num_beams,
            repetition_penalty=repetition_penalty,
        )
        self.decode_strategy = str(self.generate_config.strategy)
        maybe_set_generation_seed(seed)

        try:
            import torch
            from transformers import AutoProcessor, Qwen3VLForConditionalGeneration  # type: ignore
        except Exception as e:
            raise RuntimeError("Missing dependency: transformers/torch") from e

        require_cuda()
        if str(self.device_map).strip().lower() == "cpu":
            raise ValueError("device_map=cpu is not allowed for this project.")

        torch_dtype = torch_dtype_from_str(torch, self.dtype)

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
        assert_model_on_cuda(self.model)

    def caption(self, *, image_path: Path, prompt: str) -> CaptionResult:
        results = self.caption_batch_prompts(image_paths=[Path(image_path)], prompts=[str(prompt)])
        return results[0] if results else CaptionResult(text="")

    def caption_batch(self, *, image_paths: list[Path], prompt: str) -> list[CaptionResult]:
        prompts = [str(prompt)] * int(len(image_paths))
        return self.caption_batch_prompts(image_paths=image_paths, prompts=prompts)

    def caption_batch_prompts(self, *, image_paths: list[Path], prompts: list[str]) -> list[CaptionResult]:
        try:
            import torch
        except Exception as e:
            raise RuntimeError("Missing dependency: torch") from e

        image_paths = [Path(p) for p in image_paths]
        prompts = [str(p) for p in prompts]
        if not image_paths:
            return []
        if len(image_paths) != len(prompts):
            raise ValueError(f"image_paths and prompts must have same length, got {len(image_paths)} vs {len(prompts)}")

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

        with torch.no_grad():
            generated_ids = self.model.generate(**inputs, **self.generate_config.gen_kwargs)

        input_ids = inputs.get("input_ids", None)
        if input_ids is not None:
            prompt_len = int(input_ids.shape[1])
            prompt_lens = [int(prompt_len)] * int(generated_ids.shape[0])
        else:
            attn = inputs.get("attention_mask", None)
            if attn is None:
                raise RuntimeError("Missing both input_ids and attention_mask while trimming generated ids.")
            prompt_len = int(attn.shape[1])
            prompt_lens = [int(prompt_len)] * int(generated_ids.shape[0])

        generation_config = getattr(self.model, "generation_config", None)
        eos_token_id = getattr(generation_config, "eos_token_id", None)
        pad_token_id = getattr(generation_config, "pad_token_id", None)
        if eos_token_id is None:
            eos_token_id = getattr(self.processor.tokenizer, "eos_token_id", None)
        if pad_token_id is None:
            pad_token_id = getattr(self.processor.tokenizer, "pad_token_id", None)

        sequence_summaries = summarize_generated_sequences(
            generated_ids,
            prompt_lens,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
        )
        texts = self.processor.batch_decode(
            [item.token_ids for item in sequence_summaries],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        return [
            CaptionResult(
                text=str(text).strip(),
                generated_token_count=int(item.generated_token_count),
                ended_by_eos=bool(item.ended_by_eos),
                last_generated_token_id=item.last_generated_token_id,
            )
            for text, item in zip(texts, sequence_summaries)
        ]
