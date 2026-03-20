from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from PIL import Image

from ..shared.device import assert_model_on_cuda, require_cuda
from ..shared.qwen_dinov3 import build_generate_kwargs, maybe_set_generation_seed, summarize_generated_sequences, torch_dtype_from_str
from .sva_deepstack_ca_visual_adapter import (
    assert_sva_deepstack_ca_runtime_matches_merger,
    attach_sva_deepstack_ca_visual_adapter,
    load_sva_deepstack_ca_merger_safetensors,
)
from .vision_fixed_grid import compute_fixed_grid_resize


@dataclass(frozen=True)
class CaptionResult:
    text: str
    generated_token_count: int = 0
    ended_by_eos: bool = False
    last_generated_token_id: int | None = None


def _assert_safe_inference_device_map(device_map: str) -> None:
    raw = str(device_map).strip().lower()
    if raw == "cpu":
        raise ValueError("device_map=cpu is not allowed for this project.")
    if raw != "auto":
        return
    visible_cuda = int(torch.cuda.device_count())
    if visible_cuda > 1:
        raise RuntimeError(
            "Unsafe inference setup: device_map=auto with multiple visible CUDA devices. "
            "Expose exactly one GPU via CUDA_VISIBLE_DEVICES, or use the project shard wrapper."
        )


class SVADeepstackCACaptioner:
    """Qwen 原生视觉主路径 + DINOv3 局部 cross-attention 残差的推理封装。"""

    def __init__(
        self,
        *,
        qwen_model_dir: Path,
        dinov3_dir: Path,
        smart_resize_min_pixels: int = 256 * 256,
        smart_resize_max_pixels: int = 4096 * 4096,
        merger_ckpt: Path | None = None,
        lora_dir: Path | None = None,
        device_map: str = "auto",
        dtype: str = "bf16",
        max_new_tokens: int = 64,
        do_sample: bool | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        top_k: int | None = None,
        num_beams: int | None = None,
        repetition_penalty: float | None = None,
        no_repeat_ngram_size: int | None = None,
        merge_lora: bool = False,
        seed: int | None = None,
    ) -> None:
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

        require_cuda()
        _assert_safe_inference_device_map(str(device_map))

        self.qwen_model_dir = Path(qwen_model_dir)
        self.dinov3_dir = Path(dinov3_dir)
        self.smart_resize_min_pixels = int(smart_resize_min_pixels)
        self.smart_resize_max_pixels = int(smart_resize_max_pixels)
        self.max_new_tokens = int(max_new_tokens)
        self.generate_config = build_generate_kwargs(
            max_new_tokens=int(max_new_tokens),
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            num_beams=num_beams,
            repetition_penalty=repetition_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
        )
        self.decode_strategy = str(self.generate_config.strategy)
        maybe_set_generation_seed(seed)
        if self.smart_resize_min_pixels <= 0 or self.smart_resize_max_pixels < self.smart_resize_min_pixels:
            raise ValueError(
                "Invalid smart resize range: "
                f"min={self.smart_resize_min_pixels}, max={self.smart_resize_max_pixels}"
            )

        self.processor = AutoProcessor.from_pretrained(str(self.qwen_model_dir))
        try:
            self.processor.tokenizer.padding_side = "left"
        except Exception:
            pass
        self.image_processor = self.processor.image_processor
        self.tokenizer = self.processor.tokenizer
        self.image_token = str(getattr(self.processor, "image_token", "<|image_pad|>"))

        torch_dtype = torch_dtype_from_str(dtype)
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            str(self.qwen_model_dir),
            device_map=str(device_map),
            torch_dtype=torch_dtype,
        )
        self.model.eval()
        assert_model_on_cuda(self.model)
        self.patch_size = int(self.model.config.vision_config.patch_size)
        self.merge_size = int(self.model.config.vision_config.spatial_merge_size)

        attach_sva_deepstack_ca_visual_adapter(
            model=self.model,
            qwen_model_dir=self.qwen_model_dir,
            dinov3_dir=self.dinov3_dir,
        )

        if merger_ckpt is not None:
            assert_sva_deepstack_ca_runtime_matches_merger(
                qwen_model_dir=self.qwen_model_dir,
                dinov3_dir=self.dinov3_dir,
                smart_resize_min_pixels=int(self.smart_resize_min_pixels),
                smart_resize_max_pixels=int(self.smart_resize_max_pixels),
                merger_ckpt=Path(merger_ckpt),
                visual=self.model.model.visual,
            )
            load_sva_deepstack_ca_merger_safetensors(self.model, Path(merger_ckpt))

        if lora_dir is not None:
            from peft import PeftModel  # type: ignore

            self.model = PeftModel.from_pretrained(self.model, str(lora_dir))
            if bool(merge_lora):
                self.model = self.model.merge_and_unload()
            self.model.eval()
            assert_model_on_cuda(self.model)
        self.latent_grid_h = int(getattr(self.model.get_base_model().model.visual if hasattr(self.model, "get_base_model") else self.model.model.visual, "latent_grid_h", 16))
        self.latent_grid_w = int(getattr(self.model.get_base_model().model.visual if hasattr(self.model, "get_base_model") else self.model.model.visual, "latent_grid_w", 16))

    def caption(self, *, image_path: Path, prompt: str) -> CaptionResult:
        results = self.caption_batch_prompts(image_paths=[Path(image_path)], prompts=[str(prompt)])
        return results[0] if results else CaptionResult(text="")

    def caption_batch(self, *, image_paths: list[Path], prompt: str) -> list[CaptionResult]:
        return self.caption_batch_prompts(image_paths=image_paths, prompts=[str(prompt)] * int(len(image_paths)))

    def caption_batch_prompts(self, *, image_paths: list[Path], prompts: list[str]) -> list[CaptionResult]:
        image_paths = [Path(p) for p in image_paths]
        prompts = [str(p) for p in prompts]
        if not image_paths:
            return []
        if len(image_paths) != len(prompts):
            raise ValueError(f"image_paths and prompts must have same length, got {len(image_paths)} vs {len(prompts)}")

        def _load_rgb(p: Path) -> Image.Image:
            with Image.open(str(p)) as img:
                return img.convert("RGB")

        def _run_bucket(
            *,
            bucket_paths: list[Path],
            bucket_prompts: list[str],
            bucket_imgs: list[Image.Image],
            resized_h: int,
            resized_w: int,
            num_image_tokens: int,
            actual_grid_h: int,
            actual_grid_w: int,
        ) -> list[CaptionResult]:
            texts: list[str] = []
            for image_path, prompt in zip(bucket_paths, bucket_prompts):
                messages: list[dict[str, Any]] = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": str(image_path)},
                            {"type": "text", "text": str(prompt)},
                        ],
                    }
                ]
                text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                if text.count(self.image_token) != 1:
                    raise ValueError(
                        f"Expected exactly one {self.image_token} token, got {text.count(self.image_token)}"
                    )
                text = text.replace(self.image_token, self.image_token * int(num_image_tokens), 1)
                texts.append(text)

            text_inputs = self.tokenizer(texts, add_special_tokens=False, return_tensors="pt", padding=True)
            text_inputs = {k: v.to(self.model.device) for k, v in text_inputs.items()}

            resized_imgs = [img.resize((int(resized_w), int(resized_h)), Image.BICUBIC) for img in bucket_imgs]
            img_inputs = self.image_processor(
                images=resized_imgs,
                return_tensors="pt",
                do_resize=False,
            )
            pixel_values = img_inputs["pixel_values"].to(self.model.device, dtype=self.model.dtype)
            actual_image_grid_thw = img_inputs["image_grid_thw"].to(self.model.device, dtype=torch.long)
            if actual_image_grid_thw.ndim != 2 or int(actual_image_grid_thw.shape[0]) != int(len(resized_imgs)) or int(actual_image_grid_thw.shape[1]) != 3:
                raise ValueError(
                    "Unexpected actual_image_grid_thw shape from processor: "
                    f"got {tuple(actual_image_grid_thw.shape)}, batch={len(resized_imgs)}"
                )

            if not torch.all(actual_image_grid_thw[:, 1] == int(actual_grid_h)) or not torch.all(actual_image_grid_thw[:, 2] == int(actual_grid_w)):
                raise ValueError(
                    "actual_image_grid_thw mismatch with resized image shape: "
                    f"expected_grid=({actual_grid_h},{actual_grid_w}), got={actual_image_grid_thw.tolist()}"
                )

            fixed_image_grid_thw = torch.tensor(
                [[1, int(self.latent_grid_h), int(self.latent_grid_w)]],
                dtype=torch.long,
                device=self.model.device,
            ).repeat(int(len(resized_imgs)), 1)
            fixed_num_image_tokens = (fixed_image_grid_thw[:, 1] * fixed_image_grid_thw[:, 2]) // int(self.merge_size**2)
            if not torch.all(fixed_num_image_tokens == int(num_image_tokens)):
                raise ValueError(
                    "Expanded image token count mismatches processor output: "
                    f"expected={num_image_tokens}, got={fixed_num_image_tokens.tolist()}"
                )

            with torch.no_grad():
                generated_ids = self.model.generate(
                    **text_inputs,
                    pixel_values=pixel_values,
                    image_grid_thw=fixed_image_grid_thw,
                    actual_image_grid_thw=actual_image_grid_thw,
                    **self.generate_config.gen_kwargs,
                )

            input_ids = text_inputs.get("input_ids", None)
            if input_ids is not None:
                prompt_lens = [int(input_ids.shape[1])] * int(generated_ids.shape[0])
            else:
                attn = text_inputs.get("attention_mask", None)
                if attn is None:
                    raise RuntimeError("Missing both input_ids and attention_mask while trimming generated ids.")
                prompt_lens = [int(attn.shape[1])] * int(generated_ids.shape[0])

            generation_config = getattr(self.model, 'generation_config', None)
            eos_token_id = getattr(generation_config, 'eos_token_id', None)
            pad_token_id = getattr(generation_config, 'pad_token_id', None)
            if eos_token_id is None:
                eos_token_id = getattr(self.tokenizer, 'eos_token_id', None)
            if pad_token_id is None:
                pad_token_id = getattr(self.tokenizer, 'pad_token_id', None)

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

        imgs = [_load_rgb(p) for p in image_paths]
        resize_meta = [
            compute_fixed_grid_resize(
                height=int(img.height),
                width=int(img.width),
                patch_size=int(self.patch_size),
                latent_grid_h=int(self.latent_grid_h),
                latent_grid_w=int(self.latent_grid_w),
                merge_size=int(self.merge_size),
                min_pixels=int(self.smart_resize_min_pixels),
                max_pixels=int(self.smart_resize_max_pixels),
            )
            for img in imgs
        ]

        bucket_order: list[tuple[int, int, int, int, int]] = []
        bucket_to_indices: dict[tuple[int, int, int, int, int], list[int]] = {}
        for i, meta in enumerate(resize_meta):
            key = (
                int(meta.resized_height),
                int(meta.resized_width),
                int(meta.actual_grid_h),
                int(meta.actual_grid_w),
                int(meta.llm_image_tokens),
            )
            if key not in bucket_to_indices:
                bucket_to_indices[key] = []
                bucket_order.append(key)
            bucket_to_indices[key].append(i)

        out: list[CaptionResult | None] = [None for _ in image_paths]
        for key in bucket_order:
            resized_h, resized_w, actual_grid_h, actual_grid_w, num_image_tokens = key
            idxs = bucket_to_indices[key]
            bucket_paths = [image_paths[i] for i in idxs]
            bucket_prompts = [prompts[i] for i in idxs]
            bucket_imgs = [imgs[i] for i in idxs]
            bucket_out = _run_bucket(
                bucket_paths=bucket_paths,
                bucket_prompts=bucket_prompts,
                bucket_imgs=bucket_imgs,
                resized_h=int(resized_h),
                resized_w=int(resized_w),
                num_image_tokens=int(num_image_tokens),
                actual_grid_h=int(actual_grid_h),
                actual_grid_w=int(actual_grid_w),
            )
            for i, pred in zip(idxs, bucket_out):
                out[i] = pred

        if any(x is None for x in out):
            raise RuntimeError("Incomplete SVA deepstack CA caption outputs.")
        return [x for x in out if x is not None]
