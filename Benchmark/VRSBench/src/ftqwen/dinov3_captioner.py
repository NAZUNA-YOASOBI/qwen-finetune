from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from PIL import Image

from .dinov3_adapter import DinoV3AdapterConfig, DinoV3VisualAdapter
from .device import assert_model_on_cuda, require_cuda
from .qwen_dinov3 import (
    build_generate_kwargs,
    load_merger_safetensors,
    maybe_set_generation_seed,
    summarize_generated_sequences,
    torch_dtype_from_str,
)
from .vision_resize import compute_vision_resize


@dataclass(frozen=True)
class CaptionResult:
    text: str
    generated_token_count: int = 0
    ended_by_eos: bool = False
    last_generated_token_id: int | None = None


class DinoV3Captioner:
    def __init__(
        self,
        *,
        qwen_model_dir: Path,
        dinov3_dir: Path,
        image_size: int = 256,
        smart_resize_min_pixels: int = 256 * 256,
        smart_resize_max_pixels: int = 4096 * 4096,
        merger_ckpt: Path | None = None,
        lora_dir: Path | None = None,
        device_map: str = 'auto',
        dtype: str = 'bf16',
        max_new_tokens: int = 64,
        do_sample: bool | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        top_k: int | None = None,
        num_beams: int | None = None,
        seed: int | None = None,
        no_repeat_ngram_size: int | None = None,
        repetition_penalty: float | None = None,
        merge_lora: bool = False,
    ) -> None:
        from transformers import AutoImageProcessor, AutoProcessor, Qwen3VLForConditionalGeneration

        require_cuda()
        if str(device_map).strip().lower() == 'cpu':
            raise ValueError('device_map=cpu is not allowed for this project.')

        self.qwen_model_dir = Path(qwen_model_dir)
        self.dinov3_dir = Path(dinov3_dir)
        self.image_size = int(image_size)
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
                'Invalid smart resize range: '
                f'min={self.smart_resize_min_pixels}, max={self.smart_resize_max_pixels}'
            )

        torch_dtype = torch_dtype_from_str(dtype)

        self.processor = AutoProcessor.from_pretrained(str(self.qwen_model_dir))
        try:
            self.processor.tokenizer.padding_side = 'left'
        except Exception:
            pass
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            str(self.qwen_model_dir),
            device_map=str(device_map),
            torch_dtype=torch_dtype,
        )
        self.model.eval()
        assert_model_on_cuda(self.model)

        old_visual = self.model.model.visual
        cfg = DinoV3AdapterConfig(
            dinov3_dir=self.dinov3_dir,
            image_size=self.image_size,
            merge_size=int(old_visual.spatial_merge_size),
            deepstack_visual_indexes=tuple(int(x) for x in getattr(old_visual, 'deepstack_visual_indexes', (5, 11, 17))),
            qwen_vision_depth=int(getattr(getattr(old_visual, 'config', None), 'depth', 0) or len(getattr(old_visual, 'blocks', []))),
        )
        adapter = DinoV3VisualAdapter(
            cfg,
            merger=old_visual.merger,
            deepstack_merger_list=getattr(old_visual, 'deepstack_merger_list', None),
            torch_dtype=self.model.dtype,
        )
        adapter = adapter.to(self.model.device)
        self.model.model.visual = adapter

        if merger_ckpt is not None:
            load_merger_safetensors(self.model, Path(merger_ckpt))

        if lora_dir is not None:
            from peft import PeftModel  # type: ignore

            self.model = PeftModel.from_pretrained(self.model, str(lora_dir))
            if merge_lora:
                self.model = self.model.merge_and_unload()
            self.model.eval()
            assert_model_on_cuda(self.model)

        self.image_processor = AutoImageProcessor.from_pretrained(str(self.dinov3_dir))
        self.patch_size = int(self.model.config.vision_config.patch_size)
        self.merge_size = int(self.model.config.vision_config.spatial_merge_size)

    def caption(self, *, image_path: Path, prompt: str) -> CaptionResult:
        results = self.caption_batch_prompts(image_paths=[Path(image_path)], prompts=[str(prompt)])
        return results[0] if results else CaptionResult(text='')

    def caption_batch(self, *, image_paths: list[Path], prompt: str) -> list[CaptionResult]:
        prompts = [str(prompt)] * int(len(image_paths))
        return self.caption_batch_prompts(image_paths=image_paths, prompts=prompts)

    def caption_batch_prompts(self, *, image_paths: list[Path], prompts: list[str]) -> list[CaptionResult]:
        """按样本级 prompt 批量生成（DINOv3 视觉适配器）。

        智能缩放后按同尺寸分桶，确保每个子 batch 的像素尺寸与 grid 一致。
        """

        def _load_rgb(p: Path) -> Image.Image:
            with Image.open(str(p)) as img:
                return img.convert('RGB')

        def _run_bucket(
            *,
            bucket_paths: list[Path],
            bucket_prompts: list[str],
            bucket_imgs: list[Image.Image],
            resized_h: int,
            resized_w: int,
            grid_h: int,
            grid_w: int,
            num_image_tokens: int,
        ) -> list[CaptionResult]:
            tokenizer = self.processor.tokenizer
            texts: list[str] = []
            for image_path, prompt in zip(bucket_paths, bucket_prompts):
                messages: list[dict[str, Any]] = [
                    {
                        'role': 'user',
                        'content': [
                            {'type': 'image', 'image': str(image_path)},
                            {'type': 'text', 'text': str(prompt)},
                        ],
                    }
                ]
                text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                if text.count('<|image_pad|>') != 1:
                    raise ValueError(f"Expected exactly one <|image_pad|> token, got {text.count('<|image_pad|>')}")
                text = text.replace('<|image_pad|>', '<|image_pad|>' * int(num_image_tokens), 1)
                texts.append(text)

            text_inputs = tokenizer(texts, add_special_tokens=False, return_tensors='pt', padding=True)
            text_inputs = {k: v.to(self.model.device) for k, v in text_inputs.items()}

            img_inputs = self.image_processor(
                images=bucket_imgs,
                return_tensors='pt',
                size={'height': int(resized_h), 'width': int(resized_w)},
            )
            pixel_values = img_inputs['pixel_values'].to(self.model.device, dtype=self.model.dtype)

            image_grid_thw = torch.tensor([[1, int(grid_h), int(grid_w)]], dtype=torch.long, device=self.model.device).repeat(
                int(len(bucket_paths)), 1
            )

            with torch.no_grad():
                generated_ids = self.model.generate(
                    **text_inputs,
                    pixel_values=pixel_values,
                    image_grid_thw=image_grid_thw,
                    **self.generate_config.gen_kwargs,
                )

            input_ids = text_inputs.get('input_ids', None)
            if input_ids is not None:
                prompt_len = int(input_ids.shape[1])
                prompt_lens = [prompt_len] * int(generated_ids.shape[0])
            else:
                attn = text_inputs.get('attention_mask', None)
                if attn is None:
                    raise RuntimeError('Missing both input_ids and attention_mask while trimming generated ids.')
                prompt_len = int(attn.shape[1])
                prompt_lens = [prompt_len] * int(generated_ids.shape[0])

            generation_config = getattr(self.model, 'generation_config', None)
            eos_token_id = getattr(generation_config, 'eos_token_id', None)
            pad_token_id = getattr(generation_config, 'pad_token_id', None)
            if eos_token_id is None:
                eos_token_id = getattr(tokenizer, 'eos_token_id', None)
            if pad_token_id is None:
                pad_token_id = getattr(tokenizer, 'pad_token_id', None)

            sequence_summaries = summarize_generated_sequences(
                generated_ids,
                prompt_lens,
                eos_token_id=eos_token_id,
                pad_token_id=pad_token_id,
            )
            out_texts = self.processor.batch_decode(
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
                for text, item in zip(out_texts, sequence_summaries)
            ]

        image_paths = [Path(p) for p in image_paths]
        prompts = [str(p) for p in prompts]
        if not image_paths:
            return []
        if len(image_paths) != len(prompts):
            raise ValueError(f'image_paths and prompts must have same length, got {len(image_paths)} vs {len(prompts)}')

        imgs = [_load_rgb(p) for p in image_paths]
        resize_meta = [
            compute_vision_resize(
                height=int(img.height),
                width=int(img.width),
                patch_size=int(self.patch_size),
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
                int(meta.grid_h),
                int(meta.grid_w),
                int(meta.num_image_tokens),
            )
            if key not in bucket_to_indices:
                bucket_to_indices[key] = []
                bucket_order.append(key)
            bucket_to_indices[key].append(i)

        out: list[CaptionResult | None] = [None for _ in image_paths]
        for key in bucket_order:
            resized_h, resized_w, grid_h, grid_w, num_image_tokens = key
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
                grid_h=int(grid_h),
                grid_w=int(grid_w),
                num_image_tokens=int(num_image_tokens),
            )
            if len(bucket_out) != len(idxs):
                raise RuntimeError(f'Bucket output size mismatch: expect={len(idxs)} got={len(bucket_out)}')
            for j, pred in zip(idxs, bucket_out):
                out[j] = pred

        if any(x is None for x in out):
            raise RuntimeError('Internal error: missing predictions after smart-resize bucketing.')
        return [x for x in out if x is not None]
