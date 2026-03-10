from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


DEFAULT_PROMPT_TEMPLATE = (
    "[refer] where can I locate the <p>{ref_sentence}</p>? "
    "Return exactly one box in this format: {{<x1><y1><x2><y2>}}. "
    "Output integers in [0,100] only. "
    "If you internally use [0,1000], divide each coordinate by 10 and round to nearest integer before output. "
    "Hard rules: 0<=x1<x2<=100, 0<=y1<y2<=100. Return one line only and nothing else."
)

_SIGNED_INT_RE = re.compile(r"-?\d+")
_ANGLE_SIGNED_INT_RE = re.compile(r"<\s*(-?\d+)\s*>")


def _clip_box_100(vals: list[int]) -> list[int]:
    return [max(0, min(100, int(v))) for v in vals]


def _extract_box(text: str) -> tuple[str | None, list[int] | None]:
    """
    从模型输出中提取一个 bbox（4 个整数），并规范化成 `{<x1><y1><x2><y2>}` 字符串。
    - 优先匹配 `<num>` 形式（与标注一致）
    - 否则回退到“所有数字的前 4 个”（避免输出中包含提示数字时污染前缀）
    """

    t = str(text or "")
    nums = _ANGLE_SIGNED_INT_RE.findall(t)
    if len(nums) >= 4:
        vals = _clip_box_100([int(x) for x in nums[:4]])
        return f"{{<{vals[0]}><{vals[1]}><{vals[2]}><{vals[3]}>}}", vals

    nums2 = _SIGNED_INT_RE.findall(t)
    if len(nums2) >= 4:
        vals = _clip_box_100([int(x) for x in nums2[:4]])
        return f"{{<{vals[0]}><{vals[1]}><{vals[2]}><{vals[3]}>}}", vals

    return None, None


def _project_root() -> Path:
    # benchmark/vrsbench/scripts/*.py -> parents[3] == 项目根目录
    return Path(__file__).resolve().parents[3]


def _resolve_from_project(path: str | Path) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    return (_project_root() / p).resolve()


def _parse_shard_weights(weights: str, *, world_size: int) -> list[int] | None:
    w = str(weights).strip()
    if not w:
        return None
    vals = [int(x.strip()) for x in w.split(":") if x.strip()]
    if len(vals) != int(world_size):
        raise ValueError(f"shard_weights expects {world_size} values, got {len(vals)}: {weights}")
    if any(v <= 0 for v in vals):
        raise ValueError(f"shard_weights must be positive integers: {weights}")
    return vals


def _slice_by_shard(items: list[dict], *, world_size: int, rank: int, weights: str, key_name: str) -> list[dict]:
    if int(world_size) <= 0:
        raise ValueError(f"shard_world_size must be >=1, got {world_size}")
    if int(rank) < 0 or int(rank) >= int(world_size):
        raise ValueError(f"shard_rank out of range: rank={rank}, world_size={world_size}")

    parsed = _parse_shard_weights(weights, world_size=int(world_size))
    total = len(items)

    if parsed is None:
        shard = [it for i, it in enumerate(items) if (i % int(world_size)) == int(rank)]
    else:
        denom = int(sum(parsed))
        left = int(sum(parsed[: int(rank)]))
        right = int(sum(parsed[: int(rank) + 1]))
        start = (total * left) // denom
        end = (total * right) // denom
        shard = items[start:end]

    first_key = shard[0].get(key_name, "") if shard else ""
    last_key = shard[-1].get(key_name, "") if shard else ""
    print(
        f"[INFO] shard rank={rank}/{world_size} weights={weights or 'even'} "
        f"selected={len(shard)}/{total} first_{key_name}={first_key} last_{key_name}={last_key}",
        flush=True,
    )
    return shard


def _read_merger_expected_qwen(merger_ckpt: Path) -> str | None:
    meta_path = Path(merger_ckpt).with_suffix(".json")
    if not meta_path.is_file():
        return None
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if isinstance(meta.get("run"), dict):
        qwen = str(meta["run"].get("qwen_model_dir", "")).strip()
        if qwen:
            return qwen
    qwen = str(meta.get("qwen_model_dir", "")).strip()
    return qwen if qwen else None


def _assert_qwen_matches_merger(*, qwen_model_dir: Path, merger_ckpt: Path) -> None:
    expected_qwen = _read_merger_expected_qwen(merger_ckpt)
    if not expected_qwen:
        return

    expected_path = _resolve_from_project(expected_qwen)
    actual_path = Path(qwen_model_dir)
    if expected_path.exists() and actual_path.exists():
        ok = expected_path.resolve() == actual_path.resolve()
    else:
        # 跨机器时绝对路径可能变化，退化为目录名校验，避免明显误配。
        ok = expected_path.name == actual_path.name
    if not ok:
        raise ValueError(
            "Qwen model dir mismatches merger checkpoint metadata. "
            f"expected={expected_qwen}, got={actual_path}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="VRSBench referring (DINOv3 visual adapter).")
    parser.add_argument("--qwen-model-dir", type=str, default="models/Qwen3-VL-8B-Instruct")
    parser.add_argument("--dinov3-dir", type=str, default="models/dinov3/dinov3-vitl16-pretrain-sat493m")
    parser.add_argument("--merger-ckpt", type=str, required=True)
    parser.add_argument("--lora-dir", type=str, default="")
    parser.add_argument("--merge-lora", action="store_true")

    parser.add_argument("--data", type=str, default="benchmark/vrsbench/data/vrsbench_referring_test.jsonl")
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--prompt-template", type=str, default=DEFAULT_PROMPT_TEMPLATE)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--smart-resize-min-pixels", type=int, default=None)
    parser.add_argument("--smart-resize-max-pixels", type=int, default=None)

    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--do-sample", dest="do_sample", action="store_true", default=None)
    parser.add_argument("--no-sample", dest="do_sample", action="store_false")
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--num-beams", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--no-repeat-ngram-size", type=int, default=None)
    parser.add_argument("--repetition-penalty", type=float, default=None)

    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--device-map", type=str, default="auto")
    parser.add_argument("--dtype", type=str, default="bf16", choices=["auto", "fp16", "bf16", "fp32"])
    parser.add_argument("--max-items", type=int, default=0)
    parser.add_argument("--shard-world-size", type=int, default=1, help="Total shard workers for inference split.")
    parser.add_argument("--shard-rank", type=int, default=0, help="Current shard rank in [0, shard_world_size).")
    parser.add_argument("--shard-weights", type=str, default="", help="Optional shard weight ratio, e.g. 1:2.")
    args = parser.parse_args()

    import sys

    sys.path.insert(0, str(_project_root() / "src"))

    from tqdm import tqdm  # type: ignore

    from ftqwen.device import assert_model_on_cuda, require_cuda
    from ftqwen.dinov3_adapter import DinoV3AdapterConfig, DinoV3VisualAdapter
    from ftqwen.jsonl import append_jsonl, read_jsonl
    from ftqwen.qwen_dinov3 import (
        assert_dino_runtime_matches_merger,
        build_generate_kwargs,
        load_merger_safetensors,
        maybe_set_generation_seed,
        torch_dtype_from_str,
    )
    from ftqwen.vision_resize import compute_vision_resize

    data_path = _resolve_from_project(args.data)
    if not data_path.is_file():
        raise FileNotFoundError(f"Missing prepared file: {data_path}. Run prepare_vrsbench_referring.py first.")

    rows = read_jsonl(data_path)
    rows = sorted(rows, key=lambda x: int(x["qid"]))
    if args.max_items and int(args.max_items) > 0:
        rows = rows[: int(args.max_items)]
    rows = _slice_by_shard(
        rows,
        world_size=int(args.shard_world_size),
        rank=int(args.shard_rank),
        weights=str(args.shard_weights),
        key_name="qid",
    )

    out_path = _resolve_from_project(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    done: set[int] = set()
    if out_path.exists():
        for r in read_jsonl(out_path):
            try:
                done.add(int(r.get("qid")))
            except Exception:
                continue

    pending = [r for r in rows if int(r["qid"]) not in done]
    if not pending:
        print(f"[OK] No pending samples. Output already complete: {out_path}")
        return

    require_cuda()

    import torch
    from PIL import Image
    from transformers import AutoImageProcessor, AutoProcessor, Qwen3VLForConditionalGeneration  # type: ignore

    qwen_model_dir = _resolve_from_project(args.qwen_model_dir)
    dinov3_dir = _resolve_from_project(args.dinov3_dir)
    merger_ckpt = _resolve_from_project(args.merger_ckpt)
    resize_cfg = assert_dino_runtime_matches_merger(
        qwen_model_dir=qwen_model_dir,
        dinov3_dir=dinov3_dir,
        image_size=int(args.image_size),
        smart_resize_min_pixels=args.smart_resize_min_pixels,
        smart_resize_max_pixels=args.smart_resize_max_pixels,
        merger_ckpt=merger_ckpt,
    )
    gen_cfg = build_generate_kwargs(
        max_new_tokens=int(args.max_new_tokens),
        do_sample=args.do_sample,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        num_beams=args.num_beams,
        repetition_penalty=args.repetition_penalty,
        no_repeat_ngram_size=args.no_repeat_ngram_size,
    )
    maybe_set_generation_seed(args.seed)

    processor = AutoProcessor.from_pretrained(str(qwen_model_dir))
    tokenizer = processor.tokenizer
    # decoder-only 模型 batch 推理建议用 left padding，避免 right padding 造成生成偏差。
    try:
        tokenizer.padding_side = "left"
    except Exception:
        pass
    image_processor = AutoImageProcessor.from_pretrained(str(dinov3_dir))

    torch_dtype = torch_dtype_from_str(str(args.dtype))
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        str(qwen_model_dir),
        device_map=str(args.device_map),
        torch_dtype=torch_dtype,
    )
    model.eval()
    assert_model_on_cuda(model)

    # 附加 DINOv3 视觉适配器（复用原模型里的 merger 与 deepstack_merger_list）
    old_visual = model.model.visual
    adapter_cfg = DinoV3AdapterConfig(
        dinov3_dir=dinov3_dir,
        image_size=int(resize_cfg.image_size),
        merge_size=int(old_visual.spatial_merge_size),
        deepstack_visual_indexes=tuple(int(x) for x in getattr(old_visual, "deepstack_visual_indexes", (5, 11, 17))),
        qwen_vision_depth=int(getattr(getattr(old_visual, "config", None), "depth", 0) or len(getattr(old_visual, "blocks", []))),
    )
    adapter = DinoV3VisualAdapter(
        adapter_cfg,
        merger=old_visual.merger,
        deepstack_merger_list=getattr(old_visual, "deepstack_merger_list", None),
        torch_dtype=model.dtype,
    )
    adapter = adapter.to(model.device)
    model.model.visual = adapter

    # 加载训练后的 merger 权重（包含 deepstack_merger_list）
    load_merger_safetensors(model, merger_ckpt)

    # 可选加载 LoRA
    if str(args.lora_dir).strip():
        from peft import PeftModel  # type: ignore

        model = PeftModel.from_pretrained(model, str(_resolve_from_project(args.lora_dir)))
        if bool(args.merge_lora):
            model = model.merge_and_unload()
        model.eval()
        assert_model_on_cuda(model)

    patch_size = int(model.config.vision_config.patch_size)
    merge_size = int(model.config.vision_config.spatial_merge_size)

    def _load_rgb(p: Path) -> Image.Image:
        # 避免残留打开的文件句柄
        with Image.open(str(p)) as img:
            return img.convert("RGB")

    import gc

    max_batch_size = max(1, int(args.batch_size))
    cur_bs = int(max_batch_size)

    pbar = tqdm(total=len(pending), desc="referring:vrsbench:dinov3")
    idx = 0
    while idx < len(pending):
        chunk = pending[idx : idx + cur_bs]

        prompts: list[str] = []
        image_paths: list[Path] = []
        for it in chunk:
            q = str(it.get("question", "")).strip()
            prompt = str(args.prompt_template).format(ref_sentence=q)
            prompts.append(prompt)

            img_path = _resolve_from_project(str(it["image_path"]))
            image_paths.append(img_path)

        effective_bs = int(len(chunk))
        try:
            imgs = [_load_rgb(p) for p in image_paths]
            resize_meta = [
                compute_vision_resize(
                    height=int(img.height),
                    width=int(img.width),
                    patch_size=int(patch_size),
                    merge_size=int(merge_size),
                    min_pixels=int(resize_cfg.smart_resize_min_pixels),
                    max_pixels=int(resize_cfg.smart_resize_max_pixels),
                )
                for img in imgs
            ]

            gen_kwargs: dict[str, Any] = dict(gen_cfg.gen_kwargs)

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

            out_texts: list[str | None] = [None for _ in chunk]
            for key in bucket_order:
                resized_h, resized_w, grid_h, grid_w, num_image_tokens = key
                idxs = bucket_to_indices[key]

                bucket_imgs = [imgs[i] for i in idxs]
                bucket_paths = [image_paths[i] for i in idxs]
                bucket_prompts = [prompts[i] for i in idxs]
                bucket_bs = int(len(idxs))

                texts: list[str] = []
                for img_path, prompt in zip(bucket_paths, bucket_prompts):
                    messages: list[dict[str, Any]] = [
                        {
                            "role": "user",
                            "content": [
                                {"type": "image", "image": str(img_path)},
                                {"type": "text", "text": prompt},
                            ],
                        }
                    ]
                    t = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                    if t.count("<|image_pad|>") != 1:
                        raise ValueError(f"Expected exactly one <|image_pad|> token, got {t.count('<|image_pad|>')}")
                    t = t.replace("<|image_pad|>", "<|image_pad|>" * int(num_image_tokens), 1)
                    texts.append(t)

                text_inputs = tokenizer(texts, add_special_tokens=False, return_tensors="pt", padding=True)
                text_inputs = {k: v.to(model.device) for k, v in text_inputs.items()}

                img_inputs = image_processor(
                    images=bucket_imgs,
                    return_tensors="pt",
                    size={"height": int(resized_h), "width": int(resized_w)},
                )
                pixel_values = img_inputs["pixel_values"].to(model.device, dtype=getattr(model, "dtype", None) or torch.float16)

                grid = torch.tensor(
                    [[1, int(grid_h), int(grid_w)]],
                    dtype=torch.long,
                    device=model.device,
                ).repeat(bucket_bs, 1)

                with torch.no_grad():
                    generated_ids = model.generate(
                        **text_inputs,
                        pixel_values=pixel_values,
                        image_grid_thw=grid,
                        **gen_kwargs,
                    )

                # decoder-only generate 返回的是“输入前缀 + 新生成 token”，
                # 切分点应使用补齐后的输入长度，而不是 attention_mask 的求和。
                input_ids = text_inputs.get("input_ids", None)
                if input_ids is not None:
                    prompt_len = int(input_ids.shape[1])
                    prompt_lens = [prompt_len] * int(generated_ids.shape[0])
                else:
                    attn = text_inputs.get("attention_mask", None)
                    if attn is None:
                        raise RuntimeError("Missing both input_ids and attention_mask while trimming generated ids.")
                    prompt_len = int(attn.shape[1])
                    prompt_lens = [prompt_len] * int(generated_ids.shape[0])

                trimmed = [out[int(pl):] for out, pl in zip(generated_ids, prompt_lens)]
                bucket_out_texts = processor.batch_decode(
                    trimmed,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )
                if len(bucket_out_texts) != len(idxs):
                    raise RuntimeError(
                        f"Bucket decode size mismatch: expected={len(idxs)} got={len(bucket_out_texts)}"
                    )
                for local_i, text_out in zip(idxs, bucket_out_texts):
                    out_texts[local_i] = str(text_out)

            if any(x is None for x in out_texts):
                raise RuntimeError("Internal error: missing predictions after smart-resize bucketing.")
            out_texts = [str(x) for x in out_texts]
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            gc.collect()
            if cur_bs <= 1:
                raise
            cur_bs = max(1, int(cur_bs) // 2)
            continue

        for it, prompt, ans in zip(chunk, prompts, out_texts):
            clean_box, clean_vals = _extract_box(str(ans))
            append_jsonl(
                out_path,
                {
                    "qid": int(it["qid"]),
                    "image_id": str(it.get("image_id", it.get("filename", ""))),
                    "filename": str(it.get("filename", "")),
                    "image_path": str(Path(it.get("image_path", ""))),
                    "question": str(it.get("question", "")),
                    "ground_truth": str(it.get("ground_truth", "")),
                    "is_unique": bool(it.get("is_unique", it.get("unique", False))),
                    "answer": str(clean_box or "").strip(),
                    "answer_parsed": clean_vals or [],
                    "answer_raw": str(ans).strip(),
                    "prompt": str(prompt),
                    "prompt_template": str(args.prompt_template),
                    "qwen_model_dir": str(Path(args.qwen_model_dir)),
                    "dinov3_dir": str(Path(args.dinov3_dir)),
                    "merger_ckpt": str(Path(args.merger_ckpt)),
                    "lora_dir": str(Path(args.lora_dir)) if str(args.lora_dir).strip() else "",
                    "max_new_tokens": int(args.max_new_tokens),
                    "image_size": int(resize_cfg.image_size),
                    "smart_resize_min_pixels": int(resize_cfg.smart_resize_min_pixels),
                    "smart_resize_max_pixels": int(resize_cfg.smart_resize_max_pixels),
                    "resize_mode": str(resize_cfg.mode),
                    "do_sample": args.do_sample,
                    "temperature": args.temperature,
                    "top_p": args.top_p,
                    "top_k": args.top_k,
                    "num_beams": args.num_beams,
                    "seed": args.seed,
                    "no_repeat_ngram_size": args.no_repeat_ngram_size,
                    "repetition_penalty": args.repetition_penalty,
                    "batch_size": int(effective_bs),
                    "requested_batch_size": int(max_batch_size),
                    "decode_strategy": str(gen_cfg.strategy),
                    "shard_world_size": int(args.shard_world_size),
                    "shard_rank": int(args.shard_rank),
                    "shard_weights": str(args.shard_weights),
                },
            )

        pbar.update(len(chunk))
        idx += len(chunk)

    pbar.close()
    print(f"[OK] Wrote predictions: {out_path}")


if __name__ == "__main__":
    main()
