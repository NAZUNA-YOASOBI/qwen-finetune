from __future__ import annotations

import argparse
import re
from pathlib import Path


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


def _is_valid_box_100(vals: list[int]) -> bool:
    return len(vals) == 4 and int(vals[0]) < int(vals[2]) and int(vals[1]) < int(vals[3])


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
        if not _is_valid_box_100(vals):
            return None, None
        return f"{{<{vals[0]}><{vals[1]}><{vals[2]}><{vals[3]}>}}", vals

    nums2 = _SIGNED_INT_RE.findall(t)
    if len(nums2) >= 4:
        vals = _clip_box_100([int(x) for x in nums2[:4]])
        if not _is_valid_box_100(vals):
            return None, None
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


def main() -> None:
    parser = argparse.ArgumentParser(description="VRSBench referring (Qwen3-VL native visual + merger/LoRA).")
    parser.add_argument("--model-dir", type=str, default="models/Qwen3-VL-8B-Instruct")
    parser.add_argument("--merger-ckpt", type=str, default="", help="Merger safetensors path (optional).")
    parser.add_argument("--lora-dir", type=str, default="", help="LoRA directory (optional).")
    parser.add_argument("--merge-lora", action="store_true")
    parser.add_argument("--data", type=str, default="benchmark/vrsbench/data/vrsbench_referring_test.jsonl")
    parser.add_argument("--output", type=str, default="benchmark/vrsbench/outputs/qwen_native_referring_predictions_test.jsonl")
    parser.add_argument("--prompt-template", type=str, default=DEFAULT_PROMPT_TEMPLATE)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--do-sample", dest="do_sample", action="store_true", default=None)
    parser.add_argument("--no-sample", dest="do_sample", action="store_false")
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--num-beams", type=int, default=None)
    parser.add_argument("--repetition-penalty", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--device-map", type=str, default="auto")
    parser.add_argument("--dtype", type=str, default="auto", choices=["auto", "fp16", "bf16", "fp32"])
    parser.add_argument("--max-items", type=int, default=0)
    parser.add_argument("--shard-world-size", type=int, default=1, help="Total shard workers for inference split.")
    parser.add_argument("--shard-rank", type=int, default=0, help="Current shard rank in [0, shard_world_size).")
    parser.add_argument("--shard-weights", type=str, default="", help="Optional shard weight ratio, e.g. 1:2.")
    args = parser.parse_args()

    import sys

    sys.path.insert(0, str(_project_root() / "src"))

    from tqdm import tqdm  # type: ignore

    from ftqwen.device import assert_model_on_cuda, require_cuda
    from ftqwen.jsonl import append_jsonl, read_jsonl
    from ftqwen.qwen_dinov3 import build_generate_kwargs, load_merger_safetensors, maybe_set_generation_seed, torch_dtype_from_str

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

    import gc

    import torch
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration  # type: ignore

    model_dir = _resolve_from_project(args.model_dir)
    gen_cfg = build_generate_kwargs(
        max_new_tokens=int(args.max_new_tokens),
        do_sample=args.do_sample,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        num_beams=args.num_beams,
        repetition_penalty=args.repetition_penalty,
    )
    maybe_set_generation_seed(args.seed)
    processor = AutoProcessor.from_pretrained(str(model_dir))
    try:
        processor.tokenizer.padding_side = "left"
    except Exception:
        pass

    torch_dtype = torch_dtype_from_str(str(args.dtype))
    try:
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            str(model_dir),
            dtype=torch_dtype,
            device_map=str(args.device_map),
        )
    except TypeError:
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            str(model_dir),
            torch_dtype=torch_dtype,
            device_map=str(args.device_map),
        )
    model.eval()
    assert_model_on_cuda(model)

    if str(args.merger_ckpt).strip():
        load_merger_safetensors(model, _resolve_from_project(args.merger_ckpt))

    if str(args.lora_dir).strip():
        from peft import PeftModel  # type: ignore

        model = PeftModel.from_pretrained(model, str(_resolve_from_project(args.lora_dir)))
        if bool(args.merge_lora):
            model = model.merge_and_unload()
        model.eval()
        assert_model_on_cuda(model)

    max_batch_size = max(1, int(args.batch_size))
    cur_bs = int(max_batch_size)

    pbar = tqdm(total=len(pending), desc="referring:vrsbench:qwen-native")
    idx = 0
    while idx < len(pending):
        chunk = pending[idx : idx + cur_bs]

        prompts = []
        conversations = []
        for it in chunk:
            q = str(it.get("question", "")).strip()
            prompt = str(args.prompt_template).format(ref_sentence=q)
            prompts.append(prompt)
            img_path = _resolve_from_project(str(it["image_path"]))
            conversations.append(
                [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": str(img_path)},
                            {"type": "text", "text": prompt},
                        ],
                    }
                ]
            )

        try:
            inputs = processor.apply_chat_template(
                conversations,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
                padding=True,
            )
            inputs = inputs.to(model.device)

            gen_kwargs = dict(gen_cfg.gen_kwargs)

            with torch.no_grad():
                generated_ids = model.generate(**inputs, **gen_kwargs)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            gc.collect()
            if cur_bs <= 1:
                raise
            cur_bs = max(1, int(cur_bs) // 2)
            continue

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
        texts = processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)

        effective_bs = int(len(chunk))
        for it, prompt, ans in zip(chunk, prompts, texts):
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
                    "model": "qwen3-vl-native",
                    "model_dir": str(Path(args.model_dir)),
                    "merger_ckpt": str(Path(args.merger_ckpt)) if str(args.merger_ckpt).strip() else "",
                    "lora_dir": str(Path(args.lora_dir)) if str(args.lora_dir).strip() else "",
                    "max_new_tokens": int(args.max_new_tokens),
                    "do_sample": args.do_sample,
                    "temperature": args.temperature,
                    "top_p": args.top_p,
                    "top_k": args.top_k,
                    "num_beams": args.num_beams,
                    "repetition_penalty": args.repetition_penalty,
                    "seed": args.seed,
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
