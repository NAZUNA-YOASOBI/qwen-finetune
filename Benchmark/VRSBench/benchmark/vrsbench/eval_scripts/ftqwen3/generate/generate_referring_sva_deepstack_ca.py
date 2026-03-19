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
    # benchmark/vrsbench/eval_scripts/ftqwen*/<group>/*.py -> parents[5] == 项目根目录
    return Path(__file__).resolve().parents[5]


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
    parser = argparse.ArgumentParser(description="VRSBench referring (Qwen3-VL native + DINOv3 deepstack cross-attention SVA + merger/LoRA).")
    parser.add_argument("--qwen-model-dir", "--model-dir", dest="qwen_model_dir", type=str, default="models/Qwen3-VL-8B-Instruct")
    parser.add_argument("--dinov3-dir", type=str, default="models/dinov3/dinov3-vitl16-pretrain-sat493m")
    parser.add_argument("--smart-resize-min-pixels", type=int, default=256 * 256)
    parser.add_argument("--smart-resize-max-pixels", type=int, default=4096 * 4096)
    parser.add_argument("--merger-ckpt", type=str, required=True, help="Merger safetensors path.")
    parser.add_argument("--lora-dir", type=str, default="", help="LoRA directory (optional).")
    parser.add_argument("--merge-lora", action="store_true")
    parser.add_argument("--data", type=str, default="benchmark/vrsbench/data/vrsbench_referring_test.jsonl")
    parser.add_argument("--output", type=str, required=True, help="Where to write predictions jsonl (append/resume).")
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
    parser.add_argument("--no-repeat-ngram-size", type=int, default=None)
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

    from ftqwen3.jsonl import append_jsonl, read_jsonl
    from ftqwen3.sva_deepstack_ca_captioner import SVADeepstackCACaptioner

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

    import gc

    import torch

    qwen_model_dir = _resolve_from_project(args.qwen_model_dir)
    dinov3_dir = _resolve_from_project(args.dinov3_dir)
    captioner = SVADeepstackCACaptioner(
        qwen_model_dir=qwen_model_dir,
        dinov3_dir=dinov3_dir,
        smart_resize_min_pixels=int(args.smart_resize_min_pixels),
        smart_resize_max_pixels=int(args.smart_resize_max_pixels),
        merger_ckpt=_resolve_from_project(args.merger_ckpt),
        lora_dir=_resolve_from_project(args.lora_dir) if str(args.lora_dir).strip() else None,
        merge_lora=bool(args.merge_lora),
        device_map=str(args.device_map),
        dtype=str(args.dtype),
        max_new_tokens=int(args.max_new_tokens),
        do_sample=args.do_sample,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        num_beams=args.num_beams,
        repetition_penalty=args.repetition_penalty,
        seed=args.seed,
        no_repeat_ngram_size=args.no_repeat_ngram_size,
    )

    max_batch_size = max(1, int(args.batch_size))
    cur_bs = int(max_batch_size)

    pbar = tqdm(total=len(pending), desc="referring:vrsbench:sva-deepstack-ca")
    idx = 0
    while idx < len(pending):
        chunk = pending[idx : idx + cur_bs]

        prompts = []
        image_paths = []
        for it in chunk:
            q = str(it.get("question", "")).strip()
            prompt = str(args.prompt_template).format(ref_sentence=q)
            prompts.append(prompt)
            image_paths.append(_resolve_from_project(str(it["image_path"])))

        try:
            preds = captioner.caption_batch_prompts(image_paths=image_paths, prompts=prompts)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            gc.collect()
            if cur_bs <= 1:
                raise
            cur_bs = max(1, int(cur_bs) // 2)
            continue

        effective_bs = int(len(chunk))
        for it, prompt, pred in zip(chunk, prompts, preds):
            ans = pred.text
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
                    "image_size": None,
                    "smart_resize_min_pixels": int(args.smart_resize_min_pixels),
                    "smart_resize_max_pixels": int(args.smart_resize_max_pixels),
                    "latent_grid_h": int(getattr(captioner, "latent_grid_h", 16)),
                    "latent_grid_w": int(getattr(captioner, "latent_grid_w", 16)),
                    "resize_mode": "fixed_grid",
                    "max_new_tokens": int(args.max_new_tokens),
                    "do_sample": args.do_sample,
                    "temperature": args.temperature,
                    "top_p": args.top_p,
                    "top_k": args.top_k,
                    "num_beams": args.num_beams,
                    "repetition_penalty": args.repetition_penalty,
                    "seed": args.seed,
                    "no_repeat_ngram_size": args.no_repeat_ngram_size,
                    "batch_size": int(effective_bs),
                    "requested_batch_size": int(max_batch_size),
                    "decode_strategy": str(captioner.decode_strategy),
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
