from __future__ import annotations

import argparse
from pathlib import Path


DEFAULT_PROMPT = "Describe the image in detail."


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
    parser = argparse.ArgumentParser(description="VRSBench captioning (Qwen3-VL native visual + merger/LoRA).")
    parser.add_argument("--model-dir", type=str, default="models/Qwen3-VL-8B-Instruct")
    parser.add_argument("--merger-ckpt", type=str, default="", help="Merger safetensors path (optional).")
    parser.add_argument("--lora-dir", type=str, default="", help="LoRA directory (optional).")
    parser.add_argument("--merge-lora", action="store_true")
    parser.add_argument("--data", type=str, default="benchmark/vrsbench/data/vrsbench_images_test.jsonl")
    parser.add_argument("--output", type=str, default="benchmark/vrsbench/outputs/qwen_native_predictions_test.jsonl")
    parser.add_argument("--prompt", type=str, default=DEFAULT_PROMPT)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--do-sample", dest="do_sample", action="store_true", default=None)
    parser.add_argument("--no-sample", dest="do_sample", action="store_false")
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--num-beams", type=int, default=None)
    parser.add_argument("--repetition-penalty", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size for generation.")
    parser.add_argument("--device-map", type=str, default="auto")
    parser.add_argument("--dtype", type=str, default="auto", choices=["auto", "fp16", "bf16", "fp32"])
    parser.add_argument("--max-images", type=int, default=0, help="If >0, only run on the first N images.")
    parser.add_argument("--shard-world-size", type=int, default=1, help="Total shard workers for inference split.")
    parser.add_argument("--shard-rank", type=int, default=0, help="Current shard rank in [0, shard_world_size).")
    parser.add_argument("--shard-weights", type=str, default="", help="Optional shard weight ratio, e.g. 1:2.")
    args = parser.parse_args()

    import sys

    sys.path.insert(0, str(_project_root() / "src"))

    from tqdm import tqdm  # type: ignore

    from ftqwen.jsonl import append_jsonl, read_jsonl
    from ftqwen.qwen3_vl_native_captioner import Qwen3VLNativeCaptioner

    data_path = _resolve_from_project(args.data)
    if not data_path.is_file():
        raise FileNotFoundError(f"Missing prepared file: {data_path}. Run prepare_vrsbench_cap.py first.")

    images = read_jsonl(data_path)
    images = sorted(images, key=lambda x: int(x["imgid"]))
    if args.max_images and int(args.max_images) > 0:
        images = images[: int(args.max_images)]
    images = _slice_by_shard(
        images,
        world_size=int(args.shard_world_size),
        rank=int(args.shard_rank),
        weights=str(args.shard_weights),
        key_name="imgid",
    )

    out_path = _resolve_from_project(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    done_ids: set[int] = set()
    if out_path.exists():
        for row in read_jsonl(out_path):
            try:
                done_ids.add(int(row["imgid"]))
            except Exception:
                continue

    pending = [it for it in images if int(it["imgid"]) not in done_ids]
    if not pending:
        print(f"[OK] No pending images. Output already complete: {out_path}")
        return

    model_dir = _resolve_from_project(args.model_dir)
    merger_ckpt = _resolve_from_project(args.merger_ckpt) if str(args.merger_ckpt).strip() else None
    lora_dir = _resolve_from_project(args.lora_dir) if str(args.lora_dir).strip() else None

    captioner = Qwen3VLNativeCaptioner(
        model_dir,
        merger_ckpt=merger_ckpt,
        lora_dir=lora_dir,
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
    )

    import gc

    import torch

    max_batch_size = max(1, int(args.batch_size))
    cur_bs = int(max_batch_size)

    pbar = tqdm(total=len(pending), desc="caption:vrsbench:qwen-native")
    idx = 0
    while idx < len(pending):
        chunk = pending[idx : idx + cur_bs]
        image_paths = [_resolve_from_project(it["image_path"]) for it in chunk]
        effective_bs = int(len(chunk))
        try:
            preds = captioner.caption_batch(image_paths=image_paths, prompt=str(args.prompt))
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            gc.collect()
            if cur_bs <= 1:
                raise
            cur_bs = max(1, int(cur_bs) // 2)
            continue

        for it, pred in zip(chunk, preds):
            imgid = int(it["imgid"])
            append_jsonl(
                out_path,
                {
                    "imgid": imgid,
                    "filename": it.get("filename", ""),
                    "image_path": str(Path(it["image_path"])),
                    "prediction": pred.text,
                    "prompt": str(args.prompt),
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
