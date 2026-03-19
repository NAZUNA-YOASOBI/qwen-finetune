from __future__ import annotations

import argparse
from pathlib import Path


DEFAULT_PROMPT = "Describe the image in detail in 2 to 4 sentences."


def _project_root() -> Path:
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
        f"[INFO] shard rank={rank}/{world_size} weights={weights or 'even'} selected={len(shard)}/{total} "
        f"first_{key_name}={first_key} last_{key_name}={last_key}",
        flush=True,
    )
    return shard


def main() -> None:
    parser = argparse.ArgumentParser(description="VRSBench captioning with Qwen3.5 baseline.")
    parser.add_argument("--model-dir", type=str, required=True)
    parser.add_argument("--data", type=str, default="benchmark/vrsbench/data/vrsbench_images_test.jsonl")
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--prompt", type=str, default=DEFAULT_PROMPT)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--do-sample", dest="do_sample", action="store_true")
    parser.add_argument("--no-sample", dest="do_sample", action="store_false")
    parser.set_defaults(do_sample=True)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--num-beams", type=int, default=None)
    parser.add_argument("--repetition-penalty", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--device-map", type=str, default="auto")
    parser.add_argument("--dtype", type=str, default="bf16", choices=["auto", "fp16", "bf16", "fp32"])
    parser.add_argument("--max-images", type=int, default=0)
    parser.add_argument("--shard-world-size", type=int, default=1)
    parser.add_argument("--shard-rank", type=int, default=0)
    parser.add_argument("--shard-weights", type=str, default="")
    args = parser.parse_args()

    import sys

    sys.path.insert(0, str(_project_root() / "src"))

    from tqdm import tqdm  # type: ignore

    from ftqwen35.jsonl import append_jsonl, read_jsonl
    from ftqwen35.qwen3_5_captioner import Qwen35Captioner

    data_path = _resolve_from_project(args.data)
    if not data_path.is_file():
        raise FileNotFoundError(f"Missing prepared file: {data_path}")

    images = read_jsonl(data_path)
    images = sorted(images, key=lambda x: int(x["imgid"]))
    if args.max_images and int(args.max_images) > 0:
        images = images[: int(args.max_images)]
    images = _slice_by_shard(images, world_size=int(args.shard_world_size), rank=int(args.shard_rank), weights=str(args.shard_weights), key_name="imgid")

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

    captioner = Qwen35Captioner(
        _resolve_from_project(args.model_dir),
        device_map=str(args.device_map),
        dtype=str(args.dtype),
        max_new_tokens=int(args.max_new_tokens),
        do_sample=bool(args.do_sample),
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        num_beams=args.num_beams,
        repetition_penalty=args.repetition_penalty,
        seed=args.seed,
        enable_thinking=False,
    )

    import gc
    import torch

    max_batch_size = max(1, int(args.batch_size))
    cur_bs = int(max_batch_size)

    pbar = tqdm(total=len(pending), desc="caption:vrsbench:qwen35")
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
            append_jsonl(
                out_path,
                {
                    "imgid": int(it["imgid"]),
                    "filename": it.get("filename", ""),
                    "image_path": str(Path(it["image_path"])),
                    "prediction": pred.text,
                    "generated_token_count": int(pred.generated_token_count),
                    "generation_ended_by_eos": bool(pred.ended_by_eos),
                    "generation_last_token_id": pred.last_generated_token_id,
                    "prompt": str(args.prompt),
                    "model_dir": str(Path(args.model_dir)),
                    "max_new_tokens": int(args.max_new_tokens),
                    "do_sample": bool(args.do_sample),
                    "temperature": args.temperature,
                    "top_p": args.top_p,
                    "top_k": args.top_k,
                    "num_beams": args.num_beams,
                    "repetition_penalty": args.repetition_penalty,
                    "seed": args.seed,
                    "enable_thinking": False,
                    "batch_size": int(effective_bs),
                    "requested_batch_size": int(max_batch_size),
                    "decode_strategy": str(captioner.decode_strategy),
                    "shard_world_size": int(args.shard_world_size),
                    "shard_rank": int(args.shard_rank),
                    "shard_weights": str(args.shard_weights),
                },
            )

        idx += len(chunk)
        pbar.update(len(chunk))
    pbar.close()
    print(f"[OK] Wrote predictions: {out_path}")


if __name__ == "__main__":
    main()
