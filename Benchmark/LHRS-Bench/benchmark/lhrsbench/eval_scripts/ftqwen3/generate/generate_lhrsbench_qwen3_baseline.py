from __future__ import annotations

import argparse
import gc
from pathlib import Path


def _project_root() -> Path:
    return Path(__file__).resolve().parents[5]


def _resolve_from_project(path: str | Path) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    return (_project_root() / p).resolve()


def _parse_shard_weights(weights: str, *, world_size: int) -> list[int] | None:
    value = str(weights).strip()
    if not value:
        return None
    out = [int(x.strip()) for x in value.split(":") if x.strip()]
    if len(out) != int(world_size):
        raise ValueError(f"shard_weights expects {world_size} values, got {len(out)}: {weights}")
    if any(x <= 0 for x in out):
        raise ValueError(f"shard_weights must be positive integers: {weights}")
    return out


def _slice_by_shard(items: list[dict], *, world_size: int, rank: int, weights: str) -> list[dict]:
    if int(world_size) <= 0:
        raise ValueError(f"shard_world_size must be >=1, got {world_size}")
    if int(rank) < 0 or int(rank) >= int(world_size):
        raise ValueError(f"shard_rank out of range: rank={rank}, world_size={world_size}")

    parsed = _parse_shard_weights(weights, world_size=int(world_size))
    total = len(items)

    if parsed is None:
        shard = [item for index, item in enumerate(items) if (index % int(world_size)) == int(rank)]
    else:
        denom = int(sum(parsed))
        left = int(sum(parsed[: int(rank)]))
        right = int(sum(parsed[: int(rank) + 1]))
        start = (total * left) // denom
        end = (total * right) // denom
        shard = items[start:end]

    first_uid = shard[0].get("uid", "") if shard else ""
    last_uid = shard[-1].get("uid", "") if shard else ""
    print(
        f"[INFO] shard rank={rank}/{world_size} weights={weights or 'even'} "
        f"selected={len(shard)}/{total} first_uid={first_uid} last_uid={last_uid}",
        flush=True,
    )
    return shard


def _is_pow2(n: int) -> bool:
    return int(n) > 0 and (int(n) & (int(n) - 1) == 0)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate LHRS-Bench predictions with native Qwen3-VL.")
    parser.add_argument("--model-dir", type=str, default="../../../VRSBench/models/Qwen3-VL-8B-Instruct")
    parser.add_argument("--data", type=str, default="benchmark/lhrsbench/data/lhrsbench_attempts_r4_seed42.jsonl")
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--model-tag", type=str, default="qwen3-vl-8b-baseline")
    parser.add_argument("--max-new-tokens", type=int, default=10)
    parser.add_argument("--do-sample", dest="do_sample", action="store_true", default=False)
    parser.add_argument("--no-sample", dest="do_sample", action="store_false")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--num-beams", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--device-map", type=str, default="auto")
    parser.add_argument("--dtype", type=str, default="bf16", choices=["auto", "fp16", "bf16", "fp32"])
    parser.add_argument("--shard-world-size", type=int, default=1)
    parser.add_argument("--shard-rank", type=int, default=0)
    parser.add_argument("--shard-weights", type=str, default="")
    args = parser.parse_args()

    if not _is_pow2(int(args.batch_size)):
        raise ValueError(f"batch_size must be power of 2, got {args.batch_size}")

    import sys

    sys.path.insert(0, str(_project_root() / "src"))

    import torch
    from tqdm import tqdm  # type: ignore

    from ftqwen3.baseline.qwen3_vl_captioner import Qwen3VLCaptioner
    from ftqwen3.shared.data_io import append_jsonl, read_jsonl

    data_path = _resolve_from_project(args.data)
    out_path = _resolve_from_project(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not data_path.is_file():
        raise FileNotFoundError(f"Missing data jsonl: {data_path}")

    rows = read_jsonl(data_path)
    rows = sorted(rows, key=lambda x: str(x.get("uid", "")))
    rows = _slice_by_shard(
        rows,
        world_size=int(args.shard_world_size),
        rank=int(args.shard_rank),
        weights=str(args.shard_weights),
    )

    done_uid: set[str] = set()
    if out_path.is_file():
        for row in read_jsonl(out_path):
            uid = str(row.get("uid", "")).strip()
            if uid:
                done_uid.add(uid)

    pending = [row for row in rows if str(row.get("uid", "")).strip() not in done_uid]
    if not pending:
        print(f"[OK] no pending samples, output already complete: {out_path}")
        return

    captioner = Qwen3VLCaptioner(
        model_dir=_resolve_from_project(args.model_dir),
        device_map=str(args.device_map),
        dtype=str(args.dtype),
        max_new_tokens=int(args.max_new_tokens),
        do_sample=bool(args.do_sample),
        temperature=float(args.temperature),
        top_p=float(args.top_p),
        top_k=int(args.top_k),
        num_beams=int(args.num_beams),
        repetition_penalty=float(args.repetition_penalty),
        seed=int(args.seed) if int(args.seed) > 0 else None,
    )

    max_batch_size = int(args.batch_size)
    current_batch_size = int(max_batch_size)
    index = 0

    pbar = tqdm(total=len(pending), desc="lhrsbench:qwen3")
    while index < len(pending):
        chunk = pending[index : index + current_batch_size]
        image_paths = [_resolve_from_project(row["image_path"]) for row in chunk]
        prompts = [str(row.get("prompt", "")).strip() for row in chunk]

        try:
            preds = captioner.caption_batch_prompts(image_paths=image_paths, prompts=prompts)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            gc.collect()
            if current_batch_size <= 1:
                raise
            current_batch_size = max(1, int(current_batch_size) // 2)
            print(f"[WARN] OOM -> reduce batch_size to {current_batch_size}", flush=True)
            continue

        for sample, pred in zip(chunk, preds):
            append_jsonl(
                out_path,
                {
                    "uid": str(sample["uid"]),
                    "qid": int(sample["qid"]),
                    "attempt": int(sample["attempt"]),
                    "filename": str(sample["filename"]),
                    "image_path": str(sample["image_path"]),
                    "question": str(sample["question"]),
                    "choices_shuffled": str(sample["choices_shuffled"]),
                    "options": sample["options"],
                    "answer_letter": str(sample["answer_letter"]),
                    "answer_text": str(sample["answer_text"]),
                    "type_ids": sample["type_ids"],
                    "type_names": sample["type_names"],
                    "prompt": str(sample["prompt"]),
                    "prediction": str(pred.text).strip(),
                    "generated_token_count": int(pred.generated_token_count),
                    "generation_ended_by_eos": bool(pred.ended_by_eos),
                    "generation_last_token_id": pred.last_generated_token_id,
                    "model_tag": str(args.model_tag),
                    "model_dir": str(Path(args.model_dir)),
                    "max_new_tokens": int(args.max_new_tokens),
                    "batch_size": int(len(chunk)),
                    "requested_batch_size": int(max_batch_size),
                    "effective_max_batch_size": int(current_batch_size),
                    "do_sample": bool(args.do_sample),
                    "temperature": float(args.temperature),
                    "top_p": float(args.top_p),
                    "top_k": int(args.top_k),
                    "num_beams": int(args.num_beams),
                    "seed": int(args.seed),
                    "repetition_penalty": float(args.repetition_penalty),
                    "decode_strategy": str(captioner.decode_strategy),
                    "shard_world_size": int(args.shard_world_size),
                    "shard_rank": int(args.shard_rank),
                    "shard_weights": str(args.shard_weights),
                },
            )

        index += len(chunk)
        pbar.update(len(chunk))
    pbar.close()

    print(f"[OK] wrote predictions: {out_path}")


if __name__ == "__main__":
    main()
