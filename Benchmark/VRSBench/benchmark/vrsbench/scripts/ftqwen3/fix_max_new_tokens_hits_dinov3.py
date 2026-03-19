from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def _project_root() -> Path:
    # benchmark/vrsbench/scripts/*.py -> parents[3] == 项目根目录
    return Path(__file__).resolve().parents[3]


def _resolve_from_project(path: str | Path) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    return (_project_root() / p).resolve()


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


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            rows.append(json.loads(ln))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


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


def _slice_by_shard(items: list[int], *, world_size: int, rank: int, weights: str) -> list[int]:
    if int(world_size) <= 0:
        raise ValueError(f"shard_world_size must be >=1, got {world_size}")
    if int(rank) < 0 or int(rank) >= int(world_size):
        raise ValueError(f"shard_rank out of range: rank={rank}, world_size={world_size}")

    parsed = _parse_shard_weights(weights, world_size=int(world_size))
    total = len(items)

    if parsed is None:
        return [it for i, it in enumerate(items) if (i % int(world_size)) == int(rank)]

    denom = int(sum(parsed))
    left = int(sum(parsed[: int(rank)]))
    right = int(sum(parsed[: int(rank) + 1]))
    start = (total * left) // denom
    end = (total * right) // denom
    return items[start:end]


def _token_len(tokenizer: Any, text: str) -> int:
    return len(tokenizer(str(text), add_special_tokens=False).input_ids)


def _safe_int(v: Any, default: int = -1) -> int:
    try:
        return int(v)
    except Exception:
        return default


def _row_hits_max_new_tokens(row: dict[str, Any], *, tokenizer: Any, max_new_tokens: int) -> tuple[bool, int, bool]:
    pred = str(row.get("prediction", "")).strip()
    if not pred:
        return True, 0, False

    ended_by_eos = row.get("generation_ended_by_eos", None)
    if isinstance(ended_by_eos, bool):
        generated_token_count = _safe_int(row.get("generated_token_count", 0), default=0)
        is_hit = (not bool(ended_by_eos)) and int(generated_token_count) >= int(max_new_tokens)
        return bool(is_hit), int(generated_token_count), False

    tok_len = _token_len(tokenizer, pred)
    return bool(tok_len >= int(max_new_tokens)), int(tok_len), True


@dataclass
class RetryItem:
    idx: int
    imgid: int
    image_path: Path
    last_pred: str
    last_tok_len: int
    last_ended_by_eos: bool = False
    last_generated_token_id: int | None = None
    attempt_used: int = 0
    success: bool = False


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Re-run DINOv3 captioning for samples that hit max_new_tokens, "
            "and write patch rows for later merge."
        )
    )
    parser.add_argument("--preds", type=str, required=True, help="Predictions jsonl to inspect.")
    parser.add_argument("--output", type=str, required=True, help="Patch output jsonl (rows to overwrite by imgid).")
    parser.add_argument("--merger-ckpt", type=str, required=True, help="Merger safetensors path.")
    parser.add_argument("--lora-dir", type=str, default="", help="LoRA directory (optional).")

    parser.add_argument("--qwen-model-dir", type=str, default="models/Qwen3-VL-8B-Instruct")
    parser.add_argument("--dinov3-dir", type=str, default="models/dinov3/dinov3-vitl16-pretrain-sat493m")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--smart-resize-min-pixels", type=int, default=None)
    parser.add_argument("--smart-resize-max-pixels", type=int, default=None)
    parser.add_argument("--dtype", type=str, default="bf16", choices=["auto", "fp16", "bf16", "fp32"])
    parser.add_argument("--device-map", type=str, default="auto")

    parser.add_argument("--prompt", type=str, default="Describe the image in detail.")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--max-retries", type=int, default=10, help="Maximum regenerate attempts per sample.")
    parser.add_argument("--batch-size", type=int, default=256, help="Initial retry batch size; OOM will auto halve.")
    parser.add_argument("--do-sample", dest="do_sample", action="store_true", default=None)
    parser.add_argument("--no-sample", dest="do_sample", action="store_false")
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--no-repeat-ngram-size", type=int, default=None)
    parser.add_argument("--repetition-penalty", type=float, default=None)

    parser.add_argument("--shard-world-size", type=int, default=2, help="Total shard workers for retry split.")
    parser.add_argument("--shard-rank", type=int, default=0, help="Current shard rank in [0, shard_world_size).")
    parser.add_argument("--shard-weights", type=str, default="", help="Optional shard weight ratio, e.g. 1:1.")
    parser.add_argument("--dry-run", action="store_true", help="Only report which imgids hit max_new_tokens.")
    args = parser.parse_args()

    import sys

    sys.path.insert(0, str(_project_root() / "src"))

    from transformers import AutoProcessor

    from ftqwen3.dinov3_captioner import DinoV3Captioner
    from ftqwen3.qwen_dinov3 import assert_dino_runtime_matches_merger, path_metadata_matches

    preds_path = _resolve_from_project(args.preds)
    if not preds_path.is_file():
        raise FileNotFoundError(f"Missing preds jsonl: {preds_path}")
    rows = read_jsonl(preds_path)
    if not rows:
        raise ValueError(f"Empty preds jsonl: {preds_path}")

    out_path = _resolve_from_project(args.output)
    shard_world_size = int(args.shard_world_size)
    shard_rank = int(args.shard_rank)
    shard_weights = str(args.shard_weights)

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

    file_qwen = str(rows[0].get("qwen_model_dir", "")).strip() if rows else ""
    if file_qwen and not path_metadata_matches(file_qwen, qwen_model_dir):
        raise ValueError(
            "Qwen model dir mismatches preds metadata. "
            f"preds expect={file_qwen}, got={qwen_model_dir}"
        )
    file_dinov3 = str(rows[0].get("dinov3_dir", "")).strip() if rows else ""
    if file_dinov3 and not path_metadata_matches(file_dinov3, dinov3_dir):
        raise ValueError(
            "DINOv3 dir mismatches preds metadata. "
            f"preds expect={file_dinov3}, got={dinov3_dir}"
        )
    file_image_size = rows[0].get("image_size", None) if rows else None
    if file_image_size is not None and int(file_image_size) != int(resize_cfg.image_size):
        raise ValueError(
            "image_size mismatches preds metadata. "
            f"preds expect={file_image_size}, got={resize_cfg.image_size}"
        )
    file_min_pixels = rows[0].get("smart_resize_min_pixels", None) if rows else None
    if file_min_pixels is not None and int(file_min_pixels) != int(resize_cfg.smart_resize_min_pixels):
        raise ValueError(
            "smart_resize_min_pixels mismatches preds metadata. "
            f"preds expect={file_min_pixels}, got={resize_cfg.smart_resize_min_pixels}"
        )
    file_max_pixels = rows[0].get("smart_resize_max_pixels", None) if rows else None
    if file_max_pixels is not None and int(file_max_pixels) != int(resize_cfg.smart_resize_max_pixels):
        raise ValueError(
            "smart_resize_max_pixels mismatches preds metadata. "
            f"preds expect={file_max_pixels}, got={resize_cfg.smart_resize_max_pixels}"
        )

    processor = AutoProcessor.from_pretrained(str(qwen_model_dir))
    tokenizer = processor.tokenizer

    max_new_tokens = int(args.max_new_tokens)
    max_retries = max(1, int(args.max_retries))
    requested_batch_size = max(1, int(args.batch_size))

    hit_idx: list[int] = []
    hit_imgids: list[int] = []
    fallback_rows = 0
    for i, row in enumerate(rows):
        is_hit, _, used_fallback = _row_hits_max_new_tokens(
            row,
            tokenizer=tokenizer,
            max_new_tokens=max_new_tokens,
        )
        if used_fallback:
            fallback_rows += 1
        if is_hit:
            hit_idx.append(i)
            hit_imgids.append(_safe_int(row.get("imgid"), default=-1))

    selected_hit_idx = _slice_by_shard(
        hit_idx,
        world_size=shard_world_size,
        rank=shard_rank,
        weights=shard_weights,
    )
    selected_hit_imgids: list[int] = []
    for idx in selected_hit_idx:
        imgid = _safe_int(rows[idx].get("imgid"), default=-1)
        if imgid >= 0:
            selected_hit_imgids.append(imgid)

    hit_imgids_sorted = [x for x in sorted(hit_imgids) if x >= 0]
    selected_hit_imgids_sorted = [x for x in sorted(selected_hit_imgids) if x >= 0]

    print(
        f"[retry-policy] file={preds_path} max_new_tokens={max_new_tokens} max_retries={max_retries} "
        f"requested_batch_size={requested_batch_size}"
    )
    print(f"[check] file={preds_path} total={len(rows)} hit(max_new_tokens_without_eos)={len(hit_idx)}")
    if hit_imgids_sorted:
        print(f"[check] hit_imgids(first 30)={hit_imgids_sorted[:30]}")
    if fallback_rows:
        print(f"[warn] rows_missing_generation_eos_metadata={fallback_rows}, fallback_to_token_length=1")
    print(
        f"[shard] rank={shard_rank}/{shard_world_size} weights={shard_weights or 'even'} "
        f"selected_hit={len(selected_hit_idx)}/{len(hit_idx)}"
    )
    if selected_hit_imgids_sorted:
        print(f"[shard] selected_hit_imgids(first 30)={selected_hit_imgids_sorted[:30]}")

    if bool(args.dry_run):
        write_jsonl(out_path, [])
        print(
            f"[retry-summary] file={preds_path} total={len(rows)} hit_before={len(hit_idx)} retried=0 "
            f"resolved=0 unresolved={len(selected_hit_idx)} max_retries={max_retries} dry_run=1"
        )
        print(f"[shard-output] wrote empty patch file: {out_path}")
        return

    if not selected_hit_idx:
        write_jsonl(out_path, [])
        print(f"[shard-output] no selected hit rows. wrote empty patch file: {out_path}")
        print(
            f"[retry-summary] file={preds_path} total={len(rows)} hit_before={len(hit_idx)} retried=0 "
            "resolved=0 unresolved=0 max_retries=0"
        )
        return

    captioner = DinoV3Captioner(
        qwen_model_dir=qwen_model_dir,
        dinov3_dir=dinov3_dir,
        image_size=int(resize_cfg.image_size),
        smart_resize_min_pixels=int(resize_cfg.smart_resize_min_pixels),
        smart_resize_max_pixels=int(resize_cfg.smart_resize_max_pixels),
        merger_ckpt=merger_ckpt,
        lora_dir=_resolve_from_project(args.lora_dir) if str(args.lora_dir).strip() else None,
        device_map=str(args.device_map),
        dtype=str(args.dtype),
        max_new_tokens=max_new_tokens,
        do_sample=args.do_sample,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        seed=args.seed,
        no_repeat_ngram_size=args.no_repeat_ngram_size,
        repetition_penalty=args.repetition_penalty,
        merge_lora=False,
    )

    import gc

    import torch

    prompt = str(args.prompt)

    selected_items: list[RetryItem] = []
    for i in selected_hit_idx:
        row = rows[i]
        imgid = _safe_int(row.get("imgid"), default=-1)
        imgid_for_log = str(imgid) if imgid >= 0 else str(row.get("imgid", "NA"))
        image_path = _resolve_from_project(str(row.get("image_path", "")))
        if not image_path.is_file():
            raise FileNotFoundError(f"Missing image_path for imgid={imgid_for_log}: {image_path}")

        if imgid < 0:
            raise ValueError(f"Missing valid imgid for shard patch row: {row}")

        last_pred = str(row.get("prediction", "")).strip()
        last_generated_token_count = _safe_int(row.get("generated_token_count", 0), default=0)
        last_ended_by_eos = bool(row.get("generation_ended_by_eos", False))
        last_generated_token_id = row.get("generation_last_token_id", None)
        selected_items.append(
            RetryItem(
                idx=i,
                imgid=imgid,
                image_path=image_path,
                last_pred=last_pred,
                last_tok_len=int(last_generated_token_count),
                last_ended_by_eos=bool(last_ended_by_eos),
                last_generated_token_id=_safe_int(last_generated_token_id, default=-1)
                if last_generated_token_id is not None
                else None,
            )
        )

    resolved = 0
    unresolved_imgids: list[int] = []

    cur_bs = int(requested_batch_size)
    remaining: list[RetryItem] = list(selected_items)
    for round_idx in range(1, max_retries + 1):
        if not remaining:
            break

        round_in = len(remaining)
        print(
            f"[retry-round] rank={shard_rank} round={round_idx}/{max_retries} "
            f"current_batch_size={cur_bs} remaining={round_in}"
        )

        next_remaining: list[RetryItem] = []
        idx = 0
        while idx < len(remaining):
            chunk = remaining[idx : idx + cur_bs]
            image_paths = [it.image_path for it in chunk]

            try:
                preds = captioner.caption_batch(image_paths=image_paths, prompt=prompt)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                gc.collect()
                if cur_bs <= 1:
                    raise
                new_bs = max(1, int(cur_bs) // 2)
                print(f"[oom] rank={shard_rank} reduce_batch_size {cur_bs}->{new_bs}")
                cur_bs = new_bs
                continue

            if len(preds) != len(chunk):
                raise RuntimeError(f"Prediction count mismatch: expected={len(chunk)} got={len(preds)}")

            for item, pred_obj in zip(chunk, preds):
                item.last_pred = str(pred_obj.text).strip()
                item.last_tok_len = int(pred_obj.generated_token_count)
                item.attempt_used += 1

                is_hit = int((not bool(pred_obj.ended_by_eos)) and int(pred_obj.generated_token_count) >= max_new_tokens)
                print(
                    f"[retry] imgid={item.imgid} attempt={item.attempt_used}/{max_retries} "
                    f"generated_token_count={int(pred_obj.generated_token_count)} "
                    f"ended_by_eos={int(bool(pred_obj.ended_by_eos))} hit={is_hit}"
                )

                if bool(pred_obj.ended_by_eos) and bool(item.last_pred):
                    item.success = True
                    item.last_tok_len = int(pred_obj.generated_token_count)
                    item.last_generated_token_id = pred_obj.last_generated_token_id
                    item.last_ended_by_eos = bool(pred_obj.ended_by_eos)
                    resolved += 1
                    continue

                if round_idx >= max_retries:
                    unresolved_imgids.append(item.imgid)
                    item.last_generated_token_id = pred_obj.last_generated_token_id
                    item.last_ended_by_eos = bool(pred_obj.ended_by_eos)
                    print(
                        f"[retry-fail] imgid={item.imgid} attempts={max_retries} "
                        f"final_generated_token_count={item.last_tok_len} "
                        f"ended_by_eos={int(bool(pred_obj.ended_by_eos))}"
                    )
                    continue

                item.last_generated_token_id = pred_obj.last_generated_token_id
                item.last_ended_by_eos = bool(pred_obj.ended_by_eos)
                next_remaining.append(item)

            idx += len(chunk)

        remaining = next_remaining
        done = len(selected_items) - len(remaining)
        print(
            f"[retry-round-end] rank={shard_rank} round={round_idx}/{max_retries} "
            f"done={done}/{len(selected_items)} remaining={len(remaining)} current_batch_size={cur_bs}"
        )

    patched_rows: list[dict[str, Any]] = []
    patched_seen: set[int] = set()
    for item in sorted(selected_items, key=lambda it: int(it.imgid)):
        if item.imgid in patched_seen:
            raise ValueError(f"Duplicated imgid in shard patch rows: {item.imgid}")
        patched_seen.add(item.imgid)

        patched_rows.append(
            {
                "imgid": int(item.imgid),
                "prediction": str(item.last_pred).strip(),
                "generated_token_count": int(item.last_tok_len),
                "generation_ended_by_eos": bool(getattr(item, "last_ended_by_eos", False)),
                "generation_last_token_id": getattr(item, "last_generated_token_id", None),
                "prompt": prompt,
                "max_new_tokens": max_new_tokens,
                "smart_resize_min_pixels": int(resize_cfg.smart_resize_min_pixels),
                "smart_resize_max_pixels": int(resize_cfg.smart_resize_max_pixels),
                "resize_mode": str(resize_cfg.mode),
                "decode_strategy": str(captioner.decode_strategy),
                "do_sample": args.do_sample,
                "temperature": args.temperature,
                "top_p": args.top_p,
                "top_k": args.top_k,
                "seed": args.seed,
                "no_repeat_ngram_size": args.no_repeat_ngram_size,
                "repetition_penalty": args.repetition_penalty,
                "max_token_retry_attempts": int(item.attempt_used),
                "max_token_retry_success": bool(item.success),
            }
        )

    unresolved = len(selected_hit_idx) - resolved
    write_jsonl(out_path, patched_rows)

    print(
        f"[retry-summary] file={preds_path} total={len(rows)} hit_before={len(hit_idx)} retried={len(selected_hit_idx)} "
        f"resolved={resolved} unresolved={unresolved} max_retries={max_retries}"
    )
    if unresolved_imgids:
        unresolved_imgids = sorted(set(unresolved_imgids))
        print(f"[retry-unresolved] imgids(first 30)={unresolved_imgids[:30]} count={len(unresolved_imgids)}")
    print(f"[shard-output] wrote patch rows={len(patched_rows)} -> {out_path}")


if __name__ == "__main__":
    main()
