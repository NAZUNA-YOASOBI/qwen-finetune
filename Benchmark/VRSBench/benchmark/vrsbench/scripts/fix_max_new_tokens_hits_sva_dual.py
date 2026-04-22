from __future__ import annotations

import argparse
import json
import re
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def _project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _resolve_from_project(path: str | Path) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    return (_project_root() / p).resolve()


def read_jsonl(path: Path, *, allow_trailing_partial: bool = False) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    raw_lines = path.read_text(encoding="utf-8").splitlines()
    non_empty_lines: list[tuple[int, str]] = []
    for line_no, ln in enumerate(raw_lines, start=1):
        stripped = ln.strip()
        if not stripped:
            continue
        non_empty_lines.append((int(line_no), stripped))

    for idx, (line_no, ln) in enumerate(non_empty_lines):
        try:
            rows.append(json.loads(ln))
        except json.JSONDecodeError:
            is_last_non_empty = int(idx) == int(len(non_empty_lines) - 1)
            if bool(allow_trailing_partial) and bool(is_last_non_empty):
                warnings.warn(
                    f"Skip trailing partial JSONL line in {path} at line {line_no}.",
                    stacklevel=2,
                )
                break
            raise
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


_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")


def limit_sentences(text: str, max_sentences: int) -> str:
    text = str(text).strip()
    if not text:
        return ""
    parts = [p.strip() for p in _SENT_SPLIT.split(text) if p.strip()]
    if len(parts) <= int(max_sentences):
        return " ".join(parts).strip()
    return " ".join(parts[: int(max_sentences)]).strip()


def ensure_sentence_end(text: str) -> str:
    text = str(text).strip()
    if not text:
        return ""
    if text[-1] in ".!?":
        return text
    return text + "."


def _token_len(tokenizer: Any, text: str) -> int:
    return len(tokenizer(str(text), add_special_tokens=False).input_ids)


def _safe_int(v: Any, default: int = -1) -> int:
    try:
        return int(v)
    except Exception:
        return default


@dataclass
class RetryItem:
    idx: int
    imgid: int
    image_path: Path
    last_pred: str
    last_tok_len: int
    attempt_used: int = 0
    success: bool = False


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Re-run SVA-dual captioning for samples that hit max_new_tokens, "
            "and write patch rows for later merge."
        )
    )
    parser.add_argument("--preds", type=str, required=True, help="Predictions jsonl to inspect.")
    parser.add_argument("--output", type=str, required=True, help="Patch output jsonl (rows to overwrite by imgid).")
    parser.add_argument("--model-dir", type=str, default="models/Qwen3-VL-8B-Instruct")
    parser.add_argument("--dinov3-dir", type=str, default="models/dinov3/dinov3-vitl16-pretrain-sat493m")
    parser.add_argument("--smart-resize-min-pixels", type=int, default=224 * 224)
    parser.add_argument("--smart-resize-max-pixels", type=int, default=512 * 512)
    parser.add_argument("--merger-ckpt", type=str, required=True, help="Merger safetensors path.")
    parser.add_argument("--lora-dir", type=str, default="", help="LoRA directory (optional).")
    parser.add_argument("--merge-lora", action="store_true")
    parser.add_argument("--device-map", type=str, default="cuda:0")
    parser.add_argument("--dtype", type=str, default="cuda:0", choices=["auto", "fp16", "bf16", "fp32"])

    parser.add_argument("--prompt", type=str, default="Describe the image in detail.")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--max-retries", type=int, default=10, help="Maximum regenerate rounds.")
    parser.add_argument("--batch-size", type=int, default=256, help="Initial retry batch size; OOM will auto halve.")
    parser.add_argument("--do-sample", dest="do_sample", action="store_true", default=None)
    parser.add_argument("--no-sample", dest="do_sample", action="store_false")
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--num-beams", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--no-repeat-ngram-size", type=int, default=None)
    parser.add_argument("--repetition-penalty", type=float, default=None)

    parser.add_argument("--max-sentences", type=int, default=7, help="Post-trim to at most N sentences.")
    parser.add_argument("--shard-world-size", type=int, default=2, help="Total shard workers for retry split.")
    parser.add_argument("--shard-rank", type=int, default=0, help="Current shard rank in [0, shard_world_size).")
    parser.add_argument("--shard-weights", type=str, default="", help="Optional shard weight ratio, e.g. 1:1.")
    parser.add_argument("--dry-run", action="store_true", help="Only report which imgids hit max_new_tokens.")
    args = parser.parse_args()

    import sys

    sys.path.insert(0, str(_project_root() / "src"))

    from transformers import AutoProcessor

    from ftqwen3.qwen_dinov3 import path_metadata_matches
    from ftqwen3.sva_captioner import SVADualVisualCaptioner
    from ftqwen3.sva_dual_visual_adapter import assert_sva_runtime_matches_merger

    preds_path = _resolve_from_project(args.preds)
    if not preds_path.is_file():
        raise FileNotFoundError(f"Missing preds jsonl: {preds_path}")
    rows = read_jsonl(preds_path, allow_trailing_partial=True)
    if not rows:
        raise ValueError(f"Empty preds jsonl: {preds_path}")

    out_path = _resolve_from_project(args.output)
    shard_world_size = int(args.shard_world_size)
    shard_rank = int(args.shard_rank)
    shard_weights = str(args.shard_weights)

    qwen_model_dir = _resolve_from_project(args.model_dir)
    dinov3_dir = _resolve_from_project(args.dinov3_dir)
    merger_ckpt = _resolve_from_project(args.merger_ckpt)
    assert_sva_runtime_matches_merger(
        qwen_model_dir=qwen_model_dir,
        dinov3_dir=dinov3_dir,
        smart_resize_min_pixels=int(args.smart_resize_min_pixels),
        smart_resize_max_pixels=int(args.smart_resize_max_pixels),
        merger_ckpt=merger_ckpt,
    )

    file_qwen = str(rows[0].get("model_dir", "")).strip() if rows else ""
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
    file_min_pixels = rows[0].get("smart_resize_min_pixels", None) if rows else None
    if file_min_pixels is not None and int(file_min_pixels) != int(args.smart_resize_min_pixels):
        raise ValueError(
            "smart_resize_min_pixels mismatches preds metadata. "
            f"preds expect={file_min_pixels}, got={args.smart_resize_min_pixels}"
        )
    file_max_pixels = rows[0].get("smart_resize_max_pixels", None) if rows else None
    if file_max_pixels is not None and int(file_max_pixels) != int(args.smart_resize_max_pixels):
        raise ValueError(
            "smart_resize_max_pixels mismatches preds metadata. "
            f"preds expect={file_max_pixels}, got={args.smart_resize_max_pixels}"
        )

    processor = AutoProcessor.from_pretrained(str(qwen_model_dir))
    tokenizer = processor.tokenizer

    max_new_tokens = int(args.max_new_tokens)
    max_retries = max(1, int(args.max_retries))
    max_sentences = max(1, int(args.max_sentences))
    requested_batch_size = max(1, int(args.batch_size))

    hit_idx: list[int] = []
    hit_imgids: list[int] = []
    for i, row in enumerate(rows):
        pred = str(row.get("prediction", "")).strip()
        if not pred:
            continue
        tok_len = _token_len(tokenizer, pred)
        if tok_len >= max_new_tokens:
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
        f"max_sentences={max_sentences} requested_batch_size={requested_batch_size}"
    )
    print(f"[check] file={preds_path} total={len(rows)} hit(max_new_tokens)={len(hit_idx)}")
    if hit_imgids_sorted:
        print(f"[check] hit_imgids(first 30)={hit_imgids_sorted[:30]}")
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

    captioner = SVADualVisualCaptioner(
        qwen_model_dir=qwen_model_dir,
        dinov3_dir=dinov3_dir,
        smart_resize_min_pixels=int(args.smart_resize_min_pixels),
        smart_resize_max_pixels=int(args.smart_resize_max_pixels),
        merger_ckpt=merger_ckpt,
        lora_dir=_resolve_from_project(args.lora_dir) if str(args.lora_dir).strip() else None,
        merge_lora=bool(args.merge_lora),
        device_map=str(args.device_map),
        dtype=str(args.dtype),
        max_new_tokens=max_new_tokens,
        do_sample=args.do_sample,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        num_beams=args.num_beams,
        seed=args.seed,
        no_repeat_ngram_size=args.no_repeat_ngram_size,
        repetition_penalty=args.repetition_penalty,
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
        last_tok_len = _token_len(tokenizer, last_pred) if last_pred else 0
        selected_items.append(
            RetryItem(
                idx=i,
                imgid=imgid,
                image_path=image_path,
                last_pred=last_pred,
                last_tok_len=last_tok_len,
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
                pred = limit_sentences(str(pred_obj.text), max_sentences=max_sentences)
                pred = ensure_sentence_end(pred)

                tok_len = _token_len(tokenizer, pred)
                item.last_pred = str(pred).strip()
                item.last_tok_len = tok_len
                item.attempt_used += 1

                is_hit = int(tok_len >= max_new_tokens)
                print(
                    f"[retry] imgid={item.imgid} attempt={item.attempt_used}/{max_retries} "
                    f"tok_len={tok_len} hit={is_hit}"
                )

                if tok_len < max_new_tokens:
                    item.success = True
                    resolved += 1
                    continue

                if round_idx >= max_retries:
                    unresolved_imgids.append(item.imgid)
                    print(
                        f"[retry-fail] imgid={item.imgid} attempts={max_retries} "
                        f"final_tok_len={item.last_tok_len}"
                    )
                    continue

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
                "prompt": prompt,
                "max_new_tokens": max_new_tokens,
                "smart_resize_min_pixels": int(args.smart_resize_min_pixels),
                "smart_resize_max_pixels": int(args.smart_resize_max_pixels),
                "decode_strategy": str(captioner.decode_strategy),
                "do_sample": args.do_sample,
                "temperature": args.temperature,
                "top_p": args.top_p,
                "top_k": args.top_k,
                "num_beams": args.num_beams,
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
