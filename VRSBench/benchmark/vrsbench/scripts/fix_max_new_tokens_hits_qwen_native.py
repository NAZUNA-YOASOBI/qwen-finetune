from __future__ import annotations

import argparse
import json
import re
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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Re-run Qwen-native captioning for samples that hit max_new_tokens, and overwrite in-place."
    )
    parser.add_argument("--preds", type=str, required=True, help="Predictions jsonl to fix (in-place).")
    parser.add_argument("--model-dir", type=str, default="models/Qwen3-VL-8B-Instruct")
    parser.add_argument("--merger-ckpt", type=str, default="", help="Merger safetensors path (optional).")
    parser.add_argument("--lora-dir", type=str, default="", help="LoRA directory (optional).")
    parser.add_argument("--merge-lora", action="store_true")
    parser.add_argument("--device-map", type=str, default="auto")
    parser.add_argument("--dtype", type=str, default="auto", choices=["auto", "fp16", "bf16", "fp32"])

    parser.add_argument("--prompt", type=str, default="Describe the image in detail.")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--max-retries", type=int, default=10, help="Maximum regenerate attempts per sample.")
    parser.add_argument("--do-sample", dest="do_sample", action="store_true", default=None)
    parser.add_argument("--no-sample", dest="do_sample", action="store_false")
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--repetition-penalty", type=float, default=None)

    parser.add_argument("--max-sentences", type=int, default=7, help="Post-trim to at most N sentences.")
    parser.add_argument("--dry-run", action="store_true", help="Only report which imgids hit max_new_tokens.")
    args = parser.parse_args()

    import sys

    sys.path.insert(0, str(_project_root() / "src"))

    from transformers import AutoProcessor

    from ftqwen.qwen3_vl_native_captioner import Qwen3VLNativeCaptioner

    preds_path = _resolve_from_project(args.preds)
    if not preds_path.is_file():
        raise FileNotFoundError(f"Missing preds: {preds_path}")
    rows = read_jsonl(preds_path)
    if not rows:
        raise ValueError(f"Empty preds jsonl: {preds_path}")

    model_dir = _resolve_from_project(args.model_dir)
    merger_ckpt = _resolve_from_project(args.merger_ckpt) if str(args.merger_ckpt).strip() else None
    lora_dir = _resolve_from_project(args.lora_dir) if str(args.lora_dir).strip() else None

    processor = AutoProcessor.from_pretrained(str(model_dir))
    tokenizer = processor.tokenizer

    max_new_tokens = int(args.max_new_tokens)
    max_retries = max(1, int(args.max_retries))
    max_sentences = max(1, int(args.max_sentences))

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

    hit_imgids_sorted = [x for x in sorted(hit_imgids) if x >= 0]
    print(
        f"[retry-policy] file={preds_path} max_new_tokens={max_new_tokens} max_retries={max_retries} "
        f"max_sentences={max_sentences}"
    )
    print(f"[check] file={preds_path} total={len(rows)} hit(max_new_tokens)={len(hit_idx)}")
    if hit_imgids_sorted:
        print(f"[check] hit_imgids(first 30)={hit_imgids_sorted[:30]}")

    if bool(args.dry_run):
        print(
            f"[retry-summary] file={preds_path} total={len(rows)} hit_before={len(hit_idx)} retried=0 "
            f"resolved=0 unresolved={len(hit_idx)} max_retries={max_retries} dry_run=1"
        )
        return

    if not hit_idx:
        print(
            f"[retry-summary] file={preds_path} total={len(rows)} hit_before=0 retried=0 "
            "resolved=0 unresolved=0 max_retries=0"
        )
        print(f"[retry-summary-after] file={preds_path} hit_after=0")
        return

    captioner = Qwen3VLNativeCaptioner(
        model_dir=model_dir,
        merger_ckpt=merger_ckpt,
        lora_dir=lora_dir,
        merge_lora=bool(args.merge_lora),
        device_map=str(args.device_map),
        dtype=str(args.dtype),
        max_new_tokens=max_new_tokens,
        do_sample=args.do_sample,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        repetition_penalty=args.repetition_penalty,
        seed=args.seed,
    )

    prompt = str(args.prompt)
    resolved = 0
    retried_count = 0
    unresolved_imgids: list[int] = []

    for pos, i in enumerate(hit_idx):
        row = rows[i]
        imgid = _safe_int(row.get("imgid"), default=-1)
        imgid_for_log = str(imgid) if imgid >= 0 else str(row.get("imgid", "NA"))

        image_path = _resolve_from_project(str(row.get("image_path", "")))
        if not image_path.is_file():
            raise FileNotFoundError(f"Missing image_path for imgid={imgid_for_log}: {image_path}")

        last_pred = str(row.get("prediction", "")).strip()
        last_tok_len = _token_len(tokenizer, last_pred)
        attempt_used = 0
        success = False
        abort_after_item = False
        retried_count += 1

        for attempt in range(1, max_retries + 1):
            try:
                pred = captioner.caption(image_path=image_path, prompt=prompt).text
            except Exception as e:
                abort_after_item = True
                print(
                    f"[retry-exception] imgid={imgid_for_log} attempt={attempt}/{max_retries} "
                    f"error={type(e).__name__}: {e}"
                )
                break
            pred = limit_sentences(pred, max_sentences=max_sentences)
            pred = ensure_sentence_end(pred)

            tok_len = _token_len(tokenizer, pred)
            last_pred = str(pred).strip()
            last_tok_len = tok_len
            attempt_used = attempt

            is_hit = int(tok_len >= max_new_tokens)
            print(
                f"[retry] imgid={imgid_for_log} attempt={attempt}/{max_retries} "
                f"tok_len={tok_len} hit={is_hit}"
            )
            if tok_len < max_new_tokens:
                success = True
                break

        row["prediction"] = last_pred
        row["prompt"] = prompt
        row["max_new_tokens"] = max_new_tokens
        row["max_token_retry_attempts"] = attempt_used
        row["max_token_retry_success"] = bool(success)

        if success:
            resolved += 1
        else:
            if imgid >= 0:
                unresolved_imgids.append(imgid)
            print(
                f"[retry-fail] imgid={imgid_for_log} attempts={max_retries} "
                f"final_tok_len={last_tok_len}"
            )

        if abort_after_item:
            remaining_imgids: list[int] = []
            for j in hit_idx[pos + 1 :]:
                maybe_id = _safe_int(rows[j].get("imgid"), default=-1)
                if maybe_id >= 0:
                    remaining_imgids.append(maybe_id)
            unresolved_imgids.extend(remaining_imgids)
            print(
                f"[retry-abort] imgid={imgid_for_log} remaining_unprocessed={len(hit_idx) - (pos + 1)}"
            )
            break

    tmp_path = preds_path.with_suffix(preds_path.suffix + ".tmp")
    write_jsonl(tmp_path, rows)
    tmp_path.replace(preds_path)

    rows2 = read_jsonl(preds_path)
    hit2 = 0
    hit2_imgids: list[int] = []
    for row in rows2:
        pred = str(row.get("prediction", "")).strip()
        if not pred:
            continue
        if _token_len(tokenizer, pred) >= max_new_tokens:
            hit2 += 1
            maybe_id = _safe_int(row.get("imgid"), default=-1)
            if maybe_id >= 0:
                hit2_imgids.append(maybe_id)

    unresolved = len(hit_idx) - resolved
    print(
        f"[retry-summary] file={preds_path} total={len(rows)} hit_before={len(hit_idx)} retried={retried_count} "
        f"resolved={resolved} unresolved={unresolved} max_retries={max_retries}"
    )
    if unresolved_imgids:
        unresolved_imgids = sorted(set(unresolved_imgids))
        print(
            f"[retry-unresolved] imgids(first 30)={unresolved_imgids[:30]} count={len(unresolved_imgids)}"
        )
    if hit2_imgids:
        hit2_imgids = sorted(set(hit2_imgids))
        print(f"[retry-summary-after] file={preds_path} hit_after={hit2} hit_imgids(first 30)={hit2_imgids[:30]}")
    else:
        print(f"[retry-summary-after] file={preds_path} hit_after={hit2}")


if __name__ == "__main__":
    main()
