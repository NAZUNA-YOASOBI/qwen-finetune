from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
from typing import Any


def _project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _resolve_from_project(path: str | Path) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    return (_project_root() / p).resolve()


def _safe_int(value: Any, default: int = -1) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def _token_len(tokenizer: Any, text: str) -> int:
    return len(tokenizer(str(text), add_special_tokens=False).input_ids)


SENTENCE_SPLIT_PATTERN = re.compile(r"(?<=[.!?。！？])(?:\s+|(?=\S))")


def _trim_to_max_sentences(text: str, *, max_sentences: int) -> str:
    raw = re.sub(r"\s+", " ", str(text or "").strip())
    if not raw or int(max_sentences) <= 0:
        return raw
    parts = [part.strip() for part in SENTENCE_SPLIT_PATTERN.split(raw) if part.strip()]
    if len(parts) <= int(max_sentences):
        return raw
    return " ".join(parts[: int(max_sentences)]).strip()


def _normalize_meta_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return ""
    return str(value).strip()


def _assert_prediction_runtime_consistency(
    rows: list[dict[str, Any]],
    *,
    model_family: str,
    model_dir: Path,
    max_new_tokens: int,
) -> None:
    if not rows:
        return

    fields_to_check = [
        "task",
        "model_family",
        "model_dir",
        "max_new_tokens",
        "decode_strategy",
        "do_sample",
        "temperature",
        "top_p",
        "top_k",
        "num_beams",
        "repetition_penalty",
        "seed",
        "dtype",
        "device_map",
    ]
    multi_value_examples: dict[str, list[str]] = {}
    for field in fields_to_check:
        values = sorted(
            {
                _normalize_meta_value(row.get(field))
                for row in rows
                if field in row and _normalize_meta_value(row.get(field)) != ""
            }
        )
        missing_count = sum(1 for row in rows if field not in row)
        if len(values) > 1:
            multi_value_examples[field] = values[:5]
        elif values and missing_count > 0:
            multi_value_examples[field] = values[:5] + ["<missing>"]
    if multi_value_examples:
        raise ValueError(f"Prediction file mixes multiple runtime configurations: {multi_value_examples}")

    row0 = rows[0]
    expected_pairs = {
        "model_family": _normalize_meta_value(model_family),
        "model_dir": _normalize_meta_value(model_dir),
        "max_new_tokens": _normalize_meta_value(int(max_new_tokens)),
    }
    mismatches: dict[str, dict[str, str]] = {}
    for field, expected in expected_pairs.items():
        if field not in row0:
            continue
        observed = _normalize_meta_value(row0.get(field))
        if observed != expected:
            mismatches[field] = {"expected": expected, "observed": observed}
    if mismatches:
        raise ValueError(f"Prediction file runtime does not match current fix config: {mismatches}")


def _build_runner(model_family: str, args, row0: dict[str, Any]):
    kwargs = {
        "device_map": str(args.device_map),
        "dtype": str(args.dtype),
        "max_new_tokens": int(args.max_new_tokens),
        "do_sample": row0.get("do_sample", None),
        "temperature": row0.get("temperature", None),
        "top_p": row0.get("top_p", None),
        "top_k": row0.get("top_k", None),
        "num_beams": row0.get("num_beams", None),
        "repetition_penalty": row0.get("repetition_penalty", None),
        "seed": row0.get("seed", None),
    }

    if str(model_family) == "qwen3vl":
        from geochatbench_qwen3vl import Qwen3VLRunner

        return Qwen3VLRunner(args.model_dir, **kwargs)
    if str(model_family) == "qwen35":
        from geochatbench_qwen35 import Qwen35Runner

        return Qwen35Runner(args.model_dir, enable_thinking=False, **kwargs)
    raise ValueError(f"Unsupported model_family: {model_family}")


def _row_hits_max_new_tokens(row: dict[str, Any], *, tokenizer: Any, max_new_tokens: int) -> tuple[bool, int, bool]:
    answer = str(row.get("answer", "")).strip()
    if not answer:
        return True, 0, False

    generated_token_count = _safe_int(row.get("generated_token_count", 0), default=0)
    ended_by_eos = row.get("generation_ended_by_eos", None)
    if isinstance(ended_by_eos, bool):
        is_hit = (not bool(ended_by_eos)) and generated_token_count >= int(max_new_tokens)
        return bool(is_hit), int(generated_token_count), False

    tok_len = _token_len(tokenizer, answer)
    return bool(tok_len >= int(max_new_tokens)), int(tok_len), True


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Retry GeoChat predictions that hit max_new_tokens without EOS, and overwrite them in-place."
    )
    parser.add_argument("--preds", type=str, required=True)
    parser.add_argument("--model-family", type=str, required=True, choices=["qwen3vl", "qwen35"])
    parser.add_argument("--model-dir", type=str, required=True)
    parser.add_argument("--device-map", type=str, default="auto")
    parser.add_argument("--dtype", type=str, default="auto", choices=["auto", "fp16", "bf16", "fp32"])
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--max-retries", type=int, default=10)
    parser.add_argument("--max-sentences", type=int, default=2)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    import sys

    sys.path.insert(0, str(_project_root() / "src"))
    from geochatbench_common import read_jsonl

    preds_path = _resolve_from_project(args.preds)
    model_dir = _resolve_from_project(args.model_dir)
    if not preds_path.is_file():
        raise FileNotFoundError(f"Missing prediction file: {preds_path}")
    if not model_dir.exists():
        raise FileNotFoundError(f"Missing model directory: {model_dir}")

    rows = read_jsonl(preds_path, allow_truncated_last_line=True)
    if not rows:
        raise ValueError(f"Prediction file is empty: {preds_path}")
    _assert_prediction_runtime_consistency(
        rows,
        model_family=str(args.model_family),
        model_dir=model_dir,
        max_new_tokens=int(args.max_new_tokens),
    )

    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(str(model_dir))
    tokenizer = processor.tokenizer

    max_new_tokens = int(args.max_new_tokens)
    hit_indices: list[int] = []
    fallback_rows = 0
    for idx, row in enumerate(rows):
        is_hit, _, used_fallback = _row_hits_max_new_tokens(
            row,
            tokenizer=tokenizer,
            max_new_tokens=max_new_tokens,
        )
        if used_fallback:
            fallback_rows += 1
        if is_hit:
            hit_indices.append(idx)

    print(
        f"[check] file={preds_path} total={len(rows)} "
        f"hit_without_eos={len(hit_indices)} max_new_tokens={max_new_tokens}"
    )
    if fallback_rows:
        print(f"[warn] rows_missing_generation_eos_metadata={fallback_rows}")

    if bool(args.dry_run) or not hit_indices:
        print(f"[summary] retried=0 resolved=0 unresolved={len(hit_indices)} dry_run={int(bool(args.dry_run))}")
        return

    row0 = rows[0]
    runner = _build_runner(str(args.model_family), argparse.Namespace(**{**vars(args), "model_dir": model_dir}), row0)
    max_retries = max(1, int(args.max_retries))
    resolved = 0
    retried = 0
    unresolved_ids: list[str] = []

    for pos, idx in enumerate(hit_indices):
        row = rows[idx]
        question_id = str(row.get("question_id", f"row_{idx}"))
        prompt = str(row.get("prompt", "")).strip()
        image_path = _resolve_from_project(str(row.get("image_path", "")))
        if not prompt:
            raise ValueError(f"Missing prompt for question_id={question_id} in {preds_path}")
        if not image_path.is_file():
            raise FileNotFoundError(f"Missing image file for question_id={question_id}: {image_path}")

        last_pred = str(row.get("answer", "")).strip()
        last_generated_token_count = _safe_int(row.get("generated_token_count", 0), default=0)
        last_ended_by_eos = bool(row.get("generation_ended_by_eos", False))
        last_generated_token_id = row.get("generation_last_token_id", None)
        attempt_used = 0
        success = False
        retried += 1

        for attempt in range(1, max_retries + 1):
            pred = runner.generate_batch(image_paths=[image_path], prompts=[prompt])[0]
            last_pred = _trim_to_max_sentences(str(pred.text).strip(), max_sentences=int(args.max_sentences))
            last_generated_token_count = int(pred.generated_token_count)
            last_ended_by_eos = bool(pred.ended_by_eos)
            last_generated_token_id = pred.last_generated_token_id
            attempt_used = attempt
            still_hit = (not bool(pred.ended_by_eos)) and int(pred.generated_token_count) >= max_new_tokens
            print(
                f"[retry] question_id={question_id} attempt={attempt}/{max_retries} "
                f"generated_token_count={int(pred.generated_token_count)} "
                f"ended_by_eos={int(bool(pred.ended_by_eos))} hit={int(bool(still_hit))}"
            )
            if bool(pred.ended_by_eos) and bool(last_pred):
                success = True
                break

        row["answer"] = last_pred
        row["generated_token_count"] = int(last_generated_token_count)
        row["generation_ended_by_eos"] = bool(last_ended_by_eos)
        row["generation_last_token_id"] = last_generated_token_id
        row["max_token_retry_attempts"] = int(attempt_used)
        row["max_token_retry_success"] = bool(success)

        if success:
            resolved += 1
        else:
            unresolved_ids.append(question_id)
            print(
                f"[retry-fail] question_id={question_id} attempts={max_retries} "
                f"final_generated_token_count={last_generated_token_count} "
                f"ended_by_eos={int(bool(last_ended_by_eos))}"
            )

        if pos == len(hit_indices) - 1:
            continue

    tmp_path = preds_path.with_suffix(preds_path.suffix + ".tmp")
    _write_jsonl(tmp_path, rows)
    tmp_path.replace(preds_path)

    rows_after = read_jsonl(preds_path, allow_truncated_last_line=True)
    hit_after = 0
    fallback_rows_after = 0
    for row in rows_after:
        is_hit, _, used_fallback = _row_hits_max_new_tokens(
            row,
            tokenizer=tokenizer,
            max_new_tokens=max_new_tokens,
        )
        if used_fallback:
            fallback_rows_after += 1
        if is_hit:
            hit_after += 1

    unresolved = len(hit_indices) - resolved
    print(
        f"[summary] retried={retried} resolved={resolved} unresolved={unresolved} "
        f"hit_before={len(hit_indices)} hit_after={hit_after}"
    )
    if unresolved_ids:
        print(f"[summary-unresolved] question_ids(first 20)={unresolved_ids[:20]}")
    if fallback_rows_after:
        print(f"[warn] rows_missing_generation_eos_metadata_after={fallback_rows_after}")
    if unresolved > 0 or hit_after > 0:
        raise SystemExit(5)


if __name__ == "__main__":
    main()
