from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import re
import threading
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import requests
from tqdm import tqdm


API_BASE_DEFAULT = "https://api.gptsapi.net/v1"
JUDGE_MODEL_DEFAULT = "gpt-4o-mini"
PAPER_TYPE_MAP: dict[str, str] = {
    "object category": "Category",
    "object existence": "Presence",
    "object quantity": "Quantity",
    "object color": "Color",
    "object shape": "Shape",
    "object size": "Size",
    "object position": "Position",
    "object direction": "Direction",
    "scene type": "Scene",
    "image": "Scene",
    "rural or urban": "Scene",
    "reasoning": "Reasoning",
}
PAPER_TYPE_ORDER = [
    "Category",
    "Presence",
    "Quantity",
    "Color",
    "Shape",
    "Size",
    "Position",
    "Direction",
    "Scene",
    "Reasoning",
]
SHORT_EXACT_ANSWERS = {"yes", "no", *[str(index) for index in range(100)]}
OFFICIAL_VQA_JUDGE_PROMPT = (
    "Question: {question}\n"
    "Ground Truth Answer: {ground_truth}\n"
    "Predicted Answer: {predicted}\n"
    "Does the predicted answer match the ground truth? Answer 1 for match and 0 for not match. "
    "Use semantic meaning not exact match. Synonyms are also treated as a match, e.g., football and soccer, "
    "playground and ground track field, building and rooftop, pond and swimming pool. "
    "Do not explain the reason.\n"
)
THREAD_LOCAL = threading.local()


def resolve_from_project(path: str | Path) -> Path:
    root = Path(__file__).resolve().parents[4]
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return (root / candidate).resolve()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.is_file():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def read_api_key(args: argparse.Namespace) -> str:
    if str(args.api_key).strip():
        return str(args.api_key).strip()
    env_name = str(args.api_key_env).strip()
    if env_name and os.environ.get(env_name):
        return str(os.environ[env_name]).strip()
    if os.environ.get("OPENAI_API_KEY"):
        return str(os.environ["OPENAI_API_KEY"]).strip()
    raise ValueError(f"missing API key. Set --api-key, export {env_name}, or export OPENAI_API_KEY.")


def normalize_api_base(api_base: str) -> str:
    return str(api_base).strip().rstrip("/")


def build_chat_completions_url(api_base: str) -> str:
    return f"{normalize_api_base(api_base)}/chat/completions"


def get_thread_local_session() -> requests.Session:
    session = getattr(THREAD_LOCAL, "session", None)
    if session is None:
        session = requests.Session()
        setattr(THREAD_LOCAL, "session", session)
    return session


def official_short_circuit_match(*, ground_truth: str, predicted: str) -> str | None:
    gt = str(ground_truth).strip().lower()
    pred = str(predicted).strip().lower()
    if not gt:
        return "0"
    if gt in pred:
        return "1"
    if gt in SHORT_EXACT_ANSWERS:
        return "1" if gt == pred else "0"
    return None


def parse_binary_response(text: str) -> str | None:
    value = str(text).strip()
    if value in {"0", "1"}:
        return value
    match = re.search(r"\b([01])\b", value)
    if match is not None:
        return str(match.group(1))
    return None


def build_chat_completion_payload(
    *,
    model: str,
    prompt: str,
    max_tokens: int,
) -> dict[str, Any]:
    return {
        "model": str(model),
        "messages": [{"role": "user", "content": [{"type": "text", "text": str(prompt)}]}],
        "max_tokens": int(max_tokens),
    }


def build_chat_completion_headers(api_key: str) -> dict[str, str]:
    return {
        "Authorization": f"Bearer {str(api_key).strip()}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }


def request_chat_completion(
    *,
    api_base: str,
    api_key: str,
    payload: dict[str, Any],
    timeout: float,
) -> tuple[int, dict[str, Any] | str]:
    response = get_thread_local_session().post(
        build_chat_completions_url(api_base),
        headers=build_chat_completion_headers(api_key),
        json=payload,
        timeout=float(timeout),
    )
    try:
        body: dict[str, Any] | str = response.json()
    except Exception:
        body = response.text[:4000]
    return int(response.status_code), body


def should_retry_status(status_code: int) -> bool:
    return int(status_code) in {408, 409, 429, 500, 502, 503, 504}


def retry_delay_seconds(*, attempt_index: int, base_delay: float, max_delay: float) -> float:
    delay = float(base_delay) * (2 ** max(0, int(attempt_index)))
    return min(float(max_delay), delay)


def extract_chat_completion_text(body: dict[str, Any] | str) -> str:
    if not isinstance(body, dict):
        return ""
    try:
        content = body["choices"][0]["message"]["content"]
    except Exception:
        return ""
    return "" if content is None else str(content).strip()


def judge_one_with_gpt(
    *,
    api_base: str,
    api_key: str,
    api_timeout: float,
    model: str,
    question: str,
    ground_truth: str,
    predicted: str,
    max_tokens: int,
    max_retries: int,
    retry_backoff_base: float,
    retry_backoff_max: float,
) -> tuple[str, str]:
    prompt = OFFICIAL_VQA_JUDGE_PROMPT.format(
        question=str(question),
        ground_truth=str(ground_truth),
        predicted=str(predicted),
    )
    last_response = ""
    last_error = ""
    total_attempts = max(1, int(max_retries))
    payload = build_chat_completion_payload(
        model=str(model),
        prompt=prompt,
        max_tokens=int(max_tokens),
    )

    for attempt_index in range(total_attempts):
        try:
            status_code, body = request_chat_completion(
                api_base=api_base,
                api_key=api_key,
                payload=payload,
                timeout=float(api_timeout),
            )
            if status_code != 200:
                last_error = (
                    json.dumps(body, ensure_ascii=False)[:1000]
                    if isinstance(body, dict)
                    else str(body)
                )
                if should_retry_status(status_code) and attempt_index + 1 < total_attempts:
                    time.sleep(
                        retry_delay_seconds(
                            attempt_index=attempt_index,
                            base_delay=float(retry_backoff_base),
                            max_delay=float(retry_backoff_max),
                        )
                    )
                    continue
                raise RuntimeError(f"judge request failed with status {status_code}: {last_error}")

            last_response = extract_chat_completion_text(body)
            parsed = parse_binary_response(last_response)
            if parsed is not None:
                return parsed, last_response
            last_error = f"failed to parse judge response as 0/1: {last_response}"
        except Exception as exc:
            last_error = repr(exc)

        if attempt_index + 1 < total_attempts:
            time.sleep(
                retry_delay_seconds(
                    attempt_index=attempt_index,
                    base_delay=float(retry_backoff_base),
                    max_delay=float(retry_backoff_max),
                )
            )

    raise RuntimeError(last_error or f"failed to parse judge response as 0/1: {last_response}")


def build_judged_row(
    *,
    row: dict[str, Any],
    predicted: str,
    correct_gpt: str,
    judge_source: str,
    judge_model: str,
    judge_raw_response: str,
) -> dict[str, Any]:
    return {
        **row,
        "predicted": str(predicted),
        "correct_gpt": str(correct_gpt),
        "judge_source": str(judge_source),
        "judge_model": str(judge_model),
        "judge_raw_response": str(judge_raw_response),
    }


def judge_row_with_gpt(
    *,
    row: dict[str, Any],
    api_base: str,
    api_key: str,
    api_timeout: float,
    judge_model: str,
    judge_max_tokens: int,
    judge_max_retries: int,
    retry_backoff_base: float,
    retry_backoff_max: float,
) -> dict[str, Any]:
    question = str(row.get("question", "")).strip()
    ground_truth = str(row.get("ground_truth", "")).strip().lower()
    predicted = str(row.get("answer", "")).strip().lower()
    correct_gpt, judge_raw = judge_one_with_gpt(
        api_base=api_base,
        api_key=api_key,
        api_timeout=float(api_timeout),
        model=str(judge_model),
        question=question,
        ground_truth=ground_truth,
        predicted=predicted,
        max_tokens=int(judge_max_tokens),
        max_retries=int(judge_max_retries),
        retry_backoff_base=float(retry_backoff_base),
        retry_backoff_max=float(retry_backoff_max),
    )
    return build_judged_row(
        row=row,
        predicted=predicted,
        correct_gpt=correct_gpt,
        judge_source="gpt",
        judge_model=str(judge_model),
        judge_raw_response=judge_raw,
    )


def to_paper_type(raw_type: str) -> str:
    return PAPER_TYPE_MAP.get(str(raw_type), str(raw_type))


def compute_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_raw_type: dict[str, dict[str, int]] = defaultdict(lambda: {"total": 0, "correct": 0})
    by_paper_type: dict[str, dict[str, int]] = defaultdict(lambda: {"total": 0, "correct": 0})
    overall_total = 0
    overall_correct = 0

    for row in rows:
        raw_type = str(row.get("type", "")).strip().lower()
        paper_type = to_paper_type(raw_type)
        correct = 1 if str(row.get("correct_gpt", "")).strip() == "1" else 0
        overall_total += 1
        overall_correct += correct
        by_raw_type[raw_type]["total"] += 1
        by_raw_type[raw_type]["correct"] += correct
        by_paper_type[paper_type]["total"] += 1
        by_paper_type[paper_type]["correct"] += correct

    raw_metrics: dict[str, Any] = {}
    for raw_type in sorted(by_raw_type):
        total = by_raw_type[raw_type]["total"]
        correct = by_raw_type[raw_type]["correct"]
        raw_metrics[raw_type] = {
            "count": total,
            "correct": correct,
            "accuracy": (correct / total) if total > 0 else 0.0,
            "accuracy_x100": ((correct / total) * 100.0) if total > 0 else 0.0,
        }

    paper_metrics: dict[str, Any] = {}
    for paper_type in PAPER_TYPE_ORDER:
        total = by_paper_type[paper_type]["total"]
        correct = by_paper_type[paper_type]["correct"]
        paper_metrics[paper_type] = {
            "count": total,
            "correct": correct,
            "accuracy": (correct / total) if total > 0 else 0.0,
            "accuracy_x100": ((correct / total) * 100.0) if total > 0 else 0.0,
        }

    return {
        "accuracy": (overall_correct / overall_total) if overall_total > 0 else 0.0,
        "accuracy_x100": ((overall_correct / overall_total) * 100.0) if overall_total > 0 else 0.0,
        "correct": overall_correct,
        "num_questions": overall_total,
        "by_type_raw": raw_metrics,
        "by_type_paper_table3": paper_metrics,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate official VRSBench VQA predictions with official GPT judge logic.")
    parser.add_argument("--preds", type=str, required=True)
    parser.add_argument("--judged-output", type=str, required=True)
    parser.add_argument("--summary-out", type=str, required=True)
    parser.add_argument("--api-base", type=str, default=API_BASE_DEFAULT)
    parser.add_argument("--api-key", type=str, default="")
    parser.add_argument("--api-key-env", type=str, default="GPTSAPI_KEY")
    parser.add_argument("--api-timeout", type=float, default=60.0)
    parser.add_argument("--judge-model", type=str, default=JUDGE_MODEL_DEFAULT)
    parser.add_argument("--judge-max-tokens", type=int, default=100)
    parser.add_argument("--judge-max-retries", type=int, default=100)
    parser.add_argument("--max-workers", type=int, default=1)
    parser.add_argument("--retry-backoff-base", type=float, default=0.5)
    parser.add_argument("--retry-backoff-max", type=float, default=8.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    preds_path = resolve_from_project(args.preds)
    judged_output = resolve_from_project(args.judged_output)
    summary_out = resolve_from_project(args.summary_out)

    if not preds_path.is_file():
        raise FileNotFoundError(f"missing prediction file: {preds_path}")

    pred_rows = read_jsonl(preds_path)
    judged_by_qid = {int(row.get("qid", -1)): row for row in read_jsonl(judged_output)}
    pending = [row for row in pred_rows if int(row.get("qid", -1)) not in judged_by_qid]

    api_key = read_api_key(args)
    api_base = normalize_api_base(args.api_base)
    max_workers = max(1, int(args.max_workers))
    progress = tqdm(total=len(pending), desc="vrsbench:vqa:gpt-judge")
    gpt_pending: list[dict[str, Any]] = []

    try:
        for row in pending:
            ground_truth = str(row.get("ground_truth", "")).strip().lower()
            predicted = str(row.get("answer", "")).strip().lower()
            short_circuit = official_short_circuit_match(ground_truth=ground_truth, predicted=predicted)
            if short_circuit is not None:
                append_jsonl(
                    judged_output,
                    build_judged_row(
                        row=row,
                        predicted=predicted,
                        correct_gpt=short_circuit,
                        judge_source="official_short_circuit",
                        judge_model="",
                        judge_raw_response=short_circuit,
                    ),
                )
                progress.update(1)
                continue
            gpt_pending.append(row)

        if gpt_pending:
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_map = {
                    executor.submit(
                        judge_row_with_gpt,
                        row=row,
                        api_base=api_base,
                        api_key=api_key,
                        api_timeout=float(args.api_timeout),
                        judge_model=str(args.judge_model),
                        judge_max_tokens=int(args.judge_max_tokens),
                        judge_max_retries=int(args.judge_max_retries),
                        retry_backoff_base=float(args.retry_backoff_base),
                        retry_backoff_max=float(args.retry_backoff_max),
                    ): int(row.get("qid", -1))
                    for row in gpt_pending
                }
                for future in concurrent.futures.as_completed(future_map):
                    judged_row = future.result()
                    append_jsonl(judged_output, judged_row)
                    progress.update(1)
    finally:
        progress.close()

    judged_rows = read_jsonl(judged_output)
    summary = {
        "preds": str(preds_path),
        "judged_output": str(judged_output),
        "judge_model": str(args.judge_model),
        "api_base": str(args.api_base),
        "num_pending_before_run": len(pending),
        **compute_metrics(judged_rows),
    }
    write_json(summary_out, summary)
    print(f"[OK] Wrote judged rows: {judged_output}", flush=True)
    print(f"[OK] Wrote summary: {summary_out}", flush=True)


if __name__ == "__main__":
    main()
