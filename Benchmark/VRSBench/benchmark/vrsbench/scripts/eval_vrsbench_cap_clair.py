from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import re
import threading
import time
from pathlib import Path
from typing import Any

import requests
from tqdm import tqdm


API_BASE_DEFAULT = "https://infai.cc/v1"
JUDGE_MODEL_DEFAULT = "gpt-4o-mini"
THREAD_LOCAL = threading.local()
CLAIR_PROMPT = """You are trying to tell if a candidate set of captions is describing the same image as a reference set of captions.
Candidate set:
{candidate_statements}
Reference set:
{target_statements}
On a precise scale from 0 to 100, how likely is it that the candidate set is describing the same image as the reference set? (JSON format, with a key "score", value between 0 and 100, and a key "reason" with a string value.)
"""


def resolve_from_project(path: str | Path) -> Path:
    root = Path(__file__).resolve().parents[4]
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return (root / candidate).resolve()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


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


def build_headers(api_key: str) -> dict[str, str]:
    return {
        "Authorization": f"Bearer {str(api_key).strip()}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }


def get_thread_local_session() -> requests.Session:
    session = getattr(THREAD_LOCAL, "session", None)
    if session is None:
        session = requests.Session()
        setattr(THREAD_LOCAL, "session", session)
    return session


def should_retry_status(status_code: int) -> bool:
    return int(status_code) in {408, 409, 429, 500, 502, 503, 504}


def retry_delay_seconds(*, attempt_index: int, base_delay: float, max_delay: float) -> float:
    delay = float(base_delay) * (2 ** max(0, int(attempt_index)))
    return min(float(max_delay), delay)


def build_prompt(*, prediction: str, ground_truth: str) -> str:
    candidate_statements = f"- {str(prediction).strip()}\n"
    target_statements = f"- {str(ground_truth).strip()}\n"
    return CLAIR_PROMPT.format(
        candidate_statements=candidate_statements,
        target_statements=target_statements,
    )


def build_payload(*, model: str, prompt: str, max_tokens: int) -> dict[str, Any]:
    return {
        "model": str(model),
        "messages": [{"role": "user", "content": [{"type": "text", "text": str(prompt)}]}],
        "max_tokens": int(max_tokens),
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
        headers=build_headers(api_key),
        json=payload,
        timeout=float(timeout),
    )
    try:
        body: dict[str, Any] | str = response.json()
    except Exception:
        body = response.text[:4000]
    return int(response.status_code), body


def extract_completion_text(body: dict[str, Any] | str) -> str:
    if not isinstance(body, dict):
        return ""
    try:
        content = body["choices"][0]["message"]["content"]
    except Exception:
        return ""
    return "" if content is None else str(content).strip()


def parse_clair_response(text: str) -> tuple[float | None, str | None]:
    value = str(text).strip()
    if not value:
        return None, None

    try:
        parsed = value.split("{", 1)[1]
        parsed = "{" + parsed.split("}", 1)[0] + "}"
        data = json.loads(parsed)
        score = float(data["score"])
        reason = data.get("reason", "Unknown")
    except Exception:
        numbers = re.findall(r"\d*\.?\d+", value)
        if not numbers:
            return None, None
        score = float(numbers[0])
        reason_match = re.findall(r"(?i)reason.*", value)
        if reason_match:
            reason = reason_match[0].strip()[len("reason") :].replace(":", "").strip()
        else:
            reason = "Unknown"

    if score < 1:
        score *= 100.0
    if score < 0 or score > 100:
        return None, reason
    if reason is None:
        return float(score) / 100.0, None
    cleaned_reason = str(reason).strip()
    cleaned_reason = cleaned_reason.strip("`").strip()
    cleaned_reason = cleaned_reason.strip('"').strip()
    cleaned_reason = re.sub(r"\s+", " ", cleaned_reason)
    return float(score) / 100.0, cleaned_reason


def judge_one_with_clair(
    *,
    api_base: str,
    api_key: str,
    api_timeout: float,
    model: str,
    prediction: str,
    ground_truth: str,
    max_tokens: int,
    max_retries: int,
    retry_backoff_base: float,
    retry_backoff_max: float,
) -> tuple[float, str, str]:
    prompt = build_prompt(prediction=str(prediction), ground_truth=str(ground_truth))
    payload = build_payload(model=str(model), prompt=prompt, max_tokens=int(max_tokens))
    total_attempts = max(1, int(max_retries))
    last_response = ""
    last_error = ""

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

            last_response = extract_completion_text(body)
            score, reason = parse_clair_response(last_response)
            if score is not None:
                return float(score), "" if reason is None else str(reason), last_response
            last_error = f"failed to parse CLAIR response: {last_response}"
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

    raise RuntimeError(last_error or f"failed to parse CLAIR response: {last_response}")


def build_judged_row(
    *,
    row: dict[str, Any],
    ground_truth: str,
    clair_score: float,
    clair_reason: str,
    judge_model: str,
    judge_raw_response: str,
) -> dict[str, Any]:
    output = dict(row)
    output["ground_truth"] = str(ground_truth)
    output["clair"] = float(clair_score)
    output["clair_x100"] = float(clair_score) * 100.0
    output["clair_reason"] = str(clair_reason)
    output["judge_model"] = str(judge_model)
    output["judge_raw_response"] = str(judge_raw_response)
    return output


def build_summary(
    *,
    rows: list[dict[str, Any]],
    preds_path: Path,
    refs_path: Path,
    judged_output_path: Path,
    judge_model: str,
    api_base: str,
    num_pending_before_run: int,
) -> dict[str, Any]:
    scores = [float(row["clair"]) for row in rows]
    avg_score = float(sum(scores) / max(1, len(scores)))
    rows_sorted = sorted(rows, key=lambda row: int(str(row.get("imgid"))))
    samples = []
    for row in rows_sorted[:5]:
        samples.append(
            {
                "imgid": int(row["imgid"]),
                "prediction": str(row.get("prediction", "")),
                "ground_truth": str(row.get("ground_truth", "")),
                "clair": float(row["clair"]),
            }
        )
    return {
        "preds": str(preds_path),
        "refs": str(refs_path),
        "judged_output": str(judged_output_path),
        "judge_model": str(judge_model),
        "api_base": str(api_base),
        "num_pending_before_run": int(num_pending_before_run),
        "num_images": len(rows),
        "clair": avg_score,
        "clair_x100": avg_score * 100.0,
        "samples": samples,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate VRSBench caption predictions with official CLAIR prompt.")
    parser.add_argument("--refs", type=str, default="benchmark/vrsbench/data/vrsbench_refs_test.json")
    parser.add_argument("--preds", type=str, required=True)
    parser.add_argument("--judged-output", type=str, required=True)
    parser.add_argument("--summary-output", type=str, required=True)
    parser.add_argument("--api-base", type=str, default=API_BASE_DEFAULT)
    parser.add_argument("--api-key", type=str, default="")
    parser.add_argument("--api-key-env", type=str, default="INFAI_API_KEY")
    parser.add_argument("--judge-model", type=str, default=JUDGE_MODEL_DEFAULT)
    parser.add_argument("--judge-max-tokens", type=int, default=100)
    parser.add_argument("--max-workers", type=int, default=512)
    parser.add_argument("--judge-max-retries", type=int, default=6)
    parser.add_argument("--retry-backoff-base", type=float, default=0.25)
    parser.add_argument("--retry-backoff-max", type=float, default=4.0)
    parser.add_argument("--api-timeout", type=float, default=120.0)
    parser.add_argument("--max-images", type=int, default=0)
    args = parser.parse_args()

    api_key = read_api_key(args)
    api_base = normalize_api_base(str(args.api_base))
    refs_path = resolve_from_project(args.refs)
    preds_path = resolve_from_project(args.preds)
    judged_output_path = resolve_from_project(args.judged_output)
    summary_output_path = resolve_from_project(args.summary_output)

    if not refs_path.is_file():
        raise FileNotFoundError(f"missing refs: {refs_path}")
    if not preds_path.is_file():
        raise FileNotFoundError(f"missing preds: {preds_path}")

    refs = read_json(refs_path)
    if not isinstance(refs, dict):
        raise TypeError("refs json must be a dict keyed by imgid")

    preds_rows = read_jsonl(preds_path)
    pred_map: dict[str, dict[str, Any]] = {}
    for row in preds_rows:
        imgid = str(row.get("imgid", "")).strip()
        if imgid:
            pred_map[imgid] = row

    ref_ids = sorted(refs.keys(), key=lambda value: int(value))
    if int(args.max_images) > 0:
        ref_ids = ref_ids[: int(args.max_images)]

    missing = [imgid for imgid in ref_ids if imgid not in pred_map]
    if missing:
        raise KeyError(f"missing predictions for {len(missing)} imgids, first={missing[:5]}")

    existing_rows = read_jsonl(judged_output_path)
    existing_by_imgid: dict[str, dict[str, Any]] = {}
    for row in existing_rows:
        imgid = str(row.get("imgid", "")).strip()
        if imgid:
            existing_by_imgid[imgid] = row

    pending_imgids = [imgid for imgid in ref_ids if imgid not in existing_by_imgid]
    pending_payloads: list[tuple[str, dict[str, Any], str]] = []
    for imgid in pending_imgids:
        ref_list = refs[imgid]
        if not isinstance(ref_list, list) or len(ref_list) != 1:
            raise ValueError(f"expected exactly one ground truth caption for imgid={imgid}")
        pending_payloads.append((imgid, pred_map[imgid], str(ref_list[0])))

    if pending_payloads:
        with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, int(args.max_workers))) as executor:
            future_to_imgid: dict[concurrent.futures.Future[tuple[float, str, str]], str] = {}
            for imgid, row, ground_truth in pending_payloads:
                future = executor.submit(
                    judge_one_with_clair,
                    api_base=api_base,
                    api_key=api_key,
                    api_timeout=float(args.api_timeout),
                    model=str(args.judge_model),
                    prediction=str(row.get("prediction", "")),
                    ground_truth=str(ground_truth),
                    max_tokens=int(args.judge_max_tokens),
                    max_retries=int(args.judge_max_retries),
                    retry_backoff_base=float(args.retry_backoff_base),
                    retry_backoff_max=float(args.retry_backoff_max),
                )
                future_to_imgid[future] = imgid

            with tqdm(total=len(future_to_imgid), desc="vrsbench:caption:clair") as progress:
                for future in concurrent.futures.as_completed(future_to_imgid):
                    imgid = future_to_imgid[future]
                    row = pred_map[imgid]
                    ground_truth = str(refs[imgid][0])
                    score, reason, raw_response = future.result()
                    judged_row = build_judged_row(
                        row=row,
                        ground_truth=ground_truth,
                        clair_score=float(score),
                        clair_reason=str(reason),
                        judge_model=str(args.judge_model),
                        judge_raw_response=str(raw_response),
                    )
                    append_jsonl(judged_output_path, judged_row)
                    existing_by_imgid[imgid] = judged_row
                    progress.update(1)

    final_rows = [existing_by_imgid[imgid] for imgid in ref_ids]
    summary = build_summary(
        rows=final_rows,
        preds_path=preds_path,
        refs_path=refs_path,
        judged_output_path=judged_output_path,
        judge_model=str(args.judge_model),
        api_base=api_base,
        num_pending_before_run=len(pending_payloads),
    )
    write_json(summary_output_path, summary)
    print(f"[OK] Wrote judged rows: {judged_output_path}")
    print(f"[OK] Wrote summary: {summary_output_path}")


if __name__ == "__main__":
    main()
