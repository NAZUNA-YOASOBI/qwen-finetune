from __future__ import annotations

import argparse
import json
import re
import string
from collections import defaultdict
from pathlib import Path
from typing import Any


def _project_root() -> Path:
    return Path(__file__).resolve().parents[5]


def _resolve_from_project(path: str | Path) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    return (_project_root() / p).resolve()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _norm_text(text: str) -> str:
    value = str(text).strip().lower()
    value = re.sub(r"\b(a|an|the)\b", " ", value)
    value = "".join(ch for ch in value if ch not in set(string.punctuation))
    return " ".join(value.split())


def _extract_letter(raw: str, max_letter: str) -> str:
    upper = str(raw).upper().strip()
    if not upper:
        return ""

    patterns = [
        r"^([A-Z])[\.\)]?$",
        r"^(?:THE\s+)?(?:ANSWER|ANS|OPTION|CHOICE)\s*(?:IS|:|：|-)?\s*([A-Z])(?:[\.\)])?$",
        r"^(?:IT(?:'S|\s+IS)|I\s+THINK\s+IT(?:'S|\s+IS))\s+([A-Z])(?:[\.\)])?$",
    ]
    for pattern in patterns:
        match = re.match(pattern, upper)
        if not match:
            continue
        letter = match.group(1)
        if "A" <= letter <= max_letter:
            return letter

    labeled_letters = [
        letter
        for letter in re.findall(r"\b([A-Z])\s*[\.\)]", upper)
        if "A" <= letter <= max_letter
    ]
    labeled_letters = sorted(set(labeled_letters))
    if len(labeled_letters) == 1 and re.match(r"^\s*[A-Z]\s*[\.\)]", upper):
        return labeled_letters[0]
    return ""


def _text_candidates(raw_pred: str) -> list[str]:
    raw_text = str(raw_pred).strip()
    if not raw_text:
        return []

    candidates = [raw_text]
    prefix_patterns = [
        r"^(?:the\s+)?(?:answer|ans|option|choice)\s*(?:is|:|：|-)?\s*",
        r"^[a-z]\s*[\.\)]\s*",
    ]
    for pattern in prefix_patterns:
        stripped = re.sub(pattern, "", raw_text, flags=re.IGNORECASE).strip()
        if stripped and stripped != raw_text:
            candidates.append(stripped)

    normalized: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        candidate_norm = _norm_text(candidate)
        if candidate_norm and candidate_norm not in seen:
            seen.add(candidate_norm)
            normalized.append(candidate_norm)
    return normalized


def _predict_letter_from_text(raw_pred: str, options: list[dict[str, str]]) -> str:
    pred_norm_candidates = _text_candidates(raw_pred)
    if not pred_norm_candidates:
        return ""

    matched_letters: set[str] = set()
    for option in options:
        letter = str(option.get("letter", "")).strip().upper()
        text_norm = _norm_text(str(option.get("text", "")))
        if not letter or not text_norm:
            continue
        if text_norm in pred_norm_candidates:
            matched_letters.add(letter)

    if len(matched_letters) == 1:
        return next(iter(matched_letters))
    return ""


def _attempt_correct(pred: str, gt_letter: str, options: list[dict[str, str]]) -> tuple[bool, str, str]:
    letters = [str(item.get("letter", "")).strip().upper() for item in options]
    letters = [letter for letter in letters if letter]
    max_letter = max(letters) if letters else "D"

    by_letter = _extract_letter(pred, max_letter=max_letter)
    if by_letter:
        return by_letter == gt_letter, by_letter, "letter"

    by_text = _predict_letter_from_text(pred, options=options)
    if by_text:
        return by_text == gt_letter, by_text, "text"

    return False, "", "none"


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate LHRS-Bench with the paper 4-repeat strict rule.")
    parser.add_argument("--data", type=str, default="benchmark/lhrsbench/data/lhrsbench_attempts_r4_seed42.jsonl")
    parser.add_argument("--preds", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--num-repeats", type=int, default=4)
    args = parser.parse_args()

    data_path = _resolve_from_project(args.data)
    preds_path = _resolve_from_project(args.preds)
    out_path = _resolve_from_project(args.output)

    if not data_path.is_file():
        raise FileNotFoundError(f"Missing data jsonl: {data_path}")
    if not preds_path.is_file():
        raise FileNotFoundError(f"Missing preds jsonl: {preds_path}")

    gt_rows = _read_jsonl(data_path)
    pred_rows = _read_jsonl(preds_path)

    gt_by_uid = {str(row["uid"]): row for row in gt_rows}
    pred_by_uid: dict[str, dict[str, Any]] = {}
    for row in pred_rows:
        uid = str(row.get("uid", "")).strip()
        if uid:
            pred_by_uid[uid] = row

    per_question_attempts: dict[int, list[dict[str, Any]]] = defaultdict(list)
    parse_mode_stat: dict[str, int] = defaultdict(int)

    for uid, gt in gt_by_uid.items():
        qid = int(gt["qid"])
        pred_row = pred_by_uid.get(uid)
        pred_missing = pred_row is None
        raw_pred = str(pred_row.get("prediction", "")).strip() if pred_row else ""
        ok, pred_letter, mode = _attempt_correct(
            pred=raw_pred,
            gt_letter=str(gt["answer_letter"]).strip().upper(),
            options=list(gt.get("options", [])),
        )
        parse_mode_stat[mode] += 1
        per_question_attempts[qid].append(
            {
                "uid": uid,
                "attempt": int(gt["attempt"]),
                "ok": bool(ok),
                "pred_letter": pred_letter,
                "parse_mode": mode,
                "pred_missing": bool(pred_missing),
                "type_ids": list(gt.get("type_ids", [])),
            }
        )

    qids = sorted(per_question_attempts.keys())
    q_correct_map: dict[int, bool] = {}
    q_missing_attempts: dict[int, int] = {}
    by_type: dict[str, list[int]] = defaultdict(list)

    for qid in qids:
        attempt_rows = sorted(per_question_attempts[qid], key=lambda x: int(x["attempt"]))
        num_attempts = len(attempt_rows)
        all_ok = num_attempts == int(args.num_repeats) and all(bool(row["ok"]) for row in attempt_rows)
        q_correct_map[qid] = bool(all_ok)
        missing_pred_count = sum(1 for row in attempt_rows if bool(row.get("pred_missing", False)))
        if missing_pred_count > 0:
            q_missing_attempts[qid] = int(missing_pred_count)

        type_ids = [str(type_id) for type_id in attempt_rows[0].get("type_ids", [])] if attempt_rows else []
        for type_id in type_ids:
            by_type[type_id].append(1 if all_ok else 0)

    total_questions = len(qids)
    total_correct = sum(1 for ok in q_correct_map.values() if ok)
    total_acc = float(total_correct) / float(total_questions) if total_questions > 0 else 0.0

    type_name_map: dict[str, str] = {}
    for gt in gt_rows:
        ids = [str(x) for x in gt.get("type_ids", [])]
        names = [str(x) for x in gt.get("type_names", [])]
        for type_id, type_name in zip(ids, names):
            if type_id not in type_name_map:
                type_name_map[type_id] = type_name

    type_metrics: list[dict[str, Any]] = []
    for type_id in sorted(by_type.keys(), key=lambda x: int(x)):
        values = by_type[type_id]
        acc = float(sum(values)) / float(len(values)) if values else 0.0
        type_metrics.append(
            {
                "type_id": type_id,
                "type_name": type_name_map.get(type_id, f"type_{type_id}"),
                "num_questions": int(len(values)),
                "accuracy": acc,
                "accuracy_pct": round(acc * 100.0, 2),
            }
        )

    attempts_total = len(gt_rows)
    attempts_correct = 0
    for qid in qids:
        attempts_correct += sum(1 for row in per_question_attempts[qid] if bool(row["ok"]))
    attempts_acc = float(attempts_correct) / float(attempts_total) if attempts_total > 0 else 0.0

    summary = {
        "preds": str(preds_path),
        "data": str(data_path),
        "num_repeats": int(args.num_repeats),
        "rule_note": "Paper rule: repeat 4 times, shuffle choices each time, all attempts must be correct.",
        "num_questions": int(total_questions),
        "num_attempts": int(attempts_total),
        "question_correct": int(total_correct),
        "question_accuracy": total_acc,
        "question_accuracy_pct": round(total_acc * 100.0, 2),
        "attempt_correct": int(attempts_correct),
        "attempt_accuracy": attempts_acc,
        "attempt_accuracy_pct": round(attempts_acc * 100.0, 2),
        "parse_mode_count": dict(parse_mode_stat),
        "missing_attempt_questions": len(q_missing_attempts),
        "missing_attempt_examples": [
            {"qid": int(qid), "missing": int(missing)} for qid, missing in list(q_missing_attempts.items())[:20]
        ],
        "metrics_by_type": type_metrics,
        "metrics_by_type_pct": {row["type_name"]: row["accuracy_pct"] for row in type_metrics},
        "avg_pct": round(total_acc * 100.0, 2),
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"[OK] questions={total_questions} acc={summary['question_accuracy_pct']:.2f}%")
    print(f"[OK] wrote: {out_path}")


if __name__ == "__main__":
    main()
