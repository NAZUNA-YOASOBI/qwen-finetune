from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from shared.common import parse_shard_weights, read_jsonl


@dataclass(frozen=True)
class PredictionIntegrityReport:
    status: str
    pred_path: Path
    data_path: Path
    expected_total: int
    observed_total: int
    unique_pred_total: int
    missing_total: int
    extra_total: int
    duplicate_total: int
    empty_answer_total: int
    missing_key_total: int
    missing_examples: list[str]
    extra_examples: list[str]
    duplicate_examples: list[str]
    empty_answer_examples: list[str]
    parse_error: str | None = None

    @property
    def ok(self) -> bool:
        return self.status == "complete"

    @property
    def resumable(self) -> bool:
        return self.status == "incomplete"

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": str(self.status),
            "pred_path": str(self.pred_path),
            "data_path": str(self.data_path),
            "expected_total": int(self.expected_total),
            "observed_total": int(self.observed_total),
            "unique_pred_total": int(self.unique_pred_total),
            "missing_total": int(self.missing_total),
            "extra_total": int(self.extra_total),
            "duplicate_total": int(self.duplicate_total),
            "empty_answer_total": int(self.empty_answer_total),
            "missing_key_total": int(self.missing_key_total),
            "missing_examples": list(self.missing_examples),
            "extra_examples": list(self.extra_examples),
            "duplicate_examples": list(self.duplicate_examples),
            "empty_answer_examples": list(self.empty_answer_examples),
            "parse_error": self.parse_error,
        }

    def format_message(self) -> str:
        parts = [
            f"status={self.status}",
            f"expected_total={self.expected_total}",
            f"observed_total={self.observed_total}",
            f"unique_pred_total={self.unique_pred_total}",
            f"missing_total={self.missing_total}",
            f"extra_total={self.extra_total}",
            f"duplicate_total={self.duplicate_total}",
            f"empty_answer_total={self.empty_answer_total}",
            f"missing_key_total={self.missing_key_total}",
        ]
        if self.missing_examples:
            parts.append(f"missing_examples={self.missing_examples[:5]}")
        if self.extra_examples:
            parts.append(f"extra_examples={self.extra_examples[:5]}")
        if self.duplicate_examples:
            parts.append(f"duplicate_examples={self.duplicate_examples[:5]}")
        if self.empty_answer_examples:
            parts.append(f"empty_answer_examples={self.empty_answer_examples[:5]}")
        if self.parse_error:
            parts.append(f"parse_error={self.parse_error}")
        return "; ".join(parts)


def _sample_strings(values: list[str], *, limit: int = 20) -> list[str]:
    return list(values[:limit])
def _slice_rows_quiet(
    rows: list[dict[str, Any]],
    *,
    world_size: int,
    rank: int,
    weights: str,
) -> list[dict[str, Any]]:
    parsed = parse_shard_weights(weights, world_size=int(world_size))
    total = len(rows)
    if parsed is None:
        return [row for idx, row in enumerate(rows) if (idx % int(world_size)) == int(rank)]
    denom = int(sum(parsed))
    left = int(sum(parsed[: int(rank)]))
    right = int(sum(parsed[: int(rank) + 1]))
    start = (total * left) // denom
    end = (total * right) // denom
    return rows[start:end]


def load_benchmark_row_map(
    data_path: str | Path,
    *,
    key_field: str = "question_id",
) -> dict[str, dict[str, Any]]:
    data = Path(data_path)
    if not data.is_file():
        raise FileNotFoundError(f"Missing benchmark data file: {data}")

    data_rows = read_jsonl(data)
    if not data_rows:
        raise ValueError(f"Benchmark data file is empty: {data}")

    row_map: dict[str, dict[str, Any]] = {}
    for idx, row in enumerate(data_rows, start=1):
        if key_field not in row:
            raise ValueError(f"Benchmark data row missing `{key_field}` at {data}:{idx}")
        key = str(row[key_field])
        if key in row_map:
            raise ValueError(f"Duplicated `{key_field}` in benchmark data: {data}:{idx} -> {key}")
        row_map[key] = row
    return row_map


def inspect_prediction_file(
    pred_path: str | Path,
    data_path: str | Path,
    *,
    key_field: str = "question_id",
    answer_field: str = "answer",
    shard_world_size: int = 1,
    shard_rank: int = 0,
    shard_weights: str = "",
) -> PredictionIntegrityReport:
    pred = Path(pred_path)
    data = Path(data_path)
    data_row_map = load_benchmark_row_map(data, key_field=key_field)
    data_rows = list(data_row_map.values())
    if int(shard_world_size) > 1:
        data_rows = _slice_rows_quiet(
            data_rows,
            world_size=int(shard_world_size),
            rank=int(shard_rank),
            weights=str(shard_weights),
        )
    expected_keys = [str(row[key_field]) for row in data_rows]
    expected_set = set(expected_keys)

    if not pred.is_file():
        return PredictionIntegrityReport(
            status="incomplete",
            pred_path=pred,
            data_path=data,
            expected_total=len(expected_keys),
            observed_total=0,
            unique_pred_total=0,
            missing_total=len(expected_keys),
            extra_total=0,
            duplicate_total=0,
            empty_answer_total=0,
            missing_key_total=0,
            missing_examples=_sample_strings(expected_keys),
            extra_examples=[],
            duplicate_examples=[],
            empty_answer_examples=[],
            parse_error=None,
        )

    try:
        pred_rows = read_jsonl(pred, allow_truncated_last_line=True)
    except Exception as e:
        return PredictionIntegrityReport(
            status="invalid",
            pred_path=pred,
            data_path=data,
            expected_total=len(expected_keys),
            observed_total=0,
            unique_pred_total=0,
            missing_total=len(expected_keys),
            extra_total=0,
            duplicate_total=0,
            empty_answer_total=0,
            missing_key_total=0,
            missing_examples=_sample_strings(expected_keys),
            extra_examples=[],
            duplicate_examples=[],
            empty_answer_examples=[],
            parse_error=str(e),
        )
    pred_keys: list[str] = []
    pred_set: set[str] = set()
    duplicate_keys: list[str] = []
    empty_answer_keys: list[str] = []
    missing_key_total = 0

    for row in pred_rows:
        raw_key = row.get(key_field, None)
        if raw_key in (None, ""):
            missing_key_total += 1
            continue
        key = str(raw_key)
        pred_keys.append(key)
        if key in pred_set:
            duplicate_keys.append(key)
        else:
            pred_set.add(key)
        if answer_field:
            answer = str(row.get(answer_field, "")).strip()
            if not answer:
                empty_answer_keys.append(key)

    missing_keys = [key for key in expected_keys if key not in pred_set]
    extra_keys = sorted(pred_set - expected_set)

    has_invalid_structure = bool(
        duplicate_keys
        or extra_keys
        or empty_answer_keys
        or missing_key_total > 0
    )
    if not pred_rows or missing_keys:
        status = "invalid" if has_invalid_structure else "incomplete"
    else:
        status = "invalid" if has_invalid_structure else "complete"

    return PredictionIntegrityReport(
        status=status,
        pred_path=pred,
        data_path=data,
        expected_total=len(expected_keys),
        observed_total=len(pred_rows),
        unique_pred_total=len(pred_set),
        missing_total=len(missing_keys),
        extra_total=len(extra_keys),
        duplicate_total=len(duplicate_keys),
        empty_answer_total=len(empty_answer_keys),
        missing_key_total=int(missing_key_total),
        missing_examples=_sample_strings(missing_keys),
        extra_examples=_sample_strings(extra_keys),
        duplicate_examples=_sample_strings(duplicate_keys),
        empty_answer_examples=_sample_strings(empty_answer_keys),
        parse_error=None,
    )


def assert_prediction_integrity(
    pred_path: str | Path,
    data_path: str | Path,
    *,
    key_field: str = "question_id",
    answer_field: str = "answer",
    shard_world_size: int = 1,
    shard_rank: int = 0,
    shard_weights: str = "",
) -> PredictionIntegrityReport:
    report = inspect_prediction_file(
        pred_path,
        data_path,
        key_field=key_field,
        answer_field=answer_field,
        shard_world_size=int(shard_world_size),
        shard_rank=int(shard_rank),
        shard_weights=str(shard_weights),
    )
    if not report.ok:
        raise ValueError(
            "Prediction file failed integrity check: "
            f"{report.pred_path} against {report.data_path}; {report.format_message()}"
        )
    return report
