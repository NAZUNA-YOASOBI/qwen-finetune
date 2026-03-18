from __future__ import annotations

import argparse
import json
from pathlib import Path


def _project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _resolve_from_project(path: str | Path) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    return (_project_root() / p).resolve()


def main() -> None:
    parser = argparse.ArgumentParser(description="Check whether a GeoChat prediction file is complete and reusable.")
    parser.add_argument("--preds", type=str, required=True)
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--key-field", type=str, default="question_id")
    parser.add_argument("--answer-field", type=str, default="answer")
    parser.add_argument("--shard-world-size", type=int, default=1)
    parser.add_argument("--shard-rank", type=int, default=0)
    parser.add_argument("--shard-weights", type=str, default="")
    args = parser.parse_args()

    import sys

    sys.path.insert(0, str(_project_root() / "src"))
    from geochatbench_eval_utils import inspect_prediction_file

    report = inspect_prediction_file(
        _resolve_from_project(args.preds),
        _resolve_from_project(args.data),
        key_field=str(args.key_field),
        answer_field=str(args.answer_field),
        shard_world_size=int(args.shard_world_size),
        shard_rank=int(args.shard_rank),
        shard_weights=str(args.shard_weights),
    )
    print(json.dumps(report.to_dict(), ensure_ascii=False, indent=2))
    if report.ok:
        raise SystemExit(0)
    if report.resumable:
        raise SystemExit(3)
    raise SystemExit(4)


if __name__ == "__main__":
    main()
