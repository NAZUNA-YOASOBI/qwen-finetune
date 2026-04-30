from __future__ import annotations

import argparse
from pathlib import Path

from local_vqa_eval_utils import parse_prediction_datasets, run_local_vqa_eval_for_dataset, write_json


def resolve_path(root: Path, raw: str) -> Path:
    candidate = Path(raw)
    if candidate.is_absolute():
        return candidate
    return (root / candidate).resolve()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run local VQA evaluation on existing prediction jsonl files.")
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--eval-dir", type=str, required=True)
    parser.add_argument("--datasets", type=str, default="all")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    current = Path(__file__).resolve().parent
    project_root: Path | None = None
    for candidate in [current, *current.parents]:
        if (candidate / "VRSBench").is_dir() and (candidate / "fine-tune-qwen3-vl").is_dir():
            project_root = candidate
            break
    if project_root is None:
        raise FileNotFoundError(f"Cannot locate project root from {__file__}")
    output_dir = resolve_path(project_root, str(args.output_dir))
    eval_dir = resolve_path(project_root, str(args.eval_dir))
    output_dir.mkdir(parents=True, exist_ok=True)
    eval_dir.mkdir(parents=True, exist_ok=True)

    selected = parse_prediction_datasets(str(args.datasets), output_dir=output_dir)
    if not selected:
        raise ValueError(f"no prediction jsonl files found under {output_dir}")

    summary = {
        "output_dir": str(output_dir),
        "eval_dir": str(eval_dir),
        "eval_mode": "local_exact_match",
        "datasets": {},
    }
    for dataset_name in selected:
        pred_path = output_dir / f"{dataset_name}.jsonl"
        if not pred_path.is_file():
            raise FileNotFoundError(f"missing prediction file: {pred_path}")
        summary["datasets"][dataset_name] = run_local_vqa_eval_for_dataset(
            pred_path=pred_path,
            eval_dir=eval_dir,
        )
    write_json(eval_dir / "vqa_summary.json", summary)
    print(f"[OK] Wrote VQA summary: {eval_dir / 'vqa_summary.json'}", flush=True)


if __name__ == "__main__":
    main()
