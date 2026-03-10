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


def _rel_to_project(path: Path) -> str:
    p = Path(path)
    try:
        return str(p.resolve().relative_to(_project_root()))
    except Exception:
        return str(p.resolve())


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            rows.append(json.loads(ln))
    return rows


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def compute_iou_xyxy(bbox1: list[int], bbox2: list[int], *, return_parts: bool = False):
    """
    复刻 VRSBench 官方 eval_utils.py 的 computeIoU（面积计算含 +1）。
    bbox: [x1, y1, x2, y2]
    """

    x1, y1, x2, y2 = bbox1
    x3, y3, x4, y4 = bbox2
    ix1 = max(x1, x3)
    iy1 = max(y1, y3)
    ix2 = min(x2, x4)
    iy2 = min(y2, y4)

    inter_w = max(0, ix2 - ix1 + 1)
    inter_h = max(0, iy2 - iy1 + 1)
    inter_area = inter_w * inter_h

    area1 = (x2 - x1 + 1) * (y2 - y1 + 1)
    area2 = (x4 - x3 + 1) * (y4 - y3 + 1)
    union = area1 + area2 - inter_area
    # 保持与官方实现一致，不对 union 做额外保护；如遇异常由上层 try/except 处理。
    iou = float(inter_area) / float(union)

    if return_parts:
        return float(iou), int(inter_area), int(union)
    return float(iou)


_SIGNED_INT_RE = re.compile(r"-?\d+")


def _parse_first4_ints(text: str, *, clamp_to_100: bool = False) -> list[int] | None:
    nums = _SIGNED_INT_RE.findall(text or "")
    if len(nums) < 4:
        return None
    vals = [int(nums[0]), int(nums[1]), int(nums[2]), int(nums[3])]
    if clamp_to_100:
        vals = [max(0, min(100, v)) for v in vals]
    return vals


def _eval_split(rows: list[dict[str, Any]], *, unique_filter: str, thresholds: list[float]) -> dict[str, Any]:
    """
    unique_filter:
      - "all": 全部样本
      - "unique": is_unique==True
      - "non_unique": is_unique==False
    """

    if unique_filter not in {"all", "unique", "non_unique"}:
        raise ValueError(f"Invalid unique_filter: {unique_filter}")

    total = 0
    hit = {str(t): 0 for t in thresholds}
    mean_iou_sum = 0.0
    cum_i = 0
    cum_u = 0

    for r in rows:
        is_unique = bool(r.get("is_unique", r.get("unique", False)))
        if unique_filter == "unique" and not is_unique:
            continue
        if unique_filter == "non_unique" and is_unique:
            continue

        total += 1
        gt = _parse_first4_ints(str(r.get("ground_truth", "")))
        pred = _parse_first4_ints(str(r.get("answer", "")), clamp_to_100=True)
        if gt is None or len(gt) != 4:
            # 标注缺失属于数据异常，直接报错更安全。
            raise ValueError(f"Invalid ground_truth for qid={r.get('qid')}: {r.get('ground_truth')}")
        if pred is None or len(pred) != 4:
            continue

        try:
            iou, inter, union = compute_iou_xyxy(gt, pred, return_parts=True)
        except Exception:
            continue
        mean_iou_sum += float(iou)
        cum_i += int(inter)
        cum_u += int(union)
        for t in thresholds:
            if float(iou) >= float(t):
                hit[str(t)] += 1

    # 官方 notebook 的 Acc/meanIoU 是按 total 归一化。
    denom = float(max(1, total))
    acc = {f"Acc@{t}": float(hit[str(t)]) / denom * 100.0 for t in thresholds}
    mean_iou = float(mean_iou_sum) / denom * 100.0
    cum_iou = float(cum_i) / float(cum_u) * 100.0 if cum_u > 0 else 0.0

    return {
        "total": int(total),
        **acc,
        "meanIoU": float(mean_iou),
        "cumIoU": float(cum_iou),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate VRSBench referring (visual grounding) predictions.")
    parser.add_argument("--preds", type=str, required=True, help="Predictions jsonl path.")
    parser.add_argument(
        "--meta",
        type=str,
        default="benchmark/vrsbench/data/vrsbench_referring_meta.json",
        help="Prepared meta json path (optional, used for expected sample count).",
    )
    parser.add_argument("--output", type=str, default="", help="Output summary json path.")
    parser.add_argument("--max-items", type=int, default=0, help="Debug only: evaluate first N rows after sorting by qid.")
    args = parser.parse_args()

    preds_path = _resolve_from_project(args.preds)
    if not preds_path.is_file():
        raise FileNotFoundError(f"Missing preds: {preds_path}")

    rows = read_jsonl(preds_path)
    rows = sorted(rows, key=lambda x: int(x.get("qid", 0)))
    if args.max_items and int(args.max_items) > 0:
        rows = rows[: int(args.max_items)]

    expected_num_samples = None
    meta_path = _resolve_from_project(args.meta) if str(args.meta).strip() else None
    meta = None
    if meta_path and meta_path.is_file():
        meta = read_json(meta_path)
        try:
            expected_num_samples = int(meta.get("num_samples"))
        except Exception:
            expected_num_samples = None

    thresholds = [0.5, 0.7]
    splits = {
        "all": _eval_split(rows, unique_filter="all", thresholds=thresholds),
        "unique": _eval_split(rows, unique_filter="unique", thresholds=thresholds),
        "non_unique": _eval_split(rows, unique_filter="non_unique", thresholds=thresholds),
    }

    # 附带少量“推理配置”信息：从第一条样本里取，便于对照复现。
    cfg: dict[str, Any] = {}
    if rows:
        first = rows[0]
        for k in [
            "model",
            "model_dir",
            "qwen_model_dir",
            "dinov3_dir",
            "merger_ckpt",
            "lora_dir",
            "prompt_template",
            "max_new_tokens",
            "requested_batch_size",
            "dtype",
            "image_size",
            "smart_resize_min_pixels",
            "smart_resize_max_pixels",
            "resize_mode",
            "decode_strategy",
            "do_sample",
            "temperature",
            "top_p",
            "top_k",
            "num_beams",
            "repetition_penalty",
            "no_repeat_ngram_size",
            "seed",
        ]:
            if k in first:
                cfg[k] = first.get(k)

    out_path = (
        _resolve_from_project(args.output) if str(args.output).strip() else preds_path.with_name(preds_path.stem + "_eval_summary.json")
    )
    summary: dict[str, Any] = {
        "task": "vrsbench_referring",
        "thresholds": thresholds,
        "expected_num_samples": expected_num_samples,
        "num_rows_in_file": int(len(rows)),
        "preds": _rel_to_project(preds_path),
        "meta": _rel_to_project(meta_path) if meta_path and meta_path.is_file() else "",
        "config_hint": cfg,
        "splits": splits,
    }

    write_json(out_path, summary)
    print(f"[OK] Wrote summary: {out_path}")
    if expected_num_samples is not None and int(len(rows)) != int(expected_num_samples):
        print(
            f"[WARN] num_rows_in_file({len(rows)}) != expected_num_samples({expected_num_samples}). "
            "If the run is incomplete, resume generation then re-evaluate."
        )


if __name__ == "__main__":
    main()
