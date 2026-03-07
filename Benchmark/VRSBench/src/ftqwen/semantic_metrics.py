from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal


BertScoreAgg = Literal["max", "mean"]


@dataclass(frozen=True)
class BertScoreMetrics:
    precision: float
    recall: float
    f1: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "P": float(self.precision),
            "R": float(self.recall),
            "F1": float(self.f1),
        }


def compute_bertscore(
    refs: dict[str, list[str]],
    preds: dict[str, str],
    *,
    model_type: str = "roberta-large",
    agg: BertScoreAgg = "max",
    idf: bool = False,
    batch_size: int = 32,
    device: str | None = None,
) -> tuple[BertScoreMetrics, dict[str, Any]]:
    """Compute BERTScore for image captioning (multi-reference).

    - RSICD 每张图有多条参考 caption；这里支持两种聚合方式：
      - agg=max：按 F1 选最匹配的那条参考（然后取对应的 P/R/F1）
      - agg=mean：对同一张图的多条参考取均值
    - 返回值分两部分：
      1) BertScoreMetrics：总体均值（对所有图取平均）
      2) meta：本次计算的配置与规模信息（便于复现）
    """
    try:
        from bert_score import score as bert_score_score  # type: ignore
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(
            "Missing dependency: bert-score. Please install it via `python -m pip install bert-score==0.3.13`."
        ) from e

    try:
        import torch
    except Exception as e:  # noqa: BLE001
        raise RuntimeError("Missing dependency: torch") from e

    if device is None or not str(device).strip():
        device = "cuda" if torch.cuda.is_available() else "cpu"

    imgids = sorted(refs.keys(), key=lambda x: int(x))
    if not imgids:
        raise ValueError("Empty refs.")

    cand_flat: list[str] = []
    ref_flat: list[str] = []
    n_refs_per_image: list[int] = []

    for imgid in imgids:
        if imgid not in preds:
            raise KeyError(f"Missing prediction for imgid={imgid}")
        cand = str(preds[imgid]).strip()
        if not cand:
            raise ValueError(f"Empty prediction for imgid={imgid}")

        ref_list = [str(r).strip() for r in refs.get(imgid, []) if str(r).strip()]
        if not ref_list:
            raise ValueError(f"Empty references for imgid={imgid}")

        n_refs_per_image.append(len(ref_list))
        for r in ref_list:
            cand_flat.append(cand)
            ref_flat.append(r)

    # bert_score.score 返回的是每对句子的 P/R/F1（torch.Tensor）。
    P, R, F1 = bert_score_score(
        cand_flat,
        ref_flat,
        model_type=str(model_type),
        idf=bool(idf),
        batch_size=int(batch_size),
        device=str(device),
        verbose=False,
    )

    p_list = P.detach().cpu().tolist()
    r_list = R.detach().cpu().tolist()
    f1_list = F1.detach().cpu().tolist()

    p_sum = 0.0
    r_sum = 0.0
    f1_sum = 0.0
    offset = 0
    for n in n_refs_per_image:
        p_slice = p_list[offset : offset + n]
        r_slice = r_list[offset : offset + n]
        f1_slice = f1_list[offset : offset + n]
        offset += n

        if agg == "max":
            # 按 F1 选最匹配的参考，并取同一条参考对应的 P/R/F1。
            best_j = max(range(n), key=lambda j: float(f1_slice[j]))
            p_sum += float(p_slice[best_j])
            r_sum += float(r_slice[best_j])
            f1_sum += float(f1_slice[best_j])
        elif agg == "mean":
            p_sum += float(sum(p_slice) / n)
            r_sum += float(sum(r_slice) / n)
            f1_sum += float(sum(f1_slice) / n)
        else:
            raise ValueError(f"Unsupported agg={agg}")

    num_images = len(n_refs_per_image)
    metrics = BertScoreMetrics(
        precision=p_sum / num_images,
        recall=r_sum / num_images,
        f1=f1_sum / num_images,
    )
    meta = {
        "model_type": str(model_type),
        "agg": str(agg),
        "idf": bool(idf),
        "device": str(device),
        "batch_size": int(batch_size),
        "num_images": int(num_images),
        "num_pairs": int(len(cand_flat)),
    }
    return metrics, meta
