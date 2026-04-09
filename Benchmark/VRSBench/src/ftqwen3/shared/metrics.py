from __future__ import annotations

import contextlib
import io
from dataclasses import dataclass
from typing import Any, Literal

try:
    import torch
    TORCH_IMPORT_ERROR = None
except Exception as exc:
    torch = None
    TORCH_IMPORT_ERROR = exc

try:
    from bert_score import score as bert_score_score  # type: ignore
    BERT_SCORE_IMPORT_ERROR = None
except Exception as exc:
    bert_score_score = None
    BERT_SCORE_IMPORT_ERROR = exc

try:
    from pycocoevalcap.bleu.bleu import Bleu  # type: ignore
    from pycocoevalcap.cider.cider import Cider  # type: ignore
    from pycocoevalcap.rouge.rouge import Rouge  # type: ignore
    from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer  # type: ignore
    PYCOCOEVALCAP_IMPORT_ERROR = None
except Exception as exc:
    Bleu = None
    Cider = None
    Rouge = None
    PTBTokenizer = None
    PYCOCOEVALCAP_IMPORT_ERROR = exc


def _to_coco_dict_raw(refs: dict[str, list[str]], preds: dict[str, str]) -> tuple[dict[str, Any], dict[str, Any]]:
    gts: dict[str, Any] = {}
    res: dict[str, Any] = {}
    for img_id, ref_list in refs.items():
        if img_id not in preds:
            raise KeyError(f"Missing prediction for imgid={img_id}")
        gts[img_id] = [{"caption": str(ref)} for ref in ref_list]
        res[img_id] = [{"caption": str(preds[img_id])}]
    return gts, res


def _tokenize(gts_raw: dict[str, Any], res_raw: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    if PTBTokenizer is None:
        raise RuntimeError("Missing dependency: pycocoevalcap.tokenizer.ptbtokenizer") from PYCOCOEVALCAP_IMPORT_ERROR

    with contextlib.redirect_stdout(io.StringIO()):
        tokenizer = PTBTokenizer()
        gts_tok = tokenizer.tokenize(gts_raw)
        res_tok = tokenizer.tokenize(res_raw)
    return gts_tok, res_tok


@dataclass(frozen=True)
class CaptionMetrics:
    cider: float
    bleu4: float
    rouge_l: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "CIDEr": float(self.cider),
            "BLEU-4": float(self.bleu4),
            "ROUGE-L": float(self.rouge_l),
        }


def compute_metrics(refs: dict[str, list[str]], preds: dict[str, str]) -> CaptionMetrics:
    if Bleu is None or Cider is None or Rouge is None:
        raise RuntimeError(
            "Missing dependency: pycocoevalcap. "
            "Please install it via `python -m pip install pycocoevalcap==1.2`."
        ) from PYCOCOEVALCAP_IMPORT_ERROR

    gts_raw, res_raw = _to_coco_dict_raw(refs, preds)
    gts, res = _tokenize(gts_raw, res_raw)

    with contextlib.redirect_stdout(io.StringIO()):
        bleu_scores, _ = Bleu(4).compute_score(gts, res)
        cider_score, _ = Cider().compute_score(gts, res)
        rouge_score, _ = Rouge().compute_score(gts, res)

    return CaptionMetrics(
        cider=float(cider_score),
        bleu4=float(bleu_scores[3]),
        rouge_l=float(rouge_score),
    )


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
    if bert_score_score is None:
        raise RuntimeError(
            "Missing dependency: bert-score. Please install it via `python -m pip install bert-score==0.3.13`."
        ) from BERT_SCORE_IMPORT_ERROR
    if torch is None:
        raise RuntimeError("Missing dependency: torch") from TORCH_IMPORT_ERROR

    if device is None or not str(device).strip():
        device = "cuda" if torch.cuda.is_available() else "cpu"

    imgids = sorted(refs.keys(), key=lambda value: int(value))
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

        ref_list = [str(ref).strip() for ref in refs.get(imgid, []) if str(ref).strip()]
        if not ref_list:
            raise ValueError(f"Empty references for imgid={imgid}")

        n_refs_per_image.append(len(ref_list))
        for ref in ref_list:
            cand_flat.append(cand)
            ref_flat.append(ref)

    precision_scores, recall_scores, f1_scores = bert_score_score(
        cand_flat,
        ref_flat,
        model_type=str(model_type),
        idf=bool(idf),
        batch_size=int(batch_size),
        device=str(device),
        verbose=False,
    )

    precision_list = precision_scores.detach().cpu().tolist()
    recall_list = recall_scores.detach().cpu().tolist()
    f1_list = f1_scores.detach().cpu().tolist()

    precision_sum = 0.0
    recall_sum = 0.0
    f1_sum = 0.0
    offset = 0
    for ref_count in n_refs_per_image:
        precision_slice = precision_list[offset : offset + ref_count]
        recall_slice = recall_list[offset : offset + ref_count]
        f1_slice = f1_list[offset : offset + ref_count]
        offset += ref_count

        if agg == "max":
            best_index = max(range(ref_count), key=lambda index: float(f1_slice[index]))
            precision_sum += float(precision_slice[best_index])
            recall_sum += float(recall_slice[best_index])
            f1_sum += float(f1_slice[best_index])
        elif agg == "mean":
            precision_sum += float(sum(precision_slice) / ref_count)
            recall_sum += float(sum(recall_slice) / ref_count)
            f1_sum += float(sum(f1_slice) / ref_count)
        else:
            raise ValueError(f"Unsupported agg={agg}")

    num_images = len(n_refs_per_image)
    metrics = BertScoreMetrics(
        precision=precision_sum / num_images,
        recall=recall_sum / num_images,
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
