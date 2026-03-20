from __future__ import annotations

from dataclasses import dataclass
from typing import Any


def _to_coco_dict_raw(refs: dict[str, list[str]], preds: dict[str, str]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build raw (untokenized) COCO-style dicts.

    PTBTokenizer 期望的输入格式是：
      {imgid: [{"caption": "..."} , ...]}
    """
    gts: dict[str, Any] = {}
    res: dict[str, Any] = {}
    for img_id, ref_list in refs.items():
        if img_id not in preds:
            raise KeyError(f"Missing prediction for imgid={img_id}")
        gts[img_id] = [{"caption": str(r)} for r in ref_list]
        res[img_id] = [{"caption": str(preds[img_id])}]
    return gts, res


def _tokenize(gts_raw: dict[str, Any], res_raw: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    # 输出给 Bleu/Cider/Rouge 的格式：
    #   gts: {id: [ref1_tok_str, ref2_tok_str, ...]}
    #   res: {id: [hyp_tok_str]}
    try:
        from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer  # type: ignore
    except Exception as e:  # noqa: BLE001
        raise RuntimeError("Missing dependency: pycocoevalcap.tokenizer.ptbtokenizer") from e

    # PTBTokenizer 内部会调用 `java`。
    import contextlib
    import io

    with contextlib.redirect_stdout(io.StringIO()):
        tok = PTBTokenizer()
        gts_tok = tok.tokenize(gts_raw)
        res_tok = tok.tokenize(res_raw)
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
    """Compute CIDEr / BLEU-4 / ROUGE-L.

    - 使用 COCO/Caption 常见的 PTBTokenizer（会调用 `java`）。
    """
    try:
        from pycocoevalcap.bleu.bleu import Bleu  # type: ignore
        from pycocoevalcap.cider.cider import Cider  # type: ignore
        from pycocoevalcap.rouge.rouge import Rouge  # type: ignore
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(
            "Missing dependency: pycocoevalcap. "
            "Please install it via `python -m pip install pycocoevalcap==1.2`."
        ) from e

    gts_raw, res_raw = _to_coco_dict_raw(refs, preds)
    gts, res = _tokenize(gts_raw, res_raw)

    # pycocoevalcap 的 BLEU 实现会往 stdout 打印信息，这里统一静音，避免污染日志。
    import contextlib
    import io

    with contextlib.redirect_stdout(io.StringIO()):
        bleu_scores, _ = Bleu(4).compute_score(gts, res)
        cider_score, _ = Cider().compute_score(gts, res)
        rouge_score, _ = Rouge().compute_score(gts, res)

    return CaptionMetrics(
        cider=float(cider_score),
        bleu4=float(bleu_scores[3]),
        rouge_l=float(rouge_score),
    )
