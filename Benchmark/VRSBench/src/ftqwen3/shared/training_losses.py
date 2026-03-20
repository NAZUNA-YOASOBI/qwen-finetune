from __future__ import annotations

import torch
import torch.nn.functional as F


def causal_lm_per_sample_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    ignore_index: int = -100,
) -> torch.Tensor:
    """计算因果语言模型的逐样本 loss。

    做法与 transformers 的默认因果语言模型 loss 保持同样的时序对齐：
    - 第 t 个位置的 logits 预测第 t+1 个标签；
    - prompt 与 padding 位置通过 `ignore_index` 排除；
    - 先对每条样本自己的有效 token 求平均，再返回逐样本 loss。
    """
    if logits.ndim != 3:
        raise ValueError(f"Expected logits with shape (batch, seq, vocab), got {tuple(logits.shape)}")
    if labels.ndim != 2:
        raise ValueError(f"Expected labels with shape (batch, seq), got {tuple(labels.shape)}")
    if logits.shape[0] != labels.shape[0] or logits.shape[1] != labels.shape[1]:
        raise ValueError(
            "Logits/labels shape mismatch: "
            f"logits={tuple(logits.shape)}, labels={tuple(labels.shape)}"
        )

    logits = logits.float()
    shift_labels = F.pad(labels, (0, 1), value=ignore_index)[..., 1:].contiguous()
    flat_logits = logits.reshape(-1, logits.shape[-1])
    flat_labels = shift_labels.reshape(-1).to(device=flat_logits.device)

    token_loss = F.cross_entropy(
        flat_logits,
        flat_labels,
        ignore_index=ignore_index,
        reduction="none",
    ).view_as(shift_labels)

    valid_mask = shift_labels.ne(ignore_index)
    valid_count = valid_mask.sum(dim=-1)
    if bool((valid_count <= 0).any().item()):
        bad_positions = torch.nonzero(valid_count <= 0, as_tuple=False).view(-1).tolist()
        raise RuntimeError(
            "Found samples without supervised tokens while computing sample-average loss: "
            f"batch_positions={bad_positions}"
        )

    valid_mask_f = valid_mask.to(dtype=token_loss.dtype)
    sample_loss = (token_loss * valid_mask_f).sum(dim=-1) / valid_count.to(dtype=token_loss.dtype)
    return sample_loss


def causal_lm_sample_average_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    ignore_index: int = -100,
) -> torch.Tensor:
    """返回真正的样本平均 loss。"""
    return causal_lm_per_sample_loss(logits, labels, ignore_index=ignore_index).mean()
