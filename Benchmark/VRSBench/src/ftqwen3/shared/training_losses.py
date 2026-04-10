from __future__ import annotations

import torch
import torch.nn.functional as F

TOKEN_LOSS_CHUNK_SIZE = 32


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

    shift_labels = F.pad(labels, (0, 1), value=ignore_index)[..., 1:].contiguous()
    valid_mask = shift_labels.ne(ignore_index)
    valid_count = valid_mask.sum(dim=-1)
    invalid_positions = torch.nonzero(valid_count <= 0, as_tuple=False).view(-1)
    if invalid_positions.numel() > 0:
        bad_positions = [int(x) for x in invalid_positions.detach().cpu().tolist()]
        raise RuntimeError(
            "Found samples without supervised tokens while computing sample-average loss: "
            f"batch_positions={bad_positions}"
        )

    sample_losses: list[torch.Tensor] = []
    for sample_index in range(int(logits.shape[0])):
        sample_valid_mask = valid_mask[sample_index]
        valid_positions = torch.nonzero(sample_valid_mask, as_tuple=False).view(-1)
        sample_loss_sum = logits.new_zeros((), dtype=torch.float32)

        for start in range(0, int(valid_positions.numel()), int(TOKEN_LOSS_CHUNK_SIZE)):
            end = int(start + int(TOKEN_LOSS_CHUNK_SIZE))
            chunk_positions = valid_positions[start:end]
            chunk_logits = logits[sample_index, chunk_positions].float()
            chunk_labels = shift_labels[sample_index, chunk_positions].to(device=chunk_logits.device)
            sample_loss_sum = sample_loss_sum + F.cross_entropy(
                chunk_logits,
                chunk_labels,
                reduction="sum",
            )

        sample_loss = sample_loss_sum / float(valid_positions.numel())
        sample_losses.append(sample_loss)

    return torch.stack(sample_losses, dim=0)


def causal_lm_sample_average_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    ignore_index: int = -100,
) -> torch.Tensor:
    """返回真正的样本平均 loss。"""
    return causal_lm_per_sample_loss(logits, labels, ignore_index=ignore_index).mean()
