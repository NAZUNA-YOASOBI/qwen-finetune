from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class QwenVisualTokenOutput:
    patch_tokens: torch.Tensor
    deepstack_patch_tokens: list[torch.Tensor]


def build_qwen_postshuffle_index(*, grid_t: int, grid_h: int, grid_w: int, merge_size: int) -> torch.LongTensor:
    """按 Qwen 原生视觉分支的 token 排列逻辑构造 postshuffle 索引。

    当前仅用于显式校验：
    - Qwen 原生视觉分支输出的 patch token 顺序
    - DINO/merger 侧使用的 build_postshuffle_index 顺序

    在单帧图像（grid_t=1）时，这两者应当完全一致。
    """
    t = int(grid_t)
    h = int(grid_h)
    w = int(grid_w)
    m = int(merge_size)
    if t <= 0 or h <= 0 or w <= 0 or m <= 0:
        raise ValueError(f"Invalid qwen postshuffle args: t={t}, h={h}, w={w}, merge={m}")
    if h % m != 0 or w % m != 0:
        raise ValueError(f"grid_h/grid_w must be divisible by merge_size. Got {h}x{w}, merge={m}")

    idx = torch.arange(t * h * w, dtype=torch.long)
    idx = (
        idx.view(
            t,
            h // m,
            m,
            w // m,
            m,
        )
        .permute(0, 1, 3, 2, 4)
        .reshape(-1)
    )
    return idx


class QwenNativeVisualTokenExtractor(nn.Module):
    """从 Qwen3-VL 原生视觉分支中提取 merger 前 token。"""

    def __init__(self, visual: nn.Module) -> None:
        super().__init__()
        self.patch_embed = visual.patch_embed
        self.pos_embed = visual.pos_embed
        self.num_grid_per_side = int(visual.num_grid_per_side)
        self.rotary_pos_emb = visual.rotary_pos_emb
        self.blocks = visual.blocks
        self.deepstack_visual_indexes = tuple(int(x) for x in getattr(visual, "deepstack_visual_indexes", ()))
        self.spatial_merge_size = int(getattr(visual, "spatial_merge_size", 2))
        self.patch_size = int(getattr(visual, "patch_size", 16))
        self.gradient_checkpointing = False

    def train(self, mode: bool = True):
        super().train(mode)
        self.patch_embed.eval()
        self.pos_embed.eval()
        self.rotary_pos_emb.eval()
        self.blocks.eval()
        return self

    def rot_pos_emb(self, grid_thw: torch.Tensor) -> torch.Tensor:
        merge_size = self.spatial_merge_size

        max_hw = int(grid_thw[:, 1:].max().item())
        freq_table = self.rotary_pos_emb(max_hw)
        device = freq_table.device

        total_tokens = int(torch.prod(grid_thw, dim=1).sum().item())
        pos_ids = torch.empty((total_tokens, 2), dtype=torch.long, device=device)

        offset = 0
        for num_frames, height, width in grid_thw:
            merged_h, merged_w = int(height.item()) // merge_size, int(width.item()) // merge_size

            block_rows = torch.arange(merged_h, device=device)
            block_cols = torch.arange(merged_w, device=device)
            intra_row = torch.arange(merge_size, device=device)
            intra_col = torch.arange(merge_size, device=device)

            row_idx = block_rows[:, None, None, None] * merge_size + intra_row[None, None, :, None]
            col_idx = block_cols[None, :, None, None] * merge_size + intra_col[None, None, None, :]

            row_idx = row_idx.expand(merged_h, merged_w, merge_size, merge_size).reshape(-1)
            col_idx = col_idx.expand(merged_h, merged_w, merge_size, merge_size).reshape(-1)

            coords = torch.stack((row_idx, col_idx), dim=-1)
            if int(num_frames.item()) > 1:
                coords = coords.repeat(int(num_frames.item()), 1)

            num_tokens = int(coords.shape[0])
            pos_ids[offset : offset + num_tokens] = coords
            offset += num_tokens

        embeddings = freq_table[pos_ids]
        return embeddings.flatten(1)

    def fast_pos_embed_interpolate(self, grid_thw: torch.Tensor) -> torch.Tensor:
        grid_ts, grid_hs, grid_ws = grid_thw[:, 0], grid_thw[:, 1], grid_thw[:, 2]

        idx_list = [[] for _ in range(4)]
        weight_list = [[] for _ in range(4)]

        for t, h, w in zip(grid_ts, grid_hs, grid_ws):
            h_idxs = torch.linspace(0, self.num_grid_per_side - 1, int(h.item()), device=self.pos_embed.weight.device)
            w_idxs = torch.linspace(0, self.num_grid_per_side - 1, int(w.item()), device=self.pos_embed.weight.device)

            h_idxs_floor = h_idxs.int()
            w_idxs_floor = w_idxs.int()
            h_idxs_ceil = (h_idxs.int() + 1).clip(max=self.num_grid_per_side - 1)
            w_idxs_ceil = (w_idxs.int() + 1).clip(max=self.num_grid_per_side - 1)

            dh = h_idxs - h_idxs_floor
            dw = w_idxs - w_idxs_floor

            base_h = h_idxs_floor * self.num_grid_per_side
            base_h_ceil = h_idxs_ceil * self.num_grid_per_side

            indices = [
                (base_h[None].T + w_idxs_floor[None]).flatten(),
                (base_h[None].T + w_idxs_ceil[None]).flatten(),
                (base_h_ceil[None].T + w_idxs_floor[None]).flatten(),
                (base_h_ceil[None].T + w_idxs_ceil[None]).flatten(),
            ]

            weights = [
                ((1 - dh)[None].T * (1 - dw)[None]).flatten(),
                ((1 - dh)[None].T * dw[None]).flatten(),
                (dh[None].T * (1 - dw)[None]).flatten(),
                (dh[None].T * dw[None]).flatten(),
            ]

            for i in range(4):
                idx_list[i].extend(indices[i].tolist())
                weight_list[i].extend(weights[i].tolist())

        idx_tensor = torch.tensor(idx_list, dtype=torch.long, device=self.pos_embed.weight.device)
        weight_tensor = torch.tensor(weight_list, dtype=self.pos_embed.weight.dtype, device=self.pos_embed.weight.device)
        pos_embeds = self.pos_embed(idx_tensor) * weight_tensor[:, :, None]
        patch_pos_embeds = pos_embeds[0] + pos_embeds[1] + pos_embeds[2] + pos_embeds[3]

        patch_pos_embeds = patch_pos_embeds.split([int(h.item()) * int(w.item()) for h, w in zip(grid_hs, grid_ws)])

        patch_pos_embeds_permute = []
        merge_size = self.spatial_merge_size
        for pos_embed, t, h, w in zip(patch_pos_embeds, grid_ts, grid_hs, grid_ws):
            pos_embed = pos_embed.repeat(int(t.item()), 1)
            pos_embed = (
                pos_embed.view(
                    int(t.item()),
                    int(h.item()) // merge_size,
                    merge_size,
                    int(w.item()) // merge_size,
                    merge_size,
                    -1,
                )
                .permute(0, 1, 3, 2, 4, 5)
                .flatten(0, 4)
            )
            patch_pos_embeds_permute.append(pos_embed)
        return torch.cat(patch_pos_embeds_permute, dim=0)

    def forward_tokens(self, pixel_values: torch.Tensor, grid_thw: torch.Tensor, **kwargs: Any) -> QwenVisualTokenOutput:
        if pixel_values.ndim != 2:
            raise ValueError(
                "QwenNativeVisualTokenExtractor expects pixel_values shape [num_patches, patch_dim], "
                f"got {tuple(pixel_values.shape)}"
            )
        if grid_thw.ndim != 2 or int(grid_thw.shape[-1]) != 3:
            raise ValueError(f"grid_thw must have shape [num_images, 3], got {tuple(grid_thw.shape)}")

        grid_thw = grid_thw.to(device=pixel_values.device, dtype=torch.long)

        hidden_states = self.patch_embed(pixel_values)
        pos_embeds = self.fast_pos_embed_interpolate(grid_thw).to(dtype=hidden_states.dtype, device=hidden_states.device)
        hidden_states = hidden_states + pos_embeds

        rotary_pos_emb = self.rot_pos_emb(grid_thw).to(dtype=hidden_states.dtype, device=hidden_states.device)

        seq_len, _ = hidden_states.size()
        hidden_states = hidden_states.reshape(seq_len, -1)
        rotary_pos_emb = rotary_pos_emb.reshape(seq_len, -1)
        emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
        position_embeddings = (emb.cos(), emb.sin())

        cu_seqlens = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]).cumsum(
            dim=0,
            dtype=grid_thw.dtype if torch.jit.is_tracing() else torch.int32,
        )
        cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)

        deepstack_feature_lists: list[torch.Tensor] = []
        for layer_num, blk in enumerate(self.blocks):
            hidden_states = blk(
                hidden_states,
                cu_seqlens=cu_seqlens,
                position_embeddings=position_embeddings,
                **kwargs,
            )
            if int(layer_num) in self.deepstack_visual_indexes:
                deepstack_feature_lists.append(hidden_states)

        return QwenVisualTokenOutput(
            patch_tokens=hidden_states,
            deepstack_patch_tokens=deepstack_feature_lists,
        )
