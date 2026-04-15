from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn


def build_postshuffle_index(*, grid_h: int, grid_w: int, merge_size: int) -> torch.LongTensor:
    """构造 postshuffle 的 patch 索引序列。

    目标顺序：以 merge_size x merge_size 为一个块，先按块的行列顺序遍历，再遍历块内的行列。
    这样可以保证连续的 merge_size^2 个 patch token 对应同一个空间块，便于后续 patch merger 做 2x2 merge。
    """
    if grid_h % merge_size != 0 or grid_w % merge_size != 0:
        raise ValueError(f"grid_h/grid_w must be divisible by merge_size. Got {grid_h}x{grid_w}, merge={merge_size}")

    merged_h = grid_h // merge_size
    merged_w = grid_w // merge_size

    idx: list[int] = []
    for br in range(merged_h):
        for bc in range(merged_w):
            for ir in range(merge_size):
                for ic in range(merge_size):
                    r = br * merge_size + ir
                    c = bc * merge_size + ic
                    idx.append(r * grid_w + c)
    return torch.tensor(idx, dtype=torch.long)


@dataclass(frozen=True)
class DinoV3AdapterConfig:
    dinov3_dir: Path
    image_size: int = 512
    merge_size: int = 2
    deepstack_visual_indexes: tuple[int, ...] = (5, 11, 17)
    qwen_vision_depth: int | None = None


class DinoV3VisualAdapter(nn.Module):
    """用 DINOv3 产出的 patch token 替换 Qwen3-VL 原视觉编码器输出。

    forward 接口对齐 Qwen3VLModel.visual：输入 pixel_values + grid_thw，输出 (image_embeds, deepstack_embeds)。
    - image_embeds: (sum_tokens, out_hidden_size)
    - deepstack_embeds: list[tensor]，每个形状同 image_embeds
    """

    def __init__(
        self,
        cfg: DinoV3AdapterConfig,
        *,
        merger: nn.Module,
        deepstack_merger_list: nn.ModuleList | None,
        torch_dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.spatial_merge_size = int(cfg.merge_size)

        try:
            from transformers import DINOv3ViTModel
        except Exception as e:  # noqa: BLE001
            raise RuntimeError("Missing dependency: transformers (DINOv3ViTModel)") from e

        dino_kwargs = {}
        if torch_dtype is not None:
            dino_kwargs["torch_dtype"] = torch_dtype

        self.dino = DINOv3ViTModel.from_pretrained(str(Path(cfg.dinov3_dir)), **dino_kwargs)
        self.dino.eval()

        for p in self.dino.parameters():
            p.requires_grad = False

        self.merger = merger
        self.deepstack_merger_list = deepstack_merger_list

        # 从 DINOv3 config 读取 register token 数与 patch_size
        self.num_register_tokens = int(getattr(self.dino.config, "num_register_tokens", 0))
        self.patch_size = int(getattr(self.dino.config, "patch_size", 16))

        # DINOv3 的 patch token hidden_size 可能与 Qwen3-VL 的视觉 hidden_size 不一致。
        # - 例如：DINOv3 ViT-L/16 是 1024，而 Qwen3-VL-8B 的视觉 hidden_size 是 1152。
        # Qwen3-VL 的 merger/deepstack_merger_list 期望输入 token 的最后一维 == 视觉 hidden_size，
        # 否则内部的 reshape/view 会直接报错。
        dino_h = int(getattr(self.dino.config, "hidden_size", 0))
        norm_shape = getattr(getattr(self.merger, "norm", None), "normalized_shape", None)
        if isinstance(norm_shape, (tuple, list)) and len(norm_shape) == 1:
            qwen_h = int(norm_shape[0])
        else:
            raise RuntimeError(f"Failed to infer Qwen3-VL vision hidden_size from merger.norm.normalized_shape={norm_shape!r}")

        self.dino_hidden_size = int(dino_h)
        self.qwen_hidden_size = int(qwen_h)
        self.input_proj: nn.Linear | None = None
        if int(dino_h) != int(qwen_h):
            # 初始化为“尽量保持原维度不变”：前 min(dino_h,qwen_h) 维为单位阵，其余补 0。
            self.input_proj = nn.Linear(int(dino_h), int(qwen_h), bias=False, dtype=torch_dtype)
            with torch.no_grad():
                self.input_proj.weight.zero_()
                n = min(int(dino_h), int(qwen_h))
                eye = torch.eye(n, dtype=self.input_proj.weight.dtype)
                self.input_proj.weight[:n, :n].copy_(eye)

        self._cached_index: dict[tuple[int, int], torch.LongTensor] = {}
        # deepstack 层索引对齐表（Qwen3-VL 的 index 可能超过 DINOv3 的层数，例如 8B 配置里有 24，
        # 但 DINOv3 ViT-L 的层索引范围是 0~23）。这里把“原始 index”映射到“实际挂 hook 的层 index”，
        # 以保证 forward 不会因为越界直接报错。
        self._deepstack_index_map: dict[int, int] = {}
        num_layers = len(getattr(self.dino, "layer", []))
        raw_indexes = [int(raw) for raw in self.cfg.deepstack_visual_indexes]
        for raw_i in raw_indexes:
            if raw_i < 0:
                # 目前不支持负数索引（避免语义不明确）。如未来需要，可改成 Python 风格负索引映射。
                raise ValueError(f"Invalid deepstack layer index (negative) for DINOv3: {raw_i}")
        if num_layers <= 0:
            raise RuntimeError("Unexpected DINOv3 model: missing encoder layers.")

        # 当 Qwen 视觉层索引超出 DINO 层数时，采用“相对深度映射”对齐三层 deepstack。
        # 例如 8B 的 [8,16,24]（总深度 27）会映射到 DINO-24 层上的 [7,14,21]。
        need_realign = any(raw_i >= num_layers for raw_i in raw_indexes)
        src_depth = int(self.cfg.qwen_vision_depth) if self.cfg.qwen_vision_depth is not None else 0
        if need_realign and src_depth > 1:
            src_last = int(src_depth - 1)
            dst_last = int(num_layers - 1)
            for raw_i in raw_indexes:
                mapped = int(round(float(raw_i) * float(dst_last) / float(src_last)))
                mapped = max(0, min(dst_last, mapped))
                self._deepstack_index_map[raw_i] = int(mapped)
        else:
            for raw_i in raw_indexes:
                mapped = raw_i if raw_i < num_layers else (num_layers - 1)
                self._deepstack_index_map[raw_i] = int(mapped)

    def train(self, mode: bool = True):
        # 训练时我们只更新 merger / deepstack / input_proj，DINOv3 主干始终冻结并且 forward 在 no_grad 下执行。
        # 如果让 DINOv3 跟随整体模型切到 train()，dropout 等随机层可能会引入不必要的噪声，导致训练不稳定。
        super().train(mode)
        self.dino.eval()
        return self

    @property
    def dtype(self) -> torch.dtype:
        # Qwen3VLModel.get_image_features 会用到 visual.dtype
        return next(self.parameters()).dtype

    def _get_postshuffle_index(self, *, grid_h: int, grid_w: int, device: torch.device) -> torch.LongTensor:
        key = (int(grid_h), int(grid_w))
        if key not in self._cached_index:
            self._cached_index[key] = build_postshuffle_index(grid_h=grid_h, grid_w=grid_w, merge_size=self.spatial_merge_size)
        return self._cached_index[key].to(device=device)

    def forward(self, pixel_values: torch.Tensor, grid_thw: torch.Tensor, **kwargs):
        if pixel_values.ndim != 4:
            raise ValueError(f"DinoV3VisualAdapter expects pixel_values (B,C,H,W). Got shape={tuple(pixel_values.shape)}")
        if grid_thw is None:
            raise ValueError("grid_thw is required.")
        if grid_thw.ndim != 2 or grid_thw.shape[-1] != 3:
            raise ValueError(f"grid_thw must have shape (num_images,3). Got {tuple(grid_thw.shape)}")

        # 目前只支持图像（t=1），并且 batch 内统一网格大小（便于张量化）。
        grid_thw = grid_thw.to(device=pixel_values.device)
        t = grid_thw[:, 0]
        if not torch.all(t == 1):
            raise ValueError(f"Only t=1 is supported for now. Got t={t.tolist()}")

        grid_h = int(grid_thw[0, 1].item())
        grid_w = int(grid_thw[0, 2].item())
        if not torch.all(grid_thw[:, 1] == grid_h) or not torch.all(grid_thw[:, 2] == grid_w):
            raise ValueError("Mixed grid sizes within a batch are not supported.")

        expected_patches = grid_h * grid_w
        patch_start = 1 + self.num_register_tokens

        need_deepstack = self.deepstack_merger_list is not None and len(self.deepstack_merger_list) > 0

        # DINOv3ViTModel 在当前 transformers 版本里不会把 hidden_states 放到输出里。
        # 这里用 forward hook 抓取指定层的输出，以对齐 Qwen3-VL 的 deepstack 逻辑。
        captured: dict[int, torch.Tensor] = {}
        handles: list[Any] = []
        try:
            if need_deepstack:
                for layer_idx in self.cfg.deepstack_visual_indexes:
                    raw_idx = int(layer_idx)
                    idx = int(self._deepstack_index_map.get(raw_idx, raw_idx))

                    def _make_hook(i: int):
                        def _hook(_module, _inputs, output):
                            if isinstance(output, torch.Tensor):
                                captured[i] = output

                        return _hook

                    # hook 挂在 idx（映射后的层）上，但 captured 用 raw_idx 作为 key，
                    # 保持和 deepstack_visual_indexes 的遍历一致。
                    handles.append(self.dino.layer[idx].register_forward_hook(_make_hook(raw_idx)))

            with torch.no_grad():
                outputs = self.dino(pixel_values=pixel_values, return_dict=True)
        finally:
            for h in handles:
                try:
                    h.remove()
                except Exception:
                    pass

        last = outputs.last_hidden_state  # (B, 1 + R + N, 1024)
        if last.shape[1] < patch_start + expected_patches:
            raise ValueError(
                f"Unexpected DINOv3 token length. Got {last.shape[1]}, need >= {patch_start + expected_patches}."
            )

        patches = last[:, patch_start : patch_start + expected_patches, :]
        idx = self._get_postshuffle_index(grid_h=grid_h, grid_w=grid_w, device=patches.device)
        patches = patches.index_select(dim=1, index=idx)
        patches = patches.reshape(-1, patches.shape[-1])  # (B*N, 1024)
        if self.input_proj is not None:
            patches = self.input_proj(patches)

        deepstack_features = []
        if need_deepstack:
            # Qwen3-VL 里 deepstack_merger_list 与 deepstack_visual_indexes 一一对应。
            # 这里只取 hook 抓到的那些层，按 indexes 的顺序输出 list。
            for i, layer_idx in enumerate(self.cfg.deepstack_visual_indexes):
                if self.deepstack_merger_list is None or i >= len(self.deepstack_merger_list):
                    break
                hs = captured.get(int(layer_idx))
                if hs is None:
                    continue
                hs_patches = hs[:, patch_start : patch_start + expected_patches, :]
                hs_patches = hs_patches.index_select(dim=1, index=idx).reshape(-1, hs_patches.shape[-1])
                if self.input_proj is not None:
                    hs_patches = self.input_proj(hs_patches)
                deepstack_features.append(self.deepstack_merger_list[i](hs_patches))

        image_embeds = self.merger(patches)
        return image_embeds, deepstack_features
