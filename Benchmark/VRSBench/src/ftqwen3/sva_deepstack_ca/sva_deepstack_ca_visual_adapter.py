from __future__ import annotations

import json
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file, save_file

from ..dinov3_merger.dinov3_adapter import build_postshuffle_index
from ..shared.qwen_native_visual_tokens import QwenNativeVisualTokenExtractor, build_qwen_postshuffle_index


DEFAULT_DEEPSTACK_VISUAL_INDEXES: tuple[int, ...] = (5, 11, 17)


def _load_preprocessor_stats(model_dir: Path) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    cfg_path = Path(model_dir) / "preprocessor_config.json"
    if not cfg_path.is_file():
        raise FileNotFoundError(f"Missing preprocessor config: {cfg_path}")
    payload = json.loads(cfg_path.read_text(encoding="utf-8"))

    image_mean = tuple(float(x) for x in payload.get("image_mean", [0.5, 0.5, 0.5]))
    image_std = tuple(float(x) for x in payload.get("image_std", [0.5, 0.5, 0.5]))
    if len(image_mean) != 3 or len(image_std) != 3:
        raise ValueError(f"Invalid image_mean/std in {cfg_path}")
    return image_mean, image_std


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _resolve_meta_path(path: Path | None) -> Path | None:
    if path is None:
        return None
    return Path(path).with_suffix(".json")


def read_sva_deepstack_ca_meta(path: Path) -> dict[str, Any]:
    meta_path = _resolve_meta_path(path)
    if meta_path is None or not meta_path.is_file():
        return {}
    return json.loads(meta_path.read_text(encoding="utf-8"))


def read_sva_deepstack_ca_run_meta(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    payload = read_sva_deepstack_ca_meta(Path(path))
    run_meta = payload.get("run", {})
    return run_meta if isinstance(run_meta, dict) else {}


def assert_sva_deepstack_ca_runtime_matches_merger(
    *,
    qwen_model_dir: Path,
    dinov3_dir: Path,
    smart_resize_min_pixels: int,
    smart_resize_max_pixels: int,
    merger_ckpt: Path,
    visual: nn.Module | None = None,
) -> dict[str, Any]:
    payload = read_sva_deepstack_ca_meta(Path(merger_ckpt))
    run_meta = payload.get("run", {})
    visual_meta = payload.get("visual", {})
    if not isinstance(run_meta, dict):
        run_meta = {}
    if not isinstance(visual_meta, dict):
        visual_meta = {}

    if not run_meta and not visual_meta:
        return {}

    def _same_path(expected: Path, raw: Any) -> bool:
        try:
            candidate = Path(str(raw))
        except Exception:
            return False
        if not candidate.is_absolute():
            candidate = (_project_root() / candidate).resolve()
        else:
            candidate = candidate.resolve()
        return candidate == expected.resolve()

    checks = [
        ("qwen_model_dir", _same_path(Path(qwen_model_dir), run_meta.get("qwen_model_dir", ""))),
        ("dinov3_dir", _same_path(Path(dinov3_dir), run_meta.get("dinov3_dir", ""))),
        ("smart_resize_min_pixels", int(run_meta.get("smart_resize_min_pixels", smart_resize_min_pixels)) == int(smart_resize_min_pixels)),
        ("smart_resize_max_pixels", int(run_meta.get("smart_resize_max_pixels", smart_resize_max_pixels)) == int(smart_resize_max_pixels)),
    ]

    if visual is not None and visual_meta:
        runtime_merge_size = int(getattr(visual, "spatial_merge_size", 0))
        if "merge_size" in visual_meta:
            checks.append(("merge_size", int(visual_meta.get("merge_size", runtime_merge_size)) == int(runtime_merge_size)))

        runtime_indexes = tuple(int(x) for x in getattr(visual, "deepstack_visual_indexes", ()))
        if "deepstack_visual_indexes" in visual_meta:
            expected_indexes = tuple(int(x) for x in visual_meta.get("deepstack_visual_indexes", []))
            checks.append(("deepstack_visual_indexes", expected_indexes == runtime_indexes))

        runtime_cfg = getattr(visual, "cfg", None)
        if runtime_cfg is not None:
            if "query_base_side" in visual_meta:
                checks.append(
                    (
                        "query_base_side",
                        int(visual_meta.get("query_base_side", getattr(runtime_cfg, "query_base_side", 0)))
                        == int(getattr(runtime_cfg, "query_base_side", 0)),
                    )
                )
            if "latent_grid_h" in visual_meta:
                checks.append(
                    (
                        "latent_grid_h",
                        int(visual_meta.get("latent_grid_h", getattr(runtime_cfg, "latent_grid_h", 0)))
                        == int(getattr(runtime_cfg, "latent_grid_h", 0)),
                    )
                )
            if "latent_grid_w" in visual_meta:
                checks.append(
                    (
                        "latent_grid_w",
                        int(visual_meta.get("latent_grid_w", getattr(runtime_cfg, "latent_grid_w", 0)))
                        == int(getattr(runtime_cfg, "latent_grid_w", 0)),
                    )
                )
            if "sva_num_heads" in visual_meta:
                checks.append(
                    ("sva_num_heads", int(visual_meta.get("sva_num_heads", getattr(runtime_cfg, "sva_num_heads", 0))) == int(getattr(runtime_cfg, "sva_num_heads", 0)))
                )
            if "sva_mlp_ratio" in visual_meta:
                checks.append(
                    (
                        "sva_mlp_ratio",
                        float(visual_meta.get("sva_mlp_ratio", getattr(runtime_cfg, "sva_mlp_ratio", 0.0)))
                        == float(getattr(runtime_cfg, "sva_mlp_ratio", 0.0)),
                    )
                )
            if "sva_dropout" in visual_meta:
                checks.append(
                    (
                        "sva_dropout",
                        float(visual_meta.get("sva_dropout", getattr(runtime_cfg, "sva_dropout", 0.0)))
                        == float(getattr(runtime_cfg, "sva_dropout", 0.0)),
                    )
                )

    bad = [name for name, ok in checks if not ok]
    if bad:
        raise ValueError(
            "SVA deepstack CA runtime config mismatch with checkpoint "
            f"{merger_ckpt}: {', '.join(bad)}"
        )
    return run_meta


def load_sva_deepstack_ca_merger_safetensors(model, path: Path) -> None:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Missing SVA deepstack CA merger checkpoint: {path}")
    state = load_file(str(path))
    visual = model.model.visual

    merger_state = {k[len("merger.") :]: v for k, v in state.items() if k.startswith("merger.")}
    deepstack_state = {
        k[len("deepstack_merger_list.") :]: v for k, v in state.items() if k.startswith("deepstack_merger_list.")
    }
    sva_main_state = {k[len("sva_main.") :]: v for k, v in state.items() if k.startswith("sva_main.")}
    sva_deepstack_state = {
        k[len("sva_deepstack_list.") :]: v for k, v in state.items() if k.startswith("sva_deepstack_list.")
    }
    input_proj_state = {k[len("input_proj_dino.") :]: v for k, v in state.items() if k.startswith("input_proj_dino.")}

    visual.merger.load_state_dict(merger_state, strict=True)
    if visual.deepstack_merger_list is not None:
        visual.deepstack_merger_list.load_state_dict(deepstack_state, strict=True)
    if getattr(visual, "sva_main", None) is not None:
        visual.sva_main.load_state_dict(sva_main_state, strict=True)
    if getattr(visual, "sva_deepstack_list", None) is not None:
        visual.sva_deepstack_list.load_state_dict(sva_deepstack_state, strict=True)
    if getattr(visual, "input_proj_dino", None) is not None:
        visual.input_proj_dino.load_state_dict(input_proj_state, strict=True)


def save_sva_deepstack_ca_merger_safetensors(model, path: Path, *, extra: dict[str, Any] | None = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    visual = model.model.visual

    state: dict[str, torch.Tensor] = {}
    for k, v in visual.merger.state_dict().items():
        state[f"merger.{k}"] = v.detach().cpu()
    if visual.deepstack_merger_list is not None:
        for k, v in visual.deepstack_merger_list.state_dict().items():
            state[f"deepstack_merger_list.{k}"] = v.detach().cpu()
    if getattr(visual, "sva_main", None) is not None:
        for k, v in visual.sva_main.state_dict().items():
            state[f"sva_main.{k}"] = v.detach().cpu()
    if getattr(visual, "sva_deepstack_list", None) is not None:
        for k, v in visual.sva_deepstack_list.state_dict().items():
            state[f"sva_deepstack_list.{k}"] = v.detach().cpu()
    if getattr(visual, "input_proj_dino", None) is not None:
        for k, v in visual.input_proj_dino.state_dict().items():
            state[f"input_proj_dino.{k}"] = v.detach().cpu()

    save_file(state, str(path))
    if extra is not None:
        meta_path = path.with_suffix(".json")
        meta_path.write_text(json.dumps(extra, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def _infer_qwen_hidden_size(qwen_visual: nn.Module) -> int:
    qwen_hidden = int(getattr(getattr(qwen_visual, "config", None), "hidden_size", 0))
    if qwen_hidden > 0:
        return qwen_hidden
    norm_shape = getattr(getattr(qwen_visual.merger, "norm", None), "normalized_shape", None)
    if isinstance(norm_shape, (tuple, list)) and len(norm_shape) == 1:
        return int(norm_shape[0])
    raise RuntimeError("Failed to infer Qwen visual hidden size.")


def _infer_merger_out_hidden_size(merger: nn.Module) -> int:
    linear_fc2 = getattr(merger, "linear_fc2", None)
    weight = getattr(linear_fc2, "weight", None)
    if isinstance(weight, torch.Tensor) and weight.ndim == 2:
        return int(weight.shape[0])
    raise RuntimeError("Failed to infer merger output hidden size.")


class FixedLatentCrossAttentionResampler(nn.Module):
    """把真实 patch-grid 融合成固定 latent patch-grid。"""

    def __init__(
        self,
        *,
        hidden_size: int,
        latent_grid_h: int,
        latent_grid_w: int,
        query_base_side: int,
        num_heads: int,
        mlp_ratio: float,
        dropout: float,
    ) -> None:
        super().__init__()
        if int(hidden_size) <= 0:
            raise ValueError(f"Invalid hidden_size={hidden_size}")
        if int(latent_grid_h) <= 0 or int(latent_grid_w) <= 0:
            raise ValueError(f"Invalid latent grid={latent_grid_h}x{latent_grid_w}")
        if int(query_base_side) <= 0:
            raise ValueError(f"Invalid query_base_side={query_base_side}")
        if int(hidden_size) % int(num_heads) != 0:
            raise ValueError(f"hidden_size={hidden_size} must be divisible by num_heads={num_heads}")

        self.hidden_size = int(hidden_size)
        self.latent_grid_h = int(latent_grid_h)
        self.latent_grid_w = int(latent_grid_w)
        self.query_base_side = int(query_base_side)

        self.query_grid = nn.Parameter(
            torch.zeros(1, int(hidden_size), int(latent_grid_h), int(latent_grid_w))
        )
        self.branch_embed = nn.Parameter(torch.zeros(2, int(hidden_size)))
        self.local_pos_grid = nn.Parameter(
            torch.zeros(1, int(hidden_size), int(query_base_side), int(query_base_side))
        )

        self.query_norm = nn.LayerNorm(int(hidden_size), eps=1e-6)
        self.context_norm = nn.LayerNorm(int(hidden_size), eps=1e-6)
        self.post_norm = nn.LayerNorm(int(hidden_size), eps=1e-6)

        self.attn = nn.MultiheadAttention(
            embed_dim=int(hidden_size),
            num_heads=int(num_heads),
            dropout=float(dropout),
            batch_first=True,
        )

        mlp_hidden = int(round(float(hidden_size) * float(mlp_ratio)))
        self.mlp = nn.Sequential(
            nn.Linear(int(hidden_size), mlp_hidden),
            nn.GELU(),
            nn.Linear(mlp_hidden, int(hidden_size)),
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.query_grid, mean=0.0, std=0.02)
        nn.init.normal_(self.branch_embed, mean=0.0, std=0.02)
        nn.init.normal_(self.local_pos_grid, mean=0.0, std=0.02)

        nn.init.zeros_(self.attn.out_proj.weight)
        if self.attn.out_proj.bias is not None:
            nn.init.zeros_(self.attn.out_proj.bias)

        last_linear = self.mlp[-1]
        nn.init.zeros_(last_linear.weight)
        if last_linear.bias is not None:
            nn.init.zeros_(last_linear.bias)

    def _build_query_tokens(
        self,
        *,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        grid = self.query_grid.to(device=device, dtype=dtype)
        return grid.squeeze(0).permute(1, 2, 0).reshape(
            int(self.latent_grid_h) * int(self.latent_grid_w),
            1,
            int(self.hidden_size),
        )

    def _build_local_position_tokens(
        self,
        *,
        cell_h: int,
        cell_w: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        grid = F.interpolate(
            self.local_pos_grid.float(),
            size=(int(cell_h), int(cell_w)),
            mode="bilinear",
            align_corners=False,
        )
        grid = grid.to(device=device, dtype=dtype)
        return grid.squeeze(0).permute(1, 2, 0).reshape(1, int(cell_h) * int(cell_w), int(self.hidden_size))

    def forward(
        self,
        qwen_tokens: torch.Tensor,
        dino_tokens: torch.Tensor,
        *,
        actual_grid_h: int,
        actual_grid_w: int,
        qwen_row_major_tokens: torch.Tensor | None = None,
        dino_row_major_tokens: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if qwen_tokens.ndim != 2 or dino_tokens.ndim != 2:
            raise ValueError(
                "FixedLatentCrossAttentionResampler expects [N, D] tokens, "
                f"got qwen={tuple(qwen_tokens.shape)} dino={tuple(dino_tokens.shape)}"
            )
        if qwen_tokens.shape != dino_tokens.shape:
            raise ValueError(
                f"Qwen/DINO token shape mismatch: qwen={tuple(qwen_tokens.shape)} dino={tuple(dino_tokens.shape)}"
            )
        if int(qwen_tokens.shape[-1]) != int(self.hidden_size):
            raise ValueError(
                f"Hidden size mismatch: expect={self.hidden_size}, got={int(qwen_tokens.shape[-1])}"
            )
        if int(actual_grid_h) % int(self.latent_grid_h) != 0 or int(actual_grid_w) % int(self.latent_grid_w) != 0:
            raise ValueError(
                "Actual patch grid must be divisible by fixed latent grid, "
                f"got actual={actual_grid_h}x{actual_grid_w}, latent={self.latent_grid_h}x{self.latent_grid_w}"
            )

        expected_tokens = int(actual_grid_h) * int(actual_grid_w)
        if int(qwen_tokens.shape[0]) != expected_tokens:
            raise ValueError(
                f"Token count mismatch for grid {actual_grid_h}x{actual_grid_w}: "
                f"expect={expected_tokens}, got={int(qwen_tokens.shape[0])}"
            )

        if qwen_row_major_tokens is None:
            qwen_row_major_tokens = qwen_tokens
        if dino_row_major_tokens is None:
            dino_row_major_tokens = dino_tokens

        cell_h = int(actual_grid_h // self.latent_grid_h)
        cell_w = int(actual_grid_w // self.latent_grid_w)
        cell_tokens = int(cell_h * cell_w)

        q = qwen_row_major_tokens.view(
            int(self.latent_grid_h),
            int(cell_h),
            int(self.latent_grid_w),
            int(cell_w),
            int(self.hidden_size),
        ).permute(0, 2, 1, 3, 4).reshape(int(self.latent_grid_h) * int(self.latent_grid_w), cell_tokens, int(self.hidden_size))
        d = dino_row_major_tokens.view(
            int(self.latent_grid_h),
            int(cell_h),
            int(self.latent_grid_w),
            int(cell_w),
            int(self.hidden_size),
        ).permute(0, 2, 1, 3, 4).reshape(int(self.latent_grid_h) * int(self.latent_grid_w), cell_tokens, int(self.hidden_size))

        local_pos = self._build_local_position_tokens(
            cell_h=int(cell_h),
            cell_w=int(cell_w),
            dtype=q.dtype,
            device=q.device,
        )
        q_branch = self.branch_embed[0].view(1, 1, self.hidden_size).to(device=q.device, dtype=q.dtype)
        d_branch = self.branch_embed[1].view(1, 1, self.hidden_size).to(device=q.device, dtype=q.dtype)
        context = torch.cat([q + local_pos + q_branch, d + local_pos + d_branch], dim=1)

        query = self._build_query_tokens(
            dtype=q.dtype,
            device=q.device,
        )
        attn_out, _ = self.attn(
            self.query_norm(query),
            self.context_norm(context),
            self.context_norm(context),
            need_weights=False,
        )
        sampled = query + attn_out
        sampled = sampled + self.mlp(self.post_norm(sampled))
        return sampled.squeeze(1)


@dataclass(frozen=True)
class SVADeepstackCAVisualAdapterConfig:
    dinov3_dir: Path
    merge_size: int = 2
    latent_grid_h: int = 16
    latent_grid_w: int = 16
    deepstack_visual_indexes: tuple[int, ...] = DEFAULT_DEEPSTACK_VISUAL_INDEXES
    qwen_vision_depth: int | None = None
    qwen_image_mean: tuple[float, float, float] = (0.5, 0.5, 0.5)
    qwen_image_std: tuple[float, float, float] = (0.5, 0.5, 0.5)
    dino_image_mean: tuple[float, float, float] = (0.43, 0.411, 0.296)
    dino_image_std: tuple[float, float, float] = (0.213, 0.156, 0.143)
    query_base_side: int = 4
    sva_num_heads: int = 8
    sva_mlp_ratio: float = 4.0
    sva_dropout: float = 0.0


class SVADeepstackCAVisualAdapter(nn.Module):
    """先把 Qwen + DINO patch token 融合成固定 latent grid，再走原生 merger。"""

    def __init__(
        self,
        cfg: SVADeepstackCAVisualAdapterConfig,
        *,
        qwen_visual: nn.Module,
        torch_dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.spatial_merge_size = int(cfg.merge_size)
        self.latent_grid_h = int(cfg.latent_grid_h)
        self.latent_grid_w = int(cfg.latent_grid_w)
        self.patch_size = int(getattr(qwen_visual, "patch_size", 16))
        self.temporal_patch_size = int(getattr(getattr(qwen_visual, "config", None), "temporal_patch_size", 2))
        self.in_channels = int(getattr(getattr(qwen_visual, "config", None), "in_channels", 3))
        self.deepstack_visual_indexes = tuple(int(x) for x in cfg.deepstack_visual_indexes)
        self.qwen_token_extractor = QwenNativeVisualTokenExtractor(qwen_visual)
        self.qwen_token_extractor.deepstack_visual_indexes = self.deepstack_visual_indexes
        self.merger = qwen_visual.merger
        self.deepstack_merger_list = getattr(qwen_visual, "deepstack_merger_list", None)

        self.register_buffer(
            "qwen_image_mean",
            torch.tensor(list(cfg.qwen_image_mean), dtype=torch.float32).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "qwen_image_std",
            torch.tensor(list(cfg.qwen_image_std), dtype=torch.float32).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "dino_image_mean",
            torch.tensor(list(cfg.dino_image_mean), dtype=torch.float32).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "dino_image_std",
            torch.tensor(list(cfg.dino_image_std), dtype=torch.float32).view(1, 3, 1, 1),
            persistent=False,
        )

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
        self._dino_dtype = self._infer_module_dtype(self.dino, fallback=torch_dtype or torch.float32)

        self.num_register_tokens = int(getattr(self.dino.config, "num_register_tokens", 0))
        self.dino_patch_size = int(getattr(self.dino.config, "patch_size", 16))
        if int(self.dino_patch_size) != int(self.patch_size):
            raise ValueError(
                "Current SVA deepstack CA adapter requires identical patch_size between Qwen and DINO, "
                f"got qwen={self.patch_size} dino={self.dino_patch_size}"
            )

        self.qwen_hidden_size = _infer_qwen_hidden_size(qwen_visual)
        self.dino_hidden_size = int(getattr(self.dino.config, "hidden_size", 0))

        self.input_proj_dino: nn.Linear | None = None
        if int(self.dino_hidden_size) != int(self.qwen_hidden_size):
            self.input_proj_dino = nn.Linear(
                int(self.dino_hidden_size),
                int(self.qwen_hidden_size),
                bias=False,
                dtype=torch_dtype,
            )
            with torch.no_grad():
                self.input_proj_dino.weight.zero_()
                n = min(int(self.dino_hidden_size), int(self.qwen_hidden_size))
                eye = torch.eye(n, dtype=self.input_proj_dino.weight.dtype)
                self.input_proj_dino.weight[:n, :n].copy_(eye)

        if self.latent_grid_h % self.spatial_merge_size != 0 or self.latent_grid_w % self.spatial_merge_size != 0:
            raise ValueError(
                "Fixed latent grid must be divisible by spatial_merge_size, "
                f"got latent={self.latent_grid_h}x{self.latent_grid_w}, merge={self.spatial_merge_size}"
            )

        self.sva_main = FixedLatentCrossAttentionResampler(
            hidden_size=int(self.qwen_hidden_size),
            latent_grid_h=int(self.latent_grid_h),
            latent_grid_w=int(self.latent_grid_w),
            query_base_side=int(cfg.query_base_side),
            num_heads=int(cfg.sva_num_heads),
            mlp_ratio=float(cfg.sva_mlp_ratio),
            dropout=float(cfg.sva_dropout),
        )

        deepstack_count = int(len(self.deepstack_visual_indexes))
        self.sva_deepstack_list = nn.ModuleList(
            [
                FixedLatentCrossAttentionResampler(
                    hidden_size=int(self.qwen_hidden_size),
                    latent_grid_h=int(self.latent_grid_h),
                    latent_grid_w=int(self.latent_grid_w),
                    query_base_side=int(cfg.query_base_side),
                    num_heads=int(cfg.sva_num_heads),
                    mlp_ratio=float(cfg.sva_mlp_ratio),
                    dropout=float(cfg.sva_dropout),
                )
                for _ in range(deepstack_count)
            ]
        )

        self._cached_index: dict[tuple[int, int], torch.LongTensor] = {}
        self._cached_index_inv: dict[tuple[int, int], torch.LongTensor] = {}
        self._validated_qwen_postshuffle_contract: set[tuple[int, int]] = set()
        self._deepstack_index_map: dict[int, int] = {}
        num_layers = len(self._get_dino_layers())
        raw_indexes = [int(raw) for raw in self.deepstack_visual_indexes]
        if num_layers <= 0:
            raise RuntimeError("Unexpected DINOv3 model: missing encoder layers.")

        need_realign = any(raw_i >= num_layers for raw_i in raw_indexes)
        src_depth = int(cfg.qwen_vision_depth) if cfg.qwen_vision_depth is not None else 0
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
        super().train(mode)
        self.qwen_token_extractor.train(False)
        self.dino.eval()
        return self

    @property
    def dtype(self) -> torch.dtype:
        return next(self.parameters()).dtype

    def _get_postshuffle_index(self, *, grid_h: int, grid_w: int, device: torch.device) -> torch.LongTensor:
        key = (int(grid_h), int(grid_w))
        if key not in self._cached_index:
            idx = build_postshuffle_index(
                grid_h=int(grid_h),
                grid_w=int(grid_w),
                merge_size=int(self.spatial_merge_size),
            )
            qwen_idx = build_qwen_postshuffle_index(
                grid_t=1,
                grid_h=int(grid_h),
                grid_w=int(grid_w),
                merge_size=int(self.spatial_merge_size),
            )
            if not torch.equal(idx, qwen_idx):
                raise RuntimeError(
                    "Qwen native visual token order is inconsistent with merger postshuffle order: "
                    f"grid={int(grid_h)}x{int(grid_w)}, merge={int(self.spatial_merge_size)}"
                )
            self._cached_index[key] = idx
            self._validated_qwen_postshuffle_contract.add(key)
        elif key not in self._validated_qwen_postshuffle_contract:
            qwen_idx = build_qwen_postshuffle_index(
                grid_t=1,
                grid_h=int(grid_h),
                grid_w=int(grid_w),
                merge_size=int(self.spatial_merge_size),
            )
            if not torch.equal(self._cached_index[key], qwen_idx):
                raise RuntimeError(
                    "Cached merger postshuffle order is inconsistent with Qwen native visual order: "
                    f"grid={int(grid_h)}x{int(grid_w)}, merge={int(self.spatial_merge_size)}"
                )
            self._validated_qwen_postshuffle_contract.add(key)
        return self._cached_index[key].to(device=device)

    def _get_inverse_postshuffle_index(self, *, grid_h: int, grid_w: int, device: torch.device) -> torch.LongTensor:
        key = (int(grid_h), int(grid_w))
        if key not in self._cached_index_inv:
            idx = build_postshuffle_index(
                grid_h=int(grid_h),
                grid_w=int(grid_w),
                merge_size=int(self.spatial_merge_size),
            )
            inv = torch.empty_like(idx)
            inv[idx] = torch.arange(idx.numel(), dtype=idx.dtype)
            self._cached_index_inv[key] = inv
        return self._cached_index_inv[key].to(device=device)

    def _split_sizes_from_grid(self, grid_thw: torch.Tensor) -> list[int]:
        return [int(x) for x in (grid_thw[:, 0] * grid_thw[:, 1] * grid_thw[:, 2]).tolist()]

    def _tokens_postshuffle_to_row_major(self, tokens: torch.Tensor, *, grid_h: int, grid_w: int) -> torch.Tensor:
        inv = self._get_inverse_postshuffle_index(grid_h=int(grid_h), grid_w=int(grid_w), device=tokens.device)
        return tokens.index_select(dim=0, index=inv)

    def _tokens_row_major_to_postshuffle(self, tokens: torch.Tensor, *, grid_h: int, grid_w: int) -> torch.Tensor:
        idx = self._get_postshuffle_index(grid_h=int(grid_h), grid_w=int(grid_w), device=tokens.device)
        return tokens.index_select(dim=0, index=idx)

    @staticmethod
    def _infer_module_dtype(module: nn.Module, *, fallback: torch.dtype) -> torch.dtype:
        raw_dtype = getattr(module, "dtype", None)
        if isinstance(raw_dtype, torch.dtype):
            return raw_dtype

        for p in module.parameters(recurse=True):
            if torch.is_floating_point(p):
                return p.dtype
        for b in module.buffers(recurse=True):
            if torch.is_floating_point(b):
                return b.dtype
        return fallback

    def _get_dino_layers(self):
        layers = getattr(self.dino, "layer", None)
        if layers is not None:
            return layers

        encoder = getattr(self.dino, "encoder", None)
        layers = getattr(encoder, "layer", None)
        if layers is not None:
            return layers

        raise RuntimeError("Unexpected DINOv3 model: cannot find encoder layers.")

    def _reconstruct_rgb_image_from_qwen_pixels(self, pixel_values: torch.Tensor, grid_t: int, grid_h: int, grid_w: int) -> torch.Tensor:
        patches = pixel_values.view(
            int(grid_t),
            int(grid_h) // int(self.spatial_merge_size),
            int(grid_w) // int(self.spatial_merge_size),
            int(self.spatial_merge_size),
            int(self.spatial_merge_size),
            int(self.in_channels),
            int(self.temporal_patch_size),
            int(self.patch_size),
            int(self.patch_size),
        )
        patches = patches[:, :, :, :, :, :, 0, :, :]
        image = patches.permute(0, 5, 1, 3, 6, 2, 4, 7).contiguous()
        image = image.view(
            int(grid_t),
            int(self.in_channels),
            int(grid_h) * int(self.patch_size),
            int(grid_w) * int(self.patch_size),
        )
        image = image[:1]
        image = image.to(dtype=torch.float32)
        image = image * self.qwen_image_std + self.qwen_image_mean
        return image.clamp(0.0, 1.0)

    def _normalize_for_dino(self, image_rgb: torch.Tensor) -> torch.Tensor:
        image = (image_rgb - self.dino_image_mean.to(device=image_rgb.device)) / self.dino_image_std.to(device=image_rgb.device)
        return image.to(dtype=self._dino_dtype)

    def _run_dino_group(
        self,
        *,
        pixel_values: torch.Tensor,
        grid_h: int,
        grid_w: int,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        expected_patches = int(grid_h) * int(grid_w)
        patch_start = 1 + int(self.num_register_tokens)
        need_deepstack = self.deepstack_merger_list is not None and len(self.deepstack_merger_list) > 0
        dino_layers = self._get_dino_layers()

        captured: dict[int, torch.Tensor] = {}
        handles: list[Any] = []
        try:
            if need_deepstack:
                for layer_idx in self.deepstack_visual_indexes:
                    raw_idx = int(layer_idx)
                    idx = int(self._deepstack_index_map.get(raw_idx, raw_idx))

                    def _make_hook(i: int):
                        def _hook(_module, _inputs, output):
                            if isinstance(output, torch.Tensor):
                                captured[i] = output

                        return _hook

                    handles.append(dino_layers[idx].register_forward_hook(_make_hook(raw_idx)))

            with torch.no_grad():
                outputs = self.dino(pixel_values=pixel_values, return_dict=True)
        finally:
            for h in handles:
                try:
                    h.remove()
                except Exception:
                    pass

        last = outputs.last_hidden_state
        if int(last.shape[1]) < int(patch_start + expected_patches):
            raise ValueError(
                f"Unexpected DINO token length. Got {int(last.shape[1])}, need >= {int(patch_start + expected_patches)}."
            )

        idx = self._get_postshuffle_index(grid_h=int(grid_h), grid_w=int(grid_w), device=last.device)
        patches = last[:, patch_start : patch_start + expected_patches, :].index_select(dim=1, index=idx)

        deepstack_features: list[torch.Tensor] = []
        if need_deepstack:
            for layer_idx in self.deepstack_visual_indexes:
                hs = captured.get(int(layer_idx))
                if hs is None:
                    raise RuntimeError(f"Missing captured DINO deepstack feature for layer {int(layer_idx)}")
                hs_patches = hs[:, patch_start : patch_start + expected_patches, :].index_select(dim=1, index=idx)
                deepstack_features.append(hs_patches)

        return patches, deepstack_features

    def _extract_dino_tokens_from_qwen_input(self, pixel_values: torch.Tensor, grid_thw: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        split_sizes = self._split_sizes_from_grid(grid_thw)
        pixel_splits = list(torch.split(pixel_values, split_sizes, dim=0))

        size_to_indices: dict[tuple[int, int, int], list[int]] = {}
        images_rgb: list[torch.Tensor] = []
        for i, (pv, g) in enumerate(zip(pixel_splits, grid_thw)):
            grid_t, grid_h, grid_w = (int(g[0].item()), int(g[1].item()), int(g[2].item()))
            img = self._reconstruct_rgb_image_from_qwen_pixels(pv, grid_t=grid_t, grid_h=grid_h, grid_w=grid_w)
            images_rgb.append(img.squeeze(0))
            key = (grid_t, grid_h, grid_w)
            size_to_indices.setdefault(key, []).append(i)

        per_image_main: list[torch.Tensor | None] = [None for _ in pixel_splits]
        deepstack_count = int(len(self.deepstack_visual_indexes))
        per_image_deepstack: list[list[torch.Tensor | None]] = [
            [None for _ in pixel_splits] for _ in range(deepstack_count)
        ]

        for key, indices in size_to_indices.items():
            grid_t, grid_h, grid_w = key
            if int(grid_t) != 1:
                raise ValueError(f"Only image input with grid_t=1 is supported, got grid_t={grid_t}")
            batch_rgb = torch.stack([images_rgb[i] for i in indices], dim=0).to(device=pixel_values.device, dtype=torch.float32)
            dino_pixels = self._normalize_for_dino(batch_rgb)
            dino_main, dino_deepstack = self._run_dino_group(
                pixel_values=dino_pixels,
                grid_h=int(grid_h),
                grid_w=int(grid_w),
            )
            for local_i, global_i in enumerate(indices):
                per_image_main[global_i] = dino_main[local_i]
                for deep_i in range(deepstack_count):
                    per_image_deepstack[deep_i][global_i] = dino_deepstack[deep_i][local_i]

        if any(x is None for x in per_image_main):
            raise RuntimeError("Incomplete DINO main feature extraction.")
        for deep_i in range(deepstack_count):
            if any(x is None for x in per_image_deepstack[deep_i]):
                raise RuntimeError(f"Incomplete DINO deepstack feature extraction for branch {deep_i}.")

        main = torch.cat([x for x in per_image_main if x is not None], dim=0)
        deepstack = [
            torch.cat([x for x in branch if x is not None], dim=0)
            for branch in per_image_deepstack
        ]
        return main, deepstack

    def forward(self, pixel_values: torch.Tensor, grid_thw: torch.Tensor, **kwargs):
        if pixel_values.ndim != 2:
            raise ValueError(
                "SVADeepstackCAVisualAdapter expects pixel_values shape [num_patches, patch_dim], "
                f"got {tuple(pixel_values.shape)}"
            )
        actual_grid_thw = kwargs.pop("actual_grid_thw", None)
        if grid_thw is None and actual_grid_thw is None:
            raise ValueError("grid_thw or actual_grid_thw is required.")
        if grid_thw is not None:
            if grid_thw.ndim != 2 or int(grid_thw.shape[-1]) != 3:
                raise ValueError(f"grid_thw must have shape [num_images, 3], got {tuple(grid_thw.shape)}")
            grid_thw = grid_thw.to(device=pixel_values.device, dtype=torch.long)
        if actual_grid_thw is None:
            actual_grid_thw = grid_thw
        if actual_grid_thw is None:
            raise ValueError("actual_grid_thw is required.")
        if actual_grid_thw.ndim != 2 or int(actual_grid_thw.shape[-1]) != 3:
            raise ValueError(f"actual_grid_thw must have shape [num_images, 3], got {tuple(actual_grid_thw.shape)}")
        actual_grid_thw = actual_grid_thw.to(device=pixel_values.device, dtype=torch.long)
        if not torch.all(actual_grid_thw[:, 0] == 1):
            raise ValueError(
                "SVA deepstack CA adapter currently supports image input only (grid_t=1). "
                f"Got actual_grid_thw={actual_grid_thw.tolist()}"
            )
        if grid_thw is not None and int(grid_thw.shape[0]) != int(actual_grid_thw.shape[0]):
            raise ValueError(
                "grid_thw and actual_grid_thw batch size mismatch: "
                f"grid={tuple(grid_thw.shape)}, actual={tuple(actual_grid_thw.shape)}"
            )
        if grid_thw is not None:
            expected_fixed_grid = torch.tensor(
                [[1, int(self.latent_grid_h), int(self.latent_grid_w)]],
                dtype=torch.long,
                device=grid_thw.device,
            ).repeat(int(grid_thw.shape[0]), 1)
            if not torch.equal(grid_thw, expected_fixed_grid):
                raise ValueError(
                    "SVA fixed-grid visual adapter expects image_grid_thw to be the fixed latent grid, "
                    f"expected={expected_fixed_grid.tolist()}, got={grid_thw.tolist()}"
                )

        with torch.no_grad():
            qwen_outputs = self.qwen_token_extractor.forward_tokens(pixel_values, actual_grid_thw, **kwargs)
            dino_main_tokens, dino_deepstack_tokens = self._extract_dino_tokens_from_qwen_input(pixel_values, actual_grid_thw)

        if self.input_proj_dino is not None:
            dino_main_tokens = self.input_proj_dino(dino_main_tokens)
            dino_deepstack_tokens = [self.input_proj_dino(x) for x in dino_deepstack_tokens]

        split_sizes = self._split_sizes_from_grid(actual_grid_thw)
        qwen_main_splits = list(torch.split(qwen_outputs.patch_tokens, split_sizes, dim=0))
        dino_main_splits = list(torch.split(dino_main_tokens, split_sizes, dim=0))

        main_embed_list: list[torch.Tensor] = []
        for q_main, d_main, grid in zip(qwen_main_splits, dino_main_splits, actual_grid_thw):
            actual_grid_h = int(grid[1].item())
            actual_grid_w = int(grid[2].item())
            q_main_row_major = self._tokens_postshuffle_to_row_major(q_main, grid_h=actual_grid_h, grid_w=actual_grid_w)
            d_main_row_major = self._tokens_postshuffle_to_row_major(d_main, grid_h=actual_grid_h, grid_w=actual_grid_w)
            latent_main = self.sva_main(
                q_main,
                d_main,
                actual_grid_h=actual_grid_h,
                actual_grid_w=actual_grid_w,
                qwen_row_major_tokens=q_main_row_major,
                dino_row_major_tokens=d_main_row_major,
            )
            latent_main_post = self._tokens_row_major_to_postshuffle(
                latent_main,
                grid_h=int(self.latent_grid_h),
                grid_w=int(self.latent_grid_w),
            )
            main_embed_list.append(self.merger(latent_main_post))
        image_embeds = torch.cat(main_embed_list, dim=0)

        expected_deepstack = int(len(self.deepstack_visual_indexes))
        qwen_deepstack_count = int(len(qwen_outputs.deepstack_patch_tokens))
        dino_deepstack_count = int(len(dino_deepstack_tokens))
        sva_deepstack_count = int(len(self.sva_deepstack_list))
        merger_deepstack_count = int(len(self.deepstack_merger_list)) if self.deepstack_merger_list is not None else 0

        if not (
            qwen_deepstack_count == dino_deepstack_count == sva_deepstack_count == merger_deepstack_count == expected_deepstack
        ):
            raise RuntimeError(
                "Deepstack branch count mismatch in SVA deepstack CA adapter: "
                f"expected={expected_deepstack}, qwen={qwen_deepstack_count}, dino={dino_deepstack_count}, "
                f"sva={sva_deepstack_count}, merger={merger_deepstack_count}"
            )

        fused_deepstack_embeds: list[torch.Tensor] = []
        for deep_i in range(expected_deepstack):
            q_splits = list(torch.split(qwen_outputs.deepstack_patch_tokens[deep_i], split_sizes, dim=0))
            d_splits = list(torch.split(dino_deepstack_tokens[deep_i], split_sizes, dim=0))
            deep_outputs: list[torch.Tensor] = []
            for q_seg, d_seg, grid in zip(q_splits, d_splits, actual_grid_thw):
                actual_grid_h = int(grid[1].item())
                actual_grid_w = int(grid[2].item())
                q_seg_row_major = self._tokens_postshuffle_to_row_major(q_seg, grid_h=actual_grid_h, grid_w=actual_grid_w)
                d_seg_row_major = self._tokens_postshuffle_to_row_major(d_seg, grid_h=actual_grid_h, grid_w=actual_grid_w)
                latent_deep = self.sva_deepstack_list[deep_i](
                    q_seg,
                    d_seg,
                    actual_grid_h=actual_grid_h,
                    actual_grid_w=actual_grid_w,
                    qwen_row_major_tokens=q_seg_row_major,
                    dino_row_major_tokens=d_seg_row_major,
                )
                latent_deep_post = self._tokens_row_major_to_postshuffle(
                    latent_deep,
                    grid_h=int(self.latent_grid_h),
                    grid_w=int(self.latent_grid_w),
                )
                deep_outputs.append(self.deepstack_merger_list[deep_i](latent_deep_post))
            fused_deepstack_embeds.append(torch.cat(deep_outputs, dim=0))

        return image_embeds, fused_deepstack_embeds


def build_sva_deepstack_ca_adapter_config(
    *,
    qwen_model_dir: Path,
    dinov3_dir: Path,
    old_visual: nn.Module,
) -> SVADeepstackCAVisualAdapterConfig:
    qwen_mean, qwen_std = _load_preprocessor_stats(Path(qwen_model_dir))
    dino_mean, dino_std = _load_preprocessor_stats(Path(dinov3_dir))
    return SVADeepstackCAVisualAdapterConfig(
        dinov3_dir=Path(dinov3_dir),
        merge_size=int(old_visual.spatial_merge_size),
        latent_grid_h=16,
        latent_grid_w=16,
        deepstack_visual_indexes=tuple(
            int(x) for x in getattr(old_visual, "deepstack_visual_indexes", DEFAULT_DEEPSTACK_VISUAL_INDEXES)
        ),
        qwen_vision_depth=int(getattr(getattr(old_visual, "config", None), "depth", 0) or len(getattr(old_visual, "blocks", []))),
        qwen_image_mean=qwen_mean,
        qwen_image_std=qwen_std,
        dino_image_mean=dino_mean,
        dino_image_std=dino_std,
    )


def install_sva_deepstack_ca_dual_grid_runtime(*, model) -> None:
    if bool(getattr(model, "_sva_deepstack_ca_dual_grid_patched", False)):
        return

    qwen_cond = model
    qwen_core = model.model

    orig_model_forward = qwen_core.forward
    orig_get_image_features = qwen_core.get_image_features
    orig_prepare_inputs_for_generation = qwen_cond.prepare_inputs_for_generation
    orig_expand_inputs_for_generation = qwen_cond._expand_inputs_for_generation

    def patched_get_image_features(self, pixel_values: torch.FloatTensor, image_grid_thw: torch.LongTensor | None = None):
        actual_image_grid_thw = getattr(self, "_sva_actual_image_grid_thw", None)
        actual_grid = actual_image_grid_thw if actual_image_grid_thw is not None else image_grid_thw
        if actual_grid is None:
            raise ValueError("actual_image_grid_thw/image_grid_thw is required for SVA fixed-grid runtime.")
        if image_grid_thw is None:
            image_grid_thw = actual_grid

        pixel_values = pixel_values.type(self.visual.dtype)
        image_embeds, deepstack_image_embeds = self.visual(
            pixel_values,
            grid_thw=image_grid_thw,
            actual_grid_thw=actual_grid,
        )
        split_sizes = (image_grid_thw.prod(-1) // self.visual.spatial_merge_size**2).tolist()
        image_embeds = torch.split(image_embeds, split_sizes)
        return image_embeds, deepstack_image_embeds

    def patched_model_forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values=None,
        inputs_embeds: torch.FloatTensor | None = None,
        pixel_values: torch.Tensor | None = None,
        pixel_values_videos: torch.FloatTensor | None = None,
        image_grid_thw: torch.LongTensor | None = None,
        video_grid_thw: torch.LongTensor | None = None,
        cache_position: torch.LongTensor | None = None,
        **kwargs,
    ):
        actual_image_grid_thw = kwargs.pop("actual_image_grid_thw", None)
        self._sva_actual_image_grid_thw = actual_image_grid_thw
        try:
            return orig_model_forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                pixel_values=pixel_values,
                pixel_values_videos=pixel_values_videos,
                image_grid_thw=image_grid_thw,
                video_grid_thw=video_grid_thw,
                cache_position=cache_position,
                **kwargs,
            )
        finally:
            self._sva_actual_image_grid_thw = None

    def patched_prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        cache_position=None,
        position_ids=None,
        use_cache=True,
        pixel_values=None,
        pixel_values_videos=None,
        image_grid_thw=None,
        video_grid_thw=None,
        actual_image_grid_thw=None,
        **kwargs,
    ):
        model_inputs = orig_prepare_inputs_for_generation(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            position_ids=position_ids,
            use_cache=use_cache,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            **kwargs,
        )
        if actual_image_grid_thw is not None:
            model_inputs["actual_image_grid_thw"] = actual_image_grid_thw
        if cache_position is not None and int(cache_position[0]) != 0:
            model_inputs["actual_image_grid_thw"] = None
        return model_inputs

    def patched_expand_inputs_for_generation(
        self,
        expand_size: int = 1,
        is_encoder_decoder: bool = False,
        input_ids: torch.LongTensor | None = None,
        **model_kwargs,
    ):
        if expand_size == 1:
            return input_ids, model_kwargs

        actual_image_grid_thw = model_kwargs.get("actual_image_grid_thw", None)
        image_grid_thw = model_kwargs.get("image_grid_thw", None)
        video_grid_thw = model_kwargs.get("video_grid_thw", None)

        visual_keys = [
            "pixel_values",
            "image_grid_thw",
            "actual_image_grid_thw",
            "pixel_values_videos",
            "video_grid_thw",
            "second_per_grid_ts",
        ]

        image_nums, video_nums = self._get_image_nums_and_video_nums(
            input_ids,
            inputs_embeds=model_kwargs.get("inputs_embeds", None),
        )

        def _repeat_interleave_samples(x: torch.Tensor, lengths: list[int], repeat_times: int) -> torch.Tensor:
            samples = torch.split(x, lengths)
            repeat_args = [repeat_times] + [1] * (x.dim() - 1)
            return torch.cat([sample.repeat(*repeat_args) for sample in samples], dim=0)

        for key, value in list(model_kwargs.items()):
            if value is None:
                continue
            if key == "pixel_values":
                ref_grid = actual_image_grid_thw if actual_image_grid_thw is not None else image_grid_thw
                if ref_grid is None:
                    raise ValueError("Missing actual/image grid while expanding pixel_values for generation.")
                samples = torch.split(ref_grid, list(image_nums))
                lengths = [int(torch.prod(sample, dim=1).sum().item()) for sample in samples]
                model_kwargs[key] = _repeat_interleave_samples(value, lengths=lengths, repeat_times=expand_size)
            elif key == "image_grid_thw":
                model_kwargs[key] = _repeat_interleave_samples(value, lengths=[int(x) for x in image_nums], repeat_times=expand_size)
            elif key == "actual_image_grid_thw":
                model_kwargs[key] = _repeat_interleave_samples(value, lengths=[int(x) for x in image_nums], repeat_times=expand_size)
            elif key == "pixel_values_videos":
                samples = torch.split(video_grid_thw, list(video_nums))
                lengths = [int(torch.prod(sample, dim=1).sum().item()) for sample in samples]
                model_kwargs[key] = _repeat_interleave_samples(value, lengths=lengths, repeat_times=expand_size)
            elif key == "video_grid_thw":
                model_kwargs[key] = _repeat_interleave_samples(value, lengths=[int(x) for x in video_nums], repeat_times=expand_size)
            elif key == "second_per_grid_ts":
                model_kwargs[key] = _repeat_interleave_samples(value, lengths=[int(x) for x in video_nums], repeat_times=expand_size)
            elif isinstance(value, torch.Tensor) and key != "cache_position" and key not in visual_keys:
                model_kwargs[key] = value.repeat_interleave(expand_size, dim=0)

        if input_ids is not None:
            input_ids = input_ids.repeat_interleave(expand_size, dim=0)

        if is_encoder_decoder:
            encoder_outputs = model_kwargs.get("encoder_outputs", None)
            if encoder_outputs is None:
                raise ValueError("If `is_encoder_decoder` is True, make sure that `encoder_outputs` is defined.")
            for key, value in encoder_outputs.items():
                if value is not None and isinstance(value, torch.Tensor):
                    encoder_outputs[key] = value.repeat_interleave(expand_size, dim=0)

        return input_ids, model_kwargs

    qwen_core.get_image_features = types.MethodType(patched_get_image_features, qwen_core)
    qwen_core.forward = types.MethodType(patched_model_forward, qwen_core)
    qwen_cond.prepare_inputs_for_generation = types.MethodType(patched_prepare_inputs_for_generation, qwen_cond)
    qwen_cond._expand_inputs_for_generation = types.MethodType(patched_expand_inputs_for_generation, qwen_cond)
    qwen_cond._sva_deepstack_ca_dual_grid_patched = True


def attach_sva_deepstack_ca_visual_adapter(*, model, qwen_model_dir: Path, dinov3_dir: Path) -> Any:
    old_visual = model.model.visual
    cfg = build_sva_deepstack_ca_adapter_config(
        qwen_model_dir=Path(qwen_model_dir),
        dinov3_dir=Path(dinov3_dir),
        old_visual=old_visual,
    )
    adapter = SVADeepstackCAVisualAdapter(
        cfg,
        qwen_visual=old_visual,
        torch_dtype=model.dtype,
    )
    adapter = adapter.to(device=model.device, dtype=model.dtype)
    model.model.visual = adapter
    install_sva_deepstack_ca_dual_grid_runtime(model=model)
    return adapter
