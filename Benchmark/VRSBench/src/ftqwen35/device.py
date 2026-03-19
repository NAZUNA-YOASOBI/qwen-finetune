from __future__ import annotations

from typing import Any


def require_cuda() -> None:
    """强制要求 CUDA 可用。

    这个项目的训练/推理都默认跑在 GPU 上；如果 CUDA 不可用，直接报错，避免静默回退到 CPU。
    """
    try:
        import torch
    except Exception as e:  # noqa: BLE001
        raise RuntimeError("Missing dependency: torch") from e

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. This run requires a GPU.")


def assert_model_on_cuda(model: Any, *, max_bad: int = 5) -> None:
    """检查模型是否完整放在 CUDA 上，避免 device_map 自动 offload 到 CPU/disk。"""
    require_cuda()

    # transformers/accelerate 可能会挂上 hf_device_map，先用它做一次快速检查。
    device_map = getattr(model, "hf_device_map", None)
    if isinstance(device_map, dict):
        bad = [(k, str(v)) for k, v in device_map.items() if str(v) in {"cpu", "disk", "meta"}]
        if bad:
            items = ", ".join([f"{k}={v}" for k, v in bad[: int(max_bad)]])
            raise RuntimeError(f"Model is offloaded to non-CUDA devices: {items}")

    bad_params: list[str] = []
    for name, p in getattr(model, "named_parameters", lambda: [])():
        try:
            dev = p.device
        except Exception:
            continue
        if getattr(dev, "type", "") != "cuda":
            bad_params.append(f"{name}={dev}")
            if len(bad_params) >= int(max_bad):
                break

    if bad_params:
        raise RuntimeError("Model parameters are not fully on CUDA: " + ", ".join(bad_params))

