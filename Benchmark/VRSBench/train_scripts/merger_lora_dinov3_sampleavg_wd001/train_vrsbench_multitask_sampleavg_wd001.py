from __future__ import annotations

import argparse
import json
import math
import re
import sys
import warnings
from pathlib import Path

import torch
from accelerate import Accelerator
from peft import LoraConfig, PeftModel, get_peft_model  # type: ignore
from torch.utils.data import DataLoader, Sampler, Subset
from transformers import AutoImageProcessor, AutoProcessor, Qwen3VLForConditionalGeneration, get_scheduler

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from ftqwen3.dinov3_merger.dinov3_adapter import DinoV3AdapterConfig, DinoV3VisualAdapter
from ftqwen3.shared.device import require_cuda
from ftqwen3.shared.qwen_dinov3 import (
    assert_dino_runtime_matches_merger,
    load_merger_safetensors,
    resolve_dino_resize_config,
    save_merger_safetensors,
    torch_dtype_from_str,
)
from ftqwen3.shared.sft import CaptionSFTCollator, VrsbenchMultiTaskSFTDataset
from ftqwen3.shared.training_losses import causal_lm_sample_average_loss


def _project_root() -> Path:
    return PROJECT_ROOT


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


def _resolve_language_lora_targets(base_model, target_leaf_names: list[str]) -> list[str]:
    """把 LoRA target 限定在 language_model，避免误命中 visual 分支。"""
    leaf_set = {str(x).strip() for x in target_leaf_names if str(x).strip()}
    if not leaf_set:
        raise ValueError("Empty LoRA target module list.")

    matched: list[str] = []
    for name, _module in base_model.named_modules():
        if not name:
            continue
        leaf = name.rsplit(".", 1)[-1]
        if leaf not in leaf_set:
            continue
        if ".language_model." not in f".{name}.":
            continue
        matched.append(name)

    matched = sorted(set(matched))
    if not matched:
        want = ", ".join(sorted(leaf_set))
        raise RuntimeError(f"No language_model LoRA targets matched. wanted=[{want}]")
    return matched


def _assert_no_visual_lora_trainables(model) -> None:
    """防止把 LoRA 打到 visual 分支，避免实验口径污染。"""
    bad: list[str] = []
    for name, p in model.named_parameters():
        if "lora_" not in name:
            continue
        if ".visual." in f".{name}." and bool(getattr(p, "requires_grad", False)):
            bad.append(name)
            if len(bad) >= 8:
                break
    if bad:
        show = ", ".join(bad)
        raise RuntimeError("Found trainable visual LoRA params (expected LLM-only LoRA): " + show)


def _assert_expected_trainables_layout(model) -> None:
    bad: list[str] = []
    has_visual = False
    has_lora = False
    for name, p in model.named_parameters():
        if not bool(getattr(p, "requires_grad", False)):
            continue
        scoped = f".{name}."
        is_visual = (
            ".visual.merger." in scoped
            or ".visual.deepstack_merger_list." in scoped
            or ".visual.input_proj." in scoped
        )
        is_lora = "lora_" in name
        if is_visual:
            has_visual = True
        if is_lora:
            has_lora = True
        if not (is_visual or is_lora):
            bad.append(name)
            if len(bad) >= 8:
                break
    if bad:
        show = ", ".join(bad)
        raise RuntimeError("Found unexpected trainable params outside merger/deepstack/input_proj + LoRA: " + show)
    if not has_visual:
        raise RuntimeError("No trainable visual merger params found.")
    if not has_lora:
        raise RuntimeError("No trainable LoRA params found.")


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _parse_int_list(raw: str, *, sep: str) -> list[int]:
    values = [x.strip() for x in str(raw).split(sep) if x.strip()]
    if not values:
        return []
    out: list[int] = []
    for x in values:
        v = int(x)
        if v <= 0:
            raise ValueError(f"Batch values must be > 0, got {v} from '{raw}'.")
        out.append(v)
    return out


def _resolve_local_batch_sizes(*, world_size: int, base_batch: int, per_rank: str, ratio: str) -> list[int]:
    if world_size <= 0:
        raise ValueError(f"Invalid world_size={world_size}")

    if str(per_rank).strip():
        sizes = _parse_int_list(str(per_rank), sep=",")
        if len(sizes) != int(world_size):
            raise ValueError(
                f"--batch-size-per-rank expects {world_size} values, got {len(sizes)} from '{per_rank}'."
            )
        return sizes

    ratios = _parse_int_list(str(ratio), sep=":")
    if not ratios:
        ratios = [1] * int(world_size)
    if len(ratios) != int(world_size):
        raise ValueError(f"--batch-size-ratio expects {world_size} values, got {len(ratios)} from '{ratio}'.")

    base = int(base_batch)
    if base <= 0:
        raise ValueError(f"--batch-size must be > 0, got {base}.")

    base_ratio = int(ratios[0])
    sizes: list[int] = []
    for r in ratios:
        # 以 rank0 的 --batch-size 为基准，按 ratio 缩放各 rank 的本地 micro-batch。
        scaled = int(round(float(base) * float(r) / float(base_ratio)))
        sizes.append(max(1, scaled))
    return sizes


VRSBENCH_TASK_ORDER = ("caption", "refer", "vqa")


def _extract_vrsbench_task(item: dict) -> str:
    conv = item.get("conversations", [])
    if not isinstance(conv, list) or not conv:
        raise ValueError("Invalid conversations field while splitting VRSBench tasks.")

    human_text = ""
    for msg in conv:
        if not isinstance(msg, dict):
            continue
        if str(msg.get("from", "")).lower() == "human":
            human_text = str(msg.get("value", ""))
            break

    m = re.search(r"\[(caption|refer|vqa)\]", human_text, flags=re.IGNORECASE)
    if m is None:
        preview = str(human_text).replace("\n", " ")[:120]
        raise ValueError(f"Cannot infer task tag from human prompt: {preview}")
    task = str(m.group(1)).lower()
    if task not in VRSBENCH_TASK_ORDER:
        raise ValueError(f"Unsupported task tag {task} in VRSBench sample.")
    return task


class ProportionalDistributedSampler(Sampler[int]):
    """按每个 rank 的 micro-batch 比例切分数据，并按桶做“全局 batch=32”级别采样。"""

    def __init__(
        self,
        *,
        dataset_size: int,
        local_batch_sizes: list[int],
        rank: int,
        seed: int,
        grad_accum: int,
        sample_bucket_keys: list[tuple[int, int, int, int, int]],
    ) -> None:
        self.dataset_size = int(dataset_size)
        self.local_batch_sizes = [int(x) for x in local_batch_sizes]
        self.rank = int(rank)
        self.seed = int(seed)
        self.grad_accum = int(grad_accum)
        self.epoch = 0

        if self.dataset_size <= 0:
            raise ValueError(f"Invalid dataset_size={self.dataset_size}.")
        if self.rank < 0 or self.rank >= len(self.local_batch_sizes):
            raise ValueError(f"Invalid rank={self.rank} for local_batch_sizes={self.local_batch_sizes}.")
        if self.grad_accum <= 0:
            raise ValueError(f"Invalid grad_accum={self.grad_accum}.")
        if len(sample_bucket_keys) != self.dataset_size:
            raise ValueError(
                "sample_bucket_keys length mismatch: "
                f"expect={self.dataset_size}, got={len(sample_bucket_keys)}"
            )

        self.global_micro_batch = int(sum(self.local_batch_sizes))
        if self.global_micro_batch <= 0:
            raise ValueError(f"Invalid global_micro_batch={self.global_micro_batch}.")

        self.global_batch = int(self.global_micro_batch * self.grad_accum)
        self.local_batch_size = int(self.local_batch_sizes[self.rank])
        self.local_samples_per_update = int(self.local_batch_size * self.grad_accum)
        self.rank_offset = int(sum(self.local_batch_sizes[: self.rank]))

        bucket_to_indices: dict[tuple[int, int, int, int, int], list[int]] = {}
        for idx, key in enumerate(sample_bucket_keys):
            bucket_key = tuple(int(x) for x in key)
            bucket_to_indices.setdefault(bucket_key, []).append(int(idx))
        if not bucket_to_indices:
            raise RuntimeError("No bucket keys found for sampler.")
        self.bucket_to_indices = {key: tuple(indices) for key, indices in bucket_to_indices.items()}
        self.bucket_keys = sorted(self.bucket_to_indices.keys())

        self.bucket_sizes = {key: len(indices) for key, indices in self.bucket_to_indices.items()}
        self.bucket_used = {
            key: (len(indices) // self.global_batch) * self.global_batch
            for key, indices in self.bucket_to_indices.items()
        }
        self.bucket_dropped = {key: int(self.bucket_sizes[key] - self.bucket_used[key]) for key in self.bucket_keys}

        self.steps_per_epoch = int(sum(self.bucket_used[key] for key in self.bucket_keys) // self.global_batch)
        if self.steps_per_epoch <= 0:
            raise RuntimeError(
                "Dataset too small after per-bucket drop for one optimizer step. "
                f"dataset_size={self.dataset_size}, global_batch={self.global_batch}."
            )

        self.total_used_samples = int(self.steps_per_epoch * self.global_batch)
        self.total_dropped_samples = int(self.dataset_size - self.total_used_samples)
        self.local_num_samples = int(self.steps_per_epoch * self.local_samples_per_update)

    def __len__(self) -> int:
        return int(self.local_num_samples)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self):
        generator = torch.Generator()
        generator.manual_seed(int(self.seed) + int(self.epoch))

        global_update_groups: list[list[int]] = []
        for key in self.bucket_keys:
            indices = self.bucket_to_indices[key]
            used = int(self.bucket_used[key])
            if used <= 0:
                continue
            order = torch.randperm(len(indices), generator=generator).tolist()
            shuffled = [int(indices[i]) for i in order]
            for start in range(0, used, self.global_batch):
                global_update_groups.append(shuffled[start : start + self.global_batch])

        if len(global_update_groups) != int(self.steps_per_epoch):
            raise RuntimeError(
                f"Update group count mismatch: expect={self.steps_per_epoch}, got={len(global_update_groups)}"
            )

        if len(global_update_groups) > 1:
            order = torch.randperm(len(global_update_groups), generator=generator).tolist()
            global_update_groups = [global_update_groups[i] for i in order]

        local_indices: list[int] = []
        for update_group in global_update_groups:
            if len(update_group) != int(self.global_batch):
                raise RuntimeError(
                    f"Global batch size mismatch: expect={self.global_batch}, got={len(update_group)}"
                )

            for micro_idx in range(int(self.grad_accum)):
                start = int(micro_idx * self.global_micro_batch)
                end = int(start + self.global_micro_batch)
                micro_group = update_group[start:end]
                if len(micro_group) != int(self.global_micro_batch):
                    raise RuntimeError(
                        f"Micro group size mismatch: expect={self.global_micro_batch}, got={len(micro_group)}"
                    )
                local_start = int(self.rank_offset)
                local_end = int(local_start + self.local_batch_size)
                local_chunk = micro_group[local_start:local_end]
                if len(local_chunk) != int(self.local_batch_size):
                    raise RuntimeError(
                        f"Local chunk size mismatch: expect={self.local_batch_size}, got={len(local_chunk)}"
                    )
                local_indices.extend(int(x) for x in local_chunk)

        if len(local_indices) != int(self.local_num_samples):
            raise RuntimeError(
                f"Local sampled size mismatch: expect={self.local_num_samples}, got={len(local_indices)}"
            )
        return iter(local_indices)


def _move_batch_to_device(batch: dict[str, object], *, device: torch.device) -> dict[str, object]:
    out: dict[str, object] = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            out[key] = value.to(device=device, non_blocking=True)
        else:
            out[key] = value
    return out


def _enable_visual_trainables(target_model: torch.nn.Module) -> None:
    visual = target_model.model.visual
    for parameter in visual.merger.parameters():
        parameter.requires_grad = True
    if getattr(visual, "deepstack_merger_list", None) is not None:
        for parameter in visual.deepstack_merger_list.parameters():
            parameter.requires_grad = True
    if getattr(visual, "input_proj", None) is not None:
        for parameter in visual.input_proj.parameters():
            parameter.requires_grad = True


def main() -> None:
    parser = argparse.ArgumentParser(description="VRSBench multi-task SFT: train merger + LLM LoRA (DINOv3 visual, sample-average loss).")
    parser.add_argument("--qwen-model-dir", type=str, default="models/Qwen3-VL-8B-Instruct")
    parser.add_argument(
        "--dinov3-dir",
        type=str,
        default="models/dinov3/dinov3-vitl16-pretrain-sat493m",
    )
    parser.add_argument("--dataset-root", type=str, default="datasets/VRSBench")
    parser.add_argument("--train-json", type=str, default="datasets/VRSBench/VRSBench_train.json")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--smart-resize-min-pixels", type=int, default=None)
    parser.add_argument("--smart-resize-max-pixels", type=int, default=None)

    parser.add_argument("--output-dir", type=str, default="checkpoints/vrsbench_joint/merger_lora_sampleavg_wd001")
    parser.add_argument("--init-merger", type=str, default="", help="Merger safetensors to init from (optional).")
    parser.add_argument("--resume-lora", type=str, default="", help="LoRA dir to resume from.")

    parser.add_argument("--epochs", type=float, default=5.0)
    parser.add_argument("--max-steps", type=int, default=0, help="If >0, stop after this many optimizer steps.")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument(
        "--batch-size-per-rank",
        type=str,
        default="8,24",
        help="Optional: comma-separated local micro-batch per rank, e.g. '8,24'.",
    )
    parser.add_argument(
        "--batch-size-ratio",
        type=str,
        default="1:1",
        help="Optional: per-rank ratio list (':' separated). Used when --batch-size-per-rank is empty.",
    )
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--mixed-precision", type=str, default="bf16", choices=["no", "fp16", "bf16"])
    parser.add_argument("--gradient-checkpointing", action="store_true")

    parser.add_argument("--lora-r", type=int, default=64)
    parser.add_argument("--lora-alpha", type=int, default=64)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument(
        "--lora-target",
        type=str,
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
        help="Comma-separated module names.",
    )

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=2)
    args = parser.parse_args()

    if (args.smart_resize_min_pixels is None) ^ (args.smart_resize_max_pixels is None):
        raise ValueError("--smart-resize-min-pixels and --smart-resize-max-pixels must be both set or both unset.")
    if args.smart_resize_min_pixels is not None:
        if int(args.smart_resize_min_pixels) <= 0:
            raise ValueError(f"--smart-resize-min-pixels must be > 0, got {args.smart_resize_min_pixels}")
        if int(args.smart_resize_max_pixels) < int(args.smart_resize_min_pixels):
            raise ValueError(
                "--smart-resize-max-pixels must be >= --smart-resize-min-pixels, "
                f"got {args.smart_resize_max_pixels} < {args.smart_resize_min_pixels}"
            )

    torch.manual_seed(int(args.seed))

    require_cuda()
    accelerator = Accelerator(mixed_precision=None if args.mixed_precision == "no" else str(args.mixed_precision))
    device = accelerator.device
    if str(getattr(device, "type", "")) != "cuda":
        raise RuntimeError(f"Unexpected accelerator device: {device}. This run requires CUDA.")
    if accelerator.is_main_process:
        world_size = int(getattr(accelerator, "num_processes", 1))
        visible_cuda = int(torch.cuda.device_count())
        if world_size == 1 and visible_cuda > 1:
            print(
                "[WARN] Detected multiple CUDA devices but current run uses a single process. "
                "Use: accelerate launch --num_processes 2 ...",
                flush=True,
            )

    world_size = int(getattr(accelerator, "num_processes", 1))
    rank = int(getattr(accelerator, "process_index", 0))
    local_batch_sizes = _resolve_local_batch_sizes(
        world_size=world_size,
        base_batch=int(args.batch_size),
        per_rank=str(args.batch_size_per_rank),
        ratio=str(args.batch_size_ratio),
    )
    local_batch_size = int(local_batch_sizes[rank])
    global_micro_batch = int(sum(local_batch_sizes))
    # DDP 会对各 rank 梯度做平均；这里按每个 rank 的样本数缩放 loss，
    # 使全局梯度等价于 sample-level 平均。
    loss_scale_mode = "sample_true_mean"
    sample_ddp_loss_scale = float(world_size) * float(local_batch_size) / float(global_micro_batch)

    qwen_model_dir = _resolve_from_project(args.qwen_model_dir)
    dinov3_dir = _resolve_from_project(args.dinov3_dir)
    dataset_root = _resolve_from_project(args.dataset_root)
    train_json = _resolve_from_project(args.train_json)
    out_dir = _resolve_from_project(args.output_dir)

    resume_lora_dir = _resolve_from_project(args.resume_lora) if str(args.resume_lora).strip() else None
    init_merger_path = _resolve_from_project(args.init_merger) if str(args.init_merger).strip() else None
    resume_step = 0
    resume_optimizer_path: Path | None = None
    resume_scheduler_path: Path | None = None

    if resume_lora_dir is not None:
        # 约定：LoRA 目录与 merger.safetensors 在同一层级（epoch*/lora 对应 epoch*/merger.safetensors）。
        auto_merger = resume_lora_dir.parent / "merger.safetensors"
        if not auto_merger.is_file():
            raise FileNotFoundError(f"Missing sibling merger checkpoint for resume_lora: {auto_merger}")

        if init_merger_path is None:
            init_merger_path = auto_merger
        elif init_merger_path.resolve() != auto_merger.resolve():
            raise ValueError(
                "resume_lora and init_merger mismatch. "
                f"resume expects {auto_merger}, but got {init_merger_path}."
            )

        meta_path = auto_merger.with_suffix(".json")
        if not meta_path.is_file():
            raise FileNotFoundError(f"Missing merger metadata for strict resume: {meta_path}")
        try:
            resume_step = max(0, int(_read_json(meta_path).get("step", 0)))
        except Exception as e:
            raise RuntimeError(f"Failed to read resume step from merger metadata: {meta_path}") from e
        optimizer_path = resume_lora_dir.parent / "optimizer.pt"
        scheduler_path = resume_lora_dir.parent / "scheduler.pt"
        if not optimizer_path.is_file():
            raise FileNotFoundError(f"Missing optimizer state for strict resume: {optimizer_path}")
        if not scheduler_path.is_file():
            raise FileNotFoundError(f"Missing scheduler state for strict resume: {scheduler_path}")
        resume_optimizer_path = optimizer_path
        resume_scheduler_path = scheduler_path

    if init_merger_path is not None:
        resize_cfg = assert_dino_runtime_matches_merger(
            qwen_model_dir=qwen_model_dir,
            dinov3_dir=dinov3_dir,
            image_size=int(args.image_size),
            smart_resize_min_pixels=args.smart_resize_min_pixels,
            smart_resize_max_pixels=args.smart_resize_max_pixels,
            merger_ckpt=init_merger_path,
        )
    else:
        resize_cfg = resolve_dino_resize_config(
            image_size=int(args.image_size),
            smart_resize_min_pixels=args.smart_resize_min_pixels,
            smart_resize_max_pixels=args.smart_resize_max_pixels,
            merger_ckpt=None,
        )
    args.image_size = int(resize_cfg.image_size)
    args.smart_resize_min_pixels = int(resize_cfg.smart_resize_min_pixels)
    args.smart_resize_max_pixels = int(resize_cfg.smart_resize_max_pixels)

    out_dir.mkdir(parents=True, exist_ok=True)

    run_meta = {
        "qwen_model_dir": _rel_to_project(qwen_model_dir),
        "dinov3_dir": _rel_to_project(dinov3_dir),
        "image_size": int(args.image_size),
        "smart_resize_min_pixels": int(args.smart_resize_min_pixels),
        "smart_resize_max_pixels": int(args.smart_resize_max_pixels),
    }

    with train_json.open("r", encoding="utf-8") as f:
        items = json.load(f)
    if not isinstance(items, list) or not items:
        raise ValueError(f"Invalid train json: {train_json}")

    processor = AutoProcessor.from_pretrained(str(qwen_model_dir))
    tokenizer = processor.tokenizer
    image_processor = AutoImageProcessor.from_pretrained(str(dinov3_dir))

    model_dtype = torch_dtype_from_str(
        "bf16"
        if args.mixed_precision == "bf16"
        else "fp16"
        if args.mixed_precision == "fp16"
        else "fp32"
    )
    base_model = Qwen3VLForConditionalGeneration.from_pretrained(
        str(qwen_model_dir),
        torch_dtype=model_dtype,
    )
    base_model.config.use_cache = False

    old_visual = base_model.model.visual
    adapter_cfg = DinoV3AdapterConfig(
        dinov3_dir=dinov3_dir,
        image_size=int(args.image_size),
        merge_size=int(old_visual.spatial_merge_size),
        deepstack_visual_indexes=tuple(int(x) for x in getattr(old_visual, "deepstack_visual_indexes", (5, 11, 17))),
        qwen_vision_depth=int(getattr(getattr(old_visual, "config", None), "depth", 0) or len(getattr(old_visual, "blocks", []))),
    )
    adapter = DinoV3VisualAdapter(
        adapter_cfg,
        merger=old_visual.merger,
        deepstack_merger_list=getattr(old_visual, "deepstack_merger_list", None),
        torch_dtype=base_model.dtype,
    )
    base_model.model.visual = adapter

    # 先冻结所有参数，再启用 merger 与 LoRA 参数
    for p in base_model.parameters():
        p.requires_grad = False
    _enable_visual_trainables(base_model)

    if init_merger_path is not None:
        load_merger_safetensors(base_model, init_merger_path)

    lora_target_leaf = [x.strip() for x in str(args.lora_target).split(",") if x.strip()]
    lora_targets = _resolve_language_lora_targets(base_model, lora_target_leaf)
    lora_cfg = LoraConfig(
        r=int(args.lora_r),
        lora_alpha=int(args.lora_alpha),
        lora_dropout=float(args.lora_dropout),
        bias="none",
        target_modules=lora_targets,
        task_type="CAUSAL_LM",
    )

    if resume_lora_dir is not None:
        # 直接把 LoRA 权重加载到 base_model，避免 get_peft_model + from_pretrained 造成嵌套 PEFT。
        model = PeftModel.from_pretrained(base_model, str(resume_lora_dir), is_trainable=True)
    else:
        model = get_peft_model(base_model, lora_cfg)

    # PEFT 会默认冻结非 LoRA 参数，这里显式重新打开视觉 adapter 的 trainable 参数。
    _enable_visual_trainables(model.get_base_model())
    _assert_no_visual_lora_trainables(model)
    _assert_expected_trainables_layout(model)

    if args.gradient_checkpointing:
        try:
            base = model.get_base_model()
            base.model.language_model.gradient_checkpointing_enable()
        except Exception:
            pass

    dataset = VrsbenchMultiTaskSFTDataset(
        items,
        tokenizer=tokenizer,
        image_processor=image_processor,
        dataset_root=dataset_root,
        split="train",
        image_size=int(args.image_size),
        patch_size=int(base_model.config.vision_config.patch_size),
        merge_size=int(base_model.config.vision_config.spatial_merge_size),
        smart_resize_min_pixels=int(args.smart_resize_min_pixels),
        smart_resize_max_pixels=int(args.smart_resize_max_pixels),
        image_token="<|image_pad|>",
    )
    collator = CaptionSFTCollator(pad_token_id=int(tokenizer.pad_token_id))

    dataset.ensure_image_hw_cache(build_if_missing=bool(accelerator.is_main_process))
    accelerator.wait_for_everyone()
    dataset.ensure_image_hw_cache(build_if_missing=False)

    task_order = tuple(VRSBENCH_TASK_ORDER)
    task_indices: dict[str, list[int]] = {name: [] for name in task_order}
    for idx, item in enumerate(items):
        task_name = _extract_vrsbench_task(item)
        task_indices[task_name].append(int(idx))

    if sum(len(v) for v in task_indices.values()) != len(items):
        raise RuntimeError("Task split size mismatch for VRSBench train set.")

    task_samplers: dict[str, ProportionalDistributedSampler] = {}
    task_loaders: dict[str, DataLoader] = {}
    task_stats: dict[str, dict[str, object]] = {}
    effective_global_batch = int(global_micro_batch * int(args.grad_accum))

    for task_pos, task_name in enumerate(task_order):
        indices = task_indices[task_name]
        task_size = int(len(indices))
        if task_size <= 0:
            raise RuntimeError(f"Task subset is empty: {task_name}.")

        task_dataset = Subset(dataset, indices)
        task_bucket_keys = [dataset.get_resize_bucket_key(int(dataset_idx)) for dataset_idx in indices]
        task_sampler = ProportionalDistributedSampler(
            dataset_size=task_size,
            local_batch_sizes=local_batch_sizes,
            rank=rank,
            seed=int(args.seed) + int(task_pos) * 1000,
            grad_accum=int(args.grad_accum),
            sample_bucket_keys=task_bucket_keys,
        )
        task_loader = DataLoader(
            task_dataset,
            batch_size=int(local_batch_size),
            shuffle=False,
            sampler=task_sampler,
            num_workers=int(args.num_workers),
            pin_memory=True,
            collate_fn=collator,
            drop_last=True,
        )

        local_loader_len = int(len(task_loader))
        local_loader_len_t = torch.tensor([local_loader_len], device=device, dtype=torch.long)
        all_loader_lens = accelerator.gather(local_loader_len_t).detach().cpu().tolist()
        min_loader_len = int(min(int(x) for x in all_loader_lens))
        max_loader_len = int(max(int(x) for x in all_loader_lens))
        if min_loader_len != max_loader_len:
            raise RuntimeError(
                "Loader length mismatch across ranks under proportional sampler. "
                f"task={task_name}, lens={all_loader_lens}, batch_per_rank={local_batch_sizes}."
            )

        task_steps_per_epoch = math.floor(float(min_loader_len) / max(1, int(args.grad_accum)))
        if task_steps_per_epoch <= 0:
            raise RuntimeError(
                f"steps_per_epoch is 0 for task={task_name}. "
                "Reduce batch_size/grad_accum or check task subset size."
            )

        task_used_items_per_epoch = int(task_steps_per_epoch * effective_global_batch)
        if task_used_items_per_epoch > task_size:
            raise RuntimeError(
                f"Computed used_items_per_epoch exceeds task size: task={task_name}, "
                f"used={task_used_items_per_epoch}, size={task_size}."
            )
        task_dropped_items_per_epoch = int(task_size - task_used_items_per_epoch)
        task_samples_per_rank = [int(task_steps_per_epoch * int(args.grad_accum) * b) for b in local_batch_sizes]

        task_samplers[task_name] = task_sampler
        task_loaders[task_name] = task_loader
        task_stats[task_name] = {
            "items": task_size,
            "loader_len_per_rank": [int(x) for x in all_loader_lens],
            "min_loader_len": int(min_loader_len),
            "steps_per_epoch": int(task_steps_per_epoch),
            "samples_per_rank_per_epoch": [int(x) for x in task_samples_per_rank],
            "used_items_per_epoch": int(task_used_items_per_epoch),
            "dropped_items_per_epoch": int(task_dropped_items_per_epoch),
        }

    used_items_per_epoch = int(sum(int(task_stats[t]["used_items_per_epoch"]) for t in task_order))
    dropped_items_per_epoch = int(len(dataset) - used_items_per_epoch)
    samples_per_rank_per_epoch = [0 for _ in local_batch_sizes]
    for task_name in task_order:
        vals = [int(x) for x in task_stats[task_name]["samples_per_rank_per_epoch"]]
        for i, v in enumerate(vals):
            samples_per_rank_per_epoch[i] += int(v)

    steps_per_epoch = int(sum(int(task_stats[t]["steps_per_epoch"]) for t in task_order))
    if steps_per_epoch <= 0:
        raise RuntimeError("steps_per_epoch is 0 after task split.")

    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=float(args.lr), weight_decay=float(args.weight_decay))

    # loader 不交给 accelerate 自动切分，避免在 1:3 分片后再次被二次切分。
    model, optimizer = accelerator.prepare(model, optimizer)

    total_epochs = int(math.ceil(float(args.epochs)))
    total_steps = int(math.ceil(float(args.epochs) * steps_per_epoch))
    if int(args.max_steps) > 0:
        total_steps = min(total_steps, int(args.max_steps))

    warmup_steps = int(float(args.warmup_ratio) * total_steps)
    lr_scheduler = get_scheduler(
        name="cosine",
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    optimizer_state_loaded = False
    scheduler_state_loaded = False
    if resume_optimizer_path is not None:
        optimizer.load_state_dict(torch.load(str(resume_optimizer_path), map_location="cpu"))
        optimizer_state_loaded = True
    if resume_scheduler_path is not None:
        lr_scheduler.load_state_dict(torch.load(str(resume_scheduler_path), map_location="cpu"))
        scheduler_state_loaded = True

    start_epoch = 0
    if resume_step > 0:
        if resume_step > total_steps:
            raise RuntimeError(
                f"resume step ({resume_step}) exceeds total_steps ({total_steps}). Check --epochs/--max-steps."
            )
        if resume_step % steps_per_epoch != 0:
            raise RuntimeError(
                "Resume step is not aligned to epoch boundary. "
                f"resume_step={resume_step}, steps_per_epoch={steps_per_epoch}."
            )
        start_epoch = int(resume_step // steps_per_epoch)

        if not scheduler_state_loaded:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                for _ in range(int(resume_step)):
                    lr_scheduler.step()

    if accelerator.is_main_process:
        (out_dir / "run_config.json").write_text(
            json.dumps(
                {
                    "run": {
                        "qwen_model_dir": _rel_to_project(qwen_model_dir),
                        "dinov3_dir": _rel_to_project(dinov3_dir),
                        "dataset_root": str(Path(args.dataset_root)),
                        "train_json": str(Path(args.train_json)),
                        "image_size": int(args.image_size),
                    },
                    "adapter": {
                        "dinov3_dir": _rel_to_project(dinov3_dir),
                        "image_size": int(adapter_cfg.image_size),
                        "merge_size": int(adapter_cfg.merge_size),
                        "deepstack_visual_indexes": list(adapter_cfg.deepstack_visual_indexes),
                    },
                    "train": {
                        "epochs": float(args.epochs),
                        "max_steps": int(args.max_steps),
                        "batch_size": int(args.batch_size),
                        "batch_size_per_rank": list(local_batch_sizes),
                        "batch_size_ratio": str(args.batch_size_ratio),
                        "local_batch_size": int(local_batch_size),
                        "global_micro_batch": int(global_micro_batch),
                        "global_batch": int(effective_global_batch),
                        "micro_batch_loss_average": "sample",
                        "ddp_loss_scale_mode": str(loss_scale_mode),
                        "sample_ddp_loss_scale": float(sample_ddp_loss_scale),
                        "sampling_strategy": "bucketed_drop_tail_per_task_global_batch",
                        "samples_per_rank_per_epoch": [int(x) for x in samples_per_rank_per_epoch],
                        "used_items_per_epoch": int(used_items_per_epoch),
                        "dropped_items_per_epoch": int(dropped_items_per_epoch),
                        "steps_per_epoch": int(steps_per_epoch),
                        "total_steps": int(total_steps),
                        "grad_accum": int(args.grad_accum),
                        "lr": float(args.lr),
                        "weight_decay": float(args.weight_decay),
                        "warmup_ratio": float(args.warmup_ratio),
                        "mixed_precision": str(args.mixed_precision),
                        "gradient_checkpointing": bool(args.gradient_checkpointing),
                        "num_workers": int(args.num_workers),
                        "save_strategy": "epoch",
                    },
                    "task_schedule": {
                        "mode": "sequential_fixed",
                        "order": list(task_order),
                        "lr_reset_between_tasks": False,
                        "lr_reset_between_epochs": False,
                        "average_within_task": "sample",
                    },
                    "tasks": task_stats,
                    "lora": {
                        "r": int(args.lora_r),
                        "alpha": int(args.lora_alpha),
                        "dropout": float(args.lora_dropout),
                        "target_leaf": lora_target_leaf,
                        "target_modules": lora_targets,
                    },
                    "init_merger": _rel_to_project(init_merger_path) if init_merger_path is not None else "",
                    "resume_lora": _rel_to_project(resume_lora_dir) if resume_lora_dir is not None else "",
                    "resume_optimizer": _rel_to_project(resume_optimizer_path) if resume_optimizer_path is not None else "",
                    "resume_scheduler": _rel_to_project(resume_scheduler_path) if resume_scheduler_path is not None else "",
                    "optimizer_state_loaded": bool(optimizer_state_loaded),
                    "scheduler_state_loaded": bool(scheduler_state_loaded),
                    "resume_step": int(resume_step),
                },
                ensure_ascii=False,
                indent=2,
                default=str,
            )
            + "\n",
            encoding="utf-8",
        )

        world_size_now = int(getattr(accelerator, "num_processes", 1))
        print(
            f"[INFO] items={len(items)} batch_per_rank={local_batch_sizes} grad_accum={int(args.grad_accum)} "
            f"world_size={world_size_now} "
            f"global_batch={int(global_micro_batch) * int(args.grad_accum)} "
            f"sampling=sequential_fixed_by_task_bucketed_drop_tail order={list(task_order)} "
            f"samples_per_rank={samples_per_rank_per_epoch} used_items={used_items_per_epoch} dropped_items={dropped_items_per_epoch} "
            f"steps/epoch={steps_per_epoch} total_steps={total_steps} start_epoch={start_epoch} resume_step={resume_step} save=epoch",
            flush=True,
        )
        for task_name in task_order:
            ts = task_stats[task_name]
            task_items = int(ts["items"])
            task_loader_lens = list(ts["loader_len_per_rank"])
            task_min_loader_len = int(ts["min_loader_len"])
            task_samples_per_rank = list(ts["samples_per_rank_per_epoch"])
            task_used_items = int(ts["used_items_per_epoch"])
            task_dropped_items = int(ts["dropped_items_per_epoch"])
            task_steps = int(ts["steps_per_epoch"])
            print(
                f"[INFO] task={task_name} items={task_items} "
                f"loader_len_per_rank={task_loader_lens} min_loader_len={task_min_loader_len} "
                f"samples_per_rank={task_samples_per_rank} used_items={task_used_items} "
                f"dropped_items={task_dropped_items} steps/epoch={task_steps}",
                flush=True,
            )

    model.train()
    optimizer.zero_grad(set_to_none=True)

    global_step = int(resume_step)
    accum = 0
    running_loss = 0.0

    for epoch in range(start_epoch, total_epochs):
        epoch_updates = 0

        for task_name in task_order:
            task_sampler = task_samplers[task_name]
            task_loader = task_loaders[task_name]
            task_loader_limit = int(task_stats[task_name]["min_loader_len"])

            if hasattr(task_sampler, "set_epoch"):
                try:
                    task_sampler.set_epoch(int(epoch))
                except Exception:
                    pass

            task_step = 0
            for batch_idx, batch in enumerate(task_loader):
                if int(batch_idx) >= int(task_loader_limit):
                    break

                batch.pop("meta", None)
                batch = _move_batch_to_device(batch, device=device)

                labels = batch["labels"]
                forward_batch = {k: v for k, v in batch.items() if k != "labels"}
                outputs = model(**forward_batch)
                # 这里显式按“每条样本先对自身有效 token 求平均，再对样本求平均”来算 loss。
                local_sample_mean_loss = causal_lm_sample_average_loss(outputs.logits, labels)
                loss = local_sample_mean_loss * float(sample_ddp_loss_scale)
                loss = loss / float(args.grad_accum)
                accelerator.backward(loss)

                running_loss += float(loss.detach().item())
                accum += 1

                if accum >= int(args.grad_accum):
                    if float(args.max_grad_norm) > 0:
                        accelerator.clip_grad_norm_(trainable, float(args.max_grad_norm))
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad(set_to_none=True)

                    global_step += 1
                    epoch_updates += 1
                    task_step += 1
                    accum = 0

                    if accelerator.is_main_process:
                        lr = float(lr_scheduler.get_last_lr()[0])
                        avg_loss = running_loss
                        running_loss = 0.0
                        print(
                            f"step={global_step} epoch={epoch + 1} task={task_name} task_step={task_step} lr={lr:.3e} loss={avg_loss:.4f}",
                            flush=True,
                        )

                    if global_step >= total_steps:
                        break

            # 丢弃未凑满 grad_accum 的尾部梯度，避免跨任务混在一起。
            if accum != 0:
                optimizer.zero_grad(set_to_none=True)
                accum = 0
                running_loss = 0.0

            if global_step >= total_steps:
                break

        completed_full_epoch = int(epoch_updates) == int(steps_per_epoch)
        if completed_full_epoch:
            accelerator.wait_for_everyone()
            if accelerator.is_main_process:
                epoch_dir = out_dir / f"epoch{epoch + 1}"
                epoch_dir.mkdir(parents=True, exist_ok=True)
                model_to_save = accelerator.unwrap_model(model)
                base_to_save = model_to_save.get_base_model()
                model_to_save.save_pretrained(str(epoch_dir / "lora"))
                torch.save(optimizer.state_dict(), str(epoch_dir / "optimizer.pt"))
                torch.save(lr_scheduler.state_dict(), str(epoch_dir / "scheduler.pt"))
                save_merger_safetensors(
                    base_to_save,
                    epoch_dir / "merger.safetensors",
                    extra={
                        "step": int(global_step),
                        "epoch": int(epoch + 1),
                        "run": run_meta,
                        "train_json": _rel_to_project(train_json),
                        "task_order": list(task_order),
                        "adapter": {
                            "dinov3_dir": _rel_to_project(dinov3_dir),
                            "image_size": int(adapter_cfg.image_size),
                            "merge_size": int(adapter_cfg.merge_size),
                            "deepstack_visual_indexes": list(adapter_cfg.deepstack_visual_indexes),
                        },
                    },
                )
                print(f"[OK] Saved epoch checkpoint: {epoch_dir}", flush=True)
        elif accelerator.is_main_process and int(epoch_updates) > 0:
            print(
                f"[WARN] Epoch {epoch + 1} ended early: updates={epoch_updates}/{steps_per_epoch}.",
                flush=True,
            )

        if global_step >= total_steps:
            break

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        if int(global_step) > 0 and int(global_step) % int(steps_per_epoch) == 0:
            final_dir = out_dir / "final"
            final_dir.mkdir(parents=True, exist_ok=True)
            model_to_save = accelerator.unwrap_model(model)
            base_to_save = model_to_save.get_base_model()
            model_to_save.save_pretrained(str(final_dir / "lora"))
            torch.save(optimizer.state_dict(), str(final_dir / "optimizer.pt"))
            torch.save(lr_scheduler.state_dict(), str(final_dir / "scheduler.pt"))
            save_merger_safetensors(
                base_to_save,
                final_dir / "merger.safetensors",
                extra={
                    "step": int(global_step),
                    "run": run_meta,
                    "train_json": _rel_to_project(train_json),
                    "adapter": {
                        "dinov3_dir": _rel_to_project(dinov3_dir),
                        "image_size": int(adapter_cfg.image_size),
                        "merge_size": int(adapter_cfg.merge_size),
                        "deepstack_visual_indexes": list(adapter_cfg.deepstack_visual_indexes),
                    },
                },
            )
            print(f"[OK] Saved: {final_dir}", flush=True)
        else:
            print(
                "[WARN] Skip saving final checkpoint because current step is not on an epoch boundary: "
                f"step={global_step}, steps_per_epoch={steps_per_epoch}.",
                flush=True,
            )

    accelerator.end_training()


if __name__ == "__main__":
    main()
