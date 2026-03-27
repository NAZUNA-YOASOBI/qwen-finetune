import argparse
import json
import logging
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import torch
from PIL import Image
from torch.utils.data import Dataset
from transformers import (
    AutoModelForImageTextToText,
    AutoProcessor,
    Trainer,
    TrainingArguments,
)

try:
    from peft import LoraConfig, get_peft_model
except Exception:
    LoraConfig = None
    get_peft_model = None


LOGGER = logging.getLogger("train_sft_vrsbench")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Few-shot SFT for VRSBench visual grounding."
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="Qwen/Qwen3.5-4B",
        help="HF model name or local checkpoint path.",
    )
    parser.add_argument(
        "--train_file",
        type=str,
        default="",
        help="Training JSONL file. If empty, it will be resolved from --shots.",
    )
    parser.add_argument(
        "--shots",
        type=int,
        default=1,
        help="Resolve data/VRSBench/RL_VRSBench_VG_full_{shots}shots_vlmr1_size_new_clip.jsonl when --train_file is empty.",
    )
    parser.add_argument(
        "--image_dir",
        type=str,
        default="data/Images_train",
        help="Directory that stores images referenced by the JSONL file.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/vrsbench_sft",
        help="Directory used for checkpoints and final model.",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=1024,
        help="Maximum sequence length.",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=2e-4,
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--num_train_epochs",
        type=float,
        default=3.0,
    )
    parser.add_argument(
        "--warmup_ratio",
        type=float,
        default=0.03,
    )
    parser.add_argument(
        "--logging_steps",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--save_steps",
        type=int,
        default=50,
    )
    parser.add_argument(
        "--save_total_limit",
        type=int,
        default=2,
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=2,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    parser.add_argument(
        "--lora",
        action="store_true",
        help="Enable LoRA if peft is available.",
    )
    parser.add_argument(
        "--lora_r",
        type=int,
        default=16,
    )
    parser.add_argument(
        "--lora_alpha",
        type=int,
        default=32,
    )
    parser.add_argument(
        "--lora_dropout",
        type=float,
        default=0.05,
    )
    parser.add_argument(
        "--lora_variant",
        type=str,
        default="v2",
        choices=["v1", "v2", "v3", "v4"],
        help="LoRA target preset: v1=language, v2=language+merger, v3=language+vision_mlp+merger, v4=language+vision_attn+vision_mlp+merger.",
    )
    parser.add_argument(
        "--extra_lora_modules",
        type=str,
        default="",
        help="Optional comma-separated extra LoRA target module suffixes on top of --lora_variant.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Enable gradient checkpointing.",
    )
    parser.add_argument(
        "--bf16",
        action="store_true",
        help="Force bf16 training.",
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Force fp16 training.",
    )
    parser.add_argument(
        "--trust_remote_code",
        action="store_true",
    )
    return parser.parse_args()


def resolve_lora_target_modules(model: Any, args: argparse.Namespace) -> List[str]:
    language_suffixes = (
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    )
    vision_attn_suffixes = (".attn.qkv", ".attn.proj")
    vision_mlp_suffixes = (".mlp.linear_fc1", ".mlp.linear_fc2")
    merger_names = (
        "model.visual.merger.linear_fc1",
        "model.visual.merger.linear_fc2",
    )

    groups: Dict[str, List[str]] = {
        "language": [],
        "vision_attn": [],
        "vision_mlp": [],
        "merger": [],
    }
    for name, module in model.named_modules():
        if not hasattr(module, "weight"):
            continue
        if name.startswith("model.visual.blocks.") and name.endswith(vision_attn_suffixes):
            groups["vision_attn"].append(name)
        elif name.startswith("model.visual.blocks.") and name.endswith(vision_mlp_suffixes):
            groups["vision_mlp"].append(name)
        elif name in merger_names:
            groups["merger"].append(name)
        elif not name.startswith("model.visual.") and name.endswith(language_suffixes):
            groups["language"].append(name)

    for key in groups:
        groups[key] = sorted(groups[key])

    if args.lora_variant == "v1":
        selected_group_names = ["language"]
    elif args.lora_variant == "v2":
        selected_group_names = ["language", "merger"]
    elif args.lora_variant == "v3":
        selected_group_names = ["language", "vision_mlp", "merger"]
    elif args.lora_variant == "v4":
        selected_group_names = ["language", "vision_attn", "vision_mlp", "merger"]
    else:
        raise ValueError(f"Unsupported lora_variant: {args.lora_variant}")

    modules: List[str] = []
    for group_name in selected_group_names:
        modules.extend(groups[group_name])

    LOGGER.info(
        "LoRA groups | language=%s vision_attn=%s vision_mlp=%s merger=%s | selected=%s",
        len(groups["language"]),
        len(groups["vision_attn"]),
        len(groups["vision_mlp"]),
        len(groups["merger"]),
        ",".join(selected_group_names),
    )

    if args.extra_lora_modules:
        extra_modules = [
            item.strip()
            for item in args.extra_lora_modules.split(",")
            if item.strip()
        ]
        for module_name in extra_modules:
            if module_name not in modules:
                modules.append(module_name)
    return modules


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_user_prompt(query: str) -> str:
    return (
        "Locate the region described by the query in the image.\n"
        "Return strict JSON only with this schema:\n"
        '{"bbox_2d":[x0,y0,x1,y1]}\n'
        "Use 0..1000 normalized coordinates.\n"
        "Do not output extra text.\n"
        f"Query: {query}"
    )


def clamp_bbox_2d_1000(bbox: List[float]) -> Dict[str, List[float]]:
    x0, y0, x1, y1 = bbox
    x0 = max(0.0, min(1000.0, x0))
    y0 = max(0.0, min(1000.0, y0))
    x1 = max(0.0, min(1000.0, x1))
    y1 = max(0.0, min(1000.0, y1))
    if x1 <= x0 or y1 <= y0:
        raise ValueError(f"Invalid bbox after normalization: {bbox!r}")
    return {"bbox_2d": [x0, y0, x1, y1]}


def normalize_target_bbox(raw_value: Any, width: int, height: int) -> Dict[str, List[float]]:
    if isinstance(raw_value, dict) and "bbox_2d" in raw_value:
        bbox = raw_value["bbox_2d"]
    else:
        bbox = raw_value

    if not isinstance(bbox, Sequence) or len(bbox) != 4:
        raise ValueError(f"Unexpected bbox format: {raw_value!r}")

    values = [float(v) for v in bbox]
    max_v = max(values)

    if max_v <= 100.0:
        return clamp_bbox_2d_1000([v * 10.0 for v in values])
    if max_v <= 1000.0 and (width > 1000 or height > 1000):
        return clamp_bbox_2d_1000(values)
    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid image size: width={width}, height={height}")

    x0, y0, x1, y1 = values
    return clamp_bbox_2d_1000(
        [
            x0 * 1000.0 / float(width),
            y0 * 1000.0 / float(height),
            x1 * 1000.0 / float(width),
            y1 * 1000.0 / float(height),
        ]
    )


@dataclass
class VRSExample:
    image_path: str
    prompt: str
    target: str


class VRSBenchSFTDataset(Dataset):
    def __init__(self, examples: List[VRSExample]) -> None:
        self.examples = examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> VRSExample:
        return self.examples[idx]


def load_examples(train_file: str, image_dir: str) -> List[VRSExample]:
    examples: List[VRSExample] = []
    train_path = Path(train_file)
    image_root = Path(image_dir)

    with train_path.open("r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            conversations = record.get("conversations", [])
            if len(conversations) < 2:
                LOGGER.warning("Skip line %s: missing conversation turns", line_idx)
                continue

            human_turn = conversations[0].get("value", "")
            gpt_turn = conversations[1].get("value")
            image_name = record.get("image")

            if not human_turn or image_name is None:
                LOGGER.warning("Skip line %s: missing image or query", line_idx)
                continue

            image_path = image_root / image_name
            if not image_path.exists():
                raise FileNotFoundError(f"Image not found: {image_path}")

            with Image.open(image_path) as image:
                width, height = image.size

            target = json.dumps(
                normalize_target_bbox(gpt_turn, width=width, height=height),
                ensure_ascii=False,
            )
            prompt = build_user_prompt(human_turn.replace("<image>", "").strip())
            examples.append(
                VRSExample(
                    image_path=str(image_path),
                    prompt=prompt,
                    target=target,
                )
            )

    if not examples:
        raise ValueError(f"No training examples loaded from {train_file}")

    LOGGER.info("Loaded %s training examples from %s", len(examples), train_file)
    LOGGER.info("First target example: %s", examples[0].target)
    return examples


def resolve_train_file(args: argparse.Namespace) -> str:
    if args.train_file:
        return args.train_file
    return f"data/VRSBench/RL_VRSBench_VG_full_{args.shots}shots_vlmr1_size_new_clip.jsonl"


def load_image(image_path: str) -> Image.Image:
    image = Image.open(image_path)
    if image.mode != "RGB":
        image = image.convert("RGB")
    return image


class VLMDataCollator:
    def __init__(self, processor: Any, max_length: int) -> None:
        self.processor = processor
        self.max_length = max_length
        self.tokenizer = processor.tokenizer

    def _build_messages(self, prompt: str, target: Optional[str]) -> List[Dict[str, Any]]:
        messages: List[Dict[str, Any]] = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        if target is not None:
            messages.append(
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": target}],
                }
            )
        return messages

    def _render_text(self, messages: List[Dict[str, Any]], add_generation_prompt: bool) -> str:
        return self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )

    def __call__(self, features: List[VRSExample]) -> Dict[str, torch.Tensor]:
        images = [load_image(feature.image_path) for feature in features]
        full_texts = [
            self._render_text(
                self._build_messages(feature.prompt, feature.target),
                add_generation_prompt=False,
            )
            for feature in features
        ]
        prompt_texts = [
            self._render_text(
                self._build_messages(feature.prompt, None),
                add_generation_prompt=True,
            )
            for feature in features
        ]

        batch = self.processor(
            images=images,
            text=full_texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        prompt_batch = self.processor(
            images=images,
            text=prompt_texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        labels = batch["input_ids"].clone()
        pad_token_id = self.tokenizer.pad_token_id
        if pad_token_id is not None:
            labels[labels == pad_token_id] = -100

        prompt_lengths = prompt_batch["attention_mask"].sum(dim=1).tolist()
        for row_idx, prompt_len in enumerate(prompt_lengths):
            labels[row_idx, : int(prompt_len)] = -100

        batch["labels"] = labels
        return batch


def maybe_enable_lora(model: Any, args: argparse.Namespace) -> Any:
    if not args.lora:
        return model
    if LoraConfig is None or get_peft_model is None:
        raise ImportError("peft is not installed but --lora was requested.")

    target_modules = resolve_lora_target_modules(model, args)
    LOGGER.info(
        "LoRA variant: %s | target modules: %s",
        args.lora_variant,
        ",".join(target_modules),
    )
    config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=target_modules,
    )
    model = get_peft_model(model, config)
    model.print_trainable_parameters()
    return model


def resolve_precision(args: argparse.Namespace) -> Dict[str, bool]:
    if args.bf16 and args.fp16:
        raise ValueError("Choose only one of --bf16 or --fp16.")

    if args.bf16:
        return {"bf16": True, "fp16": False}
    if args.fp16:
        return {"bf16": False, "fp16": True}

    bf16_supported = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    fp16_supported = torch.cuda.is_available()
    return {"bf16": bf16_supported, "fp16": (not bf16_supported and fp16_supported)}


def main() -> None:
    args = parse_args()
    setup_logging()
    set_seed(args.seed)

    train_file = resolve_train_file(args)
    examples = load_examples(train_file, args.image_dir)
    dataset = VRSBenchSFTDataset(examples)

    LOGGER.info("Loading processor from %s", args.model_name_or_path)
    processor = AutoProcessor.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=args.trust_remote_code,
    )

    LOGGER.info("Loading model from %s", args.model_name_or_path)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model_name_or_path,
        torch_dtype="auto",
        trust_remote_code=args.trust_remote_code,
    )

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        if hasattr(model, "config"):
            model.config.use_cache = False
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()

    if not args.lora:
        LOGGER.warning(
            "Full fine-tuning is enabled. This is likely to OOM on a 24GB GPU for %s. "
            "Use --lora unless you have substantially more memory.",
            args.model_name_or_path,
        )

    model = maybe_enable_lora(model, args)

    precision_flags = resolve_precision(args)
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        num_train_epochs=args.num_train_epochs,
        warmup_ratio=args.warmup_ratio,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        dataloader_num_workers=args.dataloader_num_workers,
        remove_unused_columns=False,
        report_to="none",
        optim="adamw_torch",
        lr_scheduler_type="cosine",
        seed=args.seed,
        **precision_flags,
    )

    collator = VLMDataCollator(processor=processor, max_length=args.max_length)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=collator,
    )

    LOGGER.info("Start training with %s samples", len(dataset))
    trainer.train()
    trainer.save_model(args.output_dir)
    processor.save_pretrained(args.output_dir)
    LOGGER.info("Training finished. Model and processor saved to %s", args.output_dir)


if __name__ == "__main__":
    main()
