from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch


@dataclass(frozen=True)
class DinoV3RunConfig:
    qwen_model_dir: Path
    dinov3_dir: Path
    image_size: int = 512


@dataclass(frozen=True)
class DinoResizeConfig:
    image_size: int
    smart_resize_min_pixels: int
    smart_resize_max_pixels: int
    mode: str


@dataclass(frozen=True)
class GenerateKwargsConfig:
    gen_kwargs: dict[str, Any]
    strategy: str


@dataclass(frozen=True)
class GeneratedSequenceSummary:
    token_ids: list[int]
    generated_token_count: int
    ended_by_eos: bool
    last_generated_token_id: int | None


def _coerce_optional_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except Exception:
        return None


def read_merger_meta(path: Path) -> dict[str, Any]:
    meta_path = Path(path).with_suffix('.json')
    if not meta_path.is_file():
        return {}
    try:
        meta = json.loads(meta_path.read_text(encoding='utf-8'))
    except Exception:
        return {}
    return meta if isinstance(meta, dict) else {}


def read_merger_run_meta(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    meta = read_merger_meta(Path(path))
    run_meta = meta.get('run')
    return run_meta if isinstance(run_meta, dict) else {}


def path_metadata_matches(expected: str | Path | None, actual: str | Path | None) -> bool:
    if expected is None or actual is None:
        return False
    expected_path = Path(expected)
    actual_path = Path(actual)
    if expected_path.exists() and actual_path.exists():
        try:
            return expected_path.resolve() == actual_path.resolve()
        except Exception:
            pass
    return expected_path.name == actual_path.name


def assert_path_metadata_matches(*, label: str, expected: str | Path | None, actual: str | Path | None) -> None:
    expected_str = str(expected or '').strip()
    actual_path = Path(actual) if actual is not None else Path('')
    if not expected_str:
        return
    if not path_metadata_matches(expected_str, actual_path):
        raise ValueError(
            f'{label} mismatches checkpoint metadata. expected={expected_str}, got={actual_path}'
        )


def assert_dino_runtime_matches_merger(
    *,
    qwen_model_dir: Path,
    dinov3_dir: Path,
    image_size: int,
    smart_resize_min_pixels: int | None,
    smart_resize_max_pixels: int | None,
    merger_ckpt: Path,
) -> DinoResizeConfig:
    run_meta = read_merger_run_meta(merger_ckpt)
    assert_path_metadata_matches(
        label='Qwen model dir',
        expected=run_meta.get('qwen_model_dir'),
        actual=qwen_model_dir,
    )
    assert_path_metadata_matches(
        label='DINOv3 dir',
        expected=run_meta.get('dinov3_dir'),
        actual=dinov3_dir,
    )
    return resolve_dino_resize_config(
        image_size=image_size,
        smart_resize_min_pixels=smart_resize_min_pixels,
        smart_resize_max_pixels=smart_resize_max_pixels,
        merger_ckpt=merger_ckpt,
    )


def resolve_dino_resize_config(
    *,
    image_size: int,
    smart_resize_min_pixels: int | None,
    smart_resize_max_pixels: int | None,
    merger_ckpt: Path | None = None,
) -> DinoResizeConfig:
    input_image_size = int(image_size)
    if input_image_size <= 0:
        raise ValueError(f'invalid image_size: {image_size}')

    run_meta = read_merger_run_meta(merger_ckpt)
    meta_image_size = _coerce_optional_int(run_meta.get('image_size'))
    meta_min_pixels = _coerce_optional_int(run_meta.get('smart_resize_min_pixels'))
    meta_max_pixels = _coerce_optional_int(run_meta.get('smart_resize_max_pixels'))

    if (smart_resize_min_pixels is None) ^ (smart_resize_max_pixels is None):
        raise ValueError('smart resize min/max must be both set or both unset')
    if (meta_min_pixels is None) ^ (meta_max_pixels is None):
        raise ValueError(
            'checkpoint metadata contains incomplete smart resize range: '
            f'min={meta_min_pixels}, max={meta_max_pixels}'
        )

    if meta_image_size is not None and input_image_size != int(meta_image_size):
        raise ValueError(
            'image_size mismatches checkpoint metadata. '
            f'expected={meta_image_size}, got={input_image_size}'
        )
    effective_image_size = int(meta_image_size) if meta_image_size is not None else int(input_image_size)

    if smart_resize_min_pixels is not None and smart_resize_max_pixels is not None:
        effective_min = int(smart_resize_min_pixels)
        effective_max = int(smart_resize_max_pixels)
    elif meta_min_pixels is not None and meta_max_pixels is not None:
        effective_min = int(meta_min_pixels)
        effective_max = int(meta_max_pixels)
    else:
        fixed_pixels = int(effective_image_size) * int(effective_image_size)
        effective_min = int(fixed_pixels)
        effective_max = int(fixed_pixels)

    if effective_min <= 0 or effective_max < effective_min:
        raise ValueError(
            'invalid smart resize range: '
            f'min={effective_min}, max={effective_max}'
        )

    if meta_min_pixels is not None and meta_max_pixels is not None:
        if int(effective_min) != int(meta_min_pixels) or int(effective_max) != int(meta_max_pixels):
            raise ValueError(
                'smart resize range mismatches checkpoint metadata. '
                f'expected=({meta_min_pixels}, {meta_max_pixels}), '
                f'got=({effective_min}, {effective_max})'
            )
    elif merger_ckpt is not None:
        fixed_pixels = int(effective_image_size) * int(effective_image_size)
        if int(effective_min) != int(fixed_pixels) or int(effective_max) != int(fixed_pixels):
            raise ValueError(
                'checkpoint metadata implies fixed-size preprocessing, but runtime resize range is different. '
                f'expected=({fixed_pixels}, {fixed_pixels}), got=({effective_min}, {effective_max})'
            )

    mode = 'fixed' if int(effective_min) == int(effective_max) == int(effective_image_size) * int(effective_image_size) else 'smart'
    return DinoResizeConfig(
        image_size=int(effective_image_size),
        smart_resize_min_pixels=int(effective_min),
        smart_resize_max_pixels=int(effective_max),
        mode=str(mode),
    )


def build_generate_kwargs(
    *,
    max_new_tokens: int,
    do_sample: bool | None = None,
    temperature: float | None = None,
    top_p: float | None = None,
    top_k: int | None = None,
    num_beams: int | None = None,
    repetition_penalty: float | None = None,
    no_repeat_ngram_size: int | None = None,
) -> GenerateKwargsConfig:
    gen_kwargs: dict[str, Any] = {
        'max_new_tokens': int(max_new_tokens),
    }
    explicit_fields: list[str] = []

    if do_sample is not None:
        gen_kwargs['do_sample'] = bool(do_sample)
        explicit_fields.append('do_sample')
    if temperature is not None:
        gen_kwargs['temperature'] = float(temperature)
        explicit_fields.append('temperature')
    if top_p is not None:
        gen_kwargs['top_p'] = float(top_p)
        explicit_fields.append('top_p')
    if top_k is not None:
        gen_kwargs['top_k'] = int(top_k)
        explicit_fields.append('top_k')
    if num_beams is not None:
        gen_kwargs['num_beams'] = int(num_beams)
        explicit_fields.append('num_beams')
    if repetition_penalty is not None:
        gen_kwargs['repetition_penalty'] = float(repetition_penalty)
        explicit_fields.append('repetition_penalty')
    if no_repeat_ngram_size is not None:
        gen_kwargs['no_repeat_ngram_size'] = int(no_repeat_ngram_size)
        explicit_fields.append('no_repeat_ngram_size')

    strategy = 'generation_config_default' if not explicit_fields else 'cli_override:' + ','.join(explicit_fields)
    return GenerateKwargsConfig(gen_kwargs=gen_kwargs, strategy=str(strategy))


def maybe_set_generation_seed(seed: int | None) -> None:
    if seed is None:
        return
    seed_value = int(seed)
    random.seed(seed_value)
    try:
        import numpy as np  # type: ignore

        np.random.seed(seed_value)
    except Exception:
        pass
    try:
        from transformers import set_seed  # type: ignore

        set_seed(seed_value)
        return
    except Exception:
        pass

    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)


def _normalize_token_ids(tokens: Any) -> list[int]:
    if tokens is None:
        return []
    if hasattr(tokens, "detach") and hasattr(tokens, "cpu"):
        values = tokens.detach().cpu().tolist()
    elif hasattr(tokens, "tolist"):
        values = tokens.tolist()
    else:
        values = list(tokens)
    return [int(x) for x in values]


def _normalize_token_id_set(token_ids: Any) -> set[int]:
    if token_ids is None:
        return set()
    if isinstance(token_ids, (list, tuple, set)):
        out: set[int] = set()
        for token_id in token_ids:
            maybe_token_id = _coerce_optional_int(token_id)
            if maybe_token_id is not None:
                out.add(int(maybe_token_id))
        return out
    maybe_token_id = _coerce_optional_int(token_ids)
    return {int(maybe_token_id)} if maybe_token_id is not None else set()


def summarize_generated_sequences(
    generated_ids: Any,
    prompt_lens: list[int] | tuple[int, ...],
    *,
    eos_token_id: Any,
    pad_token_id: Any,
) -> list[GeneratedSequenceSummary]:
    prompt_lens_list = [int(x) for x in prompt_lens]
    batch_size = int(generated_ids.shape[0]) if hasattr(generated_ids, "shape") else len(generated_ids)
    if int(batch_size) != int(len(prompt_lens_list)):
        raise ValueError(
            "generated_ids and prompt_lens size mismatch: "
            f"generated={batch_size}, prompt_lens={len(prompt_lens_list)}"
        )

    eos_ids = _normalize_token_id_set(eos_token_id)
    pad_id = _coerce_optional_int(pad_token_id)
    summaries: list[GeneratedSequenceSummary] = []

    for row_ids, prompt_len in zip(generated_ids, prompt_lens_list):
        full_ids = _normalize_token_ids(row_ids)
        effective_ids = full_ids[int(prompt_len) :] if int(prompt_len) < len(full_ids) else []

        if pad_id is not None and (not eos_ids or int(pad_id) not in eos_ids):
            try:
                first_pad = effective_ids.index(int(pad_id))
                effective_ids = effective_ids[:first_pad]
            except ValueError:
                pass

        if eos_ids:
            first_eos: int | None = None
            for idx, token_id in enumerate(effective_ids):
                if int(token_id) in eos_ids:
                    first_eos = int(idx)
                    break
            if first_eos is not None:
                effective_ids = effective_ids[: first_eos + 1]

        ended_by_eos = bool(effective_ids) and bool(eos_ids) and int(effective_ids[-1]) in eos_ids
        last_generated_token_id = int(effective_ids[-1]) if effective_ids else None
        summaries.append(
            GeneratedSequenceSummary(
                token_ids=[int(x) for x in effective_ids],
                generated_token_count=int(len(effective_ids)),
                ended_by_eos=bool(ended_by_eos),
                last_generated_token_id=last_generated_token_id,
            )
        )
    return summaries


def torch_dtype_from_str(dtype: str) -> torch.dtype | None:
    dtype = str(dtype).lower().strip()
    if dtype == 'auto':
        return None
    if dtype in {'fp16', 'float16'}:
        return torch.float16
    if dtype in {'bf16', 'bfloat16'}:
        return torch.bfloat16
    if dtype in {'fp32', 'float32'}:
        return torch.float32
    return None


def load_merger_safetensors(model, path: Path) -> None:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f'Missing merger checkpoint: {path}')

    try:
        from safetensors.torch import load_file  # type: ignore
    except Exception as e:
        raise RuntimeError('Missing dependency: safetensors') from e

    state = load_file(str(path))

    merger_state = {k[len('merger.') :]: v for k, v in state.items() if k.startswith('merger.')}
    deepstack_state = {k[len('deepstack_merger_list.') :]: v for k, v in state.items() if k.startswith('deepstack_merger_list.')}
    input_proj_state = {k[len('input_proj.') :]: v for k, v in state.items() if k.startswith('input_proj.')}

    model.model.visual.merger.load_state_dict(merger_state, strict=True)
    if hasattr(model.model.visual, 'deepstack_merger_list') and model.model.visual.deepstack_merger_list is not None:
        model.model.visual.deepstack_merger_list.load_state_dict(deepstack_state, strict=True)
    if hasattr(model.model.visual, 'input_proj') and getattr(model.model.visual, 'input_proj') is not None:
        if not input_proj_state:
            raise ValueError(f'Missing input_proj.* weights in merger checkpoint: {path}')
        model.model.visual.input_proj.load_state_dict(input_proj_state, strict=True)


def save_merger_safetensors(model, path: Path, *, extra: dict[str, Any] | None = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    try:
        from safetensors.torch import save_file  # type: ignore
    except Exception as e:
        raise RuntimeError('Missing dependency: safetensors') from e

    state: dict[str, torch.Tensor] = {}
    for k, v in model.model.visual.merger.state_dict().items():
        state[f'merger.{k}'] = v.detach().cpu()
    if hasattr(model.model.visual, 'deepstack_merger_list') and model.model.visual.deepstack_merger_list is not None:
        for k, v in model.model.visual.deepstack_merger_list.state_dict().items():
            state[f'deepstack_merger_list.{k}'] = v.detach().cpu()
    if hasattr(model.model.visual, 'input_proj') and getattr(model.model.visual, 'input_proj') is not None:
        for k, v in model.model.visual.input_proj.state_dict().items():
            state[f'input_proj.{k}'] = v.detach().cpu()

    save_file(state, str(path))

    if extra is not None:
        meta_path = path.with_suffix('.json')
        meta_path.write_text(json.dumps(extra, ensure_ascii=False, indent=2, default=str) + '\n', encoding='utf-8')


def attach_dinov3_adapter(
    *,
    model,
    dinov3_dir: Path,
    image_size: int,
) -> Any:
    from transformers import AutoImageProcessor

    from ftqwen3.dinov3_merger.dinov3_adapter import DinoV3AdapterConfig, DinoV3VisualAdapter

    old_visual = model.model.visual
    cfg = DinoV3AdapterConfig(
        dinov3_dir=Path(dinov3_dir),
        image_size=int(image_size),
        merge_size=int(old_visual.spatial_merge_size),
        deepstack_visual_indexes=tuple(int(x) for x in getattr(old_visual, 'deepstack_visual_indexes', (5, 11, 17))),
        qwen_vision_depth=int(getattr(getattr(old_visual, 'config', None), 'depth', 0) or len(getattr(old_visual, 'blocks', []))),
    )
    adapter = DinoV3VisualAdapter(
        cfg,
        merger=old_visual.merger,
        deepstack_merger_list=getattr(old_visual, 'deepstack_merger_list', None),
        torch_dtype=model.dtype,
    )
    adapter = adapter.to(model.device)
    model.model.visual = adapter

    image_processor = AutoImageProcessor.from_pretrained(str(Path(dinov3_dir)))
    return image_processor
