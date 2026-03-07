# VRSBench 测试代码与结果整理

本目录用于整理 5 个 VRSBench 测试模型的必要评测代码与最终结果文档，便于后续上传到团队 GitHub 仓库。

## 收录范围

### 五个模型

1. `01_baseline_qwen3vl8b`
   - 原生 `Qwen3-VL-8B-Instruct`
2. `02_merger_only_epoch10_fixed256`
   - `DINOv3 + merger_only`
   - checkpoint: `checkpoints/vrsbench_joint/merger_only_8b_dinov3_micro8_24_ga1_effective32_taskseq_run_20260210_210858/epoch10/merger.safetensors`
3. `03_merger_lora_epoch10_fixed256`
   - `DINOv3 + merger + LoRA`
   - checkpoint: `checkpoints/vrsbench_joint/merger_lora_8b_dinov3_micro2_6_ga4_effective32_taskseq_resume_from_epoch5_20260208/epoch10/merger.safetensors`
   - lora: `checkpoints/vrsbench_joint/merger_lora_8b_dinov3_micro2_6_ga4_effective32_taskseq_resume_from_epoch5_20260208/epoch10/lora`
4. `04_merger_lora_epoch10_smartresize512`
   - `DINOv3 + merger + LoRA + smart resize`
   - checkpoint: `checkpoints/vrsbench_joint/merger_lora_8b_dinov3_micro8_8_ga2_effective32_taskseq_smartbucket_run_20260213_153823/epoch10/merger.safetensors`
   - lora: `checkpoints/vrsbench_joint/merger_lora_8b_dinov3_micro8_8_ga2_effective32_taskseq_smartbucket_run_20260213_153823/epoch10/lora`
5. `05_qwen_native_epoch10`
   - 原生 `Qwen3-VL` 视觉编码器 + merger + LoRA
   - checkpoint: `checkpoints/vrsbench_joint/merger_lora_8b_qwen_native_micro8_8_ga2_effective32_wd001_taskseq_run_20260302_160151/epoch10/merger.safetensors`
   - lora: `checkpoints/vrsbench_joint/merger_lora_8b_qwen_native_micro8_8_ga2_effective32_wd001_taskseq_run_20260302_160151/epoch10/lora`

### 已复制内容

- `benchmark/vrsbench/scripts/`
  - 这 5 个模型当前使用的生成、修复、评测、汇总脚本
- `src/ftqwen/`
  - 上述脚本依赖的最小运行模块
- `benchmark/vrsbench/data/`
  - 评测和生成用到的小型测试索引与引用文件
- `benchmark/vrsbench/paper/`
  - paper 对比表原始数据
- `benchmark/vrsbench/eval/prompt_modified/`
  - 最终对比报告
  - 每个模型的 `caption_summary.json`
  - 每个模型的 `grounding_summary.json`

## 未复制内容

以下内容刻意不放入本目录：

- 模型权重
- LoRA 权重
- DINOv3 权重
- 原始图片数据集
- 大体积原始预测文件 `*.jsonl`

这样做是为了控制 GitHub 仓库体积，同时避免把不必要的大文件混入版本管理。

## 结果入口

最终总表：

- `benchmark/vrsbench/eval/prompt_modified/compare_bench_vs_ours.md`

每个模型的结果摘要：

- `benchmark/vrsbench/eval/prompt_modified/01_baseline_qwen3vl8b/`
- `benchmark/vrsbench/eval/prompt_modified/02_merger_only_epoch10_fixed256/`
- `benchmark/vrsbench/eval/prompt_modified/03_merger_lora_epoch10_fixed256/`
- `benchmark/vrsbench/eval/prompt_modified/04_merger_lora_epoch10_smartresize512/`
- `benchmark/vrsbench/eval/prompt_modified/05_qwen_native_epoch10/`

## 说明

- 当前整理的是 `prompt_modified` 这一套最终口径。
- `02` 和 `03` 已重新按真正的 `fixed256` 重跑。
- 当前 `compare_bench_vs_ours.md` 已与该目录下摘要结果一致。
