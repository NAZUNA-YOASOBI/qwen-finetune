# VRSBench 测试代码与结果整理

本目录用于整理当前保留的 VRSBench 评测代码与最终结果文档，便于后续上传到团队 GitHub 仓库。

## 收录范围

### 五条模型线

1. `01_baseline_qwen3vl8b`
   - 原生 `Qwen3-VL-8B-Instruct`
2. `04_merger_lora_epoch10_smartresize512_sampleavg`
   - `DINOv3 + merger + LoRA + smart resize`
   - checkpoint: `checkpoints/vrsbench_joint/merger_lora_8b_dinov3_micro8_8_ga2_effective32_taskseq_smartbucket_sampleavg_wd001_run_20260308_025747/epoch10/merger.safetensors`
   - lora: `checkpoints/vrsbench_joint/merger_lora_8b_dinov3_micro8_8_ga2_effective32_taskseq_smartbucket_sampleavg_wd001_run_20260308_025747/epoch10/lora`
3. `05_qwen_native_epoch10`
   - 原生 `Qwen3-VL` 视觉编码器 + merger + LoRA
   - checkpoint: `checkpoints/vrsbench_joint/merger_lora_8b_qwen_native_micro8_8_ga2_effective32_wd001_taskseq_run_20260302_160151/epoch10/merger.safetensors`
   - lora: `checkpoints/vrsbench_joint/merger_lora_8b_qwen_native_micro8_8_ga2_effective32_wd001_taskseq_run_20260302_160151/epoch10/lora`
4. `06_sva_deepstack_ca_epoch10`
   - `Qwen3-VL native + DINOv3 + SVA deepstack cross-attention`
   - checkpoint: `checkpoints/vrsbench_joint/merger_lora_8b_sva_deepstack_ca_micro8_8_ga2_effective32_taskseq_sampleavg_wd001_run_20260313_083153/epoch10/merger.safetensors`
   - lora: `checkpoints/vrsbench_joint/merger_lora_8b_sva_deepstack_ca_micro8_8_ga2_effective32_taskseq_sampleavg_wd001_run_20260313_083153/epoch10/lora`
5. `07_qwen35_9b_baseline`
   - 原生 `Qwen3.5-9B`

### 已复制内容

- `benchmark/vrsbench/eval_scripts/ftqwen3/`
  - `generate/` 生成脚本
  - `fix/` 补跑与修复脚本
  - `eval/` 算分脚本
  - `report/` 汇总与表格脚本
  - `prepare/` 数据准备脚本
  - `utils/` 通用工具脚本
  - `run/` 运行入口脚本
- `benchmark/vrsbench/eval_scripts/ftqwen35/`
  - `generate/` 生成脚本
  - `fix/` 补跑与修复脚本
  - `eval/` 算分脚本
  - `report/` 汇总脚本
  - `utils/` 通用工具脚本
  - `run/` 运行入口脚本
- `src/ftqwen3/`
  - 按 `shared/`、`baseline/`、`qwen_native/`、`dinov3_merger/`、`sva_deepstack_ca/` 分层整理后的 `Qwen3-VL` 相关最小运行模块
- `src/ftqwen35/`
  - 按 `shared/`、`baseline/` 分层整理后的 `Qwen3.5` 相关最小运行模块
- `benchmark/vrsbench/data/`
  - 评测和生成用到的小型测试索引与引用文件
- `benchmark/vrsbench/paper/`
  - paper 对比表原始数据
- `benchmark/vrsbench/eval_results/prompt_modified/`
  - 最终对比报告
  - 当前保留结果的 `caption_summary.json`
  - 当前保留结果的 `grounding_summary.json`
- `benchmark/vrsbench/outputs/`
  - `17_qwen8b_baseline_noftstyle_20260308/`
  - `18_qwen8b_merger_lora_epoch10_smartresize512_sampleavg_20260310/`
  - `19_qwen8b_sva_deepstack_ca_epoch10_20260315/`
  - `20_qwen35_9b_baseline_20260315/`

## 未复制内容

以下内容默认不放入本目录：

- 模型权重
- LoRA 权重
- DINOv3 权重
- 原始图片数据集
- 其他大体积原始预测文件 `*.jsonl`

这样做是为了控制 GitHub 仓库体积，同时避免把不必要的大文件混入版本管理。当前额外保留了四组 VRSBench 输出，分别用于 baseline、smartresize512、sva_deepstack_ca 和 qwen3.5 baseline 结果复核。

## 结果入口

最终总表：

- `benchmark/vrsbench/eval_results/prompt_modified/compare_bench_vs_ours.md`

每个模型的结果摘要：

- `benchmark/vrsbench/eval_results/prompt_modified/01_baseline_qwen3vl8b/`
- `benchmark/vrsbench/eval_results/prompt_modified/04_merger_lora_epoch10_smartresize512_sampleavg/`
- `benchmark/vrsbench/eval_results/prompt_modified/06_sva_deepstack_ca_epoch10/`
- `benchmark/vrsbench/eval_results/prompt_modified/07_qwen35_9b_baseline/`
