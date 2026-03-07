# RSVQA-LR

这个目录归档了 `RSVQA-LR` 这个 VQA bench 的相关文件。

## 目录结构

- `data/`：`LR_split_test_*` 划分文件
- `paper/`：论文对比表
- `scripts/`：解压图像、运行 zero-shot、自动评测脚本
- `results/qwen3_vl_8b_instruct_zero_shot_native_default_20260307/predictions/`：逐样本预测
- `results/qwen3_vl_8b_instruct_zero_shot_native_default_20260307/reports/`：汇总评测与对比表

## 说明

- 重新运行脚本时，底层调用的是项目根目录下原始 `RSGPT-Simbench/scripts/` 里的主脚本。
