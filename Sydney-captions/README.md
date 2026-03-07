# Sydney-captions

这个目录归档了 `Sydney-captions` 这个单独 bench 的数据划分、脚本和本次原生 `Qwen3-VL-8B-Instruct` zero-shot 结果。

## 目录结构

- `data/`：当前实验用到的 bench 标注或划分文件
- `paper/`：该 bench 对应的论文表格整理结果
- `scripts/`：该 bench 的运行与评测脚本
- `results/qwen3_vl_8b_instruct_zero_shot_native_default_20260307/predictions/`：逐样本预测
- `results/qwen3_vl_8b_instruct_zero_shot_native_default_20260307/reports/`：评测汇总和对比表

## 当前结果

- 预测文件：`results/qwen3_vl_8b_instruct_zero_shot_native_default_20260307/predictions/sydney_captions.jsonl`
- 评测文件：`results/qwen3_vl_8b_instruct_zero_shot_native_default_20260307/reports/evaluation_summary.json`
- 对比表：`results/qwen3_vl_8b_instruct_zero_shot_native_default_20260307/reports/comparison.md`

## 说明

- 这里保存的是 bench 级归档结果。
- 重新运行脚本时，底层调用的是项目根目录下原始 `RSGPT-Simbench/scripts/` 里的主脚本。
- 论文里的 `RSGPT (ours)` 是微调结果；当前归档的是原生 zero-shot 结果。
