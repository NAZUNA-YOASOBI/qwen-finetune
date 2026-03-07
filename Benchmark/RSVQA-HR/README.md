# RSVQA-HR

这个目录归档了 `RSVQA-HR` 这个 VQA bench 的相关文件。

## 目录结构

- `data/`：当前实验实际使用的 `test` 与 `test_phili` 划分文件
- `paper/`：`test1` 和 `test2` 两张论文对比表
- `scripts/`：解压图像、运行 zero-shot、自动评测脚本
- `results/qwen3_vl_8b_instruct_zero_shot_native_default_20260307/predictions/`：`rsvqa_hr_test1.jsonl` 和 `rsvqa_hr_test2.jsonl`
- `results/qwen3_vl_8b_instruct_zero_shot_native_default_20260307/reports/`：汇总评测与对比表

## 说明

- `rsvqa_hr_test1` 对应 `USGS_split_test_*`
- `rsvqa_hr_test2` 对应 `USGS_split_test_phili_*`
- 重新运行脚本时，底层调用的是项目根目录下原始 `RSGPT-Simbench/scripts/` 里的主脚本。
