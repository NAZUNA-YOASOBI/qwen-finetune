# ftqwen35 脚本说明

本目录存放 `Qwen3.5` 这一条线在 VRSBench 中用到的脚本，按功能分类。

## 目录结构

- `generate/`
  - 生成 caption / referring 输出
  - 包括常规 baseline 与 githubstyle baseline
- `fix/`
  - 处理生成时触发 `max_new_tokens` 后的补跑与修复
- `eval/`
  - Caption 与 referring 的算分脚本
- `report/`
  - Qwen3.5 对比表生成脚本
- `utils/`
  - 通用工具脚本
  - 当前包括 jsonl 分片合并、按 key 打补丁
- `run/`
  - 运行入口脚本
  - 当前包含常规 baseline 与 githubstyle baseline 的完整运行脚本

## 常用入口

- `run/run_qwen35_vrsbench.sh`
  - Qwen3.5 baseline 评测入口
- `run/run_eval_qwen35_9b_baseline_githubstyle_20260315.sh`
  - Qwen3.5 githubstyle baseline 评测入口
- `report/make_report_qwen35_baselines.py`
  - 生成 Qwen3.5 对比表

## 使用约定

- 所有脚本都默认以 `Benchmark/VRSBench` 为项目根目录来解析相对路径
