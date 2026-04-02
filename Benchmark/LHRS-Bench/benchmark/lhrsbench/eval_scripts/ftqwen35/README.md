# ftqwen35 脚本说明

本目录存放 `Qwen3.5` 这一条线在 LHRS-Bench 中用到的脚本，按功能分类。

## 目录结构

- `generate/`
  - 生成 Qwen3.5 baseline 输出
- `run/`
  - 完整运行入口
  - 当前包含 Qwen3.5-9B baseline 的 prepare + generate + eval + report 链路

## 使用约定

- 所有脚本都默认以 `Benchmark/LHRS-Bench` 为项目根目录来解析相对路径
- prepare / eval / report 这类 benchmark 公共脚本复用 `ftqwen3/` 下对应目录
- 当前正式链路的多选题 prompt 由 `ftqwen3/prepare/prepare_lhrsbench_qa.py` 统一生成
- 当前有效结果目录：
  - `benchmark/lhrsbench/outputs/01_qwen35_9b_baseline_20260402/`
  - `benchmark/lhrsbench/eval_results/01_qwen35_9b_baseline_20260402/`
- 默认 Python 环境：
  - `/opt/data/private/YanZiXi/home/yzx/miniconda3/envs/qwen3.5-dinov3/bin/python`
