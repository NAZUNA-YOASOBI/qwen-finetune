# ftqwen3 脚本说明

本目录存放 `Qwen3-VL` 这一条线在 VRSBench 中用到的脚本，按功能分类。

## 目录结构

- `generate/`
  - 生成模型输出
  - 包括 baseline、DINOv3、Qwen native、SVA deepstack CA，以及对应 referring 生成脚本
- `fix/`
  - 处理生成时触发 `max_new_tokens` 后的补跑与修复
- `eval/`
  - Caption 与 referring 的算分脚本
- `report/`
  - paper 表格提取与最终 markdown 对比表生成脚本
- `prepare/`
  - 从原始数据集整理出 VRSBench 测试所需的标注与索引文件
- `utils/`
  - 通用工具脚本
  - 当前包括 jsonl 分片合并、按 key 打补丁、referring 预测规范化
- `run/`
  - 运行入口脚本
  - 用来串起完整评测流程或按 epoch 反向评测

## 常用入口

- `run/run_prompt_modified_suite.sh`
  - 当前 `prompt_modified` 这一套评测入口
  - 默认只产出并汇总 baseline、smartresize512、qwen_native 这三条结果
- `run/run_eval_reverse_epochs.sh`
  - `sva_deepstack_ca` 按 epoch 回溯评测入口
- `report/make_report_prompt_modified.py`
  - 生成 `compare_bench_vs_ours.md`
  - 如果要把 `sva` 或 `qwen3.5` 纳入总表，需要显式传 `--sva-dir` / `--qwen35-dir`

## 使用约定

- 所有脚本都默认以 `Benchmark/VRSBench` 为项目根目录来解析相对路径

## 源码分层

`src/ftqwen3/` 当前按模型线与共享模块拆分为：

- `shared/`：通用底层模块
- `baseline/`：原生未微调 baseline
- `qwen_native/`：原生 Qwen3-VL 视觉分支微调线
- `dinov3_merger/`：DINOv3 + merger 微调线
- `sva_deepstack_ca/`：SVA deepstack cross-attention 微调线

