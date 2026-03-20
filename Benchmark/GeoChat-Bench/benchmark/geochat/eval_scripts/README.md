# GeoChat 评测脚本说明

本目录按模型线与脚本职责整理 `GeoChat-Bench` 的评测代码。

## 目录结构

- `shared/`
  - `eval/`：算分脚本
  - `fix/`：补跑与截断修复脚本
  - `prepare/`：评测前的数据准备脚本
  - `utils/`：通用工具脚本
- `qwen3vl/`
  - `generate/`：`Qwen3-VL` 生成脚本
  - `run/`：`Qwen3-VL` 评测入口脚本
- `qwen35/`
  - `generate/`：`Qwen3.5` 生成脚本
  - `run/`：`Qwen3.5` 评测入口脚本

## 源码分层

`src/` 当前按共享模块与模型线拆分为：

- `shared/`：通用数据读写、生成调度、评测完整性检查
- `qwen3vl/`：`Qwen3-VL` 运行器
- `qwen35/`：`Qwen3.5` 运行器
