# GeoBench-VLM 脚本说明

本目录按链路与脚本职责整理 `GeoBench-VLM` 的评测代码。

## 目录结构

- `legacy/`
  - `run/`：`Single / Temporal / Captioning` 入口脚本
  - `eval/`：`Single / Temporal / Captioning` 评测脚本
- `xgrammar/`
  - `run/`：`Ref-Det` 的 `xgrammar` 入口脚本
  - `eval/`：`Ref-Det` 的 `xgrammar` 评测脚本

## 源码分层

`src/` 当前按共享模块与链路拆分为：

- `shared/`：通用数据读写、路径解析、分片和预测键工具
- `legacy/`：`Single / Temporal / Captioning` 生成与模型运行代码
- `xgrammar/`：`Ref-Det` 的 `xgrammar` 生成与模型运行代码
