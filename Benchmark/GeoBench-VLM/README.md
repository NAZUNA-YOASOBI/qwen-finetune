# GeoBench-VLM 测试代码与结果整理

本目录整理当前保留的 GeoBench-VLM 评测代码、脚本和结果，目录组织参考 `Benchmark/VRSBench`，默认以 `fine-tune-qwen3-vl` 为项目根目录。

## 当前保留的两条结果线

1. `01_qwen3vl_baseline_20260319_cuda1_default`
   - 模型：原生 `Qwen3-VL-8B-Instruct`
   - 任务：`Single / Temporal / Captioning / Ref-Det`
   - 环境：`qwen3-dinov3`
2. `01_qwen35_baseline_20260319_cuda1_default`
   - 模型：原生 `Qwen3.5-9B`
   - 任务：`Single / Temporal / Captioning / Ref-Det`
   - 环境：`qwen3-dinov3.5`

## 已复制内容

- `src/`
  - `shared/`：共享的数据读写、路径解析、分片与预测键工具
  - `legacy/`：当前保留的生成与模型运行代码
- `benchmark/geobench_vlm/eval_scripts/`
  - `README.md`：脚本总览
  - `legacy/`：四个任务的运行脚本与评测脚本
- `benchmark/geobench_vlm/eval_results/`
  - 当前保留的 summary / details 结果
  - `EVAL_METRIC_LOGIC.md`
  - `compare_bench_vs_ours.md`
- `benchmark/geobench_vlm/outputs/`
  - 当前保留的原始预测 `jsonl`
- `benchmark/geobench_vlm/paper/`
  - 本地抽取的 paper 文本
- `papers/`
  - 论文 PDF

## 未复制内容

以下内容默认不放入本目录：

- 原始图片数据集
- 官方仓库镜像
- 模型权重
- 其他临时调试文件与冒烟文件

## 默认外部路径

- 数据集根目录：`../../../GeoBench-VLM/dataset/GEOBench-VLM`
- Qwen3-VL 模型：`../../../VRSBench/models/Qwen3-VL-8B-Instruct`
- Qwen3.5 模型：`../../../../fine-tune-qwen3.5/models/Qwen3.5-9B`

## 当前结果入口

- `benchmark/geobench_vlm/eval_results/compare_bench_vs_ours.md`
- `benchmark/geobench_vlm/eval_results/01_qwen3vl_baseline_20260319_cuda1_default/`
- `benchmark/geobench_vlm/eval_results/01_qwen35_baseline_20260319_cuda1_default/`

## 当前摘要分数

- `Qwen3-VL`
  - `Single Acc`: 0.493242
  - `Temporal Acc`: 0.533684
  - `Captioning BERTScore-F1`: 0.875048
  - `Ref-Det Acc@0.25 / Acc@0.50`: 0.468393 / 0.354912
- `Qwen3.5`
  - `Single Acc`: 0.509063
  - `Temporal Acc`: 0.540572
  - `Captioning BERTScore-F1`: 0.870362
  - `Ref-Det Acc@0.25 / Acc@0.50`: 0.440975 / 0.348819
