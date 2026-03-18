# GeoChat Benchmark Evaluation

本目录按 `VRSBench` 的组织方式整理 `GeoChat-Benchmark` 的评测链路。

当前已补的任务：
- Scene Classification
- VQA
- Region Captioning
- Referring

当前未补：
- Grounding Description

目录约定：
- `benchmark/shared/scripts/`：共享评测脚本
- `benchmark/qwen3vl/scripts/`：`Qwen3-VL` 生成脚本
- `benchmark/qwen35/scripts/`：`Qwen3.5` 生成脚本
- `benchmark/geochat/outputs/`：模型输出
- `benchmark/geochat/eval/`：评测汇总
- `benchmark/geochat/logs/`：运行日志

说明：
- 生成 prompt 目前先尽量贴近 `GeoChat` 官方公开脚本。
- 解码参数默认尽量交给模型自身 `generation_config`，脚本只默认设置 `max_new_tokens=256`。
- `LRBEN/HRBEN` 的公开 `GeoChat-Bench` 文件不带答案，因此 `VQA` 评测脚本需要额外提供原始数据集中的 ground-truth 文件。
- 当前本地 raw 数据中：
  - `LRBEN` 的 GT 与公开 `jsonl` 键对齐，可以正式打分。
  - `HRBEN` 实际对应 Zenodo 的 `test_phili` 这套 raw 文件，不是普通 `test`。
  - 总控脚本会先调用 `benchmark/shared/scripts/prepare_geochat_hrben_gt.py` 生成对齐后的 GT，再做评测。
- `Qwen3.5` 明确关闭思考模式：`enable_thinking=False`。
- `Referring` 当前只接受严格的坐标标签格式，不再回退抓答案中的任意数字。

当前可直接使用的总控脚本：
- `benchmark/qwen3vl/scripts/run_eval_qwen3vl_baseline.sh`
- `benchmark/qwen35/scripts/run_eval_qwen35_baseline.sh`
- 本地一键入口会自动加载：
  - `benchmark/geochat/data/local_paths.sh`

运行前需要准备的路径：
- `AID_IMAGE_ROOT`
- `UCMERCED_IMAGE_ROOT`
- `HRBEN_IMAGE_ROOT`
- `HRBEN_GT_FILE`
- `LRBEN_IMAGE_ROOT`
- `LRBEN_GT_FILE`
- `GEOCHAT_IMAGE_ROOT`

任务口径：
- Scene：按官方 `question_id.split('/')[0]` 计算分类准确率。
- VQA：默认做严格字符串归一化后精确匹配；`LRBEN` 默认排除 `count`。
- Region Caption：按论文表 10 计算 `ROUGE-1`、`ROUGE-L`、`METEOR`。
- Referring：按论文表 7 的 `acc@0.5` 方向实现，支持公开数据里 `ref` 和 `grounding` 两个子集；多目标样本采用公开 generalized referring scorer 同型的“全局贪心 IoU 匹配 + 严格 `F1=1` 成功判定”。

当前本机状态：
- 已有 `GeoChat-Bench` 转换后的 `jsonl`。
- `benchmark/geochat/data/local_paths.sh` 已补齐当前机器的默认路径。
