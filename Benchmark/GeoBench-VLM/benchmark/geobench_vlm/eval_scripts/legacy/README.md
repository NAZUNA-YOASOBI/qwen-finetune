# legacy 链路

当前保留的 legacy 链路用于：

- `Single`
- `Temporal`
- `Captioning`

入口脚本：

- `run/run_geobench_vlm_qwen_baselines_20260319.sh`

说明：

- `Single` 和 `Temporal` 走选择题评测。
- `Captioning` 走 `BERTScore-F1`。
- 旧版 `Ref-Det` 评测脚本仍保留在 `eval/`，只作历史对照；当前最终保留结果不再使用这条链路。
