# legacy 链路

当前保留的 legacy 链路用于：

- `Single`
- `Temporal`
- `Captioning`
- `Ref-Det`

入口脚本：

- `run/run_geobench_vlm_qwen_baselines_20260319.sh`
- `run/run_geobench_vlm_refdet_baselines_20260321.sh`

说明：

- `Single` 和 `Temporal` 走选择题评测。
- `Captioning` 走 `BERTScore-F1`。
- `Ref-Det` 使用单独的生成脚本 `src/legacy/geobenchvlm_refdet_generate.py`。
- `Ref-Det` 输出 `0~1000` 尺度的两点框 `bbox_2d: [x0, y0, x1, y1]`，评测时转成四点矩形 polygon 再与 GT polygon 算分。
