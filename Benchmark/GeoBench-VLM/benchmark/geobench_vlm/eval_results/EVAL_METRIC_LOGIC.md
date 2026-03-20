# GeoBench-VLM 当前评测口径

## 1. Single / Temporal

- 任务类型：选择题
- 生成链路：`src/legacy/`
- prompt 版本：
  - `Single`: `official_mcq_single_v1`
  - `Temporal`: `official_mcq_temporal_all_frames_v1`
- 评测脚本：`benchmark/geobench_vlm/eval_scripts/legacy/eval/eval_geobenchvlm_mcq.py`
- 指标：`accuracy`
- 当前 summary 中的 `overall_accuracy` 按全部 prompt 的正确率统计；由于每条样本 prompt 数一致，可直接作为总分使用。

## 2. Captioning

- 生成链路：`src/legacy/`
- prompt 版本：`dataset_caption_prompt_v1`
- prompt 额外限制：`Write around 140 words.`
- 评测脚本：`benchmark/geobench_vlm/eval_scripts/legacy/eval/eval_geobenchvlm_captioning.py`
- 指标：`BERTScore-F1`
- 汇总口径：逐 prompt 算分，再对同一问题的多个 prompt 取平均。

## 3. Ref-Det

- 当前最终保留链路：`src/xgrammar/`
- prompt 版本：`bbox2d_pixel_polygon4_json_array_v9_xgrammar_countlocked`
- 输出：严格 JSON array，`bbox_2d` 为四点 polygon
- 评测脚本：`benchmark/geobench_vlm/eval_scripts/xgrammar/eval/eval_geobenchvlm_refdet.py`
- 几何口径：polygon IoU
- 匹配：最大二分匹配
- 汇总：
  - `mean_f1`：逐样本 F1 平均
  - `acc`：逐样本 `sample_F1 = 1` 才记为正确

## 4. 当前保留结果目录

- `01_qwen3vl_baseline_20260319_cuda1_default`
- `01_qwen35_baseline_20260319_cuda1_default`
- `02_qwen3vl_refdet_xgrammar_v6_pixel_20260320`
- `02_qwen35_refdet_xgrammar_v6_pixel_20260320`

