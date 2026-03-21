# GeoChat 指标对比（Bench vs Ours）

- Bench 数据来源：`GeoChat_CVPR2024_paper.pdf` 的 Table 5、6、7、8、10。
- Ours 数据来源：
  - `benchmark/geochat/eval/01_qwen3vl_baseline_20260317_cuda0_official100/*.json`
  - `benchmark/geochat/eval/01_qwen35_baseline_20260317_dualgpu_bs64/*.json`
- `Ours-qwen3-vl-baseline` 对应运行标签：`01_qwen3vl_baseline_20260317_cuda0_official100`
- `Ours-qwen3.5-baseline` 对应运行标签：`01_qwen35_baseline_20260317_dualgpu_bs64`
- Ours 结果按当前公开 `GeoChat-Bench` 数据文件及其对应标注文件计算；只要题目文件和标注文件严格对齐，这套评测就是有效的。
- 论文里的 Bench 数值保留原文写法，这里主要用于横向参考。
- Scene、VQA、Referring 的数值单位都是百分比点。
- Region Caption 的 `ROUGE/METEOR` 按 x100 展示，和论文表保持同一量纲。
- 当前公开 `region_captioning.jsonl` 共 `2797` 条，当前 caption 结果也是按这 `2797` 条公开标注算出来的。
- 当前公开 `HRBEN` 评测链路共 `62554` 条，当前 HRBEN 结果也是按这 `62554` 条公开标注算出来的。

## Table 5：Scene Classification 对比
| Group | Method | UCMerced | AID |
| --- | --- | --- | --- |
| Bench | Qwen-VL | 62.90 | 52.60 |
| Bench | MiniGPTv2 | 4.76 | 12.90 |
| Bench | LLaVA-1.5 | 68.00 | 51.00 |
| Bench | GeoChat | 84.43 | 72.03 |
| Ours | Ours-qwen3-vl-baseline | 80.62 | 68.10 |
| Ours | Ours-qwen3.5-baseline | 72.10 | 69.53 |

## Table 6：VQA on RSVQA-LRBEN 对比
| Group | Method | Presence | Comparison | Rural/Urban | Avg. Accuracy |
| --- | --- | --- | --- | --- | --- |
| Bench-ZS | LLaVA-1.5 | 55.46 | 68.20 | 59.00 | 62.77 |
| Bench-ZS | Qwen-VL-Chat | 38.57 | 67.59 | 61.00 | 55.35 |
| Bench-ZS | MiniGPTv2 | 55.16 | 55.22 | 39.00 | 54.96 |
| Bench-RS | RSVQA | 87.47 | 81.50 | 90.00 | 86.32 |
| Bench-RS | EasyToHard | 90.66 | 87.49 | 91.67 | 89.94 |
| Bench-RS | Bi-Modal | 91.06 | 91.16 | 92.66 | 91.63 |
| Bench-RS | SHRNet | 91.03 | 90.48 | 94.00 | 91.84 |
| Bench-RS | RSGPT | 91.17 | 91.70 | 94.00 | 92.29 |
| Bench | GeoChat | 91.09 | 90.33 | 94.00 | 90.70 |
| Ours | Ours-qwen3-vl-baseline | 67.21 | 74.06 | 72.00 | 71.16 |
| Ours | Ours-qwen3.5-baseline | 73.10 | 75.19 | 62.00 | 74.12 |

## Table 7：Referring / Grounding 对比
| Group | Method | Small | Medium | Large | Single-object grounding | Multi-object grounding | [refer] | [grounding] | Overall |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Bench | MiniGPTv2 | 1.70 | 9.90 | 21.90 | 9.10 | 3.60 | 8.20 | 2.60 | 7.60 |
| Bench | GeoChat | 2.90 | 13.60 | 21.70 | 16.00 | 4.30 | 10.50 | 11.80 | 10.60 |
| Ours | Ours-qwen3-vl-baseline | 36.74 | 47.32 | 64.19 | 52.62 | 27.77 | 47.22 | 38.75 | 46.60 |
| Ours | Ours-qwen3.5-baseline | 28.35 | 29.54 | 49.97 | 39.16 | 13.91 | 34.30 | 17.32 | 33.04 |

## Table 8：VQA on RSVQA-HRBEN 对比
| Group | Method | Presence | Comparison | Average Accuracy |
| --- | --- | --- | --- | --- |
| Bench | Qwen-VL | 66.44 | 60.41 | 63.06 |
| Bench | LLaVA-1.5 | 69.83 | 67.29 | 68.40 |
| Bench | MiniGPTv2 | 40.79 | 50.91 | 46.46 |
| Bench | GeoChat | 58.45 | 83.19 | 72.30 |
| Ours | Ours-qwen3-vl-baseline | 66.30 | 76.10 | 71.79 |
| Ours | Ours-qwen3.5-baseline | 62.42 | 70.44 | 66.91 |

## Table 10：Region Caption 对比
| Group | Method | ROUGE-1 | ROUGE-L | METEOR |
| --- | --- | --- | --- | --- |
| Bench | MiniGPTv2 | 32.10 | 31.20 | 10.00 |
| Bench | GeoChat | 87.30 | 87.20 | 83.90 |
| Ours | Ours-qwen3-vl-baseline | 5.96 | 5.92 | 2.17 |
| Ours | Ours-qwen3.5-baseline | 6.94 | 5.68 | 2.64 |
