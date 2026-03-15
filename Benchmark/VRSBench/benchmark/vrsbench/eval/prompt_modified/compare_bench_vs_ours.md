# VRSBench 指标对比（Bench vs Ours）
- 运行标签：`prompt_modified`。
- Caption 的 BLEU/METEOR/ROUGE_L/CIDEr 按 x100 展示。
- Grounding 的 Acc 指标直接来自各自 summary（单位：百分比点）。
- 当前 grounding 生成脚本按各自输出协议做解析与规范化。
- Bench 数据来源：`benchmark/vrsbench/paper/table3_caption_paper.json` 与 `benchmark/vrsbench/paper/table4_grounding_paper.json`。

## Caption 对比
| Group | Method | BLEU-1 | BLEU-2 | BLEU-3 | BLEU-4 | METEOR | ROUGE_L | CIDEr | CHAIR | Avg_L |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Bench | GeoChat w/o ft | 13.90 | 6.60 | 3.00 | 1.40 | 7.80 | 13.20 | 0.40 | 0.42 | 36.00 |
| Bench | GPT-4V | 37.20 | 22.50 | 13.70 | 8.60 | 20.90 | 30.10 | 19.10 | 0.83 | 67.00 |
| Bench | MiniGPT-v2 | 36.80 | 22.40 | 13.90 | 8.70 | 17.10 | 30.80 | 21.40 | 0.73 | 37.00 |
| Bench | LLaVA-1.5 | 48.10 | 31.50 | 21.20 | 14.70 | 21.90 | 36.90 | 33.90 | 0.78 | 49.00 |
| Bench | GeoChat | 46.70 | 30.20 | 20.10 | 13.80 | 21.10 | 35.20 | 28.20 | 0.77 | 52.00 |
| Bench | Mini-Gemini | 47.60 | 31.10 | 20.90 | 14.30 | 21.50 | 36.80 | 33.50 | 0.77 | 47.00 |
| Ours | Ours-baseline8b | 24.89 | 11.20 | 5.09 | 2.42 | 15.75 | 20.93 | 2.24 | - | 84.35 |
| Ours | Ours-merger_only-epoch10-fixed256 | 36.09 | 21.29 | 12.81 | 7.99 | 20.22 | 30.34 | 13.81 | - | 65.21 |
| Ours | Ours-merger_lora-epoch10-fixed256 | 44.69 | 28.20 | 18.22 | 12.04 | 20.61 | 34.15 | 24.58 | - | 50.17 |
| Ours | Ours-merger_lora-epoch10-smartresize512-sampleavg | 45.10 | 28.53 | 18.49 | 12.27 | 20.72 | 34.43 | 25.21 | - | 49.88 |
| Ours | Ours-sva_deepstack_ca-epoch10 | 45.35 | 28.56 | 18.40 | 12.15 | 20.36 | 34.18 | 25.28 | - | 48.81 |
| Ours | Ours-qwen_native-epoch10 | 45.14 | 28.76 | 18.69 | 12.41 | 21.39 | 34.97 | 27.35 | - | 51.45 |

## Grounding 对比
| Group | Method | Acc@0.5 (Unique) | Acc@0.7 (Unique) | Acc@0.5 (Non Unique) | Acc@0.7 (Non Unique) | Acc@0.5 (All) | Acc@0.7 (All) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Bench | GeoChat w/o ft | 20.70 | 5.40 | 7.30 | 1.70 | 12.90 | 3.20 |
| Bench | GPT-4V | 8.60 | 2.20 | 2.50 | 0.40 | 5.10 | 1.10 |
| Bench | MiniGPT-v2 | 40.70 | 18.90 | 32.40 | 15.20 | 35.80 | 16.80 |
| Bench | LLaVA-1.5 | 51.10 | 16.40 | 34.80 | 11.50 | 41.60 | 13.60 |
| Bench | GeoChat | 57.40 | 22.60 | 44.50 | 18.00 | 49.80 | 19.90 |
| Bench | Mini-Gemini | 41.10 | 9.60 | 22.30 | 4.90 | 30.10 | 6.80 |
| Ours | Ours-baseline8b | 63.72 | 43.45 | 52.63 | 33.14 | 57.25 | 37.44 |
| Ours | Ours-merger_only-epoch10-fixed256 | 4.38 | 1.25 | 2.00 | 0.51 | 2.99 | 0.82 |
| Ours | Ours-merger_lora-epoch10-fixed256 | 63.73 | 30.89 | 55.51 | 32.59 | 58.94 | 31.88 |
| Ours | Ours-merger_lora-epoch10-smartresize512-sampleavg | 80.37 | 60.20 | 68.40 | 48.85 | 73.39 | 53.58 |
| Ours | Ours-sva_deepstack_ca-epoch10 | 76.71 | 54.75 | 65.57 | 45.39 | 70.21 | 49.29 |
| Ours | Ours-qwen_native-epoch10 | 72.77 | 40.69 | 68.45 | 45.98 | 70.25 | 43.78 |
