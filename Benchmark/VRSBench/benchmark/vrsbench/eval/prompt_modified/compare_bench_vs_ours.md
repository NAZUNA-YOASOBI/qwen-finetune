# VRSBench 指标对比（Bench vs Ours）
- 运行标签：`prompt_modified`。
- Caption 的 BLEU/METEOR/ROUGE_L/CIDEr 按 x100 展示。
- Grounding 的 Acc 指标直接来自官方口径 summary（单位：百分比点）。
- 当前 grounding 生成脚本使用严格 prompt，并按前 4 个数字解析 / 规范化。
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
| Ours | Ours-baseline8b | 24.86 | 11.17 | 5.07 | 2.43 | 15.75 | 20.97 | 2.20 | - | 84.53 |
| Ours | Ours-merger_only-epoch10-fixed256 | 36.09 | 21.29 | 12.81 | 7.99 | 20.22 | 30.34 | 13.81 | - | 65.21 |
| Ours | Ours-merger_lora-epoch10-fixed256 | 44.69 | 28.20 | 18.22 | 12.04 | 20.61 | 34.15 | 24.58 | - | 50.17 |
| Ours | Ours-merger_lora-epoch10-smartresize512 | 44.19 | 27.75 | 17.83 | 11.72 | 20.73 | 34.16 | 25.65 | - | 51.21 |
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
| Ours | Ours-baseline8b | 23.31 | 7.07 | 12.80 | 4.06 | 17.18 | 5.32 |
| Ours | Ours-merger_only-epoch10-fixed256 | 4.38 | 1.25 | 2.00 | 0.51 | 2.99 | 0.82 |
| Ours | Ours-merger_lora-epoch10-fixed256 | 63.73 | 30.89 | 55.51 | 32.59 | 58.94 | 31.88 |
| Ours | Ours-merger_lora-epoch10-smartresize512 | 65.69 | 32.21 | 56.34 | 31.33 | 60.24 | 31.70 |
| Ours | Ours-qwen_native-epoch10 | 72.77 | 40.69 | 68.45 | 45.98 | 70.25 | 43.78 |
