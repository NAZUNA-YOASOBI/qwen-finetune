# VRSBench 指标对比（Bench vs Ours）
- 运行标签：`17_qwen8b_baseline_noftstyle_20260308`。
- 当前文件记录原生 `Qwen3-VL-8B-Instruct` 在 `noftstyle` 链路下的基线结果。
- Caption 行来自本目录下的 `caption_summary.json`。
- Grounding 行来自本目录下的 `grounding_baseline_test_summary.json`。
- Caption 的 BLEU/METEOR/ROUGE_L/CIDEr 按 x100 展示。
- Grounding 的 Acc 指标按百分比点展示。
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
| Ours | Ours-baseline8b | 24.87 | 11.16 | 5.05 | 2.42 | 15.75 | 20.94 | 2.32 | - | 84.60 |

## Grounding 对比
| Group | Method | Acc@0.5 (Unique) | Acc@0.7 (Unique) | Acc@0.5 (Non Unique) | Acc@0.7 (Non Unique) | Acc@0.5 (All) | Acc@0.7 (All) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Bench | GeoChat w/o ft | 20.70 | 5.40 | 7.30 | 1.70 | 12.90 | 3.20 |
| Bench | GPT-4V | 8.60 | 2.20 | 2.50 | 0.40 | 5.10 | 1.10 |
| Bench | MiniGPT-v2 | 40.70 | 18.90 | 32.40 | 15.20 | 35.80 | 16.80 |
| Bench | LLaVA-1.5 | 51.10 | 16.40 | 34.80 | 11.50 | 41.60 | 13.60 |
| Bench | GeoChat | 57.40 | 22.60 | 44.50 | 18.00 | 49.80 | 19.90 |
| Bench | Mini-Gemini | 41.10 | 9.60 | 22.30 | 4.90 | 30.10 | 6.80 |
| Ours | Ours-baseline8b | 63.52 | 43.72 | 53.07 | 33.29 | 57.43 | 37.64 |
