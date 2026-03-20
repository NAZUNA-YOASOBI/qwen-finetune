# GeoBench-VLM 指标对比（Bench vs Ours）

- Bench 数据来源：
  - `paper/geobench_vlm_2411.19325.pdf` 的 Table 2、Table 3、Table 4
  - `GeoBench-VLM/GEO-Bench-VLM/README.md` 中公开的 temporal / referring detection 表格
- Ours 数据来源：
  - `benchmark/geobench_vlm/eval_results/01_qwen3vl_baseline_20260319_cuda1_default/*.json`
  - `benchmark/geobench_vlm/eval_results/01_qwen35_baseline_20260319_cuda1_default/*.json`
  - `benchmark/geobench_vlm/eval_results/02_qwen3vl_refdet_xgrammar_v6_pixel_20260320/ref_det_summary.json`
  - `benchmark/geobench_vlm/eval_results/02_qwen35_refdet_xgrammar_v6_pixel_20260320/ref_det_summary.json`
- `Ours-qwen3-vl-baseline` 对应运行标签：
  - `01_qwen3vl_baseline_20260319_cuda1_default`
  - `02_qwen3vl_refdet_xgrammar_v6_pixel_20260320`
- `Ours-qwen3.5-baseline` 对应运行标签：
  - `01_qwen35_baseline_20260319_cuda1_default`
  - `02_qwen35_refdet_xgrammar_v6_pixel_20260320`
- 论文里的 Bench 数值保持原文小数口径。
- 当前 Ours 的 `Single` 不是直接拿一个总分去对齐 Table 2，而是按论文公开分组重算：
  - `Event Detection` = `Fire Risk Assessment` + `Disaster Type Classification`
  - `Object Classification` = `Aircraft Type Classification` + `Ship Type Classification`
  - `Counting` = 各类单图 counting 子任务加权汇总
  - `Scene Understanding` = `Scene Classification` + `Land Use Classification` + `Crop Type Classification`
  - 以上都按题目数加权平均
- `Spatial Relation Classification` 虽然在当前 `Single` 结果里有分数，但论文 Table 2 没单列，所以这里不混进去。
- 当前 Ours 的 `Ref-Det` 对齐 Table 4 时使用的是 `micro_precision`，因为论文 Table 4 报的是 `Precision@0.5 / Precision@0.25`，不是我们 summary 里另外保留的 strict acc。

## Table 2：Geospatial task summary 对比
| Group | Method | Event Det | Object Cls | Counting | Scene Und | Image Cap |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Bench | GPT-4o | 0.4726 | 0.5863 | 0.3965 | 0.7114 | 0.6418 |
| Bench | EarthDial | 0.5418 | 0.4039 | 0.3626 | 0.7705 | 0.5378 |
| Bench | Qwen2-VL | 0.4640 | 0.4560 | 0.4019 | 0.6761 | 0.5895 |
| Bench | LLaVA-OneVision | 0.4063 | 0.4593 | 0.4377 | 0.6636 | 0.6317 |
| Bench | SkySenseGPT | 0.3458 | 0.3094 | 0.3119 | 0.6205 | 0.6416 |
| Bench | InternVL-2 | 0.3458 | 0.3062 | 0.3280 | 0.5727 | 0.5968 |
| Bench | GeoChat | 0.3372 | 0.3127 | 0.2922 | 0.6091 | 0.4395 |
| Bench | LHRS-Bot-Nova | 0.2594 | 0.2704 | 0.3286 | 0.6330 | 0.6275 |
| Bench | LLaVA-NeXT | 0.3170 | 0.3192 | 0.2737 | 0.5477 | 0.6293 |
| Bench | LLaVA-1.5 | 0.3228 | 0.3029 | 0.2618 | 0.5625 | 0.6346 |
| Bench | RS-LLaVA | 0.2795 | 0.3094 | 0.2534 | 0.5534 | 0.5604 |
| Bench | SPHINX | 0.2363 | 0.2052 | 0.1860 | 0.2170 | 0.6451 |
| Bench | Ferret | 0.1643 | 0.1173 | 0.1956 | 0.1261 | 0.5615 |
| Ours | Ours-qwen3-vl-baseline | 0.4225 | 0.4827 | 0.3128 | 0.6564 | 0.8750 |
| Ours | Ours-qwen3.5-baseline | 0.4161 | 0.4508 | 0.3333 | 0.6720 | 0.8704 |

## Table 3：Temporal geospatial tasks 对比
| Group | Method | Crop Cls | Damaged Bldg Cnt | Disaster Cls | Farm Pond CD | Land Use Cls |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Bench | EarthDial | 0.2182 | 0.4362 | 0.5727 | 0.2105 | 0.6623 |
| Bench | GPT-4o | 0.1818 | 0.5667 | 0.6300 | 0.1711 | 0.6525 |
| Bench | LLaVA-OneVision | 0.1455 | 0.4810 | 0.4537 | 0.1842 | 0.5869 |
| Bench | Qwen2-VL | 0.1091 | 0.5000 | 0.5991 | 0.1974 | 0.5967 |
| Ours | Ours-qwen3-vl-baseline | 0.1818 | 0.5562 | 0.5357 | 0.1553 | 0.6125 |
| Ours | Ours-qwen3.5-baseline | 0.1782 | 0.5684 | 0.5057 | 0.2053 | 0.6197 |

## Table 4：Referring expression detection 对比
| Group | Method | Prec@0.5 | Prec@0.25 |
| --- | --- | ---: | ---: |
| Bench | SPHINX | 0.3408 | 0.5289 |
| Bench | EarthDial | 0.2429 | 0.4139 |
| Bench | GeoChat | 0.1151 | 0.2100 |
| Bench | Ferret | 0.0943 | 0.2003 |
| Bench | Qwen2-VL | 0.1518 | 0.2524 |
| Bench | GPT-4o | 0.0087 | 0.0386 |
| Bench | LHRS-Nova | 0.0930 | 0.2423 |
| Bench | SkySenseGPT | 0.1082 | 0.3224 |
| Ours | Ours-qwen3-vl-baseline | 0.0073 | 0.0580 |
| Ours | Ours-qwen3.5-baseline | 0.0078 | 0.0570 |
