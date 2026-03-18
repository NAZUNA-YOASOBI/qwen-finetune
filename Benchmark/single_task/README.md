# single

单任务 bench 统一按任务类型分类存放。

## 目录结构

- `caption/`
  - `RSICD/`
  - `Sydney-captions/`
  - `UCM-captions/`
- `vqa/`
  - `RSVQA-HR/`
  - `RSVQA-LR/`

## 说明

- 顶层保留综合性大型 benchmark：
  - `GeoChat-Bench/`
  - `RSIEval/`
  - `VRSBench/`
  - `XLRS-Bench/`
- 其余单任务数据集统一收拢到 `single/` 下，避免和综合 benchmark 混放。
