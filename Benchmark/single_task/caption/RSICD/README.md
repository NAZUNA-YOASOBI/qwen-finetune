# Caption / RSICD

这份目录用于集中整理 single-task caption 在 `RSICD` 数据集上的当前正式文件。

## 当前保留的五个对比模型

- `DINOv3`：`epoch2`
- `Qwen3-VL Native FT`：`epoch2`
- `Qwen3-VL-8B Base`
- `GeoChat-7B`
- `GeoGround`

## 目录说明

- `data/`
  - 训练、验证、测试所用的标注文件。
  - 不包含图片实体。
- `scripts/`
  - 训练入口、评测入口以及 caption 共用脚本。
- `results/`
  - 这五个模型当前选定结果对应的逐条 prediction、生成摘要和评测摘要。
- `table/`
  - 当前这组 caption 对比表。

## 数据口径

- 训练使用：
  - `dataset_train.json`
  - `dataset_val.json`
- 测试使用：
  - `dataset_rsicd.json`
- 标注中的 `image_id` / `filename` 就是原始 `RSICD` 图片文件名，可用于重新对应公开原图。
- 这里不复制图片实体，图片路径只在需要时由外部传入。

## 整理方式

- 这里所有内容都是独立副本，不使用符号链接。
- 标注和结果中的路径字段已经尽量改成仓库内相对路径或相对路径占位，便于单独整理到 GitHub。
