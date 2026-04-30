# VQA / RSVQA_HR_test1

这份目录用于集中整理 single-task VQA 在 `RSVQA-HR test1 10%` 上的当前正式材料。

## 当前保留的五个对比模型

- `DINOv3`：`epoch4`
- `Qwen3-VL Native FT`：`epoch3`
- `Qwen3-VL-8B Base`
- `GeoChat-7B`
- `GeoGround`

## 目录说明

- `data/`
  - 固定的训练、验证和 `test1` 评测标注文件。
  - 不包含图片实体。
- `scripts/data/`
  - 当前这份副本对应的固定切片准备脚本。
- `scripts/train/`
  - 两条可训练链路的训练代码和 `RSVQA-HR` 专用训练入口。
- `scripts/eval/`
  - 五个模型的 `test1` 评测入口，以及 VQA 共用评测脚本。
- `results/`
  - 当前表格使用的五个模型结果。
- `table/`
  - 当前这组 VQA 对比表。

## 数据口径

- 当前副本保留三组固定标注：
  - `train 20%`
  - `val 20%`
  - `test1 10%`
- `data/dataset_info.json` 记录了这三组固定切片的来源、采样比例和逐类型数量。
- 标注中的真实图片索引主键是 `img_id`。
- `images` 元数据里还保留了 `original_name`，可用于重新对应公开原图。
- 这里不复制图片文件，图片路径只在需要时由外部传入。

## 结果口径说明

- `Qwen3-VL-8B Base`、`GeoChat-7B`、`GeoGround` 这里保留的是它们现成的 `test1` 正式结果副本。
- `DINOv3 epoch4` 和 `Qwen3-VL Native FT epoch3` 已重新整理为当前固定 `test1 10%` 标注上的本地评测结果。
- 因此五个模型当前统一使用同一份 `test1` 评测集，样本数均为 `13146`。

## 整理方式

- 这里所有内容都是复制后的副本，不是符号链接。
- 原始 `VRSBench` 目录下的文件没有被剪切或改写。
- 对其他非当前口径内容的清理，仅发生在这份副本内部。
- 标注和结果中的路径字段已经尽量改成仓库内相对路径或相对路径占位，便于单独整理到 GitHub。
