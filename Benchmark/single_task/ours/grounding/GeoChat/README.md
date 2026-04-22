# Grounding / GeoChat

这份目录用于集中整理 single-task grounding 在 `GeoChat` 数据集上的当前正式文件。

## 当前保留的五个对比模型

- `DINOv3`：`epoch4`
- `Qwen3-VL Native FT`：`epoch3`
- `Qwen3-VL-8B Base`
- `GeoChat-7B`
- `GeoGround`

## 目录说明

- `data/`
  - 处理后的训练集、测试集和数据摘要。
  - 不包含图片实体，只保留标注与图片相对路径占位。
- `scripts/data/`
  - 数据集准备脚本。
- `scripts/train/`
  - 两条可训练链路的训练入口。
- `scripts/eval/`
  - 五个模型的评测入口，以及共用评测脚本。
- `results/`
  - 这五个模型当前选定结果对应的评测摘要。
- `table/`
  - 当前这组 grounding 对比表。

## 数据口径

- 训练使用 `train.json`。
- 评测使用 `test.json`。
- 验证集不是单独文件，而是训练脚本从 `train.json` 按固定随机种子和 `--geochat-val-ratio 0.1` 划分出来的。
- 标注中的真实图片索引主键是 `image_id`。
- `image_rel_path` 现在只是仓库内相对路径占位，用于表达目录结构，不代表仓库中真的附带原图。

## 整理方式

- 这里所有内容都是独立副本，不使用符号链接。
- 标注和结果中的路径字段已经尽量改成仓库内相对路径或相对路径占位，便于单独整理到 GitHub。
