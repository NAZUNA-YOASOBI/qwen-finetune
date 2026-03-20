# xgrammar 链路

当前 xgrammar 链路只用于最终保留的 `Ref-Det`。

入口脚本：

- `run/run_refdet_xgrammar_qwen3vl_20260320.sh`
- `run/run_refdet_xgrammar_qwen35_20260320.sh`

说明：

- 输出格式约束为严格 JSON array。
- `bbox_2d` 使用四点 polygon。
- prompt 版本：`bbox2d_pixel_polygon4_json_array_v9_xgrammar_countlocked`
- 评测使用 polygon IoU + 最大二分匹配。
