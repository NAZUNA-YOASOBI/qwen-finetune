# ftqwen3 脚本说明

本目录存放 `Qwen3-VL` 这一条线在 LHRS-Bench 中用到的脚本，按功能分类。

## 目录结构

- `prepare/`
  - 从原始 `qa.json` 整理出正式评测所需的 `question` 和 `attempt` 文件
  - 按论文规则做 4 次选项打乱
- `generate/`
  - 生成 Qwen3-VL baseline 输出
- `eval/`
  - 按论文正式规则算分
  - 4 次全对才算该题答对
- `report/`
  - 提取论文 Table 5
  - 生成 paper vs ours 对比表
- `utils/`
  - 通用工具脚本
  - 当前包括 jsonl 分片合并

## 使用约定

- 所有脚本都默认以 `Benchmark/LHRS-Bench` 为项目根目录来解析相对路径
- 由于原始数据集不在当前目录下，默认数据路径指向仓库外部的独立 `LHRS-Bench/datasets/LHRS-Bench`

## 评测口径

- 论文写明：
  - 每题 shuffle choices
  - 重复 4 次
  - 任一轮错误则该题记错
  - 输出正确选项字母或正确答案文本都算对
- 官方仓库 `main_bench_gen.py` 公开脚本没有完整实现这套 4 次严格口径
- 这里按论文正式 benchmark 口径实现
