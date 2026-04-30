# VRSBench Outputs 目录说明

本目录按实验阶段和任务类型整理，便于快速定位：

- `01_baseline/`
  - `caption/`：Baseline 的图像描述输出与汇总
  - `grounding/`：Baseline 的 referring/grounding 输出与汇总

- `02_rsicd_train_vrsbench_test/`
  - `caption/merger_only/`：在 RSICD 训练、在 VRSBench 测试（Merger-Only）
  - `caption/merger_lora/`：在 RSICD 训练、在 VRSBench 测试（Merger+LoRA）

- `03_vrsbench_joint_train/`
  - `caption/merger_only/`：VRSBench 联合训练后的 caption 输出（Merger-Only）
  - `caption/merger_lora/`：VRSBench 联合训练后的 caption 输出（Merger+LoRA）
  - `grounding/merger_only/`：VRSBench 联合训练后的 grounding 输出（Merger-Only）
  - `grounding/merger_lora/`：VRSBench 联合训练后的 grounding 输出（Merger+LoRA）

- `official_vqa_gpt/`
  - VRSBench VQA 在 GPT judge 前的 raw generation prediction 文件

说明：每个子目录中通常包含
- `*.jsonl`：逐样本预测结果
- `*_summary.json` 或 `*_eval_summary.json`：汇总指标
