# Benchmark 数据集关系说明

## 目的

这份文档用于说明 `Benchmark/` 目录下各个数据集或 Bench 之间的关系，重点区分以下几种情况：

- 真正独立的新 Bench
- 同一篇论文里另外拿来做对比的外部公开数据集
- 同一数据家族下的不同子集或不同命名

这样做的目的，是避免后续在整理代码、结果和论文材料时，把“同论文里一起出现”误写成“一个数据集包含另一个数据集”。

---

## 当前目录包含的条目

- `GeoChat`
- `RSICD`
- `RSIEval`
- `RSVQA-HR`
- `RSVQA-LR`
- `Sydney-captions`
- `UCM-captions`
- `VRSBench`
- `XLRS-Bench`

---

## 一、真正独立的 Bench

### 1. VRSBench

- `VRSBench` 是独立提出的多任务遥感视觉语言 Bench。
- 它主要面向 caption、grounding、referring 等任务。
- 它不是由当前目录里的其他几个小数据集简单拼出来的。

### 2. XLRS-Bench

- `XLRS-Bench` 是独立提出的超大分辨率遥感多模态 Bench。
- 它和 `VRSBench`、`RSIEval` 都不是包含关系。

### 3. RSIEval

- `RSIEval` 是 `RSGPT` 论文里单独构建的评测集。
- 它不是把 `RSICD`、`UCM-captions`、`Sydney-captions`、`RSVQA-HR`、`RSVQA-LR` 直接合并起来形成的。
- 更准确地说，`RSIEval` 是论文作者新建的一套较小评测集，用来统一评估 caption 和 VQA。
- 论文正文里给出的规模是：
  - `100` 个 image-caption 对
  - `936` 个 visual question-answer 对

### 4. GeoChat

- `GeoChat` 自己也提出了一套独立的遥感视觉语言评测体系。
- 它不是当前目录下其他几个 Bench 的简单并集。
- 但它在部分任务上会复用已有公开数据集家族做对比评测，尤其是 VQA 这一块。

---

## 二、同一论文体系下“另外拿来做对比”的外部数据集

这一类最容易和“被包含”搞混。

### 1. RSGPT / RSIEval 这一条线

在 `RSGPT` 论文里，除了作者自己提出的 `RSIEval` 外，还另外使用了几套已有公开数据集做 SOTA 对比实验：

- `UCM-captions`
- `Sydney-captions`
- `RSICD`
- `RSVQA-HR`
- `RSVQA-LR`

这几者和 `RSIEval` 的关系是：

- 它们是 `RSGPT` 论文中**另外选来做对比实验**的数据集
- 不是 `RSIEval` 的子集
- 也不是 `RSIEval` 的内部组成部分

可以把它理解成：

- `RSIEval`：作者新建的小型统一评测集
- `UCM/Sydney/RSICD/RSVQA-HR/RSVQA-LR`：论文额外拿来做标准公开基准对比的数据集

---

## 三、同一家族下的几个数据集

### 1. 遥感 caption 数据集家族

以下三个都属于遥感图像描述这个方向里常见的公开 caption 数据集：

- `UCM-captions`
- `Sydney-captions`
- `RSICD`

它们之间的关系是：

- 三者是同任务方向下的不同公开数据集
- 不是简单的包含关系
- 但在很多论文里，这三者会被一起作为 `RSIC` 对比实验来汇总展示

在 `RSGPT` 论文里，作者明确写到：

- `UCM-captions`
- `Sydney-captions`
- `RSICD`

这三个数据集一起用于 `RSIC` 对比实验。

另外，`RSICD` 的原始论文本身也会讨论和评测 `UCM-captions` 与 `Sydney-captions`，所以这三者在论文叙述层面经常一起出现。但这不等于：

- `RSICD` 包含 `UCM-captions`
- 或 `RSICD` 包含 `Sydney-captions`

更准确的说法应该是：

- `RSICD` 与 `UCM-captions`、`Sydney-captions` 同属遥感 caption 数据集家族
- 三者常被一起做横向比较

### 2. 遥感 VQA 数据集家族

以下两个都属于 `RSVQA` 这一数据家族：

- `RSVQA-HR`
- `RSVQA-LR`

它们之间的关系是：

- 同属于 `RSVQA` 家族
- 只是分成高分辨率版本和低分辨率版本
- 不是谁包含谁

在 `RSGPT` 论文里，作者明确把：

- `RSVQA-HR`
- `RSVQA-LR`

一起作为 `RSVQA` 对比实验使用。

---

## 四、GeoChat 中与现有目录的对应关系

`GeoChat` 这条线需要特别注意，因为论文里用了带 `BEN` 的命名。

### 1. `RSVQA-HRBEN` / `RSVQA-LRBEN`

在 `GeoChat` 论文里，VQA 对比部分使用了：

- `RSVQA-HRBEN`
- `RSVQA-LRBEN`

从论文描述看，它们和当前目录里的：

- `RSVQA-HR`
- `RSVQA-LR`

属于同一数据家族、同一评测方向下的命名体系，而不是一个全新无关的数据集。

因此，后续整理材料时，更稳妥的表述应当是：

- `GeoChat` 在 VQA 部分复用了 `RSVQA` 家族数据做评测

而不要直接写成：

- `GeoChat` 包含了 `RSVQA-HR`
- `GeoChat` 包含了 `RSVQA-LR`

### 2. 当前仓库中的实际整理方式

当前 `Benchmark/` 目录并没有把 `GeoChat` 的所有内部评测子集逐一拆成单独目录，而是只保留了：

- `GeoChat` 论文本身

因此，目前仓库的组织方式更适合理解成：

- `GeoChat` 是一条独立工作线
- 它在论文里复用了某些公开数据家族做评测
- 但这些复用数据并没有在当前 `GeoChat/` 目录下继续展开成完整子目录

---

## 五、一个更准确的总视图

如果从“当前仓库中的整理逻辑”去看，这些条目更适合按下面方式理解。

### A. 独立 Bench / 独立工作线

- `VRSBench`
- `XLRS-Bench`
- `RSIEval`
- `GeoChat`

### B. RSGPT 论文中额外用于公开对比的 caption 数据集

- `UCM-captions`
- `Sydney-captions`
- `RSICD`

### C. RSGPT 论文中额外用于公开对比的 VQA 数据集

- `RSVQA-HR`
- `RSVQA-LR`

---

## 六、后续写材料时建议使用的说法

为了避免表述不严谨，建议后续统一使用下面这些说法。

### 推荐说法

- `RSIEval` 是 `RSGPT` 提出的独立评测集
- `UCM-captions`、`Sydney-captions`、`RSICD` 是 `RSGPT` 论文中额外用于 caption 对比实验的公开数据集
- `RSVQA-HR`、`RSVQA-LR` 是 `RSGPT` 论文中额外用于 VQA 对比实验的公开数据集
- `GeoChat` 在部分任务上复用了 `RSVQA` 家族数据做评测
- `UCM-captions`、`Sydney-captions`、`RSICD` 属于同一遥感 caption 数据集家族
- `RSVQA-HR`、`RSVQA-LR` 属于同一 `RSVQA` 数据家族

### 不建议的说法

- `RSIEval` 包含 `RSICD`
- `RSIEval` 包含 `RSVQA-HR`
- `RSICD` 包含 `UCM-captions`
- `GeoChat` 包含 `RSVQA-HR`

这些说法会把“同论文里一起出现”误写成真正的数据集包含关系。

---

## 七、本文档的依据

这份说明基于以下材料整理：

- 当前仓库中各 Bench 的 `README.md`
- 已归档的 `RSGPT / RSIEval` 论文
- 已归档的 `RSICD` 论文
- 已归档的 `GeoChat` 论文

如果后续又补充了新的官方代码、数据划分说明或论文附录，并且出现了更精确的定义，应以官方材料为准，再同步更新本文档。
