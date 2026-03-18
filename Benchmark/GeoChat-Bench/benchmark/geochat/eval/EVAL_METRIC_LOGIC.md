# GeoChat-Benchmark 当前评测指标逻辑说明

本文档只说明当前 `GeoChat-Benchmark/benchmark` 这套评测链路里，**各任务的分数是怎么计算的**，哪些来自官方，哪些是我们自己补的，哪些部分仍然存在不确定性。

## 1. 当前实际用到的脚本

### 1.1 统一评测脚本

- Scene：
  - `GeoChat-Benchmark/benchmark/shared/scripts/eval_geochat_scene.py`
- VQA：
  - `GeoChat-Benchmark/benchmark/shared/scripts/eval_geochat_vqa.py`
- Region Caption：
  - `GeoChat-Benchmark/benchmark/shared/scripts/eval_geochat_region_caption.py`
- Referring / Grounding：
  - `GeoChat-Benchmark/benchmark/shared/scripts/eval_geochat_referring.py`

### 1.2 与正式评测直接相关、但不属于“指标公式”的辅助脚本

- HRBEN GT 对齐：
  - `GeoChat-Benchmark/benchmark/shared/scripts/prepare_geochat_hrben_gt.py`

### 1.3 官方仓库中可直接参照的入口

- 官方说明文档：
  - `GeoChat-Benchmark/GeoChat/docs/Evaluation.md`
- 官方 batch 生成脚本：
  - `GeoChat-Benchmark/GeoChat/geochat/eval/batch_geochat_scene.py`
  - `GeoChat-Benchmark/GeoChat/geochat/eval/batch_geochat_vqa.py`
  - `GeoChat-Benchmark/GeoChat/geochat/eval/batch_geochat_referring.py`
  - `GeoChat-Benchmark/GeoChat/geochat/eval/batch_geochat_grounding.py`

需要先说明一点：

- 官方仓库对 **Scene** 给了可直接参考的简单算分逻辑
- 对 **VQA / Region Caption / Referring / Grounding**，官方主要给的是生成入口、提示词方向、论文表格中的高层指标
- **没有完整公开一套可直接复现全部任务的 scorer**
- 其中 `batch_geochat_grounding.py` 当前仓库版本本身还是坏的，不能直接拿来当正式 scorer

---

## 2. 哪些是按官方 GitHub 或论文明确照着做的

### 2.1 Scene Classification

这部分是当前 GeoChat 里**最接近官方原样复刻**的一项。

官方依据：

- `GeoChat-Benchmark/GeoChat/geochat/eval/batch_geochat_scene.py`
- 论文里也明确写的是 `accuracy`

官方脚本里的核心逻辑是：

- GT：`question_id.split('/')[0].lower()`
- 预测：去空格、转小写、去句号
- 然后做精确匹配

我们当前实现：

- `GeoChat-Benchmark/benchmark/shared/scripts/eval_geochat_scene.py`

和官方一致的部分：

- GT 仍然从 `question_id` 里取类名
- 主指标仍然是 `accuracy`

我们只做了轻量归一化：

- 压缩连续空白
- 去掉句号

这个改动不改变任务本质，只是让匹配更稳。

### 2.2 GeoChat 论文里明确写过的高层指标

论文层面明确给过这些高层指标：

- Scene：`accuracy`
- VQA：`accuracy`
- Region Caption：`ROUGE-1`、`ROUGE-L`、`METEOR`
- Referring / Grounding：`acc@0.5`

也就是说：

- 我们当前采用的**主指标名字**和论文对齐
- 但论文没有把底层 scorer 细节完整公开到可逐行复刻的程度

### 2.3 LRBEN 排除 count

这一点也是按论文来的。

论文写得很明确：

- 在 `RSVQA-LRBEN` 上会排除 `count` 类问题

我们现在的 VQA 总控脚本里也是这样做的：

- `LRBEN` 默认排除 `count`
- `HRBEN` 不排除

---

## 3. 官方没给，但指标算法本身是固定的，我们按标准算法实现的

### 3.1 Region Caption 的 ROUGE / METEOR

这部分官方论文给了指标名字，但没有放完整 scorer。

不过这些指标本身是标准算法，所以我们直接按标准实现：

- `ROUGE-1`
- `ROUGE-L`
- `METEOR`

当前实现：

- `GeoChat-Benchmark/benchmark/shared/scripts/eval_geochat_region_caption.py`

所用库：

- `rouge_score`
- `nltk.meteor_score`

这部分不属于“我们自己定义了一个新指标”，而是：

- 官方给了指标名
- 我们按标准算法把它落地

### 3.2 Region Caption 这部分是否统一

这类指标整体比较统一，因为 `ROUGE` 和 `METEOR` 本身就是公开标准指标。

但仍要注意：

- 不同库版本
- 不同 tokenization 细节

会带来少量数值差异。

所以这里的结论是：

- **指标本身没有争议**
- **不同实现可能有极小数值差**

---

## 4. 哪些是我们为了工程可用性补上的

这些部分多数不是改指标公式，而是让公开数据和当前任务真正能跑通。

### 4.1 HRBEN 的 GT 对齐

这是当前 GeoChat 链路里最重要的工程修正之一。

问题在于：

- 公开的 `GeoChat-Bench/hrben.jsonl` 不带最终答案
- 直接拿它没法正式打分

我们现在的做法是：

- 去下载官方 raw 的 `test_phili` questions / answers / images
- 再把它和 `GeoChat-Bench/hrben.jsonl` 按 `question_id` 对齐
- 生成：
  - `GeoChat-Benchmark/dataset/raw/HRBEN/geochat_hrben_test_phili_gt.jsonl`

当前脚本：

- `GeoChat-Benchmark/benchmark/shared/scripts/prepare_geochat_hrben_gt.py`

这不是指标公式本身，但没有它，`HRBEN` 这项没法正式评测。

### 4.2 统一的轻归一化

对于 Scene 和 VQA，我们都做了轻量文本归一化，例如：

- 转小写
- 压缩多余空格
- 去掉常见句尾标点

这属于工程修正，不是论文单独定义的指标。

作用很直接：

- 避免模型只因为句号、大小写或多余空格被判错

### 4.3 Referring / Grounding 的坐标解释

我们当前这条 `Referring / Grounding` 链路，不再做自动猜尺度。

当前固定采用的规则是：

- 以 `GeoChat` 官方论文和官方 demo 为准
- 将模型输出的 4 个数固定按 `$0 \sim 100$` 归一化坐标解释
- 然后统一映射到像素坐标
- 最后再和 GT 像素框算 IoU

当前实现：

- `GeoChat-Benchmark/benchmark/shared/scripts/eval_geochat_referring.py`

这样做的原因是：

- `GeoChat` 论文明确写了空间框数值归一化在 `$[0, 100]$`
- `GeoChat` 官方 demo 也明确按 `$0 \sim 100$` 映射到像素坐标
- 所以这里优先遵循官方已经公开说明的口径

### 4.4 GT 多边形转普通矩形框

GeoChat 这套 `Referring` 数据里的 GT 本来是 polygon。

我们当前会先把它转成普通矩形框：

- 左上角取最小值
- 右下角取最大值

这是为了后续统一算 IoU。

这同样是工程落地步骤，不是论文单独展开说明的 scorer 细节。

---

## 5. 官方没给，而且底层并不完全唯一，于是我们参考公开文献做的

这一类主要集中在 `Referring / Grounding`。

### 5.1 Referring / Grounding：高层指标明确，底层 scorer 不完整

论文对这两项给得很明确的是：

- 用 `acc@0.5`

也就是：

- 预测框和 GT 框的 IoU 达到 `$0.5$` 视为命中

但是论文和官方仓库**没有公开完整可直接复现的多目标 scorer**，尤其是：

- 多个预测框怎么和多个 GT 一一配对
- 文本里不是严格一个框时怎么解析
- 出现多个框时到底怎么算一条样本“成功”

这些细节，官方没有完整公开。

### 5.2 我们当前采用的公开借鉴做法

当前实现：

- `GeoChat-Benchmark/benchmark/shared/scripts/eval_geochat_referring.py`

我们现在做的是两步：

#### 第一步：严格框格式解析

我们只认这种格式：

- `<num>`

并按每 4 个数解释成一个框。

也就是说：

- 不是严格框标签的任意数字串，不再硬抓进来算框

这类“严格按框标签解析”的思路，和公开 grounded MLLM 的常见做法一致。

#### 第二步：多目标一一匹配

对多目标样本，我们当前采用：

- 全局贪心 IoU 匹配
- 每次找当前最大的 IoU 配对
- 一旦配对成功，就把这一行 GT 和这一列预测都移除
- 重复到没有 `IoU >= threshold` 的配对为止

然后按：

- `TP`
- `FP`
- `FN`

算每条样本的 `F1`。

最后规定：

- 只有 `F1 = 1.0`，才算这条样本完全成功

并输出：

- `acc`
- `mean_f1`

### 5.3 这一做法借鉴自哪些公开方向

当前主要借鉴的是 generalized referring / multi-object referring 这一类公开 scorer 思路，而不是 GeoChat 作者自己的专用 scorer。

主要参考方向是：

- `gRefCOCO / GRES / MDETR / FERRET` 这一脉的多目标 referring 与 grounded MLLM 评测思路
- 核心思想都是：
  - 先做一一匹配
  - 再基于匹配结果统计成功、精确率、召回率或 F1

### 5.3.1 当前正式写入说明文档的参考文献

这里列出当前真正用来支撑这套规则的公开论文。它们不是 GeoChat 官方 scorer，但能说明这套做法有明确依据。

- `MDETR: Modulated Detection for End-to-End Multi-Modal Understanding`
  - 会议：`ICCV 2021`
  - 作用：支持“文本条件目标匹配”和“一一配对后再按 IoU 阈值判断命中”这一类评测思路
  - 链接：<https://openaccess.thecvf.com/content/ICCV2021/html/Kamath_MDETR_-_Modulated_Detection_for_End-to-End_Multi-Modal_Understanding_ICCV_2021_paper.html>

- `GRES: Generalized Referring Expression Segmentation`
  - 会议：`CVPR 2023`
  - 作用：支持“一个表达式可能对应多个目标，评测时不能把多目标样本粗暴退化成单目标样本”这一点
  - 链接：<https://openaccess.thecvf.com/content/CVPR2023/html/Liu_GRES_Generalized_Referring_Expression_Segmentation_CVPR_2023_paper.html>

- `FERRET: Refer and Ground Anything Anywhere at Any Granularity`
  - 会议：`ICLR 2024`
  - 作用：支持“把框坐标当成模型输出协议的一部分，按严格格式解析 grounded 输出”这一类 grounded MLLM 做法
  - 链接：<https://machinelearning.apple.com/research/ferret>

需要强调：

- 这些论文提供的是“合理做法方向”
- 不是 GeoChat 论文自己指定必须逐字照搬某一篇
- 所以我们当前实现是“有公开依据的合理落地”，不是“作者私有 scorer 的唯一等价物”

### 5.4 这些公开文献的做法是不是完全统一

**不是完全统一。**

统一的是大方向：

- 都会做一一匹配
- 都离不开 IoU 阈值
- 多目标一定要处理重复匹配问题

但不完全统一的地方包括：

- 用贪心还是 Hungarian
- 样本级成功是看“全部命中”还是看 F1
- 有的报 `precision / recall / F1`
- 有的只报最终 success rate
- 文本解析是严格框标签，还是宽松抓数字

所以当前这套做法可以表述为：

- **有公开文献依据**
- **方向合理**
- **但不是所有文章都完全一致**

### 5.5 我们为什么最后选了当前这套

原因很简单：

- GeoChat 论文只给了 `acc@0.5`
- 没给完整 scorer
- 公开数据里又确实存在多目标样本

如果不补一套一一匹配规则，就没法把多目标样本严肃地算进去。

当前这套：

- 规则清楚
- 可复现
- 和公开 generalized referring 的思路一致

因此它是当前最稳妥的落地方式。

---

## 6. 官方没给，而且现在仍然存在不确定性的地方

### 6.1 VQA 的底层 scorer 不是官方完整公开的

GeoChat 对 VQA 给了：

- 提示词方向
- 输出要是一个词或短语
- 最后报 `accuracy`

但是它**没有给出完整公开 scorer**，尤其没明确说明：

- 是否只做原样 exact match
- 是否做大小写归一化
- 是否做标点归一化
- 是否接受同义词

我们当前做的是：

- 轻归一化后 exact match

这对短答案任务是合理的，但需要明确：

- 这不是从官方仓库逐行复刻的 scorer
- 不能声称与作者私有评测完全一致

### 6.2 Referring / Grounding 也不能声称和作者私有 scorer 完全等价

原因同样很直接：

- 论文只给了高层指标 `acc@0.5`
- 没给完整底层 scorer

所以我们现在最多只能说：

- 高层指标方向对齐
- 多目标处理借鉴了公开文献
- 当前实现合理、严谨、可复现

但不能说它就是作者原封不动的私有 scorer。

### 6.3 Grounding Description 当前仍然不能正式测

这项我们现在没有纳入正式评测。

原因是：

- 公开 release 里没有放出足够完整的正式 GT
- 只靠当前公开文件，不足以做可靠的最终算分

因此当前 GeoChat 评测链路里：

- Scene / VQA / Region Caption / Referring 这几项可以测
- Grounding Description 还不能严肃地正式测

---

## 7. 当前结论

### 7.1 对齐官方最充分的

- Scene：
  - GT 提取方式
  - 主指标 `accuracy`
- 论文层面给定的主指标名：
  - VQA 的 `accuracy`
  - Region Caption 的 `ROUGE-1 / ROUGE-L / METEOR`
  - Referring / Grounding 的 `acc@0.5`

### 7.2 官方没给 scorer，但算法本身标准、问题不大的

- Region Caption 的：
  - `ROUGE-1`
  - `ROUGE-L`
  - `METEOR`

### 7.3 明确属于我们自己补的工程步骤

- HRBEN `test_phili` GT 对齐
- 轻归一化
- 针对当前原生 `qwen3-vl / qwen3.5` 的固定 `$0 \sim 1000$` 坐标解释
- GT polygon 转普通矩形框

### 7.4 官方没给、因此需要借公开文献补规则的

- Referring / Grounding 的多目标匹配规则

并且这一类公开文献的做法：

- **大方向一致**
- **具体实现不完全统一**

### 7.5 当前最应该怎么表述

如果后续要把这套链路写进论文或方法说明，最稳妥的表述应该是：

- Scene 基本对齐官方公开脚本
- VQA 和 Region Caption 对齐论文给定主指标，其中 Region Caption 用标准文本生成指标实现
- Referring / Grounding 对齐论文给定的 `acc@0.5` 高层指标，底层多目标匹配规则参考公开 generalized referring 评测思路实现
- HRBEN 额外做了官方 raw 数据对齐，属于必要的数据工程修正
