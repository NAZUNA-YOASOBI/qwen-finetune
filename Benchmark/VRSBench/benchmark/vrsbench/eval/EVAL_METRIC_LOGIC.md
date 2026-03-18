# VRSBench 当前评测指标逻辑说明

本文档只说明当前 `VRSBench/benchmark/vrsbench/scripts` 这套评测链路中，**分数是怎么计算出来的**，以及哪些部分来自官方，哪些部分是我们自己补的。

## 1. 当前实际用到的脚本

- Caption 评测：
  - `VRSBench/benchmark/vrsbench/scripts/eval_vrsbench_cap.py`
- Grounding 评测：
  - 微调模型与常规链路：`VRSBench/benchmark/vrsbench/scripts/eval_vrsbench_referring.py`
  - 原生未微调 baseline：`VRSBench/benchmark/vrsbench/scripts/eval_referring_baseline_noftstyle.py`
- 与评测结果直接相关、但不属于“指标公式”的修正脚本：
  - `VRSBench/benchmark/vrsbench/scripts/fix_max_new_tokens_hits_baseline.py`
  - `VRSBench/benchmark/vrsbench/scripts/fix_max_new_tokens_hits_dinov3.py`
  - `VRSBench/benchmark/vrsbench/scripts/fix_max_new_tokens_hits_qwen_native.py`
  - `VRSBench/benchmark/vrsbench/scripts/fix_max_new_tokens_hits_sva_deepstack_ca.py`
  - `VRSBench/benchmark/vrsbench/scripts/normalize_referring_predictions.py`

---

## 2. 哪些是按官方 GitHub 明确复刻的

### 2.1 Grounding 的 IoU 公式

这部分是**明确按官方代码复刻**的。

官方依据：

- `VRSBench/datasets/VRSBench/VRSBench-git/VRSBench/eval_fianl/eval_utils.py`
- `VRSBench/datasets/VRSBench/VRSBench-git/VRSBench/eval_fianl/compute_metrics.ipynb`

当前实现：

- `VRSBench/benchmark/vrsbench/scripts/eval_vrsbench_referring.py`
- `VRSBench/benchmark/vrsbench/scripts/eval_referring_baseline_noftstyle.py`

复刻点：

- 框格式按 $[x_1, y_1, x_2, y_2]$
- 交并比面积计算里，宽高都带 `$+1$`
- 即：
  - 交集宽：`ix2 - ix1 + 1`
  - 交集高：`iy2 - iy1 + 1`
  - 两个框面积也都带 `$+1$`

这个细节不能随便改，因为它会直接影响最终 IoU。

特别说明一下原生未微调 `qwen3-vl` baseline：

- 它当前走的是 `eval_referring_baseline_noftstyle.py`
- 这条链路**不是**“官方 notebook 里那种直接把前 4 个整数当作 `$0 \sim 100$` 框再去算 IoU”
- 它前面会先把：
  - GT
  - 模型输出的 `bbox_2d`
  统一解释到 `$0 \sim 1000$` 坐标系，再换算成像素框
- 但是在**真正算 IoU 的那一步**，调用的仍然是同一套带 `$+1$` 的面积公式

所以更准确的说法是：

- **原生 baseline 在 IoU 公式这一层，和官方那套 `$+1$` 逻辑是等价的**
- **它和官方不完全一样的地方，在于前面的坐标解释层，不在 IoU 公式本身**

### 2.2 Grounding 的归一化分母

这部分也是**明确按官方 notebook 复刻**的。

官方 notebook 里虽然会单独统计 `valid_count`，但真正打印：

- `Acc@0.5`
- `Acc@0.7`
- `meanIoU`

时，分母用的是 `total_count`，不是 `valid_count`。

我们当前也保持这个口径：

- 解析失败、格式错、坐标非法的样本，不会从分母里删掉
- 它们等价于该样本这次没有命中，因此对最终结果贡献就是 `$0$`

这点在：

- `VRSBench/benchmark/vrsbench/scripts/eval_vrsbench_referring.py`
- `VRSBench/benchmark/vrsbench/scripts/eval_referring_baseline_noftstyle.py`

里都是这样做的。

### 2.3 Grounding 的主指标集合

当前我们输出的这些 grounding 指标，也和官方 notebook 保持一致：

- `Acc@0.5`
- `Acc@0.7`
- `meanIoU`
- `cumIoU`

并按三个 split 汇总：

- `unique`
- `non_unique`
- `all`

其中：

- `Acc@t`：IoU 是否达到阈值的样本比例
- `meanIoU`：所有样本的 IoU 平均值，分母也是 `total`
- `cumIoU`：所有样本交集面积之和除以并集面积之和

---

## 3. 官方没给，但算法本身是固定的，我们按标准算法实现的

### 3.1 Caption 的 BLEU / METEOR / ROUGE-L / CIDEr

这部分**不是从 VRSBench 官方仓库直接抄出来的**，因为官方仓库里没有给出完整的 caption scorer 脚本。

但这些指标本身是标准文本生成指标，算法是固定的，所以我们直接按通用标准实现：

- 先做 `PTBTokenizer`
- 再算：
  - `BLEU-1`
  - `BLEU-2`
  - `BLEU-3`
  - `BLEU-4`
  - `METEOR`
  - `ROUGE_L`
  - `CIDEr`

当前实现：

- `VRSBench/benchmark/vrsbench/scripts/eval_vrsbench_cap.py`

所用标准库：

- `pycocoevalcap`

因此这一部分虽然不是“官方 GitHub 明写”，但它不是我们拍脑袋定的，而是直接走通用 caption 标准算法。

### 3.2 Caption 指标的展示单位

当前 caption summary 同时保存两套值：

- 原始小数
- `metrics_x100`

对外报告表里默认用乘以 `$100$` 后的值展示。

这不是指标公式变化，只是展示单位变化。

---

## 4. 哪些是我们自己为了工程稳定性加的修正

这一类**不改变核心指标公式**，但会影响“哪些输出能被正确拿去算分”。

### 4.1 Caption 的截断重试

我们现在对 caption 增加了 fix 阶段：

- 如果生成到 `max_new_tokens=256` 还没有正常以 EOS 结束
- 就对该样本重试，最多 `10` 次
- 如果还是不行，就保留最后一次结果

当前脚本：

- `VRSBench/benchmark/vrsbench/scripts/fix_max_new_tokens_hits_baseline.py`
- `VRSBench/benchmark/vrsbench/scripts/fix_max_new_tokens_hits_dinov3.py`
- `VRSBench/benchmark/vrsbench/scripts/fix_max_new_tokens_hits_qwen_native.py`
- `VRSBench/benchmark/vrsbench/scripts/fix_max_new_tokens_hits_sva_deepstack_ca.py`

这一步不是官方给的算分公式，它是我们为了避免“句子被截断后直接拉低 caption 指标”而加的工程修正。

### 4.2 微调模型 grounding 输出的坐标截断

对微调模型的 grounding，我们现在会把解析出的 4 个数先做坐标修正：

- 小于 `$0$` 的改成 `$0$`
- 大于 `$100$` 的改成 `$100$`
- 如果修正后不满足 `$x_1 < x_2$` 或 `$y_1 < y_2$`，该预测记为无效

当前实现：

- `VRSBench/benchmark/vrsbench/scripts/eval_vrsbench_referring.py`
- `VRSBench/benchmark/vrsbench/scripts/normalize_referring_predictions.py`

这不是官方 notebook 原样提供的逻辑，而是我们为了适配当前训练后模型“偶尔出负数、偶尔出超过 100 的值”的工程修正。

### 4.3 原生 baseline grounding 的 0~1000 坐标兼容

原生未微调 Qwen3-VL 这一条链路，输出风格和微调模型不一样：

- 它更接近 `bbox_2d` 的 `$0 \sim 1000$` 尺度

所以 baseline noftstyle 这条评测不是直接拿前 4 个数当 `$0 \sim 100$` 框，而是：

- 先把 GT 统一解释成 `$0 \sim 1000$`
- 再根据图像宽高换算到像素坐标
- 预测框也按同样方式换算到像素坐标
- 最后再算 IoU

当前实现：

- `VRSBench/benchmark/vrsbench/scripts/eval_referring_baseline_noftstyle.py`

这也是工程适配，不是官方 notebook 直接提供的 baseline scorer。

因此原生 baseline 这条链路应当理解为：

- **前处理坐标解释**：是我们为了适配 `bbox_2d` 风格输出而加的
- **后面的 IoU 公式**：仍然与官方 `$+1$` 口径一致

### 4.4 评测前的顺序和完整性检查

我们还加了几项工程性保护：

- 先按 `qid` 排序再评测，避免 shard 合并后的顺序漂移
- 用 `meta` 里的 `num_samples` 检查生成是否完整

这部分不改变算分公式，只是为了防止“少算样本”或者“结果顺序乱了但没发现”。

---

## 5. 官方没给，而且并不是完全唯一的地方

### 5.1 Caption 的 `Avg_L`

`Avg_L` 在当前实现里是：

- 对每个预测句子按空格切分
- 统计单词数
- 再对所有样本取平均

当前实现：

- `VRSBench/benchmark/vrsbench/scripts/eval_vrsbench_cap.py`

这部分官方仓库没有给出 scorer，所以它**不是官方明确定义**。

但它也不是一个完全标准唯一的算法，因为：

- 有的人按空格切词
- 有的人按 tokenizer token 数
- 有的人会先做 PTB tokenize 再统计

我们当前采用的是最直观、最容易复现的“空格分词后的平均词数”。

因此它应被看作：

- **合理实现**
- 但不是“官方唯一口径”

### 5.2 Caption 的 CHAIR

官方表格里有 `CHAIR`，但当前我们的这套脚本**没有计算 CHAIR**。

原因是：

- 官方仓库没有放出这一部分可直接复用的完整 scorer
- 我们当前比较表里对 ours 这一列也没有填 CHAIR

因此当前 VRSBench caption 链路里：

- `BLEU / METEOR / ROUGE_L / CIDEr / Avg_L` 是有的
- `CHAIR` 暂时没有

### 5.2.1 如果后续要接 CHAIR，应当参考什么

这里不再单独展开论文条目，因为当前更关键的不是论文定义，而是实际可复现的实现来源。

如果后续要把 `CHAIR` 接进来，更应该优先参考它公开可运行的实现仓库，而不是只看论文描述。

当前最直接的实现参考来源可以写成：

- 参考 GitHub 仓库：`LisaAnne/Hallucination`
  - 链接：<https://github.com/LisaAnne/Hallucination>
  - 说明：其中包含 `CHAIR` 的实际计算脚本与 COCO 标注依赖关系

所以这里真正的情况是：

- `CHAIR` 这个指标本身不是我们自己定义的
- 但 VRSBench 官方仓库没有把它完整接进当前公开 benchmark 流程
- 如果后续要补，我们更应当参考它公开的 GitHub 实现

---

## 6. 有没有“官方没给，于是我们去借鉴外部文献”的部分

### 6.1 VRSBench：当前基本没有

VRSBench 这套链路里，当前**没有哪一项核心指标公式**是通过“去找外部论文，再借它的 scorer”来定下来的。

当前基本是两类：

- **官方明确给了的**：直接复刻
- **官方没给，但本身就是标准算法的**：直接走标准库

所以 VRSBench 这里没有像 GeoChat `Referring` 那样，必须额外去借外部文献确定多目标匹配规则。

---

## 7. 当前结论

### 7.1 可以认为已经稳定对齐的部分

- Grounding 的 IoU 公式
- Grounding 的 `Acc@0.5 / Acc@0.7 / meanIoU / cumIoU`
- Grounding 按 `total_count` 做归一化
- Caption 的 `BLEU / METEOR / ROUGE_L / CIDEr`

### 7.2 明确属于工程修正的部分

- Caption 的截断重试
- 微调模型 grounding 的 `$[0,100]$` 坐标修正
- 原生 baseline grounding 的 `$[0,1000]$` 坐标换算
- 排序、完整性检查、分片合并检查

### 7.3 仍然不能说是“官方唯一口径”的部分

- `Avg_L`
- `CHAIR` 缺失

如果后续还要进一步写论文方法部分，最稳妥的说法应该是：

- grounding 公式对齐官方 notebook
- caption 主指标采用标准 COCO caption 评价口径
- 额外加入少量工程修正以避免截断和非法框直接污染结果
