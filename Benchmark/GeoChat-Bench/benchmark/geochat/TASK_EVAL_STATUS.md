# GeoChat-Benchmark 任务与评测状态整理

本文档整理当前 `GeoChat-Benchmark` 中各任务的作用、我们现在的评测方式，以及哪些部分是确定的，哪些部分仍然存在不确定性。

## 总结

- 能正式测：
  - Scene Classification
  - VQA
  - Region Captioning
- 当前不能正式测：
  - Grounding Description
- 能测，但底层 scorer 细节需要我们自己定义：
  - Referring
  - Grounding

## 1. Scene Classification

- 任务作用：
  - 给整张遥感图，判断它属于哪个场景类别。
- 对应数据：
  - `dataset/GeoChat-Bench/aid.jsonl`
  - `dataset/GeoChat-Bench/UCmerced.jsonl`
- 生成方式：
  - 直接使用数据中的 `text` 作为问题。
  - 当前实现：`src/shared/generate.py`
- 评测方式：
  - 指标为分类准确率。
  - 按官方公开 scene 脚本的逻辑，用 `question_id.split('/')[0]` 作为标准类别。
- 当前判断：
  - 这部分与官方公开逻辑对齐。
  - 这部分可以正式测，且不确定性最小。

## 2. VQA

- 任务作用：
  - 给图和一个问题，让模型输出一个词或短语作为答案。
- 对应数据：
  - `dataset/GeoChat-Bench/hrben.jsonl`
  - `dataset/GeoChat-Bench/lrben.jsonl`
- 生成方式：
  - 直接使用数据中的 `text` 作为问题。
  - 官方文档要求问题后附加 “Answer the question using a single word or phrase.”，公开转换后的 `jsonl` 已经包含这句，因此不需要再次拼接。
- 评测方式：
  - 指标为准确率。
  - 当前实现使用“轻归一化 + 精确匹配”：
    - 小写化
    - 去掉多余空格
    - 去掉常见句尾标点
    - 然后与标准答案做字符串精确匹配
  - `LRBEN` 默认排除 `count` 类问题。
- 当前判断：
  - 指标本身是合理的。
  - 当前实现已经兼容原始 GT 文件中的 `answers` 结构，并且会过滤 `active=false` 的无效项。
  - `LRBEN` 当前本地 raw GT 与公开 `jsonl` 的 `question_id` 是对齐的，因此可以正式打分。
  - `HRBEN` 当前公开 `GeoChat-Bench/hrben.jsonl` 实际对应 Zenodo 的 `USGS_split_test_phili_*` 文件。
  - 只要使用 `test_phili` 的 questions / answers / images 三份 raw 文件，就可以构造出与当前 bench 完整对齐的 GT。

## 3. Region Captioning

- 任务作用：
  - 给图和一个区域框，让模型描述这个区域里是什么。
- 对应数据：
  - `dataset/GeoChat-Bench/region_captioning.jsonl`
- 生成方式：
  - 使用 prompt：
    - `[identify] What is the object present at {question}`
- 评测方式：
  - 指标为：
    - `ROUGE-1`
    - `ROUGE-L`
    - `METEOR`
  - 当前实现基于标准库：
    - `rouge_score`
    - `nltk.meteor_score`
- 当前判断：
  - 这部分的指标类型是固定的，当前实现方式合理。
  - 官方没有公开完整 scorer，但这类指标本身就是标准算法，因此这部分可以正式测。
  - 需要注意的是，不同工具实现的细节可能会带来极小差异，但不影响总体结论。

## 4. Referring

- 任务作用：
  - 给一句指代描述，让模型返回对应目标的位置框。
- 对应数据：
  - `dataset/GeoChat-Bench/referring.jsonl` 中 `type == "ref"` 的样本
- 生成方式：
  - 使用 prompt：
    - `[refer] Give me the location of <p> {question} </p>`
- 评测方式：
  - 高层指标按论文描述使用 `acc@0.5`。
  - 当前具体实现流程：
    - 从输出文本中解析预测框
    - 将 GT 多边形转为普通矩形框
    - 计算 IoU
    - 单目标样本看该框是否满足阈值
    - 多目标样本做一一匹配，全部满足 `IoU >= 0.5` 才算正确
  - 同时统计：
    - `small / medium / large`
    - `single_object / multi_object`
    - `refer / grounding`
    - `all`
- 当前判断：
  - 高层指标 `acc@0.5` 是明确的。
  - 文本解析当前采用严格标签解析，不再抓任意数字。
  - 多目标匹配当前改成了公开 generalized referring scorer 同型的“全局贪心 IoU 匹配”，并以严格 `F1=1` 判定样本成功。
  - 由于 `GeoChat` 自身没有公开完整 scorer，所以仍然不能声称与作者私有实现完全等价，但现在这套底层逻辑有公开依据。

## 5. Grounding

- 任务作用：
  - 同样是“根据文字找框”，但在该 bench 中与 `refer` 分开统计。
- 对应数据：
  - `dataset/GeoChat-Bench/referring.jsonl` 中 `type == "grounding"` 的样本
- 生成方式：
  - 当前已经按 `type` 区分：
    - `type == "ref"` 使用 `[refer]`
    - `type == "grounding"` 使用 `[grounding] {question}`
- 评测方式：
  - 与 `Referring` 共用同一套 IoU 和 `acc@0.5` 逻辑。
  - 最后按 `type` 单独汇总 `grounding` 子集结果。
- 当前判断：
  - 高层指标仍然是明确的 `acc@0.5`。
  - 底层 scorer 细节仍然是我们自己补的，因此与 `Referring` 一样，属于“能测，但细节并非官方完整公开给定”。

## 6. Grounding Description

- 任务作用：
  - 同时涉及文字描述和目标框定位。
- 对应数据：
  - `dataset/GeoChat-Bench/grounding_description.jsonl`
- 当前状态：
  - 当前不能正式测。
- 原因：
  - 公开 release 中只有：
    - `image_id`
    - `description`
  - 但论文描述中，这个任务需要同时评文字和框。
  - 这说明正式评测还需要额外的框级 GT 或对象对应信息，而这些信息在当前公开 release 中没有提供。

## 当前最重要的结论

- `Scene Classification`
  - 可以正式测
  - 与官方公开逻辑对齐
- `VQA`
  - 当前已经可以正式测
  - `LRBEN` 直接使用原始答案文件
  - `HRBEN` 通过 `test_phili` 的 questions / answers / images 构造出与 `GeoChat-Bench/hrben.jsonl` 完整对齐的 GT
- `Region Captioning`
  - 可以正式测
  - 指标是标准文本指标，当前实现合理
- `Referring / Grounding`
  - 可以测
  - 高层指标明确
  - 底层 scorer 细节仍需继续对照公开先例
- `Grounding Description`
  - 当前不能正式测
  - 原因是缺正式评测需要的完整标注信息
