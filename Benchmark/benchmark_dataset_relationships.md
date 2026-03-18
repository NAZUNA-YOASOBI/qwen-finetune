- RSIEval：
  - 训练来源 = 无
  - 测试来源 = DOTA-v1.5

- VRSBench：
  - 训练来源 = DOTA-v2.0 + DIOR
  - 测试来源 = DOTA-v2.0 + DIOR

- GeoChat-Bench：
  - 训练来源 = SAMRS(DOTA-v2.0 + DIOR + FAIR1M-2.0) + NWPU-RESISC-45 + RSVQA-LRBEN(RSVQA-LR) + FloodNet
  - 测试来源 = AID + UCMerced + RSVQA-HRBEN(RSVQA-HR) + RSVQA-LRBEN(RSVQA-LR) + SAMRS(DOTA-v2.0 + DIOR + FAIR1M-2.0)

- GeoBench-VLM：
  - 训练来源 = 无
  - 测试来源 = AiRound + RESISC45(NWPU-RESISC45) + PatternNet + MtSCCD + FireRisk + FGSCR + FAIR1M + DIOR + DOTA + Forest Damage + Deforestation + COWC + NASA Marine Debris + RarePlanes + fMoW + xBD + PASTIS + FPCD + GVLM + DeepGlobe Land Cover + GeoNRW + So2Sat + QuakeSet

- XLRS-Bench：
  - 训练来源 = 无
  - 测试来源 = DOTA-v2.0 + ITCVD + MiniFrance + Toronto + Potsdam + HRSCD

---

## 说明

- 下面每个 benchmark 一张表。
- 只摘录论文主文或补充材料里的原文句子，或原文表格整行。
- `总结` 只基于对应摘录本身，不额外扩展到摘录之外的推断。

---

## RSIEval

| 来源数据集 | 阶段 | 原文页码 | 原文摘录 | 总结 |
|---|---|---:|---|---|
| DOTA-v1.5 | 测试来源 / 数据构建 | 主文 p.5 | "In order to benchmark various domain-specific and general VLMs on remote sensing image caption (RSIC) and remote sensing visual question answering (RSVQA) tasks, we construct an evaluation set RSIEval. We divided the images in the validation set of DOTA-v1.5 into patches with a size of 512×512 and then selected 100 images from these patches for further manual annotation." | RSIEval 测试来源来自 DOTA-v1.5 的 validation set，先切成 512×512 patch，再从中选 100 张做人工标注。 |
| DOTA-v1.5 | 相关上文背景 | 主文 p.3 | "DOTA-v1.5, which we used in this paper, covers 16 object categories;" | 文中明确使用的 DOTA 版本是 v1.5。 |

---

## VRSBench

| 来源数据集 | 阶段 | 原文页码 | 原文摘录 | 总结 |
|---|---|---:|---|---|
| DOTA-v2 | 数据构建 | 主文 p.4 | "In this study, we utilize two prominent open-access object detection datasets, DOTA-v2 [33] and DIOR [34], to develop our VRSBench dataset. Due to the unavailability of test labels for DOTA-v2, we incorporate only its training and validation sets. We divide each image into patches measuring 512 × 512 pixels." | DOTA-v2 只用了 train 和 validation，两者进入 VRSBench 前会被切成 512×512 patch。 |
| DIOR | 数据构建 | 主文 p.4 | "In this study, we utilize two prominent open-access object detection datasets, DOTA-v2 [33] and DIOR [34], to develop our VRSBench dataset." | DIOR 是 VRSBench 的两大原始检测来源之一。 |
| DOTA-v2 + DIOR | patch 与过滤 | 补充 p.15 | "Source datasets: Images are sourced from the DOTA-v2 [33] and DIOR [34] datasets and annotated with high-resolution details. We divide each image into patches measuring 512 × 512 pixels and filter out patches with no object annotations. This yields over 20,310 image patches from the DOTA-v2 dataset and 9,304 patches from the DIOR dataset." | 两个来源都会先切 patch，再过滤掉无目标 patch，最后分别保留 20,310 和 9,304 个 patch。 |
| DOTA-v2 + DIOR | benchmark train/test 划分 | 主文 p.6 | "To facilitate benchmark evaluation, we partition our VRSBench dataset into two distinct, non-overlapping splits designated for model training and evaluation. We split the datasets according to official splits of DOTA [33] and DIOR [34] datasets, where their training images are used to build the training set of VRSBench and their validation sets are used as the test set." | VRSBench 的 train/test 划分直接跟随 DOTA 和 DIOR 的官方 train/validation 划分。 |

---

## GeoChat-Bench

| 来源数据集 | 阶段 | 原文页码 | 原文摘录 | 总结 |
|---|---|---:|---|---|
| DOTA | 训练来源 / SAMRS | 主文 p.5 | "Specifically, we integrate three object detection (DOTA [35], DIOR [6], and FAIR1M [27] which together form the SAMRS [31] dataset), one scene classification (NWPU-RESISC-45 [5]), one VQA (LRBEN[20]), and one flood detection [25] VQA dataset (see Table 2)." | DOTA 不是单独直接写成训练集，而是和 DIOR、FAIR1M 一起组成 SAMRS 后进入训练。 |
| DIOR | 训练来源 / SAMRS | 主文 p.5 | "Specifically, we integrate three object detection (DOTA [35], DIOR [6], and FAIR1M [27] which together form the SAMRS [31] dataset), one scene classification (NWPU-RESISC-45 [5]), one VQA (LRBEN[20]), and one flood detection [25] VQA dataset (see Table 2)." | DIOR 也是通过和 DOTA、FAIR1M 共同组成 SAMRS 的方式进入训练。 |
| FAIR1M | 训练来源 / SAMRS | 主文 p.5 | "Specifically, we integrate three object detection (DOTA [35], DIOR [6], and FAIR1M [27] which together form the SAMRS [31] dataset), one scene classification (NWPU-RESISC-45 [5]), one VQA (LRBEN[20]), and one flood detection [25] VQA dataset (see Table 2)." | FAIR1M 也是通过 SAMRS 进入训练。 |
| DOTA | SAMRS 内部采样 | SAMRS 主文 p.3 | "Based on the available annotations, we only transform the training and validation sets of DOTA-V2.0 and FAIR1M-2.0, while for DIOR, all data has been utilized." | 在 SAMRS 内部，DOTA-V2.0 只使用了 train 和 validation。 |
| FAIR1M | SAMRS 内部采样 | SAMRS 主文 p.3 | "Based on the available annotations, we only transform the training and validation sets of DOTA-V2.0 and FAIR1M-2.0, while for DIOR, all data has been utilized." | 在 SAMRS 内部，FAIR1M-2.0 只使用了 train 和 validation。 |
| DIOR | SAMRS 内部采样 | SAMRS 主文 p.3 | "Based on the available annotations, we only transform the training and validation sets of DOTA-V2.0 and FAIR1M-2.0, while for DIOR, all data has been utilized." | 在 SAMRS 内部，DIOR 使用了全部数据。 |
| DOTA + FAIR1M + DIOR | SAMRS 预处理 | SAMRS 主文 p.4 | "Prior to transformation, we follow the common practice to crop images in DOTA and FAIR1M datasets to 1,024 × 1,024 and 600 × 600, respectively, while images in DIOR are maintained at the size of 800 × 800." | 在 SAMRS 内部，DOTA 和 FAIR1M 会先裁成固定尺寸，DIOR 保持 800×800。 |
| NWPU-RESISC-45 | 训练来源 | 主文 p.5 | "Specifically, we integrate three object detection (DOTA [35], DIOR [6], and FAIR1M [27] which together form the SAMRS [31] dataset), one scene classification (NWPU-RESISC-45 [5]), one VQA (LRBEN[20]), and one flood detection [25] VQA dataset (see Table 2)." | NWPU-RESISC-45 被明确列为训练时使用的 scene classification 数据来源。 |
| RSVQA-LRBEN | 训练来源 | 主文 p.5 | "Specifically, we integrate three object detection (DOTA [35], DIOR [6], and FAIR1M [27] which together form the SAMRS [31] dataset), one scene classification (NWPU-RESISC-45 [5]), one VQA (LRBEN[20]), and one flood detection [25] VQA dataset (see Table 2)." | RSVQA-LRBEN 被明确列为训练时使用的 VQA 来源之一。 |
| FloodNet | 训练来源 | 主文 p.5 | "Specifically, we integrate three object detection (DOTA [35], DIOR [6], and FAIR1M [27] which together form the SAMRS [31] dataset), one scene classification (NWPU-RESISC-45 [5]), one VQA (LRBEN[20]), and one flood detection [25] VQA dataset (see Table 2)." | FloodNet 被明确列为训练时使用的 flood VQA 来源。 |
| 多源 instruction 数据 | 训练集采样 | 主文 p.5 | "Specifically, from our short descriptions created using the below pipeline, we randomly sample 65k images to create multi-round conversations, 10k images to generate complex question answers and 30k images to generate detailed descriptions for the given short descriptions." | 训练 instruction 数据不是整批直接使用，而是从短描述数据里随机采样出 65k、10k、30k 三类子集。 |
| 多源 instruction 数据 | 训练集规模 | 主文 p.3 | "Table 1. Instruction following data used to train GeoChat. Instruction types and format are shown. We use a 308k set for training and a separate 10k instruction-set for testing." | GeoChat instruction 数据总训练规模是 308k，另有单独 10k instruction-set 用于测试。 |
| AID | 场景分类评测 | 主文 p.7 | "In total, the AID [34] dataset has 10,000 images within 30 classes. The images have been taken from different countries as well as different weather conditions. For evaluation, we use a 20% split of the AID [34] dataset." | AID 评测不是全量，而是只用了 20% split。 |
| UCMerced | 场景分类评测 | 主文 p.7 | "UCMerced [36] is a Land Use scene classification dataset, with 2,100 images and 21 classes. Each image is of size 256×256. We use the whole UCMerced [36] dataset as a zero-shot test set." | UCMerced 评测用了整个数据集。 |
| RSVQA-HRBEN | VQA 评测 | 主文 p.7-p.8 | "RSVQA-HRBEN [20] comprises 10,569 high-resolution photos and 1,066,316 question-answer pairs, with 61.5%, 11.2%, 20.5%, and 6.8% divided into training, validation, test 1, and test 2 sets, respectively. This dataset has three question types: presence, comparison, and count. For evaluation, we use the test set-2 for RSVQA-HRBEN [20] with 47k question answer pairs." | RSVQA-HRBEN 评测只用了 test set-2，而不是整个数据集。 |
| RSVQA-LRBEN | VQA 评测 | 主文 p.8 | "RSVQA-LR [20] is made up of 772 low-resolution images and 77,232 question-answer pairs, with 77.8%, 11.1%, and 11.1% used for training, validation, and testing, respectively. There are four different categories of questions: presence, comparison, rural/urban, and count. We omitted area and count questions during evaluation because the responses are numerical and quantifiable into numerous categories. In the RSVQA-LRBEN [20] dataset, for example, counting questions are quantified into five categories: 0, between 1 and 10, between 11 and 100, between 101 and 1000, and greater than 1000. For evaluation, we use the test set of RSVQA-LRBEN [20] with 7k question-answer pairs." | RSVQA-LRBEN 评测使用 test set，但去掉了 area 和 count 问题。 |
| SAMRS | grounding 评测 | 主文 p.8 | "Datasets for evaluation. For the evaluation of grounding tasks, we propose a new benchmark that contains different referring and grounding tasks. We use the validation set from [31] and used the same dataset creation pipeline as in Sec. 4 to construct the test benchmark." | grounding benchmark 是从 SAMRS 的 validation set 按同样流程构建出来的。 |

---

## XLRS-Bench

| 来源数据集 | 阶段 | 原文页码 | 原文摘录 | 总结 |
|---|---|---:|---|---|
| 全部来源合集 | 数据收集总量 | 主文 p.2 | "To address these challenges, we introduce XLRS-Bench, a benchmark designed to assess the perception and reasoning capabilities of MLLMs on ultra-high-resolution RS scenarios. We first collected 1,400 real-world ultra-high-resolution RS images with large sizes (8,500×8,500 pixels on average)." | XLRS-Bench 总共收集了 1,400 张超高分辨率图像。 |
| 全部来源合集 | 数据筛选口径 | 主文 p.5 | "Specifically, we compiled 1,400 images from realistic RS scenarios for different downstream tasks, selecting them rigorously based on diversity and quality." | 这 1,400 张图是经过人工严格筛选得到的，不是简单整库直用。 |
| DOTA-v2 | 数据收集 | 主文 p.5 | "For detection tasks, we sourced 270 images at 4,096×4,096 and 210 images at 7,360×4,912 from DOTA-v2 [60], and added 50 images at a size of 3,744×5,616 from the ITCVD [61] dataset." | DOTA-v2 在 XLRS-Bench 里一共贡献了 480 张 detection 图像。 |
| ITCVD | 数据收集 | 主文 p.5 | "For detection tasks, we sourced 270 images at 4,096×4,096 and 210 images at 7,360×4,912 from DOTA-v2 [60], and added 50 images at a size of 3,744×5,616 from the ITCVD [61] dataset." | ITCVD 在 XLRS-Bench 里贡献了 50 张 detection 图像。 |
| MiniFrance | 数据收集 | 主文 p.5 | "For segmentation tasks, we used 457 images at 10,000×10,000 resolution from MiniFrance [6], 13 images at 11,500×7,500 from Toronto [46], and 30 images at 6,000×6,000 from Potsdam [45]." | MiniFrance 在 XLRS-Bench 里贡献了 457 张 segmentation 图像。 |
| Toronto | 数据收集 | 主文 p.5 | "For segmentation tasks, we used 457 images at 10,000×10,000 resolution from MiniFrance [6], 13 images at 11,500×7,500 from Toronto [46], and 30 images at 6,000×6,000 from Potsdam [45]." | Toronto 在 XLRS-Bench 里贡献了 13 张 segmentation 图像。 |
| Potsdam | 数据收集 | 主文 p.5 | "For segmentation tasks, we used 457 images at 10,000×10,000 resolution from MiniFrance [6], 13 images at 11,500×7,500 from Toronto [46], and 30 images at 6,000×6,000 from Potsdam [45]." | Potsdam 在 XLRS-Bench 里贡献了 30 张 segmentation 图像。 |
| HRSCD | 数据收集 | 主文 p.5 | "Additionally, for change detection tasks, we included 185 pairs (370 images) at 10,000×10,000 resolution from the HRSCD [10] dataset." | HRSCD 在 XLRS-Bench 里贡献了 185 对变化检测图像。 |
| 全部来源合集 | 超高分辨率比例 | 主文 p.2 | "XLRS-Bench features the largest image sizes available, 10∼20× than that of existing datasets, with 840 images out of all images at a resolution of 10,000×10,000 pixels." | 1,400 张图中有 840 张达到 10,000×10,000 分辨率。 |

---

## GeoBench-VLM

| 来源数据集 | 阶段 | 原文页码 | 原文摘录 | 总结 |
|---|---|---:|---|---|
| 全部来源合集 | 总采样口径 | 主文 p.4 | "Dataset pipeline: GEOBench-VLM integrates open datasets, and manual annotation aided with automated tools. For diversity, each task samples images from multiple datasets. By combining multi-source datasets with structured question design, GEOBench-VLM enables scalable and high-quality evaluation of VLMs across geospatial tasks." | GeoBench-VLM 的总口径是多源数据集联合采样，每个任务都从多个来源中取图。 |
| 全部来源合集 | 总体说明 | 补充 p.13 | "The datasets we use in our evaluation cover a wide range of geospatial tasks, showing the variety and depth of challenges in geospatial analysis. As shown in Table A1, these datasets include tasks like scene understanding, spatial relation, instance counting, temporal understanding, referring expression segmentation, and working with non-optical data." | 补充材料确认这些来源覆盖了多个任务类型。 |
| AiRound | 表 A1 原文 | 补充 p.14 | "AiRound[35]                                                                                RGB, Sentinel-2 (10m)              2020" | Table A1 明确把 AiRound 列为来源数据集之一。 |
| RESISC45 | 表 A1 原文 | 补充 p.14 | "RESICS45[10]                                                                               RGB                                2017" | Table A1 明确把 RESISC45 列为来源数据集之一。 |
| PatternNet | 表 A1 原文 | 补充 p.14 | "PatternNet[74]                    Scene Understanding,                                     RGB                                2018" | Table A1 明确把 PatternNet 列为来源数据集之一。 |
| MtSCCD | 表 A1 原文 | 补充 p.14 | "MtSCCD[31]                        Object Classification                                    RGB (1m)                           2024" | Table A1 明确把 MtSCCD 列为来源数据集之一。 |
| FireRisk | 表 A1 原文 | 补充 p.14 | "FireRisk[45]                                                                               RGB (1m)                           2023" | Table A1 明确把 FireRisk 列为来源数据集之一。 |
| FGSCR | 表 A1 原文 | 补充 p.14 | "FGSCR[14]                                                                                  RGB                                2021" | Table A1 明确把 FGSCR 列为来源数据集之一。 |
| FAIR1M | 表 A1 原文 | 补充 p.14 | "FAIR1M[48]                           Spatial Relation                                      RGB (0.3–0.8m)                     2021" | Table A1 明确把 FAIR1M 列为来源数据集之一。 |
| DIOR | 表 A1 原文 | 补充 p.14 | "DIOR[11]                                                                                   RGB                                2020" | Table A1 明确把 DIOR 列为来源数据集之一。 |
| DOTA | 表 A1 原文 | 补充 p.14 | "DOTA[56]                                Captioning                                         RGB (0.1–1m)                       2021" | Table A1 明确把 DOTA 列为来源数据集之一。 |
| Forest Damage | 表 A1 原文 | 补充 p.14 | "Forest Damage[2]                                                                           RGB                                2021" | Table A1 明确把 Forest Damage 列为来源数据集之一。 |
| Deforestation | 表 A1 原文 | 补充 p.14 | "Deforestation[12]                                                                          RGB                                2024" | Table A1 明确把 Deforestation 列为来源数据集之一。 |
| COWC | 表 A1 原文 | 补充 p.14 | "COWC[37]                                Counting                      Bounding Box         RGB (15 cm)                        2016" | Table A1 明确把 COWC 列为来源数据集之一。 |
| NASA Marine Debris | 表 A1 原文 | 补充 p.14 | "NASA Marine Debris[44]                                                                     RGB (3m)                           2024" | Table A1 明确把 NASA Marine Debris 列为来源数据集之一。 |
| RarePlanes | 表 A1 原文 | 补充 p.14 | "The RarePlanes Dataset[46]                                                                 RGB (0.3m)                         2020" | Table A1 明确把 RarePlanes 列为来源数据集之一。 |
| fMoW | 表 A1 原文 | 补充 p.14 | "fMoW[18]                                                               Class               RGB (1m)                           2018" | Table A1 明确把 fMoW 列为来源数据集之一。 |
| xBD | 表 A1 原文 | 补充 p.14 | "xBD[15]                                                        Bounding Box, Instance      RGB (0.8m)                         2019" | Table A1 明确把 xBD 列为来源数据集之一。 |
| PASTIS | 表 A1 原文 | 补充 p.14 | "PASTIS[43]                                                        Semantic Mask            MSI (10m)                          2021" | Table A1 明确把 PASTIS 列为来源数据集之一。 |
| FPCD | 表 A1 原文 | 补充 p.14 | "FPCD[51]                                                          Semantic Mask            RGB (1m)                           2022" | Table A1 明确把 FPCD 列为来源数据集之一。 |
| GVLM | 表 A1 原文 | 补充 p.14 | "GVLM[71]                                                               Class               RGB (0.6m)                         2023" | Table A1 明确把 GVLM 列为来源数据集之一。 |
| DeepGlobe Land Cover | 表 A1 原文 | 补充 p.14 | "DeepGlobe Land Cover[13]           Referring Expression                                    RGB (0.5m)                         2018" | Table A1 明确把 DeepGlobe Land Cover 列为来源数据集之一。 |
| GeoNRW | 表 A1 原文 | 补充 p.14 | "GeoNRW[4]                             Segmentation                                         RGB (1m)                           2021" | Table A1 明确把 GeoNRW 列为来源数据集之一。 |
| So2Sat | 表 A1 原文 | 补充 p.14 | "So2Sat[76]                                                               Class             SAR, MSI (10m)                     2020" | Table A1 明确把 So2Sat 列为来源数据集之一。 |
| QuakeSet | 表 A1 原文 | 补充 p.14 | "QuakeSet[41]                                                            Number             SAR (10m)                          2024" | Table A1 明确把 QuakeSet 列为来源数据集之一。 |
| 分类来源合集 | 场景理解任务 | 主文 p.4 | "For scene understanding tasks, including scene classification, land use classification, and crop type classification, we use classification datasets [10, 31, 35, 43, 45, 74]." | 这句明确了 scene understanding 任务对应的是一组 classification datasets。 |
