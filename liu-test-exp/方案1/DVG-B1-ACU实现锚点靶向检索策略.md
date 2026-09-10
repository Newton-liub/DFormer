# DVG-B1 条件式 Oracle 门控：实验阶段、原子能力单元与实现锚点检索策略

> **分析对象：** `doc/plans/2026-09-MUSeg-几何可信RGBD双路径MVE/04-DVG-B1条件式Oracle门控.md`
>
> **产物目标：** 将基础设计拆成按时间顺序推进的实验阶段，并为每个原子能力单元（Atomic Capability Unit，ACU）提供只面向“实现锚点”的检索策略。这里的实现锚点包括可复用代码、可核验数据身份、公开协议、指标实现和统计实现，不以一般性综述为主要目标。
>
> **证据边界：** 本文是检索与实施准备文档，不表示代码已经修改、protocol 已冻结、GPU 评价已经运行或 Oracle 假设已经得到支持。原方案中的 `REXXX`、`PRXXX` 是内部参考文献编号，只用于回查来源，**不得进入任何外部检索式**。

> **大白话结论：** 这份文档把 B1 从“一个门控想法”拆成了 7 个必须按门禁推进的阶段和 11 个可独立检索的 ACU。最先要解决的是 GSA 中深度几何项能否被单独隔离、以及 corruption mask 能否无歧义传播到四级特征网格；这两项任一失败，都不应进入 GPU 完整评价。

## 0. 检索语法与缩写约束

- 下文统一使用 Scopus 风格 `TITLE-ABS-KEY(...)` 表达式。迁移到 Web of Science 时，将字段改写为 `TS=(...)`；迁移到 Google Scholar、GitHub 或 Papers with Code 时，去掉字段包装并保留布尔结构。
- `GSA` 不单独进入检索式，因为它还常指 Gravitational Search Algorithm、Global Self-Attention 等；统一使用全称 `"Geometry Self-Attention"`，或与 `DFormerv2` 强绑定。
- `MVE`、`q`、`CI`、`BIoU` 不进入检索式：它们过短或跨领域歧义较大。分别改用 `minimum viable experiment`、具体 condition 名称、`confidence interval`、`Boundary IoU`。
- `RGB-D`、`mIoU`、`FP32`、`SHA-256`、`CUDA`、`IoU` 可在计算机视觉实现语境中使用，但优先同时给出全称。
- 截词符 `*` 只用于稳定词根，如 `implement*`、`reliab*`、`downsampl*`、`bootstrap*`。专有名称、哈希算法和固定协议标识不截断。
- 代码检索时优先保留带引号的类名、论文标题和模型名；文献检索时优先使用机制全称，避免只搜项目内部 condition 名称。

# 第一部分：按实验阶段解构

## 阶段 1：证据身份核验与 protocol 冻结准备

先证明 B1 复用的模型、数据范围、mask、condition 和 evaluator 与 v3 冻结证据一致；并在看结果前补齐效应量、clean 不劣容忍度、统计口径和合法终点。任何一项无法闭合，都应停在 `protocol-blocked`，不得进入实现或完整评价。

## 阶段 2：DFormerv2 GSA 机制定位

在论文、官方代码和当前仓库之间建立一一对应，精确定位 Geometry Self-Attention（几何自注意力）中的 spatial contribution 与 depth geometry contribution，确认二者能否被独立修改。这是整个方案的首个技术硬门禁。

## 阶段 3：Oracle mask 的多尺度传播协议

把原始 `Depth16` 网格上的 condition-specific corruption mask 无歧义地传播到四级 GSA 特征网格，并闭合 resize、patch pooling、下采样、padding、原图 view 与水平翻转 view 的空间对应关系。所有规则必须在看模型结果前冻结。

## 阶段 4：最小 Oracle 门控实现与等价性门禁

只对受损位置的 depth geometry contribution 施加一个最小门控，不改 RGB、Depth 输入、spatial contribution、checkpoint、logits 融合或后处理。随后用 clean、q=0 或全可信 mask 验证门控旁路与原模型达到预先定义的数值等价。

## 阶段 5：1–2 张图的三条件 preflight

对 `clean`、`boundary-q75`、`nonboundary-q50` 做小样本运行，检查 mask 对位、四级传播、输出有限性、五尺度与水平翻转路径、原始 Label 网格恢复和日志证据。preflight 只判断实现链路能否进入完整评价，不形成科学结论。

## 阶段 6：完整 paired development evaluation

在冻结的 218 张图、138 个 location group 上运行三组同源比较和五个 condition。主要比较是 `boundary-q75` 下 `Oracle-gated - corrupted baseline`；同时记录 Boundary IoU、mIoU、有效深度比例、置零像素数和传播摘要。official test 保持 `sealed_unread`。

## 阶段 7：location-group bootstrap、裁决与收口

先按图像计算，再按 location group 聚合，以 138 个组为重采样单位，使用固定 seed `20260908` 和 10,000 次 percentile bootstrap。依据预先冻结的效应量与 clean 不劣规则，裁决为 `oracle-supported`、`oracle-not-supported`、`inconclusive`，或因证据/实现错误进入 `protocol-blocked`、`stop`。

# 第二部分：ACU 与锚点实体清单

## 阶段 1：证据身份核验与 protocol 冻结准备

### ACU 1.1：冻结输入与运行身份的可复核绑定

**锚点实体：** DFormerv2-S RGB Quick-B0、epoch 420 checkpoint、MUSeg、`val-dev`、218 张图、138 个 location group、`Depth16`、v3 corruption mask、mask manifest、`clean`、`boundary-q25`、`boundary-q50`、`boundary-q75`、`nonboundary-q50`、`rgb-imagenet-rgb-order-v1`、五尺度原图/水平翻转、原始 Label 网格、`official_test_included=false`、SHA-256、[RE326] `MUSeg: A multimodal semantic segmentation dataset for complex underground mine scenes`。

### ACU 1.2：预注册效应量、不劣界值与统计协议

**锚点实体：** Boundary IoU、mIoU、`boundary-q75`、clean baseline、corrupted baseline、Oracle-gated、location group、paired bootstrap、95% percentile interval、bootstrap seed `20260908`、10,000 次重采样、六个 mine、`oracle-supported`、`oracle-not-supported`、`inconclusive`、[RE131] `RGB-D semantic mapping for underground robotic inspection using an attention-enhanced and boundary-refined DeepLabv3+ network`、[PR090] `Towards Robust Multi-Modal Semantic Segmentation with Teacher-Student Framework and Hybrid Prototype Distillation`。

## 阶段 2：DFormerv2 GSA 机制定位

### ACU 2.1：定位并证明 GSA 两类 contribution 可分离

**锚点实体：** DFormerv2、DFormerv2-S、Geometry Self-Attention、GSA、spatial contribution、depth geometry contribution、depth geometry prior、四个 GSA stage、[PR070] `DFormerv2: Geometry Self-Attention for RGBD Semantic Segmentation`。

## 阶段 3：Oracle mask 的多尺度传播协议

### ACU 3.1：原始像素 mask 到四级 GSA 网格的确定性投影

**锚点实体：** 原始 `Depth16`、corruption mask、四个 GSA stage、patch pooling、下采样、resize、depth geometry prior、全可信 mask、condition-specific mask、[PR070] `DFormerv2: Geometry Self-Attention for RGBD Semantic Segmentation`。

### ACU 3.2：水平翻转与多尺度 view 的 mask 对齐

**锚点实体：** 五尺度、原图、水平翻转、mask resize、flip view、原始 Label 网格、logits 融合、RGB、Depth、Label、[PR117] `High-Precision Dichotomous Image Segmentation via Depth Integrity-Prior and Fine-Grained Patch Strategy`。

## 阶段 4：最小 Oracle 门控实现与等价性门禁

### ACU 4.1：仅作用于 depth geometry contribution 的 Oracle gate

**锚点实体：** Oracle corruption mask、DFormerv2 GSA、depth geometry contribution、spatial contribution、硬屏蔽、连续衰减、四个 GSA stage、[PR070] `DFormerv2: Geometry Self-Attention for RGBD Semantic Segmentation`、[PR029] `Uncertainty-Aware Modality Fusion for Unaligned RGB-T Salient Object Detection`、[PR089] `SGMA: Semantic-Guided Modality-Aware Segmentation for Remote Sensing with Incomplete Multimodal Data`。

### ACU 4.2：clean、q=0 与全可信 mask 的数值等价

**锚点实体：** clean、q=0、全可信 mask、原始模型、数值等价、FP32、输出有限性、checkpoint、DFormerv2、Oracle-gated、[PR070] `DFormerv2: Geometry Self-Attention for RGBD Semantic Segmentation`。

## 阶段 5：1–2 张图的三条件 preflight

### ACU 5.1：小样本端到端实现审计

**锚点实体：** 1–2 张图、`clean`、`boundary-q75`、`nonboundary-q50`、五尺度、水平翻转、四个 GSA stage、输出有限性、原始 Label 网格、mask 哈希、GSA stage 传播摘要、RGB-D 对齐、[RE095] `Correcting time offsets and enclosure-induced measurement distortions in LiDAR-camera systems`。

## 阶段 6：完整 paired development evaluation

### ACU 6.1：三组比较、五 condition 与指标实现

**锚点实体：** clean baseline、corrupted baseline、Oracle-gated、`clean`、`boundary-q25`、`boundary-q50`、`boundary-q75`、`nonboundary-q50`、218 张图、138 个 location group、Boundary IoU、mIoU、one-vs-rest、raw background、true ignore、原始 Label 网格、[RE326] `MUSeg: A multimodal semantic segmentation dataset for complex underground mine scenes`、[PR090] `Towards Robust Multi-Modal Semantic Segmentation with Teacher-Student Framework and Hybrid Prototype Distillation`。

## 阶段 7：location-group bootstrap、裁决与收口

### ACU 7.1：组级配对聚合与 percentile bootstrap

**锚点实体：** location group、138 个组、paired bootstrap、cluster bootstrap、bootstrap seed `20260908`、10,000 次有放回重采样、双侧 95% percentile interval、`Oracle-gated - corrupted baseline`、`boundary-q50 - nonboundary-q50` Oracle 增益差、六个 mine。

### ACU 7.2：预注册裁决状态机与证据收口

**锚点实体：** `oracle-supported`、`oracle-not-supported`、`inconclusive`、`protocol-blocked`、`stop`、预先冻结的最小效应量、clean 不劣容忍度、Boundary IoU、mIoU、condition JSON、mask 哈希、GSA stage 传播摘要、`official_test_included=false`。

> **编号处理说明：** 原方案中的 RE/PR 编号保留在“锚点实体”中，便于回查内部文献；它们已从全部外部检索式中排除。对应论文标题在确实能承担机制、数据或协议锚点时保留；仅提供应用背景、但不直接提供 B1 实现的 LFR-CMT-3D、M-SURE、地下语义建图等论文不作为核心精准检索词。

# 第三部分：每个 ACU 的动态化检索策略

## 阶段 1：证据身份核验与 protocol 冻结准备

### ACU 1.1：冻结输入与运行身份的可复核绑定

**锚点实体：** DFormerv2-S RGB Quick-B0、epoch 420 checkpoint、MUSeg、`val-dev`、218 张图、138 个 location group、`Depth16`、v3 corruption mask、mask manifest、五个 condition、五尺度原图/水平翻转、原始 Label 网格、`official_test_included=false`、SHA-256、[RE326] `MUSeg: A multimodal semantic segmentation dataset for complex underground mine scenes`。

**检索目标：** 找到可直接复用的实验身份清单、数据 manifest、文件哈希、checkpoint provenance 和不可变运行配置实现，使模型、数据、mask、condition 与 evaluator 能在运行前和运行后机械核对。

**关键词模块定义：**

**模块1 - 模型与 checkpoint 身份**

- 核心实体与锚点: DFormerv2-S、epoch 420 checkpoint、checkpoint identity、model provenance
- 扩展语义群: `["DFormerv2-S", "DFormerv2", "epoch 420 checkpoint", "checkpoint identit*", "model provenance", "weight hash*", "model artifact*", "reproducib* checkpoint*"]`
- 检索式: `TITLE-ABS-KEY("DFormerv2-S" OR "DFormerv2" OR "epoch 420 checkpoint" OR "checkpoint identit*" OR "model provenance" OR "weight hash*" OR "model artifact*" OR "reproducib* checkpoint*")`

**模块2 - 数据、mask 与 condition 身份**

- 核心实体与锚点: MUSeg、Depth16、corruption mask、mask manifest、condition manifest
- 扩展语义群: `["MUSeg", "Depth16", "corruption mask", "mask manifest", "condition manifest", "dataset manifest", "sample allowlist", "data lineage", "dataset version*", "mask hash*"]`
- 检索式: `TITLE-ABS-KEY("MUSeg" OR "Depth16" OR "corruption mask" OR "mask manifest" OR "condition manifest" OR "dataset manifest" OR "sample allowlist" OR "data lineage" OR "dataset version*" OR "mask hash*")`

**模块3 - 可复现实验身份**

- 核心实体与锚点: SHA-256、official test excluded、multi-scale flip evaluation、original Label grid
- 扩展语义群: `["SHA-256", "cryptographic hash*", "experiment manifest", "run manifest", "artifact integrit*", "multi-scale flip", "original resolution evaluat*", "sealed test set", "preregister* protocol"]`
- 检索式: `TITLE-ABS-KEY("SHA-256" OR "cryptographic hash*" OR "experiment manifest" OR "run manifest" OR "artifact integrit*" OR "multi-scale flip" OR "original resolution evaluat*" OR "sealed test set" OR "preregister* protocol")`

**资源锚点模块**

- 推断资源类型: 代码类、数据类、方法/协议类
- 锚点关键词: `"github" OR "open source" OR "code available" OR "repository" OR "dataset manifest" OR "data available" OR "protocol" OR "reproducibility checklist" OR "configuration file"`
- 检索式: `TITLE-ABS-KEY("github" OR "open source" OR "code available" OR "repository" OR "dataset manifest" OR "data available" OR "protocol" OR "reproducibility checklist" OR "configuration file")`

**组合策略菜单：**

- **策略一：精准打击（高精确度）**
  - 逻辑组合: 模块1 AND 模块2 AND 模块3
  - 说明: 优先寻找同时说明模型权重、数据版本和运行身份绑定的实现。
  - 检索式: `TITLE-ABS-KEY(("DFormerv2-S" OR "DFormerv2") AND ("MUSeg" OR "Depth16" OR "corruption mask") AND ("experiment manifest" OR "artifact integrit*" OR "cryptographic hash*"))`
- **策略二：实现优先（高相关性+可操作性）**
  - 逻辑组合: (模块1 AND 模块2 AND 模块3) AND 资源锚点模块
  - 说明: 优先返回带仓库、manifest、配置或复现清单的材料。
  - 检索式: `TITLE-ABS-KEY(("DFormerv2" AND "MUSeg") AND ("dataset manifest" OR "checkpoint identit*" OR "run manifest") AND ("github" OR "repository" OR "code available" OR "protocol"))`
- **策略三：双核探索（中等范围）**
  - 逻辑组合: 模块2 AND 模块3 AND 资源锚点模块
  - 说明: DFormerv2 精确命中不足时，转搜通用视觉实验的数据与产物身份方案。
  - 检索式: `TITLE-ABS-KEY(("dataset manifest" OR "mask manifest" OR "data lineage") AND ("artifact integrit*" OR "cryptographic hash*" OR "run manifest") AND ("github" OR "open source" OR "protocol"))`

### ACU 1.2：预注册效应量、不劣界值与统计协议

**锚点实体：** Boundary IoU、mIoU、`boundary-q75`、clean baseline、corrupted baseline、Oracle-gated、location group、paired bootstrap、95% percentile interval、seed `20260908`、10,000 次重采样、六个 mine、合法终点、[RE131] `RGB-D semantic mapping for underground robotic inspection using an attention-enhanced and boundary-refined DeepLabv3+ network`、[PR090] `Towards Robust Multi-Modal Semantic Segmentation with Teacher-Student Framework and Hybrid Prototype Distillation`。

**检索目标：** 找到与 RGB-D 分割退化鲁棒性最接近的效应量报告、clean performance retention、不劣性判据和组级配对置信区间协议，用来在完整评价前冻结 B1 的裁决门槛，而不是照搬无关领域常数。

**关键词模块定义：**

**模块1 - RGB-D 退化鲁棒性与 clean 保持**

- 核心实体与锚点: RGB-D semantic segmentation、degraded modality、clean performance、modality robustness
- 扩展语义群: `["RGB-D semantic segmentation", "multimodal semantic segmentation", "degraded depth", "corrupted depth", "missing modalit*", "modality robust*", "clean performance", "full-modality performance", "performance retention", "non-inferior*"]`
- 检索式: `TITLE-ABS-KEY("RGB-D semantic segmentation" OR "multimodal semantic segmentation" OR "degraded depth" OR "corrupted depth" OR "missing modalit*" OR "modality robust*" OR "clean performance" OR "full-modality performance" OR "performance retention" OR "non-inferior*")`

**模块2 - 边界与任务效应量**

- 核心实体与锚点: Boundary IoU、mIoU、boundary accuracy、effect size
- 扩展语义群: `["Boundary IoU", "boundary intersection over union", "mean intersection over union", "mIoU", "boundary accur*", "segmentation boundary metric*", "effect size", "minimum important difference", "practical significance"]`
- 检索式: `TITLE-ABS-KEY("Boundary IoU" OR "boundary intersection over union" OR "mean intersection over union" OR "mIoU" OR "boundary accur*" OR "segmentation boundary metric*" OR "effect size" OR "minimum important difference" OR "practical significance")`

**模块3 - 预注册与组级配对推断**

- 核心实体与锚点: location group、paired bootstrap、percentile interval、preregistered decision
- 扩展语义群: `["paired bootstrap*", "cluster bootstrap*", "grouped bootstrap*", "hierarchical bootstrap*", "percentile interval", "clustered observation*", "paired image*", "preregister*", "decision threshold*"]`
- 检索式: `TITLE-ABS-KEY("paired bootstrap*" OR "cluster bootstrap*" OR "grouped bootstrap*" OR "hierarchical bootstrap*" OR "percentile interval" OR "clustered observation*" OR "paired image*" OR "preregister*" OR "decision threshold*")`

**资源锚点模块**

- 推断资源类型: 方法/协议类、代码类
- 锚点关键词: `"protocol" OR "preregistration" OR "statistical analysis plan" OR "supplementary material" OR "github" OR "code available" OR "bootstrap implementation"`
- 检索式: `TITLE-ABS-KEY("protocol" OR "preregistration" OR "statistical analysis plan" OR "supplementary material" OR "github" OR "code available" OR "bootstrap implementation")`

**组合策略菜单：**

- **策略一：精准打击（高精确度）**
  - 逻辑组合: 模块1 AND 模块2 AND 模块3
  - 说明: 找同时报告多模态退化、边界指标和配对区间的近邻协议。
  - 检索式: `TITLE-ABS-KEY(("RGB-D semantic segmentation" OR "multimodal semantic segmentation") AND ("Boundary IoU" OR "boundary metric*") AND ("paired bootstrap*" OR "cluster bootstrap*" OR "preregister*"))`
- **策略二：实现优先（高相关性+可操作性）**
  - 逻辑组合: (模块1 AND 模块2 AND 模块3) AND 资源锚点模块
  - 说明: 优先寻找明确给出补充协议、统计代码或门槛定义的论文。
  - 检索式: `TITLE-ABS-KEY(("modality robust*" OR "degraded depth") AND ("Boundary IoU" OR "mean intersection over union") AND ("cluster bootstrap*" OR "non-inferior*" OR "performance retention") AND ("protocol" OR "supplementary material" OR "code available"))`
- **策略三：双核探索（中等范围）**
  - 逻辑组合: 模块1 AND 模块3 AND 资源锚点模块
  - 说明: 边界指标命中过少时，先找多模态 robustness 的 clean-retention 与统计设计。
  - 检索式: `TITLE-ABS-KEY(("multimodal semantic segmentation" OR "missing modalit*" OR "corrupted depth") AND ("clean performance" OR "non-inferior*" OR "paired bootstrap*") AND ("protocol" OR "supplementary material" OR "github"))`

## 阶段 2：DFormerv2 GSA 机制定位

### ACU 2.1：定位并证明 GSA 两类 contribution 可分离

**锚点实体：** DFormerv2、DFormerv2-S、Geometry Self-Attention、spatial contribution、depth geometry contribution、depth geometry prior、四个 GSA stage、[PR070] `DFormerv2: Geometry Self-Attention for RGBD Semantic Segmentation`。

**检索目标：** 找到 DFormerv2 官方论文、官方仓库、补充材料及可信复现中 attention logits 或 attention weight 的准确计算位置，明确 spatial 与 depth-derived geometry 两项的张量形状、合成顺序和四级调用链。

**关键词模块定义：**

**模块1 - 模型与官方机制**

- 核心实体与锚点: DFormerv2、Geometry Self-Attention、RGB-D semantic segmentation
- 扩展语义群: `["DFormerv2", "DFormerv2-S", "DFormerv2: Geometry Self-Attention for RGBD Semantic Segmentation", "Geometry Self-Attention", "RGB-D semantic segmentation", "depth geometry prior"]`
- 检索式: `TITLE-ABS-KEY("DFormerv2" OR "DFormerv2-S" OR "DFormerv2: Geometry Self-Attention for RGBD Semantic Segmentation" OR "Geometry Self-Attention" OR "RGB-D semantic segmentation" OR "depth geometry prior")`

**模块2 - contribution 张量分解**

- 核心实体与锚点: spatial contribution、depth geometry contribution、attention logits、geometry bias
- 扩展语义群: `["spatial contribution", "depth geometry contribution", "geometric contribution", "attention logit*", "attention bias", "geometry bias", "depth-derived attention", "relative position bias", "attention decomposition"]`
- 检索式: `TITLE-ABS-KEY("spatial contribution" OR "depth geometry contribution" OR "geometric contribution" OR "attention logit*" OR "attention bias" OR "geometry bias" OR "depth-derived attention" OR "relative position bias" OR "attention decomposition")`

**模块3 - 源码定位与张量形状**

- 核心实体与锚点: four-stage encoder、tensor shape、forward implementation
- 扩展语义群: `["four-stage encoder", "multi-stage transformer", "tensor shape", "forward pass", "attention implement*", "geometry prior implement*", "source code", "official implementation"]`
- 检索式: `TITLE-ABS-KEY("four-stage encoder" OR "multi-stage transformer" OR "tensor shape" OR "forward pass" OR "attention implement*" OR "geometry prior implement*" OR "source code" OR "official implementation")`

**资源锚点模块**

- 推断资源类型: 代码类、方法/协议类
- 锚点关键词: `"official code" OR "github" OR "repository" OR "source code" OR "implementation" OR "supplementary material" OR "pseudocode"`
- 检索式: `TITLE-ABS-KEY("official code" OR "github" OR "repository" OR "source code" OR "implementation" OR "supplementary material" OR "pseudocode")`

**组合策略菜单：**

- **策略一：精准打击（高精确度）**
  - 逻辑组合: 模块1 AND 模块2 AND 模块3
  - 说明: 直接定位 DFormerv2 中 contribution 合成公式与 forward 路径。
  - 检索式: `TITLE-ABS-KEY(("DFormerv2" OR "Geometry Self-Attention") AND ("depth geometry contribution" OR "geometry bias" OR "attention logit*") AND ("forward pass" OR "tensor shape" OR "implement*"))`
- **策略二：实现优先（高相关性+可操作性）**
  - 逻辑组合: (模块1 AND 模块2 AND 模块3) AND 资源锚点模块
  - 说明: 优先获取官方仓库、补充伪代码和可跟踪张量实现。
  - 检索式: `TITLE-ABS-KEY(("DFormerv2" AND "Geometry Self-Attention") AND ("geometry bias" OR "depth-derived attention" OR "attention decomposition") AND ("official code" OR "github" OR "source code" OR "supplementary material"))`
- **策略三：双核探索（中等范围）**
  - 逻辑组合: 模块1 AND 模块2 AND 资源锚点模块
  - 说明: 官方描述不足时，扩展到与 DFormerv2 同构的 geometry-biased attention 实现。
  - 检索式: `TITLE-ABS-KEY(("Geometry Self-Attention" OR "depth geometry prior") AND ("attention bias" OR "geometry bias" OR "depth-derived attention") AND ("github" OR "source code" OR "implementation"))`

## 阶段 3：Oracle mask 的多尺度传播协议

### ACU 3.1：原始像素 mask 到四级 GSA 网格的确定性投影

**锚点实体：** 原始 `Depth16`、corruption mask、四个 GSA stage、patch pooling、下采样、resize、depth geometry prior、全可信 mask、condition-specific mask、[PR070] `DFormerv2: Geometry Self-Attention for RGBD Semantic Segmentation`。

**检索目标：** 找到二值无效区或置信图从输入像素网格传播到多级 patch/token 网格的可审计实现，重点比较 nearest resize、max/average pooling、有效比例聚合及其边界保守性，并要求输出规则对全可信 mask 保持恒等。

**关键词模块定义：**

**模块1 - 多尺度 mask 投影**

- 核心实体与锚点: corruption mask、four-stage feature map、mask downsampling
- 扩展语义群: `["corruption mask", "validity mask", "confidence mask", "reliability map", "multi-scale mask", "mask downsampl*", "feature pyramid mask", "stage-wise mask", "token mask"]`
- 检索式: `TITLE-ABS-KEY("corruption mask" OR "validity mask" OR "confidence mask" OR "reliability map" OR "multi-scale mask" OR "mask downsampl*" OR "feature pyramid mask" OR "stage-wise mask" OR "token mask")`

**模块2 - patch 聚合与插值规则**

- 核心实体与锚点: patch pooling、resize、downsampling、invalid pixel fraction
- 扩展语义群: `["patch pooling", "mask pooling", "max pool* mask", "average pool* mask", "nearest-neighbor resiz*", "area interpolat*", "invalid pixel fraction", "patch validit*", "conservative downsampl*"]`
- 检索式: `TITLE-ABS-KEY("patch pooling" OR "mask pooling" OR "max pool* mask" OR "average pool* mask" OR "nearest-neighbor resiz*" OR "area interpolat*" OR "invalid pixel fraction" OR "patch validit*" OR "conservative downsampl*")`

**模块3 - 深度几何先验的掩膜化**

- 核心实体与锚点: depth geometry prior、invalid depth、masked attention
- 扩展语义群: `["depth geometry prior", "invalid depth", "missing depth", "depth confidence", "masked attention", "attention mask", "geometry-aware attention", "depth reliabilit*", "partial validit*"]`
- 检索式: `TITLE-ABS-KEY("depth geometry prior" OR "invalid depth" OR "missing depth" OR "depth confidence" OR "masked attention" OR "attention mask" OR "geometry-aware attention" OR "depth reliabilit*" OR "partial validit*")`

**资源锚点模块**

- 推断资源类型: 代码类、方法/协议类
- 锚点关键词: `"github" OR "code available" OR "implementation" OR "algorithm" OR "pseudocode" OR "ablation" OR "supplementary material"`
- 检索式: `TITLE-ABS-KEY("github" OR "code available" OR "implementation" OR "algorithm" OR "pseudocode" OR "ablation" OR "supplementary material")`

**组合策略菜单：**

- **策略一：精准打击（高精确度）**
  - 逻辑组合: 模块1 AND 模块2 AND 模块3
  - 说明: 找深度可靠性 mask 在层级注意力网络中的具体降采样方式。
  - 检索式: `TITLE-ABS-KEY(("validity mask" OR "depth confidence" OR "corruption mask") AND ("mask downsampl*" OR "patch pooling" OR "invalid pixel fraction") AND ("masked attention" OR "depth geometry prior" OR "geometry-aware attention"))`
- **策略二：实现优先（高相关性+可操作性）**
  - 逻辑组合: (模块1 AND 模块2 AND 模块3) AND 资源锚点模块
  - 说明: 只优先保留明确展示 pooling、resize 或 token mask 代码的来源。
  - 检索式: `TITLE-ABS-KEY(("multi-scale mask" OR "stage-wise mask") AND ("mask pooling" OR "nearest-neighbor resiz*" OR "area interpolat*") AND ("depth reliabilit*" OR "masked attention") AND ("github" OR "code available" OR "pseudocode"))`
- **策略三：双核探索（中等范围）**
  - 逻辑组合: 模块1 AND 模块2 AND 资源锚点模块
  - 说明: 深度专用结果过少时，扩展到分割、稀疏观测和层级 Transformer 的通用 mask 传播。
  - 检索式: `TITLE-ABS-KEY(("validity mask" OR "confidence mask" OR "token mask") AND ("mask downsampl*" OR "patch pooling" OR "feature pyramid mask") AND ("github" OR "implementation" OR "algorithm"))`

### ACU 3.2：水平翻转与多尺度 view 的 mask 对齐

**锚点实体：** 五尺度、原图、水平翻转、mask resize、flip view、原始 Label 网格、logits 融合、RGB、Depth、Label、[PR117] `High-Precision Dichotomous Image Segmentation via Depth Integrity-Prior and Fine-Grained Patch Strategy`。

**检索目标：** 找到 RGB、Depth、Label 与 mask 在 multi-scale flip test-time augmentation 中共享同一几何变换、逆变换 logits 并恢复原始 metric grid 的实现，排除 mask 在翻转、padding 或插值后错位。

**关键词模块定义：**

**模块1 - 多模态同步几何变换**

- 核心实体与锚点: RGB-D、mask alignment、synchronized transform
- 扩展语义群: `["RGB-D", "RGB depth alignment", "synchronized transform*", "paired transform*", "mask alignment", "label-preserving transform*", "multimodal augmentation", "geometric consistency"]`
- 检索式: `TITLE-ABS-KEY("RGB-D" OR "RGB depth alignment" OR "synchronized transform*" OR "paired transform*" OR "mask alignment" OR "label-preserving transform*" OR "multimodal augmentation" OR "geometric consistency")`

**模块2 - multi-scale flip 推理**

- 核心实体与锚点: five-scale、horizontal flip、test-time augmentation、logits fusion
- 扩展语义群: `["multi-scale flip", "horizontal flip inference", "test-time augmentation", "multi-view inference", "flip equivarian*", "inverse transform* logit*", "logit* fusion", "multi-scale semantic segmentation"]`
- 检索式: `TITLE-ABS-KEY("multi-scale flip" OR "horizontal flip inference" OR "test-time augmentation" OR "multi-view inference" OR "flip equivarian*" OR "inverse transform* logit*" OR "logit* fusion" OR "multi-scale semantic segmentation")`

**模块3 - 原始网格恢复**

- 核心实体与锚点: original Label grid、resize、padding、metric geometry
- 扩展语义群: `["original resolution", "original image grid", "label grid", "metric grid", "prediction resiz*", "padding removal", "spatial correspondence", "segmentation evaluat* geometry"]`
- 检索式: `TITLE-ABS-KEY("original resolution" OR "original image grid" OR "label grid" OR "metric grid" OR "prediction resiz*" OR "padding removal" OR "spatial correspondence" OR "segmentation evaluat* geometry")`

**资源锚点模块**

- 推断资源类型: 代码类、方法/协议类
- 锚点关键词: `"github" OR "official implementation" OR "inference script" OR "evaluation code" OR "test-time augmentation code" OR "protocol"`
- 检索式: `TITLE-ABS-KEY("github" OR "official implementation" OR "inference script" OR "evaluation code" OR "test-time augmentation code" OR "protocol")`

**组合策略菜单：**

- **策略一：精准打击（高精确度）**
  - 逻辑组合: 模块1 AND 模块2 AND 模块3
  - 说明: 查找 RGB-D 分割中同步 mask、flip 逆变换与原始网格计分的完整链路。
  - 检索式: `TITLE-ABS-KEY(("RGB-D" OR "multimodal augmentation") AND ("multi-scale flip" OR "test-time augmentation") AND ("mask alignment" OR "original resolution" OR "spatial correspondence"))`
- **策略二：实现优先（高相关性+可操作性）**
  - 逻辑组合: (模块1 AND 模块2 AND 模块3) AND 资源锚点模块
  - 说明: 优先寻找可检查 transform 顺序、插值模式和逆变换的推理代码。
  - 检索式: `TITLE-ABS-KEY(("paired transform*" OR "RGB depth alignment") AND ("horizontal flip inference" OR "inverse transform* logit*") AND ("original image grid" OR "padding removal") AND ("github" OR "inference script" OR "evaluation code"))`
- **策略三：双核探索（中等范围）**
  - 逻辑组合: 模块2 AND 模块3 AND 资源锚点模块
  - 说明: 多模态专用结果不足时，先固定通用语义分割的 view 逆变换与 metric-grid 实现。
  - 检索式: `TITLE-ABS-KEY(("multi-scale flip" OR "test-time augmentation") AND ("original resolution" OR "prediction resiz*" OR "logit* fusion") AND ("official implementation" OR "evaluation code" OR "github"))`

## 阶段 4：最小 Oracle 门控实现与等价性门禁

### ACU 4.1：仅作用于 depth geometry contribution 的 Oracle gate

**锚点实体：** Oracle corruption mask、DFormerv2 GSA、depth geometry contribution、spatial contribution、硬屏蔽、连续衰减、四个 GSA stage、[PR070] `DFormerv2: Geometry Self-Attention for RGBD Semantic Segmentation`、[PR029] `Uncertainty-Aware Modality Fusion for Unaligned RGB-T Salient Object Detection`、[PR089] `SGMA: Semantic-Guided Modality-Aware Segmentation for Remote Sensing with Incomplete Multimodal Data`。

**检索目标：** 找到可把已知像素/patch 可靠性直接作用于几何 bias、attention logits 或跨模态 contribution 的最小代码模式，同时确保 spatial contribution 与其他分支不变；检索结果用于选择一个 Oracle 形式，不用于并行调多种门控自由度。

**关键词模块定义：**

**模块1 - DFormerv2 几何 contribution 门控**

- 核心实体与锚点: DFormerv2、Geometry Self-Attention、depth geometry contribution
- 扩展语义群: `["DFormerv2", "Geometry Self-Attention", "depth geometry contribution", "depth geometry prior", "geometry bias", "geometry-aware attention", "depth-guided attention"]`
- 检索式: `TITLE-ABS-KEY("DFormerv2" OR "Geometry Self-Attention" OR "depth geometry contribution" OR "depth geometry prior" OR "geometry bias" OR "geometry-aware attention" OR "depth-guided attention")`

**模块2 - 局部可靠性与条件式门控**

- 核心实体与锚点: Oracle corruption mask、reliability gating、modality-aware fusion
- 扩展语义群: `["oracle mask", "ground-truth corruption mask", "reliability gate*", "confidence gate*", "quality-aware fusion", "uncertainty-aware fusion", "modality-aware fusion", "conditional modality weighting", "local reliabilit*"]`
- 检索式: `TITLE-ABS-KEY("oracle mask" OR "ground-truth corruption mask" OR "reliability gate*" OR "confidence gate*" OR "quality-aware fusion" OR "uncertainty-aware fusion" OR "modality-aware fusion" OR "conditional modality weighting" OR "local reliabilit*")`

**模块3 - 最小算子与隔离性**

- 核心实体与锚点: hard masking、continuous attenuation、spatial contribution unchanged
- 扩展语义群: `["hard mask*", "soft gate*", "continuous attenuation", "multiplicative mask*", "additive attention mask", "attention bias mask*", "branch isolation", "frozen backbone", "no retraining"]`
- 检索式: `TITLE-ABS-KEY("hard mask*" OR "soft gate*" OR "continuous attenuation" OR "multiplicative mask*" OR "additive attention mask" OR "attention bias mask*" OR "branch isolation" OR "frozen backbone" OR "no retraining")`

**资源锚点模块**

- 推断资源类型: 代码类、方法/协议类
- 锚点关键词: `"github" OR "official code" OR "implementation" OR "source code" OR "ablation" OR "supplementary material" OR "pseudocode"`
- 检索式: `TITLE-ABS-KEY("github" OR "official code" OR "implementation" OR "source code" OR "ablation" OR "supplementary material" OR "pseudocode")`

**组合策略菜单：**

- **策略一：精准打击（高精确度）**
  - 逻辑组合: 模块1 AND 模块2 AND 模块3
  - 说明: 搜索几何注意力中按局部深度可靠性屏蔽单一 contribution 的实现。
  - 检索式: `TITLE-ABS-KEY(("DFormerv2" OR "Geometry Self-Attention" OR "depth geometry prior") AND ("reliability gate*" OR "oracle mask" OR "uncertainty-aware fusion") AND ("multiplicative mask*" OR "attention bias mask*" OR "frozen backbone"))`
- **策略二：实现优先（高相关性+可操作性）**
  - 逻辑组合: (模块1 AND 模块2 AND 模块3) AND 资源锚点模块
  - 说明: 优先找到能直接迁移到 forward 路径的硬/软 mask 代码，而非只给概念框图的论文。
  - 检索式: `TITLE-ABS-KEY(("geometry-aware attention" OR "depth-guided attention") AND ("local reliabilit*" OR "quality-aware fusion") AND ("hard mask*" OR "soft gate*" OR "additive attention mask") AND ("official code" OR "github" OR "source code"))`
- **策略三：双核探索（中等范围）**
  - 逻辑组合: 模块2 AND 模块3 AND 资源锚点模块
  - 说明: DFormerv2 专属结果不足时，扩展到 RGB-T、遥感和缺失模态分割中的局部可靠性门控实现。
  - 检索式: `TITLE-ABS-KEY(("uncertainty-aware fusion" OR "modality-aware fusion" OR "conditional modality weighting") AND ("multiplicative mask*" OR "soft gate*" OR "frozen backbone") AND ("github" OR "code available" OR "implementation"))`

### ACU 4.2：clean、q=0 与全可信 mask 的数值等价

**锚点实体：** clean、q=0、全可信 mask、原始模型、数值等价、FP32、输出有限性、checkpoint、DFormerv2、Oracle-gated、[PR070] `DFormerv2: Geometry Self-Attention for RGBD Semantic Segmentation`。

**检索目标：** 找到神经网络旁路、恒等 mask 和干预关闭状态的数值回归测试模式，定义可复核的绝对/相对容差、确定性设置和逐 stage/最终 logits 对比，证明 gate 关闭时没有改变原模型。

**关键词模块定义：**

**模块1 - 恒等与旁路等价**

- 核心实体与锚点: all-trusted mask、identity gate、bypass equivalence
- 扩展语义群: `["identity gate", "all-one mask", "all-valid mask", "all-trusted mask", "bypass equivalence", "no-op path", "intervention off", "functional equivalence", "regression test*"]`
- 检索式: `TITLE-ABS-KEY("identity gate" OR "all-one mask" OR "all-valid mask" OR "all-trusted mask" OR "bypass equivalence" OR "no-op path" OR "intervention off" OR "functional equivalence" OR "regression test*")`

**模块2 - 浮点数值一致性**

- 核心实体与锚点: FP32、numerical equivalence、tolerance、finite outputs
- 扩展语义群: `["FP32", "float32", "numerical equivalence", "numerical consistency", "absolute tolerance", "relative tolerance", "floating-point reproducib*", "deterministic inference", "finite output*", "NaN detection"]`
- 检索式: `TITLE-ABS-KEY("FP32" OR "float32" OR "numerical equivalence" OR "numerical consistency" OR "absolute tolerance" OR "relative tolerance" OR "floating-point reproducib*" OR "deterministic inference" OR "finite output*" OR "NaN detection")`

**模块3 - 深度注意力回归测试**

- 核心实体与锚点: DFormerv2、attention output、stage-wise comparison
- 扩展语义群: `["DFormerv2", "attention output", "attention regression test*", "stage-wise comparison", "intermediate activation*", "logit equivalence", "model instrumentation", "forward hook*"]`
- 检索式: `TITLE-ABS-KEY("DFormerv2" OR "attention output" OR "attention regression test*" OR "stage-wise comparison" OR "intermediate activation*" OR "logit equivalence" OR "model instrumentation" OR "forward hook*")`

**资源锚点模块**

- 推断资源类型: 代码类、方法/协议类
- 锚点关键词: `"unit test" OR "regression test" OR "github" OR "test code" OR "reproducibility" OR "deterministic algorithm" OR "validation protocol"`
- 检索式: `TITLE-ABS-KEY("unit test" OR "regression test" OR "github" OR "test code" OR "reproducibility" OR "deterministic algorithm" OR "validation protocol")`

**组合策略菜单：**

- **策略一：精准打击（高精确度）**
  - 逻辑组合: 模块1 AND 模块2 AND 模块3
  - 说明: 寻找恒等门控下逐层 activation 与最终 logits 数值比较的方法。
  - 检索式: `TITLE-ABS-KEY(("identity gate" OR "all-valid mask" OR "no-op path") AND ("numerical equivalence" OR "absolute tolerance" OR "deterministic inference") AND ("stage-wise comparison" OR "intermediate activation*" OR "logit equivalence"))`
- **策略二：实现优先（高相关性+可操作性）**
  - 逻辑组合: (模块1 AND 模块2 AND 模块3) AND 资源锚点模块
  - 说明: 优先定位带具体 tolerance、assertion 和中间张量抓取代码的测试。
  - 检索式: `TITLE-ABS-KEY(("all-one mask" OR "bypass equivalence") AND ("FP32" OR "numerical consistency") AND ("forward hook*" OR "attention regression test*") AND ("unit test" OR "test code" OR "github"))`
- **策略三：双核探索（中等范围）**
  - 逻辑组合: 模块1 AND 模块2 AND 资源锚点模块
  - 说明: 模型专属资料不足时，借鉴神经网络 no-op 分支的数值回归测试。
  - 检索式: `TITLE-ABS-KEY(("no-op path" OR "functional equivalence" OR "intervention off") AND ("floating-point reproducib*" OR "relative tolerance" OR "finite output*") AND ("regression test" OR "test code" OR "validation protocol"))`

## 阶段 5：1–2 张图的三条件 preflight

### ACU 5.1：小样本端到端实现审计

**锚点实体：** 1–2 张图、`clean`、`boundary-q75`、`nonboundary-q50`、五尺度、水平翻转、四个 GSA stage、输出有限性、原始 Label 网格、mask 哈希、GSA stage 传播摘要、RGB-D 对齐、[RE095] `Correcting time offsets and enclosure-induced measurement distortions in LiDAR-camera systems`。

**检索目标：** 找到多模态视觉推理在正式运行前的 smoke test/preflight 检查清单和可机读证据格式，覆盖输入身份、空间对齐、各 stage mask 统计、有限值、输出 shape、逆变换与 metric-grid 恢复，但不扩大为完整测试套件。

**关键词模块定义：**

**模块1 - RGB-D 对齐与输入审计**

- 核心实体与锚点: RGB-D alignment、sensor registration、projection error
- 扩展语义群: `["RGB-D alignment", "RGB depth registration", "sensor alignment", "cross-modal alignment", "projection error", "calibration error", "time offset", "measurement distortion", "input audit"]`
- 检索式: `TITLE-ABS-KEY("RGB-D alignment" OR "RGB depth registration" OR "sensor alignment" OR "cross-modal alignment" OR "projection error" OR "calibration error" OR "time offset" OR "measurement distortion" OR "input audit")`

**模块2 - 推理 preflight 与有限性**

- 核心实体与锚点: preflight、smoke test、finite output、shape validation
- 扩展语义群: `["inference preflight", "smoke test", "sanity check", "finite output*", "NaN check", "shape validation", "tensor invariant*", "pipeline validation", "fail-fast"]`
- 检索式: `TITLE-ABS-KEY("inference preflight" OR "smoke test" OR "sanity check" OR "finite output*" OR "NaN check" OR "shape validation" OR "tensor invariant*" OR "pipeline validation" OR "fail-fast")`

**模块3 - 可审计运行证据**

- 核心实体与锚点: mask hash、stage propagation summary、original Label grid
- 扩展语义群: `["mask hash", "stage-wise summar*", "run manifest", "structured log*", "provenance record", "audit trail", "original resolution output", "machine-readable evidence", "JSON report"]`
- 检索式: `TITLE-ABS-KEY("mask hash" OR "stage-wise summar*" OR "run manifest" OR "structured log*" OR "provenance record" OR "audit trail" OR "original resolution output" OR "machine-readable evidence" OR "JSON report")`

**资源锚点模块**

- 推断资源类型: 代码类、方法/协议类
- 锚点关键词: `"checklist" OR "validation script" OR "github" OR "test code" OR "logging framework" OR "protocol" OR "example configuration"`
- 检索式: `TITLE-ABS-KEY("checklist" OR "validation script" OR "github" OR "test code" OR "logging framework" OR "protocol" OR "example configuration")`

**组合策略菜单：**

- **策略一：精准打击（高精确度）**
  - 逻辑组合: 模块1 AND 模块2 AND 模块3
  - 说明: 定位既查 RGB-D 对齐，又查推理张量和证据记录的 preflight 流程。
  - 检索式: `TITLE-ABS-KEY(("RGB-D alignment" OR "cross-modal alignment") AND ("inference preflight" OR "smoke test" OR "pipeline validation") AND ("run manifest" OR "stage-wise summar*" OR "original resolution output"))`
- **策略二：实现优先（高相关性+可操作性）**
  - 逻辑组合: (模块1 AND 模块2 AND 模块3) AND 资源锚点模块
  - 说明: 优先寻找可直接转成 1–2 样本检查脚本和 JSON 证据的材料。
  - 检索式: `TITLE-ABS-KEY(("RGB depth registration" OR "input audit") AND ("finite output*" OR "shape validation" OR "fail-fast") AND ("machine-readable evidence" OR "JSON report") AND ("validation script" OR "test code" OR "github"))`
- **策略三：双核探索（中等范围）**
  - 逻辑组合: 模块2 AND 模块3 AND 资源锚点模块
  - 说明: 对齐论文不足以提供工程实现时，转搜机器学习推理链的 fail-fast 与 provenance 模式。
  - 检索式: `TITLE-ABS-KEY(("inference preflight" OR "tensor invariant*" OR "pipeline validation") AND ("structured log*" OR "audit trail" OR "run manifest") AND ("validation script" OR "github" OR "protocol"))`

## 阶段 6：完整 paired development evaluation

### ACU 6.1：三组比较、五 condition 与指标实现

**锚点实体：** clean baseline、corrupted baseline、Oracle-gated、`clean`、`boundary-q25`、`boundary-q50`、`boundary-q75`、`nonboundary-q50`、218 张图、138 个 location group、Boundary IoU、mIoU、one-vs-rest、raw background、true ignore、原始 Label 网格、[RE326] `MUSeg: A multimodal semantic segmentation dataset for complex underground mine scenes`、[PR090] `Towards Robust Multi-Modal Semantic Segmentation with Teacher-Student Framework and Hybrid Prototype Distillation`。

**检索目标：** 找到 Boundary IoU 的权威实现和适用于语义分割的 one-vs-rest、background、ignore、empty-class 聚合约定，同时寻找成对多 condition 运行矩阵与逐图结果落盘模式，确保三组比较只有门控变量不同。

**关键词模块定义：**

**模块1 - 三组同源 paired evaluation**

- 核心实体与锚点: clean baseline、corrupted baseline、Oracle-gated、paired condition
- 扩展语义群: `["clean baseline", "corrupted baseline", "oracle-gated", "paired evaluation", "within-image comparison", "controlled intervention", "matched condition*", "ablation protocol", "single-variable ablation"]`
- 检索式: `TITLE-ABS-KEY("clean baseline" OR "corrupted baseline" OR "oracle-gated" OR "paired evaluation" OR "within-image comparison" OR "controlled intervention" OR "matched condition*" OR "ablation protocol" OR "single-variable ablation")`

**模块2 - Boundary IoU 语义实现**

- 核心实体与锚点: Boundary IoU、one-vs-rest、raw background、true ignore
- 扩展语义群: `["Boundary IoU", "boundary intersection over union", "semantic segmentation boundary", "one-vs-rest", "background class", "ignore label", "void label", "empty class", "macro aggregation", "boundary metric implementation"]`
- 检索式: `TITLE-ABS-KEY("Boundary IoU" OR "boundary intersection over union" OR "semantic segmentation boundary" OR "one-vs-rest" OR "background class" OR "ignore label" OR "void label" OR "empty class" OR "macro aggregation" OR "boundary metric implementation")`

**模块3 - MUSeg 与原始网格评价**

- 核心实体与锚点: MUSeg、RGB-D semantic segmentation、original Label grid
- 扩展语义群: `["MUSeg", "MUSeg: A multimodal semantic segmentation dataset for complex underground mine scenes", "underground mine", "RGB-D semantic segmentation", "original resolution evaluat*", "label grid", "mean intersection over union", "mIoU"]`
- 检索式: `TITLE-ABS-KEY("MUSeg" OR "MUSeg: A multimodal semantic segmentation dataset for complex underground mine scenes" OR "underground mine" OR "RGB-D semantic segmentation" OR "original resolution evaluat*" OR "label grid" OR "mean intersection over union" OR "mIoU")`

**资源锚点模块**

- 推断资源类型: 代码类、数据类、方法/协议类
- 锚点关键词: `"official implementation" OR "github" OR "evaluation code" OR "metric code" OR "dataset" OR "annotation protocol" OR "supplementary material"`
- 检索式: `TITLE-ABS-KEY("official implementation" OR "github" OR "evaluation code" OR "metric code" OR "dataset" OR "annotation protocol" OR "supplementary material")`

**组合策略菜单：**

- **策略一：精准打击（高精确度）**
  - 逻辑组合: 模块1 AND 模块2 AND 模块3
  - 说明: 找 RGB-D/MUSeg 语境中可用于三组配对比较的边界指标契约。
  - 检索式: `TITLE-ABS-KEY(("paired evaluation" OR "single-variable ablation") AND ("Boundary IoU" OR "boundary metric implementation") AND ("MUSeg" OR "RGB-D semantic segmentation" OR "original resolution evaluat*"))`
- **策略二：实现优先（高相关性+可操作性）**
  - 逻辑组合: (模块1 AND 模块2 AND 模块3) AND 资源锚点模块
  - 说明: 优先采用能核对 background、ignore 与 empty-class 行为的官方指标代码和数据协议。
  - 检索式: `TITLE-ABS-KEY(("within-image comparison" OR "matched condition*") AND ("Boundary IoU" OR "one-vs-rest") AND ("ignore label" OR "background class" OR "original resolution evaluat*") AND ("official implementation" OR "evaluation code" OR "metric code"))`
- **策略三：双核探索（中等范围）**
  - 逻辑组合: 模块2 AND 模块3 AND 资源锚点模块
  - 说明: 配对设计结果不足时，先锁定 Boundary IoU 和 MUSeg/语义分割 evaluator 的权威实现。
  - 检索式: `TITLE-ABS-KEY(("Boundary IoU" OR "semantic segmentation boundary") AND ("MUSeg" OR "RGB-D semantic segmentation" OR "label grid") AND ("github" OR "metric code" OR "annotation protocol"))`

## 阶段 7：location-group bootstrap、裁决与收口

### ACU 7.1：组级配对聚合与 percentile bootstrap

**锚点实体：** location group、138 个组、paired bootstrap、cluster bootstrap、seed `20260908`、10,000 次有放回重采样、双侧 95% percentile interval、`Oracle-gated - corrupted baseline`、`boundary-q50 - nonboundary-q50` Oracle 增益差、六个 mine。

**检索目标：** 找到“同一组内保留全部图像、按 location group 重采样、条件间保持配对”的聚合与 bootstrap 代码，稳定产出两个预注册 effect 的点估计和双侧 95% percentile interval；六个 mine 只作描述性分层，不伪装成六个独立样本。

**关键词模块定义：**

**模块1 - 组级配对重采样**

- 核心实体与锚点: location group、paired bootstrap、cluster bootstrap
- 扩展语义群: `["paired bootstrap*", "cluster bootstrap*", "grouped bootstrap*", "hierarchical bootstrap*", "cluster resampl*", "paired resampl*", "within-cluster dependenc*", "correlated image*", "group-level resampl*"]`
- 检索式: `TITLE-ABS-KEY("paired bootstrap*" OR "cluster bootstrap*" OR "grouped bootstrap*" OR "hierarchical bootstrap*" OR "cluster resampl*" OR "paired resampl*" OR "within-cluster dependenc*" OR "correlated image*" OR "group-level resampl*")`

**模块2 - 配对 effect 构造与分层聚合**

- 核心实体与锚点: Oracle-gated - corrupted baseline、boundary/nonboundary Oracle gain difference、mine stratification
- 扩展语义群: `["paired effect estimate*", "within-group difference*", "within-image difference*", "treatment effect contrast*", "difference of improvement*", "subgroup descriptive analys*", "stratified summar*", "cluster-level aggregation"]`
- 检索式: `TITLE-ABS-KEY("paired effect estimate*" OR "within-group difference*" OR "within-image difference*" OR "treatment effect contrast*" OR "difference of improvement*" OR "subgroup descriptive analys*" OR "stratified summar*" OR "cluster-level aggregation")`

**模块3 - percentile interval 与可复现实现**

- 核心实体与锚点: 95% percentile interval、10,000 resamples、fixed seed
- 扩展语义群: `["percentile bootstrap interval", "two-sided confidence interval", "bootstrap confidence interval", "ten thousand resample*", "fixed random seed", "reproducib* resampl*", "Monte Carlo error", "bootstrap implementation"]`
- 检索式: `TITLE-ABS-KEY("percentile bootstrap interval" OR "two-sided confidence interval" OR "bootstrap confidence interval" OR "ten thousand resample*" OR "fixed random seed" OR "reproducib* resampl*" OR "Monte Carlo error" OR "bootstrap implementation")`

**资源锚点模块**

- 推断资源类型: 代码类、方法/协议类
- 锚点关键词: `"github" OR "open source" OR "code available" OR "repository" OR "bootstrap implementation" OR "statistical analysis plan" OR "reporting guideline" OR "worked example"`
- 检索式: `TITLE-ABS-KEY("github" OR "open source" OR "code available" OR "repository" OR "bootstrap implementation" OR "statistical analysis plan" OR "reporting guideline" OR "worked example")`

**组合策略菜单：**

- **策略一：精准打击（高精确度）**
  - 逻辑组合: 模块1 AND 模块2 AND 模块3
  - 说明: 查找能直接对应 138 个相关组、条件内配对、effect 构造和 percentile interval 的统计协议。
  - 检索式: `TITLE-ABS-KEY(("paired bootstrap*" OR "cluster bootstrap*") AND ("within-group difference*" OR "paired effect estimate*") AND ("percentile bootstrap interval" OR "fixed random seed"))`
- **策略二：实现优先（高相关性+可操作性）**
  - 逻辑组合: (模块1 AND 模块2 AND 模块3) AND 资源锚点模块
  - 说明: 优先寻找公开代码、worked example 和统计分析计划，核对重采样单位、配对保持方式与 effect 聚合顺序。
  - 检索式: `TITLE-ABS-KEY(("group-level resampl*" OR "cluster resampl*") AND ("within-image difference*" OR "cluster-level aggregation") AND ("bootstrap confidence interval" OR "reproducib* resampl*") AND ("bootstrap implementation" OR "github" OR "statistical analysis plan"))`
- **策略三：双核探索（中等范围）**
  - 逻辑组合: 模块1 AND 模块3 AND 资源锚点模块
  - 说明: 视觉领域命中不足时，先获取通用 clustered paired bootstrap 的可靠实现，再由冻结 protocol 定义本项目的两个 effect。
  - 检索式: `TITLE-ABS-KEY(("hierarchical bootstrap*" OR "paired resampl*" OR "within-cluster dependenc*") AND ("percentile bootstrap interval" OR "fixed random seed") AND ("code available" OR "worked example" OR "bootstrap implementation"))`

### ACU 7.2：预注册裁决状态机与证据收口

**锚点实体：** `oracle-supported`、`oracle-not-supported`、`inconclusive`、`protocol-blocked`、`stop`、预先冻结的最小效应量、clean 不劣容忍度、Boundary IoU、mIoU、condition JSON、mask 哈希、GSA stage 传播摘要、`official_test_included=false`。

**检索目标：** 找到可把冻结阈值、clean 不劣约束、统计区间和证据完整性门禁映射为互斥终态的 protocol/state-machine 实现；确保实现或身份失败先进入 `protocol-blocked`/`stop`，只有证据链有效时才产生 Oracle 科学裁决。

**关键词模块定义：**

**模块1 - 预注册效应量与不劣规则**

- 核心实体与锚点: minimum effect size、clean non-inferiority margin、Boundary IoU、mIoU
- 扩展语义群: `["minimum effect size", "practical significance", "non-inferior* margin", "clean performance retention", "performance preservation", "decision threshold*", "confidence interval decision", "segmentation metric threshold*"]`
- 检索式: `TITLE-ABS-KEY("minimum effect size" OR "practical significance" OR "non-inferior* margin" OR "clean performance retention" OR "performance preservation" OR "decision threshold*" OR "confidence interval decision" OR "segmentation metric threshold*")`

**模块2 - 多状态终点与失败优先级**

- 核心实体与锚点: oracle-supported、oracle-not-supported、inconclusive、protocol-blocked、stop
- 扩展语义群: `["preregister* decision rule", "go no-go decision", "inconclusive result*", "protocol deviation", "quality gate", "fail-fast", "stopping rule*", "terminal state", "decision state machine"]`
- 检索式: `TITLE-ABS-KEY("preregister* decision rule" OR "go no-go decision" OR "inconclusive result*" OR "protocol deviation" OR "quality gate" OR "fail-fast" OR "stopping rule*" OR "terminal state" OR "decision state machine")`

**模块3 - 证据身份与机器可读收口**

- 核心实体与锚点: condition JSON、mask hash、stage propagation summary、official test excluded
- 扩展语义群: `["condition manifest", "mask hash", "stage-wise summar*", "run manifest", "artifact integrit*", "machine-readable report", "audit trail", "protocol identity", "sealed test set", "official test excluded"]`
- 检索式: `TITLE-ABS-KEY("condition manifest" OR "mask hash" OR "stage-wise summar*" OR "run manifest" OR "artifact integrit*" OR "machine-readable report" OR "audit trail" OR "protocol identity" OR "sealed test set" OR "official test excluded")`

**资源锚点模块**

- 推断资源类型: 代码类、方法/协议类
- 锚点关键词: `"github" OR "open source" OR "code available" OR "repository" OR "preregistration" OR "statistical analysis plan" OR "decision framework" OR "reporting protocol"`
- 检索式: `TITLE-ABS-KEY("github" OR "open source" OR "code available" OR "repository" OR "preregistration" OR "statistical analysis plan" OR "decision framework" OR "reporting protocol")`

**组合策略菜单：**

- **策略一：精准打击（高精确度）**
  - 逻辑组合: 模块1 AND 模块2 AND 模块3
  - 说明: 查找同时约束效应量、不劣性、协议失败和机器可读证据收口的预注册裁决设计。
  - 检索式: `TITLE-ABS-KEY(("minimum effect size" OR "non-inferior* margin") AND ("preregister* decision rule" OR "decision state machine") AND ("protocol identity" OR "artifact integrit*" OR "audit trail"))`
- **策略二：实现优先（高相关性+可操作性）**
  - 逻辑组合: (模块1 AND 模块2 AND 模块3) AND 资源锚点模块
  - 说明: 优先寻找有状态机代码、统计分析计划和结构化报告 schema 的实现，而不是只有文字性结论分类的论文。
  - 检索式: `TITLE-ABS-KEY(("confidence interval decision" OR "clean performance retention") AND ("quality gate" OR "terminal state" OR "protocol deviation") AND ("machine-readable report" OR "run manifest") AND ("github" OR "code available" OR "statistical analysis plan"))`
- **策略三：双核探索（中等范围）**
  - 逻辑组合: 模块2 AND 模块3 AND 资源锚点模块
  - 说明: 领域专用结果不足时，先获取预注册多终态决策和证据审计框架，再由本项目冻结阈值赋予 `oracle-*` 语义。
  - 检索式: `TITLE-ABS-KEY(("go no-go decision" OR "inconclusive result*" OR "decision state machine") AND ("artifact integrit*" OR "protocol identity" OR "audit trail") AND ("decision framework" OR "reporting protocol" OR "github"))`

# 第四部分：综合与行动建议

## 1. 检索结果的验收标准

每个 ACU 的检索结果只有满足以下至少一项时，才算找到“实现锚点”：

1. **代码锚点：** 能定位到仓库、文件、函数或伪代码，并能说明输入张量、输出张量及关键 shape；
2. **数据锚点：** 能提供数据版本、样本清单、manifest、哈希、split 或 annotation/ignore 契约；
3. **协议锚点：** 能明确给出变换顺序、统计单位、容差、效应量、重采样方式或裁决规则；
4. **负证据锚点：** 能证明候选方法会同时改 spatial contribution、需要训练、依赖额外模态或无法保持 clean 等价，因此应被排除。

仅有“方法可能有效”的摘要、二手博客、没有版本身份的代码片段或只报告最终指标而不披露实现的论文，不足以关闭 ACU。

## 2. 任务依赖图

```text
阶段 1 / ACU 1.1 输入身份核验 ───────────────┐
阶段 1 / ACU 1.2 门槛与统计协议 ────────┐   │
                                         │   v
阶段 2 / ACU 2.1 GSA contribution 分离 ─┼─> 阶段 4 / ACU 4.1 最小门控实现
                                         │                     │
阶段 3 / ACU 3.1 四级 mask 投影 ────────┤                     v
阶段 3 / ACU 3.2 flip/多尺度对齐 ───────┘        阶段 4 / ACU 4.2 数值等价
                                                               │
                                                               v
                                                阶段 5 / ACU 5.1 preflight
                                                               │
                                                               v
                                                阶段 6 / ACU 6.1 完整评价
                                                               │
                                                               v
                                                阶段 7 / ACU 7.1 bootstrap
                                                               │
                                                               v
                                                阶段 7 / ACU 7.2 裁决与证据收口
```

## 3. 必须顺序完成的链路

1. **ACU 1.1 必须先于任何正式实现或运行完成。** 输入身份不闭合时，后续结果无法与 v3 比较。
2. **ACU 2.1 必须先于 ACU 4.1。** 若无法证明 spatial 与 depth geometry contribution 可分离，方案直接进入 `protocol-blocked`。
3. **ACU 3.1 和 ACU 3.2 必须先于 ACU 4.2。** 未冻结传播与变换规则时，clean 等价和受损条件比较都不可解释。
4. **ACU 4.2 必须先于 ACU 5.1，ACU 5.1 必须先于 ACU 6.1。** 先证明 no-op 等价，再证明小样本链路，最后才允许完整评价。
5. **ACU 1.2 必须在 ACU 6.1 读取完整结果之前关闭。** 效应量、不劣界值和裁决规则不能根据结果回填。
6. **完整评价需要单独批准。** ACU 5.1 通过只代表实现链路具备进入完整评价的资格；在 ACU 6.1 启动前，仍需另行取得完整本地 GPU paired development evaluation 授权。
7. **ACU 7.1 只在 ACU 6.1 产物身份、配对完整性和指标契约通过后执行。** 否则应进入 `stop` 或 `protocol-blocked`，而不是计算科学区间。
8. **ACU 7.2 必须晚于 ACU 7.1。** 先产出冻结 effect 的点估计与区间，再按预先冻结的门槛映射互斥终态，不能把 bootstrap 与裁决合并成结果导向的单步判断。

## 4. 可并行检索的 ACU

- **第一并行批：** ACU 1.1、ACU 1.2、ACU 2.1 可以同时检索。三者分别面向身份治理、统计协议和模型机制，互不依赖。
- **第二并行批：** 在初步确认 DFormerv2 的四级结构后，ACU 3.1 与 ACU 3.2 可以并行：前者解决跨 stage 尺度，后者解决跨 view 几何对齐。
- **第三并行批：** ACU 4.1 的门控算子资料与 ACU 4.2 的数值等价测试资料可以并行检索，但实现必须先完成 ACU 4.1，再执行 ACU 4.2。
- **第四并行批：** ACU 5.1 的 preflight 日志/证据模板与 ACU 6.1 的 Boundary IoU 权威实现可以提前并行检索；完整评价本身仍必须等待 preflight 通过。
- **统计与裁决准备：** ACU 7.1 的 bootstrap 实现和 ACU 7.2 的终态映射框架可以提前并行检索；两者都不得读取 B1 模型结果来回填阈值。正式 bootstrap 只能在 ACU 6.1 完成后执行，正式裁决只能在 ACU 7.1 完成且证据链有效后执行。

## 5. 关键路径与优先级

**关键路径：** `ACU 1.1 → ACU 2.1 → ACU 3.1/3.2 → ACU 4.1 → ACU 4.2 → ACU 5.1 → 单独 GPU 批准门禁 → ACU 6.1 → ACU 7.1 → ACU 7.2`。

优先级建议如下：

1. **P0：ACU 2.1。** 这是最可能使整个方案提前终止的硬门禁。优先查官方 DFormerv2 论文、补充材料与仓库，要求产出“公式项—代码函数—张量 shape—四级调用点”的映射。
2. **P0：ACU 1.1。** 同步建立 v3 输入、mask、checkpoint、condition 与 evaluator 的身份核对清单，避免后续实现建立在错误证据上。
3. **P0：ACU 3.1 与 ACU 3.2。** mask 传播是第二个硬门禁；需要产出唯一规则，而不是多个可选 resize/pooling 方案。
4. **P1：ACU 1.2。** 在完整评价前关闭效应量和 clean 不劣门槛。可以晚于初始代码定位，但不能晚于完整结果生成。
5. **P1：ACU 4.1 与 ACU 4.2。** 只实现一个最小 gate；全可信 mask 无法达到预定义等价时立即停止，不进入调参。
6. **P1：ACU 5.1。** 用 1–2 张图确认三条件、四级传播和十个 view 的证据链；不得把 preflight 指标当成方案收益。
7. **P2：ACU 6.1、ACU 7.1 与 ACU 7.2。** 只有全部门禁关闭且另行取得运行批准后，才执行完整本地 GPU paired development evaluation；随后必须先完成组级 bootstrap，再执行正式裁决与证据收口。

## 6. 推荐的检索执行顺序

1. 每个 ACU 先运行**策略一**；前 20–30 条若没有可定位代码/协议，立即转策略二，不继续阅读泛泛摘要。
2. 策略二优先沿“论文标题 → 官方项目页 → 官方仓库 → 对应 commit/tag → 文件/函数”追踪；第三方复现只作为交叉核对，不覆盖官方语义。
3. 若策略二仍不足，再运行策略三，并记录它提供的是“同构实现依据”而非 DFormerv2 直接证据。
4. 对每个入选锚点记录：URL、访问日期、版本/commit、许可证、目标文件/函数、输入输出 shape、可支持的 ACU、不能支持的主张。
5. 每个 ACU 达到一个直接锚点加一个独立交叉证据后即可停止扩展；若直接锚点缺失，则明确标为“待核验”，不得用相邻方法推断 DFormerv2 的真实实现。

## 7. 当前边界

本检索方案只为 `DVG-B1-oracle-gsa-v1` 的 protocol 细化和后续实现准备服务。它不授权代码修改、GPU 运行、训练、云资源、official test、可学习质量预测、RGB-only 回退、深度补全或结果后阈值搜索。检索发现新方法时，也必须先判断它是否破坏“只改变 depth geometry contribution”这一唯一研究变量。
