# DVG-B1 WOS 检索式查准率与查全率调整建议

> **形成时间：** 2026-09-10
>
> **适用范围：** DVG-B1 的两个开放项：A）像素 corruption/validity mask 到四级 token reliability；B）单 token reliability 到 pairwise depth contribution gate。
>
> **原始依据：** `DVG-B1-必须实现细节靶向检索步骤与WOS检索式.md` 第 6.1、6.2 节，以及用户反馈的实际显示结果——两套提示词最终都显示成了第 6.2 节的 pairwise gate 检索式。
>
> **证据边界：** 当前材料没有提供 Web of Science（WOS）的命中数量、题录清单、学科分布、典型误命中文献或零结果截图。因此，本文可以根据检索目的、现有检索式结构和“两套提示词显示为同一条式子”这一结果提出定向调整，但不能把尚未提供的命中分布写成已经验证的查准率或查全率结论。

## 1. 先修正两套提示词显示同一检索式的问题

### 建议 1：把 A、B 两个检索任务彻底拆成两条独立检索式，不复用同一个“通用模板”变量

**理由：**

原文件中的两条检索式实际并不相同：

- A 检索的是 mask/confidence 如何 `downsample`、`pool`、`resize`、`propagate` 或 `aggregate`；
- B 检索的是 confidence/reliability 如何进入 attention score、attention bias、pairwise affinity 或 gate。

用户实际看到两套提示词都显示为下式：

```text
TS=((("depth confidence" OR "depth reliability" OR "validity mask" OR "corruption mask") NEAR/5 ("attention bias" OR "geometry prior" OR "pairwise attention" OR "masked attention")) AND ("RGB-D" OR "depth-guided" OR multimodal) AND ("pairwise reliabilit*" OR "query-key mask*" OR "confidence gate*" OR "multiplicative mask*" OR "attention bias mask*"))
```

这条式子只对应 B，不包含 A 所需的 `downsampl*`、`pool*`、`resiz*`、`aggregat*`、`mask update` 或 `confidence propagation` 等操作词。若用它执行 A，得到的结果即使与 attention 有关，也无法系统回答“像素 mask 如何变成每级 token reliability”。因此第一步应修复提示词、复制过程或变量替换过程，而不是先根据这条重复式子的返回结果判断 A 类文献不存在。

建议在两套提示词中分别使用固定标识，例如 `QUERY_A_PIXEL_TO_TOKEN` 和 `QUERY_B_TOKEN_TO_PAIRWISE`，并在提交 WOS 前人工核对操作词：A 必须出现 mask 聚合/传播词，B 必须出现 attention/affinity/gate 词。

## 2. 对现有检索逻辑的总体调整

### 建议 2：不要让一条超长检索式同时承担“发现术语”和“精确关闭规则”两个任务

**主要目标：提高查全率，同时保留后续提高查准率的空间。**

**理由：**

现有两条检索式都采用三个概念组强制 `AND`：

1. depth confidence/validity 概念；
2. RGB-D、depth-guided 或 multimodal 场景；
3. hierarchical/token 或 pairwise/gate 的精确表达。

这种结构适合在已知领域术语以后做精确筛选，但当前开放项的困难恰好是“论文可能不用项目内部的叫法”。例如，相关论文可能使用：

- `normalized convolution`、`partial convolution`、`mask update`，而不写 `token reliability`；
- `confidence-aware attention`、`uncertainty-guided attention`、`pairwise affinity`，而不写 `query-key mask`；
- `sparse depth` 或 `missing depth`，而不写 `corruption mask`；
- `attention score`、`attention logits` 或 `relation weights`，而不写 `attention bias mask`。

若所有概念必须同时出现，容易把“规则直接、术语不同”的论文排除。建议采用分层检索：先用较宽的发现式找术语和种子论文，再用精确式验证与 DVG-B1 的直接对应关系。

### 建议 3：把 `NEAR/5` 改为按任务分级使用，而不是两项统一固定为 5

**主要目标：兼顾查全率与查准率。**

**理由：**

`NEAR/5` 要求两个概念在很短的文本窗口内共现。它对短语明确、摘要写法紧凑的论文有较高查准率，但可能漏掉以下常见写法：

- 先介绍 depth validity/confidence，隔一句再说明如何下采样或传播；
- 先说明 uncertainty/reliability 的生成方式，后文再说明它调制 attention logits；
- 摘要使用模块名称，具体 `query`、`key`、`affinity` 或 `gate` 只在后续句子中出现。

建议：

- 发现式使用 `NEAR/10`，优先增加查全率；
- 精确式使用 `NEAR/5`；
- 若返回结果仍明显被一般网络 `downsampling` 或一般 `attention` 淹没，再收紧为 `NEAR/3`，但不要一开始就收紧。

### 建议 4：避免把项目内部术语当作文献必须使用的术语

**主要目标：提高查全率。**

**理由：**

`token reliability`、`pairwise reliability`、`query-key mask`、`depth contribution gate` 是当前项目的实现描述，不一定是已有论文的标准关键词。文献可能表达相同机制，但使用：

- `validity map`、`confidence map`、`support mask`、`sparsity mask`；
- `mask propagation`、`mask update`、`confidence propagation`；
- `attention score`、`attention logits`、`attention weights`；
- `pairwise affinity`、`relation weight`、`edge weight`、`joint confidence`；
- `confidence-aware`、`reliability-aware`、`uncertainty-aware`、`uncertainty-guided`。

因此应把内部术语保留为精确式中的一部分，但不能让它们成为发现式的硬性 `AND` 条件。

### 建议 5：暂不增加宽泛的 `NOT` 排除组

**主要目标：保护查全率。**

**理由：**

当前没有典型误命中题录，无法判断噪声主要来自医学影像、遥感、NLP、数据库、传感器可靠性还是其他领域。此时预先加入大范围 `NOT`，容易误删跨领域但可提供直接 pairwise reliability 公式的论文。

更稳妥的做法是先记录前 30–50 条结果中的主要噪声类别；只有某一噪声类别持续占据大量结果，并且与规则迁移明显无关时，才增加窄范围排除词。排除词应记录加入前后的命中数量变化，不能只凭题目印象加入。

## 3. 开放项 A：像素 mask 到四级 token reliability

### 建议 6：删去 A 发现式中的 transformer/pyramid 强制条件，补入 mask 更新和稀疏深度领域术语

**主要目标：提高查全率。**

**理由：**

A 要解决的是一个比 Transformer 更早出现的基础问题：有效性、置信度或稀疏支持区域经过卷积、池化、插值或层级下采样后如何传播。直接规则很可能来自 sparse depth completion、normalized convolution、partial convolution 或 mask-adaptive convolution，而不是标题中明确出现 `hierarchical transformer` 的论文。

现有末组：

```text
("hierarchical transformer" OR "feature pyramid" OR "multi-scale attention" OR "token mask" OR "partial validit*")
```

会漏掉只写 `encoder-decoder`、`multi-resolution`、`sparse depth`、`mask update` 或 `confidence propagation` 的方法。建议把架构限制从发现式移到精确式，并增加：

- `sparse depth`、`missing depth`、`invalid depth`；
- `mask update`、`mask propagation`、`confidence propagation`；
- `normalized convolution`、`partial convolution`；
- `valid fraction`、`valid pixel ratio`、`support ratio`；
- `mask-aware`、`validity-aware`。

这些词更贴近“部分受损 patch 怎样形成连续或离散 reliability”的实际问题。

### 建议 7：不要单独使用宽泛的 `corruption mask`，改为与 depth 局部绑定

**主要目标：提高查准率。**

**理由：**

`corruption mask` 可出现在图像修复、数据增强、鲁棒学习、NLP masking、异常检测等大量无关文献中。现有式虽然还有 RGB-D/depth 条件，但 `corruption mask` 本身仍可能与另一处无关的 `depth` 偶然共现。

建议使用局部关系表达：

```text
(depth NEAR/3 (corrupt* OR invalid* OR missing OR mask*))
```

并保留更常用的精确短语：

```text
"depth validity" OR "depth confidence" OR "sparse depth mask" OR "invalid depth mask"
```

这样既能覆盖词序变化，也能减少一般图像 corruption 论文。

### 建议 8：把 `downsampl* OR pool* OR resiz*` 扩展为“操作词 + 语义词”两类

**主要目标：同时提高查全率和结果可解释性。**

**理由：**

只搜索操作词容易命中“网络进行了下采样”，却不一定说明 mask 如何更新；只搜索语义词又可能漏掉明确给出 average/max/nearest 规则的实现。建议同时覆盖：

- 操作词：`downsampl*`、`resiz*`、`resampl*`、`pool*`、`interpolat*`、`aggregat*`；
- 传播语义：`mask update`、`mask propagation`、`confidence propagation`、`validity propagation`；
- 部分有效语义：`valid fraction`、`valid pixel ratio`、`partial valid*`、`support ratio`；
- 候选算子词：`average pooling`、`max pooling`、`nearest neighbor`、`bilinear interpolation`、`area interpolation`。

第一轮不应把所有候选算子都设为硬性条件；它们更适合作为找到种子论文后的二次精确检索词。

### A-1：推荐的平衡型主检索式

```text
TS=((
  (
    (depth NEAR/3 (valid* OR invalid* OR confid* OR reliab* OR mask* OR spars* OR missing OR corrupt*))
    OR "depth confidence map"
    OR "depth validity map"
    OR "sparse depth mask"
    OR "invalid depth mask"
  )
  NEAR/10
  (
    downsampl* OR resiz* OR resampl* OR pool* OR interpolat* OR aggregat* OR updat* OR propagat*
    OR "mask update" OR "mask propagation" OR "confidence propagation" OR "validity propagation"
    OR "valid fraction" OR "valid pixel ratio" OR "support ratio"
    OR "normalized convolution" OR "partial convolution" OR "mask-aware" OR "validity-aware"
  )
) AND ("depth completion" OR "depth estimation" OR "sparse depth" OR "RGB-D" OR RGBD))
```

**理由：**

这条式子保留 depth 领域边界，但不要求论文必须属于 Transformer 或 feature pyramid；同时把 mask/confidence 与传播、更新、聚合操作放在 `NEAR/10` 内，能减少只在摘要不同位置偶然出现的无关共现。它适合作为 A 的首轮主检索式。

### A-2：结果过少时使用的高查全率检索式

```text
TS=((
  (valid* OR invalid* OR confid* OR reliab* OR "validity map" OR "confidence map" OR "sparsity mask" OR "support mask")
  NEAR/10
  (downsampl* OR resiz* OR resampl* OR pool* OR interpolat* OR aggregat* OR updat* OR propagat* OR "valid fraction" OR "normalized convolution" OR "partial convolution")
) AND (depth OR "sparse depth" OR "depth completion" OR "RGB-D" OR RGBD))
```

**理由：**

该式去掉了“所有 mask/confidence 词都必须紧邻 depth”的限制，以覆盖摘要中先说 sparse measurement confidence、后说 depth completion 的写法。代价是噪声会增加，因此只在 A-1 结果过少或明显缺少 normalized/partial convolution 文献时使用。

### A-3：结果过多时使用的高查准率检索式

```text
TS=((
  ("depth validity" OR "depth confidence" OR "sparse depth mask" OR "invalid depth mask")
  NEAR/5
  ("confidence propagation" OR "validity propagation" OR "mask propagation" OR "mask update" OR "valid fraction" OR "valid pixel ratio" OR "normalized convolution" OR "partial convolution")
) AND ("depth completion" OR "RGB-D" OR RGBD) AND (multiscale OR "multi-scale" OR pyramid* OR hierarchical OR encoder*))
```

**理由：**

该式恢复多尺度/层级架构限制，并将一般操作词替换为更直接的 mask/confidence 传播短语，可减少仅提到普通 pooling 或 resize 的论文。它适合 A-1 返回大量一般网络结构论文时使用，不适合首轮发现。

### A-4：专门回答“与 bilinear Depth resize 如何对齐”的补充检索式

```text
TS=(((depth NEAR/3 (mask* OR valid* OR invalid* OR confid*)) NEAR/10 (bilinear OR interpolat* OR resiz* OR resampl* OR "nearest neighbor" OR "area interpolation")) AND ("depth completion" OR "RGB-D" OR RGBD OR "depth estimation"))
```

**理由：**

A-1 主要寻找从 mask 到 token reliability 的一般传播规则，但不保证结果会讨论连续 Depth 与离散 mask 使用不同插值方法时的对齐。A-4 把插值与 mask/validity 直接绑定，适合独立核对 bilinear、nearest、area 或其他 resampling 语义。该式不应替代 A-1，因为很多给出 mask 更新公式的论文不会在题录字段中写具体插值名称。

## 4. 开放项 B：token reliability 到 pairwise depth contribution gate

### 建议 9：删除现有最后一个过窄的强制短语组，改用 attention 的实际计算对象

**主要目标：显著提高查全率。**

**理由：**

现有最后一组要求论文至少出现以下一个表达：

```text
"pairwise reliabilit*" OR "query-key mask*" OR "confidence gate*" OR "multiplicative mask*" OR "attention bias mask*"
```

这些表达并不是稳定统一的文献术语。直接实现相同机制的论文更可能写：

- `attention score`、`attention logits`、`attention weights`；
- `pairwise affinity`、`relation weight`、`edge weight`；
- `confidence-aware attention`、`reliability-aware attention`；
- `uncertainty-aware attention`、`uncertainty-guided attention`；
- `gated attention`、`gated cross-attention`。

因此，最后一组不应继续作为硬性 `AND` 条件。应把 `pairwise`、`query key`、`affinity` 和 `relation` 作为精确式约束，而把 attention score/logit/weight 和 reliability-aware 等词加入发现式。

### 建议 10：删除或降级 `masked attention`，因为它常指 padding 或 causal mask

**主要目标：提高查准率。**

**理由：**

`masked attention` 在 Transformer 文献中通常表示 causal mask、padding mask、window mask 或可见性约束，不一定涉及 depth reliability，更不一定作用于 depth-only geometry contribution。它容易带来大量“有 mask、有 attention，但没有传感器可靠性”的误命中。

建议：

- 不在 B 的平衡型主检索式中单独使用 `masked attention`；
- 若保留，必须与 depth confidence/validity 使用较紧的 `NEAR/3` 或 `NEAR/5`；
- 优先用 `attention score*`、`attention logit*`、`attention weight*`、`pairwise affinity` 和 `relation weight*`，因为这些词更接近需要被 gate 调制的张量。

### 建议 11：把 `attention bias` 与 `geometry prior` 从宽泛短语改为受 reliability 限定的计算对象

**主要目标：提高查准率，同时避免漏掉非 bias 实现。**

**理由：**

`attention bias` 可能表示位置偏置、归纳偏置、公平性偏差或语言模型 bias；`geometry prior` 也可能只是一个输入特征，而不是被 reliability 调制的成对项。另一方面，有些论文把 confidence 直接乘在 attention weights 或 affinity 上，并不称其为 bias。

建议把关系写成：

```text
(confid* OR reliab* OR valid* OR uncertainty OR quality)
NEAR/10
("attention score*" OR "attention logit*" OR "attention weight*" OR "pairwise affinity" OR "relation weight*" OR gate*)
```

这样既限定了 reliability 与 attention 计算对象的局部关系，也不预设它一定采用加性 bias 或乘法 mask。

### 建议 12：增加 query/key 的拼写变体和 pairwise 的替代表达

**主要目标：提高查全率。**

**理由：**

论文可能写 `query-key`、`query key`、`query-to-key`、`token pair`、`pairwise affinity`、`relation` 或图结构中的 `edge weight`。只搜索带连字符的 `query-key mask*` 会漏掉大量变体。

建议增加：

- `"query key"`、`"query-key"`、`"query-to-key"`；
- `pairwise`、`"token pair*"`；
- `affinity`、`"pairwise affinity"`；
- `relation*`、`"relation weight*"`；
- 在补充检索中加入 `"edge confidence"`、`"edge weight*"`、`"joint confidence"`。

`edge weight` 会引入图神经网络文献，因此适合补充检索，不宜直接放入最高查准率主式。

### 建议 13：增加 confidence/reliability 的不确定性与质量同义词，但保留 depth 约束

**主要目标：提高查全率。**

**理由：**

部分论文不把连续可信度称为 confidence 或 reliability，而称为 `uncertainty`、`quality`、`validity`、`trust` 或 `certainty`。其中 `uncertainty-aware attention` 已形成较常见的表达。如果只搜索 confidence/reliability，会漏掉先预测 uncertainty、再用其衰减 attention 的方法。

建议增加 `uncertainty`、`quality`、`certainty`，但不要在没有 depth/RGB-D/multimodal 条件时单独使用，因为这些词跨领域噪声很大。

### B-1：推荐的平衡型主检索式

```text
TS=((
  (
    (depth NEAR/3 (confid* OR reliab* OR valid* OR invalid* OR uncertainty OR quality OR certainty))
    OR "depth confidence"
    OR "depth reliability"
    OR "depth uncertainty"
    OR "depth validity"
  )
  NEAR/10
  (
    attention OR "attention score*" OR "attention logit*" OR "attention weight*"
    OR "attention bias" OR "pairwise affinity" OR "relation weight*" OR gate* OR gating
  )
) AND ("RGB-D" OR RGBD OR "depth-guided" OR "depth completion" OR "depth estimation" OR multimodal))
```

**理由：**

该式不再要求论文使用 `query-key mask` 等项目内部短语，而是要求 depth reliability 与 attention 的实际计算对象在局部文本内关联。它同时覆盖加性 bias、乘性 gate、attention weight 和 affinity 等不同实现，适合作为 B 的首轮主检索式。

### B-2：结果过少时使用的高查全率检索式

```text
TS=((
  "confidence-aware attention" OR "reliability-aware attention" OR "uncertainty-aware attention"
  OR "uncertainty-guided attention" OR "confidence-guided attention" OR "gated cross-attention"
  OR ((confid* OR reliab* OR valid* OR uncertainty OR quality) NEAR/10 (attention OR "attention score*" OR "attention logit*" OR "attention weight*" OR affinity OR gate*))
) AND (depth OR "RGB-D" OR RGBD OR "depth-guided" OR multimodal))
```

**理由：**

该式通过常见模块命名提高术语覆盖，并允许摘要在不同位置分别描述 confidence 和 attention。它可能返回模态融合、NLP 或一般 uncertainty-aware attention 论文，因此应在筛选阶段要求候选文献给出局部 confidence 如何作用于 score/logit/weight/affinity 的明确公式或代码。

### B-3：结果过多时使用的高查准率检索式

```text
TS=((
  ("depth confidence" OR "depth reliability" OR "depth uncertainty" OR "depth validity")
  NEAR/5
  ("attention score*" OR "attention logit*" OR "attention bias" OR "pairwise affinity" OR "relation weight*" OR "gated attention")
) AND ("query key" OR "query-key" OR "query-to-key" OR pairwise OR affinity OR relation*) AND ("RGB-D" OR RGBD OR "depth-guided" OR "depth completion"))
```

**理由：**

该式重新加入 query/key、pairwise 或 affinity 约束，并缩短邻近距离，适合过滤只做全局 modality confidence 或通道级融合的论文。它更可能找到成对关系，但也可能漏掉正文有公式、摘要没有 query/key 字样的论文，因此不应替代 B-1 的首轮发现作用。

### B-4：专门寻找“两端 reliability 如何组成成对权重”的补充检索式

```text
TS=((
  (confid* OR reliab* OR valid* OR uncertainty)
  NEAR/5
  ("joint confidence" OR "pairwise confidence" OR "edge confidence" OR "pairwise weight*" OR "outer product" OR affinity)
) AND (attention OR transformer* OR graph* OR "non-local") AND (depth OR "RGB-D" OR RGBD OR multimodal))
```

**理由：**

B-1 至 B-3 主要寻找“可靠性进入 attention”的直接证据，但未必能在题录字段中命中两端乘积、最小值或其他对称组合。B-4 扩展到 affinity 和图关系中的 pairwise/edge confidence，可能找到可解释单点 confidence 如何提升为边权的公式。

该式只能作为规则来源的补充入口。若候选来自图神经网络或其他模态，必须进一步论证其 pairwise 语义可迁移到 DVG-B1 的 depth contribution，不能仅因出现 `outer product` 或 `edge weight` 就冻结规则。

## 5. 建议的实际检索顺序

### 建议 14：按“纠错—发现—收紧—公式追踪”四步执行，不同时混跑所有变体

**理由：**

同时提交大量相近检索式会产生重复结果，也不利于判断哪个关键词真正改善了查准率或查全率。建议按以下顺序记录：

1. **纠错：**确认 A 提交的是 A-1、B 提交的是 B-1，保存 WOS 实际解析后的完整检索式；若界面仍显示同一条式子，先停止文献判断并修复提示词或复制过程。
2. **发现：**分别运行 A-1、B-1，记录命中总数，并人工检查相关性排序前 30–50 条。
3. **收紧或放宽：**
   - 结果过少：分别改用 A-2、B-2；
   - 结果过多且多数不回答目标问题：分别改用 A-3、B-3；
   - 不要在同一次调整中同时改邻近距离、删除概念组并增加排除词，否则无法判断是哪项改变造成结果差异。
4. **公式追踪：**对 A 使用 A-4 追踪 interpolation/mask alignment；对 B 使用 B-4 追踪 joint/pairwise confidence。找到种子论文后，再使用其作者关键词、参考文献和被引文献做定向补充。

每次检索至少记录：

- 检索式编号和完整字符串；
- 检索日期与 WOS 数据库范围；
- 命中总数；
- 前 30–50 条中“直接回答、间接相关、无关”的数量；
- 主要误命中主题；
- 新发现的作者术语；
- 是否存在论文、官方代码、commit、函数和 shape 级直接证据。

## 6. 如何根据下一轮实际结果继续调整

### 建议 15：若 A 的结果主要是一般网络下采样，优先替换操作词，不先增加 `NOT`

**理由：**

一般下采样噪声说明 `downsampl*`、`pool*`、`resiz*` 太宽，而不是说明 depth 领域条件不足。应优先把操作组收紧为：

```text
"mask update" OR "mask propagation" OR "confidence propagation" OR "validity propagation" OR "valid fraction" OR "normalized convolution" OR "partial convolution"
```

这种调整直接提高目标概念密度，比加入大范围学科排除词更安全。

### 建议 16：若 A 的结果主要是深度补全性能论文但没有 mask 公式，增加实现机制词和代码筛选条件

**理由：**

许多 depth completion 论文会提 confidence 或 mask，却不公开确定性传播规则。此时可用题录中的方法名继续检索：`normalized convolution`、`partial convolution`、`mask-adaptive convolution`、`sparsity-invariant convolution`。筛选时要求正文或官方代码明确给出 mask/confidence 的更新公式；只报告精度提升的论文不能关闭 A。

### 建议 17：若 B 的结果主要是通道级或模态级 gating，增加 pairwise 计算对象，而不是只增加 `pairwise reliability`

**理由：**

B 需要的是 query-key 或 token-pair 级 gate。若结果只在整条 RGB/Depth 分支上做单一权重，应增加：

```text
"attention score*" OR "attention logit*" OR "pairwise affinity" OR "relation weight*"
```

并加入：

```text
"query key" OR "query-key" OR pairwise OR affinity
```

`pairwise reliability` 本身使用率可能很低，继续把它作为唯一限制会降低查全率。

### 建议 18：若 B 的结果主要是 causal/padding/window mask，移除 `masked attention`

**理由：**

这类结果说明 `masked attention` 的常见语义与目标不一致。移除后应保留 `attention score/logit/weight` 和 depth confidence 的邻近关系，使结果继续聚焦“可靠性怎样修改注意力”，而不是“注意力是否使用某种结构 mask”。

### 建议 19：若直接 RGB-D 文献仍没有给出两端组合公式，允许跨到 graph/affinity 文献，但单独标注为迁移证据

**理由：**

单 token reliability 到 pairwise gate 的数学结构也可能以图边权、pairwise affinity 或 joint confidence 的形式出现。跨领域检索有助于提高查全率，但其证据等级低于直接 RGB-D attention 实现。只有当公式语义、对称性、全可信恒等性和 hard/continuous 边界均可映射，并有代码或严格公式时，才可作为候选；否则只能帮助扩充术语，不能直接冻结 B1 protocol。

## 7. 建议优先复制执行的两条检索式

当前最优先执行的是 A-1 与 B-1，而不是让两套提示词继续共用原 B 式。

### A：像素 mask 到 token reliability

```text
TS=((
  (
    (depth NEAR/3 (valid* OR invalid* OR confid* OR reliab* OR mask* OR spars* OR missing OR corrupt*))
    OR "depth confidence map"
    OR "depth validity map"
    OR "sparse depth mask"
    OR "invalid depth mask"
  )
  NEAR/10
  (
    downsampl* OR resiz* OR resampl* OR pool* OR interpolat* OR aggregat* OR updat* OR propagat*
    OR "mask update" OR "mask propagation" OR "confidence propagation" OR "validity propagation"
    OR "valid fraction" OR "valid pixel ratio" OR "support ratio"
    OR "normalized convolution" OR "partial convolution" OR "mask-aware" OR "validity-aware"
  )
) AND ("depth completion" OR "depth estimation" OR "sparse depth" OR "RGB-D" OR RGBD))
```

### B：token reliability 到 pairwise gate

```text
TS=((
  (
    (depth NEAR/3 (confid* OR reliab* OR valid* OR invalid* OR uncertainty OR quality OR certainty))
    OR "depth confidence"
    OR "depth reliability"
    OR "depth uncertainty"
    OR "depth validity"
  )
  NEAR/10
  (
    attention OR "attention score*" OR "attention logit*" OR "attention weight*"
    OR "attention bias" OR "pairwise affinity" OR "relation weight*" OR gate* OR gating
  )
) AND ("RGB-D" OR RGBD OR "depth-guided" OR "depth completion" OR "depth estimation" OR multimodal))
```

## 8. 术语核对来源与使用边界

- Web of Science Core Collection 的 Search Rules 与 Search Fields 官方帮助页用于核对 `TS=`、布尔关系、短语、通配符和 `NEAR/x` 的基本使用方式：
  - <https://webofscience.zendesk.com/hc/en-us/articles/25350084904721-Search-Rules>
  - <https://webofscience.zendesk.com/hc/en-us/articles/26916258216209-Web-of-Science-Core-Collection-Search-Fields>
- Eldesokey 等的 *Confidence Propagation through CNNs for Guided Sparse Depth Regression* 用于确认 `confidence propagation`、`normalized convolution` 和 sparse depth confidence 是实际使用的论文术语：<https://doi.org/10.1109/TPAMI.2019.2929170>。
- *Gated Cross-Attention Network for Depth Completion* 用于确认 `gated cross-attention` 和 confidence propagation 是 depth completion 文献中的实际表达：<https://arxiv.org/abs/2309.16301>。

这些来源只用于调整检索词，不代表它们已经直接回答 DVG-B1 的 A 或 B，也不构成 protocol 冻结依据。候选论文仍须按原计划核对公式、代码、commit、函数、输入输出 shape、对称性和全可信恒等性。

## 9. 当前恢复点

1. 先检查生成两套提示词的过程，确保 A 显示 A-1、B 显示 B-1。
2. 分别运行 A-1 与 B-1，并记录命中总数、前 30–50 条相关性分类和典型误命中。
3. 根据结果过少或过多，单独切换到 A-2/A-3 或 B-2/B-3；不要同时改变多个逻辑条件。
4. 使用 A-4 回答 mask 与 bilinear Depth resize 的对齐问题，使用 B-4 追踪两端 reliability 的组合公式。
5. 在获得直接文献和代码证据前，A、B 均保持开放，不冻结 nearest/area/max/average、hard/continuous、乘积/最小值/query-only/key-only 等规则。
