# MUSeg `DVG-B1-oracle-gsa-v1`：Oracle GSA 深度门控条件式计划

> **文档角色：** 条件式后继子计划；项目内实现锚点已核验，外部参考门禁尚未冻结。
> **计划状态：** 项目内实现锚点已补齐；暂停于外部参考冻结门禁；代码和运行仍未授权。
> **形成或核验时点：** 2026-09-10 08:52 UTC。
> **实时入口：** [`MUSeg-current-status.md`](../../main/MUSeg-current-status.md)。
> **研究选择：** [`MUSeg-open-decisions.md`](../../main/MUSeg-open-decisions.md)。
> **上级方向：** [`00-总方向规划.md`](./00-总方向规划.md)。
> **前序关系：** `DVC-A1-valdev-boundary-zero-v3-bgcontext` 已完成正式开发评价并裁决为 `not-supported`；本设计按照用户要求保留一个独立的方案验证窗口，不把 A 的失败改写为支持，也不把 B 的结果当作 A 的问题证据。
> **后继关系：** B 实验结果不能自动授权可学习质量预测、完整联合恢复、训练、云资源或 official test；后续需在本设计基础上补充参考文献、冻结 protocol，并取得单独执行批准。

## 1. 为什么在 A 未支持时仍设计 B

`DVC-A1` 的 v3 结果只说明：在固定可构造 `val-dev` 位置组中，人工把深度边界像素置零，没有观察到预注册的边界特异敏感性。它回答的是“预设问题是否成立”，不是“少信不可靠深度是否可能有效”。这种从退化观测可靠性出发、再检查下游决策稳定性的动机，与矿山多模态退化级联和低照可靠感知研究相邻。[RE026][RE049]

因此保留一个低成本的 B 实验，用来检查一种不能提前排除的情况：

> 深度边界并不是主要受损位置，但只要根据已知的深度 corruption mask 少信这些局部深度几何信息，模型仍可能获得任务收益。

大白话说：A 是“问题找得准不准”，B 是“这个解决动作本身有没有用”。A 没通过时，B 仍然可以作为独立的方案筛选；但如果 B 有效，只能先说明“这个干预动作有经验价值”，不能反过来证明 A 的问题假设成立。

## 2. 最小问题与方案假设

### 2.1 最小问题

在与 v3 完全相同的人工深度退化输入上，若提前知道哪些像素被置为无效，只在 DFormerv2 的 Geometry Self-Attention（GSA，几何自注意力）中降低这些位置的 **depth geometry contribution**，是否比不做门控的原始 DFormerv2 输出更好？这里的研究对象直接对应 DFormerv2 将 Depth 作为几何先验注入注意力的机制。[PR070]

### 2.2 方案假设

Oracle 门控假设为：

> 已知退化位置后，保留 RGB 和 GSA 的 spatial contribution，只抑制受损位置对应的 depth geometry contribution，可以改善人工退化条件下的 Boundary IoU 或 mIoU，同时不损害 clean 条件。该假设借用的是“按局部可靠性调节不可靠跨模态信息”的方法动机，而不是声称已有工作已经实现了 DFormerv2 的 GSA 深度门控。[PR029][PR089][PR070]

这里的 Oracle 表示“直接使用真实 corruption mask 的理想情况”。它只用于测量方案上限，不代表真实系统已经具备自动识别坏深度的能力。

### 2.3 方案结果如何解释

- **B 支持：** 已知坏区的最小 GSA 深度门控在预先冻结的条件下产生稳定净收益。只能说明该干预值得继续做质量信号研究。
- **B 不支持：** Oracle 也没有稳定净收益，说明当前“在 GSA 中少信已知坏深度”的方案缺少继续投入依据。
- **A 不支持、B 支持：** 这是允许出现的结果。结论应写成“问题定位未获支持，但理想位置门控显示条件性经验收益”，不能写成“深度边界损坏机制得到支持”。
- **A、B 都不支持：** 关闭“深度边界 + GSA Oracle 门控”这条首选路线，回到蓝图中的其他候选瓶颈，例如整体深度失效、RGB 低照/粉尘退化、观测保真或标定误差。

## 3. 固定输入身份与不变量

B 设计优先复用 v3 的已核验开发证据，避免重新制造一套无法比较的输入：

- **模型：** DFormerv2-S RGB Quick-B0，固定 epoch 420 checkpoint；该模型的深度输入角色以 GSA 几何先验为核心，而非普通 RGB 颜色通道。[PR070]
- **数据：** v3 冻结的 `val-dev` 派生范围，218 张图、138 个 location group；MUSeg 的地下 RGB-D 数据集背景由 [RE326] 提供，实际运行范围仍以冻结 v3 manifest 为准；
- **Depth：** 原始 `Depth16` 上的 v3 corruption mask，随后统一经过既有量化、归一化和 resize 链；
- **输入和评估：** RGB contract `rgb-imagenet-rgb-order-v1`，五尺度原图/水平翻转，恢复到原始 Label 网格后计分；
- **标签和指标：** 使用 v3 已修正的 Boundary IoU 标签契约，raw background 作为有效几何上下文，true ignore 单独保留；
- **统计单位：** location group 是相关性边界；同一组内的全部图像保留配对结构；
- **官方测试集：** official test 继续 `sealed_unread`，protocol 必须写明 `official_test_included=false`。

### 3.1 已可直接复用的 evaluator 与证据骨架

项目内现有实现已经覆盖下列输入、几何和审计职责，B1 不再为这些内容另造一条链：

- 原始 `Depth16` corruption mask、`clean`/`boundary-q25`/`boundary-q50`/`boundary-q75`/`nonboundary-q50` 五个 condition，以及 mask 的确定性、嵌套、数量和 SHA-256；
- 五个尺度的原图/水平翻转，共 10 个 view；RGB、Depth 与后续 Oracle mask 必须共享同一 view 几何；
- 仅在右侧和底部 padding 到 32 的倍数，flip view 的 logits 逆翻转，并恢复到原始 Label 网格；
- 10 个 view 的 FP32 pre-softmax logits 算术平均；
- checkpoint `strict=True` 加载；
- q=0 时 corruption 后量化 Depth 数组与生产 Depth8 的数组级完全等价；
- finite、输出 shape、原始网格恢复和原子 JSON 写入组成的 preflight 框架。

上述复用只说明现有 evaluator 和输入证据链可承接 B1；未来 gate-specific 的 token reliability、pairwise gate 和 no-op 输出等价仍须按本文门禁补充。

B 与 baseline 的唯一研究变量是：是否在 GSA 内根据冻结的 corruption mask 抑制 depth geometry contribution。以下内容全部保持不变：RGB、受损 Depth 输入、Label、checkpoint、GSA 的 spatial contribution、五尺度翻转、logits 融合、metric grid、后处理和统计方法。

## 4. 三组最小比较

每个 condition 至少保留以下同源比较：

1. **Clean baseline：** 原始 clean Depth 输入，不做门控；
2. **Corrupted baseline：** 按 v3 mask 置零后的 Depth 输入，不做门控；
3. **Oracle-gated：** 与 corrupted baseline 使用完全相同的 RGB、Depth、checkpoint 和 evaluator，只在 GSA 中抑制 mask 指定位置的 depth geometry contribution。

clean 条件同时作为“不应改变原模型”的控制。完整模态性能保持应与退化条件鲁棒性一起检查，而不是只报告受损条件收益。[PR090] 若 clean 输入下 Oracle 与原模型不等价，不能继续解释退化条件结果。

## 5. 五个 condition 和 Oracle mask

复用 v3 的五个 condition，不新增退化类型：

- `clean`；
- `boundary-q25`；
- `boundary-q50`；
- `boundary-q75`；
- `nonboundary-q50`。

对于四个受损 condition，Oracle 使用该 condition 实际置零的像素 mask，而不是统一使用“边界 mask”。这样可以区分两种情况：

- 如果 boundary 条件有效，说明边界位置门控可能有价值；深度内部平滑、边界锐利的结构先验可作为这些条件的背景动机，但不构成门控有效性的证据。[PR117]
- 如果 nonboundary 条件也同样有效，说明收益可能来自“识别任意受损深度并降低其影响”，而不是来自边界特异机制。

各 condition 的 corruption mask、样本范围和哈希必须沿用冻结 v3 证据；不得根据 B 结果重新定义 mask、改阈值、删组或增加剂量。

## 6. 已核验的实现边界与唯一 gate 位置

### 6.1 论文公式与当前 checkpoint 代码路径并不完全相同

DFormerv2 论文把每个 depth patch 的平均深度记为 $z_{ij}$，并以

$$
D_{ij,i'j'} = |z_{ij}-z_{i'j'}|
$$

构造 depth relationship；论文文字说明用 average pooling 得到四级 depth patch，并把 GSA 写成

$$
\operatorname{GeoAttn}(Q,K,V,G)
=
\left(\operatorname{Softmax}(QK^T)\odot\beta^G\right)V.
$$

当前项目与作者原始保留副本的实际代码则在每个 `GeoPriorGen.forward` 中先执行 `F.interpolate(..., mode="bilinear", align_corners=False)`，再把空间衰减和深度衰减作为 softmax 前的可加 bias。其真实 forward 语义为

$$
\operatorname{Attention}
=
\operatorname{Softmax}\left(QK^T+w_sP_s+w_dP_d\right)V.
$$

当前 `models/encoders/DFormerv2.py` 与作者保留副本 `D:/0Project/origin/DFormer/models/encoders/DFormerv2.py` 已重新核验为完整文件一致：两个文件的 SHA-256 均为 `2b0b77ea401d56993aac915883bcb43035927ec991501dba94fb029901009332`，完整差异检查退出码为 `0`。因此，average pooling 与 bilinear interpolation 的差异属于作者论文叙述和作者发布代码之间的上游差异，不是 MUSeg/MVE 适配，也不是当前项目的意外修改。B1 的真实基线语义必须以 epoch 420 checkpoint 对应的 bilinear 代码路径为准；论文公式只用于解释机制，不得替换当前 forward。

### 6.2 GSA contribution、四级结构与关键 shape 已由代码关闭

`GeoPriorGen.forward` 已经把两类 contribution 暴露在加和之前：

- spatial contribution 是 `self.weight[0]` 乘以位置 decay，即 `mask`、`mask_h`、`mask_w` 对应的空间项；
- depth geometry contribution 是 `self.weight[1]` 乘以 `mask_d`、`mask_d_h`、`mask_d_w`。

DFormerv2-S 的实际配置已经固定为：

- `embed_dims=[64,128,256,512]`；
- `depths=[3,4,18,4]`；
- `num_heads=[4,4,8,16]`；
- `heads_ranges=[4,4,6,6]`。

四个 stage 的 attention 结构为：Stage 0–2 使用 H/W 分解式 GSA，Stage 3 使用 Full GSA。设 batch 为 $B$，head 数为 $N$，当前特征网格为 $H\times W$，且 $L=HW$：

- 分解式 H depth contribution 为 `[B,N,W,H,H]`；对应 spatial contribution 从 `[N,H,H]` 广播到 batch 和 $W$ 轴；待冻结的 H pairwise gate 形状必须为 `[B,1,W,H,H]`，并仅在 head 轴广播；
- 分解式 W depth contribution 为 `[B,N,H,W,W]`；对应 spatial contribution 从 `[N,W,W]` 广播到 batch 和 $H$ 轴；待冻结的 W pairwise gate 形状必须为 `[B,1,H,W,W]`，并仅在 head 轴广播；
- Full GSA 的 spatial contribution 为 `[N,L,L]`，depth contribution 与合成 geometry mask 为 `[B,N,L,L]`；待冻结的 Full pairwise gate 形状必须为 `[B,1,L,L]`。

前三个 stage 分别先做 W attention、再做 H attention；第四个 stage 把合成后的 geometry mask 加到 `[B,N,L,L]` 的 query-key logits 后再 softmax。不同测试尺度的 $H\times W$ 由该 view 的右侧/底部 padding 后尺寸按 $1/4、1/8、1/16、1/32$ 产生，不写死为单一输入分辨率。

### 6.3 唯一合规的 gate 插入点

唯一合规位置是 `GeoPriorGen.forward` 中 spatial contribution 与 depth geometry contribution 相加之前。目标形式只能是：

$$
M = w_sP_s + w_d\left(R\odot P_d\right),
$$

分解式 GSA 则分别为：

$$
M_h = w_sP_{s,h} + w_d\left(R_h\odot P_{d,h}\right),
$$

$$
M_w = w_sP_{s,w} + w_d\left(R_w\odot P_{d,w}\right).
$$

这里 $R$、$R_h$ 和 $R_w$ 是尚待外部参考冻结的 pairwise gate。它们只乘 depth contribution；`self.weight[0]` 对应的 spatial contribution、Q/K/V、LEPE、FFN 和其他前向语义保持不变。

以下位置明确禁止作为 gate：

- 合成后的 `mask`、`mask_h` 或 `mask_w`：此时会同时修改 spatial contribution；
- `qk_mat + mask`、`qk_mat_h + mask_h` 或 `qk_mat_w + mask_w` 之后：此时 depth 与 spatial 已不可分；
- 整个 `geo_prior`：会同时移除 spatial decay，并错误波及与 Depth 无关的 rotary position encoding；
- 原始 Depth 输入 `x_e`：这会改变输入本身，不再是只门控既有 depth geometry contribution；
- `sin/cos` rotary position encoding：它只由 token 位置生成，不是 depth contribution；
- decoder 或最终 logits：这属于输出后处理，不是 GSA depth-only gate。

Attention 类不需要直接接收 Oracle mask；它继续只消费已经合成的 geometry prior。

### 6.4 最小参数传递链与四级共同 mask 边界

未来若获代码授权，最小接口是增加可选参数 `oracle_corruption_mask=None`，并只沿以下链路传递：

`EncoderDecoder.forward/encode_decode` → `dformerv2.forward` → `BasicLayer.forward` → `RGBD_Block.forward` → `GeoPriorGen.forward`。

四个 stage 以及各 stage 内的所有 block 都接收同一个 **view-specific Oracle mask**。这里的“同一个”表示它们共享该 view 上同一份原始 corruption 事实，而不是复用一张已经 resize 到某一级的 token mask：

1. 原始 `Depth16` corruption mask 必须使用与 RGB/Depth 相同的 scale 形成该 view；
2. flip view 沿宽度轴同步翻转，且只在右侧/底部 padding；padding 区不记为 corruption；
3. Stage 0–3 分别根据自己的 $H\times W$，从这份 view-specific mask 确定性聚合 token reliability；
4. 不允许为不同 stage 另定义不同 corruption 语义，也不允许根据模型结果选择某些 stage 才接收 mask。

像素 mask 到 token reliability 的聚合规则，以及 token reliability 到 $R$、$R_h$、$R_w$ 的提升规则仍未关闭，见第 10 节 A、B。

### 6.5 no-op 必须走原始 forward 旁路

no-op 等价不再依赖外部浮点容差文献。`oracle_corruption_mask=None`、`clean`、q=0 和全可信 mask 必须先统一归一化到原始未修改 forward 旁路：不生成 token reliability、不构造 pairwise gate，直接执行当前 spatial/depth 合成表达式。

后续获授权实现时，必须用 `torch.equal` 同时检查：

1. 原始模型与 `oracle_corruption_mask=None` 的逐 stage 输出完全相等；
2. clean、q=0、全可信 mask 与同一旁路的逐 stage 输出完全相等；
3. 上述各组最终 FP32 pre-softmax logits 完全相等。

如果只有放宽绝对或相对浮点容差才能通过，说明实现没有进入同一原始旁路，应先修正实现，不能通过扩大 tolerance 解决。

### 6.6 项目内证据入口

本节结论的主要项目内证据为：

- `models/encoders/DFormerv2.py:173-212`：bilinear Depth resize、H/W 与 Full geometry prior 的 spatial/depth 加和位置；
- `models/encoders/DFormerv2.py:247-264`、`314-321`：分解式和 Full GSA 把合成 mask 加到 query-key logits 后 softmax 的真实顺序；
- `models/encoders/DFormerv2.py:414-425`、`471-484`、`620-658`：block/layer/四级调用、前三层分解/末层 Full 和 DFormerv2-S 配置；
- `models/builder.py:226-252`：`EncoderDecoder.encode_decode/forward` 当前入口；
- `tools/evaluate_museg_checkpoint.py:93-214`、`289-304`：五尺度翻转、padding、逆翻转、原图 logits 融合与 strict checkpoint load；
- `tools/mve/dvc_a1_core.py:95-190`：原始 `Depth16` corruption mask、五 condition、确定性哈希和置零量化；
- `tools/mve/run_dvc_a1.py:448-574`：q=0 输入数组等价、finite、shape 和 JSON preflight；
- `liu-test-exp/方案1/DVG-B1-必须实现细节靶向检索步骤与WOS检索式.md`：完整实现定位、原始副本哈希核对和剩余 A/B 检索字段；
- 本地 DFormerv2 论文第 3.1–3.3 节：论文 average pooling、GSA 公式和四级金字塔文字定义。

这些入口关闭的是当前代码事实；A、B、C 的选择仍必须按第 10 节处理。

## 7. 指标与统计的基础设计

### 7.1 主要比较

初步将 `boundary-q75` 作为主要受损条件进行比较。[RE131]

`Oracle-gated - corrupted baseline`

主要指标为 Boundary IoU；mIoU 作为任务表现辅助指标。这个比较直接回答“方案是否能修复人工 q75 退化下的任务损失”，不要求 A 的问题效应先显著存在。

### 7.2 辅助比较

同时记录：

- `boundary-q25`、`boundary-q50`、`boundary-q75` 和 `nonboundary-q50` 各条件的 Oracle 增益；
- Oracle 与 clean baseline 的差异，用于检查是否损害 clean 输入；
- `boundary-q50` 和 `nonboundary-q50` 的 Oracle 增益差异，用于观察收益是否具有位置特异性；
- 每图、每组的 Boundary IoU、mIoU、有效深度比例和置零像素数；
- mask 哈希、GSA stage 传播摘要、输出有限性和原始 Label 网格恢复情况。

### 7.3 统计单位和区间

沿用 v3 的配对统计骨架：

- 先计算图像级结果，再按 location group 聚合；
- 以 138 个 location group 为 bootstrap 单位，保留组内全部图像；
- 使用预先固定的 bootstrap seed `20260908` 和 `10,000` 次有放回重采样；
- 报告 Oracle 相对 corrupted baseline 的点估计和双侧 95% percentile interval；
- 六个 mine 只作描述性分层，不当作六个独立样本。

具体的 `oracle-supported` 最小实际效应量、clean 不劣容忍度，以及 Boundary IoU 与 mIoU 是否足够或还需额外指标，仍属于第 10 节 C 的正式科学裁决冻结项。现有项目规则和已引参考不能直接给出这些数值或指标选择；用户需要补充直接参考文献，或明确把它们作为项目预注册选择。没有冻结前不物化 protocol、不运行完整评价，也不根据结果回填门槛。

## 8. 门禁状态、未来验证顺序与合法终点

### 8.1 已由项目内证据关闭的门禁

以下问题不再列为待确认，也不再扩大外部检索。这里“已关闭”表示当前代码事实已经核验，或未来实现的唯一接口/验收边界已经确定；它不表示 Oracle gate 代码已经实现，也不表示相关运行检查已经通过：

1. **GSA contribution 是否可分离：已关闭。** spatial contribution 与 depth geometry contribution 可在 `GeoPriorGen.forward` 加和前独立定位；唯一 gate 点已经固定。
2. **四级 attention 结构：代码事实已关闭。** Stage 0–2 使用 H/W 分解式 GSA，Stage 3 使用 Full GSA；“四级都接收同一 view-specific Oracle mask”是已冻结但尚未实现、尚未运行验证的设计边界。
3. **最小参数链：接口设计已关闭。** Oracle mask 未来只需沿 `EncoderDecoder` → backbone → layer → block → `GeoPriorGen` 传递，Attention 不直接接收 mask；当前代码尚未增加该参数。
4. **输入和 evaluator 骨架：可复用。** 原始 `Depth16` mask、五个 condition、确定性/嵌套/数量/哈希、五尺度翻转 10 view、右侧/底部 padding、logits 逆翻转、原始 Label 网格恢复、FP32 pre-softmax 平均均已有项目内实现。
5. **工程 preflight 骨架：可复用。** strict checkpoint load、q=0 输入数组等价、finite、shape 和 JSON 落盘均已有入口；未来只增加 gate-specific 字段。
6. **no-op 判据：验收规则已关闭。** `None`、clean、q=0、全可信 mask 必须统一进入原始 forward 旁路，逐 stage 输出和最终 pre-softmax logits 均以 `torch.equal` 为通过条件；该旁路尚未实现，也未运行等价检查。

### 8.2 仍为 `reference-blocked` 的冻结门禁

以下三组内容在补充直接参考文献或取得用户明确预注册选择前保持阻塞：

- **A：像素 corruption mask → 四级 token reliability。** 包括 view-scale resize、Stage 0–3 聚合算子、部分受损 patch 语义，以及与作者 bilinear Depth resize 的对应关系。
- **B：单 token reliability → pairwise depth contribution gate。** 包括 Full/H/W 三种 shape 的公式、query/key 组合、对称性，以及 hard gate 或 continuous attenuation。
- **C：正式科学裁决。** 包括 `oracle-supported` 的最小实际效应量、clean 不劣容忍度，以及 Boundary IoU 与 mIoU 之外是否确需额外指标。

A、B、C 任一未冻结时，不得物化 `DVG-B1-oracle-gsa-v1` protocol，不得开始代码实现或 preflight。

### 8.3 冻结后仍需单独授权的最小验证顺序

只有 A、B、C 全部关闭并由用户单独批准代码实现后，才按以下顺序推进：

1. 实现可选 Oracle 参数链和 `GeoPriorGen.forward` depth-only gate，不修改其他模型语义；
2. 先做 `None`、clean、q=0、全可信 mask 的逐 stage 与最终 logits `torch.equal` 等价检查；
3. 再做 1–2 张图的 `clean`、`boundary-q75` 和 `nonboundary-q50` preflight，核对十个 view、四级 shape、finite、原始 Label 网格和 JSON；
4. preflight 通过后，仍需另行取得完整本地 GPU paired development evaluation 授权；
5. 完整评价产物身份和配对完整性通过后，才执行 location-group bootstrap 与预注册裁决。

本次文档工作不执行上述任何一步。

### 8.4 合法终点

- `reference-blocked`：A、B 或 C 尚未冻结；准确恢复点是第 10 节对应待填字段，不进入 protocol 或代码；
- `protocol-blocked`：未来冻结后仍无法只隔离 depth contribution、mask 传播不闭合、no-op 完全等价失败、condition 或哈希不一致；保留现场，若数值语义改变则建立新 protocol identity；
- `stop`：实现错误、输出非有限、输入错位或证据链不完整；不看科学结果补洞；
- `oracle-not-supported`：实现和证据链有效，但 Oracle 没有达到预先冻结的净收益；停止该门控方向；
- `oracle-supported`：实现和证据链有效，Oracle 在预先冻结条件下有稳定净收益；只允许继续设计质量信号，不自动授权训练；
- `inconclusive`：结果方向不稳定或区间不足以裁决；只按 protocol 允许的诊断解释，不追加结果导向的条件、stage 组合和阈值。

## 9. 明确不做的事情

本基础设计不授权以下操作：

- 不修改或重跑 v3 A 实验；
- 不把 B 的结果回写成 A 的 `supported`；
- 不追加 boundary 阈值、剂量、seed 或数据组；
- 不读取 official test；
- 不训练质量预测器、恢复网络或新的 RGB-only/RGB-D 模型；
- 不使用 logits 后处理冒充 GSA 深度门控；
- 不引入三维绝对误差、risk–coverage、高置信阈值或真实矿下部署结论；低照矿山可靠感知文献 [RE049] 只作为任务可靠性背景，不扩大本轮指标或安全主张；
- 不在参考文献、实现细节和数值门槛尚未补齐前运行完整 GPU 评价。

## 10. `reference-blocked` 项与待补参考文献

本节只保留项目内证据无法唯一决定的 A、B、C。GSA contribution、四级结构、插入点、参数链、evaluator、q=0 输入、finite/shape/JSON 和 no-op 判据均已关闭，不再作为“后续所需材料”。

### A. 原始像素 corruption mask 到四级 token reliability

**所需参考文献类型：** 在层级 RGB-D、深度引导 attention、稀疏/无效深度或多尺度 confidence propagation 中，明确给出 validity/confidence/corruption mask 下采样代码或伪代码的直接来源；优先要求官方仓库、固定 commit、文件和函数。

**参考文献必须回答的精确问题：**

1. 原始 `Depth16` 布尔 corruption mask 怎样随五尺度 view resize；flip 和右侧/底部 padding 后如何保持对齐？
2. Stage 0–3 分别使用 nearest、area/average、max/any-invalid，还是连续有效比例？算子参数是什么？
3. 一个 patch 只有部分像素受损时，token reliability 是二值、有效比例、置信均值还是其他定义？
4. corrupted Depth 在 evaluator 和 `GeoPriorGen.forward` 中都采用 bilinear resize 时，mask/reliability 怎样覆盖或解释双线性插值造成的受损影响扩散？
5. 全可信输入如何保证每个 view、每个 stage 都保持全可信并进入 no-op 旁路？

**WOS 靶向检索式：**

```text
TS=((("depth validity mask" OR "depth confidence map" OR "depth reliability map" OR "corruption mask") NEAR/5 (downsampl* OR pool* OR resiz* OR "validity propagation" OR "confidence propagation")) AND ("RGB-D" OR depth OR multimodal) AND ("hierarchical transformer" OR "feature pyramid" OR "multi-scale attention" OR "token mask" OR "partial validit*"))
```

**待填字段：**

- view-scale mask/reliability 变换：`<待用户补充参考后冻结>`；
- Stage 0–3 聚合算子及参数：`<待用户补充参考后冻结>`；
- 部分受损 patch 的 reliability 定义：`<待用户补充参考后冻结>`；
- 与 bilinear Depth resize 的对齐解释：`<待用户补充参考后冻结>`；
- 论文、DOI、官方仓库、commit、文件、函数、输入输出 shape 和许可证：`<待补>`。

**关闭条件：** 只保留一套从原始 mask 到每个 view、每个 stage token reliability 的确定性规则；不得在查看 B1 结果后选择插值、pooling 或阈值。

### B. 单 token reliability 到 pairwise depth contribution gate

**所需参考文献类型：** 明确把局部 depth validity/confidence 作用到 pairwise attention bias、geometry prior 或 query-key 关系的直接实现；必须能同时解释 Full attention 和轴分解 attention，优先要求官方代码。

**参考文献必须回答的精确问题：**

1. Full GSA 的单 token reliability 怎样提升为 `[B,1,L,L]` gate？
2. 分解式 H gate 怎样形成 `[B,1,W,H,H]`，分解式 W gate 怎样形成 `[B,1,H,W,W]`？
3. query 和 key 两端采用乘积、最小值、query-only、key-only，还是其他组合；该组合是否需要保持对称？
4. 部分可信 token 使用 hard gate 还是 continuous attenuation；若连续衰减，数值范围和恒等点是什么？
5. gate 全为 1 时如何保证只恢复原 depth contribution，并且 spatial contribution 完全不变？

**WOS 靶向检索式：**

```text
TS=((("depth confidence" OR "depth reliability" OR "validity mask" OR "corruption mask") NEAR/5 ("attention bias" OR "geometry prior" OR "pairwise attention" OR "masked attention")) AND ("RGB-D" OR "depth-guided" OR multimodal) AND ("pairwise reliabilit*" OR "query-key mask*" OR "confidence gate*" OR "multiplicative mask*" OR "attention bias mask*"))
```

**待填字段：**

- Full gate 公式 `[B,1,L,L]`：`<待用户补充参考后冻结>`；
- H gate 公式 `[B,1,W,H,H]`：`<待用户补充参考后冻结>`；
- W gate 公式 `[B,1,H,W,W]`：`<待用户补充参考后冻结>`；
- query/key 组合与对称性理由：`<待用户补充参考后冻结>`；
- hard 或 continuous 选择及参数：`<待用户补充参考后冻结>`；
- 论文、DOI、官方仓库、commit、文件、函数、输入输出 shape 和许可证：`<待补>`。

**关闭条件：** 得到一套同时映射 Full 与 H/W 分解式 GSA、只乘 depth contribution、全可信恒等且不改变 spatial contribution 的唯一规则。

### C. 正式科学裁决的预注册选择

**所需参考文献或用户决定：** 现有项目统计骨架可以复用 location-group paired bootstrap，但已有参考没有直接给出 B1 的实际效应量、不劣界值或额外指标要求。用户需补充与 RGB-D 分割退化鲁棒性、Oracle/可靠性门控或 clean performance retention 接近的直接参考；若没有足够直接的参考，也可以明确授权把下列项目作为项目预注册选择，但本计划不代替用户填写数值。

**参考文献或用户决定必须回答的精确问题：**

1. `oracle-supported` 至少需要多大的 Boundary IoU 和/或 mIoU 实际净增益；判据使用点估计、95% 区间下界，还是二者联合？
2. clean 条件允许的最大退化是多少；不劣判据作用于 Boundary IoU、mIoU 还是二者？
3. Boundary IoU 与 mIoU 是否已足以裁决；若增加指标，该指标回答什么独立问题，且为何不能由现有两项覆盖？
4. 如果没有文献给出可迁移常数，是否由用户明确选择项目级门槛，并把“项目预注册选择”与“文献标准”分开表述？

**待填字段：**

- `oracle-supported` 最小实际效应量及区间规则：`<待用户补充参考或明确选择>`；
- clean 不劣容忍度及适用指标：`<待用户补充参考或明确选择>`；
- Boundary IoU/mIoU 之外的额外指标：`<待用户决定：不增加，或给出名称、职责和直接依据>`；
- 对应论文、DOI、使用段落/表格/补充协议，或用户预注册决定记录：`<待补>`。

**关闭条件：** 在任何 B1 完整结果生成前，将实际效应量、clean 不劣和指标集合一次性冻结；不得根据观察结果补门槛、换主指标或追加更有利的指标。

### 10.1 准确恢复点

当前准确恢复点是：用户仅补充 A、B、C 所需的直接参考文献，或对 C 明确作出项目预注册选择；随后只做文献—代码锚点核对并填写本节待填字段。A、B、C 全部关闭前，不创建 protocol，不修改代码，不运行 preflight、GPU、训练、云资源或 official test。

## 11. 本设计保留的相关论文编号

以下编号从“问题假设与方案假设双路径”源计划及其论文集合中筛选保留。它们用于支撑研究动机、机制边界、数据背景和协议防误判；不表示这些论文已经验证了 DVG-B1，也不为本文的具体效应量、阈值或统计门槛背书。

### 11.1 机制与干预动机

- **[PR070]** Bo-Wen Yin, Jiao-Long Cao, Ming-Ming Cheng, Qibin Hou. “DFormerv2: Geometry Self-Attention for RGBD Semantic Segmentation.” 支撑把 Depth 作为 GSA 中的几何先验，并作为定位 spatial contribution 与 depth geometry contribution 的直接方法依据。
- **[PR029]** Mianzhao Wang, Fan Shi, Xu Cheng, Chen Jia, Shengyong Chen. “Uncertainty-Aware Modality Fusion for Unaligned RGB-T Salient Object Detection.” 支撑使用局部可靠性/置信图抑制不可靠跨模态信息；不应表述为现成的 DFormerv2 GSA 实现。
- **[PR089]** Lekang Wen, Liang Liao, Jing Xiao, Mi Wang. “SGMA: Semantic-Guided Modality-Aware Segmentation for Remote Sensing with Incomplete Multimodal Data.” 支撑在模态缺失、冲突或可靠性变化时进行条件式模态调节；仅作方法动机，不替代本设计的 GSA 隔离证明。
- **[RE026]** H. R. Zhang et al. “LFR-CMT-3D: Lipschitz-regularized multi-modal fusion for robust object detection in open-pit mines.” 支撑矿山退化条件下固定多模态融合可能放大下游误差、动态调整融合权重值得验证的应用层动机。

### 11.2 条件、数据与协议边界支撑

- **[PR090]** Jiaqi Tan, Xu Zheng, Yang Liu. “Towards Robust Multi-Modal Semantic Segmentation with Teacher-Student Framework and Hybrid Prototype Distillation.” 支撑同时检查退化/缺失模态鲁棒性与 clean 完整模态性能保持；不为本文 clean 等价判据背书。
- **[PR117]** Xianjie Liu, Keren Fu, Qijun Zhao. “High-Precision Dichotomous Image Segmentation via Depth Integrity-Prior and Fine-Grained Patch Strategy.” 支撑深度内部平滑与边界锐利的结构背景，可用于理解 boundary 条件；不证明 Oracle 门控有效。
- **[RE131]** G. L. Liang. “RGB-D semantic mapping for underground robotic inspection using an attention-enhanced and boundary-refined DeepLabv3+ network.” 支撑地下 RGB-D 语义任务、边界质量和语义—几何关联的场景背景；不直接证明 DFormerv2 GSA 门控。
- **[RE326]** S. Y. Li et al. “MUSeg: A multimodal semantic segmentation dataset for complex underground mine scenes.” 支撑 MUSeg 的地下 RGB-D 数据背景；本设计实际使用的 218 张图和 138 个 location group 仍以 v3 冻结证据为准。
- **[RE095]** P. Ranasinghe et al. “Correcting time offsets and enclosure-induced measurement distortions in LiDAR-camera systems.” 支撑在比较门控收益前审计时间偏移、保护罩折射、内外参和投影误差，避免把输入错位误判为门控效果。
- **[RE049]** Y. Xin, J. Ding, Z. Zhang. “M-SURE: Enhanced and reliable safety monitoring in low-light mines.” 支撑退化条件下任务可靠性和高置信错误的研究背景；本设计不据此增加高置信阈值、risk–coverage 或部署结论。

源计划中的其他论文编号仅在其对应的 RGB-only 回退、深度补全、三维重建或风险—覆盖率实验仍被保留时引用；这些内容已被本基础设计明确排除，因此不在本节扩展为 B1 的核心依据。
