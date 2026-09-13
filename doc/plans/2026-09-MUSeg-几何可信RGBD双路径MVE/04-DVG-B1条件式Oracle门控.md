# MUSeg `DVG-B1-oracle-gsa-v1`：Oracle GSA 深度门控条件式计划

> **文档角色：** 条件式后继子计划；记录 `DVG-B1-oracle-gsa-v1` 从项目预注册规则物化到低成本预检、再到完整配对开发评价的分阶段执行设计。
> **计划状态：** A/B/C 项目预注册规则与 P0 protocol 保持冻结，P1–P4 已全部完成。P4 的 218 张图完整本地 GPU paired development evaluation 通过全部协议与完整性门禁，但主条件 Boundary IoU 和 mIoU 均下降，最终裁决为 `oracle-not-supported`；`full-evaluation.json` SHA-256 为 `f5cadf93ace37868b27ecef2f5a96c18702ba821b8eda242f3c7921982a42f12`。当前恢复点是停止本方案并回到方向级研究选择；训练、云资源和 official test 均未授权。
> **形成或核验时点：** 2026-09-13 15:05 UTC。
> **实时入口：** [`MUSeg-current-status.md`](../../main/MUSeg-current-status.md)。
> **研究选择：** [`MUSeg-open-decisions.md`](../../main/MUSeg-open-decisions.md)。
> **上级方向：** [`00-总方向规划.md`](./00-总方向规划.md)。
> **前序关系：** `DVC-A1-valdev-boundary-zero-v3-bgcontext` 已完成正式开发评价并裁决为 `not-supported`；本设计按照用户要求保留一个独立的方案验证窗口，不把 A 的失败改写为支持，也不把 B 的结果当作 A 的问题证据。
> **后继关系：** P0 protocol SHA-256 保持为 `e7b9ed0a3c84053736f70a7807bdd4f270ee5bf84a6b85ca5cfd053f9cc47e46`；P1/P2/P3/P4 已完成。P4 合法终点为 `oracle-not-supported`，本 protocol 不再安排后继评价。任何 B 结果都不能自动授权可学习质量预测、完整联合恢复、训练、云资源或 official test。

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

- 分解式 H depth contribution 为 `[B,N,W,H,H]`；对应 spatial contribution 从 `[N,H,H]` 广播到 batch 和 $W$ 轴；预注册 H pairwise gate 形状为 `[B,1,W,H,H]`，并仅在 head 轴广播；
- 分解式 W depth contribution 为 `[B,N,H,W,W]`；对应 spatial contribution 从 `[N,W,W]` 广播到 batch 和 $H$ 轴；预注册 W pairwise gate 形状为 `[B,1,H,W,W]`，并仅在 head 轴广播；
- Full GSA 的 spatial contribution 为 `[N,L,L]`，depth contribution 与合成 geometry mask 为 `[B,N,L,L]`；预注册 Full pairwise gate 形状为 `[B,1,L,L]`。

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

这里 $R$、$R_h$ 和 $R_w$ 使用第 6.6 节冻结的 query/key 两端 reliability 对称乘积。它们只乘 depth contribution；`self.weight[0]` 对应的 spatial contribution、Q/K/V、LEPE、FFN 和其他前向语义保持不变。

以下位置明确禁止作为 gate：

- 合成后的 `mask`、`mask_h` 或 `mask_w`：此时会同时修改 spatial contribution；
- `qk_mat + mask`、`qk_mat_h + mask_h` 或 `qk_mat_w + mask_w` 之后：此时 depth 与 spatial 已不可分；
- 整个 `geo_prior`：会同时移除 spatial decay，并错误波及与 Depth 无关的 rotary position encoding；
- 原始 Depth 输入 `x_e`：这会改变输入本身，不再是只门控既有 depth geometry contribution；
- `sin/cos` rotary position encoding：它只由 token 位置生成，不是 depth contribution；
- decoder 或最终 logits：这属于输出后处理，不是 GSA depth-only gate。

Attention 类不需要直接接收 Oracle mask；它继续只消费已经合成的 geometry prior。

### 6.4 已选择的 A：同一原始 mask 形成四级连续 token reliability

用户已选择把下列规则作为 **项目预注册候选**。它是基于当前 evaluator、DFormerv2 实现和全文/官方代码负证据形成的项目设计，不是 [1]–[11] 的唯一文献结论；在查看任何 B1 模型结果前只允许做算子语义纠错，不允许按性能更换插值、聚合或阈值。

#### 6.4.1 变量语义

原始 `Depth16` corruption mask 记为 $m^{raw}\in\{0,1\}^{H_0\times W_0}$：

- $m^{raw}=1$ 表示该原始像素在当前 condition 中被置零、属于受损像素；
- 原始可靠性为 $r^{raw}=1-m^{raw}$，其中 $r=1$ 表示可信、$r=0$ 表示受损；
- 全部后续 reliability 使用 FP32，合法范围为 $[0,1]$；不得结果后再二值化或引入阈值。

#### 6.4.2 与 evaluator 一致的坐标链

每个尺度和 flip view 必须严格采用以下顺序：

1. 从原始 $r^{raw}$ 出发，按 evaluator 对 RGB/Depth 使用的目标 `scaled_size_hw`，用 OpenCV `INTER_LINEAR` 从原始网格 resize 到该尺度；这一步用于与已经双线性缩放的 Depth view 对齐，五个尺度中的 `1.25` 和 `1.5` 上采样也明确使用该规则；
2. 若该 view 为水平翻转，则在 resize 后沿宽度轴翻转 reliability；不得先 flip 原始 mask 再独立计算另一套 resize；
3. 按 evaluator 的 `padded_size_hw` 只在右侧和底部 padding，padding reliability 固定为 `1`，表示 evaluator 为整除 32 新增的区域不是 corruption；
4. Stage 0–3 分别读取实际运行时的 $H_s\times W_s$，对 padded view reliability 使用 OpenCV `INTER_AREA` 聚合到该 stage 网格，得到 $r^{(s)}\in[0,1]^{B\times H_s\times W_s}$。

因此，`INTER_LINEAR` 只承担“原始 mask 事实对齐到具体 view”的坐标映射；`INTER_AREA` 只承担“view 到更低分辨率 token 网格”的有效面积聚合。所有 stage 都比 padded view 小，不存在用 `INTER_AREA` 做 stage 上采样的未定义语义。

#### 6.4.3 partial、empty、padding 与 all-1 规则

- **partial token：** $r^{(s)}$ 直接解释为该 token 覆盖域内的连续有效面积比例；由于 view 级先做了双线性对齐，更严格地说它是“平滑后的有效面积比例”，不是原始布尔像素的简单计数比例。
- **fully corrupted token：** 聚合值为 `0`，其相关 depth contribution 在 B 规则下完全关闭。
- **all-trusted token：** 聚合值为 `1`；全可信原图在每个 view、每个 stage 必须保持全 1。
- **padding：** 右/下 padding 固定为可信 `1`，不得把 padding 计作坏深度。scaled content 在 resize 后必须严格水平镜像；但每个 view 都独立保持右侧 padding，且 `INTER_AREA` 的 token pooling 相位可能随单边 padding 改变，因此完整 stage 网格在 inverse-flip 后的差异不保证只落在边缘 token。该完整网格差异只作描述性记录，每个 view/stage 必须分别与冻结的 `INTER_AREA` 结果核对，不能静默把两者强制视为镜像。
- **empty support：** 正常 `INTER_AREA` 映射中每个 stage token 必须有正面积来源。若实现报告零来源面积：只有“该 token 完全落在 evaluator padding 内”时可置 `1`；其他情况一律 `protocol-blocked`，不得用 epsilon 或默认零掩盖坐标错误。

#### 6.4.4 A 的资格检查

在模型 forward 前，用 CPU 定点检查至少覆盖：五个尺度、原图/flip、非整除 32 的右/下 padding、全 1、全 0、单像素受损和跨 token 边界的部分受损样例。必须确认：

- shape 与实际四级 $H_s\times W_s$ 一致，值域有限且位于 $[0,1]$；
- 全 1 输入在所有 view/stage 严格保持全 1；
- scaled content 必须严格满足 resize 后水平翻转；原图和 flip view 的每个 stage 都必须独立与冻结的 OpenCV `INTER_AREA` 结果一致；
- 完整 stage 网格的 inverse-flip 差异按单边 padding 与 token pooling 相位效应作描述性量化，不再错误要求差异只能位于边缘 token；
- 不创建临时 `test_*.py` 或一次性脚本；优先使用未来正式运行入口的 `--qualification-only`/等价模式落盘结构化结果。若需要新增或大幅扩展永久测试文件，先取得用户确认。

#### 6.4.5 最小参数传递链

未来若获代码授权，最小接口是增加可选参数 `oracle_corruption_mask=None`，并只沿以下链路传递：

`EncoderDecoder.forward/encode_decode` → `dformerv2.forward` → `BasicLayer.forward` → `RGBD_Block.forward` → `GeoPriorGen.forward`。

四个 stage 以及各 stage 内的所有 block 共享同一个 view-specific 原始 corruption 事实；每个 stage 按本节规则形成自己的 $r^{(s)}$。Attention 类不直接接收 Oracle mask，只继续消费合成后的 geometry prior。

### 6.5 no-op 必须走原始 forward 旁路

no-op 等价不再依赖外部浮点容差文献。`oracle_corruption_mask=None`、`clean`、q=0 和全可信 mask 必须先统一归一化到原始未修改 forward 旁路：不生成 token reliability、不构造 pairwise gate，直接执行当前 spatial/depth 合成表达式。

后续获授权实现时，必须用 `torch.equal` 同时检查：

1. 原始模型与 `oracle_corruption_mask=None` 的逐 stage 输出完全相等；
2. clean、q=0、全可信 mask 与同一旁路的逐 stage 输出完全相等；
3. 上述各组最终 FP32 pre-softmax logits 完全相等。

如果只有放宽绝对或相对浮点容差才能通过，说明实现没有进入同一原始旁路，应先修正实现，不能通过扩大 tolerance 解决。

冻结 Quick-B0 的 Ham decoder 在 eval 中仍会为 NMF2D 随机初始化 bases。P2 比较必须在每个 forward 前回放同一 CPU/CUDA RNG state，使 decoder 使用相同随机基并只隔离 `oracle_corruption_mask` 接口变量；这不是把模型改成确定性推理，也不放宽 `torch.equal`。未配对 RNG 的重复 logits 差异只证明 decoder 本身会重采样，不能用于判定 no-op 接口是否改变输出。

### 6.6 已选择的 B：query/key 两端 reliability 对称乘积 gate

用户已选择把 continuous product 作为唯一项目预注册候选。理由是 depth geometry contribution 本身描述一对 token 的深度关系；只有 query 和 key 两端都可信时才完整保留该关系。乘积同时满足连续衰减、query/key 对称、全 1 恒等和任一端为 0 时关闭该 pair。该选择不是现有论文或官方代码的唯一结论，也不允许根据未来模型结果改为 `min`、query-only、key-only、hard threshold 或 stage 子集。

令第 $s$ 级 token reliability 为 $r^{(s)}\in[0,1]^{B\times H\times W}$，且 $L=HW$：

- **Full GSA：** 先按模型 token 展平顺序得到 $r_f\in[0,1]^{B\times L}$，再定义

$$
R[b,1,i,j]=r_f[b,i]r_f[b,j],
$$

  shape 固定为 `[B,1,L,L]`。
- **H 分解 GSA：** 对每个固定宽度位置 $w$，定义

$$
R_h[b,1,w,i,j]=r^{(s)}[b,i,w]r^{(s)}[b,j,w],
$$

  shape 固定为 `[B,1,W,H,H]`。
- **W 分解 GSA：** 对每个固定高度位置 $h$，定义

$$
R_w[b,1,h,i,j]=r^{(s)}[b,h,i]r^{(s)}[b,h,j],
$$

  shape 固定为 `[B,1,H,W,W]`。

三种 gate 均只广播到 head 轴，并只进入 `self.weight[1] * (R*mask_d*)`。spatial contribution、Q/K/V、rotary encoding、Depth 输入、decoder、logits 融合与指标计算保持原样。实现资格检查必须验证对称性、shape、广播轴、值域、全 1 gate 和至少一个人工 reliability 样例的行列方向；shape 正确不替代第 6.5 节的完整 forward `torch.equal` no-op。

这里的“关闭”仅表示把对应 token pair 的 depth geometry decay contribution 置为 0；它不会删除 query/key、不会直接屏蔽 attention，也不会移除 spatial decay。P3 必须据此审计 depth/spatial contribution 隔离，不能把该 gate 解释成通用 attention mask。

### 6.7 项目内证据入口

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

这些入口关闭的是当前代码事实；A/B/C 均已写入项目预注册 protocol，当前恢复点见第 8.3 和 10.4 节。

## 7. 指标与统计的基础设计

### 7.1 主要比较

固定将 `boundary-q75` 作为主要受损条件进行比较。[RE131]

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

C 已于 2026-09-12 17:03 UTC 由用户冻结为项目预注册规则：`oracle-supported` 要求 `boundary-q75` 条件下 Oracle 相对 corrupted baseline 的 Boundary IoU 点估计至少增加 `+0.10` 个百分点，且双侧 95% percentile interval 下界严格大于 `0`；mIoU 作为否决项，其点估计不得为负。三项必须同时满足。若 Boundary IoU 点估计低于 `+0.10` 或 mIoU 点估计为负，则为 `oracle-not-supported`；若 Boundary IoU 点估计达到门槛且 mIoU 不为负，但区间下界不大于 `0`，则为 `inconclusive`。clean 不劣不设置可放宽容忍度，因为 clean、q=0、`None` 和全可信 mask 必须走原始 forward 旁路，并以逐 stage 输出和最终 logits 的 `torch.equal` 作为严格零差异门禁；任何非零差异都先判为 `protocol-blocked`，不进入科学比较。指标集合固定为 Boundary IoU 主指标和 mIoU 辅助否决指标，本轮不新增第三项指标。该规则是项目预注册选择，不是文献标准，后续不得根据结果更改。

## 8. 分阶段执行计划、授权边界与合法终点

### 8.1 当前已关闭与未关闭的门禁

- **项目内代码事实：已关闭。** GSA contribution、四级 H/W/Full 结构、唯一 depth-only 插入点、参数链、现有 evaluator 与 DVC-A1 证据骨架均已核验。
- **A 规则与实现：已关闭。** 使用第 6.4 节的 view 级 `INTER_LINEAR` 对齐、resize 后 flip、右/下可信 padding、stage 级 OpenCV `INTER_AREA` 有效面积比例；model-side 已增加 padded view 双轴整除 32、stage 精确 4/8/16/32 倍缩小的 fail-closed 守卫。
- **B 规则与实现：已关闭。** Full/H/W 使用 query/key 两端 continuous product gate，仅门控 depth geometry contribution；CPU qualification 的 shape、方向、对称性、全 1、零端点和范围检查已通过。
- **C 规则：已关闭。** `oracle-supported` 要求 Boundary IoU 点估计至少 `+0.10` 个百分点且 95% interval 下界严格大于 `0`，同时 mIoU 点估计不得为负；clean 继续采用严格原路径旁路和 `torch.equal` 零差异。该规则是项目预注册选择，不是文献标准。
- **P0 protocol：已完成。** 模板位于 `protocols/dvg-b1-oracle-gsa-v1.template.json`，物化器位于 `tools/mve/dvg_b1_protocol.py`，仓库外协议位于 `cloud/DVG-B1-oracle-gsa-v1/protocol.json`，最终 SHA-256 为 `e7b9ed0a3c84053736f70a7807bdd4f270ee5bf84a6b85ca5cfd053f9cc47e46`；allowlist 摘要位于同目录 `allowlist-summary.json`。
- **P1/P2/P3/P4：已完成。** CPU qualification SHA-256 为 `e101c972a25ada6749e7d8408172938d104ecfb4eee9266f1fbab7fc632b3dc3`；GPU no-op SHA-256 为 `a9cdbe8ca3b0ef210184499a06702103e71855e8b3529472403aae00fe31df24`；真实受损 gate preflight SHA-256 为 `bd2ec0f677b89a5a3129c1823c876b5b510262f28d8d96499d023af826c1360c`；P4 full evaluation SHA-256 为 `f5cadf93ace37868b27ecef2f5a96c18702ba821b8eda242f3c7921982a42f12`。

当前准确恢复点是 P4 的 `oracle-not-supported` 合法终点：停止本 Oracle GSA 门控方案，回到方向级计划选择新的独立候选。

### 8.2 阶段 P0：C 冻结与文档 protocol 物化（已完成）

- **完成事实：** 用户于 2026-09-12 17:03 UTC 在未查看 DVG-B1 模型结果前冻结 C；protocol 已确定性物化并达到 `protocol-ready`。大白话说，成功标准和输入身份已经先写死，下一阶段只能照此实现，不能看结果后换门槛。

- **前置输入：** 本计划第 6–10 节、v3 冻结的 218 张图/138 组身份、checkpoint/split/evaluator 哈希、`MUSeg-open-decisions.md` 第 12 节。
- **只允许改变：** 填写 `oracle-supported` 的最小 Boundary IoU 实际效应量、95% 区间规则，以及 mIoU 在裁决中的辅助/否决职责。
- **固定不变：** A/B、clean `torch.equal`、指标集合、主 condition `boundary-q75`、bootstrap seed `20260908`、10,000 次 location-group 重采样、数据范围和 official-test 拒绝状态。
- **交付物：** 在本文第 10.3 节、`MUSeg-open-decisions.md` 和 protocol 模板中出现完全一致的 C 规则；明确标记“项目预注册选择，非文献标准”。
- **计划文件：** `protocols/dvg-b1-oracle-gsa-v1.template.json`；物化器为 `tools/mve/dvg_b1_protocol.py`，其 CLI 复用现有 DVC-A1 的身份绑定方式，并显式绑定 v3 `protocol.json`、`mask-manifest.json`、allowlist 和 source allowlist summary。
- **证据位置：** 仓库外/被 Git 忽略的 `cloud/DVG-B1-oracle-gsa-v1/`，当前包含 `protocol.json` 与 `allowlist-summary.json`；protocol 绑定 checkpoint、split、v3 source protocol、v3 mask manifest、allowlist、source allowlist summary、P0 代码身份和 `official_test_included=false`。
- **通过条件：** C 已由用户明确冻结，protocol 可确定性物化，输入路径、allowlist、v3 mask manifest、代码身份和 SHA-256 完整。
- **失败终点：** C 未冻结或任何身份无法核验时为 `reference-blocked`/`protocol-blocked`；停止，不进入代码。

P0 CLI 和文件已实际存在并完成一次成功物化；该成功只验收 protocol 身份与文件完整性，不代表 A/B 算子、模型 no-op 或科学结果已经验证。

### 8.3 阶段 P1：最小代码实现与 CPU 算子 qualification（已完成）

- **授权与实现：** 用户已明确批准 P1。`oracle_corruption_mask` 已沿 `EncoderDecoder` 到 `GeoPriorGen` 传递；`None`/全零 mask 进入原始分支，非零 mask 只门控 `self.weight[1] * mask_d*`。
- **几何守卫：** padded view mask 的高宽必须都整除 32；每个 stage 必须是同一 padded view 的精确 4/8/16/32 倍整数缩小，否则 fail-closed。
- **首次合法失败：** `executions/20260913T031240259385+0000-qualification.json` 以 `protocol-blocked`、退出码 2 保留。它暴露 PyTorch area 与冻结 OpenCV `INTER_AREA` 不完全一致、W gate expected shape 缺少 channel 维，以及原先对 inverse-flip 差异“只在边缘 token”的错误假设。
- **修正：** stage 聚合改为真实 OpenCV `INTER_AREA`，W expected shape 修正；flip 审计改为 scaled content 严格镜像、每个 view/stage 独立对冻结 `INTER_AREA` 核对、完整网格 inverse-flip 差异仅作描述性记录。
- **最终证据：** `cloud/DVG-B1-oracle-gsa-v1/qualification.json`，SHA-256 `e101c972a25ada6749e7d8408172938d104ecfb4eee9266f1fbab7fc632b3dc3`；execution 为 `cloud/DVG-B1-oracle-gsa-v1/executions/20260913T035625995821+0000-qualification.json`。状态 `passed`，160 个 view-stage 的 OpenCV `INTER_AREA` 均与独立精确整数块均值参考一致，几何守卫和 Full/H/W gate 检查全部通过；`checkpoint_loaded=false`、`model_forward_executed=false`、`gpu_used=false`、`official_test_included=false`。
- **边界：** OpenCV 聚合在未来 CUDA forward 中会产生 CPU round-trip，首轮 MVE 接受其冻结数值语义，性能开销留给 P3 记录；`use_checkpoint=True` 的新增 kwargs 路径尚未执行验证。

### 8.4 阶段 P2：模型 no-op 完全等价（已完成）

- **授权与范围：** 用户已单独批准本地 GPU P2；仅加载 epoch 420 checkpoint，并对 omitted argument、显式 `None`、clean、q=0 和全可信 mask 执行真实 forward，没有运行任何受损 gate condition。
- **输入与设备：** 使用 v3 allowlist 样本 `03-01-01-0066-240526121121-08-99` 的 0.5 scale 非 flip view，padded geometry 为 `480×544`；设备为本地 NVIDIA GeForce RTX 5060 Laptop GPU，checkpoint `strict=True` 加载。
- **审计修正：** 首个结构化 P2 尝试发现四级 backbone 输出已经全部 `torch.equal`，但 Ham decoder 的 NMF2D 在 eval 中仍以 `torch.rand` 初始化 bases，未配对 RNG 时最终 logits 自然不同。该失败产物已归档到 `cloud/DVG-B1-oracle-gsa-v1/attempts/noop-equivalence-e916e463d64456f2e971b638ff1dc4385dbd13e4b19232b8bb1939558cd18ec0.json`；随后在每个比较 forward 前回放相同 CPU/CUDA RNG state，以隔离 no-op 接口变量，没有修改模型或放宽 `torch.equal`。
- **最终证据：** `cloud/DVG-B1-oracle-gsa-v1/noop-equivalence.json`，SHA-256 `a9cdbe8ca3b0ef210184499a06702103e71855e8b3529472403aae00fe31df24`；execution 为 `cloud/DVG-B1-oracle-gsa-v1/executions/20260913T035646371110+0000-noop-equivalence.json`。四类比较的四级 stage 输出和最终 FP32 pre-softmax logits 全部严格相等，`failures=[]`；`checkpoint_loaded=true`、`model_forward_executed=true`、`gpu_used=true`、`official_test_included=false`。
- **大白话说明：** 门控接口在不提供坏区时没有改变模型输出；配对 RNG 只消除了原模型 decoder 每次随机初始化带来的无关差异。

### 8.5 阶段 P3：1 样本 gate preflight（已完成）

- **授权与样本：** 用户已单独批准本地 GPU P3。按冻结 v3 mask manifest 中“最大化 `boundary-q75` 与 `nonboundary-q50` 两者较小 mask 数量，再按 q75 数量和 sample ID 决定”的确定性规则，选中 allowlist 样本 `02-01-01-0283-240524103235-08-99`；没有根据模型输出换样本。
- **条件和 view：** clean、`boundary-q75`、`nonboundary-q50` 各运行五尺度 × flip 共 10 view，总计 30 个 paired corrupted-baseline/Oracle-gated view。两类受损 mask 分别含 `22,368` 与 `14,912` 个像素，重建 SHA-256 与冻结 v3 source manifest 完全一致；q=0 量化 Depth8 与生产 Depth8 数组精确相等。clean 的 mask support 和 Depth delta 均为 0；两个受损条件的 20 个 view 均有非空 Depth delta，且变化像素全部位于对应缩放、翻转、padding 后的 mask support 内。
- **四级审计：** 两个受损条件共形成 80 个 baseline 加 80 个 gated 四级首块 GSA 审计。两条生产路径均完整捕获 stage index、前三层 H/W 与第四层 Full topology、同一 corrupted Depth 输入、有限 `sin/cos` 和有限 spatial/raw/gated depth contribution；baseline 抽样 geometry prior 精确等于 spatial contribution 加 raw depth contribution，gated 抽样 geometry prior 精确等于 spatial contribution 加 gated depth contribution。gated reliability 与预计算精确相等且非全 1，Full/H/W pairwise gate 非全 1；全部 280 个拓扑抽样重建最大误差为 `0`，gated 每个 stage 均有非零 depth gate effect。
- **配对与输出：** 每对 baseline/gated forward 前回放同一 CPU/CUDA RNG state，artifact 显式记录 Ham decoder `rand_init=true`；clean 的 10/10 view 及融合后 logits 均 `torch.equal`。两个受损条件的 20 个 view 和融合 logits 均有限且与 baseline 发生差异，所有输出恢复到原始 `932×1082` Label 网格。峰值 CUDA memory 为 `5,704,256,512` bytes。受损条件四级 GPU→CPU→OpenCV→GPU 聚合计时合计 `0.191766` 秒/20 view，约 `0.009588` 秒/view；该数字只用于 P4 预算，不外推为科学结果。
- **最终证据：** 强化后的 canonical 为 `cloud/DVG-B1-oracle-gsa-v1/gate-preflight.json`，SHA-256 `bd2ec0f677b89a5a3129c1823c876b5b510262f28d8d96499d023af826c1360c`；execution 为 `cloud/DVG-B1-oracle-gsa-v1/executions/20260913T085636085128+0000-gate-preflight.json`。状态 `passed`，`failures=[]`，`checkpoint_strict_load=true`、`gpu_used=true`、`official_test_included=false`；此前 P3 产物均归档在 `cloud/DVG-B1-oracle-gsa-v1/attempts/`，不再是 canonical。
- **大白话说明：** 强化后的 P3 同时证明了坏区确实改变对应 Depth、变化区域不越出 mask，而且 baseline 与 gated 的生产 geometry prior 只按冻结规则在 depth contribution 上不同；这只关闭实现预检，不代表 Oracle 在 218 张图上有效。

### 8.6 阶段 P4：完整 paired development evaluation（已完成）

- **授权与运行：** 用户明确授权后，以 `python tools/mve/run_dvg_b1.py --protocol D:\0Project\DFormer\cloud\DVG-B1-oracle-gsa-v1\protocol.json --mode full --device cuda` 在本地 NVIDIA GeForce RTX 5060 Laptop GPU 上执行；未训练、未使用云资源、未读取 official test。
- **固定范围：** 218 张图、138 个 location group、五个 condition、每图 10 view，共 `10,900` 个 corrupted-baseline/Oracle-gated 配对 forward pair。五个 condition JSON 均为 `completed`，每个均记录 218 样本、138 组、10 view 和 2,180 个配对。
- **完整性门禁：** P0–P3 canonical SHA 前置门禁通过；mask manifest 的 218 样本、138 组、source mask hash 和 q=0 Depth8 等价通过。clean 的 `2,180/2,180` 个 view 与 `218/218` 个融合输出严格相等；四个受损 condition 的 Depth delta 均未越出 mask support。全部输出有限并恢复原始 Label 网格，paired CPU/CUDA RNG 在每对 forward 前回放，Boundary IoU 与 mIoU 的各 condition effect 均为 `138/138` 配对组。
- **主结果：** `boundary-q75` 的 Oracle-minus-baseline Boundary IoU 点估计为 `-0.1508352015` 个百分点，95% percentile interval 为 `[-0.2676580460,-0.0462325573]`；mIoU 点估计为 `-0.2316731726` 个百分点，95% interval 为 `[-0.3914881430,-0.0919002976]`。
- **辅助结果：** boundary-q50 minus nonboundary-q50 的位置特异性 Boundary IoU 为 `-0.0926464932` 个百分点，95% interval `[-0.1885406998,-0.0032951366]`；mIoU 为 `-0.1344022510` 个百分点，95% interval `[-0.2137575906,-0.0631577481]`。这些值等于两个 q50 gain 的直接差，没有重复乘 100。
- **裁决：** Boundary IoU 点估计未达到 `+0.10`，区间下界不大于 `0`，且 mIoU 点估计为负，故按冻结 C 裁决为 `oracle-not-supported`；不得改规则补跑。
- **资源与证据：** artifact 计时 `15,827.803` 秒，execution 墙钟约 `15,860.917` 秒，峰值 CUDA memory `5,703,501,824` bytes。canonical `cloud/DVG-B1-oracle-gsa-v1/full-evaluation.json` SHA-256 为 `f5cadf93ace37868b27ecef2f5a96c18702ba821b8eda242f3c7921982a42f12`；mask manifest SHA-256 为 `23675b08f8d17fd5d528d7afe181330f39578dd08958a1447cc2761af5c68656`；execution 为 `cloud/DVG-B1-oracle-gsa-v1/executions/20260913T150505939412+0000-full.json`，`official_test_included=false`。

### 8.7 合法终点

- `reference-blocked`：P0 历史合法终点，仅适用于 C 尚未冻结；C 已关闭后不再是当前状态。
- `protocol-ready`：A/B/C、身份、模板和物化 protocol 全部冻结；只表示可以请求代码授权。
- `protocol-blocked`：规则、身份、shape、对齐、no-op 或配对完整性失败；不看科学结果补洞。
- `stop`：实现错误、输出非有限、输入错位或证据链不完整；保留现场。
- `oracle-not-supported`：有效运行未达到预注册净收益；停止该门控方向。
- `oracle-supported`：有效运行达到预注册净收益；只允许继续设计真实质量信号，不自动授权训练。
- `inconclusive`：方向或区间不足以裁决；只做 protocol 允许的诊断，不追加结果导向的候选。

## 9. 明确不做的事情

本基础设计不授权以下操作：

- 不修改或重跑 v3 A 实验；
- 不把 B 的结果回写成 A 的 `supported`；
- 不追加 boundary 阈值、剂量、seed 或数据组；
- 不读取 official test；
- 不训练质量预测器、恢复网络或新的 RGB-only/RGB-D 模型；
- 不使用 logits 后处理冒充 GSA 深度门控；
- 不引入三维绝对误差、risk–coverage、高置信阈值或真实矿下部署结论；低照矿山可靠感知文献 [RE049] 只作为任务可靠性背景，不扩大本轮指标或安全主张；
- 不在 P3 门禁通过且 P4 另行授权前运行完整 GPU 评价。

## 10. 项目预注册选择、证据边界与准确恢复点

本节区分三类事实：文献/官方代码能够直接支持的机制边界、用户已经选择的项目预注册候选，以及仍需用户冻结的科学数值。不得把后三者混写。

### 10.1 A：四级连续 token reliability 已实现并通过 P1

- **唯一候选：** 原始 mask 使用 `m=1` 表示受损、`r=1-m` 表示可信；原图到五尺度 view 使用 `INTER_LINEAR`，随后按 evaluator 顺序 flip、右/下 padding `r=1`；各 stage 用 `INTER_AREA` 聚合为连续有效面积比例。
- **语义：** partial 为平滑后的有效面积比例，fully corrupted 为 0，all-trusted 为 1；正常 stage 映射不允许空 support，完全 padding token 才可定义为 1。
- **选择理由：** 与真实 Depth view 的 bilinear 几何链对齐，同时用连续比例保留部分受损信息；避免 nearest 漏点、any-invalid 过硬、any-valid 忽略少量 corruption，以及 nconv `maxpool/4`/零 padding 破坏 all-1。
- **文献能证明：** validity 可以作为连续 confidence 传播，部分有效窗口和多尺度 confidence 有直接先例。
- **文献不能证明：** 这套五尺度 `INTER_LINEAR` + 四级 `INTER_AREA` 是 DFormerv2 的唯一正确规则；它明确是项目预注册设计。
- **禁止：** 不再继续一般性窄搜，不根据模型结果改为 nearest、max/min pooling、阈值二值化或只选择部分 stage。

### 10.2 B：Full/H/W 对称乘积 gate 已实现并通过 P1

- **唯一候选：** Full、H、W 分别使用第 6.6 节公式构造 query/key 两端 reliability 的 continuous product。
- **选择理由：** depth relationship 是 token pair 属性；乘积要求两端都可信才完整保留，保持对称、连续、全 1 恒等，并在任一端为 0 时关闭该 pair。
- **文献能证明：** reliability 乘 affinity、query-only/key-only gate、Full affinity 和连续/硬门控均存在直接机制先例。
- **文献不能证明：** product 优于 min 或单边 gate，也没有论文同时给出 DFormerv2 Full/H/W 三种映射；product 是项目预注册选择。
- **禁止：** 不根据模型结果改为 `min`、query-only、key-only、hard threshold、非对称广播或 stage 子集。

### 10.3 C：已冻结并写入 P0 protocol

以下规则已由用户于 2026-09-12 17:03 UTC 在未查看 DVG-B1 模型结果前一次性冻结：

- clean/q=0/`None`/全可信 mask 必须走原始 forward 旁路；clean 容忍度为严格 0，以逐 stage 输出和最终 logits 的 `torch.equal` 验收；
- 指标集合只使用 Boundary IoU 主指标和 mIoU 辅助指标，不增加第三项指标；
- `oracle-supported` 的 Boundary IoU 最小实际净增益为 `+0.10` 个百分点；
- Boundary IoU 联合规则为：点估计必须达到 `+0.10` 个百分点，且双侧 95% percentile interval 下界严格大于 `0`；
- mIoU 作为否决项：其点估计不得为负；
- 三项同时满足才裁决为 `oracle-supported`。Boundary IoU 点估计低于门槛或 mIoU 点估计为负时裁决为 `oracle-not-supported`；Boundary IoU 点估计达到门槛且 mIoU 不为负、但区间下界不大于 `0` 时裁决为 `inconclusive`。

**用户决定记录：** 本对话中用户选择 `+0.10` 个百分点、Boundary IoU 点估计达到门槛且区间下界 `> 0`、mIoU 点估计非负否决规则。上述内容明确属于“项目预注册选择，非文献标准”，不得在查看后续结果后更改。

### 10.4 准确恢复点

新对话按以下顺序恢复，不再重复全文或常见官方仓库检索：

1. 读取 `MUSeg-current-status.md`、本目录 `01`、`00`、本文；只在复核 A/B 证据边界时读取检索证据文档第 6.7/8 节；
2. 核对 P0–P4 canonical SHA-256：P0 `e7b9ed0a3c84053736f70a7807bdd4f270ee5bf84a6b85ca5cfd053f9cc47e46`、P1 `e101c972a25ada6749e7d8408172938d104ecfb4eee9266f1fbab7fc632b3dc3`、P2 `a9cdbe8ca3b0ef210184499a06702103e71855e8b3529472403aae00fe31df24`、P3 `bd2ec0f677b89a5a3129c1823c876b5b510262f28d8d96499d023af826c1360c`、P4 `f5cadf93ace37868b27ecef2f5a96c18702ba821b8eda242f3c7921982a42f12`；
3. 保留 P4 `oracle-not-supported` 终点，不在 v1 下修改 A/B/C、追加样本、阈值、condition、stage 或 seed；
4. 若继续研究，从方向级计划选择新的独立问题或方案，并重新冻结 protocol identity 与授权边界；
5. 训练、云资源和 official test 均需新的单独授权，P4 完成不构成后继授权。

当前已完成 P0–P4，DVG-B1 v1 已收口。主条件 Boundary IoU 与 mIoU 均为负；训练、云资源和 official test 未运行。

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
