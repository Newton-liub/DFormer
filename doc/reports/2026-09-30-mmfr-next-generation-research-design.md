# 下一代 MMFR：A-v1 任务效用控制的单点残差修正

- 日期：2026-09-30；对象：上级模型研究审核。
- 状态：已审核的 A-v1 固定结构现已最小实现并通过 **Gate-B 工程核验**；直接证据、独立协议与停止点见 [实现审核报告](../../MMFR/02_evidence/report_mmfr_a_v1_implementation_gateb_20260930.md)。后续本地准备已冻结第一轮正式合同，等待4090全尺寸Proposal/Gate各3-update预检的独立云端授权；正式训练仍未授权，不代表方法有效。本报告保留设计推导；§7–8 的 design-ready/待实现表述是前一轮设计历史，§9与独立protocol记录本次正式合同，不覆盖原始实验结果。B 保留为已淘汰主创新路线，C 保留为未来性能备选，两者不再展开。
- 范围：沿用初版的 F-lite / R-OE-lite v2 事实、九篇本地论文机制与 DFormerv2-S + HAM 接口证据；本轮只修订 A，补一次窄范围动作选择近邻检查，不重新通读九篇或设计其他候选。基准 Git HEAD：`a7c5cc17c5b495f2210efa3604b05794b5fa16c4`，开始时工作区已有未提交改动。
- 授权：原设计与最小实现/Gate-B轮已完成并停止。后续用户本轮只授权冻结正式合同、身份核验、一个本地Git commit与交接文档；未正式训练、未评价、未访问official test、未操作云端或push；正式训练仍须单独授权，Git回执以实时入口为准。
- 事实与执行边界仍以 [当前状态](../main/MUSeg-current-status.md) 和 [开放决策](../main/MUSeg-open-decisions.md) 为准。本报告不替代历史裁决，也不授权实验。

## 1. 结论先行：现有实验真正提出的问题

**候选 A 已收敛为 A-v1：任务效用控制的单点残差修正。** 保留 frozen C0、stage index 2 单点残差、每张样本一个连续 gate，以及 Proposal → Gate 两阶段。用同权重、同输入、同随机状态的 off/full 任务损失差监督动作是否值得使用；首版只允许一个新 checkpoint 和一次四条件 Quick-Val 的 off/full/learned 对照。暂不加入空间可靠性图、第二模态编码器、重建器、teacher 或 prototype。

大白话：F-lite 的 residual 本来就是随输入变化的 `R(F)`，但它缺少被独立监督、能显式拒绝或减弱干预的 intervention selector（补偿动作选择器）；补偿路径在所有输入上持续生效，没有直接学习“此次补偿是否值得执行”。A-v1 的核心差异是**学习补偿动作是否值得执行，而不是让 residual 首次具有输入条件性**。R-OE-lite 只在整幅深度都空时工作，现有对照又未分离恢复器与主网络变化的贡献。下一步只验证：固定动作有没有可利用的样本差异收益，以及 selector 能否比始终补偿更好地利用它。

这是研究取舍，不是实验结论。既有证据尚不能证明可靠性控制必胜、F-lite 因果性退化、substitute 净收益为零、或所推荐机制具有足够发表的新颖性。最接近的 MoSA 已包含空间调制 adapter 与可靠性融合，DCF 已使用任务效果比较构造深度质量监督；A 的有限差异必须落在**同权重、同输入下对具体补偿动作的增益监督**，不能包装成首次提出 task-aware reliability。

### 1.1 关键概念与边界

- **MMFR**：本项目的多形式模态失效与可靠性研究，关注缺失、局部丢失、噪声、模糊、量化与错位等 Depth 故障。
- **C0**：E1 Batch 1A 不含 F-lite 的训练对照。它不是新方案冻结前的任意 baseline；本报告推荐直接绑定已核验的 C0 fixed-final 权重。
- **Quick-Val / Main-Val**：分别是四条件单视图筛选与十条件十视图正式开发集评价。不同口径的绝对分数不能互比；筛选正收益也不等于正式口径已确认收益。
- **物理有效性 / 模态质量 / 预测置信度 / 动作效用**：非零深度是否存在、信息是否可信、模型有多自信、某个干预是否改善任务，是四个不同对象。非零不保证正确，低置信不保证补偿有用，低质量也不保证应该关闭几何。
- **同权重 on/off**：固定全部主模型参数与随机状态，只改变补偿是否作用；它估计该动作对固定模型的贡献。不同训练 run 的比较不能代替它。
- **始终补偿 / 无条件残差**：指补偿路径没有独立的执行选择器，不代表 `R(F)` 与输入无关或所有样本得到相同残差。
- **保真**：零初始化或严格 bypass 可保留原路径；训练后的非零残差，即使 base 冻结，也不自动保证 clean 不变。

## 2. F-lite 与 R-OE-lite 给出的共同教训

### 2.1 F-lite：residual 已依赖输入，但缺少独立监督的动作选择器

证据：[Batch 1A Main-Val 正式分析](2026-09-22-mmfr-e1-batch1a-c0-flite-mainval.md)，以及本地两侧 `quickval-original-full/summary.json` 与 `mainval-10cond-v1/update-2560/summary.json`。两侧均完成 2560 次成功训练更新，评价样本为全部 318 条 `val-dev`。

四条件单视图 Quick-Val：clean `53.46 → 54.15`（`+0.69 pp`）；entire_missing `48.76 → 50.17`（`+1.41`）；spatial_dropout `51.40 → 52.39`（`+0.99`）；misalignment `52.15 → 52.63`（`+0.48`）；三个 hard 条件平均差约 `+0.96 pp`，历史 screening 为 promote。

十条件十视图 Main-Val：主指标 M6（六个单故障条件 mIoU 的未加权平均）`55.2883 → 54.8917`，差 `−0.3966 pp`；clean `56.69 → 56.02`，差 `−0.67`；仅 spatial_dropout `+0.40`、spatial_dropout+noise `+0.42`，其余八条件为负。四个共同条件有三个收益符号翻转。这证明**收益未跨评价口径复现**，并不证明翻转由某个单一机制造成。

机制解释与证据强度分开：

1. **已核验结构事实**：F-lite 在编码器输出的三个 stage 使用 `1×1 down → GELU → zero-init 1×1 up` 残差，共 173,152 个参数。残差内容 `R(F)` 随输入 feature 变化，并非所有样本加入相同补偿；它缺少被独立监督、可显式拒绝或减弱干预的 selector。零初始化只保证开始时不干预，不保证训练后的逐样本收益。
2. **合理假设：持续启用的补偿路径会产生正负收益混合。** 在几何已有帮助的样本上也执行输入相关残差，可能造成不必要干预。但没有同权重 adapter-off 对照、残差与局部深度质量的关联或视图级分析，**尚未证明对可靠 Depth 过度修改**。
3. **合理假设：缺少独立监督的干预选择导致跨视图不稳定。** 特征尺度、插值、翻转、多视图 logit 融合会改变残差的效果；三项翻转支持研究这个问题，却不能单独归因于 adapter 的尺度敏感性。
4. **“融合权重固定”的表述需要修正。** 当前骨干的几何先验含学习权重，但没有运行时显式模态质量控制；它并不是 RGB/Depth 两支独立特征之间的固定加权融合。下一步应针对实际接口，而非虚构的 fusion 层。
5. **替代解释不能排除。** 单 seed、无置信区间、无 Main-Val 预注册数值门槛；adapter 初始化消耗随机数导致历史已接受的训练采样顺序差异；clean 差异高度集中于少数类别。不能把这些描述性分数写成已证实的模块退化，也不能仅归结为参数/更新次数不足。

### 2.2 R-OE-lite v2：路由确实工作，但只回答了很窄的输入状态

证据：[v2 正式训练与 Quick-Val 报告](../../MMFR/02_evidence/report_e1_batch1b_roe_v2_formal_training_quickval_20260924.md)。本轮复核其正式报告；报告指向的 `cloud/mmfr-e1-batch1b-roe-v2/quickval-comparison.json` 在当前本机未找到，**不声称本轮直接复核该原 JSON**。

4090 上完成 2560/2560 更新，substitute 的 14 个参数有 2272 次 Adam 更新，机制不是“没训练起来”。四条件 Quick-Val 相对 C0 为 `0.00 / +0.01 / +0.01 / +0.01 pp`，结论 inconclusive。entire_missing `318/318` 触发，其他三个条件 `0/318` 触发且走 strict exact bypass。

1. **覆盖过窄是直接结构事实。** 全图仍有非零 Depth 时，即使局部大片缺失或错位，也不调用 substitute。它无法作为这些非空故障的主动补偿器。
2. **“完全空才恢复是否太晚”是干预时机假设。** 整体任务仍能在全空输入下工作，并不存在已证明的不可逆时间过程；准确说法是触发边界太极端，错过了大量非空退化输入。
3. **重建未必是瓶颈。** Depth-only recovery 目标与语义分割增益不等价。恢复看起来更完整的深度，也可能制造不可靠几何；但尚无同权重 bypass，不能宣布重建路线本质无效。
4. **`+0.01` 不能记为 substitute 收益。** 三个 strict-bypass 条件仍有约半数逐样本结果与 C0 不同，说明 base 漂移足以产生这一量级的聚合变化。EM 同样不能从不同权重的总差里拆出 substitute 净贡献。
5. **连续控制可以扩大覆盖，但不会自动学会效用。** 若监督只是零值比例、重建误差或 softmax confidence，可能把“看起来坏”误当成“补偿会有效”。hard routing 换成 sigmoid 不是问题的完整解答。
6. **评价资格仍待处置。** R-OE-aware Quick-Val 新入口无既有冻结资格，original-full 的 V_geom 全 1 是新决定；其数值只作 screening 参考，不能在本轮把它升级为正式确认。

### 2.3 两条路线的共同教训

- **干预的覆盖范围和干预的有益性是两个问题。** F-lite 广覆盖、残差内容依赖输入，但没有独立监督的执行选择；R-OE-lite 按全空状态选择但覆盖很窄；下一步须分别检验“可补偿”与“可判断”。
- **恢复深度与提高分割不是同一个目标。** 修正对象应直接服务当前任务，而非因为深度重建可做就默认它必要。
- **要把主网络变化与新增动作分开。** 推荐第一次机制验证冻结 base，并保留同权重 off / full / learned 三种行为，避免再次陷入 aggregate delta 无法归因。
- **先证实有效动作，再学习使用时机。** 如果固定补偿在训练开发证据上没有可选收益，gate 不能凭空产生信息；停止比堆第二个模块更合理。
- **clean 保护是约束和验收项，不是零初始化的后续保证。** 同样不应用全空标签去推断传感器失效原因；自然全空和生成器全空可能在可观测输入上无法区分。

### 2.4 旧 Oracle-A 限制了哪类方案

[Oracle-A 正式记录](2026-09-19-museg-mmfr-oracle-a-validity-aware-pairwise-geometry.md) 已裁决 validity-aware geometry suppression 为 NO-GO：使用完美有效性抑制 Depth 几何时 clean `−3.48 pp`、SD `−1.77`；strict 对 geometry_off 的 SD 仅 `+0.04`。geometry_off 的 clean `−4.45` 说明既有几何有价值；clean 自然无效 Depth 平均约 30%，不能把 clean 假设成全有效。

本轮遵守该已关闭动作：**任何候选都不通过 validity/confidence gate 去衰减或关闭 pairwise Depth geometry prior**。连续预测 gate 也不能换名复活它。该结果限制的是这个干预动作，不是证明所有 reliability 研究失败。若未来要重新研究内部几何动作，必须先申请单独复议，不能夹带在本方案中。

## 3. 九篇论文：仅提取影响本次设计的机制

机器身份源：[PAPER_LIBRARY_INDEX.json](../../MMFR/03_reference/PAPER_LIBRARY_INDEX.json)；浏览入口：[PAPER_LIBRARY_INDEX.md](../../MMFR/03_reference/PAPER_LIBRARY_INDEX.md)。全文均为仓库外 `D:\0Project\origin\论文\<canonical bundle>\<同名>.md`；RobustSeg 文件夹使用索引中的截断名。以下不比较外部性能数字，因此未读取 xlsx 作为性能依据。结构辅助查看了 MoSA、GeomPrompt 及四篇融合论文相关结构图；原任务和 MMFR 的差异始终保留。

### 3.1 LIB000014 / PR089 — SGMA: Semantic-Guided Modality-Aware Segmentation for Remote Sensing with Incomplete Multimodal Data

- **核心机制**：共享编码器分别提取模态特征，利用多尺度类别语义原型与鲁棒性相关注意力进行融合；Modality-Aware Sampling（MAS）按模态表现调整采样。定位：§III.B–E，SP/RP 式(3)–(10)、MAS 式(11)、两分支 CE 式(12)–(14)，Fig.2；可靠性没有独立标签/损失。
- **启发**：可靠性应与类别语义、任务上下文关联；训练时应真实覆盖缺失状态，而非只有推理路由。
- **已有思想**：语义引导 attention、prototype、模态鲁棒性驱动采样，均不是本项目新贡献。
- **不能照搬**：遥感多模态、独立模态特征与类别原型结构不等于当前 DFormerv2 几何输入接口。attention 不是经过动作增益标定的可靠性概率；不能仅加 prototype 就宣称语义可靠性创新。

### 3.2 LIB000023 / PR090 — Towards Robust Multi-Modal Semantic Segmentation with Teacher-Student Framework and Hybrid Prototype Distillation（RobustSeg）

- **核心机制**：完整模态 teacher 与 Anymodal Dropout student；logit KL、跨模态随机匹配的类别 prototype、dominant modality 的 intra-class feature variation，以及弱模态 student 向 teacher 的反馈。定位：§3.1–3.3，式(4)–(9)，Fig.4–5；最终推理保留哪一支在本次所读正文中未明确，不作推断。
- **启发**：缺失鲁棒性可来自训练信息迁移，不一定需要恢复输入；类别关系比纯像素模仿能携带更多结构。
- **已有思想**：teacher-student、modality dropout、类别原型及特征变异蒸馏。
- **不能照搬**：需要模态专属 feature 与复杂双向训练。DFormerv2 没有这些分支；把全部机制搬来会扩大模型/显存并混淆单核心假设。轻量 logit KD 也不能冒称忠实复现。

### 3.3 LIB000024 / PR029 — Uncertainty-Aware Modality Fusion for Unaligned RGB-T Salient Object Detection（UMFNet）

- **核心机制**：双流特征的像素级 Gaussian latent、均值/方差重参数化，方差派生 confidence 控制融合，训练包含显著性、边界和 KL 项。定位：§3.2–3.4，式(1)–(9)，Fig.2。
- **启发**：空间变化的 uncertainty 可以作为调节信息注入的信号，而非整图开关。
- **已有思想**：latent uncertainty → confidence → 自适应特征融合。
- **不能照搬**：RGB-T 显著性与 RGB-D 语义分割不同；它没有显式位移场/错位标签监督，latent variance 不能直接解释为几何错位概率。两套编码器、随机 latent、额外边界任务对当前首阶段过重。

### 3.4 LIB000026 / AI023 — GeomPrompt: Geometric Prompt Learning for RGB-D Semantic Segmentation Under Missing and Degraded Depth

- **核心机制**：冻结分割器，RGB 几何 prompt 与 RGB+degraded Depth recovery 生成输入侧几何信息；有 bounded residual、zero initialization 及平滑/幅度约束，以分割监督驱动。定位：§3.1–3.4，式(1)–(3)，Fig.2，合成退化见 §4.1。
- **启发**：输入几何可由任务监督校准，完整稠密 Depth GT 未必必要。
- **已有思想**：冻结分割器、任务驱动深度补偿、有界零初始化残差。
- **不能照搬**：原文评测的 DFormer 不等同于 DFormerv2；输入侧改动仍会改变正常几何，低通等后续处理也不等于 exact identity。前骨干模块训练需要保留穿过骨干的梯度，少参数不保证少显存。直接采用 recovery 作为下一代 MMFR 很难形成新颖性。

### 3.5 LIB000027 / AI017 — Calibrated RGB-D Salient Object Detection（DCF）

- **核心机制**：根据 RGB-only / Depth-only 显著图 IoU 挑选正负深度质量样本，ResNet18 估计图像级 P_pos；RGB 估计深度，按 `P_pos × raw + (1−P_pos) × estimated` 校准输入。另含通道融合与 triplet 训练。定位：§3.2.1–3.2.2，式(1)，Fig.3。
- **启发**：质量标签可以依据任务效果比较，而不是单纯深度零值或重建误差。
- **已有思想**：**任务相关相对质量判断已经存在**，连续混合真实/估计 Depth 也已经存在。
- **不能照搬**：图像级 SOD 质量标签不等于空间几何可靠性；它比较模态分割能力，不是比较同一模型的特定补偿动作 on/off。输入重建仍可能制造错误几何；不可宣称 A 首次把 reliability 与任务关联。

### 3.6 LIB000028 / MoSA — Modality-Aware Spatially-Adaptive Adaptation for RGB-X Semantic Segmentation

- **核心机制**：冻结 SAM 的双模态编码路径；MS-SMA 用全局和局部内容生成空间系数，调制低秩 additive adapters；RGCF 用 feature mean/std/max 预测位置级可靠性并归一化融合；训练有模态辅助预测头。定位：§IV.A 式(1)–(5)、§IV.B 式(6)–(15)、§IV.C 式(16)–(20)，Fig.2。
- **启发**：显式内容条件控制比总是加入残差更有表达能力；需要单独讨论 gate 学到的是什么。
- **已有思想**：**空间 gate + adapter、冻结 base、zero-init up、reliability-aware fusion 都已出现**。“F-lite 再加质量图”是本轮最严重的直接撞车方案。
- **原文未解问题**：§IV.B 式(15)及随后文字用 maximum softmax probability 作 soft target，紧邻段落/Fig.2 却称与 GT 比较的 correctness。当前本地正文存在定义不一致；尚未核对对应实现或原版公式图，不能擅自改写为准确的 GT-correctness supervision。
- **不能照搬**：DFormerv2 没有双模态 SAM features。即使修改为单分支 gate，结构本身仍高度相邻；差异必须由行动相对监督证明，而非减少参数或换 backbone。

### 3.7 LIB000029 / AI019 — MaskMentor: Unlocking the Potential of Masked Self-Teaching for Missing Modality RGB-D Semantic Segmentation

- **核心机制**：多模态 masked image modeling 预训练，共享权重 teacher/student，多次 teacher 可见视图与单次 student masked 视图，pixel/token reconstruction 和自教。定位：§3.1–3.5，Algorithm 1、Fig.2；teacher 三次更新后 student 一次，共享权重，微调时移除 MIM 头。
- **启发**：可以训练可缺失的语义表征，而非必须重新合成缺失模态。
- **已有思想**：多模态遮蔽、自教、重建与信息迁移。
- **不能照搬**：原设含 RGB、Depth、semantic 信息及预训练过程，不是一个 CE 上附加 logit KD 项；完整路线的成本远大于轻量补偿筛选。小规模训练若不奏效，也不能反向裁决 MaskMentor 无效。

### 3.8 LIB000030 / AI024 — Toward Reliable RGB-D Semantic Segmentation: Handling Missing Modalities via Condition Dropout

- **核心机制**：冻结原编码器与解码器，复制可训练完整 encoder，四级 zero-conv 注入；RGB-D、RGB-only、Depth-only 三种状态按各 1/3 训练。定位：§II.A–D，式(1)–(4)，Fig.2；辅助编码器参与推理。
- **启发**：显式经历缺失输入、同时保留原分割路径，可以提供不用重建器的替代方案。
- **已有思想**：模态状态采样、冻结路径与零初始化条件支路。
- **不能照搬**：完整 encoder copy 不等于 F-lite；其三种状态也不覆盖全部局部退化/错位。我们只研究 Depth failure，不能为忠实复现默认扩展到 RGB 缺失，更不能沿用旧冻结训练合同而偷偷改变分布。

### 3.9 LIB000008 / 无已确认 PR/AI 编号 — Learning Selective Mutual Attention and Contrast for RGB-D Saliency Detection（SMAC）

- **核心机制**：跨模态 mutual attention 与 contrast；RGB 估计 Depth feature，由估计残差生成图像级选择系数 α，控制 Depth 引导的信息交互。定位：§4.2–4.4 式(8)–(15)，Fig.7–8；§5/§6.2 为训练及推理。
- **启发**：RGB/Depth 不一致可提供可观测选择信号，整图连续控制也有先例。
- **已有思想**：RGB 引导特征估计、残差派生选择、控制辅助信息注入。
- **不能照搬**：估计残差同时受语义内容、纹理与估计器误差影响，不是独立监督的传感器 reliability；α 是样本级，不是局部可靠性图。SOD 双流训练和 RGB 流输出不等同于几何条件编码器。

### 3.10 机制归纳后的取舍

- **物理有效性 gate**：能检查存在性，不能告诉我们几何或补偿是否有益；旧 Oracle 禁止直接抑制几何。
- **confidence / uncertainty fusion**：适用于有模态专属特征的架构；当前直接迁移需造新分支，且 MoSA/UMFNet 已覆盖主要思路。把 confidence 当 calibrated failure probability 没有证据。
- **语义引导可靠性**：SGMA 与 DCF 支持任务关联，但 prototype 不是首阶段所必需；可先使用已有语义 feature，不再造第二套语义监督。
- **spatial soft gate**：结构合理且对局部故障有潜力，但 MoSA 撞车直接，逐位置的任务损失差也不是局部动作因果贡献。首阶段选择更可核验的样本级动作控制。
- **feature calibration**：与现有 post-backbone 接口最相容，低额外参数；新增点应是独立监督动作是否值得执行，而不是把已有输入相关残差改名。
- **reconstruction / substitution**：依然是候选，但任务动机和正常输入保护必须先说明；R-OE 的小差不够证明它无效，也不够支持继续扩大。
- **masked / condition dropout / teacher / prototype**：成熟而重要的训练路线，适合作为未来性能竞争参照；首阶段同时叠加会令 core mechanism 不可辨识。

## 4. DFormerv2 实际接口决定了什么能做

本轮只读核验：`models/encoders/DFormerv2.py` 的 `dformerv2.forward`、`GeoPriorGen`、`DFormerv2_S`；`models/builder.py` 的 `EncoderDecoder.encode_decode` 和 `reliability_auxiliary_loss`；`models/feature_adapter.py` 的 F-lite 路径；HAM 的随机 NMF 行为由正式 Main-Val 报告确认。

- Depth 的 normalized channel 0 参与 pairwise 几何先验；先验结合 position decay 与 depth-difference decay，Depth 随 stage resize。**没有独立 Depth encoder 输出可供两模态加权融合。**
- 四级 RGB 为主、受几何条件影响的 feature channels 为 `[64,128,256,512]`。现有插入顺序是 `backbone → optional feature_adapter → decode_head → resize`。
- 因此可用接口是：输入 Depth 校准、post-backbone feature 残差、或训练信息迁移。内部几何衰减不在当前候选范围。
- 既有 reliability auxiliary 是 Depth-only BCE 辅助目标，不把 reliability 返回分割路径，也没有对特定干预收益标定；不能直接把它的输出升级为已学好的融合 gate。
- HAM 含随机 NMF。比较 on/off/full、构造效用标签时必须从同一前向随机状态回放；freeze/eval 本身不能取代随机控制。

## 5. 三个候选：一个问题、一个机制、必要训练约束

### 5.1 候选 A-v1：任务效用控制的单点残差修正（design-ready）

**固定研究问题**：对同一个 frozen C0、同一输入和同一随机状态，一个固定 residual compensation action（残差补偿动作）是否存在样本间不同的任务收益；模型能否学习何时使用该动作，而不是始终补偿？“固定动作”指 proposal 参数在 Gate 阶段固定，并非其输出在所有输入上相同。

#### 5.1.1 最小结构与可观测输入

路径为 `C0 backbone → stage-2 feature → residual proposal → gated residual → 原 decoder/HAM`。只在 **0-based stage index 2、256 channels、约输入 1/16 分辨率**插入，其他三个 stage 原样。

Proposal 是轻量 `1×1 down → GELU → DWConv 3×3 → GELU → 1×1 up`，bottleneck 暂保留 32，`W_up` zero-init；不加 norm、attention、transformer、多 stage adapter、prototype、reconstruction head 或第二 encoder。

$$
R_\theta(F)=W_{up}\,\mathrm{GELU}\bigl(\mathrm{DWConv}_{3\times3}(\mathrm{GELU}(W_{down}F))\bigr),\qquad
F'=F+g_\phi(F,s_D)R_\theta(F)
$$

**每张样本一个 $g\in[0,1]$**，由 sigmoid 输出，控制这张图的残差强度。首版没有额外幅度裁剪或归一化控制；先验证 action utility，后续只有实际观察到残差爆炸才另议 bounded residual。zero-init、clean consistency 和 explicit off/full/learned 均保留；零初始化不保证训练后的 clean 保真。

Gate 输入为 stage-2 pooled semantic feature，加少量推理可观察的 Depth statistics：zero ratio、non-zero mean、non-zero std、邻域 depth difference。它们只是 **observable context（可观测上下文），不是 reliability ground truth**。统计使用当前 observed Depth 的合法区域，空集合固定取值并登记；若某项工程实现麻烦，可以删减，并同步输入维度，不为凑齐四项引入额外模块。禁止把 cause、severity、corruption metadata、未损坏 Depth 或分割标签作为 gate 输入。

Base 参数、BN running statistics 和运行模式全部固定到绑定 C0 的 frozen-base 定义。只设置 `requires_grad=False` 不足以冻结 BN；HAM 的随机 NMF 仍需显式随机状态回放。Proposal/Gate 不用额外随机层。

#### 5.1.2 Phase 1：Proposal，只训练动作

冻结 C0，固定 $g=1$，只训练 proposal，不训练 selector。使用既有 `train-dev`、项目已有 Depth corruption 类型与合法采样框架，不新增 failure 类型。目标只保留 segmentation CE 与 clean 训练输入上的必要 base consistency：

$$
\mathcal L_{proposal}=\mathrm{CE}(P_{full},Y)
+\lambda_{clean}\mathbf1_{train\ clean}\,\mathrm{KL}(P_{off}\Vert P_{full})
$$

Base reference detached，不引入独立 teacher 参数。clean consistency 用同一 clean 输入、同一随机状态的 fixed-base 输出；它是训练保护，不是无损保证。阶段末只用训练内 off/full 损失诊断动作是否有可利用收益，禁止用 `val-dev` 选阶段长度、权重或 checkpoint。**若 proposal 没有可利用收益，停止，不进入 Gate。**

#### 5.1.3 Phase 2：Gate，冻结动作后训练选择

冻结 C0 与已经训练完的 proposal，只训练 gate；冻结包括参数、运行统计与模式。这样避免 utility target 随 proposal 漂移，也避免 gate 提前收缩让 proposal 学不到有效动作。

仍用同一 `train-dev`，无需新划分或 cross-validation；但每个训练样本在 Gate 阶段**必须通过已有合法框架重新采样新的 Depth corruption realization（本次具体破坏实例）**。使用与 Proposal 阶段隔离的阶段随机流/种子，不直接复用该阶段的固定 corruption 张量、实例缓存或已生成 target。新阶段采样记录只用于复现，不进入 gate。重新采样不等于保证张量从未重复：clean 与全空等确定性状态可以相同；关键是随机退化实例不从 Proposal 训练缓存继承。这降低针对具体 corruption 实例的训练记忆风险，**不消除同一 train-dev 上的过拟合，也不证明泛化**。

先生成当前 observed 输入，再在 off/full 两个分支复用这一个输入；两个分支绝不能各自重采样。要求相同模型权重、相同输入、相同 HAM/stochastic state，**唯一区别是 residual action on/off**。gated 分支也回放该输入的相同随机状态，避免把随机输出差异混入监督。

$$
L_{off}(x,Y)=\mathrm{CE}(P_{off}(x),Y),\qquad
L_{full}(x,Y)=\mathrm{CE}(P_{full}(x),Y),\qquad
u(x)=\operatorname{stopgrad}\bigl(L_{off}-L_{full}\bigr)
$$

CE 是每张图当前有效标签上的平均交叉熵；两支使用同一个 label mask 和 reduction。设一个在运行前固定的 **margin $m>0$**，单位为上述平均 CE，不是 mIoU pp。标签仅按下面三种情况定义：

- $u>m$：full 明确有益，$t=1$；
- $u<-m$：full 明确有害，$t=0$；
- $|u|\le m$：ambiguous（收益差不足以分类），不参与 gate BCE。

令 $M_i=\mathbf1[|u_i|>m]$，gate classification 仅对非 ambiguous 样本取均值：

$$
\mathcal L_{cls}=\frac{\sum_{i:M_i=1}\mathrm{BCE}(g_i,t_i)}{\max(1,\sum_i M_i)},\qquad
\mathcal L_{gate}=\mathcal L_{cls}+\mathrm{CE}(P_g,Y)
+\lambda_{clean}\mathbf1_{train\ clean}\,\mathrm{KL}(P_{off}\Vert P_g)
$$

全部 ambiguous 时 classification 项为零，仍可参与 gated segmentation CE 与适用的 clean consistency；不要误把“未参与分类”写成整张样本不训练。BCE 与 segmentation CE 系数首版固定为 1，只留下必要 clean consistency 权重待定，不新增其他 loss。target 和 fixed-base reference 均 detached；proposal/base 不进 optimizer，也不更新梯度或统计。

**标签可用于训练监督，故障元数据不可作为推理输入。** $Y$ 只用来构造训练 utility 和 segmentation loss，推理不需要标签、不计算 off/full 两次输出。$m$ 仅根据 `train-dev` 训练内收益尺度确定后固定，禁止用 `val-dev` 调参；保留正/负/ambiguous 数量和正负组的简单 gate 均值作为训练内监督检查，避免 target 全部模糊时仍声称学到了选择。

Gate 输出是**当前 residual action 是否值得使用的控制量**，不是 calibrated reliability probability、Depth failure probability 或 sensor confidence。两个 endpoint 的收益监督不保证中间幅度的收益单调，gated CE 用来直接约束连续控制的任务输出，但仍没有逐样本收益保证。

#### 5.1.4 样本级能力上限与模态失效解释边界

本阶段保留 sample-level gate，不升级 pixel/patch gate、feature-map reliability、spatial reliability 或 local uncertainty map，理由为：先验证 action utility 值不值得研究；spatial gate 与 MoSA 更接近；HAM/decoder 非局部作用使 feature 位置的因果 utility label 难定义；若样本级 selector 都不能优于 full，暂时没有理由增加空间复杂度。

Residual 内容仍随空间位置变化，但一个 $g$ 控制全图。**A-v1 无法分别处理同一张图中的局部可靠 Depth 与局部退化 Depth。** 两个整幅输出的像素 CE 差不是局部残差动作的因果贡献，不能据此包装成已解决 spatial reliability。

为避免退化成普通 dynamic adapter，第一次四条件 Quick-Val 必须同时记录 learned gate 的 **mean、median**，仅在必要时加少量分位数；条件只含 clean、entire_missing、spatial_dropout、misalignment。统计回答“Depth degradation 状态变化时，使用补偿动作的倾向是否也发生有意义变化”，不预设 gate 随退化单调增大。即使 entire_missing 的 $g$ 更高，也只能说**不同模态状态下观察到不同 action-selection behavior**，不能说学会了 Depth reliability。需结合动作收益与任务结果判断；若只有普通内容路由收益而无模态状态关联，模态失效研究解释未获支持。

#### 5.1.5 推理、成本与近邻边界

推理只做一次 backbone、gate/proposal、HAM；没有标签、完整 Depth、teacher 或反事实前向。off 必须 strict bypass，不能用 $g=0$ 的数学等价替代数值验收；full 固定 $g=1$，learned 用连续 gate，三者共享同一 checkpoint 的全部权重。

结构草算（尚未实现、不是实测）：256→32→256 proposal 含 DWConv/bias 约 16,992 参数；四统计时 pooled 256+4 输入的 gate 为 260→16→1，约 4,193 参数，总约 21,185。统计删减时计数同步调整。低分辨率轻量残差不等于已验证 latency/显存；训练可 no-grad backbone，仍须保留穿过 frozen HAM 的梯度图。Gate 的两个 endpoint 顺序 no-grad，再计算 gated 梯度前向，不同时保存三套图。4090/5060 可行性与工程资格均待未来授权核验。

借鉴已有零初始化附加路径、受控 adapter、任务相关质量比较和内容选择。与 **MoSA** 的有限差异是：不用双模态特征可靠性融合/空间 gate，监督固定动作的 paired task loss difference；与 **DCF** 的差异是：不比较 RGB-only/Depth-only SOD IoU，也不混合 raw/estimated Depth，而是比较同一个 frozen segmenter 的同一输入上残差 off/full。冻结、低参数和换 backbone 均不是创新。窄检索结果见 §6.1；当前仍为中高创新风险。

主要失败风险：proposal 无可利用收益；单点太晚补不回信息；局部正负收益抵消；gate 记住训练语义/生成器模式；新 realization 仍不足以避免训练内过拟合；CE 与 mIoU 不一致；连续幅度非单调；clean consistency 不保证分数。**不得靠增加 spatial gate、teacher 或 prototype 来救第一阶段失败。**

保留淘汰的 A-quality 近邻：以 zero ratio/feature variance/softmax confidence 当空间 reliability 再 gate adapter，与 MoSA 高度相邻且不说明动作收益；作用于几何还越过 Oracle-A NO-GO。本轮不实现、不复活。

### 5.2 候选 B：任务驱动的输入深度校准 / 连续补全（主创新淘汰）

**核心问题**：非空但受损 Depth 在进入几何编码前已经污染几何，事后 feature adapter 可能太晚。

**输入/输出**：RGB、observed raw Depth、可观测有效区域；输出单通道校准 Depth，再完全复用原预处理和 DFormerv2。可用一个低分辨率轻量预测器输出 delta 与连续校准强度，在有效区域构造 `D'=D+q×bounded_delta`，padding 不被补成有效区域；原始零值比例不直接等于 q。

**机制/插入点**：Depth normalized 之前；一个输入校准动作，不再追加 feature adapter、prototype 或 teacher。它不是衰减 GeoPrior 权重，但因改变 Depth 值会改变几何，须重新验收正常输入保护。

**训练目标**：冻结 segmentation base，以分割 CE 驱动 calibration，clean 上惩罚校准幅度与预测变化；自然无效区域不存在可靠 Depth GT，不能直接以“原输入所有像素”充当真实深度监督。若监督使用合成破坏前 Depth，只能在其自然有效区域定义重建项，不能据此声称获得真实 dense Depth。

**推理**：所有输入可连续校准，不等全空。低质量输入也可能被错误补出几何；“q 高代表可靠”不能作为未经标定的解释。

**成本**：建议校准器预算约 0.1–0.5M 参数，仅为可设计范围，不是计数；低分辨率输出单通道可避开 R-OE v1 的高分辨率多通道瓶颈。但训练梯度必须从输出穿过 frozen backbone 回到输入校准器，显存优势明显弱于 A；4090 batch10 无保证，5060 全训练不作承诺。

**解决旧问题的可能性**：修正发生在几何利用前；非空退化也参与；任务目标优于纯重建。但会修改可靠 Depth，也可能重复 R-OE 的“制造几何却不增加语义收益”。

**防撞车**：核心与 GeomPrompt 的任务驱动 recovery / bounded residual 和 DCF 的 quality-conditioned depth calibration 已高度一致。若只换轻量预测器、增加 q 或迁移到 DFormerv2/MUSeg，属于搬运风险；没有从当前证据推出足够独立的新增训练目标。**本轮淘汰为主创新候选**，保留为未来有授权时的参考 baseline，不用高指标为其背书。

**最大风险**：恢复在视觉上正确却任务上有害；深度单位/归一化不一致；错位不能靠局部补值解决；clean 保护不足；训练显存与撞车同时成为阻塞。

### 5.3 候选 C：完整输入到受损输入的轻量自教（性能备选）

**核心问题**：当前模型未充分获得“Depth 不可靠时仍可保留的语义”，需要训练知识迁移，不一定需要检测质量或生成 Depth。

**输入/输出**：训练时同一 train-dev RGB 的原始 Depth 和原有训练故障生成器产生的 observed Depth，两条前向；推理只有 observed 输入。原始 Depth 仍有自然无效，称“未额外破坏”，不称 perfect clean GT。

**机制/插入点**：复用一个 stage index2 的轻量 residual 接口。冻结 C0 给未额外破坏输入提供 detached teacher logits，只训练学生残差，使受损输入的任务输出接近 teacher；不用独立模态 prototypes、copy encoder、重建头或 feedback teacher。训练期知识迁移是唯一核心，推理残差没有质量 gate。

**训练目标**：segmentation CE + 温度 KL；clean 训练样本的 student/base 一致性。teacher 可能在自然无效或稀有类上错误，CE 必须保留；不能对所有 teacher 高 confidence 预测都当作真值。继续使用原有 Depth failure 采样作为拟议基础，不新增 RGB missing，也不擅自改三状态概率。

**推理行为**：单次 backbone/残差/decoder，没有 teacher、质量标签或原始未破坏 Depth。它仍是无条件残差，clean 保护靠训练目标与验收，不像 A 能主动拒绝干预。

**成本**：若沿用 A 的 proposal 结构，约 16,992 trainable 参数（设计算术）；训练额外一条 no-grad teacher 前向，可顺序执行以免双计算图同时驻留。4090 预计比复制 encoder 更合理，5060 训练仍需独立显存核验；推理类似轻量 F-lite 单点版本。

**解决旧问题的可能性**：对局部/非空退化也训练语义行为；避免 substitute 伪几何；teacher 与主任务目标限制无条件残差的漂移。但 gate 缺失的问题仍在，不能声称彻底解决 F-lite。

**防撞车**：借鉴 RobustSeg 的完整→缺失 teacher-student、MaskMentor 的自教动机、Condition Dropout 的缺失输入暴露。上述不是新增。它没有 RobustSeg 的双向 prototype，也没有 MaskMentor 的 multimodal MIM；只是一个成本受控的训练迁移基线，**不能包装成完整复现或强算法创新**。成熟性能备选，不选作当前主论文贡献。

**最大风险**：teacher 错误迁移、对自合成 failure 过拟合、CE/KL 冲突、完整视图知识无法由受损输入恢复、clean 残差仍干扰。结果弱不能裁决原论文路线无效。

## 6. 为什么推荐 A，而不是预设 soft fusion

1. **实验吻合度**：直接针对缺少独立监督的 intervention selector、触发覆盖窄和贡献不清三点；F-lite 残差已经输入相关，A-v1 新增的是是否执行该动作的监督。无需假定 Depth 重建是瓶颈，也不复活已否决的 geometry suppression。
2. **接口匹配**：利用已有语义 feature，保留几何主干和 HAM；通用双流 reliability fusion 在当前架构没有现成输入。
3. **创新可核验但有限**：差异是固定动作的 paired on/off task utility、margin 正负分类与 ambiguous 排除。结构创新弱，必须承认 DCF 的任务相关先例、MoSA 的 gate+adapter 及动态路由先例；当前窄检索未发现高度相同机制，但这不是首创或发表确认。
4. **复杂度/硬件**：单点低分辨率残差和小 gate、冻结 backbone，可限制额外计算图；更适合先在 4090 做核心证伪。5060 推理可行性有既有 base 经验，完整训练没有证据。
5. **消融清楚**：同一 checkpoint 的 off、full、learned 不用训练三个新模型；能拆开 proposal 有无能力与 gate 有无判断能力。
6. **论文故事潜力**：如果能证明“坏模态不等于值得补偿、动作效用比质量 proxy 更能指导干预”，故事有清晰对象；如果只得到一点总分增加而没有可辨识的 gate 效用，应降低为工程改进。首阶段并不完成全领域创新审核或发表判断。

B 的碰撞和显存风险明显，C 的成熟路线适合性能参照但新增弱；两者本轮原样保留，不继续展开或实现。A-quality 的 reliability soft gate 虽直观，却没有先验保证监督对象正确。**因此 A-v1 的依据是动作贡献可辨识、可否证、接口合适；它研究 action utility，不把连续 gate 解释为 reliability。**

### 6.1 一次窄范围防撞车检查（2026-09-30）

只检查“同一 frozen model / 同一输入 / 固定 intervention 的 paired on/off task loss difference，用来训练 residual action selector”，不重新通读九篇、不下载或登记新 paper bundle。实际检索为六组定向查询：`adapter gating counterfactual residual loss routing`、`selective residual adapter routing utility`、`dynamic adapter gating loss oracle routing`、`SkipNet Learning Dynamic Routing ... supervised routing loss`、`Learning to Route ... counterfactual routing task loss adapter`、`dynamic adapter gating oracle supervision loss improvement residual selective adaptation arxiv`。前三组泛化关键词返回噪声较多；一次含题名的查询没有命中同名文献，未把查询中的 venue 当已核事实。以下只记录直接读取原文/官方摘要的三项，搜索摘要与低相关结果不作为机制证据。

- **SkipNet: Learning Dynamic Routing in Convolutional Networks — ECCV 2018**：[官方论文页](https://openaccess.thecvf.com/content_ECCV_2018/html/Xin_Wang_SkipNet_Learning_Dynamic_ECCV_2018_paper.html)、[作者 arXiv 摘要](https://arxiv.org/abs/1711.09485)。已直接核验：依据前层 activations 按输入选择跳过 residual/convolutional blocks，采用 supervised + reinforcement learning 的混合学习，研究计算节约。重合在 input-dependent residual execution；A-v1 则固定一个附加补偿动作，以完整 task-loss endpoints 的 margin 标签训练 selector。**本轮只核验官方摘要**；PDF 获取未成功，不能据此断言其全文从未出现 paired utility 相关子步骤，也不把它列为已排除全部监督重合。
- **Learning to Route: Per-Sample Adaptive Routing for Multimodal Multitask Prediction — 2025 arXiv，未核验正式 venue**：[原文 v1](https://arxiv.org/html/2509.12227v1)。直接核验 §3.3、式(11)–(15)：在四种 modality paths × STL/MTL 共八个 expert paths 间进行样本级概率路由；soft 路由优化加权 task loss，hard 版本使用 Gumbel-Softmax，router 与 experts 联合训练。重合在任务驱动的样本选择；原文所定义目标并非同一 frozen base 的单个固定 residual on/off margin classification，也没有 A-v1 的 Proposal 冻结后再监督选择的定义。它证明“按任务收益学习路由”这一广义思想已有先例，不证明 A-v1 组合新颖。
- **AdaFuse: Accelerating Dynamic Adapter Inference via Token-Level Pre-Gating and Fused Kernel Optimization — 2026 arXiv，未核验正式 venue**：[原文 v1](https://arxiv.org/html/2603.11873v1)。直接核验 Model Structure、式(1)–(3)：token-level Top-2 router 在首层确定跨层 LoRA experts 权重，并通过参数融合/内核优化加速动态适配。重合在 frozen pretrained backbone 上的内容条件 adapter selection；所读方法定义是多 expert、token-level pre-gating，不是单个图像残差动作的 paired off/full margin 标签。训练 classification target 的全部实现细节本轮未审计，不作不存在任何近似监督的全局断言。

#### 6.1.1 最小实现前的两篇近邻补丁（2026-09-30）

- **SpotTune: Transfer Learning through Adaptive Fine-tuning — CVPR 2019**：[原论文 HTML v1](https://arxiv.org/html/1811.08737v1)、[作者论文 PDF](https://www.cs.utexas.edu/~grauman/papers/CVPR19_spottune.pdf)。直接核验 §3.1 式(2)：per-instance policy 对每个 residual block 选择 frozen pretrained 路径或可训练 fine-tuned 路径；§3.2 式(4) 用 Gumbel-Softmax/straight-through 和分类任务损失联合优化 policy 与适配分支。§4.1/Table 2 同时含 Standard Fine-tuning（全参数微调）与 **SpotTune (running fine-tuned blocks)**（使用 SpotTune 权重、推理只运行 fine-tuned blocks）对 learned policy 的对照。因此输入条件策略、选择适配路径及 always-adapt 对照均已有先例，不能作为 A-v1 创新。A-v1 的有限区别是固定 Proposal 后，以同一 frozen segmenter、同输入/随机状态的单个 residual off/full 每图 CE 差生成显式 margin target，而非联合训练多 block 离散路径策略。
- **Keep It Frozen: Domain-Routed Conditional Residual Modulation for Multi-Domain Vision Transformers（DCRM-ViT）— CVPR 2026，pp. 21016–21025**：[CVF 正式论文页](https://openaccess.thecvf.com/content/CVPR2026/html/Khan_Keep_It_Frozen_Domain-Routed_Conditional_Residual_Modulation_for_Multi-Domain_Vision_CVPR_2026_paper.html)、[官方 PDF](https://openaccess.thecvf.com/content/CVPR2026/papers/Khan_Keep_It_Frozen_Domain-Routed_Conditional_Residual_Modulation_for_Multi-Domain_Vision_CVPR_2026_paper.pdf)。直接核验正式页与摘要：frozen ViT backbone、input-feature-conditioned Domain Router 预测 soft domain weights、Parameter Synthesizer 产生 per-sample low-rank residual modulation，注入选定 projection，并可加 domain-aware attention bias；摘要描述 task-supervised inner loop 与更新 router/synthesizer/RMB initialization 的 outer loop。与 A-v1 的 frozen base + input-conditioned residual/gate 结构高度邻近，conditional residual 本身没有创新身份。A-v1 不生成多位置 residual 参数，不做域路由，采用固定动作的 paired off/full utility selector；**本轮 DCRM PDF 正文未能读取，未直接核验其全部公式/损失，因此不能宣布已排除全文中存在近似 utility 子步骤**。

**统一创新边界：** dynamic routing、sample-level gating、conditional residual、frozen backbone + adapter 均为已有思想。A-v1 当前唯一值得继续验证的潜在新增点是：**对同一 frozen segmenter、同一输入、同一随机状态下，一个固定 compensation action 的 off/full task-loss difference 进行显式监督，用来训练该 residual action 的 selector。** 当前定向检索尚未发现与该 paired off/full action-utility supervision 高度相同的机制；未发现会直接阻塞当前最小实现的已核实高度相同方法。窄检索覆盖和 DCRM/SkipNet 全文核验仍有限，MoSA/DCF/动态路由重合明显，创新风险保持中高；不作首创、全领域排除或论文创新性确认。**两篇补丁完成即停止文献搜索。** 工程资格及后续授权见独立 [A-v1 Gate-B 报告](../../MMFR/02_evidence/report_mmfr_a_v1_implementation_gateb_20260930.md)，文献补丁不授予训练权限。

## 7. A-v1 下一步边界：最小实现与 Gate-B，随后才可能训练/筛选

本节为 **design-ready 设计，不是已冻结 protocol 或执行授权**。Gate-B 指进入实验前的工程核验，不评价方法是否有效；本轮没有实施。等待上级复核本次修订，批准后另行授权最小实现与 Gate-B，通过后再单独授权 Proposal → Gate 训练与一次四条件 Quick-Val。禁止把新两阶段目标塞入旧 E1 合同冒充同一实验。

### 7.1 最小实现与工程核验的范围

只接入 §5.1 的 stage-2 proposal、sample gate、margin/mask target、两阶段冻结边界和 explicit off/full/learned 行为。训练采样复用现有框架，但 Gate 阶段使用新的 corruption realization；不新造 failure 类型或数据划分。评价兼容接入须先审核，**原 evaluator 的算法与合同保持不变**，本轮不实现新入口。

Gate-B 只核查：strict off 与绑定 C0 数值匹配；zero-init 行为；base/proposal 权重、BN 与运行模式冻结；两阶段 optimizer membership/梯度边界正确；off/full/gated 同输入和 HAM/RNG 回放；positive/negative/ambiguous 分支及无非模糊样本时 loss finite；gate 输入不含标签或 corruption metadata。工程实现后的静态/最小定点检查范围再审批，GPU 或训练检查仍需显式授权，不能把本文清单当成已运行或 PASS。

### 7.2 一个 checkpoint、两阶段训练、三个同权重行为

C0 不重训，新候选 source 绑定 Batch 1A C0 fixed-final：

`ca618b23d18eabb201a0d11d18da383ac99576d0feae5864e3233bda527d9a1a`。

建议总预算不超过 2560 次成功更新（Proposal 1920、Gate 640），仍是待上级确认的资源安排，不声称足够收敛。$m>0$、必要 clean consistency 权重、冻结行为与新运行身份在未来运行前固定；阶段长度/checkpoint 不用 `val-dev` 选择。Proposal 没有可利用收益时提前停止，不进入 Gate；Gate 阶段 proposal 不再更新。

最终只产出**一个新 A-v1 checkpoint + 一次四条件 Quick-Val**，不训练三个模型。该 checkpoint 的三种行为为：

- **off**：strict bypass residual，必须数值匹配 frozen C0。无法匹配立即 BLOCKED，先修工程问题，不解释效果。
- **full**：$g=1$，始终启用同一个输入相关 proposal；回答 residual action 本身是否有用。
- **learned**：使用训练好的连续 gate；回答 selector 是否比始终补偿更会使用这个 action。

三个行为复用相同的 observed 输入与对应 HAM/stochastic state。C0 历史分数只在环境、输入、RNG、身份完全匹配时可复用；不匹配应先报告阻塞，不擅自重跑 C0 或扩大评价预算。

### 7.3 唯一四条件 Quick-Val 与简单 gate 统计

复用既有 Quick-Val 基础设施与冻结口径：**318 `val-dev`、original-full、scale 1.0、no flip、FP32、TF32 off、原始 Label grid**，既有 corruption identity 与 reset-per-unit 保持不变。条件仅：

1. clean；
2. `entire_missing@1.0`；
3. `spatial_dropout@0.75`；
4. `misalignment@0.75`。

同次筛选报告 off/full/learned 的每条件 mIoU、learned 相对 matched off 的三 hard 均值与 clean 差，以及 learned 相对 full 的 hard 均值差。learned gate 每条件只记 mean/median；必要时附少量分位数，无复杂图表、分组矩阵或大规模统计。监督对应性复用训练内正/负/ambiguous 计数和 gate 简单摘要，不另建实验臂，也不声称经过概率校准。

这仍须是模态失效研究：检查不同 Depth 状态是否伴随有意义的 action-selection behavior，并结合收益方向解释，不把 $g$ 排序当 reliability 证据。首轮不新增 gaussian、blur、quantization、mixed、Main-Val、multiple seed 或 official test，不再加入 hindsight 路由实验或质量 proxy 训练对照。

### 7.4 第一阶段判定：工程成立、动作有效、选择有效分别判断

拟议继续线为：learned 三 hard 平均相对 matched off **$\ge+0.50$ pp**，clean **$\ge-0.20$ pp**，且 learned hard 平均优于 full。它们只是 **Quick-Val 资源分配/研究筛选线**，不是统计显著性、最终论文标准或 Main-Val 保证；运行前仍须上级确认。

- **BLOCKED**：off 无法复现 C0；frozen base 实际变化；RNG 无法匹配；loss 非 finite；optimizer membership 错误。先修工程，不作性能结论。
- **STOP**：proposal/full 基本没有可利用收益，learned 也未改善，停止当前单点 residual action。不增加空间 gate、teacher 或 prototype 挽救，不外推所有残差路线无效。
- **Gate 不成立**：full 有明显收益，但 learned 不优于 full，或 gate 与 utility target 无明显对应。只能说“residual action 可能有价值，但 action-utility selector 暂未得到支持”，不能把 full 收益记给 gate。
- **不满足 clean 保护**：learned 达到 hard 线但 clean 越过容忍线，不建议进入下一轮。
- **inconclusive**：只有微小差值或方向混合，不自动延长训练、扩矩阵或加模块。
- **值得继续**：满足所有筛选线并通过工程条件，只说明 action utility control 值得后续研究；若缺少模态状态关联，仍须降低“模态失效控制”的解释强度。是否后续做其他故障或多视图，由上级另行决定。

## 8. 证据身份、未决项与交付核查

### 8.1 既有 checkpoint 身份

- C0：`ca618b23d18eabb201a0d11d18da383ac99576d0feae5864e3233bda527d9a1a`。
- F-lite：`ea9319e5abe55b996470ee0a75bd63b887241ef834a3b50f145b5b7d4aabd98d`。
- R-OE-lite v2：`369d7e254acadb3bec10d4b6f362d1601291b07edc604bdb671a47591d2a5fbc`（本轮来自正式报告，不重新读 checkpoint 字节）。

F-lite 本地结果根为 `cloud/MMFR_E1_Batch1A_local_transfer_20260922/MMFR_E1_Batch1A_local_transfer_20260922/{C0,FLite}/`；论文 canonical 路径和旧编号以统一索引为准，未重编号或修改 `human` 字段。没有把论文性能当成本地结果。

### 8.2 尚待上级确认的参数与执行边界

核心机制已按“有条件通过”收敛，不再把 sample-level/单点/两阶段作为重新开放的方向选择。待上级复核本次 A-v1 修订并确认：

1. 固定 $m>0$ 的具体值/仅限 train-dev 的确定方式，以及必要 clean consistency 的 $\lambda_{clean}$；BCE 与 segmentation CE 首版系数为 1，无额外收益映射参数。
2. 总预算不超过 2560 更新及建议 1920/640 分配、训练内 proposal 可利用收益的停止口径、拟议 Quick-Val 继续线；不能根据 val-dev 回调参数。
3. 是否单独授权下一步最小实现与 Gate-B；训练、四条件 Quick-Val、GPU 工程检查需要后续明确授权。新运行身份/接入方案由未来实现阶段登记，不沿用旧实验合同。

MoSA 本地正文监督定义矛盾、SkipNet 全文监督细节及更广范围新颖性仍是论文论证局限，**不以本轮窄检索冒充解决**；本轮到此停止，不自动展开。F-lite Main-Val 独立处置、R-OE 路线及入口资格仍由现有开放决策处理，本报告不作新裁决。

### 8.3 九项定向验收（文档检查，不是实验验收）

1. **核心机制收敛**：§5.1 固定 frozen C0、stage-2 单点 residual、sample gate 与 Proposal → Gate，状态为 A-v1 design-ready。
2. **F-lite 表述准确**：§1/§2.1 明确 `R(F)` 已依赖输入，缺少的是独立监督的 intervention selector。
3. **额外残差幅度控制已删除**：A-v1 直接 $F'=F+gR(F)$，保留 bottleneck/zero-init/clean consistency，无该幅度机制、公式或超参数。
4. **utility target 简化**：§5.1.3 固定正 margin，正/负分类、模糊区间排除 BCE，推理输出连续 g；loss 只含 BCE、gated CE 与必要 clean consistency。
5. **Gate 使用新 corruption realization**：阶段随机流隔离，不复用 Proposal 固定实例；同一 target 的 off/full 共享新输入，确定性退化可相同的边界明确。
6. **sample-level 上限明确**：每图一个 g，无法分别控制图内可靠/退化 Depth；不宣称 spatial reliability 或可靠性概率；四条件简单 gate 统计保留模态失效解释约束。
7. **off/full/learned 对照完整**：§7 同 checkpoint 三行为、strict off 必须匹配 C0、HAM/RNG 同态，以及 BLOCKED/STOP/Gate 不成立/继续逻辑。
8. **窄范围防撞车完成**：§6.1 六组查询、三项可核原文/官方摘要，当前未发现高度相同机制，检索和原文获取局限如实保留。
9. **只修改文档**：本轮未写模型代码/正式 config、未改 evaluator，未训练、未 Quick-Val/Main-Val、未访问 official test、未提交或推送。已有工作区代码/配置改动保持原状，不归于本轮。

### 8.4 版本、实际检查与给上级的摘要

修订前完整正式报告已保存在 [pre-A-v1 原版快照](../../MMFR/90_archive/2026-09-30_mmfr_a_v1_design_revision/2026-09-30-mmfr-next-generation-research-design.pre-a-v1.md)。编辑前用 `git hash-object --no-filters` 直接核验两个文件的 Git blob 身份均为 `7ce51c3d93b08a0943d2e9eed6271c8803f41e7d`，确认字节一致后才修订；归档正文保持原始字节与原报告链接语境，不为匹配新结论改写。初版 0.0.15 展示保持历史版本，本次不新建/修改展示，**A-v1 以本报告为准，初版展示不作为修订后的当前设计**。

本轮实际检查为：原版快照字节身份核验、六组窄查询与关键原文/官方摘要直接阅读、报告九项内容复核、报告相对原版 diff、实时入口/必要索引定点内容及差异检查。沿用初版历史实验和九篇论文证据，没有声称本轮重跑评价或重新逐篇审计。

本轮只有文档，按验证预算未运行项目测试、GPU/显存检查、训练或 Quick-Val/Main-Val；没有访问 official test、操作云资源、改冻结 evaluator/protocol、重建旧审核包或提交/push。Gate-B 尚未实施，参数量仍是设计算术，性能与硬件可行性均待验证。

**给上级的简短摘要**：已修正 F-lite 为“输入相关 residual、无独立监督 selector”；首版去掉额外幅度控制和连续收益映射，改固定 margin 正/负 BCE、模糊样本不参与分类；Gate 重新采样 Depth corruption。A-v1 的最终机制是 frozen C0 + stage-2 zero-init proposal + 每图连续 g，先训练动作再固定动作学习选择，同权重 off/full/learned 拆开动作与 selector 的贡献。窄检索未发现完全相同机制，但 MoSA/DCF/动态路由重合与检索局限仍在，不确认论文创新。建议上级复核后授权**最小实现 + Gate-B 工程核验**；尚待确认 m、clean consistency 权重、阶段预算/停止口径及执行授权。**达到文档验收后停止，等待上级审核，不进入实现。**

## 9. 后续正式训练前本地准备：合同冻结，等待4090短预检

本节是后续用户明确冻结的第一轮合同；§7–8仍作为设计轮历史，不再把其中待定预算/参数或train-dev utility早停建议当作当前边界。固定结构不变；[独立A-v1 protocol](../../MMFR/01_research/mmfr_a_v1_action_utility_protocol.md)为完整训练/筛选/交接合同。

- Proposal **1920**、Gate **640**、共**2560 successful updates**；margin **0.01**（per-image mean CE difference）、lambda_clean **0.1**（两阶段相同）、AdamW新branch LR **3e-5**、WD **0.01**。它们是预冻结筛选起点，不是val-dev调参所得最佳值，不自动追加训练。
- batch **10**、480×640、workers **8**、accumulation **1**、AMP fp16 **on**、TF32 matmul/cuDNN **on**；每阶段warmup128/poly0.9，phase corruption seeds **2026093001/2026093002**，p_clean0.25；不新增seed或consistency权重。
- Proposal只因工程/数值/身份或无法解释的明显loss异常停止，不因少数utility不理想早停；正常1920后进入Gate，Gate固定640、只因工程/数值错误停止。Proposal usefulness交给后续同权重full Quick-Val。
- transition=`proposal-update-1920.pth`，fixed-final=`update-2560.pth`，recovery每640次成功更新；不使用val selector。未来唯一四条件Quick-Val的继续线：learned hard mean相对matched off >=+0.50pp、clean >=−0.20pp、hard mean严格>full。只作研究筛选，不宣称统计显著性。
- 云端下一步只能是RTX4090真实全尺寸/正式设置的Proposal最多3-update，正常后Gate最多3-update，各阶段到3立即停止并回报；不得自动接正式1920+640。当前尚无full-resolution runner，本轮只准备调用合同，不连接云端、不训练、不运行任何Val。正式训练仍未授权。
- config import/字段断言、C0/证据/四个未改运行代码hash与定点差异已核验；正式config单独记录新hash，保留原Gate-B证据。精确Git恢复点及执行权限只以实时入口和protocol Cloud handoff为准。
