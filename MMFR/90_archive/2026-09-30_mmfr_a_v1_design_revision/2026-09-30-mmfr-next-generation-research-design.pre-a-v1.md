# 下一代 MMFR：从无条件补偿与空输入恢复，转向补偿动作的任务效用判断

- 日期：2026-09-30；对象：上级模型研究审核。
- 状态：文献与既有证据复核已完成；三个候选及最小验证为**仅提出、尚未实现/验证**。
- 范围：MUSeg、DFormerv2-S + HAM、F-lite / R-OE-lite v2、九篇本地全文。基准 Git HEAD：`a7c5cc17c5b495f2210efa3604b05794b5fa16c4`，开始时工作区已有未提交改动。
- 授权：本轮只做研究分析、设计与文档交付。没有修改模型代码、配置、冻结协议、evaluator，没有训练或评价，没有访问 official test，没有提交或推送。
- 事实与执行边界仍以 [当前状态](../main/MUSeg-current-status.md) 和 [开放决策](../main/MUSeg-open-decisions.md) 为准。本报告不替代历史裁决，也不授权实验。

## 1. 结论先行：现有实验真正提出的问题

**推荐优先审核候选 A：任务效用控制的单点残差修正。** 它预测的不是“Depth 有多可靠”，而是“对这张输入，使用这一固定补偿动作是否比不使用更有利于分割”。第一次验证采用样本级连续控制，补偿本身仍是空间变化的特征残差；暂不加入空间可靠性图、第二模态编码器、重建器、teacher 或 prototype。

大白话：F-lite 的问题是修正一直在工作，却没有单独学习什么时候修正反而有害；R-OE-lite 的问题是只在整幅深度都空时才工作，而且现有对照分不清恢复器与主网络变化各自做了多少贡献。下一步最值得问的是：**补偿本身有没有可用收益，以及模型能否识别这些收益出现的输入。** 先回答这个问题，比继续扩大恢复器、或直接把“低质量 Depth”换成融合权重，更容易得到清楚的否证结果。

这是研究取舍，不是实验结论。既有证据尚不能证明可靠性控制必胜、F-lite 因果性退化、substitute 净收益为零、或所推荐机制具有足够发表的新颖性。最接近的 MoSA 已包含空间调制 adapter 与可靠性融合，DCF 已使用任务效果比较构造深度质量监督；A 的有限差异必须落在**同权重、同输入下对具体补偿动作的增益监督**，不能包装成首次提出 task-aware reliability。

### 1.1 关键概念与边界

- **MMFR**：本项目的多形式模态失效与可靠性研究，关注缺失、局部丢失、噪声、模糊、量化与错位等 Depth 故障。
- **C0**：E1 Batch 1A 不含 F-lite 的训练对照。它不是新方案冻结前的任意 baseline；本报告推荐直接绑定已核验的 C0 fixed-final 权重。
- **Quick-Val / Main-Val**：分别是四条件单视图筛选与十条件十视图正式开发集评价。不同口径的绝对分数不能互比；筛选正收益也不等于正式口径已确认收益。
- **物理有效性 / 模态质量 / 预测置信度 / 动作效用**：非零深度是否存在、信息是否可信、模型有多自信、某个干预是否改善任务，是四个不同对象。非零不保证正确，低置信不保证补偿有用，低质量也不保证应该关闭几何。
- **同权重 on/off**：固定全部主模型参数与随机状态，只改变补偿是否作用；它估计该动作对固定模型的贡献。不同训练 run 的比较不能代替它。
- **保真**：零初始化或严格 bypass 可保留原路径；训练后的非零残差，即使 base 冻结，也不自动保证 clean 不变。

## 2. F-lite 与 R-OE-lite 给出的共同教训

### 2.1 F-lite：筛选有效，但“在所有输入上修正”缺少约束

证据：[Batch 1A Main-Val 正式分析](2026-09-22-mmfr-e1-batch1a-c0-flite-mainval.md)，以及本地两侧 `quickval-original-full/summary.json` 与 `mainval-10cond-v1/update-2560/summary.json`。两侧均完成 2560 次成功训练更新，评价样本为全部 318 条 `val-dev`。

四条件单视图 Quick-Val：clean `53.46 → 54.15`（`+0.69 pp`）；entire_missing `48.76 → 50.17`（`+1.41`）；spatial_dropout `51.40 → 52.39`（`+0.99`）；misalignment `52.15 → 52.63`（`+0.48`）；三个 hard 条件平均差约 `+0.96 pp`，历史 screening 为 promote。

十条件十视图 Main-Val：主指标 M6（六个单故障条件 mIoU 的未加权平均）`55.2883 → 54.8917`，差 `−0.3966 pp`；clean `56.69 → 56.02`，差 `−0.67`；仅 spatial_dropout `+0.40`、spatial_dropout+noise `+0.42`，其余八条件为负。四个共同条件有三个收益符号翻转。这证明**收益未跨评价口径复现**，并不证明翻转由某个单一机制造成。

机制解释与证据强度分开：

1. **已核验结构事实**：F-lite 在编码器输出的三个 stage 使用 `1×1 down → GELU → zero-init 1×1 up` 残差，共 173,152 个参数，没有显式质量/效用控制。零初始化只保证开始时不干预，不约束训练后每张输入的干预。
2. **合理假设：无条件补偿会产生正负收益混合。** 在几何已有帮助的样本上也改变特征，可能造成不必要干预。但没有同权重 adapter-off 对照、残差与局部深度质量的关联或视图级分析，**尚未证明对可靠 Depth 过度修改**。
3. **合理假设：缺少条件控制导致跨视图不稳定。** 特征尺度、插值、翻转、多视图 logit 融合会改变残差的效果；三项翻转支持研究这个问题，却不能单独归因于 adapter 的尺度敏感性。
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

- **干预的覆盖范围和干预的有益性是两个问题。** F-lite 广覆盖但不选择；R-OE-lite 强选择但只覆盖全空；下一步须分别检验“可补偿”与“可判断”。
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
- **feature calibration**：与现有 post-backbone 接口最相容，低额外参数；要以动作效用约束，而非 unconditional adaptation 的改名。
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

### 5.1 候选 A：任务效用控制的单点残差修正（主推荐）

**核心问题**：对同一固定模型，有的输入适合补偿、有的输入无需或不宜补偿；需要判断动作是否有益，而不是仅判断 Depth 看起来是否坏。

**输入/输出与插入点**：只在 stage index 2（0-based，256 通道，约输入 1/16 尺度）输出与 HAM 之间插入。输入是这个已有语义 feature F 与四个可观测整图 Depth 统计量；输出同尺寸 F'，其他三个 stage 原样。统计量建议为几何有效区域内的零值比例、非零值均值/标准差、邻接非零位置的平均绝对深度差；空集合采用固定值并明确记录。Depth 统计用于内容条件，不作为可靠性真值，也不读取 cause、severity、corruption index 或推理标签。

**机制**：一个小卷积 bottleneck 给出空间变化的残差 R；一个轻量样本级 gate 给出 g∈[0,1]。网络只有这一项受控残差，既不混合两套模态 encoder，也不改变几何先验。

$$
R_\theta(F)=W_{up}\,\mathrm{GELU}\bigl(\mathrm{DWConv}_{3\times3}(\mathrm{GELU}(W_{down}F))\bigr)
$$

$$
\bar R_\theta(F)=R_\theta(F)\min\left(1,\frac{\kappa\lVert F\rVert_{RMS}}{\lVert R_\theta(F)\rVert_{RMS}+\epsilon}\right),\qquad
F'=F+g_\phi(F,s_D)\bar R_\theta(F)
$$

这里 norm/cap 按样本计算；κ 是固定残差幅度上限，建议起点 0.1，尚未验证；ε 防零除。它是保守约束，不是 clean 保真证明。`W_up` 零初始化，base 的参数、BN running statistics 与模式固定到定义的 frozen-base 行为；不能只设置 requires_grad=False 而让 BN 继续漂移。

**第一次验证为什么不用空间 gate**：HAM 和解码器是非局部映射。两个完整输出的某像素 CE 差，只能表示该像素在整幅干预后的结果变化，不能作为“这个 feature 位置的残差造成了多少收益”的因果标签。样本级 on/off 的整体风险比较更明确。空间残差仍能针对局部内容产生不同修正，但同一个 g 控制整张图的强度；**第一版无法独立保留局部好 Depth、关闭局部坏补偿**，这是真实能力上限，不能写成 spatial reliability 已解决。

**监督对象与两阶段训练**：

1. 先冻结 C0 base，令 g=1，仅训练 proposal R。在原有 train-dev/Depth 故障训练基础上使用已有分割监督，并对训练 clean 输入加入相对 frozen base 的预测一致性约束。无需重建 Depth GT、第二 encoder 或 teacher 参数。
2. 再冻结 proposal，训练 gate。对**同一当前输入 x**计算固定 base 的 off 输出与 full-residual 输出，使用相同 HAM 随机状态，构造动作收益。两阶段用于减少 target 随 proposal 同时变化或 gate 先收缩到零导致 proposal 学不到东西；不是两个相互叠加的研究模块。

$$
\ell_0(x,Y)=\mathrm{CE}(P_0(x),Y),\quad
\ell_1(x,Y)=\mathrm{CE}(P_1(x),Y),\quad
u(x)=\operatorname{stopgrad}(\ell_0-\ell_1)
$$

$$
t(x)=\operatorname{clip}\left(\frac{u(x)-m}{\tau},0,1\right),\qquad
\mathcal L_g=\mathrm{BCE}(g,t)+\lambda_{seg}\mathrm{CE}(P_g,Y)
+\lambda_{clean}\mathbf1_{train\ clean}\,\mathrm{KL}(P_0\Vert P_g)
$$

CE 为当前有效标签上的平均交叉熵。m≥0 为最小有用 margin，τ>0 为收益缩放；须在未来运行前确定，可只依 train-dev 的收益尺度定标后固定，不用 val-dev 调参。第一阶段也保留分割 CE 与 clean KL；所有 KL 使用 fixed base 的 detached 输出。κ/m/τ/λ 现在是待批准设计值或待定训练值，**本轮没有改 loss 或协议**。

目标来自同输入上两个固定动作的任务损失差，而非完整 clean Depth 与损坏 Depth 的隐藏生成原因。即便是 natural-empty，只要观察与合成空输入相同，就不宣称辨别原因。proposal 冻结使两个 endpoint target 固定；gate 分割项处理“endpoint 有益但中间插值未必有益”的问题，但仍不能保证每个输入的收益单调。

**推理行为**：单次 backbone、单次 gate/proposal、单次 HAM；g 可在仍非空 Depth 时调整，不等待整图缺失。没有 labels、teacher、完整 Depth、额外反事实前向。初始化 R=0 可复现 base；显式 off 应跳过残差并复现冻结路径，g=0 的数学表达不替代数值 exact-bypass 验收。

**成本草算（不是实测）**：bottleneck 256→32→256、一个 32 通道 depthwise 3×3，无 norm；含 bias 约 16,992 参数。gate 的 pooled 256 feature+4 Depth statistics 输入 260→16→1，含 bias 约 4,193；总约 **21,185 参数**。这是设计算术，不是已实现模型计数。推理增量在低分辨率 feature 上，明显小于再造一套 encoder；不能由参数量推出 latency。

训练可 no-grad backbone，保留梯度穿过**冻结的 HAM**回到残差；freeze decoder 仍需其激活图。gate 阶段可以顺序算两个 no-grad endpoint，然后算 gated 梯度前向；不同时保留三套完整计算图。4090 是合理首选，但不能沿用旧 near-full batch-10 显存保证；5060 Laptop 更适合推理/小 batch 工程核验，完整训练可行性待实际最小显存检查，不能预先承诺。没有增加全分辨率多通道卷积。

**为何比旧两条路线更合理**：控制“补偿有益性”，比 F-lite 的始终修改多一道任务依据；覆盖非空退化，比 R-OE 的全空 trigger 更宽；冻结 base 与同权重 off 使新增动作贡献可辨识。上述是设计优点，不是性能预测。

**借鉴/新增/最接近差异**：借鉴 F-lite/Condition Dropout 的零初始化附加路径，MoSA 的受控适配，DCF 的任务相关比较，SMAC 的可观测内容选择。均为已有思想。有限新增是对**同一 frozen geometry-conditioned segmenter 的固定残差动作**构造 paired on/off 风险差，用它标定干预；不是模态独立预测 correctness、raw/estimated depth 混合、或 attention 强度。冻结、低参数、换到 MUSeg 都不是创新。

**撞车风险**：中高。MoSA 在结构上非常近，DCF 在监督动机上非常近；还有九篇之外的 selective prediction、residual-policy、counterfactual routing 文献，本轮未扩展审计。仅在这九篇内未见同样监督组合，不能据此宣布全领域首创。若实际 gate 只是检测 dropout、与简单质量 proxy 效果相同，论文创新可能不足。

**最大失败风险**：proposal 没有可选收益；stage 太晚无法补回丢失的信息；样本级正负区域抵消；gate 记住生成器状态而非动作效用；训练内收益 target 过拟合；CE 改善不等价于稀有类 mIoU 改善；soft amplitude 不单调；多视图时同一图 gate 随 view 不稳定；clean KL 仍不能保证 clean 分数。

**淘汰的近邻 A-quality**：把零值/feature variance/softmax confidence 直接预测为空间 reliability，再 gate adapter 或融合。它结构上与 MoSA 高度重合，监督不说明“补偿是否有用”，若作用于几何又越过旧 NO-GO。因此不作为主创新方案。A 推荐的是 action-relative target，不是 reliability-aware soft fusion 的预设胜出。

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

1. **实验吻合度**：直接针对无条件干预、覆盖窄和因果贡献不清三点；无需假定 Depth 重建是瓶颈，也不复活已否决的 geometry suppression。
2. **接口匹配**：利用已有语义 feature，保留几何主干和 HAM；通用双流 reliability fusion 在当前架构没有现成输入。
3. **创新可核验但有限**：差异是 paired action-utility supervision。结构创新弱，必须承认 DCF 的任务相关先例与 MoSA 的 gate+adapter 先例；现在只能称“值得验证的研究假设”。
4. **复杂度/硬件**：单点低分辨率残差和小 gate、冻结 backbone，可限制额外计算图；更适合先在 4090 做核心证伪。5060 推理可行性有既有 base 经验，完整训练没有证据。
5. **消融清楚**：同一 checkpoint 的 off、full、learned 不用训练三个新模型；能拆开 proposal 有无能力与 gate 有无判断能力。
6. **论文故事潜力**：如果能证明“坏模态不等于值得补偿、动作效用比质量 proxy 更能指导干预”，故事有清晰对象；如果只得到一点总分增加而没有可辨识的 gate 效用，应降低为工程改进。首阶段并不完成全领域创新审核或发表判断。

B 的碰撞和显存风险明显，C 的成熟路线适合性能参照但新增弱。A-quality 的 reliability soft gate 虽直观，却没有先验保证监督对象正确。**因此推荐 A 的依据是可辨识、可否证、接口合适，不是认为 continuous reliability 必然优于 hard routing。**

## 7. 最小第一阶段验证草案：只判断这个动作及其控制是否值得继续

本节是研究设计，**不是已冻结 protocol，不赋予运行权限**。先由上级确认核心假设、source 权重、freeze 行为和训练目标，再另行授权最小实现/资格检查；现有训练入口只作复用候选，不能把 frozen-base 两阶段新目标塞进旧 E1 合同冒充同一实验。

### 7.1 唯一研究问题与最小对照

> 固定 C0 后，单点 residual 是否产生可利用的样本差异收益，且由同权重动作收益监督训练的 gate 是否能比始终补偿更好地使用这些收益？

**一个新模型、一条训练、一次四条件 Quick-Val**。C0 不重训；新模型 source 绑定 Batch 1A C0 fixed-final：

`ca618b23d18eabb201a0d11d18da383ac99576d0feae5864e3233bda527d9a1a`。

建议总预算不超过既有 2560 次成功更新：proposal 1920、gate 640，均为待批准安排，不声称足够收敛。使用 train-dev、已有数据处理与 Depth failure 基础；BN 与权重冻结语义、optimizer 参数白名单、随机隔离、margin/scale/λ 以及 fixed-final 身份须在未来实施前登记。不得用 val-dev 选择阶段长度或 checkpoint。

**同一新 checkpoint 的三个行为**：

- off：strict bypass 单点残差，应匹配绑定 C0 的 frozen-base 输出。
- full：g=1，同一 proposal，一直补偿。
- learned：由训练 gate 决定幅度。

off 与 C0 若数值不匹配，先判工程对照 blocked，不解释模型效果。三种行为是同权重诊断，不是额外三条训练；无需再训练 F-lite 或 R-OE。

### 7.2 两个停止点，避免无意义训练

- **proposal 阶段停止点**：仅在 train-dev 的固定训练内诊断上查看 off/full 的损失变化与是否存在有用样本，不接触 val-dev 选模型。若 proposal 没有可用收益、loss 非 finite 或显存无法满足批准预算，停止，不让 gate 弥补不存在的动作能力。
- **gate 阶段停止点**：冻结 proposal 后，检查 target 并非全零/饱和、base 与 BN 未变、endpoint 两次使用同随机状态；若目标生成或梯度边界错误，停止，不以继续训练掩盖问题。

### 7.3 唯一四条件筛选

复用既有 `318 val-dev`、单视图 original-full、clean / entire_missing@1.0 / spatial_dropout@0.75 / misalignment@0.75、既有 corruption identity 与 reset-per-unit、FP32/TF32-off。只复用基础设施与口径，**不改 evaluator**。

现有入口尚不支持新模块与同权重行为选择；未来必须在不改变 frozen evaluator 的前提下先审核兼容接入或独立诊断入口。本轮没有实现该入口。C0 旧分数只在运行环境、输入、RNG、identity 完全匹配时复用，否则需未来授权 matched C0 重跑，不能强行跨设备/版本复用。

报告 learned 相对 C0/off 的 clean 与三 hard 均值，同时列 full；额外保留每条件配对变化、gate 分布和 off/full/learned 的同输入损失。优先记录新臂的原有显存/时间遥测，不另做完整 benchmark。HAM 三个行为必须每次回放同一个对应输入随机状态，不能把连续调用的随机差当作干预效果。

可在同次诊断内计算**输出级 off/full hindsight 选择**：按标签选择低 CE 的整幅 endpoint，仅估计“两选一动作池”在 CE 指标下的上限，明确它不是推理方法，也不保证 dataset mIoU 上限。它不能发现中间幅度独有的收益，不能据此训练/选择 checkpoint。局部逐像素 endpoint 拼接不用于宣称可实现空间 policy。

### 7.4 提议的 go / stop / inconclusive 条件（运行前仍须批准）

沿用 screening 的保守量级作为**新方案的拟议判断线**，不回写旧门槛：learned 相对 matched off 的三 hard 平均 `≥+0.50 pp`，clean `≥−0.20 pp`，且 learned 的 hard 平均优于 full，才建议“值得继续研究”。这些数值是研究资源分配线，不是统计显著性或 Main-Val 保证。

- off 不匹配 C0、freeze/RNG/身份不一致：**blocked**，没有效果结论。
- full 与 hindsight 诊断未显示可利用的动作收益，learned 也无收益：**停止这个单点动作**；不能外推所有 post-backbone 机制无效。
- full 有用而 learned 不优于 full、或 gate 行为与目标无对应：**效用控制未获支持**；不以 full 的增益给 gate 记功。
- learned 达到 hard 线但 clean 超出容忍线：**不满足保守补偿要求**，不进入下一轮。
- 只有 `+0.01` 等微差，或收益/保护方向混合：**inconclusive**，不自动扩矩阵、延长训练或增加模块。
- 达到所有拟议线：只是 **Quick-Val 核心机制值得继续**；上级另行决定是否验证十视图与更多故障，不能称已解决 F-lite 的 Main-Val 泛化问题。

第一次验证不做多 seed、完整论文矩阵、新 teacher/prototype、九篇外大规模文献遍历、十条件 Main-Val 或 official test。三种行为已足够判断动作与 gate；任务效用对质量 proxy 的优势、空间版本、新颖性扩展和正式泛化属于之后才可能讨论的事项。

## 8. 证据身份、未决项与交付核查

### 8.1 既有 checkpoint 身份

- C0：`ca618b23d18eabb201a0d11d18da383ac99576d0feae5864e3233bda527d9a1a`。
- F-lite：`ea9319e5abe55b996470ee0a75bd63b887241ef834a3b50f145b5b7d4aabd98d`。
- R-OE-lite v2：`369d7e254acadb3bec10d4b6f362d1601291b07edc604bdb671a47591d2a5fbc`（本轮来自正式报告，不重新读 checkpoint 字节）。

F-lite 本地结果根为 `cloud/MMFR_E1_Batch1A_local_transfer_20260922/MMFR_E1_Batch1A_local_transfer_20260922/{C0,FLite}/`；论文 canonical 路径和旧编号以统一索引为准，未重编号或修改 `human` 字段。没有把论文性能当成本地结果。

### 8.2 仍需上级审核

1. 是否接受 A 的研究对象为“补偿动作效用”，以及首阶段样本级 gate 的能力上限。
2. 是否接受绑定 C0 并冻结 base、单点 proposal 与两阶段预算；κ、margin/scale、loss 权重和新运行身份仍需在实施前定案。
3. 是否同意先进行最小实现/资格核验，再独立授权一条训练与四条件筛选；本轮全部未获执行授权。
4. F-lite Main-Val 的独立历史处置、R-OE 路线及其入口资格仍未被本报告裁决。
5. MoSA 的监督定义矛盾、九篇之外 action-utility 近邻新颖性仍待核；不为完成报告自动扩展研究范围。

### 8.3 本轮实际检查与停止边界

实际检查为：必读实时入口与正式报告复核、F-lite 本地 summary 复核、九篇 canonical 正文的定点机制阅读及相关结构图、模型接口只读核验、交付文档/索引/展示的内容与差异检查。文献机制的条件和原文矛盾保留，不以子任务摘要替代关键证据。

没有执行项目测试、GPU 显存验证、新训练、Quick-Val/Main-Val、official test、云资源操作或统计推断。本次是文档与研究设计，依据项目验证预算不运行项目测试；参数与资源影响均为草算/待验证，不能写成已通过。

新报告与展示完成后更新实时入口及 MMFR 导航；历史 blueprint、实验协议、报告、原始论文、旧审核包保持不动。**交付到此停止，等待上级模型审核。**
