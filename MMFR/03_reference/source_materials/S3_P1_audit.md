# MMFR-P1 新增文献与 Idea 防撞车综合审计报告

> **审计范围：** 本报告仅综合本轮 8 篇 P1 文献，不替代已有 P0 审计。
> **审计原则：** 按既定六维框架——“输入与失效、质量表示、监督信号、控制位置、动作、证据”——判断实质性重合，而不是根据模块名称是否相似判断撞车。
> **涉及文献：** MAGIC、ANGA、Calibrated RGB-D SOD、SimMLM、D3Net、RobustSeg、Condition Dropout、UMFNet。

------

# 一、综合结论

这 8 篇 P1 文献进一步确认：MMFR 已经不能把下列宽泛思想作为核心创新点：

- “坏 Depth 会伤害多模态模型”；
- “需要判断 Depth / 模态是否可靠”；
- “根据模态质量动态调整融合权重”；
- “模态缺失时应该降低或取消其作用”；
- “根据 strong / weak / robust / fragile modality 做不同处理”；
- “利用 uncertainty / confidence 抑制不可靠模态”；
- “学习一个连续 reliability/confidence map 再执行局部 feature gating”；
- “通过 modality dropout / missing-modality training 提高 DFormer 的鲁棒性”。

其中，**UMFNet 是本轮 P1 对 MMFR 原始创新边界压缩最大的一篇**：它已经明确实现了 pixel-wise uncertainty、continuous confidence map、spatial modulation 和 unreliable-region suppression，因此过去还相对安全的“**空间连续 reliability → 局部自适应融合**”也已经不能单独作为 novelty。

综合这 8 篇之后，当前仍然值得 MMFR 守住的不是一个单独模块，而是下面这条**组合链条**：

$$
\boxed{ \text{controlled Depth corruption state} \rightarrow \text{explicitly supervised geometric reliability }Q_D(x,y) \rightarrow \text{DFormerv2-specific geometry-path intervention} \rightarrow \text{reliability-to-utility validation} \rightarrow \text{counterfactual evidence} }
$$

更具体地说，MMFR 应避免把贡献描述成“提出了一种 reliability-aware fusion”，而应收窄为：

> **针对受控 Depth degradation 后的最终几何状态，构造具有明确 corruption 含义的像素级连续可靠性，并让该可靠性直接作用于 DFormerv2 的 geometry-prior / geometry interaction 路径；再通过 oracle、predicted、constant、shuffle、inverse 以及 reliability-to-utility 分析证明，性能提升来自“诊断正确 + 几何处置正确”，而不仅仅是 corruption augmentation 或 generic confidence gating。**

这一方向与既有《MMFR-新增文献与Idea撞车审计》中保留的狭化边界一致，但经过 UMFNet 后，必须比原先更强调 **corruption-grounded、geometry-specific 和 causal evidence**。

------

# 二、8 篇 P1 文献的综合定位

| 文献                             | 已占据的关键空间                                             | 对 MMFR 的主要约束                                           | 当前撞车压力 |
| -------------------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------ |
| **MAGIC / AI022**                | robust/fragile modality 排序、arbitrary-modal segmentation   | 不能把“判断模态强弱并改变使用方式”作为创新                   | 中高         |
| **ANGA**                         | sample reliability → curriculum → gradient alignment         | 不能把“可靠性决定训练/优化行为”作为宽泛创新                  | 中低         |
| **Calibrated RGB-D SOD / AI017** | Depth reliability classifier → raw/estimated Depth 连续校正  | “学习 Depth reliability + reliability-conditioned Depth action”早已存在 | 高           |
| **SimMLM / AI026**               | dynamic modality gating + More-vs-Fewer ranking              | 动态模态贡献、模态 availability ranking 已存在               | 中           |
| **D3Net / AI016**                | low-quality Depth 判断 → RGB/RGB-D hard path selection       | “坏 Depth 就降低/取消其贡献”早已存在                         | 高           |
| **RobustSeg / PR090**            | missing/noisy MMSS、dominant/non-dominant modality、差异化蒸馏 | 强弱模态检测及 robustness/full-performance balance 已被覆盖  | 高           |
| **Condition Dropout / AI024**    | DFormer + missing Depth adaptation                           | DFormer missing-modality robustness 已有直接先例             | 高           |
| **UMFNet / PR029**               | pixel-wise uncertainty → confidence → spatial/channel modulation | “空间连续可靠性 → 局部 feature suppression”已高度重合        | **极高**     |

MAGIC 通过 fused semantic representation 对不同模态 feature 进行排序，明确区分 robust 和 fragile modality，并面向 arbitrary modality input 建立鲁棒分割框架。 ANGA 则将 entropy 判断出的可靠 reconstruction sample 纳入 optimization anchor，并从 gradient 层面对 reconstructed sample 的优化方向进行约束。

DCF 更早已经训练 Depth-quality discriminator，输出 reliability probability $P_{pos}$，并依据该值连续插值 raw Depth 与 RGB-estimated Depth：

$$
Depth_{cal} = P_{pos}Depth_{raw} + (1-P_{pos})Depth_{est}.
$$

因此“学习 Depth reliability，再据此改变 Depth 使用方式”已有非常直接的 RGB-D 先例。

D3Net甚至更早采用 DDU 自动过滤低质量 Depth，并在 RGB-only 与 RGB-D prediction path 之间选择，还通过 upper/lower bound 分析讨论理想路径选择器的潜力。

SimMLM 用 DMoME gating 动态调整 modality expert contribution，同时用 MoFe loss 约束 modality-rich input 的任务 loss 不应高于 modality-poor input。

RobustSeg 则进一步在 RGB、Depth、Event、LiDAR 等 MMSS 环境中通过 dominant/non-dominant modality selection 决定跨模态 prototype distillation、IFV transfer 和 teacher feedback，并同时测试 missing 与 noisy modality robustness。

ConD 与 MMFR 的工程关系尤其直接：它明确将 DFormer-B 作为对象，通过 full/RGB-missing/Depth-missing 二阶段 continued training，使 DFormer 的 missing-modality 性能大幅恢复，因此“DFormer + sensor failure robustness”已经不是可独占的研究定位。

最后，UMFNet 将每个像素 feature 建模成 Gaussian latent distribution，以 variance 表达 uncertainty，再形成 $Conf(p)\in[0,1]$ 的局部 confidence map，用其同时控制 spatial/channel fusion，在低 confidence 位置主动压制不可靠 Thermal 信息。它已经直接覆盖了“pixel-wise continuous confidence → local reliability-aware multimodal modulation”这一抽象链。

------

# 三、六维综合撞车审计

## 1. 输入与失效：missing 已高度拥挤，MMFR 应重点占据“坏而未失”的 Depth

这 8 篇文献已经完整覆盖了一条明显的 missing-modality 研究谱系：

$$
\text{full modality} \rightarrow \text{partial modality} \rightarrow \text{arbitrary modality} \rightarrow \text{missing-modality robustness}.
$$

MAGIC、SimMLM、RobustSeg 和 ConD 都直接处理 modality missing；ANGA处理 missing 后 reconstruction；D3Net 和 DCF则更早研究 low-quality Depth。尤其 ConD 已经直接在 DFormer 上模拟 RGB missing / Depth missing。

因此，MMFR 不应再以：

> “现实中 Depth 可能缺失，所以我们研究模态失效”

作为创新出发点。

MMFR真正应该强化的是：

$$
\boxed{ D\text{ 仍然存在} \quad\land\quad D\text{ 局部、连续、异质地失真} }
$$

例如：

- local dropout；
- Gaussian noise；
- blur；
- quantization；
- local invalidity；
- misalignment；
- continuous severity；
- 不同位置不同 corruption strength。

这与“整模态置零”是本质不同的问题。

尤其应强调：

$$
D_c\neq\varnothing
$$

但：

$$
\operatorname{Reliability}(D_c(x,y))
$$

并不均匀。

### 对 MMFR 的直接要求

**entire missing 应作为 robustness condition，而不能成为核心创新场景。**

MMFR 的主体实验应该把重心放在：

> **present-but-unreliable Depth**

而不是：

> **absent Depth**。

------

# 四、质量表示：从“global quality”到“pixel confidence”均已有先例

这 8 篇已经覆盖了多个层次的 quality / reliability representation。

### D3Net：图像级二值判断

$$
F_{cu}\in\{0,1\}
$$

决定整张 Depth 是否应该参与。

### DCF：图像级连续 Depth reliability

$$
P_{pos}\in[0,1]
$$

代表整张 Depth 的可靠程度。

### MAGIC / RobustSeg：模态级 semantic dominance

依据 unimodal feature 与 fused semantic feature 的 similarity 判断：

$$
\text{robust / fragile}
$$

或：

$$
\text{dominant / non-dominant}.
$$



### ANGA：样本级 reconstruction reliability

利用预测 entropy 排序 reconstructed samples，再决定哪些样本进入 optimization anchor。

### SimMLM：instance-dependent modality importance

DMoME gating 输出不同 modality expert 的动态贡献权重。

### UMFNet：像素级连续 uncertainty / confidence

UMFNet进一步建立：

$$
Conf(x,y)\in[0,1]
$$

并且直接解释为不同空间位置的 feature reliability。

因此，经过本轮 P1 后必须正式放弃：

> **“pixel-wise continuous reliability”本身就是 MMFR 创新**

这一说法。

------

## MMFR 还能保住什么？

区别必须从“粒度”升级到“**语义来源**”。

UMFNet 的 $Conf(x,y)$ 本质是：

> **由 task supervision 学出的 latent feature uncertainty/confidence。**

而 MMFR 应尽量保持：

$$
Q_D^{GT}(x,y) = V_D^{final}(x,y) \odot R_D^{syn}(x,y)
$$

即可靠性与实际 Depth corruption 后的最终状态明确对应。已有审计文件也将这一点作为当前 MMFR 最重要的狭化空间。

因此真正需要强调的是：

$$
\boxed{ \text{learned attention/confidence} \neq \text{corruption-grounded geometric reliability} }
$$

------

# 五、监督信号：这是当前 MMFR 仍然比较有价值的一条分界

这 8 篇中的 reliability / weighting 大多数是**隐式 task-driven**：

- MAGIC：semantic similarity；
- RobustSeg：semantic similarity + task distillation；
- SimMLM：task loss；
- ANGA：prediction entropy；
- UMFNet：task loss + latent KL；
- D3Net：heuristic prediction consistency；
- DCF：task-IoU-derived image-level binary pseudo-label。

其中 DCF 已经证明“Depth reliability 可以被显式监督”，所以 MMFR不能说：

> “首次监督 Depth reliability。”

DCF通过 Depth-only prediction 与 saliency GT 的 IoU 排序构造 positive / negative Depth，再训练二分类 discriminator。

但是 DCF 的 supervision 是：

$$
Y_Q\in\{0,1\}
$$

且是：

> image-level + task-derived。

MMFR仍然可以争取：

$$
Q_D^{GT}(x,y)\in[0,1]
$$

这种：

> **spatial + continuous + corruption-state-derived**

监督。

所以以后应写：

> **explicit corruption-grounded spatial reliability target**

而不能只写：

> explicit reliability supervision。

------

# 六、控制位置：这是 MMFR 当前必须牢牢守住的主要结构边界

这 8 篇论文的 control location 已经遍布多个位置：

- D3Net：最终 RGB / RGB-D path selection；
- DCF：raw Depth calibration；
- MAGIC：high-level modality feature selection；
- SimMLM：expert-output/logit weighting；
- ANGA：optimization gradient；
- RobustSeg：distillation / encoder fine-tuning；
- ConD：generic encoder residual feature injection；
- UMFNet：generic cross-modal feature fusion。

但这批论文中，尚未看到与当前 MMFR 完全同构的：

$$
\boxed{ Q_D \rightarrow \text{DFormerv2-specific geometry-prior computation} }
$$

因此，“控制哪里”已经比“有没有 reliability”更加关键。

------

## 风险最高的错误实现

如果 MMFR 最终只是：

$$
F_D' = Q_D\odot F_D
$$

然后：

$$
F_{fused} = F_R+F_D',
$$

那么它实际上很容易被归类为：

> **another confidence-guided feature fusion method**

而 UMFNet 已经有：

$$
\text{pixel-wise confidence} \rightarrow \text{spatial/channel modulation} \rightarrow \text{unreliable-region suppression}.
$$



这种情况下，仅仅把 RGB-T 换成 RGB-D，或者把 SOD 换成 semantic segmentation，都不足以构成强方法差异。

------

## 更安全的 MMFR 控制对象

MMFR应尽量直接作用于：

- depth-derived geometry prior；
- geometry relation；
- geometry attention；
- geometry token interaction；
- 或 DFormerv2 中其它可明确归因于 Depth geometric inductive bias 的路径。

概念上应该更接近：

$$
G_D \rightarrow \widetilde G_D(Q_D)
$$

而不是：

$$
F_D \rightarrow Q_D\odot F_D.
$$

例如在抽象意义上，可以是：

$$
\widetilde G_{ij}^{(\ell)} = \phi \left( Q_i^{(\ell)}, Q_j^{(\ell)} \right) G_{ij}^{(\ell)},
$$

重点在于：

> reliability 改变的是 **geometry relation 是否值得相信**，而不是简单重新给一个模态 feature 加权。

这才是与 UMFNet、MAGIC、DCF 等路线真正产生结构距离的位置。

------

# 七、动作：从 discard 到 soft local suppression 都已经存在

这 8 篇已经几乎覆盖了 reliability-aware action 的主要通用形态：

### D3Net

$$
\text{good Depth}\rightarrow RGBD
$$

$$
\text{bad Depth}\rightarrow RGB
$$

即 hard reject。

### DCF

$$
\text{reliable Depth}\rightarrow Depth_{raw}
$$

$$
\text{unreliable Depth}\rightarrow Depth_{est}
$$

即 continuous replacement / calibration。

### MAGIC / RobustSeg

$$
\text{robust/dominant modality} \rightarrow \text{preferential semantic/detail transfer}
$$

### SimMLM

$$
\text{input condition} \rightarrow \text{dynamic expert weighting}
$$

### ANGA

$$
\text{reliable reconstruction} \rightarrow \text{optimization anchor}
$$

$$
\text{conflicting gradient} \rightarrow \text{attenuate / suppress}
$$

### UMFNet

$$
Conf(x,y)\downarrow \rightarrow \text{local feature injection}\downarrow
$$

已经完成 continuous local suppression。

------

## 综合结论

因此下面这条逻辑：

$$
\text{reliable}\rightarrow\text{retain}
$$

$$
\text{partially unreliable}\rightarrow\text{attenuate}
$$

$$
\text{highly unreliable}\rightarrow\text{suppress}
$$

**本身已经不是新的。**

MMFR需要创新的不是“action rule”，而是：

> **这个 action 精确作用于何种 geometry mechanism，以及 reliability 是否真的对应 geometry utility。**

------

# 八、证据体系：这是 MMFR 最有机会明显超过 P1 先例的地方

8 篇论文已经提供了多种有价值的证据：

- D3Net：RGB/RGB-D oracle-style upper bound；
- SimMLM：More-vs-Fewer ranking 与 Counterintuitive Rate；
- RobustSeg：missing/noisy robustness 与 full-modality performance trade-off；
- ConD：same backbone missing-modality improvement 与 architecture ablation；
- UMFNet：`w/o Conf`、`w/o spatial modulation`、`w/o channel modulation` 等 component ablation；
- ANGA：complete/reconstructed sample 分群分析；
- DCF：calibrated Depth 可迁移至其它模型。

但是这些工作总体仍主要回答：

> **加入这个模块以后，最终任务指标有没有上涨？**

尚不足以回答 MMFR 希望建立的：

> **这个 reliability 是否真的“正确”，以及正确 reliability 是否真的导致正确 geometry action？**

这正是 MMFR 最值得建立的差异。

------

# 九、建议 MMFR 必须建立“诊断”和“处置”两层因果证据

## 第一层：先证明 action 是对的

首先不要使用 predicted reliability，而使用已知 corruption 构造：

$$
Q_D^{oracle}.
$$

比较：

$$
\text{Base}
$$

$$
\text{Base + Oracle Action}
$$

如果：

$$
\text{Oracle Action} \le \text{Base},
$$

说明 action 本身就不合理，此时继续训练 reliability predictor 没有意义。

这也是原审计已经明确要求的 B1 冻结原则。

------

## 第二层：再证明预测 reliability 是对的

至少比较：

$$
Q_D^{oracle}
$$

$$
\hat Q_D
$$

$$
Q_D^{constant}
$$

$$
Q_D^{shuffle}
$$

$$
1-\hat Q_D.
$$

理想关系应接近：

$$
Perf(Q_D^{oracle}) > Perf(\hat Q_D) > Perf(Q_D^{constant})
$$

并且：

$$
Perf(Q_D^{shuffle}),\; Perf(1-\hat Q_D)
$$

明显退化。

这样才能说明：

> 性能不是因为“加了一个 mask / attention module”，而是因为 mask 的 reliability 结构具有正确含义。

------

# 十、必须建立 reliability-to-utility，而不仅是 reliability-to-corruption

这一点在 SimMLM、DCF 和 UMFNet 出现以后尤其重要。

可以定义 Depth 的任务边际 utility，例如：

$$
\Delta U_D = \mathcal L_{RGB} - \mathcal L_{RGB+D}.
$$

则：

$$
\Delta U_D>0
$$

意味着 Depth 有帮助，

而：

$$
\Delta U_D<0
$$

意味着 Depth 具有负效用。

真正重要的问题不是：

> 模型能不能识别 corruption type？

而是：

$$
\boxed{ Q_D\uparrow \Rightarrow \Delta U_D\uparrow ? }
$$

也就是：

> **预测为可靠的 Depth 是否真的更值得被 geometry mechanism 使用？**

进一步最好做 spatial / region-level utility，而不仅仅是 image-level utility。

这一步能明显把 MMFR 与 UMFNet 式“learned confidence gating”区分开。

------

# 十一、ConD 对 MMFR baseline 设计提出了直接要求

ConD已经证明，仅通过让 DFormer 见到 modality-missing condition，再配合 copied encoder、freezing 和 zero-conv residual injection，就能显著恢复 missing-modality performance。

因此 MMFR 不能只比较：

$$
\text{Clean-trained DFormerv2}
$$

vs.

$$
\text{MMFR trained with corruptions}.
$$

否则无法回答：

> 提升到底来自 reliability mechanism，还是因为模型单纯见过 corruption？

必须至少加入：

### Baseline A：Original

$$
\text{clean training only}
$$

### Baseline B：Augmentation-only

$$
\text{same corruption distribution} + \text{same training budget}
$$

但：

$$
\text{without }Q_D
$$

### Baseline C：Generic feature gating

$$
\hat Q_D\odot F_D
$$

用于正面回应 UMFNet 类方法。

### MMFR

$$
\text{same corruption exposure} + \hat Q_D + \text{geometry-specific action}.
$$

真正重要的是：

MMFR>Baseline BMMFR>Baseline\ B

以及：

MMFR>Baseline C.MMFR>Baseline\ C.

否则很难证明 MMFR 的贡献来自 geometry reliability。

------

# 十二、建议的 P1 防撞强基线矩阵

后续主实验至少应考虑以下类型：

| 基线                            | 目的                              |
| ------------------------------- | --------------------------------- |
| 原始 DFormerv2                  | clean/full 基础性能               |
| corruption augmentation only    | 排除“只是见过坏 Depth”            |
| entire-modality dropout         | 对应 ConD / missing-modality 路线 |
| global binary Depth gate        | 对应 D3Net 式策略                 |
| global scalar reliability       | 对应 DCF 式策略                   |
| modality-level dynamic gate     | 对应 MAGIC / SimMLM 思路          |
| generic pixel-wise feature gate | **正面对齐 UMFNet**               |
| parameter-matched adapter       | 排除参数量收益                    |
| Oracle geometry reliability     | 测 action 上限                    |
| Predicted geometry reliability  | 实际 MMFR                         |
| Constant reliability            | 测 reliability 是否必要           |
| Shuffle reliability             | 测空间位置是否必要                |
| Inverse reliability             | 测 reliability 方向是否正确       |

这套基线的重要性在 UMFNet 和 ConD 加入后明显高于之前。

------

# 十三、必须同时报告 clean-performance preservation

RobustSeg 和 ConD 都已经明确把：

$$
\text{failure robustness}
$$

与：

$$
\text{full-modality accuracy}
$$

同时作为目标。

因此 MMFR不能只汇报 corruption gain。

建议明确报告：

$$
\Delta_{\text{clean}} = mIoU_{\text{MMFR,clean}} - mIoU_{\text{base,clean}}
$$

以及：

$$
\Delta_{\text{failure}} = mIoU_{\text{MMFR,failure}} - mIoU_{\text{base,failure}}.
$$

目标应当证明：

> failure robustness 的提升不是通过永久削弱 Depth geometry path 换来的。

否则一个简单“始终少用 Depth”的模型也可能获得类似 robustness。

------

# 十四、当前已明确不能再写的 novelty claim

经过这 8 篇 P1 后，以下表述建议全部进入**禁止列表**：

> ❌ 首次考虑 Depth 质量不可靠问题。

> ❌ 首次证明坏 Depth 会损害 RGB-D 模型。

> ❌ 首次学习 Depth reliability。

> ❌ 首次根据 Depth reliability 改变 Depth 的使用程度。

> ❌ 首次动态识别 robust / fragile modality。

> ❌ 首次根据 strong / weak modality 实施不同操作。

> ❌ 首次动态调整不同模态贡献。

> ❌ 首次在模态缺失时自适应重加权。

> ❌ 首次使用 confidence / uncertainty 抑制不可靠模态。

> ❌ 首次学习 pixel-wise continuous modality reliability。

> ❌ 首次用局部 reliability map 执行 spatially varying feature fusion。

> ❌ 首次针对 misalignment 使用 reliability-aware local fusion。

> ❌ 首次提升 DFormer 在 Depth missing 时的 robustness。

> ❌ 首次兼顾 missing-modality robustness 和 full-modality accuracy。

> ❌ 首次证明增加一个模态有时反而可能损害任务性能。

以上概念分别已经受到 D3Net、DCF、MAGIC、SimMLM、RobustSeg、ConD 和尤其 UMFNet 的直接限制。

------

# 十五、当前相对安全的 MMFR 表述

在只考虑这 8 篇 P1 的前提下，更合适的表述不是“first reliability-aware fusion”，而是：

> **MMFR studies corruption-grounded geometric reliability for RGB-D semantic segmentation, explicitly modeling the spatially varying trustworthiness of degraded Depth and using it to regulate a geometry-specific pathway rather than generic multimodal feature fusion.**

中文可压成：

> **MMFR关注受控 Depth degradation 下的局部几何可信度，并将与实际 corruption 后状态相对应的连续可靠性用于调节 RGB-D backbone 中特定的几何先验路径，而非对整个 Depth 模态进行图像级舍弃、输入修复或通用 feature gating。**

进一步，证据贡献可以表述为：

> **我们将可靠性估计与可靠性动作分开验证，并利用 oracle、constant、shuffle、inverse 以及 reliability-to-utility analysis 检验模型收益是否真正来自正确的几何可靠性判断。**

注意：这些表述目前只能称为**相对安全定位**，不能仅凭本轮 P1 就写成“首次”，最终仍需与 P0 及完整先例集合联合判断。

------

# 十六、8 篇 P1 在论文 Related Work 中应分别承担什么角色

建议不要把这 8 篇全部混在一个“missing modality”段落里，而是形成四条发展线。

### 1. Depth-quality-aware RGB-D

核心文献：

- D3Net；
- Calibrated RGB-D SOD。

它们证明：

$$
\text{Depth quality} \rightarrow \text{discard / calibrate}
$$

已经是成熟早期路线。D3Net强调低质量 Depth filtering，DCF则进一步学习 image-level reliability probability 并连续校正 Depth。

MMFR与它们的区别应放在：

> **local continuous geometry reliability + geometry-path control，而不是 image-level Depth rejection / raw-depth reconstruction。**

------

### 2. Missing / arbitrary modality robustness

核心文献：

- MAGIC；
- SimMLM；
- RobustSeg；
- ConD。

它们分别代表：

- robust/fragile modality ranking；
- dynamic MoE gating；
- teacher-student missing/noisy robustness；
- pretrained RGB-D/DFormer missing-modality adaptation。



MMFR与其区别：

> **不是“模态是否存在”，而是“Depth仍然存在时，其局部 geometry 是否可信”。**

------

### 3. Reliability-aware optimization

核心文献：

- ANGA。

它代表：

$$
\text{reliability} \rightarrow \text{optimization action}.
$$



MMFR需要强调：

> reliability 不作用于 sample weighting / gradient alignment，而作用于 inference-time geometry computation。

------

### 4. Pixel-wise uncertainty-aware fusion

核心近邻：

- **UMFNet**。

这篇必须单独重点讨论，因为它已经覆盖：

$$
\text{pixel uncertainty} \rightarrow \text{continuous confidence} \rightarrow \text{local multimodal suppression}.
$$



MMFR与其真正的差异必须写成：

> **learned feature uncertainty vs. explicit corruption-grounded geometry reliability**

以及：

> **generic cross-modal fusion vs. DFormerv2 geometry-specific intervention**

再加上：

> **performance ablation vs. reliability-to-utility counterfactual validation。**

------

# 十七、P1 对 B1 结构冻结的最终约束

综合 8 篇 P1 后，建议在 B1 冻结前强制检查以下问题：

-  B1 是否只是整幅 Depth 的 scalar score？若是，与 DCF/D3Net 距离过近。
-  B1 是否只是根据 $Q_D$ 给 Depth feature 做乘法？若是，与 UMFNet 类 confidence-guided fusion 距离过近。
-  B1 是否只是 modality selection / dynamic weighting？若是，与 MAGIC/SimMLM 重合。
-  B1 是否主要通过 masking/distillation 学鲁棒表示？若是，与 RobustSeg/ConD 主线重合。
-  B1 是否主要在 loss / gradient 层控制 unreliable samples？若是，应正面对照 ANGA。
-  reliability 是否有明确 corruption-state supervision，而不是自由学习 attention map？
-  action 是否进入 DFormerv2 的 geometry-specific path，而不是 generic feature fusion？
-  oracle $Q_D$ 是否已经证明 action 本身有效？
-  predicted $Q_D$ 是否能够接近 oracle action？
-  constant / shuffle / inverse 是否明显劣于正确 reliability？
-  同 corruption augmentation、同训练预算、无 reliability 的模型是否明显弱于 MMFR？
-  clean performance 是否保持，避免通过简单长期抑制 Depth 换 robustness？
-  是否证明 $\hat Q_D$ 与实际 Depth utility 之间存在正确关系？

如果最后四项证据不足，即使模型在 corrupted benchmark 上上涨，也仍容易被审稿人理解成：

> **corruption augmentation + another confidence gate。**

------

# 十八、P1 综合防撞后的最终 MMFR 边界

经过这 8 篇之后，MMFR最不应该变成的是：

$$
\boxed{ \text{Depth quality estimation} \rightarrow \text{adaptive fusion} }
$$

因为这一区域从 D3Net、DCF 到 MAGIC、SimMLM，再到 UMFNet，已经形成了非常完整的先例链。

也不应该只是：

$$
\boxed{ \text{pixel-wise confidence} \rightarrow Q_D\odot F_D }
$$

因为 UMFNet 已经使这种方案的 novelty 风险明显升高。

目前更值得守住的是：

Depth corruption after-state↓explicit spatial continuous geometric reliability↓DFormerv2 geometry-prior/path intervention↓local geometry utility↓oracle/predicted/shuffle/inverse causal validation\boxed{ \begin{aligned} &\text{Depth corruption after-state}\\ &\downarrow\\ &\text{explicit spatial continuous geometric reliability}\\ &\downarrow\\ &\text{DFormerv2 geometry-prior/path intervention}\\ &\downarrow\\ &\text{local geometry utility}\\ &\downarrow\\ &\text{oracle/predicted/shuffle/inverse causal validation} \end{aligned} }

换句话说，**MMFR真正应该回答的已经不是“Depth坏了怎么办”，而是三个更严格的问题：**

1. **Depth 的哪部分几何信息现在值得信？**
2. **DFormerv2 的哪部分 geometry computation 应该因此改变？**
3. **如何证明这个改变确实因为“可靠性判断正确”而有效，而不是一个普通 attention/gating 模块恰好涨点？**

------

# 十九、P1 最终审计结论

基于本轮 8 篇全文，**当前 MMFR 主线没有被整体撞死，但创新空间已经明显比初始蓝图更窄。**

其中：

- **D3Net + DCF** 卡住“Depth quality → action”的历史优先权；
- **MAGIC + RobustSeg** 卡住“robust/fragile 或 dominant/weak modality → differentiated treatment”；
- **SimMLM** 卡住“dynamic gating + task-level modality utility/ranking”；
- **ANGA** 卡住“reliability → optimization control”；
- **Condition Dropout** 卡住“DFormer + missing Depth robustness/adaptation”；
- **UMFNet** 则进一步卡住最危险的一层——“pixel-wise continuous confidence → local multimodal suppression”。

因此，本轮 P1 的核心结论可以压缩成一句：

> **MMFR不能再把“质量感知、可靠性估计、动态融合、局部软门控或 DFormer 缺失模态鲁棒性”写成原创点；当前仍值得争取的空间是“与受控 Depth corruption 最终状态一致的显式几何可靠性 → DFormerv2 特定 geometry path 的局部干预 → reliability-to-utility 与反事实证据”这一完整组合。**

在这 8 篇中，**UMFNet 应升级为后续方法冻结时的核心近邻对照，ConD 应升级为 augmentation/adaptation 强基线，D3Net 与 DCF 应承担历史 Depth-quality-aware 先例，MAGIC/RobustSeg/SimMLM 应承担 missing/arbitrary-modal 与 dynamic weighting 边界，ANGA 则作为 optimization-level reliability 的侧向边界。**

这也是目前 P1 层面对 MMFR 最重要的防撞结论。