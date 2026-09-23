# MMFR P0 六篇核心文献综合防撞车报告

> **审计范围：** 本报告针对原 P0 清单中的六篇全文进行合并审计：
> **AI019 MaskMentor、AI020 ECoLaF、AI021 QMF、PR089 SGMA、AI023 GeomPrompt / GeomPrompt-Recovery、AI025 CoReFuse-Med**。原审计要求从“输入与失效—质量表示—监督信号—控制位置—动作—证据”六个维度判断撞车，而不是只看模块名称是否相同。

------

## 一、P0 总结论

六篇论文合起来，已经基本覆盖了下面这条宽泛研究链：

$$
\text{Missing / Noisy / Degraded Modality} \rightarrow \text{Quality / Uncertainty / Reliability} \rightarrow \text{Adaptive Suppression / Repair / Reweighting} \rightarrow \text{Robust Multimodal Prediction}
$$

因此，MMFR **不能再把“可靠性感知的鲁棒多模态融合”作为抽象创新**。

尤其以下概念均已有直接或强相邻先例：

- missing-modality training；
- corruption-aware training；
- uncertainty / reliability guided dynamic weighting；
- 像素级或空间级 reliability map；
- 低质量模态动态抑制；
- reliability-guided feature fusion；
- task-driven degraded Depth correction；
- feature-space corruption suppression；
- 模态贡献重新校准；
- robustness 与 task performance 的相关性验证；
- whole-modality occlusion 衡量实际模态贡献。

六篇全文审完以后，MMFR 剩余最值得保护的研究位置已经明显收窄为：

$$
\boxed{ \text{corruption-grounded pixel-wise Depth diagnosis} \rightarrow \text{DFormerv2 geometry-prior-specific intervention} \rightarrow \text{counterfactual geometry-utility verification} }
$$

更具体地说：

$$
\boxed{ Q_D \rightarrow \hat r_D \rightarrow A_{\mathrm{geo}} \rightarrow U_D^{\mathrm{geo}} }
$$

其中四个量必须有明确区别：

- $Q_D$：corruption 后 Depth 的**真实质量状态**；
- $\hat r_D$：模型预测的 Depth reliability；
- $A_{\mathrm{geo}}$：对 DFormerv2 几何路径采取的 action；
- $U_D^{\mathrm{geo}}$：Depth geometry 对最终任务的真实边际效用。

**这四者不能继续混成一个普通 attention weight。**

------

## 二、六篇 P0 文献的功能覆盖图

| 文献                          | 主要问题                                       | 质量/可靠性表示                         | 控制位置                          | Action                                                       | 对 MMFR 的主要阻断                                           |
| ----------------------------- | ---------------------------------------------- | --------------------------------------- | --------------------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| **AI019 MaskMentor**          | RGB/Depth 整模态缺失                           | 无显式 reliability                      | representation pretraining        | modality-level + patch-level masking；complete teacher → missing student | 阻断“训练阶段模拟模态缺失”“单模型适应 RGB-D/RGB-only/Depth-only”作为创新 |
| **AI020 ECoLaF**              | sensor missing + 模态冲突                      | **逐像素 conflict-derived reliability** | segmentation output / late fusion | Dempster–Shafer discounting                                  | 阻断“像素 reliability → 动态降低坏模态贡献”作为创新          |
| **AI021 QMF**                 | low-quality/noisy multimodal data              | sample-level uncertainty                | decision-level late fusion        | uncertainty-aware dynamic weighting                          | 阻断“质量越差权重越低”的抽象思想，并给出理论 criterion       |
| **PR089 SGMA**                | arbitrary missing modalities、fragile modality | **多尺度空间 robustness map**           | intermediate features             | reliability-weighted adaptive fusion + fragile-modality sampling | 阻断“空间 reliability → feature-level adaptive fusion”作为创新 |
| **AI023 GeomPrompt-Recovery** | missing / degraded Depth                       | 无显式 reliability                      | **raw Depth / geometry input**    | task-driven bounded residual correction                      | 直接阻断 raw-Depth residual repair 与“只恢复任务有用几何”路线 |
| **AI025 CoReFuse-Med**        | 存在但低质量的模态                             | 无独立 reliability target               | **浅层+深层 feature space**       | corruption suppression + channel/spatial calibration + attention | 阻断“坏模态不修输入、而在内部过滤和重新加权”这一宽泛路线     |

MaskMentor 的 M²IM 直接在训练中随机整模态 masking，并由完整模态 teacher 通过 token/pixel reconstruction 指导 missing-modality student；同一个训练后的模型被用于 RGB-D、Only-RGB 和 Only-Depth。

ECoLaF 则更进一步：它对各模态 evidential segmentation outputs 逐像素计算冲突，产生 $M\times H'\times W'$ 的 discounting map，再降低高冲突模态的证据后做 Dempster fusion；sensor failure 测试采用单模型并在测试时 zero-out 缺失模态。

QMF 已经从理论上把动态 fusion weight 与模态 loss 联系起来：其核心条件是动态权重应与对应单模态 loss 非正相关，即坏模态得到更低权重；具体实现使用 uncertainty estimation 产生动态融合权重。

SGMA 已经产生真正的多尺度空间 robustness map $r_m^i\in\mathbb R^{H_i\times W_i}$，根据 semantic prototype 与各模态 feature 的匹配关系进行 adaptive feature fusion，而且训练时又将 robustness 反转为 fragile-modality sampling probability。

GeomPrompt-Recovery 已明确实现 RGB + degraded Depth → bounded residual correction，并直接修正 raw Depth；它只依赖 segmentation supervision，而不要求恢复真实 metric Depth。

CoReFuse-Med 又补上了另一条重要路线：低质量模态无需先恢复输入，可以在浅层 feature transmission 中抑制 corruption，再在深层通过 channel calibration、spatial gating 与 symmetric cross-attention 重新调整模态贡献。

------

## 三、六维综合防撞车审计

### 3.1 输入与失效：单纯“模拟失效”已经没有创新空间

六篇已经覆盖：

- whole-modality missing；
- Gaussian / salt-pepper 等 noise；
- degraded Depth；
- quantization；
- hole/dropout；
- blur；
- banding；
- scale shift；
- resolution-induced degradation；
- arbitrary modality subsets。

MaskMentor 证明 modality-level masking 本身就能大幅提高 missing robustness；GeomPrompt-Recovery 训练时则以 $20%$ clean、$80%$ corruption 的形式从 quantize、hole、dropout、noise、blur、banding、scale shift 中随机选择 degradation，并随机采 severity。

因此 MMFR 的：

> clean + corruption mixed training、random severity、entire missing、local dropout 等

**只能作为实验 protocol，不能作为算法创新。**

MMFR 仍然有价值的，是将这些失效统一映射到：

$$
Q_D(x,y)\in[0,1]
$$

而不是仅仅让网络“见过这些 corruption”。

------

## 四、真正需要守住的核心：MMFR 的 reliability 必须和已有 reliability 语义不同

六篇论文已经出现至少三种“可靠性”：

### 1. QMF：prediction uncertainty

回答：

> 这个模态自己的分类器当前有多容易犯错？

大致对应：

$$
u_m\uparrow \Rightarrow w_m\downarrow.
$$

### 2. ECoLaF：relative cross-modal disagreement

回答：

> 这个模态和其他模态是否意见冲突？

$$
\mathrm{Conf}_m\uparrow \Rightarrow \alpha_m\downarrow.
$$

### 3. SGMA：task-conditioned semantic usefulness

回答：

> 当前语义/尺度下，这个模态 feature 和 semantic prototype 有多匹配、多值得使用？

这三个都不严格等价于：

> **Depth 传感数据本身到底有没有被 corruption 破坏。**

因此 MMFR 最值得坚持的定义应该是：

$$
Q_D = V_D^{\mathrm{final}} \odot R_D^{\mathrm{syn}}
$$

即 $Q_D$ 首先描述 **corruption 后最终 Depth 状态**，而不是由 segmentation 输出是否正确、是否和 RGB 一致、或者是否获得较高 attention 反推。

于是必须严格区分：

$$
\boxed{ \text{Depth quality} \neq \text{cross-modal agreement} \neq \text{task usefulness} }
$$

例如：

- 一个**完全正确**的 Depth 像素可能对某个纯颜色类别作用有限；
- 一个轻微受损的 Depth 像素仍可能对边界非常有价值；
- 多个错误模态形成一致意见时，正确的少数模态反而可能表现为高 conflict。

因此 MMFR 如果最终只是学习：

$$
r_D=\text{attention weight}
$$

那么和 SGMA/ECoLaF 的边界会迅速消失。

------

## 五、B1 最大红线：不能退化成已有的五种 Action

### 红线 A：Raw Depth repair

以下方案风险最高：

$$
D' = D+\Delta D(RGB,D).
$$

甚至加入 reliability：

$$
D' = D+(1-r_D)\Delta D
$$

也没有解决根本问题。

GeomPrompt-Recovery 已经明确实现：

$$
\mathrm{corr} = s\tanh(\Delta_{\mathrm{full}})
$$

$$
p_{\mathrm{raw}} = \mathrm{clamp} \left( \tilde D+\mathrm{corr},0,255 \right),
$$

并且从 zero-initialized approximate identity mapping 开始训练。

所以：

> **raw corrupted Depth + bounded task-driven residual repair**

应直接列为 B1 禁区。

------

### 红线 B：普通 Feature Gate

如果 B1 最终只是：

$$
F'_D = r_D\odot F_D,
$$

然后正常融合，也不够安全。

因为：

- QMF 已经做 quality/uncertainty weighting；
- ECoLaF 已有 pixel-wise reliability discount；
- SGMA 已有 multi-scale spatial reliability weighted feature fusion；
- CoReFuse-Med 已有 channel calibration + spatial gating。

因此：

> **pixel-wise 并不会自动带来新颖性。**

------

### 红线 C：Conflict/Agreement Reliability

如果：

$$
r_D = f \left( \mathrm{difference}(F_R,F_D) \right),
$$

随后降低 Depth 权重，

它很容易被归入 ECoLaF/SGMA 一类：

> cross-modal inconsistency → reliability → suppression。

可以作为 baseline，但不宜成为 MMFR 主方法。

------

### 红线 D：Feature purification / denoising

例如：

```text
Depth feature
→ low/high-frequency decomposition
→ noise suppression
→ gated fusion
```

同样危险。

CoReFuse-Med 已经利用 spatial-scale decomposition 将 feature 分为 structural base 和 detail residual，再分别做竞争性加权与 residual gating，目标就是防止 corruption 在 feature transmission 中继续传播。

所以：

> “我们不修 raw Depth，而是在 feature space 中过滤坏信息”

也已经不能作为主创新。

------

### 红线 E：Modality-aware training 本身

任何：

- modality dropout；
- missing exposure；
- robustness-guided sampling；
- hard-example fragile modality sampling；
- complete teacher → missing student；

都已经有 MaskMentor/SGMA 等直接先例。

这些可以使用，但必须定位为：

> **训练策略 / necessary control**

而不是 MMFR 的 novelty。

------

## 六、B1 应该锁定的结构边界：Geometry-specific，而不是 Modality-specific

六篇 P0 审完以后，**“feature-level”已经不够窄**。

MMFR 必须进一步进入：

> **DFormerv2 中由 Depth 构造或调制 geometry prior 的具体计算路径。**

最理想的抽象不是：

$$
F'_D = r_DF_D,
$$

而更接近：

$$
G_D^l = \mathcal G_l(D),
$$

然后 reliability 改变的是：

$$
A_{\mathrm{geo}}^l = \mathcal A \left( G_D^l,\hat r_D \right),
$$

最终：

$$
F_{\mathrm{B1}}^l = F_{\mathrm{base}}^l + \beta_l(\hat r_D)\Delta F_{\mathrm{geo}}^l.
$$

这里关键不在具体公式，而在结构原则：

> **Action 对象应当是 Depth-derived geometry prior / geometry relation / geometry-attention contribution，而不是普通 Depth feature amplitude。**

因此论文问题应被定义成：

> **“当局部 Depth 不可信时，一个依赖 Depth 几何关系进行表示学习的 RGB-D Transformer 应如何改变几何先验的使用方式？”**

而不是：

> “如何降低低质量 Depth 的权重？”

后一句已经基本被六篇 P0 文献占满。

------

## 七、六篇论文共同要求 MMFR 把“诊断”和“动作”彻底拆开

GeomPrompt、SGMA、CoReFuse 的共同特点是：

$$
\text{input/features} \rightarrow \text{action}
$$

而 MMFR 最值得保留的逻辑应是：

$$
\text{Depth state} \rightarrow \underbrace{\hat r_D}_{\text{diagnosis}} \rightarrow \underbrace{A_{\mathrm{geo}}(\hat r_D)}_{\text{action}}.
$$

这样才能分别问两个问题：

### 问题一：诊断是否正确？

$$
\hat r_D \approx Q_D ?
$$

### 问题二：即使知道了真实质量，Action 是否正确？

$$
A_{\mathrm{geo}}(Q_D) > A_{\mathrm{none}} ?
$$

这是目前六篇中没有形成完整分解验证链的地方。

因此 B1 的优先顺序必须是：

$$
\boxed{ \text{先验证 Oracle Action} \rightarrow \text{再接 Predicted Reliability} }
$$

如果用真实 $Q_D$ 控制 geometry action 都不能明显提升，那么继续训练 reliability estimator 没有意义。

------

## 八、reliability-to-utility 不能再泛泛地写，需要升级成 geometry-specific utility

QMF 已经要求动态 weight 与单模态 loss 负相关。

SGMA 已经比较 robustness score 与各模态 segmentation performance 排名。

CoReFuse-Med 更直接使用：

> modality occlusion → DSC drop

衡量不同模态的实际 predictive impact，并发现 gradient contribution 与 inference utility 可能严重不一致。

因此 MMFR 不能再宣称：

> “首次把 reliability 与 task utility 建立联系。”

真正可守的应该是：

$$
\boxed{ \text{pixel/local Depth reliability} \leftrightarrow \text{DFormerv2 geometry-specific marginal utility} }
$$

例如可定义一个反事实 geometry utility：

$$
U_D^{\mathrm{geo}} = \mathcal L_{\mathrm{geo\ suppressed}} - \mathcal L_{\mathrm{geo\ enabled}}.
$$

若 $U_D^{\mathrm{geo}}>0$，说明使用 Depth geometry 对该样本/区域有益；若 $U_D^{\mathrm{geo}}<0$，说明 geometry 当前产生负效应。

随后验证：

$$
\hat r_D\uparrow \quad\Rightarrow\quad U_D^{\mathrm{geo}}\uparrow
$$

是否在 reliability bins、corruption types 和 severity 上稳定成立。

这比 QMF 的：

$$
w_m \leftrightarrow \ell_m
$$

和 CoReFuse 的：

$$
\text{whole-modality occlusion} \leftrightarrow \Delta DSC
$$

更接近 MMFR 真正需要建立的机制证据。

------

## 九、A2 在六篇 P0 之后的准确定位

当前 A2 的：

> failure-aware training + Depth reliability estimator

仍然可以保留，但**不能作为最终创新主体**。

MaskMentor 已证明 missing exposure 本身具有巨大收益；GeomPrompt 已使用大范围 random corruption/severity training；SGMA 又证明 training-time fragile-modality rebalancing 本身可能贡献很大。

所以最终实验必须拆开：

$$
\text{augmentation effect}
$$

和

$$
\text{reliability/action effect}.
$$

至少需要：

| Variant                                        | 回答的问题                                |
| ---------------------------------------------- | ----------------------------------------- |
| Original DFormerv2                             | clean baseline                            |
| DFormerv2 + matched corruption augmentation    | 仅“见过 corruption”能提升多少             |
| + reliability estimator，但不参与 segmentation | diagnosis 是否只是辅助任务 regularization |
| Oracle geometry action                         | action 本身有没有价值                     |
| Predicted reliability geometry action          | 完整 MMFR                                 |
| Parameter-matched generic gate                 | 是否普通加参数/feature gate 已足够        |

否则最终 robustness 增益很容易被解释成：

> “只是因为 MMFR 在训练阶段见过这些 corruption。”

------

## 十、B1 必须加入的 P0 级强基线

经过六篇全文，简单只与 DFormerv2 比已经不够。

建议最终至少覆盖以下机制类别：

| Baseline 类别                               | 对应 P0 先例            | 它要回答的问题                                             |
| ------------------------------------------- | ----------------------- | ---------------------------------------------------------- |
| **Augmentation-only**                       | MaskMentor / GeomPrompt | 收益是不是只来自 failure exposure？                        |
| **Uncertainty scalar gate**                 | QMF                     | 普通 uncertainty weighting 是否已经足够？                  |
| **Conflict-derived reliability**            | ECoLaF                  | 跨模态 disagreement 能否替代 $Q_D$？                       |
| **Spatial feature reliability gate**        | SGMA                    | 普通 spatial weighting 能否替代 geometry-specific action？ |
| **Raw-Depth residual repair**               | GeomPrompt-Recovery     | 直接修 Depth 是否已经足够？                                |
| **Generic feature suppression/calibration** | CoReFuse-Med            | 普通 feature purification + gate 是否已经足够？            |
| **MMFR geometry-specific action**           | Proposed                | 是否必须针对 geometry prior 本身干预？                     |

不一定需要逐篇完整重现，但必须至少有**概念和参数量匹配的控制版本**。

------

## 十一、反事实实验必须成为 MMFR 的核心证据，而不能只是普通消融

完整 B1 至少应该包含：

$$
A_{\mathrm{none}}
$$

$$
A_{\mathrm{constant}}
$$

$$
A_{\mathrm{oracle}}
$$

$$
A_{\mathrm{pred}}
$$

$$
A_{\mathrm{shuffle}}
$$

$$
A_{\mathrm{inverse}}
$$

其中期望看到的关系不是机械要求某个固定排序，而是至少满足：

1. **Oracle 比 no-action 明显好**
   → 证明 action 有潜力。
2. **Predicted 接近 Oracle 且优于 constant**
   → 证明 reliability prediction 带来实际价值。
3. **Shuffle 后明显下降**
   → 证明 reliability 的空间对应关系有意义。
4. **Inverse 明显恶化**
   → 证明 action 依赖 reliability 的“方向正确”。
5. **Parameter-matched generic gate 不足以复制收益**
   → 证明不是单纯增加一个 gating block。

这样论文才能从：

> “加模块以后 corruption mIoU 更高”

升级成：

> **“识别对了什么 + 为什么该这样处理 + 如果 reliability 错了系统会怎样”**

的可证伪机制论证。

------

## 十二、测试协议也必须防止与现有工作退化为同一证据

### 1. 一个模型覆盖全部 conditions

GeomPrompt-Recovery 已覆盖多个 corruption，但 CoReFuse-Med 的 10%、20%、30% slice-retention 实验会针对相应条件分别重新训练模型。

因此 MMFR 最好坚持：

$$
\boxed{ \text{one checkpoint} \rightarrow \text{multiple corruption types and severities} }
$$

而不是每个 failure condition 单独训练模型。

------

### 2. 保留 held-out corruption

这一项非常有价值。

训练：

$$
\mathcal C_{\mathrm{train}}
$$

测试再加入：

$$
\mathcal C_{\mathrm{heldout}} \notin \mathcal C_{\mathrm{train}}.
$$

这可以区分：

> 网络记住了 corruption 模式

和

> 网络真的学习了 transferable Depth quality。

------

### 3. misalignment 应单独保留

六篇 P0 中都没有系统解决：

> **RGB 与 Depth 都各自有正常数值，但它们之间空间对应关系错误**

这个问题。

misalignment 的本质不是简单：

$$
D\rightarrow\tilde D
$$

而是：

$$
D(x,y) \rightarrow D(x+\Delta x,y+\Delta y).
$$

它尤其适合验证：

> corruption-grounded quality 是否比纯 semantic agreement / feature weighting 更可靠。

因此这是 MMFR benchmark 中值得重点保留的 condition。

------

## 十三、最终禁止使用的 Novelty 表述

以下表述建议直接加入“禁用 claim”清单：

> “首次提出可靠性感知的 RGB-D 语义分割。”

不成立。

> “首次预测像素级模态可靠性。”

ECoLaF 和 SGMA 已有直接空间级先例。

> “首次根据可靠性动态降低低质量模态贡献。”

QMF、ECoLaF、SGMA 均已覆盖。

> “首次利用模态冲突识别坏模态。”

ECoLaF 已覆盖。

> “首次通过训练阶段模拟模态缺失提高 RGB-D robustness。”

MaskMentor 等已覆盖。

> “首次在不恢复真实 Depth 的情况下学习任务驱动几何修正。”

GeomPrompt 已明确提出。

> “首次不修复输入，而是在内部 feature space 抑制低质量模态。”

CoReFuse-Med 已覆盖。

> “首次证明低质量辅助模态可能使融合性能下降。”

QMF 与 CoReFuse-Med 均已有直接证据。

> “首次验证 reliability 与任务性能/utility 的关系。”

QMF、SGMA、CoReFuse-Med 都已经进行了不同形式的相关性或 occlusion-based contribution 分析。

------

## 十四、建议采用的 MMFR 核心定位

最终可以将研究问题收敛成：

> 现有工作已经从模态 masking、自蒸馏、预测不确定性、跨模态冲突、语义鲁棒性、task-driven Depth correction 以及 corruption-aware feature calibration 等不同角度处理缺失或低质量多模态输入。然而，这些方法通常将模态质量隐式表示为 prediction uncertainty、cross-modal disagreement 或 task-conditioned contribution，或直接学习修复和融合动作。MMFR 进一步关注 RGB-D 网络内部由 Depth 建立的几何先验：首先以 corruption 后 Depth 最终状态构造显式像素级连续质量目标 $Q_D$，再独立预测 Depth reliability，并研究该 reliability 应如何作用于 DFormerv2 的 geometry-prior path，而非直接修复 raw Depth 或进行通用 feature reweighting。最后，通过 oracle、constant、shuffle、inverse 和 geometry-specific marginal utility 等反事实实验，将“失效诊断是否正确”与“几何处置是否正确”分别验证。

核心可以压缩成：

$$
\boxed{ \text{Quality} \rightarrow \text{Diagnosis} \rightarrow \text{Geometry-specific Action} \rightarrow \text{Utility} }
$$

而不是已有工作普遍采用的：

$$
\boxed{ \text{Quality / Uncertainty} \rightarrow \text{Weight / Gate / Repair} \rightarrow \text{Prediction} }
$$

------

## 十五、B1 冻结前 P0 硬检查表

### A. 结构防撞车

-  B1 不直接修改 raw Depth。
-  B1 不以 pseudo Depth / geometric prompt synthesis 为主机制。
-  B1 不只是 $r_D\odot F_D$。
-  B1 不只是 semantic prototype / RGB-D disagreement 产生 reliability。
-  B1 不只是 feature denoising / base-residual filtering。
-  B1 action 明确位于 **DFormerv2 Depth-derived geometry-prior / Geometry Self-Attention path**。
-  能在结构图和代码中指出“若移除此 geometry intervention，则退化回 baseline DFormerv2”的最小控制点。

### B. 可靠性定义

-  $Q_D$ 描述 corruption 后最终 Depth 状态，而不是 segmentation confidence。
-  natural invalid 与 corruption-induced invalid 均进入最终质量。
-  有值但低质量的 Depth 不简单二值化为 valid。
-  reliability 为连续量而不是 clean/corrupt 二分类。
-  明确区分 $Q_D$、$\hat r_D$ 与 $U_D^{\mathrm{geo}}$。

### C. Action 正确性

-  Oracle reliability action 明显优于 no-action。
-  Predicted reliability 能复制相当部分 oracle gain。
-  Constant control 无法复制完整收益。
-  Shuffle reliability 明显削弱效果。
-  Inverse reliability 产生预期方向的性能恶化。
-  Parameter-matched generic feature gate 无法完全复制 geometry-specific gain。

### D. 训练混淆控制

-  有 augmentation-only baseline。
-  所有主要方法使用相同 corruption manifest。
-  checkpoint selection 不利用 corruption test 结果。
-  clean 性能单独报告，而不是被 robustness average 掩盖。

### E. 泛化证据

-  单 checkpoint 覆盖完整 corruption × severity 矩阵。
-  至少有 held-out corruption。
-  至少有 unseen severity / severity extrapolation。
-  misalignment 单独报告。
-  报告 clean retention、average robustness 与 worst-case robustness。

### F. 机制证据

-  $\hat r_D$ 与 $Q_D$ 有 calibration / ranking 证据。
-  $\hat r_D$ 与 $U_D^{\mathrm{geo}}$ 有 utility-bin 证据。
-  对局部 corruption 有 spatial correspondence。
-  证明收益来自“正确 reliability + 正确 geometry action”，而不仅是额外参数或 corruption training。

------

## 十六、P0 最终审计判定

### 针对这六篇 P0 文献：**条件通过，可以继续 B1，但研究边界必须进一步收窄。**

六篇全文**没有发现与下面完整组合完全同构的工作**：

$$
\boxed{ Q_D^{\mathrm{corruption-grounded}} \rightarrow \hat r_D^{\mathrm{explicit}} \rightarrow A_{\mathrm{DFormerv2\ geometry}} \rightarrow U_D^{\mathrm{geometry-specific}} }
$$

但它们已经分别占据了这条链外围的几乎所有宽泛概念。

因此 MMFR 当前真正还可辩护的 novelty，不再是：

> reliability、adaptive fusion、missing robustness、corruption robustness、Depth repair 或 feature suppression。

而是这三个东西的**特定组合**：

$$
\boxed{ \text{1. corruption-grounded Depth diagnosis} }
$$

$$
\boxed{ \text{2. DFormerv2 geometry-prior-specific action} }
$$

$$
\boxed{ \text{3. diagnosis/action 分离的 counterfactual causal evidence} }
$$

其中第二项是现在最关键的结构护城河，第三项是最关键的证据护城河。

**如果 B1 最终退化为 raw-Depth residual、普通 feature gate、spatial reliability weighting 或 feature purification，则 P0 应重新判为高风险，不能冻结。**

**如果 B1 能明确落在 DFormerv2 的 geometry-prior / Geometry Self-Attention 内部，并且 oracle-first 与 counterfactual evidence 能成立，那么至少针对这六篇 P0 核心文献，当前收窄后的 MMFR 方向仍然存在清晰可辩护的研究空间。**