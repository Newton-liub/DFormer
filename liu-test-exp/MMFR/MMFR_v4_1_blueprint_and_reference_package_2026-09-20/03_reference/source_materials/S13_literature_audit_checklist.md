- # MMFR 新增文献与 Idea 防撞车文献整理清单（修订版）

  > **修订日期：** 2026-09-18
  > **修订依据：** 原《MMFR 新增文献与 Idea 撞车审计》以及后续 P0、P1、P2 全文综合审计。
  > **用途：** 当前仅用于后续 B1 结构设计与 Related Work 的防撞参考，不在此阶段展开具体模块设计。

  ------

  ## 一、当前防撞总边界

  经过 P0、P1、P2 补全文献后，以下宽泛思想均不能再作为 MMFR 的核心创新点：

  - Depth / 模态质量评估后抑制低质量信息；
  - uncertainty / confidence / reliability guided fusion；
  - missing-modality masking、dropout、distillation 或 adaptation；
  - pixel-wise / spatial reliability map；
  - reliability-guided generic feature gating；
  - reliability-guided generic adapter modulation；
  - degraded Depth 的 task-driven residual repair；
  - feature-space corruption suppression / purification；
  - quality-aware MoE / expert routing；
  - “诊断—恢复—重新评估—融合”闭环；
  - 同时处理 missing 与 noisy modalities；
  - 根据 robust / fragile 或 dominant / weak modality 改变处理方式。

  当前仍值得继续保护的 MMFR 主线应进一步收窄为：

  > **针对受控 Depth corruption 后的最终状态定义显式像素级连续质量 $Q_D=V_D^{\mathrm{final}}\odot R_D^{\mathrm{syn}}$，研究该状态如何影响 DFormerv2 内部 Depth-derived geometry prior / geometry interaction 的实际效用，并通过 geometry-specific intervention 与反事实实验分别验证“诊断正确”和“处置正确”。**

  因此当前核心链条应保持为：

  **Depth final-state quality → explicit reliability diagnosis → DFormerv2 geometry-specific action → geometry utility → counterfactual validation**

  而不是一般性的：

  **quality / uncertainty → gate / fusion / repair → prediction**。

  ------

  # 二、P0：B1 冻结前必须重点对照

  P0 六篇全文已经完成审计，是当前 B1 结构设计最主要的直接边界。

  | 编号      | 文献/方法                        | 对 MMFR 的主要限制                                           |
  | --------- | -------------------------------- | ------------------------------------------------------------ |
  | **AI019** | MaskMentor                       | missing-modality masking、自蒸馏与 RGB-D/RGB-only/Depth-only 单模型鲁棒训练已有直接先例 |
  | **AI020** | ECoLaF                           | 已有逐像素 conflict-derived reliability，并利用 reliability discounting 降低坏模态贡献 |
  | **AI021** | QMF                              | uncertainty-aware dynamic fusion 已建立，并明确要求动态权重与对应模态 loss/utility 建立合理关系 |
  | **PR089** | SGMA                             | 已有多尺度空间 robustness map、reliability-weighted feature fusion 和 fragile-modality sampling |
  | **AI023** | GeomPrompt / GeomPrompt-Recovery | degraded Depth + task-driven bounded raw-Depth residual correction 已有直接先例 |
  | **AI025** | CoReFuse-Med                     | corruption-aware feature suppression、channel/spatial calibration 和模态贡献重新平衡已有先例 |

  ### P0 对 B1 的主要红线

  - 不应采用 raw Depth residual repair 作为核心结构；
  - 不应退化成 $F'_D=r_D\odot F_D$ 一类普通 feature gate；
  - 不应只依据 RGB/Depth disagreement 构造 reliability；
  - 不应仅做 feature denoising / purification；
  - modality dropout、corruption augmentation 等只能作为训练策略或基线，而不能作为主要 novelty；
  - B1 的 action 应尽量落入 **DFormerv2 的 geometry-prior / Geometry Self-Attention 相关路径**。

  ------

  # 三、P1：重要近邻与强基线

  P1 八篇全文已经完成审计，主要用于限制 MMFR 在 Depth-quality、missing-modality、dynamic gating 和 dense confidence 等方向上的表述。

  | 编号              | 文献/方法                                                    | 对 MMFR 的主要限制                                           |
  | ----------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
  | **AI022 / MAGIC** | Centering the Value of Every Modality                        | robust / fragile modality 排序及 arbitrary-modality segmentation 已存在 |
  | **ANGA**          | Anchor-Guided Gradient Alignment for Incomplete Multimodal Learning | reliability 决定 sample selection / optimization / gradient behavior 已存在；当前仍保留待正式 AI 编号状态 |
  | **AI017**         | Calibrated RGB-D Salient Object Detection                    | image-level Depth reliability + reliability-conditioned raw/estimated Depth calibration 已存在 |
  | **AI026**         | SimMLM                                                       | dynamic modality gating、MoE 和“增加模态不应恶化任务性能”的约束已有先例 |
  | **AI016**         | D3Net                                                        | low-quality Depth 判定后在 RGB 与 RGB-D path 之间选择已有早期直接先例 |
  | **PR090**         | RobustSeg                                                    | dominant / non-dominant modality、missing/noisy robustness 和差异化 teacher-student transfer 已存在 |
  | **AI024**         | Condition Dropout                                            | 直接针对 DFormer 的 RGB/Depth missing-modality adaptation 已存在 |
  | **PR029**         | UMFNet                                                       | pixel-wise uncertainty → continuous confidence → spatial/channel modulation 已高度覆盖普通局部 reliability gating |

  ### P1 后的重要修正

  尤其由于 **PR029 / UMFNet**，以下表述应正式删除：

  > “pixel-wise continuous reliability 本身是 MMFR 的创新。”

  MMFR 的区别必须从“空间粒度”进一步收窄到：

  > **corruption-grounded Depth final-state reliability + DFormerv2 geometry-specific control**。

  同时，AI024 表明：

  > “DFormer + Depth missing robustness”

  本身也不能作为研究定位。

  ------

  # 四、P2：新思路风险哨兵

  P2 四篇全文已经完成审计。它们不与 P0/P1 核心先例完全等权，但用于监控新方法链和术语边界。

  | 编号      | 文献/方法                                                    | 当前定位              | 对 MMFR 的意义                                               |
  | --------- | ------------------------------------------------------------ | --------------------- | ------------------------------------------------------------ |
  | **AI018** | Uncertainty Inspired RGB-D Saliency Detection                | P2-L，概念边界        | 主要研究 annotation/output uncertainty，不是 Depth reliability 的直接撞车，但要求严格区分不同 uncertainty |
  | **AI027** | UMQ                                                          | P2-H                  | explicit quality estimation → enhancement → quality-aware MoE 已形成完整链条 |
  | **AI028** | PRIME                                                        | P2-H                  | corruption reliability → restoration → reassessment → precision fusion 已覆盖诊断—修复—复诊闭环 |
  | **MoSA**  | Modality-Aware Spatially-Adaptive Adaptation for RGB-X Semantic Segmentation | **P2-H / 近 P0 警戒** | RGB-D/RGB-T segmentation 中已有 spatial reliability、spatial adapter modulation 和 reliability-guided fusion，是当前最直接的新近风险之一 |

  ### P2 后新增的重要红线

  - 不能以“spatial reliability”本身作为创新；
  - 不能以 $Q_D(x,y)$ 直接控制普通 fusion weight 为核心；
  - 不能以 $Q_D(x,y)$ 调节 generic adapter / residual branch 强度为核心；
  - 不能以 quality-guided restoration 为核心；
  - 不能以 quality-aware expert routing 为核心；
  - 不能把“diagnose → repair → reassess → fuse”描述为新闭环；
  - prediction entropy、variance 或 segmentation confidence 不能直接等同于 Depth sensor/state reliability。

  特别需要与 MoSA 区分：

  > **MoSA 更接近“当前位置哪个模态应该贡献更多？”；MMFR 应转向“受损 Depth 何时使 DFormerv2 的 geometry prior 产生负效用，以及怎样针对该 geometry mechanism 进行有效干预？”**

  ------

  # 五、需要长期保留的历史近邻

  除上述 P0/P1/P2 外，以下早期工作仍应长期保留在防撞清单中。

  | 编号      | 文献                                                         | 用途                                                         |
  | --------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
  | **AI015** | Incomplete RGB-D Salient Object Detection: Conceal, Correlate and Fuse | Depth Quality Assessment Regression + 图像级低质量 Depth 丢弃，是 Depth-quality-aware action 的重要历史先例 |
  | **AI016** | D3Net                                                        | 低质量 Depth filtering / path selection 的早期先例           |
  | **AI017** | Calibrated RGB-D SOD                                         | reliability-conditioned Depth calibration 先例               |
  | **AI018** | Uncertainty Inspired RGB-D Saliency Detection                | uncertainty 概念边界参考                                     |
  | **AI021** | QMF                                                          | low-quality multimodal dynamic fusion 与理论关系参考         |

  因此论文中不能把“发现坏 Depth 会损害 RGB-D 模型”或“根据 Depth 质量决定是否使用 Depth”描述为新发现。

  ------

  # 六、当前需要严格区分的四个量

  后续方法与实验中建议始终明确区分：

  - **$Q_D$：Depth final-state quality / sensor-state reliability**
    描述 corruption 后 Depth observation 本身是否可信。
  - **$\hat r_D$：predicted reliability**
    网络对 $Q_D$ 的预测，而不是直接等同于 attention weight。
  - **$U_D^{\mathrm{geo}}$：geometry-specific marginal utility**
    描述当前 Depth geometry 对 DFormerv2 任务预测究竟有帮助还是产生负作用。
  - **$G_A$：action gain**
    描述执行某个 reliability-conditioned geometry action 后实际获得的收益。

  因此必须避免：

  **$Q_D=\text{attention}=\text{task confidence}=\text{utility}$**

  这种隐式混用。

  ------

  # 七、当前建议的 B1 防撞基线

  后续 B1 至少应保留以下类型的控制实验：

  | 基线                                     | 主要回答的问题                                     |
  | ---------------------------------------- | -------------------------------------------------- |
  | Original DFormerv2                       | 原始 clean / failure 基线                          |
  | Corruption augmentation only             | 收益是否仅来自见过 corruption                      |
  | Entire-modality dropout / adaptation     | 对照 Condition Dropout、MaskMentor 等 missing 路线 |
  | Global binary Depth gate                 | 对照 D3Net / CCF 类策略                            |
  | Global scalar reliability                | 对照 Calibrated RGB-D 类策略                       |
  | Uncertainty / conflict gate              | 对照 QMF、ECoLaF                                   |
  | Generic pixel-wise feature gate          | 对照 SGMA、UMFNet、MoSA 类方法                     |
  | Generic spatial adapter modulation       | 对照 MoSA                                          |
  | Raw-Depth residual repair                | 对照 GeomPrompt-Recovery                           |
  | Generic feature suppression/calibration  | 对照 CoReFuse-Med                                  |
  | Parameter-matched generic module         | 排除参数量收益                                     |
  | Oracle geometry action                   | 判断 action 本身是否成立                           |
  | Predicted reliability action             | 完整 MMFR                                          |
  | Constant / Shuffle / Inverse reliability | 验证 reliability 信息与方向是否真正有效            |

  ------

  # 八、B1 冻结前的核心检查项

  -  B1 不直接修复 raw Depth。
  -  B1 不只是 $r_D\odot F_D$。
  -  B1 不只是 reliability-weighted generic fusion。
  -  B1 不只是 reliability-scaled generic adapter。
  -  B1 不只是 semantic prototype / RGB-D disagreement 产生的 reliability。
  -  B1 不只是 feature denoising / purification。
  -  Action 能明确定位到 DFormerv2 的 Depth-derived geometry-prior / geometry interaction 路径。
  -  Oracle $Q_D$ 控制 action 时已经优于 no-action。
  -  Predicted reliability 能复现较大部分 oracle gain。
  -  Constant、Shuffle、Inverse 无法复制正确 reliability 的收益。
  -  有 parameter-matched generic gate / adapter 对照。
  -  有相同 corruption exposure 的 augmentation-only 对照。
  -  clean performance 单独报告。
  -  单 checkpoint 覆盖主要 corruption × severity。
  -  保留 held-out corruption。
  -  misalignment 单独报告。
  -  验证 $Q_D$ / $\hat r_D$ 与 $U_D^{\mathrm{geo}}$ 的关系。
  -  验证相同 corruption severity 下可能存在不同 geometry utility，避免模型仅学习 corruption 类型或强度。

  ------

  # 九、当前禁止使用的宽泛 Novelty Claim

  后续论文中原则上避免以下表述：

  > ❌ 首次研究低质量 Depth 对 RGB-D 模型的影响。

  > ❌ 首次学习 Depth reliability。

  > ❌ 首次根据 reliability 动态降低低质量模态贡献。

  > ❌ 首次提出 uncertainty/confidence-aware multimodal fusion。

  > ❌ 首次提出 pixel-wise / spatial modality reliability。

  > ❌ 首次利用局部 reliability 进行 adaptive fusion。

  > ❌ 首次利用局部 reliability 调节 adapter。

  > ❌ 首次统一 missing 与 noisy modality。

  > ❌ 首次通过训练阶段模拟 modality missing 提高鲁棒性。

  > ❌ 首次进行 task-driven degraded Depth correction。

  > ❌ 首次不修输入而在 feature space 抑制 corruption。

  > ❌ 首次利用 quality 进行 expert routing。

  > ❌ 首次提出 diagnosis–restoration–reassessment–fusion 闭环。

  > ❌ 首次提高 DFormer 在 Depth missing 条件下的鲁棒性。

  > ❌ 首次证明增加低质量模态可能降低多模态性能。

  > ❌ 首次验证 reliability 与任务 performance / utility 有关。

  ------

  # 十、当前相对安全的 MMFR 定位

  现阶段更合适的定位为：

  > **MMFR 不把模态可靠性视为普通 fusion weight，而关注受控 Depth corruption 后 observation 的最终几何可信状态，并研究这一状态如何影响 DFormerv2 中由 Depth 驱动的 geometry-prior computation。进一步通过 oracle、predicted、constant、shuffle、inverse 和 geometry-specific utility 等对照，将“Depth 是否被正确诊断”与“对应 geometry action 是否真正有效”分开验证。**

  可进一步压缩为：

  **Quality → Diagnosis → Geometry-specific Action → Utility**

  当前真正需要保护的是以下组合，而不是任何单一组件：

  1. **corruption-grounded Depth final-state reliability；**
  2. **DFormerv2 geometry-prior-specific intervention；**
  3. **reliability-to-geometry-utility calibration；**
  4. **diagnosis / action 分离的 counterfactual validation。**

  ------

  # 十一、全文审计状态修正

  原清单中的“需要提供/下载全文”状态可更新如下：

  ### P0

  AI019、AI020、AI021、PR089、AI023、AI025：

  **均已完成全文审计。**

  ### P1

  AI022、ANGA、AI017、AI026、AI016、PR090、AI024、PR029：

  **均已完成全文审计。**

  其中 ANGA 暂继续保留：

  > **未单独编 AI 编号，待正式 DOI / 预印本标识最终核定。**

  ### P2

  AI018、AI027、AI028、MoSA：

  **均已完成全文审计。**

  其中 MoSA 应从原来的普通 P2 哨兵提升为：

  > **P2-H / 近 P0 警戒**

  因为其已经直接覆盖 RGB-D semantic segmentation、spatial reliability、spatial adaptive adapter 与 reliability-guided fusion。

  ------

  # 十二、AI 编号参考文献

  - **[AI015]** *Incomplete RGB-D Salient Object Detection: Conceal, Correlate and Fuse* — DOI: `10.1016/j.patcog.2024.110700`
  - **[AI016]** *Rethinking RGB-D Salient Object Detection: Models, Data Sets, and Large-Scale Benchmarks* — DOI: `10.1109/TNNLS.2020.2996406`
  - **[AI017]** *Calibrated RGB-D Salient Object Detection* — DOI: `10.1109/CVPR46437.2021.00935`
  - **[AI018]** *Uncertainty Inspired RGB-D Saliency Detection* — DOI: `10.1109/TPAMI.2021.3073564`
  - **[AI019]** *MaskMentor: Unlocking the Potential of Masked Self-Teaching for Missing Modality RGB-D Semantic Segmentation* — DOI: `10.1145/3664647.3681698`
  - **[AI020]** *A Conflict-Guided Evidential Multimodal Fusion for Semantic Segmentation* — DOI: `10.1109/WACV61041.2025.00141`
  - **[AI021]** *Provable Dynamic Fusion for Low-Quality Multimodal Data* — DOI: `10.48550/arXiv.2306.02050`
  - **[AI022]** *Centering the Value of Every Modality: Towards Efficient and Resilient Modality-Agnostic Semantic Segmentation* — DOI: `10.1007/978-3-031-72890-7_12`
  - **[AI023]** *GeomPrompt: Geometric Prompt Learning for RGB-D Semantic Segmentation Under Missing and Degraded Depth* — DOI: `10.48550/arXiv.2604.11585`
  - **[AI024]** *Toward Reliable RGB-D Semantic Segmentation: Handling Missing Modalities via Condition Dropout* — DOI: `10.48550/arXiv.2607.20326`
  - **[AI025]** *When Fusion Fails: Corruption-Aware Rebalanced Fusion for Multi-Modal Medical Image Segmentation* — DOI: `10.1145/3767308.3836234`
  - **[AI026]** *SimMLM: A Simple Framework for Multi-Modal Learning with Missing Modality* — DOI: `10.1109/ICCV51701.2025.02231`
  - **[AI027]** *Addressing Missing and Noisy Modalities in One Solution: Unified Modality-Quality Framework for Low-Quality Multimodal Data* — DOI: `10.48550/arXiv.2603.02695`
  - **[AI028]** *Adaptive Modality Reliability Diagnosis and Restoration for Robust Multimodal Intent Recognition* — DOI: `10.48550/arXiv.2608.03475`
  - **[MoSA]** *Modality-Aware Spatially-Adaptive Adaptation for RGB-X Semantic Segmentation* — DOI: `10.1109/ACCESS.2026.3694496`

  另外长期保留：

  - **[PR029] UMFNet** — pixel-wise uncertainty / confidence 与局部 multimodal modulation 的核心近邻；
  - **[PR089] SGMA** — spatial robustness map 与 reliability-weighted adaptive fusion 的核心近邻；
  - **[PR090] RobustSeg** — missing/noisy multimodal semantic segmentation 与 dominant/non-dominant modality handling 的核心近邻；
  - **ANGA** — *Anchor-Guided Gradient Alignment for Incomplete Multimodal Learning*，当前保留待正式编号状态。

  ------

  # 十三、当前最终判定

  基于已补全的 P0、P1、P2 文献，**MMFR 主线目前仍可继续，但研究空间已经明显收窄。**

  当前不应再围绕：

  **reliability estimation、adaptive fusion、pixel-wise gating、missing robustness、Depth repair、feature suppression、generic adapter 或 quality-aware routing**

  建立主要 novelty。

  当前仍未在这批先例中发现完整同构于以下组合的工作：

  **Depth final-state reliability + DFormerv2 geometry-prior failure mechanism + geometry-specific marginal utility + geometry-specific intervention + counterfactual action validation**

  因此 B1 下一阶段最重要的判断标准仍然是：

  > **先用 oracle $Q_D$ 证明 geometry-specific action 本身正确，再接 predicted reliability；如果最终结构退化为 raw-Depth repair、generic feature gate、generic spatial adapter 或普通 reliability-weighted fusion，则应重新判为高撞车风险。**
