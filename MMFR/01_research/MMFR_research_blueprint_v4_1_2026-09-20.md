# MMFR v4.1 研究蓝图
## Oracle-A 后继执行修订｜R/F/T 全文复核与 E1 分批筛选

> **文档角色：** MMFR 方向总计划的 Oracle-A 后继执行修订；不承担项目实时状态或执行授权。  
> **计划状态：** 全文硬门禁待完成；当前只允许定向全文复核与 E1 子计划编写。  
> **形成或核验时点：** 2026-09-20。  
> **实时入口：** [`MUSeg-current-status.md`](../../doc/main/MUSeg-current-status.md)。  
> **研究选择：** [`MUSeg-open-decisions.md`](../../doc/main/MUSeg-open-decisions.md)。  
> **后继关系：** 本修订以后继报告 [`MMFR-Oracle-A`](../../doc/reports/2026-09-19-museg-mmfr-oracle-a-validity-aware-pairwise-geometry.md) 的 `NO-GO` 为依据，覆盖本文旧版中“G 进入首轮 E1、六组同时筛选、G 可直接胜出”等冲突执行文字；历史章节、论文编号、参考索引与来源编号继续保留。若本计划与实时入口冲突，以实时入口为准。
>
> 基线文档为 S5《MMFR v4.0 研究蓝图》；研究对象仍为 MUSeg + DFormerv2-S + HAM。  
> 本次只修订研究顺序、证据门禁和 G 的处置，没有修改代码、protocol（实验协议）、索引注册表或 `../03_reference/source_materials/`，也没有训练、GPU/云任务、checkpoint 效果评价或 official test 读取。  
> 原 v4.1 新增的 34 个实施步骤溯源、6 个读文献包、135 条论文总索引、13 项冲突说明和 26 组网络来源记录继续有效；135 条仍只是可定位条目，不等于已逐篇全文复核。  
> S1–S4、W1–W3 保持 v4 的编号；既有 PR/RE/AI 不重编号，MoSA/ANGA 也保留原代号。

### 使用入口

正文保留 v4 的 0–13 节，并在相关操作旁补入来源。完整题名、DOI/标识、原文件行号、状态与别名见 **[MMFR_reference_index_v4_1_2026-09-20.md](../03_reference/MMFR_reference_index_v4_1_2026-09-20.md)**；机器可读对应关系见 **[MMFR_reference_registry_v4_1_2026-09-20.json](../03_reference/MMFR_reference_registry_v4_1_2026-09-20.json)**。整包中的 `../03_reference/source_materials/` 保存本轮14份依据的原样副本，SHA-256 可核对。

### 四种证据不能混写

| 标记 | 指什么 | 可以支持什么 |
| --- | --- | --- |
| [项目] | S6 实验报告、S1 项目账本 | 已记录结果/历史协议；不等于本轮重跑 |
| [资料] | PR/RE 条目与 P0/P1/P2 审计 | 你提供的文献信息；“原审计已核”不升级为本轮重读全文 |
| [外核] | W1–W26 的原始页面、源码或检索入口 | 只支持记录中写明的核验层级；标题可见不等于方法全文可见 |
| [设计] | v4/v4.1 的候选、对照、公式和参数 | 待验证提案；不可冒充原论文默认配置或预期结果 |

**执行优先级：** v4/v4.1 决定采用路线；历史清单保留“已有先例”和机制信息，不恢复“不得采用修复/适配/门控”等旧限制。数据、权重、代码的实际运行身份最终由本地 manifest 决定。

### 进入子计划前的全文硬门禁

当前不再按“候选近邻尽可能多读”启动实现，而是先取得并结构化复核以下 **五篇核心全文**。五篇必须全部完成，才能冻结 E1；仅定位条目、摘要、搜索片段、既有审计或网页方法段落均不能替代本轮全文复核。

| 路线 | 论文与固定版本 | 本轮用途 | 必须提取的实施字段 |
| --- | --- | --- | --- |
| R | **AI023 — GeomPrompt / GeomPrompt-Recovery，arXiv:2604.11585** | 输入补偿、RGB 几何提示、受损 Depth 的任务驱动残差修正 | 两种输入模式；补偿器/提示器结构与插入点；输出量纲和范围；有效 Depth 的保护方式；missing/degraded 的区分；loss、正则和权重；初始化；冻结/训练参数；optimizer 参数组、学习率、训练长度；推理所需信息；官方代码/版本 |
| F | **AI024 — Condition Dropout，arXiv:2607.20326v1** | DFormer 缺失模态 continued training 与残差适应的首要工程参考 | DFormer 版本；复制/冻结哪些模块；注入层与张量形状；condition 编码；missing 输入表示；零初始化；训练状态采样；loss；参数组和学习率；训练长度；checkpoint/selector；推理路径与 full-modality 保护 |
| T | **AI019 — MaskMentor，DOI 10.1145/3664647.3681698** | complete teacher 到 missing student 的最小知识迁移参考 | teacher/student 来源；遮罩/缺失生成；监督和蒸馏位置；loss 与权重；teacher 更新/冻结；视图与学生更新次数；初始化；训练长度；推理是否仅保留 student；对照与消融 |
| T | **PR090 — RobustSeg，官方 CVPR 2026 固定版本** | 完整教师、受损学生和蒸馏机制的语义分割直接近邻 | 正式版与作者预印本差异；最终采用的模块名和结构；teacher/student 身份；缺失/噪声条件；输出/特征/原型蒸馏；loss 权重；训练安排；推理信息；matched baseline；计算成本。**DOI 待核，禁止猜造** |
| R/F/T 共同骨干 | **PR070 — DFormerv2，DOI 10.1109/CVPR52734.2025.01802** | 冻结本地可实现的输入、特征和几何接口 | 论文正式算子与官方代码差异；本地 fork 对应函数；四级特征形状；Depth 唯一消费路径；插值/归一化；可插入点；参数初始化；optimizer 是否覆盖新增/既有参数；训练与推理契约 |

每篇全文的结构化摘录都必须记录：固定版本与来源、原文章节/公式/图表、官方代码入口与 commit（如有）、原论文做法、可直接借用部分、与 DFormerv2-S + A2 的不兼容点、本项目适配、仍属 `[设计]` 的字段和待核项。PR090 必须先消解 CVPR 正式版与相关作者预印本的差异，不能拼接多个版本的模块或超参数。

**硬门禁：** 五篇全文全部取得并形成 `../03_reference/literature_fulltext_audit.md` 前，不得写实现代码、运行新实验或冻结 E1；全文复核完成后，还必须在 `e1_screening_plan.md` 中分别冻结 R/F/T 的 minimum viable implementation（最小可行实现）、接口、loss、初始化、训练/冻结参数、optimizer 参数组、训练/推理可用信息、matched control（匹配对照）、预算、checkpoint 规则、promotion/stop 条件，之后才可申请执行授权。

**条件补充，不是当前五篇门禁的一部分：**

- **AI017 — Calibrated RGB-D Salient Object Detection，DOI 10.1109/CVPR46437.2021.00935**：只有 R-lite 实际采用 RGB 估计 Depth 与原始 Depth 的质量条件校正/连续插值机制时，才升级为 R 路线阻塞项。
- **MoSA — Modality-Aware Spatially-Adaptive Adaptation，附件记录 DOI 10.1109/ACCESS.2026.3694496，题名—DOI 对应仍待核**：只有 F-lite 实际采用其空间自适应调制机制时，才升级为 F 路线阻塞项。

其余索引继续用于查重、近邻定位和后续扩展，不构成本轮大规模文献调研。全文门禁不是要求忠实复现五篇论文，而是为了在实施前把“原论文做法 / 本项目适配 / 新建议”分开。

---


## 0. 本次修订的结论

### 0.1 Oracle-A 后的执行裁决

`MMFR-Oracle-A` 已完成 318 条 `val-dev`、3 个 condition、4 个 variant 的 inference-only（只推理）配对评价，裁决为 **`NO-GO`**。关键事实为：strict 在 `clean` 上相对 original 为 `-3.48` mIoU，在 `spatial_dropout@0.75` 上为 `-1.77`；该条件的 location-group 配对 bootstrap 区间为 `[-0.503, -0.158]`；strict 相对 `geometry_off` 只高 `+0.04`；`entire_missing@1.0` 下 strict 与 `geometry_off` 均为 `52.61`，逐位等价。权威细节见后继报告，不从本蓝图摘要反推更多结论。

因此当前执行主线固定为：

$$
\boxed{R/F \rightarrow T \rightarrow \text{必要时测试 }T+R\text{ 或 }T+F}
$$

G 中的具体动作族 **`validity-aware geometry suppression`（按 validity 降低或关闭不可信 Depth geometry）正式冻结**，不进入首轮 E1，也不训练 predicted reliability gate 去抢救同一个动作。这个裁决只说明 suppression 动作不值得继续投入；它不说明 reliability 诊断无用，不说明 Depth geometry 不重要，也不关闭 R 的输入补偿、F 的特征适配、T 的训练迁移。`geometry_off` 在 clean 上从 `57.06` 降到 `52.61`，反而说明正常条件下 Depth geometry 对任务具有显著价值；平均 attention 变化小不能推出任务效应小。

G 只可在以下条件全部满足时重开：R/F/T 均未显示实用收益，或出现一个在机制上不再等价于 geometry suppression 的新假设；为其建立新的 protocol identity 和独立 go/no-go；取得用户明确授权。历史 G 设计与编号继续保留用于解释 Oracle-A 和避免重复试验，不再构成当前候选合同。

### 0.2 v4/v4.1 原始原则的保留边界

用户关于“防撞车压缩了有效设计空间”的批评成立，但应把两个问题分开：

**可以采用什么方法**，主要由技术适配性、可复现性、性能收益和成本决定。原始深度修复、空间门控、蒸馏、特征适配、缺失模态训练、任务驱动几何提示都可以采用。

**可以把什么写成原创贡献**，取决于相对先例究竟新增了什么、为什么有必要以及证据是否充分。更换名称或故事不会改变方法实质；更换场景或骨干可以形成有价值的应用研究，但不自动等于强算法创新。

旧 P0 报告曾把 raw-Depth residual repair 直接列为“B1 禁区”；P2 又把 generic fusion/adaptation 写成必须避开的方案。这把“不能单独声称首创”扩大成了“不能使用”，本版撤销这种推导。[S2，§五；S4，§三十四]

同时，v3 并非完全禁止借鉴：它已经安排通用适配器对照、匹配短训练，并说明 Oracle 失败不能否定整个方法家族。这些合理部分保留；需要改变的是让几何局部干预成为排他主线的优先级，而不是把 v3 的所有内容推倒。[S1，§S2.2、S2.6、S2.7]

**新版执行主线：**

> 保留可信实验底盘 → 复用并筛选有希望的成熟方法 → 找到强性能版本 → 围绕真实短板做针对性改造 → 精简无效组件 → 公平比较、独立确认与贡献界定。

不再要求先获得完整机制证明，才允许开展小规模学习式实验；也不把所有机制分析都推迟到最后。开始前写清工作假设，实验中记录失败，入围后完成与论文主张相匹配的机制验证。

---

## 1. 哪些规定撤销，哪些规定保留

| v3/P0–P2 中的限制倾向 | v4 的处理 |
| --- | --- |
| raw Depth 修复与 GeomPrompt 接近，因此不能作为主方法 | 可以复现、移植和改造；直接承认来源，比较朴素移植与针对性版本 |
| 空间门控、adapter 已有先例，因此只能当陪衬 | 可以成为最终系统的主体；是否承担算法原创贡献另行判断 |
| entire missing 已有很多论文，因此不应优先开发 | 按本项目实际损失重新排序；目前它是优先性能目标 |
| 必须独立估计合成质量，再用该质量驱动动作 | 允许质量监督、任务监督及混合监督竞争，不预定哪种最优 |
| A2 已有质量头，B1 必须把它接进几何计算 | 可复用，也可替换、降级为辅助监督或删掉 |
| 零训练 Oracle 没涨点，学习式路线就很难继续 | Oracle 只裁决被直接检验的动作：Oracle-A 已关闭 validity-aware geometry suppression，但不关闭 R/F/T 的有限、匹配学习式验证，也不否定 Depth geometry 本身 |
| 永久采用 clean-only checkpoint selector | 历史记录不变；新实验可事前登记“clean 约束下的鲁棒验证选择” |
| 只有 geometry-specific 路线才值得写 | 允许输入补偿、特征适应、训练策略和系统组合；贡献随证据确定 |

以下规则不放松：官方 test 不用于开发；不把额外训练收益全算给模块；不把简化移植叫作原论文忠实复现；不把合成故障当作真实矿下故障分布；不隐瞒 clean 或个别条件退化。

---

## 2. 从已有数值重新判断值得投入的方向

### 2.1 已知开发参照

> **v4.1 一级项目溯源：** 本节数值现在直接定位到 S6《A2实验结果报告》§1、§5（24–40、180–199），而非只经 S1 转述。计算差距属于由这些数值进行的情景算术，不是论文预测。


以下数值取自 v3 对 A2 epoch-420 的记录，不是本次重新测量。[S1，第一部分 §2]

| 条件 | A2 mIoU | 相对 A2 clean 的下降 |
| --- | ---: | ---: |
| clean | 57.06 | — |
| entire_missing@1.0 | 52.55 | 4.51 |
| spatial_dropout@0.75 | 54.42 | 2.64 |
| misalignment@0.75 | 55.80 | 1.26 |
| gaussian_noise@0.75 | 56.48 | 0.58 |
| quantization@0.75 | 56.99 | 0.07 |
| blur@0.75 | 57.06 | 0.00 |

六个单故障条件的宏平均为：

$$
M_6=\frac{1}{6}\sum_{c\in\mathcal C_{\mathrm{single}}}\operatorname{mIoU}_c=55.55.
$$

仅作机会判断：假设一个改动只把 spatial dropout 和 misalignment 恢复到当前 clean 的 57.06，其他条件不变，则：

$$
\Delta M_6=\frac{2.64+1.26}{6}=0.65.
$$

而仅把 entire missing 恢复到同一个参考点，对主平均的贡献为：

$$
\Delta M_6=\frac{4.51}{6}\approx0.75.
$$

把全部单故障都恢复到当前 clean，对应平均差距为 $57.06-55.55=1.51$ 个百分点。

**这不是任何方法的理论上限。** 新训练、新表征和补偿机制可能同时提高 clean 与失效性能，也不保证能恢复上述全部差距。这组算术只是说明：若主要投入“抑制局部坏几何”，却不开发“Depth 缺失后如何继续获得有效表示”，会错过当前最大的一项缺口。

### 2.2 clean 本身也需要恢复或提升

> **项目溯源：** S6 §2–3（50–114）明确 B0 是历史 RGB-D 参照而非 paired clean-control；不能把 1.73 差距归因于某一个训练变化。


历史 B0 clean 为 58.79，A2 为 57.06，差 1.73 个百分点。但 B0 不是正式匹配的 clean-control，不能直接认定差距由 corruption training 引起，也不能把 58.79 当成新方法必然达到的数值。[S1，第一部分 §2.2]

新版同时追求：

- 不可靠 Depth 下的绝对分割性能更高；
- 可靠 Depth 下的性能不被长期牺牲；
- 整模态缺失时有可学习的补偿或退路，而不仅是继续降权。

暂不为了“更难”而修改旧 blur/quantization 的测试强度。扩展故障协议可以另行登记，不能替换对本方法不利或不显著的旧结果。

---

## 3. 把“防撞清单”改成“借鉴与贡献清单”

### 3.1 借鉴优先级

| 方法来源 | 值得借用的部分 | 在本项目中的角色 |
| --- | --- | --- |
| MaskMentor [AI019]、RobustSeg [PR090] | 完整输入指导受损输入、蒸馏、缺失暴露 | 优先研究的训练策略；不能假定具体损失可直接照搬 |
| GeomPrompt / Recovery [AI023]、Calibrated RGB-D [AI017] | 任务驱动几何补偿、输入校正、原始与替代证据的使用 | 原始 Depth 修复/提示路线的主要来源 |
| Condition Dropout [AI024] | 保留原模型、学习残差适应、缺失输入继续训练 | 无深度运行与特征补偿的主要来源 |
| MoSA、UMFNet [PR029]、SGMA [PR089] | 空间条件化、任务相关可靠性、局部适配 | 门控与轻量 adapter 的主要来源 |
| ECoLaF [AI020]、QMF [AI021] | 冲突、不确定性与质量信号的不同定义 | 信号对照和设计参考，不要求全部实现 |
| CoReFuse-Med [AI025] | 特征抑噪与重校准 | 第一轮有明确未解决问题时再引入 |
| UMQ [AI027]、PRIME [AI028] | 质量增强、恢复后再评估、条件路由 | 保留候选，不因“闭环已有”而禁用，也不为拼故事强行加入 |

上述方法角色主要依据附件报告。[S2，§二、§十；S3，§二；S4，§一]

外部核验支持两点：GeomPrompt/Recovery 确实以分割任务监督学习几何提示或修正；Condition Dropout 确实采用冻结原模型与训练附加残差路径的思路。它们值得作为设计来源，但这些论文不证明同样收益会出现在 MUSeg + DFormerv2-S 上。[W2][W3]

### 3.2 每个采用组件只需要回答四件事

1. 原文做了什么，本项目借用了什么。
2. 在当前骨干和任务上改了什么。
3. 改动是否超过朴素移植，或者以更低成本取得相近收益。
4. 论文中哪些归于先例，哪些是本项目有证据支持的增量。

不再给方法贴“相似，所以禁止”的标签。按需使用“忠实复现”“骨干适配”“受其启发的简化对照”三个标签，不能混写。

最接近的方法往往应该成为优先参考和正面对照，但“接近”本身也不保证它最容易复现或最适合当前结构。

---

## 4. Oracle-A 后的候选体系：R/F 第一优先，T 第二优先，G 冻结

### 4.1 总体建议

当前更值得优先尝试的组合假设是：

> **先验证一条在 Depth 缺失时仍能增加 fallback/substitute capability（退路/替代能力）的结构路径，再验证完整输入对受损输入的知识迁移是否带来额外净收益。**

当前活动候选只有三个：

- **R：输入或几何提示补偿**——直接修正 Depth 输入或生成任务有用的替代几何；必须区分 valid、natural-invalid（自然无效）与 synthetic-missing（合成缺失），不得默认覆盖 natural-invalid。
- **F：特征补偿**——在现有 RGB 主特征上学习残差适应；第一轮不接 A2 reliability predictor。
- **T：训练策略**——完整/受损视图联合监督与最小输出级蒸馏；用 `Cpair` 分离成对视图暴露与蒸馏净收益。

**G：几何关系使用控制**保留为历史/条件重启代号，不是当前活动候选。Oracle-A 已关闭“validity-aware geometry suppression”这一具体动作；不得把它重新包装成 learned gate 后放回首轮。

这些字母是本项目实验代号，不是提前命名的原创算法。E1 必须先做 Batch 1 的 `C0/R/F`，形成可核验证据后才做 Batch 2 的 `Cpair/T`；不得同时启动两批。只有单候选证据完成后才讨论 `T+R` 或 `T+F`，当前不设计或训练 `T+R+F`。

### 4.2 为什么不能只保留“抑制”

> **骨干依据：** [PR070; W1]。S7 的论文整理与当前公开代码有实现表述差异，详见 K02；本节只作固定算子下的推理，不宣称已核过你的本地训练仓库。


DFormerv2 官方实现以深度差构造几何关系偏置，并在注意力 softmax 前加入。它不等同于拥有一个可以任意加权的对称 Depth 语义分支。[W1]

如果整幅缺失在实际几何入口表现为常数深度，则端点深度差为零。仅把已经为零的深度偏置继续乘小，不能凭空恢复信息。这里说的是固定参数、固定输入下的这一条操作；训练改变表征，或引入新补偿路径，是不同的问题。

因此，需要让候选至少包含一种“没有 Depth 时仍有能力改变有效表示”的机制：RGB 生成的任务几何提示、RGB 特征残差适应，或不依赖缺失 Depth 的训练收益。不能把 52.55 视为无深度条件下的上限。

---

## 5. 三个活动候选与一个冻结分支的落地边界

### 5.1 R：任务驱动的输入补偿——解除禁区后的优先候选

**本节溯源与实施编号（v4.1 补充）**

| 步骤 | 参考论文及其用途 | 参数来源与引用边界 |
| --- | --- | --- |
| **R.1** 朴素任务驱动 Depth 补偿或 RGB 几何提示 | [AI023](../03_reference/MMFR_reference_index_v4_1_2026-09-20.md#ref-AI023)、[AI017](../03_reference/MMFR_reference_index_v4_1_2026-09-20.md#ref-AI017)、[PR070](../03_reference/MMFR_reference_index_v4_1_2026-09-20.md#ref-PR070) | 残差接口与分割监督为 v4 候选；输出范围、网络宽度和归一化待本地确定。 GeomPrompt 与 Recovery 输入不同；本文式子是适配示意，非原论文逐式复现。 |
| **R.2** 可靠区域保持、局部缺失与整幅缺失的补偿幅度 | [AI023](../03_reference/MMFR_reference_index_v4_1_2026-09-20.md#ref-AI023)、[AI017](../03_reference/MMFR_reference_index_v4_1_2026-09-20.md#ref-AI017)、[AI016](../03_reference/MMFR_reference_index_v4_1_2026-09-20.md#ref-AI016) | 幅度上限、valid 区辅助约束及缺失模式处理是本项目待验证选择。 不能照搬其他量纲的 0–255，不能将替代几何记为真实传感观测。 |
| **R.3** 用分割结果和关系变化评估补偿而非只看像素误差 | [PR070](../03_reference/MMFR_reference_index_v4_1_2026-09-20.md#ref-PR070)、[AI023](../03_reference/MMFR_reference_index_v4_1_2026-09-20.md#ref-AI023)、[RE266](../03_reference/MMFR_reference_index_v4_1_2026-09-20.md#ref-RE266)、[RE043](../03_reference/MMFR_reference_index_v4_1_2026-09-20.md#ref-RE043)、[RE049](../03_reference/MMFR_reference_index_v4_1_2026-09-20.md#ref-RE049) | 区域/边界定义、关系抽样和辅助 loss 权重是研发建议，不是以上论文的统一配置。 跨任务矿下增强仅支撑任务评价动机；不声称其结果可直接转移。 |

定位：S2 §二（55–76）、§五；S3 §二（38–69）；S5 §5.1（163–187）；S5 §5.1（175–186）；S6 §7（252–280）；S1 §M1（639–670）；S9/S10 按对应编号；S5 §6、§10。网络原始入口：W2、W9、W1、W4（见末尾网络索引）。
以下操作是 v4 的项目方案；相邻论文支持借鉴方向，不代表每项操作或公式都原封不动来自某篇论文。


**大白话：** 不只是告诉模型“这张深度不好”，还尝试给它一份更适合完成分割的几何输入。

从简单残差形式开始：

$$
\widetilde D=D_c+\Delta_\theta(I,D_c,V),
$$

其中 $I$ 为 RGB，$D_c$ 为实际 Depth，$V$ 为部署时可观察的有效性信息。输出边界、归一化和通道数必须按本地输入契约处理，不能直接照搬其他论文中的 0–255 范围。

第一版优先使用分割损失；不强求恢复真实米制深度。需要训练参考 Depth 的辅助损失时，只在定义正确、有效且坐标对应的区域使用，并另做消融。原始参考 Depth 不能进入普通测试前向。

本项目值得检查的针对性改造包括：

- 有效 Depth 是否被不必要地改坏，能否保留其作用；
- 局部缺失与整幅缺失是否需要不同的修正幅度；
- 输入插值与 validity 是否共同制造虚假中间值；
- 修正是否真正有利于 DFormerv2 的成对关系，而不仅是像素误差变小。

这不是要求一次实现四个组件，而是朴素 R 之后的改造菜单。尤其不应使用只能做很小修正的输出范围，再据此宣布“全缺失补偿无效”。

**命名边界：** 简化网络并适配到 DFormerv2 时，写成 GeomPrompt-inspired / task-driven input repair adaptation；未经忠实实现不能写“复现了原 GeomPrompt”。[S2，§二、§五；W2]

### 5.2 F：RGB 主路径的轻量特征补偿

**本节溯源与实施编号（v4.1 补充）**

| 步骤 | 参考论文及其用途 | 参数来源与引用边界 |
| --- | --- | --- |
| **F.1** 在 RGB 主特征中加入残差适配 | [AI024](../03_reference/MMFR_reference_index_v4_1_2026-09-20.md#ref-AI024)、[MoSA](../03_reference/MMFR_reference_index_v4_1_2026-09-20.md#ref-MoSA)、[AI009](../03_reference/MMFR_reference_index_v4_1_2026-09-20.md#ref-AI009)、[PR070](../03_reference/MMFR_reference_index_v4_1_2026-09-20.md#ref-PR070) | 小型 adapter 的宽度、插入 stage 与共享方式属于本项目简化方案。 ConD 原始复制编码器不是小 adapter；简化应标为 inspired/适配，不标忠实复现。 |
| **F.2** 身份初始化、梯度启动与冻结策略核验 | [AI024](../03_reference/MMFR_reference_index_v4_1_2026-09-20.md#ref-AI024)、[PR070](../03_reference/MMFR_reference_index_v4_1_2026-09-20.md#ref-PR070) | 零输出投影是候选初始化；本地测试 logits、梯度与一步 optimizer update。 初始化近似/精确 identity 必须实测；不声称训练后 clean 必然保持。 |
| **F.3** 质量条件、抑噪或专家扩展与无条件版本竞争 | [MoSA](../03_reference/MMFR_reference_index_v4_1_2026-09-20.md#ref-MoSA)、[PR029](../03_reference/MMFR_reference_index_v4_1_2026-09-20.md#ref-PR029)、[PR089](../03_reference/MMFR_reference_index_v4_1_2026-09-20.md#ref-PR089)、[AI025](../03_reference/MMFR_reference_index_v4_1_2026-09-20.md#ref-AI025)、[AI026](../03_reference/MMFR_reference_index_v4_1_2026-09-20.md#ref-AI026)、[PR038](../03_reference/MMFR_reference_index_v4_1_2026-09-20.md#ref-PR038)、[PR165](../03_reference/MMFR_reference_index_v4_1_2026-09-20.md#ref-PR165) | 只在 F 朴素版不足时单变量扩展；空间 gate、专家数、插入层不预先照搬。 先保留无质量头 F；不同任务的 routing/抑噪不等于可直接拼接。 |

定位：S3 §二（38–69）及 ConD 段落；S4 §九（380 起）；S5 §5.2（188–207）；S5 §5.2（188–207）、§E1（291–311）；S2 §二；S3 §二；S4 §九及 MoSA 段落；S8 PR038/PR165。网络原始入口：W3、W10、W1、W4、W11、W12、W17、W16（见末尾网络索引）。
以下操作是 v4 的项目方案；相邻论文支持借鉴方向，不代表每项操作或公式都原封不动来自某篇论文。


**大白话：** Depth 不可用时，让原模型多一小段能够处理这种情况的能力，而不是只能关闭原有信息。

示意形式：

$$
\widetilde F^\ell
=
F^\ell+\operatorname{Adapter}_\ell(F^\ell,V^\ell).
$$

先用小型残差适配器，插入点与张量布局按本地实现确定。不要假定存在独立 Depth encoder。参考 Condition Dropout 的残差注入思想与 MoSA 的空间适配，但不必一开始复制整个编码器。[S4，§九；W3]

采用输出投影零初始化等方式让新增路径从不扰动原模型开始；这是初始化性质，不保证训练后 clean 自动保持。避免把残差输出和外部乘法系数同时设为零而让新增路径无法启动学习。

第一轮不强制 reliability gate。若不带质量头的 F 已经有效，先保留；质量头能否增加收益另行验证。

若小适配器表达能力不足，允许后续评估更大的补偿分支，但必须报告参数、延迟和训练成本。轻量是优先策略，不是凌驾于性能之上的教条。

### 5.3 G：Oracle-A 后冻结的历史分支

本节保留 G 的历史编号，但其当前状态为 **暂停，不进入 E1**。Oracle-A 已直接检验以下具体动作：在 DFormerv2 的位置项保持不变时，用真实 validity 对 `weight[1] * depth geometry` 做 strict 或 continuous pairwise suppression。结果在 `clean` 和 `spatial_dropout@0.75` 上均为负，且 spatial strict 与 `geometry_off` 仅差 `+0.04` mIoU。

因此以下旧提案不再执行：

- 不把 validity、A2 预测可靠性或更复杂任务 gate 接到同一个 suppression 动作；
- 不为 G 安排 matched short training；
- 不把 G 放入 `C0/R/F` 或 `Cpair/T` 的首轮筛选；
- 不做“Oracle validity → predicted reliability → complex gate”的递进。

历史公式

$$
\widetilde B^\ell=B_S^\ell+a^\ell\odot B_D^\ell
$$

只作为已测试动作族的说明，不是当前待实现合同。Oracle-A 没有否定 Depth geometry：`geometry_off` 在 clean 上比 original 低 `4.45` mIoU；也没有否定 reliability 诊断、R/F/T 或与 suppression 机制不同的新假设。

G 只有在第 0.1 节的重启条件满足后才能另建 protocol 和 go/no-go。届时必须先证明新机制为什么不等价于“降低/关闭不可信 Depth geometry”，再由用户单独授权；不得在当前 E1 计划中预留隐式实现。

**非阻塞诊断边界：** Oracle-B、`weight[1]` geometry-weight 剂量曲线，以及 invalid sentinel / 输入表示诊断均只保留为后续研究候选，不是当前任务，当前不执行，也不阻塞 R/F/T 的全文复核和子计划冻结。若以后启动，必须另行定义问题、协议和授权；sentinel 不得因本次观察被直接改值。

### 5.4 T：完整与受损输入之间的训练知识迁移

**本节溯源与实施编号（v4.1 补充）**

| 步骤 | 参考论文及其用途 | 参数来源与引用边界 |
| --- | --- | --- |
| **T.1** 完整/受损成对视图与教师输出指导 | [AI019](../03_reference/MMFR_reference_index_v4_1_2026-09-20.md#ref-AI019)、[PR090](../03_reference/MMFR_reference_index_v4_1_2026-09-20.md#ref-PR090) | v4 的 clean CE + corrupt CE + KD 是候选组合；各权重未预填，须单独登记。 MaskMentor 的遮罩重建与 RobustSeg 的原型蒸馏不等同于该简化输出 KD。 |
| **T.2** 固定教师来源并防止无条件复制教师错误 | [AI019](../03_reference/MMFR_reference_index_v4_1_2026-09-20.md#ref-AI019)、[PR090](../03_reference/MMFR_reference_index_v4_1_2026-09-20.md#ref-PR090)、[AI022](../03_reference/MMFR_reference_index_v4_1_2026-09-20.md#ref-AI022) | 用 A2 clean 输出作为低成本教师、教师筛选与替换均为本项目研发建议。 教师未被证明最优；训练标签继续监督学生；换教师单列变化与成本。 |
| **T.3** 以 Cpair 分离多视图暴露和蒸馏收益 | [AI019](../03_reference/MMFR_reference_index_v4_1_2026-09-20.md#ref-AI019)、[PR090](../03_reference/MMFR_reference_index_v4_1_2026-09-20.md#ref-PR090)、[AI024](../03_reference/MMFR_reference_index_v4_1_2026-09-20.md#ref-AI024) | 成对视图与学生更新数匹配属于本项目对照；教师计算单列。 不能只因 epoch 相同就称算力相同；不把成熟论文当本项目两批五组 E1 设计的直接出处。 |
| **T.4** 按需扩展原型、采样或梯度优化 | [PR090](../03_reference/MMFR_reference_index_v4_1_2026-09-20.md#ref-PR090)、[PR089](../03_reference/MMFR_reference_index_v4_1_2026-09-20.md#ref-PR089)、[ANGA](../03_reference/MMFR_reference_index_v4_1_2026-09-20.md#ref-ANGA)、[AI026](../03_reference/MMFR_reference_index_v4_1_2026-09-20.md#ref-AI026)、[PR074](../03_reference/MMFR_reference_index_v4_1_2026-09-20.md#ref-PR074) | 扩展仅在基础 T 不足时设计；原型数、阈值、课程和 loss 需查固定版本原文。 不把全部近邻模块堆成默认网络；同作者多版本不可混合复现。 |

定位：S2 §二（55–76）；S3 §二（38–69）；S5 §5.4（230–260）；S5 §5.4（251–260）；S3 §二；S5 §5.4、§E1（230–260、291–311）；S1 §S2.7；S3 §二及 ANGA 段落；S2 SGMA 段落；S8 PR074。网络原始入口：W6、W7、W8、W15、W3、W12、W20、W16（见末尾网络索引）。
以下操作是 v4 的项目方案；相邻论文支持借鉴方向，不代表每项操作或公式都原封不动来自某篇论文。


**大白话：** 训练时让模型看同一场景的完整与受损版本，用完整版本能提供的知识帮助受损版本，但测试时只用实际输入。

可从以下目标开始：

$$
\mathcal L
=
\mathcal L_{\mathrm{seg}}^{\mathrm{corrupt}}
+\lambda_c\mathcal L_{\mathrm{seg}}^{\mathrm{clean}}
+\lambda_{\mathrm{KD}}
\mathcal L_{\mathrm{KD}}
\left(
\operatorname{stopgrad}(p_T(I,D_0)),
p_S(I,D_c)
\right).
$$

这是候选损失结构，不是原论文复现公式，也不预填未经本地核实的权重。

廉价筛选时可用同一个冻结 A2 checkpoint 在训练样本 clean 视图上的输出作为老师，避免一开始新增教师训练。其能力有限，不应称为最优教师。后续采用正式 clean 教师时，教师来源与训练成本必须一致、公开记录，并把换教师作为独立变化。

教师不是绝对正确：学生始终受训练标签监督；可用训练标签或教师置信筛除明显错误的蒸馏目标。是否加入特征蒸馏，在输出级方案表现不足后再判断。

**关键对照：** 不蒸馏的比较组也要看同样的 clean/corrupt 视图，拥有同样的学生更新次数。否则无法区分“多看了一遍数据”与“蒸馏有益”。教师前向的额外计算另报，不能说同 epoch 就等于同算力。

这里借鉴完整到缺失的教学思想，不声称自己首次使用该策略。[S2，§二；S3，§二]

---

## 6. 矿下与模型特化：从真实问题出发，不靠换名

**本节溯源与实施编号（v4.1 补充）**

| 步骤 | 参考论文及其用途 | 参数来源与引用边界 |
| --- | --- | --- |
| **S.1** 矿下场景动机与特化证据 | [RE326](../03_reference/MMFR_reference_index_v4_1_2026-09-20.md#ref-RE326)、[RE240](../03_reference/MMFR_reference_index_v4_1_2026-09-20.md#ref-RE240)、[RE026](../03_reference/MMFR_reference_index_v4_1_2026-09-20.md#ref-RE026)、[RE094](../03_reference/MMFR_reference_index_v4_1_2026-09-20.md#ref-RE094)、[RE049](../03_reference/MMFR_reference_index_v4_1_2026-09-20.md#ref-RE049)、[RE266](../03_reference/MMFR_reference_index_v4_1_2026-09-20.md#ref-RE266)、[RE418](../03_reference/MMFR_reference_index_v4_1_2026-09-20.md#ref-RE418)、[RE452](../03_reference/MMFR_reference_index_v4_1_2026-09-20.md#ref-RE452)、[RE360](../03_reference/MMFR_reference_index_v4_1_2026-09-20.md#ref-RE360) | 本项目类权重、区域与传感失效映射须由实际标签/样本验证。 露天矿、地表裂缝、SLAM、偏振、PPE检测仅相邻证据；不把它们写成MUSeg真实Depth失效标签。 |
| **E3.1** 朴素借用与场景/模型特化直接对比 | [PR070](../03_reference/MMFR_reference_index_v4_1_2026-09-20.md#ref-PR070)、[RE326](../03_reference/MMFR_reference_index_v4_1_2026-09-20.md#ref-RE326)、[AI023](../03_reference/MMFR_reference_index_v4_1_2026-09-20.md#ref-AI023)、[AI024](../03_reference/MMFR_reference_index_v4_1_2026-09-20.md#ref-AI024)、[MoSA](../03_reference/MMFR_reference_index_v4_1_2026-09-20.md#ref-MoSA) | 特化菜单与改造增量定义为本项目设计。 借鉴组件贡献与新增改造贡献分开；移到 backbone 内不自动增加新颖性。 |

定位：S9/S10 按编号；S12 §一–二（5–52）；S5 §6；S5 §6、§E3（261–277、322–337）。网络原始入口：W5、W1、W4、W2、W3、W10（见末尾网络索引）。
以下操作是 v4 的项目方案；相邻论文支持借鉴方向，不代表每项操作或公式都原封不动来自某篇论文。


值得形成贡献的“特化”应能落实到可复核的差异，例如：

**表示适配。** DFormerv2 如何使用 Depth，决定应该修正原始输入、关系偏置，还是中间 RGB 特征；必须与朴素移植比较，而不是把代码移进 backbone 就称新机制。

**失效适配。** 缺失没有观测，错位有观测但位置不对，噪声则可能仍含可用结构。允许处理方式不同，但不能把合成故障类别标签偷偷交给推理模型。

**尺度适配。** 处理 Depth 与 validity 在缩放后的语义，避免修复产生的新值被误标为真实传感器有效观测。

**任务适配。** 在真实标签空间中检查边界、稀少类和几何敏感区域，决定训练目标与特征设计。不能凭矿下想象新增“异物”类别或宣称某个类最重要。

**部署适配。** 同一 checkpoint 处理全部条件；不依赖原始 clean Depth、真实移位量或故障 severity；报告同设备下的运行成本。

这些只是可能产生实质差异的位置，不能保证任意一个微小修改都足够支撑强方法创新。矿下真实性也需要数据证据；人工 dropout、平移和噪声只能先作为受控压力测试。

---

## 7. 执行路线：小规模候选竞争，而不是一次堆满模块

### 阶段 E0：继承现状与建立新实验分支

**本节溯源与实施编号（v4.1 补充）**

| 步骤 | 参考论文及其用途 | 参数来源与引用边界 |
| --- | --- | --- |
| **E0.1** 冻结数据划分、样本/故障清单与历史结果 | [RE326](../03_reference/MMFR_reference_index_v4_1_2026-09-20.md#ref-RE326)、[AI010](../03_reference/MMFR_reference_index_v4_1_2026-09-20.md#ref-AI010)、[AI011](../03_reference/MMFR_reference_index_v4_1_2026-09-20.md#ref-AI011)、[PR040](../03_reference/MMFR_reference_index_v4_1_2026-09-20.md#ref-PR040) | 318 样本、40/40 评价、旧 10-condition 来自项目文件；哈希/白名单为工程约束。 数据集论文不提供本项目 A2 checkpoint 或主验证成绩。 |
| **E0.2** 核对几何入口、Depth/validity 缩放与无动作等价 | [PR070](../03_reference/MMFR_reference_index_v4_1_2026-09-20.md#ref-PR070)、[AI029](../03_reference/MMFR_reference_index_v4_1_2026-09-20.md#ref-AI029)、[AI030](../03_reference/MMFR_reference_index_v4_1_2026-09-20.md#ref-AI030) | 按本地代码确定量纲、张量与缩放；32 样本仅是 v4 排错建议。 公开代码核验不能代替本地 fork；不把 PR070 摘要中的 pooling/公式直接当实施代码。 |
| **E0.3** 区分合成状态、任务系数和推理可用信息 | [AI018](../03_reference/MMFR_reference_index_v4_1_2026-09-20.md#ref-AI018)、[AI020](../03_reference/MMFR_reference_index_v4_1_2026-09-20.md#ref-AI020)、[AI021](../03_reference/MMFR_reference_index_v4_1_2026-09-20.md#ref-AI021)、[AI027](../03_reference/MMFR_reference_index_v4_1_2026-09-20.md#ref-AI027)、[AI028](../03_reference/MMFR_reference_index_v4_1_2026-09-20.md#ref-AI028) | 质量目标和现有头的 14/2 维契约以本地记录核实；不补造未知损失权重。 推理不能读取 clean Depth、真实 shift 或注入 severity；任务 gate 不是自动校准的物理概率。 |
| **E0.4** 补齐 matched clean-control 与无动作续训对照 | [AI019](../03_reference/MMFR_reference_index_v4_1_2026-09-20.md#ref-AI019)、[AI024](../03_reference/MMFR_reference_index_v4_1_2026-09-20.md#ref-AI024)、[PR090](../03_reference/MMFR_reference_index_v4_1_2026-09-20.md#ref-PR090) | 原训练预算来自项目；新旧分支/随机性/额外步数必须记录。 历史 B0 不是正式配对 control；C0、R、F 必须作为 Batch 1 的匹配合同一起冻结。当前全文复核与子计划编写期间不得启动训练。 |

定位：S6 §1、§5（24–40、180–199）；S1 §S1.1、冻结账本（228–244、284–302）；S1 §S1.3–S1.4（325–382）；S7 PR070（24–82）；S1 §必须分开的量、S1.2（133–154、304–323）；S4 §四–七；S6 §2–4（50–174）；S1 §S2.1（386–404）；S5 §E0。网络原始入口：W5、W23、W1、W4、W21、W22、W13、W14、W18、W19、W6、W3、W7、W8（见末尾网络索引）。
以下操作是 v4 的项目方案；相邻论文支持借鉴方向，不代表每项操作或公式都原封不动来自某篇论文。


保留历史 A2 四个 checkpoint 与 40/40 开发评价，不重新挑选最有利的历史结果。A2 epoch-420 仍是廉价筛选的共同起点。[S1，第一部分 §2.1]

核验只针对新实验确实会动到的接口：输入/输出契约、Depth 几何入口、有效性缩放、无动作等价性、梯度是否接通。已经完成且代码路径未改变的检查不重复当作启动门槛。

配对 clean-control 与候选合同都需要在 `e1_screening_plan.md` 中冻结；当前不得与文献复核或子计划编写并行启动任何训练。最终归因时不能省略 matched control。

本阶段的当前产物不是运行清单，而是全文审计和 E1 子计划。代码与权重身份、新旧协议差异、候选开关、数据白名单、训练预算、选择规则和停止条件都必须在子计划中先写清并通过上级审核。

### 阶段 E1：五组、两批、顺序执行

**进入 E1 的共同前置门禁：** 五篇核心全文全部取得并完成结构化摘录；`../03_reference/literature_fulltext_audit.md` 可复核；`e1_screening_plan.md` 将 R/F/T 的最小实现和比较合同冻结；上级模型完成最终审计；用户另行授权代码修改与相应运行。任一条件未满足时，E1 状态保持 `blocked-before-implementation`。

| 批次 | 组 | 固定比较 | 解锁条件 | 回答的问题 |
| --- | --- | --- | --- | --- |
| **Batch 1：结构能力筛选** | `C0 / R / F` | `R - C0`、`F - C0` | R/F 合同均冻结并获执行授权 | 替代几何或 RGB 特征适配是否增加有效 fallback capability |
| **Batch 2：训练方式筛选** | `Cpair / T` | `T - Cpair` | T 合同冻结、Batch 1 已形成可核验证据并获新的执行授权 | complete-to-corrupt 知识迁移是否超出成对视图暴露本身 |

两批不得同时启动。Batch 2 不能因为 T 的全文先读完而越过 Batch 1；Batch 1 结束也不自动授权 Batch 2。

#### Batch 1：`C0 / R / F`

- **C0：** 从同一个 A2 epoch-420 起点做 matched continuation（匹配续训），不加新动作；与 R/F 使用相同原始故障暴露、学生更新数、优化器恢复语义、checkpoint 机会和评价预算。它回答“只是继续训练是否会涨”。
- **R-lite：** 在 C0 合同上只增加最小输入/替代几何补偿。必须单独记录 original validity、synthetic corruption validity、实际补偿区域、untouched valid depth 和 natural-invalid region；natural-invalid 不得默认被覆盖或与 synthetic-missing 合并。
- **F-lite：** 在 C0 合同上只增加小型 residual adapter；第一轮不接 A2 reliability predictor，不把 reliability-conditioned adapter 与简单适配能力混为一个变量。

R/F 的具体网络宽度、插入层、输出范围、loss 权重、冻结策略、parameter groups、学习率和训练长度均等待全文与本地接口复核后标为 `[设计]` 冻结。旧版“额外 50 epochs”只保留为历史预算建议，不自动成为本轮配置。

#### Batch 2：`Cpair / T`

- **Cpair：** 同一场景生成 clean/corrupt 配对视图，无知识蒸馏；必须与 T 使用相同视图、相同学生更新数、相同 segmentation supervision、相同 checkpoint/selector 机会。
- **T-lite：** 在 Cpair 合同上只增加全文复核后冻结的最简单 output-level KD（输出级知识蒸馏）。教师来源、温度、loss 权重、错误教师过滤和计算成本都必须显式登记；第一版不默认加入 feature/prototype distillation。

净蒸馏收益定义为：

$$
\Delta_{\mathrm{KD}}=T-Cpair.
$$

教师前向成本另报；不能把多看 clean 数据、额外学生更新或更多 checkpoint 选择机会算给 KD。

#### 两批共同的最小验证与升级边界

实现获批后，先执行 identity/no-op、梯度与 optimizer 接通、有限值和 16/32 固定样本 sanity；这些只用于排错，不是涨点证据。短训练后的第一层报告固定包含 clean、`entire_missing@1.0`、`spatial_dropout@0.75` 与学习曲线；只有达到 `e1_screening_plan.md` 事前冻结的 promotion 条件，才申请完整 10-condition Main-Val。具体阈值、预算和 checkpoint 规则当前尚未冻结，禁止从结果倒推。

合法终点包括 `stop`、`inconclusive`、`protocol-blocked` 和 `promote`。单个 R/F/T-lite 失败只关闭该实现合同，不自动否定整个方法族；但不得无限调参直到出现正结果。G 不在任何一批中。

### 阶段 E2：只保留两个优先候选，最多测试两个组合

**本节溯源与实施编号（v4.1 补充）**

| 步骤 | 参考论文及其用途 | 参数来源与引用边界 |
| --- | --- | --- |
| **E2.1** 选少量互补候选并做共同基础/+A/+B/+A+B | [AI023](../03_reference/MMFR_reference_index_v4_1_2026-09-20.md#ref-AI023)、[AI024](../03_reference/MMFR_reference_index_v4_1_2026-09-20.md#ref-AI024)、[PR090](../03_reference/MMFR_reference_index_v4_1_2026-09-20.md#ref-PR090)、[PR070](../03_reference/MMFR_reference_index_v4_1_2026-09-20.md#ref-PR070) | 最多两条优先路线、最多两个组合是预算管理建议。 T+R/T+F 只是工作假说；以收益/成本选，不以相似性或命名选。 |

定位：S5 §E2（312–321）。网络原始入口：W2、W3、W7、W8、W1、W4（见末尾网络索引）。
以下操作是 v4 的项目方案；相邻论文支持借鉴方向，不代表每项操作或公式都原封不动来自某篇论文。


优先保留在 clean、$M_6$、最坏条件和成本之间不被其他版本全面压过的候选，不按“看起来更原创”排序。

例如，T 与 R 有互补迹象，再评估 T+R；T 与 F 有互补迹象，再评估 T+F。F 已解决主要问题时，也不为保留 A2 的故事硬塞回质量头。G 已由 Oracle-A 从本轮候选中移除，不参与组合选择。

组合的最低比较为共同基础、+A、+B、+A+B。只比较基础与最终全模块版本无法判断是否多余。

**优先组合候选仅为 T+R 或 T+F；这只是工作假设，不是已经认定的最佳结构。** 当前不测试 T+R+F；只有单候选筛选证据完成、互补假设明确并另行冻结组合计划后，才允许启动组合实验。

### 阶段 E3：针对获胜路线做实质改造

**本节溯源与实施编号（v4.1 补充）**

| 步骤 | 参考论文及其用途 | 参数来源与引用边界 |
| --- | --- | --- |
| **E3.1** 朴素借用与场景/模型特化直接对比 | [PR070](../03_reference/MMFR_reference_index_v4_1_2026-09-20.md#ref-PR070)、[RE326](../03_reference/MMFR_reference_index_v4_1_2026-09-20.md#ref-RE326)、[AI023](../03_reference/MMFR_reference_index_v4_1_2026-09-20.md#ref-AI023)、[AI024](../03_reference/MMFR_reference_index_v4_1_2026-09-20.md#ref-AI024)、[MoSA](../03_reference/MMFR_reference_index_v4_1_2026-09-20.md#ref-MoSA) | 特化菜单与改造增量定义为本项目设计。 借鉴组件贡献与新增改造贡献分开；移到 backbone 内不自动增加新颖性。 |

定位：S5 §6、§E3（261–277、322–337）。网络原始入口：W1、W4、W5、W2、W3、W10（见末尾网络索引）。
以下操作是 v4 的项目方案；相邻论文支持借鉴方向，不代表每项操作或公式都原封不动来自某篇论文。


此时才从第 6 节的特化菜单选择一到两个真正有证据的短板，不预先要求同时具备恢复、对齐、门控和蒸馏。

必须保留“朴素借用版本”作为对照：

$$
\Delta_{\mathrm{adapt}}
=
M_6(\text{针对性改造})
-
M_6(\text{同骨干上的朴素借用版本}).
$$

该差值帮助界定自身增量。整套系统的大收益可以来自成熟组件，但不能把全部收益都称为新改造的贡献。

### 阶段 E4：设计冻结与正式确认

**本节溯源与实施编号（v4.1 补充）**

| 步骤 | 参考论文及其用途 | 参数来源与引用边界 |
| --- | --- | --- |
| **E4.1** 冻结少量入围模型的完整方法与独立确认 | [AI010](../03_reference/MMFR_reference_index_v4_1_2026-09-20.md#ref-AI010)、[RE326](../03_reference/MMFR_reference_index_v4_1_2026-09-20.md#ref-RE326)、[PR070](../03_reference/MMFR_reference_index_v4_1_2026-09-20.md#ref-PR070) | 最终三 seed 等来自旧项目规划，非某篇基线的统一要求。 所有候选开发不是已完成验证；只对锁定版本做论文确认。 |

定位：S1 §S2.8（596–635）；S5 §E4、§11.2。网络原始入口：W23、W5、W1、W4（见末尾网络索引）。
以下操作是 v4 的项目方案；相邻论文支持借鉴方向，不代表每项操作或公式都原封不动来自某篇论文。


确定一个性能优先版本；必要时另保留一个更轻版本。最终只对少量入围方法做完整匹配训练、多 seed 与独立评价，避免把所有探索方案都扩成大型训练矩阵。

---

## 8. 允许优化训练与选择策略，但建立新的可追踪协议

### 8.1 不把历史规则永久化

**本节溯源与实施编号（v4.1 补充）**

| 步骤 | 参考论文及其用途 | 参数来源与引用边界 |
| --- | --- | --- |
| **V.1** 预登记 clean 约束下的鲁棒验证 selector | [AI010](../03_reference/MMFR_reference_index_v4_1_2026-09-20.md#ref-AI010)、[AI011](../03_reference/MMFR_reference_index_v4_1_2026-09-20.md#ref-AI011) | 选择目标、候选集合、参考模型及 epsilon=0.50 起点均是项目建议。 历史 A2 不追溯重选；论文只是鲁棒评价背景，不是该 selector 的来源。 |

定位：S5 §8.1（346–369）。网络原始入口：W23（见末尾网络索引）。
以下操作是 v4 的项目方案；相邻论文支持借鉴方向，不代表每项操作或公式都原封不动来自某篇论文。


历史 A2 的 clean-only selector 不修改、不追溯重选。新 v4 实验可以事前登记：

$$
\theta^*
=
\arg\max_{\theta\in\mathcal K} M_6^{\mathrm{val}}(\theta)
\quad
\text{subject to }
M_{\mathrm{clean}}^{\mathrm{val}}(\theta)
\ge
M_{\mathrm{clean}}^{\mathrm{val}}(C_{\mathrm{ref}})
-\epsilon.
$$

其中候选 checkpoint 集合 $\mathcal K$、评价频率、参考模型 $C_{\mathrm{ref}}$、容许 clean 损失 $\epsilon$ 和并列规则都在对应新运行前固定；$\epsilon=0.50$ 可作为沿用旧 clean 保护尺度的开发起点。

正式配对 clean-control 尚未完成时，可暂用 C0 作筛选参考，但明确这只是在开发阶段防止相对 C0 退化，不能据此宣称已经保持正式 clean-control 水平。

使用验证集优化鲁棒性不等于测试泄漏。真正不能做的是根据 official test 结果反选模型，或在看完结果后反复改变选择规则而隐藏这些选择。

所有新对照组采用相同选择机会与评价预算；不能只让 MMFR 享有新规则，再直接与按旧规则选出的 A2 比较并归因。

### 8.2 可以改训练配方，但分清比较层次

**本节溯源与实施编号（v4.1 补充）**

| 步骤 | 参考论文及其用途 | 参数来源与引用边界 |
| --- | --- | --- |
| **V.2** 分开系统总收益与模块净收益 | [AI019](../03_reference/MMFR_reference_index_v4_1_2026-09-20.md#ref-AI019)、[AI023](../03_reference/MMFR_reference_index_v4_1_2026-09-20.md#ref-AI023)、[AI024](../03_reference/MMFR_reference_index_v4_1_2026-09-20.md#ref-AI024)、[PR090](../03_reference/MMFR_reference_index_v4_1_2026-09-20.md#ref-PR090) | 同配方同预算的增量控制由项目定义；额外训练/教师成本同时记录。 普通强基线和朴素成熟适配都能胜出，不把全部收益包装成自研模块。 |

定位：S5 §8.2、§9.1（370–389）。网络原始入口：W6、W2、W3、W7、W8（见末尾网络索引）。
以下操作是 v4 的项目方案；相邻论文支持借鉴方向，不代表每项操作或公式都原封不动来自某篇论文。


允许研究 clean/corrupt 比例、缺失暴露、课程安排、损失与冻结策略。冻结测试条件不意味着永久冻结所有训练方法。

报告分两层：

- **系统层比较：** 完整 v4 配方相对原系统提高多少，训练与推理成本是什么。
- **机制层比较：** 同配方、同骨干、同新增预算下，某模块或某改造额外提高多少。

这样既不压制有效训练方法，也不制造“所有提升都来自一个新模块”的错觉。

---

## 9. 性能目标：提高研发雄心，不预告实验结果

### 9.1 主评价保持清楚

**本节溯源与实施编号（v4.1 补充）**

| 步骤 | 参考论文及其用途 | 参数来源与引用边界 |
| --- | --- | --- |
| **V.3** 锁定 M6、clean、逐条件/逐类和成本 | [AI010](../03_reference/MMFR_reference_index_v4_1_2026-09-20.md#ref-AI010)、[AI011](../03_reference/MMFR_reference_index_v4_1_2026-09-20.md#ref-AI011)、[PR040](../03_reference/MMFR_reference_index_v4_1_2026-09-20.md#ref-PR040)、[RE266](../03_reference/MMFR_reference_index_v4_1_2026-09-20.md#ref-RE266) | M6 与旧门槛继承项目；约+1.0、进取+1.5–2.0为新研发目标。 不是论文给出的预期增益；不换分母/删条件制造涨点。 |

定位：S6 §5（180–199）；S1 §指标（246–280）；S5 §9。网络原始入口：W23（见末尾网络索引）。
以下操作是 v4 的项目方案；相邻论文支持借鉴方向，不代表每项操作或公式都原封不动来自某篇论文。


主指标仍为六单故障宏平均 $M_6$；同时报告 clean、整个 10-condition 表、最坏单条件、每类 IoU 和成本。不以更好看的十条件平均替代原主指标。

比较基准不能只选旧 A2。至少包含正式 clean-control、匹配增强/续训对照，以及获胜路线中最有竞争力的成熟方法适配版本。

### 9.2 新版投入目标

> **参数归属：** 本节数字来自 S5 的研发目标，标记为 [设计]，不引用其他论文的涨点作为本项目预期。


对后续正式实验，建议把**相对匹配强基线的 $M_6$ 提升约 1.0 个百分点**作为值得优先投入的研发目标；把 **1.5–2.0 个百分点**视为进取目标，同时争取 clean 不降或回升、整幅缺失有明显改善。

这些是事前设定的投入方向，不是预测值、置信区间或成功保证。若只是把旧 A2 的故障恢复到旧 clean，平均差距只有 1.51；更大的平均提升需要更强表征、clean 改善或超出这种简单恢复假设，不能靠换平均口径得到。

若某模块只有很小收益，仍可作为低成本改进，但不应围绕它无限增加复杂度或把它包装成“大幅提升”。若简单借用已达到最佳性能，则诚实选择它，再判断应用、系统或方法贡献的适当定位。

旧 A2 相对 clean-v3 的预注册门槛属于旧比较，不因本次目标变化而事后撤销，也不自动成为每个小模块必须满足的门槛。[S1，第三部分 §1–2]

---

## 10. 机制分析保留，但由最终主张决定做哪些

**本节溯源与实施编号（v4.1 补充）**

| 步骤 | 参考论文及其用途 | 参数来源与引用边界 |
| --- | --- | --- |
| **M.R** R 入围后检查补偿收益位置与可靠输入保持 | [AI023](../03_reference/MMFR_reference_index_v4_1_2026-09-20.md#ref-AI023)、[AI017](../03_reference/MMFR_reference_index_v4_1_2026-09-20.md#ref-AI017)、[PR070](../03_reference/MMFR_reference_index_v4_1_2026-09-20.md#ref-PR070)、[RE266](../03_reference/MMFR_reference_index_v4_1_2026-09-20.md#ref-RE266) | 区域分层/原图对照是项目机制分析；规模参数待登记。 修复看起来更好不等于分割提升；不能只挑成功样例。 |
| **M.F** F 入围后分离额外容量、适应和质量条件 | [AI024](../03_reference/MMFR_reference_index_v4_1_2026-09-20.md#ref-AI024)、[MoSA](../03_reference/MMFR_reference_index_v4_1_2026-09-20.md#ref-MoSA)、[AI025](../03_reference/MMFR_reference_index_v4_1_2026-09-20.md#ref-AI025) | 同规模 adapter 与有无质量信号由项目设置。 来自有效适应的收益可保留，但不强称可靠性诊断必不可少。 |
| **M.G（历史）** Oracle-A 后保留 suppression 机制记录 | [PR070](../03_reference/MMFR_reference_index_v4_1_2026-09-20.md#ref-PR070)；[`MMFR-Oracle-A`](../../doc/reports/2026-09-19-museg-mmfr-oracle-a-validity-aware-pairwise-geometry.md) | Oracle-A 已完成 validity / aggregated / geometry-off 对照并裁决 `NO-GO`。 当前不做 G 入围分析；历史分析不得被改写成待执行路线。 |
| **M.T** T 入围后比较无蒸馏/输出/可选特征蒸馏 | [AI019](../03_reference/MMFR_reference_index_v4_1_2026-09-20.md#ref-AI019)、[PR090](../03_reference/MMFR_reference_index_v4_1_2026-09-20.md#ref-PR090) | 比较层级与匹配视图为项目设计；具体原型/特征损失需再读选定原文。 教师指导收益须超出成对视图暴露收益。 |

定位：S5 §10；S1 §M1（639–670）；S1 §S2.7（572–594）；S1 §M2（672–748）；S5 §10、§5.4。网络原始入口：W2、W9、W1、W4、W3、W10、W17、W14、W6、W7、W8（见末尾网络索引）。
以下操作是 v4 的项目方案；相邻论文支持借鉴方向，不代表每项操作或公式都原封不动来自某篇论文。


Oracle 指提供额外理想信息的诊断实验，不是任何可训练方法的全局性能上界。

**R 获胜：** 重点比较不修复、朴素修复、针对性修复；检查可靠区域是否被改坏，缺失是否获得任务有用信息。不要只报告恢复图像“更好看”。

**F 获胜：** 比较同规模普通 adapter、残差补偿与有无质量信号；检查收益是不是主要来自新增表示容量与训练。即使来自有效适应，也可以使用，只需相应调整贡献表述。

**G 的历史处置：** Oracle-A 已用真实 validity 对 suppression 动作完成最理想信息版本之一的检验并裁决 `NO-GO`。相关 validity、constant、predicted reliability 与 inverse 等扩展不再作为当前机制分析；只有满足第 0.1 节重启条件并建立新 protocol 时才重新设计。

**T 获胜：** 重点比较同视图无蒸馏、输出蒸馏和可选特征蒸馏，分离教师知识与额外训练暴露。

质量目标 $Q_D$、任务使用系数和最终效用继续区分。允许任务监督改变门控的含义；若它不再预测输入质量，就把它称为任务条件系数，不冒充物理可靠性。

所有分箱、热图和相关性只支持相应解释，不单独证明现实世界因果机制。合成质量目标更不是天然的传感器真实质量。

---

## 11. 最终实验与可信度

### 11.1 基线组织

**本节溯源与实施编号（v4.1 补充）**

| 步骤 | 参考论文及其用途 | 参数来源与引用边界 |
| --- | --- | --- |
| **V.4** 选择普通强基线与相近鲁棒适配 | [PR070](../03_reference/MMFR_reference_index_v4_1_2026-09-20.md#ref-PR070)、[AI003](../03_reference/MMFR_reference_index_v4_1_2026-09-20.md#ref-AI003)、[AI004](../03_reference/MMFR_reference_index_v4_1_2026-09-20.md#ref-AI004)、[AI005](../03_reference/MMFR_reference_index_v4_1_2026-09-20.md#ref-AI005)、[AI019](../03_reference/MMFR_reference_index_v4_1_2026-09-20.md#ref-AI019)、[AI022](../03_reference/MMFR_reference_index_v4_1_2026-09-20.md#ref-AI022)、[AI024](../03_reference/MMFR_reference_index_v4_1_2026-09-20.md#ref-AI024)、[PR090](../03_reference/MMFR_reference_index_v4_1_2026-09-20.md#ref-PR090)、[MoSA](../03_reference/MMFR_reference_index_v4_1_2026-09-20.md#ref-MoSA) | 只选代表性入围方法；同骨干机制比较与跨骨干系统比较分表。 预训练/decoder升级允许，但须对照升级后的无MMFR版本。 |

定位：S5 §11.1（423–429）；S1 §S2.7。网络原始入口：W1、W4、W24、W25、W6、W15、W3、W7、W8、W10（见末尾网络索引）。
以下操作是 v4 的项目方案；相邻论文支持借鉴方向，不代表每项操作或公式都原封不动来自某篇论文。


主表至少覆盖原始骨干、强训练对照、获胜路线的朴素适配、针对性改造版。资源允许时再加入代表性 RGB-D 方法与相近鲁棒方法，不必完整复现 P0/P1/P2 的全部论文。

同骨干受控实验用于解释增量；不同骨干的完整系统比较用于展示实际性能。后者不能因参数、预训练或计算不同而被混称为严格的单变量比较。

更换 backbone、decoder 或预训练也可以另开分支探索，不列禁区。但第一轮优先保留 DFormerv2-S + HAM，目的是复用现有投入和减少混杂；最终升级要对照升级后的无 MMFR 基线，不能拿更强骨干的收益充当鲁棒模块收益。

### 11.2 独立确认

**本节溯源与实施编号（v4.1 补充）**

| 步骤 | 参考论文及其用途 | 参数来源与引用边界 |
| --- | --- | --- |
| **V.5** 组级配对统计、多 seed 与封存 test | [RE326](../03_reference/MMFR_reference_index_v4_1_2026-09-20.md#ref-RE326)、[AI010](../03_reference/MMFR_reference_index_v4_1_2026-09-20.md#ref-AI010) | 196 location groups、10000次、3 seed 来自项目旧规划；测试组数不能照搬196。 bootstrap不代替训练随机性；full-train并入的318张不再是独立验证集。 |
| **V.6** 真正未见故障与外部数据复验 | [AI010](../03_reference/MMFR_reference_index_v4_1_2026-09-20.md#ref-AI010)、[PR070](../03_reference/MMFR_reference_index_v4_1_2026-09-20.md#ref-PR070)、[AI003](../03_reference/MMFR_reference_index_v4_1_2026-09-20.md#ref-AI003)、[PR090](../03_reference/MMFR_reference_index_v4_1_2026-09-20.md#ref-PR090)、[RE326](../03_reference/MMFR_reference_index_v4_1_2026-09-20.md#ref-RE326) | 未见故障选择、NYUv2适配和超参数冻结策略为项目计划。 整条训练链排除才是未见；跨数据集重训不叫零样本迁移。 |

定位：S1 §S2.8（596–635）；S5 §11.2（431–443）；S1 §S2.8（619–635）；S5 §11.2（439–443）。网络原始入口：W5、W23、W1、W4、W7、W8（见末尾网络索引）。
以下操作是 v4 的项目方案；相邻论文支持借鉴方向，不代表每项操作或公式都原封不动来自某篇论文。


开发统计可沿用原记录的 196 个 location groups 与 10,000 次配对重采样；从组级/样本级混淆矩阵重新聚合 mIoU，不把逐图 mIoU 简单平均当作原指标。探索选出的赢家，其开发区间不等于独立确认。[S1，§S2.8]

最终入围方法至少按原计划做 3 个预登记训练 seed，报告单次与均值/标准差。样本 bootstrap 不代替训练随机性。

official test 当前继续保持 `sealed_unread`。未来只有在独立门禁通过、设计与选择规则预先冻结且用户明确授权后才可一次性使用；本蓝图及 E1 子计划均不构成解封授权。若最终把 318 张 val-dev 并入 1,595 张 full train，它们不能继续作为独立选模型集合；采用事先固定的训练时长/checkpoint 规则，或明确保留验证集。

真正未见的故障须在该模型整条训练链中排除；A2 曾经见过的故障，不能因为新 head 没见过就称为整个模型 held-out。新增故障也不能反复调整到本方法最有利。

外部数据验证可沿用旧计划优先考虑 NYUv2，保持算法与选择原则一致并公开必要适配。重新训练后的外部数据集实验不等于零样本迁移。[S1，§S2.8]

---

## 12. 论文叙事怎么写：解释真实工作，不用措辞制造差异

### 12.1 允许的写法

> 我们借鉴任务驱动几何补偿和鲁棒适应方法，在地下矿山 RGB-D 分割中研究可靠 Depth 的利用与失效时的补偿。针对 DFormerv2 的实际信息使用路径，我们对朴素适配方案作出具体修改，并在匹配训练条件下验证这些修改的收益及成本。

这只是写作模板，具体修改和“验证”必须等结果存在后才能填入完成时。

### 12.2 不预先绑定某个故事

如果 R 获胜，围绕任务有用的几何补偿与可靠输入保持组织论文；若 F 获胜，围绕无深度运行和轻量适应；若 T 或 T+R/T+F 获胜，围绕 complete-to-corrupt 知识迁移及其超出成对视图暴露的净收益。G 只作为 Oracle-A 排除过的历史动作边界，不再预设“G 获胜”叙事。最终论文不需要同时讲三个故事。

如果成熟组件组合就已经很强，真实的系统贡献、场景验证和清楚的对照优于虚构一个万能原创模块。但这不保证任何特定期刊或会议的录用，也不能仅凭数值更高就认定算法创新充分。

### 12.3 应避免的做法

不把别人模块换名后称首创；不把“首次在某数据集使用”自动等同于强算法贡献；不声称公开文献没有同类方案而只核对了少量论文；不把通用消融实验本身称为独有机制；不把开发集最好结果写成已确认的普遍 SOTA。

本版不是取消贡献审计，而是把它从“限制能做什么”改为“准确解释做成了什么”。

---

## 13. 当前阶段交付物与授权边界

> **溯源交付要求：** 每个子计划继承本版步骤号与论文 ID，至少给出固定版本、原文章节/公式或代码函数、朴素移植、改造点、参数出处及待核项。没有全文或本地证据支持的配置统一写 `[设计]`，不得补造作者设置。

当前唯一允许的任务是：**收集并复核五篇核心全文，以及在全文门禁通过后编写 E1 子计划供上级模型审核。** 本阶段不允许修改实现代码、启动训练或新实验、使用 GPU/云资源、评价 checkpoint、运行完整 evaluator 或读取 official test。

交付物按以下顺序形成：

| 顺序 | 交付物 | 必须说清楚的内容 | 解锁关系 |
| --- | --- | --- | --- |
| 1 | `../03_reference/literature_fulltext_audit.md` | 五篇固定版本是否取得；逐篇章节/公式/代码锚点；结构、接口、loss、初始化、冻结/训练参数、parameter groups、训练长度、推理信息、对照和成本；“原论文做法 / 本项目适配 / 新建议”；版本差异与待核项 | 五篇全部完成才解锁 E1 冻结 |
| 2 | `e1_screening_plan.md` | 可内含 component provenance（组件溯源）、R/F/T candidate contracts（候选合同）与 screening matrix（筛选矩阵）；分别冻结 R-lite/F-lite/T-lite 的 minimum viable implementation、代码插入点、输入输出、loss、初始化、训练/冻结参数、optimizer 参数组、训练/推理信息、matched controls、预算、快速断言、checkpoint 规则、quick-val 到 10-condition 的 promotion/stop 条件；明确 Batch 1 后才能 Batch 2 | 上级模型审核和用户授权后才可交给执行模型 |
| 3 | `screening_results/` | 按获批批次保留逐条件原始分数、clean、$M_6$、成本、失败版本、学习曲线、协议阻塞与恢复点 | 仅在对应批次实际获批并执行后形成 |
| 4 | `promotion_decision.md` | 哪 1–2 个候选值得继续，是否有证据测试 T+R 或 T+F，哪些主张尚无证据，G 是否仍保持冻结 | 只依据已完成筛选证据形成 |

`../03_reference/literature_fulltext_audit.md` 未完成时，不能用现有索引摘要代替全文门禁，也不能先写“暂定实现”后让执行模型边查边改。`e1_screening_plan.md` 未经上级审计前，不得交给执行模型。即使计划冻结，代码、训练、GPU、云、完整评价和 official test 仍需分别取得明确授权。

本蓝图修订没有运行实验，也没有虚构修改行数、实测收益、预算或运行时间。当前准确恢复点是：用户收集五篇核心全文；全文到齐后，先编写上述两份文档，再停下等待上级审核。

**最终行动结论：先完成五篇核心全文的实施级复核，再冻结 R/F/T 的 E1 合同；随后只按 Batch 1 `C0/R/F` → Batch 2 `Cpair/T` 的顺序申请执行。**

---


---

## 附录 A：参数与证据归属账本

| 项目/参数 | 性质 | 来源 | 不可误写为 |
| --- | --- | --- | --- |
| Oracle-A：strict 的 clean `-3.48`、spatial dropout `-1.77`；spatial strict vs geometry-off `+0.04`；entire missing strict = geometry-off = `52.61` | 已完成的 development 级项目结果 | [`MMFR-Oracle-A`](../../doc/reports/2026-09-19-museg-mmfr-oracle-a-validity-aware-pairwise-geometry.md) | 只关闭 validity-aware geometry suppression；不否定 Depth geometry、reliability 诊断或 R/F/T，不授权 predicted gate 训练。 |
| 五篇核心全文全部取得并形成结构化摘录 | E1 前硬门禁，当前待完成 | AI023 `arXiv:2604.11585`；AI024 `arXiv:2607.20326v1`；AI019 DOI `10.1145/3664647.3681698`；PR090 官方 CVPR 2026 固定版；PR070 DOI `10.1109/CVPR52734.2025.01802` | 文献已定位、摘要可见、既有审计或网页片段不等于本轮全文复核；PR090 DOI 待核且禁止猜造。 |
| AI017 与 MoSA 全文 | 条件补充门禁 | AI017 仅在 R 采用其校正机制时阻塞 R；MoSA 仅在 F 采用其空间适配机制时阻塞 F | 不是当前五篇共同硬门禁，也不能在未采用对应机制时阻塞全部 E1。 |
| E1 Batch 1 `C0/R/F` → Batch 2 `Cpair/T` | Oracle-A 后继执行设计 | 本修订 §4、§7、§13 | 两批不得同时启动；Batch 1 证据不自动授权 Batch 2；G 不在首轮。 |
| A2 epoch-420、clean 57.06、M6 55.55 | 项目结果转录 | S6 §1、§5 | 不代表本轮重新评测；B0 未成为 matched control。 |
| 500 epochs、batch 10、主干 LR 6e-5、25% clean/75% corruption | 项目账本继承 | S1 §冻结项目账本（228–244） | 实施须核配置/调度器；不能当作所有文献通用配置。 |
| 318 val-dev、196 location groups、10000 次配对 bootstrap | 项目协议继承 | S1 §S2.8；S6 §1 | test 分组另核；不把像素当独立训练。 |
| 3 个预登记训练 seed | 项目确认阶段计划 | S1 §S2.8 | 不是已完成 3 seed 的事实。 |
| 额外 50 epochs、小型 adapter、最多两路线/两组合 | v4 研发建议 | S5 §5、§E1–E2 | 依据学习曲线与预算登记；不是论文原设。 |
| clean 约束 epsilon=0.50 与鲁棒验证 selector | v4 新实验建议 | S5 §8.1 | 新组同机会；不追溯重选 A2，不用 test。 |
| 约+1.0、进取+1.5–2.0 mIoU 点 | v4 研发目标 | S5 §9.2 | 不是预测/保证/既有结果；不得改指标口径。 |
| GeomPrompt 原 loss 有分割项、TV 与 L1 正则 | 本轮原文核验 | AI023；W2 §3.4 | 无 Depth GT 不等于无正则；v4 简化须标明。 |
| ConD 原三种模态状态等概率、复制编码器 | 本轮原文核验 | AI024；W3 §II-A–II-C | 不是 A2 的 25/75；不是小型 adapter 的现成配置。 |

## 附录 B：关键文献索引（完整135条另见总索引）

以下57条覆盖本轮主线、重要近邻与评价/场景备用来源；不是57个必做实验。缺 DOI 时保留题名和原始入口，绝不猜造。

| 编号 | 题名 | DOI / 标识 | v4.1 用途 |
| --- | --- | --- | --- |
| [PR070](../03_reference/MMFR_reference_index_v4_1_2026-09-20.md#ref-PR070) | DFormerv2: Geometry Self-Attention for RGBD Semantic Segmentation | doi: `10.1109/CVPR52734.2025.01802` | 骨干信息使用路径；实现前核对本地几何算子 |
| [PR029](../03_reference/MMFR_reference_index_v4_1_2026-09-20.md#ref-PR029) | Uncertainty-Aware Modality Fusion for Unaligned RGB-T Salient Object Detection | 未提供／未核；按完整题名检索，禁止补造 | 局部不确定性与空间/通道调制近邻 |
| [PR089](../03_reference/MMFR_reference_index_v4_1_2026-09-20.md#ref-PR089) | SGMA: Semantic-Guided Modality-Aware Segmentation for Remote Sensing with Incomplete Multimodal Data | doi: `10.1109/TGRS.2026.3692798` | 多尺度语义引导可靠性、融合与脆弱模态采样 |
| [PR090](../03_reference/MMFR_reference_index_v4_1_2026-09-20.md#ref-PR090) | Towards Robust Multi-Modal Semantic Segmentation with Teacher-Student Framework and Hybrid Prototype Distillation | 未提供／未核；按完整题名检索，禁止补造 | 教师—学生与混合原型蒸馏的直接任务近邻 |
| [PR040](../03_reference/MMFR_reference_index_v4_1_2026-09-20.md#ref-PR040) | CLP: A Real-World Dataset of Contaminated Lens Protectors for Robust Semantic Segmentation | 未提供／未核；按完整题名检索，禁止补造 | 真实污染与条件化评价设计参考 |
| [PR042](../03_reference/MMFR_reference_index_v4_1_2026-09-20.md#ref-PR042) | Robust Promptable Video Object Segmentation | 未提供／未核；按完整题名检索，禁止补造 | 条件信号是否有效的消融思路 |
| [PR024](../03_reference/MMFR_reference_index_v4_1_2026-09-20.md#ref-PR024) | ELVIS: Enhance Low-Light for Video Instance Segmentation in the Dark | 未提供／未核；按完整题名检索，禁止补造 | 退化估计与下游任务驱动增强 |
| [PR038](../03_reference/MMFR_reference_index_v4_1_2026-09-20.md#ref-PR038) | M4-SAM: Multi-Modal Mixture-of-Experts with Memory-Augmented SAM for RGB-D Video Salient Object Detection | 未提供／未核；按完整题名检索，禁止补造 | 模态专家路由、卷积适配与多层门控 |
| [PR053](../03_reference/MMFR_reference_index_v4_1_2026-09-20.md#ref-PR053) | VarSplat: Uncertainty-aware 3D Gaussian Splatting for Robust RGB-D SLAM | 未提供／未核；按完整题名检索，禁止补造 | 不确定性影响下游处理的跨任务参考 |
| [PR074](../03_reference/MMFR_reference_index_v4_1_2026-09-20.md#ref-PR074) | A Mixed Diet Makes DINO An Omnivorous Vision Encoder | 未提供／未核；按完整题名检索，禁止补造 | 预训练表征锚定与跨模态对齐 |
| [PR059](../03_reference/MMFR_reference_index_v4_1_2026-09-20.md#ref-PR059) | Efficient RGB-D Scene Understanding via Multi-task Adaptive Learning and Cross-dimensional Feature Guidance | doi: `10.48550/arXiv.2603.07570` | 正常 RGB-D 多任务场景理解 |
| [PR117](../03_reference/MMFR_reference_index_v4_1_2026-09-20.md#ref-PR117) | High-Precision Dichotomous Image Segmentation via Depth Integrity-Prior and Fine-Grained Patch Strategy | 未提供／未核；按完整题名检索，禁止补造 | 深度完整性与几何分割背景 |
| [PR138](../03_reference/MMFR_reference_index_v4_1_2026-09-20.md#ref-PR138) | Beyond Appearance: Camouflaged Object Detection via Geometric Structure | 未提供／未核；按完整题名检索，禁止补造 | 几何与语义互补的背景 |
| [PR158](../03_reference/MMFR_reference_index_v4_1_2026-09-20.md#ref-PR158) | Joint Spectral Image Reconstruction and Semantic Segmentation with Cooperative Unfolding | 未提供／未核；按完整题名检索，禁止补造 | 重建与语义任务联合设计的跨任务参考 |
| [PR165](../03_reference/MMFR_reference_index_v4_1_2026-09-20.md#ref-PR165) | REL-SF4PASS: Panoramic Semantic Segmentation with REL Depth Representation and Spherical Fusion | 未提供／未核；按完整题名检索，禁止补造 | 区域自适应融合与几何表征 |
| [AI001](../03_reference/MMFR_reference_index_v4_1_2026-09-20.md#ref-AI001) | Are Multimodal Transformers Robust to Missing Modality? | doi: `10.1109/CVPR52688.2022.01764` | 缺失模态鲁棒性历史参考 |
| [AI002](../03_reference/MMFR_reference_index_v4_1_2026-09-20.md#ref-AI002) | Multimodal Token Fusion for Vision Transformers | doi: `10.1109/CVPR52688.2022.01187` | 多模态 token 融合历史参考 |
| [AI003](../03_reference/MMFR_reference_index_v4_1_2026-09-20.md#ref-AI003) | CMX: Cross-Modal Fusion for RGB-X Semantic Segmentation With Transformers | doi: `10.1109/TITS.2023.3300537` | 普通 RGB-X 强基线候选 |
| [AI004](../03_reference/MMFR_reference_index_v4_1_2026-09-20.md#ref-AI004) | DFormer: Rethinking RGBD Representation Learning for Semantic Segmentation | doi: `10.48550/arXiv.2309.09668` | DFormer 原骨干及 ConD 的结构背景 |
| [AI005](../03_reference/MMFR_reference_index_v4_1_2026-09-20.md#ref-AI005) | GeminiFusion: Efficient Pixel-wise Multimodal Fusion for Vision Transformer | doi: `10.48550/arXiv.2406.01210` | 普通多模态强基线与 GeomPrompt 的原始承载模型背景 |
| [AI006](../03_reference/MMFR_reference_index_v4_1_2026-09-20.md#ref-AI006) | Missing Modality Robustness in Semi-Supervised Multi-Modal Semantic Segmentation | doi: `10.1109/WACV57701.2024.00106` | 半监督场景下缺失模态鲁棒性历史参考 |
| [AI007](../03_reference/MMFR_reference_index_v4_1_2026-09-20.md#ref-AI007) | Multi-Modal Learning with Missing Modality via Shared-Specific Feature Modelling | doi: `10.1109/CVPR52729.2023.01524` | 共享/特有表征历史参考 |
| [AI008](../03_reference/MMFR_reference_index_v4_1_2026-09-20.md#ref-AI008) | Learning Modality-Agnostic Representation for Semantic Segmentation from Any Modalities | doi: `10.1007/978-3-031-72754-2_9` | 任意模态表示学习历史参考 |
| [AI009](../03_reference/MMFR_reference_index_v4_1_2026-09-20.md#ref-AI009) | Robust Multimodal Learning With Missing Modalities via Parameter-Efficient Adaptation | doi: `10.1109/TPAMI.2024.3476487` | 参数高效缺失模态适配历史参考 |
| [AI010](../03_reference/MMFR_reference_index_v4_1_2026-09-20.md#ref-AI010) | Benchmarking Multi-Modal Semantic Segmentation Under Sensor Failures: Missing and Noisy Modality Robustness | doi: `10.1109/CVPRW67362.2025.00146`；arxiv: `2503.18445` | 传感器缺失/噪声条件的鲁棒分割评价参考 |
| [AI011](../03_reference/MMFR_reference_index_v4_1_2026-09-20.md#ref-AI011) | Benchmarking the Robustness of Semantic Segmentation Models with Respect to Common Corruptions | doi: `10.1007/s11263-020-01383-2` | 分割常见腐蚀评价历史参考 |
| [AI012](../03_reference/MMFR_reference_index_v4_1_2026-09-20.md#ref-AI012) | RoboDepth: Robust Out-of-Distribution Depth Estimation under Corruptions | doi: `10.52202/075280-0932` | 深度估计的退化鲁棒性参考 |
| [AI013](../03_reference/MMFR_reference_index_v4_1_2026-09-20.md#ref-AI013) | On Calibration of Modern Neural Networks | doi: `10.48550/arXiv.1706.04599` | 预测概率校准 |
| [AI014](../03_reference/MMFR_reference_index_v4_1_2026-09-20.md#ref-AI014) | SelectiveNet: A Deep Neural Network with an Integrated Reject Option | doi: `10.48550/arXiv.1901.09192` | 选择性预测与拒绝决策参考 |
| [AI015](../03_reference/MMFR_reference_index_v4_1_2026-09-20.md#ref-AI015) | Incomplete RGB-D Salient Object Detection: Conceal, Correlate and Fuse | doi: `10.1016/j.patcog.2024.110700` | Depth 质量判断与低质量输入处理 |
| [AI016](../03_reference/MMFR_reference_index_v4_1_2026-09-20.md#ref-AI016) | Rethinking RGB-D Salient Object Detection: Models, Data Sets, and Large-Scale Benchmarks | doi: `10.1109/TNNLS.2020.2996406` | 低质量深度筛选、RGB/RGB-D 路径选择 |
| [AI017](../03_reference/MMFR_reference_index_v4_1_2026-09-20.md#ref-AI017) | Calibrated RGB-D Salient Object Detection | doi: `10.1109/CVPR46437.2021.00935` | 原始深度与 RGB 估计深度的质量条件校正 |
| [AI018](../03_reference/MMFR_reference_index_v4_1_2026-09-20.md#ref-AI018) | Uncertainty Inspired RGB-D Saliency Detection | doi: `10.1109/TPAMI.2021.3073564` | 输出/标注不确定性与输入质量的区分 |
| [AI019](../03_reference/MMFR_reference_index_v4_1_2026-09-20.md#ref-AI019) | MaskMentor: Unlocking the Potential of Masked Self-Teaching for Missing Modality RGB-D Semantic Segmentation | doi: `10.1145/3664647.3681698` | 完整教师到缺失学生、自监督遮罩学习 |
| [AI020](../03_reference/MMFR_reference_index_v4_1_2026-09-20.md#ref-AI020) | A Conflict-Guided Evidential Multimodal Fusion for Semantic Segmentation | doi: `10.1109/WACV61041.2025.00141` | 输出证据冲突与可靠性折扣 |
| [AI021](../03_reference/MMFR_reference_index_v4_1_2026-09-20.md#ref-AI021) | Provable Dynamic Fusion for Low-Quality Multimodal Data | doi: `10.48550/arXiv.2306.02050` | 质量相关动态融合及任务效用关系 |
| [AI022](../03_reference/MMFR_reference_index_v4_1_2026-09-20.md#ref-AI022) | Centering the Value of Every Modality: Towards Efficient and Resilient Modality-Agnostic Semantic Segmentation | doi: `10.1007/978-3-031-72890-7_12`；arxiv: `2407.11344` | 稳健/脆弱模态与任意模态分割 |
| [AI023](../03_reference/MMFR_reference_index_v4_1_2026-09-20.md#ref-AI023) | GeomPrompt: Geometric Prompt Learning for RGB-D Semantic Segmentation Under Missing and Degraded Depth | doi: `10.48550/arXiv.2604.11585` | 无深度几何提示及受损深度任务驱动残差修正 |
| [AI024](../03_reference/MMFR_reference_index_v4_1_2026-09-20.md#ref-AI024) | Toward Reliable RGB-D Semantic Segmentation: Handling Missing Modalities via Condition Dropout | doi: `10.48550/arXiv.2607.20326` | 冻结原网络、复制编码器和零初始化注入 |
| [AI025](../03_reference/MMFR_reference_index_v4_1_2026-09-20.md#ref-AI025) | When Fusion Fails: Corruption-Aware Rebalanced Fusion for Multi-Modal Medical Image Segmentation | doi: `10.1145/3767308.3836234`；arxiv: `2609.10261v1`；doi: `10.48550/arXiv.2609.10261` | 特征传输抑噪和模态贡献重平衡 |
| [AI026](../03_reference/MMFR_reference_index_v4_1_2026-09-20.md#ref-AI026) | SimMLM: A Simple Framework for Multi-Modal Learning with Missing Modality | doi: `10.1109/ICCV51701.2025.02231` | 专家门控和多/少模态约束 |
| [AI027](../03_reference/MMFR_reference_index_v4_1_2026-09-20.md#ref-AI027) | Addressing Missing and Noisy Modalities in One Solution: Unified Modality-Quality Framework for Low-Quality Multimodal Data | doi: `10.48550/arXiv.2603.02695` | 质量排序、增强与质量感知专家路由 |
| [AI028](../03_reference/MMFR_reference_index_v4_1_2026-09-20.md#ref-AI028) | Adaptive Modality Reliability Diagnosis and Restoration for Robust Multimodal Intent Recognition | doi: `10.48550/arXiv.2608.03475` | 合成退化监督、恢复后复诊与融合 |
| [AI029](../03_reference/MMFR_reference_index_v4_1_2026-09-20.md#ref-AI029) | Confidence Propagation through CNNs for Guided Sparse Depth Regression | doi: `10.48550/arXiv.1811.01791` | 稀疏深度置信传播和归一化聚合 |
| [AI030](../03_reference/MMFR_reference_index_v4_1_2026-09-20.md#ref-AI030) | Uncertainty-Aware CNNs for Depth Completion: Uncertainty from Beginning to End | doi: `10.48550/arXiv.2006.03349` | 深度补全中的输入/传播不确定性 |
| [MoSA](../03_reference/MMFR_reference_index_v4_1_2026-09-20.md#ref-MoSA) | Modality-Aware Spatially-Adaptive Adaptation for RGB-X Semantic Segmentation | doi: `10.1109/ACCESS.2026.3694496` | 空间可靠性、适配器调制与融合 |
| [ANGA](../03_reference/MMFR_reference_index_v4_1_2026-09-20.md#ref-ANGA) | Anchor-Guided Gradient Alignment for Incomplete Multimodal Learning | 未提供／未核；按完整题名检索，禁止补造 | 可靠重建样本的选择与梯度对齐 |
| [RE326](../03_reference/MMFR_reference_index_v4_1_2026-09-20.md#ref-RE326) | MUSeg: A multimodal semantic segmentation dataset for complex underground mine scenes | doi: `10.1038/s41597-025-05493-9` | MUSeg 数据集原始来源 |
| [RE240](../03_reference/MMFR_reference_index_v4_1_2026-09-20.md#ref-RE240) | Mine-DW-Fusion: BEV Multiscale-Enhanced Fusion Object-Detection Model for Underground Coal Mine Based on Dynamic Weight Adjustment | doi: `10.3390/s25165185` | 矿下局部置信图、动态融合与补偿 |
| [RE026](../03_reference/MMFR_reference_index_v4_1_2026-09-20.md#ref-RE026) | LFR-CMT-3D: Lipschitz-regularized multi-modal fusion for robust object detection in open-pit mines | doi: `10.1088/1361-6501/ae58c7` | 物理退化先验、局部权重与恢复协同 |
| [RE094](../03_reference/MMFR_reference_index_v4_1_2026-09-20.md#ref-RE094) | Zero-Shot Polarization-Intensity Physical Fusion Monocular Depth Estimation for High Dynamic Range Scenes | doi: `10.3390/photonics13030268` | 物理有效性与可靠性条件输入处理 |
| [RE043](../03_reference/MMFR_reference_index_v4_1_2026-09-20.md#ref-RE043) | Task-Aware Low-Light Image Enhancement Method for Underground Coal Mine Monitoring | 未提供／未核；按完整题名检索，禁止补造 | 矿下任务感知低照增强 |
| [RE049](../03_reference/MMFR_reference_index_v4_1_2026-09-20.md#ref-RE049) | M-SURE: Enhanced and reliable safety monitoring in low-light mines | doi: `10.1016/j.aei.2026.104702` | 低照矿下可靠感知与预测过置信 |
| [RE266](../03_reference/MMFR_reference_index_v4_1_2026-09-20.md#ref-RE266) | Scene Understanding System of Underground Pipeline Corridors Under Characteristic Degradation Conditions | doi: `10.3390/s26010141` | 任务驱动处理与均值掩盖类别退化 |
| [RE418](../03_reference/MMFR_reference_index_v4_1_2026-09-20.md#ref-RE418) | Adaptive Image Enhancement Method for Coal-Mine Underground Image Based on No-Reference Quality Evaluation | doi: `10.1109/TIM.2024.3470234` | 质量代理驱动自适应增强 |
| [RE452](../03_reference/MMFR_reference_index_v4_1_2026-09-20.md#ref-RE452) | Graph-based adaptive weighted fusion SLAM using multimodal data in complex underground spaces | doi: `10.1016/j.isprsjprs.2024.08.007` | 残差/内点率驱动动态多传感器权重 |
| [RE360](../03_reference/MMFR_reference_index_v4_1_2026-09-20.md#ref-RE360) | DeepFissureNets-Infrared-Visible: Infrared visible image fusion for boosting mining-induced ground fissure semantic segmentation | doi: `10.1016/j.jrmge.2025.03.045` | 红外/可见互补与下游分割 |

## 附录 C：冲突、版本与引用边界（不静默改写原资料）

### K01｜附件“全文已核”与本轮核验范围
记录：P0/P1/P2 记载此前已进行全文审计。
处理：本轮可使用其审计结论作为所提供资料，但不改写为本轮已逐篇重读原 PDF。W 记录单独标明代码/方法段落/摘要/检索入口/访问失败。
来源：S2、S3、S4、S13。

### K02｜PR070：论文整理表述与当前公开实现
记录：S7 PR070 描述 average pooling 与 softmax 后乘几何 mask；S1 已提醒公开实现差异。
处理：W1 当前代码使用双线性插值，并在 softmax 前加 mask/bias。保留原整理文本；实施前核对本地 fork、归一化、位置/深度混合权重。二者不能未经推导直接当同一算子。
来源：S7、S1、W1、W4。

### K03｜PR059：两份原始条目与出版时间冲突
记录：相同题名既被列为 KBS 327 (2025), 114107，又记录 arXiv:2603.07570v1 (2026-03-08)。
处理：一个 PR059 下保留两条 source record；本轮确认预印本题名/作者/日期/DOI，未确认 KBS 对应，正式刊物元数据仍待核。
来源：S7、S8、S11、W26。

### K04｜MoSA：RGB-X / Multimodal 题名与 DOI 对应
记录：附件题名为 Modality-Aware Spatially-Adaptive Adaptation for RGB-X Semantic Segmentation；官方检索显示 MoSA: ... for Multimodal Semantic Segmentation。
处理：两种题名都保留，继续使用同一既有代号 MoSA。DOI 10.1109/ACCESS.2026.3694496 为附件记录；出版商正文受限，未宣称完整匹配已核。
来源：S4、S13、S1、W10。

### K05｜PR090：CVPR RobustSeg 与相关 arXiv 多版本
记录：CVPR 题名为 Towards Robust Multi-Modal ... Teacher-Student ... Hybrid Prototype Distillation；相关 arXiv:2505.12861 存在题名/模块变化。
处理：复现前指定 CVPR 版本或某一 arXiv vN；不能拼接不同版本的 RRM/FSM/HPDM 等模块。W8 只作相关版本入口，不证明完全同构。
来源：S7、S3、W7、W8。

### K06｜AI023：两种输入模式、监督与发表身份
记录：GeomPrompt 用 RGB 生成提示；Recovery 另使用受损 Depth。摘要称无需 Depth 监督，方法段落同时列出 TV 和 L1 正则。
处理：“只依赖下游分割监督”不简写成“原 loss 只有 CE”；v4 先用分割损失是简化候选。arXiv 注明 CVPR 2026 URVIS Workshop 接收，不能标成 CVPR 主会。
来源：S2、S5、W2。

### K07｜AI024：复制编码器不等于小型 adapter
记录：ConD 原方法训练复制编码器并零初始化注入；v4 F 只借鉴残差适应思路。
处理：F 属于适配/启发式简化，除非补齐原结构和协议，否则不称忠实复现。代码是否公开、推理成本以实际仓库和测量为准；摘要仍使用未来公开的措辞。
来源：S3、S5、W3。

### K08｜AI025：补充预印本与接收状态
记录：附件有 ACM DOI；本轮找到 arXiv:2609.10261v1，提交 2026-09-09，摘要注明 ACM MM 2026 接收。
处理：沿用 AI025，不新编 AI；追加预印本入口。arXiv DOI 页面标 pending registration，不能声称注册已完成；接收不等于会议已经举行。
来源：S13、S2、W17。

### K09｜ANGA：找到正式检索入口，但未核 DOI
记录：原清单使用 ANGA 临时代号，并要求后续核对正式标识。
处理：本轮补充 CVPR 2026 官方检索入口；保留 ANGA，不凭篇名或页码猜造 DOI，也不擅自新增 AI 编号。
来源：S13、S3、W20。

### K10｜项目结果与研发参数分离
记录：A2 结果来自 S6；25/75 exposure、196 组等来自项目账本；50 epochs 与新 selector/目标是设计建议。
处理：不能把 50 epochs、epsilon=0.50、+1.0 或 +1.5–2.0 指为论文原设；不能把历史 B0 当 matched control 或把 52.55 当无深度上限。
来源：S6、S1、S5。

### K11｜保留文献证据，不恢复旧方法禁令
记录：旧清单按几何专用主线给出禁区/降级/排除建议。
处理：保留原文件作为历史证据；执行路线以 v4/v4.1 为准。“已有先例”只要求标明来源、适配与比较，不构成不用该方法的理由。背景条目保留索引，按需要重新评估。
来源：S2、S3、S4、S5、S11、S12、S13。

### K12｜合成质量、效用及物理真实性
记录：旧审计部分措辞把 Q_D 称真实质量、把 utility 称真实边际效用；v3/v4 已限制这种解释。
处理：Q_D 仅是已定义合成协议下的状态目标；任务系数与固定模型干预收益另列。真实矿下传感质量/因果安全性须另有证据，不通过引用术语获得。
来源：S2、S4、S1、S5。


### K13｜PR040：作者拼写的已有更正
原记录：原 PR040 条目作者含 Hyunsuh Koh；S11 元数据修正（209–217）明确写应为 Hyunseo Koh。
处理：原始字段保留不覆盖；在 PR040 条目与 JSON 追加 supplied_corrections。正式引用使用前核作者页；不把保留原文误作认可其拼写。
依据：S8、S11。

## 附录 D：原始资料身份与后续子计划入口

| 文件ID | 原文件名 | SHA-256 前12位 | 物理行数 | 包内原样副本 |
| --- | --- | --- | ---: | --- |
| S1 | `MMFR_research_blueprint_v3_2026-09-18.md` | `10fc73aeb654` | 1020 | [S1_blueprint_v3.md](../03_reference/source_materials/S1_blueprint_v3.md) |
| S2 | `防撞车P0.md` | `0be5e394d4c4` | 776 | [S2_P0_audit.md](../03_reference/source_materials/S2_P0_audit.md) |
| S3 | `防撞车P1.md` | `3858ea3525d7` | 947 | [S3_P1_audit.md](../03_reference/source_materials/S3_P1_audit.md) |
| S4 | `防撞车P2.md` | `93bb36120d0f` | 1592 | [S4_P2_audit.md](../03_reference/source_materials/S4_P2_audit.md) |
| S5 | `MMFR_research_blueprint_v4_2026-09-20(1).md` | `dac01bf7e7d6` | 514 | [S5_blueprint_v4_original.md](../03_reference/source_materials/S5_blueprint_v4_original.md) |
| S6 | `A2实验结果报告.md` | `2f9d82de2be2` | 1347 | [S6_A2_report.md](../03_reference/source_materials/S6_A2_report.md) |
| S7 | `PR核心.md` | `5de41a4549b6` | 105 | [S7_PR_core.md](../03_reference/source_materials/S7_PR_core.md) |
| S8 | `PR其他.txt` | `c728325bc524` | 573 | [S8_PR_other.txt](../03_reference/source_materials/S8_PR_other.txt) |
| S9 | `RE核心.md` | `4d603dc6407a` | 344 | [S9_RE_core.md](../03_reference/source_materials/S9_RE_core.md) |
| S10 | `RE全.txt` | `4bc45d751001` | 1068 | [S10_RE_all.txt](../03_reference/source_materials/S10_RE_all.txt) |
| S11 | `MMFR-PR文献整理清单.md` | `84b8cc56c05a` | 246 | [S11_PR_checklist.md](../03_reference/source_materials/S11_PR_checklist.md) |
| S12 | `MMFR-RE文献整理清单.md` | `752acf9c997d` | 94 | [S12_RE_checklist.md](../03_reference/source_materials/S12_RE_checklist.md) |
| S13 | `MMFR-新增文献与Idea撞车审计.md` | `6ef3ea7289b7` | 341 | [S13_literature_audit_checklist.md](../03_reference/source_materials/S13_literature_audit_checklist.md) |
| S14 | `生成研究蓝图使用的提示词.txt` | `5f5ae01f538d` | 65 | [S14_blueprint_instructions.txt](../03_reference/source_materials/S14_blueprint_instructions.txt) |

完整 SHA-256 在 JSON 中；这里的哈希是附件身份，不是训练代码或模型权重哈希。S5 的原版和带 `(1)` 的重上传版字节相同，仅计一个来源。早期 v2 原件本轮未提供，其历史信息只能按 S1 的转述使用。

### 后续请求子计划的固定写法

> 依据 MMFR v4.1 的【步骤号，例如 R.1–R.3】及总索引的【PACK-R】，先读取必读论文的已提供全文/审计与所选原始版本。逐项列出：论文编号、版本/DOI、原文章节或代码函数、借用机制、朴素移植、针对性改造、参数出处、匹配对照、产物和待核项。摘要没有的参数写成建议，不冒称作者原设。保持 v4 性能优先路线；不要把存在先例改回采用禁令。


## 附录 E：网络来源索引（可离线保存的最终溯源表）

访问/检索日期均为 **2026-09-20**。W1–W3 沿用 v4 编号，W4–W26 为本次补充；W 编号标识网页，不是新论文编号。
“打开条目”“检索可见”“方法段落”“公开代码”是不同核验深度。失败/受限的入口也保留，避免后续把“本轮没读到”误作“不存在”。代码/项目链接不表示代码已运行。

<a id="W1"></a>
### W1｜DFormerv2 官方公开代码
- 对应论文：PR070。核验范围：**已读取公开代码片段**。
- 原始入口：<https://raw.githubusercontent.com/VCIP-RGBD/DFormer/main/models/encoders/DFormerv2.py>
- 支持内容：GeoPriorGen 的双线性 resize；Decomposed_GSA / Full_GSA 中 softmax 前加入几何偏置。
- 限制：可变 main 分支；未取得本地 fork / checkpoint；本轮不声称已锁定公开 commit。首次实施时固定 commit 并对照本地函数。

<a id="W2"></a>
### W2｜GeomPrompt / GeomPrompt-Recovery 原始 HTML 与 arXiv 条目
- 对应论文：AI023。核验范围：**已读取方法/训练段落及元数据**。
- 原始入口：<https://arxiv.org/html/2604.11585>
- 支持内容：§3：区分 RGB-only 几何提示与受损 Depth 残差修正；无 Depth GT；训练含分割目标与正则。arXiv 条目标注获 CVPR 2026 URVIS Workshop 接收。
- 限制：不是 CVPR 主会身份；未独立核定 workshop 正式 DOI。原输入量纲、幅度、损失权重不自动适用于 MUSeg。
- 同源补充入口：<https://arxiv.org/abs/2604.11585>。

<a id="W3"></a>
### W3｜Condition Dropout (ConD) 原始 HTML 与 arXiv 条目
- 对应论文：AI024。核验范围：**已读取方法/训练段落及元数据**。
- 原始入口：<https://arxiv.org/html/2607.20326v1>
- 支持内容：§II：冻结原编码器、训练复制编码器、零初始化特征注入；三种模态可用性训练。
- 限制：原方法不等同于轻量 adapter；对 DFormer 的结果不等同于 DFormerv2-S。摘要仍称代码将在接收后公开，不能断言代码已发布或推理零开销。
- 同源补充入口：<https://arxiv.org/abs/2607.20326>。

<a id="W4"></a>
### W4｜DFormerv2 CVPR 2025 官方论文条目
- 对应论文：PR070。核验范围：**官方检索条目可见；直开失败**。
- 原始入口：<https://openaccess.thecvf.com/content/CVPR2025/html/Yin_DFormerv2_Geometry_Self-Attention_for_RGBD_Semantic_Segmentation_CVPR_2025_paper.html>
- 支持内容：题名、作者、CVPR 2025 身份和摘要中的几何先验定位。
- 限制：不能把检索条目当本轮重新读完主文/补充材料；实现细节另看 W1。

<a id="W5"></a>
### W5｜MUSeg 数据集原始论文
- 对应论文：RE326。核验范围：**已打开出版商原始页面**。
- 原始入口：<https://www.nature.com/articles/s41597-025-05493-9>
- 支持内容：MUSeg 数据集与其地下矿山 RGB-D 场景来源；DOI 页面可定位。
- 限制：本项目 train-dev / val-dev 划分、A2 结果与失效 manifest 仍来自 S6/S1，不能归为数据集论文给出的数值。

<a id="W6"></a>
### W6｜MaskMentor DOI 与作者投稿入口
- 对应论文：AI019。核验范围：**DOI 直开失败；原始 PDF 入口已定位，未读取**。
- 原始入口：<https://doi.org/10.1145/3664647.3681698>
- 支持内容：保留 S13 的题名—DOI 索引和 OpenReview 原始检索入口。
- 限制：本轮没有重新核 PDF 方法；MaskMentor 机制仍引 S2。OpenReview forum 出现访问验证，不能据此认定论文不存在。
- 同源补充入口：<https://openreview.net/forum?id=j520wKxnf6>；<https://openreview.net/pdf?id=j520wKxnf6>；<https://openreview.net/pdf/903463b76ae48357e1feb57bbd45e76afc9235c3.pdf>。

<a id="W7"></a>
### W7｜RobustSeg CVPR 2026 官方条目
- 对应论文：PR090。核验范围：**官方检索题名/摘要可见；直开失败**。
- 原始入口：<https://openaccess.thecvf.com/content/CVPR2026/html/Tan_Towards_Robust_Multi-Modal_Semantic_Segmentation_with_Teacher-Student_Framework_and_Hybrid_CVPR_2026_paper.html>
- 支持内容：正式题名、作者 Jiaqi Tan / Xu Zheng / Yang Liu 与 teacher-student / hybrid prototype distillation 定位。
- 限制：具体结构、损失与超参数须锁定 CVPR 版本；不要与 W8 的不同 arXiv 版本拼接。

<a id="W8"></a>
### W8｜与 PR090 相关的同作者 arXiv 多版本记录
- 对应论文：PR090。核验范围：**已检索到相关版本；不自动合并方法**。
- 原始入口：<https://arxiv.org/abs/2505.12861>
- 支持内容：相关预印本题名/模块曾变化：RMMSS；v1 另有 Representation Regularization 表述。
- 限制：本轮未完成版本逐段比对，不能把全部版本的模块都算进 CVPR RobustSeg。只作版本核对入口。
- 同源补充入口：<https://arxiv.org/html/2505.12861v1>。

<a id="W9"></a>
### W9｜Calibrated RGB-D Salient Object Detection 官方条目
- 对应论文：AI017。核验范围：**已检索官方题名/作者/摘要**。
- 原始入口：<https://openaccess.thecvf.com/content/CVPR2021/html/Ji_Calibrated_RGB-D_Salient_Object_Detection_CVPR_2021_paper.html>
- 支持内容：CVPR 2021 深度校正方法身份。
- 限制：具体连续插值和 DCF 机制本轮沿用 S3；未重读原 PDF。

<a id="W10"></a>
### W10｜MoSA 出版商检索入口
- 对应论文：MoSA。核验范围：**官方索引题名可见；页面正文受限**。
- 原始入口：<https://ieeexplore.ieee.org/document/11523456>
- 支持内容：检索题名为 MoSA: Modality-Aware Spatially-Adaptive Adaptation for Multimodal Semantic Segmentation。
- 限制：与附件 RGB-X 题名保留为版本差异；未独立确认该入口与附件 DOI 的完整对应或全部方法细节。
- 同源补充入口：<https://ieeexplore.ieee.org/abstract/document/11523456>。

<a id="W11"></a>
### W11｜UMFNet CVPR 2026 官方条目
- 对应论文：PR029。核验范围：**官方检索题名/摘要可见；直开失败**。
- 原始入口：<https://openaccess.thecvf.com/content/CVPR2026/html/Wang_Uncertainty-Aware_Modality_Fusion_for_Unaligned_RGB-T_Salient_Object_Detection_CVPR_2026_paper.html>
- 支持内容：未对齐 RGB-T SOD 的 uncertainty-aware fusion 方法身份。
- 限制：空间置信与调制细节沿用 S7/S3；不是重新完成 PDF 全文复核。

<a id="W12"></a>
### W12｜SGMA 出版商入口
- 对应论文：PR089。核验范围：**官方检索条目可见；直开失败**。
- 原始入口：<https://ieeexplore.ieee.org/abstract/document/11517493>
- 支持内容：SGMA 题名与不完整遥感多模态分割定位。
- 限制：DOI 仍按 S11/S1 记录；本轮没有独立完成注册元数据与全文核验。
- 同源补充入口：<https://ieeexplore.ieee.org/document/11517493/authors>。

<a id="W13"></a>
### W13｜ECoLaF WACV 2025 官方条目
- 对应论文：AI020。核验范围：**已检索官方题名/摘要**。
- 原始入口：<https://openaccess.thecvf.com/content/WACV2025/html/Deregnaucourt_A_Conflict-Guided_Evidential_Multimodal_Fusion_for_Semantic_Segmentation_WACV_2025_paper.html>
- 支持内容：证据冲突指导的多模态分割身份。
- 限制：本轮不从摘要推定具体 DS 折扣参数；原始机制细节继续引 S2。

<a id="W14"></a>
### W14｜QMF：ICML 2023 / PMLR 原始页面
- 对应论文：AI021。核验范围：**已打开原始论文条目及 arXiv 元数据**。
- 原始入口：<https://proceedings.mlr.press/v202/zhang23ar.html>
- 支持内容：Provable Dynamic Fusion for Low-Quality Multimodal Data；ICML 2023，PMLR 202。
- 限制：理论条件的使用范围按 S2 原审计且需再核定理；不把样本层结果直接当几何关系定理。
- 同源补充入口：<https://arxiv.org/abs/2306.02050>。

<a id="W15"></a>
### W15｜MAGIC arXiv 原始条目
- 对应论文：AI022。核验范围：**已打开题名/摘要条目**。
- 原始入口：<https://arxiv.org/abs/2407.11344>
- 支持内容：为原 ECCV DOI 索引补充原始预印本检索入口。
- 限制：没有在本轮重新核对全部实现与正式版差异。

<a id="W16"></a>
### W16｜SimMLM ICCV 2025 官方条目
- 对应论文：AI026。核验范围：**已检索官方题名/摘要**。
- 原始入口：<https://openaccess.thecvf.com/content/ICCV2025/html/Li_SimMLM_A_Simple_Framework_for_Multi-modal_Learning_with_Missing_Modality_ICCV_2025_paper.html>
- 支持内容：Sijie Li / Chen Chen / Jungong Han；缺失模态学习与动态专家背景。
- 限制：具体 DMoME / MoFe 细节沿用 S3；不能直接推为有噪声模态越多越好。

<a id="W17"></a>
### W17｜CoReFuse-Med arXiv 原始条目与 HTML
- 对应论文：AI025。核验范围：**已读取元数据/摘要，定位原始 HTML**。
- 原始入口：<https://arxiv.org/abs/2609.10261>
- 支持内容：题名、作者、2026-09-09 v1；摘要标注 ACM MM 2026 已接收；主要考虑分辨率退化。
- 限制：ACM DOI 源记录保留；不宣称会议已召开。新增 arXiv DOI 页面标为 pending registration，不能写注册已完成。
- 同源补充入口：<https://arxiv.org/html/2609.10261v1>。

<a id="W18"></a>
### W18｜UMQ arXiv 原始条目
- 对应论文：AI027。核验范围：**已打开题名/摘要条目**。
- 原始入口：<https://arxiv.org/abs/2603.02695>
- 支持内容：补充可定位的原始论文入口。
- 限制：细粒度质量排序/路由机制沿用 P2，不冒称重新逐式核验或确认正式发表。

<a id="W19"></a>
### W19｜PRIME arXiv 原始条目
- 对应论文：AI028。核验范围：**已打开题名/摘要条目**。
- 原始入口：<https://arxiv.org/abs/2608.03475>
- 支持内容：补充可定位的原始论文入口。
- 限制：恢复—复诊细节沿用 P2，不冒称重新逐式核验或确认正式发表。

<a id="W20"></a>
### W20｜ANGA CVPR 2026 官方条目
- 对应论文：ANGA。核验范围：**已检索官方题名/作者/摘要**。
- 原始入口：<https://openaccess.thecvf.com/content/CVPR2026/html/Guan_Anchor-Guided_Gradient_Alignment_for_Incomplete_Multimodal_Learning_CVPR_2026_paper.html>
- 支持内容：Zhi-Hao Guan / Longfei Huang / Yang Yang；正式检索入口已找到。
- 限制：DOI 尚未确认；不猜造 DOI，不另分 AI 号。

<a id="W21"></a>
### W21｜Confidence Propagation 原始条目
- 对应论文：AI029。核验范围：**已打开题名/摘要条目**。
- 原始入口：<https://arxiv.org/abs/1811.01791>
- 支持内容：稀疏深度回归/置信传播历史近邻；恢复 v3 外部文献索引。
- 限制：本轮未重新核网络与公式；不是已移植实验。

<a id="W22"></a>
### W22｜Uncertainty-Aware CNNs for Depth Completion 原始条目
- 对应论文：AI030。核验范围：**已打开题名/摘要条目**。
- 原始入口：<https://arxiv.org/abs/2006.03349>
- 支持内容：深度补全不确定性历史近邻；恢复 v3 外部文献索引。
- 限制：本轮未重新核网络与公式；不是已移植实验。

<a id="W23"></a>
### W23｜多模态语义分割传感器失效 benchmark 原始条目
- 对应论文：AI010。核验范围：**已打开题名/摘要条目**。
- 原始入口：<https://arxiv.org/abs/2503.18445>
- 支持内容：为 AI010 补充原始预印本入口；缺失/噪声模态评价背景。
- 限制：本项目的具体故障表、bootstrap 与阈值不是该文参数。
- 同源补充入口：<https://arxiv.org/html/2503.18445>；<https://ieeexplore.ieee.org/abstract/document/11147584>。

<a id="W24"></a>
### W24｜DFormer 原始条目
- 对应论文：AI004。核验范围：**已打开题名/摘要条目**。
- 原始入口：<https://arxiv.org/abs/2309.09668>
- 支持内容：区分 DFormer 与 DFormerv2；普通强基线来源。
- 限制：未重新核所有训练和预训练版本。

<a id="W25"></a>
### W25｜GeminiFusion 原始条目
- 对应论文：AI005。核验范围：**已打开题名/摘要条目**。
- 原始入口：<https://arxiv.org/abs/2406.01210>
- 支持内容：普通多模态强基线检索入口。
- 限制：未重新核所有训练和预训练版本。

<a id="W26"></a>
### W26｜PR059 原始预印本元数据
- 对应论文：PR059。核验范围：**已核题名/作者/提交日期及预印本 DOI**。
- 原始入口：<https://arxiv.org/abs/2603.07570>
- 支持内容：2026-03-08 v1 与附件标题/作者相符；预印本 DOI 为 10.48550/arXiv.2603.07570。
- 限制：未证实附件 KBS 327 (2025),114107 的正式版对应；两条记录不静默合并。

### 已定位的作者项目/代码入口（不等于代码审计）

| 论文 | 入口 | 来源 | 状态 |
| --- | --- | --- | --- |
| AI023 | <https://geomprompt.github.io/> | W2 | 原始 arXiv 条目链接；未检查代码内容/复现可用性 |
| AI025 | <https://github.com/lrever/CoReFuse> | W17 | 原始论文链接；未检查仓库 commit 或运行 |
| AI026 | <https://github.com/LezJ/SimMLM> | W16 | 原始论文检索结果提供的代码入口；未运行 |
| AI022 | <https://vlislab22.github.io/MAGIC/> | W15 | 原始论文关联入口；未核代码 |
| AI010 | <https://github.com/Chenfei-Liao/Multi-Modal-Semantic-Segmentation-Robustness-Benchmark> | W23 | 原始论文关联入口；未核代码 |
| PR070 | <https://github.com/VCIP-RGBD/DFormer> | W1 | 官方代码仓库入口；已读特定源码，不代表全仓库审计 |

**ConD 代码说明：** W3 摘要仍写接收后公开；本轮没有确认可用的正式代码仓库，不编造下载入口。