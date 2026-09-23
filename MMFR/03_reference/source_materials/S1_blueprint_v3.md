# 面向地下矿山 RGB-D 语义分割的深度几何证据失效与可靠性条件干预研究蓝图
## MMFR v3.0：从失效诊断走向可验证的几何处置

> 修订日期：2026-09-18  
> 研究对象：MUSeg + DFormerv2-S + HAM decoder。  
> 当前阶段：A2 v3 主验证已完成；正式配对 clean-control、下一模块 Oracle 和完整方法验证仍待完成。  
> 资料基础：生成蓝图提示词、上一版 v2.2 蓝图、A2 结果报告、PR/RE 补全文献、三组防撞审计及最新整理清单。  
> 证据约定：**[项目事实]** 指所提供项目文件的记录，不代表本次重新运行；**[文献依据]** 指论文或所提供全文审计支持的内容；**[外部核验]** 指本次核对的官方资料；**[本版建议]** 指待执行的设计，不是既有结果。  
> 重要限制：本次没有访问你的本地模型代码、训练日志原件、checkpoint 或数据集，也没有执行模型实验。公开官方实现不能自动代表本地 fork。  
> 编号：沿用 RE、PR、AI 编号；本次新增机制近邻使用 AI029、AI030。所有公式使用 Markdown 的 `$...$` / `$$...$$` 定界符。

---

# 0. 本版的总判断

**现在不应把研究主线写成“再设计一个可靠性感知融合模块”，也不应预先承诺一个同时解决所有 Depth 故障的双模块架构。**

更合适的研究问题是：

> **局部无效或空间错位的 Depth，究竟如何改变 DFormerv2 内部的几何关系；能否在保留可靠几何和 RGB 主路径的前提下，修正这些关系的使用方式，并证明收益确实来自正确的诊断与处置？**

执行顺序确定为：

**冻结现状与补齐对照 → 核对几何计算 → 两类 Oracle → 选择最小模块 → 匹配训练与反事实对照 → 独立确认。**

当前优先级不是按照概念新颖程度，而是按照已有证据和可检验性排列：

| 优先级 | 问题 | 当前证据 | 下一步 |
| --- | --- | --- | --- |
| 主线优先 | 局部 Depth 无效是否制造错误的成对几何关系 | spatial dropout 降幅明显，位置与有效性可明确记录 | 先做 validity-aware pairwise geometry Oracle |
| 条件性第二方向 | 有效 Depth 位于错误坐标是否误导几何关系 | misalignment 产生明确降幅 | 先做带有效支持域约束的 inverse-shift Oracle |
| 单独保留 | 整模态缺失后的无深度运行能力 | entire missing 是当前最大降幅 | 保留评价；不能假定继续抑制 Depth 就能改善 |
| 监测项 | Gaussian noise 与复合退化 | 有一定下降 | 检查主模块能否自然受益，不先加专用组件 |
| 暂不专门开发 | 当前强度下的 blur、quantization | 分割指标变化很小 | 保留冻结协议，不为“制造困难”事后改变强度 |

以上排序来自当前 A2 开发结果，不是对所有模型、数据或退化强度的普遍结论。[项目-A2]

本版将最新报告的候选 C1、C2 分别记为 **B1-V** 和 **B1-A**；正式论文确认阶段记为 **P-confirm**，避免与旧蓝图中“C1 论文确认阶段”混淆。

---

# 任务一：从参考文献提炼研究配方

## 1. 核心问题与贡献的共同结构

这些文献并不研究完全相同的任务，但可抽象出共同的问题：

> **系统通常按照“输入证据可用且可信”的假设设计；当证据质量随样本、位置或环境变化时，固定的信息使用方式可能失效。**

对应的贡献不只是“加入某个模块”，而是重新建立：

**证据状态 → 使用规则 → 任务结果**

之间的关系。

质量驱动融合可以参考 QMF、ECoLaF、SGMA、UMFNet；缺失模态适应可以参考 MaskMentor、Condition Dropout、RobustSeg；任务与恢复不完全一致可以参考 GeomPrompt、CLP 与矿下增强研究；质量代理和任务可靠性必须单独检验，可以参考 RE049、RE094、RE418、RE452。[AI019][AI020][AI021][AI023][AI024][PR029][PR040][PR089][PR090][RE049][RE094][RE418][RE452]

这不是说每篇论文都完成了下列全部证据链，而是将它们的研究范式及其证据缺口合并为本项目的操作框架。

## 2. 中心假说的通用句式

> **如果一种外部状态通过可定位的内部机制导致任务退化，那么，对该状态进行独立表征，并在这一机制上采取状态条件化干预，应当比无条件处理、固定处理和错误条件处理更有效；这种优势应在匹配资源与独立评价中保持。**

需要分别检验三个命题：问题存在；机制确实可干预；状态诊断比简单规则提供了额外价值。不能因为最终指标上涨，就默认三个命题全部成立。

## 3. 三阶段研究范式

| 阶段 | 抽象方法论 | MMFR 的具体落地 | 主要溯源 |
| --- | --- | --- | --- |
| 问题提出与初步验证 | 受控扰动、配对观察、失败分层 | 同一图像、同一 checkpoint、不同 Depth 状态；区分缺失、错误位置和有值失真 | AI010、AI011、PR040、项目-A2 |
| 核心分析与假说检验 | 单变量干预、理想信息对照、资源匹配 | Oracle-A/B；augmentation-only；observed-validity rule；predicted/constant/generic control | PR070、AI016、AI019、AI023、AI024、PR042、PR090 |
| 机制探索与深化 | 中介量追踪、反事实替换、外部复验 | 几何偏置失真、关系有效性、局部 utility、shuffled/inverse maps、held-out corruption | AI021、AI025、PR029、PR089、MoSA；具体组合为本版建议 |

---

# 第一部分：研究基本盘——问题、假说与边界

## 1. 研究背景与核心冲突

### 1.1 已有基础

MUSeg 为本研究提供地下矿山 RGB-D 语义分割数据：3,171 对 RGB/Depth 图像，覆盖六个矿区、15 类语义标注。它支持矿下场景研究，但不能因此把人工注入的 Gaussian noise、平移或 dropout 自动解释为真实矿尘、真实传感器故障分布。[RE326]

DFormerv2 将 Depth 用于构造几何先验，而不是把 Depth 作为与 RGB 对称的独立语义编码流。这决定了本研究应追踪 **Depth → 成对关系 → 几何偏置 → RGB token 交互**，而非默认存在一个可以直接加权的独立 Depth feature branch。[PR070]

矿下多传感器研究已经使用局部置信度、物理质量指标和自适应权重。**“在矿下使用可靠性”“根据质量调整模态贡献”均不能单独承担新颖性。**[RE240][RE026][RE094][RE452]

### 1.2 真正的核心冲突

本课题的矛盾不是“有 Depth 比没有 Depth 好不好”，而是：

> **同一个几何机制，在输入关系可信时可能有益，在关系不成立时可能有害；但是数值有效、传感质量、空间对齐、几何关系正确和语义任务有用，并不是同一个概念。**

因此需要避免三个跳跃：

- 从“见过 corruption”直接跳到“识别了 corruption”；
- 从“预测了质量图”直接跳到“采取了正确动作”；
- 从“改变了几何计算”直接跳到“获得了独立算法创新”。

## 2. 当前 A2 提供了什么证据

### 2.1 已完成事实

A2 v3 已完成 4 个候选 checkpoint、每个 10 个冻结条件、每条件 318 个 val-dev 样本的正式开发评价，共 40/40 个 evaluation units。official test 未使用。epoch-420 是当前主开发 checkpoint；390、370、latest 保留用于 checkpoint 敏感性，不再作为继续寻找好结果的自由度。[项目-A2]

| 条件 | mIoU | 相对 clean 变化，百分点 |
| --- | ---: | ---: |
| clean | 57.06 | — |
| entire_missing@1.0 | 52.55 | -4.51 |
| spatial_dropout@0.75 | 54.42 | -2.64 |
| misalignment@0.75 | 55.80 | -1.26 |
| gaussian_noise@0.75 | 56.48 | -0.58 |
| quantization@0.75 | 56.99 | -0.07 |
| blur@0.75 | 57.06 | 0.00 |
| spatial_dropout@0.5 + gaussian_noise@0.5 | 55.15 | -1.91 |
| blur@0.5 + misalignment@0.5 | 56.45 | -0.61 |
| quantization@0.5 + misalignment@0.5 | 56.47 | -0.59 |

主指标为六个单失效条件的等权平均，当前为 **55.55**。九个受损条件平均 **55.7078**、十条件平均 **55.8430** 是不同汇总口径，不能替代主指标。[项目-A2]

### 2.2 不可越过的解释边界

B0 是历史 **RGB-D DFormerv2-S baseline**，不是 RGB-only，也不是当前 A2 的正式 clean-control。B0 历史 clean 为 58.79，A2 为 57.06，差 1.73 个百分点；目前只能描述该差距，不能归因为 corruption training 或可靠性辅助任务。[项目-A2]

A2 的 reliability estimator 尚未控制 segmentation/geometry。当前资料也没有提供可用于认定“质量估计已经泛化、校准良好”的完整独立指标。因此，A2 是 **failure-aware training + parallel depth-reliability estimation 原型**，不是已证明有效的 reliability-conditioned segmentation 方法。[项目-v2][项目-A2]

如果现有梯度隔离记录确认辅助损失不进入分割主干，就不能把 A2 的分割结果解释为“可靠性辅助任务改善了语义表征”。下一阶段应核验该隔离契约，而不是预设辅助监督存在正则化收益。

**52.55 不是缺失 Depth 下的性能上限。** 它只对应当前参数、当前预处理和当前零深度输入条件。全缺失可能需要更好的无深度运行方式，但不能由此直接决定添加第二网络、恢复分支或大型 MoE。[项目-A2]

blur/quantization 的输入 hash 变化仅证明扰动执行；小降幅不能证明这些故障普遍无害，也不足以证明具体几何失真大小，后者需要直接测量。[项目-A2]

## 3. 必须分开的量

设未额外注入退化的参考 Depth 为 $D_0$，实际输入为 $D_c$，RGB 为 $I$。

| 符号 | 定义 | 不应等同于 |
| --- | --- | --- |
| $V_D^{\mathrm{final}}$ | 最终 Depth 的有效性状态 | 数值存在就一定正确 |
| $Q_D^{\mathrm{syn}}=V_D^{\mathrm{final}}\odot R_D^{\mathrm{syn}}$ | 当前合成协议定义的连续质量目标 | 真实物理传感器质量真值 |
| $\hat r_D$ | 对该质量目标的预测 | attention weight、语义正确率 |
| $\rho_{ij}^{\ell}$ | 第 $\ell$ 级两端点形成几何关系的使用系数 | 已校准的联合正确概率 |
| $U_D^{\mathrm{geo}}$ | 指定模型和干预定义下，Depth 几何项的任务边际效用 | 单模态准确率或质量标签 |
| $G_A$ | 执行某个动作相对指定基线的任务收益 | Oracle 信息本身的价值 |

沿用项目中的 $Q_D$ 写法时，正文应明确它首先是 **corruption-grounded supervision target**。自然有效的 Depth 仍可能含误差；合成 clean 不意味着物理完美。[项目-v2][防撞-P0][防撞-P1][防撞-P2]

不能强制要求所有类别、位置和退化类型中 $\hat r_D$ 与 utility 都严格单调。例如，在实际几何输入层给所有有效深度加上同一个常数 $b$，有：

$$
|(d_i+b)-(d_j+b)|=|d_i-d_j|.
$$

像素数值改变并不必然改变这种成对关系。这是对关系算子的数学观察，不是本项目新增实测结论。质量与 utility 的联系必须按故障类型、强度和区域分别检验。

## 4. 核心科学问题与可证伪假说

### Q1 / H1：无效观测是否产生了可避免的错误关系？

**假说 H1-V：** 在局部 Depth 缺失时，经过当前预处理与尺度变换的无效值，可能参与成对距离计算，形成不应存在的深度几何偏置。保持其他路径不变，仅去除这些无效关系对应的 Depth 项，可恢复部分分割损失。

证伪条件：代码和张量追踪显示相关 Depth 项并未异常，或者按正确坐标和有效支持域实现的 Oracle 在目标条件下仍无实际收益。

### Q2 / H2：错位是否具有可回收的几何损失？

**假说 H1-A：** 对 misalignment，核心问题是观测位置错误，而不仅是无效。对可恢复支持域执行正确的坐标补偿，可能恢复一部分几何作用。

证伪条件：正确 inverse-shift 已通过坐标单元测试，仍无法恢复具有实际意义的性能；或收益完全来自另一个被同时改变的预处理步骤。

### Q3 / H3：学习式诊断是否超过简单可观察规则？

**假说 H2：** 在同样的几何动作和训练条件下，预测的连续可靠性可比已知 invalid sentinel / validity flag、固定几何强度和全局质量系数提供额外收益，尤其针对“仍有数值但已失真”的 Depth。

证伪条件：可观察 validity 规则已经复制全部收益，或者把 $\hat r_D$ 换成均值、空间错配图以后效果不变。

这时可以保留有效的简单规则，但必须降低“学习式 reliability 必要”的贡献表述。

### Q4 / H4：收益能否归因于正确机制并迁移？

**假说 H3：** 收益在固定输入、相同参数预算和相同 corruption exposure 下成立；主要发生于原几何项负效用或关系失真的区域，并在独立训练 seed、真正未见故障或外部数据集上保持。

证伪条件：只有额外训练步数、特定 checkpoint、单一合成模式或参数增量能解释改善。

## 5. 预期贡献与研究范围

若上述假说获得支持，贡献限定为三项：

**贡献一：失效机制与评价。** 解释局部无效、空间错位和整模态缺失的不同机制，并给出可复核的矿下 RGB-D Depth 失效评价。不能将 synthetic stress test 声称为真实传感器失效率评价。[RE326][AI010][PR040]

**贡献二：几何证据条件化使用。** 在明确的 DFormerv2 Depth-derived relation 上建立有效性/可靠性条件干预，并与简单规则、全局强度和通用适配器比较。[PR070]

**贡献三：诊断与动作分离的证据链。** 用 Oracle、预测、常数、空间错配和 geometry utility 分开验证“是否知道状态”和“知道后是否处置正确”。该组合是本研究拟建立的证据，不声称反事实或可靠性校准本身首次出现。[AI021][AI025][PR042][RE049]

主体仍为二维语义分割、Depth 失效和轻量几何干预。RGB 增强、完整 Depth 恢复、三维重建、SLAM、安全决策认证、大型 MoE 均不列为必做贡献。B1-V/B1-A 是否组合由 Oracle 决定，而不是预先写死。

## 6. 文献防撞边界

| 已有方法族 | 已被覆盖的内容 | MMFR 必须正面区分 |
| --- | --- | --- |
| D3Net、CCF、Calibrated RGB-D | Depth 质量判断、丢弃、路径选择、连续校正 | 不是再次发现“坏 Depth 应少用” |
| MaskMentor、Condition Dropout、RobustSeg | 缺失暴露、适应、蒸馏；部分覆盖 noisy 评价 | 同 corruption exposure 对照不可缺 |
| ECoLaF、SGMA、UMFNet、MoSA | 像素/空间可靠性与局部融合，MoSA 还覆盖空间 adapter | 不能用“空间可靠性图”或 generic gate/adapter 作为主创新 |
| GeomPrompt、CoReFuse-Med | task-driven raw-Depth repair、内部特征去噪与重加权 | 不能仅将修复位置换一下就宣称新机制 |
| UMQ、PRIME | 质量监督、增强/路由、诊断—恢复—复诊 | 合成质量监督与闭环处理本身不新 |
| RE240、RE026、RE094、RE452 | 矿下/物理质量驱动动态使用模态 | 矿下场景不能补偿结构同构 |
| AI029、AI030（本次补充） | 稀疏深度的有效性/置信传播、输入噪声置信估计 | mask-aware aggregation、confidence propagation 不能独立宣称新颖 |

对应具体文献见文末。前三组防撞审计的结论仍适用；本版并未获得“全球不存在相似工作”的保证。[AI015–AI030][PR029][PR089][PR090][MoSA][RE026][RE094][RE240][RE452]

**仅把 $\rho_{ij}=r_ir_j$ 放入某个 backbone，不足以保证有强创新。** 需要证明其针对真实可定位的算子失败，比普通去无效、标量降权和通用适配更有必要性。

---

# 第二部分：核心实验方案——从设计到执行

## 1. 研究设计与变量

| 类型 | 本项目定义 |
| --- | --- |
| 自变量 | Depth 故障类型/强度；训练是否见过故障；诊断信号来源；动作是否改变 Depth 几何；动作粒度/插入级；有无坐标补偿 |
| 主要因变量 | 六单失效宏平均 mIoU、clean mIoU、逐条件 mIoU、最坏单条件；B1 相对匹配 A2/augmentation-only 的增量 |
| 诊断与机制量 | $Q_D$ 预测误差、invalid detection、几何偏置误差、$U_D^{\mathrm{geo}}$、$G_A$、正确/错配条件信号的效果差 |
| 控制变量 | 数据分组、预训练、训练预算、batch/学习率/优化器、RGB/Depth 预处理、随机输入实例、checkpoint 规则、主 evaluator、指标实现 |
| 泛化变量 | 独立训练 seed；真正未见的 corruption family；明确位于训练范围外的 severity；外部 RGB-D 数据集 |

不能一次同时改变模块、数据增强分布、训练时长和选择规则，再把总变化归因为一个新模块。

## 2. 冻结项目账本

| 项目 | 沿用规则 |
| --- | --- |
| 开发划分 | train-dev 1,277；val-dev 318 |
| 官方划分 | train 1,595；test 1,576；test 在设计冻结前 `sealed_unread` |
| 训练基础 | 当前相同 pretrained、优化器/调度器、学习率 $6\times10^{-5}$、500 epochs、batch size 10 |
| 第一配对 seed | 772961337；其他确认 seed 在开跑前登记 |
| A2 exposure | 25% clean、75% Depth corruption；六类故障；curriculum 精确边界从现有配置导出 |
| checkpoint selector | 沿用 clean-only selector；保留 top-3 + latest 与既有并列规则，不用 corruption 反选 |
| 主评价 | 现有冻结 Main-Val 配置；不能擅自用 selector 的推理配置代替 |
| 六单失效 | entire missing=1.0；其余单故障=0.75 |
| 三复合失效 | 现有类型组合和顺序、severity=0.50；不重新抽取更有利的组合 |
| 开发统计 | 196 个 location group，10,000 次 paired percentile bootstrap |
| 当前参照 | A2 epoch-420；B0 仅 historical reference |

这些数值来自项目已有方案，不是本次从论文摘要推定的“推荐超参数”。特别要区分 clean checkpoint selector 和 reliability estimator：两者不是同一个模块，也不是同一种“selector”。[项目-v2][项目-A2]

## 3. 主要指标及比较关系

六单故障主指标：

$$
M_{\mathrm{single}}
=\frac{1}{6}\sum_{c\in\mathcal C_{\mathrm{single}}}\operatorname{mIoU}_c.
$$

最坏单故障：

$$
M_{\mathrm{worst}}
=\min_{c\in\mathcal C_{\mathrm{single}}}\operatorname{mIoU}_c.
$$

主比较必须分两层：

$$
\Delta_{\mathrm{exposure}}
=M_{\mathrm{single}}(\mathrm{A2/aug})
-M_{\mathrm{single}}(\mathrm{clean\ control}),
$$

$$
\Delta_{\mathrm{action}}
=M_{\mathrm{single}}(\mathrm{B1})
-M_{\mathrm{single}}(\mathrm{matched\ A2/aug}).
$$

前者检验训练暴露，后者检验干预的额外价值。总体 MMFR 相对 clean-control 的改善不能替代 $\Delta_{\mathrm{action}}$。

同时报告 clean、六单条件、三复合条件、每类 IoU 和实际 label map 中预登记的关键类别。不要凭场景想象新增“异物”类别。mAcc、mF1、boundary 指标和相对保持率均为辅助结果，不替代绝对 mIoU。[RE326][PR040][RE266][项目-v2]

---

## 步骤一：数据、标签与计算路径预处理

### S1.1 数据与 manifest 固化

**目的：** 让不同方法处理完全相同的样本、故障实例和评价区域。

**方法依据：** 同图像受控故障、逐条件报告参考 AI010、AI011、PR040；数据分组参考 RE326；具体字段沿用项目并补全。[AI010][AI011][PR040][RE326]

**操作：**

1. 固化 train/val/test 文件列表、location group 映射和 SHA-256；同一位置或同一原始图的所有衍生版本不得跨划分。
2. 保存每条样本的 `sample_id`、`location_group`、RGB/Depth/label 路径或哈希、corruption seed/type/severity、真实执行参数、复合顺序、validity 与 $Q_D$ 生成版本。
3. 对 train/dev 的既有标签和图像读取进行断言；不要为了做审计提前读取封存 test 的图像/标注。
4. eval 中采用固定 per-sample corruption seed 或冻结 manifest，不依赖 batch 顺序重新抽噪声。
5. 不改变 A2 已冻结的 corruption。发现生成器或标签错误，应另建协议版本并重做受影响比较，不能悄悄改后继续沿用原结论。

**新增建议参数：** 先固定 32 个开发样本做管线单元测试，再跑完整 318 个 val-dev；32 只是本版排错规模，不是文献参数。

**参数来源声明：** 划分和 corruption 比例来自项目；32 个样本、哈希及字段细节为本版工程建议，非上述论文摘要中的配置。复现某篇 benchmark 时，仍须核对其全文与官方配置。

**产出：** `data_manifest`、`corruption_manifest`、`group_mapping`、`protocol_hash`。

### S1.2 最终状态标签与可部署输入分离

**目的：** 避免把仅训练/审计时知道的信息泄漏给部署模型。

**方法依据：** corruption-grounded 监督的边界参考 PRIME，task-derived reliability 的不同含义参考 MoSA、SGMA；具体 $Q_D$ 沿用项目定义。[AI028][MoSA][PR089][项目-v2]

**操作：**

1. 保留自然 invalid 和合成 invalid 的来源，但最终有效状态统一记录为 $V_D^{\mathrm{final}}$。
2. 检查噪声不会“复活”自然 invalid；blur 保持既有 mask-normalized 生成规则；misalignment 同步搬运 Depth 与 validity。
3. 明确 $R_D^{\mathrm{syn}}$ 到底来自哪一阶段的状态，复合退化是如何合成的；不得不加说明就改用 $1-\mathrm{severity}$ 作为所有类型的真实质量。
4. 将模型可用信息列为一张白名单：当前 RGB、当前 Depth、部署传感器实际提供的 validity flag，以及仅由这些数据计算的统计量。
5. 原始 clean Depth、合成 corruption mask、真实 shift、真实 severity、$Q_D$ 和 segmentation GT 只能用于监督或 Oracle，不能进入普通测试前向。
6. 若 invalid sentinel 或 validity flag 本来就可从实际输入取得，必须建立**可部署的确定性 validity baseline**；此时不能把“知道零值无效”包装成学习式诊断能力。
7. 核对 A2 的 14 维输入和 2 维输出各自含义，沿用真实读出与损失，不能根据“2”自行推断为 RGB/Depth 两个可靠度，也不能擅自解释成 validity/quality 双头。
8. 对“有值但错误”“无值”“有效但错位”分别统计。移位后仍有效的像素，不应仅因 validity=1 就被认为几何正确。

**参数来源声明：** $Q_D$ 的当前实现应从项目代码导出；上述信息白名单和分组分析为本版控制设计，不是 PRIME 或 MoSA 的复现协议。项目文件没有给出的系数、阈值和通道定义保持待核，不补造数值。

**产出：** `target_definition.md`、`inference_input_contract.md`。

### S1.3 核对真实几何入口——本版的重要实现修正

**目的：** 不让论文级公式与实际干预对象错位。

**外部核验：** 本次读取的官方 `models/encoders/DFormerv2.py`，在 `GeoPriorGen` 中以双线性插值调整 Depth 尺度，分别产生位置项和 Depth 项，并在 GSA 的 softmax 之前加入几何偏置。其旋转位置处理和局部位置增强仍属于注意力实现的一部分。[PR070-代码]

因此本版使用下列**计算抽象**，而不把附件中的未归一化后乘形式直接当成代码：

$$
B_{\mathrm{geo}}^{\ell,h}
=w_s^\ell B_S^{\ell,h}+w_d^\ell B_D^{\ell,h},
$$

$$
A^{\ell,h}
=\operatorname{Softmax}\left(L^{\ell,h}+B_{\mathrm{geo}}^{\ell,h}\right),
$$

其中 $L$ 表示实际实现中的视觉相似度 logits，$B_D$ 为含 head decay 的深度关系偏置。公式省略 block 下标，实际必须核对每个 block。

**操作：**

1. 保存本地 git commit、文件 hash、checkpoint hash；检查是否与公开实现一致。若不一致，以实际产生当前 checkpoint 的实现为准，单独说明差异。
2. 核对 Depth 单位、归一化、invalid sentinel、实际 resize 算子、head broadcast、各 stage 与 block 的调用。
3. 在第一次候选实验中只改 $B_D$，保留 $B_S$、RGB token、Q/K/V、decoder、位置处理与原 attention 归一化。
4. 导出 checkpoint 中位置/深度混合权重的值与符号；不要未经检查就假设学习后它们必为正，或“距离越大必然抑制越强”。
5. 前三级按实际横/纵分解构造关系系数，第四级按其完整关系实现；不能把前三级强行改为完整 $N\times N$ 注意力。
6. 区分三种诊断：仅关闭 Depth 几何项；关闭位置+Depth 总几何偏置；输入全零 Depth。三者不是同一个对照。
7. 全几何偏置关闭后，也不能直接称其为独立训练的 RGB-only 模型或完全普通 Transformer。

**数值测试建议：** `no-action` 先走原始分支；要求输出与既有 evaluator 一致。显式全一关系系数用 FP32 检查 `atol=1e-6, rtol=1e-5`，超限先排查算子路径、精度和非确定性，不立即放宽阈值。该容差是本版工程起点，最终需按原环境的无改动重复误差登记。

**参数来源声明：** 公开代码事实不替代本地核验；容差及测试流程为本版建议。本文不宣称已经对你的本地模型完成了上述测试。

**产出：** `geometry_path_audit.md`、输入/输出张量断言和最小修改 diff。

### S1.4 有效性和可靠性的尺度运输

**目的：** 避免 Depth 经过一种采样方式，可靠性图却表示另一片区域。

**方法依据：** 几何尺度对应参考 PR070；带有效性/置信度的聚合与传播有 AI029、AI030 等先例，不作为独立创新。[PR070][AI029][AI030]

**操作：**

1. 从实际 Depth resize 操作导出每个 token 的采样支持域。
2. Oracle-A 首轮保持 Depth resize 不变，仅构造两种预登记关系证据：
   - **严格支持域有效性：** 该 token 所有非零权重来源均有效时记为有效；
   - **有效覆盖率：** 对 validity 使用同一采样权重，输出 $[0,1]$ 的支持程度。
3. 先固定两者，不再根据 val 结果任意搜索十几种 mask pooling 规则。
4. 对连续 $\hat r_D$ 同步记录尺度映射方式；任何独立阈值都只能由开发数据确定。
5. mask-normalized 的**模型内** Depth resampling 可作为另一个单独消融，不能与关系干预同时偷偷加入。它不同于生成 blur corruption 时已经冻结的 mask normalization。
6. 构造“一个有效值邻近一个 invalid 值”的人工张量，检查是否因插值产生虚假中间深度；同时保存对应的粗尺度 validity。

**参数来源声明：** 两种支持域方案是本版候选设计；不是 AI029/AI030 的官方实现参数。最终方案须通过本地采样算子的数值测试确定。

**产出：** `scale_support_audit` 与“仅关系干预 / 仅采样修正 / 二者组合”的清晰开关。

---

## 步骤二：核心假说验证——先 Oracle，再学习式模块

### S2.1 建立参照，补齐归因条件

**目标：** 分开历史开发参照、训练暴露效应和新动作效应。

**方法依据：** missing/corruption exposure 本身即可改变鲁棒性，参考 MaskMentor、Condition Dropout、RobustSeg；不能把训练策略收益算到 reliability action 上。[AI019][AI024][PR090]

**操作：**

1. 将历史 B0 checkpoint 放入现有冻结 10-condition evaluator，禁止继续训练或改变输入，使其成为 `historical stress reference`。
2. B0 的报告注明旧训练代码、checkpoint、预处理及兼容加载情况；不把它升级为正式配对 control。
3. 按 v3 代码路径执行 `MMFR-A2-clean-control-v3`，仅改变预登记的故障暴露条件，其他主训练条件与 A2 配对。
4. clean-control 的辅助头处理必须预先明确。若梯度隔离成立，可保留同一辅助计算框架以避免流程差异；分割侧不得受到辅助 loss、全局梯度裁剪、AMP scaler 或 RNG 顺序的意外影响。
5. 为证明“辅助头不改变分割”而做的单元测试，应比较固定 batch 下的分割 logits、分割参数梯度及一次 optimizer update，而不只检查计算图里是否调用了 `detach`。
6. 保存 epoch-420 的选择依据。它可以作为既定开发起点，但其合法性应来自 clean-only 规则，而不是“在 10 个 corruption 条件都获胜”。本版不授权事后更换 selector。
7. clean-control 训练与 Oracle 诊断在工作计划上可并行；最终归因必须等配对证据齐全，不能以“不阻塞模块设计”为由省略 control。

**参数来源声明：** 500 epochs、batch 10、学习率和 seed 沿用项目；本版不提供新的 clean-control 优化参数。上述隔离检查为工程验证，不是所引论文的既有实验结论。

**产出：** B0 历史压力曲线、clean-v3 配对配置、辅助损失隔离记录。

### S2.2 Oracle-A：局部无效 Depth 的成对几何处置

**目标：** 排除“诊断再准确也没有可用动作”的情况。

**方法依据：** 几何入口来自 PR070；低质量 Depth 的选择和理想信息对照可参考 D3Net；普通质量门控已有 QMF、SGMA、UMFNet，故本实验限定在 Depth 几何关系上。[PR070][AI016][AI021][PR029][PR089]

#### A. 固定动作定义

先用无新增网络的最简规则：

$$
\rho_{ij}^{\ell}=r_i^{\ell}r_j^{\ell},
$$

$$
\widetilde B_{\mathrm{geo},ij}^{\ell,h}
=w_s^\ell B_{S,ij}^{\ell,h}
+w_d^\ell \rho_{ij}^{\ell} B_{D,ij}^{\ell,h}.
$$

这里的乘积仅为端点共同支持几何关系的候选使用规则，**不是已经证明的最优关系，也不自动构成两点联合可信概率**。第一次 Oracle 不改 RGB token，不把无效 Depth 所在位置的语义 token 删除，不修复 raw Depth。

其两个边界为：

- $\rho_{ij}=1$：保留原几何偏置；
- $\rho_{ij}=0$：只去掉该关系的 Depth 项，保留位置项。

对加性 log-bias 而言，“不使用 Depth 关系”的中性值是 **0**；不能误设为负无穷，也不能把整个 attention 输出乘到 0。

#### B. 固定输入与比较组

使用 A2 epoch-420，不训练。首轮评价 clean、spatial_dropout@0.75、entire_missing@1.0。

| 组别 | 处理 | 作用 |
| --- | --- | --- |
| A0 | 原始 no-action | 对照 |
| A1 | 严格支持域 validity → pair rule | 检验无效几何关系 |
| A2-O | 有效覆盖率 → pair rule | 检验边界与尺度聚合敏感性 |
| A3-O | 真实合成 $Q_D$ → 同一 pair rule | 理想质量信息实验，不是理论上界 |
| A4 | 整体关闭 Depth 几何项，位置项保留 | 判断局部选择是否超过永久不用 Depth |
| A5（诊断用） | 同时关闭位置与 Depth 总偏置 | 检查错误中和位置先验的影响，不作为主候选 |

如果 validity 在部署时可观察，A1/A2-O 中对应的信息不必称为 privileged Oracle，应同步标注为“observed-validity rule”。A3-O 的合成质量目标才包含通常不可直接获得的额外信息。

#### C. 关键输出

保存三个条件的 mIoU、逐类 IoU、逐组混淆矩阵，以及 invalid-related pair 的数量、原始/修改后的 Depth bias 和 attention 变化。先用 32 个固定样本完成断言，再跑全 318 张；不能只凭样例图判断成功。

当输入在实际几何路径上为常数深度时，原 $B_D$ 应为 0；此时“仅抑制 Depth bias”的 A1/A3 不应凭空改变缺失条件的输出。如果变化明显，先检查是否意外改变了位置项、mask、归一化、其他参数或数据路径。

clean 中可能本来就存在 natural invalid，因此“全一 $\rho$ 精确恢复基线”的单元测试，不等于自然 clean 样本经过真实 validity rule 后一定不变。二者都要测试。

#### D. 继续／停止条件

**本版建议的开发筛选门槛：** spatial dropout 的增益至少达到 +0.50 个百分点，配对 group-bootstrap 区间下界大于 0，clean 损失不超过 0.50 个百分点，且数值边界测试通过，再优先进入学习式 B1-V。该 +0.50 是新增研发筛选标准，不替代已冻结的论文 primary gate。

若差值为小幅正值但区间跨零，标记为证据不足，不直接宣布失败或成功；优先检查支持域和原权重，而非扩展网络。若正确实现的两个预登记聚合版本都无收益，暂停**当前冻结 checkpoint 上这一类直接插入动作**。

这不能证明所有 geometry learning 都无效。若有明确的分布适应迹象，可登记一次有限的匹配短训练作为例外检验；不能在连续失败后无限改变定义，直到找出正数。

**参数来源声明：** pair 规则、候选组、+0.50 开发门槛是本版建议；条件和样本数来自项目，10,000 次/196 组沿用冻结统计。它们不是 D3Net 或 PR070 摘要给出的实验参数。

### S2.3 Oracle-B：错位深度的几何坐标补偿

**目标：** 测量“错位本身可逆的部分”能恢复多少，避免提前训练对齐网络。

**方法依据：** 未对齐模态会误导融合可参考 UMFNet；将实验局限于当前几何入口来自 PR070 和项目 A2 报告。以下 inverse-shift 及支持域对照为本版设计。[PR029][PR070][项目-A2]

**操作：**

1. 固定 epoch-420、原始 RGB 和标签，从 frozen corruption manifest 读取真实 shift。
2. 先用带坐标编码的人工图确认 shift 正负方向、单位、padding 和 resize 顺序；区分输入分辨率像素与 token 分辨率。
3. 对 Depth 及 validity 做相反变换，只改变送入几何生成过程的证据坐标，不改变 RGB/label。
4. 分别跑 clean/no correction、misalignment/no correction、misalignment/true inverse；增加一个方向错误的 shift 作为坐标实现 sanity check。
5. 对每个样本输出可恢复支持域。经过裁剪、边界填充或插值后，inverse-shift **不等于完美恢复 clean Depth**；禁止用循环平移将移出画面的内容绕回，禁止用 clean Depth 偷补丢失边界。
6. 全图 mIoU 仍是主结果。额外报告同一个共同支持域上的误差；所有比较使用同一支持域，不为某个方法挑选更有利区域。
7. 可单独增加“直接使用 clean 几何”的 privileged reference，标为理想参照，不与 inverse-shift 或可部署方法混写。
8. 若可回收损失明确，再考虑低自由度 offset 或少量粗网格 offset；第一轮不做全分辨率光流，不同时增加重建、confidence 网络和蒸馏。

**重要防撞提醒：** 如果 Depth 在整个网络只通过几何分支进入，那么“先 warp Depth 再生成几何”与“在几何入口 warp Depth”可能数学等价。仅把代码移到 backbone 内部，不足以区别于已有配准方法。进入 B1-A 时必须对照相同自由度的普通 warp，并解释关系有效性/支持域处理带来的额外价值。

**继续条件：** true inverse 在实际可恢复区域和全图上显示一致、具有实际意义的改善；可沿用 +0.50 个百分点作为本版开发筛选起点，但必须同时报告当前 misalignment 总损失只有 1.26 个百分点这一事实。不要强求其达到超出可回收空间的固定改善幅度。

**参数来源声明：** shift 数值、插值和 padding 从现有 generator 导出，本版不发明像素偏移范围；+0.50、共同支持域和错误方向对照为研发建议，非 PR029 的实验配置。

### S2.4 Oracle 后的决策

| Oracle-A | Oracle-B | 研究决策 |
| --- | --- | --- |
| 明确有收益 | 无明确收益 | B1-V 为主；错位先作为未解决条件报告 |
| 无明确收益 | 明确有收益 | 转为 B1-A 可行性研究，并补普通 warp 的同自由度对照 |
| 二者明确有收益 | 二者明确有收益 | 先完成 B1-V；B1-A 作为独立增量，最后验证是否真有互补 |
| 二者均无明确收益 | — | 暂停当前门控/对齐方案；审查无深度运行、表征适应等替代假说，不继续机械堆模块 |

**不预先承诺最终一定有两个组件。** 一个有清楚机制证据的简单方法，优于两个靠联合训练掩盖无效性的模块。

### S2.5 从理想信息到预测信息：先接现有头，再决定是否训练

**目标：** 把 action gap 和 diagnosis gap 分开。

**方法依据：** DCF、SGMA、UMFNet、MoSA 已经表明质量/置信度控制是成熟路径；MMFR 必须证明自己的状态预测在特定动作中真正有额外信息。[AI017][PR029][PR089][MoSA]

**操作：**

1. 在 Oracle-A 固定动作上直接接入 epoch-420 现有 $\hat r_D$，不先重训。
2. 同时运行可观察 validity、oracle $Q_D$、预测 $\hat r_D$、按图像/尺度取均值的常数图。这样能直接分辨“动作有效但预测不够”与“预测已足够”。
3. 按 invalid、valid-but-corrupted、clean-valid 三组报告 $Q_D$ 的 MAE/RMSE 和偏差；不要让大量易识别零值掩盖非零错误。
4. 按 corruption 类型与 severity 分别评价，避免只报告跨类型总相关系数。
5. 对具有明确二值事件定义的 validity/错误检测，使用 AUROC、AUPRC 和误报率；对连续合成质量 $Q_D$，优先使用回归误差和“预测均值—目标均值”分箱。
6. 只有输出明确代表事件概率时，才使用相应 Brier/NLL/ECE。不能把任意连续质量分数直接套概率校准指标，再宣称传感器概率已校准。
7. 分割输出 confidence 的校准和 Depth quality 的校准分别报告，RE049 的 detection calibration 不能直接充当 Depth reliability 证据。
8. 若相同预测 map 的全局均值与局部对应效果相当，先怀疑“局部诊断必要性”，而不是立刻增大头部容量。

**新增建议参数：** 质量图用 10 个预登记等频分箱；以 location group 计算不确定区间；每箱同时报告样本/有效像素数量。若没有相应概率输出，不为凑指标另行引入不必要的概率模块。

**参数来源声明：** 10 箱和分组方式为本版建议，非 SGMA、UMFNet 或 RE049 的统一官方设置。已有 A2 loss/读出保持原定义，缺失细节从代码核实。[RE049][AI013]

### S2.6 最小 B1-V 及可选 B1-A

**目标：** 将已经证明有用的动作变为稳定可训练模块，而不是增加一组未经检验的新机制。

#### B1-V：Depth 几何偏置的可靠性条件干预

保留既有 reliability estimator，以 $\rho_{ij}=r_ir_j$ 为起点。需要 baseline-safe 初始化时，可以采用：

$$
a_{ij}^{\ell}
=1-\lambda_\ell(1-\rho_{ij}^{\ell}),\qquad \lambda_\ell\in[0,1],
$$

$$
\widetilde B_{\mathrm{geo},ij}^{\ell,h}
=w_s^\ell B_{S,ij}^{\ell,h}
+w_d^\ell a_{ij}^{\ell}B_{D,ij}^{\ell,h}.
$$

$\lambda_\ell=0$ 时恢复原计算，$\lambda_\ell=1$ 时采用完整 pair rule。实现精确零初始值时要使用真正的 bypass 或相应有界参数方案，不能把“sigmoid 非常接近零”写成精确 identity。

第一轮只增加少量 stage-wise 强度参数，不并行添加 MoE、重建损失、语义原型或对齐网络。是否需要按 block 独立强度，必须作为后续单变量消融，而非默认扩张。

初始训练保持质量目标不变：

$$
\mathcal L
=\mathcal L_{\mathrm{seg}}
+\lambda_Q\mathcal L_Q^{\mathrm{existing}}.
$$

$\lambda_Q$、损失形式、输出读出来自已有配置，不在本蓝图中补造。第一版可继续隔离 segmentation loss 到 reliability head 的梯度，让 $\hat r_D$ 保持质量预测语义；“允许任务梯度共同优化质量头”仅作为单独消融，需重新检查其是否退化为普通 task gate。

#### B1-A：低自由度的几何坐标适应

只有 Oracle-B 支持时才开展。先预测全局或极粗网格 offset，将 offset 转换到实际几何尺度，保留显式支持域和 identity 选项。

固定三个对照：普通同自由度 warp；仅降低错位区域几何作用；坐标补偿加关系有效性处理。offset confidence 是否必要，应由前两者结果决定，不预先增加。

#### 训练预算与初始化公平性

若先从 epoch-420 继续训练做可行性研究，所有比较组必须从同一 checkpoint 出发并获得相同新增步数，A2 本身也要有继续训练对照。**不能拿“420 + 新训练”的 B1 对比未继续训练的 A2，然后把差异全算作模块效果。**

本版提供一个非强制的可行性起点：额外 50 epochs、主干学习率 $6\times10^{-6}$，其余设置保持一致；corruption curriculum 不能重新从早期轻度阶段开始。它只服务研发，不替代最终从相同预训练起点、相同主训练预算的配对实验。

更保守的选择是先做完全无训练的 predicted-map 接入；若已经失败，优先定位诊断/动作差距，而不是直接启动长训练。

**参数来源声明：** stage-wise $\lambda$、50 epochs、$6\times10^{-6}$ 和梯度隔离策略是本版候选方案，不是 PR070 或其他文献摘要参数。正式执行前应登记并核对原优化器状态、调度器和已有 loss 设置，不能混入已经冻结的 A2 结果。[PR070][AI024][PR029][MoSA]

### S2.7 最低充分对照矩阵

**目标：** 排除暴露、参数量、全局减弱几何和简单 invalid rule 四类替代解释。

| 组别 | 配置 | 主要问题 |
| --- | --- | --- |
| C-clean | v3 路径 clean-only control | 原始训练策略下的失效表现 |
| C-aug | 相同 corruption exposure，无 action | 训练暴露本身的收益 |
| C-A2 | C-aug + 既有并行质量估计，无 action | 诊断与分割行为分离；隔离成立时可证明与 C-aug 的分割等价 |
| C-valid | 可部署 observed-validity + 同一 pair action | 学习式连续诊断是否真有必要 |
| B1-V | predicted reliability + Depth-specific pair action | 核心增量 |
| C-mean | 同来源信号的全局/逐图平均或可学习常数几何强度 | 是否只是普遍降低 Depth 依赖 |
| C-generic | 同参数规模、同位置范围的通用空间 feature/adapter 调节 | 是否只是额外适应能力 |

C-generic 必须尊重 DFormerv2 实际结构，不能虚构一个不存在的独立 Depth encoder。可在对应 RGB 主特征上做小型空间残差适配，使用相同可靠性输入、相近参数规模，并单独报告 FLOPs、显存和延迟；参数匹配不等于计算完全匹配。[PR029][PR089][MoSA]

上述均为**机制类别对照**；未经忠实实现不能直接命名为“复现的 SGMA/UMFNet/MoSA”。

另外，Oracle、shuffle、inverse 属于固定模型上的机制诊断，可共享训练结果，不必每一张图都重新训练一个模型。预算有限时，先完成 C-clean、C-aug/C-A2、C-valid、B1-V、C-generic；C-mean 的固定和轻量学习版本应尽量保留，因为它对“动态信号有用”的结论很关键。

进入论文比较时，再选择 2–3 个代表性外部方法，而不是完整复现全部防撞论文：一个普通 RGB-D 强基线（如 CMX 或 GeminiFusion）、一个缺失模态适应方法、一个空间可靠性/适应机制。所有可报告数字的方法应在相同数据划分与失效 manifest 下公平评价。无法公平适配者保留文献对照，不能虚构复现结果。[AI003][AI005][AI019][AI024][PR090][MoSA]

**参数来源声明：** 对照组组合与匹配方式为本版实验设计；正式复现具体论文仍须查阅其全文和官方实现。不能将简化控制组的结果说成原论文方法的性能。

### S2.8 正式统计与独立确认

**目标：** 把开发阶段“看起来有效”升级为受控证据。

**操作：**

1. 开发阶段严格沿用 **196 location groups、10,000 次配对 percentile bootstrap**。每次抽组时，对同一组的所有方法、所有条件使用相同抽样权重。
2. 从抽到样本的混淆矩阵重新聚合并计算 dataset-level mIoU，再计算条件宏平均及方法差；不要以“逐图 mIoU 的平均”替代原指标。
3. 保存组映射、重采样 seed 和全部差值；多模型/多插入位置筛选属于开发，不将未经修正的最佳开发区间当最终显著性结论。
4. P-confirm 至少使用 3 个预登记训练 seed，报告每个 seed 及均值/标准差。测试样本 bootstrap 不等同于训练随机性估计，不能用大量像素或 10,000 次重采样冒充更多独立训练。
5. design freeze 后，按预登记计划在 official test 上统一评价所有锁定方法与 seed，不用 test 调阈值、选 epoch、挑 checkpoint 或重设 corruption。
6. test 的 location group 数量由对应元数据确定；196 是当前 val-dev 的组数，不直接搬到 test。

#### 全部 1,595 张训练数据的 checkpoint 规则

旧蓝图提出最终用 full train=1,595 重训。本版补上一个必须明确的条件：

> **当原 val-dev 318 张被并入训练后，它们不能再作为独立 checkpoint 验证集。**

推荐做法是：在 1,277/318 开发阶段完成结构和训练时长选择，随后为 full-train 阶段预先固定保存/报告规则，例如固定训练末尾 epoch=500 的 checkpoint。该规则必须在 full-train 开始前登记，对配对方法一致执行；不能因某个方法结果不好又改成其他 epoch。

若项目选择保留独立 318 张用于 clean selector，就继续用 1,277 训练，并如实报告，不能同时声称 full-train 1,595 和独立318验证。

#### 真正 held-out 的要求

A2 已经见过六类 corruption，因此：

- 只在测试时挑其中一类叫 held-out，不成立；
- 只让新 head 没见过、但 backbone 曾见过，也不能叫整个模型的未见故障；
- 连续随机训练范围内的某个没单独列过的 severity，不等于强度外推。

建议确认阶段预登记一类有现实意义且具有判别力的故障，完整地从该实验的 backbone/head/控制组训练暴露中排除，再测试。也可新增未训练的结构化故障，但不得先在 val 上反复调到本方法最有优势。

#### 外部验证

优先 NYUv2，采用该数据集自己的标签空间、划分与必要训练，保持算法结构和主要超参数选择原则不变；不要把跨数据集重新训练称为零样本跨域。SUN RGB-D 或 DeLiVER 仅作为有预算时的扩展。若要证明跨矿泛化，需要真实矿区/位置分组划分，不能从文件名猜一个矿区标签。[PR070][AI003][AI010][PR090][RE326]

**参数来源声明：** 196 组/10,000 次与三 seed 规划来自既有项目蓝图；full-train checkpoint 规则的补充为本版防泄漏建议；held-out 类型、额外 severity 和外部训练配置须在执行前登记，不能说是所引论文的统一推荐值。

---

## 步骤三：机理探索——只保留两个核心补充实验

### M1. 从输入失效追踪到几何关系失真

**目标：** 解释“损坏的 Depth 到底经过哪一步影响任务”，而不只展示好看的 reliability heatmap。

**方法依据：** 路径来自 PR070；质量不等于任务收益的警示来自 CLP、QMF 及任务驱动矿下研究；内部传播分析可参考 CoReFuse-Med。以下关系级测量是本版针对当前算子的设计。[PR070][PR040][AI021][AI025][RE266][RE418]

**操作：**

1. 对同一个样本保存参考 $D_0$、受损 $D_c$、最终 validity、各 stage 实际采样后的 depth 和关系偏置。$D_0$ 只是未注入本次合成扰动的参考，不宣称为绝对无误差深度。
2. 保持模型权重固定，分别构造参考/受损的 Depth bias：

$$
E_{ij}^{\ell,h}
=\left|
B_{D,ij}^{\ell,h}(D_c)
-B_{D,ij}^{\ell,h}(D_0)
\right|.
$$

3. 按 valid–valid、valid–invalid、invalid–invalid 分组，检查受损关系是否主要出现在理论预期的集合。misalignment 另按共同支持域及失去支持的边界分组。
4. 对比 no-action、observed-validity、predicted action：记录哪些 Depth bias 被删除/保留，哪些 RGB attention 分配随之改变，相关区域分割误差是否改善。
5. 同时保留“低 bias 误差但动作有害”“高 bias 误差但任务无变化”的样本，避免只挑支持假说的可视化。
6. 不只看数值较大区域。记录实际 checkpoint 的 $w_d$、attention 使用程度和最终任务变化，避免把一个几乎不被网络利用的几何误差误当核心原因。
7. 对 dropout 的模型内 resampling 单独开关；用“原采样+关系干预”“修正采样无关系干预”“二者组合”判定收益究竟来自哪一层。

**新增建议参数：** 每图每 stage 固定抽样最多 4,096 对实际存在的 token 关系，用于离线分析；不把前三层改成全 $N^2$ 矩阵。样本选择按固定 group/condition规则覆盖，不按收益挑图。可补充原图 3 像素宽的标签边界带，作为预登记区域定义。

**需提取结果：** 关系类型占比、bias 误差分布、Depth 项强度变化、attention 变化、区域分割 loss/IoU 和失败样例。相关性用于支持解释，不能单凭 heatmap 证明因果。

**预期可支持的结论：** “方法针对已定位的无效关系/错位关系起效”。若收益来自全图恒定减弱而非局部对应，应改写机制叙述。

**参数来源声明：** 4,096 对、3 像素边界和具体 bias 误差为本版测量方案，非 PR070 或 CoReFuse-Med 的原始参数；正式执行前需结合输出尺度核对。

### M2. 反事实几何效用与条件信号替换

**目标：** 区分质量、几何帮助和动作收益，并检验空间对应信息是否不可替代。

**方法依据：** QMF、CoReFuse-Med 已有权重/模态效用关联，PR042 支持专门检验 conditioning 信息；本项目进一步固定模型参数并只干预 Depth 几何项。[AI021][AI025][PR042][PR070]

#### A. 固定模型参数下定义 utility

对固定参数 $\theta$、同一输入和预登记区域 $\Omega$：

$$
U_D^{\mathrm{geo}}(\Omega)
=
\mathcal L_{\Omega}
\left(f_{\theta}^{\mathrm{DepthBiasOff}}(I,D_c),Y\right)
-
\mathcal L_{\Omega}
\left(f_{\theta}^{\mathrm{Base}}(I,D_c),Y\right).
$$

正值表示该定义下原 Depth 几何项有帮助；负值表示它使任务 loss 变差。位置项保持不变。

动作收益另外定义：

$$
G_A(\Omega)
=
\mathcal L_{\Omega}
\left(f_{\theta}^{\mathrm{Base}}(I,D_c),Y\right)
-
\mathcal L_{\Omega}
\left(f_{\theta}^{A}(I,D_c),Y\right).
$$

两个量均依赖模型、输入、区域和干预定义。它们是**模型内部干预效应**，不是与上下文无关的物理真值；也不能把局部 loss 收益直接相加称为全图 mIoU 收益。

#### B. 全局与局部两种干预

先做整图 Depth-bias off 以获得稳健参照；再在一个固定区域中，仅改变至少一个端点落在该区域的 Depth 关系，其他关系与 RGB token 均保留。

由于 attention 的行归一化和跨位置传播，局部动作可能影响区域外预测。报告区域内外效果，不能声称“改哪个 Depth 像素就只影响哪个语义像素”。

建议局部区域先采用固定 $4\times4$ 网格，而不是每次根据输出寻找最有利区域；有完整 group 标签时按 group 聚合。

#### C. 条件信号替换

对同一个训练后模型，至少执行：

| 信号 | 保持什么 | 检查什么 |
| --- | --- | --- |
| 原始预测图 | 原预测质量与空间结构 | 完整动作 |
| 合成 $Q_D$ | 理想合成状态信息 | 状态诊断差距；不要求一定优于 predicted |
| 图内均值图 | 每图/每尺度平均强度 | 动态局部性是否必要 |
| 图内空间置换/块平移 | 尽量保留数值分布，破坏位置对应 | 空间可靠性是否真正起作用 |
| 反向图 $1-\hat r_D$ | 相反排序 | 方向敏感性；须注明均值改变的混淆 |
| no-action / Depth-bias off | 两个边界 | 是否只是永久忽略几何 |

仅用随机图恶化不能证明“语义正确的可靠性”是唯一原因，因为替换图可能分布外。因此还要有**同均值、同直方图、仅改变位置**的控制，以及正常训练得到的 C-mean/C-generic。

反向图若均值差异很大，应补充保持原数值集合但反向排序的版本；不要用简单 clipping 后就声称完全均值匹配。shuffle 的固定种子建议为 5 个，每个方法共用；报告平均与范围，不选最差一次。

#### D. 分箱与解释

在同一种故障、同一强度内，按 $\hat r_D$ 或 $Q_D$ 分箱，报告 $E_{ij}$、$U_D^{\mathrm{geo}}$ 与 $G_A$。不能只利用“重度故障比轻度故障都更差”的总趋势来证明局部质量有效。

重点查看四种情形：

- 诊断准确，动作有益：支持完整链条；
- 诊断准确，动作有害：需要改 action，不能怪 estimator；
- 诊断不准，动作仍有益：可能是任务适应或固定降权效应；
- 诊断不准，动作无益：没有足够证据继续增加模块。

不机械要求 `Oracle > Predicted > Constant > Shuffle > Inverse` 在所有条件成立。$Q_D$ 只是合成状态目标，任务相关预测可能在某些条件超过它；关键是预登记对照能否排除合理替代解释。

**参数来源声明：** 固定 $4\times4$ 网格、10 分箱、5 个置换种子及以上 utility 定义是本版建议，不是 QMF/CoReFuse-Med 的原实验参数。指标使用范围和区域定义必须预先写入协议。

---

# 第三部分：成功门槛、失败解释与阶段出口

## 1. 保留旧 primary gate，不事后降低门槛

A2 相对正式 clean-v3 的既有预注册要求继续保留：

| 项目 | 冻结要求 |
| --- | --- |
| 六单失效宏平均 | 提升至少 +1.00 个百分点 |
| 配对统计 | 95% percentile bootstrap 区间下界 > 0 |
| clean | 降幅不超过 0.50 个百分点 |
| 条件覆盖 | 至少 5/6 单条件差值不为负 |
| 单条件退化 | 任一单条件差值不得低于 -1.00 个百分点 |
| 统计单元 | 当前 val-dev 的196个 location groups，10,000次重采样 |

以上要求来自旧蓝图。最新 A2 报告中“> -1.0”与旧冻结表的“不得低于 -1.0”在恰好等于边界时不同；执行时应以实际已签定 protocol 的比较符为准，并在结果前记录，不能事后挑选更宽松写法。[项目-v2][项目-A2]

**当前缺正式 clean-control，所以 A2 是否通过 primary gate 仍不能判定。**

## 2. B1 的增量门槛

B1 必须单独回答：相对相同 exposure 和预算的 A2/C-aug，是否有稳定正增量，clean 是否仍受控，C-valid/C-mean/C-generic 是否不能复制完整收益。

旧 +1.00 门槛针对 A2 与 clean-v3 的既定比较，不能不加说明地改成“每个新小模块也必须额外提升1.00”，也不能反过来用总体 +1.00 隐藏 B1 本身零增益。

B1 的正式增量评价以 $\Delta_{\mathrm{action}}>0$、配对区间下界大于0、clean约束、预登记对照和独立确认综合判断；本版建议在开发阶段就登记最小实际有意义增益，避免把非常小但样本统计显著的差异包装成强方法贡献。

## 3. 结果如何改变论文结论

| 观察 | 合理解释与下一步 |
| --- | --- |
| B0 压力表现差，A2 好 | 有研究导航价值；正式 exposure 归因仍需 clean-v3 |
| Oracle-V 好，predicted 不好 | action 有潜力，优先诊断目标/估计误差，不先扩大 action |
| observed-validity 与 learned 相当 | 贡献应收缩为确定性几何有效性处理，不能坚持 learned reliability 是核心 |
| predicted 与图内均值相当 | 缺乏局部诊断必要性证据 |
| generic adapter 与几何 action 相当 | 缺乏算子专用设计必要性证据；继续查是否仅额外训练/参数 |
| 只改善 entire missing | 重点可能是无深度表征学习，不应叙述为消除错误局部几何 |
| 只对合成 translation 有效 | 限制为该错位模型，不声称一般传感配准鲁棒性 |
| 全图平均提高但关键类明显下降 | 报告负面结果，不能只以平均值宣传安全性 |
| 开发有效、held-out/外部消失 | 泛化证据不足，收缩结论范围 |
| Oracle-A/B 都无明确空间 | 暂停当前动作方案；保留失效与负结果记录 |

负结果应进入研究账本；不能靠不断改名、换 checkpoint 或删条件维持既定故事。

## 4. 下一阶段实际执行顺序

| 阶段 | 执行任务 | 阶段出口 |
| --- | --- | --- |
| 已完成 | A2 v3 的 40/40 Main-Val units | 固定结果表与 epoch-420 开发起点 |
| P0：现状核验 | 本地几何入口、no-action identity、validity支持域、数据/协议hash；补B0历史stress | 确认干预对象与数值边界 |
| P1：可行性 | Oracle-A、Oracle-B；同时安排clean-control | 选择B1-V、B1-A或暂停当前方案 |
| P2：最小模块 | 先接现有预测图，再做必要的匹配训练；C-valid/C-mean/C-generic | 证明不是简单规则或额外适应收益 |
| P3：机制与设计冻结 | M1/M2；冻结loss、输入契约、对照、checkpoint和统计规则 | Design Freeze |
| P-confirm | 配对seed、锁定test、真正held-out或外部验证 | 论文主证据，而非开发展示 |

这一顺序是一份研究计划，不表示本次已经执行了这些实验。当前不应越过 Oracle 与 paired-control，直接进入大规模完整模型训练。

## 5. 论文叙事及主表安排

论文叙事建议为：

> 既有多模态鲁棒方法已处理模态缺失、质量估计和动态融合。本研究聚焦另一层问题：当 Depth 作为 RGB 表征的成对几何先验时，观测缺失和空间错位会以不同方式破坏几何证据。我们先定位这些影响，再设计最小关系级干预，并通过理想状态、可观察有效性、预测质量和错误条件信号的对照验证其作用。

这是**待实验支持的研究叙事**，不能现在改写成“我们已经证明”。

建议形成五组最终结果：

1. 原始/augmentation/A2/B1 的10条件主表，主分数与辅助均值清晰分开。
2. Oracle-A/B、observed-validity、constant、generic的机制对照。
3. 几何关系失真与局部utility分析，包括失败样例。
4. 三seed、clean保持、held-out与外部验证。
5. 新增参数、总计算量、峰值显存和相同设备/分辨率/精度下的延迟。

方法尚未完成前，不预写 SOTA、真实安全保障、任意模态鲁棒或普遍适用结论。如果最终仅在 MUSeg 上优于若干公平复现模型，应准确写成“在该划分与协议下优于所比较方法”，不能扩张为整个RGB-D领域SOTA。

## 6. 交给实现者的最低产物

| 产物 | 必须包含 |
| --- | --- |
| 实验身份文件 | 代码/配置/数据/权重hash，选择规则，随机种子，硬件与精度 |
| 输入与目标契约 | 14维输入、2维读出、最终validity、$R_D^{syn}$、禁止推理使用的信息 |
| 几何审计 | 实际bias位置、Depth resizer、位置项/深度项分离、stage/block支持域、no-action测试 |
| Oracle报告 | A/B原始数值、CI、支持域、失败条件；不是只报最好的一个 |
| 配对对照表 | exposure、训练步数、初始化、参数和checkpoint规则是否匹配 |
| 主指标文件 | 每样本/每组混淆矩阵，六单条件宏平均，三复合条件，clean与每类结果 |
| 决策记录 | 为什么继续/停止某条路线，是否修改协议，哪些结果仅属开发 |

本版的直接行动结论是：

> **先验证“无效或错位的 Depth 关系是否有可回收的负效用”，再训练模型判断这些关系；不要把“已有一个可靠性头”当成它必须进入最终方法的理由。**

---

# 附录 A：资料优先级、修订说明与未决事项

## A1. 本版对旧蓝图的关键修订

| 旧表述或隐含问题 | v3.0处理 |
| --- | --- |
| A2还在训练/等待主验证 | 更新为已完成40/40开发评价 |
| B0可能被当clean-control或RGB-only | 明确为历史RGB-D参照 |
| generic reliability residual可作为主线 | 收窄到Depth-derived关系；普通spatial adapter已有MoSA等先例 |
| 后softmax乘mask直接作为实现公式 | 公开代码核验为pre-softmax加性bias；本地fork仍待核 |
| 默认Depth平均池化到stage | 公开代码当前为双线性插值；先核真实支持域 |
| $Q_D$直接称“真实物理可靠性” | 改为合成协议质量监督目标，物理效度另证 |
| confidence与utility混用 | 分离state、prediction、relation coefficient、utility、action gain |
| entire missing52.55是上限 | 删除；只代表当前模型当前条件 |
| Oracle负结果否定整个研究方向 | 限定否定当前实现/动作/起点，保留有依据的后续假说 |
| reliability辅助任务必然影响表征 | 先检查梯度与优化隔离；隔离成立则不得这样归因 |
| full train后仍用原val选epoch | 明确禁止；改为事前固定checkpoint规则或继续保留holdout |
| A2见过的故障也叫held-out | 要求整条训练链排除或明确仅组件未见 |
| raw-depth warp移进geometry函数就算新 | 指出可能数学等价；要求同自由度warp对照 |
| 将全部腐蚀平均替代primary | 保留六单失效宏平均及旧统计门槛 |

## A2. 证据优先级

项目当前状态以最新 A2 报告为主；冻结阈值与协议以既有正式 ledger/配置为准，旧蓝图用于核对。论文机制以原文为准，本次未重新取得全部原文的条目沿用所提供的全文审计，并标注其来源。官方代码只能证明被读取的公开版本；本地实现必须另核。

不能因为“新文件”日期更晚，就自动认定其中所有推断比老协议更正确。例如“任何Oracle负结果都应停止整个家族”以及“Q是真实物理质量”仍需要本版所作的限定。

## A3. 尚未由所提供资料解决的事项

- 本地DFormerv2代码、checkpoint中的几何混合权重与实际输入归一化。
- A2可靠性头的准确14维输入、2维输出定义、loss公式/权重、各类型$R_D^{syn}$完整映射。
- 正式paired clean-control、B0 frozen stress profile、Oracle-A/B结果。
- A2预测质量的完整独立指标。
- full-train确认阶段最终登记的checkpoint规则及测试group映射。
- 部分文献元数据的一致性。

这些缺口不会阻止制定实验，但不能在方法或结论中当成已经确认的事实。

## A4. 文献元数据处理

PR070沿用CVPR2025。PR089沿用已整理的TGRS正式记录。PR090、PR029本次可查到CVPR2026官方条目，但不能因此宣称其具体实验设置已在本次重新逐项复核。

PR059的出版信息在现有资料中冲突，继续列待核，不用来支撑关键方法结论。

MoSA的用户整理题名为“...RGB-X Semantic Segmentation”，本次官方IEEE检索显示题名尾部为“...Multimodal Semantic Segmentation”。全文机制仍以提供的P2审计为依据；正式引文题名、DOI与版本对应关系须核对。这个元数据问题不构成忽略其空间可靠性/adapter先例的理由。

AI023、AI024、AI027、AI028使用已给出的arXiv标识，未确认正式发表时保留预印本身份。AI029/AI030是本次外部补充的机制近邻，不冒称已完成与P0相同深度的全文六维审计。

---

# 附录 B：项目与文献来源索引

以下索引用于离线阅读。文献中的参数不自动转化为MMFR配置；正文标为“本版建议”的参数均属于拟议设计。

## B1. 项目依据与附件身份

| 引用简称 | 对应附件 | SHA-256前12位 |
| --- | --- | --- |
| 项目-A2 | `A2实验结果报告.md` | `2f9d82de2be2` |
| MMFR-PR文献整理清单(3) | `MMFR-PR文献整理清单(3).md` | `84b8cc56c05a` |
| MMFR-RE文献整理清单(2) | `MMFR-RE文献整理清单(2).md` | `752acf9c997d` |
| MMFR-新增文献与Idea撞车审计(2) | `MMFR-新增文献与Idea撞车审计(2).md` | `6ef3ea7289b7` |
| PR其他(1) | `PR其他(1).txt` | `c728325bc524` |
| PR核心(1) | `PR核心(1).md` | `5de41a4549b6` |
| RE全(1) | `RE全(1).txt` | `4bc45d751001` |
| RE核心(1) | `RE核心(1).md` | `4d603dc6407a` |
| 生成提示词 | `生成研究蓝图使用的提示词(1).txt` | `5f5ae01f538d` |
| 项目-v2 | `研究蓝图-MMFR模态失效鲁棒性-v2(1).md` | `9180392ccd20` |
| 防撞-P0 | `防撞车P0(1).md` | `0be5e394d4c4` |
| 防撞-P1 | `防撞车P1(1).md` | `3858ea3525d7` |
| 防撞-P2 | `防撞车P2(1).md` | `93bb36120d0f` |


这里的哈希只标识本次读取的附件，不是训练代码、数据集或checkpoint的哈希。

## B2. 核心PR/RE文献

以下题名和已记录DOI沿用提供的整理资料；“待核”不影响正文对方法角色的保守使用。

| 编号 | 文献 | 标识/备注 |
| --- | --- | --- |
| PR070 | DFormerv2: Geometry Self-Attention for RGBD Semantic Segmentation | CVPR2025；DOI `10.1109/CVPR52734.2025.01802` |
| PR029 | Uncertainty-Aware Modality Fusion for Unaligned RGB-T Salient Object Detection | CVPR2026官方条目已核；本文使用UMFNet简称 |
| PR089 | SGMA: Semantic-Guided Modality-Aware Segmentation for Remote Sensing with Incomplete Multimodal Data | DOI `10.1109/TGRS.2026.3692798`，沿用整理 |
| PR090 | Towards Robust Multi-Modal Semantic Segmentation with Teacher-Student Framework and Hybrid Prototype Distillation | CVPR2026官方条目已核；RobustSeg |
| PR040 | CLP: A Real-World Dataset of Contaminated Lens Protectors for Robust Semantic Segmentation | 方法依据来自提供的全文补充 |
| PR042 | Robust Promptable Video Object Segmentation | MoGA；条件化适应的实验设计参考 |
| RE326 | MUSeg: A multimodal semantic segmentation dataset for complex underground mine scenes | DOI `10.1038/s41597-025-05493-9` |
| RE240 | Mine-DW-Fusion: BEV Multiscale-Enhanced Fusion Object-Detection Model for Underground Coal Mine Based on Dynamic Weight Adjustment | DOI `10.3390/s25165185` |
| RE026 | LFR-CMT-3D: Lipschitz-regularized multi-modal fusion for robust object detection in open-pit mines | DOI `10.1088/1361-6501/ae58c7` |
| RE094 | Zero-Shot Polarization-Intensity Physical Fusion Monocular Depth Estimation for High Dynamic Range Scenes | DOI `10.3390/photonics13030268` |
| RE049 | M-SURE: Enhanced and reliable safety monitoring in low-light mines | DOI `10.1016/j.aei.2026.104702` |
| RE266 | Scene Understanding System of Underground Pipeline Corridors Under Characteristic Degradation Conditions | DOI `10.3390/s26010141`；使用其任务驱动评价结论 |
| RE418 | Adaptive Image Enhancement Method for Coal-Mine Underground Image Based on No-Reference Quality Evaluation | DOI `10.1109/TIM.2024.3470234` |
| RE452 | Graph-based adaptive weighted fusion SLAM using multimodal data in complex underground spaces | DOI `10.1016/j.isprsjprs.2024.08.007`；题名依据所提供RE目录 |

## B3. AI编号题名与DOI

编号沿用旧蓝图及修订后的防撞清单；列出全部既有AI编号便于连续使用，不代表本版逐篇重新核验了全部书目信息。arXiv DOI标识所引预印本版本，不代表正式会议DOI。

- **[AI001]** *Are Multimodal Transformers Robust to Missing Modality?* — `10.1109/CVPR52688.2022.01764`

- **[AI002]** *Multimodal Token Fusion for Vision Transformers* — `10.1109/CVPR52688.2022.01187`

- **[AI003]** *CMX: Cross-Modal Fusion for RGB-X Semantic Segmentation With Transformers* — `10.1109/TITS.2023.3300537`

- **[AI004]** *DFormer: Rethinking RGBD Representation Learning for Semantic Segmentation* — `10.48550/arXiv.2309.09668`

- **[AI005]** *GeminiFusion: Efficient Pixel-wise Multimodal Fusion for Vision Transformer* — `10.48550/arXiv.2406.01210`

- **[AI006]** *Missing Modality Robustness in Semi-Supervised Multi-Modal Semantic Segmentation* — `10.1109/WACV57701.2024.00106`

- **[AI007]** *Multi-Modal Learning with Missing Modality via Shared-Specific Feature Modelling* — `10.1109/CVPR52729.2023.01524`

- **[AI008]** *Learning Modality-Agnostic Representation for Semantic Segmentation from Any Modalities* — `10.1007/978-3-031-72754-2_9`

- **[AI009]** *Robust Multimodal Learning With Missing Modalities via Parameter-Efficient Adaptation* — `10.1109/TPAMI.2024.3476487`

- **[AI010]** *Benchmarking Multi-Modal Semantic Segmentation Under Sensor Failures: Missing and Noisy Modality Robustness* — `10.1109/CVPRW67362.2025.00146`

- **[AI011]** *Benchmarking the Robustness of Semantic Segmentation Models with Respect to Common Corruptions* — `10.1007/s11263-020-01383-2`

- **[AI012]** *RoboDepth: Robust Out-of-Distribution Depth Estimation under Corruptions* — `10.52202/075280-0932`

- **[AI013]** *On Calibration of Modern Neural Networks* — `10.48550/arXiv.1706.04599`

- **[AI014]** *SelectiveNet: A Deep Neural Network with an Integrated Reject Option* — `10.48550/arXiv.1901.09192`

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

- **[AI029，本次补充]** *Confidence Propagation through CNNs for Guided Sparse Depth Regression* — DOI: `10.48550/arXiv.1811.01791`。使用其对稀疏深度有效性、置信传播与归一化聚合的机制说明；不将其完整模型作为已复现baseline。
- **[AI030，本次补充]** *Uncertainty-Aware CNNs for Depth Completion: Uncertainty from Beginning to End* — CVPR2020；所引预印本 DOI: `10.48550/arXiv.2006.03349`。作为输入置信估计与深度补全的历史机制近邻。
- **[MoSA]** 用户整理题名为 *Modality-Aware Spatially-Adaptive Adaptation for RGB-X Semantic Segmentation*；本次IEEE检索题名显示为 *MoSA: Modality-Aware Spatially-Adaptive Adaptation for Multimodal Semantic Segmentation*。提供资料记录 DOI: `10.1109/ACCESS.2026.3694496`；正式题名与该DOI的版本对应仍待核。
- **[ANGA]** *Anchor-Guided Gradient Alignment for Incomplete Multimodal Learning*。沿用未单独分配AI编号的状态；本版不将其作为关键结构依据。

## B4. 本次外部核验的原始来源

访问日期均为2026-09-18。URL保留为代码文本用于溯源；公开分支会变化，本地复现应固定commit。

| 核验对象 | 原始来源 |
| --- | --- |
| PR070-代码：GeoPriorGen、Decomposed_GSA、Full_GSA | `https://raw.githubusercontent.com/VCIP-RGBD/DFormer/main/models/encoders/DFormerv2.py` |
| PR070官方论文条目 | `https://openaccess.thecvf.com/content/CVPR2025/html/Yin_DFormerv2_Geometry_Self-Attention_for_RGBD_Semantic_Segmentation_CVPR_2025_paper.html` |
| RE326数据与分组说明 | `https://www.nature.com/articles/s41597-025-05493-9` |
| AI023预印本 | `https://arxiv.org/abs/2604.11585` |
| AI024预印本 | `https://arxiv.org/abs/2607.20326` |
| AI027预印本 | `https://arxiv.org/abs/2603.02695` |
| AI028预印本 | `https://arxiv.org/abs/2608.03475` |
| AI029预印本 | `https://arxiv.org/abs/1811.01791` |
| AI030预印本 | `https://arxiv.org/abs/2006.03349` |

最重要的外部核验结果不是增添更多论文，而是修正了干预位置的实现假设。下一次执行必须用本地代码和权重复核这一点；本蓝图不以公开main分支替代你的实际实验身份。
