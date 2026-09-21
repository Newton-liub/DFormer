# MMFR：B0 → A2 → 下一模块设计最终审计报告

## 0. 本报告的定位

本报告用于回答：

1. 当前 A2 到底暴露了什么问题；
2. B0 与 A2 现在能比较到什么程度；
3. 哪些现象值得进入下一模块；
4. 哪些推断目前证据不足；
5. 下一模块应该优先设计什么，而不应该设计什么；
6. 后续最低成本实验应该按什么顺序执行。

当前仍然没有正式配对 clean-control，因此本报告属于：

**研究方向审计 + 模块设计依据**

而不是：

**A2 有效性的最终实验结论。**

------

# 1. 当前实验事实

A2 v3 已经完成：

- 4 个候选 checkpoint；
- 每个 checkpoint 10 个 frozen conditions；
- 每个 condition 318 个 `val-dev` 样本；
- 总计 40/40 formal evaluation units；
- official test 未使用。

正式结果显示 `selector-epoch-420` 是当前最稳定的主 checkpoint：

- clean mIoU：57.06；
- protocol main score：55.55；
- 9 corrupted-condition mean：55.7078；
- 10-condition mean：55.8430；
- 在四个候选 checkpoint 的 10/10 conditions 中均排名第一。

因此后续设计阶段可以把：

**epoch-420 作为 A2 的主开发 checkpoint。**

390 / 370 / latest 保留为 checkpoint sensitivity evidence，不需要继续围绕它们开发。

------

# 2. B0 与 A2 的正确关系

B0 是 RGB-D DFormerv2-S baseline，不是 RGB-only baseline。

现有资料确认 B0 与 A2 在以下方面高度接近：

- 同 DFormerv2-S + HAM decoder；
- 同 pretrained；
- 同 `train-dev / val-dev`；
- 同主要 optimizer / scheduler / epoch budget；
- 同主评价几何；
- 都属于 RGB-D segmentation。

但两者并不是正式配对实验：

- B0 使用旧训练代码路径；
- A2 使用 v3 corruption training 路径；
- A2 增加训练期 corruption 与 reliability auxiliary objective；
- B0 当前没有 frozen 10-condition failure profile；
- A2 当前没有正式 clean-control。

因此：

**B0 可以作为 historical development reference。**

不能称为：

**A2 的 clean-control。**

------

# 3. B0 → A2 当前只能得到什么方向信息

现有 clean 数值：

- B0 historical clean mIoU：58.79；
- A2 epoch-420 clean mIoU：57.06。

数值差：

$58.79-57.06=1.73$

训练 selector 也表现出近似量级：

- B0：56.39；
- A2：54.63；
- 差约 1.76。

这说明有一个值得认真关注的现象：

> 当前 A2 identity 的 clean performance 大约低于历史 B0 reference 1.7–1.8 mIoU。

但现在不能解释这个差值的来源。

不能写：

- A2 比 B0 差 1.73；
- corruption training 损失了 1.73；
- A2 用 clean performance 换取了 robustness。

因为缺少严格配对 control。

正确说法是：

> A2 与历史 B0 clean reference 存在约 1.7 mIoU 的描述性差距，该差距需要正式 clean-control 才能归因。

------

# 4. 建议额外补一个低成本 B0 stress profile

下级报告建议“不重跑 B0”。

从正式因果评价角度，这个判断没有问题。

但从**下一模块设计**角度，我建议改变：

> 建议把 B0 checkpoint 放进当前 frozen 10-condition evaluator，得到一个 historical stress reference。

目的不是证明 A2 比 B0 好，而是回答：

> 原始 DFormerv2 在这些 failure 下到底是什么形状？

这样我们才能区分两种完全不同的情况。

### 情况 A

例如 B0：

```text
clean       58.79
missing     46
dropout     48
misalign    51
```

而 A2：

```text
clean       57.06
missing     52.55
dropout     54.42
misalign    55.80
```

这种结果虽然仍然不能形成正式因果 claim，但会强烈提示：

> corruption-trained A2 很可能已经改变 robustness profile。

那么下一模块就应该以：

**在保持 A2 robustness 的同时追回 clean performance**

为重要目标。

### 情况 B

如果 B0 failure profile 本来就与 A2 接近，那么说明：

> A2 当前 robustness 很可能主要来自 DFormerv2 本身，而不是 A2 机制。

那么下一模块的研究重点就应该更彻底地重新设计。

因此：

**B0 × 10-condition 的价值主要是科研导航，而不是论文证明。**

如果单 checkpoint 本地只需约 4 小时，这个实验值得做。

------

# 5. A2 当前 failure profile

定义 condition 相对 clean 的下降：

$\Delta_c=mIoU_c-mIoU_{clean}$

epoch-420：

| condition            | mIoU  | $\Delta_c$ |
| -------------------- | ----- | ---------- |
| clean                | 57.06 | —          |
| entire_missing@1.0   | 52.55 | -4.51      |
| spatial_dropout@0.75 | 54.42 | -2.64      |
| misalignment@0.75    | 55.80 | -1.26      |
| gaussian_noise@0.75  | 56.48 | -0.58      |
| quantization@0.75    | 56.99 | -0.07      |
| blur@0.75            | 57.06 | 0.00       |
| sd@0.5 + gn@0.5      | 55.15 | -1.91      |
| blur@0.5 + mis@0.5   | 56.45 | -0.61      |
| quant@0.5 + mis@0.5  | 56.47 | -0.59      |

这个 profile 很清楚：

### 第一层：明显问题

1. entire missing；
2. spatial dropout；
3. misalignment。

### 第二层：次要问题

1. gaussian noise；
2. mixed corruption。

### 第三层：当前协议判别力很低

1. blur；
2. quantization。

因此下一模块不应该平均对待所有 corruption。

------

# 6. 关于 blur / quantization 的正确解释

目前：

- blur@0.75 几乎不影响 mIoU；
- quantization@0.75 几乎不影响 mIoU；
- 四个 checkpoint 都表现出相同趋势；
- corrupted Depth 的 hash 与 clean 不同，因此 corruption 确实执行了。

所以比较合理的解释是：

> 当前 frozen severity 下，这两类 corruption 对 DFormerv2 的有效几何证据扰动不足。

这首先是：

**measurement sensitivity 问题**

而不是：

**architecture failure 问题。**

因此下一模块不应该为了 blur / quantization 专门加组件。

它们可以继续保留在 frozen protocol 中保证可比性。

如果未来重新设计 robustness benchmark，可以另立 protocol v2 调整 severity，但不要修改当前已经冻结的实验。

------

# 7. 对 entire_missing 的关键纠偏

下级模型有一个非常重要但推断过强的结论：

> 52.55 是关闭 Depth 后的“上限”，门控/fallback 没有空间。

这个结论不能成立。

更准确的是：

> 52.55 是当前 epoch-420 参数、当前零 Depth 输入语义下，Depth pairwise geometry term 被中和后的实测性能。

它不是模型理论上或结构上的 upper bound。

例如下面这些方案仍然可能超过 52.55：

- 为 missing-depth 专门学习一套 adapter；
- mixture-of-experts 中切换到 missing-modality expert；
- conditional normalization；
- RGB-derived substitute geometry；
- missing-depth-specific positional prior；
- teacher-student distillation；
- modality-specific token；
- learned RGB fallback；
- feature hallucination。

它们都不要求恢复原始 Depth。

因此：

**entire_missing 不能被排除。**

但它的机制与 spatial_dropout 不一样。

------

# 8. entire_missing 真实说明了什么

Depth 全零后，DFormer 的 Depth difference geometry 项可能退化或被中和。

这个代码事实非常有价值。

它说明：

> 对 entire_missing 来说，问题不一定是“错误 Depth 仍然污染模型”。

更可能是：

> 模型失去了原来由 Depth 提供的有效几何信息。

因此下一模块针对 entire_missing 时，重点不应该只是：

**抑制 Depth**

因为它已经没有什么可以继续抑制。

应该考虑：

### 路线 A

学习更好的：

**no-depth operating mode**

### 路线 B

利用 RGB 产生：

**substitute geometry**

### 路线 C

使用教师或多模态训练：

**让 RGB feature 学到更强的几何表征**

因此下级模型“entire_missing 不是简单门控问题”这一点正确。

但“已经没有提升空间”是不正确的。

------

# 9. 是否能说模型“不依赖 Depth”

不能。

A2 clean：

57.06

missing：

52.55

下降：

4.51。

这是当前最大的 failure degradation。

92.1% retention 可以说明：

> 模型具有相当强的缺失模态容忍能力。

不能说明：

> 模型不依赖 Depth。

更不能直接说明：

> 过度依赖 Depth 已被否定。

“过度”本身需要一个参照，例如：

- RGB-only 模型；
- clean-control；
- 其他 RGB-D backbone；
- theoretical oracle。

当前没有这样的参照。

因此最终采用：

> DFormerv2/A2 在 Depth 完全缺失时仍表现出较强的 graceful degradation，但 Depth 缺失仍造成当前最大的 mIoU 降幅。

------

# 10. 当前最值得研究的问题：spatial dropout

这一点我基本同意下级模型。

spatial dropout 是当前**最干净的结构问题**。

原因：

1. 降幅足够明显：-2.64；
2. 181/318 样本退化；
3. 已经有 validity ground truth；
4. failure location 明确；
5. 很容易构造 Oracle；
6. 与 DFormer 的 pairwise geometry 机制直接相关。

如果 invalid Depth 被写为 0，而几何项使用类似：

$|d_i-d_j|$

那么 valid-invalid pair 可能产生没有物理意义的深度差。

因此下一模块一个非常合理的目标是：

> **不要让不存在的 Depth 产生错误几何证据。**

这是比“预测一个 generic reliability score”更具体的科学问题。

------

# 11. 对 C1 的重新定义

原 C1：

> reliability-conditioned geometry prior adaptation

方向合理，但建议进一步收窄。

我建议把核心问题定义成：

## Validity-aware Pairwise Geometry

不是：

> “给 Depth 加一个 confidence gate”。

而是：

> “只有当一个 pair 的几何关系本身有足够证据时，Depth-derived geometry bias 才应该存在。”

例如定义 pair reliability：

$r_{ij}=f(r_i,r_j)$

最简单的 Oracle 情况甚至可以先用：

$r_{ij}=r_i r_j$

然后几何项：

$G_{ij}=P_{ij}+r_{ij}D_{ij}$

其中：

- $P_{ij}$ 为 position prior；
- $D_{ij}$ 为 Depth-derived geometry；
- $r_{ij}$ 为 pair reliability。

当：

$r_i=r_j=1$

完全恢复原始 DFormer。

当任一 endpoint 无效：

$r_{ij}=0$

则：

$G_{ij}=P_{ij}$

不会把 invalid Depth=0 当成真实几何信息。

这一设计的价值在于：

> 它处理的是“pairwise evidence validity”，而不仅仅是普通 feature gating。

这比简单 reliability fusion 更有清晰的问题定义。

------

# 12. 为什么必须先做 C1 Oracle

历史 DVG-B1 Oracle 为负，这个证据不能忽略。

但它也不能直接否定新的 C1。

因为旧实验与当前候选存在：

- checkpoint 不同；
- corruption family / setting 不完全相同；
- gate 定义不同；
- query/key 作用位置不同；
- 当前问题已经明确聚焦到 pairwise invalid-depth evidence。

所以最划算的一步不是直接训练 C1。

而是：

> 在 A2 epoch-420 上构造一个无需学习的理想 validity Oracle。

只测：

- clean；
- spatial_dropout；
- entire_missing；
- optionally misalignment。

若 Oracle 仍然不能改善 spatial_dropout：

> 不应该继续做 reliability gating 家族。

若 Oracle 能显著改善：

> 才证明这个几何入口存在真实可利用空间。

这是非常高价值的 go/no-go experiment。

------

# 13. 当前第二优先问题：misalignment

misalignment 的性质与 missing 完全不同。

missing 是：

**evidence unavailable**

misalignment 是：

**evidence available but spatially wrong**

因此简单 validity mask 无法解决。

这也是为什么 C1 不应该试图一次解决所有 failure。

针对 misalignment 更合理的方向是：

## Geometry Consistency / Alignment

核心思想：

> 在计算 Depth geometry prior 前，先判断 RGB 与 Depth 的结构是否在同一坐标位置。

潜在线索：

- edge correspondence；
- gradient orientation；
- local correlation；
- local RGB-depth structure consistency；
- coarse displacement estimation。

如果存在偏移，再：

- 校正 geometry coordinate；
- 搬运 depth evidence；
- 或降低对应区域 pair evidence。

这里我更推荐：

**低自由度 correction**

而不是：

**pixel-wise optical-flow-style correction**

因为后者：

- 参数多；
- 容易过拟合 synthetic translation；
- clean performance 风险大；
- 与已有 alignment 文献撞车更严重。

下一阶段最好先研究：

- global translation；
- block-wise translation；
- coarse offset bins。

------

# 14. C2 的定位

因此 C2 可以定义为：

## Consistency-Guided Geometry Realignment

而不是普通：

> RGB-D registration network。

输入 RGB / Depth structure evidence。

输出一个非常受限的：

- offset；
- offset confidence；
- geometry correction。

仅修正：

**Depth → GeoPriorGen**

而不是重新对齐完整 Depth feature stream。

这样可以保持：

- 模块小；
- identity-safe；
- 容易解释；
- 容易做消融；
- 更接近 DFormer 特有的 geometry path。

------

# 15. C3 暂不建议进入正式开发

受限 prior-level Depth evidence completion 有一定研究空间。

但当前不推荐作为第一轮新模块。

原因：

1. 深度补全领域非常拥挤；
2. 容易逐步变成完整 reconstruction；
3. 训练目标复杂；
4. clean preservation 更困难；
5. entire_missing 下没有邻域有效 Depth；
6. 需要回答“为什么不直接用已有 depth completion network”。

因此：

**C3 保留为 fallback route。**

只有 C1 Oracle 和 C2 feasibility 都不理想时，再考虑。

------

# 16. 新文献对下一模块真正有价值的启发

当前新增论文大致分成三组。

## 第一组：模态可靠性

例如 SGMA、UMFNet 等。

启发：

> reliability 不应该只是一个 auxiliary prediction；它应该真正改变模型行为。

这正是 A2 当前最大结构缺口之一：

> A2 已经学习 reliability，但 reliability 不进入 backbone / decoder / geometry action。

这是下一阶段最值得利用的已有资产。

但需要注意：

> 直接做 reliability-weighted fusion 撞车风险很高。

因此最好进入：

**DFormer-specific geometry mechanism**

而不是普通 feature fusion。

------

## 第二组：missing modality robustness

例如 RobustSeg 一类方法。

核心启发不是简单 fallback，而是：

> full-modality 模型可以通过训练机制显式学习 partial-modality operating mode。

这对 entire_missing 很有价值。

因此如果 C1 后续扩展，可以考虑：

**geometry reliability + missing-mode training**

而不是直接维护一个独立 RGB-only model。

------

## 第三组：uncertainty / consistency

例如 uncertainty-aware fusion 类工作。

它们最值得借鉴的是：

> 不确定性首先用于判断“某条信息是否应当成为证据”。

这一思想与 pairwise geometry validity 很契合。

但下一模块不要一开始同时加入：

- uncertainty distribution；
- calibration；
- fusion；
- reconstruction；
- alignment。

否则模块会迅速失去可解释性。

------

# 17. 我推荐的研究主线

基于当前实验，我建议把下一阶段主线收敛为：

## Reliable Geometry under Invalid or Misaligned Depth

即：

> **不是泛化地研究 multimodal reliability，而是研究“不可靠 Depth 如何安全进入 DFormer geometry prior”。**

这比：

> uncertainty-aware multimodal fusion

更具体，也更贴合当前 backbone。

核心问题拆为两个正交 failure family：

### Family A：Evidence Missing / Invalid

代表：

- spatial dropout；
- entire missing。

核心目标：

> invalid evidence 不应产生 false geometry。

### Family B：Evidence Misregistered

代表：

- misalignment。

核心目标：

> valid-but-wrong-location evidence 不应产生 wrong geometry。

这样一个整体研究问题是统一的，但模块仍可以保持两个清晰组件。

------

# 18. 推荐下一模块总体架构思路

不是最终实现，只作为设计约束：

```text
Depth + RGB
   │
   ├── reliability / validity estimation
   │
   ├── consistency / alignment estimation
   │
   ▼
Geometry Evidence Controller
   │
   ├── invalid → suppress / replace pairwise depth evidence
   ├── misaligned → correct / relocate geometry evidence
   └── reliable → exact original DFormer geometry
   │
   ▼
GeoPriorGen
   │
   ▼
DFormer GSA
```

设计原则是：

### clean identity

当 Depth 正常时：

$r_{ij}=1$

且：

$\Delta_{ij}=0$

模型应尽量退化回原 DFormer geometry。

这是非常重要的 clean-preservation 机制。

------

# 19. 不建议下一模块立即做的事情

目前不建议：

### 1. 独立 RGB-only fallback network

理由：

- 增加完整第二模型；
- 计算成本大；
- 与 missing-modality 文献重合高；
- 当前尚未证明需要独立 RGB branch。

### 2. 完整 Depth reconstruction

理由：

- 问题太大；
- 研究重心容易从 segmentation robustness 变成 depth completion。

### 3. 大型 MoE

理由：

- 当前数据量只有 1277 train-dev；
- 很容易把简单 reliability 问题复杂化。

### 4. 同时解决全部 6 corruption

没有必要。

目前实验已经告诉我们：

> failure importance 并不均匀。

------

# 20. 下一模块的开发优先级

建议：

## P0：Oracle validity-aware pairwise geometry

只改 geometry evidence。

不训练 reliability predictor。

直接用 synthetic corruption 已知的 ground-truth validity。

目的：

> 检查“正确知道哪里无效”是否真的能改善 spatial dropout。

这是最关键的一步。

------

## P1：如果 P0 为正

设计：

**learned reliability → pairwise geometry adapter**

即 C1。

------

## P2：并行/随后测试 alignment oracle

对于 synthetic misalignment，真实 shift 本来就是已知的。

因此同样可以做：

> 使用 ground-truth inverse shift 校正 Depth，再评价。

这会直接回答：

> 如果 misalignment 被完美纠正，mIoU 最多能回到哪里？

甚至这个 Oracle 比学习式 alignment 更重要。

如果完美 correction 都只能提升很少：

> 就没有必要训练 C2。

如果能恢复接近 clean：

> C2 具有明确上限空间。

------

# 21. 这一点比下级报告更重要：建议做两个 Oracle

下一阶段不要只做 reliability Oracle。

应该做：

### Oracle-A：Validity Oracle

针对：

```
spatial_dropout
```

问题：

> 完美知道哪些 Depth 无效，并让这些 pair 不使用 Depth evidence，能否恢复性能？

### Oracle-B：Alignment Oracle

针对：

```
misalignment
```

使用 corruption generator 已知的真实 shift 做 perfect inverse correction。

问题：

> 完全纠正错位后，损失能恢复多少？

这两个实验：

- 不需要训练；
- 不需要新 checkpoint；
- 成本低；
- 能直接决定 C1 / C2 是否有研究空间。

它们应该是下一模块开发前最优先的实验。

------

# 22. 关于 entire_missing 的第三个 Oracle

还可以考虑一个可选诊断：

### Oracle-C：RGB-derived / no-depth operating reference

这里不要马上训练。

先确认：

- 当前模型 zero-depth performance = 52.55；
- B0 zero-depth performance 是多少；
- 如果存在兼容的 RGB-only/reference inference，性能是多少。

它可以帮助判断：

> 52.55 的瓶颈究竟来自“没有 Depth 本身”，还是 A2 对 missing mode 没有学好。

但这不是第一优先级。

------

# 23. clean-control 应该什么时候跑

这里我对下级模型的结论稍作修改。

它建议：

> 现在立即跑 clean-control。

从实验科学上没问题。

但从当前工作流效率上：

**clean-control 不应该阻塞模块设计。**

我建议：

### 现在

先完成：

1. B0 × frozen 10-condition historical stress reference；
2. Oracle-A；
3. Oracle-B；
4. 冻结新模块设计。

这些都是低成本诊断。

### 同期或模块方案冻结后立即

启动：

```
MMFR-A2-clean-control-v3
```

因为无论未来选择 C1 还是 C2：

> 最终要判断 corruption training 本身是否有效，都需要 clean-control。

因此最佳实践是：

> **模块设计与 clean-control 可以并行，而不是二选一。**

如果现在云 GPU 正好可用、约 20 元成本可以接受，也完全可以立即启动 clean-control。

但不需要等它跑完才设计下一模块。

------

# 24. 下一阶段建议的实验顺序

建议严格按照：

### Step 1 — B0 stress reference

B0：

10-condition frozen Main-Val。

目的：

建立 historical failure profile。

------

### Step 2 — Oracle-A

A2 epoch-420：

spatial_dropout validity-aware geometry Oracle。

至少评价：

- clean；
- spatial_dropout@0.75；
- entire_missing@1.0。

------

### Step 3 — Oracle-B

A2 epoch-420：

perfect inverse misalignment correction。

至少评价：

- clean；
- misalignment@0.75。

------

### Step 4 — Decision Gate

根据 Oracle：

#### 如果 Oracle-A 明显为正

进入 C1。

#### 如果 Oracle-A 无收益，但 Oracle-B 明显为正

进入 C2。

#### 如果二者都明显为正

以 C1 为主模块，C2 作为扩展或第二 component。

#### 如果二者都没有明显空间

停止 geometry reliability / alignment 主线，重新考虑：

- learned missing mode；
- feature-level robustness；
- teacher-student；
- representation invariance。

------

### Step 5 — Freeze module protocol

模块编码之前冻结：

- model modification；
- training objective；
- ablations；
- selector；
- evaluation；
- success gate。

------

### Step 6 — Clean-control

如果尚未并行启动，则此时正式训练 clean-control。

------

### Step 7 — New module training

再进入真实模型开发。

------

# 25. 下一模块成功标准

不建议只看某一个 failure。

建议继续使用已有六 single-condition macro score 作为主 robustness metric。

同时新增设计目标：

### Robustness target

相对正式 control：

六 single-failure macro mIoU：

$\Delta_{robust}\ge1.0$ 

### Clean preservation

clean：

$\Delta_{clean}\ge-0.5$ 

### Breadth

至少 5/6 single conditions：

$\Delta_c\ge0$

且任何单条件：

$\Delta_c>-1.0$ pp

### Statistics

location-group paired bootstrap：

95% interval lower bound：

$>0$

若这些 gate 已在项目协议中冻结，则沿用，不重新根据结果调门槛。

------

# 26. 下一模块最低消融矩阵

如果最终进入 C1：

1. A2；
2. A2 + pairwise validity rule；
3. A2 + learned reliability only；
4. A2 + reliability-conditioned geometry；
5. A2 + missing-mode objective；
6. Full。

如果最终进入 C2：

1. A2；
2. Oracle alignment；
3. learned global shift；
4. learned block-wise shift；
5. correction confidence；
6. Full。

不要一开始就做十几个消融。

------

# 27. 防撞车风险的最终判断

## C1

### 风险

高。

因为 reliability-aware fusion 已经是成熟方向。

### 可保留的新颖空间

必须把创新点收窄为：

> **DFormer geometry prior 中 pairwise evidence validity 的显式建模。**

关键不是：

“我们预测 reliability。”

而是：

“我们重新定义不可靠深度进入 geometry attention bias 的方式。”

如果未来论文写成 generic reliability fusion，撞车风险很大。

------

## C2

### 风险

中高。

RGB-D alignment / registration 已经成熟。

### 可保留的新颖空间

限定：

> **只校正 geometry prior evidence，而不校正完整 feature / depth image。**

再配合：

- low-DoF；
- identity-safe；
- segmentation-driven；
- DFormer GSA-specific。

新颖性空间会明显更好。

------

## C3

风险最高。

需要非常谨慎。

当前不推荐。

------

# 28. 最终模块设计方向

我建议当前把研究主题暂时定义为：

## Reliability-Conditioned Geometry Adaptation for Degraded RGB-D Segmentation

更具体的问题陈述是：

> DFormer assumes depth-derived pairwise geometry is valid and spatially aligned. Under local missing depth or RGB-D misregistration, this assumption breaks: invalid depth can create false pairwise geometry, while misaligned depth provides geometrically meaningful but spatially misplaced evidence. The next module should make geometry evidence conditional on its validity and spatial consistency while preserving the original geometry path under clean inputs.

这个问题定义比“做一个 reliability module”更清晰。

------

# 29. 当前三个核心科学问题

下一阶段最值得回答的依次是：

## 1. Invalid Depth

> 当 Depth 局部无效时，错误的 pairwise geometry 是不是主要性能瓶颈？

由 Oracle-A 回答。

## 2. Misregistered Depth

> 当 Depth 与 RGB 错位时，geometry coordinate correction 能恢复多少性能？

由 Oracle-B 回答。

## 3. Missing Depth

> 在没有任何 Depth evidence 时，应该学习 no-depth mode，还是需要生成 substitute geometry？

这个问题留在前两个 Oracle 后进一步处理。

------

# 30. 最终结论

### 对 A2 当前状态

A2 是一个有价值的 robustness screening identity，但目前还不能证明其 corruption training 带来了因果提升。

### 对 B0

B0 应继续作为 historical reference。

建议额外补跑：

**B0 × frozen 10-condition**

以获得方向性 stress profile。

但它永远不自动等于 clean-control。

### 对下一模块

暂不直接训练 C1/C2/C3。

先运行：

1. validity-aware geometry Oracle；
2. perfect alignment Oracle。

这两个实验将比继续阅读更多泛 reliability 文献更直接地决定路线。

### 最推荐候选

如果 validity Oracle 为正：

> **C1：Validity-/Reliability-Conditioned Pairwise Geometry**

如果 alignment Oracle 的空间更大：

> **C2：Consistency-Guided Geometry Realignment**

C3 暂缓。

### 对 entire_missing

它仍然是最严重的 failure。

但不能称 52.55 为性能上限。

正确理解是：

> 当前模型在关闭有效 Depth geometry evidence 后得到 52.55；下一阶段若想进一步提升，需要改变 no-depth operating mode 或提供 substitute geometry，而不是仅继续抑制已不存在的 Depth evidence。

### 对 clean-control

正式因果结论最终必须有 clean-control。

但它不需要阻塞下一模块的 Oracle 与设计工作。

最合适的是：

> **低成本 Oracle / B0 stress profile 与 clean-control 训练并行推进。**

------

# 31. 建议下一步执行清单

按优先级：

1. **B0 × frozen 10-condition Main-Val**：历史方向参考；
2. **Oracle-A：spatial_dropout validity-aware pairwise geometry**；
3. **Oracle-B：misalignment perfect inverse correction**；
4. 根据 Oracle 选择 C1 或 C2；
5. 冻结下一模块 protocol + ablation matrix；
6. 同期启动或完成 `MMFR-A2-clean-control-v3`；
7. 再进入模块编码与训练；
8. 模块完成后用统一 10-condition + clean-control 做正式比较；
9. 最后才考虑 official test。

official test 继续保持 `sealed_unread`。

------

# 32. 当前最重要的防偏原则

后续开发过程中始终坚持：

> **先证明“这个机制理论上有可回收空间”，再训练可学习模块。**

也就是：

**Oracle → Mechanism → Learnable Module → Paired Control → Statistical Confirmation**

而不是：

**看到 failure → 堆模块 → 跑一次 → 看 mIoU。**

这是当前阶段最能减少试错成本、也最能保护论文创新性的路线。