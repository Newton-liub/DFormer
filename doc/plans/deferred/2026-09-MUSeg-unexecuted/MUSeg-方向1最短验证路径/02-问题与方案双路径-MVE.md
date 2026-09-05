# MUSeg 方向1最短验证路径：问题与方案双路径 MVE

> **文档角色：** 方向1的两条互补最小可行实验（MVE）设计；补充总方向规划，不是代码、GPU、训练、云资源或 official test 授权。
> **形成或核验时点：** 2026-09-01。
> **上级方向：** [`00-总方向规划.md`](./00-总方向规划.md)。
> **原拟议任务卡：** [`01-新对话最小上下文与当前任务.md`](./01-新对话最小上下文与当前任务.md)。
> **计划状态：** 2026-09-05 起延期且未执行；下文步骤只在未来重新启用并取得新授权后适用。
> **研究输入：** [`研究方案设计专用提示词.md`](./参考资料/研究方案设计专用提示词.md) 与 [`段落主题1_论文编号.txt`](./补充内容/段落主题1_论文编号.txt)。

## 0. 共同前提：两条路径怎样并行

路径 A 和路径 B 是两个互补的科学验证问题：A 检验问题是否真实且关键，B 检验最低复杂度方案是否可行。两条路径可以在共享协议和同源 logits 准备完成后分别执行，但不应在身份未闭合时无条件并跑。[PR003][PR005]

两条路径共同固定：epoch 420 `DFormerv2-S + ham` Quick-B0、独立的 calibration 集 `C`、独立评价集 `E`、RGB/Depth 输入契约、FP32 evaluator、原始 Label 网格、ignore mask、样本顺序和 official-test `sealed_unread` 状态。[PR001][PR005]

`C` 只用于拟合校准器和冻结风险阈值；`E` 只用于路径 A 问题判断和路径 B A/B 判断，不参与方法选择或阈值回调。`val-dev` 已参与 B0 checkpoint 选择，不能直接充当 `C` 或 `E` 的独立证据。[PR001][PR003]

---

# 路径 A：问题—假设验证路径

## A1. 核心假设识别

**核心假设：** 当只提高输入中无效 Depth 的可控比例时，冻结 RGB-D 分割模型会出现可重复的条件正确率下降，并伴随高置信错误、概率失校准或风险排序恶化；若这些现象不出现，“退化条件下概率不可信”就不足以作为后验校准的主瓶颈。[RE068][RE131][RE666][RE752]

这里把“Depth 有效性下降”作为可控变量，把置信度和风险指标作为结果变量。该设计检验的是预注册合成退化条件下的方向关系，不等同于真实传感器故障概率或部署安全保证。[RE346][RE460]

## A2. 最小可行实验设计（MVE）

### 实验目标

直接回答：**在 checkpoint、RGB、标签、样本、推理几何和评价代码均不变时，只增加无效 Depth 比例，是否会在独立评价集 `E` 上造成可量化且可复核的可信度损失？**[RE429][RE068]

### 实验步骤

1. **冻结身份。** 绑定 epoch 420 checkpoint、checkpoint SHA-256、RGB/Depth 输入契约、FP32 推理、原始 Label 网格、ignore mask、代码提交、protocol 和 evaluator identity。[PR001][PR003]
2. **冻结数据。** 建立与 train-dev、val-dev、official test 隔离的 `C/E`，以 location group 划分；路径 A 只使用 `E`，每张样本保留 clean 版本和一一对应的退化版本。[PR003][PR005]
3. **冻结退化变量。** 选择一个 Depth corruption 主变量，例如无效像素比例 `q`；在运行前冻结 `q` 水平、mask 类型、fill 类型、seed 和实际覆盖率记录。具体数值当前仍待确认。[PR003][PR005]
4. **生成 paired 条件。** 对同一 sample、同一 `q`、mask 类型和 seed 复用完全相同的 mask，只改变 Depth；RGB、Label、样本顺序和 evaluator 保持不变。全零输入只能命名为 Depth 缺失/退化，不能命名为严格 Depth-only 模型。[RE346][RE461]
5. **一次性导出同源输出。** 对 clean 和预注册退化条件生成 15 类逐像素 float32 logits，记录 sample ID、condition ID、mask 身份、实际覆盖率、checkpoint/split/protocol SHA-256 和产物哈希；后续所有方法只读取这批 logits。[PR004][PR005]
6. **计算最小指标。** 以有效像素为基础计算 ECE、NLL、Brier Score、高置信错误率（例如预注册 `HCE@t`）、mIoU 以及风险—覆盖指标；正常和每个退化条件分开报告，不能只给一个混合平均数。[RE666][RE131]。这些指标的规范定义和密集预测适配来源仍待补充。
7. **进行 paired 汇总。** 以图像或 location group 为重采样单位，比较 clean 与退化条件的指标差，检查退化强度和可信度损失是否保持预注册方向；不得把所有像素当作完全独立重复。[RE447]
8. **执行停止门禁。** 若数据身份、mask 覆盖、logits 有限性、标签/掩码对齐、样本顺序或 official-test allowlist 失败，停止本批次并保留现场，不解释科学指标。[PR003][PR004]

### 所需数据 / 材料

- 独立且 group-aware 的 `C/E` 标注数据池；路径 A 需要 `E` 的 RGB、Depth、Label 和清单。[PR003][PR005]
- 已核验 epoch 420 checkpoint、RGB/Depth 解码与归一化、ignore index、原始 Label 网格 evaluator。[PR001][PR005]
- 确定性 Depth mask 生成器、预注册 condition manifest、mask seed、实际注入比例记录和 SHA-256。[PR003][PR004]
- 现有 MUSeg 推理链、PyTorch/NumPy、离线 ECE/NLL/Brier、HCE 和风险—覆盖计算；不需要新训练。[PR004][RE666]

现有 `RE136` 只能作为后验校准邻近案例，不是上述指标或温度缩放的直接方法来源。

### 关键成功 / 失败指标

以下数值必须在任务 1 中结合实际数据覆盖和统计预算确认；这里给出判据类型，不把未经项目核验的数字伪装成固定事实。[PR003][PR005]

- **成功（G-A）：** 至少一个主可信度/风险指标随 `q` 增大呈稳定、可复核且达到预注册最小效应的恶化方向；paired group-level 不确定区间支持该方向，并且不只由单一 group 或单一 seed 驱动。[RE447][RE068][RE666]
- **失败（N-G-A）：** `q` 增大没有稳定增加失校准或高置信错误，或效应只在单一条件/单一 group 出现且不确定区间覆盖零；停止扩展复杂校准器，重新审视问题定义。[RE447][PR005]
- **不确定（I-A）：** mIoU 明显下降但概率指标不变，或总体与主要分层方向冲突；保留“分割退化与后验失真未被证明是同一问题”的结论，不强行判为成功或失败。[RE666]

### 所需最简资源

- 一份已审计的独立 `E` 数据和一个可复算的 Depth corruption manifest。[PR003][PR005]
- 一次冻结模型推理环境；不需要反向传播、重新训练或云端训练。[PR001][PR005]
- 能存储同源 float32 logits 和结构化 manifest 的本地磁盘；后续指标计算可在普通 CPU 完成。[PR004][PR005]

### 备选及简化方案

若暂时没有退化推理能力，可在同一份 clean logits 上做预注册的 logit sharpness/temperature 扰动，观察是否能制造过度自信和失校准。该方案只检验“后验尺度失真是否足以形成问题”，不能证明 Depth 退化是根因；结果只能作为工程筛查。当前 `RE136` 不是 temperature scaling 的直接方法来源；对应基础论文待补充。

### 路径原理解释

路径 A 最短，是因为它只改变一个候选根因变量 `q`，并固定 checkpoint、样本、标签、几何、输入契约和评价代码；paired 设计把样本差异从因果链中排除。[PR003][RE447]

路径 A 最关键，是因为只有先证明问题会被主动制造并重复观察到，才有理由投入更高自由度的校准器、退化迁移或风险门控。若只看到 mIoU 下降而没有可信度损失，后验校准未必是正确的解决方向。[RE666]

---

# 路径 B：方案—假设验证路径

## B1. 核心假设识别

**核心假设：** 在独立 `C` 上只拟合一个全局正温度 `T`，相对于未校准 Softmax，能够在未参与拟合的 `E` 上改善预注册概率指标，同时保持逐像素 argmax、mIoU 和逐类分割预测不变。[PR005]

全局温度缩放是最低复杂度方案：它只学习一个标量并作用于冻结 logits。正温度对每个像素的所有类别做统一缩放，理论上不改变 argmax，因此能把概率数值改善与分割预测改变分开。该方法主张的直接 temperature scaling 文献待补充；`RE136` 仅作后验校准邻近背景。

## B2. 最小可行实验设计（MVE）

### 实验目标

直接回答：**在同一冻结 checkpoint、同一份 clean logits、同一 `C/E`、同一标签和同一有效像素掩码上，全局温度缩放相对未校准 Softmax 是否在独立 `E` 上改善概率可靠性？**[PR005]

### 实验步骤

1. **复用同源 clean logits。** 直接读取任务 2产生的、绑定 checkpoint/input/evaluator/protocol 身份的 clean logits；Baseline 与方案不重复推理、不改变样本过滤。[PR004][PR005]
2. **只在 `C` 拟合标量。** 以有效像素为单位按预注册 NLL 目标拟合 `T>0`，记录目标、参数约束、收敛状态、回退规则和产物哈希；`E` 在拟合阶段不可用于调参。对应的 temperature scaling 直接方法来源待补充。
3. **冻结并应用。** 在 `E` 上比较 `softmax(z)` 与 `softmax(z/T)`，两臂共享同一 logits、标签、ignore mask、样本顺序和评价实现。[PR004]
4. **执行最小 A/B。** 报告 NLL、ECE、Brier Score、可靠性分箱、HCE 和风险—覆盖指标；同步逐像素比较 argmax、mIoU、逐类 IoU 和改变像素比例。[RE666][RE131]。这些指标的规范定义和 dense prediction 适配文献仍待补充。
5. **可选迁移检查。** 只有 clean A/B 完成后，才把已冻结的同一个 `T` 应用于路径 A 的一个预注册退化条件；不在退化结果出来后重新调温度。该步骤用于发现迁移边界，不是 B 的 clean 可行性前置条件。[PR005][RE226]
6. **进行 paired 汇总。** 以图像或 location group 为重采样单位；若 ECE 改善但 AURC 不变，分别记录概率数值改善和错误排序未改善。[RE131][RE447]

### 所需数据 / 材料

- 路径 A 共享的独立 `C/E` 身份、样本清单和 group 泄漏审计；`C` 拟合、`E` 评价。[PR003][PR005]
- 同一冻结 checkpoint 产生的 clean float32 logits、标签、ignore mask、protocol 和产物 manifest。[PR004][PR005]
- 未校准 Softmax、全局温度缩放、NLL/ECE/Brier、可靠性和风险—覆盖计算；不需要修改模型结构或训练参数。[PR004]
- `RE136` 仅作为“冻结主干之外进行后验校准”的邻近案例；temperature scaling 及其指标的直接方法来源仍待补充。
- 温度拟合与离线指标可在 CPU 完成；只有 logits 尚不存在时才需要一次冻结模型推理。[PR004][PR005]

### 关键成功 / 失败指标

- **成功（G-B）：** 独立 `E` 上至少一个预注册概率指标达到最小改善，且 argmax 完全一致、改变像素比例为 0、mIoU 和逐类 IoU 在预注册容差内一致；具体改善比例和区间规则待任务 1确认，直接 temperature scaling 与 calibration 评价来源待补充。[PR005]
- **增强证据：** 在相同覆盖率下选择性风险不升高，或相同风险目标下覆盖率提高；这支持风险分数用途，但不能由 ECE 单独推出。[RE131][RE226]
- **失败（N-G-B）：** 独立 `E` 上 NLL、ECE、Brier 均无稳定改善或至少一个主指标明显恶化；或出现 logits 身份、样本过滤、mask 或 argmax 契约失败。此时不直接升级向量缩放或类别条件温度。[PR005][RE447]
- **部分成功（P-B）：** 概率指标改善但 AURC 不变，写成“概率数值改善、错误排序未改善”；clean 有效但退化迁移失败，写成“同域校准有效、迁移存在边界”。[RE131][RE226]

### 所需最简资源

- 已审计的 `C/E` clean float32 logits、标签、ignore mask 和 manifest。[PR004][PR005]
- 普通 CPU 即可拟合一个温度和计算指标；不需要 GPU、训练或云资源，除非首次导出 logits。[PR004][PR005]
- 两个离线方法注册项：`uncalibrated_softmax` 和 `global_temperature_scaling`。[PR004]

### 备选及简化方案

先以少量已审计 group 进行两臂 smoke check，只检查输入、NLL、argmax、mask 和 manifest 契约；结果只能标为工程可行性信号，不能替代独立正式评价。若只能使用 `val-dev` 做 smoke check，必须明确它已参与 B0 模型选择，不能作为正式 `C/E` 证据。[PR001][PR003][RE447]

### 路径原理解释

路径 B 最短，是因为它只学习一个标量，Baseline 与方案共享冻结 logits、数据、标签、掩码和 evaluator；观察到差异时，不需要把收益归因于模型容量、训练随机种子或数据增强。[PR004]

路径 B 最关键，是因为全局温度缩放是低成本、可解释的第一道方案。它失败时，能够阻止项目无证据地堆叠复杂校准器；它成功时，能够以最小投入支持继续检查退化迁移和风险阈值。其直接方法依据仍待补充，`RE136` 仅为后验校准邻近案例。[PR004]

---

# 3. 两条路径互为补充

- **A 支持、B 支持：** 问题和最低复杂度方案均得到初步支持；冻结温度和协议，进入预注册的 clean→单退化迁移，再考虑一个受约束的高自由度方法。[PR003][RE226]
- **A 支持、B 不支持：** 保留“Depth 退化制造可信度瓶颈”的问题证据；停止无依据增加模型自由度，优先重审风险分数、条件化校准或问题边界。[RE131][PR005]
- **A 不支持、B 支持：** 将方向收窄为 clean/同域概率校准；不再把 Depth 退化作为主问题叙事，也不把 B 的收益写成 A 的证据。[PR005]
- **A、B 均不支持：** 进入 `stop` 或 `inconclusive`，保留可复核失败现场；不通过调阈值、换分箱、追加方法或读取 official test 挽救方向。[PR003][RE447]
- **任一路径身份失败：** 若 checkpoint、数据职责、logits、mask、标签对齐、有限性或 official-test allowlist 失败，立即标记 `protocol-blocked`；修复后使用新的 protocol identity 重新开始，原失败现场不覆盖。[PR003][PR004]

A 与 B 的成功均只表示早期 MVE 的支持，不表示三 seed 论文级结论、真实故障概率、部署安全保证或训练后性能结论。[PR001][PR005]

# 4. 成功/失败后的行动路径

1. **A 支持且 B 支持：** 进入任务 3 的正式 clean→退化迁移；只使用运行前冻结的条件和阈值，随后再决定是否加入保序回归或一个受约束的高自由度方法。[PR003][RE226]
2. **A 支持且 B 不支持：** 保留问题证据，停止直接扩大方法复杂度；另立问题重审协议，比较风险分数或条件化校准前先重新确认研究价值。[PR005][PR005]
3. **A 不支持且 B 支持：** 收窄为同域概率校准方向；不投入以 Depth 退化为中心的迁移或机制叙事。[PR005]
4. **A/B 均不支持或任一路径不确定：** 标记 `stop` 或 `inconclusive`，保留结果和恢复点；只有用户批准新 protocol 后才能补充一次有针对性的验证。[PR003][RE447]
5. **身份/隔离/完整性失败：** 先修复数据、工具或 manifest，再重新冻结清单和哈希；任何身份不闭合的产物不进入科学结论。[PR003][PR004]

# 5. 参考文献与证据映射

- **PR001**：DFormer 项目，[`doc/main/MUSeg-current-status.md`](../../../../main/MUSeg-current-status.md)，Quick-B0、checkpoint、`val-dev`、official-test 和当前授权边界的实时事实入口。
- **PR002**：DFormer 项目，[`../MUSeg-A2-B2深度有效性/00-总方向规划.md`](../MUSeg-A2-B2深度有效性/00-总方向规划.md)，并行计划的身份冻结、任务门禁和恢复点结构参考。
- **PR003**：本计划 [`00-总方向规划.md`](./00-总方向规划.md) 与共享任务约定，定义独立 `C/E`、数据职责和 protocol 门禁。
- **PR004**：本计划共享工具链约定，定义同源 float32 logits、标签、ignore mask、manifest 和最小离线检查。
- **PR005**：DFormer 项目，[`./参考资料/研究方案设计专用提示词.md`](./参考资料/研究方案设计专用提示词.md)，DFormerv2-S 后验校准研究设计、退化边界和结论限制。
- **PR006**：DFormer 项目，[`doc/plans/deferred/2026-09-MUSeg-unexecuted/MUSeg-方向1最短验证路径/补充内容/段落主题1_论文编号.txt`](./补充内容/段落主题1_论文编号.txt)，RE/PR 编号索引。
- **RE068**：Xin, Y. et al., “M-SURE: Enhanced and reliable safety monitoring in low-light mines,” *Advanced Engineering Informatics*, 2026；支持低照矿山中的不确定性和高置信错误问题背景。
- **RE131**：Pan, Y. et al., “Sequential Probabilistic Descriptor via Uncertainty-Aware Multi-Modal Fusion for Safety-Critical Place Recognition,” *IEEE Robotics and Automation Letters*, 2026；支持多模态不确定性估计和高不确定性结果过滤背景。
- **RE136**：Myagmarsuren, D. et al., “Multimodal Uncertainty-Aware Gating Fusion and Iterative Feedback Refinement for HSI-LiDAR Open-Set Classification,” *Remote Sensing*, 2026；仅支持冻结主干之外进行后验校准和独立验证阈值的邻近思路，不是经典 global temperature scaling 的基础来源。
- **RE165**：Liang, G. L., “RGB-D semantic mapping for underground robotic inspection using an attention-enhanced and boundary-refined DeepLabv3+ network,” *Industrial Robot*, 2026；支持地下 RGB-D 场景、边界和空间连贯性评价动机。
- **RE346**：Xu, J. H. et al., “Modality-Resilient Multimodal Industrial Anomaly Detection via Cross-Modal Knowledge Transfer and Dynamic Edge-Preserving Voxelization,” *Sensors*, 2025；支持模态内不完整和整模态缺失条件的退化背景。
- **RE429**：Li, S. Y. et al., “MUSeg: A multimodal semantic segmentation dataset for complex underground mine scenes,” *Scientific Data*, 2025；支持 MUSeg RGB-D 数据、矿下场景和 15 类标注背景。
- **RE447**：Huang, S. et al., “Ground-Type Classification from Earth-Pressure-Balance Shield Operational Data with Uncertainty Quantification,” *Applied Sciences*, 2025；支持 bootstrap 区间和不确定性量化作为统计设计参考。
- **RE460**：Jiang, F. et al., “Closing the Calibration Gap: A Real-Time Multi-Modal Fusion Framework for 3D Semantic Segmentation,” *IEEE Transactions on Intelligent Vehicles*, 2025；支持跨模态弱标定/错位导致的性能退化背景。
- **RE461**：Park, E., “Prompt the Missing: Efficient and Robust Audio-Visual Classification Under Uncertain Modalities,” 2025；支持缺失模态和不确定模态条件的鲁棒性背景。
- **RE666**：Hildebrand, M. et al., “Assessing Distribution Shift in Probabilistic Object Detection Under Adverse Weather,” *IEEE Access*, 2023；支持分布偏移下概率可靠性和不确定性评价背景。
- **RE752**：Geng, K. K. et al., “Robust dual-modal image quality assessment aware deep learning network for traffic targets detection of autonomous vehicles,” *Multimedia Tools and Applications*, 2022；支持低照、模糊、噪声和质量感知退化背景。
- **PR005**：本计划的设计性判据登记；A/B 的退化强度、最小效应、分箱、覆盖率、重采样和置信区间数值均待任务 1结合项目数据确认，不是外部文献的普适阈值。

## 6. 运行前冻结清单

在任何正式 MVE 前，必须冻结并记录：

- `C/E` 来源、group 划分、样本清单和 SHA-256；
- checkpoint、输入契约、evaluator、ignore mask 和 logits 格式；
- 路径 A 的 `q`、mask、fill、seed、实际覆盖率和主指标；
- 路径 B 的温度拟合目标、`T` 约束、方法注册和 Baseline 定义；
- ECE/NLL/Brier/HCE、风险—覆盖、AURC 的分箱与聚合规则；
- 最小效应、方向一致性、重采样/置信区间和 `supported/not-supported/inconclusive` 判据；
- 任务授权、命令、环境、产物哈希、失败/恢复历史和 official-test 拒绝状态。[PR003][PR004][PR005]

未冻结的项目不得通过结果后补写；若无法冻结，合法输出为 `protocol-blocked`。[PR003]
