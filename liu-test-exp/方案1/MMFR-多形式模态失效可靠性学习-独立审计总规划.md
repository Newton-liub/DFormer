# MMFR-多形式模态失效可靠性学习：独立审计总规划

> **文档状态：** 2026-09-14（v2 修订版）独立审计快照，2026-09-15 补入门禁核验与第二轮本地资格证据；A1 v1 已完成 CPU qualification，A2 v1 已完成 code qualification 与本地 GPU 单步 qualification，但随后在正式 500 epoch 尚未执行的时点进行了协议修订，v1 已冻结归档为 `superseded-before-formal-training`；v2 已完成代码修订、A1 v2 severity/burden 定点审计、可复跑的 A2 v2 CPU qualification（`74` 项断言）、`initial-state-equivalence`、`gradient-path isolation` 与 `amp-update-path isolation` 三项门禁，以及 `train-dev` Depth 有效性分布与 validity transport 两项事实审计。**当前状态标识为 `local-qualifications-passed-awaiting-senior-review-of-dataset-validity-audit`：本地资格已补齐，但尚未由高级模型确认“数据集有效性审计不需要改变监督语义”，也尚未完成 source identity 冻结，因此还不能升级为 `eligible-for-cloud-capacity-probe-authorization`。** 正式训练、云端 probe、MMFR checkpoint 评价和 official test 均未授权、未完成。
> **事实入口：** `doc/main/MUSeg-current-status.md`；本文只在该入口基础上重组审计叙事，不替代实时状态。
> **本次修订性质：** 只改协议与代码语义，不产生任何模型效果结论。v1 的历史 CPU/GPU 证据、DVC/DVG 负结果和已记录哈希均未被改写。

## 0. 文档用途、形成时点与阅读规则

- **文档用途：** 本文件是一份可以直接交给“完全看不到 DFormer 仓库内容”的外部 agent 审计的自包含总规划。它把项目背景、历史依据、当前实现、冻结设计、已完成资格证据、正式实验计划、授权边界、风险和审计问题放在同一份叙事中。
- **形成时点：** 2026-09-14；v2 修订在同一时点完成，未跨越任何正式训练或评价事件，因此本次修订是“结果产生前的协议修订”，不是对已有结果的追认或解释。
- **外部 agent 的使用方式：** 外部 agent 不应假定自己能够读取仓库、云端目录、checkpoint、日志或本文引用的文件。审计应先把本文中的“事实”“冻结设计”“建议”“未授权”“未执行”五种状态分开，再逐项要求补交证据。本文引用的路径是证据索引，不是外部 agent 已经看见这些文件的声明。
- **事实优先级：**
  1. 直接核验的当前状态、工作区代码、协议模板、已有运行产物及其哈希；
  2. `doc/main/MUSeg-current-status.md` 的实时状态和恢复点；
  3. `doc/main/MUSeg-open-decisions.md` 的研究选择与边界，重点是第 8、12、13 节以及本次新增的 v2 开放项；
  4. 形成时点的计划和历史报告；
  5. 本文对未来实验的重组说明。
- **语言规则：** “已完成”只表示实现或资格检查已经发生；不表示模型效果已证明。“已通过但不构成效果”表示链路或接口通过了定点检查；“冻结”表示实验开始前不得随意改变；“建议”表示尚未成为当前主门槛；“未授权”表示即使计划已经写好也不能执行；“未执行”表示该项检查连一次都没有跑过。
- **硬边界：** 截至本文形成时，没有正式 500 epoch MMFR 训练、没有新训练 checkpoint、没有云端 probe、没有 MMFR checkpoint 评价，也没有读取 official test。本文不能被解释为这些工作已经完成。
- **v1 与 v2 的关系：** v1 是已冻结的历史身份，其模板、CPU qualification 和本地 GPU 单步 preflight 事实保持不变；v2 是替代它的当前身份。两份模板同时存在、互不导入，v1 只作为审计参照。

## 1. 一页式执行摘要

### 1.1 现在已经知道什么

- DFormer 是面向 RGB-D 语义分割的模型族；本项目使用其 DFormerv2-S 变体，在 MUSeg 矿井场景数据上处理对齐的 RGB 图像、Depth 图像和语义标签。DFormerv2-S 是 **RGB-primary semantic path + Depth geometry-prior path 的非对称架构**：Depth 作为 geometry prior 进入 GSA（Geometry-aware Self-Attention，几何感知自注意力），不是与 RGB 对称的显式语义编码分支。
- MUSeg 当前开发职责为 `train-dev=1277`、`val-dev=318`，另有封存的 official test（`1576` 条，`sealed_unread`）。`val-dev` 参与 checkpoint 选择，因此它是开发证据，不是独立测试集。
- Quick-B0 是已冻结的 RGB single-seed development 基线：官方预训练权重、epoch 420 checkpoint、五尺度加水平翻转的 10-view evaluator，主指标为 mIoU `58.79`、mAcc `69.91`、mF1 `72.73`。它用于内部模块对照，不是三 seed 论文级复现，也不是 official test 结果。
- 过去把 Depth 置零作为失效动作的 DVC 方向没有得到支持性结论；随后“知道真实坏区后，直接关闭 GSA 的 Depth contribution”的 DVG-B1 Oracle 方案在完整配对评价中使主指标下降，最终以 `oracle-not-supported` 关闭。这些是历史结论，不能改写成成功，也不能从中推出所有可靠性学习都无效。
- 当前方向 MMFR（Multi-Modal Failure Robustness，多模态失效鲁棒性）在 v2 修订后的阶段含义是：**A1 = 合成失效生成 + 由 corruption 推导的 reliability surrogate**；**A2 = Depth corruption robustness training + 并行 reliability estimation**；**B1 = 第一个真正的 reliability-aware segmentation 阶段**，即首次让预测 reliability 影响 DFormerv2 的模型动作。A2 的 predicted reliability 不进入 backbone、decoder 或 geometry prior，因此 A2 的分割性能变化不能归因于 reliability head。
- A2 v2 的正式研究问题仍不是“最终可靠性融合是否有效”，而是先验证：在不改 DFormerv2 geometry prior、不做 clean/corrupt 双前向的条件下，Depth 失效输入、连续 target、辅助损失和公平训练身份是否能被正确接入。
- v2 修订解决了四条经审计的协议缺陷：(1) `gaussian_noise`/`blur`/`quantization` 的 severity 被二次编码；(2) `blur` 与 `misalignment` 的 severity 是绝对像素量，在 `480×640` 训练 crop 与 `932×1082` 原始网格上不等价；(3) Depth reliability target 忽略了 MUSeg 原生无效 Depth；(4) RGB reliability 通道被计分，但它在 A2 中恒等于 1。
- 定点证据：`tools/mmfr/severity_burden_audit.py` 在 CPU 上退出码 `0`、`OVERALL: PASS`、`violations=0`，确认 severity 只编码一次、burden 单调不减、target 单调不增、blur/misalignment 相对尺度一致、结构性 burden 只取两个值、空 spec 严格 no-op；报告 SHA-256 为 `86e468c040648a790675456db1b828a5a71e560b849c75b4b7e9a3ca926c2f52`，主代理复跑得到同一哈希。
- A2 v2 CPU 定点检查（主代理临时内联脚本，真实 `train-dev` 样本、冻结 crop）通过 47 项断言、0 项失败：clean 样本对模型输入是精确 no-op；clean 的 Depth target 等于原生 Depth 有效性；crop/pad 区保持中性 target 并恒为精确零；corrupt 样本只改 Depth、RGB 逐位不变；invalidity 记账闭合且与独立重算的掩码一致；同 seed/epoch/iteration/rank/slot 复现完全相同的输出。该脚本为一次性临时脚本，未留在工作区。
- 定点检查同时暴露并修复了一个真实缺陷：v2 invalidity 记账曾断言 `post_corruption_invalid_pixels == natural_invalid_pixels + synthetic_missing_pixels`，但 `gaussian_noise` 与 `misalignment` 会把原生无效（raw Depth == 0）像素变成非零，使该等式必然失败并中止首个真实 corrupt 样本。修正为分开记录 `newly_valid_pixels` 并断言闭合平衡 `post + newly_valid == natural + synthetic`，仍保持 fail-closed。v1 没有这套记账，未受影响。
- **gradient-path isolation 门禁（已执行并通过）：** `tools/mmfr/gradient_path_isolation.py` 已写好，并已于 2026-09-15 01:32 在本地 GPU 上实际执行通过，不再是未执行项。它证明的是：在完全相同的模型状态、输入、corruption、RNG 与 segmentation loss 下，`lambda_rel=0` 与 `lambda_rel=0.1` 两次 backward 的共享参数（backbone/decoder/geometry prior）梯度必须在容差内相同，reliability head 参数只在后者有有效梯度，head 开/关时 segmentation logits 一致。实测结果：`714` 个共享参数梯度**逐位完全相同**（`max_abs_difference=0.0`、`over_tolerance_count=0`、`absent_in_both_runs_count=0`），reliability head 的 `6` 个参数在 `lambda_rel=0` 时为恒零梯度、在 `lambda_rel=0.1` 时梯度范数为 `0.05855773380379924`，两次 segmentation loss 与两种情况下的 logits 均逐位相同。因此**通过后应冻结的结论是：A2 分割性能变化的直接原因是 corruption exposure，而 reliability head 在该阶段只承担 estimator qualification**；这条数学事实同时说明，本项目不需要再额外消耗一次完整 500 epoch 的“corruption-only”训练来证明它。**证据边界：** 该检查是 batch size `1`、FP32、单进程、单样本、一次 corrupted 抽样的定点检查，不覆盖冻结 batch size `10`、AMP、DDP、完整 epoch、checkpoint save/load、evaluator、云资源或 official test，也不提供任何 mIoU 或鲁棒性收益结论。**正式训练本身仍未授权。**
- 正式训练职责已冻结为云端单 GPU，默认 RTX 4090；RTX 5090 只有在 4090 缺货、冻结 batch size 10 无法装入 24 GiB，或配对 probe 证明单位样本成本更低时才使用。当前没有云实例，没有训练授权，也没有容量/吞吐 probe 授权。

### 1.2 大白话版方案

先拿两个完全公平的模型从同一份官方预训练权重重新训练：一个始终看干净的 RGB-D 输入，另一个在每个训练样本上按固定随机规则损坏 Depth，并额外学习“这块 Depth 现在有多可信”。目前这个可靠性预测只接受辅助监督，不会直接替 DFormer 关掉某条融合路径；A2 阶段也不该把分割性能变化解释成可靠性头的功劳。v2 修好了四件事：损坏强度不再被算两次、模糊和错位的强度不再依赖图像分辨率、真实缺失的深度像素不再被当成“可信度 1”，以及只在 Depth 通道上算可靠性损失。那道“证明打开可靠性头不会改变分割网络梯度”的检查已经在本地显卡上真实跑过并通过：打开后分割网络的 714 个参数梯度与关闭时逐位完全相同，而可靠性头自己只在打开时拿到梯度。所以现在数学前提已经满足，接下来仍然是训练本身尚未授权。再往后才是 B1（首次让可靠性影响模型动作）和 B2（若研究目标包含 RGB 完全失效，需要真正的 Depth 语义路径）。

### 1.3 本次 v2 修订带来什么、没带来什么

- **带来：** 修正后的 A1 v2 失效基函数与 burden 语义；修正后的 A2 v2 监督目标 `R_D^sup = V_D^pre · R_D^syn` 与 Depth-only 监督；分开记录的无效像素总体；一个可复跑的 severity→burden 审计工具与报告；一个已写好并已于 2026-09-15 执行通过的 gradient-path isolation 工具与报告；三个结果前冻结的补充协议身份（R1/S1/C1）；v2 协议模板与 v1 的 superseded 标记。
- **没带来：** 没有模型训练、没有 checkpoint、没有 mIoU、没有可靠性校准结果、没有云资源、没有 official test。本文中所有 mIoU、mAcc、mF1、Boundary IoU 的 MMFR 新模型结果仍然全部是未产生。

## 2. 项目背景与数据职责

### 2.1 DFormer、DFormerv2-S 与 RGB-D 语义分割

- **DFormer：** 本项目中的 DFormer 是一个 RGB-D 语义分割模型族，同时使用彩色图像和深度图像，为每个像素预测语义类别。
- **DFormerv2-S：** 当前模型为 DFormerv2-S，即 DFormerv2 的 Small 规模配置。它是 **RGB-primary semantic path + Depth geometry-prior path 的非对称架构**：Depth 主要通过 geometry prior 进入 GSA 相关计算，而 Q/K/V 的主要语义信息仍然来自 RGB。因此它是 RGB-centric 的 RGB-D 模型，不是已经具备独立 Depth 语义推理能力的双语义模型；把 Depth 描述成与 RGB 对称的第二条语义分支是错误表述。
- **RGB-D 语义分割：** 输入是空间对齐的 RGB 与 Depth，标签是语义类别图；输出是每个像素的类别 logits 或类别预测。Depth 不是标签，也不是天然的传感器健康真值。
- **MUSeg：** 当前材料把 MUSeg 定位为矿井域 RGB-D 语义分割数据集，包含精确对齐的 RGB/Depth 与语义标签。本文不把这些数据自动解释成带有故障类型、故障位置或故障严重度的传感器数据集。

### 2.2 数据划分和职责

- **`train-dev=1277`：** 用于模型训练和训练过程中的开发职责。MMFR A2 v2 的两个公平训练身份都应使用这一划分。
- **`val-dev=318`：** 用于 clean checkpoint selector 和冻结后的开发评价。该划分包含 196 个 location groups（位置组），后续配对 bootstrap 以 location group 为重采样单位。
- **official test=`sealed_unread`：** official test 共 `1576` 条，是封存、未读取的最终测试职责。它不能参与训练、checkpoint 选择、阈值选择、方向筛选或当前开发评价。
- **输入职责：** RGB 是彩色观测，Depth 是深度观测；A2 的 backbone 继续消费现有 normalized tensor。可靠性 head 另消费从最终 crop 恢复的 raw `[0,1]` 信号。
- **标签职责：** Label 是语义分割的 ground-truth 标签，用于 segmentation loss 和评价；A2 的 reliability target 来自合成器已知的失效 burden，不是 MUSeg 提供的自然传感器可靠性标签。
- **评价职责：** checkpoint selector 和主 evaluator 都固定在开发划分上；最终评价必须报告输入 geometry、metric geometry、evaluator、checkpoint、split、seed、condition 以及产物哈希。

## 3. 稳定 Quick-B0 基线

### 3.1 身份

- 模型：`DFormerv2-S RGB Quick-B0`。
- RGB 输入契约：`rgb-imagenet-rgb-order-v1`。当前 quick B0 明确把原始 OpenCV BGR 转成 RGB，再使用 RGB 顺序 ImageNet mean/std。
- 官方预训练权重：`DFormerv2_Small_pretrained.pth`，SHA-256 为 `19116988fc86dc9f3e879282237941e11b9b1b5c480edb51e92807311dbc11a6`。
- 最终 checkpoint：epoch 420 的 `selector-epoch-420.pth`，SHA-256 为 `f246a3afc50334c81302b7bfebdadf7cf37d00326bf1c3aa54f6a151754e3a1c`。
- `val-dev` split SHA-256：`1d0719d8f64f016d48995c25ab66d4004d76b7155d9efeef7cbb7454c0dd0e83`。
- official test：`official_test_included=false`，保持 `sealed_unread`。

### 3.2 主 evaluator 与指标

- 主 evaluator 身份：`msflip-whole-original-grid-v1`。
- 每张图采用五个尺度 `0.5、0.75、1.0、1.25、1.5`，每个尺度使用原图和水平翻转，共 10 个 view。
- 每个 view 的输出恢复到 MUSeg 原始 Label 网格；在 FP32 中平均 pre-softmax logits 后计分。
- epoch 420 主指标：mIoU `58.79`、mAcc `69.91`、mF1 `72.73`，单位为百分比。
- 选择过程说明：训练期 selector 的单尺度排序与主 evaluator 的排序不同；epoch 480 的 selector 分数更高，但按预登记的主 evaluator，epoch 420 最终胜出。这说明不能用单尺度 selector 分数代替最终 evaluator。
- B0 的性质：single-seed、RGB、development Quick-B0。它足以作为同口径模块探索的内部共同起点，但不能估计随机方差，不能声称三 seed 完整复现，不能声称 official-test 性能。
- 后续模块的公平要求：必须从相同官方 pretrained 独立训练，保持相同 split、seed、数据顺序、训练预算、优化器、增强、selector 和主 evaluator。不能从 epoch 420 checkpoint 续训后再称为公平消融。

## 4. 关键术语表与判断边界

- **MMFR（Multi-Modal Failure Robustness，多模态失效鲁棒性）：** 本研究方向，研究 RGB/Depth 完全缺失、局部缺失、噪声、模糊、量化、几何错位和混合失效下的模型行为。
- **corruption（合成失效/输入损坏）：** 在原始对齐输入上按冻结规则制造的受控退化。它是 **model-input / representation-level synthetic corruption**（模型输入/表示层合成损坏），不是自然故障标签，不是对真实故障发生率的估计，也不是 Kinect 真实物理噪声模型。`noise` 与 `quantization` 属于 intensity-space corruption，其 severity 仍定义在输入表示空间。
- **reliability target（可靠性目标）：** 合成器根据每个像素的失效 burden 计算出的 `[0,1]` 连续目标。A2 v2 实际用于监督的 Depth surrogate 是 `R_D^sup(p) = V_D^pre(p) · R_D^syn(p)`，其中 `R_D^syn` 是 A1 的合成目标。它描述“按当前合成规则，这个像素被认为还剩多少可信度”，不是自然世界传感器的真值。
- **`depth_valid_pre` / `depth_valid_post`：** corruption 前后实际输入的有效性。`depth_valid_pre = raw_depth_pre > 0 AND valid_mask` 表示真实输入有效性（也是 target 的乘子）；`depth_valid_post = raw_depth_post > 0 AND valid_mask` 表示模型实际看到的 corrupted Depth 的有效性（供 reliability head 的特征使用）。
- **reliability head（可靠性头）：** 从固定信号特征预测 RGB/Depth 可靠性 logits 的小型卷积网络。A2 v2 中它只承担 Depth 通道的辅助监督，预测不会反馈给 DFormerv2 的 backbone、decoder 或 geometry prior；其 RGB 输出通道是未计分的 all-ones scaffold，不能称为已训练的 RGB reliability estimator。
- **clean control（干净对照）：** 使用与 corruption 模型完全相同的官方 pretrained、seed、split、优化器、schedule 和 selector，但训练输入始终不做 MMFR corruption，并且不增加 reliability head。
- **geometry prior（几何先验）：** DFormerv2 根据 RGB/Depth 的空间和深度关系形成的几何贡献信息。它不是最终语义 logits。
- **GSA（Geometry-aware Self-Attention，几何感知自注意力）：** DFormerv2 中消费合成 geometry prior 的注意力机制。DVG-B1 曾尝试只关闭其中的 Depth contribution；该动作失败不能等同于所有 reliability 方法失败。
- **adapter（适配器）：** 位于可靠性估计和几何贡献之间的小型可学习模块。A2 v2 不启用它。B1 的最小 baseline 记为 `B1a-global-reliability-baseline`（`mean/std reliability → global scale`），不得作为 B1 最终默认方案。
- **gradient-path isolation qualification（梯度路径隔离资格检查）：已执行并通过。** 它用两次 backward 证明 reliability 辅助损失不会改变共享分割参数的梯度；身份为 `MMFR-A2-v2-gradient-path-isolation`，实测结果为 `gradient-path-isolation: PASS`（退出码 `0`、墙钟 `15.3` 秒）。
- **preflight（预检）：** 在正式长程训练或完整评价前，对真实模型链做小规模、受限的运行检查。preflight 通过只说明接口、数值和资源路径在该小规模条件下可运行，不说明模型效果。
- **checkpoint selector（checkpoint 选择器）：** 训练期用低成本、冻结规则选择候选 checkpoint 的过程。当前 A2 selector 只看 clean `val-dev`、`original-full`、scale `1.0`、无 flip 的 mIoU；失效条件不参与 selector。
- **paired bootstrap（配对 bootstrap）：** 保持同一图像各条件的配对关系，以冻结的 location group 为相关性单位进行有放回重采样，形成效应量区间。
- **mIoU / mAcc / mF1：** 分别为各类别 IoU、准确率、F1 的宏平均。主 robustness score 使用六个单失效条件的 mIoU 宏平均。
- **Boundary IoU（边界交并比）：** 在定义的边界带几何范围内，按 one-vs-rest、ignore、空类和宏平均规则计算边界评价。历史 `boundary_band_mIoU` 不能直接冒充标准 Boundary IoU。
- **AMP / FP16 / GradScaler：** 自动混合精度、16 位浮点与动态 loss 缩放机制。A2 的 reliability auxiliary 分支固定用 FP32，segmentation backbone/decoder 仍可用 AMP；preflight 中的“尝试步数”不等于“成功更新步数”。
- **single-seed development-supported：** 即便主门槛未来达到，也只能表示单个随机种子、开发划分上的支持性证据，不表示多 seed 稳定性、独立测试泛化或真实部署收益。

## 5. 历史依据：从 DVC/DVG 转到 MMFR

### 5.1 Depth 置零的历史问题

- 早期 DVC-A1 方向把 Depth 的局部边界失效编码为置零，试图比较模型在 boundary/non-boundary 位置的变化。
- v1 的定义门禁扫描了全部 318 个 `val-dev` 样本和 196 个 location group；有 58/196 个组无法形成非空 `boundary-q75`，比例为 `29.5918%`，高于预注册的 `5%` 上限。因此 v1 以 `protocol-blocked` 结束，没有生成可用的五条件科学裁决。
- v2 固定保留 0.05 相对深度跳变阈值、218 张图和 138 个组，但完整统计时一个组只有 137 个有效配对组，仍按预注册门禁阻塞。后续诊断发现 Boundary IoU 的标签域把 raw background 与 evaluator ignore 混用了。
- v3 只修正了 Boundary IoU 的标签域契约：raw background 保留为有效边界几何上下文，true ignore 仍单独处理；数据范围、Depth 置零动作、checkpoint、evaluator 和统计单位没有改变。v3 完整运行主 `dose_effect` 为 `+0.0731348717` 个百分点，95% 区间为 `[-0.0620441424,+0.2226503089]`，区间跨过 0，最终裁决为 `not-supported`。
- 历史结论的准确含义是：在该冻结数据、失效定义、模型和统计协议下，没有支持 DVC 假设的充分证据。它不表示 Depth 可靠性研究从原则上不可能，也不表示 v1/v2 的 protocol 阻塞可以被改写成模型效果失败。

### 5.2 DVG-B1 Oracle 负结果

- DVG-B1 的问题更窄：假设已经知道真实 corruption mask，只在 GSA 的 Depth contribution 上施加 Oracle gate；不改 Depth 输入、不改 Q/K/V、不改 decoder、不改 logits 后处理。
- A/B/C 规则在查看结果前冻结：A 是像素 mask 经 `INTER_LINEAR`、flip、padding 和 OpenCV `INTER_AREA` 聚合到四级 token；B 是 Full/H/W 拓扑上的 query/key 对称乘积；C 要求 `boundary-q75` 下 Boundary IoU 至少增加 `+0.10` 个百分点、95% percentile interval 下界严格大于 0，且 mIoU 点估计不得为负。
- P1 CPU qualification、P2 no-op、P3 受损 gate preflight 和 P4 218 张图完整本地 GPU 配对评价均完成。P4 使用 218 张图、138 个 location group、五个 condition、每图 10 view，共 `10,900` 个 baseline/gated 配对 forward pair。
- clean 条件的全部 `2,180/2,180` view 和 `218/218` 融合结果严格相等；这证明 no-op 和变量隔离，不证明受损条件收益。
- 主 `boundary-q75` 条件的 Oracle-minus-baseline Boundary IoU 为 `-0.1508352015` 个百分点，95% 区间为 `[-0.2676580460,-0.0462325573]`；mIoU 为 `-0.2316731726` 个百分点，95% 区间为 `[-0.3914881430,-0.0919002976]`。因此按冻结 C 裁决为 `oracle-not-supported`。
- 该负结果不能被改写为“可靠性学习失败”。它说明：corruption mask 不等于模型应使用的贡献权重；未训练的固定回退动作可能破坏已学习的 geometry prior；同一可靠性值可能需要随 stage、失效类型和上下文改变动作；DFormerv2 的 RGB-centric 结构不能只靠 GSA Depth gate 覆盖 RGB 完全失效。

### 5.3 对 MMFR 架构的含义

1. 先学习失效分布和可靠性信号，再研究可靠性如何影响融合动作；不能把人工 mask 直接当作最终 gate。
2. A2 先隔离“训练分布、target 和辅助 loss 是否接对”；B1 才研究 learned geometry adapter，减少把多个新变量同时引入的风险。
3. A2 当前不声称最终可靠性融合模型；它没有把 predicted reliability 送入 backbone、decoder 或 geometry prior。
4. 如果研究目标包括 RGB complete-missing，必须另建 Depth 语义路径或双语义路径，不能把当前 RGB-centric DFormerv2 的 A2 结果扩写成 RGB 缺失解决方案。

## 6. MUSeg 自然故障与 Depth 稀疏性证据边界

- MUSeg 图像中可能出现弱光、粉尘、反光、噪声或深度无效等现象，但当前数据和被核对的材料没有为这些现象提供正式的自然故障类型标签、像素位置标签、严重度标签、故障率、重复采集、跨时刻配对或标定漂移真值。
- 因此本文不声称“数据中只有 Depth=0”，也不把未审计的视觉现象写成已确认的自然故障类别。
- **单样本核对事实（不得外推为全数据集统计）：** 在 `train-dev` 的样本 `06-01-01-0035-230920140169-12-99` 上，Depth8 与 Depth16 的原始网格均为 `932×1082`，其中 `993,379 / 1,008,424` 个像素为 `0`，约占 `98.5%`。这意味着该样本的 Depth 观测本身极度稀疏，A1 v2 的“原生无效 Depth”定义会把大部分像素的监督目标设为 0。这是 MUSeg 作者保留“Depth 信息缺失但 RGB 清晰”样本的直接后果，也是 v2 必须区分原生无效与合成缺失的原因；但它只是一个样本的核对结果，不能当作数据集整体统计。**2026-09-15 全量审计更新：** 该推断已被全量扫描取代——全部 `1277` 条 `train-dev` 的原始网格有效率 mean `0.6833`、median `0.7413`、min `0.0125`、max `0.9999`，`<50%` 有 `309` 条、`<5%` 有 `13` 条，**没有任何样本 Depth 全为 0**；训练几何（crop/pad）下 mean `0.6978`、median `0.8177`，另有 `8` 条样本在 `valid_mask` 内完全没有有效 Depth。监督像素中 target 恰为 `0` 的占比为 `28.33%`/`28.41%`/`43.84%`（early/mid/late）。详情与哈希见 §13.6。
- A1/A2 的 `entire_missing`、局部 dropout、noise、blur、quantization 和 misalignment 都是受控合成条件。它们可以支持“在 MUSeg 语义域和冻结合成协议下的开发研究”，不能单独支持真实矿井传感器故障率、自然故障分布、现实部署可靠性或跨设备校准稳定性。
- 未来如果要提出自然故障结论，必须另行获得自然故障标签、采集和标定证据，或明确把结论限制为“合成失效外推假设”，并报告 synthetic-to-real gap（合成到真实的差距）作为未解决风险。

## 7. 当前 MMFR 总体架构与阶段图

### 7.1 阶段关系

```text
A1：通用 RGB/Depth 失效基函数
    ├─ 受控 corruption（v1 basis 已冻结；v2 basis 为当前身份）
    ├─ 连续 reliability target
    ├─ audit metadata / deterministic semantics
    └─ fixed signal features + future adapter scaffold
             │
             ▼
A2 v2：Depth corruption robustness training + parallel reliability estimation
    ├─ post-mirror/scale/crop/pad、pre-GPU corruption
    ├─ segmentation loss + 0.1 × Depth-only reliability auxiliary loss
    ├─ R_D^sup = depth_valid_pre × R_D^syn
    ├─ predicted reliability 不进入 backbone/decoder/geometry prior
    └─ 门禁已通过：gradient-path isolation 已于 2026-09-15 执行并通过
             │
             ▼
B1：第一个真正的 reliability-aware segmentation 阶段（待细化、待授权）
    ├─ 默认方案：DFormerv2-native reliability-conditioned geometry adaptation
    ├─ 最小 baseline：B1a-global-reliability-baseline（mean/std → global scale）
    └─ 必须从官方 pretrained 独立训练并建立新 protocol
             │
             ▼
B2：若要覆盖 RGB 完全失效（待设计，与 B1 严格分离）
    ├─ B2a explicit Depth semantic fallback（默认首先研究）
    ├─ B2b feature reconstruction/compensation
    ├─ B2c modality-agnostic representation/pretraining
    └─ 不能由当前 RGB-centric A2/B1 静默替代
```

### 7.2 当前 A2 v2 的准确定位

- 当前正式 config 是 **Depth-only corruption + segmentation loss + Depth-only reliability auxiliary loss**。
- v2 的 reliability head 仍输出 RGB/Depth 两通道，但只有 Depth 通道计分；RGB 通道是 all-ones scaffold，其 target 全为 1 且不参与 loss，不报告任何 calibration 指标。
- predicted reliability 不进入 DFormerv2 backbone、decoder 或 geometry prior；四级 area pyramid 和 geometry adapter 虽已定义，但 A2 的辅助 loss 只使用固定特征、head 和连续 BCE。
- 当前 A2 不是最终可靠性融合模型。即使未来两个训练身份产生结果，也只能先回答训练分布和 auxiliary supervision 的开发问题；B1 需要独立证明 learned adapter 是否带来下游收益。
- B1 的冻结设计原则：reliability 是“condition”，fusion action 必须由 segmentation objective 学出来，禁止令 action 直接等于 reliability 或 corruption mask；adapter 至少同时接收 stage-specific Depth reliability 和当前 stage 的语义/context 信息；局部失效必须保留 spatial reliability structure，不能只留下 mean/std；所有新调制层必须 zero-init/no-op-init，使初始化时旧 DFormerv2 行为严格不变，并使用连续软调制而非 0/1 硬开关。DVG-B1 的负结果因此继续保留，它说明“知道哪里坏”不等于“知道模型该采取什么动作”。

## 8. A1 失效基函数：v1 与 v2

### 8.1 身份与支持的类型

- **v1 身份（冻结、历史）：** `MMFR-A1-corruption-basis-v1`，实现于 `utils/dataloader/multimodal_failure.py`，其 synthetic burden 历史语义不得修改。
- **v2 身份（当前）：** `MMFR-A1-corruption-basis-v2`，实现于 `utils/dataloader/multimodal_failure_v2.py`（SHA-256 `8734cf090e58eca17a54add761bbddf8baa06712481333d35a45e8bf4b8e94b1`）；它不导入、也不改写 v1 模块。
- 支持的六类失效：`entire_missing`（整模态完全缺失）、`spatial_dropout`（块状局部缺失）、`gaussian_noise`（加性高斯噪声，`sigma` 上限 `48` 个 uint8 强度单位）、`blur`（Gaussian blur）、`quantization`（位深/台阶损失）、`misalignment`（Depth 相对 RGB 的整数平移，越界区显式置零并记录为无效）。`quantization` 与 `misalignment` 是 Depth-only。
- 明确不实现 haze、dust 合成、非刚性 warp、Poisson shot noise 或真实相机响应模型；这些名称不能在审计中被当作已实现功能。

### 8.2 severity 与 curriculum（冻结不变）

- 每个 `FailureSpec` 显式记录 `modality`、`kind` 和 `severity`；severity 合法范围为 $(0,1]$。
- curriculum progress 固定在 `[0,1]`：前 1/3 为轻度阶段（severity 上限 `0.30`，每个样本 1 个 spec，不能抽 `entire_missing`）；中间 1/3 为中度阶段（上限 `0.60`，1–2 个 spec）；后 1/3 为重度阶段（上限 `1.0`，最多 `max_specs` 个 spec，允许 `entire_missing`）。
- 当前 A2 `max_specs=2`；`entire_missing` 只在重度阶段出现并固定为 severity `1.0`；severity 抽样下限为 `0.05`。
- 采样器只返回 spec，不修改数据；所有随机性来自调用方提供的 NumPy Generator，不读写进程级全局随机状态。

### 8.3 v2 的 burden 语义（本次修订的核心之一）

每个失效基函数在像素 $p$ 对模态 $m$ 产生非负局部 burden $b_{m,k}(p)$；多种失效按调用顺序作用，可靠性按乘法组合：

$$
R_m^*(p)=\prod_k \exp\left(-b_{m,k}(p)\right)=\exp\left(-\sum_k b_{m,k}(p)\right).
$$

- 结构性缺失（`entire_missing`、`spatial_dropout`）的 burden 使用 `MISSING_BURDEN=1.0e4`，在合法计算范围内 `exp(-MISSING_BURDEN)` 下溢为 0，表示完全不可信；这两类的 burden 取值集合只有 $\{0,\ \text{MISSING\_BURDEN}\}$。
- **severity 只编码一次。** `gaussian_noise`、`blur`、`quantization` 的连续 burden 就是 realized normalized damage：

$$
b_{\text{graded}}(p)=\operatorname{clip}_{[0,1]}\left(\frac{\max_c |\Delta_c(p)|}{255\cdot \mathrm{DAMAGE\_REFERENCE}}\right),\qquad \mathrm{DAMAGE\_REFERENCE}=0.25,
$$

  不再额外乘一次 severity。v1 曾经同时用 severity 设定 corruption 强度并再乘一次 severity，因此 v2 是一次显式的协议修订，不是对历史结果的重新解释。
- `gaussian_noise` 的 corruption 参数为 `sigma = severity × 48`；`blur` 的 sigma 是**相对尺度量**：

$$
\sigma_{\text{blur}} = \text{severity}\cdot \mathrm{BLUR\_SIGMA\_FRACTION}\cdot \min(H,W),\qquad \mathrm{BLUR\_SIGMA\_FRACTION}=1/80,
$$

  该比例等价于 v1 在冻结 `480` 像素 crop 上的绝对上限 `6.0` px，但换到原始 `932×1082` 网格上会自动换算，不再出现“同一 severity 在两个分辨率上含义不同”的问题。
- `quantization` 的 corruption 参数为 `levels = clip(round(256 × (1 − severity)), 2, 256)`。
- `misalignment` 的每轴最大位移是相对尺度量 `severity × MISALIGN_MAX_SHIFT_FRACTION × H`（或 `× W`），`MISALIGN_MAX_SHIFT_FRACTION=1/30`；有效区 burden 由已知位移的几何幅度决定，而不是由平坦图像的像素差决定：

$$
b_{\text{in-bounds}}=\min\left(1,\frac{\sqrt{dx^2+dy^2}}{\mathrm{reference\_displacement}}\right),\qquad \mathrm{reference\_displacement}=\sqrt{(\mathrm{frac}\cdot H)^2+(\mathrm{frac}\cdot W)^2}.
$$

  越界像素使用 missing burden。这样同样的平移不会因为图像恰好平坦而被错误判为可靠。
- 噪声与 misalignment 的核先抽一次单位离差、再按 severity 驱动的参数缩放，因此固定 RNG 流下 severity 增大时 realized damage 与 burden 单调不减、target 单调不增。

### 8.4 severity→burden 定点审计（已执行）

- 工具：`tools/mmfr/severity_burden_audit.py`，SHA-256 `7300a9e48c5f0dca59762f3fa47f99f81e4f6e2f7dae9a5308a2e44d8f96a796`；命令 `python tools/mmfr/severity_burden_audit.py`；退出码 `0`；结论 `OVERALL: PASS`；`violations=0`。
- 报告：`outputs/mmfr-a1-v2-severity-audit/severity-burden-audit.json`，SHA-256 `86e468c040648a790675456db1b828a5a71e560b849c75b4b7e9a3ca926c2f52`；主代理复跑得到完全相同的哈希。
- 通过项：`no_double_severity_encoding`、`monotonic_burden`、`monotonic_target`、`relative_scale_consistent`、`dropout_extent_monotone`、`empty_spec_strict_noop` 全部为真。
- 审计对每类失效输出机器可检查的映射链：`severity → 实际 corruption 参数 → realized damage → burden → target`；graded 三类的 burden 与独立重算的归一化损伤逐位一致，misalignment 有效区 burden 恰为 `displacement / reference_displacement`、越界像素恰为 `MISSING_BURDEN`，结构性两类 burden 只取两个值，空 spec 严格 no-op。
- 敏感性反证：把 `gaussian_noise` 的核临时替换成 v1 式二次编码后审计立即失败，把 `blur` 的核替换成随 severity 递减后单调性检查立即失败；因此上述 PASS 不是空跑。
- **证据边界：** 审计使用两个确定性合成 fixture（`480×640` 与 `932×1082`）而不是真实 MUSeg 样本，因此它证明的是代数关系与基函数的单调性，不是真实数据上的行为。

### 8.5 输出与可审计 metadata

- 输出保持 RGB 的 `uint8 H×W×3` 和 Depth 的 `uint8 H×W` 或三通道同值 `H×W×3` shape/dtype。
- reliability target 为 `float32 [2,H,W]`，通道顺序固定为 RGB、Depth，范围为 `[0,1]`。
- 空 spec 是严格 no-op：数组逐元素相同，target 全 1，metadata 为空。
- 非空 metadata 记录 spec 数量、顺序、模态、kind、severity、severity 编码、corruption 参数、realized damage 统计、每个 spec 的 burden min/max/mean、受影响模态的 reliability min/mean，以及 misalignment 的位移、参考位移与越界像素数等。
- 任何不支持的 modality/kind、非有限 severity、越界 severity、shape/dtype 不一致或非有限 target 都 fail-closed，而不是静默修正。

## 9. A2 v2 数据流、确定性语义与无效总体记账

### 9.1 具体接入位置

A2 v2 的 corruption 在 `DataLoader` 已完成 mirror、scale、crop、pad 之后、batch 搬到 GPU 之前，由主训练进程逐样本执行。于是 reliability target 天然与最终 `480×640` crop 几何一致，不需要在 target 上重新播放 mirror/scale/crop，也不改变 worker 内 `TrainPre`、`RGBXDataset` 的原有几何增强语义。

### 9.2 raw、normalized、pad 和 valid mask

- segmentation backbone 继续消费 normalized RGB/Depth tensor。
- reliability estimator 消费从最终 normalized crop 严格恢复的 raw `[0,1]` tensor。
- RGB inverse normalization 使用当前配置的 mean/std：mean `[0.485,0.456,0.406]`，std `[0.229,0.224,0.225]`。
- Depth inverse normalization 固定使用 `TrainPre(sign=True)` 的三个相同通道 mean `[0.48,0.48,0.48]` 和 std `[0.28,0.28,0.28]`。
- qualification 对全部 uint8 值 `0..255` 做 exhaustive round-trip：`uint8 -> normalize -> float32 -> inverse+round` 必须逐值恢复原 byte；失败即 `qualification-blocked`。
- 训练 crop/pad 的 normalized pad 值是精确 `0`。若 RGB 与 Depth 在某像素都逐通道精确为 0，则该像素是几何无效区；这里的 valid mask 只排除 crop/pad，不因为语义 Label 为 255 就排除真实图像区域。
- inverse normalization 会把 normalized zero 还原到通道均值，Depth 约为 uint8 `122`。这不是观测值，所以 corruption 前先把 pad 的 raw bytes 置零，避免 misalignment 或 blur 把伪造的均值移入有效区域。
- corruption 后 pad 的 normalized 输入恢复为原始精确 `0`，raw pad 也恢复为 `0`，target 在 pad 区设为中性 `1`，同时由 `valid_mask` 排除；clean sample 直接复用原 normalized tensor，不做 round-trip 重建。
- 输出接口包括 `rgb`、`depth`、`raw_rgb`、`raw_depth`、`reliability_target`、`valid_mask`、`depth_valid_pre`、`depth_valid_post` 和 CPU-only metadata。v2 故意不再导出单一 `depth_valid`：`V1_ONLY_KEYS` 记录了该键，调用方若沿用旧字段会显式失败，而不是静默读到错误的语义。

### 9.3 监督目标与无效总体记账（本次修订的核心之二）

- A1 返回的 synthetic target 记为 $R_D^{syn}(p)$；A2 v2 实际用于监督的 Depth surrogate 为

$$
R_D^{sup}(p)=V_D^{pre}(p)\cdot R_D^{syn}(p),
$$

  其中 $V_D^{pre}$ 是 corruption 前的真实输入有效性 `raw_depth_pre > 0 AND valid_mask`。
- 因而在 clean 样本中：原生有效 Depth 像素 target=1，原生 Depth=0 的无效像素 target=0；crop/pad 区仍保持中性值 1，但继续由 `valid_mask` 排除。corruption 后的实际输入有效性单独记为 `depth_valid_post`，供 reliability head 的特征使用。
- metadata 分开记录四类互不混同的总体：`natural_invalid_pixels`（原生无效）、`synthetic_missing_pixels`（原生有效但被 corruption 置零）、`newly_valid_pixels`（原生无效但被 corruption 变成非零）和 `post_corruption_invalid_pixels`（corruption 后无效），另有 `implicit_quality_pixels`（保持非零但被 corruption 改变，是 `depth_valid` 看不见的那一类）。
- 三者混为一类是禁止的：前两类是无效像素的两个来源，但它们**不构成** `post_corruption_invalid_pixels` 的划分，因为 `gaussian_noise` 与 `misalignment` 会把原生无效像素变成非零。代码因此断言闭合平衡

$$
\mathrm{post\_corruption\_invalid} + \mathrm{newly\_valid}
= \mathrm{natural\_invalid} + \mathrm{synthetic\_missing},
$$

  并且保持 fail-closed：真正的记账错误仍然会中止运行。
- 这个平衡式是本次定点检查发现并修正的真实缺陷：原实现断言 `post == natural + synthetic`，在首个真实 corrupt 样本上就会中止。该缺陷只存在于 v2 新引入的记账中，v1 不受影响。
- **开放项（不在本次静默修改）：** 是否允许 corruption 在原生无效像素上“伪造”出非零 Depth（即 `newly_valid_pixels` 是否应被重新置零），是一个需要单独决定的研究口径问题。当前实现保留 corruption 的原始语义，只把该总体显式记录出来；它已经在 `MUSeg-open-decisions.md` 中登记为开放决策。

### 9.4 A2 v2 Depth-only 采样

- 每个样本先由独立 RNG 抽 clean/corrupt：`p_clean=0.25`，`p_corrupt=0.75`。
- corrupt 样本在抽 spec 数量和种类前，先限制候选为 Depth；这样不会先抽 RGB+Depth 再过滤 RGB，避免改变 spec 数量分布。
- Depth-only 当前允许六类 A1 kind：`entire_missing`、`spatial_dropout`、`gaussian_noise`、`blur`、`quantization`、`misalignment`。
- clean sample 的 RGB/Depth normalized segmentation 输入保持 exact no-op；Depth-corruption sample 的 RGB 保持原输入，Depth 按 result 重新 normalized，且仍保持原输入 channel count。

### 9.5 per-sample PCG64 SeedSequence

- A2 corruption seed 与模型 seed 分离：corruption seed 为 `2026091402`，模型 seed 为 `772961337`。每个样本建立独立 `numpy.random.Generator(PCG64(SeedSequence(words)))`，不读取或修改全局 RNG。
- 固定 seed words 顺序为：`train_seed`、`epoch_1_based`、`iteration_0_based`、`global_rank`、`sample_slot_0_based`、`sample_id_sha256_u32_be_0..3`（sample id 先统一路径分隔符为 `/`，再取 SHA-256 前 16 bytes 按大端拆成 4 个 uint32）。
- 训练进度使用训练位置，不使用 wall clock：

$$
progress=\operatorname{clip}_{[0,1]}\left(\frac{(epoch-1)N_{iter}+iteration}{N_{epoch}N_{iter}-1}\right).
$$

- 相同 sample id、epoch、iteration、rank、slot 和 protocol 必须得到相同 spec、corruption、target 和 metadata。这个保证只针对 A2 corruption；DataLoader 内的 mirror/scale/crop 仍属于旧训练 RNG 语义。若未来要求任意中途 resume 后几何增强也逐像素相同，必须建立新 protocol。

## 10. 模型、可靠性特征与损失（v2 只监督 Depth）

### 10.1 14 通道固定信号特征

`SignalFeatureExtractor` 没有可学习参数，只注册固定卷积核和常数。输入为 raw `[0,1]` RGB/Depth 及 Depth validity，输出 `[B,14,H,W]`：RGB 的 intensity、local mean、local standard deviation、Sobel magnitude、Laplacian magnitude、Gaussian high-frequency residual 共 6 个；Depth 同样的 6 个；Depth validity 1 个；RGB/Depth cross-modal gradient alignment 1 个。局部统计使用 3×3 window，高频残差使用 5×5 Gaussian kernel，cross-modal alignment 是经有效性约束的 cosine-based 对齐量并映射到 `[0,1]`。这些信号只是 reliability head 的输入，不能被审计成 hard gate，也不能被解释为传感器真值。

### 10.2 reliability head、pyramid 和未来 adapter

- 当前 head 结构为三层卷积：`14 -> 16` 的 3×3 Conv、ReLU，`16 -> 16` 的 3×3 Conv、ReLU，`16 -> 2` 的 1×1 Conv，输出 RGB/Depth 两个 reliability logits。
- A1 同时定义四级 `ReliabilityPyramid`，尺度为 `4、8、16、32`，每级使用 PyTorch `area` 插值。它保持 `[0,1]` 语义，但不声称等同于 DVG-B1 使用的 OpenCV `INTER_AREA`。
- 未来 `GeometryContributionAdapter` 对每级 reliability 做均值和标准差统计，输入为 `[B,4L]`，默认 `L=4`、hidden size `8`，输出每级 spatial/depth 两个 scale，范围严格在开区间 `(0,2)`，输出层零初始化时每个 scale 严格等于 `1`，因此未训练时对旧 geometry prior 是 exact no-op。它现在的身份只能是最小 baseline `B1a-global-reliability-baseline`，不是 B1 默认方案。
- 未来 B1 的抽象组合可以写成

$$
G_l=a_l(\hat R_d)G_{s,l}+b_l(\hat R_d)G_{d,l},\qquad 0<a_l,b_l<2,
$$

  注意 A2 v2 中只有 Depth 通道是被训练的可靠性估计，因此 B1 只能消费 Depth reliability；这只是后续 adapter 设计接口，不是当前 A2 已启用的模型路径。

### 10.3 FP32 reliability 分支

首次真实 AMP forward（v1 preflight）在信号特征中发现，梯度平方和的乘积在 FP16 下溢为 0，导致 cosine 分母为 0 并产生非有限值。修复后：segmentation backbone 和 Ham decoder 继续使用外层 AMP/FP16；reliability 固定特征、head 和连续 BCE 在 FP32 autocast-disabled 区域计算；该修复不改变 reliability 不进入 backbone/geometry prior 的协议；FP32 分支的显存、吞吐和完整 batch size 10 成本仍需要正式 probe/训练测量。

### 10.4 总损失与 Depth-only 监督

A2 v2 每个样本只执行一次 segmentation forward。segmentation loss 为带有效语义标签 mask 的 `safe_masked_mean(cross_entropy(input,label))`；reliability loss 为连续 target 的稳定 BCE，`valid_mask` 只排除 crop/pad，并只对 Depth 通道计分：

$$
\mathcal L=\mathcal L_{seg}^{input}+\lambda_{rel}\mathcal L_{rel},\qquad \lambda_{rel}=0.1,
$$

其中 $\mathcal L_{rel}$ 的 `supervised_channels` 固定为 `["depth"]`；`models/builder.py` 用通道权重掩码实现，并在“掩码后没有任何被监督像素”时 fail-closed 报错，而不是继续跑一个静默的零损失。RGB 通道不计入 loss、不报告 calibration 指标、不得称为已训练的 RGB reliability estimator。当前冻结的其他项为：`lambda_consistency=0.0`；不做 clean teacher；不做 clean-corrupt consistency；不做蒸馏；不启用 geometry adapter；reliability prediction 不进入 backbone、decoder 或 geometry prior。

### 10.5 当前配置的参数计数

按当前配置的 CPU 构建计数，clean control 为 `26,673,193` 个参数，Depth-corruption 配置为 `26,677,579` 个参数，reliability 分支新增 `4,386` 个参数，增幅约 `0.0164435%`：

$$
14\times16\times3\times3+16+16\times16\times3\times3+16+16\times2\times1\times1+2=4,386.
$$

这些是当前配置的 CPU 构建计数事实，不是训练后效果或资源成本结论。

## 11. 两个公平训练身份（v2）

### 11.1 共同冻结项

`MMFR-A2-clean-control-v2` 与 `MMFR-A2-depth-corruption-train-v2` 必须共同使用：

- 同一官方 `DFormerv2_Small_pretrained.pth`；
- 同一模型 seed `772961337`；
- 同一 `train-dev=1277`、`val-dev=318` 和 split identity；
- 同一 RGB input contract `rgb-imagenet-rgb-order-v1`；
- 同一 AdamW、learning rate `6e-5`、weight decay `0.01`、500 epoch、warmup 10 epoch、poly power `0.9`；
- 同一 batch size `10`、workers `8`、尺度增强 `[0.5,0.75,1.0,1.25,1.5,1.75]`；
- 同一 clean `val-dev` selector、评估间隔、tie-break 和主 evaluator；
- 同一 checkpoint 保存/恢复语义、运行身份、协议和证据 manifest 规则。

### 11.2 唯一科学变量

- clean control 保持 pre-A2 模块集合和 clean 输入，不实例化 reliability head，不执行 corruption。
- Depth-corruption model 执行冻结的 Depth-only corruption，增加 reliability head 和 `0.1 × Depth-only continuous BCE` 辅助项。
- 任何单独改变增强、优化器、seed、epoch、split、selector、checkpoint 起点或 evaluator 的运行都不再是该公平对照的一部分，必须使用新 protocol identity。
- 两个身份都必须从官方 pretrained 独立开始；不能从 Quick-B0 epoch 420 checkpoint 续训后与 clean control 或 Quick-B0 直接宣称公平。
- 失效条件不参与 checkpoint selector。两个身份可以按同一 selector 规则选出不同 epoch，但不能为其中一个追加候选、改变 tie-break 或使用 corruption 条件筛选。

### 11.3 代码身份

- `utils/dataloader/mmfr_training_v2.py`：SHA-256 `c5fb035eee040d8afd05007a4d84a2912afe70d2f2b3cc198ad5eafb441e7d0c`（本次修订中为修正 invalidity 记账而改动）。
- `local_configs/MUSeg/DFormerv2_S_MMFR_A2_Common_v2.py`：`16bf4ec91737c452c13f8d8d0c63dfea3b20c079bc6e4cb572417fa66caf3620`；`..._Clean_v2.py`：`650c7e3a0ee22f5bcd233783a1210d7167e4ad0083977e5fc3c0a17ac18d7d54`；`..._DepthCorrupt_v2.py`：`8a9ca0abdb8e15e48ecbc0656bcc1e16adba5a9470380568bae13522e871ff26`。
- `models/builder.py`：`96ce2f4a61e62ae80ff732e387d3ec83486be5c63f8d528d28996e719901c278`；`utils/train.py`：`f7a24349dea295de42b322c97c263fcfde44e3e606ba41034b15589b5ca79099`（本次修订中增加 v2 记账遥测）。
- 工作区当前为未提交状态；这些哈希对应未提交的工作区字节，不是某个 Git commit 的身份。

## 12. 云端执行边界、预算与生命周期

### 12.1 冻结执行配置

- 正式训练环境：云端单 GPU；默认主选 RTX 4090 24GB（价格快照 `1.88 元/小时`），备用 RTX 5090 32GB（`2.78 元/小时`）。
- 5090/4090 价格比与单位样本成本 break-even 吞吐比：

$$
\frac{2.78}{1.88}=1.4787234043\approx1.4787.
$$

  只有 5090 在相同科学配置下实测吞吐超过约 `1.4787×` 4090，单位时间价格更高的 5090 才可能有更低的单位样本成本。
- 冻结 global batch size `10`、DataLoader workers `8`、AdamW、learning rate `6e-5`、500 epoch、warmup 10 epoch、AMP 开启、SyncBN 开启、首轮 `torch.compile` 关闭、seed `772961337`。换 GPU 不能改变科学超参数。

### 12.2 当前未授权的云操作

- 当前没有云实例，也没有创建实例或产生云费用。
- 当前没有授权 batch size 10 的容量/吞吐 probe。
- 当前没有授权 `MMFR-A2-clean-control-v2` 或 `MMFR-A2-depth-corruption-train-v2` 训练。
- 当前没有授权完整 epoch、checkpoint save/load、DDP、完整 evaluator 或 official test。
- 本机 NVIDIA GeForce RTX 5060 Laptop 只承担推理、想法初步验证和小规模 preflight，不作为正式长训练环境候选。

### 12.3 生命周期审计要求

正式云任务即使未来获批，也必须先完成一次该运行的生命周期授权和最晚停止 schedule：启动前明确实例、最长运行时长和预计费用；使用控制面 schedule 设置并复核最晚停止时间；workload 成功、失败或人工中止后都先取回必要证据并核验 SHA-256，再调用控制面 stop；stop 后复查平台状态确实为 `Stopped`（实例内 `shutdown -h` 不能单独证明平台已停止计费）；验收 pass/fail 只决定研究结论，不决定是否停止计费；失败路径也必须自动停止。A2 v2 尚未运行云端生命周期测试，不能把历史其他方向的生命周期通过自动移植成 A2 v2 已经通过。

## 13. 已完成的资格证据及其边界

### 13.1 v1 CPU qualification（历史，保持）

已完成的受影响 Python 文件静态编译、静态诊断、限定 CPU probe 和 `git diff --check`，覆盖 exhaustive uint8 normalization round-trip、各类 corruption 的 shape/dtype/range/finite/metadata、同位置确定性与 slot 分离、Depth-only curriculum、clean exact normalized no-op、pad/raw/target 中性语义、misalignment 不把伪造 Depth 均值字节 `122` 移入有效区、完整 supervision 字段守卫、finite scalar loss/backward、pyramid 与 adapter 初始 exact-one。该 qualification 没有读取真实 checkpoint、没有执行真实 Ham forward。

### 13.2 v1 本地 GPU 单步 qualification（历史，保持）

- 环境：Python `3.10.20`、PyTorch `2.7.0+cu128`、CUDA `12.8`、cuDNN `90701`、driver `610.88`、NVIDIA GeForce RTX 5060 Laptop GPU；真实 `train-dev`、官方 pretrained、Ham decoder、SyncBN、AMP、Depth corruption 与 reliability auxiliary loss。
- 尝试 `6` 个 batch，前 `5` 次由 GradScaler 跳过 update，第 `6` 个 batch 完成 `1` 次 optimizer update；6 个 loss 均有限；最终更新步 loss 为 `3.3470830917`；峰值 allocated/reserved CUDA memory 为 `2069.91/2260` MiB。
- canonical 输出目录 `outputs/mmfr-a2-gpu-preflight-20260914T0138Z/`，其 `run_config.json` 记录 `attempted_steps=6`、`completed_optimizer_steps=1`、`skipped_optimizer_steps=5`、`checkpoint=null`、`official_test_included=false`、`dirty=true`，并记录 protocol manifest SHA-256 `e9825c4a4818cf2860a529b6c0fa542f5412a7153a1e8b9b8c8d000de6ca7f07` 与 pretrained SHA-256 `19116988fc86dc9f3e879282237941e11b9b1b5c480edb51e92807311dbc11a6`。
- **证据边界：** 这是 v1 的资格证据，只证明当时接线在 batch size 1、小步数、单本地 GPU 条件下能完成一次有效参数更新。它不覆盖 v2 代码，不能移植成 v2 的 GPU 通过证据，也不覆盖冻结 batch size 10 的显存、完整 500 epoch、checkpoint save/load、DDP、`torch.compile`、云端资源、正式 evaluator 或 official test。

### 13.3 A1 v2 severity→burden 定点审计（本次新增，已通过）

见 8.4。工具与报告哈希、六项结论与证据边界均已列出；主代理已复跑并得到同一报告哈希。

### 13.4 A2 v2 CPU 定点检查（本次新增，已通过）

- 方式：主代理在 CPU 上以真实 `train-dev` 样本（`RGB/06-01-01-0035-230920140169-12-99.jpg`，缩放到冻结 `480×640` crop，附合成 pad 与注入的原生无效 Depth 区块）调用 v2 batch helper；检查项 `47`、失败 `0`。
- 通过内容：v2 config 模块导入成功且其自身冻结字段守卫接受绑定身份；protocol id、schedule version、corruption basis 与 Depth-only supervised channels 与冻结一致；输出键/形状/dtype 正确；clean 样本对 normalized 模型输入是精确 no-op；clean 的 Depth target 等于原生 Depth 有效性；pad 像素被 valid_mask 排除、normalized 与 raw 均恒为精确零、target 保持中性 `1`；corrupt 样本只改 Depth、RGB 逐位不变、RGB target 通道全 1；target 落在 `[0,1]`、raw 信号落在 `[0,1]`、所有字段有限；`depth_valid_pre` 与 `depth_valid_post` 都是 `valid_mask` 子集；invalidity 记账闭合且与独立重算的掩码一致；同 seed/epoch/iteration/rank/slot 复现完全相同的输出与 metadata；在冻结 `p_clean` 下 clean 与 corrupt 两条分支都可达（40 次迭代中 13 次 clean、27 次 corrupt）。
- 同一检查中的实测数值：`valid_pixels=272384`、`natural_invalid_pixels=267802`、`synthetic_missing_pixels=871`、`newly_valid_pixels=132827`、`post_corruption_invalid_pixels=135846`，闭合平衡成立。
- **证据边界：** 该脚本是一次性临时内联脚本，未留在工作区；本文记录了检查清单与观测值以便复现。它没有运行任何真实模型 forward/backward，因此 segmentation 阶段、AMP 路径、reliability head 接线和 Depth-only 监督掩码**只经过静态复核**，没有执行验证。

### 13.5 gradient-path isolation qualification（本次新增，已执行并通过）

- 工具：`tools/mmfr/gradient_path_isolation.py`，SHA-256 `522a475f8a524bb4fe27379be7882535208de6db19940fb8ec62f0431ef44435`（LF 归一化值；同一文件以 CRLF 存储时的原始字节哈希为 `002dbf0559a1d2989fbe3c3c45f7fb2c39a8e319bd041192d1bc7fa0b1233474`，也正是本次运行报告里记录的 `tool.sha256`）；状态 `executed-and-passed`，执行时间 2026-09-15 01:32（本地时间），命令 `python tools/mmfr/gradient_path_isolation.py`，退出码 `0`，墙钟 `15.3` 秒。
- 结论：`gradient-path-isolation: PASS`，`failures` 与 `warnings` 均为空。
- 要求：在完全相同的模型状态、输入 batch、corruption、RNG 和 segmentation loss 下执行两次 backward（`lambda_rel=0` 与 `lambda_rel=0.1`），逐参数比较所有非 `reliability_estimator.*` 参数（backbone、decoder、geometry prior 及所有头）的梯度并要求在预定义容差内相同；要求 reliability head 参数只在 `lambda_rel=0.1` 时产生有效梯度；要求 head 开/关时 segmentation logits 一致；任何共享参数梯度差异都必须阻塞并报告未记录耦合。
- 实测结果：`714` 个共享参数（backbone `672`、decoder `13`、geometry prior `29`）在两次 backward 下**逐位完全相同**，`over_tolerance_count=0`、`absent_in_both_runs_count=0`，`max_abs_difference=0.0`、`max_relative_difference=0.0`，容差为 `ATOL=1e-6`、`RTOL=1e-6`；reliability head 的 `6` 个参数在 `lambda_rel=0` 时梯度张量存在且恒为零（范数 `0.0`），在 `lambda_rel=0.1` 时存在且非零（范数 `0.05855773380379924`，最大绝对值 `0.04926614463329315`）；两次的 segmentation loss 逐位相同（`2.545051336288452`），reliability head 开/关的 logits 与 `lambda_rel=0` 的 logits 均逐位相同（差异 `0.0`）。
- 输入：真实 `train-dev` 的 `sample_index=0`、`iteration=0`（无需向后扫描），抽到 corrupted 样本（`clean=False`），`1` 个 spec，kind 为 `quantization`（`levels=195`、`step_uint8=1.3144329896907216`，severity 约 `0.24`），Depth 通道监督，分割有效像素 `11932`。
- 证据：`outputs/mmfr-a2-gradient-path-isolation/gradient-path-isolation.json`，SHA-256 `dd0e351b418f0228650830c70cf0750c21a446502d3a9c11ebc047581aedb763`；stdout 为同目录 `run-stdout.txt`。该 JSON 不写 checkpoint、不读取 official test。
- 同一运行核到的官方 pretrained 加载事实（只作事实记录）：checkpoint key `785` 个、backbone key `780` 个、重叠 `774` 个、重叠张量不匹配 `0`；checkpoint 缺少 `extra_norms.0/1/2.{weight,bias}` 共 `6` 个 key，因此这 6 个张量由仓库自身初始化提供；checkpoint 另有 `proj.*`、`norm.*`、`head.*`、`aux_head.*` 等模型未使用的 key。
- **证据边界：** 该检查是 batch size `1`、FP32（关闭 autocast 与 TF32）、单进程（无 process group 时 SyncBN 回退为逐进程 BN）、一个真实样本、一次 corrupted 抽样的定点检查；它不覆盖冻结 batch size `10`、AMP、DDP、多样本、完整 epoch、checkpoint save/load、evaluator、云资源或 official test，也不提供任何 mIoU 或鲁棒性收益结论。
- 通过后冻结的结论：**A2 分割性能变化的直接原因是 corruption exposure，而 reliability head 在该阶段只承担 estimator qualification。** 正是这条数学事实使本项目不需要再额外消耗一次完整 500 epoch 的“corruption-only”训练来证明它。
- 该门禁只在实测通过前阻塞正式训练；现在它已通过，**正式训练本身仍然未授权**。
- **复跑确认：** 主代理随后以 `--output outputs/mmfr-a2-gradient-path-isolation/repro-check.json` 复跑一次，判定与数值完全相同（同一 corruption 抽样 kind、同一 segmentation loss `2.545051336288452`、同一 head 梯度范数 `0.0`/`0.05855773380379924`、两次 sweep 的 logits SHA-256 均为 `82ec467dd97a8e44d7c3e11842ff15d9410d0f7033762d6670c3a43f5ed95288`），复跑报告 SHA-256 为 `54a6c183877c32f720854461c0d62ffe0b3be512c729fff2c6a1b95367cbc6a5`；该判定是确定性的，不是单次抽样的偶然结果。

### 13.6 第二轮本地门禁与事实审计（2026-09-15，全部已执行）

第二轮审计判定为 `需修订后进入`，并要求先补齐本地资格。以下三项门禁与两项事实审计均已实际执行；它们只覆盖本地单卡条件，不构成模型效果结论。

- **`MMFR-A2-v2-initial-state-equivalence`（通过）**：工具 `tools/mmfr/a2_v2_initial_state_equivalence.py`（SHA-256 `7637b67d211e43a872bfa7e9141e9fc6ff04c3647e51b35409401797ca7c1088`），六个独立子进程构建（每个身份 3 次），退出码 `0`、墙钟 `42.7` 秒、`40` 项断言 `failed=0`。两个身份的 `714` 个共有参数与 `88` 个共有 buffer **逐位相同**，`max_abs_difference=0.0`；参数集合差异只有 corruption 侧 `reliability_estimator.head.net.{0,2,4}.{weight,bias}` 共 `4,386` 个元素。`extra_norms.{0,1,2}.{weight,bias}` 六张量在六次构建中哈希唯一（weight 恒 `1.0`、bias 恒 `0.0`，官方 pretrained 提供 `0/6`），因此**该确定性从“待核验”转为“已核验”**。证据 SHA-256 `17dcb4e42f914b095a8d4187adad4c404e89e9eaeb857bd218e3ae47923c61fa`。
- **同一核验的附带事实（重要，未处置）**：`utils/init_func.py:group_weight` 只把 `720` 个参数张量中的 `677` 个放进 AdamW 的 param group，**`43` 个张量不在任何 group 中、从不更新、也从不进入 GradScaler 的 inf 检查**：`29` 个 `backbone.*.Geo.weight`、`8` 个 `backbone.patch_embed.proj.{1,4,7,10}.{weight,bias}`、`6` 个 `backbone.layers.{0,1,2}.downsample.norm.{weight,bias}`。两个身份同样受影响，不破坏公平性，但意味着这些层在任何以此代码库运行的训练中停留在初始化值。是否处置登记为开放项。
- **`MMFR-A2-v2-amp-update-path-isolation`（通过，正式训练前硬门禁）**：工具 `tools/mmfr/a2_v2_amp_update_path_isolation.py`，在本地 GPU 上以**冻结 batch size `10`**、真实 `train-dev` 批次、真实 v2 Depth-corruption 模型与正式 AMP + GradScaler 运行，退出码 `0`、墙钟 `88` 秒。`16` 次尝试获得 `10` 次真实成功更新（`13` 个 corrupt step）；两条轨迹（`lambda_rel=0` 与 `0.1`）的 **step/skip 序列完全相同、scale 轨迹完全相同**（`65536→…→1024`，共 6 次 skip）；每次成功更新后共享参数与共享 optimizer state **逐位相同**（各 `0` 处不一致，`max_abs_difference=0.0`），最终共享参数仍逐位相同；reliability head 自身哈希变化，两次 loss 不同，证明辅助项确实激活而未带动共享轨迹。峰值显存约 `2.50 GB`。证据 SHA-256 `7020c354a948cb876a369dd196d03e21fd17fb1b3e051734b1563931ae358cdb`。**边界：** 单卡单进程、`16` 步、一次配对运行，不覆盖 DDP、完整 epoch、checkpoint save/load、evaluator 或 official test。
- **可复跑 `MMFR-A2-v2-cpu-qualification`（通过）**：`tools/mmfr/a2_v2_cpu_qualification.py`（SHA-256 `b3737f1b7ea6947be9cd9db6ef6b910a4f661e9ddb29da72cc3ddf97475cc999`）把原先“运行后删除的一次性 47 项检查”固化为仓库脚本并扩展到 `74` 项断言；主代理复跑得到 `assertions=74 failed=0 status=PASS exit_code=0`（`2.8` 秒）。证据 SHA-256 `75436cf432a4877d3af63d6b16d8f6413f7007d08c8072cada08a1b2dccd7b57`。
- **`train-dev` Depth 有效性分布审计（事实）**：`tools/mmfr/train_dev_depth_validity_audit.py`（SHA-256 `62a52cf3f4c74352359506d753c69e502cbf3d35314220e1df361ba5b50e4d30`），扫描全部 `1277` 条样本，耗时 `215` 秒。原始网格有效率 mean `0.6833`、median `0.7413`、min `0.0125`、max `0.9999`；`<1%`/`<5%`/`<10%`/`<25%`/`<50%` 分别为 `0`/`13`/`26`/`91`/`309` 条，**没有全 0 样本**；训练几何下 mean `0.6978`、median `0.8177`，另有 `8` 条样本在 `valid_mask` 内无有效 Depth。location group 按既有冻结规则共 `762` 组。监督像素构成（early/mid/late）中 target 恰为 `0` 的占比为 `28.33%`/`28.41%`/`43.84%`，其中约 `25.82` 个百分点始终来自原生无效 Depth。**结论：此前“约 98.5% 像素为 0”的单样本观测不能外推**；BCE 未被恒定 0 完全支配，但有约四分之一到五分之二的监督像素提供恒定 0 信号。证据 SHA-256 `63ea91947b699a99d57358239ccdbb93f1edf1d1aa6a81a67c559f0ad0f5af78`。
- **validity transport 审计（事实，含开放决策）**：`tools/mmfr/validity_transport_audit.py`（SHA-256 `f574d461eb7da9a895251ca1cbc127ae72e7f5c611c401b193cda7e40407f7c4`）。`gaussian_noise`、`blur`、`misalignment` 会在原生无效 Depth 像素上制造非零值；`quantization`、`spatial_dropout`、`entire_missing` 从不。最稀疏真实样本上 `gaussian_noise@0.25` 单个 crop 制造 `129,563` 个此类像素（`1.0` 时 `132,929`），约占该样本有效区内原生无效像素的 `48%`；`R_D^sup = V_D^pre · R_D^syn` 逐位成立、记账恒等式在 `84` 条记录上闭合，**说明记账闭合不等于 target 语义正确**。misalignment 的“源无效→目标非零”结构性恒为 `0`，平移前后有效判定一致率 `77.57%–99.39%`。工具输出 `decision_required`（选项 A 吸收哨兵 / 选项 B 显式命名 spurious-measurement corruption，另有 MID-A/MID-B 子问题），**未做选择**。证据 SHA-256 `4cc92b5f6885e9e8163062062e3d53a843752699e4e8bebd77e5d9dad81639fa`。

### 13.7 尚不存在的资格证据

- 没有 v2 的 GPU preflight；v2 的真实模型 forward/backward、AMP 数值稳定性与显存占用均未测量。
- 没有 v2 的 checkpoint save/load 验收、没有 DDP/`torch.compile` 覆盖、没有完整 epoch、没有 evaluator、没有云资源、没有 official test。

## 14. 正式开发评价设计（未来授权后执行，v2 继承 v1）

### 14.1 checkpoint selector

每个训练身份分别运行相同 clean `val-dev` selector：geometry `original-full`、scale `1.0`、flip `false`、metric mIoU、tie-break `earlier_epoch`；failure condition 不参与 selector。selector 完成后冻结 checkpoint，再进入主 evaluator；不得看到 corruption 结果后为某一身份追加 epoch 候选。

### 14.2 主 evaluator

- evaluator：`msflip-whole-original-grid-v1`；5 scales `0.5、0.75、1.0、1.25、1.5`；每个 scale 原图与水平翻转，共 10 view。
- evaluation corruption 在原始对齐 RGB/Depth 上按 evaluation seed `2026091401` 和 sample/condition identity 一次生成，再由 10-view evaluator 共同做 geometry transform；不能为每个 view 重新抽取不同 failure。
- 每个 view 的 logits 回到原始 Label 网格，在 FP32 中平均 pre-softmax logits 后计分。
- 评价范围固定为全部 `318` 张 `val-dev`，不把其中一部分事后改名为独立 test。
- 配对 bootstrap 单位固定为 `196` 个 location groups，重采样组内全部图像并保持 condition pairing。

### 14.3 固定条件

六个 Depth 单失效条件：`spatial_dropout@0.75`、`gaussian_noise@0.75`、`blur@0.75`、`quantization@0.75`、`misalignment@0.75`、`entire_missing@1.0`。三个固定混合条件：`spatial_dropout@0.5 + gaussian_noise@0.5`、`blur@0.5 + misalignment@0.5`、`quantization@0.5 + misalignment@0.5`。混合 spec 的顺序固定，不能根据结果交换顺序、修改 severity 或追加组合。clean、六个单失效和三个混合条件都应报告；混合条件不进入主单失效 score。

### 14.4 主成功门槛与声明上限

比较对象为 `corruption model minus clean control`。主 robustness score 是六个单失效 mIoU 的等权宏平均。主成功必须同时满足：宏平均 mIoU 点估计至少为 `+1.00` 个百分点；该宏平均的 paired 95% percentile interval 下界严格大于 `0`；clean mIoU 点估计下降不超过 `0.50` 个百分点；六个单失效中至少五个 mIoU 点估计不为负，且任一单条件不得低于 `-1.00` 个百分点。

- mAcc、mF1、Boundary IoU、clean Boundary IoU 和三个混合条件是辅助结果，不能事后替代主门槛，把主失败改写成成功。
- 因 `val-dev` 同时承担 checkpoint 选择和开发评价，达到门槛时最高只能写成 `single-seed development-supported`。
- 这些门槛一句都没有改变；它们不因新论文采用 50% missing 等设置而照抄：KSTrack 的 50% modal-incomplete 属 tracking 场景，不能推出 MUSeg 最优 corruption ratio。若未来发现当前 exposure 不合适，只能新开 protocol，不能看完结果后修改 v2。

## 15. 三个结果前冻结的补充协议

三个身份都已在查看任何正式结果之前建立并冻结为独立模板；它们都不能替代主 robustness score，也不能在主门槛失败后重新定义“成功”。

### 15.1 `MMFR-R1-reliability-supplemental-v1`（reliability 专门评价）

- 模板：`protocols/mmfr-r1-reliability-supplemental-v1.template.json`，SHA-256 `5ea5ff1df795f14c3c92345c80ac6112814e7e3f49b85e479904632b0419960b`；状态 `frozen-preregistered-not-authorized-not-executable-before-a2-results`。
- 三层问题必须分开报告：第一层 Depth surrogate target fidelity，固定报告 MAE 与 Brier score（对象是 `R_D^sup`，仅 Depth 通道）；第二层 reliability 与实际 segmentation risk 的关系，固定报告 reliability-error monotonicity、risk-coverage curve 和 AURC；第三层下游 gain，只能等 B1 之后用 segmentation 指标回答。第一、二层不得替代主 robustness score。
- Reliability 结果必须分成 `explicit-invalid`（`entire_missing`、`spatial_dropout`、`misalignment` 越界区）与 `implicit-quality`（`noise`、`blur`、`quantization`、`misalignment` 有效区）两组分别报告：前一组可以通过 `depth_valid` 非常容易识别，不能用它掩盖后者识别失败。
- target 仍是合成 target，不能把校准图写成现实传感器校准证明。

### 15.2 `MMFR-S1-severity-sweep-v1`（severity-response 补充评价）

- 模板：`protocols/mmfr-s1-severity-sweep-v1.template.json`，SHA-256 `bc3fb11666239885b7985778923dee55f37c7bbd1209c395b5bfc667cbe0eb9d`；状态同上。
- 固定评价 `spatial_dropout / gaussian_noise / blur / quantization / misalignment` 的 severity `0.25、0.50、0.75`，另加 `entire_missing@1.0` 和 clean；报告每类 mIoU-vs-severity 曲线、五类非完全缺失 corruption 的宏平均 robustness-AUC，并检查性能是否总体随 severity 恶化。
- 不得用于 checkpoint selection，也不得用它在主门槛失败后重新定义“成功”。文献依据是 CMCAF 的 controlled degradation levels、Yu 等按 missing-modality ratio 的曲线和 KSTrack 的随机 missing rate；这些依据只支持“应做 severity 响应评价”，不改变本项目的主门槛。

### 15.3 `MMFR-C1-paper-confirmation-v1`（论文/SOTA 确认阶段）

- 模板：`protocols/mmfr-c1-paper-confirmation-v1.template.json`，SHA-256 `f5ef2288fffff79185156ed1245aa0e00ba8b6dca73bcea0d2f79ee9ec29a3cd`；状态 `reserved-preregistered-not-executable-until-a2-b1-locked`。
- “开发验证”和“论文/SOTA 确认”是两个完全不同的实验阶段：现有 `1277 train-dev + 318 val-dev + single seed` 继续只用于 architecture screening，最高声明 `single-seed development-supported`。
- C1 只在 A2/B1 架构和全部 protocol 完全锁死以后才建立与执行：使用 MUSeg 官方 `1595` training samples 与 `1576` test samples；固定 500 epoch 或预登记固定 checkpoint 规则，不得通过 test 选 epoch；clean control、最终方法及关键 baseline 使用 **3 个 paired seeds**；最后一次性在 sealed official test 上运行 clean 和所有预注册 robustness conditions。
- MUSeg 原论文的公开模型比较就是 `1,595/1,576` official split、500 epoch、3 个不同 seed 取平均。若以后要和论文中 CMX `61.83`、DFormer `59.74` 等数字谈 MUSeg SOTA，这一步是必要的；当前 `318` 张 val-dev 结果不能承担这个声明。

### 15.4 补充协议中尚未确认的数值

三个模板中有一批**不是改动指令所固定、而是起草时自行拟定**的数值（R1 的 Spearman 方向阈值 `-0.10`、AURC/Brier 相对改进阈值 `0.05`、Holm-Bonferroni 家族大小 `12` 与 α `0.05`、bootstrap 重采样 `10000`、risk-coverage 固定点 `20`；S1 的连续退化容忍度 `0.25` 个百分点与允许违反次数 `0`、bootstrap 重采样 `10000`；C1 的 bootstrap 重采样 `10000`）。它们已在模板内显式标注为 `provisional-default-not-frozen-requires-user-confirmation` 并列成清单，**在执行前必须由用户确认**，当前不得当作已冻结的科学阈值；指标本身的定义（MAE、Brier、单调性、risk-coverage、AURC、三层结构与两组划分）是改动指令固定的，不属于待确认项。

## 16. 预期变化与禁止的结果叙事

以下是实验前的合理预期，不是实验结论：

- 参数量预计几乎不变；当前 CPU 构建计数显示新增约 `0.0164%`，这不是性能保证。
- clean 指标的目标是基本稳定，但不能保证超过 Quick-B0 的 mIoU `58.79`；clean control 与 corruption model 还没有正式结果。
- 主要期待是 corruption 条件下的分割性能改善，尤其是六个单失效的宏平均；是否发生必须由冻结 evaluator 和 paired bootstrap 决定。
- **归因约束（v2 明确）：** A2 的 reliability 不反馈模型动作，因此任何潜在收益只能来自训练期间暴露于 corruption 分布及其辅助 target，不能归因于 reliability head；禁止使用“辅助 target 促进了 segmentation robustness”一类因果表述。reliability head 在 A2 的职责是 estimator qualification。
- 不能写“clean 已保持”“corruption 已提升”“可靠性已经校准”“模型已经适合真实故障”或“已具备 RGB 完全失效能力”，除非未来有对应冻结身份、完整产物和直接核验的结果。

## 17. 风险清单与停止条件

### 17.1 研究解释风险

- **自然故障外推：** 没有自然故障类型、位置、严重度、故障率和标定漂移标签。若审计发现结论越过合成开发边界，停止并收窄结论。
- **Depth 稀疏性被误读：** 单个 `train-dev` 样本的 Depth8 有约 `98.5%` 的像素为 `0`，意味着 v2 的监督目标在这些像素上恒为 0。若据此声称“可靠性估计器已学会识别矿井失效”，属于把数据结构当成模型能力。
- **RGB-centric 风险：** A2 只做 Depth corruption；即使 Depth 失效结果支持，也不能声称 RGB complete-missing 已解决。
- **synthetic-to-real gap：** target 可能只识别 A1 合成器的统计痕迹。若 reliability head 只拟合 synthetic target 而与 segmentation risk 无关，应将其写成 target fidelity，不写成现实可靠性。
- **单 seed 风险：** 单 seed 不能估计随机方差；小增益、接近训练波动或重要结论应增加成对 seed，并建立新预算和 protocol。
- **val-dev 双重职责：** 同一 `val-dev` 用于 selector 和开发评价，不能称独立测试。

### 17.2 实现和统计风险

- **gradient-path isolation 的结论边界：** 该门禁已实测通过（`714` 个共享参数梯度逐位相同），所以“reliability 头不影响分割梯度路径”现在是实测事实而不是静态判断。边界是：它只在 batch size `1`、FP32、单进程、单样本、一次 corrupted 抽样下成立，不能扩展成“冻结 batch size 10、AMP、DDP 也已覆盖”。**第二批边界：** 第二轮新增的 `amp-update-path-isolation` 已在 batch size `10`、真实 AMP + GradScaler 下通过（`16` 次尝试、`10` 次成功更新、step/skip 与 scale 轨迹完全相同、共享参数与共享 optimizer state 逐位相同），因此“实际 AMP 更新轨迹也隔离”在单卡单进程、`16` 步范围内已是实测事实；但它仍不覆盖 DDP、完整 epoch、checkpoint save/load、evaluator、云资源或 official test。若有人把这两项定点结论越界成完整训练已验收，属于停止条件。
- **43 个参数不在 optimizer 中（新发现）：** `29` 个 `Geo.weight`、`8` 个 `patch_embed.proj.*`、`6` 个 `downsample.norm.*` 从不被更新、也不进入 GradScaler 的 inf 检查。两个身份同样受影响，不破坏公平对照，但任何“这些层已被训练”的表述都是错误的；是否修复属于新 protocol 决策。
- **`newly_valid_pixels` 语义未定（新发现）：** 噪声、模糊与错位会在原生无效 Depth 像素上制造非零输入（最稀疏样本上 `gaussian_noise@0.25` 即制造 `129,563` 个），而监督目标在那里恒为 `0`。记账闭合不等于语义正确；A/B 与 MID-A/MID-B 必须由用户或高级模型决定，低级模型不得自行“修复”，也不得在未决定前把该行为写成已定口径。
- **target/geometry 错位：** corruption 必须在最终 mirror/scale/crop/pad 后执行；任何中途 corruption 或独立 target 几何会使监督错位，立即停止。
- **pad 伪值泄漏：** inverse-normalized pad 的均值字节必须在 corruption 前置零，corruption 后 pad 必须回到 normalized exact zero 与 raw zero；发现 pad 被当作观测，立即停止。
- **原生无效像素被 corruption 改造：** `gaussian_noise` 与 `misalignment` 会把原生无效像素变成非零（记为 `newly_valid_pixels`）。当前实现只记录不改语义；如果未来决定在原生无效区重新置零，必须作为新的口径决定并写进 protocol，不能静默修改。
- **Depth-only 泄漏：** 当前 config 只能抽 Depth；发现 RGB spec 被混入、先抽 RGB 再过滤、或 RGB target 非 1，立即停止。
- **监督通道漂移：** `supervised_channels=["depth"]` 是 v2 的冻结项；若 RGB 通道被重新计入 loss、或被报告为已训练的 RGB reliability estimator，立即停止。
- **supervision 静默缺失：** reliability head 启用时辅助字段必须完整，且掩码后必须有被监督像素；缺字段、掩码为空或错配必须 fail-closed，不能以零 loss 继续。
- **Ham decoder RNG：** Ham decoder eval 可能随机初始化 NMF bases。paired forward 或基线/模型比较必须在每对 forward 前回放相同 CPU/CUDA RNG state，否则不能把 logits 差异归因于 MMFR。
- **指标定义漂移：** Boundary IoU、mIoU、metric geometry、ignore/background 域和 bootstrap 单位不能随结果变化。
- **checkpoint selector 越权：** 失效条件不得参与 selector；不得看到结果后追加 checkpoint、改 tie-break 或使用不同候选数量。

### 17.3 资源和运行风险

- **FP32 auxiliary 开销：** reliability 分支需要 FP32 以避免 FP16 下溢，可能改变显存和吞吐；batch size 10 云端容量/吞吐必须实测。
- **batch size 10 容量：** batch size 1 的证据不能证明 batch size 10 可容纳。若 4090 24GB 容量门禁失败，不能直接改 batch size、学习率或科学 schedule。
- **checkpoint save/load：** 尚无正式 checkpoint 生成，也没有验证完整模型与 optimizer state 的 strict 保存/恢复。
- **DDP/SyncBN/torch.compile：** 单 GPU 证据不覆盖 DDP、多卡同步语义或 compile。
- **云生命周期：** 没有最晚停止 schedule、控制面 stop 和 `Stopped` 复查就不能启动付费训练。
- **official test：** 任何意外读取、解封或将 official test 用于选择的行为都是硬阻塞。
- **哈希指针不可复现（待核验）：** v1 模板被记录的 `b5b5c979352cca451cfbca35abc77232a9ae82c3f22ec0f33fcdb964ae03e519` 与 v1 manifest 被记录的 `e9825c4a4818cf2860a529b6c0fa542f5412a7153a1e8b9b8c8d000de6ca7f07` 无法从当前工作区字节复现（已测试原始字节、LF 归一化和六种 JSON 规范序列化）；当前 v1 模板为 `6cb29725e4972526089eb85e8f9e07cf383c19b2389601e189cb69e2d28cb23a`（本次仅追加状态块后的字节），`outputs/mmfr-a2-gpu-preflight-20260914T0138Z/protocol.json` 为 `03443995dbaf03b0750fe736193ef2a1bc3b6f0711987ab634fdff6af5bbef4a`。该差异未做任何历史改写，已标为待核验的完整性事项。

### 17.4 通用停止条件

遇到以下任一情况，应停止扩大修改或运行，保留已完成部分并回到主代理裁决：

1. 需要修改 DFormerv2 encoder、evaluator、checkpoint loader、A1 v1 数值语义或 official-test 文件；
2. 需要改变已冻结 split、seed、condition、severity、selector、成功门槛、supervised channel 或 bootstrap 单位；
3. 实际运行的身份、pretrained、Git 工作区、protocol hash 与预登记记录不一致；
4. clean no-op、Depth-only、pad neutral、finite loss、reliability 完整字段或 seed words 检查失败；
5. 启动正式 v2 训练前，gradient-path isolation 的结论被推翻、被改写，或被越界解释（例如把 batch size 1 / FP32 的单点结论写成覆盖冻结 batch size 10、AMP 或 DDP）；
6. batch size 10 显存不足且拟通过更改科学超参数规避；
7. checkpoint 不能 strict save/load，或 selector 需要不存在的文件；
8. 云实例不能在成功、失败或中止后按控制面规则停止；
9. 需要把 preflight、计划、资源 probe 或 target fidelity 写成模型效果；
10. 任何人要求在结果后修改规则、追加 condition、改阈值、删样本、删 location group 或追加 seed 以获得支持性结论。

## 18. 外部审计清单

外部 agent 应逐项给出“通过、需补证据、需修订、阻塞”及理由，而不是只给总体印象。

### 18.1 高风险项目

- [ ] 事实一致性：Quick-B0 checkpoint、pretrained、split、指标、DVC/DVG 历史终态、v1 与 v2 的身份与哈希是否与本文一致。
- [ ] 当前状态：是否明确没有正式 MMFR checkpoint、没有正式 MMFR 指标、没有云实例、没有云 probe、没有 v2 GPU preflight、没有 official test。
- [ ] 门禁完整性：是否明确 `gradient-path isolation`（batch size 1 / FP32）与 `amp-update-path-isolation`（batch size `10` / AMP + GradScaler）都已执行并通过、`initial-state-equivalence` 已通过、CPU qualification 已可复跑且 `74` 项断言通过，其证据路径与哈希是否可复核；是否**没有**把任何单点结论越界解释成覆盖冻结 batch size 10、DDP、完整 epoch、evaluator 或 official test；是否明确状态仍停在 `local-qualifications-passed-awaiting-senior-review-of-dataset-validity-audit`，因为“高级模型确认监督语义无需改变”这一前置条件尚未满足。
- [ ] 新发现的事实是否被正确记录：`43` 个参数不在任何 optimizer param group（从不更新）；`train-dev` 全部 `1277` 条样本的 Depth 有效性分布（原始网格 mean `0.6833`、无全 0 样本）以及单样本 `98.5%` 观测不可外推；validity transport 的 A/B 与 MID-A/MID-B 仍是开放选择。
- [ ] 修订合法性：v2 是否确实发生在正式 500 epoch 之前；v1 的历史证据、哈希与负结果是否未被改写。
- [ ] 公平性：两个 v2 训练身份是否都从相同 official pretrained 独立开始，是否共享 seed/split/optimizer/schedule/selector。
- [ ] 结果后选择空间：condition、severity、样本、group、checkpoint 候选、阈值、主/辅助指标、R1/S1 补充协议与 supervised channel 是否都在结果前冻结。
- [ ] 数据结论边界：是否把合成 corruption 误写成自然故障标签、故障率或部署证据；是否把单样本 Depth 稀疏性写成数据集统计。
- [ ] A2 定位与归因：是否把 reliability head 的辅助监督误写成已经参与 fusion，是否把 A2 描述成最终 reliability fusion model，是否出现“辅助 target 促进 segmentation robustness”一类因果表述。
- [ ] RGB 边界：是否把 Depth-only 训练结果扩展成 RGB complete-missing 能力；是否把 RGB scaffold 通道写成已训练的 RGB reliability estimator。
- [ ] 评价支持性：主门槛是否确实由六个单失效 mIoU 宏平均、paired 95% interval、clean drop 和逐条件保护共同决定。
- [ ] 实现泄漏：reliability target 是否仅来自合成器与原生有效性，valid mask 是否只排除 pad，Label 是否没有泄漏进 reliability head 输入，corruption 是否发生在最终 crop 后。
- [ ] 无效监督：head 启用时是否 fail-closed 检查 raw inputs、target、valid mask、pre/post validity；pad target 是否中性且被 mask 排除；invalidity 记账闭合式是否为 `post + newly_valid == natural + synthetic`。
- [ ] 云成本和停止：是否先做容量/吞吐 probe、设置最晚 stop schedule、核验控制面 `Stopped`，并在失败时停止计费。
- [ ] 完整性待核验项：v1 模板与 v1 manifest 的历史哈希不可复现问题是否被如实标为待核验，而不是被解释成文件被篡改或直接忽略。

### 18.2 中风险项目

- [ ] 14 通道特征是否真的保持 `[0,1]`、finite、无意中变成 hard gate。
- [ ] A1 v2 六类 failure 的 severity 单次编码、相对尺度语义、curriculum、组合顺序、burden 与 target 是否与 protocol 一致，且 severity 审计报告哈希可复现。
- [ ] per-sample PCG64 SeedSequence words 是否顺序固定，sample id 是否统一 `/` 后取 hash 前 16 bytes。
- [ ] normalized/raw/pad round-trip 是否对全部 uint8 逐值通过，Depth channel count 是否在 clean/corrupt 两条路径一致。
- [ ] FP32 reliability 分支是否只修复数值稳定性而没有偷偷改变 loss 权重或模型反馈路径。
- [ ] Ham decoder 的随机 NMF bases 是否在 paired comparison 中使用相同 CPU/CUDA RNG state。
- [ ] R1/S1/C1 中起草期自拟的数值是否都被显式标为待确认，是否没有任何一个被当作已冻结阈值引用。
- [ ] single-seed 和 `val-dev` 双重职责是否在最终报告中被明确限制。
- [ ] 新增参数计数 `4,386`、clean `26,673,193`、corruption `26,677,579` 是否由实际当前配置复核。
- [ ] v2 代码哈希是否与本文一致，且是否明确这些哈希对应未提交工作区字节。

### 18.3 低风险项目

- [ ] 4090/5090 价格快照、1.4787 break-even 规则和换卡不改科学超参数是否被记录。
- [ ] 首轮 `torch.compile` 关闭、AMP/SyncBN 配置和本机/云端职责是否写清楚。
- [ ] 证据文件、JSON、日志和 checkpoint 是否都有路径、身份和 SHA-256 指针。
- [ ] 本文是否没有使用 Markdown 表格，所有 Markdown 数学公式是否使用 `$...$` 或独立 `$$...$$`。

## 19. 状态矩阵（用项目符号表达）

### 19.1 已完成

- A1 v1 standalone 模块 `utils/dataloader/multimodal_failure.py` 与 reliability feature/head/pyramid/adapter 模块 `models/modal_reliability.py` 已实现并通过 CPU qualification，保持 standalone 语义。
- A1 v2 模块 `utils/dataloader/multimodal_failure_v2.py` 已实现：severity 单次编码、blur/misalignment 相对尺度、结构性/渐变 burden 语义与完整 metadata。
- A2 v1 的 `utils/dataloader/mmfr_training.py`、三个 v1 config 和 `protocols/mmfr-a2-train-integration-v1.template.json` 已冻结并归档为 `superseded-before-formal-training`。
- A2 v2 的 `utils/dataloader/mmfr_training_v2.py`、三个 v2 config、`models/builder.py` 与 `utils/train.py` 的 v1/v2 双协议守卫已实现；v2 增加 `depth_valid_pre`/`depth_valid_post`、`R_D^sup` 目标、Depth-only 监督与四类无效总体记账。
- `protocols/mmfr-a2-train-integration-v2.template.json` 已创建并冻结 v2 身份；R1/S1/C1 三个补充协议模板已建立并冻结。
- `tools/mmfr/severity_burden_audit.py` 已实现并运行通过；`tools/mmfr/gradient_path_isolation.py` 已写好，并已于 2026-09-15 执行通过（`gradient-path-isolation: PASS`）。
- A2 v2 CPU 定点检查已完成（47 项通过）并修复 invalidity 记账缺陷。

### 19.2 已通过但不构成效果

- A1 v2 severity/burden 审计：退出码 `0`、`violations=0`、六项结论为真，报告哈希可复现。
- A2 v2 CPU 定点检查：clean no-op、target 语义、pad 中性、Depth-only、记账闭合、确定性、p_clean 双分支可达。
- gradient-path isolation 门禁：`gradient-path-isolation: PASS`、退出码 `0`，`714` 个共享参数梯度在 `lambda_rel=0` 与 `lambda_rel=0.1` 下逐位相同，reliability head 只在 `lambda_rel=0.1` 有非零梯度，两次 segmentation loss 与两种情况下的 logits 逐位相同；证据哈希可复核。它的证据边界是 batch size `1`、FP32、单进程、单样本、一次 corrupted 抽样。
- 第二轮新增：`MMFR-A2-v2-initial-state-equivalence` PASS（`714` 个共有参数与 `88` 个共有 buffer 逐位相同、`extra_norms` 六张量可复现）；`MMFR-A2-v2-amp-update-path-isolation` PASS（batch size `10`、`10` 次真实成功更新、step/skip 与 scale 轨迹完全相同、共享参数与共享 optimizer state 逐位相同）；可复跑 CPU qualification `74` 项断言 PASS；`train-dev` Depth 有效性分布已实测（原始网格 mean `0.6833`，无全 0 样本）；validity transport 已实测（噪声/模糊/错位会在原生无效像素上制造非零值）。
- 这些证据只说明当前代码链在 CPU、合成 fixture 或真实样本的定点条件下符合协议，不能说明 MMFR 提升 mIoU、校准可靠性、改善自然故障或带来部署收益。

### 19.3 未执行

- **高级模型对“`train-dev` Depth 有效性审计与 validity transport 审计不需要改变监督语义”的确认**（第二轮指令第 9 条要求此前置条件）。在这一确认与 source identity 冻结同时成立之前，状态不得升级为 `eligible-for-cloud-capacity-probe-authorization`。
- validity transport 的 A/B 选择与 misalignment 的 MID-A/MID-B 子问题（由用户或高级模型决定，低级模型不得自行选择）。
- 43 个不在任何 optimizer param group 中的参数是否处置（修复会改变训练语义，必须新开 protocol）。
- v2 的真实 AMP 路径与冻结 batch size 10 的 GPU 行为（无 v2 的 GPU 单步 preflight，AMP 数值稳定性与显存占用未测量）。
- 云端 batch size 10 容量/吞吐 probe；云实例创建、连接、执行和停止。
- `MMFR-A2-clean-control-v2` 与 `MMFR-A2-depth-corruption-train-v2` 的 500 epoch 正式训练。
- 正式 checkpoint save/load 验收、DDP、compile 路径和完整 epoch。
- 冻结 checkpoint 后的 318 样本、10-view、六单失效和三混合条件评价。
- R1/S1 的实际执行；C1 的任何部分。
- official test 的解封、读取或评价。

### 19.4 待设计

- `MMFR-B1-learned-geometry-adapter-v1` 的独立代码、loss、checkpoint 和成功门槛；B1 默认方案为 DFormerv2-native reliability-conditioned geometry adaptation，`B1a-global-reliability-baseline` 只是最小 baseline。
- `MMFR-B2` 的三个预研选项（B2a/B2b/B2c）及其协议；CMNeXt 不能当作 RGB-complete-missing 的直接模板。
- R1/S1 中起草期自拟数值的用户确认。
- 是否允许 corruption 在原生无效像素上产生非零 Depth（`newly_valid_pixels` 的处置）。
- 自然故障数据、故障率、重复采集和标定漂移证据。

### 19.5 禁止

- 从 Quick-B0 epoch 420 checkpoint 续训后冒充公平 A2 消融。
- 修改 DVC/DVG 历史 protocol、负结果或已关闭的 `oracle-not-supported` 结论。
- 把 A1 脚手架、CPU qualification、severity 审计或 v1 preflight 写成 v2 的模型效果或 GPU 通过证据。
- 把已通过的 gradient-path isolation 单点结论（batch size `1`、FP32、单进程、单样本）越界写成覆盖冻结 batch size 10、AMP、DDP 或 official test 的通过证据。
- 在 gradient-path isolation 结论被推翻或越界解释的情况下启动正式 v2 训练。
- 在结果后修改 failure 定义、severity、condition、sample/group、selector、bootstrap、阈值、主 score 或 supervised channel。
- 把 `val-dev` 称为独立测试，或读取 official test。
- 因 4090 容量不足而静默修改 batch size、learning rate、epoch、seed 或其他科学超参数。
- 只训练 clean control 或只训练 corruption model 后宣称公平对照已经完成。
- 把 reliability target fidelity 直接写成自然传感器可靠性或实际分割风险证明。

## 20. 文件地图与证据指针

以下是本文形成时限定读取的关键文件及其职责。外部 agent 看不到这些文件时，应把它们当作待补证据清单，而不是默认已核验。

- `doc/main/MUSeg-current-status.md`：MUSeg 当前事实、阶段、checkpoint/指标身份、已完成历史、授权边界和准确恢复点的唯一实时入口。
- `doc/main/MUSeg-open-decisions.md`：研究口径和处置状态；本次新增 v2 开放项。
- `doc/plans/2026-09-MUSeg-多形式模态失效可靠性学习/00-总方向规划.md`：MMFR 总方向、自然故障证据边界、A1/A2/B1/B2 阶段关系与方向级禁止事项（已同步 v2）。
- 同目录 `01-新对话最小上下文与当前任务.md`：当前任务边界、恢复点和未授权的下一项任务卡（已同步 v2 与门禁）。
- 同目录 `02-MMFR-A1失效基函数与可靠性脚手架.md`：A1 v1/v2 类型、target、burden 语义、信号特征、adapter 接口与审计结果。
- 同目录 `03-MMFR-A2训练接入与公平对照协议.md`：A2 v1 历史记录与 v2 修订记录、数据流、RNG、loss、selector、评价条件、主门槛、云端职责、CPU 检查与已通过的 gradient-path isolation 门禁证据。
- `protocols/mmfr-a2-train-integration-v1.template.json`：已 superseded 的 v1 结构化身份（SHA-256 `6cb29725e4972526089eb85e8f9e07cf383c19b2389601e189cb69e2d28cb23a`）。
- `protocols/mmfr-a2-train-integration-v2.template.json`：当前 v2 结构化身份、代码哈希、审计结果、已通过的 gradient-path isolation 门禁证据与复跑记录（现为 CRLF 存储，原始字节 SHA-256 `8ce4e7c40e04d080ab93cf8b299500f3ca621a0b9ce9757c788fe1435f1a990a`；LF 归一化值 `39b43fb7bd905b4ca49a9e9707227c833dea23c7ea7fb3259f3711ae2571b58d`）。
- **行尾约定说明（供外部 agent 核对哈希）：** 本仓库在 Windows 检出下部分文本文件以 CRLF 存储，而规划文档中的若干哈希是按 LF 归一化内容计算的，两者指同一份内容、只差行尾。已知成对关系：v2 模板 `8ce4e7c4…`（CRLF）/`39b43fb7…`（LF）；v1 模板 `a3002b731019517cbd62b83cfd14e71260099efdd8290504a3ad8bbf6d46ef51`（当前 LF 存储）/`6cb29725e4972526089eb85e8f9e07cf383c19b2389601e189cb69e2d28cb23a`（CRLF）；`tools/mmfr/gradient_path_isolation.py` `002dbf05…`（CRLF 原始字节，亦为运行报告中的 `tool.sha256`）/`522a475f…`（LF）；`tools/mmfr/severity_burden_audit.py` `185135ef…`（CRLF）/`7300a9e4…`（LF）；R1、S1、C1 模板文档中记录的是 LF 归一化值，其 CRLF 原始字节分别为 `c36e239b…`、`a4382b14…`、`d0832faf…`。核对时请先归一化行尾，不要把行尾差异误判为文件被改写。
- `protocols/mmfr-r1-reliability-supplemental-v1.template.json`、`protocols/mmfr-s1-severity-sweep-v1.template.json`、`protocols/mmfr-c1-paper-confirmation-v1.template.json`：三个结果前冻结的补充协议身份。
- `local_configs/MUSeg/DFormerv2_S_MMFR_A2_Common_v2.py`、`..._Clean_v2.py`、`..._DepthCorrupt_v2.py`：v2 两个公平身份的共同冻结项与各自 run identity。
- `utils/dataloader/multimodal_failure_v2.py`：A1 v2 的失效核、单次 severity 编码、相对尺度与 metadata。
- `utils/dataloader/mmfr_training_v2.py`：A2 v2 最终 CPU batch helper、raw/normalized round-trip、pad/valid 语义、Depth-only 抽样、per-sample seed words、`R_D^sup` 目标与无效总体记账。
- `models/modal_reliability.py`：14 通道 fixed signal features、reliability head、area pyramid、连续 BCE 与未来 geometry contribution adapter。
- `models/builder.py`、`utils/train.py`：可选 reliability estimator 实例化、Depth-only 监督掩码与 fail-closed 检查、A2 config 守卫、CPU batch helper 调用、AMP forward、telemetry 与 checkpoint selector。
- `tools/mmfr/severity_burden_audit.py`：severity→parameter→damage→burden→target 的机器可检查审计。
- `tools/mmfr/gradient_path_isolation.py`：梯度路径隔离检查；已于 2026-09-15 执行并通过（报告 `outputs/mmfr-a2-gradient-path-isolation/gradient-path-isolation.json`，SHA-256 `dd0e351b418f0228650830c70cf0750c21a446502d3a9c11ebc047581aedb763`）。
- `outputs/mmfr-a1-v2-severity-audit/severity-burden-audit.json`：severity 审计报告（SHA-256 `86e468c040648a790675456db1b828a5a71e560b849c75b4b7e9a3ca926c2f52`）。
- `outputs/mmfr-a2-gpu-preflight-20260914T0138Z/`：v1 GPU 单步 preflight 证据（含 `run_config.json`、`training_result.json`、`probe-telemetry.jsonl`、`protocol.json`、`train.log`）。
- `doc/reports/2026-08-31-museg-quick-b0-main-evaluation.md`：Quick-B0 的 baseline 身份、epoch 420 checkpoint、10-view evaluator 与 single-seed 边界。
- `liu-test-exp/方案1/改动细节.md`：本次 v2 修订的逐文件改动细节、13 条改动指令逐条处置、三项已执行资格检查的命令与实测数值、未执行事项、已知不一致（v1 历史哈希不可复现、行尾双写法、R1/S1/C1 自拟数值待确认、`newly_valid_pixels` 开放口径、`extra_norms` 确定性待核验）与哈希总表；它是本文件的外部审计配套材料。
- 仓库外证据：DVC-A1 v1–v3 与 DVG-B1 P0–P4 的 protocol、allowlist、execution、full-evaluation 与 failure 记录仍位于 `cloud/DVC-A1-valdev-boundary-zero-*` 与 `cloud/DVG-B1-oracle-gsa-v1/`。

## 21. 审计结论模板

外部 agent 可按以下三种结论之一收口，不得用模糊的“基本可以”。

### `可进入云端容量/吞吐 probe 授权`

使用条件：当前事实、v2 protocol/template、两个 v2 config、pretrained、split、seed、selector、工作区身份与代码哈希已经一致核验；severity 审计报告与 gradient-path isolation 报告哈希可复现；CPU 检查与门禁检查的边界被正确理解为“链路、记账与梯度隔离符合协议，且仅在 batch size 1 / FP32 / 单样本条件下”而非效果；用户已明确授权云端容量/吞吐短 probe。该 probe 只回答 batch size 10 是否能装入、吞吐和单位样本成本，不能顺便运行 500 epoch 或评价模型。

### `需修订后进入`

适用情形：发现文档、代码、template 或运行 artifact 的普通身份字段、解释或证据指针不一致，但可在不改变科学问题、冻结门槛和 supervised channel 的前提下修正；发现需要补一项有限的 CPU/小规模检查（例如 v2 GPU 单步 preflight）；发现 R1/S1 中起草期自拟数值尚未经用户确认。修订后必须重新检查差异、协议哈希、授权范围和最小定点验证，不得直接进入正式训练。

### `阻塞`

以下任一情形都足以阻塞：无法确认工作区代码、protocol/template、pretrained、split、checkpoint 或运行身份；需要从 Quick-B0 续训；需要把自然故障、真实部署或 RGB complete-missing 写入 A2 能力声明；**把已通过的 gradient-path isolation 单点结论越界解释成覆盖冻结 batch size 10、AMP、DDP 或 official test，或在该结论被推翻后仍要启动正式 v2 训练**；batch size 10 容量、checkpoint save/load 或云 lifecycle 尚未达到必要门禁却要开始付费长训练；需要在结果后改变 condition、severity、selector、样本/group、阈值、主指标或补充指标；发现 target/valid/pad/Depth-only/supervised channel/FP32 分支存在泄漏、错位、缺失字段静默继续或非有限 loss；任何 official test 已被意外读取，或已有证据不能证明其仍为 `sealed_unread`。

## 22. 准确恢复点与不得执行事项

### 22.1 准确恢复点

当前准确恢复点是：`MMFR-A2-train-integration-v2` 已完成代码修订与全部本地资格，状态标识为 `local-qualifications-passed-awaiting-senior-review-of-dataset-validity-audit`。已通过本地门禁：可复跑 CPU qualification（`74` 项断言）、`MMFR-A2-v2-initial-state-equivalence`、`MMFR-A2-v2-gradient-path-isolation`、`MMFR-A2-v2-amp-update-path-isolation`（batch size `10`）。已完成事实审计：`train-dev` Depth 有效性分布、validity transport（含 A/B 开放决策）。`MMFR-A2-train-integration-v1` 已归档为 `superseded-before-formal-training`。source identity 以本地 commit + annotated tag `MMFR-A2-v2-pretrain-freeze` 冻结（未推送远端）。

下一步只有在以下条件同时满足后才成立：高级模型审阅两项审计并确认监督语义无需改变 → 用户单独授权云端 batch size `10` 容量/吞吐短 probe → 用户单独授权 `MMFR-A2-clean-control-v2` 与 `MMFR-A2-depth-corruption-train-v2` 两个公平对照训练。当前没有任何正式训练、checkpoint、MMFR 指标、完整评价、云实例或 official test 结果。

### 22.2 不得执行

- 不得在当前授权下创建云实例、运行容量/吞吐 probe、启动 500 epoch 训练或产生云费用。
- 不得在 gradient-path isolation 结论被推翻或越界解释的情况下启动正式 v2 训练，也不得把该单点结论写成覆盖冻结 batch size 10、AMP、DDP 或 official test。
- 不得运行完整 MMFR evaluator、读取 official test 或生成新的 MMFR checkpoint 评价。
- 不得把本文中的“冻结计划”“主成功门槛”“预期变化”写成已完成实验结论。
- 不得修改 DVC/DVG 历史 protocol、负结果或 `oracle-not-supported` 终态；不得重写 v1 的历史证据、哈希与 preflight 事实。
- 不得扩大 A2 为 RGB corruption、learned geometry adapter、Depth semantic fallback 或最终可靠性融合模型。
- 不得根据未来结果删样本、删组、调 severity、加 condition、改 selector、改阈值、改 supervised channel 或挑选补充指标。
- 不得从 Quick-B0 epoch 420 续训后宣称公平；不得只训练一侧对照后宣称比较完成。
- 不得修改 official test 的封存状态。

## 23. 本次修订对 13 条改动指令的逐条处置

1. **冻结并归档 A2 v1、创建 v2 身份：** 已完成。v1 模板状态改为 `superseded-before-formal-training` 并保留其科学语义、qualification 记录与历史哈希；v2 模板、两个 v2 config 已创建。未启动云实例、500 epoch 训练、完整 evaluator 或 official test。
2. **重写 A1/A2/B1 科学含义、不删除现有模块：** 已完成。A2 的归因边界与 B1 的“首次影响模型动作”定义写入 v2 模板、计划文档与本文；DFormerv2 统一描述为非对称 RGB-primary + Depth geometry-prior 架构；已删除“辅助 target 促进 segmentation robustness”一类表述。
3. **修正 Depth reliability target 对原生无效 Depth 的定义：** 已完成。新增 `depth_valid_pre`/`depth_valid_post`，监督目标为 `R_D^sup = V_D^pre · R_D^syn`，无效总体分开记录，并在 CPU 检查中逐项核对。
4. **只监督 Depth、停止把 RGB 恒 1 当学习任务：** 已完成。`supervised_channels=["depth"]` 写入 v2 模板、config 与 `builder.py` 掩码；RGB 通道为未计分 scaffold，并写入禁止把它报告为已训练估计器的措辞。
5. **gradient-path isolation qualification：** **已完成（已执行并通过）**。2026-09-15 01:32 在用户单独授权下于本地 GPU 实际运行：`gradient-path-isolation: PASS`、退出码 `0`、墙钟 `15.3` 秒，`714` 个共享参数梯度在 `lambda_rel=0` 与 `lambda_rel=0.1` 下逐位相同（`max_abs_difference=0.0`、`over_tolerance_count=0`、`absent_in_both_runs_count=0`），reliability head 的 `6` 个参数只在 `lambda_rel=0.1` 有非零梯度（范数 `0.05855773380379924`），两次 segmentation loss 与两种情况下的 logits 逐位相同；证据为 `outputs/mmfr-a2-gradient-path-isolation/gradient-path-isolation.json`（SHA-256 `dd0e351b418f0228650830c70cf0750c21a446502d3a9c11ebc047581aedb763`）。边界为 batch size `1`、FP32、单进程、单样本、一次 corrupted 抽样。通过后冻结的结论已写入 v2 模板与本文：A2 分割性能变化的直接原因是 corruption exposure，reliability head 只承担 estimator qualification，因此不需要再额外跑一次 500 epoch 的 corruption-only 对照。
6. **统一 train/eval 的 spatial corruption 尺度语义：** 已完成。blur 与 misalignment 改为相对尺度，noise/quantization 保持表示空间定义，并在审计中证明跨分辨率一致；统一称为 model-input/representation-level synthetic corruption。
7. **severity→corruption→burden 定点审计：** 已完成。审计工具与报告已落地，确认不存在二次 severity 编码、burden/target 单调性成立；结论为“保留现实现”，没有为了看起来更合理而盲改语义（唯一的代码改动是修复记账平衡式，见第 12 条说明的缺陷）。
8. **冻结项保持：** 已完成。`p_clean=0.25`、三阶段 curriculum、`max_specs=2`、`lambda_rel=0.1`、500 epoch、optimizer、LR、batch size、clean-only selector、六单失效主 score、clean drop ≤0.50、paired location-group bootstrap 与主门槛全部保持不变并写入 v2 模板；明确禁止照抄新论文设置。
9. **结果前冻结 reliability supplemental protocol：** 已完成。`MMFR-R1-reliability-supplemental-v1` 模板冻结三层结构、两组划分与固定指标；起草期自拟数值已标为待确认。
10. **结果前冻结 severity-response evaluator：** 已完成。`MMFR-S1-severity-sweep-v1` 模板冻结 severity 网格、曲线与宏平均 robustness-AUC，并写明不得用于 checkpoint selection、不得重新定义成功；同样标注待确认数值。
11. **重写 B1 规划：** 已完成。最小 baseline 记为 `B1a-global-reliability-baseline`，B1 冻结设计原则（condition 而非 action、保留空间结构、优先 geometry adaptation、zero-init 软调制）写入 v2 模板、计划文档与本文；DVG-B1 负结果的正确解释保留。
12. **B2 与 B1 严格分离：** 已完成。B2a/B2b/B2c 三个预研选项及默认顺序写入 v2 模板与方向规划，并写明 CMNeXt 不能作为 RGB-complete-missing 的直接模板。
13. **补上开发验证与论文/SOTA 确认两个阶段：** 已完成。`MMFR-C1-paper-confirmation-v1` 模板以 `reserved-preregistered-not-executable-until-a2-b1-locked` 状态冻结官方 `1595/1576` 划分、500 epoch、3 个 paired seeds 与 sealed official test 单次读取策略；development 阶段最高声明仍限制为 `single-seed development-supported`。

**第一轮执行边界声明：** 第一轮修订本身没有启动云实例、没有运行 500 epoch 训练、没有运行完整 evaluator、没有读取 official test；其中的 gradient-path isolation qualification 已在用户单独授权后实际执行并通过（batch size `1`、FP32、单进程、单样本、一次 corrupted 抽样）。v1 的历史哈希、DVC/DVG 负结果与既有 preflight 事实均未被改写；v1 模板与 v1 manifest 的历史哈希不可复现问题已如实标为待核验。

## 23.2 第二轮 9 条指令的处置（2026-09-15）

1. **状态降级与措辞收窄：** 已完成。状态改为 `needs-local-qualification-before-cloud-probe` 语义（现为 `local-qualifications-passed-awaiting-senior-review-of-dataset-validity-audit`）；gradient isolation 的 PASS 未被撤销，但结论已收窄为“FP32 数学梯度层面证明 auxiliary reliability loss 不改变共享参数梯度；实际 AMP/GradScaler optimizer-update trajectory 在本轮之前未证明”——该缺口已由第 2 条补齐。
2. **新增阻塞门禁 `MMFR-A2-v2-amp-update-path-isolation`：** 已实现为可复跑工具并执行通过（batch size `10`、`16` 次尝试、`10` 次成功更新、step/skip 与 scale 轨迹完全相同、共享参数与共享 optimizer state 逐位相同、head 自身允许不同），证据见 §13.6。
3. **新增 `MMFR-A2-v2-initial-state-equivalence`：** 已实现为可复跑工具并执行通过（两身份共有 `714` 参数与 `88` buffer 逐位相同、`extra_norms` 六张量三次重建可复现），并顺带证实参数集合差异仅为 reliability head 的 `4,386` 个元素。
4. **CPU 47 项检查固化为可复跑脚本：** 已完成（`tools/mmfr/a2_v2_cpu_qualification.py`，扩展到 `74` 项断言，主代理复跑 PASS；原 47/0 作为历史保留）。
5. **`train-dev` 全量 Depth 有效性分布审计：** 已完成（`1277` 条，原始网格 mean `0.6833`、median `0.7413`，无全 0 样本；训练几何 mean `0.6978`，`8` 条样本在 `valid_mask` 内无有效 Depth；监督像素构成 early/mid/late 的恒定 0 占比 `28.33%`/`28.41%`/`43.84%`）。**该结果同时更正了上一轮基于单样本的“约 98.5% 像素为 0”印象。** 本项只收集事实，未改动 loss 或监督语义。
6. **validity transport 审计与 A/B 选择：** 已完成审计（按 kind 拆分 `V_pre`/`V_post`/值变化/`R_syn`/`R_sup`，含 misalignment 四类运输关系与 MID-A/MID-B 子问题），**未做任何选择**，`decision_required` 区块留给用户或高级模型。没有“修复” `newly_valid_pixels`。
7. **删除 R1/S1 自拟阈值：** 已完成。删除了 Spearman `-0.10`、AURC/Brier `5%`、Holm-Bonferroni 家族 `12` 与 α `0.05`、S1 `0.25` 个百分点容忍度与允许违反 `0`；三个模板只预注册指标、分组、估计量与置信区间，并声明不设置新的成功/失败 gate；`bootstrap=10000` 与 95% 区间保留为计算精度设置。
8. **source identity 冻结：** 已完成。以本地 commit + annotated tag `MMFR-A2-v2-pretrain-freeze` 固定本轮科学代码、config、protocol 模板、工具与证据指针；未推送远端。此后任何科学代码变化都必须新 identity。
9. **恢复点更新：** 已按本条第 9 项执行：本地资格全部通过，但因缺少“高级模型确认监督语义无需改变”这一前置条件，状态**未**升级为 `eligible-for-cloud-capacity-probe-authorization`，并已写明 v2 的 AMP/update-path qualification 是硬门禁而非可选项（它已执行并通过）。

**执行边界声明（第二轮）：** 本轮新增的门禁与审计只在本地单卡、单进程、限步范围内执行；没有云实例、没有 500 epoch 训练、没有 evaluator、没有读取 official test。所有失效仍是合成条件；`train-dev` 分布与 validity transport 的结论只描述合成协议与观测事实，不构成模型效果或真实部署结论。
