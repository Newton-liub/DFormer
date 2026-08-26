# MUSeg 实验口径与处置状态

> 状态时间：2026-08-26 09:42 UTC
> 本文件保留问题缘由，同时明确区分“仍待决定”“本轮已处置”和“仅保留历史解释”。
> 已完成的 seed 1 不回写 protocol 或原始证据；影响后续运行的变更必须使用新 protocol 身份并重新 qualification。

## 1. Validation 空间尺寸

**大白话问题：** 文档曾把“输入 640×480”写成统一模型输入，但 seed 1 实际只把训练样本随机裁剪到高 480、宽 640；`sliding=false` 时，validation 使用转换数据的原始高 932、宽 1082 整图前向。不同几何会改变 val mIoU，不能把结果直接混为同一口径。

**当前状态：仍待决定，正在补充本地后评估证据。**

- seed 1 的训练与在线 validation 事实保持为：训练裁剪 480×640，validation 原分辨率整图，`sliding=false`。
- 已冻结同一 val-dev、BGR、batch 1 和 official-test 不参与的本地后评估工具。
- 计划比较最佳 checkpoint 的原分辨率整图、固定 resize 480×640、sliding-window 480×640，并比较 epoch-500 的后两种几何。
- 当前尚无有效后评估指标；模型构造路径正在修正，因此不能提前选择几何。

决定前，所有报告把“480×640”写成训练裁剪尺寸，不概括为统一 validation 输入尺寸；后评估结果只用于确定未来协议，不改写 seed 1 的原始曲线或 best 身份。

## 2. MUSeg 颜色通道顺序

**大白话问题：** 文档通常用“RGB”表示彩色模态，但 OpenCV loader 对 MUSeg 实际保留 BGR 通道顺序；如果改成 RGB，预训练兼容性和全部结果身份都会变化。

**当前状态：已处置。当前 baseline 谱系冻结 BGR。**

- seed 1 以及后续同一 baseline 谱系继续使用 BGR，不在 seed 间切换。
- protocol、评估证据和报告应显式记录 `channel_order=BGR`。
- 只有决定重建 baseline 谱系时才重新讨论 RGB；届时必须新建 protocol、重新 qualification，并重跑全部 seeds。
- 文档仍可用“彩色图像模态”描述数据目录，但涉及张量事实时必须写明当前 loader 为 BGR。

## 3. A2 自然无效深度分层是否为 B2 硬门槛

**大白话问题：** 人工 corruption 可以证明模型对深度破坏敏感，但当前自然缺失深度证据可能不足以证明现实世界中存在同样机制。若把两者都设为硬门槛，会让“能否做 B2”和“能否声称现实机制”混在一起。

**当前状态：已处置。人工 corruption 可作为进入 B2 的工程门槛，自然证据限制结论强度。**

- A2 人工 corruption 达标后可以进入 B2 开发，不要求自然无效深度分层先成为硬前提。
- 若自然缺失/无效深度分层证据不足，只能声明“在人工 corruption 条件下观察到敏感性或改进”，不得扩展为真实缺失机制、现实鲁棒性或部署收益。
- 正式 A2/B2 开发筛查只使用 `val-dev`；official test 等最终模型和协议冻结后再通过独立门禁一次性解封。

## 4. Qualification 与长程训练的 step 计数

**大白话问题：** Stage-04 计划为 3×128=384 次 loop 尝试，报告记录 376 次成功 optimizer update；Stage-05 理论网格为 64,000 次，最终记录 63,973 次有效更新。AMP 可能跳过少量更新，但旧遥测把“尝试”和“成功”混写，导致验收误判。

**当前状态：历史差异不再阻塞，未来遥测已修正，旧 run 不追溯改写。**

- Stage-04 的 8 次差异缺少完整 trace，无法事后证明每次具体原因；该缺口保留为历史限制，不推翻 Gate D 的连续/恢复等价证据。
- Stage-05 的 27 次差异按少量 AMP 跳过更新处理，不作为训练失败条件；500 个 epoch、50 个 validation 点、checkpoint 身份和最终结果已由 v2 裁决独立核验。
- 未来非 probe 运行分别记录实际 loop attempts、completed optimizer updates 和 skipped optimizer steps，并写入遥测 schema 版本。
- 学习率与调度语义必须在新运行中由结构化计数验证，不用修改原始 `acceptance.json` 或 `training_result.json` 来补齐旧证据。

## 5. `run_kind=qualification` 的历史字段名

**大白话问题：** seed 1 明明是 development 长程训练，命令却记录 `run_kind=qualification`。这是旧代码把“所有非 probe 运行”都叫 qualification，不代表研究 phase 真的是 qualification。

**当前状态：已处置。未来使用 `standard`，历史身份保持不变。**

- 启动器和训练入口已允许未来 `run_kind=standard`，并继续兼容旧的 `qualification`。
- seed 1 的原始命令、manifest 和结果仍保留 `run_kind=qualification`，不得改写；其真实研究阶段继续由 `experiment_phase=development` 和 protocol role 决定。
- 新的 development 长程运行应使用 `standard`；`qualification` 只为历史兼容或真正 qualification 保留。

## 6. 云端终态与关机

**大白话问题：** 本次 SwanLab 已显示完成，但自动流程没有及时关机，人工等待约 23 分钟后仍需手动处理，验收失败路径还曾明确记录 `automatic_shutdown=false`。如果让验收结果决定是否关机，失败时会持续计费。

**当前状态：策略已确定，指南与模板仍待最终落地核验。**

- 成功、失败和人工中止都先同步必要证据，完成本地或独立位置的哈希复核，再停止实例。
- 验收 pass/fail 只决定研究结论，不决定实例是否继续运行和计费。
- 每次启动前在 CompShare 控制面设置最晚停止兜底；脚本内关机只作为第一道保障。
- 本次原始失败行为与证据保持不变，不为符合新策略而改写。
