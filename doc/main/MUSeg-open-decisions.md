# MUSeg 实验口径与处置状态

> 状态时间：2026-08-27 07:28 UTC
> 本文件保留问题缘由，同时明确区分“仍待决定”“本轮已处置”和“仅保留历史解释”。
> 已完成的 seed 1 不回写 protocol 或原始证据；影响后续运行的变更必须使用新 protocol 身份并重新 qualification。

## 1. Validation 空间尺寸

**大白话问题：** 文档曾把“输入 640×480”写成统一模型输入，但 seed 1 实际只把训练样本随机裁剪到高 480、宽 640；`sliding=false` 时，validation 使用转换数据的原始高 932、宽 1082 整图前向。不同几何会改变 val mIoU，不能把结果直接混为同一口径。

**当前状态：仍待决定。原图计分契约已修复并通过聚焦 CPU 测试，五项 val-dev 后评估尚未运行。**

- seed 1 的训练与在线 validation 事实保持为：训练裁剪 480×640，validation 原分辨率整图，`sliding=false`。
- post-evaluator 已改为所有 geometry 保留原始 Label：resize 只改变模型输入，logits 恢复到原图计分；sliding 保持全图覆盖。报告显式记录 input/metric geometry、插值、stride、padding 和输出尺寸。
- production `ValPre`/original-full、resize 原图计分、sliding 覆盖、strict checkpoint load 和 official-test 拒绝的聚焦 CPU 测试已通过；完整验收仍待完成。
- 五项后评估仍保持待运行：best 的 original/resize/sliding，以及 epoch-500 的 resize/sliding。当前没有新指标，不能提前选择 geometry。
- 选择顺序为：在线 original-full 可复现、原始像素支持一致、无长宽比扭曲、显存可控、确定性和 per-class 稳定；固定 resize 只作诊断，original-full/sliding-480×640 为主要候选。

决定前，“480×640”只表示训练裁剪或明确命名的推理输入，不概括为统一 validation 尺寸；后评估只冻结未来 protocol，不改写 seed 1 原始曲线或 best 身份。

## 2. MUSeg 颜色通道顺序

**大白话问题：** 文档通常用“RGB”表示彩色模态，但 OpenCV loader 对 MUSeg 实际保留 BGR 通道顺序；如果改成 RGB，预训练兼容性和全部结果身份都会变化。

**当前状态：用户已重新打开。legacy BGR 只作为历史 reference；未来颜色谱系待 pretrained provenance、三臂诊断和 paired calibration 决定。**

- seed 1 的历史事实保持为 OpenCV BGR 数组，并按位置应用 `[0.485,0.456,0.406]` / `[0.229,0.224,0.225]`；不回写其 protocol 或结果。
- 当前权重已核验身份为 110,203,103 bytes、SHA-256 `19116988fc86dc9f3e879282237941e11b9b1b5c480edb51e92807311dbc11a6`，但仓库内未发现把该 SHA 绑定到上游发布资产及训练通道语义的元数据，来源语义仍待核验。
- 配置、protocol v3、launcher/run manifest、production loader 和 post-evaluator 已加入显式 `channel_order` 与 normalization identity；v2 历史 protocol 只按 legacy 来源补录运行记录，不伪装成原始 manifest 字段。
- 固定 checkpoint 的 legacy BGR、RGB+RGB mean、BGR+反向 mean 三臂只诊断敏感性，不能直接与 seed 1 的 `52.84` 决定新 baseline。
- 真正选择必须使用独立 `color-geometry-screening-B0` protocol，使候选与 legacy BGR 从相同 pretrained、seed、数据顺序、预算和 evaluator 成对重训；接近噪声容差时两臂一起补第二 seed。
- 若最终离开 legacy BGR，当前 seed 1 只保留 `development-reference-B0` 历史身份；新谱系重新 qualification 和 B0，不能跨谱系混入 mean±std。

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

## 7. Development 三 seed 的执行时机

**大白话问题：** 三 seed 能估计随机方差，但在模块方向尚未冻结时为每个候选都跑三次成本很高。当前需要区分“快速发现可行模块”和“形成可发表的正式消融结论”。

**当前状态：已处置。经用户于 2026-08-27 确认，development seeds 2/3 暂缓；先完成 seed 1 后评估，再进入单 seed 配对筛选，正式三 seed 延后到架构与消融组合冻结之后。**

- 当前 seed 1 只作为经过长程训练和独立裁决的 development 参考 B0，不称为三 seed 正式 baseline，也不直接与论文 official-test 数值作严格复现比较。
- 快速筛选必须在同一 train-dev/val-dev、seed、预训练、训练预算、优化器、增强、checkpoint 规则和 validation 几何下成对重跑 B0 与候选模块；改变总 epoch 时必须新建 screening protocol，并在同一短协议下重跑 B0，不能直接对比现有 500-epoch 的 `52.84`。
- 单 seed 结果只用于淘汰和候选排序；方向性较好但增益接近噪声的候选可增加第二 seed 作为中间确认，但第二 seed 不替代正式三 seed。
- 架构、超参数和最终消融组合冻结后，B0 与每个最终模块必须使用同一预注册三 seed 成对从头训练，报告每 seed 配对差、mean、sample std 和 per-class 指标；不得只给模块跑三 seed 而复用当前单 seed B0。
- official test 在开发与筛选期间继续 `sealed_unread`；只有正式 B0/最终模块的 checkpoint、哈希和协议全部冻结并通过 Gate F 后，才按预登记清单一次性评估。
- B2 不在当前 seed 1 baseline 中，也不能作为普通候选绕过专用门禁。若要纳入本轮正式矩阵，必须在 Gate E 前完成新 Stage-06 的 `A2-pass + 用户批准`、规格/金标准、B2-zero-train、B2-short，再回到新 Stage-07 完成 B2-screening；Gate E 未纳入而在 official test 后启动时，登记为独立后续研究。
