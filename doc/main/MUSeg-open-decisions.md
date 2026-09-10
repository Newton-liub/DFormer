# MUSeg 实验口径与处置状态

> **状态时间：** 2026-09-10 08:52 UTC。
> **文档角色：** 研究选择与边界记录，不承担实时状态或执行授权。
> **实时入口：** [`MUSeg-current-status.md`](MUSeg-current-status.md)。稳定基准与分支规则见 [`research-branch-governance.md`](../guides/project/research-branch-governance.md)。
> **候选计划：** A2/B2 与方向1均已延期、未执行、未授权；索引见 [`doc/plans/deferred/2026-09-MUSeg-unexecuted/README.md`](../plans/deferred/2026-09-MUSeg-unexecuted/README.md)。
> 本文件保留问题缘由，并区分“仍待决定”“本轮已处置”和“仅保留历史解释”。已完成的 seed 1 不回写 protocol 或原始证据；影响后续运行的变更必须使用新 protocol 身份并重新 qualification。

## 1. 新 DFormerv2-MUSeg baseline 方向

**大白话结论：** 新计划使用 DFormerv2-S 和其公开训练/测试方法建立内部 B0，作为后续模块消融的共同起点；目标是结果量级合理、链路可信和比较口径一致，不是三 seed 完整复现论文。

**当前状态：方向、RGB、single-seed B0 角色、训练参数、主 evaluator、历史运行所用的 top 3 + latest 和 protocol v3 均已冻结并执行完成。唯一 Quick-B0 已完成 500 epoch 和 4 个候选的五尺度翻转主评估，最终 B0 为 epoch 420，主 mIoU `58.79`、mAcc `69.91`、mF1 `72.73`；official test 继续保持 `sealed_unread`。后续训练的独立 v2 top 8 + latest 设置已提交，但不改变本次 v1 结论。**

- 训练方向：采用官方公开的随机尺度训练增强，尺度候选为 `0.5、0.75、1.0、1.25、1.5、1.75`，之后裁剪到 `480×640`，并保持 RGB/Depth/Label 同步变换。
- 测试方向：采用官方论文公开的 multi-scale flip 推理，尺度为 `0.5、0.75、1.0、1.25、1.5`，暂不把滑动窗口静默混入主基线。
- 输出方向：每个尺度的预测恢复到 MUSeg 原始 Label 网格后融合和计分；`480×640` 是训练或明确命名的模型输入尺寸，不自动等于最终 metric geometry。
- 这是公开 DFormerv2 方法在 MUSeg 上的适配，用于建立后续模块的内部对照；不声称复现 MUSeg 作者未公开的测试代码，也不以论文数值完全相等或三 seed 统计作为当前 B0 门槛。具体 evaluator、预算、checkpoint 规则和运行位置由当前执行方案冻结。
- 旧 `Stage-01` 至 `Stage-05` 计划和其未完成的 Protocol Gate 已封存；历史 seed 1 的单尺度结果只作为 reference，不与新 baseline 混合统计。

## 2. Validation 空间尺寸

**当前边界：** Quick-B0 的主 evaluator 已固定为 `msflip-whole-original-grid-v1`；本节其余关于 seed 1 的 validation geometry 和后评估只保留为历史诊断，不构成当前 Quick-B0 的待决选择。

**当前状态：历史 seed 1 的五项后评估已完成；新 baseline 的 `msflip-whole-original-grid-v1` 已按冻结契约完成 4 个候选的正式主评估。最终 epoch 420 在 318 个 `val-dev` 样本的原始 Label 网格上取得 mIoU `58.79`、mAcc `69.91`、mF1 `72.73`；FP32、TF32 disabled、RGB 输入和 `official_test_included=false` 身份均已核验。**

- 技术检查样本为 `06-01-01-0352-230920140646-10-99`，原始 `932×1082`，尺度 1.5 后 `1398×1623`，padding 后 `1408×1632`；两个 view 用时 `2.095559`/`1.079008` 秒，峰值 allocated/reserved 为 `4,977,021,952`/`6,511,656,960` bytes，未 OOM。证据见 `cloud/DFormer-stage05-evidence/posteval/quick-b0-scale1.5-max-sample-fp32-technical-check.json`，其中 `metrics_computed=false`。按最大样本保守外推最多 4 个候选约 `2.8` 小时，低于 8 小时硬上限。大白话说，本次只确认本机能承载冻结 evaluator，不产生任何模型好坏结论。

- 历史 seed 1 的训练与在线 validation 事实保持为：训练裁剪 480×640，validation 原分辨率整图，`sliding=false`。
- post-evaluator 已改为所有 geometry 保留原始 Label：resize 只改变模型输入，logits 恢复到原图计分；sliding 保持全图覆盖。报告显式记录 input/metric geometry、插值、stride、padding 和输出尺寸。
- 历史后评估链的 production `ValPre`/original-full、resize 原图计分、sliding 覆盖、strict checkpoint load 和 official-test 拒绝的聚焦 CPU 测试已通过；这些检查覆盖历史后评估链路，不再作为当前 Quick-B0 的未闭合事项。
- 五项后评估已完成：best 的 original-full/resize/sliding mIoU 为 `52.98`/`56.31`/`51.89`，epoch-500 的 resize/sliding 为 `56.73`/`52.08`；五项都在原始 Label grid 计分，均为 318 样本且 official test 未参与。结果只能用于 geometry 诊断，不改写 seed 1 原始曲线或 best 身份。
- 历史五项后评估的几何排序只用于诊断，不作为新 baseline 的冻结依据。新计划优先实现 DFormerv2 论文的 multi-scale flip；单尺度 original-full、固定 resize 和 sliding 保留为命名清晰的对照或资源备选。

当前口径是：`480×640`只表示训练裁剪或明确命名的模型输入，不概括为统一 validation 尺寸；Quick-B0 的当前计分口径由 `msflip-whole-original-grid-v1` 固定。历史后评估只作为 geometry 诊断，不改写 seed 1 原始曲线或 best 身份。

## 3. MUSeg 颜色通道顺序

**大白话问题：** 历史 MUSeg loader 使用 OpenCV BGR，但官方预训练模型看到的是 RGB。项目刚起步时，是先做两种颜色的配对训练，还是先选择与预训练一致的输入？

**当前状态：本轮已处置。用户于 2026-08-30 确认 quick B0 直接使用 RGB，取消 RGB/BGR 双臂；这是输入一致性选择，不是颜色性能胜负结论。**

- seed 1 的历史事实保持为 OpenCV BGR 数组，并按位置应用 `[0.485,0.456,0.406]` / `[0.229,0.224,0.225]`；不回写其 protocol 或结果。
- 当前权重已闭合为官方上游资产：Hugging Face `bbynku/DFormerv2` 中 `DFormerv2/pretrained/DFormerv2_Small_pretrained.pth` 的大小为 110,203,103 bytes，LFS SHA-256 为 `19116988fc86dc9f3e879282237941e11b9b1b5c480edb51e92807311dbc11a6`，与本项目权重完全一致。官方 README 将其列为 ImageNet-1K RGB-D pretrained；官方 `VCIP-RGBD/RGBD-Pretrain` 数据代码默认以 PIL `RGB` 读取彩色图并使用 RGB 顺序 ImageNet mean/std，因此 pretrained 上游通道语义判定为 RGB。
- 新 quick B0 明确执行 OpenCV BGR→RGB，再使用 RGB 顺序 ImageNet mean/std。大白话说，这让下游输入保持官方预训练模型已经学习过的通道含义，是当前变量最少、依据最直接的起点。
- 固定历史 best checkpoint 的三臂 original-full 诊断结果仍保留：legacy BGR、RGB+RGB mean/std、BGR+反向 mean/std 的 mIoU 分别为 `52.98`、`33.85`、`49.53`。它只证明旧 checkpoint 对输入契约强敏感，不能用于判断重新训练后的 RGB/BGR 胜负。
- 本轮不做 `color-geometry-screening-B0`、短程颜色训练或第二 seed，也不把“选择 RGB”表述为“RGB 在 MUSeg 上统计显著优于 BGR”。如果未来研究问题明确变成颜色谱系比较，才需要另立 paired calibration protocol 并从相同 pretrained 成对重训。
- 历史 BGR seed 1 继续保留 `development-reference-B0` 身份；新 RGB quick B0 使用独立 protocol identity，两者不混入同一 mean±std 或当作配对实验。

## 4. A2 自然无效深度分层是否为 B2 硬门槛

**大白话问题：** 人工 corruption 可以证明模型对深度破坏敏感，但当前自然缺失深度证据可能不足以证明现实世界中存在同样机制。若把两者都设为硬门槛，会让“能否做 B2”和“能否声称现实机制”混在一起。

**当前状态：已处置。人工 corruption 可作为进入 B2 的工程门槛，自然证据限制结论强度。**

- A2 人工 corruption 达标后可以进入 B2 开发，不要求自然无效深度分层先成为硬前提。
- 若自然缺失/无效深度分层证据不足，只能声明“在人工 corruption 条件下观察到敏感性或改进”，不得扩展为真实缺失机制、现实鲁棒性或部署收益。
- 正式 A2/B2 开发筛查只使用 `val-dev`；official test 等最终模型和协议冻结后再通过独立门禁一次性解封。

## 5. Qualification 与长程训练的 step 计数

**大白话问题：** Stage-04 计划为 3×128=384 次 loop 尝试，报告记录 376 次成功 optimizer update；Stage-05 理论网格为 64,000 次，最终记录 63,973 次有效更新。AMP 可能跳过少量更新，但旧遥测把“尝试”和“成功”混写，导致验收误判。

**当前状态：历史差异不再阻塞，未来遥测已修正，旧 run 不追溯改写。**

- Stage-04 的 8 次差异缺少完整 trace，无法事后证明每次具体原因；该缺口保留为历史限制，不推翻 Gate D 的连续/恢复等价证据。
- Stage-05 的 27 次差异按少量 AMP 跳过更新处理，不作为训练失败条件；500 个 epoch、50 个 validation 点、checkpoint 身份和最终结果已由 v2 裁决独立核验。
- 未来非 probe 运行分别记录实际 loop attempts、completed optimizer updates 和 skipped optimizer steps，并写入遥测 schema 版本。
- 学习率与调度语义必须在新运行中由结构化计数验证，不用修改原始 `acceptance.json` 或 `training_result.json` 来补齐旧证据。

## 6. `run_kind=qualification` 的历史字段名

**大白话问题：** seed 1 明明是 development 长程训练，命令却记录 `run_kind=qualification`。这是旧代码把“所有非 probe 运行”都叫 qualification，不代表研究 phase 真的是 qualification。

**当前状态：已处置。未来使用 `standard`，历史身份保持不变。**

- 启动器和训练入口已允许未来 `run_kind=standard`，并继续兼容旧的 `qualification`。
- seed 1 的原始命令、manifest 和结果仍保留 `run_kind=qualification`，不得改写；其真实研究阶段继续由 `experiment_phase=development` 和 protocol role 决定。
- 新的 development 长程运行应使用 `standard`；`qualification` 只为历史兼容或真正 qualification 保留。

## 7. 云端终态与关机

**大白话问题：** 本次 SwanLab 已显示完成，但自动流程没有及时关机，人工等待约 23 分钟后仍需手动处理，验收失败路径还曾明确记录 `automatic_shutdown=false`。如果让验收结果决定是否关机，失败时会持续计费。

**当前状态：策略与实现均已验证。无卡 lifecycle-test、正式 B0 训练、证据取回、普通 stop 和 `Stopped` 复查均已完成；本次主评估仅使用本地 RTX 5060 Laptop，没有执行云端或生命周期操作。**

- 生产生命周期由本地控制器处理共同终态：workload 成功、失败或人工中止后，都先取回必要证据并核验哈希，再调用 CompShare 控制面 stop；验收 pass/fail 只决定研究结论，不决定是否停止计费。
- 实例内 `shutdown -h` 不能单独证明平台进入 `Stopped`。自动关机验收必须使用控制面 stop，并等待和复查实例状态为 `Stopped`。
- 正式 RTX 4090 前，用 `run_kind=lifecycle-test`、`simulation=true` 的无卡任务模拟成功 workload、测试报告、证据 manifest 和 SHA-256；测试产物不得进入 B0 指标或被训练裁决器接受。
- 无卡实测通过条件为报告与哈希匹配、自动 stop 成功、实例在 timeout 内进入 `Stopped`，且不需要人工补发普通停止命令；失败则阻塞正式 B0。
- 每次无卡测试和正式训练启动前都使用 `instance schedule set --at` 设置控制面最晚停止兜底，并用 `instance schedule show` 复核；脚本或本地控制器自动 stop 是第一道保障，schedule 是断联兜底。
- 2026-08-30 无卡门禁使用实例 `cpod-1tyvjsiu6ahe`：`GPU=0`，durable job `job-20260830T101210Z-8c5b24ea` 退出码 0，证据 manifest SHA-256 为 `f9f00d7bdee84cfa8c5cab5ab47b3388fb2ad709ec03ff404c1b8207d5d37742`；自动 stop 后实例于 `2026-08-30T10:13:25Z` 达到 `Stopped`，无需人工补发普通停止命令。证据见 `cloud/museg-lifecycle-gates/museg-lifecycle-cpod-1tyvjsiu6ahe-20260830T1012Z/`。
- 每次未来正式训练启动前仍需用户对训练实例、最长时间和预计费用单独授权，并重新设置与复核该次运行的最晚停止 schedule。当前 v1 已完成该次启动授权，不构成后续训练的持续授权。

## 8. Single-seed B0 与后续模块消融

**大白话问题：** 当前需要的是模块设计的可信共同起点，而不是先花三倍成本形成论文级随机方差统计。怎样既节省资源，又避免后续比较失去公平性？

**当前状态：已处置并完成。single-seed RGB B0 已冻结为 epoch 420，主 mIoU `58.79`、mAcc `69.91`、mF1 `72.73`；它是后续模块消融的固定内部基线，不是三 seed 完整论文复现。**

- B0 的验收重点是训练与评估链可信、指标量级合理、没有明显类别或数值异常，并完整绑定 pretrained、split、seed、config、checkpoint 和 evaluator 身份；不要求与论文数字完全相等。
- 后续模块可以复用这一个 B0 结果作为对照，但模块版本必须从同一 pretrained 独立训练，并保持相同 `train-dev`/`val-dev`、seed、数据顺序、epoch、优化器、增强、checkpoint 规则和主 evaluator。不能从 B0 最终 checkpoint 接着训练模块后再称为公平消融。
- 若后续改变训练预算、优化器、增强、数据或 evaluator，现有 B0 不再是严格配对对照；需要限定结论，或在新协议下重训匹配的 B0。
- 单 seed 足够用于模块探索、淘汰和初步消融，但不能估计随机方差。若模块增益很小、接近训练波动或要支撑重要结论，应对 B0 和该模块增加成对重复或额外 seed；当前不预先要求三 seed，也不因此阻塞模块设计。
- 主 evaluator 已在本地 RTX 5060 Laptop 上完成：4 个候选全部绑定 checkpoint/split 哈希、冻结代码与协议、RGB 输入契约、FP32、环境和 `official_test_included=false`；内部计时合计 `94.044` 分钟，低于 8 小时硬上限。
- official test 在 B0 和模块开发期间继续 `sealed_unread`；是否以及何时解封由未来独立门禁决定，当前 single-seed 方向不构成解封授权。

## 9. 几何可信 RGB-D 双路径 MVE 的统计来源处置

**大白话问题：** 原草案把 `RE447` 当作 paired/cluster bootstrap 的依据，但全文实际是深海采矿车辆路径规划论文，只使用 AHP-FCE 专家判断矩阵和一致性检验，不能说明怎样对 MUSeg 的相关样本做置信区间。

**当前状态：已处置。处置 1 已全文核对并判定不适用；用户于 2026-09-08 同意，对通用且相对简单的分析操作不再强制补参考文献，因此采用项目预注册统计设计，不继续扩大文献检索。**

- **已排除的处置 1：** `RE447`（Lu et al., *Ocean Engineering*, 2024，DOI `10.1016/j.oceaneng.2024.119500`）没有 paired bootstrap、cluster bootstrap、95% confidence interval、scene/location 重采样或扩样后重复裁决规则。详细证据见 [`00-待补充论文内容清单.md`](../plans/2026-09-MUSeg-几何可信RGBD双路径MVE/参考资料/00-待补充论文内容清单.md) 的 P0-6。
- **采用的处置：** 同一图像各条件保持配对，以冻结的 location group 为相关性边界，对 group 有放回重采样并保留组内全部样本；重采样次数、seed、效应量、95% percentile interval 和联合裁决在查看结果前写入 `DVC-A1-valdev-boundary-zero-v1` protocol。该做法明确标为本项目预注册统计流程，不归因于 `RE447`，也不声称是唯一统计选择。
- **数据职责边界：** `val-dev` 已参与 checkpoint 选择，因此不再把其中一部分命名为独立评价集 `E`。首轮 DVC 没有拟合自由度，直接在全 `val-dev` 上形成 paired development evidence；不改称 independent test。
- **Boundary IoU 处置：** Cheng et al., *Boundary IoU: Improving Object-Centric Image Segmentation Evaluation*, CVPR 2021，DOI `10.1109/CVPR46437.2021.01508` 及作者官方 API 已核对。后继 protocol 固定 one-vs-rest、ignore、空类和 macro aggregation，历史 corruption-band mIoU 不复用为 Boundary IoU。
- **共同边界：** official test 继续 `sealed_unread`；不为获得显著结果而追加剂量、seed、样本或阈值。若未来引入复杂层级模型、BCa 区间、序贯检验或多重比较，再单独补直接统计依据并建立新 protocol identity。

## 10. 后续 checkpoint 数量与数据盘清理



**大白话问题：** 训练期使用的是低成本单尺度 validation，最终选择使用五尺度翻转主 evaluator；如果只保留少量单尺度高分点，可能漏掉主 evaluator 更好的 checkpoint。增加候选又会增加磁盘和本地评估时间，怎样取得可控平衡？

**当前状态：已处置。当前冻结 v1 的 top 3 + latest 已完成主评估，并实际观察到 selector 排名与主 evaluator 排名不同：selector 第一的 epoch 480 只排主评估第三，最终胜者为 epoch 420。后续独立 v2 使用 top 8 + latest、最多 9 个去重候选；v2 与只读清理门禁已提交，不回写本次 v1。**

- protocol v3 继续兼容历史 top 3，并允许正整数 `top_k`，上限固定为 8；后续配置和模板使用独立 `museg-dformerv2-s-rgb-quick-b0-v2-top8` 身份，不回写当前运行的 config、protocol 或候选清单。
- top 8 仍按同一 original-full、尺度 1.0、无 flip 的 mIoU 排序，同分优先更早 epoch；`latest.pth` 持续覆盖，最终清单按 checkpoint SHA-256 去重。因此候选最多是 9 个，而不是每 10 epoch 的全部 50 个 checkpoint。
- 增加候选只能降低 selector 与主 evaluator 排序不一致导致的漏选概率，不能声称完全消除风险。最终 checkpoint 仍由冻结的五尺度翻转主 evaluator 决定。
- 本机保守外推从当前 v1 的最多 4 个约 `2.8` 小时扩展到后续最多 9 个约 `6.3` 小时，仍低于 8 小时硬上限；继续串行评估，不并发复制模型争抢显存。
- `tools/audit_museg_cloud_storage.py` 只生成候选占用、剩余空间和 checkpoint 纯文件预算，不提供删除参数。候选必须是数据盘下的显式现存路径，且不能与仓库、当前输出、共享数据、official-test/split authority、预训练权重或其他保护路径重叠。
- 删除前必须先把归档取回本地并重新核验 SHA-256；OpenList 个人云盘副本是额外备份，不以任务页面的“成功”单独替代哈希证据。实际删除必须由用户确认每个规范化绝对路径后人工逐项执行，禁止通配符或模糊名称清理。
- v2 与清理门禁已随提交 `773c508e68d21491ad71d53f5967c3f76dc69ae6` 推送到 `origin/main`。后续使用 v2 仍需从干净 commit 物化新 protocol、通过正式 preflight，并分别取得训练和云生命周期授权。

## 11. DVC-A1 边界候选覆盖不足的后继协议

**大白话问题：** 当前规则只把相邻有效深度的相对跳变不低于 `0.05` 视为边界。全量扫描后，约三成位置组完全没有能形成非空 q75 的样本；继续沿用 v1 会让这些组无法接受预定干预，直接放宽规则又会改变原来预注册的问题。

**当前状态：本轮已处置为条件性开发验证方案，并完成 v3 定义修正门禁。** `DVC-A1-valdev-boundary-zero-v1` 与 `DVC-A1-valdev-boundary-zero-v2` 保持各自的 `protocol-blocked` 历史终态；独立 `DVC-A1-valdev-boundary-zero-v3-bgcontext` 保留 v2 数据和 corruption 不变量，仅分离有效 background 与 true ignore 的 Boundary IoU 标签契约。

- 直接核验事实：318 条 `val-dev` 的 mask 扫描全部完成；58/196 个 location group 的 `boundary-q75` 不可构造，占 `29.5918%`，高于 v1 预注册 `5%` 上限；非边界 q50 同面积候选不足为 0 条。
- v2 固定范围：138 个纳入组包含全部 218 张图；其中 31 张图的 q75 实际置零数仍为 0，123 个组内每张图均可构造 q75，另有 15 个组为部分可构造。主分析保留 138 组全部图像，并预注册 123 组敏感性分析，不在模型结果后择优。
- 研究对象：只估计“具有至少一个可执行深度边界干预的 `val-dev` 位置组中，冻结 Quick-B0 对人工边界失效的开发期敏感性”。58 个未纳入组只能解释为当前 `0.05` 操作定义无法施加 q75，不能解释为模型没有问题。
- v1 合法终点：保持 `protocol-blocked`，不生成五条件 mIoU、Boundary IoU、bootstrap 或问题假设裁决；不得改写成 `not-supported`。
- v2 不变量：不降低 `0.05`，不重新划分 `train-dev`/`val-dev`，不重新训练或选择 checkpoint，不读取 official test；继续保留 `clean`、`boundary-q25/q50/q75` 和 `nonboundary-q50` 五个条件，以 location group 为 bootstrap 单位。
- 当前禁止：不得覆盖或回写 v1，不得把 31 张 q75 空图伪装成实际受干预样本，不得根据模型结果删除组、修改阈值、追加条件或改成功门槛。
- 当前执行状态：v2 protocol 与 138 组/218 样本 evaluation allowlist 已物化，两样本本地 GPU preflight 已通过；218 张图的完整五条件本地 GPU 评价已结束，五个 condition 均写出 218 样本/138 组，但 `dose_effect` 只有 137/138 个有效配对组，状态为 `protocol-blocked`。定义层只读诊断已确认无效组 `06-01-01-0346` 包含 4 张图；四张图虽有 `cable`、`tube`、`rescue equipment` 前景，但在 29 像素 ignore 安全距离下计分安全区均为空，故 15 个类别全部双空 `None`，图像级和组级 Boundary IoU 均未定义。详细证据见 `doc/reports/2026-09-09-museg-dvc-a1-v2-group-definition-diagnosis.md`；123 组敏感性范围的两个 effect 均为 123/123。
- **v3 定义修正与处置：** v2 的唯一无效组 `06-01-01-0346` 的根因已确认是标签域混用：训练输入需要 raw background `0 -> evaluator ignore 255`，但 Boundary IoU 几何计算需要把 raw background 保留为有效 one-vs-rest 上下文。已建立独立 `DVC-A1-valdev-boundary-zero-v3-bgcontext`，不改变 v2 的 0.05 阈值、218 张图/138 组 allowlist、五个 condition、checkpoint、evaluator、bootstrap 和裁决门槛。v3 的 metric target 使用 raw foreground `1..15 -> 0..14`、background `0 -> 15`、true ignore `255`；15 个前景类继续报告，背景不作为报告类别。218 张图 CPU 标签域审计通过，`06-01-01-0346` 四张图的有效安全域均为 `1,008,424` 像素，定义层不再为空；两样本本地 GPU preflight 也明确覆盖该组并通过。大白话说，这次修正让真实背景参与边界几何计算，但没有把背景变成待报告类别，也没有修改原 v2 的数据范围或退化强度。
- **v3 当前边界：** v3 protocol SHA-256 为 `f9960904f51cec11797ada6952c2102da4b2b6832d0bf7b529898bfae9c0f216`，allowlist SHA-256 为 `5589eb3378ed2e23180f6205e2d88cea39702ad4bfd5d4e1b739cf2f920a8d89`，allowlist summary SHA-256 为 `6fa94de96f1b5b4e94c1feecdc4d821e05db1828be05b011f3b48f43ce408dfd`；CPU 标签域审计 SHA-256 为 `17a4ec36ba231c3be6ea1ed1f4f6e3b9f8380d8d0a619cc3530a6bfd903ad1db`，GPU preflight SHA-256 为 `829b580b6ed4ace977cf578e8391bcc759fc6d135fad72c2bd0dba958712dedd`。完整 218 张图 × 5 condition GPU 评价、bootstrap 和科学裁决均已完成，主裁决为 `not-supported`；训练、云资源、official test 和可学习门控仍未授权。用户明确要求在 A 未支持的情况下先推进 B，因此 `DVG-B1-oracle-gsa-v1` 已补齐项目内实现锚点并暂停于 A/B/C 外部参考冻结门禁；尚未创建 protocol，也未进入代码或运行阶段。
- 证据：v1 门禁报告为 `doc/reports/2026-09-08-museg-dvc-a1-protocol-gate.md`，仓库外权威运行证据位于 `cloud/DVC-A1-valdev-boundary-zero-v1/attempt-2/`，其中 mask manifest SHA-256 为 `60b988b3f9ffaabc5f6540cfccda48ddb5efd4d44ce360691bfaeea047e63f29`；v2 物化与 preflight 报告为 `doc/reports/2026-09-08-museg-dvc-a1-v2-materialization-preflight.md`，仓库外证据位于 `cloud/DVC-A1-valdev-boundary-zero-v2/`。

## 12. DVG-B1 Oracle mask 到 GSA depth contribution 的冻结门禁

**大白话问题：** 门控插在哪里、怎样传参数以及哪些 evaluator 能复用都已经核清；现在真正未定的是坏像素怎样变成每级 token 可靠性、token 可靠性怎样变成成对 gate，以及用多大收益和 clean 保持标准裁决方案。

**当前状态：项目内实现锚点已关闭，计划暂停于 A/B/C 三组 `reference-blocked` 项。** `04-DVG-B1条件式Oracle门控.md` 已按当前代码和作者原始保留副本收紧；A、B 必须补直接实现参考，C 必须补裁决参考或由用户明确作出项目预注册选择。三组关闭前不创建 protocol，不修改代码，不运行 preflight 或 GPU。

- **已闭合的代码事实：** `models/encoders/DFormerv2.py` 的 `GeoPriorGen.forward` 在 spatial/depth 两项加和前明确暴露 `mask_d_h`、`mask_d_w` 和 `mask_d`；前三个 stage 使用 H/W 分解 GSA，第四个 stage 使用 Full GSA。最小实现只允许门控 `self.weight[1] * mask_d*`，不得改 spatial contribution、`sin/cos`、Q/K/V、Depth 输入、decoder 或最终 logits。最小参数链为 `EncoderDecoder.forward/encode_decode` → `dformerv2.forward` → `BasicLayer.forward` → `RGBD_Block.forward` → `GeoPriorGen.forward`；Attention 继续只消费合成后的 geometry prior。
- **已闭合的上游差异：** 当前和作者原始保留副本的 `DFormerv2.py` SHA-256 均为 `2b0b77ea401d56993aac915883bcb43035927ec991501dba94fb029901009332`，完整差异检查退出码为 `0`，两者都使用 `F.interpolate(..., mode="bilinear", align_corners=False)`。论文文字描述 average pooling，但这不是 MUSeg/MVE 适配或当前项目意外修改；B1 基线语义以 checkpoint 对应的 bilinear 路径为准。
- **已闭合的复用边界：** 原始 `Depth16` corruption mask、五个 condition、mask 确定性/嵌套/数量/哈希、五尺度翻转 10 view、右侧/底部 padding、逆 flip、原始 Label 网格、FP32 pre-softmax 平均、strict checkpoint load、q=0 输入数组等价、finite/shape/JSON preflight 框架均可复用。
- **已闭合的 no-op 验收规则：** `None`、clean、q=0 和全可信 mask 必须归一化到原始未修改 forward 旁路；逐 stage 输出和最终 pre-softmax logits 都要求 `torch.equal` 完全相等，不通过放宽浮点容差解决旁路不等价。该旁路尚未实现，也未运行等价检查。
- **开放项 A：像素 mask → 四级 token reliability。** 必须冻结 view-scale resize、Stage 0–3 聚合算子、部分受损 patch 的 reliability 语义，以及它与 bilinear Depth resize 的对齐解释。候选 nearest、area/average、max/any-invalid 或连续有效比例均不得由项目自行猜测，也不得在结果后选择。
- **开放项 B：token reliability → pairwise gate。** 必须冻结 Full `[B,1,L,L]`、H `[B,1,W,H,H]`、W `[B,1,H,W,W]` 的公式，以及两端乘积、最小值、query-only、key-only 或其他组合、对称性和 hard/continuous 选择。
- **开放项 C：正式科学裁决。** 必须冻结 `oracle-supported` 的最小实际效应量、clean 不劣容忍度，以及 Boundary IoU 与 mIoU 是否足够或还需额外指标。现有项目规则和已引参考不能直接填入数值；等待用户补充直接参考，或明确记录为项目预注册选择。
- **恢复点和证据：** A、B 的 WOS 靶向检索式与待填字段、C 的精确待决问题和全部项目内证据入口见 `doc/plans/2026-09-MUSeg-几何可信RGBD双路径MVE/04-DVG-B1条件式Oracle门控.md`；实现定位详证见 `liu-test-exp/方案1/DVG-B1-必须实现细节靶向检索步骤与WOS检索式.md`。本次没有执行外部 Web 搜索。代码、preflight、GPU、训练、云资源和 official test 仍未授权。

**大白话说明：** 以后不用再搜索“GSA 到底在哪里”或“现有 evaluator 能不能复用”；只需补 A、B、C 的直接依据并把唯一规则填回计划，之后再决定是否授权 protocol 和代码。
