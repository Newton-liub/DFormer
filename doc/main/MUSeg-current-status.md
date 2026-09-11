# MUSeg 当前状态与唯一实时入口

> **状态时间：** 2026-09-11 03:20 UTC
> **当前阶段：** `DVC-A1-valdev-boundary-zero-v1` 与 v2 保持各自的 `protocol-blocked` 历史终态，独立 v3 已完成并裁决为 `not-supported`。`DVG-B1-oracle-gsa-v1` 的项目内锚点、11 篇全文和常见官方代码核验已完成；用户已在查看模型结果前选择 A/B 项目预注册候选：四级连续 token reliability 使用 view 级双线性对齐与 stage 级有效面积比例，Full/H/W gate 使用 query/key 两端 reliability 的对称乘积。A/B 尚未实现或验证；C 的最小实际效应量、95% 区间规则和 mIoU 裁决职责仍待用户冻结。尚未创建 B1 protocol。
> **大白话说明：** A/B 已经选定，不再让后续模型重新搜索或自行换公式。新对话先只补 C；C 关闭后再按 protocol 物化、CPU 算子检查、严格 no-op、1–2 样本 preflight 的顺序推进，完整 GPU 评价最后单独申请。本轮只有文档计划变化，没有模型代码或实验结果。
> 本文件是 MUSeg 当前事实、授权边界、证据入口和恢复规则的唯一实时入口；计划、报告、审计和 Canvas 只承担各自形成时点的历史或详细证据职责。

## 1. 稳定基线

- **模型与输入：** `DFormerv2-S RGB Quick-B0`，使用 RGB 输入契约 `rgb-imagenet-rgb-order-v1`；B1 `safe_masked_mean` 数值稳定性修复保留在基线中。
- **最终 checkpoint：** epoch 420 的 `selector-epoch-420.pth`，SHA-256 为 `f246a3afc50334c81302b7bfebdadf7cf37d00326bf1c3aa54f6a151754e3a1c`。
- **开发 split：** `val-dev` 共 318 条，split SHA-256 为 `1d0719d8f64f016d48995c25ab66d4004d76b7155d9efeef7cbb7454c0dd0e83`；official test 仍为 `sealed_unread`（封存未读）。
- **主 evaluator：** `msflip-whole-original-grid-v1`，5 个尺度分别使用原图与水平翻转，共 10 个 view；输出恢复到 MUSeg 原始 Label 网格，以 FP32 平均 pre-softmax logits 后计分。
- **主指标：** mIoU `58.79`、mAcc `69.91`、mF1 `72.73`。
- **基线性质：** 这是 single-seed、RGB、development Quick-B0，作为后续模块探索的固定内部对照；它不是三 seed 的完整论文复现，也不是颜色优劣实验结论。
- **权威证据：** 主评估裁决见 `doc/reports/2026-08-31-museg-quick-b0-main-evaluation.md`；综合报告见 `doc/reports/2026-08-31-museg-dformerv2-quick-baseline-comprehensive.md`；完整本地证据位于仓库外的 `cloud/DFormer-quick-b0-evidence/museg-dformerv2-s-rgb-quick-b0-v1/`（该目录不纳入 Git）。

## 2. 当前研究方向状态

- 2026-09-08 的计划门禁已确认：DFormerv2 的 Depth 在四级 GSA 中形成几何先验；现有置信融合/回退文献依赖专门训练，首轮不能支持冻结 logits 门控；三维评价、严格 RGB-only 回退、亮度引导补全、高置信阈值和可学习 logits 门控均已从首轮删除。`RE447` 不适合作 bootstrap 来源，配对 location-group 重采样采用项目预注册流程；历史 `boundary_band_mIoU` 不等同标准 Boundary IoU。详细裁决保留在本计划目录与 `MUSeg-open-decisions.md`。
- `DVC-A1-valdev-boundary-zero-v1` 的 protocol 模板、专用物化/校验器、确定性 Depth16 corruption、标准 Boundary IoU、location-group bootstrap/裁决和五尺度翻转运行入口已实现于当前未提交工作区；聚焦测试为 `5 passed`，两条真实 `val-dev` 样本的 GPU preflight 已通过 q=0 decoded-array 等价、mask 确定性/嵌套/面积匹配、有限值、strict checkpoint load 和原始 `932×1082` Label 网格恢复。
- 2026-09-08 的完整 mask 门禁已读取 318 条 `val-dev` 的 Depth16 并完成全部 mask 扫描：58/196 个 location group 无法构造非空 `boundary-q75`，比例 `29.5918%`，高于预注册上限 `5%`；非边界 q50 候选不足为 0 条。因此裁决为 `protocol-blocked`，完整 5 condition × 10 view 模型评价没有开始，也没有生成 mIoU、Boundary IoU、bootstrap 区间或 `supported/not-supported/inconclusive` 科学裁决。大白话说，v1 的操作定义覆盖不了足够多的位置，所以按原规则停止；这个历史结论保持不变。
- 2026-09-08 已处置并物化后继研究选择：独立 `DVC-A1-valdev-boundary-zero-v2` 保持全局相对深度跳变阈值 `0.05` 和冻结 `val-dev` 不变，预先纳入至少一张图可构造非空 q75 的全部 138 个 location group，共 218 张图。31 张图在组级纳入后仍无实际 q75 置零，主分析必须保留并显式报告；另对 123 个组内每张图均可构造 q75 的组做预注册敏感性分析。修正前 v2 protocol、allowlist 和两样本本地 GPU preflight 已物化/通过；保护修正后的 protocol 已重新物化，修正后的 preflight 已通过，完整评价随后按批准执行并以 `protocol-blocked` 结束。大白话说，评价范围已固定，正式推理已完成，但统计保护发现一个预注册组的 Boundary IoU 未定义，当前没有模型好坏或问题是否成立的结论。
- 2026-09-08 的最终差异审查已落实：v2 运行入口现在会把新生成的 218 样本、五个 condition 的 mask 哈希逐一比对冻结 v1 manifest，并在形成统计量前强制主分析 138 组和敏感性分析 123 组的两个 effect 都有对应数量的有效配对组。修正后的两样本 preflight 已通过，完整评价实际于 `2026-09-08T14:34:00.225007+00:00` 启动，并于约 90 分 45.565 秒后结束。运行实际写出五个 condition 产物，但 `dose_effect` 的 138 组门禁发现只有 137 个有效配对组，遂以 `protocol-blocked` 结束。只读核对显示无效组为 `06-01-01-0346`；在 123 组敏感性范围内，两个 effect 均为 123 个有效配对组。大白话说，模型推理跑完了，但预注册统计范围与 Boundary IoU 可定义范围仍有一个组不一致，所以当前数值不能作为科学结果。
- **当前授权与恢复点：** v3 定义修正、218 张图 CPU 标签域审计、两样本 CUDA preflight 和完整 218 张图 × 5 condition GPU 评价均已完成，主裁决为 `not-supported`。用户已要求继续独立 `DVG-B1-oracle-gsa-v1` 方案筛选，并已把 A/B 记录为项目预注册候选：A 使用 view 级 `INTER_LINEAR` 对齐、resize 后 flip、右/下 padding `r=1` 和 stage 级 `INTER_AREA` 连续有效面积比例；B 使用 Full/H/W query/key 对称乘积 gate，只门控 depth geometry contribution。clean/q=0/`None`/全可信 mask 固定走原 forward 旁路并以 `torch.equal` 验收，指标固定 Boundary IoU + mIoU。当前唯一第一门禁是 C 的最小实际效应量、95% 区间规则和 mIoU 裁决职责。C 未冻结前不创建最终 protocol、不修改模型代码、不运行模型 forward；之后仍须依次通过 protocol 物化、CPU qualification、no-op 和 1–2 样本 preflight，完整 GPU paired development evaluation 另行授权。大白话说，A/B 已选但还没做；下一次先定科学成功门槛，再逐级检查，不能直接跑完整实验。
- 2026-09-10 已完成 B1 的项目内实现锚点核对并收紧目标计划：`GeoPriorGen.forward` 中 spatial contribution 与 depth geometry contribution 可在加和前分离，前三个 stage 使用 H/W 分解 GSA、第四个 stage 使用 Full GSA；最小 Oracle 参数链和唯一 depth-only 插入点已明确，Attention 继续只消费合成后的 geometry prior。现有五尺度翻转、Depth16 corruption、q=0 输入等价和 preflight/JSON 框架可直接复用。当前 `models/encoders/DFormerv2.py` 与作者原始保留副本 `D:/0Project/origin/DFormer/models/encoders/DFormerv2.py` 已再次完整核对：两者 SHA-256 均为 `2b0b77ea401d56993aac915883bcb43035927ec991501dba94fb029901009332`，完整差异检查退出码为 `0`，且都使用 bilinear interpolation；论文文字描述 average pooling。因此该差异属于作者论文与作者代码之间的上游差异，不是 MUSeg/MVE 适配或当前项目意外修改，B1 基线语义以 checkpoint 对应的 bilinear 路径为准。截至该核验时点仍处于 `reference-blocked` 的内容分为三组：A 为像素 corruption mask 到四级 token reliability 的确定性聚合，B 为 token reliability 到 Full/H/W pairwise depth contribution gate 的提升，C 为 `oracle-supported` 最小实际效应量、clean 不劣容忍度和额外指标必要性。A、B 保留 WOS 靶向检索式和待填字段；C 等待用户补充参考或明确项目预注册选择。本次未执行外部 Web 搜索。证据入口为 `04-DVG-B1条件式Oracle门控.md` 与 `liu-test-exp/方案1/DVG-B1-必须实现细节靶向检索步骤与WOS检索式.md`，研究选择见 `MUSeg-open-decisions.md` 第 12 节。代码、preflight、GPU、训练、云资源和 official test 仍未授权。大白话说，现有 GSA 实现已经核清；下一步只补 A、B、C 的规则依据，不写代码也不跑模型。
- 2026-09-11 已只读核对 `doc/paper/补充材料2` 中对应优先清单的 11 篇全文，并把公式、方向性、归一化、hard/continuous、代码 URL 和负证据回填到 `liu-test-exp/方案1/DVG-B1-必须实现细节靶向检索步骤与WOS检索式.md`。A 的最强直接证据来自 *Confidence Propagation through CNNs*：二值 validity 可作为连续 confidence，逐层 confidence 使用 normalized aggregation，多尺度下采样采用 confidence max pooling 与 argmax feature selection；但没有 DFormerv2 四级 kernel/stride/padding、空窗口严格规则或与 `bilinear, align_corners=False` 的共同对齐。B 的直接证据包括 NL-3A 与 GSCS/SPN 的邻居/key-only 连续公式 $w_{p,q}=c_q\hat w_{p,q}$、NR-MVSNet 的 query-only hard threshold，以及 SMAC 的 $HW\times HW$ Full attention 拓扑；这些机制互不等价，也没有任何一篇同时给出 Full/H/W 三种 gate。因此在全文核验完成的该时点，A/B 仍为 `reference-blocked`，恢复点是补官方代码或由用户作项目预注册选择；后续官方代码核验与用户选择见下两条更新。
- 2026-09-11 已继续完成 11 篇论文的官方代码核验并更新同一检索文档：确认 [1]、[3]、[5]、[9]、[10] 和 [11] 存在与论文绑定的作者/官方仓库，但 [11] SMAC 仓库只含 ReDWeb-S 数据集、统计与结果，没有模型代码；作者前作 S2MA 只能作为邻近 Full-affinity 拓扑参考。`D:/0Project/origin` 下 `nconv`、`nconv-nyu`、`LightDepth`、`OPM-MVS`、`NR-MVSNet`、`LFDA`、`SMAC`、`S2MA` 已固定分支、commit、许可证状态和函数锚点，复核时 8 个仓库均无工作区改动。nconv 代码的 `maxpool/4` 与零 padding 不满足 strict all-1 reliability，NR-MVSNet 是 `[B,1,H,W]` query/location hard gate，LFDA 是 candidate attention，S2MA 不是 SMAC 实现；代码仍没有给出统一 Full/H/W gate。两个内联 CPU probe 仅核对 resize/pooling/padding/flip 和 query/key/product/min 的算子语义，未加载模型或 checkpoint。在官方代码核验完成的该时点，A/B 仍为 `reference-blocked`，等待用户决定继续窄搜或作项目预注册选择；后续选择已记录在下一条 03:20 UTC 更新中。该时点仍未创建 B1 protocol，也未修改模型代码或运行模型 forward、preflight、GPU、训练、云资源、official test。
- 2026-09-11 03:20 UTC，用户已选择把 A/B 固定为项目预注册候选，并完成 `04-DVG-B1条件式Oracle门控.md`、新对话入口、总规划、检索证据和开放决策同步。A 采用“原图 reliability `INTER_LINEAR` 到 view → resize 后 flip → 右/下可信 padding → `INTER_AREA` 到四级 token”的连续有效面积比例；B 采用 Full/H/W query/key 对称乘积。clean 旁路严格 `torch.equal`，指标集合固定 Boundary IoU + mIoU；C 的最小实际效应量、95% 区间规则和 mIoU 职责仍待冻结。计划已细化为 P0 protocol、P1 CPU qualification、P2 no-op、P3 1–2 样本 preflight、P4 另行授权完整评价。没有创建 protocol，没有修改模型代码，也没有运行模型 forward、GPU、训练、云资源或 official test。大白话说，下一模型不用再选 A/B；只需先补 C，再照门禁顺序执行。
- **v2 证据：** 仓库外 `cloud/DVC-A1-valdev-boundary-zero-v2/`；修正后 protocol SHA-256 为 `3c6f33562692c8baee85786261de431583e9bdd9b6f8f51cd5ad0f042406f6bb`，allowlist SHA-256 仍为 `5589eb3378ed2e23180f6205e2d88cea39702ad4bfd5d4e1b739cf2f920a8d89`，allowlist 摘要 SHA-256 仍为 `f451955d91143d118f4445181ceb8ff7212ced5ba8cf2add12981cd1e75287c2`，修正后 preflight SHA-256 为 `95afc2f071656569a7f3ee18d3244c5f0abfc5ef59c349825e170ccc314c0e9d`。完整运行记录为 `cloud/DVC-A1-valdev-boundary-zero-v2/executions/20260908T143400225007+0000-full.json`，失败记录为 `cloud/DVC-A1-valdev-boundary-zero-v2/full-failure.json`；均记录 `official_test_included=false`。五个 condition JSON 均已写出，每个包含 218 样本/138 组，但统计因 `dose_effect` 只有 137 个有效配对组而阻塞。无效组定义诊断见 `doc/reports/2026-09-09-museg-dvc-a1-v2-group-definition-diagnosis.md`：4 张图的 29 像素 ignore 安全区均为空，三类原始前景在计分安全区中均被排除，故 15 类全部双空 `None`。组级诊断报告见 `doc/reports/2026-09-09-museg-dvc-a1-v2-protocol-blocked.md`；物化/preflight 历史报告见 `doc/reports/2026-09-08-museg-dvc-a1-v2-materialization-preflight.md`，保护修正报告见 `doc/reports/2026-09-08-museg-dvc-a1-v2-protection-fix.md`，修正后 preflight 报告见 `doc/reports/2026-09-08-museg-dvc-a1-v2-protection-preflight.md`。
- **v1 证据：** 日期化报告为 `doc/reports/2026-09-08-museg-dvc-a1-protocol-gate.md`；仓库外权威运行证据位于 `cloud/DVC-A1-valdev-boundary-zero-v1/attempt-2/`。最终 protocol SHA-256 为 `7bca3c109905d7d4ed228359bf8cd2a20bee18cbcd3d4cbdfed871e09870fba2`，preflight SHA-256 为 `aeb0afff829fc5abcd86196cb6c27822055b90c7910711c2da956a2133dc0c45`，mask manifest SHA-256 为 `60b988b3f9ffaabc5f6540cfccda48ddb5efd4d44ce360691bfaeea047e63f29`，failure SHA-256 为 `b36d9bddfb33a4f69657dae976f94afa6e97282a42aec27663949502be08e216`；均记录 `official_test_included=false`。
- **v3 证据：** 定义门禁证据位于仓库外 `cloud/DVC-A1-valdev-boundary-zero-v3-bgcontext/`；完整评价证据位于 `cloud/DVC-A1-valdev-boundary-zero-v3-bgcontext-full-20260909/`。定义门禁 protocol SHA-256 为 `f9960904f51cec11797ada6952c2102da4b2b6832d0bf7b529898bfae9c0f216`，CPU 标签域审计 SHA-256 为 `17a4ec36ba231c3be6ea1ed1f4f6e3b9f8380d8d0a619cc3530a6bfd903ad1db`，原 preflight SHA-256 为 `829b580b6ed4ace977cf578e8391bcc759fc6d135fad72c2bd0dba958712dedd`。完整评价 protocol SHA-256 为 `d52b3dba2c7a34894b9f4cdf1d8e313e304415d1ea821a191b7753b97be74f5d`，mask manifest SHA-256 为 `3da28ae84806c61b0178c9563e811eed9e6f16e3f0f4040ab5753b26abdc9e79`，summary SHA-256 为 `951d8e2005a4c64ac809ffd3de9d7db367df955a46e81c46e89516bd64d9f220`。完整运行退出码为 `0`，状态 `completed`，时长 `5507.055` 秒；主分析 `dose_effect` 为 `+0.0731348717` 个百分点（95% 区间 `[-0.0620441424,+0.2226503089]`），`specificity_effect` 为 `+0.0323919541` 个百分点（95% 区间 `[-0.3002343847,+0.2899672010]`），两者均为 `138/138` 有效配对组，主裁决为 `not-supported`。完整运行记录 `official_test_included=false`。
- A2/B2 深度有效性方向已迁入 `doc/plans/deferred/2026-09-MUSeg-unexecuted/MUSeg-A2-B2深度有效性/`，状态为**延期、未执行、未授权、当前不处于恢复点**。
- 方向1后验校准与 Depth 退化双路径计划已迁入 `doc/plans/deferred/2026-09-MUSeg-unexecuted/MUSeg-方向1最短验证路径/`，状态同样为**延期、未执行、未授权、当前不处于恢复点**。
- 两套延期计划只保留未来重新启用时的候选设计。重新启用必须先从稳定基准建立独立研究分支，重新确认数据、protocol、evaluator 和授权；计划中的“当前任务”“下一步”或“恢复点”不构成执行依据。
- 方向1仍需使用的研究设计提示词和论文编号索引已经随计划收进其 `参考资料/` 与 `补充内容/`，不再作为活跃草稿目录入口。

## 3. 数据、评估和 official-test 边界

- B0、A2/B2 候选方向和未来模块消融都必须显式绑定 checkpoint、split、输入契约、metric geometry、evaluator、protocol 和产物哈希。
- `val-dev` 已参与 B0 checkpoint 选择；它不能在没有新数据职责和新 protocol 的情况下直接改作方向1的独立 calibration/evaluation 集。
- A2/B2 若未来重新启用，只能先在 `val-dev` 上按新 protocol 处理；人工 corruption 证据与自然无效深度证据的外推边界按 `MUSeg-open-decisions.md` 执行。
- official test 在模型选择、阈值冻结、方向筛选、开发评估和恢复流程中继续保持 `sealed_unread`；任何解封都需要独立门禁和单独授权。
- 本轮已完成 v1 DVC-A1 代码、聚焦测试、两样本本地 GPU preflight 和全量 Depth16 mask 门禁，v1 因覆盖门禁 `protocol-blocked`；随后完成独立 v2 protocol/allowlist 物化、范围核验、完整五条件 GPU 评价和 `protocol-blocked` 终止；最后完成 v3 标签契约修正、218 张图 CPU 标签域审计、新 protocol 物化、两样本本地 GPU preflight、完整 218 张图 × 5 condition GPU 评价、location-group bootstrap 和预注册 `not-supported` 裁决。训练、云端操作和 official test 均未运行。

## 4. 证据位置与历史解释

- 2026-09-11 的当前方向入口：`doc/plans/2026-09-MUSeg-几何可信RGBD双路径MVE/01-新对话最小上下文与当前任务.md`；`04-DVG-B1条件式Oracle门控.md` 已记录 A/B 项目预注册候选、C 第一门禁和 P0–P4 分阶段执行顺序。当前未创建 B1 protocol，未授权代码、模型 forward、preflight、GPU、训练、云资源或 official test。`03-共享协议与DVC-A1问题验证.md` 保留 v1/v2 的 `protocol-blocked` 记录，并记录 v3 完整评价和 `not-supported` 收口。日期化历史证据报告与完整评价证据位置保持不变。
- 2026-09-10 已生成覆盖 2026-08-17 至 2026-09-10 的组会正式报告 `doc/reports/2026-09-10-museg-dformerv2-to-mve-group-meeting.md`，并生成、发布版本化 Canvas `doc/canvases/0.0.13-museg-dformerv2-to-mve-group-meeting.canvas.tsx`；两者已登记到 `doc/reports/report-index.json`，下一个 Canvas 版本为 `0.0.14`。大白话说，明日组会需要的长报告和可视化展示版已经落地，但本次没有产生新的训练、评价或科学裁决。
- 正式报告和 Canvas 元数据入口：`doc/reports/report-index.json`。
- 历史计划入口：`doc/plans/`；已完成 Quick-B0 与更早阶段计划位于 `doc/plans/archive/`。
- 未执行候选计划入口：`doc/plans/deferred/2026-09-MUSeg-unexecuted/README.md`。
- 稳定项目指南：`doc/guides/project/README.md`；当前文件职责索引：`doc/guides/project/file-catalog.md`；完整旧目录作为日期化历史快照保留在同目录。
- 2026-09-05 上下文归档清单：`doc/archive-manifests/2026-09-05-context-cleanup.md`；归档 ZIP 在本地 `archive-export/`，已被 `.gitignore` 排除。
- 历史报告中的“正在运行”“当前计划”或“待闭合”只按报告形成时点理解；若报告正文没有后继指针，以本文件和报告索引为准，不把历史措辞当作当前事实。

## 5. Git 基准与治理状态

- **整理状态：** 已完成本轮文档、归档清单、目录迁移、引用治理和 Git 元数据治理；整理提交为 `de731aa611db1ba3a6511153c29f7e568f91da09`，随后在 `main` 上追加 Cursor skill 提交 `be07e96390a75b1d7651c1bf29979153d7f74a49`。本地 annotated tag `museg-research-base-v1` 当前指向 `main` 的最新提交，远端不推送。
- **2026-09-07 Git 状态：** `main` 已包含稳定文档提交和其后的 skill 提交，当前本地 `main` 比 `origin/main` 超前 1 个提交；本地标签名 `museg-research-base-v1` 已移动到 `main` 最新提交。研发分支 `research/lowlight-dust-rgbd-recovery` 已从该标签创建并作为当前工作分支；此前的临时维护分支和旧研发分支已删除。
- **大白话说明：** `doc` 和 `skill` 现在属于同一条 `main` 历史；标签代表包含这两个提交的最新本地基准，新研发工作从这个位置开始，远端 `origin/main` 没有被推送或改写。
- 整理提交和 tag 创建后，工作区应保持干净；最终直接核验 tag 指向、工作区状态以及 `main` 与 `origin/main` 的差异。
- 稳定基准治理指南：`doc/guides/project/research-branch-governance.md`。未来每篇论文或独立研究使用 `research/<topic>` 分支，并从该稳定基准创建。
- Git 整理完成后，后续方向不得从 B0 最终 checkpoint 续训后冒充公平消融；必须从同一官方 pretrained 独立训练，并建立新的 config、protocol 和证据身份。

## 6. 恢复规则

1. 每个 MUSeg 对话先读取本文件；涉及研究选择时再读取 `doc/main/MUSeg-open-decisions.md`。
2. 先确认本文件中的当前授权边界和恢复点；延期计划、历史报告和旧交接材料不能单独触发恢复。
3. 新方向先建立独立 `research/<topic>` 分支，冻结 protocol/config/evaluator/split 身份，再按具体操作取得授权。
4. 任何训练、checkpoint、指标、official-test、云实例、证据位置、阻塞项、恢复步骤、提交或发布状态变化，都必须在最终答复前更新本文件。
5. 纯文档或只读检查没有持久事实变化时，不更新时间；历史指标和原始 acceptance/result 文件不得为匹配新口径而改写。
