# MUSeg 当前状态与唯一实时入口

> **状态时间：** 2026-09-08 08:29 UTC
> **当前阶段：** `DVC-A1-valdev-boundary-zero-v1` 保持 `protocol-blocked`；后继 `DVC-A1-valdev-boundary-zero-v2` 已冻结为阈值 `0.05` 下全部 138 个可构造 location group 的条件性开发验证，尚未修改代码、物化 protocol、执行 preflight 或运行完整模型评价。
> **大白话说明：** v1 因约三成位置组无法施加 q75 而合法停止；现在已决定不降低阈值、不重划数据，改用独立 v2 在可施加该干预的全部 138 个位置组上验证，并如实报告全数据覆盖率。当前只保存了设计，实验还没有开始。
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
- 2026-09-08 已处置后继研究选择：建立独立 `DVC-A1-valdev-boundary-zero-v2`，保持全局相对深度跳变阈值 `0.05` 和冻结 `val-dev` 不变，预先纳入至少一张图可构造非空 q75 的全部 138 个 location group，共 218 张图。31 张图在组级纳入后仍无实际 q75 置零，主分析必须保留并显式报告；另对 123 个组内每张图均可构造 q75 的组做预注册敏感性分析。大白话说，v2 只回答“在当前强边界定义确实可施加干预的地点上，模型是否对边界深度失效敏感”，不能代表全部 MUSeg 地点。
- **当前授权与恢复点：** 本次只批准保存 `DVC-A1-valdev-boundary-zero-v2` 的研究设计和全 138 组范围，未授权代码修改、protocol 物化、preflight、完整 GPU 评价、训练、云资源或 official test。准确恢复点是先按 [`03-共享协议与DVC-A1问题验证.md`](../plans/2026-09-MUSeg-几何可信RGBD双路径MVE/03-共享协议与DVC-A1问题验证.md) 第 11 节物化并审核 v2 protocol 与 138 组/218 样本 evaluation allowlist；之后才可另行确认最小 preflight 和完整五条件本地 GPU 评价。v1 的 protocol、失败证据和 `protocol-blocked` 结论不得覆盖或回写，`DVG-B1` 继续未解锁。
- **证据：** 日期化报告为 `doc/reports/2026-09-08-museg-dvc-a1-protocol-gate.md`；仓库外权威运行证据位于 `cloud/DVC-A1-valdev-boundary-zero-v1/attempt-2/`。最终 protocol SHA-256 为 `7bca3c109905d7d4ed228359bf8cd2a20bee18cbcd3d4cbdfed871e09870fba2`，preflight SHA-256 为 `aeb0afff829fc5abcd86196cb6c27822055b90c7910711c2da956a2133dc0c45`，mask manifest SHA-256 为 `60b988b3f9ffaabc5f6540cfccda48ddb5efd4d44ce360691bfaeea047e63f29`，failure SHA-256 为 `b36d9bddfb33a4f69657dae976f94afa6e97282a42aec27663949502be08e216`；均记录 `official_test_included=false`。
- A2/B2 深度有效性方向已迁入 `doc/plans/deferred/2026-09-MUSeg-unexecuted/MUSeg-A2-B2深度有效性/`，状态为**延期、未执行、未授权、当前不处于恢复点**。
- 方向1后验校准与 Depth 退化双路径计划已迁入 `doc/plans/deferred/2026-09-MUSeg-unexecuted/MUSeg-方向1最短验证路径/`，状态同样为**延期、未执行、未授权、当前不处于恢复点**。
- 两套延期计划只保留未来重新启用时的候选设计。重新启用必须先从稳定基准建立独立研究分支，重新确认数据、protocol、evaluator 和授权；计划中的“当前任务”“下一步”或“恢复点”不构成执行依据。
- 方向1仍需使用的研究设计提示词和论文编号索引已经随计划收进其 `参考资料/` 与 `补充内容/`，不再作为活跃草稿目录入口。

## 3. 数据、评估和 official-test 边界

- B0、A2/B2 候选方向和未来模块消融都必须显式绑定 checkpoint、split、输入契约、metric geometry、evaluator、protocol 和产物哈希。
- `val-dev` 已参与 B0 checkpoint 选择；它不能在没有新数据职责和新 protocol 的情况下直接改作方向1的独立 calibration/evaluation 集。
- A2/B2 若未来重新启用，只能先在 `val-dev` 上按新 protocol 处理；人工 corruption 证据与自然无效深度证据的外推边界按 `MUSeg-open-decisions.md` 执行。
- official test 在模型选择、阈值冻结、方向筛选、开发评估和恢复流程中继续保持 `sealed_unread`；任何解封都需要独立门禁和单独授权。
- 本轮已完成 DVC-A1 代码、聚焦测试、两样本本地 GPU preflight 和全量 Depth16 mask 门禁；因 mask 门禁 `protocol-blocked`，没有运行完整五条件模型评价、训练、云端操作或 official test。

## 4. 证据位置与历史解释

- 2026-09-08 的新候选方向入口：`doc/plans/2026-09-MUSeg-几何可信RGBD双路径MVE/01-新对话最小上下文与当前任务.md`；`03-共享协议与DVC-A1问题验证.md` 已保留 v1 的 `protocol-blocked` 记录并追加 v2 全 138 个可构造组规划，日期化 v1 证据报告为 `doc/reports/2026-09-08-museg-dvc-a1-protocol-gate.md`；`04-DVG-B1条件式Oracle门控.md` 未解锁。
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
