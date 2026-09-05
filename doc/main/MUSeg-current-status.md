# MUSeg 当前状态与唯一实时入口

> **状态时间：** 2026-09-05 UTC
> **当前阶段：** 已完成的 RGB Quick-B0 作为稳定研究起点；文档与目录治理已完成并已在本地基准提交/tag 中固定。当前没有已授权的 MUSeg 研究执行项。
> **大白话说明：** 现在先把已完成基线和历史材料整理成不会误导后续工作的 Git 起点；A2/B2 与方向1都还没有开始，不能从计划文字直接恢复实验。
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

- 当前没有已授权的代码实现、实验、GPU、训练、云资源或 official test 操作。
- A2/B2 深度有效性方向已迁入 `doc/plans/deferred/2026-09-MUSeg-unexecuted/MUSeg-A2-B2深度有效性/`，状态为**延期、未执行、未授权、当前不处于恢复点**。
- 方向1后验校准与 Depth 退化双路径计划已迁入 `doc/plans/deferred/2026-09-MUSeg-unexecuted/MUSeg-方向1最短验证路径/`，状态同样为**延期、未执行、未授权、当前不处于恢复点**。
- 两套延期计划只保留未来重新启用时的候选设计。重新启用必须先从稳定基准建立独立研究分支，重新确认数据、protocol、evaluator 和授权；计划中的“当前任务”“下一步”或“恢复点”不构成执行依据。
- 方向1仍需使用的研究设计提示词和论文编号索引已经随计划收进其 `参考资料/` 与 `补充内容/`，不再作为活跃草稿目录入口。

## 3. 数据、评估和 official-test 边界

- B0、A2/B2 候选方向和未来模块消融都必须显式绑定 checkpoint、split、输入契约、metric geometry、evaluator、protocol 和产物哈希。
- `val-dev` 已参与 B0 checkpoint 选择；它不能在没有新数据职责和新 protocol 的情况下直接改作方向1的独立 calibration/evaluation 集。
- A2/B2 若未来重新启用，只能先在 `val-dev` 上按新 protocol 处理；人工 corruption 证据与自然无效深度证据的外推边界按 `MUSeg-open-decisions.md` 执行。
- official test 在模型选择、阈值冻结、方向筛选、开发评估和恢复流程中继续保持 `sealed_unread`；任何解封都需要独立门禁和单独授权。
- 本轮只做文档、归档、链接、上下文和 Git 元数据治理，不运行项目测试、完整 A2、GPU、训练、长耗时评估、云端操作或 official test。

## 4. 证据位置与历史解释

- 正式报告和 Canvas 元数据入口：`doc/reports/report-index.json`。
- 历史计划入口：`doc/plans/`；已完成 Quick-B0 与更早阶段计划位于 `doc/plans/archive/`。
- 未执行候选计划入口：`doc/plans/deferred/2026-09-MUSeg-unexecuted/README.md`。
- 稳定项目指南：`doc/guides/project/README.md`；当前文件职责索引：`doc/guides/project/file-catalog.md`；完整旧目录作为日期化历史快照保留在同目录。
- 2026-09-05 上下文归档清单：`doc/archive-manifests/2026-09-05-context-cleanup.md`；归档 ZIP 在本地 `archive-export/`，已被 `.gitignore` 排除。
- 历史报告中的“正在运行”“当前计划”或“待闭合”只按报告形成时点理解；若报告正文没有后继指针，以本文件和报告索引为准，不把历史措辞当作当前事实。

## 5. Git 基准与治理状态

- **整理状态：** 已完成本轮文档、归档清单、目录迁移、引用治理和 Git 元数据治理；整理提交为本文件所在的 `HEAD`，annotated tag `museg-research-base-v1` 将指向该最终提交，远端不推送。
- 整理提交和 tag 创建后，工作区应保持干净；最终直接核验 tag 指向、工作区状态以及 `main` 与 `origin/main` 的差异。
- 稳定基准治理指南：`doc/guides/project/research-branch-governance.md`。未来每篇论文或独立研究使用 `research/<topic>` 分支，并从该稳定基准创建。
- Git 整理完成后，后续方向不得从 B0 最终 checkpoint 续训后冒充公平消融；必须从同一官方 pretrained 独立训练，并建立新的 config、protocol 和证据身份。

## 6. 恢复规则

1. 每个 MUSeg 对话先读取本文件；涉及研究选择时再读取 `doc/main/MUSeg-open-decisions.md`。
2. 先确认本文件中的当前授权边界和恢复点；延期计划、历史报告和旧交接材料不能单独触发恢复。
3. 新方向先建立独立 `research/<topic>` 分支，冻结 protocol/config/evaluator/split 身份，再按具体操作取得授权。
4. 任何训练、checkpoint、指标、official-test、云实例、证据位置、阻塞项、恢复步骤、提交或发布状态变化，都必须在最终答复前更新本文件。
5. 纯文档或只读检查没有持久事实变化时，不更新时间；历史指标和原始 acceptance/result 文件不得为匹配新口径而改写。
