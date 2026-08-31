# MUSeg 阶段计划与历史执行记录

> **文档角色：计划目录索引，不承担实时状态。** 当前事实、正在进行事项、授权边界和恢复点只看 `doc/main/MUSeg-current-status.md`；研究选择及其处置状态只看 `doc/main/MUSeg-open-decisions.md`。

本目录保存 MUSeg 阶段设计、执行计划和历史记录。当前计划与已封存计划分开管理：

- **当前计划：** `doc/plans/MUSeg-A2-B2深度有效性/00-总方向规划.md`。目标是在冻结的 RGB Quick-B0 上完成正式 A2，并仅在 `A2-pass + 用户确认` 后进入 B2 depth-validity gating 的规格、实现和零训练验证。
- **当前最小入口：** `doc/plans/MUSeg-A2-B2深度有效性/01-新对话最小上下文与当前任务.md`；唯一当前方案为同目录 `02-A2正式验证与B2条件分支.md`。
- **已封存计划：** `doc/plans/archive/2026-08-MUSeg-DFormerv2快速Baseline/00-总方向规划.md` 和 `doc/plans/archive/2026-08-MUSeg阶段二长程Baseline与MVE/00-总索引与执行门禁.md`。归档只保留形成时点的设计、执行历史和证据指针，不再作为新的运行授权依据。
- **归档说明：** `doc/plans/archive/README.md`。归档文件不替代实时状态入口。

当前 A2/B2 小计划只有 3 个顺序任务：A2 协议与本地准备、另获授权后的正式 A2 与裁决、仅在 A2 通过并再次授权后的 B2 实现与零训练验证。B2 短训、完整独立训练、配对 screening、GPU、云资源和 official test 不由该计划自动授权。

任何计划与实时状态或开放决策冲突时，以 `doc/main/` 下两份入口文件为准，并在执行前停止确认。

新增或实质更新计划时，按 `doc/guides/README.md` 的文档状态头规范写明文档角色、形成/核验时点、实时入口和后继关系。新计划放入 `doc/plans/` 下的明确主题子目录，不再创建依赖时效判断的 `doc/临时`、`待执行` 等计划目录。
