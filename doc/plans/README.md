# MUSeg 阶段计划与历史执行记录

> **文档角色：计划目录索引，不承担实时状态。** 当前事实、正在进行事项、授权边界和恢复点只看 `doc/main/MUSeg-current-status.md`；研究选择及其处置状态只看 `doc/main/MUSeg-open-decisions.md`。

本目录保存 MUSeg 阶段设计、执行计划和历史记录。当前计划与已封存计划分开管理：

- **当前执行计划：** [`doc/plans/MUSeg-A2-B2深度有效性/00-总方向规划.md`](MUSeg-A2-B2深度有效性/00-总方向规划.md)。目标是在冻结的 RGB Quick-B0 上完成正式 A2，并仅在 `A2-pass + 用户确认` 后进入 B2 depth-validity gating 的规格、实现和零训练验证。
- **当前执行最小入口：** [`doc/plans/MUSeg-A2-B2深度有效性/01-新对话最小上下文与当前任务.md`](MUSeg-A2-B2深度有效性/01-新对话最小上下文与当前任务.md)；唯一当前执行方案为同目录 [`02-A2正式验证与B2条件分支.md`](MUSeg-A2-B2深度有效性/02-A2正式验证与B2条件分支.md)。
- **并行研究设计计划：** [`doc/plans/MUSeg-方向1最短验证路径/00-总方向规划.md`](MUSeg-方向1最短验证路径/00-总方向规划.md)。该计划围绕冻结输出后验校准设计问题—假设和方案—假设两条互补 MVE 路径；它是未授权的文档计划，不改变 A2/B2 当前执行恢复点。
- **方向1最小入口：** [`doc/plans/MUSeg-方向1最短验证路径/01-新对话最小上下文与当前任务.md`](MUSeg-方向1最短验证路径/01-新对话最小上下文与当前任务.md)；唯一双路径方案为同目录 [`02-问题与方案双路径-MVE.md`](MUSeg-方向1最短验证路径/02-问题与方案双路径-MVE.md)。
- **已封存计划：** [`doc/plans/archive/2026-08-MUSeg-DFormerv2快速Baseline/00-总方向规划.md`](archive/2026-08-MUSeg-DFormerv2快速Baseline/00-总方向规划.md) 和 [`doc/plans/archive/2026-08-MUSeg阶段二长程Baseline与MVE/00-总索引与执行门禁.md`](archive/2026-08-MUSeg阶段二长程Baseline与MVE/00-总索引与执行门禁.md)。归档只保留形成时点的设计、执行历史和证据指针，不再作为新的运行授权依据。
- **归档说明：** [`doc/plans/archive/README.md`](archive/README.md)。归档文件不替代实时状态入口。

A2/B2 小计划只有 3 个顺序任务：A2 协议与本地准备、另获授权后的正式 A2 与裁决、仅在 A2 通过并再次授权后的 B2 实现与零训练验证。B2 短训、完整独立训练、配对 screening、GPU、云资源和 official test 不由该计划自动授权。

方向1最短验证计划的两条路径共享数据职责、冻结 checkpoint、同源 logits 和证据门禁，但不共享未经冻结的数据、协议或结论；正式 logits 导出、校准器拟合、GPU、训练、云资源和 official test 均需单独授权。

任何计划与实时状态或开放决策冲突时，以 `doc/main/` 下两份入口文件为准，并在执行前停止确认。

新增或实质更新计划时，按 `doc/guides/README.md` 的文档状态头规范写明文档角色、形成/核验时点、实时入口和后继关系。新计划放入 `doc/plans/` 下的明确主题子目录，不再创建依赖时效判断的 `doc/临时`、`待执行` 等计划目录。
