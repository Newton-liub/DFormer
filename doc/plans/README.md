# MUSeg 阶段计划与历史执行记录

> **文档角色：** 计划目录索引，不承担实时状态。
> **核验时点：** 2026-09-05。
> **实时入口：** `doc/main/MUSeg-current-status.md`；研究选择见 `doc/main/MUSeg-open-decisions.md`。
> **后继关系：** 计划只记录设计和形成时点；当前事实与恢复步骤以实时状态为准。

本目录保存 MUSeg 阶段设计、执行计划和历史记录。计划不单独构成代码、GPU、训练、云资源或 official test 授权。

## 未执行候选计划

- [`deferred/2026-09-MUSeg-unexecuted/README.md`](deferred/2026-09-MUSeg-unexecuted/README.md)：延期区总说明。
- [`deferred/2026-09-MUSeg-unexecuted/MUSeg-A2-B2深度有效性/00-总方向规划.md`](deferred/2026-09-MUSeg-unexecuted/MUSeg-A2-B2深度有效性/00-总方向规划.md)：A2/B2 深度有效性候选方向；未执行、未授权。
- [`deferred/2026-09-MUSeg-unexecuted/MUSeg-方向1最短验证路径/00-总方向规划.md`](deferred/2026-09-MUSeg-unexecuted/MUSeg-方向1最短验证路径/00-总方向规划.md)：后验校准与 Depth 退化双路径候选方向；未执行、未授权。

重新启用任一方向时，必须先读取实时状态和开放决策，从稳定基准建立独立研究分支，重新冻结数据、config、protocol、evaluator 和授权边界。延期计划中的“当前任务”“下一步”和“恢复点”只代表原形成时点的拟议流程。

## 已封存计划

- [`archive/README.md`](archive/README.md)：归档说明。
- [`archive/2026-08-MUSeg-DFormerv2快速Baseline/`](archive/2026-08-MUSeg-DFormerv2快速Baseline/)：已完成 RGB Quick-B0 阶段的历史设计和证据入口。
- [`archive/2026-08-MUSeg阶段二长程Baseline与MVE/`](archive/2026-08-MUSeg阶段二长程Baseline与MVE/)：Stage-01 至 Stage-08 及历史 MVE/门禁设计。

## 使用规则

- 当前事实、阻塞项、授权边界和恢复点只看 `doc/main/MUSeg-current-status.md`。
- 研究选择及其处置状态只看 `doc/main/MUSeg-open-decisions.md`。
- 历史计划中的指标、命令和下一步按其形成时点理解，不回写历史实验结果。
- 新增或实质更新计划时，按 `doc/guides/README.md` 的状态头规范记录角色、时点、实时入口和后继关系。
