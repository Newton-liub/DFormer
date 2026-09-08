# MUSeg 阶段计划与历史执行记录

> **文档角色：** 计划目录索引，不承担实时状态。
> **核验时点：** 2026-09-05。
> **实时入口：** `doc/main/MUSeg-current-status.md`；研究选择见 `doc/main/MUSeg-open-decisions.md`。
> **后继关系：** 计划只记录设计和形成时点；当前事实与恢复步骤以实时状态为准。

本目录保存 MUSeg 阶段设计、执行计划和历史记录。计划不单独构成代码、GPU、训练、云资源或 official test 授权。

## 未执行候选计划

- [`2026-09-MUSeg-几何可信RGBD双路径MVE/00-总方向规划.md`](2026-09-MUSeg-几何可信RGBD双路径MVE/00-总方向规划.md)：几何可信 RGB-D 问题/方案双路径 MVE 总规划；文献与项目事实门禁已完成，代码和实验未授权。
- [`2026-09-MUSeg-几何可信RGBD双路径MVE/03-共享协议与DVC-A1问题验证.md`](2026-09-MUSeg-几何可信RGBD双路径MVE/03-共享协议与DVC-A1问题验证.md)：当前待用户审批的详细问题验证子计划，合并共享 protocol、最小实现门禁与开发评价。
- [`2026-09-MUSeg-几何可信RGBD双路径MVE/04-DVG-B1条件式Oracle门控.md`](2026-09-MUSeg-几何可信RGBD双路径MVE/04-DVG-B1条件式Oracle门控.md)：仅在 `DVC-A1=supported` 且用户再次批准后才细化的 Oracle GSA 门控方向。
- [`2026-09-MUSeg-几何可信RGBD双路径MVE/参考资料/00-待补充论文内容清单.md`](2026-09-MUSeg-几何可信RGBD双路径MVE/参考资料/00-待补充论文内容清单.md)：已完成的关键论文方法、实验片段和收缩裁决。
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
