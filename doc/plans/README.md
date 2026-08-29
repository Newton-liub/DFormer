# MUSeg 阶段计划与历史执行记录

> **文档角色：计划目录索引，不承担实时状态。** 当前事实、正在进行事项、授权边界和恢复点只看 `doc/main/MUSeg-current-status.md`；研究选择及其处置状态只看 `doc/main/MUSeg-open-decisions.md`。

本目录保存 MUSeg 阶段设计、执行计划和历史记录。当前计划与已封存计划分开管理：

- **当前计划：** `doc/plans/MUSeg-DFormerv2快速Baseline/00-总方向规划.md`。目标是快速使用 DFormerv2 跑通 MUSeg，并建立接近 DFormerv2 官方论文口径、可作为后续模块对照的 baseline。
- **已封存计划：** `doc/plans/archive/2026-08-MUSeg阶段二长程Baseline与MVE/00-总索引与执行门禁.md`。该计划的 Stage-01 至 Stage-05 进度、MVE、Protocol Gate、历史 checkpoint 和相关执行证据保留在归档目录，不再作为新的运行授权依据。
- **归档说明：** `doc/plans/archive/README.md`。归档文件只承担历史设计和证据导航，不替代实时状态入口。

当前计划先只保留总方向和阶段边界；训练细节、评估器实现、预算、seed、云资源和具体运行命令在方向确认后分别建立子计划。任何计划与实时状态或开放决策冲突时，以 `doc/main/` 下两份入口文件为准，并在执行前停止确认。

新增或实质更新计划时，按 `doc/guides/README.md` 的文档状态头规范写明文档角色、形成/核验时点、实时入口和后继关系。新计划放入 `doc/plans/` 下的明确主题子目录，不再创建依赖时效判断的 `doc/临时`、`待执行` 等计划目录。
