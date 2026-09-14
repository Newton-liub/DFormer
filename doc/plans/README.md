# MUSeg 阶段计划与历史执行记录

> **文档角色：** 计划目录索引，不承担实时状态。
> **核验时点：** 2026-09-14。
> **实时入口：** `doc/main/MUSeg-current-status.md`；研究选择见 `doc/main/MUSeg-open-decisions.md`。
> **后继关系：** 计划只记录设计和形成时点；当前事实与恢复步骤以实时状态为准。

本目录保存 MUSeg 阶段设计、执行计划和历史记录。计划不单独构成代码、GPU、训练、云资源或 official test 授权。

## 当前与候选计划

- [`2026-09-MUSeg-多形式模态失效可靠性学习/00-总方向规划.md`](2026-09-MUSeg-多形式模态失效可靠性学习/00-总方向规划.md)：当前 MMFR（多模态失效鲁棒性）方向；A1 standalone 脚手架与 A2 post-crop/pre-GPU Depth corruption、确定性 RNG、辅助 reliability loss 和公平 clean/corruption config 均已完成 CPU qualification。GPU preflight、训练、完整评价、云执行和 official test 均未授权。
- [`2026-09-MUSeg-多形式模态失效可靠性学习/01-新对话最小上下文与当前任务.md`](2026-09-MUSeg-多形式模态失效可靠性学习/01-新对话最小上下文与当前任务.md)：新对话恢复入口；当前没有已授权执行任务，下一拟议步骤是单独批准本地 GPU 单步 preflight。
- [`2026-09-MUSeg-多形式模态失效可靠性学习/03-MMFR-A2训练接入与公平对照协议.md`](2026-09-MUSeg-多形式模态失效可靠性学习/03-MMFR-A2训练接入与公平对照协议.md)：冻结 A2 数据流、RNG、raw/normalized 职责、Depth-only curriculum、$\lambda_{rel}=0.1$、clean control、评价条件和成功门槛，并记录限定代码接入与 CPU qualification。

- [`2026-09-MUSeg-几何可信RGBD双路径MVE/00-总方向规划.md`](2026-09-MUSeg-几何可信RGBD双路径MVE/00-总方向规划.md)：几何可信 RGB-D 问题/方案双路径 MVE 总规划；v1/v2 历史终态保持不变，v3 为 `not-supported`，DVG-B1 的 P0–P4 已完成并裁决为 `oracle-not-supported`，当前回到方向级候选选择。
- [`2026-09-MUSeg-几何可信RGBD双路径MVE/03-共享协议与DVC-A1问题验证.md`](2026-09-MUSeg-几何可信RGBD双路径MVE/03-共享协议与DVC-A1问题验证.md)：保留 v1/v2 的 `protocol-blocked` 记录；新增独立 `DVC-A1-valdev-boundary-zero-v3-bgcontext`，记录完整评价和 `not-supported` 正式收口。
- [`2026-09-MUSeg-几何可信RGBD双路径MVE/04-DVG-B1条件式Oracle门控.md`](2026-09-MUSeg-几何可信RGBD双路径MVE/04-DVG-B1条件式Oracle门控.md)：A/B/C 与 P0 保持冻结，P1–P4 已完成；218 张图完整本地 GPU 配对开发评价的主条件 Boundary IoU 与 mIoU 均下降，最终为 `oracle-not-supported`。训练、云执行和 official test 未授权。
- [`2026-09-MUSeg-几何可信RGBD双路径MVE/参考资料/00-待补充论文内容清单.md`](2026-09-MUSeg-几何可信RGBD双路径MVE/参考资料/00-待补充论文内容清单.md)：已完成的关键论文方法、实验片段和收缩裁决。
- [`deferred/2026-09-MUSeg-unexecuted/README.md`](deferred/2026-09-MUSeg-unexecuted/README.md)：延期区总说明。
- [`deferred/2026-09-MUSeg-unexecuted/MUSeg-A2-B2深度有效性/00-总方向规划.md`](deferred/2026-09-MUSeg-unexecuted/MUSeg-A2-B2深度有效性/00-总方向规划.md)：A2/B2 深度有效性候选方向；未执行、未授权。
- [`deferred/2026-09-MUSeg-unexecuted/MUSeg-方向1最短验证路径/00-总方向规划.md`](deferred/2026-09-MUSeg-unexecuted/MUSeg-方向1最短验证路径/00-总方向规划.md)：后验校准与 Depth 退化双路径候选方向；未执行、未授权。

延期目录中的任一方向重新启用时，必须先读取实时状态和开放决策，从稳定基准建立独立研究分支，重新冻结数据、config、protocol、evaluator 和授权边界。延期计划中的“当前任务”“下一步”和“恢复点”只代表原形成时点的拟议流程。

## 已封存计划

- [`archive/README.md`](archive/README.md)：归档说明。
- [`archive/2026-08-MUSeg-DFormerv2快速Baseline/`](archive/2026-08-MUSeg-DFormerv2快速Baseline/)：已完成 RGB Quick-B0 阶段的历史设计和证据入口。
- [`archive/2026-08-MUSeg阶段二长程Baseline与MVE/`](archive/2026-08-MUSeg阶段二长程Baseline与MVE/)：Stage-01 至 Stage-08 及历史 MVE/门禁设计。

## 使用规则

- 当前事实、阻塞项、授权边界和恢复点只看 `doc/main/MUSeg-current-status.md`。
- 研究选择及其处置状态只看 `doc/main/MUSeg-open-decisions.md`。
- 历史计划中的指标、命令和下一步按其形成时点理解，不回写历史实验结果。
- 新增或实质更新计划时，按 `doc/guides/README.md` 的状态头规范记录角色、时点、实时入口和后继关系。
