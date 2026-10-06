---
name: subagent-dispatch
description: Decides when to delegate work to subagents in this project, which work stays with the main agent, and how to handle subagent failures. Use when a task involves large but low-difficulty reading, search, log or inventory work, or mechanical implementation after the approach is fixed.
---

# 子代理调度

## 优先委派

- 大量文件阅读与整理、仓库搜索、日志整理、批量检查、重复性代码调查、方案明确后的机械实现、token 消耗大但推理难度低的任务。
- 先由主代理定位候选范围，再给出简短执行单：目标、允许访问或修改的目录/文件、已知入口、约束与易错点、禁止事项、验收标准、最小验证方式、期望输出、停止条件。
- 默认先用一个子代理；只有互不依赖的交付物才增加到 2–3 个；不嵌套委派。
- 尊重现有子代理模型配置，不修改模型设置。

## 保留给主代理

- 科研方向判断、核心方法设计、重要实验裁决、异常结果解释、高风险项目操作、会显著影响项目结构或研究路线的决策、最终验收。
- 子代理的修改、验证结果与摘要都是待复核材料；主代理必须直接核对关键差异与证据后再下结论。

## 失败保护

- 如果主代理决定把任务交给子代理，但子代理启动失败、不可用、异常退出或无法正常工作：**不允许主代理自动接管原本准备交给子代理的大型任务。**
- 停止该部分并询问用户：**“子代理当前不可用，本任务原计划由子代理执行。是否改由主代理继续，还是先处理子代理问题？”**
- 报告已完成部分、阻塞原因、受影响文件与准确恢复点。
- 非常小的任务无需机械套用本规则。

## 验证边界

- 子代理只运行覆盖本次行为风险的最小定点检查，遵守项目验证预算；完整测试、全仓扫描、GPU、训练、长耗时评价、云端或付费操作未经用户批准不得执行。
