---
name: project-state
description: Maintains the single live state file doc/state/current.md for this DFormer research repository. Use in DFormer project work (research, source code, data, experiments, cloud runs, reports, releases) to read the current stage, confirmed results, blockers and recovery point, and to update it when facts change. Do not use for unrelated questions.
---

# 项目实时状态维护

## 何时使用

- 涉及本仓库研究、源码、数据、实验、云资源、报告或发布的任务：开始先读 `doc/state/current.md`。
- 只有发生持久事实变化时才更新；纯问答、解释和只读检查不更新。

## 读什么

- `doc/state/current.md`：当前阶段、已确认事实、下一步、阻塞、授权边界、恢复点。
- 需要目录与产物约定时读 `doc/guides/project.md`；涉及云端时读 `doc/guides/cloud.md`。
- 不把 `doc/plans/` 与 `doc/reports/` 当状态，它们是详细材料。

## 写什么

- 就地改写，保持简短：阶段、确认结果、当前任务、下一步、阻塞、边界、必要 commit/run 信息。
- 证据用链接指向 `doc/reports/` 或产物路径，不复制报告正文。
- 只写直接核验过的事实；未核验内容标为“待核验”，不按计划或预计时间补造结论。
- 需要更新的典型事件：实验开始或结束、验收结论、指标或 checkpoint 身份、official-test 状态、云实例状态、证据位置、阻塞与恢复点、提交或推送状态。

## 归档

- 目标约 100 行；超过约 120 行或一个大阶段结束时，把有追溯价值的旧内容整段移到 `doc/archive/<日期-事件>/`，然后重写当前摘要。
- 小修复、单次启动失败、单个 hash 更新不单独建档。
- 不为此建立后台服务、Hook 或归档脚本。

## 责任

- 状态正文与研究结论由主代理写入；子代理只能提供草稿与证据。
- 每个涉及本项目的对话在最终答复前更新，并在答复中说明是否更新及原因。
