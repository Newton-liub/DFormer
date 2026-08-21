---
name: dformer-plan-execute
description: 使用此 Skill 时，先用复杂模型规划，再自动用简单模型执行并验证 DFormer 任务。用户只需说“使用 dformer-plan-execute：任务”。
---

# DFormer 自动规划执行

用户只需输入：

`使用 dformer-plan-execute：<任务>`

收到后自动完成以下流程，不要求用户分别调用代理：

1. 调用 `dformer-planner`，使用 `.cursor/agents/dformer-planner.md` 中配置的复杂模型，只读调查并生成计划。
2. 将计划直接交给 `dformer-executor`，使用配置的简单模型完成代码、配置、测试和必要实验。
3. 调用 `dformer-verifier` 独立验证实现和证据。
4. 汇总修改文件、验证结果、未完成事项和风险。

只有以下情况需要暂停并询问用户：任务目标无法判断、涉及破坏性资源操作、或计划需要超出用户授权范围的变更。普通代码任务不要在规划完成后再次询问确认。

## 代理限制

- 直接子代理最多 4 个；默认依次使用规划、执行、验证三个代理。
- 子代理不得创建子代理，不得递归委派。
- 多个代理修改同一文件时串行执行；并行编辑必须使用隔离 worktree。
- 不自动提交 Git。

## 任务规则

- 规划代理只读，输出目标、文件、步骤、验证命令和风险。
- 执行代理先核对计划，再分阶段实施并验证。
- 验证代理只读，独立检查实现、测试和实验依据。
- 不修改数据集、checkpoint 或原始实验结果来配合结论。
- 实验结论必须有配置、日志、命令或结果证据。

## 现有 Skill

涉及 CompShare GPU 云资源时继续遵守 `compshare-cli` 的 JSON、dry-run、授权和敏感信息规则。涉及阶段报告、Markdown 或 Canvas 时继续遵守 `research-progress-report` 的边界、证据和版本规则；更具体的现有 Skill 优先。

## 模型切换

实际模型写在三个代理文件的 `model` 字段中；`.dformer/agent-workflow.yaml` 是集中对照配置。更换模型时同步修改两处：

- 复杂模型：`dformer-planner.md` 的 `model`；
- 简单模型：`dformer-executor.md` 和 `dformer-verifier.md` 的 `model`。

当前默认值为规划 `gpt0.1`，执行和验证 `gpt-terra`。