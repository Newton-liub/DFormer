---
name: research-progress-report
description: Generates evidence-based DFormer research, development, experiment, data-processing, literature, handoff, and group-meeting progress reports. Use when the user asks for a 阶段总结、工作汇报、实验汇报、组会汇报、代码改动说明、项目交接、Markdown 报告或 Canvas 汇报。
---

# DFormer 阶段工作汇报

## 职责

把指定时间范围内的实际工作整理为可复核的 Markdown 正文，并在适合展示时生成版本化 Canvas。不要重新总结整个项目，也不要把讨论或计划写成已完成事项。

## 快速调用

用户可以只提供本次变化的信息：

- `汇报上次正式报告之后的全部进展，面向组会，生成 Markdown 和 Canvas。`
- `只汇报本周代码与实验，详细说明复现方法，不需要 Canvas。`
- `汇报指定提交区间，重点说明数据处理和当前风险。`

如果范围、对象和输出已经明确，不重复提问。

## 工作流

### 1. 确定报告边界

按以下优先级确定起点和终点：

1. 用户明确指定的日期、提交、对话或任务范围；
2. `doc/reports/report-index.json` 中最新正式报告的截止边界；
3. 当前对话中可明确识别的上一次正式报告；
4. 无法唯一确定时，向用户确认。

提示词讨论、Skill 设计、格式讨论和选型分析不视为正式报告。不得只凭“上次”“最近”等模糊表述猜测跨会话边界。

### 2. 选择证据

先阅读 [PROJECT_EVIDENCE.md](PROJECT_EVIDENCE.md)，只调查与本次范围有关的材料。不要默认扫描整个仓库。

每个可验证事实至少关联一种证据。实验指标必须同时说明配置、数据集或划分、checkpoint/日志来源和验证状态；缺失时明确标记。

### 3. 分类工作状态

所有主要事项归入且只归入以下一种状态：

- **已完成并验证**：实现完成且存在测试、运行、实验或人工核验结果；
- **已完成但尚未验证**：改动已落地，但没有充分验证结果；
- **正在进行**：已开始且仍有未完成步骤；
- **仅提出或计划实施**：只有讨论、方案或待办，尚未落地。

证据冲突时采用更保守的状态，并在“当前问题”中说明冲突。

### 4. 生成 Markdown 事实正文

按 [REPORT_FORMAT.md](REPORT_FORMAT.md) 选择必要章节，不输出空章节。每项主要工作尽量按以下逻辑组织：

`问题 -> 处理原因 -> 实际措施 -> 原理或取舍 -> 结果与证据 -> 后续事项`

默认将正式报告保存为：

`doc/reports/YYYY-MM-DD-<topic-slug>.md`

同名文件已存在时不要覆盖；增加更具体的主题后缀或向用户确认。

### 5. 按需生成 Canvas

满足任一条件时生成 Canvas：

- 用户明确要求 Canvas、可视化或组会展示版；
- 报告包含适合视觉布局的多项指标、时间线、状态分组、流程或风险对比。

生成前遵循 Cursor Canvas Skill，并满足本项目约束：

1. Markdown 是事实源，Canvas 只重排已核实内容，不引入新结论；
2. Git 管理的源文件保存到 `doc/canvases/`；
3. 文件名必须匹配 `MAJOR.MINOR.PATCH-<descriptive-name>.canvas.tsx`，例如 `0.0.2-weekly-progress.canvas.tsx`；
4. 版本号取 `doc/reports/report-index.json` 的 `nextCanvasVersion`，并与现有文件核对；
5. 每个新建或修订后的 Canvas 都使用新版本号，禁止覆盖或删除旧版本；
6. 标题或显著元数据中显示相同版本号；
7. 使用 `tools/publish-canvas.ps1` 发布到 Cursor 受管目录；
8. 发布后向用户提供受管 Canvas 的绝对路径链接。

版本号是全项目单调递增的保护标识，不表示模型或实验版本。常规情况下递增 PATCH；只有用户明确要求重置版本策略时才改变 MAJOR/MINOR。

### 6. 更新报告索引

只有正式报告或版本化 Canvas 已成功生成并完成必要验证后，才更新 `doc/reports/report-index.json`：

- 正式报告：记录覆盖边界、基准提交、产物路径和验证状态；
- Canvas：记录版本、名称、源文件、关联报告和创建日期；
- 将 `nextCanvasVersion` 更新为下一个未使用版本。

索引更新失败时，产物不得宣称已经正式登记。

### 7. 完成前检查

- 报告边界可追溯；
- 文件、函数、参数、命令和指标均来自证据；
- 四类状态没有混淆；
- 没有空章节、虚构统计或未说明来源的结论；
- Markdown 与 Canvas 内容一致；
- Canvas 文件名和页面内版本一致；
- 未覆盖或删除既有 Canvas；
- 报告索引 JSON 有效且版本号未重复；
- 最近修改文件通过可用的 lint、类型或脚本检查。

## 项目约束

- 不修改原始实验结果来配合叙述；
- 不把 README 中的官方论文结果当作本阶段本地实验结果；
- 不提交 checkpoint、数据集或大型运行产物；
- 不自动提交 Git；
- 用户只要求整理对话时，不额外展开全项目调查；
- 信息不足时使用“尚未验证”“当前证据未提供”或 `<待补充>`。