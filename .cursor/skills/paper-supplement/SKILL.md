---
name: paper-supplement
description: Add evidence-based reading notes to one specified paper in the external library. Use when the user asks to 补充/补论文/记录某篇 LIB 的阅读证据, or to save a reviewed supplement note. Defaults to appending evidence and keeping the reading status; writing the index needs a separately approved plan.
disable-model-invocation: true
---

# 论文证据补充（paper-supplement）

为外部论文库中的**一篇指定论文**追加阅读证据笔记。默认只出草稿与审核计划；索引写入必须另有一次明确批准。

## 先读

1. 本 Skill 所在仓库的 `doc/state/current.md`。
2. 共享 Rule `.cursor/rules/paper-library.mdc`（人工字段保护、审批与冲突停止、证据类型区分）。
3. 需要字段约定时读 `D:\0Project\origin\_index\README.md`。

## 定位与只读定位模式

```text
supplement-plan --id <LIB0000xx 或已确认人工编号>
```

只读输出 F0（索引指纹）、LIB、人工编号、标题、正文路径与正文 hash、已有补充长度、`source_note`、`sources` 条数、阅读状态。此模式不写任何文件。

- 零匹配、多归属、编号互相矛盾即停止；未核验历史编号只作线索。
- 只读指定正文、必要图表/表格/公式，以及获准的官方实现；未实际打开的文件不算已读。

## 草稿模板（四个短区块）

```markdown
## 阅读范围与未读范围
- 已读：<正文路径 + 章节/图/表/公式>
- 未读：<明确列出>

## 论文作者事实及来源
- <结论>（来源：<正文路径 + 章节/页/图/表/公式>）

## 官方实现事实及来源
- <结论>（来源：<URL 或本地路径>，<文件/函数/行范围>，commit <hash>）

## 模型判断、与当前研究的关系及不确定项
- <判断与理由>；不确定项：<...>
```

结论分别使用“未阅读”“已读范围未报告”“尚未核实”，不把局部缺项扩大成全文未报告；不虚补事实。科研判断由上级模型或用户负责。

## 命令

工具：`D:\2Env\anaconda\python.exe D:\0Project\origin\_index\tools\paper_workflow.py`

```text
# 1) 定位（只读，得到 F0）
supplement-plan --id <LIB 或人工编号>

# 2) 生成审核计划（读取期间的索引/证据变化会被拒绝）
supplement-plan --id <...> --draft <草稿 md> --expected-fingerprint <F0> --plan <审核计划 JSON> \
  [--evidence <实际读过的本地文件> ...] [--scope "<阅读范围标签>"] \
  [--reading-status <abstract_only|needs_full_text|full_text_completed>]

# 3) 获批后保存
supplement-apply --plan <审核计划 JSON> --approved-plan-sha256 <该文件的 SHA-256>
```

隔离或非默认索引时追加 `--index <JSON>` 与配套 `--root <资产根>`（必须与索引的 `library_root` 一致）。

计算批准指纹：

```powershell
(Get-FileHash -Algorithm SHA256 -LiteralPath "<审核计划 JSON>").Hash.ToLower()
```

## 流程

1. 定位唯一 LIB，完整读取已有 `human.supplement`、阅读状态与本篇书目，保留旧正文、旧来源、旧限定语。
2. 只读实际需要的正文与证据；对正文与将引用的本地证据记录 hash（用 `--evidence` 逐个传入实际打开的文件）。
3. 写草稿，运行 `supplement-plan` 生成计划。默认在旧 `content_md` 后追加一段带范围标记的补充，旧文本作为完整前缀保留；旧 `sources` 与其他扩展键保留。
4. **送审并停止**：提交草稿、计划路径、字段差异（内容长度、来源说明、阅读状态是否变化）。追加内容已存在即 no-op，不重复追加。
5. 获批后运行 `supplement-apply`。它校验计划 hash、F0、草稿与正文/证据 hash，只把批准对象补入从 F0 加载的文档，并按白名单核对后保存一次。
6. 报告保存后的实际指纹、阅读状态与验收摘要。

再次审批、再次保存都要在新消息里重新显式调用本 Skill。

## 阅读状态

- 只使用现有枚举 `abstract_only` / `needs_full_text` / `full_text_completed`；默认 keep。
- 局部阅读不自动完成全文。`full_text_completed` 需要明确的全文范围完成记录（`--full-text-scope`）并单独批准；笔记非空或读了方法/实验都不满足条件。
- 不新增 `partial_read` 之类的状态。

## 停止条件

定位不唯一、指定附件不可读、旧笔记冲突、审批缺失或指纹不符、正文/草稿/证据/索引在审批后变化、越权字段差异、保存失败。证据缺项可以明示后交审，不虚补。若索引 JSON 已保存而派生 Markdown 重建失败，按「JSON 已提交、视图失败」报告：不重跑 `supplement-apply`、不二次保存，管理视图另行单独重建。

## 不做

不覆盖人工编号/标签/核心标记/书目 override，不重写论文资产，不改标题格式，不调用模型 API，不生成科研结论，不执行 Git 提交或推送。
