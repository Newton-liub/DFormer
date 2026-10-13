# 外部资源入口（论文库、索引、外部源码）

> 给 Agent 的短入口：只说明外部资源在哪、索引在哪、怎么读。按 2026-10-07 裁决，索引真源**不在本仓库**，这里不放第二份需要同步维护的副本。

## 位置

- **论文全文库**：`D:\0Project\origin\论文\`（canonical bundle；当前篇数以下方 JSON 为准）。
- **索引真源与维护工具**：`D:\0Project\origin\_index\`
  - `PAPER_LIBRARY_INDEX.json`（唯一权威机器索引，当前 `schema_version` 5）、`PAPER_LIBRARY_INDEX.md`（由 JSON 生成的只读视图）
  - `paper-index.md`（编号 ↔ 本地全文 ↔ 本地代码对照，历史快照）
  - `code-index.md`（外部 clone 清单）、`external_reference_provenance_2026-09-21.md`（许可证与再分发边界）
  - `pending-and-duplicates.md`（Pending 与 Duplicate Review 清单）
  - `tools\paper_library_store.py`（共享数据层，唯一索引读写实现）、`tools\paper_library.py` / `.cmd`（接入与索引生成，支持 `--dry-run`）
  - `tools\paper_workflow.py`（审批制工作流，七个子命令：`intake-plan` / `intake-apply` / `supplement-plan` / `supplement-apply` / `maintenance-plan` / `maintenance-apply` / `views-refresh`；plan 只写审核 JSON，apply 需审核文件 SHA-256，`views-refresh` 只读重建派生视图）
  - `tools\paper_library_ui.py` / `.cmd`（论文索引可视化工具，只在本机 `127.0.0.1` 运行）、`tools\requirements-ui.txt`
  - `exports\`（AI 阅读版 Markdown 的最新版本：全部 / 仅核心 / 仅已有补充三个固定文件）、`backups\`（`PAPER_LIBRARY_INDEX.json.bak1`–`.bak3`）
- **退库恢复目录**：`D:\0Project\origin\论文_retired\<LIB>\<原目录名>\`，在正式扫描根之外、同卷；只有明确授权的退库/恢复才移动其中内容。
- **外部源码 clone**：`D:\0Project\origin\<repo>\`，共 12 个仓库，清单与许可证见 `_index\code-index.md`。

## 状态分层（引用前必须区分）

- `论文\` 的 canonical 正式库可以作为已核验依据引用；当前活动篇数以 `PAPER_LIBRARY_INDEX.json` 为准（`status=retired` 的记录保留在索引中但不属于活动集合）。
- `论文_retired\<LIB>\` 是明确授权退库后的可恢复目录，其中的 bundle 不是活动论文，不得当作已核验论文引用，也不得在未授权时移动或删除。
- `论文待处理\`（9 项，Pending）与 `论文_duplicates_review\`（1 项，Duplicate Review）状态独立，**不得**在方向审计或方案设计里当作已核验论文使用。
- `origin\DFormer\` 这个 clone 已标 deprecated：停在 `814799b`，落后于作者最新且与工作副本重复。当前唯一 DFormer 工作基线是 `D:\0Project\DFormer\`。

## 怎么读

- 单个 bundle：正文 `<bundle>\<bundle>.md`；表格数值优先 `Tables\*.xlsx`；图用 `Figure\*.jpg`；公式优先 `Formula\*_formula.md`，并用公式图片核对。
- 查实体用 `PAPER_LIBRARY_INDEX.json`：`auto.*` 是扫描字段；`human.*`（人工编号、核心标记、标签、阅读状态、补充、外部确认书目含 `rating`）由人工维护，自动工具不得覆盖。
- 需要整篇论文的书目 + 摘要 + 补充给外部 AI 阅读时，用 `_index\tools\paper_library_ui.cmd` 的“AI 阅读导出”，或 `paper_workflow.py views-refresh`（结果写到 `_index\exports\`，只含最新版本与活动记录，不改索引）。
- 需要新增论文或更新索引时用 `_index\tools\paper_library.cmd`，先跑 `--dry-run` 看计划再执行；改索引必须经过 `_index\tools\paper_library_store.py`，不要直接手改 JSON。
- 入库、证据补充、书目/等级维护、退库与恢复的同轮授权方式与来源清理边界见 `paper-library.md`；通过 `/paper-intake`、`/paper-supplement` 和 `/paper-maintenance` 显式调用相应 Skill。用户明确授权时，可在同一会话完成受支持的 plan→apply；只要求预检时在 plan 后停止。真实计划 SHA-256、补充基线指纹和 store 并发校验仍必须按 CLI 执行。
- 指定书目字段（`year` / `venue` / `rating` / `doi`）修改与精确退库/恢复走 `paper_workflow.py maintenance-plan` / `maintenance-apply`，或索引 UI 的受支持字段编辑、差异确认与冲突检查；`doi` 的人工确认值优先于扫描值且不改动 `auto.doi`。常规扫描不代替人工维护，等级不写进导出文本冒充入库。恢复必须另有明确授权。
- 需要人工浏览和编辑编号、补充、书目与等级时启动 `_index\tools\paper_library_ui.cmd`（本机页面，保存前有差异确认与冲突校验）；只编辑用户明确指定的字段，已退库记录在页面里只读。
- 索引只在 `_index\` 维护；需要编号历史依据时去归档 `D:\0Project\DFormer-archive-20261007\MMFR\03_reference\`。
