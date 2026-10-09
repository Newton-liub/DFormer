# 外部资源入口（论文库、索引、外部源码）

> 给 Agent 的短入口：只说明外部资源在哪、索引在哪、怎么读。按 2026-10-07 裁决，索引真源**不在本仓库**，这里不放第二份需要同步维护的副本。

## 位置

- **论文全文库**：`D:\0Project\origin\论文\`，45 篇 canonical bundle。
- **索引真源与维护工具**：`D:\0Project\origin\_index\`
  - `PAPER_LIBRARY_INDEX.json`（唯一权威机器索引，当前 `schema_version` 4）、`PAPER_LIBRARY_INDEX.md`（由 JSON 生成的只读视图）
  - `paper-index.md`（编号 ↔ 本地全文 ↔ 本地代码对照，历史快照）
  - `code-index.md`（外部 clone 清单）、`external_reference_provenance_2026-09-21.md`（许可证与再分发边界）
  - `pending-and-duplicates.md`（Pending 与 Duplicate Review 清单）
  - `tools\paper_library_store.py`（共享数据层，唯一索引读写实现）、`tools\paper_library.py` / `.cmd`（接入与索引生成，支持 `--dry-run`）
  - `tools\paper_workflow.py`（审批制入库与证据补充工作流，四个子命令；plan 只写审核 JSON，apply 需审核文件 SHA-256）
  - `tools\paper_library_ui.py` / `.cmd`（论文索引可视化工具，只在本机 `127.0.0.1` 运行）、`tools\requirements-ui.txt`
  - `exports\`（AI 阅读版 Markdown 的最新版本：全部 / 仅核心 / 仅已有补充三个固定文件）、`backups\`（`PAPER_LIBRARY_INDEX.json.bak1`–`.bak3`）
- **外部源码 clone**：`D:\0Project\origin\<repo>\`，共 12 个仓库，清单与许可证见 `_index\code-index.md`。

## 状态分层（引用前必须区分）

- `论文\` 的 45 篇是 canonical 正式库，可以作为已核验依据引用。
- `论文待处理\`（9 项，Pending）与 `论文_duplicates_review\`（1 项，Duplicate Review）状态独立，**不得**在方向审计或方案设计里当作已核验论文使用。
- `origin\DFormer\` 这个 clone 已标 deprecated：停在 `814799b`，落后于作者最新且与工作副本重复。当前唯一 DFormer 工作基线是 `D:\0Project\DFormer\`。

## 怎么读

- 单个 bundle：正文 `<bundle>\<bundle>.md`；表格数值优先 `Tables\*.xlsx`；图用 `Figure\*.jpg`；公式优先 `Formula\*_formula.md`，并用公式图片核对。
- 查实体用 `PAPER_LIBRARY_INDEX.json`：`auto.*` 是扫描字段；`human.*`（人工编号、核心标记、标签、阅读状态、补充、外部确认书目）由人工维护，自动工具不得覆盖。
- 需要整篇论文的书目 + 摘要 + 补充给外部 AI 阅读时，用 `_index\tools\paper_library_ui.cmd` 的“AI 阅读导出”（结果写到 `_index\exports\`，只含最新版本，不改索引）。
- 需要新增论文或更新索引时用 `_index\tools\paper_library.cmd`，先跑 `--dry-run` 看计划再执行；改索引必须经过 `_index\tools\paper_library_store.py`，不要直接手改 JSON。
- 需要接入新论文 bundle 时用 `/paper-intake`，需要为某一篇追加阅读证据时用 `/paper-supplement`（两者都是显式调用的项目 Skill，见 `.cursor/skills/`）。默认只产出审核计划并停止；获批后才执行 apply，索引写入仍只经共享 store。
- 需要人工浏览和编辑编号 / 补充时启动 `_index\tools\paper_library_ui.cmd`（本机页面，保存前有差异确认与冲突校验）。
- 索引只在 `_index\` 维护；需要编号历史依据时去归档 `D:\0Project\DFormer-archive-20261007\MMFR\03_reference\`。
