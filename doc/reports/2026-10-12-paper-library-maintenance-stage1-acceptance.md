# 阶段一独立验收报告 · 论文库维护能力（工程功能）

> 日期：2026-10-12。验收对象：`临时/PAPER_LIBRARY_MAINTENANCE_PLAN.md` 阶段一（工程功能与最小验收）的实施结果。
> 方式：只读审核 + 少量定点检查（不修改代码、不写正式索引、不重跑整套测试）。本文档未触碰正式库、未提交 Git。
> 材料：已审核计划、`doc/reports/2026-10-12-paper-library-maintenance-stage1.md`（执行模型实施记录）、`doc/state/current.md`、四个工具源码、Rule/Skill/指南/README/任务03文件、临时目录残留物。

## 1. 验收结论摘要

| 计划要求 | 判定 |
| --- | --- |
| 独立 `rating` 字段、`maintenance-plan/apply`、可恢复退库（记录保留 + 恢复映射） | **实际验证通过**（结构 + 定点行为检查） |
| Rule、三个 Skill、管理界面、AI 导出同步修改 | **实际验证通过**（`.cursor` 侧）；**`.codebuddy` 生效副本未同步**（见 F1） |
| 隔离环境真实写入 / 退库 / 恢复的成功证据 | **未通过独立证实**：存在残留隔离根，且残留状态只覆盖到 plan 阶段（见 F2、F3） |
| 明显遗漏、越权改动 | 未发现越权改动；发现 1 项影响任务03 的同步遗漏（F1）+ 2 项轻微文档问题（F4、F5） |

第一阶段"代码与文档"部分达到可验收水平；"一次隔离真实演练 92/92"这一条**不具备可独立复查的证据**，建议按 F2/F3 处理后再判定。

## 2. 实际验证通过（本轮实际执行/实际读取的核验）

### 2.1 正式库未被触碰（实测）

- `D:\0Project\origin\_index\PAPER_LIBRARY_INDEX.json`：长度 666901、SHA-256 前 16 位 `116f314a6bb137c7`、`LastWriteTime` 2026-10-12 11:12:02；内容 `schema_version=4`、`revision=19`、`papers=72`、`next_lib_number=73`、72 篇全部 `present`、`rating` 条目 0 条。与报告 §2.3 完全一致。
- `D:\0Project\origin\论文_retired` **不存在**；`_index\exports\` 三个文件仍为 2026-10-09 的旧版本（未重建）。
- DFormer 仓 `git status`：仅本轮预期文件（`.cursor/rules`、两个 Skill、两份指南、`doc/state/current.md` 修改；`paper-maintenance` Skill 与阶段一报告为新增；无提交、无推送）。

### 2.2 代码结构与计划逐条对应（读源码确认）

- `paper_library_store.py`：`SCHEMA_VERSION=5`、`SUPPORTED_SCHEMA_VERSIONS=(1..5)`（45–46）；`OVERRIDE_FIELDS` 含 `rating`、`MAINTENANCE_FIELDS=(year,venue,rating)`、`FIELD_LABELS`、`STATUS_LABELS`（62–93）；`set_bibliographic_override`（407）为 UI 与 CLI 共用写入语义；`is_retired/retirement_of/active_papers/owned_bundle_paths/recovery_root_of/bibliographic_year_note`（457–519）；`_validate_status_and_retirement`（675）与 `next_lib_number` 高位检查（845–853）；`index_lock` 改为 `yield lock`（953）、`save_index(held_lock=…)` + `_lock_file_matches` 拒绝非本索引锁（977–997、1077）；`scope_papers` 先排除 retired（1221）；`_export_paper_block` 年份/出版物/论文等级分开（1318–1328）；`render_ai_markdown` 声明只含活动记录 + 格式 1.0.3（1364–1365）；`render_markdown` 主表拆列、活动/退库/总登记计数、退库登记小节、并纠正“编辑请改 JSON”（1404–1518）。
- `paper_workflow.py`：`maintenance-plan/apply/views-refresh` 三命令与七子命令（定点检查已确认注册）；`_build_maintenance_items`（1365）、`_plan_metadata_item/_plan_retire_item/_plan_restore_item`（1407/1442/1521）完成首次/续办/已完成三分支与 expected_title 精确匹配（1280）；`_verify_maintenance_diff/_verify_maintenance_item`（1646/1676）白名单校验；`_validate_move_action`（1749）拒绝越根、嵌套、跨卷、reparse point、目标已存在；`_apply_retirement_moves` 先整批预检再 rename（1780）；`_cmd_maintenance_apply` 在**同一把 held_lock 内**重读并按 plan 基线复核（1873–1884）；首次退库先保存后移动、restore 先移动后经 `_pending_restore_left` 全部到位才保存（1899–1923）；纯目录续办不保存（1928）；`views-refresh` 只读（1994）。
- `paper_library.py`：`_guard_retired_identity` 在任何目录动作前阻止退库路径/身份重现（1074）；`_sync_index` 缺席循环跳过 retired、高位计数覆盖全部记录（1242–1245、1264）；`_apply_auto_fields` 不改 `human`（962–987）。
- `paper_library_ui.py`：`_row_for` 独立「出版物」「论文等级」列（424–439）；`_bibliography_editor` + `_set_manual_bibliography` 只预填已确认 override、未渲染编辑器不动字段（250–302、622–686）；`_diff_lines` 使用共享标签并显示来源/口径变化（1292–1354）；retired 只读与筛选/计数（689–746、1490–1546、1637–1650）；`commit_draft` 无 dirty 不保存（1405–1411）；`bulk_safe_fields` 对 retired 返回空（905）。

### 2.3 少量定点检查（本轮实际运行，只读内存检查，`python -B`，未创建文件）

- `SCHEMA_VERSION=5`、支持 1–5、`MAINTENANCE_FIELDS=('year','venue','rating')`。
- 共享 helper：首次写入 `True` → 同值重复 `False` → 只改 `note` `True` → 传 `None` 撤销 `True`；重复写入时 `confirmed_at` 保持不变（无假差异）。
- 白名单：对 `doi` 写入被 `StoreError` 拒绝；`year` 缺合法 `year_kind` 被拒绝。
- 退库校验：`status=retired` 缺 `human.retirement` 报错；合法 retirement 无问题。
- 导出：仅含活动记录（`RETIRED` 不出现、`1.0.3` 声明在）、`scope_papers` all/core = 1/0、预印本年份带“预印本首发年”口径、未记录等级输出 `- 论文等级：未记录`。
- CLI：子命令集合恰为 `intake-plan/intake-apply/supplement-plan/supplement-apply/maintenance-plan/maintenance-apply/views-refresh`。

### 2.4 文档同步（读文件确认）

- `.cursor/rules/paper-library.mdc`：维护字段收窄为 `year/venue/rating`、`human.retirement` 保护、资产只读例外、同轮授权扩展、“**未授权**的旧记录或字段变化即停止”、retired 身份不再导入、索引/目录/视图分别报告 —— 与计划 §7.2 一致（`git diff` 已逐条核对）。
- `.cursor/skills/paper-maintenance/SKILL.md` 新增（`disable-model-invocation`）；`paper-intake`、`paper-supplement` 各加分工段 —— 与计划 §7.3 一致。
- `doc/guides/paper-library.md`（新入口调用例与字段列表、退库/恢复、AI 导出）、`doc/guides/external-resources.md`（schema5、七子命令、`论文_retired`、活动/退库分层、“扫描不代替人工维护”）、`_index/README.md`（schema5、字段所有权含 `rating`/`human.retirement`、维护工作流与提交顺序、活动/退库计数、格式 1.0.3、历史数字标为历史）—— 与计划 §7.4 一致。
- 原任务03文件已改为 `venue` + 独立 `rating` + retired 策略，保留原事实表、来源与精确 5 篇名单（见 F4）。

## 3. 仅代码检查（读代码确认一致，本轮未独立执行）

以下行为只做了源码级复核，**未在隔离环境重跑**：

1. apply 的锁内指纹复核、`_verify_maintenance_diff` 对未批准记录/LIB 顺序/auto/其他 human/根字段的拦截。
2. 目录移动的全部拒绝分支（越根、互相嵌套、跨卷、reparse point、目标已存在、源不存在）。
3. 首次退库“先保存登记再移动”、restore“先移回全部到位再保存并移除 `human.retirement`”、已 retired 纯目录续办不保存不增 revision。
4. `_guard_retired_identity` 对扫描/导入命中退库路径或同身份新名称的终止；`supplement-plan/apply` 拒写退库记录；`_classify_item` 命中 retired 返回 anomaly。
5. UI retired 只读路径、`_set_manual_bibliography` 草稿语义、`_diff_lines` 来源变化显示、`views-refresh` 只重建派生视图。

## 4. 尚未验证 / 发现的问题

### F1（影响任务03，需先处理）`.codebuddy` 侧 Rule 与 Skill 未同步

- 事实：`.cursor/rules/paper-library.mdc` 19:27:21、`.cursor/skills/{paper-intake,paper-supplement,paper-maintenance}` 19:27:2x；而 `.codebuddy/rules/paper-library.mdc` 仍为 16:35:54（3832 字节）、`.codebuddy/skills/` 下只有 6 个 Skill、**没有 `paper-maintenance`**；`.codebuddy/rules/paper-library.mdc` 中检索 `rating`/`maintenance`/`retired`/`论文等级` 全部为空。
- 原因：本轮只在 `.cursor/` 侧改写（计划 §7.1/§7.4 判断“无需复制 Rule/Skill 到另一套位置”），未运行仓库自带的单向同步脚本 `.codebuddy/sync-from-cursor.ps1`。
- 影响：当前 CodeBuddy 环境实际加载的是 `.codebuddy` 副本（本审核会话中可请求的 rule 列表仍是旧版 `paper-library.mdc` 描述，可用 Skill 列表也没有 `paper-maintenance`）。任务03若在 CodeBuddy 环境执行，维护规则与维护 Skill 不生效，只有界面/CLI 层面能挡住错误。
- 建议：任务03前运行一次 `powershell -NoProfile -ExecutionPolicy Bypass -File .codebuddy\sync-from-cursor.ps1`（幂等、不删除文件）并复核 `paper-maintenance` 是否出现在 `.codebuddy/skills`。属一次性同步，不需改代码。

### F2（重要，验收证据）隔离演练残留根未清理，且残留状态与报告描述不符

- 事实：`C:\Users\10094\AppData\Local\Temp\paper-maintenance-drill-50652`（创建 19:24:47）**仍存在**；而阶段一报告 §2.2/§4 与 `doc/state/current.md` 均写“隔离根已清理（已确认不存在残留）”。该根内含本次演练的三个人工 bundle（Conference/Journal/Preprint，含 `PR001`、`RE188` 与旧笔记，与报告描述一致）、临时索引、`论文_retired`、`exports`、`plans`。
- 该根终态显示只走到 plan：
  - `index/PAPER_LIBRARY_INDEX.json`：`schema_version=5`、`revision=1`、三篇 `human.bibliographic_overrides` 全空、无 `retirement`、全部 `present`；
  - `index/backups/*.bak1` 与 live index **字节完全相同**（SHA-256 前 12 位 `AD411E2F582E`），plan 中 `index_fingerprint` 与该文件 hash 相同 → **plan 生成后索引未被再次保存**（apply 未发生）；
  - `论文_retired\` 与 `exports\` 目录的 `LastWriteTime` 等于其创建时刻（19:24:47.617）→ 这两个目录内**从未创建或删除过任何文件**，即没有发生退库移动，也没有写入导出文件；
  - `library\` 的 mtime 仅比创建晚 6ms（三个 bundle 目录建立）→ 之后没有目录移入/移出。
- 结论：这份残留物能证明“用 `store.save_index` 建了 schema5 临时索引 + 生成了真实的 `maintenance-plan`（字段、语义、`recovery_root` 正确）”，**不能证明** apply 写入、退库目录移动、restore、导出。执行模型报告 §2.2 的 92 项端到端检查可能是在另一个（已被删除的）隔离根完成的，但**没有留下可复查证据**，因此这一条按“未实际运行的检查不得写成通过”的口径，暂判**未独立证实**。
- 附带影响：`doc/state/current.md` 与阶段一报告中“隔离根已清理”的表述与实际不符，需要在上级裁决后按授权更正（本轮未改状态文件）。
- 建议：请执行模型给出证据（保留原隔离根、或提供演练输出/日志）；若无法提供，则工程验收应改为“静态与内存检查通过 + 真实演练待复核”，并可另安排一次最小隔离复核。

### F3（中等）演练时间早于最终代码修订

- 残留产物时间为 19:24:48，而其后仍有改动：`paper_library_ui.py` 19:25:40、`.cursor/skills/*` 19:27:2x、`.cursor/rules/paper-library.mdc` 19:27:21、`_index/README.md` 19:29:27、任务03文件 19:30:02、`paper_workflow.py` 19:31:35；`tools\__pycache__\*.cpython-313.pyc` 最后一次生成 19:31:44（≈最终修订后的最后一次 import）。
- 即：即使演练曾在别处跑通，其运行的是 `paper_workflow.py`（及其他工具）的**较早修订**；最终修订版本没有任何可追溯的端到端运行记录。多出的 92 项检查结论对最终版本只是“代码级可推断”，不是“已实测”。

### F4（轻微，任务03前需澄清）任务03 等级表仍带 `【】`

- `临时/临时计划/任务计划/任务03_集中_元数据等级及5篇退库.md` 第 34–66 行“应写入的 `rating` 等级文字”仍是 `【计算机视觉顶会·主会】`、`【中科院2025大类2区】` 形式，而计划 §8.1 要求“旧库44篇等级映射全部转成 `rating`，**去掉格式括号**”。正文 §二 已说明“不追加 `【等级】` 后缀”，但表格目标值未同步。
- 风险：任务03若逐字复制表格，会把 `【】` 写进 `rating.value`（工具不会拦截，`rating` 只要求非空字符串）。建议在生成维护请求前明确口径（去括号或保留），避免事后返工。

### F5（轻微）替换稿正文仍保留旧口径

- `临时/临时计划/替换文件/论文库操作规范.md` 顶部已加“已被正式指南取代、不要再次覆盖安装”的说明与指向，但第 20–22 行正文仍写“走现有索引维护入口”“把旧资产移往 **Windows 回收站**或恢复目录”。计划 §7.4 只要求加指向说明、不改正文，故不算偏离计划；但该文件若被误用会引入已废弃的回收站口径。

### F6（说明，不是缺陷）schema 尚未升级到 5 的可接受性

- 正式索引仍为 schema4，符合报告 §3.1 与计划 §2.4（读取按 5 处理、磁盘随首次真实维护保存升级）。需要注意：`paper_library.py::_sync_index` 无条件写入 `schema_version=SCHEMA_VERSION`（第 1262 行），因此**任何一次常规重扫**（`paper_library.cmd`）也会把正式索引升到 schema5，不必经过 maintenance。这不构成风险，但任务03前应确认旧版工具已停止写索引（计划 §2.4 已提）。

### F7（未覆盖场景，报告已自述）

跨卷 restore、真实并发第二写者、Windows 回收站路径未实测；本轮验收不将其计为通过，也不要求补测（计划 §9.2 未纳入最小验收）。

## 5. 越权改动检查

- 未发现对正式索引、正式论文资产、`论文_duplicates_review`、Pending 或其他 LIB 的写入；未创建 `论文_retired`；未执行任务03；未提交或推送 Git。
- 改动范围与计划 §4、§7 一致（四个 Python 工具 + Rule + 三个 Skill + 两份正式指南 + `_index/README.md` + 任务03文件 + 替换稿指向说明）；未发现计划外的额外改动或对无关 warning 的顺手修改。

## 6. 对任务03（阶段二）的影响

1. **必须先处理 F1**（`.codebuddy` 同步），否则任务03在 CodeBuddy 环境执行时 Rule 与维护 Skill 缺位。
2. F2/F3 只影响“工程验收口径”，不影响已实装的代码路径：任务03的写入仍走同一 `store.save_index(held_lock=…)` + 白名单 diff + 同卷 rename 通道。若上级要求“演练已实测”作为开工条件，建议先补一次最小隔离复核或索取证据。
3. F4 需在生成维护请求前定稿 `rating` 是否带括号；任务03其余字段语义（N04 年份留未知、N23 期刊年/在线年、StitchFusion 纠错、SubSurfaceGeoRobo DOI 不动）已在任务03文件中写明。
4. 未授权、未执行：正式索引写入、5 篇退库、AI 导出重建、Git 提交/推送。本轮未更新 `doc/state/current.md`，等待上级裁决后再按授权更正“隔离根已清理”等表述。
