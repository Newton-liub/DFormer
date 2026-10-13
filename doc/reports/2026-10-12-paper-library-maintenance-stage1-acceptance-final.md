# 阶段一最终验收报告 · 最小隔离真实演练（最终版本代码）

> 日期：2026-10-12。执行方：独立验收模型（第二轮）。对象：`临时/PAPER_LIBRARY_MAINTENANCE_PLAN.md` 阶段一工程功能的**最终版本代码**。
> 隔离根：`%TEMP%\paper-acceptance-final`（本例自建；演练成功后已按授权清理，`paper-acceptance*` 残留数 0）。
> 未触碰正式库、未改代码、未提交 Git。本报告替代上一轮验收报告中“尚未验证/矛盾”的结论项（见 §4）。

## 1. 演练命令与结果（可复查）

所有命令均显式指定 `--index` / `--root` / `--recovery-root` / `--export-dir`，隔离根内自建临时索引（经 `store.save_index(expected_fingerprint="")`，revision 1、fingerprint `ac4f9786e854514e`）、3 个人工 bundle（Conference / Journal / Preprint）与恢复、导出目录。

| 步骤 | 命令（工具为 `_index\tools\paper_workflow.py`） | 关键输出 |
| --- | --- | --- |
| 1 metadata plan | `maintenance-plan --changes metadata-request.json --plan metadata-plan.json --recovery-root <iso>\论文_retired` | `Items: 3; index-writing items: 3; path actions: 0`；自动带出标题与 `— -> 2026 / CVPR 2026 / 计算机视觉顶会·主会` |
| 1 metadata apply | `maintenance-apply --plan metadata-plan.json --approved-plan-sha256 fe0b1a6d…` | `saved … (revision 2; backups: .bak1)`；`rebuilt …INDEX.md`；`exported …FOR_AI.md (3 篇) / CORE (1 篇) / SUPPLEMENTED (1 篇)` |
| 2 retire plan | `maintenance-plan --changes retire-request.json …`（LIB000001 + 完整题名 + reason） | `Items: 1; index-writing items: 1; path actions: 1`；`[retire] … 待移 1 个目录` |
| 2 retire apply | `maintenance-apply … c4a8172c…` | `moved: library\Acceptance Conference Paper One -> 论文_retired\LIB000001\Acceptance Conference Paper One`；`saved (revision 3; backups: .bak1,.bak2)`；导出降为 2 篇 |
| 3 restore plan/apply | `maintenance-plan/apply … restore-request.json`（同 LIB + 完整题名） | `待移回 1 个目录` → `moved: 论文_retired\LIB000001\… -> library\…`；`saved (revision 4)`；导出回到 3 篇 |
| 4 views-refresh | `views-refresh --export-dir <iso>\exports` | 重建 `INDEX.md` + 三份导出；`index unchanged by views-refresh: True` |
| 幂等 | 以最新基线重跑同一 metadata 请求 | `index-writing items: 0` / `Nothing to do … complete no-op`；apply 输出 `no index change: directory continuation only`，revision 仍 4、backups 不变 |

## 2. 关键状态（执行前后）

**metadata 后（revision 2，schema 5）**：三篇的年份/出版物/等级分别独立写入：

- LIB000001 `year=2026, year_kind=conference_year` / `venue=CVPR 2026` / `rating="计算机视觉顶会·主会"`（rating 键为 `value/source_type/confidence/confirmed_at/note`，与 venue 分离）
- LIB000002 `year=2021, issue_date` / `Journal of Acceptance Testing` / `rating="中科院2025大类2区（参考）"`
- LIB000003 `year=2024, preprint_date` / `arXiv (preprint)` / **无 rating 条目**；导出显示 `- 论文等级：未记录`，年份显示 `2024（预印本首发年，非正式发表年）`
- 未授权字段未变：LIB000002 保留 `manual_ids=["PR001"]`、`tags=["acceptance"]`、`is_core=true`、`reading_status=needs_full_text`、旧 supplement `旧笔记 acceptance note`；LIB000003 保留 `unverified_manual_ids=["RE188"]`；`next_lib_number=4` 不变

**retire 后（revision 3）**：LIB000001 `status=retired`；`human.retirement = {retired_at, reason, previous_status:"present", bundles:[{bundle_path, recovery_path}]}`；原目录消失、恢复目录出现且正文可读（111 字节）；LIB 与已批准书目保留；管理视图 `活动记录：2；已退库：1；总登记：3` 并列出退库登记与原路径→恢复路径；三份导出均不含该 LIB。

**restore 后（revision 4）**：`status=present`、`human.retirement` 已移除、目录回到原位置、正文可读；**恢复记录与该篇退库前（revision 2）逐字段相同（整体 `papers` 列表也完全相同）**；`next_lib_number=4`；管理视图回到 `活动记录：3；已退库：0；总登记：3`，退库小节显示“无。”；导出重新包含该 LIB（3 篇）。恢复根下仅遗留空的 `论文_retired\LIB000001\` 容器目录（工具按设计不自行清理恢复目录）。

**正式库**（本轮与上一轮一致，未被触碰）：`PAPER_LIBRARY_INDEX.json` 长度 666901、SHA-256 前 16 位 `116f314a6bb137c7`、mtime 2026-10-12 11:12:02、schema4/revision19/72 篇/`next_lib_number=73`；`论文_retired` 不存在；`_index\exports` 仍为 2026-10-09。

## 3. 第 5 项检查（.codebuddy 同步与任务03 等级格式）

- `.codebuddy` 已同步：`sync-from-cursor.ps1 -WhatIf` 输出 Rules `[same] paper-library.mdc`、Skills `[same] paper-intake / paper-maintenance / paper-supplement`（`side-effect-authorization.mdc` 为 CodeBuddy 独有，`[keep]`）；`.codebuddy\skills\paper-maintenance\SKILL.md` 存在且 frontmatter 已按脚本规则去掉 `disable-model-invocation`。**上一轮 F1 已解决。**
- 任务03 等级格式已改：正文表格的目标值改为 `` `计算机视觉顶会·主会` ``（无 `【】`），并新增“表格文字即 `rating.value`，不带 `【】` 或装饰性括号；口径内部括号如（ESCI）（收录性质待核）保留”的说明。全文检索 `【` 无残留。**上一轮 F4 已解决。**

## 4. 与上一轮验收报告的关系

| 上一轮条目 | 本轮结论 |
| --- | --- |
| F1 `.codebuddy` 未同步 | 已解决（§3） |
| F2 隔离演练无独立证据 | 已解决：本轮在最终代码上实测通过 plan→写入→退库→续办幂等→恢复→导出（§1、§2），命令与输出可复查 |
| F3 演练早于最终代码 | 已解决：本轮演练直接运行当前落盘的最终版本工具 |
| F4 任务03 等级带括号 | 已解决（§3） |
| F5 替换稿正文残留“回收站”旧口径 | 未复核（不影响任务03 执行路径） |
| F6/F7 schema 升级路径、未覆盖场景（跨卷/真并发/回收站） | 仍未覆盖；计划 §9.2 未纳入最小验收，本轮不据此判失败 |

说明：上一轮在 `%TEMP%\paper-maintenance-drill-50652` 发现的残留根**保留未动**（属上一轮证据，未获清理授权）；本报告只清理本轮自建的 `paper-acceptance-final`。

## 5. 是否具备执行任务03 的条件

**具备。** 依据：

1. 最终版本代码在隔离环境真实完成 metadata 写入、退库移动、恢复回迁、视图/导出重建，索引 revision/备份/幂等行为与计划一致；未见计划外字段变化或越权改动。
2. `.codebuddy` 侧 Rule 与三个 Skill 已与 `.cursor` 一致，维护 Skill 在 CodeBuddy 环境可用。
3. 任务03 的字段语义（独立 `rating`、无括号目标值、N04 年份留未知、N23 期刊年/在线年、StitchFusion 纠错、SubSurfaceGeoRobo DOI 不动）在任务03文件中已明确。
4. 正式库仍是 schema4/revision19/72 篇、未被本轮改动；任务03 第一次真实维护保存将按计划把 schema5 持久化。

执行前建议（不阻塞）：确认旧版工具已停止写索引；`retire` 的 `expected_title` 与索引题名需精确一致（不一致会在 plan 阶段停止）；退库 5 篇与任务01 式来源清理仍是两种独立授权。
