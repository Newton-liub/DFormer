---
name: paper-maintenance
description: Authorized bibliography/rating maintenance and exact retire/restore of specified papers in the external library. When the user explicitly names LIBs, fields and sources, complete the supported plan/apply flow in the same session; preflight-only requests stop after planning. Restore always needs its own explicit authorization.
disable-model-invocation: true
---

# paper-maintenance：指定书目/等级维护与精确退库、恢复

入口：`D:\2Env\anaconda\python.exe D:\0Project\origin\_index\tools\paper_workflow.py`（子命令 `maintenance-plan` / `maintenance-apply` / `views-refresh`）。遵守 `.cursor/rules/paper-library.mdc`；命令或参数不清楚时先看对应子命令 `--help`。不做科研取舍，只执行用户已明确指定的对象与字段。

## 必要上下文

- 本轮尚未读取 `doc/state/current.md` 时先读取一次，确认授权边界与恢复点；同一会话、同一任务范围已读取则复用。
- 字段所有权、`rating` 语义或恢复目录约定不清楚时，查 `D:\0Project\origin\_index\README.md` 与 `doc/guides/paper-library.md`。

## 授权前置

- 只处理用户已明确指定的 LIB、字段与依据；`metadata` 只允许 `year` / `venue` / `rating` / `doi`（`doi` 仅用于点名 LIB 的定点纠错，保留 `auto.doi` 扫描值），`retire` / `restore` 只允许点名的 LIB 加完整题名。
- 未指定的 override、人工编号、标签、核心标记、阅读状态、旧笔记与其他记录一律保留。
- 恢复（restore）必须另有明确授权；不得因为此前执行过退库就自动恢复。任务 01 式来源清理与退库是两种授权。

## 流程

1. 生成一份短维护请求 JSON（根为 `items`）：`metadata` 项给出准确 `lib_id` 与 `changes`（值为 override 载荷或 `null` 表示撤销）；`retire` / `restore` 项给出 `lib_id`、完整 `expected_title`，`retire` 首办还要给 `reason`。同一请求不重复 LIB，不混用 retire 与 restore。
2. `maintenance-plan --changes <请求.json> --plan <审核.json> [--recovery-root <库外恢复根>]`；隔离运行时另加 `--index` 与配套 `--root`。计划从当前索引自动带出标题、原有效值与目标值、路径动作供核对。
3. 用户已明确授权且计划在范围内时，同一会话计算真实计划文件 SHA-256 并执行 `maintenance-apply --plan <审核.json> --approved-plan-sha256 <sha> [--export-dir <导出目录>]`。只要求“预检”时，plan 后停止。
4. 成功后核对：书目项写入 `year` / `venue` / `rating` / `doi`；退库项 `status=retired` 且资产已同卷移入恢复路径；恢复项资产已移回原位置且 `human.retirement` 已移除。apply 会自动重建管理视图与三个 AI 导出。

## 停止条件与简报

- 题名与索引不一致、请求含白名单外字段、索引并发修改、原位置与恢复位置同时存在或同时缺失、目标已存在、路径含链接/联接、跨卷移动、恢复根冲突或退库身份被重新扫描命中：立即停止并报告。
- 计划提交后旧指纹即失效；不得改指纹、自动回滚或自动重试。
- 纯目录续办（索引字段无变化）不保存、不增加 revision；失败时报告索引是否已提交、已移与未移目录、恢复前置条件。

简报实际改动的 LIB 与字段、退库/恢复对象与目录去向、活动/退库/总登记数与源 revision；未执行或未核实的项如实注明。不推送 Git，不做科研裁决。
