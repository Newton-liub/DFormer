# 2026-10-07 项目干净重建与轻量机制

> 角色：本次重建的实施计划与验收依据（已获用户批准）。实时状态见 `doc/state/current.md`。

## 目标

Clean upstream + thin research layer + centralized docs + lightweight project skills + selective migration。

不做“把旧项目整理得更规范”，而是完整封存旧项目、以作者最新 DFormer++ 为基线重建，只迁移下一轮研究最低限度需要的东西。

## 已确认边界

- 先建立干净基线与工作机制，再确定论文方向；本轮不续跑旧 Natural missing 实验。
- 保留作者原有 `README.md`、`LICENSE`、`figs/` 说明；我们新增的正式文档进入 `doc/`；`.cursor/` 中必须存在的 Rule/Skill 属于工具配置。
- 旧项目不做清理式重构，整目录封存。

## 实施步骤

1. **重建前检查**：Git 状态、未跟踪/忽略内容、归档目标占用、外部链接与目录占用。（已核对：归档目标 `D:\0Project\DFormer-archive-20261007` 空闲；仓库内无 junction/reparse point；用户级子代理目录为空，无项目内模型配置可迁移。）
2. **封存旧项目**：把 `D:\0Project\DFormer\` 的完整内容移到 `D:\0Project\DFormer-archive-20261007\`，保留隐藏文件、`.git`、未提交修改、未跟踪文件、被忽略产物。归档后复核 HEAD、分支、修改与未跟踪清单一致。
3. **重建基线**：在 `D:\0Project\DFormer\` 建立新仓库，`origin` = 个人 fork，新增 `upstream` = 作者仓库，fetch 后从 `upstream/main` 创建 `research/dformerpp-clean-start`。不改写 fork 的 `main`、不恢复旧 `cloud` remote、不 merge 旧分支、不推送。
4. **最小文档与约定**：`doc/state/current.md`、`doc/ideas/ideas.md`、`doc/guides/project.md`、`doc/guides/cloud.md`、`doc/plans/`；`outputs/` 作为本地与云端产物的统一入口，默认忽略；`research/`、`local_configs/research/`、`doc/reports/`、`doc/archive/` 按需创建。
5. **Cursor 机制**：两个短 Rule（`project-context.mdc`、`project-safety.mdc`）与四个项目 Skill（`project-state`、`research-idea`、`subagent-dispatch`、`compshare-cloud`）。不修改现有子代理模型配置，不复制旧 MMFR 管理体系。
6. **验证**：定点检查 Git 基线、归档一致性、`outputs/` 不进入 Git、新文档链接与 Rule/Skill 元数据；在新会话确认 Skill 可发现与触发行为。

## 失败与恢复

- 目录占用或 clone/fetch 失败即停止并报告，保留归档与准确恢复点；不覆盖已有归档，不自动删除失败残留。
- 恢复原目录需先确认原路径占用情况，不覆盖新内容。

## 本轮不包含

依赖安装升级、配置 import、模型 forward、完整测试、GPU/训练/评价、云实例启停或删除、实验结果重验、Git 提交与推送。

## 验收结果

- 旧项目完整归档并逐项复核一致（HEAD `8c274c59…`、分支 `perf/mmfr-a2-v3-pipeline-opt1`、13 个修改、3 个未跟踪条目）。
- 新分支 `research/dformerpp-clean-start` = `e3273009b759b578945483828ff315d560be94c9`，工作区仅含本轮新增文档与 Cursor 文件。
- 实际执行偏差：Windows 下 Cursor 侧进程持有目录句柄，无法整目录改名，改为把全部内容移入归档目录；残留空 `.pytest_cache` 被占用无法删除（Git 已忽略），因此用 `git init` + `git fetch` 建立仓库而非 `git clone`，结果等价。
