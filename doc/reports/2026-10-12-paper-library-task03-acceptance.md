# 论文库阶段二（任务03 正式数据维护）独立验收报告

> 日期：2026-10-12。方式：**严格只读**（未修改正式索引、论文资产、Rule、Skill、项目状态；未恢复退库论文；未重跑阶段一演练；未联网复核分区）。
> 依据文件：`临时/PAPER_LIBRARY_FINAL_MAINTENANCE.md`、`临时/PAPER_LIBRARY_MAINTENANCE_PLAN.md`、请求/计划 JSON、`PAPER_LIBRARY_INDEX.json`、`PAPER_LIBRARY_INDEX.md`、`exports/` 三份导出、`backups/PAPER_LIBRARY_INDEX.json.bak1`（revision 19）、`论文_retired/` 资产、任务03 原指令。

## 1. 通过项（均以实际文件核对，非引用执行模型结论）

**1.1 索引基线**：`schema_version=5`、`revision=20`、`papers=72`、`next_lib_number=73`、`bootstrap_complete=true`、`updated_at=2026-10-12T21:22:28+08:00`；活动 **67** / 退库 **5**，状态取值仅 `present`/`retired`。与任务03 要求完全一致。

**1.2 计划与批准基线**：`临时/PAPER_LIBRARY_MAINTENANCE_PLAN_2026-10-12.json` 为 72 项（`metadata` 67 + `retire` 5），`pending_moves=5`，基线指纹 `116f314a6bb137c7d34be3a19…`（= 保存前指纹）；计划文件实际 SHA-256 = `ba0ec18334294fefc292ec3d1e546f070225ab44d516cc0b35704b92139f7344`，与报告 §1 一致；retire 项 5 个 `expected_title` 与索引题名逐字相同。

**1.3 元数据（对照请求逐项比对，0 处不符）**：把请求文件中 67 个 `metadata` 项的目标值与索引实际 override 逐字段比较（含 `year_kind`）——**mismatch = 0**。变更分布与报告一致：`rating` 67 篇、`year` 26 篇、`venue` 26 篇（`venue` 变更集合 = 023/025/030 + 新 23 篇，**LIB000044 未提交 venue 变更**，与清单一致）；`LIB000068` **未写 year**（N04 年份按语义留未知，接收事实写入 venue 的 `note`）。

抽查（实际索引值）：

| LIB | year / year_kind | venue | rating |
| --- | --- | --- | --- |
| LIB000023 / 025 | 2026 / conference_year | CVPR 2026 | 计算机视觉顶会·主会 |
| LIB000030 | 2026 / preprint_date | arXiv (preprint) | 预印本·尚无正式发表记录 |
| LIB000044 | 2026 / preprint_date | arXiv (preprint) | 预印本·尚无正式发表记录 |
| LIB000068 | **未知**（note：2024 接收，卷期年待补） | IEEE TPAMI | TPAMI顶刊 / CCF A；中科院1区Top（2025参考转录） |
| LIB000069 | 2025 / conference_year | **ACM Multimedia 2025**（`auto.venue=Neural Networks` 保留，note 记纠错依据） | 多媒体顶会·主会 |
| LIB000070 | 2026 / issue_date（note 保留 2025 在线，`auto.year=2025` 未动） | PFG – J. of Photogrammetry… | 测绘/地学专业同行评审期刊；分区本轮未官方核验 |

`venue` 与 `rating` 分离：`rating` 条目键为 `value/source_type/confidence/confirmed_at/note`（必要时含 `source_url`），未出现等级拼进出版物字符串；全文检索无 `【】`。`rating` 实际分布（67 篇、19 种取值）与报告 §3.2 各类计数逐项吻合（唯一差异见 §2.1）。

**1.4 退库对象与资产**：`status=retired` 集合**恰好 = {LIB000004, LIB000009, LIB000011, LIB000021, LIB000022}**，无其他论文被退库（LIB000017/019/025 仍为 `present`）。逐篇实测：

- 原位置 `论文\<目录>` 均不存在；`论文_retired\<LIB>\<原目录名>\` 均存在且含完整附件（文件数 17/21/49/38/84）。
- 正文 Markdown 可读，其 SHA-256 **与索引 `body_sha256` 逐篇一致**（资产字节未变）。
- `human.retirement` 结构完整（`retired_at=2026-10-12T21:22:28+08:00`、`reason`、`previous_status=present`、`bundles` 映射）；LIB 编号、`auto.*`、已有人工编号（LIB000021 `RE188`、LIB000022 `RE447` 保留）、标签/核心/阅读状态/已批准书目均保留；5 篇**均无 `rating`**（符合任务03 §三）。
- `论文_retired` 下仅这 5 个 `<LIB>` 目录；`论文\` 内无 `论文_retired`；`论文_duplicates_review` 仍为 1 项，与索引 `duplicate_review=1` 一致。

**1.5 与保存前备份（rev19）的全库白名单差分**：LIB 集合与顺序完全一致，根字段仅 `papers`/`revision`/`updated_at` 变化，`library_root`/`bootstrap_complete`/`next_lib_number`/`duplicate_review` 不变。逐记录比对结果：

- `auto.*` 字段**逐键完全相同**（`raw auto identical: True`）；
- 状态变化仅 5 篇 `present → retired`；
- `human` 变化仅：67 篇的授权 override、5 篇的 `retirement`，以及任务01 新增的 23 篇（LIB000050–072）原本缺失的默认键（`manual_ids/unverified_manual_ids/primary_manual_id/tags/is_core/reading_status/supplement`）被这次常规保存持久化（报告已单列为“既有加载规范化”）；
- **未发现任何越权/非预期变化（unexpected: none）**，无 human 键被删除。

**1.6 派生产物**：`PAPER_LIBRARY_INDEX.md`（21:22:28）显示 `活动记录：67；已退库：5；总登记：72`，主表 67 行且**活动章节不含 5 篇退库论文**，另列退库登记（LIB / 时间 / 原因 / 原状态 / 原路径→恢复路径）。三份导出（21:22:28）为 67 / 15 / 21 篇，均标注 `源索引 revision：20`、`源 schema_version：5`、导出格式 1.0.3 与“只含活动记录”，**标题与 LIB 均不含 5 篇退库论文**。`_index` 下无残留 `.lock`/`.tmp`。

**1.7 正式库与工作区状态**：`论文\` 下 67 篇活动论文原目录全部存在（无缺失）；根内 2 个未登记目录为既有状态（见 §2.4）。Git 无提交/推送，改动仅本项目文档与规则/Skill。

## 2. 发现的问题（均**不阻塞**）

1. **报告统计口径小误差**：`PAPER_LIBRARY_FINAL_MAINTENANCE.md` §3.2 表最后一行写“其余 **9** 类各 1 篇”，实际为 **10** 类各 1 篇；合计 67 正确（21+8+8+5+5+3+3+2+2+10=67），索引数据无误。属报告文字计数错误，建议日后一并更正，不影响验收。
2. **同一期刊存在两套 `rating` 措辞**（报告 §6.4 已自述）：如 TPAMI 在旧库 LIB000006/008/037 写 `中科院2025大类1区·Top`，在新库 LIB000058/068 写 `TPAMI顶刊 / CCF A；中科院1区Top（2025参考转录）`；会议类同样存在中英混排。来源文档不同导致，本轮未统一——按 `rating` 字符串做筛选/统计时需注意，将来若需统一口径要另立一次 `rating` 重写授权。
3. **遗留待定点项**（报告 §6 已列，均超出本轮授权字段）：LIB000068 年份仍未知；LIB000070 的 DOI 仍是数据集 DOI `10.1594/pangaea.975532`；LIB000054 本地题名仍有 `Eficient` 拼写；中科院分区为第三方 2025 转录、未登录官方订阅库核验（每条 `note` 已声明）。
4. **正式库根内 2 个未登记目录**：`论文\DFormer-doc-paper`（4 个文件）、`论文\MMFR-附件`（53 个文件；其中含 LIB000002 的 alternate bundle，索引中已登记该子路径）。均为既有状态、非本轮产生，索引与磁盘一致，不阻塞。
5. **阶段一残留隔离根** `%TEMP%\paper-maintenance-drill-50652` 仍保留（上一轮验收已记录，本轮无清理授权）；不涉及正式库数据。

## 3. 结论

**验收通过；可以结束论文库维护阶段，转入后续科研任务（任务04 等）。**

- 5 项核对要求全部满足：索引 schema5/revision20/活动 67/退库 5/总登记 72/`next_lib_number=73`；67 篇 `rating` 与 `venue` 独立且与批准请求逐项一致（重点 6 篇抽查全部正确）；5 篇退库对象准确、资产完整可读、原 LIB 与人工信息未丢失、无其他论文被退库；与 rev19 备份的白名单差分 0 处越权变化；管理视图与三份导出与 revision20 一致且不含退库论文。
- 发现的问题均为文档计数、口径措辞与已声明的遗留待定点，**不阻塞**后续科研使用；建议在转入科研任务时不依赖 `rating` 字符串做精确统计，并在将来需要时另行授权统一 `rating` 口径或补 LIB000068 年份、LIB000070 DOI。
- 正式库现处于 `schema_version=5`：**旧版工具不得再写该索引**；如需重建导出只用只读入口 `views-refresh`。
