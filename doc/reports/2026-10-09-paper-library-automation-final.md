# 论文库自动化：最终收尾报告（2026-10-09）

> 上级审核结论：两个工作流的开发、隔离验收及首次真实试用全部通过，停止新增功能与测试。本报告只做收尾汇总，不重复既有报告的详细验收过程。详细材料见 [实施与隔离验收报告](2026-10-09-paper-library-automation-implementation.md)、[批量入库与审核](2026-10-08-paper-library-batch-intake.md)、[质量审核](2026-10-08-paper-library-quality-audit.md)、[年份与出版物补充](2026-10-08-new-paper-bibliography-fill.md)。

## 1. 最终实现

新增四个实现文件（工具本体自实施起未再改动）：

- `DFormer/.cursor/skills/paper-intake/SKILL.md`（67 行）——`/paper-intake` 入口：五类分类、默认只出审核计划、获批后 apply、跨消息重新调用。
- `DFormer/.cursor/skills/paper-supplement/SKILL.md`（96 行）——`/paper-supplement` 入口：定位 → 定点阅读 → 四区块草稿 → 审核计划 → apply。
- `DFormer/.cursor/rules/paper-library.mdc`（39 行）——共享 Rule：原资产只读、唯一 store 写入、人工身份字段保护、审批范围与冲突停止、证据类型区分。
- `D:\0Project\origin\_index\tools\paper_workflow.py`（1306 行）——四子命令薄封装：`intake-plan` / `intake-apply` / `supplement-plan` / `supplement-apply`，复用现有扫描器与 `paper_library_store.py`，未新增依赖、测试框架、数据库或常驻服务。

两个 Skill 均使用与目录同名的 `name`、设置 `disable-model-invocation: true`、不设 `paths`，已在 Customize → Skills 确认可被发现（Rule `paper-library` 的 Rules 标签页尚未目视确认）。

配套文档改动：外部 `_index/README.md`（工具清单、工作流与审批/中断说明、使用规则、库规模）、`DFormer/doc/guides/external-resources.md`（工具与两个 Skill 入口）、`DFormer/doc/guides/project.md`（库规模）。

两个入口各补了一条部分成功语义：索引 JSON 已保存而派生 Markdown 重建失败时按「JSON 已提交、视图失败」报告，不重跑 apply、不二次保存。

## 2. 两次真实试用结果

**入库（GeminiFusion）** — 计划 SHA-256 `e6adc67d…`；来源 `临时/临时论文/Jia 等 - 2024 - GeminiFusion …`（39 个文件，正文 sha256 `d5f07e09…`）分类为唯一一项 `new`。经同盘暂存 → 全资产校验 + 正文改名 → 目标落位 → 再校验 → 内存 dry-run → 白名单检查 → 一次保存：新增 **LIB000045**，revision 12→13，`next_lib_number` 45→46。旧 44 篇逐字节不变、顺序不变；`bak1` 等于审批基线；无暂存残留；来源保留。这是该流程第一次对真实库运行全库 dry-run，**未发现任何既有记录变化**，此前标注为未实测的逐记录稳定性由此关闭。

**补充（`LIB000028` MoSA）** — 计划 SHA-256 `e85c3d11…`（v2 版，含上级要求的两处定点修订：Eq.4 维度疑点记录、把「两模态都不可靠」限定为归一化融合权重本身）。经一次 `supplement-apply`：`human.supplement` 由 null 变为 10,459 字符，与批准计划逐字节一致，阅读状态保持 `abstract_only`，revision 13→14。该篇仅 `supplement` 一个字段变化；其余 44 篇（含 LIB000045）逐字节不变；人工编号、标签、核心标记、`bibliographic_overrides`、`auto`、`duplicate_review`、`next_lib_number` 均不变。

两次试用的审核材料（计划与草稿）保留在外部 `_index\intake-plans\2026-10-09\` 与 `_index\supplement-evidence\2026-10-09\`，含被取代的两份历史计划。

## 3. 最终论文库状态

- 权威索引 `D:\0Project\origin\_index\PAPER_LIBRARY_INDEX.json`：schema **4** / revision **14** / **45 篇** / `next_lib_number` **46**，更新时间 2026-10-09 09:21:07（UTC+8）。
- 索引指纹：`94af0e46f786483f56a50b2f62a61b167644318a60c23c25fcb34d67cac483de`。
- 管理视图 `PAPER_LIBRARY_INDEX.md`：SHA-256 `d7b253a88eed902a16de0d038fa0efe589b196264cd04b58f4b3de7ce5fa507d`（revision 14 重建）。
- 备份 `backups\`：`bak1` = revision 13（`c3d6fe53…`）、`bak2` = revision 12（`17de87b2…`）、`bak3` = revision 11（`b29d4289…`）。
- 库结构：论文库根目录 47 个目录 = 45 篇 canonical（全部顶层）+ 2 个容器目录（`DFormer-doc-paper`、`MMFR-附件`，后者内含 LIB000002 的 1 个 alternate 抽取）。
- 内容状态：45 篇阅读状态**全部仍为 `abstract_only`**；非空 `human.supplement` **12 篇**（其余 33 篇无笔记）；`alternate_bundles` 1 篇、`duplicate_review` 1 条（LIB000023，已移入 `论文_duplicates_review`），均未变。

## 4. AI 阅读导出同步

三份导出原本停在 revision 12 / 44 / 0 / 11，本轮用现有 `store.write_ai_export` 重新生成，现均为 revision 14、导出格式 1.0.2：

- `exports\PAPER_LIBRARY_FOR_AI.md`：45 篇，218,841 字节，SHA-256 `09d66683…`。
- `exports\PAPER_LIBRARY_FOR_AI_CORE.md`：0 篇（核心仍为 0，固定文件照写），595 字节，SHA-256 `226e57a4…`。
- `exports\PAPER_LIBRARY_FOR_AI_SUPPLEMENTED.md`：12 篇，148,946 字节，SHA-256 `cce69bee…`。

抽查确认：GeminiFusion（LIB000045）与 MoSA（LIB000028）都出现在 `all` 导出中，MoSA 落在 `supplemented` 导出中且包含本轮 v2 修订后的正文。导出只更新派生文件，权威 JSON 指纹在导出前后一致。

## 5. 已知限制与待处理事项

不影响正常使用，但建议上级知悉：

1. **外部索引与工具不在任何 Git 仓库内**。`D:\0Project\origin\`（含 `_index\` 与 `论文\`）不是 Git 仓库，`paper_workflow.py` 与索引只有 store 自身的 JSON 备份保护，没有版本控制。是否需要单独纳入版本管理属于新决定，本轮未做。
2. **LIB000045（GeminiFusion）书目缺项**：正文不含 DOI / arXiv / venue，年份与出版物仍缺；入库判重因此只依据规范化题名（已独立复核：与索引 45 条记录最高相似度 0.538，`_relation` 全部为 none）。补书目属入库后的独立任务。
3. **MoSA 笔记内记录的证据问题未消除**：Table 4 行标签与数值在正文与 XLSX 均错位（数値自洽但归属不可确定）、Table 5 一项 FLOPs 缺失、Table 1 符号表错行、式(20) 括号位置异常、Fig.5 正文均值 0.78 与图内 0.80 不一致、Pearson 0.72 与 Fig.6 均值仅存在于正文文字；Eq.4 的矩阵维度疑点按上级要求只作记录，未修改公式也未推断实现。这些是原 bundle 的提取质量问题，不是笔记或工具缺陷。
4. **工具的一处命名不一致**：计划里的 `draft_sha256` 是按行规整后的文本摘要，与草稿文件的字节 SHA-256 不同（草稿为 CRLF 行尾）；`apply` 用同一方式重算，绑定自洽，但对照文件哈希时会显得不一致。统一需要改工具，本轮未授权。
5. **复制过程中断（进程被杀造成目标半成品）未做确定性构造**，隔离验收只覆盖了来源变化、目标占用、并发写锁与索引并发修改等同类保护路径。
6. **Rule `paper-library` 的 Rules 标签页未目视确认**；两个 Skill 已确认可发现。
7. **阅读状态区分度仍低**：45 篇全为 `abstract_only`，12 篇有笔记也保持 `abstract_only`（按设计，笔记非空不等于全文读完）。若要跟踪全文工作，需要有意识地改用 `needs_full_text` 并记录范围。
8. **审核材料会持续累积**：`_index\intake-plans\` 与 `_index\supplement-evidence\` 按批次保留（含被取代的历史计划）。是否设保留/清理策略属新决定，本轮未清理。

## 6. 本轮未做

未修改工具代码；未做新的论文补充或正式入库；未运行完整测试套件；未删除任何历史数据（含被取代的历史计划与旧报告）；未提交或推送 Git。
