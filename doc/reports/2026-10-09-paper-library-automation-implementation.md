# 论文库长期自动化：实施与隔离验收报告（待上级审核）

> 日期：2026-10-09。依据：上级已审核通过的 [`临时/PAPER_LIBRARY_AUTOMATION_FINAL_PLAN.md`](../../临时/PAPER_LIBRARY_AUTOMATION_FINAL_PLAN.md)，并吸收「入库复制安全 / 验收精简 / 保持既定架构 / 执行范围」四项修订。本轮只完成 Skill、Rule、工具与隔离验收；**未做正式论文入库、未修改权威索引、未清理来源、未批量补充、未 Git 提交或推送**。

## 1. 实际修改文件

新增（4 个，即计划的全部实现文件）：

| 文件 | 行数 | 作用 |
| --- | --- | --- |
| `DFormer/.cursor/skills/paper-intake/SKILL.md` | 67 | `/paper-intake` 入库入口：默认只出审核计划、获批后 apply、跨消息重新调用 |
| `DFormer/.cursor/skills/paper-supplement/SKILL.md` | 96 | `/paper-supplement` 证据补充入口：定位→草稿→审核→保存 |
| `DFormer/.cursor/rules/paper-library.mdc` | 39 | 共享 Rule：原资产只读、唯一 store 写入、人工身份字段保护、审批范围与冲突停止、证据类型区分 |
| `origin/_index/tools/paper_workflow.py` | 1306 | 四子命令薄封装：`intake-plan` / `intake-apply` / `supplement-plan` / `supplement-apply` |

修改（2 个，均在计划推荐范围内）：

| 文件 | 改动 |
| --- | --- |
| `origin/_index/README.md` | 文件清单加入 `paper_workflow.py`；新增「入库与证据补充工作流」小节（四子命令、审批与中断语义）；使用规则加入审批制入口一条 |
| `DFormer/doc/guides/external-resources.md` | 工具清单加入 `paper_workflow.py`；「怎么读」加入 `/paper-intake`、`/paper-supplement` 两个 Skill 入口 |

本报告与随后的 `doc/state/current.md` 更新是第 3 处文档改动。扫描器、store、UI 与启动器**保持不变**；无新增依赖、测试框架、数据库或常驻服务；未固化篇数、revision 或格式版本。

## 2. Skill / Rule 识别结果

- frontmatter、路径、调用示例、审批边界已完成内容复核：两个 Skill 的 `name` 与目录同名，`description` 分别限定「处理后 bundle 接入」与「指定论文证据补充」，均设置 `disable-model-invocation: true`，均不设 `paths`；Rule 为 `alwaysApply: false`、无 `globs`。
- Skill 各约 100 行以内，命令串与阶段 2 实测的 `--help` 逐项对齐（子命令名、`--source` / `--plan` / `--approved-plan-sha256` / `--id` / `--draft` / `--expected-fingerprint` / `--evidence` / `--scope` / `--reading-status` / `--full-text-scope` / `--replace` / `--index` / `--root`）。
- 两个 Skill 互相独立：调用一个不需要读取另一个；各自明确要求先读 `doc/state/current.md` 与共享 Rule。
- **未核验项（诚实说明）**：本轮无法从工具层观察 Cursor 的 Customize → Skills/Rules 与 `/` 列表。Cursor 在启动时发现 Skill，新增文件不保证热刷新。请上级在 **Customize → Skills/Rules** 与 `/` 搜索中确认 `paper-intake`、`paper-supplement` 与 `paper-library` Rule；若未出现，先重新加载窗口再确认（不把「新建会话」当作保证刷新机制）。

## 3. 最小验收结果（隔离）

全部命令显式指向系统临时目录 `%TEMP%\paper-wf-acceptance\`（合成 bundle + 隔离索引，LIB 由现有 `_sync_index` 生成）。验收前后**权威索引指纹不变**，所有写入都落在隔离目录。

### 3.1 入库分类（一次 `intake-plan`，12 个来源）

| 样本 | 期望 | 实测 |
| --- | --- | --- |
| 独立新 bundle | 全新 | `new` |
| 同批重复的第二份 | 批内完全重复 | `full_duplicate`（reason 指向批内代表） |
| 与已索引论文正文+附件逐字节相同 | 全资产重复 | `full_duplicate` |
| 正文相同、文件数相同、一个附件 hash 不同（UMIS 类） | 正文同附件异 | `body_same_attachments_differ`，`comparison.changed` 精确列出该文件 |
| 同题名、正文不同 | 疑似版本 | `suspected_version`（relation=title，body 不同） |
| 多个候选身份 | 疑似版本 | `suspected_version`（multiple candidates） |
| 目标目录名被占用、且无身份候选 | 异常 | `anomaly`（refusing automatic suffixing） |
| 同身份候选、但目标名恰好被占用 | 疑似版本 | `suspected_version`（不自动加后缀） |
| 目录内无 Markdown 正文 | 异常 | `anomaly` |
| 含 0 字节附件 | 异常 | `anomaly`（zero-byte file 明确列出） |

计划中的 `approved_count` 只包含 4 个 `new`；其余分类全部留原处并写出理由。

### 3.2 复制与提交（`intake-apply`）

- 计划 SHA-256 不匹配 → 退出码 2，**未复制任何目录**。
- `--root` 与索引 `library_root` 不一致 → 停止；同名审核文件已存在 → 拒绝覆盖；`intake-plan` 前后索引指纹不变。
- 正确批准 → 4 个 `new` 复制成功，目标目录名与预期一致；**正文改名映射生效**（`odd-source-name.md` → `Rename Target.md`，字节不变，附件名不变）。
- 临时暂存目录（复制到正式扫描范围之外的**同盘**位置、校验后再移入目标）**未留残留**；来源 12 个目录全部保留。
- 一次保存后逐项核对：`next_lib_number` 4→8、`revision` +1、新 LIB 为 LIB000004–LIB000007 且顺序稳定；**原有 LIB 记录逐字节不变**，`human`（tags / is_core / override / supplement / 人工编号）、`auto` 书目、`alternate_bundles`、`duplicate_review` 及其他根字段全部不变。
- 子集审批：批准 3 个中的 1 个 → 只复制并登记该 1 篇（LIB000008），其余不复制。
- 审批把一个弱判断项（`full_duplicate`）标为 approved → apply 直接拒绝（"refusing to copy it"），阻止弱 exact/alternate 自动决策。
- **完全重复 / 无新增**：对同一来源再 `intake-plan` → `approved_count` 0，apply 报「Nothing to do」，索引指纹不变、备份未轮转、revision 未增加。
- 审批后改变来源文件 → 停止，目标未创建，暂存无残留，来源保留、索引未写。
- 审批后改变隔离索引（revision 前进）→ 停止，未复制任何目录。

### 3.3 白名单安全网（纯内存定点检查）

对 `to_disk_document` 前后差异比对函数直接构造被篡改文档，逐一确认拒绝：清单外新增记录、既有记录被重扫改变、被判为 alternate、给新记录写入人工编号；入库侧干净的新增一条记录返回空问题列表。supplement 侧确认：无关记录变化、未批准的 `human` 字段变化（如 `is_core`）、`next_lib_number` 变化均被拒绝，而只有目标 LIB `human.supplement` 单字段变化返回空问题列表。

### 3.4 补充证据（`supplement-plan` / `supplement-apply`）

- 定位：`--id PR0007` 与 `--id PR007` 规范化后命中同一篇（`PR24 == PR0024` 语义）；`LIB000001` 直接命中；未知人工编号、未知 LIB 均零匹配停止。
- 追加语义：保存后旧 `content_md` 为完整前缀、旧 `source_note` 被追加而非替换、旧 `sources` 条目与新 `extension_key` 全部保留、`updated_at` 更新；`tags` / `is_core` / override / 人工编号不变，其他记录、`alternate_bundles`、`duplicate_review` 不变，`revision` +1。
- **no-op**：再次提交同一段草稿 → `no_op: true`，apply 报「Nothing to do」，索引指纹不变。
- 阅读期间 F0 变化 → 停止；审批后正文或所引用本地证据 hash 变化 → 停止。
- 阅读状态：省略则保持原值；`needs_full_text` 可批准保存；`full_text_completed` 未提供 `--full-text-scope` → 停止。局部阅读未自动标记全文完成。
- 计划 SHA-256 不匹配 → 停止。
- **store 写锁被其他进程持有**：apply 在 15.2 秒后停止，索引指纹不变；释放锁后同一计划正常保存（证明停止仅由并发写锁引起）。
- **JSON 已提交但派生 Markdown 失败**：退出码 0，明确输出「JSON is committed, but the Markdown view failed… Do not re-run」，重读确认 JSON 已含本次改动、revision 已前进——不把已成功的 JSON 保存报告成失败，也不二次保存。

### 3.5 未运行的检查

- 未运行项目测试套件（不能覆盖论文工作流的主要风险，计划明确不要求）。
- 未运行真实索引 dry-run / 同步 / 迁移，未重哈希全库资产。
- **未定点覆盖**：复制过程真正中断（进程被杀造成目标目录半成品）没有确定性构造。设计上以「同盘暂存 → 全资产校验 → 移入目标 → 再校验」限制影响面，并以「来源在审批后变化即停止」覆盖了同类保护路径，但该中断分支本身未实测，如实记录。

## 4. 已知限制

1. 全资产比较只把「已识别的主正文文件」映射为统一正文角色，其余按原相对路径比较；正文角色识别依赖现有 `_analyse_bundle`，多正文歧义时整项归入异常。
2. 「缺少正文引用的本地资产」以结构化近似实现（Figure/Tables/Formula/Word 之外的附件、0 字节文件、链接/junction 均记为异常），未逐条解析正文引用。
3. 正文同附件异、疑似版本、完全重复全部留待人工裁决；工具**不提供**把版本写成新 LIB、alternate 或 duplicate-review 的自动路径。
4. 已有笔记的**替换**需显式 `--replace --replace-reason`；工具记录替换理由与旧内容摘要，但不做逐段增删 diff。
5. `human.supplement.sources` 没有 schema 级契约；工具追加 `{kind, path, sha256, recorded_at}` 条目并完整保留已有条目。
6. plan / apply 的停止原因可能来自共享 store 的中文错误信息；在 PowerShell 控制台可能显示为乱码（**显示层编码问题**，索引内容与判定不受影响）。
7. 工具不触发任何 AI 阅读导出；需要时由上级按现有导出函数单独执行。
8. 隔离验收证明的是逻辑与安全边界；真实库的 44 篇在 dry-run 中的逐记录稳定性仍需在真实试用时由实际数据确认（若某篇重扫结果与既有记录不同，按设计会停止并要求重新送审）。

## 5. 真实试用所需授权

本轮在此停止并交上级审核。真实试用需分别授权，缺一不可：

1. **入库试用**：一篇由用户确认全新的 bundle 及其当前来源路径；授权后按「预检 → 审核 → 复制 → 资产核验 → 内存 dry-run → 正式同步 → 验收」执行，并接受若 44 篇 dry-run 出现旧记录刷新变化则停止重审。
2. **补充试用**：一个指定 LIB（或已确认人工编号）、具体阅读问题与范围、是否允许查官方实现；默认追加、阅读状态 keep、只读本地获准证据。
3. **默认边界确认**：UMIS-Mine 不作为自动合并对象；不清理来源目录；不自动 Git 提交或推送；不新建测试框架；不升级 schema。

需要哪一篇做入库试用、哪一篇做补充试用，请上级指定。
