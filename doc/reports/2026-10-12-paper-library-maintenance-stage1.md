# 论文库维护能力扩展 · 阶段一实施与验收报告

> 日期：2026-10-12。依据：已审核的 `临时/PAPER_LIBRARY_MAINTENANCE_PLAN.md`（阶段一：工程功能实现与最小验收）。
> 本轮只实施工程功能并做一次隔离环境真实演练。**未修改正式论文索引、未移动正式论文资产、未执行任务03数据维护、未提交或推送 Git。**
>
> **验收口径更正（2026-10-12 晚，独立验收后）：** 本报告初稿曾把“一次隔离真实演练 92/92 通过”与“隔离根已清理（已确认不存在残留）”写成已完成结论。独立验收（见 [独立验收报告](2026-10-12-paper-library-maintenance-stage1-acceptance.md)，F2/F3）指出这两条**不成立**：磁盘上仍留有本次演练的一个隔离根 `%TEMP%\paper-maintenance-drill-50652`，且该残留只覆盖到 plan 阶段；报告所述 92 项端到端检查的原始输出已随临时日志一并删除，无法独立复查，且其运行时间早于工具的最终修订。**当前准确口径是：代码与定点检查通过；一次隔离真实演练按原执行输出记录为 92/92，但独立验收尚未充分证实；待最终代码版本的一次独立隔离演练复核。** 下文 §2.2 与 §3 已按此口径改写，未删除原记录，只标注其证据等级。

## 1. 实际改动

### 1.1 `D:\0Project\origin\_index\tools\paper_library_store.py`（共享数据层）

- `SCHEMA_VERSION` 升为 5，读取支持 1–5；schema4 在内存中兼容升级，**读取本身不写索引**。
- `OVERRIDE_FIELDS` 增加 `rating`；新增共享 `MAINTENANCE_FIELDS=("year","venue","rating")`、`FIELD_LABELS`、资源状态常量（`present` / `missing` / `retired`）与 `STATUS_LABELS`。
- 新增 `set_bibliographic_override(paper, field, payload)`：UI 与维护 CLI 共用的唯一写入语义。`payload=None` 删除 override；业务键（`value` / `year_kind` / 来源 / `note` / `raw_value`）不变时返回 `False` 且不刷新 `confirmed_at`；只改来源或说明也算真实变化。
- 新增 `is_retired` / `retirement_of` / `active_papers` / `owned_bundle_paths` / `recovery_root_of` / `effective_year_kind` / `bibliographic_year_note`；`YEAR_NOTE_PREPRINT = "预印本首发年，非正式发表年"`。
- `validate_document` 增加：`status` 取值检查、`retired` 必须有合法 `human.retirement`（`retired_at` / `reason` / `previous_status` / `bundles`，映射覆盖该 LIB 全部 bundle、不重复）、非 retired 不得残留 `retirement`、`next_lib_number` 不得低于已用最高编号。
- `ensure_paper_defaults` 不新增空 `rating` / `retirement`。
- `index_lock` 改为 `yield lock`（原忽略返回值的调用者不受影响）；`save_index` 新增 `held_lock`，只接受 `lock_file` 对应当前索引 `.lock` 的 FileLock，**没有跳过锁的开关**，其余检查、备份与原子保存不变。
- `scope_papers` 先排除 retired；`_export_paper_block` 将年份、出版物、论文等级分开输出（预印本年份附口径，`rating` 的 `note` / 来源标识独立成行）；`render_ai_markdown` 升到导出格式 1.0.3 并声明只含活动记录；`render_markdown` 主表拆出「出版物」「论文等级」列、显示活动/退库/总登记数、新增“已退库登记（不在活动集合）”小节，并纠正原来的“编辑请改 JSON”提示。

### 1.2 `D:\0Project\origin\_index\tools\paper_workflow.py`（审批制维护入口）

- 新增 `maintenance-plan` / `maintenance-apply` / `views-refresh` 三个子命令，复用现有 `--index` / `--root`。
- 维护请求为一份短 JSON：`items` 每项含准确 `lib_id` 与 `operation`（仅 `metadata` / `retire` / `restore`）。`metadata` 的 `changes` 只允许 `year` / `venue` / `rating`，值为 override 载荷或 `null`；`retire` / `restore` 必须带完整 `expected_title`，`retire` 首办还要 `reason`。同一请求不重复 LIB、不混用 retire 与 restore。
- plan 只读：取基线指纹，校验 LIB 唯一性、字段白名单、来源载荷与全部路径，自动带出标题与 before/after；不扫描整库资产。
- apply 在**同一把已持有的写锁**内重读索引并复核批准基线，从最新基线深拷贝、按白名单重建修改，再用 `_verify_maintenance_diff` 校验：未批准记录、LIB 顺序、`auto` 字段、其他 `human` 字段、`duplicate_review` 与根字段一律不得变化。
- 提交顺序：首次退库**先保存 retired 登记与恢复映射，再同卷移动目录**；restore **先移回目录，全部到位后才保存原状态并移除 `human.retirement`**；已 retired 的纯目录续办只移动目录、不保存、不增加 revision。全批至多一次 JSON 保存，`working != locked_document` 才算有变化。
- 目录移动 `_validate_move_action` / `_apply_retirement_moves`：先整批预检（原路径须在资产根内、恢复路径须在批准恢复根内、同卷、非 reparse point、源存在、目标不存在、互不嵌套），再执行 rename；拒绝覆盖、合并、跨卷与回收站 fallback。
- 报告既有 `human` 默认规范化（`_normalised_human_defaults`），并把“已提交 / 未移目录 / 恢复前置条件”分开打印；失败不自动回滚、不重试、不改指纹。
- `_cmd_supplement_plan` / `_cmd_supplement_apply` 拒绝写已退库记录；`_index_candidates` 保留 retired 身份，`_classify_item` 命中 retired 时返回 `anomaly` 并提示走明确授权的 restore。
- `views-refresh` 只读索引，重建管理 Markdown 与三份固定导出，绝不调用 `save_index`。

### 1.3 `D:\0Project\origin\_index\tools\paper_library.py`（扫描器）

- `_sync_index` 缺席循环跳过 retired 记录（不改写成 `missing` / `present`），高位编号继续覆盖全部记录（retired 仍占用 LIB）。
- 新增 `_guard_retired_identity`：扫描/导入命中已退库记录的路径或身份（`strong` / `title` 关系）时，在**任何目录动作之前**抛 `LibraryError` 终止，阻止重新导入、身份合并或另发 LIB。
- `_apply_auto_fields` 继续完整保护 `human`（含 `rating` / `retirement`）；最终汇总新增活动/退库计数。

### 1.4 `D:\0Project\origin\_index\tools\paper_library_ui.py`（最小兼容改动）

- 列表新增独立「出版物」「论文等级」列与共享资源状态标签；年份列在预印本语义下附口径。
- 详情页新增 `_set_manual_bibliography(lib_id, field)` 共用回调与「人工确认书目：年份 / 出版物 / 论文等级」编辑器（含 `year_kind` 与共用来源/note/URL/来源类型/置信度）：**输入框只预填已确认的 override 值**，因此渲染或未编辑的字段不会被降级为人工来源；留空即撤销该字段覆盖；未渲染的编辑器一律不动字段。
- `_diff_lines` 使用共享字段标签，除 value 变化外还显示 `year_kind` / 来源类型 / URL / 置信度 / 说明等真实来源变化，并显示退库登记的增删；不产生 `confirmed_at` 假差异。
- retired 记录在页面内只读（原书目、原因、恢复路径与只读笔记，目录按钮走恢复映射的真实路径），不参与候选批量接受与批量编号编辑；筛选新增资源状态「已退库」与可选的“显示已退库记录（只读，仅论文库视图）”。
- `commit_draft` 增加“无未保存修改时不保存”的防御；`bulk_safe_fields` 对 retired 直接返回空。

### 1.5 Rule、Skill 与指南

- `.cursor/rules/paper-library.mdc`：把受控维护字段收窄为 `year` / `venue` / `rating`（`rating` 独立、未知显示「未记录」）、明确 `preprint_date` / `arxiv_v1` 不是正式发表年、保护 `human.retirement` 只允许精确授权更新、资产只读条款加入“指定 LIB 可恢复整 bundle 移出/移回”例外、把“旧记录变化即停止”改为“**未授权**的旧记录或字段变化即停止”、加入退库身份不再导入与“JSON 已提交 / 目录 / 派生视图分别报告”，并把同轮授权扩展到已明确指定的 metadata 与精确退库。
- 新增 `.cursor/skills/paper-maintenance/SKILL.md`（`disable-model-invocation: true`）：授权前置、请求与 plan/apply、精确退库/恢复、冲突与部分完成报告、`views-refresh`；恢复须另行明确授权。
- `paper-intake` 增加一条“需要书目/等级补录时转 `/paper-maintenance`，字段依据与授权须明确”；`paper-supplement` 增加分工段（书目/等级/退库交 paper-maintenance，退库记录只读）。
- `doc/guides/paper-library.md`：用新的受支持维护入口替换“没有批量 CLI / 退库入口”，给出 `maintenance-plan` / `maintenance-apply` / `views-refresh` 的调用例与字段列表。
- `doc/guides/external-resources.md`：同步 schema 5、七个子命令、`论文_retired` 恢复目录、retired 活动/非活动分层与“扫描不代替人工维护、等级不写进导出冒充入库”。
- `_index/README.md`：schema5、字段所有权新增 `rating` 与 `human.retirement`、三个新增命令与维护工作流章节（至多一次保存、提交顺序、续办/中断语义）、活动/退库计数、可恢复目录、UI 独立编辑与 retired 只读、导出 1.0.3；把已结束的 44 篇 / 72 篇等数字标为历史记录，当前数量以 JSON 为准。
- 原任务03文件：把“出版物（附等级）”“末尾追加【等级】”“后缀验收”改为 `venue` + 独立 `rating`，并写入 retired 策略、N04 年份留未知、N23 两种年份语义、StitchFusion 定点纠错与 SubSurfaceGeoRobo DOI 不顺手修改；保留原事实表、来源与精确 5 篇名单。**（本轮按独立验收 F4 更正）** 第二节等级表的“目标 `rating` 文字”已去掉装饰性 `【】`，只保留口径自身自带的括号（如“（ESCI）”“（收录性质待核）”），并加一句“表格里的文字就是 `rating.value` 的目标值，原样写入”；出版物名称、论文题名与分区依据未改动。同时把第一节里对 arXiv 等级的重述改为指向该表，消除两处措辞分歧。
- 临时替换稿 `临时/临时计划/替换文件/论文库操作规范.md` 只加“已被正式 guide 取代、不要再次覆盖安装”的说明与指向；任务00备份未改。**（按用户指示）该文件正文的旧口径不做额外整理。**
- **（本轮补齐，详见 §2.4）** 运行仓库自带单向同步脚本 `.codebuddy/sync-from-cursor.ps1`，把最新 Cursor Rule 与三个论文库 Skill 同步到 `.codebuddy`，使 CodeBuddy 环境实际加载的副本与本轮改动一致。

## 2. 验收

### 2.1 静态与内存检查

- **实施轮实际运行（静态、可复现，本轮未重跑）：** 四个工具语法编译通过；只读加载正式索引成功（schema4 文件按 5 在内存处理，72 条记录、全部 active；三份导出与 Markdown 视图均能生成，未写入）；七个 CLI 子命令注册正确。
- **独立验收轮实际运行并确认（见 [独立验收报告](2026-10-12-paper-library-maintenance-stage1-acceptance.md) §2.3）：** 共享 helper 的首次写入→同值重复→只改 `note`→撤销语义且 `confirmed_at` 无假差异；白名单拒绝对 `doi` 写入与缺 `year_kind` 的 `year`；`status=retired` 缺 `human.retirement` 报错、合法 retirement 通过；导出只含活动记录、预印本年份带口径、未记录等级输出 `未记录`、`scope_papers` 计数；子命令集合恰为七项。
- **仅在隔离演练中覆盖、证据等级待复核（见 §2.2）：** `_apply_auto_fields` 的 human 保护、`held_lock` 的接受与拒绝分支、`_diff_lines` 的来源变化显示、UI 三个字段编辑回调、扫描器退库身份保护。这些在本轮**没有**重新运行。

### 2.2 一次隔离环境真实演练（原执行输出记录为 92 项全部通过；**独立验收尚未充分证实，待复核**）

隔离根为系统临时目录下 `paper-maintenance-<pid>\`，含临时索引、3 个人工测试 bundle（journal / conference / preprint）、恢复目录与导出目录；初始临时索引经 `store.save_index(expected_fingerprint="")` 创建，所有 plan/apply/refresh 均显式指定配套 `--index` / `--root` / `--recovery-root` / `--export-dir`。**未复制或引用任何正式论文资产，未在正式 `_index` 或论文根下建目录。**

以下是**本次执行的实际输出所记录**的通过项（同一套检查的两次运行均打印 `checks failed: 0`）：

- **metadata**：三篇真实 `maintenance-plan` → 真实计划 SHA-256 → `maintenance-apply` 一次保存；年份 / 出版物 / 论文等级分别正确写入；期刊、会议、预印本三种语义正确；预印本年份带口径，未记录的等级保持「未记录」；`schema_version` 4→5 随该次保存持久化；`next_lib_number` 不变；人工编号（含 `unverified_manual_ids=RE188`）、核心标记、标签、阅读状态、旧补充笔记完整保留。
- **幂等**：同一请求重跑 plan 为 no-op（`index-writing items: 0`），apply 不保存、revision 与 `confirmed_at` 不变；旧计划再 apply 被旧指纹拒绝。
- **UI**：`_set_manual_bibliography` 经共享 helper 写入评级草稿；只编辑一个字段不改变其他字段来源；未渲染的编辑器不清空字段；`_diff_lines` 报出“只改来源/说明”的变化且无 `confirmed_at` 假差异；`_row_for` 分开给出年份 / 出版物 / 论文等级。
- **retire**：`status=retired` + 完整 `human.retirement`；资产同卷移到恢复路径、原路径消失、正文可读；LIB、人工编号、笔记、标签与已批准书目全部保留；三份导出与活动集合不含该 LIB；管理视图显示“活动 2 / 退库 1 / 总登记 3”与退库登记小节。
- **续办与已完成**：索引已 retired 但目录仍在原位置时，续办只报告 1 个待移目录并移动目录、**不保存 JSON、不增加 revision、不改写既有 retirement**；全部移完后重跑为 no-op。
- **冲突与保护**：原位置与恢复位置同时存在时 plan 停止且不写计划文件；扫描命中退库路径或同身份新名称时在目录动作前报 `LibraryError`；`supplement-plan` 拒绝写退库记录。
- **restore**：按已有映射移回目录并经 store 提交 `previous_status`、移除 `human.retirement`；恢复后记录与其退库前状态逐字段一致，`next_lib_number` 不变；重建导出重新包含该 LIB 且源 revision 正确；管理视图回到“活动 3 / 退库 0 / 总登记 3”；已完成 restore 重跑为 no-op。
- **锁**：`index_lock` 正确产出锁对象、`save_index(held_lock=…)` 可用、外部锁被拒绝且未写入、无 `held_lock` 的普通保存仍工作。

**这项结论的证据等级（按独立验收口径据实说明）：**

1. **磁盘上没有留下可复查的成功证据。** 仍然存在的隔离根 `%TEMP%\paper-maintenance-drill-50652`（创建 19:24:47）是其中**一次中途失败运行**的残留：该次运行在父进程打印失败清单时遇到控制台 GBK 编码错误，`finally` 里的清理与“成功”打印都没有执行到底，因此这个根被留下。它的终态只到 plan：`index\PAPER_LIBRARY_INDEX.json` 为 `revision=1`、三篇 `bibliographic_overrides` 全空、无 `retirement`、全部 `present`；`index\backups\*.bak1` 与 live index 字节相同（SHA-256 前 12 位 `AD411E2F582E`，与 plan 里的 `index_fingerprint` 一致），说明 plan 之后没有再保存；`exports\` 与 `论文_retired\` 自创建后未被写入或删除过任何文件；`library\` 也没有目录移入/移出。**即：这份残留物只能证明“用 `store.save_index` 建了 schema5 临时索引 + 生成了真实的 `maintenance-plan`”，不能证明 apply 写入、退库目录移动、restore 与导出。**
2. **原始输出未留存。** 演练的 stdout 当时写到系统临时日志文件用于绕过控制台编码问题，收尾时与演练脚本一同删除，因此 92 项清单没有可独立复查的运行证据。
3. **运行版本早于最终修订。** 演练发生在 19:24–19:2x，此后仍有改动：`paper_library_ui.py`、`.cursor/skills/*`、`.cursor/rules/paper-library.mdc`、`_index/README.md`、任务03文件，以及 `paper_workflow.py` 的模块说明文字（最后一次）。即使演练确曾在别处跑通，其对象也不是最终修订版本；对最终版本而言，这些端到端结论目前只是**代码级可推断**，不是已实测。
4. 本轮未重跑演练（用户明确不要求），也**未删除**上述残留根：它是目前唯一留在磁盘上的演练中间状态证据。建议由后续独立隔离演练在同一最终代码版本上重跑后，再连同新隔离根一并清理。

因此 §2.2 的 92 项清单应读作“**本次执行的原始记录**”，而非“已经独立验证通过”。正式的工程验收口径见 §3 第 1 条。

### 2.3 正式库未被触碰（核验）

- `PAPER_LIBRARY_INDEX.json` 长度 666901、`LastWriteTime` 2026-10-12 11:12:02、SHA-256 前 16 位 `116f314a6bb137c7`，与实施前完全一致；磁盘上仍为 schema4 / revision19 / 72 条 / `next_lib_number=73`。
- `D:\0Project\origin\论文_retired` 不存在；正式论文根与 `论文_duplicates_review` 未变动。
- 未执行任务03数据维护；未提交或推送 Git。

### 2.4 `.codebuddy` 侧 Rule 与 Skill 同步（本轮补齐）

独立验收 F1 指出：本轮只在 `.cursor/` 侧改写，CodeBuddy 实际加载的 `.codebuddy` 副本仍是旧版，任务03若在 CodeBuddy 环境执行会缺维护规则与 `paper-maintenance` Skill。本轮先核对仓库自带单向同步脚本的同步范围，再运行一次并复核结果。

- **同步范围核对（先做，再写盘）：** `.codebuddy\sync-from-cursor.ps1` 只做三件事——`.cursor/rules/*.{mdc,md}` → `.codebuddy/rules/`、`.cursor/skills/*/SKILL.md` → `.codebuddy/skills/`、以及把 `.cursor/skills` 下目标缺失的附属文件补一份（不覆盖）。它**从不删除、从不重建目录**，`.codebuddy` 独有文件原样保留；正文中的 `.cursor/rules` / `.cursor/skills` 引用自动改写为 `.codebuddy`；frontmatter 去掉 CodeBuddy 不支持的 `disable-model-invocation`，但保留目标文件里的 CodeBuddy 专有字段（如 `description_zh`）；内容无变化时不写盘（幂等）。
- 先跑 `-WhatIf` 预览，实际待改对象只有：Rules 的 `paper-library.mdc`（另两份 rule 无变化，`.codebuddy` 独有的 `side-effect-authorization.mdc` 保持不动）；Skills 的 `paper-intake` / `paper-supplement` 更新、`paper-maintenance` 新增，其余四个 Skill 无变化。**未发现任何非预期覆盖或删除。**
- 实际运行后复核：`.codebuddy/rules/paper-library.mdc` 正文与 `.cursor` 版逐字符一致，frontmatter 已去掉 `disable-model-invocation`；检索 `rating` / `maintenance-plan` / `retired` / `论文等级` / `未授权` 全部命中；`.codebuddy/skills/paper-maintenance/SKILL.md` 已出现（共 7 个 Skill 目录，与 `.cursor` 侧一致）。三个 Skill 与来源的差异**只有**设计内的两类：去掉 `disable-model-invocation`、以及 `.cursor/rules/...` → `.codebuddy/rules/...` 路径改写（两个既有 Skill 还保留了各自的 `description_zh`）。`.codebuddy/rules` 仍是 4 个文件，未增未减。
- 本轮同步不涉及正式索引或论文资产。

## 3. 验收状态、遗留问题与已知边界

1. **工程验收口径（准确表述）：代码与定点检查通过；一次隔离真实演练按原执行输出记录为 92/92，但独立验收尚未充分证实；待最终代码版本的一次独立隔离演练复核。** 具体：
   - 已通过且可复查：四个工具的语法编译、只读加载正式索引成功（schema4 按 5 在内存处理、72 条记录全部 active、三份导出与 Markdown 视图可生成且未写入）、七个子命令注册正确，以及独立验收轮实际运行的共享 helper 语义 / 白名单拒绝 / 退库校验 / 导出与 `scope_papers` / 预印本口径等定点检查（清单见 §2.1）。
   - 只在隔离演练中覆盖、本轮未重跑，因而证据待复核：`_apply_auto_fields` 的 human 保护、`held_lock` 的接受与拒绝分支、`_diff_lines` 的来源变化显示、UI 三个字段编辑回调、扫描器退库身份保护（清单见 §2.1 第三条）。
   - 未获独立证实：apply 写入、退库目录移动、restore、导出重建这一整条端到端链路（原因见 §2.2 的证据等级说明）。**本轮明确不重跑**（用户要求后续由另一执行模型做一次独立隔离演练）。
2. **临时目录残留（未清理，据实说明）**：`%TEMP%\paper-maintenance-drill-50652` 仍在磁盘上，是一次中途失败运行的残留，状态只到 plan 阶段。本轮**未删除**它，因为它是目前唯一留在磁盘上的演练中间状态证据；建议在后续独立隔离演练以最终代码版本重跑、确认结果之后，连同新隔离根一并清理。除该根外没有发现其他演练残留（未留下长期 fixture、脚本或测试框架），演练脚本与 stdout 日志已在收尾时删除（这也是原始输出不可复查的原因）。
3. **索引 schema 仍是 4，磁盘尚未升级到 5**：读取已按 5 兼容处理，schema5 会随任务03第一次真实维护保存一并持久化。另注意 `paper_library.py::_sync_index` 无条件写入 `schema_version=SCHEMA_VERSION`，因此任何一次常规重扫（`paper_library.cmd`）也会把正式索引升到 schema5。切换前需结束旧 UI 写会话；旧版本工具不得继续写 schema5。
4. **正式 5 篇退库未执行**，任务03的 metadata 与 retire 数据操作整体待授权；正式库未做实际恢复演练。
5. 任务03执行前仍需重新读取 `NEW_PAPERS_IMPORT_REPORT.md`、`doc/state/current.md` 并核对最新索引；`retire` 的 `expected_title` 要求与索引题名**精确一致**，若正式题名与任务03表格有差异会在 plan 阶段停止并报告（不模糊匹配）。
6. 任务03的 `rating` 目标值已按审核意见去掉装饰性 `【】`，只保留口径自身自带的括号（如“（ESCI）”“（收录性质待核）”）；出版物名称、论文题名与分区依据未改动。
7. LIB000068 的年份按要求保持未知，接收事实写进该篇出版物 override 的 `note`；这是既定语义，不是工具失败。LIB000070 的数据集 DOI `10.1594/pangaea.975532` 不在本批授权字段内，未修改，仍待定点处理。
8. `rating` 的准确性依赖用户提供的 2025 第三方转录口径，工具只忠实记录并保留 `note`，不生成科研结论，也不把第三方转录写成官方订阅库认定。
9. 演练中未覆盖的场景：跨卷 restore、真实并发第二写者、Windows 回收站路径；这些不在本计划的最小验收范围内，也未声称已实测。
10. `.codebuddy` 侧已按仓库自带单向同步脚本补齐 Rule 与三个论文库 Skill（详见 §2.4）。

## 4. 结束状态

代码、Rule / Skill（`.cursor` 与 `.codebuddy` 两侧）、正式指南与任务03口径已一致；正式索引与正式论文资产未被触碰。**阶段一工程验收按 §3 第 1 条的口径计为“代码与定点检查通过，端到端隔离演练待最终版本复核”。** 按用户要求本轮**到此停止**：不重跑 92 项测试、不改正式索引、不移动正式资产、不执行任务03；后续由另一执行模型做一次独立隔离演练，之后再按另行授权进入阶段二。
