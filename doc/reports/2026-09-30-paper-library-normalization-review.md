# 论文资源库规范化整理与人工维护工具：上级模型审核报告

- 汇报周期：2026-09-30 本轮论文库整理；本报告只复核该轮交付，不扩展为 MMFR 研究阶段报告。
- 报告对象：上级模型审核。
- 证据边界：本地提交 `a7c5cc17c5b495f2210efa3604b05794b5fa16c4`、当前统一索引及两份实时状态文档；仓库外论文实体不由 Git 提交证明。
- 当前状态：论文库首次整理和索引工具已落地并做过定点检查；有争议的抽取版本、附属资产一致性及部分旧编号仍待人工裁决。

## 一、先看结论

原先分散于主题目录的完整论文文件夹（paper bundle，即一篇论文的正文及图、表、公式等资源所在文件夹）已整理出统一查找入口。当前索引有 **32 条 canonical 记录**：canonical 指目前供查询使用的主记录，并不表示同论文其他抽取已被证明无用。每条采用稳定 `LIB` 物理身份编号，原有经核验的 `PR`、`RE`、`AI` 编号及 `MoSA` 代号继续作为人工编号保留。日后可双击或拖入文件夹运行 `human/paper_library.cmd` 手动维护索引；没有后台服务。

本次整理**不构成**论文内容分析、所有资源逐字节相同的证明，也不改变训练、验证或 official test 的授权状态。以下数据属于本地资源管理统计，而非实验指标。

## 二、范围、事实源与判断边界

1. 目标主库为本机仓库外 `D:\0Project\origin\论文\`。整理前识别 **34 个可作论文正文的 bundle**，另有 **1 份 DFormerv2 正式 supplemental**（论文补充材料，不算独立正文）。已知历史来源 `D:\0Project\DFormer-archive-20260922\doc\paper` 当时不存在，因此仅扫描现有主库，没有全盘检索或从该旧来源补扫。34 是当次输入 bundle 数，不是新确认的不同论文数。
2. `MMFR/03_reference/PAPER_LIBRARY_INDEX.json` 是当前论文实体的机器索引；同目录 `PAPER_LIBRARY_INDEX.md` 供人浏览。每条记录分 `auto`（扫描可再生信息）与 `human`（人工维护信息），含 `auto.library_id`、完整标题、`auto.entry_md`、旧目录名 `auto.previous_folder_names`、正文哈希、简单资源计数和内容能力标志；必要时另记 `alternate_bundles`、`duplicate_review`。`MMFR/03_reference/paper-index.md` 保留旧编号映射和历史说明，其旧物理路径是整理前快照，不应当成当前正文入口。
3. Git 提交只覆盖工具、索引、规则及必要的项目入口说明。真正的论文正文、图片、表格、文档、review 目录均在仓库外，不随这份报告分发；上级若要核验其字节，需访问上述本机路径。`LIB` 是索引身份，`PR/RE/AI` 是人工文献编号，两者不可互相替代。

## 三、已完成工作及证据

### 3.1 论文文件夹与入口规范化——已完成并做过定点核验

- 形成 `LIB000001`–`LIB000032` 共 **32 条 canonical 记录**。**31 个**一级论文文件夹规范化重命名：该数来自 JSON 中非空的 `auto.previous_folder_names` 记录；相应正文入口 Markdown 文件名与新目录名同步。完整标题另存在索引中，文件夹名可能为符合路径长度约束而截断。
- 没有拆分或重写 bundle 内的 Figure、Tables、Formula、Word 等资源或论文正文内容；未把历史 PDF-only 材料硬认作 canonical 论文。另一份 DFormerv2 抽取和正式 supplemental 仍在旧主题目录，所以“canonical 论文一级扁平”**不等于**主库完全没有历史非 canonical 目录。
- `LIB000002` 记录 DFormerv2 的另一份抽取为 `alternate_bundles`，保留原处，不自动合并或选优。旧 `paper-index.md` 曾指出该组两次抽取有文件字节差异；当前主版本只是查找入口，不是质量裁决。
- `LIB000023` 对应 PR090：一份 bundle 的正文 Markdown SHA-256 为 `7a72f547f18ea82de0ef7b4ac562cb6632c5b66ccb96dbd1754bf17a7d8b783d`，并与主要资源**数量**一致，工具按该简单签名列为 exact duplicate；整包移至 `D:\0Project\origin\论文_duplicates_review\Towards Robust Multi-Modal Semantic Segmentation with Teacher-Student Framework and Hybrid Prototype__7a72f547`，并没有删除。**“exact”是工具分类名，不表示每一张图、每一张表和每个 docx 都逐字节比对相同**；历史 `paper-index.md` 对这两次独立抽取曾记为字节不同，若要得出全包等同结论须逐文件复核。

证据：统一 JSON 的 `papers`、`alternate_bundles`、`duplicate_review` 字段；浏览索引末尾的另一抽取与重复审查条目；`doc/main/MUSeg-current-status.md` 的本轮整理记录。仓库外物理文件不在提交中，上述移动事实依据本轮操作记录及其索引，远端仅凭 Git 不能重验。

### 3.2 人工编号保护与统一索引——已完成并做过定点核验

- 继承 **16 个已核验人工编号**（PR/RE/AI 与 MoSA），实际迁移中 **0 个**需覆盖原映射的 manual-ID 冲突。`RE042`、`RE053`、`RE188`、`RE447` 只进入 `human.unverified_manual_ids`，**不计入 16 个**、不绑定成已确认编号；旧目录名不等于总索引核验通过。
- 自动刷新保留 `human` 字段，不允许新 bundle 占用另一个实体的已确认人工编号；`auto` 负责位置、正文入口、简易签名和能力标志。旧历史映射继续保留在 `paper-index.md`，其顶部已标明旧路径失效、应转查统一索引。

证据：`PAPER_LIBRARY_INDEX.json` 的 `human.manual_ids` / `human.unverified_manual_ids`，`human/paper_library.py` 的索引同步与 `_record_manual_conflicts`，以及 `paper-index.md` 顶部说明。

### 3.3 手动维护入口——已完成并做过定点核验

- 新增标准库 Python 脚本 `human/paper_library.py` 和 Windows 包装入口 `human/paper_library.cmd`。无参数双击会扫描/刷新；拖单个完整 bundle 或含多个 bundle 的父目录会按路径导入；命令行支持路径及 `--dry-run`。首次整理与日常刷新分开，bootstrap 完成后不可再次误触发同一首次流程。
- 日常扫描不再重命名稳定的 canonical bundle；工具以完整文件夹为单位处理，且不改动内部正文与附属资源。人工身份冲突应阻止错误覆盖，而不是自动猜选新的 PR/RE/AI 对应关系。

证据：提交 `a7c5cc1` 的 `human/paper_library.py`、`human/paper_library.cmd`、`.cursor/rules/mmfr-research-maintenance.mdc` 和 `MMFR/00_control/{FILE_RULES.md,PACKAGE_INDEX.md}`。

## 四、定点检查与未验证范围

上一轮对已落地工具做过以下**局部检查**，本次写报告没有重新操作仓库外论文文件夹：

- 连续两次 `cmd /c "echo.|human\paper_library.cmd"`：每次扫描 33 个 bundle（含另一抽取）、维持 32 条 LIB 记录、**0 次文件操作**。这些是日常刷新幂等性的定点观察，不是全盘完整性证明。
- 将 review 目录中 PR090 的 bundle 再拖入：现有重复项被跳过，没有复制第二份。预览显示 possible duplicates 为 **0**，仅在当前工具识别规则下成立，不代表人工全文/全资产检索结果。
- 临时给首条记录的 `human.tags` 增加 `paper-library-preservation-check`，刷新后标签仍在；之后已移除该临时标签并再次刷新。纯函数冲突模拟将 `PR037` 从 `LIB000032` / Paper A 指向 `LIB000094` / Paper B，输出 `[MANUAL ID CONFLICT]`，旧映射未变，新候选人工 ID 被清除；模拟不是一次真实迁移冲突。
- 上轮 `python -m py_compile human/paper_library.py`、定点静态诊断和 `git diff --check` 未报问题。上述定点操作结果来自上一轮操作记录；本报告未另附可移植终端日志，要求独立重验时应在同一主库环境重新运行相应只读预览或局部检查。

**未运行：**完整测试套件、全仓扫描、GPU、训练、Quick-Val/Main-Val、official test、论文正文重读或资源逐文件哈希比对。本轮报告是纯文本交付，不能把索引管理检查写成研究实验结论。

## 五、请上级模型审核与裁决

1. **DFormerv2 主抽取版本：**先核对两次抽取的正文 OCR 与表格资源质量，再决定 `LIB000002` 当前主入口是否保留；在明确标准前维持另一份为 `alternate_bundles`，不删除、不合并。
2. **正式 supplemental 的挂载方式：**确定是否作为 `LIB000002` 的附属资源登记，并保留“补充材料不是另一篇正文”的身份边界；本次没有代为迁入或拼接正文。
3. **PR090 重复包的证据等级：**若审核需要证明两包每个附属文件完全相同，应逐文件比对哈希并处理与历史“字节不同”记录的冲突；在此之前只接受“正文哈希与主要资源计数相同、重复包已保留待审”的结论。
4. **旧 RE 编号：**依据权威总索引或论文原始证据分别裁决 `RE042`、`RE053`、`RE188`、`RE447` 对应题名；未经核验不得从 `unverified_manual_ids` 提升为正式 `manual_ids`。
5. **缺失旧来源：**决定是否恢复 `D:\0Project\DFormer-archive-20260922\doc\paper` 并补扫。本次“34 输入 bundle”仅覆盖实际可访问的主库，不能据此断言历史来源已完整并入。

上述均为**待裁决**，不是已完成的论文内容判断，也不自动授予新训练、评价或发布权限。裁决后以新证据修改统一索引及必要实时状态，保留本报告作为本轮快照。

## 六、交接入口、提交与边界

- 审核机器索引：`MMFR/03_reference/PAPER_LIBRARY_INDEX.json`；快速浏览：`MMFR/03_reference/PAPER_LIBRARY_INDEX.md`；旧编号与历史差异：`MMFR/03_reference/paper-index.md`。
- 下次手动接入：双击 `human/paper_library.cmd` 刷新，或将完整 bundle / 批量父目录拖至该入口；先用 `python human/paper_library.py --dry-run <bundle-or-parent-path>` 查看预计操作，再确认身份及编号。论文实体仍只在本机仓库外主库；勿从 Git 索引推断远端已有论文附件。
- 上轮交付本地提交为 `a7c5cc17c5b495f2210efa3604b05794b5fa16c4`，**未推送**。当前工作区另有 Gate-B 证据、代码索引、旧 Canvas/报告和配置等既存未提交修改，不属于该提交，也不应混入本报告的成果归属。本次报告及报告登记同样不自动提交或推送。
- 研究执行状态及权限以 `doc/main/MUSeg-current-status.md` 和 `doc/main/MUSeg-open-decisions.md` 为准；本报告只为本轮论文资产整理提供审核材料。
