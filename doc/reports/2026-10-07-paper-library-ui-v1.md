# 论文索引可视化工具 V1 实施报告

> 日期：2026-10-07。执行依据：`临时/优化后的计划.md`（基于已审核的《2026-10-07-论文索引可视化管理工具实施规划》）。本轮按计划完成 V1 并停止，不继续开发 AI 人工处理、逻辑合并或其他 V1.1 功能。
> 运行代码全部在仓库外的外部资产侧 `D:\0Project\origin\_index\tools\`，未改动论文资产、模型代码或研究代码。

## 1. 实际新增 / 修改文件

新增（`D:\0Project\origin\_index\tools\`）：

| 文件 | 作用 |
| --- | --- |
| `paper_library_store.py` | 共享数据层：加载 / 校验 / v1→v2 内存迁移、字段保留、人工编号规范化与冲突检查、文件锁、指纹检查、原子保存、最近备份、Markdown 视图生成 |
| `paper_library_ui.py` | 单个 Streamlit 应用：三个视图 + 详情编辑 + 会话草稿 + 保存 / 重置 / 刷新 |
| `paper_library_ui.cmd` | 可视化工具启动器（绑定本机 `127.0.0.1`，使用普通工具 Python 环境） |
| `requirements-ui.txt` | 实际依赖版本记录（streamlit 1.51.0、filelock 3.20.0、pandas 2.3.3，均为复用已有环境，未安装新包） |

修改：

| 文件 | 改动 |
| --- | --- |
| `tools\paper_library.py` | 索引读写改由 store 承担；新增 v2 兼容与数据安全定点修复（见第 4 节） |
| `tools\paper_library.cmd` | 明确使用工具 Python 环境（索引保存需要 filelock），不再依赖 `py -3` |
| `PAPER_LIBRARY_INDEX.json` | 一次性迁移到 `schema_version` 2，新增 `revision` / `updated_at` 与每篇的 V1 human 字段 |
| `PAPER_LIBRARY_INDEX.md` | 由新渲染器重建，声明自己是 JSON 只读视图，并补充核心 / 阅读状态 / 补充列 |
| `_index\README.md` | 更新真源说明、字段所有权表、启动方式、保存协议、备份与 AI 写入约定 |
| `doc\guides\external-resources.md` | 小幅补充 store / UI 入口与 schema 变化 |

未改动：论文目录、正文、Figure / Tables / Formula / Word、`code-index.md`、`pending-and-duplicates.md`、`paper-index.md`、DFormer 模型与研究代码。运行环境：复用已存在的普通工具环境 `D:\2Env\anaconda\python.exe`（Python 3.13.9），**没有向 GPU 训练环境 `dformer` 安装任何依赖**。

## 2. JSON schema 实际变化（v1 → v2）

根对象新增：

- `schema_version`：1 → 2。
- `revision`：整数，每次成功保存 +1；界面显示磁盘版本。
- `updated_at`：带时区的 ISO 8601，仅在成功保存时写入。

单篇 `human` 新增（旧论文一律写入安全缺省，不伪造）：

- `primary_manual_id`：列表主要显示的人工编号，缺省 `null`。
- `tags`：字符串列表，缺省 `[]`。
- `is_core`：布尔，缺省 `false`。
- `reading_status`：`abstract_only`（缺省）/ `needs_full_text` / `full_text_completed`。
- `supplement`：缺省 `null`；写入后为 `{content_md, updated_at, source_note, sources}`。

保持不变：`library_root`、`next_lib_number`、`bootstrap_complete`、`papers[].auto.*`、`human.manual_ids`、`human.unverified_manual_ids`、`status`、`alternate_bundles`、`duplicate_review`，以及所有未知扩展字段（磁盘写入采用“已知键在前、扩展键保留”的稳定顺序）。

本轮未做：`review_items`、`merged_into`、`manual_id_history`、`bibliographic_overrides`、`ai_summary`。

## 3. 页面已具备的功能

- 顶部：论文数、磁盘 revision、未保存修改数量、刷新索引（只重读磁盘）、搜索结果计数。
- 视图：论文库 / 核心论文（`is_core`）/ 待全文补充（`reading_status == needs_full_text`）。
- 搜索：LIB、人工编号（含规范化形式，`PR24` 能找到 `PR024`）、标题、年份、标签；筛选：阅读状态、资源状态、只看已有补充。
- 列表：人工编号、LIB、标题、年份、核心、阅读状态、有无补充、资源状态。
- 论文选择：显式下拉框（按 LIB 稳定映射）。用表格行选择时依赖在 canvas 单元格内点击，实测不可靠，按计划许可改用独立选择控件，未引入前端插件。
- 详情：LIB 只读、当前全部人工编号与未核验编号、主要人工编号编辑、标题 / 作者 / 年份 / 出版物 / DOI / 摘要（缺失显示“未知”）、核心标记、标签、阅读状态、supplement 大文本编辑（460 高）+ Markdown 预览、来源说明、最近保存时间、正文与资源绝对路径、“在资源管理器中打开目录 / 打开正文”。
- 批量编号编辑：一次改多篇，只改草稿；同一篇在详情与批量区之间的编辑不会互相覆盖，输入框内改动（含改回原值）都能正确落到草稿。
- 保存：差异预览 → 确认写入 → 校验 → 加锁 → 指纹复核 → 原子替换 → 备份轮转 → 重建 Markdown；成功后重建 base/draft 并重设控件代次。
- 重置未保存修改、刷新前提示（放弃草稿刷新 / 取消）、下载未保存草稿 JSON、高级区重建 Markdown 视图。

## 4. 旧脚本的必要兼容修改（`paper_library.py`）

1. **索引读写交给 store**：`_load_index` 委托 `store.load_index`（支持 v1/v2、结构校验、LIB 与路径唯一性、`next_lib_number` 修正）；`_write_indexes` 改为 `store.save_index` + `store.write_markdown_view`；`_render_markdown` 改为调用 store。删除了脚本内重复的 `_internal_record` / `_external_record` / `_validate_human` / `_validate_bundle_record` 与 `_write_text_atomic`，避免出现第二套读写逻辑。
2. **不再清空 `year` / `authors`**：`_apply_auto_fields` 对扫描无法可靠判定的字段（`PRESERVE_WHEN_UNKNOWN`）在扫描结果为空时保留旧值。
3. **不再从旧种子回灌人工编号**：`_sync_index` 的“同一论文不同抽取版本”分支不再把 bundle 携带的历史编号追加到已存在 LIB 的 `manual_ids`（改为警告）；`_parse_seed_mappings` 使用 store 的规范化形式。已实测“用户删掉编号后重新扫描不会回灌”。
4. **编号正则兼容四位及更长**：`MANUAL_ID_RE` 与标题可信度检查改为 `\d{3,}`；`_record_manual_conflicts` 按“前缀 + 数值”比较，`PR024` 与 `PR0024` 视为同一编号。
5. **不降级 v2**：`SCHEMA_VERSION` 取自 store（=2）；读到未知 / 更高 `schema_version` 时拒绝写入。
6. **CLI 保存也做并发保护**：`main()` 在读取时记录磁盘指纹，写入前由 store 复核；期间被其他程序改动则报错退出，索引保持原样。

## 5. 最小验收结果（实际运行过）

隔离数据位置：`%TEMP%\pl_v1_accept\`（store/扫描）、`%TEMP%\pl_ui_apptest\`（页面）。真实论文库只被读取，真实索引只在最后一节迁移时写入一次。

数据层与扫描（全部 PASS）：37 篇加载无校验问题；一次保存完成两篇编号交换 + 写入 supplement + 阅读状态 + 核心 + 标签，并保留未知扩展字段；`.bak1` 等于迁移前原始字节；过期指纹被拒绝且文件未变；外部改动后拒绝覆盖且外部内容仍在；跨 LIB 编号冲突（`PR0024` vs `PR024`）阻止保存且文件未变；删除某篇编号后重新扫描不回灌，supplement / 阅读状态 / 未核验编号 / `alternate_bundles` / `duplicate_review` 均未丢失，`year` / `authors` 未被清空；未知新版本 schema 被拒绝。

页面（Streamlit `AppTest` 驱动真实 `paper_library_ui.py`，全部 PASS）：打开无异常、37 篇；修改一篇编号不保存时磁盘完全未变（仍是 v1、无备份、无 md）；两篇编号一次保存后 revision=2、`.bak1` 等于迁移前字节、重新打开页面仍正确；构造编号冲突后保存被阻止且磁盘未变；写 supplement + 来源 + “已全文补充” + 核心标记保存后重新打开内容仍在；修改 supplement 后重置回到已保存内容且已保存补充未被删除；页面编辑期间外部改写 JSON 后保存被拒绝、外部内容与已有补充都未被覆盖、草稿仍在会话中；有未保存修改时刷新先提示，放弃草稿刷新后回到磁盘内容。

浏览器端另做了一次真实渲染验证：列表、详情、筛选、状态栏与保存 / 重置按钮均正常显示，重置流程（提示 → 确认 → 恢复原值）在真实浏览器中走通。

真实索引迁移（`_index\PAPER_LIBRARY_INDEX.json`）：先 `--dry-run` 预览（38 个受管 bundle，无文件系统动作），再执行一次真实扫描迁移。逐字段核对结果：37 篇 LIB 集合与顺序不变；每篇 `auto` 事实、人工编号、未核验编号、状态、`alternate_bundles`（含交替版本路径与其 human）逐字段一致；`duplicate_review`、`next_lib_number`、`library_root`、`bootstrap_complete` 不变；`schema_version`=2、`revision`=2、`updated_at` 已写入；英文编号 16 个、`MoSA` 等特殊编号仍在；无人工编号冲突；`.bak1` 即迁移前的 v1 原文；迁移后再跑一次 `--dry-run` 读取正常。

未运行的检查：GPU、模型训练、DFormer 全仓测试、全库逐字节哈希、物理文件操作。

## 6. 未解决风险与已知边界

- **未保存草稿不跨重启**：浏览器刷新、断连或应用重启后草稿丢失（符合计划边界），只提供“下载未保存草稿”。已验证：刷新页面会重新读取磁盘。
- **目录改名不会自动重新关联**：文件夹改名后，扫描把它当成“重复候选”记入 `duplicate_review`，旧记录仍指向旧路径。路径重新关联属于 `review_items` 与 V1.1。
- **文件锁是合作式保护**：绕过 store 直接整文件写 JSON 的第三方操作仍可能造成并发覆盖；已在 `README.md` 的使用约定中禁止。
- **`revision` 起始值**：旧文件没有 `revision`，V1 迁移保存后显示 2（载入按 1 计）；仅影响显示，不影响数据。
- **`paper-index.md` 仍是过时快照**：本轮只停止从中回灌编号，未重写它；编号历史依据仍需回归档查。
- **页面表格行选择依赖 canvas 点击**：已按计划改用独立下拉选择控件；数据表本身仍可排序 / 筛选，但不作为唯一选择入口。
- **`bundle_signature` 与资源统计仍是 v1 语义**：只证明正文哈希 + 资源数量，不能证明每个 xlsx / jpg 内容一致。

## 7. 本轮明确未实现（留到 V1.1 及以后）

`review_items` 审核系统与“待人工处理”视图；AI 推荐结构界面与接受 / 忽略 / 延期；AI Summary 五字段；`bibliographic_overrides` 与人工校正界面；`manual_id_history` 与历史编号搜索；`merged_into` 与逻辑归并、多 LIB 合并链；物理文件夹合并 / 移动 / 删除；内部 ID 修复界面；草稿导入；备份恢复 UI；自动全文阅读；内置大模型 API；重复论文自动合并；SQLite；React / Node；Docker；登录权限系统；大规模测试框架。

开发中发现的未来需求只记录，未实现。
