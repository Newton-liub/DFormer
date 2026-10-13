# 论文库遗留书目问题收尾报告（执行模型2，2026-10-12）

> 范围：仅处理用户点名的 5 项遗留问题；不做论文库重构、不重跑阶段一演练、不恢复退库论文、不联网复核分区。
> 入口：`_index\tools\paper_workflow.py` 的 `maintenance-plan` / `maintenance-apply`；索引写入只经 `paper_library_store.py`。未手工改 JSON、未修改模型/训练/历史实验。未提交、未推送 Git。
> 索引：revision 20 →（题名修正重扫）21 →（书目维护）22；当前指纹 `46692a972f289474…`。

## 1. 实际修复项

### 1.1 LIB000068 补正式出版年份（TPAMI 2025）

| 字段 | 结果 |
| --- | --- |
| `year` | **2025**，`year_kind=issue_date`，`confidence=high`，`source_url=https://pubmed.ncbi.nlm.nih.gov/39388325/`，`source_identifier=10.1109/TPAMI.2024.3476487` |
| year.note | 「TPAMI 2025, 47(2):742–754；DOI 10.1109/TPAMI.2024.3476487；2024 年接收（接收年不等于卷期年，接收事实保留）」 |
| `venue` | 值不变（IEEE TPAMI）；note 更新为「2024 接收；2025 年卷期发表（TPAMI 47(2):742–754，DOI 10.1109/TPAMI.2024.3476487）；接收年不作为卷期年」——**保留了原接收记录**，未把 2024 当作卷期年 |
| `rating` | 见 1.4（同篇一并统一为 `中科院2025大类1区·Top`） |

### 1.2 LIB000070 修正论文 DOI（与数据集 DOI 分离）

| 字段 | 结果 |
| --- | --- |
| `human.bibliographic_overrides.doi`（新增） | **10.1007/s41064-025-00361-y**，`source_type=manual`，`confidence=high`，`source_url=https://link.springer.com/article/10.1007/s41064-025-00361-y`，note：「论文正式 DOI（Springer 文章页）；数据集 DOI 10.1594/PANGAEA.975532 仍是 auto.doi 扫描值（未改动），两者不得混用」 |
| `auto.doi` | 仍为 `10.1594/pangaea.975532`（**扫描值未动**），索引/导出按人工确认值显示论文 DOI，两个 DOI 并存可区分 |
| year / venue / rating 的 note | 同步去掉已失效的「DOI 为数据集 DOI，不在本轮字段范围」表述，保留「2026 卷期年 / 2025 在线 / `auto.year=2025` 未改动」事实（字段值本身未变） |

### 1.3 LIB000054 修正题名拼写

- 错误位置核对：**来源 Markdown 第 1 行 H1**、**索引 `title`（取自 H1）**、**目录名与正文文件名**三处含 `Eficient`（目录名另有 ` - ` 代替 `:` 的转换差异）。
- 只修必要位置：把 bundle 正文 H1 改为正式题名 `Centering the Value of Every Modality: Towards Efficient and Resilient Modality-agnostic Semantic Segmentation`（来源：DBLP `conf/eccv/ZhengLZW24`），再用**常规扫描**（`paper_library.py`，无参数）刷新 `auto.*`。
- 重扫结果：68 个 bundle、0 个目录动作，仅 LIB000054 的 `title` / `normalized_title` / `md_sha256` / `body_sha256` / `bundle_signature.md_sha256` 更新（预先内存模拟确认无其他差异），revision 20 → 21。
- **未改**目录名与文件名（含旧拼写），因此 `bundle_path` / `entry_md` 未变、LIB 身份不变、无重复记录；管理视图该行的“一级目录”列仍显示旧目录名（唯一残留拼写位置，非题名字段）。
- 扫描持久性：修正后再次用同一 `_refresh_plan` + `_sync_index` 代码路径模拟重扫，**字段差异 0、human 差异 0** → 后续常规重扫不会写回错误拼写。

### 1.4 小范围统一论文等级（1 次 apply，5 项）

只处理“同一出版物、同年份”下明显不一致的表述；期刊统一为中科院分区参考写法，其余信息保留在 note：

| LIB | 出版物 | 旧 `rating` | 新 `rating` | 保留信息 |
| --- | --- | --- | --- | --- |
| LIB000058 | IEEE TPAMI | TPAMI顶刊 / CCF A；中科院1区Top（2025参考转录） | **中科院2025大类1区·Top** | note 保留「TPAMI 顶刊（CCF A）」+ 第三方转录口径 + 分区来源 URL |
| LIB000068 | IEEE TPAMI | 同上 | **中科院2025大类1区·Top** | 同上 + 「原来源：DFormer_论文补充清单_2026-10-12 第三节」 |
| LIB000064 | ECCV 2024 | 计算机视觉顶会·主会；数据集论文 | **计算机视觉顶会·主会** | note 保留「论文属性（数据集论文）与原表述」 |
| LIB000067 | ICCV 2023 | 计算机视觉顶会·主会；鲁棒性基准 | **计算机视觉顶会·主会** | note 保留「论文属性（鲁棒性基准）与原表述」 |

统一后同一出版物的 `rating` 取值唯一（ECCV 2024、ICCV 2023、TPAMI 各 1 种写法）；未做会议/分区/CCF 之间的硬换算，未猜未知分区，分区仍标为第三方参考。

### 1.5 清理阶段一残留测试目录

- 位置：`%TEMP%\paper-maintenance-drill-50652`。清理前确认：仅含 3 个人工演练 bundle（`Drill Conference/Journal/Preprint Paper …`）、临时索引/计划与空的 `论文_retired`、`exports`；**无任何正式论文资产**（无 PDF、无 `D:\0Project\origin\论文\` 路径引用，索引内 `0Project` 匹配数 0）。
- 已整目录删除并复核不存在；`%TEMP%` 下无 `paper-*` 残留。未扫描或清理其他临时目录。

## 2. 接口与规则的实际变化（因为有真实变化才同步）

- **代码最小扩展**：`paper_library_store.MAINTENANCE_FIELDS` 由 `(year, venue, rating)` 扩为 **`(year, venue, rating, doi)`**；`paper_workflow.py` 文档字符串同步。DOI 走既有 `set_bibliographic_override` + override 机制，`auto`/`human` 分工不变，UI 直接编辑器仍只渲染年/出版物/等级（其余逻辑与既有候选确认路径不受影响）。
- **同步文档**：`.cursor/rules/paper-library.mdc`（受控维护字段增加 `doi` 及其边界）、`.cursor/skills/paper-maintenance/SKILL.md`（`metadata` 允许字段与核对项）、`doc/guides/paper-library.md`、`doc/guides/external-resources.md`、`_index/README.md`（请求格式与维护字段）。
- **`.codebuddy` 同步**：运行仓库自带 `sync-from-cursor.ps1`，输出 `[update] paper-library.mdc`、`[update] paper-maintenance/SKILL.md`，其余 `[same]`；未机械改动无关 Rule/Skill。

## 3. 最小验收（以文件为准）

| 检查 | 结果 |
| --- | --- |
| 索引范围 | 72 条（**活动 67 / 退库 5**）、`next_lib_number=73`、retired 集合仍为 {004,009,011,021,022}，未变 |
| 与 rev20 备份（`backups\…bak2`）逐记录比对 | 差异**恰为 15 个字段**：LIB000054 的 5 个 auto 字段（题名修正连带哈希）+ 10 个 override 字段（058/064/067 rating；068 year/venue/rating；070 doi/year/venue/rating）；**human 其他键无增删改、无状态变化、无新增/删除记录**；根字段仅 `revision`/`updated_at` |
| 人工编号与补充笔记 | 未变化（上述比对中 human 无差异），人工编号 16 个、补充 21 篇与导出计数一致 |
| 扫描持久性 | 见 1.3：修正后模拟重扫 0 差异 |
| 管理视图 | `活动记录：67；已退库：5；总登记：72`；LIB000054 题名单元格为修正后题名；LIB000068 年份列 2025 |
| AI 导出 | 三份均源 `revision=22`、schema 5、格式 1.0.3，篇数 67 / 15 / 21；无退库论文；LIB000068 年份 2025、note 完整；LIB000070 `- DOI：10.1007/s41064-025-00361-y`；等级为统一后取值 |

## 4. 未修复 / 未处理项及原因

1. **目录名与正文文件名仍含 `Eficient`**（`bundle_path` / `entry_md` 未变）：改名会改动论文身份路径、需要路径迁移，超出“只修必要位置”，故未动；管理视图对应行的目录列因此仍显示旧目录名（题名字段已正确）。
2. **LIB000054 正文中多处小写 `eficient`**（转换器正文噪声，非题名）：属“不修正文”范围，未处理。
3. **LIB000068 的 `doi` 仍未记录**：本轮只授权补年份；其论文 DOI `10.1109/TPAMI.2024.3476487` 已写入 `year` override 的 `source_identifier` 与 note。若需在索引 `doi` 字段落位，需另行点名授权（接口现已支持）。
4. **LIB000058 的 `rating` note 未重复「来源：DFormer_论文补充清单_2026-10-12 第三节」**：该来源仍记录在同篇 `year` override 的 note 中，信息未丢失；如要求逐字段重复可再走一次 maintenance。
5. **LIB000068 的标题为全大写写法的论文 H1**（`ROBUST MULTIMODAL LEARNING …`）：本轮未授权改题名，未处理。
6. 其余 66 篇的分区、其他期刊/会议等级未复核（不在本轮范围）；第三方分区口径未升级为官方认定。

## 5. Git 改动范围（均为工作区未提交，未推送）

- 本轮修改：`.cursor/rules/paper-library.mdc`、`.cursor/skills/paper-maintenance/SKILL.md`、`doc/guides/paper-library.md`、`doc/guides/external-resources.md`；由同步脚本更新 `.codebuddy/rules/paper-library.mdc`、`.codebuddy/skills/paper-maintenance/SKILL.md`；本报告及请求/计划 JSON 位于 `临时/`（未纳入 Git）。
- 仓库外（不在 Git）：`_index\tools\paper_library_store.py`、`paper_workflow.py`、`_index\README.md`、`PAPER_LIBRARY_INDEX.json`（rev22，含 `.bak1–.bak3`）、`PAPER_LIBRARY_INDEX.md`、`exports\` 三份、`论文\...Centering…md` 的 H1。
- 未提交、未推送；`doc/state/current.md` 本轮**未更新**（状态写入留给主力模型审核后处理）。

**结论**：5 项点名的遗留问题已按要求处理完毕（题名修正附带常规重扫），范围、人工信息与派生视图均一致；发现项均为已声明或超出本轮授权的小项，不阻塞。等待主力模型只读审核。
