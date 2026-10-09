# 论文库 UI V1.0.1：书目信息抽取补丁报告

> 日期：2026-10-07。执行依据：用户下达的《论文库 UI V1.0.1：书目信息抽取补丁》。范围只到“从正文 Markdown 高置信度抽取 → 写入 `auto` → 页面显示”，未接入历史 registry、未做人工审核界面、未开发 V1.1。

## 1. 修改的文件

| 文件 | 改动 |
| --- | --- |
| `_index\tools\paper_library.py` | 新增受限书目抽取：`Bibliographic` 结果结构与 `_extract_bibliographic()`，含 `_extract_abstract` / `_extract_authors` / `_extract_year` / `_extract_venue` / `_extract_identifiers`（重写）/ `_split_author_line` / `_reference_format_block` 与配套常量；`Bundle` 与 `Bundle.record()` 增加 `year`/`authors`/`venue`/`abstract`/`bibliographic_sources`；`PRESERVE_WHEN_UNKNOWN` 扩展到 6 个字段；`_apply_auto_fields()` 对 `bibliographic_sources` 做逐字段幂等合并；`_analyse_bundle()` 调用新抽取 |
| `_index\tools\paper_library_store.py` | `SCHEMA_VERSION` 2 → 3（`SUPPORTED_SCHEMA_VERSIONS` = 1/2/3，旧版仍可迁移，未知更高版本继续拒绝写入）；`validate_document()` 增加 `authors`/`year`/`venue`/`abstract`/`bibliographic_sources` 类型校验；`render_markdown()` 增加“作者”列与“书目信息抽取状态（auto 字段）”表 |
| `_index\tools\paper_library_ui.py` | 仅 1 行最小修复：列表“年份”列强制转字符串。原因是新写入的年份是整数，与 `未知` 混在一列会让 Streamlit 的 pyarrow 序列化报 `Expected bytes, got a 'int' object`，导致列表列渲染失败。未改任何布局 |
| `_index\README.md` | 字段所有权表补充 4 个新 `auto` 字段与 `bibliographic_sources` 的取值规则，并明确历史材料不自动回填 |
| `PAPER_LIBRARY_INDEX.json` / `.md` | 由一次真实扫描自然写入（`.bak1` 为扫描前的 v2 原文） |

## 2. schema 变化

`schema_version` 2 → 3。相对 v2 只新增 `auto` 侧字段，`human.*`、`status`、`alternate_bundles`、`duplicate_review` 结构完全未动：

- `auto.authors`：字符串列表（散文姓名，按论文原始写法保留大小写，如 `WEI ZHAO`；机构角标与 IEEE 会员头衔已剥离）。
- `auto.year`：整数或 `null`。
- `auto.venue`：字符串或 `null`。
- `auto.abstract`：单段纯文本（空白折叠）或 `null`。
- `auto.bibliographic_sources`：`{字段: {source, method, confidence, extracted_at}}`，`source` 是 bundle 正文 Markdown 文件名，`confidence` 取 `high`/`medium`。

## 3. 抽取结果（真实索引，扫描后实测）

37 篇中：

| 字段 | 成功 | 说明 |
| --- | --- | --- |
| 标题 | 37/37 | 未变（仍全部来自 H1） |
| `authors` | 31 | 6 篇保持未知（见第 4 节） |
| `abstract` | 34 | 3 篇保持未知（见第 4 节） |
| `year` | 11 | 其中 8 篇 `high`（出版日期/版权行/ACM reference format），3 篇 `medium`（接收日期） |
| `venue` | 3 | `Pattern Recognition`（LIB000005）、`ACM International Conference on Multimedia`（LIB000029、LIB000034） |
| `doi` | 7（**新补 2**） | 新增 `LIB000029` `10.1145/3664647.3681698`、`LIB000034` `10.1145/3767308.3836234`；原 5 篇值未变 |
| `arxiv` | 0 | 37 篇正文都没有可用的 arXiv 编号（LIB000033 只出现 `arxiv` 字样，无编号），保持未知 |

摘要长度 1016–2277 字符，全部落在 300–8000 的阈值内。

## 4. 仍保持 unknown 的典型论文与原因

作者（6 篇）：

- `LIB000029`、`LIB000034`：ACM 版式为“作者／单位／邮箱”逐行交错，单行无法代表完整名单，按“不可错抽”拒绝。
- `LIB000026`：作者行内联单位（`Krishna Jaganathan, Patricio Vela Georgia Institute of Technology {…}@gatech.edu`），边界歧义 → 拒绝。
- `LIB000016`：`Jana Koseckˇ a´` 是 OCR 断字后的姓名，无法确认末词属于姓名还是单位 → 拒绝。
- `LIB000036`：`<sup>` 角标内容被 OCR 抹掉，`Sijie Li Chen Chen` 已粘连成“一个人”，触发“同一元素重复词”门禁 → 拒绝（此前会写出错误值）。
- `LIB000001`：末两位作者被 OCR 粘成 `PenghaoWang` / `& KehuYang`，作者区内无机构边界可切 → 拒绝。

摘要（3 篇）：

- `LIB000001`（Nature Data Descriptor）：摘要没有任何标记，按方案保持未知，不用“标题后第一段”猜。
- `LIB000019`（MDPI）：摘要区只有 `## Highlights` 等标记，没有 `Abstract` 标记 → 未知。
- `LIB000020`（Emerald）：`## Abstract` 后实际只有 Keywords 与 Paper type，被否定词表拦下 → 未知（这正是必须避免的误抽）。

年份未知 26 篇：正文没有可靠出版日期字段（如 `LIB000002`、`LIB000023`、`LIB000037` 全篇头部无日期），按方案保持未知，未从随机四位数字、引用年份或 DOI 年份推断。

## 5. 人工抽查结果（9 篇）

| LIB | authors | year（方法/置信度） | venue | 摘要开头 |
| --- | --- | --- | --- | --- |
| LIB000002（PR070） | Bo-Wen Yin, Jiao-Long Cao, Ming-Ming Cheng, Qibin Hou | — | — | Recent advances in scene understanding… |
| LIB000007 | WEI ZHAO, CHEOLKON JUNG, JAEKWANG KIM | 2023（published_date/high） | — | Image-guided depth completion aims to… |
| LIB000023（PR090） | Jiaqi Tan, Xu Zheng, Yang Liu | — | — | Multimodal semantic segmentation (MMSS) faces… |
| LIB000037 | Bo-Wen Yin, Jiao-Long Cao, Dan Xu, Ming-Ming Cheng, Qibin Hou | — | — | We explore the potential of pretrain-and-finetune… |
| LIB000029 | —（拒收） | 2024（reference_format/high） | ACM International Conference on Multimedia | Existing RGB-D semantic segmentation methods… |
| LIB000003 | Minjie Wan, Yirong Chen, Pengqiang Ge, Xiaofang Kong, Guohua Gu, Qian Chen | — | — | Depth completion is a method for scene depth… |
| LIB000020 | Guanglei Liang | — | — | —（未知） |
| LIB000005 | Haipeng Guo, Junbao Li, Huanyu Liu | 2026（copyright/high） | Pattern Recognition | Depth completion aims to recover dense depth maps… |
| LIB000033 | Chenfei Liao, Kaiyu Lei, Xu Zheng, Junha Moon, Zhixiong Wang, Yixuan Wang, Danda Pani Paudel, Luc Van Gool, Xuming Hu | — | — | Multi-modal semantic segmentation (MMSS) addresses… |

抽查的 31 篇作者列表逐条核对与正文一致，未发现脏数据；`LIB000027`（11 人）、`LIB000022`（7 人）、`LIB000028`（7 人）等长名单也正确。

## 6. 是否发现误提取

- 开发过程中确实出现过并被门禁拦下：`Neural Networks` 被从摘要里“convolutional neural networks”误判为期刊（已用大小写敏感 + 短行/venue 线索门禁修掉）；`Pattern Recognition` 同类；`Central China Normal`（单位片段）曾被当成作者；`Sijie Li Chen Chen`（OCR 粘连）曾被当成一个人。以上都已在最终版本中拒绝，最终写入的 37 篇数据里**未发现误提取**。
- 唯一带不确定性的值：`LIB000017` 的 `year=2026` 来自 “Accepted: 24 August 2026”（`accepted_date_line` / `medium`）。若该刊 2027 年才正式出版，该年会偏早一年；已在 `sources` 里标明方法与置信度，便于以后人工改。

## 7. 验收（真实运行）

隔离副本（`%TEMP%\pl_bib_accept\`）先跑通全部 18 项检查后再写真实索引：LIB 集合与顺序不变、`human`/`supplement`/`alternate_bundles`/`duplicate_review` 逐项不变、标题 37/37、作者 ≥30、邮箱/图注/单位未进入作者、五类摘要标记各自命中、`LIB000020` 未把 Keywords 当摘要、`LIB000001` 保持未知、两篇 ACM DOI 补上、年份均在 1990–2027 且随机四位数字未被采用、venue 只来自明确名称、`sources` 结构完整、重扫不清空且来源时间幂等；Streamlit 用 `AppTest` 确认详情页显示作者/年份/DOI（列表年份列的类型问题即在此步发现并修复）。

真实索引扫描后再逐项复核：`schema_version`=3、37 篇 LIB 与顺序不变、`human` 逐篇不变、`title`/`content_capabilities`/`body_sha256` 不变、`duplicate_review` 与 `alternate_bundles` 不变、`.bak1` 为扫描前 v2 原文、结构校验与人工编号冲突检查通过、所有抽取字段都带来源、Markdown 视图含作者列。真实性检查：真实论文库只被读取，没有改动任何 bundle。

## 8. 留给 V1.0.2 / V1.1

- 作者仍缺 6 篇，都属版式难点：ACM 逐行作者块（可从 “ACM Reference Format” 段取作者，需新增受限规则）、单位内联、OCR 粘连姓名；建议按需要单独立项，不要放宽现有门禁。
- 摘要仍缺 3 篇：MDPI 的 `Highlights` 结构与 Nature 的无标记结构，需要在有人工确认通道后再处理。
- 年份/出版物的覆盖率受数据本身限制：37 篇里正文真正写明刊名的只有 3 篇。要补齐只能走“历史 registry 候选 + 人工确认”（本轮按规定未接入）。
- 人工 metadata 校正界面（`bibliographic_overrides`）仍属 V1.1。
