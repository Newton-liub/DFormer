# 2026-10-08 临时论文批量入库与审核

> 本报告记录入库完成时的 revision 8 状态。随后经用户单独授权联网补书目，索引更新至 revision 9，结果见 [新入库论文年份与出版物补充](2026-10-08-new-paper-bibliography-fill.md)；下文保留入库时的审核证据，不将后补信息改写成入库时已知事实。

## 范围与迁移前分类

- 来源：`D:\0Project\DFormer\临时论文\`，最终预检清单为用户追加后的 8 个 bundle；共 274 个文件。
- 正式库：`D:\0Project\origin\论文\`；唯一索引：`D:\0Project\origin\_index\PAPER_LIBRARY_INDEX.json`。
- 迁移前：schema 4 / revision 7 / 37 个 canonical LIB / 1 个 alternate bundle / 1 个既有 duplicate-review 记录 / next_lib_number 38。
- 迁移前 JSON SHA256：`261573096db4926df3fe7d20e91e473527de66bee57befbc6dbb35cae881b566`。
- 分类已经在移动前完成：全新 7；全资产完全重复 0；同论文不同提取版本待审核 1；不完整或结构异常 0。
- 8 个 bundle 均有一个非空顶层正文 Markdown，以及非空 `Figure` / `Tables` / `Formula` / `Word` 目录；无零字节文件。
- 依据：正文标题、作者、明确自身 DOI/arXiv、正文 SHA256、既有 37 篇及 alternate 的身份记录、临时 bundle 互比。主题或个别作者重叠不作为同论文依据；参考文献里的 DOI/arXiv 不用作自身身份。
- 本轮不联网补书目，不改正文或实际论文资产，不调整抽取规则，不分配 PR / RE / AI 人工编号。

## 七篇确认的新论文

下面记录原始来源目录、规范化标题目录、文件数、四类目录文件数及正文哈希。规范化复用现有工具 `_safe_component` 的规则（非法路径字符替换，最大 110 字符）；主正文只改文件名，与规范化目录名一致。资源目录中的名称与内容保持不动。四类数量是实际文件数量，包含 mapping 和辅助文件，并非索引的语义资源计数。

### Provable Dynamic Fusion for Low-Quality Multimodal Data

- 来源目录：`2306.02050v2`；规范化目录同本节标题。
- 作者：Qingyang Zhang；Haitao Wu；Changqing Zhang；Qinghua Hu；Huazhu Fu；Joey Tianyi Zhou；Xi Peng。
- 文件 51；Figure / Tables / Formula / Word：4 / 12 / 32 / 2。
- 正文 SHA256：`8ba4f320cd41bdd07f0baba7ddb068767bcc51ea866a2654eaa67fdb9e76fe6a`。
- 来源文件名提供 arXiv 样式线索 `2306.02050v2`，不是扫描确认的正文标识；不据此推断出版年份。

### Learning Modality-agnostic Representation for Semantic Segmentation from Any Modalities

- 来源目录：`2407.11351v1`；规范化目录同本节标题。
- 作者：Xu Zheng；Yuanhuiyi Lyu；Lin Wang。
- 文件 35；Figure / Tables / Formula / Word：7 / 16 / 9 / 2。
- 正文 SHA256：`caab5a7500619dd43b35fa6cde0b7c751e6d665451ca0c438b6ba8de5d8ef5f1`。
- 来源文件名提供 arXiv 样式线索 `2407.11351v1`，未自动写入索引。与旧 LIB000023 / LIB000033 及本批 Reducing Unimodal Bias 有作者重叠，但标题、作者集合和方法不同，没有版本关系证据。

### When Depth Hurts: Reliability-Aware Geometry Distillation for Depth-Free RGB-D Salient Object Detection

- 来源目录：`2609.03378v1`。
- 规范化目录：`When Depth Hurts - Reliability-Aware Geometry Distillation for Depth-Free RGB-D Salient Object Detection`。
- 作者：Xuehao Wang；Jiaxin Hua；Runmei Li；Zhenyu Wu；Chenglizhao Chen；Ke Gu；Aimin Hao。
- 文件 32；Figure / Tables / Formula / Word：5 / 10 / 14 / 2。
- 正文 SHA256：`bdb08e214c5fe0b432063f7d644b5089a696a85a83661f183a5e3d8b9fcc097d`。
- 来源文件名提供 arXiv 样式线索 `2609.03378v1`，未自动写入索引；与旧显著目标检测论文只有主题关联。

### A Conflict-Guided Evidential Multimodal Fusion for Semantic Segmentation

- 来源目录：`Deregnaucourt_A_Conflict-Guided_Evidential_Multimodal_Fusion_for_Semantic_Segmentation_WACV_2025_paper`；规范化目录同本节标题。
- 作者：Lucas Deregnaucourt；Hind Laghmara；Alexis Lechervy；Samia Ainouz。
- 文件 34；Figure / Tables / Formula / Word：4 / 8 / 19 / 2。
- 正文 SHA256：`13bf63bd8027a32657667ac2ba7c58f6bf13f6cdf51c3044aaaff5068ec3d023`。

### Delivering Arbitrary-Modal Semantic Segmentation

- 来源目录：`Zhang_Delivering_Arbitrary-Modal_Semantic_Segmentation_CVPR_2023_paper`；规范化目录同本节标题。
- 作者：Jiaming Zhang；Ruiping Liu；Hao Shi；Kailun Yang；Simon Reiß；Kunyu Peng；Haodong Fu；Kaiwei Wang；Rainer Stiefelhagen。
- 文件 33；Figure / Tables / Formula / Word：7 / 14 / 9 / 2。
- 正文 SHA256：`acb2d56281e3d13df9cef6f3a9c0d8ee87189d08c456a00c504efb5b88881b0e`。

### Reducing Unimodal Bias in Multi-Modal Semantic Segmentation with Multi-Scale Functional Entropy Regularization

- 来源目录：`Zheng_Reducing_Unimodal_Bias_in_Multi-Modal_Semantic_Segmentation_with_Multi-Scale_Functional_ICCV_2025_paper`；规范化目录同本节标题。
- 作者：Xu Zheng；Yuanhuiyi Lyu；Lutao Jiang；Danda Pani Paudel；Luc Van Gool；Xuming Hu。
- 文件 35；Figure / Tables / Formula / Word：8 / 14 / 10 / 2。
- 正文 SHA256：`9228df2896c1d24095fa1543ecd8390d58daa5cc8c6cdbfaf3459682984b20f7`。
- 与旧 LIB000033 共享部分作者，但题目与作者集合不同，无同论文版本证据。

### CoRiM: Conflict-driven Risk Minimization for Dynamic Multimodal Fusion

- 来源目录：`Zou_CoRiM_Conflict-driven_Risk_Minimization_for_Dynamic_Multimodal_Fusion_CVPR_2026_paper`。
- 规范化目录：`CoRiM - Conflict-driven Risk Minimization for Dynamic Multimodal Fusion`。
- 正文作者：Shihao Zou；Wei Wei；现有严格抽取器未接受作者行，索引自动作者为空，仅记录，不放宽规则。
- 文件 33；Figure / Tables / Formula / Word：6 / 8 / 16 / 2。
- 正文 SHA256：`705a1812013a06e9da3f65345e3c90321bc1005b9781ccc97df0433c93e5659d`。
- 与本批 Provable Dynamic Fusion / A Conflict-Guided Evidential 只有主题关联，未见同论文身份线索。

## 保留的人工审核项

- 原目录：`D:\0Project\DFormer\临时论文\202610081158188243\`，不迁入正式库，也不移动到完全重复审核目录。
- 新旧标题完全一致：UMIS-Mine: Robust RGB-D Instance Segmentation in Visually Degraded Underground Mine Scenes。
- 五位作者完全一致：Kaiyu Li；Feiteng Han；Yu Wang；Ming Xue；Xiao Zheng。
- 对应旧记录：`LIB000032`；正文入口在 `UMIS-Mine - Robust RGB-D Instance Segmentation in Visually Degraded Underground Mine Scenes`。
- 新旧正文 SHA256 相同：`33a0ec8b48549f9997fe8642fffe797ec2c4cfc7b8739b8f6b33410764c720b1`。
- 双方都是 21 个实际文件；正文、7 张 Figure、5 张公式图片、2 张表格图片、image_mapping 共 16 个文件字节一致。
- 5 个附属文件 SHA256 不同：`Formula/*_formula.md`、`Tables/Table_1.xlsx`、`Tables/Table_2.xlsx`、`Word/*_图表.docx`、`Word/*_图表汇总.md`。不据此宣称科学内容发生修改；可能包含提取命名、容器元数据或资产差异，尚未人工裁决。
- 双方正文自身 DOI/arXiv 均未被抽取器确认。旧 `human.bibliographic_overrides.year/venue` 的来源标识为 `10.1007/978-981-92-3432-5_1`，这只是既有人工证据来源，不将它冒称已存于 `auto.doi` 或 override.doi。
- 现有工具的“exact”仅检查正文哈希和资源数量，会把本项判为 exact。本次全文件核验发现上述差异，因此按更保守的“同论文不同提取 bundle 待人工审核”保留；不交给扫描器自动合并或新增 duplicate-review。

## 执行与后审核

### 实际执行

1. 只对确认的 7 篇新论文做同卷目录移动和主 Markdown 文件名规范化，未复制出第二份正文；Figure / Tables / Formula / Word 内文件名和字节均未修改。
2. 移动前重核全部 274 个来源文件 SHA256，确认清单与预检一致；移动后核验迁入的 253 个文件 SHA256 不变。
3. 通过现有正式入口先执行 `D:\0Project\origin\_index\tools\paper_library.cmd --dry-run`。输出为 45 个扫描 bundle，无目录操作、无 warning。45 = 44 个 canonical + 1 个既有 alternate，不能把 bundle 数误写为论文实体数。
4. 用同一工具现有 `_refresh_plan` / `_sync_index` / store 的只读内存结果检查：原 37 篇记录完整不变，human / bibliographic_overrides / supplement / alternate_bundles / duplicate_review 不变，schema 仍为 4，新增 7 个独立条目，人工编号为空，无编号冲突；JSON 在 dry-run 后哈希未变。
5. 安全核验后执行 `D:\0Project\origin\_index\tools\paper_library.cmd --root D:\0Project\origin\论文`，由共享 store 正式写入一次索引，未手改 JSON。

### 新增 LIB（由正式扫描生成）

- `LIB000038` — A Conflict-Guided Evidential Multimodal Fusion for Semantic Segmentation。
- `LIB000039` — CoRiM: Conflict-driven Risk Minimization for Dynamic Multimodal Fusion。
- `LIB000040` — Delivering Arbitrary-Modal Semantic Segmentation。
- `LIB000041` — Learning Modality-agnostic Representation for Semantic Segmentation from Any Modalities。
- `LIB000042` — Provable Dynamic Fusion for Low-Quality Multimodal Data。
- `LIB000043` — Reducing Unimodal Bias in Multi-Modal Semantic Segmentation with Multi-Scale Functional Entropy Regularization。
- `LIB000044` — When Depth Hurts: Reliability-Aware Geometry Distillation for Depth-Free RGB-D Salient Object Detection。

### 入库后审核结果

- 最终：8 个原始 bundle = 新入库 7 + 全资产完全重复 0 + 同论文不同提取版本待审核 1 + 结构异常 0；正式论文实体 37 → 44。
- schema：4 → 4；revision：7 → 8；next_lib_number：38 → 45。新增 LIB 与 dry-run 预测的路径身份一致，44 个 LIB 均 present。
- 原 `LIB000001`–`LIB000037` 的完整序列化记录逐项相等，既有 LIB、human、override、supplement、auto 与 alternate 全部保持不变；有人工书目 override 的旧篇数仍为 32。
- 既有 alternate 仍 1 个；既有 duplicate_review 仍 1 条且内容不变。UMIS-Mine 待审核项没有被写成 alternate 或再次入库。
- 新 7 篇 manual_ids / unverified_manual_ids 全部为空，无 PR / RE / AI 分配，无人工编号冲突，未发生新条目之间或与旧条目的错误合并。
- 标题 7/7 正确来自正文 H1；自动作者 6/7；摘要 7/7。CoRiM 作者自动值为空，正文已直接读到 Shihao Zou / Wei Wei，仅在本报告记录。
- **缺摘要：无。缺年份和出版物：LIB000038–LIB000044 全部 7 篇。DOI / arXiv：新增 7 篇自动值均为空。** 不以来源文件名中的 CVPR / ICCV / WACV 或 arXiv 日期推断正式 publication year，不把参考文献里的标识当作自身标识；未联网或自动补 override。
- 正式刷新后的只读重扫显示，经过共享 store 默认字段规范化的完整文档与当前已加载文档相等，LIB/路径映射不变，无 warning、无人工编号冲突。新条目部分 human 可选字段在磁盘上省略、由既有读取层补默认值，这是现有 schema 4 行为，不改工具、不再为此写第二次 revision。
- `PAPER_LIBRARY_INDEX.md` 已由正式工具重建；实际内容与同一 store 从已保存索引重新渲染的内容完全相同，包含 44 篇及 7 个新增 LIB。
- UI 实际显示总数 44；逐个搜索 LIB000038–LIB000044 均匹配 1 篇并显示正确标题。未编辑人工字段，保存按钮始终禁用，UI 未写索引。
- `backups` 当前 revision 为 7 / 6 / 5；bak1 的字节哈希等于本次迁移前 JSON，可作为本次索引恢复材料。正式库目录移动不随索引备份自动回滚。
- 最终 JSON SHA256：`790bd9e7b96865ef22d2e00bdb2143e80a4170790a29058c6f9c46169196990b`。
- 最终 Markdown SHA256：`7dc3a14ede3239438f58719f60bd9782ee1f3f29e3b652509ef9d43ca2b62b31`。
- 对 8 个来源共 274 个文件的最终核验均与预检 SHA256 一致（新入库 253 + 临时保留 21）；另核验旧 LIB000032 全 21 文件未变。

### 临时目录剩余与验证边界

- `临时论文\` 只剩 `202610081158188243\` 及其原 21 个文件，原因是与 LIB000032 同论文但 5 个附属文件哈希不同，等待人工决定；主正文和实际资产未改写，原目录名也保留。
- 剩余文件清单（均相对上述目录）：正文 `202610081158188243.md`；Figure 的 `Fig.1.jpg`–`Fig.7.jpg` 与 `image_mapping.json`；Formula 的 `202610081158188243_formula.md` 和 `formula_1.jpg`–`formula_5.jpg`；Tables 的 `Table_1.jpg` / `Table_1.xlsx` / `Table_2.jpg` / `Table_2.xlsx`；Word 的 `202610081158188243_图表.docx` 与 `202610081158188243_图表汇总.md`。
- 人工可进一步核验资产差异后决定按重复归档、作为 alternate 保留，或其他处置。本次不代为决定、不覆盖、不合并、不删除。
- 仅执行用户授权的论文目录检查、正式 CLI dry-run / 刷新、字段与资产差异核验及本地 UI 只读验收。未新建验证脚本或测试文件，未运行项目测试套件、训练或评价；本次没有模型代码改动，完整测试不覆盖论文入库的主要风险。
