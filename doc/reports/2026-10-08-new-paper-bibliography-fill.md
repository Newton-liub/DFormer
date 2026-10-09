# 2026-10-08 新入库论文年份与出版物补充

## 授权与范围

- 用户在本轮明确授权联网核验新入库的 7 篇论文，并补充已确认的年份和出版物。
- 唯一索引仍在 `D:\0Project\origin\_index\PAPER_LIBRARY_INDEX.json`；本仓库不复制索引。
- 仅写 `LIB000038`–`LIB000044` 的 `human.bibliographic_overrides.year/venue`；不改正文、资源、自动抽取规则或 `auto.*`，不追加人工编号，不修改作者、DOI/arXiv、摘要、标签、阅读状态或 supplement。
- 原批量入库完成于 revision 8；本次只补书目，正式库仍为 44 篇，不重新接入或移动论文。

## 逐篇核验与补充

### LIB000038 — A Conflict-Guided Evidential Multimodal Fusion for Semantic Segmentation

- 已写：年份 **2025**，出版物 **WACV 2025**。
- 来源：[CVF 官方 WACV 论文页](https://openaccess.thecvf.com/content/WACV2025/html/Deregnaucourt_A_Conflict-Guided_Evidential_Multimodal_Fusion_for_Semantic_Segmentation_WACV_2025_paper.html)。标题、四位作者及摘要对应本地正文；BibTeX 为 `Deregnaucourt_2025_WACV`，会议年 2025，pp. 1373–1382。
- 出版物原文：Proceedings of the Winter Conference on Applications of Computer Vision (WACV)。

### LIB000039 — CoRiM: Conflict-driven Risk Minimization for Dynamic Multimodal Fusion

- 已写：年份 **2026**，出版物 **CVPR 2026**。
- 来源：[CVF 官方 CVPR 论文页](https://openaccess.thecvf.com/content/CVPR2026/html/Zou_CoRiM_Conflict-driven_Risk_Minimization_for_Dynamic_Multimodal_Fusion_CVPR_2026_paper.html)。标题及 Shihao Zou / Wei Wei 与本地正文一致；BibTeX 为 `Zou_2026_CVPR`，会议年 2026，pp. 37821–37830。
- 正文作者已知但不在本次年份／出版物授权范围，未额外补作者 override；`auto.authors` 仍为空。

### LIB000040 — Delivering Arbitrary-Modal Semantic Segmentation

- 已写：年份 **2023**，出版物 **CVPR 2023**。
- 来源：[CVF 官方 CVPR 论文页](https://openaccess.thecvf.com/content/CVPR2023/html/Zhang_Delivering_Arbitrary-Modal_Semantic_Segmentation_CVPR_2023_paper.html)。标题、九位作者及摘要对应本地正文；BibTeX 为 `Zhang_2023_CVPR`，会议年 2023，pp. 1136–1147。

### LIB000041 — Learning Modality-agnostic Representation for Semantic Segmentation from Any Modalities

- 已写：年份 **2024**，出版物 **ECCV 2024**。
- 来源：[Crossref 的 DOI 精确记录](https://api.crossref.org/works/10.1007/978-3-031-72754-2_9)；标题与 Xu Zheng / Yuanhuiyi Lyu / Lin Wang 对应本地正文。
- 交叉核验：[ECCV 官方论文页面](https://eccv2024.ecva.net/virtual/2024/poster/1593)；[arXiv 官方记录](https://arxiv.org/abs/2407.11351) 明确注明 Accepted to ECCV 2024。
- 日期裁决：Crossref `conference_year=2024`、`published-online=2024-10-31`、`published-print=2025`。本索引按明确的会议年填 2024，`year_kind=conference_year`，不是从 arXiv 文件编号推算，也不采用后出的印刷年份。
- 出版物原文保存在 `raw_value`：Computer Vision – ECCV 2024 / Lecture Notes in Computer Science。DOI 仅记录为核验来源标识，不额外写 DOI override。
- Springer 正文页面返回客户端验证页，未把该未读到的页面冒称已核验；采用已成功读取的 Crossref DOI 元数据与官方 ECCV 页面。

### LIB000042 — Provable Dynamic Fusion for Low-Quality Multimodal Data

- 已写：年份 **2023**，出版物 **ICML 2023**。
- 来源：[PMLR 正式 proceedings 论文页](https://proceedings.mlr.press/v202/zhang23ar.html)。标题、七位作者及摘要对应本地正文；BibTeX 为 `pmlr-v202-zhang23ar`，会议年 2023，PMLR 202:41753–41769。
- 出版物原文：Proceedings of the 40th International Conference on Machine Learning, PMLR 202。

### LIB000043 — Reducing Unimodal Bias in Multi-Modal Semantic Segmentation with Multi-Scale Functional Entropy Regularization

- 已写：年份 **2025**，出版物 **ICCV 2025**。
- 来源：[CVF 官方 ICCV 论文页](https://openaccess.thecvf.com/content/ICCV2025/html/Zheng_Reducing_Unimodal_Bias_in_Multi-Modal_Semantic_Segmentation_with_Multi-Scale_Functional_ICCV_2025_paper.html)。标题、六位作者及摘要对应本地正文；BibTeX 为 `Zheng_2025_ICCV`，会议年 2025，pp. 21166–21176。

### LIB000044 — When Depth Hurts: Reliability-Aware Geometry Distillation for Depth-Free RGB-D Salient Object Detection

- 已写：出版物 **arXiv (preprint)**；正式出版年份仍为未知。
- 来源：[arXiv 官方记录](https://arxiv.org/abs/2609.03378)，标题与摘要对应本地正文；v1 提交于 **2026-09-03 05:29:10 UTC**。
- 本次核验只确认预印本，官方页面未列 journal reference 或正式会议，限定搜索未找到可核验的正式发表记录。不能据此断言从未发表，但也不能捏造会议或期刊。
- 2026 是预印本提交年，不写成正式 publication year；提交日期与未确认发表的说明写入 venue override 的 `note`。因此 UI 继续显示“年份：未知”，同时显示“出版物：arXiv (preprint)”。

## 保存与保护核验

- 共补 13 个字段：6 个正式会议年 + 7 个出版物值（6 个会议 + 1 个预印本标记）。
- 全部值写到既有 `human.bibliographic_overrides`；包含来源 URL、来源标识、置信度 high、确认时间与核验说明。6 个年份均标为 `conference_year`。
- CVF / PMLR 官方页面人工核验使用既有允许的 `source_type=manual`，UI 标签显示“人工录入”；这不表示凭猜测填写。Any2Seg 使用 `crossref_doi`，When Depth Hurts 使用 `arxiv`。
- 保存前用共享 store 做内存预览、结构和冲突校验、保护字段对比；通过后使用 `paper_library_store.save_index` 的锁、文件指纹校验、原子替换及备份机制提交，未手改 JSON。
- revision **8 → 9**；schema 仍 4，论文仍 44，next_lib_number 仍 45。人工书目 override 论文数由 32 → 39。
- 原 LIB000001–LIB000037 完整磁盘记录逐项一致；44 篇 `auto.*` 全部不变；alternate_bundles 和 duplicate_review 不变。旧 override / supplement 不变，新论文的其他 human 字段保留既有语义（共享读取层默认字段在保存时显式化）。
- `PAPER_LIBRARY_INDEX.md` 重建后与共享 store 重渲染内容完全相同。保存后的只读扫描显示规范化文档完全一致，证明已确认 override 不会被普通扫描覆盖；未为复核再保存一次 revision。
- 无结构校验问题、人工编号冲突或扫描 warning；backups revision 8 / 7 / 6，bak1 字节哈希与本次保存前索引一致。
- UI 已逐篇读取 LIB000038–LIB000044，核验上述年份和出版物显示正确，保存按钮均禁用，界面未再次写索引。
- 保存后 JSON SHA256：`9e70625326839412e9bf5e314524725dfccb160d3195b47398110dc04798f270`。
- 保存后 Markdown SHA256：`b82a60a39814410dca688570e6dd784160efd3076a6e0463304add59ffa49ebb`。
- 本次未改工具代码、未新建验证脚本或测试文件；只做来源、数据差异及 UI 定点核验，未运行项目测试套件。
