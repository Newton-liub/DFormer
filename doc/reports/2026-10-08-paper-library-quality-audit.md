# 论文库质量审核与定点修正报告

> 审核批次：2026-10-08；正式保存完成于 2026-10-09 00:02:45（UTC+8）。范围：外部正式库44篇书目/摘要、11篇已有人工补充（supplement）。本报告供上级模型审核；本轮完成后停止，不继续全文研究。

## 结果

- 权威索引仍在 `D:\0Project\origin\_index\PAPER_LIBRARY_INDEX.json`，通过现有共享store一次保存，从 **schema4 / revision11 → schema4 / revision12**；44篇、`next_lib_number=45` 均不变。
- 仅写入 `human.bibliographic_overrides` 的 **37个字段，涉及21篇**：作者1、年份4、出版物1、DOI5、arXiv13、摘要13。其中36个有效值改变；LIB000017年份数值仍为2026，只新增正式出版来源确认。
- 两份逐篇审核材料合计覆盖44篇；“完成审核”表示每篇已检查并作采纳/保留处置，不表示每个字段均成功联网核实。无法取得同版本可靠证据的缺项及疑点保留待审。
- 有效值缺项：作者1→0、正式年份7→4、出版物4→3、DOI19→14、arXiv44→31；摘要仍为44篇均非空。
- 已重建管理Markdown与三个AI阅读导出：全部44篇、核心0篇、已有补充11篇，均来自revision12、导出格式1.0.2。核心范围为0也写入固定文件，不遗留旧内容。

## 已保存修正及来源

以下 `null` 或 `[]` 表示保存前有效值。摘要只做列明的精确片段替换，其余文字保留；完整存储值及来源元数据在权威JSON中，精确操作输入保留在外部审核材料中。

1. **LIB000002 / DFormerv2**：arXiv `null → 2504.04701`。[官方同题记录](https://arxiv.org/abs/2504.04701)，保留2025会议年及原出版DOI。
2. **LIB000004 / LightDepth**：DOI `null → 10.1016/j.robot.2024.104784`。[Crossref](https://api.crossref.org/works/10.1016/j.robot.2024.104784)，标题、作者、期刊及2024出版日期匹配。
3. **LIB000008 / Learning Selective Mutual Attention and Contrast**：DOI `null → 10.1109/tpami.2021.3122139`，[Crossref](https://api.crossref.org/works/10.1109/tpami.2021.3122139)；arXiv `null → 2010.05537`，[官方期刊扩展记录](https://arxiv.org/abs/2010.05537)。摘要 `a twostream CNN → a two-stream CNN`、`evaluation o deep models → evaluation of deep models`，依据[NCBI期刊摘要XML](https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=pubmed&id=34699348&retmode=xml)。保留正式期刊年2022，不按DOI中的2021改年，不合并旧题会议版。
4. **LIB000012 / NL-3A**：摘要 `afinity → affinity`、`afinities → affinities`、`diferent → different`、`eficiency → efficiency`；删除尾随 ` © 2023 Optica Publishing Group under the terms of the Optica Open Access Publishing Agreement`。修复后与[Crossref正式摘要](https://api.crossref.org/works/10.1364/oe.492187)一致。
5. **LIB000014 / SGMA**：arXiv `null → 2603.02505`；摘要 `cross-modal hetero- geneity → cross-modal heterogeneity`、`modalityspecific cues → modality-specific cues`、`plugand-play modules → plug-and-play modules`。[官方记录](https://arxiv.org/abs/2603.02505)支持共同短语；期刊与预印本其他措辞不同，保留当前期刊文字及RGB/DSM/NIR/SAR例子，不整段覆盖。
6. **LIB000015 / PDFNet**：arXiv `null → 2503.06100`；摘要 `depthaware structure → depth-aware structure`、`F <sup>max</sup> → Fmax`。[官方v5](https://arxiv.org/abs/2503.06100v5)。保留0.915指标及CVPR2026出版身份，不把2025提交年改为正式年。
7. **LIB000016 / VarSplat**：年份 `null → 2026`，`year_kind=conference_year`；arXiv `null → 2603.09673`。摘要 `real world scenes → real-world scenes`、`lowtexture regions → low-texture regions`、`singlepass rasterization → single-pass rasterization`、`Ex- perimental results → Experimental results`。[官方记录](https://arxiv.org/abs/2603.09673)明确Accepted to CVPR2026。
8. **LIB000017 / UAV-Borne RGB-D tunnel**：有效年份 `2026 → 2026`，新增 `year_kind=published_date` 的[Crossref正式出版确认](https://api.crossref.org/works/10.1007/s42461-026-01692-z)，published-online/issued为2026-08-28。原auto中accepted_date_line的日期和中等置信度证据不改。
9. **LIB000024 / UMFNet**：年份 `null → 2026`（`conference_year`）、出版物 `null → CVPR 2026`。[CVF正式会议页与BibTeX](https://openaccess.thecvf.com/content/CVPR2026/html/Wang_Uncertainty-Aware_Modality_Fusion_for_Unaligned_RGB-T_Salient_Object_Detection_CVPR_2026_paper.html)，标题及5位作者匹配。
10. **LIB000025 / MV3DIS**：arXiv `null → 2604.08916`；摘要 `Scan-Net200 → ScanNet200`。[官方记录](https://arxiv.org/abs/2604.08916)，标题及3位作者元数据匹配；保留其余版本措辞和代码链接，正式年份/出版物未知。
11. **LIB000026 / GeomPrompt**：年份 `null → 2026`（`conference_year`）；摘要将 `down- Supported in part by NSF Award #2345057. Project page: https://geomprompt.github.io stream` 精确替换为 `downstream`，移除串入单词的脚注。[官方项目页摘要及BibTeX](https://geomprompt.github.io/)确认CVPR2026 Workshops身份。
12. **LIB000027 / Calibrated RGB-D SOD**：摘要 `cuttingedge → cutting-edge`。[CVF正式摘要](https://openaccess.thecvf.com/content/CVPR2021/html/Ji_Calibrated_RGB-D_Salient_Object_Detection_CVPR_2021_paper.html)，标题及11位作者一致，其他文本保留。
13. **LIB000030 / ConD**：arXiv `null → 2607.20326`。[官方记录](https://arxiv.org/abs/2607.20326)，标题、摘要方法及6位作者匹配；2026提交年不填入正式年份。
14. **LIB000032 / UMIS-Mine**：DOI `null → 10.1007/978-981-92-3432-5_1`。[Crossref](https://api.crossref.org/works/10.1007/978-981-92-3432-5_1)，标题、5位作者、Springer书章匹配。线上出版2026-07-14、ICIC会议年2026、印刷年2027，保留既有2026年；不处理待审同文异附件副本。
15. **LIB000033 / sensor-failure benchmark**：arXiv `null → 2503.18445`。[官方v3](https://arxiv.org/abs/2503.18445v3)支持摘要中将 `Based on <sub>EMM</sub> , mIoU E <sub>EMM</sub>, mIoU<sup>Avg</sup> <sub>RMM</sub> , and mIoU <sup>E</sup><sub>RM</sub> —to assess` 替换为 `Based on these, we propose four metrics—$mIoU^{Avg}_{EMM}$, $mIoU^{E}_{EMM}$, $mIoU^{Avg}_{RMM}$, and $mIoU^{E}_{RMM}$—to assess`；恢复来源已有导语及四个指标，不凭正文推造公式。
16. **LIB000034 / When Fusion Fails**：arXiv `null → 2609.10261`；摘要 `singlemodality baselines → single-modality baselines`、`inputs difer in quality → inputs differ in quality`、`substantially afect predictions → substantially affect predictions`、`sliceretention ratios → slice-retention ratios`。[官方摘要](https://arxiv.org/abs/2609.10261)，保留既有ACM MM2026身份。
17. **LIB000039 / CoRiM**：作者 `[] → [Shihao Zou, Wei Wei]`。[CVF论文页与BibTeX](https://openaccess.thecvf.com/content/CVPR2026/html/Zou_CoRiM_Conflict-driven_Risk_Minimization_for_Dynamic_Multimodal_Fusion_CVPR_2026_paper.html)。官方摘要本身也有 `simplex.We`，不擅自修订。
18. **LIB000040 / DeLiVER**：arXiv `null → 2303.01480`，依据[CVF Related Material](https://openaccess.thecvf.com/content/CVPR2023/html/Zhang_Delivering_Arbitrary-Modal_Semantic_Segmentation_CVPR_2023_paper.html)；DOI `null → 10.1109/cvpr52729.2023.00116`，依据[Crossref同题及9位作者](https://api.crossref.org/works/10.1109/CVPR52729.2023.00116)。摘要 `of<sup>D-E-D-E-D-E-</sup> modalities → of modalities`、`baseline.<sup>1</sup> → baseline.`，依据同一CVF摘要，保留其他canonical措辞。
19. **LIB000041 / Any2Seg**：DOI `null → 10.1007/978-3-031-72754-2_9`，[Crossref](https://api.crossref.org/works/10.1007/978-3-031-72754-2_9)；arXiv `null → 2407.11351`，[官方记录](https://arxiv.org/abs/2407.11351)。两者与既有Xu Zheng、Yuanhuiyi Lyu、Lin Wang三位作者一致，原委派口述的作者冲突未复现。摘要 `stateof-the-art → state-of-the-art`、`setting(+19.79 → setting (+19.79`，依据[ECCV官方摘要](https://eccv2024.ecva.net/virtual/2024/poster/1593)；保留会议年2024，不采用印刷年2025，指标不变。
20. **LIB000043 / multi-scale functional entropy**：arXiv `null → 2505.06635`，依据[ICCV官方Related Material](https://openaccess.thecvf.com/content/ICCV2025/html/Zheng_Reducing_Unimodal_Bias_in_Multi-Modal_Semantic_Segmentation_with_Multi-Scale_Functional_ICCV_2025_paper.html)；摘要 `realworld → real-world`。该官方摘要本身包含 `plug-and-paly`，保留原拼写。
21. **LIB000044 / GeoDistill**：arXiv `null → 2609.03378`。[官方记录](https://arxiv.org/abs/2609.03378)。v1提交2026-09-03，正式出版年/出版DOI未知；不把官方摘要中的未展开宏混入当前干净摘要。

所有新override均保留合法来源类型、URL、标识符、high置信度、确认时间及说明；年份另带合法出版语义。后22篇13条新增建议中的LIB000041两处摘要修改合并为一个字段，故新增12字段，加原25字段共37字段。

## supplement与阅读状态

- 11篇已有笔记全部符合现有结构，共67,512字符；`content_md/updated_at/source_note/sources`整个对象在本轮保存前后逐记录一致，科研判断和限定语均保留。
- 11条 `sources` 列表仍为空；正文内有canonical路径、章节/表号及官方链接。本轮只核对结构和已有证据边界，不把未逐项核对的链接批量转成已确认结构化来源。
- 全44篇阅读状态仍为 `abstract_only`，核心标记仍为0篇。已有正文笔记包含草稿、未读附录、未核实表格或附件，不因supplement非空就自动标记全文阅读完成。
- 逐篇边界见 `D:\0Project\origin\_index\quality-audit\2026-10-08\supplement-review.md`。本轮不新增阅读状态或改变科研结论。

## 未解决项及未采纳建议

- 正式年份仍未知：LIB000023、025、030、044；出版物仍未知：023、025、030。025/030已关联官方预印本，但v1年不当正式出版年；044已有arXiv出版物标记。
- 出版DOI仍缺14篇：015、016、021、023、024、025、026、027、030、038、039、042、043、044。arXiv仍缺31篇，详见权威JSON；缺arXiv不表示必然存在未补记录。
- LIB000015/016的arXiv-issued DOI不填入现有单一出版DOI槽位；避免把会议与预印本身份混用，不为此升级schema。
- LIB000018标题 `M− SURE → M−SURE` 有来源支持，但现有override不支持title，保留待审；不改auto、不扩大schema。
- LIB000006候选arXiv1805.11913、LIB000023候选2505.12861的题名/摘要版本关系不明，不关联或合并。
- 保留未取得同版本可靠摘要的OCR疑点：001、003、005、006、010、013、017、018、021、023、029、031、032等；021的 `GT!Pred/Pred!GT`、031的 `instinct expertise` 不凭语感修改。部分Crossref429、出版社挑战/403按失败留待审，不算已核验来源。
- LIB000039 `simplex.We`、043 `plug-and-paly` 属于官方原文已有形式，未作为抽取污染修复。保留未确定摘要边界的代码URL，不擅自删除。
- UMIS-Mine同文异附件副本仍待人工审核，未合并、覆盖、移动或删除。

## 最小工具修复与验收

工具仅修改外部 `tools/paper_library_store.py` 的AI导出呈现，README同步说明；UI、书目抽取、schema、保存协议均未修改。

- `_export_supplement_markdown` 在导出时将现有笔记ATX一至三级标题映射为四至六级，避免与论文二级标题混淆；围栏代码、换行、公式、链接和其他正文保留，存储正文不改。
- 笔记来源及时间放在正文前；页首增加源schema与格式版本1.0.2，并明确非空supplement不代表全文阅读完成。
- 已执行此前的定点检查：共享store结构/编号校验；all/core/supplemented渲染；反引号/波浪号围栏、未闭合围栏、CRLF、公式/链接保护；只读渲染内存与磁盘指纹不变；修改区域Pyright诊断为空。本次不重复这些边界检查。
- 本次提交前断言37字段旧值/片段及基线指纹，校验通过后只调用一次 `save_index(expected_fingerprint=...)`；保存后重读，逐记录对比仅允许指定override与根revision/updated_at变化。全部auto、LIB顺序、人工编号、其他human、整个supplement、alternate、duplicate_review、next_lib_number及其他根字段不变；已保存值与准备值一致。
- 备份轮转实测为11/10/9：bak1指纹等于本轮原revision11，bak2/3分别等于原bak1/2指纹。
- 重建三范围导出实测44/0/11，源revision12、格式1.0.2；每个文件仅一个文档H1，论文H2数量匹配。逐篇确认完整已存摘要和仅变换标题的完整supplement均存在于导出；导出前后索引与三份备份指纹不变。
- 保存后的首轮导出验收在中文字符串断言处失败：PowerShell管道把源代码中文字面量转为问号。直接诊断确认导出头及H1/H2正确；改用Unicode转义的验收表达后全部通过。没有再次保存索引、回滚或修改工具代码。

指纹与落点：

- 原revision11 SHA-256：`b29d4289b24d7b1c7cdc5261526f8eaee935294e7c336e4881f86f6306b95a6b`。
- 新revision12 SHA-256：`17de87b21649aba0964d7028856421eb3ceabf0d7fe13304cfa655db27b036b1`。
- 管理视图：`D:\0Project\origin\_index\PAPER_LIBRARY_INDEX.md`。
- AI导出：同目录 `exports\PAPER_LIBRARY_FOR_AI.md`、`PAPER_LIBRARY_FOR_AI_CORE.md`、`PAPER_LIBRARY_FOR_AI_SUPPLEMENTED.md`。
- 审核材料：同目录 `quality-audit\2026-10-08\part-01-22.*`、`part-23-44.*`、`main-verified.json`、`main-verified-additional.json`、`supplement-review.md`、`acceptance.json`。候选材料保留为审计输入，不是第二份活动索引。

## 保护范围与停止点

本轮未写原bundle或外部科研代码；保护结论基于索引字段逐记录对比及操作范围，**没有重新哈希全库资产**，不宣称全资产已重新验收。未全库重新抽取、未扩大全文研究、未新建测试或临时验证脚本、未运行完整项目测试（按验证预算只检查本次保存/导出风险），未启动UI回归、训练/GPU/云操作，未Git提交或推送。

`doc/state/current.md`与外部README更新为真实完成状态。本轮在此停止，交上级审核；待审字段、标题能力扩展、阅读状态细分与科研判断需另行决定。
