# 最后一轮论文定点补充审核（2026-10-09）

## 最新结果：已批准并保存

用户于2026-10-09 14:34明确“批准保存”。核验原联合清单SHA-256 `9d337082ad103feb39361dbfd5e065280e7c8fe8732eb067d816d6d8fb993e96`、共同F0、11个原计划及正文/草稿/证据，共111个不同文件后，通过既有`paper_library_store.save_index(expected_fingerprint=F0)`一次联合保存，未开发写入器、未重生成计划。

- **正式索引schema4/revision18、49篇、next_lib_number50**，保存时间2026-10-09T14:41:53+08:00，实际指纹`c7c0419e6bb3f1ddb4c554ae07d1ad9bf659ad191cc1a2809a152e201b96280e`。
- 仅批准的16篇记录变化：11个supplement追加、9个reading_status改为`needs_full_text`、15个is_core设为true、GeminiFusion缺失year=2024及venue=ICML 2024。12篇旧`full_text_completed`状态不动；局部阅读没有新增全文完成。保存后共21篇非空supplement、15篇核心、28篇abstract_only/9篇needs_full_text/12篇full_text_completed。
- 保存前精确字段白名单与原笔记前缀/来源已核；保存后实际文档逐项等于批准对象。原始书目/摘要、人工编号/标签、auto、其它记录/字段及canonical资产不变。备份为revision17/16/15，bak1与审批基线F0完全一致。
- 管理视图及全部/核心/已有补充AI导出均已从实际revision18重建，范围49/15/21篇，时间2026-10-09T14:43:23+08:00；文件内容与既有renderer一致，导出未再次保存索引或轮转备份。
- 初检曾把草稿的原始字节hash与原工作流的文本hash比较，因Windows换行口径不同而在保存前停止；按原工作流`read_text`后的UTF-8文本hash核验，11篇全部一致。这不是草稿变化，没有改草稿/计划，没有失败的索引提交或二次保存。
- 原11篇草稿、11个计划和`INDEX_FIELD_REVIEW.json`原样保留，执行回执另存审核根`SAVE_RESULT.json`；它们已作为历史审核材料，旧revision17计划不再可apply。论文搜索、PDF下载和重复阅读继续停止，不重试DFormer++附录。

## 审批前记录（以下待审措辞为历史状态）

完成11篇指定论文的证据草稿、主代理关键复核与现有工具生成的11个 `supplement-plan`。**停在待审批阶段，未保存索引、未执行apply。** 唯一索引仍为schema4/revision17、49篇；指纹仍为 `a8b55ecc597edcc578e2ba7c8d6c440e51b2fcc0c84e08b0ae5e4d0fdcc54f13`。四篇新指定论文已是LIB000046–049，沿用身份，没有重复入库。

审核根：`D:\0Project\origin\_index\supplement-review\2026-10-09-final\`。
- `drafts/LIB0000xx.md`：11篇本轮追加内容，论文、实现和主代理判断分列。
- `plans/LIB0000xx.json`：现有supplement工具生成、含body/draft/evidence hashes的单篇计划。
- `INDEX_FIELD_REVIEW.json`：11篇计划SHA256、9个阅读状态调整、15篇核心集、GeminiFusion缺失year/venue的联合审批清单；不是第二份索引，也不是可直接apply的工作流计划。
- `evidence-D.json`、`evidence-C.json`、`drafts/evidence-A.json`、`drafts/evidence-B.json`及`evidence/main-verification*.md`：实际打开材料与主代理核验边界。

所有单篇计划绑定同一F0；**不能依次盲目apply**，第一篇保存后其余计划即过期。需另行明确批准通过现有UI/store的一次共同基线提交，先核所有计划/正文/草稿/证据哈希、旧笔记前缀及字段白名单；若现有通道不能满足保护就停止，不开发写入器、不用新指纹绕过批准。

## 补齐的指定证据

1. **LIB000002 DFormerv2**：CVPR官方supplementary三页与Tables1–3完成阅读/视觉核验；补预训练及NYU/SUN schedule/stage配置。SUN lr8e-5、batch16、300epoch、warmup10有附录依据。正文poly、附录linear与当前power.9分别保留；decoder、batch/drop-path/eval配置不完全一致。确认depth/spatial独立距离、每block两个head-shared可学习scalar及固定head decay；归一化depth与孔洞/padding含义。只去depth项的位置为GeoPriorGen.forward轴向L198–199/full L208，未改代码。
2. **LIB000037 DFormer++**：补本地T/S/B stage、NYU-T配置、与v2的depth特征QKV/距离prior区别、预训练依赖；不重推旧公式。IEEE官方supplement返回418，附录未读，代码不冒充附录。2026-10-09读取README仍显示++权重Coming soon。
3. **LIB000001 MUSeg**：15个对象类、rail area而非可通行地面/road类、SDK对齐/裁剪、16-bit原depth与8-bit显示、位置组与作者1595/1576 split、真实低光等证据等级。numeric IDs/ignore仍未知；毫米/zero以Microsoft传感器文档分列，不冒称raw PNG已核。
4. **LIB000014 SGMA**：prototype、RP空间robustness、MAS逆robustness后空间均值抽样、两路GT-CE、关键消融。Eq3公式/声明维度不自洽，原式保留，不猜实现。
5. **LIB000024 UMFNet**：pixel-wise Gaussian variance、inverse uncertainty/confidence、隐式latent alignment及残差融合；明确RGB-T显著目标任务。Table3表头与Table2数值重复/unaligned主表不一致，不将消融数值当确定错位增益。
6. **LIB000042 QMF**：energy与负线性sample/modal权重、历史loss监督ranking；Theorem2是有条件上界比较，非准确率保证；与CoRiM同为sample-level decision层而目标不同。
7. **LIB000045 GeminiFusion**：PMLR正式ICML2024书目、Eq6图片与Table9核验、同位置attention/relation与learned self-K/V noise；固定commit源码、NYU/SUN公开模型链接和接入接口。公开权重未下载；独立MiT/Swin接入与公平初始化/数据协议仍未验收。
8. **LIB000049 SPGNet**：Sobel feature-edge consistency/feature discrepancy结构prior，source gate/KV correction先于DEF；训练/标签协议、Table4 depth-only noise/1–5px shift及Table5控制消融完成关键复核。大块孔洞未测，单扰动seed未留存，不能声称每条件降幅最小。
9. **LIB000046 AGWNet**：本地全文主线可读，不存在等待用户提供全文的阻塞。补channel/spatial gate、decoder deformable offset、相机质量子集/消融；正文SUN8268/2067而非标准5285/5050，不横比标准结果。合成noise/hole severity在已读正文未见。
10. **LIB000048 RGB-D Mirror Segmentation**：双depth原响应/对数/梯度差、validity、局部variance；entropy×learned gate作用于最终logit residual；final CE与GT条件safe loss。式25图确认L1范数；不称residual有独立真值标签。
11. **LIB000047 GeoSphere-DETR**：仅指定题录/摘要/有限正文、RGB-only与depth退化对照；Table5和Fig9核验。检测而非语义分割；Fig9 noise=0 E4 FPPI与dropout=0/Table5不一致，保留，不补造逐点数值。

可选LIB000030/041/043只定位，不扩展阅读/补充、不变状态。

## 阅读状态与核心标记（原提议已按批准保存）

- LIB000002/037保持此前用户批准的 `full_text_completed`，其语义是正文已读并有笔记，未读附录/资产仍写明；DFormer++不因无法访问附录而抹掉旧批准状态。
- 另9篇目标提议 `abstract_only`→`needs_full_text`：已定点读正文、仍有未读范围，不自动转全文完成。其它40篇状态不动。
- 15篇核心集：LIB000001、002、014、024、028、033、037、038、039、042、044、045、046、048、049。用途分别为数据/基线/传感器故障协议，局部特征可靠性，预测冲突与理论，geometry injection/残差校正近邻。旧LIB000028/033/038/039/044仅复阅已有笔记、不重提取全文。GeoSphere47作为非核心检测旁证。
- GeminiFusion仅提议填现有空缺year=2024、venue=ICML2024；作者/摘要/其它原书目不改，卷页记入supplement，不新增索引字段。

## 直接撞车风险与真正剩余问题

- SPGNet已有“结构一致性/差异→局部门控→聚合前纠正→depth-only故障评价”；SGMA/UMFNet/MoSA/AGWNet已有语义/不确定性/统计/质量特征融合。泛称“局部可靠性”“先校正再融合”不够构成贡献。
- Mirror已有sensor–estimated depth discrepancy、预测熵、reliability gate、保守logit残差校正；GeoDistill已有appearance–geometry差异输入和逐像素geometry残差注入。若候选落入这些组合，直接近邻风险很高。
- ECoLaF已有pixel-wise预测冲突折扣，CoRiM/QMF有sample-wise决策冲突/质量加权；直接密集化不能仅靠“pixel-wise”宣称新颖。QMF理论不覆盖attention-bias校准。
- 当前可确认的作用点区别是DFormerv2 attention内部depth-derived token-pair log-bias；此区别既不证明创新，也不证明退化depth真的劣于spatial-only。两套方法尚未设计/裁定，本轮没有产生模块。
- 真正影响设计的未决项：MUSeg numeric labels/background-ignore与split provenance；真实depth到PNG/归一化的单位和无效值约定；RGB语义边界与正确几何不一致时如何归因；DFormerv2已训练depth权重符号及错误prior主动伤害的实证（本轮不做实验）；DFormer++附录/公平预训练发布；外部baseline相同表示/初始化/协议的可比性。UMFNet、SGMA与GeoSphere原材料内部疑点限制可引用证据强度。

## 导出、状态与验证预算

审核阶段曾导出revision17旧笔记；获批保存后，已使用可视化工具共用的既有store导出实现重建`D:\0Project\origin\_index\exports\PAPER_LIBRARY_FOR_AI.md`（全部49篇）、`PAPER_LIBRARY_FOR_AI_CORE.md`（核心15篇）、`PAPER_LIBRARY_FOR_AI_SUPPLEMENTED.md`（已有补充21篇）。三个header均为revision18/schema4/格式1.0.2，导出时间2026-10-09T14:43:23+08:00。**全部版与已有补充版现已包含本轮11篇笔记，核心版按15篇范围输出**；内容与既有renderer逐项一致，导出未改权威JSON/备份。

已更新项目实时状态，恢复点为“本轮保存与导出已完成、停止继续扩展”；此前观察本地HEAD `79f8e81acd74b9f4ba65be9beaf6e3ee32736eac`。本次仅核验已有审批材料/字段、执行批准保存及重建派生视图，没有新论文阅读/检索/下载，没有项目测试、训练、评价、GPU、数据集或权重下载、代码修改、commit/push。此前浏览器阅读留下 `.playwright-mcp/jia24b.pdf`、`sensors-26-03739.pdf`与页面快照；不计为已读原PDF，不自动删除。用户原有临时来源和working changes保持。
