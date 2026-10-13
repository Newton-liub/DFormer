# 科研深度提取与论文Supplement保存记录（2026-10-12任务，2026-10-13完成）

> 本轮已完成并停止。用户交付物仅两项：正式论文库已保存的Supplement，以及独立的`临时/PROJECT_EVIDENCE_PACK.md`。本报告是内部保存回执，不是第三份研究综合包。没有创建或编辑`RESEARCH_PREPARATION_PACK.md`或其他综合包；没有决定最终研究方向或开展实验。

## 最终保存状态

- 本轮共追加补充**33篇不同论文：10篇旧论文＋LIB000050–072共23篇新论文**；Cocoon另追加一次措辞更正，不重复计为第34篇。
- 索引从schema5/revision22经过旧10篇revision23–32、新23篇revision33–55，再由LIB000055更正达到**schema5/revision56**。
- 最终索引指纹：`8145a2f2278bce184b24fa3a62170531057a66f874f781462dd98779928e9405`；末次保存时间`2026-10-13T09:26:30+08:00`。
- 活动67、退库5、总登记72、`next_lib_number=73`保持不变；全库当前有非空Supplement的活动论文为50篇。33是本轮涉及论文数，50是全库累计有补充论文数，含义不同。
- 所有写入均用受支持的定位→fingerprint约束append计划→真实计划SHA-256→apply，串行提交；原补充保留，阅读状态未升级。23篇新论文仍为`abstract_only`，不因局部深读自动改成全文完成。

## 已提交的旧论文补充

本轮从schema5/revision22开始，已用`paper_workflow.py supplement-plan`→真实计划SHA-256→`supplement-apply`串行追加10篇，达到revision32。每次先取得当前定位指纹，计划/草稿/证据校验由现有工具完成；append模式保留原补充，阅读状态不变。

- LIB000002 DFormerv2：revision23；计划SHA-256 `6e0f8a9a3eb4c6bbb65f32bd4b674974aea953bbabcd23ff47d9ffa73deb73f6`。
- LIB000006 Confidence Propagation：revision24；`49da2eec3334f68fde55dd0f185415ece911d9535983cc89d098fac8d5fbc6b2`。
- LIB000029 MaskMentor：revision25；`09b709a39928d5c803f84772833123d91ec1be5cd82cc9b224430d1134f2fc50`。
- LIB000030 ConD：revision26；`d11f75da996096b7a44b39010cc4d5c466ff82d2f390f15c461b7e4c605614bb`。
- LIB000035 OmniSegmentor：revision27；`7c5aabeae9b64270f343794b7db6cd15d7461858b61c8ebfdb6a90e301dd1aa3`。
- LIB000036 SimMLM：revision28；`88664a0769ac557a87d3b7b7274ac17fdf5595441add21e378a1d8271bf80503`。
- LIB000037 DFormer++：revision29；`96987cf00b48a6e63cfdaa8a936e0567da1e7ed03f56c5b884811aa6754fa9aa`。
- LIB000042 QMF：revision30；`54eaac8a92955d61e176e33e386f6a361ea859af4f56172524374bd25ff7eccd`。
- LIB000043 Functional Entropy：revision31；`63b3c02bf3186823e1fdd8e05433b2983b8260f6f695dfb48ab63376634b1f32`。
- LIB000049 SPGNet：revision32；`f1400617ef6ff711f767bbeb91e0d65f576a5328981a2ebe54a2b7ce28d3b4a3`。

草稿与完整计划保留在`临时/research-extraction-20261012/LIBxxxxxx.md`及同名`.plan.json`，未清理。revision32指纹`9ec59f37dd9f2f906ebca42e26cbe69aac75e29756a271c696b91b349edbdff1`仅为历史中间点，不可作为新一轮提交基线。旧10篇本次收尾未重新提取或追加保存。

## 23篇新论文：全部正式保存

主代理已读全部23份补充稿，结合所需原文、精选公式/图表完成定点核验，并依LIB顺序提交。内容覆盖任意模态分割、缺模态适配、可信融合、训练平衡、信息瓶颈、多尺度融合/上采样与鲁棒性数据集。各篇实际阅读范围、未核理论/官方实现与迁移限制保留在正式Supplement，不把所引用的全部论文或整份附录冒称已独立阅读。

保存revision映射：

- LIB000050–055：依次revision33、34、35、36、37、38。
- LIB000056–061：依次revision39、40、41、42、43、44。
- LIB000062–067：依次revision45、46、47、48、49、50。
- LIB000068–072：依次revision51、52、53、54、55。

上述23份计划均使用scope `2026-10-12科研深度提取`，实际计划SHA-256已在每次apply前计算并提交。全部草稿和批准计划仍在同一工作目录，作为逐篇回执；此处不重新制造一套计划或索引副本。revision55指纹为`594b861bc9551f2d62a7ee33f1c809a770c2abf25cf6463104146156d8b216b4`，现已被下述更正保存取代。

### Cocoon收尾更正

主代理回读原始Markdown §4.2–4.4后发现旧措辞错误：FI应为Feature Impression（与对齐器联合学习的参照节点），稳定性来自同一Transformer解码器不同层的top-1 FI变化，而非多个独立decoder；相应开销应归于对齐器、校准与跨层处理。按append追加明确替代说明，原文与先前补充保留。

- 最终计划：`临时/research-extraction-20261012/LIB000055.correction-final.plan.json`。
- 真实计划SHA-256：`b6be99638d34ae4d2ad5f2e12b706b9f84b68ef4a9db276647587487f02b759d`。
- 保存基线为上述revision55指纹；实际apply成功到revision56，回读LIB000055补充1518字符、`abstract_only`，无状态升级。
- 早期`LIB000055.correction.plan.json`未apply。补齐开销说明后工具拒绝覆盖已有审核计划，因此保留旧计划，以同一未变化基线生成`correction-final`计划并审核；这不是索引并发冲突，也没有换指纹绕过校验。早期计划与当前草稿不匹配，不能再用于apply。

## 派生AI导出与定点回读

受支持的`paper_workflow.py views-refresh`在最终revision56成功重建管理视图及三份导出；直接打开三份文件头确认源schema5/revision56与时间`2026-10-13T09:26:47+08:00`：

- `D:\0Project\origin\_index\exports\PAPER_LIBRARY_FOR_AI.md`：全部活动论文67篇。
- 同目录`PAPER_LIBRARY_FOR_AI_CORE.md`：核心15篇。
- 同目录`PAPER_LIBRARY_FOR_AI_SUPPLEMENTED.md`：有非空补充50篇。

在revision55保存完成后，以受支持定位入口逐篇回读23篇的新补充存在性、长度与阅读状态，并回读旧10篇补充仍在、状态保留；最终revision56再定点回读Cocoon更正及全部论文导出中的更正正文。此前全部论文导出末段已分段打开，确认新补充、公式和限制进入导出；一次超长读取截断后，用定点分段读取补齐所需内容，截断部分本身不算已读。最终三份导出头部均已直接回读，未声称逐字审核全部导出或做完整元数据差分。

以上是定点存在性/状态/导出内容核验，不是全库逐字段、逐字节审计。旧10篇保留状态为：002/037 `full_text_completed`，042/049 `needs_full_text`，006/029/030/035/036/043 `abstract_only`。旧人工编号与已有笔记未作为本轮编辑对象。

## 项目证据交付

[PROJECT_EVIDENCE_PACK.md](../../临时/PROJECT_EVIDENCE_PACK.md)已最终更新：保留SUN R1混杂/R2结果、DeLiVER已验收但未正式训练、旧MMFR失败与未完成运行、DFormer++双流源码事实，以及几何可靠性、表示学习、缺模态适配、训练平衡与解码器等开放问题。更正UniME为单一编码器的多模态MRI掩码预训练，以及Cocoon的FI/跨层稳定性；补充解码机制、标注可观测性、额外模态非单调收益和RMT乘prior位置的区别。历史几何诊断示例保留为条件性材料，不是本轮选定方案。

早期`EVIDENCE_CHECKLIST.md`只加历史说明：它保留早期委派/草稿范围，已被各LIB正式补充中的最终阅读范围取代，不作为当前保存状态或完整阅读范围真源。

## 证据与操作边界

- 使用canonical原始Markdown及精选Figure、表格JPG和公式图；未读取原PDF/XLSX，未读资产不算证据。公式图片文件序号不自动当作正文公式编号。区分论文事实、官方/本地源码事实、项目实测与模型迁移判断。
- 定点Glob/Grep曾返回错误工作区结果，已丢弃；以CLI定位和精确绝对路径读取核验。失败路径、转码错误和截断部分均不冒称已读，未因此扩大扫描。
- 旧LIB000002/037已有历史`full_text_completed`，工具仍要求`--full-text-scope`记录；如实说明保留历史完成状态、本轮仅局部复读，计划`changed=false`。一次证据路径Unicode捕获错误导致计划被拒，纠正路径后保存；全部正式apply没有索引冲突或失败，没有借新指纹绕过拒绝。
- 仅追加授权LIB的补充及相应来源记录；原始论文、现有Supplement、阅读状态、人工身份、核心标记及无关索引字段保留。退库资产、历史与临时资料不移动、不清理、不删除。
- 没有修改代码/模型/配置/数据，未训练、推理、GPU/云操作、下载、转换、提交或推送。按文档验证预算只复核内容与差异，未运行项目测试、完整测试套件或复杂审计。
- 已更新现有报告、独立项目证据包与项目状态；没有另建综合包，没有选择最终方向。所有研究线索都不构成实验授权。

## 用户后续交付步骤

用户在论文库UI手动导出“全部活动论文”的AI阅读Markdown（67篇，`PAPER_LIBRARY_FOR_AI.md`），与`临时/PROJECT_EVIDENCE_PACK.md`一起交给云端高级模型。导出包含书目、已存摘要和正式Supplement，不自动包含原始论文全文；仅核心或仅已有补充范围不能替代全部活动论文。本轮到此停止。
