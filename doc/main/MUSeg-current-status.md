# MUSeg 当前状态与唯一实时入口

> **事实截至：** 2026-10-01 本地完整 Git 收口与云端连续执行授权。当前为 **`A-v1 Gate-B PASS; formal contract frozen; cloud preflight → formal training → four-condition Quick-Val authorized; complete local snapshot committed and pushed`**。本会话只提交、推送和整理指令；尚未登录云端、运行模型、训练或评价。执行口径见 [A-v1 protocol](../../MMFR/01_research/mmfr_a_v1_action_utility_protocol.md)，仍开放的选择见 [开放决策](MUSeg-open-decisions.md)。

## 当前阶段与实际意义

用户于本会话明确确认：**提交全部需要保留的本地改动并推送；云端实现必要入口，Proposal/Gate 各最多3次更新预检通过后，连续执行1920+640次正式训练及唯一四条件 Quick-Val，完成后停止。** Proposal 是补偿残差分支，Gate 是每图补偿强度选择器，Quick-Val 是小范围开发集筛选。**大白话：** 不再每一步都等待重复授权，但不能跳过容量预检，也不能擅自扩大实验或改变冻结合同。旧文档中的“正式训练未授权”和“预检后等待授权”属于2026-09-30历史边界，已被本次明确授权替代。

## 已核验事实与身份

- A-v1 identity：`MMFR-A-v1-action-utility-v1`；run：`MMFR-A-v1-action-utility-v1-formal-v1`；config：`local_configs.MUSeg.DFormerv2_S_MMFR_AV1`。
- 现有运行代码基线：`e26d670e279970ecb8907aef1de470e992be500e`。本次直接检查5个关键代码/config文件相对HEAD无内容差异；本地没有实现新的runner/evaluator。只有阶段接口，云端需补齐全尺寸预检/正式训练和A-v1评价入口，不能直接套旧E1训练命令。
- C0 fixed-final SHA-256：`ca618b23d18eabb201a0d11d18da383ac99576d0feae5864e3233bda527d9a1a`；原Gate-B JSON SHA-256：`bf87cc2fc46dd413131f7a6778d8df82388cf7fffa08852126d1f62178be01bf`。沿用2026-09-30直接核验结果，本次不重复计算大权重hash、不重跑GPU资格检查。
- 原Gate-B仅为synthetic B1/64×64资格证据，不代表全尺寸容量、方法收益或论文创新性。4090 batch10仍待预检。原始证据及历史结果保持不变；历史pre-A-v1归档仅换行差异已恢复到HEAD原字节。
- config的本地CRLF/Git LF哈希区别及5个Git代码hash仍见 [复现信息](../../MMFR/02_evidence/reproducibility_current.json)。后续有实质代码修复时记录新提交和影响，不要求每次文档修改都刷新全部文件hash。
- official test保持 **`sealed_unread`**；A-v1没有训练或评价结果。

## 本次云端执行授权

1. 在用户指定的现有RTX4090环境同步本轮提交；只核对执行版本、真实train-dev路径/样本划分及C0权重身份等必要前提。不自行新建付费实例或扩大资源。
2. 复用已有阶段接口、数据/损坏生成和评价工具，补齐最薄执行入口。只做覆盖新入口风险的最小检查；无需新测试框架、全仓扫描、完整测试套件、文献检索、重复Gate-B或benchmark。
3. Proposal最多3次成功更新；正常后Gate最多3次，使用同一预检Proposal权重。每阶段短运行到上限即退出，记录显存、时间、loss、finite/scaler skip；Gate另报utility正/负/模糊计数。两阶段通过后，**本次已授权自动继续正式训练**，无需再等待重复确认。
4. 从原C0及冻结初始化/种子干净开始正式训练，预检权重和步数不计正式预算。Proposal1920/Gate640；固定batch10、480×640、workers8、accumulation1、AMP fp16/TF32 on；margin0.01、lambda_clean0.1、AdamW LR3e-5/WD0.01、各阶段独立128-update warmup/poly0.9。其余以protocol为准。
5. Proposal1920保存`proposal-update-1920.pth`，Gate使用同一Proposal权重；全局2560保存唯一`update-2560.pth`；每640次成功更新保存完整恢复状态。只因工程/数值/身份/基础设施错误或无法解释的明显loss异常停止，不以utility或gate分布提前终止。
6. 正式成功后，仅对同一fixed-final执行一次四条件Quick-Val，off/full/learned同输入、配对随机状态；318 val-dev、original-full、scale1/no flip、FP32/TF32 off、eval seed2026091401/reset-per-unit。条件为clean、entire_missing@1.0、spatial_dropout@0.75、misalignment@0.75。评价薄入口需定点确认输入、随机回放和指标一致后再产生结果，不额外跑整套资格实验。
7. 汇总三行为每条件mIoU、hard平均、相对matched off差值、gate mean/median及冻结筛选结论。继续线为learned hard增益>=+0.50pp、clean>=−0.20pp、hard严格优于full；pp为百分点。只作单seed筛选，不宣称统计显著性。完成后停止；Main-Val、official test、新seed、调参、追加训练均未授权。

## 云端修复与Git同步约定

- 本地完成全部收口后以云端为主要开发位置，本地尽量只读；云端普通工程修复可连续完成最小检查、提交并推送至同名origin分支，再由本地拉取。
- 每次运行记录实际启动的完整commit SHA；代码/config在运行前提交。修复先停止原运行、保留日志、创建新提交再启动；按是否兼容完整恢复状态决定能否续训，不能静默weights-only重启冒称连续训练。错误修复后只重做受影响的最小检查，不自动无限重试。
- OOM、NaN/Inf、GradScaler skip、身份错误立即停止。改变batch/结构/loss/种子/预算/评价口径或无法确认恢复正确性时交人工确认，不做自动sweep。
- 数据划分、初始/最终权重、原始证据仍保留必要身份核验；普通工程文件hash和审查可以简化或交人工确认，待确认不得写成PASS。
- 文档里的SHA可指向对应代码基线或已经完成的交付快照，不追逐每次最新HEAD；最终交付SHA直接从Git读取，不为回填自身SHA制造反复dirty/commit。当前审核包记录生成时的Git快照，不冒称包含自身提交SHA。

## 本地Git交付与恢复点

- 分支`perf/mmfr-a2-v3-pipeline-opt1`；目标远端`origin`（`https://github.com/Newton-liub/DFormer.git`）同名分支；用户本次明确授权普通push，不force、不改写历史、不推upstream或其他remote。
- 本次纳入最新状态/审核材料、论文库报告、历史0.0.15 Canvas及对应索引；历史Canvas只是历史展示，不作当前实验口径。无内容diff的状态标记随暂存刷新，不人为修改代码。历史归档换行噪声不作为新成果提交。
- 完整收口快照 **`884ea1d2b9b5b08be30b53218f4c8feffab393e0`**（`chore(mmfr): close out workspace and authorize cloud results run`，14个文件）已成功推送至origin同名分支；直接`ls-remote`核验远端SHA相同，推送后工作区clean、无ahead/behind。原先两笔本地提交亦随普通push同步。此为2026-10-01 04:25 UTC前直接确认的事实，不代表云端已拉取。
- 本文件及复现信息现在记录上述已确认快照，随一笔文档回执提交同步；回执不改变运行代码或训练合同。为保持最终工作区干净，不回填回执自身SHA；最新HEAD直接从Git读取。审核包Git字段是生成时快照，不要求它等于包含该包的新提交。
- 本地实际检查仅限定Git差异、文本/JSON内容和既有审核包生成；未运行项目测试、GPU、训练、Val或云端，因为本次只交付版本与执行指令。
- 下一恢复点：云端先读取本文件和protocol §7，确认同步版本后实现入口并按上述授权链执行；代码/config里的2026-09-30未授权保护字段需由新入口显式落实本次授权，test禁用保持不变。

## 少量证据与独立事项

- [A-v1 protocol及云端指令](../../MMFR/01_research/mmfr_a_v1_action_utility_protocol.md)；[原Gate-B报告](../../MMFR/02_evidence/report_mmfr_a_v1_implementation_gateb_20260930.md)；[原证据](../../MMFR/02_evidence/mmfr_a_v1_gateb.json)；[现行审核入口](../../MMFR/99_review_packet_current/REVIEW_BRIEF.md)。2026-09-30完整本地收口事实由`e26d670`及原报告追溯，不改历史正文。
- R-OE-lite v2的inconclusive、F-lite Main-Val未复现及论文库实体裁决仍独立待决定，见开放决策和 [论文库报告](../reports/2026-09-30-paper-library-normalization-review.md)，不阻塞本次A-v1执行。旧入口修复已在`1b061cb`提交，不恢复为未提交阻塞。
