# MUSeg 当前状态与唯一实时入口

> **事实截至：2026-10-01 15:33 UTC。** A-v1正式训练与唯一四条件Quick-Val已完成，筛选 **`stop`**；本轮已完成上级审核报告和条件性本地val转移包，正在做文档Git交付。用户要求无GPU继续，本轮只做文件/哈希/文档工作，不加载模型或运行新实验。新的本地val及A-v1 Main-Val适配仍需明确授权。真正未决选择见 [开放决策](MUSeg-open-decisions.md)。

## 当前结论与执行边界

A-v1的Proposal是stage2候选补偿残差，Gate是每图连续补偿强度选择器；off禁用补偿、full全量补偿、learned按Gate强度补偿。matched off是同一checkpoint（完整权重/状态文件）、同输入和配对随机状态的禁用补偿对照。mIoU是平均交并比，pp是百分点。

**大白话：** 训练完成，但困难条件平均收益没有通过冻结门槛。本次报告和包可以用于汇报、下载保存；它们不会自动授权本地评价，也没有补齐Main-Val运行入口。

- 正式逻辑Proposal1920/Gate640/global2560、skip0；用户批准`fp32-full1280-resume`，保留原前1280完整状态，仅训练NMF（原HAM解码头内的非负矩阵分解）局部FP32，执行剩640+640。其他AMP（自动混合精度）/scaler1024/TF32、结构/loss/batch/seed/预算不变；是混合历史精度续训，不是全轮FP32重训。
- 唯一四条件评分：318 val-dev/条件、original-full、scale1/no flip、FP32/TF32 off、eval seed2026091401、reset-per-unit、off/full/learned配对HAM随机状态。首个评价初始化数组比较错误发生在样本forward前，修复后仅一轮实际评分，旧停止证据保留。
- 三hard未加权平均off/full/learned：**50.770364019291435 / 50.766105718012604 / 50.76849101659144**；learned−off **−0.0018730026999946858pp**，要求≥+0.50pp，FAIL。clean差值−0.010031370364394832pp≥−0.20pp，PASS；learned hard−full+0.0023852985788366254pp>0，PASS；full hard−off−0.004258301278831311pp。full/learned均低于matched off，原规则分类`stop`。
- 单seed筛选，不作统计显著性、通用模型优劣或论文创新结论。当前停止训练/评价，不自动Main-Val、official test、新seed、sweep、预算追加或模型重设计。

## 本次上级报告与转移包（已完成）

- [正式上级报告](../../MMFR/02_evidence/report_mmfr_a_v1_formal_quickval_upper_review_20261001.md)：覆盖上一份2026-09-30Gate-B报告之后的入口实现、原失败/限定诊断、批准精度修订、正式续训与真实评分；精确表格、结果限制和上级裁决项均有证据入口。
- [本地条件性交接](../../MMFR/02_evidence/handoff_mmfr_a_v1_local_val_conditional_20261001.md)：收包/核hash/源码与数据路径准备；明确现有A-v1入口只支持四条件单视图，十条件多尺度/flip Main-Val adapter未实现，不能直接套用旧入口或伪造Git身份。
- [交付收据](../../MMFR/02_evidence/delivery_mmfr_a_v1_local_val_20261001.md)：ZIP在仓库外 `/root/rivermind-data/cloud/MMFR_AV1_local_val_conditional_20261001.zip`，配套`.zip.sha256`；**429993998 bytes、996成员**，整包SHA **`1975bde09fcde5924dc3c947f6762e59db51b1e0a2ae1ed25ca4410a07fadcd7`**。
- 包含A-v1最终/C0原完整checkpoint、基线`4f84b469c4b04de657eaf2c455bd44e991b7f869`的973个限定源码/配套文件、train/val-dev清单、报告/protocol/交接与选定原始小证据。dataset、official-test清单/图片/标签/cache、大日志、父1280/transition/诊断checkpoint、外部论文/clone、Git历史和旧生成审核包不入包。包不是全训练恢复归档。
- 打包核验PASS：全部成员字节SHA与原输入一致、CRC、唯一安全路径、两份原checkpoint和dev split身份；4条件×318及stop与final身份匹配。两份checkpoint不重新序列化；未访问dataset/test或执行模型。打包前JSON解析与26份canonical Markdown路径链接检查PASS；最终27份MMFR canonical文档及2份实时文档的路径链接、报告登记/profile源与收据身份检查均PASS（不认证全部fragment或跨平台渲染）。

## 关键身份与原始证据恢复点

- 训练runtime **`812507385a1b4b805966799bffb9a13f4704e492`**；评价runtime **`c798ed8f27483157ba6b164dafec4995bc47fce1`**，均执行前已push/远端核验。数值合同SHA **`5f437c5b470077476f28b8e1ecceddf119cc402045e723f5a1aadbc519c6cc2e`**；删除唯一precision字段后与原合同SHA`96e93e02bfb2ad8214ab26f826ca66728a80231e16133f81a04709036578f7cd`精确一致。定义见 [protocol §8](../../MMFR/01_research/mmfr_a_v1_action_utility_protocol.md)。
- 最终 `cloud/mmfr-av1-formal-fp32-resume1280-v1/formal/checkpoint/update-2560.pth`，SHA **`87ec54d10192d3aedc1d6edb864b7b60b94ce66220a748a12f1ca50ca605d336`**；C0仓库外 `/root/rivermind-data/cloud/MMFR_E1_Batch1A_local_transfer_20260922/C0/checkpoint/update-2560.pth`，SHA **`ca618b23d18eabb201a0d11d18da383ac99576d0feae5864e3233bda527d9a1a`**。
- 唯一批准父完整1280：原runtime`91c7f96d3768fde0f478e03a4ccda8c6f19368dd`，原`cloud/mmfr-av1-formal-v1/formal/checkpoint/update-1280.pth`，SHA`05714d165925c4549aa1111564bbca7d8b8d9896d337c5b68f0b7bb98590564d`。恢复model+AdamW+scaler+scheduler+四类RNG+cursor；未用nonresumable诊断1870权重。
- train-dev1277 SHA`a6b15b63f6d5193e3928ea24ada25be403a48e68d1c1f9372cdbbc3fe5cd8470`；val-dev318 SHA`1d0719d8f64f016d48995c25ab66d4004d76b7155d9efeef7cbb7454c0dd0e83`。原dataset `/root/rivermind-data/dataset/MUSeg_DFormer`；本地目标数据和跨硬件行为未验证，official test **`sealed_unread`**。
- 原运行权威结果：`cloud/mmfr-av1-formal-fp32-resume1280-v1/training-result.json`、`precision-amendment-compatibility.json`、两阶段result与步骤日志；`quickval-config-equality-fix-v1/summary.json`及4份metrics。训练恢复段1539.0116258771159s，评价607.959805s；不是完整冷启动训练耗时。
- 既有独立核验PASS：final/transition完整状态finite及lineage，812个冻结C0张量精确一致、Proposal与transition一致；实际首clean strict off logits相等/max_abs_error0、分支调用0；3816逐样本confusion重算指标与门槛，1280训练记录连续/finite/applied/scale1024/skip0。无额外GPU forward。历史1871失败、590条重放和三现场对照边界由正式报告及 [既有复现JSON](../../MMFR/02_evidence/reproducibility_current.json) 留存，不重复加入实时流水账。

## Git交付、生成包限制与云资源

分支 `perf/mmfr-a2-v3-pipeline-opt1`，origin `https://github.com/Newton-liub/DFormer.git`。已有实验结果提交`65d89c1dc03f762fc91316ebd4aef2b3b655acc2`及回执基线`4f84b469c4b04de657eaf2c455bd44e991b7f869`已普通push并远端核验。本轮报告/收据/索引/profile/实时文档的提交与push尚待收口；用户已明确授权同分支普通push，禁止强推/改历史。checkpoint、ZIP、大日志和逐样本cache不进Git/MMFR。

`MMFR/99_review_packet_current/`仍为旧生成快照：15:33 UTC再次直接确认无`pwsh`，未安装PowerShell、未手工改生成产物。既有profile源已指向本轮报告；当前可直接转交正式报告。正式六文件入口仍需在PowerShell可用环境先核验canonical链接，再由原`MMFR/98_tools/rebuild_review_packet.ps1`重建，旧包不可冒充最新结果。

云资源**最后直接核验**仍为09:30:44 UTC：无A-v1实验进程，GPU1MiB/0%，实例开启。用户15:27 UTC要求无GPU模式继续，本轮不重新检查GPU、不关闭/销毁资源；不能据旧观测声称现在GPU或计费状态已核验。

下一恢复点：完成文档Git交付后等待上级对报告的回复，默认维持`stop`；若例外批准本地val，先明确范围与适配/最小资格/运行预算再执行。本次未运行完整测试、模型导入、GPU、训练、重复Quick-Val、Main-Val或test，检查只覆盖报告与文件交付风险。
