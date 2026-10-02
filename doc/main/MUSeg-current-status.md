# MUSeg 当前状态与唯一实时入口

> **事实截至：2026-10-02（本地事实摸底收口）。** 已完成 MMFR 两个候选论文方向的限定调查：自然 Depth missing、F-lite 推理协议差异、本地 baseline/NYUv2 与官方参考代码。唯一主要交付为 [方向调查报告](../reports/2026-10-02-direction-audit-local-evidence.md)。本轮只做CPU输入统计、限定源码/旧结果/资料核验，不训练、不新评价、不用GPU；完成后停止，等待上级选择方向。真正未决事项见 [开放决策](MUSeg-open-decisions.md)。

## 当前事实与实际意义

Depth missing 指原始深度中没有有效观测的零值区域；train-dev/val-dev 是冻结的训练开发集/验证开发集，均不包含 official test。checkpoint 是保存的模型权重/状态；mIoU 是平均交并比，pp 是百分点。

- 只按dev allowlist读取 RGB/Depth/Depth16：train-dev1277图/762组、val-dev318图/196组；每图1082×932。Label与official-test清单/图片/标签/cache未读。固定视场裁剪外区域已不在分母中。
- train自然Depth8无效率mean **31.674541%**、median25.875029%、P10/P90=3.966645%/69.449140%；val仅描述为mean30.148527%/median26.876492%。所有dev图固定16→8位映射一致，Depth16>0但Depth8==0的新增零像素为0；不外推到test。
- 自然洞兼有大连通区域与小区域，矿井组成有明显差异；当前spatial_dropout是随机网格块删除，不是自然mask或pixel Bernoulli。train-only replay只有接入事实清单，未保存mask库或实现增强。val统计不得用于定训练规则。
- F-lite四个共同条件在Quick-Val均正差，Main-Val clean/EM/Mis转负、SD仍正；HAM（含随机非负矩阵分解的解码头）随机消耗、视图、padding、batch分组和硬件/软件栈也同时改变，不能单独归因多尺度/flip。统一matched单/双/十view入口未实现/运行；各profile内可同预算，跨profile1/2/10view不等成本。
- 指定本地checkpoint路径盘点未找到可立即推理的Windows MUSeg权重；已有代码/config/远端身份不等于本地文件。NYUv2已知本地数据路径未找到，raw/filled版本待核验；loader无显式补洞不证明磁盘raw或保留16位单位。
- 仅新增官方MMSS浅clone于 `D:\0Project\origin\MMSS`，复用已有官方DFormer；未取大数据/权重。MMSS依赖DELIVER，非现成MUSeg evaluator，随机条件无sample-keyed观测复用接口，许可未确认。GeomPrompt官方来源未找到可确认源码链接，未clone第三方。

**大白话：** 方向A有真实输入分布证据，方向B有历史评分反转与协议混杂证据；两者都尚无新的收益实验。本次调查没有决定论文方向、创造新方法或复活旧训练授权。

## 已完成旧路线的结论仍有效

A-v1的Proposal是stage2候选补偿残差，Gate是每图连续补偿强度；off/full/learned分别是禁用/全量/按Gate补偿，同checkpoint与输入/随机状态配对。正式Proposal1920/Gate640/global2560、skip0；用户批准保留完整前1280，仅训练NMF局部FP32执行剩640+640，是混合历史精度续训，不是全轮FP32重训。

唯一四条件Quick-Val为318/条件、original-full scale1/no flip、FP32/TF32off、seed2026091401/reset-per-unit。三hard off/full/learned **50.770364019291435 / 50.766105718012604 / 50.76849101659144**；learned−off **−0.0018730026999946858pp** 未达冻结+0.50pp，结论 **stop**；clean保护与learned>full通过不改变该结论。单seed筛选，不作统计显著性、通用优劣或创新结论。

F-lite Main-Val无预注册去留门槛，R-OE Quick-Val为inconclusive且资格/净归因限制仍在；Oracle-A NO-GO只关闭具体validity-aware pairwise geometry suppression动作族。旧结果不因本次调查改写，旧路线裁决与方向A/B选择分开。

## 证据入口与身份恢复点

- [本次十二章报告](../reports/2026-10-02-direction-audit-local-evidence.md)：自然统计、corruption/replay入口、完整Quick/Main矩阵、matched限制、baseline/NYU/官方代码、四历史摘要及8个上级问题。
- [统计工具](../../tools/mmfr/audit_museg_natural_missing.py)，本地 `outputs/direction-audit-20261002/summary.json` 与两份per-image CSV；中间证据说明也在该输出目录，不是第二份实时状态。没有保存mask、标签或预测。
- [A-v1正式上级报告](../../MMFR/02_evidence/report_mmfr_a_v1_formal_quickval_upper_review_20261001.md)、[既有复现JSON](../../MMFR/02_evidence/reproducibility_current.json)：训练runtime `812507385a1b4b805966799bffb9a13f4704e492`、评价runtime `c798ed8f27483157ba6b164dafec4995bc47fce1`；final SHA `87ec54d10192d3aedc1d6edb864b7b60b94ce66220a748a12f1ca50ca605d336`，沿用既有核验，不重hash。本地原AV1/ROE cloud目录未读到，本轮不声称重算其评分。
- [条件性本地交接](../../MMFR/02_evidence/handoff_mmfr_a_v1_local_val_conditional_20261001.md)、[转移收据](../../MMFR/02_evidence/delivery_mmfr_a_v1_local_val_20261001.md)：原ZIP/checkpoint仍是远端仓库外资产；未下载或验证本地模型。A-v1十条件多视图Main adapter尚未实现，收包不授权评价。
- [MMFR导航](../../MMFR/00_control/PACKAGE_INDEX.md)、[外部代码索引](../../MMFR/03_reference/code-index.md)。正式报告登记于现有report-index，不新建manifest/hash系统。

## 执行、交付与下一恢复点

本轮基准HEAD `459908b0ed5f75b38858ea5b708864ff4cc4beb6`，分支 `perf/mmfr-a2-v3-pipeline-opt1`；仅开始时做一次Git状态确认，未pull/push/commit或创建空commit。新增统计工具、报告与导航/状态小文档留作本地未提交交付；原checkpoint、数据、旧结果与外部源码没有改写。历史远端交付回执仍见原正式报告，不据此声称当前工作区clean。

本轮检查已完成：1595图限定CPU输入统计、主代理关键统计/源码/旧结果/官方来源复核；统计工具定点静态诊断无报告项；report-index与summary的JSON解析成功，限定文档实际diff与格式检查通过。未重复全量图像统计或新增临时测试脚本。

`MMFR/99_review_packet_current/`仍为历史生成快照；本轮明确不创建manifest/审核包，不手改生成产物。既有A-v1 profile继续服务上一次实验审核，本次直接提交独立方向报告供上级阅读，不能把旧包当本次调查入口。

云资源最后直接观测仍来自2026-10-01旧记录；本轮未连接云端、检查GPU/计费或关闭/销毁资源，当前云状态待核验。official test继续 **sealed_unread**。

下一恢复点：将本次报告交上级裁决方向A/B、baseline与第二数据集/最小实现预算。默认保持停止；只有新的明确授权才进入mask replay、matched薄入口、权重转入、模型/数据准备或任何评价。A-v1例外local-val与R-OE/F-lite旧去留独立开放。内容/差异与限定输入证据检查不等于新实验；完整测试、GPU、训练、新Quick/Main/test均未运行。
