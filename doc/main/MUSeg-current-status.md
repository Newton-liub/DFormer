# MUSeg 当前状态与唯一实时入口

> **事实截至：2026-10-02（新方向计划与项目接入审计收口）。** 用户提供了方向 A（自然深度空洞形态重放）和方向 B（固定观测下的推理协议研究）两份 v1.0 方案。本轮完成限定现状核验与 [上级审计报告](../reports/2026-10-02-direction-plans-project-readiness-upper-review.md)，推荐 A 优先、B 暂只贡献共用单视图入口；这是待审建议，尚未选择/授权实现或实验。真正未决事项见 [开放决策](MUSeg-open-decisions.md)。

## 当前事实与实际意义

C0 是 E1 Batch 1A 的训练对照；fixed-final 是成功更新 2560 次后的固定权重。checkpoint 指权重/训练状态快照；mIoU 是平均交并比，pp 是百分点。support 指不含 padding 的图像区域；validity 指其中有无深度观测。S1 是新计划拟议的统一 padding 单视图，不等于旧 Quick-Val。

- 已有自然缺失调查仍有效：train-dev1277图/762组，val-dev318图/196组；train Depth 缺失率mean31.674541%，val30.148527%只描述。已核验dev的Depth16/Depth8零值支持一致，量化新增零0，不外推test。train P25/P75为0.10761048923865359/0.4944140559923207，对应val低/中/高80/174/64图。
- 本轮直接复核现存summary与限定源码，未重跑1595图统计或读取标签。真实空洞形态与随机网格删除不同，但 Replay 尚未实现、训练或取得收益；高缺失性能缺口仍待验证。
- **新接入风险已核验：**实际训练helper为`utils/dataloader/mmfr_training_v3.py::build_mmfr_training_batch_v3`，两份方案的`utils/mmfr_training.py`路径需在执行单纠正。loader没有返回空间变换/support元信息，旧helper用normalized零值推断support。旧C0配置继承A2混合故障与辅助可靠性头；`EncoderDecoder.forward`训练时要求完整可靠性监督并添加辅助loss。`utils/train.py`的旧E1入口固定candidate/protocol、base LR1e-5、预算及AMP/SyncBN等合同，不能仅复制旧配置改名接新三组。
- 上级方案提出原骨干/解码器三组全参数微调；首轮辅助头/损失、具体LR、BN/精度/scheduler、mask空间传递、删除量匹配及矩形不可达语义仍待定。本地建议仅分割目标、独立配置和薄入口，不改旧冻结合同。
- 方案A拟议三组各2560更新，共7680正式成功更新；四checkpoint×318图×三输入×S1共3816 view前向。方案B完整B1a拟议33072 view前向。均为预算计数，不是授权、耗时或实际运行结果。
- 本地C0/F-lite与NYUv2资产就绪证据仍不足；本轮未扩大搜盘、取回权重、下载数据或接云。恢复C0建议优先于重训；A首轮不必先恢复F-lite。CMX/GeminiFusion/ConD等完整MUSeg对照也未就绪。

**大白话：** 两份计划已经可以送审，但还不能直接开跑。先确认三组只差新增删除方式、取得正确C0并统一评价，再决定训练；本轮没有把上级方案自动升级为执行许可。

## 旧路线结论与执行边界

- A-v1正式Proposal1920/Gate640/global2560、skip0；批准NMF局部FP32从完整1280恢复，是混合历史精度续训，不是全轮FP32重训。唯一四条件Quick-Val三hard off/full/learned为50.770364019291435/50.766105718012604/50.76849101659144；learned−off−0.0018730026999946858pp未达冻结+0.50pp，仍 **stop**。
- F-lite历史Quick共同项均正，Main的clean/EM/Mis转负、SD仍正；padding、views、HAM随机、batch与环境同时变化，不能单因归因。Main无预注册去留线，原去留独立待审。
- R-OE v2 Quick约+0.01pp仍inconclusive、净归因/入口资格限制不变；Oracle-A NO-GO只关闭具体几何抑制动作族。新A不是旧A-v1，任何旧路线均未因本报告获得新实验权限。
- official test继续 **sealed_unread**。云资源最后直接观测仍来自2026-10-01旧记录；当前实例/GPU/计费状态待核验，本轮未操作资源生命周期。

## 证据入口与文档位置

- [本轮上级审计报告](../reports/2026-10-02-direction-plans-project-readiness-upper-review.md)：现状、代码风险、最小文件职责、分阶段预算与8项裁决；当前交付入口。
- 上级原方案：[A原稿](../../临时/MMFR_direction_A_natural_missing_2026-10-02.md)、[B原稿](../../临时/MMFR_direction_B_inference_protocol_2026-10-02.md)。保持原文与位置，未迁移；拟在接受后移至MMFR/01_research单一正文位置，不保留双份可编辑当前副本。
- [原本地调查](../reports/2026-10-02-direction-audit-local-evidence.md)：自然统计、完整Quick/Main差异、baseline/NYU/官方来源和历史结果。统计工具仍为`tools/mmfr/audit_museg_natural_missing.py`，summary/CSV为本地Git排除产物，无mask库。
- [A-v1正式报告](../../MMFR/02_evidence/report_mmfr_a_v1_formal_quickval_upper_review_20261001.md)、[既有复现JSON](../../MMFR/02_evidence/reproducibility_current.json)：训练runtime812507385a1b4b805966799bffb9a13f4704e492、评价runtimec798ed8f27483157ba6b164dafec4995bc47fce1与final身份沿用既有核验，不重hash。
- [MMFR导航](../../MMFR/00_control/PACKAGE_INDEX.md)、[计划索引](../plans/README.md)、现有report-index已登记本轮报告。最新候选与旧A-v1设计分开展示，不回写历史计划/报告。

## 本轮交付与下一恢复点

本轮基准HEAD `8b5860173749b999bdf272b77505a34e79e75a46`，分支`perf/mmfr-a2-v3-pipeline-opt1`；开始时ahead2且已有dirty。没有commit/push/fetch/reset、模型/配置改动、代码迁移、权重恢复、数据下载或云操作。旧归档行尾差异与用户原稿保持，不宣称当前工作区clean。

只新增审计报告并更新现有索引/导航/实时入口；检查为内容、关键源码/产物、限定链接/JSON结构和Git差异。完整测试、配置import、GPU/forward/测时、训练、新Quick/Main/test均未运行，纯报告按验证预算不扩验证。

`MMFR/99_review_packet_current/`仍是历史生成快照，既有A-v1 profile继续服务上次实验；本轮不手改生成产物、不新增profile/manifest/hash库或复制第二正文。直接提交本报告供上级阅读；若要求正式生成包，待明确canonical方案位置与审核范围后再用既有生成器重建。

**准确恢复点：**把本报告及两份原方案交上级，裁决分割-only/辅助头、实际LR、mask/support与匹配/压力测试合同，以及C0恢复、限定代码、GPU预检和正式训练各自预算。建议A先准备、B仅S1共用；未获新授权时保持停止。旧F-lite/R-OE/A-v1例外local-val决策独立，不自动关闭也不自动复活。
