# MUSeg 当前状态与唯一实时入口

> **事实与执行边界截至：2026-10-03。** 用户通过综合执行单授权最小目录整理与 Direction A 首轮 readiness。A（自然深度空洞形态重放）优先，B（推理协议研究）备用；本轮允许独立分割配置、LR `1e-6`、确定性输入配对与统一S1实现，精确C0和本机GPU就绪时才允许每组最多3次成功更新及极少量val-dev预检。正式三组各2560更新、完整评价、B1a、official test及付费云未授权。当前正在执行，最终证据与停止点将在收口时更新。

## 当前事实与实际意义

C0 是 E1 Batch 1A 的训练对照；fixed-final 是成功更新 2560 次后的固定权重。checkpoint 指权重/训练状态快照；mIoU 是平均交并比，pp 是百分点。support 指不含 padding 的图像区域；validity 指其中有无深度观测。S1 是新计划拟议的统一 padding 单视图，不等于旧 Quick-Val。

- 已有自然缺失调查仍有效：train-dev1277图/762组，val-dev318图/196组；train Depth 缺失率mean31.674541%，val30.148527%只描述。已核验dev的Depth16/Depth8零值支持一致，量化新增零0，不外推test。train P25/P75为0.10761048923865359/0.4944140559923207，对应val低/中/高80/174/64图。
- 本轮直接复核现存summary与限定源码，未重跑1595图统计或读取标签。真实空洞形态与随机网格删除不同，但 Replay 尚未实现、训练或取得收益；高缺失性能缺口仍待验证。
- **新接入风险已核验：**实际训练helper为`utils/dataloader/mmfr_training_v3.py::build_mmfr_training_batch_v3`，两份方案的`utils/mmfr_training.py`路径需在执行单纠正。loader没有返回空间变换/support元信息，旧helper用normalized零值推断support。旧C0配置继承A2混合故障与辅助可靠性头；`EncoderDecoder.forward`训练时要求完整可靠性监督并添加辅助loss。`utils/train.py`的旧E1入口固定candidate/protocol、base LR1e-5、预算及AMP/SyncBN等合同，不能仅复制旧配置改名接新三组。
- 本轮已采用原骨干/解码器三组全参数segmentation-only continuation，LR `1e-6`；BN/精度/scheduler按实际正式E1 C0 runner核对后写入新protocol。输入mask/support、删除量匹配及矩形不可达语义由本轮限定实现明确，不改旧冻结合同。实现与真实更新尚待本轮检查。
- 正式Round-1合同预算仍为三组各2560成功更新、合计7680，C0+三组×318图×三条件S1共3816 view前向；本轮不执行这两个正式预算。B完整B1a33072 view前向未授权。所有预算计数不表示实际运行结果。
- 本地C0尚在限定搜索/恢复中，F-lite/NYUv2资产不属本轮必须恢复范围；不取回额外baseline、不下载数据或启动付费云。恢复精确C0优先于重训，禁止另训C0代替。

**大白话：** 本轮已经获准准备三组只差新增删除方式的输入与训练链，并严格限量预检；正式训练仍要等就绪报告审核。目录整理和代码实现不代表Replay已取得收益。

## 旧路线结论与执行边界

- A-v1正式Proposal1920/Gate640/global2560、skip0；批准NMF局部FP32从完整1280恢复，是混合历史精度续训，不是全轮FP32重训。唯一四条件Quick-Val三hard off/full/learned为50.770364019291435/50.766105718012604/50.76849101659144；learned−off−0.0018730026999946858pp未达冻结+0.50pp，仍 **stop**。
- F-lite历史Quick共同项均正，Main的clean/EM/Mis转负、SD仍正；padding、views、HAM随机、batch与环境同时变化，不能单因归因。Main无预注册去留线，原去留独立待审。
- R-OE v2 Quick约+0.01pp仍inconclusive、净归因/入口资格限制不变；Oracle-A NO-GO只关闭具体几何抑制动作族。新A不是旧A-v1，任何旧路线均未因本报告获得新实验权限。
- official test继续 **sealed_unread**。云资源最后直接观测仍来自2026-10-01旧记录；当前实例/GPU/计费状态待核验，本轮未操作资源生命周期。

## 证据入口与文档位置

- [10月3日目录与代码职责审计](../reports/2026-10-03-project-directory-responsibility-audit.md)：当前结构、十项风险、文件职责/冻结边界与仅建议的处置清单；当前交付入口。
- [10月2日上级接入审计](../reports/2026-10-02-direction-plans-project-readiness-upper-review.md)：现状、代码风险、最小文件职责、分阶段预算与8项裁决；新方向研究裁决仍以此及开放决策为依据。
- canonical方案：[A正文（优先）](../../MMFR/01_research/MMFR_direction_A_natural_missing_2026-10-02.md)、[B正文（备用）](../../MMFR/01_research/MMFR_direction_B_inference_protocol_2026-10-02.md)。已原样迁至MMFR/01_research，临时目录不留第二份可编辑正文；历史代码/报告/配置冻结原位。
- [原本地调查](../reports/2026-10-02-direction-audit-local-evidence.md)：自然统计、完整Quick/Main差异、baseline/NYU/官方来源和历史结果。统计工具仍为`tools/mmfr/audit_museg_natural_missing.py`，summary/CSV为本地Git排除产物，无mask库。
- [A-v1正式报告](../../MMFR/02_evidence/report_mmfr_a_v1_formal_quickval_upper_review_20261001.md)、[既有复现JSON](../../MMFR/02_evidence/reproducibility_current.json)：训练runtime812507385a1b4b805966799bffb9a13f4704e492、评价runtimec798ed8f27483157ba6b164dafec4995bc47fce1与final身份沿用既有核验，不重hash。
- [MMFR导航](../../MMFR/00_control/PACKAGE_INDEX.md)、[计划索引](../plans/README.md)、现有report-index已登记本轮报告。最新候选与旧A-v1设计分开展示，不回写历史计划/报告。

## 本轮交付与下一恢复点

2026-10-03综合执行正在进行。开始HEAD为 `34a4dea41c1e08a5a1f85711eb97107471445272`，分支 `perf/mmfr-a2-v3-pipeline-opt1`；既有审计报告、report-index及导航/状态dirty保留，不把无关产物纳入提交。A/B科研正文已迁到canonical位置，默认旧输入/历史runner/config/model/证据不搬移或删除；必要loader扩展只允许opt-in。下一恢复点：目录整理本地提交后，冻结实际C0参数、限定恢复精确权重、完成三组输入/独立配置/S1与最小检查；满足条件才做授权内预检，最终报告后停止。

这轮是就绪准备，不是正式实验。`official test=sealed_unread`，不push，不启动付费云。历史目录审计详情继续由日期化报告留存；新阶段最终事实待实际检查后更新。
