# MUSeg 当前开放问题

> **事实截至：2026-10-02（新方案接入审计）。** 用户提供方向A/B两份v1.0方案；已形成 [上级审计报告](../reports/2026-10-02-direction-plans-project-readiness-upper-review.md)，推荐A优先、B备用但未裁决/授权实现或实验。A-v1原筛选仍stop。这里只保留影响下一步的未决选择；事实与恢复点见 [实时状态](MUSeg-current-status.md)，原输入与历史证据见 [本地调查](../reports/2026-10-02-direction-audit-local-evidence.md)。

## 0. 新方向、训练目标与分阶段权限（待上级裁决）

方向A拟比较等预算自然输入续训、有效删除量匹配的网格增强和train-only真实空洞形态重放；方向B拟研究固定观测下S1/S2/S10（1/2/10视图）的协议依赖。两份原稿已读，尚无实现或新收益结果。推荐先A，仅借用B的统一padding单视图S1；S1是新口径，不是旧Quick-Val。

需要上级冻结以下下一步选择（完整8项见上级审计报告§8）：

- 是否接受A优先、B暂不立项；旧F-lite/R-OE仅本轮搁置，不把未裁决误写为STOP。
- 是否采用仅分割目标、关闭旧辅助可靠性训练。源码确认旧C0继承A2故障与辅助头，旧forward训练时强制完整辅助监督；新合同不能默默沿用。若不实例化辅助头，是否接受明确排除其旧键并严格加载全部分割键。
- 首轮实际LR及BN/精度/scheduler。C0 E1起始LR1e-5，方案的0.1倍按此解释为1e-6；旧A2起始6e-5不是同一版本。新值不能通过假冒旧E1合同接入。
- 源mask变换、非padding support、目标当前有效点、确定性配对输入、有限匹配/共同跳过与矩形压力不可达语义；源只用train-dev，val规则不参与训练。实际旧builder在`utils/dataloader/mmfr_training_v3.py`，loader未传空间元信息、旧helper只推断support，不能按原稿错误路径假定接口已就绪。
- 是否接受方案A门槛、固定train阈值分层，以及类别支持/采集组敏感性口径。高缺失64图不是64个完全独立场景，门槛不等于显著性。
- 精确C0恢复与限定代码准备范围；是否单独批准GPU每组最多3成功更新预检、正式三组各2560更新/一次S1评价，并明确设备、时间/费用和资源责任。以上预算均未获本轮授权。
- 是否同意方案正文接受后迁至MMFR/01_research单一位置、少量新文件和必要公共原语抽取；旧冻结入口暂不搬移删除，不做全仓重构。

本地C0/F-lite与NYUv2 raw/filled数据仍缺就绪证据。A首轮建议只恢复C0，NYUv2/外部baseline/B1a在方向通过或另获授权后再准备。当前不授权代码、权重下载、GPU、训练/评价、official test或云资源操作。

**大白话：** 新计划已经明确了研究问题，但旧配置不能直接拿来运行；先定训练目标和预算、取得正确权重，再实施最小输入增强与统一评价。

## 1. A-v1 stop后是否例外授权本地val（待上级回复）

learned三hard−matched off为−0.0018730026999946858pp，未达冻结+0.50pp，默认继续停止。原报告与条件性转移包只是资料交付，本次方向调查也没有复活local-val授权。

若上级例外同意，必须明确四条件重评或十条件Main-Val、off/full/learned行为、视图/随机配对/指标、硬件/时间预算和最小资格权限。A-v1现入口只有四条件单view，Main adapter尚缺。见 [A-v1正式报告](../../MMFR/02_evidence/report_mmfr_a_v1_formal_quickval_upper_review_20261001.md)、[条件性交接](../../MMFR/02_evidence/handoff_mmfr_a_v1_local_val_conditional_20261001.md)。不开放official test、新seed或训练。

## 2. R-OE-lite v2 去留与入口资格（待用户/上级裁决）

已有2560成功更新，四条件Quick-Val相对C0 clean0.00、EM/SD/Mis各+0.01pp，三hard50.77→50.78，inconclusive。EM318/318触发substitute，其余各0/318 exact bypass；base权重有漂移、未运行同权重forced-bypass，EM+0.01不能净归因substitute。需要决定停止、例外进入Main-Val或先补该对照，任何新运行都需独立授权；原收益远低于EM+0.50pp继续线。

R-OE-aware薄入口的original-full `V_geom`全1定义尚无既有冻结资格；旧身份修复与入口已纳入Git，开放的是接受/改写/替换资格，不再是“未提交产物处置”。现有数值仅screening参考，不自动触发重跑。见 [R-OE v2报告](../../MMFR/02_evidence/report_e1_batch1b_roe_v2_formal_training_quickval_20260924.md) 和 [E1 protocol](../../MMFR/01_research/e1_batch1_protocol.md)。

## 3. F-lite Main-Val 的独立处置（待上级裁决）

十条件十view Main-Val未复现四条件单view Quick-Val优势，M6差−0.3966pp；四共同项中clean/EM/Mis转负、SD仍正。事前无Main数值门槛，单seed描述结果不能自行判stop或C0更好。本次补齐协议差异，但仍无法单独归因多尺度/flip；去留和是否验证协议因素需另行裁决。见 [Main-Val报告](../reports/2026-09-22-mmfr-e1-batch1a-c0-flite-mainval.md) 与本次调查§6–7，不自动授权训练/评价。

## 4. 论文库遗留实体与人工编号归属（待用户确认）

既有DFormerv2另一份抽取与supplemental/PDF-only材料的归属仍待裁决；PR090重复实体留在仓库外duplicates_review，已有正文/资源数量判断不等于全包逐字节相同。RE042/RE053/RE188/RE447旧编号—题名未经总索引确认，仍保留unverified_manual_ids，不自动绑定。旧归档 `D:\0Project\DFormer-archive-20260922\doc\paper` 的恢复与补扫仍需用户决定。

现行实体数量以 [机器总索引](../../MMFR/03_reference/PAPER_LIBRARY_INDEX.json) 为准，不把旧32条快照当当前数量；本次方向调查不扩大到全论文库重审，也不自动处置这些遗留项。见 [paper-index](../../MMFR/03_reference/paper-index.md)。它们不改变实验边界。
