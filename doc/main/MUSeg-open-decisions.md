# MUSeg 当前开放问题

> **执行边界截至：2026-10-03（NaturalMissing readiness获授权）。** A优先、B备用及本轮代码/限量预检边界已由综合执行单决定；这里只保留尚影响下一步的正式预算、资源与独立旧路线选择。事实与恢复点见 [实时状态](MUSeg-current-status.md)，历史输入与证据见 [本地调查](../reports/2026-10-02-direction-audit-local-evidence.md)。

## 0. NaturalMissing 正式训练与完整评价授权（readiness后待裁决）

2026-10-03综合执行单已接受 A 优先、B 备用，并授权正文迁移、独立 segmentation-only（只优化分割目标）配置、LR `1e-6`、严格分割权重加载及必要输入/S1代码准备。精确C0与本机现有GPU就绪时，每组最多3成功更新及极少量val-dev预检已获本轮条件授权；这些已决定事项不再列为未决。

仍未授权的是 Natural/Grid/Replay 各2560成功更新、C0+三组完整318图三条件S1、Direction B B1a与付费云。readiness报告完成后，按真实C0/加载/CPU/GPU结果决定是否批准正式预算、设备与费用责任；预检权重不能用作正式初始化，必须重新从同一精确C0干净启动。NYUv2及外部baseline不是本轮前置工作。

[A canonical正文](../../MMFR/01_research/MMFR_direction_A_natural_missing_2026-10-02.md)、[B备用正文](../../MMFR/01_research/MMFR_direction_B_inference_protocol_2026-10-02.md) 科研设计保留原文。类别支持与不可达矩形等最小工程口径在本轮执行protocol中明示，正式结果前冻结，不根据val分数修改。

**大白话：** 已经可以准备代码并做严格限量的本机预检，但只有就绪证据通过审核后，才可能开始三组正式训练。

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
