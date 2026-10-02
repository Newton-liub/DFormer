# MUSeg 当前开放问题

> **事实截至：2026-10-02。** 本次方向A/B本地摸底已完成，尚未选择论文方向或授权实现/实验。A-v1原四条件筛选仍为stop。本文只保留影响下一步的未决选择；事实与恢复点见 [实时状态](MUSeg-current-status.md)，完整证据见 [本次调查报告](../reports/2026-10-02-direction-audit-local-evidence.md)。已关闭事项不重新开放。

## 0. 两个候选方向与下一步最小授权（待上级裁决）

方向A关注MUSeg原始Depth无观测区域的分布与train-only mask replay接入；方向B关注F-lite在不同推理协议中的相对收益反转及严格matched推理。**本次只有输入/源码/旧结果事实，没有新方法或收益结果。**

上级需选择方向、最小baseline和第二数据集，并决定是否另行授权replay或统一推理薄入口。方向A若推进，mask来源只能是train-dev，须先明确原始support与resize后零值的语义；val-dev统计只描述。方向B若推进，须冻结padding、视图分组、HAM随机政策与环境，区分各profile内同预算和1/2/10view跨profile不等成本。

当前本地MUSeg权重与NYUv2 raw/filled数据就绪证据不足；需决定先恢复哪些既有权重、是否补第二数据集，以及允许的适配/运行预算。具体8个问题见调查报告§12。本项不授权训练、模型开发、新评价、GPU、official test或云资源操作。

**大白话：** 先选值得继续验证的问题，再明确能改什么、能跑什么；本次调查完成后保持停止。

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
