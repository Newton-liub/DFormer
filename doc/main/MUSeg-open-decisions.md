# MUSeg 当前开放问题

> **执行边界截至：2026-10-03（NaturalMissing readiness获授权）。** A优先、B备用及本轮代码/限量预检边界已由综合执行单决定；这里只保留尚影响下一步的正式预算、资源与独立旧路线选择。事实与恢复点见 [实时状态](MUSeg-current-status.md)，历史输入与证据见 [本地调查](../reports/2026-10-02-direction-audit-local-evidence.md)。

## 0. NaturalMissing 正式训练与完整评价授权（readiness后待裁决）

2026-10-03真实执行收口仍为 **PARTIALLY READY**，见[就绪报告](../reports/2026-10-03-natural-missing-round1-readiness.md)与[执行合同](../../MMFR/01_research/natural_missing_round1_protocol.md)。精确C0已CPU-only取回、立即确认云实例Stopped，关机后一次SHA匹配，真实严格加载802分割键/只排除10辅助键/missing0/unexpected0。2图真实C0 S1完成6次finite前向，仅工程预检；资产恢复/登录途径不再是待决事项。

**当前直接阻塞下一步的是设备和限量预检授权：** Natural/Grid/Replay各从同C0独立启动，batch10/480×640在RTX5060 Laptop 8GB各首次forward OOM，successful均0，尚无loss/backward/optimizer/scaler/保存通过或有效step测时。是否另批4090 24GB或同等级更大显存设备的每组≤3成功更新预检，并明确设备、费用和时间上限？当前不改变合同制造本机PASS，也不从失败进程wall外推正式耗时。本轮CPU-only取回权限已经执行并停机，不允许据此再次开机或启动GPU云资源。

通过新的限量预检后，才裁决Natural/Grid/Replay各2560成功更新、C0+三组完整318图三条件S1及正式设备/费用责任。正式训练/完整评价/Main-Val/B1a和其他数据集/baseline预算仍独立关闭。预检权重不得用作正式初始化，正式运行必须从同一个核验C0干净启动；类别支持、原始全图固定分层和矩形固定几何语义已确定，不按小样本分数改写。

[A canonical正文](../../MMFR/01_research/MMFR_direction_A_natural_missing_2026-10-02.md)、[B备用正文](../../MMFR/01_research/MMFR_direction_B_inference_protocol_2026-10-02.md)科研内容保持原文。NYUv2和外部baseline不是本轮恢复前置项。

**大白话：** 共同底座和极小真实评价已通过工程检查，本机训练因显存不足停在首次前向；先独立裁决更大显存的限量预检及费用，通过后再讨论正式预算。

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
