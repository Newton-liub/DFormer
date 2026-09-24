# MUSeg 当前开放问题

> **事实截至：** 2026-09-24；只记录会影响下一步行动且尚未裁决的选择，不承担运行状态或执行授权。当前事实与恢复边界见 [`MUSeg-current-status.md`](MUSeg-current-status.md)，既往已处置问题与完整决策史见 [`整理前归档`](../../MMFR/90_archive/2026-09-24_live_docs_rolling_window/README.md)。

## 1. E1 Batch 1B R-OE-lite 的 OOM 后处置（待用户决定，阻塞下一步）

现有冻结训练在 RTX 4090 上因 CUDA 显存不足中止，没有 fixed-final checkpoint 或 Quick-Val 结果。需要用户决定：**停止该批次**，或**提出并单独批准新的可行训练方案**。若改变模型、batch size 或共同合同，须建立新身份、重新资格化，并核对是否需要重训匹配的 C0；原训练/条件评价授权不自动覆盖新方案。先核验云实例控制面状态及 C0 checkpoint 所在位置，这两项是执行核查事项，不是科学方案的已作决定。证据：[`中止报告`](../../MMFR/02_evidence/report_e1_batch1b_roe_formal_training_attempt_20260924.md)、[`冻结协议`](../../MMFR/01_research/e1_batch1_protocol.md)。

## 2. E1 Batch 1A F-lite 的 Main-Val 处置（待上级独立裁决）

Batch 1A 十条件、十视图 Main-Val 的 F-lite 相对 C0 未复现四条件单视图 Quick-Val 的优势；事先没有为该批 Main-Val 预注册数值门槛，现有单 seed 描述性结果**不能自行判定 F-lite 去留或 C0 更好**。需要上级决定其后续研究处置；该判断独立于 Batch 1B 的 OOM 修复选择，不据此自动开放新的训练/评价。证据：[`Main-Val 分析报告`](../reports/2026-09-22-mmfr-e1-batch1a-c0-flite-mainval.md)。

已决定、已执行、失效和历史背景事项只在归档、协议或正式报告中追溯，不再追加到本文件。
