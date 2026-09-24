# MUSeg 当前开放问题

> **事实截至：** 2026-09-24；只记录会影响下一步行动且尚未裁决的选择，不承担运行状态或执行授权。R-OE-lite v2 已完成正式训练与四条件 Quick-Val，执行边界见 [`MUSeg-current-status.md`](MUSeg-current-status.md)；既往完整决策史见 [`整理前归档`](../../MMFR/90_archive/2026-09-24_live_docs_rolling_window/README.md)。

## 1. R-OE-lite 路线的去留（待用户裁决）

R-OE-lite v2 已在 RTX 4090 上完成 2560/2560 次成功更新并生成 fixed-final checkpoint，机制证据确认 substitute 真的被训练。但四条件 Quick-Val（单视图 318/318 `val-dev`，matched control 为 Batch 1A C0）相对 C0 为 clean `0.00`、`entire_missing@1.0` `+0.01`、`spatial_dropout@0.75` `+0.01`、`misalignment@0.75` `+0.01` pp，$M_{3,\mathrm{hard}}$ `50.77 → 50.78`，判定 `inconclusive`。可选处置：

- **按停止处理、不再进入十条件 Main-Val**：理由是该 screening 差值远低于 `e1_batch1_protocol.md` §11/§13.5 中 Batch 1B Main-Val promote 所需的 `entire_missing@1.0` `+0.50` pp 门槛；十条件 Main-Val 未获授权，需要新的单独授权才会启动。
- **仍进入十条件 Main-Val**：只在前一项被显式否决时考虑，且同样需要单独的运行授权。
- **先补“同权重下强制 bypass”对照**：用来把 substitute 的净因果效果与 base 权重漂移分开。本轮已观察到 clean/SD/Mis 三个条件路由是严格 exact bypass，其逐样本 mIoU 仍有约一半样本与 C0 不同（clean 聚合值相同），说明 base 漂移本身就能产生 0.00–0.01 pp 量级的聚合变化，因此 EM 的 `+0.01` pp 不能单独归因于 substitute。该对照尚未运行、也未获授权。

证据：[`v2 训练与 Quick-Val 报告`](../../MMFR/02_evidence/report_e1_batch1b_roe_v2_formal_training_quickval_20260924.md)、`cloud/mmfr-e1-batch1b-roe-v2/quickval-comparison.json`。

## 2. 本轮两个未提交产物的处置（待用户裁决）

- `utils/train.py` 入口白名单的 v2 身份修复（未提交；补丁 `cloud/mmfr-e1-batch1b-roe-v2/roe-v2-entry-fix.patch`）：需要用户决定是在本地提交并推送，还是在云端补提交，或改用其他实现。**提交与推送未经用户确认不得执行。**
- 新增的 R-OE-aware Quick-Val 入口 `tools/mmfr/e1_quickval_roe.py`（未提交、无既有冻结资格，`V_geom` 在无 crop/pad 的 `original-full` 视图下取全 1 单位掩码）：需要用户决定接受、改写或替换实现；在复核前其数值只作 screening 参考。

## 3. E1 Batch 1A F-lite 的 Main-Val 处置（待上级独立裁决）

Batch 1A 十条件、十视图 Main-Val 的 F-lite 相对 C0 未复现四条件单视图 Quick-Val 的优势；事先没有为该批 Main-Val 预注册数值门槛，现有单 seed 描述性结果**不能自行判定 F-lite 去留或 C0 更好**。需要上级决定其后续研究处置；该判断独立于 Batch 1B 的路线选择，不据此自动开放新的训练/评价。证据：[`Main-Val 分析报告`](../reports/2026-09-22-mmfr-e1-batch1a-c0-flite-mainval.md)。

已决定、已执行、失效和历史背景事项只在归档、协议或正式报告中追溯，不再追加到本文件。
