# MUSeg 当前开放问题

> **事实截至：** 2026-10-01 15:33 UTC。A-v1唯一四条件Quick-Val已完成并筛选`stop`；上级报告和条件性本地val资料包已准备，当前实验授权执行结束。新增裁决仅为上级是否例外同意新的本地验证，不阻塞资料交付。只记录仍真正未裁决事项，参数/预算/筛选线不重复开放。执行边界见 [`MUSeg-current-status.md`](MUSeg-current-status.md)；既往完整决策史见 [`整理前归档`](../../MMFR/90_archive/2026-09-24_live_docs_rolling_window/README.md)。

## 0. A-v1 筛选后是否例外授权本地val（待上级回复）

本轮learned三hard相对matched off为−0.0018730026999946858pp，未达冻结+0.50pp门槛，结论`stop`。**默认继续停止实验**，报告和打包只是资料交付；用户提出“若上级模型同意后的本地val”没有授权立即运行。

上级需决定是否有充分研究理由例外追加验证。若同意，必须明确四条件重评或十条件Main-Val、off/full/learned行为范围、视图/随机配对/指标口径、硬件与时间预算，以及必要适配和最小资格核验权限。A-v1现有入口仅四条件单视图，Main-Val adapter未实现，收到包不等于可直接启动。参见 [`上级报告裁决项`](../../MMFR/02_evidence/report_mmfr_a_v1_formal_quickval_upper_review_20261001.md) 和 [`条件性交接`](../../MMFR/02_evidence/handoff_mmfr_a_v1_local_val_conditional_20261001.md)。本项不开放official test、新seed、训练或调参。

**大白话：** 先看上级是否认为值得再评，再确认怎么评；在明确同意之前保持停止。原文献创新性与全文缺口仍为论证局限，不据此重新搜索或重设计。以下旧路线问题保持独立。

## 1. R-OE-lite 路线的去留（待用户裁决）

R-OE-lite v2 已在 RTX 4090 上完成 2560/2560 次成功更新并生成 fixed-final checkpoint，机制证据确认 substitute 真的被训练。但四条件 Quick-Val（单视图 318/318 `val-dev`，matched control 为 Batch 1A C0）相对 C0 为 clean `0.00`、`entire_missing@1.0` `+0.01`、`spatial_dropout@0.75` `+0.01`、`misalignment@0.75` `+0.01` pp，$M_{3,\mathrm{hard}}$ `50.77 → 50.78`，判定 `inconclusive`。可选处置：

- **按停止处理、不再进入十条件 Main-Val**：理由是该 screening 差值远低于 `e1_batch1_protocol.md` §11/§13.5 中 Batch 1B Main-Val promote 所需的 `entire_missing@1.0` `+0.50` pp 门槛；十条件 Main-Val 未获授权，需要新的单独授权才会启动。
- **仍进入十条件 Main-Val**：只在前一项被显式否决时考虑，且同样需要单独的运行授权。
- **先补“同权重下强制 bypass”对照**：用来把 substitute 的净因果效果与 base 权重漂移分开。本轮已观察到 clean/SD/Mis 三个条件路由是严格 exact bypass，其逐样本 mIoU 仍有约一半样本与 C0 不同（clean 聚合值相同），说明 base 漂移本身就能产生 0.00–0.01 pp 量级的聚合变化，因此 EM 的 `+0.01` pp 不能单独归因于 substitute。该对照尚未运行、也未获授权。

证据：[`v2 训练与 Quick-Val 报告`](../../MMFR/02_evidence/report_e1_batch1b_roe_v2_formal_training_quickval_20260924.md)、`cloud/mmfr-e1-batch1b-roe-v2/quickval-comparison.json`。

## 2. R-OE-aware Quick-Val 入口的资格处置（待用户裁决）

旧 `utils/train.py` v2身份修复和 `tools/mmfr/e1_quickval_roe.py` 已在本地commit `1b061cbabaf76575b0812e1513d953b02cdb6254` 纳入Git，当前无内容差异，已不是未提交产物处置。仍真正开放的是 R-OE-aware Quick-Val入口**资格**：`original-full` 的 `V_geom` 全1定义及新薄入口未有既有冻结资格，需要上级决定接受、改写或替换；在此之前原数值只作screening参考。此问题不改变A-v1合同或自动授权重跑。

## 3. E1 Batch 1A F-lite 的 Main-Val 处置（待上级独立裁决）

Batch 1A 十条件、十视图 Main-Val 的 F-lite 相对 C0 未复现四条件单视图 Quick-Val 的优势；事先没有为该批 Main-Val 预注册数值门槛，现有单 seed 描述性结果**不能自行判定 F-lite 去留或 C0 更好**。需要上级决定其后续研究处置；该判断独立于 Batch 1B 的路线选择，不据此自动开放新的训练/评价。证据：[`Main-Val 分析报告`](../reports/2026-09-22-mmfr-e1-batch1a-c0-flite-mainval.md)。

## 4. 论文库遗留实体与人工编号归属（待用户确认）

本次已形成 32 条 canonical LIB 记录，DFormerv2 的另一份不同抽取仍在旧目录，正式 supplemental 与 PDF-only 资料也未当成独立论文迁入；不能自动决定哪个抽取更好或是否将 supplemental 作为主 bundle 的附属资源。PR090 两份材料按正文 SHA-256 和主要资源数量一致判为 exact，重复实体已移到 `D:\0Project\origin\论文_duplicates_review\` 而非删除；旧 `paper-index.md` 曾记录两份独立提取字节不同，如需证明所有附属文件相同仍应人工复核，不把简单签名写成逐字节全包相同。RE042、RE053、RE188、RE447 的旧编号—题名对应未经总索引核实，统一索引只记录为 `human.unverified_manual_ids`，未绑定为已确认 `manual_ids`。已知旧归档 `D:\0Project\DFormer-archive-20260922\doc\paper` 当前不可访问，是否恢复来源再补扫需要用户决定。证据与现行位置：[`PAPER_LIBRARY_INDEX.json`](../../MMFR/03_reference/PAPER_LIBRARY_INDEX.json)、[`paper-index.md`](../../MMFR/03_reference/paper-index.md)。这些未决项不改变现有训练/评价授权。

已决定、已执行、失效和历史背景事项只在归档、协议或正式报告中追溯，不再追加到本文件。
