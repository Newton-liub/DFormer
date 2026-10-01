# MUSeg 当前开放问题

> **事实截至：** 2026-10-01 07:30 UTC。原连续训练/唯一Quick-Val及提交push授权保持；单次受控恢复诊断已经执行结束，下一数值对照/精度合同及正式恢复仍需裁决。只记录仍未裁决事项，参数/预算/筛选线不重复开放。执行边界见 [`MUSeg-current-status.md`](MUSeg-current-status.md)；既往完整决策史见 [`整理前归档`](../../MMFR/90_archive/2026-09-24_live_docs_rolling_window/README.md)。

## 0. A-v1 合同变更或筛选后扩展（仅触发时需人工决定）

预检、1920+640正式训练、唯一四条件Quick-Val及必要薄入口实现已于2026-10-01授权，移出未决清单。准确执行边界见实时状态和 [protocol §7](../../MMFR/01_research/mmfr_a_v1_action_utility_protocol.md)。**大白话：** 正常路径可以连续执行，不再逐步等批准；以下超出冻结合同的情况才需人工裁决。

- 若batch10显存不足，或修复必须改变结构/loss/种子/预算/数据或评价口径，停止并决定是否修改合同；不自动降batch、accumulation改约或sweep。中断恢复兼容性无法确认时同样先问人工。
- 四条件筛选结束后，是否追加Main-Val或其他研究轮次；无论promote-for-next-review与否，本次均不自动授权追加实验或official test。

原文献创新性与全文缺口仍为论证局限，不以此重新搜索或重设计。以下旧路线问题保持独立。

## 0.1 A-v1 失败batch数值对照、精度合同与正式恢复（当前阻塞）

单次受控恢复诊断已完成：590条成功记录精确匹配，在原1871坐标定位到HAM/NMF矩阵乘反向NaN。其执行选择已关闭；证据/资源/恢复点只看实时状态。

- 是否授权利用已保存的同一失败batch/模型/pre-forward RNG做最多3组定点loss/backward对照：原AMP+scale1024、原AMP+scale1、仅HAM/NMF局部FP32+scale1024。每组沿用off/full配对，最多6次模型forward+3次backward，无optimizer更新，不重新跑590步，不进入Gate/评价。
- 若对照支持精度修复，是否批准调整正式数值合同，以及从完整1280恢复还是另立干净正式起点；正式跨提交兼容性必须显式审定，原身份不可改写，不静默weights-only续训。
- 也可以停止本轮并保留原证据。**大白话：** 错误位置已经找到，但降低scale或局部FP32能否修复尚未测过；本次授权已用完，不能自动改精度继续出最终成绩。

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
