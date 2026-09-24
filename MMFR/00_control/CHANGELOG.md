# MMFR v4.1 package change log

## 2026-09-24 — R-OE-lite v2 单通道上采样调整与条件训练授权

- 将 `d2 → GELU → head` 移到最终全分辨率双线性插值之前；不增删可训练层，参数仍为 `3,302,785`。原 v1 逐像素函数改变，v2 使用新 protocol/run identity，严禁继承 v1 中止训练的更新状态。
- 本地仅核查 v2 config/substitute import、CPU 输入输出尺寸、单通道、掩码覆盖区严格为零及参数量；路由与 reliability auxiliary 代码未改。RTX 4090 三步及正式训练结果尚未取得。
- 用户授权：保留 batch size 10 和其余 Batch 1B 训练设置，先连续 3 次成功更新并记录显存/损失/触发数/插值输入；满足无 OOM、loss finite 和显存余量条件后，从 A2 epoch-420 source 干净开始 2560 次更新，再进入已授权四条件 Quick-Val。OOM 则立即停止。v1 报告与旧审核包保持历史身份，未重跑 Gate-B、未重建审核包。

## 2026-09-24 — E1 Batch 1B R-OE-lite formal training stopped by CUDA OOM

- Added `02_evidence/report_e1_batch1b_roe_formal_training_attempt_20260924.md` recording the cloud run identity, OOM evidence, update-count boundary, absent fixed-final checkpoint, unrun Quick-Val, and unresolved C0 checkpoint path.
- Updated the authoritative current-status and open-decisions documents: the run ended at 01:45:30 UTC after at least 168 successful updates; the exact final count was not persisted; no automatic retry is authorized by this recovery point.
- Updated `PACKAGE_INDEX.md` to point to the new attempt report. The frozen R-OE-lite protocol, Gate-B evidence, prior review packet, and official-test seal were not rewritten; no review packet rebuild was requested or performed.

## 2026-09-23 — refreshed Batch 1B L1 packet for senior review

- Rebuilt the active `e1-batch1b-roe-gateb` packet at HEAD `2c3176e375562cd62be31bb10cb9363537edb4f3`; generation ID `8213dfb94396d4c65f4200c94029308573332f210665c578ccf323577e6a0415`.
- Confirmed the six expected flat packet files, zero broken canonical/packet/copied-packet links, and `official_test=sealed_unread`.
- Refreshed reproducibility metadata for the current commit and clean-at-start worktree; no training, evaluation, cloud task, commit, or push was performed.

## 2026-09-23 — E1 Batch 1B R-OE-lite Gate-B review packet

- Recorded the completed R-OE-lite implementation and Gate-B PASS in the protocol, screening plan, design record, Gate-B report, and implementation-diff audit. The canonical Gate-B JSON SHA-256 is `35297b490c3e3eb54b5e66d3f06784688fca65038e7d09874c30c60cde820231`.
- Froze the ten-condition Main-Val promote/stop/inconclusive/blocked rules and clarified that Quick-Val is screening only, `entire_missing@1.0` is a stress condition rather than hidden-cause detection, and Gate-B does not authorize training or evaluation.
- Changed the active L1 review profile to `e1-batch1b-roe-gateb`, with the Batch 1B report and implementation-diff audit as exact-copy attachments; rebuilt the six-file packet with generation ID `86a92d119a8a1e0b2c33a2ad6fba8876aef0e2c26c1233f8fb7a28c9d2272691` and zero broken links.
- Updated the authoritative MUSeg status and open-decision pointers, package navigation, reproducibility metadata, and report index to distinguish the pending Batch 1B training authorization from the independent Batch 1A Main-Val review.
- Formal training, Quick-Val, Main-Val, cloud execution, Batch 2/T, and official-test access remain unauthorized; `official_test` remains `sealed_unread`.

## 2026-09-23 — restored into the project repository, external-reference indexes added

- Moved the package from `D:\0Project\DFormer-archive-20260922\liu-test-exp\MMFR\MMFR_v4_1_blueprint_and_reference_package_2026-09-20\` back to the project root as `D:\0Project\DFormer\MMFR\` and brought it under Git.
- Moved paper full texts and extraction artifacts out of the package and out of Git to `D:\0Project\origin\论文\`, under two provenance roots (`MMFR-附件`, `DFormer-doc-paper`). The package now keeps only Markdown, TXT, small JSON, indexes, and packet tooling.
- Repaired the package-relative links the move invalidated: `../../../../` -> `../../` in `00_control/PACKAGE_INDEX.md`, `01_research/e1_screening_plan.md`, `01_research/MMFR_research_blueprint_v4_1_2026-09-20.md`, and `03_reference/literature_fulltext_audit.md`; local-only attachment paths now read `D:\0Project\origin\论文\MMFR-附件\`; the authority paths in `review_profile.json` now resolve from the new location.
- Added `03_reference/paper-index.md` (paper number -> local full text -> local code) and `03_reference/code-index.md` (read-only local clone inventory). Neither renumbers existing PR/RE/AI/MoSA/ANGA codes.
- `PACKAGE_INDEX.md` gained a quick-navigation table and an explicit packet-state note; `FILE_RULES.md` now places paper full texts outside the repository.
- Research content, Gate-B evidence, authorization boundary, and the official-test seal were not changed. The review packet was **not** regenerated: the packet in `99_review_packet_current/` still belongs to profile `e1-batch1a-gateb`, and its recorded canonical source hashes predate these link repairs.
- Known blocker for the next packet rebuild: `01_research/e1_screening_plan.md` links to `02_evidence/report_e1_batch1b_roe_gateb.md`, which is not yet present in the package, and `98_tools/rebuild_review_packet.ps1` fails hard on a broken canonical link.
- Repaired `98_tools/rebuild_review_packet.ps1`'s `$repoRoot`, which still assumed the old three-level-deep package location; it now resolves the Git work tree one level above the package root, so the generator can read the repository identity again.

## 2026-09-21 — second directory simplification

- Consolidated the long-term package into `00_control`, `01_research`, `02_evidence`, and `03_reference`, while retaining separate archive, tooling, and generated-review areas.
- Replaced the package-local README/status pair with `PACKAGE_INDEX.md`; research facts and authorization remain authoritative only in `doc/main/MUSeg-current-status.md` and `doc/main/MUSeg-open-decisions.md`.
- Moved the active review profile into machine-readable `00_control/review_profile.json`.
- Preserved the 14 source-material files byte-for-byte while moving them to `03_reference/source_materials/`.
- Replaced pre-created empty archive categories with an event-based migration archive containing the pre-migration manifest and superseded control documents.
- Redesigned the current L1 review packet as a flat six-file interface with one human entry, one machine manifest, three exact-copy attachments, and one generated evidence summary.
- Kept the research conclusions, Gate-B decision, authorization boundary, external-code isolation, and official-test seal unchanged.

## 2026-09-21 — package directory reorganization

- Created the numbered package layout for control, blueprint, references, plans/protocols, engineering audits, experiment reports, validation/reproducibility, archives, tools, and the current review packet.
- Moved the current blueprint, reference index/registry, literature audit, E1 plans, engineering audits, Gate-B report, and validation report to their fixed directories without changing their research content.
- Preserved all 14 original `source_materials` files under `02_reference/03_source_materials/`.
- Recorded external paper-code provenance separately from external source trees; external repositories remain under `D:\0Project\origin`.
- Added package-management status, file-placement rules, current reproducibility metadata, and deterministic review-packet generation.
- Repaired package-relative links and registry source paths after the move.

## 2026-09-20 — v4.1 blueprint and provenance package

### 保持不变

保留 v4 正文0–13节的路线、候选体系、比较逻辑和历史结果。原正文全部非空行按原顺序保留；借鉴表只插入既有论文编号。未执行训练，未修改项目仓库或评价结果。

### 补充内容

- 正文对应位置加入34个实施步骤的论文/原材料/参数边界说明。
- 57条关键文献快速表；单独总索引含135条带题名文献和双向步骤映射。
- 六个子计划读文献包、13项元数据/版本/实现边界台账。
- S1–S14原材料清单与实际SHA-256；W1–W26网络来源记录。
- 保留124个旧清单中只有编号的归档项；RE067明确为提示词示例，不计为论文。
- 原附件与本轮网络补充字段分开，不用后者覆盖附件。
- 以JSON保存全量元数据、源定位、步骤关系和网络状态。

### 范围限制

这是索引与溯源修订，不是135篇论文的重新全文审计。
未验证本地模型代码、数据、checkpoint或实际模块收益。
部分出版商入口受限；明确标记，未猜造DOI、代码仓库或原文超参数。

### 文件一致性检查

校验结果见 `06_validation_and_reproducibility/validation_report.json`；全部论文/步骤/网页编号引用可解析，原附件副本哈希一致，原v4正文无非空行丢失。
