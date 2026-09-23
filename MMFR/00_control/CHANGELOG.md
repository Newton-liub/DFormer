# MMFR v4.1 package change log

## 2026-09-23 — restored into the project repository, external-reference indexes added

- Moved the package from `D:\0Project\DFormer-archive-20260922\liu-test-exp\MMFR\MMFR_v4_1_blueprint_and_reference_package_2026-09-20\` back to the project root as `D:\0Project\DFormer\MMFR\` and brought it under Git.
- Moved paper full texts and extraction artifacts out of the package and out of Git to `D:\0Project\origin\论文\`, under two provenance roots (`MMFR-附件`, `DFormer-doc-paper`). The package now keeps only Markdown, TXT, small JSON, indexes, and packet tooling.
- Repaired the package-relative links the move invalidated: `../../../../` -> `../../` in `00_control/PACKAGE_INDEX.md`, `01_research/e1_screening_plan.md`, `01_research/MMFR_research_blueprint_v4_1_2026-09-20.md`, and `03_reference/literature_fulltext_audit.md`; local-only attachment paths now read `D:\0Project\origin\论文\MMFR-附件\`; the authority paths in `review_profile.json` now resolve from the new location.
- Added `03_reference/paper-index.md` (paper number -> local full text -> local code) and `03_reference/code-index.md` (read-only local clone inventory). Neither renumbers existing PR/RE/AI/MoSA/ANGA codes.
- `PACKAGE_INDEX.md` gained a quick-navigation table and an explicit packet-state note; `FILE_RULES.md` now places paper full texts outside the repository.
- Research content, Gate-B evidence, authorization boundary, and the official-test seal were not changed. The review packet was **not** regenerated: the packet in `99_review_packet_current/` still belongs to profile `e1-batch1a-gateb`, and its recorded canonical source hashes predate these link repairs.
- Known blocker for the next packet rebuild: `01_research/e1_screening_plan.md` links to `02_evidence/report_e1_batch1b_roe_gateb.md`, which is not yet present in the package, and `98_tools/rebuild_review_packet.ps1` fails hard on a broken canonical link.

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
