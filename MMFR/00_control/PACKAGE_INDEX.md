# MMFR package index

## Package identity

- Package: `MMFR_v4_1_blueprint_and_reference_package_2026-09-20`
- Research blueprint: MMFR v4.1
- In-repository root: `D:\0Project\DFormer\MMFR`（2026-09-23 从外部归档恢复到项目内并纳入 Git）
- Current blueprint: [`01_research/MMFR_research_blueprint_v4_1_2026-09-20.md`](../01_research/MMFR_research_blueprint_v4_1_2026-09-20.md)（历史冻结设计保持不变）
- Latest design and implementation: [`A-v1 任务效用控制残差设计`](../../doc/reports/2026-09-30-mmfr-next-generation-research-design.md)；[`最小实现/Gate-B 审核报告`](../02_evidence/report_mmfr_a_v1_implementation_gateb_20260930.md)（2026-09-30；implementation complete、Gate-B PASS；正式合同已冻结，2026-10-01已授权云端两阶段各3-update预检，通过后连续正式训练和唯一四条件Quick-Val，尚未执行）
- Pre-revision snapshot: [`pre-A-v1 初版报告`](../90_archive/2026-09-30_mmfr_a_v1_design_revision/2026-09-30-mmfr-next-generation-research-design.pre-a-v1.md)（字节一致归档；初版 0.0.15 展示仅对应此历史设计）
- Active review profile source: [`review_profile.json`](review_profile.json)
- Current active profile: `mmfr-a-v1-action-utility-gateb`（独立 A-v1 工程审核；不代表训练/评价授权）
- Review level: `L1`

## Authoritative research state

Research facts, authorization boundaries, open decisions, and recovery points remain authoritative only in:

- [`doc/main/MUSeg-current-status.md`](../../doc/main/MUSeg-current-status.md)
- [`doc/main/MUSeg-open-decisions.md`](../../doc/main/MUSeg-open-decisions.md)

This package is a maintained evidence library and review-packet source. It is not a second research-status authority.

## Quick navigation（新会话快速入口）

| 需要知道 | 去哪里看 |
| --- | --- |
| 当前 MMFR 做到哪里 | [`doc/main/MUSeg-current-status.md`](../../doc/main/MUSeg-current-status.md)（唯一实时入口）；本包目录与变更历史见本文件与 [`CHANGELOG.md`](CHANGELOG.md) |
| 最新研究蓝图 | [`01_research/MMFR_research_blueprint_v4_1_2026-09-20.md`](../01_research/MMFR_research_blueprint_v4_1_2026-09-20.md) |
| 当前工程与实验事实 | A-v1 implementation complete、Gate-B PASS（仅资格，无新性能实验），见 [`A-v1 审核报告`](../02_evidence/report_mmfr_a_v1_implementation_gateb_20260930.md)；历史 R-OE-lite v2 正式训练已完成 2560/2560 次更新并生成 fixed-final checkpoint，四条件 Quick-Val（单视图 screening）相对 C0 为 `0.00 / +0.01 / +0.01 / +0.01` pp、判定 `inconclusive`，证据见 [`02_evidence/report_e1_batch1b_roe_v2_formal_training_quickval_20260924.md`](../02_evidence/report_e1_batch1b_roe_v2_formal_training_quickval_20260924.md)；执行边界见 [`doc/main/MUSeg-current-status.md`](../../doc/main/MUSeg-current-status.md)；v1 正式训练 CUDA OOM 历史报告见 [`02_evidence/report_e1_batch1b_roe_formal_training_attempt_20260924.md`](../02_evidence/report_e1_batch1b_roe_formal_training_attempt_20260924.md)；v1 Gate-B PASS 历史证据见 [`02_evidence/report_e1_batch1b_roe_gateb.md`](../02_evidence/report_e1_batch1b_roe_gateb.md)；Batch 1A Main-Val 描述性分析在 `doc/reports/2026-09-22-mmfr-e1-batch1a-c0-flite-mainval.md` |
| 下一阶段计划 | A-v1 采用独立 [`mmfr_a_v1_action_utility_protocol.md`](../01_research/mmfr_a_v1_action_utility_protocol.md)，正式合同冻结，2026-10-01授权的下一步为必要入口实现、Proposal/Gate各3-update全尺寸预检、通过后正式1920+640及一次四条件Quick-Val；完成后停止，不自动Main-Val；旧 E1 [`e1_screening_plan.md`](../01_research/e1_screening_plan.md)、[`e1_batch1_protocol.md`](../01_research/e1_batch1_protocol.md)、[`r_oe_lite_design.md`](../01_research/r_oe_lite_design.md)只作对应旧身份追溯 |
| 相关论文在哪里 | 统一实体见 [`03_reference/PAPER_LIBRARY_INDEX.md`](../03_reference/PAPER_LIBRARY_INDEX.md)（机器索引 `03_reference/PAPER_LIBRARY_INDEX.json`）；既有 PR/RE/AI 编号与本地全文映射见 [`03_reference/paper-index.md`](../03_reference/paper-index.md)。实体在仓库外 `D:\0Project\origin\论文\`，新 bundle 由 `human/paper_library.cmd` 接入 |
| 相关代码在哪里 | [`03_reference/code-index.md`](../03_reference/code-index.md)；本地 clone 在仓库外 `D:\0Project\origin\` |
| 当前需要上级审核什么 | [`A-v1 实现/Gate-B 报告`](../02_evidence/report_mmfr_a_v1_implementation_gateb_20260930.md)与独立 protocol；[`99_review_packet_current/REVIEW_BRIEF.md`](../99_review_packet_current/REVIEW_BRIEF.md)由当前 A-v1 profile 生成，不赋予训练或评价授权；实时阶段与边界以实时入口为准 |

完整文献编号表（135 条）仍是 [`03_reference/MMFR_reference_index_v4_1_2026-09-20.md`](../03_reference/MMFR_reference_index_v4_1_2026-09-20.md) 与 [`03_reference/MMFR_reference_registry_v4_1_2026-09-20.json`](../03_reference/MMFR_reference_registry_v4_1_2026-09-20.json)；`paper-index.md` 只补“本地全文/代码实际在哪”这一层，不重编号、不替代编号总索引。

## Directory map

- `00_control/`: package navigation, placement rules, change history, and the machine-readable review profile.
- `01_research/`: current blueprint, screening plan, protocol, and frozen design specifications.
- `02_evidence/`: engineering audits, experiment reports, validation, and reproducibility metadata.
- `03_reference/`: reference index/registry, local paper index, local code index, literature audit, external-code provenance, and the 14 preserved source materials.
- `90_archive/`: event-based historical records created only when material is actually superseded or a migration record is required.
- `98_tools/`: deterministic package and review-packet automation.
- `99_review_packet_current/`: generated, flat, upload-ready material for the active review profile.

## Review packet

From the package root, run:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\98_tools\rebuild_review_packet.ps1
```

The script reads `00_control/review_profile.json`, generates a flat packet, records source and packet SHA-256 identities, and enforces the packet exclusion rules. Upload the complete `99_review_packet_current/` directory for the current senior-model review.

`99_review_packet_current/` is generated output. Do not edit files there; edit canonical sources and rebuild.

**Packet state for the 2026-09-30 A-v1 review**

1. The active L1 profile is `mmfr-a-v1-action-utility-gateb`; it points to the independent A-v1 protocol, Gate-B report and implementation audit. Prior E1 reports/protocols remain unchanged.
2. Rebuild the current frozen-contract handoff packet with `98_tools/rebuild_review_packet.ps1` after canonical link checks; the generation ID and Git identity are recorded in the existing `packet_manifest.json`, not duplicated as a stale value here.
3. The six-file generated `99_review_packet_current/` is the sole upload-ready entry. It includes Gate-B evidence and the frozen formal/preflight contract; it does not authorize formal training/evaluation or cloud operations. `official_test` remains `sealed_unread`.


## External code boundary

External paper repositories remain exclusively under `D:\0Project\origin`. This package stores provenance and boundaries, not copied repositories, external checkpoints, or vendored source trees. The read-only local inventory of those clones is `03_reference/code-index.md`.

## Local paper full texts（仓库外，不纳入 Git）

MMFR uses two evidence layers:

- Paper full texts and extraction artifacts (Markdown conversions, extracted figures/tables, supplementary originals) live outside the Git repository, under `D:\0Project\origin\论文\`. They are read-only reference material and are intentionally not committed.
- Portable conclusions, source identities, provenance, and review evidence remain tracked in this package, primarily under `03_reference/`.

The local files remain readable by tools and AI when they exist on the current machine. Canonical package documents must show local-only paths as plain text rather than portable Markdown links, so a clean Git checkout does not claim that those files are included. `03_reference/PAPER_LIBRARY_INDEX.json` is the machine-readable current bundle directory, with `03_reference/PAPER_LIBRARY_INDEX.md` for browsing; `03_reference/paper-index.md` preserves the established PR/RE/AI research numbering and pre-normalization location evidence. New paper bundles enter through `human/paper_library.cmd`, whose refresh may update only automatic metadata and must preserve human fields.

## Source-material integrity

The 14 original source-material files are retained under `03_reference/source_materials/`. Their content, byte sizes, and registry SHA-256 identities must remain unchanged across directory migrations.
