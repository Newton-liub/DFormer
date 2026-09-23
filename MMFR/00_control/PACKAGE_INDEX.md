# MMFR package index

## Package identity

- Package: `MMFR_v4_1_blueprint_and_reference_package_2026-09-20`
- Research blueprint: MMFR v4.1
- In-repository root: `D:\0Project\DFormer\MMFR`（2026-09-23 从外部归档恢复到项目内并纳入 Git）
- Current blueprint: [`01_research/MMFR_research_blueprint_v4_1_2026-09-20.md`](../01_research/MMFR_research_blueprint_v4_1_2026-09-20.md)
- Active review profile source: [`review_profile.json`](review_profile.json)
- Current active profile: `e1-batch1b-roe-gateb`
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
| 当前实验结论 | Batch 1B R-OE-lite Gate-B 报告与实现差异审计见 [`02_evidence/`](../02_evidence/)；Batch 1A Main-Val 的描述性分析在 `doc/reports/2026-09-22-mmfr-e1-batch1a-c0-flite-mainval.md` |
| 下一阶段计划 | [`01_research/e1_screening_plan.md`](../01_research/e1_screening_plan.md)、[`01_research/e1_batch1_protocol.md`](../01_research/e1_batch1_protocol.md)、[`01_research/r_oe_lite_design.md`](../01_research/r_oe_lite_design.md) |
| 相关论文在哪里 | [`03_reference/paper-index.md`](../03_reference/paper-index.md)；本地全文在仓库外 `D:\0Project\origin\论文\` |
| 相关代码在哪里 | [`03_reference/code-index.md`](../03_reference/code-index.md)；本地 clone 在仓库外 `D:\0Project\origin\` |
| 当前需要上级审核什么 | [`99_review_packet_current/REVIEW_BRIEF.md`](../99_review_packet_current/REVIEW_BRIEF.md)；实时阶段、授权边界与“待审核事项”始终以实时入口为准 |

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

**Packet state for the 2026-09-23 Batch 1B review**

1. The active L1 profile is `e1-batch1b-roe-gateb`; it points to the Batch 1B protocol, Gate-B report, and implementation-diff audit.
2. The current packet was rebuilt successfully with generation ID `8213dfb94396d4c65f4200c94029308573332f210665c578ccf323577e6a0415`. It contains the six expected files, and the generator reported zero broken canonical, packet, or copied-packet links.
3. `99_review_packet_current/` is generated output and is the sole upload-ready entry. Its manifest records source and packet hashes; do not edit generated files by hand. `official_test` remains `sealed_unread`.


## External code boundary

External paper repositories remain exclusively under `D:\0Project\origin`. This package stores provenance and boundaries, not copied repositories, external checkpoints, or vendored source trees. The read-only local inventory of those clones is `03_reference/code-index.md`.

## Local paper full texts（仓库外，不纳入 Git）

MMFR uses two evidence layers:

- Paper full texts and extraction artifacts (Markdown conversions, extracted figures/tables, supplementary originals) live outside the Git repository, under `D:\0Project\origin\论文\`. They are read-only reference material and are intentionally not committed.
- Portable conclusions, source identities, provenance, and review evidence remain tracked in this package, primarily under `03_reference/`.

The local files remain readable by tools and AI when they exist on the current machine. Canonical package documents must show local-only paths as plain text rather than portable Markdown links, so a clean Git checkout does not claim that those files are included. `03_reference/paper-index.md` is the mapping from MMFR paper numbers to those local paths.

## Source-material integrity

The 14 original source-material files are retained under `03_reference/source_materials/`. Their content, byte sizes, and registry SHA-256 identities must remain unchanged across directory migrations.
