# MMFR package index

## Package identity

- Package: `MMFR_v4_1_blueprint_and_reference_package_2026-09-20`
- Research blueprint: MMFR v4.1
- In-repository root: `D:\0Project\DFormer\MMFR`（2026-09-23 从外部归档恢复到项目内并纳入 Git）
- Current blueprint: [`01_research/MMFR_research_blueprint_v4_1_2026-09-20.md`](../01_research/MMFR_research_blueprint_v4_1_2026-09-20.md)
- Active review profile source: [`review_profile.json`](review_profile.json)
- Current active profile: `e1-batch1a-gateb`
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
| 当前实验结论 | [`02_evidence/`](../02_evidence/) 的 Gate-B 审计与报告；Main-Val 分析的详细报告在包外 `D:\0Project\DFormer\doc\reports\2026-09-22-mmfr-e1-batch1a-c0-flite-mainval.md` |
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

**Packet state after the 2026-09-23 relocation（必须知道的三点）**

1. 当前目录内是最后一次生成的结果，profile 为 `e1-batch1a-gateb`，generation ID 为 `8d5657e0f1de834c1a6becc911ea6fb03922c88d88a7b70edac015a13fa0f6bf`，对应 E1 Batch 1A C0/F-lite 的 Gate-B 工程资格审核，不是最新阶段。
2. 本轮只做路径修复（包内相对链接与本地资料位置随目录迁移更新），没有重建该 packet；因此 `packet_manifest.json` 里记录的部分 canonical 源文件 SHA-256 早于本轮修复，**下一次真正提交给上级审核前必须重新生成**。
3. 重新生成当前被一个已知缺件阻塞：`01_research/e1_screening_plan.md` 链接到 `02_evidence/report_e1_batch1b_roe_gateb.md`，该报告尚未写入本包，而 `98_tools/rebuild_review_packet.ps1` 遇到坏链会直接失败。补齐该报告（或把该链接改成纯文本指针）后才能重建。

## External code boundary

External paper repositories remain exclusively under `D:\0Project\origin`. This package stores provenance and boundaries, not copied repositories, external checkpoints, or vendored source trees. The read-only local inventory of those clones is `03_reference/code-index.md`.

## Local paper full texts（仓库外，不纳入 Git）

MMFR uses two evidence layers:

- Paper full texts and extraction artifacts (Markdown conversions, extracted figures/tables, supplementary originals) live outside the Git repository, under `D:\0Project\origin\论文\`. They are read-only reference material and are intentionally not committed.
- Portable conclusions, source identities, provenance, and review evidence remain tracked in this package, primarily under `03_reference/`.

The local files remain readable by tools and AI when they exist on the current machine. Canonical package documents must show local-only paths as plain text rather than portable Markdown links, so a clean Git checkout does not claim that those files are included. `03_reference/paper-index.md` is the mapping from MMFR paper numbers to those local paths.

## Source-material integrity

The 14 original source-material files are retained under `03_reference/source_materials/`. Their content, byte sizes, and registry SHA-256 identities must remain unchanged across directory migrations.
