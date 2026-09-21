# MMFR package index

## Package identity

- Package: `MMFR_v4_1_blueprint_and_reference_package_2026-09-20`
- Research blueprint: MMFR v4.1
- Current blueprint: [`01_research/MMFR_research_blueprint_v4_1_2026-09-20.md`](../01_research/MMFR_research_blueprint_v4_1_2026-09-20.md)
- Active review profile source: [`review_profile.json`](review_profile.json)
- Current active profile: `e1-batch1a-gateb`
- Review level: `L1`

## Authoritative research state

Research facts, authorization boundaries, open decisions, and recovery points remain authoritative only in:

- [`doc/main/MUSeg-current-status.md`](../../../../doc/main/MUSeg-current-status.md)
- [`doc/main/MUSeg-open-decisions.md`](../../../../doc/main/MUSeg-open-decisions.md)

This package is a maintained evidence library and review-packet source. It is not a second research-status authority.

## Directory map

- `00_control/`: package navigation, placement rules, change history, and the machine-readable review profile.
- `01_research/`: current blueprint, screening plan, protocol, and frozen design specifications.
- `02_evidence/`: engineering audits, experiment reports, validation, and reproducibility metadata.
- `03_reference/`: reference index/registry, literature audit, external-code provenance, and the 14 preserved source materials.
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

## External code boundary

External paper repositories remain exclusively under `D:\0Project\origin`. This package stores provenance and boundaries, not copied repositories, external checkpoints, or vendored source trees.

## Local full-text attachments

MMFR uses two evidence layers:

- Local-only full texts and extraction artifacts remain under `liu-test-exp/MMFR/附件/`. This directory is excluded by Git and must be restored separately on a new machine.
- Portable conclusions, source identities, provenance, and review evidence remain tracked in this package, primarily under `03_reference/`.

The local files remain readable by tools and AI when they exist on the current machine. Canonical package documents must show local-only attachment paths as plain text rather than portable Markdown links, so a clean Git checkout does not claim that those files are included.

## Source-material integrity

The 14 original source-material files are retained under `03_reference/source_materials/`. Their content, byte sizes, and registry SHA-256 identities must remain unchanged across directory migrations.
