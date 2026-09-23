# MMFR v4.1 blueprint and reference package

This directory is the maintained package for the MMFR v4.1 research blueprint, source provenance, E1 plans, engineering audits, experiment reports, reproducibility records, and current senior-model review material.

## Start here

1. Read [`CURRENT_STATUS.md`](CURRENT_STATUS.md) for the package state and the current review profile.
2. Read [`FILE_PLACEMENT_RULES.md`](FILE_PLACEMENT_RULES.md) before adding or archiving a file.
3. Use [`98_tools/rebuild_review_packet.ps1`](../98_tools/rebuild_review_packet.ps1) through the package root to rebuild the minimum current review packet.
4. The research source of truth remains [`DFormer/doc/main/MUSeg-current-status.md`](../../../../doc/main/MUSeg-current-status.md); this package status file only describes package organization and evidence pointers.

## Current package scope

- Current research stage: MMFR v4.1 E1 Batch 1A C0/F-lite Gate-B passed; formal training is not authorized.
- Current review profile: `e1-batch1a-gateb`.
- External paper repositories are kept outside this package under `D:\0Project\origin`. The package stores provenance and version records, not copied external source trees.
- The 14 original `source_materials` files are retained unchanged under `02_reference/03_source_materials/`.

## Directory map

- `00_project_control/`: package README, package status, change log, and placement rules.
- `01_blueprint/`: the current v4.1 research blueprint.
- `02_reference/`: reference index/registry, literature audits, external provenance, and original source materials.
- `03_plans_and_protocols/`: current screening plans, protocols, and frozen design specifications.
- `04_engineering_audits/`: input, optimizer, implementation, and parameter-contract audits.
- `05_experiment_reports/`: Gate reports and later experiment reports.
- `06_validation_and_reproducibility/`: package validation and current reproducibility metadata.
- `90_archive/`: superseded material, kept by category and never used as the current source.
- `98_tools/`: deterministic package-management scripts.
- `99_review_packet_current/`: generated, upload-ready material for the current review profile only.

The package root intentionally contains only numbered directories. New documents must not be placed directly in the root.
