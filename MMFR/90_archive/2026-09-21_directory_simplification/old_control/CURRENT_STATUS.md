# MMFR v4.1 package current status

> Status date: 2026-09-21
>
> This file is the package-management status, not a replacement for the MUSeg research status. Research facts and authorization boundaries remain authoritative in [`doc/main/MUSeg-current-status.md`](../../../../doc/main/MUSeg-current-status.md).

## Current package state

- The package has been reorganized into numbered directories.
- The current research material is MMFR v4.1 E1 Batch 1A: C0 and F-lite passed Gate-B; formal 20-epoch/2,560-update training is not authorized.
- R-EM-lite is retired by observability. R-OE-lite is design-frozen but not implemented or authorized.
- Active review profile (machine-readable): `e1-batch1a-gateb`.
- The 14 original source-material files are preserved under `02_reference/03_source_materials/`.
- No external paper repository is copied into this package. External repositories remain under `D:\0Project\origin`; package provenance is recorded in `02_reference/02_literature_audits/external_reference_provenance_2026-09-21.md`.

## Evidence entry points

| Purpose | Canonical path |
| --- | --- |
| Research facts and authorization | `../../../../doc/main/MUSeg-current-status.md` |
| Research choices and open decisions | `../../../../doc/main/MUSeg-open-decisions.md` |
| Current E1 screening plan | `../03_plans_and_protocols/e1_screening_plan.md` |
| Current E1 Batch 1 protocol | `../03_plans_and_protocols/e1_batch1_protocol.md` |
| R-OE-lite frozen design | `../03_plans_and_protocols/r_oe_lite_design.md` |
| Engineering audits | `../04_engineering_audits/` |
| Gate-B report | `../05_experiment_reports/e1_batch1a_gateb_report.md` |
| Package validation | `../06_validation_and_reproducibility/validation_report.json` |
| Current reproducibility metadata | `../06_validation_and_reproducibility/CURRENT_REPRODUCIBILITY.json` |
| Generated review packet | `../99_review_packet_current/` |

## Review packet rule

Run `98_tools/rebuild_review_packet.ps1` from this package root when the current review packet is needed. With the default `current` profile, the script reads the machine-readable active profile from `00_project_control/CURRENT_STATUS.md`, resolves that profile's selection, and writes a manifest with SHA-256 values. It excludes original source-material copies, the full reference index/registry, archives, external repositories, checkpoints, raw logs, and official-test material unless a future profile explicitly changes the selection.

## Recovery point

After a future research-stage change, update the authoritative MUSeg status first, then update the package status and review profile, and finally rebuild `99_review_packet_current/`. Until that sequence is completed, the packet must not be presented as current.
