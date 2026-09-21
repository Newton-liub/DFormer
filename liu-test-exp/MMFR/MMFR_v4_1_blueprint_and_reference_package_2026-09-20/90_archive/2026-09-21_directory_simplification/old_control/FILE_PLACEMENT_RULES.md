# MMFR package file-placement and version rules

## 1. Root rule

The package root keeps numbered directories only. New Markdown, JSON, TXT, experiment reports, audit reports, logs, or manifests must be placed in a numbered directory. The root must not become an informal inbox.

## 2. Fixed placement

| File purpose | Fixed directory |
| --- | --- |
| README, package status, change log, placement rules | `00_project_control/` |
| Current effective research blueprint | `01_blueprint/` |
| Reference index, registry, paper-number tables | `02_reference/01_index_and_registry/` |
| Literature full-text audits, provenance, PR/RE collision analysis | `02_reference/02_literature_audits/` |
| Original source files and supplied reference material | `02_reference/03_source_materials/` |
| Screening plans, experiment plans, protocols, frozen design specifications | `03_plans_and_protocols/` |
| Input, optimizer, parameter, implementation-difference, and engineering checks | `04_engineering_audits/` |
| Gate reports, training results, Quick-Val, Main-Val, and ablation reports | `05_experiment_reports/` |
| Validation, Git identity, checkpoint identity, manifests, environment, and reproducibility metadata | `06_validation_and_reproducibility/` |
| Superseded current files | `90_archive/<category>/` |
| Package and review-packet automation | `98_tools/` |
| Generated material for the current senior-model review | `99_review_packet_current/` |

## 3. Current-versus-history rule

- One file has one current canonical location.
- A revised protocol, plan, or report receives a new identity when its scientific meaning changes; the previous file is copied to the matching `90_archive/` category and is not overwritten.
- Historical evidence is never rewritten to make it match a newer interpretation. Add a new report or an explicit status note instead.
- Archive paths should preserve the former category and include the archival date or superseding identity when a file is moved.
- `99_review_packet_current/` is generated output, not an authoring location. Edit the source files, then rebuild the packet.

## 4. External-code isolation

External paper code is stored only under `D:\0Project\origin`. The MMFR package may contain a provenance ledger with repository URL, branch, commit, license, source anchors, intended use, and boundaries. It must not contain a copied external repository, vendored source tree, external checkpoint, or mixed implementation directory.

A mechanism is not considered imported merely because a paper or repository is listed. Any later port must record the original symbol, license/attribution, MMFR destination, exact changes, experiment identity, and authorization state before code modification.

## 5. Review-packet selection

The default packet is built by `98_tools/rebuild_review_packet.ps1` using the current profile in `00_project_control/CURRENT_STATUS.md`. The default E1 Batch 1A packet contains only the current status, relevant literature/provenance audits, current E1 plans/design, engineering audits, Gate-B report, and reproducibility metadata.

The default packet excludes:

- the 14 original source-material copies;
- the complete reference index and registry unless a reference-audit profile explicitly requests them;
- archived files;
- external repositories under `D:\0Project\origin`;
- checkpoints, datasets, raw telemetry, raw logs, and generated caches;
- official-test data or any sealed material.

This makes packet construction deterministic. A reviewer receives the same sufficient evidence set for the same profile and does not need to guess which files to upload.

## 6. Link and identity rule

Use package-relative Markdown links for package files. Use repository-relative paths or stable absolute evidence pointers for files outside this package. When a file moves, repair its links in the same change and update the package status if an authoritative evidence pointer changed.

## 7. Validation rule

After a reorganization, verify the numbered directory layout, source-material count, JSON parseability, package-relative links, SHA-256 identities, and a dry-run or real rebuild of the current review packet. This is a focused package check; it is not a project test suite or a research experiment.
