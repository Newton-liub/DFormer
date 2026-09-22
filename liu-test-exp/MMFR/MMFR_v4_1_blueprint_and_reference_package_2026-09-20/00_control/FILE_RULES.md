# MMFR package file and version rules

## 1. Package root

The package root keeps only these numbered directories:

- `00_control/`
- `01_research/`
- `02_evidence/`
- `03_reference/`
- `90_archive/`
- `98_tools/`
- `99_review_packet_current/`

New Markdown, JSON, TXT, audit, report, manifest, or log files must not be placed directly in the package root.

## 2. Canonical placement

- Package index, file rules, change history, and review profile: `00_control/`.
- Current blueprint, plans, protocols, and frozen research designs: `01_research/`.
- Engineering audits, experiment reports, validation, and reproducibility metadata: `02_evidence/`.
- Reference index/registry, literature audits, provenance, and original source materials: `03_reference/`.
- Local paper full texts and extraction artifacts: repository-local `liu-test-exp/MMFR/附件/`, excluded from Git and not part of this package.
- Superseded files and migration evidence: event-based subdirectories under `90_archive/`.
- Automation: `98_tools/`.
- Current generated senior-model packet: `99_review_packet_current/`.

Use filename prefixes in `02_evidence/` to preserve evidence type:

- `audit_...`
- `report_...`
- `validation_...`
- `reproducibility_...`

## 3. Current and historical material

- One canonical file has one current location.
- When scientific meaning changes, create a new identity and move the superseded file into a dated, reason-named archive event directory.
- Historical evidence must not be rewritten to match a newer interpretation.
- Do not pre-create empty archive categories. Create an archive event directory only when it contains actual historical or migration evidence.
- `99_review_packet_current/` is never a canonical authoring location.

## 4. Research authority

The only live research-state authorities are:

- `doc/main/MUSeg-current-status.md`
- `doc/main/MUSeg-open-decisions.md`

`PACKAGE_INDEX.md` may point to them but must not become a second research authorization source.

## 5. Review profile and packet

- The active profile is machine-readable in `00_control/review_profile.json`.
- The user must not manually select packet files.
- Each profile determines the review level, exact-copy attachments, generated summaries, review questions, and exclusions.
- L1 is the default engineering/experiment review level and should normally remain within five or six packet files.
- L2 is reserved for major route, blueprint, or literature review and may add only profile-declared attachments.
- `REVIEW_BRIEF.md` is the packet's single human entry point.
- `packet_manifest.json` is the packet's single machine identity entry point.
- Generated summaries must record all canonical inputs and their SHA-256 values.
- Exact-copy attachments record identical source and packet hashes.
- A transformed packet file records `source_sha256`, `packet_sha256`, and the named transform.

## 6. Default exclusions

Every normal review profile excludes:

- `03_reference/source_materials/`;
- the full reference index and registry unless an L2 profile explicitly requires them;
- `90_archive/`;
- external repositories under `D:\0Project\origin`;
- local paper full texts and extraction artifacts under `liu-test-exp/MMFR/附件/`;
- datasets, checkpoints, raw logs, telemetry, caches, and generated training artifacts;
- official-test data, predictions, reports, or sealed artifacts.

An unauthorized official-test reference is a hard generation failure, not a warning.

## 7. External-code isolation

External paper code is stored only under `D:\0Project\origin`. The package may record repository URL, local path, branch, commit, license, source anchors, intended use, direct-copy status, and implementation boundaries. It must not contain an external repository, vendored source tree, or external checkpoint.

## 8. Source materials

The 14 files under `03_reference/source_materials/` are immutable source evidence:

- do not rename them;
- do not rewrite or format them;
- do not change encoding or line endings;
- verify byte size and SHA-256 against `03_reference/MMFR_reference_registry_v4_1_2026-09-20.json` after any move.

## 9. Links and paths

- Canonical Markdown uses links relative to the canonical file location.
- Files outside the package use repository-relative evidence pointers or stable absolute paths where necessary.
- Git-excluded local attachments are written as plain-text paths and explicitly marked local-only; canonical Markdown must not present them as portable links.
- Generated packet files must not contain broken local links.
- If a canonical target is not included in the packet, generated text must show the canonical path as plain text instead of leaving a broken link.
- Path-only repairs may update authoritative status documents only when a package evidence path actually moved; research facts and authorization text must remain unchanged.

## 10. Migration and validation

Before a structural migration, create a manifest containing each canonical path, byte size, and SHA-256. After migration, minimally verify:

- the expected seven root directories;
- 14 source materials with unchanged bytes and hashes;
- JSON parseability;
- zero broken canonical Markdown links;
- a six-file packet for the current L1 profile;
- zero broken packet links, including after copying the packet to another directory;
- deterministic repeated packet builds;
- exclusion of nested repositories, datasets, checkpoints, logs, caches, telemetry, and official-test material;
- `git diff --check` exit code `0`.
