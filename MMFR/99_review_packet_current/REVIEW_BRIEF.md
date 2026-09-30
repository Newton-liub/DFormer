# MMFR review brief

## Review identity

- Profile: `mmfr-a-v1-action-utility-gateb`
- Review level: `L1`
- Generation identity: `52fdaa881070430927f50d05a14e19171dd95c11866a2033b7dde314436c3e22`
- Official test: `sealed_unread`

## Authoritative research state

- Research status: `doc/main/MUSeg-current-status.md`
- Open decisions: `doc/main/MUSeg-open-decisions.md`

This generated packet is not a research-authorization authority. The two repository documents above remain authoritative.

## Current task

MMFR A-v1 frozen formal contract and local cloud-preflight handoff

Review the frozen first-round contract and exact local Git handoff. The next possible cloud action is only full-resolution RTX 4090 Proposal 3-update then Gate 3-update preflight followed by a mandatory stop; formal training remains unauthorized.

## Current state and boundary

- A-v1 implementation complete; original Gate-B PASS; identity MMFR-A-v1-action-utility-v1 unchanged; first-round formal contract frozen.
- Source C0 SHA-256 rechecked: ca618b23d18eabb201a0d11d18da383ac99576d0feae5864e3233bda527d9a1a; source checkpoint bytes unchanged.
- The four runtime/code files remain byte-identical to Gate-B; only formal config and documents changed in this closeout. Config import and bounded field assertions pass without model/data/GPU execution.
- Proposal/Gate successful updates 1920/640; total 2560; margin 0.01 per-image mean CE difference; lambda_clean 0.1 in both phases; AdamW new LR 3e-5, WD 0.01 and existing bias/norm no-decay.
- Target batch10, 480x640, workers8, accumulation1, AMP fp16 on and explicit matmul/cuDNN TF32 on; phase corruption seeds 2026093001/2026093002, p_clean 0.25. Full-resolution 4090 feasibility is not yet verified.
- Proposal transition at 1920, fixed-final at global update2560, recovery every640 successful updates. No validation selector, utility early-stop, budget extension or val-dev parameter tuning.
- One future four-condition off/full/learned Quick-Val: learned hard mean vs matched off >=+0.50pp, clean >=-0.20pp, learned hard mean strictly greater than full. This is a screening rule, not statistical significance or evaluation authorization.
- Cloud preflight contract caps each phase at three successful updates; Proposal must pass before Gate; OOM/NaN/Inf/scaler skip/identity error stops immediately; no automatic continuation to formal training.
- Only phase interfaces exist; full-resolution preflight/formal runner and A-v1 evaluation entry are not implemented. This round performs no cloud access, training, Val, official test or push. Commit receipt is filled after the single local commit.

## Please review

1. Does the source and original Gate-B binding support unchanged A-v1 structure/forward/loss/RNG with config-only formal contract extension?
2. Are phase-local warmup/poly, corruption coordinates, phase seeds and full-state recovery unambiguous and independent of the old E1 identity?
3. Are all frozen Quick-Val conditions and the three numerical continuation requirements recorded without claiming significance?
4. Is the RTX 4090 preflight capped at three successful updates per phase with immediate engineering-error stop and no automatic formal continuation?
5. Does the exact local commit exclude unrelated dirty files, and are post-commit SHA receipts clearly distinguished from executable changes?

## Do not infer or authorize

- Segmentation benefit, trained selector ability or publication novelty
- Full-resolution batch10 memory feasibility or stable training throughput
- Formal Proposal/Gate training, Quick-Val/Main-Val, optimizer searches or budget extensions
- Cloud login/sync/execute in this local-only round, official-test access, remote push, or extra structures/failure types/seeds

## Recommended reading

1. [CURRENT_PROTOCOL.md](CURRENT_PROTOCOL.md)
2. [CURRENT_REPORT.md](CURRENT_REPORT.md)
3. [IMPLEMENTATION_DIFF.md](IMPLEMENTATION_DIFF.md)
4. [EVIDENCE_SUMMARY.md](EVIDENCE_SUMMARY.md)

## Changed since previous review

- The previously undecided margin/clean weight/budget/continuation line are now explicitly frozen by the user, not selected using val-dev.
- Runtime implementation and original Gate-B evidence are unchanged; formal config records a new hash separately.
- A-v1 local Git preparation is authorized; cloud operations and formal training remain outside this round.
- Old R-OE/F-lite results and dispositions remain independent historical evidence/open decisions.

## Background not included

- Full literature library
- Full old E1 experiment history
- Raw checkpoints/datasets/logs
- Archives

The excluded material remains in the canonical package and may be added only by a profile that explicitly requires it.
