# E1 Batch 1B R-OE-lite implementation diff audit

- Scope: the local, uncommitted E1 Batch 1B R-OE-lite implementation, as checked by the canonical Gate-B artifact on 2026-09-23.
- Purpose: summarize the implementation boundary for the L1 review packet. This file is not a substitute for the source diff or the Gate-B JSON.
- Result: the R-OE-lite implementation path passed its minimum Gate-B; no formal training or model-effect evaluation was performed.

## Changed implementation surface

- `models/roe_substitute.py` adds the frozen RGB-only observable-empty substitute network. It has no skip connection or normalization layer, uses the frozen channel sequence `[3,122,398,256,256,398,122,1]`, and contains exactly `3,302,785` trainable parameters.
- `models/builder.py` adds per-sample observable-empty routing. The detector reads the current raw Depth and geometry-valid support; triggered samples route through the substitute, while non-triggered samples preserve exact Depth bypass. Mixed batches are routed per sample.
- `utils/train.py` connects the R-OE candidate to the shared E1 training path. The substitute output is used for segmentation on triggered samples; the reliability auxiliary path retains the original corrupted raw inputs and target.
- `local_configs/MUSeg/DFormerv2_S_MMFR_E1_Batch1B_R_OE.py` identifies the R-OE candidate and keeps the shared Batch 1A C0 training fields equal to the matched control.
- `tools/mmfr/e1_batch1b_gateb.py` implements the bounded Gate-B evidence checks; its canonical output is `outputs/mmfr-e1-batch1b-gateb/e1-batch1b-gateb.json`.

## Contract evidence recorded by Gate-B

- Shared config comparison: `exact_equal=true`, with no mismatches.
- Post-build CPU/CUDA RNG state and the 1280-item first-epoch sample permutation match C0 exactly.
- Mixed-batch route: `[False, True, False, False]`; the substitute executes once. Nonempty and no-geometry cases bypass; non-triggered samples receive no substitute forward.
- Substitute output is finite, padding is exact zero, and three-channel replication plus Depth normalization are exact.
- Every trainable parameter has optimizer membership exactly one. The 29 geometry weights enter `base_decay`, 14 SyncBN parameters remain in `base_no_decay`, and the seven substitute Conv weights/biases enter `new_decay`/`new_no_decay`.
- One AMP update was applied with finite loss and finite, nonzero substitute gradients. This is a minimal update-path check, not a training run.

## Scope exclusions and reading boundary

- The Gate-B evidence does not establish segmentation improvement, full-training stability, batch-size-10 memory feasibility, or full training cost.
- No Quick-Val, Main-Val, cloud job, Batch 2/T run, or official-test access was performed.
- `entire_missing@1.0` remains a stress condition; this route does not infer the hidden cause of an observable-empty Depth input.
- The full protocol and Gate-B figures are in `01_research/e1_batch1_protocol.md` and `02_evidence/report_e1_batch1b_roe_gateb.md`; the machine-readable evidence remains the canonical JSON artifact.
