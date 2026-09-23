# Evidence summary

- Profile: `e1-batch1b-roe-gateb`
- Generation identity: `86a92d119a8a1e0b2c33a2ad6fba8876aef0e2c26c1233f8fb7a28c9d2272691`
- Scope: review-relevant conclusions extracted from canonical sources; canonical files remain authoritative.

## Observable-empty substitute route

- Status: Implementation complete; Gate-B PASS; formal training unauthorized
- Current relevance: The review is limited to whether the implemented substitute follows the frozen observable-empty contract and passes the bounded engineering gate.
- Canonical sources:
  - `02_evidence/report_e1_batch1b_roe_gateb.md` - SHA-256 `fa550e995aa07546935e6c5b05e79c475d07f445f7163388092d1db7d67d7a97`
  - `02_evidence/audit_implementation_diff_e1_batch1b_roe.md` - SHA-256 `39881d7ada1905a2a739e4f77ba7645b2d88a83964010235aef0644855a55fce`
  - `01_research/r_oe_lite_design.md` - SHA-256 `8a2ce33222c52ff530e1213105a1e8ec1b2d3b6c85d34236b672fa40e2e8966a`
- Key conclusions:
  - The detector observes only current raw Depth and geometry-valid support; it does not infer hidden failure causes.
  - Non-triggered samples preserve exact Depth bypass; triggered samples use the RGB-generated substitute for segmentation.
  - The reliability auxiliary path continues to use the original corrupted raw inputs and target.

## Matched C0 control and optimizer coverage

- Status: Gate-B checks PASS
- Current relevance: The Batch 1B candidate reuses the Batch 1A C0 checkpoint only because the shared training fields were verified equal for this Gate-B identity.
- Canonical sources:
  - `02_evidence/report_e1_batch1b_roe_gateb.md` - SHA-256 `fa550e995aa07546935e6c5b05e79c475d07f445f7163388092d1db7d67d7a97`
  - `01_research/e1_batch1_protocol.md` - SHA-256 `6ea5d9dc143102cdb3cfc56170606ad2907bd86daa530e0ea94b10c33fcb17b8`
- Key conclusions:
  - Shared config, post-build CPU/CUDA RNG, and first-epoch sample permutation match the C0 control.
  - Every trainable parameter has exactly one optimizer group membership; the substitute is split across new_decay and new_no_decay.
  - The one-step AMP result establishes only the minimum update path, not full training stability or capacity.

## Reproducibility identity

- Status: Canonical Gate-B artifact identity recorded; report and digest are in the review packet
- Current relevance: The canonical artifact binds the PASS result, implementation identity, and authorization boundary.
- Canonical sources:
  - `02_evidence/reproducibility_current.json` - SHA-256 `d0035a9bffac8341715cef658b30fa7884eca8077a3392d8f88f671d4871033e`
  - `02_evidence/report_e1_batch1b_roe_gateb.md` - SHA-256 `fa550e995aa07546935e6c5b05e79c475d07f445f7163388092d1db7d67d7a97`
- Key conclusions:
  - The Gate-B artifact SHA-256 is 35297b490c3e3eb54b5e66d3f06784688fca65038e7d09874c30c60cde820231.
  - The artifact records official_test_included=false and formal_training_started=false.
  - The official-test state remains sealed_unread.
- Review identity fields:
  - Git branch: `perf/mmfr-a2-v3-pipeline-opt1`
  - Git commit: `97de9f02aa543085f3085fbc59c231cf8b4867e2`
  - Tracked workspace dirty: `true`
  - Evaluator: `N/A - Gate-B engineering qualification`
  - Checkpoint effect selection: `N/A - not performed`
  - Official test: `sealed_unread`

