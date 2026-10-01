# MMFR v4.1 package change log

## 2026-10-01 — 完整Git收口及云端连续执行授权

- 用户明确授权本地全部改动收口并push至origin同名分支；论文库报告、历史0.0.15 Canvas及索引一起保留。历史pre-A-v1归档的纯换行差异恢复原字节；运行代码/config无新内容变化。
- 云端可实现必要薄入口、依次Proposal/Gate各最多3次成功更新预检，通过后从原C0干净开始1920+640正式训练，再做一次同checkpoint off/full/learned四条件Quick-Val，完成后停止。无需各阶段重复等待授权，不扩Main-Val/official test/新seed/超参搜索。
- 普通工程修复允许最小检查后提交/push；每次运行记录实际代码提交，合同变更、OOM需改batch或恢复身份不明确时交人工。不追逐文档自身SHA；重要数据/权重/原始证据身份仍须匹配。
- 更新滚动实时入口、protocol最新授权、既有profile/复现信息和生成审核包。本地只Git/文档检查，未运行测试/GPU/训练/Val/云端。

## 2026-09-30 — A-v1 正式训练前本地收口：合同冻结，等待4090短预检

- 用户冻结第一轮Proposal/Gate=1920/640 successful updates、margin0.01、lambda_clean0.1、AdamW新branch LR3e-5/WD0.01；batch10、480×640、AMP/TF32 on、workers8、accumulation1与原phase seeds不变。config补最少正式合同字段，不修改结构/forward/loss/RNG或旧evaluator/corruption。
- 固定transition=`proposal-update-1920.pth`、fixed-final=`update-2560.pth`、recovery每640成功更新；无val selector、自动追加epoch或utility early-stop。Quick-Val唯一三项继续线与selector-not-supported/stop/inconclusive归类已预冻结，仅合同、未执行。
- 开始时五个关键文件hash与Gate-B完全一致；source C0和原Gate-B证据hash再次匹配。正式config独立新hash记入既有reproducibility JSON，config import/合同字段断言PASS；不重跑GPU Gate-B。
- 4090交接限定Proposal最多3-update，正常后Gate最多3-update，各阶段停止并回报显存/速度/loss/utility/稳定性，禁止自动接正式训练。当前尚无全尺寸runner，本轮只写调用合同，不编造可执行命令。
- 更新滚动实时入口、开放决策、当前设计补充与既有审核profile；按现有generator重建审核包。精确暂存A-v1文件形成一个本地commit，完整SHA提交后回填为元数据回执；既存无关dirty保留，不push、不训练、不运行Val、不访问official test、不操作云端。

## 2026-09-30 — A-v1 最小实现与 Gate-B PASS，等待正式训练授权

- 实现独立 `MMFR-A-v1-action-utility-v1`：stage2 单点 zero-init proposal、260→16→1 每图 gate、off/full/learned；新增 21,185 参数，内部 geometry/HAM 算法与旧 E1 protocols/evaluator 不改。
- source C0 SHA-256 实现前/加载/工程更新后均为 `ca618b23d18eabb201a0d11d18da383ac99576d0feae5864e3233bda527d9a1a`；strict off 和 zero-init full 与 C0 bitwise equal、max abs error=0。
- 限定本地 RTX 5060、synthetic B1 64×64、真实 C0/HAM；8 forwards、2 backward/step（每阶段一次），全部必要 Gate-B PASS。base 参数/buffers/BN 不变，phase optimizer isolation、detached utility branches/all-ambiguous、同 observed input/label/RNG/NMF bases 已直接核验。
- 增加 SpotTune/DCRM-ViT 两篇近邻补丁，结构 routing/gating 不作为创新；DCRM PDF 正文未核验，不声称论文创新性。补完即停止搜索。
- 新增 protocol、Gate-B JSON、审核报告/差异说明，更新实时入口、报告索引与当前审核 profile；由既有 generator 重建当前审核包，旧证据与源论文保持不变。Gate-B JSON SHA-256 `bf87cc2fc46dd413131f7a6778d8df82388cf7fffa08852126d1f62178be01bf`。
- 正式 margin/lambda_clean 未选；Gate-B 后停止，无正式训练/Quick-Val/Main-Val/official test/云端，未 commit/push。未来训练须上级审核及独立授权。

## 2026-09-30 — A-v1 定向修订，design-ready，等待复核与工程授权

- 按上级“候选 A 有条件通过”修订既有研究设计报告；不重新设计 B/C，不改历史 F-lite/R-OE/Oracle 结果与裁决。
- F-lite 明确为输入相关 R(F)、缺少独立监督的 intervention selector；A-v1 删除额外残差幅度控制及连续收益映射，采用固定正 margin 的 off/full utility 正负分类，ambiguous 排除 BCE，保留 gated CE 与必要 clean consistency。
- Gate 阶段冻结 C0/proposal、重新采样合法 Depth corruption realization；首版每图一个连续 g，只解释动作选择。保留同 checkpoint off/full/learned 与四条件 gate mean/median，strict off 不匹配 C0 即 BLOCKED。
- 六组窄查询、三项关键原文/官方摘要核验，当前未发现高度相同机制；SkipNet 全文监督与 MoSA 矛盾仍有限制，创新风险中高，不宣称首创。
- 修订前报告已在 `90_archive/2026-09-30_mmfr_a_v1_design_revision/` 字节一致归档，Git blob `7ce51c3d93b08a0943d2e9eed6271c8803f41e7d`；初版 0.0.15 展示保留历史，不代表 A-v1。同步实时入口、必要导航与报告索引。
- 状态为 A-v1 design-ready，等待上级复核 m、clean consistency 权重、阶段预算/停止线与最小实现/Gate-B 单独授权。本轮纯文档内容/diff 检查，未实现、未运行 Gate-B/测试/GPU/训练/Val，未改 evaluator/protocol，未重建旧审核包、提交或推送；official test 仍 sealed_unread。

## 2026-09-30 — 下一代 MMFR 研究设计，等待上级审核

- 新增 `doc/reports/2026-09-30-mmfr-next-generation-research-design.md`，复核 F-lite / R-OE-lite v2、九篇 canonical 论文及 DFormerv2 实际接口，比较任务效用控制残差、输入深度校准、轻量自教三个候选。
- 主推荐为同输入/同权重动作收益监督的单点残差与样本级连续控制；它是待审核研究假设，不是可靠性融合已胜出或全领域创新裁决。输入校准淘汰为主创新，自教保留性能备选；MoSA/DCF 重合与原文监督定义矛盾已显式记录。
- 第一阶段仅提出一条新训练与四条件单视图筛选，以及同 checkpoint off/full/learned 诊断。没有实现、训练、评价或修改旧 protocol/evaluator；Oracle-A 已关闭的 geometry suppression 不复活。
- 更新唯一实时入口、开放研究决策及本包导航；旧 blueprint、历史实验/文献编号、原始论文、review profile 与生成审核包保持不动。设计交付后停止等待上级审核；official test 保持 sealed_unread，没有提交/推送。

## 2026-09-24 — R-OE-lite v2 formal training PASS and four-condition Quick-Val result

- Added `02_evidence/report_e1_batch1b_roe_v2_formal_training_quickval_20260924.md` recording the entry-point blocker and its one-line non-committed fix, the 3-update GPU memory check, the 2560/2560 formal run identity, the fixed-final checkpoint SHA-256, the substitute-training evidence, the new R-OE-aware Quick-Val entry, the four-condition numbers, and the routing telemetry.
- Recorded that commit `255780f` alone cannot run: `utils/train.py` only accepted the `...-v1` protocol identity, so the approved v2 identity required a whitelist addition. The fix is uncommitted and un-pushed; its diff is archived as `cloud/mmfr-e1-batch1b-roe-v2/roe-v2-entry-fix.patch`.
- Recorded that the frozen `tools/mmfr/e1_quickval.py` cannot score an R-OE-lite checkpoint (the model forward requires `raw_depth` and the geometry mask), so a thin new entry `tools/mmfr/e1_quickval_roe.py` was added following `r_oe_lite_design.md` §11–§12; it has no prior frozen qualification and its `V_geom` unit-mask choice for the non-padded `original-full` view is recorded in every artefact.
- Result: Quick-Val clean `53.46` vs C0 `53.46` (delta `0.00`), `entire_missing@1.0` `48.77` vs `48.76` (`+0.01`), `spatial_dropout@0.75` `51.41` vs `51.40` (`+0.01`), `misalignment@0.75` `52.16` vs `52.15` (`+0.01`); $M_{3,\mathrm{hard}}$ `50.77 -> 50.78`; frozen screening verdict `inconclusive`.
- Updated the authoritative current-status and open-decisions documents: v2 is no longer pending execution, and the remaining live question is the disposition of the R-OE route given a screening delta far below the Main-Val promote threshold. Batch 1B ten-condition Main-Val, Batch 2, T and official test were not run; `official_test` remains `sealed_unread`. No Gate-B rerun, no review-packet rebuild, no C0 retraining, and no commit or push.

## 2026-09-24 — R-OE-lite v2 单通道上采样调整与条件训练授权

- 将 `d2 → GELU → head` 移到最终全分辨率双线性插值之前；不增删可训练层，参数仍为 `3,302,785`。原 v1 逐像素函数改变，v2 使用新 protocol/run identity，严禁继承 v1 中止训练的更新状态。
- 本地仅核查 v2 config/substitute import、CPU 输入输出尺寸、单通道、掩码覆盖区严格为零及参数量；路由与 reliability auxiliary 代码未改。RTX 4090 三步及正式训练结果尚未取得。
- 用户授权：保留 batch size 10 和其余 Batch 1B 训练设置，先连续 3 次成功更新并记录显存/损失/触发数/插值输入；满足无 OOM、loss finite 和显存余量条件后，从 A2 epoch-420 source 干净开始 2560 次更新，再进入已授权四条件 Quick-Val。OOM 则立即停止。v1 报告与旧审核包保持历史身份，未重跑 Gate-B、未重建审核包。

## 2026-09-24 — E1 Batch 1B R-OE-lite formal training stopped by CUDA OOM

- Added `02_evidence/report_e1_batch1b_roe_formal_training_attempt_20260924.md` recording the cloud run identity, OOM evidence, update-count boundary, absent fixed-final checkpoint, unrun Quick-Val, and unresolved C0 checkpoint path.
- Updated the authoritative current-status and open-decisions documents: the run ended at 01:45:30 UTC after at least 168 successful updates; the exact final count was not persisted; no automatic retry is authorized by this recovery point.
- Updated `PACKAGE_INDEX.md` to point to the new attempt report. The frozen R-OE-lite protocol, Gate-B evidence, prior review packet, and official-test seal were not rewritten; no review packet rebuild was requested or performed.

## 2026-09-23 — refreshed Batch 1B L1 packet for senior review

- Rebuilt the active `e1-batch1b-roe-gateb` packet at HEAD `2c3176e375562cd62be31bb10cb9363537edb4f3`; generation ID `8213dfb94396d4c65f4200c94029308573332f210665c578ccf323577e6a0415`.
- Confirmed the six expected flat packet files, zero broken canonical/packet/copied-packet links, and `official_test=sealed_unread`.
- Refreshed reproducibility metadata for the current commit and clean-at-start worktree; no training, evaluation, cloud task, commit, or push was performed.

## 2026-09-23 — E1 Batch 1B R-OE-lite Gate-B review packet

- Recorded the completed R-OE-lite implementation and Gate-B PASS in the protocol, screening plan, design record, Gate-B report, and implementation-diff audit. The canonical Gate-B JSON SHA-256 is `35297b490c3e3eb54b5e66d3f06784688fca65038e7d09874c30c60cde820231`.
- Froze the ten-condition Main-Val promote/stop/inconclusive/blocked rules and clarified that Quick-Val is screening only, `entire_missing@1.0` is a stress condition rather than hidden-cause detection, and Gate-B does not authorize training or evaluation.
- Changed the active L1 review profile to `e1-batch1b-roe-gateb`, with the Batch 1B report and implementation-diff audit as exact-copy attachments; rebuilt the six-file packet with generation ID `86a92d119a8a1e0b2c33a2ad6fba8876aef0e2c26c1233f8fb7a28c9d2272691` and zero broken links.
- Updated the authoritative MUSeg status and open-decision pointers, package navigation, reproducibility metadata, and report index to distinguish the pending Batch 1B training authorization from the independent Batch 1A Main-Val review.
- Formal training, Quick-Val, Main-Val, cloud execution, Batch 2/T, and official-test access remain unauthorized; `official_test` remains `sealed_unread`.

## 2026-09-23 — restored into the project repository, external-reference indexes added

- Moved the package from `D:\0Project\DFormer-archive-20260922\liu-test-exp\MMFR\MMFR_v4_1_blueprint_and_reference_package_2026-09-20\` back to the project root as `D:\0Project\DFormer\MMFR\` and brought it under Git.
- Moved paper full texts and extraction artifacts out of the package and out of Git to `D:\0Project\origin\论文\`, under two provenance roots (`MMFR-附件`, `DFormer-doc-paper`). The package now keeps only Markdown, TXT, small JSON, indexes, and packet tooling.
- Repaired the package-relative links the move invalidated: `../../../../` -> `../../` in `00_control/PACKAGE_INDEX.md`, `01_research/e1_screening_plan.md`, `01_research/MMFR_research_blueprint_v4_1_2026-09-20.md`, and `03_reference/literature_fulltext_audit.md`; local-only attachment paths now read `D:\0Project\origin\论文\MMFR-附件\`; the authority paths in `review_profile.json` now resolve from the new location.
- Added `03_reference/paper-index.md` (paper number -> local full text -> local code) and `03_reference/code-index.md` (read-only local clone inventory). Neither renumbers existing PR/RE/AI/MoSA/ANGA codes.
- `PACKAGE_INDEX.md` gained a quick-navigation table and an explicit packet-state note; `FILE_RULES.md` now places paper full texts outside the repository.
- Research content, Gate-B evidence, authorization boundary, and the official-test seal were not changed. The review packet was **not** regenerated: the packet in `99_review_packet_current/` still belongs to profile `e1-batch1a-gateb`, and its recorded canonical source hashes predate these link repairs.
- Known blocker for the next packet rebuild: `01_research/e1_screening_plan.md` links to `02_evidence/report_e1_batch1b_roe_gateb.md`, which is not yet present in the package, and `98_tools/rebuild_review_packet.ps1` fails hard on a broken canonical link.
- Repaired `98_tools/rebuild_review_packet.ps1`'s `$repoRoot`, which still assumed the old three-level-deep package location; it now resolves the Git work tree one level above the package root, so the generator can read the repository identity again.

## 2026-09-21 — second directory simplification

- Consolidated the long-term package into `00_control`, `01_research`, `02_evidence`, and `03_reference`, while retaining separate archive, tooling, and generated-review areas.
- Replaced the package-local README/status pair with `PACKAGE_INDEX.md`; research facts and authorization remain authoritative only in `doc/main/MUSeg-current-status.md` and `doc/main/MUSeg-open-decisions.md`.
- Moved the active review profile into machine-readable `00_control/review_profile.json`.
- Preserved the 14 source-material files byte-for-byte while moving them to `03_reference/source_materials/`.
- Replaced pre-created empty archive categories with an event-based migration archive containing the pre-migration manifest and superseded control documents.
- Redesigned the current L1 review packet as a flat six-file interface with one human entry, one machine manifest, three exact-copy attachments, and one generated evidence summary.
- Kept the research conclusions, Gate-B decision, authorization boundary, external-code isolation, and official-test seal unchanged.

## 2026-09-21 — package directory reorganization

- Created the numbered package layout for control, blueprint, references, plans/protocols, engineering audits, experiment reports, validation/reproducibility, archives, tools, and the current review packet.
- Moved the current blueprint, reference index/registry, literature audit, E1 plans, engineering audits, Gate-B report, and validation report to their fixed directories without changing their research content.
- Preserved all 14 original `source_materials` files under `02_reference/03_source_materials/`.
- Recorded external paper-code provenance separately from external source trees; external repositories remain under `D:\0Project\origin`.
- Added package-management status, file-placement rules, current reproducibility metadata, and deterministic review-packet generation.
- Repaired package-relative links and registry source paths after the move.

## 2026-09-20 — v4.1 blueprint and provenance package

### 保持不变

保留 v4 正文0–13节的路线、候选体系、比较逻辑和历史结果。原正文全部非空行按原顺序保留；借鉴表只插入既有论文编号。未执行训练，未修改项目仓库或评价结果。

### 补充内容

- 正文对应位置加入34个实施步骤的论文/原材料/参数边界说明。
- 57条关键文献快速表；单独总索引含135条带题名文献和双向步骤映射。
- 六个子计划读文献包、13项元数据/版本/实现边界台账。
- S1–S14原材料清单与实际SHA-256；W1–W26网络来源记录。
- 保留124个旧清单中只有编号的归档项；RE067明确为提示词示例，不计为论文。
- 原附件与本轮网络补充字段分开，不用后者覆盖附件。
- 以JSON保存全量元数据、源定位、步骤关系和网络状态。

### 范围限制

这是索引与溯源修订，不是135篇论文的重新全文审计。
未验证本地模型代码、数据、checkpoint或实际模块收益。
部分出版商入口受限；明确标记，未猜造DOI、代码仓库或原文超参数。

### 文件一致性检查

校验结果见 `06_validation_and_reproducibility/validation_report.json`；全部论文/步骤/网页编号引用可解析，原附件副本哈希一致，原v4正文无非空行丢失。
