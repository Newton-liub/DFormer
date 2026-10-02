# MMFR package index

## Package identity

- Package: `MMFR_v4_1_blueprint_and_reference_package_2026-09-20`；仓库内 `MMFR/` 是证据与审核材料入口，不是第二份实时状态。
- Historical blueprint: [`MMFR v4.1`](../01_research/MMFR_research_blueprint_v4_1_2026-09-20.md)，冻结历史不追改。
- Latest candidate proposals: [`方向A：自然空洞形态覆盖`](../../临时/MMFR_direction_A_natural_missing_2026-10-02.md)、[`方向B：推理协议研究`](../../临时/MMFR_direction_B_inference_protocol_2026-10-02.md)（上级v1.0原稿；待审，暂保留原位置，未授权实现/实验）。
- Latest formal report: [`新方向计划、项目现状与最小文件治理上级审计`](../../doc/reports/2026-10-02-direction-plans-project-readiness-upper-review.md)（2026-10-02；推荐A优先、B备用与S1共用，明确旧C0继承/辅助loss及入口硬约束；仅审计）。
- Prior local evidence: [`两个候选方向的本地事实摸底`](../../doc/reports/2026-10-02-direction-audit-local-evidence.md)（自然Depth缺失、F-lite协议差异、baseline/NYU与官方代码；原始事实/局限保留）。
- Previous implemented design: [`A-v1任务效用控制残差`](../../doc/reports/2026-09-30-mmfr-next-generation-research-design.md)；对应正式筛选已stop，不再作为新方向入口。
- Latest experiment report: [`A-v1正式训练/唯一Quick-Val上级报告`](../02_evidence/report_mmfr_a_v1_formal_quickval_upper_review_20261001.md)（2026-10-01；Proposal1920/Gate640完成、skip0，四条件筛选`stop`，不因调查自动复活）。
- Local conditional handoff: [`资料接收与本地val前置条件`](../02_evidence/handoff_mmfr_a_v1_local_val_conditional_20261001.md)；[`转移包交付收据`](../02_evidence/delivery_mmfr_a_v1_local_val_20261001.md)记录整包SHA、大小与逐成员核验。打包不授权新实验，Main-Val适配尚缺。
- Historical implementation evidence: [`Gate-B报告`](../02_evidence/report_mmfr_a_v1_implementation_gateb_20260930.md)、[`差异审核`](../02_evidence/audit_mmfr_a_v1_implementation_20260930.md)，原结论/字节保留。
- Pre-revision snapshot: [`pre-A-v1设计`](../90_archive/2026-09-30_mmfr_a_v1_design_revision/2026-09-30-mmfr-next-generation-research-design.pre-a-v1.md)，历史0.0.15展示仅对应此设计。
- Existing active profile: `mmfr-a-v1-action-utility-gateb`，review level `L1`；[profile源](review_profile.json)已滚动更新为完成结果审核，沿用原类型。

## Authoritative research state

事实、权限、阻塞和恢复点只以 [`MUSeg-current-status.md`](../../doc/main/MUSeg-current-status.md) 与 [`MUSeg-open-decisions.md`](../../doc/main/MUSeg-open-decisions.md) 为准。

**大白话：** 当前已收到两份新方向方案，建议先用三组训练验证自然空洞形态是否有价值，推理协议研究暂作备用。但旧C0配置带有历史增强/辅助训练，需先定新合同并恢复真实权重；本轮只有审计报告，没有实现或新实验。A-v1仍stop，任何新准备、GPU或评价都需明确授权。

## Quick navigation

| 需要知道 | 权威材料或证据 |
| --- | --- |
| 当前阶段和授权 | [`唯一实时状态`](../../doc/main/MUSeg-current-status.md)；[`真正未决事项`](../../doc/main/MUSeg-open-decisions.md) |
| 本次新方案、接入风险与8项裁决 | [`新方向项目准备审计`](../../doc/reports/2026-10-02-direction-plans-project-readiness-upper-review.md)；原稿A/B见本页Package identity，尚未移至canonical计划目录 |
| 原方向A/B输入与历史协议事实 | [`2026-10-02本地调查报告`](../../doc/reports/2026-10-02-direction-audit-local-evidence.md)，保留原证据；本轮不重跑统计 |
| 上次A-v1结果、失败修复、精确分数与局限 | [`2026-10-01上级报告`](../02_evidence/report_mmfr_a_v1_formal_quickval_upper_review_20261001.md) |
| 正式合同和局部FP32完整1280恢复 | [`A-v1 protocol §8`](../01_research/mmfr_a_v1_action_utility_protocol.md#8-2026-10-01-nmf局部fp32修订与完整1280恢复最新授权)；旧章节保留历史，当前完成结果见实时状态 |
| 运行身份与结果JSON | [`reproducibility_current.json`](../02_evidence/reproducibility_current.json)，`current_cloud_outcome_pointer`指向`cloud_precision_amended_resume`；旧pending字段不当当前权限 |
| 条件性本地转移包 | [`本地val交接`](../02_evidence/handoff_mmfr_a_v1_local_val_conditional_20261001.md)；包在仓库外，不含dataset/test，大checkpoint不进Git |
| 文献/参考代码实体 | [`论文机器索引浏览入口`](../03_reference/PAPER_LIBRARY_INDEX.md)、[`paper-index`](../03_reference/paper-index.md)、[`code-index`](../03_reference/code-index.md) |
| 历史路线结论 | [`R-OE-lite v2报告`](../02_evidence/report_e1_batch1b_roe_v2_formal_training_quickval_20260924.md)、[`Batch1A Main-Val报告`](../../doc/reports/2026-09-22-mmfr-e1-batch1a-c0-flite-mainval.md)，独立于本轮A-v1 |
| 审核包生成与当前限制 | 本文件下节；[`变更历史`](CHANGELOG.md) |

完整文献编号仍由 [`reference index`](../03_reference/MMFR_reference_index_v4_1_2026-09-20.md) 与 [`registry`](../03_reference/MMFR_reference_registry_v4_1_2026-09-20.json) 管理，不重新编号或覆盖human字段。

## Directory map

- `00_control/`：导航、放置规则、变更历史与现有review profile。
- `01_research/`：蓝图、当前方案与protocol。
- `02_evidence/`：工程审核、实验报告、验证与复现记录。
- `03_reference/`：索引/注册表、文献分析、外部代码来源与14份原始source materials。
- `90_archive/`：按重大事件保留的历史版本，不为每个小动作建档。
- `98_tools/`：现有审核包生成器。
- `99_review_packet_current/`：仅生成产物，禁止手工编辑。

## Review packet: source current, generated output stale

既有profile的 `CURRENT_REPORT.md` 源仍指向2026-10-01 A-v1实验报告；生成包仍是更早的历史快照。上一轮本地调查按用户限定未生成包；本轮收到方向A/B方案后仅准备独立上级审计报告，沿用旧A-v1 profile作为历史审核身份，不手改生成产物或新建profile类型。当前直接转交上述新方向项目准备审计及两份原稿，旧包不能冒充本次交付。生成器只接受MMFR包内canonical源；待明确方案正式位置与新的审核范围后再用现有generator重建，不为适配它复制第二份报告。

正式六文件审核入口仍限定 `99_review_packet_current/`；未来另获授权时先核验对应profile的canonical Markdown链接，再用既有生成器重建。从MMFR目录执行：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\98_tools\rebuild_review_packet.ps1
```

生成器读取既有profile、注册表、复现/验证证据，记录source/packet SHA与Git身份；不新增manifest体系。历史实现差异附件是2026-09-30版本，本轮训练/数值/评价修复以新报告与实际runtime身份补充。待重建后再上传完整六文件目录，不能把独立报告或转移ZIP伪称为该生成产物。

## External material boundary

论文全文/抽取图片表格/supplementary仍在仓库外 `D:\0Project\origin\论文\`；外部clone仍在 `D:\0Project\origin\`。包内只保留小型索引、分析、来源和证据，不存checkpoint、dataset、大日志/cache。新paper bundle优先通过 `human/paper_library.cmd` 接入，机器索引为 `03_reference/PAPER_LIBRARY_INDEX.json`，人工编号字段不自动覆盖。

14份 `03_reference/source_materials/` 原文件字节、大小和registry SHA保持不变。本次不改历史报告正文、实验结果、归档蓝图、原论文或参考代码。
