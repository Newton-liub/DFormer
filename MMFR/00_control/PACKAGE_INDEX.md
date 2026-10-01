# MMFR package index

## Package identity

- Package: `MMFR_v4_1_blueprint_and_reference_package_2026-09-20`；仓库内 `MMFR/` 是证据与审核材料入口，不是第二份实时状态。
- Historical blueprint: [`MMFR v4.1`](../01_research/MMFR_research_blueprint_v4_1_2026-09-20.md)，冻结历史不追改。
- Latest design: [`A-v1任务效用控制残差`](../../doc/reports/2026-09-30-mmfr-next-generation-research-design.md)。
- Latest formal report: [`A-v1正式训练/唯一Quick-Val上级报告`](../02_evidence/report_mmfr_a_v1_formal_quickval_upper_review_20261001.md)（2026-10-01；Proposal1920/Gate640完成、skip0，四条件筛选`stop`）。
- Local conditional handoff: [`资料接收与本地val前置条件`](../02_evidence/handoff_mmfr_a_v1_local_val_conditional_20261001.md)；[`转移包交付收据`](../02_evidence/delivery_mmfr_a_v1_local_val_20261001.md)记录整包SHA、大小与逐成员核验。打包不授权新实验，Main-Val适配尚缺。
- Historical implementation evidence: [`Gate-B报告`](../02_evidence/report_mmfr_a_v1_implementation_gateb_20260930.md)、[`差异审核`](../02_evidence/audit_mmfr_a_v1_implementation_20260930.md)，原结论/字节保留。
- Pre-revision snapshot: [`pre-A-v1设计`](../90_archive/2026-09-30_mmfr_a_v1_design_revision/2026-09-30-mmfr-next-generation-research-design.pre-a-v1.md)，历史0.0.15展示仅对应此设计。
- Existing active profile: `mmfr-a-v1-action-utility-gateb`，review level `L1`；[profile源](review_profile.json)已滚动更新为完成结果审核，沿用原类型。

## Authoritative research state

事实、权限、阻塞和恢复点只以 [`MUSeg-current-status.md`](../../doc/main/MUSeg-current-status.md) 与 [`MUSeg-open-decisions.md`](../../doc/main/MUSeg-open-decisions.md) 为准。

**大白话：** 训练跑完不等于有性能收益；本轮learned困难条件平均比matched off低0.001873pp，未通过+0.50pp门槛，当前停止实验。上级若例外同意本地val，还需明确新授权和完成A-v1评价适配。

## Quick navigation

| 需要知道 | 权威材料或证据 |
| --- | --- |
| 当前阶段和授权 | [`唯一实时状态`](../../doc/main/MUSeg-current-status.md)；[`真正未决事项`](../../doc/main/MUSeg-open-decisions.md) |
| 本轮结果、失败修复、精确分数与局限 | [`2026-10-01上级报告`](../02_evidence/report_mmfr_a_v1_formal_quickval_upper_review_20261001.md) |
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

既有profile的 `CURRENT_REPORT.md` 源现已指向本轮正式报告；生成包仍是旧快照。当前Linux环境直接核验无`pwsh`，本次未安装PowerShell、未运行生成器、未手改六文件包。**转交本轮上级报告即可完成此次汇报；旧包不可当作最新完成结果审核包。**

正式六文件审核入口仍限定 `99_review_packet_current/`，只可在PowerShell可用环境先核验canonical Markdown链接，再用既有生成器重建。从MMFR目录执行：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\98_tools\rebuild_review_packet.ps1
```

生成器读取既有profile、注册表、复现/验证证据，记录source/packet SHA与Git身份；不新增manifest体系。历史实现差异附件是2026-09-30版本，本轮训练/数值/评价修复以新报告与实际runtime身份补充。待重建后再上传完整六文件目录，不能把独立报告或转移ZIP伪称为该生成产物。

## External material boundary

论文全文/抽取图片表格/supplementary仍在仓库外 `D:\0Project\origin\论文\`；外部clone仍在 `D:\0Project\origin\`。包内只保留小型索引、分析、来源和证据，不存checkpoint、dataset、大日志/cache。新paper bundle优先通过 `human/paper_library.cmd` 接入，机器索引为 `03_reference/PAPER_LIBRARY_INDEX.json`，人工编号字段不自动覆盖。

14份 `03_reference/source_materials/` 原文件字节、大小和registry SHA保持不变。本次不改历史报告正文、实验结果、归档蓝图、原论文或参考代码。
