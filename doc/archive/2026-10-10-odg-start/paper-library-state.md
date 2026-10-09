# ODG 开始前的论文库状态摘要

> 从 `doc/state/current.md` 的2026-10-09阶段移出（2026-10-10）。这是已结束阶段的历史事实与审批恢复点；实时入口仍为 `doc/state/current.md`。完整开始前状态还保存在研究分支的 Git 提交 `08d59409e5251379c3305d2058a05d972fe1eb2d` 中。

## 已完成事实

- 论文库、索引与工具真源在仓库外：`D:\0Project\origin\论文\`（49篇 canonical）、`D:\0Project\origin\_index\`（唯一索引、工具、备份及导出）。没有纳入 Git；只有 store 自身的 JSON 备份。本仓库不复制第二份索引。
- schema4 / revision18 / 49篇 / next_lib_number50；索引更新时间2026-10-09 14:41:53（UTC+8）。当时索引SHA-256：`c7c0419e6bb3f1ddb4c554ae07d1ad9bf659ad191cc1a2809a152e201b96280e`。backups 为revision17/16/15，bak1与批准前F0逐字节一致；管理视图为revision18。
- 2026-10-09 14:34用户批准联合清单，核验111个不同文件后经既有store一次提交（17→18）。仅16篇记录的11个supplement、9个reading_status、15个is_core及GeminiFusion两个缺失书目override变化；其他原始与人工字段不变。回执：`_index\supplement-review\2026-10-09-final\SAVE_RESULT.json`。
- 最新入库四篇：AGWNet→LIB000046、GeoSphere-DETR→047、RGB-D Mirror Segmentation→048、SPGNet→049。来源仅复制，原目录保留。四篇已在库，不重复入库。
- 12篇原批准`full_text_completed`：LIB000002/023/026/028/032/033/034/037/038/039/040/044。9篇局部阅读为`needs_full_text`：LIB000001/014/024/042/045/046/047/048/049。其余28篇`abstract_only`；supplement21篇、核心15篇。旧全文状态表示正文已读并有证据笔记，不保证附录与全部资产核完。
- 三范围AI导出：全部49、核心15、已有补充21，均为schema4/revision18/format1.0.2。全部版实际header时间2026-10-09T14:51:59+08:00，核心/已有补充版为14:43:23。11篇批准草稿归一标题层级后均完整包含于相应条目。
- 自动化实现：项目两个论文 Skill、paper-library Rule，以及外部`paper_workflow.py`四子命令薄封装，复用既有扫描器/store，无新增数据库或常驻服务。两次真实试用为GeminiFusion入库（LIB45，revision12→13）与MoSA supplement（LIB28，13→14，仅supplement变化，阅读状态未自动提升）。

## 授权历史与失效计划

- 2026-10-09 14:32用户确认论文补充足够，停止新的论文搜索、PDF下载和重复阅读，不扩展范围、不重试DFormer++附录。14:34联合保存已完成，论文库工作线结束，不自动进入实验。
- 原审核根`D:\0Project\origin\_index\supplement-review\2026-10-09-final\`中的联合清单及11个单篇计划作为审批历史保留，绑定revision17/F0，保存后均失效。不得重复apply，也不得换新指纹绕过批准。
- index只能经既有store提交且携带审批基线指纹；无实际变化不保存。已有论文原资产只读，人工身份字段受保护，不手动分配LIB编号。
- 论文库后续写入、来源清理或工具维护仍需新授权。旧阶段的“不改代码、不提交推送、不启动云端”只描述该已结束任务，2026-10-10 ODG指令对本任务另行授权，未扩展论文库权限。

## 证据入口与仍未决项

- [定点补充审核与保存](../../reports/2026-10-09-paper-final-targeted-supplement-review.md)。
- [自动化最终收尾](../../reports/2026-10-09-paper-library-automation-final.md)、[实施与隔离验收](../../reports/2026-10-09-paper-library-automation-implementation.md)。
- [质量审核](../../reports/2026-10-08-paper-library-quality-audit.md)、[入库与审核](../../reports/2026-10-08-paper-library-batch-intake.md)。
- DFormer++官方附录418、未读；MoSA提取不一致、UMIS-Mine待审副本及Rules页目视确认仍未处理，不自动扩展。UMFNet数值冲突、SGMA公式维度及GeoSphere图表差异仍是引用限制，不由工具裁决。
- 核心区分：DFormerv2正文写softmax后乘prior，官方实现为softmax前加depth/spatial log bias；官方pretrained存在负几何融合权重。ECoLaF/GeoDistill/MoSA等已有可靠性/几何门控近邻。ODG未因此被证明创新或有效；最终科研裁决仍由用户或上级模型负责。
