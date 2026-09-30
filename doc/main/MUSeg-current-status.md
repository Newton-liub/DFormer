# MUSeg 当前状态与唯一实时入口

> **事实截至：** 2026-09-30 论文 paper bundle 首次整理及索引定点验证。当前研究授权与未决判断仍见 [`MUSeg-open-decisions.md`](MUSeg-open-decisions.md)；本次不运行训练、验证或文献调研。

## 当前阶段与最近一个大动作

**研究阶段未切换：** MMFR（多形式模态失效与可靠性）E1 Batch 1B 仍处于 `R-OE-lite-v2-training-and-quickval-complete`。2026-09-24 v2 在 RTX 4090 完成 2560/2560 次成功更新，fixed-final checkpoint SHA-256 `369d7e254acadb3bec10d4b6f362d1601291b07edc604bdb671a47591d2a5fbc`；四条件 Quick-Val 相对 C0 为 `0.00 / +0.01 / +0.01 / +0.01` pp，筛选结论 `inconclusive`。十条件 Main-Val、Batch 2、T、official test 仍未获授权，official test 保持 `sealed_unread`。详见 [`v2 训练与 Quick-Val 报告`](../../MMFR/02_evidence/report_e1_batch1b_roe_v2_formal_training_quickval_20260924.md) 与 [`开放决策`](MUSeg-open-decisions.md)。

**本次大动作：论文库实体整理。** `D:\0Project\origin\论文\` 中整理前识别到 34 个可作论文正文的 bundle，另有 1 份 DFormerv2 正式 supplemental（不是独立论文正文）。现有 32 条 canonical 论文记录（`LIB000001`–`LIB000032`）；31 个一级论文文件夹已规范化重命名，1 份正文 SHA-256 相同且主要资源数量一致的 bundle **按简单签名**判为 exact duplicate 并整包移至仓库外 `D:\0Project\origin\论文_duplicates_review\`，1 组 DFormerv2 同论文、不同抽取保留原位作为 `alternate_bundles`，没有合并或选优。索引继承 16 个已核验的人工编号（含 PR/RE/AI 和 MoSA），未发生需要覆盖旧映射的冲突；RE042、RE053、RE188、RE447 仅记为待核验旧编号，不作为已确认的人工映射。**大白话说明：** 今后可用 LIB ID 查正文和图表资源，旧 PR/RE/AI 编号继续保留；版本拿不准的材料没有为求目录整齐而合并或删除。

机器事实源为 [`PAPER_LIBRARY_INDEX.json`](../../MMFR/03_reference/PAPER_LIBRARY_INDEX.json)，人工浏览入口为 [`PAPER_LIBRARY_INDEX.md`](../../MMFR/03_reference/PAPER_LIBRARY_INDEX.md)，手动接入工具为 `human/paper_library.cmd`。本次工具、索引和必要入口说明已本地提交，**未推送**；提交以本地 Git 记录为准。已知旧来源 `D:\0Project\DFormer-archive-20260922\doc\paper` 当前不存在，本次只扫描当前主库，未进行全盘搜索。仍在旧主题目录中的不同抽取、正式 supplemental 和 PDF-only 材料未作为 canonical paper 自动搬动；因此“32 篇 canonical 论文一级扁平”不等于主库已没有任何历史非 canonical 目录。

## 授权边界与下一恢复点

- 本次只整理论文资产与索引；未重新分析论文内容，未开展新文献检索、训练、Val 或新模块设计。DFormerv2 两份抽取的选优、正式 supplemental 是否另行挂载为附属资源、四个待核验 RE 编号归属、不可访问的旧归档目录是否需补扫，由用户另行决定；不依据相似标题推断正式版本关系。
- 上轮 `utils/train.py` 的 v2 身份入口修复与 `tools/mmfr/e1_quickval_roe.py` 的冻结资格处置仍开放，R-OE-lite 路线去留及 F-lite Main-Val 的研究判断仍见 [`MUSeg-open-decisions.md`](MUSeg-open-decisions.md)，本轮未改变任何实验授权。
- 下次新增论文使用 `human/paper_library.cmd` 双击刷新或拖入完整 bundle/批量父目录。先核验有争议的新身份、编号和版本，再决定是否调整 canonical；仅索引内的 `human` 字段由人工维护，自动刷新不得覆盖。
