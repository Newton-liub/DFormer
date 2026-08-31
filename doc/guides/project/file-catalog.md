# DFormer 完整文件目录

> 本目录按 2026-08-27 的 Git 索引建立，说明稳定职责而非复制实时实验状态。当前事实见 [`../main/MUSeg-current-status.md`](../../main/MUSeg-current-status.md)，开放研究选择见 [`../main/MUSeg-open-decisions.md`](../../main/MUSeg-open-decisions.md)。

## 1. 覆盖基线

- 分析分支：`main`
- 分析 HEAD：`c9ad268b5ee3ab685a4f93c945bfcdd843c49ab9`
- 上游：`origin/main`
- 基线分歧：ahead 0 / behind 0
- Git 跟踪文件总数：**1,141**
- 已说明跟踪文件数：**1,141**
- 明确排除跟踪文件数：**0**
- 未覆盖跟踪文件数：**0**
- 重复统计数：**0**
- 分析日期：2026-08-27
- 初始工作区：不干净；只有用户已有的未跟踪 `doc/临时/计划.md`，不属于覆盖分母且未修改。
- 本轮新增非基线路径：`doc/guides/project/README.md`、`doc/guides/project/file-catalog.md`。
- 后续文档治理差异（2026-08-27）：经用户确认删除早期 4090 云端训练交接计划；活动计划旧 05–11 又迁移为新 05–08，并新增日期化迁移审计。下方职责目录按当前工作树列新路径，不再列已删除路径；这些变化不改写上述 `c9ad268...` 历史基线计数。

覆盖方法：以 `git -c core.quotepath=false ls-files -z` 为唯一分母，从本文件的逐路径条目提取仓库相对路径：普通条目使用反引号列表项，`.mim/configs/` 使用清单代码块中的裸路径行。只统计这些条目，不把正文中的交叉引用重复计数；随后检查集合差、重复项、路径存在性和旧路径残留。二进制和 official-test 路径也纳入说明，只限制内容读取，不从覆盖中排除。

### 条目约定

- “当前”只表示文件职责，不复制其中会变化的事实。
- “上游/兼容”表示随论文实现或 MMSegmentation 兼容树保存；没有逐提交来源证据时不推断具体改动史。
- “历史”表示只能按形成时点解释。
- 同组清单前的共享说明适用于组内每个路径；路径仍逐一出现。复杂入口另列关键符号、输入、输出和消费者。
- 二进制资源只说明类型、已找到的引用和边界；无法确认用途时标“待核验”。

## 2. 根文件

- `.DS_Store` — 已跟踪的 macOS Finder 元数据；不参与运行，通常不应手工修改，保留原因待核验。
- `.gitattributes` — Git 属性；强制 `*.sh` 使用 LF，并把冻结 `official-test.txt` 设为二进制式内容以防行尾改写。
- `.gitignore` — 忽略数据、checkpoint、输出、缓存、物化 protocol 和多数根训练脚本，同时显式放行 MUSeg 配置、冻结 split、当前云指南与审计训练脚本。
- `LICENSE` — 非商业使用许可文本；分发和复用前必须阅读。
- `README.md` — 仓库总入口、MUSeg 导航、上游安装/数据/checkpoint/脚本/性能说明和 Canvas 发布约定。
- `train.sh` — 上游 NYUv2 双 GPU `utils/train.py` 示例；不是 MUSeg 审计入口。
- `eval.sh` — 上游 NYUv2 八 GPU `utils/eval.py` 示例，固定 checkpoint 和滑窗参数。
- `infer.sh` — 上游 NYUv2 双 GPU `utils/infer.py` 可视化示例。
- `requirements-monitoring.txt` — MUSeg 在线实验跟踪额外依赖，固定 `swanlab==0.9.7`。
- `skills-lock.json` — Cursor/Agent Skill 依赖锁定元数据；由 Skill 管理流程消费，不承载研究结论。

## 3. Agent、Cursor 与符号工具配置

- `.agents/skills/compshare-cli/SKILL.md` — CompShare CLI 的项目级操作 Skill，覆盖实例、SSH、文件、存储和计费操作；使用前必须核对外部状态与凭据边界。
- `.agents/skills/compshare-cli/agents/openai.yaml` — 上述 Skill 的代理界面/元数据配置。
- `.cursor/rules/museg-current-status.mdc` — 强制 MUSeg 唯一实时入口、同对话状态更新和提交前一致性规则。
- `.cursor/rules/subagent-delegation.mdc` — 仅允许把简单、低风险、边界清晰且耗上下文的机械任务委派出去。
- `.cursor/skills/research-progress-report/PROJECT_EVIDENCE.md` — 报告证据优先级与代码、实验、数据、文献、Git 的取证路径。
- `.cursor/skills/research-progress-report/REPORT_FORMAT.md` — 研究进展 Markdown/Canvas 的结构和表达规范。
- `.cursor/skills/research-progress-report/SKILL.md` — 研究进展报告 Skill 入口和工作流。
- `.serena/.gitignore` — Serena 本地缓存忽略规则。
- `.serena/project.yml` — Serena 项目语言、编码和符号分析配置；供代码导航工具读取。

## 4. 冻结数据身份 `data/`

本目录只保存 split 身份和审计证据，不保存完整图像。五个路径均由配置、protocol/preflight、split 工具和测试读取。

- `data/splits/MUSeg/dev-v1/manifest.json` — 冻结 split 权威清单；包含生成器、Git、哈希、样本/组计数、统计、集合关系、warnings 和 Gate A 签署元数据。当前候选状态为 frozen。
- `data/splits/MUSeg/dev-v1/audit-report.json` — 对 manifest schema、来源、成员、隔离、计数、哈希和统计的独立结构化复核；`pass=true` 不代表模型训练完成。
- `data/splits/MUSeg/dev-v1/train-dev.txt` — 1,277 个 development 训练样本身份；`RGB/<name>.jpg` 每行一项。
- `data/splits/MUSeg/dev-v1/val-dev.txt` — 318 个 development validation 样本身份；用于在线验证和本地后评估。
- `data/splits/MUSeg/dev-v1/official-test.txt` — official test 的冻结字节副本和身份锚点；内容在本次分析中未读取，development 不得消费。

## 5. 文档 `doc/`

### 5.1 当前、稳定入口与指南

- `doc/dataset.md` — MUSeg 数据准备、目录、固定深度量化、标签映射、split 和 BGR/official-test 边界的稳定入口。
- `doc/main/MUSeg-current-status.md` — 当前事实、正在进行事项、边界、证据位置和恢复点的唯一实时入口；允许按规则更新。
- `doc/main/MUSeg-open-decisions.md` — 开放问题、已处置选择及历史解释；只有真实研究选择或执行边界变化时更新。
- `doc/guides/README.md` — 稳定操作指南总索引及计划/报告/审计的文档状态头规范，不承担实验状态。
- `doc/guides/cloud/README.md` — 云端操作指南入口与生命周期提示。
- `doc/guides/cloud/compshare-filebrowser-legacy.md` — CompShare File Browser 旧界面说明；历史/legacy，使用前核对平台现状。
- `doc/guides/cloud/openlist-quark-download.md` — OpenList/夸克文件获取的稳定操作说明；外部服务与凭据可能变化。
- `doc/guides/cloud/museg-storage-backup-cleanup.md` — MUSeg 训练归档、本地哈希核验、OpenList 可选副本、只读空间审计与人工确认删除门禁。

### 5.2 审计、报告和索引

以下均是日期化证据或交接，不替代当前状态；`report-index.json` 提供元数据索引和 Canvas 版本登记。

- `doc/audits/2026-08-26-markdown-consistency-audit.md` — 2026-08-26 文档一致性审计的历史时点结论。
- `doc/audits/2026-08-27-museg-plan-conflict-migration-audit.md` — 活动计划冲突、颜色/geometry 契约与旧 05–11 到新 05–08 的日期化迁移审计。
- `doc/reports/2026-08-19-museg-data-reproduction-update.md` — MUSeg 全量转换复现与核验报告。
- `doc/reports/2026-08-19-museg-dformer-data-processing-review.md` — 数据集论文、格式、深度、标签和 DFormer 接口评审。
- `doc/reports/2026-08-19-museg-minimum-validation-paths.md` — 最小验证路径与证据阈值设计报告。
- `doc/reports/2026-08-21-museg-mve-a1-b1-a2-git-alignment.md` — MVE A1/B1/A2 与 Git 对齐审计。
- `doc/reports/2026-08-21-museg-mve-cleanup-and-disposition.md` — MVE 清理和历史 workload 处置报告。
- `doc/reports/2026-08-25-museg-stage-2-dev-split-freeze.md` — development split 冻结与审计报告。
- `doc/reports/2026-08-25-museg-stage-2-gate-b-auditable-training-pipeline.md` — 可审计训练管线 Gate B 报告。
- `doc/reports/2026-08-25-museg-stage04-cloud-qualification-handoff.md` — Stage-04 云端 qualification 历史交接。
- `doc/reports/2026-08-26-museg-stage05-seed1-local-closure-handoff.md` — seed 1 本地收口工作中交接；当前状态将其作为详细支持材料。
- `doc/reports/2026-08-26-museg-stage05-seed1-prelaunch.md` — seed 1 启动前历史快照。
- `doc/reports/2026-08-26-museg-stage05-seed1-running-handoff.md` — 训练运行中历史快照。
- `doc/reports/report-index.json` — 正式报告和 Canvas 元数据索引；其中“latest”字段按索引维护时点理解，不能覆盖当前状态。

### 5.3 当前计划

- `doc/plans/MUSeg-A2-B2深度有效性/00-总方向规划.md` — 当前 A2/B2 深度有效性小阶段的目标、条件分支和授权边界。
- `doc/plans/MUSeg-A2-B2深度有效性/01-新对话最小上下文与当前任务.md` — 新对话最小读取入口、搜索/验证预算和当前任务卡。
- `doc/plans/MUSeg-A2-B2深度有效性/02-A2正式验证与B2条件分支.md` — 正式 A2、条件式 B2 与零训练验证的三任务唯一当前方案。

### 5.4 历史计划

`doc/plans/README.md` 规定计划目录的历史语义；归档文件记录阶段设计、门禁和形成时点的执行状态，不能作为当前 workload 授权依据。

- `doc/plans/archive/README.md` — 计划归档说明。
- `doc/plans/archive/2026-08-MUSeg-DFormerv2快速Baseline/00-总方向规划.md` — 已完成 RGB Quick-B0 的历史总方向、共同身份和阶段边界。
- `doc/plans/archive/2026-08-MUSeg-DFormerv2快速Baseline/01-新对话最小上下文与当前任务.md` — Quick-B0 阶段的历史接续规范和任务入口。
- `doc/plans/archive/2026-08-MUSeg-DFormerv2快速Baseline/02-一次性B0执行方案.md` — 已完成的四任务 B0 训练、主评估和证据收口协议。
- `doc/plans/archive/2026-08-MUSeg阶段二长程Baseline与MVE/00-总索引与执行门禁.md` — 历史基础 01–04、活动 05–08 及 Protocol/MVE Evidence/Gate E/F 的依赖索引。
- `doc/plans/archive/2026-08-MUSeg阶段二长程Baseline与MVE/01-开发划分协议与生成工具.md` — development split 协议与生成工具设计。
- `doc/plans/archive/2026-08-MUSeg阶段二长程Baseline与MVE/02-训练验证Checkpoint与恢复改造.md` — checkpoint、validation 和恢复改造设计。
- `doc/plans/archive/2026-08-MUSeg阶段二长程Baseline与MVE/03-三种子编排Preflight与SwanLab改造.md` — seed 编排、preflight 和 SwanLab 设计。
- `doc/plans/archive/2026-08-MUSeg阶段二长程Baseline与MVE/04-静态验收与4090Qualification.md` — 静态验收、probe 和 qualification 设计。
- `doc/plans/archive/2026-08-MUSeg阶段二长程Baseline与MVE/05-协议校准与Development-B0收口.md` — 颜色/归一化、validation geometry、五项后评估和 development reference B0 收口计划。
- `doc/plans/archive/2026-08-MUSeg阶段二长程Baseline与MVE/06-MVE问题验证与B2条件分支.md` — B1 稳定性回归、A2 判据及 B2 规格/实现/zero-train/short 条件分支。
- `doc/plans/archive/2026-08-MUSeg阶段二长程Baseline与MVE/07-模块配对筛选与GateE冻结.md` — 通用模块/B2 的统一 paired screening、B0-only 分支和 Gate E 冻结。
- `doc/plans/archive/2026-08-MUSeg阶段二长程Baseline与MVE/08-正式三种子与Official-Test解封.md` — formal B0/最终 variant paired 三 seed、Gate F 与一次性 official-test 解封。

### 5.5 Canvas 与论文

Canvas 是可视化汇报源，已发布版本只读；`old/` 为归档。TSX 由 Canvas 运行时消费，不参与模型训练。

- `doc/canvases/0.0.8-museg-stage05-seed1-prelaunch.canvas.tsx` — seed 1 prelaunch 参考 Canvas。
- `doc/canvases/0.0.9-markdown-consistency-audit.canvas.tsx` — 文档一致性审计的历史 running 快照。
- `doc/canvases/old/0.0.1-report-workflow-decision.canvas.tsx` — 报告工作流决策归档。
- `doc/canvases/old/0.0.2-museg-dformer-data-review.canvas.tsx` — 数据处理评审归档。
- `doc/canvases/old/0.0.3-museg-data-reproduction-update.canvas.tsx` — 数据复现更新归档。
- `doc/canvases/old/0.0.4-museg-minimum-validation-paths.canvas.tsx` — 最小验证路径归档。
- `doc/canvases/old/0.0.5-museg-mve-outcome-and-disposition.canvas.tsx` — MVE 结果处置归档。
- `doc/canvases/old/0.0.6-museg-stage-2-dev-split-freeze.canvas.tsx` — split 冻结归档。
- `doc/canvases/old/0.0.7-museg-gate-b-auditable-training-pipeline.canvas.tsx` — Gate B 管线归档。
- `doc/paper/25CVPR_RGBDSeg-CN.pdf` — DFormerv2 中文论文 PDF；二进制参考资料，不手工修改。
- `doc/paper/Li 等 - 2025 - 1-MUSeg A multimodal semantic segmentation dataset for complex underground mine scenes.pdf` — 1-MUSeg/MUSeg 数据集论文 PDF；由数据评审报告引用。

## 6. 图片资源 `figs/`

图片为静态二进制资源；根 README 直接引用其中主要架构/性能图。没有文本引用证据的用途不按文件名继续推断。

- `figs/DFormer.png` — 根 README 的 DFormer 对比图。
- `figs/Sal.jpg` — 静态上游图片；当前直接引用待核验。
- `figs/Semseg.jpg` — 根 README 的 DFormer 语义分割性能图。
- `figs/dformerv2_table.jpg` — 根 README 的 DFormerv2 性能表图。
- `figs/geo_attention.png` — 根 README 的 geometry attention 示例图。
- `figs/manner.jpg` — 根 README 的 RGB-D 管线比较图。
- `figs/overview.jpg` — 静态上游概览图；当前直接引用待核验。
- `figs/zhihu.png` — 静态上游图片；当前直接引用待核验。
- `figs/application_new_dataset/README.md` — 上游“应用到新数据集”操作说明。
- `figs/application_new_dataset/application.jpg` — 上述说明直接引用的目录结构示意图。

## 7. 用户思路与历史草案 `liu-test-exp/`

这些路径是用户思路设计/历史研究材料，不属于当前状态或已验证实验，禁止本任务修改。

- `liu-test-exp/对抗 copy.md` — 广义对抗/传播主题的历史副本草案，具体形成关系待核验。
- `liu-test-exp/对抗.md` — 广义对抗主题研究草案。
- `liu-test-exp/广义战略信息传播与群体认知偏移的数学模型.md` — 战略信息传播与群体认知建模草案。
- `liu-test-exp/方案1/最短验证路径.md` — MUSeg 相关方案的历史最短验证路径。
- `liu-test-exp/方案1/段落主题1_论文编号.txt` — 方案写作的论文编号辅助清单。
- `liu-test-exp/方案1/研究方案设计专用提示词.md` — 研究方案设计提示词/草案材料。

## 8. 配置 `local_configs/`

配置模块由 `utils/train.py`、`utils/eval.py`、`utils/infer.py`、preflight 和 protocol import。文件名编码数据集和模型变体；同一数据集配置共享数据字段，模型文件覆盖 backbone、decoder、优化器和训练参数。

### 8.1 MUSeg

- `local_configs/MUSeg/DFormerv2_S_MVE.py` — MUSeg 基础配置：15 前景类、背景 ignore 255、BGR loader、Depth 单模态、HAM decoder、480×640 训练尺寸；默认 official split 字段仅是历史/基础值。
- `local_configs/MUSeg/DFormerv2_S_4090.py` — 当前审计运行配置入口：绑定冻结 train-dev/val-dev、split SHA/计数、环境可覆盖数据/输出/预训练路径和 4090 训练参数。
- `local_configs/MUSeg/DFormerv2_S_QuickB0.py` — 当前冻结 RGB Quick-B0 v1 配置，使用 top 3 + latest；历史运行身份不得由后续策略覆盖。
- `local_configs/MUSeg/DFormerv2_S_QuickB0_Top8.py` — 后续训练的独立 v2 top-8 配置；保持同一训练口径并最多形成 8 个 selector 候选加 latest。
- `local_configs/MUSeg/DFormerv2_S_20Epoch.py` — 旧 20-epoch 云绝对路径配置，直接指向 official train/test；当前 development 已作废，不应恢复。

### 8.2 公共基础、NYUv2 与 SUNRGBD

- `local_configs/_base_/__init__.py` — 基础配置对象 `C`、公共默认值和数据配置导入入口。
- `local_configs/_base_/datasets/NYUDepthv2.py` — NYUv2 路径、类别、格式、归一化和样本数基础配置。
- `local_configs/_base_/datasets/SUNRGBD.py` — SUNRGBD 路径、类别、格式、归一化和样本数基础配置。
- `local_configs/template/DFormer_Large.py` — 新配置的 DFormer-Large 模板/示例。
- `local_configs/NYUDepthv2/DFormer_Tiny.py` — NYUv2 DFormer Tiny 配置。
- `local_configs/NYUDepthv2/DFormer_Small.py` — NYUv2 DFormer Small 配置。
- `local_configs/NYUDepthv2/DFormer_Base.py` — NYUv2 DFormer Base 配置。
- `local_configs/NYUDepthv2/DFormer_Large.py` — NYUv2 DFormer Large 配置。
- `local_configs/NYUDepthv2/DFormerv2_S.py` — NYUv2 DFormerv2 Small 配置；根训练/评估示例使用。
- `local_configs/NYUDepthv2/DFormerv2_B.py` — NYUv2 DFormerv2 Base 配置。
- `local_configs/NYUDepthv2/DFormerv2_L.py` — NYUv2 DFormerv2 Large 配置。
- `local_configs/SUNRGBD/DFormer_Tiny.py` — SUNRGBD DFormer Tiny 配置。
- `local_configs/SUNRGBD/DFormer_Small.py` — SUNRGBD DFormer Small 配置。
- `local_configs/SUNRGBD/DFormer_Base.py` — SUNRGBD DFormer Base 配置。
- `local_configs/SUNRGBD/DFormer_Large.py` — SUNRGBD DFormer Large 配置。
- `local_configs/SUNRGBD/DFormerv2_S.py` — SUNRGBD DFormerv2 Small 配置。
- `local_configs/SUNRGBD/DFormerv2_B.py` — SUNRGBD DFormerv2 Base 配置。
- `local_configs/SUNRGBD/DFormerv2_L.py` — SUNRGBD DFormerv2 Large 配置。

## 9. 本地模型实现 `models/`

### 9.1 装配与公共网络工具

- `models/__init__.py` — 模型包导出入口。
- `models/builder.py` — `EncoderDecoder` 装配 backbone/decoder/aux head，加载预训练，前向解码并计算 masked loss；训练、评估和后评估的核心构造器。
- `models/net_utils.py` — 权重初始化、参数分组和网络辅助函数，由 builder/训练使用。

### 9.2 编码器

- `models/encoders/DFormer.py` — `LayerNorm`、`MLP`、`attention`、`Block`、`DFormer` 及 Tiny/S/B/L 工厂；统一编码 RGB-D 多尺度特征。
- `models/encoders/DFormerv2.py` — `GeoPriorGen`、GSA、`RGBD_Block`、`BasicLayer`、`dformerv2` 及 S/B/L 工厂；以深度和位置构造 geometry prior。

### 9.3 解码器

- `models/decoders/decode_head.py` — decode head 基类和输入变换，复用 `mmseg.ops.resize`。
- `models/decoders/MLPDecoder.py` — 多尺度 MLP 解码头。
- `models/decoders/LMLPDecoder.py` — 轻量 MLP 解码变体。
- `models/decoders/UPernet.py` — UPerNet 风格 PSP/FPN 解码头。
- `models/decoders/deeplabv3plus.py` — DeepLabV3+ 解码头。
- `models/decoders/fcnhead.py` — FCN 主/辅助头。
- `models/decoders/ham_head.py` — LightHamHead/Hamburger 解码实现，当前 MUSeg 配置选择此头。
- `models/decoders/nl_head.py` — non-local 解码头。
- `models/decoders/test.py` — 解码器开发测试/示例脚本；不属于正式 `tests/` 验收集，作用边界待核验。

### 9.4 损失

- `models/losses/__init__.py` — loss 包导出。
- `models/losses/accuracy.py` — 分类/像素 accuracy 计算。
- `models/losses/cross_entropy_loss.py` — 交叉熵封装。
- `models/losses/dice_loss.py` — Dice loss。
- `models/losses/focal_loss.py` — Focal loss。
- `models/losses/lovasz_loss.py` — Lovász loss。
- `models/losses/tversky_loss.py` — Tversky loss。
- `models/losses/safe_masked_loss.py` — `safe_masked_mean`；避免全 ignore batch 的 NaN，并保持可反传零损失。
- `models/losses/utils.py` — loss reduction、权重与辅助函数。

## 10. Protocol 模板 `protocols/`

- `protocols/museg-qualification-v1.template.json` — 3-epoch qualification 模板；Git、路径、预训练 SHA 等字段需物化，允许 qualification 范围的 batch/output override。
- `protocols/museg-development-long500-v2.template.json` — 500-epoch development 三 seed 模板；固定 required commit、seed、batch、schedule、评估/checkpoint 策略、split authority 和预训练身份。模板不证明 seeds 已全部运行。
- `protocols/museg-dformerv2-s-rgb-quick-b0-v1.template.json` — 当前冻结 single-seed RGB Quick-B0 v1 模板，预登记 top 3 + latest。
- `protocols/museg-dformerv2-s-rgb-quick-b0-v2-top8.template.json` — 后续运行的独立 v2 模板，预登记 top 8 + latest 且不改写 v1 身份。

## 11. MUSeg 工具 `tools/`

### 11.1 数据、split 与 protocol

- `tools/prepare_museg.py` — `Split`、`Sample`、`quantize_depth`、全量验证和原子发布；将原始 MUSeg 整理为 DFormer 输入并写 `dataset_meta.json`。
- `tools/splits/create_museg_dev_split.py` — 按采集组、矿井和多项统计目标生成 train-dev/val-dev 候选及 manifest；输出需经审计和用户门禁。
- `tools/splits/audit_museg_splits.py` — 从冻结输入重算 schema、成员、哈希、组隔离、统计和工具源身份，生成 `audit-report.json`。
- `tools/museg_protocol.schema.json` — protocol JSON Schema，供工具/测试检查字段结构。
- `tools/museg_protocol.py` — `SplitAuthority`、`ProtocolManifest`、`load_protocol`；绑定冻结 manifest/audit，严格验证协议字段、路径、phase 和 consumed splits。
- `tools/materialize_museg_protocol.py` — 把模板中的 Git、输出根、official train、预训练路径/SHA 和可选运行参数物化到机器本地 manifest。
- `tools/preflight_train.py` — `Preflight`、`AuditReport`、`audit_protocol`；检查包、Git、配置、数据、GPU、SwanLab、split、phase、预训练和输出边界。
- `tools/audit_museg_cloud_storage.py` — 对显式清理候选、保护路径、符号链接和 checkpoint 空间预算生成只读 JSON 审计；不实现删除。

### 11.2 训练、probe、qualification 与汇总

- `tools/train_museg_4090.sh` — 当前 MUSeg shell 入口：protocol preflight 后调用三 seed 编排器；本身不决定研究门禁。
- `tools/run_museg_3seed.py` — 校验 protocol 声明 seeds，逐个调用单 seed 启动器并汇总退出状态。
- `tools/run_museg_seed.py` — 构造审计训练 argv、约束 overrides/resume/输出目录、运行子进程、校验 `training_result.json`，写 command/environment/exit/run manifest。
- `tools/probe_museg_4090.sh` — qualification protocol 下测试多个 batch 的短 probe，写显存/吞吐结构化结果；执行完要求用户确认 batch。
- `tools/summarize_museg_probe.py` — 汇总单个 probe 的步数、吞吐、显存、安全阈值和异常类别。
- `tools/qualify_museg_b1.py` — 在精确 Git、干净工作区、split/pretrained/protocol 身份下裁决 B1/qualification 条件。
- `tools/summarize_museg_runs.py` — 汇总多 seed/run 结构化产物；汇总不替代原始证据。
- `tools/inspect_museg_checkpoint.py` — CLI 检查 checkpoint schema、进度和 protocol/run 身份。

### 11.3 后评估、裁决与 MVE

- `tools/evaluate_museg_checkpoint.py` — `MUSegPostEvalDataset`、`load_model`、`sliding_logits`、confusion/metrics；只用显式 val split，支持三种几何并输出结构化 JSON。
- `tools/adjudicate_museg_seed_acceptance.py` — 独立重算报告引用证据的哈希与完成条件，写新裁决，不改写原始 acceptance/failed/result。
- `tools/mve/a2_prepare_masks.py` — 从 Depth16/Depth/Label 统计自然无效深度并生成 A2 mask/清单。
- `tools/mve/a2_run_sensitivity.py` — 对指定验证数据运行 depth corruption 敏感性评估。
- `tools/mve/a2_evaluate_results.py` — 汇总 A2 指标和门槛；自然证据不足时限制结论强度。
- `tools/mve/run_museg_20epoch_screen.sh` — 已作废的历史 20-epoch 筛查入口；不得作为当前 development workload。

### 11.4 Canvas 发布

- `tools/publish-canvas.cmd` — Windows 双击包装器，调用 PowerShell 发布脚本。
- `tools/publish-canvas.ps1` — 校验版本、唯一性和内容冲突后复制 Canvas 到 Cursor 受管预览目录；不删除已有版本。

## 12. 测试 `tests/`

每个文件是 pytest 测试模块；synthetic/mock 通过只证明被测边界，不证明真实 GPU 训练或数据指标。

- `tests/test_a2_masks.py` — A2 mask 统计、生成和异常输入。
- `tests/test_masked_loss.py` — 全 ignore/部分有效标签下安全 masked loss 回归。
- `tests/test_museg_checkpoint_posteval.py` — 后评估 split 拒绝规则、几何和不依赖单独预训练的 strict checkpoint 加载。
- `tests/test_museg_cloud_storage_audit.py` — 只读数据盘审计的路径越界、保护重叠、符号链接和无删除行为测试。
- `tests/test_museg_dev_split.py` — split 生成、冻结 manifest/audit、成员和哈希回归。
- `tests/test_museg_protocol_materialization.py` — protocol 模板物化、字段替换、哈希和错误路径。
- `tests/test_museg_seed_acceptance_adjudication.py` — 独立 acceptance v2 哈希与裁决回归。
- `tests/test_museg_stage03.py` — preflight、seed 编排、run kind、tracking 和运行证据的 Stage-03 测试。
- `tests/test_museg_stage04_preconditions.py` — Stage-04 Git/protocol/split/pretrained 前置条件。
- `tests/test_museg_stage04_probe.py` — probe telemetry、阈值、异常和汇总测试。
- `tests/test_training_checkpoint.py` — checkpoint schema、原子保存、身份检查、恢复和目录连续性。
- `tests/test_training_ops.py` — AMP/optimizer step、训练操作和计数语义测试。

## 13. 通用运行代码 `utils/`

### 13.1 数据、训练、评估与推理

- `utils/__init__.py` — utils 包入口。
- `utils/dataloader/RGBXDataset.py` — `get_path` 与 `RGBXDataset`；读取同名 RGB/Depth/Label、选择 BGR/RGB、标签减一并返回张量和样本身份。
- `utils/dataloader/dataloader.py` — `TrainPre`、`ValPre`、`get_train_loader`、`get_val_loader`；实现增强、归一化、固定 epoch 长度和分布式 sampler。
- `utils/train.py` — 主训练程序；CLI、protocol/split/Git/resume 验证、模型/优化器/scheduler/AMP、validation、checkpoint、遥测和 `training_result.json`。
- `utils/eval.py` — 上游独立评估 CLI；从配置和 `continue_fpath` 构造模型并计算指标。
- `utils/infer.py` — 上游推理/可视化 CLI；加载 checkpoint 并保存预测。
- `utils/val_mm.py` — `evaluate`、滑窗/多尺度评估和 confusion/指标，是在线 validation 的主要实现。
- `utils/transforms.py` — RGB-X resize、crop、mirror、normalize 等图像变换。

### 13.2 Checkpoint、engine 与跟踪

- `utils/training_checkpoint.py` — `CheckpointProtocol`、checkpoint 创建/检查/加载/恢复、目录检查和原子写入；绑定运行身份和 RNG 状态。
- `utils/experiment_tracker.py` — SwanLab/TensorBoard 运行配置、指标和结束状态适配层。
- `utils/engine/__init__.py` — engine 包导出。
- `utils/engine/engine.py` — 分布式上下文、state 注册、checkpoint 保存和日志链接。
- `utils/engine/dist_test.py` — engine 分布式测试/示例入口，实际验收角色待核验。
- `utils/engine/evaluator.py` — engine 评估器抽象/辅助类。
- `utils/engine/logger.py` — 日志格式、文件与控制台 logger。

### 13.3 指标、优化与通用工具

- `utils/init_func.py` — 权重初始化和参数分组辅助函数。
- `utils/load_utils.py` — state dict/checkpoint 加载兼容工具。
- `utils/loss_opr.py` — 历史/通用 loss 操作实现；当前 builder 主要使用 `models/losses/`。
- `utils/lr_policy.py` — `WarmUpPolyLR` 等学习率策略。
- `utils/metric.py` — confusion matrix 与旧指标工具。
- `utils/metrics_new.py` — 新指标计算辅助实现。
- `utils/pyt_utils.py` — 文件、链接、分布式 reduce 和通用 PyTorch 工具。
- `utils/visualize.py` — 分割结果着色/保存辅助函数。
- `utils/nyucmap.npy` — NYUv2 颜色映射 NumPy 二进制资源，由可视化代码消费，不手工编辑。

### 13.4 性能与演示

- `utils/benchmark.py` — 参数量/FLOPs 基准 CLI，结果依赖配置和环境。
- `utils/latency.py` — 延迟基准；只能在同设备/条件下比较。
- `utils/demo_geometry_prior.py` — DFormerv2 geometry prior 可视化/演示代码。

## 14. 内嵌 MMSegmentation 兼容树 `mmseg/`

该树共有 **949** 个跟踪文件：`.mim/` 830、`models/` 86、`core/` 18、`utils/` 6、`apis/` 4、`ops/` 3 和包根 2。它主要是上游/兼容代码，不等同于本仓库 `models/` 主实现。仓库外部的主动直接依赖集中在 `models/decoders/decode_head.py`、`ham_head.py`、`nl_head.py` 对 `mmseg.ops.resize` 的导入；其他动态调用待核验。

### 14.1 运行时代码

运行时文件共享说明：包/API/core 提供训练、测试、评估、hook、优化器和像素采样基础设施；`models/` 提供标准分割 backbone/head/loss/neck/segmentor；`ops/` 提供 encoding 与 resize wrapper；`utils/` 提供环境、日志和分布式辅助。每个路径按其末级模块名承担对应实现，复杂行为应查看公开类/函数，不能仅凭存在推断本地正在使用。

- `mmseg/__init__.py`
- `mmseg/version.py`
- `mmseg/apis/__init__.py`
- `mmseg/apis/inference.py`
- `mmseg/apis/test.py`
- `mmseg/apis/train.py`
- `mmseg/core/__init__.py`
- `mmseg/core/builder.py`
- `mmseg/core/evaluation/__init__.py`
- `mmseg/core/evaluation/class_names.py`
- `mmseg/core/evaluation/eval_hooks.py`
- `mmseg/core/evaluation/metrics.py`
- `mmseg/core/hook/__init__.py`
- `mmseg/core/hook/wandblogger_hook.py`
- `mmseg/core/optimizers/__init__.py`
- `mmseg/core/optimizers/layer_decay_optimizer_constructor.py`
- `mmseg/core/seg/__init__.py`
- `mmseg/core/seg/builder.py`
- `mmseg/core/seg/sampler/__init__.py`
- `mmseg/core/seg/sampler/base_pixel_sampler.py`
- `mmseg/core/seg/sampler/ohem_pixel_sampler.py`
- `mmseg/core/utils/__init__.py`
- `mmseg/core/utils/dist_util.py`
- `mmseg/core/utils/misc.py`
- `mmseg/models/__init__.py`
- `mmseg/models/builder.py`
- `mmseg/models/backbones/__init__.py`
- `mmseg/models/backbones/beit.py`
- `mmseg/models/backbones/bisenetv1.py`
- `mmseg/models/backbones/bisenetv2.py`
- `mmseg/models/backbones/cgnet.py`
- `mmseg/models/backbones/erfnet.py`
- `mmseg/models/backbones/fast_scnn.py`
- `mmseg/models/backbones/hrnet.py`
- `mmseg/models/backbones/icnet.py`
- `mmseg/models/backbones/mae.py`
- `mmseg/models/backbones/mit.py`
- `mmseg/models/backbones/mobilenet_v2.py`
- `mmseg/models/backbones/mobilenet_v3.py`
- `mmseg/models/backbones/resnest.py`
- `mmseg/models/backbones/resnet.py`
- `mmseg/models/backbones/resnext.py`
- `mmseg/models/backbones/scnet.py`
- `mmseg/models/backbones/stdc.py`
- `mmseg/models/backbones/swin.py`
- `mmseg/models/backbones/timm_backbone.py`
- `mmseg/models/backbones/twins.py`
- `mmseg/models/backbones/unet.py`
- `mmseg/models/backbones/vit.py`
- `mmseg/models/decode_heads/__init__.py`
- `mmseg/models/decode_heads/ann_head.py`
- `mmseg/models/decode_heads/apc_head.py`
- `mmseg/models/decode_heads/aspp_head.py`
- `mmseg/models/decode_heads/cascade_decode_head.py`
- `mmseg/models/decode_heads/cc_head.py`
- `mmseg/models/decode_heads/da_head.py`
- `mmseg/models/decode_heads/decode_head.py`
- `mmseg/models/decode_heads/dm_head.py`
- `mmseg/models/decode_heads/dnl_head.py`
- `mmseg/models/decode_heads/dpt_head.py`
- `mmseg/models/decode_heads/ema_head.py`
- `mmseg/models/decode_heads/enc_head.py`
- `mmseg/models/decode_heads/fcn_head.py`
- `mmseg/models/decode_heads/fpn_head.py`
- `mmseg/models/decode_heads/gc_head.py`
- `mmseg/models/decode_heads/isa_head.py`
- `mmseg/models/decode_heads/knet_head.py`
- `mmseg/models/decode_heads/lraspp_head.py`
- `mmseg/models/decode_heads/mix_conv_head.py`
- `mmseg/models/decode_heads/nl_head.py`
- `mmseg/models/decode_heads/ocr_head.py`
- `mmseg/models/decode_heads/point_head.py`
- `mmseg/models/decode_heads/psa_head.py`
- `mmseg/models/decode_heads/psp_head.py`
- `mmseg/models/decode_heads/segformer_head.py`
- `mmseg/models/decode_heads/segmenter_mask_head.py`
- `mmseg/models/decode_heads/sep_aspp_head.py`
- `mmseg/models/decode_heads/sep_fcn_head.py`
- `mmseg/models/decode_heads/setr_mla_head.py`
- `mmseg/models/decode_heads/setr_up_head.py`
- `mmseg/models/decode_heads/stdc_head.py`
- `mmseg/models/decode_heads/uper_head.py`
- `mmseg/models/losses/__init__.py`
- `mmseg/models/losses/accuracy.py`
- `mmseg/models/losses/cross_entropy_loss.py`
- `mmseg/models/losses/dice_loss.py`
- `mmseg/models/losses/focal_loss.py`
- `mmseg/models/losses/lovasz_loss.py`
- `mmseg/models/losses/tversky_loss.py`
- `mmseg/models/losses/utils.py`
- `mmseg/models/necks/__init__.py`
- `mmseg/models/necks/featurepyramid.py`
- `mmseg/models/necks/fpn.py`
- `mmseg/models/necks/ic_neck.py`
- `mmseg/models/necks/jpu.py`
- `mmseg/models/necks/mla_neck.py`
- `mmseg/models/necks/multilevel_neck.py`
- `mmseg/models/segmentors/__init__.py`
- `mmseg/models/segmentors/base.py`
- `mmseg/models/segmentors/cascade_encoder_decoder.py`
- `mmseg/models/segmentors/encoder_decoder.py`
- `mmseg/models/utils/__init__.py`
- `mmseg/models/utils/embed.py`
- `mmseg/models/utils/inverted_residual.py`
- `mmseg/models/utils/make_divisible.py`
- `mmseg/models/utils/res_layer.py`
- `mmseg/models/utils/se_layer.py`
- `mmseg/models/utils/self_attention_block.py`
- `mmseg/models/utils/shape_convert.py`
- `mmseg/models/utils/up_conv_block.py`
- `mmseg/ops/__init__.py`
- `mmseg/ops/encoding.py`
- `mmseg/ops/wrappers.py` — `resize` 是本地多个解码器的直接兼容依赖。
- `mmseg/utils/__init__.py`
- `mmseg/utils/collect_env.py`
- `mmseg/utils/logger.py`
- `mmseg/utils/misc.py`
- `mmseg/utils/set_env.py`
- `mmseg/utils/util_distribution.py`

### 14.2 `.mim` 模型索引、工具与配置库

`.mim/model-index.yml` 组织上游模型元数据；`.mim/tools/` 是训练、测试、分析、部署和数据转换 CLI；`.mim/configs/` 是 46 个模型族的配置/README/YAML 元数据。配置路径中的族名、backbone、数据集、crop、schedule 和 runtime 变体构成其精简用途说明；它们不属于 MUSeg 当前 protocol，除非出现直接引用证据。

- `mmseg/.mim/model-index.yml` — 上游模型索引元数据。
- `mmseg/.mim/tools/analyze_logs.py`
- `mmseg/.mim/tools/benchmark.py`
- `mmseg/.mim/tools/browse_dataset.py`
- `mmseg/.mim/tools/confusion_matrix.py`
- `mmseg/.mim/tools/convert_datasets/chase_db1.py`
- `mmseg/.mim/tools/convert_datasets/cityscapes.py`
- `mmseg/.mim/tools/convert_datasets/coco_stuff10k.py`
- `mmseg/.mim/tools/convert_datasets/coco_stuff164k.py`
- `mmseg/.mim/tools/convert_datasets/drive.py`
- `mmseg/.mim/tools/convert_datasets/hrf.py`
- `mmseg/.mim/tools/convert_datasets/isaid.py`
- `mmseg/.mim/tools/convert_datasets/loveda.py`
- `mmseg/.mim/tools/convert_datasets/pascal_context.py`
- `mmseg/.mim/tools/convert_datasets/potsdam.py`
- `mmseg/.mim/tools/convert_datasets/stare.py`
- `mmseg/.mim/tools/convert_datasets/vaihingen.py`
- `mmseg/.mim/tools/convert_datasets/voc_aug.py`
- `mmseg/.mim/tools/deploy_test.py`
- `mmseg/.mim/tools/dist_test.sh`
- `mmseg/.mim/tools/dist_train.sh`
- `mmseg/.mim/tools/get_flops.py`
- `mmseg/.mim/tools/model_converters/beit2mmseg.py`
- `mmseg/.mim/tools/model_converters/mit2mmseg.py`
- `mmseg/.mim/tools/model_converters/stdc2mmseg.py`
- `mmseg/.mim/tools/model_converters/swin2mmseg.py`
- `mmseg/.mim/tools/model_converters/twins2mmseg.py`
- `mmseg/.mim/tools/model_converters/vit2mmseg.py`
- `mmseg/.mim/tools/model_converters/vitjax2mmseg.py`
- `mmseg/.mim/tools/model_ensemble.py`
- `mmseg/.mim/tools/onnx2tensorrt.py`
- `mmseg/.mim/tools/print_config.py`
- `mmseg/.mim/tools/publish_model.py`
- `mmseg/.mim/tools/pytorch2onnx.py`
- `mmseg/.mim/tools/pytorch2torchscript.py`
- `mmseg/.mim/tools/slurm_test.sh`
- `mmseg/.mim/tools/slurm_train.sh`
- `mmseg/.mim/tools/test.py`
- `mmseg/.mim/tools/torchserve/mmseg2torchserve.py`
- `mmseg/.mim/tools/torchserve/mmseg_handler.py`
- `mmseg/.mim/tools/torchserve/test_torchserve.py`
- `mmseg/.mim/tools/train.py`

以上工具按文件名分别执行日志分析、基准、数据浏览/转换、部署、分布式/Slurm 训练测试、FLOPs、checkpoint 转换、模型集成、ONNX/TorchScript/TensorRT/TorchServe、配置打印和发布；它们是上游 CLI，不是 MUSeg protocol 入口。

#### Base、ANN、APCNet、BEiT、BiSeNet、CCNet、CGNet、ConvNeXt

以下每行都是一个独立上游配置/元数据路径；`_base_` 提供模型与 schedule 基片，其余目录的 README/YAML 记录模型族元数据，Python 文件名记录模型、backbone、输入、schedule 和数据集变体。

```text
mmseg/.mim/configs/_base_/default_runtime.py
mmseg/.mim/configs/_base_/models/ann_r50-d8.py
mmseg/.mim/configs/_base_/models/apcnet_r50-d8.py
mmseg/.mim/configs/_base_/models/bisenetv1_r18-d32.py
mmseg/.mim/configs/_base_/models/bisenetv2.py
mmseg/.mim/configs/_base_/models/ccnet_r50-d8.py
mmseg/.mim/configs/_base_/models/cgnet.py
mmseg/.mim/configs/_base_/models/danet_r50-d8.py
mmseg/.mim/configs/_base_/models/deeplabv3_r50-d8.py
mmseg/.mim/configs/_base_/models/deeplabv3_unet_s5-d16.py
mmseg/.mim/configs/_base_/models/deeplabv3plus_r50-d8.py
mmseg/.mim/configs/_base_/models/dmnet_r50-d8.py
mmseg/.mim/configs/_base_/models/dnl_r50-d8.py
mmseg/.mim/configs/_base_/models/dpt_vit-b16.py
mmseg/.mim/configs/_base_/models/emanet_r50-d8.py
mmseg/.mim/configs/_base_/models/encnet_r50-d8.py
mmseg/.mim/configs/_base_/models/erfnet_fcn.py
mmseg/.mim/configs/_base_/models/fast_scnn.py
mmseg/.mim/configs/_base_/models/fastfcn_r50-d32_jpu_psp.py
mmseg/.mim/configs/_base_/models/fcn_hr18.py
mmseg/.mim/configs/_base_/models/fcn_r50-d8.py
mmseg/.mim/configs/_base_/models/fcn_unet_s5-d16.py
mmseg/.mim/configs/_base_/models/fpn_poolformer_s12.py
mmseg/.mim/configs/_base_/models/fpn_r50.py
mmseg/.mim/configs/_base_/models/gcnet_r50-d8.py
mmseg/.mim/configs/_base_/models/icnet_r50-d8.py
mmseg/.mim/configs/_base_/models/isanet_r50-d8.py
mmseg/.mim/configs/_base_/models/lraspp_m-v3-d8.py
mmseg/.mim/configs/_base_/models/nonlocal_r50-d8.py
mmseg/.mim/configs/_base_/models/ocrnet_hr18.py
mmseg/.mim/configs/_base_/models/ocrnet_r50-d8.py
mmseg/.mim/configs/_base_/models/pointrend_r50.py
mmseg/.mim/configs/_base_/models/psanet_r50-d8.py
mmseg/.mim/configs/_base_/models/pspnet_r50-d8.py
mmseg/.mim/configs/_base_/models/pspnet_unet_s5-d16.py
mmseg/.mim/configs/_base_/models/segformer_mit-b0.py
mmseg/.mim/configs/_base_/models/segmenter_vit-b16_mask.py
mmseg/.mim/configs/_base_/models/setr_mla.py
mmseg/.mim/configs/_base_/models/setr_naive.py
mmseg/.mim/configs/_base_/models/setr_pup.py
mmseg/.mim/configs/_base_/models/stdc.py
mmseg/.mim/configs/_base_/models/twins_pcpvt-s_fpn.py
mmseg/.mim/configs/_base_/models/twins_pcpvt-s_upernet.py
mmseg/.mim/configs/_base_/models/upernet_beit.py
mmseg/.mim/configs/_base_/models/upernet_convnext.py
mmseg/.mim/configs/_base_/models/upernet_mae.py
mmseg/.mim/configs/_base_/models/upernet_r50.py
mmseg/.mim/configs/_base_/models/upernet_swin.py
mmseg/.mim/configs/_base_/models/upernet_vit-b16_ln_mln.py
mmseg/.mim/configs/_base_/schedules/schedule_160k.py
mmseg/.mim/configs/_base_/schedules/schedule_20k.py
mmseg/.mim/configs/_base_/schedules/schedule_320k.py
mmseg/.mim/configs/_base_/schedules/schedule_40k.py
mmseg/.mim/configs/_base_/schedules/schedule_80k.py
mmseg/.mim/configs/ann/README.md
mmseg/.mim/configs/ann/ann.yml
mmseg/.mim/configs/ann/ann_r101-d8_512x1024_40k_cityscapes.py
mmseg/.mim/configs/ann/ann_r101-d8_512x1024_80k_cityscapes.py
mmseg/.mim/configs/ann/ann_r101-d8_512x512_160k_ade20k.py
mmseg/.mim/configs/ann/ann_r101-d8_512x512_20k_voc12aug.py
mmseg/.mim/configs/ann/ann_r101-d8_512x512_40k_voc12aug.py
mmseg/.mim/configs/ann/ann_r101-d8_512x512_80k_ade20k.py
mmseg/.mim/configs/ann/ann_r101-d8_769x769_40k_cityscapes.py
mmseg/.mim/configs/ann/ann_r101-d8_769x769_80k_cityscapes.py
mmseg/.mim/configs/ann/ann_r50-d8_512x1024_40k_cityscapes.py
mmseg/.mim/configs/ann/ann_r50-d8_512x1024_80k_cityscapes.py
mmseg/.mim/configs/ann/ann_r50-d8_512x512_160k_ade20k.py
mmseg/.mim/configs/ann/ann_r50-d8_512x512_20k_voc12aug.py
mmseg/.mim/configs/ann/ann_r50-d8_512x512_40k_voc12aug.py
mmseg/.mim/configs/ann/ann_r50-d8_512x512_80k_ade20k.py
mmseg/.mim/configs/ann/ann_r50-d8_769x769_40k_cityscapes.py
mmseg/.mim/configs/ann/ann_r50-d8_769x769_80k_cityscapes.py
mmseg/.mim/configs/apcnet/README.md
mmseg/.mim/configs/apcnet/apcnet.yml
mmseg/.mim/configs/apcnet/apcnet_r101-d8_512x1024_40k_cityscapes.py
mmseg/.mim/configs/apcnet/apcnet_r101-d8_512x1024_80k_cityscapes.py
mmseg/.mim/configs/apcnet/apcnet_r101-d8_512x512_160k_ade20k.py
mmseg/.mim/configs/apcnet/apcnet_r101-d8_512x512_80k_ade20k.py
mmseg/.mim/configs/apcnet/apcnet_r101-d8_769x769_40k_cityscapes.py
mmseg/.mim/configs/apcnet/apcnet_r101-d8_769x769_80k_cityscapes.py
mmseg/.mim/configs/apcnet/apcnet_r50-d8_512x1024_40k_cityscapes.py
mmseg/.mim/configs/apcnet/apcnet_r50-d8_512x1024_80k_cityscapes.py
mmseg/.mim/configs/apcnet/apcnet_r50-d8_512x512_160k_ade20k.py
mmseg/.mim/configs/apcnet/apcnet_r50-d8_512x512_80k_ade20k.py
mmseg/.mim/configs/apcnet/apcnet_r50-d8_769x769_40k_cityscapes.py
mmseg/.mim/configs/apcnet/apcnet_r50-d8_769x769_80k_cityscapes.py
mmseg/.mim/configs/beit/README.md
mmseg/.mim/configs/beit/beit.yml
mmseg/.mim/configs/beit/upernet_beit-base_640x640_160k_ade20k_ms.py
mmseg/.mim/configs/beit/upernet_beit-base_8x2_640x640_160k_ade20k.py
mmseg/.mim/configs/beit/upernet_beit-large_fp16_640x640_160k_ade20k_ms.py
mmseg/.mim/configs/beit/upernet_beit-large_fp16_8x1_640x640_160k_ade20k.py
mmseg/.mim/configs/bisenetv1/README.md
mmseg/.mim/configs/bisenetv1/bisenetv1.yml
mmseg/.mim/configs/bisenetv1/bisenetv1_r101-d32_in1k-pre_lr5e-3_4x4_512x512_160k_coco-stuff164k.py
mmseg/.mim/configs/bisenetv1/bisenetv1_r101-d32_lr5e-3_4x4_512x512_160k_coco-stuff164k.py
mmseg/.mim/configs/bisenetv1/bisenetv1_r18-d32_4x4_1024x1024_160k_cityscapes.py
mmseg/.mim/configs/bisenetv1/bisenetv1_r18-d32_in1k-pre_4x4_1024x1024_160k_cityscapes.py
mmseg/.mim/configs/bisenetv1/bisenetv1_r18-d32_in1k-pre_4x8_1024x1024_160k_cityscapes.py
mmseg/.mim/configs/bisenetv1/bisenetv1_r18-d32_in1k-pre_lr5e-3_4x4_512x512_160k_coco-stuff164k.py
mmseg/.mim/configs/bisenetv1/bisenetv1_r18-d32_lr5e-3_4x4_512x512_160k_coco-stuff164k.py
mmseg/.mim/configs/bisenetv1/bisenetv1_r50-d32_4x4_1024x1024_160k_cityscapes.py
mmseg/.mim/configs/bisenetv1/bisenetv1_r50-d32_in1k-pre_4x4_1024x1024_160k_cityscapes.py
mmseg/.mim/configs/bisenetv1/bisenetv1_r50-d32_in1k-pre_lr5e-3_4x4_512x512_160k_coco-stuff164k.py
mmseg/.mim/configs/bisenetv1/bisenetv1_r50-d32_lr5e-3_4x4_512x512_160k_coco-stuff164k.py
mmseg/.mim/configs/bisenetv2/README.md
mmseg/.mim/configs/bisenetv2/bisenetv2.yml
mmseg/.mim/configs/bisenetv2/bisenetv2_fcn_4x4_1024x1024_160k_cityscapes.py
mmseg/.mim/configs/bisenetv2/bisenetv2_fcn_4x8_1024x1024_160k_cityscapes.py
mmseg/.mim/configs/bisenetv2/bisenetv2_fcn_fp16_4x4_1024x1024_160k_cityscapes.py
mmseg/.mim/configs/bisenetv2/bisenetv2_fcn_ohem_4x4_1024x1024_160k_cityscapes.py
mmseg/.mim/configs/ccnet/README.md
mmseg/.mim/configs/ccnet/ccnet.yml
mmseg/.mim/configs/ccnet/ccnet_r101-d8_512x1024_40k_cityscapes.py
mmseg/.mim/configs/ccnet/ccnet_r101-d8_512x1024_80k_cityscapes.py
mmseg/.mim/configs/ccnet/ccnet_r101-d8_512x512_160k_ade20k.py
mmseg/.mim/configs/ccnet/ccnet_r101-d8_512x512_20k_voc12aug.py
mmseg/.mim/configs/ccnet/ccnet_r101-d8_512x512_40k_voc12aug.py
mmseg/.mim/configs/ccnet/ccnet_r101-d8_512x512_80k_ade20k.py
mmseg/.mim/configs/ccnet/ccnet_r101-d8_769x769_40k_cityscapes.py
mmseg/.mim/configs/ccnet/ccnet_r101-d8_769x769_80k_cityscapes.py
mmseg/.mim/configs/ccnet/ccnet_r50-d8_512x1024_40k_cityscapes.py
mmseg/.mim/configs/ccnet/ccnet_r50-d8_512x1024_80k_cityscapes.py
mmseg/.mim/configs/ccnet/ccnet_r50-d8_512x512_160k_ade20k.py
mmseg/.mim/configs/ccnet/ccnet_r50-d8_512x512_20k_voc12aug.py
mmseg/.mim/configs/ccnet/ccnet_r50-d8_512x512_40k_voc12aug.py
mmseg/.mim/configs/ccnet/ccnet_r50-d8_512x512_80k_ade20k.py
mmseg/.mim/configs/ccnet/ccnet_r50-d8_769x769_40k_cityscapes.py
mmseg/.mim/configs/ccnet/ccnet_r50-d8_769x769_80k_cityscapes.py
mmseg/.mim/configs/cgnet/README.md
mmseg/.mim/configs/cgnet/cgnet.yml
mmseg/.mim/configs/cgnet/cgnet_512x1024_60k_cityscapes.py
mmseg/.mim/configs/cgnet/cgnet_680x680_60k_cityscapes.py
mmseg/.mim/configs/convnext/README.md
mmseg/.mim/configs/convnext/convnext.yml
mmseg/.mim/configs/convnext/upernet_convnext_base_fp16_512x512_160k_ade20k.py
mmseg/.mim/configs/convnext/upernet_convnext_base_fp16_640x640_160k_ade20k.py
mmseg/.mim/configs/convnext/upernet_convnext_large_fp16_640x640_160k_ade20k.py
mmseg/.mim/configs/convnext/upernet_convnext_small_fp16_512x512_160k_ade20k.py
mmseg/.mim/configs/convnext/upernet_convnext_tiny_fp16_512x512_160k_ade20k.py
mmseg/.mim/configs/convnext/upernet_convnext_xlarge_fp16_640x640_160k_ade20k.py
```

#### DANet 与 DeepLab

以下路径共享上游模型族说明；每行文件名给出 decoder/backbone、输入尺度、schedule、数据集、fp16 或增强变体。

```text
mmseg/.mim/configs/danet/README.md
mmseg/.mim/configs/danet/danet.yml
mmseg/.mim/configs/danet/danet_r101-d8_512x1024_40k_cityscapes.py
mmseg/.mim/configs/danet/danet_r101-d8_512x1024_80k_cityscapes.py
mmseg/.mim/configs/danet/danet_r101-d8_512x512_160k_ade20k.py
mmseg/.mim/configs/danet/danet_r101-d8_512x512_20k_voc12aug.py
mmseg/.mim/configs/danet/danet_r101-d8_512x512_40k_voc12aug.py
mmseg/.mim/configs/danet/danet_r101-d8_512x512_80k_ade20k.py
mmseg/.mim/configs/danet/danet_r101-d8_769x769_40k_cityscapes.py
mmseg/.mim/configs/danet/danet_r101-d8_769x769_80k_cityscapes.py
mmseg/.mim/configs/danet/danet_r50-d8_512x1024_40k_cityscapes.py
mmseg/.mim/configs/danet/danet_r50-d8_512x1024_80k_cityscapes.py
mmseg/.mim/configs/danet/danet_r50-d8_512x512_160k_ade20k.py
mmseg/.mim/configs/danet/danet_r50-d8_512x512_20k_voc12aug.py
mmseg/.mim/configs/danet/danet_r50-d8_512x512_40k_voc12aug.py
mmseg/.mim/configs/danet/danet_r50-d8_512x512_80k_ade20k.py
mmseg/.mim/configs/danet/danet_r50-d8_769x769_40k_cityscapes.py
mmseg/.mim/configs/danet/danet_r50-d8_769x769_80k_cityscapes.py
mmseg/.mim/configs/deeplabv3/README.md
mmseg/.mim/configs/deeplabv3/deeplabv3.yml
mmseg/.mim/configs/deeplabv3/deeplabv3_r101-d16-mg124_512x1024_40k_cityscapes.py
mmseg/.mim/configs/deeplabv3/deeplabv3_r101-d16-mg124_512x1024_80k_cityscapes.py
mmseg/.mim/configs/deeplabv3/deeplabv3_r101-d8_480x480_40k_pascal_context.py
mmseg/.mim/configs/deeplabv3/deeplabv3_r101-d8_480x480_40k_pascal_context_59.py
mmseg/.mim/configs/deeplabv3/deeplabv3_r101-d8_480x480_80k_pascal_context.py
mmseg/.mim/configs/deeplabv3/deeplabv3_r101-d8_480x480_80k_pascal_context_59.py
mmseg/.mim/configs/deeplabv3/deeplabv3_r101-d8_512x1024_40k_cityscapes.py
mmseg/.mim/configs/deeplabv3/deeplabv3_r101-d8_512x1024_80k_cityscapes.py
mmseg/.mim/configs/deeplabv3/deeplabv3_r101-d8_512x512_160k_ade20k.py
mmseg/.mim/configs/deeplabv3/deeplabv3_r101-d8_512x512_20k_voc12aug.py
mmseg/.mim/configs/deeplabv3/deeplabv3_r101-d8_512x512_40k_voc12aug.py
mmseg/.mim/configs/deeplabv3/deeplabv3_r101-d8_512x512_4x4_160k_coco-stuff164k.py
mmseg/.mim/configs/deeplabv3/deeplabv3_r101-d8_512x512_4x4_20k_coco-stuff10k.py
mmseg/.mim/configs/deeplabv3/deeplabv3_r101-d8_512x512_4x4_320k_coco-stuff164k.py
mmseg/.mim/configs/deeplabv3/deeplabv3_r101-d8_512x512_4x4_40k_coco-stuff10k.py
mmseg/.mim/configs/deeplabv3/deeplabv3_r101-d8_512x512_4x4_80k_coco-stuff164k.py
mmseg/.mim/configs/deeplabv3/deeplabv3_r101-d8_512x512_80k_ade20k.py
mmseg/.mim/configs/deeplabv3/deeplabv3_r101-d8_769x769_40k_cityscapes.py
mmseg/.mim/configs/deeplabv3/deeplabv3_r101-d8_769x769_80k_cityscapes.py
mmseg/.mim/configs/deeplabv3/deeplabv3_r101-d8_fp16_512x1024_80k_cityscapes.py
mmseg/.mim/configs/deeplabv3/deeplabv3_r101b-d8_512x1024_80k_cityscapes.py
mmseg/.mim/configs/deeplabv3/deeplabv3_r101b-d8_769x769_80k_cityscapes.py
mmseg/.mim/configs/deeplabv3/deeplabv3_r18-d8_512x1024_80k_cityscapes.py
mmseg/.mim/configs/deeplabv3/deeplabv3_r18-d8_769x769_80k_cityscapes.py
mmseg/.mim/configs/deeplabv3/deeplabv3_r18b-d8_512x1024_80k_cityscapes.py
mmseg/.mim/configs/deeplabv3/deeplabv3_r18b-d8_769x769_80k_cityscapes.py
mmseg/.mim/configs/deeplabv3/deeplabv3_r50-d8_480x480_40k_pascal_context.py
mmseg/.mim/configs/deeplabv3/deeplabv3_r50-d8_480x480_40k_pascal_context_59.py
mmseg/.mim/configs/deeplabv3/deeplabv3_r50-d8_480x480_80k_pascal_context.py
mmseg/.mim/configs/deeplabv3/deeplabv3_r50-d8_480x480_80k_pascal_context_59.py
mmseg/.mim/configs/deeplabv3/deeplabv3_r50-d8_512x1024_40k_cityscapes.py
mmseg/.mim/configs/deeplabv3/deeplabv3_r50-d8_512x1024_80k_cityscapes.py
mmseg/.mim/configs/deeplabv3/deeplabv3_r50-d8_512x512_160k_ade20k.py
mmseg/.mim/configs/deeplabv3/deeplabv3_r50-d8_512x512_20k_voc12aug.py
mmseg/.mim/configs/deeplabv3/deeplabv3_r50-d8_512x512_40k_voc12aug.py
mmseg/.mim/configs/deeplabv3/deeplabv3_r50-d8_512x512_4x4_160k_coco-stuff164k.py
mmseg/.mim/configs/deeplabv3/deeplabv3_r50-d8_512x512_4x4_20k_coco-stuff10k.py
mmseg/.mim/configs/deeplabv3/deeplabv3_r50-d8_512x512_4x4_320k_coco-stuff164k.py
mmseg/.mim/configs/deeplabv3/deeplabv3_r50-d8_512x512_4x4_40k_coco-stuff10k.py
mmseg/.mim/configs/deeplabv3/deeplabv3_r50-d8_512x512_4x4_80k_coco-stuff164k.py
mmseg/.mim/configs/deeplabv3/deeplabv3_r50-d8_512x512_80k_ade20k.py
mmseg/.mim/configs/deeplabv3/deeplabv3_r50-d8_769x769_40k_cityscapes.py
mmseg/.mim/configs/deeplabv3/deeplabv3_r50-d8_769x769_80k_cityscapes.py
mmseg/.mim/configs/deeplabv3/deeplabv3_r50b-d8_512x1024_80k_cityscapes.py
mmseg/.mim/configs/deeplabv3/deeplabv3_r50b-d8_769x769_80k_cityscapes.py
mmseg/.mim/configs/deeplabv3plus/README.md
mmseg/.mim/configs/deeplabv3plus/deeplabv3plus.yml
mmseg/.mim/configs/deeplabv3plus/deeplabv3plus_r101-d16-mg124_512x1024_40k_cityscapes.py
mmseg/.mim/configs/deeplabv3plus/deeplabv3plus_r101-d16-mg124_512x1024_80k_cityscapes.py
mmseg/.mim/configs/deeplabv3plus/deeplabv3plus_r101-d8_480x480_40k_pascal_context.py
mmseg/.mim/configs/deeplabv3plus/deeplabv3plus_r101-d8_480x480_40k_pascal_context_59.py
mmseg/.mim/configs/deeplabv3plus/deeplabv3plus_r101-d8_480x480_80k_pascal_context.py
mmseg/.mim/configs/deeplabv3plus/deeplabv3plus_r101-d8_480x480_80k_pascal_context_59.py
mmseg/.mim/configs/deeplabv3plus/deeplabv3plus_r101-d8_4x4_512x512_80k_vaihingen.py
mmseg/.mim/configs/deeplabv3plus/deeplabv3plus_r101-d8_512x1024_40k_cityscapes.py
mmseg/.mim/configs/deeplabv3plus/deeplabv3plus_r101-d8_512x1024_80k_cityscapes.py
mmseg/.mim/configs/deeplabv3plus/deeplabv3plus_r101-d8_512x512_160k_ade20k.py
mmseg/.mim/configs/deeplabv3plus/deeplabv3plus_r101-d8_512x512_20k_voc12aug.py
mmseg/.mim/configs/deeplabv3plus/deeplabv3plus_r101-d8_512x512_40k_voc12aug.py
mmseg/.mim/configs/deeplabv3plus/deeplabv3plus_r101-d8_512x512_80k_ade20k.py
mmseg/.mim/configs/deeplabv3plus/deeplabv3plus_r101-d8_512x512_80k_loveda.py
mmseg/.mim/configs/deeplabv3plus/deeplabv3plus_r101-d8_512x512_80k_potsdam.py
mmseg/.mim/configs/deeplabv3plus/deeplabv3plus_r101-d8_769x769_40k_cityscapes.py
mmseg/.mim/configs/deeplabv3plus/deeplabv3plus_r101-d8_769x769_80k_cityscapes.py
mmseg/.mim/configs/deeplabv3plus/deeplabv3plus_r101-d8_fp16_512x1024_80k_cityscapes.py
mmseg/.mim/configs/deeplabv3plus/deeplabv3plus_r101_512x512_C-CM+C-WO-NatOcc-SOT.py
mmseg/.mim/configs/deeplabv3plus/deeplabv3plus_r101b-d8_512x1024_80k_cityscapes.py
mmseg/.mim/configs/deeplabv3plus/deeplabv3plus_r101b-d8_769x769_80k_cityscapes.py
mmseg/.mim/configs/deeplabv3plus/deeplabv3plus_r18-d8_4x4_512x512_80k_vaihingen.py
mmseg/.mim/configs/deeplabv3plus/deeplabv3plus_r18-d8_4x4_896x896_80k_isaid.py
mmseg/.mim/configs/deeplabv3plus/deeplabv3plus_r18-d8_512x1024_80k_cityscapes.py
mmseg/.mim/configs/deeplabv3plus/deeplabv3plus_r18-d8_512x512_80k_loveda.py
mmseg/.mim/configs/deeplabv3plus/deeplabv3plus_r18-d8_512x512_80k_potsdam.py
mmseg/.mim/configs/deeplabv3plus/deeplabv3plus_r18-d8_769x769_80k_cityscapes.py
mmseg/.mim/configs/deeplabv3plus/deeplabv3plus_r18b-d8_512x1024_80k_cityscapes.py
mmseg/.mim/configs/deeplabv3plus/deeplabv3plus_r18b-d8_769x769_80k_cityscapes.py
mmseg/.mim/configs/deeplabv3plus/deeplabv3plus_r50-d8_480x480_40k_pascal_context.py
mmseg/.mim/configs/deeplabv3plus/deeplabv3plus_r50-d8_480x480_40k_pascal_context_59.py
mmseg/.mim/configs/deeplabv3plus/deeplabv3plus_r50-d8_480x480_80k_pascal_context.py
mmseg/.mim/configs/deeplabv3plus/deeplabv3plus_r50-d8_480x480_80k_pascal_context_59.py
mmseg/.mim/configs/deeplabv3plus/deeplabv3plus_r50-d8_4x4_512x512_80k_vaihingen.py
mmseg/.mim/configs/deeplabv3plus/deeplabv3plus_r50-d8_4x4_896x896_80k_isaid.py
mmseg/.mim/configs/deeplabv3plus/deeplabv3plus_r50-d8_512x1024_40k_cityscapes.py
mmseg/.mim/configs/deeplabv3plus/deeplabv3plus_r50-d8_512x1024_80k_cityscapes.py
mmseg/.mim/configs/deeplabv3plus/deeplabv3plus_r50-d8_512x512_160k_ade20k.py
mmseg/.mim/configs/deeplabv3plus/deeplabv3plus_r50-d8_512x512_20k_voc12aug.py
mmseg/.mim/configs/deeplabv3plus/deeplabv3plus_r50-d8_512x512_40k_voc12aug.py
mmseg/.mim/configs/deeplabv3plus/deeplabv3plus_r50-d8_512x512_80k_ade20k.py
mmseg/.mim/configs/deeplabv3plus/deeplabv3plus_r50-d8_512x512_80k_loveda.py
mmseg/.mim/configs/deeplabv3plus/deeplabv3plus_r50-d8_512x512_80k_potsdam.py
mmseg/.mim/configs/deeplabv3plus/deeplabv3plus_r50-d8_769x769_40k_cityscapes.py
mmseg/.mim/configs/deeplabv3plus/deeplabv3plus_r50-d8_769x769_80k_cityscapes.py
mmseg/.mim/configs/deeplabv3plus/deeplabv3plus_r50b-d8_512x1024_80k_cityscapes.py
mmseg/.mim/configs/deeplabv3plus/deeplabv3plus_r50b-d8_769x769_80k_cityscapes.py
```

#### DMNet 至 FastSCNN

```text
mmseg/.mim/configs/dmnet/README.md
mmseg/.mim/configs/dmnet/dmnet.yml
mmseg/.mim/configs/dmnet/dmnet_r101-d8_512x1024_40k_cityscapes.py
mmseg/.mim/configs/dmnet/dmnet_r101-d8_512x1024_80k_cityscapes.py
mmseg/.mim/configs/dmnet/dmnet_r101-d8_512x512_160k_ade20k.py
mmseg/.mim/configs/dmnet/dmnet_r101-d8_512x512_80k_ade20k.py
mmseg/.mim/configs/dmnet/dmnet_r101-d8_769x769_40k_cityscapes.py
mmseg/.mim/configs/dmnet/dmnet_r101-d8_769x769_80k_cityscapes.py
mmseg/.mim/configs/dmnet/dmnet_r50-d8_512x1024_40k_cityscapes.py
mmseg/.mim/configs/dmnet/dmnet_r50-d8_512x1024_80k_cityscapes.py
mmseg/.mim/configs/dmnet/dmnet_r50-d8_512x512_160k_ade20k.py
mmseg/.mim/configs/dmnet/dmnet_r50-d8_512x512_80k_ade20k.py
mmseg/.mim/configs/dmnet/dmnet_r50-d8_769x769_40k_cityscapes.py
mmseg/.mim/configs/dmnet/dmnet_r50-d8_769x769_80k_cityscapes.py
mmseg/.mim/configs/dnlnet/README.md
mmseg/.mim/configs/dnlnet/dnl_r101-d8_512x1024_40k_cityscapes.py
mmseg/.mim/configs/dnlnet/dnl_r101-d8_512x1024_80k_cityscapes.py
mmseg/.mim/configs/dnlnet/dnl_r101-d8_512x512_160k_ade20k.py
mmseg/.mim/configs/dnlnet/dnl_r101-d8_512x512_80k_ade20k.py
mmseg/.mim/configs/dnlnet/dnl_r101-d8_769x769_40k_cityscapes.py
mmseg/.mim/configs/dnlnet/dnl_r101-d8_769x769_80k_cityscapes.py
mmseg/.mim/configs/dnlnet/dnl_r50-d8_512x1024_40k_cityscapes.py
mmseg/.mim/configs/dnlnet/dnl_r50-d8_512x1024_80k_cityscapes.py
mmseg/.mim/configs/dnlnet/dnl_r50-d8_512x512_160k_ade20k.py
mmseg/.mim/configs/dnlnet/dnl_r50-d8_512x512_80k_ade20k.py
mmseg/.mim/configs/dnlnet/dnl_r50-d8_769x769_40k_cityscapes.py
mmseg/.mim/configs/dnlnet/dnl_r50-d8_769x769_80k_cityscapes.py
mmseg/.mim/configs/dnlnet/dnlnet.yml
mmseg/.mim/configs/dpt/README.md
mmseg/.mim/configs/dpt/dpt.yml
mmseg/.mim/configs/dpt/dpt_vit-b16_512x512_160k_ade20k.py
mmseg/.mim/configs/emanet/README.md
mmseg/.mim/configs/emanet/emanet.yml
mmseg/.mim/configs/emanet/emanet_r101-d8_512x1024_80k_cityscapes.py
mmseg/.mim/configs/emanet/emanet_r101-d8_769x769_80k_cityscapes.py
mmseg/.mim/configs/emanet/emanet_r50-d8_512x1024_80k_cityscapes.py
mmseg/.mim/configs/emanet/emanet_r50-d8_769x769_80k_cityscapes.py
mmseg/.mim/configs/encnet/README.md
mmseg/.mim/configs/encnet/encnet.yml
mmseg/.mim/configs/encnet/encnet_r101-d8_512x1024_40k_cityscapes.py
mmseg/.mim/configs/encnet/encnet_r101-d8_512x1024_80k_cityscapes.py
mmseg/.mim/configs/encnet/encnet_r101-d8_512x512_160k_ade20k.py
mmseg/.mim/configs/encnet/encnet_r101-d8_512x512_20k_voc12aug.py
mmseg/.mim/configs/encnet/encnet_r101-d8_512x512_40k_voc12aug.py
mmseg/.mim/configs/encnet/encnet_r101-d8_512x512_80k_ade20k.py
mmseg/.mim/configs/encnet/encnet_r101-d8_769x769_40k_cityscapes.py
mmseg/.mim/configs/encnet/encnet_r101-d8_769x769_80k_cityscapes.py
mmseg/.mim/configs/encnet/encnet_r50-d8_512x1024_40k_cityscapes.py
mmseg/.mim/configs/encnet/encnet_r50-d8_512x1024_80k_cityscapes.py
mmseg/.mim/configs/encnet/encnet_r50-d8_512x512_160k_ade20k.py
mmseg/.mim/configs/encnet/encnet_r50-d8_512x512_20k_voc12aug.py
mmseg/.mim/configs/encnet/encnet_r50-d8_512x512_40k_voc12aug.py
mmseg/.mim/configs/encnet/encnet_r50-d8_512x512_80k_ade20k.py
mmseg/.mim/configs/encnet/encnet_r50-d8_769x769_40k_cityscapes.py
mmseg/.mim/configs/encnet/encnet_r50-d8_769x769_80k_cityscapes.py
mmseg/.mim/configs/encnet/encnet_r50s-d8_512x512_80k_ade20k.py
mmseg/.mim/configs/erfnet/README.md
mmseg/.mim/configs/erfnet/erfnet.yml
mmseg/.mim/configs/erfnet/erfnet_fcn_4x4_512x1024_160k_cityscapes.py
mmseg/.mim/configs/fastfcn/README.md
mmseg/.mim/configs/fastfcn/fastfcn.yml
mmseg/.mim/configs/fastfcn/fastfcn_r50-d32_jpu_aspp_4x4_512x1024_80k_cityscapes.py
mmseg/.mim/configs/fastfcn/fastfcn_r50-d32_jpu_aspp_512x1024_80k_cityscapes.py
mmseg/.mim/configs/fastfcn/fastfcn_r50-d32_jpu_aspp_512x512_160k_ade20k.py
mmseg/.mim/configs/fastfcn/fastfcn_r50-d32_jpu_aspp_512x512_80k_ade20k.py
mmseg/.mim/configs/fastfcn/fastfcn_r50-d32_jpu_enc_4x4_512x1024_80k_cityscapes.py
mmseg/.mim/configs/fastfcn/fastfcn_r50-d32_jpu_enc_512x1024_80k_cityscapes.py
mmseg/.mim/configs/fastfcn/fastfcn_r50-d32_jpu_enc_512x512_160k_ade20k.py
mmseg/.mim/configs/fastfcn/fastfcn_r50-d32_jpu_enc_512x512_80k_ade20k.py
mmseg/.mim/configs/fastfcn/fastfcn_r50-d32_jpu_psp_4x4_512x1024_80k_cityscapes.py
mmseg/.mim/configs/fastfcn/fastfcn_r50-d32_jpu_psp_512x1024_80k_cityscapes.py
mmseg/.mim/configs/fastfcn/fastfcn_r50-d32_jpu_psp_512x512_160k_ade20k.py
mmseg/.mim/configs/fastfcn/fastfcn_r50-d32_jpu_psp_512x512_80k_ade20k.py
mmseg/.mim/configs/fastscnn/README.md
mmseg/.mim/configs/fastscnn/fast_scnn_lr0.12_8x4_160k_cityscapes.py
mmseg/.mim/configs/fastscnn/fastscnn.yml
```

#### FCN、GCNet 与 HRNet

```text
mmseg/.mim/configs/fcn/README.md
mmseg/.mim/configs/fcn/fcn.yml
mmseg/.mim/configs/fcn/fcn_d6_r101-d16_512x1024_40k_cityscapes.py
mmseg/.mim/configs/fcn/fcn_d6_r101-d16_512x1024_80k_cityscapes.py
mmseg/.mim/configs/fcn/fcn_d6_r101-d16_769x769_40k_cityscapes.py
mmseg/.mim/configs/fcn/fcn_d6_r101-d16_769x769_80k_cityscapes.py
mmseg/.mim/configs/fcn/fcn_d6_r101b-d16_512x1024_80k_cityscapes.py
mmseg/.mim/configs/fcn/fcn_d6_r101b-d16_769x769_80k_cityscapes.py
mmseg/.mim/configs/fcn/fcn_d6_r50-d16_512x1024_40k_cityscapes.py
mmseg/.mim/configs/fcn/fcn_d6_r50-d16_512x1024_80k_cityscapes.py
mmseg/.mim/configs/fcn/fcn_d6_r50-d16_769x769_40k_cityscapes.py
mmseg/.mim/configs/fcn/fcn_d6_r50-d16_769x769_80k_cityscapes.py
mmseg/.mim/configs/fcn/fcn_d6_r50b-d16_512x1024_80k_cityscapes.py
mmseg/.mim/configs/fcn/fcn_d6_r50b-d16_769x769_80k_cityscapes.py
mmseg/.mim/configs/fcn/fcn_r101-d8_480x480_40k_pascal_context.py
mmseg/.mim/configs/fcn/fcn_r101-d8_480x480_40k_pascal_context_59.py
mmseg/.mim/configs/fcn/fcn_r101-d8_480x480_80k_pascal_context.py
mmseg/.mim/configs/fcn/fcn_r101-d8_480x480_80k_pascal_context_59.py
mmseg/.mim/configs/fcn/fcn_r101-d8_512x1024_40k_cityscapes.py
mmseg/.mim/configs/fcn/fcn_r101-d8_512x1024_80k_cityscapes.py
mmseg/.mim/configs/fcn/fcn_r101-d8_512x512_160k_ade20k.py
mmseg/.mim/configs/fcn/fcn_r101-d8_512x512_20k_voc12aug.py
mmseg/.mim/configs/fcn/fcn_r101-d8_512x512_40k_voc12aug.py
mmseg/.mim/configs/fcn/fcn_r101-d8_512x512_80k_ade20k.py
mmseg/.mim/configs/fcn/fcn_r101-d8_769x769_40k_cityscapes.py
mmseg/.mim/configs/fcn/fcn_r101-d8_769x769_80k_cityscapes.py
mmseg/.mim/configs/fcn/fcn_r101-d8_fp16_512x1024_80k_cityscapes.py
mmseg/.mim/configs/fcn/fcn_r101b-d8_512x1024_80k_cityscapes.py
mmseg/.mim/configs/fcn/fcn_r101b-d8_769x769_80k_cityscapes.py
mmseg/.mim/configs/fcn/fcn_r18-d8_512x1024_80k_cityscapes.py
mmseg/.mim/configs/fcn/fcn_r18-d8_769x769_80k_cityscapes.py
mmseg/.mim/configs/fcn/fcn_r18b-d8_512x1024_80k_cityscapes.py
mmseg/.mim/configs/fcn/fcn_r18b-d8_769x769_80k_cityscapes.py
mmseg/.mim/configs/fcn/fcn_r50-d8_480x480_40k_pascal_context.py
mmseg/.mim/configs/fcn/fcn_r50-d8_480x480_40k_pascal_context_59.py
mmseg/.mim/configs/fcn/fcn_r50-d8_480x480_80k_pascal_context.py
mmseg/.mim/configs/fcn/fcn_r50-d8_480x480_80k_pascal_context_59.py
mmseg/.mim/configs/fcn/fcn_r50-d8_512x1024_40k_cityscapes.py
mmseg/.mim/configs/fcn/fcn_r50-d8_512x1024_80k_cityscapes.py
mmseg/.mim/configs/fcn/fcn_r50-d8_512x512_160k_ade20k.py
mmseg/.mim/configs/fcn/fcn_r50-d8_512x512_20k_voc12aug.py
mmseg/.mim/configs/fcn/fcn_r50-d8_512x512_40k_voc12aug.py
mmseg/.mim/configs/fcn/fcn_r50-d8_512x512_80k_ade20k.py
mmseg/.mim/configs/fcn/fcn_r50-d8_769x769_40k_cityscapes.py
mmseg/.mim/configs/fcn/fcn_r50-d8_769x769_80k_cityscapes.py
mmseg/.mim/configs/fcn/fcn_r50b-d8_512x1024_80k_cityscapes.py
mmseg/.mim/configs/fcn/fcn_r50b-d8_769x769_80k_cityscapes.py
mmseg/.mim/configs/gcnet/README.md
mmseg/.mim/configs/gcnet/gcnet.yml
mmseg/.mim/configs/gcnet/gcnet_r101-d8_512x1024_40k_cityscapes.py
mmseg/.mim/configs/gcnet/gcnet_r101-d8_512x1024_80k_cityscapes.py
mmseg/.mim/configs/gcnet/gcnet_r101-d8_512x512_160k_ade20k.py
mmseg/.mim/configs/gcnet/gcnet_r101-d8_512x512_20k_voc12aug.py
mmseg/.mim/configs/gcnet/gcnet_r101-d8_512x512_40k_voc12aug.py
mmseg/.mim/configs/gcnet/gcnet_r101-d8_512x512_80k_ade20k.py
mmseg/.mim/configs/gcnet/gcnet_r101-d8_769x769_40k_cityscapes.py
mmseg/.mim/configs/gcnet/gcnet_r101-d8_769x769_80k_cityscapes.py
mmseg/.mim/configs/gcnet/gcnet_r50-d8_512x1024_40k_cityscapes.py
mmseg/.mim/configs/gcnet/gcnet_r50-d8_512x1024_80k_cityscapes.py
mmseg/.mim/configs/gcnet/gcnet_r50-d8_512x512_160k_ade20k.py
mmseg/.mim/configs/gcnet/gcnet_r50-d8_512x512_20k_voc12aug.py
mmseg/.mim/configs/gcnet/gcnet_r50-d8_512x512_40k_voc12aug.py
mmseg/.mim/configs/gcnet/gcnet_r50-d8_512x512_80k_ade20k.py
mmseg/.mim/configs/gcnet/gcnet_r50-d8_769x769_40k_cityscapes.py
mmseg/.mim/configs/gcnet/gcnet_r50-d8_769x769_80k_cityscapes.py
mmseg/.mim/configs/hrnet/README.md
mmseg/.mim/configs/hrnet/fcn_hr18_480x480_40k_pascal_context.py
mmseg/.mim/configs/hrnet/fcn_hr18_480x480_40k_pascal_context_59.py
mmseg/.mim/configs/hrnet/fcn_hr18_480x480_80k_pascal_context.py
mmseg/.mim/configs/hrnet/fcn_hr18_480x480_80k_pascal_context_59.py
mmseg/.mim/configs/hrnet/fcn_hr18_4x4_512x512_80k_vaihingen.py
mmseg/.mim/configs/hrnet/fcn_hr18_4x4_896x896_80k_isaid.py
mmseg/.mim/configs/hrnet/fcn_hr18_512x1024_160k_cityscapes.py
mmseg/.mim/configs/hrnet/fcn_hr18_512x1024_40k_cityscapes.py
mmseg/.mim/configs/hrnet/fcn_hr18_512x1024_80k_cityscapes.py
mmseg/.mim/configs/hrnet/fcn_hr18_512x512_160k_ade20k.py
mmseg/.mim/configs/hrnet/fcn_hr18_512x512_20k_voc12aug.py
mmseg/.mim/configs/hrnet/fcn_hr18_512x512_40k_voc12aug.py
mmseg/.mim/configs/hrnet/fcn_hr18_512x512_80k_ade20k.py
mmseg/.mim/configs/hrnet/fcn_hr18_512x512_80k_loveda.py
mmseg/.mim/configs/hrnet/fcn_hr18_512x512_80k_potsdam.py
mmseg/.mim/configs/hrnet/fcn_hr18s_480x480_40k_pascal_context.py
mmseg/.mim/configs/hrnet/fcn_hr18s_480x480_40k_pascal_context_59.py
mmseg/.mim/configs/hrnet/fcn_hr18s_480x480_80k_pascal_context.py
mmseg/.mim/configs/hrnet/fcn_hr18s_480x480_80k_pascal_context_59.py
mmseg/.mim/configs/hrnet/fcn_hr18s_4x4_512x512_80k_vaihingen.py
mmseg/.mim/configs/hrnet/fcn_hr18s_4x4_896x896_80k_isaid.py
mmseg/.mim/configs/hrnet/fcn_hr18s_512x1024_160k_cityscapes.py
mmseg/.mim/configs/hrnet/fcn_hr18s_512x1024_40k_cityscapes.py
mmseg/.mim/configs/hrnet/fcn_hr18s_512x1024_80k_cityscapes.py
mmseg/.mim/configs/hrnet/fcn_hr18s_512x512_160k_ade20k.py
mmseg/.mim/configs/hrnet/fcn_hr18s_512x512_20k_voc12aug.py
mmseg/.mim/configs/hrnet/fcn_hr18s_512x512_40k_voc12aug.py
mmseg/.mim/configs/hrnet/fcn_hr18s_512x512_80k_ade20k.py
mmseg/.mim/configs/hrnet/fcn_hr18s_512x512_80k_loveda.py
mmseg/.mim/configs/hrnet/fcn_hr18s_512x512_80k_potsdam.py
mmseg/.mim/configs/hrnet/fcn_hr48_480x480_40k_pascal_context.py
mmseg/.mim/configs/hrnet/fcn_hr48_480x480_40k_pascal_context_59.py
mmseg/.mim/configs/hrnet/fcn_hr48_480x480_80k_pascal_context.py
mmseg/.mim/configs/hrnet/fcn_hr48_480x480_80k_pascal_context_59.py
mmseg/.mim/configs/hrnet/fcn_hr48_4x4_512x512_80k_vaihingen.py
mmseg/.mim/configs/hrnet/fcn_hr48_4x4_896x896_80k_isaid.py
mmseg/.mim/configs/hrnet/fcn_hr48_512x1024_160k_cityscapes.py
mmseg/.mim/configs/hrnet/fcn_hr48_512x1024_40k_cityscapes.py
mmseg/.mim/configs/hrnet/fcn_hr48_512x1024_80k_cityscapes.py
mmseg/.mim/configs/hrnet/fcn_hr48_512x512_160k_ade20k.py
mmseg/.mim/configs/hrnet/fcn_hr48_512x512_20k_voc12aug.py
mmseg/.mim/configs/hrnet/fcn_hr48_512x512_40k_voc12aug.py
mmseg/.mim/configs/hrnet/fcn_hr48_512x512_80k_ade20k.py
mmseg/.mim/configs/hrnet/fcn_hr48_512x512_80k_loveda.py
mmseg/.mim/configs/hrnet/fcn_hr48_512x512_80k_potsdam.py
mmseg/.mim/configs/hrnet/hrnet.yml
```

#### ICNet 至 Non-local

```text
mmseg/.mim/configs/icnet/README.md
mmseg/.mim/configs/icnet/icnet.yml
mmseg/.mim/configs/icnet/icnet_r101-d8_832x832_160k_cityscapes.py
mmseg/.mim/configs/icnet/icnet_r101-d8_832x832_80k_cityscapes.py
mmseg/.mim/configs/icnet/icnet_r101-d8_in1k-pre_832x832_160k_cityscapes.py
mmseg/.mim/configs/icnet/icnet_r101-d8_in1k-pre_832x832_80k_cityscapes.py
mmseg/.mim/configs/icnet/icnet_r18-d8_832x832_160k_cityscapes.py
mmseg/.mim/configs/icnet/icnet_r18-d8_832x832_80k_cityscapes.py
mmseg/.mim/configs/icnet/icnet_r18-d8_in1k-pre_832x832_160k_cityscapes.py
mmseg/.mim/configs/icnet/icnet_r18-d8_in1k-pre_832x832_80k_cityscapes.py
mmseg/.mim/configs/icnet/icnet_r50-d8_832x832_160k_cityscapes.py
mmseg/.mim/configs/icnet/icnet_r50-d8_832x832_80k_cityscapes.py
mmseg/.mim/configs/icnet/icnet_r50-d8_in1k-pre_832x832_160k_cityscapes.py
mmseg/.mim/configs/icnet/icnet_r50-d8_in1k-pre_832x832_80k_cityscapes.py
mmseg/.mim/configs/isanet/README.md
mmseg/.mim/configs/isanet/isanet.yml
mmseg/.mim/configs/isanet/isanet_r101-d8_512x1024_40k_cityscapes.py
mmseg/.mim/configs/isanet/isanet_r101-d8_512x1024_80k_cityscapes.py
mmseg/.mim/configs/isanet/isanet_r101-d8_512x512_160k_ade20k.py
mmseg/.mim/configs/isanet/isanet_r101-d8_512x512_20k_voc12aug.py
mmseg/.mim/configs/isanet/isanet_r101-d8_512x512_40k_voc12aug.py
mmseg/.mim/configs/isanet/isanet_r101-d8_512x512_80k_ade20k.py
mmseg/.mim/configs/isanet/isanet_r101-d8_769x769_40k_cityscapes.py
mmseg/.mim/configs/isanet/isanet_r101-d8_769x769_80k_cityscapes.py
mmseg/.mim/configs/isanet/isanet_r50-d8_512x1024_40k_cityscapes.py
mmseg/.mim/configs/isanet/isanet_r50-d8_512x1024_80k_cityscapes.py
mmseg/.mim/configs/isanet/isanet_r50-d8_512x512_160k_ade20k.py
mmseg/.mim/configs/isanet/isanet_r50-d8_512x512_20k_voc12aug.py
mmseg/.mim/configs/isanet/isanet_r50-d8_512x512_40k_voc12aug.py
mmseg/.mim/configs/isanet/isanet_r50-d8_512x512_80k_ade20k.py
mmseg/.mim/configs/isanet/isanet_r50-d8_769x769_40k_cityscapes.py
mmseg/.mim/configs/isanet/isanet_r50-d8_769x769_80k_cityscapes.py
mmseg/.mim/configs/knet/README.md
mmseg/.mim/configs/knet/knet.yml
mmseg/.mim/configs/knet/knet_s3_deeplabv3_r50-d8_8x2_512x512_adamw_80k_ade20k.py
mmseg/.mim/configs/knet/knet_s3_fcn_r50-d8_8x2_512x512_adamw_80k_ade20k.py
mmseg/.mim/configs/knet/knet_s3_pspnet_r50-d8_8x2_512x512_adamw_80k_ade20k.py
mmseg/.mim/configs/knet/knet_s3_upernet_r50-d8_8x2_512x512_adamw_80k_ade20k.py
mmseg/.mim/configs/knet/knet_s3_upernet_swin-l_8x2_512x512_adamw_80k_ade20k.py
mmseg/.mim/configs/knet/knet_s3_upernet_swin-l_8x2_640x640_adamw_80k_ade20k.py
mmseg/.mim/configs/knet/knet_s3_upernet_swin-t_8x2_512x512_adamw_80k_ade20k.py
mmseg/.mim/configs/mae/README.md
mmseg/.mim/configs/mae/mae.yml
mmseg/.mim/configs/mae/upernet_mae-base_fp16_512x512_160k_ade20k_ms.py
mmseg/.mim/configs/mae/upernet_mae-base_fp16_8x2_512x512_160k_ade20k.py
mmseg/.mim/configs/mobilenet_v2/README.md
mmseg/.mim/configs/mobilenet_v2/deeplabv3_m-v2-d8_512x1024_80k_cityscapes.py
mmseg/.mim/configs/mobilenet_v2/deeplabv3_m-v2-d8_512x512_160k_ade20k.py
mmseg/.mim/configs/mobilenet_v2/deeplabv3plus_m-v2-d8_512x1024_80k_cityscapes.py
mmseg/.mim/configs/mobilenet_v2/deeplabv3plus_m-v2-d8_512x512_160k_ade20k.py
mmseg/.mim/configs/mobilenet_v2/fcn_m-v2-d8_512x1024_80k_cityscapes.py
mmseg/.mim/configs/mobilenet_v2/fcn_m-v2-d8_512x512_160k_ade20k.py
mmseg/.mim/configs/mobilenet_v2/mobilenet_v2.yml
mmseg/.mim/configs/mobilenet_v2/pspnet_m-v2-d8_512x1024_80k_cityscapes.py
mmseg/.mim/configs/mobilenet_v2/pspnet_m-v2-d8_512x512_160k_ade20k.py
mmseg/.mim/configs/mobilenet_v3/README.md
mmseg/.mim/configs/mobilenet_v3/lraspp_m-v3-d8_512x1024_320k_cityscapes.py
mmseg/.mim/configs/mobilenet_v3/lraspp_m-v3-d8_scratch_512x1024_320k_cityscapes.py
mmseg/.mim/configs/mobilenet_v3/lraspp_m-v3s-d8_512x1024_320k_cityscapes.py
mmseg/.mim/configs/mobilenet_v3/lraspp_m-v3s-d8_scratch_512x1024_320k_cityscapes.py
mmseg/.mim/configs/mobilenet_v3/mobilenet_v3.yml
mmseg/.mim/configs/nonlocal_net/README.md
mmseg/.mim/configs/nonlocal_net/nonlocal_net.yml
mmseg/.mim/configs/nonlocal_net/nonlocal_r101-d8_512x1024_40k_cityscapes.py
mmseg/.mim/configs/nonlocal_net/nonlocal_r101-d8_512x1024_80k_cityscapes.py
mmseg/.mim/configs/nonlocal_net/nonlocal_r101-d8_512x512_160k_ade20k.py
mmseg/.mim/configs/nonlocal_net/nonlocal_r101-d8_512x512_20k_voc12aug.py
mmseg/.mim/configs/nonlocal_net/nonlocal_r101-d8_512x512_40k_voc12aug.py
mmseg/.mim/configs/nonlocal_net/nonlocal_r101-d8_512x512_80k_ade20k.py
mmseg/.mim/configs/nonlocal_net/nonlocal_r101-d8_769x769_40k_cityscapes.py
mmseg/.mim/configs/nonlocal_net/nonlocal_r101-d8_769x769_80k_cityscapes.py
mmseg/.mim/configs/nonlocal_net/nonlocal_r50-d8_512x1024_40k_cityscapes.py
mmseg/.mim/configs/nonlocal_net/nonlocal_r50-d8_512x1024_80k_cityscapes.py
mmseg/.mim/configs/nonlocal_net/nonlocal_r50-d8_512x512_160k_ade20k.py
mmseg/.mim/configs/nonlocal_net/nonlocal_r50-d8_512x512_20k_voc12aug.py
mmseg/.mim/configs/nonlocal_net/nonlocal_r50-d8_512x512_40k_voc12aug.py
mmseg/.mim/configs/nonlocal_net/nonlocal_r50-d8_512x512_80k_ade20k.py
mmseg/.mim/configs/nonlocal_net/nonlocal_r50-d8_769x769_40k_cityscapes.py
mmseg/.mim/configs/nonlocal_net/nonlocal_r50-d8_769x769_80k_cityscapes.py
```

#### OCRNet、PointRend、PoolFormer 与 PSANet

```text
mmseg/.mim/configs/ocrnet/README.md
mmseg/.mim/configs/ocrnet/ocrnet.yml
mmseg/.mim/configs/ocrnet/ocrnet_hr18_512x1024_160k_cityscapes.py
mmseg/.mim/configs/ocrnet/ocrnet_hr18_512x1024_40k_cityscapes.py
mmseg/.mim/configs/ocrnet/ocrnet_hr18_512x1024_80k_cityscapes.py
mmseg/.mim/configs/ocrnet/ocrnet_hr18_512x512_160k_ade20k.py
mmseg/.mim/configs/ocrnet/ocrnet_hr18_512x512_20k_voc12aug.py
mmseg/.mim/configs/ocrnet/ocrnet_hr18_512x512_40k_voc12aug.py
mmseg/.mim/configs/ocrnet/ocrnet_hr18_512x512_80k_ade20k.py
mmseg/.mim/configs/ocrnet/ocrnet_hr18s_512x1024_160k_cityscapes.py
mmseg/.mim/configs/ocrnet/ocrnet_hr18s_512x1024_40k_cityscapes.py
mmseg/.mim/configs/ocrnet/ocrnet_hr18s_512x1024_80k_cityscapes.py
mmseg/.mim/configs/ocrnet/ocrnet_hr18s_512x512_160k_ade20k.py
mmseg/.mim/configs/ocrnet/ocrnet_hr18s_512x512_20k_voc12aug.py
mmseg/.mim/configs/ocrnet/ocrnet_hr18s_512x512_40k_voc12aug.py
mmseg/.mim/configs/ocrnet/ocrnet_hr18s_512x512_80k_ade20k.py
mmseg/.mim/configs/ocrnet/ocrnet_hr48_512x1024_160k_cityscapes.py
mmseg/.mim/configs/ocrnet/ocrnet_hr48_512x1024_40k_cityscapes.py
mmseg/.mim/configs/ocrnet/ocrnet_hr48_512x1024_80k_cityscapes.py
mmseg/.mim/configs/ocrnet/ocrnet_hr48_512x512_160k_ade20k.py
mmseg/.mim/configs/ocrnet/ocrnet_hr48_512x512_20k_voc12aug.py
mmseg/.mim/configs/ocrnet/ocrnet_hr48_512x512_40k_voc12aug.py
mmseg/.mim/configs/ocrnet/ocrnet_hr48_512x512_80k_ade20k.py
mmseg/.mim/configs/ocrnet/ocrnet_r101-d8_512x1024_40k_b16_cityscapes.py
mmseg/.mim/configs/ocrnet/ocrnet_r101-d8_512x1024_40k_b8_cityscapes.py
mmseg/.mim/configs/ocrnet/ocrnet_r101-d8_512x1024_80k_b16_cityscapes.py
mmseg/.mim/configs/point_rend/README.md
mmseg/.mim/configs/point_rend/point_rend.yml
mmseg/.mim/configs/point_rend/pointrend_r101_512x1024_80k_cityscapes.py
mmseg/.mim/configs/point_rend/pointrend_r101_512x512_160k_ade20k.py
mmseg/.mim/configs/point_rend/pointrend_r50_512x1024_80k_cityscapes.py
mmseg/.mim/configs/point_rend/pointrend_r50_512x512_160k_ade20k.py
mmseg/.mim/configs/poolformer/README.md
mmseg/.mim/configs/poolformer/fpn_poolformer_m36_8x4_512x512_40k_ade20k.py
mmseg/.mim/configs/poolformer/fpn_poolformer_m48_8x4_512x512_40k_ade20k.py
mmseg/.mim/configs/poolformer/fpn_poolformer_s12_8x4_512x512_40k_ade20k.py
mmseg/.mim/configs/poolformer/fpn_poolformer_s24_8x4_512x512_40k_ade20k.py
mmseg/.mim/configs/poolformer/fpn_poolformer_s36_8x4_512x512_40k_ade20k.py
mmseg/.mim/configs/poolformer/poolformer.yml
mmseg/.mim/configs/psanet/README.md
mmseg/.mim/configs/psanet/psanet.yml
mmseg/.mim/configs/psanet/psanet_r101-d8_512x1024_40k_cityscapes.py
mmseg/.mim/configs/psanet/psanet_r101-d8_512x1024_80k_cityscapes.py
mmseg/.mim/configs/psanet/psanet_r101-d8_512x512_160k_ade20k.py
mmseg/.mim/configs/psanet/psanet_r101-d8_512x512_20k_voc12aug.py
mmseg/.mim/configs/psanet/psanet_r101-d8_512x512_40k_voc12aug.py
mmseg/.mim/configs/psanet/psanet_r101-d8_512x512_80k_ade20k.py
mmseg/.mim/configs/psanet/psanet_r101-d8_769x769_40k_cityscapes.py
mmseg/.mim/configs/psanet/psanet_r101-d8_769x769_80k_cityscapes.py
mmseg/.mim/configs/psanet/psanet_r50-d8_512x1024_40k_cityscapes.py
mmseg/.mim/configs/psanet/psanet_r50-d8_512x1024_80k_cityscapes.py
mmseg/.mim/configs/psanet/psanet_r50-d8_512x512_160k_ade20k.py
mmseg/.mim/configs/psanet/psanet_r50-d8_512x512_20k_voc12aug.py
mmseg/.mim/configs/psanet/psanet_r50-d8_512x512_40k_voc12aug.py
mmseg/.mim/configs/psanet/psanet_r50-d8_512x512_80k_ade20k.py
mmseg/.mim/configs/psanet/psanet_r50-d8_769x769_40k_cityscapes.py
mmseg/.mim/configs/psanet/psanet_r50-d8_769x769_80k_cityscapes.py
```

#### PSPNet

```text
mmseg/.mim/configs/pspnet/README.md
mmseg/.mim/configs/pspnet/pspnet.yml
mmseg/.mim/configs/pspnet/pspnet_r101-d8_480x480_40k_pascal_context.py
mmseg/.mim/configs/pspnet/pspnet_r101-d8_480x480_40k_pascal_context_59.py
mmseg/.mim/configs/pspnet/pspnet_r101-d8_480x480_80k_pascal_context.py
mmseg/.mim/configs/pspnet/pspnet_r101-d8_480x480_80k_pascal_context_59.py
mmseg/.mim/configs/pspnet/pspnet_r101-d8_4x4_512x512_80k_potsdam.py
mmseg/.mim/configs/pspnet/pspnet_r101-d8_4x4_512x512_80k_vaihingen.py
mmseg/.mim/configs/pspnet/pspnet_r101-d8_512x1024_40k_cityscapes.py
mmseg/.mim/configs/pspnet/pspnet_r101-d8_512x1024_40k_dark.py
mmseg/.mim/configs/pspnet/pspnet_r101-d8_512x1024_40k_night_driving.py
mmseg/.mim/configs/pspnet/pspnet_r101-d8_512x1024_80k_cityscapes.py
mmseg/.mim/configs/pspnet/pspnet_r101-d8_512x512_160k_ade20k.py
mmseg/.mim/configs/pspnet/pspnet_r101-d8_512x512_20k_voc12aug.py
mmseg/.mim/configs/pspnet/pspnet_r101-d8_512x512_40k_voc12aug.py
mmseg/.mim/configs/pspnet/pspnet_r101-d8_512x512_4x4_160k_coco-stuff164k.py
mmseg/.mim/configs/pspnet/pspnet_r101-d8_512x512_4x4_20k_coco-stuff10k.py
mmseg/.mim/configs/pspnet/pspnet_r101-d8_512x512_4x4_320k_coco-stuff164k.py
mmseg/.mim/configs/pspnet/pspnet_r101-d8_512x512_4x4_40k_coco-stuff10k.py
mmseg/.mim/configs/pspnet/pspnet_r101-d8_512x512_4x4_80k_coco-stuff164k.py
mmseg/.mim/configs/pspnet/pspnet_r101-d8_512x512_80k_ade20k.py
mmseg/.mim/configs/pspnet/pspnet_r101-d8_512x512_80k_loveda.py
mmseg/.mim/configs/pspnet/pspnet_r101-d8_769x769_40k_cityscapes.py
mmseg/.mim/configs/pspnet/pspnet_r101-d8_769x769_80k_cityscapes.py
mmseg/.mim/configs/pspnet/pspnet_r101-d8_fp16_512x1024_80k_cityscapes.py
mmseg/.mim/configs/pspnet/pspnet_r101b-d8_512x1024_80k_cityscapes.py
mmseg/.mim/configs/pspnet/pspnet_r101b-d8_512x1024_80k_dark.py
mmseg/.mim/configs/pspnet/pspnet_r101b-d8_512x1024_80k_night_driving.py
mmseg/.mim/configs/pspnet/pspnet_r101b-d8_769x769_80k_cityscapes.py
mmseg/.mim/configs/pspnet/pspnet_r18-d8_4x4_512x512_80k_potsdam.py
mmseg/.mim/configs/pspnet/pspnet_r18-d8_4x4_512x512_80k_vaihingen.py
mmseg/.mim/configs/pspnet/pspnet_r18-d8_4x4_896x896_80k_isaid.py
mmseg/.mim/configs/pspnet/pspnet_r18-d8_512x1024_80k_cityscapes.py
mmseg/.mim/configs/pspnet/pspnet_r18-d8_512x512_80k_loveda.py
mmseg/.mim/configs/pspnet/pspnet_r18-d8_769x769_80k_cityscapes.py
mmseg/.mim/configs/pspnet/pspnet_r18b-d8_512x1024_80k_cityscapes.py
mmseg/.mim/configs/pspnet/pspnet_r18b-d8_769x769_80k_cityscapes.py
mmseg/.mim/configs/pspnet/pspnet_r50-d32_512x1024_80k_cityscapes.py
mmseg/.mim/configs/pspnet/pspnet_r50-d32_rsb-pretrain_512x1024_adamw_80k_cityscapes.py
mmseg/.mim/configs/pspnet/pspnet_r50-d8_480x480_40k_pascal_context.py
mmseg/.mim/configs/pspnet/pspnet_r50-d8_480x480_40k_pascal_context_59.py
mmseg/.mim/configs/pspnet/pspnet_r50-d8_480x480_80k_pascal_context.py
mmseg/.mim/configs/pspnet/pspnet_r50-d8_480x480_80k_pascal_context_59.py
mmseg/.mim/configs/pspnet/pspnet_r50-d8_4x4_512x512_80k_potsdam.py
mmseg/.mim/configs/pspnet/pspnet_r50-d8_4x4_512x512_80k_vaihingen.py
mmseg/.mim/configs/pspnet/pspnet_r50-d8_4x4_896x896_80k_isaid.py
mmseg/.mim/configs/pspnet/pspnet_r50-d8_512x1024_40k_cityscapes.py
mmseg/.mim/configs/pspnet/pspnet_r50-d8_512x1024_40k_dark.py
mmseg/.mim/configs/pspnet/pspnet_r50-d8_512x1024_40k_night_driving.py
mmseg/.mim/configs/pspnet/pspnet_r50-d8_512x1024_80k_cityscapes.py
mmseg/.mim/configs/pspnet/pspnet_r50-d8_512x1024_80k_dark.py
mmseg/.mim/configs/pspnet/pspnet_r50-d8_512x1024_80k_night_driving.py
mmseg/.mim/configs/pspnet/pspnet_r50-d8_512x512_160k_ade20k.py
mmseg/.mim/configs/pspnet/pspnet_r50-d8_512x512_20k_voc12aug.py
mmseg/.mim/configs/pspnet/pspnet_r50-d8_512x512_40k_voc12aug.py
mmseg/.mim/configs/pspnet/pspnet_r50-d8_512x512_4x4_160k_coco-stuff164k.py
mmseg/.mim/configs/pspnet/pspnet_r50-d8_512x512_4x4_20k_coco-stuff10k.py
mmseg/.mim/configs/pspnet/pspnet_r50-d8_512x512_4x4_320k_coco-stuff164k.py
mmseg/.mim/configs/pspnet/pspnet_r50-d8_512x512_4x4_40k_coco-stuff10k.py
mmseg/.mim/configs/pspnet/pspnet_r50-d8_512x512_4x4_80k_coco-stuff164k.py
mmseg/.mim/configs/pspnet/pspnet_r50-d8_512x512_80k_ade20k.py
mmseg/.mim/configs/pspnet/pspnet_r50-d8_512x512_80k_loveda.py
mmseg/.mim/configs/pspnet/pspnet_r50-d8_769x769_40k_cityscapes.py
mmseg/.mim/configs/pspnet/pspnet_r50-d8_769x769_80k_cityscapes.py
mmseg/.mim/configs/pspnet/pspnet_r50-d8_rsb-pretrain_512x1024_adamw_80k_cityscapes.py
mmseg/.mim/configs/pspnet/pspnet_r50b-d32_512x1024_80k_cityscapes.py
mmseg/.mim/configs/pspnet/pspnet_r50b-d8_512x1024_80k_cityscapes.py
mmseg/.mim/configs/pspnet/pspnet_r50b-d8_769x769_80k_cityscapes.py
```

#### ResNeSt、SegFormer、Segmenter、Semantic FPN、SETR 与 STDC

```text
mmseg/.mim/configs/resnest/README.md
mmseg/.mim/configs/resnest/deeplabv3_s101-d8_512x1024_80k_cityscapes.py
mmseg/.mim/configs/resnest/deeplabv3_s101-d8_512x512_160k_ade20k.py
mmseg/.mim/configs/resnest/deeplabv3plus_s101-d8_512x1024_80k_cityscapes.py
mmseg/.mim/configs/resnest/deeplabv3plus_s101-d8_512x512_160k_ade20k.py
mmseg/.mim/configs/resnest/fcn_s101-d8_512x1024_80k_cityscapes.py
mmseg/.mim/configs/resnest/fcn_s101-d8_512x512_160k_ade20k.py
mmseg/.mim/configs/resnest/pspnet_s101-d8_512x1024_80k_cityscapes.py
mmseg/.mim/configs/resnest/pspnet_s101-d8_512x512_160k_ade20k.py
mmseg/.mim/configs/resnest/resnest.yml
mmseg/.mim/configs/segformer/README.md
mmseg/.mim/configs/segformer/segformer.yml
mmseg/.mim/configs/segformer/segformer_mit-b0_512x512_160k_ade20k.py
mmseg/.mim/configs/segformer/segformer_mit-b0_8x1_1024x1024_160k_cityscapes.py
mmseg/.mim/configs/segformer/segformer_mit-b1_512x512_160k_ade20k.py
mmseg/.mim/configs/segformer/segformer_mit-b1_8x1_1024x1024_160k_cityscapes.py
mmseg/.mim/configs/segformer/segformer_mit-b2_512x512_160k_ade20k.py
mmseg/.mim/configs/segformer/segformer_mit-b2_8x1_1024x1024_160k_cityscapes.py
mmseg/.mim/configs/segformer/segformer_mit-b3_512x512_160k_ade20k.py
mmseg/.mim/configs/segformer/segformer_mit-b3_8x1_1024x1024_160k_cityscapes.py
mmseg/.mim/configs/segformer/segformer_mit-b4_512x512_160k_ade20k.py
mmseg/.mim/configs/segformer/segformer_mit-b4_8x1_1024x1024_160k_cityscapes.py
mmseg/.mim/configs/segformer/segformer_mit-b5_512x512_160k_ade20k.py
mmseg/.mim/configs/segformer/segformer_mit-b5_640x640_160k_ade20k.py
mmseg/.mim/configs/segformer/segformer_mit-b5_8x1_1024x1024_160k_cityscapes.py
mmseg/.mim/configs/segmenter/README.md
mmseg/.mim/configs/segmenter/segmenter.yml
mmseg/.mim/configs/segmenter/segmenter_vit-b_mask_8x1_512x512_160k_ade20k.py
mmseg/.mim/configs/segmenter/segmenter_vit-l_mask_8x1_640x640_160k_ade20k.py
mmseg/.mim/configs/segmenter/segmenter_vit-s_linear_8x1_512x512_160k_ade20k.py
mmseg/.mim/configs/segmenter/segmenter_vit-s_mask_8x1_512x512_160k_ade20k.py
mmseg/.mim/configs/segmenter/segmenter_vit-t_mask_8x1_512x512_160k_ade20k.py
mmseg/.mim/configs/sem_fpn/README.md
mmseg/.mim/configs/sem_fpn/fpn_r101_512x1024_80k_cityscapes.py
mmseg/.mim/configs/sem_fpn/fpn_r101_512x512_160k_ade20k.py
mmseg/.mim/configs/sem_fpn/fpn_r50_512x1024_80k_cityscapes.py
mmseg/.mim/configs/sem_fpn/fpn_r50_512x512_160k_ade20k.py
mmseg/.mim/configs/sem_fpn/sem_fpn.yml
mmseg/.mim/configs/setr/README.md
mmseg/.mim/configs/setr/setr.yml
mmseg/.mim/configs/setr/setr_mla_512x512_160k_b16_ade20k.py
mmseg/.mim/configs/setr/setr_mla_512x512_160k_b8_ade20k.py
mmseg/.mim/configs/setr/setr_naive_512x512_160k_b16_ade20k.py
mmseg/.mim/configs/setr/setr_pup_512x512_160k_b16_ade20k.py
mmseg/.mim/configs/setr/setr_vit-large_mla_8x1_768x768_80k_cityscapes.py
mmseg/.mim/configs/setr/setr_vit-large_naive_8x1_768x768_80k_cityscapes.py
mmseg/.mim/configs/setr/setr_vit-large_pup_8x1_768x768_80k_cityscapes.py
mmseg/.mim/configs/stdc/README.md
mmseg/.mim/configs/stdc/stdc.yml
mmseg/.mim/configs/stdc/stdc1_512x1024_80k_cityscapes.py
mmseg/.mim/configs/stdc/stdc1_in1k-pre_512x1024_80k_cityscapes.py
mmseg/.mim/configs/stdc/stdc2_512x1024_80k_cityscapes.py
mmseg/.mim/configs/stdc/stdc2_in1k-pre_512x1024_80k_cityscapes.py
```

#### Swin、Twins、U-Net、UPerNet 与 ViT

```text
mmseg/.mim/configs/swin/README.md
mmseg/.mim/configs/swin/swin.yml
mmseg/.mim/configs/swin/upernet_swin_base_patch4_window12_512x512_160k_ade20k_pretrain_384x384_1K.py
mmseg/.mim/configs/swin/upernet_swin_base_patch4_window12_512x512_160k_ade20k_pretrain_384x384_22K.py
mmseg/.mim/configs/swin/upernet_swin_base_patch4_window7_512x512_160k_ade20k_pretrain_224x224_1K.py
mmseg/.mim/configs/swin/upernet_swin_base_patch4_window7_512x512_160k_ade20k_pretrain_224x224_22K.py
mmseg/.mim/configs/swin/upernet_swin_large_patch4_window12_512x512_pretrain_384x384_22K_160k_ade20k.py
mmseg/.mim/configs/swin/upernet_swin_large_patch4_window7_512x512_pretrain_224x224_22K_160k_ade20k.py
mmseg/.mim/configs/swin/upernet_swin_small_patch4_window7_512x512_160k_ade20k_pretrain_224x224_1K.py
mmseg/.mim/configs/swin/upernet_swin_tiny_patch4_window7_512x512_160k_ade20k_pretrain_224x224_1K.py
mmseg/.mim/configs/twins/README.md
mmseg/.mim/configs/twins/twins.yml
mmseg/.mim/configs/twins/twins_pcpvt-b_fpn_fpnhead_8x4_512x512_80k_ade20k.py
mmseg/.mim/configs/twins/twins_pcpvt-b_uperhead_8x2_512x512_160k_ade20k.py
mmseg/.mim/configs/twins/twins_pcpvt-l_fpn_fpnhead_8x4_512x512_80k_ade20k.py
mmseg/.mim/configs/twins/twins_pcpvt-l_uperhead_8x2_512x512_160k_ade20k.py
mmseg/.mim/configs/twins/twins_pcpvt-s_fpn_fpnhead_8x4_512x512_80k_ade20k.py
mmseg/.mim/configs/twins/twins_pcpvt-s_uperhead_8x4_512x512_160k_ade20k.py
mmseg/.mim/configs/twins/twins_svt-b_fpn_fpnhead_8x4_512x512_80k_ade20k.py
mmseg/.mim/configs/twins/twins_svt-b_uperhead_8x2_512x512_160k_ade20k.py
mmseg/.mim/configs/twins/twins_svt-l_fpn_fpnhead_8x4_512x512_80k_ade20k.py
mmseg/.mim/configs/twins/twins_svt-l_uperhead_8x2_512x512_160k_ade20k.py
mmseg/.mim/configs/twins/twins_svt-s_fpn_fpnhead_8x4_512x512_80k_ade20k.py
mmseg/.mim/configs/twins/twins_svt-s_uperhead_8x2_512x512_160k_ade20k.py
mmseg/.mim/configs/unet/README.md
mmseg/.mim/configs/unet/deeplabv3_unet_s5-d16_128x128_40k_chase_db1.py
mmseg/.mim/configs/unet/deeplabv3_unet_s5-d16_128x128_40k_stare.py
mmseg/.mim/configs/unet/deeplabv3_unet_s5-d16_256x256_40k_hrf.py
mmseg/.mim/configs/unet/deeplabv3_unet_s5-d16_64x64_40k_drive.py
mmseg/.mim/configs/unet/deeplabv3_unet_s5-d16_ce-1.0-dice-3.0_128x128_40k_chase-db1.py
mmseg/.mim/configs/unet/deeplabv3_unet_s5-d16_ce-1.0-dice-3.0_128x128_40k_stare.py
mmseg/.mim/configs/unet/deeplabv3_unet_s5-d16_ce-1.0-dice-3.0_256x256_40k_hrf.py
mmseg/.mim/configs/unet/deeplabv3_unet_s5-d16_ce-1.0-dice-3.0_64x64_40k_drive.py
mmseg/.mim/configs/unet/fcn_unet_s5-d16_128x128_40k_chase_db1.py
mmseg/.mim/configs/unet/fcn_unet_s5-d16_128x128_40k_stare.py
mmseg/.mim/configs/unet/fcn_unet_s5-d16_256x256_40k_hrf.py
mmseg/.mim/configs/unet/fcn_unet_s5-d16_4x4_512x1024_160k_cityscapes.py
mmseg/.mim/configs/unet/fcn_unet_s5-d16_64x64_40k_drive.py
mmseg/.mim/configs/unet/fcn_unet_s5-d16_ce-1.0-dice-3.0_128x128_40k_chase-db1.py
mmseg/.mim/configs/unet/fcn_unet_s5-d16_ce-1.0-dice-3.0_128x128_40k_stare.py
mmseg/.mim/configs/unet/fcn_unet_s5-d16_ce-1.0-dice-3.0_256x256_40k_hrf.py
mmseg/.mim/configs/unet/fcn_unet_s5-d16_ce-1.0-dice-3.0_64x64_40k_drive.py
mmseg/.mim/configs/unet/pspnet_unet_s5-d16_128x128_40k_chase_db1.py
mmseg/.mim/configs/unet/pspnet_unet_s5-d16_128x128_40k_stare.py
mmseg/.mim/configs/unet/pspnet_unet_s5-d16_256x256_40k_hrf.py
mmseg/.mim/configs/unet/pspnet_unet_s5-d16_64x64_40k_drive.py
mmseg/.mim/configs/unet/pspnet_unet_s5-d16_ce-1.0-dice-3.0_128x128_40k_chase-db1.py
mmseg/.mim/configs/unet/pspnet_unet_s5-d16_ce-1.0-dice-3.0_128x128_40k_stare.py
mmseg/.mim/configs/unet/pspnet_unet_s5-d16_ce-1.0-dice-3.0_256x256_40k_hrf.py
mmseg/.mim/configs/unet/pspnet_unet_s5-d16_ce-1.0-dice-3.0_64x64_40k_drive.py
mmseg/.mim/configs/unet/unet.yml
mmseg/.mim/configs/upernet/README.md
mmseg/.mim/configs/upernet/upernet.yml
mmseg/.mim/configs/upernet/upernet_r101_512x1024_40k_cityscapes.py
mmseg/.mim/configs/upernet/upernet_r101_512x1024_80k_cityscapes.py
mmseg/.mim/configs/upernet/upernet_r101_512x512_160k_ade20k.py
mmseg/.mim/configs/upernet/upernet_r101_512x512_20k_voc12aug.py
mmseg/.mim/configs/upernet/upernet_r101_512x512_40k_voc12aug.py
mmseg/.mim/configs/upernet/upernet_r101_512x512_80k_ade20k.py
mmseg/.mim/configs/upernet/upernet_r101_769x769_40k_cityscapes.py
mmseg/.mim/configs/upernet/upernet_r101_769x769_80k_cityscapes.py
mmseg/.mim/configs/upernet/upernet_r18_512x1024_40k_cityscapes.py
mmseg/.mim/configs/upernet/upernet_r18_512x1024_80k_cityscapes.py
mmseg/.mim/configs/upernet/upernet_r18_512x512_160k_ade20k.py
mmseg/.mim/configs/upernet/upernet_r18_512x512_20k_voc12aug.py
mmseg/.mim/configs/upernet/upernet_r18_512x512_40k_voc12aug.py
mmseg/.mim/configs/upernet/upernet_r18_512x512_80k_ade20k.py
mmseg/.mim/configs/upernet/upernet_r50_512x1024_40k_cityscapes.py
mmseg/.mim/configs/upernet/upernet_r50_512x1024_80k_cityscapes.py
mmseg/.mim/configs/upernet/upernet_r50_512x512_160k_ade20k.py
mmseg/.mim/configs/upernet/upernet_r50_512x512_20k_voc12aug.py
mmseg/.mim/configs/upernet/upernet_r50_512x512_40k_voc12aug.py
mmseg/.mim/configs/upernet/upernet_r50_512x512_80k_ade20k.py
mmseg/.mim/configs/upernet/upernet_r50_769x769_40k_cityscapes.py
mmseg/.mim/configs/upernet/upernet_r50_769x769_80k_cityscapes.py
mmseg/.mim/configs/vit/README.md
mmseg/.mim/configs/vit/upernet_deit-b16_512x512_160k_ade20k.py
mmseg/.mim/configs/vit/upernet_deit-b16_512x512_80k_ade20k.py
mmseg/.mim/configs/vit/upernet_deit-b16_ln_mln_512x512_160k_ade20k.py
mmseg/.mim/configs/vit/upernet_deit-b16_mln_512x512_160k_ade20k.py
mmseg/.mim/configs/vit/upernet_deit-s16_512x512_160k_ade20k.py
mmseg/.mim/configs/vit/upernet_deit-s16_512x512_80k_ade20k.py
mmseg/.mim/configs/vit/upernet_deit-s16_ln_mln_512x512_160k_ade20k.py
mmseg/.mim/configs/vit/upernet_deit-s16_mln_512x512_160k_ade20k.py
mmseg/.mim/configs/vit/upernet_vit-b16_ln_mln_512x512_160k_ade20k.py
mmseg/.mim/configs/vit/upernet_vit-b16_mln_512x512_160k_ade20k.py
mmseg/.mim/configs/vit/upernet_vit-b16_mln_512x512_80k_ade20k.py
mmseg/.mim/configs/vit/vit.yml
```

All 788 `.mim/configs/` paths are listed above. README files describe model families, YAML files provide model-index metadata, and Python files compose upstream recipes; they should be modified only when intentionally maintaining the embedded compatibility distribution.

## 15. 待核验项与修改边界

- 未逐一追溯 949 个 `mmseg/` 文件相对其上游版本的具体 commit 改动史；目录按当前代码职责标为上游/兼容。
- 根 `figs/` 中 `Sal.jpg`、`overview.jpg`、`zhihu.png` 未发现当前跟踪文本的直接引用；只确认其静态资源身份。
- `.DS_Store` 的保留理由和 `models/decoders/test.py`、`utils/engine/dist_test.py` 是否仍有维护用途缺少直接证据。
- `report-index.json` 的 latest 元数据仍按其维护时点理解；实时恢复不得据此覆盖 current status。
- 可直接修改的主要范围是本地模型、配置、工具、测试和稳定指南，但实验身份变更需要新 protocol/qualification。冻结 split、历史证据、已发布 Canvas 和 `liu-test-exp/**` 不能为适配新结论而改写。
