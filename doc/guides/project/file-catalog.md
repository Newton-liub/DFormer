# DFormer 当前文件职责索引

> **文档角色：** 当前职责导航，不承担 MUSeg 实时状态或完整文件清单职责。
> **核验时点：** 2026-09-05。
> **实时入口：** `doc/main/MUSeg-current-status.md`；研究选择见 `doc/main/MUSeg-open-decisions.md`。
> **后继关系：** 2026-08-27 的完整逐文件历史快照见 [`file-catalog-2026-08-27.md`](file-catalog-2026-08-27.md)；当前结构变化以本索引和 Git 为准。

本索引按职责提供最小阅读入口。需要逐文件覆盖、历史计数或旧时点说明时，再读取完整快照；它不是实时实验状态来源。

## 首次阅读

1. `README.md`：仓库总入口和上游 DFormer/DFormerv2 说明。
2. `doc/main/MUSeg-current-status.md`：MUSeg 当前事实、授权边界、证据位置和恢复规则。
3. `doc/main/MUSeg-open-decisions.md`：仍影响后续研究口径的选择及其处置。
4. `doc/guides/project/README.md`：架构、数据流、训练、评估和证据边界。
5. `doc/guides/project/research-branch-governance.md`：稳定基准与独立研究分支治理。

## 关键目录

- `data/splits/MUSeg/dev-v1/`：冻结 development split、manifest 和审计结果；official-test 文件只作为封存身份锚点。
- `doc/main/`：唯一当前状态和开放研究决策。
- `doc/guides/`：稳定项目/云操作指南。
- `doc/plans/deferred/2026-09-MUSeg-unexecuted/`：A2/B2 与方向1等未执行、未授权候选计划。
- `doc/plans/archive/`：已完成阶段和历史计划；不构成新运行授权。
- `doc/reports/`：日期化报告、交接材料和 `report-index.json`。
- `doc/canvases/`：当前可视化汇报源；`doc/canvases/old/` 为只读历史版本。
- `local_configs/`：模型、数据、训练和评估配置；新方向需新 config 身份。
- `protocols/`：可提交的 protocol 模板；物化 manifest 位于被忽略路径。
- `models/`：DFormer/DFormerv2 模型、解码器和损失实现。
- `tools/`：数据、split、protocol、preflight、训练编排、后评估、裁决和 Canvas 工具。
- `utils/`：dataloader、训练、评估、checkpoint、指标和通用运行代码。
- `tests/`：协议、split、checkpoint、训练操作和 MVE 的定点测试。
- `mmseg/`：内嵌 MMSegmentation 兼容/上游代码，不等同于本地 MUSeg 主入口。

## MUSeg 稳定基线入口

- 数据与 split：`doc/dataset.md`、`data/splits/MUSeg/dev-v1/`。
- Quick-B0 配置模板：`local_configs/MUSeg/DFormerv2_S_QuickB0.py`。
- Quick-B0 protocol 模板：`protocols/museg-dformerv2-s-rgb-quick-b0-v1.template.json`。
- 主评估工具：`tools/evaluate_museg_checkpoint.py`；当前主 evaluator 身份以实时状态为准。
- 训练/恢复入口：`tools/preflight_train.py`、`tools/run_museg_seed.py`、`utils/train.py`；启动前必须重新核对授权和 protocol。
- 正式报告索引：`doc/reports/report-index.json`。

## 边界提醒

- 当前 Quick-B0 使用 RGB 输入契约；历史 BGR 结果不得与其混合统计。
- 训练 crop `480×640` 不等于统一 validation geometry；评估 geometry 必须随 evaluator 明确记录。
- 单 seed development B0 是内部共同对照，不是三 seed 论文级 baseline。
- official test 继续保持 `sealed_unread`；计划、测试通过、文件存在或进程退出都不能单独证明实验完成。
- 大型数据、checkpoint、云证据、物化 protocol 和归档包不进入 Git；归档 ZIP 仅在本地 `archive-export/` 保存并由 `.gitignore` 排除。
