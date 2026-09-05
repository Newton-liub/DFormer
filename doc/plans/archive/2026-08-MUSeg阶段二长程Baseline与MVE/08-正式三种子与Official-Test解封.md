# 08：正式三种子与 Official Test 解封

> **后继候选计划：** `doc/plans/deferred/2026-09-MUSeg-unexecuted/MUSeg-A2-B2深度有效性/00-总方向规划.md`（该后继计划现已延期，未执行）。

> **文档角色：** Gate E 后的正式执行计划，不承担实时状态或自动授权。
> **形成或核验时点：** 2026-08-27。
> **实时入口：** `doc/main/MUSeg-current-status.md`。
> **后继关系：** 取代旧 `06-正式三种子Baseline与Test解封.md`；只有 Gate E 与用户运行授权均满足后才能执行。

## 1. 前置条件

- Gate E 已冻结 `B0-only` 或 `B0 + 最终 variant`、代码开关、协议、三 seed allowlist、预算、checkpoint/evaluator 与资源边界；
- 正式 commit 包含全部实现和测试，工作区干净；
- `formal-B0` 与所有 variant 在同一最终代码基线上，B0 只关闭预登记模块；
- 每个结构完成匹配当前显存/数据流的 qualification；
- official train 为完整 1595 样本；official test 1576 样本在训练期保持 sealed unread；
- development reference、短训和 screening checkpoint 均不复用。

## 2. 正式 paired 三 seed

对 `formal-B0` 和 Gate E 中每个最终 variant：

- 使用相同的三个预注册 seed；
- 每个 `variant × seed` 从相同 pretrained 独立开始；
- 使用完整 official train，不配置 training-time official test evaluator；
- 同一 seed 内优先顺序运行 B0 与 variant，实际顺序在 protocol 预登记；
- 每个 run 使用独立输出目录、SwanLab identity、manifest 和 checkpoint；
- 运行中不调参、不替换 seed、不取消不利 variant、不挑周期 checkpoint。

如果 Gate E 输出 `B0-only`，只运行 formal B0 三 seed并形成正式 baseline，不虚构模块提升结论。

## 3. 每个 run 的验收与恢复

每项必须有退出码 0、预定 checkpoint、完整 commit/protocol/split/pretrained/seed/variant 身份、checkpoint SHA-256、输入/几何契约和 sealed-test 记录。

只允许从 CPU 可读、schema/hash/协议一致的最高完整 epoch checkpoint 恢复；child run 使用新 ID 并绑定 parent。代码、模型、数据、优化或随机语义修复会使所有受影响 paired 两臂失效，必须一起重跑。失败 seed 不得替换；增加 seed 时 B0 与所有 variant一起增加并升级 protocol。

## 4. Gate F

全部正式 run 完成后向用户提交不可变 allowlist，列出：

- variant、seed、checkpoint 路径/SHA、epoch/global step；
- protocol、commit、split/pretrained 和 evaluator identity；
- 退出码、耗时、qualification、失败/恢复历史；
- training 与 validation 期间 `official_test_included=false` 的证据。

用户明确批准前不读取 official test。Gate F 只批准 allowlist，不允许再改变模型、checkpoint、seed、geometry 或指标定义。

## 5. Official test 一次性评估

Gate F 后，对 allowlist 中每个 checkpoint 恰好按冻结的单一 evaluator/geometry 运行一次：

- 显式记录 `split_role=official_test`、`official_test_included=true`；
- 记录 official-test split SHA、checkpoint SHA、commit、seed、variant、input contract、metric geometry 和命令；
- 输出 mIoU/mAcc/mF1、per-class 指标、样本数和环境身份；
- 不扫描周期 checkpoint，不运行多种 geometry 后挑最好；
- 纯基础设施失败可在身份不变时重试；修复影响数值定义时，全部 allowlist checkpoint 使用新 evaluator version 统一重评并保留旧失败史。

Test 结果不得反向改变任何开发或训练选择。

## 6. 正式统计与声明

报告必须包含：

- B0 与每个 variant 的逐 seed 指标；
- 每个 seed 的 `variant - B0` paired Δ；
- 各臂 mean、sample std、min/max，以及 paired Δ 的 mean/sample std；
- per-class、参数量、训练/推理吞吐、峰值显存、耗时和恢复史；
- current development seed 1 的 `52.84` 只作历史开发参考，不混入正式统计；
- 论文 DFormer-S 59.74 只作外部参考，并说明模型与协议差异。

只有 paired 三 seed 与预登记 official-test 结果支持，才能形成正式模型改进结论。方向不稳定、成本不可接受或 test 退化时按事实报告，不挑最好 seed。

## 7. 完成标准

- formal B0 与全部最终 variant 的 paired 三 seed 证据闭合；
- Gate F 有用户批准记录；
- official test 仅按 allowlist 一次性评估且未参与选择；
- 正式报告和机器可读汇总可重算；
- 所有大产物、数据、checkpoint 和凭据留在 Git 外；
- 最终提交/发布状态同步到实时状态入口，远端推送仍需用户确认。
