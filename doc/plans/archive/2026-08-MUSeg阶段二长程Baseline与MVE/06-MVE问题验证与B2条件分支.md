# 06：MVE 问题验证与 B2 条件分支

> **后继候选计划：** `doc/plans/deferred/2026-09-MUSeg-unexecuted/MUSeg-A2-B2深度有效性/00-总方向规划.md`（该后继计划现已延期，未执行）。

> **文档角色：** 条件式研究计划，不承担实时状态或运行授权。
> **形成或核验时点：** 2026-08-27。
> **实时入口：** `doc/main/MUSeg-current-status.md`；A2 自然证据边界见 `doc/main/MUSeg-open-decisions.md`。
> **后继关系：** 合并并取代旧 Stage-07 至 Stage-11；若 B2 通过本阶段，候选必须回到新 Stage-07 使用统一 screening 规则。

## 1. B1 固定职责

B1 `safe_masked_mean` 是基础链路的数值稳定性修复，不是性能模块。已有 tensor、真实 CUDA mixed/all-background 与 development seed 1 使用证据作为权威历史指针；未来每次相关改动必须运行 B1 回归。不得安排“去掉 B1”的性能长训，也不得把 B1 与 B2 混称一个模块。

## 2. A2 前置与矩阵

A2 只在 Protocol Gate 冻结 channel order、normalization、validation geometry、evaluator 和 `development-reference-B0` 后运行；只使用 val-dev，official test 明确拒绝。

预注册矩阵至少包括：

- 全 val-dev（资源预审可固定 64 图，但不能替代正式矩阵）；
- `q=0,0.1,0.3,0.5`；
- block/random mask；zero/图内有效深度中位数 fill；
- 至少两个 mask seeds；
- 至少一个冻结 model checkpoint，可选第二 checkpoint 只能增强开发证据；
- 注入区域只来自原始 `Depth16 > 0`，zero/median 对同 sample/q/seed 复用同一 mask。

manifest 记录 model/checkpoint SHA、mask seed/type/fill/q、实际注入数、input contract、metric geometry、split/tool commit 和 condition ID。q=0 必须先与 baseline evaluator 等价；失败即停止。

## 3. A2 聚合与判据

聚合单位是“每个 model checkpoint × mask seed 条件下，跨 val-dev 样本得到的 foreground mIoU 和同图 paired Δ”；先保留条件级结果，再对条件做等权汇总。方向一致定义为所有预注册 mask seeds 的聚合 Δ 同号；若有第二 model checkpoint，其聚合方向也必须同号。浮点复算容差为绝对 `1e-6`；门槛比较按未四舍五入值执行，恰好等于阈值视为满足。

### `A2-pass`

必须同时满足：

1. `q=0.3 block zero` 相对 q=0 的平均 foreground mIoU 下降至少 2.0 个百分点；
2. 至少 75% 图像的同图 paired Δ 为负；
3. q=0.1→0.3→0.5 的条件均值总体不升；相邻反弹超过 0.2 个百分点视为破坏剂量趋势；
4. 所有预注册 mask seeds 方向一致；有第二 model checkpoint 时 model 方向也一致；
5. zero 相对 median 负对照的额外退化门槛必须在 A2 protocol 运行前冻结，不能看结果后补定。

单 model checkpoint 通过时只能声明“单 checkpoint 工程证据”，不能声称跨 model-seed 稳定。

### `A2-stop`

满足任一项：

- q=0.5 平均下降不足 0.5 个百分点；
- 剂量趋势不成立且超过上面的 0.2 容差；
- 效应只存在于单个 mask seed 或已预注册的 model checkpoint；
- zero/median 不支持预注册的无效零值机制；
- 完整性、q=0 等价或 official-test 封存失败。

### `A2-inconclusive`

门槛附近、置信区间跨阈值、缺少预注册条件或仅部分 mask 类型成立时记为 inconclusive。最多提出一次新 protocol 的受限补充实验；未补齐前不得进入 B2。

自然无效深度分层不足不单独把 `A2-pass` 改为 stop，但结论只能限定在人工 corruption，不能声称现实缺失机制、部署鲁棒性或收益。

## 4. MVE Evidence Gate

只有 `A2-pass + 用户明确批准` 才能开始 B2 数学/shape 规格或生产实现。`A2-stop` 终止本轮 B2；`A2-inconclusive` 保持阻塞。该门禁替代旧 A2-G、N-G 和 Gate G 命名。

## 5. B2 规格与金标准

B2 目标：两个 depth patch 都有效时保留原 depth decay；任一无效时只移除该 pair 的 depth decay，保留 positional decay；`validity=None` 和 all-valid 在容差内等价 B0，不新增参数或 state-dict key。

实现前必须冻结：

- validity 来源为量化前 `Depth16 > 0`；与 RGB/Depth/Label 同步 scale/crop/flip，nearest resize，padding invalid；
- 每个 stage 的 validity 下采样语义；
- full GSA 与 decomposed GSA 的 batch/head/H/W shape、pair mask、transpose、dtype/device/AMP；
- full 2×2、decomposed 非方形 2×3、batch=2、多 head、all-valid/all-invalid/mixed 的手算金标准；
- strict checkpoint、参数量、forward/backward、legacy NYU/SUNRGBD/B1 回归。

规格仍有 shape、广播或下采样歧义时不得编码。

## 6. B2 实现与专用资格

按失败金标准测试→数据 validity→端到端可选参数→full GSA→decomposed GSA→strict compatibility→真实 forward/backward 顺序实现。核心 validity、pair mask 与 geometry prior 必须由主代理逐行复核。

一个 B2 专用 GPU qualification 统一承担真实 mixed/all-invalid forward/backward、AMP、显存、吞吐、checkpoint 和资源安全验证。它只证明链路与资源可用，不是效果证据，也不复用通用 Stage-04 的旧 batch 结论。

## 7. `B2-zero-train`

使用同一冻结 checkpoint，不更新权重，在相同 val-dev/A2 条件下比较 B0 与启用 B2：

- pass：q=0.3 block zero 恢复至少 1.0 mIoU，q=0 干净下降不超过 0.5，所有已预注册 model/mask seeds 方向一致，数值和资源可接受；
- stop：任一安全条件失败、恢复不足，或收益依赖单个已预注册 seed；
- inconclusive：结果在阈值容差附近或预注册条件不完整。

聚合单位、方向和阈值比较规则与 A2 相同；不得反复改 gating 追结果。

## 8. `B2-short`

zero-train pass 且用户批准后，使用独立 5-epoch protocol 和固定 20% train-dev location-group 子集；B0/B2 必须从相同初始化、同 seed、数据顺序、优化器、LR 与预算对称训练，不能用 checkpoint 微调一臂而另一臂从 pretrained 开始。

pass 必须满足至少一项：自然高无效率组提升 ≥1.0 mIoU，或 q=0.3 block zero 提升 ≥2.0；同时全量干净 val 下降 ≤0.5、方向不冲突、数值/显存/吞吐可接受。单 seed 只决定能否进入 Stage-07，不形成正式改进结论；可选第二 seed 必须两臂一起补。

## 9. 回接 Stage-07

`B2-short-pass` 只授予 B2 进入统一 module screening 的资格，结果名转为 `B2-screening`；不直接进入正式三 seed。`B2-short-stop/inconclusive` 不进入 Gate E 候选。所有 A2/B2 预测、mask、checkpoint 和日志保存在实验盘，不进 Git。
