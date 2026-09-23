# MMFR E1 Batch 1A C0 / F-lite 十条件 Main-Val 阶段工作汇报

- 汇报周期：2026-09-22 至 2026-09-23
- 报告对象：上级审计（研究结论与处置裁决）
- 证据边界：冻结评测器 `msflip-whole-original-grid-v1`、十条件、318 条 `val-dev` 全量、本地单机推理；不含 official test
- 当前状态：C0 与 F-lite 两侧各 `10/10` 条件完成且冻结身份一致；十视图口径下 F-lite 未复现 Quick-Val 优势，主指标 $M_6$ 为 `55.2883`（C0）对 `54.8917`（F-lite），差 `-0.3966` 个百分点；本轮**没有**预注册的 Main-Val 数值门禁，因此不产生 `promote`/`stop` 判定，处置权在上级

---

## 一、本阶段工作概述

**大白话结论：** 两个候选模型（C0 与 F-lite）的十条件正式评价已经在本地跑完，十个条件、两种模型全部有结果，没有任何一个条件是缺失或半途中断的。结果是：在十视图这个更严格、也更有代表性的口径下，F-lite 没有像之前的单视图筛查那样领先，反而在主指标（六类单故障的宏平均 mIoU，记作 $M_6$）上比 C0 低 `0.40` 个百分点，在干净条件（clean）上低 `0.67` 个百分点；十个条件里只有两个对 F-lite 有利。这不等于说 F-lite 变差了，因为本轮只有一个 seed、一个 checkpoint，也没有做置信区间或显著性检验，而且本轮**事先没有为 Main-Val 定过任何数值门槛**，所以这些数字是描述性的，不能直接翻译成“该保留”或“该放弃”。具体怎么处置，需要上级判断。

**精确概述：**

1. **运行完成性。** 在本地 `NVIDIA GeForce RTX 5060 Laptop GPU` 上，使用冻结命令与冻结评测器对 C0、F-lite 的 fixed final checkpoint 各跑完 10 个条件 × 318 条 `val-dev`，即每侧 `3180` 个 `(sample, condition)` 单元、每单元 10 个 view，每侧 `31,800` 次前向单元。两侧 `summary.json` 的 `coverage.is_full_ten_condition_full_val_dev` 均为 `true`，每个条件的 `metrics.json` 均为 `completed=true`。
2. **身份与配对性。** 两侧 `runner`、`protocol` 原文、条件定义、样本集合、条件顺序、视图分批、evaluation seed、PyTorch 版本与 TF32 开关逐项一致；每个条件 `metrics.json` 的 `identity_sha256` 与其 `run_manifest.json` 中记录的 expected 值完全相同；两侧同一条件的 corrupted Depth 聚合 SHA-256 完全相同，标签像素支持同为 `155,829,149`，说明比较是真正配对的（跑的是同一批损坏输入、同一批标签）。断言检查共 `0` 项失败。
3. **主结果。** 主指标（`unweighted-macro-mean-single-condition-mIoU`，六个单故障条件的未加权宏平均 mIoU）为 C0 `55.2883`、F-lite `54.8917`，即 $\Delta_F=-0.3966$ pp。clean 为 `56.69` 对 `56.02`（`-0.67` pp）。十个条件中 F-lite 仅在 `spatial_dropout@0.75`（`+0.40` pp）与 `spatial_dropout@0.5+gaussian_noise@0.5`（`+0.42` pp）上更好。
4. **与 Quick-Val 的关系。** 早前 4-condition Quick-Val（单视图 `original-full`）判定 F-lite `promote`（$\Delta_F=+0.96$ pp）。本轮十视图 Main-Val 覆盖其中 4 个条件，结果是 3 个符号翻转：clean `+0.69 → -0.67`、`entire_missing@1.0` `+1.41 → -0.63`、`misalignment@0.75` `+0.48 → -0.33`；只有 `spatial_dropout@0.75` 同为正值（`+0.99 → +0.40`）。**两套口径的绝对数字不可互相比较**（冻结文本已明确禁止），但符号方向的变化本身说明：Quick-Val 的单视图筛查增益没有在十视图口径下复现。
5. **本轮没有做的事。** 没有训练、没有重选 checkpoint、没有修改评测代码/协议/数据、没有跑 R-OE、T、Batch 1B、Batch 2，也没有读取 official test。official test 继续为 `sealed_unread`。
6. **不能据此声称的事。** 由于单 seed、单 checkpoint、无 bootstrap/置信区间、无 Main-Val 预注册数值门禁，且两个训练 run 的 DataLoader shuffle 顺序存在已接受的 screening-level 差异，本轮数字**不能**支持“F-lite 造成实质性退化”这一因果或统计结论；同样也**不能**支持“C0 更好”。

---

## 二、关键概念与判断边界

| 术语 | 技术定义 | 大白话说明 | 对本报告结论的影响 |
|---|---|---|---|
| `msflip-whole-original-grid-v1` | 冻结主评测器：整幅图、5 个尺度（`0.5/0.75/1.0/1.25/1.5`）× 原图与水平翻转，共 10 个 view；FP32 平均 pre-softmax logits 后恢复到 MUSeg 原始 Label 网格计分 | 同一张图由 10 种“看的方式”各推理一次，把结果合并后再打分 | 这是本轮的正式口径；它与 Quick-Val 的单视图口径是两套口径，数字不可互相比较 |
| `M6`（`unweighted-macro-mean-single-condition-mIoU`） | 六个单故障条件（`spatial_dropout@0.75`、`gaussian_noise@0.75`、`blur@0.75`、`quantization@0.75`、`misalignment@0.75`、`entire_missing@1.0`）各自 mIoU 的未加权宏平均 | 把六种“坏 Depth”场景各自的成绩简单平均，避免某一种场景因像素多而主导 | 这是冻结的主指标；本轮 $\Delta_F=-0.3966$ pp 就发生在它上面 |
| unit（单元） | 一个 `(sample, condition)` 组合，内部包含 10 个 view 的前向与融合 | 一次“某张图 × 某种故障”的完整评测 | 每侧 3180 个单元，是耗时与进度统计的单位 |
| `condition-major` | 编排顺序：先跑完一个条件的 318 条样本，再进入下一个条件 | 一种故障一批全跑完，再换下一种故障 | 使单个条件的结果可以独立、原子地写出，是暂停/恢复能保住已完成条件的前提 |
| `forward-rng reset-per-unit` | 每个单元前把 torch RNG 回放到同一 base 种子，因为 Ham decoder 的 NMF 初始化本身含随机抽样 | 每个单元都用同一份随机数，保证结果与调用顺序无关 | 这是两侧可配对、可复现的前提 |
| official test `sealed_unread` | 项目封存的官方测试划分，任何情况下未经独立门禁不得读取 | 最终考卷还在封条里，没人看过 | 本轮所有数字都只来自开发集 `val-dev` |
| pp（百分点） | 百分比度量的绝对差，例如 `56.69 → 56.02` 记作 `-0.67` pp | 直接相减的百分点差，不是相对变化 | 本报告所有 delta 单位均为 pp |

---

## 三、主要工作进展

### 3.1 十条件 Main-Val 运行完成

- **状态：** 已完成并验证
- **问题：** Batch 1A 需要一个与冻结主评测器口径一致、覆盖全部十条件、覆盖全部 `318` 条 `val-dev` 的正式对照评价，用来回答“F-lite 的轻量适配是否值得继续”，而不是只看单视图筛查。
- **措施：** 按冻结命令与冻结参数在本地对两个 fixed final checkpoint 顺序执行十条件评价：`--split-role val_dev`、`--resume`、`--order condition-major`、`--view-batching 2`（按冻结的 per-scale 有效分批 `{0.5:2, 其余:1}`）、`--forward-rng reset-per-unit`、`--device cuda`。C0 完成后再启动 F-lite，两侧不并行。
- **原理与取舍：** 主指标保持 $M_6$ 不变，同时报告 clean、完整 10 条件表、最坏单条件、逐类 IoU 与成本；**不使用**更好看的十条件平均替代原主指标。`condition-major` 与 `reset-per-unit` 是暂停可恢复、结果与调用顺序无关的前提。
- **结果与证据：** 两侧 `coverage.conditions_completed` 与 `conditions_requested` 的并集均为 10 个条件，`conditions_present` 与之一致；每条件 `completed=true`、`sample_count=318`；`stored_sample_counts=[318]`、`stored_sample_set_sha256` 唯一，说明 10 个条件文件来自同一样本集合。证据：两侧 `update-2560/summary.json` 与 10 个 `update-2560/<condition>/metrics.json`（路径见第八节）。
- **后续：** 若上级要求更强的统计判断，需要另立预注册实验（见第七节）。

### 3.2 运行中断与恢复的如实记录

- **状态：** 已完成并验证
- **问题：** C0 第一次运行在写完前 6 个条件后被用户要求暂停，需要确认暂停没有损坏已完成的证据，且恢复时不会重算或漏算。
- **措施：** 暂停时按用户裁决强制终止进程；随后用**同一冻结命令**加 `--resume` 恢复，仅重算缺失条件。
- **结果与证据：** C0 `run_manifest.json` 的 `resume.decisions` 逐条记录：`clean`、`spatial_dropout_075`、`gaussian_noise_075`、`blur_075`、`quantization_075`、`misalignment_075` 六个条件的判定均为 `skipped`，理由是与 expected identity `完成且身份匹配`；`entire_missing_100`、`spatial_dropout_050__gaussian_noise_050`、`blur_050__misalignment_050`、`quantization_050__misalignment_050` 四个条件为 `pending` 并被重新计算。C0 恢复会话 `duration_seconds=6775.076`（约 1.88 小时，只覆盖后 4 个条件）；全部 10 个条件的 `identity_sha256` 均等于 manifest 记录的 expected 值，因此“跳过”与“重算”两条路径产出的结果处在同一身份契约下。
- **后续：** 无。该项是边界说明，不影响结果解释。

### 3.3 冻结身份与配对性断言

- **状态：** 已完成并验证
- **问题：** 两个候选的比较只有在输入、评测器、条件定义、样本集合与数值精度完全一致时才有意义。
- **措施：** 对两侧 `run_manifest.json`、`summary.json` 与全部 20 个 `metrics.json` 做程序化断言。
- **结果与证据（`0` 项失败）：**

| 检查项 | C0 | F-lite | 结论 |
|---|---|---|---|
| 条件文件数 | `10/10` | `10/10` | 通过 |
| `completed` | 全部 `true` | 全部 `true` | 通过 |
| `official_test_included` | `false`（manifest/summary/每个条件） | `false`（同） | 通过 |
| checkpoint SHA-256 | `ca618b23…d9a1a` | `ea9319e5…abd98d` | 两侧各自一致，且为预期的两个不同 checkpoint |
| split SHA-256 | `1d0719d8…0dd0e83` | 同 | 通过 |
| runner SHA-256 | `1c6736a67939fc8b7de571396a6ce113c02daf0726cc981dac3f85b55b9bec7a` | 同 | 通过 |
| protocol 原文 SHA-256 | `814da56321fc5c4e0118b762177877553dfa3ed71e8f72721cb3d65764849da3` | 同 | 通过 |
| 条件定义 SHA-256 | `6b4b129dd72a6976bf069655ffd77cd59ec63218b46a5bceea40043d606d4225` | 同 | 通过 |
| 样本集合 SHA-256 | `27001f07…e53f42` | 同 | 通过 |
| evaluation seed / 视图数 / 分批 | `2026091401` / `10` / `{0.5:2, 其余:1}` | 同 | 通过 |
| FP32 / TF32 off | `forward_precision=fp32`、`tf32_enabled=false` | 同 | 通过 |
| 环境 | `torch 2.7.0+cu128`、`cuda 12.8`、`cudnn 90701`、Python `3.13.9` | 同 | 通过 |
| 主指标自洽 | `protocol_primary_score.value=55.2883` 等于六个单条件 mIoU 重算值 | `54.8917` 等于重算值 | 通过 |
| 每条件 `identity_sha256` 与 manifest expected 一致 | `10/10` | `10/10` | 通过 |
| 配对性：同条件 corrupted Depth 聚合 SHA-256 | 两侧逐条件相同（10/10） | — | 通过 |
| 标签像素支持 | `155,829,149` | `155,829,149` | 通过 |

冻结条件契约在运行时被硬校验：两个 manifest 的 `protocol.frozen_hard_check` 均为 `status: PASS` 且 `mismatches: []`，即实际解析出的 evaluator、六个单条件、三个混合条件、主指标名、样本数 `318` 与 seed `2026091401` 与冻结文本逐项相同。

### 3.4 主结果：完整十条件表

mIoU / mAcc / mF1 均为百分比（`%`），delta 为 F-lite 减 C0 的百分点（pp）。负值表示 F-lite 更低。

| # | condition | C0 mIoU | F-lite mIoU | Δ mIoU | C0 mAcc | F-lite mAcc | Δ mAcc | C0 mF1 | F-lite mF1 | Δ mF1 |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | clean | 56.69 | 56.02 | **-0.67** | 69.53 | 69.09 | -0.44 | 71.02 | 70.36 | -0.66 |
| 1 | `spatial_dropout@0.75` | 53.92 | 54.32 | **+0.40** | 68.37 | 68.52 | +0.15 | 68.83 | 69.14 | +0.31 |
| 2 | `gaussian_noise@0.75` | 56.34 | 55.85 | -0.49 | 69.33 | 69.14 | -0.19 | 70.77 | 70.35 | -0.42 |
| 3 | `blur@0.75` | 56.65 | 55.95 | -0.70 | 69.50 | 69.04 | -0.46 | 70.97 | 70.30 | -0.67 |
| 4 | `quantization@0.75` | 56.62 | 55.99 | -0.63 | 69.47 | 69.06 | -0.41 | 70.95 | 70.33 | -0.62 |
| 5 | `misalignment@0.75` | 55.57 | 55.24 | -0.33 | 69.03 | 68.60 | -0.43 | 70.18 | 69.75 | -0.43 |
| 6 | `entire_missing@1.0` | 52.63 | 52.00 | -0.63 | 67.15 | 66.35 | -0.80 | 67.75 | 67.04 | -0.71 |
| 7 | `spatial_dropout@0.5+gaussian_noise@0.5` | 54.94 | 55.36 | **+0.42** | 68.83 | 69.08 | +0.25 | 69.72 | 70.02 | +0.30 |
| 8 | `blur@0.5+misalignment@0.5` | 56.14 | 55.33 | -0.81 | 69.14 | 68.63 | -0.51 | 70.56 | 69.75 | -0.81 |
| 9 | `quantization@0.5+misalignment@0.5` | 56.00 | 55.37 | -0.63 | 69.32 | 68.63 | -0.69 | 70.54 | 69.81 | -0.73 |

十个条件中 F-lite 在 mIoU 上更好的有 2 个（第 1、7 行），其余 8 个更低。单条件最大正差为 `+0.42` pp（第 7 行），最大负差为 `-0.81` pp（第 8 行）。两列 delta 的绝对量级集中在 `±0.4 ~ ±0.8` pp 之间。

### 3.5 主结果：汇总量与判读

| 汇总量 | C0 | F-lite | Δ (pp) |
|---|---|---|---|
| **$M_6$（六个单故障宏平均 mIoU，主指标）** | **55.2883** | **54.8917** | **-0.3966** |
| clean mIoU | 56.69 | 56.02 | -0.67 |
| 三个混合条件宏平均 mIoU | 55.6933 | 55.3533 | -0.34 |
| 九个受损条件宏平均 mIoU | 55.4233 | 55.0456 | -0.3777 |
| 十个条件宏平均 mIoU | 55.55 | 55.143 | -0.407 |
| $M_6$ 的 mAcc | 68.8083 | 68.4517 | -0.3566 |
| $M_6$ 的 mF1 | 69.9083 | 69.485 | -0.4233 |
| 最坏单条件 mIoU | `entire_missing@1.0` = 52.63 | `entire_missing@1.0` = 52.00 | -0.63 |

按冻结文本，$M_6$ 是唯一主指标，其余为必须一并报告的描述性量（clean、十条件表、最坏单条件、逐类 IoU、成本）。**本轮没有为 Main-Val 预注册任何数值门槛**：冻结的 `promote`/`stop` 门槛只属于 Quick-Val，且已在 2026-09-22 使用过。因此上表只描述“十视图口径下 F-lite 未显示优势”，不构成处置判定。

### 3.6 逐类 IoU（15 类，clean 与最坏条件）

逐类 IoU 在未加权宏平均下与 mIoU 同权，因此少量稀有类的剧烈变化会显著影响 mIoU。下表给出 clean 与 `entire_missing@1.0` 两个条件的逐类 IoU（`%`）。

| idx | class | clean C0 | clean F-lite | Δ | EM C0 | EM F-lite | Δ |
|---|---|---|---|---|---|---|---|
| 0 | person | 52.82 | 59.10 | +6.28 | 51.59 | 55.46 | +3.87 |
| 1 | cable | 60.05 | 60.07 | +0.02 | 53.75 | 53.66 | -0.09 |
| 2 | tube | 67.92 | 66.33 | -1.59 | 60.28 | 59.53 | -0.75 |
| 3 | indicator | 73.61 | 72.59 | -1.02 | 73.32 | 72.70 | -0.62 |
| 4 | metal fixture | 46.10 | 43.95 | -2.15 | 45.53 | 41.83 | -3.70 |
| 5 | container | 25.55 | 25.71 | +0.16 | 24.86 | 25.04 | +0.18 |
| 6 | tools & materials | 72.59 | 72.91 | +0.32 | 68.84 | 69.61 | +0.77 |
| 7 | door | 75.18 | 74.24 | -0.94 | 61.60 | 61.74 | +0.14 |
| 8 | electrical equipment | 62.63 | 66.35 | +3.72 | 60.90 | 64.51 | +3.61 |
| 9 | electronic equipment | 55.08 | 48.89 | -6.19 | 46.40 | 44.20 | -2.20 |
| 10 | mining equipment | 30.79 | 33.38 | +2.59 | 27.50 | 31.83 | +4.33 |
| 11 | anchoring equipment | 59.33 | 58.85 | -0.48 | 58.12 | 57.83 | -0.29 |
| 12 | support equipment | 45.64 | 35.01 | **-10.63** | 43.23 | 29.64 | **-13.59** |
| 13 | rescue equipment | 43.82 | 43.58 | -0.24 | 40.87 | 39.97 | -0.90 |
| 14 | rail area | 79.31 | 79.32 | +0.01 | 72.63 | 72.41 | -0.22 |

**可核验的分解：** 15 个类别等权，contributions 之和除以 15 必须等于 mIoU 差。clean 上正贡献之和 `+13.10`、负贡献之和 `-23.24`，净值 `-10.14`，除以 15 得 `-0.676`，与报告的 `-0.67` pp 一致；`entire_missing@1.0` 上正 `+12.90`、负 `-22.36`，净值 `-9.46`，除以 15 得 `-0.631`，与 `-0.63` pp 一致。也就是说，clean 的 `-0.67` pp 中约 `-0.71` pp 单独来自 `support equipment` 一类（该类标签像素仅 `1,501,030`，约占 1%），其余类别净贡献约 `+0.03` pp。这是一个事实观察，不是因果结论：本轮未做逐类显著性检验，逐类指标也未预注册为门禁。

### 3.7 成本与资源

| 项目 | C0 | F-lite |
|---|---|---|
| 逐条件耗时之和 | `16,323.43` s（约 4.53 h） | `15,883.17` s（约 4.41 h） |
| 恢复会话 `run.duration_seconds` | `6,775.076` s（仅后 4 个条件） | `16,080.757` s（全部 10 个条件） |
| 单条件耗时范围 | `1,558.64` ~ `1,776.44` s | `1,573.81` ~ `1,605.85` s |
| 单元平均耗时 `mean_unit_seconds`（clean） | `4.9998` s / `(sample, condition)` | `4.9491` s / `(sample, condition)` |
| 峰值显存（clean，allocated / reserved） | `4700.619` / `6578.0` MiB | `4701.280` / `6580.0` MiB |
| 设备 / 精度 | RTX 5060 Laptop / FP32、TF32 off | 同 |

两侧的单元平均耗时差约 `1.0%`，与 F-lite 适配器带来的推理开销量级一致（Gate-B 记录 batch-1 推理中位增量 `+0.78%`），未观察到异常放大。C0 的逐条件耗时之和（`4.53 h`）大于其恢复会话时长，是因为其中前 6 个条件在暂停前完成。

---

## 四、实验与验证结果

### 4.1 配置与身份

- **数据集与划分：** MUSeg `val-dev` 全量 `318` 条，split SHA-256 `1d0719d8f64f016d48995c25ab66d4004d76b7155d9efeef7cbb7454c0dd0e83`；official test 未参与。
- **模型与 checkpoint：** C0 `update-2560.pth` SHA-256 `ca618b23d18eabb201a0d11d18da383ac99576d0feae5864e3233bda527d9a1a`；F-lite `update-2560.pth` SHA-256 `ea9319e5abe55b996470ee0a75bd63b887241ef834a3b50f145b5b7d4aabd98d`。两者均为 Batch 1A 正式训练（各 `2560/2560` successful updates、`skipped=0`）产出的 fixed final checkpoint，本轮未做任何 checkpoint 选择。
- **配置模块：** `local_configs.MUSeg.DFormerv2_S_MMFR_E1_Batch1A_C0` 与 `local_configs.MUSeg.DFormerv2_S_MMFR_E1_Batch1A_FLite`（两侧唯一差异为候选身份本身）。
- **评测器与条件契约：** runner `tools/evaluate_museg_10condition.py`（SHA-256 `1c6736a67939fc8b7de571396a6ce113c02daf0726cc981dac3f85b55b9bec7a`）；协议文件 `protocols/mmfr-a2-train-integration-v3.template.json`（原文 SHA-256 `814da56321fc5c4e0118b762177877553dfa3ed71e8f72721cb3d65764849da3`），其 `frozen_hard_check` 内嵌 Batch 1A 的十条件字面契约并在运行时通过（`status: PASS`、`mismatches: []`）。
- **评测口径：** 整幅图、尺度 `[0.5, 0.75, 1.0, 1.25, 1.5]` × {原图, 水平翻转} 共 `10` view，`pad_divisor=32`，`forward_precision=fp32`、`logits_fusion_precision=fp32`、`amp=false`、`tf32_enabled=false`，corruption 只作用于 Depth（RGB 无 corruption），有效性定义为 `(raw Depth uint8 > 0) & validity_mask`，`validity_mask` 为原始对齐网格上的全 1（无 crop、无 pad）。
- **可分复现性设置：** corruption 使用 `numpy.PCG64 + SeedSequence`，seed words 为 `evaluation_seed=2026091401`、`condition_index_0_based`、`sample_id` SHA-256 前四个 big-endian `uint32`；`shared_progressing_generator=false`；模型前向使用 `reset-per-unit` 回放 base RNG（原因：`models/decoders/ham_head.py` 的 `NMF2D` 含 `torch.rand`，不做控制时每次前向都随机）。
- **环境：** Windows `10.0.26200`、Python `3.13.9`、`torch 2.7.0+cu128`、CUDA `12.8`、cuDNN `90701`、`numpy 2.3.5`、`NVIDIA GeForce RTX 5060 Laptop GPU`。

### 4.2 验证状态与证据完整性

- **已验证（直接证据）：** 完成性、两侧身份一致性、配对性、主指标自洽、冻结条件契约硬校验，全部由上表断言覆盖，`0` 项失败。
- **未被保留的证据（如实记录）：** 本轮 Main-Val 的 evaluator **stdout 与进程退出码没有落盘为文件**；当前工作区只保留训练期的 `C0/train.log` 与 `FLite/train.log`，终端会话记录已不在本机保留。因此“运行完成”的证据是**产物完整性 + 身份一致性**（两侧 `summary.json` + 20 个 `metrics.json` 齐全、覆盖与身份全部通过），而不是退出码记录。这一点足以支持“结果完整且身份正确”，但弱于“有退出码 + 完整日志”的证据等级。
- **未运行的高成本检查：** 未做 bootstrap 或配对显著性检验、未做 location-group 重采样、未做多 seed 复现、未读取 official test。上述检查按项目分级验证预算，需要另行授权。

---

## 五、当前问题与风险

1. **没有 Main-Val 的预注册数值门禁。** 冻结文本只为 Quick-Val 定义了 `promote`/`stop` 门槛，并已在 2026-09-22 使用；Main-Val 的角色是产出完整十条件表与成本，不是再裁决一次。因此本轮结果的性质是**描述性证据**，任何“保留/放弃 F-lite”的判断都需要上级显式裁决，不能在结果产生后补写门槛。
2. **统计强度不足。** 单 seed、单 checkpoint、单次运行，$M_6$ 的 `-0.3966` pp 与 clean 的 `-0.67` pp 都没有置信区间，也没有运行间噪声估计。因此**无法区分真实差异与运行间波动**；本轮不声称因果。
3. **Quick-Val 与 Main-Val 口径不同且方向翻转。** 4 个共同条件中有 3 个符号翻转（clean、`entire_missing@1.0`、`misalignment@0.75`），只有 `spatial_dropout@0.75` 保持正值。冻结文本明确禁止跨口径比较绝对数字，但符号不一致本身说明：单视图筛查的增益没有自动迁移到十视图口径。这属于本轮最重要的解释边界。
4. **训练期 DataLoader shuffle 顺序差异（已接受，未消除）。** 两个训练 run 的样本-槽位排列不同（C0 合计 clean `6359/25600`、F-lite `6369/25600`），原因是 F-lite adapter 初始化额外消耗全局 torch RNG。用户已裁决接受为 screening-level 随机性、不重跑、不改采样器。该差异会影响 $\Delta_F$ 的解读粒度。
5. **逐类变化高度不均衡，且未做多重比较校正。** clean 的 mIoU 差几乎全部由 `support equipment`（`-10.63` pp，占约 `-0.71` pp）和 `electronic equipment`（`-6.19` pp）等少数类驱动；同时 `person`（`+6.28` pp）、`electrical equipment`（`+3.72` pp）、`mining equipment`（`+2.59` pp）方向相反。未做逐类检验，逐类指标也未预注册为门禁，因此逐类结论只能作为观察。
6. **C0 的十个条件跨两次会话完成。** 前 6 个条件与恢复会话之间存在设备状态差异；缓解证据是同条件 corrupted Depth 哈希一致、全部 identity 一致、精度与环境一致，但“单次连续会话”这一更强条件未被满足。
7. **历史 selector 与主评测排序不一致的先例。** 项目历史上出现过 selector 排序与主评测排序不同的情况（Quick-B0 中 selector 第一的 epoch 480 在主评测只排第三）。这提示“不同口径下的排序不一定稳定”，本轮 Quick-Val 与 Main-Val 的方向差异属于同类现象，需要在解释时保留。

---

## 六、下一步计划

1. **由上级裁决本轮 Main-Val 的处置。** 明确本轮十视图结果是（a）作为描述性证据保留并维持 Quick-Val 的 `promote`，还是（b）触发新的预注册确认实验，或（c）判定为需要更强证据后再决定。裁决前不启动任何新的研究分支。
2. **若要判断 `-0.3966` pp 是否为真实差异，另立预注册实验。** 冻结的检查方式应包含：同一 checkpoint 的重复运行以估计十视图逐条件运行间波动，以及对逐样本配对差异使用冻结的 location-group 单位做 bootstrap；估计量、区间规则与裁决门槛必须在查看新结果之前写入 protocol，并需用户单独授权。
3. **核查 Quick-Val 与 Main-Val 的口径差来源。** 用同一 checkpoint 在单视图 `original-full` 与十视图 `msflip-whole-original-grid-v1` 两个口径下分别复算 4 个共同条件，确认单视图增益是否只存在于单视图口径；该核查属于诊断，不构成新的效果结论。
4. **在获得授权前保持停止状态。** 不实现 R-OE、不运行 T、不启动 Batch 1B/Batch 2、不重选 checkpoint、不读取 official test；official test 继续为 `sealed_unread`。

---

## 七、复现或交接说明

**输出根目录：**

- C0：`cloud/MMFR_E1_Batch1A_local_transfer_20260922/MMFR_E1_Batch1A_local_transfer_20260922/C0/mainval-10cond-v1/`
- F-lite：`cloud/MMFR_E1_Batch1A_local_transfer_20260922/MMFR_E1_Batch1A_local_transfer_20260922/FLite/mainval-10cond-v1/`

每个目录下的证据结构为：`run_manifest.json`、`update-2560/summary.json`，以及 `update-2560/<condition>/metrics.json` 共 10 份，条件目录名为 `clean`、`spatial_dropout_075`、`gaussian_noise_075`、`blur_075`、`quantization_075`、`misalignment_075`、`entire_missing_100`、`spatial_dropout_050__gaussian_noise_050`、`blur_050__misalignment_050`、`quantization_050__misalignment_050`。这些目录位于仓库外，不进入 Git。

**C0 命令：**

```text
python -m tools.evaluate_museg_10condition --checkpoint "cloud\...\C0\checkpoint\update-2560.pth" --split "data\splits\MUSeg\dev-v1\val-dev.txt" --split-role val_dev --expected-checkpoint-sha256 ca618b23d18eabb201a0d11d18da383ac99576d0feae5864e3233bda527d9a1a --expected-split-sha256 1d0719d8f64f016d48995c25ab66d4004d76b7155d9efeef7cbb7454c0dd0e83 --protocol "protocols\mmfr-a2-train-integration-v3.template.json" --config local_configs.MUSeg.DFormerv2_S_MMFR_E1_Batch1A_C0 --dataset-root "D:\0Project\dataset\MUSeg_DFormer" --device cuda --output-dir "cloud\...\C0\mainval-10cond-v1" --resume --order condition-major --view-batching 2 --forward-rng reset-per-unit
```

F-lite 使用完全相同参数，仅替换 checkpoint 路径与 `--expected-checkpoint-sha256`（`ea9319e5…abd98d`）、`--config local_configs.MUSeg.DFormerv2_S_MMFR_E1_Batch1A_FLite`、`--output-dir` 指向 F-lite 目录。

**环境变量：** 运行前需 `PYTHONPATH=D:\0Project\DFormer`，并设置 `KMP_DUPLICATE_LIB_OK=TRUE`。

**复核入口：** 主指标与分组均值见两侧 `update-2560/summary.json` 的 `protocol_primary_score`、`groups`、`mean_over_all_conditions`；恢复决策见两侧 `run_manifest.json` 的 `resume.decisions`；逐类 IoU 与混淆矩阵见每个条件的 `metrics.json` 的 `metrics_percent.per_class` 与 `metrics_percent.confusion_matrix`。

**关联材料：** 冻结筛选计划与协议位于 `D:/0Project/DFormer-archive-20260922/liu-test-exp/MMFR/MMFR_v4_1_blueprint_and_reference_package_2026-09-20/01_research/e1_screening_plan.md` 与同目录 `e1_batch1_protocol.md`；实时状态入口为 `doc/main/MUSeg-current-status.md`。

---

## 八、本阶段未执行项（按分级验证预算说明）

- 未运行完整测试套件、全仓扫描或与本任务无关的检查；
- 未训练、未重跑 Quick-Val、未重选 checkpoint；
- 未做 bootstrap、配对显著性检验、多 seed 复现或逐类检验；
- 未运行 R-OE、T、Batch 1B、Batch 2；
- 未读取或解封 official test（保持 `sealed_unread`）；
- 本次为文档与证据复核类交付，除上述断言脚本外未产生新的计算任务，断言脚本为一次性内联命令，未留在工作区。
