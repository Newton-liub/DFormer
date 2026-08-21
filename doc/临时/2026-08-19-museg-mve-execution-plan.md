# MUSeg / DFormerv2 MVE 执行总计划

> 计划类型：阶段执行清单
> 状态更新：2026-08-21
> 适用范围：A1、B1、A2 筛查，以及条件触发后的 B2 方案验证
> 当前结论：A1/B1 已完成并验证；A2 已完成但未达到进入 B2 的预注册门槛；当前优先统一证据、完成正式 MUSeg 基线，再决定是否扩大 Depth=0 实验。

## 一、统一实验与代码基线

- 本机 `main`、GitHub `origin/main`、云端运行副本 `/root/DFormer`、云端数据盘仓库 `/root/rivermind-data/DFormer` 已统一到提交：
  `5c134092d0126eb0430415ee0cebe7b2b2a19ed1`。
- 本机下载的云端证据目录：`cloud/museg-epoch10-a2-20260821`，已加入 `.gitignore`，不上传 checkpoint、预测结果或数据。
- 云端实际训练和 A2 使用的代码副本是 `/root/DFormer`；长期保存和后续同步使用 `/root/rivermind-data/DFormer`。
- 云端当前实例为无卡运行状态（`GPU=0`），不在该状态启动 GPU 训练或推理。

## 二、问题—处理—证据逻辑链

### 2.1 A1：全背景样本导致空有效像素归约风险

**问题。** MUSeg 原始标签中存在全背景图；训练映射将背景设为 `255=ignore`。原始归约形式是对 `pixel_loss[target != 255]` 求均值。当某个本地 batch 或分布式 rank 没有有效像素时，均值作用于空张量，可能产生非有限值。

**验证。** `tests/test_masked_loss.py` 使用项目同款 `CrossEntropyLoss(reduction="none", ignore_index=255)`，比较有效像素数为 0、1、5、100 等条件。

**结果。** 旧归约在 0 个有效像素时非 finite；1 pixel、1%、100% 条件 finite。该结果只证明空集合归约风险，不把它归因于 Depth=0。

**结论。** A1-G：全 ignore 本地 batch 是确定性的数值稳定性风险，应修复。

### 2.2 B1：safe masked loss 修复

**处理。** 新增 `models/losses/safe_masked_loss.py` 中的 `safe_masked_mean`：

- 非空有效集合：返回原 masked mean；
- 空有效集合：返回 `values.sum() * 0.0`，保持计算图连接，loss 为有限 0，梯度为有限零梯度。

`models/builder.py` 已将 safe reduction 接入主头和辅助头路径。

**验证。** 本地执行：

```text
python -m unittest discover -s tests -p "test_*.py" -v
```

共 8 项测试通过，包括：旧实现空集合非 finite、safe loss 空集合有限零、非空 loss/gradient 与旧实现一致、辅助头路径有限。

**结论。** B1-G：该修复关闭了 A1 数值风险，且不改变非空有效像素 batch 的 loss 与梯度。当前仍可补做“真实模型 + 1 张全背景图 + 1 张普通图”的混合 batch 回归，但不影响本地张量级结论。

### 2.3 A2：Depth=0 是否构成实际性能瓶颈

**假设。** DFormerv2 的 geometry prior 使用深度差异；若把无效 `Depth=0` 当成真实深度参与几何关系，额外置零应造成可重复的性能退化。

**实验设计。** 固定 epoch-10 MUSeg checkpoint、官方 test 中 16 张含前景且有效深度比例较高的图、seed `20260819`、block mask、RGB/Label/resize/推理代码。只在原始 `Depth16 > 0` 像素中注入额外 0：

- `q=0`：不注入，baseline；
- `q=0.3`：约 30% 有效深度置零；
- `q=0.5`：约 50% 有效深度置零。

共 16 × 3 = 48 个样本—条件前向，使用 checkpoint：
`/root/rivermind-data/mve_outputs/museg_20epoch/checkpoint/epoch-10.pth`。

**结果。** 独立评估结果如下：

| 条件 | 前景 mIoU | 相对 q=0 |
|---|---:|---:|
| q=0 | 0.3389857875 | 0 |
| q=0.3 | 0.3378786890 | -0.0011070985（-0.1107 个百分点） |
| q=0.5 | 0.3356490187 | -0.0033367688（-0.3337 个百分点） |

结果在该 16 图 screening 上随 q 增加而单调下降，但效应较小。

**判定。** A2-N-G / 中间态：q=0.3 未达到预注册的至少 2.0 个百分点下降；q=0.5 下降小于 0.5 个百分点。因此当前证据不足以把 Depth=0 定性为主要独立瓶颈，也不足以触发 B2 validity mask/gating 开发。A2 只证明了轻微敏感性，不能证明问题已解决或已经构成另一类主要问题。

## 三、云端 MVE/A2 证据

- 训练输出：`/root/rivermind-data/mve_outputs/museg_20epoch`。
- A2 输出：`/root/rivermind-data/mve_outputs/museg_a2_screening`。
- epoch-10 验证 mIoU：`20.49`；该 checkpoint 不是完整 20 epoch 正式结果。
- A2 独立评估：`evaluation/per_image_metrics.csv`（49 行，含表头）和 `evaluation/summary.json`。
- 云端证据包：`/root/rivermind-data/exports/museg-epoch10-a2-evidence-20260821.tar.gz`。
- 本机证据包：`cloud/museg-epoch10-a2-20260821/museg-epoch10-a2-evidence-20260821.tar.gz`。
- 证据包 SHA-256：`4d5da8a26974124d2cd3979c5166f40e12dbeae9a3f412a7d026649db492519d`。
- 原始任务记录和失败纠正已保留在 `doc/临时/云/MUSeg-20epoch-A2云端执行流程.md`。

## 四、B2 validity mask/gating 当前状态

B2 尚未实现，原因是 A2 未达到进入门槛。计划中的 B2 是：从 `Depth16 > 0` 得到 validity mask，在每个 DFormerv2 stage 对齐到 depth patch 网格，构造 `pair_valid = valid_i & valid_j`；有效 pair 使用 depth decay，无效 pair 仅使用 positional decay，不改 decoder、不把无效深度对应 Label 设为 ignore、不使用深度补全伪造观测值。

只有在更大规模 A2 复核后仍观察到稳定且具有实际效应量的退化，才做 B2 零训练 A/B。B2 的预注册门槛是：q=0.3 至少恢复 1.0 个百分点，同时 q=0 干净条件下降不超过 0.5 个百分点；未达到则不进入微调。

## 五、当前完成状态

- [x] MUSeg 数据转换、模态和官方 split 验收。
- [x] A1 空有效像素风险复现。
- [x] B1 safe masked loss 实现、接入主头/辅助头并通过 8 项测试。
- [x] 16 图 A2 block mask manifest、48 个预测和独立评估。
- [x] 本机、GitHub、云端运行副本、数据盘仓库 Git 对齐。
- [x] 云端 MVE/A2 证据包下载到本机 `cloud/` 目录并校验。
- [ ] 完整 20 epoch MUSeg 正式训练和严格的 `epoch-20.pth` 验收。
- [ ] 64 图或完整 test 的 A2 扩大复核。
- [ ] random mask、中位数填充负对照和多 seed 复核。
- [ ] B2 validity mask/gating 实现与零训练 A/B。
- [ ] 正式 MUSeg baseline 报告。

## 六、下一步顺序

1. 在有 GPU 的实例上完成正式 MUSeg baseline，使用当前统一提交和固定配置；不要把 epoch-10 screening 结果写成正式性能。
2. 如需要确认 A2 的弱趋势，再扩展到 64 张或完整 test，增加 `q={0,0.1,0.3,0.5}`、random mask、中位数填充负对照和至少一个额外 seed。
3. 只有扩大 A2 达到预注册门槛，才实现 B2 gating，并先做零训练 A/B；否则保留 A2-N-G 结论，转向数据、预处理或其他性能瓶颈。
4. 补做真实模型混合 batch 的 A1/B1 回归测试，作为训练前的工程验收，不改变 A1/B1 已有逻辑结论。

## 七、明确不做的事项

- 不把 ImageNet backbone 或 epoch-10 screening checkpoint当作正式 MUSeg baseline。
- 不在无卡实例上启动训练、A2 推理或 B2 实验。
- 不将云端 checkpoint、预测、数据和日志提交到 GitHub。
- 不因 A2 的轻微单调下降直接宣称 Depth=0 已被证明为主要瓶颈。