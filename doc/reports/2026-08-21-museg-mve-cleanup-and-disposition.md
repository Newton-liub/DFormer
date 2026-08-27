# MUSeg MVE 实验、云端状态与项目处置总结报告

> **历史报告：** 本文保留 2026-08-21 的 MVE 处置和实例状态。其“实例已停止”和后续 20-epoch 路线不是当前实时状态；当前 Stage-05 seed 1 已运行，见 `doc/main/MUSeg-current-status.md`。

- 汇报日期：2026-08-21
- 报告对象：项目整理、实验交接与后续执行决策
- 代码基线：`27437c1ddf5ae6c8f5da05b7ae94fc6b29fc80af`
- 证据边界：本地 Git、GitHub 同步记录、云端两个 DFormer 仓库、MVE/A2 训练与推理产物、A1/B1 测试和当前实例状态
- 当前状态：MVE/A2 筛查已完成，代码和仓库已对齐，实例已停止；本次清理不删除 `liu-test-exp`，也不删除仍具有复现价值的实验文档和证据。

## 一、为什么要执行 MVE 实验

本阶段的目标不是直接宣称 MUSeg 的最终性能，也不是立即实现新的模型结构，而是用较低成本回答两个会影响后续方向的问题。

### 1. A1：全背景样本是否会造成训练数值风险

MUSeg 中存在全背景图。训练标签映射后，背景像素为 `255=ignore`。如果一个本地 batch 或分布式 rank 中没有任何有效前景像素，旧的 masked loss 会对空集合直接求均值，可能产生非有限 loss，并进一步影响反向传播和训练稳定性。

A1 的目的，是先确认这个风险是否真实存在，并把它与 Depth=0 问题区分开。A1 不是性能调参，而是训练工程稳定性验证。

### 2. A2：Depth=0 是否是独立且主要的性能瓶颈

DFormerv2 的 geometry prior 使用深度差异。项目中原始无效深度以 `Depth=0` 表示，因此需要先判断：把有效深度额外置零，是否会造成足够大、可重复的性能下降。

如果影响达到预注册门槛，才有理由进入 B2 validity mask/gating 方案；如果影响很小，直接改模型可能会把有限的工程资源投入到并非主要的问题上。MVE 使用 epoch-10 checkpoint 和 16 张样本，是为了快速筛查方向，不替代完整 MUSeg baseline。

## 二、已经完成的实验与结论

### 2.1 A1/B1：空有效像素归约风险

- 旧实现对全 ignore 的 loss 张量求均值时出现非 finite。
- 新增 `models/losses/safe_masked_loss.py`，实现 `safe_masked_mean`。
- 空集合返回与计算图连接的有限零值；非空集合保持原有均值行为。
- `models/builder.py` 已将安全归约接入主分割头和辅助头。
- 执行命令：`python -m unittest discover -s tests -p "test_*.py" -v`。
- 结果：8 项测试全部通过。

结论：A1 是已复现的确定性数值风险，B1 已完成并通过张量级测试。正式训练前仍建议补做“1 张全背景图 + 1 张普通图”的真实模型混合 batch 回归，但这不改变现有 A1/B1 结论。

### 2.2 A2：Depth block mask 筛查

实验条件如下：

- 模型：DFormerv2-S；
- checkpoint：`/root/rivermind-data/mve_outputs/museg_20epoch/checkpoint/epoch-10.pth`；
- checkpoint：epoch 10 screening 结果，不是完整 20 epoch 正式结果；
- epoch-10 全量验证 mIoU：`20.49`；
- 数据：官方 test 中固定 16 张含前景且有效深度比例较高的样本；
- mask：seed `20260819`、block mask，只在原始 `Depth16 > 0` 像素中注入额外零值；
- 条件：`q=0`、`q=0.3`、`q=0.5`；
- 总量：16 张样本 × 3 个条件 = 48 个样本—条件前向；
- 评估：独立执行 `tools/mve/a2_evaluate_results.py`。

结果：

| 条件 | 前景 mIoU | 相对 q=0 |
|---|---:|---:|
| q=0 | 0.3389857875 | 0 |
| q=0.3 | 0.3378786890 | -0.0011070985（-0.1107 个百分点） |
| q=0.5 | 0.3356490187 | -0.0033367688（-0.3337 个百分点） |

结果在 16 张 screening 样本上随缺失率增加而单调下降，但下降幅度很小。按照预注册门槛，A2 判定为 **A2-N-G**：当前证据只支持“模型可能对额外深度缺失有轻微敏感性”，不支持“Depth=0 已被证明是主要独立瓶颈”。

因此暂不进入 B2 validity mask/gating 实现。当前也不能把 B2 写成已完成或已解决事项。

## 三、三个 DFormer 项目的情况

这里的“三个项目”指本地工作副本、云端数据盘持久化副本和云端运行副本。GitHub 是三者共同使用的远端代码基线，不另算一个本地工作副本。

### 3.1 本地项目

- 路径：`D:\0Project\DFormer`；
- 分支：`main`；
- HEAD：`27437c1ddf5ae6c8f5da05b7ae94fc6b29fc80af`；
- `origin/main`：同一提交；
- 工作树：已确认干净；
- 本地证据：`cloud/museg-epoch10-a2-20260821`；
- 证据目录：已加入 `.gitignore`，不提交 checkpoint、预测、日志和解压产物。

### 3.2 云端数据盘持久化项目

- 路径：`/root/rivermind-data/DFormer`；
- 用途：长期保存代码仓库，并作为后续云端同步基准；
- 分支：`main`；
- HEAD：`27437c1ddf5ae6c8f5da05b7ae94fc6b29fc80af`；
- `origin/main`：同一提交；
- 工作树：最后一次云端核验时干净；
- 实验产物不放在仓库内，而位于 `/root/rivermind-data/mve_outputs` 和 `/root/rivermind-data/exports`。

### 3.3 云端实际运行项目

- 路径：`/root/DFormer`；
- 用途：本次训练、checkpoint 推理和 A2 评估的实际运行副本；
- 分支：`main`；
- HEAD：`27437c1ddf5ae6c8f5da05b7ae94fc6b29fc80af`；
- `origin/main`：同一提交；
- 工作树：最后一次云端核验时干净；
- 当前不再承担运行任务，实例已停止。

### 3.4 GitHub 远端

- 远端：`https://github.com/Newton-liub/DFormer.git`；
- `origin/main`：`27437c1ddf5ae6c8f5da05b7ae94fc6b29fc80af`；
- 三份 DFormer 副本均通过 fast-forward 同步到该提交；
- 当前没有本地、云端和 GitHub 之间的提交分叉。

## 四、云端实验产物和实例状态

### 4.1 已保留的云端产物

- 训练输出：`/root/rivermind-data/mve_outputs/museg_20epoch`；
- A2 输出：`/root/rivermind-data/mve_outputs/museg_a2_screening`；
- 证据包：`/root/rivermind-data/exports/museg-epoch10-a2-evidence-20260821.tar.gz`；
- 本地证据包：`cloud/museg-epoch10-a2-20260821/museg-epoch10-a2-evidence-20260821.tar.gz`；
- 证据包 SHA-256：`4d5da8a26974124d2cd3979c5166f40e12dbeae9a3f412a7d026649db492519d`；
- checkpoint 仍保留在云端导出位置，未重复下载大文件。

### 4.2 当前云实例

- 实例：`cpod-1tyvjsiu6ahe`；
- 区域：`cn-bj2`；
- 最后查询状态：`Stopped`；
- 当前 GPU：`0`；
- 结论：不在该实例上启动新的训练、推理或 B2 实验。若未来需要继续实验，应重新申请或启动有 GPU 的实例，并重新确认 CUDA、显卡型号和费用状态。

## 五、临时文件与重复文档清理结论

### 5.1 `liu-test-exp` 明确不清理

`liu-test-exp` 是用户用于打开思路的草稿区，不属于本次项目清理范围。本次已恢复此前误删的 `liu-test-exp/对抗 copy.md`，并确认该目录没有工作树改动。后续不对该目录执行删除、重命名或内容整理。

### 5.2 后续临时计划清理

本报告生成时，以下两份临时文件分别承担总计划和云端执行流水职责：

- `doc/临时/2026-08-19-museg-mve-execution-plan.md`；
- `doc/临时/云/MUSeg-20epoch-A2云端执行流程.md`。

其中需要长期保留的 A1/B1/A2 结论、任务结果、checkpoint 和停止状态已经写入本报告及 `doc/reports/2026-08-21-museg-mve-a1-b1-a2-git-alignment.md`。在 4090 正式训练准备完成后，上述临时计划已过时并被删除，避免新对话误用旧 commit、旧仓库路径和 epoch-10 screening 流程。

### 5.3 形成时点的待执行入口（后续已删除）

本文形成时，唯一待执行计划是后来删除的早期 4090 云端训练交接计划。其历史职责已由 Stage-01 至 Stage-11 计划、Stage-04 日期化报告和当前状态入口覆盖；当前事实、执行边界和恢复点只看 `doc/main/MUSeg-current-status.md`。

该计划曾汇总已完成事项、无需重复的检查、当时项目基线、RTX 4090 batch 探测、SwanLab online 正式训练和结果验收步骤。历史结论继续以 `doc/reports/` 下的报告为证据，不再以已删除计划作为审计入口。

### 5.4 清理边界

`liu-test-exp` 仍不属于项目计划清理范围。本次只清理 `doc/临时` 中已经被报告吸收的旧计划和旧云端指南，不删除数据、checkpoint、本地 `cloud/` 证据或正式报告。

## 六、后续处置方案

### P0：完成正式 MUSeg baseline

1. 获取有 GPU 的实例，先确认 `GPU>0`、CUDA 可用、PyTorch 版本和显存。
2. 使用统一提交 `27437c1ddf5ae6c8f5da05b7ae94fc6b29fc80af`，不要回到旧提交 `5c13409`。
3. 补做真实模型混合 batch 回归：1 张全背景图 + 1 张普通图，覆盖主头和辅助头。
4. 运行完整 20 epoch 正式训练，严格验收 `epoch-20.pth`、训练日志、完整 test 指标和环境信息。
5. 将 epoch-10 screening 与正式 baseline 分开报告，不混用性能结论。

### P1：仅在研究目标仍要求确认 A2 时扩大筛查

1. 将样本规模从 16 张扩大到 64 张或完整官方 test。
2. 增加 `q={0,0.1,0.3,0.5}`、random mask 和中位数填充负对照。
3. 至少增加一个额外 seed，报告配对差值、置信区间或 bootstrap 区间。
4. 按自然无效深度比例分层，避免把数据分布差异误判为 mask 效应。
5. 只有效应量稳定达到预注册门槛，才进入 B2。

### P2：条件触发后再做 B2

1. 从 `Depth16 > 0` 构造 validity mask。
2. 在每个 geometry prior stage 对齐到对应 patch 网格。
3. 使用 `pair_valid = valid_i & valid_j`；有效 pair 保留 depth decay，无效 pair 只保留 positional decay。
4. 先做固定 checkpoint、固定样本、固定 mask、固定 seed 的零训练 A/B。
5. 只有 q=0.3 至少恢复 1.0 个百分点且 q=0 干净条件下降不超过 0.5 个百分点时，才进行 5 epoch 小规模微调。

## 七、复现入口与证据路径

- A1/B1 测试：`python -m unittest discover -s tests -p "test_*.py" -v`；
- 当前事实与恢复点：`doc/main/MUSeg-current-status.md`；
- 阶段计划索引：`doc/plans/MUSeg阶段二长程Baseline与MVE/00-总索引与执行门禁.md`，不承担实时状态；
- 历史 A1/B1/A2 综合分析：`doc/reports/2026-08-21-museg-mve-a1-b1-a2-git-alignment.md`；
- 历史清理与处置记录：`doc/reports/2026-08-21-museg-mve-cleanup-and-disposition.md`；
- 本地证据：`cloud/museg-epoch10-a2-20260821`；
- 云端产物根目录：`/root/rivermind-data/mve_outputs`。

## 八、最终判断

MVE 实验已经完成它的筛查任务：A1/B1 的数值稳定性问题得到明确修复，A2 没有达到触发 B2 的效应量门槛。当前最合理的处置不是继续在无卡实例上堆叠小规模实验，也不是直接改 geometry prior，而是保留证据、完成正式 MUSeg baseline，并根据正式 baseline 的研究目标决定是否扩大 A2。