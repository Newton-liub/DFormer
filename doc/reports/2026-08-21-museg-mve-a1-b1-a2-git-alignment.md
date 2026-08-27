# MUSeg / DFormerv2 MVE、A1/B1 与 Git 对齐阶段分析报告

> **历史报告：** 本文的指标和当时处置结论保持有效，但“20-epoch 正式 baseline”和 official-test 下一步已被阶段二长程协议取消。A2 在本文中指 epoch-10/16 图 pilot，不代表成熟 baseline 上的正式 A2。当前入口见 `doc/main/MUSeg-current-status.md`。

- 汇报周期：2026-08-19 至 2026-08-21
- 报告对象：实验复盘、代码交接与后续研究决策
- 证据边界：本机 Git 检查、云端两个项目副本检查、A1/B1 本地测试、epoch-10 MVE/A2 云端任务与独立评估
- 代码基线：`5c134092d0126eb0430415ee0cebe7b2b2a19ed1`
- 当前状态：A1/B1 已验证并形成低风险修复；A2 已完成但只观察到小幅退化，未达到进入 validity mask/gating 的预注册门槛；本机、GitHub、云端两个项目副本已统一。

## 一、本阶段结论

1. **A1 是确定性的训练数值风险。** 当本地 batch 或分布式 rank 没有任何有效标签像素时，旧的 masked mean 对空集合求均值，测试复现为非 finite。
2. **B1 已关闭 A1 风险。** `safe_masked_mean` 在空集合时返回与计算图连接的有限零值；非空条件的 loss 和梯度保持旧实现行为。8 项本地测试全部通过。
3. **A2 只支持“Depth=0 可能有轻微影响”，不支持“Depth=0 已被证明是主要独立瓶颈”。** 16 张图的 block mask 筛查中，q=0.3 相对 baseline 下降 0.1107 个百分点，q=0.5 下降 0.3337 个百分点，效应小于预注册门槛。
4. **当前不直接进入 B2 gating。** 应先完成正式 MUSeg baseline；若研究目标仍要求确认 A2 弱趋势，再做扩大样本、random mask、填充负对照和额外 seed 的复核。
5. **云端代码已统一，但云端实例目前无卡。** 当前不在该实例上启动训练、推理或 B2 实验。

## 二、Git 与云端目录对齐

### 2.1 处理前的差异

- 本机 `main` 与 GitHub `origin/main` 原本都在 `a2a66e5`，但本机已有暂存改动：3 个 agent 文件、一个 workflow skill、一个配置文件，以及 MVE/A2 执行文档。
- 云端运行副本 `/root/DFormer` 在 `a2a66e5`。
- 云端数据盘仓库 `/root/rivermind-data/DFormer` 在 `7581228`，fetch 后确认落后 GitHub 6 个提交。
- 实验产物位于 `/root/rivermind-data/mve_outputs`，不属于 Git 仓库内容。

### 2.2 实际处理

1. 给本机原始基线创建保护标签：`backup/local-before-align-20260821`。
2. 将已有本机暂存改动提交为 `5c13409`，并推送到 GitHub `origin/main`。
3. 对 `/root/rivermind-data/DFormer` 执行 `fetch` 与 `merge --ff-only`，无冲突同步到 `5c13409`。
4. 对 `/root/DFormer` 执行 `fetch` 与 `merge --ff-only`，无冲突同步到 `5c13409`。
5. 复核结果：本机、GitHub、云端运行副本、云端数据盘仓库均指向完整提交 `5c134092d0126eb0430415ee0cebe7b2b2a19ed1`，工作树均干净。
6. 本地云端证据目录已加入 `.gitignore`，避免 checkpoint、日志和预测文件进入 Git。

### 2.3 当前目录职责

- `/root/rivermind-data/DFormer`：数据盘上的持久化项目仓库，作为云端长期保存和后续同步位置。
- `/root/DFormer`：本次训练与 A2 实际运行副本；已完成同一提交对齐。
- `/root/rivermind-data/mve_outputs`：训练、checkpoint、预测和评估产物，不提交 Git。
- 本机 `cloud/museg-epoch10-a2-20260821`：下载的证据副本，不提交 Git。

## 三、A1/B1 验证—结论逻辑链

### 3.1 问题来源

MUSeg 原始标签中 `0` 是背景；训练标签映射将背景转为 `255=ignore`，前景类别映射为 `0–14`。因此全背景图或某个 crop/rank 只含背景时，有效监督像素数可能为 0。

旧逻辑等价于：

```text
pixel_loss = CrossEntropyLoss(..., reduction="none", ignore_index=255)
loss = pixel_loss[target != 255].mean()
```

当 `target != 255` 为空时，`mean()` 作用于空张量，导致非 finite。这个链条与 Depth=0 不同：A1 是标签有效像素集合为空造成的 loss reduction 问题，不是深度输入或几何先验问题。

### 3.2 A1 证据

`tests/test_masked_loss.py` 构造随机 `1×15×10×10` logits 和不同有效像素数量：

- 0 个有效像素：旧归约非 finite；
- 1 pixel、5 pixels、100 pixels：旧归约 finite。

这给出清晰边界：故障发生在空集合，而不是普通稀疏监督本身。

### 3.3 B1 处理

新增 `models/losses/safe_masked_loss.py`：

- 非空集合：返回选中元素的普通均值；
- 空集合：返回 `values.sum() * 0.0`。

后者保持 loss 与 logits 计算图连接，forward 得到有限 0，backward 得到有限零梯度。`models/builder.py` 已把它接入主分割头和辅助头归约路径。

### 3.4 B1 证据与结论

执行命令：

```text
python -m unittest discover -s tests -p "test_*.py" -v
```

结果：8 项测试全部通过。覆盖内容包括：旧实现空集合非 finite、safe loss 空集合有限零、非空 loss/gradient 与旧实现一致、辅助头路径 finite、mask 局部计数行为正确，以及 A2 block mask 的精确计数和相对路径。

结论：A1-G 与 B1-G 同时成立。B1 是局部、低风险、无新增参数的数值稳定性修复；它不等于已经解决 Depth=0 问题，也不应把两个问题合并为同一个故障。

当前证据缺口：尚未做“真实模型 + 1 张全背景图 + 1 张普通图”的混合 batch 回归；该测试应在正式训练前补做，但不会推翻已有张量级 A1/B1 结论。

## 四、A2 实验与解释边界

### 4.1 实验条件

- 模型：DFormerv2-S；
- checkpoint：epoch-10 MUSeg checkpoint，路径 `/root/rivermind-data/mve_outputs/museg_20epoch/checkpoint/epoch-10.pth`；
- epoch-10 全量验证 mIoU：`20.49`；
- 数据：官方 test 中固定 16 张含前景且原始有效深度比例较高的图；
- mask：seed `20260819`、block mask，只在原始 `Depth16 > 0` 像素中注入额外 0；
- 条件：q=0、q=0.3、q=0.5；共 48 个样本—条件前向；
- 评估：独立执行 `tools/mve/a2_evaluate_results.py`，生成 49 行 CSV（含表头）和 summary JSON。

### 4.2 指标

| 条件 | 前景 mIoU | 相对 q=0 |
|---|---:|---:|
| q=0 | 0.3389857875 | 0 |
| q=0.3 | 0.3378786890 | -0.0011070985（-0.1107 个百分点） |
| q=0.5 | 0.3356490187 | -0.0033367688（-0.3337 个百分点） |

### 4.3 决策

实验在这 16 张图上表现为随 q 增加而小幅下降，但：

- q=0.3 未达到预注册的至少 2.0 个百分点下降；
- q=0.5 下降小于 0.5 个百分点；
- 当前证据范围小，且 checkpoint 只训练到 epoch 10；
- 未完成 64 张/完整 test、random mask、中位数填充负对照和多 seed 复核。

因此 A2 判定为 **A2-N-G / 中间态**：可以保留“模型对额外深度缺失有轻微敏感性”的观察，但不能宣称 Depth=0 已被证明为主要独立瓶颈，也不能宣称 validity mask/gating 已解决该问题。

## 五、云端证据与本地下载

### 5.1 云端原始位置

- 训练：`/root/rivermind-data/mve_outputs/museg_20epoch`；
- A2：`/root/rivermind-data/mve_outputs/museg_a2_screening`；
- 持久化代码：`/root/rivermind-data/DFormer`；
- 运行代码：`/root/DFormer`。

### 5.2 本地证据位置

- `cloud/museg-epoch10-a2-20260821/museg-epoch10-a2-evidence-20260821.tar.gz`；
- 已解压的 `code/`、`training/` 和 `museg_a2_screening/` 子目录。

证据包 SHA-256：

```text
4d5da8a26974124d2cd3979c5166f40e12dbeae9a3f412a7d026649db492519d
```

本次没有下载 307 MB checkpoint；它仍保存在云端导出位置，避免在无卡实例和本机重复占用空间。

## 六、风险与证据缺口

1. epoch-10 checkpoint 只用于 MVE screening，不代表完整 20 epoch 正式性能。
2. A2 样本规模为 16 张，不能支持强因果或普遍性结论。
3. A2 脚本曾出现一次并行输出目录冲突；q=0.5 已串行补跑并独立验收，最终评估使用串行补跑的有效结果。
4. A1/B1 尚缺真实混合 batch 回归，但核心空集合和梯度等价测试已通过。
5. B2 validity mask/gating 尚未实现，不能写成已完成或已解决。
6. 云实例当前无卡（`GPU=0`），只能用于代码、文件和 Git 操作，不能用于新的 GPU 实验。

## 七、需要完成的事情

### 优先级 P0：正式基线前置

1. 获取有 GPU 的实例，确认 `GPU>0`、CUDA 可用和运行环境版本。
2. 使用统一提交 `5c134092d0126eb0430415ee0cebe7b2b2a19ed1`，重新执行真实模型混合 batch 回归：1 张全背景图 + 1 张普通图，验证主头和辅助头训练归约。
3. 明确并记录正式 MUSeg baseline 的训练周期、checkpoint 保存、完整 test 评估和环境版本。
4. 运行完整 20 epoch 训练，严格验收 `epoch-20.pth`，不要沿用 epoch-10 screening 结论。

### 优先级 P1：如果仍需确认 Depth=0

1. 将 A2 从 16 张扩大到 64 张或完整官方 test。
2. 增加 `q={0,0.1,0.3,0.5}`、random mask 和中位数填充负对照。
3. 至少增加一个额外 seed，报告配对差值、置信区间或 bootstrap 区间、边界带 mIoU，以及自然无效深度比例分层结果。
4. 只有扩大 A2 达到预注册效应量门槛，才实现 B2 validity mask/gating。

### 优先级 P2：B2 方案验证

1. 从 `Depth16 > 0` 构造 validity mask，并在每个 geometry prior stage 对齐到 patch 网格。
2. 使用 `pair_valid = valid_i & valid_j`：有效 pair 保留 depth decay，无效 pair 只保留 positional decay。
3. 固定同一 checkpoint、样本、mask、seed，先做零训练 B0/B2 A/B。
4. 仅当 q=0.3 至少恢复 1.0 个百分点且 q=0 下降不超过 0.5 个百分点时，再做 5 epoch 小规模微调。

## 八、复现命令

A1/B1 本地测试：

```powershell
python -m unittest discover -s tests -p "test_*.py" -v
```

历史云端 MVE/A2 的执行结论、任务记录和结果已归档在本报告及 `doc/reports/2026-08-21-museg-mve-cleanup-and-disposition.md`。原 `doc/临时` 执行文档已在后续计划清理中合并删除；本文形成时使用的早期 4090 待执行入口后来也已删除，当前唯一实时入口为 `doc/main/MUSeg-current-status.md`。