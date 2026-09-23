# MMFR E1 Batch 1B R-OE-lite Gate-B 报告

- 日期：2026-09-23
- 报告范围：R-OE-lite 冻结实现、最小 Gate-B 与 matched C0 复用资格
- 当前状态：`Gate-B PASS; awaiting separate formal-training authorization`
- 证据身份：`outputs/mmfr-e1-batch1b-gateb/e1-batch1b-gateb.json`
- 授权边界：没有启动 R-OE-lite 正式训练、Quick-Val、Main-Val、云端任务或 official test；official test 继续为 `sealed_unread`

**大白话结论：** R-OE-lite 的实现和最小 Gate-B 检查已通过，现有 Batch 1A C0 可以在共同训练合同不变的前提下作为对照复用。Gate-B 证明的是实现、数据路由、优化器和最小更新路径符合冻结合同；它不证明模型效果，也没有启动正式训练。下一步必须先取得正式训练的单独授权。

## 1. 关键概念与判断边界

- **R-OE-lite（Observable-Empty Geometry Substitute，可观测空几何替代）**：只识别当前几何有效区域内是否没有任何非零 raw Depth，不推断空值的隐藏原因。大白话说，它只看见“当前有效区域没有深度值”，不能区分传感器整幅丢失、自然空洞或 dropout 删空。
- **Exact bypass（逐位旁路）**：未触发样本直接沿用原 corrupted Depth，数值逐位相同，且 substitute 不执行。
- **Gate-B**：实现进入正式训练申请前的工程资格检查，不是训练授权或效果评价。
- `entire_missing@1.0` 是固定压力条件；它不能证明模型识别出了 `entire_missing` 这个原因。

## 2. Source、共同合同与 C0 复用

- 协议身份：`MMFR-E1-Batch1B-R-OE-lite-v1`；候选 config：`local_configs/MUSeg/DFormerv2_S_MMFR_E1_Batch1B_R_OE.py`。
- source config：`local_configs.MUSeg.DFormerv2_S_MMFR_A2_DepthCorrupt_v3`。
- source checkpoint：A2 epoch-420；SHA-256 `2b72eb7f28e5c4b4f52bdc43e2ac5c7298169325a412bfbcbc7a022c29822597`；812 个 model keys；completed/next epoch `420/421`；global optimizer step `53735`。
- `train-dev`：1277 条，SHA-256 `a6b15b63f6d5193e3928ea24ada25be403a48e68d1c1f9372cdbbc3fe5cd8470`；`val-dev`：318 条，SHA-256 `1d0719d8f64f016d48995c25ab66d4004d76b7155d9efeef7cbb7454c0dd0e83`。
- 随机种子：`772961337`。两配置共同字段逐项相等，`exact_equal=true`、`mismatches={}`。
- matched C0：复用 Batch 1A fixed final checkpoint，SHA-256 `ca618b23d18eabb201a0d11d18da383ac99576d0feae5864e3233bda527d9a1a`。
- 共同预算和训练身份：20 nominal epochs、每 epoch 128 attempts、2560 successful updates、batch size 10、workers 8、AMP 与 SyncBN 开启、DDP 关闭；base/new LR 为 `1e-5/3e-5`、warmup 128 successful updates、poly power `0.9`、`p_clean=0.25`、`max_specs=2`、六类 Depth corruption 与 virtual curriculum 保持一致；共同损失、四组 optimizer 与 checkpoint 机会保持一致；source 权重按 weights-only restart，不恢复旧 optimizer、scheduler、GradScaler 或 RNG。
- TF32 勘误：Batch 1B 沿用 Batch 1A 的实际训练行为。训练入口保留 `torch.set_float32_matmul_precision("high")`，不能将旧文本中的 `TF32 off` 当作实际训练设置；Gate-B 检查确认该行为仍存在。

## 3. C0 RNG、sampler 与共同初始化证据

- 构建后 CPU RNG：C0 与 R-OE-lite SHA-256 均为 `787cd37cbb6948f81470deb8e68301578671f2d50bfb1e677683e7344d417c0c`。
- 构建后 CUDA RNG：两侧 SHA-256 均为 `ec4cb796f83ce9ed8d44081cd347df2f48de3ad02f364904692dc84c58a99a89`。
- 第一 epoch sample permutation：两侧均为 1280 项，逐项 exact equal；digest 为 `196564b61f9f5349dc6c4b77b7c993ccbf9c0660646bfde637fb5a87da1bc028`。
- C0 与 R-OE-lite 的 812 个共有 source state keys 均保持一致；R-OE-lite 新增的 14 个 state tensors 只属于 substitute。
- Gate-B 因此确认 `RNG identity=PASS`、`sampler identity=PASS`、`existing C0 reusable=YES`。这确认首轮顺序与共同起点，不替代训练期的完整行为记录。

## 4. 实现与最小训练路径

实现文件与职责见 `audit_implementation_diff_e1_batch1b_roe.md`。关键合同与检查如下：

- Substitute 固定为无 skip、无 normalization 的 RGB CNN，通道序列 `[3,122,398,256,256,398,122,1]`，使用 AvgPool2d、GELU、bilinear resize 和 straight-through clamp。
- substitute 新增可训练参数精确为 `3,302,785`；全部 7 个卷积 weight（`3,301,232` 个元素）和 7 个 bias（`1,553` 个元素）分别进入 `new_decay` 与 `new_no_decay`。
- detector 只读取当前 raw Depth 与 $V_{\mathrm{geom}}$；RGB 只供 substitute 使用。mixed batch 路由 `[False, True, False, False]`，substitute 只调用一次。nonempty 与 no-geometry 都走 bypass，未触发样本不执行 substitute、clamp 或重新 normalization。
- substitute 输出 finite，观测范围为 `[0.0, 127.50041961669922]`；padding exact zero；三通道复制和 Depth normalization exact。
- Reliability auxiliary 继续使用原 raw RGB、原 raw Depth 与原 target；substitute 不进入 auxiliary loss。
- 所有 trainable parameters 的 optimizer membership 恰为 1；29 个 `Geo.weight` 属于 `base_decay`，14 个 SyncBN 参数属于 `base_no_decay`。
- AMP 单步 loss `0.7119939327` finite，optimizer step 已应用；14 个 substitute 参数张量的梯度均 finite 且非零，代表性前层与 head 参数均发生更新。该单步检查不等价于 epoch 训练。

## 5. Gate-B 成本记录

本地 NVIDIA GeForce RTX 5060 Laptop GPU，batch size 1，latency warmup 1 次、repeats 2 次：

- non-trigger inference median：`130.693645 ms`；trigger inference median：`263.437180 ms`。
- peak allocated memory：C0 `2,284,511,744` bytes，R-OE-lite `3,875,250,688` bytes，差 `1,517.046875 MiB`。
- peak reserved memory：C0 `2,428,502,016` bytes，R-OE-lite `6,211,764,224` bytes，差 `3,608 MiB`。

这些是 Gate-B batch-1 工程成本记录，不是 batch-size-10 的正式训练可行性证明，也不是完整训练耗时或费用估计。

## 6. 冻结的 10-condition Main-Val 处置门槛

正式 Main-Val 必须另行授权，使用 matched C0 和 R-OE-lite fixed final checkpoint，在完整 `val-dev`、冻结 `msflip-whole-original-grid-v1` 与十个条件上完成配对比较。Quick-Val 只用于筛查，不承担最终 `promote/stop` 判定；未获授权前不运行。

所有差值定义为 R-OE-lite 减 matched C0，单位为百分点（pp）。$M_6$ 是六个单故障 mIoU 的未加权宏平均。冻结的 Main-Val 门槛为：

- **Promote**：`entire_missing@1.0` 的 delta `>= +0.50 pp`；$M_6$ delta `>= 0`；clean delta `>= -0.25 pp`；六个单故障中没有任一 delta `< -0.50 pp`。
- **Stop**：满足任一项即 stop：`entire_missing@1.0` delta `<= 0`；$M_6$ delta `<= -0.50 pp`；clean delta `< -0.50 pp`；六个单故障中任一 delta `< -1.00 pp`。
- **Inconclusive**：身份和完整性均合格的评价完成，但既不满足 promote 也未触发 stop。
- **Blocked**：评价未完整完成或任一冻结身份/配对性断言失败；不得把 blocked 改称 inconclusive。

其余三个混合条件仍须报告，但没有独立 promote/stop 阈值。`entire_missing@1.0` 仅为压力条件；任何指标结果均不能据此声称 detector 识别了 synthetic cause。单 seed、单 checkpoint、单次 Main-Val 也不构成统计显著性或现实部署可靠性结论。

## 7. Canonical artifact、复现命令与停止点

命令：

```text
PYTHONPATH=D:\\0Project\\DFormer python tools\\mmfr\\e1_batch1b_gateb.py --latency-warmup 1 --latency-repeats 2
```

- Gate-B JSON：`outputs/mmfr-e1-batch1b-gateb/e1-batch1b-gateb.json`
- SHA-256：`35297b490c3e3eb54b5e66d3f06784688fca65038e7d09874c30c60cde820231`
- `status=PASS`，`failed_checks=[]`
- `official_test_included=false`，`formal_training_started=false`
- Gate-B duration：`11.80315530000371` s

**恢复点：** `ready-for-R-OE-formal-training-authorization`。现阶段等待 R-OE-lite 正式训练的单独授权；本报告不授权正式训练、Quick-Val、Main-Val、云端任务、Batch 2、T 或 official test。Batch 1A C0/F-lite Main-Val 的上级处置仍独立待审，不由本 Gate-B 结论裁决。
