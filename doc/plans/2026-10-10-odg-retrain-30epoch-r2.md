# ODG 第二轮 30 epoch 重训计划（修复后语义，待授权）

日期：2026-10-10。依据：上级已裁决“ODG 在正确 eval 模式下 clean +1.45pp、25% 孔洞 +2.96pp，具有继续研究价值；train 模式验证的负向趋势不再作为淘汰依据”，并指示只做本地修复与准备。执行边界与恢复点见[项目状态](../state/current.md)，协议与命令见[ODG 入口](../guides/odg.md)，修复与对齐证据见[评价对齐复测报告](../reports/2026-10-10-odg-bn-align-reevaluation.md)。

**目标：** 用修复后的训练入口（周期验证为 eval 模式、全部 714 个可训练参数张量入优化器）从**同一官方预训练权重**重跑 `original` 与 `odg` 各 30 epoch（300 epoch 日程的暂停点），随后用对齐后的独立评价入口比较 clean 与 25% 人工孔洞。本轮之前的两组 checkpoint 只作历史探索结果保留，不从其 optimizer 续训。

## 已完成的准备（本地，无需 GPU）

- `research/train_odg.py` 周期验证改为 eval 模式，验证后恢复训练模式，并在验证前后比对 BatchNorm running 统计量指纹，发生变化即报错停止。
- 优化器分组由研究层实现：714 个可训练参数张量恰好各入组一次（decay 302 / no_decay 412），缺失、重复或非可训练即报错；checkpoint 的 state 内记录该报告。
- 训练合同新增 `validation_mode=eval` 与 `optimizer_param_scope=all_trainable`，因此修复前（round-1）的 checkpoint 无法被修复后的代码 resume。
- `research/run_odg_gpu.sh` 的运行目录后缀默认 `r2`：`outputs/sun-dev-{original,odg}-seed12345-r2/<run-id>/`。
- 未改变：ODG 设计、优化器类型与超参、数据划分、300 epoch 学习率日程、`--pad_SUNRGBD`、继承的尺度/翻转增强。

## 授权后执行步骤

1. 用户启动实例（有卡）并确认实际规格；无卡准备与数据传输沿用既有 CompShare 流程。
2. **先设置并回读平台计划关机（保险）**，再启动任何长时间训练。
3. 最小 GPU 检查：`bash research/run_odg_gpu.sh smoke`（`MICRO_BATCH=4 ACCUM_STEPS=4`），确认两组 loss/梯度有限、`optimizer parameter groups` 报告为 714/714、显存与吞吐可接受，并核对验证模式日志。
4. 两组并行（同一张卡、两个进程）各跑到 30 epoch：`bash research/run_odg_gpu.sh baseline` 与 `bash research/run_odg_gpu.sh odg`；`--stop-after-epoch 30` 只是暂停，日程仍为 300 epoch。
5. 取回产物（含 `validation.csv` 的 `model_mode=eval` 列、`val_per_class/epoch-030.json` 的 BN 指纹），校验后停机并回读 `Stopped`。
6. 用同一独立评价入口做四组对齐评价：clean 与 25% 孔洞 × original 与 odg（命令见[ODG 入口](../guides/odg.md)），比较绝对 mIoU、各自退化量与主要类别。

## 预计资源

- 主体训练约 **3.6 GPU 小时**（沿用 round-1 同卡并行 `4×4` 的实测：两组 30 epoch 约 3.6 h）。
- 加上最小 GPU 检查与验证/保存开销，整轮约 **4.0–4.5 GPU 小时**，按实测规格价 ¥1.88/小时约 **¥8–8.5**。
- 需用户明确授权；不新建实例、不扩容、不删除磁盘，沿用既有实例与 ¥40/20 GPU 小时的额度口径。

## 风险与边界

- 30 epoch 仍只是暂停点；单一 seed、单一 dev 划分、单一孔洞 seed。
- 孔洞对照中“分布机制”与“显式删除 mask 的有效处理”仍无法分离，需要 `mean`/无 mask 消融才能区分；本轮不新增消融。
- 修复改变了训练语义（验证不再影响 BN 统计量、29 个几何权重进入优化器），因此新结果**不能与 round-1 数值直接相减**。
- 正式 test、100/300 epoch 续训、云端 GPU 均未授权、未运行。
