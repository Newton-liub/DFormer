# ODG 首轮 SUN RGB-D 30 epoch 实验结果（2026-10-10）

本报告记录 2026-10-10 首轮 GPU 实验：DFormerv2-S + HAM 在 SUN RGB-D 开发划分上，`original`（作者几何路径）与 `odg`（局部观测分布几何先验）各训练 30 epoch 的对照结果、性能诊断与实际费用。执行依据为用户批准的[首轮 GPU 实验计划](../../临时/ODG_GPU_实验计划_20261010.md)；协议与命令见 [ODG 入口](../guides/odg.md)，实时状态见[项目状态](../state/current.md)。

**结论摘要：** 两组均正常完成 30 epoch（各 8940 次 optimizer 更新，无失败、无显存溢出、无持续 AMP 跳步）。dev 单尺度无翻转 mIoU 上，`original` 为 26.01 / 36.82 / **39.39**（epoch 10/20/30），`odg` 为 24.92 / 34.83 / **36.97**，ODG 分别落后 1.09 / 1.99 / **2.42** 个点，且差距在三个验证点上持续扩大；训练 loss 也始终略高。按计划的 30 epoch 停止条件（20、30 epoch 均落后约 2 点以上、差距未收窄、伴随类别退化），**不建议未经复核直接续训到 100 epoch**。

## 训练合同与执行方式

- 两组完全同条件：`local_configs.research.ODG_SUNRGBD`，DFormerv2-S + HAM（宽度 1024）、37 类、480×480；同一官方 encoder 预训练（`DFormerv2_Small_pretrained.pth`）、新分割头与新 optimizer/scaler；seed 12345、train-dev 4757 / dev 528；AdamW lr $8\times10^{-5}$、weight decay 0.01、betas 0.9/0.999；有效 batch 16、warmup 10、poly 0.9、**总日程 300 epoch**，30 epoch 只是暂停点。两组的 `contract.json` 除 `geometry_mode` 外字段一致，训练/验证清单指纹一致。
- batching：`MICRO_BATCH=4 × ACCUM_STEPS=4`（有效 batch 16），两组相同。该组合由最小 GPU 检查在 `2×8` 与 `4×4` 之间选定：`4×4` 单进程吞吐 0.955–1.099 s/attempt（`2×8` 为 1.745–1.834），显存峰值 5.2–5.9 GB，`16×1` 因需 >17 GB 显存未采用。
- 执行方式：两组**并行**在同一 4090 上运行（`screen`），而不是计划中的串行。理由见后文性能诊断：并行总吞吐比串行高约 55%。原“训练完自动串行启动 odg”的链在诊断期间被挂起，避免重复启动。
- 未做：人工孔洞对照、五尺度+翻转评价、官方 test、mean 消融、100 epoch 续训。

## 主要指标

dev 528 张，单尺度、无翻转（`val_per_class/epoch-0X0.json` 为逐类来源）：

| epoch | original mIoU | odg mIoU | 差 | original mAcc | odg mAcc | original mF1 | odg mF1 |
|---|---|---|---|---|---|---|---|
| 10 | 26.01 | 24.92 | -1.09 | 33.24 | 31.87 | 37.55 | 36.28 |
| 20 | 36.82 | 34.83 | -1.99 | 46.42 | 43.95 | 50.29 | 47.99 |
| 30 | **39.39** | **36.97** | **-2.42** | 49.23 | 45.39 | 53.31 | 50.55 |

每 epoch 平均训练 loss（298 次更新/epoch）：

| epoch | 1 | 5 | 10 | 15 | 20 | 25 | 30 |
|---|---|---|---|---|---|---|---|
| original | 3.8939 | 1.1421 | 0.7347 | 0.5638 | 0.4768 | 0.4159 | 0.3488 |
| odg | 3.8987 | 1.1464 | 0.7364 | 0.5695 | 0.4887 | 0.4213 | 0.3625 |

两组均单调下降、未出现异常；ODG 的 loss 始终略高（+0.005 → +0.014），与 mIoU 差距方向一致。

## epoch 30 逐类 IoU（差异较大的类）

ODG 明显更好的类：`fridge` 44.62→58.53（+13.9）、`person` 22.75→33.01（+10.3）、`towel` 25.52→30.88（+5.4）、`ceiling` 55.50→58.58（+3.1）、`books` 30.07→31.58（+1.5）。
ODG 明显更差的类：`tv` 63.91→37.30（-26.6）、`bathtub` 23.67→12.06（-11.6）、`mirror` 21.41→10.14（-11.3）、`toilet` 71.22→61.33（-9.9）、`blinds` 45.39→35.65（-9.7）、`clothes` 22.58→14.99（-7.6）、`desk` 17.17→10.79（-6.4）、`bag` 28.90→22.91（-6.0）、`curtain` 40.37→34.90（-5.5）、`sofa` 54.39→49.07（-5.3）。
差异方向零散：ODG 在少数类上有提升，但没有出现“缺失观测相关类别普遍受益”的模式；三个空支持域类（`floor_mat`、`shower_curtain`、`night_stand`）两组均为 0，无区分信息。预测图（`predictions/epoch-030/train_31|33|35_pred.png`）在两组间结构相近，未做逐像素统计。

## 性能诊断与吞吐优化

诊断在正式训练运行中用 `SIGSTOP` 挂起（不终止、不丢进度）后完成，随后立刻恢复；共 5 个 50 次尝试的短测：

| 组合 | 吞吐 | 峰值显存 | 说明 |
|---|---|---|---|
| 4×4 | 15.82 samples/s | 5.9 GB | 单进程，GPU 利用率约 41% |
| 8×2 | 22.11 samples/s | 11.1 GB | 单进程，+40% |
| 16×1 | 未测成 | — | CUDA OOM：该进程需 >17.2 GB |
| 并行 4×4（original+odg） | 12.84 + 11.62 = 24.46 samples/s | 5.9 / 5.2 GB | 各降约 20%，合计 +55% |

瓶颈定位：8 个 DataLoader worker 各仅 7–24% CPU、整机 CPU 约 17%，数据供给与 CPU 均未饱和；主进程约 1 核串行推进每步。由两组 micro-batch 对照反推，每次 micro-batch 迭代约有 0.116 s 固定开销，另每次 optimizer 尝试约有 0.5 s 非迭代开销。实际效果：并行后 GPU 利用率升至 87–99%，两组 30 epoch 于约 3.6 小时内完成；按串行方案估算约需 5.2 小时，**节省约 1.4–1.6 小时**（约 ¥3）。未修改任何代码。

## 时间与费用

- 有卡会话：2026-10-10 09:04:34 → 13:23:02（UTC+8），平台计 15508 秒 = **4.31 GPU 小时**；按实测规格价 ¥1.88/小时计 **约 ¥8.1**，远低于本阶段 ¥40 / 20 GPU 小时的授权上限。
- 明细：最小 GPU 检查（4 次 200 尝试）约 0.35 h；性能诊断短测约 0.12 h；两组 30 epoch 主体约 3.6 h；其余为建环境/验证/保存开销。
- 训练进度时间：original 09:33 开始、12:28 结束 30 epoch；odg 10:05 开始、13:19 结束 30 epoch。
- 实例于 13:23 主动停止并回读 `Stopped`；随后以无卡 A 模式（¥0.13/小时）短窗取回产物，取回完成后再次停止。
- 两组每组 8932/8940 次更新成功应用，AMP 跳步各 8 次（均为 scaler 预热期的前段），无失败记录（无 `failure.json`）。

## 产物位置

本地（`outputs/` 不进 Git）：

- `outputs/sun-dev-original-seed12345/20261010-013324/`：`loss_lr.csv`、`validation.csv`、`val_per_class/epoch-0{10,20,30}.json`、`predictions/epoch-0{10,20,30}/`、`resolved_config.json`、`args.json`、`contract.json`、`train.log`、`last.pth`、`best-dev.pth`、`swanlab/`。
- `outputs/sun-dev-odg-seed12345/20261010-020527/`：同结构，`last.pth`、`best-dev.pth` 已取回。
- `outputs/odg-gpu-20261010/`：`smoke-screen.log`、`smoke4-screen.log`、`stage1-screen.log`、`odg-par-screen.log`、`probe-mb4|mb8|mb16|par-original|par-odg.log`；`outputs/sun-odg-smoke-*/`、`outputs/probe-*/` 的 `smoke-report.json`。
- 本地已用本地环境（torch 2.7.0，CPU）实际加载 4 个 checkpoint，`snapshot_kind`/`completed_epochs=30`/`best_dev_miou=39.39`（original）与 `36.97`（odg）/802 个模型条目均与云端一致；预测 PNG 可正常打开。字节数与云端一致。

云端保留（不删除）：`/root/rivermind-data/DFormer/outputs/sun-dev-*/<run-id>/` 全部文件，含未取回的 `stage-epoch-30.pth`（与 `last.pth` 同为 epoch 30 状态，`snapshot_kind=stage`）与完整 `swanlab/` 离线记录。

SwanLab 在线 run：original `dt6txln8`、odg `e0bqsosm`（项目 `dformer-Newton_liub/dformer-research`）。

## 下一步建议

1. **不自动续训 100 epoch。** 依据：30 epoch 三个验证点差距持续扩大（-1.09 → -1.99 → -2.42），loss 同向偏高，逐类差异未显示预期方向上的收益；计划中的正向条件（最近多个验证点持续提升且类别/孔洞证据支持预期）未满足。
2. 若要继续投入有卡时间，最先做的应是计划中已有的**缺失观测对照**：用现有 30 epoch checkpoint 在 dev 上做一次两组同 seed 的 25% 人工连通孔洞评价（纯评价、不训练，成本很小）。若在该对照上 ODG 同样没有优势，应回到方法层面复核（例如核 bias 的实际量级与 $w_d$ 的相互作用、`mean` 消融位置），而不是先花 100 epoch 的费用。
3. 若用户仍决定看 100 epoch 趋势：按本机实测速率（约 5.7–5.9 分钟/epoch 单进程、并行约 1.25 倍）估计再需约 8.4 小时有卡时间（约 ¥16），须另行授权。
4. 本报告不含任何 test 集结果；所有数字为 dev 单尺度无翻转，不可与五尺度+翻转或官方 test 数字混比。

## 未运行 / 未验证项

- 未做人工孔洞、五尺度+翻转、官方 test、mean 消融、NYUv2、100/300 epoch 续训。
- 未做逐像素预测统计与可视化对比，仅有 3 张 dev 预测图。
- 未对每个配置逐一验证 16×1 的可行性（仅知在另一进程占显存时 OOM）。
- checkpoint 的“可读”验证为本地 torch 2.7.0 CPU 加载并检查字段，未做与云端 2.1.2 的逐张量数值一致性比对。
