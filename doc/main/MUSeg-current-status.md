# MUSeg 当前状态与唯一实时入口

> **事实截至：** 2026-09-24 R-OE-lite v2 本地修改与云端实例控制面只读核验；GPU 三步验证及新正式训练尚未开始。
> **用途：** 当前阶段、授权边界与恢复点。仍开放的研究选择见 [`MUSeg-open-decisions.md`](MUSeg-open-decisions.md)；v1 中止的历史事实见 [`Batch 1B 中止报告`](../../MMFR/02_evidence/report_e1_batch1b_roe_formal_training_attempt_20260924.md)。

## 当前阶段与最近一个大动作

**阶段：** MMFR（多形式模态失效与可靠性）E1 Batch 1B，`R-OE-lite-v2-memory-fix-preflight`。v1 正式训练曾真实完成至少 168 次 optimizer update，后因 RTX 4090 CUDA OOM 中止；未生成 fixed-final checkpoint，也未运行四条件 Quick-Val。v1 原始训练状态和旧 Gate-B 结果不能作为 v2 的训练进度或资格结果。

**本次授权与修改：** 用户于 2026-09-24 批准 R-OE-lite v2：保留原有七层可训练卷积、参数量 `3,302,785`、可观测无深度（observable-empty）检测及只处理触发样本的路由；在 `models/roe_substitute.py` 将 `d2 → GELU → head` 移到完整分辨率的双线性插值之前，使该处只插值单通道 logits。v2 改变逐像素函数，建立新身份 `MMFR-E1-Batch1B-R-OE-lite-v2`，不得续接 v1 的至少 168 次更新。**大白话说明：** 先在小特征图算完输出通道，再把一张单通道图放大，以降低显存峰值；这不是声称新旧模型计算结果完全相同。

**已核验：** 本地项目 Python 环境中 substitute 可 import、输出单通道且尺寸与输入一致、掩码覆盖区严格为零、可训练参数为 `3,302,785`；v2 config 可 import，仍为 batch size 10、480×640。`models/builder.py` 的非触发 exact bypass、triggered subset 路由和原 reliability auxiliary 输入均未改动。以上不是 GPU 显存或训练效果证明。CompShare 控制面只读查询确认此前使用的 RTX 4090 实例 `cpod-1vbh7faqcauq` 在本次查询时为 `Stopped`；未据此推断未来状态。

## 授权边界与阻塞

- **条件授权：** 本地提交后将新 commit 提供给云端，在 RTX 4090 上用原 Batch 1B 完整正式训练配置连续运行 3 次成功 optimizer update，记录 loss、OOM、峰值 allocated/reserved、实际触发数及最后一次全分辨率插值输入 shape/dtype。若无 OOM、loss finite 且显存余量合理，直接从 A2 epoch-420 source checkpoint **干净重启** 2560 次 successful updates，seed `772961337`，SwanLab online；训练成功后才进入现有四条件 Quick-Val。若仍 OOM，立即停止并报告，不擅改 batch、梯度累积或网络结构。
- **仍待执行：** 云端尚未 pull v2 commit，三步 GPU 验证和完整训练未开始；最终 checkpoint 和新效果指标不存在。A2 epoch-420 源 checkpoint 身份需在启动前按冻结 SHA-256 `2b72eb7f28e5c4b4f52bdc43e2ac5c7298169325a412bfbcbc7a022c29822597` 核验。Batch 1A C0 fixed-final checkpoint 真实位置/身份仍待核验，配对 Quick-Val 前须确认；冻结 SHA-256 `ca618b23d18eabb201a0d11d18da383ac99576d0feae5864e3233bda527d9a1a`。
- Batch 1B 十条件 Main-Val、Batch 2、T 与 official test 均未获授权；official test 仍为 `sealed_unread`（封存未读）。Batch 1A F-lite Main-Val 的独立上级裁决仍开放。

## 下一恢复点与证据

1. 复核本地提交的 v2 代码、config 身份与干净工作区；同步云端 commit 前确认远端写入授权与正确目标分支。
2. 若云实例仍停机，在授权范围内启动/连接，核对环境、source checkpoint、GPU 空闲及训练入口，运行 3 次连续成功更新并记录规定的显存与形状证据；依据条件授权决定正式重训或停止。
3. 若正式训练达成 `2560/2560`，核验 fixed-final 权重和匹配 C0 权重身份后，才按已有授权执行四条件 Quick-Val。

**证据入口：** [`v2 最小结构调整说明`](../../MMFR/01_research/e1_batch1_protocol.md)、[`v1 中止报告`](../../MMFR/02_evidence/report_e1_batch1b_roe_formal_training_attempt_20260924.md)、[`v1 Gate-B 报告`](../../MMFR/02_evidence/report_e1_batch1b_roe_gateb.md)。v1 报告与旧审核包保留历史身份，不解释为 v2 已通过。
