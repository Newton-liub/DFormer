# MUSeg 当前状态与唯一实时入口

> **事实截至：** 2026-09-24 的 E1 Batch 1B 正式训练中止记录；本次仅整理文档，未重新核验云端实时状态或启动实验。
> **用途：** 只回答当前阶段、最近一个大动作、授权边界和恢复点。研究选择见 [`MUSeg-open-decisions.md`](MUSeg-open-decisions.md)；整理前的完整历史见 [`归档快照`](../../MMFR/90_archive/2026-09-24_live_docs_rolling_window/README.md)。

## 当前阶段与最近一个大动作

**阶段：** MMFR（多形式模态失效与可靠性）E1 Batch 1B，`R-OE-formal-training-blocked-oom`。R-OE-lite 是仅依据当前深度输入判断“当前裁剪区域没有非零深度观测”并尝试用 RGB 生成替代深度的候选；实现资格 Gate-B 已通过，**不代表训练或效果通过**。

**最近动作：** 用户授权的 R-OE-lite 云端正式训练于 2026-09-24 01:40:43 UTC 启动，01:45:30 UTC 在 `models/roe_substitute.py:85` 的 `F.interpolate` 遇 CUDA 显存不足而退出。日志至少确认 **168 次成功 optimizer update**，最终精确计数未落盘；未生成 fixed-final `update-2560.pth`，所以四条件 Quick-Val（开发集快速筛查）未运行。进程与 screen 会话已退出，OOM 后未自动重试。**大白话说明：** 训练确实开始了，但在生成最终权重前因显存不足中止，现在没有 R-OE-lite 的效果结论。

## 当前边界与阻塞

- **授权：** 此前授权涵盖该次 Batch 1B 正式训练及**训练成功后**的四条件 Quick-Val；本次因训练失败不满足评价前提。OOM 后不自动重训、不改冻结配置；是否以新方案恢复或停止 Batch 1B 等待用户决定。Batch 1B 十条件 Main-Val、Batch 2、T 和 official test 未获授权；official test 仍为 `sealed_unread`（封存未读）。
- **关键阻塞：** 现有 batch size 10 的 R-OE-lite 在 RTX 4090 上发生 CUDA OOM，缺少最终 checkpoint。Batch 1A 的匹配对照 C0 fixed-final checkpoint 尚未在已查的云端候选路径找到，真实位置待核验；运行任何配对 Quick-Val 前必须核实其可读性与身份。
- **云资源待核验：** 中止后 GPU 空闲、训练进程已退出，但实例 `cpod-1vbh7faqcauq` 的**控制面停机状态未核验**；不能用 GPU 空闲推断不再计费。应先用有权限的控制面核对实例状态，再决定是否采取停机操作；本次文档整理未触碰云资源。

## 恢复点与证据

1. 先核对云实例控制面状态和 Batch 1A C0 fixed-final checkpoint 的实际位置/身份；C0 冻结 SHA-256 为 `ca618b23d18eabb201a0d11d18da383ac99576d0feae5864e3233bda527d9a1a`，路径目前待核验。
2. 等用户对 OOM 后方案作决定。若改变模型、batch size 或共同训练合同，应建立新身份并重新资格化/核对匹配 C0；不得以旧 Gate-B 或旧授权直接启动新训练。

**证据入口：** [`Batch 1B 中止报告`](../../MMFR/02_evidence/report_e1_batch1b_roe_formal_training_attempt_20260924.md)（云端原始 `train.log` 的位置、SwanLab run 与计数边界）、[`Batch 1B Gate-B 报告`](../../MMFR/02_evidence/report_e1_batch1b_roe_gateb.md)、[`E1 冻结协议`](../../MMFR/01_research/e1_batch1_protocol.md)。历史实验事实和已关闭选择保留在归档快照、正式报告及原始证据中，不在此重复。
