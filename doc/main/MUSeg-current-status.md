# MUSeg 当前状态与唯一实时入口

> **事实与执行边界截至：2026-10-04（Direction A 数值定位准备阶段）。** 用户已授权按“Natural短数值定位 → 证据驱动最小修复 → 短资格 → 三组从C0重新正式训练 → 完整S1 → 冻结gate裁决”推进。当前科研状态仍为 **INVALID/BLOCKED**；尚未得到新的诊断、修复、正式训练或S1结果。实例刚直接查询为 **Stopped/GPU0**。**大白话：** 获批查清基础组为何出现坏梯度，但还没有证据确定原因；不能续跑旧1597更新，也不能提前跑其他组或评价。

## 当前事实与诊断入口

- 上轮正式源码 `edb660a83da1ec67626149d06dc59460565e2c20`，Natural attempted1598 / successful1597 / 已记录optimizer skip0；attempt1598在optimizer执行前非有限梯度退出。最后成功loss0.1476122885942459、LR4.151961920329593e-07、AMP scale1024，epoch13。具体tensor/算子与失败步loss尚未定位，不直接归因AMP、NMF或数据。
- 上轮Grid/Replay及完整S1未运行，无合格fixed-final，无科研GO/STOP。失败日志已在本地 `outputs/natural-missing-round1-formal-restart-20261003/monitor/`；权威历史报告见[2026-10-03正式报告](../reports/2026-10-03-natural-missing-round1-formal.md)。历史源码与证据不改写。
- 唯一实例 `cpod-1vbh7faqcauq`，本对话直接API确认Stopped/GPU0。原环境历史身份Python3.10.16 / torch2.1.2+cu118 / CUDA11.8，下一次启动后仍需直接复核，不重建或升级依赖。
- 原C0为 `/root/rivermind-data/cloud/MMFR_E1_Batch1A_local_transfer_20260922/C0/checkpoint/update-2560.pth`，321150608 bytes；已核验SHA-256 `ca618b23d18eabb201a0d11d18da383ac99576d0feae5864e3233bda527d9a1a`。只加载分割权重，不继承旧optimizer/scaler/RNG，也不使用旧Natural或预检权重。

## 最新授权与停止边界

- 本地诊断准备中：记录首个非有限参数/层/算子、loss/梯度/AMP scale、attempt/successful、epoch/batch/sample身份及NMF前后/中间数值；只新增观测，不改变模型结构、数据、LR、训练预算和评价口径。NMF指分割解码器中的非负矩阵分解；AMP指混合精度训练，scaler负责缩放梯度。
- 先必要选择性commit/push至当前origin分支，再云端Git fast-forward与直接HEAD/源码干净核验；禁止复制不同版本源码训练。原有无关dirty保留，不reset，不将outputs/checkpoint/大日志加入Git。诊断埋点及持久screen入口已落地，主代理已直接复核NMF默认算术路径、诊断模式隔离、计数/异常停止与日志字段；静态诊断/编排语法/限定差异检查通过。本机额外小型tensor检查因既有OpenMP重复runtime冲突未完成，没有采用不安全绕过或修改环境；不记为运行通过。
- 先有限排查SwanLab既有配置/凭据来源及非交互初始化，不泄露token、不扩大环境改动；合理尝试失败则持久记录原因并LOG_ONLY，不长时间阻塞诊断。
- 诊断仅Natural从C0干净开始，沿原2560-update scheduler，限定到1664 successful updates，重点从attempt1536启用细粒度日志；诊断没有正式合格checkpoint，不能转正式resume。非有限立即停止，不跳坏update或吞异常。约原失败区间仍未复现则停止并报告证据边界，不擅自扩预算。
- 只在直接证据支持后采用最小数值修复；不借机改研究模块。Natural/Grid/Replay统一数值策略。修复后重新commit → push → 云端fast-forward → 同HEAD核验，再短资格，资格通过才开启新正式轮。
- 新正式顺序Natural → Grid → Replay，各自原C0干净初始化、各2560 successful updates、attempted=successful、optimizer skip0、每640 recovery和2560 fixed-final。任一资格/工程故障立即停止后续流程，不自行改protocol。
- 三组全部合法完成才运行完整S1（四权重×318样本×三条件=3816 views）。S1是冻结原尺度、无翻转、FP32开发评价；按[Direction A §6.5](../../MMFR/01_research/MMFR_direction_A_natural_missing_2026-10-02.md)固定门槛裁决，不因结果差改阈值。
- 云GPU启动后训练前必须直接核验自动定时关机保险，覆盖当前任务并留少量收尾；screen与日志持久保留。GPU任务结束立即主动关GPU；下载只在无卡/CPU-only直接Running/GPU0下完成，另有短时保险，取完立即关机并直接确认。
- Main-Val、official test、NYUv2、其他baseline、Round-2、Direction B/B1a、新seed/调参仍关闭；official test保持sealed_unread。A-v1 stop及F-lite/R-OE独立开放处置不变。

## 准确恢复点与证据

当前恢复点是完成诊断代码最小复核/门禁、选择性commit/push，再启动现有4090并核验Git/环境/保险，优先有限排查SwanLab后仅运行Natural诊断。实例尚未重启，尚无新云任务或数值修复结论；旧1597不是续训起点。授权持久记录见[首轮protocol §13](../../MMFR/01_research/natural_missing_round1_protocol.md)；旧阶段全部历史事实、生命周期与收口Git身份见上述正式报告及Git history。
