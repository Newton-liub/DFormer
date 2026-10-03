# MUSeg 当前状态与唯一实时入口

> **事实与执行边界截至：2026-10-04 01:33:29+08:00（Direction A数值定位完成，最小修复本地准备）。** Natural从原C0复现attempt1598 / successful1597的非有限反向传播，直接证据指向AMP下NMF内部数值不稳定。NMF局部FP32修复已在本地落地，**尚未通过短资格，当前仍INVALID/BLOCKED，不能科研GO/STOP**。诊断GPU已主动关闭，日志已通过CPU-only取回，直接确认实例**Stopped/GPU0**。**大白话：** 已查到坏梯度发生在矩阵分解的反向计算；修复还必须真正运行检查，合格后才能重新做三组对照。

## 已核验数值证据

- 诊断源码 `1cf9a8af517a1cf060846461cd31be5aba667950` 已commit/push，origin同分支直接ref一致；云端Git fast-forward、同HEAD/无tracked dirty核验后运行。Python3.10.16 / torch2.1.2+cu118 / CUDA11.8 / RTX4090 25280839680 bytes，原环境未升级或重建。
- 只运行Natural诊断，原C0干净初始化，cap1664、重型观测从1536；实际attempt1598 / successful1597 / optimizer skip0，epoch13/batch position61停止。last successful loss0.1476122885942459，与旧失败位置一致；失败步loss **0.18105733394622803** finite，AMP scale始终1024，backward未完成、unscale未开始、坏attempt未执行optimizer。
- batch RGB/Depth/Label均finite，NMF及全部已观察前向激活没有non-finite。首个观测坏梯度是 `decode_head.hamburger.ham.iteration.2.bases.divide`（NMF第二次乘法更新得到的基底张量），shape `[10,512,64]`、39 Inf / 0 NaN，其他有限值max55446.61328125。该节点梯度虽已经是FP32，周围autocast bmm/分母仍是FP16，不能把它误读为完整NMF已在FP32运行。
- anomaly traceback直接定位 `models/decoders/ham_head.py::NMF2D.local_step` 的 `denominator = bases.bmm(gram)`，`BmmBackward0`返回NaN；loop2分母前向min0、加epsilon后min1.0132789611816406e-06，相关反向有39 Inf，multiply梯度还有finite max54719987712。证据支持**AMP下NMF反向溢出/非有限传播**，不是已发现的输入NaN/Inf污染或scaler增长/解缩放故障；不据此宣称所有数据质量均已核验。
- anomaly提前停止后参数梯度只部分生成，首个坏named parameter为null，不能把partial currently_available finite摘要当作完整反向通过。首次坏中间tensor、NMF阶段和产生NaN的operator已有直接证据。

## 最小修复与下一门禁

- 本地修复仅将NaturalMissing三组共同 `nmf_training_precision` 设为 `fp32_local`，共享builder只给原NMF2D设置训练期flag；在训练态/外层CUDA autocast启用时，对NMF内部关闭autocast并输入float32。外围AMP fp16/scaler1024、TF32、模型结构/参数、NMF算法/迭代/随机基底、数据/LR/scheduler/正式预算/S1不变。旧A-v1分支不启用；S1 eval态策略仍按原FP32/TF32off。
- 修复源码尚待最小静态/差异检查、commit/push和云端Git同HEAD核验；**尚无修复运行成功证据**。下一步Natural从原C0运行至1664的限定资格，确认越过已复现1598故障区间且全程finite；不使用诊断/旧Natural权重，不产生正式资格checkpoint。
- 资格通过才从原C0分别干净重启 Natural → Grid → Replay，各2560 successful updates、attempted=successful、optimizer skip0、每640 recovery/2560 fixed-final。任一组资格性故障立即停止后续流程，不吞坏update、不自行改protocol。
- 三组全部合法完成才完整S1：C0/Natural/Grid/Replay×318样本×3条件=3816 views。沿[Direction A §6.5](../../MMFR/01_research/MMFR_direction_A_natural_missing_2026-10-02.md)原门槛GO/科研STOP/工程INVALID，不改阈值。三组新正式/S1尚未运行。
- Main-Val、official test、NYUv2、其他baseline、Round-2、Direction B/B1a、新seed/调参仍关闭；official test保持sealed_unread；A-v1 stop及F-lite/R-OE独立处置不变。

## SwanLab、资源与证据入口

- SwanLab0.9.7：云端登录shell中相关key/token/mode环境为空、默认netrc不存在；一次bounded online初始化在SDK `swanlab/sdk/cmd/init.py::prompt_init_mode` line323抛RuntimeError、interactive=False。当前**LOG_ONLY**，没有online链接/上传完成证据；完整凭据来源逻辑还需有限SDK源检查，不把“无环境key”单独当作所有配置均不存在的证明。没有泄露token或改环境，不因监控继续阻塞诊断。
- 唯一实例 `cpod-1vbh7faqcauq`。正式诊断窗口API StartTime1791047409 / StopTime1791048748（22分19秒）；保险1791050100在启动前后直接核验。screen `natural-missing-numerical`、durable job `natural-missing-numerical-20261004-1cf9a8a`；receipt wall约1320.56秒。任务Failed后主动停GPU，CPU-only保险1791049661、Running/GPU0取回exit0，最终01:33:29直接Stopped/GPU0。
- 先前两个零训练准备窗口因远端命令引号及可选SDK introspection KeyError主动停机，GPU窗口55秒/18秒，无Natural尝试，不作为数值诊断结果；CPU-only准备窗口21秒完成Git同步/原环境检查。细节保留本轮run identity和命令证据，不把报错终端的本地Bash PASS误标当作真实通过；Bash语法已在云端实际通过。
- 本地证据根 `outputs/natural-missing-numerical-20261004-1cf9a8a/`：`run-identity.json`、`numerical-lifecycle.json`、`monitor/numerical-run-receipt.json`、Natural stdout/stderr、screen.log。远端同名根在 `/root/rivermind-data/cloud/`。无诊断checkpoint，不取回旧不完整权重。旧失败事实见[2026-10-03正式报告](../reports/2026-10-03-natural-missing-round1-formal.md)，原报告/日志不改写。
- 本轮实际检查：直接代码差异复核、diagnostic AST/独立parser门禁、Python编排语法/PowerShell parser/云端Bash语法、Git/原环境/保险/诊断日志/CPU-only取回与最终云状态。额外本机torch小检查受既有OpenMP runtime冲突未完成，未不安全绕过。未运行完整测试、正式三组、S1、额外数据审计或official test。

**准确恢复点：** 实例Stopped/GPU0；数值定位已完成，不重复诊断。提交并push最小NMF FP32修复，云端Git同步后核验HEAD/环境/保险，有限确认SwanLab凭据分支，再仅Natural限定资格；资格前不得启动Grid/Replay或S1。授权与精度修订见[首轮protocol §13–14](../../MMFR/01_research/natural_missing_round1_protocol.md)。
