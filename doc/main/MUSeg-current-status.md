# MUSeg 当前状态与唯一实时入口

> **事实与执行边界截至：2026-10-03 23:11:39+08:00（Direction A正式运行因非有限梯度停止，证据取回且实例已关闭）。** 运行源码`edb660a83da1ec67626149d06dc59460565e2c20`已push，云端同HEAD/无tracked dirty；Natural **attempted1598 / successful1597 / 已记录optimizer skip0**，第1598次尝试在optimizer更新前非有限梯度退出。Grid/Replay与完整S1未运行，结论**INVALID/BLOCKED，不能给科研GO/STOP**。最终直接确认`cpod-1vbh7faqcauq` **Stopped/GPU0**。**大白话：** 基础对照组先出现数值问题，还无法比较Replay效果；原执行已按合同停止，不能自动改精度续跑。

## 当前结果与边界

- Natural最后成功更新1597，loss0.1476122885942459、LR4.151961920329593e-07、AMP scale1024；第1598次attempt_started后，stderr明确`FloatingPointError: NaturalMissing gradients are non-finite; refusing to skip the optimizer update`。失败前所有成功更新均skip0，失败尝试不算成功或被跳过后继续。具体非有限tensor/产生算子未记录，不能直接归因NMF、AMP或数据。
- 三组各2560成功更新与四权重完整S1（3816 views）的授权没有完成；非有限停止条款已触发，**当前无自动重试、resume或改变科研合同权限**。没有合格fixed-final/S1结果或科研门槛裁决。Natural输入15980槽含失败尝试，全部clean、新增删除0、已观测输入违例0；不代表未运行Grid/Replay匹配与跨组排除已验证。
- 原C0仍为`/root/rivermind-data/cloud/MMFR_E1_Batch1A_local_transfer_20260922/C0/checkpoint/update-2560.pth`，321150608 bytes，既有核验SHA-256`ca618b23d18eabb201a0d11d18da383ac99576d0feae5864e3233bda527d9a1a`；本次不重复hash/下载，不使用预检或组间权重。流程经过640/1280保存点，但recovery文件/schema/hash未直接读回，不授予resume资格。
- 合同未改：全部分割参数/segmentation-only；batch10/480×640/workers8、LR1e-6/AdamW/WD.01、warmup128/poly.9、AMP fp16/scaler1024/TF32on/SyncBN、原生NMF autocast；每640 recovery、2560fixed-final，无训练val/best选择。旧aux/A2混合故障/F-lite/R-OE/A-v1关闭。
- Direction B/B1a、Main-Val、NYUv2/baseline、Round-2、新seed/调参及追加预算仍关闭；official test保持**sealed_unread**。A-v1保持stop，F-lite/R-OE处置独立未决。本次INVALID不能自动变成Direction A科研STOP或开放下一方向。

## 资源、监控与实际时间

- 唯一实例`cpod-1vbh7faqcauq`；资源不足启动失败后用户手动启动，API GPU StartTime1791038472（22:41:12），RTX4090/GPU1/CPU14/32768MiB；CUDA报告显存25280839680 bytes。原环境Python3.10.16/torch2.1.2+cu118/CUDA11.8，未安装/升级依赖。
- 原GPU保险截止2026-10-04 02:11:47+08:00 / Unix1791051107在Running直接复核，不延长。任务`natural-missing-formal-20261003-edb660a`及screen `natural-missing-round1-formal`持久运行；receipt wall1344.1179秒（22分24.12秒），Natural child1343.4608秒。
- 监督器获得Failed后主动stop，API StopTime1791039984（23:06:24），23:06:32直接Stopped/GPU1配置规格；GPU运行窗口1512秒（25分12秒），未为下载保留GPU。
- CPU-only失败证据取回沿同一30分钟保险23:37:54 / Unix1791041874；前两次Initializing状态与PowerShell UTF-8 JSON解码问题均立即stop、未取回；第三次明确等待Running/GPU0，monitor transfer exit0于23:10:57完成，API StopTime1791040259（23:10:59）。显式等待后23:11:39直接**Stopped/GPU0**，三个CPU API运行窗口合计34秒。最终账单未核验。
- SwanLab（在线实验记录服务）仍为**LOG_ONLY**，初始化RuntimeError、finish_returned=true，无online链接/上传完成证据。screen/日志/receipt完整保留；监控故障不是已证明的非有限梯度原因。

## 交付、证据与准确恢复点

- [正式执行报告](../reports/2026-10-03-natural-missing-round1-formal.md)已落盘：实际训练、停止位置、输入边界、无S1/无GO-STOP、资源时间和下一授权。报告与实时入口/必要索引按用户许可选择性Git收口；formal-run SHA与收口提交区分，实际提交/推送身份以canonical同分支Git记录为准。原有无关审计dirty保留，权重/outputs/大日志/cache不入Git。
- 本地根`outputs/natural-missing-round1-formal-restart-20261003/`：`run-identity.json`、`restart-capacity-receipt.json`、原始`lifecycle.json`、`nonfinite-closeout.json`及`monitor/`中的formal receipt、Natural summary、stdout/stderr、screen.log。远端根`/root/rivermind-data/cloud/natural-missing-round1-formal-20261003-edb660a/`；没有三组合格final或完整S1本地产物，不运行成功final专用读回工具。
- 首次源码`8d2490ef433807ec60d475cab27fbadd04d18e1f`漏传`--successful-updates 2560`，Natural参数门禁0尝试/0成功更新；修复后用户另批继续原预算，已用于当前实际运行。首次证据在`outputs/natural-missing-round1-formal-20261003/`独立保留，不覆盖历史结果。
- 收口仅修正资源监督的UTF-8 JSON解码与显式start/stop `--wait`，避免Initializing/Stopping竞态；未用于再次GPU运行。实际检查包括原CPU CLI门禁、Python静态检查、本次PowerShell parser/差异复核、运行日志及直接云状态核验；未运行完整测试、额外GPU验证、真实resume、非有限定位训练或S1，不把未运行写成通过。
- 授权/科研门槛见[首轮protocol](../../MMFR/01_research/natural_missing_round1_protocol.md)与[canonical Direction A §6.5](../../MMFR/01_research/MMFR_direction_A_natural_missing_2026-10-02.md)，旧工程预检只见[readiness报告 §13](../reports/2026-10-03-natural-missing-round1-readiness.md)。历史生成review packet仍是旧A-v1快照，不冒充本次结果、不重建或手改。
- **准确恢复点：** 实例保持Stopped/GPU0，原正式任务已终态Failed，取回与监督进程已退出，不重复提交。先由用户/上级裁决是否授权限定非有限梯度定位与重新资格，或终止本次执行准备；任何GPU诊断、AMP/NMF调整、resume/重训与新预算必须另获明确授权。现有日志最后成功1597、失败attempt1598（epoch13）是定位入口，不是允许续训的位置。
