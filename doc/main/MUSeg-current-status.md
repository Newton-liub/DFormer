# MUSeg 当前状态与唯一实时入口

> **事实与执行边界截至：2026-10-03（Direction A 正式首轮已授权，正在准备 Git 同步与顺序编排）。** 用户本轮执行指令批准现有4090实例上运行Natural/Grid/Replay各2560成功更新，全部合法完成后进行C0+三组完整318图三条件S1（3816 views）；尚未启动正式训练或云实例。本轮允许必要提交及同分支push，正常结束主动停GPU，再CPU-only取回并停机。

## 当前阶段与实际意义

Natural保留自然输入，Grid增加规则网格删除，Replay重放其他训练采集组的真实空洞；checkpoint是模型参数与训练状态快照。S1是冻结的原尺度、无翻转单视图开发评价，GO/STOP是预定资源筛选门槛，不是统计显著性。

- 科研合同保持：全部分割参数、仅分割loss；旧辅助/A2混合故障/F-lite/R-OE/A-v1关闭。每组2560成功更新，batch10、480×640、workers8、LR1e-6、AdamW/WD.01、warmup128/poly.9、AMP fp16/scaler1024、TF32on/SyncBN、原生NMF autocast。每640成功更新保存recovery，最终固定`update-2560.pth`；无训练val或best选择。
- 三组固定Natural→Grid→Replay，每组重新独立加载精确C0：`/root/rivermind-data/cloud/MMFR_E1_Batch1A_local_transfer_20260922/C0/checkpoint/update-2560.pth`，321150608 bytes，冻结SHA-256 `ca618b23d18eabb201a0d11d18da383ac99576d0feae5864e3233bda527d9a1a`；不重复下载/hash，不用preflight权重或组间续训。
- 完整S1仅在三组全部合法完成后执行：C0/Natural/Grid/Replay×val-dev318×natural/冻结矩形/entire-missing，3816 views。原尺度1/no flip/batch1/whole image、右下pad32、FP32/TF32off、裁padding回Label网格，固定观测与sample×condition RNG、原冻结缺失分层/15类政策/采集组confusion。按canonical Direction A §6.5裁决，不改门槛或救结果。
- Non-finite、未解释optimizer skip、checkpoint/模型/数据合同错误即停止；不改LR、batch、precision、NMF、geometry或预算，不自动重试/接续。SwanLab（在线实验记录服务）仅有限排障，成功用ONLINE，合理尝试仍失败用LOG_ONLY继续，不把监控故障当科研失败。

**大白话：** 之前4090只证明能按原合同训练几步；这次获批的是完整首轮对照和统一评价，结果出来前还不能判断Replay是否值得继续。

## 云资源、授权与已核验边界

- 唯一实例`cpod-1vbh7faqcauq`，本轮开始直接查询仍**Stopped/GPU0**；已有4090限量预检READY_ENGINEERING_ONLY，三组各attempted3/successful3/skipped0及保存/CPU读回通过，不代表长期稳定或真实GPU resume。
- GPU启动前设置并直接确认**6小时平台自动关机保险**，Running后复核RTX4090及同截止；不新建/resize实例，不无限延长。任务正常结束或失败主动stop并查询Stopped/GPU0；完整S1与收据落盘后不为下载保持GPU运行。
- GPU停止后同实例CPU-only取回必要正式权重/S1/summary/日志，先设20–30分钟保险，取回完成或失败主动停机；不取preflight权重、大cache或数据集。
- 现有本地分支`perf/mmfr-a2-v3-pipeline-opt1`，开始HEAD`6ec6bf8eee92e6374cabf81c6c49c16d05a25488`，origin为canonical DFormer仓库。已直接检查已有目录审计/执行单及共享索引dirty，保持不覆盖、不混入本轮提交；formal-run SHA待必要编排提交推送后记录。
- Direction B/B1a、NYUv2、外部baseline、Main-Val、新seed/调参/补实验仍关闭；official test仍**sealed_unread**。A-v1保持stop，F-lite/R-OE独立未决。用户本轮授权原文为本地`临时/2026-10-03-mmfr-direction-a-round1-formal-execution.md`，持久执行边界见[protocol §10](../../MMFR/01_research/natural_missing_round1_protocol.md#10-2026-10-03-direction-a正式首轮最新授权)。

## 证据与准确恢复点

- Canonical [Direction A](../../MMFR/01_research/MMFR_direction_A_natural_missing_2026-10-02.md) §6.5为冻结门槛；[readiness报告](../reports/2026-10-03-natural-missing-round1-readiness.md) §13保留4090预检事实，历史实时快照从Git`6ec6bf8`追溯，不改原始证据。
- 已完成：本轮授权、冻结合同/已有训练与S1入口、Git与实例初态只读核对；最小正式编排、screen入口、本地自动停机/CPU取回监督和CPU-only正式产物读回工具已准备，并由主执行者直接复核关键合同、目录、遥测与资源边界。Python/Bash/PowerShell语法及两Python文件静态诊断通过，文档/差异检查通过；未运行完整测试或额外GPU验证，正式更新/S1尚未开始。
- **准确恢复点：** 完成并复核最小编排，做必要静态/差异检查，选择性提交push；记录formal-run SHA后才设置6小时保险和启动4090。云端只轻量fast-forward与HEAD核对；有限SwanLab排障后进入screen顺序正式流程。任何中断保留当前receipt/日志与最新合法recovery，不擅自追加训练。
- 最终正式报告目标`doc/reports/2026-10-03-natural-missing-round1-formal.md`，目前尚未生成。旧生成审核包为历史快照，当前本轮实时权限只以本页为准。
