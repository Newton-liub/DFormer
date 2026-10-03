# MUSeg 当前状态与唯一实时入口

> **事实与执行边界截至：2026-10-03（4090限量预检启动前准备）。** 用户已批准现有实例有卡模式，仅Natural/Grid/Replay各最多3成功更新、云端workers8；先提交并推送，再设置并核验30分钟平台关机保险，启动后拉取同一代码提交。当前直接观测实例仍Stopped，尚未执行本轮有卡启动或训练；既有C0/极小S1通过、本机OOM事实保留。正式训练和完整评价仍关闭。

## 当前事实与实际意义

C0是已有E1训练对照；fixed-final是2560成功更新后的固定权重。checkpoint是参数/训练状态快照。Natural保留原输入，Grid新增规则网格删除，Replay重放其他训练采集组的真实空洞。support是不含padding的真实图像区域；validity是其中有深度观测的位置。OOM是GPU内存不足，不能完成当前分配；S1是新统一原尺度单视图评价，不是旧Quick-Val。

- A/B正文已原样迁至 `MMFR/01_research/`，Git100% rename；历史runner/config/evidence冻结，仅loader显式opt-in扩展。代码/初始CPU检查详见[现有readiness报告](../reports/2026-10-03-natural-missing-round1-readiness.md)。
- 科研合同保持：同精确C0、全部分割参数、仅分割loss，旧可靠性辅助/A2混合故障/F-lite/R-OE/A-v1关闭；LR1e-6、AdamW/WD.01、warmup128/poly.9、batch10/480×640、正式workers8、AMP fp16/TF32on/SyncBN、原生NMF autocast。输入取消额外删除不等于optimizer跳步；后者或非有限必须停止。未为本机OOM改batch/geometry/precision/allocator或追加尝试。
- 精确C0本地路径：`D:/0Project/DFormer/outputs/natural-missing-readiness-20261003/C0/update-2560.pth`，321150608 bytes；SHA-256 **`ca618b23d18eabb201a0d11d18da383ac99576d0feae5864e3233bda527d9a1a`**。直接确认关机后hash **1次**；真实strict segmentation load **loaded802 / intentionally dropped10 / missing0 / unexpected0**，只过滤已知辅助键，无全局strict=False。各后续进程重载原C0但不重复hash；未重训/替换C0。
- 本机df2 Python3.10.20/torch2.7.0+cu128、RTX5060 Laptop GPU8151MiB；Natural/Grid/Replay分别独立从C0启动，各 **attempted1 / successful0 / optimizer skipped0**。三组首次forward在width方向depth-decay分配额外470MiB时OOM，尚未返回loss/进入backward/optimizer更新。Grid/Replay实测peak allocated7054.634MiB、reserved7354MiB；Natural仅异常当时内存，不冒记峰值。无成功step时间、loss finite/scaler运行通过、保存或真实GPU resume证据。
- 真实首批配对输入：两组同10目标槽位/源身份，4clean、5matched、1paired_skip；5个已应用匹配最大误差0.00939993≤.02。第7槽8候选后匹配失败，两组实际新增0；最后失败候选误差0.02546536不是已应用超标。输入配对工程行为通过，不代表训练或Replay研究效果通过。
- 真实C0 S1：allowlisted val-dev前2图×3条件 **6次FP32/TF32off前向**，finite与原Label网格crop通过，6次sample×condition RNG复位；同1采集组、原全图low2/high0，每条件独立698762有效Label像素。矩形新增476439/467727点，均为原当前有效点精确50%；自然新增0、entire-missing新增952878/935454点。示例932×1082→960×1088，右6/底28 pad后metric前crop。小样本仅工程证据，不公布或使用正式Round-1分数，不验证high集合/跨checkpoint复用/完整评价。

**大白话：** 共同起点权重已找回且真正加载成功，少量真实评价也能跑；本机8GB GPU在合同batch下连首次训练前向都无法完成，因此还不能正式训练，也没有有效速度可估算总耗时。

## 云资源、授权与未完成项

- 唯一现有实例 `cpod-1vbh7faqcauq`，2026-10-03 **08:37:26+08:00**请求 `--without-gpu A` 启动，08:37:37直接确认Running、CPU2/内存4096MiB/**GPU0**。只取回原远端C0文件，transfer exit0；未取整ZIP/F-lite/R-OE，未执行云模型、训练或hash。
- 下载成功后 **08:41:10+08:00立即请求stop**，**08:41:23+08:00直接查询确认State=Stopped、GPU0**，API StopTime1790988072。实例名称/GpuType4090只是历史标签，实际本次GPU数量0；未启动GPU/新建资源/再次开机。API InstancePrice0.13，最终费用未核验，不将价格字段当最终账单。
- **本轮新授权：** 用户批准现有 `cpod-1vbh7faqcauq` 有卡模式，仅三组各≤3成功更新，Linux workers8、同C0/科研合同不变；训练类检查全部云端，不重复本机GPU。先提交必要代码，授权将已有3提交和本轮必要提交推送到origin同分支，云端HEAD一致才运行；既有审计dirty不纳入。
- 启动前优先设置并直接核验30分钟平台关机保险；若只能运行态设置，启动后立即设置，成功前不提交训练，失败立即停机。复用已有SwanLab配置，有条件时提供在线整体进度和完成/失败状态；训练子进程与监控进程隔离，不改变训练RNG。
- 正常结束主动停机并确认Stopped，不等待20分钟；若用户因SwanLab完成/中断后20分钟无下一步人工关机，发现后记录并退出，不自动重启。正常结束后如需取小型证据可CPU-only短时取回再停机，不因大文件下载保持GPU运行。
- 正式训练 **3×2560=7680成功更新**、完整S1 **C0+三组×318×3=3816 views**、Main-Val/B1a33072 views、NYUv2/外部baseline及其他高成本验证仍未授权。official test仍 **sealed_unread**。
- 当前待验证：云端成功loss/backward/optimizer/scaler/save链、成功step测时和allocated/reserved峰值。发生OOM/非有限/身份冲突或需要改变科研合同时本轮停止，不改batch/precision挽救；云端通过后只提交工程资格和正式预算估计，正式预算仍需裁决。

## 旧路线结论保持

A-v1仍stop：learned三hard−off为−0.0018730026999946858pp，未达+0.50pp；F-lite Main去留、R-OE v2约+0.01pp及入口资格仍独立待裁决；Oracle-A仅关闭具体抑制动作族。新A不是旧A-v1，不自动复活任何旧路线。历史正式报告/Git history保留，不回写旧结果。

## 证据、提交与准确恢复点

- [本轮readiness报告](../reports/2026-10-03-natural-missing-round1-readiness.md)已更新原报告，记录真实取回/停机/加载/OOM/极小S1、最小改动和预算边界；[首轮protocol](../../MMFR/01_research/natural_missing_round1_protocol.md)不改科研合同。canonical [A优先](../../MMFR/01_research/MMFR_direction_A_natural_missing_2026-10-02.md)/[B备用](../../MMFR/01_research/MMFR_direction_B_inference_protocol_2026-10-02.md)。
- 本地忽略证据：`outputs/natural-missing-readiness-20261003/C0/recovery-receipt.json`（完整键清单/停机收据）、`execution-summary.json`、三组`*-gpu.log`、`s1/checkpoints/update-2560-ca618b23d18e.json`；权重/日志/outputs不入Git。既有[C0转移收据](../../MMFR/02_evidence/delivery_mmfr_a_v1_local_val_20261001.md)保留原始事实。
- 本次代码只给训练入口增加attempt/reserved telemetry，评价入口显式复用已核验C0身份避免第二次hash；实际差异/输出定点核对与两文件静态诊断已完成。未重复完整测试或初始CPU套件、未新建test脚本；失败不记PASS。既有MMFR导航/changelog与report-index同步，旧生成审核包仍历史快照，不重建/手改。
- 本轮执行起始HEAD `d47a4be1c139cde1a9c304b5bf36239069ec2bd0`（前一轮readiness本地收口），此前实现提交 `caaeda289cb68767c9fa3e0ea011e2f4e4ede1ec`、目录提交 `ca65f4bdfc76e687746745df12e7c767a0a46049`。此前未推送；本轮已获明确推送授权。新增云端wrapper仅顺序调用三个独立preflight子进程，固定3成功更新/workers8，训练入口只增加flush；两文件静态诊断无报错。监控与训练RNG隔离，compact receipt保留计数、step、checkpoint位置与输入telemetry。用户既有目录审计/执行单及其索引/导航dirty选择性排除并保留。

**准确恢复点：** 完成启动前监控最小代码/静态复核与授权记录提交，推送origin同分支；设置并确认30分钟平台关机保险，现有实例有卡启动后拉取并核对同一代码SHA，再顺序三组各≤3成功更新。所有结束路径优先停机；人工保险关机后记录退出。小动作不逐条写状态，整体启动/结束或中断统一更新。C0不重新hash/取回，不运行正式实验。
