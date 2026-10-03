# MUSeg 当前状态与唯一实时入口

> **事实与执行边界截至：2026-10-03（4090限量云端预检已完成并停机）。** Natural/Grid/Replay各从同一精确C0独立完成 **attempted3 / successful3 / skipped0**，工程结论 **READY_ENGINEERING_ONLY**。云端运行源码与本地已推送提交一致，三个预检checkpoint已保存并在云端CPU-only读回核对。最终2026-10-03 **09:46:59+08:00直接确认实例Stopped/GPU0**。正式训练、完整评价和official test仍关闭。

## 当前事实与实际意义

C0是已有E1训练对照；checkpoint是模型参数和训练状态快照。Natural保留自然输入，Grid新增规则网格删除，Replay重放其他训练采集组的真实空洞。AMP是混合精度训练；optimizer skip是没有成功执行参数更新，和输入配对失败后取消额外删除是两回事。

- 科研合同保持：全部分割参数、仅分割loss，旧辅助/A2混合故障/F-lite/R-OE/A-v1关闭；LR1e-6、AdamW/WD.01、warmup128/poly.9、batch10/480×640、workers8、AMP fp16/TF32on/SyncBN、原生NMF autocast。不改batch/geometry/precision、没有预算外重试或接续预检权重。
- 精确C0沿用原远端文件 `/root/rivermind-data/cloud/MMFR_E1_Batch1A_local_transfer_20260922/C0/checkpoint/update-2560.pth`，321150608 bytes；冻结SHA-256 **`ca618b23d18eabb201a0d11d18da383ac99576d0feae5864e3233bda527d9a1a`**。先前本地关机后单次hash及strict load802/已知aux dropped10/missing0/unexpected0证据保留；本轮三个进程各严格重载原C0，**没有重复hash或重新下载C0**。
- 云端干净运行提交 **`18271ad0b0c3e7ba81c630c099d02592c02a45f4`**；分支 `perf/mmfr-a2-v3-pipeline-opt1`，origin为DFormer canonical仓库。先推送已有3本地提交及必要监控代码，云端从已确认干净checkout fast-forward并直接核对HEAD后才运行；无reset或历史dirty覆盖。
- 实际环境：RTX4090 **24564MiB**、CPU14/32768MiB；既有Python3.10.16、torch2.1.2+cu118/CUDA11.8。只核对所需数据目录/C0大小/train-dev1277条，未重建环境或安装依赖。
- 持久任务 `natural-missing-20261003-18271ad`，09:41:27–09:42:32+08:00，**Succeeded/exit0**。三组各3成功更新，共9；九次loss finite、gradient检查、optimizer/scaler路径通过，scale均1024。首次warmup LR0仍记录optimizer调用成功，后两次LR分别7.8125e-9/1.5625e-8；不把9步称为正式训练完成。
- 三组allocated/reserved峰值：Natural、Grid **18955.576/20556MiB**；Replay **18954.139/20554MiB**。成功step2–3均值约Natural0.7713s、Grid0.7117s、Replay0.6831s；只含输入上GPU后的计算链，短样本仍在warmup，**不含取batch/配对构建/保存/加载/评价**。正式7680步的条件性纯计算线性估计约92.42分钟，实际总墙钟/完整费用需另批预算。
- 每组 `preflight-update-000003.pth` 云端CPU-only读回：802模型键、685 optimizer state且step均3、global update3、cursor3/3/0、scaler growth tracker3、训练/epoch RNG保存、同Git/C0/合同身份。均标记 **preflight-only / formal eligible false**；只是保存/读回通过，**没有执行真实GPU resume**。
- 三批输入共每组30槽；三组目标顺序一致，Natural新增删除0；Grid/Replay源身份/status/候选历史逐槽相同，均7clean、20matched、3paired_skip。20matched均跨源目标采集组，最大误差0.01973145≤.02，已应用Replay率0.11626508–0.46178596。clean/skip的实际新增删除均0；失败候选不当作已应用值。输入取消额外删除后仍成功训练，不是optimizer skip；小样本不估计长期skip率或策略效果。
- SwanLab（在线实验记录服务）显式请求online，但初始化 **RuntimeError**，本轮 **LOG_ONLY**，没有在线实验链接或可见上传完成证据；根因未进一步核验，未暴露凭据/安装依赖。receipt终态SUCCEEDED、terminal exit0；在线完成/失败刷新路径本轮未实际验证。

**大白话：** 4090能在原合同下完成少量真实训练更新，并把状态可靠保存、读回，之前8GB本机的容量阻塞已解除。现在可以提交正式预算供审批，但仍不知道完整训练是否长期稳定，也没有证明Replay效果更好。

## 云资源与已执行授权

- 唯一实例 `cpod-1vbh7faqcauq`；启动前设置 **30分钟平台保险，10:07:28+08:00 / Unix1790993248**，读取确认scheduled=true；09:37:50请求有卡start，直接确认Running/GPU1，启动后再次核验同截止，没有延长或改变付费规格。
- GPU任务正常结束后 **09:43:01+08:00主动stop，09:43:13+08:00直接查询确认Stopped**，API StopTime1790991782；没有等20分钟或为下载保持GPU运行。运行态API InstancePrice1.88，不把CPU价0.13/resize-price0当GPU价；最终账单未核验。
- 正常结束后按授权，仅为checkpoint CPU读回与5个小型证据取回，另设 **10分钟保险，09:54:10+08:00 / Unix1790992450**，09:44:16请求 `--without-gpu A`，直接确认CPU2/4096MiB/**GPU0**。没有模型前向、训练或新增更新；不下载三个大checkpoint。
- 五个文件transfer均exit0后 **09:46:49+08:00立即stop，09:46:59+08:00直接确认Stopped/GPU0**，API StopTime1790992011。全过程无人工保险关机事件、无自动重启训练；CPU取回阶段是正常结束后的已授权操作。**10:01:58+08:00最终复查仍Stopped/GPU0**；CPU保险已到期，API SchedulerStopTime=null，没有延长保险或自动开机。
- 本轮≤3/组预算已经用完并关闭。正式训练 **3×2560=7680成功更新**、完整S1 **3816 views**、Main-Val/B1a33072 views、NYUv2/外部baseline仍未授权，official test仍 **sealed_unread**。真正未决事项见[open-decisions](MUSeg-open-decisions.md)。

## 证据、提交与恢复点

- [现有readiness报告](../reports/2026-10-03-natural-missing-round1-readiness.md) §13记录本轮完整证据，§11为条件性预算；[首轮protocol](../../MMFR/01_research/natural_missing_round1_protocol.md) §8保留授权，科研内容不改。canonical [A优先](../../MMFR/01_research/MMFR_direction_A_natural_missing_2026-10-02.md)/[B备用](../../MMFR/01_research/MMFR_direction_B_inference_protocol_2026-10-02.md)。
- 远端持久证据根 `/root/rivermind-data/cloud/natural-missing-preflight-20261003-18271ad/`；本地 `outputs/natural-missing-cloud-preflight-20261003/` 的 `preflight-result.json`、`checkpoint-readback.json`、`Natural.log`/`Grid.log`/`Replay.log` 与 `lifecycle.json`。三个预检权重只留远端 `checkpoints/NaturalMissing-<strategy>-preflight/development/seed-772961337/checkpoint/preflight-only/`，不入Git。
- 前阶段原始C0/本机OOM/2图S1结果留在同报告§3/§8/§9及 `outputs/natural-missing-readiness-20261003/`；旧实时全量版本可从 `d47a4be` 与 `18271ad` Git history追溯，不复制历史流水。A-v1仍stop，F-lite/R-OE独立未决，不因新预检复活旧路线。
- 最小代码仅新增顺序监控wrapper、训练 `_emit` flush，模型/输入/optimizer合同未改；主执行者直接复核差异、实际结果与checkpoint关键证据，两代码文件静态诊断无报错。本轮没有完整测试、初始CPU重复检查、额外GPU训练、评价或一次性test脚本。
- 本轮代码已提交并推送；本次文档收口按 `docs(mmfr): close bounded 4090 preflight` 提交并推送同origin分支，确切收口SHA以Git为准，不自引用追加。既有目录审计报告/执行单及共享索引中的无关dirty选择性排除并保留；生成审核包仍历史快照，不重建/手改。

**准确恢复点：** 保持实例Stopped，向用户/上级提交正式三组训练与完整S1预算、设备/时间/责任审批。获新授权前不重启GPU、不继续预检、不用预检权重正式初始化、不重复C0 hash/取回或本机OOM尝试。若正式阶段要求在线监控，应先独立明确SwanLab既有配置问题与验收条件。
