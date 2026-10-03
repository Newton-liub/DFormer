# Natural Missing Round-1 正式执行报告

> **覆盖范围：** 2026-10-03正式授权执行、首次零更新启动失败、获批修复后原预算运行、非有限梯度停止及CPU-only证据取回。结果截至2026-10-03 23:11:39+08:00；后续Git收口单独以提交记录为准。
> **科研合同：** [首轮protocol §1–7、§10–11](../../MMFR/01_research/natural_missing_round1_protocol.md)；冻结裁决见[canonical Direction A §6.5](../../MMFR/01_research/MMFR_direction_A_natural_missing_2026-10-02.md)。[实时状态](../main/MUSeg-current-status.md)仍是唯一当前事实与权限入口。

## 1. Executive Summary

本轮已真正进入正式训练，但Natural在第1598次尝试发生非有限梯度，只完成1597次成功更新，未达到2560。runner按冻结合同立即退出，未跳过坏更新继续训练；Grid、Replay和完整S1均未运行。**Direction A本次执行结论为INVALID/BLOCKED，不是科研GO或STOP。** 失败证据已取回，实例最终直接确认**Stopped/GPU0**。

**大白话：** 基础对照组先出现数值问题，因此还无法比较Replay是否有效。当前停止的是这次无效运行；没有证据证明该研究方向有效或无效，也没有授权自动改变精度后重训。

关键术语：successful update是确实执行的optimizer更新；attempt是已经取到输入并开始计算的一次尝试；非有限梯度指梯度包含NaN/Inf等不可用数值；checkpoint是保存的模型和训练状态快照；fixed-final指预先指定成功更新2560时的权重，不能用不完整或中间权重替换。S1是冻结原尺度、无翻转、FP32单视图开发评价，不是Main-Val或official test。

## 2. Run Identity

- 正式运行源码：`edb660a83da1ec67626149d06dc59460565e2c20`，分支`perf/mmfr-a2-v3-pipeline-opt1`；已push到canonical origin。云端`/root/rivermind-data/DFormer`轻量fast-forward后直接确认同HEAD，前后tracked dirty为空；未reset用户改动。
- 唯一实例：`cpod-1vbh7faqcauq`；用户在平台资源不足启动失败后手动启动。RTX4090/GPU1/CPU14/32768MiB；CUDA直接报告显存25280839680 bytes。Python3.10.16、torch2.1.2+cu118、CUDA11.8；原解释器`/usr/local/miniconda3/envs/py310/bin/python`，未安装/升级依赖。
- 原C0：`/root/rivermind-data/cloud/MMFR_E1_Batch1A_local_transfer_20260922/C0/checkpoint/update-2560.pth`，321150608 bytes；沿用已核验SHA-256 `ca618b23d18eabb201a0d11d18da383ac99576d0feae5864e3233bda527d9a1a`，没有重复hash/下载。使用严格分割加载；未继承旧optimizer/scaler/RNG或预检权重。
- 持久任务：`natural-missing-formal-20261003-edb660a`；screen `natural-missing-round1-formal`直接确认存在，stdout/stderr、screen和receipt持久落盘。任务API StartedTime1791038608、FinishedTime1791039952、State Failed/ExitCode1。
- GPU保险沿原绝对截止**2026-10-04 02:11:47+08:00 / Unix1791051107**，用户手动启动后Running查询直接确认有效；没有重新增加六小时或延长。SwanLab为LOG_ONLY。

正式参数未变：train-dev1277，全部分割参数/segmentation-only；三组计划各2560、batch10、480×640、workers8、LR1e-6、AdamW/WD.01、warmup128/poly.9、AMP fp16/scaler初始1024、TF32on、SyncBN、原生NMF autocast；每640 recovery，无训练val/best选择，旧aux/A2混合故障/F-lite/R-OE/A-v1关闭。

## 3. Training与停止位置

| Strategy | attempted | successful | 已记录optimizer skip | 合格fixed-final | 编排wall time | peak allocated / reserved VRAM |
| --- | ---: | ---: | ---: | --- | --- | --- |
| Natural | 1598 | 1597 | 0 | 无；未达2560 | 1343.4608秒 | 18956.5898 / 20556.0 MiB |
| Grid | 0（NOT_RUN） | 0 | 不适用 | 无合格产物 | 未运行 | 未运行 |
| Replay | 0（NOT_RUN） | 0 | 不适用 | 无合格产物 | 未运行 | 未运行 |

- 最后一次成功更新为1597，epoch13、batch cursor61，loss `0.1476122885942459`、LR `4.151961920329593e-07`、AMP scale1024，`optimizer_step_applied=true`、`loss_finite=true`。
- 最后一条attempt_started为1598，batch形状`[10,3,480,640]`，此前successful1597。stderr在`natural_missing_train.py::_run_training`的梯度有限性检查抛出：`FloatingPointError: NaturalMissing gradients are non-finite; refusing to skip the optimizer update`。
- 代码路径在loss有限性检查、scaled backward与unscale之后，在`scaler.step`之前停止。**该次失败尝试没有执行optimizer更新**；已记录skip0不意味着第1598次成功，也不能把1598=1597+0作为合法完成计数。失败尝试单列，不吞掉、不更换样本续跑。
- 具体非有限tensor、产生算子与失败步loss数值没有记录，本次未做额外forward/backward定位，**不能据此直接归因NMF、AMP、数据或某个模型层**。不是此次参数门禁失败，也不是已记录的OOM。
- 流程经过640/1280保存点后继续到了1597，但未取回/逐文件读回recovery；其当前文件存在性、schema与hash未直接核验，不作为合格fixed-final或resume资格。没有完成summary的`complete=true`，没有2560固定final验收。

## 4. Input Telemetry

Natural已观察1598批、15980输入槽，包含最后失败尝试：clean15980、matched0、paired_skip0；实际新增删除像素总量/均值/最大值均为0，validated slots15980、contract violations0。Natural没有选源，不把source/target检查0记为跨组排除已验证。

Grid/Replay尚未运行，因此matching error、Replay rate、删除率匹配和三组完整配对没有正式证据；未生成合法三组input summary。这里的0违例只覆盖已执行Natural输入，不等于三策略正式输入合同全部通过。

## 5. Full S1

**完整S1未运行，已执行正式S1 views为0/3816。** 三组全部合法完成是进入条件，本次在Natural失败后保持该门禁。C0/Natural/Grid/Replay在natural、rectangle、entire-missing条件的all/low/high指标均无本轮正式结果，不填写0分或沿用旧极小预检分数。

未运行旧Quick-Val、Main-Val、S2/S10、NYUv2、baseline或official test；official test保持sealed_unread。没有通过中间checkpoint补评价。

## 6. Frozen Gate Decision

- Replay自然high相对C0/Natural/Grid最强者≥+1.0pp：未评价。
- 自然all和low各相对最强者≥−0.3pp：未评价。
- rectangle相对Natural/Grid强者≥+0.5pp：未评价。
- entire-missing相对Grid≥−0.5pp：未评价。
- 正式三组删除量、类别支持与采集组解释：未完成，不能以Natural局部遥测代替。

**最终为INVALID/BLOCKED：非有限梯度使正式运行不合格，完整S1不存在。** 冻结阈值不改；既不能宣布GO，也不能把工程/数值失败写成未过收益门槛的科研STOP。

## 7. Resources与实际时间

以下时间为+08:00，运行窗口按直接API StartTime/StopTime计算，不替代最终账单。

- 修复后用户手动GPU启动：22:41:12 / Unix1791038472；正式receipt开始22:43:28.536，结束23:05:52.654，编排总wall **1344.1179秒（22分24.12秒）**。Natural child wall1343.4608秒，不能把单步elapsed字段误作整组时间。
- 监督器23:06:22获得Failed终态，主动stop；API StopTime23:06:24 / Unix1791039984，23:06:32直接Stopped。GPU运行窗口**1512秒（25分12秒）**；当时GPU1是保留的配置规格，不是停机后仍在运行。
- 失败monitor取回设置CPU-only30分钟保险，截止23:37:54 / Unix1791041874；三次CPU操作沿同一截止，未延长：23:08:04–23:08:08因Initializing异步状态立即停止；23:09:23–23:09:31因PowerShell原生管道UTF-8 JSON解码错误立即停止；23:10:37–23:10:59明确等待Running/GPU0、取回monitor exit0后立即stop。三个API运行窗口合计**34秒**。
- 最后stop响应曾短暂为Stopping；显式wait后，**23:11:39直接确认Stopped/GPU0**。未等待保险触发，没有因下载保留GPU，没有取回不完整权重或数据集。
- 首次零更新启动失败的GPU窗口20:13:57–20:21:04（427秒）、CPU-only取回20:24:22–20:25:20（58秒）为独立历史窗口。含两次正式启动的GPU窗口合计1939秒（32分19秒）；本报告不是持续GPU三小时计费。平台API运行价GPU1.88、CPU0.13，**最终账单未核验，不报估算为实付**。

## 8. SwanLab、启动修复与资源监督

SwanLab0.9.7已安装；首次运行前两次有限online初始化及两次正式初始化均RuntimeError，既有排障定位至SDK `prompt_init_mode`，API key环境缺失、interactive=false设置可接受。本次receipt为LOG_ONLY/init_error detail redacted、finish_returned=true；没有online run/link或上传完成证据。未输出token，未重建环境。监控故障不是第1598次非有限梯度的已证明原因。

首次源码`8d2490ef433807ec60d475cab27fbadd04d18e1f`遗漏必需`--successful-updates 2560`，Natural参数门禁exit2，attempted0/successful0；其余组与S1未运行。用户明确选择continue_original_budget后，修复调用参数、终态failed-job JSON保留与Stopped配置数判据，提交push为edb660a；22:37启动曾因4090容量不足ret_code226604失败，用户手动启动后才执行本次实际训练。首次失败不算科研结果，原始证据不覆盖。

收口又修正**仅资源监督**的两项操作问题：start/stop显式`--wait`以等待稳定状态；PowerShell native JSON管道设置UTF-8，避免平台中文字段破坏解析。模型、训练入口、数据、AMP/NMF、预算与评价实现未改。这些收口修复未用于重跑本次实验，最终提交与formal-run SHA应明确区分。

## 9. Local Artifacts与实际检查

本次本地根：`outputs/natural-missing-round1-formal-restart-20261003/`。

- `run-identity.json`：修复SHA、环境、原保险、持久任务身份。
- `restart-capacity-receipt.json`：平台资源不足请求及直接Stopped确认。
- `lifecycle.json`：原监督器Failed终态和主动GPU stop；保留原始记录，不补造CPU取回。
- `nonfinite-closeout.json`：主执行者直接核验后的三次CPU操作、monitor transfer与最终Stopped/GPU0。
- `monitor/formal-run-receipt.json`、`Natural-training-summary.json`、`Natural.stdout.log`、`Natural.stderr.log`、`screen.log`：计数、遥测、错误与终态原始证据。
- 远端持久根：`/root/rivermind-data/cloud/natural-missing-round1-formal-20261003-edb660a/`。没有三份合格`update-2560.pth`或完整S1本地产物，未运行成功final专用schema/hash读回工具。
- 首次零更新失败独立保留：`outputs/natural-missing-round1-formal-20261003/`，含`startup-failure-closeout.json`，不覆盖历史receipt/log。

实际核验：云端同SHA及tracked-clean、Running GPU/环境与原保险；直接复核Natural最后attempt/success日志、stderr梯度停止位置、receipt/summary的输入计数；CPU-only取回exit0和最终直接Stopped/GPU0；修复的CPU CLI参数门禁、Python/PowerShell静态检查及交付文档差异检查。未运行完整测试、追加GPU资格、真实resume、非有限定位训练、完整S1或final checkpoint读回：前者不覆盖本次风险，后者受非有限停止合同/缺少合格产物限制。没有临时测试文件、大mask cache或额外实验。

## 10. Next Step

当前执行授权已经因非有限梯度停止。**下一步需独立裁决是否开展限定数值定位与重新资格检查，或终止这次执行准备；任何GPU诊断、精度/NMF变更、resume/重训都需明确新授权。** 现有证据不足以自动决定技术原因或将剩余预算续跑。Direction B/B1a、Round-2、NYUv2/baseline与official test仍关闭；本次只进行报告/状态/必要索引与Git收口。
