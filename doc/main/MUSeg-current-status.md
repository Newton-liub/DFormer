# MUSeg 当前状态与唯一实时入口

> **事实与执行边界截至：2026-10-03（0更新启动失败已停机，修复后继续原预算已获用户明确批准）。** 首次运行Git `8d2490ef433807ec60d475cab27fbadd04d18e1f`；Natural **attempted0 / successful0**，Grid/Replay与完整S1均未运行。失败是新增编排器漏传必需的`--successful-updates 2560`，不是loss、显存、模型或数据失败。GPU停止后仅CPU-only取回失败证据并立即停机，最终直接确认**Stopped/GPU0**。用户随后明确同意补齐启动命令后继续尚未执行的原三组预算与S1，保持原2026-10-04 02:11:47+08:00关机截止、不延长；尚未重新上云。

## 当前结果与实际意义

- 用户已批准Natural→Grid→Replay各2560成功更新、四权重×val-dev318×三条件完整S1（3816 views）、现有4090、screen/SwanLab及必要Git提交push。此次启动在训练入口`argparse`门禁退出：Natural child exit2，receipt FAILED，0输入槽/0 optimizer更新；未加载C0进入正式训练，没有formal/recovery checkpoint或S1分数。
- **科研裁决：BLOCKED/INVALID（工程启动失败），不能给科研GO/STOP。** Natural保留自然输入，Grid增加网格删除，Replay重放其他训练采集组空洞；S1是冻结的原尺度、无翻转单视图开发评价。**大白话：** 正式训练还没开始，因此这次失败不能用来判断Replay有没有效果。
- 同一个精确原C0身份仍为`ca618b23d18eabb201a0d11d18da383ac99576d0feae5864e3233bda527d9a1a`，来源`/root/rivermind-data/cloud/MMFR_E1_Batch1A_local_transfer_20260922/C0/checkpoint/update-2560.pth`、321150608 bytes。不重复下载/hash，不使用preflight或组间权重；旧4090预检READY_ENGINEERING_ONLY证据保留，不冒称正式加载/训练通过。
- 合同不变：全部分割参数、仅分割loss；关闭旧aux/A2混合故障/F-lite/R-OE/A-v1；每组2560、batch10/480×640/workers8、LR1e-6/AdamW/WD.01、warmup128/poly.9、AMP fp16/scaler1024/TF32on/SyncBN、原生NMF autocast；每640 recovery/fixed-final，无训练val或best选择。完整S1仅在三组全部合法完成后进入，按canonical A §6.5原门槛裁决。

## 资源、运行身份与监控

- 唯一实例`cpod-1vbh7faqcauq`，没有创建/resize。启动前6小时平台保险直接确认截止**2026-10-04 02:11:47+08:00 / Unix1791051107**；Running直接记录仍为同截止。GPU启动请求20:11:52+08:00，API Running StartTime1791029637（20:13:57）；RTX4090、CUDA报告可用显存25280839680 bytes、CPU14/32768MiB，API运行价1.88。
- 云端canonical origin轻量fast-forward到同SHA，启动前源码tracked dirty为空；原环境`/usr/local/miniconda3/envs/py310/bin/python`、Python3.10.16/torch2.1.2+cu118/CUDA11.8，未安装/升级依赖。screen `natural-missing-round1-formal`由持久任务`natural-missing-formal-20261003-8d2490e`启动；receipt始末Unix1791030035.742–1791030038.342（约2.60秒），终态FAILED。
- 停机监督主动stop，API StopTime1791030064（20:21:04），20:21:11直接查询Stopped。该次API仍保留**GPU1配置规格**，不冒称当时查询GPU0；这也暴露原监督器把停机配置数当作活动GPU数的问题，已修正第一阶段Stopped判据，最终CPU-only停机仍必须Stopped/GPU0。
- 失败证据取回另设30分钟保险20:54:10+08:00 / Unix1791032050，20:24:16请求without-gpu A，API StartTime1791030262（20:24:22），直接Running/CPU2/4096MiB/GPU0。monitor目录transfer exit0，20:25:19完成后立即stop，API StopTime1791030320（20:25:20）；最终直接查询**Stopped/GPU0**。无训练/评价、无大权重取回；最终账单未核验。
- SwanLab（在线实验记录服务）0.9.7已安装，API key环境变量不存在、`Settings(interactive=False)`可用；两次限时online初始化及正式初始化均RuntimeError，安全定位至SDK `prompt_init_mode`。本轮**LOG_ONLY**，没有在线run链接/上传完成证据；未输出token或原始敏感异常，未重建环境，监控失败不是训练失败原因。

## 修复、证据与准确恢复点

- 主执行者直接核验失败receipt、Natural stderr/stdout及summary，确认训练parser要求显式`--successful-updates 2560`而调用遗漏，**0尝试/0成功更新**。本地已补参数；同时保留CLI终态失败JSON、修正停机配置数解释。模型、输入、optimizer、训练预算与评价入口未改；修复尚未提交/推送或重新上云。
- 本地证据根`outputs/natural-missing-round1-formal-20261003/`：`run-identity.json`、原始监督`lifecycle.json`、`startup-failure-closeout.json`、`monitor/formal-run-receipt.json`、`Natural-training-summary.json`、stdout/stderr及`screen.log`。远端对应`/root/rivermind-data/cloud/natural-missing-round1-formal-20261003-8d2490e/`。失败原始证据不覆盖。
- 科研/授权依据：[canonical Direction A §6.5](../../MMFR/01_research/MMFR_direction_A_natural_missing_2026-10-02.md)、[首轮protocol §10](../../MMFR/01_research/natural_missing_round1_protocol.md#10-2026-10-03-direction-a正式首轮最新授权)；[readiness报告 §13](../reports/2026-10-03-natural-missing-round1-readiness.md)保留旧预检。正式最终报告尚未生成，不能写成三组训练或S1已完成。
- **准确恢复点：** 用户已明确批准修复命令后继续原首轮预算；先选择性提交push新SHA，云端轻量同步，用新输出根保留首次0更新失败证据。三组从精确原C0独立开始，沿原02:11:47保险截止、不延长，依次正式训练与完整S1；当前实例保持Stopped/GPU0直到必要Git同步准备完成。科研合同、门槛及额外实验关闭边界不变。
- 原有目录审计/执行单、PACKAGE_INDEX/report-index dirty未覆盖或混入已推送提交；checkpoint/outputs/大日志不入Git。Direction B/B1a、Main-Val、NYUv2/baseline、新seed/调参仍关闭，official test仍**sealed_unread**；A-v1保持stop，F-lite/R-OE独立未决。完整测试、追加GPU验证和真实resume均未运行。
