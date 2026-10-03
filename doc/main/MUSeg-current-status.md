# MUSeg 当前状态与唯一实时入口

> **事实与执行边界截至：2026-10-04 02:30:06+08:00（Direction A数值修复资格通过，重新正式被版本同步阻塞）。** Natural从冻结C0资格运行1664次尝试/1664次成功更新、optimizer skip0，越过原第1598次故障区间，无资格checkpoint。正式final验收适配已提交推送，但云端SSH/GitHub TLS同步失败，同HEAD门禁未通过；**新三组正式及S1均NOT_RUN，仍INVALID/BLOCKED，不能科研GO/STOP**。准备仅CPU-only，最新直接确认唯一实例 **Stopped/GPU0**。**大白话：** 修复已经通过限定检查；完整对照还没启动，因为云端版本没能同步，不能拿资格结果当方法收益。

## 已核验根因、修复与资格

- 诊断源码 `1cf9a8af517a1cf060846461cd31be5aba667950` 从原C0复现attempt1598/successful1597/skip0；失败loss0.18105733394622803 finite、AMP scale1024。输入及已观察前向均finite，首个坏中间梯度 `decode_head.hamburger.ham.iteration.2.bases.divide` 有39 Inf；异常trace定位NMF（非负矩阵分解）基底更新分母的 `BmmBackward0` 返回NaN。backward未完成、unscale未开始、坏attempt未执行optimizer。结论为AMP下NMF内部反向数值不稳定/溢出传播，不是已发现的输入NaN/Inf污染或scaler增长/解缩放故障；不扩大为所有数据质量已核验。
- 修复源码 `7d6246a7db77f77e26e20d84074dff6e191e9fa5` 已commit/push，云端Git fast-forward、同HEAD/tracked-clean后运行。三组共同 `nmf_training_precision=fp32_local`：原NMF2D在CUDA训练autocast中仅内部关闭autocast并输入FP32。外围AMP fp16/scaler1024、TF32、模型结构/参数、NMF算法/迭代/随机基底、数据/LR/scheduler/正式预算/S1不变；不启用旧A-v1分支。推理仍为冻结FP32/TF32off。
- Natural限定资格从原C0独立初始化，receipt **CAP_REACHED_FINITE**，attempted1664/successful1664/skip0/complete=true、workers8、scheduler2560。实际策略日志确认NMF local FP32 enabled；最后loss0.15081989765167236、scale1024，peak allocated18988.9429/reserved20556 MiB，child elapsed1439.2121秒。runner逐步loss/gradient和optimizer检查通过，checkpoint_dir=null/checkpoint_written=false；资格权重不具正式初始化或resume资格。
- 精确C0 SHA256 `ca618b23d18eabb201a0d11d18da383ac99576d0feae5864e3233bda527d9a1a`，321150608 bytes；原云路径 `/root/rivermind-data/cloud/MMFR_E1_Batch1A_local_transfer_20260922/C0/checkpoint/update-2560.pth`。不重复hash/download，不继承旧optimizer/scaler/RNG。原环境Python3.10.16/torch2.1.2+cu118/CUDA11.8/RTX4090未重建升级。

## 当前动作、授权与停止边界

- 正式验收适配 `cfca1c579f5e29388528c3b9b1fce70fac831be1` 已提交推送，交接提交 `00ca7375fbaa4a42dc781bdd368e7d1b2738c9d3` origin ref已直接核验。精度合同改为 `fp32_local`；每组完成后CPU map_location读回final schema/provenance/计数/optimizer/RNG与SHA，失败立即停止下一组/S1。已核验hash交给同进程S1，取回后仍本地CPU读回并对齐S1。Python语法、直接差异及修正末尾空白后的whitespace检查通过；云端新Bash/Python检查及真实final验收未运行。
- **当前阻塞：** CPU-only同步发生SSH连接关闭，一次已进入Git fetch后exit128/gnutls TLS握手失败，未merge；最后一次限时重试在SSH readiness失败，未执行fetch。最后直接观察云端HEAD为资格提交 `7d6246a`，尚未同步新验收版本。没有提交新formal job、没有启动GPU或绕过同HEAD门禁。本轮报告/导航收口与训练源码身份分开记录，不因此重启云端。
- 恢复SSH/Git连接、云端fast-forward并直接核验最新已push HEAD/运行源码tracked-clean、原环境及平台保险后，才从原C0分别干净启动 **Natural → Grid → Replay**，各2560 successful updates、attempted=successful、optimizer skip0、每640 recovery与2560 fixed-final。不得续跑旧Natural1597或资格1664，不得组间继承权重。
- 任一正式组再次资格性故障，立即停止后续流程并报告INVALID/BLOCKED，不跳坏update、不改protocol救结果、不自动重试。三组全部合法完成才唯一完整S1：C0/Natural/Grid/Replay ×318样本×3条件=3816 views。
- 沿[Direction A §6.5](../../MMFR/01_research/MMFR_direction_A_natural_missing_2026-10-02.md)冻结gate：Replay自然high≥最强C0/Natural/Grid+1.0pp；natural all/low各≥最强−0.3pp；rectangle≥最强Natural/Grid+0.5pp；entire≥Grid−0.5pp，并且删除量/类别支持/采集组解释合格。门槛不改，工程失效不能科研裁决。
- Main-Val、official test、NYUv2、其他baseline、Round-2、Direction B/B1a、新seed/调参仍关闭，official test保持sealed_unread；A-v1 stop、F-lite/R-OE独立处置不变。完整授权与精度修订见[首轮protocol §13–15](../../MMFR/01_research/natural_missing_round1_protocol.md)。

## SwanLab、资源与证据

- SwanLab0.9.7已有限定位：online客户端未登录、settings.api_key=None、interactive=False，在SDK `prompt_init_mode` line323拒绝初始化；相关key/token/mode环境为空、默认netrc不存在。保持 **LOG_ONLY**，没有online链接/上传完成证据，不泄露凭据、不升级SDK或长期阻塞科研。诊断和资格均一次bounded初始化，finish_returned=true。
- 唯一实例 `cpod-1vbh7faqcauq`，资格GPU API起点1791049456、保险1791052144；job `natural-missing-qualification-20261004-7d6246a` Succeeded/exit0。结束主动停GPU，15分钟保险CPU-only Running/GPU0取回exit0，02:09:46直接Stopped/GPU0。随后正式准备仅四个短时保险CPU-only窗口，均失败后主动关机；最新02:30:06直接Stopped/GPU0、StopTime1791052134。没有新正式训练、S1或监督器运行。
- 资格证据：`outputs/natural-missing-qualification-20261004-7d6246a/monitor/numerical-run-receipt.json`、Natural stdout/stderr与screen.log、`numerical-lifecycle.json`；云端同名根位于 `/root/rivermind-data/cloud/`。诊断证据：`outputs/natural-missing-numerical-20261004-1cf9a8a/`，其早期run-identity是准备快照，真实终态以receipt/lifecycle为准。重新正式准备记录：`outputs/natural-missing-round1-renewed-20261004/run-identity.json`。完整定位/资格/同步阻塞见[2026-10-04阶段报告](../reports/2026-10-04-natural-missing-numerical-repair-qualification.md)；旧失败事实见[2026-10-03正式报告](../reports/2026-10-03-natural-missing-round1-formal.md)，原始证据不改写。
- 实际检查限于直接代码diff/语法、既有parser门禁、Git/原环境/保险、真实诊断/资格日志与资源/取回身份；本机额外torch检查因既有OpenMP runtime冲突未完成，未不安全绕过。未运行完整测试、额外训练/数据审计、三组新正式/S1或official test。

**准确恢复点：** 实例Stopped/GPU0；诊断和1664资格已完成，不重复。SSH/GitHub同步是唯一当前正式启动阻塞；恢复后先云端Git fast-forward/同最新已push HEAD/tracked-clean/新入口语法及原环境核验，再设置本轮GPU保险从冻结C0执行新三组与S1。禁止接旧1597或资格1664；新三组/S1/final读回/gate均未执行，资源已主动关闭。
