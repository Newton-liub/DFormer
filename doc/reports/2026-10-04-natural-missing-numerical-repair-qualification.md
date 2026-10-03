# Direction A 数值定位、最小修复与资格通过报告

- 汇报周期：2026-10-03旧正式非有限停止之后，至2026-10-04 02:30:06+08:00的资源直接核验。
- 对象：Direction A执行交接与阶段复核。
- 证据边界：诊断源码 `1cf9a8af517a1cf060846461cd31be5aba667950`；修复/资格源码 `7d6246a7db77f77e26e20d84074dff6e191e9fa5`；正式验收适配 `cfca1c579f5e29388528c3b9b1fce70fac831be1`；已推送交接提交 `00ca7375fbaa4a42dc781bdd368e7d1b2738c9d3`。
- 当前结论：**数值修复资格通过；重新正式启动被SSH/Git同步失败阻塞，仍INVALID/BLOCKED，尚不能科研GO/STOP。**
- 实时权限与恢复点：[当前状态](../main/MUSeg-current-status.md)；[开放决策](../main/MUSeg-open-decisions.md)。授权与精度修订：[首轮protocol §13–15](../../MMFR/01_research/natural_missing_round1_protocol.md)。旧失败历史：[2026-10-03正式报告](2026-10-03-natural-missing-round1-formal.md)。

## 1. 概述与判断边界

已从原冻结C0重现坏梯度，并将有直接故障证据的矩阵分解计算限定为FP32。修复后的Natural完成1664次成功更新、没有跳过坏步骤，越过了原第1598次故障区间。但重新正式需要云端Git同步到包含新权重验收门禁的已推送版本；该同步因SSH连接关闭与GitHub TLS握手失败未完成。因此没有启动新的三组正式训练，也没有S1评价或可用于判定Replay收益的分数。实例已经关闭，没有为排障保留GPU。

- NMF（非负矩阵分解）：HAM分割解码器内部计算；本次只改其训练数值精度，不改算法或模型结构。
- AMP（自动混合精度）：外围仍使用FP16与GradScaler；NMF内部单独FP32不是全模型FP32重训。
- C0：既有2560-update固定底座权重；每个新运行从其模型参数独立初始化，不继承optimizer/scaler/RNG。
- 资格：限定运行检查数值稳定性，不保存正式checkpoint（模型参数快照），不能代替完整三组实验。
- S1：冻结的原尺度、无flip、单view、FP32/TF32off开发评价；完整规模为四权重×318样本×三条件=3816 views。
- INVALID/BLOCKED：当前证据不足或执行门禁未通过；不等于研究方法已达GO或已被科研STOP。

## 2. 已完成并验证：数值定位

### 2.1 执行身份

诊断仅Natural，从冻结C0独立开始，cap1664、重型观测从attempt1536，scheduler仍为2560。源码已commit/push，云端同HEAD/tracked-clean后运行；没有保存checkpoint，不续跑旧Natural1597。

环境为Python3.10.16、torch2.1.2+cu118、CUDA11.8、RTX4090，显存25280839680 bytes。没有升级SDK/PyTorch或重建环境。数据仍为冻结train-dev1277、batch10、480×640、workers8、LR1e-6及原普通增强/采样顺序；不读取official test。

冻结C0路径为 `/root/rivermind-data/cloud/MMFR_E1_Batch1A_local_transfer_20260922/C0/checkpoint/update-2560.pth`，321150608 bytes，先前已核验SHA256 `ca618b23d18eabb201a0d11d18da383ac99576d0feae5864e3233bda527d9a1a`。本轮不重复hash或下载C0。

### 2.2 直接故障证据

- 再现attempt1598/successful1597/optimizer skip0，epoch13、batch position61；最后成功loss0.1476122885942459，与旧故障一致。
- 失败步loss0.18105733394622803 finite，AMP scale1024；RGB/Depth/Label与全部已观察前向激活均finite。
- 首个观察到的非有限中间梯度为 `decode_head.hamburger.ham.iteration.2.bases.divide`，shape `[10,512,64]`，39 Inf/0 NaN，其余有限值max55446.61328125。
- 该节点梯度dtype虽为FP32，周围autocast bmm/分母仍为FP16，不能据此宣称原NMF内部已经完整FP32。
- anomaly forward traceback定位 `NMF2D.local_step` 的 `denominator = bases.bmm(gram)`，`BmmBackward0` 的第1个输出返回NaN。loop2分母前向min0，加epsilon后min1.0132789611816406e-06；相关divide梯度39 Inf，multiply梯度还有有限max54719987712。
- backward未完成、unscale未开始、optimizer未执行。异常提前截断后首个坏named parameter为null，partial参数finite摘要不构成完整反向通过证据。

证据支持**AMP下NMF内部反向数值不稳定/溢出传播**。没有发现该失败batch的输入NaN/Inf污染，也没有证据指向scaler增长或解缩放故障；本报告不扩大为全数据质量已验证。

直接依据：`outputs/natural-missing-numerical-20261004-1cf9a8a/monitor/numerical-run-receipt.json` 的 `failed_attempt.diagnostics`、`Natural.stderr.log` 与 `Natural.stdout.log`。顶层 `nonfinite_events=[]` 不抵消 `failed_attempt` 中的坏梯度证据：异常反向提前终止，必须结合失败对象和trace判断。该根目录早期 `run-identity.json` 是准备快照，真实终态以receipt和 `numerical-lifecycle.json` 为准，不改写历史快照。

## 3. 已完成并验证：三组统一最小修复及Natural限定资格

### 3.1 改动范围与原理

修复提交 `7d6246a`：

- `local_configs/MUSeg/DFormerv2_S_NaturalMissing.py`：三组统一 `nmf_training_precision=fp32_local`。
- `utils/natural_missing_runtime.py::build_model`：校验精度字段与原NMF2D身份，只给该模块设置训练flag。
- `models/decoders/ham_head.py::NMF2D.forward`：新flag仅训练态、CUDA且autocast启用时生效；仅NMF内部关闭autocast并转换输入FP32。保留旧A-v1条件，但NaturalMissing不启用A-v1研究分支。
- `tools/mmfr/natural_missing_train.py`：记录实际 `numerical_policy`。

外围AMP fp16/初始scale1024、TF32、模型参数结构、NMF算法/六次训练迭代/随机基底、数据/LR/2560 scheduler及各组正式预算均不变。eval态新flag不改变冻结S1 FP32/TF32off定义。

### 3.2 资格运行结果

源码修复已推送，云端Git fast-forward、同HEAD/tracked-clean后，从原C0独立开始Natural diagnosis-only资格。job `natural-missing-qualification-20261004-7d6246a` 为Succeeded/exit0；receipt **CAP_REACHED_FINITE**。

- attempted1664 / successful1664 / skipped0 / complete=true。
- workers8，scheduler2560，实际日志 `nmf_local_fp32_enabled=true`、`nmf_training_precision=fp32_local`、外围float16/initial1024。
- 全程runner loss/gradient与optimizer检查通过；越过原第1598次故障区间。
- 最后loss0.15081989765167236，scale1024；peak allocated18988.94287109375 MiB、reserved20556 MiB；child elapsed1439.212100441102秒。
- checkpoint_dir=null、checkpoint_written=false；资格不提供正式初始化或resume权重。

依据：`outputs/natural-missing-qualification-20261004-7d6246a/monitor/numerical-run-receipt.json`、Natural stdout/stderr、screen.log及根目录 `numerical-lifecycle.json`。这只证明该限定资格运行通过，不保证Grid/Replay及完整2560预算必然稳定。

## 4. 已完成但尚未云端验证：正式权重验收适配

发现既有 `natural_missing_formal_artifacts.py` 仍要求旧 `amp_autocast`；若不适配，会错误拒绝合法新FP32 final。原正式编排仅核对完成计数和checkpoint存在/大小，S1前没有schema/hash CPU读回。

提交 `cfca1c5` 完成最小适配，并经Python语法及直接差异复核：

- `read_final` 的合同精度改为 `fp32_local`，允许 `s1_hashes=None` 用于S1前直接读回；有S1 hash集合时仍强制比对。
- 每组训练成功后、下一组之前，CPU map_location加载本项目可信fixed-final，检查schema/provenance/源码SHA/formal标记、C0/数据/预算/精度、cursor与2560/2560/skip0、模型802keys、optimizer685states/step2560、RNG及无best选择，直接记录final SHA256。失败即停止下一组和S1。
- `natural_missing_formal.py` 保存每组final-readback，全部通过后才进入S1；复用已核验final hash并核对S1报告身份，避免同一不变权重重复hash。
- `natural_missing_formal_screen.sh` 增加Git工作区及index tracked-clean拒绝门禁。

已直接核对evaluator `verified_checkpoint_sha256` 支持本次调用内的非C0路径与合法SHA256；不改任何模型forward、输入或S1条件。真正的final CPU读回、云端新Bash/Python检查及S1链尚未执行，不能写为通过。

## 5. SwanLab有限排障结论

SwanLab0.9.7，云端相关key/token/mode环境为空、默认netrc不存在。有限初始化失败stack为SDK `swanlab/sdk/cmd/init.py::prompt_init_mode` line323；此前CPU-only直接读取SDK分支，确认online客户端未登录、settings.api_key=None、interactive=False，因而拒绝无凭据的非交互初始化。

诊断和资格各一次bounded初始化、finish_returned=true，记录 **LOG_ONLY**。没有在线链接或上传成功证据，没有暴露凭据，没有因监控升级依赖、改环境或长时间阻塞科研。资格成功不意味着SwanLab在线成功。

## 6. 重新正式启动阻塞、三组与gate状态

适配及交接提交 `00ca7375fbaa4a42dc781bdd368e7d1b2738c9d3` 已推送origin当前分支，`git ls-remote` 直接确认同ref。第一次push后的额外ref查询遇到本地TLS失败，后续查询成功；未把失败查询记通过。

云端准备只使用短时保险CPU-only/GPU0。期间出现SSH连接关闭；一次安全编码同步连接确实进入Git fetch，但返回exit128，错误为 `gnutls_handshake() failed: The TLS connection was non-properly terminated.`，未进入merge或后续源码/环境检查。最后一次限时同步重试在SSH readiness已失败，计划中的90秒timeout/HTTP1.1 Git fetch未被执行。没有复制异版源码或绕过Git同HEAD门禁。

最后直接观察的云端HEAD仍为资格提交 `7d6246a7db77f77e26e20d84074dff6e191e9fa5`；**云端最新同HEAD/tracked-clean门禁未通过**。所有准备窗口均主动停机，最新2026-10-04 02:30:06+08:00直接查询为Stopped/GPU0。

- 新Natural正式2560：**NOT_RUN**；1664资格不计正式训练。
- 新Grid正式2560：**NOT_RUN**。
- 新Replay正式2560：**NOT_RUN**。
- 完整S1 3816 views：**NOT_RUN**；没有新fixed-final或评价分数。
- Direction A冻结gate：**INVALID/BLOCKED，尚无科研GO/STOP裁决**。根因为正式启动版本同步门禁未满足，而不是新正式再次发生数值故障，也不是收益不足。

冻结门槛仍为Replay自然high≥最强C0/Natural/Grid+1.0pp；natural all/low各≥最强−0.3pp；rectangle≥最强Natural/Grid+0.5pp；entire≥Grid−0.5pp，并要求删除量/类别支持/采集组解释合格。未修改阈值或研究变量。Main-Val、official test、NYUv2、baseline、Round-2、Direction B/B1a仍关闭；official test保持sealed_unread。

准备记录为 `outputs/natural-missing-round1-renewed-20261004/run-identity.json`，记录未提交formal job、未启动GPU、四个CPU-only准备窗口、同步错误分类和最终直接资源状态；不作为训练receipt。

## 7. 云资源与CPU-only取回

唯一实例 `cpod-1vbh7faqcauq`，未新建或resize。

- 诊断GPU API窗口1791047409→1791048748，共22分19秒；平台保险1791050100，失败后主动停GPU。必要监控证据CPU-only取回exit0，01:33:29+08:00最终Stopped/GPU0。前两个零训练准备窗口55秒/18秒及CPU准备21秒见既有lifecycle/运行证据，不能当诊断训练成功。
- 资格GPU起点1791049456，保险1791052144；完成后主动停GPU，15分钟保险CPU-only Running/GPU0取回exit0，02:09:46+08:00直接Stopped/GPU0。
- 本次正式准备只有CPU-only窗口：1791051692→1791051711、1791051784→1791051805、1791051860→1791051988、1791052120→1791052134；启动前设置短时平台保险，失败后主动关机。没有新GPU训练窗口，未下载新final（尚不存在）。
- 最新直接状态02:30:06.828876+08:00：Stopped/GPU0，StopTime1791052134，SchedulerStopTime1791052712。没有宣称最终账单已核验。

## 8. 实际检查与未运行项目

完成的相关检查：诊断日志与异常trace/资格receipt和实际精度日志直接复核；修复配置/builder/NMF差异复核；修复代码及本轮正式编排/CPU验收Python语法检查；此前独立实际parser门禁、PowerShell parser和诊断版云端Bash语法检查；本轮交付物diff/whitespace及JSON内容复核；本地push/ref、cloud最后直接HEAD、保险、CPU-only取回与最终实例状态核验。

本次曾发现protocol末尾空白行，另一次文档提交修正后diff --check通过；未将初次失败检查写为通过。本机完整导入/额外torch小检查因既有重复OpenMP runtime未完成，未设置不安全绕过或改环境。

按分级验证预算未运行完整测试或新测试文件；没有重复C0 hash/download、重跑诊断/资格、额外训练、完整S1或official test。新正式链未运行是版本同步门禁阻塞，不是这些项目通过；新final schema/hash的真实验收仍待合法训练产物。

## 9. 准确恢复步骤

1. 在限时CPU-only保险内恢复SSH/GitHub连接，云端Git fast-forward到已推送的最终工作提交；直接核验HEAD与tracked-clean，再检查新Bash/Python语法和原环境。不复制独立源码、不重建依赖、不重复诊断或1664资格。
2. 按protocol §13现有授权，为正式任务设置并复核有限平台关机保险和screen/监督器；三组各从冻结C0独立开始Natural→Grid→Replay，每组2560成功更新/attempted=successful/skip0，逐组final验收。禁止接旧1597、资格1664或组间权重。任一资格性故障立即停止，不自动重试或改protocol。
3. 三组合法完成才唯一完整S1；任务结束主动停GPU，仅CPU-only/短时保险取回三个final与完整证据，取完停机。直接本地CPU读回/hash与S1对应后沿原gate裁决。
4. 更新两份实时入口、正式结果报告与必要MMFR导航/索引；当前本报告只记录已完成定位、资格及未完成正式的精确边界，不能替代未来结果报告。
