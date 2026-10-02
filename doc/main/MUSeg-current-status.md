# MUSeg 当前状态与唯一实时入口

> **事实与执行边界截至：2026-10-03。** Direction A NaturalMissing首轮readiness收口为 **PARTIALLY READY**：独立三策略配置、配对输入、continuation训练与新S1已实现，最小CPU检查通过；精确C0尚未恢复，`C0_RECOVERY=BLOCKED_REMOTE_ACCESS`，`GPU_PREFLIGHT=BLOCKED`。正式训练/完整评价仍未授权。A优先、B备用；下一恢复点是恢复精确C0并完成条件授权内预检，不是直接开跑。

## 当前事实与实际意义

C0是已有E1训练对照；fixed-final是2560成功更新后的固定权重。checkpoint是参数/训练状态快照。Natural保留原输入，Grid新增规则网格删除，Replay重放不同训练采集组的真实空洞形态。support是不含padding的真实图像区域；validity是其中有深度观测的位置。S1是新统一原尺度单视图评价，不等于旧Quick-Val。

- A/B正文已原样迁至 `MMFR/01_research/`，Git100% rename；临时目录不留第二份。历史runner/config/evidence冻结原位，loader只增加显式opt-in元信息/当前Depth/support接口。
- 新Round-1合同：同精确C0、全部分割参数、仅分割loss，关闭可靠性辅助训练/A2 mixed corruption/F-lite/R-OE/A-v1；LR1e-6（实际E1 C0的1e-5×0.1）、AdamW/WD.01、warmup128/poly.9、batch10/480×640、workers8、SyncBN、AMP fp16/TF32 on、原生NMF autocast。optimizer非有限/跳步停止，预检权重不能用于正式初始化。
- 配置隔离和CPU分割构造/optimizer覆盖通过：802 state keys、26,673,193 trainable参数；29个Geo.weight都在decay。真实C0载入尚未验证，不把旧812 keys与10个aux键的对应关系当作真实load通过。
- Replay源只允许冻结canonical train-dev，在打开前拒绝val/test/其他列表；1277样本/762组，源目标不同采集组。源mask nearest、目标Depth保持linear，新增删除仅作用当前有效且非padding区域，raw0对应normalized约−1.7142857。25%clean、新增有效删除率.10–.50、Grid误差≤.02、最多8候选，失败时两组共同取消额外删除而继续普通训练。
- 主执行者CPU检查：合成seed0/1/2旧/opt-in预处理原输出完全相等；2真实train-dev target×3计划位置共6槽位，4 matched/2 clean，源组/删除量/原洞/padding/no-mutation/Natural no-op通过。计划位置不代表模型更新。
- 新S1为scale1/noflip/B1/whole/right-bottom pad32/FP32/TF32off/crop原Label网格。自然/矩形50%当前有效删除/entire-missing观测在同一次运行的所有checkpoint间复用，forward前sample×condition稳定RNG。全图原始缺失率按既有train阈值固定分层；有效标签区域缺失率只作解释，不改分层。合成pad/crop、矩形固定几何可达性、RNG、公共mIoU与独立condition×group混淆计数通过；没有真实模型评价；另有合成CPU恢复epoch1 position0/2与epoch2 position0的顺序/普通增强/模型RNG一致，完整已知aux过滤与未知键/shape拒绝通过。这些不等于实际C0载入或GPU恢复。
- 本轮本机已有df2环境可用：Python3.10.20、torch2.7.0+cu128，CUDA枚举为RTX5060 Laptop GPU；没有GPU tensor/forward/backward或更新。Anaconda base的OpenMP导入问题未用不安全环境变量绕过，也未改全局环境。

**大白话：** 输入和配置等有限工程风险已检查，代码可以保存和继续恢复；仍缺真正的共同初始化权重，所以还不能说训练链、显存或Replay效果已经验证。

## 资产阻塞与授权边界

- 所需C0 SHA为 `ca618b23d18eabb201a0d11d18da383ac99576d0feae5864e3233bda527d9a1a`；canonical本地路径未取得，实际候选hash0次。限定项目/转移/历史记录盘点没有可核验候选，不声称穷尽所有外盘。
- 远端定位：`/root/rivermind-data/cloud/MMFR_E1_Batch1A_local_transfer_20260922/C0/checkpoint/update-2560.pth`；既有ZIP `/root/rivermind-data/cloud/MMFR_AV1_local_val_conditional_20261001.zip` 的 `checkpoints/C0/update-2560.pth`。本轮未建立远端登录、下载或启动付费资源；禁止重训另一个C0。
- 真实C0 strict segmentation load、每组≤3成功更新GPU预检及极少量val-dev C0 S1均 **未运行/BLOCKED**。实际GPU attempted/successful均0，无实测时间/显存，不能估计正式耗时。
- 正式合同预算3×2560=7680成功更新、C0+三组×318×3=3816 S1 view前向；这些计数只是待批预算。完整val-dev、B1a33072 view、NYUv2/外部baseline和paid cloud均未授权。本轮未做完整测试/全集统计/hash库或额外审计系统。
- official test仍 **sealed_unread**。当前云实例/GPU/计费状态待核验；最后直接观测仍是2026-10-01历史记录，本轮不操作生命周期。

## 旧路线结论保持

A-v1仍stop：learned三hard−off为−0.0018730026999946858pp，未达+0.50pp；F-lite Main去留、R-OE v2约+0.01pp及入口资格仍独立待裁决；Oracle-A仅关闭具体抑制动作族。新A不是旧A-v1，不自动复活任何旧路线权限。历史详细身份/结果由既有正式报告与Git history留存，不改写。

## 证据、提交与准确恢复点

- [本轮readiness报告](../reports/2026-10-03-natural-missing-round1-readiness.md)：实际完成、CPU证据、C0阻塞、未运行GPU、12节交付与Git身份收据；[首轮protocol](../../MMFR/01_research/natural_missing_round1_protocol.md)：最终执行合同。
- canonical [A优先](../../MMFR/01_research/MMFR_direction_A_natural_missing_2026-10-02.md)/[B备用](../../MMFR/01_research/MMFR_direction_B_inference_protocol_2026-10-02.md)；[MMFR导航](../../MMFR/00_control/PACKAGE_INDEX.md)、[计划索引](../plans/README.md)、report-index已登记。旧生成review packet不代表本轮。
- [原调查](../reports/2026-10-02-direction-audit-local-evidence.md)、[接入审计](../reports/2026-10-02-direction-plans-project-readiness-upper-review.md)、[目录审计](../reports/2026-10-03-project-directory-responsibility-audit.md)、[C0转移收据](../../MMFR/02_evidence/delivery_mmfr_a_v1_local_val_20261001.md)保留原始结论。
- 分支 `perf/mmfr-a2-v3-pipeline-opt1`，起始HEAD `34a4dea41c1e08a5a1f85711eb97107471445272`；目录提交 `ca65f4bdfc76e687746745df12e7c767a0a46049`。第二阶段按指定消息 `feat(mmfr): prepare natural missing round1` 本地保存，准确SHA见报告提交后收据/Git；不push。报告自身SHA收据将在提交后只回填一行本地注记，不为自引用再创建第三次提交。既有用户审计文件/索引entry与导航dirty保留，outputs/checkpoint不入Git。

下一恢复顺序：提供/取回现存精确C0→对实际候选一次hash核对→真实严格分割加载→现有本机限量预检（Windows workers0仅preflight）→补测时/显存/finite/telemetry/checkpoint证据→请求正式三组和完整S1授权。若本机容量不足，4090只作为需另批费用/设备合同的选项。此前停止，不运行正式实验。
