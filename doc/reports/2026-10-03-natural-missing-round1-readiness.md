# Natural Missing Round-1 Readiness

> 范围：2026-10-03目录整理、独立训练/输入/S1实现、初始CPU检查、CPU-only C0恢复、本机限量GPU/2图S1，以及后续获批的4090云端三组各3成功更新与停机收口；不包含正式实验。
> 本机恢复阶段起始HEAD：`caaeda289cb68767c9fa3e0ea011e2f4e4ede1ec`，收口提交 `d47a4be1c139cde1a9c304b5bf36239069ec2bd0`，当时未push。云端阶段起始HEAD为该收口提交；先将已有3提交及必要监控代码推送origin，云端实际干净运行提交 **`18271ad0b0c3e7ba81c630c099d02592c02a45f4`**。本次后续文档收口另行提交推送，不追逐报告自身SHA。
> 起始HEAD：`34a4dea41c1e08a5a1f85711eb97107471445272`；branch：`perf/mmfr-a2-v3-pipeline-opt1`。
> 第一阶段docs commit：`ca65f4bdfc76e687746745df12e7c767a0a46049`（`docs(mmfr): normalize direction plan locations`）。
> 第二阶段readiness commit：`caaeda289cb68767c9fa3e0ea011e2f4e4ede1ec`（`feat(mmfr): prepare natural missing round1`）。此行为第二阶段提交后的历史SHA收据，未纳入该commit；本次真实执行收口保留该收据，不为自身SHA追加提交。
> 实时事实与权限：[current-status](../main/MUSeg-current-status.md)；待裁决事项：[open-decisions](../main/MUSeg-open-decisions.md)。本报告是交付快照，不替代这两个实时入口。

## 1. Executive Summary

**READY_ENGINEERING_ONLY — 4090限量训练链与保存读取通过，正式实验待授权。** Natural/Grid/Replay分别独立从精确C0开始，在同一云端代码提交、batch10/480×640、workers8和原生NMF AMP合同下，各 **attempted3 / successful3 / skipped0**；持久任务Succeeded/exit0，共9次成功更新，没有追加预算或合同变更。九次loss、gradient检查与optimizer/scaler路径均通过；三个预检checkpoint已在云端CPU-only读取核对计数、802模型键、685 optimizer state、scaler及RNG保存，均为preflight-only、不可正式初始化或formal resume。详见§13。

初始精确C0恢复与严格加载（802 loaded/10已知aux dropped/missing0/unexpected0）、本机RTX5060三组首次forward OOM/successful0、2图真实C0 S1的6次FP32前向工程通过，均保留为先前阶段事实；本次未重复这些初始检查或任何评价。§3/§8/§9中的旧结果不被云端成功覆盖或改写。

C0是已有E1训练对照，checkpoint是保存的参数/训练状态。**实际意义：4090能在原训练合同下完成少量真实更新和可靠保存，硬件容量阻塞已解除；这不证明完整训练长期稳定，更不证明Replay优于Natural/Grid。** 本轮没有正式Round-1分数、科研GO/STOP或正式训练授权。

启动前已核验30分钟平台定时关机，运行后再次确认；GPU预检正常结束后09:43:01+08:00主动stop，直接确认Stopped；仅为小型取回另启CPU-only/GPU0并设置10分钟保险，最终09:46:59+08:00直接确认 **Stopped/GPU0**。SwanLab在线初始化返回RuntimeError，按条件性授权降为LOG_ONLY，保留明确SUCCEEDED终态；没有在线实验链接或上传已完成证据。正式训练、完整评价、official test均未运行；official test仍 `sealed_unread`。

## 2. 本轮目录整理

科研正文原样迁移到：

- `MMFR/01_research/MMFR_direction_A_natural_missing_2026-10-02.md`：A优先。
- `MMFR/01_research/MMFR_direction_B_inference_protocol_2026-10-02.md`：B备用，不运行完整B1a。

Git确认两份正文100% rename，`临时/`不留第二份可编辑副本。根README、`doc/plans/README.md`、MMFR package index/changelog与两份实时文档只收口必要当前导航。没有搬移、改名、删除或重写A2-v3、E1/C0/F-lite、R-OE、A-v1、Oracle-A、旧evaluator/config、历史evidence或实验目录；仅增加显式选择启用的loader接口，旧分支保持原行为。未重建或手工编辑旧审核包。

## 3. C0 Recovery

- 所需SHA-256：`ca618b23d18eabb201a0d11d18da383ac99576d0feae5864e3233bda527d9a1a`。
- 状态：`PASS`；canonical本地路径：`D:/0Project/DFormer/outputs/natural-missing-readiness-20261003/C0/update-2560.pth`；实际大小321150608 bytes；本次候选SHA计算次数 **1**，在直接确认关机后执行。
- 初始准备阶段限定盘点未取得本地候选，当时`BLOCKED_REMOTE_ACCESS`、hash0、无云操作；这是先前阶段事实，后续CPU-only取回由用户另行明确授权，未重复全盘扫描。
- 实际远端来源：`/root/rivermind-data/cloud/MMFR_E1_Batch1A_local_transfer_20260922/C0/checkpoint/update-2560.pth`。SSH直接确认该文件321150608 bytes，只下载该文件，transfer exit0。
- 备用ZIP `/root/rivermind-data/cloud/MMFR_AV1_local_val_conditional_20261001.zip` / `checkpoints/C0/update-2560.pth` 本次未下载/打开；既有转移收据 `MMFR/02_evidence/delivery_mmfr_a_v1_local_val_20261001.md` 保留。
- 唯一现有实例 `cpod-1vbh7faqcauq`，请求启动时间2026-10-03 **08:37:26+08:00**；显式 `--without-gpu A`，08:37:37直接确认Running、2CPU、4096MiB内存、GPU=0。API显示InstancePrice0.13，但本轮未取得最终账单，不将该字段自行换算成实际费用。
- 下载成功后 **08:41:10+08:00** 立即请求stop，**08:41:23+08:00** 再次直接查询确认`State=Stopped`、GPU=0（API StopTime1790988072）。先stop再本地hash/load，未在云运行模型、hash权重、创建新实例、启动GPU或重新开机；无须下载F-lite/R-OE/整包。

关机后在df2中真实strict segmentation load：**loaded802 / intentionally dropped10 / missing0 / unexpected0**。只排除`reliability_estimator.features`的4个固定buffer和`reliability_estimator.head.net.{0,2,4}.{weight,bias}`的6个已知辅助参数；全体分割键与shape匹配，无全局`strict=False`。之后每个训练进程和S1都再次严格加载原C0，但**没有再次哈希**。

本地资产/完整键清单收据：`outputs/natural-missing-readiness-20261003/C0/recovery-receipt.json`。资产恢复门禁已关闭，下一阻塞是本机训练显存，不是C0身份或登录。

## 4. Round-1 Protocol

最终合同见 [`natural_missing_round1_protocol.md`](../../MMFR/01_research/natural_missing_round1_protocol.md)，科学门槛沿canonical A正文，不改历史合同。

- Natural/Grid/Replay：同精确C0、train-dev、数据顺序、普通增强、全部分割参数与分割CE目标；原C0保留评价锚点。
- LR `1e-6`：实际E1 C0起始 `1e-5` ×0.1，不采用A2的 `6e-5`。
- AdamW，betas `.9/.999`，decay `.01`，bias/归一化WD0；所有29个Geo.weight纳入decay。
- warmup128成功更新，poly power `.9`，总2560成功更新/组；accumulation1、batch10、480×640、seed772961337。
- 普通mirror、scale `[.5,.75,1,1.25,1.5,1.75]`、crop/pad；正式workers8，Windows预检默认workers0只为preflight执行设置。
- SyncBN训练态、eps `.001`、momentum `.1`；单GPU单进程、DDP off，全部分割参数可训练。
- AMP fp16，scaler initial1024/growth2/backoff.5/interval2000；TF32 on，compile off。NMF保留原路径的原生autocast，不继承A-v1专用局部FP32修订。
- 不做training val或选最优epoch；每640成功更新保存recovery，最终 `update-2560.pth` fixed-final。预检checkpoint单独标记，不能用于正式初始化或formal resume。
- optimizer非有限/跳步即停止；输入配对失败只取消该槽位新增删除，仍正常训练，二者严格区分。

参数按实际E1 Common及 `utils/train.py` 最终执行覆盖读取；模型/几何/optimizer分类来源列在protocol。正式预算仍为3×2560=7680成功更新，**本轮执行0**。C0已有鲁棒训练历史，未来科学主张只能是该底座上的额外收益。

## 5. Config Isolation

`local_configs/MUSeg/DFormerv2_S_NaturalMissing.py::make_config(strategy)` 返回独立deepcopy；三组只改变strategy/run身份/输出路径，不复制三份完整配置。必要模型和数据字段显式恢复，legacy实验块设为None。

CPU直接检查：配置对象及可变字段互不污染；原segmentation模型中reliability_estimator、feature_adapter、roe_substitute、av1均为None；全部参数trainable。实际参数量26,673,193；optimizer base_decay302个参数张量、base_no_decay412个，新模块组为空，29个Geo.weight均被覆盖。没有新增可靠性loss、F-lite/R-OE/A-v1或A2 mixed corruption。

实际Depth16默认为 `D:/0Project/dataset/MUSeg_DFormer/Depth16`，支持既有环境变量覆盖。首次真实输入检查曾显式传入该转换数据集根，不能把那个检查误记为错误旧默认路径通过；默认路径修正后的readiness检查单独记入第8节。

## 6. Replay / Grid

源库仅接受冻结canonical `data/splits/MUSeg/dev-v1/train-dev.txt`，在打开前拒绝val/test/任意其他列表，核对既有train SHA；索引1277样本/762组。只读源Depth16的零值，不读源Label、val/test或预测。先抽不同目标采集组的源组，再抽组内样本，最多8候选。

显式元信息包含原尺寸、mirror、scaled尺寸、crop与四边pad。源二值mask nearest变换；目标Depth仍linear插值。support是不含padding的图像区域；当前validity是其中Depth>0的位置，不从normalized零值推断。新增删除仅在源mask∩target当前validity∩support；raw uint8置0，输入对应normalized约−1.7142857，padding保持normalized0。输出同步更新raw_depth、modal_x与current_validity，源batch不被修改。

私有稳定key包含seed、epoch、成功更新位置、sample/group与batch slot；普通augmentation RNG与配对RNG分开。25%clean槽位；Replay新增当前有效点删除率10%–50%；Grid按随机规则块顺序匹配，绝对删除率误差≤.02。有限候选失败时Grid/Replay共同取消新增删除，保留输入并记录attempts/reason，不扩大mask优化算法。

真实CPU输入小样本：2个train-dev target，在计划位置0/1/2共6槽位；4 matched、2 clean、0匹配失败。matched Replay/Grid新增数分别 `(38525,38666)`、`(44806,47200)`、`(31508,32007)`、`(58772,58288)`；全部误差合格、源目标不同组、原洞/padding不计新增且不被改写。target为 `06-01-01-0061-230920140058-02-99` 与 `06-01-01-0092-230920140132-08-99`。这些是CPU计划位置，**不是模型更新**；batch2/workers0仅用于检查输入。

合成无洞source另验证共同失败skip的重复性。小样本不能估计正式全程skip率或删除量分布。轻量本地记录位于 `outputs/natural-missing-readiness-20261003/cpu-input-check.json`，不进Git。

## 7. S1

新统一身份：original scale1、noflip、batch1、whole-image、右/底normalized0 pad到32倍数，FP32、TF32 off，forward后恢复padded网格并裁掉padding，再回原Label网格。这是新S1，不能叫旧Quick-Val；没有S2/S10。

三条件：原自然Depth、单连续矩形新增删除、entire-missing。每sample×condition在单次调用中生成一次raw观测，所有checkpoint复用内存对象，不建立持久PNG/mask/cache manifest。每次forward用sample×condition稳定key恢复相同Python/NumPy/Torch随机起点，配对HAM/NMF；不依赖checkpoint遍历顺序，不承诺跨硬件逐位一致。

矩形目标为原当前有效点50%（整数四舍五入），先固定一个依密度/长宽比得到的H×W，再用积分计数穷举该固定尺寸的合法平移；选最接近目标的位置，sample-key打破并列。记录geometry、bounds、actual/target与可达性；“不可达”只指该固定H×W平移集合，不指所有尺寸的矩形。既有自然洞不计新增，不用Label定位。

分层保持既有原始**全图**缺失率和train P25/P75；有效标签区域中的缺失率只作诊断，不改变原低/高集合。每condition独立累计dataset/采集组的all/low/high confusion，记录类别GT像素支持；调用统一公共mIoU实现，15类固定，union0记IoU0并纳入均值。保存混淆计数供后续精确阈值复核，不用公共展示四舍五入值取代原始计数。

默认只允许2个val-dev样本；更大开发集评价需要显式formal-dev授权参数。test路径在打开前拒绝。当前已完成2图真实C0 S1，细节与已验证边界见第9节；只证明极小工程预检，不是完整val-dev或正式策略比较。

## 8. CPU Checks

下列是最初准备阶段已完成并由主执行者复核的最小CPU检查，未在本次重复运行；后续真实C0/GPU/S1结果见第3/9节：

1. 独立三组配置/legacy关闭/合同值检查：PASS。
2. 合成seed0/1/2旧与新opt-in预处理前三个输出逐元素相等；raw normalization/support/padding：PASS。
3. 第6节2真实train-dev target/6槽位输入与新增删除telemetry：PASS；未运行模型。
4. 原分割模型CPU构造、参数trainable与optimizer分类/Geo覆盖：PASS；未做真实模型forward/backward。
5. canonical train索引与在打开前拒绝其他列表、确定性共同失败skip：PASS。
6. `conda run -n df2 python -m tools.mmfr.natural_missing_eval --self-check`：PASS；合成37×65→64×96 pad、下采样dummy输出恢复/crop、固定RNG、矩形geometry、公共confusion/无支持类政策。
7. 合成reader/runtime的S1三条件采集组累计：每条件独立计数2405像素，不混合三种条件；有效标签缺失率1.0而全图缺失率.05的合成例子仍保留low分层：PASS。
8. 训练恢复的合成CPU检查：epoch1 position0/2、epoch2 position0均恢复相同样本顺序、Python/NumPy/CPU-Torch普通增强输出及模型侧RNG：PASS（workers0；不是实际GPU checkpoint恢复）。严格过滤完整10个旧aux键、拒绝未知aux/分割missing/shape mismatch、接受无aux的新分割checkpoint：PASS（小型CPU模型与内存mock，无权重文件读取）。
9. readiness CLI的Grid配置默认Depth16/data目录存在、配置隔离/legacy off、C0未提供且未启动GPU：PASS。7个改动Python文件的定点语法编译与S1静态诊断通过。

环境：现有df2 Python3.10.20、torch2.7.0+cu128；CUDA枚举可用，设备为RTX5060 Laptop GPU，枚举不是GPU运行验证。Anaconda base导入torch遇到OpenMP重复运行库错误，未使用 `KMP_DUPLICATE_LIB_OK` 绕过，也未修改全局环境。

本次恢复执行未重复上述CPU测试或7文件编译；仅真实strict-load、三次限量GPU尝试、2图S1及实际输出定点核对。两个最小改动Python入口静态诊断无报错；现有报告索引/收据JSON、导航和working/staged差异在收口复核。没有完整测试套件、全仓扫描、临时test脚本、完整评价或其他高成本验证；失败的训练预检明确记录OOM，不记PASS。

## 9. 本机 GPU / 极小 S1 Preflight（此前阶段）

**FAILED_OOM，三组各在首次forward停止。** RTX5060 Laptop GPU总8151MiB；启动前nvidia-smi观察占用1559MiB（包含桌面/其他占用，不将其归给模型）。三组分别新建进程，从§3的同一个C0初始化，不resume、不接续其他预检，batch10/480×640、workers0只用于preflight，其余科研合同不变。

- Natural：attempted1 / successful0 / optimizer skipped0，第一次forward失败。
- Grid：attempted1 / successful0 / optimizer skipped0，第一次forward失败。
- Replay：attempted1 / successful0 / optimizer skipped0，第一次forward失败。
- 三组都在`DFormerv2.py::generate_1d_depth_decay`的width方向depth-decay分配处OOM，额外请求470MiB；异常报告当时PyTorch allocated约6.43GiB、reserved但未分配664.78MiB、GPU可用0。不按OOM提示自动切换allocator、batch、geometry或NMF精度，也未重试。
- Grid/Replay异常时实测**peak allocated7054.634MiB / peak reserved7354MiB**；Natural原CLI仅留下异常中的当时内存值，未采集独立peak计数，不把另外两组的数值冒记为Natural实测峰值。
- Grid/Replay失败进程wall分别7.6453s/5.9977s，含构造/加载/I/O/前向失败，**不是成功step时间**。初始scaler1024在attempt日志可见；未进入完整loss/backward/optimizer/scaler.update，不得写finite或数值稳定通过。无成功权重保存、真实GPU resume证据，无预检权重可用于正式初始化。

**真实配对输入已到达GPU前向入口：** Grid/Replay首批10槽位的目标顺序、源组/样本和候选匹配完全对应，4clean、5matched、1paired_skip；5个实际匹配槽最大误差0.00939993≤.02，源目标组不同且Replay率在.10–.50。第7槽在8候选后`grid_match_not_found`，最后候选误差0.02546536，两组实际新增删除均0；该失败候选不是已应用的超标配对。输入取消额外删除与optimizer skip不是同一件事。10槽的小样本不用于估计长期skip率或策略效果。

**极小真实C0 S1：PASS_ENGINEERING_ONLY。** allowlisted val-dev前2图 `03-01-01-0066-240526121121-08-99`、`03-01-01-0066-240526121123-12-99`，同1采集组、都是原始全图low；C0×2图×3条件完成6次FP32/TF32off前向，6次sample×condition RNG复位。finite logits与回原Label网格检查通过；示例932×1082→960×1088，pad bottom28/right6，metric前crop。每条件独立698762有效Label像素，采集组/类别support和all/low/high confusion计数落盘。矩形新增476439/467727点，两图均精确删除原当前有效点50%、声明固定几何下可达；原自然新增0，entire-missing新增952878/935454点。未验证high stratum、跨checkpoint观测复用或真实GPU恢复。

轻量执行汇总与日志在 `outputs/natural-missing-readiness-20261003/execution-summary.json`、`natural-gpu.log`/`grid-gpu.log`/`replay-gpu.log`；S1 JSON为 `s1/checkpoints/update-2560-ca618b23d18e.json`。S1 JSON内指标只为调试产物，**本报告不公布正式Round-1分数、不据此比较策略**。资产、大日志/权重/outputs不入Git。

复现边界：训练使用 `python -m tools.mmfr.natural_missing_train --mode preflight --strategy <Natural|Grid|Replay> --c0-checkpoint <§3路径> --verified-c0-sha256 <§3 SHA> --successful-updates 3 --preflight-workers 0`；在本机已确认OOM后不重复运行。S1使用 `python -m tools.mmfr.natural_missing_eval --dataset-root D:/0Project/dataset/MUSeg_DFormer --checkpoint <§3路径> --output-dir <本地outputs>/s1 --device cuda --sample-limit 2 --verified-c0-sha256 <§3 SHA>`，复用已核验身份不重复hash。当前权限到此停止。

## 10. Changed Files

第一阶段：A/B原样rename；根 `README.md`、`doc/plans/README.md`、MMFR `PACKAGE_INDEX.md`/`CHANGELOG.md` 与两份实时状态文档的必要入口收口。

第二阶段新增：

- `MMFR/01_research/natural_missing_round1_protocol.md`：冻结执行合同。
- `local_configs/MUSeg/DFormerv2_S_NaturalMissing.py`：独立三策略factory。
- `utils/dataloader/natural_missing.py`：纯CPU输入、train-only source与deterministic pairing。
- `utils/natural_missing_runtime.py`：原分割模型构造与严格加载。
- `tools/mmfr/natural_missing_train.py`：独立continuation入口，复用公共optimizer/scheduler/checkpoint/RNG，不继承旧科研gate。
- `tools/mmfr/natural_missing_eval.py`：仅新S1、三条件、观测/RNG配对和公共指标。
- 本报告。

第二阶段修改：`utils/dataloader/dataloader.py`/`RGBXDataset.py`只作opt-in元信息扩展；实时状态/open-decisions、report-index、MMFR导航/changelog记录实际收口。

此前本机恢复执行仅最小修改 `tools/mmfr/natural_missing_train.py`（attempt入口日志和reserved峰值记录，便于失败定位）与 `natural_missing_eval.py`（显式复用恢复责任者已核验身份，CLI只允许单checkpoint+冻结C0 SHA，其他checkpoint默认仍直接hash，输出明确provenance）。不修改backbone/decoder/loss、batch、geometry、NMF精度或科研协议。其余修改为本报告、两份实时文档和既有索引/导航/changelog。

开始时已有目录审计报告、执行单、report-index中的审计entry及PACKAGE_INDEX的审计入口原样保留；不把它们误作本轮新产物或随代码提交。checkpoint、数据、大日志、outputs/mask cache不纳入Git。既有生成审核包保持历史快照。

## 11. 正式 Round-1 预算估计（4090短样本）

正式训练预算3×2560=7680成功更新；完整S1为C0+三组×318图×3条件=3816 view前向。两者均未执行、未获得自动执行权限。本机8GB的既有OOM仍然成立；本节只依据新4090实测，不从失败进程wall外推。

各进程首次成功step明显慢于其后两step，Natural/Grid/Replay为11.0994/3.3473/4.3114s；本次没有进一步剖析首步开销。只取每组step2–3的均值：Natural **0.771277s**、Grid **0.711740s**、Replay **0.683076s**。按$2560t_s$作条件性线性估计，分别为32.91/30.37/29.14分钟，合计 **92.42分钟（约1.54小时）**。运行态API显示InstancePrice **1.88**；完整计费口径/费用报价及最终账单未核验，正式总费用仍待确认。

计时从输入传入GPU且同步后开始，到optimizer完成再同步结束，**不含取batch、Grid/Replay构建输入、checkpoint保存、环境/加载和评价**。每组只有2个后续step，仍在warmup最初位置，不能宣称稳态全程均值或长期稳定；估计未覆盖完整训练墙钟，应另列额外预算。整个三组持久预检任务实测65s，但包含三个独立初始化/限量保存，不能直接乘为正式耗时；完整S1耗时/费用尚无新证据，不拿2图结果硬推。

本估计只供下一轮预算审批，不授权启动正式任务、重启GPU或接续预检权重。

## 12. 下一步申请

下一步是裁决 **正式3×2560训练、完整3816-view S1、设备/费用/时间上限及运行责任**，不是再重复本机OOM或自动扩展预检。若获准，正式三组仍从§3同一精确C0独立初始化，保持batch/geometry/精度/输入合同；本轮三个preflight-only权重均不得作正式初始化或formal resume。SwanLab在线监控本次不可用，若下一阶段必须在线，需另行确认既有配置/登录问题，而非冒称已有可用链接。

完整训练长期finite、workers8实际恢复一致性与完整S1的high集合/跨checkpoint复用尚未真实验证；checkpoint读取通过不等于运行resume通过。本轮不为这些缺口追加更新或评价。Main-Val/B1a、NYUv2/外部baseline和official test继续关闭。最终实例Stopped/GPU0，小型证据已取回；无需再次开机、取回C0或重复hash。

## 13. 已批准4090限量云端预检：真实执行收口

### 13.1 身份、环境与最小代码

- 已批准计划仅现有 `cpod-1vbh7faqcauq` 有卡启动、三组各≤3成功更新；先提交推送再云端核对同SHA，30分钟平台关机保险和正常结束主动停机；正式预算不开放。
- 本轮必要代码提交 **`18271ad0b0c3e7ba81c630c099d02592c02a45f4`**，与已有3本地提交一起推送 `origin/perf/mmfr-a2-v3-pipeline-opt1`。云端 `/root/rivermind-data/DFormer` 原origin实际为 `https://github.com/Newton-liub/DFormer.git`，原checkout干净，fetch/fast-forward后直接确认同一HEAD及空git status；没有强制reset、fork误拉或覆盖历史dirty。
- `tools/mmfr/natural_missing_cloud_preflight.py` 顺序调用三个独立 `sys.executable -u -m tools.mmfr.natural_missing_train` 子进程；固定preflight/同C0/3成功更新/workers8，无resume/formal参数或重复hash。监控父进程复用 `ExperimentTracker`，不导入训练随机状态；训练文件唯一新差异为 `_emit` 的flush。主执行者直接复核实际diff，两文件静态诊断无报错；没有新增test脚本或全套测试。
- 真实云环境Python3.10.16、torch2.1.2+cu118/CUDA11.8；GPU1 **NVIDIA GeForce RTX4090/24564MiB**、CPU14/内存32768MiB。数据环境 `DFORMER_DATA_ROOT=/root/rivermind-data/dataset`、`MUSEG_DEPTH16_ROOT=/root/rivermind-data/dataset/MUSeg_DFormer/Depth16`；仅确认所需RGB/Label/Depth/Depth16目录、train-dev1277条和原C0文件大小321150608 bytes，没有重建环境或再hash。

### 13.2 三组成功更新与保存读取

持久任务 `natural-missing-20261003-18271ad`，Created/Started Unix1790991687（09:41:27+08:00），Finished1790991752（09:42:32），**Succeeded/exit0**，wall65s。各组result均complete=true、attempt cap3、workers8、formal eligible false：

- **Natural：attempted3 / successful3 / skipped0**；loss依次0.150492117/0.075245894/0.149992928；成功step依次11.099412/0.708800/0.833753s；peak allocated18955.576171875MiB、reserved20556MiB。训练循环至最终保存elapsed19.625719s。
- **Grid：attempted3 / successful3 / skipped0**；loss0.146257848/0.071628839/0.139067933；step3.347309/0.735882/0.687599s；peak allocated18955.576171875MiB、reserved20556MiB。循环至保存elapsed10.361942s。
- **Replay：attempted3 / successful3 / skipped0**；loss0.150824711/0.070011154/0.178284675；step4.311388/0.742372/0.623779s；peak allocated18954.138671875MiB、reserved20554MiB。循环至保存elapsed9.853987s。

九次loss finite=true、optimizer_step_applied=true；已运行代码在每次optimizer之前检查gradient finite，任何非有限都会异常退出，因此这9次均通过该检查。scaler均1024；LR依次0、7.8125e-9、1.5625e-8，第一步LR0仍计数optimizer成功调用，后两步进入正LR；全部处于最初warmup。loss只为工程finite证据，不能据小样本值比较科研效果或训练收敛。

三个 `preflight-update-000003.pth` 正常保存到持久盘，GPU先停后在CPU-only读回：Natural321113947 bytes，Grid/Replay各321117339 bytes；schema为dformer-training-checkpoint-v2，model802键，optimizer685 state且保存step均3，global_optimizer_step3，cursor epoch1/batch_position3/attempted3/successful3/skipped0，AMP scale1024/growth tracker3，训练RNG与epoch起点RNG均存在。三个保存protocol的Git/C0、batch10/workers8/原生NMF AMP与策略身份一致，preflight_only=true/execution_mode=preflight。**保存和读回通过，不是resume运行通过；预检权重不能用于正式初始化或formal resume。** CPU读回检查曾因日志中的JSON整数行出现TypeError，增加dict类型筛选后只重做只读核对；没有新增模型执行、更新或预算。

### 13.3 输入配对与监控边界

三批每组共30槽：Natural全为clean_control/新增0；Grid/Replay逐批为4clean/5matched/1skip、2clean/8matched/0skip、1clean/7matched/2skip，合计均 **7clean / 20matched / 3paired_skip**。三组全部30目标顺序相同；Grid/Replay源组/样本、status与有序候选历史逐槽相同。20个matched源目标组均不同，已应用Replay率 **0.1162650849–0.4617859556**，最大matched误差 **0.0197314467≤.02**。clean/paired_skip两组applied deletion均0且applied rate为null；成功训练没有把输入skip计为optimizer skip。

paired_skip保留失败候选的grid/replay_rate字段，但它们不是实际应用值。例如首批skip的最后候选Replay率0.007575、误差0.025465，以及第三批skip候选Replay率0.625027，均没有应用删除，不能冒记为已应用超界。父执行者直接复核result/readback与Grid原始批记录的关键边界；限定四文件独立只读复核也确认上述配对无违例。证据为三组日志第18/20/22行的成功记录（不重复计attempt记录），全量来源日志已取回；小样本不推断长期skip率或策略效果。

SwanLab显式请求online，现有包可找到，但初始化返回 **RuntimeError**；按条件性授权记录LOG_ONLY继续，没有在线实验链接、在线可见性或上传成功证据。receipt保存terminal/succeeded1、failed0、interrupted0、exit_code0。异常只打印类型，根因未进一步核验；不暴露凭据、不安装监控依赖、不为在线刷新占用GPU等待。在线完成/失败SDK路径及信号中断路径本轮未真实触发验证。

### 13.4 定时关机与最终资源状态

- **09:37:28+08:00**设置30分钟后的保险：Unix1790993248、**10:07:28+08:00**。启动前schedule show直接确认scheduled=true；09:37:50请求有卡启动、API StartTime1790991474，实际Running/GPU1；启动后再次直接核验同截止，不延长或改变规格。
- 正常任务结束后 **09:43:01+08:00**主动stop，**09:43:13+08:00**再次直接确认Stopped（API StopTime1790991782）。启动到平台stop约308s；没有等20分钟、没有为大文件取回保持GPU运行，也没有人工保险关机事件。
- 小型证据取回另设10分钟CPU保险：Unix1790992450、**09:54:10+08:00**，短于原GPU硬截止；09:44:16显式without-gpu A启动，API StartTime1790991862，直接确认CPU2/4096MiB/**GPU0**并再次核验保险。只读三个保存权重和五个小型文件，无训练/模型前向。
- 5个文件transfer均exit0后，**09:46:49+08:00**立即stop；**09:46:59+08:00直接确认Stopped/GPU0**，API StopTime1790992011。GPU阶段运行态InstancePrice1.88，CPU-only0.13；最终账单未核验，不把CPU价格或升级查询0当GPU运行价。

### 13.5 证据位置、验证预算与交付

远端根 `/root/rivermind-data/cloud/natural-missing-preflight-20261003-18271ad/`，`monitor/`有result/readback/三组日志，`checkpoints/NaturalMissing-<strategy>-preflight/development/seed-772961337/checkpoint/preflight-only/`有三个预检权重。只取小文件，不取大权重；本地根 `outputs/natural-missing-cloud-preflight-20261003/` 包含 `preflight-result.json`、`checkpoint-readback.json`、`Natural.log`/`Grid.log`/`Replay.log` 和主执行者记录的 `lifecycle.json`，均不进Git。

本次实际检查为必要diff/两文件静态诊断、同HEAD/相关源码干净、平台保险/实际GPU和最终Stopped直接查询、三组3成功更新、输入日志与checkpoint读回及五个小文件取回。未运行完整测试或重复初始CPU、本机GPU、真实resume、S1/完整val-dev/Main-Val/B1a、正式训练、其他数据集/baseline或official test；原因是限定预算且本次资格风险已经由所授权的九步与保存核对覆盖，未运行项不冒称通过。

本报告、两份实时入口、既有report-index/MMFR导航/changelog/protocol按实际收口更新并提交推送同分支；用户既有目录审计报告/执行单及共享索引无关dirty选择性排除并保留。旧生成审核包仍历史快照，不重建或手改；历史本机OOM/C0/S1原结果及日志不追改。最终停止于正式预算审批，不自动重启或继续训练。
