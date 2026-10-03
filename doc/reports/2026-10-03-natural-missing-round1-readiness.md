# Natural Missing Round-1 Readiness

> 范围：2026-10-03综合执行单的目录整理、独立训练/输入/S1实现、初始CPU检查，以及后续获准的CPU-only C0取回、确认停机后单次哈希/真实加载、本机限量GPU与2图S1；不包含正式实验。
> 本次真实执行起始HEAD：`caaeda289cb68767c9fa3e0ea011e2f4e4ede1ec`。执行收口使用 `test(mmfr): validate natural missing round1 readiness` 本地提交，不push；准确身份以Git提交为准，不追逐报告自引用。
> 起始HEAD：`34a4dea41c1e08a5a1f85711eb97107471445272`；branch：`perf/mmfr-a2-v3-pipeline-opt1`。
> 第一阶段docs commit：`ca65f4bdfc76e687746745df12e7c767a0a46049`（`docs(mmfr): normalize direction plan locations`）。
> 第二阶段readiness commit：`caaeda289cb68767c9fa3e0ea011e2f4e4ede1ec`（`feat(mmfr): prepare natural missing round1`）。此行为第二阶段提交后的历史SHA收据，未纳入该commit；本次真实执行收口保留该收据，不为自身SHA追加提交。
> 实时事实与权限：[current-status](../main/MUSeg-current-status.md)；待裁决事项：[open-decisions](../main/MUSeg-open-decisions.md)。本报告是交付快照，不替代这两个实时入口。

## 1. Executive Summary

**PARTIALLY READY — C0已恢复，RTX5060训练容量阻塞。** 独立配置、配对输入、分割训练与新S1入口已实现；精确C0在CPU-only云取回后已经本地落盘，云实例已直接确认Stopped。关机后仅计算一次SHA-256，身份匹配；真实严格分割加载802键、只排除已知10个辅助键、missing/unexpected均0，`C0_RECOVERY=PASS`。

Natural/Grid/Replay各自从该C0启动独立进程，保留batch10/480×640/AMP合同；各在第一次前向发生显存不足（OOM，GPU内存无法满足分配），各attempted=1/successful=0/optimizer skipped=0，`GPU_PREFLIGHT=FAILED_OOM`。不存在有效更新耗时、loss finite或backward/scaler运行通过证据。2张val-dev的真实C0 S1完成6次FP32前向，finite、pad/crop、三条件和采集组统计通过，`S1_TINY_PREFLIGHT=PASS_ENGINEERING_ONLY`；不披露或使用小样本分数作为正式Round-1结果。

C0是已有E1训练对照，fixed-final是成功更新2560次的固定权重；checkpoint是保存的参数/训练状态。**实际意义：共同底座和真实评价入口已确认，但本机8GB GPU不能按当前合同完成训练预检，尚不具备立即正式训练资格。** 建议另行申请4090或同等级更大显存设备的限量预检权限，不在本轮启动。

本轮GPU成功更新总数0、真实模型评价前向6；云操作仅现有实例CPU-only取回→立即停机，无GPU云启动、完整val-dev、正式训练、official test或push。official test仍为 `sealed_unread`。

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

## 9. GPU Preflight

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

本次真实执行仅最小修改 `tools/mmfr/natural_missing_train.py`（attempt入口日志和reserved峰值记录，便于失败定位）与 `natural_missing_eval.py`（显式复用恢复责任者已核验身份，CLI只允许单checkpoint+冻结C0 SHA，其他checkpoint默认仍直接hash，输出明确provenance）。不修改backbone/decoder/loss、batch、geometry、NMF精度或科研协议。其余修改为本报告、两份实时文档和既有索引/导航/changelog。

开始时已有目录审计报告、执行单、report-index中的审计entry及PACKAGE_INDEX的审计入口原样保留；不把它们误作本轮新产物或随代码提交。checkpoint、数据、大日志、outputs/mask cache不纳入Git。既有生成审核包保持历史快照。

## 11. 正式 Round-1 预算估计

正式训练预算3×2560=7680成功更新；完整S1为C0+三组×318图×3条件=3816 view前向。两者均未执行、未获得自动执行权限。

本机没有完成一次成功更新，因此**正式本地训练在当前合同下不可执行，无法给出可信完成时间**。Grid/Replay的7.6453s/5.9977s是含初始化的失败进程wall，不能乘7680，更不能外推4090速度。完整S1也不从这2图推出可靠全集耗时/费用。

建议未来另批RTX4090 24GB或同等级大显存设备的每组≤3次预检，用实测成功step时间及allocated/reserved峰值评估正式预算；设备容量更大是候选理由，不代表其finite、NMF AMP路径或训练资格已经通过。以各组实测稳定step时间$t_s$估计单组$2560t_s$、三组之和，并单列加载/保存/I/O/初始化与短样本不确定性；当前$t_s$未取得，不填写数值。该建议**不授权也未执行本轮任何GPU云启动/再次开机**。

## 12. 下一步申请

下一步适合申请的是**更大显存设备的限量预检及明确费用/设备合同**，不是立即无条件开跑3×2560。精确C0资产和2图真实S1的工程门禁已通过；当前阻塞是batch10/480×640在本机首次前向OOM，完整训练loss/backward/optimizer/scaler/save/resume链尚未验证。

若另行批准更大显存预检，应仍分别从同一精确C0干净初始化，三组每组≤3成功更新，保持batch/geometry/精度/输入合同，禁止接续预检权重作正式初始化；取得成功step测时/显存/finite与保存证据后再申请正式7680成功更新和完整3816 S1 views。本轮CPU-only资产恢复授权已执行并停机，**不得据此再次启动云实例**。

正式训练、完整val-dev、B1a/Main-Val、NYUv2/外部baseline、official test继续关闭。当前执行已停止；恢复点是用户对更大显存预检/费用的独立裁决，不是重新hash、重新取回C0或本机OOM重复尝试。
