# Natural Missing Round-1 Readiness

> 范围：2026-10-03综合执行单的目录整理、独立训练/输入/S1实现与最小CPU就绪检查；不包含正式实验。
> 起始HEAD：`34a4dea41c1e08a5a1f85711eb97107471445272`；branch：`perf/mmfr-a2-v3-pipeline-opt1`。
> 第一阶段docs commit：`ca65f4bdfc76e687746745df12e7c767a0a46049`（`docs(mmfr): normalize direction plan locations`）。
> 第二阶段readiness commit：待最终本地提交后回填收据；提交消息固定为 `feat(mmfr): prepare natural missing round1`。正文无法包含其自身commit的最终SHA；收据回填只形成明确的一行本地注记，不额外创建第三次提交。
> 实时事实与权限：[current-status](../main/MUSeg-current-status.md)；待裁决事项：[open-decisions](../main/MUSeg-open-decisions.md)。本报告是交付快照，不替代这两个实时入口。

## 1. Executive Summary

**PARTIALLY READY。** 独立配置、Natural/Grid/Replay配对输入、分割训练入口和新S1评价入口已实现，输入、模型构造/参数分组及合成评价的最小CPU检查通过。唯一已确认的资产阻塞是**精确C0 fixed-final尚未恢复到本地**：`C0_RECOVERY=BLOCKED_REMOTE_ACCESS`。因此真实C0严格加载、GPU训练预检、真实C0的极少量val-dev前向均未运行，`GPU_PREFLIGHT=BLOCKED`。

C0指已有E1训练对照，fixed-final指成功更新2560次后的固定权重；checkpoint是保存的参数/训练状态。CPU检查只证明准备代码的有限工程行为，不证明训练能跑、显存足够、数值稳定或Replay有效。当前不能无条件申请立即开跑正式三组训练；应先恢复该资产并完成已获条件授权的限量预检。

本轮GPU成功更新数为0，真实模型评价前向数为0；没有完整val-dev、official test、付费云资源操作或push。official test仍为 `sealed_unread`。

## 2. 本轮目录整理

科研正文原样迁移到：

- `MMFR/01_research/MMFR_direction_A_natural_missing_2026-10-02.md`：A优先。
- `MMFR/01_research/MMFR_direction_B_inference_protocol_2026-10-02.md`：B备用，不运行完整B1a。

Git确认两份正文100% rename，`临时/`不留第二份可编辑副本。根README、`doc/plans/README.md`、MMFR package index/changelog与两份实时文档只收口必要当前导航。没有搬移、改名、删除或重写A2-v3、E1/C0/F-lite、R-OE、A-v1、Oracle-A、旧evaluator/config、历史evidence或实验目录；仅增加显式选择启用的loader接口，旧分支保持原行为。未重建或手工编辑旧审核包。

## 3. C0 Recovery

- 所需SHA-256：`ca618b23d18eabb201a0d11d18da383ac99576d0feae5864e3233bda527d9a1a`。
- 状态：`BLOCKED_REMOTE_ACCESS`；canonical本地路径：未取得；本轮实际候选hash次数：0。
- 限定盘点覆盖项目checkpoint、cloud/transfer、outputs/experiments及历史转移定位记录，没有找到可核验的本地候选。仓库外查询结果不足以证明所有外盘都不存在，未全盘扫描。
- 已有远端定位：`/root/rivermind-data/cloud/MMFR_E1_Batch1A_local_transfer_20260922/C0/checkpoint/update-2560.pth`。
- 已有ZIP定位：`/root/rivermind-data/cloud/MMFR_AV1_local_val_conditional_20261001.zip`；成员 `checkpoints/C0/update-2560.pth`，既有收据记录321150608 bytes。来源为 `MMFR/02_evidence/delivery_mmfr_a_v1_local_val_20261001.md`，本轮没有重新取得/核验该成员。
- 远端访问/登录未建立，当前实例与计费状态未核验；未启动资源、下载或重训C0，也不使用其他权重代替。

新模型在CPU上构造为802个state keys。旧C0记录812个keys，差额与已核验10个旧可靠性auxiliary键一致；**这不是实际C0加载通过**。加载器只允许显式过滤这10个键，所有分割keys必须完整且shape匹配，未知同前缀键同样拒绝；报告列loaded/intentionally dropped/missing/unexpected，禁止全局 `strict=False`。

恢复步骤：用户提供现有本地包/权重，或在不创建付费资源的前提下建立获准访问；对实际候选计算一次SHA并核对，记录本地路径，然后做真实严格加载。当前无需先恢复其他baseline或NYUv2。

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

默认只允许2个val-dev样本；更大开发集评价需要显式formal-dev授权参数。test路径在打开前拒绝。当前只有合成CPU验证，没有读取val-dev标签或执行真实模型评价。

## 8. CPU Checks

实际完成并由主执行者复核的最小检查：

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

遵循验证预算，没有完整测试套件、全集统计重跑、临时test脚本、真实模型CPU前向或昂贵验证。最终报告索引JSON解析、新canonical入口文件存在性、内容复核及 `git diff --check`（working/index）均通过；未执行内容不记PASS。

## 9. GPU Preflight

**BLOCKED，未运行。** 精确C0未恢复，真实segmentation strict-load前置条件尚未满足；本机CUDA存在不足以解除这个门禁。Natural/Grid/Replay attempted=0、successful=0；loss finite、AMP/scaler、峰值显存、wall time和保存/真实resume均未取得实测证据。

极少量val-dev的C0 S1预检同样BLOCKED；不能把合成dummy CPU检查写成真实C0 forward通过。恢复资产后仅能在已授权范围执行每组最多3成功更新、极少量val-dev，失败就停止报告；正式三组必须重新从同一C0干净启动。

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

开始时已有目录审计报告、执行单、report-index中的审计entry及PACKAGE_INDEX的审计入口原样保留；不把它们误作本轮新产物或随代码提交。checkpoint、数据、大日志、outputs/mask cache不纳入Git。既有生成审核包保持历史快照。

## 11. 正式 Round-1 预算估计

正式训练预算3×2560=7680成功更新；完整S1为C0+三组×318图×3条件=3816 view前向。两者均未执行、未获得自动执行权限。

本轮没有有效GPU测时，因此**不给训练时间、显存余量或云费用估计**，也不套用历史速度。恢复C0并取得限量预检测时后，才能按实测step时间给区间；3步短样本受首次初始化、I/O、worker与不同缺失输入影响，估计必须说明不确定性。

## 12. 下一步申请

当前值得提交**条件性正式预算审核材料**，但还不值得请求“立即无条件开跑3×2560”。先恢复精确C0、核验一次hash与真实严格加载，再做本机已获授权的≤3成功更新/组及极少量S1预检，确认finite/AMP、显存、时间、输入telemetry和checkpoint保存；补齐证据后再申请正式训练与完整评价。

优先利用现有本机做限量预检；RTX5060 Laptop对batch10/480×640的容量未实测，不能先保证本机正式运行。若本机不足，4090仅是待批准的正式资源选项，应明确费用、设备合同和时间预算；当前无需也未获准创建/启动云实例。恢复远端既有文件需先明确可访问途径，不以启动付费实例替代资产恢复授权。

正式训练、完整val-dev、B1a、NYUv2/外部baseline、official test继续关闭。下一恢复点是精确C0资产及真实预检，不是直接运行正式三组。
