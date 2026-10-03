# Natural Missing Round-1 执行合同

> **文档角色：** Direction A 首轮执行合同，不替代科研方案或实时权限。
> **形成/核验时点：** 2026-10-03。
> **实时入口：** [当前状态](../../doc/main/MUSeg-current-status.md)；[开放决策](../../doc/main/MUSeg-open-decisions.md)。
> **科学依据：** [Direction A canonical 正文](MMFR_direction_A_natural_missing_2026-10-02.md)，§5–6；[接入审计](../../doc/reports/2026-10-02-direction-plans-project-readiness-upper-review.md)与[目录审计](../../doc/reports/2026-10-03-project-directory-responsibility-audit.md)。
> **权限：** 最新数值定位、证据驱动最小修复及重新正式首轮授权见§13，必须逐阶段通过门禁；旧非有限运行仍INVALID/BLOCKED，不续跑1597。初始实现/限量预检及2026-10-03正式授权与停止过程见§8–12历史记录。§1–7科研合同和canonical门槛不变，局部精度修订只允许在直接诊断证据支持后明确登记。

## 1. 对象、身份与合法主张

Natural、Grid、Replay 三组从同一个 C0 fixed-final（成功更新2560次后的固定权重）初始化；另保留未续训的C0作为评价锚点。精确来源SHA-256为 `ca618b23d18eabb201a0d11d18da383ac99576d0feae5864e3233bda527d9a1a`。本轮不得重训另一个C0代替。

模型为原DFormerv2-S骨干与HAM分割解码器；segmentation-only continuation 指继续训练全部分割参数、只使用分割损失，关闭旧辅助可靠性头/损失、A2混合退化、F-lite、R-OE和A-v1分支。C0历史上已有A2/E1鲁棒训练，合法主张仅为“在已有鲁棒训练底座上研究自然缺失形态覆盖的额外任务收益”，不是首次学会缺失鲁棒性。

三组固定同源权重、数据、样本顺序策略、普通几何增强、optimizer/scheduler、预算、保存和评价规则；唯一核心变量为是否新增Depth删除及删除空间形态。运行身份采用 `NaturalMissing-Natural`、`NaturalMissing-Grid`、`NaturalMissing-Replay`。预检模型/optimizer/RNG全部丢弃，正式运行必须重新从同一C0干净初始化。

## 2. 数据边界

- 只用冻结train-dev 1277条训练、val-dev 318条开发评价，按已有采集组身份排除源目标同组；不打开official-test清单或任何test数据。既有dev清单SHA继续为来源合同，不另建manifest或hash库。
- RGB/Depth8/Label模型数据根默认 `D:/0Project/dataset/MUSeg_DFormer`；源mask使用该转换数据中直接保留的原始 `Depth16==0`。实际目录由独立配置明确声明，保持 `DFORMER_DATA_ROOT` 的数据根覆盖语义。
- Replay库只来自train-dev原始Depth16，先均匀抽源采集组，再抽组内样本；不使用val、Label、预测或val分数选源。
- train固定缺失率阈值：低 `q <= 0.10761048923865359`，高 `q > 0.4944140559923207`；中间为中缺失。旧val计数80/174/64只作历史输入证据，本轮不重跑全集统计。

## 3. 实际E1 C0参数与新训练合同

最终值按实际正式runner覆盖顺序核对，不以A2历史 `mmfr_a2.frozen` 字段当作C0当前schedule。

- 新base LR为 `1e-6`，即E1 C0起始 `1e-5` 的0.1；不是A2 `6e-5` 的0.1。
- AdamW，betas `(0.9, 0.999)`；decay组WD `0.01`，bias/归一化组WD `0`；29个 `*.Geo.weight` 全部纳入decay，全部分割参数恰好一次入optimizer。E1公共分组函数可复用参数分类，但不继承candidate whitelist或旧stage gate。
- warmup 128成功更新，随后poly power `0.9`，总2560成功更新；复用 `WarmUpPolyLR` 的算术，不用预检3步缩短scheduler。成功更新坐标与attempted计数分别记录；非有限loss/gradient或AMP跳过optimizer step即停止，不能吞掉失败继续换样本；合格运行要求attempted=successful、optimizer skip=0。输入槽位的共同skip只是取消新增删除，仍参与正常分割更新，与optimizer skip不同。
- batch 10，accumulation 1，480×640；128 batch槽位/epoch，20个epoch-equivalent。普通mirror、随机scale `[0.5, 0.75, 1.0, 1.25, 1.5, 1.75]`、crop/pad沿已有loader。
- workers 8为正式配置；Windows预检可显式workers0，标记preflight-only，不据此改变正式科学合同。
- SyncBatchNorm训练态，单GPU单进程、DDP off。BN eps `1e-3`、momentum `0.1`，不冻结BN或C0分割参数。
- 训练AMP fp16、TF32 on；GradScaler initial1024、growth2、backoff0.5、interval2000；torch compile off。
- NMF训练保留原分割路径及原生autocast行为，不继承A-v1特定局部FP32修订。NMF算法、随机基底、迭代数不改。推理另固定FP32/TF32 off。
- 训练时validation disabled，无val selector或早停；每640成功更新保存recovery、最终 `update-2560.pth` fixed-final。预检保存单独标识的checkpoint，禁止用于正式初始化/resume。
- train seed `772961337` 为三组共同起点。样本普通增强按seed/epoch/index固定Python、NumPy和CPU-Torch私有状态，并恢复调用者状态；sampler和worker-generator各用独立epoch-key，不消耗模型全局RNG。恢复时重建当epoch顺序并重放到cursor，再恢复模型RNG；position0也不依赖lazy sampler从全局RNG抽seed。配对输入使用另一个稳定key，同位置各组共用ordinary augmentation。

源码依据：`local_configs/MUSeg/DFormerv2_S_MMFR_E1_Batch1A_Common.py` 的schedule覆盖与 `configure_e1_batch1a_candidate`；`utils/train.py` E1强制AMP/SyncBN、model norm、AdamW及scaler构造；`DFormerv2_S_MMFR_A2_Common_v3.py` 公共数据/几何字段；`DFormerv2_S_Base.py` 模型几何；`utils/init_func.py::build_e1_optimizer_param_groups` 参数分类。E1新模块LR `3e-5`、可靠性训练与A2 curriculum均不迁入新合同。

## 4. checkpoint加载与恢复

关闭reliability head后，只允许10个已核验旧auxiliary键被过滤：`reliability_estimator.features.{gaussian_kernel,sobel_x,sobel_y,laplacian}`与`reliability_estimator.head.net.{0,2,4}.{weight,bias}`。未知同前缀键也必须拒绝；分割路径必须完整匹配。报告分别给loaded、intentionally dropped、missing、unexpected，未知键/shape mismatch或分割缺失直接BLOCKED；禁止全局 `strict=False`。

C0是weights-only新起点，不恢复旧optimizer/scaler/scheduler/RNG。新运行的recovery必须校验strategy、source身份、数据合同、预算、精度与预检/正式模式，保存模型/optimizer/scaler/RNG/数据位置/成功与尝试计数；无法精确恢复时拒绝恢复而不是默默跳过数据。跨Windows worker执行设置与不同环境精确等价不作未经检查的保证。

## 5. 纯输入与确定性配对

Support指不含padding的真实图像区域；validity指当前数值Depth>0的位置，不能用normalized零值推断两者。opt-in预处理返回mirror/resize/crop/pad元信息、当前uint8 Depth及显式support，默认旧loader三元输出/旧行为不变。

源mask用nearest变换，目标Depth保留既有linear插值。若源尺寸不同，先nearest到目标原尺寸，再应用目标mirror、scale、crop、pad；这只是形态重放，不宣称物理传感器仿真。新增删除只在 `replay_mask & target_valid & support`。当前自然洞不计新增删除，不补洞；raw uint8置0后按Depth mean `.48` / std `.28`归一化，padding保持normalized0。

稳定key至少含train seed、epoch/成功更新位置、sample/group identity与batch slot，不含strategy/model遍历顺序。同位置先生成Replay可行删除量，再按随机网格块顺序找到Grid近似匹配。网格cell沿旧思想 `max(8,min(H,W)//16)`，不直接调用固定severity `.75`。

- 保留25%clean槽位。
- Replay新增删除率在当前有效Depth点中为 `[0.10,0.50]`。
- Grid/Replay绝对新增删除率误差不超过 `.02`。
- 最多8个源候选；任一不可行/超容差，Grid与Replay共同skip该槽位。
- `|V|=0`保留原输入并skip；不为降低skip率增加复杂mask优化器。
- 输出source sample/group、attempts、target/replay/grid新增数、matching error、skip reason。Natural保持原输入，Grid/Replay共享相同计划，不读取对方模型结果。

## 6. 新S1评价身份

S1为original scale1、noflip、whole image、batch1、右/下normalized0 pad到32倍数、FP32 inference、TF32 off；forward后crop掉padding，logits用统一插值回原Label网格。只支持Natural原始输入、连续矩形压力、entire-missing三条件；不称作旧Quick-Val，不实现S2/S10。

同sample×condition只生成一次raw observation，所有checkpoint复用；每个checkpoint forward前从稳定sample/condition key恢复同推理RNG起点，配对HAM/NMF随机基底，不依赖checkpoint排序/字典/文件系统顺序。随机性配对不保证跨硬件逐位一致。

矩形不使用Label定位，目标删除当前原有效点的50%，计数四舍五入到整数。先按全图有效密度和原图长宽比确定一个固定H×W矩形，再用积分计数穷举该固定几何的合法平移位置，选择删除量最接近目标的位置，以sample-key打破并列。实际比例/误差/固定几何必须记录。“不可达”仅指该声明H×W几何的所有平移不能达到精确计数，不宣称其他尺寸矩形也不可达；保留最接近位置，不另叠第二矩形、不按分数筛压力样本。

先累积dataset confusion matrix再用公共mIoU实现，不平均逐图mIoU；所有候选统一15类，沿公共实现把union为0的类IoU记0并纳入15类均值。高/低分层严格沿原始全图未padding Depth缺失率及已冻结train阈值，不因Label改变成员。有效标签区域中的缺失率仅是附加解释统计，不用于重新分层。分别按condition×采集组×all/low/high累计混淆矩阵，记录类别真实像素支持与组计数，不能用剔除困难类别抬分。无真实标签支持类政策运行前固定；压力条件不作为自然数据收益证据。

## 7. 正式停止线与本轮合法终点

沿A正文§6.5：Replay自然高缺失比C0/Natural/Grid最强者至少+1.0pp；全体/低缺失各不劣于最强者超过0.3pp；矩形比Natural/Grid强者至少+0.5pp；entire-missing相对Grid不下降超过0.5pp；删除量、类别支持与采集组解释合格。任一关键条件不满足STOP，不增加模块/预算挽救。门槛是资源筛选线，不是统计显著性。

本轮最多READY / PARTIALLY READY / BLOCKED工程结论，不产出上述科学GO/STOP。初始实现阶段未运行正式7680更新、完整3816 view评价、B1a、NYUv2/外部baseline或official test；后续限量GPU资源权限仅按§8，不改变上述科研合同。

## 8. 2026-10-03 已批准4090限量云端预检

只启动现有`cpod-1vbh7faqcauq`有卡模式，不新建或改变付费规格。Natural/Grid/Replay各独立从原精确C0开始，最多3成功更新、Linux workers8，其余合同不变；C0沿用先前核验身份/原远端文件，不重复hash。预检前先提交必要代码，用户明确批准已有3提交和本轮必要提交推送origin同分支；云端拉取且HEAD一致/相关源码干净后才能运行。

启动前优先设置并核验30分钟平台关机保险；仅支持运行态时立即在启动后设置，未成功前不提交训练，失败立即停机，不自动延长。SwanLab有条件时复用已有配置记录整体进度和成功/失败，监控与训练子进程隔离。持久任务顺序执行；OOM/非有限/身份不符或需改变合同就停，不增加尝试/改batch/precision挽救，不自动进入正式阶段。

正常结束立即主动stop并直接确认Stopped，不等用户20分钟保险。若发现用户因完成/中断后无下一步操作已人工停机，记录并退出，不自动重启；正常结束后仍需小型证据取回时可短时CPU-only、取完停机。只在大阶段开始/结束/中断更新实时文档，小动作不写流水账。完整测试、正式训练/全集评价和official test仍关闭。

## 9. 已执行的限量预检工程资格（2026-10-03）

云端干净运行提交18271ad0b0c3e7ba81c630c099d02592c02a45f4，现有RTX4090 24564MiB、workers8，科研合同不变。Natural/Grid/Replay各attempted3/successful3/skipped0，finite loss/gradient、optimizer/scaler和保存链通过，三个preflight-only checkpoint云端CPU-only读回计数/状态/身份通过。工程结论READY_ENGINEERING_ONLY；没有科研分数/GO/STOP、真实GPU resume或长期稳定证据，预检权重不作正式初始化/正式resume。

平台30分钟保险启动前/后已核验，任务正常结束主动停GPU；必要小型CPU-only取回后最终09:46:59+08:00直接确认Stopped/GPU0。SwanLab初始化RuntimeError，本轮LOG_ONLY且无在线链接；不把监控上传记通过。精确结果、计时边界和收据见[同一readiness报告§13](../../doc/reports/2026-10-03-natural-missing-round1-readiness.md#13-已批准4090限量云端预检真实执行收口)。三组各3步预算已用完，正式预算仍待独立授权，不重复预检或C0 hash。

## 10. 2026-10-03 Direction A正式首轮最新授权

用户通过本地`临时/2026-10-03-mmfr-direction-a-round1-formal-execution.md`正式执行指令批准一次完整首轮；本节为其持久授权记录，不改变§1–7科研内容或canonical Direction A §6.5门槛。

- 必要正式编排/logging与文档选择性提交，push到origin `perf/mmfr-a2-v3-pipeline-opt1`；已有无关dirty保留，不reset。云端只轻量fast-forward，正式运行HEAD等于本地formal-run SHA，运行源码无冲突tracked dirty；不使用旧snapshot。
- 只启动现有`cpod-1vbh7faqcauq` RTX4090，不创建/resize；启动前设置并确认**6小时平台自动关机保险**，Running后复核同截止。接近截止优先合法保存并停机，不无限延长。
- Natural→Grid→Replay各从同一精确原C0独立weights-only初始化，2560 successful updates/group，attempted=successful、optimizer skip0。禁止preflight权重、组间续训、自动改变合同或重试；每640 recovery与固定final。
- 全部三组合法完成后唯一一次四权重×318图×3条件完整S1（3816 views）；按原冻结门槛GO/STOP，工程/输入/身份失效写INVALID/BLOCKED，不能用无效分数裁决方法。训练与S1均使用既有原合同入口，不启用旧Quick/Main/S2/S10。
- SwanLab先按既有登录/环境/项目设置做有限排障，不泄露token或重建依赖；ONLINE成功则使用，合理尝试失败记录LOG_ONLY并继续。screen持久终端`natural-missing-round1-formal`，保存阶段/计数/loss/LR/time/memory/checkpoint及轻量输入遥测，不保留大mask cache。
- GPU任务正常结束或失败后主动stop并直接确认Stopped/GPU0，不为下载保留GPU。正常完成后同实例CPU-only取回三个正式final、完整S1/confusion/summary、receipt、training/input summary和必要日志，启动前20–30分钟保险；取完立即停机并确认Stopped/GPU0。本地CPU schema/身份与权重hash核验不增加模型forward或训练。
- 最终报告`doc/reports/2026-10-03-natural-missing-round1-formal.md`，同步两份实时入口与必要导航/索引，按执行指令必要收口提交push。checkpoint/outputs/大日志/cache不入Git，历史原始证据不改写。
- Direction B/B1a、Main-Val、NYUv2、外部baseline、official test、新seed/调参/救结果均未授权；本轮结束即停止，GO也只请求下一阶段授权。

**大白话：** 获批的是一轮完整且预先定好门槛的对照实验；监控故障允许降级，科研训练或身份错误必须停，结果不够好就结束候选，不继续修模块。

## 11. 零更新启动失败后的原预算继续授权（2026-10-03）

首个正式编排提交`8d2490ef433807ec60d475cab27fbadd04d18e1f`遗漏训练入口必需的`--successful-updates 2560`，Natural在参数门禁退出，attempted0/successful0；其余组及S1未运行。失败证据已取回，最终直接确认实例Stopped/GPU0。该次属于工程INVALID，不能给科研GO/STOP。

用户随后明确选择`continue_original_budget`：允许修正调用参数与停机监督判据，必要选择性提交push后用新SHA、新输出根执行尚未消耗的原三组2560预算与完整S1。首次失败原始证据保留在`outputs/natural-missing-round1-formal-20261003/`，不得覆盖。GPU平台保险仍为原绝对截止**2026-10-04 02:11:47+08:00 / Unix1791051107**，重新启动前及Running后必须核验；不得重新增加六小时。其余科学合同、停止规则与关闭边界不变，没有授权自动重试或追加预算。

**大白话：** 只修复尚未进入训练的启动命令，继续原实验；第一次失败不算训练结果，也不换来额外时间。

## 12. 已执行正式运行的非有限停止与当前边界（2026-10-03）

正式源码edb660a83da1ec67626149d06dc59460565e2c20，云端同HEAD/无tracked dirty。Natural attempted1598/successful1597/已记录optimizer skip0，第1598次在梯度有限性检查、optimizer执行前退出；Grid/Replay和完整S1未运行。按§3及§10停止规则，本次INVALID/BLOCKED，不能科研GO/STOP或继续剩余更新；没有改变AMP/NMF/LR/输入/预算。监督器主动停GPU，失败证据CPU-only取回后23:11:39直接Stopped/GPU0。运行、计时与证据详见[正式报告](../../doc/reports/2026-10-03-natural-missing-round1-formal.md)。

原正式继续授权已执行并因非有限停止，不保留自动重试权限。任何数值定位GPU操作、合同修订、resume/重训或新预算均须独立授权；当前只完成报告/状态/必要索引与Git收口。**大白话：** 基础组没能合法完成，先决定是否单独查数值问题，不能直接续跑或拿不完整结果评价Replay。

## 13. 2026-10-04 数值定位、最小修复与重新正式首轮最新授权

用户在本对话明确批准顺序：**Natural短数值定位 → 证据驱动最小数值修复 → 短资格 → Natural/Grid/Replay重新正式训练 → 完整S1 → Direction A冻结gate裁决**。本节替代§12中“待独立授权”的权限边界，不追改旧失败事实或§1–7科学定义。当前仍INVALID/BLOCKED，不是科研STOP。

- 先基于现有报告/源码增加必要non-finite日志：首次非有限参数/层/算子、loss/gradient/AMP scale、attempt/successful、batch/sample身份、NMF前后及关键中间tensor。诊断阶段模型结构、数据、LR、正式预算和S1口径不变。
- 本地选择性commit/push至当前origin分支，云端Git fast-forward；每次GPU运行前直接确认HEAD等于已push commit且运行源码tracked-clean，禁止不同版本复制运行。无关dirty保留，checkpoint/outputs/大日志不入Git。
- 只启动现有RTX4090实例 `cpod-1vbh7faqcauq`，沿用原CUDA/PyTorch环境，不重建/无意义升级依赖。每个当前任务有平台自动定时关机保险与少量收尾余量，训练前直接核验；保留screen与日志，结束立即关GPU，取回只用CPU-only/无卡及短时保险，完成后关机。
- SwanLab优先有限排查已有登录/配置/API key来源与非交互初始化，不泄露token，不大改环境。合理尝试仍失败则记录步骤/原因/错误，LOG_ONLY继续，不长期阻塞科研。
- 数值定位只Natural从原C0干净起跑，沿原2560-update scheduler，诊断cap1664成功更新，细粒度观测窗口默认从attempt1536；不追求完整2560，不保存正式合格checkpoint或resume旧Natural1597。任何非有限停止，不跳坏update或吞异常。若cap内仍无异常，停止报告未复现，不擅自增加诊断预算。
- 优先区分AMP/scaler、NMF、异常输入及首次tensor/layer/operator。只有直接诊断证据支持后才做最小数值修复（例如被证据确认的NMF局部FP32）；算法、随机基底、迭代数和研究模块不变，三组统一数值策略。
- 修复再次commit → push → 云端git pull/fast-forward → 同HEAD直接核验，先短资格确认finite再正式。局部数值策略若改变，明确记录精度修订与资格证据，不能把旧原生autocast协议冒充实际修订策略。
- 新正式必须从原冻结C0分别干净初始化，Natural → Grid → Replay，各2560 successful updates，attempted=successful、optimizer skip0，必要final/schema/hash/完成状态验收；不续跑旧1597或组间权重。任一组再次资格性故障立即停止后续流程并报告，不自行改合同。
- 三组全部合法完成才运行冻结完整S1四权重×318×3=3816 views；沿canonical §6.5原阈值GO / 科研STOP / 工程INVALID/BLOCKED，结果差不临时调门槛。Main-Val、official test、NYUv2、其他baseline、Round-2、Direction B/B1a均关闭。

**大白话：** 先用有限额外算力查清坏梯度，只有修复有证据且资格通过才重新做完整对照；旧失败不能接着算正式结果，下载文件也不能继续占GPU。

## 14. 2026-10-04 已复现NMF反向非有限与最小FP32数值修订

诊断运行源码 `1cf9a8af517a1cf060846461cd31be5aba667950`，从冻结C0独立启动Natural，cap1664/观测起点1536。实际attempt1598/successful1597/skip0，失败loss0.18105733394622803 finite，scale1024、backward未完成、unscale未开始；输入及已观察前向均finite。首个观测坏中间梯度 `decode_head.hamburger.ham.iteration.2.bases.divide` 为39 Inf，异常反向trace定位NMF基底更新的分母batch-matmul `BmmBackward0`产生NaN。具体证据在 `outputs/natural-missing-numerical-20261004-1cf9a8a/monitor/numerical-run-receipt.json` 和 `Natural.stderr.log`。这支持AMP下NMF反向数值不稳定，不是输入NaN/Inf污染或已证明的scaler增长/解缩放bug；不能将未完成反向中的partial参数finite摘要记成完整通过。

按§13已授权，仅修订**训练NMF局部精度为FP32**：`nmf_training_precision=fp32_local`，共享NaturalMissing builder为原NMF2D设置训练flag，CUDA训练autocast中仅NMF范围关闭autocast并输入float32。Natural/Grid/Replay统一使用此策略；§3原生NMF autocast是修订前历史合同，不冒充修订后的实际精度。外围AMP/scaler1024、NMF算法/六次训练迭代/随机基底、模型结构/参数、数据/顺序/普通增强、LR/scheduler、三组各2560预算及S1/科研门槛全不变；不启用旧A-v1。

修复完成必要commit/push、云端fast-forward和同HEAD核验后，先Natural从原C0限定资格至1664，确认越过已复现1598区间且attempted=successful、skip0、loss/gradient全程finite。资格为diagnosis-only，无正式checkpoint/正式resume资格，不继承诊断或旧Natural1597权重；失败立即停止，未通过不得进入三组正式。资格当前尚未运行，局部修复不自动等于稳定性已获证明。GPU任务结束立即关闭，证据只CPU-only取回；本诊断取回后01:33:29已直接确认Stopped/GPU0。

**大白话：** 只把已经出错的矩阵分解计算改用更稳的数值精度，其他研究变量不动；先验证能通过原失败区间，再重新做完整对照，不能把诊断的1597更新接成正式结果。
