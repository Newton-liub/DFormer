# Natural Missing Round-1 执行合同

> **文档角色：** Direction A 首轮执行合同，不替代科研方案或实时权限。
> **形成/核验时点：** 2026-10-03。
> **实时入口：** [当前状态](../../doc/main/MUSeg-current-status.md)；[开放决策](../../doc/main/MUSeg-open-decisions.md)。
> **科学依据：** [Direction A canonical 正文](MMFR_direction_A_natural_missing_2026-10-02.md)，§5–6；[接入审计](../../doc/reports/2026-10-02-direction-plans-project-readiness-upper-review.md)与[目录审计](../../doc/reports/2026-10-03-project-directory-responsibility-audit.md)。
> **权限：** 本轮只实现与就绪检查。精确C0和本机现有GPU满足条件时，每组最多3成功更新、极少量val-dev预检；正式三组训练和完整评价仍需下一轮授权。

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

本轮最多READY / PARTIALLY READY / BLOCKED工程结论，不产出上述科学GO/STOP。完整测试、正式7680更新、完整3816 view评价、B1a、NYUv2/外部baseline、official test与付费云均不运行。精确C0/本机资源阻塞不阻止纯代码与CPU准备，但不得宣称真实训练链通过。最终只提交就绪报告、真实检查与恢复点后停止。
