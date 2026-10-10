# DFormer 项目实时状态

> 事实时间：2026-10-10。本文件是当前事实、授权边界与恢复点的唯一入口。ODG实现/验收证据见[无卡准备记录](../reports/2026-10-10-odg-no-gpu-preparation.md)，首轮30 epoch结果见[30epoch报告](../reports/2026-10-10-odg-sunrgbd-30epoch.md)，固定协议与命令见[ODG入口](../guides/odg.md)。已结束论文库阶段摘要见[历史状态](../archive/2026-10-10-odg-start/paper-library-state.md)。

## 当前阶段

用户已批准[首轮GPU实验计划](../plans/2026-10-10-odg-gpu-experiment.md)并确认SUN包来自作者入口。**2026-10-10首轮GPU实验已完成：original与odg各30 epoch（300 epoch日程的暂停点）在云端并行跑完，实例已主动停止并回读`Stopped`，产物已在无卡A窗口取回本地。** dev单尺度无翻转mIoU：original 26.01/36.82/**39.39**、odg 24.92/34.83/**36.97**（epoch 10/20/30），ODG落后1.09/1.99/2.42点且差距扩大。**这三个验证点是在 train 模式下取得的（见下一段），不是标准 eval 结果**；上级已裁决该负向趋势不再作为淘汰依据。按计划30epoch停止条件，**不自动续训100epoch**；正式test、mean消融、300epoch均未授权、未运行。

2026-10-10[科研诊断报告](../reports/2026-10-10-odg-research-diagnosis.md)已获上级审核通过。CPU直接核验发现两组29个`Geo.weight`均遗漏于优化器（714个可训练参数张量，仅685个入组），30epoch与预训练逐元素相同；此前“几何权重已更新/所有可训练层更新”的描述已更正。三张dev深度关系抽样显示ODG bias较原关系减弱，天然零值语义仍未知。

**2026-10-10评价配置对齐与孔洞复测已完成，等待上级科研裁决**（[复测报告](../reports/2026-10-10-odg-bn-align-reevaluation.md)；此前未对齐的孔洞结果见[孔洞评价](../reports/2026-10-10-odg-holes25-evaluation.md)）：独立评价入口已恢复训练时的解码器 BatchNorm 数值（`eps=0.001`，CPU 三路径定点核对，792 个浮点张量不变），但四组结果只变化 ≤0.01 点——**`eps` 不是 clean 与训练内不一致的原因**。修复后 clean 为 Original **41.32**、ODG **42.77**，**仍不能复现**训练内 39.39/36.97；定点诊断（同权重、同 528 图、仅令模型保持 train 模式）得 39.09/37.22，并复现训练记录中 `floor_mat`/`shower_curtain`/`night_stand` 三类 IoU 为 0 的特征，证实 `research/train_odg.py` 的周期验证从未调用 `model.eval()`（作者的 `utils/train.py` 在验证前调用了）。25% 孔洞下 ODG 仍 **+2.96** 点、退化改善 1.51 点，正向信号保留；上级已据此裁决 ODG 具有继续研究价值，“ODG 落后 2.42 点”的 train 模式趋势不再作为淘汰依据。

**2026-10-10 训练代码修复与第二轮准备已完成（本地，未启动 GPU）。** 周期验证改为 eval 模式（验证后恢复训练模式，验证前后比对 BatchNorm running 统计量指纹，发生变化即报错）；优化器参数分组改由研究层实现，714 个可训练参数张量恰好各入组一次（decay 302 / no_decay 412，含此前遗漏的 29 个 `GeoPriorGen.weight`），缺失或重复即报错；训练合同新增 `validation_mode=eval` 与 `optimizer_param_scope=all_trainable`，使修复前的 checkpoint 无法被新代码 resume。ODG 设计、超参、数据划分、300 epoch 日程均未改。[第二轮重训计划](../plans/2026-10-10-odg-retrain-30epoch-r2.md)已就绪，等待用户授权 GPU。


### 首轮30 epoch结果与执行事实（2026-10-10，详见报告）

- 组合`MICRO_BATCH=4 ACCUM_STEPS=4`（有效batch16），两组同组合、同合同（`contract.json`仅`geometry_mode`不同）；两组**并行**在同一4090上运行，GPU利用87–99%，比串行方案省约1.4–1.6小时。
- 指标与逐类差异、loss趋势、性能诊断短测（`4×4`15.82、`8×2`22.11、`16×1`OOM未测、并行`4×4`合计24.46 samples/s）见[30epoch报告](../reports/2026-10-10-odg-sunrgbd-30epoch.md)。
- 费用：有卡会话2026-10-10 09:04:34–13:23:02，4.31 GPU小时、约¥8.1（上限¥40/20小时未触及）。两组各8940次尝试、8932次应用、8次AMP预热跳步，无失败记录。
- 性能瓶颈定位：数据加载与CPU均未饱和，瓶颈是每次迭代约0.116s加每次optimizer尝试约0.5s的固定开销；未改任何代码。

## 已确认事实

### 研究合同与实现

- 主数据集SUN RGB-D，NYUv2第二验证集；MUSeg仅可选应用扩展，DeLiVER本轮不适配。DFormerv2-S + 当前HAM宽度1024、37类，只实施ODG，不叠加教师/补偿/新损失。
- `research/geometry.py`构造输入观测16区间软分布→实际stage面积池化→归一化核log bias；固定区间覆盖作者归一化深度域。空观测只中性化depth项，原spatial/QKV/RoPE/FFN/可学习权重符号保留。无新增可学习参数；前三stage轴向、最后full，同forward复用stage关系，不跨batch缓存。
- `geometry_mode=original|mean|odg`：original复用作者双线性stage深度差；mean先求同支持域均值再走相同分箱核，是正结果后的关键消融。区间中心点质量可退化，off-grid仅近似，不声称已证明创新或收益。
- 研究层为`research/{data,train_odg,evaluate_odg,odg_schedule,prepare_odg}.py`；独立配置`local_configs.research.ODG_SUNRGBD`及`ODG_NYUv2`。作者`utils/train.py`未改；模型只改必要接口、几何切换、单卡BN与HAM设备分配。
- 新baseline/candidate都只加载`checkpoints/pretrained/DFormerv2_Small_pretrained.pth`，新头、新optimizer/scaler，从头训练；不使用旧MUSeg/MMFR完整分割权重。常规编码器/分割头参数参与优化；**round-1 两组训练时29个`Geo.weight`因作者分组函数遗漏而未进入优化器（2026-10-10诊断直接核验），该问题已在训练入口修复，现要求714个可训练张量恰好各入组一次，缺失或重复即报错**；单卡关闭SyncBN且`norm_eval=False`。
- SUN：AdamW8e-5、wd.01、betas.9/.999、有效batch16、480×480、seed12345、warmup10、poly.9、总300epoch，继承尺度/翻转，不新增损坏增强、不compile。micro-batch/累积须GPU授权后确认并在两模型一致。
- LR在optimizer尝试之前写入真实param-group；AMP成功/跳过分别计数。`stop_after_epoch=30/100`只是暂停，不压缩300epoch。保存last、best-dev、30/100阶段点与optimizer/scaler/RNG/合同；resume限同实验epoch边界，smoke快照不可恢复正式训练。

### 数据、开发验证与CPU验收

- 本地`D:\0Project\dataset\SUNRGBD`：RGB/Depth/labels各10335，官方train5285/test5050。路径配对0缺失。固定图像级seed12345划分4757train-dev/528dev，清单在`research/splits/sunrgbd_seed12345/`，互斥且并集等于官方train；原train/test不改，不声称场景隔离。
- 周期验证只读dev，每10epoch、单尺度无flip。round-1 实际在 train 模式下进行；已修复为 eval 模式（验证后恢复训练模式，验证不得更新 BatchNorm running 统计量）。显式fulltrain用官方全train并关闭周期验证；正式test仅独立入口显式选择，不反复调参。五尺度+flip只在指定节点显式评价，数字不得与单尺度混比。
- 开发训练298次optimizer尝试/epoch，补11张到4768；总89400、warmup2980。全train331次/epoch。原始/ODG从同encoder预训练共同起步；旧MUSeg和论文分数不替代匹配SUN baseline。
- 灰度读取保留作者8位读法及`(D8/255-.48)/.28`；uint16 PNG被读为高8位，不直接除255。天然0保留为观测，支持域排除padding/非有限值/人工删除；不能用normalized tensor==0推断物理缺失。
- `D:\0Project\dataset\SUNRGBD.zip`已存在（2452204576bytes，SUNRGBD单顶层，作者整理格式`RGB/Depth/labels`+`train.txt/test.txt`，三模态各10335）；结构与体积与作者整理版一致，**2026-10-10已由用户确认为作者入口下载**。米制编码仍未知，本轮未发明映射、未改16位读取。
- 三模式各26966591参数，均成功加载官方encoder。missing为extra_norms标准LayerNorm初值，unexpected为不用的预训练头；没有补key、没有重哈希权重。
- 小张量常量/自关系/全空/部分空、轴向/full尺寸与有限性、FP32 autocast隔离、分块一致性均通过。ODG一次CPU batch2、64×64完整前后向loss4.1066256，714个可训练参数张量均有有限梯度。实际没有GPU或正式训练验收。
- 人工25%/50%孔洞会同时填共同depth输入与删mask，按矩形图像支持域精确取整计数；padding不改。非矩形支持域先拒绝并要求单独协议，不以错误严重度评价。五张数据示例在`outputs/odg-preparation/data-preview/`，只是输入对齐检查，不是收益图。
- NYUv2数据本地/云端均缺，已准备作者40类入口，不把test伪作dev；不阻塞SUN。本轮未下载NYU或适配DeLiVER。可信整理版入口和目标结构见准备报告。

### Git、环境、监控与云端

- 作者基线仍为`e3273009b759b578945483828ff315d560be94c9`。任务从实际HEAD`08d5940`建立`research/odg-sunrgbd`。核心`71aa4e4`、训练入口`2b2d754`、启动脚本修正`4f3e405`均已推送用户origin；upstream push继续禁用。主体实现提交`4f3e40529399aff91d7af9fc9eb2450a1d752498`已在两端；收尾提交包含状态/报告和数据检查断言/同步临时文件清理入口，其身份用`git rev-parse HEAD`获取，不在文件内自指尚未生成的hash。
- 本机继续用`D:\2Env\anaconda\envs\dformer`：Python3.10.22、torch2.7.0+cu128、mmcv1.7.2、timm1.0.30、SwanLab0.10.1；旧df2冻结，未升级依赖栈。
- 当前会话未发现CompShare MCP工具，已改用既有官方CLI0.3.6与配置，没要求新密钥。实例`cpod-1vbh7faqcauq`、cn-bj2-03、Postpay、50GB系统盘；启动前Stopped且无其它screen/训练任务。
- 2026-10-10有卡会话（用户手动启动）：09:04:34开始、13:23:02主动停止并回读`Stopped`，平台计15508秒=4.31 GPU小时、约¥8.1。保险先设10:48:43、扩容到16:18:40、按实际进度调整为14:45:00并回读确认。随后以无卡A档（2CPU/4GiB/GPU0、`InstancePrice=0.13`，非免费）短窗取回产物，取回校验后再次主动停止并回读`Stopped`。挂起的`stage1`串行链与其screen、`odgpar` screen、诊断脚本进程均已清理，无遗留训练或自动任务。
- 云端工程`/root/rivermind-data/DFormer`，复用`/usr/local/miniconda3/envs/py310`：Python3.10.16、torch2.1.2+cu118、mmcv2.1.0、timm1.0.28、SwanLab0.10.1，官方pretrained已存在110203103bytes。没有开GPU或在4GB机器模拟全尺寸训练。
- 云端GitHub fetch发生TLS握手中止，已按许可用Git bundle fetch/fast-forward同步正常提交关系，不手工覆盖源码、不留云端未记录补丁。手动启动脚本`bash -n`及CPU日程已通过；未创建训练session、cron/start.d或将来自动GPU任务。
- SUN现成zip已上传至`/root/rivermind-data/dataset/SUNRGBD-odg-20261010.zip`并解压到相邻`SUNRGBD/`，原包保留。三模态各10335、train5285/test5050；云端四清单路径配对0缺失、少量train/dev样本CPU可读及padding通过，磁盘空闲约20GiB。8MiB短传5.513秒、保守预计26.86分钟才启动；实际1089.322秒（18分09秒），低于3小时。编码来源仍待确认，不把工程可读性冒称作者来源已闭环。
- 本地在线SwanLab init/log/finish实际上传完成16records：[CPU准备run](https://swanlab.cn/@Newton_liub/dformer-research/runs/bwmgfa13)。GBK错误后只针对性改`-X utf8`重试一次成功。云端登录存储存在，disabled和offline生命周期成功；云端online本轮未新测。key未打印、纳入Git或配置，未直接读取`临时/key.txt`。
- 旧云端工程`/root/rivermind-data/DFormer-archive-20261008`、旧cloud结果/MUSeg/pretrained库，以及本地`D:\0Project\DFormer-archive-20261007\`均保留，不迁回或覆盖。旧实验结论仍以[交接报告](../reports/2026-10-09-research-handoff.md)为准。

### 已结束的论文库工作

- 2026-10-09已批准保存并验收：schema4/revision18、49篇、核心15、supplement21；三范围AI导出和回执就绪。本轮仅定点读取既有DFormerv2条目，不保存索引，不继续全文阅读/搜索/PDF下载或工具开发。
- 索引唯一真源仍在`D:\0Project\origin\_index\`，论文原资产在`D:\0Project\origin\论文\`；人工字段与原资产只读边界不变。旧revision17/F0审核计划均已失效，不能重复apply；保存回执和历史授权见[归档摘要](../archive/2026-10-10-odg-start/paper-library-state.md)。
- DFormer++附录等旧未决项不阻塞本轮，不自动扩展。外部索引未纳入版本控制的维护事项继续暂不处理。

## 本轮授权与边界

- **DeLiVER独立规划（2026-10-10）：** 已生成[待审接入计划](../plans/2026-10-10-deliver-integration.md)，建议本地原位使用RGB+单通道Depth、独立Dataset/25类配置并复用研究入口；本轮只读代表样本与接口、写计划，未改工程或运行模型/训练。仅此规划任务替代此前“暂不适配”的准备限制；接入实施、GPU与正式实验仍待审核/另行授权，不改变SUN主线诊断及停止续训结论。

- **训练代码修复与第二轮准备（2026-10-10，已执行）：** 上级批准只做本地训练代码修复与准备、暂不启动 GPU：周期验证改为 `model.eval()` 并恢复训练模式；修复优化器参数分组遗漏 29 个 `Geo.weight`；保留现有 ODG 设计、超参数、数据划分与 300 epoch 日程；只做最小 CPU 检查。本轮改动限于 `research/{train_odg,odg_schedule}.py` 与 `research/run_odg_gpu.sh`（运行目录后缀默认 `r2`，改动 1 处 experiment 名），未启动云端 GPU、未重训、未新增消融、未改模型结构与作者代码。

- **评价修复与复测授权（2026-10-10，已执行）：** 上级批准修复独立评价入口未恢复训练时 BatchNorm 数值配置的问题，并用现有两个 epoch30 checkpoint 在本地 RTX 5060 重做 Original/ODG 的 clean 与 25% 孔洞四组；优先确认修复后 clean 能否复现训练内 39.39/36.97，仍不一致时只针对实际差异排查。本轮实现限于研究层 `evaluate_odg.py` / `odg_schedule.py`，另做一次 train 模式定点诊断定位差异（同权重、同 528 图，仅改变模型模式）；未修优化器、未重训、未新增消融、未启动云端 GPU。

- **上一轮孔洞评价授权（2026-10-10，已执行）：** 现有Original/ODG epoch30、同528图dev、同seed/config的25%孔洞评价在本地5060完成；因与历史clean不一致，仅补两次同入口clean基准，未添加故障/seed/消融。禁止新训练、修改模型/优化器；付费云GPU未授权且未启动。该轮的“评价BN配置恢复及重复配对仍需批准”已由本轮修复复测替代；没有提交或推送。
- **已结束诊断任务（2026-10-10）：** 只读科研诊断及CPU定点取证、写报告；资料本地齐备，未使用无卡云取证，未修改代码或启动GPU。优化器修正重验仍需另行授权。

- 2026-10-10指令授权本任务代码/配置/简短文档、限定CPU检查、现有实例无卡启动和关停、数据传输/必要下载、一次最小在线日志验证、Git提交及推送用户远端。此授权替代已结束论文库任务的工程限制，但没有扩展论文库写入权。
- **GPU阶段授权范围（2026-10-10用户批准）：** 现有实例有卡运行、最小GPU检查、original与odg各30 epoch正式训练（用户事后明确批准双模型并行4×4方案）、必要的最小代码修复与提交；预算上限¥40且20 GPU小时。**该额度只用掉4.31小时/约¥8.1。** 续至100 epoch、正式test、300 epoch、mean消融与云端GPU孔洞评价仍需另行授权；本轮本地孔洞评价已按最新授权完成，未经批准不启动100 epoch训练。
- 有卡启动、付费规格切换与云资源生命周期由用户手动执行或明确授权；不新建实例、不扩容、不删除磁盘/实例、不覆盖旧实验。GPU检查通过前不开始长训练；训练前必须设置并回读平台关机保险。
- 不跑完整旧测试、全量模型推理、长CPU训练或权重全量重哈希。异常与未运行检查必须如实记录；正式GPU能力不能由CPU验收替代。
- 验证范围曾有偏差：首次CPU检查遇到作者硬编码CUDA基向量分配后在设备一致性检查处终止，已修复并屏蔽CUDA重检；数据整理曾全清单头部尺寸扫描（15620条含重复），超出约5组抽样，已停止并移除全量入口。细节见报告，不隐瞒或继续扩展。

## 下一步、阻塞与恢复点

- 文档归档（2026-10-10）：`临时/`中4份Markdown已留档至`doc/`：交接报告进`reports/`，数据下载指南进`guides/`，ODG无卡指令与GPU计划进`plans/`；相关链接已更新，历史材料已标注时效与授权边界。临时原件保留供用户自行清理，本次不改变实验事实或授权。

- 2026-10-10首轮GPU实验计划已执行完毕（[归档计划](../plans/2026-10-10-odg-gpu-experiment.md)）：最小GPU检查、限时性能诊断、两组30 epoch均已结束，实例已停机。结果与产物位置见[30epoch报告](../reports/2026-10-10-odg-sunrgbd-30epoch.md)。

1. 云端工程与数据保留不覆盖；旧云端资产、原zip与本地来源均不动，不继续下载或转换。
2. 本地产物：`outputs/sun-dev-original-seed12345/20261010-013324/`与`outputs/sun-dev-odg-seed12345/20261010-020527/`（CSV、逐类IoU、预测、配置、`last.pth`、`best-dev.pth`、swanlab），诊断与训练日志在`outputs/odg-gpu-20261010/`；4个checkpoint已用本地torch2.7.0 CPU实际加载核对字段。云端保留全部文件含未取回的`stage-epoch-30.pth`。
3. 续训100 epoch、正式test、300 epoch、mean消融仍未授权、未运行；30 epoch差距持续扩大（-1.09→-1.99→-2.42）**为 train 模式验证数值，上级已裁决不再作为淘汰依据**。
4. [评价对齐复测](../reports/2026-10-10-odg-bn-align-reevaluation.md)已完成：解码器BN数值已与训练对齐，但对结果影响≤0.01点；修复后clean为41.32/42.77，孔洞为39.78/42.74，ODG孔洞仍+2.96、退化改善1.51点。产物在`outputs/odg-holes25-bnfix-20261010/{original-clean,original-holes25,odg-clean,odg-holes25}/`，诊断证据在其`diagnostic-trainmode.json`。**已确认原因并修复：`research/train_odg.py`周期验证未切eval模式**，故训练内39.39/36.97为train模式数值，同权重eval模式clean为41.32/42.77（ODG领先1.45）。mask处理与分布贡献仍未分离。
5. **训练代码修复与第二轮准备已完成**（[重训计划](../plans/2026-10-10-odg-retrain-30epoch-r2.md)）：验证模式、优化器分组（714/714，decay302/no_decay412）、合同新字段（阻断旧checkpoint resume）、`RUN_TAG=r2` 运行目录均已就绪；CPU 定点检查通过（eval下BN统计量不变、train模式forward会改变统计量、29个`Geo.weight`在一次step后全部更新）。**下一步只需用户授权 GPU**：预计约4.0–4.5 GPU小时、约¥8–8.5（含最小GPU检查与验证开销），随后用对齐评价比较clean与25%孔洞。旧checkpoint保留为历史探索、不续训；未改模型结构/作者代码，未启动云端。
6. 性能诊断结论（若后续续训仍需提速）：数据加载与CPU未饱和；每次迭代约0.116s加每次optimizer尝试约0.5s固定开销是主要限制；可选优化是经审批后削减每次尝试的逐张量梯度有限性扫描（714张量逐个同步，属监测开销，不改变优化与公平性）；`16×1`仅在整卡独占时可行。
