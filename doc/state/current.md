# DFormer 项目实时状态

> 事实时间：2026-10-10。本文件是当前事实、授权边界与恢复点的唯一入口。ODG实现/验收证据见[无卡准备记录](../reports/2026-10-10-odg-no-gpu-preparation.md)，固定协议与命令见[ODG入口](../guides/odg.md)。已结束论文库阶段摘要见[历史状态](../archive/2026-10-10-odg-start/paper-library-state.md)。

## 当前阶段

用户已批准[首轮GPU实验计划](../../临时/ODG_GPU_实验计划_20261010.md)，并确认本地SUN包来自DFormer作者入口；当前进入首轮GPU实验阶段。**ODG实现、三模式构建、CPU梯度验收、独立训练/评价入口、SUN云端数据准备均已完成，主无卡窗口此前已回读Stopped。** 本阶段预算上限为¥40且20 GPU小时：最小GPU检查后按计划串行训练original与odg各30 epoch（300 epoch日程与两组条件不变）。有卡启动由用户手动执行；实例`cpod-1vbh7faqcauq`现为Stopped、GPU0、未计费。尚无任何GPU实测吞吐、显存或分割指标；30 epoch只作工程与趋势判断，不宣称方法有效。

## 已确认事实

### 研究合同与实现

- 主数据集SUN RGB-D，NYUv2第二验证集；MUSeg仅可选应用扩展，DeLiVER本轮不适配。DFormerv2-S + 当前HAM宽度1024、37类，只实施ODG，不叠加教师/补偿/新损失。
- `research/geometry.py`构造输入观测16区间软分布→实际stage面积池化→归一化核log bias；固定区间覆盖作者归一化深度域。空观测只中性化depth项，原spatial/QKV/RoPE/FFN/可学习权重符号保留。无新增可学习参数；前三stage轴向、最后full，同forward复用stage关系，不跨batch缓存。
- `geometry_mode=original|mean|odg`：original复用作者双线性stage深度差；mean先求同支持域均值再走相同分箱核，是正结果后的关键消融。区间中心点质量可退化，off-grid仅近似，不声称已证明创新或收益。
- 研究层为`research/{data,train_odg,evaluate_odg,odg_schedule,prepare_odg}.py`；独立配置`local_configs.research.ODG_SUNRGBD`及`ODG_NYUv2`。作者`utils/train.py`未改；模型只改必要接口、几何切换、单卡BN与HAM设备分配。
- 新baseline/candidate都只加载`checkpoints/pretrained/DFormerv2_Small_pretrained.pth`，新头、新optimizer/scaler，从头训练；不使用旧MUSeg/MMFR完整分割权重。所有可训练层更新，单卡关闭SyncBN且`norm_eval=False`。
- SUN：AdamW8e-5、wd.01、betas.9/.999、有效batch16、480×480、seed12345、warmup10、poly.9、总300epoch，继承尺度/翻转，不新增损坏增强、不compile。micro-batch/累积须GPU授权后确认并在两模型一致。
- LR在optimizer尝试之前写入真实param-group；AMP成功/跳过分别计数。`stop_after_epoch=30/100`只是暂停，不压缩300epoch。保存last、best-dev、30/100阶段点与optimizer/scaler/RNG/合同；resume限同实验epoch边界，smoke快照不可恢复正式训练。

### 数据、开发验证与CPU验收

- 本地`D:\0Project\dataset\SUNRGBD`：RGB/Depth/labels各10335，官方train5285/test5050。路径配对0缺失。固定图像级seed12345划分4757train-dev/528dev，清单在`research/splits/sunrgbd_seed12345/`，互斥且并集等于官方train；原train/test不改，不声称场景隔离。
- 周期验证只读dev，每10epoch、单尺度无flip。显式fulltrain用官方全train并关闭周期验证；正式test仅独立入口显式选择，不反复调参。五尺度+flip只在指定节点显式评价，数字不得与单尺度混比。
- 开发训练298次optimizer尝试/epoch，补11张到4768；总89400、warmup2980。全train331次/epoch。原始/ODG从同encoder预训练共同起步；旧MUSeg和论文分数不替代匹配SUN baseline。
- 灰度读取保留作者8位读法及`(D8/255-.48)/.28`；uint16 PNG被读为高8位，不直接除255。天然0保留为观测，支持域排除padding/非有限值/人工删除；不能用normalized tensor==0推断物理缺失。
- `D:\0Project\dataset\SUNRGBD.zip`已存在（2452204576bytes，SUNRGBD单顶层）；3个depth成员和两份清单与落盘大小/CRC相同。**尚无直接绑定作者下载源的记录**；米制编码仍未知，正式GPU前须确认来源或替换为明确作者整理包，不静默发明映射。
- 三模式各26966591参数，均成功加载官方encoder。missing为extra_norms标准LayerNorm初值，unexpected为不用的预训练头；没有补key、没有重哈希权重。
- 小张量常量/自关系/全空/部分空、轴向/full尺寸与有限性、FP32 autocast隔离、分块一致性均通过。ODG一次CPU batch2、64×64完整前后向loss4.1066256，714个可训练参数张量均有有限梯度。实际没有GPU或正式训练验收。
- 人工25%/50%孔洞会同时填共同depth输入与删mask，按矩形图像支持域精确取整计数；padding不改。非矩形支持域先拒绝并要求单独协议，不以错误严重度评价。五张数据示例在`outputs/odg-preparation/data-preview/`，只是输入对齐检查，不是收益图。
- NYUv2数据本地/云端均缺，已准备作者40类入口，不把test伪作dev；不阻塞SUN。本轮未下载NYU或适配DeLiVER。可信整理版入口和目标结构见准备报告。

### Git、环境、监控与云端

- 作者基线仍为`e3273009b759b578945483828ff315d560be94c9`。任务从实际HEAD`08d5940`建立`research/odg-sunrgbd`。核心`71aa4e4`、训练入口`2b2d754`、启动脚本修正`4f3e405`均已推送用户origin；upstream push继续禁用。主体实现提交`4f3e40529399aff91d7af9fc9eb2450a1d752498`已在两端；收尾提交包含状态/报告和数据检查断言/同步临时文件清理入口，其身份用`git rev-parse HEAD`获取，不在文件内自指尚未生成的hash。
- 本机继续用`D:\2Env\anaconda\envs\dformer`：Python3.10.22、torch2.7.0+cu128、mmcv1.7.2、timm1.0.30、SwanLab0.10.1；旧df2冻结，未升级依赖栈。
- 当前会话未发现CompShare MCP工具，已改用既有官方CLI0.3.6与配置，没要求新密钥。实例`cpod-1vbh7faqcauq`、cn-bj2-03、Postpay、50GB系统盘；启动前Stopped且无其它screen/训练任务。
- 本轮显式无卡A启动，运行期间平台回读2CPU/4GiB/GPU0，`InstancePrice=0.13`，不是免费状态。保险为2026-10-10 02:44:43（UTC+8）。主窗口已实际主动停止并回读`Stopped / GPU0`；收尾短窗口只同步Git/清理本轮传输临时文件，仍明确A档、保留保险并主动停止，不把任务退出写成资源停止。
- 云端工程`/root/rivermind-data/DFormer`，复用`/usr/local/miniconda3/envs/py310`：Python3.10.16、torch2.1.2+cu118、mmcv2.1.0、timm1.0.28、SwanLab0.10.1，官方pretrained已存在110203103bytes。没有开GPU或在4GB机器模拟全尺寸训练。
- 云端GitHub fetch发生TLS握手中止，已按许可用Git bundle fetch/fast-forward同步正常提交关系，不手工覆盖源码、不留云端未记录补丁。手动启动脚本`bash -n`及CPU日程已通过；未创建训练session、cron/start.d或将来自动GPU任务。
- SUN现成zip已上传至`/root/rivermind-data/dataset/SUNRGBD-odg-20261010.zip`并解压到相邻`SUNRGBD/`，原包保留。三模态各10335、train5285/test5050；云端四清单路径配对0缺失、少量train/dev样本CPU可读及padding通过，磁盘空闲约20GiB。8MiB短传5.513秒、保守预计26.86分钟才启动；实际1089.322秒（18分09秒），低于3小时。编码来源仍待确认，不把工程可读性冒称作者来源已闭环。
- 本地在线SwanLab init/log/finish实际上传完成16records：[CPU准备run](https://swanlab.cn/@Newton_liub/dformer-research/runs/bwmgfa13)。GBK错误后只针对性改`-X utf8`重试一次成功。云端登录存储存在，disabled和offline生命周期成功；云端online本轮未新测。key未打印、纳入Git或配置，未直接读取`临时/key.txt`。
- 旧云端工程`/root/rivermind-data/DFormer-archive-20261008`、旧cloud结果/MUSeg/pretrained库，以及本地`D:\0Project\DFormer-archive-20261007\`均保留，不迁回或覆盖。旧实验结论仍以[交接报告](../../临时/RESEARCH_HANDOFF.md)为准。

### 已结束的论文库工作

- 2026-10-09已批准保存并验收：schema4/revision18、49篇、核心15、supplement21；三范围AI导出和回执就绪。本轮仅定点读取既有DFormerv2条目，不保存索引，不继续全文阅读/搜索/PDF下载或工具开发。
- 索引唯一真源仍在`D:\0Project\origin\_index\`，论文原资产在`D:\0Project\origin\论文\`；人工字段与原资产只读边界不变。旧revision17/F0审核计划均已失效，不能重复apply；保存回执和历史授权见[归档摘要](../archive/2026-10-10-odg-start/paper-library-state.md)。
- DFormer++附录等旧未决项不阻塞本轮，不自动扩展。外部索引未纳入版本控制的维护事项继续暂不处理。

## 本轮授权与边界

- 2026-10-10指令授权本任务代码/配置/简短文档、限定CPU检查、现有实例无卡启动和关停、数据传输/必要下载、一次最小在线日志验证、Git提交及推送用户远端。此授权替代已结束论文库任务的工程限制，但没有扩展论文库写入权。
- **GPU阶段授权范围（2026-10-10用户批准）：** 现有实例有卡运行、最小GPU检查、original/与odg各30 epoch正式训练、必要的最小代码修复与提交；预算上限¥40且20 GPU小时。续至100 epoch、正式test、300 epoch与消融仍需另行授权。
- 有卡启动、付费规格切换与云资源生命周期由用户手动执行或明确授权；不新建实例、不扩容、不删除磁盘/实例、不覆盖旧实验。GPU检查通过前不开始长训练；训练前必须设置并回读平台关机保险。
- 不跑完整旧测试、全量模型推理、长CPU训练或权重全量重哈希。异常与未运行检查必须如实记录；正式GPU能力不能由CPU验收替代。
- 验证范围曾有偏差：首次CPU检查遇到作者硬编码CUDA基向量分配后在设备一致性检查处终止，已修复并屏蔽CUDA重检；数据整理曾全清单头部尺寸扫描（15620条含重复），超出约5组抽样，已停止并移除全量入口。细节见报告，不隐瞒或继续扩展。

## 下一步、阻塞与恢复点

- 2026-10-10首轮GPU实验计划已获用户批准（[临时计划](../../临时/ODG_GPU_实验计划_20261010.md)），SUN包来源已由用户确认为作者入口。第一步为最小GPU检查，随后original/odg各30 epoch；当前等待用户手动启动有卡实例。

1. SUN传输与云端无卡验收已完成，原zip与本地来源保留，旧云端资产不覆盖；不继续下载或转换。
2. 有卡启动后先回读规格并设置+回读平台关机保险（约90分钟），再从`7abf51d`同一commit执行`bash research/run_odg_gpu.sh smoke`（起点`MICRO_BATCH=2 ACCUM_STEPS=8`，两组同组合），确认loss有限、存在实际成功更新、无显存溢出并记录峰值显存与稳定吞吐；随后串行`baseline`、`odg`各30 epoch。
3. 预算上限¥40且20 GPU小时；实际吞吐明显超出预期或任一上限触顶即暂停报告，不自行扩预算。100 epoch续训、正式test、300 epoch与消融均需另行授权。
4. 本阶段仍待实测：micro-batch显存余量、AMP稳定更新、吞吐与时长、resume后设备一致性。NYU缺失不是SUN阻塞。
5. 没有新科研裁决或匹配SUN baseline分数。30epoch只看工程/趋势；100epoch后才按上级预算作实质判断；有正结果再mean消融、正式fulltrain/test和NYU验证。所有test结果与开发/全train训练口径分开记账。
