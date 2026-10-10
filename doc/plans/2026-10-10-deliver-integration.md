# DeLiVER RGB-D 接入计划（待上级审核）

日期：2026-10-10。检查时工程 HEAD：`5c082ba`。**本轮仅规划；本计划获批前不实施、不训练。** 审核后由执行模型接入，主力模型依据实际产物最终验收。目标是准备可用的实验入口，不改变 SUN-RGBD 当前研究主线。

## 1. 现状判断

**具备薄层接入条件，尚未接入；深度物理编码未知不阻塞按固定图像表示读取，但限制科研解释与公开分数比较。**

### 本地数据

数据保留在 `D:\0Project\dataset\DELIVER`。本轮实际打开下列同一训练样本的 RGB、Depth、语义标签和 HHA（水平视差、离地高度、法向角的三通道编码），确认模态目录、层级和文件名配对规则，视觉上场景结构对应：

- `img/cloud/train/MAP_1_point102/110100_rgb_front.png`
- `depth/cloud/train/MAP_1_point102/110100_depth_front.png`
- `semantic/cloud/train/MAP_1_point102/110100_semantic_front.png`
- `hha/cloud/train/MAP_1_point102/110100_depth_front.png`

已有[本地交接记录](../reports/2026-10-09-research-handoff.md)记录结构为 `<modality>/<weather>/<train|val|test>/<scene>/*.png`，天气为 `cloud/fog/night/rain/sun`；RGB、Depth、HHA、semantic 各有 train 3983 / val 2005 / test 1897。该记录的 Depth 样本为单通道 uint8、1–255，四模态样本尺寸为 1042×1042。**计数、原生位深和值域来自已有实测记录，本轮没有重新遍历或数值解码；当前打开代表样本不等于全数据验收。** 执行时生成所需 RGB 清单并抽样确认即可，不重复完整性审计。

本地代表标签呈红通道 ID 图。为确定类别顺序和精确读取规则，本轮只补读了[官方 loader](https://github.com/InSAI-Lab/DELIVER/blob/main/semseg/datasets/deliver.py)：标签读取 `io.read_image(...)[0]`，原始 255 先置 0，再做 uint8 减 1；官方名为 `depth` 的模态实际读取 `hha/`，不是 `depth/`。这是实现事实，不是数据来源调查。

### 现有工程

- SUN-RGBD 已通过 `research/data.py` 的 `ObservationDataset`、同步预处理、独立研究配置及 `research/train_odg.py` / `research/evaluate_odg.py` 接入；其 dev 清单和 SUN 专用 padding 不能直接套到 DeLiVER。
- MUSeg 整理版的 `dataset_meta.json` 本轮已读：原始 uint16 保留，模型使用固定全局量化的 uint8 Depth，标签不在整理阶段重编码。历史接入和 RGB 约定见交接记录；**当前干净仓库没有可直接运行的 MUSeg 研究配置**，不恢复旧研究系统。
- `utils/dataloader/RGBXDataset.py::get_path` 的默认分支丢弃完整层级；现有标签灰度读法也不适合 DeLiVER 红通道 ID。直接换数据根目录或仅改 25 类不可行。
- `research/odg_schedule.py` 的两个 loader factory 固定导入 `research.data`，但模型、累积训练、checkpoint、原生整图评价和指标计算可复用。DFormerv2 只取辅助张量第 0 通道，适合本次固定单通道 Depth 表示，不应把 HHA 灰度化后冒称距离。
- 当前已确认作者参数分组遗漏 29 个 `Geo.weight`。新实验入口必须覆盖这些已有参数；本任务不重写或重启 SUN 实验。

## 2. 推荐方案

### 2.1 固定输入与标签合同

只接入 **RGB + `depth/` 单通道灰度表示**，暂不实现 HHA、LiDAR、Event 或故障合成。

- 路径：清单每行保存相对 `img/`、使用 `/` 分隔的完整路径，含天气、split、scene、视角后缀与扩展名；对应 Depth/semantic 只替换模态根目录和文件名中的 `_rgb`，保留 `_front` 等后缀。使用 `Path` 拼路径，避免 Windows 下字符串替换 `/img` 失败或同名样本冲突。
- RGB：明确解码为 RGB 三通道，按现有 ImageNet mean/std 归一化；新配置和产物记录 `rgb_order=RGB`，不沿用“非 SUN 默认 BGR”的隐式分支。
- Depth：`IMREAD_UNCHANGED` 保留原生位深，抽样符合单通道 uint8 后按 $d=(D_8/255-0.48)/0.28$ 归一化，并复制三通道以满足现有张量接口。预期归一化域约为 $[-1.714286,1.857143]$。不做逐图 min-max、不反转、不套 CARLA 三通道解码、不推算米制单位。若抽样与 uint8 单通道约定不符，先报告该局部格式差异，再做最小适配，不能静默截位或取任意通道。
- 支持域：原图有限像素为 1，padding 为 0；当前未知天然无效值约定，保留 raw 0 为观测，不用 `tensor==0` 推断缺失。以后人工删除必须同时改两组共同输入及显式 mask，另行批准。
- 标签：原生单通道时直接取 ID；三/四通道时取 **RGB 红通道，即 OpenCV BGR/BGRA 的索引 2**。不能 `IMREAD_GRAYSCALE`，不能套调色板颜色匹配。显式映射原始 1–25→0–24，0/255→255；其他 ID 报出具体样本并停止，不静默吞成 ignore。
- 类别顺序（训练 ID 0–24）：`Building, Fence, Other, Pedestrian, Pole, RoadLine, Road, SideWalk, Vegetation, Cars, Wall, TrafficSign, Sky, Ground, Bridge, RailTrack, GroundRail, TrafficLight, Static, Dynamic, Water, Terrain, TwoWheeler, Bus, Truck`。loss 和评价均使用 `ignore_index/background=255`。
- 输出复用 `data/label/modal_x/depth_support/fn/n`；RGB、Depth、support 为 float32，label 为 int64。同步镜像、尺度、裁剪，RGB/Depth 用双线性、标签/support 用最近邻；训练 padding label=255、support=0。

### 2.2 划分、配置与评价

- 保持官方目录 train/val/test，不另切 dev、不合并 val/test。训练只读 train，周期评价只读 val；现有入口的 `--split dev` **在本配置中映射官方 val**，`--split test` 明确映射官方 test。test 只在模型和协议冻结后评价；本次可读一张 test 作格式检查，但不计分。
- 配置名建议 `local_configs.research.DFormerv2_S_DeLiVER`，`dataset_name=DeLiVER`，`data_module=research.deliver`；以 `DFORMER_DATASET_ROOT` 为数据父目录，再拼接 `DELIVER`。配置对象独立复制并明确覆盖数据、类别、输出与 padding 字段，不修改 SUN 配置。
- 模型默认 DFormerv2-S + 作者现有 HAM 分割头（宽度1024）、25 类、`geometry_mode=original`，只用现有官方 encoder pretrained，新建分割头/optimizer，不载 SUN/MUSeg 完整分割 checkpoint。为方便日后同条件比较，保留现有研究入口和可选几何模式，但本次不验收候选方法。
- 工程起始默认沿用 480×480 crop、随机尺度/镜像、AdamW lr8e-5/wd0.01、有效 batch16、seed12345、300 epoch 日程/warmup10、每10 epoch验证；只是可运行默认值，**不是已批准的正式实验合同，也不是官方 DeLiVER 最优配置**。实际 micro-batch 后续按授权硬件选择。
- `pad=False`，不使用 SUN 的 531×730 padding；验证/测试保留原图网格，batch1，默认单尺度无翻转，复用现有整图评价，不新增滑窗。主要指标为 25 类 mIoU（平均类别交并比）和逐类 IoU，辅助输出 mAcc（平均类别准确率）和 mF1（平均类别F1分数）；沿用现有混淆矩阵实现，无 union 的类别记 0 并参与25类平均。小样本检查的指标仅验证流程，不能报告为性能。
- 五尺度+flip入口已有，融合为各视图 softmax 概率求和；现在不运行。正式公开对比前再固定输入表示、分辨率、融合和计分规则；本次 raw-Depth 协议不等同官方 HHA RGB-D 协议。

### 2.3 最小文件范围

1. **新增 `research/deliver.py`**：薄 `DeLiVERDataset`、类别/预览 palette，复用现有 `ObservationTrainPre/ObservationValPre`；对外暴露 `ObservationDataset=DeLiVERDataset` 以匹配 factory。原图只解码一次 Depth，复用该数组构造 support。提供小型 `prepare/preview/smoke`（清单准备／预览／单次模型检查）命令，不另建管理系统或测试文件。
2. **新增 `local_configs/research/DFormerv2_S_DeLiVER.py`**：独立配置；`train_dev_source/full_train_source` 指向官方 train 清单，`dev_eval_source` 指向 val，`full_test_source` 指向 test；实际清单齐备才标记 ready。RGB/Depth/标签协议明确写入 resolved 配置。
3. **小改 `research/odg_schedule.py`**：`import_data_module(config)` 读取可选 `config.data_module`，缺省仍为 `research.data`；两个 loader factory 都显式传入 config。DeLiVER 用现有 setting 键集即可，不泛化 dataset registry。SUN/NYU 调用行为保持原样。
4. **优化器最小保护**：优先复用并行主线已经批准并落地的漏参修正。若届时仍未修复，只在 `research/train_odg.py` 创建 optimizer 前对 `dataset_name==DeLiVER` 调用新模块的补齐函数：按参数对象 ID 收集未入组的 requires-grad 参数、去重，新增这些几何系数的 no-decay 组并记录；断言全部可训练参数恰好入组一次。该条件补齐单独保留小补丁，不改作者 `utils/init_func.py`，不改变 SUN 默认行为；遇到非预期漏参先报告，不扩大修复。
5. **生成 `research/splits/deliver_official/{train,val,test}.txt`**：仅小型相对路径清单，可版本管理；数据与预览不进 Git。若已有同名清单内容不同，先报告，不覆盖并行产物。

除上面的 DeLiVER 条件参数补齐外，训练主体和评价入口直接复用，不复制新 train/eval 脚本，不改模型主干。执行结束只需补一段使用命令和简短验收结果。

### 2.4 论文用途与准备边界

**推荐角色：后续独立训练的室外合成场景扩展验证集，优先检验天气和相机质量下降下的适用边界。** MUSeg 是真实地下矿井、15 类；SUN-RGBD 是真实室内、37 类；DeLiVER 是合成驾驶场景、25 类，距离范围与图像统计也不同。三者无需承担相同实验角色。

- 各数据集分别训练匹配 baseline/候选，可支持“方法在不同场景有效”的结论；**不能把 SUN 的37类模型直接在 DeLiVER 上计25类 IoU并称跨数据集零样本泛化**。真正迁移评价需额外标签空间、训练域和评测协议，现在不做。
- 天气优先 `fog/night/rain`，保留 `sun/cloud` 分项参照。原生相机故障优先 `underexposure/motionblur`，随后 `overexposure`；按官方完整相对路径含关键字筛选，执行时只从生成的 RGB 清单确认非空，不预设新的故障目录。`all` 包含天气和故障，不命名为纯 clean；没有匹配正常样本时，不声称故障子集差异是配对退化。
- `lidarjitter/eventlowres` 主要作用于本次不用的模态，不能充当 Depth 故障证据，也不列入本轮优先验收。天气和相机故障可评价 RGB-D 互补，但不直接证明深度孔洞鲁棒性。
- 现在准备：固定表示、官方划分、加载/训练/评价入口及轻量 case 筛选。以后再定：候选方法、损坏训练、人工 Depth 缺失、消融、正式预算及是否进入论文主表。SUN 当前去留仍按已提交的科研诊断决定，接入不授权续训或转移主线。

## 3. 执行步骤（审核通过后）

1. **隔离修改。** 从届时确认的主线 HEAD 新建本地分支和相邻 Git worktree（同一仓库的独立工作目录），例如 `D:\0Project\DFormer-deliver-worktree`；不 stash/搬走主目录未提交文件，不依赖其临时报告。先确认 worktree 引用的主线源码版本。主要新增文件先做，两个共享文件的小补丁放最后并单独保留；合入由主力模型检查，提交与推送均等待用户确认。无需创建云资源。
2. **实现薄 Dataset 与配置。** 按第2节合同处理完整路径、RGB通道、红通道标签、Depth和同步变换；不在作者 loader 加新的多模态分支。
3. **生成必要清单。** 只枚举 `img/*/<split>/*/*.png` 一遍并排序，写相对路径 train/val/test；同时输出各 split、天气、相机故障的路径计数。预期总量参考3983/2005/1897，差异仅报告，不自行补齐、删除或重划分。这里是加载所需清单生成，不扫描全部模态、不逐张解码或哈希。
4. **接通两个 factory，处理 optimizer 漏参。** 默认分派不变；复用已有修复或启用 DeLiVER 条件补齐。只对本次25类模型核对参数覆盖，不重新核验历史 SUN checkpoint。
5. **按第4节一次完成最小验收。** 数据预览→CPU日程→一份实际样本的小尺寸完整模型检查；发生明确路径/标签/位深错误，只修该问题并重跑受影响检查，不扩为全量审计。
6. **交付复核。** 记录变更、实际命令/结果、少量预览和未运行项；主力模型验收后更新项目状态。不能把 smoke checkpoint 当正式实验起点。正式训练和最终 test 仍需另外明确授权。

拟提供的命令接口如下，**执行模型实现这些接口后再运行；本轮均未执行**。在执行 worktree 根目录使用现有 `dformer` 环境：

```powershell
$env:DFORMER_DATASET_ROOT = 'D:\0Project\dataset'
$env:CUDA_VISIBLE_DEVICES = '-1'
$py = 'D:\2Env\anaconda\envs\dformer\python.exe'
& $py -X utf8 -m research.deliver prepare --root D:\0Project\dataset\DELIVER --split-dir research/splits/deliver_official
& $py -X utf8 -m research.deliver preview --config local_configs.research.DFormerv2_S_DeLiVER --limit 6 --out outputs/deliver-preparation
& $py -X utf8 -m research.train_odg --config local_configs.research.DFormerv2_S_DeLiVER --micro-batch 1 --accum-steps 16 --print-schedule
& $py -X utf8 -m research.deliver smoke --config local_configs.research.DFormerv2_S_DeLiVER --device cpu --size 128 --out outputs/deliver-preparation
```

后续正式训练仍用 `research.train_odg --config local_configs.research.DFormerv2_S_DeLiVER --geometry-mode original`，补齐获批 batch/阶段参数后启动；val 评价用 `research.evaluate_odg --config ... --checkpoint ... --split dev --no-pad_SUNRGBD`，最终 test 改 `--split test`。这些只是入口约定，不是训练执行授权。

## 4. 最小验收

**达到数据与入口可用即可；本轮规划没有执行以下检查。**

- **清单与路径：** train/val/test 来源于对应官方目录、排序稳定、完整路径保留，清单内样本标识互斥；不对整个资产逐文件审计。case 过滤只在已生成清单上做，空子集清楚报错。
- **不超过6组样本读取：** 覆盖 train/val/test（test仅格式读取），在限额内优先覆盖不同天气和一种非空相机故障。只对这些样本检查三模态存在、尺寸一致、原生 dtype/通道、Depth值域和标签唯一值；raw 1/25/0/255 的映射做一次小数组断言。验证训练crop输出480×480、val原生尺寸、支持mask和padding语义。
- **少量可视化：** 从上述样本保存2–3张 RGB／灰度Depth／按官方类别palette着色GT及叠图，检查道路、车辆、人物边界基本对应；不要把红通道 ID 原图的暗色显示误认为空标签。预览名保留天气/split/scene，避免同名覆盖。
- **CPU日程：** 只运行一次 `--print-schedule`，train计数来自实际清单，周期评价路径为val，不是test；此命令不建训练会话。
- **一次模型 smoke：** batch1，取一个真实 train 样本、同步缩到128×128，仅 `original` 模式；加载现有 encoder、25类新头，FP32完成一次前向、ignore255 loss、反向和一次 AdamW 更新。核对输出25通道、loss/有效梯度有限、可训练参数唯一入组（含全部Geo.weight），保存不可恢复为正式训练的smoke记录。用这次输出复用现有 `SegMetrics` 验证25×25矩阵和ignore计数，并以小型已知预测数组检查ignore不计分；不额外跑整份val或第二模型。
- **隔离复核：** 内容/差异确认新增配置不会继承SUN路径/padding/37类，默认data_module不变，DeLiVER条件补齐不影响SUN；不因此启动SUN重新训练或完整项目测试。

验收结论须区分“CPU数据/模型链路可用”与“正式GPU显存、吞吐已测”。后者本计划不要求；正式运行前在获批GPU做一次所选crop/batch的定点检查即可，不需要再做大规模工程适配。

## 5. 成本与风险

- **预计工程工作量：约2–4小时**（薄loader/配置/清单约1–2小时，小补丁与抽样验收约0.5–1小时，其余为隔离与交付复核）；是估计，不是已测耗时。CPU smoke控制在数分钟内，不安排长CPU训练。此阶段无需GPU或云端费用，也不升级依赖、不复制大规模数据。正式训练成本未知，本次不估算或申请。
- **最重要的数据风险是标签被灰度化。** 采用红通道ID和显式映射解决；未知ID定位单样本，不扩为全量标签审计。
- **Depth编码、单位、无效值仍未知。** 固定uint8图像表示可先接入并启动该表示下的匹配实验；不能据此声称米制几何、真实深度缺失或与官方HHA分数同协议。只有研究主张确实依赖这些语义时再做必要技术确认，不重新追溯下载来源。
- **现有优化器漏参及并行冲突。** 共享factory分派与DeLiVER条件补齐均需小diff复核；主线修复若先落地就复用，不做两套重复补丁。状态/报告由主力模型维护，不在执行worktree覆写主目录文档。
- **原生整图评价显存可能高于480 crop。** 当前不为未知风险开发滑窗；正式GPU检查若明确OOM，先降低eval batch（已默认为1），仍不行再提出单一固定推理resize并保留原始标签计分，作为显式新评价协议，不静默更改。

## 6. 需要裁决的问题

**本轮只保留一个影响接入边界的决策：是否批准先按 RGB + 原始单通道 `depth/` 的固定非米制表示接入，接受其不同于官方 HHA RGB-D 协议。推荐批准。** 若上级要求首轮即对齐官方HHA公开分数，应暂停该路线并另定适配方案，不能由执行模型偷偷切换到HHA。

其余细节按本计划自主实施。批准本计划只包含有限工程改动、数据抽样和上述CPU单次检查，不包括完整测试套件、正式训练、整份val/test推理、GPU/云端操作、提交推送或论文主线变更；GPU/实验合同与最终研究方法在需要时另行审批。
