# MMFR 两个候选论文方向：本地事实摸底

> 日期：2026-10-02。调查对象是方向 A 的自然 Depth missing（原始深度中的无观测区域）与方向 B 的推理协议敏感性；本报告是供上级选择方向的事实材料，不是方法方案或实验授权。当前事实与权限仍以 [实时状态](../main/MUSeg-current-status.md) 和 [开放决策](../main/MUSeg-open-decisions.md) 为准。

## 1. 摘要与调查边界

本地 MUSeg train-dev 原图中的无效深度比例均值为 **31.6745%**，中位数为 **25.8750%**；val-dev 对应为 **30.1485% / 26.8765%**。自然缺失同时包含大连通区域和许多小区域，与当前随机网格块删除并非同一个生成分布。这支持继续讨论自然缺失的训练分布覆盖问题，但尚不能证明 replay 会提高分割性能。

F-lite 的四个共同条件在单视图 Quick-Val 中均优于 C0；十视图 Main-Val 中 clean、entire-missing、misalignment 转为负差，spatial-dropout 仍为正。该差异同时涉及视图、padding、随机数消耗、batch 分组和执行环境，不能单独归因于 test-time augmentation（测试时多视图增强）。现有 view builder 可复用，严格 matched 比较的统一薄入口尚缺；本轮没有实现或运行它。

- **已完成的范围：** 指定 dev 划分 RGB/Depth/Depth16 的 CPU 输入统计；限定数据、corruption、训练和 evaluator 路径阅读；旧评分材料复核；本地资产及官方参考源码摸底。
- **未执行：** 训练、模型开发、backbone/evaluator 修改、新 Quick-Val/Main-Val、GPU forward/benchmark、official test、完整测试、全仓 hash/manifest/重复 Git 审计。
- **数据隔离：** 本轮不读取 Label 文件、official-test 清单/图片/标签/cache。val-dev 统计仅描述；未来规则、阈值、mask 库与采样分布只能由 train-dev 决定。
- **证据等级：** 本轮直接输入统计与代码观察、可读本地旧结果、正式报告中既有结果分别注明。官方 README 的论文成绩不是本地成绩；资产存在不是可运行资格。

关键概念：checkpoint 是某训练时点保存的权重/状态；C0 是 E1 Batch 1A 对照；mIoU 是平均交并比，数值以百分数表示；pp 是百分点。matched 指同一次比较固定数据、损坏输入、权重/行为、视图和随机策略，不意味着所有协议之间推理成本相等。Quick-Val 是四条件筛选，Main-Val 是十条件完整开发集评价，两者都不是 official test。

## 2. MUSeg 本地数据、原始单位与划分

数据根为 `D:\0Project\dataset\MUSeg_DFormer`，原始来源为 `D:\0Project\dataset\MUSeg`。`dataset_meta.json` 记载 RGB uint8、Depth16 uint16、Depth uint8、Label uint8，共 3171 对；本轮只打开 dev allowlist 指定的 RGB/Depth16/Depth。

- 原图统一尺寸为宽1082×高932，即每图1,008,424像素。MUSeg 论文描述 Azure Kinect SDK 对齐后，从2048×1536图像的中央六角有效视场中统一裁取矩形。**已裁掉的固定视场外区域不纳入自然 missing 分母**；当前932×1082内的零深度才进入本次统计。
- Azure Kinect 官方 `k4atypes.h` 将 `K4A_IMAGE_FORMAT_DEPTH16` 定义为两字节 little-endian unsigned、单位为相机原点距离的毫米；原始0表示无效。传感器格式证据与本地转换元数据共同支持单位解释，不从8位灰度反推物理精度。
- 固定8位映射为 `round(Depth16 * 255 / 13932)`。本轮逐像素验证所有1595张dev图与该映射一致；原始缺失定义为Depth16==0，模型输入缺失定义为Depth==0。
- 冻结 split：`data/splits/MUSeg/dev-v1/train-dev.txt` 为1277图/762组；`val-dev.txt` 为318图/196组。组ID取文件名首四段，矿井取首段。既有 `audit-report.json` 确认组不拆分与隔离；本轮沿用而不重建manifest。
- 论文预过滤“有效深度<40%且RGB均值<40”的双低质量图，保留RGB好/Depth差及RGB差/Depth好的图。这是亮度相关性解释的重要选择偏差；本轮统计不代表未过滤传感器总体。

来源：[MUSeg 数据文档](../dataset.md)、本地 `dataset_meta.json`、[冻结 split 审计](../../data/splits/MUSeg/dev-v1/audit-report.json)、LIB000001 的 Methods（本地论文主库正文）、[微软官方格式声明](https://raw.githubusercontent.com/microsoft/Azure-Kinect-Sensor-SDK/develop/include/k4a/k4atypes.h)。

## 3. 自然 Depth missing 的 CPU 统计

### 3.1 定义、总体分布和直方图

统计脚本为 [audit_museg_natural_missing.py](../../tools/mmfr/audit_museg_natural_missing.py)，本地原始输出为 `outputs/direction-audit-20261002/summary.json`、`train-dev_per-image.csv`、`val-dev_per-image.csv`。这些是输入统计，不含标签、预测或性能分数；CSV不是训练mask库，未写出任何mask。

所有图尺寸相同，因此像素加权均值与逐图等权均值数值相同。分位数采用NumPy linear interpolation。下表缺失/有效率均为百分数。

| 指标 | train-dev 1277图 | val-dev 318图（只描述） |
| --- | ---: | ---: |
| 缺失率mean | 31.6745 | 30.1485 |
| 缺失率median/P50 | 25.8750 | 26.8765 |
| 缺失率P10 | 3.9666 | 5.4968 |
| 缺失率P25 | 10.7610 | 10.7827 |
| 缺失率P75 | 49.4414 | 43.3985 |
| 缺失率P90 | 69.4491 | 63.0460 |
| 有效率mean | 68.3255 | 69.8515 |
| 有效率median | 74.1250 | 73.1235 |
| 有效率P10 / P90 | 30.5509 / 96.0334 | 36.9539 / 94.5032 |
| Depth16>0但Depth8==0新增像素 | 0 | 0 |
| 全图Depth8==0图数 | 0 | 0 |

| 单图缺失率区间 | train-dev图数 | val-dev图数 |
| --- | ---: | ---: |
| [0,1%) | 23 | 4 |
| [1%,5%) | 135 | 26 |
| [5%,10%) | 135 | 39 |
| [10%,25%) | 331 | 82 |
| [25%,50%) | 344 | 103 |
| [50%,75%) | 218 | 47 |
| [75%,100%) | 91 | 17 |
| 恰好100% | 0 | 0 |

“量化新增零为0”只适用于这1595张已核验dev图，不扩展为official test或所有传感器值域的声明。clean条件也保留这些自然洞，因此这里的clean是“未额外施加人工损坏”，不是“完整深度”。

### 3.2 矿井、组与亮度

| 矿井前缀 | train图/组 | train缺失mean / median (%) | val图/组 | val缺失mean / median (%) |
| --- | ---: | ---: | ---: | ---: |
| 01 | 273 / 227 | 26.1815 / 19.8151 | 68 / 59 | 22.4391 / 17.4109 |
| 02 | 178 / 143 | 24.9394 / 19.5397 | 44 / 32 | 18.5640 / 13.1531 |
| 03 | 247 / 182 | 23.8541 / 17.1506 | 62 / 45 | 21.8209 / 18.6313 |
| 04 | 166 / 35 | 22.0026 / 15.8457 | 41 / 13 | 30.0248 / 31.1398 |
| 05 | 40 / 25 | 64.1373 / 61.6737 | 10 / 7 | 63.1985 / 59.8282 |
| 06 | 373 / 150 | 44.9108 / 44.3533 | 93 / 40 | 43.3189 / 40.9135 |

亮度是RGB uint8码值的BT.601 luma `mean((.299R+.587G+.114B)/255)`，未做gamma/ICC校正，不是物理照度。train亮度mean/median为0.239348/0.183401；val为0.239188/0.199294。亮度与单图缺失率的Pearson相关系数train **+0.171228**（1277图）、val **+0.125564**（318图）。这是弱的正相关描述；不支持“暗就缺失更多”的简单全局规则，也不能证明亮度导致缺失。矿井组成、预过滤、场景与组内重复均可能影响它；本轮未作回归、组级推断或显著性分析。

### 3.3 连通区域、缺失分层与代表样本

连通区域使用Depth8==0的**8连通**定义。train区域总数522,427；单图区域数mean409.10、median252、P10/P90=87/1014.4；全部区域等权面积median12像素、P90=98像素。最大区域占整图面积mean24.7737%、median14.9844%、P90=65.2286%。val对应为123,077个区域、单图mean387.03/median274、区域面积median12/P90=103、最大区域整图占比mean23.0805%/median15.4573%。**区域等权面积与图像等权最大洞面积是两种统计量，不相互替代。**

为描述形态而设的“最大区域≥整图1%且≥该图无效像素50%”标记命中train782/1277、val190/318；“至少20区域且最大区域≤无效像素5%”标记命中train1/1277、val0/318。它们是脚本的描述性定义，不是数据集类别、训练规则或“大洞/散洞”完备互斥分类；阈值很严，后一个计数低不等于没有小散洞。

低/中/高缺失分层仅用train P25=0.1076104892与P75=0.4944140560，分别定义为≤P25、(P25,P75]、>P75。train人数320/638/319、均值5.3676%/26.8500%/67.7131%；相同train阈值描述val为80/174/64图。这不是建议采用分层采样。

| train代表 | 文件名 | 缺失率 (%) | 最大洞占整图 (%) |
| --- | --- | ---: | ---: |
| 低缺失层均值附近 | `03-01-01-0338-240526112222-04-99.jpg` | 5.3722 | 1.2537 |
| 中缺失层均值附近 | `02-01-01-0168-240524100513-04-99.jpg` | 26.8311 | 18.1576 |
| 高缺失层均值附近 | `06-01-01-0225-230920140510-04-99.jpg` | 67.6935 | 60.4027 |
| 最大缺失/最大洞 | `03-01-01-0126-240526100815-06-99.jpg` | 98.7497 | 98.6987 |
| 最少缺失 | `04-01-01-0027-230808080117-12-99.jpg` | 0.0128 | 0.0051 |
| 小区域数量相对无效面积较多候选 | `01-01-01-0327-240523095529-02-99.jpg` | 4.3466 | 0.4861 |

代表来自确定性数值选择，尚未作逐样本视觉机制标注。CSV保留矿井/组字段，支持后续经授权仅从train回取mask；不能用val代表挑选训练库。

## 4. 自然缺失与现有 synthetic corruption 的差异

现有实现为 [multimodal_failure_v3.py](../../utils/dataloader/multimodal_failure_v3.py)。输入是对齐uint8 Depth，初始validity state为Depth>0与geometry support的交集；多个故障按spec顺序作用，不是独立并行叠加。最后invalid严格0，仍有效点至少1；强度扰动不填原洞。RGB保持不变。

| 故障 | 实际操作/冻结强度例 | 与自然缺失的关系与边界 |
| --- | --- | --- |
| spatial_dropout | cell=max(8,min(H,W)//16)，ceil网格中无放回选round(severity×格数)个块；@.75约删75%块，Depth/state置0 | 原图cell58、17×19格；480×640训练crop cell30。不是pixel Bernoulli，也不保证删75%原有效点；选块可覆盖原洞，且不看RGB/场景。 |
| entire_missing | 全Depth/state置0；冻结severity1 | 与本轮dev中0张全图自然缺失不同，是明确压力条件，不是自然发生率估计。 |
| gaussian_noise | sigma=severity×48；@.75为36灰度级，只改current-valid强度 | 增加噪声但不造缺失state，不模拟自然洞形态；单位是8位灰度而非毫米。 |
| blur | sigma=severity×minside/80，mask-normalized Gaussian，REFLECT_101；仅valid，state不变 | 使有效区模糊，原洞保持无效，不是填洞。 |
| quantization | levels=clip(round(256(1-severity)),2,256)；@.75为64级/step255÷63 | 在原8位映射上再量化，只改valid强度，state不变；区别于本次原始16→8位转换审计。 |
| misalignment | 随机整数平移Depth与state；axismax=severity×dimension/30，零位移强制至少1；外缘0 | 搬运原洞并造边缘缺失，空间几何关系改变；不是仅有效点加噪。 |
| mixed | SD.5→noise.5；blur.5→mis.5；quant.5→mis.5 | 顺序明确，已缺失对后续强度操作保持缺失；平移会搬运缺失。 |

自然mask由观测图本身提供，矿井间缺失率差异已直接见到；与材质、距离、反射、光照等成因的关联未被本轮输入统计识别。相同缺失比例并不意味着相同空间分布。当前块删除叠加于已有自然缺失上，不能把人工severity与最终观测缺失率等同。

## 5. Train-only mask replay 的最小接入事实

**可复用的是输入侧训练batch构造接口；本轮没有增加replay、保存mask库或拟定新训练方法。** replay在这里仅指将train-dev原图的无效区域作为额外遮挡形态施加给训练输入，不表示恢复自然真深度。

现有数据链先读取Depth8，mirror、resize、normalize、crop/pad，再由 `utils/mmfr_training.py::build_mmfr_training_batch_v3` 在CPU反归一化为uint8、施加corruption、重新normalize，最后送GPU。它不是Depth16毫米域或feature域操作。natural raw0归一化后为−1.7142857；pad normalized0与之不同，不能把normalized0当自然缺失。

最小事实入口有两个：
1. 若先在原图网格施加train mask，需在数据preprocess之前让mask与Depth共用同一mirror/scale/crop/pad；适合保留原始洞的空间变换，但需要显式mask流转。本轮未改loader。
2. 若复用现有batch corruption点，需把train源mask按当前训练网格变换、仅在geometry support内置0，并同步validity state。不能在归一化张量中直接写0冒充缺失，也不能从Label/val/test或故障名称生成判别信息。本轮未改helper。

v3训练合同的基础seed为2026091402，clean概率.25、最多2个spec；前1/3使用单故障severity .05–.30且无entire-missing，中1/3扩至.60/最多2故障，后1/3扩至1并允许entire-missing（severity1）。这不是从自然mask分布学习出的课程。A-v1 Proposal/Gate使用各自phase seed2026093001/2026093002，仍复用batch构造helper，不把故障metadata送入Gate。

训练随机源已有独立PCG64/SeedSequence，以train seed、epoch、iteration、rank、slot、规范化sample ID哈希词组合，避免全局RNG；A-v1 Proposal/Gate也复用此构造链。新增来源选择若获授权，应继承可追踪train-only来源与确定性种子，保持现有clean比例/旧故障合同独立，不悄悄覆盖冻结v3。

当前train原始mask在原图上可按Depth>0读取，但训练resize使用线性插值，边缘零/非零与原始二值mask的最近邻变换不一定相同。上级需先决定replay要表达“原始无观测support”还是“变换后输入的零值support”，再授权最小实现。这里列的是接入限制，不替上级选方案。

## 6. F-lite Quick-Val → Main-Val 完整差异

### 6.1 协议差异矩阵

| 维度 | Quick-Val | Main-Val | 可以据此说什么 |
| --- | --- | --- | --- |
| checkpoint | C0/F-lite各自fixed-final2560 | 同一对checkpoint | 排除两次评价之间换权重；候选本身权重不同。 |
| split/样本 | val-dev318，四条件 | val-dev318，十条件 | 四共同项可看相对差符号；条件aggregate不同。 |
| corruption | 原始对齐Depth一次损坏，之后建输入 | 原始对齐Depth一次损坏，之后建views | 源码支持共享同一raw realization的确定性前提；未做跨runner318份数组逐一核验。 |
| corruption seed | 2026091401+canonical condition.index+规范化sample ID哈希词 | 同规则与seed | seed数字相同不足以独立证明同输入；index/ID/原输入/generator都必须相同。 |
| scales/flip/views | 1/no flip/1 view | .5,.75,1,1.25,1.5，各original+horizontal flip/10views | 输入view集合变化，不能称相同模型观测。 |
| whole/sliding | whole image，无滑窗 | whole image多view，无滑窗 | 不是whole与sliding差异。 |
| resize/normalization | 无spatial resize；Depth /255、.48/.28、3通道 | RGB/Depth OpenCV linear resize后同normalize | resize发生在corruption之后。 |
| padding | 无pad | 右下pad到32倍数，normalized0 | 对932×1082原图scale1也不相同；pad不等于raw0。 |
| label/metric grid | 原Label网格，必要logit bilinear align_corners=False | 每view裁pad、反flip、回原网格后融合 | 最终标签网格相同，logit变换路径不同。 |
| precision/TF32 | FP32/TF32 off | FP32/TF32 off，融合FP32 | 开关同，不保证跨硬件逐位相同。 |
| HAM随机性 | reset-per-unit后单次forward | reset-per-unit后依序全部views | NMF每forward随机初始化bases，scale1 view不保证复用单视图NMF抽样。 |
| view batching | batch1 | scale.5 original+flip batch2，其余batch1 | 形状、调用次数、随机数消耗同时变化。 |
| logits/指标 | 单view，累计数据集confusion算mIoU | 十view pre-softmax logits FP32平均，再累计confusion | 条件内计分定义同，输入预测生成方式不同。 |
| evaluator资格 | original-full薄入口，输入318/318等价检查；无既有冻结资格 | 冻结msflip-whole-original-grid入口/身份检查 | 资格差异必须保留，不能视为只换view开关。 |
| environment | 云RTX4090、torch2.1.2+cu118 | 本地RTX5060Laptop、torch2.7.0+cu128、Python3.13.9/Windows11 | 环境同时变化；各批次内部C0/F-lite仍环境匹配。 |
| primary aggregate | EM/SD/Mis三hard未加权M3 | 六种单故障未加权M6 | M3和M6不可直接做同口径收益对比。 |
| gate/完成性 | 冻结Quick gate，promote | 无预注册Main gate；10×318/侧结果覆盖齐全 | Main结果描述性，不事后制造stop线；Main stdout/exit code未落盘。 |

候选间另有既有训练限制：F-lite额外初始化消耗torch RNG，shuffle与clean slot数有小差异（6359 vs6369/25600）。它是两候选训练来源差异，不是Quick→Main新发生的变化。

### 6.2 四共同条件与十条件结果

| 共同条件 | Quick C0 | Quick F-lite | Δpp | Main C0 | Main F-lite | Δpp | 符号 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| clean | 53.46 | 54.15 | +0.69 | 56.69 | 56.02 | −0.67 | 正→负 |
| SD@.75 | 51.40 | 52.39 | +0.99 | 53.92 | 54.32 | +0.40 | 仍正 |
| Mis@.75 | 52.15 | 52.63 | +0.48 | 55.57 | 55.24 | −0.33 | 正→负 |
| EM@1 | 48.76 | 50.17 | +1.41 | 52.63 | 52.00 | −0.63 | 正→负 |

另外六个Main条件：noise56.34→55.85/−.49；blur56.65→55.95/−.70；quant56.62→55.99/−.63；SD.5+noise.5 54.94→55.36/+.42；blur.5+Mis.5 56.14→55.33/−.81；quant.5+Mis.5 56.00→55.37/−.63。表中分数沿用旧报告两位小数，Δ亦按旧报告显示精度，不据舍入值重设门槛。

Quick M3为50.77→51.73/+0.96pp，原筛选promote；Main M6为55.2883→54.8917/−0.3966pp。**三个共同项翻负是观测事实；其机制、显著性及F-lite去留仍未裁决。**

来源：[Batch1A Main报告](2026-09-22-mmfr-e1-batch1a-c0-flite-mainval.md)、[E1 protocol](../../MMFR/01_research/e1_batch1_protocol.md)、本地 `cloud/MMFR_E1_Batch1A_local_transfer_20260922/MMFR_E1_Batch1A_local_transfer_20260922/` 下comparison、两侧Quick summary/Main manifest与summary；代码为 `tools/mmfr/e1_quickval.py`、`tools/evaluate_museg_10condition.py`、`tools/evaluate_museg_checkpoint.py`、fast runner与HAM head。

## 7. 严格 matched single / flip / multi-scale 推理的可行性

现有函数 `build_original_full_input` 支持无pad单视图；`build_msflip_views(scales=[1.0])` 支持两view；默认五尺度支持十view。`build_rgb_view_cache`/`assemble_views` 可从同一raw corrupted Depth派生各view。**函数可复用不等于三协议统一入口已实现或资格已确认。**

最小需要补的控制范围（仅事实清单，本轮未写代码）：
- 每sample-condition只读一次/只生成一次raw corruption；各候选或off/full/learned复用完全相同的views，不在view内重新抽损坏。
- 显式profile记录scales、flip、pad、view顺序、batch grouping、logit融合、precision和metric grid；共同条件名不足以保证协议兼容，旧 `e1_quickval.compare()` 不能直接用于跨profile gate。
- 同一profile内部对候选/行为恢复相同Python/NumPy/torch CPU/CUDA随机起点；HAM/NMF每forward `torch.rand`，reset-per-unit只配对整套调用序列，不保证跨profile每view抽样相同。若需跨profile单view耦合，需另行明确view级随机政策，不悄改原冻结评价。
- 单view若沿用Quick无pad、双view沿用Main的32倍pad，仍同时改变padding。要隔离flip/scale，需显式统一这条路径；统一pad会与旧Quick历史口径不同，应新标身份。
- A-v1必须显式控制off/full/learned，R-OE必须传当前raw_depth与geometry support。旧generic forward的默认行为不能代替三行为比较；A-v1现成四条件三行为入口不是十条件Main adapter。
- 结果必须区分“同一raw损坏”“同view tensor”“同forward随机策略”和“同推理成本”。1/2/10views天生预算不同；能做到各profile内跨方法相同预算，不能将跨profile比较称严格等成本。

代码证据：E10 `unit_seed_words`、`corrupt_depth`、`build_rgb_view_cache`、`build_depth_view`、`assemble_views`；EV `build_msflip_views`；FAST RNG capture/restore与view-group后处理；HAM `NMF2D._build_bases`。本轮仅源码与旧材料核验，未做新forward或跨硬件等价实验。

## 8. 本地 baseline、checkpoint 与可复用程度

**本次限定路径盘点未找到可立即使用的Windows本地MUSeg checkpoint。** 代码、配置、候选身份JSON、远端权重记录和本地权重文件必须分开。以下“未发现”仅指主工程encoder/MUSeg配置及指定 `checkpoints/`、`experiments/`、`cloud/` checkpoint范围，不声称整个磁盘无资产；未读取权重或重新计算hash。

| 方法 | 本地代码/配置 | MUSeg checkpoint与本地直接推理 | 下一步缺口（未授权） |
| --- | --- | --- | --- |
| DFormerv2-S | `models/encoders/DFormerv2.py`、`local_configs/MUSeg/DFormerv2_S_Base.py`，HAM头 | 配置中的Small pretrained是初始化引用，非MUSeg训练权重；本次范围未找到本地权重，未验证推理 | 先取得真实MUSeg权重；不能取得才讨论微调/重训。 |
| C0 | `DFormerv2_S_MMFR_E1_Batch1A_C0.py`与Common | 本地A2 selector-epoch-420.pth未发现；身份JSON记录远端A2，正式C0另有远端fixed-final2560记录 | 可取回已有精确C0，不默认重训；转入后仍需config/runner资格。 |
| F-lite | `DFormerv2_S_MMFR_E1_Batch1A_FLite.py` | 本地transfer结果JSON可读，但所查checkpoint目录没有权重文件 | 找回同一fixed-final；结果JSON不等于权重。 |
| A-v1 | `DFormerv2_S_MMFR_AV1.py`，独立三行为Quick入口 | 远端final身份已在正式报告核验；本地对应cloud目录未读到。现入口仅四条件单view | 转入权重不自动授权；十条件Main adapter缺口仍在。 |
| R-OE-lite v2 | `DFormerv2_S_MMFR_E1_Batch1B_R_OE.py` | 本次指定范围未找到本地权重；正式训练报告提供远端结果 | 先确认精确远端权重/入口资格，不默认重训。 |
| DFormer | `models/encoders/DFormer.py`；NYU系列配置有 | 未发现MUSeg配置+训练权重组合 | 若纳入对照，需MUSeg适配及真实权重，通常另需微调授权。 |
| CMX | 所查主工程encoder/MUSeg配置未发现 | 未发现对应MUSeg权重，不能直接推理 | 实现、配置、权重三项缺口；不为本轮全面clone。 |
| CMNeXt | 同上 | 同上 | 同上。 |
| GeminiFusion | 同上 | 同上 | 同上。 |
| DFormer++ | 所查实现/配置未发现同名资产；不当作DFormer/DFormerv2别名 | 未发现对应MUSeg权重 | 先确认实现身份；论文LIB000037接入不等于代码/权重就绪。 |
| ConD / RobustSeg | 已有官方clone `RGBD-MMCD` / `RMMSS` | 主工程未核到对应MUSeg配置/权重；不是现成推理baseline | 本轮仅确认remote/已有索引，不运行或移植。 |

`experiments/MMFR_A2_v3/checkpoints/checkpoint-candidates.json` 直接可读，selector420/390/370/latest均指 `/root/rivermind-data/cloud/...`；Common config要求的本地selector420路径只是配置引用。最新A-v1报告记载的C0/final与条件性ZIP是远端已核验资产，本轮未连接云端、下载、解包或读checkpoint。**恢复已有权重通常比重新训练更直接，但是否采用该baseline由上级决定。**

来源：上述实际配置、候选JSON、[A-v1正式报告](../../MMFR/02_evidence/report_mmfr_a_v1_formal_quickval_upper_review_20261001.md)、[代码索引](../../MMFR/03_reference/code-index.md)；本地限定盘点材料为 `outputs/direction-audit-20261002/assets-external-evidence.md`。

## 9. NYUv2 的本地数据与第二数据集缺口

本次在已知 `D:\0Project\dataset\NYUDepthv2` 与项目 `datasets/` 路径未找到数据；**NYUv2的实际raw/filled/inpainted文件版本尚未核验，当前不能作为就绪的第二自然missing数据集。** 未下载或读取NYU图像/标签。

已核到 `local_configs/_base_/datasets/NYUDepthv2.py`：`Depth/.png`、单通道、train.txt/test.txt、795/654、40类，输入480×640，`is_test=True`；“raw depth”仅为 `x_is_single_channel` 的通用注释，不证明磁盘内容。`local_configs/NYUDepthv2/DFormerv2_S.py` 引用 `checkpoints/pretrained/DFormerv2_Small_pretrained.pth`、scale1+flip；该引用本轮未确认为本地文件，不是MUSeg训练权重。配置有创建log目录副作用，本轮只读文本、没有import它。

主代理直接复核本项目 `utils/dataloader/RGBXDataset.py::__getitem__/_open_image` 与 `dataloader.py::TrainPre.__call__`：d模态用OpenCV `IMREAD_GRAYSCALE`、复制三通道、交给通用preprocess；所读读取/预处理函数没有显式hole filling。上游官方DFormer的对应读取路径亦没有补洞。**loader不补洞不能证明磁盘文件是raw；更不能证明IMREAD_GRAYSCALE保留原始16位毫米值。**

最小后续准备应按实际文件分支，而不是仅改目录名：
- 已核验的8位输入/预处理版本：可最小调整 `dataset_path/x_root_folder/train_source/eval_source`，保留单通道与明确Depth映射；raw与filled文件需来源与配对证据。
- 真正uint16传感器raw：当前grayscale读取和归一化合同不能直接当毫米原值通路；需独立授权明确保留读取、invalid sentinel、单位、固定映射与对齐/裁剪。仅换路径不足以验证自然missing实验。
- 先明确NYU数据与split角色，再决定第二数据集评价。现成 `test.txt/is_test=True` 配置不是本次数据访问或执行授权；MUSeg official test继续sealed。

已具备：模型/loader/config参考入口。未具备：本地数据、raw/filled版本证明、适用训练checkpoint、自然missing统计与matched evaluator资格。它们均未由本轮补齐。

## 10. MMSS Benchmark 与 GeomPrompt 官方代码

### 10.1 来源、去重与实际取得范围

克隆前核对已有外部代码索引及既有remote，复用已存在的官方DFormer。仅新增一次浅clone：`D:\0Project\origin\MMSS`，origin为 [Chenfei-Liao/Multi-Modal-Semantic-Segmentation-Robustness-Benchmark](https://github.com/Chenfei-Liao/Multi-Modal-Semantic-Segmentation-Robustness-Benchmark)，main；README题名与 [arXiv2503.18445](https://arxiv.org/abs/2503.18445) 一致。跳过LFS smudge，不取大权重/数据。重点源码读取只有已有DFormer与新增MMSS两仓；没有克隆DELIVER或第三方GeomPrompt。

MMSS根目录常见LICENSE/COPYING未找到，许可**未确认**；只读参考不授权代码复制。README要求将三个val脚本替换进DELIVER工程，imports依赖 `semseg.*`；该clone不是自足runner，也不是MUSeg两模态配置。已有11-repo索引保持历史日期，仅补本次新增事实。

### 10.2 MMSS的实际实现与可复用限制

| 项目 | 直接代码观察 | 对本项目的限制 |
| --- | --- | --- |
| EMM 整模态缺失 | `val_mm_EMM.py::evaluate_with_missing_modalities`：RGB/D/Event/LiDAR四模态，枚举缺失数0–3，指定tensor `zero_()` | 包含clean子集，排除四模态全缺失；tensor后置零的语义不等于MUSeg raw0。 |
| RMM 随机部分缺失 | `val_mm_RMM.py::evaluate_with_random_missing_modalities`：当前硬编码r=.25，`torch.rand(image.shape)<r`后写0 | 按完整tensor shape采样，若含通道维则各通道独立；不是本项目网格块删除/共享空间mask。README展示多r结果不等于当前脚本自动跑多r。 |
| NM 噪声 | `val_mm_NM.py`：salt/pepper写当前tensor max/min到随机坐标；Gaussian用torch.randn；level固定0 | 实际density .05/std .1；salt-pepper四模态、Gaussian RGB/D/L不含Event，非MUSeg Depth-only valid-state规则。 |
| 随机观测复用 | RMM/NM未见sample-keyed seed、损坏数组缓存或sample-condition-view记录 | 不同模型/重复运行不能据同条件名保证同realization；若借用须另定可复现输入合同。 |
| case与计分 | main `cases=[None] # all`；每缺失子集遍历dataloader，batch调用Metrics、末尾compute | 不是按DELIVER每case分别报告；DELIVER数据类/Metrics不在clone，样本/view底层组织与精确聚合未核验。 |
| 条件聚合 | EMM/RMM对15个保留子集取等权均值；按p=.2/.1/.05独立模态故障概率加权并以保留概率和归一化 | 是排除全缺失后的条件期望，不是本项目六单故障M6；换两模态需改组合与指标定义。 |
| 多尺度/flip | `evaluate_msf` resize到32倍尺寸，bilinear align_corners=True，累加softmax概率；main与损坏分支二选一 | 不是现成同一次损坏+多views runner；区别于本项目右下pad及pre-softmax logits平均。 |

最直接可借鉴的是“显式条件循环”与mask/noise生成操作的阅读依据；移植需要MUSeg两模态/原始零值/种子/观测复用/指标适配和许可确认。**本轮没有移植或运行MMSS，不将其README成绩写作本地结果。**

### 10.3 GeomPrompt：官方代码截至本轮未找到

直接核到作者 [项目页](https://geomprompt.github.io/) 与官方CVPR Workshops/arXiv论文入口，未发现可确认官方源码链接；已知本地 `GeomPrompt/.git/config` 未读到。因此状态是“本次限定官方来源未找到代码”，不是证明作者永远未发布；未以第三方仓库替代官方实现。

项目页说明GeomPrompt从RGB合成任务相关几何prompt给冻结RGB-D segmenter；Recovery从RGB+退化Depth预测有界修正残差，仅分割监督、不是毫米真深度恢复。论文文字描述clean概率.2，否则均匀一种quantize/hole/dropout/noise/blur/banding/scale-shift，severity .10–.90；DFormer五尺度+flip、GeminiFusion单尺度480×480无flip。**这些属于论文描述，不是源码核验**；raw/inpainted输入版本、hole/dropout具体mask、seed/sample-condition-view复用及聚合实现仍待核验。

关联风险只作事实提示：任务驱动输入补偿/几何prompt已有近邻，不能凭本轮缺失统计宣称方法创新。GeomPrompt公开成绩、速度和参数不是MUSeg本地实验，本报告不据此选择新结构或重设计A-v1。

## 11. 四条历史路线：已知结果与不能外推的边界

| 路线 | 已有本地/历史事实 | 当前能支持的结论 |
| --- | --- | --- |
| F-lite | Quick三hard+.96pp/promote；Main M6−.3966pp，四共同项三翻负 | 推理协议敏感性值得审查；无Main预注册门槛，不自动判stop或某模型更好。 |
| R-OE-lite v2 | 2560成功更新；Quick clean0、EM/SD/Mis各+.01pp，M3 50.77→50.78/inconclusive；EM318/318路由，其余0/318 | substitute确实训练/触发；base共有812键中734变动、无同权重forced-bypass，+.01不可净归因substitute。入口资格与路线去留仍开放。 |
| A-v1 | Proposal1920/Gate640、skip0；同checkpoint配对off/full/learned；hard均值50.7703640193/50.7661057180/50.7684910166 | learned−off−.0018730027pp未达+.50门槛，正式stop；不支持续训/重评自动授权，也不否定所有动作效用选择方法。 |
| Oracle-A | original clean/SD/EM=57.06/54.42/52.55；strict53.58/52.65/52.61；aggregated53.80/52.72/52.61；geometry_off52.61/52.61/52.61 | NO-GO关闭具体validity-aware pairwise geometry suppression动作族；不是所有Depth可靠性或缺失处理均无效。 |

A-v1 clean learned−off−.0100313704pp通过≥−.20保护线、hard learned−full+.0023852986pp通过，但困难条件收益线失败；gate均值约.541–.543只是未明显按条件区分的观察，不是可靠性概率或机制证明。其最终SHA沿用已有核验 `87ec54d10192d3aedc1d6edb864b7b60b94ce66220a748a12f1ca50ca605d336`，本轮不重新hash。训练仅NMF局部FP32从完整1280恢复，非完整FP32冷启动。

R-OE报告可读；本地 `cloud/mmfr-e1-batch1b-roe-v2/` 与 `cloud/mmfr-av1-formal-fp32-resume1280-v1/` 原始目录本轮未读到，相关数值依据正式报告与既有 `reproducibility_current.json`，不声称本轮重算原始评分。Oracle aggregate analysis JSON与Batch1A transfer小结果本轮直接可读。

来源：[R-OE v2报告](../../MMFR/02_evidence/report_e1_batch1b_roe_v2_formal_training_quickval_20260924.md)、[A-v1正式报告](../../MMFR/02_evidence/report_mmfr_a_v1_formal_quickval_upper_review_20261001.md)、[既有复现信息](../../MMFR/02_evidence/reproducibility_current.json)、[Oracle-A报告](2026-09-19-museg-mmfr-oracle-a-validity-aware-pairwise-geometry.md)、`experiments/MMFR_OracleA/eval/oracle-a-analysis.json`。原报告与原始证据均未改写。

## 12. 上级裁决问题、交付检查与停止点

### 12.1 需要上级决定的8个问题

1. 方向A是否值得进入独立立项：目标是自然缺失分布覆盖/输入增强，还是需要性能证据后再选？本轮输入统计不能替代收益实验。
2. 若允许train-only replay，来源mask是否仅用train-dev，表达原始无观测support还是resize后零值support；是否先限定最小输入侧实现，而不碰backbone/evaluator？
3. 方向B是优先核清F-lite历史反转的协议因素，还是作为新方法评价原则；是否先冻结padding、view batching、HAM随机配对及环境，再单独授权统一薄入口？
4. matched比较是否按各profile内同预算，明确single/flip/msflip跨profile不等成本；若要求跨profileview级NMF耦合，是否接受与旧冻结评价不同的新身份？
5. 最小首轮对照采用哪些本地可读模型/权重；已有MUSeg训练权重与外部预训练/参考实现如何分级，哪些baseline缺口需要后续独立训练授权？
6. NYUv2是否作为第二数据集，以及采用filled输入还是必须取得raw/filled配对；若要验证自然missing，版本、split与单位/映射缺口由谁补齐、预算多少？
7. MMSS只借鉴输入损坏/指标实现，还是要求协议对齐；GeomPrompt若无可确认官方完整实现，是否允许论文级对照而不把第三方代码当官方？
8. A-v1继续保持stop、R-OE/Flite原去留独立未决；上级是否明确关闭其中旧路线，以免本次候选调查被误读成复活旧训练或Main-Val授权？

### 12.2 复现与检查边界

输入统计的可复用命令为：

```powershell
python D:\0Project\DFormer\tools\mmfr\audit_museg_natural_missing.py
```

只查train可加 `--skip-val`。命令输出到固定本地统计目录，不加载模型或标签。该命令记录复现入口，不要求重复执行已完成的全dev统计。

已实际运行 `python tools/mmfr/audit_museg_natural_missing.py`，完成1595图限定CPU统计并生成summary/两份CSV；脚本同时检查dtype、对齐尺寸与固定量化逐像素一致性。主代理直接复核统计代码、关键summary数值与CSV结构、corruption/view/RNG代码、旧结果和资产/官方来源，未重复全量图像统计。该工具的定点静态诊断无报告项；report-index与summary的JSON解析成功，限定Git差异人工复核及格式检查通过。没有新增临时测试文件、验证脚本或完整测试。

完整测试与GPU/训练/新评价按本轮边界未运行，不能写成通过。报告列出的可行入口均为静态调查结论，不是模型运行验收。

交付为本报告及其统计工具/本地小输出；更新滚动实时状态、现有report-index与MMFR导航/CHANGELOG/code-index。没有创建manifest/hash库、生成新审核包、提交或推送；旧生成审核包仍是历史快照，不冒充本报告。完成后停止，等待上级选择方向和任何新的实现/运行授权。
