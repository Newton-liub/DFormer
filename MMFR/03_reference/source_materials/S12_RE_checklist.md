# MMFR 的 RE 编号文献整理清单（修正版）

> 用途：依据“数据/场景直接性—失效机制直接性—可用实验证据”整理 RE 文献。RE 合集主要用于支撑矿下场景退化、多传感器必要性、质量/可靠性表征与工程动机，不应单独替代鲁棒多模态学习的核心方法文献。

## 一、主文献夹：保留 4 篇

| 等级 | 编号 | 处理建议 | 主要用途 |
| --- | --- | --- | --- |
| S | RE326 | 核心必引 | MUSeg 数据集直接来源：地下矿山 RGB-D 语义分割，包含 3171 对对齐 RGB/Depth 图像、6 个矿区与 15 类语义标注，是 MMFR 主实验场景与数据依据。 |
| A | RE240 | 核心邻近工作 / 已核 | 地下煤矿 Camera+LiDAR BEV 检测；以 BEV 局部特征统计构造空间置信图，并通过 MoE-FLIM 与 PMS-FFEM 实现局部动态模态加权和跨模态补偿。可作为“矿下局部 reliability-conditioned fusion”直接先例，但不包含 RGB-D segmentation、完整模态缺失或严格 reliability 因果对照。DOI: 10.3390/s25165185。 |
| A- | RE026 | 强邻近工作 / 已核 | 露天矿 RGB+LiDAR 3D 检测；利用 dark-channel 等物理指标估计粉尘/质量状态，并执行“全局权重 + 局部 pixel/projection-point 修正”的动态融合。属于物理先验驱动的确定性质量估计，而非 learned reliability。DOI: 10.1088/1361-6501/ae58c7。 |
| A- | RE094 | 强跨任务邻近工作 / 已核 | 偏振—强度单目深度估计；以饱和度、局部对比度和 DoLP 构造全局 reliability scalar，并在输入级进行 deterministic modality/channel gating。已有 gated-vs-ungated 对照，但不是像素级 reliability，也非 RGB-D segmentation。DOI: 10.3390/photonics13030268。 |

### 主文献夹的引用边界

- `RE240` 已经实现“局部模态置信度 → 空间动态权重 → 跨模态补偿”，因此 MMFR 不应再声称“首次在矿下根据局部模态可靠性动态调整融合”。
- `RE026` 已经实现“显式退化/物理质量估计 → 动态跨模态权重”，且包含局部空间修正，因此 MMFR 不应声称“首次利用退化程度动态调整多模态融合”。
- `RE094` 已经实现“显式 reliability → modality-specific action”，因此 MMFR 也不宜使用过宽的“首次依据模态可靠性动态调整信息贡献”表述。
- MMFR 与上述工作的主要区分仍应落在：RGB-D dense semantic segmentation、Depth-specific failure diagnosis、与最终 Depth 状态对应的空间连续可靠性、DFormerv2 geometry pathway 条件干预，以及 oracle / constant / shuffle / inverse / parameter-matched 等因果控制实验。

## 二、背景子文件夹：继续保留，但不要写成 MMFR 核心机制证据

### 2.1 场景退化、质量与任务驱动处理

`RE023` `RE043` `RE049` `RE117` `RE135` `RE140` `RE166` `RE183` `RE191` `RE218` `RE241` `RE259` `RE266` `RE285` `RE350` `RE354` `RE366` `RE397` `RE404` `RE418` `RE455` `RE474` `RE603`

用途：支撑矿下低照、粉尘、雾/水汽、眩光、噪声、弱纹理等真实退化，以及“低层图像质量提升不等于下游任务必然改善”的工程动机。

其中建议重点保留：

- `RE049`：真实煤矿低照、粉尘和眩光会诱发高置信错误；其 calibration / risk-coverage 评价可作为 MMFR reliability evaluation 的参考。DOI: 10.1016/j.aei.2026.104702。
- `RE266`：固定低照增强虽提高总体 mIoU，但会放大噪声并损害部分安全关键类别，可支撑“恢复必须由下游任务验证，平均指标可能掩盖局部失败”。DOI: 10.3390/s26010141。
- `RE418`：使用煤矿专用无参考质量分数驱动逐图像自适应增强参数搜索，直接体现“quality evidence → adaptive action”；同时说明错误的质量代理可能导致错误处理。DOI: 10.1109/TIM.2024.3470234。

### 2.2 矿下多模态 / 语义感知背景

`RE081` `RE086` `RE120` `RE131` `RE171` `RE194` `RE233` `RE257` `RE263` `RE304` `RE343` `RE360` `RE365` `RE376` `RE419` `RE426` `RE460` `RE465` `RE535` `RE551` `RE554` `RE564` `RE579` `RE586` `RE606` `RE609` `RE613` `RE663` `RE679`

用途：支撑地下空间对多传感器、深度感知、语义分割、检测与导航的实际需求，并说明不同传感器在矿下复杂环境中具有互补性。

其中：

- `RE360` 为红外—可见光矿区裂缝语义分割，证明多模态互补和下游任务评价的重要性，但其融合为固定策略，不含 reliability-aware 动态模态控制。DOI: 10.1016/j.jrmge.2025.03.045。
- `RE465` 使用 RGB-D 进行矿工检测与距离估计，并用轨迹信息过滤低置信检测，可作为“矿下 RGB-D 实际应用 + 输出置信度处理”的背景，但并非 Depth reliability-conditioned semantic fusion。

### 2.3 动态加权 / 系统鲁棒性旁证

`RE143` `RE153` `RE386` `RE392` `RE413` `RE417` `RE428` `RE452` `RE549` `RE561` `RE574` `RE646` `RE667`

用途：支撑“不同传感器可靠性随环境变化，固定权重存在局限；动态权重应建立在可验证质量/残差信号之上”。

其中 `RE452` 最值得保留：其以 LiDAR 配准残差和视觉匹配内点率作为质量代理，动态调整 factor weight，并结合一致性检查隔离异常观测，构成较完整的“quality evidence → reliability judgment → reject/down-weight → multimodal fusion”链条。任务仍为 SLAM，可靠性粒度为观测/factor 级，而非像素级语义可靠性。DOI: 10.1016/j.isprsjprs.2024.08.007。

## 三、建议整体移出 MMFR 主合集的主题

以下主题中未被上述清单点名的条目，默认移入独立资料夹或归档，不再放在 MMFR 直接相关文献中：

1. 点云分割、补全、注册、变形监测；
2. 3D 重建、数字孪生、Gaussian Splatting 与一般语义建图；
3. LiDAR/IMU/UWB/visual SLAM、定位、标定和导航，且不涉及可解释质量驱动的动态融合；
4. 外来物、人员、矿卡、煤矸检测，且不含可信多模态融合机制；
5. 纯 RGB 增强、除雾、去尘或亮度调整，且无下游任务可靠性证据；
6. 岩体裂隙、螺栓、崩落、表面结构量测，与 RGB-D 语义分割失效无直接联系。

典型需移出的条目包括：`RE003` `RE017` `RE022` `RE040` `RE042` `RE044` `RE053` `RE063` `RE070` `RE073` `RE076` `RE095` `RE099` `RE111` `RE118` `RE122` `RE124` `RE129` `RE136` `RE138` `RE141` `RE150` `RE168` `RE172` `RE178` `RE188` `RE189` `RE195` `RE209` `RE214` `RE222` `RE239` `RE246` `RE248` `RE251` `RE278` `RE284` `RE289` `RE292` `RE293` `RE297` `RE305` `RE307` `RE309` `RE336` `RE344` `RE356` `RE359` `RE361` `RE377` `RE382` `RE395` `RE398` `RE400` `RE408` `RE409` `RE410` `RE412` `RE423` `RE425` `RE431` `RE433` `RE439` `RE446` `RE449` `RE462` `RE497` `RE501` `RE509` `RE512` `RE519` `RE521` `RE525` `RE530` `RE545` `RE550` `RE553` `RE555` `RE557` `RE568` `RE569` `RE570` `RE575` `RE582` `RE589` `RE591` `RE593` `RE595` `RE596` `RE607` `RE616` `RE618` `RE621` `RE623` `RE624` `RE627` `RE628` `RE629` `RE631` `RE633` `RE635` `RE636` `RE637` `RE641` `RE642` `RE643` `RE648` `RE649` `RE652` `RE653` `RE655` `RE660` `RE661` `RE662` `RE668` `RE690` `RE696` `RE698` `RE700` `RE702` `RE705` `RE707` `RE713` `RE716`。

## 四、已完成全文复核的重点 RE

### P0：直接影响 novelty 判断

- `RE240` — **已核**：局部 BEV confidence map + MoE/Fuzzy 动态模态权重 + 跨模态补偿；未发现 fixed/equal、shuffle、inverse、oracle 等严格 reliability 因果对照，也未覆盖完整模态缺失。
- `RE094` — **已核**：reliability 为 sample/frame-level global scalar；由 intensity saturation / local contrast 与 mean DoLP 构造，作用于输入级 channel gating，并非 pixel-level confidence map。
- `RE026` — **已核**：粉尘浓度为局部图，融合属于“全局权重 + 局部修正”；reliability/权重由物理先验和解析公式产生，不是学习式 estimator。

### P1：矿下可靠感知论述

- `RE049` — **已核**：重点用于 prediction calibration、ECE/MCE 与 risk-coverage 评价参考，不属于 modality reliability。
- `RE266` — **已核**：用于固定增强可能产生负迁移、类别级退化与任务驱动评价证据；不是 RGB-D 方法。
- `RE360` — **已核**：用于矿区多模态互补与任务驱动融合背景；没有 reliability estimator 或动态模态 gate。
- `RE418` — **已核**：用于“quality evidence → adaptive enhancement action”以及 quality proxy 必须经过实际任务验证的旁证；quality 为全图 handcrafted scalar。
- `RE452` — **已核**：用于地下多传感器 quality/residual-driven adaptive weighting 的方法学旁证；不是 learned / pixel-level semantic reliability。

当前这 8 篇重点 RE 已无“待补全文”项，可直接作为后续研究蓝图和模块设计阶段的 RE 侧参考。

## 五、整理与引用时的硬规则

RE 文献主要支撑以下四类论述：

1. **矿下有哪些真实退化**：低照、非均匀照明、粉尘、雾/水汽、眩光、弱纹理、传感器质量波动等；
2. **为什么需要多模态**：单一传感器在复杂地下环境中存在条件性失效，多传感器具有互补价值；
3. **为什么需要质量/可靠性条件化处理**：固定参数、固定权重或无条件融合难以适应时变环境；
4. **可靠性必须单独验证**：任务精度提升不能自动证明 reliability 正确，应检查 calibration、utility 对应关系以及 reliability→action 的因果控制。

RE 文献不应单独支撑“MMFR 结构新颖”。结构 novelty 仍需与 PR 侧核心机制文献（如 PR089、GeomPrompt、CCF、QMF、MaskMentor、ECoLaF 等）联合比对。
