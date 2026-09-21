# MMFR v4.1 文献全文审计（实施级）

> **文档角色：** 本文件是 MMFR v4.1 在 E1 实现前的文献全文审计，不是实验报告、代码说明或执行授权书。  
> **审计状态：** 已完成本轮允许范围内的五篇核心正文、DFormerv2 正式 supplemental、两个条件补充和指定本地接口的结构化复核；仍把来源元数据、未提供的 supplemental-only 细节和本地数据量纲列为待核项。  
> **日期：** 2026-09-20。  
> **实时入口：** [`MUSeg-current-status.md`](../../../../doc/main/MUSeg-current-status.md)。本文件不替代实时入口。  
> **授权边界：** 本轮只读蓝图、索引、登记表、S13 审计清单、允许的论文/补充材料和四个本地接口文件；没有修改代码，没有训练、GPU/云任务、评估、official test 或提交/推送。本文件本身不授权 E1 实现或运行。
> **本地资料边界：** 论文全文与提取产物保留在 Git 忽略的 `liu-test-exp/MMFR/附件/`，只在当前机器存在时可直接读取；本文件保存可携带的审计结论与来源身份。下文附件位置均以 plain-text local-only 路径记录，不表示 Git 仓库包含论文全文。
>
> **大白话：** 这份审计把“论文原来怎么做”“哪些思想可以借用”“哪些地方和 DFormerv2-S+A2 对不上”“本项目准备怎样适配”“哪些参数还必须由上级审核冻结”分开。读完后可以知道下一步该冻结哪些合同，但不能把任何候选方案当成已经实现或已经有效。

## 0. 审计口径与取得结论

### 0.1 固定版本、来源和本地取得状态

“取得”在本表中分成两层：一是正文或正式 supplemental 是否已在允许目录中取得并实际读取；二是 DOI、PDF 哈希、官方代码 commit 等来源元数据是否已经独立核验。后者没有证据时不补造。

| 编号 | 固定版本/来源范围 | 允许目录中的材料 | 正文/补充取得状态 | 来源元数据与版本边界 |
| --- | --- | --- | --- | --- |
| **AI023** | GeomPrompt / GeomPrompt-Recovery，`arXiv:2604.11585` | 正文、公式摘录、图表与 figure mapping | **正文已取得并审计** | 本文件按该 arXiv 标识审计；论文项目页只作为入口，不把代码可用性或 commit 写成事实。 |
| **AI024** | Condition Dropout，蓝图固定为 `arXiv:2607.20326v1` | 正文、公式摘录、图表与 figure mapping | **正文已取得并审计** | 本文件按 v1 审计；论文正文只说代码接受后公开，未取得可核对的 commit。 |
| **AI019** | MaskMentor，`DOI: 10.1145/3664647.3681698` | ACM/MM 论文正文、图表与 figure mapping | **正文已取得并审计** | DOI 来自固定索引和正文元数据；论文给出代码入口，但本轮未核对具体 commit。 |
| **PR090** | RobustSeg，**只采用 CVPR 2026 正式版范围** | 本地 RobustSeg 正文、公式摘录、图表与 figure mapping | **正式版正文已取得并审计**；允许目录中未见独立 PR090 supplemental 文件 | 本文件不拼接相关作者 arXiv 预印本；正式版 DOI、PDF 哈希和 supplemental-only 细节仍列为待核，不猜造 DOI。 |
| **PR070** | DFormerv2，CVPR 2025 正文与正式 supplemental | 正文、正式 supplemental、图表与 figure mapping；本地 DFormerv2 代码 | **正文、正式 supplemental 和指定本地接口均已取得并审计** | 论文索引给出 `DOI: 10.1109/CVPR52734.2025.01802`；本地代码事实只归于本地代码，不反写为论文原设。 |
| **AI017**（条件补充） | Calibrated RGB-D Salient Object Detection，`DOI: 10.1109/CVPR46437.2021.00935` | 正文、图表与 figure mapping | **正文已取得并审计** | 只用于判断 R-lite 是否触发条件阻塞；不把其 SOD 结果迁移为 MUSeg 结果。 |
| **MoSA**（条件补充） | Modality-Aware Spatially-Adaptive Adaptation，附件正文标示 `DOI: 10.1109/ACCESS.2026.3694496` | 正文、公式、图表与 figure mapping | **正文已取得并审计** | 正文页眉给出 DOI；蓝图此前记录题名—索引入口仍需区分，本文件只按附件正文的模块和 DOI 记载，不补充外部版本。 |

### 0.2 证据标签

- **[原论文]**：来自上述本地正文、公式摘录、正式 supplemental 或论文中明确的图表/章节。
- **[本地代码]**：来自允许只读的 `models/encoders/DFormerv2.py`、`models/builder.py`、`utils/train.py`、`utils/init_func.py`。
- **[蓝图]**：来自同目录 `../01_research/MMFR_research_blueprint_v4_1_2026-09-20.md`、reference index/registry 或 `source_materials/S13_literature_audit_checklist.md`，只能支持项目边界和来源索引，不能替代原文。
- **[设计—待上级审核冻结]**：本项目的适配提议、候选参数、比较合同或默认选择，不是任何论文的原设。
- **[待核项]**：本轮材料不足以定案的版本、量纲、代码、训练或部署信息。

以下每篇核心文献均固定使用六个互不混写的栏目：**原论文做法、可直接借用、与 DFormerv2-S+A2 不兼容、本项目适配、[设计—待上级审核冻结]、待核项**。

## 1. 本地 DFormerv2-S+A2 接口基线

这一节只记录本地代码事实，用来限定后面“适配”可以落在哪里；它不把当前代码新增的 A2 诊断分支归给 PR070 原论文。

| 本地接口 | 只读核验结果 | 实施含义 |
| --- | --- | --- |
| DFormerv2-S 变体 | [`DFormerv2.py`](../../../../models/encoders/DFormerv2.py) 中 `DFormerv2_S` 使用 `embed_dims=[64,128,256,512]`、`depths=[3,4,18,4]`、`num_heads=[4,4,8,16]`；对应 builder 通道为 `[64,128,256,512]`。 | 本项目的 S 规模不是论文正文中可任意替换的抽象 backbone；新增模块必须服从四级输出和既有 decoder 接口。 |
| RGB/Depth 输入路径 | `dformerv2.forward` 先对 RGB 做 `patch_embed`；Depth 输入 `x_e` 只取第一个通道，然后传给各 `BasicLayer`。 | 本地实现没有独立的 Depth encoder；Depth 的主要消费路径是 geometry prior。 |
| Geometry prior | `GeoPriorGen.forward` 对 Depth 使用 `F.interpolate(..., mode="bilinear", align_corners=False)`，再构造 depth decay；`RGBD_Block` 每个 block 调用它。 | 论文的 patch average pooling 与本地 checkpoint 对应代码的 bilinear interpolation 必须分开写，不能把二者当成同一实现。 |
| GSA 计算位置 | 本地 `Full_GSA`/`Decomposed_GSA` 把 geometry mask 加到 attention logits 后再 softmax；论文 Eq. (4) 的形式是 softmax 后与 $\beta^G$ 做逐元素乘法。 | PR070 的正式算子和当前本地实现存在明确的实现边界；任何新动作都必须针对实际本地路径说明。 |
| Encoder→decoder 契约 | [`models/builder.py`](../../../../models/builder.py) 的 `encode_decode` 接收 backbone 的多级 tuple，再交给 decoder；输出恢复到输入空间使用 bilinear interpolation。`ham` decoder 使用后三级通道。 | **[设计—待上级审核冻结]** F-lite 的自然候选插入点是 backbone stage 输出到 decoder 之间，但具体 stage、形状和 decoder 适配尚未冻结。 |
| A2 reliability head | `EncoderDecoder.forward` 只在训练传入完整 A2 supervision 时增加 reliability auxiliary loss；`reliability_auxiliary_loss` 的输出不回馈 backbone、geometry prior 或 decoder。 | A2 头不是现成的 geometry gate。**[设计—待上级审核冻结]** F-lite 第一版不接 reliability predictor，避免把 A2 诊断、空间 gate 和 adapter 适应混成一个变量。 |
| 参数初始化 | backbone `_init_weights` 对 Linear 使用截断正态、对 LayerNorm 使用单位权重/零偏置；GSA 的 q/k/v 与 out projection 在各自 `reset_parameters` 中采用 Xavier/零偏置；`GeoPriorGen.weight` 以全 1 参数开始。 | 新增模块的初始化不能写成“沿用论文”除非论文确有同一初始化；新增 adapter 的精确初始化属于 [设计—待上级审核冻结]。 |
| 当前训练优化器 | [`utils/train.py`](../../../../utils/train.py) 通过 `group_weight` 生成 decay/no-decay 两组，再按 `config.optimizer` 构造 AdamW/SGD；学习率按 `WarmUpPolyLR` 和 `config.nepochs` 控制。[`utils/init_func.py`](../../../../utils/init_func.py) 的 `configure_optimizers` 是另一套防御式分组函数，但训练脚本当前使用 `group_weight`。 | E1 不能直接照搬论文 optimizer 参数；应在子计划中明确新增参数是否进入当前 param groups，以及主干是否冻结。 |

### 1.1 PR070 论文平均池化与本地 bilinear 的强制边界

PR070 正文 §3.1、Eq. (1) 明确写的是：对每个 depth patch 内像素做 **average pooling** 得到 patch 深度 $z_{ij}$，再计算成对深度差。正文 §3.3 又写四个 stage 对 Depth 做不同 kernel/stride 的 average pooling。正式 supplemental 的配置表只给 stage 尺度和模型设置，没有把本地 fork 的插值代码改写成论文公式。

本地 checkpoint 对应的代码路径是 `GeoPriorGen.forward`：先把 `depth_map` 用 `F.interpolate` 的 `mode="bilinear"` 调整到当前 stage 的 $H\times W$，然后由 `generate_depth_decay` 或一维版本计算差值。因此：

1. **论文原设**是 patch average pooling；
2. **本地当前实现**是 bilinear interpolation 后的 depth grid；
3. 本文件以及后续 E1 子计划不得把本地 bilinear 写成“忠实复现论文 average pooling”；
4. 如需复现实验算子，必须另行决定是跟论文公式、跟当前 checkpoint、还是新增一组明确标识的算子对照；这不是本文件代替上级冻结的决定。

## 2. AI023 — GeomPrompt / GeomPrompt-Recovery

### 2.1 固定来源与原文锚点

- 正文（本机资料，不纳入 Git）：`../../附件/Jaganathan和Vela - 2026 - GeomPrompt Geometric prompt learning for RGB-D semantic segmentation under missing and degraded dep/Jaganathan和Vela - 2026 - GeomPrompt Geometric prompt learning for RGB-D semantic segmentation under missing and degraded dep.md`。
- 公式摘录（本机资料，不纳入 Git）：`../../附件/Jaganathan和Vela - 2026 - GeomPrompt Geometric prompt learning for RGB-D semantic segmentation under missing and degraded dep/Formula/Jaganathan和Vela - 2026 - GeomPrompt Geometric prompt learning for RGB-D semantic segmentation under missing and degraded dep_formula.md`。
- 方法锚点：§3.1 `Prompting Framework`；§3.2 `GeomPrompt Architecture`；§3.3 `GeomPrompt-Recovery Architecture`；§3.4 `Training Objective and Protocol`；Eq. (1)–(3)；Fig. 1–2。
- 训练/实验锚点：§4.1 `Implementation Details`；§4.2 `Experimental Setup`；§4.3 `RGB Geometric Prompting`；§4.4 `GeomPrompt-Recovery Under Depth Failures`；§4.5.1–§4.5.3；Table 1–4；Fig. 3–7。
- 代码/版本锚点：正文只给出项目页 `https://geomprompt.github.io`；本轮未取得可核对的官方代码 commit，不能把项目页写成代码已公开事实。

### 2.2 实施字段

| 字段 | [原论文]核验 |
| --- | --- |
| 输入与输出 | GeomPrompt 输入 ImageNet-normalized RGB $x\in\mathbb{R}^{3\times H\times W}$，ViT-S/16 加 CNN decoder 输出低分辨率 residual，再得到单通道 prompt 并复制成三通道；GeomPrompt-Recovery 输入 RGB 与 degraded Depth，输出 bounded correction，直接加到 corrupted Depth 后得到 prompt。两者最终都把 prompt 送入冻结 RGB-D segmenter。 |
| 结构与插入点 | GeomPrompt：ViT-S/16 去掉 prefix token，重排为二维网格；先 $\times2$ bilinear upsampling，再两个 $3\times3$ Conv-BN-ReLU 和 $1\times1$ 单通道 projection，残差先在 $H/8\times W/8$ 预测，再 anti-aliased upsample 到全分辨率；PromptAdapter 为 $1\times1$、$3\times3$、$1\times1$ 的 3→16→16→3 残差模块，之后做 hard low-pass projection。Recovery 另有四层 stride-2 depth-condition CNN，并与 RGB 特征 concat 后用 $1\times1$ 融合。 |
| 输出量纲与公式 | GeomPrompt Eq. (1)：$\Delta_{full}=\mathcal U(\Delta)$，$p_{raw}=127.5+s\tanh(\Delta_{full})$，再归一化、PromptAdapter 和低通；Recovery Eq. (2)：$corr=s\tanh(\Delta_{full})$，$p_{raw}=\operatorname{clamp}(\tilde d+corr,0,255)$。论文明确这是 task-relevant geometry-like prompt，不是 metric depth。 |
| loss 与正则 | Eq. (3)：OHEM cross-entropy segmentation loss 加 TV smoothness 和低分辨率 residual 的 L1 magnitude penalty：$\mathcal L=\mathcal L_{seg}+\lambda_{tv}\mathcal L_{tv}+\lambda_\delta\|\Delta\|_1$。 |
| 初始化与冻结 | 冻结 RGB-D segmenter；prompt 生成模块训练。PromptAdapter 最后一层 zero-initialized；Recovery decoder correction head zero-initialized，训练初始接近输入 corrupted Depth 的 identity。ViT encoder 默认需要训练，冻结 ViT 的消融下降 3.5 mIoU。 |
| optimizer、参数和训练长度 | 300 epochs；AdamW；poly power 0.9；10 warmup；weight decay 0.01；encoder LR $3\times10^{-5}$、decoder LR $1\times10^{-4}$；effective batch size 32；8×A40；$\lambda_{tv}=10^{-5}$、$\lambda_\delta=5\times10^{-4}$；$s$ 从 15 线性升到 80。Recovery 训练样本以 0.2 概率 clean，否则从 quantize/hole/dropout/noise/blur/banding/scale shift 中均匀选一种，severity 在 [0.10,0.90]。 |
| 推理、对照和成本 | 推理只需要 RGB（GeomPrompt）或 RGB+degraded Depth（Recovery）和冻结 segmenter；不需要 Depth GT。对照包括 GT Depth、RGB-only zero depth、DA2、Metric3Dv2、手工 pseudo-depth、不同 degradation severity，以及关闭 PromptAdapter/low-pass/regularizer/curriculum/ViT adaptation。Table 3 报 prompt 侧 7.8 ms、44.0 GFLOPs、23.4M 参数（单 A40、batch 1、fp16）；这不是 DFormerv2-S+A2 全模型成本。 |

### 2.3 原论文做法

1. 论文把“missing Depth”和“degraded Depth”分成两种输入模式，不能把 GeomPrompt 的 RGB-only prompt 与 GeomPrompt-Recovery 的 RGB+corrupted-Depth residual 混成一个模块。
2. missing 模式以 127.5 为中心的 bounded residual 生成 prompt；degraded 模式在输入 corrupted Depth 上加 bounded residual，并 clamp 到 [0,255]。
3. 论文冻结 segmenter，而不是冻结 prompt 生成器；其 ViT-S/16 是 prompt 分支的一部分。
4. 训练只用 segmentation supervision，但加了 OHEM CE、TV 和 L1；没有 Depth GT reconstruction loss。
5. Recovery 的部署需要其额外 depth-condition encoder，且其输入是三通道复制后的 degraded Depth。

### 2.4 可直接借用

- **任务驱动而非 metric-depth 复原的判断：** 可以借用“只要求输入对分割有用，不把替代几何称为真实深度”的研究口径。
- **bounded residual 和 identity 起点：** 可以借用 residual correction、输出投影 zero-init 以及先保持原输入再学习修正的工程思想。
- **评估方式：** 可以借用 clean/缺失/受损分层、GT/zero-depth/手工 proxy 对照和推理成本单独报告的结构。
- **不能直接借用的量纲：** 0–255、127.5、[0,255]、$s=15\rightarrow80$ 和论文归一化统计只能作为来源记录，不能作为 DFormerv2 本地默认值。

### 2.5 与 DFormerv2-S+A2 不兼容

- **输入通道语义不兼容：** GeomPrompt 的 prompt 是可复制到三通道 geometric input 的连续图；本地 DFormerv2-S 的 `modal_x` 在 `dformerv2.forward` 中只取第一个通道进入 geometry prior。
- **量纲不兼容：** GeomPrompt 的 0–255/127.5 量纲不能未经核验进入本地 checkpoint；本地 Depth 的数据预处理、归一化和 invalid sentinel 不能由论文表格反推。
- **Recovery 结构不兼容：** GeomPrompt-Recovery 复制了一个 depth-condition CNN；DFormerv2 没有独立 Depth encoder，Depth 主要进入 `GeoPriorGen`，因此不能把 Recovery 直接称为本地复现。
- **冻结边界不兼容：** 论文冻结整个 segmenter；本项目 R/F/T 需要和 A2 的继续训练、matched control 及本地 decoder 预算分开登记。
- **算子边界不兼容：** 本地 geometry prior 是 bilinear-resized depth grid 加 logits mask，论文 prompt 侧的 upsampling、low-pass 和 segmenter 输入契约不是本地 geometry 算子的同义替换。

### 2.6 本项目适配

- **[设计—待上级审核冻结] R-lite 的角色：** 只把 GeomPrompt 的“任务有用替代几何”作为 R 路线来源，不声称复现 GeomPrompt。R-lite 当前建议只处理推理时可判定的**整幅 Depth missing**；输入不属于整幅 missing 时精确 bypass，不能把局部 synthetic missing 自动扩展到 natural-invalid 区域。此边界是本项目适配口径，不是 AI023 原设。
- **[设计—待上级审核冻结] 接口方向：** 适配目标应是给本地 DFormerv2 的 geometry prior 入口提供一个可用的 Depth-like 输入，或在明确批准后直接改变 geometry 输入；不能未经冻结把 RGB prompt 接到 decoder logits 或把 A2 reliability head当作恢复器。
- **[设计—待上级审核冻结] Recovery 与 prompt 必须分开记账：** RGB-only prompt 和 RGB+degraded-Depth recovery 是两个候选，不以一次实现同时代表两种论文模式。

### 2.7 [设计—待上级审核冻结]

- R-lite 的输入检测、替代几何输出范围、归一化、是否保留原 Depth、插入 `GeoPriorGen` 前还是替代 Depth grid、网络宽度、激活、最后投影初始化。
- R-lite 是否只使用 segmentation loss，是否增加 TV/L1 或本地关系约束；所有 loss 权重、residual 上限、warmup、optimizer 参数组和训练长度。
- 是否冻结 DFormerv2-S+A2 主干、是否只训练 R 分支，以及 clean-control 的 checkpoint/selector 规则。
- 任何使用 0–255、127.5、Recovery 的额外 depth-condition encoder 或 local low-pass 的方案，均不得在审核前写成默认实现。

### 2.8 待核项

- 本地 MUSeg/DFormer 数据管线中 `modal_x` 的真实数值范围、dtype、invalid sentinel 与缩放后语义。
- R-lite 是否能仅凭当前推理输入可靠判定“整幅 Depth missing”；不能读取 clean Depth、真实故障 severity 或 future privileged mask。
- 若未来采用 recovery，是否有与本地输入契约一致的训练 corruptions，以及有效区域保护规则。
- AI023 官方代码是否公开、哪个 commit 对应正文，当前没有本轮证据，不作结论。

## 3. AI024 — Condition Dropout（ConD）

### 3.1 固定来源与原文锚点

- 正文（本机资料，不纳入 Git）：`../../附件/Zhu 等 - 2026 - Toward reliable RGB-D semantic segmentation Handling missing modalities via condition dropout/Zhu 等 - 2026 - Toward reliable RGB-D semantic segmentation Handling missing modalities via condition dropout.md`。
- 公式摘录（本机资料，不纳入 Git）：`../../附件/Zhu 等 - 2026 - Toward reliable RGB-D semantic segmentation Handling missing modalities via condition dropout/Formula/Zhu 等 - 2026 - Toward reliable RGB-D semantic segmentation Handling missing modalities via condition dropout_formula.md`。
- 方法锚点：§II-A `Overview of the Training Paradigm`；§II-B `Modality-Stochastic Input Strategy`；§II-C `Zero Convolution Feature Injection`；§II-D `Training Objective and Stability`；Eq. (1)–(4)；Fig. 1–2。
- 实验锚点：§III-A–§III-G；Table I–II；Fig. 3–4。特别关注 §III-B 的实现说明、§III-F 的四组消融和 §III-G 的额外 encoder 成本限制。
- 代码/版本锚点：正文称 code will be made publicly available upon acceptance；本轮没有取得可核对官方代码或 commit。

### 3.2 实施字段

| 字段 | [原论文]核验 |
| --- | --- |
| 输入与输出 | 以已经训练好的 RGB-D segmenter 为 Stage 1；Stage 2 在 complete、RGB-missing、Depth-missing 三种输入之间随机采样，缺失模态在实践中用与输入同形状的 zero tensor 表示。输出仍是原 decoder 的 segmentation map。 |
| 结构与插入点 | 复制原 encoder 得到可训练 $E_{aux}$；冻结原 encoder $E_{pre}$ 和 decoder；$E_{aux}$ 的特征经过 zero-initialized $1\times1$ Conv，再加到 frozen encoder 的对应 feature 上。原 decoder 保留。论文强调复制的是 encoder，不是只增加一个小 adapter。 |
| loss | 使用原任务 loss；正文举例是 cross-entropy 或 panoptic quality，没有给出一个对所有实验统一的新 loss 公式或权重。 |
| 初始化 | $E_{aux}$ 从 $E_{pre}$ 拷贝权重；注入 Conv 的参数 zero-init，使 Stage 2 起点尽量保持原模型输出。 |
| 冻结/训练 | $\theta_{pre}$ frozen，$\theta_{aux}$ trainable；完整/缺失状态各以 $1/3$ 概率采样。原 decoder 不训练。 |
| optimizer、训练长度和硬件 | §III-B 只说遵循原论文的 optimizer、learning rate、schedule；2× RTX 3090、480×640、随机 flip/color jitter。正文没有给出精确 continued-training epoch 数，不能补造。 |
| 推理、对照和成本 | 推理保留 frozen encoder、aux encoder 和原 decoder；因此参数、显存和 latency 增加。Table II 对比 baseline、dropout only、copy without freeze、full ConD；Table I 报 full/missing 条件。论文 limitation 明确指出额外 encoder 也在 inference 使用。 |

### 3.3 原论文做法

1. ConD 是从完整模态 checkpoint 出发的 continued training，不要求从头训练。
2. 原 encoder/decoder 冻结，复制的 encoder 学习 partial-modal residual features，注入层 zero-init。
3. 三种 condition 等概率：完整、RGB 缺失、Depth 缺失；缺失张量是 zero tensor。
4. 方法目标是保留 full-modality 能力同时减轻 missing-modality 退化，而不是显式估计 Depth quality 或修改 geometry prior。
5. 论文的实验证据依赖 DFormer-B、Sigma-S 等模型；不能把 DFormer-B 的配置或结果直接归给 DFormerv2-S+A2。

### 3.4 可直接借用

- 继续训练、冻结既有主干和 zero-initialized residual injection 的工程动机。
- 用 full/RGB-missing/Depth-missing 三状态暴露 missing-modality 风险的对照思路。
- 通过 copy+freeze 与 dropout-only 的消融分离“见过缺失输入”和“新增可训练容量”的作用。
- 额外参数、显存和推理 latency 必须单列，不能只报 mIoU。

### 3.5 与 DFormerv2-S+A2 不兼容

- **复制范围不兼容：** 原文复制完整 encoder；本项目 F-lite 明确不复制整个 DFormerv2 encoder。
- **Depth 结构不兼容：** ConD 叙述面向可复制的 RGB/Depth encoder；本地 DFormerv2-S 没有独立 Depth encoder，Depth 只参与 geometry prior。
- **推理成本不兼容：** 原文必须保留额外 encoder 到推理；本项目首轮 F-lite 目标是轻量 stage adapter，不应承诺同一成本或同一鲁棒性。
- **训练预算不完整：** 原文没有精确 continued-training 长度；不能把旧蓝图中的额外 epoch 建议写成 ConD 作者配置。
- **A2 语义不兼容：** 本地 A2 reliability head 是训练辅助分支，不是 ConD 的 copied encoder，也没有替代原 decoder。

### 3.6 本项目适配

- **[设计—待上级审核冻结] ConD 的项目角色：** ConD 只作为 F-lite 的 residual adaptation 和 condition exposure 来源；F-lite 不是 ConD 的忠实复现。
- **[设计—待上级审核冻结] 本地适配边界：** 本地适配可利用 DFormerv2 的四级 feature tuple 到 decoder 的接口，避免复制完整 encoder；该方向是项目适配边界，不是 ConD 原文结构。
- **[设计—待上级审核冻结] 第一版范围：** F-lite 第一版不接 reliability predictor、不做空间 modulation、不复制完整 encoder，保持与 MoSA/ConD 原结构的差异可审计。

### 3.7 [设计—待上级审核冻结]

- F-lite 的插入 stage、bottleneck 通道、是否共享 adapter、激活、zero-init 的具体层和参数量。
- 是否使用完整/RGB-missing/Depth-missing 三状态、各状态比例、zero tensor 与本地 missing sentinel 的关系。
- 主干和 decoder 的冻结范围、continued-training 长度、学习率、weight decay、optimizer param groups、checkpoint/selector。
- 训练时是否保持 original corruption exposure，以及 C0 matched continuation 的步数和学生更新数。

### 3.8 待核项

- ConD 正文没有给出精确 Stage 2 epoch；需取得固定版本 supplemental 或作者实现后再决定能否做严格预算对照。
- 本地 DFormerv2-S 的全模型 copy 与显存成本未在本轮运行测量；本文件不运行模型、不估算实测 latency。
- 论文所称“inference architecture unchanged”是 decoder/interface unchanged，不等于参数和计算不增加；这一点必须在 E1 计划中单独记账。

## 4. AI019 — MaskMentor

### 4.1 固定来源与原文锚点

- 正文（本机资料，不纳入 Git）：`../../附件/Zhao 等 - 2024 - MaskMentor Unlocking the potential of masked self-teaching for missing modality RGB-D semantic segm/Zhao 等 - 2024 - MaskMentor Unlocking the potential of masked self-teaching for missing modality RGB-D semantic segm.md`。
- 方法锚点：§3.1 `Problem Setting`；§3.2 `Overview`；§3.3 `Masked Modality and Image Modeling`；§3.4 `Self-Teaching via Token-Pixel Joint Reconstruction`；Algorithm 1；Fig. 1–2。
- 训练/实验锚点：§3.5 `Overall Training Pieline`；§4.1 `Setting Up`；§4.3 `Ablation Study`；Table 1–5；Fig. 3–4。
- 代码入口：正文写明 `https://github.com/Zhao-ZD/MaskMentor`；本轮未核对具体 commit 或当前公开状态。

### 4.2 实施字段

| 字段 | [原论文]核验 |
| --- | --- |
| 输入与输出 | 预训练阶段使用 $K=3$ 个模态：RGB、Depth、semantic segmentation map；测试和 fine-tuning 面向 complete、Only-RGB、Only-Depth。预训练的 MIM heads 在 fine-tuning 时丢弃，换成随机初始化的 ConvNeXt decoder，输出 segmentation map。 |
| 结构 | M²IM 先做 modality-level mask，再对保留模态做 patch-level mask；至少保留一个模态。STTP 中共享参数的 encoder 同时作 teacher/student：teacher 输入 complete modalities，student 输入 modality-missing data；teacher 提供 token-level supervision，二者还做 pixel-level reconstruction。 |
| loss 与监督位置 | MIM 使用 pixel-level reconstruction；STTP 增加 teacher token reconstruction，正文说明 token-level 使用 cosine similarity，pixel-level 遵循 MultiMAE。它不是简单 logits KD，也不是只在 segmentation logits 上做蒸馏。 |
| teacher 更新方式 | teacher/student 参数共享，不保存一套独立 teacher；每个 batch 先更新 teacher 3 iterations，再更新 student 1 iteration。 |
| 初始化与训练长度 | ViT-B encoder；其他网络参数随机初始化。约 100 epochs MIM warmup，400 epochs STTP，再进行 500 epochs segmentation fine-tune。modality mask rate $\phi_m=0.5$。 |
| optimizer 与 batch | 预训练 AdamW，初始 LR $1\times10^{-5}$，cosine，batch 12；fine-tuning 初始 LR $3\times10^{-5}$，cosine，batch 2。patch size、输入分辨率等沿用 MultiMAE。 |
| 推理、对照和成本 | 推理只保留一个训练好的模型，不需要独立 teacher；对照包括 direct fine-tune、MIM、M²IM、M²IM+STTP、separate teacher/student、pixel-only/token-only MIM、不同 missing modality。模型部署没有额外 teacher 参数，但预训练阶段有多视图和 teacher/student 计算。 |

### 4.3 原论文做法

1. M²IM 直接在训练时模拟 missing modality，并同时做 patch mask；随机排序模态以平衡被 mask 的模态。
2. STTP 让 complete-input teacher 指导 missing-input student，但 teacher/student 共享参数，且监督目标是 token+pixel reconstruction。
3. 论文把 MIM 预训练与 downstream segmentation fine-tuning 分开，MIM heads 不进入推理。
4. 论文并非 RGB-D DFormerv2 级别的 geometry prior 修正，也没有以 segmentation output logits 为唯一迁移对象。

### 4.4 可直接借用

- complete view 指导 missing/corrupt view 的训练动机。
- 需要用 paired views 和同场景条件暴露来区分“见过缺失输入”和“teacher guidance”的必要性。
- teacher 不一定在推理保留，推理成本和训练成本应分开报告。
- missing-modality 训练收益不能直接写成 geometry-specific novelty。

### 4.5 与 DFormerv2-S+A2 不兼容

- **模态数不兼容：** MaskMentor 预训练的第三个输入是 semantic segmentation map；本地 DFormerv2-S+A2 的模型输入是 RGB+Depth，不能把 label map 放入普通推理路径。
- **token 长度不兼容：** MaskMentor 依赖可变长度 Transformer 输入和多 MIM heads；DFormerv2 是固定四级 pyramid encoder。
- **结构不兼容：** DFormerv2 的 Depth 不经过独立编码器；M²IM 的跨模态重建结构不能直接映射到 `GeoPriorGen`。
- **loss 不兼容：** token/pixel reconstruction 不是 output-level KD，不能把 MaskMentor 直接称为 T-lite 复现。
- **预算不兼容：** 100+400+500 的三段训练是 ViT-B/MIM 配方，不是 DFormerv2-S+A2 的继续训练长度。

### 4.6 本项目适配

- **[设计—待上级审核冻结] T-lite 借用范围：** T-lite 只借用 complete→corrupt/missing 的迁移动机和 paired-view 对照逻辑。
- **[设计—待上级审核冻结] 第一版结构：** 本项目第一版只考虑 output-level student logits guidance；不引入 M²IM、patch mask、token reconstruction、semantic-map input 或共享 teacher/student MIM head。
- **[设计—待上级审核冻结] Cpair 对照：** Cpair 用来匹配 clean/corrupt 视图暴露和学生更新数，避免把额外数据视图误算成蒸馏收益。

### 4.7 [设计—待上级审核冻结]

- T-lite 的 teacher 来源（例如固定 clean checkpoint 或同一模型的 full-input branch）、是否 stop-gradient、teacher 更新/冻结和学生更新次数。
- 输出 KD 的温度、loss 权重、是否只蒸馏 valid pixels、teacher 错误过滤、clean/corrupt CE 的具体组合。
- T-lite 的训练 epoch、batch、optimizer、checkpoint/selector 和额外 teacher forward 成本。
- 是否只覆盖整幅 Depth missing，还是同时覆盖局部 corruption；这必须与 R/F 的 protocol identity 分开登记。

### 4.8 待核项

- 原文正文没有给出可直接迁移到 DFormerv2 logits 的 segmentation fine-tuning loss 全部实现细节；不能把 reconstruction 公式转换成默认 KD 权重。
- 论文代码入口存在，但具体 commit、配置文件和 checkpoint 身份未核验。
- MIM 与 STTP 的 exact wall-clock/FLOPs 在正文中没有统一表格，不能用预估值代替成本测量。

## 5. PR090 — RobustSeg（只按 CVPR 2026 正式版）

### 5.1 固定来源与原文锚点

- 正文（本机资料，不纳入 Git）：`../../附件/Tan 等 - 2026 - Towards robust multi-modal semantic segmentation with teacher-student framework and hybrid prototype/Tan 等 - 2026 - Towards robust multi-modal semantic segmentation with teacher-student framework and hybrid prototype.md`。
- 公式摘录（本机资料，不纳入 Git）：`../../附件/Tan 等 - 2026 - Towards robust multi-modal semantic segmentation with teacher-student framework and hybrid prototype/Formula/Tan 等 - 2026 - Towards robust multi-modal semantic segmentation with teacher-student framework and hybrid prototype_formula.md`。
- 版本约束：本节只记录本地附件所对应的 **CVPR 2026 正式版范围**；相关作者 arXiv 版本不作为本节证据，不把两个版本的模块、超参数或补充实验拼接。
- 方法锚点：§3.1 `Overall Framework`；§3.2 `Hybrid Prototype Distillation`；§3.3 `Feedback Strategy`；Eq. (1)–(9)；Fig. 1–5。
- 训练/实验锚点：§4.1 `Experiment Setup`；§4.2；§4.3；§5.1–§5.4；Table 1–4；Fig. 6–8。
- 版本元数据：允许目录中未提供独立 PR090 supplemental 文件；正式 DOI、PDF 哈希和 supplemental-only 内容列为待核，不猜造。

### 5.2 实施字段

| 字段 | [原论文]核验 |
| --- | --- |
| teacher/student 输入 | teacher 使用 full-modality input；student 使用 Anymodal Dropout 后的 partially missing input。各模态有对应 encoder，抽取四级 feature；teacher/student 通过 segmentation head 输出预测。 |
| 基础 loss | Eq. (1)：$\mathcal L_{origin}=\mathcal L_{CE}+\lambda\mathcal L_{KL}$；CE 监督 student，KL 做 teacher/student logits distillation。 |
| HPD 结构 | CPD：GT nearest-neighbor resize 到 feature size，按类别平均池化形成 class-wise prototypes（Eq. 4），样本级随机 modality matching 后做 prototype KL（Eq. 5）。IFV：由 prototype 构造 CFM（Eq. 6），生成 intra-class feature variation map，再对 dominant modalities 做 IFV distillation（Eq. 7）。总 HPD loss 为 Eq. (8)：$\mathcal L_{hp}=\mathcal L_{origin}+\alpha\mathcal L_{cp}+\beta\mathcal L_{ifv}$。 |
| feedback | 先用 ASM 区分 dominant/non-dominant modalities；teacher 的 dominant modality encoders 和 fusion block 冻结，只训练 non-dominant modality encoders；feedback loss 为 CE+IFV（Eq. 9）。这是 teacher 更新，不是简单冻结 teacher。 |
| 初始化、optimizer、训练长度 | M-SegFormer/CM-NeXt；1024×1024；预训练 AdamW、LR $6\times10^{-5}$、warmup 10、poly power 0.9、200 epochs。teacher/student 同初始化并同时训练 120 epochs；student LR $6\times10^{-5}$，teacher LR $6\times10^{-6}$。超参数消融给出 $\lambda=50$、$\alpha=100$、$\beta=12$。 |
| 推理、对照和成本 | 评估包括 Anymodal、EMM、RMM、NM，覆盖 full/missing/noisy。主文比较 M-SegFormer、CM-NeXt、AnySeg、MAGIC 等，并做 basic/HPD/HPD+Feedback 对照。论文声称不增加模型参数，但训练同时维护 teacher/student、prototype/IFV 和 feedback 计算；正文没有统一 latency/FLOPs 表，exact training overhead 待核。 |
| 推理保留信息 | 正文明确 student 面向 missing-modality 学习，但没有在可见正文中给出一个单独的 deployment graph，明确说明推理时是否只保存 student、是否保留 teacher 或哪些辅助头。因此不能把“推理只保留 student”写成已核事实。 |

### 5.3 原论文做法

1. RobustSeg 不是单一 output KD，而是 logits CE/KL、cross-modal prototype distillation、dominant-modality IFV distillation 和 feedback 的组合。
2. teacher full-modality、student incomplete-modality；二者同初始化、同时训练，teacher 以更小 LR 更新，不是完全冻结 teacher。
3. 其关键特征依赖多模态独立 encoder、四级 per-modality features、GT class prototypes 和 ASM dominant-modality selection。
4. 论文给出的 missing/noisy benchmark 和 matched backbone 对照是方法论参考，但它们不是 MUSeg 或 DFormerv2-S+A2 的结果。

### 5.4 可直接借用

- complete teacher 指导 corrupt/missing student 的基本迁移动机。
- 以 matched student exposure、相同 segmentation supervision 和单独 teacher 计算成本来分离蒸馏收益。
- 将 full-modality 与 missing/noisy 条件同时报告，避免只报最有利的失效条件。
- 仅可把 Eq. (1) 的最简 CE+logits-KL 作为 T-lite 的候选起点；这仍是项目简化，不能称 RobustSeg 复现。

### 5.5 与 DFormerv2-S+A2 不兼容

- **独立 modality encoder 不存在：** 本地 DFormerv2-S 只有 RGB backbone，Depth 通过 geometry prior 使用；无法直接获得 RobustSeg 所需的 per-modality four-stage features。
- **HPD 目标不兼容：** class-wise prototype、IFV map、ASM 和 feedback 都需要模态级 feature；不能从一个 `modal_x` geometry map 直接推断同构目标。
- **训练状态不兼容：** RobustSeg teacher 低 LR 更新且部分冻结；T-lite 设计只借用最简 output KD，不自动继承 120 epoch、teacher LR 或 HPD 权重。
- **版本混用风险：** 相关预印本中的模块或补充超参数不得与本地 CVPR 2026 正文拼接。

### 5.6 本项目适配

- **[设计—待上级审核冻结] T-lite 最小迁移：** T-lite 仅采用“complete→corrupt 的 output-level teacher guidance”作为最小候选。
- **[设计—待上级审核冻结] Cpair 先行匹配：** 先用 Cpair 匹配 clean/corrupt exposure，再比较 T-lite；不把 HPD、CPD、IFV、ASM、feedback 作为第一版默认结构。
- **[设计—待上级审核冻结] 本地 logits 接口：** 以本地 DFormerv2-S+A2 的 segmentation logits 为接口，教师和学生都使用同一 DFormerv2-S+A2 结构的 full/corrupt views；这只是项目适配假设，不是 PR090 的原始网络实现。

### 5.7 [设计—待上级审核冻结]

- teacher checkpoint 身份、是否冻结、student 初始化、是否 stop-gradient、teacher forward 频率。
- output KD 的温度、KL 方向、权重、valid mask、teacher 置信过滤和 clean/corrupt segmentation loss。
- T-lite 训练长度、学习率、optimizer parameter groups、teacher 计算预算和 checkpoint 选择。
- 是否只评估整幅 missing，还是加入局部 dropout/noise；每种 corruption exposure 必须与 Cpair 完全匹配。

### 5.8 待核项

- PR090 正式版 DOI、PDF 哈希和是否存在独立正式 supplemental，当前允许材料没有提供可核元数据。
- 正文没有统一 teacher/student inference cost 表；不能把“不增加参数”解读为训练或推理零额外成本。
- 正文没有明确写出部署时 teacher、prototype、IFV 和 feedback 辅助模块的保留边界。
- 本文件不读取或合并任何相关预印本；如果上级要求正式版本差异审计，需要另取得正式版 supplemental，不能从预印本补空缺。

## 6. PR070 — DFormerv2: Geometry Self-Attention for RGBD Semantic Segmentation

### 6.1 固定来源与原文锚点

- 正文（本机资料，不纳入 Git）：`../../附件/Yin 等 - DFormerv2 Geometry self-attention for RGBD semantic segmentation/Yin 等 - DFormerv2 Geometry self-attention for RGBD semantic segmentation.md`。
- 正式 supplemental（本机资料，不纳入 Git）：`../../附件/Yin_DFormerv2_Geometry_Self-Attention_CVPR_2025_supplemental/Yin_DFormerv2_Geometry_Self-Attention_CVPR_2025_supplemental.md`。
- 论文锚点：正文 §3.1 `Geometry Prior Generation`、Eq. (1)–(2)、Fig. 3；§3.2 `Geometry Self-Attention`、Eq. (3)–(7)、Fig. 4(c)；§3.3 `DFormerv2 Architecture`、Fig. 4(a–b)；§4.1、Table 1–7、Fig. 5–9。
- supplemental 锚点：§1 `More insights about geometry prior`；§2 `Configuration of our models`；§3 `Details about the decomposition`；§4.1 `Pretraining settings`；§4.2 `Finetuning settings`；Table 1–3。
- 本地代码锚点：[`models/encoders/DFormerv2.py`](../../../../models/encoders/DFormerv2.py) 的 `GeoPriorGen`、`Full_GSA`、`Decomposed_GSA`、`RGBD_Block`、`BasicLayer`、`dformerv2.forward`、`DFormerv2_S`；[`models/builder.py`](../../../../models/builder.py) 的 `EncoderDecoder.encode_decode`、`forward`；[`utils/train.py`](../../../../utils/train.py) 的 optimizer/epoch loop；[`utils/init_func.py`](../../../../utils/init_func.py) 的 `group_weight` 与 `configure_optimizers`。

### 6.2 实施字段

| 字段 | [原论文]与[本地代码]核验 |
| --- | --- |
| 输入与输出 | 论文：RGB 为主视觉输入，Depth 不经独立神经 encoder，而是形成 geometry prior；四级 RGB feature 输出给 decoder。论文正文 §3.3 说 stage 分辨率为 $1/4,1/8,1/16,1/32$。本地代码：`dformerv2.forward` 只取 `x_e[:,0,:,:]`，输出四级 tuple。 |
| geometry prior | 论文 Eq. (1)：对 depth patch average pooling 得 $z_{ij}$，再用绝对深度差；Eq. (2) 用 Manhattan spatial distance。两类 prior 由两个 learnable memories 融合为 $G$。本地代码 `GeoPriorGen` 先 bilinear interpolate 到 stage size，再构造 depth decay；这是强制记录的实现差异。 |
| attention | 论文 Eq. (4)：$\operatorname{Softmax}(QK^T)\odot\beta^G$；Eq. (5)–(7) 是 decomposed horizontal/vertical attention。本地 `Full_GSA`/`Decomposed_GSA` 将 mask 加到 qk logits 后再 softmax，不能把本地代码写成论文 Eq. (4) 的逐字实现。 |
| 模型规模 | supplemental Table 2：DFormerv2-S 的 stage channels $64,128,256,512$，blocks $3,4,18,4$，decoder dimension 512，约 26.7M；本地 `DFormerv2_S` 和 builder 通道一致。 |
| 初始化 | supplemental pretraining table：truncated normal 0.2 的摘要记录；本地 `_init_weights` 对 Linear 使用 `trunc_normal_(std=0.02)`、LayerNorm 单位权重/零偏置；GSA 内部另有 Xavier 初始化。论文配置与本地实现必须分开。 |
| pretraining | 正文 §4.1：RGB-D ImageNet-1K pretraining，CE，300 epochs，AdamW，LR $1\times10^{-3}$，weight decay $5\times10^{-2}$，batch 1024。supplemental Table 1 补充 8×3090、cosine、warmup 5、beta=(0.9,0.999)、RandAugment/mixup/cutmix 等。 |
| finetuning | 正文：CE、AdamW、初始 LR $6\times10^{-5}$、poly decay、NYU/SUN 输入尺寸和 MS-flip scales；supplemental Table 3：LR `6e-5/8e-5`、weight decay 0.01、batch `8/16`、epochs `500/300`、warmup 10、linear decay 字样。本文只把用户已核验的 LR 6e-5、500/300、warmup 10 作为正式事实；scheduler 表述冲突列为待核。 |
| 推理与成本 | 论文 Table 1/6 报 DFormerv2-S 26.7M、NYU 33.9G、SUN 43.7G、NYU latency 43.9 ms（3090、480×640）；这些是论文模型/设置，不是本地 A2 成本。Depth geometry 在推理仍需输入。 |
| 对照 | vanilla attention、only depth prior、only spatial prior、both priors、decomposition；不同 fusion operation、decay rate、RGB/Depth/RGB+Depth 贡献和 latency。 |

### 6.3 原论文做法

1. Depth 作为 geometry prior，而不是对 Depth 建立与 RGB 对称的独立 encoder。
2. Geometry prior 结合深度差与空间距离，并通过 GSA 调整 attention。
3. DFormerv2-S 的正式结构是四级 pyramid，stage block 数为 3/4/18/4，通道 64/128/256/512。
4. 论文平均池化、正文 Eq. (4) 和 supplemental 训练表属于论文原设；本地 checkpoint 代码的 bilinear 和 pre-softmax mask 属于当前实现边界。

### 6.4 可直接借用

- DFormerv2-S 四级输出、Depth 唯一 geometry 消费路径和 decoder 接口，是 R/F 适配的直接本地边界。
- geometry prior 不是普通 Depth feature fusion；新增动作应明确作用于输入 geometry 或 stage feature，而不是默认增加独立 Depth encoder。
- 论文的 prior/attention/decoder ablation 与参数、FLOPs、latency 报告方式可作为成本审计模板。
- 论文的 average pooling 不能直接替换本地 bilinear；二者应作为不同证据层。

### 6.5 与 DFormerv2-S+A2 不兼容

- **论文正式算子与本地实现不完全一致：** average pooling vs bilinear；post-softmax multiplication vs local pre-softmax additive mask。
- **论文训练设置与当前项目脚本不完全一致：** 当前本地 `utils/train.py` 使用 `group_weight` 和 `WarmUpPolyLR`，不能未经配置核验写成 supplemental 的全部配方。
- **A2 不是 PR070 原设：** 本地 `reliability_estimator`、oracle 参数和 telemetry 是项目代码分支；不能归因给 DFormerv2 论文。
- **F-lite 不应复制 Depth encoder：** PR070 明确没有独立 Depth encoder，复制一个完整 modality encoder 会改变骨干结构和成本解释。

### 6.6 本项目适配

- **[设计—待上级审核冻结] R-lite 适配位置：** R-lite 如进入 local geometry input，必须以本地 `GeoPriorGen` 的 bilinear/grid 语义为实施边界，并单独记录与论文 average pooling 的差异。
- **[设计—待上级审核冻结] F-lite 适配位置：** F-lite 如进入 stage output→decoder 之间，必须保留原有 geometry path 不变，先验证轻量残差是否影响四级 tuple 和 decoder。
- **[设计—待上级审核冻结] T-lite 适配位置：** T-lite 只在 segmentation output 接口上借用 complete/corrupt 训练，不把 DFormerv2 geometry equation 改成 teacher/student 机制。

### 6.7 [设计—待上级审核冻结]

- 是否跟随论文 average pooling、跟随本地 bilinear，或把二者作为独立 implementation variants；不能在同一 protocol 中隐式切换。
- R/F/T 各自是否改变输入、stage feature 或 logits；新增参数、初始化、冻结范围和 optimizer groups。
- local DFormerv2-S+A2 的 exact checkpoint、输入归一化、训练 scheduler、epoch、selector、matched control 和成本预算。
- 如果新增模块接入 `EncoderDecoder.encode_decode`，是否对所有 decoder（尤其 `ham` 后三级输入）保持形状兼容。

### 6.8 待核项

- 本地 fork 的 checkpoint 身份、预处理配置与论文公开实现的对应关系。
- supplemental Table 3 的 scheduler 字样与正文 poly decay 的最终解释。
- DFormerv2-S+A2 当前实际参数量、FLOPs、latency；本轮没有运行测量。
- 本地代码中新增 oracle/A2 分支的具体 protocol 不在 PR070 原文内，须在项目文件中单独追踪。

## 7. 条件补充 AI017 — Calibrated RGB-D Salient Object Detection

### 7.1 固定来源与原文锚点

- 正文（本机资料，不纳入 Git）：`../../附件/Ji 等 - Calibrated RGB-D salient object detection/Ji 等 - Calibrated RGB-D salient object detection.md`。
- 方法锚点：§3.1 `Overview`；§3.2 `Depth Calibration`；§3.2.1 `Difficulty-aware Selection Strategy`；§3.2.2 `Depth Calibration Module`；Eq. (1)；Fig. 2–4。
- 融合/损失锚点：§3.3 `Feature Fusion`；Eq. (2)–(9)；Fig. 5；§4.3；§4.5；Table 2–4。

### 7.2 实施字段

| 字段 | [原论文]核验 |
| --- | --- |
| 输入输出 | 两流 RGB-D SOD；先产生 raw Depth reliability，再用 RGB 估计 Depth 与 raw Depth 的加权结果生成 calibrated Depth；后续进入双流 ResNet-50/encoder-decoder 与 CRM，输出 saliency map。 |
| reliability 与 calibration | 先用 RGB-only/depth-only baseline 的 saliency IoU 选 top 20% positive、bottom 20% negative，并在 $IoU_{depth}>IoU_{RGB}$ 时增加 positive；ResNet-18 discriminator 产生 $P_{pos}$。Depth estimator 用 positive set 的 RGB+Depth pairs 训练。Eq. (1)：$Depth_{cal}=Depth_{raw}P_{pos}+Depth_{est}(1-P_{pos})$。 |
| loss、训练、推理 | CRM 使用 channel attention；triplet loss Eq. (8)，margin 1.0；总 loss Eq. (9)，$\alpha=0.2$。ResNet-50 ImageNet init，352×352，Adam，LR $1\times10^{-4}$，batch 16，总 250 epochs（120 selection、60 calibration、70 fusion），推理无 CRF 后处理。 |
| 对照与成本 | RGB stream、raw Depth stream、calibrated Depth stream、direct fusion、CRM without triplet、full CRM；没有本地 DFormerv2 成本。 |

### 7.3 原论文做法

AI017 的核心是“图像级/样本级 Depth quality classifier + RGB 估计 Depth + reliability-conditioned continuous interpolation”，而不是整幅 missing 的 RGB-only prompt。它保留可靠 raw Depth，低质量时才更多使用 RGB 估计 Depth。

### 7.4 可直接借用

- “可靠 raw Depth 应被保护，低质量观测不能无条件覆盖”的评价原则。
- raw/estimated 两种证据的连续组合可作为未来 R 路线的风险对照。
- 质量判断、替代观测和任务评价必须分别报告。

### 7.5 与 DFormerv2-S+A2 不兼容

- AI017 是二值/显著目标任务，使用独立 RGB/depth streams 和 ResNet-18/50，不是 DFormerv2 geometry prior。
- Eq. (1) 的 Depth 估计器输出与本地 Depth 数值契约未对齐；不能把 $P_{pos}$ 当作本地 A2 reliability target。
- AI017 需要 raw Depth 与 RGB 估计 Depth 同时可得；整幅 Depth missing 的 R-lite 没有可直接用于插值的 raw Depth。

### 7.6 本项目适配

**[设计—待上级审核冻结]** AI017 作为条件补充，不升级当前 R-lite 阻塞项，因为当前 R-lite **不采用**其 quality discriminator、RGB estimated Depth 与 raw Depth 的加权校正机制，而只讨论可判定的整幅 Depth missing fallback。若未来 R-lite 改为保留/修正局部 raw Depth，并引入类似 $P_{pos}$ 的质量条件插值，AI017 才需要升级为 R 路线阻塞项和直接对照。

### 7.7 [设计—待上级审核冻结]

任何使用 AI017 的 quality predictor、raw/estimated Depth interpolation、positive/negative selection、triplet loss、ResNet estimator 或其训练轮数的配置，均属于项目设计，当前不冻结、不实现。

### 7.8 待核项

- 若未来采用 AI017 风格校正，需要明确本地 Depth 的物理/归一化量纲、RGB 估计器监督和 natural-invalid 保护规则。
- 当前 R-lite 不依赖 AI017，因此这些待核项不阻塞 R/F/T 全文门禁。

## 8. 条件补充 MoSA — Modality-Aware Spatially-Adaptive Adaptation

### 8.1 固定来源与原文锚点

- 正文（本机资料，不纳入 Git）：`../../附件/Zhang 等 - 2026 - MoSA Modality-aware spatially-adaptive adaptation for RGB-X semantic segmentation/Zhang 等 - 2026 - MoSA Modality-aware spatially-adaptive adaptation for RGB-X semantic segmentation.md`。
- 公式摘录（本机资料，不纳入 Git）：`../../附件/Zhang 等 - 2026 - MoSA Modality-aware spatially-adaptive adaptation for RGB-X semantic segmentation/Formula/Zhang 等 - 2026 - MoSA Modality-aware spatially-adaptive adaptation for RGB-X semantic segmentation_formula.md`。
- 方法锚点：§III `Problem Formulation`；§IV `Design of MoSA`；§IV-A `MS-SMA`；§IV-B `RGCF`；§IV-C `Overall Framework and Training`；Eq. (1)–(20)；Fig. 1–2。
- 实验锚点：§V-A、§V-B、§V-C、§V-D、§V-E；Table 2–7；Fig. 3–8。

### 8.2 实施字段

| 字段 | [原论文]核验 |
| --- | --- |
| 结构输入输出 | RGB+X 两个并行 encoder branch，冻结 SAM ViT-H；MS-SMA 在每个 Transformer block 的 MHSA 和 FFN 后插入 modality-specific low-rank adapter，并由 global context+local content 生成空间 modulation $\alpha$；RGCF 在 encoder output 根据 feature mean/std/max 估计每位置 reliability，再归一化加权融合，送 semantic decoder。 |
| 初始化与参数 | LoRA down projection Gaussian、up projection zero-init；rank $r=16$，modulation MLP hidden $d_h=64$；每模态 adapter 独立，跨 layer 共享 modulation generator。 |
| loss | Eq. (16)：$\mathcal L_{total}=\mathcal L_{seg}+\lambda_{rel}\mathcal L_{rel}$；Eq. (17) segmentation=CE+Dice；Eq. (20) reliability BCE；$\lambda_{rel}=0.1$，前 10 epochs warmup。辅助 per-modality prediction head 只训练使用，不进入推理。 |
| optimizer、长度、硬件 | AdamW，LR $1\times10^{-4}$，weight decay $1\times10^{-2}$，batch 8，50 epochs，cosine；A100；输入 480×640；5 independent seeds。 |
| 对照与成本 | equal/global/channel/self-attention reliability estimator、MS-SMA/RGCF ablation、missing/noise/blur/fog/brightness 条件。NYU Table 2：MoSA 15.3M trainable params、412.04G FLOPs、38.16 ms；包含 dual ViT-H encoder 的推理成本。 |

### 8.3 原论文做法

MoSA 同时做两件事：空间调制 adapter 输出，和基于 feature statistics 的 reliability-guided cross-modal fusion。其 reliability 由模型 feature 与 per-modality prediction correctness 监督，不是本地 A2 的 fixed Depth final-state target，也不是 DFormerv2 geometry-specific action。

### 8.4 可直接借用

- 冻结大 backbone、只训练低秩/残差适配器并报告参数/延迟/FLOPs 的工程纪律。
- zero-init output projection 作为保持初始路径的候选初始化思想。
- 用有/无条件化模块的 parameter-matched 对照，区分新增容量与条件信号收益。

### 8.5 与 DFormerv2-S+A2 不兼容

- **并行 encoder 不兼容：** MoSA 需要 RGB/X 两个 encoder branch；DFormerv2-S 不存在独立 Depth encoder。
- **动作目标不兼容：** MoSA 的核心是 spatial modality modulation 与 reliability-weighted fusion；如果直接用于 F-lite，将把项目候选变成已被 MoSA 直接覆盖的 generic spatial adapter/reliability fusion。
- **backbone/decoder 不兼容：** MoSA 依赖 SAM ViT-H 和 SAM mask decoder 改造，不是 DFormerv2 四级 pyramid + HAM/MLP decoder。
- **监督语义不兼容：** MoSA 的 per-location modality prediction correctness 不能自动等同于 DFormerv2 Depth geometry utility 或 A2 sensor-state reliability。

### 8.6 本项目适配

**[设计—待上级审核冻结]** MoSA 只作为 F-lite 的近邻风险和 parameter-efficient adapter 参考。当前 F-lite 定义为**无条件**的 stage-output residual adapter：不复制整个 encoder、不接 reliability predictor、不做 spatial modulation、不做 reliability-weighted fusion。这样可以先验证“轻量 feature residual adaptation”本身，避免把 MoSA 的核心模块换名后写成本项目新设。

### 8.7 [设计—待上级审核冻结]

若未来决定采用 MoSA 式空间调制、feature-statistics reliability、per-location auxiliary correctness、LoRA rank、$d_h=64$、$\lambda_{rel}=0.1$、50 epochs 或其 dual-branch 结构，均必须另立 F 路线版本并经上级批准；不能悄悄并入当前 F-lite。

### 8.8 待核项

- MoSA 代码入口、commit 和是否有与正文完全对应的公开实现，本轮未核验。
- 若将来作为正式对照，需要确认标题索引与附件正文版本、数据预处理和成本统计是否同一版本。
- 当前 F-lite 不采用 spatial modulation/reliability fusion，因此 MoSA 不升级为 F 路线的当前阻塞项。

## 9. R/F/T 最小适配边界（只记录候选合同，不冻结参数）

本节把五篇核心文献和两个条件补充转换成当前项目需要审核的最小边界。下列所有候选配置都明确标成 `[设计—待上级审核冻结]`；它们不是论文原设，也没有授权实现。

### 9.1 R-lite：整幅 Depth missing 的输入/几何 fallback

#### 原论文做法

- AI023：RGB-only task-driven prompt 或 RGB+degraded-Depth bounded residual recovery。
- AI017：质量判别后在 raw Depth 与 RGB estimated Depth 之间连续校正。
- PR070：Depth 只进入 geometry prior，且论文 average pooling 与本地 bilinear 实现要分开。

#### 可直接借用

- task loss 可以直接评价替代几何是否帮助分割；不把 prompt 叫 metric Depth。
- 输入/替代路径要保护可靠 Depth，并单独报告 missing/degraded。
- 对照必须包含 no-action、原始输入、替代输入和成本。

#### 与 DFormerv2-S+A2 不兼容

- AI023 的三通道 0–255 prompt、Recovery depth-condition CNN 和 DFormerv2-S 的单通道 geometry input 不同。
- AI017 的 raw/estimated interpolation 需要 raw Depth 和质量 predictor，不适用于当前只判定整幅 missing 的边界。
- PR070 论文算子与本地 bilinear/pre-softmax mask 不能混写。

#### 本项目适配

**[设计—待上级审核冻结]** R-lite 只处理可由当前输入判定的整幅 Depth missing；非整幅 missing 精确 bypass；natural-invalid 不默认覆盖；不接 A2 reliability predictor；不把局部 missing、natural-invalid、noise、misalignment 统一成同一路径。

#### [设计—待上级审核冻结]

输出是输入 Depth-like map、geometry grid 还是直接 geometry prior；残差幅度、归一化、valid 区保护、训练 corruption、loss、optimizer、训练长度、checkpoint 与 selector 均未冻结。不得照搬 AI023 的 0–255、127.5、$s$ 或 TV/L1 权重。

#### 待核项

本地 `modal_x` 真实范围、整幅 missing 判定、bilinear 后 sentinel 语义、geometry prior 中替代值的可解释性和推理成本。

### 9.2 F-lite：stage output→decoder 的无条件 residual adapter

#### 原论文做法

- AI024：复制整个 encoder、冻结原 encoder/decoder、zero-init $1\times1$ 注入。
- MoSA：冻结 foundation encoder、使用低秩 adapter 和 spatial modulation/reliability fusion。
- PR070：本地 DFormerv2-S 输出四级 feature tuple，Depth 不经独立 encoder。

#### 可直接借用

- residual path 从 identity/no-op 起步；新增路径的输出投影可 zero-init。
- adapter 参数、延迟和训练成本必须单列。
- 首先做无条件 adapter，再决定是否需要质量条件或空间调制。

#### 与 DFormerv2-S+A2 不兼容

- 不复制完整 DFormerv2 encoder；否则变成 ConD 风格且成本、归因和部署都改变。
- 不直接接 MoSA 的 global/local reliability modulation 或 RGCF。
- 不假设 A2 reliability head 可以直接给 adapter 作为 gate；本地代码显示它是辅助 loss 分支。

#### 本项目适配

**[设计—待上级审核冻结]** F-lite 位于 DFormerv2 stage 输出和 decoder 之间，做无条件 residual correction；第一版不接 reliability predictor、不复制整个 encoder、不改变 geometry prior；第一版优先保持 decoder 输入 tuple 形状不变。

#### [设计—待上级审核冻结]

插入哪些 stage、每个 stage 的 bottleneck 通道、共享/独立、激活、最后 `1×1` zero-init、冻结范围、optimizer 参数组、学习率、训练长度和参数/延迟预算均未冻结。`1×1 bottleneck→activation→zero-init 1×1` 只是候选形态，不是已经批准的网络。

#### 待核项

各 stage 与 HAM decoder 的实际形状、zero-init 后梯度能否启动、主干冻结时 optimizer 是否遗漏新增参数、clean 输入是否保持 no-op，以及参数匹配 generic adapter 对照。

### 9.3 T-lite：complete→corrupt 的最简 output-level KD

#### 原论文做法

- AI019：complete teacher→missing student，但目标是 token/pixel reconstruction，共享参数。
- PR090：teacher full modality、student Anymodal Dropout，完整方法为 CE/KL+prototype+IFV+feedback。
- 两篇都说明完整输入可以作为受损输入训练的知识来源，但具体机制不同。

#### 可直接借用

- complete/corrupt paired views 和 teacher guidance 的动机。
- Cpair 必须匹配 clean/corrupt exposure、student update 数和 segmentation supervision。
- teacher 前向成本与 student 训练成本分别报告。

#### 与 DFormerv2-S+A2 不兼容

- 不引入 AI019 的 token/pixel MIM、semantic-map modality 或共享 MIM heads。
- 不引入 PR090 的 per-modality encoder、prototype、IFV、ASM、feedback 或 dominant-modality teacher update。
- 本地 DFormerv2 logits 蒸馏不能写成复现任一完整方法。

#### 本项目适配

**[设计—待上级审核冻结]** T-lite 只在 Cpair 上增加最简 output-level KD：teacher 使用完整输入，student 使用同一场景 corrupt/missing 输入；训练和推理只保留项目批准的实际 student path，第一版不默认保留 teacher。

#### [设计—待上级审核冻结]

teacher checkpoint、冻结/更新、KD temperature、KL 方向、权重、valid mask、teacher error filter、clean/corrupt CE 比例、teacher forward 频率、训练长度、optimizer、selector 和成本上限均未冻结。

#### 待核项

A2 checkpoint 是否作为 teacher、完整输入是否能在训练中合法可用、teacher/student 是否共享权重、missing condition 是否只限整幅 Depth missing，以及如何防止 teacher 错误被学生复制。

## 10. 当前不应混入 E1 合同的内容

1. **AI023 的 0–255/127.5 和 Recovery depth encoder**：它们是论文量纲和结构，不是本地 DFormerv2 默认值。
2. **AI024 的完整 encoder copy**：它是 ConD 原文，不等于 F-lite；若采用必须单列为更重的 ConD-inspired 分支。
3. **AI019 的 MIM/STTP**：它是 token/pixel reconstruction 预训练，不能叫 output KD。
4. **PR090 的 HPD/feedback**：正式版完整方法不等于 T-lite；不拼接作者预印本。
5. **MoSA 的 spatial modulation/RGCF**：若加入则触发 F 路线条件阻塞，不是当前无条件 adapter。
6. **AI017 的 quality-conditioned raw/estimated Depth interpolation**：若加入则触发 R 路线条件阻塞，不是当前整幅 missing fallback。
7. **PR070 的论文 average pooling 与本地 bilinear**：不得在同一 checkpoint 说明中隐去实现差异。
8. **任何未标 `[设计—待上级审核冻结]` 的 E1 epoch、LR、loss weight、adapter width、teacher temperature、checkpoint 规则**：本文件不替上级冻结这些值。

## 11. 待核项清单与恢复点

### 11.1 来源与版本

- PR090 正式版 DOI、PDF 哈希、独立 supplemental 是否存在；当前只按本地 CVPR 2026 正文，未并入预印本。
- 五篇论文的官方 code/commit 是否与正文固定版本对应；当前只有论文中给出的若干入口，未把公开状态写成事实。
- PR070 本地 fork checkpoint 与论文公开实现的具体对应关系。

### 11.2 本地接口

- `modal_x` 的真实量纲、预处理、missing sentinel、validity 和 bilinear 后语义。
- DFormerv2-S+A2 当前 decoder、A2 head、optimizer param groups 和 checkpoint 的完整 protocol identity。
- R/F/T 新增参数是否进入当前 `group_weight` 参数分组；主干冻结时是否会产生 unused parameter 或错误的 no-op。

### 11.3 实验合同

- R-lite 的 exact input/output、valid 区保护、loss、optimizer、长度、成本和 stop 条件。
- F-lite 的 stage、bottleneck、activation、zero-init、冻结和 parameter-matched generic control。
- T-lite 的 teacher 身份、temperature、KD 权重、Cpair exposure、student update 数和推理保留信息。
- 上述实验合同字段已写入同期 `../01_research/e1_screening_plan.md`，但仍需由上级审核冻结；本文件不代替该子计划，也不授权实现。

## 12. 门禁裁决与未运行检查

### 12.1 门禁裁决

**文献全文门禁完成。** 这里的“完成”仅表示：AI023、AI024、AI019、PR090（只按 CVPR 2026 正式版范围）和 PR070 已形成实施级结构化审计，并且 AI017、MoSA 两个条件补充已覆盖；PR070 正式 supplemental 已审计；PR090 未取得的 DOI/supplemental-only 元数据仍明确列为待核，未被预印本补齐。

这项裁决**不授权**：

- 编写或修改 R/F/T 实现代码；
- 启动 E1 Batch 1 或 Batch 2；
- 训练、GPU/云任务、完整评估、checkpoint 选择或 official test；
- 把任何 `[设计—待上级审核冻结]` 参数当成已批准配置；
- 把本审计写成五篇论文的忠实复现承诺。

下一门禁仍是：由上级模型共同复核本文件与已形成的 `../01_research/e1_screening_plan.md`，并冻结 R-lite、F-lite、T-lite 的接口、loss、初始化、冻结/训练参数、optimizer groups、matched controls、预算和 stop/promotion 条件；之后还需用户单独授权代码和运行。

### 12.2 本轮未运行检查

- 未运行项目测试、静态全仓扫描、GPU、训练、评估、云任务或 official test。
- 未运行模型 forward、参数量/FLOPs/latency 测量或 checkpoint 加载验证。
- 只做了允许范围内的文本、公式、图表、代码锚点阅读和差异边界整理。
