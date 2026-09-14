# MMFR-多形式模态失效可靠性学习：独立审计总规划

> **文档状态：** 2026-09-14 独立审计快照；A1 已完成 CPU qualification，A2 已完成 code qualification 与本地 GPU 单步 qualification；正式训练、云端 probe、MMFR checkpoint 评价和 official test 均未授权、未完成。
> **事实入口：** `doc/main/MUSeg-current-status.md`；本文只在该入口基础上重组审计叙事，不替代实时状态。

## 0. 文档用途、形成时点与阅读规则

- **文档用途：** 本文件是一份可以直接交给“完全看不到 DFormer 仓库内容”的外部 agent 审计的自包含总规划。它把项目背景、历史依据、当前实现、冻结设计、已完成资格证据、正式实验计划、授权边界、风险和审计问题放在同一份叙事中。
- **形成时点：** 2026-09-14。本文按该时点实际工作区内容和 `MUSeg-current-status.md` 的事实入口重组，不以 Git HEAD 替代工作区事实。
- **外部 agent 的使用方式：** 外部 agent 不应假定自己能够读取仓库、云端目录、checkpoint、日志或本文引用的文件。审计应先把本文中的“事实”“冻结设计”“建议”“未授权”四种状态分开，再逐项要求补交证据。本文引用的路径是证据索引，不是外部 agent 已经看见这些文件的声明。
- **事实优先级：**
  1. 直接核验的当前状态、工作区代码、协议模板、已有运行产物及其哈希；
  2. `doc/main/MUSeg-current-status.md` 的实时状态和恢复点；
  3. `doc/main/MUSeg-open-decisions.md` 的研究选择与边界，重点是第 8、12、13 节；
  4. 形成时点的计划和历史报告；
  5. 本文对未来实验的重组说明。
- **语言规则：** “已完成”只表示实现或资格检查已经发生；不表示模型效果已证明。“已通过但不构成效果”表示链路或接口通过了定点检查；“冻结”表示实验开始前不得随意改变；“建议”表示尚未成为当前 v1 主门槛；“未授权”表示即使计划已经写好也不能执行。
- **硬边界：** 截至本文形成时，没有正式 500 epoch MMFR 训练、没有新训练 checkpoint、没有云端 probe、没有 MMFR checkpoint 评价，也没有读取 official test。本文不能被解释为这些工作已经完成。

## 1. 一页式执行摘要

### 1.1 现在已经知道什么

- DFormer 是面向 RGB-D 语义分割的模型族；本项目使用其 DFormerv2-S 变体，在 MUSeg 矿井场景数据上处理对齐的 RGB 图像、Depth 图像和语义标签。
- MUSeg 当前开发职责为 `train-dev=1277`、`val-dev=318`，另有封存的 official test。`val-dev` 参与 checkpoint 选择，因此它是开发证据，不是独立测试集。official test 身份为 `sealed_unread`，即封存且未读取。
- Quick-B0 是已冻结的 RGB single-seed development 基线：官方预训练权重、epoch 420 checkpoint、五尺度加水平翻转的 10-view evaluator，主指标为 mIoU `58.79`、mAcc `69.91`、mF1 `72.73`。它用于内部模块对照，不是三 seed 论文级复现，也不是 official test 结果。
- 过去把 Depth 置零作为失效动作的 DVC 方向没有得到支持性结论；随后“知道真实坏区后，直接关闭 GSA 的 Depth contribution”的 DVG-B1 Oracle 方案在完整配对评价中使主指标下降，最终以 `oracle-not-supported` 关闭。这些是历史结论，不能改写成成功，也不能从中推出所有可靠性学习都无效。
- 当前转入 MMFR（Multi-Modal Failure Robustness，多模态失效鲁棒性）方向：用受控合成的多种 RGB/Depth 失效和连续可靠性目标训练一个小型可靠性估计器。A1 只提供失效基函数、target 和未来 adapter 的 standalone 脚手架；A2 已把 Depth-only corruption 和 reliability auxiliary loss 接入训练链，但当前可靠性预测不进入 backbone、decoder 或 geometry prior。
- A2 的正式研究问题不是“最终可靠性融合是否有效”，而是先验证：在不改 DFormerv2 geometry prior、不做 clean/corrupt 双前向的条件下，Depth 失效输入、连续 target、辅助损失和公平训练身份是否能被正确接入。
- A2 代码已经通过 CPU qualification；本地 NVIDIA GeForce RTX 5060 Laptop GPU 已完成一次真实单步 preflight：尝试 6 个 batch，前 5 次因 GradScaler 跳过，完成 1 次 optimizer update，6 个 loss 均有限。该证据只证明训练链能够运行一次，不能证明模型鲁棒性或收益。
- 正式训练职责已经冻结为云端单 GPU，默认 RTX 4090；RTX 5090 只有在 4090 缺货、冻结 batch size 10 无法装入 24 GiB，或配对 probe 证明单位样本成本更低时才使用。当前没有云实例，没有训练授权，也没有容量/吞吐 probe 授权。

### 1.2 大白话版方案

先拿两个完全公平的模型从同一份官方预训练权重重新训练：一个始终看干净的 RGB-D 输入，另一个在每个训练样本上以固定随机规则损坏 Depth，并额外学习“这块 RGB/Depth 现在有多可信”。目前这个可靠性预测只接受辅助监督，不会直接替 DFormer 关掉某条融合路径。等 A2 训练和开发评价有结果后，才考虑 B1：让一个小型 adapter 学会如何把可靠性转成 geometry prior 的贡献缩放；如果研究目标还包括 RGB 完全失效，则需要 B2，增加真正的 Depth 语义路径或双语义路径。当前还没有进入长程训练，所以不能说方案有效或失败。

## 2. 项目背景与数据职责

### 2.1 DFormer、DFormerv2-S 与 RGB-D 语义分割

- **DFormer：** 本项目中的 DFormer 是一个 RGB-D 语义分割模型族，同时使用彩色图像和深度图像，为每个像素预测语义类别。
- **DFormerv2-S：** 当前模型为 DFormerv2-S，即 DFormerv2 的 Small 规模配置。模型有 RGB 分支和 Depth 分支；现有实现中 Depth 主要通过 geometry prior（几何先验）进入 GSA（Geometry-aware Self-Attention，几何感知自注意力）相关计算，而 Q/K/V 的主要语义信息仍然来自 RGB。因此它是 **RGB-centric（以 RGB 为中心）** 的 RGB-D 模型，不是已经具备独立 Depth 语义推理能力的双语义模型。
- **RGB-D 语义分割：** 输入是空间对齐的 RGB 与 Depth，标签是语义类别图；输出是每个像素的类别 logits 或类别预测。Depth 不是标签，也不是天然的传感器健康真值。
- **MUSeg：** 当前材料将 MUSeg 定位为矿井域 RGB-D 语义分割数据集，包含 3,171 对精确对齐的 RGB/Depth、六个矿区和 15 个语义类别。本文不把这些数据自动解释成带有故障类型、故障位置或故障严重度的传感器数据集。

### 2.2 数据划分和职责

- **`train-dev=1277`：** 用于模型训练和训练过程中的开发职责。MMFR A2 的两个公平训练身份都应使用这一划分。
- **`val-dev=318`：** 用于 clean checkpoint selector 和冻结后的开发评价。该划分包含 196 个 location groups（位置组），后续配对 bootstrap 以 location group 为重采样单位。
- **official test=`sealed_unread`：** official test 是封存、未读取的最终测试职责。它不能参与训练、checkpoint 选择、阈值选择、方向筛选或当前开发评价。
- **输入职责：** RGB 是彩色观测，Depth 是深度观测；A2 的 backbone 继续消费现有 normalized tensor。可靠性 head 另消费从最终 crop 恢复的 raw `[0,1]` 信号。
- **标签职责：** Label 是语义分割的 ground-truth 标签，用于 segmentation loss 和评价；A2 的 reliability target 来自合成器已知的失效 burden，不是 MUSeg 提供的自然传感器可靠性标签。
- **评价职责：** checkpoint selector 和主 evaluator 都固定在开发划分上；最终评价必须报告输入 geometry、metric geometry、evaluator、checkpoint、split、seed、condition 以及产物哈希。

## 3. 稳定 Quick-B0 基线

### 3.1 身份

- 模型：`DFormerv2-S RGB Quick-B0`。
- RGB 输入契约：`rgb-imagenet-rgb-order-v1`。当前 quick B0 明确把原始 OpenCV BGR 转成 RGB，再使用 RGB 顺序 ImageNet mean/std。
- 官方预训练权重：`DFormerv2_Small_pretrained.pth`，SHA-256 为 `19116988fc86dc9f3e879282237941e11b9b1b5c480edb51e92807311dbc11a6`。
- 最终 checkpoint：epoch 420 的 `selector-epoch-420.pth`，SHA-256 为 `f246a3afc50334c81302b7bfebdadf7cf37d00326bf1c3aa54f6a151754e3a1c`。
- `val-dev` split SHA-256：`1d0719d8f64f016d48995c25ab66d4004d76b7155d9efeef7cbb7454c0dd0e83`。
- official test：`official_test_included=false`，保持 `sealed_unread`。

### 3.2 主 evaluator 与指标

- 主 evaluator 身份：`msflip-whole-original-grid-v1`。
- 每张图采用五个尺度 `0.5、0.75、1.0、1.25、1.5`，每个尺度使用原图和水平翻转，共 10 个 view。
- 每个 view 的输出恢复到 MUSeg 原始 Label 网格；在 FP32 中平均 pre-softmax logits 后计分。
- epoch 420 主指标：mIoU `58.79`、mAcc `69.91`、mF1 `72.73`，单位为百分比。
- 选择过程说明：训练期 selector 的单尺度排序与主 evaluator 的排序不同；epoch 480 的 selector 分数更高，但按预登记的主 evaluator，epoch 420 最终胜出。这说明不能用单尺度 selector 分数代替最终 evaluator。
- B0 的性质：single-seed、RGB、development Quick-B0。它足以作为同口径模块探索的内部共同起点，但不能估计随机方差，不能声称三 seed 完整复现，不能声称 official-test 性能。
- 后续模块的公平要求：必须从相同官方 pretrained 独立训练，保持相同 split、seed、数据顺序、训练预算、优化器、增强、selector 和主 evaluator。不能从 epoch 420 checkpoint 续训后再称为公平消融。

## 4. 关键术语表与判断边界

- **MMFR（Multi-Modal Failure Robustness，多模态失效鲁棒性）：** 本研究方向，研究 RGB/Depth 完全缺失、局部缺失、噪声、模糊、量化、几何错位和混合失效下的模型行为。大白话是：先用可复现的方式把传感器输入损坏，再看模型能否学习到更稳健的行为。
- **corruption（合成失效/输入损坏）：** 在原始对齐输入上按冻结规则制造的受控退化。它不是自然故障标签，也不是对真实故障发生率的估计。
- **reliability target（可靠性目标）：** 合成器根据每个像素的失效 burden 计算出的 `[0,1]` 连续目标。它描述“按当前合成规则，这个像素被认为还剩多少可信度”，不是自然世界传感器的真值。
- **reliability head（可靠性头）：** 从固定信号特征预测 RGB/Depth 可靠性 logits 的小型卷积网络。A2 中它只承担辅助 supervision（监督），预测不会反馈给 DFormerv2 的 backbone、decoder 或 geometry prior。
- **clean control（干净对照）：** 使用与 corruption 模型完全相同的官方 pretrained、seed、split、优化器、schedule 和 selector，但训练输入始终不做 MMFR corruption，并且不增加 reliability head。
- **geometry prior（几何先验）：** DFormerv2 根据 RGB/Depth 的空间和深度关系形成的几何贡献信息。它不是最终语义 logits。
- **GSA（Geometry-aware Self-Attention，几何感知自注意力）：** DFormerv2 中消费合成 geometry prior 的注意力机制。DVG-B1 曾尝试只关闭其中的 Depth contribution；该动作失败不能等同于所有 reliability 方法失败。
- **adapter（适配器）：** 位于可靠性估计和几何贡献之间的小型可学习模块，未来 B1 计划用它学习如何缩放 spatial/depth contribution。A2 当前不启用它。
- **preflight（预检）：** 在正式长程训练或完整评价前，对真实模型链做小规模、受限的运行检查。preflight 通过只说明接口、数值和资源路径在该小规模条件下可运行，不说明模型效果。
- **checkpoint selector（checkpoint 选择器）：** 训练期用低成本、冻结规则选择候选 checkpoint 的过程。当前 A2 selector 只看 clean `val-dev`、`original-full`、scale `1.0`、无 flip 的 mIoU；失效条件不参与 selector。
- **paired bootstrap（配对 bootstrap）：** 保持同一图像各条件的配对关系，以冻结的 location group 为相关性单位进行有放回重采样，形成效应量区间。它是本项目预注册统计流程，不归因于某篇不适用的历史文献。
- **mIoU（mean Intersection over Union，平均交并比）：** 对各语义类别的 IoU 做宏平均，主 robustness score 使用六个单失效条件的 mIoU 宏平均。
- **mAcc（mean accuracy，平均类别准确率）：** 对各类别准确率做宏平均，作为辅助语义指标。
- **mF1（mean F1，平均 F1）：** 对各类别 F1 做宏平均，作为辅助语义指标。
- **Boundary IoU（边界交并比）：** 在定义的边界带几何范围内，按 one-vs-rest、ignore、空类和宏平均规则计算边界评价。历史 `boundary_band_mIoU` 不能直接冒充标准 Boundary IoU。
- **AMP（Automatic Mixed Precision，自动混合精度）：** 让部分计算使用低精度以降低显存和提高速度，同时保留必要的高精度计算。
- **FP16：** 16 位浮点格式。A2 首次真实 AMP forward 发现固定信号特征中梯度平方乘积会在 FP16 下溢；因此 reliability auxiliary 分支固定用 FP32，而 segmentation backbone/decoder 仍可用 AMP。
- **GradScaler：** AMP 训练中动态缩放 loss 以避免梯度下溢，并在发现非有限梯度时跳过 optimizer update。A2 preflight 的前 5 次跳过正是这种计数语义，不能把“尝试次数”写成“成功更新次数”。
- **SyncBN（同步 Batch Normalization）：** 在分布式训练中跨设备同步 batch 统计量的 BatchNorm 形式。当前正式配置保留 SyncBN，但当前正式环境冻结为单 GPU；SyncBN/DDP 的多卡行为未由 A2 preflight 覆盖。
- **DDP（DistributedDataParallel，分布式数据并行）：** 多 GPU 分布式训练方式。当前没有运行 DDP，也没有把单 GPU preflight 当作 DDP 通过证据。
- **`torch.compile`：** PyTorch 的编译优化路径。正式首轮明确关闭，以减少编译路径引入的变量；本地 preflight 也不构成 `torch.compile` 通过证据。
- **single-seed development-supported：** 即便主门槛未来达到，也只能表示单个随机种子、开发划分上的支持性证据，不表示多 seed 稳定性、独立测试泛化或真实部署收益。

## 5. 历史依据：从 DVC/DVG 转到 MMFR

### 5.1 Depth 置零的历史问题

- 早期 DVC-A1 方向把 Depth 的局部边界失效编码为置零，试图比较模型在 boundary/non-boundary 位置的变化。
- v1 的定义门禁扫描了全部 318 个 `val-dev` 样本和 196 个 location group；有 58/196 个组无法形成非空 `boundary-q75`，比例为 `29.5918%`，高于预注册的 `5%` 上限。因此 v1 以 `protocol-blocked` 结束，没有生成可用的五条件科学裁决。
- v2 固定保留 0.05 相对深度跳变阈值、218 张图和 138 个组，但完整统计时一个组只有 137 个有效配对组，仍按预注册门禁阻塞。后续诊断发现 Boundary IoU 的标签域把 raw background 与 evaluator ignore 混用了。
- v3 只修正了 Boundary IoU 的标签域契约：raw background 保留为有效边界几何上下文，true ignore 仍单独处理；数据范围、Depth 置零动作、checkpoint、evaluator 和统计单位没有改变。v3 完整运行主 `dose_effect` 为 `+0.0731348717` 个百分点，95% 区间为 `[-0.0620441424,+0.2226503089]`，区间跨过 0，最终裁决为 `not-supported`。
- 历史结论的准确含义是：在该冻结数据、失效定义、模型和统计协议下，没有支持 DVC 假设的充分证据。它不表示 Depth 可靠性研究从原则上不可能，也不表示 v1/v2 的 protocol 阻塞可以被改写成模型效果失败。

### 5.2 DVG-B1 Oracle 负结果

- DVG-B1 的问题更窄：假设已经知道真实 corruption mask，只在 GSA 的 Depth contribution 上施加 Oracle gate；不改 Depth 输入、不改 Q/K/V、不改 decoder、不改 logits 后处理。
- A/B/C 规则在查看结果前冻结：A 是像素 mask 经 `INTER_LINEAR`、flip、padding 和 OpenCV `INTER_AREA` 聚合到四级 token；B 是 Full/H/W 拓扑上的 query/key 对称乘积；C 要求 `boundary-q75` 下 Boundary IoU 至少增加 `+0.10` 个百分点、95% percentile interval 下界严格大于 0，且 mIoU 点估计不得为负。
- P1 CPU qualification、P2 no-op、P3 受损 gate preflight 和 P4 218 张图完整本地 GPU 配对评价均完成。P4 使用 218 张图、138 个 location group、五个 condition、每图 10 view，共 `10,900` 个 baseline/gated 配对 forward pair。
- clean 条件的全部 `2,180/2,180` view 和 `218/218` 融合结果严格相等；这证明 no-op 和变量隔离，不证明受损条件收益。
- 主 `boundary-q75` 条件的 Oracle-minus-baseline Boundary IoU 为 `-0.1508352015` 个百分点，95% 区间为 `[-0.2676580460,-0.0462325573]`；mIoU 为 `-0.2316731726` 个百分点，95% 区间为 `[-0.3914881430,-0.0919002976]`。因此按冻结 C 裁决为 `oracle-not-supported`。
- 该负结果不能被改写为“可靠性学习失败”。它说明：corruption mask 不等于模型应使用的贡献权重；未训练的固定回退动作可能破坏已学习的 geometry prior；同一可靠性值可能需要随 stage、失效类型和上下文改变动作；DFormerv2 的 RGB-centric 结构不能只靠 GSA Depth gate 覆盖 RGB 完全失效。

### 5.3 对 MMFR 架构的含义

1. 先学习失效分布和可靠性信号，再研究可靠性如何影响融合动作；不能把人工 mask 直接当作最终 gate。
2. A2 先隔离“训练分布、target 和辅助 loss 是否接对”；B1 才研究 learned geometry adapter，减少把多个新变量同时引入的风险。
3. A2 当前不声称最终可靠性融合模型；它没有把 predicted reliability 送入 backbone、decoder 或 geometry prior。
4. 如果研究目标包括 RGB complete-missing，必须另建 Depth 语义路径或双语义路径，不能把当前 RGB-centric DFormerv2 的 A2 结果扩写成 RGB 缺失解决方案。

## 6. MUSeg 自然故障证据边界

- MUSeg 图像中可能出现弱光、粉尘、反光、噪声或深度无效等现象，但当前数据和被核对的材料没有为这些现象提供正式的自然故障类型标签、像素位置标签、严重度标签、故障率、重复采集、跨时刻配对或标定漂移真值。
- 因此本文不声称“数据中只有 Depth=0”，也不把未审计的视觉现象写成已确认的自然故障类别。
- A1/A2 的 `entire_missing`、局部 dropout、noise、blur、quantization 和 misalignment 都是受控合成条件。它们可以支持“在 MUSeg 语义域和冻结合成协议下的开发研究”，不能单独支持真实矿井传感器故障率、自然故障分布、现实部署可靠性或跨设备校准稳定性。
- 未来如果要提出自然故障结论，必须另行获得自然故障标签、采集和标定证据，或明确把结论限制为“合成失效外推假设”，并报告 synthetic-to-real gap（合成到真实的差距）作为未解决风险。

## 7. 当前 MMFR 总体架构与阶段图

### 7.1 阶段关系

```text
A1：通用 RGB/Depth 失效基函数
    ├─ 受控 corruption
    ├─ 连续 reliability target
    ├─ audit metadata / deterministic semantics
    └─ fixed signal features + future adapter scaffold
             │
             ▼
A2：当前正式训练接入（Depth-only）
    ├─ post-mirror/scale/crop/pad、pre-GPU corruption
    ├─ segmentation loss + 0.1 × reliability auxiliary loss
    ├─ predicted reliability 不进入 backbone/decoder/geometry prior
    └─ clean control 与 Depth-corruption 公平身份尚未正式训练
             │
             ▼
B1：未来 learned geometry adapter（待细化、待授权）
    ├─ reliability pyramid
    ├─ 学习 spatial/depth geometry contribution 缩放
    └─ 必须从官方 pretrained 独立训练并建立新 protocol
             │
             ▼
B2：若要覆盖 RGB 完全失效（待设计）
    ├─ Depth semantic path 或双语义 path
    └─ 不能由当前 RGB-centric A2/B1 静默替代
```

### 7.2 当前 A2 的准确定位

- 当前正式 config 是 **Depth-only corruption + segmentation loss + reliability auxiliary loss**。
- A2 的 reliability head 预测 RGB/Depth 两通道 reliability，但当前训练只采样 Depth；因此 RGB target 在 A2 中始终为 1，RGB complete-missing 不进入该训练身份。
- predicted reliability 不进入 DFormerv2 backbone、decoder 或 geometry prior；四级 area pyramid 和 geometry adapter 虽然已在 A1 模块中定义，但 A2 的辅助 loss 只使用固定特征、head 和连续 BCE，未计算未使用的 pyramid 参与模型动作。
- 当前 A2 不是最终可靠性融合模型。即使未来两个训练身份产生结果，也只能先回答训练分布和 auxiliary supervision 的开发问题；B1 需要独立证明 learned adapter 是否带来下游收益。

## 8. A1 失效基函数、severity、组合与 metadata

### 8.1 当前 A1 支持的类型

- `entire_missing`：整模态完全缺失，输出全零；当前 A2 只允许它作用于 Depth。
- `spatial_dropout`：块状局部缺失。采样网格中的块被置零，severity 控制被选中的块数。
- `gaussian_noise`：连续加性高斯噪声，最大标准差参数为 `48.0` 个 uint8 强度单位。
- `blur`：Gaussian blur，最大 sigma 为 `6.0`。
- `quantization`：Depth 的位深/台阶损失，Depth-only。
- `misalignment`：Depth 相对 RGB 的整数平移，Depth-only；越界区域显式置零并记录为 invalid。
- A1 第一版明确不实现 haze、dust 合成、非刚性 warp、Poisson shot noise 或真实相机响应模型；这些名称不能在审计中被当作已实现功能。

### 8.2 severity 和 curriculum

- 每个 `FailureSpec` 显式记录 `modality`、`kind` 和 `severity`；severity 合法范围为 $(0,1]$。
- curriculum progress 固定在 `[0,1]`：
  - 前 1/3 为轻度阶段，severity 上限 `0.30`，每个样本只抽 1 个 spec，不能抽 `entire_missing`；
  - 中间 1/3 为中度阶段，severity 上限 `0.60`，可抽 1–2 个 spec，不能抽 `entire_missing`；
  - 后 1/3 为重度阶段，severity 上限 `1.0`，最多 `max_specs` 个 spec，允许 `entire_missing`；
  - 当前 A2 `max_specs=2`；`entire_missing` 只在重度阶段出现，并固定为 severity `1.0`；severity 抽样下限为 `0.05`。
- 采样器只返回 spec，不偷偷修改数据；所有随机性来自调用方提供的 NumPy Generator，不读写 Python、NumPy 或 PyTorch 的进程级全局随机状态。

### 8.3 组合可靠性公式

每个失效基函数在像素 $p$ 对模态 $m$ 产生非负局部 burden $b_{m,k}(p)$。多种失效按调用顺序作用，后一个 spec 看到前一个 spec 的输出；可靠性按乘法组合：

$$
R_m^*(p)=\prod_k \exp\left(-b_{m,k}(p)\right)=\exp\left(-\sum_k b_{m,k}(p)\right).
$$

- 结构性缺失的 burden 使用 `MISSING_BURDEN=1.0e4`；在合法计算范围内 `exp(-MISSING_BURDEN)` 下溢为 0，表示完全不可信。
- `gaussian_noise`、`blur`、`quantization` 使用 realized local damage：逐像素最大通道变化按 `255 × DAMAGE_REFERENCE` 归一化，其中 `DAMAGE_REFERENCE=0.25`，再乘 severity 并截断到 `[0,1]`。
- `misalignment` 的有效区域 burden 由已知位移的几何幅度决定，而不是由平坦图像的像素差决定：

$$
 b_{\mathrm{in\text{-}bounds}}=\min\left(1,\frac{\sqrt{dx^2+dy^2}}{\sqrt{2}\,\mathrm{MISALIGN\_MAX\_PX}}\right),\qquad \mathrm{MISALIGN\_MAX\_PX}=16.
$$

越界像素使用 missing burden。这样同样的平移不会因为图像恰好平坦而被错误判为可靠。

### 8.4 输出和可审计 metadata

- 输出保持 RGB 的 `uint8 H×W×3` 和 Depth 的 `uint8 H×W` 或三通道同值 `H×W×3` shape/dtype。
- reliability target 为 `float32 [2,H,W]`，通道顺序固定为 RGB、Depth，范围为 `[0,1]`。
- 空 spec 是严格 no-op：数组逐元素相同，target 全 1，metadata 为空。
- 非空 metadata 记录 spec 数量、顺序、模态、kind、severity、每个 spec 的 burden min/max/mean、受影响模态的 reliability min/mean，以及 misalignment 的位移、越界像素数等信息。
- 任何不支持的 modality/kind、非有限 severity、越界 severity、shape/dtype 不一致或非有限 target 都 fail-closed，而不是静默修正。

## 9. A2 数据流和确定性语义

### 9.1 具体接入位置

A2 的 corruption 在 `DataLoader` 已完成 mirror、scale、crop、pad 之后、batch 搬到 GPU 之前，由主训练进程逐样本执行。于是 reliability target 天然与最终 `480×640` crop 几何一致，不需要在 target 上重新播放 mirror/scale/crop，也不改变 worker 内 `TrainPre`、`RGBXDataset` 的原有几何增强语义。

大白话是：先照旧得到训练批次，再在主进程中损坏 Depth；损坏动作不被放进 worker，所以不依赖 worker 调度和 worker 数量。

### 9.2 raw、normalized、pad 和 valid mask

- segmentation backbone 继续消费 normalized RGB/Depth tensor。
- reliability estimator 消费从最终 normalized crop 严格恢复的 raw `[0,1]` tensor。
- RGB inverse normalization 使用当前配置的 mean/std：mean `[0.485,0.456,0.406]`，std `[0.229,0.224,0.225]`。
- Depth inverse normalization 固定使用 `TrainPre(sign=True)` 的三个相同通道 mean `[0.48,0.48,0.48]` 和 std `[0.28,0.28,0.28]`。
- qualification 对全部 uint8 值 `0..255` 做 exhaustive round-trip：`uint8 -> normalize -> float32 -> inverse+round` 必须逐值恢复原 byte；失败即 `qualification-blocked`。
- 训练 crop/pad 的 normalized pad 值是精确 `0`。若 RGB 与 Depth 在某像素都逐通道精确为 0，则该像素是几何无效区；这里的 valid mask 只排除 crop/pad，不因为语义 Label 为 255 就排除真实图像区域。
- inverse normalization 会把 normalized zero 还原到通道均值，Depth 约为 uint8 `122`。这不是观测值，所以 corruption 前先把 pad 的 raw bytes 置零，避免 misalignment 或 blur 把伪造的均值移入有效区域。
- corruption 后 pad 的 normalized 输入恢复为原始精确 `0`，target 在 pad 区设为中性 `1`，同时由 `valid_mask` 排除；clean sample 直接复用原 normalized tensor，不做 round-trip 重建。
- 输出接口包括 `rgb`、`depth`、`raw_rgb`、`raw_depth`、`reliability_target`、`valid_mask`、`depth_valid` 和 CPU-only metadata。`depth_valid` 是 corrupted raw Depth `>0` 与 `valid_mask` 的交集。

### 9.3 A2 Depth-only 采样

- 每个样本先由独立 RNG 抽 clean/corrupt：`p_clean=0.25`，`p_corrupt=0.75`。
- corrupt 样本在抽 spec 数量和种类前，先限制候选为 Depth；这样不会先抽 RGB+Depth 再过滤 RGB，避免改变 spec 数量分布。
- Depth-only 当前允许六类 A1 kind：`entire_missing`、`spatial_dropout`、`gaussian_noise`、`blur`、`quantization`、`misalignment`。
- clean sample 的 RGB/Depth normalized segmentation 输入保持 exact no-op，reliability target 全 1；Depth-corruption sample 的 RGB 保持原输入，Depth 按 result 重新 normalized，且仍保持原输入 channel count。

### 9.4 per-sample PCG64 SeedSequence

A2 corruption seed 与模型 seed 分离：corruption seed 为 `2026091402`，模型 seed 为 `772961337`。每个样本建立独立 `numpy.random.Generator(PCG64(SeedSequence(words)))`，不读取或修改全局 RNG。

固定 seed words 顺序为：

1. `train_seed`；
2. `epoch_1_based`；
3. `iteration_0_based`；
4. `global_rank`；
5. `sample_slot_0_based`；
6. `sample_id_sha256_u32_be_0`；
7. `sample_id_sha256_u32_be_1`；
8. `sample_id_sha256_u32_be_2`；
9. `sample_id_sha256_u32_be_3`。

sample id 先统一路径分隔符为 `/`，再取 SHA-256 前 16 bytes，按大端拆成 4 个 uint32。当前训练进度使用训练位置，不使用 wall clock：

$$
progress=\operatorname{clip}_{[0,1]}\left(\frac{(epoch-1)N_{iter}+iteration}{N_{epoch}N_{iter}-1}\right).
$$

相同 sample id、epoch、iteration、rank、slot 和 protocol 必须得到相同 spec、corruption、target 和 metadata。这个保证只针对 A2 corruption；DataLoader 内的 mirror/scale/crop 仍属于旧训练 RNG 语义。若未来要求任意中途 resume 后几何增强也逐像素相同，必须建立新 protocol。

## 10. 模型、可靠性特征与损失

### 10.1 14 通道固定信号特征

`SignalFeatureExtractor` 没有可学习参数，只注册固定卷积核和常数。输入为 raw `[0,1]` RGB/Depth 及 Depth validity，输出 `[B,14,H,W]`。14 个通道为：

- RGB：intensity、local mean、local standard deviation、Sobel magnitude、Laplacian magnitude、Gaussian high-frequency residual，共 6 个；
- Depth：同样的 6 个信号，共 6 个；
- Depth validity，共 1 个；
- RGB/Depth cross-modal gradient alignment，共 1 个。

局部统计使用 3×3 window；高频残差使用 5×5 Gaussian kernel；Sobel 和 Laplacian 使用固定核。cross-modal alignment 是经有效性约束的 cosine-based 对齐量，范围映射到 `[0,1]`。这些信号只是 reliability head 的输入，不能被审计成 hard gate，也不能被解释为传感器真值。

### 10.2 reliability head、pyramid 和未来 adapter

- 当前 head 结构为三层卷积：`14 -> 16` 的 3×3 Conv、ReLU，`16 -> 16` 的 3×3 Conv、ReLU，`16 -> 2` 的 1×1 Conv。输出是 RGB/Depth 两个 reliability logits，之后使用 sigmoid 得到 `[0,1]` reliability。
- A1 同时定义四级 `ReliabilityPyramid`，尺度为 `4、8、16、32`，每级使用 PyTorch `area` 插值。它保持 `[0,1]` 语义，但不声称等同于 DVG-B1 使用的 OpenCV `INTER_AREA`，也不复用 DVG-B1 Oracle protocol。
- 未来 `GeometryContributionAdapter` 对每级 reliability 做均值和标准差统计，输入为 `[B,4L]`，默认 `L=4`、hidden size `8`，输出每级 spatial/depth 两个 scale。输出范围严格在开区间 `(0,2)`，输出层零初始化时每个 scale 严格等于 `1`，因此未训练时对旧 geometry prior 是 exact no-op。
- 未来 B1 的抽象组合可以写成：

$$
G_l=a_l(\hat R_{rgb},\hat R_d)G_{s,l}+b_l(\hat R_{rgb},\hat R_d)G_{d,l},
\qquad 0<a_l,b_l<2.
$$

这只是后续 adapter 设计接口，不是当前 A2 已启用的模型路径。

### 10.3 FP32 reliability 分支

首次真实 AMP forward 在信号特征中发现，梯度平方和的乘积在 FP16 下溢为 0，导致 cosine 分母为 0 并产生非有限值。修复后：

- segmentation backbone 和 Ham decoder 继续使用外层 AMP/FP16；
- reliability 固定特征、head 和连续 BCE 在 FP32 autocast-disabled 区域计算；
- 该修复不改变 reliability 不进入 backbone/geometry prior 的 A2 协议；
- FP32 分支的显存、吞吐和完整 batch size 10 成本仍需要正式 probe/训练测量，不能根据单步 preflight 猜测。

### 10.4 总损失

A2 每个样本只执行一次 segmentation forward。segmentation loss 为带有效语义标签 mask 的 `safe_masked_mean(cross_entropy(input,label))`；reliability loss 为连续 target 的稳定 BCE，`valid_mask` 只排除 crop/pad：

$$
\mathcal L=\mathcal L_{seg}^{input}+\lambda_{rel}\mathcal L_{rel},
\qquad \lambda_{rel}=0.1.
$$

当前冻结的其他项为：

- `lambda_consistency=0.0`；
- 不做 clean teacher；
- 不做 clean-corrupt consistency；
- 不做蒸馏；
- 不启用 geometry adapter；
- reliability prediction 不进入 backbone、decoder 或 geometry prior；
- A2 中 RGB target 始终为 1，Depth target 来自 Depth corruption。

### 10.5 当前配置的参数计数

按当前配置的 CPU 构建计数，clean control 为 `26,673,193` 个参数，Depth-corruption 配置为 `26,677,579` 个参数，reliability 分支新增 `4,386` 个参数，增幅约 `0.0164435%`（约 `0.0164%`）。新增量可由当前 head 结构直接计算：

$$
14\times16\times3\times3+16
+16\times16\times3\times3+16
+16\times2\times1\times1+2
=4,386.
$$

这些是当前配置的 CPU 构建计数事实，不是训练后效果或资源成本结论；正式 batch size 10 的显存和吞吐仍待云端 probe。

## 11. 两个公平训练身份

### 11.1 共同冻结项

`MMFR-A2-clean-control-v1` 与 `MMFR-A2-depth-corruption-train-v1` 必须共同使用：

- 同一官方 `DFormerv2_Small_pretrained.pth`；
- 同一模型 seed `772961337`；
- 同一 `train-dev=1277`、`val-dev=318` 和 split identity；
- 同一 RGB input contract `rgb-imagenet-rgb-order-v1`；
- 同一 AdamW、learning rate `6e-5`、weight decay `0.01`、500 epoch、warmup 10 epoch、poly power `0.9`；
- 同一 batch size `10`、workers `8`、尺度增强 `[0.5,0.75,1.0,1.25,1.5,1.75]`；
- 同一 clean `val-dev` selector、评估间隔、tie-break 和主 evaluator；
- 同一 checkpoint 保存/恢复语义、运行身份、协议和证据 manifest 规则。

### 11.2 唯一科学变量

- clean control 保持 pre-A2 模块集合和 clean 输入，不实例化 reliability head，不执行 corruption。
- Depth-corruption model 执行冻结的 Depth-only corruption，增加 reliability head 和 `0.1 × continuous BCE` 辅助项。
- 任何单独改变增强、优化器、seed、epoch、split、selector、checkpoint 起点或 evaluator 的运行都不再是该公平对照的一部分，必须使用新 protocol identity。
- 两个身份都必须从官方 pretrained 独立开始；不能从 Quick-B0 epoch 420 checkpoint 续训后与 clean control 或 Quick-B0 直接宣称公平。
- 失效条件不参与 checkpoint selector。两个身份可以按同一 selector 规则选出不同 epoch，但不能为其中一个追加候选、改变 tie-break 或使用 corruption 条件筛选。

## 12. 云端执行边界、预算与生命周期

### 12.1 冻结执行配置

- 正式训练环境：云端单 GPU。
- 默认主选：RTX 4090 24GB，价格快照 `1.88 元/小时`。
- 备用：RTX 5090 32GB，价格快照 `2.78 元/小时`。
- 5090/4090 价格比和单位样本成本 break-even 吞吐比：

$$
\frac{2.78}{1.88}=1.4787234043\approx1.4787.
$$

只有 5090 在相同科学配置下实测吞吐超过约 `1.4787×` 4090，单位时间价格更高的 5090 才可能有更低的单位样本成本。

- 冻结 global batch size：`10`。
- DataLoader workers：`8`。
- 优化器：AdamW。
- learning rate：`6e-5`。
- 总训练：`500` epoch。
- warmup：`10` epoch。
- AMP：开启。
- SyncBN：开启。
- `torch.compile`：首轮关闭。
- seed：`772961337`。
- 换 GPU 不能改变科学超参数；5090 的营销 AI TOPS 不能替代同口径 FP16 训练吞吐证据。

### 12.2 当前未授权的云操作

- 当前没有云实例，也没有创建实例或产生云费用。
- 当前没有授权 batch size 10 的容量/吞吐 probe。
- 当前没有授权 `MMFR-A2-clean-control-v1` 训练。
- 当前没有授权 `MMFR-A2-depth-corruption-train-v1` 训练。
- 当前没有授权完整 epoch、checkpoint save/load、DDP、完整 evaluator 或 official test。
- 本机 NVIDIA GeForce RTX 5060 Laptop 只承担推理、想法初步验证和小规模 preflight，不作为正式长训练环境候选。

### 12.3 生命周期审计要求

正式云任务即使未来获批，也必须先完成一次该运行的生命周期授权和最晚停止 schedule：

1. 在启动前明确实例、最长运行时长和预计费用；
2. 使用控制面 schedule 设置最晚停止时间，并复核 schedule；
3. workload 成功、失败或人工中止后，都先取回必要证据并核验 SHA-256，再调用控制面 stop；
4. stop 后复查平台状态确实为 `Stopped`；实例内 `shutdown -h` 不能单独证明平台已停止计费；
5. 验收 pass/fail 只决定研究结论，不决定是否停止计费；失败路径也必须自动停止；
6. A2 当前尚未运行云端生命周期测试，因此不能把历史其他方向的生命周期通过自动移植成 A2 本次训练已经通过。

## 13. 已完成的资格证据及其边界

### 13.1 CPU qualification

已完成的受影响 Python 文件静态编译、静态诊断、限定 CPU probe 和 `git diff --check`。定点检查覆盖：

- exhaustive uint8 normalization round-trip；
- A1 各类 corruption 的 shape、dtype、range、finite 和 metadata；
- 同一位置的确定性与不同 sample slot 的分离；
- Depth-only curriculum 和三阶段 severity/count 语义；
- clean exact normalized no-op；
- pad/raw/target 中性语义；
- misalignment 不把伪造的 Depth 均值字节 `122` 移入有效区；
- 完整 reliability supervision 字段守卫；
- finite scalar loss、finite backward；
- A1 reliability pyramid 和 adapter 初始 exact-one。

CPU qualification 没有读取真实 checkpoint、没有执行真实 Ham forward、没有运行完整训练或评价。

### 13.2 本地 RTX 5060 Laptop 单步 preflight

最终 canonical preflight 的实际事实：

- 环境：Python `3.10.20`、PyTorch `2.7.0+cu128`、CUDA `12.8`、cuDNN `90701`、driver `610.88`、NVIDIA GeForce RTX 5060 Laptop GPU。
- 使用真实 `train-dev`、官方 pretrained、Ham decoder、SyncBN、AMP、Depth corruption 和 reliability auxiliary loss。
- 尝试 `6` 个 batch；前 `5` 次由 GradScaler 跳过 optimizer update；第 `6` 个 batch 完成 `1` 次 optimizer update。
- 6 个 loss 均有限，累计为 `1` 个 clean sample 和 `5` 个 corrupted sample。
- 最终更新步 loss 为 `3.3470830917`。
- 峰值 allocated/reserved CUDA memory 为 `2069.91/2260` MiB；结束时 free ratio 为 `0.5812`。
- `training_result.json` 记录 `attempted_steps=6`、`completed_optimizer_steps=1`、`skipped_optimizer_steps=5`、`checkpoint=null`、`official_test_included=false`。
- 首次运行发现 reliability 固定信号特征在外层 FP16 下溢，修复为 reliability 分支 FP32；随后发现短 probe 不应 finalize 需要 `latest.pth` 的 selector，修复为 probe 不创建或 finalize selector，但仍保存 telemetry 和 `training_result.json`。
- canonical 输出目录：`outputs/mmfr-a2-gpu-preflight-20260914T0138Z/`。
- protocol manifest SHA-256：`e9825c4a4818cf2860a529b6c0fa542f5412a7153a1e8b9b8c8d000de6ca7f07`。
- repository template（preflight 状态回填后）SHA-256：`b5b5c979352cca451cfbca35abc77232a9ae82c3f22ec0f33fcdb964ae03e519`。
- `run_config.json` SHA-256：`2ff6bf4668efbb58012fced89af698ce4cfec5181c43a4ee4dec9e56dce4edd3`。
- `training_result.json` SHA-256：`bb3d71dacc99d7b1b97eb03d702f0eff621f78669b7dbb5824aaafad832550c0`。
- `probe-telemetry.jsonl` SHA-256：`6ae56d55552aee9cacda71b09fea7269069814e18d81e1d3548539d0995d462c`。
- run config 显式记录工作区 `dirty=true`；这意味着该证据不能被误写成来自干净提交的正式训练结果。

**证据边界：** 这次 preflight 只证明当前接线在 batch size 1、小步数、单本地 GPU 条件下能够完成一次有效参数更新。它不覆盖冻结 batch size 10 的显存容量、完整 500 epoch、checkpoint save/load、DDP、`torch.compile`、云端资源、正式 evaluator、完整测试或 official test；更不提供 mIoU、鲁棒性收益或研究结论。

## 14. 正式开发评价设计（未来授权后执行）

### 14.1 checkpoint selector

- 每个训练身份分别运行相同 clean `val-dev` selector。
- selector geometry：`original-full`。
- selector scale：`1.0`。
- selector flip：`false`。
- selector metric：mIoU。
- tie-break：`earlier_epoch`。
- failure condition 不参与 selector。
- selector 完成后冻结 checkpoint，再进入主 evaluator；不得看到 corruption 结果后为某一身份追加 epoch 候选。

### 14.2 主 evaluator

- evaluator：`msflip-whole-original-grid-v1`。
- 5 scales：`0.5、0.75、1.0、1.25、1.5`。
- 每个 scale 原图与水平翻转，共 10 view。
- evaluation corruption 在原始对齐 RGB/Depth 上按 evaluation seed `2026091401` 和 sample/condition identity 一次生成，再由 10-view evaluator 共同做 geometry transform；不能为每个 view 重新抽取不同 failure。
- 每个 view 的 logits 回到原始 Label 网格，在 FP32 中平均 pre-softmax logits 后计分。
- 评价范围固定为全部 `318` 张 `val-dev`，不把其中一部分事后改名为独立 test。
- 配对 bootstrap 单位固定为 `196` 个 location groups，重采样组内全部图像并保持 condition pairing。

### 14.3 固定条件

六个 Depth 单失效条件：

1. `spatial_dropout@0.75`；
2. `gaussian_noise@0.75`；
3. `blur@0.75`；
4. `quantization@0.75`；
5. `misalignment@0.75`；
6. `entire_missing@1.0`。

三个固定混合条件：

1. `spatial_dropout@0.5 + gaussian_noise@0.5`；
2. `blur@0.5 + misalignment@0.5`；
3. `quantization@0.5 + misalignment@0.5`。

混合 spec 的顺序固定，不能根据结果交换顺序、修改 severity 或追加组合。clean、六个单失效和三个混合条件都应报告；混合条件不进入主单失效 score。

### 14.4 主成功门槛与声明上限

比较对象为 `corruption model minus clean control`。主 robustness score 是六个单失效 mIoU 的等权宏平均。主成功必须同时满足：

1. 六个单失效的宏平均 mIoU 点估计至少为 `+1.00` 个百分点；
2. 该宏平均的 paired 95% percentile interval 下界严格大于 `0`；
3. clean mIoU 点估计下降不超过 `0.50` 个百分点；
4. 六个单失效中至少五个 mIoU 点估计不为负，且任一单条件不得低于 `-1.00` 个百分点。

- mAcc、mF1、Boundary IoU、clean Boundary IoU 和三个混合条件是辅助结果，用于检查类别准确率、F1、边界几何和混合失效行为。
- 辅助结果不能事后替代主门槛，把主失败改写成成功。
- 因 `val-dev` 同时承担 checkpoint 选择和开发评价，达到门槛时最高只能写成 `single-seed development-supported`。不能写成独立测试通过、真实部署可靠或自然故障已解决。
- 不能把预期门槛写成已取得结果。当前所有 mIoU、mAcc、mF1、Boundary IoU 的 MMFR 新模型结果均为未产生。

## 15. 可靠性专门评价：建议的新补充协议

本节不是当前 `MMFR-A2-train-integration-v1` 的主成功门槛，而是建议在正式结果出现前单独决定是否采纳的补充协议。它不能在看到结果后用来挽救一个未达到主门槛的训练结果。

### 15.1 三层问题必须分开

1. **预测合成 target：** reliability head 是否能复现合成器定义的 $R^*$。可考虑 MAE、Brier score、校准误差和 reliability diagram。
2. **预测实际分割风险：** 预测的 reliability 是否与真实像素分割错误、类别错误或失效严重度相关。可考虑 AUPRC、校准曲线、严重度单调性、risk-coverage 和 AURC（Area Under the Risk-Coverage curve）。
3. **带来下游收益：** reliability 是否通过 adapter 或其他已冻结融合动作改善 clean/单失效/混合条件的分割指标。只有这一层涉及下游模型收益，不能用第一层的 target 拟合好替代它。

### 15.2 采纳条件

- 如果采纳任何 reliability 专门指标，应在正式结果查看前建立新的 protocol identity，或建立明确标记为 supplemental 的独立 protocol。
- 新 protocol 必须提前写明 target、错误定义、有效像素域、采样单位、bootstrap、阈值、是否多重比较以及失败处理。
- reliability target 仍是合成 target；没有自然故障真值时，不能把校准图写成现实传感器校准证明。
- 不得结果后从 MAE 改成 Brier、从像素 risk 改成图像 risk、从 AUPRC 改成 AUROC，或只挑对当前结果最有利的层级。

## 16. 预期变化与禁止的结果叙事

以下是实验前的合理预期，不是实验结论：

- 参数量预计几乎不变；当前 CPU 构建计数显示新增约 `0.0164%`，这不是性能保证。
- clean 指标的目标是基本稳定，但不能保证超过 Quick-B0 的 mIoU `58.79`；clean control 与 corruption model 还没有正式结果。
- 主要期待是 corruption 条件下的分割性能改善，尤其是六个单失效的宏平均；是否发生必须由冻结 evaluator 和 paired bootstrap 决定。
- 当前 A2 reliability 不反馈模型动作，因此任何潜在收益主要来自训练期间暴露于 corruption 分布及其辅助 target，而不是测试时主动关闭 geometry prior。
- 不能写“clean 已保持”“corruption 已提升”“可靠性已经校准”或“模型已经适合真实故障”，除非未来有对应冻结身份、完整产物和直接核验的结果。

## 17. 风险清单与停止条件

### 17.1 研究解释风险

- **自然故障外推：** 没有自然故障类型、位置、严重度、故障率和标定漂移标签。若审计发现结论越过合成开发边界，停止并收窄结论。
- **RGB-centric 风险：** A2 只做 Depth corruption；即使 Depth 失效结果支持，也不能声称 RGB complete-missing 已解决。若目标要求 RGB 完全失效，转入 B2 设计，不向 A2 静默加功能。
- **synthetic-to-real gap：** target 可能只识别 A1 合成器的统计痕迹。若 reliability head 只拟合 synthetic target 而与 segmentation risk 无关，应将其写成 target fidelity，不写成现实可靠性。
- **单 seed 风险：** 单 seed 不能估计随机方差。小增益、接近训练波动或重要结论应增加成对 seed，并建立新预算和 protocol。
- **val-dev 双重职责：** 同一 `val-dev` 用于 selector 和开发评价，不能称独立测试；若未来需要无选择偏差评价，应另建数据职责或独立协议。

### 17.2 实现和统计风险

- **target/geometry 错位：** corruption 必须在最终 mirror/scale/crop/pad 后执行；任何中途 corruption 或独立 target 几何会使监督错位，立即停止。
- **pad 伪值泄漏：** inverse-normalized pad 的均值字节必须在 corruption 前置零，corruption 后 pad 必须回到 normalized exact zero；发现 pad 被当作观测，立即停止。
- **Depth-only 泄漏：** 当前 config 只能抽 Depth；发现 RGB spec 被混入、先抽 RGB 再过滤、或 RGB target 非 1，立即停止。
- **supervision 静默缺失：** reliability head 启用时五个辅助字段必须完整；缺字段或错配必须 fail-closed，不能以零 loss 继续。
- **Ham decoder RNG：** Ham decoder eval 可能随机初始化 NMF bases。paired forward 或基线/模型比较必须在每对 forward 前回放相同 CPU/CUDA RNG state，否则不能把 logits 差异归因于 MMFR。
- **指标定义漂移：** Boundary IoU、mIoU、metric geometry、ignore/background 域和 bootstrap 单位不能随结果变化；发现定义不一致，停止裁决并建立新 protocol。
- **checkpoint selector 越权：** 失效条件不得参与 selector；不得看到结果后追加 checkpoint、改 tie-break 或使用不同候选数量。

### 17.3 资源和运行风险

- **FP32 auxiliary 开销：** reliability 分支需要 FP32 以避免 FP16 下溢，可能改变显存和吞吐；batch size 10 云端容量/吞吐必须实测。
- **batch size 10 容量：** batch size 1 preflight 不能证明 batch size 10 可容纳。若 4090 24GB 容量门禁失败，不能直接改 batch size、学习率或科学 schedule；只能按已冻结规则考虑 5090 或回报阻塞。
- **checkpoint save/load：** preflight 没有生成正式 checkpoint，也没有验证完整模型与 optimizer state 的 strict 保存/恢复。该检查未通过前不能开始依赖 checkpoint 的正式评价。
- **DDP/SyncBN/torch.compile：** 单 GPU preflight 不覆盖 DDP，多卡同步语义或 compile；正式首轮 compile 关闭，任何打开 compile 或多卡都需新定点验证。
- **云生命周期：** 没有最晚停止 schedule、控制面 stop 和 `Stopped` 复查就不能启动付费训练；验收失败不能成为继续计费的理由。
- **official test：** 任何意外读取、解封或将 official test 用于选择的行为都是硬阻塞，需要保留证据并停止当前方向。

### 17.4 通用停止条件

遇到以下任一情况，应停止扩大修改或运行，保留已完成部分并回到主代理裁决：

1. 需要修改 DFormerv2 encoder、evaluator、checkpoint loader、A1 数值语义或 official-test 文件；
2. 需要改变已冻结 split、seed、condition、severity、selector、成功门槛或 bootstrap 单位；
3. 实际运行的身份、pretrained、Git 工作区、protocol hash 与预登记记录不一致；
4. clean no-op、Depth-only、pad neutral、finite loss、reliability 完整字段或 seed words 检查失败；
5. batch size 10 显存不足且拟通过更改科学超参数规避；
6. checkpoint 不能 strict save/load，或 selector 需要不存在的文件；
7. 云实例不能在成功、失败或中止后按控制面规则停止；
8. 需要把 preflight、计划、资源 probe 或 target fidelity 写成模型效果；
9. 任何人要求在结果后修改规则、追加 condition、改阈值、删样本、删 location group 或追加 seed 以获得支持性结论。

## 18. 外部审计清单

外部 agent 应逐项给出“通过、需补证据、需修订、阻塞”及理由，而不是只给总体印象。

### 18.1 高风险项目

- [ ] 事实一致性：Quick-B0 checkpoint、pretrained、split、指标、DVC/DVG 历史终态和 A2 preflight 数字是否与本文一致。
- [ ] 当前状态：是否明确没有正式 MMFR checkpoint、没有正式 MMFR 指标、没有云实例、没有云 probe、没有 official test。
- [ ] 公平性：两个训练身份是否都从相同 official pretrained 独立开始，是否误用 Quick-B0 checkpoint 续训，是否共享 seed/split/optimizer/schedule/selector。
- [ ] 结果后选择空间：condition、severity、样本、group、checkpoint 候选、阈值、主/辅助指标和可靠性 supplemental protocol 是否都在结果前冻结。
- [ ] 数据结论边界：是否把合成 corruption 误写成自然故障标签、故障率或部署证据。
- [ ] A2 定位：是否把 reliability head 的辅助监督误写成已经参与 fusion，是否把 A2 误写成最终 reliability fusion model。
- [ ] RGB 边界：是否把 Depth-only 训练结果扩展成 RGB complete-missing 能力。
- [ ] 评价支持性：主门槛是否确实由六个单失效 mIoU 宏平均、paired 95% interval、clean drop 和逐条件保护共同决定。
- [ ] 实现泄漏：reliability target 是否仅来自合成器 metadata，valid mask 是否只排除 pad，Label 是否没有泄漏进 reliability head 输入，corruption 是否发生在最终 crop 后而不是使用未来信息。
- [ ] 无效监督：head 启用时是否 fail-closed 检查 raw inputs、target、valid mask、depth_valid；pad target 是否中性且被 mask 排除。
- [ ] 云成本和停止：是否先做容量/吞吐 probe、设置最晚 stop schedule、核验控制面 `Stopped`，并在失败时停止计费。

### 18.2 中风险项目

- [ ] 14 通道特征是否真的保持 `[0,1]`、finite、无意中变成 hard gate。
- [ ] A1 六类 failure 的 severity、curriculum、组合顺序、burden 和 target 是否与 protocol 一致。
- [ ] per-sample PCG64 SeedSequence words 是否顺序固定，sample id 是否统一 `/` 后取 hash 前 16 bytes。
- [ ] normalized/raw/pad round-trip 是否对全部 uint8 逐值通过，Depth channel count 是否在 clean/corrupt 两条路径一致。
- [ ] FP32 reliability 分支是否只修复数值稳定性而没有偷偷改变 loss 权重或模型反馈路径。
- [ ] Ham decoder 的随机 NMF bases 是否在 paired comparison 中使用相同 CPU/CUDA RNG state。
- [ ] 可靠性专门评价是否在查看结果前建立新 protocol 或 supplemental identity，是否区分 synthetic target fidelity、actual segmentation risk 和 downstream gain。
- [ ] single-seed 和 `val-dev` 双重职责是否在最终报告中被明确限制。
- [ ] 新增参数计数 `4,386`、clean `26,673,193`、corruption `26,677,579` 是否由实际当前配置复核，而不是从预期数字猜测。

### 18.3 低风险项目

- [ ] 4090/5090 价格快照、1.4787 break-even 规则和换卡不改科学超参数是否被记录。
- [ ] 首轮 `torch.compile` 关闭、AMP/SyncBN 配置和本机/云端职责是否写清楚。
- [ ] 证据文件、JSON、日志和 checkpoint 是否都有路径、身份和 SHA-256 指针。
- [ ] 本文是否没有使用 Markdown 表格，所有 Markdown 数学公式是否使用 `$...$` 或独立 `$$...$$`。

## 19. 状态矩阵（用项目符号表达）

### 19.1 已完成

- A1 standalone corruption 模块 `utils/dataloader/multimodal_failure.py` 已实现并通过 CPU qualification。
- A1 reliability feature/head/pyramid/adapter 模块 `models/modal_reliability.py` 已实现并保持 standalone 语义。
- A2 post-crop/pre-GPU batch helper `utils/dataloader/mmfr_training.py` 已接入限定训练路径。
- `models/builder.py` 和 `utils/train.py` 已增加可选 reliability auxiliary 接口；配置关闭时保持旧 scalar loss 路径。
- common、clean control、Depth-corruption 三个独立 A2 config 和 protocol template 已形成。
- A2 protocol、RNG、target、raw/normalized、pad、loss、selector、云端单卡职责和两训练身份已冻结。
- CPU qualification 已完成；本地 RTX 5060 Laptop 单步 GPU preflight 已完成。

### 19.2 已通过但不构成效果

- exhaustive uint8 round-trip、确定性、Depth-only、curriculum、pad neutral、clean exact no-op、finite loss/backward 已通过。
- 真实本地 GPU preflight 已完成 6 batch 尝试、5 次跳过、1 次 optimizer update，全部 loss 有限。
- FP16 下溢和 probe selector finalize 两个具体运行阻塞已发现并修复。
- 这些证据只说明当前代码链在限定条件下可运行，不能说明 MMFR 提升 mIoU、校准可靠性、改善自然故障或带来部署收益。

### 19.3 未授权

- 云端 batch size 10 容量/吞吐 probe。
- RTX 4090 或 RTX 5090 云实例创建、连接、执行和停止。
- `MMFR-A2-clean-control-v1` 500 epoch 正式训练。
- `MMFR-A2-depth-corruption-train-v1` 500 epoch 正式训练。
- 正式 checkpoint save/load 验收、DDP、compile 路径和完整 epoch。
- 冻结 checkpoint 后的 318 样本、10-view、六单失效和三混合条件评价。
- reliability 专门 supplemental protocol 的建立和执行。
- official test 的解封、读取或评价。

### 19.4 待设计

- `MMFR-B1-learned-geometry-adapter-v1` 的独立代码、loss、checkpoint 和成功门槛。
- 如果要覆盖 RGB complete-missing，`MMFR-B2` 的 Depth semantic path 或双语义 path。
- reliability 专门评价是否采纳，以及其新 protocol/supplemental identity、指标、风险单位和阈值。
- 自然故障数据、故障率、重复采集和标定漂移证据；没有这些证据时只能保持合成开发结论。

### 19.5 禁止

- 从 Quick-B0 epoch 420 checkpoint 续训后冒充公平 A2 消融。
- 修改 DVC/DVG 历史 protocol、负结果或已关闭的 `oracle-not-supported` 结论。
- 把 A1 脚手架、CPU qualification 或单步 preflight 写成模型效果。
- 在结果后修改 failure 定义、severity、condition、sample/group、selector、bootstrap、阈值或主 score。
- 把 `val-dev` 称为独立测试，或读取 official test。
- 因 4090 容量不足而静默修改 batch size、learning rate、epoch、seed 或其他科学超参数。
- 只训练 clean control 或只训练 corruption model 后宣称公平对照已经完成。
- 把 reliability target fidelity 直接写成自然传感器可靠性或实际分割风险证明。

## 20. 文件地图与证据指针

以下是本文形成时限定读取的关键文件及其职责。外部 agent 看不到这些文件时，应把它们当作待补证据清单，而不是默认已核验。

- `doc/main/MUSeg-current-status.md`：MUSeg 当前事实、阶段、checkpoint/指标身份、已完成历史、授权边界和准确恢复点的唯一实时入口。
- `doc/main/MUSeg-open-decisions.md`：研究口径和处置状态；第 8 节说明 single-seed B0 与模块消融，第 12 节记录 DVG-B1 Oracle 负结果，第 13 节记录 MMFR 方向和 A2 当前开放授权。
- `doc/plans/2026-09-MUSeg-多形式模态失效可靠性学习/00-总方向规划.md`：MMFR 总方向、自然故障证据边界、A1/A2/B1/B2 阶段关系和方向级禁止事项。
- `doc/plans/2026-09-MUSeg-多形式模态失效可靠性学习/01-新对话最小上下文与当前任务.md`：当前任务边界、恢复点和未授权的下一项任务卡。
- `doc/plans/2026-09-MUSeg-多形式模态失效可靠性学习/02-MMFR-A1失效基函数与可靠性脚手架.md`：A1 类型、target、信号特征、未来 adapter 接口及 CPU qualification 结果。
- `doc/plans/2026-09-MUSeg-多形式模态失效可靠性学习/03-MMFR-A2训练接入与公平对照协议.md`：A2 数据流、RNG、loss、selector、评价条件、主门槛、云端职责和 preflight 证据。
- `protocols/mmfr-a2-train-integration-v1.template.json`：A2 结构化 protocol 身份、数据样本数与 hash、训练超参数、corruption、selector、评价和授权字段。
- `local_configs/MUSeg/DFormerv2_S_MMFR_A2_Common.py`：两个 A2 身份共同继承的 split、RGB input、official pretrained、optimizer、500 epoch schedule、selector、云端 profile 和 protocol bookkeeping。
- `local_configs/MUSeg/DFormerv2_S_MMFR_A2_Clean.py`：clean control 的独立 run identity；不启用 corruption 和 reliability head。
- `local_configs/MUSeg/DFormerv2_S_MMFR_A2_DepthCorrupt.py`：Depth-corruption 独立 identity；启用 Depth-only corruption、hidden size 16 reliability head 和 `0.1 × BCE`。
- `utils/dataloader/multimodal_failure.py`：A1 的 uint8 RGB/Depth failure kernels、连续 burden/target、组合和 metadata。
- `utils/dataloader/mmfr_training.py`：A2 的最终 CPU batch helper、raw/normalized round-trip、pad/valid 语义、Depth-only spec 抽样、per-sample seed words 和 audit metadata。
- `models/modal_reliability.py`：14 通道 fixed signal features、reliability head、PyTorch area pyramid、连续 BCE 和未来 geometry contribution adapter。
- `models/builder.py`：可选 reliability estimator 的实例化、A2 辅助 loss、FP32 reliability 分支和 supervision 字段 fail-closed 检查；配置关闭时旧路径保持不变。
- `utils/train.py`：A2 config 守卫、CPU batch helper 调用、GPU 搬运、AMP forward、optimizer step/skipped step 计数、checkpoint selector 和 telemetry。
- `doc/reports/2026-08-31-museg-quick-b0-main-evaluation.md`：Quick-B0 的 baseline 身份、epoch 420 checkpoint、10-view evaluator、mIoU/mAcc/mF1 和 single-seed 边界。
- 仓库外或被忽略输出中还应补交 A2 preflight 的 `run_config.json`、`training_result.json`、`probe-telemetry.jsonl` 及其 SHA-256；DVC/DVG 历史证据也应按本文列出的 protocol/manifest/execution hash 补交。

## 21. 审计结论模板

外部 agent 可按以下三种结论之一收口，不得用模糊的“基本可以”：

### `可进入云端 probe`

使用条件：

- 当前事实、A2 protocol/template、两个 config、pretrained、split、seed、selector 和工作区身份已经一致核验；
- CPU qualification 与本地单步 preflight 的边界已经被正确理解为链路资格，而非效果；
- 云端单 GPU 容量/吞吐 probe 的命令、最长时长、费用上限、4090/5090 选择规则和停止 schedule 已独立授权；
- probe 只回答 batch size 10 是否能装入、吞吐和单位样本成本，不能顺便运行 500 epoch 或评价模型。

### `需修订后进入`

适用情形：

- 发现文档、代码、template 或运行 artifact 的普通身份字段、解释或证据指针不一致，但可以在不改变科学问题和冻结门槛的情况下修正；
- 发现需要补一项有限的 CPU/小规模 preflight，例如 checkpoint save/load 或云生命周期门禁，但没有必要改变架构或评价口径；
- 发现可靠性 supplemental protocol 尚未在结果前建立，需先明确是否采纳和如何独立登记。

修订后必须重新检查差异、协议 hash、授权范围和最小定点验证，不得直接进入正式训练。

### `阻塞`

以下任一情形都足以阻塞：

- 无法确认工作区代码、protocol/template、pretrained、split、checkpoint 或运行身份；
- 需要从 Quick-B0 续训，或两个公平身份无法从同一 official pretrained 独立开始；
- 需要把自然故障、真实部署或 RGB complete-missing 写入 A2 能力声明；
- batch size 10 容量、checkpoint save/load 或云 lifecycle 尚未达到必要门禁却要开始付费长训练；
- 需要在结果后改变 condition、severity、selector、样本/group、阈值、主指标或补充指标；
- 发现 target/valid/pad/Depth-only/FP32 分支存在泄漏、错位、缺失字段静默继续或非有限 loss；
- 任何 official test 已被意外读取，或已有证据不能证明其仍为 `sealed_unread`。

## 22. 准确恢复点与不得执行事项

### 22.1 准确恢复点

截至 2026-09-14 02:02 UTC 的准确恢复点是：`MMFR-A2-train-integration-v1` 已达到 `code-qualified` 和 `GPU-single-step-qualified`；A1 已完成；A2 protocol、限定代码接入、CPU qualification 和本地 GPU 单步 preflight 已完成。正式训练职责为云端单 GPU，默认 RTX 4090；本机只做推理和小规模 preflight。当前没有云实例、没有正式训练 checkpoint、没有 MMFR 新指标、没有完整评价，也没有 official test 结果。

下一步只有在单独授权后：

1. 先做冻结 batch size 10 的云端容量/吞吐短 probe；
2. 按 4090 优先、5090 仅在冻结规则触发时备用的资源选择；
3. 在相同官方 pretrained、seed、split、optimizer、schedule、selector 和 protocol 下，独立启动 clean control 与 Depth-corruption 两个公平训练；
4. 在保存/恢复和证据 hash 门禁通过后，再按冻结的 318 样本、10-view、六单失效、三混合条件和 196 location-group paired bootstrap 评价；
5. 若未来进入 B1/B2 或采纳 reliability supplemental protocol，必须建立独立 identity，不得借用 A2 v1 的授权。

### 22.2 不得执行

- 不得在当前授权下创建云实例、运行容量/吞吐 probe、启动 500 epoch 训练或产生云费用。
- 不得运行完整 MMFR evaluator、读取 official test 或生成新的 MMFR checkpoint 评价。
- 不得把本文中的“冻结计划”“主成功门槛”“预期变化”写成已完成实验结论。
- 不得修改 DVC/DVG 历史 protocol、负结果或 `oracle-not-supported` 终态。
- 不得扩大 A2 为 RGB corruption、learned geometry adapter、Depth semantic fallback 或最终可靠性融合模型。
- 不得根据未来结果删样本、删组、调 severity、加 condition、改 selector、改阈值或挑选 supplemental 指标。
- 不得从 Quick-B0 epoch 420 续训后宣称公平；不得只训练一侧对照后宣称比较完成。
- 不得修改 official test 的封存状态。

本文件是 2026-09-14 的独立审计快照；项目实时事实仍以 `doc/main/MUSeg-current-status.md` 为准。为满足本次“只允许写入一个新文件”的边界，本文形成过程中没有修改状态文件、开放决策文件、现有计划、代码、配置或其他文件，也没有提交、推送、运行 GPU/训练/云任务或完整测试。
