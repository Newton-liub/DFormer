# MUSeg 全背景样本与 Depth=0 最短验证路径

- 设计目标：以最少代码、最少训练和最少 GPU 时间，判断两个风险是否真实，以及两个最小修复是否值得继续投入。[RE001][RE003][RE006]
- 路径 A：问题—假设验证；通过受控注入复现负面结果。[RE009]
- 路径 B：方案—假设验证；通过最小 A/B 对照检验 safe loss 与 depth validity mask。[RE001][RE003][RE004]
- 预注册原则：先固定变量、指标和 G/N-G 门槛，再运行实验；路径 A 不改模型权重，路径 B 除目标修复外保持数据、split、checkpoint、seed 与评估代码一致。[RE009]

## 一、共同实验口径

1. 保留官方完整 split，不删除 11 张全背景样本；原始 train/test 为 1595/1576，11 张全背景图中 train 5 张、test 6 张。[RE005][RE006]
2. 原始标签 `0=background`，经 `gt_transform=True` 后为 `255=ignore`；原始 `1–15` 映射为训练 ID `0–14`。[RE001][RE006]
3. 原始 Depth=0 表示无效/缺失深度，不等于最近距离；当前数据中该值约占 30.7351%。[RE005][RE006]
4. 首选 DFormerv2-S 作为 MVE 模型；DFormerv2 直接以 patch 深度差构造几何先验，因此 Depth=0 可能改变注意力，而不只是改变输入通道。[RE003][RE004]
5. 主指标只使用：NaN 触发率、前景 mIoU、每图 mIoU、无效深度区域边界带 mIoU；辅助记录 loss、梯度是否 finite 和 wall-clock，不扩展到完整指标矩阵。[RE001][RE003][RE009]
6. Path A 首先使用一个固定可用 checkpoint 做推理级实验，避免训练随机性；只有 Path B 才进行最小训练/微调。[RE009]

---

# 路径 A：问题—假设验证路径

## A1. 全背景样本风险：受控制造全 ignore 本地 batch

### 核心假设识别

当前 `loss = per_pixel_loss[valid].mean()` 在某个 GPU/rank 的本地 batch 没有任何有效像素时会对空张量求均值并产生 NaN；若 batch 至少有一个有效像素，则这个特定空均值机制不会触发。[RE001][RE002]

### 最小可行实验设计（MVE）

#### 实验目标

直接回答：“将本地 batch 的全部标签人为改为 `255`，能否稳定复现非有限 loss/gradient；放回一个有效像素后是否恢复正常？”[RE001][RE002]

#### 实验步骤

1. 不训练完整模型，构造形状为 `B×15×H×W` 的随机 logits，并使用项目同款 `CrossEntropyLoss(reduction="none", ignore_index=255)`。[RE001][RE002]
2. 固定 `B=1`、小尺寸 `H=W=32`，生成四个标签条件：有效像素比例分别为 `100%`、`1%`、`1 pixel`、`0%`；其余像素全部设为 255。[RE001][RE002]
3. 对每个条件运行项目现有归约表达式，记录 loss 是否 finite；随后调用 `backward()`，记录梯度是否全部 finite。[RE001]
4. 将 `0%` 条件重复 100 次并无必要，因为该运算是确定性的；只需覆盖主头和启用 aux head 时的两条归约路径。[RE001][RE009]
5. 再做一个分布式最小模拟：rank 0 使用 `0%`，rank 1 使用 `1 pixel`，确认风险判据必须按“每个 rank 的本地 batch”而不是全局 batch 判断。[RE001]

#### 所需数据/材料

- 不需要 MUSeg 图像、GPU 或完整 checkpoint；只需要 PyTorch 和 `models/builder.py` 中的现有 loss 表达式。[RE001][RE002]
- 5 张 train 全背景图只用于证明真实采样条件存在，不用于本 MVE 的计算。[RE006]

### 关键成功/失败指标

- **成功（G）**：`0%` 有效像素时 loss 或 gradient 非 finite，而 `1 pixel`、`1%`、`100%` 均 finite；说明“本地 batch 全 ignore”是充分且边界清晰的数值风险，应修复。[RE001][RE002]
- **失败（N-G）**：当前运行环境下 `0%` 仍稳定返回有限零 loss 且梯度 finite；说明实际 loss 实现已规避该风险，应停止为此修改并核对代码版本。[RE001][RE002]
- **异常分支**：非 `0%` 条件也出现 NaN，说明存在空均值之外的问题，不能把故障归因于全背景样本。[RE009]

### 所需最简资源

- CPU、当前 base 环境、数秒级单元测试；无需数据加载、训练或 GPU。[RE001][RE002]

### 备选及简化方案

直接在 Python REPL 中对 `torch.empty(0).mean()` 和项目 per-pixel loss 空筛选各运行一次；它只能证明算子行为，不能覆盖 aux head 和分布式本地 batch 语义。[RE001][RE002]

### 路径原理解释

该实验把“有效像素数量”作为唯一变量，并在 `1 pixel→0 pixel` 的边界处观察故障，因此比启动完整训练更快，也能排除数据增强、优化器和模型结构的干扰。[RE009]

---

## A2. Depth=0 风险：受控注入不同缺失率

### 核心假设识别

如果 DFormerv2 将无效 Depth=0 当作真实深度参与 patch 深度差，那么在固定 RGB、Label、权重和有效深度的条件下，逐步增加人为 Depth=0 应造成单调或剂量相关的性能下降，且下降应集中在被遮挡区域及其边界附近。[RE003][RE004][RE005]

### 最小可行实验设计（MVE）

#### 实验目标

直接回答：“仅增加 Depth=0 的面积，是否会使同一 checkpoint 的前景分割性能系统性下降？”[RE003][RE004]

#### 实验步骤

1. 从官方 test 中固定抽取 64 张原始有效深度比例较高且含前景标签的图；固定 checkpoint、RGB、Label、resize 和推理参数。[RE005][RE006][RE009]
2. 仅在原始有效深度像素上注入额外 0，避免把已有无效点重复计数；设置新增缺失率 `q∈{0, 0.1, 0.3, 0.5}`。[RE005][RE006]
3. 每个 `q` 生成两种固定掩码并保存：`block`（1–3 个矩形，模拟连续传感器缺失）和 `random`（像素级缺失）；同一图、同一 `q` 使用固定 seed。[RE008][RE009]
4. 不重新训练，分别推理原始 Depth 和各受控缺失版本，记录前景 mIoU、每图 mIoU，以及注入区域向外 5 像素边界带的 mIoU。[RE003][RE004]
5. 计算同图配对差值 `ΔmIoU(q)=mIoU(q)-mIoU(0)`，并画出缺失率—性能曲线；核心看趋势和效应量，不先做大规模多 seed 训练。[RE009]
6. 加一个负对照：将同样掩码应用到 Depth，但把被遮挡区域填为该图有效深度中位数；若 0 值的退化明显大于中位数填充，说明风险与“无效 0 被当作真实极近深度”有关，而不只是信息删除。[RE003][RE004][RE009]

#### 所需数据/材料

- MUSeg_DFormer 的 RGB、Depth、Depth16、Label 和官方 test 索引；Depth16 用于定义原始有效 mask。[RE005][RE006][RE007]
- 一个可用 DFormerv2-S checkpoint；路径 A 不要求重新训练。[RE003]
- OpenCV/NumPy/PyTorch 以及一个保存注入 mask 和汇总 CSV 的短脚本。[RE007]

### 关键成功/失败指标

- **成功（G）**：预注册判据同时满足：`q=0.3` 时 block 条件的平均前景 mIoU 相对 `q=0` 下降至少 `2.0` 个百分点；至少 75% 图像的配对差值为负；且 `q=0.1→0.3→0.5` 的平均性能总体不升。[RE003][RE004][RE009]
- **失败（N-G）**：`q=0.5` 仍下降不足 `0.5` 个百分点，或不同 `q` 无方向一致的趋势；说明当前模型/数据上 Depth=0 不是关键瓶颈，不应优先投入复杂 mask 结构。[RE009]
- **中间态**：只在 synthetic block 下明显退化、自然高缺失样本上不退化；说明模型对极端人工缺失敏感，但尚不能证明 MUSeg 自然缺失是实际瓶颈，应先做自然无效率分层分析。[RE005][RE006][RE009]

### 所需最简资源

- 1 张可推理 DFormerv2-S 的 GPU；只跑 `64×4×2=512` 次单尺度前向，不训练。[RE003]
- 若没有 GPU，可降为 16 张图、`q∈{0,0.3,0.5}`、只测 block，共 48 次前向。[RE009]

### 备选及简化方案

不计算完整 mIoU，只比较原始与注入后 logits 的像素翻转率、平均 KL divergence 和注入边界带预测变化；它不能证明任务指标下降，但能快速判断模型是否对 Depth=0 敏感。[RE003][RE004]

### 路径原理解释

该实验不重新训练，只改变同一输入中的 Depth 有效性，并使用同图配对比较，因此最短地建立“Depth=0 剂量增加→预测退化”的因果证据。[RE009]

---

# 路径 B：方案—假设验证路径

## B1. 全背景风险解决方案：safe masked loss

### 核心假设识别

在没有有效像素时返回与 logits 计算图相连的零 loss，可以完全消除 NaN，且在存在有效像素时与原实现数值和梯度一致。[RE001][RE002]

### 最小可行实验设计（MVE）

#### 实验目标

直接回答：“safe masked loss 是否只修复空集合边界，而不改变正常 batch 的优化行为？”[RE001]

#### 实验步骤

1. 实现单一函数：有有效像素时返回 `pixel_loss[valid].mean()`；无有效像素时返回 `pixel_loss.sum()*0.0`。[RE001]
2. 复用 A1 的四个条件，对旧实现和新实现逐项比较 loss、logits gradient 和 finite 状态。[RE001][RE002]
3. 对 `100%`、`1%`、`1 pixel` 条件要求 loss 与梯度在浮点容差内相等；对 `0%` 条件要求新实现 loss=0 且梯度 finite。[RE001]
4. 用一个真实混合 batch（1 张全背景图+1 张普通图）和一个全背景 batch 各做一次 forward/backward；无需跑完整 epoch。[RE006]

#### 所需数据/材料

- A1 单元测试张量；另取 1 张已人工确认为纯背景的 train 图和 1 张普通 train 图。[RE006]
- PyTorch 和 `models/builder.py`。[RE001][RE002]

### 关键成功/失败指标

- **成功（G）**：正常条件下新旧 loss 与梯度最大绝对差 `<1e-6`；全 ignore 条件下新 loss 精确为有限 0、梯度 finite；主头和 aux head 均通过。[RE001]
- **失败（N-G）**：新实现改变任何非空条件的 loss/gradient，或全 ignore 条件仍产生非有限值；停止训练并修正实现。[RE001]

### 所需最简资源

- CPU、数秒级单元测试；无需改变 split、无需训练、无需 GPU。[RE001][RE002]

### 备选及简化方案

在 DataLoader 中拒绝全背景样本只能降低已知风险，不能覆盖随机 crop 或某个分布式 rank 的本地 batch 全 ignore；因此仅可作效率消融，不能代替 safe loss。[RE001][RE006]

### 路径原理解释

该修复没有可学习参数，也不改变任何含有效像素的 batch；只要梯度等价测试通过，就能以最小代价关闭确定性的数值故障。[RE001]

---

## B2. Depth=0 解决方案：无参数 validity mask

### 核心假设识别

仅在两个 patch 都有效时启用 depth decay、无效 pair 退化为 positional decay，可在不伪造深度、不丢弃 RGB/Label 监督的前提下减轻 Depth=0 对几何先验的污染。[RE003][RE004][RE005]

### 最小可行实验设计（MVE）

#### 实验目标

直接回答：“最小的 depth-pair gating 是否比原始 DFormerv2 更耐受自然和人为 Depth=0，同时不明显损害干净样本？”[RE003][RE004]

#### 实验步骤

1. 从量化前 `Depth16>0` 生成 validity mask，并与 RGB/Depth 同步做几何增强；mask resize 使用 nearest，避免产生非二值有效性。[RE005][RE007]
2. 在每个 stage 将 mask resize 到 depth patch 网格；构造 `pair_valid=valid_i & valid_j`，只对有效 pair 保留 depth decay，无效 pair 只保留现有 positional decay。[RE003][RE004]
3. 不增加新参数，不改 decoder，不把对应 Label 设为 ignore，不做深度填补。[RE003][RE004][RE005]
4. 第一阶段做零训练 A/B：同一 checkpoint 分别运行原实现 B0 与 gating B2，先在 A2 的 64 张、相同注入 mask 上比较；若 B2 没有方向性收益，立即停止。[RE009]
5. 只有零训练测试有正向信号时，进行最小微调：B0/B2 使用同一初始化、同一固定 20% group-disjoint train 子集、同一 5 epoch、同一 seed；评估完整官方 test 和 A2 corruption sweep。[RE005][RE009]
6. 同时报告自然无效深度比例高/低两组的 mIoU，避免改进只来自人工 corruption。[RE005][RE006]

#### 所需数据/材料

- A2 的固定 64 张测试样本与注入 mask；若进入微调，再使用固定 20% group-disjoint train 子集。[RE005][RE009]
- DFormerv2-S、现有 checkpoint、Depth16 有效 mask 和当前训练代码。[RE003][RE004][RE007]

### 关键成功/失败指标

- **零训练筛选成功（G0）**：在 `q=0.3 block` 上，B2 相对 B0 恢复至少 `1.0` 个 mIoU 百分点，同时 `q=0` 下降不超过 `0.5` 个百分点；满足后才进入 5 epoch 微调。[RE009]
- **最终成功（G）**：微调后自然高无效率组前景 mIoU 提升至少 `1.0` 个百分点，或 `q=0.3 block` 提升至少 `2.0` 个百分点；同时完整 test 前景 mIoU 不下降超过 `0.5` 个百分点。[RE009]
- **失败（N-G）**：B2 在自然高无效率组和 corruption sweep 上都提升不足 `0.5` 个百分点，或完整 test 下降超过 `0.5` 个百分点；停止复杂 mask 开发，保留 B0 并转向简单填补对照。[RE008][RE009]
- **解释限制**：若只在人工 corruption 上成功，则只能证明鲁棒性可行，不能宣称已解决 MUSeg 自然 Depth=0 瓶颈。[RE005][RE009]

### 所需最简资源

- 第一阶段仅需 1 个已有 checkpoint 和 512 次前向；第二阶段只在 G0 后使用 1 张 GPU、两个 5-epoch 小子集微调。[RE003][RE009]
- 代码改动仅限 validity mask 数据流与 geometry prior gating，不引入深度补全网络。[RE004][RE008]

### 备选及简化方案

- **低配版**：先在输入层用最近有效邻域或 RGB 引导方法填补 Depth=0，再保持模型不变；优点是代码简单，缺点是会生成未经传感器观测的伪深度。[RE008]
- **巧妙版**：不训练模型，只将无效 pair 的 depth difference 强制为 0 并比较 logits 稳定性；若预测变化极小，可提前终止 mask 路线。[RE003][RE004][RE009]

### 路径原理解释

该方案直接切断 DFormerv2 中“无效 0→错误深度差→错误 geometry prior”的最短链路，不引入补全网络和新参数，因此是检验 mask 思路本身的最小 A/B。[RE003][RE004]

---

# 两条路径如何互补及行动路径

路径 A 先回答“风险是否真实”：A1 验证全 ignore 是否构成确定性数值故障，A2 验证 Depth=0 是否形成具有剂量关系的性能瓶颈；路径 B 再回答“最小修复是否有效”：B1 要求正常梯度严格等价，B2 要求自然/人工缺失场景有收益且干净性能基本不损失。[RE001][RE003][RE009]

- **A1-G / B1-G**：立即合入 safe masked loss；这是低成本稳定性修复，不需要把它包装为模型创新。[RE001]
- **A1-G / B1-N-G**：风险真实但实现错误，停止训练，先修正归约和 aux 路径。[RE001]
- **A1-N-G**：不再投入全背景 loss 修复，只保留回归测试并核对当前代码版本。[RE001][RE002]
- **A2-G / B2-G**：Depth=0 是真实瓶颈且 validity mask 初步有效，才值得进入完整 3-seed、完整 train 和 mask 设计消融。[RE003][RE004][RE009]
- **A2-G / B2-N-G**：问题真实但 gating 方案不足，下一步只比较一种简单填补方案，不立即扩展复杂网络。[RE008][RE009]
- **A2-N-G / B2-N-G**：终止 Depth=0 主线，把资源投入 MUSeg 配置、训练复现或更强瓶颈。[RE009]
- **A2-N-G / B2-G**：可能是正则化或评估偶然性，不宣称因果成立；增加自然无效率分层和第二个 seed 后再判断。[RE005][RE009]

## 参考文献与项目证据

- **RE001**：DFormer 当前实现，`models/builder.py`，`EncoderDecoder.forward()`；使用 `criterion(...)[label != background].mean()`，访问日期 2026-08-19。
- **RE002**：PyTorch 2.13 Documentation, `torch.nn.CrossEntropyLoss`；`ignore_index` 不贡献梯度，mean 按非 ignore 目标归约。https://docs.pytorch.org/docs/2.13/generated/torch.nn.CrossEntropyLoss.html
- **RE003**：Yin, B. et al. *DFormerv2: Geometry Self-Attention for RGBD Semantic Segmentation*. arXiv:2504.04701, 2025. https://arxiv.org/abs/2504.04701
- **RE004**：DFormer 当前实现，`models/encoders/DFormerv2.py`，`generate_depth_decay()` 与 `forward()`；深度先插值到 patch 网格，再以绝对差参与 geometry prior，访问日期 2026-08-19。
- **RE005**：Li, et al. *MUSeg: A multimodal semantic segmentation dataset for complex underground mine scenes*. Scientific Data, 2025. https://www.nature.com/articles/s41597-025-05493-9
- **RE006**：项目全量审计，`doc/museg-dformer-data-processing-review.md`；3171 样本、Depth=0 比例 30.7351%、11 张全背景图及 train/test 分布，修订日期 2026-08-19。
- **RE007**：项目转换实现，`tools/prepare_museg.py` 与生成的 `dataset_meta.json`；固定 `max_raw=13932`、保留 invalid 0、官方 split 和全量验证，验证日期 2026-08-19。
- **RE008**：Zhang, Y. and Funkhouser, T. *Deep Depth Completion of a Single RGB-D Image*. CVPR 2018. https://github.com/yindaz/DeepCompletionRelease
- **RE009**：Montgomery, D. C. *Design and Analysis of Experiments*. Wiley；受控变量、配对对照、预先规定决策规则等实验设计原则。