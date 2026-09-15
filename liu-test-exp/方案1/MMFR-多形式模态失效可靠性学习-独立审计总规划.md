# MMFR-多形式模态失效可靠性学习：独立审计总规划

> **文档状态：** 2026-09-15 v3 当前审计版。高级审计已正式选择 **A + MID-A**；当前身份为 `MMFR-A2-train-integration-v3`，分析身份为 `MMFR-A2-clean-control-v3` 与 `MMFR-A2-depth-corruption-train-v3`。v3 要求的 severity/burden、CPU qualification、validity transport、initial-state equivalence、AMP update-path isolation 五项本地资格均已形成独立结构化 `PASS` 证据，状态已升级为 `eligible-for-cloud-capacity-probe-authorization`。
> **授权边界：** 上述状态只表示现在可以向用户申请云端单 GPU、batch size `10` 的容量/吞吐短 probe；它不表示 probe 已获授权。云实例创建、正式 500 epoch 训练、checkpoint save/load 验收、完整 evaluator、MMFR 效果评价和 official test 均未运行、未授权。
> **身份冻结：** v2 以 commit `3c8ebddeb76e63a5be37e261381117abfba117ca` 和 annotated tag `MMFR-A2-v2-pretrain-freeze` 保留为历史；v3 以 commit `4ce7b67c5707991750461ef2f806e12b37e203f9` 和 annotated tag `MMFR-A2-v3-pretrain-freeze` 冻结。两个标签均仅在本地，未推送远端。
> **大白话说明：** v2 审计发现的“普通噪声或模糊会把原生空洞变成假深度”和“错位后有效性坐标不明确”已经在独立 v3 中修正，并通过五道本地门禁。下一步只能先询问是否允许租用云端单卡做短容量/速度测试，不能直接开始正式训练。
> **事实入口：** `doc/main/MUSeg-current-status.md` 是唯一实时状态入口；本文负责重组自包含审计叙事，不替代实时状态。

## 0. 文档用途、形成时点与阅读规则

- **用途：** 本文件供无法直接看到仓库、运行产物和云资源的外部审计者使用。它集中说明项目背景、历史依据、当前 v3 设计、资格证据、正式实验计划、风险、授权边界和恢复点。
- **形成时点：** 2026-09-15，v3 代码、配置、协议模板、五个资格工具及五份 canonical 报告已经冻结；正式 MMFR 训练和效果评价尚未开始。
- **证据优先级：**
  1. 当前工作区中可直接核验的代码、配置、协议模板、运行产物与哈希；
  2. `doc/main/MUSeg-current-status.md` 的实时事实与授权边界；
  3. `doc/main/MUSeg-open-decisions.md` 第 13 节的研究裁决；
  4. `liu-test-exp/方案1/改动说明.md` 的高级审计原文与 v3 执行回填；
  5. 本文及其他计划文档对未来实验的说明。
- **状态词规则：** “已实现”不等于“已训练”；“资格通过”不等于“模型效果通过”；“可申请授权”不等于“已授权”；“冻结”表示结果产生前不得静默改变；“历史”表示只用于追溯，不再是当前执行身份。
- **硬边界：** 本文没有任何 v3 checkpoint、mIoU、可靠性校准、真实故障收益或部署收益结论。所有资格报告均记录 `official_test_included=false`。
- **版本关系：** v1、v2、v3 是互不覆盖的协议身份。v1/v2 文件和历史证据保留；v3 通过新文件、新配置、新协议和新标签承载语义变化。

## 1. 一页式执行摘要

### 1.1 已知事实

- 本项目使用 DFormerv2-S 在 MUSeg 矿井场景数据上做 RGB-D 语义分割。DFormerv2-S 是 **RGB-primary semantic path + Depth geometry-prior path** 的非对称架构：RGB 承担主要语义路径，Depth 主要形成几何先验，不是与 RGB 对称的第二条完整语义分支。
- 开发职责固定为 `train-dev=1277`、`val-dev=318`；official test 共 `1576` 条，保持 `sealed_unread`。`val-dev` 已参与 checkpoint 选择，因此只能提供开发证据，不能称为独立测试。
- 稳定 Quick-B0 为 single-seed RGB development baseline：epoch 420 checkpoint、五尺度水平翻转 10-view evaluator，mIoU `58.79`、mAcc `69.91`、mF1 `72.73`。
- DVC-A1 的 Depth 置零开发验证最终为 `not-supported`；DVG-B1 在已知真实坏区的 Oracle GSA gate 下使 Boundary IoU 与 mIoU 下降，最终为 `oracle-not-supported`。这些历史结果说明固定回退动作不值得继续，不等于所有可靠性学习都失败。
- v2 解决了 severity 二次编码、空间失效绝对像素尺度、原生无效 Depth 未进入 target、RGB 恒 1 通道被计分等第一轮问题；随后全量有效性与 transport 审计发现，v2 的 `gaussian_noise`、`blur`、`misalignment` 可能在原生无效位置产生非零 Depth，且 misalignment 的有效性坐标语义未冻结。
- 高级审计依据全量 `train-dev` 分布裁决：natural-invalid 监督保留，`lambda_rel=0.1` 不变，不做重加权或 masking，不删除 8 个训练几何下全无有效 Depth 的 crop；43 个不在 optimizer param group 的参数保持上游语义，本轮不修。
- 高级审计对 validity semantics 选择 **A + MID-A**：原生无效与后续结构性无效是吸收状态；intensity corruption 不能复活 invalid Depth；blur 使用 mask-normalized blur；misalignment 用同一整数平移搬运 Depth 和 validity；最终 target 使用最终状态。
- v3 五项本地资格均已通过，当前可以申请云端容量/吞吐短 probe 授权；probe、云资源和训练本身仍未授权。

### 1.2 大白话版方案

未来要公平地训练两个模型：clean control 始终看干净输入，Depth-corruption model 按冻结随机规则损坏 Depth，并额外学习像素级 Depth 可靠性。两者从同一官方预训练、同一 seed、同一数据和同一训练预算独立开始。A2 的可靠性预测仍只承担辅助监督，不进入 backbone、decoder 或 geometry prior，所以 A2 的分割差异只能直接归因于 corruption exposure，不能归功于可靠性头。

v3 新增的关键保护是：一个位置如果当前没有有效 Depth，普通噪声、模糊或量化就不能凭空生成“新测量”；如果 Depth 被平移，表示“这里有无测量”的 validity 也一起平移。这样实际模型输入、有效性状态和监督 target 使用同一套坐标与语义。

### 1.3 v2→v3 带来什么、没带来什么

- **带来：** 新的 A1/A2 v3 身份；显式 sequential validity state；A + MID-A；mask-normalized blur；最终 target `V_state_final * R_syn`；四类互斥 telemetry；v3 协议模板；五个可复跑资格工具和五份 `PASS` 报告；新的本地冻结 commit/tag。
- **保持：** 六类 failure、severity、curriculum、`p_clean=0.25`、`max_specs=2`、`lambda_rel=0.1`、Depth-only 监督、500 epoch、optimizer、学习率、batch size、selector、主 evaluator、成功门槛、8 个全无有效 Depth crop、43 个 optimizer-missing 参数。
- **没有带来：** 没有云实例、没有容量/吞吐 probe、没有正式训练、没有 checkpoint、没有 MMFR mIoU、没有 R1/S1/C1 结果、没有 official test。

## 2. 项目背景与数据职责

### 2.1 模型和任务

- **DFormer / DFormerv2-S：** 面向 RGB-D 语义分割的模型；当前 Small 配置以 RGB 为主要语义路径，Depth 通过 geometry prior 参与 Geometry-aware Self-Attention（GSA，几何感知自注意力）。
- **MUSeg：** 矿井场景 RGB/Depth/Label 数据。它适合受控合成失效的开发研究，但没有自然故障类型、位置、严重度、故障率、重复采集或标定漂移真值。
- **reliability target：** 由合成器和显式有效性状态推导的训练 surrogate，不是自然传感器健康真值。

### 2.2 数据划分

- `train-dev=1277`：A2 两个公平身份的训练职责。
- `val-dev=318`：clean selector 和冻结 checkpoint 后的开发评价职责；包含 `196` 个 location group，用于配对 bootstrap。
- official test=`1576`、`sealed_unread`：不得用于训练、选择、调参、方向筛选或当前开发评价。
- evaluation condition 必须在原始对齐 RGB/Depth 上按冻结 seed 一次生成，再由全部 evaluator view 共用，不能每个 view 重抽 corruption。

## 3. 稳定 Quick-B0 基线

### 3.1 身份

- 模型：`DFormerv2-S RGB Quick-B0`。
- RGB 输入契约：`rgb-imagenet-rgb-order-v1`。
- 官方预训练：`DFormerv2_Small_pretrained.pth`，SHA-256 `19116988fc86dc9f3e879282237941e11b9b1b5c480edb51e92807311dbc11a6`。
- 最终 checkpoint：epoch 420 `selector-epoch-420.pth`，SHA-256 `f246a3afc50334c81302b7bfebdadf7cf37d00326bf1c3aa54f6a151754e3a1c`。
- `val-dev` split SHA-256：`1d0719d8f64f016d48995c25ab66d4004d76b7155d9efeef7cbb7454c0dd0e83`。
- official test：`official_test_included=false`。

### 3.2 evaluator 与指标

- 主 evaluator：`msflip-whole-original-grid-v1`。
- 五个尺度 `0.5、0.75、1.0、1.25、1.5`，每尺度原图与水平翻转，共 10 view。
- 每个 view 的 logits 恢复到原始 Label 网格，在 FP32 中平均 pre-softmax logits 后计分。
- epoch 420：mIoU `58.79`、mAcc `69.91`、mF1 `72.73`。
- 该基线是 single-seed development reference，不是三 seed 论文复现，也不是 official-test 结果。

## 4. 关键概念与判断边界

- **MMFR（Multi-Modal Failure Robustness，多形式模态失效鲁棒性）：** 当前研究方向，覆盖整模态缺失、局部缺失、噪声、模糊、量化、几何错位和组合失效。
- **synthetic corruption（合成损坏）：** 模型输入或表示层的受控操作，不是 Kinect 等设备的真实物理噪声模型。
- **validity state（有效性状态）：** 二值状态 `V_state`，表示当前位置当前是否存在合法 Depth 测量。大白话说，它回答“这里有没有测量”，不回答“测量质量有多好”。
- **synthetic reliability `R_syn`：** 合成器根据 realized damage 与 burden 形成的连续质量目标。大白话说，它回答“现有测量还剩多少质量”。
- **A + MID-A：** A 表示 intensity corruption 只能在当前有效位置作用，invalid 状态不可被复活；MID-A 表示 misalignment 同时搬运 Depth 和 validity。
- **mask-normalized blur：** 只以有效测量参与 Gaussian blur 的归一化卷积，避免无效零值污染邻近有效测量。
- **reliability head：** 由固定信号特征预测 RGB/Depth reliability logits 的小网络。A2 只监督 Depth 通道，RGB 是未计分 all-ones scaffold；预测不反馈分割路径。
- **qualification（资格检查）：** 验证接口、语义、确定性、初始公平性或限步训练轨迹。资格通过不提供模型效果结论。
- **capacity/throughput probe：** 云端短运行，只回答冻结 batch size 是否容纳、速度与单位样本成本；不等于正式训练。
- **single-seed development-supported：** 即使未来主门槛达到，也只支持单 seed、开发划分上的结论。

## 5. 历史依据：DVC、DVG 与 MMFR

### 5.1 DVC-A1

- v1 因 `58/196` 个 location group 无法构造非空 `boundary-q75`，超过预注册上限，以 `protocol-blocked` 结束。
- v2 固定 218 张图、138 组，但完整统计只有 `137/138` 个有效配对组，再次 `protocol-blocked`。
- v3 只修复 Boundary IoU 的 background/ignore 标签域；主 `dose_effect=+0.0731348717` 个百分点，95% 区间 `[-0.0620441424,+0.2226503089]`，最终为 `not-supported`。

### 5.2 DVG-B1

- 已知真实坏区后，仅门控 GSA depth contribution；A/B/C 在结果前冻结。
- P4 包含 218 张图、138 个组、五个 condition、每图 10 view，共 `10,900` 个 baseline/gated 配对 forward pair。
- `boundary-q75` 的 Oracle-minus-baseline Boundary IoU 为 `-0.1508352015` 个百分点，mIoU 为 `-0.2316731726` 个百分点，最终为 `oracle-not-supported`。
- 这说明 corruption mask 不能直接等同于模型动作；它不否定后续学习型 reliability-conditioned adaptation。

### 5.3 对当前 A2 的含义

1. A2 先隔离 corruption exposure、target 与辅助 estimation 是否正确接入。
2. A2 不让 predicted reliability 影响模型动作；B1 才是第一项 reliability-aware segmentation 研究。
3. RGB complete-missing 需要未来独立 Depth 语义路径或双语义架构，不能由当前 RGB-centric A2 静默宣称解决。

## 6. `train-dev` Depth 有效性事实与高级审计裁决

### 6.1 全量事实

- 全部 `1277` 条原始对齐网格的 `depth_valid_pre` 比例：mean `0.6833`、median `0.7413`、std `0.2482`、min `0.0125`、max `0.9999`。
- 原始网格有效率 `<1% / <5% / <10% / <25% / <50%` 的样本数为 `0 / 13 / 26 / 91 / 309`；没有任何样本 Depth 全为 0。
- 训练 crop/pad 几何下：mean `0.6978`、median `0.8177`；有 `8` 条 crop 在 `valid_mask` 内完全没有有效 Depth。
- early/mid/late 三档 target 恰为 0 的监督像素占 `28.33% / 28.41% / 43.84%`，其中约 `25.82` 个百分点始终来自 natural-invalid。
- 历史单样本约 `98.5%` 为 0 的事实只适用于该样本，不能外推到数据集。

### 6.2 已冻结裁决

- natural-invalid 保留在 Depth reliability BCE 中；不重加权、不 masking、不删除 8 个 crop。
- `lambda_rel=0.1` 不变。
- 43 个 optimizer-missing 参数保持 Quick-B0 与上游 DFormerv2 的既有 optimizer 语义；若未来研究 optimizer completeness，必须新建身份并同时重训 clean 与 MMFR。
- A/B 与 MID-A/MID-B 已正式收口为 **A + MID-A**，不再是当前开放项。

## 7. 当前 MMFR 阶段关系

```text
A1 v3：Depth 合成失效 + sequential validity state + continuous reliability surrogate
    ├─ 六类冻结 corruption
    ├─ A：intensity corruption 不复活 invalid Depth
    ├─ MID-A：Depth 与 validity 同步平移
    ├─ mask-normalized blur
    └─ 审计 metadata 与确定性 RNG
             │
             ▼
A2 v3：Depth corruption robustness training + parallel reliability estimation
    ├─ post-mirror/scale/crop/pad、pre-GPU corruption
    ├─ segmentation loss + 0.1 × Depth-only reliability BCE
    ├─ target = V_state_final × R_syn
    ├─ predicted reliability 不进入 backbone/decoder/geometry prior
    └─ 五项本地资格全部 PASS
             │
             ▼
B1：首次让 reliability 影响模型动作（待独立设计、协议和授权）
             │
             ▼
B2：覆盖 RGB 完全失效的 Depth 语义 fallback 或双语义路径（待设计）
```

## 8. A1 v3 失效基函数与 sequential validity state

### 8.1 身份和不变量

- 当前 A1：`MMFR-A1-corruption-basis-v3`，实现于 `utils/dataloader/multimodal_failure_v3.py`。
- v1/v2 模块保持不改；v3 不通过原地改写历史文件实现。
- 六类 failure 保持：`entire_missing`、`spatial_dropout`、`gaussian_noise`、`blur`、`quantization`、`misalignment`。
- `quantization` 与 `misalignment` 为 Depth-only；当前 A2 config 只抽 Depth。
- severity 范围、三阶段 curriculum、`max_specs=2` 与单次 severity 编码沿用 v2。

### 8.2 burden 和 severity

多种失效按顺序作用，synthetic reliability 为：

$$
R_m^{syn}(p)=\exp\left(-\sum_k b_{m,k}(p)\right).
$$

- `entire_missing` 与 `spatial_dropout` 使用 `MISSING_BURDEN=1.0e4`。
- graded intensity corruption 的 burden 为 realized normalized damage，`DAMAGE_REFERENCE=0.25`，不再二次乘 severity。
- `gaussian_noise` 的 `sigma` 上限为 48 个 uint8 强度单位。
- `blur` 的 sigma 上限为实际网格短边的 `1/80`。
- `misalignment` 每轴最大位移为该轴尺寸的 `1/30`。
- `quantization` 的 levels 由 severity 从 256 降至最少 2。

### 8.3 validity 状态转移

初始状态为：

$$
V_D^{state,0}=V_D^{pre}=(D_{pre}>0)\land V_{geometry}.
$$

按 spec 顺序更新：

- `gaussian_noise`、`blur`、`quantization`：`V_state` 不变，只在 `V_state=1` 处更新 Depth；无效处保持 uint8 0。
- `spatial_dropout`：`V_state <- V_state AND keep_mask`。
- `entire_missing`：`V_state <- 0`。
- `misalignment`：Depth 与 `V_state` 使用同一整数 `(dy, dx)` 平移，再与 `valid_mask` 相交。
- 最终强制：`V_state_final=1` 的 Depth uint8 必须在 `[1,255]`，`V_state_final=0` 的 Depth 必须为 0。

### 8.4 mask-normalized blur

普通 Gaussian blur 会让无效零值参与卷积并压低邻近有效测量。v3 改为：

$$
D_{blur}(p)=\frac{G*(D\cdot V)(p)}{G*V(p)+\epsilon},
$$

只在 `V_state=1` 处输出，`V_state=0` 继续保持 0。该处理只解决有效性语义，不声称模拟真实传感器 blur。

## 9. A2 v3 数据流、target、RNG 与 telemetry

### 9.1 接入位置和输入语义

- corruption 在 DataLoader 完成 mirror/scale/crop/pad 后、batch 搬到 GPU 前，由主训练进程逐样本执行。
- segmentation backbone 继续消费 normalized RGB/Depth。
- reliability estimator 消费恢复后的 raw `[0,1]` RGB/Depth 与 `depth_valid_post`。
- clean sample 直接复用原 normalized tensor，保持 exact no-op。
- crop/pad 区 normalized/raw 都为精确 0，target 为中性 1，并由 `valid_mask` 排除。

### 9.2 最终监督目标

v3 Depth target 为：

$$
R_D^{sup,v3}(p)=V_D^{state,K}(p)\,R_D^{syn}(p).
$$

硬一致性条件为：

$$
V_D^{state,K}=depth\_valid\_post=(raw\_depth\_post>0)\land valid\_mask.
$$

大白话说，最终有没有测量由最后的 validity state 决定，剩余质量由 `R_syn` 决定，两者相乘得到监督目标。

### 9.3 四类 telemetry

训练几何有效区被四个互斥 mask 精确划分：

- `natural-invalid`：原始 Depth 无效；
- `synthetic-invalid`：原始有效但被结构性 corruption 变为无效；
- `implicit-quality`：前后均有效但数值被 corruption 改变；
- `valid-clean`：前后均有效且数值未改变。

每类单独记录像素数、占比和 Depth BCE；空类输出 detached NaN 占位并以 `pixel_count=0` 排除聚合。telemetry 不改变 aggregate reliability loss 或优化权重。

### 9.4 确定性 RNG

- corruption seed：`2026091402`；模型 seed：`772961337`。
- 每样本使用 `numpy.PCG64(SeedSequence(words))`。
- words 顺序为 `train_seed`、`epoch_1_based`、`iteration_0_based`、`global_rank`、`sample_slot_0_based`、sample-id SHA-256 前四个大端 uint32。
- 训练进度按 epoch/iteration 位置计算，不使用 wall clock。

## 10. 模型、固定信号特征与损失

### 10.1 reliability head

- `SignalFeatureExtractor` 输出 14 通道固定信号：RGB 六类、Depth 六类、Depth validity 一类、跨模态 gradient alignment 一类。
- head 为 `14→16→16→2` 的卷积网络，新增 `4,386` 个参数。
- reliability 分支固定 FP32；segmentation backbone/decoder 继续使用 AMP。
- A2 的 reliability prediction 不进入 backbone、decoder、geometry prior 或 checkpoint selector。

### 10.2 损失

每样本只做一次 segmentation forward：

$$
\mathcal L=\mathcal L_{seg}^{input}+0.1\,\mathcal L_{rel,depth}.
$$

- `lambda_consistency=0`；无 clean teacher、无 clean-corrupt consistency、无 distillation、无 geometry adapter。
- reliability loss 只监督 Depth 通道；RGB 通道保持全 1 scaffold，不得报告为已训练的 RGB reliability estimator。
- supervision 字段缺失、通道漂移、监督 mask 为空或非有限 loss 均 fail-closed。

## 11. 两个公平训练身份（v3）

### 11.1 共同冻结项

`MMFR-A2-clean-control-v3` 与 `MMFR-A2-depth-corruption-train-v3` 共同使用：

- 同一官方 pretrained 与 SHA-256；
- 模型 seed `772961337`；
- `train-dev=1277`、`val-dev=318`；
- RGB contract `rgb-imagenet-rgb-order-v1`；
- AdamW、learning rate `6e-5`、weight decay `0.01`；
- 500 epoch、warmup 10 epoch、poly power `0.9`；
- batch size `10`、workers `8`；
- train scales `[0.5,0.75,1.0,1.25,1.5,1.75]`；
- 同一 clean selector、tie-break、主 evaluator、checkpoint 与证据规则；
- inherited upstream optimizer semantics，包括已知 43 个 optimizer-missing 参数。

### 11.2 唯一科学变量

- clean control：clean 输入，不实例化 reliability head，不执行 corruption。
- Depth-corruption model：冻结的 v3 Depth corruption、reliability head 与 `0.1 × Depth-only BCE`。
- 两者必须从官方 pretrained 独立训练；不得从 Quick-B0 或 v1/v2 checkpoint 续训。
- failure condition 不参与 selector；不得看到结果后改变候选、tie-break 或选择规则。

### 11.3 当前代码身份

完整 raw/LF 双哈希以 `protocols/mmfr-a2-train-integration-v3.template.json` 为权威索引。主要 raw SHA-256：

- `utils/dataloader/multimodal_failure_v3.py`：`0a729bcb1b60a71120445894d71459895d2fd61a6bc30dfbccb3ac94108b110f`
- `utils/dataloader/mmfr_training_v3.py`：`449aa42585bf2b170cfc8924c8fefc1f114ddc426be601fd1a78d2d997402ee7`
- `DFormerv2_S_MMFR_A2_Common_v3.py`：`3f309effc1a8e005844d8885bfd31cdb05c816e686632aa0d34f0f23bcffa595`
- `DFormerv2_S_MMFR_A2_Clean_v3.py`：`9bdf78b464589991e358e52044f045543e707d52ea473172fb96219bc561fdf6`
- `DFormerv2_S_MMFR_A2_DepthCorrupt_v3.py`：`b8ba2dd0c5ac4a3c445779bc44ce1d4e9a283fe309a8e190c8c84b5da2c02bef`
- `models/builder.py`：`972d855b906fcea19534b163d8a0299bf2bf236d50a42e2c28fe750fda2166f1`
- `utils/train.py`：`aba1b407d6b53af623145aebeae28cfab0803cbe8d1c2e96d5a3e1a7ea036ad7`

## 12. 云端执行、预算与生命周期

### 12.1 冻结执行配置

- 正式环境：云端单 GPU。
- 默认 RTX 4090 24GB，价格快照 `1.88 元/小时`；备用 RTX 5090 32GB，`2.78 元/小时`。
- 5090 的单位样本成本 break-even 吞吐比为：

$$
\frac{2.78}{1.88}=1.4787234043\approx1.4787.
$$

- 只有 4090 缺货、24GB 无法容纳冻结 batch，或配对 probe 证明 5090 吞吐超过 `1.4787×` 时才使用 5090。
- 换卡不得改变 batch size、学习率、epoch、seed 或其他科学超参数。

### 12.2 当前授权边界

- **可以做：** 向用户申请云端单 GPU、batch size 10 容量/吞吐短 probe 的授权。
- **尚未授权：** 创建云实例、运行 probe、正式训练、checkpoint save/load 验收、完整 evaluator、R1/S1/C1、official test。
- **本机角色：** 推理、想法初步验证和小规模 preflight，不承担正式长训练。

### 12.3 生命周期要求

未来任何付费运行必须在启动前确认实例、最长时长和预计费用，设置并复核控制面最晚停止 schedule；成功、失败或中止后都先取回必要证据并核验哈希，再调用控制面 stop，最后确认平台状态为 `Stopped`。验收 pass/fail 不决定是否停止计费。

## 13. 资格证据及边界

### 13.1 v1/v2 历史证据

- v1 CPU qualification 与本地 GPU 单步 preflight 保持历史：batch size 1，6 次尝试、5 次 GradScaler skip、1 次 optimizer update。它不覆盖 v2/v3。
- v2 severity、CPU、gradient-path、initial-state、AMP update-path 和 validity transport 证据全部保留。v2 validity transport 报告是建立 v3 的问题证据，不得改写为 v3 的通过证据。
- v2 终态：`superseded-before-formal-training-after-validity-semantic-review`；正式训练前被 v3 取代。

### 13.2 v3 五项 canonical 资格

1. **severity/burden：PASS**
   - 工具：`tools/mmfr/a2_v3_severity_burden_audit.py`。
   - 覆盖 2 个分辨率 fixture、6 类、9 个 severity、108 条记录。
   - 检查单次编码、burden/target 单调性、相对尺度、invalid-state 一致性和空 spec strict no-op。
   - 报告：`outputs/mmfr-a1-v3-severity-audit/severity-burden-audit.json`。
   - raw/LF SHA-256：`34d9d70d1e1b9c7420fc7b293f2c2b3eb880caeb7fb8d4f8f6935078bdc8bbe0` / `75ac1f73b42f7430c9d671de7b44894f2c7f5141080ccfaff008bb8f3a27d2da`。

2. **CPU qualification：PASS**
   - `121/121` 断言，`failed=0`；未读 checkpoint、未用 GPU。
   - 覆盖 clean no-op、state transition、target、pad、Depth-only、determinism、telemetry partition 和 fail-closed 守卫。
   - 报告：`outputs/mmfr-a2-v3-cpu-qualification/a2-v3-cpu-qualification.json`。
   - raw/LF SHA-256：`b8740689adccb84249b35c7a98fe936dd229694efa6af071bf31582f085f934a` / `dd24acfdda7e193194bf03f319d195cfa388ad87bc81a05cf22d4073a206f4a7`。

3. **validity transport：PASS**
   - `48/48` 断言，`failed=0`；CPU。
   - `gaussian_noise`、`blur`、`quantization` 的 `newly_valid_pixels=0`。
   - misalignment 验证 MID-A transported validity 与实际 transported Depth 输入一致。
   - 报告：`outputs/mmfr-a2-v3-validity-transport/validity-transport-audit.json`。
   - raw/LF SHA-256：`9f3066d47e82a2ad84e2dad457a6b68c0d5dac4d28882249fdaca7495eb4cd90` / `495e27fa0d2059207e0ae06c60fd3cab048498f9926eff311aa96eb420de0510`。

4. **initial-state equivalence：PASS**
   - 6 次独立构建，每个身份 3 次；`38/38` 断言。
   - `714/714` 个共有参数与 `88/88` 个共有 buffer 逐位相同，`max_abs_difference=0.0`；`extra_norms` 可复现。
   - 报告：`outputs/mmfr-a2-v3-initial-state-equivalence/a2-v3-initial-state-equivalence.json`。
   - raw/LF SHA-256：`4035612e5e314a9044626503c80c2af869a6be1a5cf4d89aab8383b7cb0515cc` / `362b1951d82ff2fb131bf808858926e2ea0e0c00c90a43332ab0acd4d6f4532b`。

5. **AMP update-path isolation：PASS**
   - 本地 NVIDIA GeForce RTX 5060 Laptop GPU；batch size `10`；真实 `train-dev`；AMP + GradScaler。
   - `17` 次尝试、`10` 次成功更新、`17` 个 corrupt step。
   - 两条轨迹的 batch identity、step/skip、GradScaler scale、每次成功更新后的共享参数与共享 optimizer state 均逐位一致；mismatch 均为 0，`max_abs_difference=0.0`。
   - 只有 `lambda_rel=0.1` 侧的 reliability head 学习。
   - 峰值 allocated `3,132,888,576` bytes；耗时 `2112.328` 秒。
   - 报告：`outputs/mmfr-a2-v3-amp-update-path-isolation/a2-v3-amp-update-path-isolation.json`。
   - raw/LF SHA-256：`770daaeab50da4e593b0a722ef0b7482b4c67d8616a37d95348b6a4432036872` / `ed579c1af7daddfffdb59881d5e7a50977f0b0178c6366ed139fbdc72c1c95c3`。

### 13.3 资格证据不能证明什么

- 不覆盖云端 RTX 4090/5090 的容量与吞吐。
- 不覆盖完整 epoch、500 epoch、checkpoint save/load、完整 evaluator、DDP 或 `torch.compile`。
- 不提供 mIoU、校准、真实故障、跨设备或部署收益。
- 不授权任何云资源或训练。

## 14. 正式开发评价设计（未来训练和评价分别获批后）

### 14.1 selector

每个身份独立使用 clean `val-dev`、`original-full`、scale `1.0`、无 flip、mIoU、`earlier_epoch` tie-break。failure condition 不参与 selector；冻结 checkpoint 后才能进入主 evaluator。

### 14.2 主 evaluator 和条件

- evaluator：`msflip-whole-original-grid-v1`；318 张 `val-dev`；每图 10 view。
- evaluation seed：`2026091401`。
- 六个单失效：`spatial_dropout@0.75`、`gaussian_noise@0.75`、`blur@0.75`、`quantization@0.75`、`misalignment@0.75`、`entire_missing@1.0`。
- 三个混合：`spatial_dropout@0.5 + gaussian_noise@0.5`、`blur@0.5 + misalignment@0.5`、`quantization@0.5 + misalignment@0.5`。
- 配对 bootstrap 单位：196 个 location group。

### 14.3 主成功门槛

比较 `Depth-corruption model minus clean control`：

- 六单失效 mIoU 等权宏平均至少 `+1.00` 个百分点；
- paired 95% percentile interval 下界严格大于 0；
- clean mIoU 下降不超过 `0.50` 个百分点；
- 至少五个单条件 mIoU 不为负，且任一不得低于 `-1.00` 个百分点。

mAcc、mF1、Boundary IoU、混合条件和 R1/S1 均为辅助结果，不得替代主门槛。达到门槛时最高声明仍是 `single-seed development-supported`。

## 15. 补充协议

- `MMFR-R1-reliability-supplemental-v1`：target fidelity、reliability-risk relation、B1 downstream gain 三层分开；`explicit-invalid` 与 `implicit-quality` 分开报告。
- `MMFR-S1-severity-sweep-v1`：severity `0.25/0.50/0.75` × 五类，加 `entire_missing@1.0` 和 clean；报告曲线与 robustness-AUC。
- `MMFR-C1-paper-confirmation-v1`：仅在 A2/B1 锁定后，按官方 `1595/1576`、500 epoch、3 paired seeds、sealed official test 单次读取执行。
- R1/S1/C1 只预注册指标、分组、估计量和区间，不设置新的成功/失败 gate；`bootstrap=10000` 与 95% 区间只是计算精度设置。
- 三者当前均未授权、未执行。

## 16. 预期变化与禁止叙事

- 参数增量约 `4,386`，不构成性能保证。
- clean 指标目标是基本稳定，但当前不能写“clean 已保持”。
- corruption 条件可能改善，也可能恶化；必须由冻结评价决定。
- A2 分割变化的直接变量是 corruption exposure；reliability head 只承担 estimator qualification，禁止把潜在分割收益归因于 head。
- 禁止声称“可靠性已经校准”“真实矿井故障已解决”“RGB complete-missing 已解决”“模型可部署”或“达到 SOTA”。

## 17. 风险与停止条件

### 17.1 研究解释风险

- synthetic-to-real gap：合成 target 可能只反映生成器痕迹。
- single seed：不能估计随机方差。
- `val-dev` 双重职责：不能称独立测试。
- RGB-centric：Depth-only A2 不能推出 RGB 缺失能力。

### 17.2 实现风险

- validity state、实际非零 Depth 和 `depth_valid_post` 任一不一致，立即阻塞。
- intensity corruption 产生 `newly_valid_pixels>0`，立即阻塞。
- blur 退回普通含零卷积、misalignment 未同步 transport validity，立即阻塞。
- telemetry 四类不互斥或不能覆盖 `valid_mask`，立即阻塞。
- target/pad/Depth-only/supervised channel/RNG 身份漂移，立即阻塞。
- Ham decoder 配对比较未回放相同 CPU/CUDA RNG，不能归因 MMFR。
- 43 个 optimizer-missing 参数若被静默修复，公平身份失效，必须新 protocol。

### 17.3 资源风险

- 本地 batch size 10 的限步资格不等于云端吞吐结论。
- 4090 容量失败时不得静默改 batch size、学习率或 epoch。
- checkpoint save/load 和完整 epoch 尚未验收。
- 未设置云端最晚停止 schedule 或无法确认 `Stopped` 时不得启动付费长训练。
- official test 任何意外读取都是硬阻塞。

## 18. 外部审计清单

### 18.1 高风险

- [ ] 当前身份是否确为 v3，v1/v2 是否只作为历史保留。
- [ ] A + MID-A 是否同时落在 A1 kernel、A2 helper、config 守卫、训练入口和协议模板中。
- [ ] `V_state_final = depth_valid_post = (raw_depth_post>0) AND valid_mask` 是否被硬断言。
- [ ] mask-normalized blur 是否排除了 invalid zero 对有效邻域的污染。
- [ ] 四类 telemetry 是否互斥、完备且不改变 aggregate loss。
- [ ] 五项 v3 报告、工具身份和 raw/LF 哈希是否可复核。
- [ ] 资格是否被错误写成训练效果、云端通过或 official-test 结果。
- [ ] natural-invalid、8 个 crop、43 个参数是否按高级裁决保持不变。
- [ ] 两个 v3 身份是否从同一 pretrained、seed、split、optimizer 和 schedule 独立开始。
- [ ] 当前是否只具备“申请 probe 授权”的资格，而没有 probe 或训练授权。

### 18.2 中风险

- [ ] severity、curriculum、`p_clean`、`max_specs`、Depth-only 监督是否未漂移。
- [ ] raw/normalized/pad 的 exact-zero 和 uint8 round-trip 是否保持。
- [ ] FP32 reliability branch 是否未改变分割路径。
- [ ] selector、评价条件、bootstrap 和主成功门槛是否未因 v3 改变。
- [ ] v2 validity transport 问题是否作为历史原因保留，而没有重写历史报告。

### 18.3 低风险

- [ ] 4090/5090 价格快照和 `1.4787` break-even 是否正确。
- [ ] raw/LF 哈希约定是否明确。
- [ ] 所有报告是否记录 `official_test_included=false`。
- [ ] Markdown 公式是否只使用 `$...$` 或独立 `$$...$$`。

## 19. 状态矩阵

### 19.1 已完成并验证

- 高级审计 A + MID-A 裁决。
- A1/A2 v3 代码、三个 v3 config、训练入口守卫与四类 telemetry。
- v3 协议模板及完整 raw/LF 身份索引。
- 五项 v3 本地资格，全部 `PASS`。
- v3 本地 commit/tag 冻结；v2 tag 保留、未覆盖。

### 19.2 已完成但不构成效果

- severity/burden、CPU、validity transport、initial-state、AMP update-path 的结构化本地资格。
- 本地 batch size 10 限步运行及约 3.13GB peak allocated 事实。
- `train-dev` 全量 Depth 有效性分布审计。

### 19.3 未执行、未授权

- 云端 batch size 10 容量/吞吐 probe。
- 云实例创建、连接、执行与停止。
- 两个 v3 身份的 500 epoch 正式训练。
- checkpoint save/load、完整 epoch、完整 evaluator。
- R1、S1、C1。
- official test。

### 19.4 待未来独立设计

- `MMFR-B1-learned-geometry-adapter-v1`。
- 覆盖 RGB complete-missing 的 B2 Depth 语义 fallback 或双语义路径。
- optimizer completeness 新身份。
- 自然故障数据与 synthetic-to-real 验证。

### 19.5 禁止

- 覆盖 v1/v2 文件、tag 或历史证据。
- 从 Quick-B0/v1/v2 checkpoint 续训后冒充公平 v3。
- 只训练一侧后宣称公平对照完成。
- 根据结果修改 condition、severity、selector、样本/group、阈值或主指标。
- 把资格、probe、target fidelity 或 telemetry 写成模型效果。
- 未授权创建云资源、开始训练或读取 official test。

## 20. 文件地图与证据指针

- `doc/main/MUSeg-current-status.md`：唯一实时状态入口。
- `doc/main/MUSeg-open-decisions.md`：研究裁决与开放项，第 13 节是 MMFR 当前口径。
- `liu-test-exp/方案1/改动说明.md`：高级审计原文与 v3 执行结果回填。
- `liu-test-exp/方案1/改动细节.md`：v1→v2 与第二轮 v2 本地资格的历史改动快照。
- `liu-test-exp/方案1/改动细节2.md`：v2→v3 的逐文件改动、五项证据、哈希、授权边界与恢复点。
- `protocols/mmfr-a2-train-integration-v3.template.json`：v3 协议和源码、工具、报告 raw/LF 双哈希的权威索引。
- `utils/dataloader/multimodal_failure_v3.py`：A1 v3 corruption 与 sequential validity state。
- `utils/dataloader/mmfr_training_v3.py`：A2 v3 batch helper、target、telemetry masks、确定性 RNG 与 fail-closed 守卫。
- `local_configs/MUSeg/DFormerv2_S_MMFR_A2_*_v3.py`：共同配置与 clean/corruption 两个身份。
- `models/builder.py`：Depth-only reliability BCE 与四类 per-category loss telemetry。
- `utils/train.py`：v1/v2/v3 协议分派、v3 守卫、训练期 telemetry 累积与日志。
- `tools/mmfr/a2_v3_*.py`：五个 v3 资格工具。
- `outputs/mmfr-a1-v3-severity-audit/`、`outputs/mmfr-a2-v3-*/`：五份 canonical 报告；`outputs/` 被 Git 忽略，只通过哈希引用。

## 21. 审计结论模板

### `当前结论：可申请云端容量/吞吐 probe 授权`

适用事实：A + MID-A 已实现；五项 v3 本地资格全部 `PASS`；v3 source identity 已由本地 commit/tag 冻结；没有新的 validity contradiction。该结论只允许下一步向用户询问是否授权短 probe，不授权云资源创建、probe 执行或正式训练。

### `需修订后重新资格`

发现 v3 代码、config、protocol、工具或证据身份不一致；A + MID-A、target、telemetry 或 fail-closed 守卫存在普通可修正缺陷；修订后必须建立新的 source identity，并只重跑与变化风险直接相关的最小资格。

### `阻塞`

出现 invalid Depth 被 intensity corruption 复活、Depth/state transport 不一致、target 与实际输入不一致、四类 telemetry 影响优化权重、两个训练身份初始共享状态或 AMP 更新轨迹不公平、需要改主科学门槛、需要读取 official test，或要求未经授权创建云资源/开始训练。

## 22. 准确恢复点与不得执行事项

### 22.1 准确恢复点

当前身份 `MMFR-A2-train-integration-v3` 已达到 `eligible-for-cloud-capacity-probe-authorization`。v3 冻结提交为 `4ce7b67c5707991750461ef2f806e12b37e203f9`，本地 annotated tag 为 `MMFR-A2-v3-pretrain-freeze`；v2 commit/tag 保持历史，未覆盖、未推送。

下一步是：**先取得用户对云端单 GPU、batch size 10 容量/吞吐短 probe 的单独授权。** 若 probe 获批并通过，仍需用户另行授权 `MMFR-A2-clean-control-v3` 与 `MMFR-A2-depth-corruption-train-v3` 两个公平对照的 500 epoch 正式训练。probe 授权不自动等于训练授权。

### 22.2 不得执行

- 不得在当前文档更新授权下创建云实例、运行 probe 或产生费用。
- 不得启动 500 epoch 训练、生成正式 checkpoint 或运行完整 evaluator。
- 不得读取 official test。
- 不得修改 A + MID-A、natural-invalid 权重、8 个 crop、43 个 optimizer-missing 参数或主成功门槛。
- 不得覆盖 v1/v2 历史文件、报告、commit 或 tag。
- 不得把本地资格写成 MMFR 性能已通过。

## 23. v1→v2 历史修订摘要

v2 在正式训练前完成以下修订：severity 单次编码；blur/misalignment 使用相对尺度；target 改为 `V_pre * R_syn`；Depth-only 监督；invalidity 分项记账；可复跑 CPU、gradient-path、initial-state 与 AMP update-path 门禁；全量 `train-dev` 有效性和 validity transport 审计；R1/S1/C1 去除无依据的新成功 gate。详细逐文件记录保留在 `改动细节.md`，不在本文重写。

v2 的关键终点不是“训练失败”，而是 validity semantics 在正式训练前发现需要改变，因此以 `superseded-before-formal-training-after-validity-semantic-review` 归档。

## 24. 高级审计 v3 指令的逐条处置

1. **保留 v2 历史证据，建立 v3：** 已完成；v2 文件/tag 不覆盖。
2. **主 condition、severity、curriculum、`p_clean`、`lambda_rel`、500 epoch、selector、evaluator 和 success gate 不变：** 已完成。
3. **显式 sequential validity state：** 已完成。
4. **A：intensity corruption 不复活 invalid Depth，valid state 不得编码为 sentinel zero：** 已完成，并由 validity transport `48/48 PASS` 验证。
5. **mask-normalized blur：** 已完成。
6. **MID-A：Depth 与 validity 同步整数平移：** 已完成，并由 actual input 一致性验证。
7. **target 改为 `V_state_final * R_syn`，增加最终三方一致性断言：** 已完成。
8. **保留 natural-invalid、权重和 8 个 crop，新增四类 telemetry 且不改优化权重：** 已完成。
9. **43 个 optimizer-missing 参数保持不修：** 已完成并写入 protocol compatibility。
10. **重跑五项本地资格：** 已完成，五项均 `PASS`。
11. **重新冻结 source identity：** 已完成，commit `4ce7b67c...` 与 tag `MMFR-A2-v3-pretrain-freeze`；未推送远端。
12. **全部 PASS 后升级状态：** 已完成，当前为 `eligible-for-cloud-capacity-probe-authorization`。

**执行边界声明：** v3 收口只完成本地 CPU/GPU 资格和 source freeze；没有云端操作、正式训练、checkpoint、完整评价或 official-test 读取。本文本轮更新本身只做文档重组，不重复运行任何资格、GPU、训练或云任务。
