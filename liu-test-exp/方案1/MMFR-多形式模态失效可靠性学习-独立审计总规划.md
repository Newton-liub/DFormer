# MMFR-多形式模态失效可靠性学习：独立审计总规划

> **文档状态：** 2026-09-15 云端 probe 后审计版。高级审计已正式选择 **A + MID-A**；当前身份为 `MMFR-A2-train-integration-v3`，分析身份为 `MMFR-A2-clean-control-v3` 与 `MMFR-A2-depth-corruption-train-v3`。五项 v3 本地资格、云端 batch size `10` 容量/吞吐 probe（两个身份各一次，各 60 个有效更新步）以及 checkpoint save → 进程销毁 → strict reload → 继续 step 工程门禁均已完成并通过；用户另把执行流程修订为"两阶段省钱策略"，新增阶段 `MMFR-A2-v3-exploratory-one-arm-screening`。当前状态为 `awaiting-user-authorization-for-one-arm-corruption-v3-500e-screening`。
> **授权边界：** probe 与工程门禁的通过**不构成训练授权**。两个 500 epoch 正式训练（第一段 `MMFR-A2-depth-corruption-train-v3` 与条件性的第二段 `MMFR-A2-clean-control-v3`）、完整 evaluator、checkpoint 效果评价、R1/S1/C1 与 official test 均未授权；控制面停机流程验证与 SwanLab `online` 监控尚未完成；batch size 10 在 4090 上的显存余量处置待用户决定。云端逐项事实、数值、哈希与失败归档见 `liu-test-exp/方案1/改动细节3-云服务器.md`。
> **身份冻结：** v2 以 commit `3c8ebddeb76e63a5be37e261381117abfba117ca` 和 annotated tag `MMFR-A2-v2-pretrain-freeze` 保留为历史；v3 以 commit `4ce7b67c5707991750461ef2f806e12b37e203f9` 和 annotated tag `MMFR-A2-v3-pretrain-freeze` 冻结。两个标签均仅在本地，未推送远端。
> **大白话说明：** v2 审计发现的“普通噪声或模糊会把原生空洞变成假深度”和“错位后有效性坐标不明确”已经在独立 v3 中修正，并通过五道本地门禁；随后云端短测试也跑完了：两张卡容量测试确认 batch size 10 能跑，存档-杀进程-重载-继续训练这条链被证明与不中断完全一致。但实测同时暴露两条新事实——"深度损坏"那一路每秒只处理约 5 张、比干净那一路慢 3.4 倍，所以省钱策略先跑的那一段反而更贵（约 36 小时、约 67 元）；而且每次 validation 之后只剩约 1.73 GB 显存余量。现在只能等你决定显存余量怎么处置，并单独批准"只跑一个 500 epoch"的省钱筛选，不能直接开两段训练。
> **事实入口：** `doc/main/MUSeg-current-status.md` 是唯一实时状态入口；本文负责重组自包含审计叙事，不替代实时状态。

## 0. 文档用途、形成时点与阅读规则

- **用途：** 本文件供无法直接看到仓库、运行产物和云资源的外部审计者使用。它集中说明项目背景、历史依据、当前 v3 设计、资格证据、正式实验计划、风险、授权边界和恢复点。
- **形成时点：** 2026-09-15，v3 代码、配置、协议模板、五个资格工具及五份 canonical 报告已经冻结；正式 MMFR 训练和效果评价尚未开始。
- **证据优先级：**
  1. 当前工作区中可直接核验的代码、配置、协议模板、运行产物与哈希；
  2. `doc/main/MUSeg-current-status.md` 的实时事实与授权边界；
  3. `doc/main/MUSeg-open-decisions.md` 第 13 节的研究裁决；
  4. `liu-test-exp/方案1/改动说明.md` 的高级审计原文与 v3 执行回填；
  5. `liu-test-exp/方案1/改动细节2.md`（v2→v3 代码改动）与 `liu-test-exp/方案1/改动细节3-云服务器.md`（云端 probe、工程门禁、实测事实与失败归档）；
  6. 本文及其他计划文档对未来实验的说明。
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
- v3 五项本地资格均已通过；云端 batch size `10` 容量/吞吐 probe 已在单张 RTX 4090 上对**两个身份**各完成一次（各 `60` 个有效更新步、`68` 次尝试、`8` 次 GradScaler 跳过、loss 全有限、`official_test_included=false`）；checkpoint save → 进程销毁 → strict reload → 继续 step 工程门禁也已通过。
- **云端实测吞吐：** `clean-control-v3` 稳定中位数 `16.43` 张/秒（约 `0.609` 秒/步），峰值 allocated `18,951.5 MiB`、reserved `20,556 MiB`，最小剩余 `3,068.6 MiB`（`12.73%`）；`depth-corruption-v3` 稳定中位数 `4.94` 张/秒（约 `2.02` 秒/步），峰值 allocated `19,046.4 MiB`、reserved `21,286 MiB`，最小剩余 `2,330.6 MiB`（`9.67%`）。按 `128` 步/epoch 与 `1.88 元/小时` 保守外推：corruption 500 epoch 约 `36.0` 小时、约 `67` 元；clean 约 `10.8` 小时、约 `20 元`；即"省钱策略"的第一段反而是贵的一段。
- **云端实测显存事实：** 经过一次 validation 之后，batch size `10` 的第 2 个 epoch 首个训练步只剩 `1.73 GiB` 自由显存（低于历史 `2 GiB` 余量，**不是 OOM**），且在"续跑"与"不中断"两条独立轨迹上同样复现。
- **checkpoint 门禁判定：** 从 parent 的 epoch-1 存档 strict reload 的续跑轨迹，与不中断跑完 2 个 epoch 的参照轨迹，在只忽略协议 `run_id` 的条件下 `mismatches=[]`，model / optimizer / amp_scaler / rng_state 四个 component SHA-256 逐位相同。
- **执行计划修订：** 用户要求改为"两阶段省钱策略"，新增 `MMFR-A2-v3-exploratory-one-arm-screening`（probe 之后、正式 paired 训练之前，只做成本筛选，不改变主 success gate，不新建 v4）。下一步只申请一个 500 epoch（corruption-v3）；probe 与门禁通过不构成训练授权。

### 1.2 大白话版方案

未来要公平地训练两个模型：clean control 始终看干净输入，Depth-corruption model 按冻结随机规则损坏 Depth，并额外学习像素级 Depth 可靠性。两者从同一官方预训练、同一 seed、同一数据和同一训练预算独立开始。A2 的可靠性预测仍只承担辅助监督，不进入 backbone、decoder 或 geometry prior，所以 A2 的分割差异只能直接归因于 corruption exposure，不能归功于可靠性头。

v3 新增的关键保护是：一个位置如果当前没有有效 Depth，普通噪声、模糊或量化就不能凭空生成“新测量”；如果 Depth 被平移，表示“这里有无测量”的 validity 也一起平移。这样实际模型输入、有效性状态和监督 target 使用同一套坐标与语义。

### 1.3 v2→v3 带来什么、没带来什么

- **带来：** 新的 A1/A2 v3 身份；显式 sequential validity state；A + MID-A；mask-normalized blur；最终 target `V_state_final * R_syn`；四类互斥 telemetry；v3 协议模板；五个可复跑资格工具和五份 `PASS` 报告；新的本地冻结 commit/tag。
- **保持：** 六类 failure、severity、curriculum、`p_clean=0.25`、`max_specs=2`、`lambda_rel=0.1`、Depth-only 监督、500 epoch、optimizer、学习率、batch size、selector、主 evaluator、成功门槛、8 个全无有效 Depth crop、43 个 optimizer-missing 参数。
- **没有带来：** 没有正式训练、没有正式 checkpoint、没有 MMFR mIoU、没有 R1/S1/C1 结果、没有 official test；probe 与工程门禁只回答容量、吞吐、AMP 有限性和存档恢复能力，不是效果证据。
- **本轮云端带来（`改动细节3-云服务器.md`）：** 一个可复跑的云端执行链路（数据集落地并逐位核验、官方 pretrained 逐位核验、环境补齐、三份 qualification 清单与 GPU preflight）；两个身份的 batch size 10 容量/吞吐实测；checkpoint 存档/重载等价的工程判定；以及两条新的实测事实（corruption 慢 3.4 倍导致成本不对称、validation 后只剩 1.73 GiB 余量）。

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
云端 probe + checkpoint 工程门禁（已完成，见 `改动细节3-云服务器.md`）
    ├─ batch size 10 容量/吞吐：clean 16.43 张/秒、corruption 4.94 张/秒
    ├─ 存档 → 进程销毁 → strict reload → 继续 step：与不中断轨迹逐位一致
    └─ 实测：validation 后余量 1.73 GiB；corruption 单步成本约为 clean 的 3.3 倍
             │
             ▼
MMFR-A2-v3-exploratory-one-arm-screening（待用户单独授权，成本筛选）
    ├─ 只跑一个 500 epoch：MMFR-A2-depth-corruption-train-v3
    ├─ 冻结 checkpoint 后用现有主 evaluator、evaluation seed 与 failure conditions 评价
    ├─ Quick-B0 仅作 historical RGB development reference，不是 clean control
    └─ 结论只有 screening-promising / screening-unpromising / screening-inconclusive，然后停下等高级审查
             │
             ▼
正式 paired 评价（条件性，需再次单独授权）
    ├─ 只有存在研究苗头才授权 MMFR-A2-clean-control-v3 的 500 epoch
    └─ 正式结论只比较 Depth-corruption-v3 − Clean-control-v3
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

- **已完成（用户单独授权）：** 创建云实例 `cpod-1vbh7faqcauq`；云端 batch size `10` 容量/吞吐 probe（两个身份各一次）；checkpoint save → 进程销毁 → strict reload → 继续 step 工程门禁。运行提交为 `9c4059d0…`（probe）与 `cdd9ba09…`（门禁），两次均 `dirty=false`。
- **可以做：** 向用户申请 `MMFR-A2-v3-exploratory-one-arm-screening` 阶段的**一个** 500 epoch 训练（`MMFR-A2-depth-corruption-train-v3`）授权。
- **尚未授权：** 任何 500 epoch 正式训练（含条件性的 `MMFR-A2-clean-control-v3`）、完整 evaluator、checkpoint 效果评价、R1/S1/C1、official test，以及本次已授权操作之外的云资源操作。
- **尚未完成：** 控制面停机流程验证（本实例无 `compshare` CLI 与凭据，需用户侧执行）、SwanLab `online` 监控凭据（`/root/.config/dformer/swanlab.env` 缺失）、batch size `10` 显存余量处置决定。
- **本机角色：** 推理、想法初步验证和小规模 preflight，不承担正式长训练。

### 12.3 生命周期要求

未来任何付费运行必须在启动前确认实例、最长时长和预计费用，设置并复核控制面最晚停止 schedule；成功、失败或中止后都先取回必要证据并核验哈希，再调用控制面 stop，最后确认平台状态为 `Stopped`。验收 pass/fail 不决定是否停止计费。

### 12.4 云端实测结果与预算（2026-09-15）

- 两个身份的 batch size `10` 容量/吞吐实测、显存极值、运行时长、两次安全闸中止与放宽、以及 500 epoch 时间/费用外推，完整数值见 `改动细节3-云服务器.md` 第 5、7、8 节。
- 摘要：`clean-control-v3` 稳定 `16.43` 张/秒、最小剩余 `3,068.6 MiB`（`12.73%`）；`depth-corruption-v3` 稳定 `4.94` 张/秒、最小剩余 `2,330.6 MiB`（`9.67%`）。按 `128` 步/epoch 与 `1.88 元/小时`：corruption 500 epoch 约 `36.0` 小时 / 约 `67` 元，clean 约 `10.8` 小时 / 约 `20` 元。
- 显存安全闸差异：corruption 侧曾按历史 `0.10` 余量在 `attempt 8` 中止（`free ratio 0.097`，**非 OOM**），最终以显式放宽为 `0.05` 完成完整 60 步；clean 侧使用 `0.10`。该差异记录在 `probe-result.json` 的阈值字段中，两次运行与中止归档均保留。
- validation 之后 batch size `10` 只剩 `1.73 GiB` 自由显存（在续跑与不中断两条轨迹上同样复现），属于"能跑但余量偏紧"；这是启动 500 epoch 前必须由用户处置的开放项。
- 因此 one-arm screening 应预算约 `36–40` 小时、`68–75` 元；若高级审查后追加 clean control，再加约 `11–14` 小时、`21–27` 元。以上均不含云盘与镜像费用。

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

- 五项本地资格本身不覆盖云端 RTX 4090/5090 的容量与吞吐；该缺口已由本轮云端 probe 单独补齐（见 13.4），但 probe 只覆盖两个身份各 `60` 个测量步。
- 不覆盖 500 epoch 完整训练、训练期周期性 validation 的长期稳定性、完整 evaluator、DDP 或 `torch.compile`。
- 不提供 mIoU、可靠性校准、真实故障、跨设备或部署收益。
- 不授权任何云资源或训练。

### 13.4 云端 probe 与 checkpoint 工程门禁证据（2026-09-15）

- 证据根目录（仓库外）：`/root/rivermind-data/cloud/mmfr-a2-v3-probe/`；逐项数值、命令与哈希见 `liu-test-exp/方案1/改动细节3-云服务器.md`。
- 两份 probe 清单与一份门禁清单均通过完整 GPU preflight（`pass=true`、`errors=[]`）；probe preflight 报告 SHA-256 为 `4997904524d988eecc5c47f029e090eda0480599bd9c6005035c6ace8ab6e934`（clean）与 `8e15c63c9ac04594fafb864210d492a4efa25327306c71b4dbf0390a9392d920`（corruption），门禁清单 SHA-256 为 `d4d741ace5f1549586bc0bfebb1e194284eafe92f036c9c1c60a69b65e418db9`。
- probe 结果：`clean-control-v3` `probe-result.json` SHA-256 `8d8ac335acc0f4a803f0c1e7ca9935adefb781482843ba3354aec1cc6a3d0626`、`depth-corruption-v3` `a8f3a614606572ccdb0f801f2ad9f196e81c55a748b231de3114841bbe3e24f0`；两者均 `eligible=true`、`exact_target_met=true`、`all_steps_passed=true`、`anomaly=null`。
- 门禁结果：`compare-child-vs-reference.json` SHA-256 `e0189aa5178692b9c6627676e6892fc919db8612ec271b87c61ab828adf0ce4c`，`pass=true`、`mismatches=[]`；model / optimizer / amp_scaler / rng_state 四个 component 哈希在续跑与不中断两侧逐位相同。
- **证据边界：** probe 的 `step_seconds` 含 probe 专有 telemetry 哈希开销（约 `35 ms/步`）且不含每 `10` epoch 的 validation；门禁为了隔离存档/重载变量而关闭了显存安全闸；corruption 侧 probe 使用放宽到 `0.05` 的余量门槛。所有失败与降级归档都在 `attempts/` 下，不得作为验收证据。

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

### 14.4 `MMFR-A2-v3-exploratory-one-arm-screening`（用户第 2 轮要求的执行计划修订）

- **位置与性质：** 位于云端容量 probe **之后**、正式 paired 训练 **之前**；它是**成本筛选**，不是正式因果实验，**不改变主 success gate**，也**不新建科学协议 v4**（只改执行顺序、授权状态与结果解释，v3 科学身份继续使用）。
- **它只回答一个问题：** "`MMFR-A2-depth-corruption-train-v3` 是否表现出值得继续投入第二个 500 epoch clean control 的明显研究苗头？"
- **执行顺序：** probe（已完成）→ 只申请并执行**一个** 500 epoch（corruption-v3，从冻结官方 pretrained 独立开始，不从 Quick-B0/v1/v2 续训）→ 冻结 checkpoint 后用现有主 evaluator、evaluation seed 与 failure conditions 评价 → 生成 `one-arm-screening-report` → 状态置为 `awaiting-senior-review-of-one-arm-screening` 并**停下** → 高级审查 → 视情况单独申请 `MMFR-A2-clean-control-v3` 的 500 epoch → 通过后 Quick-B0 降级为普通历史参考 → 正式结论只比较 `Depth-corruption-v3 − Clean-control-v3`。
- **允许的第一阶段结论只有：** `screening-promising`、`screening-unpromising`、`screening-inconclusive`。**禁止：** `A2 supported`、`MMFR improves robustness`、`causal gain`、`正式成功`。
- **Quick-B0 的角色：** 只能作为 `historical RGB development reference`（single-seed RGB 开发基线，clean mIoU `58.79`），**不得写成 `clean control`**；禁止把 `Depth-corruption-v3 − Quick-B0` 写成 MMFR robustness gain。必须保留的一句话：**第一阶段 Quick-B0 只能帮我们决定"还值不值得继续花钱"，不能帮我们证明"MMFR 有效"；正式证明依旧需要第二阶段的 clean-v3 配对实验。**
- **报告最小字段：** corruption-v3 选中的 epoch；clean mIoU；六个单 failure mIoU；六单 macro-average；三个 mixed failure；相对 Quick-B0 的逐条件差值；四类 telemetry 统计；checkpoint 哈希；source commit/tag；evaluator identity；GPU、耗时、费用；以及"Quick-B0 不是配对 clean control"的显式声明。
- **不得自动升级：** 第一阶段结束后不得自动启动第二个 500 epoch；也不得根据第一阶段结果修改主 success gate、condition、severity、selector、样本/group 或 bootstrap。

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
- 不得把云端 probe 的 `eligible=true`、吞吐数字或 checkpoint 门禁 `PASS` 写成 MMFR 效果、鲁棒性收益或训练成功。
- 不得把 one-arm screening 的 `screening-promising` 写成 `A2 supported`、`MMFR improves robustness`、`causal gain` 或“正式成功”；也不得把 Quick-B0 当作配对 clean control。

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

- 云端吞吐已实测，但 probe 只覆盖两个身份各 `60` 个测量步；500 epoch 的长期稳定性与周期性 validation 行为仍未验证。
- corruption 的实测单步成本约为 clean 的 `3.3` 倍（`4.94` vs `16.43` 张/秒），因此 one-arm screening 约 `36–40` 小时 / `68–75` 元；不能沿用"第一段更便宜"的直觉预算。
- validation 之后 batch size `10` 只剩 `1.73 GiB` 自由显存（**非 OOM**）；在该余量处置被用户明确之前，不得静默改 batch size、学习率或 epoch，也不得静默改变显存安全阈值。
- checkpoint save/reload 工程门禁已通过，但控制面停机流程与 SwanLab `online` 监控尚未验证；未设置最晚停止 schedule 或无法确认 `Stopped` 时不得启动付费长训练。
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
- [ ] 云端 probe 是否只被写成容量/吞吐/AMP 有限性与存档恢复证据，而没有写成效果结论。
- [ ] Quick-B0 是否只被写成 `historical RGB development reference`，而不是 v3 的 clean control。
- [ ] 显存安全闸的两次中止数值、放宽后的门槛与全部失败归档是否完整保留、未被隐藏。

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
- [ ] 云端实测吞吐、显存、时长与预算数值是否与 `改动细节3-云服务器.md` 一致（`16.43` / `4.94` 张/秒；`3,068.6` / `2,330.6` MiB；validation 后 `1.73 GiB`；约 `36.0` / `10.8` 小时）。

## 19. 状态矩阵

### 19.1 已完成并验证

- 高级审计 A + MID-A 裁决。
- A1/A2 v3 代码、三个 v3 config、训练入口守卫与四类 telemetry。
- v3 协议模板及完整 raw/LF 身份索引。
- 五项 v3 本地资格，全部 `PASS`。
- 云端 batch size `10` 容量/吞吐 probe（两个身份各一次，`eligible=true`、`exact_target_met=true`）。
- checkpoint save → 进程销毁 → strict reload → 继续 step 工程门禁（`pass=true`、`mismatches=[]`）。
- 云端执行链路的环境补齐，以及数据集与官方 pretrained 的逐位核验。
- v3 本地 commit/tag 冻结；v2 tag 保留、未覆盖。

### 19.2 已完成但不构成效果

- severity/burden、CPU、validity transport、initial-state、AMP update-path 的结构化本地资格。
- 本地 batch size 10 限步运行及约 3.13GB peak allocated 事实。
- `train-dev` 全量 Depth 有效性分布审计。
- 云端吞吐与显存实测（`16.43` vs `4.94` 张/秒；最小剩余 `3,068.6` / `2,330.6 MiB`；validation 之后 `1.73 GiB`）以及 500 epoch 时间/费用外推。
- checkpoint 存档等价性判定。

### 19.3 未执行、未授权

- 两个 v3 身份的 500 epoch 正式训练（含 one-arm screening 的第一段）。
- 完整 evaluator 与 checkpoint 效果评价。
- R1、S1、C1。
- official test。
- 控制面停机流程验证与 SwanLab `online` 监控。

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

### 19.6 待用户决定（2026-09-15）

- batch size `10` 在 4090 上的显存余量处置：保持 4090 并接受该余量 / 先补一次余量验证 / 改用 5090。换卡不得改变任何科学超参数。
- 是否授权 `MMFR-A2-v3-exploratory-one-arm-screening` 的一个 500 epoch（`MMFR-A2-depth-corruption-train-v3`），预算约 `36–40` 小时 / `68–75` 元。
- 控制面停机流程验证的执行方式与 SwanLab `online` 凭据提供方式。

## 20. 文件地图与证据指针

- `doc/main/MUSeg-current-status.md`：唯一实时状态入口。
- `doc/main/MUSeg-open-decisions.md`：研究裁决与开放项，第 13 节是 MMFR 当前口径。
- `liu-test-exp/方案1/改动说明.md`：高级审计原文与 v3 执行结果回填。
- `liu-test-exp/方案1/改动细节.md`：v1→v2 与第二轮 v2 本地资格的历史改动快照。
- `liu-test-exp/方案1/改动细节2.md`：v2→v3 的逐文件改动、五项证据、哈希、授权边界与恢复点。
- `liu-test-exp/方案1/改动细节3-云服务器.md`：本轮云端 probe、显存安全闸中止与放宽、checkpoint 工程门禁、实测速度与显存事实、成本外推、one-arm screening 执行计划、失败归档与准确恢复点。
- `/root/rivermind-data/cloud/mmfr-a2-v3-probe/`（仓库外，不进入 Git）：本轮云端证据根目录，含 `manifests/`、两个 probe 运行目录、`depth-corruption-v3-checkpoint-gate/`、以及全部失败与降级归档 `attempts/`。
- `protocols/mmfr-a2-train-integration-v3.template.json`：v3 协议和源码、工具、报告 raw/LF 双哈希的权威索引。
- `utils/dataloader/multimodal_failure_v3.py`：A1 v3 corruption 与 sequential validity state。
- `utils/dataloader/mmfr_training_v3.py`：A2 v3 batch helper、target、telemetry masks、确定性 RNG 与 fail-closed 守卫。
- `local_configs/MUSeg/DFormerv2_S_MMFR_A2_*_v3.py`：共同配置与 clean/corruption 两个身份。
- `models/builder.py`：Depth-only reliability BCE 与四类 per-category loss telemetry。
- `utils/train.py`：v1/v2/v3 协议分派、v3 守卫、训练期 telemetry 累积与日志。
- `tools/mmfr/a2_v3_*.py`：五个 v3 资格工具。
- `outputs/mmfr-a1-v3-severity-audit/`、`outputs/mmfr-a2-v3-*/`：五份 canonical 报告；`outputs/` 被 Git 忽略，只通过哈希引用。

## 21. 审计结论模板

### `当前结论：probe 与工程门禁通过；等待 one-arm screening 训练授权`

适用事实：五项 v3 本地资格 `PASS`；云端 batch size `10` 容量/吞吐 probe 对两个身份各完成一次（`eligible=true`、`exact_target_met=true`）；checkpoint save → strict reload → 继续 step 工程门禁 `pass=true`、`mismatches=[]`；执行计划已修订为"两阶段省钱策略"，新增 `MMFR-A2-v3-exploratory-one-arm-screening`。该结论只允许下一步：先处置显存余量、控制面停机流程与 SwanLab 凭据，然后**只申请一个** 500 epoch（corruption-v3）的 one-arm screening 授权。**不授权**任何 500 epoch 训练、完整 evaluator、checkpoint 效果评价或 official test。

### `历史结论：可申请云端容量/吞吐 probe 授权（已被上一条取代）`

适用事实：A + MID-A 已实现；五项 v3 本地资格全部 `PASS`；v3 source identity 已由本地 commit/tag 冻结。该结论只允许向用户询问是否授权短 probe。probe 已于 2026-09-15 完成，故本结论降级为历史。

### `需修订后重新资格`

发现 v3 代码、config、protocol、工具或证据身份不一致；A + MID-A、target、telemetry 或 fail-closed 守卫存在普通可修正缺陷；修订后必须建立新的 source identity，并只重跑与变化风险直接相关的最小资格。

### `阻塞`

出现 invalid Depth 被 intensity corruption 复活、Depth/state transport 不一致、target 与实际输入不一致、四类 telemetry 影响优化权重、两个训练身份初始共享状态或 AMP 更新轨迹不公平、需要改主科学门槛、需要读取 official test，或要求未经授权创建云资源/开始训练。

## 22. 准确恢复点与不得执行事项

### 22.1 准确恢复点

当前身份 `MMFR-A2-train-integration-v3`。v3 冻结提交为 `4ce7b67c5707991750461ef2f806e12b37e203f9`，本地 annotated tag 为 `MMFR-A2-v3-pretrain-freeze`；本轮 probe 运行在 `9c4059d0fd8ffbdfcebdb3dffbe9b9e7d1f8c1d8`、checkpoint 门禁运行在 `cdd9ba092d296458593009361dcb12011e4e18ce`（两者均为干净工作区）；v2 commit/tag 保持历史，未覆盖、未推送。

当前准确恢复点：

`v3 scientific protocol frozen; local qualifications PASS; cloud batch-10 capacity/throughput probe COMPLETED for both identities; checkpoint save / strict-reload / continue-step engineering gate PASS (resumed child == uninterrupted reference on model, optimizer, amp_scaler and rng_state component hashes); next sequence = control-plane stop-flow verification + SwanLab online credentials + user decision on the batch-10 VRAM margin -> user authorization for one-arm MMFR-A2-depth-corruption-train-v3 500e screening -> senior screening review -> conditional clean-control-v3 500e authorization -> formal paired evaluation`

下一步是：先处置 batch size `10` 的显存余量、控制面停机流程验证与 SwanLab `online` 凭据，然后**只取得一个 500 epoch（`MMFR-A2-depth-corruption-train-v3`）的 one-arm screening 授权**。若高级审查认定存在值得继续的苗头，才另行授权 `MMFR-A2-clean-control-v3` 的 500 epoch 配对训练。probe 与工程门禁的通过不自动等于训练授权。

### 22.2 不得执行

- 不得在当前文档更新授权下创建云实例、运行 probe 或产生费用。
- 不得启动 500 epoch 训练、生成正式 checkpoint 或运行完整 evaluator。
- 不得读取 official test。
- 不得修改 A + MID-A、natural-invalid 权重、8 个 crop、43 个 optimizer-missing 参数或主成功门槛。
- 不得覆盖 v1/v2 历史文件、报告、commit 或 tag。
- 不得把本地资格写成 MMFR 性能已通过。
- 不得把 probe 或 checkpoint 门禁写成 MMFR 效果。
- 不得自动启动第二个 500 epoch，也不得把 one-arm screening 结果写成正式配对结论。
- 不得把 Quick-B0 写成 clean control，或把 `Depth-corruption-v3 − Quick-B0` 写成 MMFR robustness gain。
- 不得静默修改显存安全阈值、batch size、学习率或 epoch，也不得把"门槛放宽"写成默认口径。

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
12. **全部 PASS 后升级状态：** 已完成，当时状态为 `eligible-for-cloud-capacity-probe-authorization`（该状态已被 2026-09-15 云端 probe 与工程门禁的完成取代，见 24.2 与第 22.1 节）。

**执行边界声明：** v3 本地收口那一轮只完成本地 CPU/GPU 资格和 source freeze；那一次没有云端操作、正式训练、checkpoint、完整评价或 official-test 读取。本文档更新本身不重复运行任何资格、GPU、训练或云任务。**后续云端轮次的边界见 24.2 末尾。**

### 24.2 用户第 2 轮执行计划指令的逐条处置（2026-09-15）

1. **改成"两阶段省钱策略"但不改 v3 科学变量：** 已落实为执行顺序、授权矩阵和结果解释的修订；六类 failure、severity、curriculum、`p_clean=0.25`、`lambda_rel=0.1`、500 epoch、batch size `10`、optimizer、selector、evaluator、主 success gate 与 43 个 optimizer-missing 参数全部未改（第 14.4 节）。
2. **新增 `MMFR-A2-v3-exploratory-one-arm-screening` 阶段：** 已写入本文第 7、14.4 节与 `doc/main/MUSeg-current-status.md`；`protocols/mmfr-a2-train-integration-v3.template.json` 与 `doc/main/MUSeg-open-decisions.md` 的同步状态以实时状态文件为准。
3. **云端 probe 仍须先执行且需单独授权：** 已执行完成（第 12.4、13.4 节）；probe 授权没有升级为训练授权。
4. **probe 通过后只申请一个 500 epoch（corruption-v3）：** 当前恢复点已按该逻辑书写；训练本身尚未授权。
5. **冻结 checkpoint 后再评价，并输出 clean / 六单 / 三混合 / mIoU / mAcc / mF1 / Boundary IoU / 六单宏平均：** 已写入第 14.4 节的报告最小字段；尚未执行。
6. **Quick-B0 只作 `historical RGB development reference`：** 已写入第 14.4 节，并在 16、22.2 节明确禁止写成 `clean control`。
7. **禁止把 `Depth-corruption-v3 − Quick-B0` 写成 MMFR robustness gain：** 已写入第 14.4、16、22.2 节；第一阶段允许结论只有 `screening-promising` / `screening-unpromising` / `screening-inconclusive`。
8. **不得由下级模型发明新的 screening 数值门槛：** 已写入第 14.4 节；主 success gate 未新增任何 screening 门槛，报告字段不构成成功条件。
9. **高级审查后两条路径：** 已写入第 7 节阶段图与 14.4 节（无苗头则暂停 A2 并保留探索性负结果；有苗头才单独申请 clean-control-v3 的 500 epoch）。
10. **第二阶段必须从同一 pretrained、seed、`train-dev`、optimizer、LR、500 epoch、batch size、selector 独立训练，不得从 corruption checkpoint 反向构造 clean control：** 与第 11.1 节共同冻结项一致，未改。
11. **第二阶段完成后 Quick-B0 降级为普通历史参考，正式结论只比较 `Depth-corruption-v3 − Clean-control-v3`：** 已写入第 14.4 节。
12. **不新建科学 protocol v4：** 本轮只新增执行计划记录与仓库外清单；v3 科学身份、生产代码、corruption、loss、target、selector、evaluator 全部未改。本轮仓库内唯一提交为只包含状态文档的 `cdd9ba092d296458593009361dcb12011e4e18ce`。
13. **更新准确恢复点：** 已完成，见第 22.1 节。

**本轮云端边界更新声明：** 与 24.1 末尾"没有云端操作"的旧声明不同，本轮**确实在云端执行了 batch size 10 容量/吞吐 probe 与 checkpoint 工程门禁**（用户单独授权，见 `liu-test-exp/方案1/改动细节3-云服务器.md` 第 2 节），但仍未执行任何 500 epoch 训练、完整 evaluator、checkpoint 效果评价或 official test。
