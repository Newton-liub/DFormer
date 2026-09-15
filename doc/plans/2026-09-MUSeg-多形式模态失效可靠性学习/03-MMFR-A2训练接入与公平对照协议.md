# MMFR-A2 训练接入与公平对照协议

> **文档角色：** 条件式子计划与实现执行单。
> **计划状态：** v1 的设计冻结、限定训练接入、CPU qualification 与本地 GPU 单步 preflight 已完成（历史事实与哈希原样保留）；v2 修订（`MMFR-A2-train-integration-v2`，supersede v1）已实现、已通过 CPU severity/burden 审计；v2 的 `gradient_path_isolation` 门禁已于 2026-09-15 01:32（本地时间）在本地 RTX 5060 Laptop GPU 上执行并通过（退出码 `0`、墙钟 `15.3` 秒、结论 `gradient-path-isolation: PASS`，证据见第 13.2 节）。正式训练资源已固定为云端单 GPU、默认 RTX 4090，本机仅做推理与小规模验证。云端 probe、正式训练、checkpoint 评价和 official test 未授权。
> **形成或核验时点：** 2026-09-14 02:02 UTC；v2 语义修订时点 2026-09-14（UTC）。
> **实时入口：** [`MUSeg-current-status.md`](../../main/MUSeg-current-status.md)。
> **研究选择：** [`MUSeg-open-decisions.md`](../../main/MUSeg-open-decisions.md)。
> **上级方向：** [`00-总方向规划.md`](00-总方向规划.md)。
> **当前/后继关系：** 前序 [`02-MMFR-A1失效基函数与可靠性脚手架.md`](02-MMFR-A1失效基函数与可靠性脚手架.md) 已完成（v1）并修订为 v2；本阶段 v1 已接入 Depth 合成失效和辅助可靠性监督，并通过真实本地 GPU 单步 preflight，但未接入 geometry adapter；v2 修订了 corruption basis 与监督目标身份。正式训练固定为云端单卡并默认使用 RTX 4090，本机只做推理与小规模验证。v2 `gradient_path_isolation` 门禁已于 2026-09-15 在本地 GPU 上执行并通过，该前置条件已满足、不再阻塞；恢复点仍保持云资源与训练未授权，由用户单独授权后依次推进：（1）可选地先补一次 v2 本地 GPU 单步 preflight，覆盖真实模型 AMP 路径（v2 目前尚无 GPU preflight）；（2）云端冻结 batch size `10` 容量/吞吐短 probe；（3）clean control/Depth-corruption 公平对照训练；`MMFR-B1-learned-geometry-adapter-v1` 尚未细化或授权。

## 1. 实验族身份与本阶段问题

- **协议身份（当前）：** `MMFR-A2-train-integration-v2`，supersede `MMFR-A2-train-integration-v1`。
- **supersede 原因：** “正式 500 epoch 尚未执行，所以在结果产生前进行协议修订”，修订内容是 v1 的 severity 双编码、分辨率相关的 `blur`/`misalignment` severity、对原生无效 Depth 的未限定 reliability target，以及被监督但未定义的 RGB reliability 通道。
- **clean control（v2）：** 分析身份 `MMFR-A2-clean-control-v2`，run id `museg-dformerv2-s-mmfr-a2-clean-control-v2`；沿用原 DFormerv2-S 结构，只使用 clean 训练输入。
- **corruption model（v2）：** 分析身份 `MMFR-A2-depth-corruption-train-v2`，run id `museg-dformerv2-s-mmfr-a2-depth-corruption-v2`；使用 Depth 合成失效、corrupted segmentation loss 和辅助 reliability loss。
- **v1 身份（冻结、历史）：** 协议 `MMFR-A2-train-integration-v1`，分析身份 `MMFR-A2-clean-control-v1` 与 `MMFR-A2-depth-corruption-train-v1`，run id `museg-dformerv2-s-mmfr-a2-clean-control-v1` 与 `museg-dformerv2-s-mmfr-a2-depth-corruption-v1`。v1 模块与 config 保持冻结、未被改写；v2 只在协议语义无关处复用 v1 的基础设施与常量，并在 import 时断言一致。
- **本阶段问题：** 在不改 DFormerv2 geometry prior、不做 clean/corrupt 双前向的前提下，把 A1 v2 的 Depth corruption 与连续 target 接进现有训练链，并冻结可复现、公平、可审计的输入和损失语义。
- **本阶段科学含义（v2）：** A2 = **Depth corruption robustness training + parallel reliability estimation**，即“Depth corruption 鲁棒性训练 + 并行可靠性估计”。A2 的 predicted reliability 不影响 segmentation backbone、decoder 或 geometry prior，因此 A2 的 segmentation gain **不能**归因于 reliability head；B1 才是第一个真正的 reliability-aware segmentation 阶段。
- **不主张：** A2 接入通过不代表模型已获得鲁棒性；当前 DFormerv2 是 **RGB-primary semantic path + Depth geometry-prior path 的非对称架构**，不能把结果外推到 RGB complete-missing，也不能外推到真实矿井传感器故障率。

## 2. 固定输入、训练起点与公平性

- 数据职责保持 `train-dev=1277`、`val-dev=318`；official test 保持 `sealed_unread`。
- 两个训练身份都从同一官方 `DFormerv2_Small_pretrained.pth` 独立初始化，使用同一 seed `772961337`、AdamW、学习率、500 epoch、batch size、尺度增强、clean selector 和 evaluator 规则。
- 不从 Quick-B0 的 `selector-epoch-420.pth` 续训。
- checkpoint selector 只看 clean `val-dev` 的原图网格 mIoU，规则保持 `original-full`、scale `1.0`、无 flip、同样的评估间隔与 tie-break；失效条件不参与选 epoch。
- 两个模型允许各自按同一 selector 规则选出不同 epoch，但不得为其中一个模型追加额外候选或改变 tie-break。

## 3. 数据流冻结

### 3.1 接入位置

corruption 在 `DataLoader` 已完成 mirror、scale、crop/pad 之后、批次搬到 GPU 之前，于主训练进程逐样本执行。这样 reliability target 天然处于最终 `480×640` crop 几何，不再跨 worker 同步 epoch 状态，也不需要在 target 上重放 mirror/scale/crop。

现有 `TrainPre`、`RGBXDataset` 和 worker 内几何增强语义保持不变。A2 只把最终 CPU batch 转换为 corrupted batch，不修改 worker 数量、worker seed 或样本采样方式。大白话说，先得到与旧训练完全相同的裁剪批次，再在主进程里损坏 Depth。

### 3.2 raw `[0,1]` 与 normalized tensor 职责

- segmentation backbone 继续消费现有 normalized RGB/Depth tensor。
- reliability estimator 只消费从最终 normalized crop 严格恢复的 uint8 信号再除以 255 得到的 raw `[0,1]` tensor。
- RGB 逆变换使用配置 `norm_mean/norm_std`；Depth 逆变换固定使用现有 `TrainPre(sign=True)` 的 `[0.48,0.48,0.48]` 与 `[0.28,0.28,0.28]`。
- 恢复过程必须对全部 uint8 值 `0..255` 做资格检查：`normalize -> float32 -> inverse+round` 必须逐值恢复原 uint8；失败即 `qualification-blocked`。
- 原训练 pad 的 normalized 值是精确 `0`。几何有效 mask 定义为 RGB 与 Depth 不同时为逐通道精确全零；corruption 后必须把无效 pad 位置恢复为原 tensor 的精确 `0`，不能把 pad 重新解释成黑色观测。
- clean sample 直接复用原 normalized tensor，不做 round-trip 重建，保证 clean 数据路径不因 A2 helper 产生数值扰动。

### 3.3 batch 接口

A2 corruption helper 输出：

- `rgb`、`depth`：供 segmentation model 使用的 normalized tensor；
- `raw_rgb`、`raw_depth`：供 reliability estimator 使用的 `[0,1]` tensor；
- `reliability_target`：形状 `[B,2,H,W]`，通道顺序固定为 RGB、Depth；
- `valid_mask`：形状 `[B,1,H,W]` 的 bool 几何有效 mask，只排除 crop/pad 区，不因语义 Label 为 `255` 而排除真实图像区域；
- `depth_valid`：形状 `[B,1,H,W]`，由 corrupted raw Depth `>0` 与 `valid_mask` 共同得到；
- 结构化 metadata：至少记录 sample id、seed words、spec 顺序、severity 和 reliability 摘要，仅用于审计，不搬到 GPU 或进入 checkpoint 参数。

### 3.4 A2 v2 batch 接口与监督目标（v2 修订，新增）

v2 helper `utils/dataloader/mmfr_training_v2.py` 声明 `PROTOCOL_ID = "MMFR-A2-train-integration-v2"`、`SUPERSEDES = "MMFR-A2-train-integration-v1"`、`CORRUPTION_BASIS = "MMFR-A1-corruption-basis-v2"`。它只从 v1 helper 复用与协议语义无关的基础设施（sample-id hash、seed words 顺序、uint8 round-trip、curriculum position），并在 import 时断言 v1/v2 的六类 kind 与三阶段 severity 常量一致。

相对 v1，v2 只改三处：

1. **Depth 有效性拆成 corruption 前/后两个字段。** `depth_valid_pre` 是 corruption 之前的真实输入有效性 `raw_depth_pre > 0 AND valid_mask`；`depth_valid_post` 是交给模型的实际输入有效性 `raw_depth_post > 0 AND valid_mask`。v1 那个含义含混的单一 `depth_valid` 被刻意不再导出（列入 `V1_ONLY_KEYS`），使调用方无法静默读到错误字段。`models/builder.py` 与 `utils/train.py` 在 v2 下显式把 `depth_valid_post` 作为 reliability head 的输入有效性，pre-corruption 有效性则折进监督目标。
2. **Depth 监督目标尊重原生 MUSeg Depth 无效性。**

$$
R_D^{sup}(p) = depth\_valid\_pre(p)\cdot R_D^{syn}(p).
$$

其中 $R_D^{syn}$ 是 A1 v2 返回的合成 Depth reliability surrogate。clean 样本中原生有效 Depth 像素 target=1、原生 Depth=0 的无效像素 target=0；crop/pad 区仍为中性 1 并由 `valid_mask` 排除。RGB 通道保持 all-ones scaffold：A2 v2 只监督 Depth 通道，RGB 通道不计入 loss、不报告 calibration 指标，也不得称为已训练的 RGB reliability estimator。
3. **三类无效总体分开记录，从不合并。** metadata 记录 `natural_invalid_pixels`（corruption 前就无效）、`synthetic_missing_pixels`（corruption 前有效、被 corruption 置零）与 `post_corruption_invalid_pixels`（两者并集，断言等于前两者之和），另有 `implicit_quality_pixels`（保持非零但被 corruption 改变，`depth_valid` mask 看不到）。这三类不得混为一类。

helper 输出接口在 v1 的字段上把 `depth_valid` 替换为 `depth_valid_pre` 与 `depth_valid_post`，其余字段（`rgb`、`depth`、`raw_rgb`、`raw_depth`、`reliability_target`、`valid_mask`）保持同一语义，并附带 `protocol`、`supervised_channels`（固定 `["depth"]`）与 CPU-only metadata。`depth` 的通道数在 clean 与 corrupted 两条路径上必须一致。

## 4. 确定性 RNG 冻结

- 训练 corruption 基础 seed 固定为 `2026091402`，与模型 seed 分离。
- 每个样本建立独立 `numpy.random.Generator(PCG64(SeedSequence(words)))`，不读取或修改 Python、NumPy、PyTorch 的进程级随机状态。
- `words` 顺序固定为：`[corruption_seed, epoch, iteration, global_rank, sample_slot, sample_id_sha256_u32_0, sample_id_sha256_u32_1, sample_id_sha256_u32_2, sample_id_sha256_u32_3]`。
- `epoch` 为训练循环的一基编号，`iteration` 为 epoch 内零基编号，`global_rank` 使用分布式全局 rank，`sample_slot` 为本 rank batch 内零基位置；sample id 使用仓库相对路径统一为 `/` 后做 SHA-256，取前 16 bytes 按大端拆成四个 uint32。
- curriculum progress 使用当前尝试位置，而不是 wall clock：

$$
progress=\frac{(epoch-1)N_{iter}+iteration}{N_{epoch}N_{iter}-1},
$$

并截断到 `[0,1]`。相同输入身份和训练位置必须得到相同 spec、corruption、target 与 metadata。
- DataLoader 的 mirror/scale/crop 继续属于旧训练 RNG 语义；A2 只保证 corruption 不受 worker 调度和 worker 数量影响。若未来要求任意中途 resume 后连几何增强也逐像素相同，必须建立新协议，不能把本保证扩大解释。

## 5. corruption 分布与 loss 冻结

- 当前 A2/B1 路径只采样 `modality="depth"`；RGB corruption 保留在 A1 通用库中，但不进入本训练 config。RGB complete-missing 留给独立 B2（见第 12 节与 [`00-总方向规划.md`](00-总方向规划.md)）。
- 每个样本先以独立 RNG 采样 clean/corrupt：`p_clean=0.25`，`p_corrupt=0.75`。
- corrupt 样本调用 A1 v2 curriculum，`max_specs=2`，允许 `entire_missing`、`spatial_dropout`、`gaussian_noise`、`blur`、`quantization`、`misalignment`；A1 v2 已冻结的三阶段 severity 与完整缺失仅重度阶段出现的规则保持不变。v2 的 severity 只编码一次：severity 只选择 corruption 参数，graded kinds 的连续 burden 就是 realized normalized damage；`blur` 与 `misalignment` 使用相对尺度 severity（`BLUR_SIGMA_FRACTION=1/80`、`MISALIGN_MAX_SHIFT_FRACTION=1/30`）。这些失效是 model-input/representation-level synthetic corruptions，不是 Kinect 真实物理噪声模型。
- 每个样本只做一次 segmentation forward。clean control 使用 clean 输入；corruption model 使用采样后的 clean 或 corrupted 输入。
- 不做 clean teacher、不做 clean-corrupt consistency、不做蒸馏，固定 `lambda_cons=0`。
- corruption model 的总损失固定为：

$$
\mathcal L=\mathcal L_{seg}^{input}+0.1\mathcal L_{rel},
\qquad \mathcal L_{rel}=\operatorname{BCE}\!\left(\hat R_d,\;R_D^{sup}\right).
$$

其中 `reliability_target` 的 Depth 通道为 A1 v2 连续 target 与 pre-corruption Depth 有效性的乘积 $R_D^{sup}(p)=depth\_valid\_pre(p)\cdot R_D^{syn}(p)$，`valid_mask` 只排除 pad。crop/pad 的 target 保持中性 1 并由 `valid_mask` 排除；RGB target 在本 Depth-only protocol 中始终为 1，且不计入 loss、不报告 calibration 指标。
- reliability estimator 只产生辅助 loss，不把预测 reliability 送入 DFormerv2；`GeometryContributionAdapter` 在 A2 保持未接入。这样 A2 隔离“训练分布与 target 是否接对”，B1 才单独检验 learned adapter。

### 5.1 冻结项不变（不得在冻结身份内更改）

以下项在 v2 修订中保持不变，必须在正式训练前逐项复核一致：`p_clean=0.25` / `p_corrupt=0.75`、三阶段 curriculum、`max_specs=2`、`lambda_rel=0.1`、500 epoch、optimizer（AdamW）、learning rate、batch size、clean-only checkpoint selector、六个单失效主 score、clean drop ≤ `0.50` 个百分点、paired location-group bootstrap、主成功门槛，以及云端单卡与 4090/5090 规则。

明确写出：**不因新论文采用 50% missing 等设置而照抄**；KSTrack 的 50% modal-incomplete 属 tracking 场景，不能推出 MUSeg 最优 corruption ratio；若未来发现 exposure 不合适，只能新开 protocol，不能在当前冻结身份内调整 `p_clean`/`p_corrupt`、curriculum 或 `max_specs`。

## 6. config、模型与 checkpoint 边界

- 新建独立 common、clean control 和 depth-corruption config；不得覆盖 Quick-B0 config。v2 的 config 为 `local_configs/MUSeg/DFormerv2_S_MMFR_A2_Common_v2.py`、`DFormerv2_S_MMFR_A2_Clean_v2.py`、`DFormerv2_S_MMFR_A2_DepthCorrupt_v2.py`（v1 的同名 config 冻结保留）。
- `DFormerv2_S_MMFR_A2_Common_v2.py` 声明 `MMFR_A2_PROTOCOL = "MMFR-A2-train-integration-v2"`、`MMFR_A2_SUPERSEDES = "MMFR-A2-train-integration-v1"`、schedule version `mmfr-a2-v2-adamw-6e-5-warmup10-poly0.9-500e-v1`，两个子 config 通过 `configure_mmfr_identity_v2(...)` 绑定各自 run id 与 analysis identity；它把不影响科学问题的数据职责、pretrained、seed、optimizer、LR、500 epoch、warmup、batch size、workers、尺度增强、clean selector 与云端 profile 原样继承 v1，以保持公平对与跨协议可比。
- 冻结常量（与实际 config 一致）：`p_clean=0.25`、`p_corrupt=0.75`、`max_specs=2`、corruption seed `2026091402`、`blur_sigma_fraction_of_min_side_at_severity_1 = 1/80`、`misalignment_max_shift_fraction_of_axis_at_severity_1 = 1/30`、`noise_sigma_uint8_at_severity_1 = 48.0`、`quantization_min_levels_at_severity_1 = 2`、`lambda_reliability = 0.1`、`lambda_consistency = 0.0`，`supervised_channels = ["depth"]`。
- 只有 corruption config 实例化 `ModalReliabilityEstimator(hidden_channels=16)`；clean control 不增加该 head。
- 官方 pretrained 只加载 backbone，新增 reliability head 使用其模块默认初始化；训练 checkpoint 必须 strict 保存/恢复完整模型与 optimizer state。
- `models/builder.py` 与 `utils/train.py` 增加 v1/v2 双协议守卫：`mmfr_a2.protocol` 只允许 `MMFR-A2-train-integration-v1` 或 `MMFR-A2-train-integration-v2`，未知身份报错；v2 额外强制 corruption basis 为 `MMFR-A1-corruption-basis-v2`、`severity_encoding == "single"`、target composition 为 `depth_valid_pre * R_depth_synthetic`、`supervised_channels == ["depth"]`、`relative_scale_spatial_corruptions == True`。把 v2 config 喂给 v1 helper（或反向）是硬错误。builder 只对 `supervised_channels` 指定的通道计分，缺字段或错配时 fail-closed。
- A2 不修改 `models/encoders/DFormerv2.py`、DVG-B1 Oracle 参数链、decoder、evaluator 或 existing checkpoint key。
- A2 代码必须在配置关闭时保持现有 `model(imgs, modal_xs, gts)` 路径和 scalar loss 行为不变。

## 7. 冻结的开发评价设计

### 7.1 evaluator 与条件

- checkpoint selection：clean `val-dev`、`original-full`、scale `1.0`、无 flip。
- 最终开发评价：冻结 checkpoint 后使用 `msflip-whole-original-grid-v1`，即 5 scale × 原图/水平翻转共 10 view，回到 MUSeg 原始 Label 网格后平均 FP32 pre-softmax logits。
- evaluation corruption 在原始对齐 RGB/Depth 上一次生成，再由 10-view evaluator 做共同几何变换；不得为每个 view 重新采样。
- evaluation seed 固定 `2026091401`，按 sample id 与 condition index 生成，不使用训练位置。
- 主单失效条件固定为 Depth：`spatial_dropout@0.75`、`gaussian_noise@0.75`、`blur@0.75`、`quantization@0.75`、`misalignment@0.75`、`entire_missing@1.0`。
- 次要混合条件固定为顺序组合：`spatial_dropout@0.5 + gaussian_noise@0.5`、`blur@0.5 + misalignment@0.5`、`quantization@0.5 + misalignment@0.5`。
- 同时报告 clean、每个单失效、每个混合条件的 mIoU 与 Boundary IoU。主 robustness score 是六个单失效 mIoU 的等权宏平均；混合条件不进入主 score，避免结果后调整权重。

### 7.2 统计单位与成功门槛

- 比较为 corruption model minus clean control；全体 318 张 `val-dev` 均保留，以 196 个 location group 为 paired bootstrap 单位。
- 主成功要求同时满足：
  1. 六单失效宏平均 mIoU 点估计至少 `+1.00` 个百分点；
  2. 该宏平均 paired 95% percentile interval 下界严格大于 `0`；
  3. clean mIoU 点估计下降不超过 `0.50` 个百分点；
  4. 六个单失效中至少五个 mIoU 点估计不为负，且任一单条件不得低于 `-1.00` 个百分点。
- Boundary IoU、三个混合条件和 clean Boundary IoU 为预注册辅助结果，不作为把主失败改写为成功的替代门槛。
- 因 checkpoint 选择和最终失效评价都使用 `val-dev`，即使达到门槛也只能写成 single-seed development-supported，不能写成独立测试或真实部署结论。

### 7.3 结果前冻结的补充协议身份（只引用，文件已存在且不得改写）

以下三个补充协议在结果产生前冻结，只扩展报告口径，不能替代或挽救主门槛。它们的实现/执行仍需用户单独授权；本节只引用其身份与关键约束。

- **`protocols/mmfr-r1-reliability-supplemental-v1.template.json`（`MMFR-R1-reliability-supplemental-v1`）：** 三层 reliability 专门评价。第一层是 Depth surrogate target fidelity（MAE、Brier），第二层是 reliability 与真实 segmentation risk 的关系（reliability-error 单调性、risk-coverage、AURC），第三层是下游 gain，只能等 B1 之后用 segmentation 指标回答。`explicit-invalid` 与 `implicit-quality` 两组必须分开报告：前一组可由 `depth_valid` 轻易识别，不能用它掩盖后者（`implicit-quality` 像素保持非零，必须靠学到的质量证据识别）。第一、二层不得替代主 robustness score；`inherited_at_lock` 字段需在 A2 身份锁定时绑定。
- **`protocols/mmfr-s1-severity-sweep-v1.template.json`（`MMFR-S1-severity-sweep-v1`）：** severity `0.25/0.50/0.75` × 五类非完全缺失 corruption（`spatial_dropout`、`gaussian_noise`、`blur`、`quantization`、`misalignment`），另加 `entire_missing@1.0` 与 `clean`；报告每类 mIoU-vs-severity 曲线与五类宏平均 robustness-AUC。明确：不得用于 checkpoint selection，不得在主门槛失败后重新定义成功，severity 只编码一次。
- **`protocols/mmfr-c1-paper-confirmation-v1.template.json`（`MMFR-C1-paper-confirmation-v1`）：** 论文/SOTA 确认阶段，官方 `1595` train 与 `1576` test、500 epoch 或预登记固定 checkpoint 规则、3 个 paired seeds，最后**一次性**在 sealed official test 上运行（单次读取、禁止测试时选 epoch/condition/method）。必须与 `1277 train-dev + 318 val-dev + single seed` 的 development screening 严格区分：后者即使达到门槛，最高声明仍只是 `single-seed development-supported`。

## 8. 限定代码执行单

### 8.1 允许修改

- 新增 `utils/dataloader/mmfr_training.py`（v1）；
- 修改 `models/builder.py`，仅增加可选 reliability head 和辅助 loss；
- 修改 `utils/train.py`，仅在 A2 config 开启时调用 batch helper 并传入可选字段；
- 新增 `local_configs/MUSeg/DFormerv2_S_MMFR_A2_Common.py`、`DFormerv2_S_MMFR_A2_Clean.py`、`DFormerv2_S_MMFR_A2_DepthCorrupt.py`（v1）。

v2 修订另允许（且只允许）以下新增，v1 文件与 config 保持冻结：

- 新增 `utils/dataloader/multimodal_failure_v2.py`（A1 v2 corruption basis `MMFR-A1-corruption-basis-v2`）；
- 新增 `utils/dataloader/mmfr_training_v2.py`（A2 v2 batch helper）；
- 新增 `local_configs/MUSeg/DFormerv2_S_MMFR_A2_Common_v2.py`、`DFormerv2_S_MMFR_A2_Clean_v2.py`、`DFormerv2_S_MMFR_A2_DepthCorrupt_v2.py`；
- 在 `models/builder.py`、`utils/train.py` 中增加 v1/v2 双协议守卫与 Depth-only 监督通道选择，不改动 v1 路径语义；
- 新增只读审计工具 `tools/mmfr/severity_burden_audit.py`（已运行）与 `tools/mmfr/gradient_path_isolation.py`（已写好并已于 2026-09-15 执行、通过）。

### 8.2 禁止修改

不得修改 `TrainPre`、`RGBXDataset`、`models/encoders/DFormerv2.py`、A1 corruption 数值语义、DVG-B1 文件、evaluator、checkpoint loader、official-test 文件或 MUSeg 关键状态文档；不得新增依赖、测试文件或临时脚本。

### 8.3 验收与停止条件

- config 关闭时旧训练调用和 loss 接口不变；
- exhaustive uint8 round-trip、同 seed 确定性、不同 sample slot 分离、clean exact normalized no-op、pad exact-zero 保持、target/valid/depth-valid shape/range 通过 CPU 定点检查；
- builder 的可选 head 在 CPU 小 tensor 上得到 finite scalar loss 和 finite backward；
- v2 额外验收：v1/v2 双协议守卫在错配时拒绝；`depth_valid_pre` 与 `depth_valid_post` 均为 `valid_mask` 子集；Depth 监督 target 在原生无效 Depth 处严格为 0、在 crop/pad 区为中性 1 且被 `valid_mask` 排除；RGB scaffold 通道严格恒为 1；三类无效总体满足 `post_corruption_invalid_pixels == natural_invalid_pixels + synthetic_missing_pixels`；clean 路径 `torch.equal` no-op；Depth 通道数在 clean/corrupt 两条路径一致；
- 只运行受影响 Python 文件静态编译、内联 CPU probe、静态诊断，以及专门 CPU 审计 `python tools/mmfr/severity_burden_audit.py`。
- 任一接口需要修改 DFormerv2、evaluator、checkpoint 语义或 A1 数值定义时立即停止并报告 `scope-blocked`。

## 9. 未授权操作与后继门禁

**v2 首要门禁（已执行并通过）：** `tools/mmfr/gradient_path_isolation.py` 已于 2026-09-15 01:32（本地时间）在本地 RTX 5060 Laptop GPU 上实际执行，退出码 `0`、墙钟 `15.3` 秒、结论 `gradient-path-isolation: PASS`，`failures` 与 `warnings` 均为空列表。它实现的检查是：同一模型状态/输入/corruption/RNG/segmentation loss 下，`lambda_rel=0` 与 `lambda_rel=0.1` 两次 backward 的共享参数（backbone/decoder/geometry prior，规则为所有非 `reliability_estimator.*` 参数）梯度必须在容差内相同，reliability head 参数只在后者有有效梯度，head 开/关时 segmentation logits 一致。实测结论是这三项全部成立：共享参数 `714` 个（backbone `672`、decoder `13`、geometry prior `29`）在 `ATOL=1e-6`、`RTOL=1e-6` 下 `over_tolerance_count=0`、`absent_in_both_runs_count=0`，且全部逐位相同（`max_abs_difference=0.0`、`max_relative_difference=0.0`、`bitwise_difference_count=0`）；reliability head `6` 个参数在 `lambda_rel=0` 时梯度张量存在且恒为零（梯度范数 `0.0`），在 `lambda_rel=0.1` 时存在且非零（梯度范数 `0.05855773380379924`、最大绝对值 `0.04926614463329315`）；`lambda_rel=0` 与 `lambda_rel=0.1` 两次的 segmentation loss 逐位相同（`2.545051336288452`），reliability head 开/关两种 forward 的 logits 与 `lambda_rel=0` 的 logits 逐位相同（两组比较的 `max_abs_difference=0.0`）。输入为真实 `train-dev` 的 `sample_index=0`、`iteration=0`（无需向后扫描），抽到 corrupted 样本（`clean=False`），`1` 个 spec、kind `quantization`（`levels=195`、`step_uint8=1.3144329896907216`，severity 约 `0.24`），Depth 通道监督，分割有效像素 `11932`。证据为 `outputs/mmfr-a2-gradient-path-isolation/gradient-path-isolation.json`，SHA-256 `dd0e351b418f0228650830c70cf0750c21a446502d3a9c11ebc047581aedb763`；同目录 `run-stdout.txt` 记录 stdout。该 JSON 不记录 checkpoint，也不读取 official test。门禁状态记为 `executed-and-passed`：**“通过之前禁止启动正式 A2 v2 训练”这一前置条件已经满足，该检查不再构成阻塞；正式训练本身仍未授权**（完整证据、边界与冻结结论见第 13.2 节）。

本地 GPU 单步 preflight 已由用户单独授权并完成（v1 协议身份）。正式训练固定为云端单 GPU，本机只做推理、想法初步验证和小规模 preflight；默认租用 RTX 4090 24GB，RTX 5090 32GB 只在 4090 缺货、24GB 无法容纳冻结 batch size 10，或同配置配对短 probe 证明单位样本成本更低时使用。无论换卡与否，batch size `10`、workers `8`、AdamW、learning rate `6e-5`、500 epoch、warmup 10 epoch、AMP、SyncBN、首轮关闭 `torch.compile`、seed `772961337` 与 selector 都保持不变。

当前仍不运行正式训练、完整 epoch、checkpoint save/load、冻结 batch size 10 云端容量/吞吐 probe、DDP、完整评价、云资源或 official test。开始 probe、clean control 或 Depth-corruption 任一正式训练，仍需用户单独授权；B1 learned geometry adapter 必须另建独立代码计划，不能因 A2 preflight 通过而顺带接入。

**大白话说明：** 本协议把损坏动作放在最终裁剪批次上，用训练位置和样本身份生成独立随机数；模型每次只看一个输入，可靠性头先作为辅助监督，不直接改 GSA。GPU 预检现在证明这条链能完成一次真实更新，但还没有开始长程训练或比较模型效果。v2 修订后的 `gradient_path_isolation` 门禁已经在本地 GPU 上实测通过（第 13.2 节），所以“先证明可靠性头不会把梯度漏回主分割网络”这道数学前提已经满足、不再阻塞正式训练；但正式训练本身仍未授权，也没有开始长程训练或比较模型效果。

## 10. 实际完成与资格检查

以下条目是 A2 v1（`MMFR-A2-train-integration-v1`）的原始记录，原样保留；v2 修订见第 12、13 节。

- 已新增 `utils/dataloader/mmfr_training.py`：在最终 CPU crop batch 上恢复严格 uint8/raw 信号，构造 per-sample stateless NumPy RNG，只采样 Depth curriculum，并输出 normalized input、raw input、连续 target、pad-valid mask、Depth validity 和审计 metadata。
- 已修改 `models/builder.py`：只有 corruption config 创建 `ModalReliabilityEstimator(hidden_channels=16)`；训练时强制完整 reliability supervision，计算 segmentation loss 加 `0.1` 倍连续 BCE。A2 只运行固定 signal features 与 reliability head，不计算未使用的四级 pyramid，也不把预测 reliability 送入 backbone、decoder 或 geometry prior。
- 已修改 `utils/train.py`：仅在 A2 corruption config 开启时，于 GPU 搬运前调用 batch helper；配置关闭时继续走原 `model(imgs, modal_xs, gts)` 标量 loss 路径。逐样本 metadata 留在 CPU，只汇总 clean/corrupt 数量和有效区 Depth reliability 均值。
- 已新增 common、clean control、Depth corruption 三个独立 MUSeg config；两身份共享 official pretrained、seed、优化器、500 epoch、尺度增强和 clean selector，不覆盖 Quick-B0 config。
- 主代理 CPU 复核后修正三个边界问题：clean/受损路径 Depth 通道数必须一致；Depth-only 候选必须在抽取 spec 数量前限制，不能先抽 RGB+Depth 再过滤；normalized pad 逆变换得到的均值字节必须在 corruption 前置零，防止 misalignment 把伪 Depth `122` 移入有效区域。builder 同时改为缺少或错配 reliability 字段时 fail-closed。
- CPU qualification：受影响 Python 文件 `py_compile` 通过；聚焦 CPU probe 通过 exhaustive uint8 round-trip、三阶段 Depth-only curriculum、同位置确定性、sample slot 分离、clean `torch.equal` no-op、pad/raw/target 中性语义、misalignment 不引入伪 `122`、完整/缺失 supervision 守卫、finite scalar loss 与 finite backward；六个受影响 Python 文件静态诊断无问题，`git diff --check` 通过。

## 11. 本地 GPU 单步 preflight

以下条目绑定 v1 协议身份，属历史事实，原样保留（数字与哈希不得改写）。

- 首次真实 AMP forward 在 `SignalFeatureExtractor` 中触发非有限值。根因是梯度平方和的乘积在 FP16 下溢为 0，余弦对齐分母随之为 0；这在 CPU FP32 小张量资格检查中不会出现。修复后 reliability 固定特征、小型 head 与连续 BCE 在 FP32 中计算，segmentation backbone 和 Ham decoder 继续使用 AMP，损失定义及 reliability 不进入 backbone/geometry prior 的协议保持不变。
- 第二次运行已完成真实 forward/backward/AdamW update，但 probe 在退出阶段错误 finalize checkpoint selector，并因短运行按设计未生成 `latest.pth` 而失败。修复后 `run_kind=probe` 不创建或 finalize selector，仍写出 `run_config.json`、逐步 telemetry 和 `training_result.json`；正式训练的 selector/checkpoint 路径不变。
- 最终 canonical 运行绑定 protocol `MMFR-A2-train-integration-v1`、schedule `mmfr-a2-adamw-6e-5-warmup10-poly0.9-500e-v1` 和物化 `protocol.json` SHA-256 `e9825c4a4818cf2860a529b6c0fa542f5412a7153a1e8b9b8c8d000de6ca7f07`。preflight 状态回填后的 repository template SHA-256 为 `b5b5c979352cca451cfbca35abc77232a9ae82c3f22ec0f33fcdb964ae03e519`。环境为 Python `3.10.20`、PyTorch `2.7.0+cu128`、CUDA `12.8`、cuDNN `90701`、driver `610.88` 和 NVIDIA GeForce RTX 5060 Laptop GPU；官方 pretrained SHA-256 为 `19116988fc86dc9f3e879282237941e11b9b1b5c480edb51e92807311dbc11a6`。
- 运行使用真实 `train-dev`、`480×640`、batch size 1、worker 0、SyncBN、AMP、Ham decoder、Depth corruption 与 reliability auxiliary loss。初始 GradScaler 连续跳过 5 次 update，并将 scale 从首个已记录的 `32768` 降到 `2048`；第 6 个 batch 完成第 1 次 optimizer update。6 个 loss 均有限，累计包含 1 个 clean 与 5 个 corrupted sample，最终更新步 loss 为 `3.34708309173584`。
- 峰值 allocated/reserved CUDA memory 为 `2069.91/2260` MiB，最终可用显存 `4737/8150.56` MiB、free ratio `0.5812`，安全检查通过。`training_result.json` 记录 `attempted_steps=6`、`completed_optimizer_steps=1`、`skipped_optimizer_steps=5`、telemetry invariant 成立、exit code `0`，并明确 `checkpoint=null`、`official_test_included=false`。
- canonical 证据位于仓库内忽略目录 `outputs/mmfr-a2-gpu-preflight-20260914T0138Z/`：`protocol.json` SHA-256 `e9825c4a4818cf2860a529b6c0fa542f5412a7153a1e8b9b8c8d000de6ca7f07`，`run_config.json` SHA-256 `2ff6bf4668efbb58012fced89af698ce4cfec5181c43a4ee4dec9e56dce4edd3`，`training_result.json` SHA-256 `bb3d71dacc99d7b1b97eb03d702f0eff621f78669b7dbb5824aaafad832550c0`，`probe-telemetry.jsonl` SHA-256 `6ae56d55552aee9cacda71b09fea7269069814e18d81e1d3548539d0995d462c`。run config 记录当前未提交工作区 `dirty=true`。
- preflight 未覆盖冻结 batch size 10 的容量、完整 epoch、checkpoint save/load、DDP、`torch.compile`、evaluator、完整测试、云任务或 official test；这些项目不得写成通过。

## 12. A2 v2 修订（协议 supersede 与科学语义）

### 12.1 身份与修订原因

- **协议身份：** `MMFR-A2-train-integration-v2`，supersede `MMFR-A2-train-integration-v1`。
- **修订原因（结果前修订）：** “正式 500 epoch 尚未执行，所以在结果产生之前进行协议修订”。具体是 v1 的 severity 双编码、分辨率相关的 `blur`/`misalignment` severity、对原生无效 Depth 未限定的 reliability target，以及被监督却未定义的 RGB reliability 通道。
- **两个分析身份：** `MMFR-A2-clean-control-v2`、`MMFR-A2-depth-corruption-train-v2`。
- **两个 run id（以实际 config 为准）：** `museg-dformerv2-s-mmfr-a2-clean-control-v2`、`museg-dformerv2-s-mmfr-a2-depth-corruption-v2`。
- **schedule version：** `mmfr-a2-v2-adamw-6e-5-warmup10-poly0.9-500e-v1`。
- **v1 冻结：** v1 模块、helper、config 与 protocol template 保持冻结、未被改写；v2 只在协议语义无关处复用 v1 的基础设施与常量，并在 import 时断言一致。

### 12.2 代码身份

- A1 v2 corruption basis：`utils/dataloader/multimodal_failure_v2.py`（`MMFR-A1-corruption-basis-v2`）。
- A2 v2 batch helper：`utils/dataloader/mmfr_training_v2.py`。
- v2 config：`local_configs/MUSeg/DFormerv2_S_MMFR_A2_Common_v2.py`、`DFormerv2_S_MMFR_A2_Clean_v2.py`、`DFormerv2_S_MMFR_A2_DepthCorrupt_v2.py`。
- `models/builder.py` 与 `utils/train.py` 增加 v1/v2 双协议守卫与 Depth-only 监督通道选择。

### 12.3 科学语义修订（逐条）

1. **severity 只编码一次。** `gaussian_noise`/`blur`/`quantization` 的连续 burden 就是 realized normalized damage $\operatorname{clip}(\max_{\mathrm{ch}}|\Delta_{\mathrm{ch}}|/(255\cdot \mathrm{DAMAGE\_REFERENCE}),0,1)$，`DAMAGE_REFERENCE=0.25`，不再二次乘 severity；`misalignment` 有效区 burden 为实际位移除以当前尺度允许的最大位移（`MISALIGN_MAX_SHIFT_FRACTION=1/30`，按轴换算像素），越界像素用 `MISSING_BURDEN=1e4`。
2. **`blur` severity 改为相对尺度量。** $\sigma = \text{severity}\times \mathrm{BLUR\_SIGMA\_FRACTION}\times\min(H,W)$，`BLUR_SIGMA_FRACTION=1/80`，等价于 v1 在冻结 `480` 像素 crop 上的 `6.0` px 上限。
3. **Depth reliability 监督目标改变。** $R_D^{sup}(p)=depth\_valid\_pre(p)\cdot R_D^{syn}(p)$，`depth_valid_pre = raw_depth_pre > 0 AND valid_mask`；corruption 后的实际输入有效性记为 `depth_valid_post`。clean 样本中原生有效 Depth 像素 target=1、原生 Depth=0 的无效像素 target=0；crop/pad 区仍为中性 1 并由 `valid_mask` 排除。metadata 分开记录 `natural_invalid_pixels`、`synthetic_missing_pixels`、`post_corruption_invalid_pixels`（另有 `implicit_quality_pixels`），三者不得混为一类。
4. **A2 v2 只监督 Depth 通道。** RGB 通道保留为两通道接口的 all-ones scaffold，不计入 loss、不报告 calibration 指标、不得称为已训练的 RGB reliability estimator；B1-v1 同样只能消费 Depth reliability。
5. **冻结项不变。** `p_clean=0.25` / `p_corrupt=0.75`、三阶段 curriculum、`max_specs=2`、`lambda_rel=0.1`、500 epoch、optimizer、LR、batch size、clean-only selector、六单失效主 score、clean drop ≤ `0.50`、paired location-group bootstrap、主成功门槛、云端单卡与 4090/5090 规则。明确：不因新论文采用 50% missing 等设置而照抄；KSTrack 的 50% modal-incomplete 属 tracking 场景，不能推出 MUSeg 最优 corruption ratio；若未来发现 exposure 不合适只能新开 protocol。
6. **阶段科学含义重写。** A1 = “synthetic failure generation + corruption-derived reliability surrogate”；A2 = “Depth corruption robustness training + parallel reliability estimation”；A2 的 predicted reliability 不影响 segmentation backbone/decoder/geometry prior，因此 A2 的 segmentation gain 不能归因于 reliability head；B1 = “第一个真正的 reliability-aware segmentation 阶段”，即首次让预测 reliability 影响 DFormerv2 的模型动作。删除“辅助 target 促进了 segmentation robustness”一类因果表述。
7. **DFormerv2 架构描述统一。** 写作 **RGB-primary semantic path + Depth geometry-prior path 的非对称架构**：Depth 作为 geometry prior 进入 GSA，而非与 RGB 对称的显式语义编码；不得写成普通 RGB/Depth 双语义分支。
8. **corruption 定性统一。** A1/A2 v2 的 corruption 是 **model-input/representation-level synthetic corruptions**，不得称为 Kinect 真实物理噪声模型。

### 12.4 B1 与 B2 边界（不属于 A2 执行范围）

- B1 的最小 baseline 记为 `B1a-global-reliability-baseline`（`mean/std reliability -> global scale`），不得作为 B1 最终默认方案；B1 冻结设计原则见 [`00-总方向规划.md`](00-总方向规划.md) 第 7 节。
- DVG-B1 Oracle 负结果的正确解释保留：“知道哪里坏”不等于“知道模型该采取什么动作”。
- B2 与 B1 严格分离，扩写为 `B2a explicit Depth semantic fallback`（默认首先研究，对当前 DFormerv2 改动最可控）、`B2b feature reconstruction/compensation`（参考 KSTrack 的 prototype + feature reconstruction）、`B2c modality-agnostic representation/pretraining`（参考 Yu 的 modality-asymmetric reconstruction）；不把 CMNeXt 当作解决 RGB-complete-missing 的直接模板。

## 13. A1 v2 severity/burden 审计与 `gradient_path_isolation` 门禁

### 13.1 已完成的 CPU 审计（可引用，不夸大）

- **脚本：** `tools/mmfr/severity_burden_audit.py`，纯 CPU、只读，审计链条 `severity -> corruption parameter -> realized damage -> burden -> Depth target`。
- **运行结果：** `python tools/mmfr/severity_burden_audit.py` 退出码 `0`，`OVERALL: PASS`，`violations=0`。
- **通过的具体检查（全部为真）：** `no_double_severity_encoding`、`monotonic_burden`、`monotonic_target`、`relative_scale_consistent`、`dropout_extent_monotone`、`empty_spec_strict_noop`。
- **报告：** `outputs/mmfr-a1-v2-severity-audit/severity-burden-audit.json`，SHA-256 `86e468c040648a790675456db1b828a5a71e560b849c75b4b7e9a3ca926c2f52`（上级代理已复跑并得到同一哈希）。
- **边界：** 审计使用两个确定性合成 fixture（`480×640` 与 `932×1082`），不是真实样本；它证明 v2 编码与数值语义在 CPU 上自洽，不证明真实样本行为、GPU、训练或模型效果。

### 13.2 已执行的 `gradient_path_isolation` 门禁（已通过）

- **脚本与状态：** `tools/mmfr/gradient_path_isolation.py` 已于 2026-09-15 01:32（本地时间）在本地 RTX 5060 Laptop GPU 上**实际执行**，退出码 `0`、墙钟 `15.3` 秒、结论 `gradient-path-isolation: PASS`，`failures` 与 `warnings` 均为空列表。门禁状态记为 `executed-and-passed`。
- **检查内容与三项实测结论：** 同一模型状态/输入/corruption/RNG/segmentation loss 下，`lambda_rel=0` 与 `lambda_rel=0.1` 两次 backward 的共享参数（backbone/decoder/geometry prior，规则为所有非 `reliability_estimator.*` 参数）梯度必须在容差内相同；reliability head 参数只在后者有有效梯度；head 开/关时 segmentation logits 一致。三项全部成立。
- **共享参数（`714` 个：backbone `672`、decoder `13`、geometry prior `29`）：** `over_tolerance_count=0`、`absent_in_both_runs_count=0`，且全部**逐位相同**（`max_abs_difference=0.0`、`max_relative_difference=0.0`、`bitwise_difference_count=0`）；容差为 `ATOL=1e-6`、`RTOL=1e-6`。
- **reliability head（`6` 个参数）：** `lambda_rel=0` 时梯度张量存在且恒为零（梯度范数 `0.0`）；`lambda_rel=0.1` 时存在且非零（梯度范数 `0.05855773380379924`，最大绝对值 `0.04926614463329315`）。
- **segmentation 一致性：** `lambda_rel=0` 与 `lambda_rel=0.1` 两次的 segmentation loss 逐位相同（`2.545051336288452`）；reliability head 开/关两种 forward 的 logits 与 `lambda_rel=0` 的 logits 逐位相同（两组比较的 `max_abs_difference=0.0`）。
- **输入与 corruption：** 真实 `train-dev` 的 `sample_index=0`、`iteration=0`（无需向后扫描）；抽到 **corrupted** 样本（`clean=False`），`1` 个 spec，kind 为 `quantization`（`levels=195`、`step_uint8=1.3144329896907216`，即 severity 约 `0.24`），Depth 通道监督，分割有效像素 `11932`。
- **证据：** `outputs/mmfr-a2-gradient-path-isolation/gradient-path-isolation.json`，SHA-256 `dd0e351b418f0228650830c70cf0750c21a446502d3a9c11ebc047581aedb763`；stdout 记录为同目录 `run-stdout.txt`。该 JSON 不记录 checkpoint，也不读取 official test。
- **同时核到的官方 pretrained 加载事实（只记录，不作延伸推论）：** checkpoint key `785` 个、backbone key `780` 个、重叠 `774` 个，重叠张量不匹配数为 `0`；checkpoint 缺少 `extra_norms.0/1/2.{weight,bias}` 共 `6` 个 key（即模型有而 checkpoint 没有的 key），checkpoint 另有 `proj.*`、`norm.*`、`head.*`、`aux_head.*` 等模型未使用的 key。因此 `extra_norms` 的那 `6` 个张量不由该 checkpoint 提供，这一点只作为已核到的加载事实记录，不延伸出任何性能或因果推论。
- **证据边界（必须与结论同时引用）：** 该检查是 batch size `1`、FP32（关闭 autocast 与 TF32）、单进程（无 process group 时 SyncBN 回退为逐进程 BN）、一个真实样本、一次 corrupted 抽样的定点检查；它**不覆盖**冻结 batch size `10`、AMP、DDP、多样本、完整 epoch、checkpoint save/load、evaluator、云资源或 official test，也**不提供任何 mIoU 或鲁棒性收益结论**。
- **通过后冻结的结论：** A2 分割性能变化的直接原因是 corruption exposure，reliability head 在该阶段只承担 estimator qualification；因此本项目不需要再额外消耗一次完整 500 epoch 的“corruption-only”训练来证明这条数学事实。
- **门禁规则（已满足）：** 原约束“**在该检查通过之前，禁止启动正式 A2 v2 训练**”作为已满足的前置条件保留；该检查不再构成阻塞，但正式训练仍未授权。

**大白话说明：** 这次实测确认可靠性头的辅助损失不会通过梯度沿任何共享参数泄漏回分割主路径，也不会改变前向 logits 和 segmentation loss，所以 A2 里“分割变化只能来自 corruption exposure”这个前提是被验证过的、不用再单训一个 500 epoch 的 corruption-only 模型去证明。但这是 batch size 1、FP32、单进程、一个样本的定点检查，既不代表训练已完成，也不代表模型在失效下变得更好。

**完成边界：** `MMFR-A2-train-integration-v1` 是 code-qualified 且 GPU-single-step-qualified（历史事实与哈希原样保留，见第 10、11 节）。`MMFR-A2-train-integration-v2` 目前是 code-qualified、通过 CPU severity/burden 审计，且 `gradient_path_isolation` 门禁已执行并通过（`executed-and-passed`，证据与边界见第 13.2 节）；它仍未开始 500 epoch 训练、没有训练 checkpoint 或模型指标，也不因此产生任何 mIoU 或鲁棒性收益结论。正式训练资源已冻结为云端单卡、默认 RTX 4090；门禁不再阻塞，准确恢复点是在用户单独授权后依次：（1）可选地先补一次 v2 本地 GPU 单步 preflight，覆盖真实模型 AMP 路径（v2 目前尚无 GPU preflight）；（2）云端 batch size `10` 容量/吞吐短 probe；（3）`MMFR-A2-clean-control-v2` 与 `MMFR-A2-depth-corruption-train-v2` 两个公平对照训练。B1/B2/C1 与 R1/S1 仍各自需要独立授权。
