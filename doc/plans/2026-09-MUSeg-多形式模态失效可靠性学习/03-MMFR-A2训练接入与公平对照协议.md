# MMFR-A2 训练接入与公平对照协议

> **文档角色：** 条件式子计划与实现执行单。
> **计划状态：** 设计冻结、限定训练接入、CPU qualification 与本地 GPU 单步 preflight 已完成；正式训练资源已固定为云端单 GPU、默认 RTX 4090，本机仅做推理与小规模验证。云端 probe、正式训练、checkpoint 评价和 official test 未授权。
> **形成或核验时点：** 2026-09-14 02:02 UTC。
> **实时入口：** [`MUSeg-current-status.md`](../../main/MUSeg-current-status.md)。
> **研究选择：** [`MUSeg-open-decisions.md`](../../main/MUSeg-open-decisions.md)。
> **上级方向：** [`00-总方向规划.md`](00-总方向规划.md)。
> **当前/后继关系：** 前序 [`02-MMFR-A1失效基函数与可靠性脚手架.md`](02-MMFR-A1失效基函数与可靠性脚手架.md) 已完成；本阶段已接入 Depth 合成失效和辅助可靠性监督，并通过真实本地 GPU 单步 preflight，但未接入 geometry adapter。正式训练固定为云端单卡并默认使用 RTX 4090，本机只做推理与小规模验证。当前恢复点是保持云资源与训练未授权，由用户单独授权 batch size 10 容量/吞吐 probe 及 clean control/Depth-corruption 公平对照；`MMFR-B1-learned-geometry-adapter-v1` 尚未细化或授权。

## 1. 实验族身份与本阶段问题

- **协议身份：** `MMFR-A2-train-integration-v1`。
- **clean control：** `MMFR-A2-clean-control-v1`，沿用原 DFormerv2-S 结构，只使用 clean 训练输入。
- **corruption model：** `MMFR-A2-depth-corruption-train-v1`，使用 Depth 合成失效、corrupted segmentation loss 和辅助 reliability loss。
- **本阶段问题：** 在不改 DFormerv2 geometry prior、不做 clean/corrupt 双前向的前提下，把 A1 的 Depth corruption 与连续 target 接进现有训练链，并冻结可复现、公平、可审计的输入和损失语义。
- **不主张：** A2 接入通过不代表模型已获得鲁棒性；当前结构仍是 RGB-centric，不能把结果外推到 RGB complete-missing，也不能外推到真实矿井传感器故障率。

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

- 当前 A2/B1 路径只采样 `modality="depth"`；RGB corruption 保留在 A1 通用库中，但不进入本训练 config。RGB complete-missing 留给独立 B2 双语义路径。
- 每个样本先以独立 RNG 采样 clean/corrupt：`p_clean=0.25`，`p_corrupt=0.75`。
- corrupt 样本调用 A1 curriculum，`max_specs=2`，允许 `entire_missing`、`spatial_dropout`、`gaussian_noise`、`blur`、`quantization`、`misalignment`；A1 已冻结的三阶段 severity 与完整缺失仅重度阶段出现的规则保持不变。
- 每个样本只做一次 segmentation forward。clean control 使用 clean 输入；corruption model 使用采样后的 clean 或 corrupted 输入。
- 不做 clean teacher、不做 clean-corrupt consistency、不做蒸馏，固定 `lambda_cons=0`。
- corruption model 的总损失固定为：

$$
\mathcal L=\mathcal L_{seg}^{input}+0.1\mathcal L_{rel}.
$$

其中 `reliability_target` 为 A1 连续 target，`valid_mask` 只排除 pad。clean sample 的 target 为全 1；RGB target 在本 Depth-only protocol 中始终为 1。
- reliability estimator 只产生辅助 loss，不把预测 reliability 送入 DFormerv2；`GeometryContributionAdapter` 在 A2 保持未接入。这样 A2 隔离“训练分布与 target 是否接对”，B1 才单独检验 learned adapter。

## 6. config、模型与 checkpoint 边界

- 新建独立 common、clean control 和 depth-corruption config；不得覆盖 Quick-B0 config。
- 只有 corruption config 实例化 `ModalReliabilityEstimator(hidden_channels=16)`；clean control 不增加该 head。
- 官方 pretrained 只加载 backbone，新增 reliability head 使用其模块默认初始化；训练 checkpoint 必须 strict 保存/恢复完整模型与 optimizer state。
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

## 8. 限定代码执行单

### 8.1 允许修改

- 新增 `utils/dataloader/mmfr_training.py`；
- 修改 `models/builder.py`，仅增加可选 reliability head 和辅助 loss；
- 修改 `utils/train.py`，仅在 A2 config 开启时调用 batch helper 并传入可选字段；
- 新增 `local_configs/MUSeg/DFormerv2_S_MMFR_A2_Common.py`、`DFormerv2_S_MMFR_A2_Clean.py`、`DFormerv2_S_MMFR_A2_DepthCorrupt.py`。

### 8.2 禁止修改

不得修改 `TrainPre`、`RGBXDataset`、`models/encoders/DFormerv2.py`、A1 corruption 数值语义、DVG-B1 文件、evaluator、checkpoint loader、official-test 文件或 MUSeg 关键状态文档；不得新增依赖、测试文件或临时脚本。

### 8.3 验收与停止条件

- config 关闭时旧训练调用和 loss 接口不变；
- exhaustive uint8 round-trip、同 seed 确定性、不同 sample slot 分离、clean exact normalized no-op、pad exact-zero 保持、target/valid/depth-valid shape/range 通过 CPU 定点检查；
- builder 的可选 head 在 CPU 小 tensor 上得到 finite scalar loss 和 finite backward；
- 只运行受影响 Python 文件静态编译、内联 CPU probe 和静态诊断。
- 任一接口需要修改 DFormerv2、evaluator、checkpoint 语义或 A1 数值定义时立即停止并报告 `scope-blocked`。

## 9. 未授权操作与后继门禁

本地 GPU 单步 preflight 已由用户单独授权并完成。正式训练固定为云端单 GPU，本机只做推理、想法初步验证和小规模 preflight；默认租用 RTX 4090 24GB，RTX 5090 32GB 只在 4090 缺货、24GB 无法容纳冻结 batch size 10，或同配置配对短 probe 证明单位样本成本更低时使用。无论换卡与否，batch size `10`、workers `8`、AdamW、learning rate `6e-5`、500 epoch、warmup 10 epoch、AMP、SyncBN、首轮关闭 `torch.compile`、seed `772961337` 与 selector 都保持不变。

当前仍不运行正式训练、完整 epoch、checkpoint save/load、冻结 batch size 10 云端容量/吞吐 probe、DDP、完整评价、云资源或 official test。开始 probe、clean control 或 Depth-corruption 任一正式训练，仍需用户单独授权；B1 learned geometry adapter 必须另建独立代码计划，不能因 A2 preflight 通过而顺带接入。

**大白话说明：** 本协议把损坏动作放在最终裁剪批次上，用训练位置和样本身份生成独立随机数；模型每次只看一个输入，可靠性头先作为辅助监督，不直接改 GSA。GPU 预检现在证明这条链能完成一次真实更新，但还没有开始长程训练或比较模型效果。

## 10. 实际完成与资格检查

- 已新增 `utils/dataloader/mmfr_training.py`：在最终 CPU crop batch 上恢复严格 uint8/raw 信号，构造 per-sample stateless NumPy RNG，只采样 Depth curriculum，并输出 normalized input、raw input、连续 target、pad-valid mask、Depth validity 和审计 metadata。
- 已修改 `models/builder.py`：只有 corruption config 创建 `ModalReliabilityEstimator(hidden_channels=16)`；训练时强制完整 reliability supervision，计算 segmentation loss 加 `0.1` 倍连续 BCE。A2 只运行固定 signal features 与 reliability head，不计算未使用的四级 pyramid，也不把预测 reliability 送入 backbone、decoder 或 geometry prior。
- 已修改 `utils/train.py`：仅在 A2 corruption config 开启时，于 GPU 搬运前调用 batch helper；配置关闭时继续走原 `model(imgs, modal_xs, gts)` 标量 loss 路径。逐样本 metadata 留在 CPU，只汇总 clean/corrupt 数量和有效区 Depth reliability 均值。
- 已新增 common、clean control、Depth corruption 三个独立 MUSeg config；两身份共享 official pretrained、seed、优化器、500 epoch、尺度增强和 clean selector，不覆盖 Quick-B0 config。
- 主代理 CPU 复核后修正三个边界问题：clean/受损路径 Depth 通道数必须一致；Depth-only 候选必须在抽取 spec 数量前限制，不能先抽 RGB+Depth 再过滤；normalized pad 逆变换得到的均值字节必须在 corruption 前置零，防止 misalignment 把伪 Depth `122` 移入有效区域。builder 同时改为缺少或错配 reliability 字段时 fail-closed。
- CPU qualification：受影响 Python 文件 `py_compile` 通过；聚焦 CPU probe 通过 exhaustive uint8 round-trip、三阶段 Depth-only curriculum、同位置确定性、sample slot 分离、clean `torch.equal` no-op、pad/raw/target 中性语义、misalignment 不引入伪 `122`、完整/缺失 supervision 守卫、finite scalar loss 与 finite backward；六个受影响 Python 文件静态诊断无问题，`git diff --check` 通过。

## 11. 本地 GPU 单步 preflight

- 首次真实 AMP forward 在 `SignalFeatureExtractor` 中触发非有限值。根因是梯度平方和的乘积在 FP16 下溢为 0，余弦对齐分母随之为 0；这在 CPU FP32 小张量资格检查中不会出现。修复后 reliability 固定特征、小型 head 与连续 BCE 在 FP32 中计算，segmentation backbone 和 Ham decoder 继续使用 AMP，损失定义及 reliability 不进入 backbone/geometry prior 的协议保持不变。
- 第二次运行已完成真实 forward/backward/AdamW update，但 probe 在退出阶段错误 finalize checkpoint selector，并因短运行按设计未生成 `latest.pth` 而失败。修复后 `run_kind=probe` 不创建或 finalize selector，仍写出 `run_config.json`、逐步 telemetry 和 `training_result.json`；正式训练的 selector/checkpoint 路径不变。
- 最终 canonical 运行绑定 protocol `MMFR-A2-train-integration-v1`、schedule `mmfr-a2-adamw-6e-5-warmup10-poly0.9-500e-v1` 和物化 `protocol.json` SHA-256 `e9825c4a4818cf2860a529b6c0fa542f5412a7153a1e8b9b8c8d000de6ca7f07`。preflight 状态回填后的 repository template SHA-256 为 `b5b5c979352cca451cfbca35abc77232a9ae82c3f22ec0f33fcdb964ae03e519`。环境为 Python `3.10.20`、PyTorch `2.7.0+cu128`、CUDA `12.8`、cuDNN `90701`、driver `610.88` 和 NVIDIA GeForce RTX 5060 Laptop GPU；官方 pretrained SHA-256 为 `19116988fc86dc9f3e879282237941e11b9b1b5c480edb51e92807311dbc11a6`。
- 运行使用真实 `train-dev`、`480×640`、batch size 1、worker 0、SyncBN、AMP、Ham decoder、Depth corruption 与 reliability auxiliary loss。初始 GradScaler 连续跳过 5 次 update，并将 scale 从首个已记录的 `32768` 降到 `2048`；第 6 个 batch 完成第 1 次 optimizer update。6 个 loss 均有限，累计包含 1 个 clean 与 5 个 corrupted sample，最终更新步 loss 为 `3.34708309173584`。
- 峰值 allocated/reserved CUDA memory 为 `2069.91/2260` MiB，最终可用显存 `4737/8150.56` MiB、free ratio `0.5812`，安全检查通过。`training_result.json` 记录 `attempted_steps=6`、`completed_optimizer_steps=1`、`skipped_optimizer_steps=5`、telemetry invariant 成立、exit code `0`，并明确 `checkpoint=null`、`official_test_included=false`。
- canonical 证据位于仓库内忽略目录 `outputs/mmfr-a2-gpu-preflight-20260914T0138Z/`：`protocol.json` SHA-256 `e9825c4a4818cf2860a529b6c0fa542f5412a7153a1e8b9b8c8d000de6ca7f07`，`run_config.json` SHA-256 `2ff6bf4668efbb58012fced89af698ce4cfec5181c43a4ee4dec9e56dce4edd3`，`training_result.json` SHA-256 `bb3d71dacc99d7b1b97eb03d702f0eff621f78669b7dbb5824aaafad832550c0`，`probe-telemetry.jsonl` SHA-256 `6ae56d55552aee9cacda71b09fea7269069814e18d81e1d3548539d0995d462c`。run config 记录当前未提交工作区 `dirty=true`。
- preflight 未覆盖冻结 batch size 10 的容量、完整 epoch、checkpoint save/load、DDP、`torch.compile`、evaluator、完整测试、云任务或 official test；这些项目不得写成通过。

**完成边界：** `MMFR-A2-train-integration-v1` 现在是 code-qualified 且 GPU-single-step-qualified。它仍未开始 500 epoch 训练、没有训练 checkpoint 或模型指标；正式训练资源已冻结为云端单卡、默认 RTX 4090，准确恢复点是用户单独授权 batch size 10 容量/吞吐 probe 与两个公平对照训练。
