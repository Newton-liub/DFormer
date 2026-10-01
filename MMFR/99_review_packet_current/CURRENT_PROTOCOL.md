# MMFR-A-v1-action-utility-v1：固定实现与第一轮正式训练合同

- 日期：2026-09-30；唯一协议 identity：`MMFR-A-v1-action-utility-v1`；正式 run name：`MMFR-A-v1-action-utility-v1-formal-v1`。
- 设计依据：A-v1 研究设计（`../../doc/reports/2026-09-30-mmfr-next-generation-research-design.md`）。独立于旧 E1 identity，只复用经核验的 C0、既有 corruption 与工程工具。
- 当前事实：implementation complete；Gate-B PASS；第一轮正式合同已冻结。**2026-10-01用户已授权云端必要入口实现、两阶段各3-update预检，通过后连续1920+640正式训练及唯一四条件Quick-Val，完成后停止**；本地本轮仅Git收口/push与指令整理，未执行实验。事实/授权以 当前状态（`../../doc/main/MUSeg-current-status.md`） 为准。§6保留2026-09-30历史交接边界，最新授权及可直接执行的指令见§7。

## 1. 固定 source 与结构

Source 是 Batch 1A C0 fixed-final `update-2560.pth`；SHA-256 为 `ca618b23d18eabb201a0d11d18da383ac99576d0feae5864e3233bda527d9a1a`。本地位置为 `cloud/MMFR_E1_Batch1A_local_transfer_20260922/MMFR_E1_Batch1A_local_transfer_20260922/C0/checkpoint/update-2560.pth`。通过既有 `load_weights_only_model_state` 加载完整 812 个源 model keys；允许缺失键仅 `av1.*`，不恢复 source optimizer/scheduler/GradScaler/RNG。C0 不重训、不改 checkpoint。

固定在 DFormerv2-S encoder output 的 0-based stage index 2，256 channels、约输入 1/16：backbone → A-v1(stage2 only) → 原 HAM/decode head。内部 pairwise geometry prior、其他 stage 与原 HAM 不修改。

- Proposal：`Conv1×1 256→32 → GELU → DWConv3×3 32→32 (padding=1, groups=32) → GELU → Conv1×1 32→256`；所有 Conv 带 bias；最后 weight/bias zero-init。16,992 参数。
- Gate：stage2 global-mean pool 256 维 + 4 维当前 observed raw Depth 统计 → `Linear260→16 → GELU → Linear16→1 → sigmoid`。4,193 参数，每图一个连续标量；新增共 21,185 参数。
- 四统计：geometry-valid 区域内 zero ratio、非零 Depth population mean/std、两端有效且非零的水平/垂直邻域 absolute difference mean。Depth 单通道 `[0,1]`；support 只排除 crop/pad，不表示故障质量。空集合定义为 0；全零且区域有效时为 `[1,0,0,0]`。
- Gate 只见 observed feature/Depth/support，不见 GT、cause/type/severity/index/condition、未损坏 Depth。无新增 norm、attention、多 stage adapter、reconstruction、teacher、prototype、second encoder 或 residual cap。

$$
F_{out}=F+gR(F)
$$

`off` 原样返回 feature 对象、零次 proposal/gate 调用；`full` 执行同一 proposal、g=1；`learned` 执行同模型同参数的连续 gate。g 是补偿动作强度，不是 Depth reliability/failure probability。**大白话：** 模型结构和输入边界保持 Gate-B 已核验版本，本轮只固定未来怎么训练和筛选。

## 2. 两阶段、冻结与 action utility

Config：`local_configs.MUSeg.DFormerv2_S_MMFR_AV1`。接口：`utils/mmfr_av1_training.py` 的 `build_phase_batch`、`configure_phase`、`phase_loss`。当前只有两阶段函数接口，**没有可执行的正式训练循环、3-update 全尺寸 runner、自动阶段切换或 A-v1 评价入口**；交接合同不冒充已实现入口，也不提供会误用旧 `utils/train.py` 的启动命令。

- Proposal：只训练 proposal，base/gate 冻结；full segmentation CE + clean frozen-base KL。
- Gate：只训练 gate，base/proposal 冻结；BCE + learned segmentation CE + clean KL。
- `EncoderDecoder.train` 对 A-v1 固定所有 C0 模块 eval，包括 BN/SyncBN；optimizer 复用既有 Conv/Linear decay 与 bias/norm no-decay 规则，移除空 base groups。未启用 A-v1 的旧模型路径保持不变。
- C0 reliability auxiliary keys 仅为 source 兼容保留，完全冻结，不参与 A-v1 gate/loss/optimizer。
- 同一个 observed batch 的 RGB、normalized/observed Depth、support、label 在 off/full/learned 间复用；metadata 仅用于 clean mask/复现，不进 gate。

每图 endpoint CE 使用相同 non-ignore label support，ignore index 255，空支持 denominator clamp 到 1；margin 单位为 per-image mean CE difference。

$$
u=\operatorname{stopgrad}(L_{off}-L_{full}),\quad M=\mathbf1[|u|>m],\quad t=\mathbf1[u>m]
$$

`u>m` positive，`u<-m` negative，`|u|<=m` ambiguous（含等于边界）。off/full no-grad；utility/target/mask detached；BCE 仅取非模糊样本均值，全模糊时为可微 exact 0，learned CE 仍训练。

$$
\mathcal L_{proposal}=\mathrm{CE}(P_{full},Y)+\lambda_{clean}\mathrm{KL}_{clean}(P_{off}\Vert P_{full})
$$

$$
\mathcal L_{gate}=\mathrm{BCE}_{nonambiguous}(g,t)+\mathrm{CE}(P_g,Y)+\lambda_{clean}\mathrm{KL}_{clean}(P_{off}\Vert P_g)
$$

仅上述 loss，BCE 与 segmentation CE 系数为 1；不新增其他 consistency 权重。使用既有 `capture_rng_state`/`restore_rng_state` 回放同一 pre-forward snapshot（Python/NumPy/torch CPU/CUDA），最后 active pass 按一次 forward 推进随机流；原 HAM 随机 NMF 不改。

## 3. 第一轮正式合同（预冻结，尚未运行）

### 3.1 成功更新、参数与停止逻辑

- Proposal **1920 successful updates**；Gate **640 successful updates**；共 **2560**。Successful update 指 optimizer 真正完成一次更新，不包括 GradScaler skip。
- `margin=0.01`：Gate-B 已验证这一数值路径；第一轮不做超参数搜索。这是预冻结工程/研究起点，不宣称最佳，不用 val-dev 调 margin。
- `lambda_clean=0.1`：两阶段相同的预冻结保护权重，不是调参所得最佳值。
- AdamW；新 branch peak LR **3e-5**；weight decay **0.01**；bias/norm 按现有 no-decay；C0 不进 optimizer。
- 继承 E1 的 successful-update linear warmup **128**、poly power **0.9**、GradScaler initial scale **1024**、growth factor 2、backoff 0.5、growth interval 2000。每阶段使用独立 optimizer/scheduler/GradScaler，从该阶段第 1 次成功更新重新计数；这是新两阶段合同，不恢复旧 E1 schedule。

设该阶段 successful update 为 $k=1,\dots,N$，$N$ 分别为 1920/640；scheduler 设置供本次更新使用的 LR：

$$
\mathrm{LR}(k)=3\times10^{-5}\begin{cases}k/128,&k\le128\\((N-k)/(N-128))^{0.9},&k>128\end{cases}
$$

本轮只冻结该规则，尚无 runner 执行。阶段计数和 scheduler 只由成功更新推进，最终阶段 LR 到 0；不通过 val-dev 或 Quick-Val 更改。

Proposal 正常完成 1920 后进入 Gate；仅 NaN/Inf、GradScaler skip/optimizer 异常、A-v1/frozen-base/source 身份破坏、显存/基础设施无法满足合同、loss 明显异常且无法解释时停止。单个或几个 batch 的 off/full utility 不理想不是停止理由；**不增加 train-dev utility early-stop gate**。Proposal 是否有用主要留给之后同权重 full Quick-Val 判断。

Gate 正常完成 640；只因工程/数值/身份错误停止，不因中间 gate 分布不好看而停止。禁止 val-dev 延长预算、自动追加 epoch、Quick-Val 后回调阶段长度或参数。

### 3.2 数据、随机流与执行设置

- 仅真实 `train-dev`（1277 样本）；沿用现有训练增强/normalization，最终训练 crop **480×640**；保留目标 **batch 10、workers 8、accumulation 1、single GPU、DDP off**。batch10 是否适合 4090 尚未证明，预检失败即停止，不自动 sweep/降 batch。
- AMP on（CUDA autocast float16 + GradScaler）；TF32 matmul **on**、cuDNN **on**，显式采用 `float32_matmul_precision=high`，继承 E1 经勘误确认的实际行为。Gate-B 的 FP32/TF32 off/B1/64×64 不作为正式设置。BN/SyncBN 保持 frozen C0 eval。
- train/data seed **772961337**；A-v1 initialization seed **2026093000**；Proposal/Gate phase corruption seeds **2026093001 / 2026093002**。不新增 seed。
- `p_clean=0.25`、Depth only、max_specs=2；六类 `entire_missing/spatial_dropout/gaussian_noise/blur/quantization/misalignment`；既有 v3/MID-A validity、uint8、归一化与 pad 语义不变。
- 复用 `build_phase_batch` 的现有 **phase-local v3 curriculum 0..1**，而不是旧 E1 caller 的 0.84..0.88 continuation remap。Proposal 数据坐标为 15×128、Gate 为 5×128，epoch 从 1、iteration 从 0；这是 corruption/data cursor 的坐标，不是可自动延长的 epoch 合同。Gate 重建新 corruption 流，无 batch/target cache 继承。预检用各阶段起始 3 个坐标与完整正式分母，不把正式分母改为 3。
- 稳定记录 attempted/completed/skipped，skip 不计预算且立即停止回报；不无限重试。official test **sealed_unread**，config 的 val/eval/test sources 为 None、training validation disabled。

### 3.3 Checkpoint / recovery

- Proposal 成功更新 1920：保存 **`proposal-update-1920.pth` transition/recovery**；Gate 必须从同一 Proposal 权重进入，重置 Gate optimizer/scheduler/GradScaler、使用独立 phase seed，不复用 Proposal optimizer。
- fixed-final 唯一性能候选：Gate 完成后的全局 **`update-2560.pth`**；不选 best/val-dev/top-k，不用 transition 替代 fixed-final 筛选。
- 每 **640 successful updates** 保存 recovery，并保留 transition 和 fixed-final。恢复内容包含 model、当前 phase、phase/global counters、optimizer、scheduler、GradScaler、全部 RNG、data/sampler/augmentation cursor 及 config/source identity；同阶段中断需原样恢复，不能静默 weights-only 重启冒充连续运行。未来 runner 必须落实，当前不声称已实现。
- 容量预检更新和产物不计正式预算、不作为正式起点；正式运行若日后获授权，应从原 C0、zero-init Proposal 和原初始化/phase seeds 干净重启。

## 4. 正式 Quick-Val 筛选线（2026-10-01已获条件执行授权，尚未运行）

正式训练成功且薄入口完成必要定点复核后，只允许 **一次**四条件 Quick-Val：`clean`、`entire_missing@1.0`、`spatial_dropout@0.75`、`misalignment@0.75`。同一 fixed-final A-v1 checkpoint 输出 off/full/learned；off 为 matched off，不用旧 C0 summary 代替当前同输入/RNG对照。

沿用既有 318 val-dev、original-full、scale 1.0、no flip、FP32、TF32 off、原 Label grid、corruption identity/evaluation seed **2026091401** 和 reset-per-unit；新 A-v1 薄入口仍待资格审核，本轮不修改旧 evaluator。

三个 hard condition 的 mIoU 作未加权平均；pp 为百分点。`promote-for-next-review` 必须同时满足：

1. learned hard 平均相对 matched off **>= +0.50 pp**；
2. learned clean 相对 matched off **>= -0.20 pp**；
3. learned hard 平均 **严格 > full**。

full 用来判断 Proposal 是否有用；learned 用来判断 selector 是否好于始终补偿；clean 限制牺牲正常输入。full 有收益但 learned 不优于 full，记 **`selector-not-supported`**；full/learned 都基本无收益，记 **`stop`**；微小变化或方向混合，记 **`inconclusive`**；全部满足记 **`promote-for-next-review`**。后三种描述性归类不另造数值阈值，存在边界歧义时保留原分数交上级裁决；唯一数值继续线是上述三项。单 seed 筛选线不是统计显著性，不自动授权 Main-Val 或下一轮训练。

## 5. Gate-B 原始证据与本地收口核验

原审核报告（`../02_evidence/report_mmfr_a_v1_implementation_gateb_20260930.md`）、原差异审核（`../02_evidence/audit_mmfr_a_v1_implementation_20260930.md`）、Gate-B JSON（`../02_evidence/mmfr_a_v1_gateb.json`） 保留历史正文/字节。证据 SHA-256 **`bf87cc2fc46dd413131f7a6778d8df82388cf7fffa08852126d1f62178be01bf`**。

Gate-B 原始范围为 synthetic B1 64×64、真实 source/HAM、FP32/TF32 off，八次 forward、每阶段一次 backward/step，无 dataset/split 读取。必要工程项 PASS 只证明实现资格，不证明性能、收敛、全尺寸容量或创新性。其 smoke m/lambda/LR 当时不是正式超参数；本次由用户独立冻结为正式起点，历史证据不改写。

本地四个未改运行文件可能保留既有CRLF，Git按原策略归一为LF；既有复现JSON同时记录本地正式config hash与5个Git LF blob SHA，云端使用Git LF版本核对，不能把换行差异误判为模型修改。`.gitattributes`对A-v1证据JSON、hash绑定的canonical sources/生成审核包以及历史pre-A-v1快照定点设 `-text`，保证恢复commit时source/packet hash和原证据字节仍匹配；运行代码不改变换行策略。

收口开始时五个关键代码文件 SHA-256 全部匹配原 `implementation_sha256`；本轮只允许 config 发生正式合同差异。`models/mmfr_av1.py`、`models/builder.py`、`utils/mmfr_av1_training.py`、`tools/mmfr/av1_gateb.py` 保持原字节；正式 config 新哈希记入既有 `reproducibility_current.json`，与 Gate-B 配置版本区分。不改 forward/loss/RNG，因此不重跑 GPU Gate-B；只做 config import、source/代码/证据身份、定点差异及文档坏链检查。

## 6. 2026-09-30历史 Cloud handoff：4090 全尺寸容量/测速边界

本节记录当时仅预检、预检后等待授权的交接快照；2026-10-01授权变更见§7，预检上限和技术合同保持不变。

- Branch：`perf/mmfr-a2-v3-pipeline-opt1`；本地交接 commit SHA：**`e26d670e279970ecb8907aef1de470e992be500e`**（`feat(mmfr): prepare A-v1 formal training handoff`，26个A-v1相关文件；未push）。完整 SHA 是提交后回执，无法作为该 commit 自身的内容；提交前快照用此标记，提交后只回填元数据，不 amend 或另建 commit。
- Identity：`MMFR-A-v1-action-utility-v1`；config import：`local_configs.MUSeg.DFormerv2_S_MMFR_AV1`；run name `MMFR-A-v1-action-utility-v1-formal-v1`。
- Source C0 SHA：`ca618b23d18eabb201a0d11d18da383ac99576d0feae5864e3233bda527d9a1a`；同步 commit 后必须独立匹配 checkpoint，checkpoint 不进 Git/MMFR。
- 正式合同：Proposal/Gate **1920/640**，m **0.01**，lambda_clean **0.1**，LR **3e-5**，WD **0.01**；phase seeds **2026093001/2026093002**；expected batch **10**，480×640、AMP fp16 on、TF32 matmul/cuDNN on、workers8、accumulation1。
- 当前没有 full-resolution preflight runner。下一会话若获云端权限，应先实现/审核只允许每阶段最多3次成功更新的最薄调用器，使用现有 phase 接口；禁止自动接正式 schedule。该缺口不等于本轮合同/提交失败，但意味着不能直接用旧 E1 train 命令执行 A-v1。

### 6.1 Proposal full-resolution preflight

真实 train-dev/正式分辨率/正式 batch/AMP/TF32/config，Proposal phase，最多 **3 successful updates**，达到3立即退出。OOM、NaN/Inf、GradScaler skip、身份/optimizer 异常立即停止，不能等到3或自动重试/改 batch。

记录每步 update time（CUDA synchronize 计时，涵盖 phase loss 各 forward/backward/optimizer；另列含数据加载的 wall time）、loss、finite、OOM、GradScaler attempted/completed/skipped、batch，及该 phase reset 后的 peak allocated/reserved（bytes 和 MiB）。只报告3步原始记录和简单均值，不扩展 benchmark。

### 6.2 Gate full-resolution preflight

只有 Proposal 3-update 正常后才允许启动 Gate，使用同一 preflight Proposal 权重供工程检查（不是正式 transition），冻结 Proposal，新 Gate optimizer/scaler 和独立 phase seed；同分辨率/batch/AMP/TF32，最多 **3 successful updates**。

Gate 包含 off/full endpoints + learned active pass，必须单独测显存/速度。记录 peak allocated/reserved、每步 update time、loss/CE/BCE/clean KL、每图 utility、positive/negative/ambiguous 数量、finite/OOM/NaN、GradScaler skip 和 batch；计时涵盖全部三行为和一次 optimizer update。all-ambiguous 本身不是失败，数值/身份/梯度错误才是。

**每阶段3 updates后强制停止；两阶段都完成后立即回报显存、速度、loss、utility与稳定性。预检不会自动授权或启动1920+640。正式训练尚未授权，本轮也未授权云端登录/同步/运行。** 下一步仅允许在单独云端授权后同步该 commit 到 RTX 4090，并依次执行 Proposal 3-update 与 Gate 3-update full-resolution preflight；完成后停止等待上级审核。

## 7. 2026-10-01最新授权与云端执行指令

用户已明确授权本地全量收口/push，并选择云端“必要入口实现 → 两阶段各3-update预检 → 通过后1920+640正式训练 → 唯一四条件Quick-Val → 停止”。本节替代§6的历史授权限制，结构、损失、预算和筛选线不变。本地当前只交付Git与指令，没有执行云端实验。以下指令可直接交给云端执行会话：

> 目标：尽快完成MMFR A-v1第一轮训练和四条件筛选，用真实结果决定是否继续，而非扩充测试、文档或模型。
>
> 1. 同步origin的`perf/mmfr-a2-v3-pipeline-opt1`分支，先读`doc/main/MUSeg-current-status.md`和本protocol §1–4、§7。当前只有阶段函数，预检/正式runner与A-v1评价入口尚缺；复用`utils/mmfr_av1_training.py`及已有数据、corruption、checkpoint、评价工具实现最薄入口，不直接套旧E1训练循环。允许显式落实本次训练/四条件评价授权；保留训练中Val关闭与test禁用。用用户指定现有4090，不自行创建付费资源。
> 2. 必要启动检查仅确认代码提交、真实train-dev/val-dev划分和C0初始权重SHA；C0 SHA为`ca618b23d18eabb201a0d11d18da383ac99576d0feae5864e3233bda527d9a1a`。不全仓扫描、全量测试、重复Gate-B、文献搜索、batch sweep或benchmark。新入口只做定点import/字段检查及下一项真实预检，评价复核并入必要入口检查/首个样本，不另开一轮实验。
> 3. 正式设置batch10、480×640、workers8、accumulation1、AMP fp16/TF32 on。先Proposal最多3次成功更新，再用同一预检Proposal权重做Gate最多3次；记录显存、每步/含数据加载时间、loss与scaler skip，Gate另报utility正/负/模糊计数。每个短运行到3退出；两阶段都通过后，无需再等授权，从原C0、冻结初始化和原种子干净开始正式训练，预检不计预算、不作正式起点。
> 4. 严格Proposal1920/Gate640，margin0.01、lambda_clean0.1、AdamW LR3e-5/WD0.01；每阶段独立optimizer/scaler、128-successful-update warmup/poly0.9；种子、phase-local corruption和配对RNG按§2–3，不改C0/BN/结构/loss。每640次成功更新保存完整恢复状态；Proposal1920保存`proposal-update-1920.pth`，Gate从同一Proposal进入，最终只用`update-2560.pth`。不因utility或gate分布不好看提前停，不自动追加预算。
> 5. 成功后，对同一fixed-final仅做一次四条件Quick-Val：clean、entire_missing@1.0、spatial_dropout@0.75、misalignment@0.75；每条件off/full/learned，同输入/配对HAM随机状态，318 val-dev/original-full/scale1/no flip/FP32/TF32 off/eval seed2026091401/reset-per-unit。最小评价入口复核确认observed Depth/support、原标签网格、strict off与C0一致及指标累积；不得用旧C0 summary替代同checkpoint matched off。输出三行为每条件mIoU/hard平均、learned相对off差值、gate mean/median和§4筛选结论，完成后停止。不得擅自Main-Val/official test、新seed、调参或追加训练。
> 6. 普通工程错误允许最小修复、受影响定点检查、提交并push至origin同名分支；每次运行代码/config先提交并记录完整SHA。错误先停止并留日志，按恢复兼容性决定续训或新运行，不无限重试、不静默weights-only重启冒称连续训练。OOM/NaN/Inf/scaler skip立即停止；修复需要改变batch、模型、loss、种子、预算或评价口径，以及恢复正确性无法确认时，交人工决定。
> 7. 最终交付实际运行/修复commit、原始日志和结果位置、fixed-final SHA、训练计数/耗时、四条件分数及筛选结论；结果不论正负都如实报告。同步更新两份实时入口和已有MMFR证据/审核包，代码和小型结果材料提交并push，供本地Git拉取。数据/checkpoint/大日志不入Git；文档可引用对应代码提交，不为追逐最新HEAD反复回填hash。优先完成结果，报告保持简短，单seed筛选不冒称论文结论或统计显著性。

运行版本以实际提交为准，必要数据/权重/原始证据身份仍须核对；普通工程文件hash限制可简化或由人工确认，未确认必须标为待核验。本次授权不保证运行成功，也不改变历史原始证据。
