# MMFR E1 Batch 1A/1B Protocol Freeze

> **2026-09-24 v2 执行补充（以下 §1–§13 的 v1 冻结结果均为历史记录，不代表 v2 Gate-B）：** RTX 4090 上 v1 R-OE-lite 正式训练在至少 168 次 optimizer update 后因 398 通道全分辨率插值 OOM 中止；未生成 fixed-final checkpoint。用户批准 R-OE-lite v2 新实现身份 `MMFR-E1-Batch1B-R-OE-lite-v2`：将原 `d2 → GELU → head` 移至最终双线性插值之前，使完整分辨率插值只接收 1-channel logits；其余七层参数定义、`3,302,785` 参数、observable-empty 路由、padding 和 reliability auxiliary 不变。逐像素函数已改变，不继承 v1 训练状态或 v1 Gate-B 结论。新训练从 A2 epoch-420 source checkpoint 干净启动，保留 batch size 10、workers 8、AMP/SyncBN on、DDP off、原 seed `772961337` 与其余 Batch 1B 训练设置。先在 RTX 4090 上做 3 次连续成功 update 并记录 loss/显存/触发数/最终插值输入 shape 和 dtype；无 OOM、loss finite、显存余量合理即按用户条件授权启动全新 2560-update 正式 run，训练成功后执行现有四条件 Quick-Val。若 OOM 则停止。当前实际执行与授权状态只以 [`MUSeg-current-status.md`](../../doc/main/MUSeg-current-status.md) 为准；v1 原始中止证据见 [`中止报告`](../02_evidence/report_e1_batch1b_roe_formal_training_attempt_20260924.md)，不得改写。

> **当前 protocol identity：** `MMFR-E1-Batch1B-R-OE-lite-v1`；历史 Batch 1A identity：`MMFR-E1-Batch1A-C0-F-v1`
> **状态：** `Batch 1B R-OE-lite implementation complete; Gate-B PASS; formal training not authorized`
> **日期：** 2026-09-23
> **性质：** Batch 1A implementation、正式训练、Quick-Val 与十条件 Main-Val 历史记录，以及 Batch 1B R-OE-lite 的冻结协议、实现与最小 Gate-B 结果；Batch 1B 尚未正式训练，不涉及 Quick-Val、Main-Val、T 或 official test。
> **official test：** `sealed_unread`

## 1. 协议裁决

- `Gate-A-common = PASS`；
- `Gate-A-F = PASS`；
- `R-EM-lite = retired-by-observability`；
- `Batch 1A C0/F = implementation-authorized, Gate-B-passed, training-complete-PASS`；
- `Quick-Val = authorized, completed, F-lite promote`；
- `R-OE-lite = implementation-complete, Gate-B-PASS`；
- `Batch 1B R-OE = formal-training-authorization-pending`。

Batch 1A 正式训练已完成且 C0/F-lite 均 PASS；Quick-Val 已完成并判定 F-lite `promote`；十条件 Main-Val 已于 2026-09-23 在本地完成，两侧各 10/10 条件且身份与配对性断言 0 失败，但本轮未为 Main-Val 预注册数值门禁，因此十条件数字仍是描述性对照结果（见 §12）。Batch 1B 的冻结协议、实现与最小 Gate-B 已完成并通过；它只证明 R-OE-lite 具备按冻结合同进入正式训练的实现资格，不等于正式训练授权或效果结论。official test 仍在边界之外。

## 2. Batch 结构

### 2.1 Batch 1A

候选：

- `C0`：matched continuation；
- `F-lite`：stage 1/2/3 unconditional residual adapter。

比较：

$$
\Delta_F=F\text{-lite}-C0.
$$

### 2.2 Batch 1B

候选：

- `R-OE-lite`：Observable-Empty Geometry Substitute；
- matched control：C0。

比较：

$$
\Delta_{R\text{-OE}}=R\text{-OE-lite}-C0.
$$

只有当 checkpoint、optimizer、base loss、training budget、corruption manifest、seed 与 evaluation protocol 和 Batch 1A 完全一致时，Batch 1B 才可复用 Batch 1A C0。任一共同合同变化均要求重跑对应 C0。

## 3. `R-EM-lite` 正式关闭

`R-EM-lite` 原要求识别 hidden cause=`entire_missing`。当前 observable Depth 无法区分：

1. synthetic entire missing；
2. natural all-invalid crop；
3. spatial dropout emptied crop。

因此状态固定为 `retired-by-observability`。禁止以后用 corruption type、severity、synthetic mask、clean Depth、`depth_valid_pre/post`、reliability target、generator RNG 或 manifest cause 重新引入该路线。

## 4. R-OE detector 合同

允许输入：当前实际 RGB、当前实际 raw Depth、preprocessing 合法知道的 crop/pad geometry support。trigger 只读取当前 raw Depth 与 $V_{\mathrm{geom}}$：

$$
\phi_{\mathrm{OE}}
=
\mathbf 1
\left[
\sum V_{\mathrm{geom}}>0
\land
\max_{V_{\mathrm{geom}}=1}D_{\mathrm{u8}}=0
\right].
$$

其语义只能是：

> 当前 crop 的真实图像支持域中没有任何非零 Depth observation。

它故意同时触发 synthetic entire missing、natural-empty crop 和 dropout-emptied input。不得命名为 `entire-missing detector` 或 `failure-cause classifier`。

R-OE-lite 的 substitute 结构、exact bypass、数值合同、参数量与 loss 边界见 `r_oe_lite_design.md`；实际实现与 Gate-B 证据见 §13。本协议不授权正式训练、Quick-Val、Main-Val、T 或 official test。

## 5. Source identity

- checkpoint：`experiments/MMFR_A2_v3/checkpoints/selector-epoch-420.pth`；
- size：`321103318` bytes；
- SHA-256：`2b72eb7f28e5c4b4f52bdc43e2ac5c7298169325a412bfbcbc7a022c29822597`；
- schema：`dformer-training-checkpoint-v2`；
- completed/next epoch：`420/421`；
- global optimizer step：`53735`；
- embedded source commit：`d82d83482722776f5dc5059c80c34975e828e402`；
- model keys：`812`；
- candidate manifest SHA-256：`5e9b6f95d6c2aad5e474ad974f6a06d543c8d68940eb146965c5a0d761e46bb1`；
- `train-dev`：1277，SHA-256 `a6b15b63f6d5193e3928ea24ada25be403a48e68d1c1f9372cdbbc3fe5cd8470`；
- `val-dev`：318，SHA-256 `1d0719d8f64f016d48995c25ab66d4004d76b7155d9efeef7cbb7454c0dd0e83`。

任一 identity 不匹配直接 `protocol-blocked`。

## 6. Restart 与基础 loss

- 从同一 epoch-420 完整 model weights 开始；
- 不恢复旧 optimizer、scheduler、GradScaler 或 RNG；
- 新建 AdamW；
- 所有组保留：

$$
L_{\mathrm{base}}^{A2}
=
L_{\mathrm{seg}}
+
0.1L_{\mathrm{reliability,depth}}.
$$

- segmentation：`safe_masked_mean(cross_entropy)`；
- auxiliary segmentation：关闭；
- reliability：Depth-only BCE；
- consistency：0；
- reliability auxiliary 不进入 segmentation network。

## 7. Optimizer identity

固定四组与顺序：

1. `base_decay`：LR `1e-5`，WD `0.01`；
2. `base_no_decay`：LR `1e-5`，WD `0`；
3. `new_decay`：LR `3e-5`，WD `0.01`；
4. `new_no_decay`：LR `3e-5`，WD `0`。

要求：

- 29 个 `backbone.layers.*.blocks.*.Geo.weight` 全部属于 `base_decay`；
- 14 个审计到的 SyncBN 参数属于 `base_no_decay`；
- reliability head weight/bias 分别进入 base decay/no-decay；
- F adapter Conv weight → `new_decay`；
- F adapter bias → `new_no_decay`；
- C0 的 new groups 允许为空；
- 所有 `requires_grad=True` 参数 membership 恰好为 1；
- 不迁移 epoch-420 optimizer state。

## 8. F-lite 结构

插入点：backbone tuple 后、decoder 前。stage 0 不改，stage 1/2/3 各使用独立 adapter：

$$
F_l'
=
F_l+
W_{up}^{(l)}
\operatorname{GELU}
\left(
W_{down}^{(l)}F_l+b_{down}^{(l)}
\right)
+b_{up}^{(l)}.
$$

冻结字段：

- bottleneck ratio `1/4`；
- no normalization；
- down weight `trunc_normal_(std=0.02)`；
- down bias 0；
- up weight/bias 0；
- 不接 reliability、condition、severity、oracle；
- geometry path、decoder 不改；
- trainable parameters 精确 `173152`。

梯度资格：第一次 successful update 后 6 个 up weight/bias 都必须变化；第二次 backward/update 时 6 个 down weight/bias 都必须出现有限非零梯度并变化。

成本资格：training peak allocated memory delta `<= 512 MiB`；batch-1 inference median latency delta `<= 5%`。

## 9. Gate-B 结果

Canonical：`outputs/mmfr-e1-batch1a-gateb/e1-batch1a-gateb.json`，SHA-256 `5d5f0526e95ed81b6da8fbcd3cc395911f2266132a9148826b5ad261042e4e89`。

- `C0: PASS`；
- `F-lite: PASS`；
- `failed_checks=[]`；
- source/C0 812 shared state keys exact equal；
- C0 logits/prediction/loss exact equal；
- F stage 1/2/3 residual exact zero；
- F decoder inputs/logits/prediction/loss exact equal；
- optimizer membership 全部为 1；
- C0/F 的 29 个 Geo 全覆盖，19 个在该 batch 上有非零 gradient，29 个均发生合法 AdamW parameter change；
- F step 1：6/6 up finite nonzero gradient、6/6 updated，down gradient 合法为 0；
- F step 2：6/6 down finite nonzero gradient、6/6 updated；
- F memory delta：`213.713867 MiB` allocated、`152 MiB` reserved；
- F inference median delta：`+1.026519775390625 ms / +0.7825612403259408%`；
- F forward-loss median delta：`+0.9180450439453125 ms / +0.5561633978759639%`。

详细数值见 `../02_evidence/report_e1_batch1a_gateb.md`。

## 10. 正式训练结果（已完成，C0/F-lite 均 PASS）

- single GPU；batch 10；workers 8；accumulation 1；
- SyncBN on；DDP off；AMP on；
- **TF32 勘误：** 原字段 `TF32 off` 与实际训练入口不一致。两次正式训练均保留 `utils/train.py` 中既有的 `torch.set_float32_matmul_precision("high")` 行为，实际允许 CUDA matmul TF32，cuDNN TF32 也保持 PyTorch 2.1.2 默认开启；本轮接受该既有行为，不重跑；
- seed `772961337`；
- 20 nominal epochs × 128 attempts = 2560 slots；
- C0 与 F-lite 均完成 `2560/2560 successful updates`；
- 两者均为 `skipped=0`，正式训练判定均为 `PASS`；
- base LR `1e-5`；new LR `3e-5`；
- weight decay `0.01`，bias/norm no-decay；
- 128 successful-update linear warmup；
- poly power `0.9`，update 2560 到 0；
- GradScaler initial 1024、growth 2.0、backoff 0.5、growth interval 2000；
- A2 v3 virtual curriculum 约 `0.84 → 0.88`；
- `p_clean=0.25`，`max_specs=2`，六类 Depth corruption；
- recovery 每 640 successful updates；
- C0 final checkpoint：`cloud/mmfr-e1-batch1a-v1/C0/development/seed-772961337/checkpoint/update-2560.pth`，SHA-256 `ca618b23d18eabb201a0d11d18da383ac99576d0feae5864e3233bda527d9a1a`；
- F-lite final checkpoint：`cloud/mmfr-e1-batch1a-v1/FLite/development/seed-772961337/checkpoint/update-2560.pth`，SHA-256 `ea9319e5abe55b996470ee0a75bd63b887241ef834a3b50f145b5b7d4aabd98d`；
- selector/top-k/best/latest selection 关闭；
- **DataLoader shuffle 记录：** 两个 run 的样本 shuffle 顺序存在轻微差异，原因是 F-lite adapter 初始化额外消耗全局 torch RNG；本轮接受为 screening-level 随机性差异，不修改采样器、不重跑训练。

## 11. Quick-Val（已授权、已执行，F-lite 判定 `promote`）

评价固定为：318 条完整 `val-dev`、original-full、scale 1.0、no flip、FP32、TF32 off、原始 Label grid、fixed final checkpoint only。

条件：clean、`entire_missing@1.0`、`spatial_dropout@0.75`、`misalignment@0.75`。

**执行入口（新增，需记录为偏离）：** 上述“单视图 `original-full` + 四个 condition 计分”在既有冻结工具中没有对应 runner —— `tools/evaluate_museg_checkpoint.py` 提供 `original-full` 几何但没有 condition 计分入口，`tools/evaluate_museg_10condition.py` 能计分但视图硬编码为十视图 `msflip-whole-original-grid-v1`（其 scale 与 view 常量不从 `--protocol` 读取）。经用户单独裁决，采用“严格按文本新写最小单视图条件计分入口”的方案：新增 `tools/mmfr/e1_quickval.py`，SHA-256 `928c4229d937552e00128e9291b915204291f58c1e80b4b5a005efa9365e4073`。该入口是薄复用层：模型构建与 strict 载入、FP32/TF32-off、`original-full` 输入契约、logits 回原网格、confusion 与指标均调用冻结的 `tools.evaluate_museg_checkpoint`；condition 定义、corruption 应用与样本读取均调用冻结的 `tools.evaluate_museg_10condition`（冻结 evaluation seed `2026091401`、冻结 condition index）；RNG 沿用冻结 `load_eval_model` 约定（构建后用 evaluation seed 播种 torch，并 `reset-per-unit` 回放同一 base RNG 状态）。**该入口没有既有冻结资格检查**，其 `original-full` 输入契约已用 `--self-check` 证明与冻结 `MUSegPostEvalDataset` 在 `318/318` 样本上 `rgb`/`depth`/`label` 逐位相等。

**结果（mIoU %，单视图，318 样本）：**

| condition | C0 | F-lite | delta (F-lite − C0) |
| --- | --- | --- | --- |
| clean | 53.46 | 54.15 | +0.69 pp |
| `spatial_dropout@0.75` | 51.40 | 52.39 | +0.99 pp |
| `misalignment@0.75` | 52.15 | 52.63 | +0.48 pp |
| `entire_missing@1.0` | 48.76 | 50.17 | +1.41 pp |

$$
M_{3,\mathrm{hard}}: 50.77 \rightarrow 51.73,\quad \Delta_F=+0.96\ \text{pp}
$$

**判定 `promote`：** 满足 $\Delta_F\ge+0.50$ pp、clean $\ge-0.25$ pp、每个 hard condition $\ge-0.50$ pp；stop 条件（$\Delta_F\le0$、clean $<-0.50$ pp、任一 hard $<-1.00$ pp）均未触发。checkpoint 身份：C0 `ca618b23d18eabb201a0d11d18da383ac99576d0feae5864e3233bda527d9a1a`、F-lite `ea9319e5abe55b996470ee0a75bd63b887241ef834a3b50f145b5b7d4aabd98d`。证据：`cloud/mmfr-e1-batch1a-v1/quickval-comparison.json`（SHA-256 `1f00d4face8587183c2e235eac689cb6c77556dc0b6c8636d95a8c5471176f7f`）与两侧 `quickval-original-full/`（含逐条件 `metrics.json` 与 `summary.json`）。**最小验证：** 输入契约逐位等价；同 checkpoint 两进程重复运行给出相同 clean mIoU 与相同 confusion matrix；C0/F-lite 的逐样本顺序、corrupted Depth SHA-256 与像素支持在四个条件上完全相同。**边界：** 该结果是单视图口径的 screening 证据，不得与十视图数字直接比较，也不构成 Main-Val、10-condition、Batch 1B 或 official test 授权。

F-lite：

$$
M_{3,\mathrm{hard}}=(EM+SD+Mis)/3.
$$

- promote：$\Delta_F\ge +0.50$ pp；clean `>= -0.25 pp`；每个 hard condition `>= -0.50 pp`；
- stop：$\Delta_F\le 0$，或 clean `< -0.50 pp`，或任一 hard condition `< -1.00 pp`；
- 其他合法完成：inconclusive。

R-OE-lite 不继承 R-EM 的 cause-specific gate。Batch 1B 的冻结 Main-Val gate、Quick-Val screening-only 边界与压力条件解释已在 §13.5 独立确认。

## 12. 授权边界与恢复点

Batch 1A C0/F-lite 正式训练均已完成并 PASS，各完成 `2560/2560 successful updates` 且 `skipped=0`；4-condition Quick-Val 已在 318 条 `val-dev` 上完成并判定 F-lite `promote`；十条件 Main-Val 已于 2026-09-23 在本地完成；Batch 1B R-OE-lite 实现与 Gate-B 已通过。本轮未启动 R-OE 正式训练、未运行 Quick-Val/Main-Val、未运行 T，也未读取 official test。

**十条件 Main-Val 描述性结果（无预注册门禁）：** 两侧各完成 10 个条件 × 318 条 `val-dev` × 每图 10 view；`is_full_ten_condition_full_val_dev=true`，每条件 `completed=true`、`official_test_included=false`；两侧 runner、协议原文、条件定义、样本集合、split、evaluation seed、有效视图分批与 FP32/TF32-off 身份逐项一致，每条件 `identity_sha256` 等于 manifest 期望值，两侧同条件 corrupted Depth 聚合 SHA-256 全同、标签像素支持同为 `155,829,149`（断言 0 项失败）。

| 量 | C0 | F-lite | Δ (pp) |
| --- | --- | --- | --- |
| $M_6$（六个单故障宏平均 mIoU，主指标） | 55.2883 | 54.8917 | -0.3966 |
| clean mIoU | 56.69 | 56.02 | -0.67 |
| 三个混合条件宏平均 mIoU | 55.6933 | 55.3533 | -0.34 |
| 九个受损条件宏平均 mIoU | 55.4233 | 55.0456 | -0.3777 |
| 最坏单条件 mIoU（`entire_missing@1.0`） | 52.63 | 52.00 | -0.63 |

十个条件中仅 `spatial_dropout@0.75`（`+0.40` pp）与 `spatial_dropout@0.5+gaussian_noise@0.5`（`+0.42` pp）为 F-lite 更高，最大负差为 `blur@0.5+misalignment@0.5`（`-0.81` pp）。与 §11 的单视图 Quick-Val 相比，4 个共同条件中 clean（`+0.69 → -0.67`）、`entire_missing@1.0`（`+1.41 → -0.63`）、`misalignment@0.75`（`+0.48 → -0.33`）三个符号翻转，仅 `spatial_dropout@0.75` 同为正值；按冻结约定两套口径的绝对数字不可直接比较。

**解释边界：** 本协议未为 Main-Val 定义 promote/stop 门禁；§11 的数值门槛只适用于 Quick-Val 且已在 2026-09-22 使用。结合单 seed、单 checkpoint、单次运行与无 bootstrap 或置信区间，上述数字是描述性证据，不构成 F-lite 去留判定，也不构成 C0 更好的结论。已如实记录的缺口：Main-Val 的 evaluator stdout 与退出码未落盘，完成性证据是产物完整性加身份一致性；C0 的 10 个条件跨暂停前后两次会话完成。详细分析见 `doc/reports/2026-09-22-mmfr-e1-batch1a-c0-flite-mainval.md`。

**恢复点：** `ready-for-R-OE-formal-training-authorization`。Batch 1B Gate-B 已通过，但 formal training authorization 尚未获得；在授权前不启动 20 epoch 正式训练、Quick-Val、Main-Val、T、Batch 2 或 official test。Batch 1A Main-Val 的上级处置仍保持独立，不修改 evaluator、不重选 checkpoint、不追加研究设计；official test 保持 `sealed_unread`。

## 13. Batch 1B R-OE-lite protocol 与 Gate-B 状态

### 13.1 冻结的共同训练合同

Batch 1B 与 Batch 1A C0 共享同一 source、数据、seed、训练预算、corruption 规则、loss、optimizer 语义、checkpoint 机会和 TF32 实际行为。合同为：

- source：A2 epoch-420，checkpoint `experiments/MMFR_A2_v3/checkpoints/selector-epoch-420.pth`，SHA-256 `2b72eb7f28e5c4b4f52bdc43e2ac5c7298169325a412bfbcbc7a022c29822597`，model keys `812`，epoch `420/421`，global optimizer step `53735`；
- seed：`772961337`；
- 20 nominal epochs、每 epoch `128` attempts、`2560` successful updates；batch size `10`、workers `8`；
- AMP on、SyncBN on、DDP off；base LR `1e-5`、new-module LR `3e-5`；warmup `128` successful updates、poly power `0.9`；
- `p_clean=0.25`、`max_specs=2`、六类 Depth corruption：`entire_missing`、`spatial_dropout`、`gaussian_noise`、`blur`、`quantization`、`misalignment`；virtual curriculum `0.84 → 0.88`；
- 共同损失仍为 $L_{\mathrm{base}}^{A2}=L_{\mathrm{seg}}+0.1L_{\mathrm{reliability,depth}}$，不增加 substitute reconstruction、consistency、condition、severity 或 oracle loss；
- 四组 optimizer 保持 `base_decay / base_no_decay / new_decay / new_no_decay`，base/new LR 分别为 `1e-5 / 3e-5`；checkpoint 机会和 fixed final checkpoint 规则保持不变；
- 从 source 做 weights-only restart，不恢复 epoch-420 optimizer、scheduler、GradScaler 或 RNG state；
- TF32 沿用 Batch 1A 已核验的实际训练行为，记为 `batch-1a-actual-preserved`：训练入口保留 `torch.set_float32_matmul_precision("high")`，该设置允许 CUDA matmul 使用 TF32；PyTorch 2.1.2 下 cuDNN TF32 也保持默认开启。不得把旧文本中的 `TF32 off` 当作实际训练设置。

共同合同逐字段比较结果为 `exact_equal=true`、`mismatches={}`。因此现有 Batch 1A C0 fixed final checkpoint 可复用为 matched control；Gate-B 记录的 C0 路径为 `cloud/MMFR_E1_Batch1A_local_transfer_20260922/MMFR_E1_Batch1A_local_transfer_20260922/C0/checkpoint/update-2560.pth`，`existing_c0_checkpoint=true`。

### 13.2 R-OE-lite implementation identity

- config：`local_configs/MUSeg/DFormerv2_S_MMFR_E1_Batch1B_R_OE.py`；
- implementation：`models/roe_substitute.py` 中的 `ObservableEmptyGeometrySubstitute`，以及 `models/builder.py` 中的 `_route_roe_modal_x`；
- training integration：`utils/train.py` 的 E1 Batch 1B candidate path；
- Gate-B runner：`tools/mmfr/e1_batch1b_gateb.py`；
- substitute trainable parameters：精确 `3,302,785`；7 个 Conv weight 共 `3,301,232` 个参数，7 个 bias 共 `1,553` 个参数；
- protocol identity：`MMFR-E1-Batch1B-R-OE-lite-v1`；matched control：`MMFR-E1-Batch1A-C0`。

实现严格保持冻结设计：detector 只读取当前 raw Depth 与 $V_{\mathrm{geom}}$；RGB 只进入 substitute；非触发样本 exact bypass 且不运行 substitute；mixed batch 按样本路由；A2 reliability auxiliary 继续读取原 raw RGB、原 raw Depth、原 target，不读取 substitute 输出。

### 13.3 Gate-B 命令与最终结果

命令：

```text
PYTHONPATH=D:\\0Project\\DFormer python tools\\mmfr\\e1_batch1b_gateb.py --latency-warmup 1 --latency-repeats 2
```

Canonical artifact：`outputs/mmfr-e1-batch1b-gateb/e1-batch1b-gateb.json`；SHA-256 `35297b490c3e3eb54b5e66d3f06784688fca65038e7d09874c30c60cde820231`；`status=PASS`；`failed_checks=[]`；`official_test_included=false`；`formal_training_started=false`；duration `11.80315530000371` s。

Gate-B 直接通过的最小检查包括：

- source checkpoint、train/val split、config identity 与 official-test sealed 状态正确；
- post-build CPU/CUDA RNG state 与 C0 exact equal；第一 epoch permutation 共 `1280` 项 exact equal，digest `196564b61f9f5349dc6c4b77b7c993ccbf9c0660646bfde637fb5a87da1bc028`；
- detector mixed batch 路由为 `[False, True, False, False]`，substitute 只调用一次；nonempty 与 no-geometry 均 bypass；
- 非触发样本 segmentation Depth bitwise exact equal，未触发样本无 substitute forward；
- substitute output finite，范围 `[0.0, 127.50041961669922]`，padding exact zero，三通道复制与现有 Depth normalization exact；
- reliability 原始输入/target identity 保持，substitute 不进入 auxiliary loss；
- 所有 trainable parameter membership 精确为 1；29 个 `Geo.weight` 在 `base_decay`，14 个 SyncBN 参数在 `base_no_decay`，R-OE 7 个 Conv weight/bias 分别进入 `new_decay/new_no_decay`；
- AMP 单步 loss `0.7119939327` finite，optimizer step applied；14 个新参数梯度均 finite/nonzero，代表性前层与 head 参数均发生 update。

### 13.4 成本记录与授权边界

在本地 NVIDIA GeForce RTX 5060 Laptop GPU、batch size 1、warmup `1`、repeats `2` 下：non-trigger median latency `130.693645 ms`，trigger median latency `263.437180 ms`。training peak allocated memory 为 C0 `2,284,511,744` bytes、R-OE `3,875,250,688` bytes，allocated delta `1,517.046875 MiB`；reserved peak 为 C0 `2,428,502,016` bytes、R-OE `6,211,764,224` bytes，reserved delta `3,608 MiB`。这些是 Gate-B 成本记录，不是完整训练成本估计，也不是正式训练可行性的最终保证。

Gate-B PASS 只关闭实现资格，不授权正式训练。当前状态为 `ready-for-R-OE-formal-training-authorization`；正式训练、Quick-Val、Main-Val、T、Batch 2、云任务与 official test 均保持未启动，official test 继续 `sealed_unread`。

### 13.5 冻结的 10-condition Main-Val 门槛

正式 Main-Val 必须在另行授权后，使用 matched C0 与 R-OE-lite fixed final checkpoint，在完整 `val-dev`、冻结 `msflip-whole-original-grid-v1` 与十个条件上完成。所有差值定义为 R-OE-lite 减 C0，单位为百分点（pp）。$M_6$ 是六个单故障 mIoU 的未加权宏平均。Quick-Val 只作 screening，不承担最终 `promote/stop` 判定；本条冻结门槛本身不授权任何评价。

- **Promote：** `entire_missing@1.0` delta `>= +0.50 pp`；$M_6$ delta `>= 0`；clean delta `>= -0.25 pp`；六个单故障中没有任何 delta `< -0.50 pp`。
- **Stop：** 满足任一项即 stop：`entire_missing@1.0` delta `<= 0`；$M_6$ delta `<= -0.50 pp`；clean delta `< -0.50 pp`；六个单故障中任一 delta `< -1.00 pp`。
- **Inconclusive：** 身份与完整性均合格的完整评价既不满足 promote，也未触发 stop。
- **Blocked：** 评价未完整完成，或冻结身份/配对性断言失败；不得将 blocked 改称 inconclusive。

另外三个混合条件仍须报告，但不设置单独 promote/stop 门槛。`entire_missing@1.0` 是压力条件，不是 cause label；任何分数都不能支持 detector 识别了 hidden cause。该单 seed、单 checkpoint、单次门槛不构成统计显著性或现实部署可靠性结论。