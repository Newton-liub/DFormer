# MMFR E1 Batch 1A/1B Protocol Freeze

> **Protocol identity：** `MMFR-E1-Batch1A-C0-F-v1`  
> **状态：** `Batch 1A Quick-Val complete, F-lite promote, awaiting local 10-condition Main-Val`  
> **日期：** 2026-09-22  
> **性质：** Batch 1A implementation protocol、正式训练与 Quick-Val 结果与 Batch 1B 边界；Main-Val 尚未运行，计划按冻结命令迁回本地执行；不涉及 checkpoint 重选、R-OE、T 或 official test。  
> **official test：** `sealed_unread`

## 1. 协议裁决

- `Gate-A-common = PASS`；
- `Gate-A-F = PASS`；
- `R-EM-lite = retired-by-observability`；
- `Batch 1A C0/F = implementation-authorized, Gate-B-passed, training-complete-PASS`；
- `Quick-Val = authorized, completed, F-lite promote`；
- `R-OE-lite = design-frozen, implementation-not-authorized`；
- `Batch 1B R-OE = protocol-pending`。

Batch 1A 正式训练已完成且 C0/F-lite 均 PASS；Quick-Val 已完成并判定 F-lite `promote`。Main-Val 尚未运行，因此本协议当前不包含十条件效果结论；Batch 1B 与 official test 仍在边界之外。

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

本协议不实现 R-OE。substitute 的唯一冻结设计、exact bypass、数值合同、参数量与 loss 边界见 `r_oe_lite_design.md`。

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

R-OE 不自动继承原 R-EM cause-specific gate；Batch 1B 数值 gate 必须在其独立协议审核时确认。

## 12. 授权边界与恢复点

Batch 1A C0/F-lite 正式训练均已完成并 PASS，各完成 `2560/2560 successful updates` 且 `skipped=0`；4-condition Quick-Val 已在 318 条 `val-dev` 上完成并判定 F-lite `promote`。本轮没有运行 Main-Val 或 10-condition 评价，因此没有 Main-Val 指标、比较结论或 checkpoint 效果选择；也没有实现 R-OE、运行 T 或读取 official test。

**恢复点：** `awaiting local 10-condition Main-Val`。下一步只按冻结命令将 Main-Val 迁回本地执行；不修改 evaluator、不重选 checkpoint、不增加研究设计。