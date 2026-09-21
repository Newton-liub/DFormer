# MMFR E1 Batch 1A/1B Protocol Freeze

> **Protocol identity：** `MMFR-E1-Batch1A-C0-F-v1`  
> **状态：** `batch1a-gateb-passed-training-not-authorized`  
> **日期：** 2026-09-21  
> **性质：** Batch 1A implementation protocol 与 Batch 1B 边界；不是训练授权。  
> **official test：** `sealed_unread`

## 1. 协议裁决

- `Gate-A-common = PASS`；
- `Gate-A-F = PASS`；
- `R-EM-lite = retired-by-observability`；
- `Batch 1A C0/F = implementation-authorized, Gate-B-passed, training-not-authorized`；
- `R-OE-lite = design-frozen, implementation-not-authorized`；
- `Batch 1B R-OE = protocol-pending`。

`implementation-authorized` 与 Gate-B PASS 都不等于 `training-authorized`。

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

## 10. 正式训练合同（未授权、未运行）

- single GPU；batch 10；workers 8；accumulation 1；
- SyncBN on；DDP off；AMP on；TF32 off；
- seed `772961337`；
- 20 nominal epochs × 128 attempts = 2560 slots；
- 必须 2560 successful updates；
- 任一 GradScaler skip 直接 blocked；
- base LR `1e-5`；new LR `3e-5`；
- weight decay `0.01`，bias/norm no-decay；
- 128 successful-update linear warmup；
- poly power `0.9`，update 2560 到 0；
- GradScaler initial 1024、growth 2.0、backoff 0.5、growth interval 2000；
- A2 v3 virtual curriculum 约 `0.84 → 0.88`；
- `p_clean=0.25`，`max_specs=2`，六类 Depth corruption；
- recovery 每 640 successful updates；
- fixed final checkpoint `update-2560.pth`；
- selector/top-k/best/latest selection 关闭。

## 11. Quick-Val（未授权、未运行）

Batch 1A 若获正式训练授权，评价固定为：318 条完整 `val-dev`、original-full、scale 1.0、no flip、FP32、TF32 off、原始 Label grid、fixed final checkpoint only。

条件：clean、`entire_missing@1.0`、`spatial_dropout@0.75`、`misalignment@0.75`。

F-lite：

$$
M_{3,\mathrm{hard}}=(EM+SD+Mis)/3.
$$

- promote：$\Delta_F\ge +0.50$ pp；clean `>= -0.25 pp`；每个 hard condition `>= -0.50 pp`；
- stop：$\Delta_F\le 0$，或 clean `< -0.50 pp`，或任一 hard condition `< -1.00 pp`；
- 其他合法完成：inconclusive。

R-OE 不自动继承原 R-EM cause-specific gate；Batch 1B 数值 gate 必须在其独立协议审核时确认。

## 12. 授权边界与恢复点

本轮没有运行 20 epoch、2560 updates、Quick-Val、318 样本评价、Main-Val、云端正式训练或 official test，也没有实现 R-OE。

**恢复点：** 当前停止于 `Batch 1A C0/F Gate-B PASS`，等待上级审计和用户对正式 20-epoch / 2560-update 训练的单独授权。