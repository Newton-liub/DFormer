# MMFR E1 候选筛选计划：Batch 1A/1B 分阶段冻结

> **文档角色：** E1 Batch 1 的研究合同、授权边界与恢复入口。  
> **状态：** `batch1a-gateb-passed-awaiting-training-authorization`  
> **日期：** 2026-09-21  
> **实时入口：** [`MUSeg-current-status.md`](../../../../doc/main/MUSeg-current-status.md)  
> **配套协议：** [`e1_batch1_protocol.md`](e1_batch1_protocol.md)  
> **Gate-B 报告：** [`../02_evidence/report_e1_batch1a_gateb.md`](../02_evidence/report_e1_batch1a_gateb.md)  
> **R-OE 设计：** [`r_oe_lite_design.md`](r_oe_lite_design.md)

**大白话说明：** 原 `R-EM-lite` 想识别“为什么 Depth 为空”，但当前输入只能看见“Depth 已经为空”，无法恢复隐藏原因。该路线因此正式退休，不再阻塞与它无关的 C0/F-lite。C0 与 F-lite 已完成最小实现资格检查，但还没有获准开始 20 epoch、2560 update 正式训练。

---

## 1. Gate 结构与当前裁决

### 1.1 Gate-A-common = PASS

共同字段已冻结：

- checkpoint/config/code/split identity；
- 共同基础损失 $L_{\mathrm{base}}^{A2}$；
- 29 个 `Geo.weight` optimizer coverage 修复；
- weights-only restart；
- optimizer/scheduler 与 2560 successful-update 预算；
- failure manifest、fixed final checkpoint；
- Quick-Val evaluator；
- 日志、恢复和成本规则。

### 1.2 Gate-A-F = PASS

F-lite 专属字段已冻结：

- stage 1/2/3 独立 adapter；
- `1×1 down → GELU → zero-init 1×1 up`；
- bottleneck ratio `1/4`；
- 无 normalization；
- 新增参数精确 `173152`；
- 不读取 reliability、condition、severity 或 oracle；
- 参数、显存、延迟、promotion 与 stop gate。

### 1.3 Gate-A-R-EM = retired-by-observability

`R-EM-lite` 要求从当前输入识别 cause=`entire_missing`。`../02_evidence/audit_depth_input_contract.md` 已证明 synthetic `entire_missing`、natural all-invalid crop 与 spatial dropout emptied crop 可以形成完全相同的 observable Depth。因此：

- `R-EM-lite = retired-by-observability`；
- 不继续研究 cause detector；
- 永久禁止用 corruption type、severity、hidden mask、clean Depth、`depth_valid_pre/post`、reliability target、generator RNG 或 manifest cause 抢救该路线。

### 1.4 Gate-B：Batch 1A C0/F-lite = PASS

C0 与 F-lite 已实现并通过最小 Gate-B：

- `C0: PASS`；
- `F-lite: PASS`；
- 29 个 Geo 参数全部进入 `base_decay`；
- 所有 trainable parameter optimizer membership 均为 1；
- C0 source forward identity 与 F-lite 初始 no-op 均为 exact equal；
- C0/F 均出现合法 Geo gradient/update；
- F 第一步 up 路径启动、第二步 down 路径启动；
- F 显存和 batch-1 latency 增量低于冻结上限。

这只表示代码具备按协议运行的资格，**不等于 training-authorized**。

---

## 2. Batch 1 分阶段执行

### 2.1 Batch 1A：C0 + F-lite

比较量：

$$
\Delta_F = F\text{-lite} - C0.
$$

当前状态：

- implementation：已完成；
- Gate-B：已通过；
- formal training：未授权；
- Quick-Val：未授权；
- Main-Val：未授权。

### 2.2 Batch 1B：R-OE-lite vs C0

新候选名称：

> **R-OE-lite = Observable-Empty Geometry Substitute**

其研究问题为：

$$
\text{observable no-depth state}
\rightarrow
\text{RGB-derived substitute geometry}.
$$

不是：

$$
\text{hidden corruption cause}
\rightarrow
\text{special route}.
$$

R-OE 的设计已单独写入 `r_oe_lite_design.md`，但 substitute 尚未实现，Batch 1B 完整 protocol、Gate-B 与运行授权仍为 pending。

若未来 Batch 1B 的 checkpoint、optimizer、base loss、training budget、corruption manifest、seed 和 evaluation protocol 与 Batch 1A 完全相同，可复用 Batch 1A C0 作为 matched control。任何共同训练合同发生变化，都必须重跑对应 C0。

---

## 3. Source identity 与共同起点

共同 source：

- checkpoint：`experiments/MMFR_A2_v3/checkpoints/selector-epoch-420.pth`；
- size：`321103318` bytes；
- SHA-256：`2b72eb7f28e5c4b4f52bdc43e2ac5c7298169325a412bfbcbc7a022c29822597`；
- schema：`dformer-training-checkpoint-v2`；
- model keys：`812`；
- completed/next epoch：`420/421`；
- global optimizer step：`53735`；
- embedded source commit：`d82d83482722776f5dc5059c80c34975e828e402`；
- `train-dev`：1277 条，SHA-256 `a6b15b63f6d5193e3928ea24ada25be403a48e68d1c1f9372cdbbc3fe5cd8470`；
- `val-dev`：318 条，SHA-256 `1d0719d8f64f016d48995c25ab66d4004d76b7155d9efeef7cbb7454c0dd0e83`；
- official test：`sealed_unread`。

所有候选只加载完整 model weights；旧 optimizer、scheduler、GradScaler 与 RNG state 不恢复。

---

## 4. 共同损失与 optimizer

共同基础损失保持：

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
- reliability auxiliary 不进入 backbone、decoder 或 geometry path。

四个 optimizer group 固定为：

1. `base_decay`：base LR `1e-5`、weight decay `0.01`；
2. `base_no_decay`：base LR `1e-5`、weight decay `0`；
3. `new_decay`：new LR `3e-5`、weight decay `0.01`；
4. `new_no_decay`：new LR `3e-5`、weight decay `0`。

分组规则：

- 29 个 `backbone.layers.*.blocks.*.Geo.weight` → `base_decay`；
- 14 个已覆盖 SyncBN weight/bias → `base_no_decay`；
- reliability head weight/bias 按 decay/no-decay 分组；
- F adapter Conv weight → `new_decay`；
- F adapter bias → `new_no_decay`；
- C0 的两个 new group 保留但为空；
- 每个 `requires_grad=True` 参数 membership 必须精确为 1。

---

## 5. F-lite 冻结结构

F-lite 位于 backbone 四级 tuple 返回后、decoder 前，只修改 stage 1/2/3：

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

- stage 1/2/3 独立；stage 0 不改；
- bottleneck ratio `1/4`；
- no normalization；
- down weight：`trunc_normal_(std=0.02)`；
- down bias：0；
- up weight/bias：0；
- geometry path、decoder、reliability、condition、severity、oracle 均不改。

参数量：stage 1=`8352`，stage 2=`33088`，stage 3=`131712`，合计 `173152`。

---

## 6. R-OE detector 语义

允许使用当前实际 RGB、当前实际 raw Depth 和 preprocessing 合法知道的 crop/pad geometry support。detector 只依赖当前 raw Depth 与 $V_{\mathrm{geom}}$：

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

其唯一合法语义是：

> 当前 crop 的真实图像支持域中没有任何非零 Depth observation。

它故意共同覆盖 synthetic entire missing、natural-empty crop 与 dropout-emptied input。这是方法定义，不是 detector error。不得称为 `entire-missing detector` 或 `failure-cause classifier`。

R-OE 本轮只完成设计冻结，不实现 substitute，不进入 Gate-B 或训练。

---

## 7. 正式训练合同（保留但未运行）

- single GPU、SyncBN on、DDP off、AMP on、TF32 off；
- batch size 10、workers 8、accumulation 1；
- seed `772961337`；
- 20 nominal epochs × 128 attempts = 2560 slots；
- 必须 2560 successful updates，任何 GradScaler skip 直接阻塞；
- base LR `1e-5`，new LR `3e-5`；
- warmup 128 successful updates；
- poly power `0.9`，update 2560 到 0；
- GradScaler：initial 1024、growth 2.0、backoff 0.5、growth interval 2000；
- A2 v3 curriculum virtual progress 约 `0.84 → 0.88`；
- `p_clean=0.25`、`max_specs=2`、六类 Depth corruption；
- recovery 每 640 successful updates；
- fixed final checkpoint：`update-2560.pth`；
- 不使用 selector/top-k/best/latest 选择效果。

本轮没有开始该训练。

---

## 8. Quick-Val 与 promotion

Batch 1A 若未来获准训练，Quick-Val 仍固定为：完整 318 条 `val-dev`、original-full、scale 1.0、no flip、FP32、TF32 off、原始 Label grid、fixed final checkpoint；条件为 clean、`entire_missing@1.0`、`spatial_dropout@0.75`、`misalignment@0.75`。

F-lite gate 保持：

$$
M_{3,\mathrm{hard}}=(EM+SD+Mis)/3.
$$

- promote：$\Delta_F\ge +0.50$ pp，clean `>= -0.25 pp`，每个 hard condition `>= -0.50 pp`；
- stop：$\Delta_F\le 0$，或 clean `< -0.50 pp`，或任一 hard condition `< -1.00 pp`；
- 其余合法完成：inconclusive。

R-OE 的 Quick-Val 解释见设计文档；Batch 1B 在单独协议审核前不继承 R-EM 的 cause-specific gate。

---

## 9. 当前授权边界与停止点

已完成：

- Gate-A-common PASS；
- Gate-A-F PASS；
- `R-EM-lite` retired-by-observability；
- Batch 1A C0/F 实现；
- Batch 1A Gate-B PASS；
- R-OE-lite 设计冻结。

未执行且未授权：

- 20 epoch / 2560 update 正式训练；
- Quick-Val、318 样本评价或 Main-Val；
- 云端正式任务；
- Batch 1B R-OE 实现；
- Batch 2；
- official test。

**准确恢复点：** 当前停止于 `Batch 1A C0/F Gate-B PASS`，等待上级审计和用户对 20-epoch / 2560-update 正式训练的单独授权。