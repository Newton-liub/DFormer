# MMFR E1 候选筛选计划：Batch 1A/1B 分阶段冻结

> **文档角色：** E1 Batch 1 的研究合同、授权边界与恢复入口。  
> **状态：** `Batch 1B R-OE-lite Gate-B PASS; awaiting formal-training authorization`  
> **日期：** 2026-09-23  
> **实时入口：** [`MUSeg-current-status.md`](../../doc/main/MUSeg-current-status.md)  
> **配套协议：** [`e1_batch1_protocol.md`](e1_batch1_protocol.md)  
> **Gate-B 报告：** [`../02_evidence/report_e1_batch1b_roe_gateb.md`](../02_evidence/report_e1_batch1b_roe_gateb.md)  
> **R-OE 设计：** [`r_oe_lite_design.md`](r_oe_lite_design.md)

**大白话说明：** 原 `R-EM-lite` 想识别“为什么 Depth 为空”，但当前输入只能看见“Depth 已经为空”，无法恢复隐藏原因。该路线因此正式退休，不再阻塞与它无关的 C0/F-lite。C0 与 F-lite 正式训练均已 PASS，各完成 `2560/2560 successful updates` 且 `skipped=0`；4-condition Quick-Val 已完成，F-lite 相对 C0 的 $\Delta_F=+0.96$ pp 满足冻结 promote 门槛。十条件 Main-Val 已于 2026-09-23 在本地完成，两侧各 10/10 条件、身份与配对性断言 0 失败；十视图口径下 F-lite 未复现 Quick-Val 优势（$M_6$ 为 `55.2883` 对 `54.8917`，$\Delta_F=-0.3966$ pp；clean 为 `56.69` 对 `56.02`），且本轮未为 Main-Val 预注册数值门禁，故处置待上级裁决。随后 Batch 1B 已按冻结设计完成 R-OE-lite implementation 与最小 Gate-B，最终 `PASS`；这只表示代码可以公平进入正式训练，不代表正式训练或效果评价已经开始，当前等待正式训练授权。

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
- formal training：C0/F-lite 均已完成并 PASS；
- Quick-Val：已授权、已完成，F-lite `promote`；
- Main-Val：已完成（本地，2026-09-23；两侧各 10/10 条件、身份与配对性断言 0 失败，结果为描述性、无预注册门禁）。

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

R-OE-lite 已按冻结设计完成实现并通过最小 Gate-B。Gate-B 确认 detector 只读取当前 raw Depth 与 $V_{\mathrm{geom}}$，非触发样本 exact bypass，mixed batch 按样本路由，substitute 参数精确为 `3,302,785`，共同 C0 的 post-build RNG 与首 epoch permutation 保持 exact equal，optimizer 与梯度路径完整。Batch 1B 仍未正式训练；Gate-B PASS 不自动授予 formal training、Quick-Val、Main-Val 或云任务授权。

若未来 Batch 1B 的 checkpoint、optimizer、base loss、training budget、corruption manifest、seed 和 evaluation protocol 与 Batch 1A 完全相同，可复用 Batch 1A C0 作为 matched control。Gate-B 已直接核验共同配置 `exact_equal=true`、RNG identity、sampler identity 与 existing C0 checkpoint 可复用。

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

## 7. 正式训练结果（已完成，C0/F-lite 均 PASS）

- single GPU、SyncBN on、DDP off、AMP on；
- **TF32 勘误：** 原合同字段 `TF32 off` 与实际训练入口不一致。两次正式训练均保留 `utils/train.py` 中既有的 `torch.set_float32_matmul_precision("high")` 行为，实际允许 CUDA matmul TF32，cuDNN TF32 也保持 PyTorch 2.1.2 默认开启；本轮接受该既有行为，不重跑；
- batch size 10、workers 8、accumulation 1；
- seed `772961337`；
- 20 nominal epochs × 128 attempts = 2560 slots；
- C0 与 F-lite 均完成 `2560/2560 successful updates`，`skipped=0`，正式训练判定均为 `PASS`；
- base LR `1e-5`，new LR `3e-5`；
- warmup 128 successful updates；
- poly power `0.9`，update 2560 到 0；
- GradScaler：initial 1024、growth 2.0、backoff 0.5、growth interval 2000；
- A2 v3 curriculum virtual progress 约 `0.84 → 0.88`；
- `p_clean=0.25`、`max_specs=2`、六类 Depth corruption；
- recovery 每 640 successful updates；
- C0 final checkpoint：`cloud/mmfr-e1-batch1a-v1/C0/development/seed-772961337/checkpoint/update-2560.pth`，SHA-256 `ca618b23d18eabb201a0d11d18da383ac99576d0feae5864e3233bda527d9a1a`；
- F-lite final checkpoint：`cloud/mmfr-e1-batch1a-v1/FLite/development/seed-772961337/checkpoint/update-2560.pth`，SHA-256 `ea9319e5abe55b996470ee0a75bd63b887241ef834a3b50f145b5b7d4aabd98d`；
- 不使用 selector/top-k/best/latest 选择效果；
- **DataLoader shuffle 记录：** 两个 run 的样本 shuffle 顺序存在轻微差异，原因是 F-lite adapter 初始化额外消耗全局 torch RNG；本轮接受为 screening-level 随机性差异，不修改采样器、不重跑训练。

---

## 8. Quick-Val 与 promotion

Batch 1A Quick-Val 已执行完毕，评价口径固定为：完整 318 条 `val-dev`、original-full、scale 1.0、no flip、FP32、TF32 off、原始 Label grid、fixed final checkpoint；条件为 clean、`entire_missing@1.0`、`spatial_dropout@0.75`、`misalignment@0.75`。

**执行入口（新增，需记录为偏离）：** 既有冻结工具中没有“单视图 `original-full` + four-condition 计分”的 runner（`tools/evaluate_museg_checkpoint.py` 无 condition 计分，`tools/evaluate_museg_10condition.py` 硬编码十视图）。经用户单独裁决新增 `tools/mmfr/e1_quickval.py`（SHA-256 `928c4229d937552e00128e9291b915204291f58c1e80b4b5a005efa9365e4073`），为薄复用层：前向与指标调用冻结 evaluator，condition/corruption/样本读取调用冻结 ten-view evaluator，RNG 沿用冻结 `load_eval_model` 约定。输入契约已用 `--self-check` 证明与冻结 `MUSegPostEvalDataset` 在 `318/318` 样本上逐位相等。

**结果（mIoU %，单视图）：** C0 clean `53.46` / SD@0.75 `51.40` / Mis@0.75 `52.15` / EM@1.0 `48.76`；F-lite clean `54.15` / SD@0.75 `52.39` / Mis@0.75 `52.63` / EM@1.0 `50.17`；delta `+0.69 / +0.99 / +0.48 / +1.41` pp；$M_{3,\mathrm{hard}}$ `50.77 → 51.73`，$\Delta_F=+0.96$ pp；判定 `promote`。证据见 `cloud/mmfr-e1-batch1a-v1/quickval-comparison.json`（SHA-256 `1f00d4face8587183c2e235eac689cb6c77556dc0b6c8636d95a8c5471176f7f`）与两侧 `quickval-original-full/`。该结果是单视图口径 screening 证据，不得与十视图数字直接比较。

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
- Batch 1A C0/F-lite 正式训练均 PASS，各完成 `2560/2560 successful updates` 且 `skipped=0`；
- 4-condition Quick-Val 完成，F-lite 相对 C0 判定 `promote`（$\Delta_F=+0.96$ pp）；
- 两个 fixed final checkpoint 及 SHA-256 已记录；
- R-OE-lite 设计、Batch 1B protocol 与实现；
- Batch 1B 最小 Gate-B PASS；
- C0 reuse fairness：RNG identity、sampler identity、共同配置 identity 和 existing C0 checkpoint 均 PASS。

未执行或尚未形成结果：

- 10-condition Main-Val：已完成（本地，2026-09-23）。$M_6$ 为 C0 `55.2883`、F-lite `54.8917`，即 $\Delta_F=-0.3966$ pp；clean 为 `56.69` / `56.02`。本轮未为 Main-Val 预注册数值门禁，故该结果不构成 F-lite 去留判定，处置待上级裁决。详细证据见 `doc/reports/2026-09-22-mmfr-e1-batch1a-c0-flite-mainval.md`；
- Batch 1B R-OE-lite 20 epoch / 2560 successful updates 正式训练；
- Batch 1B Quick-Val、Main-Val、Batch 2、T；
- official test（保持 `sealed_unread`）。

**准确恢复点：** `ready-for-R-OE-formal-training-authorization`。Batch 1B 实现与 Gate-B 已通过，但 R-OE-lite 正式训练仍需单独授权；训练授权不自动包含 Quick-Val、Main-Val、云任务、Batch 2、T 或 official test。Batch 1A 十条件 Main-Val 的上级处置独立待审。official test 继续 `sealed_unread`。

## 10. Batch 1B Gate-B 与冻结 Main-Val 门槛

### 10.1 Gate-B 身份与结论

R-OE-lite 实现和最小 Gate-B 已于 2026-09-23 完成。canonical JSON 为 `outputs/mmfr-e1-batch1b-gateb/e1-batch1b-gateb.json`，SHA-256 `35297b490c3e3eb54b5e66d3f06784688fca65038e7d09874c30c60cde820231`，`status=PASS`、`failed_checks=[]`、`official_test_included=false`、`formal_training_started=false`。完整结果见 [`../02_evidence/report_e1_batch1b_roe_gateb.md`](../02_evidence/report_e1_batch1b_roe_gateb.md)。

共同字段 exact equal；post-build CPU/CUDA RNG 与 C0 exact equal；1280 项第一 epoch permutation digest 为 `196564b61f9f5349dc6c4b77b7c993ccbf9c0660646bfde637fb5a87da1bc028`。Gate-B 确认 mixed-batch 路由、非触发 exact bypass、substitute 输出/padding/Depth normalization、reliability auxiliary 分流、optimizer membership 与一次 AMP 更新路径。substitute 可训练参数精确 `3,302,785`。这些证据只证明最小实现资格，不是训练、性能或 batch-size-10 容量结论。

### 10.2 10-condition Main-Val gate

正式 Main-Val 需要单独授权，评价完整 318 条 `val-dev` 的十个 condition，差值均为 R-OE-lite 减 matched C0，单位为 pp；$M_6$ 为六个单故障 mIoU 的未加权宏平均。Quick-Val 仅用于 screening，不作最终 `promote/stop` 判定；`entire_missing@1.0` 是 observable-empty stress condition，不能据此声称 detector 识别 hidden cause。

- **Promote：** `entire_missing@1.0` delta `>= +0.50` pp；$M_6$ delta `>= 0`；clean delta `>= -0.25` pp；六个单故障中无一 delta `< -0.50` pp。
- **Stop：** `entire_missing@1.0` delta `<= 0`，或 $M_6$ delta `<= -0.50` pp，或 clean delta `< -0.50` pp，或任一单故障 delta `< -1.00` pp。
- **Inconclusive：** 合法完整评价不满足 promote，且没有触发 stop。
- **Blocked：** 评价未完成或身份/配对性检查失败，不归入 inconclusive。

三个混合条件照常报告但无独立门槛。该数值 gate 已冻结，不代表任何正式训练或评价已获授权。