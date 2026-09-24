# MUSeg 当前状态与唯一实时入口

> **事实截至：** 2026-09-24 R-OE-lite v2 云端 RTX 4090 正式训练（2560/2560）完成、fixed-final checkpoint 已核验、四条件 Quick-Val 已跑完；入口修复与新版 Quick-Val 入口尚未提交。
> **用途：** 当前阶段、授权边界与恢复点。仍开放的研究选择见 [`MUSeg-open-decisions.md`](MUSeg-open-decisions.md)；v1 中止与 v2 本轮结果的历史事实见 [`Batch 1B v1 中止报告`](../../MMFR/02_evidence/report_e1_batch1b_roe_formal_training_attempt_20260924.md) 与 [`Batch 1B v2 训练与 Quick-Val 报告`](../../MMFR/02_evidence/report_e1_batch1b_roe_v2_formal_training_quickval_20260924.md)。

## 当前阶段与最近一个大动作

**阶段：** MMFR（多形式模态失效与可靠性）E1 Batch 1B，`R-OE-lite-v2-training-and-quickval-complete`。v2 用新的 protocol/run identity `MMFR-E1-Batch1B-R-OE-lite-v2` 从 A2 epoch-420 source 干净训练了 2560 次成功更新并生成 fixed-final checkpoint；这是 v1 因 CUDA OOM 中止后的独立重训，不继承 v1 的任何训练状态。

**正式训练（PASS）：** 云端 RTX 4090（主机 `cpod-1vbh7faqcauq`）上，`2026-09-24 06:46:37 UTC` 启动、用时 `3590.197` 秒（约 `59 分 50 秒`）；`attempted_steps=2560`、`completed_optimizer_steps=2560`、`skipped_optimizer_steps=0`，20/20 个 epoch 遥测一致，`exit_code=0`；无 OOM、无 NaN、无 AMP skip，AMP scale 由 `1024` 增长到 `2048`。fixed-final checkpoint `cloud/mmfr-e1-batch1b-roe-v2/R-OE-lite/development/seed-772961337/checkpoint/update-2560.pth`，SHA-256 `369d7e254acadb3bec10d4b6f362d1601291b07edc604bdb671a47591d2a5fbc`；SwanLab online run `mmfr-e1-batch1b-roe-v2-seed772961337-4090`（[`uhqkywfw`](https://swanlab.cn/@Newton_liub/DFormer-liu/runs/uhqkywfw)）。output 目录交付物：`cloud/mmfr-e1-batch1b-roe-v2/`。

**四条件 Quick-Val（`inconclusive`）：** matched control 为 Batch 1A C0 fixed-final checkpoint（SHA-256 `ca618b23d18eabb201a0d11d18da383ac99576d0feae5864e3233bda527d9a1a`，本轮未重训，其 `quickval-original-full/summary.json` 按 `e1_screening_plan.md` §109 复用）。单视图 318/318 `val-dev` 上，clean `53.46 → 53.46`（`0.00`）、`entire_missing@1.0` `48.76 → 48.77`（`+0.01`）、`spatial_dropout@0.75` `51.40 → 51.41`（`+0.01`）、`misalignment@0.75` `52.15 → 52.16`（`+0.01`）；$M_{3,\mathrm{hard}}$ `50.77 → 50.78`。按 `e1_batch1_protocol.md` §11 的 screening 阈值，既未触发 stop 也未达到 promote（`+0.50` pp），判定 `inconclusive`。触发覆盖：`entire_missing@1.0` 为 `318/318`，其余三个条件均为 `0/318`（严格 exact bypass）。**大白话说明：** v2 的显存修复确实成立——v1 在同一台 4090 上 epoch 2 就显存不足退出，v2 完整跑满一小时且一次都没跳过；但四个条件的 Quick-Val 指标相对 C0 几乎完全不动（0.00 到 +0.01 个百分点），离进入十条件 Main-Val 所需的 `+0.50` pp 门槛差得很远。

**已核验的机制证据：** 14 个 `roe_substitute.*` 参数的 AdamW `step=2272`（其余 720 个参数为 `2560`），说明 substitute 在 `2272/2560` 次更新中拿到梯度；其权重绝对值已超过 `trunc_normal_(std=0.02)` 的 ±2σ 初始化上界 `0.04`（最大 `0.1007`），确认它真的被训练；相对 C0 有 `734/812` 个共有键发生变化（`max|Δ|=0.0619`），且两个 run 的逐 epoch `clean_samples/corrupt_samples` 在 `20/20` 个 epoch 完全相同（合计 clean `6359/25600`），说明差异来自 substitute 路由而非样本顺序。四条件逐样本 mIoU 各有约一半样本与 C0 不同，但 clean 聚合值仍与 C0 完全相同，说明“约一半样本变化 + 聚合 0.00–0.01 pp”属该训练规模下 base 权重漂移的正常量级。

## 授权边界与阻塞

- **必须记录的入口阻塞与未提交修复：** 提交 `255780f` 只改了 config 的 protocol identity，`utils/train.py:460` 的入口白名单仍只接受 `...-v1`，按该 commit 原样无法启动 v2。实际执行的最小修复是在白名单中加入 v2 身份（候选、`roe_substitute.` missing 前缀、AMP/SyncBN/DDP/2560-update 等全部约束不变，对 v1 与 Batch 1A 无行为影响）：`utils/train.py` SHA-256 由 `030b2c6c65876b36dc053257ad43bc28a12b2249950f691ebefee9c769169fa5` 变为 `d9a2d5f616bc04f3fc86271228b9c1a9d3a38b40235667a3c4850859ffa0e167`，补丁存为 `cloud/mmfr-e1-batch1b-roe-v2/roe-v2-entry-fix.patch`（SHA-256 `56d46abb2d39ebba99ee2a34f72c64658b4d7d4c2e1157584d5038289376f52f`）。**该修复与新增的 Quick-Val 入口都未提交、未推送**；本轮训练与前序所有正式 run 一样以 `identity.dirty=true` 记录（`git_commit=a5293b31d467ed91943498f3a53a3ee8eb2a241f`，该 commit 含 `255780f`）。
- **新增 Quick-Val 入口无冻结资格：** 冻结的 `tools/mmfr/e1_quickval.py` 只调用 `model(rgb, depth)`，无法评价 R-OE-lite checkpoint（`models/builder.py::forward` 无条件调用 `_route_roe_modal_x`，缺 `raw_depth`/几何掩码即 fail closed）。新增 `tools/mmfr/e1_quickval_roe.py`（SHA-256 `6734cb91c82167f84cbecdfe740f7996e2808fd51e19db4d5d05f980e6c4c63e`）沿用全部冻结数值步骤，只增加路由输入与 `oe_routing` telemetry；其 `V_geom` 对无 crop/pad 的 `original-full` 视图取全 1 单位掩码，属本入口的新决定，已写入产物。该入口尚未经用户复核或其冻结资格检查。
- Batch 1B 十条件 Main-Val、Batch 2、T 与 official test 均未获授权；official test 仍为 `sealed_unread`（封存未读）。Batch 1A F-lite Main-Val 的独立上级裁决仍开放。C0 与 F-lite 的既有 Quick-Val 证据未被本轮改写。

## 下一恢复点与证据

1. 用户复核本轮两个未提交产物：入口白名单修复（决定在本地提交推送、还是在云端补提交）与新 Quick-Val 入口 `tools/mmfr/e1_quickval_roe.py`（决定接受、改写或替换实现）。
2. 若接受 `V_geom` 取全 1 的 `original-full` 口径，可继续做一次“同权重下强制 bypass”的对照，用来把 substitute 的净因果效果与 base 权重漂移分开；该对照尚未运行、也未获授权。
3. R-OE 路线的去留（是否进入十条件 Main-Val，或直接按停止处理）见 [`MUSeg-open-decisions.md`](MUSeg-open-decisions.md)。在裁决前不启动 Main-Val、Batch 2、T 或 official test。

**证据入口：** [`v2 训练与 Quick-Val 报告`](../../MMFR/02_evidence/report_e1_batch1b_roe_v2_formal_training_quickval_20260924.md)、[`v1 中止报告`](../../MMFR/02_evidence/report_e1_batch1b_roe_formal_training_attempt_20260924.md)、[`v1 Gate-B 报告`](../../MMFR/02_evidence/report_e1_batch1b_roe_gateb.md)、[`v2 最小结构调整说明`](../../MMFR/01_research/e1_batch1_protocol.md)。比对产物：`cloud/mmfr-e1-batch1b-roe-v2/quickval-comparison.json`（SHA-256 `995b6fab51cc2c7d4bbc9f6679f44a421ae30f97a9793876288c2864ee96406f`）。上一版实时状态文档已由提交 `a5293b3` 保存在 Git 历史中，未单独归档。
