# MMFR E1 Batch 1B R-OE-lite v2 云端正式训练与四条件 Quick-Val 结果

- 日期：2026-09-24
- 训练身份：`MMFR-E1-Batch1B-R-OE-lite-v2`，seed `772961337`
- 云端 GPU：NVIDIA GeForce RTX 4090（24564 MiB），主机 `cpod-1vbh7faqcauq`
- Git HEAD：`a5293b31d467ed91943498f3a53a3ee8eb2a241f`（含 v2 commit `255780fbbe16a8b95cff833e99752afdaaf31841`），工作区 `dirty=true`（见下文入口修复与 Quick-Val 新入口）
- 输出目录：`cloud/mmfr-e1-batch1b-roe-v2/R-OE-lite/development/seed-772961337`
- SwanLab online run：[`mmfr-e1-batch1b-roe-v2-seed772961337-4090`](https://swanlab.cn/@Newton_liub/DFormer-liu/runs/uhqkywfw)
- 源 checkpoint：A2 epoch-420，SHA-256 `2b72eb7f28e5c4b4f52bdc43e2ac5c7298169325a412bfbcbc7a022c29822597`（启动前复核一致）

## 1. 前置核验与一个入口阻塞（必须记录）

最小核对全部通过：GPU 为 RTX 4090；HEAD 包含 `255780f`；v2 config 与 `models/roe_substitute.py` 可读取；A2 epoch-420 checkpoint 可读且 SHA-256 与冻结身份一致；MUSeg 数据（`/root/rivermind-data/dataset/MUSeg_DFormer` 与 `dev-v1` 三个 split）可读。

**阻塞：** 提交 `255780f` 只把 config 的 protocol identity 改成 `MMFR-E1-Batch1B-R-OE-lite-v2`，但 `utils/train.py:460` 的入口白名单仍只接受 `...-v1`，直接启动报 `train.py: error: unsupported E1 Batch 1 protocol identity 'MMFR-E1-Batch1B-R-OE-lite-v2'`。因此按该 commit 原样，v2 身份无法启动。远端 `origin/perf/mmfr-a2-v3-pipeline-opt1` 在核验时为同一 commit，没有后续修复，故不是云端落后。

**实际执行的最小修复（未提交、未推送）：** 在 `utils/train.py` 的白名单分支加入 v2 身份，候选、`roe_substitute.` missing 前缀、Batch 1B 全部约束（AMP/SyncBN on、DDP off、`--gpus 1`、2560 updates、weights-only restart）完全不变，对 v1 与 Batch 1A 路径无行为影响。

- `utils/train.py` 修改前 SHA-256 `030b2c6c65876b36dc053257ad43bc28a12b2249950f691ebefee9c769169fa5`，修改后 `d9a2d5f616bc04f3fc86271228b9c1a9d3a38b40235667a3c4850859ffa0e167`。
- 补丁留档：`cloud/mmfr-e1-batch1b-roe-v2/roe-v2-entry-fix.patch`，SHA-256 `56d46abb2d39ebba99ee2a34f72c64658b4d7d4c2e1157584d5038289376f52f`。
- 该修复只让已批准的 v2 身份可被入口接受，未改 batch size、梯度累积、AMP、LR、loss、corruption、curriculum、网络结构或输入尺寸。

## 2. 三步 GPU 验证（`R-OE-lite v2 GPU memory PASS`）

用完整正式配置在干净进程中连续执行 3 次成功 optimizer update（额外参数 `--max-train-iters 3 --log-interval 1`，`--swanlab-mode disabled`）：

| 项目 | 结果 |
| --- | --- |
| 成功更新 | **3/3**（`attempted_steps=3`、`completed_optimizer_steps=3`、`skipped_optimizer_steps=0`） |
| CUDA OOM | 无；进程正常退出（`exit_code=0`，`short run completed normally`） |
| loss | iter1 `0.3163`、iter2 `0.3459`、iter3 `0.2115`；均 finite，`amp_scale=1024.0` |
| peak allocated | **20385 MiB**（iter1 `20139`、iter2 `19084`、iter3 `20385`） |
| peak reserved | **23016 MiB**（iter1 `23004`、iter2/3 `23016`），`free=601/24110 MiB`，`free_ratio=0.025` |
| 触发样本数 | iter1 `3`、iter2 `0`、iter3 `3`（每个 batch 10 个样本） |
| v2 最后一次全分辨率插值输入 | `shape=[3, 1, 120, 160]`、`dtype=torch.float16`，`size=(480, 640)`、`mode=bilinear`、`align_corners=False` |

**与 v1 的对比边界：** v1 日志唯一一条显存遥测是 epoch 1 末步 `allocated=21375 MiB / reserved=22756 MiB / free=861 MiB`，发生在低触发区间；v1 实际 OOM 发生在 epoch 2 iter 40，那一刻的计数没有落盘。因此三步验证给出的 `allocated` 比 v1 记录值低约 990 MiB、`reserved` 高约 260 MiB，**不足以单独证明“显存明显低于 v1”**；判定依据是 v2 已彻底移除 v1 在该路径上的全部全分辨率算子（v1 在插值后又做了 122 通道的 3×3 全分辨率卷积与 1×1 head，v2 只插值单通道 logits），使该处的触发相关瞬时显存从每触发样本数百 MB 降到约 0.03 MB。

**这三项非日志证据的取得方式：** 用一个仓库外的 `sitecustomize` 探针（`PYTHONPATH` 加载，不改仓库、不改训练命令行）只记录 v2 内部单通道 `F.interpolate` 的输入 `shape`/`dtype` 与 `last_roe_route` 触发计数；探针与原始日志留档为 `cloud/mmfr-e1-batch1b-roe-v2/probe/sitecustomize.py`（SHA-256 `fbba31ac4c091c96911de45f87b2d33adb8d0a5fcbf9961b515eeedc1b38a589`）、`.../gpucheck.log`（SHA-256 `d088cb807c900905e8c7ded4f37d48d25d2c2185a8c42b748da6d48b76900369`）与 `.../roe-v2-interpolate-and-trigger-evidence.json`（SHA-256 `3cd57696733dcfcc58fbab21e9413e1a2ba0bb178f57e7bd176fd7c380c3428d`）。峰值显存取自训练器既有的 `max_memory_allocated`/`max_memory_reserved` 遥测，未新增 instrumentation。

**v1 失败点的实际穿越：** 正式训练顺利通过 epoch 2 iter 40（v1 的 OOM 位置），全程 `reserved` 稳定在 `23016 MiB`，没有再次增长。

## 3. 正式训练（PASS）

- 启动：`2026-09-24 06:46:37 UTC`，detached `screen` 会话 `mmfr-e1-batch1b-roe-v2`，`LOCAL_RANK=0`，单 GPU，DDP off，AMP on，SyncBN on，batch size 10，workers 8。
- 更新计数：`attempted_steps=2560`、`completed_optimizer_steps=2560`、`skipped_optimizer_steps=0`，20/20 个 epoch 遥测均为 `attempted == completed` 且 `skipped=0`，`exit_code=0`。
- 总训练时间：`duration_seconds=3590.197`（约 `59 分 50 秒`）。
- 异常：无 CUDA OOM、无 NaN、无 GradScaler skip；AMP scale 由 `1024` 稳定增长到 `2048`。
- loss 走势：epoch 1–20 的 epoch 末 batch loss 在 `0.087–0.448` 之间波动，epoch 内 running `total_loss` 从 `0.2039` 微降到 `0.1935`，整体平稳无发散。
- fixed-final checkpoint：`.../checkpoint/update-2560.pth`，size `360801864`，SHA-256 `369d7e254acadb3bec10d4b6f362d1601291b07edc604bdb671a47591d2a5fbc`；`latest.pth` SHA-256 `ebcb69b4040b484099d6c3a3f4d38f6bc5d9dfd45639da8a246508057c9816f0`。
- checkpoint 元数据：`schema_version=dformer-training-checkpoint-v2`、`completed_epoch=20`、`next_epoch=21`、`global_optimizer_step=2560`、model keys `826`（C0 的 `812` 加 14 个 `roe_substitute.*`）。

**R-OE-lite 是否真的参与了训练（直接核验）：**

- 优化器仍是 4 组；组 2/3 各 7 个参数、`lr_scale=3.0`，即 14 个 `roe_substitute.*` 参数。
- 优化器状态里这 14 个参数的 AdamW `step=2272`，其余 720 个参数为 `step=2560`：substitute 在 `2272/2560` 次更新中拿到梯度，在另外 288 次更新中整批无触发样本。
- substitute 权重绝对值超过 `trunc_normal_(std=0.02)` 的 ±2σ 初始化上界 `0.04`（`d2.weight` `0.1007`、`b2.weight` `0.0881`、`e1.weight` `0.0667`、`head.weight` `0.0655`），说明它确实被训练出初始化范围，而非停留在初始值。
- 相对 C0：812 个共有键中 `734` 个发生变化，`max|Δ|=0.0619`（`backbone.layers.2.downsample.norm.running_var`）。两个 run 的逐 epoch `clean_samples/corrupt_samples` 在 `20/20` 个 epoch 上完全相同（合计 clean `6359/25600`），说明 v2 的 RNG 隔离生效、样本排列与 C0 一致，因此权重差异来自 substitute 路由而不是数据顺序。

## 4. 四条件 Quick-Val

**matched control：** Batch 1A C0 fixed-final checkpoint `ca618b23d18eabb201a0d11d18da383ac99576d0feae5864e3233bda527d9a1a`，复用其既有 `quickval-original-full/summary.json`（`e1_screening_plan.md` §109 允许在该批评价协议与配置完全相同时复用）；该 control 记录的平台与本次一致（`compushare` Linux、RTX 4090、torch `2.1.2+cu118`、`base_rng_seed=2026091401`、`reset-per-unit`、FP32/TF32 off）。**未重训 C0**，其 checkpoint 已在 `cloud/mmfr-e1-batch1a-v1/C0/development/seed-772961337/checkpoint/update-2560.pth` 与 `cloud/MMFR_E1_Batch1A_local_transfer_20260922/C0/checkpoint/update-2560.pth` 两处找到且 SHA-256 与冻结值一致。

**新入口（必须记录为偏离）：** 冻结的 `tools/mmfr/e1_quickval.py` 只调用 `model(rgb, depth)`，而 R-OE-lite checkpoint 的 `models/builder.py::forward` 会无条件调用 `_route_roe_modal_x`，缺少 `raw_depth` 与几何掩码时 fail closed，因此该入口无法评价 R-OE-lite。按 `r_oe_lite_design.md` §11–§12 已冻结的推理输入与 telemetry 要求，新增薄入口 `tools/mmfr/e1_quickval_roe.py`（SHA-256 `6734cb91c82167f84cbecdfe740f7996e2808fd51e19db4d5d05f980e6c4c63e`）：condition 定义、corruption 播种、`original-full` 输入契约、FP32/TF32-off、logits 回原网格、RNG 策略、confusion 与指标全部沿用冻结模块，只新增两项——把当前 corrupted `raw_depth`（uint8/255）与 `V_geom` 传入 forward，以及汇总 `last_roe_route` telemetry。**该入口没有任何既有冻结资格，且其中 `V_geom` 取值为新决定**：`original-full` 视图无 crop/pad，故按设计对 `V_geom` 的“几何（padding）有效区”定义取全 1 单位掩码，已写入产物 `v_geom_definition`。它不读取 cause、severity 或生成器标签。

**结果（mIoU %，单视图 318/318 `val-dev`）：**

| condition | C0 | R-OE-lite v2 | delta |
| --- | ---: | ---: | ---: |
| clean | 53.46 | 53.46 | 0.00 |
| entire_missing@1.0 | 48.76 | 48.77 | +0.01 |
| spatial_dropout@0.75 | 51.40 | 51.41 | +0.01 |
| misalignment@0.75 | 52.15 | 52.16 | +0.01 |

$M_{3,\mathrm{hard}}$ `50.77 → 50.78`（$\Delta=+0.01$ pp）；clean delta `0.00` pp。按 `e1_batch1_protocol.md` §11 的 screening 规则，$\Delta_{M_{3,\mathrm{hard}}}\le 0$ 未触发、`>= +0.50` 也未达到，判定为 **`inconclusive`**。

**路由 telemetry（R-OE-lite v2 侧）：**

| condition | 触发样本 | substitute 前向 | 非空 exact bypass | 无几何 exact bypass |
| --- | ---: | ---: | ---: | ---: |
| clean | 0/318 | 0 | 318 | 0 |
| spatial_dropout@0.75 | 0/318 | 0 | 318 | 0 |
| misalignment@0.75 | 0/318 | 0 | 318 | 0 |
| entire_missing@1.0 | 318/318 | 318 | 0 | 0 |

**逐样本观察：** 四个条件中 R-OE-lite 与 C0 的逐样本 mIoU 都有约一半样本不同（clean `154/318`、SD `159/318`、Mis `141/318`、EM `156/318`），confusion matrix 也不同。clean、SD、Mis 三条件路由是严格 exact bypass，其逐样本差异**只能**来自 base network 的权重漂移；clean 的聚合值仍与 C0 完全相同，说明“约一半样本变化 + 聚合 0.00–0.01 pp”是本训练规模下 base 漂移的正常量级。因此 EM 的 `+0.01` pp 与 base 漂移不可区分，**本轮没有做“同权重下强制 bypass”的对照**，不能据此宣称 substitute 的净因果效果恰为 0。

**产物：**

- `cloud/mmfr-e1-batch1b-roe-v2/quickval-comparison.json`，SHA-256 `995b6fab51cc2c7d4bbc9f6679f44a421ae30f97a9793876288c2864ee96406f`
- `.../R-OE-lite/development/seed-772961337/quickval-original-full/{clean,spatial_dropout_075,misalignment_075,entire_missing_100}/metrics.json` 与 `summary.json`
- `cloud/mmfr-e1-batch1b-roe-v2/console.log`（含训练与 SwanLab 记录）、`launch.sh`、`run_train.sh`、`roe-v2-entry-fix.patch`

## 5. 边界与未完成项

- 未运行十条件 Main-Val、Batch 2、T 或 official test；official test 仍为 `sealed_unread`，Quick-Val 产物均记录 `official_test_included=false`。
- 未重跑 Gate-B、未重建审核包、未重训 C0、未重跑 C0 Quick-Val。
- 上述 Quick-Val 是单视图 screening 证据，不得与十视图数字直接比较，也不产生 Batch 1B 的 promote/stop 裁决。
- 入口修复与新 Quick-Val 入口都**未提交、未推送**；工作区 `dirty=true` 已记录在 `run_config.json` 的 `identity` 中。
- Batch 1A F-lite Main-Val 的独立上级处置、Batch 1A C0 与 F-lite 既有 Quick-Val 证据均未被本轮改写。

**大白话说明：** v2 的显存修复确实生效——同一台 4090 上 v1 在 epoch 2 就因显存不足退出，v2 完整跑满 2560 次更新、一次都没跳过，用时约一小时，最终权重已生成并核验，substitute 也确认真正参与了训练。但四个条件的 Quick-Val 指标相对 C0 几乎完全不动（0.00 到 +0.01 个百分点）；虽然约一半样本的预测发生了变化，聚合提升仍在噪声量级。按冻结的 screening 规则这是 `inconclusive`，但距离进入十条件 Main-Val 所需的 `+0.50` pp 门槛差得很远。
