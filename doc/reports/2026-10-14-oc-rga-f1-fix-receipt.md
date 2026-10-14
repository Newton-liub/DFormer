# OC-RGA：F1 收尾修复回执（P1 + 两项非阻断）

> 日期：2026-10-14。依据：[最终定点复核](2026-10-14-oc-rga-final-code-acceptance.md) 第 2 节 F1，以及第 3 节 R6/R8 的非阻断建议。
> 分支 `research/observation-relational-geometry`。本轮只做局部修复，未重新设计方法、未重构训练框架、未重跑 R1–R8 全量检查、未启动云端 GPU 或正式训练。
> 本文只写实际执行过的检查与结果。

## 1. 修改文件与代码位置

| 项 | 修改 | 位置 |
|---|---|---|
| **F1（训练阻断）** | 收尾几何块抽成 `epoch_geometry_summary()` / `write_geometry_logs()`，以 `model.geometry.uses_prior` 判据分流：OC 臂照常统计；基线写入显式 `skipped` 条目而非伪造概率字段，并继续进入验证、日志与 checkpoint 保存 | `research/train_oc_rga.py`（L279–L330、训练循环调用点） |
| F1 防御 | `geometry_statistics()` 先检查 `OC_BASE_FIELDS`，缺字段时抛出**具名 `ValueError`**（列出缺失字段并指向 `uses_prior`），不再以裸 `KeyError` 中断；无 `try/except` 吞错、无虚构字段 | `research/oc_rga_losses.py`（L669–L684） |
| **R6 统计口径/开销** | 关系读出改为**每 head 定长跨步取样**（`SATURATION_SAMPLES_PER_HEAD = 64`），`\|E\|>tau` 计数只在这批取样项上统计，量化与饱和率口径统一且覆盖全部 head（原先只取 head0 样本、却对 head0 全张量做归约） | `research/oc_rga.py::_axis_evidence_fn`（L925–） |
| R6 设备端累积 | `_build_stage` 的饱和计数保持 **device 标量**（不再 `float()`）；`evidence_sample_stats` 不再 `.cpu()`；训练循环以 CUDA 标量累加计数、按均匀跨步保留有界样本（`EVIDENCE_SAMPLE_ROWS = 256`），统计只在 epoch 收尾解析一次 | `research/oc_rga.py::_build_stage`、`research/oc_rga_losses.py::evidence_sample_stats/merge_evidence_stats`、`research/train_oc_rga.py`（累积块与收尾块） |
| **R8 GradScaler 回执** | 成功更新路径改为与正式训练一致的 `scaler.unscale_` → `clip` → `scaler.step` → `scaler.update`（不再用裸 `optimizer.step()`），并记录 `scaler_applied_attempts`；成功若被跳过则判失败 | `research/train_oc_rga.py::run_check` |
| 验收入口 | 新增 `--check-epoch-end`：对两种模式各跑一次真实收尾块（几何统计→几何日志→周期验证→checkpoint 保存），失败返回非零 | `research/train_oc_rga.py::run_epoch_end_check` |

## 2. F1 实际检查结果

**负控（证明 gate 是真保护）**：真实基线 builder + 32×32 输入，`bundle.base` 实测字段为 `['cell','grid','k','n','zbar']`；直接调用 OC 统计得到具名错误而非 `KeyError`：

```
geometry_statistics needs the OC probability fields
['s2','omega','f','m','delta','F','U'] but this bundle only has
['cell','grid','k','n','zbar']; gate the call on model.geometry.uses_prior
```

**`--check-epoch-end`（CPU，小合成 batch，两种模式各一次真实收尾）**：`passed=True`，`failures=[]`

| 模式 | uses_prior | geometry | 验证 | checkpoint |
|---|---|---|---|---|
| `masked_baseline` | False | `skipped`（无任何 OC 字段） | `['clean']` 正常完成 | 已写出 |
| `oc_rga` | True | 完整 OC 统计 | `['clean']` 正常完成 | 已写出 |

OC 臂 epoch 几何 JSON 含 `s_quantiles / omega_quantiles / f_quantiles / m_fraction / posterior_marginal_var / u_energy / ell_quantiles / beta_quantiles / cholesky_fallback / raw_E_quantiles / raw_E_quantile_sample_count / saturation`，其中饱和口径写入为（`same entries` 原表述已按第 6 节修正）：

```
{"saturated": …, "entries": …, "rate": …,
 "definition": "share of every per-head pre-saturation raw-E sample of the epoch
                with |E| > tau; counted on the device for all micro-batches,
                denominator = entries"}
```

**真实训练循环（真实 DeLiVER 480，micro1/accum1，`--smoke-steps 2`）**

- `masked_baseline`：**退出码 0**；写出 `geometry_stats/epoch-000.json`（内容为显式 skip 条目）、`geometry_stats.csv`、`smoke.pth`、`loss_lr.csv`；即审计复现的崩溃路径现在完整走通收尾与保存分支。
- `oc_rga`：退出码 0；几何 JSON 统计完整，`saturation = {saturated: 1170, entries: 6144, rate: 0.1904}`，`entries` 恰为 2 个 micro-batch × 3072（每 micro-batch = 4/4/8/16 heads 的取样和），证明计数确实**跨 micro-batch 累积**；当时记录的 `raw_E_quantiles = [-18.76, -0.019, 0.275]` 取自仅 2 个微批次的 smoke，**且当时分位数仍走前缀截取**；第 6 节修复后该字段改为覆盖 epoch 的有界均匀子样本估计，该 smoke 未重跑。

## 3. 两项非阻断问题

**均已处理**（改动量小，未改变训练数学）：

1. **R6 统计**：关系读出改为每 head 定长取样后，饱和率与分位数都覆盖全部 head，且同源于这套每 head 取样；**但两者样本规模不同**——饱和计数累计 epoch 内每个微批次，分位数只在保留的有界均匀子样本上估计（原“同一批取样项”的描述不实，已按第 6 节更正）。计数与样本留在 device，逐 micro-batch 不再有统计相关的 `.cpu()`；epoch 收尾统一解析一次。仍保留的每 micro-batch 主机读取只有 loss 分量/计数标量（CSV 行所需，两臂一致，非 pair 张量统计），未改动。
2. **R8 GradScaler**：成功更新的回执已收窄为真实 `scaler.step/update` 路径。CUDA 小网格验证输出 `updated 64/64 key parameters with weight decay disabled (attempts=1, scaler applied=1, AMP skipped=0)`，即无跳步时成功分支确实经过 GradScaler。

## 4. Git

- 提交：`3e6d9ee`（F1 门控 + R6 统计口径/开销 + R8 scaler 成功路径 + `--check-epoch-end`）。
- 变更文件：`research/train_oc_rga.py`、`research/oc_rga_losses.py`、`research/oc_rga.py`（源码 3 个，未改模型数学与骨干）。
- 回执提交：`bf593c4`。
- R6 日志小修提交：`e8f42d4`（`research/train_oc_rga.py`、`research/oc_rga_losses.py`），详见第 6 节。

## 5. 是否仍有阻断问题

**无新的阻断问题。** F1 已修复并在真实训练循环中由两臂分别验证；两项非阻断建议已处理。以下仍属 GPU 授权后的阶段 A 实测项，本轮未运行、也不影响 F1 关闭：

- 4090 上 480 完整前后向的峰值显存/余量、micro4×accum4 可行性、吞吐与成本。
- 640 推理显存与耗时；云端的 SwanLab、关机保险、持久任务与多 worker 数据管线实测。
- 本轮未做 CUDA 上真实 GradScaler 的跳过→恢复全序列（小网格无跳步），正式训练前 1–2 epoch 的效率检查覆盖该点。

修复后只回读本轮差异即可，不需要重跑 R1–R8 或新开一轮完整审核。

## 6. 追加修复：R6 日志采样与定义（2026-10-14 第二轮，提交 `e8f42d4`）

依据[F1 修复差异验收](2026-10-14-oc-rga-f1-fix-acceptance.md)第 4 节，只修 R6 的 epoch 采样与日志定义。未改模型数学、注意力接线、loss、优化器、训练超参数、rank20 与 120 epoch 合同；未重跑 R1–R8、F1 收尾、R8 链路或任何 GPU 检查；未启动云端资源。

| 项 | 修改 | 位置 |
|---|---|---|
| epoch 保留索引 | 删除 `total_micro // EVIDENCE_SAMPLE_ROWS` 跨步 + 收满 256 条即停的写法，改为预计算均匀索引集合 `epoch_evidence_steps(total_micro)`：覆盖 `0 … total_micro-1`、必含首尾微批次、至多 256 个且按构造互不重复。饱和计数仍累计全部微批次 | `research/train_oc_rga.py`（新增 `epoch_evidence_steps()`、训练循环累积块） |
| raw-E 二次采样 | 新增 `_uniform_bounded_sample()`：对拼好的整段 raw-E 样本做一次跨步均匀有界采样（≤ `RAW_E_QUANTILE_LIMIT = 4096` 个值），全程留在 device，最后一次性转 CPU 求分位数，替代连续 `narrow(0, keep)` 的前缀截取；其它几何字段的 `_sample_tensor()` 行为未改 | `research/oc_rga_losses.py`（新增 `_uniform_bounded_sample()`、`merge_evidence_stats()`） |
| 日志定义 | `saturation.definition` 改为“全 epoch 所有微批次的每 head 饱和前 raw-E 样本中 `\|E\| > tau` 的比例，分母为 `entries`”；`raw_E_quantiles` 明示为覆盖 epoch 的有界均匀子样本估计；新增 `raw_E_quantile_sample_count` 记录实际参与分位数计算的样本数；删除 `same entries as raw_E_quantiles` 表述 | `research/oc_rga_losses.py::merge_evidence_stats` |

**CPU 局部检查（退出码 0；无模型前后向、无 GPU、无临时脚本或测试文件）**

| 检查 | 输入 | 结果 |
|---|---|---|
| 分位数样本覆盖后段 | 3072 个 0 + 3072 个 8，`tau=4` | `raw_E_quantiles = [0.0, 4.0, 8.0]`（与全部 6144 个值的真实分位数一致），`raw_E_quantile_sample_count = 4096` |
| 饱和率与条目 | 同上（`saturated=3072`、`entries=6144`） | `saturation = {saturated: 3072.0, entries: 6144.0, rate: 0.5, threshold: 4.0}`，`definition` 不再含 `same entries` |
| 同上，按 2 个微批次的 `[2, 3072]` 形状 | `torch.stack` 的两行 | 同样得 `[0.0, 4.0, 8.0]`、样本数 4096 |
| 尾段极端值确实进入样本 | `arange(8192)` 的 `[2, 4096]` | `raw_E_quantiles = [81.9, 4095.0, 8108.1]`，99% 分位 > 8000（旧前缀截取只会看到开头） |
| 小样本不截断 | 16 个值 | 样本数 16，全部保留 |
| epoch 保留索引 | `total_micro = 996` | 256 个索引，最小 0、最大 995、无重复（旧写法最大只到 765，约 23% 尾段被丢弃） |
| 索引边界 | `n = 1/2/255/256/257/3983/3984/3985` | 数量恒为 `min(n, 256)`，首尾索引分别为 `0` 与 `n-1` |

**与最小验收的对应**：分位数约为 `[0, 4, 8]`、饱和率 `0.5`、总条目 `6144`（第 1、2 行）；996 微批次保留索引 ≤ 256、覆盖首尾且无重复（第 6、7 行）；实际分位数样本数、饱和计数与日志定义一致（第 2 行 `definition` 与 `raw_E_quantile_sample_count`）。三个修复要求（epoch 均匀保留索引、raw-E 覆盖全范围的有界均匀采样、按真实总体标注定义）均已满足。

**残留状态**：本轮三个源码以外只动本回执与 `doc/state/current.md`；上一轮遗留的三个未跟踪审核报告与 `临时/` 未被提交或修改。未在真实 DeLiVER 数据上重跑训练入口 smoke，因此新口径在真实累积路径上的 JSON 输出未实测，只由上述定点 CPU 检查覆盖。
