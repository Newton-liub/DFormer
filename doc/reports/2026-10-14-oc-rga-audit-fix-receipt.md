# OC-RGA v0.3 训练前审核 R1–R8 修复回执

> 日期：2026-10-14。执行对象：执行模型。依据：[正式训练前独立代码审核](2026-10-14-oc-rga-pretraining-code-audit.md)。
> 分支 `research/observation-relational-geometry`；修复提交 `0e4498ff326f72feeda4463073ec5b6be96636fb`，父提交 `bbf8630095497ffcac24ecf60d22ab8373d042c3`，已推送 `origin`（`bbf8630..0e4498f`，快进，无 force push）。
> 本轮按审核报告 R1–R8 定点修复实现与验收入口；未重新设计网络、未改 rank20 / 三种 loss / 120 epoch 日程 / 匹配训练协议，未启动 GPU 或正式训练，未重新做六大类全面审计。
> 本文只写实际执行过的检查与结果；未运行项在第 5 节列出。

## 1. R1–R8 修改与源码位置

| 项 | 修改 | 源码位置 |
|---|---|---|
| R1 基线 head decay | 显式基线组装路径按 W/H/full 把池化关系先乘一次 `Geo.decay`，再交给 `GeoPriorGen`（后者只应用 signed `weight[1]`），使 `w_d·decay_h·\|Δȳ\|·m_i m_j` 成立且 signed 系数只应用一次；未改 `GeoPriorGen` 公共外部关系语义 | `models/encoders/DFormerv2.py::RGBD_Block.observational_research`（L538、L560、L584） |
| R2 概率精度边界 | 新增 `probability_dtype` / `autocast_off`；`bounded_std`/`bounded_mean`/`window_basis`/`_column_basis`/`merge_2x2`/`_marginal_relation`/`axis_relation_variance`（Gram）/`relational_log_kernel`/`_axis_evidence_fn`/`joint_low_rank_nll` 全部在关闭 autocast 下按概率 dtype 计算；`condition_low_rank` 的 `work` 由全部输入决定且输出不再留在 AMP dtype；`RGBGeometryPrior.forward` 把原始头输出转概率 dtype | `research/oc_rga.py`（L127、L135、L154、L283、L467、L521、L592、L925、L1356） |
| R2 附带 `_quantiles` | 数据与分位网格统一为 FP32，并用 `narrow` 视图采样，不再对整张 pair 张量做 reshape 拷贝 | `research/oc_rga_losses.py::_quantiles/_sample_tensor` |
| R3 矩形主导孔洞 | 新增 `_rectangle_holes`（按剩余预算封顶地持续新增/局部扩展矩形）+ `_trim_rectangle_boundary`（只沿并集边界剥离到精确整数）；删除旧的“6 窗 + 全图随机补足”路径；`pixel_missing` 仍为独立随机抽样 | `research/oc_rga_data.py`（L154、L177）；调用点 `degrade_depth()` 与 `research/evaluate_oc_rga.py::apply_condition` |
| R4 local_value_shift | 先施加局部数值偏移，再用独立 mask **替换**异常像素（三通道一致），O 对非零污染保持；条件名统一为 `local_value_shift`，`localshift` 保留为 CLI 别名 | `research/evaluate_oc_rga.py`（L90、L65）；配置 `local_configs/research/OC_RGA_DeLiVER.py`（L116） |
| R5 随机性与恢复 | 验证外层完整保存/恢复 Python/NumPy/CPU/CUDA RNG，每图在 `torch.random.fork_rng()` 内重设 seed；DataLoader 使用独立 `loader_generator` base seed；解析后的 seed 一次性写回 `config.seed`；恢复合同补齐 holdout 稀疏概率/上下限、全部故障强度与局部窗范围、cell/rank/head 结构；缺 optimizer/scaler/RNG/contract/config/args 的 payload 明确拒绝；非默认 cell/rank 明确拒绝，head 数取自实际构造的 backbone | `research/evaluate_oc_rga.py::evaluate_conditions`；`research/oc_rga_data.py`（L621、build_train_loader/build_eval_loader）；`research/train_oc_rga.py`（L52、L95、L486）；`research/oc_rga_model.py`（L181、L182） |
| R6 GPU 开销与日志 | 辅助上下文改为单个 coarse cell mask 一次展开成像素 mask 后再 `where`；stage0/1 关系构造走非重入 checkpoint；几何统计改为定长采样 + epoch 级汇总（不再每 micro-batch 全量分位与同步）；raw E 改读饱和前的真实值并统计 `\|E\|>tau`；更新级 loss/计数改为累积组均值与组内总和 | `research/oc_rga_losses.py`（L126、L600、L651）；`research/oc_rga.py::RelationalEvidencePyramid.evidence`/`_axis_evidence_fn`；`research/train_oc_rga.py`（L714、L820、L844） |
| R7 padding 与窗质心 | 评价先按真实 grid 建 S，再 S/O 右下 pad 0；RGB 先 normalize 再 pad 0；`_support_centroids` 按真实未 pad 像素 extent 归一化，只有整个图像轴为单像素时才取 0.5 | `research/oc_rga_data.py::OCRGAEvalDataset.__getitem__`；`research/oc_rga.py`（L283） |
| R8 验收入口 | holdout 检查改为改变目标整格的**实际观测**（Depth 与 O）后重建辅助上下文，比较 `y/m/f/a_obs`、完整 69 通道 precision 输入、`omega2` 与辅助后验；保留非目标正控与“模型输出不依赖 reference”；`--check` 走真实累积/AMP/GradScaler/unscale/clip，断言关键参数梯度非零且 weight decay 关闭后真实更新；`--check`/`--leak-check` 失败返回非零 | `research/oc_rga_losses.py::holdout_leak_check`（L426）；`research/train_oc_rga.py::run_check`（L279） |

## 2. 实际执行的检查与结果

全部命令使用 `D:\2Env\anaconda\envs\dformer\python.exe -X utf8`，数据根 `D:/0Project/dataset`。

| 检查 | 命令要点 | 结果 |
|---|---|---|
| 核心数学自检 | `python -m research.oc_rga --self-check` | **13/13 通过**（FP64 稠密后验均值/协方差、Cholesky 转置方向、稠密联合 NLL、相对协方差非对角项、自关系、全缺测、协方差收缩、尺度传播计数与有限性、两次 FP64 gradcheck、张量排列） |
| 模型自检 | `python -m research.oc_rga_model --self-check --device cpu --size 128` | **24/24 通过，1 项未运行（CPU 无 CUDA，AMP 边界留给 GPU）** |
| 评价协议自检 | `python -m research.evaluate_oc_rga --self-check --device cuda --limit 2` | **14/14 通过**（真实 DeLiVER 1042→640→pad，7 条件，故障确定性，R3 矩形主导性，R4 替换语义与别名，R7 padding） |
| 留出泄漏检查（真实 480） | `python -m research.train_oc_rga --leak-check --device cuda` | **全部通过**，正控 `max\|Δmu_post\|=3.554e-02` |
| 完整链路检查（CPU 小网格） | `--check --check-size 128 --accum-steps 4 --device cpu` | **PASS，退出码 0** |
| 完整链路检查（真实 480 + AMP + 4 累积） | `--check --micro-batch 1 --accum-steps 4 --device cuda` | **PASS，退出码 0**；attempts=8、AMP 预热跳步 7 次后真实更新 |
| 日程检查 | `--print-schedule --stop-after-epoch 30` | `total_updates=29880`、`warmup_updates=1245`，与 120 epoch 日程一致；暂停不压缩曲线 |
| 数据协议抽查 | `python -m research.oc_rga_data --inspect --limit 10` | 10 张真实 crop：`O≤S`、`Vref≤S`、缺失位置为 raw0 归一化值、编码范围内 |

### R1 定点证据（逐 head）

两个 token、差值 1、signed `weight[1]=1`、`weight[0]=0`：

```
head decay = [-0.287682, -0.133531, -0.064539, -0.031749]   （与合同数值一致）
max|dW| = 0.00e+00, max|dFull| = 0.00e+00
```

### R2 定点证据

在**启用** autocast 的条件下调用概率算子，输出 dtype 全部为 `torch.float32`：

```
{'gram_dtype': 'torch.float32', 'mean_dtype': 'torch.float32',
 'scale_dtype': 'torch.float32', 'kernel_dtype': 'torch.float32',
 'nll_dtype': 'torch.float32'}
```

`_quantiles` 接受 FP16 输入（此前会因 dtype 不一致报错）。

### R3 定点证据（真实 640 输入）

```
holes25 exact=102400  windows=4  rectMean=34133  scatterMean=1.95   trim=0.0008
holes50 exact=204800  windows=20 rectMean=51200  scatterMean=7.54   trim=0.0006
holes75 exact=307200  windows=47 rectMean=307200 scatterMean=217.10 trim=0.0005
```

即：精确整数计数、删除像素连通块平均规模比独立像素删除大 3–4 个数量级、边界剥离量 < 0.1%。

### R6/R7 定点证据

```
r6_stage01_relation_checkpointed  PASS  4 relation checkpoint calls, all use_reentrant=False
r7_centroids_use_real_pixel_extent PASS  cx=0.92105 cy=0.97059 (与审核报告期望值一致)
r7_padding_band_has_zero_support  PASS  20x30 -> (32,32)，真实区 S=1、band S=0
```

### R8 定点证据（真实 480）

```
target_observation_really_changed   PASS  目标整格 4096 像素的 Depth 与 O 都被改变
auxiliary_input_unchanged           PASS  辅助 depth/O 逐位相同
auxiliary_tensors_unchanged         PASS  y/m/f/omega2/mu_post/delta/F max|d|=0.000e+00
precision_head_input_unchanged      PASS  完整 69 通道精度头输入 max|d|=0.000e+00
model_output_independent_of_reference PASS 改 reference 后几何 bundle 逐位相同
positive_control_non_target_...     PASS  max|d mu_post|=3.554e-02
```

## 3. 关键梯度与参数更新证据

真实 480、AMP、`micro_batch=1, accum_steps=4`：

```
grads finite=True
key grads max = {oc_injection.a_beta: 7e-06, head_factor.weight: 4.97e-04,
                 head_factor.bias: 1.39e-04, precision.out.weight: 3.6e-05,
                 precision.out.bias: 1.6e-05, a_ell.0: 3e-06}
updated 34/34 key parameters with weight decay disabled (attempts=8, AMP skipped=7)
```

- 34 个探针参数覆盖：U 因子头权重/偏置、精度头输出权重/偏置、每 stage 的核长度 `a_ell`、每个 block 的注入 logit `a_beta`。
- 每条 `|grad|max > 0`，**不是**仅有参数变化；随后的有效更新在**关闭 weight decay** 的条件下完成，因此参数变化不能由 weight decay 解释。
- 优化器覆盖：835 个可训练张量、groups `[369, 466]`、`duplicates=0`、`missing=[]`。
- AMP 预热跳步 7 次属于正常现象（与既有 SUN 运行记录的 8 次同量级），随后真实应用。

真实 480 上的组均值 loss（accum=4）：`seg 3.7969 / cond 0.4497 / obs -0.5113 / total 3.8003`；`obs` 为负是概率 NLL 的正常取值（合同明确不 clamp）。

## 4. 是否有阻断问题

**未发现新的阻断问题。** R1–R8 全部定点修复并通过上述检查；原审核报告列出的最高优先项（R1 基线核、R2 AMP 边界、R3 退化协议、R5 恢复随机性、R6 训练执行风险）均已有实测证据。

仍需遵守的边界（非缺陷，属合同或已知风险）：

- R7 的 padding 仅在非 32 倍数输入触发；当前 480 训练与 640 评价都是 32 的倍数，因此该分支在首轮实际不生效（接口已正确）。
- `Vref` 端点排除会减少近/远深度区间的几何监督；实测出现整张 crop 的参考合法比例为 0 的样本（如 night 场景），训练期该项按合同跳过并计数，不属泄漏或错误。
- 概率 NLL 可为负；几何统计只保存小样本与计数，不代表全对张量分布。

## 5. 尚待 4090 smoke 验证的事项

以下**本轮未运行**，必须在 GPU 授权后由 4090 阶段 A 实测，不能用本轮 CPU/5060 结果替代：

- 480 完整前后向（含几何 loss/backward）的**峰值显存**（allocated/reserved）与余量，`4×4` 是否可训；不足时按既定方案统一改 `2×8`。
- 单步/吞吐（samples/s）与据此推算的首轮成本；数据读取、CPU 增强、同步日志是否造成 GPU 等待。
- 640 评价的显存与耗时（batch1 推理）。
- 训练 automic 下 SwanLab 在线记录、计划关机保险、`screen`/持久任务的实际可用性。
- 本地 5060 8GB 不足以代表 4090 的显存与吞吐，本轮未做大 batch/大分辨率测试。

## 6. Git 与云端状态

- 提交：`0e4498ff326f72feeda4463073ec5b6be96636fb`（父 `bbf8630`），仅含本轮修复涉及的 8 个源码/配置文件。
- 推送：`bbf8630..0e4498f research/observation-relational-geometry -> origin`，快进成功，未 force push、未动 `upstream` 与其他分支。
- 本地工作区仍保留前序轮次的未提交文档（`doc/state/current.md` 修改、审核报告等），本轮未代为提交。
- **云端未同步**：实例 `cpod-1vbh7faqcauq` 当前为 `Stopped`（`StopTime=1791948355`，无卡 A 档会话已由前序轮次结束）。本轮任务的条件是“若云端实例可用则尝试同步”，实例不可用；且启动/停止属需要明确授权的生命周期操作，本轮未授权，故**未启动实例、未尝试 fetch**。前序轮次已在云端记录 GitHub fetch 的 TLS 握手中止，未改用其它同步方式。恢复云端代码同步需一次明确的无卡启动授权（或按 `doc/guides/cloud.md` 的 Git bundle 路线）。

## 7. 结论与交接

R1–R8 已按审核报告的修正范围完成，未扩大方法或结构；rank20、三种 loss、120 epoch 日程、baseline 与 OC-RGA 匹配训练协议保持不变。本轮交付为：修复代码 + 上述实测证据 + 本回执。

请求主力模型**只复核本次代码差异与检查结果**，不需要重新做六大类完整审计；若无实质问题，可给出“代码与结构通过，GPU 能力待验收”的结论，再交云端高级模型复核。**未经两级审核与 GPU 授权，不启动 4090 正式训练。**
