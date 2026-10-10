# ODG epoch30：评价配置对齐与 25% 孔洞复测（2026-10-10）

本报告记录上级批准的“最小评价修复与复测”：修复独立评价入口未恢复训练时 BatchNorm 数值配置的问题，并用现有两个 epoch30 `last.pth` 在本地 RTX 5060 重做 Original/ODG 的 clean 与 25% 孔洞共四组评价。执行边界见[项目状态](../state/current.md)与[ODG 入口](../guides/odg.md)；上一轮未对齐的孔洞报告见[25% 孔洞评价](2026-10-10-odg-holes25-evaluation.md)。

**结论摘要：** ①修复已完成并定点验收，评价模型的解码器 BatchNorm 现为训练时的 `eps=0.001`，但**它只让四组数值变化 ≤0.01 个百分点**，原先怀疑的 `eps` 不是 clean 与训练内不一致的原因；②修复后 clean 为 Original **41.32**、ODG **42.77**，**仍不能复现**训练内 39.39/36.97，实际原因是**训练入口的周期验证从未切换到 eval 模式**，该差异已用定点诊断复现（同权重同数据在 train 模式下得 39.09/37.22，并复现训练记录中三类 IoU 为 0 的特征）；③25% 孔洞下 ODG 相对 Original 的 **+2.96** 个百分点优势在修复后完全保留，退化量 1.54 → 0.03（改善 1.51 点）。**评价口径现已对齐；正向信号保留；但“训练内 39.39/36.97”与据此得出的“ODG 落后 2.42 点”属 train 模式数值，是否改变原停止续训结论需上级科研裁决。**

## 1. 修复内容与定点验收

只改研究层两处，未改模型结构、checkpoint、训练代码、优化器与数据管线：

- `research/odg_schedule.py` 新增 `restore_decoder_norm_config()`：把 `C.bn_eps` / `C.bn_momentum` 写回解码器 norm 层。
- `research/evaluate_odg.py` 在构建模型后、加载 checkpoint 前调用它，并在 `eval_result.json` 中记录 `decoder_bn_eps` / `decoder_bn_momentum` / 实际层列表。

机制（与诊断报告一致）：`models/builder.py` 只在 `criterion` 非空时才调用 `init_weights`，而写 `eps`/`momentum` 的是 `utils/init_func.__init_weight`；`state_dict` 只携带张量，因此独立评价入口（`criterion=None`）此前一直使用 PyTorch 默认 `eps=1e-5`。

CPU 定点验收（三路径对比，无前向、无 GPU）：

| 构建路径 | `decode_head` 三个 BN 的 eps/momentum | 编码器 7 个 BN 的 eps |
|---|---|---|
| 训练路径（传 criterion，作者 init 生效） | 0.001 / 0.1 | 1e-5 |
| 评价路径修复前 | 1e-5 / 0.1 | 1e-5 |
| 评价路径修复后 | **0.001 / 0.1（与训练逐项相同）** | 1e-5（未触碰） |

修复前后全部 792 个浮点张量逐元素不变，运行统计量、仿射参数与骨干均未改动；checkpoint 仍是唯一权重来源。

## 2. 四组复测结果（本地 RTX 5060）

同一 dev 528 图、seed 12345、`--pad_SUNRGBD`、单尺度无翻转、AMP 开、batch 1、workers 2、`holeseed=12345`；两次孔洞的删除像素为 **41,265,537 / 165,061,806 = 25.000052%**，与修复前完全一致。

| 组 | 修复后 mIoU | mAcc | mF1 | 修复前（eps=1e-5） | 差 |
|---|---|---|---|---|---|
| Original clean | **41.32** | 58.54 | 55.25 | 41.31 | +0.01 |
| Original 25% 孔洞 | **39.78** | 56.84 | 53.66 | 39.77 | +0.01 |
| ODG clean | **42.77** | 57.58 | 56.76 | 42.76 | +0.01 |
| ODG 25% 孔洞 | **42.74** | 57.70 | 56.81 | 42.74 | 0.00 |

四次推理各 63.0 / 67.9 / 70.8 / 72.7 秒，均正常退出、无 OOM。**修复的数值影响 ≤0.01 点**：解码器 BN `eps` 不是 clean 与训练内不一致的解释，上一轮的全部孔洞结论在数值上原样成立。

## 3. clean 不能复现训练内 39.39/36.97：原因已定位并复现

**定位（代码事实）：** `research/train_odg.py` 的周期验证在 `model.train()` 状态下直接调用 `evaluate_loader`（该函数只有 `@torch.no_grad()`），全程没有 `model.eval()`；而作者的 `utils/train.py` 在验证前明确 `with torch.no_grad(): model.eval()`。因此训练内的三个验证点是在 train 模式前向下产生的，后果是：解码器与编码器 BN 用单张图的 batch 统计替代 running 统计；`DropPath(0.1)` 仍然生效（timm 的 DropPath 在 `self.training` 时随机丢弃）；HAM 用 `train_steps=6` 而非 `eval_steps=7`。

**定点复现（诊断，不计入四组正式评价）：** 同 checkpoint、同 528 图、同入口逻辑，仅令模型保持 train 模式：

| 组 | train 模式（本次诊断） | 训练内记录（epoch 30） | 差 |
|---|---|---|---|
| Original | 39.09 | 39.39 | −0.30 |
| ODG | 37.22 | 36.97 | +0.25 |

并复现了训练记录的特征：记录的逐类 IoU 中 `floor_mat`、`shower_curtain`、`night_stand` 三类为 0，本次 train 模式诊断同为这三类为 0；而正确的 eval 模式下只有 `floor_mat` 为 0（另两类为 22.24/10.98 与 9.95/23.25）。残留约 0.3 点来自 train 模式本身的随机性（DropPath、以及 `rand_init=True` 时 HAM 每次前向重抽随机基）。

**因此：39.39/36.97 是 train 模式数值，不能作为独立 clean 评价的复现目标；同一批权重在正确 eval 协议下为 41.32/42.77。** 需要同时记录两点归因边界：其一，原报告“ODG 落后 2.42 点且差距扩大”的趋势同样建立在 train 模式验证上，同一批 checkpoint 在 eval 模式下 clean 反而领先 1.45 点，是否因此改变“不续训”的结论属于科研裁决，本报告只给事实；其二，模型仍为 30 epoch 暂停点、单一 seed、单一 dev 划分。

## 4. 孔洞对照：正向信号保留

| 组 | clean | 25% 孔洞 | 退化 |
|---|---|---|---|
| Original | 41.32 | 39.78 | **−1.54** |
| ODG | 42.77 | 42.74 | **−0.03** |

- ODG−Original：clean **+1.45**，孔洞 **+2.96**，退化量少 **1.51** 点——与修复前数值一致（±0.01）。
- 主要类别（本地配对 IoU，孔洞优势/各自退化）：`bathtub` **+22.83**（Original 退化 18.63，ODG 退化 0.18）、`person` **+20.43**（ODG 受损后反升 +3.75）、`fridge` **+16.44**（优势主要在 clean 已存在）、`night_stand` +10.34、`sofa` +10.10、`towel` +8.19、`picture` +7.54、`clothes` +6.60、`bag` +5.84、`floor` +5.79、`counter` +5.14。
- ODG 仍存在明确弱项：`shower_curtain` **−11.21**、`blinds` **−8.37**、`mirror` −3.71、`desk` −3.13、`ceiling` −2.83；整体退化仅 0.03 不等于所有类别或孔洞区域都不受损。
- 归因边界不变：两组共享相同深度输入与人工删除 mask，但 ODG 另有支持域中性化，当前对照**不能分离**分布与显式 mask 的贡献；未运行 `mean` 或去掉 mask 的消融。

## 5. 结论与下一步建议

1. **评价已对齐**：解码器 BN 数值现与训练一致，评价在标准 eval 模式下进行；孔洞对照的输入、seed、比例与像素口径均可直接核对。
2. **修复不改变孔洞结论**：25% 孔洞 +2.96 点优势与 1.51 点退化改善在修复后原样保留，说明该正向信号不是评价数值配置造成的假象。
3. **但“训练内验证”本身不可用于绝对比较**：建议下一步单独授权把 `research/train_odg.py` 的周期验证改为 `model.eval()` 后再验证、之后恢复 `model.train()`（改动很小，但会改变已有验证语义与 best-dev 选择口径，需上级决定；本轮未授权未执行）。
4. **不因本轮结果自动续训或新增消融**：若继续，优先级建议为（a）修正验证模式后重取验证点，（b）用同协议检查 `mean`/无 mask 消融以分离机制，（c）再决定 100 epoch。正式 test、云端 GPU 均未授权、未运行。

产物：`outputs/odg-holes25-bnfix-20261010/{original-clean,original-holes25,odg-clean,odg-holes25}/`（各含 `eval_result.json`、`eval_per_class.csv`）与诊断证据 `outputs/odg-holes25-bnfix-20261010/diagnostic-trainmode.json`。checkpoint 仍为 `outputs/sun-dev-original-seed12345/20261010-013324/last.pth` 与 `outputs/sun-dev-odg-seed12345/20261010-020527/last.pth`，未改写。

复现命令（仓库根目录，`DFORMER_DATASET_ROOT=D:\0Project\dataset`）：

```bash
python -X utf8 -m research.evaluate_odg --config local_configs.research.ODG_SUNRGBD \
  --checkpoint outputs/sun-dev-original-seed12345/20261010-013324/last.pth \
  --pad_SUNRGBD --hole-ratio 0.25 --hole-seed 12345 --num-workers 2 \
  --out outputs/odg-holes25-bnfix-20261010/original-holes25
```

## 未运行 / 未验证

- 未修改优化器、未重新训练、未新增消融、未跑正式 test、未启动云端 GPU（本轮全部为本地评价与一次本地 train 模式诊断）。
- 未测量不同全局 seed 下 HAM 随机基造成的评价波动幅度（仅记录该机制为代码事实），因此 clean 的“完全复现”本身受该方法随机性限制。
- 编码器 BN 在训练与评价中同为 `eps=1e-5`，本轮未改动，也未与云端 torch 2.1.2 做逐张量数值比对。
