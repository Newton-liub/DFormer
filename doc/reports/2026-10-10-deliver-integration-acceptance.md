# DeLiVER 接入最终验收报告

日期：2026-10-10。依据：[批准接入计划](../plans/2026-10-10-deliver-integration.md)、[执行报告](2026-10-10-deliver-integration-execution.md)及实际代码/CPU产物。

## 1. 结论与合并

**验收通过，已完成主线合并。** 现有工程已具备 DeLiVER RGB + 原始单通道 Depth 的读取、25类训练及独立 val/test 评价入口；正式 GPU 能力尚未实测，不作为本次工程验收前提。

- 执行提交：`72ec726f001d63c94c069adba9cdafa1ac8f98fe`，独立分支 `research/deliver-integration` 保留不动。
- 复核开始时主线有进行中的修复，因此先独立验收。修复提交为 `da0cace` 后，确认工作区干净，未发现研究入口的训练/评价进程，再合并。
- **合并提交：`5d61a8c192e28b131a4ff613ce52af17414b7ec8`**，父提交为 `da0cace`、`72ec726`，位于 `research/odg-sunrgbd`。
- 唯一冲突在优化器创建处：保留最新主线的完整参数分组，移除重复的 DeLiVER 条件补齐接线。保留 `7d06f05` 的独立评价 BatchNorm（BN，批归一化）数值恢复，以及 `da0cace` 的周期验证 eval 模式、统计量保护和训练合同；评价入口与 GPU 启动脚本相对 `da0cace` 无变化。没有重训 SUN、覆盖未提交工作或推送远端。

## 2. 核验结果

- **Dataset/标签正确。** 完整天气/split/scene/视角路径保留；RGB 显式转 RGB，Depth 读取单通道 uint8 并复用 $(D_8/255-0.48)/0.28$ 归一化。标签按 OpenCV 红通道取 ID，1–25→0–24，0/255→ignore255，未知 ID 报错；类别顺序与计划一致。复核了实际代码、6组 `preview-summary.json` 和已有 RGB/Depth/GT 叠图，没有重复读整份数据。
- **官方划分正确。** train/val/test 清单为3983/2005/1897，来自对应官方目录，不重新划分；`--split dev` 指官方 val，`--split test` 指官方 test。独立配置25类、`pad=False`，没有继承 SUN 的531×730 padding。
- **训练/评价已接通且隔离。** 两个 loader factory 按配置选 `research.deliver`；合并后实际导入检查确认 SUN 默认仍为 `research.data`、37类。入口的loss、指标、checkpoint流程复用主线；本次没有新训练好的25类checkpoint，因此未声称完整独立评价已经实际计分。
- **29个几何权重确实可更新。** 直接读取并核对执行者的 CPU smoke 产物及生成代码：128×128真实 train 样本、输出 `[1,25,128,128]`、loss `4.005939`、714/714可训练参数有有限梯度；原遗漏的29个 `Geo.weight` 全部加入优化器，并在一次 AdamW 更新后29/29变化。ignore255/25×25混淆矩阵的小数组检查通过。这足以证明基本数据—模型—损失—更新链路，不证明收敛或正式分辨率下的GPU能力。
- **合并后再次确认实际分组。** 一次CPU模型构造、无前向/反向，核对主线优化器：714/714张量唯一入组，missing/duplicate均0，组大小302/412，全部29个 `Geo.weight` 在组内。正式训练现沿用主线 decay 组、wd0.01；这与旧分支 smoke 的额外 no-decay 组不同，旧smoke只作基本链路证据，不作为新实验的优化协议或恢复点。
- **BN修复保留。** 合并后检查确认3个解码器BN恢复为eps0.001/momentum0.1；验证上下文进入eval模式并恢复原训练模式。未重新执行BN前向或整份val推理，原主线修复证据不重复运行。
- **预训练可配置。** `DFORMER_PRETRAINED` 覆盖默认相对路径；执行报告已记录绝对路径下成功加载encoder的smoke。本轮确认配置解析到下列真实主目录路径，没有重复加载权重；只初始化encoder，不使用SUN/MUSeg分割checkpoint。

已有证据：独立worktree的 `outputs/deliver-preparation/{preview-summary.json,smoke-report.json,preview-*.png}` 保留原位。本次新增入口证据：主目录 `outputs/deliver-acceptance-20261010/entry-check.json`。合并后日程实际输出249更新/epoch、300epoch总74700、warmup2490，周期评价目标为val。首次内联入口检查因PowerShell引号处理发生SyntaxError、未进入模型检查；仅改用标准输入重试该检查成功，没有重跑已成功的日程或CPU smoke。

## 3. 实际使用命令

在 **`D:\0Project\DFormer`** 根目录执行。以下训练/评价命令本次均未运行，需先取得对应GPU与实验授权。`1×16` 是保守可配置示例，正式micro-batch按硬件定点检查调整，乘积保持16；配置默认300epoch，可由已批准的 `--stop-after-epoch` 暂停点控制预算。

```powershell
Set-Location D:\0Project\DFormer
$py = 'D:\2Env\anaconda\envs\dformer\python.exe'
$env:DFORMER_DATASET_ROOT = 'D:\0Project\dataset'
$env:DFORMER_PRETRAINED = 'D:\0Project\DFormer\checkpoints\pretrained\DFormerv2_Small_pretrained.pth'
$env:DFORMER_EXPERIMENT_NAME = 'deliver-original'
$env:CUDA_VISIBLE_DEVICES = '0' # 正式GPU运行时设置；CPU检查时的-1会隐藏GPU

# 官方train训练，周期验证只读官方val
& $py -X utf8 -m research.train_odg --config local_configs.research.DFormerv2_S_DeLiVER --geometry-mode original --micro-batch 1 --accum-steps 16 --num-workers 2

# 改为本次DeLiVER训练生成的25类checkpoint实际路径，不能用SUN的37类权重
$ckpt = 'D:\0Project\DFormer\outputs\deliver-original\<run-id>\best-dev.pth'

# 官方val：dev在本配置中就是val；默认单尺度、无翻转
& $py -X utf8 -m research.evaluate_odg --config local_configs.research.DFormerv2_S_DeLiVER --checkpoint $ckpt --split dev --no-pad_SUNRGBD --batch-size 1 --num-workers 2

# 官方test：仅在方法、checkpoint及评价协议冻结后使用
& $py -X utf8 -m research.evaluate_odg --config local_configs.research.DFormerv2_S_DeLiVER --checkpoint $ckpt --split test --no-pad_SUNRGBD --batch-size 1 --num-workers 2
```

默认已经使用全部官方train，无需加 `--fulltrain`；该开关对DeLiVER主要作用是关闭周期val评价，而不是多获得训练数据。新worktree或其他机器没有版本控制外的权重时，仍通过 `DFORMER_PRETRAINED` 指向实际权重文件；数据根目录同理可配置。

## 4. 正式GPU实验前的必要事项与结束边界

1. 明确训练阶段、预算和运行硬件；在所选480×480 crop/micro-batch上做一次最小GPU检查，确认显存、数值及实际更新，不安排额外长性能测试。
2. 确认1042×1042原生整图、batch1评价可用；本次没有测其显存。若实际OOM，再提出最小固定推理resize方案并记录为不同评价协议，当前不预先增加滑窗。
3. 使用新DeLiVER25类训练checkpoint与当前训练合同；不从旧SUN或smoke恢复。跨Windows/Linux续训时保留同一清单字节格式，避免LF/CRLF引起现有resume指纹不一致；本次不修改全局Git换行设置。

**本任务完成。** 已达到后续实验可用的工程准备程度；原始单通道Depth协议按批准结果保留，不再讨论HHA或开展物理编码研究。没有运行GPU、正式训练、全量val/test推理、完整回归测试或数据审计；完整测试未运行是遵守最小验证预算，并非报告为已通过。后续只需按实验授权启动，不继续扩展接入工程。
