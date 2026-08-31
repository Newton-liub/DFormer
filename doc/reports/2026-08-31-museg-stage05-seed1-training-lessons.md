# MUSeg Stage-05 seed 1 训练数据复盘与经验教训

- 报告形成时间：2026-08-31
- 训练运行时间：2026-08-25 16:46 UTC 至 2026-08-26 04:53 UTC
- 报告对象：历史 Stage-05 seed 1（seed `772961337`）
- 证据边界：已取回归档、SwanLab run `4qbda9xh`、训练日志、结构化运行记录及五项 val-dev 后评估
- 当前状态：历史单 seed 训练已完成并复核；本报告只总结经验，不改变正在进行的 RGB quick B0

## 一、结论先行

大白话结论：这次历史训练本身是完整且稳定跑完的，训练 loss 总体下降但有局部回升，val-dev mIoU 从 `17.42` 上升到最高 `52.84`；但验证曲线明显有波动，最终 epoch 500 的 `52.07` 低于 epoch 460 的峰值。它最直接的经验不是“500 epoch 一定太长”或“模型已经过拟合”，而是**不能只保留最后一个 checkpoint，也不能只看训练 loss 判断模型是否还在变好**。

进一步的后评估表明，同一 checkpoint 在不同推理几何下的 mIoU 可从 `51.89` 变到 `56.31`，单类别 IoU 的变化更大。因此，checkpoint 选择和 evaluator（评估器，即把模型快照转换为指标的固定程序）必须一起冻结；不同几何的数字不能直接混成一条性能曲线。

这些结论只属于历史 legacy BGR、固定训练尺度 `[1.0]`、单尺度在线验证的 seed 1。当前 RGB quick B0 使用不同输入契约、随机尺度训练和五尺度翻转主评估。本报告**不比较两轮性能、不修改当前协议、不要求当前训练停机或重跑**。

## 二、关键概念与判断边界

| 术语 | 精确定义 | 本报告中的边界 |
| --- | --- | --- |
| checkpoint | 某个训练时点保存的模型参数快照 | epoch 460 的训练期 best 与 epoch 500 的 final 不是同一快照；最终快照不自动等于最佳快照 |
| validation mIoU | 在冻结 `val-dev` 上按类别计算 IoU 后取均值 | 每 10 epoch 观测一次，共 50 点；两个观测点之间可能存在未记录的更高值 |
| evaluator geometry | 输入 resize、整图或滑窗方式，以及预测恢复到标签网格的规则 | 几何变化会改变指标；只能在同一 evaluator 内比较 checkpoint |
| SwanLab | 训练过程的在线与本地结构化记录系统 | 本次已从归档读取原始 `.swanlab` 文件，不需要重新下载曲线 |
| single seed | 只使用一个随机种子完成一次训练 | 可以总结该次运行的工程经验，不能估计随机波动或统计显著性 |

## 三、运行身份与证据完整性

### 3.1 冻结身份

- 模型：`DFormerv2-S`
- 数据：`train-dev` 1,277 样本；`val-dev` 318 样本
- seed：`772961337`
- Git commit：`56a7ed711df2252e6228fc777d7cb92eb2510ef6`
- protocol：`museg-development-long500-v2`
- protocol SHA-256：`47d1a52a20bbb73b9d1a1b609819335d07ca84b49e8373d4046318b414ad324d`
- pretrained SHA-256：`19116988fc86dc9f3e879282237941e11b9b1b5c480edb51e92807311dbc11a6`
- 优化器：AdamW，base learning rate `6e-5`，warmup 2 epoch，poly power `0.9`，weight decay `0.01`
- 预算：500 epoch，128 iteration/epoch，batch 10，AMP 开启
- 训练增强：`train_scale=[1.0]`；在线验证为单尺度、非滑窗
- 运行设备：NVIDIA GeForce RTX 4090，PyTorch `2.1.2+cu118`
- 输入历史事实：legacy BGR；不是当前 quick B0 的 RGB 输入契约

### 3.2 SwanLab 原始记录

归档内训练 run 为 `4qbda9xh`，原始文件：

`museg-stage05-development/museg-development-long500-v2/development/seed-772961337/run-20260825_164642-4qbda9xh/run-4qbda9xh.swanlab`

直接核验结果：

- 文件大小：`11,063,726` bytes
- SHA-256：`d12241bc797461fc0865172325d9ec5a96727cbe5987dc65766741d54acd08cd`
- 记录格式版本：SwanLab `0.9.7`
- start/finish 生命周期记录：各 1 条
- scalar records：117,602 条
- metric columns：31 条
- terminal log records：7,728 条
- `train/epoch_loss`：500 点
- `validation/miou`、`validation/best_miou`、`validation/mean_accuracy`、`validation/mean_f1`：各 50 点
- 稀疏训练指标 `train/loss`、learning rate、吞吐、显存和 AMP scale：各 5,339 点
- SwanLab 系统监控：4,357 个采样时点

交叉核对：SwanLab 与 `train.log` 都有 500 个 epoch loss 点；日志保留 4 位小数，因此两者最大绝对差约 `5.0e-5`。50 个 validation mIoU 点逐点完全一致。大白话说，曲线不是截图转录，而是原始结构化记录和文本日志相互对得上。

### 3.3 完成与封存状态

- `final_epoch=500`，训练退出码为 0。
- 训练总时长 `43,623.94` 秒，约 12 小时 07 分 04 秒。
- `acceptance-v2.json` 记录 500 个 epoch 末日志和 50 个 validation 点齐全，`errors=[]`。
- 日志未发现 NaN、Inf、CUDA OOM、Traceback 或异常中断迹象。
- official test 保持封存未读：`official_test_read=false`、`official_test_included=false`、`sealed_unread=true`。

## 四、训练曲线分析

### 4.1 Loss：总体下降但有局部回升，不能单独代表泛化继续改善

`train/epoch_loss` 从 epoch 1 的 `3.046055` 降到 epoch 500 的 `0.057900`，下降约 `98.10%`；最低值为 epoch 492 的 `0.056886`。关键节点如下：

| epoch | epoch loss | validation mIoU |
| ---: | ---: | ---: |
| 1 | 3.046055 | 未评估 |
| 10 | 1.677240 | 17.42 |
| 50 | 0.541382 | 35.90 |
| 100 | 0.287613 | 42.80 |
| 200 | 0.133996 | 47.84 |
| 300 | 0.091678 | 51.05 |
| 400 | 0.066131 | 51.32 |
| 460 | 0.061468 | **52.84** |
| 500 | 0.057900 | 52.07 |

这里的 `epoch_loss` 是训练代码在一个 epoch 内累计 batch loss 后得到的均值；文本日志中的 `total_loss` 是其四位小数显示。它可以描述训练目标的下降趋势，但与 mIoU 的数值尺度和统计对象不同，不能把二者作数值差或称为“泛化差距”。

从 epoch 400 到 500，loss 仍从 `0.066131` 降到 `0.057900`，而 mIoU 在 `49.63–52.84` 之间波动。这是“训练目标继续被优化，但验证指标已进入高位波动区”的证据；它**不足以单独证明过拟合**，因为只有一个 seed、验证间隔为 10 epoch，且没有匹配的重复实验。

### 4.2 Validation：前期快速增长，后期平台且有回撤

50 个 mIoU 观测点完整覆盖 epoch 10–500：

- 前段 epoch 10–160：`17.42 → 45.12`，净增 `27.70` 点；主要能力在这一段建立，但中间已有明显波动。
- 中段 epoch 170–330：`47.04 → 51.92`，净增 `4.88` 点；增益速度明显降低。
- 后段 epoch 340–500：`51.28 → 52.07`，净增 `0.79` 点；表现为平台期内波动。
- 全程峰值：epoch 460，mIoU `52.84`。
- 最终点：epoch 500，mIoU `52.07`；比峰值低 `0.77` 点。
- 最大历史峰值回撤：epoch 110 的 `46.13` 到 epoch 120 的 `42.60`，下降 `3.53` 点。
- epoch 410–500 的 10 个点均值为 `51.612`，总体标准差约 `0.902`，范围为 `49.63–52.84`。

这些数字说明，单个验证点的上升或下降不宜被过度解释。每 10 epoch 验证一次足以做低成本候选筛选，但不能证明相邻两个验证点之间没有更高峰值。

### 4.3 Learning rate 与运行资源

学习率完成了已登记的 2-epoch warmup 和 poly 衰减：

- epoch 1 末：`2.9766e-5`
- epoch 2 末：`5.9766e-5`
- epoch 100 末：`4.9084e-5`
- epoch 300 末：`2.6304e-5`
- epoch 500 末：`2.8352e-9`

SwanLab 稀疏训练遥测的主要结果：

| 指标 | 中位数 | p95 / 高位值 | 最大值或最小余量 |
| --- | ---: | ---: | ---: |
| images/s | 17.01 | 18.33 | max 18.73 |
| step seconds | 0.588 s | 0.654 s | max 6.978 s（首批等冷启动时点包含在内） |
| PyTorch max allocated | 18,956 MiB | 18,956 MiB | 18,956 MiB |
| PyTorch max reserved | 20,632 MiB | 20,632 MiB | 20,632 MiB |
| free VRAM | 2,993 MiB | — | min 2,993 MiB |
| GPU utilization | 56% | 93% | max 100% |
| GPU temperature | 58°C | 61°C | max 64°C |
| GPU power | 221.25 W | 233.75 W | max 241.46 W |

GPU utilization 的监控点包含 validation、checkpoint 保存和数据等待，不能直接当作纯训练 kernel 利用率。现有证据支持“运行未出现显存或温度异常并完整结束”，但不足以据此诊断数据加载是否已最优。

### 4.4 Optimizer update 遥测的历史缺口

协议理论网格为 `500 × 128 = 64,000` 个 iteration slot；历史 `training_result.json` 同时记录：

- `attempted_steps=63,973`
- `completed_optimizer_steps=63,973`

两者与理论网格相差 27。AMP 动态 scale 的 SwanLab 曲线确实发生变化，取值范围为 `128–32,768`，但旧遥测没有直接记录每次 `optimizer_step_completed` 或 skip 的位置。因此，本报告只把 27 记为**理论网格与有效 update 的差额**，不把它无条件写成“已证明的 27 次 AMP overflow skip”。

这项历史缺口给出的工程教训是：attempted、completed、skipped 必须从训练时直接分开记录，并验证 `attempted = completed + skipped`，不能在训练结束后依赖差值反推原因。

## 五、Checkpoint 与评估几何

### 5.1 五项后评估

五项后评估都使用同一 318 样本 `val-dev`、legacy BGR 输入、原始 Label 网格计分，且 `official_test_included=false`：

| checkpoint | geometry | mIoU | mAcc | mF1 |
| --- | --- | ---: | ---: | ---: |
| epoch-460 best | original-full | 52.98 | 65.67 | 67.96 |
| epoch-460 best | resize-480x640 | 56.31 | 68.76 | 70.62 |
| epoch-460 best | sliding-480x640 | 51.89 | 65.45 | 66.28 |
| epoch-500 final | resize-480x640 | 56.73 | 68.88 | 70.99 |
| epoch-500 final | sliding-480x640 | 52.08 | 65.60 | 66.21 |

训练期 epoch 460 original-full mIoU 为 `52.84`，post-evaluator v2 对同一 best checkpoint 重算得到 `52.98`，相差 `+0.14`。两次运行的代码路径和环境不同；现有证据没有隔离出差异来源。因此，报告保留两个原始结果，不用其中一个改写另一个，也不把 `0.14` 解释为模型提升。

### 5.2 不能跨 geometry 选“绝对最好”

固定 epoch-460 best checkpoint 时：

- resize 比 original-full 高 `3.33` mIoU；
- sliding 比 original-full 低 `1.09` mIoU；
- 三种 geometry 的跨度为 `4.42` mIoU。

类别层面变化更大。以 best checkpoint 为例：

- resize 相对 original-full：`mining equipment +12.00`、`rail area +9.14`、`tube +8.68`，但 `support equipment -6.11`、`person -6.06`；
- sliding 相对 original-full：`rescue equipment +8.42`、`rail area +7.12`、`cable +5.03`，但 `support equipment -17.86`、`electronic equipment -9.94`、`container -7.22`。

这证明推理几何会重排类别表现，但不能证明 resize 的长宽比改变在研究上更正确，也不能据最高 mIoU 把它追认成历史主协议。

同一 geometry 内，epoch-500 相对 epoch-460 best 在 resize 上高 `0.42`，在 sliding 上高 `0.19`。这说明“训练期 original-full best”不保证在其他 evaluator 下仍排名第一；它不推翻当时按 original-full 选择 best 的协议。epoch-500 没有对应的 post-evaluation v2 original-full 结果，因此不能补出该项比较。

### 5.3 原始 geometry 下的类别薄弱点

在 best checkpoint 的 original-full 结果中：

- 最低类别 IoU：`container 21.40`；
- 其次为 `mining equipment 34.74`、`rescue equipment 37.76`、`support equipment 38.85`；
- 最高为 `door 80.52`、`rail area 71.56`、`indicator 70.81`。

这些是单次、单 seed、单 geometry 的诊断值，可用于提醒后续报告不能只给总体 mIoU；它们不能单独证明类别难度的因果来源，也不能直接指定当前 B0 的模块设计。

## 六、可复用的经验教训

### 6.1 保留候选，而不是只保留 final

训练期 best 在 epoch 460，final 在 epoch 500，二者的在线 mIoU 相差 `0.77`。历史结果直接支持保留 best checkpoint；考虑 evaluator 排序可能改变，更稳妥的做法是先用低成本 validation 保留 top-k 与 latest，再用冻结主 evaluator 统一重排。

这条是流程经验，不是对当前 quick B0 的新要求。当前计划已经采用 top 3 + latest 和训练后五尺度翻转主评估，本报告不改变该规则。

### 6.2 Loss、validation 和最终 evaluator 分层判断

- loss 回答训练目标是否仍在下降；
- validation 曲线回答冻结开发集上的低成本筛选表现；
- 主 evaluator 回答冻结最终评估协议下应选择哪个候选。

三者用途不同。loss 总体下降不能替代 validation；训练期单尺度 best 也不能替代主 evaluator 的最终排序。

### 6.3 明确记录 telemetry 语义

历史运行能确认 63,973 个有效 update，却不能从原始字段直接证明 27 个差额的逐次原因。未来运行应直接记录 attempted、completed、skipped、AMP scale 和发生位置；验收规则应读取结构化计数，而不是从理论预算反推。

### 6.4 总指标必须配合类别和 geometry

同一 checkpoint 的总体 mIoU 跨 geometry 相差 `4.42`，部分单类别差异达到两位数。报告模型性能时至少绑定 checkpoint SHA、split SHA、输入契约、geometry、计分网格和 per-class 指标。

### 6.5 运行稳定不等于方法优越

500 epoch 完成、loss 有限、无 OOM/NaN，只能说明该训练链路在该环境下跑通。B1 `safe_masked_mean` 已存在于该历史基线，但没有“关闭 B1”的匹配对照，因此不能把稳定完成或 mIoU 归因于 B1 的性能收益。

## 七、明确不能据此声称什么

本报告不支持以下结论：

1. 不支持 official test 性能；official test 仍封存未读。
2. 不支持三 seed 均值、方差、统计显著性或论文级 baseline。
3. 不支持“500 epoch 导致过拟合”；现有证据只显示后期平台和波动。
4. 不支持 RGB 与 BGR 的胜负；历史运行是 legacy BGR，当前 B0 是独立 RGB 协议。
5. 不支持某个 geometry 在研究上天然更正确；后评估只证明 geometry 敏感。
6. 不支持把 resize、sliding 和 original-full 的指标混为同一曲线。
7. 不支持把 27 个 update 差额全部定性为 AMP skip。
8. 不支持据历史曲线改变当前 RGB quick B0 的训练、候选或主 evaluator 计划。

## 八、SwanLab 曲线取回与复核方法

本次原始 `.swanlab` 文件已经在不可变归档中，无需从网页重新下载。复核时只需定点提取训练 run，不要解包 checkpoint：

```powershell
$archive = 'D:\0Project\DFormer\cloud\DFormer-stage05-archive\museg-stage05-seed772961337\museg-stage05-seed772961337-original.tar'
$member = 'museg-stage05-development/museg-development-long500-v2/development/seed-772961337/run-20260825_164642-4qbda9xh'
tar -xf $archive -C <临时目录> `
  "$member/run-4qbda9xh.swanlab" `
  "$member/files/swanlab-metadata.json" `
  "$member/files/config.yaml"
```

匹配原始 SDK 版本后，可用官方离线看板查看：

```powershell
python -m pip install swanlab==0.9.7
python -m swanlab watch --logdir <包含该 run 目录的路径>
```

当前官方文档也支持登录后通过 OpenAPI 按 `workspace/project/run_id` 获取 metric keys 与完整标量。该路径需要用户自己的本地登录态或 `SWANLAB_API_KEY`；凭据不应写入报告、仓库或聊天。本次本地原始文件已足够，因此没有调用云端 API，也没有要求用户提供凭据。

## 九、证据索引

- 完整归档：`cloud/DFormer-stage05-archive/museg-stage05-seed772961337/museg-stage05-seed772961337-original.tar`
- 归档 SHA-256：`4f6b079b707266ee358d2522fc6e4e034a5380d09ba8c65696df7aaa3e383c66`
- 训练日志：`cloud/DFormer-stage05-evidence/train.log`
- 运行结果：`cloud/DFormer-stage05-evidence/training_result.json`
- 运行配置：`cloud/DFormer-stage05-evidence/run_config.json`
- 运行 manifest：`cloud/DFormer-stage05-evidence/run_manifest.json`
- 独立裁决：`cloud/DFormer-stage05-evidence/acceptance-v2.json`
- 五项后评估：`cloud/DFormer-stage05-evidence/posteval/*-v2.json`
- 历史后评估中间报告：`doc/reports/2026-08-28-museg-stage05-posteval-protocol-gate.md`

## 十、与当前 RGB quick B0 的隔离声明

本报告是历史证据解释层，不是当前执行授权。它没有读取当前训练日志、SwanLab run、checkpoint 或云实例状态，没有更改当前 protocol、训练预算、top 3 + latest 规则、五尺度翻转主 evaluator、official-test 封存状态或下一恢复步骤。

大白话说：上一次训练告诉我们“以后要怎样看曲线和保留证据”，但不替当前这一轮做决定。
