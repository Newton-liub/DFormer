# MUSeg DFormerv2-S Quick-B0 主评估与证据收口报告

- 汇报周期：2026-08-31 04:33 UTC 至 2026-08-31 06:31 UTC
- 报告对象：MUSeg 后续模块消融的内部基线交接
- 证据边界：冻结训练提交 `3975f7d66c78e9bed6b9053071bb274199d550e9`、protocol SHA-256 `6822e4cdfd9c6985c323123fc4d24a9f06ed269fada55203b1707fc5ab612bbd`、4 个预登记候选和本地 `val-dev` 主评估
- 当前状态：任务 4“主评估与证据收口”已完成，epoch 420 被选为最终 single-seed development B0

## 一、本阶段工作概述

4 个训练后候选均已在本地 RTX 5060 Laptop 上完成五尺度翻转主评估。最终胜者是 epoch 420：主评估 mIoU `58.79%`、mAcc `69.91%`、mF1 `72.73%`，checkpoint SHA-256 为 `f246a3afc50334c81302b7bfebdadf7cf37d00326bf1c3aa54f6a151754e3a1c`。

大白话说，训练期单尺度分数最高的 epoch 480 并不是最终严格评估最好的模型；按训练前写明的规则比较 4 个候选后，epoch 420 才是后续模块应共同使用的 B0 对照。该结果只属于一个 seed 的冻结开发划分，不能估计随机波动，也不是 official test 结果。

## 二、关键概念与判断边界

- `selector`（训练期低成本候选筛选器）：使用尺度 1.0、无翻转的 original-full validation 保留 top 3 和 latest。它只决定哪些 checkpoint 进入最终比较，不决定最终 B0。
- `msflip-whole-original-grid-v1`（主评估器）：对每张图执行 `0.5/0.75/1.0/1.25/1.5` 五个尺度及水平翻转，共 10 个 view；FP32 平均 pre-softmax logits，并恢复到原始 Label 网格计分。它是最终 checkpoint 的选择依据。
- single-seed development B0：只在 `val-dev` 上形成的内部比较起点。它可用于同口径模块筛选，但不能提供多 seed 方差或 official-test 泛化结论。

## 三、主评估执行与结果

### 3.1 冻结身份和运行边界

- 状态：已完成并验证。
- 评估代码：从提交 `3975f7d66c78e9bed6b9053071bb274199d550e9` 建立 detached 干净 worktree；没有使用当前 `main` 的后续 evaluator 改动。
- 协议：本地按原始字节复原 materialized protocol，SHA-256 为 `6822e4cdfd9c6985c323123fc4d24a9f06ed269fada55203b1707fc5ab612bbd`。
- 数据：`val-dev` 共 318 个样本，split SHA-256 为 `1d0719d8f64f016d48995c25ab66d4004d76b7155d9efeef7cbb7454c0dd0e83`。
- 输入：RGB 与 `rgb-imagenet-rgb-order-v1` normalization。
- 数值与几何：FP32 前向、FP32 logits 融合、TF32 disabled、原始 Label 网格计分、batch size 1、4 个 checkpoint 串行执行。
- 环境：Python `3.10.20`、PyTorch `2.7.0+cu128`、CUDA `12.8`、cuDNN `90701`、NVIDIA GeForce RTX 5060 Laptop GPU。
- official test：`official_test_included=false`，继续保持 `sealed_unread`。

首次命令因未导出仓库 `PYTHONPATH`，在导入 `models.builder` 前退出；未读取 checkpoint、未加载模型、未产生指标。只修正该启动环境后，从第一个候选重新运行，后续 4 项均正常完成，未改变协议、数据、checkpoint 或 evaluator 参数。

### 3.2 四个候选的主评估排序

1. epoch 420：selector mIoU `56.39%`；主 mIoU `58.79%`、mAcc `69.91%`、mF1 `72.73%`；耗时 `1402.672` 秒。
2. epoch 440：selector mIoU `56.10%`；主 mIoU `58.73%`、mAcc `69.54%`、mF1 `72.67%`；耗时 `1405.922` 秒。
3. epoch 480：selector mIoU `56.87%`；主 mIoU `58.43%`、mAcc `69.34%`、mF1 `72.44%`；耗时 `1427.140` 秒。
4. epoch 500：latest，无独立 selector mIoU；主 mIoU `57.68%`、mAcc `68.84%`、mF1 `71.81%`；耗时 `1406.891` 秒。

4 项 evaluator 内部计时合计 `5642.625` 秒，即 `94.044` 分钟；完整串行命令从 `2026-08-31 12:57:07 +08:00` 到 `14:31:32 +08:00`，明显低于 8 小时异常停手上限。每项记录的峰值显存均为 allocated `5,276,340,224` bytes、reserved `7,201,619,968` bytes，未发生 OOM。

epoch 420 比 epoch 440 的主 mIoU 高 `0.06` 个百分点，因此无需使用 mAcc 或更早 epoch 的同分规则。selector 排名与主评估排名不同，直接验证了“训练后对所有预登记候选运行主 evaluator”的必要性。

### 3.3 最终 B0 的逐类结果

最终 epoch 420 的 15 类 IoU / accuracy / F1（单位均为百分比）如下：

- person：`58.54 / 72.41 / 73.85`
- cable：`61.41 / 78.73 / 76.10`
- tube：`67.71 / 74.66 / 80.74`
- indicator：`73.26 / 80.20 / 84.57`
- metal fixture：`53.65 / 74.11 / 69.83`
- container：`24.67 / 26.83 / 39.58`
- tools & materials：`69.91 / 90.41 / 82.29`
- door：`86.01 / 86.63 / 92.48`
- electrical equipment：`59.08 / 82.16 / 74.28`
- electronic equipment：`56.31 / 60.95 / 72.05`
- mining equipment：`41.92 / 48.97 / 59.07`
- anchoring equipment：`65.46 / 79.70 / 79.13`
- support equipment：`40.23 / 43.57 / 57.38`
- rescue equipment：`43.56 / 55.51 / 60.69`
- rail area：`80.10 / 93.84 / 88.95`

全部逐类指标为有限值，没有空类归约或 NaN。最低 IoU 为 container 的 `24.67%`；这说明 B0 没有数值崩溃，但类别间差异仍明显，后续模块不能只报告总体 mIoU。

## 四、最终身份与证据位置

最终 B0 checkpoint：

- 文件：`cloud/DFormer-quick-b0-evidence/museg-dformerv2-s-rgb-quick-b0-v1/active-seed/checkpoint/selector-epoch-420.pth`
- epoch：`420`
- 大小：`321,011,270` bytes
- SHA-256：`f246a3afc50334c81302b7bfebdadf7cf37d00326bf1c3aa54f6a151754e3a1c`

主证据：

- 最终胜者完整评估：`cloud/DFormer-quick-b0-evidence/museg-dformerv2-s-rgb-quick-b0-v1/posteval/epoch-420-msflip-whole-original-grid-v1.json`
- 胜者报告 SHA-256：`d2dbd68a998fe03931babad839c6381dde943b8d3a2ce5c5e4d16c16eaf53c5b`
- 结构化裁决：`cloud/DFormer-quick-b0-evidence/museg-dformerv2-s-rgb-quick-b0-v1/posteval/main-evaluation-adjudication.json`
- 裁决文件 SHA-256：`20336eabfdefcf8661ed33c781a0104784950cde07d59e4d1c343c8579c27b05`
- 其余 3 个候选完整结果位于同一 `posteval/` 目录；裁决文件记录每份结果的大小、SHA-256 和逐项身份检查。

## 五、结论边界与后续使用

该 B0 已满足“训练与评估链可信、指标量级合理、无明显数值异常、身份完整绑定”的内部基线要求。后续模块只有在 pretrained、`train-dev`/`val-dev`、seed、数据顺序、500 epoch 预算、优化器、增强、checkpoint 规则和主 evaluator 全部不变时，才能直接与本 B0 做严格配对比较；模块必须从同一 pretrained 独立训练，不能从 epoch 420 继续训练后称为公平消融。

本次没有运行第二或第三 seed、完整测试套件、额外训练、云端命令或 official test。未运行的项目不能写成通过；当前结果也不能用于估计随机方差或声称 official-test 性能。
