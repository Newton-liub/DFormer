# 路径 A：Depth 退化与可信度问题验证

> **实验状态：** 计划中，未执行。
> **实验类型：** 单变量、paired、冻结模型的最小可行实验（MVE）。
> **上级计划：** [`../00-总方向规划.md`](../00-总方向规划.md)；双路径原方案见 [`../02-问题与方案双路径-MVE.md`](../02-问题与方案双路径-MVE.md)。
> **实时状态：** [`../../../../../main/MUSeg-current-status.md`](../../../../../main/MUSeg-current-status.md)。

## 1. 要回答的问题

在 checkpoint、RGB、标签、样本、推理几何和评价代码均不变时，只提高输入中无效 Depth 的比例，是否会在独立评价集 `E` 上造成可量化、可复核的可信度损失？

核心假设是：Depth 有效性下降可能同时带来条件正确率下降、高置信错误增加、概率失校准或风险排序恶化。如果只看到 mIoU 下降而没有可信度损失，不能把“退化条件下概率不可信”认定为后验校准的主要瓶颈。[RE068][RE078][RE131][RE666][RE752]

该实验只检验预注册合成退化条件下的方向关系，不证明真实传感器故障概率、现实部署收益或安全保证。[RE346][RE460]

## 2. 固定身份

- **模型：** `DFormerv2-S + ham`，MUSeg 15 类 RGB-D 语义分割；`DFormerv2` 的正式论文为 CVPR 2025 的 `DFormerv2: Geometry Self-Attention for RGBD Semantic Segmentation`。[RE006][RE429]
- **checkpoint：** epoch 420 `selector-epoch-420.pth`；SHA-256：`f246a3afc50334c81302b7bfebdadf7cf37d00326bf1c3aa54f6a151754e3a1c`。[PR001]
- **数据：** 必须建立与 `train-dev`、`val-dev` 和 official test 隔离的独立标注数据池，再按 location group 划分 `C` 与 `E`。路径 A 的正式结果只使用 `E`。[PR003][PR005]
- **输入：** RGB 使用 `rgb-imagenet-rgb-order-v1`；Depth 使用当前全局 8-bit 单通道复制 3 通道、mean `0.48`、std `0.28`。[PR001][PR005]
- **输出：** 15 类逐像素 float32 logits；恢复到 MUSeg 原始 Label 网格计分；ignore index 和无效标签像素不参与指标。[PR004]
- **共享约束：** clean 与退化样本一一对应，复用同一份 RGB、Label、样本顺序、有效像素掩码和 evaluator。
- **官方测试：** `sealed_unread`，不参与数据划分、退化参数选择、阈值冻结或结果回调。[PR001][PR003]

## 3. 运行前必须冻结

以下项目全部确定并写入 protocol 后，才能进入正式运行：

1. 独立 `E` 的来源、location group 划分、样本清单和 SHA-256；
2. Depth 无效比例 `q` 的水平；
3. mask 类型、fill 类型、固定 seed；
4. 每张样本的目标覆盖率和实际覆盖率记录方式；
5. 主指标、分层指标、最小效应、重采样单位、置信区间和“不支持/不确定”边界；
6. checkpoint、输入契约、evaluator、代码提交和输出 manifest 身份。[PR003][PR005]

当前具体 `q`、mask、fill、seed 和统计数值均为**待确认**，不得从结果反推或事后调整。

## 4. 实验步骤

1. **冻结身份。** 绑定 checkpoint、权重哈希、输入契约、FP32 推理、原始 Label 网格、ignore mask、代码提交、protocol 和 evaluator identity。
2. **冻结数据。** 建立独立 `C/E`，用 location group 防止同一采集位置跨职责泄漏；路径 A只取 `E`。
3. **确定 clean 条件。** 保留每个样本的原始 RGB、Depth、Label 和有效像素掩码，形成可审计输入清单。
4. **生成 paired 退化条件。** 对同一 sample 和同一 `q` 使用预注册 mask，只改变 Depth；RGB、Label、样本顺序和 evaluator 保持不变。全零输入只能称为 Depth 缺失/退化，不能称为严格 Depth-only 模型。[RE346][RE461]
5. **一次导出同源 logits。** 对 clean 和所有预注册条件各生成 15 类逐像素 float32 logits。记录 sample ID、condition ID、mask 身份、目标/实际覆盖率、checkpoint/split/protocol 哈希和产物哈希。后续指标只读取这批 logits。[PR004][PR005]
6. **计算指标。** clean 与每个退化条件分别计算 ECE、NLL、Brier Score、高置信错误率（如预注册 `HCE@t`）、mIoU 和风险—覆盖指标。现有文献只能提供分布偏移/不确定性背景；ECE、NLL、Brier 和风险—覆盖的规范定义仍待补充直接方法来源。[RE666][RE131]
7. **进行 paired 汇总。** 以图像或 location group 为重采样单位，比较 clean 与退化条件的差值，检查 `q` 与可信度损失是否保持预注册方向；不能把全部像素当作完全独立重复。[RE447]
8. **执行停止门禁。** 数据身份、mask 覆盖、logits 有限性、标签/掩码对齐、样本顺序或 official-test allowlist 任一失败时，标记 `protocol-blocked`，保留现场，不解释科学指标。

## 5. 最简资源

- 一份已审计的独立 `E` 数据及可复算的 Depth corruption manifest；
- 一次冻结模型推理环境；不需要反向传播、重新训练或云端训练；
- 可保存同源 float32 logits 和结构化 manifest 的本地磁盘；指标计算可在 CPU 完成。[PR001][PR004]

正式推理、完整数据读取、GPU、长耗时评价和云资源均需另行授权。

## 6. 判据与合法终点

### `A-problem-supported`

至少一个预注册的主可信度/风险指标随 `q` 增大呈稳定、可复核并达到最小效应的恶化方向；paired group-level 不确定区间支持该方向，且证据不只由单一 group 驱动。[RE447][RE068][RE666]

### `A-problem-not-supported`

`q` 增大没有稳定增加失校准或高置信错误，或效应只在单一条件/单一 group 出现且不确定区间覆盖零。此时停止扩展复杂校准器，重新审视问题定义。

### `A-inconclusive`

mIoU 明显下降但概率指标不变，或总体与主要分层方向冲突。只能记录“分割退化与后验失真未被证明是同一问题”，不能强行判为支持或失败。

### `protocol-blocked`

独立数据职责、checkpoint、输入、logits、mask、标签对齐或 official-test 边界无法闭合时使用。修复后必须以新的 protocol identity 重新开始，原失败现场不覆盖。[PR003][PR004]

## 7. 备选方案边界

如果暂时无法生成 Depth 退化输入，可以在同一份 clean logits 上做预注册的 logit sharpness/temperature 扰动，作为工程筛查。该简化实验只能判断后验尺度失真是否足以制造失校准，不能证明 Depth 退化是根因，也不能替代路径 A 正式实验。当前 `RE136` 不是 temperature scaling 的直接方法来源；对应基础论文待补充。

## 8. 待补充参考资料

- 重点寻找：Depth 缺失或局部无效、RGB-D 退化、模态可靠性、分布偏移下的概率失真、风险—覆盖评价。
- 现有背景依据：`RE429`（MUSeg 数据集与矿下场景）、`RE078`（不完整多模态语义分割）、`RE346`（模态内不完整与整模态缺失）、`RE461`（不确定模态）、`RE752`（质量感知退化）、`RE666`（分布偏移下的不确定性）、`RE447`（bootstrap 与不确定性量化）。这些大多是邻近任务，不能替代 Depth corruption 与密集预测校准的直接证据。
- `RE006` 仅用于固定 DFormerv2 模型和 Depth 几何先验背景。
- 待补充：Depth 缺失/局部无效的 RGB-D 语义分割、corruption calibration、dense prediction calibration、ECE/NLL/Brier 与 selective risk 的直接方法论文。
- 新论文应在 `./段落主题1_论文编号.txt` 中登记，并在此处补充其与本实验的直接关系。
