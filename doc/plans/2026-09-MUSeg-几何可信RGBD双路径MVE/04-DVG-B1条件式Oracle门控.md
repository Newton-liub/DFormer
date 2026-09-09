# MUSeg `DVG-B1-oracle-gsa-v1`：Oracle GSA 深度门控基础设计

> **文档角色：** 条件式后继子计划；基础设计版本，尚未进入实现或运行授权。
> **计划状态：** 基础设计完成；代码修改、GPU 运行、训练和 official test 均未授权。
> **形成或核验时点：** 2026-09-09 08:48 UTC。
> **实时入口：** [`MUSeg-current-status.md`](../../main/MUSeg-current-status.md)。
> **研究选择：** [`MUSeg-open-decisions.md`](../../main/MUSeg-open-decisions.md)。
> **上级方向：** [`00-总方向规划.md`](./00-总方向规划.md)。
> **前序关系：** `DVC-A1-valdev-boundary-zero-v3-bgcontext` 已完成正式开发评价并裁决为 `not-supported`；本设计按照用户要求保留一个独立的方案验证窗口，不把 A 的失败改写为支持，也不把 B 的结果当作 A 的问题证据。
> **后继关系：** B 实验结果不能自动授权可学习质量预测、完整联合恢复、训练、云资源或 official test；后续需在本设计基础上补充参考文献、冻结 protocol，并取得单独执行批准。

## 1. 为什么在 A 未支持时仍设计 B

`DVC-A1` 的 v3 结果只说明：在固定可构造 `val-dev` 位置组中，人工把深度边界像素置零，没有观察到预注册的边界特异敏感性。它回答的是“预设问题是否成立”，不是“少信不可靠深度是否可能有效”。

因此保留一个低成本的 B 实验，用来检查一种不能提前排除的情况：

> 深度边界并不是主要受损位置，但只要根据已知的深度 corruption mask 少信这些局部深度几何信息，模型仍可能获得任务收益。

大白话说：A 是“问题找得准不准”，B 是“这个解决动作本身有没有用”。A 没通过时，B 仍然可以作为独立的方案筛选；但如果 B 有效，只能先说明“这个干预动作有经验价值”，不能反过来证明 A 的问题假设成立。

## 2. 最小问题与方案假设

### 2.1 最小问题

在与 v3 完全相同的人工深度退化输入上，若提前知道哪些像素被置为无效，只在 DFormerv2 的 Geometry Self-Attention（GSA，几何自注意力）中降低这些位置的 **depth geometry contribution**，是否比不做门控的原始 DFormerv2 输出更好？

### 2.2 方案假设

Oracle 门控假设为：

> 已知退化位置后，保留 RGB 和 GSA 的 spatial contribution，只抑制受损位置对应的 depth geometry contribution，可以改善人工退化条件下的 Boundary IoU 或 mIoU，同时不损害 clean 条件。

这里的 Oracle 表示“直接使用真实 corruption mask 的理想情况”。它只用于测量方案上限，不代表真实系统已经具备自动识别坏深度的能力。

### 2.3 方案结果如何解释

- **B 支持：** 已知坏区的最小 GSA 深度门控在预先冻结的条件下产生稳定净收益。只能说明该干预值得继续做质量信号研究。
- **B 不支持：** Oracle 也没有稳定净收益，说明当前“在 GSA 中少信已知坏深度”的方案缺少继续投入依据。
- **A 不支持、B 支持：** 这是允许出现的结果。结论应写成“问题定位未获支持，但理想位置门控显示条件性经验收益”，不能写成“深度边界损坏机制得到支持”。
- **A、B 都不支持：** 关闭“深度边界 + GSA Oracle 门控”这条首选路线，回到蓝图中的其他候选瓶颈，例如整体深度失效、RGB 低照/粉尘退化、观测保真或标定误差。

## 3. 固定输入身份与不变量

B 设计优先复用 v3 的已核验开发证据，避免重新制造一套无法比较的输入：

- **模型：** DFormerv2-S RGB Quick-B0，固定 epoch 420 checkpoint；
- **数据：** v3 冻结的 `val-dev` 派生范围，218 张图、138 个 location group；
- **Depth：** 原始 `Depth16` 上的 v3 corruption mask，随后统一经过既有量化、归一化和 resize 链；
- **输入和评估：** RGB contract `rgb-imagenet-rgb-order-v1`，五尺度原图/水平翻转，恢复到原始 Label 网格后计分；
- **标签和指标：** 使用 v3 已修正的 Boundary IoU 标签契约，raw background 作为有效几何上下文，true ignore 单独保留；
- **统计单位：** location group 是相关性边界；同一组内的全部图像保留配对结构；
- **官方测试集：** official test 继续 `sealed_unread`，protocol 必须写明 `official_test_included=false`。

B 与 baseline 的唯一研究变量是：是否在 GSA 内根据冻结的 corruption mask 抑制 depth geometry contribution。以下内容全部保持不变：RGB、受损 Depth 输入、Label、checkpoint、GSA 的 spatial contribution、五尺度翻转、logits 融合、metric grid、后处理和统计方法。

## 4. 三组最小比较

每个 condition 至少保留以下同源比较：

1. **Clean baseline：** 原始 clean Depth 输入，不做门控；
2. **Corrupted baseline：** 按 v3 mask 置零后的 Depth 输入，不做门控；
3. **Oracle-gated：** 与 corrupted baseline 使用完全相同的 RGB、Depth、checkpoint 和 evaluator，只在 GSA 中抑制 mask 指定位置的 depth geometry contribution。

clean 条件同时作为“不应改变原模型”的控制。若 clean 输入下 Oracle 与原模型不等价，不能继续解释退化条件结果。

## 5. 五个 condition 和 Oracle mask

复用 v3 的五个 condition，不新增退化类型：

- `clean`；
- `boundary-q25`；
- `boundary-q50`；
- `boundary-q75`；
- `nonboundary-q50`。

对于四个受损 condition，Oracle 使用该 condition 实际置零的像素 mask，而不是统一使用“边界 mask”。这样可以区分两种情况：

- 如果 boundary 条件有效，说明边界位置门控可能有价值；
- 如果 nonboundary 条件也同样有效，说明收益可能来自“识别任意受损深度并降低其影响”，而不是来自边界特异机制。

各 condition 的 corruption mask、样本范围和哈希必须沿用冻结 v3 证据；不得根据 B 结果重新定义 mask、改阈值、删组或增加剂量。

## 6. Oracle 的基础实现方向

### 6.1 只改 GSA 的 depth contribution

基础方案不做 logits 后处理，不混合 RGB-only 输出，不训练门控网络，也不做深度补全。它只在 DFormerv2 GSA 内部改变 depth geometry contribution 的权重或参与程度，同时保留 spatial contribution。

需要由后续代码审查和参考文献共同确认的实现点包括：

- depth geometry contribution 在当前实现中的准确张量位置；
- 原始图像 mask 到四级 GSA 特征网格的传播规则；
- 采用硬屏蔽、连续衰减还是其他最小形式；
- 是否四个 GSA stage 均使用 mask；
- mask 在 patch pooling、下采样和翻转 view 下的对应关系。

这些内容现在只冻结“必须回答的问题”，不提前把可能错误的张量公式写死。正式运行前必须把其中一种实现写入新 protocol；若无法证明只改变 depth contribution，则终点为 `protocol-blocked`。

### 6.2 推荐的最小实现原则

在后续细化时优先遵守以下原则：

1. 先选一种最简单、可逐项核对的 Oracle 形式，不并行搜索软门控、硬门控、stage 组合和多个阈值；
2. 只允许一个新自由度：mask 对 depth contribution 的作用；
3. 所有 mask resize、pooling、翻转和 stage 传播规则在看模型结果前冻结；
4. q=0 或全可信 mask 时，输出必须与原始模型达到预先定义的数值等价；
5. 不引入 RGB-only checkpoint、临时训练、补全算法、额外质量分数或新的后处理。

## 7. 指标与统计的基础设计

### 7.1 主要比较

初步将 `boundary-q75` 作为主要受损条件，比较：

`Oracle-gated - corrupted baseline`

主要指标为 Boundary IoU；mIoU 作为任务表现辅助指标。这个比较直接回答“方案是否能修复人工 q75 退化下的任务损失”，不要求 A 的问题效应先显著存在。

### 7.2 辅助比较

同时记录：

- `boundary-q25`、`boundary-q50`、`boundary-q75` 和 `nonboundary-q50` 各条件的 Oracle 增益；
- Oracle 与 clean baseline 的差异，用于检查是否损害 clean 输入；
- `boundary-q50` 和 `nonboundary-q50` 的 Oracle 增益差异，用于观察收益是否具有位置特异性；
- 每图、每组的 Boundary IoU、mIoU、有效深度比例和置零像素数；
- mask 哈希、GSA stage 传播摘要、输出有限性和原始 Label 网格恢复情况。

### 7.3 统计单位和区间

沿用 v3 的配对统计骨架：

- 先计算图像级结果，再按 location group 聚合；
- 以 138 个 location group 为 bootstrap 单位，保留组内全部图像；
- 使用预先固定的 bootstrap seed `20260908` 和 `10,000` 次有放回重采样；
- 报告 Oracle 相对 corrupted baseline 的点估计和双侧 95% percentile interval；
- 六个 mine 只作描述性分层，不当作六个独立样本。

具体的 `oracle-supported` 效应量、clean 不劣容忍度和是否需要恢复比例指标，等后续参考文献补充、GSA 代码核对和 protocol 细化后，在正式运行前一次性冻结。没有冻结前不运行完整评价，也不根据结果回填门槛。

## 8. 基础门禁和合法终点

### 8.1 实现前门禁

正式实现前至少确认：

1. `DVC-A1-valdev-boundary-zero-v3-bgcontext` 的完整结果、mask manifest 和 condition 文件身份一致；
2. GSA 的 spatial/depth contribution 可以在代码中明确分离；
3. mask 能从原始 Label/Depth 网格稳定传播到各 GSA stage；
4. clean 与 q=0 门控输出满足预先定义的等价要求；
5. Oracle 不依赖 RGB-only checkpoint、额外训练或结果后参数选择；
6. `official_test_included=false`，且执行清单仍只来自冻结的 218 张图/138 个组。

### 8.2 最小验证顺序

在获得后续实现批准后，按以下顺序推进：

1. 先做代码级 GSA contribution 定位和 mask 尺度传播小例；
2. 再做 clean/q=0 等价检查；
3. 再做 1–2 张图的 clean、boundary-q75 和 nonboundary-q50 preflight；
4. preflight 通过后，另行取得完整本地 GPU paired development evaluation 批准；
5. 完整评价结束后才执行 location-group bootstrap 和预注册裁决。

### 8.3 合法终点

- `protocol-blocked`：无法隔离 depth contribution、mask 传播不闭合、clean 等价失败、condition 或哈希不一致；保留现场，修正后建立新的 protocol identity；
- `stop`：实现错误、输出非有限、输入错位或证据链不完整；不看科学结果补洞；
- `oracle-not-supported`：实现和证据链有效，但 Oracle 没有预先冻结的净收益；停止该门控方向；
- `oracle-supported`：实现和证据链有效，Oracle 在预先冻结的条件下有稳定净收益；只允许继续设计可学习质量信号，不自动授权训练；
- `inconclusive`：结果方向不稳定或区间不足以裁决；只按 protocol 允许的诊断解释，不追加结果导向的条件和阈值。

## 9. 明确不做的事情

本基础设计不授权以下操作：

- 不修改或重跑 v3 A 实验；
- 不把 B 的结果回写成 A 的 `supported`；
- 不追加 boundary 阈值、剂量、seed 或数据组；
- 不读取 official test；
- 不训练质量预测器、恢复网络或新的 RGB-only/RGB-D 模型；
- 不使用 logits 后处理冒充 GSA 深度门控；
- 不引入三维绝对误差、risk–coverage、高置信阈值或真实矿下部署结论；
- 不在参考文献、实现细节和数值门槛尚未补齐前运行完整 GPU 评价。

## 10. 后续细化所需材料

后续用户提供参考文献后，优先补齐以下内容：

1. DFormerv2 GSA 中 spatial 与 depth geometry contribution 的准确数学和代码对应；
2. mask 传播到 patch-level geometry prior 的合理方式；
3. Oracle 门控属于机制上限还是已有方法可直接对应的实验设计；
4. Boundary IoU 与 mIoU 之外是否需要增加任务相关指标；
5. `oracle-supported` 所需的最小效应量、clean 不劣标准和配对统计说明。

补齐这些内容后，再生成独立 protocol template 和执行清单。当前文件只完成基础设计，不构成代码修改、GPU、训练、云资源或 official test 授权。
