# MUSeg `DVG-B1-oracle-gsa-v1`：条件式 Oracle 门控

> **文档角色：** 条件式后继子计划；只冻结方向、触发条件和不越界边界。
> **计划状态：** 条件式待规划、未授权、当前不在恢复点。
> **形成或核验时点：** 2026-09-08。
> **实时入口：** [`MUSeg-current-status.md`](../../main/MUSeg-current-status.md)。
> **研究选择：** [`MUSeg-open-decisions.md`](../../main/MUSeg-open-decisions.md)。
> **上级方向：** [`00-总方向规划.md`](./00-总方向规划.md)。
> **前序关系：** 仅在 [`03-共享协议与DVC-A1问题验证.md`](./03-共享协议与DVC-A1问题验证.md) 裁决为 `supported` 后触发。
> **后继关系：** 当前无；Oracle 结果不能自动解锁可学习门控或训练。

## 1. 触发条件

必须同时满足：

1. `DVC-A1-valdev-boundary-zero-v1=supported`；
2. 前序 protocol、代码、checkpoint、split、evaluator、condition 和结果哈希完整；
3. q=0 等价与 Boundary IoU 门禁通过；
4. 用户在看到 DVC 裁决后再次明确批准本子计划的细化或实现。

任一条件不满足时，本文件只保留方向，不得开始编码、GPU 运行或参数选择。

## 2. 方向与最小问题

候选问题是：已知人工置零位置时，只抑制这些位置对 DFormerv2 GSA 中 **depth geometry contribution** 的影响，能否在不引入 RGB-only 模型、补全网络或可学习质量预测器的情况下恢复一部分 Boundary IoU。

【文献依据：DFormerv2 的几何先验由 spatial 与 depth 关系加权组成，见参考资料 P0-1；PR029/PR089/PR090 仅支持“按置信度减少不可靠模态影响”的动机，其方法均需专门训练，不能作为本 Oracle 的直接实现依据，见 P0-2、P0-3。】

大白话说，这一阶段只问理想情况下“少信坏深度”有没有上限价值，不声称已经学会识别坏深度。

## 3. 候选 A/B 唯一差异

- Baseline：复用前序 `boundary-q50` 与 `boundary-q75` 的冻结 DFormerv2 输出。
- Oracle：复用相同 RGB、受损 Depth、checkpoint 和 evaluator，只根据已知 corruption mask 在对应 GSA 尺度抑制 depth prior contribution；spatial prior 保留。
- 不使用另一个 RGB-only checkpoint，不做 logits 混合，不加入亮度引导补全，不训练质量分数，不改变 RGB 或 Label。

具体 mask 下采样规则、软/硬门控公式和 stage 应用位置会受 DVC 错误分布影响，当前**不提前冻结**。如果无法做到“只改变 depth contribution”，该方向判 `protocol-blocked`，不退化成任意 logits 后处理。

## 4. 主要交付物

触发后只需一份细化 protocol，包含：

- GSA depth/spatial prior 的明确切分和最小代码改动；
- mask 从原始网格到四级 GSA 网格的传播规则；
- clean、受损 baseline、Oracle 三者的同源比较；
- q=0/全可信时与原 checkpoint 数值等价的门禁；
- Boundary IoU、mIoU 和 paired location-group 差值；
- `oracle-supported/not-supported/inconclusive/protocol-blocked` 裁决。

## 5. 不提前冻结

- 不冻结软门控还是硬门控、阈值、四个 stage 是否全部应用；
- 不冻结成功效应量，需先依据 DVC 的实际损失幅度定义“恢复比例”，再在运行 Oracle 前写入新 protocol；
- 不冻结可学习特征、参数规模、训练集或优化器；这些均不属于 Oracle；
- 不冻结三维、risk–coverage、高置信阈值、真实退化或补全指标。

若细化时引入论文中的具体模块、损失、阈值或置信估计公式，必须在对应位置写【需补参考文献】并先补齐直接依据；简单的 mask resize、张量乘法、配对差值和哈希记录不需要额外文献。

## 6. 合法终点

- `oracle-supported`：已知坏区的最小 GSA 抑制恢复了预先冻结的有意义比例，可另立“质量识别”研究计划；不自动授权训练。
- `oracle-not-supported`：Oracle 也无净收益，停止门控主线。
- `inconclusive`：结果方向不稳定；只检查既定 evidence，不追加模型或退化类型。
- `protocol-blocked`：无法隔离 depth prior、clean 等价失败或 mask 几何不闭合；停止并记录恢复点。

## 7. 禁止越界与验证预算

不提前运行 GPU、训练、云任务或 official test；不把本方向写成已授权；不使用结果后阈值搜索。触发后验证仍只保留最小检查：clean 等价、mask 尺度传播小例、1–2 样本 preflight，以及另行授权后的完整 paired development evaluation。状态、授权或恢复点变化时，最终答复前更新实时状态文件。
