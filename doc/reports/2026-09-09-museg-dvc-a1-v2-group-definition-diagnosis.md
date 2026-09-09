# MUSeg DVC-A1-v2：`06-01-01-0346` 组级 Boundary IoU 定义诊断

> **文档角色：** 对 v2 完整评价阻塞组的只读定义层诊断。
> **核验时点：** 2026-09-09。
> **实时入口：** [`MUSeg-current-status.md`](../main/MUSeg-current-status.md)。
> **前序终态报告：** [`2026-09-09-museg-dvc-a1-v2-protocol-blocked.md`](./2026-09-09-museg-dvc-a1-v2-protocol-blocked.md)。

## 一、大白话结论

`06-01-01-0346` 的组级 `boundary_iou = None` 不是因为四张图没有前景，也不是因为模型没有输出分割结果。真正原因是：这四张图的前景区域都没有像素距离 Label 背景/ignore 区域超过 Boundary IoU 所需的 29 像素安全距离，因此按当前冻结规则，四张图在计分安全区内所有 15 个前景类别都同时为空。双空类别按规则记为未定义 `None`；图像级没有任何已定义类别，随后组级也只能是 `None`。

该诊断只读取既有 v2 condition 结果、v2 mask manifest、冻结数据 Label 和当前 Boundary IoU 实现；没有修改 v2 protocol、任何产物或缺失值，也没有把 `None` 改成 `0`。它确认了阻塞机制，但**没有选择新的有效配对组口径**，因此尚不能重新运行评价。

## 二、组内具体图像

该组包含以下 4 张 `val-dev` 图像，来自 `data/splits/MUSeg/dev-v1/val-dev.txt`：

- `06-01-01-0346-230921160051-12-99`
- `06-01-01-0346-230921160052-12-99`
- `06-01-01-0346-230921160056-12-99`
- `06-01-01-0346-230921160058-12-99`

四张图均为原始 Label 网格 `932×1082`。当前 Boundary IoU 距离带宽为 `29` 像素。

## 三、前景类别与安全区核对

Label 原始值 `0` 是背景，原始前景值 `1..15` 在 evaluator 中转换为类别 `0..14`，对应 15 个前景类别。该组四张图的原始 Label 只出现三类前景：

- 原始类别 `2`：`cable`；
- 原始类别 `3`：`tube`；
- 原始类别 `14`：`rescue equipment`。

四张图的原始前景像素数量分别为：

- `160051`：`cable 25,724`、`tube 48,223`、`rescue equipment 13,173`，合计 `87,120`；
- `160052`：`cable 26,379`、`tube 40,841`、`rescue equipment 13,752`，合计 `80,972`；
- `160056`：`cable 23,502`、`tube 44,312`、`rescue equipment 8,118`，合计 `75,932`；
- `160058`：`cable 24,211`、`tube 44,494`、`rescue equipment 7,913`，合计 `76,618`。

当前实现先把距离 ignore 区不超过 `29` 像素的区域排除。四张图距离背景/ignore 区的最大有效距离分别只有：

- `160051`：`21.095022` 像素；
- `160052`：`20.000000` 像素；
- `160056`：`18.000000` 像素；
- `160058`：`19.000000` 像素。

所以四张图的 Boundary IoU 安全区像素数均为 `0`。这意味着三类原始前景虽然在整张 Label 中存在，但在计分安全区中都不存在；其余 12 个前景类别也同样不存在。

## 四、当前规则如何产生 `None`

`tools/mve/dvc_a1_core.py` 的 `semantic_boundary_iou` 对每个 evaluator 类别分别构造安全区内的 ground-truth 与 prediction mask：

1. 安全区为空后，每个类别的 ground-truth mask 都为空；
2. prediction mask 也在同一个空安全区内计算，因此同样为空；
3. 当前冻结规则对“ground truth 与 prediction 都为空”的类别不计入平均，并把该类别写为 `None`；
4. 四张图的 `boundary_iou_per_class` 都是 15 个 `None`，`defined_class_count = 0`，因此每张图的 `boundary_iou` 都是 `None`；
5. `aggregate_condition_records` 会过滤图像级 `None`。该组四张图没有任何已定义值，所以五个 condition 的组级 `boundary_iou` 都是 `None`。

因此，`clean`、`boundary-q75`、`boundary-q50` 和 `nonboundary-q50` 的组级缺失是同一个目标 Label 安全区规则造成的，而不是某个 condition 单独导致的。`image_miou` 在四个 condition 中仍有有限数值，进一步说明这里是 Boundary IoU 定义域为空，不是整张图推理产物为空。

## 五、与 Depth corruption 的关系

v2 mask manifest 对四张图的边界候选和置零数量为：

- `160051`：边界候选 `0`，`boundary-q75` 置零 `0`；
- `160052`：边界候选 `121`，`boundary-q75` 置零 `90`；
- `160056`：边界候选 `0`，`boundary-q75` 置零 `0`；
- `160058`：边界候选 `0`，`boundary-q75` 置零 `0`。

所以该组确实只有一张图执行了非空的 q75 深度置零，但这不是组级 Boundary IoU 变为 `None` 的直接原因：即使在 `clean` 中不做任何深度置零，四张图的安全区仍然为空；在五个 condition 中，目标 Label 和安全区定义也没有改变。

## 六、证据与当前边界

直接证据：

- 图像清单：`data/splits/MUSeg/dev-v1/val-dev.txt`；
- v2 mask 构造证据：`cloud/DVC-A1-valdev-boundary-zero-v2/mask-manifest.json`；
- 五个 condition 的图像级与组级结果：`cloud/DVC-A1-valdev-boundary-zero-v2/conditions/{clean,boundary-q25,boundary-q50,boundary-q75,nonboundary-q50}.json`；
- Boundary IoU 实现：`tools/mve/dvc_a1_core.py` 中的 `semantic_boundary_iou` 与 `aggregate_condition_records`；
- v2 运行终态：`cloud/DVC-A1-valdev-boundary-zero-v2/full-failure.json` 与 `cloud/DVC-A1-valdev-boundary-zero-v2/executions/20260908T143400225007+0000-full.json`。

本诊断支持的结论是：`06-01-01-0346` 在当前 Boundary IoU 定义下没有可定义的图像级或组级 Boundary IoU，原因是 29 像素 ignore 安全区规则将四张图的计分域全部排空。

本诊断不支持以下未经重新决策的操作：

- 将 `None` 改成 `0`；
- 修改距离带宽、ignore 处理或双空类别规则；
- 静默删除该组并继续使用 v2 identity；
- 直接使用 123 组敏感性结果替代预注册 138 组主分析；
- 重跑评价或形成科学裁决。

在任何重跑前，仍须明确新的有效配对组口径，并为改变主分析范围、缺失值处理、统计门禁或 protocol 语义建立新的 protocol identity，随后取得用户确认。
