# MUSeg DVC-A1-v2 完整评价阻塞与交接报告

> **文档角色：** 日期化完整评价终态与下一对话交接报告。
> **形成或核验时点：** 2026-09-09 00:36 UTC。
> **实时入口：** [`MUSeg-current-status.md`](../main/MUSeg-current-status.md)。
> **前序报告：** [`2026-09-08-museg-dvc-a1-v2-protection-preflight.md`](2026-09-08-museg-dvc-a1-v2-protection-preflight.md)。

## 一、结论

`DVC-A1-valdev-boundary-zero-v2` 的 218 张图、5 个 condition 的完整本地 GPU 推理已经结束，实际运行时长为 `5445.565` 秒，即 **1 小时 30 分 45.565 秒**。运行状态为 `protocol-blocked`，退出码为 `2`。

大白话说，模型推理和五个 condition 产物都已经写出，但统计保护门禁发现预注册的 138 个主分析组中有 1 个组没有可定义的 Boundary IoU，因此本次不能形成科学结果，也不能继续把现有数值解释为支持或不支持该问题。

## 二、运行身份与直接证据

- protocol：`DVC-A1-valdev-boundary-zero-v2`。
- protocol SHA-256：`3c6f33562692c8baee85786261de431583e9bdd9b6f8f51cd5ad0f042406f6bb`。
- 命令：`python tools/mve/run_dvc_a1.py --protocol "D:\\0Project\\DFormer\\cloud\\DVC-A1-valdev-boundary-zero-v2\\protocol.json" --mode full --device cuda`。
- 开始时间：`2026-09-08T14:34:00.225007+00:00`；失败记录生成时间：`2026-09-08T16:04:45.788707+00:00`。
- 执行记录：`cloud/DVC-A1-valdev-boundary-zero-v2/executions/20260908T143400225007+0000-full.json`。
- 失败记录：`cloud/DVC-A1-valdev-boundary-zero-v2/full-failure.json`。
- 环境：Python `3.13.9`、PyTorch `2.7.0+cu128`、CUDA `12.8`。
- `official_test_included=false`；official test 仍保持 `sealed_unread`。

## 三、实际写出和组级只读诊断

五个 condition 产物均已写出，且每个都是 `218` 个样本、`138` 个 location group：

- `clean.json`
- `boundary-q25.json`
- `boundary-q50.json`
- `boundary-q75.json`
- `nonboundary-q50.json`

失败信息为：`v2 analysis dose_effect does not contain exactly 138 valid paired groups`。

随后对上述 condition JSON 的 `per_group.boundary_iou` 做了只读组级计数，没有重跑评价、没有修改结果、没有读取 official test：

- `dose_effect = boundary-q75 - clean`：有效配对组 `137/138`，无效 `1` 组；无效组为 `06-01-01-0346`。
- `specificity_effect = boundary-q50 - nonboundary-q50`：在全部 138 组上同样为有效 `137/138`，无效组为 `06-01-01-0346`。
- 123 组敏感性范围：`dose_effect` 有效 `123/123`，`specificity_effect` 有效 `123/123`，没有无效组。
- `boundary-q75`、`clean`、`boundary-q50` 和 `nonboundary-q50` 各自均有 1 个组的 `boundary_iou` 为 `None`；对应的都是 `06-01-01-0346`。
- 该组不属于预注册的 123 组“组内每张图均可构造 q75”的敏感性范围，因此不能通过删去该组来凑足主分析的 138 组门禁。

这些计数解释了为什么运行能够完成推理，却在统计阶段合法停止：样本和 condition 覆盖完整，但“138 个组”与“Boundary IoU 在两个 condition 中均有定义”的有效统计对象不是同一个集合。

## 四、当前科学边界

当前不能报告：

- `dose_effect` 或 `specificity_effect` 的正式点估计；
- 10,000 次 location-group bootstrap 区间；
- `supported`、`not-supported` 或 `inconclusive` 裁决；
- 基于本次 v2 的 A1 问题支持或不支持结论。

v1 的 `protocol-blocked` 结论和原始证据保持不变；v2 本次失败也不能改写为科学上的 `not-supported`。已有五个 condition JSON 是运行证据，不是已经通过统计协议的科学结果。

## 五、下一对话的准确恢复点

1. 先读取 `doc/main/MUSeg-current-status.md`，再读取本报告和 `doc/plans/2026-09-MUSeg-几何可信RGBD双路径MVE/03-共享协议与DVC-A1问题验证.md`。
2. 保留 v2 的 protocol、allowlist、mask manifest、五个 condition JSON、执行记录和失败记录，不删除、不覆盖、不把 `None` 改成 `0`。
3. 针对 `06-01-01-0346` 只做定义层面的诊断：确认该组在 clean 与 boundary-q75、以及 boundary-q50 与 nonboundary-q50 中 Boundary IoU 未定义的具体原因和图像级分布。该诊断不改变现有 v2 protocol。
4. 在重新运行前，必须明确“有效配对组”的统计口径：是否要求每个组的两个 condition 都有组级 Boundary IoU，若是，主分析范围将发生变化；该变化不能通过静默删组实现。
5. 任何新的有效组口径、主分析范围、缺失值处理或统计门禁，都必须建立新的 protocol identity，并由用户重新确认后，才可考虑新的最小 preflight 或完整评价。
6. 在新 protocol 建立并获得明确授权前，不重跑完整评价，不改 `0.05` 阈值、allowlist、样本范围或裁决门槛，不训练、不使用云资源、不读取 official test，不解锁 `DVG-B1`。

## 六、验证边界

- 已执行：执行记录/失败记录读取；五个 condition 的样本与组覆盖核对；138 组和 123 组两个 effect 的有效配对组只读计数。
- 未执行：完整测试套件、重新推理、bootstrap、科学裁决、训练、云端操作和 official test 读取。
- 未执行这些高成本或会改变证据状态的操作，是为了保留当前失败现场并等待新的统计定义与 protocol 身份，而不是把未运行内容误报为已通过。
