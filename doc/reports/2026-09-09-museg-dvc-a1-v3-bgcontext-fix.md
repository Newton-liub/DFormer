# MUSeg DVC-A1 v3 背景上下文修正与最小门禁报告

> **报告范围：** 2026-09-09；仅覆盖 DVC-A1 Boundary IoU 标签域修正、定义层审计、v3 protocol 物化和两样本最小模型链门禁。
> **状态：** 已完成并验证；完整 v3 GPU 评价尚未授权、尚未运行。
> **实时入口：** [`MUSeg-current-status.md`](../main/MUSeg-current-status.md)。

## 一句话结论

本次修正解决了一个标签域定义问题：训练标签需要把 background `0` 映射为 ignore `255`，但 Boundary IoU 需要把 background 保留为有效几何上下文。新建的 `DVC-A1-valdev-boundary-zero-v3-bgcontext` 已通过 218 张图的 CPU 标签域审计和覆盖原阻塞组 `06-01-01-0346` 的两样本 CUDA（Compute Unified Device Architecture，图形处理器计算平台）preflight；这证明评价定义与最小推理链已闭合，但还没有产生模型敏感性、Bootstrap 区间或科学裁决。

## 1. 问题与处理

v2 的唯一无效配对组是 `06-01-01-0346`。四张图的原始 Label 都含有 `cable`、`tube` 和 `rescue equipment` 前景，问题不是前景不存在，而是旧链路先将 raw background `0` 映射为 `255`，随后 Boundary IoU 又按 ignore 安全距离排除了距该区域不超过 29 像素的像素。结果是四张图的计分安全区为空，15 个前景类别全部为双空 `None`，主分析因此只有 137/138 个有效配对组。

本次建立独立 protocol identity：`DVC-A1-valdev-boundary-zero-v3-bgcontext`。v3 只改变 Boundary IoU 的标签域，不回写 v1/v2，不改变以下研究不变量：

- `0.05` 全局相对深度跳变阈值；
- 冻结 v1 派生的 218 张图、138 个 location group allowlist；
- `clean`、`boundary-q25`、`boundary-q50`、`boundary-q75`、`nonboundary-q50` 五个 condition；
- epoch 420 checkpoint、RGB 输入、五尺度翻转 evaluator、原始 Label 网格、FP32 logits 融合；
- location group 配对、Bootstrap（按冻结位置组重采样的区间估计）和裁决门槛；
- `official_test_included=false`，official test 继续 `sealed_unread`。

## 2. 实现定义

代码中保留两套明确契约：

1. **训练/普通 mIoU 契约：** raw Label `0 -> 255`，raw Label `1..15 -> 0..14`，保持模型输入和普通评价既有语义。
2. **v3 Boundary IoU 契约：** raw foreground `1..15 -> 0..14`，raw background `0 -> 15`，true ignore 使用 `255`。背景 `15` 只作为有效 one-vs-rest 几何上下文，不作为待报告前景类别。

`semantic_boundary_iou` 新增显式 `background_label` 参数。Boundary IoU 安全域只排除 true ignore；双空类别继续返回 `None` 并排除该类 pair，单侧为空继续计 `0`。只有图像或组没有任何可定义前景类别时才触发定义层阻塞。

相关实现文件为：

- `tools/mve/dvc_a1_core.py`：增加有效背景上下文参数与契约说明；
- `tools/mve/run_dvc_a1.py`：同时保留训练标签和 raw Label，v3 使用独立 Boundary IoU target，并在 preflight 检查至少一个定义类别；
- `tools/mve/dvc_a1_protocol.py`：注册 v3 schema、identity、独立 evaluator identity 和标签契约校验；
- `protocols/dvc-a1-valdev-boundary-zero-v3-bgcontext.template.json`：新增 v3 模板；
- `tests/test_dvc_a1.py`：增加 v3 identity、背景上下文、true ignore、mask hash 和配对组门禁的聚焦断言。

## 3. 定义层审计结果

CPU（Central Processing Unit，中央处理器）标签域审计直接覆盖 v3 allowlist 的全部 218 张图和 138 个 location group：

- 215 张图含前景，3 张图为全背景；
- 138 个组均至少包含一张前景图；
- 原始标签值为 `0..15`，冻结标签中没有 true ignore；
- 因此 v3 Boundary IoU 的有效安全域为全图像素；
- `06-01-01-0346` 的四张图均有 metric foreground ids `[1, 2, 13]`，每张图有效安全域为 `1,008,424` 像素；
- 定义层审计状态为 `passed`。

三张全背景图仍保留在评价范围内，没有静默删除。它们不被预先改写成前景图；若预测与目标只有一侧有前景，继续执行单侧空计 `0` 的冻结规则。

审计证据：`cloud/DVC-A1-valdev-boundary-zero-v3-bgcontext/label-domain-audit.json`。

## 4. Protocol 与最小模型链门禁

v3 protocol 已成功物化，且沿用冻结 v1 派生 allowlist：

- protocol SHA-256：`f9960904f51cec11797ada6952c2102da4b2b6832d0bf7b529898bfae9c0f216`；
- allowlist SHA-256：`5589eb3378ed2e23180f6205e2d88cea39702ad4bfd5d4e1b739cf2f920a8d89`；
- allowlist summary SHA-256：`6fa94de96f1b5b4e94c1feecdc4d821e05db1828be05b011f3b48f43ce408dfd`；
- 覆盖范围：218 张图、138 个 location group；
- v3 protocol 标记 `official_test_included=false`。

两样本 GPU preflight 已通过，并明确覆盖 v2 阻塞的 `06-01-01-0346` 组。门禁直接核对了：

- q=0 decoded array 等价；
- 五个 condition 的 mask 确定性、嵌套关系和 q50 同面积；
- condition mask hash 与冻结 v1 manifest 一致；
- strict checkpoint load；
- 有限 logits；
- 原始 `932×1082` Label 网格输出；
- v3 Boundary IoU 至少有一个定义前景类别。

preflight 证据为 `cloud/DVC-A1-valdev-boundary-zero-v3-bgcontext/preflight.json`，状态为 `passed`，SHA-256 为 `829b580b6ed4ace977cf578e8391bcc759fc6d135fad72c2bd0dba958712dedd`。该产物记录的阻塞组样本为 `06-01-01-0346-230921160051-12-99`，其 clean 和 boundary-q50 的 Boundary IoU 均有定义，定义类别数为 7。

## 5. 聚焦验证

执行的最小代码检查为：

```text
python -m pytest tests/test_dvc_a1.py -q
```

结果为 `9 passed`，耗时约 4.13 秒；唯一 warning 是既有 pytest cache 目录权限 warning，不影响本次断言结果。

## 6. 结论边界与未执行事项

本次结果支持以下窄结论：v3 已把 raw background 与 true ignore 分离，原 v2 阻塞组在定义层形成了有效 Boundary IoU 安全域，且最小模型推理链通过身份、输入、mask 和数值门禁。

本次结果不支持以下结论：

- 不支持任何 condition 的模型优劣或敏感性结论；
- 不支持 Boundary IoU 的完整统计结果或 Bootstrap 区间；
- 不支持 `supported`、`not-supported` 或 `inconclusive` 科学裁决；
- 不支持自然传感器故障、真实低照/粉尘机制、部署安全或论文级泛化结论。

按已批准范围，未运行完整 218 张图 × 5 condition GPU 评价、Bootstrap、训练、云资源操作、official test 或 `DVG-B1`。完整 GPU 评价仍需单独授权。

## 7. 代码与证据身份

本报告基于提交 `90eee4a4ad5297cd513073f4791fe99d41eac90c` 的未提交工作区实现。直接核验的实现文件 SHA-256 为：

- `tools/mve/dvc_a1_protocol.py`：`2d226271836dcae435ca7953dcec16fd4c1905fa5d212b4acacb0e38b82cf97b`；
- `tools/mve/dvc_a1_core.py`：`f4cc8f6d03858a9057d171e3950946fda768cf379b16dcd358a3d8d725a04015`；
- `tools/mve/run_dvc_a1.py`：`4ad8da15e2c546da44004c6c69eb55c7c2f9207c26e5ffc7eb04c0d2459f62c2`。

## 8. 后续恢复点

当前恢复点是：保留 v1/v2 历史 protocol-blocked 现场和全部原始产物；以 v3 protocol、CPU 标签域审计和两样本 preflight 作为已通过的定义门禁。下一步如继续，只能在用户单独批准后运行完整 v3 GPU 评价，并沿用 v3 的独立身份和当前已核验哈希；在此之前不训练、不读取 official test、不解锁 `DVG-B1`。
