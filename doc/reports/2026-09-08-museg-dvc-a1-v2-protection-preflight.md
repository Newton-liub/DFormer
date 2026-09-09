# MUSeg DVC-A1-v2 修正后 Preflight 报告

> **文档角色：** 日期化修正后 preflight 证据报告。
> **形成或核验时点：** 2026-09-08 14:25 UTC。
> **实时入口：** [`MUSeg-current-status.md`](../main/MUSeg-current-status.md)。
> **前序报告：** [`2026-09-08-museg-dvc-a1-v2-protection-fix.md`](2026-09-08-museg-dvc-a1-v2-protection-fix.md)。

## 一、结论

修正后 `DVC-A1-valdev-boundary-zero-v2` 的两样本本地 GPU preflight 已通过。该结果只证明新 protocol 身份下的最小运行链闭合，不产生模型指标、科学裁决或完整评价结论。

大白话说，正式评价前新增的两道保险已经在实际样本上生效，但 218 张图的完整评价仍未开始。

## 二、固定身份

- protocol SHA-256：`3c6f33562692c8baee85786261de431583e9bdd9b6f8f51cd5ad0f042406f6bb`。
- checkpoint SHA-256：`f246a3afc50334c81302b7bfebdadf7cf37d00326bf1c3aa54f6a151754e3a1c`。
- v2 allowlist：218 个样本、138 个位置组；allowlist SHA-256 为 `5589eb3378ed2e23180f6205e2d88cea39702ad4bfd5d4e1b739cf2f920a8d89`。
- official test：`official_test_included=false`，仍为 `sealed_unread`。
- preflight 证据：仓库外 `cloud/DVC-A1-valdev-boundary-zero-v2/preflight.json`，SHA-256 为 `95afc2f071656569a7f3ee18d3244c5f0abfc5ef59c349825e170ccc314c0e9d`。

## 三、实际检查

本次选择两个不同位置组的样本：

- `01-01-01-0078-240523102138-04-99`：q75 不可构造，五个 condition 的置零计数均为 0；
- `03-01-01-0066-240526121123-12-99`：q75 可构造，boundary-q25/q50/q75 分别置零 5/11/16 个像素，nonboundary-q50 为 11 个像素。

两样本均通过：

- 五个 condition 的 mask hash 在重复生成中稳定；
- 五个 condition 的 mask hash 与冻结 v1 manifest 逐项匹配；
- q25、q50、q75 mask 保持嵌套；
- boundary-q50 与 nonboundary-q50 置零数量相等；
- clean condition 的量化 Depth decoded array 与生产 Depth 完全相等；
- clean 与 boundary-q50 的五尺度翻转推理输出均为有限值；
- strict checkpoint load 成功；
- 输出恢复到原始 `932×1082` Label 网格。

运行设备为本地 CUDA，运行时长 `32.548` 秒，退出码为 0。终端同时出现既有第三方依赖弃用和 `torch.meshgrid` indexing warning；它们不影响本次门禁，未为消除无关 warning 扩大改动。

## 四、当前边界与下一步

本次 preflight 没有运行完整评价，也没有生成 mIoU、Boundary IoU、bootstrap 区间或 `supported/not-supported/inconclusive` 裁决。v1 的 `protocol-blocked` 结论和证据保持原样。

只有用户再次明确批准后，才运行固定 138 个位置组、218 张图的五 condition、五尺度翻转本地开发评价。训练、云资源和 official test 仍不在本次授权范围内。
