# MUSeg DVC-A1-v2 保护修正与重新物化报告

> **文档角色：** 日期化实现与 protocol 重新物化记录。
> **形成或核验时点：** 2026-09-08 14:00 UTC。
> **实时入口：** [`MUSeg-current-status.md`](../main/MUSeg-current-status.md)。
> **前序报告：** [`2026-09-08-museg-dvc-a1-v2-materialization-preflight.md`](2026-09-08-museg-dvc-a1-v2-materialization-preflight.md)。

## 一、本阶段结论

已在不改变 DVC-A1-v2 的阈值、condition、allowlist、评价范围或统计定义的前提下，补齐完整运行前审查发现的两项保护：

1. 完整运行和新的 preflight 会把重新生成的 condition mask 哈希逐样本、逐 condition 与冻结 v1 mask manifest 的记录直接比较；运行内重新计算与扫描结果之间的确定性比较仍保留。
2. 完整运行在计算统计量前，显式强制主分析包含 138 个位置组，并强制主分析与 123 组敏感性分析的 dose effect、specificity effect 各自拥有对应数量的有效配对组；15 个部分可构造组只保留描述性分析。

大白话说，正式统计前现在同时防止 mask 被悄悄替换，以及有效配对组数量悄悄减少。

## 二、代码改动

- `tools/mve/run_dvc_a1.py` 新增冻结 v1 condition hash 读取和逐项比较逻辑。
- `tools/mve/run_dvc_a1.py` 的 preflight 增加冻结 v1 hash 检查，并把结果写入检查记录。
- `tools/mve/run_dvc_a1.py` 的完整运行增加 138 组主分析、123 组敏感性分析和两个 effect 的有效配对组数门禁；门禁在统计量和裁决前执行。
- `tests/test_dvc_a1.py` 增加 hash 漂移拒绝和有效配对组数门禁的聚焦测试。

v1 protocol、v1 mask manifest、v1 `protocol-blocked` 证据均未覆盖或回写。

## 三、重新物化结果

- v2 protocol 路径：仓库外 `cloud/DVC-A1-valdev-boundary-zero-v2/protocol.json`。
- 修正后 protocol SHA-256：`3c6f33562692c8baee85786261de431583e9bdd9b6f8f51cd5ad0f042406f6bb`。
- v2 allowlist SHA-256：`5589eb3378ed2e23180f6205e2d88cea39702ad4bfd5d4e1b739cf2f920a8d89`，未改变。
- allowlist 摘要 SHA-256：`f451955d91143d118f4445181ceb8ff7212ced5ba8cf2add12981cd1e75287c2`，未改变。
- 重新读取冻结 v1 manifest 的结果：218 个 allowlist 样本、每个样本 5 个 condition hash，且派生 allowlist 仍覆盖 138 个位置组。
- checkpoint、`val-dev` split 和 `official_test_included=false` 身份保持不变。

## 四、实际检查

- `python -m py_compile tools/mve/dvc_a1_protocol.py tools/mve/run_dvc_a1.py tests/test_dvc_a1.py`：通过。
- `python -m pytest tests/test_dvc_a1.py -q`：`7 passed`。
- Python 语言服务器对本次涉及的三个文件未报告诊断。
- `git diff --check`：通过。
- Protocol 重新物化命令：通过，返回 `protocol-ready` 和上述新 SHA-256。
- 冻结 manifest hash 参考读取检查：通过，读取 218 个样本、5 个 condition、138 个位置组。

pytest 仍报告工作区 `.pytest_cache` 无写权限的既有 warning；未为消除该无关 warning 扩大改动。

## 五、当前边界与恢复点

修正前的两样本 preflight 仍只作为历史证据，不能替代新 protocol 身份下的检查。新的两样本本地 GPU preflight 尚未运行，完整 218 张图、五 condition、五尺度翻转评价也尚未运行；因此当前没有 mIoU、Boundary IoU、bootstrap 区间或科学裁决。

下一步需要用户确认后，使用修正后 protocol 重跑两样本本地 GPU preflight。只有新的 preflight 通过并再次取得完整评价确认后，才可运行 138 个位置组、218 张图的完整本地开发评价。训练、云资源和 official test 仍不在授权范围内。
