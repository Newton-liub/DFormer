# MMFR E1 Batch 1 Optimizer Coverage Audit

> **状态：** `completed-read-only-audit`  
> **日期：** 2026-09-21  
> **裁决：** `coverage bug + Batch 1 全组统一修复`  
> **授权边界：** 本文件只冻结后续协议语义；未修改 optimizer 实现，未训练或运行 GPU。

## 1. 结论

`utils/init_func.py::group_weight` 通过 `module.modules()` 遍历模块，再用 `elif isinstance(m, nn.Parameter)` 尝试捕获裸参数。裸 `nn.Parameter` 不是子模块，因此该分支不能发现注册为模块属性的 `Geo.weight`。当前路径也没有“所有 `requires_grad=True` 参数恰好进入一个 group”的完整性断言。

历史 H43 清单包含 43 个参数，但当前 A2 v3 使用 SyncBN 后，实际 optimizer 缺组数是 **29**：全部为 `Geo.weight`。其余 14 个 BatchNorm 参数已经进入 no-decay group 并更新。

**大白话说明：** 旧审计把 43 个都当作没训练到；对当前 A2 v3 来说，真正没进入 optimizer 的只有 29 个几何权重。Batch 1 必须三组一起修，不能只给候选修。

## 2. 审计身份

- source checkpoint：`experiments/MMFR_A2_v3/checkpoints/selector-epoch-420.pth`；
- checkpoint SHA-256：`2b72eb7f28e5c4b4f52bdc43e2ac5c7298169325a412bfbcbc7a022c29822597`；
- model parameter tensors：720；
- saved optimizer group sizes：276 + 415 = 691；
- optimizer state entries：691；
- current missing membership count：29。

所有 H43 参数均满足：

- `requires_grad=True`；
- checkpoint 中存在对应 key；
- 历史 Quick-B0 `--no-syncbn` 路径下 membership 为 0，且与 pretrained 保持相同。

当前 A2 v3 的差异是 SyncBN 让其中 14 个 norm 参数成为可识别模块参数并进入 optimizer。

## 3. 完整 H43 清单

### 3.1 当前实际缺组：29 个 `Geo.weight`

Stage 0，3 个 block：

1. `backbone.layers.0.blocks.0.Geo.weight`
2. `backbone.layers.0.blocks.1.Geo.weight`
3. `backbone.layers.0.blocks.2.Geo.weight`

Stage 1，4 个 block：

4. `backbone.layers.1.blocks.0.Geo.weight`
5. `backbone.layers.1.blocks.1.Geo.weight`
6. `backbone.layers.1.blocks.2.Geo.weight`
7. `backbone.layers.1.blocks.3.Geo.weight`

Stage 2，18 个 block：

8. `backbone.layers.2.blocks.0.Geo.weight`
9. `backbone.layers.2.blocks.1.Geo.weight`
10. `backbone.layers.2.blocks.2.Geo.weight`
11. `backbone.layers.2.blocks.3.Geo.weight`
12. `backbone.layers.2.blocks.4.Geo.weight`
13. `backbone.layers.2.blocks.5.Geo.weight`
14. `backbone.layers.2.blocks.6.Geo.weight`
15. `backbone.layers.2.blocks.7.Geo.weight`
16. `backbone.layers.2.blocks.8.Geo.weight`
17. `backbone.layers.2.blocks.9.Geo.weight`
18. `backbone.layers.2.blocks.10.Geo.weight`
19. `backbone.layers.2.blocks.11.Geo.weight`
20. `backbone.layers.2.blocks.12.Geo.weight`
21. `backbone.layers.2.blocks.13.Geo.weight`
22. `backbone.layers.2.blocks.14.Geo.weight`
23. `backbone.layers.2.blocks.15.Geo.weight`
24. `backbone.layers.2.blocks.16.Geo.weight`
25. `backbone.layers.2.blocks.17.Geo.weight`

Stage 3，4 个 block：

26. `backbone.layers.3.blocks.0.Geo.weight`
27. `backbone.layers.3.blocks.1.Geo.weight`
28. `backbone.layers.3.blocks.2.Geo.weight`
29. `backbone.layers.3.blocks.3.Geo.weight`

当前 A2 v3 中，这 29 个参数 membership 为 0。

### 3.2 当前已覆盖：8 个 patch-embed SyncBN 参数

30. `backbone.patch_embed.proj.1.weight`
31. `backbone.patch_embed.proj.1.bias`
32. `backbone.patch_embed.proj.4.weight`
33. `backbone.patch_embed.proj.4.bias`
34. `backbone.patch_embed.proj.7.weight`
35. `backbone.patch_embed.proj.7.bias`
36. `backbone.patch_embed.proj.10.weight`
37. `backbone.patch_embed.proj.10.bias`

当前 A2 v3 中，这 8 个参数 membership 为 1，属于 no-decay group，并已相对初始化发生更新。

### 3.3 当前已覆盖：6 个 downsample SyncBN 参数

38. `backbone.layers.0.downsample.norm.weight`
39. `backbone.layers.0.downsample.norm.bias`
40. `backbone.layers.1.downsample.norm.weight`
41. `backbone.layers.1.downsample.norm.bias`
42. `backbone.layers.2.downsample.norm.weight`
43. `backbone.layers.2.downsample.norm.bias`

当前 A2 v3 中，这 6 个参数 membership 为 1，属于 no-decay group，并已相对初始化发生更新。

## 4. 冻结修复合同

Batch 1 的 C0、R-EM-lite、F-lite 必须共同使用同一新 optimizer identity：

1. 继续使用 AdamW；
2. 保留现有 `group_weight` 对已覆盖共享参数的 decay/no-decay 分类；
3. 显式把 29 个 `Geo.weight` 加入 base decay group；
4. 8+6 个 SyncBN 参数继续位于 base no-decay group；
5. 新模块 Conv/Linear weight 位于 new-module decay group；
6. 新模块 bias/norm 位于 new-module no-decay group；
7. 每个共享及新增 `requires_grad=True` 参数的 membership 必须恰好为 1；
8. 禁止只对 R/F 修复，C0 必须采用同一覆盖修复；
9. 禁止把 optimizer 修复收益归因于 R/F；
10. weights-only restart 不恢复旧 optimizer，因此不迁移 epoch-420 optimizer state。

建议固定 group 名称：

- `base_decay`；
- `base_no_decay`；
- `new_decay`；
- `new_no_decay`。

C0 的两个 new group 可以为空，但 group schema 与审计字段必须一致。

## 5. 实施验收

只有未来取得代码授权后，最小验收才包括：

- 枚举全部 `named_parameters()`；
- 对每个 `requires_grad=True` 参数计算 membership count；
- count 必须全部为 1；
- 29 个 `Geo.weight` 全部且仅在 `base_decay`；
- 14 个 SyncBN 参数全部且仅在 `base_no_decay`；
- F-lite 的 173152 个新增参数按 weight/bias 分入 new groups；
- reliability head 3 个 weight decay、3 个 bias no-decay；
- 记录每组参数数、元素数、LR、weight decay 与 name digest；
- 任一缺组、重复、意外改组或非有限梯度直接 `protocol-blocked`。

## 6. 未运行项

本轮只做只读 checkpoint/source 语义审计和文档冻结。没有修改 `group_weight`，没有运行模型、optimizer step、项目测试、GPU、训练、云任务或 official test。
