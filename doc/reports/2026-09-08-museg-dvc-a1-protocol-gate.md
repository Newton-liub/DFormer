# MUSeg DVC-A1 协议实现与门禁报告

> **文档角色：** 日期化实现与协议门禁报告。
> **形成或核验时点：** 2026-09-08 03:00 UTC。
> **实时入口：** [`MUSeg-current-status.md`](../main/MUSeg-current-status.md)。
> **研究选择：** [`MUSeg-open-decisions.md`](../main/MUSeg-open-decisions.md)。
> **后继关系：** `DVC-A1-valdev-boundary-zero-v1` 已在完整评价前裁决为 `protocol-blocked`；下一步只能先处置无可构造 q75 边界候选的开放问题，不能直接进入 `DVG-B1`。

- 汇报周期：2026-09-08 02:23 UTC 至 2026-09-08 03:00 UTC
- 报告对象：MUSeg 几何可信 RGB-D 双路径最小可行实验的研发与交接
- 证据边界：Git HEAD `d7c175372a7c58c71b0ebc9f79989e722d92fe8b` 上的未提交 DVC-A1 实现，以及仓库外 `cloud/DVC-A1-valdev-boundary-zero-v1/attempt-2/` 运行证据
- 当前状态：最小实现与两样本 GPU preflight 已通过，但 318 条 `val-dev` 的全量 mask 门禁发现 58/196 个 location group 无法构造非空 q75，超过预注册 5% 上限，故在任何完整五条件模型评价前停止。

## 一、本阶段结论

本轮已经把 `DVC-A1-valdev-boundary-zero-v1` 从计划落实为可审计代码和独立 protocol（实验身份与不可变口径），并验证了确定性 mask、q=0 输入等价、标准 Boundary IoU 小例和 2 个真实 `val-dev` 样本的五尺度翻转推理。全量 mask 扫描随后触发预注册停止条件：58/196 个 location group 没有任何样本能形成非空 `boundary-q75`，占 `29.5918%`，高于 `5%` 上限；非边界同面积对照不足为 0 条。

大白话说，实现链路能工作，但当前“相对深度跳变不低于 0.05”的边界定义在太多地点里找不到可施加 75% 剂量的边界像素。协议要求这种情况必须先停，因此没有产生五条件全量 mIoU、Boundary IoU、bootstrap 区间或 `supported/not-supported/inconclusive` 科学裁决。

## 二、关键概念与判断边界

- **Protocol（实验协议）：** 固定 checkpoint、split、输入、变量、指标和停止条件的机器可读合同。本轮物化协议 SHA-256 为 `7bca3c109905d7d4ed228359bf8cd2a20bee18cbcd3d4cbdfed871e09870fba2`。
- **Preflight（最小运行前检查）：** 在少量样本上检查输入、数值和推理链，而不是科学评价。本轮 2 样本 preflight 通过只能证明实现可运行，不能证明研究假设成立。
- **Location group（位置组）：** 样本名按 ASCII 连字符切分后的前四段；同一位置组内图像相关，计划以 196 个组作为 bootstrap 单位。
- **q75 可构造性：** 一个位置组至少有一个样本的边界候选集合能产生非空 75% 前缀 mask。本轮 58 个组不满足，触发 `protocol-blocked`。
- **`protocol-blocked`：** 协议输入或操作定义无法覆盖预注册要求，必须在看完整模型结果前停止。它不是模型效果的负结论。

## 三、代码与协议交付

### 3.1 独立协议与物化

- 状态：已完成并验证。
- `protocols/dvc-a1-valdev-boundary-zero-v1.template.json` 固定 epoch 420 checkpoint、318 样本/196 组 `val-dev`、RGB/Depth/Label 契约、五尺度翻转 evaluator、五个 condition、Boundary IoU、10,000 次 location-group bootstrap、联合裁决和 official-test 拒绝规则。
- `tools/mve/dvc_a1_protocol.py` 提供 DVC 专用严格 schema、资产哈希/样本身份检查、源码哈希记录和仓库外物化。
- 最终物化协议位于 `cloud/DVC-A1-valdev-boundary-zero-v1/attempt-2/protocol.json`；其 SHA-256 为 `7bca3c109905d7d4ed228359bf8cd2a20bee18cbcd3d4cbdfed871e09870fba2`。

### 3.2 Corruption、Boundary IoU 与统计核心

- 状态：已完成并完成定点验证，但未进入全量模型评价。
- `tools/mve/dvc_a1_core.py` 实现原始 `uint16 Depth16` 上的相邻相对跳变、边界候选/guard band、稳定 SplitMix64 排序、q25/q50/q75 嵌套前缀、同面积非边界对照，以及复用生产 `quantize_depth` 的 16→8 bit 量化。
- 同一文件实现 one-vs-rest 内侧 Boundary IoU、ignore 邻域排除、图内类 macro、组内图像等权、固定 PCG64 bootstrap 和冻结联合裁决。
- `tests/test_dvc_a1.py` 覆盖 protocol 身份、mask 确定性与嵌套、q50 面积匹配、q=0 量化、Boundary IoU 的 identical/one-empty/both-empty/ignore 小例、bootstrap 可重复和裁决阈值。

### 3.3 五尺度翻转评价入口

- 状态：已完成并通过两样本 GPU preflight。
- `tools/mve/run_dvc_a1.py` 复用 `tools/evaluate_museg_checkpoint.py` 的 RGB 顺序、五尺度翻转 view、strict checkpoint load 和 FP32 pre-softmax logits 融合；只在内存中生成各 condition 的 Depth，不保存 logits、预测图或退化数据集。
- 运行入口先核验协议/源码/资产身份，再执行 mask 全量门禁；只有门禁通过才加载模型执行 318×5 条件评价。实际全量尝试在 mask 门禁停止，因此没有浪费后续 GPU 评价预算。

## 四、实际验证与结果

### 4.1 定点代码检查

- `python -m py_compile tools/mve/dvc_a1_core.py tools/mve/dvc_a1_protocol.py tools/mve/run_dvc_a1.py tests/test_dvc_a1.py`：通过。
- `python -m pytest tests/test_dvc_a1.py -q`：`5 passed`。pytest 因工作区 `.pytest_cache` 无写权限产生 1 条 cache warning；它不影响测试断言，也未为消除无关 warning 修改项目。
- 变更文件静态诊断：未发现诊断错误。

### 4.2 两样本 GPU preflight

- 模型：DFormerv2-S RGB Quick-B0，epoch 420 checkpoint，SHA-256 `f246a3afc50334c81302b7bfebdadf7cf37d00326bf1c3aa54f6a151754e3a1c`。
- 数据：`val-dev` 前 2 条；split SHA-256 `1d0719d8f64f016d48995c25ab66d4004d76b7155d9efeef7cbb7454c0dd0e83`。
- 环境：Python `3.13.9`、PyTorch `2.7.0+cu128`、CUDA `12.8`、本地 CUDA 设备。
- 结果：mask hash 稳定、q25/q50/q75 嵌套、boundary/nonboundary q50 数量相等、q=0 与现有 `Depth/` decoded array 完全相等；clean 与 boundary-q50 的 4 次样本-condition 推理均为有限值并恢复到 `932×1082` 原始 Label 网格。
- 证据：`cloud/DVC-A1-valdev-boundary-zero-v1/attempt-2/preflight.json`，SHA-256 `aeb0afff829fc5abcd86196cb6c27822055b90c7910711c2da956a2133dc0c45`；执行退出码 0，用时 `28.346` 秒；`official_test_included=false`。

### 4.3 全量 mask 门禁

- 范围：完整 318 条 `val-dev`、196 个 location group；只读取构造 mask 所需的 Depth16，没有执行五条件全量模型推理。
- 结果：318 条均完成 mask 扫描；58/196 个组 q75 不可构造，比例 `29.5918%`，超过上限 `5%`；非边界 q50 候选不足为 0 条。
- 裁决：`protocol-blocked`，执行退出码 2，用时 `31.410` 秒；停止发生在完整 GPU 模型评价前。
- 证据：
  - `cloud/DVC-A1-valdev-boundary-zero-v1/attempt-2/mask-manifest.json`，SHA-256 `60b988b3f9ffaabc5f6540cfccda48ddb5efd4d44ce360691bfaeea047e63f29`；
  - `cloud/DVC-A1-valdev-boundary-zero-v1/attempt-2/full-failure.json`，SHA-256 `b36d9bddfb33a4f69657dae976f94afa6e97282a42aec27663949502be08e216`；
  - `cloud/DVC-A1-valdev-boundary-zero-v1/attempt-2/executions/20260908T025721804739+0000-full.json`。

## 五、结论边界与当前风险

- 当前证据证明 DVC-A1 的最小实现、输入等价门禁和真实样本推理链可以运行，也证明预注册边界候选定义在当前 `val-dev` 上覆盖不足。
- 当前证据不能说明边界置零是否降低 Boundary IoU，因为五条件完整模型评价没有开始。
- 不得把 `protocol-blocked` 写成 `not-supported`；后者要求协议有效并实际得到不支持假设的数值。
- 不得直接降低 `0.05` 阈值、删除 58 个不利组、把空 q75 当正常 clean 副本或提高 5% 上限。任何一种处置都会改变数值语义，必须在看完整结果前明确选择并建立新 protocol identity。
- `DVG-B1-oracle-gsa-v1` 仍未解锁；它要求有效 `DVC-A1=supported`，而本轮停在协议门禁。

## 六、准确恢复点

下一步是处置“如何定义足够覆盖全 `val-dev` 的深度边界候选”这一开放研究选择。允许的恢复动作仅限先检查本次 mask manifest 中零候选/少候选的分布，并在不读取模型五条件结果的前提下预注册新定义；若改变相对跳变阈值、候选构造、剂量可构造规则或 5% 上限，使用新的 protocol identity，例如 `DVC-A1-valdev-boundary-zero-v2`，重新物化并重跑 preflight 和 mask 门禁。

本轮没有运行完整项目测试套件、完整五条件 GPU 评价、训练、云任务、多 seed、额外 condition、阈值搜索或 official test。完整五条件评价未运行的原因是预注册 mask 门禁已经失败，继续运行会违反停止条件。
