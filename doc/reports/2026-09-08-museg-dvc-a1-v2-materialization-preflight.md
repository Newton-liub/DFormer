# MUSeg DVC-A1-v2 物化与 Preflight 报告

> **文档角色：** 日期化实现、协议物化与最小运行前检查报告。
> **形成或核验时点：** 2026-09-08 09:43 UTC。
> **实时入口：** [`MUSeg-current-status.md`](../main/MUSeg-current-status.md)。
> **研究选择：** [`MUSeg-open-decisions.md`](../main/MUSeg-open-decisions.md)。
> **后继关系：** `DVC-A1-valdev-boundary-zero-v2` 已完成独立 protocol、138 组/218 样本 evaluation allowlist 物化和两样本本地 GPU preflight；完整五条件评价尚未运行，需用户另行确认。

- 汇报周期：2026-09-08 08:29 UTC 至 2026-09-08 09:43 UTC
- 报告对象：MUSeg 几何可信 RGB-D 双路径最小可行实验的研发与交接
- 证据边界：Git HEAD `90eee4a4ad5297cd513073f4791fe99d41eac90c` 上的当前未提交 v2 实现，以及仓库外 `cloud/DVC-A1-valdev-boundary-zero-v2/` 运行证据
- 当前状态：v2 已达到 `preflight-passed`，可以请求一次 138 组/218 样本的完整五条件本地 GPU 开发评价；该评价、训练、云资源与 official test 均尚未获批或运行。

## 一、本阶段结论

本轮把已冻结的 `DVC-A1-valdev-boundary-zero-v2` 从研究规划落实为机器可校验的独立协议和只读派生评价清单，并完成了计划要求的两样本本地 GPU preflight（运行前最小检查）。评价范围精确为 138 个位置组、218 张图；其中 31 张图的 q75 实际置零数为 0，123 个组内每张图都可构造非空 q75，15 个组为部分可构造。

大白话说，v2 的“评哪些地点、评哪些图片、哪些图片实际上没有被 q75 改动”都已经固定并核验，少量真实样本也确认能按新身份正确进入模型。现在还没有运行完整评价，因此没有 mIoU、Boundary IoU、bootstrap 区间或问题假设裁决。

## 二、关键概念与判断边界

- **Evaluation allowlist（评价允许清单）：** 从冻结 `val-dev` 和 v1 mask manifest 按预注册规则派生的 218 条样本清单。它只限定本次最小可行实验的评价范围，不是新的训练/验证划分。
- **Preflight（运行前最小检查）：** 用少量样本检查身份、mask、输入等价性、模型加载、数值和输出网格。通过只说明链路可运行，不说明研究假设成立。
- **部分可构造组：** 组内至少一张图可构造非空 q75，但仍包含 q75 置零数为 0 的图。主分析保留这些图并如实记录，不能把它们描述成实际接受了 q75 干预。
- **开发证据：** `val-dev` 已参与 checkpoint 选择；后续完整结果只能称为配对开发证据，不能称为独立测试结果。

## 三、代码、协议与清单

### 3.1 独立 v2 protocol

- 状态：已完成并验证。
- 新增 `protocols/dvc-a1-valdev-boundary-zero-v2.template.json`，固定 v2 身份、阈值 `0.05`、五种 condition、五尺度翻转 evaluator、138/218 范围、31/123/15 构造性统计、123 组敏感性分析和 `official_test_included=false`。
- `tools/mve/dvc_a1_protocol.py` 现同时严格校验 v1 与 v2；v2 物化必须绑定 v1 mask manifest 的 SHA-256，并生成只读 allowlist 和构造性摘要。v1 原 protocol 与证据未被覆盖或改写。
- 最终 v2 protocol SHA-256：`bc71ee97421a97f1ae172ec5f1096bf026750a3e4fae539eccfffc2c53aca535`。

### 3.2 评价清单与构造性摘要

- 状态：已完成并验证。
- allowlist：`cloud/DVC-A1-valdev-boundary-zero-v2/val-dev-constructable-v2.txt`，218 条且 218 条唯一，SHA-256 `5589eb3378ed2e23180f6205e2d88cea39702ad4bfd5d4e1b739cf2f920a8d89`。
- 派生来源：v1 `mask-manifest.json`，SHA-256 `60b988b3f9ffaabc5f6540cfccda48ddb5efd4d44ce360691bfaeea047e63f29`；原 `val-dev` split SHA-256 `1d0719d8f64f016d48995c25ab66d4004d76b7155d9efeef7cbb7454c0dd0e83`。
- `allowlist-summary.json` SHA-256：`f451955d91143d118f4445181ceb8ff7212ced5ba8cf2add12981cd1e75287c2`。
- 核验结果：138 个纳入组、58 个排除组、218 张图、31 张 q75 为空图、123 个全可构造组和 15 个部分可构造组，与预注册计划完全一致。

### 3.3 运行入口

- 状态：已完成并通过定点验证。
- `tools/mve/run_dvc_a1.py` 按 protocol 自动选择 v1 全 `val-dev` 或 v2 allowlist；v2 preflight 强制选择两个不同位置组，且覆盖一个 q75 为空样本和一个 q75 非空样本。
- v2 完整评价入口已准备为主分析 138 组，并预先生成 123 个全可构造组的敏感性分析及 15 个部分可构造组的描述性分析。该分支尚未执行，因此不能视为结果已验证。

## 四、实际验证

### 4.1 聚焦代码检查

- `python -m py_compile tools/mve/dvc_a1_core.py tools/mve/dvc_a1_protocol.py tools/mve/run_dvc_a1.py tests/test_dvc_a1.py`：通过。
- `python -m pytest tests/test_dvc_a1.py -q`：`5 passed`。pytest 仅报告工作区 `.pytest_cache` 无写权限的 cache warning；未为消除无关 warning 扩大改动。
- Python 语言服务器对 `dvc_a1_protocol.py` 未发现诊断；`run_dvc_a1.py` 只保留当前编辑器环境无法解析 `cv2`、`numpy`、`torch` 的既有环境诊断，实际同一 Python 环境的编译、测试和 GPU 运行均成功。

### 4.2 两样本本地 GPU preflight

- 模型：冻结的 DFormerv2-S RGB Quick-B0 epoch 420 checkpoint，SHA-256 `f246a3afc50334c81302b7bfebdadf7cf37d00326bf1c3aa54f6a151754e3a1c`。
- 评价身份：v2 protocol SHA-256 `bc71ee97421a97f1ae172ec5f1096bf026750a3e4fae539eccfffc2c53aca535`；allowlist 共 218 张图、138 个位置组。
- 样本覆盖：`01-01-01-0078-240523102138-04-99` 的 q75 为空；`03-01-01-0066-240526121123-12-99` 的 q75 置零数为 16，来自另一个位置组。
- 检查结果：两个样本的 mask hash 稳定、q25/q50/q75 嵌套、boundary-q50 与 nonboundary-q50 面积相等、q=0 decoded array 完全等价。
- 推理结果：两个样本的 clean 与 boundary-q50 共 4 次五尺度翻转推理均为有限值，strict checkpoint load 成功，输出均恢复到原始 `932×1082` Label 网格。
- 环境：Python `3.13.9`、PyTorch `2.7.0+cu128`、CUDA `12.8`、本地 CUDA 设备；用时 `38.283` 秒，退出码 0。
- 证据：`preflight.json` SHA-256 `9129103d5b3612f21dd7c86291a46a6a3102b4313b30357c4904ced639736e15`；执行记录 `20260908T094227207884+0000-preflight.json` SHA-256 `aa7f32aa036bf9eb9ab086a43eeb17dda59da05d24fea120afb6da826881a89f`；均记录 `official_test_included=false`。

## 五、当前边界与准确恢复点

- v2 当前只证明 protocol、allowlist 和最小运行链闭合；没有科学指标或 `supported/not-supported/inconclusive` 裁决。
- v1 的 `protocol-blocked`、protocol、mask manifest、preflight 和失败证据保持历史原样，v2 没有覆盖或回写它们。
- `DVG-B1-oracle-gsa-v1` 仍未解锁；它要求有效的 `DVC-A1=supported`，而完整 v2 评价尚未运行。
- 准确恢复点：用户另行确认后，使用当前 v2 protocol 对 138 个纳入位置组的全部 218 张图运行一次五条件、五尺度翻转本地 GPU 开发评价；完成后先核验 condition 覆盖、mask hash、有限值和证据哈希，再形成主分析、123 组敏感性分析与科学裁决。

本轮没有运行完整五条件 GPU 评价、完整项目测试套件、训练、云任务、多 seed、额外 condition、阈值搜索或 official test。完整评价未运行的原因是当前授权只推进到 protocol/allowlist 物化与最小 preflight，且计划明确要求 preflight 通过后再次确认。
