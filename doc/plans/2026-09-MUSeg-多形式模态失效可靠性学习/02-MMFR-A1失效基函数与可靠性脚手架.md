# MMFR-A1 失效基函数与可靠性脚手架

> **文档角色：** 条件式子计划。
> **计划状态：** 已完成代码脚手架与 CPU qualification；未接入训练主链。
> **形成或核验时点：** 2026-09-13。
> **实时入口：** [`MUSeg-current-status.md`](../../main/MUSeg-current-status.md)。
> **研究选择：** [`MUSeg-open-decisions.md`](../../main/MUSeg-open-decisions.md)。
> **上级方向：** [`00-总方向规划.md`](00-总方向规划.md)。
> **当前/后继关系：** 本计划已完成；恢复点是由主代理冻结 A2 数据接入、RNG、以及 loss/protocol 身份，用户另行批准后才可修改训练主链或运行训练。

## 1. 实验族身份

- **语义名称：** `MMFR-A1-train-corruption-basis-v1`。
- **MMFR：** Multi-Modal Failure Robustness，多模态失效鲁棒性。
- **A1 职责：** 提供受控失效输入、连续可靠性监督和未来 adapter 的独立算子，不评价模型效果。
- **不主张：** A1 通过不代表 MUSeg 已适合真实失效结论，不代表 reliability head 已学会可靠性，也不代表 GSA 改进有效。

## 2. 目标与不变量

目标是新增两个不接入现有主链的模块，使后续训练可以在不重写数据加载和 DFormerv2 的前提下复用。现有 `TrainPre`、`RGBXDataset`、`EncoderDecoder`、`DFormerv2.py`、Quick-B0 配置与 checkpoint key 必须保持不变。

## 3. 代码执行单

### 3.1 `utils/dataloader/multimodal_failure.py`

必须提供：

- 显式 failure spec 与结构化结果；
- 调用方传入 `numpy.random.Generator`，不使用 `np.random`/`random` 全局状态；
- `entire_missing`、`spatial_dropout`、`gaussian_noise`、`blur`、`quantization`、Depth `misalignment`；
- 多 spec 顺序组合，连续 reliability target 使用各基函数负担的乘法组合；
- curriculum sampler 接受 $progress\in[0,1]$，只返回 spec，不隐式修改数据；
- RGB 为 uint8 HWC 三通道，Depth 接受 uint8 HW 或三通道同值 HWC，输出保持原 shape/dtype；
- 空 spec 返回逐元素严格相同数组、全 1 target 和空 metadata；
- 所有 severity、shape、modality、kind 和有限值 fail-closed。

首版不实现 haze、dust 合成、非刚性 warp、Poisson shot noise 或真实相机响应模型；这些需要单独 protocol，不能用名称暗示已实现。

### 3.2 `models/modal_reliability.py`

必须提供：

- 固定可微信号特征：亮度/Depth 强度、局部均值、局部标准差、Sobel 梯度、Laplacian、高频残差、Depth 有效值和跨模态梯度方向一致性；
- 小型卷积 head，输出 RGB/Depth 两通道 reliability logits，不在模块内部偷偷 detach；
- reliability pyramid 使用 PyTorch `area` 插值，输入输出保持 `[0,1]`；
- 四级 geometry contribution adapter，学习 spatial/depth 两个正值 scale，零初始化时严格输出 1；
- 连续 target 的稳定 BCE helper；
- shape、dtype、范围和有限值守卫。

首版 adapter 不接入 `GeoPriorGen`。它只定义未来接口，避免在 protocol 未冻结前改变 DVG-B1 已验证路径。

## 4. 易错点

- RGB 全零可能是真正缺失，也可能是极暗图；因此手工特征只能作为学习输入，不能直接成为 hard gate。
- RGB/Depth 边缘不总是一一对应；跨模态梯度一致性是提示而非真值。
- `torch.nn.functional.interpolate(mode="area")` 与 OpenCV `INTER_AREA` 不保证逐元素相等；A1 模型侧只声明 PyTorch pyramid 语义，不复用 DVG-B1 Oracle protocol。
- 新模块有参数，不能用 `strict=True` 直接加载 Quick-B0 checkpoint 并声称兼容。当前不接入主模型，因此原 checkpoint 身份不变。
- adapter 的初始输出必须是 1，而不是 0.5；否则即使未训练也会改变现有 prior。

## 5. 最小验证

只允许：

1. 对两个新文件运行 `python -m py_compile`；
2. 内联 CPU probe 核对空 spec no-op、同 seed 确定性、每类 corruption shape/dtype/range/metadata、混合可靠性不高于各单项；
3. 内联 PyTorch CPU probe 核对特征/head/pyramid shape 与 finite、target loss finite、adapter 初始 scale 严格为 1、一次 backward 梯度 finite。

不新建 `test_*.py`，不运行完整测试、GPU、训练、checkpoint load、评价、云任务或 official test。

## 6. 完成、阻塞与恢复

- **完成：** 两个文件满足执行单，最小验证通过，主代理查看实际 diff 并记录未接入边界。
- **`qualification-blocked`：** 任一确定性、no-op、shape、范围、finite 或 backward 检查失败；保留代码和准确错误，不扩大修改到现有主链。
- **恢复点：** A1 已完成；主代理先冻结 A2 的 dataset output、RNG/epoch progression、loss 权重、clean/corrupt forward 和 checkpoint identity，再决定是否请求训练接入授权。

## 7. 实际完成与资格检查

- 已新增 `utils/dataloader/multimodal_failure.py`：支持完整缺失、局部 dropout、Gaussian noise、Gaussian blur、Depth quantization 和 Depth translation misalignment；所有随机性来自调用方 `numpy.random.Generator`，多失效可靠性按 $R=\exp(-\sum b_k)$ 组合。完整缺失只在 curriculum 重度阶段出现且 severity 固定为 `1.0`；错位 target 由已知位移而不是图像纹理差值决定。
- 已新增 `models/modal_reliability.py`：提供 14 通道固定可微信号特征、两模态 reliability logits head、四级 PyTorch area pyramid、连续 target BCE，以及初始化严格为 scale `1`、学习范围严格在 $(0,2)$ 内的 geometry contribution adapter。
- 两个模块保持 standalone，未修改 `TrainPre`、`RGBXDataset`、`EncoderDecoder`、`DFormerv2.py`、配置或 evaluator；因此现有 checkpoint key 和运行路径未改变。
- 主代理复跑 `python -m py_compile` 与两个内联 CPU probe：空 spec no-op、同 seed 确定性、Depth-only guard、curriculum 重度完整缺失、输出 shape/range/finite、reliability pyramid、bool mask BCE、adapter exact-one 和 backward finite 全部通过；两个文件静态诊断无问题。
- 未运行完整测试、GPU、训练、checkpoint、评价、云任务或 official test。

**大白话说明：** 现在只有可复用的失效生成器和可学习可靠性模块，尚未接入训练，更没有模型效果结果；下一步不能直接开训，必须先固定 A2 怎样把 target、随机种子、loss 和四级 adapter 接进现有数据/模型链。
