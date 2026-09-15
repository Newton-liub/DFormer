# MMFR-A1 失效基函数与可靠性脚手架

> **文档角色：** 条件式子计划。
> **计划状态：** v1 已完成代码脚手架与 CPU qualification；v2 修订（`MMFR-A1-corruption-basis-v2`）已实现并通过 CPU severity/burden 审计；两个版本都不接入训练主链。
> **形成或核验时点：** 2026-09-13；v2 语义修订时点 2026-09-14（UTC）。
> **实时入口：** [`MUSeg-current-status.md`](../../main/MUSeg-current-status.md)。
> **研究选择：** [`MUSeg-open-decisions.md`](../../main/MUSeg-open-decisions.md)。
> **上级方向：** [`00-总方向规划.md`](00-总方向规划.md)。
> **当前/后继关系：** v1 计划已完成；v2 修订（`MMFR-A1-corruption-basis-v2`）已实现并通过 CPU severity/burden 审计。恢复点是由主代理冻结 A2 v2 数据接入、RNG、target 与 loss/protocol 身份，用户另行批准后才可修改训练主链或运行训练。

## 1. 实验族身份

- **v1 语义名称（冻结、历史身份）：** `MMFR-A1-train-corruption-basis-v1`。
- **v2 语义名称（当前身份）：** `MMFR-A1-corruption-basis-v2`，supersede `MMFR-A1-corruption-basis-v1`。v1 模块 `utils/dataloader/multimodal_failure.py` 保持冻结，v2 不导入也不改写它。
- **MMFR：** Multi-Modal Failure Robustness，多模态失效鲁棒性。
- **A1 职责（v2 口径）：** **synthetic failure generation + corruption-derived reliability surrogate**，即“受控合成失效生成 + 由 corruption 推导的可靠性代理目标”，并提供未来 adapter 的独立算子；A1 不评价模型效果。
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

### 3.3 `utils/dataloader/multimodal_failure_v2.py`（v2 修订，新增）

v2 模块声明 `PROTOCOL_ID = "MMFR-A1-corruption-basis-v2"`、`SUPERSEDES = "MMFR-A1-corruption-basis-v1"`、`SEVERITY_ENCODING = "single"`，是 v1 的修订版，不是别名：v1 文件保持冻结，v2 不导入 v1 的 kernel。修订原因是 v1 有两个性质在新 protocol 下不可审计：

1. **severity 被编码两次。** v1 的 `gaussian_noise`、`blur`、`quantization` 既用 severity 选择损坏强度，又把实得 damage burden 再乘一次 severity，使报告 burden 与实得归一化损伤不一致。v2 中 severity 只选择 corruption 参数，连续 burden 就是 realized normalized damage 本身。
2. **`misalignment` 与 `blur` 的 severity 是绝对像素量。** 同一 severity 在 `480×640` 训练 crop 与 `932×1082` 原始对齐网格上代表不同退化。v2 把两者定义为实际网格的相对比例，再按分辨率换算为像素。

v2 的 burden 语义（必须与 `local_configs/MUSeg` 中 v2 冻结常量、A2 v2 helper 及审计脚本一致）：

- **graded kinds（`gaussian_noise`、`blur`、`quantization`）：** realized normalized damage

$$
b(p)=\operatorname{clip}\!\left(\frac{\max_{\mathrm{channel}}|\Delta_{\mathrm{channel}}(p)|}{255\cdot \mathrm{DAMAGE\_REFERENCE}},\,0,\,1\right),
$$

其中 `DAMAGE_REFERENCE=0.25`。不再乘 severity。
- **`gaussian_noise` 参数：** $\sigma = \text{severity}\times 48$ 个 uint8 单位（`NOISE_SIGMA_MAX=48.0`，与 v1 强度含义一致）。
- **`blur` 参数（相对尺度）：** $\sigma = \text{severity}\times \mathrm{BLUR\_SIGMA\_FRACTION}\times \min(H,W)$，`BLUR_SIGMA_FRACTION=1/80`；该比例在冻结 `480` 像素 crop 上等价于 v1 的 `6.0` px 上限。
- **`quantization` 参数：** $\text{levels}=\operatorname{clip}(\operatorname{round}(256(1-\text{severity})),2,256)$，仍按强度空间量化。
- **`misalignment` 参数（相对尺度）：** 每轴最大位移为 $\text{severity}\times \mathrm{MISALIGN\_MAX\_SHIFT\_FRACTION}\times H$（y 轴）与 $\times W$（x 轴），`MISALIGN_MAX_SHIFT_FRACTION=1/30`；有效区 burden 为实际位移除以该尺度允许的最大位移：

$$
b_{\mathrm{in\text{-}bounds}}(p)=\min\!\left(1,\frac{\sqrt{dy^2+dx^2}}{\sqrt{\left(\mathrm{frac}\cdot H\right)^2+\left(\mathrm{frac}\cdot W\right)^2}}\right),
\qquad \mathrm{frac}=\mathrm{MISALIGN\_MAX\_SHIFT\_FRACTION}.
$$

越界像素显式无效，使用 `MISSING_BURDEN=1.0e4`。
- **`entire_missing` 与 `spatial_dropout`：** 结构性缺失，burden 为 `MISSING_BURDEN=1.0e4`；`spatial_dropout` 的 extent 仍由 severity 控制，网格 cells 数随 severity 单调。

v2 原样保留的冻结语义：三阶段 curriculum（轻度/中度/重度 severity 上限 `0.30`/`0.60`/`1.0`，spec 数 `1`/`1–2`/最多 `max_specs`，`entire_missing` 仅重度阶段且 severity 固定 `1.0`，severity 下限 `0.05`）、乘法可靠性组合 $R_m(p)=\prod_k\exp(-b_{m,k}(p))$、输出契约（uint8 原 shape/dtype、float32 `[2,H,W]` target、通道顺序 RGB 后 Depth、空 spec 严格 no-op）以及 unsupported spec/shape/dtype/非有限值的 fail-closed 校验。

**定性边界：** v2 的全部失效都是 **model-input/representation-level synthetic corruptions**（模型输入/表示层合成损坏），不是 Kinect 真实物理噪声模型，不得按物理噪声模型报告。

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
3. 内联 PyTorch CPU probe 核对特征/head/pyramid shape 与 finite、target loss finite、adapter 初始 scale 严格为 1、一次 backward 梯度 finite；
4. v2 修订另允许一次专门的 CPU 审计 `python tools/mmfr/severity_burden_audit.py`（只读脚本、纯 CPU、两个确定性合成 fixture）。

不新建 `test_*.py`，不运行完整测试、GPU、训练、checkpoint load、评价、云任务或 official test。

## 6. 完成、阻塞与恢复

- **完成：** 两个文件满足执行单，最小验证通过，主代理查看实际 diff 并记录未接入边界。
- **`qualification-blocked`：** 任一确定性、no-op、shape、范围、finite 或 backward 检查失败；保留代码和准确错误，不扩大修改到现有主链。
- **恢复点：** A1 已完成；主代理先冻结 A2 的 dataset output、RNG/epoch progression、loss 权重、clean/corrupt forward 和 checkpoint identity，再决定是否请求训练接入授权。

## 7. 实际完成与资格检查

以下条目是 A1 v1（`MMFR-A1-train-corruption-basis-v1`，`utils/dataloader/multimodal_failure.py`）的原始记录，原样保留；v2 修订见第 8 节。

- 已新增 `utils/dataloader/multimodal_failure.py`：支持完整缺失、局部 dropout、Gaussian noise、Gaussian blur、Depth quantization 和 Depth translation misalignment；所有随机性来自调用方 `numpy.random.Generator`，多失效可靠性按 $R=\exp(-\sum b_k)$ 组合。完整缺失只在 curriculum 重度阶段出现且 severity 固定为 `1.0`；错位 target 由已知位移而不是图像纹理差值决定。
- 已新增 `models/modal_reliability.py`：提供 14 通道固定可微信号特征、两模态 reliability logits head、四级 PyTorch area pyramid、连续 target BCE，以及初始化严格为 scale `1`、学习范围严格在 $(0,2)$ 内的 geometry contribution adapter。
- 两个模块保持 standalone，未修改 `TrainPre`、`RGBXDataset`、`EncoderDecoder`、`DFormerv2.py`、配置或 evaluator；因此现有 checkpoint key 和运行路径未改变。
- 主代理复跑 `python -m py_compile` 与两个内联 CPU probe：空 spec no-op、同 seed 确定性、Depth-only guard、curriculum 重度完整缺失、输出 shape/range/finite、reliability pyramid、bool mask BCE、adapter exact-one 和 backward finite 全部通过；两个文件静态诊断无问题。
- 未运行完整测试、GPU、训练、checkpoint、评价、云任务或 official test。

**大白话说明：** 现在只有可复用的失效生成器和可学习可靠性模块，尚未接入训练，更没有模型效果结果；下一步不能直接开训，必须先固定 A2 怎样把 target、随机种子、loss 和四级 adapter 接进现有数据/模型链。

## 8. A1 v2 修订与 severity/burden 审计

### 8.1 v2 实际完成与身份

- 已新增 `utils/dataloader/multimodal_failure_v2.py`：声明 `protocol_id=MMFR-A1-corruption-basis-v2`、supersede `MMFR-A1-corruption-basis-v1`、`severity_encoding=single`。六类 kind、乘法可靠性组合、structural missing 的 `MISSING_BURDEN=1.0e4`、空 spec 严格 no-op 与 fail-closed 校验沿用 v1；severity 双编码与绝对像素尺度两处已按第 3.3 节修订。
- v1 模块 `utils/dataloader/multimodal_failure.py` 与 v1 config 保持冻结、未被改写；A1 v2 模块不导入 v1 kernel。A2 v2 helper 只从 v1 helper 与 v1 basis 复用与协议语义无关的基础设施与常量（sample-id hash、seed words 顺序、uint8 round-trip、curriculum 常量），并在 import 时断言两边 curriculum 维度一致，因此 v1 的数值语义不会被 v2 修改。
- 修订时点说明：正式 500 epoch 训练尚未执行，因此这是**结果产生前的协议修订**，不是结果后调整。

### 8.2 severity -> burden 审计证据

- **脚本：** `tools/mmfr/severity_burden_audit.py`，纯 CPU、只读，审计链条为 `severity -> corruption parameter -> realized damage -> burden -> Depth target`。
- **运行结果（已在 CPU 上运行通过）：** `python tools/mmfr/severity_burden_audit.py` 退出码 `0`，stdout 末行 `OVERALL: PASS`，报告 `conclusions.violation_count=0`。
- **报告路径：** `outputs/mmfr-a1-v2-severity-audit/severity-burden-audit.json`。
- **报告 SHA-256：** `86e468c040648a790675456db1b828a5a71e560b849c75b4b7e9a3ca926c2f52`（上级代理已复跑并得到同一哈希）。
- **通过的具体检查（全部为真）：** `no_double_severity_encoding`、`monotonic_burden`、`monotonic_target`、`relative_scale_consistent`、`dropout_extent_monotone`、`empty_spec_strict_noop`。
- **审计方法要点：** 每次 `(kind, severity)` 运行从固定 `FIXED_SEED` 重建 RNG；预期 burden 由脚本独立于模块、从返回的 corrupted array 与 fixture 重算，并要求与模块可靠性目标逐位一致；审计记录 `severity -> corruption parameter -> realized damage -> burden -> target` 映射。

### 8.3 审计边界（不得夸大）

- 审计使用两个**确定性合成 fixture**（`train_crop_480x640` 与 `aligned_grid_932x1082`），不是真实 MUSeg 样本；它证明的是 v2 编码与数值语义在 CPU 上自洽，不证明真实样本行为、模型效果、GPU 或训练行为。
- 审计是 CPU-only，不覆盖 AMP/FP16、Synced BN、真实 forward/backward 或任何训练/评价路径。
- A1 v1 与 v2 都不接入训练主链；A1 通过不等于 reliability head 已学会可靠性，也不等于 segmentation 已获得鲁棒性。
