# MMFR E1 Depth Input Contract Audit

> **状态：** `completed-read-only-audit`  
> **日期：** 2026-09-21  
> **裁决：** R-EM-lite input detector `protocol-blocked`  
> **授权边界：** 未修改数据、模型或训练代码；未运行模型 forward、GPU、训练或评价。

## 1. 最终结论

当前模型可见输入只能判断“几何有效区中的 Depth 是否全为 0”，不能判断这些 0 是由 `entire_missing`、自然全无效 crop，还是 spatial dropout 恰好删空造成。三种输入可以逐值完全相同。

用户规则要求 R-EM-lite 只处理可合法区分的 `entire_missing`，且禁止读取隐藏 corruption label。因此 detector 必须判为 `protocol-blocked`。

**大白话说明：** 模型眼前只看到一张全零深度图，看不出它是传感器整幅丢失、这次 crop 天然没有有效深度，还是局部删除刚好删光。用生成器标签告诉模型答案属于作弊，所以当前不能实现这条路由。

## 2. 实际数值链路

### 2.1 原始数据到训练张量

1. raw Depth 以灰度 `uint8` 读取；
2. 复制为三个相同通道；
3. `TrainPre` 顺序：mirror → bilinear scale → normalization → crop/pad → CHW；
4. Depth normalization：mean `0.48`、std `0.28`；
5. raw zero 在真实图像有效区域归一化为：

$$
\frac{0/255-0.48}{0.28}
\approx -1.7142857.
$$

6. crop/pad 新增的 padding 在 normalized tensor 中是 exact `0.0`；
7. `modal_x` 的三个通道相同；
8. DFormerv2 只取 channel 0；
9. `GeoPriorGen` 再用 bilinear resize 生成四级 Depth grid。

### 2.2 四类像素值

- `clean-valid`：raw `1..255`，按 mean/std 正常归一化；
- `natural-invalid`：raw `0`，在真实图像区为 normalized `-1.7142857`；
- `entire_missing`：真实图像有效区全部 raw `0`，同样为 normalized `-1.7142857`；
- `spatial-dropout-invalid`：被删位置 raw `0`，同样为 normalized `-1.7142857`；
- padding：normalized exact `0.0`，不能与 raw-zero invalid 混为同一 sentinel。

因此 detector 若在 normalized space 中使用 `0` 作为 missing sentinel，会把 padding 当作 missing；若使用 `-1.7142857`，又无法区分三种 raw-zero 原因。

## 3. Observable-empty proxy

若只使用当前可部署输入和几何有效区，可以定义：

$$
\phi_{\mathrm{empty}}
=
\mathbf 1
\left[
\sum V_{\mathrm{geom}}>0
\land
\max_{V_{\mathrm{geom}}=1}D_{\mathrm{u8}}=0
\right].
$$

这里 `V_geom` 只用于排除 crop/pad 几何外区域；该 proxy 的语义是：

> 当前可见 crop 的真实图像区域中，没有任何非零 Depth。

它**不是**：

> 当前 corruption cause 一定是 `entire_missing`。

这个差异不能通过调整浮点容差、换 raw/normalized space 或查看三复制通道解决。

## 4. 冲突情形

### 4.1 `entire_missing`

generator 把几何有效区中的 Depth 全部置为 raw zero。observable tensor 满足 empty proxy。

### 4.2 natural-invalid-only crop

项目全量训练几何审计已记录 8 个在 `valid_mask` 内完全没有有效 Depth 的 crop。它们没有 synthetic `entire_missing`，但 observable tensor 同样满足 empty proxy。

### 4.3 spatial dropout 删空

spatial dropout 在某些原有效点很少的 crop 上可以删除全部剩余有效点。此时 observable tensor也满足 empty proxy。

因此存在：

$$
D_{\mathrm{observed}}^{\mathrm{entire\ missing}}
=
D_{\mathrm{observed}}^{\mathrm{natural\ empty}}
=
D_{\mathrm{observed}}^{\mathrm{dropout\ emptied}}
$$

而三者的 hidden cause 不同。不存在只依赖该 observable tensor 的确定性函数可以恢复原因标签。

## 5. 真实单样本 probe

只读 CPU 数值 probe 使用：

- sample：`06-01-01-0035-230920140169-12-99`；
- crop shape：`[3,480,640]`；
- geometry-valid：`307200`；
- natural-invalid：`305172`；
- clean-valid：`2028`；
- `entire_missing` 后：全部 `307200` 个几何有效像素为 raw 0；
- `spatial_dropout@0.75`：新增 invalid `1218`，剩余 valid `810`。

该 probe 说明真实 crop 中本来就可能只剩很少有效 Depth；继续抽样或在其他 crop 上应用 spatial dropout，可以到达与 entire-missing 相同的全零 observable state。该 probe 未运行模型 forward，也不提供 mIoU 或 detector 性能结论。

## 6. Privileged information 边界

R-EM-lite detector 和 substitute logits 路径禁止读取：

- corruption type；
- severity；
- synthetic corruption mask；
- clean Depth；
- `depth_valid_pre`；
- `depth_valid_post`；
- reliability supervision target；
- generator RNG state、seed、sample spec 或 manifest cause；
- “这条样本来自 entire_missing 分支”的任何布尔标记。

这些字段可用于数据生成、离线路由审计或报告，但不能作为训练/推理 detector 的输入。否则训练时和部署时信息不一致，并且比较不再是合法的 input-observable fallback。

## 7. 对 R-EM-lite 的裁决

### 7.1 不能采用的偷换

- 不能把 `observable-empty` 改名为 `entire_missing`；
- 不能声称 raw-space all-zero 就证明 corruption cause；
- 不能用 `depth_valid_pre/post` 排除 natural-invalid；
- 不能在训练时用 hidden label，推理时再改成 heuristic；
- 不能先实现并靠结果决定 detector 定义。

### 7.2 合法后继选择

只能由上级另行选择并建立新 identity：

1. 终止 R-EM-lite；
2. 改为明确命名的 `observable-empty` 路线，并承认它会同时处理三类全零输入；
3. 提供部署时合法可见、且能区分原因的新信号。

在该选择前，R-EM-lite detector 和 substitute network 都不得冻结或实现。

## 8. 未运行项

本轮没有模型 forward/backward、GPU、训练、Quick-Val、Main-Val、完整测试、云任务或 official test。只完成输入数值链、真实单样本数值和信息可见性边界的定点审计。
