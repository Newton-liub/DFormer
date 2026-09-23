# R-OE-lite 设计冻结

> **名称：** R-OE-lite（Observable-Empty Geometry Substitute）  
> **文档状态：** `design-frozen, implementation-complete, Gate-B-PASS, formal-training-not-authorized`；架构冻结文本保持不变，当前实现身份与资格证据见文末 17 节。
> **候选语义：** observable-empty（可观测为空），不是 `entire_missing` 原因识别。  
> **适用边界：** 本文把原先被阻塞的 R-EM-lite 改写为一个明确承认输入不可辨因的候选；它不把既有 `R-EM-lite` 协议追溯标记为已批准。

## 1. 一句话裁决

R-OE-lite 只在当前几何有效区 `V_geom` 内的当前 raw Depth 全为零时触发，用 RGB 生成一个 task-driven Depth substitute；否则对 Depth 执行 exact bypass。该 observable-empty 状态故意同时覆盖 synthetic `entire_missing`、natural-empty crop 和被 spatial dropout 删空的输入，因此 R-OE-lite 不能声称识别 corruption cause。

**大白话说明：** 模型只确认“眼前这块几何有效区域没有任何非零 Depth”，不判断它究竟是整幅传感器丢失、自然没有有效深度，还是 dropout 恰好删光。

## 2. 允许使用的信息与明确禁止的信息

### 2.1 Detector 的唯一输入

Detector 的函数签名固定为当前样本的：

- `raw_depth`：当前 corruption 后导出的 raw Depth；
- `V_geom`：当前 crop/pad 产生的几何有效区。

Detector 不读取 RGB，也不读取已经归一化的 Depth；RGB 只供 substitute 使用。Detector 不读取任何原因标签或训练辅助字段。

### 2.2 禁止输入

以下信息不得进入 detector、substitute 的输入、路由逻辑或 substitute 的 loss：

- corruption type、severity、synthetic corruption mask；
- clean Depth；
- `depth_valid_pre`、`depth_valid_post`；
- reliability target；
- generator state、seed、sample spec、manifest cause 或分支标签；
- reliability、condition、severity 或 oracle 信号。

这些字段即使在数据生成或审计中存在，也不构成部署时可见输入。

## 3. Detector 精确定义

在每个样本内，从 v3 batch builder 导出的 `raw_depth` 恢复当前 raw uint8-like 平面：

$$
D_{\mathrm{u8}} = \operatorname{round}(255\,D_{\mathrm{raw}}),
\qquad
D_{\mathrm{raw}}=\operatorname{raw\_depth}[:,0,:,:].
$$

其中 `raw_depth` 的值来自 uint8 transport 除以 255，因此不得使用模糊阈值。几何有效区定义为：

$$
V_{\mathrm{geom}}=\operatorname{valid\_mask}[:,0,:,:].\operatorname{bool}.
$$

R-OE-lite trigger 为：

$$
\phi_{\mathrm{OE}}
=
\mathbf 1\left[
\sum V_{\mathrm{geom}}>0
\ \land\ 
\max_{V_{\mathrm{geom}}=1}D_{\mathrm{u8}}=0
\right].
$$

等价的实现语义是：几何有效像素数大于零，并且有效区内的 raw Depth 非零像素数为零。`V_geom` 全空时不触发并走 exact bypass；不能把全 padding 样本当作 observable-empty。

### 3.1 Trigger 的语义边界

当 `\phi_OE=1` 时，唯一允许的结论是：

> 当前可观测 crop 的几何有效区域中没有非零 raw Depth。

它**不**允许写成：

> 当前 corruption cause 是 `entire_missing`。

以下三类输入有意共享同一个 R-OE-lite trigger：

1. synthetic `entire_missing`；
2. natural-empty crop，即几何有效区域天然没有有效 Depth；
3. spatial dropout 恰好删除全部剩余有效 Depth 的输入。

因此有：

$$
D_{\mathrm{observed}}^{\mathrm{entire\_missing}}
=
D_{\mathrm{observed}}^{\mathrm{natural\text{-}empty}}
=
D_{\mathrm{observed}}^{\mathrm{dropout\text{-}emptied}}
$$

时，R-OE-lite 必须使用同一 observable-empty 路由，而不是尝试恢复原因标签。

## 4. raw、normalized 与 padding 的实现位置

### 4.1 raw Depth

当前 v3 输入合同在 `utils/dataloader/mmfr_training_v3.py` 的 `build_mmfr_training_batch_v3` 中完成：

1. 输入 `depth` 是已经归一化的浮点 tensor，形状为 `[B,C,H,W]`，其中 `C` 为 1 或 3；三通道时必须完全相同。
2. 几何有效区由 RGB/Depth padding 共同计算为 `valid`，R-OE-lite 将其作为 `V_geom`。
3. Depth corruption 在 raw uint8-like 平面上执行，得到 `corrupted_depth_uint8`。
4. `raw_depth` 由 `corrupted_depth_uint8 / 255` 导出，形状固定为 `[B,1,H,W]`，范围为 `[0,1]`。

因此 detector 应在 batch builder 已经导出 `raw_depth` 和 `valid_mask` 后、segmentation Depth 路由进入模型前读取这两个字段；不应从 normalized Depth 反推 raw zero。

### 4.2 normalized Depth

同一函数先将 corruption 后的 raw uint8-like 平面通过 `uint8_to_normalized` 归一化，再形成 batch 输出的 `depth`。现有 Depth 合同为 mean `0.48`、std `0.28`：

$$
D_{\mathrm{norm}}
=
\frac{D_{\mathrm{u8}}/255-0.48}{0.28}.
$$

真实图像区域内 raw zero 对应约为 `-1.7142857`；这与 padding 的 normalized exact zero 不同。因此 detector 禁止使用 normalized zero 作为 missing sentinel。

在 `models/encoders/DFormerv2.py` 的 `dformerv2.forward` 中，现有模型只取 Depth 的 channel 0；随后 `GeoPriorGen.forward` 会把该 normalized Depth resize 到各 stage 并生成 geometry prior。R-OE-lite 的 substitute normalized 输出必须在进入该现有路径前完成。

### 4.3 Padding

`build_mmfr_training_batch_v3` 中的 padding 区由 `valid` 排除：

- `raw_depth` 在 `~V_geom` 上为 exact `0.0`；
- normalized `depth` 保留 preprocessing 产生的 exact zero padding；
- `_validate_batch_outputs_v3` 和 `_assert_tensor_is_zero_at_pad` 对 RGB、Depth、raw 信号执行 exact-zero 检查；
- detector 只在 `V_geom=1` 上检查 raw Depth，因此不会把 padding 当作 observable-empty 的有效区域。

## 5. Exact bypass

Exact bypass 是按样本定义的路由合同：

- `\phi_OE=0`：segmentation Depth 输入必须是原 corrupted normalized `depth`，值保持 bitwise identical；不调用 substitute，不做 clamp，不做 RGB-to-Depth normalization，不做 mask blending。
- `\phi_OE=1`：segmentation Depth 输入才替换为第 6 节定义的 substitute normalized output。
- `V_geom` 全空属于 `\phi_OE=0`，仍然 exact bypass。
- 混合 batch 必须逐样本路由；未触发样本不能因为同 batch 中有触发样本而经过 substitute。

这里的 exact bypass 是行为要求，不是“先对全部样本运行 substitute，再用数值相同的 `where` 伪装 bypass”。未触发样本的 substitute forward、clamp 和重新归一化均不发生。

## 6. Substitute 输入输出合同

### 6.1 输入

Substitute 只接收：

- 当前 RGB normalized tensor `rgb`，形状 `[B,3,H,W]`，使用现有 RGB preprocessing 合同；
- `V_geom` 仅用于最终输出的 padding mask，不作为 detector 输入，也不进入 learned convolution。

Substitute 不接收当前 Depth、`depth_valid_*`、reliability target、corruption metadata 或 oracle。RGB padding 已由现有合同保持 exact zero；即使不依赖该性质，最终 output mask 仍必须把 padding 清为 exact zero。

### 6.2 输出

Substitute 产生单通道浮点的 uint8-like output：

- 形状 `[B,1,H,W]`；
- 有效区范围 `[0,255]`；
- 训练时保持 floating point，以便 segmentation gradient 通过 straight-through clamp；不能在训练 forward 中实际 cast 成不可微 `uint8`；
- `~V_geom` 上为 exact zero；
- 触发样本将该单通道复制为三个相同通道，再使用现有 Depth mean/std 做 normalization；
- 该 normalized 三通道结果只送入 segmentation backbone 的 Depth 路径。

令 substitute 的最后一个卷积输出为 `z`，straight-through clamp 固定为：

$$
\operatorname{STClamp}_{[0,255]}(z)
=
 z+\left(\operatorname{clip}(z,0,255)-z\right).\operatorname{detach}().
$$

最终单通道 raw substitute 为：

$$
D_{\mathrm{sub,u8}}
=
V_{\mathrm{geom}}\odot\operatorname{STClamp}_{[0,255]}(z).
$$

乘以布尔几何 mask 后，padding 为 exact zero。触发样本的 segmentation input 为：

$$
D_{\mathrm{sub,norm}}^{(3)}
=
\operatorname{Normalize}_{\mathrm{Depth}}
\left(
\operatorname{repeat}_3(D_{\mathrm{sub,u8}})
\right).
$$

其中 `Normalize_Depth` 必须复用现有 Depth normalization；不能另设 substitute 专用 mean、std 或 missing sentinel。

## 7. 唯一低自由度 architecture

本设计冻结一个且仅一个 substitute architecture。它不包含 attention、normalization、skip connection、dropout、stochastic depth、condition branch、reliability branch 或 oracle branch。`AvgPool2d`、bilinear resize 和 GELU 不含可训练参数。

空间路径固定为：

1. `AvgPool2d(kernel=2,stride=2,ceil_mode=True)`；`Conv3x3(3,122)`；GELU；
2. 再次 `AvgPool2d(kernel=2,stride=2,ceil_mode=True)`；`Conv3x3(122,398)`；GELU；
3. `Conv3x3(398,256)`；GELU；
4. `Conv3x3(256,256)`；GELU；
5. bilinear resize 到上一层空间尺度；`Conv3x3(256,398)`；GELU；
6. bilinear resize 到输入空间尺度；`Conv3x3(398,122)`；GELU；
7. `Conv1x1(122,1)`，再按输入的精确 `(H,W)` 做必要的无参数 resize。

不加入 RGB encoder 的可选冻结分支；本冻结采用上述直接 RGB CNN，以避免“冻结 encoder / 可训练 encoder”成为第二个自由度。通道数也不得替换为其他候选。

### 7.1 精确参数量

每个卷积均带 bias，参数公式为：

$$
P_{k\times k}(C_{in},C_{out})
=
k^2C_{in}C_{out}+C_{out}.
$$

| 模块 | 结构 | 参数量 |
|---|---|---:|
| E1 | `Conv3x3(3,122)` | 3,416 |
| E2 | `Conv3x3(122,398)` | 437,402 |
| B1 | `Conv3x3(398,256)` | 917,248 |
| B2 | `Conv3x3(256,256)` | 590,080 |
| D1 | `Conv3x3(256,398)` | 917,390 |
| D2 | `Conv3x3(398,122)` | 437,126 |
| Head | `Conv1x1(122,1)` | 123 |
| **合计** |  | **3,302,785** |

这里的 **3,302,785** 是 R-OE-lite substitute 的全部 trainable parameters：卷积 weight 共 `3,301,232`，卷积 bias 共 `1,553`。没有未列出的 affine、embedding、校准向量或额外参数。

## 8. 初始化

初始化也固定为单一方案：

- 所有卷积 weight：`trunc_normal_(std=0.02)`；
- E1–D2 的 bias：exact `0`；
- Head bias：`127.5`，使初始 uint8-like 输出落在 clamp 内部，避免整个 substitute 从第一步开始处于下界；该 bias 已计入上表的 123 个参数；
- 无 normalization，因此不存在额外 norm weight/bias 初始化；
- GELU、pooling、resize 无状态；
- 不加载外部 RGB-to-Depth checkpoint，不恢复旧 optimizer state，不使用 source checkpoint 中不存在的 substitute state。

Head bias 的数值只影响初始输出，不改变 detector 语义、输出范围、padding mask 或 loss 合同。

## 9. Optimizer groups

R-OE-lite 沿用 E1 Batch 1 的共同 optimizer identity，并将 substitute 参数加入新模块组：

1. **base decay**：既有 base decay 参数，加上当前实际缺失的 29 个 `backbone.layers.*.blocks.*.Geo.weight`；LR `1e-5`，weight decay `0.01`。
2. **base no-decay**：既有 bias/norm 参数，包括当前已覆盖的 14 个 SyncBN 参数；LR `1e-5`，weight decay `0`。
3. **new decay**：R-OE-lite 七个卷积的全部 weight，共 `3,301,232` 个参数；LR `3e-5`，weight decay `0.01`。
4. **new no-decay**：R-OE-lite 七个卷积的全部 bias，共 `1,553` 个参数；LR `3e-5`，weight decay `0`。

所有 `requires_grad=True` 参数的 optimizer membership 必须恰好为 1。不得把 substitute bias 放入 decay，不得把 29 个 Geo weight 留在组外，也不得只修复 R 组而不修复 C0/F 的共同 base coverage。R-OE-lite 从 epoch-420 完整模型权重做 weights-only restart，不迁移旧 optimizer、scheduler、GradScaler 或 RNG state。

## 10. Loss 合同

R-OE-lite 只使用共同 `L_base^A2`：

$$
L_{\mathrm{base}}^{A2}
=
L_{\mathrm{seg}}
+0.1L_{\mathrm{reliability,depth}}.
$$

具体路由固定为：

- `L_seg` 使用 `D_seg`：触发样本使用 substitute normalized Depth，未触发样本使用 exact-bypass 的原 corrupted normalized Depth；
- `L_seg` 继续使用 `safe_masked_mean(cross_entropy)`；
- A2 reliability auxiliary 仍使用原 corrupted raw Depth 与原 `reliability_target`，不能把 substitute output 回写到 reliability 输入或 target；
- reliability supervision 仍为 Depth-only BCE，权重 `0.1`；
- 不增加 reconstruction loss、RGB-to-Depth loss、substitute reliability loss、condition loss、severity loss、oracle loss 或 consistency loss；
- reliability output 不进入 segmentation backbone、decoder 或 `GeoPriorGen`。

因此 substitute 只通过 segmentation task gradient 学习，不被赋予一个未经批准的“重建真实 Depth”目标。

## 11. 推理信息与路由输出

推理时不读取任何生成器或实验标签。每个样本只按当前 `raw_depth` 和 `V_geom` 计算 `\phi_OE`：

- `observable-empty-substitute`：`\phi_OE=1`，运行 substitute，mask padding，复制三通道并走现有 Depth normalization；
- `nonempty-exact-bypass`：`\phi_OE=0` 且有效区存在 raw nonzero，原 Depth exact bypass；
- `no-geometry-exact-bypass`：`V_geom` 全空，原 Depth exact bypass。

建议在 inference telemetry 中保留以下可复核字段：

- `oe_semantics = "observable-empty"`；
- `oe_trigger`；
- `oe_route`；
- `oe_geometry_pixels = sum(V_geom)`；
- `oe_raw_depth_nonzero_pixels`；
- `oe_raw_depth_zero_pixels`；
- `oe_raw_depth_max_u8`；
- `oe_raw_depth_zero_fraction`，仅在 geometry pixels 大于零时定义；
- `oe_substitute_forward`。

Telemetry 只能由当前 raw Depth 和 `V_geom` 推导，不得写入或推断 cause、severity、`depth_valid_pre/post`、reliability target、generator spec 或 oracle。Telemetry 不参与路由梯度和 loss。

## 12. Quick-Val 解释

Quick-Val evaluator 继续遵守 E1 Batch 1 的固定合同：完整 318 条 `val-dev`、original-full、scale 1.0、no flip、FP32、TF32 off、原始 Label grid、固定 final checkpoint。条件仍记录：

1. clean；
2. synthetic `entire_missing@1.0`；
3. `spatial_dropout@0.75`；
4. `misalignment@0.75`。

R-OE-lite 的每个条件必须同时报告：

- segmentation mIoU 及相对 C0 的 delta；
- `oe_trigger` 样本数与比例；
- geometry pixels、raw nonzero pixels 的汇总；
- 是否存在 no-geometry bypass；
- fixed final checkpoint 与完整 sample coverage。

### 12.1 如何解释四个条件

- `entire_missing@1.0` 是一个压力条件，不是 R-OE-lite 的原因标签。通常它会产生较高 trigger coverage，但结果只能称为 observable-empty stress 下的表现。
- `clean` 中若存在 natural-empty crop，也会合法触发；因此 clean delta 不能被解释为“没有 corruption 时 substitute 从未运行”。
- `spatial_dropout@0.75` 只有在删空当前有效区时才触发；未删空的 dropout 样本必须 exact bypass。该条件同时检验 partial dropout bypass 与 dropout-emptied substitute。
- `misalignment@0.75` 的 trigger coverage 必须实测并报告，不能根据 condition 名称猜测；其非触发样本仍是 exact bypass。

因此 Quick-Val 的 R-OE-lite delta 是“固定条件下 observable-empty 路由的 task effect”，不是 `entire_missing` cause 的纯因果效果。不得把触发比例写成 synthetic entire-missing 比例。

### 12.2 Quick-Val 与 Main-Val 的裁决边界

- Quick-Val 只用于 screening，不产生最终 `promote/stop` 判定；它不能替代 Batch 1B 的十条件 Main-Val。
- Batch 1B 的冻结 Main-Val gate 见 `e1_batch1_protocol.md` §13.5：比较 matched C0 与 R-OE-lite，按 `entire_missing@1.0`、$M_6$、clean 与六个单故障的指定阈值作 `promote`、`stop` 或 `inconclusive` 判定。
- `entire_missing@1.0` 是压力条件，不是原因标签。R-OE-lite 的 trigger coverage 只能说明 observable-empty 路由覆盖，不能作为 synthetic entire-missing 的比例或原因识别准确率。
- 该设计说明不授权 Quick-Val、Main-Val、正式训练或云端运行；每个运行阶段仍须取得相应单独授权。
## 13. 与 AI023 / GeomPrompt 的来源关系

R-OE-lite 只能描述为：

> **GeomPrompt-inspired task-driven observable-empty substitute**。

来源关系的边界固定如下：

- AI023/GeomPrompt 只提供概念层面的启发：利用 RGB/geometry context 产生面向任务的 fallback，而不是进行严格真实 Depth reconstruction。
- 本设计不是 AI023 的 faithful reproduction，不声称复现其 architecture、训练数据、loss、checkpoint、指标或实现细节。
- R-OE-lite 的 detector、observable-empty 语义、exact bypass、现有 Depth normalization、A2 reliability 分流和 3,302,785 参数结构都是本 DFormer/MMFR 合同下的独立冻结设计。
- 不得把“GeomPrompt-inspired”写成“来自 AI023 的官方实现”，也不得把 synthetic `entire_missing` 结果包装为原因识别结果。

## 14. 实现与验证边界

原始设计冻结阶段（2026-09-21）的 Gate-B 准入清单如下。该段保留其设计时范围；当前实现及已执行的资格检查见第 17 节。

1. detector 只读取当前 `raw_depth` 与 `V_geom`；
2. trigger 覆盖三类 observable-empty 输入，且不读取 privileged information；
3. 非触发样本 Depth exact bypass；
4. substitute 参数总数精确为 `3,302,785`；
5. output 范围、straight-through clamp、padding exact zero 和三通道复制符合合同；
6. substitute 只进入 segmentation，A2 reliability 仍使用原 corrupted raw Depth/target；
7. optimizer membership 恰好为 1，29 个 Geo weight 和 14 个 SyncBN 参数归组正确；
8. logits、loss、gradient、parameter finite。

原始设计冻结时未运行测试、模型 forward、GPU、训练、Quick-Val、Main-Val、云任务或 official test，也未创建临时测试文件。2026-09-23 的最小 Gate-B 执行范围、结果与授权边界见第 17 节。

## 15. 已复核的冻结边界

主代理已完成以下定案：

- 本文件建立 `R-OE-lite` 的设计身份，原 `R-EM-lite` 保持 `retired-by-observability`；
- substitute architecture 固定为无 skip、无 normalization、通道 `122/398/256` 的直接 RGB CNN，trainable parameters 精确为 `3,302,785`；
- 2026-09-23 已按第 14 节检查项完成最小 Gate-B，canonical 结果见第 17 节；该资格不等于正式训练或效果评价授权；
- R-OE-lite 不继承 R-EM 的 cause-specific numeric gate；Batch 1B Main-Val 数值门槛已在 `e1_batch1_protocol.md` §13.5 单独冻结，Quick-Val 仅用于 screening；
- 第 17 节记录当前实现身份、Gate-B evidence 与成本边界，不改变本设计冻结架构。

因此当前状态为 `design-frozen, implementation-complete, Gate-B-PASS, formal-training-not-authorized`。不得将 Gate-B 写成训练授权或效果结论；未经单独授权不得启动正式训练、Quick-Val、Main-Val、云任务或 official test。

## 16. 读取依据

本设计只基于以下获准材料：

- `e1_batch1_protocol.md`；
- `e1_screening_plan.md`；
- `../02_evidence/audit_depth_input_contract.md`；
- `models/encoders/DFormerv2.py`；
- `utils/dataloader/mmfr_training_v3.py`。

## 17. 当前实现身份与 Gate-B 记录（2026-09-23）

- config：`local_configs/MUSeg/DFormerv2_S_MMFR_E1_Batch1B_R_OE.py`；substitute class：`ObservableEmptyGeometrySubstitute` in `models/roe_substitute.py`；model routing：`models/builder.py` `_route_roe_modal_x`；training integration：`utils/train.py`；Gate-B runner：`tools/mmfr/e1_batch1b_gateb.py`。
- R-OE-lite trainable parameters：`3,302,785`，由 7 个 Conv weight 与 7 个 bias 构成；weight/bias 分别归入 `new_decay`/`new_no_decay`。
- Canonical Gate-B JSON：`outputs/mmfr-e1-batch1b-gateb/e1-batch1b-gateb.json`；SHA-256 `35297b490c3e3eb54b5e66d3f06784688fca65038e7d09874c30c60cde820231`；`status=PASS`、`failed_checks=[]`、`official_test_included=false`、`formal_training_started=false`。
- C0 common-config comparison、post-build CPU/CUDA RNG 与 1280 项第一 epoch permutation 均 exact equal；已有 Batch 1A C0 final checkpoint SHA-256 为 `ca618b23d18eabb201a0d11d18da383ac99576d0feae5864e3233bda527d9a1a`，Gate-B 确认其可复用。
- Gate-B 单步 AMP loss `0.7119939327` finite，optimizer step 已应用；R-OE 新参数梯度 finite/nonzero。该资格仅覆盖最小更新路径，不代表完整训练稳定性或分割收益。
- batch-size-1 成本记录：non-trigger/trigger latency median `130.693645/263.437180 ms`；allocated memory 增量 `1,517.046875 MiB`，reserved memory 增量 `3,608 MiB`。这不是 batch-size-10 训练可行性结论。
- 复核报告：`../02_evidence/report_e1_batch1b_roe_gateb.md`；实现差异摘要：`../02_evidence/audit_implementation_diff_e1_batch1b_roe.md`。
- 停止点：`ready-for-R-OE-formal-training-authorization`。正式训练、Quick-Val、Main-Val、云任务、Batch 2、T 与 official test 均未授权；Batch 1A Main-Val 的上级处置独立保留。
