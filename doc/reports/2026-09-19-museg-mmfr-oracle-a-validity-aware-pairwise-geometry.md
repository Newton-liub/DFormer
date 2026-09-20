# MMFR-Oracle-A：Validity-Aware Pairwise Geometry 的 go/no-go 验证（阶段报告）

> **身份：** `MMFR-Oracle-A`（Validity-Aware Pairwise Geometry），inference-only
> **checkpoint：** `experiments/MMFR_A2_v3/checkpoints/selector-epoch-420.pth`，SHA-256 `2b72eb7f28e5c4b4f52bdc43e2ac5c7298169325a412bfbcbc7a022c29822597`
> **数据：** `val-dev` 318 条，split SHA-256 `1d0719d8f64f016d48995c25ab66d4004d76b7155d9efeef7cbb7454c0dd0e83`
> **evaluator：** 冻结的 `msflip-whole-original-grid-v1`（同一 `tools/evaluate_museg_checkpoint[_fast]`，同一 corruption 生成器，同一 RNG 策略）
> **official test：** `sealed_unread`，本报告全部产物 `official_test_included=false`
> **状态：** 已完成（3 condition × 4 variant × 318 样本），裁决 **NO-GO**

## 一句话结论（大白话）

**在这份 checkpoint 上，即使我们「开天眼」百分之百知道哪些 Depth 像素是无效的，按有效性去关掉对应的 Depth 几何关系，也只会让结果变差 —— 在 `clean` 上掉 3.48 分，在最重要的 `spatial_dropout@0.75` 上掉 1.77 分，唯一「变好」的 `entire_missing@1.0` 只涨 0.06 分（本质是无变化）。**

更关键的是：**「只关掉涉及无效像素的那些 Depth 几何关系」（strict / aggregated）与「干脆把 Depth 几何整体关掉」（geometry_off）在 `spatial_dropout` 上几乎完全一样（差 0.04 分）。** 也就是说，我们没有证据表明「局部、选择性地抑制错误几何」比「一刀切不使用 Depth 几何」更好 —— 这条路线在动作层面没有体现出选择性价值。

因此按预先写好的门槛，本阶段判定 **NO-GO**：不继续投入「validity-aware geometry suppression」这一族，也不应转而训练一个更复杂的 reliability gate 去「抢救」它。

### 本报告 variant 名称与需求中 A/B/C/D 的对应

| 需求中的名字 | 本报告 / 代码中的名字 | 含义 |
|---|---|---|
| Variant A：Original / No-action | `original` | 完全不传 oracle 参数，走 checkpoint 自己的历史路径 |
| Variant B：Strict Validity Pair Oracle | `strict` | $\rho_{ij}=r_ir_j$，$r_i\in\{0,1\}$，只门控 `weight[1]*mask_d` |
| Variant C：Scale-Aware / Aggregated Validity Oracle | `aggregated` | $\rho_{ij}=c_ic_j$，$c_i\in[0,1]$ |
| Variant D：Depth-Geometry-Off | `geometry_off` | 完全移除 `weight[1]*mask_d`，只留位置先验 |
| （额外，用于 identity 验证） | `noop_continuous` / `noop_strict` | 用「全有效」掩码驱动同一条 Oracle 代码通路，必须逐位等于 `original` |

### 正式结果表（318 样本，`selector-epoch-420`）

| Variant | Clean | Spatial Dropout@0.75 | Entire Missing@1.0 |
|---|---:|---:|---:|
| Original A2 | 57.06 | 54.42 | 52.55 |
| Strict Validity Pair Oracle | 53.58 | 52.65 | 52.61 |
| Aggregated Validity Oracle | 53.80 | 52.72 | 52.61 |
| Depth Geometry Off | 52.61 | 52.61 | 52.61 |

### 差值表（Oracle − Original，单位为 mIoU 百分点）

| Variant | ΔClean | ΔSpatialDropout | ΔEntireMissing |
|---|---:|---:|---:|
| Strict Validity Pair Oracle | **−3.48** | **−1.77** | +0.06 |
| Aggregated Validity Oracle | **−3.26** | **−1.70** | +0.06 |
| Depth Geometry Off | **−4.45** | **−1.81** | +0.06 |

**参考基线可比性：** `original` 在 `clean` 与 `spatial_dropout@0.75` 上的 confusion matrix 与 318/318 逐样本 mIoU 与冻结 Main-Val 证据**完全一致**（`57.06` / `54.42`）；`entire_missing@1.0` 亦为 `52.55`，与冻结值一致。因此上表所有差值都是与正式 Main-Val 同一输入的配对比较，不需要任何换算。

---

## 0. 这份报告要回答什么

本阶段只回答一个问题：

> 即使我们已经**完美知道** Depth 的 validity，用这个信息去修改 DFormerv2 的 geometry computation，是否真的能在 `spatial_dropout` 下改善分割？

如果答案是否定的，就没有理由继续投入一个更复杂的 reliability-conditioned geometry module。

本阶段**不训练任何参数、不新增任何网络模块**，只做 inference-only 的 Oracle 对照。

---

## 1. 本地 DFormerv2 geometry path 到底是什么（代码事实）

### 1.1 调用链与代码位置

```text
EncoderDecoder.forward(rgb, modal_x, ..., geometry_oracle=None)        models/builder.py
└─ EncoderDecoder.encode_decode                                     models/builder.py
   └─ dformerv2.forward(x, x_e, oracle_corruption_mask=None,
                        geometry_oracle=None)                        models/encoders/DFormerv2.py
      ├─ PatchEmbed(x)                                                -> [B, H/4, W/4, 64]
      └─ BasicLayer.forward(x, x_e, ..., geometry_oracle=...)         models/encoders/DFormerv2.py
         └─ RGBD_Block.forward(x, x_e, ..., geometry_oracle=...)      models/encoders/DFormerv2.py
            ├─ cnn_pos_encode(x)                                       DWConv2d(3,1,1)
            ├─ GeoPriorGen.forward((H,W), x_e, split_or_not, ...)      models/encoders/DFormerv2.py  ← 唯一使用 Depth 的地方
            └─ Attention(layer_norm1(x), geo_prior, split_or_not)      Decomposed_GSA / Full_GSA
```

关键事实：**Depth 在整个 DFormerv2-S 里只进入几何先验**。`x_e`（Depth 特征）除了 `GeoPriorGen` 以外没有任何其它消费者，backbone 主干、decoder、分类头都不看 Depth。

### 1.2 Depth 输入预处理（代码事实）

`tools/evaluate_museg_10condition.py` → `tools/evaluate_museg_checkpoint.build_msflip_views` / `10condition.build_depth_view`：

1. 读原图 Depth（`cv2.imread(..., IMREAD_GRAYSCALE)`，uint8，与 RGB/Label 同为 `932×1082` 这类原始网格）；
2. `cv2.resize(depth, (scaled_w, scaled_h), INTER_LINEAR)`，缩放尺度 `0.5/0.75/1.0/1.25/1.5`；
3. 归一化：`depth_float = (depth/255 − 0.48) / 0.28`；
4. 复制成 3 通道；
5. 水平翻转（每个尺度有 original / flip 两个 view）；
6. 右下补零 padding 到 32 的整数倍（`constant_values=0`）。

因此：

| 输入状态 | 原始 uint8 | 归一化后的值 |
|---|---:|---:|
| 有效 Depth | 1…255 | 约 `−1.7143` 到 `1.8714` |
| 无效 Depth（原始 0） | 0 | **`−1.7143`**（= `(0/255 − 0.48)/0.28`） |
| padding 区 | 补的是归一化后的 0 | **`0.0`**（等价于原始 Depth ≈ `122.4`） |

**这里有两个必须记住的点：**

* 无效 Depth 在模型眼里**不是 0**，而是 `−1.7143`，这是一个比任何真实有效 Depth 都更极端、且**所有无效像素完全相同**的数值；
* padding 区在模型眼里是 `0.0`，也就是一个「中等深度」的伪测量值。padding 不是「没有 Depth」，而是「有一个假的 Depth」。

一个 val-dev 样本的实测（`outputs/mmfr-oracle-a-geometry-audit/geometry-audit.json`，样本 `03-01-01-0066-240526121121-08-99`，scale 1.0）：

* 原始网格有效率 `94.49%`（自然无效 `5.51%`）；
* 在 stage `/4` 网格上，**有效** token 的归一化 Depth 均值 `−1.2259`，**无效** token 的均值 `−1.6451`。

也就是说，有效/无效在 Depth 数值上只差约 `0.42`，但无效那一侧紧贴 `−1.7143`（原始 0 的编码值）。

### 1.2b 原始 Depth 的 sentinel 结构（全量 `val-dev` 扫描，代码事实）

对全部 318 条 `val-dev` 逐张扫描原始 `Depth/*.png`：

| 事实 | 数值 |
|---|---|
| 几何 | 全部 `932×1082`（与 RGB/Label 完全一致） |
| 无效 sentinel | 只有 `0`；**没有任何一张图出现 `255`**；也不存在 NaN（uint8 PNG） |
| 非零取值 | 246 个不同值，最小 `2`，最大 `249` |
| 无效像素比例 | mean `30.15%`、median `26.88%`、min `0.16%`、max `93.25%` |
| 完全无无效像素的样本 | **`0` 张** |
| 无效比例 > 50% 的样本 | `64` 张 |

**这意味着 `clean` 条件根本不是「Depth 全有效」**：平均来说每张 `val-dev` 图像有约 30% 的像素本来就是缺失的。因此 Strict Validity Oracle 在 `clean` 上**必然**会改动一部分几何关系，它不等于原版是正常的、可预期的，而不是 bug。

### 1.3 geometry prior 的构造（公式 + 真实 tensor shape）

`GeoPriorGen` 里 `self.weight` 是一个 `nn.Parameter(torch.ones(2,1,1,1))`：`weight[0]` 管位置项，`weight[1]` 管 Depth 项。`decay` 是 buffer：

$$
\mathrm{decay}[n] = \log\!\left(1 - 2^{-\,\mathrm{init} - \mathrm{range}\cdot n / N}\right) < 0
$$

（Stage 1-3：`N=4`、`init=2`、`range=4`；Stage 4：`N=16`、`init=2`、`range=6`。实测 `|decay| ∈ [0.005, 0.288]`，**恒为负**。）

**Stage 1–3（i_layer ≠ 3）= Decomposed_GSA，H/W 分解**，其中 `l = H+W`：

```python
mask_d_w[b,n,i,p,q] = |d[b,0,i,p] − d[b,0,i,q]| * decay[n]        # [B,N,H,W,W]
mask_d_h[b,n,j,p,q] = |d[b,0,p,j] − d[b,0,q,j]| * decay[n]        # [B,N,W,H,H]
mask_w = weight[0] * pos_w.unsqueeze(0).unsqueeze(2) + weight[1] * mask_d_w
mask_h = weight[0] * pos_h.unsqueeze(0).unsqueeze(2) + weight[1] * mask_d_h
```

**Stage 4（i_layer == 3）= Full_GSA**：

```python
mask[b,n,p,q] = weight[0] * (|Δrow| + |Δcol|) * decay[n]
              + weight[1] * |d[b,0,p] − d[b,0,q]| * decay[n]          # [B,N,L,L]
```

其中 `d` 是 `F.interpolate(depth_patches, size=(H,W), mode="bilinear", align_corners=False)` 的结果，即 **Depth 在 stage 网格上的取值本身就是一次双线性重采样**。

以 `932×1082` 样本、scale 0.5、padded `480×544` 为例的真实 shape：

| Stage | 网格 | Attention | `mask_d` / `mask` shape | `qk_mat` shape |
|---|---|---|---|---|
| layer 0 | `120×136` | Decomposed_GSA | `mask_w [B,4,120,136,136]`，`mask_h [B,4,136,120,120]` | `qk_mat_w [B,120,4,136,136]` |
| layer 1 | `60×68` | Decomposed_GSA | `mask_w [B,4,60,68,68]`，`mask_h [B,4,68,60,60]` | 类似 |
| layer 2 | `30×34` | Decomposed_GSA | 同上模式 | 同上模式 |
| layer 3 | `15×17` | Full_GSA | `mask [B,16,255,255]` | `qk_mat [B,16,255,255]` |

### 1.4 geometry 到 attention 的真实作用位置（代码事实，重要）

本地实现属于第一种形式——**把 geometry prior 当作加到 attention logits 上的偏置（bias），softmax 之前相加**：

```python
qk_mat_w = qr_w @ kr_w.transpose(-1,-2)
qk_mat_w = qk_mat_w + mask_w.transpose(1, 2)     # ← 加法 bias
qk_mat_w = torch.softmax(qk_mat_w, -1)
```

**不是** `softmax(QK) × geometry mask`，也**不是**把 pair 置成 `−∞`。

因为 `decay < 0` 且 `|Δd| ≥ 0`、`|Δpos| ≥ 0`，所以几何先验是一个 **≤ 0 的惩罚项**：位置越远、Depth 差越大，被减得越多，注意力权重越小。它从来不是「允许/禁止」，而是「按几何距离衰减」。

### 1.5 实测：学习到的权重、Depth 项的真实量级、对 attention 的真实影响

权重（checkpoint 里 29 个 `Geo.weight`，即 3+4+18+4 个 block，每个 2 个数）：

| layer | `weight[0]`（位置）min/mean/max | `weight[1]`（Depth）min/mean/max |
|---|---|---|
| 0 | `0.852 / 1.256 / 1.727` | `0.427 / 3.859 / 10.053` |
| 1 | `0.673 / 1.381 / 2.029` | `0.177 / 0.806 / 1.254` |
| 2 | `−0.042 / 0.388 / 1.377` | `−0.195 / 1.481 / 4.102` |
| 3（Full） | `0.095 / 0.334 / 0.740` | `−0.062 / **−0.009** / 0.059` |

**Stage 4（Full GSA）的 Depth 权重几乎为 0**，也就是说最后一级的 depth geometry 基本不参与。

对全部 29 个 block 的 attention 影响（同一真实样本、同一 view，`outputs/mmfr-oracle-a-geometry-audit/`）：

| 指标 | W 方向 | H 方向 | Full |
|---|---:|---:|---:|
| 原版注意力落在「无效 key」上的质量占比（均值） | `0.1230` | `0.0581` | `0.1163` |
| 均匀分布下应有的占比（均值） | `0.0758` | `0.0674` | `0.0725` |
| 施加 **Strict Oracle** 后占比（均值） | `0.1231` | `0.0591` | `0.1163` |
| Strict Oracle 引起的注意力总变差（均值 / 最大） | `0.00326` / `0.0119` | `0.00191` / `0.0091` | `0.00282` / `0.0119` |
| **Depth-Geometry-Off** 引起的注意力总变差（均值 / 最大） | `0.00367` / `0.0130` | `0.00371` / `0.0127` | `0.00318` / `0.0130` |

（总变差 = `0.5 * Σ|p−q|`，1.0 表示注意力分布完全改变。）

**结论（代码事实 + 实测）：** 在这个 checkpoint 上，无论「严格取消无效参与的 Depth 几何项」还是「完全关掉 Depth 几何项」，对任意单个 block 的注意力分布的影响都在 **0.2%–1.3%** 量级。

---

## 2. invalid Depth 原来是如何进入 geometry computation 的？是否真的产生了「数学上成立但物理无意义」的几何证据？

**它确实进入了，而且是这样进入的（代码事实）：**

无效 Depth 的归一化值是 `−1.7143`，一个所有无效像素共享的常数。它被双线性重采样到 stage 网格后与邻近有效值混合，因此 stage 上的「无效 token」只是**接近**这个常数，不是精确等于。

于是 Depth 几何项 `|d_i − d_j| · decay` 对三类 pair 给出完全不同的、但都不是真实测量关系的数值（实测，`|Δd|` 与原始 depth 项）：

| pair 类型 | `mean |Δd|` | 原始 depth 项（`|Δd| × mean|decay|`） |
|---|---:|---:|
| valid–valid（stage /4） | `0.1456` | `0.0188` |
| valid–invalid | `0.4228` | `0.0547` |
| invalid–invalid | `0.1078` | `0.0140` |

**这产生了两条「数学上成立、物理上无意义」的几何证据：**

1. **无效区域内部被当作「同一深度平面」**：所有无效像素数值相同 → `|Δd| ≈ 0.108`（比有效–有效 pair 的 `0.146` 还小）→ 无效 token 之间几乎不被 Depth 项惩罚，互相吸引；
2. **无效区域与有效区域被强行拉开**：`|Δd| ≈ 0.423`，约为有效–有效 pair 的 `2.9` 倍 → 包含无效 token 的 pair 受到的 Depth 惩罚约为有效 pair 的 `2.9` 倍。

**但必须诚实说明边界（合理推断，不是已证结论）：**

* 上述差异是**代码事实 + 单样本实测**，它证明「无效 Depth 产生了非零、且与有效性高度相关的几何偏置」；
* 它**不自动等于**「这个偏置正在伤害分割」——1.5 节实测显示这个偏置对注意力的总体影响只有百分之零点几；
* 是否「有害」，只能由第 3–5 节的 318 样本配对结果回答；
* 注意 1.5 节的「无效 key 注意力质量 `0.123` vs 均匀 `0.076`」说明：原版模型在 W 方向上**反而更偏向**无效 token（因为无效 token 之间 `|Δd|≈0` 互相吸引，而 q·k 项本身并不排斥它们），并非简单地「被压制」。

---

## 3. Oracle 实际修改了哪个 tensor

```text
文件：   models/encoders/DFormerv2.py
类：     GeoPriorGen
函数：   GeoPriorGen.forward
tensor： 唯一被改写的对象是「Depth 派生几何项」这一项，即
         self.weight[1] * mask_d        （Full）
         self.weight[1] * mask_d_h / mask_d_w （Stage 1–3 的 H/W 分解）
```

具体做法：

```python
mask = self.weight[0] * mask + self.weight[1] * (pairwise_gate * mask_d)
```

* `pairwise_gate` 为 `[B,1,L,L]`（或 H/W 分解的 `[B,1,W,H,H]` / `[B,1,H,W,W]`），由 `build_oracle_pairwise_gates` 生成；
* **只有 `weight[1]` 的那一项**被乘以 gate；`weight[0] * position` 一项、`sin/cos`、Q/K/V、`lepe`、FFN、Depth 输入 tensor、decoder、logits 后处理**全部不动**；
* `Depth-Geometry-Off` 用「全零 reliability 场」走同一条表达式（而不是删掉该项），因此 tensor shape 与所有其它路径完全一致，且 `x + weight[1]*(0*mask_d)` 与 `x` 逐位相同（每一加数都是带符号零）。

**有效性如何被聚合到 stage（关键设计说明）：**

```text
原始网格 validity（V_state_final, bool, 932×1082）
  → invalidity = 1 − validity                              (float32, {0,1})
  → 每个 view：cv2.resize(INTER_LINEAR)  ← 与 Depth 完全同一个调用、同一插值核
  → 水平翻转（若该 view 是 flip）
  → 右下补 0（补 0 表示「有效 / 不干预」）
  → 送进模型后：F.interpolate(bilinear, align_corners=False) 到 stage 网格
                ← 与 GeoPriorGen 对 Depth 用的**同一个算子**
```

因此 **stage token 的 reliability 与它配对的 Depth 值拥有完全相同的 receptive support**（都由同一次双线性重采样决定），这比「对整个 token 感受野做 area 平均」更严格地对应模型真实使用的采样点。

两种 gate：

* **Strict**：$r_i = 1 \iff$ 该 token 插值后的 invalidity **精确等于 0**。因为双线性权重全部非负，这个测试是**精确**的（无 epsilon），含义是「贡献到该 token 的 Depth 值的每一个原始像素都有效」。
* **Aggregated**：$c_i = 1 - \text{interpolated invalidity}$，即「贡献到该 token 的原始像素中有效部分的加权比例」，然后 $\rho_{ij}=c_i c_j$。

**padding 约定（有意选择，需报告）：** 右/下 padding 一律记为**有效（invaldity = 0）**，也就是 `Oracle-A` 完全不改动 padding 区。这与冻结的 `DVG-B1` 预注册约定一致。后果：`entire_missing@1.0` 下 padding 条带仍保留自己的 Depth 几何；单元测试实测该条带的 Depth 值精确为 `0.0`，因此条带内部 `|Δd| = 0`，Depth 项恒为 0，于是 Strict 与 Depth-Geometry-Off 在该条件下**逐位相同**（见第 4 节）。

---

## 4. Oracle 是否满足 identity test（数值误差）

单元测试：`python -m tools.mmfr.oracle_a_unit_tests --checkpoint ... --device cuda --samples 2`
证据：`outputs/mmfr-oracle-a-unit-tests/oracle-a-unit-tests.json`

**`status = PASS`，`assertions_total = 66`，`assertions_failed = 0`，wall clock `106.064` 秒；checkpoint SHA-256 前 16 位 `2b72eb7f28e5c4b4`。**

### Test 1 — All-valid identity（$\rho_{ij}\equiv 1$）

构造「全有效」validity（invalidity 全 0），分别走 `noop_continuous`（$c_i\equiv 1$）与 `noop_strict`（$r_i\equiv 1$）两条 Oracle 通路，与**完全不传 oracle 参数**的原版路径比较。覆盖 2 个真实样本 × 3 个 condition（`clean` / `spatial_dropout@0.75` / `entire_missing@1.0`），每单位 10 个 view：

| 比较 | `torch.equal(logits)` | `max_abs_diff` | `mean_abs_diff` | prediction mismatch |
|---|---:|---:|---:|---:|
| original vs noop_continuous（6/6 单位） | **True** | `0.0` | `0.0` | `0` |
| original vs noop_strict（6/6 单位） | **True** | `0.0` | `0.0` | `0` |

几何先验张量层面的定点检查（不含模型前向）同样逐位相等：`full_all_valid_mask_bitwise_equal_reference`、`split_all_valid_mask_h/w_bitwise_equal_reference` 全部为 True。

### Test 2 — All-invalid boundary（$r_i\equiv 0$）

在真实样本上把 validity 强制置成全 0（即全部无效），比较 `strict` 与 `geometry_off`：

* 最终 logits **`torch.equal` 为 True**，`max_abs_diff = 0.0`，`prediction_mismatch = 0`；
* 几何先验层面：`geometry_off` 的 mask **逐位等于** 独立重算的 `weight[0] * position`；all-invalid 的 `strict` mask 与之**逐位相同**；
* 位置先验仍在且非零（`|mask|` 之和 > 0，取值全部 ≤ 0）；
* 深度项在参考路径中确实激活（`reference_mask` 与 off mask 不相等）。

**为什么这两个变体在 all-invalid 下逐位相同（可解释的巧合，不是实现漏改）：** padding 区的 Depth 归一化值恒为 `0.0`，因此 stage 上「完全落在 padding 条带内」的 token 其插值 Depth 精确为 `0.0`，它们两两之间 `|Δd| = 0`，Depth 项 `weight[1]·(gate·mask_d)` 恒为 0；其余（真实）token 在 strict 下 gate 为 0。所以 Strict 与 Depth-Geometry-Off 在真实 token 上等价。实测该条带在 stage `/4` 上占 `525/16320 = 3.22%` 的 token，全部满足上述条件。

### Test 3 — Half-valid synthetic mask

在 `6×9` 的合成 stage 网格上使用「左半有效 / 右半无效」的 invalidity 图案：

* `strict_reliability` 严格二值（`(r==0)|(r==1)` 全覆盖），且与独立重算的 `(interp(invalidity)==0)` 逐位相等；
* `continuous_reliability` 与独立重算的 `1−interp(invalidity)` 逐位相等；
* `valid-valid` pair gate **全为 1**；`valid-invalid` 与 `invalid-invalid` pair gate **全为 0**；
* 该合成图案下：`54` 个 token 中 `46` 个被判为无效（`0.852`），`continuous` 均值 `0.25`。

### Test 4 — Natural clean invalid（真实自然缺失）

使用真实 `val-dev` 的 `clean` validity（定义即原始 `Depth == 0`）：

| 检查 | 结果 |
|---|---|
| `invalidity == (raw Depth == 0)` 逐位 | 两样本均为 **True** |
| **支撑集一致性**：Depth 缩放所改动的 view 像素集合 == invalidity 缩放所标记的集合（scale 0.5 / 0.75 / 1.0） | 三个尺度**完全一致**（`1` 像素对 `1` 像素） |
| **与独立 PyTorch 双线性重实现的差异**（同 `align_corners=False` 约定） | 二值判定不一致像素 `0`；`max_abs ≤ 1.11e−4`（仅 OpenCV 定点系数量化） |
| **单点扩散**：只让 1 个原始像素无效，检查 view 上有多少像素被标记 | `support_pixels = 1`（scale 0.5 与 1.0 均如此），即**没有被错误扩散或复活** |
| padding 区在 10 个 view 上是否精确为「有效」 | 全部 **True** |
| 非 padding 区 view 平均无效比例 vs 原始网格无效比例 | 样本一：`0.0551` vs `0.0551`；样本二：`0.0724` vs `0.0724`（一致） |
| stage `/4` 上 strict 无效 token 比例（min–max，10 个 view） | 样本一：`0.0624 – 0.0844`；样本二：`0.0822 – 0.1074` |
| stage `/4` 上 continuous 平均 reliability（min–max） | 样本一：`0.9455 – 0.9474`；样本二：`0.9286 – 0.9303` |

注意最后两行：**strict 在 stage 上会把无效比例放大到原始比例的约 1.1–1.5 倍**，这是「只要贡献到该 token 的任一原始像素无效，就判该 token 无效」这一保守规则的必然结果，属于设计意图，不是缺陷。

### Test 5 — no-action 端到端复现（runner 内）

在 `experiments/MMFR_OracleA/smoke2/`（4 样本 × 3 condition × 6 variant）中，`noop_continuous` / `noop_strict` 与 `original` 的 **confusion matrix 与逐样本 confusion matrix 全部逐位相同**（3 个 condition 全部成立）。

### 补充：与冻结 Main-Val 的数值可比性（发现并修复的两个真实缺陷）

冻结 Main-Val 的 `selector-epoch-420` 证据是本阶段唯一的正式对照。为保证「可以直接比较」，我在正式运行前做了**输入级**核对：把本 runner 的 `original` 变体与冻结证据逐样本对比，并逐个比较 corruption 摘要。

**缺陷 1（RNG 基点）：** 首次启动时，本 runner 的 `original` 在 `clean` 上与冻结证据逐样本 mIoU 不一致（首个样本 `14.29` vs 冻结 `14.36`；318 张里 250 张不同）。
根因：冻结 runner 的 `tools/evaluate_museg_10condition.load_eval_model` 在捕获 base RNG 之前会显式执行 `torch.manual_seed(base_rng_seed)` 与 `torch.cuda.manual_seed_all(base_rng_seed)`（`base_rng_seed = 2026091401`），本 runner 当时没有这一步；而冻结的 Ham decoder 每次前向都用 `torch.rand` 重抽 NMF bases。
修复后：`original/clean` 的 **confusion matrix 与冻结证据逐位相同**，318/318 张逐样本 mIoU 完全相等（`miou 57.06 / macc 69.86 / mf1 71.33`，与冻结值一致）。

**缺陷 2（corruption 种子身份，严重）：** 修复缺陷 1 后 `clean` 已完全一致，但 corruption 摘要仍对不上：

| condition | 本 runner（修复前） | 冻结 Main-Val | 一致 |
|---|---|---|---|
| `clean` | `676019b178cb31525c6ea345ba600165` | `676019b178cb31525c6ea345ba600165` | ✅ |
| `spatial_dropout@0.75` | `d01fa0affd57f98a87fe272b8eb3eda1` | `0897b1e9fc9ded07f67e3d23ed74b7c5` | ❌ |

根因通过冻结的 model-free 摘要基准 `experiments/MMFR_A2_v3/benchmarks/10cond-corruption-digest-process.json`（其中记录了每个 unit 的 `seed_words`）定位到：

```text
冻结 seed words: [2026091401, 1,  96155078, 629898586, 1696300807, 2672834079]
本 runner 当时: [2026091401, 1, 1413192147, 2620420522,  693253843, 2172760572]
```

后 4 个是 sample id 的 SHA-256 前 16 字节。经逐一验证：`sha256("03-01-01-0066-240526121121-08-99")` 给出**本 runner** 的那组，而 `sha256("RGB/03-01-01-0066-240526121121-08-99.jpg")` 给出**冻结**的那组。

即冻结 runner 的调用约定是 `normalize_sample_id(entry)`（**split 条目**，形如 `RGB/<id>.jpg`），而不是 `normalize_sample_id(Path(entry).stem)`（裸 id）。本 runner 当时用了后者。

**这个缺陷只影响需要随机抽样的 condition**：`clean`（不抽样）与 `entire_missing@1.0`（确定性清空）不受影响，`spatial_dropout@0.75` 的 dropout 掩码则来自另一个随机实现。

**处置：** 按用户「发现数据协议不一致必须立即停止并报告、不得静默修复后继续」的要求，我停止了当时正在进行的正式运行，把调用改成 `normalize_sample_id(entry)`，并重新逐样本验证：

* 修复后 `clean` / `spatial_dropout@0.75` / `entire_missing@1.0` 三个 condition 的前 3 张样本 corruption 摘要**全部与冻结值逐位一致**；
* 随后重跑单元测试与 smoke，并从头重跑正式 318 样本评测。修复前的那一次运行产物全部作废。

### 补充：`MMFR-Oracle-A-NoAction` 基线（16 样本，用户要求的 16–32 样本范围）

在正式 318 样本之外，另跑了一次独立的 no-action 基线：`experiments/MMFR_OracleA/noaction16/`，**16 个固定样本 × 3 个 condition × 6 个 variant**（4 个正式 variant + 2 个 no-op identity variant）。目的只有一个：证明「Oracle 功能关闭时，重构后的代码必须恢复原始模型」。

结论：**全部通过。**

| Condition | `original` vs `noop_continuous` | `original` vs `noop_strict` |
|---|---|---|
| clean | confusion matrix 逐位相同 + 16/16 逐样本 confusion matrix 逐位相同 | 同左 |
| spatial_dropout@0.75 | 同左 | 同左 |
| entire_missing@1.0 | 同左 | 同左 |

除此之外还有两项交叉验证：

- **跨运行一致性**：这次 16 样本运行的 `original` 与先前 4 样本 smoke 运行（`smoke3`）在前 4 个样本上 **3 个 condition 全部逐样本一致**；
- **边界一致性**：这 16 样本里 `entire_missing@1.0` 的 `strict` 与 `geometry_off` confusion matrix **逐位相同**，与 318 样本的结论一致。

大白话说：把「完美的有效性信息」用在一个「全有效」的输入上，模型输出和原版**一模一样，连一个像素都不差** —— 这证明我们对 geometry path 的改动在功能关闭时确实没有任何副作用，正式结果里的所有差异都只来自 Oracle 本身。

---

## 5. 修改文件与 diff 摘要

### 5.1 被修改的既有文件

| 文件 | 目的 | +/- 行 |
|---|---|---|
| `models/encoders/DFormerv2.py` | 新增 `geometry_oracle` 可选通路：`normalize_geometry_oracle` / `build_oracle_stage_reliability` / `build_oracle_pairwise_gates`，并在 `GeoPriorGen.forward` 增加「只门控 `weight[1]*(…)`」的分支；`RGBD_Block` / `BasicLayer` / `dformerv2` 透传该参数 | `+137 / −5` |
| `models/builder.py` | `EncoderDecoder.forward` / `encode_decode` 透传 `geometry_oracle`（默认 `None`，旧路径逐字节不变） | `+10 / −2` |

### 5.2 新增文件

| 文件 | 作用 | 行数 |
|---|---|---|
| `utils/dataloader/oracle_a_validity.py` | 纯 NumPy/OpenCV 的 validity → 每 view invalidity 变换（与冻结 view 几何严格对齐） | 125 |
| `tools/mmfr/oracle_a_core.py` | variant 定义、每 view oracle 注入包装器（`GeometryOracleModel`） | 189 |
| `tools/mmfr/oracle_a_unit_tests.py` | Test 1–5 单元测试与证据落盘 | 737 |
| `tools/mmfr/run_oracle_a.py` | inference-only 评测 runner（复用冻结 evaluator / corruption / view 构建） | 471 |
| `tools/mmfr/analyze_oracle_a.py` | 结果表、per-class、per-sample、location-group paired bootstrap、go/no-go 判定 | 274 |
| `tools/mmfr/oracle_a_geometry_audit.py` | 只读 CPU 几何审计（权重、量级、pair 统计、注意力影响） | 401 |

### 5.3 明确没有改动的模块

```text
dataset / split               未改
corruption generator          未改（仍调用 utils.dataloader.multimodal_failure_v3.apply_failures）
RGB preprocessing             未改
Depth preprocessing           未改（同一 cv2.resize / 同一 0.48,0.28 归一化 / 同一 padding）
normalization                 未改
decoder                       未改
loss                          未改（本轮不训练）
checkpoint selector           未改
evaluator                     未改：tools/evaluate_museg_checkpoint.py 与 _fast.py 均未修改
                              （run_oracle_a.py 只以库的方式 import 它们）
checkpoint                    未改（strict=True 加载，本阶段未产生任何新权重）
```

---

## 6. spatial_dropout 是否得到明确恢复？

**没有。它在所有三个 Oracle 变体下都变差。**

| Variant | mIoU | ΔmIoU（corpus） | 逐样本 win / tie / loss | 逐样本平均 Δ | location-group 配对 bootstrap 点估计 | 95% 区间 |
|---|---:|---:|---|---:|---:|---|
| original | **54.42** | — | — | — | — | — |
| strict | 52.65 | **−1.77** | 99 / 44 / 175 | −0.252 | −0.325 | **[−0.503, −0.158]** |
| aggregated | 52.72 | **−1.70** | 104 / 38 / 176 | −0.248 | −0.324 | **[−0.500, −0.161]** |
| geometry_off | 52.61 | **−1.81** | 100 / 44 / 174 | −0.257 | −0.328 | **[−0.508, −0.160]** |

（corpus 列是「把所有样本的 confusion matrix 加起来再算 mIoU」，与冻结 evaluator 口径一致；「逐样本平均 Δ」与 bootstrap 是「先算每张图 mIoU、再对 location group 求均值」的口径，两者权重不同，所以数值不同但结论一致。196 个 location group 全部参与配对。）

**结论：**
- 需要的门槛是 `ΔmIoU ≥ +0.50` 且 bootstrap 区间下界 > 0。实测是 **−1.77**，区间是 **[−0.503, −0.158]**，**上界都远小于 0**，方向完全相反。
- 逐样本胜负是 **99 胜 / 44 平 / 175 负**（strict），即大多数图像是变差的。
- bootstrap 区间在两代 Oracle 上都排除了 0，但排除的方向是「明确更差」，不是「明确更好」。
- 集中度检查（strict / spatial_dropout，318 张图）：逐样本 delta 求和 `−80.05` 分，其中正贡献合计 `+47.82`、负贡献合计 `−127.87`；最大的 5% 样本合计 `+29.87`、单张最大 `+4.67`、单张最小 `−7.33`、中位数 `−0.020`。也就是说收益确实存在但很小，而且**被多数样本的损失盖过**；这里不存在「少数样本带来全部收益」可以争取的情形 —— 净效果本身就是负的。

### 每个 condition 的「无效」比例（Oracle 实际介入强度）

| Condition | 原始网格无效像素比例（均值） | stage `/4` strict 判为无效的 token 比例 | stage `/4` continuous 平均 reliability | stage `/32` strict 无效 token 比例 |
|---|---:|---:|---:|---:|
| clean | 30.15% | 33.46% | 0.7096 | 34.56% |
| spatial_dropout@0.75 | 82.48% | 81.76% | 0.2051 | 85.10% |
| entire_missing@1.0 | 100.00% | 96.78% | 0.0363 | 100.00% |

大白话说：`spatial_dropout@0.75` 之后，**八成以上的 Depth 都是无效的**，Strict Oracle 会把同样八成以上的 token 的 Depth 几何关系切断 —— 干预力度非常大，但结果是变差。

## 7. per-class 与 sample-level 细分（strict vs original）

### Clean（ΔmIoU = −3.48）

| 类别 | original IoU | strict IoU | Δ |
|---|---:|---:|---:|
| door | 74.71 | 64.87 | **−9.84** |
| support equipment | 45.86 | 37.66 | **−8.20** |
| tube | 67.57 | 60.01 | **−7.56** |
| rail area | 79.58 | 73.08 | **−6.50** |
| cable | 59.21 | 53.87 | **−5.34** |
| rescue equipment | 44.73 | 41.29 | −3.44 |
| metal fixture | 48.09 | 45.00 | −3.09 |
| person | 54.92 | 52.57 | −2.35 |
| mining equipment | 30.82 | 29.06 | −1.76 |
| tools & materials | 73.03 | 71.43 | −1.60 |
| anchoring equipment | 60.88 | 59.33 | −1.55 |
| electrical equipment | 63.21 | 61.80 | −1.41 |
| container | 25.77 | 25.21 | −0.56 |
| indicator | 74.85 | 74.76 | −0.09 |
| electronic equipment | 52.65 | 53.81 | **+1.16** |

### Spatial Dropout@0.75（ΔmIoU = −1.77）

| 类别 | original IoU | strict IoU | Δ |
|---|---:|---:|---:|
| support equipment | 43.12 | 37.88 | **−5.24** |
| door | 67.28 | 63.83 | −3.45 |
| mining equipment | 32.54 | 29.14 | −3.40 |
| metal fixture | 47.18 | 44.58 | −2.60 |
| tube | 61.58 | 59.14 | −2.44 |
| cable | 55.14 | 53.01 | −2.13 |
| rail area | 74.82 | 72.74 | −2.08 |
| electronic equipment | 48.01 | 46.05 | −1.96 |
| tools & materials | 70.62 | 69.11 | −1.51 |
| rescue equipment | 42.81 | 41.61 | −1.20 |
| indicator | 75.29 | 74.67 | −0.62 |
| anchoring equipment | 59.56 | 58.96 | −0.60 |
| electrical equipment | 61.90 | 61.44 | −0.46 |
| container | 25.30 | 25.24 | −0.06 |
| person | 51.10 | 52.38 | **+1.28** |

损失不是集中在某一个类，而是**15 个类里 14 个都变差**；变差的幅度与「该类是否依赖细几何/边界」并不呈现清晰关系，更像是一次整体的表征扰动。

### 逐样本层面

| Condition | 胜 | 平 | 负 | 逐样本平均 Δ |
|---|---:|---:|---:|---:|
| clean | 96 | 34 | 188 | −0.599 |
| spatial_dropout@0.75 | 99 | 44 | 175 | −0.252 |
| entire_missing@1.0 | 122 | 69 | 127 | −0.012 |

配对 bootstrap（196 个 location group，10000 次重采样，seed `20260908`，沿用项目预注册规则 `tools.mve.dvc_a1_core.paired_effect`）：

| Variant | Condition | 点估计 | 95% 区间 |
|---|---|---:|---|
| strict | clean | −0.765 | [−1.057, −0.494] |
| strict | spatial_dropout@0.75 | −0.325 | [−0.503, −0.158] |
| strict | entire_missing@1.0 | +0.002 | [−0.017, +0.020] |
| aggregated | spatial_dropout@0.75 | −0.324 | [−0.500, −0.161] |
| geometry_off | spatial_dropout@0.75 | −0.328 | [−0.508, −0.160] |

## 8. clean 有没有受到伤害？为什么？

**受到了严重伤害：Δclean = −3.48（strict）、−3.26（aggregated）、−4.45（geometry_off）。容忍度是 0.50，实测超出约 7 到 9 倍。**

原因（代码事实 + 实测支持）：

1. **`clean` 本来就不是「Depth 全有效」。** 全量扫描 318 张 `val-dev`：无效像素比例 mean `30.15%`、median `26.88%`、**没有任何一张图完全没有无效像素**。所以 Strict Oracle 在 `clean` 上也会切断 stage `/4` 上 **33.46%** 的 token 的 Depth 几何关系 —— 大约 **56%** 的 token 对因此失去 Depth 项（$1-0.665^2$）。
2. **这些被切断的几何关系并不是无用的。** 由于无效 Depth 在模型眼里是常数 `−1.7143`，模型在训练中**学会了利用这个常数**：它让无效区域内部几乎不互相惩罚（`|Δd|≈0`），同时把无效区域与有效区域明显推开。这套「用无效标记当地理线索」的行为虽然是人为伪影，但训练后已经成为模型表示能力的一部分，删掉它是净损失。
3. **越激进删得越多，损失越大**：`geometry_off`（全删）−4.45 > `strict`（只删涉及无效的）−3.48 > `aggregated`（按比例软删除）−3.26。这个顺序与「删掉多少 Depth 几何信息」完全一致，说明损失就是来自「删掉了有用的几何信息」这一件事本身。

## 9. 边界条件 `entire_missing@1.0` 的行为是否符合预期？

**符合。** 

- 此时原始网格 validity 全为 0，因此所有真实 token 的 reliability 都是 0，Depth 几何项在真实 token 上被完全取消，只剩位置先验 + RGB 语义路径（以及 Q/K/V、decoder 全部不变）。
- 实测 `strict 52.61` vs `geometry_off 52.61`：**confusion matrix 逐位相同（0 个 cell 不同、0/318 张样本不同）**。也就是说「全部无效」时 Strict Oracle 精确退化成 Depth-Geometry-Off，没有任何多余改动。
- 之所以还能有 3.2% 的 token 保持 validity，是因为**padding 条带被有意当作有效**（见第 3 节）；但那条带内插值出来的 Depth 精确等于 `0.0`，两两 `|Δd| = 0`，Depth 项恒为 0，所以数值上仍与「全关」完全一致。
- 与 `original` 相比 Δ = **+0.06**，逐样本平均 −0.012、区间 [−0.017, +0.020]。**这是「无变化」而不是「改善」**，不应当作任何上限或收益来解读。

**一个实现层面的副产品（值得记录）：** `geometry_off` 在三个 condition 上给出**完全相同**的 confusion matrix（`52.61 / 52.61 / 52.61`，逐位相同）。这从数值上证明了「Depth 只通过 `GeoPriorGen` 进入模型」这一代码事实 —— 关掉 Depth 几何项之后，模型对 Depth 输入的依赖被完全切断。

**同时必须指出一个与 strict 不同的地方：** `aggregated` 在 `entire_missing` 下**不等于** `strict`（185 个 confusion cell 不同、288/318 张样本不同）。原因是软 reliability 在 padding/真实交界处不为 0，会保留一部分 Depth 几何项。这不违反边界预期，因为边界预期只约束「严格 validity」这一族。

## 10. selective Oracle 是否优于 Geometry-Off？（本轮最重要的问题）

| Condition | strict | geometry_off | strict − off |
|---|---:|---:|---:|
| clean | 53.58 | 52.61 | **+0.97** |
| spatial_dropout@0.75 | 52.65 | 52.61 | **+0.04** |
| entire_missing@1.0 | 52.61 | 52.61 | **0.00**（逐位相同） |

**结论：在真正关心的 `spatial_dropout` 上，选择性 Oracle 与一刀切几乎完全一样（+0.04 分）。**

这意味着用户提出的那个判据（“如果 `Oracle ≈ Geometry-Off`，说明可能只是 Depth geometry 整体减弱更安全，会削弱 spatial reliability 的必要性”）成立：

- 我们**没有**获得「局部精确关闭错误几何 > 整体放弃 Depth 几何」的证据；
- 在 `clean` 上选择性确实少损失 0.97 分，说明「少删一点就少伤一点」，但这是「删得少」的效果，不是「删得准」的效果；
- 一旦进入 `spatial_dropout`（82% 的 Depth 无效），选择性已经没有可选择性可言 —— 可保留的有效 pair 太少（stage `/4` 上约 18% 的 token 有效，有效–有效 pair 只占约 3.4%），于是它与「全关」自然趋同。

**大白话：在坏区占八成以上时，「聪明地只关坏的部分」和「干脆全关」是一回事；而在坏区占三成时，「关掉涉及坏区的几何」本身就伤得太重。两种情形都不支持继续做 spatial reliability 门控。**

## 11. 最终判断：GO / NO-GO

### **NO-GO**

按预先写好的门槛（`spatial_dropout` 上 `ΔmIoU ≥ +0.50`、配对 bootstrap 区间下界 > 0、clean 损失不超过 0.50 个百分点）：

| 判据 | 门槛 | 实测 | 通过 |
|---|---|---:|---|
| strict 在 `spatial_dropout` 的增益 | ≥ +0.50 | **−1.77** | ✗ |
| strict 的 bootstrap 区间下界 | > 0 | **−0.503**（区间 [−0.503, −0.158]） | ✗ |
| clean 损失 | ≥ −0.50 | **−3.48** | ✗ |

三条判据**全部不通过，且三条的差距都很大**，不是「接近门槛但差一点」。`aggregated` 与 `geometry_off` 的结论同样为负。

**因此：**
- 本阶段**不能**进入 B1-V（`Oracle validity → 预测 reliability → 同样的 geometry action`）的训练；
- **不应**转而训练一个更复杂的 reliability gate 来「抢救」这一族方法；
- `MMFR-B1-learned-geometry-adapter-v1` 中「validity-aware geometry suppression」这一族应当在当前证据下暂缓，而不是加码。

**这个 NO-GO 是「动作不成立」，不是「诊断不成立」。** 本轮完全没有测试 reliability 预测器的准确度 —— 我们测的是：**在诊断完全正确的前提下，这个动作本身值不值得做**。答案是不值得。所以不需要再去优化诊断能力。

## 12. NO-GO 之后：失败最可能来自哪里，下一步应该验证什么

### 12.1 最可能的原因（按证据强度排序）

1. **Depth 几何项的「正确性」与「有用性」是两件事，而模型已经把它训练成了有用的伪影。**
   无效 Depth 的常数编码 `−1.7143` 让「无效区域」成为一个几何上连贯的平面，并在训练中被模型当作可观测量使用。取消它等于删除一个模型已依赖的特征，净效果必然是负的。这是最直接、证据最强的解释：损失随「删除量」单调增加（−3.26 → −3.48 → −4.45）。
2. **Depth 几何项对注意力的影响力本来就很小，因此它不是一个好的干预切入口。**
   对全部 29 个 block 的实测：Strict gate 引起的注意力总变差均值仅 **0.33%（W）/0.19%（H）**，完全关掉也只有 **0.37%**；stage 4（Full GSA）学到的 Depth 权重均值只有 `−0.009`。**一个对注意力影响力不到 1% 的量，很难承载「修复错误几何」这种强干预** —— 要么它不够强、改不动结果，要么改了也只是噪声级别的扰动（实测正是后者：结果普遍略差）。
3. **在 `spatial_dropout` 下可保留的几何关系太少，动作退化成「整体削弱」。**
   82.48% 的 Depth 无效、stage `/4` 上 81.76% 的 token 被判无效、有效–有效 pair 只剩约 3.4%。在这种稀疏度下，任何 pointwise validity 门控都近似等价于全局关闭（实测 +0.04）。
4. **本轮的 action 定义（只乘 `weight[1]*mask_d`）可能过于保守。** 它不改变 `sin/cos`、Q/KV、`lepe`、FFN、decoder。如果性能下降来自「Depth 几何的缺失让 backbone 表征整体偏移」，那么只在 bias 上做乘法可能既不够精确也不够充分。

### 12.2 下一步建议验证的方向（只列建议，不实施）

**方向 1：先验证「Depth 几何是否真的在 spatial_dropout 下被误用」，而不是继续做门控。**
本轮已经说明「关掉它会更差」，但没有证明「它在坏区产生的几何关系是错的」。一个更便宜、更直接的诊断是：在 `spatial_dropout` 下，**保持 Depth 几何不变、只改变坏区的 Depth 数值**（例如把坏区填成有效区域的均值、或填成全局中位数），看结果是否改善。如果改善，说明问题在「坏区的数值编码」而不是「几何关系是否应存在」，那么正确的方向是**改变无效 Depth 的表示**（例如加一个显式的 invalid 通道），而不是门控几何。

**方向 2：如果还要碰 geometry，先做「影响力放大」的门槛检查。**
在投入任何新模块之前，先确认「Depth 几何项对注意力的影响力是否会随干预强度单调放大」。可以做一个纯推理的 dose 实验：把 `weight[1]` 乘 `0 / 0.25 / 0.5 / 1 / 2 / 4`，观察 `spatial_dropout` 上 mIoU 的变化曲线。如果曲线在 `weight[1]` 附近非常平坦，就说明这条路径天花板很低，不值得继续投入。

**方向 3：把「无效 Depth 的数值编码」当作独立的研究问题。**
本轮最强证据（clean 30% 无效却仍然变差 3.48 分）指向一个更基本的假设：**当前 `raw 0 → −1.7143` 的编码方式本身在制造伪几何线索**。这是一条与 validity 门控不同、且更可能产生收益的路线（例如为 Depth 增加显式 validity 通道、或把无效位置改为对几何项中性的表示）。它属于「改变输入表示」，不属于本轮被否决的「validity-aware geometry suppression」。

**方向 4：`misalignment` 仍然完全未动。**
本轮只处理了「invalid / unavailable evidence」，`misalignment` 属于「valid-but-wrong-location evidence」，机制不同，仍然是一个独立的、尚未被本轮任何结果否决的问题。Oracle-B（misalignment geometry realignment）仍然是可选的独立路线。

**明确不在建议内：** 不训练 reliability estimator；不做普通 Depth feature gate $F_D'=r_DF_D$；不做 Depth reconstruction/completion；不加 MoE / expert / router。

### 12.3 本报告没有做的事（边界声明）

- 没有训练任何参数，没有产生任何新 checkpoint；
- 没有改变 dataset、corruption generator、RGB/Depth 预处理、normalization、decoder、loss、checkpoint selector 或 evaluator；
- 没有运行完整测试套件，没有运行云端任务，没有触碰 official test（仍为 `sealed_unread`）；
- 结论只覆盖 `val-dev` 的 318 张图、单一 checkpoint、单一 seed，属于 development 级证据，不是论文级统计结论。

## 13. 复现命令

### 13.1 身份与证据清单

```text
git HEAD（本批次运行时的仓库 HEAD）: 11354a1be5357980a0240785868765cb69f4cd9e（提交信息 "推理完成"）
工作区状态                          : 有未提交改动（本阶段新增的 6 个文件与 models 两处改动；见 5.1 / 5.2）
checkpoint                          : experiments/MMFR_A2_v3/checkpoints/selector-epoch-420.pth
                                      SHA-256 2b72eb7f28e5c4b4f52bdc43e2ac5c7298169325a412bfbcbc7a022c29822597
config                              : local_configs.MUSeg.DFormerv2_S_MMFR_A2_DepthCorrupt_v3
protocol                            : protocols/mmfr-a2-train-integration-v3.template.json
                                      raw SHA-256 814da56321fc5c4e0118b762177877553dfa3ed71e8f72721cb3d65764849da3
split                               : data/splits/MUSeg/dev-v1/val-dev.txt（val_dev）
                                      SHA-256 1d0719d8f64f016d48995c25ab66d4004d76b7155d9efeef7cbb7454c0dd0e83，318 条
dataset root                        : D:/0Project/dataset/MUSeg_DFormer
evaluation seed                     : 2026091401（同时是 corruption seed 与 forward base RNG seed）
正式结果                            : experiments/MMFR_OracleA/eval/<variant>/<condition>/metrics.json（12 份）
                                      experiments/MMFR_OracleA/eval/summary.json
                                      experiments/MMFR_OracleA/eval/oracle-a-analysis.json
no-action 基线                      : experiments/MMFR_OracleA/noaction16/（16 样本 × 3 condition × 6 variant）
单元测试                            : outputs/mmfr-oracle-a-unit-tests/oracle-a-unit-tests.json（66/66 PASS）
几何审计                            : outputs/mmfr-oracle-a-geometry-audit/geometry-audit.json
smoke                               : experiments/MMFR_OracleA/smoke3/（4 样本 × 3 condition × 6 variant）
本报告结果表（纯表格版）            : doc/reports/2026-09-19-museg-mmfr-oracle-a-tables.md
```

每个 `metrics.json` 内含：`variant`（名字、mode、是否 depth_geometry_off、`geometry_implementation`）、`condition`（含两个 spec 的 kind/severity）、`identity`（runner/checkpoint/split/protocol/condition-definition/config/seed/sample-count/sample-set 哈希）、`metrics_percent`（mIoU/mAcc/mF1 与**逐类 IoU**）、`confusion_matrix`、`per_sample`（逐样本 mIoU/mAcc/mF1 + **逐样本 confusion matrix** + corrupted depth 与 validity 的 SHA-256）、`validity`（validity 来源、聚合方式、pair 公式、干预位置、逐样本原始网格无效比例、**逐 stage 覆盖率**）、`timings`、`runtime`（峰值显存、view batching）、`official_test_included=false`、`timestamp_utc`。

**与需求中要求的记录字段的差异（如实说明）：** 本 runner 不在每个 `metrics.json` 里写 `git commit` 字段（git 身份记录在本节的报告里）；`corruption seed` 以 seed-words 规则的形式记录在 `summary.json` 的 `identity.evaluation_seed` 与冻结 runner 的既有口径中，逐样本 depth 摘要已落盘可复核。

```bash
# 单元测试（Test 1–5）
python -m tools.mmfr.oracle_a_unit_tests \
  --checkpoint experiments/MMFR_A2_v3/checkpoints/selector-epoch-420.pth \
  --output-dir outputs/mmfr-oracle-a-unit-tests --device cuda --samples 2

# 只读几何审计（CPU）
python -m tools.mmfr.oracle_a_geometry_audit \
  --checkpoint experiments/MMFR_A2_v3/checkpoints/selector-epoch-420.pth \
  --output outputs/mmfr-oracle-a-geometry-audit/geometry-audit.json --sample-index 0 --scale 1.0

# 正式 318 样本评测（3 condition × 4 variant）
python -m tools.mmfr.run_oracle_a \
  --checkpoint experiments/MMFR_A2_v3/checkpoints/selector-epoch-420.pth \
  --expected-checkpoint-sha256 2b72eb7f28e5c4b4f52bdc43e2ac5c7298169325a412bfbcbc7a022c29822597 \
  --dataset-root D:/0Project/dataset/MUSeg_DFormer \
  --split data/splits/MUSeg/dev-v1/val-dev.txt \
  --expected-split-sha256 1d0719d8f64f016d48995c25ab66d4004d76b7155d9efeef7cbb7454c0dd0e83 \
  --output-dir experiments/MMFR_OracleA/eval \
  --variants original,strict,aggregated,geometry_off \
  --conditions "clean,spatial_dropout@0.75,entire_missing@1.0" \
  --resume --view-batching 2

# 结果分析
python -m tools.mmfr.analyze_oracle_a --output-dir experiments/MMFR_OracleA/eval
```
