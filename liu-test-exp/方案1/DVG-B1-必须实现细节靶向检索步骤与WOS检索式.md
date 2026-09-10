# DVG-B1 项目内实现定位结果与剩余 WOS 靶向检索式

> **核对范围：** 当前 DFormerv2 项目源码、作者原始保留副本 `D:/0Project/origin/DFormer`、本项目 MUSeg 五尺度翻转 evaluator、DVC-A1 corruption/preflight 实现，以及本地论文 `doc/paper/Yin 等 - DFormerv2 Geometry self-attention for RGBD semantic segmentation/`。
>
> **当前结论：** GSA 的 depth geometry contribution、四级调用链、最小 gate 插入点、五尺度翻转几何链、q=0 输入等价、有限值检查和 JSON 证据框架都能从项目内直接定位，不再需要外部文献检索。真正仍需外部参考标准的只有两项：**像素 corruption mask 如何确定性聚合为四级 token reliability**，以及 **token reliability 如何提升为作用于成对 depth geometry contribution 的 pairwise gate**。
>
> **边界：** 本文只完成代码和论文层面的实现定位，没有修改模型代码、运行 GPU 或形成 Oracle 实验结果。

## 1. 已由项目和本地论文闭合：GSA 真实计算链

### 1.1 论文定义

本地 DFormerv2 论文把每个深度 patch 的平均深度记为 $z_{ij}$，并用 patch 间绝对深度差构造 depth relationship：

$$
D_{ij,i'j'} = |z_{ij} - z_{i'j'}|
$$

空间关系使用 Manhattan distance：

$$
S_{ij,i'j'} = |i-i'| + |j-j'|
$$

论文说明使用两个可学习权重融合 depth prior 与 spatial prior，形成 geometry prior。论文证据位于：

- `doc/paper/Yin 等 - DFormerv2 Geometry self-attention for RGBD semantic segmentation/Yin 等 - DFormerv2 Geometry self-attention for RGBD semantic segmentation.md:68-87`；
- GSA 公式与分解式计算位于同文件 `91-127`；
- 四级金字塔与 depth 四尺度处理说明位于同文件 `127-140`。

论文写出的 GSA 为：

$$

\operatorname{GeoAttn}(Q,K,V,G)

\left(\operatorname{Softmax}(QK^T)\odot\beta^G\right)V
$$

### 1.2 当前项目的实际实现

当前代码使用 `GeoPriorGen` 生成 geometry prior。它先将 Depth 双线性插值到当前 block 的 $H\times W$，分别计算 spatial decay 与 depth decay，再用两个可学习权重相加。

```173:212:models/encoders/DFormerv2.py
    depth_map = F.interpolate(depth_map, size=HW_tuple, mode="bilinear", align_corners=False)
    # ...
    mask_h = self.weight[0] * mask_h.unsqueeze(0).unsqueeze(2) + self.weight[1] * mask_d_h
    mask_w = self.weight[0] * mask_w.unsqueeze(0).unsqueeze(2) + self.weight[1] * mask_d_w
    # ...
    mask = self.weight[0] * mask + self.weight[1] * mask_d
```

因此当前代码中的两类 contribution 已经明确分离：

- `self.weight[0] * mask`、`self.weight[0] * mask_h`、`self.weight[0] * mask_w`：spatial contribution；
- `self.weight[1] * mask_d`、`self.weight[1] * mask_d_h`、`self.weight[1] * mask_d_w`：depth geometry contribution。

对于第四级 Full GSA，设当前 token 数 $L=H\times W$：

- spatial contribution：`[N, L, L]`；
- depth contribution：`[B, N, L, L]`；
- 合成 geometry mask：`[B, N, L, L]`。

实际 attention 顺序是先把合成 geometry mask 加到 query-key logits，再做 softmax：

```314:321:models/encoders/DFormerv2.py
        qk_mat = qr @ kr.transpose(-1, -2)
        qk_mat = qk_mat + mask
        qk_mat = torch.softmax(qk_mat, -1)
        output = torch.matmul(qk_mat, vr)
```

即当前项目实际执行：

$$
\operatorname{Attention}

\operatorname{Softmax}
\left(
QK^T + w_s P_s + w_d P_d
\right)V
$$

论文采用“softmax 后乘几何衰减”的表达，当前代码采用“把对数衰减作为 bias 加到 softmax 前”的实现。DVG-B1 必须以当前 checkpoint 对应的项目代码为实际执行对象，不能用论文公式替换现有 forward 语义。

### 1.3 分解式 GSA

前三个 stage 使用 `Decomposed_GSA`，先沿宽度方向计算 attention，再沿高度方向计算 attention：

```247:264:models/encoders/DFormerv2.py
        qk_mat_w = qr_w @ kr_w.transpose(-1, -2)
        qk_mat_w = qk_mat_w + mask_w.transpose(1, 2)
        qk_mat_w = torch.softmax(qk_mat_w, -1)
        v = torch.matmul(qk_mat_w, v)

        qk_mat_h = qr_h @ kr_h.transpose(-1, -2)
        qk_mat_h = qk_mat_h + mask_h.transpose(1, 2)
        qk_mat_h = torch.softmax(qk_mat_h, -1)
        output = torch.matmul(qk_mat_h, v)
```

形状为：

- 高度方向 depth contribution：`[B, N, W, H, H]`；
- 宽度方向 depth contribution：`[B, N, H, W, W]`；
- spatial contribution 先分别生成为 `[N, H, H]` 与 `[N, W, W]`，再广播到 batch 和另一空间轴。

`sin/cos` rotary position encoding 只由 token 索引生成，不依赖 Depth，不能被当作 depth geometry contribution 一起门控。

## 2. 已由项目闭合：四级结构和 gate 的唯一插入点



### 2.1 DFormerv2-S 四级配置

`DFormerv2_S` 的实际配置为：

```650:658:models/encoders/DFormerv2.py
def DFormerv2_S(pretrained=False, **kwargs):
    model = dformerv2(
        embed_dims=[64, 128, 256, 512],
        depths=[3, 4, 18, 4],
        num_heads=[4, 4, 8, 16],
        heads_ranges=[4, 4, 6, 6],
```

四级调用由统一循环完成：

```620:637:models/encoders/DFormerv2.py
        for i in range(self.num_layers):
            layer = self.layers[i]
            x_out, x = layer(x, x_e)
```

`split_or_not=(i_layer != 3)` 决定前三个 stage 使用 `Decomposed_GSA`，第四个 stage 使用 `Full_GSA`。

对于未缩放且 padding 后输入为 `480×640` 的 view，四级 RGB 特征尺寸为：

- Stage 0：`120×160×64`，3 blocks，4 heads；
- Stage 1：`60×80×128`，4 blocks，4 heads；
- Stage 2：`30×40×256`，18 blocks，8 heads；
- Stage 3：`15×20×512`，4 blocks，16 heads。

其他测试尺度使用各自 padding 后输入尺寸的 `1/4、1/8、1/16、1/32` 网格，不应写死为上述四组尺寸。

### 2.2 唯一 depth-only gate 插入点

最小且不改变 spatial contribution 的插入点是 `GeoPriorGen.forward` 中 spatial/depth 两项相加之前：

```188:208:models/encoders/DFormerv2.py
        mask_d_h = self.generate_1d_depth_decay(...)
        mask_d_w = self.generate_1d_depth_decay(...)
        mask_h = self.generate_1d_decay(...)
        mask_w = self.generate_1d_decay(...)
        mask_h = self.weight[0] * mask_h.unsqueeze(0).unsqueeze(2) + self.weight[1] * mask_d_h
        mask_w = self.weight[0] * mask_w.unsqueeze(0).unsqueeze(2) + self.weight[1] * mask_d_w
        # ...
        mask_d = self.generate_depth_decay(...)
        mask = self.weight[0] * mask + self.weight[1] * mask_d
```

目标形式只能是：

$$
M = w_s P_s + w_d\left(R\odot P_d\right)
$$

其中 $R$ 是由 Oracle corruption mask 生成的 pairwise gate。`self.weight[0] * spatial`、`sin/cos`、Q/K/V、LEPE、FFN、decoder 和 logits 融合保持原样。

以下位置必须排除：

- 对合成后的 `mask`、`mask_h` 或 `mask_w` 整体乘 gate：会同时修改 spatial contribution；
- 在 `qk_mat + mask` 后门控：此时 depth 与 spatial 已不可分；
- 将整个 `geo_prior` 置零：会同时移除 spatial decay 和 rotary position encoding；
- 修改或清零 `x_e`：这属于改变 Depth 输入，不是只抑制既有 depth geometry contribution；
- 修改最终 logits：不再属于 GSA depth-only gate。



### 2.3 最小参数传递路径

当前接口只接收 RGB 和 Depth：

```226:239:models/builder.py
    def encode_decode(self, rgb, modal_x):
        x = self.backbone(rgb, modal_x)
```

```414:425:models/encoders/DFormerv2.py
    def forward(self, x: torch.Tensor, x_e: torch.Tensor, split_or_not=False):
        geo_prior = self.Geo((h, w), x_e, split_or_not=split_or_not)
        x = x + self.drop_path(self.Attention(self.layer_norm1(x), geo_prior, split_or_not))
```

最小改动路径应增加可选 `oracle_corruption_mask=None`，并按以下链路传递：

`EncoderDecoder.forward/encode_decode` → `dformerv2.forward` → `BasicLayer.forward` → `RGBD_Block.forward` → `GeoPriorGen.forward`。

Attention 类不需要接收 Oracle mask；它继续只消费已经合成的 geometry prior。四个 stage 全部传入同一个 view-specific 原始 mask，由各 stage 根据自身 $H\times W$ 形成 reliability 和 pairwise gate。

## 3. 已由项目闭合：现有 corruption mask 和五尺度翻转链



### 3.1 原始 corruption mask

`depth_edge_candidates` 与 `build_corruption_masks` 已在原始二维 `uint16 Depth16` 网格生成五个 condition 的布尔 mask，并冻结确定性、嵌套关系、匹配面积和 SHA-256。

```95:182:tools/mve/dvc_a1_core.py
def depth_edge_candidates(...):
    # Depth16 相邻像素相对跳变、3×3 boundary 膨胀、11×11 guard

def build_corruption_masks(...):
    # clean、boundary-q25/q50/q75、nonboundary-q50
```

`quantized_condition_depth` 只把 mask 指定的原始 Depth16 像素置零：

```185:190:tools/mve/dvc_a1_core.py
def quantized_condition_depth(...):
    corrupted = depth16.copy()
    corrupted[mask] = 0
    return quantize_depth(corrupted, depth_max_raw)
```

当前 mask 没有作为独立张量进入模型。DVG-B1 需要保留现有 corrupted Depth 输入，同时把同一 condition 的布尔 corruption mask 作为额外 Oracle 输入。

### 3.2 五尺度、水平翻转和原始网格恢复

`build_msflip_views` 已固定 RGB/Depth 的同步几何链：

```93:144:tools/evaluate_museg_checkpoint.py
def build_msflip_views(...):
    # 五尺度同步 resize RGB/Depth
    # 每个尺度生成原图和水平翻转 view
    # 右侧/底部 padding 到 32 的倍数
```

`msflip_whole_logits` 已固定：

- FP32 forward；
- 必要时把模型输出恢复到 padding 后尺寸；
- 去除右侧和底部 padding；
- flip view 的 logits 逆水平翻转；
- 每个 view 恢复到原始 Label 网格；
- 10 个 view 的 pre-softmax logits 做 FP32 算术平均。

```155:214:tools/evaluate_museg_checkpoint.py
def msflip_whole_logits(...):
    # unpad -> inverse flip -> original Label grid -> FP32 mean
```

DVG-B1 的 mask view 必须复用相同的 `scale`、`flipped`、`scaled_size_hw` 和 `padded_size_hw` 元数据：

1. 从原始 Depth16 corruption mask 生成对应尺度的 mask/reliability；
2. flip view 同步沿宽度轴翻转；
3. 只在右侧和底部 padding；
4. padding 区域固定为“不属于 corruption”，避免把 evaluator padding 新增为实验受损区域；
5. 将 view-specific mask 随 RGB/Depth 一起传入模型。

其中“mask 从原图缩放到 view 的插值/聚合规则”仍需外部标准，见第 6 节。

## 4. 已由项目闭合：no-op 数值等价的实现方式

现有 DVC-A1 preflight 已检查：

- corruption mask 两次生成的 hash 一致；
- boundary mask 嵌套；
- boundary/nonboundary q50 面积匹配；
- clean/q=0 的量化 Depth 数组与生产 Depth8 完全相同；
- 输出有限；
- logits 尺寸等于原始 Label 网格；
- checkpoint strict load；
- `preflight.json` 等结构化证据落盘。

```448:574:tools/mve/run_dvc_a1.py
def run_preflight(...):
    # mask_hash_stable、nested、matched_q50_count、q0_decoded_array_equal
    # finite、output_size_hw、expected_size_hw
```

但现有 q=0 只证明输入数组等价，不能证明未来 gated model 与原模型输出等价。

DVG-B1 不需要为 no-op 容差继续检索文献。实现时应设置明确的原路径旁路：

```text
oracle_corruption_mask is None 或 mask 中没有任何受损位置
    -> 不生成 reliability
    -> 不计算 pairwise gate
    -> 直接执行当前未修改的 spatial + depth 合成表达式
```

因此必须检查：

1. 原模型与 `oracle_corruption_mask=None` 的四级输出、最终 pre-softmax logits 使用 `torch.equal` 完全相等；
2. clean/q=0/all-trusted mask 归一化到同一个旁路后，也使用 `torch.equal` 完全相等；
3. 若实现未走同一旁路而需要浮点容差，说明引入了不必要的新数值路径，应先修正实现，而不是外部搜索一个更宽容的 tolerance。



## 5. 已由项目闭合：最小 preflight 审计字段

复用现有 `run_preflight`、`msflip_whole_logits` 和原子 JSON 写入方式即可，不需要外部检索 smoke-test 或日志标准。DVG-B1 只需增加 gate-specific 字段。

每个样本/condition/view 至少记录：

- 原始 corruption mask shape、受损像素数和 SHA-256；
- `scale`、`flipped`、scaled/padded size；
- view mask/reliability 的 shape、最小值、最大值、受损或非全可信数量；
- 四个 stage 的 $H\times W$；
- 每级 token reliability shape 与统计；
- Full GSA 或 H/W 分解的 pairwise gate shape 与统计；
- gate 前后 depth contribution 的 shape；
- spatial contribution 未修改的确认；
- 四级输出和最终 logits 是否有限；
- 最终输出是否恢复到原始 Label 网格；
- no-op 的逐级与最终 logits 完全等价结果。

失败条件：

- mask 与 Depth16 原图不对齐；
- view mask 与 RGB/Depth view 几何不一致；
- stage reliability 或 pairwise gate shape 不匹配；
- gate 影响 spatial contribution；
- q=0/all-trusted 未进入原路径旁路；
- 任一中间量或 logits 非有限；
- 输出未恢复到原始 Label 网格。



## 6. 真正需要外部文献检索的参考标准



### 6.1 待检索项 A：像素 corruption mask 到四级 token reliability 的确定性聚合

**项目内为何不能直接决定：**

- 论文写的是对每个 depth patch 做 average pooling；
- 作者原始保留副本 `D:/0Project/origin/DFormer/models/encoders/DFormerv2.py` 与当前 `models/encoders/DFormerv2.py` 都在 `GeoPriorGen.forward` 中使用 `F.interpolate(..., mode="bilinear", align_corners=False)`；两个完整文件的 SHA-256 均为 `2b0b77ea401d56993aac915883bcb43035927ec991501dba94fb029901009332`，`git diff --no-index` 退出码为 `0`；
- 因此 average pooling 与 bilinear interpolation 的差异来自论文叙述和作者原始代码本身，不是 MUSeg/MVE 适配，也不是当前项目对该文件的意外改动。若此前曾临时改动过该位置，当前内容已经与作者原始副本完全一致；
- 二值 corruption mask 若使用 nearest、area、max pooling 或 average pooling，会分别对应不同的“patch 是否可信”语义；
- corrupted Depth 在 evaluator 和 GSA 内都经历双线性插值，坏像素影响可能扩散到相邻位置，不能把原始布尔 mask 直接按坐标抽样后声称已经覆盖全部受影响 geometry。

**需要文献给出的最小证据：**

- 无效/受损深度 mask、validity mask 或 confidence map 在层级 RGB-D/Depth 网络中的下采样规则；
- 部分有效 patch 使用 any-invalid、valid fraction、average confidence 或其他聚合时的明确语义；
- mask 与经过双线性 resize 的 depth 对齐时，如何定义输出 token reliability；
- 全可信输入必须在所有尺度保持全可信。

**WOS 靶向检索式：**

```text
TS=((("depth validity mask" OR "depth confidence map" OR "depth reliability map" OR "corruption mask") NEAR/5 (downsampl* OR pool* OR resiz* OR "validity propagation" OR "confidence propagation")) AND ("RGB-D" OR depth OR multimodal) AND ("hierarchical transformer" OR "feature pyramid" OR "multi-scale attention" OR "token mask" OR "partial validit*"))
```

**检索后必须填回：**

- view-scale mask/reliability 变换：`<待外部检索后冻结>`；
- Stage 0–3 聚合算子及参数：`<待外部检索后冻结>`；
- 部分受损 patch 的 reliability 定义：`<待外部检索后冻结>`；
- 与双线性 Depth resize 的对齐解释：`<待外部检索后冻结>`；
- 直接代码锚点、commit、文件和函数：`<待外部检索>`。

**关闭条件：** 只能保留一套从原始布尔 mask 到每个 view、每个 stage reliability 的确定性规则；不得在查看模型结果后选择 nearest、max、average 或阈值。

### 6.2 待检索项 B：token reliability 到 pairwise depth contribution gate 的提升规则

**项目内为何不能直接决定：**

当前 depth contribution 表示成对 token 的深度距离，而原始 Oracle mask 表示单个像素或 token 是否受损。项目代码没有规定一个受损 query、受损 key 或两者之一受损时，成对 depth contribution 应如何处理。

需要在以下候选语义中只冻结一种：

- 两端都可信才保留 depth contribution；
- 只按 query reliability 门控；
- 只按 key reliability 门控；
- 使用两端 reliability 的乘积、最小值或其他对称组合；
- 对部分可信 token 采用连续衰减还是先阈值化为硬 gate。

**需要文献给出的最小证据：**

- 局部 depth confidence/validity 如何进入 pairwise attention bias、geometry prior 或 query-key 关系；
- 成对 gate 是否保持 query/key 对称性；
- hard invalid mask 与 continuous reliability 的适用边界；
- gate 为全 1 时保持原 depth contribution 不变。

**WOS 靶向检索式：**

```text
TS=((("depth confidence" OR "depth reliability" OR "validity mask" OR "corruption mask") NEAR/5 ("attention bias" OR "geometry prior" OR "pairwise attention" OR "masked attention")) AND ("RGB-D" OR "depth-guided" OR multimodal) AND ("pairwise reliabilit*" OR "query-key mask*" OR "confidence gate*" OR "multiplicative mask*" OR "attention bias mask*"))
```

**检索后必须填回：**

- 单 token reliability 到 Full GSA `[B, 1, L, L]` gate 的公式：`<待外部检索后冻结>`；
- 单 token reliability 到分解式 H gate `[B, 1, W, H, H]` 的公式：`<待外部检索后冻结>`；
- 单 token reliability 到分解式 W gate `[B, 1, H, W, W]` 的公式：`<待外部检索后冻结>`；
- hard 或 continuous gate 的选择：`<待外部检索后冻结>`；
- 直接代码锚点、commit、文件和函数：`<待外部检索>`。

**关闭条件：** 已得到一个可同时映射到 Full GSA 与 H/W 分解 GSA、只乘 depth contribution、保持 spatial contribution 不变且全可信恒等的 pairwise gate 规则。

## 7. 不再需要执行的外部检索

以下内容已由本项目代码和本地论文直接闭合，不再使用 WOS 扩大检索：

- DFormerv2 正式论文、GSA 公式和四级结构；
- DFormerv2-S 的 blocks、heads、embed dims 和 H/W 分解实现；
- spatial contribution 与 depth geometry contribution 的代码边界；
- 唯一 depth-only gate 插入位置；
- 五尺度、水平翻转、padding、逆 flip 和原始 Label 网格融合；
- corruption mask 的原始 Depth16 身份、确定性和 q=0 输入等价；
- checkpoint strict load；
- FP32、有限值、shape 和 JSON preflight 框架；
- no-op 输出等价策略。



## 8. 下一步

1. 只执行第 6.1 和 6.2 的两条 WOS 检索式。
2. 每个待检索项优先寻找带官方代码的直接实现；只给概念框图的论文不能关闭条目。
3. 对命中结果记录论文、DOI、官方仓库、commit、文件、函数、输入输出 shape 和许可证。
4. 将两项唯一规则填回本文后，才能形成独立 B1 protocol；本文仍不授权代码修改或 GPU 运行。

