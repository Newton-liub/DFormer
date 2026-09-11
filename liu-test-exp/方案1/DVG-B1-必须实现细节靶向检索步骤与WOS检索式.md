# DVG-B1 项目内实现定位结果与剩余 WOS 靶向检索式

> **核对范围：** 当前 DFormerv2 项目源码、作者原始保留副本 `D:/0Project/origin/DFormer`、本项目 MUSeg 五尺度翻转 evaluator、DVC-A1 corruption/preflight 实现，以及本地论文 `doc/paper/Yin 等 - DFormerv2 Geometry self-attention for RGBD semantic segmentation/`。
>
> **当前结论：** GSA 的 depth geometry contribution、四级调用链、最小 gate 插入点、五尺度翻转几何链、q=0 输入等价、有限值检查和 JSON 证据框架都能从项目内直接定位。11 篇候选论文的全文与常见官方代码已逐篇核对，但没有任何实现同时给出 **像素 corruption mask → 四级 token reliability → Full/H/W pairwise gate** 的唯一完整规则。用户现已把 A/B 选为项目预注册候选：A 使用 view 级双线性对齐与 stage 级有效面积比例，B 使用 query/key 两端 reliability 的对称乘积；两项尚未实现或验证。C 的科学效应量数值仍需单独冻结。[1]–[11]
>
> **大白话说明：** 能找到的常见官方代码已经查完。它们说明了不同候选的真实边界，却不能替项目唯一选型；用户因此在看模型结果前先固定 A/B，下一步不再重复一般性检索，而是冻结 C、物化 protocol 并按计划做低成本资格检查。
>
> **边界：** 本文完成的是只读代码核验和两个小型 CPU 算子语义实验，并记录用户随后作出的 A/B 项目预注册选择。没有修改第三方仓库或模型代码，没有运行模型 forward、GPU、训练、完整测试、official test 或形成 Oracle 性能结果。CPU probe 只能说明 shape、方向性、边缘/padding 和恒等性质，不能证明 A/B 会提升模型。

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



## 6. 真正仍需关闭的参考与预注册标准



### 6.1 待冻结项 A：像素 corruption mask 到四级 token reliability 的确定性聚合

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

**11 篇全文核对后的回填状态：**

- **view-scale mask/reliability 变换：仍未冻结。** *Confidence Propagation through CNNs* 明确采用 confidence max pooling 下采样，并用 argmax 选择最高 confidence 的 feature；跨尺度融合使用 nearest-neighbor 上采样。BCAP-Net 则对 sparse depth 使用带有效性分母的动态 weighted pooling，并只在多尺度预测监督中明确 bilinear 上采样。两者处理对象和语义不同，不能直接替代 DVG-B1 的 corruption mask view resize。
- **Stage 0–3 聚合算子及参数：仍未冻结。** 当前最强候选机制是 normalized convolution 的连续 confidence propagation：

$$
C_{i,j}^{l}
=
\frac{
\sum_{m,n} C_{i+m,j+n}^{l-1}\Gamma(W_{m,n}^{l})+\epsilon
}{
\sum_{m,n}\Gamma(W_{m,n}^{l})
}.
$$

  但全文没有给出可直接移植到 DFormerv2 四级 token 网格的 kernel、stride、padding 和四级参数；max pooling confidence 也没有给出项目所需的四级固定设置。
- **部分受损 patch 的 reliability 定义：已有候选公式，仍未冻结为项目规则。** Normalized convolution 明确把二值 validity mask 视为连续 confidence field，部分有效窗口采用 confidence-weighted normalized aggregation；BCAP-Net 使用 $\mathbb I(S_y>0)$ 排除无效 sparse depth。两篇都没有把“corruption mask 中受损面积比例”直接定义为 DFormerv2 token reliability，也没有给出满足项目 strict no-op 的空窗口与全 1 规则。
- **与双线性 Depth resize 的对齐解释：仍缺。** 已核对全文分别使用 nearest-neighbor、bilinear prediction upsampling、joint bilateral upsampling、未说明模式的 Resize 或局部 bilinear sampling；没有一篇同时规定 corruption mask 与 `F.interpolate(..., mode="bilinear", align_corners=False)` Depth 路径的共同坐标和影响域语义。
- **直接代码锚点：已完成关键核验，但没有关闭 A。** *Confidence Propagation through CNNs* 的作者官方主仓库为 `https://github.com/abdo-eldesokey/nconv`，NYU 辅助仓库为 `https://github.com/abdo-eldesokey/nconv-nyu`。[1] `NConv2d.forward(data, conf)` 对 `data*conf` 与 `conf` 使用同一卷积核，输出 `nomin/(denom+1e-20)+bias` 与 `denom/sum(weight)`；多尺度网络用 `F.max_pool2d(c, 2, 2, return_indices=True)` 选最高 confidence 位置的 feature，随后显式执行 `c_ds /= 4`，上采样使用 nearest。因此它不是简单的 any-valid 二值 max pooling：全 1 confidence 经一次该下采样会变为 `0.25`；卷积零 padding 也会使边缘 confidence 低于 1。该官方实现不能满足 DVG-B1 的 strict all-1 reliability/no-op 要求，只能作为“连续 confidence 可传播”的直接代码证据。
- LightDepth 作者官方仓库 `https://github.com/fatemehkarimii/LightDepth` 的 PyTorch 与 TensorFlow `dataloaders.py::dilation` 都在训练阶段对 sparse ground-truth depth 重复做 MaxPool；TensorFlow 版本随后用 nearest resize 恢复尺寸。[3] 这是训练标签稀疏度课程，不是输入 corruption reliability 传播，不能关闭 A。
- NR-MVSNet、LFDA、OPM-MVS 的官方代码分别处理 query/location entropy gate、depth/angular candidate attention与多视图 bilateral/geometry confidence，[5][9][10] 都没有给出 corruption mask 到四级 token reliability 的确定性规则。完整 commit、许可证与函数锚点见第 6.4–6.5 节。

**全文证据入口：**

- `doc/paper/补充材料2/Confidence_Propagation_through_CNNs_for_Guided_Sparse_Depth_Regression/Confidence_Propagation_through_CNNs_for_Guided_Sparse_Depth_Regression.md:17-44,60-86,130-143,178-191`；
- `doc/paper/补充材料2/s13042-026-03156-8/s13042-026-03156-8.md:84-98,155-169,179-209,236-243`；
- 其余 9 篇的逐篇用途和负证据见第 6.3 节。

**关闭条件：** 只能保留一套从原始布尔 mask 到每个 view、每个 stage reliability 的确定性规则；不得在查看模型结果后选择 nearest、max、average 或阈值。

### 6.2 待冻结项 B：token reliability 到 pairwise depth contribution gate 的提升规则

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

**11 篇全文核对后的回填状态：**

- **Full GSA `[B, 1, L, L]` gate：仍未冻结。** *Learning Selective Mutual Attention and Contrast* 给出完整 $HW\times HW$ query-key affinity 与 row-wise softmax 拓扑，但其 pairwise matrix 来自 feature similarity，不是 reliability gate。NL-3A 和 GSCS/SPN 给出的直接 reliability 规则都是邻居/key-only：

$$
w_{p,q}=c_q\hat w_{p,q}.
$$

  该式能证明“邻居可靠性乘基础 affinity”确实存在，也能在 $c_q=1$ 时局部恢复基础 affinity；但它没有 query 端、两端对称组合或 DFormerv2 Full GSA 的全局 shape 说明。
- **分解式 H gate `[B, 1, W, H, H]`：未找到直接公式。** 11 篇全文没有沿固定宽度分组、在高度轴构造 reliability pair 的实现。
- **分解式 W gate `[B, 1, H, W, W]`：未找到直接公式。** 11 篇全文没有沿固定高度分组、在宽度轴构造 reliability pair 的实现。
- **hard 或 continuous：不能由现有全文唯一选择。** NL-3A 使用 $a^{i,j}\in[0,1]$ 连续乘 affinity；GSCS/SPN 使用连续邻居 confidence；NR-MVSNet 则先以 entropy 构造连续 confidence，再用阈值 $\tau=0.3$ 对当前位置的 attention cost volume 做 hard query-only 保留/置零。文献同时存在互不等价的连续 key-only 与 hard query-only 设计。
- **归一化与方向性：已有机制证据，但不等于最终 gate。** NL-3A 先用带学习 normalization factor 的双曲正切式 affinity，再乘邻居 reliability；BCAP-Net 先归一化 affinity，再分别乘 self/neighbor/scale confidence；Multi-Affinity Matrix 使用有向局部 affinity 和非中心 L1 归一化；这些都不是 query/key 对称 gate。
- **全 1 恒等：只得到局部代数性质，未得到完整 forward 证明。** 对 $w_{p,q}=c_q\hat w_{p,q}$，$c_q\equiv1$ 可恢复基础 affinity；但没有一篇论文证明四级输出和最终 logits 与原 DFormerv2 forward 使用 `torch.equal` 完全相等。项目仍必须依靠第 4 节规定的原路径旁路验收。
- **直接代码锚点：已完成可用仓库核验，但没有关闭 B。** NR-MVSNet 官方实现 `models/refinenet.py::DepthUpdate.forward` 的输入 shape 为 `ref_feat [B,C,H,W]`、`cost_volume [B,C,D,H,W]`、`prob_volume [B,D,H,W]`、`depth [B,1,H,W]`；归一化 entropy 后使用 `mask = (1 - entropy) > 0.3`，再把 `[B,1,H,W]` mask 乘到当前位置的 cost feature。[9] 这是 query/location-only hard gate，不是 pairwise query-key gate。LFDA 官方实现 `LFDA.py::Depth_MCA.forward` 对 depth/angular candidate query 与 center-view key 计算 attention，`softmax(dim=-2)` 后执行 `out = query * attn`；它不是空间 token 的 `[B,L,L]` reliability gate。[10] SMAC 官方仓库没有模型代码；作者前作 S2MA 的 `NonLocalBlock` 确实生成 `[B,HW,HW]` Full affinity，并把原 `[B,HW,1]` 的逐 query 权重扩展为 `[B,HW,HW]`，但 S2MA 不是 SMAC，只能作为邻近拓扑参考，不能代替论文 [11] 的正式实现证据。

**全文证据入口：**

- `doc/paper/补充材料2/oe-31-13-22012/oe-31-13-22012.md:92-100,105-140`；
- `doc/paper/补充材料2/507024/507024.md:114-133,247-282`；
- `doc/paper/补充材料2/NR-MVSNet_Learning_Multi-View_Stereo_Based_on_Normal_Consistency_and_Depth_Refinement/NR-MVSNet_Learning_Multi-View_Stereo_Based_on_Normal_Consistency_and_Depth_Refinement.md:188-213,252-257`；
- `doc/paper/补充材料2/Learning_Selective_Mutual_Attention_and_Contrast_for_RGB-D_Saliency_Detection/Learning_Selective_Mutual_Attention_and_Contrast_for_RGB-D_Saliency_Detection.md:150-182,205-265`；
- 其余论文的逐篇用途和负证据见第 6.3 节。

**关闭条件：** 已得到一个可同时映射到 Full GSA 与 H/W 分解 GSA、只乘 depth contribution、保持 spatial contribution 不变且全可信恒等的 pairwise gate 规则。

### 6.3 11 篇论文的逐篇提取结论与编号

以下“可用”只表示能作为机制或负证据写入设计，不表示论文已经替本项目冻结最终规则。

1. **Confidence Propagation through CNNs for Guided Sparse Depth Regression [1]**
   - 最强 A 类证据。明确 `0=missing、1=valid`，把 validity mask 作为连续 confidence field；给出 normalized convolution、逐层 confidence propagation、confidence max pooling、argmax feature selection 和 nearest-neighbor 跨尺度融合。
   - 作者官方主仓库和 NYU 辅助仓库均已固定。代码确认 confidence 下采样后还显式除以 4，且 NConv 零 padding 会让全 1 confidence 的边缘低于 1；它不能直接满足 DVG-B1 strict all-1 reliability/no-op。
2. **BCAP-Net [2]**
   - 给出 $\mathbb I(S_y>0)$ 有效性指示、动态 weighted pooling、$k=3,5,7$ 的局部传播，以及 self、neighbor、scale 三类连续 confidence 与 normalized affinity 的组合。
   - 未找到可确认的官方仓库。`https://github.com/JackChenStrive/BCAP-Net` 的 README 是另一项工作的 partial code release statement，未绑定本文完整题名或 DOI，当前只能记为未核验候选，不能作为官方实现证据。
3. **LightDepth [3]**
   - 对 sparse ground-truth depth 使用重复 MaxPool2D 与 Resize；作者官方仓库已固定。
   - 代码确认它处理训练标签稀疏度课程，不输出输入 reliability，不能关闭 A。
4. **Image-guided dense depth completion network based on hierarchical feature reconstruction and dynamic weight sampling [4]**
   - 给出局部 bilinear sampling kernel、learned attention 和 confidence gating 的邻近机制。
   - 未找到可确认的官方仓库；当前 Markdown 中最终 confidence-gating 公式不完整，且 confidence 是学习产生的，不是 corruption mask 的确定性聚合。
5. **Octagram Propagation Matching for Multi-Scale View Stereopsis [5]**
   - 给出几何、深度、法向、光度 confidence 的乘积与多视图平均，并使用 coarse-to-fine joint bilateral upsampling；官方仓库已固定。
   - 代码包含 dual bilateral weight、bilateral NCC、view weight 与 joint bilateral upsampling，但 confidence 属于 view/depth-hypothesis consistency，不属于 pixel corruption validity，不能关闭 A 或 B。
6. **Non-local affinity adaptive acceleration propagation network（NL-3A）[6]**
   - 最直接 B 类证据之一。预测 $a^{i,j}\in[0,1]$，将邻居像素 reliability 连续乘到 normalized affinity 上；属于有向 key/source-only gate。
   - 未找到可确认的官方仓库。reliability 由网络预测，不是由 corruption mask 聚合；没有 query 端、对称性、Full/H/W 映射或 strict no-op。
7. **Deep Sparse Depth Completion Using Multi-Affinity Matrix [7]**
   - 给出有向局部 CSPN affinity、非中心 L1 归一化和中心 residual weight；另有逐像素 branch confidence 融合。
   - 未找到可确认的官方仓库。branch confidence 没有直接进入 pairwise affinity，也没有 all-1 reliability 恒等和 Full/H/W 映射。
8. **Gaussian Splatting Confidence Supervision for SPN-based Depth Completion [8]**
   - 明确给出 $w_{p,q}=c_q\hat w_{p,q}$，即邻居/key-only confidence 乘基础 affinity；同时区分连续 confidence 与局部 attention 的硬 padding mask。
   - 未找到可确认的官方仓库。这是局部传播公式，不是 DFormerv2 四级 gate；全 1 只局部恢复基础 affinity，不能证明完整 forward 恒等。
9. **NR-MVSNet [9]**
   - 由归一化 entropy 得到 query/location confidence，再以 $\tau=0.3$ hard gate 保留或清零 attention cost volume；官方仓库已固定。
   - 代码输出的 gate 是 `[B,1,H,W]` 当前位置 mask，不是 `[B,L,L]` pairwise gate。它与 NL-3A/GSCS 的 continuous key-only 语义相冲突，证明文献不能唯一决定 hard/continuous 或 query/key 方向。
10. **LFDA [10]**
    - 给出 depth-candidate query 对 center-view key 的连续 cross-attention，并在 softmax 后乘 candidate cost volume；官方仓库已固定。
    - 代码中的 attention 轴是 depth/angular candidates，不是空间 token reliability gate，没有 all-1 identity 或 Full/H/W 映射。
11. **Learning Selective Mutual Attention and Contrast for RGB-D Saliency Detection（SMAC）[11]**
    - 论文给出 $HW\times HW$ Full query-key affinity、row-wise softmax 和跨模态 mutual attention。
    - 作者官方 SMAC 仓库只发布 ReDWeb-S 数据集、统计与结果，没有 SMAC 模型代码。作者前作 S2MA 官方代码确实生成 `[B,HW,HW]` Full affinity，并把 `[B,HW,1]` 的逐 query 权重扩展为 `[B,HW,HW]`；但 S2MA 不是 SMAC，不能用来替代 [11] 的正式实现证据。

### 6.4 官方代码、固定版本与许可证核验

所有已克隆仓库均位于 `D:/0Project/origin`。本轮只读复核时 8 个仓库的 `git status --porcelain` 均为空，说明没有改动第三方仓库。

| 编号 | 官方代码结论 | 本地仓库、分支与固定 HEAD | 许可证状态 |
|---|---|---|---|
| [1] | 作者官方主仓库 `https://github.com/abdo-eldesokey/nconv`；NYU 辅助仓库 `https://github.com/abdo-eldesokey/nconv-nyu` | `nconv`, `master`, `d85d4b659f2207b397c62d81f27f363baf3397be`；`nconv-nyu`, `master`, `a708c2dbeba9679d8474e493ef2c40e07846cf8e` | 主仓库 GPL-3.0；NYU 仓库未声明 |
| [2] | 未找到可确认官方仓库；同名 GitHub 候选未通过论文身份绑定 | 未克隆为官方证据 | 未确认 |
| [3] | 作者官方仓库 `https://github.com/fatemehkarimii/LightDepth` | `LightDepth`, `main`, `785575ad9baa32e476a3f5df84d00f7e764ef915` | 未声明 |
| [4] | 未找到可确认官方仓库 | — | 未确认 |
| [5] | README 声明 official implementation 并绑定论文 DOI 的仓库 `https://github.com/RayKhuboni/OPM-MVS` | `OPM-MVS`, `main`, `da9635893f81608a997036d8a48e907e03e70d99` | 未声明 |
| [6] | 未找到可确认官方仓库 | — | 未确认 |
| [7] | 未找到可确认官方仓库 | — | 未确认 |
| [8] | 未找到可确认官方仓库 | — | 未确认 |
| [9] | 论文正文给出实现 URL `https://github.com/wdkyh/NR-MVSNet` | `NR-MVSNet`, `main`, `0c1e3d14c4ee6a54f5fdbabca6f50687067761c9` | 未声明 |
| [10] | 论文正文给出且 README 声明官方实现的仓库 `https://github.com/syt06007/LFDA` | `LFDA`, `main`, `f5eee3ce1c91fbbb2d0a2e6dc2064e45792670cf` | MIT |
| [11] | 作者官方 SMAC 仓库 `https://github.com/nnizhang/SMAC`，但无模型代码；作者前作参考仓库 `https://github.com/nnizhang/S2MA` 不是 SMAC 实现 | `SMAC`, `main`, `e40f4579fcbd5da2f4ea3426f93a5d6988676dc4`；`S2MA`, `master`, `f94fceede09d644f285c271b1d8d41e384e0f8ed` | 两者均未声明 |

“未声明”只表示仓库根目录没有可确认的许可证文件，不等于可以自由复制代码。后续若实现 DVG-B1，应依据论文机制自行编写最小代码，不直接搬运许可证不明或不兼容的第三方实现。

### 6.5 官方代码的关键实现边界

#### 6.5.1 Normalized convolution [1]

- `D:/0Project/origin/nconv/modules/nconv.py::NConv2d.forward`
  - 输入 `data, conf` 为 `[B,C_{in},H,W]`；输出为卷积输出网格上的 `nconv, cout`。
  - 实际数据计算为 `conv(data*conf, weight)/(conv(conf, weight)+1e-20)+bias`。
  - 实际 confidence 输出为 `conv(conf, weight)/sum(weight)`。
- `D:/0Project/origin/nconv/workspace/exp_unguided_depth/network_exp_unguided_depth.py`、对应 disparity 网络与 `nconv-nyu/nconv_sd.py`
  - 下采样为 `F.max_pool2d(c, 2, 2, return_indices=True)`；feature 使用 confidence argmax 的索引选点；随后 `c_ds /= 4`。
  - 上采样为 nearest。
- **对 DVG-B1 的含义：** 该实现支持“validity 可以转成连续 confidence 并逐层传播”，但不支持把 `maxpool/4` 直接当作本项目 reliability。全 1 输入经过下采样后不是 1，零 padding 下边缘也不是严格 1；空窗口 confidence 为 0，但数据输出仍保留 bias。

#### 6.5.2 LightDepth [3]

- `D:/0Project/origin/LightDepth/torch_implementation/scripts/dataloaders.py::dilation`：重复 `MaxPool2d(kernel_size=pool_size)`。
- `D:/0Project/origin/LightDepth/tf_implementation/scripts/dataloaders.py::dilation`：重复 `MaxPooling2D` 后使用 nearest resize 回到训练尺寸。
- **对 DVG-B1 的含义：** 操作对象是训练 ground-truth depth 的稀疏度课程，不是模型输入的 corruption mask，也没有输出 token reliability。

#### 6.5.3 NR-MVSNet [9]

- `D:/0Project/origin/NR-MVSNet/models/refinenet.py::DepthUpdate.forward`
- 关键 shape：`cost_volume [B,C,D,H,W]`、`prob_volume [B,D,H,W]`，entropy 在深度候选维聚合。
- 代码执行：`mask = (1 - entropy) > 0.3`，随后 `sum(cost_volume * prob_volume.unsqueeze(1), dim=2) * mask`。
- **对 DVG-B1 的含义：** gate 是当前位置的 `[B,1,H,W]` hard query/location mask，不涉及另一 token，不能映射成 Full/H/W pairwise gate。

#### 6.5.4 LFDA [10]

- `D:/0Project/origin/LFDA/LFDA.py::Depth_MCA.forward`
- `query` shape 为 `[B,d,N_q,C]`，key/value 来自 center view；计算 `q @ k.transpose(-2,-1)`，执行 `softmax(dim=-2)`，再 `out = query * attn`。
- **对 DVG-B1 的含义：** 它在 depth/angular candidate 轴做 attention，不是在空间 token 轴构造 `[B,L,L]` reliability gate。

#### 6.5.5 SMAC 与 S2MA [11]

- `D:/0Project/origin/SMAC`：README 与仓库内容只覆盖 ReDWeb-S 数据集、统计和结果，没有 SMAC 网络 forward。
- `D:/0Project/origin/S2MA/ImageDepthNet/ImageDepthNet.py::NonLocalBlock`：`theta_x @ phi_x` 形成 `[B,HW,HW]` Full affinity，`softmax(alpha*f + self_f, dim=-1)` 后与 value 相乘。
- `ImageDepthNet.forward` 中 `alphaD/alphaR` 先为 `[B,HW,1]`，再扩展到 `[B,HW,HW]`，表示逐 query 位置权重，不是单个 image-wise scalar。
- **对 DVG-B1 的含义：** S2MA 可以校验 Full affinity 的 shape 和 row-wise softmax 拓扑，但不能当作 SMAC 官方模型实现，更不能提供 corruption reliability gate。

#### 6.5.6 OPM-MVS [5]

- `D:/0Project/origin/OPM-MVS/OCMM.cu`：包含 `ComputeDualBilateralWeight`、bilateral NCC、view weights 及其归一化。
- `D:/0Project/origin/OPM-MVS/main.cpp::JointBilateralUpsampling`：执行 coarse-to-fine joint bilateral upsampling。
- **对 DVG-B1 的含义：** 这些 confidence 评估多视图匹配和假设一致性，没有可直接映射为“corruption reliability × pairwise attention”的统一实现。

### 6.6 自行快速 CPU 对比实验：只验证操作语义

> **标记：自行实验，非论文结果、非模型性能实验。** 本节使用内联 Python 和 CPU PyTorch，没有创建 `test_*.py` 或修改仓库。首次导入 Torch 遇到本机重复 OpenMP runtime；仅对该进程设置 `KMP_DUPLICATE_LIB_OK=TRUE` 后运行。实验未加载 DFormer checkpoint，也未执行模型 forward。

#### 6.6.1 mask/reliability 聚合

构造 `[1,1,5,7]` reliability，坏点坐标为 `(0,1)`、`(2,3)`、`(4,5)`，缩放到 `3×4`：

- nearest 只保留一个 0，另外两个坏点被采样遗漏；输出仍是离散 0/1。
- bilinear 输出最小值约 `0.75`，出现 `0.875` 等连续值。
- area 输出最小值 `0.75`，出现约 `0.8333`、`0.8889` 等有效面积比例。
- 在该 `5×7 -> 3×4` 的精确样例上，bilinear 与 area 和水平翻转交换；nearest 不交换。这说明 nearest 不仅可能漏掉坏点，还可能因源/目标网格取样位置在 flip view 中产生不同结果。

再构造 `3×3` reliability，中心一个坏点；先把右侧和底部 padding 为可信，再做 `2×2, stride=2`：

- average/valid fraction：受影响 patch 为 `0.75`；
- min/all-valid：受影响 patch 为 `0`；
- max/any-valid：四个 patch 全为 `1`；
- nconv 代码同形态的 `maxpool/4`：四个 patch 全为 `0.25`；
- “先右/下可信 padding 再 average pool”与水平翻转不交换，说明 evaluator 的单边 padding 会带来坐标不对称风险。

用全 1 的 `5×5` 卷积核、padding 2 复现 `NConv2d` confidence 更新：

- 全 1 输入的输出范围约为 `0.36–1.00`，边缘因零 padding 低于 1；
- 全 0 输入为 0；
- 仅中心一个坏点时范围约为 `0.32–0.96`。

**能说明什么：** nearest、area、bilinear、all-valid、any-valid、valid fraction 与 nconv 的 `maxpool/4` 是不同操作，padding/flip 顺序也是协议的一部分。

**不能说明什么：** 这些输出没有模型指标，不能据此选择某个算子“更科学”或“效果更好”。

#### 6.6.2 pairwise gate

令连续 token reliability 为 $r=[1,0.5,0,1]$，矩阵轴固定为 row=query、column=key。比较：

$$
G_{\mathrm{query}}(i,j)=r_i,
\qquad
G_{\mathrm{key}}(i,j)=r_j,
$$

$$
G_{\mathrm{product}}(i,j)=r_i r_j,
\qquad
G_{\min}(i,j)=\min(r_i,r_j).
$$

CPU 结果：

- query-only 与 key-only 都非对称，分别缩放整行与整列；该连续样例中各有 8 个矩阵元素小于 1。
- product 与 min 都对称；各有 12 个矩阵元素小于 1。
- reliability 为 `0.5` 的 token 自配对时，product 为 `0.25`，min 为 `0.5`，说明两者对部分可信 token 的衰减强度不同。
- 若只有一个坏 token，即 $r=[1,1,0,1]$，query-only/key-only 分别影响一行/一列共 4 个元素；product/min 影响该行与该列，共 7 个唯一 pair。
- 对 `[B,H,W]=[1,2,3]`，product gate 的可实现 shape 分别为 Full `[1,1,6,6]`、H `[1,1,3,2,2]`、W `[1,1,2,3,3]`；全 1 reliability 在三种 shape 下均生成全 1 gate。

**能说明什么：** 该 probe 可提前排除 row/column 方向、广播 shape、对称性与 all-1 gate 构造错误。

**不能说明什么：** shape 正确和全 1 gate 正确并不能证明最终模型 `torch.equal` no-op，也不能替文献或项目预注册选择 query/key/product/min。

### 6.7 官方代码核验后的门禁判断与用户选择

- **A 的参考检索已关闭，项目预注册候选已选。** [1] 的连续 confidence 传播是最强直接机制证据，但 `maxpool/4` 与零 padding 违反 strict all-1；其余代码也没有给出本项目五尺度与四级规则。用户因此选择：原始 `m=1` 为受损、`r=1-m` 为可信；先按 evaluator 用 `INTER_LINEAR` 映射到每个 view，再 flip、右/下 padding `r=1`，随后对四级实际 token 网格用 `INTER_AREA` 得到连续有效面积比例。partial/empty/all-1 的严格语义以 `04-DVG-B1条件式Oracle门控.md` 第 6.4 节为准。
- **B 的参考检索已关闭，项目预注册候选已选。** [9] 的 query-only hard gate、[10] 的 candidate attention、S2MA Full affinity 和 [5] 的多视图 confidence 互不等价。用户选择 query/key 两端 continuous product，同时构造 Full `[B,1,L,L]`、H `[B,1,W,H,H]` 和 W `[B,1,H,W,W]` gate；只乘 depth geometry contribution。严格公式以 04 计划第 6.6 节为准。
- **证据边界不变。** A/B 是看模型结果前形成的项目设计，不是 [1]–[11] 的唯一结论；CPU probe 只帮助排除方向、shape、padding 和 all-1 错误，不能证明性能更好。
- **C 仍是唯一 `reference-blocked` 项。** clean 使用原路径旁路和 `torch.equal` 严格零差异，指标集合固定为 Boundary IoU + mIoU；`oracle-supported` 的最小实际效应量、95% 区间规则和 mIoU 裁决职责仍需用户项目预注册选择。
- **准确恢复点：** 不再继续一般性的 hierarchical reliability/decomposed attention 窄搜。新对话先关闭 C，再物化 protocol；之后按 CPU qualification → no-op → 1–2 样本 preflight 推进，完整 GPU 评价另行授权。

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

1. 不再重复获取或通读 11 篇全文，不再重复核对已固定的 8 个常见仓库，也不再继续一般性的 A/B 窄搜；如未来出现直接覆盖同一实现问题的新证据，只能作为 protocol 形成前的勘误，不能按模型结果触发搜索。
2. A 按 04 计划第 6.4 节物化：view 级 `INTER_LINEAR`、resize 后 flip、右/下 padding `r=1`、stage 级 `INTER_AREA` 连续有效面积比例，并显式审计 partial/empty/all-1 与单边 padding 的 flip 不对称。
3. B 按 04 计划第 6.6 节物化：Full/H/W 使用 query/key 两端 reliability 对称乘积，只乘 `self.weight[1] * mask_d*`，不得结果后改为 min、query-only、key-only、hard gate 或 stage 子集。
4. 新对话第一步关闭 C：clean 固定严格 `torch.equal` 零差异，指标固定 Boundary IoU + mIoU；用户仍需填写最小实际效应量、95% 区间规则和 mIoU 裁决职责。C 未关闭时不创建最终 B1 protocol。
5. C 关闭后按 04 计划 P0–P4 顺序执行：protocol 物化 → CPU qualification → no-op → 1–2 样本 preflight → 另行授权的完整 paired development evaluation。本文不授权模型代码、模型 forward、GPU、训练、云资源或 official test。

## 9. 参考文献

[1] A. Eldesokey, M. Felsberg, and F. S. Khan, “Confidence Propagation through CNNs for Guided Sparse Depth Regression,” *IEEE Transactions on Pattern Analysis and Machine Intelligence*, vol. 42, no. 10, pp. 2423–2436, 2020, doi: `10.1109/TPAMI.2019.2929170`.

[2] H. Chen, Y. Wang, S. Jia, X. Jin, and J. Zhang, “BCAP-Net: A Bidirectional Cross-Fusion and Adaptive Depth Propagation Network for Depth Completion,” *International Journal of Machine Learning and Cybernetics*, vol. 17, no. 7, article 324, 2026, doi: `10.1007/s13042-026-03156-8`.

[3] F. Karimi, A. Mehrpanah, and R. Rawassizadeh, “LightDepth: A Resource Efficient Depth Estimation Approach for Dealing with Ground Truth Sparsity via Curriculum Learning,” *Robotics and Autonomous Systems*, vol. 181, article 104784, 2024, doi: `10.1016/j.robot.2024.104784`.

[4] M. Wan, Y. Chen, P. Ge, X. Kong, G. Gu, and Q. Chen, “Image-Guided Dense Depth Completion Network Based on Hierarchical Feature Reconstruction and Dynamic Weight Sampling,” *Optics and Lasers in Engineering*, vol. 195, article 109296, 2025, doi: `10.1016/j.optlaseng.2025.109296`.

[5] R. L. Khuboni and H. Xu, “Octagram Propagation Matching for Multi-Scale View Stereopsis (OPM-MVS),” *IEEE Access*, vol. 13, pp. 86203–86217, 2025, doi: `10.1109/ACCESS.2025.3569913`.

[6] H. Zhang and J. Huo, “Non-Local Affinity Adaptive Acceleration Propagation Network for Generating Dense Depth Maps from LiDAR,” *Optics Express*, vol. 31, no. 13, starting at p. 22012, 2023, doi: `10.1364/OE.492187`.

[7] W. Zhao, C. Jung, and J. Kim, “Deep Sparse Depth Completion Using Multi-Affinity Matrix,” *IEEE Access*, vol. 11, pp. 78251–78261, 2023, doi: `10.1109/ACCESS.2023.3295133`.

[8] H. Guo, J. Li, and H. Liu, “Gaussian Splatting Confidence Supervision for SPN-Based Depth Completion,” *Pattern Recognition*, article 114460, 2026, doi: `10.1016/j.patcog.2026.114460`.

[9] J. Li, Z. Lu, Y. Wang, J. Xiao, and Y. Wang, “NR-MVSNet: Learning Multi-View Stereo Based on Normal Consistency and Depth Refinement,” *IEEE Transactions on Image Processing*, vol. 32, pp. 2649–2662, 2023, doi: `10.1109/TIP.2023.3272170`.

[10] H. Kim, S. Han, and Y. Kim, “LFDA: A Framework for Light Field Depth Estimation With Depth Attention,” *IEEE Access*, vol. 12, pp. 65032–65040, 2024, doi: `10.1109/ACCESS.2024.3393576`.

[11] N. Liu, N. Zhang, L. Shao, and J. Han, “Learning Selective Mutual Attention and Contrast for RGB-D Saliency Detection,” *IEEE Transactions on Pattern Analysis and Machine Intelligence*, vol. 44, no. 12, pp. 9026–9042, 2022, doi: `10.1109/TPAMI.2021.3122139`.

