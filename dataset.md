
# MUSeg 数据集基础说明与使用指南

## 1. 数据集概况

**MUSeg（Multimodal Semantic Segmentation Dataset）**面向复杂地下矿山场景，核心任务是**RGB-D多模态语义分割**。

数据集默认放置在仓库外部；运行训练时通过数据集配置指定实际路径。

| 项目       | 内容                                               |
| ---------- | -------------------------------------------------- |
| 数据模态   | 对齐的RGB图像、Depth深度图、像素级语义标签         |
| 数据规模   | **3,171组RGB-D数据对**                       |
| 数据分组   | **1,916个采集组**                            |
| 采集地点   | 6座矿山，包括5座煤矿和1座金矿                      |
| 图像分辨率 | $1082\times932$                                  |
| RGB格式    | JPG                                                |
| Depth格式  | PNG，像素包含实际距离信息                          |
| 标签格式   | PNG彩色标签、PNG类别ID标签、Labelme JSON多边形标注 |
| 语义类别   | 15个矿下目标类别                                   |
| 主要任务   | RGB、Depth或RGB-D语义分割                          |
| 发布方式   | Figshare公开发布                                   |

过滤后的各矿井数据量如下：

| 矿井目录       |        采集组数 |   RGB-D数据对数 |
| -------------- | --------------: | --------------: |
| Coal Mine No.1 |             609 |             754 |
| Coal Mine No.2 |             361 |             481 |
| Coal Mine No.3 |             434 |             549 |
| Coal Mine No.4 |              92 |             417 |
| Coal Mine No.5 |              66 |             115 |
| Gold Mine No.6 |             354 |             855 |
| **总计** | **1,916** | **3,171** |

这里的“采集组”和“图像对”不是同一概念：同一组可能包含多个视角或相近样本。因此，划分训练集和测试集时应尽可能**按组划分**，而不是把所有图像随机打散，以免高度相似的图像同时进入训练集与测试集。

---

## 2. 数据集的基本目录结构

论文描述的目录结构可整理为：

```text
MUSeg/
├── Mine_01/                     # 1号煤矿
│   ├── Image/                   # RGB图像
│   ├── Depth/                   # 深度图
│   └── Label/                   # 语义分割标注
│       ├── *_color.png          # 彩色可视化标签
│       ├── *_label.png          # 像素类别ID标签
│       └── *_polygons.json      # Labelme多边形标注
│
├── Mine_02/
│   ├── Image/
│   ├── Depth/
│   └── Label/
│
├── Mine_03/
│   ├── Image/
│   ├── Depth/
│   └── Label/
│
├── Mine_04/
│   ├── Image/
│   ├── Depth/
│   └── Label/
│
├── Mine_05/
│   ├── Image/
│   ├── Depth/
│   └── Label/
│
├── Mine_06/                     # 金矿
│   ├── Image/
│   ├── Depth/
│   └── Label/
│
└── Experimental files/         # 实验配置、划分或相关实验文件
```

实际下载后的矿井文件夹名称可能与上面的示意名称略有差别，应以发布包中的名称为准；但根目录组织逻辑是**一个矿井一个子目录，每个矿井内部按数据类型组织**。

> The root directory contains 6 subfolders (one per mine, numbered as in Table 2) and experimental files. Under each mine subdirectory, data is stored by type: The Image folder stores RGB images (file format: JPG, resolution: 1082 × 932); The Depth folder stores depth images (file format: PNG, resolution: 1082 × 932, each pixel contains actual distance information, specific technical details refer to the oficial Microsoft Azure Kinect DK documentation).
>
> (Li 等, 2025)

这段说明确定了三个关键事实：

1. 数据按矿井划分，适合做跨矿井泛化实验；
2. RGB与Depth分开存储，需要通过文件名建立对应关系；
3. Depth PNG保存的是距离数据，不应直接当作普通8位灰度图读取。

---

# 3. 各目录的作用和内容

## 3.1 `Image/`：RGB图像

### 内容

```text
Image/
├── sample_001.jpg
├── sample_002.jpg
└── ...
```

- 格式：JPG；
- 分辨率：$1082\times932$；
- 内容：地下矿井场景的彩色图像；
- 用途：提供纹理、颜色、亮度、物体外观和边缘等信息。

### 在模型中的使用方式

RGB图像通常读取为：

$$
I_{\mathrm{rgb}}\in\mathbb{R}^{3\times H\times W}
$$

典型预处理：

1. BGR转RGB——如果使用OpenCV；
2. 缩放至模型输入尺寸；
3. 转为浮点数；
4. 除以255；
5. 使用ImageNet均值和标准差归一化。

```python
rgb = cv2.imread(rgb_path, cv2.IMREAD_COLOR)
rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
rgb = cv2.resize(rgb, (input_w, input_h))
rgb = rgb.astype(np.float32) / 255.0

mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
rgb = (rgb - mean) / std
```

如果模型使用ImageNet预训练的RGB骨干网络，这种归一化通常是较稳妥的起点。

---

## 3.2 `Depth/`：深度图

### 内容

```text
Depth/
├── sample_001.png
├── sample_002.png
└── ...
```

- 格式：PNG；
- 分辨率：$1082\times932$；
- 每个像素记录对应位置的距离信息；
- 与同名RGB图像对齐；
- 数据来自Microsoft Azure Kinect DK体系。

深度图可表示为：

$$
D\in\mathbb{R}^{1\times H\times W}
$$

其中 $D(u,v)$ 表示RGB像素位置 $(u,v)$ 对应的深度或距离。

### 必须注意：不要默认按8位图读取

论文中的深度可视化图是把原始深度归一化后显示的，但原始文件包含实际距离信息。因此不应使用会强制转成8位的读取方式。

推荐：

```python
depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)

print(depth.dtype)
print(depth.shape)
print(depth.min(), depth.max())
```

常见情况下，深度PNG可能是16位整数：

```python
assert depth.dtype == np.uint16
```

如果直接写成：

```python
depth = cv2.imread(depth_path)
```

OpenCV可能把它读取为三通道8位图，导致距离精度丢失。

### 深度值处理

先检查发布说明中给出的深度单位。若像素值以毫米为单位，可转换为米：

```python
depth_m = depth.astype(np.float32) / 1000.0
```

但不要在没有确认单位前直接假设缩放系数。论文建议参考Azure Kinect DK的技术说明，因此应结合数据包README和样本数值范围进行确认。

### 无效深度

深度相机在粉尘、反光、遮挡或超量程区域可能产生无效值。通常可建立有效性掩码：

```python
valid_mask = depth > 0
```

然后进行截断和归一化：

```python
depth_m = depth.astype(np.float32) * depth_scale

min_depth = 0.2
max_depth = 10.0

valid_mask = (
    np.isfinite(depth_m)
    & (depth_m >= min_depth)
    & (depth_m <= max_depth)
)

depth_norm = np.zeros_like(depth_m, dtype=np.float32)
depth_norm[valid_mask] = (
    depth_m[valid_mask] - min_depth
) / (max_depth - min_depth)

depth_norm = np.clip(depth_norm, 0.0, 1.0)
```

其中 `depth_scale`、`min_depth` 和 `max_depth` 应根据数据的单位、设备量程和样本统计确定。

### 建议把有效性掩码输入模型

单纯把无效深度设置为0，会让模型难以区分：

- “实际距离很近”；
- “这个位置没有有效深度”。

因此可以把输入组织为：

$$
X_d=[D_{\mathrm{norm}},M_{\mathrm{valid}}]
$$

即深度值加一个有效性通道：

```python
depth_input = np.stack(
    [depth_norm, valid_mask.astype(np.float32)],
    axis=0
)
```

也可以让深度分支只输入一通道，把有效性掩码用于注意力或损失加权。

---

## 3.3 `Label/`：语义分割标签

Label目录包含三种表达形式，但它们的作用不同。

> The Label folder stores multi-category annotation files, including: Colored Label with sufix ‘\_color’ (file format: PNG, resolution: 1082 × 932), labels with sufix ‘\_label (file format: PNG, resolution: $1 0 8 2 \times 9 3 2$ , each pixel’s value represents the corresponding category), annotation files with sufix ‘\_polygons’ (file format: JSON, following Labelme standard).
>
> (Li 等, 2025)

---

### A. `*_label.png`：模型训练用类别ID标签

这是训练语义分割模型时最重要的文件。

```text
Label/
├── sample_001_label.png
├── sample_002_label.png
└── ...
```

每个像素的数值表示一个语义类别：

$$
Y(u,v)\in\{0,1,\ldots,C-1\}
$$

其中 $C$ 是模型输出类别数。论文描述了15个目标类别，但实际训练前需要确认：

- 背景是否单独占用一个ID；
- 15类是否包含背景；
- 是否存在 `ignore` 类；
- `ignore_index` 是255还是其他数值；
- 标签ID是否连续。

建议首先检查标签中的全部唯一值：

```python
from pathlib import Path
import cv2
import numpy as np

all_ids = set()

for path in Path(dataset_root).rglob("*_label.png"):
    label = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    all_ids.update(np.unique(label).tolist())

print("Label IDs:", sorted(all_ids))
```

必须使用：

```python
cv2.IMREAD_UNCHANGED
```

并保留原始整数ID。不要对类别标签做归一化。

#### 标签缩放必须用最近邻插值

```python
label = cv2.resize(
    label,
    (input_w, input_h),
    interpolation=cv2.INTER_NEAREST
)
```

不能使用双线性或双三次插值，否则会生成不存在的类别ID。

---

### B. `*_color.png`：标签可视化文件

```text
Label/
├── sample_001_color.png
├── sample_002_color.png
└── ...
```

它通过不同颜色显示不同语义类别，主要用于：

- 人工查看标注；
- 数据质量检查；
- 预测结果对比；
- 论文和报告可视化；
- 验证类别ID与颜色映射关系。

通常**不要把彩色标签直接用于监督训练**。JPEG/PNG读写、颜色空间转换或压缩都可能破坏颜色到类别ID的稳定对应关系。训练时优先使用 `*_label.png`。

---

### C. `*_polygons.json`：Labelme多边形标注

```text
Label/
├── sample_001_polygons.json
├── sample_002_polygons.json
└── ...
```

JSON文件遵循Labelme标准，通常包含：

- 图像尺寸；
- 多边形顶点；
- 类别名称；
- 图像对应关系；
- 标注形状等信息。

典型结构可能类似：

```json
{
  "imageHeight": 932,
  "imageWidth": 1082,
  "shapes": [
    {
      "label": "Person",
      "points": [[x1, y1], [x2, y2], [x3, y3]],
      "shape_type": "polygon"
    }
  ]
}
```

JSON适合用于：

- 修改或补充标注；
- 合并类别；
- 生成新的类别ID标签；
- 转换为COCO格式；
- 生成实例级多边形；
- 对特定类别重新定义；
- 检查类别名称与ID的映射。

如果只是训练标准语义分割模型，直接读取 `*_label.png` 即可，不需要每轮训练时解析JSON。

---

## 3.4 `Experimental files/`：实验相关文件

论文说明根目录中包含实验文件，但当前目录描述没有进一步列出其中每个文件的确切名称。实际使用时应优先检查它是否包含：

- 官方训练集/测试集划分；
- 类别名称与类别ID映射；
- 颜色映射表；
- 配置文件；
- 基线模型代码或参数；
- 数据列表文件；
- 评价脚本。

使用原则是：

1. **如果有官方划分，优先按官方划分复现实验**；
2. **如果有类别映射文件，以官方映射为准**；
3. 不要仅凭彩色标签自行猜测类别ID；
4. 在做新划分前保留官方测试集，便于与论文结果对比。

---

# 4. 文件如何建立一一对应关系

RGB、Depth和Label通常共享同一个样本主文件名，只是所在目录和后缀不同。

示意：

```text
Image/ABC.jpg
Depth/ABC.png
Label/ABC_label.png
Label/ABC_color.png
Label/ABC_polygons.json
```

可将 `ABC` 作为样本ID：

```python
sample = {
    "rgb":     "Image/ABC.jpg",
    "depth":   "Depth/ABC.png",
    "label":   "Label/ABC_label.png",
    "color":   "Label/ABC_color.png",
    "polygon": "Label/ABC_polygons.json",
}
```

不要依赖三个目录中 `os.listdir()` 的返回顺序，因为不同文件系统的顺序可能不一致。应使用**主文件名匹配**。

---

## 文件名包含的信息

论文给出的命名字段包括：

- 矿井编号；
- 采集设备；
- 数据组编号；
- 采集时间戳；
- RGB亮度等级；
- 保留扩展字段；
- 文件扩展名。

其中：

- 矿井编号：`01–06`；
- 数据组编号：对应采集分组策略；
- 时间戳：采集时间；
- RGB亮度等级：可用于按照度分析；
- 扩展字段：为未来扩展预留。

文件名解析可用于构造元数据：

```python
metadata = {
    "mine_id": "...",
    "device_id": "...",
    "group_id": "...",
    "timestamp": "...",
    "brightness_level": "..."
}
```

由于原文中的命名格式排版存在一定混杂，实际编写解析器时应以下载数据中的真实文件名为准。不要在没有检查样本文件名之前把字段位置硬编码。

---

# 5. 在新模型中如何组织输入和输出

## 5.1 推荐的数据样本结构

每个训练样本至少返回：

```python
{
    "rgb": rgb_tensor,             # [3, H, W], float32
    "depth": depth_tensor,         # [1, H, W], float32
    "depth_valid": valid_tensor,   # [1, H, W], float32/bool
    "label": label_tensor,         # [H, W], int64
    "sample_id": sample_id,
    "mine_id": mine_id,
    "group_id": group_id,
}
```

其张量形式为：

| 数据       | 推荐形状              | 类型                  | 说明                 |
| ---------- | --------------------- | --------------------- | -------------------- |
| RGB        | $3\times H\times W$ | `float32`           | 归一化后输入         |
| Depth      | $1\times H\times W$ | `float32`           | 截断、归一化后的深度 |
| Valid mask | $1\times H\times W$ | `float32`或`bool` | 深度有效区域         |
| Label      | $H\times W$         | `int64`             | 每个像素的类别ID     |

模型输出为：

$$
Z=f_\theta(I_{\mathrm{rgb}},D)
\in\mathbb{R}^{C\times H\times W}
$$

其中 $C$ 为最终确认的训练类别数。

预测类别为：

$$
\hat Y(u,v)=\arg\max_c Z_c(u,v)
$$

---

## 5.2 三种基础使用方案

### 方案一：RGB单模态基线

```text
RGB → RGB Encoder → Decoder → Segmentation
```

用途：

- 检查数据管线是否正确；
- 与普通语义分割方法比较；
- 衡量加入Depth后的实际收益。

论文中的参考结果：

| 模型             | 输入 |   mIoU |
| ---------------- | ---- | -----: |
| DeepLabV3+       | RGB  | 43.64% |
| SegFormer MiT-B2 | RGB  | 47.93% |

---

### 方案二：四通道早期融合

将RGB和Depth直接拼接：

$$
X=\operatorname{Concat}(I_{\mathrm{rgb}},D)
\in\mathbb{R}^{4\times H\times W}
$$

```python
x = torch.cat([rgb, depth], dim=1)
logits = model(x)
```

优点：

- 实现简单；
- 可快速验证Depth是否有效；
- 计算量较小。

缺点：

- RGB和Depth的统计分布不同；
- 浅层直接混合可能降低预训练权重的利用率；
- 难以处理某个模态失效；
- 不容易显式建模模态可靠性。

如果第一层卷积原本接收3通道，需要改为4通道：

```python
old_conv = model.backbone.patch_embed.proj

new_conv = nn.Conv2d(
    4,
    old_conv.out_channels,
    kernel_size=old_conv.kernel_size,
    stride=old_conv.stride,
    padding=old_conv.padding,
    bias=old_conv.bias is not None
)
```

可用RGB卷积权重的均值初始化第4通道：

```python
with torch.no_grad():
    new_conv.weight[:, :3] = old_conv.weight
    new_conv.weight[:, 3:4] = old_conv.weight.mean(dim=1, keepdim=True)
```

---

### 方案三：双编码器中间融合

这是更推荐的新模型框架：

```text
RGB   → RGB Encoder   ─┐
                       ├→ Multimodal Fusion → Decoder → Label
Depth → Depth Encoder ─┘
```

数学表示：

$$
F_r^l=E_r^l(I_{\mathrm{rgb}})
$$

$$
F_d^l=E_d^l(D,M_{\mathrm{valid}})
$$

$$
F_f^l=\Phi^l(F_r^l,F_d^l)
$$

其中：

- $E_r$：RGB编码器；
- $E_d$：Depth编码器；
- $\Phi$：融合模块；
- $l$：特征尺度。

融合方式可以从简单到复杂依次实验：

1. 特征相加；
2. 通道拼接；
3. 门控融合；
4. 交叉注意力；
5. 质量或可靠性感知融合。

例如门控融合：

$$
G=\sigma\left(\operatorname{Conv}
\left([F_r,F_d,M_{\mathrm{valid}}]\right)\right)
$$

$$
F_f=G\odot F_r+(1-G)\odot F_d
$$

这种结构能让模型根据区域情况动态选择RGB或Depth，对低照度和深度空洞更实用。

论文基线显示RGB-D方法整体优于RGB基线：

| 模型      | 输入         |    PA |   MPA |            mIoU |
| --------- | ------------ | ----: | ----: | --------------: |
| SegFormer | RGB          | 78.75 | 58.54 |           47.93 |
| SegFormer | Depth（HHA） | 81.16 | 62.21 |           51.28 |
| SegFormer | RGB-D（HHA） | 83.00 | 65.25 |           55.20 |
| CMNeXt    | RGB-D（HHA） | 84.05 | 66.77 |           56.06 |
| SA-Gate   | RGB-D（HHA） | 84.77 | 71.34 |           59.38 |
| DFormer   | RGB-D        | 85.63 | 70.37 |           59.74 |
| CMX       | RGB-D（HHA） | 86.30 | 72.72 | **61.83** |

因此，新模型至少应同时设置RGB-only和RGB-D两个版本，否则难以说明性能提升究竟来自模型结构还是增加了深度信息。

---

# 6. Depth原始值与HHA编码

论文基线中部分模型使用了`Depth (HHA)`或`RGB-D (HHA)`。HHA通常把原始深度转换为三通道几何表达，包括：

1. 水平视差；
2. 距离地面的高度；
3. 局部表面法向与重力方向的夹角。

这样可以让深度输入适配原本为三通道图像设计的编码器。

但矿下场景未必存在稳定、平整的地面和可靠重力参考，因此建议同时比较：

| Depth表达      | 通道数 | 优点              | 风险                      |
| -------------- | -----: | ----------------- | ------------------------- |
| 原始归一化深度 |      1 | 简单、保留距离    | 与RGB预训练网络不完全匹配 |
| 深度+有效掩码  |      2 | 显式表示缺失值    | 需要修改输入层            |
| 复制深度       |      3 | 可直接使用RGB骨干 | 三通道信息重复            |
| HHA            |      3 | 引入几何含义      | 依赖相机参数和几何估计    |
| 深度+法向量    |      4 | 强化局部形状      | 对深度噪声敏感            |

如果你的创新重点是多模态融合，建议至少报告：

- RGB；
- 原始Depth；
- RGB+原始Depth；
- RGB+HHA。

否则难以判断收益来自融合模块还是Depth编码方式。

---

# 7. 数据划分建议

## 7.1 复现论文时

优先使用数据包中的官方实验文件和官方划分。这样得到的PA、MPA和mIoU才能与论文表5直接比较。

## 7.2 开发新模型时

由于数据中存在采集组，应避免同组的近似样本泄漏：

```text
正确：
group_001的全部样本 → Train
group_002的全部样本 → Val
group_003的全部样本 → Test

不推荐：
group_001/image_1 → Train
group_001/image_2 → Test
```

建议建立以下三套协议：

### 标准协议

使用官方训练/测试划分，用于与已有模型比较。

### 按组划分协议

```text
Train / Validation / Test = 70% / 10% / 20%
```

但必须以`group_id`为划分单位。

### 跨矿井协议

例如：

```text
Mine 01–05 → Train
Mine 06    → Test
```

或者进行Leave-One-Mine-Out：

$$
\mathcal D_{\mathrm{train}}
=\bigcup_{i\neq k}\mathcal D_i,
\qquad
\mathcal D_{\mathrm{test}}=\mathcal D_k
$$

跨矿井协议更能衡量新模型在真实部署中的泛化能力。

---

# 8. 数据增强必须保持多模态同步

RGB、Depth和Label在空间上对齐，因此几何增强必须使用完全相同的随机参数：

- 随机裁剪；
- 水平翻转；
- 缩放；
- 旋转；
- 仿射变换。

即：

$$
T_{\mathrm{geo}}(I_{\mathrm{rgb}}),
\quad
T_{\mathrm{geo}}(D),
\quad
T_{\mathrm{geo}}(Y)
$$

必须共享同一个 $T_{\mathrm{geo}}$。

标签使用最近邻插值；RGB通常使用双线性插值；深度最好根据变换类型谨慎选择，并同步更新有效性掩码。

```python
rgb, depth, label = synchronized_random_crop(
    rgb, depth, label
)

if random.random() < 0.5:
    rgb = np.flip(rgb, axis=1).copy()
    depth = np.flip(depth, axis=1).copy()
    label = np.flip(label, axis=1).copy()
```

RGB光度增强则不应作用于Depth和Label：

- 亮度；
- 对比度；
- 色偏；
- Gamma；
- RGB噪声；
- 模拟灯光过曝。

Depth可单独进行：

- 随机深度空洞；
- 高斯噪声；
- 区域遮挡；
- 随机模态丢弃；
- 距离相关噪声。

MUSeg有意保留了部分模态信息缺失的样本：

> We retained such RGB-D cases in the dataset to support robustness research for multimodal fusion models under partial modality missing conditions.
>
> (Li 等, 2025)

因此，新模型可以把“模态退化鲁棒性”作为正式训练目标，而不是简单删除所有质量不佳的样本。

---

# 9. 一个可直接改造的PyTorch数据集骨架

```python
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


class MUSegDataset(Dataset):
    def __init__(
        self,
        records,
        image_size=(512, 512),
        depth_scale=1.0,
        min_depth=0.0,
        max_depth=None,
        transform=None,
    ):
        """
        records中的每个元素：
        {
            "rgb": Path,
            "depth": Path,
            "label": Path,
            "sample_id": str,
            "mine_id": str,
            "group_id": str
        }
        """
        self.records = records
        self.image_size = image_size
        self.depth_scale = depth_scale
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.transform = transform

        self.rgb_mean = np.array(
            [0.485, 0.456, 0.406], dtype=np.float32
        )
        self.rgb_std = np.array(
            [0.229, 0.224, 0.225], dtype=np.float32
        )

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        item = self.records[index]

        # RGB
        rgb = cv2.imread(str(item["rgb"]), cv2.IMREAD_COLOR)
        if rgb is None:
            raise FileNotFoundError(item["rgb"])
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)

        # Depth：保留原始位深
        depth = cv2.imread(
            str(item["depth"]),
            cv2.IMREAD_UNCHANGED
        )
        if depth is None:
            raise FileNotFoundError(item["depth"])

        # Label：保留整数类别ID
        label = cv2.imread(
            str(item["label"]),
            cv2.IMREAD_UNCHANGED
        )
        if label is None:
            raise FileNotFoundError(item["label"])

        # 防止标签被错误读取成多通道
        if label.ndim == 3:
            raise ValueError(
                f"Expected an ID label, but got shape {label.shape}: "
                f"{item['label']}"
            )

        depth = depth.astype(np.float32) * self.depth_scale

        valid = np.isfinite(depth) & (depth > self.min_depth)
        if self.max_depth is not None:
            valid &= depth <= self.max_depth

        # 几何增强必须同步作用于三种数据
        if self.transform is not None:
            rgb, depth, valid, label = self.transform(
                rgb, depth, valid, label
            )

        target_h, target_w = self.image_size

        rgb = cv2.resize(
            rgb,
            (target_w, target_h),
            interpolation=cv2.INTER_LINEAR
        )
        depth = cv2.resize(
            depth,
            (target_w, target_h),
            interpolation=cv2.INTER_NEAREST
        )
        valid = cv2.resize(
            valid.astype(np.uint8),
            (target_w, target_h),
            interpolation=cv2.INTER_NEAREST
        ).astype(bool)
        label = cv2.resize(
            label,
            (target_w, target_h),
            interpolation=cv2.INTER_NEAREST
        )

        # RGB归一化
        rgb = rgb.astype(np.float32) / 255.0
        rgb = (rgb - self.rgb_mean) / self.rgb_std
        rgb = rgb.transpose(2, 0, 1)

        # Depth归一化
        depth_norm = np.zeros_like(depth, dtype=np.float32)

        if self.max_depth is not None:
            denom = max(self.max_depth - self.min_depth, 1e-6)
            depth_norm[valid] = (
                depth[valid] - self.min_depth
            ) / denom
            depth_norm = np.clip(depth_norm, 0.0, 1.0)
        else:
            # 没有确定量程时，建议在训练集上计算固定统计量，
            # 不建议最终版本逐图归一化。
            if np.any(valid):
                scale = np.percentile(depth[valid], 99)
                scale = max(scale, 1e-6)
                depth_norm[valid] = depth[valid] / scale
                depth_norm = np.clip(depth_norm, 0.0, 1.0)

        return {
            "rgb": torch.from_numpy(rgb).float(),
            "depth": torch.from_numpy(
                depth_norm[None]
            ).float(),
            "depth_valid": torch.from_numpy(
                valid[None]
            ).float(),
            "label": torch.from_numpy(
                label.astype(np.int64)
            ).long(),
            "sample_id": item["sample_id"],
            "mine_id": item["mine_id"],
            "group_id": item["group_id"],
        }
```

正式训练时，深度归一化参数最好从**训练集整体统计**得到，而不是每张图独立归一化。逐图归一化会丢失跨图像的绝对距离关系。

---

# 10. 损失函数和评价指标

## 基础损失

最简单的语义分割损失：

$$
\mathcal L_{\mathrm{CE}}
=
-\frac{1}{N}
\sum_{i=1}^{N}\log p_{i,y_i}
$$

```python
criterion = torch.nn.CrossEntropyLoss(
    ignore_index=ignore_index
)
```

对于类别不均衡，可以使用：

$$
\mathcal L
=
\mathcal L_{\mathrm{CE}}
+\lambda\mathcal L_{\mathrm{Dice}}
$$

也可以使用类别权重或Focal Loss，但应根据训练集类别像素统计计算权重。

## 推荐评价指标

至少报告：

- PA：Pixel Accuracy；
- MPA：Mean Pixel Accuracy；
- mIoU：Mean Intersection over Union；
- 每一类别的IoU；
- 每一类别的Precision和Recall。

论文显示数据存在明显的类别数量差异，例如Cable实例超过10,000，而部分类别仅有数百个实例。因此不能只看总体PA。总体PA可能主要反映大面积或高频类别的表现，而掩盖人员、门、救援设备等小类别的失败。

---

# 11. 建议的新模型实验矩阵

为了证明新模型有效，建议至少完成以下消融：

| 实验 |       RGB |         Depth | Valid mask | 融合模块   | 目的           |
| ---- | --------: | ------------: | ---------: | ---------- | -------------- |
| A    |        ✓ |               |            | 无         | RGB基线        |
| B    |           |            ✓ |       可选 | 无         | 深度基线       |
| C    |        ✓ |            ✓ |            | 拼接/相加  | 简单融合基线   |
| D    |        ✓ |            ✓ |         ✓ | 新融合模块 | 完整模型       |
| E    |        ✓ |     Depth置零 |         ✓ | 新融合模块 | 深度失效测试   |
| F    |   RGB置零 |            ✓ |         ✓ | 新融合模块 | RGB失效测试    |
| G    | 低亮度RGB |            ✓ |         ✓ | 新融合模块 | 黑暗鲁棒性     |
| H    |        ✓ |  局部深度空洞 |         ✓ | 新融合模块 | 深度缺失鲁棒性 |
| I    |        ✓ | 空间偏移Depth |         ✓ | 新融合模块 | 标定误差测试   |

核心比较包括：

$$
\Delta_{\mathrm{RGB-D}}
=
\mathrm{mIoU}_{\mathrm{RGB-D}}
-\mathrm{mIoU}_{\mathrm{RGB}}
$$

以及故障条件下的退化率：

$$
R_{\mathrm{drop}}
=
\frac{
\mathrm{mIoU}_{\mathrm{normal}}
-
\mathrm{mIoU}_{\mathrm{failure}}
}{
\mathrm{mIoU}_{\mathrm{normal}}
}
$$

这样可以区分“正常条件精度高”和“真实矿下环境鲁棒”这两个不同目标。

---

# 12. 开始训练前的检查清单

## 文件一致性

- [ ] 每张RGB都能找到同名Depth；
- [ ] 每张RGB都能找到对应的`*_label.png`；
- [ ] 三者原始分辨率一致；
- [ ] RGB、Depth和Label空间对齐；
- [ ] 没有按目录遍历顺序错误配对。

## 标签检查

- [ ] 枚举所有标签ID；
- [ ] 确认背景ID；
- [ ] 确认类别数；
- [ ] 确认`ignore_index`；
- [ ] 确认ID到类别名称的映射；
- [ ] 确认标签缩放使用最近邻插值。

## 深度检查

- [ ] 使用`IMREAD_UNCHANGED`读取；
- [ ] 确认数据位深；
- [ ] 确认距离单位；
- [ ] 统计0值和无效值比例；
- [ ] 选定固定的截断和归一化范围；
- [ ] 保留深度有效性掩码。

## 数据划分

- [ ] 优先保留官方划分；
- [ ] 自定义划分按数据组进行；
- [ ] 避免同组相似图像跨训练集和测试集；
- [ ] 单独设计跨矿井测试。

## 模型实验

- [ ] RGB-only基线；
- [ ] Depth-only基线；
- [ ] 简单RGB-D融合基线；
- [ ] 新模型；
- [ ] 逐类IoU；
- [ ] 模态缺失和低照度测试。

---

## 最推荐的使用方式

如果你正在设计一个新的矿下多模态分割模型，建议把MUSeg组织成以下训练链路：

```text
Image/*.jpg
    → RGB预处理
    → RGB Encoder ─────────┐
                           │
Depth/*.png                ├→ 可靠性感知融合
    → 深度值转换与归一化   │
    → Valid Mask           │
    → Depth Encoder ───────┘
                           ↓
                    Segmentation Decoder
                           ↓
                  与 *_label.png 计算损失
                           ↓
              PA / MPA / mIoU / per-class IoU
```

其中：

- `*_label.png`用于训练和评价；
- `*_color.png`用于结果可视化；
- `*_polygons.json`用于修改、转换和扩展标注；
- `Image/`和`Depth/`通过主文件名配对；
- 矿井编号和数据组编号用于无泄漏划分；
- Depth有效性掩码应进入模型或融合模块；
- 官方实验文件应优先用于复现论文基线。

最稳妥的第一版是：**RGB双分支编码器 + 原始深度/有效性掩码 + 多尺度融合 + 语义分割解码器**，并同时设置RGB-only、Depth-only和简单拼接三个基线。
