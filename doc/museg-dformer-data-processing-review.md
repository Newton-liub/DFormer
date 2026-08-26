# MUSeg 数据集接入 DFormer/DFormerv2 的处理评审与复现报告

> **历史数据审计：** 本文保留 2026-08-17 至 2026-08-19 的数据复现证据。文中“缺 MUSeg config、缺 development validation、全背景 loss 未处理”等状态后来已解决；仍未冻结的是 validation 尺寸、BGR/RGB 和无效深度/B2 口径。当前状态见 `doc/main/MUSeg-current-status.md`，未决项见 `doc/main/MUSeg-open-decisions.md`。

> 用途：组会汇报、数据处理评审、新项目复现  
> 审计日期：2026-08-17  
> 状态修订：2026-08-19（标签语义、转换脚本与本地重建状态）
> 审计对象：DFormer 仓库、MUSeg 3171 组数据、DFormerv2 原论文、MUSeg 数据集论文

## 1. 执行摘要

### 1.1 总结论

当前处理方向可分成两部分评价：

1. **把 MUSeg 原始 16-bit Depth 保存在 `Depth16/`，再通过固定的全数据集线性映射生成 8-bit `Depth/`，作为当前 DFormer 代码的输入：方向正确。**
2. **数据转换复现闭环已经完成，但训练闭环尚未完成。** 仓库已新增 `tools/prepare_museg.py`，本地已从官方六矿区原始目录重建 `MUSeg_DFormer`，并对 3171 组四模态、官方划分、标签 ID 和全部深度映射执行验证。当前仍缺 MUSeg 专属训练配置；无效深度和全背景样本还存在明确风险。

当前 8-bit 深度不是简单截低 8 位，也不是逐图归一化。对全部 3,197,712,504 个像素核验后，现有数据精确满足：

\[
D_8=\operatorname{round}\left(D_{16}\times\frac{255}{13932}\right)
\]

其中 `13932` 是当前全数据集的最大原始深度值。该映射 100% 复现现有 `Depth/`。

### 1.2 必须区分的结论边界

- **论文没有规定 DFormerv2 必须接收 8-bit 深度。** 8-bit 是当前仓库加载与归一化实现形成的接口约束，不是模型论文给出的唯一深度表示。
- **MUSeg 论文明确原始 Depth 是 16-bit PNG。** 论文中的 8-bit 深度只用于可视化，是由原始 16-bit 深度归一化得到的显示图，不是发布数据的物理格式。
- **为了不改当前 DFormer 数据链路并复用其预训练尺度，生成 8-bit 输入是合理的兼容方案。** 如果希望直接使用 16-bit 物理深度，则必须同步修改读取、归一化、几何先验尺度和训练策略，不能只替换目录。
- **HHA 不应直接代替 DFormerv2 的 `modal_x`。** 当前 DFormerv2 在前向中只取辅助模态第一个通道；把三通道 HHA 放进去会丢掉另外两个通道，并改变论文中的 raw-depth geometry prior 语义。

### 1.3 当前状态判定

- 已验证正确：数据数量、文件配对、原始深度保留、固定线性 16→8 映射、官方 train/test 样本与采集组无交集、标签 `0=background`、`1–15=15` 个前景类别，以及训练映射 `0→255`、`1–15→0–14`。
- 已完成工程化：确定性转换脚本、原子输出替换、`dataset_meta.json`、官方划分 SHA-256 记录、转换前源数据检查和转换后全量验证。
- 仍需补齐：MUSeg 训练配置、验证集策略、无效深度策略、全背景样本处理、验证尺寸策略、RGB/BGR 明示。
- 可选研究改进：有效深度 mask、直接 16-bit/float 深度输入、几何先验尺度校准、HHA 分支消融。

---

## 2. 论文依据

## 2.1 DFormerv2 原论文给出的约束

来源：`paper/25CVPR_RGBDSeg-CN.pdf`。

### 模型输入与几何先验

DFormerv2 面向 RGB-D 语义分割。与 DFormer v1 的双模态统一编码不同，DFormerv2 主要编码 RGB，Depth 不再经过独立深度编码器，而是作为几何先验参与注意力计算。

论文第 3 页第 3.1 节、公式 (1) 定义深度关系：

\[
D_{ij,i'j'}=|z_{ij}-z_{i'j'}|
\]

其中 \(z_{ij}\) 是 depth patch 的代表值。论文第 3 页公式 (2)、(4) 将深度距离与空间距离融合，并通过几何衰减调制注意力。第 5 页第 3.3 节说明多尺度使用 Depth 构造几何先验。

因此，如果深度做线性变换 \(z'=az+b\)：

\[
|z'_i-z'_j|=|a|\,|z_i-z_j|
\]

偏移 \(b\) 会抵消，但尺度 \(a\) 会直接改变几何先验强度。这说明：

- 深度的绝对偏移不重要；
- 深度比例、量化分辨率和无效值处理重要；
- 固定线性映射保留顺序与相对差异，但会改变整体几何尺度；
- 逐图 min-max 会让同一物理差值在不同图像中产生不同几何权重，不适合复现。

### 论文未规定的内容

DFormerv2 论文没有明确规定：

- Depth 文件必须是 8-bit 还是 16-bit；
- 深度单位是 m、mm 还是相对深度；
- Depth 的 min/max、无效值和归一化均值方差；
- 是否必须使用传感器物理深度。

论文第 5 页第 4.1 节还说明 ImageNet 预训练阶段的 Depth 可由深度估计模型产生，因此模型并不以“必须具有物理单位”为前提。但这不意味着任意缩放都等价，因为公式 (1)、(4) 直接依赖深度数值差。

### 训练预处理

DFormerv2 论文第 5 页第 4.1 节明确：

- 随机水平翻转；
- 随机缩放范围 0.5–1.75；
- NYUv2 训练尺寸 480×640；
- SUNRGBD 训练尺寸 480×480；
- Cross Entropy、AdamW、初始学习率 `6e-5`、poly 衰减；
- 多尺度加翻转推理。

论文没有给出 Depth 文件位深、插值方式或数值归一化细节，这些需要以代码实现为准。

## 2.2 MUSeg 数据集论文给出的约束

来源：`paper/Li 等 - 2025 - 1-MUSeg A multimodal semantic segmentation dataset for complex underground mine scenes.pdf`。

### 数据规模与采集

论文摘要、Methods 和表 1–4 给出：

- 3171 对对齐 RGB–Depth；
- 1916 个拍摄位置组；
- 来自 6 个矿区：5 个生产煤矿和 1 个教学金矿；
- Microsoft Azure Kinect DK 同步采集；
- 使用 Native SDK 对齐 RGB 与 Depth；
- 原始分辨率 2048×1536；
- 裁掉深度无效六边形边界后发布尺寸为 1082×932。

### 深度格式与无效值

论文 Data Records、图 4 和图 8 明确：

- 发布 Depth 为 16-bit PNG；
- 每个像素保存实际距离信息；
- 图中的 8-bit depth 是由原始 16-bit depth 归一化得到的显示图；
- 作者通过 non-zero pixels 统计有效深度，因此原始值 `0` 表示无效/缺失深度，而不是“最近距离”。

论文没有给出：

- 深度单位；
- 有效最小/最大距离；
- 深度 16→8 的训练转换公式；
- 剩余局部无效点的填补策略。

### 标签与划分

论文列出 15 个前景类别：

1. person
2. cable
3. tube
4. indicator
5. metal fixture
6. container
7. tools & materials
8. door
9. electrical equipment
10. electronic equipment
11. mining equipment
12. anchoring equipment
13. support equipment
14. rescue equipment
15. rail area

本地官方 `Label_ID.pdf` 已确认原始标签语义和顺序：`0` 为 background，`1–15` 依次对应上述 15 个前景类别。当前基线不训练 background，故进入 DFormer 后使用 `0→255`（ignore）、`1–15→0–14`（15 个训练类别）。

官方发布 train/test：

- train：1595；
- test：1576；
- 同一拍摄位置组整体进入同一划分，防止同组多视角泄漏；
- 没有独立 validation split。

论文基线输入尺寸设为 640×480，训练 500 epochs；其他增强和归一化沿各模型官方实现，没有统一详细说明。

---

## 3. 当前项目代码实际如何处理数据

## 3.1 当前数据目录

当前转换结果位于仓库上一级：

```text
../dataset/MUSeg_DFormer/
├── RGB/       # 3171 个 8-bit RGB JPEG
├── Depth/     # 3171 个 8-bit 单通道 PNG，当前 DFormer 输入
├── Depth16/   # 3171 个原始 16-bit 单通道 PNG
├── Label/     # 3171 个 8-bit 类别 ID PNG
├── train.txt  # 1595 行
├── test.txt   # 1576 行
└── dataset_meta.json
```

RGB、Depth、Depth16、Label 的主文件名集合完全一致，均为 1082×932（宽×高）。

## 3.2 Loader 行为

关键文件：`utils/dataloader/RGBXDataset.py`。

- 通用数据集路径由 `get_path()` 根据相同主文件名拼接 RGB、Depth 和 Label；
- RGB 默认以 BGR 读取；仅 `SUNRGBD + DFormerv2` 特判为 RGB；
- Label 使用 `cv2.IMREAD_GRAYSCALE` 并强制为 `np.uint8`；
- Depth 使用 `cv2.IMREAD_GRAYSCALE` 读取，再复制为三个相同通道；
- `x_is_single_channel=True` 只控制归一化参数，不改变读取方式；
- 预处理后 RGB/Depth 转 float tensor，Label 转 long tensor。

因此，直接把 `Depth16/` 指给当前 loader 并不能正确保留 16-bit：`IMREAD_GRAYSCALE` 不是 `IMREAD_UNCHANGED/ANYDEPTH` 路径。

## 3.3 数值归一化

关键文件：

- `utils/transforms.py`：所有图像先除以 255；
- `utils/dataloader/dataloader.py`：单通道辅助模态使用 mean `[0.48, 0.48, 0.48]`、std `[0.28, 0.28, 0.28]`；
- `local_configs/_base_/datasets/NYUDepthv2.py`：RGB 使用 ImageNet mean/std。

当前 Depth 的完整数值链路为：

\[
x=\frac{D_8/255-0.48}{0.28}
\]

结合当前 16→8 公式，可以近似写成：

\[
x\approx\frac{D_{16}/13932-0.48}{0.28}
\]

原始深度每变化约 54.64 个原始单位，8-bit 值变化 1。若要评价量化后的物理分辨率，必须先确认原始单位；MUSeg 论文未给出单位，报告不把它擅自解释为 mm。

## 3.4 DFormerv2 如何使用 Depth

关键文件：`models/encoders/DFormerv2.py`。

- 前向中只取 `x_e[:, 0, :, :]`，即 Depth 三个复制通道中的一个；
- Depth 会被双线性插值到各 stage 的 patch 尺度；
- 代码直接计算 patch 深度之间的绝对差；
- 深度差与空间距离按可学习权重融合为 geometry prior。

这与论文中“使用深度差构造几何先验”的核心一致。代码使用双线性插值，而论文描述 patch average pooling，两者实现细节并不完全相同，复现实验时应以当前代码为准并在报告中注明。

## 3.5 训练增强

关键文件：`utils/dataloader/dataloader.py`。

- 50% 概率同步水平翻转；
- 从配置的尺度数组中随机缩放；
- RGB、Depth 用双线性插值；
- Label 用最近邻插值；
- 同步随机裁剪/填充至 `image_height × image_width`；
- RGB/Depth 填充值为 0，Label 填充值为 255。

该增强方向与 DFormerv2 论文一致。

---

## 4. 当前 MUSeg 数据实测审计

## 4.1 文件、位深与范围

| 模态 | 数量 | 格式 | 位深/通道 | 全数据范围 |
|---|---:|---|---|---|
| RGB | 3171 | JPEG | 8-bit, 3 通道 | 各通道 0–255 |
| Depth | 3171 | PNG | 8-bit, 灰度 | 0–255 |
| Depth16 | 3171 | PNG | 16-bit, 灰度 | 0–13932 |
| Label | 3171 | PNG | 8-bit, 灰度 | ID 0–15 |

总像素数：3,197,712,504。

## 4.2 16→8 映射验证

全像素 100% 匹配：

```python
depth8 = np.rint(depth16.astype(np.float64) * 255.0 / 13932.0).astype(np.uint8)
```

量化性质：

- 理论斜率：`255 / 13932 = 0.0183031869`；
- 抽样回归 Pearson `r = 0.999907992`；
- 抽样回归 `R² = 0.999815993`；
- 低 8 位相等率仅 0.40256%，排除直接 `astype(uint8)`/截低 8 位；
- 非零像素：2,214,892,384；
- 原始零值：982,820,120，占 30.7351%；
- 所有原始零值均映射为 Depth=0；
- Depth=255 仅 388 像素，占约 0.00001213%，没有明显饱和。

结论：当前是**全数据固定线性量化**，不是逐图归一化，能保留全局相对尺度；这是正确方向。

## 4.3 无效深度风险

MUSeg 论文把原始 Depth=0 当作无效点。当前映射保留 0，而 DFormer 新数据说明把灰度 0 描述为最近距离。两者语义冲突。

对 DFormerv2 而言，geometry prior 使用绝对深度差：

- 无效区内部都为 0，彼此差为 0，会被错误视为同一几何平面；
- 无效区与有效远距离之间会产生较大的深度差，可能形成伪边界；
- 当前模型没有 validity mask，无法区分“无效 0”和“真实近距离 0”。

这是当前数据处理的首要技术问题。建议至少做一个对照实验：

- A：原始方案，0 原样保留；
- B：对无效深度做最近邻/引导式填补后再量化；
- C：修改模型，引入有效深度 mask，屏蔽无效值产生的 depth decay。

其中 C 最符合数据语义，但需要改模型；A 最兼容现有代码；B 是低成本工程改进。

## 4.4 标签统计与风险

Label 全数据唯一值严格为 0–15。像素占比：

| ID | 占比 | ID | 占比 |
|---:|---:|---:|---:|
| 0 | 50.9459% | 8 | 1.8583% |
| 1 | 0.1599% | 9 | 4.1974% |
| 2 | 5.6111% | 10 | 0.7995% |
| 3 | 4.0558% | 11 | 1.2193% |
| 4 | 2.1136% | 12 | 0.5225% |
| 5 | 2.2368% | 13 | 0.4876% |
| 6 | 1.2068% | 14 | 5.4265% |
| 7 | 6.9704% | 15 | 12.1885% |

ID=0 占约 50.95%，与 MUSeg 论文“约一半像素属于已标注类别”的描述一致。官方 `Label_ID.pdf` 已确认原始 `0=background`、`1–15` 为按既定顺序排列的 15 个前景类别。当前基线不训练背景，因此配置确定为：

```python
C.num_classes = 15
C.gt_transform = True
C.background = 255
```

`gt_transform=True` 会让 uint8 标签执行 `gt - 1`：

- 原始 0 → 255 ignore；
- 原始 1..15 → 训练 ID 0..14。

该映射已经确认：原始数据中的 `0` 语义是 background；只有经过 `gt_transform` 后，它才成为训练管线中的 ignore=255。

### 全背景标签

共有 11 张 Label 全部为原始 ID 0，即全背景图：train 5 张、test 6 张。启用 `gt_transform` 后整张图才会全部变为 255 ignore。

训练端 `models/builder.py` 对有效像素做布尔筛选再 `.mean()`。若一个本地 batch 没有任何有效像素，空 tensor 的 `.mean()` 会得到 NaN。单个全 ignore 样本与有效样本在同一个 batch 时不会 NaN，但该样本没有监督梯度；batch size=1 或分布式某个 rank 恰好得到全 ignore batch 时有风险。

必须选择并记录一种策略：

1. 推荐基线：从训练和评估 split 排除 11 张全背景图，并生成排除清单；
2. 或修改 `models/builder.py`：当 valid mask 为空时返回与 logits 图相连的零损失；
3. 评估时同样跳过没有有效像素的样本。

## 4.5 划分审计

- `train.txt`：1595 个唯一条目；
- `test.txt`：1576 个唯一条目；
- 样本交集为 0；
- 合集恰好覆盖 3171 个样本；
- 按文件名前四段 `MM-RR-DD-GGGG` 解析采集组：train 958 组、test 957 组，组交集为 0；
- 实际解析得到 1915 个唯一组，与论文/说明中的 1916 组相差 1，需保留为数据版本差异记录；
- 六个矿区均同时出现在 train/test，官方划分是 group-disjoint，不是 mine-disjoint。

不得逐图随机重划 train/test。若需要 validation，应只从官方 train 的采集组中按组划出，test 保持封存。

---

## 5. 当前已经改动了什么

## 5.1 数据侧已经发生的改动

1. 将外部 MUSeg 重组为 `RGB/Depth/Depth16/HHA/Label`；
2. 统一 RGB、Depth、Label 主文件名；
3. 保留 16-bit 原始深度到 `Depth16/`；
4. 使用全局最大值 13932 线性量化得到 `Depth/` 8-bit PNG；
5. 使用官方 1595/1576 train/test 索引。

## 5.2 仓库代码侧现状

截至 2026-08-19，数据转换工程化已经落地：

- 已新增 `tools/prepare_museg.py`；
- 脚本从官方六矿区原始目录收集 RGB、16-bit Depth 和 ID Label；
- 固定使用 `depth_max_raw=13932` 生成 8-bit `Depth/`；
- 保留官方 1595/1576 划分并记录 `DatasetSplit.zip` SHA-256；
- 输出 `dataset_meta.json`；
- 在临时目录中完成四模态集合、位深、尺寸、标签 ID、split 和逐像素映射验证后，再原子替换目标目录；
- 本地 `../dataset/MUSeg_DFormer` 已完成 3171 组全量重建并通过验证。

尚未完成：

- 没有 `MUSeg` 专属训练 config；
- `train.sh` 仍指向 NYUDepthv2；
- 全背景 batch 的安全 loss、无效深度策略、RGB/BGR 和验证尺寸尚未落地。

因此当前结论是：**数据转换已经可复现，训练链路仍未闭环。**

---

## 6. 工程改动与后续文件职责

以下区分已经实现的转换模块和仍待实施的训练模块。

## 6.1 `tools/prepare_museg.py`（已实现并验证）

职责单一：把原始官方发布数据转换成 DFormer 数据目录。当前实现包括：

- 复制 RGB、Depth16、Label；
- 统一文件名并检测跨矿区重名；
- 使用固定 `depth_max_raw=13932` 生成 8-bit Depth；
- 明确 `invalid_value=0` 和 `preserve_zero` 策略；
- 保留官方 split，不按图片随机划分；
- 输出 `dataset_meta.json`，记录公式、参数、划分来源哈希和文件数量；
- 原子写入，避免中断后得到半成品；
- 转换后全量验证四模态、位深、尺寸、标签 ID、split 和深度量化公式。

核心转换由脚本中的 `quantize_depth()` 实现；不能用 `depth16.astype(np.uint8)`，也不能对每张图单独 min-max。

如果后续采用无效值填补，必须在量化前执行，并单独记录 valid mask 或处理参数。

## 6.2 独立审计入口（可选增强）

当前 `tools/prepare_museg.py` 已内置可复用的 `verify_output()`，且只有全量验证通过才替换输出目录。若后续需要 CI 或只读复核命令，可再将该函数封装为独立 `tools/audit_museg.py`，但这已不是数据重建的阻塞项。

## 6.3 `local_configs/_base_/datasets/MUSeg.py`（建议新增）

建议配置：

```python
from .. import *

C.dataset_name = "MUSeg"
C.dataset_path = osp.join(C.root_dir, "MUSeg_DFormer")
C.rgb_root_folder = osp.join(C.dataset_path, "RGB")
C.rgb_format = ".jpg"
C.x_root_folder = osp.join(C.dataset_path, "Depth")
C.x_format = ".png"
C.x_is_single_channel = True
C.gt_root_folder = osp.join(C.dataset_path, "Label")
C.gt_format = ".png"
C.gt_transform = True
C.background = 255
C.num_classes = 15
C.class_names = [
    "person", "cable", "tube", "indicator", "metal fixture",
    "container", "tools & materials", "door", "electrical equipment",
    "electronic equipment", "mining equipment", "anchoring equipment",
    "support equipment", "rescue equipment", "rail area",
]
C.num_train_imgs = 1595
C.num_eval_imgs = 1576
C.train_source = osp.join(C.dataset_path, "train.txt")
C.eval_source = osp.join(C.dataset_path, "test.txt")
C.image_height = 480
C.image_width = 640
C.eval_crop_size = [480, 640]
```

还必须补：

- 明确 `rgb_mode`；
- 训练中是否排除 5 张全背景图；
- validation 从 train 按组划分后的索引；
- depth quantization metadata 路径。

## 6.4 `local_configs/MUSeg/DFormerv2_S.py`（建议新增）

职责：组合 MUSeg 数据配置与 DFormerv2-S 模型/优化配置。不要复制整份模板造成配置漂移，优先复用 `_base_` 模块。

建议先以 DFormerv2-S 建立可运行基线，再扩展 B/L。报告和实验日志必须记录：

- checkpoint 来源；
- RGB/BGR 模式；
- depth policy；
- batch size、seed、GPU 数；
- train/val/test split 哈希；
- 是否 sliding/multi-scale inference。

## 6.5 `utils/dataloader/RGBXDataset.py`（建议小幅修改）

当前 RGB/BGR 由 dataset name 和 backbone 隐式决定。建议改为配置显式传入：

```python
C.rgb_mode = "RGB"  # 或 BGR，但必须与 checkpoint 处理一致
```

Loader 只读取 `config.rgb_mode`，避免新数据集默默落入 BGR 分支。

如果决定直接读取 16-bit Depth，则应另建清晰的 depth reader/normalizer，而不是继续扩展 `_open_image()` 参数分支。

## 6.6 `utils/dataloader/dataloader.py`（建议修改）

MUSeg 原图为 932×1082，而论文基线输入为 480×640。当前训练会随机 crop 到 480×640，但验证预处理默认不 resize。

必须明确选择：

- 为与 MUSeg 基线比较：验证时固定 resize 到 480×640；
- 为保留原分辨率：使用 sliding-window 评估，并在报告中标明不与固定 resize 基线直接比较。

建议增加配置驱动的 eval resize，而不是为 MUSeg 写 dataset-name 特判。RGB/Depth 用双线性，Label 用最近邻。

## 6.7 `models/builder.py`（建议修改）

防止全 ignore batch 的空均值 NaN：

```python
def masked_mean_loss(pixel_loss, label, ignore_index):
    valid = label != ignore_index
    if not torch.any(valid):
        return pixel_loss.sum() * 0.0
    return pixel_loss[valid].mean()
```

主头与辅助头统一调用该函数。该修改是通用稳定性修复，不应只在 MUSeg config 中临时绕过。

## 6.8 `models/encoders/DFormerv2.py`（可选研究改动）

若解决无效 Depth 的根本问题，可让 geometry prior 接收 `depth_valid_mask`：

- 插值 depth 时同步以 nearest 插值 mask；
- 只有两个 patch 都有效时才使用 depth difference；
- 无效 pair 的 depth decay 置零，仅保留 positional decay；
- 不把无效 0 解释为真实深度。

这是模型层改动，必须与“原始不加 mask 的 DFormerv2 基线”做消融对比。

## 6.9 `train.sh`、`eval.sh`、`infer.sh`（建议修改）

把配置路径改为 MUSeg 专属配置，并明确是否启用 sliding、multi-scale、AMP、seed。不要继续使用 NYUDepthv2 配置运行 MUSeg。

---

## 7. 新项目完整复现流程

## 7.1 准备输入

需要：

- 官方 MUSeg 发布数据；
- 官方 `train.txt`/`test.txt`；
- 官方标签 ID 映射或标注脚本；
- DFormer 仓库固定 commit；
- Python 与依赖锁定文件；
- 转换参数：`max_raw=13932`、`invalid_raw=0`、invalid policy。

## 7.2 转换

1. 解压原始数据，不修改源目录；
2. 解析矿区、采集组和样本文件名；
3. 复制或软链接 RGB；
4. 原样保存 16-bit Depth 到 `Depth16/`；
5. 根据固定参数生成 `Depth/`；
6. 复制类别 ID 标签，不使用 `_color.png`；
7. 应用官方 group-disjoint split；
8. 写入 `dataset_meta.json`；
9. 运行审计脚本，审计通过后才启动训练。

## 7.3 必须固化的元数据

建议 `dataset_meta.json` 至少包含：

```json
{
  "dataset": "MUSeg",
  "expected_samples": 3171,
  "image_size_wh": [1082, 932],
  "depth_source_dtype": "uint16",
  "depth_model_dtype": "uint8",
  "depth_quantization": {
    "method": "global_linear_round",
    "min_raw": 0,
    "max_raw": 13932,
    "invalid_raw": 0,
    "invalid_policy": "preserve_zero"
  },
  "label_source_ids": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
  "label_training_map": "0->255, 1..15->0..14",
  "split_policy": "official_group_disjoint"
}
```

## 7.4 验收标准

转换完成后必须全部满足：

- `[PASS]` RGB/Depth/Depth16/Label 各 3171 个；
- `[PASS]` 主文件名集合完全一致；
- `[PASS]` 所有尺寸 1082×932；
- `[PASS]` Depth16 为 16-bit，Depth 为 8-bit；
- `[PASS]` 全像素量化公式一致；
- `[PASS]` Label 唯一值为 0–15；
- `[PASS]` train/test 数量 1595/1576；
- `[PASS]` train/test 样本交集为 0；
- `[PASS]` train/test 采集组交集为 0；
- `[PASS]` 全 ignore 样本已排除或 loss 已安全处理；
- `[PASS]` `rgb_mode`、eval resize/sliding、invalid depth policy 已写入配置和日志；
- `[PASS]` 类别 ID 与名称顺序已由官方 `Label_ID.pdf` 确认。

---

## 8. 推荐实验矩阵

为了判断 8-bit 兼容方案是否损失有效几何信息，至少完成以下消融：

| 实验 | Depth 输入 | 无效值 | 目的 |
|---|---|---|---|
| B0 | 当前 8-bit 全局线性 | 保留 0 | 复现当前最小基线 |
| B1 | 当前 8-bit 全局线性 | 填补无效点 | 测试无效 0 的影响 |
| B2 | 当前 8-bit 全局线性 | validity mask | 验证 mask 几何先验 |
| B3 | 16-bit/float 全局归一化 | validity mask | 测试量化精度损失 |
| B4 | RGB only | 无 Depth | 衡量 Depth 的净贡献 |
| B5 | HHA 专用分支 | HHA | 与 MUSeg 论文的 HHA 基线对照 |

B5 不能通过把 HHA 直接送入当前 DFormerv2 第一个 Depth 通道实现；应建立真正的 HHA 编码分支或选用支持三通道辅助模态的结构。

评价时至少报告：mIoU、mAcc、各类别 IoU、3 次 seed 均值与方差，并按弱光/矿区或深度有效率做分组分析。

---

## 9. 组会汇报建议

建议按以下逻辑汇报：

1. **问题**：MUSeg 发布的是带实际距离的 16-bit Depth，但当前 DFormer loader 强制走 8-bit grayscale + `/255` 归一化；两者接口不一致。
2. **已有改动**：保留 `Depth16/`，按全数据固定最大值 13932 生成 8-bit `Depth/`，并按官方组级 split 重组 3171 组数据。
3. **为何合理**：DFormerv2 使用 patch 深度差；固定线性量化保留全局顺序与相对差，且兼容当前 loader 和预训练数值范围。
4. **定量验证**：全部约 31.98 亿像素 100% 符合统一转换公式；不是截位，也不是逐图归一化；饱和像素极少。
5. **尚存问题**：30.74% 原始零值是无效深度，但当前会被当成数值 0；11 张全背景图经 `gt_transform` 后成为全 ignore，可能触发 NaN；MUSeg 配置、验证尺寸与 RGB/BGR 未明确。
6. **下一步**：新增 MUSeg config 并建立 B0；补全背景样本安全 loss，再做无效深度填补/mask、16-bit float 和 RGB-only 消融。

一句话总结：

> 当前“保留 16-bit 原始深度 + 固定全局线性量化为 8-bit 模型输入”的数据复现闭环已经完成；下一阶段重点是 MUSeg 训练配置、全背景样本安全处理和无效深度策略。

---

## 10. 证据索引

### 论文

- `paper/25CVPR_RGBDSeg-CN.pdf`
  - 第 3 页 §3.1，公式 (1)(2)：深度关系与几何先验；
  - 第 3 页 §3.2，公式 (4)：geometry-aware attention；
  - 第 5 页 §3.3：多尺度 depth prior；
  - 第 5 页 §4.1：训练增强、尺寸和优化器。
- `paper/Li 等 - 2025 - 1-MUSeg A multimodal semantic segmentation dataset for complex underground mine scenes.pdf`
  - 摘要、Methods、表 1–4：规模、采集与分辨率；
  - Data filtering：non-zero depth 有效性；
  - Data Records、图 7：发布文件格式；
  - 图 4、图 8：16-bit 原始深度与 8-bit 可视化说明；
  - Technical Validation、表 5：官方 split、输入尺寸与基线。

### 当前代码

- `tools/prepare_museg.py`：确定性转换、metadata、官方 split 校验和全量输出验证；
- `doc/dataset.md`：转换目录、命令、深度公式和标签映射说明；
- `utils/dataloader/RGBXDataset.py`：路径、读取 flag、通道、`gt_transform`；
- `utils/dataloader/dataloader.py`：同步增强、插值、裁剪、Depth 归一化；
- `utils/transforms.py`：`/255` 标准化；
- `models/encoders/DFormerv2.py`：单通道 Depth、深度差和 geometry prior；
- `models/builder.py`：ignore 像素筛选和空均值风险；
- `utils/train.py`：CrossEntropyLoss 与 ignore index；
- `local_configs/_base_/datasets/NYUDepthv2.py`：当前参考数据配置；
- `local_configs/template/DFormer_Large.py`：新数据集配置模板；
- `train.sh`、`eval.sh`、`infer.sh`：当前仍使用 NYUv2 配置。

### 数据实测

- RGB/Depth/Depth16/Label：各 3171；
- train/test：1595/1576；
- 量化公式：全像素 100% 匹配；
- 采集组交集：0；
- Label ID：0–15；
- 原始零深度率：30.7351%；
- 全背景标签：11 张（train 5、test 6），经 `gt_transform` 后成为全 ignore。