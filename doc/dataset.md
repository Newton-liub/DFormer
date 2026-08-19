# MUSeg 数据集说明

## 数据集位置与转换

项目约定原始 MUSeg 和 DFormer 输入均放在仓库上一级的 `dataset` 目录：

```text
DFormer/                         # 本仓库
../dataset/MUSeg/                # 官方原始数据，保持不变
../dataset/MUSeg_DFormer/        # 脚本生成的 DFormer 输入
```

在仓库根目录运行：

```bash
python tools/prepare_museg.py
```

如果输出目录已经存在，并确认需要完整重建：

```bash
python tools/prepare_museg.py --overwrite
```

脚本固定使用全数据集映射 `round(depth16 * 255 / 13932)`，保留原始 16-bit 深度，并使用官方 `Experiment/DatasetSplit.zip` 生成 `train.txt` 和 `test.txt`。转换会先在临时目录完成全量验证，再替换目标目录；因此本地和云端应使用同一脚本与参数，不能分别计算量化范围。

数据配置应指向转换结果：

```python
C.root_dir = "../dataset"
C.dataset_name = "MUSeg"
C.dataset_path = osp.join(C.root_dir, "MUSeg_DFormer")
```

## 项目需要的数据集信息

MUSeg 是面向地下矿山场景的 RGB-D 语义分割数据集。本项目需要关注以下信息：

- 3,171 组对齐的 RGB-D 数据；
- 1,916 个采集组，来自 5 座煤矿和 1 座金矿；
- 图像分辨率为 `1082 × 932`；
- RGB 图像为 JPG；
- Depth 图像为 PNG，原始像素保存实际距离信息；
- 语义标注包含类别 ID PNG、彩色 PNG 和 Labelme JSON；
- 共 15 个矿下前景类别；原始标签 `0` 为背景，`1–15` 按官方 `Label_ID.pdf` 顺序对应 15 个前景类别。

同一采集组可能包含多个相似视角。生成训练集和测试集时应按采集组划分，避免同组样本跨集合造成数据泄漏。

## DFormer 使用结构

当前 DFormer 通用数据加载器要求 RGB、Depth 和 Label 使用相同主文件名，并通过 `train.txt` 和 `test.txt` 建立索引。整理后的目录应为：

```text
../dataset/MUSeg_DFormer/
├── RGB/
│   ├── sample_001.jpg
│   └── sample_002.jpg
├── Depth/               # 全局线性量化后的 8-bit 输入
│   ├── sample_001.png
│   └── sample_002.png
├── Depth16/             # 原始 16-bit 深度
├── Label/
│   ├── sample_001.png
│   └── sample_002.png
├── train.txt
├── test.txt
└── dataset_meta.json    # 映射参数、数据规模和划分来源哈希
```

每个索引文件一行记录一个 RGB 相对路径：

```text
RGB/sample_001.jpg
RGB/sample_002.jpg
```

原始 MUSeg 按矿井分别存放 `Image/`、`Depth/` 和 `Label/`，接入本项目之前需要整理为上述结构。整理时必须保证：

- RGB、Depth 和 Label 空间对齐且主文件名一致；
- `*_label.png` 作为训练标签，整理后去掉 `_label` 后缀；
- `*_color.png` 仅用于可视化，不作为训练标签；
- `*_polygons.json` 不参与常规语义分割训练；
- 不同矿井出现同名文件时，重命名后仍需保持三种数据一致；
- `train.txt` 和 `test.txt` 中的样本数量与配置一致。

## 接入规则

### Depth

MUSeg 原始深度图保存实际距离，而当前 DFormer 加载器按 8 位灰度图读取深度。`tools/prepare_museg.py` 会保留原始深度到 `Depth16/`，并将所有样本统一转换为 8 位单通道 `Depth/`：

\[
D_8=\operatorname{round}\left(D_{16}\times\frac{255}{13932}\right)
\]

其中原始 `0` 作为无效深度保留为 `0`。禁止逐图、按训练集或按测试集分别计算 min-max，也不能直接把 16 位深度截断为 8 位。

### Label

MUSeg 原始类别 ID 的语义已经确认：

- `0`：background；
- `1`：person；
- `2`：cable；
- `3`：tube；
- `4`：indicator；
- `5`：metal fixture；
- `6`：container；
- `7`：tools & materials；
- `8`：door；
- `9`：electrical equipment；
- `10`：electronic equipment；
- `11`：mining equipment；
- `12`：anchoring equipment；
- `13`：support equipment；
- `14`：rescue equipment；
- `15`：rail area。

当前基线不训练背景类，因此使用：

```python
C.num_classes = 15
C.gt_transform = True
C.background = 255
```

`gt_transform=True` 对 uint8 标签执行减 1：原始背景 `0` 回绕为 `255`（ignore），原始前景 `1–15` 映射为训练 ID `0–14`。因此必须区分“原始 `0` 是 background”和“进入损失函数后该像素是 ignore=255”。标签缩放只能使用最近邻插值，不能对类别 ID 做归一化或双线性插值。
