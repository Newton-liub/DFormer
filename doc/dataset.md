# MUSeg 数据集说明

## 数据集位置

本项目约定数据集统一放在仓库上一级的 `dataset` 目录中，MUSeg 的默认路径为：

```text
DFormer/                 # 本仓库
../dataset/MUSeg/        # MUSeg 数据集
```

相对于仓库根目录，配置路径应为：

```python
C.root_dir = "../dataset"
C.dataset_name = "MUSeg"
C.dataset_path = osp.join(C.root_dir, C.dataset_name)
```

## 项目需要的数据集信息

MUSeg 是面向地下矿山场景的 RGB-D 语义分割数据集。本项目需要关注以下信息：

- 3,171 组对齐的 RGB-D 数据；
- 1,916 个采集组，来自 5 座煤矿和 1 座金矿；
- 图像分辨率为 `1082 × 932`；
- RGB 图像为 JPG；
- Depth 图像为 PNG，原始像素保存实际距离信息；
- 语义标注包含类别 ID PNG、彩色 PNG 和 Labelme JSON；
- 共 15 个矿下目标类别，实际训练前须以数据集提供的类别 ID 映射为准。

同一采集组可能包含多个相似视角。生成训练集和测试集时应按采集组划分，避免同组样本跨集合造成数据泄漏。

## DFormer 使用结构

当前 DFormer 通用数据加载器要求 RGB、Depth 和 Label 使用相同主文件名，并通过 `train.txt` 和 `test.txt` 建立索引。整理后的目录应为：

```text
../dataset/MUSeg/
├── RGB/
│   ├── sample_001.jpg
│   └── sample_002.jpg
├── Depth/
│   ├── sample_001.png
│   └── sample_002.png
├── Label/
│   ├── sample_001.png
│   └── sample_002.png
├── train.txt
└── test.txt
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

## 接入前必须确认

### Depth

MUSeg 原始深度图保存实际距离，而当前 DFormer 加载器按 8 位灰度图读取深度。训练前需要统一把原始深度转换为项目使用的 8 位单通道 PNG，并保持“近处为 0、远处为 255”。转换范围必须根据 MUSeg 的深度单位和有效值范围确定，不能直接把 16 位深度图当作普通 8 位图使用。

### Label

当前 DFormer 配置中的 `gt_transform=True` 会对标签 ID 执行减 1。接入前必须检查 MUSeg 标签中的全部唯一值，并根据官方类别映射确定：

- 类别数和类别名称；
- 背景是否参与训练；
- 是否存在忽略标签；
- 是否需要启用 `gt_transform`；
- 损失函数使用的 `ignore_index`。

标签缩放只能使用最近邻插值，不能对类别 ID 做归一化或双线性插值。
