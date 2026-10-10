# 数据集下载与配置指南

> 迁移说明（2026-10-10）：原文从 `临时/DATASET_DOWNLOAD_GUIDE.md` 迁入，保留2026-10-08的来源核实记录与操作示例。以下研究顺序、未下载表述和配置名属于当时背景；当前路线为SUN主集、NYUv2第二验证集、DeLiVER暂不适配、MUSeg可选扩展。当前数据与配置以[项目状态](../state/current.md)和[ODG入口](odg.md)为准；本地后续落盘记录见[交接报告](../reports/2026-10-09-research-handoff.md)。本指南不是下载或云端操作授权。

核实日期：2026-10-08。本文件只记录下载和配置方法；**本轮没有执行任何数据集下载、解压、转换、训练或评价**。下列命令均为以后获得数据下载授权时的操作示例。

第一阶段使用 SUN RGB-D 快速开发；DeLiVER 用作正式 robustness benchmark；NYU Depth V2 为第二阶段可选；MUSeg 用作最终 underground real-world validation。

## 1. SUN RGB-D：第一阶段快速开发

### 来源与版本选择

- 原始官方页面：https://rgbd.cs.princeton.edu/
- 原始 SUNRGBD V1：https://rgbd.cs.princeton.edu/data/SUNRGBD.zip ，官方说明包含 10,335 张 RGB-D 图像；原始 annotation / MATLAB toolbox：https://rgbd.cs.princeton.edu/data/SUNRGBDtoolbox.zip 。这些原始资料不等于 DFormer 整理格式。
- **优先使用 DFormer 作者整理版**。依据：https://github.com/VCIP-RGBD/DFormer/blob/main/README.md ，其中 Datasets 一节提供：
  - Google Drive：https://drive.google.com/drive/folders/1RIa9t7Wi4krq0YcgjR3EWBxWWJedrYUl
  - OneDrive：https://mailnankaieducn-my.sharepoint.com/:f:/g/personal/bowenyin_mail_nankai_edu_cn/EqActCWQb_pJoHpxvPh4xRgBMApqGAvUjid-XK3wcl08Ug?e=VcIVob
  - 百度网盘：https://pan.baidu.com/s/1-CEL88wM5DYOFHOVjzRRhA?pwd=ij7q ，提取码 `ij7q`。
- 本次仅浏览 Google Drive 文件列表，确认作者文件夹含 `SUNRGBD.zip`（页面显示 2.28 GB）和 `NYUDepthv2.zip`；**只选择 SUNRGBD.zip**，不整文件夹下载。
- 从作者文件夹页面直接核实的 SUNRGBD.zip 文件 ID：`180rzPMudFK5jVtYFnBpsGrx9wU8QVNG7`，单文件页面：https://drive.google.com/file/d/180rzPMudFK5jVtYFnBpsGrx9wU8QVNG7/view 。链接可能受共享权限 / 流量配额影响；遇限流改用作者其他镜像，不能把 HTML 提示页当 zip。

### Windows 本地下载与解压

1. 在浏览器打开作者百度网盘或 Google Drive，单独下载 `SUNRGBD.zip`，建议保存到 `D:\0Project\dataset\downloads\`。百度网盘可使用官方客户端；Google Drive 可使用浏览器下载。不要下载训练权重来替代数据包。
2. 可选命令行单文件下载，使用普通 Python 环境，不需要 GPU；示例中的安装 / 下载本轮未执行：

```powershell
python -m pip install gdown
gdown --fuzzy "https://drive.google.com/file/d/180rzPMudFK5jVtYFnBpsGrx9wU8QVNG7/view" -O "D:\0Project\dataset\downloads\SUNRGBD.zip"
```

3. 使用 7-Zip 先查看包内顶层目录，再解压。若包内已有 `SUNRGBD/` 顶层，解压到 `D:\0Project\dataset\`；若顶层直接是 RGB / Depth / labels，则解压到 `D:\0Project\dataset\SUNRGBD\`。避免 `SUNRGBD\SUNRGBD\` 双重目录。

```powershell
7z l "D:\0Project\dataset\downloads\SUNRGBD.zip"
# 仅适用于压缩包自身包含 SUNRGBD/ 顶层的情况：
7z x "D:\0Project\dataset\downloads\SUNRGBD.zip" -o"D:\0Project\dataset"
# 在将来启动研究配置之前设置，值为 SUNRGBD 的父目录：
$env:DFORMER_DATASET_ROOT = "D:\0Project\dataset"
```

也可直接放在 `D:\0Project\DFormer\datasets\SUNRGBD\` 使用配置默认值，但不要重复保留两份解压数据。

### Linux 云端下载与解压

当前工程为 `/root/rivermind-data/DFormer`，已有数据资产父目录为 `/root/rivermind-data/dataset`。以后可在无卡模式下载单一作者包，或者上传本地已下载 zip，二者任选，避免重复传输：

```bash
source /usr/local/miniconda3/bin/activate py310
mkdir -p /root/rivermind-data/dataset/downloads
python -m pip install --no-cache-dir gdown
gdown --fuzzy 'https://drive.google.com/file/d/180rzPMudFK5jVtYFnBpsGrx9wU8QVNG7/view' \
  -O /root/rivermind-data/dataset/downloads/SUNRGBD.zip
unzip -l /root/rivermind-data/dataset/downloads/SUNRGBD.zip
# 仅适用于包内已有 SUNRGBD/ 顶层：
unzip /root/rivermind-data/dataset/downloads/SUNRGBD.zip \
  -d /root/rivermind-data/dataset
export DFORMER_DATASET_ROOT=/root/rivermind-data/dataset
```

若包内没有 SUNRGBD/ 顶层，将 `-d` 改为 `/root/rivermind-data/dataset/SUNRGBD`。如希望使用作者默认 `datasets/` 路径，也可只建立单一软链接：

```bash
cd /root/rivermind-data/DFormer
mkdir -p datasets
# datasets/SUNRGBD 尚不存在时执行；不覆盖现有目录：
ln -s /root/rivermind-data/dataset/SUNRGBD datasets/SUNRGBD
```

这里没有承诺本次验证过下载吞吐、配额或压缩包内部内容；只核实了官方列表、下载入口与代码预期格式。

### DFormer 预期目录、split 与 depth

依据：`local_configs/_base_/datasets/SUNRGBD.py`、`local_configs/SUNRGBD/DFormerv2_S.py`、`utils/dataloader/RGBXDataset.py`。

```text
SUNRGBD/
├── RGB/          # 同名 .jpg
├── Depth/        # 同名 .png，作者转换的 depth 表示
├── labels/       # 同名 .png，注意小写 labels
├── train.txt
└── test.txt
```

- 作者协议：train=5,285，test=5,050，37 类。保留作者提供的 txt 文件及原行格式，不重新随机切分，不重写为裸文件名。
- 当前加载代码对 SUNRGBD 走通用路径解析：从 split 行的第二个 `/` 分段提取图像名称，例如 `RGB/<name>.jpg`；以这个名称拼接 RGB / Depth / labels。下载后先抽查实际 txt 行和对应三文件是否一致，不自行猜测替换路径。
- 标签设置 `gt_transform=True`，读取后减1；原始标签0映射到 ignore=255，类编号变为0–36。不要先把文件标签减1后又启用这个转换。
- 作者 README 明确说明把原 depth `.npy` 经 `plt.imsave(save_path, np.load(depth), cmap='Greys_r')` 转为 `.png`。**它是作者提供的深度表示，不应假定为原始米制值或可逆的 uint16 度量深度。**
- 当前加载器用 `cv2.IMREAD_GRAYSCALE` 读取 depth，并复制到3通道。不要擅自改为16位读取、额外归一化、HHA 或另一种 colormap；这些改变会形成不同输入协议。
- SUNRGBD + DFormerv2 的作者入口要求 `--pad_SUNRGBD`，且此组合使用 RGB 读取方式。新研究配置为 `local_configs.research.DFormerv2_S_SUNRGBD`。
- 作者 config 的 `eval_source=test.txt`。正式实验前必须确定开发验证策略和最终 test 使用边界；本轮不评价 test、不编造 val split。新 baseline 与候选方法使用完全相同的数据预处理和 evaluator。

## 2. DeLiVER：正式 robustness benchmark

### 官方页面与公开下载范围

- Dataset / project：https://jamycheung.github.io/DELIVER.html
- 官方 dataset / CMNeXt code：https://github.com/jamycheung/DELIVER （目前 GitHub 可重定向至 https://github.com/InSAI-Lab/DELIVER ）。
- 官方 README 发布的 front-view 包：https://drive.google.com/file/d/1P-glCmr-iFSYrzCfNawgVI9qKWfP94pm/view?usp=share_link ，约12.2 GB。
- **完整下载这个官方公开 front-view 包**即可取得该公开发布范围；没有核实到官方 RGB+Depth 独立分包，不杜撰分包地址。数据集设计描述有 front / rear / left / right / up / down 六视角，但官方更新记录明确说 release front-view，不能把上述链接称作“完整六视角包”。如以后需要完整六视角，应向作者确认额外发布入口，不自行推断。

### 完整公开包的下载方法

Windows 可用浏览器下载上述 Google Drive 文件；本次仅浏览文件页面，确认官方包名为 **`DELIVER.tar.gz`**。命令行可选：

```powershell
gdown --fuzzy "https://drive.google.com/file/d/1P-glCmr-iFSYrzCfNawgVI9qKWfP94pm/view" -O "D:\0Project\dataset\downloads\DELIVER.tar.gz"
tar -tf "D:\0Project\dataset\downloads\DELIVER.tar.gz"
# 仅适用于包内已有 DELIVER/ 顶层：
tar -xf "D:\0Project\dataset\downloads\DELIVER.tar.gz" -C "D:\0Project\dataset"
```

Linux 可选：

```bash
gdown --fuzzy 'https://drive.google.com/file/d/1P-glCmr-iFSYrzCfNawgVI9qKWfP94pm/view' \
  -O /root/rivermind-data/dataset/downloads/DELIVER.tar.gz
tar -tzf /root/rivermind-data/dataset/downloads/DELIVER.tar.gz
# 仅适用于包内已有 DELIVER/ 顶层：
tar -xzf /root/rivermind-data/dataset/downloads/DELIVER.tar.gz \
  -C /root/rivermind-data/dataset
```

解压目标建议为 `D:\0Project\dataset\DELIVER\` / `/root/rivermind-data/dataset/DELIVER/`，先检查顶层避免多嵌套一层；若包内直接是模态目录，创建 DELIVER 目录后将 `-C` 指向该目录。官方代码默认根路径是 `data/DELIVER`，以后通过其配置 `DATASET.ROOT` 指向实际位置，无需复制数据。

### 第一阶段 RGB + Depth 需要保留什么

- 对 DFormerv2 的 geometry-prior 研究，保留 `img/`（RGB）、`depth/`（原 depth）、`semantic/`（标签）和官方 train / val / test 组织。当前 DFormer 工程尚未接入 DeLiVER，不应声称这三目录可以直接用 SUNRGBD config 训练。
- 为保留官方 RGB-D benchmark 复现能力，同时保留 `hha/`：官方 `semseg/datasets/deliver.py` 把配置中的 `depth` 模态实际映射至 `/hha/`，并将 `_rgb` 文件名替换为 `_depth`；其 RGB-D 输入组合是 `['img','depth']`，但读取的并非原 `depth/` 文件。
- `event/` 和 `lidar/` 不用于第一阶段 RGB+Depth 输入。官方当前给的是统一公开包而非按模态分包，因此下载阶段仍下载整个包，不声称可按模态减少官方下载体积；解压后先保留，不在未确认研究协议前删资产。
- 官方 README 的 HHA 生成参考：https://github.com/charlesCXK/Depth2HHA-python 。原 depth 和 HHA 不是可互换的 geometry 输入。需要复现哪套协议由正式实验前裁决，本轮不运行转换、不编写适配器。

### 官方 split、天气和 sensor failure 位置

官方目录树与加载器：

```text
DELIVER/
├── img/
│   ├── cloud/{train,val,test}/<scene>/*.png
│   ├── fog/{train,val,test}/<scene>/*.png
│   ├── night/{train,val,test}/<scene>/*.png
│   ├── rain/{train,val,test}/<scene>/*.png
│   └── sun/{train,val,test}/<scene>/*.png
├── depth/       # 按相同 weather / split / scene 层次组织
├── hha/
├── semantic/
├── event/
└── lidar/
```

- 官方 split 为 `train`、`val`、`test`；加载器从 `img/*/<split>/*/*.png` 获取样本。保持官方目录 split，不合并 val/test，也不重新随机切分；准确文件数量以后解压后按该 glob 核对，本轮未下载，不编造实际样本盘点。
- 天气路径名是 `cloud`、`fog`、`night`、`rain`、`sun`。原始页面描述5种天气条件，其中4种为 adverse 条件；不要把 all-cases 结果冒称某一个天气子集结果。
- 五个 sensor failure case 名：`motionblur`（运动模糊）、`overexposure`（过曝）、`underexposure`（欠曝）、`lidarjitter`（LiDAR抖动）、`eventlowres`（事件低分辨率）。
- 官方加载器采用 `case in full_file_path` 筛选。case 信息在公开数据的路径 / 文件名中识别，不存在本轮已核实的统一顶层 `sensor_failure/` 目录；下载后按这些关键字检查实际位置，保留全部场景路径和文件名。LiDAR / event 专属失效也不能自动解释为 depth 失效。

### Robustness benchmark 代码来源

- 官方评价入口：https://github.com/jamycheung/DELIVER/blob/main/tools/val_mm.py
- 官方数据 / case 筛选：https://github.com/jamycheung/DELIVER/blob/main/semseg/datasets/deliver.py
- 官方配置示例：https://github.com/jamycheung/DELIVER/blob/main/configs/deliver_rgbdel.yaml
- 评价脚本默认 split=`val`，`test` 为注释中的替代构造；正式报告分别保留 val 开发与 test 最终评价。
- `cases=[None]` 为全部；天气分项使用 `['cloud','fog','night','rain','sun']`；失效分项使用上述5个 failure 名。后续 DFormer evaluator 需遵循相同筛选和标签协议，并固定评价尺度 / flip；本轮不运行官方 CMNeXt evaluator，也不把它误写成 DFormer 已有原生 DeLiVER 支持。

## 3. NYU Depth V2：第二阶段可选，不立即下载

- 官方页面：https://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html （可重定向到 https://cs.nyu.edu/~fergus/datasets/nyu_depth_v2.html ）。
- 官方标注版含1,449对对齐 RGB/Depth；DFormer 配置使用795 train / 654 test、40类。原始 MATLAB / raw 包不是作者整理 PNG 数据目录。
- 将来若使用，优先作者同一 Datasets 镜像中的 `NYUDepthv2.zip`；结构为 `RGB/`、`Depth/`、`Label/`（注意大写）、`train.txt`、`test.txt`。
- 当前本地 / 云端均不下载，不启动预处理或实验。

## 4. MUSeg：最终 underground real-world validation

- 继续复用 `D:\0Project\dataset\MUSeg\` 与 `D:\0Project\dataset\MUSeg_DFormer\`；本地整理版已存在 RGB / Label / Depth / Depth16，各3,171文件，train1,595 / test1,576。云端原有 MUSeg 资产保留在 `/root/rivermind-data/dataset/`。
- 不重新下载，不重新转换，不把原 MUSeg/MMFR checkpoints 删除或当成新论文主 baseline。旧项目与旧云端工程保留历史参考。
- 新数据集正式实验开始后，应从同一官方 DFormerv2-S pretrained 重训 baseline，baseline 与候选方法共用 seed、schedule、augmentation、evaluator；本轮未重训 baseline。
