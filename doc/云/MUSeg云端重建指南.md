# 云端 MUSeg 重建与 DFormer 操作指南

> 用途：将本指南直接交给云端 AI 执行。
>
> 约束：本云端机器只同时运行一个项目，**不创建、不激活 Conda/venv 虚拟环境**，直接使用当前 `base` 环境。所有命令均在 DFormer 仓库根目录执行。

---

## 一、给云端 AI 的执行要求

请按本文件顺序执行，不要跳过数据检查。执行过程中：

1. 直接使用当前 Conda `base` 环境，不创建虚拟环境。
2. 依赖只安装到当前 `base` 环境。
3. 不删除原始 MUSeg 数据，除非用户明确确认原始数据已备份。
4. 不把数据集、checkpoint、日志和转换结果加入 Git。
5. 修改代码前先读取相关文件；完成修改后运行语法检查和必要验证。
6. 不自行改变深度量化公式、最大值、官方划分或目录命名。
7. 如果发现路径、文件数量、深度位深或划分不一致，先停止转换并报告，不要猜测修复。

推荐先执行：

```bash
conda activate base
cd /path/to/DFormer
```

如果当前已经在 `base` 环境，不需要重复激活。确认环境：

```bash
which python
python --version
python -c "import sys; print(sys.executable)"
```

Windows 云端使用 PowerShell 时，将 `which python` 替换为：

```powershell
Get-Command python
python --version
```

---

## 二、同步代码

如果云端已有仓库：

```bash
git status --short
git pull --ff-only
```

如果需要首次克隆：

```bash
git clone <GitHub仓库地址>
cd DFormer
git checkout <目标分支>
```

确认转换脚本存在：

```bash
test -f tools/prepare_museg.py
```

Windows PowerShell：

```powershell
Test-Path tools/prepare_museg.py
```

如果脚本不存在，先同步正确分支或报告“代码版本不完整”，不要自行重写脚本。

---

## 三、准备 base 环境依赖

项目当前没有单独的 `requirements.txt`、`environment.yml` 或 `pyproject.toml`。因此不要盲目安装一整套新环境，先检查 base 环境：

```bash
python - <<'PY'
import importlib.util

packages = [
    "torch",
    "cv2",
    "numpy",
    "PIL",
]
for package in packages:
    print(f"{package}: {'OK' if importlib.util.find_spec(package) else 'MISSING'}")
PY
```

缺少转换脚本所需的包时，只在当前 `base` 环境安装：

```bash
conda install -y numpy opencv
```

若 Conda 源不可用，再使用：

```bash
python -m pip install numpy opencv-python
```

转换脚本只需要 `Python`、`numpy`、`opencv-python`。训练所需的 PyTorch、CUDA 和其他依赖必须根据云端已有 GPU/CUDA 环境检查后再安装，不要为了转换数据更换现有 PyTorch 版本。

检查训练环境：

```bash
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("cuda version:", torch.version.cuda)
PY
```

---

## 四、确认数据目录

项目约定数据集放在仓库上一级的 `dataset` 目录：

```text
<parent>/
├── DFormer/
└── dataset/
    └── MUSeg/
```

转换前原始 MUSeg 至少应包含：

```text
../dataset/MUSeg/
├── 01-Mine/
│   ├── Image/
│   ├── Depth/
│   └── Label/
├── 02-Mine/
├── 03-Mine/
├── 04-Mine/
├── 05-Mine/
├── 06-Mine/
└── Experiment/
    └── DatasetSplit.zip
```

检查目录：

```bash
ls -ld ../dataset/MUSeg
find ../dataset/MUSeg -maxdepth 2 -type d | sort
ls -lh ../dataset/MUSeg/Experiment/DatasetSplit.zip
```

Windows PowerShell：

```powershell
Get-ChildItem ..\dataset\MUSeg
Get-Item ..\dataset\MUSeg\Experiment\DatasetSplit.zip
```

如果云端数据集使用了其他绝对路径，可以通过参数显式传入，但不要修改脚本中的默认规则：

```bash
python tools/prepare_museg.py \
  --source-root /data/dataset/MUSeg \
  --output-root /data/dataset/MUSeg_DFormer \
  --split-zip /data/dataset/MUSeg/Experiment/DatasetSplit.zip
```

---

## 五、删除旧转换结果前的要求

如果云端已有旧的处理结果，先确认它不是原始数据：

```bash
ls -ld ../dataset/MUSeg ../dataset/MUSeg_DFormer
```

**只允许删除或覆盖 `MUSeg_DFormer`，禁止删除 `MUSeg` 原始目录。**

推荐直接让转换脚本原子重建：

```bash
python tools/prepare_museg.py --overwrite
```

脚本会：

- 从原始 `MUSeg` 读取数据；
- 在临时目录生成结果；
- 完成全量验证后替换 `MUSeg_DFormer`；
- 不修改原始数据。

如果需要人工删除旧结果，只能执行：

```bash
rm -rf ../dataset/MUSeg_DFormer
python tools/prepare_museg.py
```

执行 `rm -rf` 前必须再次确认路径确实是 `MUSeg_DFormer`，不能使用变量不明确的删除命令。

---

## 六、执行统一转换

在 DFormer 仓库根目录执行：

```bash
python tools/prepare_museg.py --overwrite
```

默认参数：

- 原始输入：`../dataset/MUSeg`
- 输出目录：`../dataset/MUSeg_DFormer`
- 官方划分：`../dataset/MUSeg/Experiment/DatasetSplit.zip`
- 全数据集原始深度最大值：`13932`
- 深度映射：

\[
D_8=\operatorname{round}\left(D_{16}\times\frac{255}{13932}\right)
\]

- 原始深度 `0`：保留为 `0`
- 不允许逐图 min-max
- 不允许分别使用 train/test 的最大最小值
- 不允许直接截断 16-bit 为 8-bit

成功时应看到类似输出：

```text
Converted 3171 samples to .../dataset/MUSeg_DFormer
Official split: train=1595, test=1576
Depth mapping: round(depth16 * 255 / 13932)
Observed source depth maximum: 13932
```

如果输出不是这些数量，或者出现异常，停止后报告完整错误信息。

---

## 七、转换结果验收

转换脚本在替换目标目录前已经执行全量验证。云端 AI 仍需执行以下人工复核：

```bash
find ../dataset/MUSeg_DFormer/RGB -type f | wc -l
find ../dataset/MUSeg_DFormer/Depth -type f | wc -l
find ../dataset/MUSeg_DFormer/Depth16 -type f | wc -l
find ../dataset/MUSeg_DFormer/Label -type f | wc -l
wc -l ../dataset/MUSeg_DFormer/train.txt ../dataset/MUSeg_DFormer/test.txt
cat ../dataset/MUSeg_DFormer/dataset_meta.json
```

预期：

- `RGB`：3171
- `Depth`：3171
- `Depth16`：3171
- `Label`：3171
- `train.txt`：1595 行
- `test.txt`：1576 行
- `dataset_meta.json` 中 `max_raw` 和 `observed_source_max_raw` 都为 `13932`

使用脚本内置验证再次检查已生成结果：

```bash
python - <<'PY'
from pathlib import Path
from tools.prepare_museg import collect_samples, load_split, verify_output

source = Path("../dataset/MUSeg").resolve()
output = Path("../dataset/MUSeg_DFormer").resolve()
samples = collect_samples(source)
split = load_split(source / "Experiment" / "DatasetSplit.zip", samples)
verify_output(output, set(samples), split, (1082, 932), 13932)
print("PASS: all 3171 converted samples verified")
PY
```

---

## 八、训练配置路径

训练配置必须指向转换结果，而不是原始目录：

```python
C.root_dir = "../dataset"
C.dataset_name = "MUSeg_DFormer"
C.dataset_path = osp.join(C.root_dir, C.dataset_name)
```

如果新建 MUSeg 配置，目录应使用：

```text
local_configs/_base_/datasets/MUSeg.py
local_configs/MUSeg/
```

配置中的数据目录应为：

```text
RGB      -> ../dataset/MUSeg_DFormer/RGB
Depth    -> ../dataset/MUSeg_DFormer/Depth
Label    -> ../dataset/MUSeg_DFormer/Label
train    -> ../dataset/MUSeg_DFormer/train.txt
test     -> ../dataset/MUSeg_DFormer/test.txt
```

不要把 `Depth16` 直接交给当前默认 loader。当前 loader 的 DFormer 输入是 8-bit `Depth`；`Depth16` 只用于保留原始数据和后续研究。

---

## 九、Git 和数据保护

数据集、转换结果和训练产物不上传 GitHub。转换完成后检查：

```bash
git status --short
```

只应看到代码或文档改动，例如：

```text
 M doc/dataset.md
?? tools/prepare_museg.py
```

如果出现以下目录或文件，不要提交：

- `../dataset/`
- `MUSeg/`
- `MUSeg_DFormer/`
- `checkpoints/`
- `work_dirs/`
- `*.pth`
- `*.pt`
- 大型日志和可视化结果

提交前确认 `.gitignore` 已覆盖这些输出；如果没有覆盖，先补充忽略规则，不要 `git add` 数据。

---

## 十、云端 AI 的直接执行提示词

可以将以下内容直接发给云端 AI：

```text
请在当前 DFormer 仓库根目录执行 MUSeg 数据重建，不创建任何虚拟环境，不运行 conda create、python -m venv 或 virtualenv，直接使用当前 Conda base 环境。

要求：
1. 先检查当前 Python、PyTorch、numpy、cv2 和 CUDA；缺少转换脚本依赖时只安装到 base。
2. 原始数据位于仓库上一级 dataset/MUSeg，禁止删除或修改原始目录。
3. 只使用仓库中的 tools/prepare_museg.py。
4. 如果存在旧的 dataset/MUSeg_DFormer，使用 python tools/prepare_museg.py --overwrite 重建。
5. 不改变 depth-max-raw=13932，不改变官方 Experiment/DatasetSplit.zip，不改变目录命名。
6. 转换后确认 RGB/Depth/Depth16/Label 各 3171 个，train/test 为 1595/1576。
7. 确认 dataset_meta.json 记录统一映射 round(depth16 * 255 / 13932)。
8. 执行脚本内置全量验证；失败时停止并报告错误，不要猜测修复。
9. 训练配置必须指向 dataset/MUSeg_DFormer，不得直接读取原始 MUSeg。
10. 最后运行 git status，确保没有数据集、checkpoint 或大型产物进入 Git。

完成后报告：执行命令、Python 环境、转换统计、验证结果和任何异常。
```

---

## 十一、完成标准

只有同时满足以下条件，才认为云端数据已与本地一致：

- 使用相同 Git 提交中的 `tools/prepare_museg.py`；
- 使用相同官方 `DatasetSplit.zip`；
- `dataset_meta.json` 中划分文件 SHA-256 一致；
- `sample_count=3171`；
- train/test 数量为 `1595/1576`；
- `max_raw=13932`；
- `observed_source_max_raw=13932`；
- 四个输出模态各有 3171 个文件；
- 内置全量验证通过；
- 训练配置指向 `MUSeg_DFormer`；
- Git 中没有数据集和训练产物。

本流程的核心原则是：**原始数据只保留一份，转换过程只保留一份脚本，云端和本地使用同一脚本生成结果。**