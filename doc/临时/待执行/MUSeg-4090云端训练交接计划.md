# MUSeg DFormerv2-S 单卡 RTX 4090 云端训练交接计划

> 文档状态：待执行
>
> 使用方式：在准备好 RTX 4090 有卡实例后，把本文件完整交给新的 Cursor 对话，并明确发送：`已开启有卡模式，请严格按交接计划执行；探测完成后先报告并等待我确认，不要直接开始正式训练。`
>
> 当前阶段只保存计划，不连接云端、不运行 GPU 任务。本文是 `doc/临时` 下唯一有效的待执行计划；旧的数据重建、MVE/A2 筛查和 4090 工具开发计划已经完成并清理。

## 1. 给新对话的强制执行规则

1. 先完整阅读本文以及以下实际入口，不根据旧记忆重写命令：
   - `tools/preflight_train.py`
   - `tools/probe_museg_4090.sh`
   - `tools/train_museg_4090.sh`
   - `local_configs/MUSeg/DFormerv2_S_4090.py`
   - `utils/train.py`
   - `utils/experiment_tracker.py`
   - `.cursor/skills/dformer-plan-execute/SKILL.md`
2. 云端固定项目目录是 `~/rivermind-data/DFormer#`，末尾 `#` 是目录名的一部分；Shell 中必须整体加引号。
3. 固定使用 `/usr/local/miniconda3/envs/py310/bin/python`，不创建新 Conda、venv 或 virtualenv，不盲目替换已有 PyTorch/CUDA。
4. 在用户明确开启有卡模式前不执行本文；开启后按“同步→预检→batch 探测→用户确认→正式训练→验收”的顺序执行。
5. batch 探测结束后必须先向用户报告并等待确认，不能自动开始正式训练。
6. 不删除或修改原始数据，不把数据集、预训练权重、checkpoint、日志、SwanLab 凭据或大型输出加入 Git。
7. 遇到路径不一致、云端工作区不干净、非 fast-forward、GPU 不是 RTX 4090、preflight error、OOM、非有限 loss、CUDA Xid、数据进程退出或长时间无进展时停止并报告，不猜测修复。
8. 若确需修改代码或脚本，先报告原因和最小修改方案；修改后必须独立验证并单独提交，不能在云端留下未提交补丁后继续正式训练。

## 2. 当前项目基线

- 仓库：DFormer，目标数据集为 MUSeg，目标模型为 DFormerv2-S，单张 NVIDIA GeForce RTX 4090。
- 当前训练代码基线为提交 `bb44831`（`skill优化`）；本文档清理提交位于其后，执行时以 `git pull --ff-only origin main` 后的实际 `HEAD` 为准。
- 4090 训练准备提交为 `b290410`（`准备 MUSeg 单卡 4090 训练与监控`），已包含在 `bb44831` 中。
- 开始本次临时计划清理前，本地 `main` 与 GitHub `origin/main` 对齐且工作区干净；本文只引入文档清理与新交接计划。
- 本地数据默认位于项目上一级 `../dataset/MUSeg_DFormer`；云端默认数据根为 `~/rivermind-data/dataset`，训练目录为 `~/rivermind-data/dataset/MUSeg_DFormer`。
- MUSeg 共 3171 个样本：`train.txt=1595`、`test.txt=1576`。
- 当前训练输入为：
  - `RGB/*.jpg`
  - `Depth/*.png`（8-bit，交给当前 loader）
  - `Label/*.png`
- `Depth16/*.png` 仅保留原始深度和供研究使用，不直接交给当前默认训练 loader。
- 预训练权重默认云端路径：`~/rivermind-data/pretrained/DFormerv2_Small_pretrained.pth`。
- 已知预训练权重验收值：
  - 大小：`110203103` bytes
  - SHA-256：`19116988fc86dc9f3e879282237941e11b9b1b5c480edb51e92807311dbc11a6`
- 输出根目录默认：`~/rivermind-data/dformer_outputs`。
- SwanLab 默认项目：`DFormer-liu`；默认工作区：`Newton_liub`；固定依赖为 `swanlab==0.9.7`。
- SwanLab online 由自定义训练循环直接接入，仅 rank 0 初始化；采用非交互设置，online 初始化失败会在训练开始前失败，不允许静默降级到 offline。

## 3. 已经完成的工作

### 3.1 数据与历史实验

- MUSeg 数据转换流程、官方划分、RGB/Depth/Depth16/Label 模态和数量已完成全量验收。
- 固定深度映射为 `round(depth16 * 255 / 13932)`；原始深度 0 保持为 0；不允许逐图 min-max 或分别按 train/test 量化。
- A1 已复现：全背景样本映射为全 ignore 时，旧 masked mean 存在空集合非有限风险。
- B1 已完成：`safe_masked_mean` 已接入主头和辅助头，空有效像素时返回保持计算图连接的有限零 loss；非空条件与旧实现一致。
- 历史 epoch-10 MVE screening 已完成，验证 mIoU 为 `20.49`；它不是正式 20-epoch baseline。
- 历史 A2 已完成 16 图、q=0/0.3/0.5 共 48 个条件：退化随 q 单调增加但幅度很小，未达到触发 B2 validity mask/gating 的预注册门槛。
- 历史 A1/B1/A2 证据保留在：
  - `doc/reports/2026-08-21-museg-mve-a1-b1-a2-git-alignment.md`
  - `doc/reports/2026-08-21-museg-mve-cleanup-and-disposition.md`
  - `cloud/museg-epoch10-a2-20260821`（本地忽略目录，不进入 Git）

### 3.2 4090 正式训练准备

- 已新增可移植配置 `local_configs/MUSeg/DFormerv2_S_4090.py`：
  - AdamW，`lr=6e-5`，`weight_decay=0.01`，poly power 0.9；
  - 候选 `batch_size=8`，`val_batch_size=1`，`num_workers=8`；
  - `nepochs=20`，warmup 2 epochs，训练尺度固定 1.0；
  - 每 5 epochs 保存 checkpoint，并维护 `latest.pth`。
- 已新增 `tools/preflight_train.py`，检查依赖、配置导入、数据路径/划分/抽样解码、预训练权重、输出目录、CUDA/RTX 4090 和 SwanLab。
- 已新增 `tools/probe_museg_4090.sh`，按 batch 4/8/12/16 各探测 60 步，启用 AMP，关闭 compile、多尺度验证、SyncBN 和 sliding；显存安全阈值为至少 2 GiB 且至少 10% free VRAM。
- 已新增 `tools/train_museg_4090.sh`，使用单卡 AMP 和 SwanLab online 启动正式训练，正式验证默认单尺度。
- `utils/train.py` 已支持 epoch、batch、worker、验证 batch、checkpoint 目录、有限训练步数和日志间隔覆盖；学习率在 optimizer step 前更新。
- 训练已记录 loss、学习率、step time、吞吐、allocated/reserved/free VRAM、free ratio、AMP scale，以及 epoch 与验证指标；非有限 loss 和显存安全阈值可快速失败。
- 详细 GPU 同步和显存采样仅在日志采样步或安全探测步执行，避免正常训练每一步同步。
- 本地验证已完成：Python compile、两个 Shell 脚本语法、diff whitespace、换行检查和 `tests` 共 18 项测试均通过。
- 相关实现已提交并推送，不存在“本地改好但云端拉不到”的待处理代码。

## 4. 后续不需要重复的检查

以下内容已有固定证据，除非相关代码、数据或官方划分发生变化，否则新对话不要重新开展同类调查或重建：

1. 不重新研究 MUSeg 目录结构、样本总数和 1595/1576 官方划分。
2. 不重新转换 `MUSeg_DFormer`，不运行 `tools/prepare_museg.py --overwrite`，除非云端 preflight 明确发现数据缺失或损坏并经用户确认。
3. 不重新推导深度最大值 13932 或量化公式，不比较逐图 min-max 等已排除方案。
4. 不重复 A1 空集合风险复现、B1 safe masked loss 设计和张量级单元测试论证。
5. 不重复历史 16 图 A2 q=0/0.3/0.5 screening，也不直接开发 B2；只有用户之后明确启动扩大 A2 研究且达到门槛，才另立计划。
6. 不重新设计 SwanLab 接入，不改用 MMCV/MMEngine Runner、WandB hook 或 `SwanlabVisBackend`；当前项目使用 `utils/train.py` 自定义循环和 `utils/experiment_tracker.py`。
7. 不重复本地 compile、18 项单元测试和 Shell 语法检查，除非新对话修改了受影响代码。
8. 不重复讨论云端项目目录；唯一项目目录是 `~/rivermind-data/DFormer#`，不得改用 `~/DFormer` 或不带 `#` 的旧路径。

以下检查虽然本地或历史上做过，仍必须在新的有卡实例上执行，因为它们依赖本次云端运行时状态：

- 云端 Git 工作区和实际 commit；
- `py310` 中的实际包版本与 CUDA 可用性；
- 当前 GPU 型号和显存；
- 云端数据路径及抽样可读性；
- 当前预训练权重文件及哈希；
- 当前输出目录可写性；
- 当前 SwanLab 凭据和 online 初始化；
- 当前 4090 对新训练链路的真实 batch、吞吐和显存余量。

## 5. 尚未完成的核心目标

1. 尚未在当前 `DFormerv2_S_4090` + 新监控链路上完成 batch 4/8/12/16 的真实 RTX 4090 探测。
2. 正式 batch 尚未确定；配置中的 8 只是候选值。
3. 尚未使用当前统一 Git 提交完成 SwanLab online 的正式 20-epoch MUSeg baseline。
4. 尚未验收正式运行产生的 `epoch-20.pth`、完整 test 指标、SwanLab 页面和最终实验记录。
5. 当前探测脚本没有实现旧计划中提到的“120 秒无进展自动停止”；执行时需要人工监控。若需要自动 watchdog，必须先提出最小代码修改并等待用户确认。
6. 正式训练脚本当前未暴露 checkpoint 恢复环境变量；若训练中断，先报告可用 checkpoint 和恢复方案，不要擅自改脚本或自动重启。

## 6. 有卡模式开启后的执行计划

### 阶段 0：等待执行授权

在用户明确发送“已开启有卡模式，开始执行本计划”之前停止在这里，不连接云端、不安装依赖、不运行命令。

### 阶段 1：云端同步与环境留档

在云端执行：

```bash
cd "$HOME/rivermind-data/DFormer#"
git status --short
git branch --show-current
git rev-parse HEAD
git pull --ff-only origin main

git rev-parse HEAD
/usr/local/miniconda3/envs/py310/bin/python --version
/usr/local/miniconda3/envs/py310/bin/python -c "import sys; print(sys.executable)"
nvidia-smi
```

要求：

- `git status --short` 必须为空；非空时停止并报告。
- 分支必须为 `main`，拉取只能 fast-forward。
- 拉取后至少包含 `bb44831` 和 `b290410`；若 `origin/main` 已有更新，记录新 commit 并检查训练相关 diff 后再继续。
- GPU 必须为 RTX 4090 且显存不少于 20 GiB。

安装 SwanLab 依赖：

```bash
PYTHON=/usr/local/miniconda3/envs/py310/bin/python
"$PYTHON" -m pip install --index-url https://pypi.org/simple -r requirements-monitoring.txt
"$PYTHON" -m pip check
```

设置路径与凭据。API key 只通过环境提供，不回显、不写文件：

```bash
export PYTHON_BIN=/usr/local/miniconda3/envs/py310/bin/python
export CUDA_VISIBLE_DEVICES=0
export DFORMER_DATA_ROOT="$HOME/rivermind-data/dataset"
export DFORMER_PRETRAINED="$HOME/rivermind-data/pretrained/DFormerv2_Small_pretrained.pth"
export DFORMER_OUTPUT_ROOT="$HOME/rivermind-data/dformer_outputs"
export SWANLAB_API_KEY='由用户在云端安全设置'
```

上面的值是说明性占位符，不能原样执行；新对话必须等待用户通过安全方式在云端设置真实 key，且不得读取、回显或写入计划/日志。
记录但不要上传敏感信息：Git commit、Python/PyTorch/CUDA/cuDNN、驱动、GPU、包版本和输出根目录。

### 阶段 2：强制 preflight

先校验预训练权重：

```bash
stat -c '%s %n' "$DFORMER_PRETRAINED"
sha256sum "$DFORMER_PRETRAINED"
```

预期大小和 SHA-256 必须与第 2 节一致。

执行：

```bash
"$PYTHON_BIN" tools/preflight_train.py --swanlab-mode online
```

继续条件：

- preflight 为 0 error；
- RTX 4090、数据 1595/1576、抽样三模态、权重、输出目录、必需包和 SwanLab 均通过；
- warning 必须逐项报告，不能自动忽略。

preflight 失败时停止，不进入 batch 探测。

### 阶段 3：batch 4/8/12/16 探测

执行：

```bash
export DFORMER_PROBE_STEPS=60
export DFORMER_WORKERS=8
export DFORMER_PROBE_MIN_FREE_GIB=2
export DFORMER_PROBE_MIN_FREE_RATIO=0.10
bash tools/probe_museg_4090.sh
```

脚本会把日志写入默认 `outputs/museg_4090_probe/batch-*`；如需持久化到数据盘，先设置：

```bash
export DFORMER_PROBE_OUTPUT_ROOT="$HOME/rivermind-data/dformer_outputs/museg_4090_probe"
```

每档必须汇总：

- 是否完成 60 步及退出码；
- 预热后的稳定吞吐；
- 峰值 allocated 和 reserved VRAM；
- 最低 free VRAM 与 free ratio；
- loss 是否有限、是否异常波动；
- AMP scale 是否稳定；
- 是否出现 OOM、CUDA、数据加载或长时间无进展问题。

选择规则：在 loss/AMP 稳定且至少保留 2 GiB 和 10% 显存余量的候选中，选择稳定吞吐最高的 batch；不能只按“能跑的最大 batch”选择。

**强制暂停点：** 探测结束后向用户给出四档对比、推荐 batch 和理由，等待用户明确确认。不得直接运行正式训练。

### 阶段 4：正式 20-epoch SwanLab online 训练

仅在用户确认 batch 后设置，例如确认 batch 12：

```bash
export DFORMER_BATCH_SIZE=12
export DFORMER_VAL_BATCH_SIZE=1
export DFORMER_WORKERS=8
export DFORMER_EPOCHS=20
export DFORMER_MST=0
export DFORMER_LOG_INTERVAL=20
export SWANLAB_RUN_NAME='可选的明确实验名'

mkdir -p "$DFORMER_OUTPUT_ROOT/museg_dformerv2_s_4090"
bash tools/train_museg_4090.sh 2>&1 | tee "$DFORMER_OUTPUT_ROOT/museg_dformerv2_s_4090/launcher.log"
status=${PIPESTATUS[0]}
printf '%s\n' "$status" > "$DFORMER_OUTPUT_ROOT/museg_dformerv2_s_4090/train.exit_code"
exit "$status"
```

启动验收：

- SwanLab online 初始化成功并返回实验页面；
- 实验配置中记录正确 commit、数据集、backbone、20 epochs、确认后的 batch、worker、AMP 和输出目录；
- 前 60 步 loss 有限，AMP scale 无持续崩落，吞吐和显存稳定。

持续监控：

- `train/loss`、`train/loss_mean`、`train/learning_rate`；
- `train/step_seconds`、`train/images_per_second`；
- allocated/reserved/free VRAM、free ratio、AMP scale；
- epoch loss/time；
- validation mIoU、mAcc、mF1、best mIoU；
- epoch 5/10/15/20 checkpoint 和 `latest.pth`。

停止条件：OOM、非有限 loss、持续 AMP scale 崩落、CUDA Xid、吞吐持续异常下降、数据进程退出、SwanLab 或训练进程中断。停止后保留日志和 checkpoint，不自动重启。

### 阶段 5：checkpoint 与结果验收

训练完成后检查：

```bash
find "$DFORMER_OUTPUT_ROOT/museg_dformerv2_s_4090" -maxdepth 2 -type f -printf '%p %s bytes\n' | sort
```

使用 CPU 方式检查 checkpoint，不把文件加入 Git：

```bash
"$PYTHON_BIN" - <<'PY'
from pathlib import Path
import os
import torch

root = Path(os.environ["DFORMER_OUTPUT_ROOT"]) / "museg_dformerv2_s_4090" / "checkpoint"
for path in sorted(root.glob("*.pth")):
    obj = torch.load(path, map_location="cpu")
    print(path, path.stat().st_size, sorted(obj.keys()), obj.get("epoch"), obj.get("iteration"))
PY

sha256sum "$DFORMER_OUTPUT_ROOT"/museg_dformerv2_s_4090/checkpoint/*.pth
```

最终报告必须包含：

- 实际 Git commit 和完整启动命令；
- Python/PyTorch/CUDA/驱动/GPU；
- 最终 batch、worker、epoch、AMP、多尺度设置；
- preflight 结果和四档 batch 探测对比；
- SwanLab 实验链接；
- 最终与 best mIoU、mAcc、mF1；
- 总耗时、稳定吞吐、峰值显存和最低显存余量；
- checkpoint 路径、大小、epoch/iteration 和 SHA-256；
- `train.exit_code`；
- `git status --short`，确认 Git 中没有数据、权重、凭据或训练产物。

## 7. 中断恢复规则

- 优先选择最高编号且能用 `torch.load(..., map_location="cpu")` 正常读取的 `epoch-N.pth`，其次才考虑 `latest.pth`。
- 恢复前先报告原运行 commit、参数、已完成 epoch、checkpoint 哈希、日志最后位置和 SwanLab 状态。
- 当前 `tools/train_museg_4090.sh` 没有暴露恢复参数；不要直接编辑云端脚本。先向用户提出最小改动或一次性受控命令，得到确认后再执行。
- 恢复后必须明确这是新的 SwanLab run 还是原 run 延续；不得把两段运行误写为单一连续实验。

## 8. 本计划完成标准

- 当前阶段：仅保存本文件，未执行任何有卡操作。
- 有卡模式后：云端 preflight 零错误；真实探测确定并由用户确认正式 batch。
- SwanLab online 从训练开始持续记录训练、性能、显存和验证指标。
- 正式 20-epoch baseline 完成，`epoch-20.pth` 和完整 test 指标通过验收；若中断，则按受控恢复规则处理。
- 所有结论均可由 commit、命令、日志、SwanLab 页面和 checkpoint 哈希追溯。
- Git 中不包含数据集、模型权重、checkpoint、API key 或大型训练输出。
