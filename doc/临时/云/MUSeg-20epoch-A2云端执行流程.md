# MUSeg 20 Epoch 与 A2 云端执行流程

> 目标：在固定代码版本和 `py310` 环境中完成 DFormerv2-S 的 MUSeg 15 类 20 epoch 训练，随后执行 A2 筛查。
>
> 授权状态：用户已授权启动云端任务，并要求 checkpoint、日志、A2 结果验收后自动停止实例。预训练权重仅用于 backbone 初始化，不得作为 A2 checkpoint。

## 一、固定资源

- 实例：`cpod-1tyvjsiu6ahe`，单张 RTX 4090。
- Python：`/usr/local/miniconda3/envs/py310/bin/python`，Python `3.10.16`。
- 数据：`/root/rivermind-data/dataset/MUSeg_DFormer`，3171 样本，train/test 为 1595/1576。
- 训练配置：`local_configs.MUSeg.DFormerv2_S_20Epoch`。
- 训练输出：`/root/rivermind-data/mve_outputs/museg_20epoch`。
- A2 输出：`/root/rivermind-data/mve_outputs/museg_a2_screening`。
- Git 基线：`11c20fcf0e670aee764c17b84cdc96afbd9fe327`；实际训练必须记录包含 20 epoch 配置的新 commit。

## 二、预训练权重

固定 URL：

`https://huggingface.co/bbynku/DFormerv2/resolve/48a60260aa3e0696be0b51e202e7d1fdda7d7e8c/DFormerv2/pretrained/DFormerv2_Small_pretrained.pth`

验收值：

- 路径：`/root/rivermind-data/pretrained/DFormerv2_Small_pretrained.pth`
- 文件大小：`110203103` bytes
- SHA-256：`19116988fc86dc9f3e879282237941e11b9b1b5c480edb51e92807311dbc11a6`

该文件是 ImageNet 预训练 backbone，不是 MUSeg 分割 checkpoint。只有训练输出的完整 checkpoint 才能用于 A2。

## 三、执行阶段

### 阶段 1：代码同步与权重校验

1. 本地提交并推送受 Git 跟踪的训练配置、checkpoint 保存逻辑和本文档。
2. 云端仅允许 fast-forward 到该精确 commit，随后确认工作树干净。
3. 下载固定版本预训练权重并校验大小、SHA-256。

状态：**进行中**。

任务记录：

- 新 Git commit：待本地提交后回填。
- 权重下载任务：待提交后回填。

### 阶段 2：单卡 dry-run

必须依次验证：

- `torch/cv2/numpy` 可导入，CUDA 可用，GPU 为 RTX 4090；
- 训练配置解析为 15 类、20 epoch、持久化输出路径；
- ImageNet 权重可加载到 DFormerv2-S backbone；
- 一个真实 batch 可完成 forward、有限 loss、backward；
- batch size `2` 的显存满足训练要求。

任何一项失败均不得启动 20 epoch 训练。若仅因显存不足，新增并提交更小 batch 的配置后重新同步，不在云端手工修改配置。

状态：**未完成**。

任务记录：

- dry-run 任务：待提交后回填。

### 阶段 3：20 epoch 训练

训练必须在命名的 detached `screen` 会话中启动，不能把训练进程直接挂在 SSH 前台。受 Git 跟踪的启动器会把 stdout/stderr 和退出码写入持久化目录：

```bash
cd /root/DFormer
bash tools/mve/run_museg_20epoch_screen.sh museg20
```

连接断开不会终止训练。重新连接后的检查命令：

```bash
screen -ls
screen -r museg20
```

查看后使用 `Ctrl+A`、`D` 脱离会话，不使用 `Ctrl+C`。无需进入会话时可直接查看持久化日志：

```bash
tail -n 100 -f /root/rivermind-data/mve_outputs/museg_20epoch/screen-train.log
```

启动器内部的单卡参数固定为：

```bash
CUDA_VISIBLE_DEVICES=0 LOCAL_RANK=0 /usr/local/miniconda3/envs/py310/bin/python utils/train.py \
  --config local_configs.MUSeg.DFormerv2_S_20Epoch \
  --gpus 1 \
  --no-syncbn \
  --no-sliding \
  --no-compile \
  --no-mst \
  --amp \
  --val_amp \
  --no-pad_SUNRGBD \
  --use_seed
```

训练退出码写入 `/root/rivermind-data/mve_outputs/museg_20epoch/train.exit_code`。只有该值为 `0` 且 checkpoint 严格验收通过，才进入 A2。

配置每 5 epoch 保存完整 checkpoint，并更新 `checkpoint/latest.pth`；第 20 epoch 的确定路径是：

`/root/rivermind-data/mve_outputs/museg_20epoch/checkpoint/epoch-20.pth`

训练验收必须包括：

- 日志出现 `Epoch 20/20` 且任务成功退出；
- `epoch-20.pth` 与 `latest.pth` 均存在、非空；
- checkpoint 包含 `model`、`optimizer`、`epoch`、`iteration`；
- `epoch == 20`，模型头输出类别数为 15；
- checkpoint 可在 CPU 上重新加载，参数无缺失或意外 key。

状态：**未完成**。

任务记录：

- 训练任务：待提交后回填。
- 首 epoch 与总耗时：待训练后回填。

### 阶段 4：A2 baseline 与筛查

先基于云端数据确定性重建 mask manifest：16 张官方 test 样本，seed `20260819`，q 为 `0/0.3/0.5`，block mask。

“48 条件”定义为 16 张图 × 3 个 q，共 48 个样本-条件前向：

- q=0 baseline：16 个；
- q=0.3：16 个；
- q=0.5：16 个；
- 两个受损 q 合计 32 个，不能误写为各 48 个。

执行顺序：

1. 仅运行 q=0，验收 16 张预测、CSV 和汇总 JSON；
2. q=0 成功后运行 q=0.3、q=0.5；
3. 运行 `tools/mve/a2_evaluate_results.py` 独立汇总并记录相对 baseline 的变化。

A2 必须使用 `epoch-20.pth`，配置使用 `local_configs.MUSeg.DFormerv2_S_MVE`，避免再次加载 ImageNet 预训练权重。

状态：**阻断**，等待 20 epoch checkpoint 验收。

任务记录：

- mask 任务：待提交后回填。
- q=0 任务：待提交后回填。
- q=0.3/q=0.5 任务：待提交后回填。
- 评估任务：待提交后回填。

### 阶段 5：实例停止

只有以下产物全部位于 `/root/rivermind-data` 且验收成功后，才执行独立控制命令停止实例：

- 训练日志、`epoch-20.pth`、`latest.pth`；
- A2 manifest、mask、48 个预测、CSV、summary；
- 独立评估结果。

停止命令：

```bash
compshare --json instance stop cpod-1tyvjsiu6ahe --yes --timeout 600
```

训练或 A2 失败时保留实例和日志，不自动停止，先报告失败位置。停止后必须再次查询实例状态，不能仅以 stop 命令返回作为最终证据。

状态：**未完成**。

## 四、完成判据

- [ ] 云端检出并核对包含训练配置的新 commit，工作树干净。
- [ ] 固定预训练权重大小和 SHA-256 完全匹配。
- [ ] 单卡真实 batch dry-run 完成，loss 与梯度有限。
- [ ] 20 epoch 训练任务成功，完整 checkpoint 可严格重载。
- [ ] q=0 baseline 的 16 个条件完成。
- [ ] q=0.3/q=0.5 的 32 个条件完成；三组总计 48 个条件。
- [ ] 独立评估完成，结果写入持久化目录。
- [ ] 文档回填 commit、任务 ID、路径、耗时和结果。
- [ ] 实例停止并确认状态。