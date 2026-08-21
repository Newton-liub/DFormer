# MUSeg Epoch 10 MVE 与 A2 云端执行流程

> 目标：使用 DFormerv2-S 的 epoch-10 checkpoint 快速完成 MUSeg 15 类 MVE/A2 筛查，验证 Depth block mask 扰动是否产生可测的性能变化；本流程不是 20 epoch 正式训练结果。
>
> 授权状态：用户已授权以 epoch-10 checkpoint 完成 MVE/A2。已停止继续到 epoch 20 的训练和原先等待 epoch-20 的后处理任务。预训练权重仅用于 backbone 初始化，不得作为 A2 checkpoint。

## 一、固定资源

- 实例：`cpod-1tyvjsiu6ahe`，单张 RTX 4090。
- Python：`/usr/local/miniconda3/envs/py310/bin/python`，Python `3.10.16`。
- 数据：`/root/rivermind-data/dataset/MUSeg_DFormer`，3171 样本，train/test 为 1595/1576。
- 训练配置：`local_configs.MUSeg.DFormerv2_S_20Epoch`（实际运行至 epoch 10，作为 MVE screening）。
- MVE checkpoint：`/root/rivermind-data/mve_outputs/museg_20epoch/checkpoint/epoch-10.pth`。
- MVE 验证指标：epoch 10 全量验证 mIoU `20.49`。
- 训练输出：`/root/rivermind-data/mve_outputs/museg_20epoch`。
- A2 输出：`/root/rivermind-data/mve_outputs/museg_a2_screening`。
- Git 基线：`11c20fcf0e670aee764c17b84cdc96afbd9fe327`；本次云端实际训练任务已使用包含 20 epoch 配置的工作树，但精确 commit 尚未回填。

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

状态：**已完成（云端实际训练与 checkpoint 已验证；精确 commit 和历史任务记录仍待补录）**。

任务记录：

- 新 Git commit：`<待补录>`。
- 权重下载任务：固定权重已用于训练，独立任务 ID `<待补录>`。

### 阶段 2：单卡 dry-run

必须依次验证：

- `torch/cv2/numpy` 可导入，CUDA 可用，GPU 为 RTX 4090；
- 训练配置解析为 15 类、20 epoch、持久化输出路径；
- ImageNet 权重可加载到 DFormerv2-S backbone；
- 一个真实 batch 可完成 forward、有限 loss、backward；
- batch size `2` 的显存满足训练要求。

任何一项失败均不得启动 20 epoch 训练。若仅因显存不足，新增并提交更小 batch 的配置后重新同步，不在云端手工修改配置。

状态：**已完成（epoch 10 screening 前置验证通过）**。

任务记录：

- dry-run 任务：`job-20260819T140545Z-f8db4b88`（已成功）。
- 训练实际使用 `batch_size=2`、`num_workers=2`、AMP、单卡 RTX 4090。

### 阶段 3：Epoch 10 MVE screening 训练

原计划的 20 epoch 训练已运行至 epoch 10 并完成验证。为快速完成 MVE/A2，已取消继续训练到 epoch 20 的远程任务，保留所有已生成产物，不再等待 `epoch-20.pth`。

原训练启动命令仍记录如下，便于追溯：

```bash
cd /root/DFormer
bash tools/mve/run_museg_20epoch_screen.sh museg20
```

实际训练参数：

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

MVE checkpoint 验收结果：

- 训练日志完成 `Epoch 10/20`；
- epoch 10 全量验证完成，`mIoU = 20.49`；
- `/root/rivermind-data/mve_outputs/museg_20epoch/checkpoint/epoch-10.pth` 存在且非空；
- `/root/rivermind-data/mve_outputs/museg_20epoch/checkpoint/latest.pth` 已更新；
- `/root/rivermind-data/mve_outputs/museg_20epoch/epoch-10_miou_20.49.pth` 已保存；
- checkpoint 使用 15 类 MUSeg 配置，作为 MVE screening checkpoint；
- 该 checkpoint 不是完整 20 epoch 正式训练结果，不用于宣称最终模型性能。

任务记录：

- 训练启动任务：`job-20260819T141557Z-128ce01f`，已取消，退出码 `137`；
- 原 A2 等待任务：`job-20260819T141714Z-db0285c3`，已取消，退出码 `137`；
- 取消时训练进程及 GPU 占用已确认停止；
- 实际 MVE checkpoint：`epoch-10.pth`，验证 mIoU `20.49`。

状态：**已完成（MVE screening checkpoint 已验收）**。

### 阶段 4：A2 baseline 与筛查

使用 epoch-10 MVE screening checkpoint，配置使用 `local_configs.MUSeg.DFormerv2_S_MVE`，不再加载 ImageNet 预训练权重。

先基于云端数据确定性重建 mask manifest：16 张官方 test 样本，seed `20260819`，q 为 `0/0.3/0.5`，block mask。

“48 条件”定义为 16 张图 × 3 个 q，共 48 个样本-条件前向：

- q=0 baseline：16 个；
- q=0.3：16 个；
- q=0.5：16 个；
- 两个受损 q 合计 32 个，不能误写为各 48 个。

执行顺序：

1. 生成并验收 16 张样本的 manifest 与 mask；
2. 仅运行 q=0，验收 16 张预测、CSV 和汇总 JSON；
3. q=0 成功后运行 q=0.3、q=0.5；
4. 运行 `tools/mve/a2_evaluate_results.py` 独立汇总并记录相对 baseline 的变化。

MVE 解释边界：A2 结果用于快速验证扰动实验流程和相对变化方向；由于 checkpoint 仅训练至 epoch 10，不能替代完整 20 epoch 正式结果。

状态：**已完成（q=0、q=0.3、q=0.5 三组预测均验收）**。

任务记录：

- checkpoint：`/root/rivermind-data/mve_outputs/museg_20epoch/checkpoint/epoch-10.pth`；
- mask 任务：`job-20260819T152827Z-3b37377b`，状态 `Succeeded`，manifest 与 q=0/q=0.3/q=0.5 block mask 已生成；
- q=0 首次任务：`job-20260819T152934Z-62543eba`，状态 `Failed`，stderr 为 `ModuleNotFoundError: No module named 'models'`；
- q=0 重跑任务：`job-20260819T153307Z-6766ff62`，状态 `Succeeded`；
- q=0 产物：`q_0_block/` 共 16 张预测，前景 mIoU `0.3389857875225665`；
- q=0.3 任务：`job-20260819T153505Z-3eae5fc9`，状态 `Succeeded`，前景 mIoU `0.3378786890205167`，`q_0.3_block/` 已验收 16 张预测；
- q=0.5 并行任务：`job-20260819T153505Z-eaff689b`，任务状态 `Succeeded` 但输出被并发写入冲突覆盖，不作为产物验收依据；
- q=0.5 串行补跑任务：`job-20260819T153709Z-3c1fed95`，状态 `Succeeded`，前景 mIoU `0.33564901872796205`，`q_0.5_block/` 已验收 16 张预测；
- 当前已验收三组共 48 张预测；
- 评估任务：`job-20260819T153832Z-c985b24f`，状态 `Succeeded`；
- 独立评估产物：`/root/rivermind-data/mve_outputs/museg_a2_screening/evaluation/per_image_metrics.csv`（49 行，含表头）与 `summary.json`；
- 独立评估前景 mIoU：q=0 为 `0.3389857875225665`，q=0.3 为 `0.3378786890205167`，q=0.5 为 `0.33564901872796205`；
- 相对 q=0：q=0.3 变化 `-0.0011070985020498`（`-0.1107` 个百分点），q=0.5 变化 `-0.00333676879460445`（`-0.3337` 个百分点）；
- 解释：16 图 MVE screening 中随 block 缺失率增加呈单调下降，但下降幅度较小；不能据此替代完整 20 epoch 正式结论。

### 阶段 5：实例停止

以下产物已全部位于 `/root/rivermind-data` 且通过验收：

- `screen-train.log`、`train.exit_code=0`、`epoch-10.pth`、`latest.pth`；
- A2 manifest、三组 block mask、48 个预测、推理 CSV/summary；
- 独立评估 CSV 与 summary；
- 停止前 `nvidia-smi` 无运行中的计算进程。

停止命令：

```bash
compshare --json instance stop cpod-1tyvjsiu6ahe --yes --timeout 600
```

训练或 A2 失败时保留实例和日志，不自动停止，先报告失败位置。停止后必须再次查询实例状态，不能仅以 stop 命令返回作为最终证据。

状态：**已完成，实例已停止并确认状态为 `Stopped`**。

停止记录：

- 停止命令返回成功，`failed=[]`；
- 停止后查询：实例 `cpod-1tyvjsiu6ahe`，区域 `cn-bj2`，状态 `Stopped`；
- 停止时间：`1787154047`（云端记录）。

## 四、完成判据

- [ ] 云端检出并核对包含训练配置的新 commit，工作树干净（精确 commit 尚未补录）。
- [x] 固定预训练权重大小和 SHA-256 完全匹配。
- [x] 单卡真实 batch dry-run 完成，loss 与梯度有限。
- [x] epoch-10 MVE screening 训练产物成功，`epoch-10.pth` 与 `latest.pth` 可用；不宣称完成 20 epoch 正式训练。
- [x] q=0 baseline 的 16 个条件完成。
- [x] q=0.3/q=0.5 的 32 个条件完成；三组总计 48 个条件。
- [x] 独立评估完成，结果写入持久化目录。
- [x] 文档已回填任务 ID、路径和结果；代码 commit 仍待补录。
- [x] 实例停止并确认状态。