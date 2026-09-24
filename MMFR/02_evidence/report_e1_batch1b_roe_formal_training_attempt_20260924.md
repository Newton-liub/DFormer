# MMFR E1 Batch 1B R-OE-lite 云端正式训练中止记录

- 日期：2026-09-24
- 状态：`blocked-oom-before-completion`
- 训练身份：`MMFR-E1-Batch1B-R-OE-lite-v1`，seed `772961337`
- Git HEAD：`789852be1d817c1b60d23ed002321619ffc63b7c`
- 云端硬件：NVIDIA RTX 4090；Python `3.10.16`、PyTorch `2.1.2+cu118`、CUDA `11.8`
- 配置：原冻结 config；20 epochs × 128 iterations；batch size 10；workers 8；AMP/SyncBN 开启；单 GPU、DDP 关闭；source 为 A2 epoch-420，SHA-256 `2b72eb7f28e5c4b4f52bdc43e2ac5c7298169325a412bfbcbc7a022c29822597`
- 输出目录：`cloud/mmfr-e1-batch1b-roe-v1/R-OE-lite/development/seed-772961337`
- SwanLab online run：[`mmfr-e1-batch1b-roe-seed772961337-4090`](https://swanlab.cn/@Newton_liub/DFormer-liu/runs/er8ao0f2)

**结果与停止原因：** 正式训练于 `2026-09-24 01:40:43 UTC` 开始，于 `01:45:30 UTC` 遇到 CUDA OOM 并退出，约 `4 分 47 秒`。异常发生在 `models/roe_substitute.py:85` 的 `F.interpolate` forward：当时尝试分配 `2.73 GiB`，设备仅剩 `198.56 MiB`；进程使用 `23.34 GiB / 23.54 GiB`。训练父进程 PID `14335` 与 screen 会话 `14333.mmfr-e1-batch1b-roe` 均已结束。OOM 后只读核查的主机为 `cpod-1vbh7faqcauq`，RTX 4090 显存占用 `1 MiB / 24564 MiB`、利用率 `0%`。当前环境未安装 `compshare-cli` 且没有控制面凭据，云实例是否已停止待核验；GPU 空闲不代表实例停机。

**更新计数边界：** 日志最后确认完成 `Epoch 2/20 Iter 40/128`，累计至少 `168` 次成功 optimizer update（epoch 1 的 128 次加 epoch 2 已记录的 40 次）。OOM 发生后，attempted/completed 的精确最终计数没有落盘，不能把 `168` 当作最终精确总数。OOM 前已记录 loss finite、AMP scale `1024`；没有记录 NaN 或 AMP skip。最近一条定期日志为 epoch 2 iter 40：batch loss `0.2024`、running total loss `0.2365`。

**产物与边界：** 没有达到 `2560/2560`，fixed final `update-2560.pth` 未生成；Quick-Val 未运行，official test 未读取且仍为 `sealed_unread`。未重跑 Gate-B、未重训 C0、未改 batch size、precision、LR、模型结构或冻结合同；OOM 后没有重启训练。训练原始输出日志位于云端输出目录的 `train.log`，SwanLab run 保存在线记录。

**matched C0 checkpoint 搜索：** 在三个文档记录候选路径及当前 `/root`、`/mnt`、`/workspace` 下按 `update-2560.pth` 搜索未找到；对根目录 `/` 的广域搜索在工具时限内超时，因此真实路径仍为“待核验”。该 checkpoint 未被重训或替代，Quick-Val 未启动。

**大白话说明：** R-OE-lite 确实进入了正式训练并完成至少 168 个更新，但 RTX 4090 在 substitute 的双线性插值 forward 中显存不足。按冻结规则这次运行停止，不能通过缩小 batch 或改模型来补救；目前没有 fixed-final checkpoint，也没有效果评价。
