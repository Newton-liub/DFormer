# MUSeg Stage-05 development seed 1 启动前报告

> **历史启动前快照：** 本报告准确记录 v1 protocol 尚未启动时的状态。后续已在提交 `56a7ed7...` 上重新物化 `museg-development-long500-v2` 并启动 seed 1；seed 1 后来已完成，当前状态见 `doc/main/MUSeg-current-status.md`，完成后的详细进度见 `doc/reports/2026-08-26-museg-stage05-seed1-local-closure-handoff.md`。
>
> **口径勘误：** 正文“输入 640×480”只表示训练随机裁剪尺寸，不表示 seed 1 的 validation 统一 resize 到 640×480；validation 几何事实与待定方案见 `doc/main/MUSeg-open-decisions.md`。其余指标、哈希和形成时点事实保持不变。

- 生成时间：2026-08-25
- 当前状态：**技术 preflight 通过；训练未启动；等待用户手动授权与启动**
- official test：保持 `sealed_unread`；本轮未读取、未抽样、未评估
- 仓库提交：`56544eacf83b535c49095d7b21060634f36fe1f3`
- 协议：`museg-development-long500-v1`
- 协议 SHA-256：`1ecdc05528bf90de26b980675a4dcc21197d842ff2e8753542e916538990f7e0`

## 1. Stage-04 Gate D 证据结论

Stage-04 计划文档中的早期交接状态已经落后于数据盘上的最终结构化证据。当前 Gate D 可用证据绑定同一提交与 qualification protocol：

- batch 10 的连续 3-epoch qualification 完成，退出码 0，376 个成功 optimizer steps，best validation mIoU 为 7.54（epoch 3）；该指标只证明 qualification 铯路可用，不是 baseline 性能结论。
- 受控 parent 在 epoch 1 停止，child 从 parent checkpoint 恢复到 epoch 3。
- 连续 epoch 1 与 parent epoch 1 对比无 mismatch；连续 epoch 3 与 resumed epoch 3 对比无 mismatch。
- 训练结果与恢复摘要均记录 `official_test_included=false`。
- Gate D 摘要：`/root/cloud-ssd/museg-stage04-qualification/runs/commit-56544ea-protocol-0c356824/resume-rehearsal-summary.json`。

结论：batch 10、checkpoint 和 resume 链路具备进入 Stage-05 seed 1 的工程资格。

## 2. 冻结的 development protocol

### 2.1 训练与优化

- 模型：DFormerv2-S，RGB + 单通道 Depth，输入 640×480。
- 数据：`train-dev` 1277 samples / 762 groups；`val-dev` 318 samples / 196 groups。
- batch：10；每 epoch 128 iterations；workers 8；validation batch 1。
- 总周期：500 epochs，共 64,000 planned iterations。
- optimizer：AdamW，base LR `6e-5`，betas `(0.9, 0.999)`，weight decay `0.01`。
- LR：`WarmUpPolyLR`，前 2 epochs / 256 iterations 从 0 线性 warmup；随后按总长度 64,000 iterations 的 poly schedule 衰减，power `0.9`。
- AMP：开启；compile、SyncBN、multi-scale test、sliding 均关闭；train scale 固定 `[1.0]`。

**理由：** 500 epochs 保持论文预算上限，先完整观察单 seed 的学习、平台与过拟合，不提前把 qualification 的 3-epoch 趋势当作收敛结论。optimizer、LR、warmup、增强和 batch 10 沿用 Gate D 已验证候选，避免在首次长程 seed 中同时改变多个变量。poly LR 的分母绑定完整 500-epoch 长度，因此不能把中间 checkpoint 解释为独立短 schedule 的结果。

### 2.2 validation 与 checkpoint

- validation：从 epoch 10 开始，每 10 epochs 一次，并强制包含 epoch 500；预期 50 个 validation 点：10、20、…、500。
- 周期 checkpoint：每 50 epochs 保存，预期 `epoch-50.pth` 至 `epoch-500.pth` 共 10 个。
- `latest.pth`：每个完整 epoch 原子覆盖，支持受控恢复。
- `best-val-miou.pth`：只根据 val-dev 前景 mIoU 严格改善保存；相等时保留最早已评估 epoch。
- 自动 early stop：关闭。只有预注册工程失败条件触发才停止，且失败不自动重启。

**理由：** 10-epoch validation 提供 50 个长程观察点，同时把单次约 70.84 秒的 validation 开销控制在约 59 分钟；50-epoch 周期保存能覆盖 50/100/200/300/400/500 等关键审查点，并把累计 checkpoint 数量控制在 12 个。`latest`、periodic 和 `best` 分别承担恢复、固定里程碑和 val 选择，角色不混用。

### 2.3 预注册 seeds

- seed 1：`772961337`
- seed 2：`1101528019`
- seed 3：`1126246545`

三个值由字符串 `museg-development-long500-v1:seed:{1,2,3}` 的 SHA-256 前 4 bytes 取大端整数后模 `2^31` 确定，避免根据性能挑选 seed。本轮脚本只允许启动 seed 1；seed 2/3 必须在 seed 1 曲线审查和用户再次确认后顺序运行。

## 3. 完整 preflight 结果

### 3.1 仓库与身份

- Git：`main`，HEAD 与 `origin/main` 均为 `56544eacf83b535c49095d7b21060634f36fe1f3`，工作区干净。
- 预训练权重：110,203,103 bytes，SHA-256 `19116988fc86dc9f3e879282237941e11b9b1b5c480edb51e92807311dbc11a6`。
- `train-dev` SHA-256：`a6b15b63f6d5193e3928ea24ada25be403a48e68d1c1f9372cdbbc3fe5cd8470`。
- `val-dev` SHA-256：`1d0719d8f64f016d48995c25ab66d4004d76b7155d9efeef7cbb7454c0dd0e83`。
- 协议已验证 `phase=development`、train role=`train_dev`、val role=`val_dev`、test role=`sealed_unread`。

### 3.2 数据、GPU 与软件

- 四模态目录 RGB / Depth / Depth16 / Label 各 3171 files。
- preflight 只从 train-dev 和 val-dev 各确定性抽样 3 个样本；RGB、Depth、Label、Depth16 均可解码且尺寸一致。未抽样 official test。
- GPU：NVIDIA GeForce RTX 4090，24,564 MiB；当前无训练进程。
- Driver 610.57.04；Python 3.10.16；PyTorch 2.1.2+cu118；CUDA 11.8；cuDNN 8.7.0。
- SwanLab 0.9.7；仓库外 credential file 权限 600；non-interactive online 初始化成功，远端 preflight run 为 `s3soc6ab`。
- GNU screen 4.09.00 和 `/usr/sbin/shutdown` 均可用。

### 3.3 磁盘

- 当前 filesystem：53,619,982,336 bytes 总量，约 38,626,435,072 bytes / 35.97 GiB 可用。
- Stage-04 证据当前占约 3.85 GB。
- 单 checkpoint 实测约 320.98 MB；seed 1 将保留 10 个 periodic、1 个 latest、1 个 best，checkpoint 约 3.85 GB。
- 加上日志、SwanLab、本地 JSON、TensorBoard 和验收报告，seed 1 预计约 4.0 GB；协议按 4.5 GB 规划，并要求启动时至少 15 GiB 可用。
- 当前容量可满足 seed 1，并仍保留约 31.5 GiB 名义余量。

完整 preflight：`pass=true`、0 errors、0 warnings。报告 SHA-256：

- `/root/cloud-ssd/museg-stage05-development/prelaunch/preflight-full.json`：`bc6876f4b588a21de0e549238b4a0037d6ab3f9e2779bf7904f5ba2610618573`
- `/root/cloud-ssd/museg-stage05-development/prelaunch/stage05-audit.json`：`f6b062644f60b30a43f8b1a84617c2777d6f869ecd0c190b64f248769458ab54`
- 未执行命令预览：`/root/cloud-ssd/museg-stage05-development/prelaunch/seed1-command-preview.json`，SHA-256 `af757eaac61ccb95934efa8974c23f7f05fee701f6c05f6e9aaadb99990fc803`

## 4. 预计时间、计算费用与磁盘费用

Stage-04 batch 10 连续 3-epoch qualification 的结构化结果记录总耗时 `483.13936644996284 s`，退出码 0，376 个成功 optimizer steps；该耗时包含训练与当时的验证/保存开销。按同一端到端耗时线性外推，500 epochs 的计算下限为：

- `500 / 3 × 483.13936644996284 s = 80,523.23 s`，约 **22.37 小时**。
- 该值是基于 3 个 epoch 的下限估计，不是承诺时长。Stage-05 有 50 次 validation、10 次 periodic checkpoint、SwanLab 在线写入、文件系统波动和最终验收，因此建议预留 **24–28 小时**；按 15% 余量计算约 **25.72 小时**。
- CompShare 官方文档/价目证据支持“平台实例处于 `Stopped` 后 CPU/GPU/内存算力计费停止”；当前实例的实际计费类型、折扣和订单价未能通过本机 CLI 读取。若按已记录的 RTX 4090 参考价 `1.98 元/小时`，计算费用下限约 **44.29 元**，按 15% 余量约 **50.93 元**，最终以控制台订单为准。
- Stage-04 实测 checkpoint 文件尺寸约 `320,976,778–320,996,589 bytes`。按 seed 1 保留 10 个 periodic、1 个 latest、1 个 best，12 个 checkpoint 的实测尺寸估算为 `3,851,769,357 bytes`，约 **3.59 GiB / 3.85 GB**；加上日志、SwanLab、本地 JSON、TensorBoard 和验收报告，协议按 **4.5 GB** 规划，并要求启动时至少保留 **15 GiB** 可用空间。
- 当前 output filesystem 总量 `53,619,982,336 bytes`，可用 `38,626,193,408 bytes`（约 **35.97 GiB**），专项审计要求通过，满足 seed 1 预算。

## 5. 关机与计费确认

CompShare 官方计费文档明确：**按量后付费实例在平台 `Stopped` 状态下，CPU、GPU 和内存被回收并停止计费；云盘与镜像继续计费。** 官方停止接口说明实例会从 `Running` 经 `Stopping` 进入 `Stopped`。

边界：官方文档没有明确说明“实例操作系统内执行 `shutdown -h`”是否必然被平台控制面识别为 `Stopped`；CompShare CLI 的问答服务两次返回 HTTP 502，本机也没有 API 密钥可查询当前实例订单。因此脚本使用标准 ACPI poweroff，并在成功路径 `sync` 后调用 `shutdown -h +5`；关机后仍应在控制台确认实例状态为 `Stopped`。只有达到 `Stopped` 才能把计算停止计费视为已落实。

## 6. 脚本与成功/失败语义

- 主训练/验收/关机脚本：`/root/cloud-ssd/museg-stage05-development/scripts/train_accept_sync_shutdown_seed1.sh`
- screen 启动包装：`/root/cloud-ssd/museg-stage05-development/scripts/start_seed1_screen.sh`
- preflight 审计器：`/root/cloud-ssd/museg-stage05-development/scripts/stage05_preflight.py`
- seed 1 验收器：`/root/cloud-ssd/museg-stage05-development/scripts/accept_seed1.py`

主脚本会在启动时重新执行 full online preflight 和 Stage-05 专项审计；随后仅运行 seed 1。训练完成后，验收器检查 run manifest、training result、run config、50 个 validation 点、里程碑日志、12 个 checkpoint 的 schema/identity/schedule/SHA、SwanLab 中 mIoU/mAcc/mF1 records 和 `official_test_included=false`。所有必需项通过后才写 success record、执行 `sync`，再安排 5 分钟后 poweroff。

任何命令、训练或证据验证失败都会进入 failure trap：写 `failed.json`、执行 `sync` 保存现场，并明确记录 `automatic_shutdown=false`；不会安排关机，也不会自动重启训练。

已知限制：当前 trainer 不把 per-class IoU 写入本地结构化曲线；SwanLab 记录 mIoU、mAcc、mF1。per-class IoU 应在 seed 1 完成后、Gate E 曲线审查前，用 best-val checkpoint 在 val-dev 上执行受控评估并结构化留档；这不影响本次训练启动链路，但不能在最终研究汇报中省略。

## 7. 手动授权边界

当前没有 seed 1 输出目录、没有训练进程、没有新的 screen 训练 session、没有已安排关机。本轮停在手动授权边界。

用户授权后手动执行：

```bash
/root/cloud-ssd/museg-stage05-development/scripts/start_seed1_screen.sh
```

查看状态：

```bash
screen -r museg-stage05-seed1
# 或
 tail -F /root/cloud-ssd/museg-stage05-development/seed-772961337-screen.log
```

启动后不要临时更改 epochs、LR、validation/save interval、seed 或 checkpoint 规则；发生失败时保留现场并回到受控恢复分析。
