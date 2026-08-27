# MUSeg Stage-05 development seed 1 运行中交接报告

> **文档角色：历史运行中快照。** 本报告只描述 2026-08-26 03:33 UTC 的运行状态；seed 1 后来已完成 500/500 epochs，实例已停止。当前事实和恢复点只看 `doc/main/MUSeg-current-status.md`，完成后的详细进度见 `doc/reports/2026-08-26-museg-stage05-seed1-local-closure-handoff.md`。
>
> 正文中的“当前”“正在运行”“下一门禁”均按核验时点解释，不得据此恢复旧进程、连接旧端口或重新启动 workload。

- 核验时间：2026-08-26 03:33 UTC
- 工作状态：**正在进行；不是已完成 baseline**
- 核验方式：通过 CompShare CLI 对运行实例和云端文件做只读检查
- 云端实例：`cpod-1tyvjsiu6ahe`，状态 `Running`
- Git：`main`，HEAD 与 `origin/main` 均为 `56a7ed711df2252e6228fc777d7cb92eb2510ef6`，工作区干净
- protocol：`museg-development-long500-v2`
- protocol SHA-256：`47d1a52a20bbb73b9d1a1b609819335d07ca84b49e8373d4046318b414ad324d`

## 1. 启动身份与 preflight

本次 run 使用重新物化的 v2 protocol，不沿用绑定旧提交 `56544ea...` 的 v1 protocol。下载后的 protocol 原始字节重新计算 SHA-256，与云端 preflight 和进程命令中的 `47d1a52a...` 一致。

启动前证据：

- native preflight：`/root/cloud-ssd/museg-stage05-development/prelaunch/preflight-launch-seed-772961337.json`，`pass=true`、0 errors、0 warnings；
- Stage-05 audit：`/root/cloud-ssd/museg-stage05-development/prelaunch/stage05-audit-launch-seed-772961337.json`，`pass=true`；
- Git exact commit、干净工作区、冻结 split、预训练权重、RTX 4090、SwanLab online、磁盘和 Gate D 绑定检查均通过；
- protocol phase 为 `development`，roles 为 `train_dev`、`val_dev`、`sealed_unread`。

当前 Git 中归档的可移植副本是 `protocols/museg-development-long500-v2.template.json`。其中绝对机器路径替换为 materialization 占位符，因此模板自身 SHA 不等于云端 materialized protocol SHA；原始 SHA 作为 provenance 保留在本报告和文档审计中。

## 2. 运行中证据

只读检查时存在 detached screen `museg-stage05-seed1`，主训练进程参数包括：

- seed `772961337`；
- 500 epochs，batch 10，128 iterations/epoch；
- validation 从 epoch 10 开始、每 10 epochs；
- checkpoint 每 50 epochs；
- AMP 开启，compile/SyncBN/sliding/MST 关闭；
- train-dev 1277、val-dev 318；
- required Git commit `56a7ed7...`；
- protocol ID `museg-development-long500-v2`。

2026-08-26 03:33 UTC 的 `train.log` 已完成 epoch 445。已存在 `epoch-50.pth` 至 `epoch-400.pth`、`latest.pth` 和 `best-val-miou.pth`。epoch 440 的运行中 best val-dev mIoU 为 52.41，日志 ETA 约为 04:54 UTC。

这些值只证明 run 正在正常推进。训练完成、验收器通过和三个 seeds 汇总前，不得把 52.41 写成正式 baseline。

## 3. Official test 边界

- preflight 与 Stage-05 audit 均记录 official test role=`sealed_unread` 和 `official_test_read=false`。
- `run_config.json` 记录 official-test 路径、样本数和 SHA 用于身份约束，并明确 `sealed_unread=true`。
- development 进程命令携带 test identity 参数不等于读取 test 样本；当前 phase role 不建立 test loader。
- 运行尚未生成最终 `training_result.json`，因此最终的 `official_test_included=false` 必须在 seed 1 验收时再次检查，当前不能提前宣称最终验收已完成。

## 4. 已知边界

- 之前存在一个 `museg-stage05-seed-failure-v1` 记录，退出码 1、失败行 49、未自动关机；随后 v2 protocol 的当前 run 成功启动。该失败记录保留，不改写。
- CLI 中 `run_kind=qualification` 是当前代码对“非 probe 完整运行”的历史字段名；真正研究 phase 由 `experiment_phase=development` 决定。
- validation 原分辨率、BGR 通道和 A2 自然无效深度门槛仍按 `doc/main/MUSeg-open-decisions.md` 登记，当前 run 不变更。

## 5. 下一门禁

seed 1 结束后必须核验训练退出码、验收器结果、50 个 validation 点、12 个 checkpoint、best/latest 身份、SwanLab 记录和 `official_test_included=false`。完成这些检查后再进入 Gate E 曲线审查；seed 2/3 不自动启动。
