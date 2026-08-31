# MUSeg Quick-B0 外部控制面 schedule 回执

> 回执形成时间：2026-08-30 14:31 UTC
> 最终追加时间：2026-08-31 04:33 UTC
> 回执范围：记录 CompShare 控制面最晚停止 schedule、训练终态证据拉取与本地复核、普通 stop 和 `Stopped` 复核。

## 1. 执行环境

- CompShare CLI 版本：`0.3.6`
- 实例 ID：`cpod-1tyvjsiu6ahe`
- 计划动作：实例 shutdown（控制面最晚停止兜底）
- 目标 UTC 时间：`2026-08-31T22:30:00Z`
- CLI 查询显示的等价本地时间：`2026-09-01T06:30:00+08:00`

## 2. schedule 设置回执

- 执行开始 UTC 时间：`2026-08-30T14:31:17.748Z`
- `schedule set` JSON：`ok=true`
- 返回实例 ID：`cpod-1tyvjsiu6ahe`

执行命令：

```bash
compshare --json instance schedule set cpod-1tyvjsiu6ahe --at '2026-08-31T22:30:00Z'
```

## 3. 即时查询复核

- 复核完成 UTC 时间：`2026-08-30T14:31:24.990Z`
- `schedule show` JSON：`ok=true`
- 实例 ID：`cpod-1tyvjsiu6ahe`
- `scheduled=true`
- `scheduler_stop_time=1788215400`
- `scheduler_stop_at=2026-09-01T06:30:00+08:00`，等价于目标时间 `2026-08-31T22:30:00Z`
- 计划动作：shutdown
- 即时实例状态查询 JSON：`ok=true`
- 当前实例状态：`Running`
- 状态记录中的 `SchedulerStopTime=1788215400`

复核命令：

```bash
compshare --json instance schedule show cpod-1tyvjsiu6ahe
compshare --json instance show cpod-1tyvjsiu6ahe --status
```

## 4. schedule 设置时的结论与边界

最晚停止 schedule 已设置并经控制面即时查询复核，设置时实例保持 `Running`。大白话说，训练主窗口当时已获得明确的最晚关机兜底；这一历史回执本身不代表训练已经启动、完成或停止。

- 未启动或恢复训练；
- 未运行评估或读取 official test；
- 未执行普通 stop；
- 训练终态证据、普通 stop 请求和 `Stopped` 状态将在训练终态后追加，不能据本回执写成已完成。

## 5. 训练终态

- 训练结束 UTC 时间：`2026-08-31T04:16:27.983518Z`；本地控制器于 `2026-08-31T04:18:43.130332Z` 检出终态。
- `run_museg_seed.py` 已退出，`train.exit_code=0`。
- `training_result.json`、`run_manifest.json` 和 `checkpoint-candidates.json` 均已生成；候选清单 `finalized=true`。
- 完成 `500/500` epoch；optimizer attempted/completed/skipped 为 `64,000/63,971/29`，直接复核满足 `64,000 = 63,971 + 29`。
- seed `772961337`、run ID `museg-dformerv2-s-rgb-quick-b0-v1-development-DFormerv2-S-development-single-b0-500epoch-v1-seed-772961337`、Git commit `3975f7d66c78e9bed6b9053071bb274199d550e9` 和 protocol SHA-256 `6822e4cdfd9c6985c323123fc4d24a9f06ed269fada55203b1707fc5ab612bbd` 与冻结身份一致。
- `official_test_included=false`，run manifest 中 official test 仍为 `sealed_unread=true`。

大白话说，唯一一次 Quick-B0 已按冻结身份完整跑完，没有启动第二个 seed 或第二个 run。

## 6. 证据下载与本地复核

- 本地证据目录：`D:\0Project\DFormer\cloud\DFormer-quick-b0-evidence\museg-dformerv2-s-rgb-quick-b0-v1`。
- 本地复核完成 UTC 时间：`2026-08-31T04:32:59.646650Z`。
- 已下载 active seed 完整目录、materialized protocol、`preflight.json`、development 目录中的两份 Screen 日志、首次失败启动证据目录，以及候选清单列出的全部 checkpoint。
- active seed 共 `21` 个文件、`1,620,776,692` bytes；首次失败启动证据共 `5` 个文件、`9,912` bytes。两份 Screen 日志在云端和本地均为 0 bytes，文件本身已保留。
- materialized protocol 本地重算 SHA-256 为 `6822e4cdfd9c6985c323123fc4d24a9f06ed269fada55203b1707fc5ab612bbd`。
- 候选清单本地重算 SHA-256 为 `9c7297e1c6c4a8e5929108bccfd9b720ac110640f3bc28cdec2cb40c796d1942`，与 `training_result.json` 记录一致。
- 本地逐项解析 seed、run ID、Git commit、split、`official_test_included=false` 和 optimizer 遥测不变量均通过；结构化结果见同目录 `local-verification.json`。

候选 checkpoint 本地回读结果：

- `selector-epoch-480.pth`：`321,011,270` bytes，SHA-256 `50a8febc4a4876fc4b0b3f882f92361b8deb2ad0bc02216c6d28fde6ef5e12f8`；
- `selector-epoch-420.pth`：`321,011,270` bytes，SHA-256 `f246a3afc50334c81302b7bfebdadf7cf37d00326bf1c3aa54f6a151754e3a1c`；
- `selector-epoch-440.pth`：`321,011,270` bytes，SHA-256 `096125ac3cca1c215f085cfb164aed3a9e25b6784ce8334d3bc051f9419aec47`；
- `latest.pth`（epoch 500）：`320,977,354` bytes，SHA-256 `00ba9f7dbd31677b630664040406eed809e7204f46941e1f5a2baa9d37d44f67`。

4 个候选 SHA-256 两两不同，所有本地重算值均与候选清单一致。候选清单条目本身未包含显式 `size_bytes` 字段，因此本回执和 `local-verification.json` 使用下载文件直接回读的大小补充该字段；没有改写云端原始清单。

## 7. 控制面 stop 与 `Stopped` 复核

- 普通 stop 请求 UTC 时间：`2026-08-31T04:33:16.795325Z`。
- `instance stop` JSON：`ok=true`，实例操作成功。
- `Stopped` 确认 UTC 时间：`2026-08-31T04:33:31.827620Z`。
- `instance wait --state Stopped` JSON：`ok=true`；随后 `instance show --status` 再次返回 `State=Stopped`。
- 本次未使用固定 schedule 触发停止；由获授权的外部控制器在证据下载和本地哈希复核后执行了一次普通 stop。
- 未发生 stop 失败、重试或人工补发第二次普通 stop。

## 8. 最终边界与未确认项

- 训练和外部控制面收口成功；CompShare 实例当前为 `Stopped`。
- official test 继续保持 `sealed_unread`，本次未运行主 evaluator、完整测试或额外 GPU probe。
- 尚未确定最终 B0 checkpoint：训练期 original-full selector 最优为 epoch 480、mIoU `56.87`，但它不是五尺度翻转主 evaluator 的最终结果。
- 本地 evaluator 启动前仍需从训练提交 `3975f7d66c78e9bed6b9053071bb274199d550e9` 建立干净 worktree，并处理冻结 protocol 中云端绝对 `val-dev` 路径在 Windows 上的路径映射；当前 `main` 已修改 `tools/museg_protocol.py`，不得直接替代冻结 evaluator 代码身份，也不得通过改写 materialized protocol 改变其 SHA-256 身份。
