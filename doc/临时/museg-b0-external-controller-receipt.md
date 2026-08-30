# MUSeg Quick-B0 外部控制面 schedule 回执

> 回执形成时间：2026-08-30 14:31 UTC
> 回执范围：仅记录 CompShare 控制面最晚停止 schedule 的设置与即时复核；训练终态证据拉取、普通 stop 和 `Stopped` 复核将在训练终态后追加。

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

## 4. 当前结论与边界

最晚停止 schedule 已设置并经控制面即时查询复核，实例保持 `Running`。大白话说，训练主窗口现在可以看到明确的最晚关机兜底；本回执尚不代表训练已经启动、完成或停止。

- 未启动或恢复训练；
- 未运行评估或读取 official test；
- 未执行普通 stop；
- 训练终态证据、普通 stop 请求和 `Stopped` 状态将在训练终态后追加，不能据本回执写成已完成。
