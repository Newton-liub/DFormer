---
name: compshare-cloud
description: Runs the repeated CompShare GPU cloud workflow for this project, covering instance start and stop, GPU and no-GPU modes, Git sync, artifact upload and retrieval, durable remote training with screen or instance job, SwanLab environment, scheduled shutdown insurance, and result recovery. Use when the user mentions CompShare, 云端, 算力, GPU 实例, remote training, or pulling cloud results.
---

# CompShare 云端流程

详细步骤与命令见 `doc/guides/cloud.md`（唯一正文）。本 Skill 只保留触发条件、顺序与审批边界。

## 依据

- 官方 CLI 文档 <https://www.compshare.cn/docs/gpus/cli>、官方 Skill <https://github.com/compshare-cn/compshare-cli>。
- 以本机 `compshare` 实际版本的 `--help` 为准；陌生命令先查帮助，不照搬旧 wrapper 或过时参数。

## 顺序

1. 只读确认：`doctor`、`config list`、`instance list` / `instance show`，明确实例、GPU 或无卡模式、预算与取回范围。
2. 准备与回收优先无卡模式；有卡模式属于计费状态，只在授权明确时启动。
3. 代码只用 Git 同步到同一已确认 commit，并核对云端 `git rev-parse HEAD` 与工作区状态。
4. 长任务用 `instance job`（首选）或 `screen`，二者择一，不叠加多层 supervisor。
5. 训练前设置并回读平台计划关机保险；未确认保险不开始训练；正常结束或失败都主动关机。
6. SwanLab 凭据只来自环境变量或本机登录配置，先确认可用模式，未登录时按 log-only 处理，不写进仓库或日志。
7. 结果取回到 `outputs/<experiment>/<run-id>/`；需要大量下载时先停卡，再用短时无卡窗口。
8. 收尾确认退出码、产物与实例状态，最后更新 `doc/state/current.md`。

## 边界

- 资源生命周期操作、创建或销毁资源、产生费用的操作由主代理负责并需要用户明确授权；`--yes` 只在授权后使用。
- 不回显敏感字段（密码、IP、访问链接、密钥）；除非用户明确需要，不使用 `--show-sensitive`。
- 超时后先查实际状态再决定是否重试，避免重复训练或重复计费。
- 不封装镜像、US3、团队、账单与媒体生成等当前未使用的功能。
