# CompShare 云端流程

> 角色：本仓库重复使用的云端训练流程，供 `.cursor/skills/compshare-cloud/SKILL.md` 按需读取。
> 依据：官方 CLI 文档 <https://www.compshare.cn/docs/gpus/cli> 与官方 Skill <https://github.com/compshare-cn/compshare-cli>，本机已安装 `compshare 0.3.6` 并核对了本文涉及命令的 `--help`。
> 边界：只覆盖“起实例 → 同步代码 → 跑训练 → 取回结果 → 关机”这一条重复链路；镜像、US3、团队、账单、媒体生成等不在此封装。

## 0. 前置

- 凭证只放本机 profile 或环境变量（`compshare config --name <profile>`、`COMPSHARE_PUBLIC_KEY` / `COMPSHARE_PRIVATE_KEY`）。不在仓库、Skill、日志或聊天里输出密钥、密码、IP 与访问链接。
- 先做只读确认：`compshare --json doctor`、`compshare config list`、`compshare --json instance list --all`、`compshare --json instance show <id> --status --spec --billing`。
- 陌生命令一律先 `compshare --json <command> --help`，以本机实际版本为准，不照搬旧 wrapper 或过时参数。
- 自动化统一加 `--json` 解析；`--yes` 只在用户明确授权该次变更后使用。

## 1. 实例与有卡/无卡模式

- 无卡（CPU）模式用于准备与回收：`compshare --json instance start <id> --without-gpu A --wait --timeout 600`（`A` / `B` 是平台的无卡规格档位，先按 `--help` 与平台说明核对，不要预设免费或性能）。本项目现有实例实测 A 档为 2 CPU / 4 GiB / GPU=0；平台报价字段 `InstancePrice=0.13`，不能因此声称无卡免费。
- 有卡模式用于训练，属于计费状态：只在本次授权明确、代码与数据就绪、关机保险设置完成后启动。
- 启动/停止均显式给超时并回读状态；超时后先查实际状态，不盲目重试：`compshare --json instance stop <id> --yes --wait --timeout 600`。
- 关机不等于零成本：系统盘可能继续计费，实例保留还是释放需单独决策。`compshare --json instance billing` 是规格价格估算（必须给 `--gpu`），不是当前账单，不能当作实时费用依据。

## 2. 关机保险（训练前必须完成）

- 设置计划关机：`compshare --json instance schedule set <id> --at <ISO8601|unix|relative>`，随后用 `compshare --json instance schedule show <id>` 回读确认；需要时可 `extend`、`cancel`。
- 保险必须覆盖预计训练时长并留出取回余量；未确认保险生效前不开始训练。
- 训练正常结束或失败收尾时都主动关机，不只依赖定时器。
- SSH 或本地网络中断不影响平台 CLI 关机；平台本身不可用时立即告警并保留最后一次确认到的状态。

## 3. 代码同步

- 代码只通过 Git 同步，不用旧目录快照或压缩包覆盖工作区。云端与本地必须指向同一个已确认 commit：`git rev-parse HEAD` 与 `git status --porcelain` 都要核对（目标为同 HEAD、受跟踪文件干净）。
- 推送由主代理在获得用户确认后执行；云端只做 pull/fetch 到该 commit。若只授权本地提交，使用 Git bundle：本地 `git bundle create <outside-worktree>.bundle <branch> ^<base>`，经 SSH 传到云端，云端 `git fetch <bundle> <branch>` 后 `git checkout -B <branch> FETCH_HEAD`，同步完成删除临时 bundle；仍是 Git 同步，不推送远端。
- 训练所需的小文件用 `compshare --json instance cp <id> <local> :<remote>`；数据集与权重若已在云端，不要重复上传。
- 取回：`compshare --json instance cp <id> :<remote> <local>`。

## 4. 长任务与断线保护

- 首选 CLI 持久任务：`compshare --json instance job submit <id> --name <name> --cwd <remote-dir> -- <command>`，配合 `job list` / `job show` / `job logs --tail` / `job wait --timeout`。
- 已有环境继续用 `screen` 也可以：`screen -dmS <name> <command>`、`screen -ls`、`screen -r <name>`、`screen -S <name> -X quit`。二者择一，不叠加多层 supervisor。
- 日志读取优先用带偏移的增量方式，避免把完整日志拉进上下文。

## 5. SwanLab 环境

- 先确认实际安装版本与可用模式（online / offline / local / log-only），以该版本的 `swanlab` 帮助与文档为准，不硬编码未核验的参数名。
- 凭据只来自环境变量或本机登录配置，绝不写进仓库、Skill 或日志。未登录或模式不明时按 log-only 处理并记录，不长期阻塞训练。
- 记录本次 run 的 SwanLab 项目名、模式与产物位置，产物取回到 `outputs/<experiment>/<run-id>/swanlab/`。
- 作者代码保留 TensorBoard；新研究配置通过 `research/tracking.py` 的可选调用启用 SwanLab，公共训练入口只增加最小日志调用，不改变训练 / 评价算法。配置与字段见 [新研究配置](research-setup.md)。官方持久登录使用 `swanlab.login(..., save=True)`，凭据保存在用户目录 `~/.swanlab/.netrc`，不得入 Git。

## 6. 结果回收

- 统一落到本地 `outputs/<experiment>/<run-id>/`：`logs/`、`checkpoints/`、`swanlab/`、`predictions/`、`eval/`。
- 大文件取回优先“先停卡、再用无卡窗口”，减少计费时间；取回后用一个小记录保留 commit、命令、配置、实例 ID 与产物路径（不新建 manifest/hash 数据库）。
- 需要的代码或配置在复核后归位到 `research/` 或 `local_configs/research/`；结论写进 `doc/reports/`。不要把下载的源码副本当作第二套当前代码。

## 7. 收尾清单

1. 确认任务退出码与关键日志，判定成功或失败。
2. 取回必要产物并核对文件存在与大小。
3. 主动关机并回读状态；确认不存在计划外仍在计费的实例。
4. 更新 `doc/state/current.md`（实例状态、产物位置、下一步、阻塞）。
