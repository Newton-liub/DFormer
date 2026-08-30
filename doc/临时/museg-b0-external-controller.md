# MUSeg Quick-B0 外部控制面代理交接

> 形成时间：2026-08-30 14:24 UTC
> 角色：只负责 CompShare 控制面最晚停止计划、训练终态证据拉取、控制面 stop 与 `Stopped` 复核；不修改训练协议，不启动第二个训练，不运行评估。
> 训练主机：`cpod-1tyvjsiu6ahe`
> 仓库：`/root/rivermind-data/DFormer`
> official test：必须保持 `sealed_unread`，不得读取、抽样或评估。

## 1. 立即执行：设置最晚停止兜底

本次预计在 2026-08-30 14:30 UTC 后启动，训练预计 24–28 小时。请把控制面最晚停止时间设为：

- `2026-08-31T22:30:00Z`（约为预计启动后 32 小时）

先用你已配置凭据的 CompShare CLI 查看当前安装版本的准确参数：

```bash
compshare --json instance schedule set --help
compshare --json instance schedule show --help
```

然后对实例 `cpod-1tyvjsiu6ahe` 设置最晚停止计划并立即查询复核。当前仓库计划所依据的命令形态是：

```bash
compshare --json instance schedule set cpod-1tyvjsiu6ahe --at '2026-08-31T22:30:00Z'
compshare --json instance schedule show cpod-1tyvjsiu6ahe
```

若当前 CLI 帮助显示参数顺序不同，以当前帮助为准，不要猜测或重复提交。请记录：

1. schedule set JSON 的 `ok`；
2. schedule show JSON 中的实例 ID、计划动作、UTC 时间；
3. 执行 UTC 时间；
4. 当前实例状态仍为 `Running`。

未收到这份回执前，训练主窗口不会启动正式训练。

## 2. 通过 GitHub 交付 schedule 回执

在已同步本提交的仓库中新增：

- `doc/临时/museg-b0-external-controller-receipt.md`

只写控制面回执，不写 API 密钥、密码、SSH 地址或其他敏感字段。回执至少包含第 1 节四项，并注明执行命令的 CompShare CLI 版本。提交并推送到 GitHub 后，把提交哈希告知用户；训练主窗口会 `git pull --ff-only`，直接核验该文件与提交后再物化正式 protocol。

外部控制代理除这份回执和训练终态回执外，不修改项目代码、protocol、config 或当前状态文件，避免与训练主窗口发生冲突。

## 3. 冻结的训练边界与路径

- 模型：DFormerv2-S；RGB 输入；single-seed development B0。
- seed：`772961337`；500 epoch；batch 10；`run_kind=standard`。
- Git commit：以训练主窗口拉取 schedule 回执后、物化 protocol 时的干净 `main` HEAD 为准。
- materialized protocol：`/root/rivermind-data/DFormer/protocols/generated/museg-dformerv2-s-rgb-quick-b0-v1-20260830.json`。
- protocol template：`/root/rivermind-data/DFormer/protocols/museg-dformerv2-s-rgb-quick-b0-v1.template.json`。
- output root：`/root/rivermind-data/DFormer/outputs`。
- seed output：`/root/rivermind-data/DFormer/outputs/museg-dformerv2-s-rgb-quick-b0-v1/development/seed-772961337`。
- 候选 manifest：`/root/rivermind-data/DFormer/outputs/museg-dformerv2-s-rgb-quick-b0-v1/development/seed-772961337/checkpoint-candidates.json`。
- Screen 会话：`museg-quick-b0-772961337`。
- 默认 run ID：`museg-dformerv2-s-rgb-quick-b0-v1-development-DFormerv2-S-development-single-b0-500epoch-v1-seed-772961337`。
- 预计训练：24–28 小时。
- 费用口径：按历史参考价 1.98 元/小时，预计 47.52–55.44 元；32 小时兜底上限对应 63.36 元，实际以控制台订单为准。
- 只允许同一 run 的受控恢复；不得自动增加 epoch、seed、候选数或旁路实验。
- 主 evaluator 不在云端执行；训练收口后在本地 RTX 5060 Laptop 运行。

训练主窗口会在启动后把 materialized protocol SHA-256、实际 Git commit 和实际 run ID 写入 `doc/main/MUSeg-current-status.md`。外部控制代理可在需要时从 GitHub 同步该状态，但不得据旧计划猜测身份。

## 4. 训练终态后的证据拉取

训练正常完成、失败或人工中止后，都先取回必要证据，再调用普通控制面 stop。不得用验收 pass/fail 决定是否停止计费，也不得仅依赖实例内 `shutdown -h`。

正常完成时至少取回：

- materialized protocol manifest；
- `run_manifest.json`；
- `training_result.json`；
- `train.exit_code`；
- `environment.json`；
- `command.json`；
- `checkpoint-candidates.json`；
- 候选 manifest 列出的最多 4 个 checkpoint；
- `launcher.log`、训练日志、验证日志；
- preflight 报告和 Screen 日志。

失败或中止时，取回当时已存在的上述文件、Screen 日志、最新 checkpoint、失败 traceback 和目录清单，不要伪造缺失的成功证据。

使用已配置凭据的控制机执行 `compshare instance cp`。目标目录由控制机选择，但必须记录其绝对路径。逐项核对：

1. 候选文件大小；
2. 候选 manifest 记录的 SHA-256 与拉取文件重算值一致；
3. protocol SHA、Git commit、seed、run ID 一致；
4. `official_test_included=false`；
5. 500 epoch 正常完成时，optimizer attempted/completed/skipped 满足已冻结计数不变量。

## 5. 控制面 stop 与最终回执

证据拉取和哈希核验结束后立即执行普通控制面 stop，并等待实例进入 `Stopped`。先查看当前 CLI 帮助，再按当前版本执行；命令形态参考：

```bash
compshare --json instance stop cpod-1tyvjsiu6ahe --yes --timeout 600
compshare --json instance wait cpod-1tyvjsiu6ahe --state Stopped --timeout 600
compshare --json instance show cpod-1tyvjsiu6ahe --status
```

最终回执必须包含：

- 证据拉取完成 UTC 时间和控制机绝对目录；
- 候选 checkpoint 文件名、大小、SHA-256；
- stop 请求 UTC 时间和 JSON `ok`；
- 达到 `Stopped` 的 UTC 时间及状态查询结果；
- 是否使用了 schedule 兜底；
- 是否发生人工补发普通 stop；
- 未确认项或失败原因。

将最终回执追加到 `doc/临时/museg-b0-external-controller-receipt.md`，提交并推送；不要改写第一次 schedule 回执。若训练主窗口尚未推送最新状态，先等待或使用单独提交，不强推、不覆盖远端历史。

## 6. 禁止事项

- 不启动训练、恢复训练或第二个 run；
- 不修改项目代码、protocol、config、seed、epoch、batch、增强或 checkpoint 规则；
- 不运行主 evaluator、完整测试或额外 GPU probe；
- 不读取 official test；
- 不在证据未拉取时提前普通 stop，除非实例安全、失控计费或用户明确要求；出现这类紧急情况时先 stop，再如实报告证据未完整取回；
- 不提交密钥、密码、SSH 地址、访问令牌或其他敏感信息；不强推 GitHub。
