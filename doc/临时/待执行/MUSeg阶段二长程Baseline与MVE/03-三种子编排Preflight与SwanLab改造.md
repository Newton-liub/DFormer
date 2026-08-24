# 03：三种子编排、preflight、4090 脚本与 SwanLab 改造计划

> 任务类型：外围编排、可观测性和防误操作工具。
>
> 模型要求：Terra 可按本计划实现；**Sol 必须复核字段契约、失败传播和最终端到端接口。**
>
> 前置：02 的 CLI、checkpoint schema、phase 和输出目录规则已经冻结。

## 1. 目标

- 将单次 20-epoch 脚本升级为 qualification、development、official 三类明确入口。
- 支持固定 seed、独立输出目录、顺序三种子长训、受控 resume 和机器可读汇总。
- 扩展 preflight，使 split、Git、权重、输出和 SwanLab 条件在耗费 GPU 前全部可审计。
- 保留 4090 batch 探测，但探测结果只决定资源参数，不决定研究结论。

## 2. 建议修改和新增文件

- 修改 `tools/preflight_train.py`
- 修改 `tools/probe_museg_4090.sh`
- 重构或替换 `tools/train_museg_4090.sh`
- 新增 `tools/run_museg_seed.py` 或等效单 seed launcher
- 新增 `tools/run_museg_3seed.py` 或 Shell 顺序 orchestrator
- 新增 `tools/summarize_museg_runs.py`
- 修改 `utils/experiment_tracker.py` 和 `utils/train.py` 的元数据构建处
- 新增相应测试和 fake launcher 夹具

## 3. seed 和 run 身份规格

- seed 列表由 05 最终冻结；开发前可使用预先指定的候选列表，但不得运行后挑最好 seed。
- 每个 run ID 至少含 phase、model、schedule version、seed 和 UTC 时间或协议 ID。
- 输出必须隔离：`<output-root>/<protocol-id>/<phase>/seed-<seed>/`。
- 三 seed 在单卡上严格顺序执行；不得后台并发。
- 已存在非空输出目录时失败；resume 只能显式指定且必须通过 02 的兼容检查。
- launcher 写入 `launcher.log`、`train.exit_code`、`command.json`、`environment.json` 和 `run_manifest.json`。

## 4. preflight 扩展

除已有包、CUDA、4090、数据抽样、权重和 SwanLab 外，增加：

1. Git 分支、完整 commit、工作区干净；非干净时失败并列出路径。
2. 当前 commit 包含所需阶段提交；不硬编码过时 HEAD，改由计划提供最低提交或 protocol manifest。
3. train-dev/val-dev/official train/test 路径、数量、重复、样本交集、group 交集和 SHA-256。
4. phase 角色校验：development 不得将 test 设为 val；official 训练入口不得读取 test。
5. 预训练权重大小和 SHA-256 精确匹配计划值。
6. 输出目录父目录可写、目标目录不存在或满足显式 resume。
7. 配置导入后输出 epochs、iterations、warmup、eval/save interval、seed 和 phase 摘要。
8. 在线 SwanLab 必须非交互、凭据可用且初始化失败快速终止；不回显 key。
9. Python/PyTorch/CUDA/cuDNN/驱动/GPU/包版本写机器可读环境记录。

warning 必须有明确分类：允许继续的建议性 warning 与阻塞 error 不可混淆。

## 5. 4090 batch 探测

- 保留 batch 4/8/12/16，每档固定相同 seed、相同步数、相同输入和监控设置。
- 每档使用独立目录，输出结构化 JSON：退出码、完成步数、稳定吞吐、step time、allocated/reserved/free、free ratio、loss、AMP scale、异常。
- 某档 OOM 后停止更大 batch；非 OOM 的环境/数据错误立即停止全部探测。
- 安全阈值继续要求至少 2 GiB 且至少 10% free VRAM。
- 选择稳定吞吐最高且满足安全阈值的 batch，不选择“仅仅能跑的最大值”。
- 探测后强制人工暂停，用户确认 batch 才能进入 04 的短训。

## 6. SwanLab 元数据契约

每个 run config 至少包含：

- protocol ID、phase、run ID、seed、完整 Git commit、dirty=false；
- dataset/model/backbone、输入模态和 Depth 版本；
- split 路径、数量和 SHA-256；
- epochs、iterations/epoch、batch、workers、AMP、compile、SyncBN；
- optimizer、基础 LR、poly power、warmup、weight decay；
- train/eval scale、flip、sliding、MST；
- eval/save interval、best metric 和 tie-break；
- 输出目录、checkpoint schema；
- resume parent、resume checkpoint SHA；
- Python/PyTorch/CUDA/驱动/GPU；
- pretrained checkpoint SHA。

测试要求 rank 0 only、显式 run name 优先、online fail-fast、finish 仅一次。

## 7. 三种子 orchestrator

- 接受 protocol manifest 和明确 seed 列表，不从目录猜参数。
- 每次只启动一个 seed；当前 seed 非零退出时停止后续 seed并报告。
- 不自动重试 OOM、Xid、非有限 loss、SwanLab 失败或数据错误。
- 允许用户批准后对指定 seed 从已验证 epoch checkpoint 恢复；恢复作为新 SwanLab run 并记录 parent。
- 所有 seed 完成后只汇总训练/validation 和 checkpoint 状态；official test 汇总留给 06。
- 汇总 JSON 包含每 seed 最佳 val、最佳 epoch、最终 epoch、总时长、退出码、checkpoint 路径和 SHA。

## 8. Terra 测试要求

- `bash -n` 覆盖所有 Shell。
- fake Python/launcher 验证参数透传、顺序执行、失败即停、退出码保存。
- Windows/Unix 路径与云端目录末尾 `#` 的安全引用。
- seed 输出目录不碰撞，已有目录拒绝覆盖。
- resume 参数必须带 checkpoint SHA 和 parent run。
- preflight 对 dirty Git、split 交集、哈希错误、错误 phase、权重错误、非 4090 分别失败。
- fake SwanLab 验证完整 config 和 online fail-fast。
- 汇总器对缺 run、重复 seed、非零 exit、损坏 JSON 快速失败。

## 9. 执行门禁与 Git

1. Terra 完成实现和静态测试。
2. Sol 审核所有会影响实验身份、失败传播和 test 封存的接口。
3. 只提交脚本、代码和测试，不提交运行输出。
4. 推荐提交标题：`增加 MUSeg 三种子长程训练编排与审计`。
5. 04 开始前要求工作区干净，preflight 能在无 GPU模式完成静态部分。

## 10. 完成标准

- 任一 run 可由 manifest 唯一重建。
- 三 seed 严格隔离并顺序执行，失败不会被吞掉。
- SwanLab 和本地 JSON 都能追溯 commit、split、seed、超参数、环境和 resume。
- preflight 在 GPU 分配前发现协议、路径和工作区错误。
- Sol 签署端到端接口，允许进入 04。
