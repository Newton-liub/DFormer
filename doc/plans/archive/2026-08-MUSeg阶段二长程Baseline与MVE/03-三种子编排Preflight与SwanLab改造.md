# 03：三种子编排、preflight、4090 脚本与 SwanLab 改造计划

> **后继候选计划：** `doc/plans/deferred/2026-09-MUSeg-unexecuted/MUSeg-A2-B2深度有效性/00-总方向规划.md`（该后继计划现已延期，未执行）。

> **历史阶段记录：** Stage-03 与 Gate B 已完成。以下尚未运行真实 GPU/SwanLab 的表述是 Stage-03 结束时快照，已被 Stage-04/05 证据取代；当前入口见 `doc/main/MUSeg-current-status.md`。

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

## 11. 执行完成记录（2026-08-25）

### 11.1 已实现接口

- 新增 `museg-training-protocol-v2` 协议清单契约及 JSON Schema；契约冻结 protocol/schedule/phase/model、完整 Git commit、固定 seeds、隔离输出根、四类 split 的数量/group/SHA-256、预训练权重精确身份、训练参数和 SwanLab 模式。加载器拒绝未知顶层字段、重复或负 seed、非十六进制身份哈希、不安全 protocol 路径段，并保留 Windows/Unix 路径和末尾 `#`。
- 新增 `tools/run_museg_seed.py`：单 seed 输出目录为 `<output-root>/<protocol-id>/<phase>/seed-<seed>`，透传 02 冻结的 source/hash/count/schedule/seed/output/checkpoint/SwanLab 参数；默认拒绝非空目录；resume 必须同时提供 parent run ID 与实际匹配的 checkpoint SHA-256。无论 trainer 返回非零还是进程启动失败，均写入 `launcher.log`、`command.json`、`environment.json`、`train.exit_code` 和 `run_manifest.json`；即使 trainer 返回 0，也必须生成身份完全匹配、未包含 official test 的 `training_result.json`，development/official 还必须提供可读且 SHA-256 匹配的 checkpoint，否则 launcher 把该运行记录为失败。
- 新增 `tools/run_museg_3seed.py`：严格前台顺序启动 manifest 声明的 seed，首个非零退出立即停止且始终写入 `orchestrator.json`；Python/launcher 进程本身无法启动时也以结构化失败记录对应 seed。支持只对一个明确 seed 透传受控 resume 三元组；全部成功后调用训练/validation 汇总器，若汇总失败则改写整体退出状态并写入 `summary_error`，汇总不包含 official test。
- 新增 `tools/summarize_museg_runs.py` 与 `tools/summarize_museg_probe.py`：训练汇总器拒绝缺 run、重复/未声明 seed、损坏 JSON、非零退出、跨 protocol/manifest/phase 产物、official test 未封存和 checkpoint SHA 不匹配；development/official 还必须有有效 checkpoint。probe 汇总结构化记录完成步数、吞吐、step time、显存、loss、AMP scale、OOM/非 OOM 异常。4090 shell 在 OOM 或其他失败后停止更大 batch，同时仍保留部分 probe summary；batch 推荐只在绝对和比例显存阈值均满足时按稳定吞吐选择，并强制人工确认。
- `tools/preflight_train.py` 新增机器可读协议审计：Git branch/commit/dirty/required-commit ancestry，四 split 数量、重复、样本/group 交集、闭合关系与 SHA-256，phase 角色和 sealed official test，预训练权重大小/SHA，输出父目录与碰撞，Python/包/PyTorch/CUDA/cuDNN/驱动/GPU 环境，以及 SwanLab 包、非交互 online 凭据和 warning/error 分级。
- `utils/experiment_tracker.py` 和 `utils/train.py` 写出完整版本化 run config：protocol、phase/run/seed/commit/dirty、数据与模态、split 封存元数据、schedule、optimizer/LR/warmup/weight decay、augmentation/evaluation、checkpoint schema、resume parent/SHA、预训练 SHA 和运行环境。SwanLab 保持 rank 0 only、显式名称优先、`interactive=False`、初始化失败直传和 finish 幂等；训练结束由 rank 0 写 `training_result.json`。
- official test 在训练入口继续使用 `read_test_source=False`：只把 manifest 中已审计的路径、数量和 SHA 写为 `sealed_unread` 元数据；official phase 不创建 validation loader，也不读取 test 清单。

### 11.2 测试先行与静态验证

- 新增 `tests/test_museg_stage03.py`；新增约束先观察到 `3 failed, 7 passed`（退出码 1），随后实现到 `15 passed`（退出码 0）。覆盖参数透传、空/非空输出、Windows 路径和末尾 `#`、resume parent+SHA、顺序/失败即停、指定 seed resume、Git dirty 与生成输出排除、split/phase/权重/输出/SwanLab preflight 错误、启动失败结构化证据、成功但缺失/错绑训练结果、orchestrator 启动/汇总失败、SwanLab 元数据和汇总器失败矩阵。
- `python -m pytest -q tests/test_museg_stage03.py tests/test_training_ops.py tests/test_training_checkpoint.py`：退出码 0，`49 passed, 9 warnings`。
- `python -m pytest -q tests`：退出码 0，`85 passed, 9 warnings`。warning 均来自 02 的 CPU checkpoint 测试使用旧 `torch.cuda.amp.GradScaler` API，不是本阶段失败。
- `python -m compileall -q utils tools tests local_configs`：退出码 0；协议 JSON Schema 已通过 PowerShell `ConvertFrom-Json` 解析。
- `bash -n tools/probe_museg_4090.sh tools/train_museg_4090.sh`：当前系统 Bash 可用，退出码 0；`git -c core.whitespace=cr-at-eol diff --check`：退出码 0。
- 协议 Python 文件未发现新增语言服务诊断。preflight 中 PIL/PyTorch 的编辑器环境缺包提示属于可选运行环境诊断，真实 preflight 会将缺包写为阻塞 error。

### 11.3 Sol 最终复核、边界与剩余风险

- Sol 已逐项复核 protocol 字段契约、launcher 参数透传、失败传播、训练结果与 checkpoint 身份绑定、SwanLab rank 0/显式名称/fail-fast/finish-once，以及 official test `sealed_unread` 的 preflight—launcher—trainer—summary 路径；本阶段端到端接口复核通过。
- 本阶段基础实现已提交为 `e9e3c1c`，门禁 B 冻结 authority 修复已作为独立提交。未修改原始数据，未运行 GPU/CUDA 调用、4090 probe 或任何真实训练，也未在线初始化 SwanLab。
- fake launcher 证明了编排和失败传播；真实 DDP、GPU 遥测、OOM 分类和在线 SwanLab 凭据链路尚未做运行验证，按计划留给 04。
### 11.4 门禁 B 修复记录（已通过最终无卡复核）

- protocol 的 split 身份不再接受仅由 protocol 自描述的路径、数量与 SHA-256；加载、preflight、launcher、三 seed 编排和汇总均从 `data/splits/MUSeg/dev-v1/manifest.json` 这个冻结 authority 派生并校验各角色的清单身份。篡改 authority、冻结 artifact 或 protocol 的重复声明均必须在 GPU 分配前失败。
- `development`、`official` 与 `qualification` 的 source 解析均受冻结 split role 约束；checkpoint phase 只能消费其规定的上游 phase/source，不能通过手填路径跨越 phase 语义。
- qualification 没有可跟踪的通用可运行 manifest：仓库只跟踪模板。`tools/materialize_museg_protocol.py` 在有卡机器为本机参数生成可审计 manifest，生成物受 `.gitignore` 保护并作为运行证据保存；无卡环境只验证该物化和静态 preflight 链路。
- `tests/test_museg_dev_split.py` 以确定性 canonical 重建候选 manifest 并验证其 SHA-256 与冻结 `source_candidate_manifest_sha256` 的关系，补齐来源证明。
- 门禁 B 已签署：冻结 authority 绑定、候选来源 canonical SHA-256 证明、三相位 source 闭合、qualification 模板/物化可追溯性、完整无卡测试、静态检查和干净工作区均已通过正式整体复核；阶段 04 仍必须等待用户开启有卡模式。