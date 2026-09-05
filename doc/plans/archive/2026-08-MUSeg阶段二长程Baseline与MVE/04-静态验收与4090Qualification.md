# 04：静态验收与 RTX 4090 qualification 计划

> **后继候选计划：** `doc/plans/deferred/2026-09-MUSeg-unexecuted/MUSeg-A2-B2深度有效性/00-总方向规划.md`（该后继计划现已延期，未执行）。

> **文档角色：Stage-04 历史计划与执行记录，不承担实时状态。** 本阶段的 Gate C、3-epoch qualification、连续/恢复等价演练和 Gate D 后来均已完成。以下计划和状态段落按形成时点保留；当前事实、授权和恢复点只看 `doc/main/MUSeg-current-status.md`。
>
> **状态标记核验：2026-08-27。** 正文中的“当前”“下一步”“尚未运行”均表示 2026-08-25 的交接切面，不得据此重跑旧 workload。

> 任务类型：代码资格验收、短跑、恢复演练和 batch 选择。
>
> 模型要求：Terra 可严格执行；遇到异常、指标解释或方案变更必须停止并升级给 Sol。
>
> 本阶段不是 baseline，不产生可用于论文结论的性能结果。

## 1. 前置条件

- 01–03 的代码已分别提交并通过最终无卡复核，门禁 B 已签署；阶段 04 仍须由用户明确开启有卡模式后才能开始。阶段 03 protocol 已强制绑定阶段 01 冻结 authority，候选来源哈希可由 canonical 重建证明。
- qualification 使用跟踪的不可运行模板 `protocols/museg-qualification-v1.template.json`；仅在有卡机器上用 `tools/materialize_museg_protocol.py` 生成本机、可审计的 manifest，并将生成物保存于被忽略的实验输出目录。
- 云端开始前用户明确开启有卡模式。
- 云端仓库目录、py310、数据、预训练权重和输出根已按总索引核实。
- 工作区干净，当前 commit 和 materialized protocol manifest 已记录。

阶段 04 在本文形成时已完成代码/无卡验收、云端 RTX 4090 环境核验、真实 CUDA B1、protocol 物化、完整 preflight 和 SwanLab online smoke，并停在 batch probe 的独立授权边界；当时尚未运行 batch 4/8/12/16 probe、门禁 C、3-epoch qualification、连续/恢复演练或门禁 D。形成时云端证据绑定代码提交 `4f84ee33c93c3c8be83cb2ad029879c26a5346e9`；后续文档提交同步到云端后，继续 exact-commit qualification 前必须重新物化协议并处理身份变化。后续完成状态只看当前入口。

## 2. 静态验收

Terra 按受影响范围执行并留档：

- Python compile；
- `python -m pytest -q tests`；
- 所有新增/修改 Shell 的 `bash -n`；
- `git diff --check`；
- split 工具确定性测试；
- validation/test 封存测试；
- checkpoint schema、best、poly LR 和 resume CPU 测试；
- fake SwanLab、fake launcher 和三 seed 失败传播测试。

禁止直接运行仓库根的无约束 pytest；若测试收集计划外的 `cloud/` 或分布式工具，使用计划指定入口并说明原因。

## 3. 真实模型 B1 混合 batch 回归

补足历史证据缺口：

1. 固定一张官方 train 全背景图和一张普通图，记录样本 ID。
2. 真实 DFormerv2-S + 主头/辅助头配置分别运行：普通+全背景混合 batch、全背景 batch。
3. 检查 forward loss 有限、backward 梯度有限；全背景 batch 的安全归约为图连接零值。
4. 非空条件下与旧归约的数值/梯度等价仍由已有单测保证。
5. 输出小型 JSON 报告，不保存模型权重。

失败时停止，不能进入 GPU batch 探测。

## 4. 云端 preflight

依次验证：

- Git 分支、commit、工作区；
- `/usr/local/miniconda3/envs/py310/bin/python`；
- PyTorch/CUDA/cuDNN/驱动；
- RTX 4090 且显存不少于 20 GiB；
- split 数量、样本/group 零交集和 SHA-256；
- 抽样 RGB/Depth/Label/Depth16；
- 预训练权重大小和 SHA-256；
- 输出目录写入与冲突检查；
- SwanLab online 非交互初始化；
- phase=qualification，明确不读取 official test。

preflight 必须 0 error；每个 warning 先报告并由 Sol/用户决定。

## 5. batch 4/8/12/16 探测

- 固定同一个 qualification seed、同一个 train-dev 列表、60 步、AMP、单尺度、无 compile、单卡。
- 每档独立输出，记录结构化 JSON 和完整日志。
- OOM 后停止更大 batch；非 OOM 错误停止全部。
- 安全阈值：至少 2 GiB 且至少 10% free VRAM。
- 汇总稳定吞吐、step time、allocated/reserved/free、free ratio、loss、AMP scale 和异常。
- 推荐满足安全阈值且稳定吞吐最高的 batch。

**人工门禁 C：** 向用户报告四档对比并等待确认；不得自动短训。

## 6. qualification 短训

确认 batch 后运行独立 qualification：

- 目的只验证数据、优化、验证、日志、best 和 checkpoint 链路。
- 使用 train-dev/val-dev；禁止 official test。
- 建议 2–3 epochs，每 epoch 验证和保存；总 LR 日程专属于 qualification，结果不得与 500-epoch性能比较。
- 前 60 步检查 loss、LR、吞吐、显存、AMP scale 和非有限值。
- 检查 SwanLab config 中 commit、split hash、seed、phase 和输出目录。
- 检查 `best-val-miou.pth`、`latest.pth`、`epoch-N.pth` 可读且 schema 完整。

## 7. 强制中断恢复演练

1. 启动一个独立 3-epoch qualification run，在第 1 个完整 epoch 后正常终止或使用专用测试入口停止。
2. 验证 checkpoint 可读、哈希和已完成 epoch。
3. 用显式 resume 启动新 SwanLab run，完成剩余 epochs。
4. 与连续 3-epoch对照比较：下一 epoch、global step、LR、best、scaler 和关键随机序列。
5. 确认旧输出未覆盖，新 run 记录 parent 和 resume SHA。

任何关键状态不一致都使 02/03 返工；不得“能继续跑就算通过”。

## 8. 停止条件

- dirty Git、split/hash 变化、test 被打开；
- GPU/环境不符、OOM、安全余量不足；
- loss/metric 非有限、AMP scale 持续崩落；
- checkpoint 不可读、resume 重复 epoch 或 LR 不连续；
- SwanLab 丢失关键元数据或静默 offline；
- 数据进程退出、CUDA Xid、长时间无进展。

## 9. 证据与 Git 边界

- 运行证据保存到数据盘 qualification 目录，不进 Git。
- 如无需修代码，本阶段不创建代码提交；只产出验收报告。
- 若发现代码问题，立即停止当前 run，回到对应 01–03 修复、测试、独立提交，再完整重跑 04。

## 10. 完成标准

- 静态测试、真实 B1 回归、preflight、batch 探测、短训和 resume 演练全部通过。
- 用户确认正式 batch。
- qualification 输出明确标注“非 baseline、非正式性能”。
- Sol 解释所有 warning/异常并签署门禁 D，允许进入 05。

## 11. 形成时点交接状态（2026-08-25）

### 11.1 状态结论

- **正在进行：** 阶段 04 已完成代码加固、无卡静态验收、云端 RTX 4090 环境核验、真实 CUDA B1、完整 preflight 和 SwanLab online smoke；当前停在 batch probe 的独立用户授权边界。
- **已完成并验证：** 当前代码范围的 CPU/fake 测试、Python compile、Shell 语法和 diff whitespace 检查；云端 Git/协议/数据/权重/四模态样本/GPU 环境；B1 四种真实模型场景；SwanLab 非交互式 online 初始化。
- **尚未开始：** batch 4/8/12/16 probe、门禁 C、3-epoch qualification、连续/恢复对照和门禁 D。
- 本阶段仍然不是 baseline；B1 报告明确 `official_test_included=false`，qualification preflight 明确 `test_role=sealed_unread`。当前没有正式性能结论。

### 11.2 Git 提交与云端同步边界

- 阶段 04 代码加固的主提交为 `fda65ad5c1dc3a5e6aae4ffdcfa3951ef38a46f7`（`stage04: harden qualification evidence`）。后续修复依次为 `8c9f15822a56be4dd4a55136c3c457facf34f7d2`（冻结测试字节）、`22e217303367ff51e2ece8319822e61f4a668967`（B1 云端绝对路径入口）和 `4f84ee33c93c3c8be83cb2ad029879c26a5346e9`（辅助特征选择）。
- 当前已通过的云端 B1/full preflight 证据绑定 `4f84ee33c93c3c8be83cb2ad029879c26a5346e9`，不是更早的 `185db0e`。
- 这些提交只包含代码、测试和文档；没有包含数据、权重、protocol 物化产物、日志或运行证据。
- 云端运行前后均已确认工作区干净。交接文档的新提交同步后会改变 `HEAD`；继续 exact-commit qualification 前必须重新物化协议，不得修改旧证据中的 commit。

### 11.3 已落地的前置修复

1. **冻结 validation 数量修复。** `local_configs/MUSeg/DFormerv2_S_4090.py` 已显式设置 `C.num_eval_imgs = 318`，不再继承 MUSeg 官方 test 的 1576，避免 preflight 错误拒绝冻结的 `val-dev.txt`。
2. **qualification best checkpoint。** `utils/training_checkpoint.py` 增加 validation phase 判定；`utils/train.py` 允许 `qualification` 与 `development` 在 validation 严格改善时保存 `best-val-miou.pth`，`official` 仍没有 validation loader。
3. **qualification checkpoint 身份。** `tools/run_museg_seed.py` 已将 qualification 纳入成功运行 checkpoint 路径和 SHA-256 校验，不再只约束 development/official。
4. **epoch 边界受控停止。** trainer 与单 seed launcher 已增加 qualification-only 的 `--stop-after-completed-epoch`。该入口在完整 epoch 的 validation、latest、periodic 和 best 保存之后退出，并在 `training_result.json` 记录停止 epoch；它不复用会跳过 validation/checkpoint 的 `--max-train-iters`。
5. **四模态 preflight。** `tools/preflight_train.py` 的抽样检查已从 RGB/Depth/Label 扩展到 RGB/Depth/Label/Depth16；Depth16 必须存在、尺寸一致，并以二维 `uint16` 解码。
6. **SwanLab online smoke。** 非 `--static-only` 的 protocol preflight 使用 `interactive=False` 做在线初始化/立即结束检查；凭据、认证、网络或初始化失败形成阻塞 error。最终真实 SwanLab 0.9.7 online smoke 已通过，见 11.11。
7. **真实模型 B1 入口。** `tools/qualify_museg_b1.py` 只从冻结 official train 确定性选取一张全背景图和一张普通图，分别运行主头和主头+辅助头的 mixed/all-background batch，检查 loss 和梯度有限性以及全背景图连接精确零；只写版本化 JSON，不保存模型权重，代码路径不读取 official test。最终四种真实 CUDA case 已通过，见 11.11。
8. **测试覆盖。** `tests/test_museg_stage04_preconditions.py` 覆盖 validation phase、318 张验证集声明、Depth16 解码和 B1 样本选择；`tests/test_museg_stage03.py` 增加受控 epoch-stop 的 qualification 转发及非 qualification 拒绝测试。

### 11.4 本地前置修复完成记录（已提交）

本轮已在无卡模式补齐以下代码和 CPU/fake 覆盖；尚未进行真实 GPU、数据或在线 SwanLab 操作：

1. **checkpoint 审计与恢复身份。** `utils/training_checkpoint.py` 新增 CPU inspect、规范化 component SHA-256 和字段比较；`tools/inspect_museg_checkpoint.py` 可检查单 checkpoint 或比较两份 checkpoint 及可选 JSONL trace。恢复路径可校验父 checkpoint 的逻辑 `parent_run_id`，而恢复子 run 使用独立 `run_id`。
2. **probe 证据。** trainer/launcher 增加 probe run kind、逐步 JSONL telemetry、精确完成 optimizer step 计数和 10-step warmup；summary 改为 `museg-4090-probe-result-v2`，从 run 工件交叉读取身份，统计第 11–60 步并输出显存阈值、异常类别和可用性。Shell 统一传递阈值、为各 batch 分配不同 run ID，并不再用硬编码阈值推荐 batch。
3. **B1 与受控停止加固。** B1 对全部可训练参数检查有限梯度与全背景零梯度，并在真实运行前校验 exact commit、干净工作区、权重大小和 SHA-256。launcher 对普通 qualification 的完成 epoch、probe 的精确步数和 telemetry 进行区分验证；qualification resume 必须显式给出与 parent 不同的 child run ID。
4. **新增测试。** `tests/test_museg_stage04_probe.py` 覆盖 warmup 排除、精确 60 步与不完整/非单调 telemetry 拒绝；checkpoint 测试覆盖组件状态摘要与 parent run ID 绑定。原有 B1/Depth16/受控停止覆盖仍保留。

SwanLab online smoke 保持 fail-fast，但本地只使用 fake/disabled 路径；实际安装版本的认证方式、远端 record ID/URL 和用户凭据均必须在有卡 cloud preflight 时验证。若需要用户通过 `swanlab login` 或配置 API key，本阶段将先暂停。

### 11.5 形成时点无卡验证

本轮最终无卡验收命令：

```bash
python -m pytest -q tests
python -m compileall -q utils tools tests local_configs
bash -n tools/probe_museg_4090.sh tools/train_museg_4090.sh
git -c core.whitespace=cr-at-eol diff --check
```

结果：

- `python -m pytest -q tests`：退出码 0，`99 passed, 10 warnings`。
- 10 个 warning 均为 `tests/test_training_checkpoint.py` 使用旧式 `torch.cuda.amp.GradScaler` API 产生的 `FutureWarning`；没有失败。
- Python compile、两个 4090 Shell 的 `bash -n` 和 CRLF 兼容的 `git diff --check` 均通过。
- 这些结果仅证明本地 CPU/fake/static 链路；不证明真实 DFormerv2-S、CUDA、4090 遥测、SwanLab online、B1 或连续/恢复等价性。

### 11.6 形成时点下一步与暂停边界

1. 先提交并推送本轮交接文档，在云端 fast-forward 后确认本地/云端文档提交一致且工作区干净。
2. 文档提交改变了 Git commit；保留当前绑定 `4f84ee3...` 的 B1/preflight 证据不动，归档旧 protocol，并为新干净提交重新物化 protocol。严格 exact-commit 门禁下，batch probe 前必须在最终文档提交上重跑 B1 和完整 preflight，不得继续消费旧 protocol 或改写旧报告身份。
3. **当前暂停等待 batch probe 的单独用户授权。** 未获授权前不运行 `tools/probe_museg_4090.sh`、不启动短训或恢复演练。
4. 获得授权后运行 batch 4/8/12/16 各 60 optimizer steps、10-step warmup probe；OOM 停更大 batch，非 OOM 错误停全部。完成后报告结构化证据并停在门禁 C。
5. SwanLab online 需要同一进程环境中的 `SWANLAB_API_KEY`。只能由用户本人隐藏输入；执行者不得自动登录、读取、请求或回显密钥。

### 11.7 明确禁止事项

- 不把本地 `99 passed` 写成阶段 04 已完成；它只覆盖无卡静态验收。
- 不读取 official test，不用 official test 选择 batch、epoch、checkpoint 或任何研究参数。
- 不提交数据、权重、protocol 物化产物、日志或运行证据。
- 不在有卡 B1、preflight、batch 门禁 C、短训和恢复演练全部完成前进入阶段 05。

### 11.8 云端协议物化前的跨平台冻结字节修复

云端静态核验发现，`official-test.txt` 的冻结 SHA-256 对应原始 CRLF 字节，但该文件此前被 Git 作为普通文本规范化为 LF。Windows 的 `core.autocrlf=true` 会在检出时恢复 CRLF，因此本地测试未暴露问题；Linux 云端则检出 LF，导致“干净工作树”和“冻结哈希正确”无法同时成立。修复将该特定冻结文件标记为 `-text` 并按原始 CRLF 字节重新入库，同时增加直接比较跟踪文件与冻结 manifest 输出哈希的回归测试。修复后本地无卡验收为 `100 passed, 10 warnings`，其余 compile、Shell 语法和 diff whitespace 检查均通过；warning 仍全部是既有的 AMP API 弃用提示。

### 11.9 云端静态 preflight 与首次 4090 核验

- 跨平台修复提交 `8c9f15822a56be4dd4a55136c3c457facf34f7d2` 已推送并同步云端；云端 Git 工作树干净。
- 云端 qualification protocol 已物化到 `/root/cloud-ssd/museg-stage04-qualification/protocols/museg-qualification-v1.json`，SHA-256 为 `e3431e511c1a04b4e442f3992f7d37f3f23969ea563c1c3163e3cdb4049042cf`。该 manifest 在下一次修复提交后必须重新物化，旧 manifest 不得继续用于 B1/probe。
- 云端已安装仓库固定的 `swanlab==0.9.7`。`--static-only` protocol preflight 已通过，报告为 `/root/cloud-ssd/museg-stage04-qualification/preflight-static.json`，结果为 0 error、0 warning；SwanLab online 初始化被明确延后，没有登录或创建在线 run。
- 真实硬件核验确认 NVIDIA GeForce RTX 4090、24,564 MiB、驱动 610.57.04；PyTorch 2.1.2+cu118、CUDA 11.8、cuDNN 8.7.0，`torch.cuda.is_available()` 为 true。
- 首次 B1 在任何模型 forward/backward 之前失败：以绝对脚本路径启动时仓库根未加入 `sys.path`，触发 `ModuleNotFoundError: models`。本轮因此没有 B1 JSON、loss、梯度或性能结果，也没有进入完整 preflight、SwanLab online smoke 或 batch probe。入口已补仓库根 bootstrap 和任意工作目录绝对启动回归测试；本地复核为 `101 passed, 10 warnings`，compile、Shell 语法和 diff whitespace 检查通过。修复提交后必须重新物化协议和完整重跑 B1。

### 11.10 第二次 B1 的辅助头结构修复

入口修复提交 `22e217303367ff51e2ece8319822e61f4a668967` 同步云端并重新物化协议后，第二次 B1 已实际完成主头 mixed 和全背景两种 forward/backward；主+辅助头 mixed 在 forward 中失败。根因是 `EncoderDecoder.encode_decode()` 已将双模态 backbone 输出归一化为 RGB 四级特征列表后，辅助头仍使用 `x[0][aux_index]`，错误地在第一级特征的 batch 维取索引 2；B1 固定 batch 为 2，因此触发 `IndexError`。修复改为按特征级选择 `x[aux_index]`，并用 batch=2 的 synthetic 双模态 backbone 测试证明辅助头收到第三级特征而非 batch 切片。本地复核为 `102 passed, 10 warnings`，compile 和 diff whitespace 检查通过。该段记录的是第二次失败时的历史状态；最终结果见 11.11。

### 11.11 形成时点的云端交接状态（B1 与完整 preflight 已通过）

- 云端官方仓库：`/root/rivermind-data/DFormer`；qualification 根：`/root/cloud-ssd/museg-stage04-qualification`。
- 形成时 SSH：`ssh -p 24569 root@cpod-1tyvjsiu6ahe.podtcp.compshare.cn`；该端口仅作历史记录，不得作为当前连接入口。
- 通过证据绑定 Git commit `4f84ee33c93c3c8be83cb2ad029879c26a5346e9`，当时本地/云端工作区均干净。
- 通过证据绑定 protocol SHA-256 `cb741c26897ac32f5a76204180932b03a80ff5ef9fd29ddd218a5ddfb64387e7`。
- RTX 4090 环境：24,564 MiB，driver 610.57.04，Python 3.10.16，PyTorch 2.1.2+cu118，CUDA 11.8，cuDNN 8.7.0。
- B1 报告：`/root/cloud-ssd/museg-stage04-qualification/b1-real-model.json`，SHA-256 `e6dafd8656a783b9569df773381de43ba652e970b7f2b4ee0e971adc455510e5`，`pass=true`。
- B1 四种 case 全部通过：主头 mixed loss `3.204871892929077`、714 个梯度张量有限且无缺失；主头全背景 loss/梯度精确为零；主+辅助 mixed loss `5.559028625488281`、720 个梯度张量有限且无缺失；主+辅助全背景 loss/梯度精确为零。报告明确 `official_test_included=false`。
- 预训练加载仍会输出既有 non-strict key mismatch warning。它未导致 B1 失败，且 B1 JSON 没有 warning 字段；后续说明不得静默省略，也不要把它误判为 CUDA 故障。
- 完整 preflight：`/root/cloud-ssd/museg-stage04-qualification/preflight-full.json`，SHA-256 `c127a792d85b2f315a53b5494b80b68293a435887abeb551de6c973a570e7b4d`，`pass=true`、0 errors、0 warnings。
- full preflight 已验证 Git、config、冻结 split/authority、`phase=qualification`、`test_role=sealed_unread`、预训练身份、输出根、四模态抽样、RTX 4090/CUDA 环境和 SwanLab 0.9.7 非交互式 online 初始化。
- SwanLab online smoke 必须由用户在同一进程环境中隐藏输入 `SWANLAB_API_KEY`；`swanlab login` 的本地登录状态不会通过当前项目的环境凭据门禁。执行者不得自动登录或接触密钥。
- 形成时尚未运行 batch probe、短训或恢复演练；当时记录的下一 GPU workload 是 batch 4/8/12/16 各 60 steps 的 probe。后续这些工作和 Gate D 已完成，不得按本段重复执行。
- 形成时点的跨对话基础知识和 token 节省规则见 `doc/reports/2026-08-25-museg-stage04-cloud-qualification-handoff.md`；当前恢复点只看 `doc/main/MUSeg-current-status.md`。
