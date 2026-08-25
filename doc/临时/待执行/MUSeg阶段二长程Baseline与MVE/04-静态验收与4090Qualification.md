# 04：静态验收与 RTX 4090 qualification 计划

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

阶段 04 当前未启动：无卡环境只能验证模板与物化流程，不得执行本阶段的 GPU 探测、训练、恢复或在线实验记录。

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
