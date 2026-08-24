# 06：正式三种子 baseline 与 official test 解封计划

> 任务类型：冻结协议的正式执行与最终统计。
>
> 模型要求：Terra 可模板化顺序执行；**Sol 和用户控制协议一致性、test 解封及结果解释。**
>
> 禁止：运行中调参、从 test 挑 epoch/seed、把失败 seed 替换成更好 seed。

## 1. 前置条件

- 05 的 protocol ID、训练周期、LR schedule、seed 列表和 checkpoint 规则已由用户确认。
- 01–04 的实现 commit 全部包含在正式 commit 中，工作区干净。
- official train 为完整 1595 张；official test 为原始 1576 张，group 零交叉和哈希匹配。
- 04 的 batch、安全余量和 resume 演练有效。

## 2. 正式训练语义

- 每个 seed 都从同一个预训练权重独立开始，不能从 development checkpoint 续训。
- 使用完整 official train；不得把 val-dev 从 1595 中排除。
- 训练期间不读取 official test。
- 若冻结协议要求固定 500 epochs，就以 epoch-500 为预定正式 checkpoint。
- 若冻结协议要求固定 `E*` 或其他 schedule，必须使用 05 验证过的完整 LR 设计，不能临时改 `--epochs`。
- 每 seed 独立输出、独立 SwanLab run；单卡严格顺序。

## 3. 每个 seed 的标准步骤

1. 运行正式 preflight，核对 protocol、commit、split、权重、seed、输出和 SwanLab。
2. 生成命令预览和 `run_manifest.json`，人工确认没有 test evaluator。
3. 启动训练并记录 launcher exit code。
4. 在预定里程碑报告 loss、LR、吞吐、显存、AMP、checkpoint 和异常；不报告 test。
5. 完成后 CPU 验证所有 checkpoint，记录大小、epoch/global step、schema 和 SHA-256。
6. 检查 SwanLab config 与 protocol 完全一致。
7. `git status --short` 必须为空。
8. 当前 seed 验收后，用户批准才启动下一 seed。

建议将三个 seed 的执行记录分别保存，Terra 不得把三个 run 写入同一日志或 checkpoint 目录。

## 4. 中断与恢复

- 只从最高编号、CPU 可读、schema/hash/协议一致的 epoch checkpoint 恢复。
- 先报告完成 epoch、checkpoint SHA、最后日志、故障原因和恢复命令。
- 用户批准后恢复；恢复使用新 SwanLab run 并链接 parent。
- OOM、非有限 loss、Xid、数据错误和代码缺陷不自动恢复。
- 若需修代码，当前及此前受影响的正式 seed 失效；修复独立提交后由 Sol 决定重跑范围。

## 5. test 解封前验收

三个 seed 全部必须具备：

- 退出码 0；
- 预定正式 checkpoint 可读；
- commit、protocol、split、pretrained 和 seed 一致；
- checkpoint SHA 已记录且不同 seed 不应意外相同；
- 无 test 访问记录；
- SwanLab 与本地 manifest 闭合；
- Git 工作区干净。

缺一项则不得 test。

## 6. 人工门禁 F：official test 解封

Sol 向用户提交解封报告，列出三个正式 checkpoint 的路径、SHA、epoch、seed、训练耗时和协议一致性。只有用户明确同意后才运行 test evaluator。

## 7. official test 评估

- 每个预定 checkpoint 恰好按冻结设置评估；不能先多尺度再单尺度后挑更好者。
- evaluator 只读 official test，输出每 seed JSON：PA/mPA 或项目对应 mAcc、mIoU、mF1、per-class IoU/F1、样本数、split/checkpoint SHA、commit、seed、命令。
- 评估失败只允许修复纯评估基础设施；若修复影响数值定义，全部 seed 重评并版本化 evaluator。
- test 结果不能反向改变 epoch、seed、LR、增强或 checkpoint。

## 8. 三种子汇总

汇总：

- 每 seed 指标；
- mean、sample std、min/max；
- 训练时长、吞吐、峰值显存、恢复历史；
- 与论文 DFormer-S 59.74 mIoU 只作外部参考，明确模型为 DFormerv2-S、深度处理和硬件不同，不能写成严格数值复现；
- 与历史 epoch-10 mIoU 20.49 明确区分成熟度和协议。

禁止只报告最好 seed。

## 9. baseline 与后续 MVE 的关系

- 正式 baseline 证明全量数据上的性能和稳定性。
- A2 开发决策仍使用 05 冻结的 development checkpoints + val-dev，避免用 official test 调 B2。
- 06 的 test 结果只用于 baseline 报告，不作为 A2/B2 门槛输入。
- 若以后做正式 B2，对照 B0/B2 都必须按相同正式协议和 paired seeds 重训；不能直接拿本阶段最好 seed 对比。

## 10. 证据和完成标准

- 三 seed 正式 checkpoint、SwanLab、manifest、exit code和 SHA 全部可追溯。
- official test 只在门禁 F 后运行，结果没有用于调参或挑 checkpoint。
- 产出 baseline 报告和机器可读汇总；训练产物不进 Git。
- Sol 签署结果边界并指定 07 使用的 development checkpoint 集合。
- 完成后方可进入 A2 正式扩展。
