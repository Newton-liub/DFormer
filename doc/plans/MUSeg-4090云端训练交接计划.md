# MUSeg 4090 云端训练交接计划（历史阶段二入口）

> **历史快照：** 本文件记录 Stage-04 交接和旧定义替代过程，不再是当前执行入口。当前唯一入口为 `doc/main/MUSeg-current-status.md`；Stage-05 seed 1 的运行中证据见 `doc/reports/2026-08-26-museg-stage05-seed1-running-handoff.md`。
>
> 以下“batch probe 尚未执行”等文字是当时状态，已由后续 Gate D 与 Stage-05 证据取代。

## 1. 当前唯一执行入口

后续新对话或执行模型优先读取：

1. `doc/main/MUSeg-stage04-云端对话基础知识.md`；
2. `doc/plans/MUSeg阶段二长程Baseline与MVE/04-静态验收与4090Qualification.md` 的当前交接章节。

只有发生跨阶段或门禁争议时，再读取 `00-总索引与执行门禁.md`；不要默认重读全部 01–03 和历史日志。阶段 04 完成后，再严格按 `05` 至 `11` 的顺序和门禁执行。

## 2. 计划起点

新计划直接从“阶段二：建立 DFormerv2-S 长程 baseline”开始，阶段一的论文与数据协议核对已经完成。已确认：

- MUSeg 原论文使用官方 train=1595、test=1576；
- 同一 location group 不得跨 split；
- 论文训练协议为 500 epochs、3 个随机种子；
- 当前研究目标模型是 DFormerv2-S，不把它误写成论文 DFormer-S 的严格数值复现；
- 历史 A1/B1 已完成；历史 A2 仅为 epoch-10 checkpoint、16 图的 pilot；B2 尚未实现。

## 3. 被替代的旧定义

本文件旧版把以下内容合并成一次“20-epoch 正式 baseline”：

- 4090 preflight 和 batch 探测；
- 20-epoch 训练；
- 训练期直接使用官方 `test.txt` 验证；
- 从少量 checkpoint 中选择结果。

该定义现已取消。20 epochs 只允许作为 qualification 或工程试跑，不能用于确定成熟 baseline 或正式 A2/B2 结论。官方 test 在开发阶段必须封存。

## 4. 新计划文件

1. `01-开发划分协议与生成工具.md`
2. `02-训练验证Checkpoint与恢复改造.md`
3. `03-三种子编排Preflight与SwanLab改造.md`
4. `04-静态验收与4090Qualification.md`
5. `05-开发长程训练与协议冻结.md`
6. `06-正式三种子Baseline与Test解封.md`
7. `07-A2正式筛查工具扩展.md`
8. `08-A2正式运行与B2门禁.md`
9. `09-B2几何规格与金标准.md`
10. `10-B2实现与零训练筛选.md`
11. `11-B2短微调与完整对照.md`

以上文件位于：

`doc/plans/MUSeg阶段二长程Baseline与MVE/`

## 5. 执行模型要求

- Terra 可以执行已冻结、边界明确、能够机械验收的工具实现、测试、脚本编排和运行任务。
- 涉及数据划分协议、训练/恢复核心语义、曲线与收敛判断、test 解封、A2 门槛解释、B2 几何设计和最终研究结论时，计划文件会明确标注：**继续使用 Sol 模型执行或复核**。
- Terra 遇到计划外改动、协议歧义、异常结果或需要改变门槛时必须停止，不得自行补设计。

## 6. 当前门禁

阶段 04 已完成到 full preflight。当前执行边界为：

- 未获用户单独明确授权前，不运行 batch 4/8/12/16 probe；
- probe 完成后只汇总证据并停在门禁 C，不自动短训；
- 门禁 C 后才可运行 qualification 和连续/恢复演练；完成后停在门禁 D；
- 不读取 official test，不把 qualification/probe 结果写成 baseline 或论文性能；
- 不创建计划外提交，不提交数据、权重、protocol 物化产物、日志、checkpoint 或凭据。

本文件只承担总入口和旧计划作废声明。跨对话基础知识见 `doc/main/MUSeg-stage04-云端对话基础知识.md`，当前执行细节见阶段 04 文档。
