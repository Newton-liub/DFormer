# MUSeg 研究分支治理

> **文档角色：** 稳定项目治理指南，不承担实验实时状态或运行授权。
> **核验时点：** 2026-09-05。
> **实时入口：** `doc/main/MUSeg-current-status.md`；研究选择见 `doc/main/MUSeg-open-decisions.md`。
> **后继关系：** 具体运行事实和恢复点以实时状态为准；历史计划与报告按各自形成时点解释。

## 1. 稳定起点

- `main` 与本地 annotated tag `museg-research-base-v1`共同标识已整理的 MUSeg 稳定研究起点；tag 创建后不得移动或复用。
- 稳定起点包含已完成的 single-seed RGB Quick-B0 身份、必要的文档治理和历史证据入口，但不自动授权任何新实验、训练或云操作。
- Quick-B0 的最终 checkpoint 是 epoch 420；其指标和哈希只以实时状态与正式报告为准。

## 2. 新研究方向

每篇论文、独立研究问题或实质性模块都使用独立分支，推荐命名为 `research/<topic>`。新分支必须从 `museg-research-base-v1` 创建，而不是从另一个研究方向的中间提交创建。

新方向开始前必须建立新的：

- config 身份：明确模型、数据、训练预算和输入契约；
- protocol 身份：绑定 Git 提交、split、pretrained、seed、evaluator 和 official-test 边界；
- 证据身份：为日志、checkpoint、manifest、报告和裁决提供独立路径与哈希。

如果方向修改训练预算、优化器、增强、数据职责、输入契约或主 evaluator，必须说明它已经不再是与稳定 B0 严格配对的消融；需要时在新 protocol 中重训匹配的 B0 对照。

## 3. 公平比较边界

- 不得从 B0 最终 checkpoint 续训模块后，将结果称为公平模块消融。
- 后续模块训练必须从同一官方 pretrained 独立初始化，并按新 protocol 冻结 seed、数据顺序、预算、优化器、增强、checkpoint 规则和主 evaluator。
- 单 seed 结果可以用于探索和淘汰，但小增益或重要结论需要成对重复或额外 seed；不能把单 seed 结果写成论文级随机方差结论。
- 历史指标、原始 `acceptance.json`、`failed.json`、`training_result.json`、日志和报告不为新口径改写；修订使用新 protocol、报告或裁决。

## 4. official test 与运行授权

official test（最终测试集）在开发、模型选择、阈值冻结、方向筛选和恢复过程中保持 `sealed_unread`。任何解封必须通过独立门禁，并取得与该操作匹配的明确授权。

分支创建不等于运行授权。代码修改、CPU 定点检查、GPU、训练、长耗时评估、云资源和 official test 应分别按风险取得授权；计划文件不能单独触发这些操作。

## 5. 合并回 `main`

合并前应直接核对：

1. 研究分支的 config、protocol、split、checkpoint、evaluator 和证据身份；
2. 代码/文档差异与历史文件是否保持不可变；
3. 仅运行覆盖本次行为风险的定点检查，并在报告中区分未运行的高成本验证；
4. 当前状态、开放决策、报告索引和导航是否已同步；
5. official-test 标志、数据边界和提交身份是否一致。

合并只表示变更进入项目历史，不表示结果自动成为正式论文结论。新的稳定研究起点应在合并后重新创建不可移动的 annotated tag，并在状态文件登记。
