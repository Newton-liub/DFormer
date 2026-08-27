# MUSeg 当前状态与唯一入口

> 状态时间：2026-08-27 01:20 UTC
> 当前阶段：Stage-05 development seed 1 已完成并通过独立 v2 裁决；后评估工具与文档治理已收口，五项本地后评估待运行；全仓稳定指南与逐文件目录已完成
> 本文件是 MUSeg 当前状态的唯一入口；其他阶段计划、审计和正式报告按各自日期保留为历史证据。

## 1. 当前结论

- Stage-01 至 Stage-03 已完成，Gate B 已签署。
- Stage-04 的 batch 选择、3-epoch qualification、checkpoint 连续/恢复等价性和 Gate D 已完成；batch 10 已用于开发长程训练。
- Stage-05 seed 1（seed `772961337`）已完成 500/500 个 epoch，训练进程退出码为 0；云端运行绑定提交 `56a7ed711df2252e6228fc777d7cb92eb2510ef6` 和 protocol `museg-development-long500-v2`。
- 50/50 个 val-dev 验证点齐全；最佳结果为 mIoU `52.84`，对应 epoch 460。该结果是单 seed development 结果，不是三 seed 正式 baseline。
- 最终记录 `63,973` 次有效 optimizer update。相对 64,000 次理论 loop 网格的少量 AMP 跳过不阻塞本次训练完成结论；未来遥测已改为分别记录尝试、完成和跳过计数。
- 原始 v1 `acceptance.json` 仍为失败且未改写，唯一失败项是旧版 `milestones_complete` 遥测规则；原始 `failed.json` 同样保持不变。
- 独立 `acceptance-v2.json` 为 `pass=true`：重新哈希原报告列出的全部 18 项证据（含 12 个 checkpoint），核验 500 个 epoch 末核心日志、50 个验证点、best/final 身份和 official-test 封存状态。
- official test 继续保持 `sealed_unread`：原始裁决记录 `official_test_read=false`，训练结果记录 `official_test_included=false`，运行配置记录 `sealed_unread=true`。
- 已在 `main` 的 `c9ad268b5ee3ab685a4f93c945bfcdd843c49ab9` 基线上完成稳定项目指南 `doc/guides/project/README.md` 与逐文件目录 `doc/guides/project/file-catalog.md`：Git 跟踪路径 1,141/1,141 已说明，排除 0、缺失 0、重复 0。该目录是结构与导航证据，不替代本文件的实时事实。

## 2. 证据取回与云实例状态

- 完整不可变归档已保存到仓库外 `D:\0Project\DFormer-stage05-archive\museg-stage05-seed772961337`。
- 原始归档大小为 `3,865,057,280` bytes，本地 SHA-256 为 `4f6b079b707266ee358d2522fc6e4e034a5380d09ba8c65696df7aaa3e383c66`，与云端清单一致。
- 最佳 checkpoint 与 epoch-500 checkpoint 已保存到 `D:\0Project\DFormer-stage05-evidence\checkpoints`，本地 SHA-256 分别与云端期望值 `b62ca049...`、`0b88ab02...` 一致。
- val-dev 本地后评估包、日志、运行 JSON、原始报告和 `acceptance-v2.json` 已保存到 `D:\0Project\DFormer-stage05-evidence`；不含 official test。
- CompShare 实例 `cpod-1tyvjsiu6ahe` 已在本地哈希复核完成后停止，复查状态为 `Stopped`。
- 本次暴露了自动关机风险：训练完成后未按预期及时停止。未来所有终态都必须先同步证据并核验哈希，再停止实例，同时在控制面预设最晚停止兜底；验收结果不得控制是否继续计费。

## 3. 当前正在进行

- `tools/evaluate_museg_checkpoint.py` 已改为构造模型时跳过单独的预训练初始化，再以 `strict=True` 加载完整 checkpoint；新增回归测试后本地全量测试为 114 passed。
- 在冻结 val-dev、BGR、batch 1、official test 不参与的条件下，运行最佳 checkpoint 的原图、resize 和 sliding 三种后评估，以及 epoch-500 的 resize 和 sliding 后评估。
- 原分辨率整图若超过本地 RTX 5060 Laptop GPU 的 8 GB 显存，将记录为本地 `environment_limit`，不写成模型失败。
- 唯一实时入口维护规则已写入 `.cursor/rules/museg-current-status.mdc` 和 README；每次 MUSeg 对话必须在最终答复前同步持久状态变化，无事实变化时不更新时间。
- 文档职责已进一步收口：Stage-04 云端交接和 2026-08-19 数据处理评审已归入 `doc/reports/`，File Browser/OpenList 操作说明已归入 `doc/guides/cloud/`，`doc/dataset.md` 继续作为稳定数据入口；有效引用已同步更新。最终一致性检查已通过：旧路径无残留，报告索引 JSON 可解析，归档 Canvas 无诊断，完整暂存 whitespace/rename 检查通过。逐文件复核确认普通差异与忽略 CRLF 后的实质差异一致，没有纯换行污染进入提交；本轮迁移与状态记录由同一个本地文档治理提交保存，远端未推送。
- 全仓指南任务已完成静态验收：目录条目与 1,141 个基线路径集合完全一致，全部路径存在，跟踪 JSON 均可解析，两份新指南的本地 Markdown 链接有效，两份新指南及 `doc/guides/README.md` 无尾随空白且以换行结尾，目标文档无静态诊断，`git diff --check` 通过（根 README 的既有尾随空白未被本轮新增行扩散）。本轮未运行训练/GPU、未操作云资源、未读取 official-test 内容；指南、两级导航和本状态更新由本文件所在的本地提交一并保存，远端未推送。
- 五项后评估完成后继续收口正式完成与风险处置报告、Canvas 0.0.10 和发布状态。

## 4. 当前权威链路

1. 当前状态与恢复点：本文件。
2. 本地收口进度与剩余任务：`doc/reports/2026-08-26-museg-stage05-seed1-local-closure-handoff.md`。
3. 运行完成前的历史只读快照：`doc/reports/2026-08-26-museg-stage05-seed1-running-handoff.md`。
4. 当前可移植协议模板：`protocols/museg-development-long500-v2.template.json`。
5. 冻结 development split：`data/splits/MUSeg/dev-v1/manifest.json` 与 `audit-report.json`。
6. 研究口径及其最新处置状态：`doc/main/MUSeg-open-decisions.md`。
7. 训练运行中时点的文档冲突审计：`doc/audits/2026-08-26-markdown-consistency-audit.md`。

## 5. 当前边界

- 不改写原始 `acceptance.json`、`failed.json`、`training_result.json` 或 `liu-test-exp/**`。
- 不把数据、checkpoint、归档、运行大文件或凭据提交到 Git。
- 不把单 seed、单 checkpoint 或后评估几何比较写成三 seed 正式 baseline。
- 不用 official test 选择 epoch、checkpoint、seed、validation 几何、A2 条件或 B2 结构。
- 不再运行已作废的 `tools/mve/run_museg_20epoch_screen.sh`，也不把 `local_configs/MUSeg/DFormerv2_S_20Epoch.py` 当作当前 development 协议。
- seed 2/3 不自动启动；需先完成 seed 1 收口和后续门禁决定。

## 6. 下一次更新条件

完成五项本地后评估并核对结构化 JSON 后，写入正式完成与风险处置报告、更新历史材料的后继指针并生成 Canvas 0.0.10。完成后重新运行全量验证、敏感/大文件检查和远端分支核对，再把状态更新为“本地后评估与发布收口完成”。
