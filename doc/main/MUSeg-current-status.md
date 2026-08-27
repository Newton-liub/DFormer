# MUSeg 当前状态与唯一入口

> 状态时间：2026-08-27 08:41 UTC
> 当前阶段：计划与预处理/后评估契约重构已完成 CPU 验收；Protocol Gate 尚未通过，五项 val-dev 后评估、颜色 provenance 与 paired calibration 待执行
> 本文件是 MUSeg 当前状态的唯一入口；其他阶段计划、审计和正式报告按各自日期保留为历史证据。

## 1. 当前结论

- Stage-01 至 Stage-03 已完成，Gate B 已签署。
- 活动计划已于 2026-08-27 收敛：历史 01–04 保留，旧未执行 05–11 由新 `05-协议校准与Development-B0收口.md`、`06-MVE问题验证与B2条件分支.md`、`07-模块配对筛选与GateE冻结.md`、`08-正式三种子与Official-Test解封.md` 取代；活动门禁统一为 Protocol Gate、MVE Evidence Gate、Gate E、Gate F。迁移审计见 `doc/audits/2026-08-27-museg-plan-conflict-migration-audit.md`。
- 颜色/归一化契约已显式进入配置、protocol v3、materializer、launcher、run manifest/config、preflight、production loader 和 post-evaluator；历史 v2 protocol 仍可读取并标记补录来源。当前 seed 1 仍是 legacy BGR 历史事实，未来颜色谱系已重新打开；具体 pretrained 来源通道语义尚未由发布资产证明，保持待核验。
- post-evaluator 已修正为所有 geometry 保留原始 Label：resize 只改变模型输入并恢复 logits 到原图计分，sliding 使用全图覆盖 count map；legacy `utils/val_mm.py` 的 sliding list/tensor 接口与小图 geometry 已修复。聚焦测试 31 passed，全量 CPU 测试 118 passed，compile 通过；没有产生新实验指标。
- Stage-04 的 batch 选择、3-epoch qualification、checkpoint 连续/恢复等价性和 Gate D 已完成；batch 10 已用于开发长程训练。Stage-04 当前没有未完成执行项；376/384 次有效 update 的旧遥测差异只保留为历史限制，不要求重跑。
- Stage-05 seed 1（seed `772961337`）已完成 500/500 个 epoch，训练进程退出码为 0；云端运行绑定提交 `56a7ed711df2252e6228fc777d7cb92eb2510ef6` 和 protocol `museg-development-long500-v2`。直接核验该提交确认：基线已使用历史 MVE 的 B1 `safe_masked_mean` 数值稳定性修复；B2 depth-validity mask/gating 尚未实现，未进入本次基线。
- Stage-05 整体尚未完成：seed 1 五项后评估、曲线/几何审查和本地发布收口仍待完成。经用户于 2026-08-27 确认，development seeds 2/3 不再作为当前前置任务，暂缓到架构和消融组合冻结后的正式阶段；届时 B0 与最终模块必须使用同一三 seed 成对从头训练。
- 50/50 个 val-dev 验证点齐全；最佳结果为 mIoU `52.84`，对应 epoch 460。该结果是单 seed development 结果，不是三 seed 正式 baseline。
- 最终记录 `63,973` 次有效 optimizer update。相对 64,000 次理论 loop 网格的少量 AMP 跳过不阻塞本次训练完成结论；未来遥测已改为分别记录尝试、完成和跳过计数。
- 原始 v1 `acceptance.json` 仍为失败且未改写，唯一失败项是旧版 `milestones_complete` 遥测规则；原始 `failed.json` 同样保持不变。
- 独立 `acceptance-v2.json` 为 `pass=true`：重新哈希原报告列出的全部 18 项证据（含 12 个 checkpoint），核验 500 个 epoch 末核心日志、50 个验证点、best/final 身份和 official-test 封存状态。
- official test 继续保持 `sealed_unread`：原始裁决记录 `official_test_read=false`，训练结果记录 `official_test_included=false`，运行配置记录 `sealed_unread=true`。
- 已在 `main` 的 `c9ad268b5ee3ab685a4f93c945bfcdd843c49ab9` 基线上完成稳定项目指南 `doc/guides/project/README.md` 与逐文件目录 `doc/guides/project/file-catalog.md`：Git 跟踪路径 1,141/1,141 已说明，排除 0、缺失 0、重复 0。该目录是结构与导航证据，不替代本文件的实时事实。

## 2. 证据取回与云实例状态

- 完整不可变归档当前可访问于仓库内暂存路径 `D:\0Project\DFormer\cloud\DFormer-stage05-archive\museg-stage05-seed772961337\museg-stage05-seed772961337-original.tar`；该路径是用户从 `D:\0Project\DFormer-stage05-archive` 手动移动后的当前位置。文件大小为 `3,865,057,280` bytes，本地 SHA-256 为 `4f6b079b707266ee358d2522fc6e4e034a5380d09ba8c65696df7aaa3e383c66`，与 sidecar 清单一致。归档内已直接列出 `best-val-miou.pth` 与 `epoch-500.pth`，但两个 checkpoint 尚未从归档提取为独立本地实体。
- 当前 `D:\0Project\DFormer\cloud\DFormer-stage05-evidence` 仍主要包含日志、配置、验收 JSON、manifest 和 split 列表；完整 val-dev 数据包实体及独立 checkpoint 尚未在该目录发现。
- CompShare 实例 `cpod-1tyvjsiu6ahe` 已在本地哈希复核完成后停止，复查状态为 `Stopped`。
- 本次暴露了自动关机风险：训练完成后未按预期及时停止。未来所有终态都必须先同步证据并核验哈希，再停止实例，同时在控制面预设最晚停止兜底；验收结果不得控制是否继续计费。

## 3. 当前正在进行

- `tools/evaluate_museg_checkpoint.py` 已使用 `strict=True` 加载完整 checkpoint、跳过单独 pretrained 初始化，并升级到 post-evaluation v2 契约：显式 input contract，所有 geometry 在原始 Label grid 计分，报告 input/metric geometry。
- 在完成提交边界复核并获用户单独授权后，先运行同一 val-dev、batch 1、official test 不参与的五项后评估；颜色三臂诊断和真正 paired calibration 使用独立 protocol，不把 seed 1 的 `52.84` 与不同颜色谱系直接比较。
- 原分辨率整图若超过本地 RTX 5060 Laptop GPU 的 8 GB 显存，将记录为本地 `environment_limit`，不写成模型失败。
- 唯一实时入口维护规则已写入 `.cursor/rules/museg-current-status.mdc` 和 README；每次 MUSeg 对话必须在最终答复前同步持久状态变化，无事实变化时不更新时间。
- 文档职责已进一步收口：Stage-04 云端交接和 2026-08-19 数据处理评审已归入 `doc/reports/`，File Browser/OpenList 操作说明已归入 `doc/guides/cloud/`，`doc/dataset.md` 继续作为稳定数据入口；有效引用已同步更新。最终一致性检查已通过：旧路径无残留，报告索引 JSON 可解析，归档 Canvas 无诊断，完整暂存 whitespace/rename 检查通过。逐文件复核确认普通差异与忽略 CRLF 后的实质差异一致，没有纯换行污染进入提交；文档治理提交 `c9ad268b5ee3ab685a4f93c945bfcdd843c49ab9` 已直接核验为当前 `origin/main`。
- 2026-08-27 经用户确认删除语义重复且含过时“当前门禁”的 `doc/plans/MUSeg-4090云端训练交接计划.md`；其导航和三份历史报告中的旧引用已改为当前状态入口、Stage-04 计划或日期化报告。`00`/`04` 的形成时点边界、`05` 的训练裁剪与 validation 几何区别、`08` 已处置的自然证据门禁、seed 1 预启动报告的 validation 几何勘误及运行中交接的后继状态均已澄清；`doc/guides/README.md` 与 `doc/plans/README.md` 已规定统一文档状态头。Stage-01 至 Stage-04 的设计证据继续保留，不据此重跑历史 workload。
- 全仓指南任务的历史静态验收和提交 `1fbb0da` 保持不变；本轮新增计划/代码迁移是在其后的未提交差异，未改写该指南基线计数。当前工作区还包含用户此前已有的文档治理和旧交接计划删除/引用更正。
- 本轮计划/代码迁移仍是工作区未提交差异，并叠加在用户此前已有的文档治理修改上；未运行训练、GPU、五项后评估、颜色诊断、云资源或 official test。当前 `main` 相对 `origin/main` ahead 1，远端未推送。
- Protocol Gate 的下一证据顺序为：提交边界复核后运行五项 val-dev 后评估，完成曲线/geometry 审查，再执行 provenance/三臂诊断与独立 paired 颜色 calibration。Gate 通过后，通用模块进入新 Stage-07；若考虑 B2，先在新 Stage-06 完成 `A2-pass + 用户批准`、规格/金标准、zero-train 和 short，再回到 Stage-07。Gate E 后才启动 formal B0/最终 variant paired 三 seed，Gate F 后才一次性读取 official test。

## 4. 当前权威链路

1. 当前状态与恢复点：本文件。
2. 本地收口进度与剩余任务：`doc/reports/2026-08-26-museg-stage05-seed1-local-closure-handoff.md`。
3. 运行完成前的历史只读快照：`doc/reports/2026-08-26-museg-stage05-seed1-running-handoff.md`。
4. 历史 seed 1 协议模板：`protocols/museg-development-long500-v2.template.json`；新运行使用显式 input contract 的 protocol v3。
5. 冻结 development split：`data/splits/MUSeg/dev-v1/manifest.json` 与 `audit-report.json`。
6. 研究口径及其最新处置状态：`doc/main/MUSeg-open-decisions.md`。
7. 当前活动计划：`doc/plans/MUSeg阶段二长程Baseline与MVE/00-总索引与执行门禁.md` 与新 05–08。
8. 本轮计划/代码迁移审计：`doc/audits/2026-08-27-museg-plan-conflict-migration-audit.md`。
9. 训练运行中时点的历史冲突审计：`doc/audits/2026-08-26-markdown-consistency-audit.md`。

## 5. 当前边界

- 不改写原始 `acceptance.json`、`failed.json`、`training_result.json` 或 `liu-test-exp/**`。
- 不把数据、checkpoint、归档、运行大文件或凭据提交到 Git。
- 不把单 seed、单 checkpoint 或后评估几何比较写成三 seed 正式 baseline。
- 不用 official test 选择 epoch、checkpoint、seed、validation 几何、A2 条件或 B2 结构。
- 不再运行已作废的 `tools/mve/run_museg_20epoch_screen.sh`，也不把 `local_configs/MUSeg/DFormerv2_S_20Epoch.py` 当作当前 development 协议。
- development seeds 2/3 已明确暂缓，不在 seed 1 收口后自动启动；快速筛选只使用 val-dev 和单 seed 配对 B0/模块，正式三 seed 延后到架构与消融组合冻结后，并要求同时重跑 B0 与最终模块。

## 6. 下一次更新条件

先在提交边界复核和用户单独授权后运行五项 val-dev 后评估，核对 post-evaluation v2 JSON、318 样本、checkpoint/split SHA、原始 metric grid 与 `official_test_included=false`；完成曲线/geometry 审查。随后补 pretrained provenance、三臂敏感性诊断和独立 paired 颜色 calibration，只有证据闭合后才冻结 channel order/normalization/validation geometry 并通过 Protocol Gate。Gate 通过后按新 Stage-06/07 恢复 MVE 或模块 screening；Gate E 后才授权 formal paired 三 seed，Gate F 后才一次性读取 official test。任何训练、指标、checkpoint、门禁、云状态、提交或发布变化都在同一对话更新本文件。
