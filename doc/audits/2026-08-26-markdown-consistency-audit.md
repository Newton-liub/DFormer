# DFormer Markdown 一致性审计

> 后继状态：本审计冻结于 Stage-05 seed 1 运行中时点；训练完成后的实时事实以 `doc/main/MUSeg-current-status.md` 和 `doc/reports/2026-08-26-museg-stage05-seed1-local-closure-handoff.md` 为准。

- 审计日期：2026-08-26
- 范围：项目自有 README、MUSeg 数据文档、正式报告、阶段计划和云端操作指南
- 明确排除：`liu-test-exp/**`、`mmseg/.mim/**` 第三方 README、`.cursor/**` 与 `.agents/**` 内部 Skill/Agent 说明
- 处理原则：历史报告保留当时事实；只增加状态标记、替代关系和当前入口，不用今天的结论重写过去

## 1. 当前权威结论

当前唯一状态入口为 `doc/main/MUSeg-current-status.md`。Stage-04 Gate D 已完成；Stage-05 seed 1 在 `56a7ed711df2252e6228fc777d7cb92eb2510ef6` 上以 `museg-development-long500-v2` 运行，云端 materialized protocol 原始 SHA-256 为 `47d1a52a20bbb73b9d1a1b609819335d07ca84b49e8373d4046318b414ad324d`。2026-08-26 03:33 UTC 的只读证据显示 epoch 445 已完成，运行中 best val-dev mIoU 为 52.41；该值不是最终 baseline。

## 2. 已确认并处理的冲突

### 2.1 旧入口仍称 Gate C/D 未完成

影响：执行者可能重复 batch probe、qualification 和恢复演练。处理：Stage-04、总交接和总索引保留为历史执行快照，并统一链接当前状态；当前入口改为 Stage-05 running handoff。

### 2.2 最新正式报告索引落后

影响：自动汇报边界退回 Gate B。处理：`report-index.json.latestFormalReport` 更新到本次 Stage-05 running handoff；旧报告保留在列表中。

### 2.3 旧 20-epoch 路径可能读取 official test

影响：旧配置的 `eval_source=test.txt` 与 development test 封存政策冲突。处理：旧脚本和配置在文档中明确标成禁用历史入口；当前入口只允许 protocol 驱动链路。

### 2.4 旧 MVE 从 official test 抽 64 张

影响：破坏 development 阶段 test 隔离。处理：旧设计标为已被阶段二协议替代；正式 A2/B2 开发筛查只使用 `val-dev`，official test 在最终模型和协议冻结后经门禁一次性解封。

### 2.5 500-epoch 参数与跟踪配置默认值不同

影响：直接导入 `DFormerv2_S_4090.py` 会看到 batch 8/20 epochs 等 qualification 默认值。处理：文档明确运行事实源是 materialized protocol；归档 v2 可移植模板，并保留云端原始 SHA。

### 2.6 Resume run ID 语义冲突

影响：旧说明要求父子 run ID 严格相同，但恢复子 run 必须使用新 ID。当前事实：加载 checkpoint 时严格核验父 run ID；连续/恢复等价比较只允许 `run_id` 不同，其他 protocol 与状态保持一致。

### 2.7 Qualification best checkpoint 描述过时

影响：旧说明称只有 development 保存 best。当前事实：qualification 和 development 都属于 validation phase，可保存 `best-val-miou.pth`；official phase 无 validation loader。

### 2.8 MUSeg group 数来源混写

影响：1916 与 1915 被误认为同一统计冲突。处理：论文/官方说明报告 1916 个采集位置；当前发布文件按文件名前四段可解析出 1915 个唯一 groups。两者分别保留来源。

### 2.9 计划 iteration 与 successful update 混写

影响：3×128=384 planned iterations 与报告 376 successful optimizer steps 看似冲突。处理：暂按 attempted iteration 与 successful update 两个口径登记，等待 AMP/GradScaler trace 解释 8 次差异。

### 2.10 README 与实际入口不一致

影响：新人可能按上游 NYUv2/SUNRGBD 多卡脚本运行当前 MUSeg。处理：README 顶部增加项目分流，标明根脚本仅为上游示例，当前 MUSeg 使用审计 protocol 入口。

### 2.11 OpenList 命令错误

影响：`server DATA start --force-bin-dir` 混合两个子命令，数据目录不按文档生效。处理：改为 `openlist --data /opt/openlist/data --log-std server`，同步修正进程检测、初始化和密码闭环。

### 2.12 Canvas 路径与索引不一致

影响：0.0.1–0.0.7 移到 `doc/canvases/old/` 后索引失效。处理：确认 old/ 为长期归档，默认发布只处理根目录当前 Canvas；索引改为归档路径和 `reference` 状态。

## 3. 历史正确但不再代表当前状态

- 2026-08-19 的数据评审和最短验证路径：保留为阶段一审计与预注册设计，其中“无 MUSeg config、无 development validation、全背景 loss 未处理”等状态已被后续工作解决。
- 2026-08-21 的 MVE 报告：保留 epoch-10/16 图 pilot 指标；其“20-epoch 正式 baseline”和 test 评估下一步已取消。
- Stage-04 计划和云端基础知识：保留 qualification 的失败史、证据路径和门禁定义；其“下一步 probe”状态已被 Stage-05 取代。
- 2026-08-26 prelaunch 报告：保留启动前快照；当前 run 已使用 v2 protocol 启动，不能继续把“未启动”当实时状态。

## 4. 待讨论，不在本轮改代码

1. validation 原分辨率整图、固定 480×640 resize 或 480×640 sliding-window；
2. MUSeg 保持 BGR 还是切换 RGB；
3. 自然无效深度分层是否为进入 B2 的硬门槛；
4. 是否重命名 `run_kind=qualification` 这个“非 probe”历史字段；
5. qualification 的 376/384 差异如何由 AMP trace 解释。

详见 `doc/main/MUSeg-open-decisions.md`。

## 5. 已核实一致的核心事实

- 总样本 3171；官方 train/test 为 1595/1576。
- train-dev 1277/762 groups，val-dev 318/196 groups，official test 1576/957 groups。
- train-dev 与 val-dev 精确闭合为 official train，样本与 group 两两零交叉。
- 冻结 audit 为 `pass=true`、102/102 checks。
- 原始标签 0 为 background，1–15 为前景；训练映射为 255 ignore 与 0–14。
- 深度映射为 `round(depth16 × 255 / 13932)`，原始无效 0 保留。
- 历史 A2 pilot 指标在正式报告之间一致；B2 尚未实现。
- 20.49、A2 子集指标、qualification 7.54 和当前运行中 52.41 来自不同 checkpoint、split/子集和目的，不应横向比较。

## 6. 文档职责规则

- `doc/main/MUSeg-current-status.md`：唯一实时入口。
- `doc/reports/*.md`：按日期冻结的正式事实快照。
- `doc/plans/**`：计划与历史执行记录，不承担实时状态。
- `doc/main/MUSeg-open-decisions.md`：尚未冻结的研究口径。
- `protocols/*.template.json`：可移植协议事实源；云端 materialized manifest 是具体 run 的机器身份。
- Canvas 只重排 Markdown 已核实内容，不增加事实。
