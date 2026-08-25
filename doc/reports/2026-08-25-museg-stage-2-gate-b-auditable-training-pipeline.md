# MUSeg 阶段二 Stage 01–03 项目汇报：从数据冻结到可审计训练链路

- 汇报日期：2026-08-25
- 汇报对象：组会；面向不熟悉语义分割、训练工程和实验审计的老师
- 汇报范围：上次正式报告 `doc/reports/2026-08-25-museg-stage-2-dev-split-freeze.md` 之后，补充 Stage 02、Stage 03、门禁 B 阻塞修复及 01–03 整体复核；为便于理解，同时保留 Stage 01 的必要上下文
- 证据边界：Git 提交 `764ac4e`、`d9ef428`、`e9e3c1c`、`4365edd`，冻结目录 `data/splits/MUSeg/dev-v1`，阶段文档 00–04，以及当前无卡测试结果
- 当前 HEAD：`4365edd7071f5b68e5e3306e9726ae782637fcdd`
- 当前状态：**门禁 B 已签署，数据划分、训练状态恢复、三种子编排和实验身份审计已形成闭环；Stage 04 尚未启动，仍需用户明确开启有卡模式。**

## 一、30 秒结论

这一阶段没有直接给出一个新的模型精度数字，而是解决了“一个训练结果能否被相信、能否被重做、失败后能否正确恢复、是否意外看过最终测试集”这一组基础问题。

如果把正式实验类比为考试：

1. Stage 01 固定了练习题、模拟卷和最终考卷，避免开发过程中反复查看最终答案；
2. Stage 02 让训练过程能够保存完整进度，断电或中断后从同一状态恢复，而不是仅加载一份模型权重后假装继续；
3. Stage 03 建立了考前检查、单次考试、三次独立考试和成绩汇总的统一流程；
4. 门禁 B 复核进一步发现：流程虽然记录了数据名单的“指纹”，但最初仍允许协议自己填写这份指纹。修复后，所有训练协议必须从 Stage 01 已批准的冻结清单派生数据身份，不能另写一套自我证明的名单。

最终形成的不是一条“能跑起来”的命令，而是一条从数据划分、配置、Git 提交、预训练权重、随机种子、checkpoint、运行退出状态到汇总结果都可以追溯的证据链。

## 二、为什么没有直接开始 4090 训练

### 2.1 “程序能运行”和“实验可信”是两件不同的事

深度学习训练往往持续数小时或数天。一个脚本能够开始计算，并不代表最终结果可以用于论文：

- 如果开发阶段读取了官方 test，最终分数可能包含选择偏差；
- 如果 checkpoint 只保存模型参数而没有保存优化器、学习率、随机数和 AMP 状态，中断恢复后得到的是另一条训练轨迹；
- 如果三个随机种子共用输出目录，日志或权重可能相互覆盖；
- 如果 trainer 返回退出码 0，但没有产生与本次协议匹配的结果文件，外层脚本仍可能把空结果当成功；
- 如果协议可以自己声明数据路径和哈希，那么一份看似完整的 JSON 仍可能绕开已冻结的数据基线；
- 如果把 qualification 的短跑结果与 500-epoch 正式 baseline 混在一起，工程验证会被误写成研究结论。

因此，本阶段选择先修复“实验如何被定义和证明”，再消耗 GPU 预算。这样做把最便宜的静态错误拦截在 GPU 分配之前，也避免长训结束后才发现结果无法解释。

### 2.2 本阶段的门禁思想

项目把关键决策拆成不同门禁：

- Gate A：数据划分获批并冻结；
- Gate B：代码链路、协议绑定和无卡验证通过；
- Gate C：4090 batch 探测后由用户确认 batch；
- 后续门禁再分别控制长训、正式协议、official test 解封和 B2 研究方向。

门禁的作用不是增加手续，而是把“机器检查通过”“研究者做出选择”和“允许消耗正式数据或算力”分开记录。当前只完成并签署到 Gate B。

## 三、起点问题：旧链路具体缺在哪里

### 3.1 数据角色混在一起

旧配置在训练期间把 `eval_source` 指向官方 `test.txt`。这意味着 epoch、checkpoint 或方案选择可能无意中受到最终 test 结果影响。即使没有主观挑选，也很难在论文中证明 test 从未参与开发决策。

### 3.2 checkpoint 只能“载入权重”，不能证明等价恢复

可靠恢复至少要回答：

- 完成到第几个 epoch；
- 优化器动量和参数组状态是什么；
- AMP GradScaler 是否刚因溢出跳过了更新；
- Python、NumPy、PyTorch CPU/CUDA 随机数走到哪里；
- 学习率 warmup 和 poly schedule 走到哪个真实 optimizer step；
- 当前 best 指标来自哪一轮；
- 恢复时的数据划分、配置、Git 提交和训练总周期是否仍相同。

只保存 `model.state_dict()` 无法回答这些问题。加载后继续训练可以运行，但不等价于原轨迹继续。

### 3.3 多 seed 编排缺少失败证据

MUSeg 正式协议要求三个 seed。旧 shell 入口缺少统一的 protocol manifest、隔离输出、失败即停、结构化退出记录和最终汇总校验。常见风险包括：

- 前一个 seed 失败，后一个 seed 仍继续；
- 目录已有文件时静默覆盖；
- 子进程无法启动时没有结构化记录；
- trainer 返回 0 但未生成结果或 checkpoint；
- 汇总器读取了错误 phase、错误 seed 或错误 protocol 的历史文件；
- official test 是否被读取只能依赖口头约定。

### 3.4 第一版 protocol 的根本缺口：自我声明不等于权威证明

Stage 03 初版已经为 split 记录路径、数量、group 数和 SHA-256，但这些字段仍由 protocol 自己填写。

这类似于要求考生在答卷上写“我使用的是指定试卷”，却没有让监考系统把答卷与教务处封存试卷进行核对。字段本身很完整，但它的权威来源不闭合。

正式复核因此拒绝直接签署 Gate B，并提出四项要求：

1. protocol 必须强制绑定 Stage 01 冻结 manifest，而不是自描述 split；
2. Stage 01 冻结文件中记录的候选来源哈希必须能够被重建证明；
3. Stage 04 必须有一份真实、机器本地、可审计的 qualification protocol；
4. qualification、development、official 三个 phase 的 source 语义必须从数据冻结一直闭合到 checkpoint 与运行汇总。

## 四、Stage 01：把开发数据和最终 test 真正分开

### 4.1 难点：同一拍摄位置不能拆到训练和验证两边

MUSeg 的图片来自多个矿区和拍摄位置。同一位置的连续画面高度相似。如果按单张图片随机切分，模型可能在验证时看到训练中出现过的地点背景，得到偏乐观的分数。

因此，划分的最小单位不是图片，而是由文件名前四段定义的 location group。一个位置组必须整体进入 train-dev 或 val-dev。

### 4.2 为什么不能简单随机抽 20%

在“组不能拆”的前提下，还要同时兼顾：

- 六个矿区在训练和验证两侧都有覆盖；
- 15 个前景类别的图片出现频次；
- 类别像素量；
- 全背景图片；
- 深度有效率；
- RGB 亮度；
- 单图包含类别数；
- 验证集图片数量接近目标 319。

这些目标相互制约。项目采用固定 seed、精确 Fraction 计算和六级字典序目标，并通过 add/remove/swap 的确定性 best-improvement 搜索生成唯一结果。真实数据中接受 59 次严格改进后，以 `no_strict_improvement` 停止，没有人工挪组、换 seed 或多版本择优。

### 4.3 冻结结果

| 数据角色 | 图片数 | 位置组数 | 作用 |
|---|---:|---:|---|
| train-dev | 1277 | 762 | 开发训练与调试 |
| val-dev | 318 | 196 | 选择 epoch、checkpoint 和实验方案 |
| official test | 1576 | 957 | 最终一次性验收，开发期封存 |

train-dev 与 val-dev 精确闭合为官方 train；三侧样本和位置组均两两零交叉。冻结目录的独立审计为 `102/102 checks`、`pass=true`。

### 4.4 来源哈希为什么还要重建一次

冻结 manifest 记录了获批候选 manifest 的 SHA-256。早期审计能确认“字段存在且格式正确”，但仅凭字段不能证明历史候选内容确实对应这个哈希。

本次修复在测试中从冻结 manifest 移除冻结元数据，把状态还原为 `candidate/pending`，再按照 Stage 01 固定的 canonical JSON 规则重建候选字节。重建 SHA-256 必须精确等于：

`1fa8fbc1613bad8a5d251cf8949f4dd5e5cd2dc9c82ab3cbf5ee4384f556dd63`

这一步把“冻结文件声称自己来自某个候选”升级为“可以由当前仓库中的规则重新构造并验证来源”。

## 五、Stage 02：把训练中断恢复变成可验证的状态恢复

### 5.1 三个 phase 的数据职责

训练入口只接受三类 phase：

| phase | 训练源 | 开发期验证源 | official test |
|---|---|---|---|
| qualification | train-dev | val-dev | 只记录封存身份，不读取 |
| development | train-dev | val-dev | 只记录封存身份，不读取 |
| official | 完整 official train | 无 validation | 只记录封存身份，不读取 |

qualification 是短周期工程资格验证；development 用 validation 冻结训练规则；official 在规则冻结后使用完整官方 train。三者的结果用途不同，不能互相冒充。

### 5.2 显式验证和保存规则

验证调度冻结为 `start + k × interval`，并强制最后一个 epoch 验证；周期 checkpoint 按固定间隔保存，并强制最后一个 epoch 保存。

best 规则固定为 `strict-greater-keeps-earliest`：

- 只接受有限 mIoU；
- 只有严格提升才更新 best；
- 分数相等时保留更早的 epoch；
- 只有 development 写 `best-val-miou.pth`。

这样可以避免同分时因文件写入顺序改变最终选择。

### 5.3 checkpoint 保存的不是一份权重，而是一份训练现场

新 checkpoint schema 为 `dformer-training-checkpoint-v2`，覆盖：

- model、optimizer 和 AMP scaler；
- completed epoch、next epoch 和真实 optimizer step；
- best mIoU 与 best epoch；
- Python、NumPy、PyTorch CPU/CUDA RNG；
- 总 epoch、每轮 iteration、warmup、poly power、base LR；
- train/val/test 的路径、数量和 SHA-256；
- Git commit、run ID、phase、模型和优化器身份；
- 关键配置摘要及其哈希。

恢复时逐项比较这些身份。字段缺失、checkpoint 截断、非有限 best、配置摘要篡改、split 变化或训练协议不兼容都会快速失败，不允许“尽量加载”。

### 5.4 为什么恢复必须使用新输出目录

resume 被视为一条新的、带父运行关系的运行，而不是覆盖原运行。这样可以保留：

- 中断前的日志和 checkpoint；
- 恢复使用的父 checkpoint 及 SHA-256；
- 恢复后的新运行 ID；
- 两段运行之间的因果关系。

这避免了“恢复后日志继续写入原目录，最后无法判断哪部分来自哪次进程”的问题。

## 六、Stage 03：把单次训练扩展为可审计的三种子实验系统

### 6.1 preflight：在占用 GPU 前拒绝错误

preflight 检查以下内容并输出机器可读报告：

- Git 分支、完整 commit、工作区是否干净、最低提交祖先关系；
- 冻结 split 身份、数量、group、哈希和 phase 角色；
- 预训练权重大小与 SHA-256；
- 输出目录是否可写、是否碰撞；
- 配置导入后的 epoch、batch、seed、验证与保存摘要；
- Python、包、PyTorch、CUDA、cuDNN、驱动和 GPU 环境；
- SwanLab 模式、非交互凭据条件和 warning/error 分级。

原则是：路径错误、旧权重、dirty Git、错误 phase 或数据身份错误，都应在创建正式运行目录或启动 trainer 之前被发现。

### 6.2 单 seed launcher：退出码 0 也不自动算成功

`tools/run_museg_seed.py` 为每个 seed 创建独立目录，并记录：

- `launcher.log`；
- `command.json`；
- `environment.json`；
- `train.exit_code`；
- `run_manifest.json`；
- trainer 生成的 `training_result.json` 和 checkpoint 身份。

trainer 返回 0 只是必要条件。launcher 还会检查结果文件的 protocol ID、manifest SHA-256、phase、seed、run ID、official-test 未参与标记以及 checkpoint 路径和哈希。缺一项都把运行判为失败。

### 6.3 三 seed orchestrator：严格顺序、失败即停

`tools/run_museg_3seed.py` 不并发启动三个 seed。它每次只等待当前 seed 完成：

- 当前 seed 非零退出，停止后续 seed；
- Python 或 launcher 无法启动，也写结构化失败；
- resume 只允许一个明确 seed，且必须同时给出 checkpoint、parent run ID 和 SHA-256；
- 全部完成后调用汇总器；汇总失败会反向把整体状态改为失败。

这防止“部分 seed 失败，但最终表格只展示成功 seed”的选择性汇报。

### 6.4 汇总器和 SwanLab：记录完整身份，而不是只记一个分数

汇总器拒绝：缺 run、重复 seed、未声明 seed、损坏 JSON、非零退出、跨 protocol/phase 产物、checkpoint SHA 不匹配或 official test 未封存。

本地 JSON 与 SwanLab config 同时记录数据、模型、commit、seed、训练超参数、增强、验证、checkpoint、resume 和环境。在线初始化仍保持非交互、失败快速终止，不读取或写出 API key。

## 七、Gate B 复核为什么没有在 Stage 03 完成后立即通过

### 7.1 发现的问题

Stage 03 初版从工程行为看已经覆盖 launcher、preflight 和汇总，但 protocol 中四份 split 的身份仍可由 protocol 自己填写。只要伪造路径、数量和哈希之间彼此一致，就可能形成一份“内部自洽、外部不受冻结基线约束”的协议。

这不是普通的字段遗漏，而是证据权威方向错误：应该由冻结 manifest 约束训练协议，而不是由训练协议复述并解释冻结数据。

### 7.2 修复：训练协议 v2 与冻结 authority

`museg-training-protocol-v2` 用必需的 `split_authority` 替代四份可任意自描述的 split。协议加载时必须验证：

- authority 路径就是仓库冻结目录；
- manifest 和 audit-report SHA-256 等于 Gate A 已批准值；
- manifest schema、protocol ID、`candidate_status=frozen`；
- `user_gate_a.status=approved`；
- audit schema、`pass=true`；
- audit 对 manifest SHA-256 的绑定；
- 样本数和 group 数摘要。

train-dev、val-dev、official-test 的身份随后直接从 authority 派生。official train 的哈希、数量和 group 身份也来自冻结 manifest，仅本机路径由 materialization 提供并重新校验。

### 7.3 authority 如何贯穿整条链路

authority 身份现在进入：

1. protocol load；
2. static preflight；
3. 单 seed run manifest；
4. 三 seed orchestrator；
5. training result 校验；
6. 最终 summary。

任一环节看到的 manifest SHA-256、audit SHA-256、split role 或 phase 不一致，结果不能进入后续环节。

## 八、qualification 模板与物化：为什么仓库不直接放一份“可运行 JSON”

### 8.1 通用协议与机器事实不能混为一体

一份真实可运行的 qualification protocol 必须包含：

- 当前机器上的 official train 绝对路径；
- 预训练权重绝对路径、文件大小和 SHA-256；
- 当前干净 Git HEAD；
- 本机输出根；
- 已确认或待探测 batch；
- SwanLab 运行模式和既有 project/workspace。

这些值在本地和云端不同。若仓库跟踪一份写死路径的可运行 JSON，它到另一台机器后要么失效，要么被手工修改，审计身份随之丢失。

### 8.2 两层交付

因此采用两层设计：

1. 跟踪的 `protocols/museg-qualification-v1.template.json`：冻结模型、phase、三 epoch 日程、seed、每 epoch eval/save 和 SwanLab 默认目的地，但保留明确占位符，故意不可直接运行；
2. `tools/materialize_museg_protocol.py`：在有卡机器上读取模板，要求干净 Git，校验冻结 authority、official train 和预训练权重，再生成不可覆盖的本地 manifest 并打印其 SHA-256。

materialized manifest 放入被忽略的实验输出目录，作为本次运行证据保留，而不是提交到仓库。工具不接触 SwanLab API key。

### 8.3 为什么拒绝覆盖

协议一旦物化就代表一次具体实验定义。若确认 batch 后直接改写原文件，probe 和短训会共享同一文件名却对应不同内容。拒绝覆盖能够迫使每个阶段保留独立 manifest 和独立 SHA-256。

## 九、验证结果与可核验交付范围

### 9.1 自动验证

最终无卡验证结果：

| 检查 | 结果 | 说明 |
|---|---|---|
| 冻结目录独立审计 | 102/102 checks | Stage 01 数据与产物审计通过 |
| 聚焦 MUSeg CPU 测试 | 72 passed，9 warnings | 覆盖 authority、物化、编排、checkpoint 和候选来源证明 |
| 完整测试 | 90 passed，9 warnings | `python -m pytest -q tests` |
| materializer 专项 | 3 passed | 覆盖不可运行模板、干净 Git、拒绝覆盖和静态 preflight |
| Python compile | 通过 | `utils tools tests local_configs` |
| JSON 解析 | 通过 | protocol schema 与模板均可解析 |
| Shell 语法 | 通过 | 4090 probe 与训练脚本 `bash -n` |
| Git diff check | 通过 | 使用 CRLF-aware 检查 |
| 最终工作区 | 干净 | Gate B 复核时无未提交修改 |

9 个 warnings 均来自 CPU checkpoint 测试使用旧 `torch.cuda.amp.GradScaler` API 的弃用提示，不是功能失败，也没有触发 CUDA。

### 9.2 交付链条

| 层级 | 主要交付 | 解决的问题 |
|---|---|---|
| 数据协议 | 确定性划分生成器、独立审计器、五份冻结文件 | 开发/test 隔离与位置泄漏 |
| 训练内核 | 显式 source、验证调度、best/latest/periodic、完整恢复 | 中断续训与 checkpoint 可信度 |
| 实验协议 | v2 schema、冻结 authority、qualification 模板/物化 | 实验身份不再自我声明 |
| 运行控制 | preflight、单 seed launcher、三 seed orchestrator | 错误前置拦截、失败传播与目录隔离 |
| 结果证据 | command/environment/run/result/orchestrator/summary JSON | 每个数字可追溯到配置和运行 |
| 测试体系 | 数据、checkpoint、编排、物化与负向篡改测试 | 防止后续改动绕开冻结约束 |

### 9.3 提交边界

阶段成果沿四个独立提交推进：

- `764ac4e`：Stage 01 数据划分、冻结产物和审计；
- `d9ef428`：Stage 02 训练验证、checkpoint 和恢复；
- `e9e3c1c`：Stage 03 protocol、preflight、三种子编排和汇总；
- `4365edd`：Gate B authority 绑定、来源证明、物化工具与整体复核修复。

独立提交使数据规则、训练内核、外围编排和跨阶段修复能够分别复核，也便于定位未来回归来自哪一层。

## 十、哪些取舍是有意为之

### 10.1 official test 只记录身份，不在训练进程中打开

即使只是为了重新计算哈希，训练进程打开 official test 也会削弱“开发期封存”的证据。因此训练只携带已审计的路径、数量和哈希元数据，实际 test 内容留到后续明确解封阶段。

### 10.2 三 seed 串行而不是并发

目标设备是单卡 RTX 4090。并发不仅争用显存，也会让 OOM、吞吐和日志归因复杂。严格串行使每个 seed 的资源和失败边界清晰。

### 10.3 快速失败而不是自动修复

数据哈希、checkpoint 协议、输出碰撞、dirty Git 或结果身份不一致时，系统选择停止，不尝试猜测或修复。自动容错在普通应用中可能友好，但在研究实验中可能悄悄改变实验定义。

### 10.4 qualification 与正式 baseline 分开

qualification 只有 2–3 epochs，用于检查数据、显存、日志、验证、保存和恢复链路。它不能用于比较论文性能，也不能决定研究结论。正式 baseline 仍需后续 development 和 official 三 seed 流程。

## 十一、当前边界与尚未验证事项

### 已完成并验证

- Stage 01 开发划分、Gate A、冻结发布和独立审计；
- Stage 02 训练 source、验证、checkpoint 和 CPU 恢复语义；
- Stage 03 protocol、preflight、单/三 seed 编排、结果校验和汇总；
- v2 frozen authority、候选来源证明和 qualification 物化流程；
- 01–03 整体无卡复核；
- Gate B 签署。

### 尚未在真实设备验证

- RTX 4090 环境、驱动、CUDA 和显存余量；
- 真实 DFormerv2-S 混合 batch 回归；
- batch 4/8/12/16 的吞吐与 OOM 分类；
- 真实 GPU checkpoint 恢复轨迹；
- DDP 进程、GPU 遥测和在线 SwanLab 凭据链路；
- 长程 development 训练、正式三 seed baseline 和 official test 评估。

这些事项没有被写成“已完成”。当前证据只支持代码链路和无卡契约通过，不支持任何新的模型性能结论。

## 十二、下一步计划

### P0：Stage 04 有卡 qualification

前提是用户明确开启有卡模式，并在云端：

1. 保持仓库位于干净提交；
2. 用 tracked template 物化真实 qualification manifest；
3. 对该 manifest 执行静态 preflight；
4. 运行真实模型“普通图 + 全背景图”混合 batch 回归；
5. 探测 batch 4/8/12/16，OOM 后停止更大 batch；
6. 汇报吞吐、step time、显存余量和异常，等待 Gate C；
7. 用户确认 batch 后再进行 2–3 epoch qualification 和恢复演练。

### P1：Stage 05 development 长训

只有 Stage 04 通过并由用户批准后，才在 train-dev/val-dev 上运行长程三 seed，使用 validation 曲线冻结 epoch 和 checkpoint 规则。

### P2：Stage 06 正式 baseline

训练规则冻结后，使用完整 official train 运行三个正式 seed；三个预定 checkpoint、哈希和退出码全部合格后，才申请 official test 解封。

## 十三、组会可用的一句话总结

**本阶段把 MUSeg 实验从“有一个训练脚本”推进到“数据、训练状态、三次重复实验和最终结果都有统一身份证明，并且任何绕开冻结数据或吞掉失败的行为都会在正式训练前被拒绝”；因此 Gate B 已通过，但模型性能结论仍等待有卡 qualification 和后续长程实验。**

## 十四、复核入口

- 总索引与门禁：`doc/临时/待执行/MUSeg阶段二长程Baseline与MVE/00-总索引与执行门禁.md`
- Stage 01：`doc/临时/待执行/MUSeg阶段二长程Baseline与MVE/01-开发划分协议与生成工具.md`
- Stage 02：`doc/临时/待执行/MUSeg阶段二长程Baseline与MVE/02-训练验证Checkpoint与恢复改造.md`
- Stage 03：`doc/临时/待执行/MUSeg阶段二长程Baseline与MVE/03-三种子编排Preflight与SwanLab改造.md`
- Stage 04：`doc/临时/待执行/MUSeg阶段二长程Baseline与MVE/04-静态验收与4090Qualification.md`
- 冻结数据证据：`data/splits/MUSeg/dev-v1`
- Protocol v2：`tools/museg_protocol.py`、`tools/museg_protocol.schema.json`
- Qualification 物化：`protocols/museg-qualification-v1.template.json`、`tools/materialize_museg_protocol.py`
- 训练恢复：`utils/training_checkpoint.py`
- 编排与验证：`tools/preflight_train.py`、`tools/run_museg_seed.py`、`tools/run_museg_3seed.py`、`tools/summarize_museg_runs.py`