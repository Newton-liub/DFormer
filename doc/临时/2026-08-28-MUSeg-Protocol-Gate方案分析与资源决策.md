# MUSeg Protocol Gate 全套方案、当前进度与资源决策说明

> 文档角色：临时分析与新窗口交接材料，不替代实时状态、活动计划或运行授权。
> 核验时间：2026-08-28 02:35 UTC。
> 实时事实入口：`doc/main/MUSeg-current-status.md`。
> 当前结论：五项 val-dev 后评估和固定 checkpoint 三臂颜色诊断已经完成；paired calibration 尚未定义预算，也没有启动云实例或训练。已核实上游 pretrained 权重的彩色通道语义为 RGB，而历史 MUSeg seed 1 下游训练实际采用 BGR；这两个事实不能混为一谈。Protocol Gate 仍未通过。

## 1. 先用一句话说明整个项目现在在做什么

现在不是在继续堆新模型，而是在先把“模型到底看到了什么输入、用什么尺寸验证、结果是否可比”这几个基础口径固定下来。

如果颜色顺序、归一化方式或验证尺寸没有先冻结，后面即使某个模块提高了 mIoU，也无法判断提升来自模块本身，还是来自输入颜色、图像缩放或评估方式改变。因此当前 Stage-05 的任务是先通过 **Protocol Gate（协议门禁）**，再允许后续模块筛选和正式训练。

## 2. 关键专业名词的大白话解释

### 2.1 B0、B1、B2

- **B0**：不包含待研究新模块的基础模型，是所有候选方案的对照组。
- **B1**：`safe_masked_mean` 数值稳定性修复，用于避免极端 mask 条件下出现非法数值。它已经进入基础训练链，不再作为性能模块反复消融。
- **B2**：尚未实现的 depth-validity gating，即让模型识别“这块深度值是否有效”，再决定是否使用对应的深度信息。它必须先经过 A2 问题证据门禁，不能直接加入正式模型。

### 2.2 train-dev、val-dev、official test

- **train-dev**：开发阶段训练用的 1,277 个样本。
- **val-dev**：开发阶段比较方案用的 318 个样本。本轮全部后评估和颜色诊断只使用它。
- **official test**：最终测试集，共 1,576 个样本。当前继续封存，不能参与颜色、尺寸、checkpoint、模块或 seed 选择。

这样做的原因是：如果反复查看 official test 再调整方案，test 就变成了调参集，最后数字会偏乐观，不能代表真实泛化能力。

### 2.3 checkpoint

checkpoint 是训练某一时刻保存的完整模型状态。本轮使用两个：

- `best-val-miou.pth`：历史在线验证中表现最好的 epoch 460 checkpoint；
- `epoch-500.pth`：500 epoch 训练结束时的 checkpoint。

### 2.4 channel order 与 normalization

彩色图片通常口头叫 RGB，但必须区分“权重上游预训练时的通道语义”和“MUSeg 下游数据 loader 实际送入模型的通道顺序”。

- **channel order**：三个颜色通道在张量中的实际排列顺序，例如 BGR 或 RGB。
- **normalization**：把像素按指定 mean/std 做标准化。mean/std 也是按数组位置应用的，因此颜色顺序变了，统计数组是否反向也会影响数值。
- **pretrained 上游语义**：当前 `DFormerv2_Small_pretrained.pth` 是官方 DFormer 的 RGB-D ImageNet-1K 预训练权重；其上游彩色图像按 RGB 语义读取，并按 RGB 顺序使用 ImageNet mean/std。
- **MUSeg 下游契约**：历史 seed 1 使用 OpenCV 读取彩色图并保留 BGR 数组顺序，再按位置使用 `[0.485, 0.456, 0.406]` 和 `[0.229, 0.224, 0.225]`。因此历史 checkpoint 的下游输入契约是 legacy BGR。

上游权重是 RGB，并不意味着历史 MUSeg checkpoint 可以直接改用 RGB 推理，也不意味着未来 MUSeg 必须采用 RGB。改变下游颜色契约需要重新训练并重新建立结果身份。

### 2.5 validation geometry

validation geometry 是模型验证时如何处理图像空间尺寸：

- **original-full**：整张原图直接输入，输出在原始 Label 网格计分；
- **resize-480x640**：先把输入强制缩放到 480×640，再把 logits 恢复到原始 Label 网格计分；
- **sliding-480x640**：在原图上用 480×640 窗口滑动推理，再把重叠区域 logits 平均。

这里的 **Label grid** 是人工标签所在的原始像素网格。本轮 evaluator 保证三种方式最终都在原始 Label grid 上计分，避免因为缩放 Label 而改变真实计分支持。

### 2.6 paired calibration

paired calibration 是“成对校准训练”。它不是把同一个历史 checkpoint 分别用 BGR 和 RGB 推理，而是从同一个已核验的官方 RGB-D pretrained 权重出发，分别训练两条下游颜色谱系。比较 BGR 和 RGB 时，两臂必须做到：

- 使用同一个 `DFormerv2_Small_pretrained.pth`，其上游预训练通道语义固定为 RGB；
- 使用相同 seed；
- 使用相同样本顺序；
- 使用相同训练预算、batch、优化器、学习率和增强；
- 使用相同 checkpoint 选择规则与 evaluator；
- 唯一实验变量是下游颜色/归一化契约：legacy BGR 臂保留 OpenCV BGR 顺序，RGB 臂显式转换为 RGB 顺序并使用对应 RGB mean/std。

只有这样，结果差异才主要对应下游颜色方案，而不是随机性、初始化权重或训练预算差异。

## 3. 项目阶段全貌

### 3.1 已完成的历史基础

1. Stage-01：冻结 development split，确认 train-dev、val-dev 和 official test 互不混入。
2. Stage-02：完成 validation、checkpoint、best/latest 和可靠 resume 改造。
3. Stage-03：完成协议、preflight、seed 编排与运行证据链。
4. Stage-04：完成 4090 probe、3-epoch qualification、checkpoint 连续性和恢复等价性，历史 Gate D 已通过。
5. Stage-05 seed 1：完成 500/500 epoch 长程训练，退出码为 0，历史在线最佳 mIoU 为 `52.84`，对应 epoch 460。

### 3.2 当前所在位置

当前位于 Stage-05 的后半段：

1. 显式输入契约和 post-evaluator 修复：已完成；
2. CPU 契约与关键测试：已完成；
3. 两个 checkpoint、split、归档哈希复核：已完成；
4. 五项 val-dev 后评估：已完成；
5. 固定 checkpoint 三臂颜色诊断：已完成；
6. validation geometry 最终冻结：尚未正式裁决；
7. pretrained 上游资产与 RGB 通道语义 provenance：已闭合；
8. 独立 paired calibration：尚未冻结预算，未启动训练；
9. Protocol Gate：因此仍未通过。

### 3.3 Protocol Gate 之后才能做什么

- Stage-06 是可选 B2 支线：先做 A2 人工深度 corruption 证据；只有 `A2-pass + 用户批准` 才能实现 B2。
- Stage-07 是通用模块 paired screening：每个候选都必须和同协议 B0 成对训练。
- Gate E 冻结最终架构，可以合法输出 `B0-only`，也可以输出 `B0 + 最终模块组合`。
- Stage-08 才执行 formal B0/最终 variant 的正式 paired 三 seed。
- Gate F 通过后，才一次性读取 official test。

## 4. 本轮五项后评估已经得到什么

五项都满足以下共同边界：

- val-dev 318 个样本；
- split SHA-256 为 `1d0719d8f64f016d48995c25ab66d4004d76b7155d9efeef7cbb7454c0dd0e83`；
- 所有指标在原始 `932×1082` Label grid 计算；
- `official_test_included=false`；
- input contract 为历史 legacy BGR；
- checkpoint SHA 与预登记值一致。

具体结果：

- best × original-full：mIoU `52.98`，mAcc `65.67`，mF1 `67.96`；
- best × resize-480x640：mIoU `56.31`，mAcc `68.76`，mF1 `70.62`；
- best × sliding-480x640：mIoU `51.89`，mAcc `65.45`，mF1 `66.28`；
- epoch-500 × resize-480x640：mIoU `56.73`，mAcc `68.88`，mF1 `70.99`；
- epoch-500 × sliding-480x640：mIoU `52.08`，mAcc `65.60`，mF1 `66.21`。

### 4.1 应该如何理解这些数字

第一，`original-full` 的 `52.98` 与历史在线最佳 `52.84` 接近，差值为 0.14 个百分点。这支持“修复后的独立 evaluator 基本复现历史整图验证”的判断，但不改写历史记录。

第二，resize 数字最高，但它把原图的长宽比例从约 1.161 改成 1.333，会发生形状拉伸。因此不能只看最高 mIoU 就把 resize 设为正式验证方式。

第三，sliding 保留原图空间支持，也避免整图显存过大，但不同 crop 上下文会改变预测；本轮结果低于 original-full。

第四，本地 RTX 5060 Laptop GPU 的显存约 8 GB，`original-full` 已经成功跑完，没有触发环境限制。

### 4.2 当前对 geometry 的建议

建议新窗口优先讨论并冻结 `original-full` 作为后续默认 validation geometry，理由是：

1. 它最接近 seed 1 历史在线验证事实；
2. 它不扭曲长宽比；
3. 它保持完整图像上下文；
4. 本地 8 GB GPU 已证明可以运行；
5. 独立结果与历史在线结果接近。

`resize-480x640` 保留为诊断方式，不建议因为 mIoU 最高就成为正式口径；`sliding-480x640` 可作为整图显存不足环境的备选，但不是当前首选。

## 5. 三臂颜色诊断已经得到什么

三臂使用同一个 best checkpoint、同一 val-dev、同一 original-full geometry：

- legacy BGR + 原数组位置 mean/std：mIoU `52.98`；
- RGB + RGB mean/std：mIoU `33.85`；
- BGR + 反向 mean/std：mIoU `49.53`。

### 5.1 大白话解释

现有 checkpoint 是在 legacy BGR 下游契约中训练出来的。推理时突然换成 RGB，相当于把模型熟悉的颜色输入重新排列，mIoU 大幅下降 19.13 个百分点；只反向 mean/std 也下降 3.45 个百分点。

这证明“下游颜色契约不是无关小细节”，必须被显式记录和冻结。它同时提醒我们：上游 pretrained 的 RGB 语义不能直接当作历史 MUSeg checkpoint 的输入语义。

### 5.2 这个诊断不能证明什么

它不能证明未来从头训练时 BGR 一定优于 RGB，也不能证明 RGB 与官方 pretrained 的上游语义不兼容。

原因是：当前模型已经适应了 BGR。拿同一个 BGR 模型直接喂 RGB，只能测试它对下游输入变化是否敏感，不能公平比较“BGR 训练出来的模型”和“RGB 训练出来的模型”。公平选择仍需要以同一个 RGB 语义的 pretrained 权重初始化、再分别进行 BGR/RGB 下游微调的 paired calibration。

## 6. pretrained provenance 当前到哪里

已核验的 pretrained 身份：

- 文件名：`DFormerv2_Small_pretrained.pth`；
- 大小：110,203,103 bytes；
- SHA-256：`19116988fc86dc9f3e879282237941e11b9b1b5c480edb51e92807311dbc11a6`。

上游身份现已闭合：Hugging Face 官方 `bbynku/DFormerv2` 仓库中 `DFormerv2/pretrained/DFormerv2_Small_pretrained.pth` 的文件大小同为 110,203,103 bytes，LFS SHA-256 同为 `19116988fc86dc9f3e879282237941e11b9b1b5c480edb51e92807311dbc11a6`，与云端运行使用的权重逐项一致。官方 DFormer README 将该目录列为 DFormerv2 的 ImageNet-1K pretrained；官方 `VCIP-RGBD/RGBD-Pretrain` 数据代码默认通过 PIL `RGB` 读取彩色图，并按 RGB 顺序使用 ImageNet mean/std。因此该 pretrained 的上游彩色通道语义判定为 RGB。

这个来源结论不能直接替代 paired calibration：它只说明初始化权重的上游彩色通道语义是 RGB，不说明 MUSeg 下游在相同预算下最终采用 RGB 一定优于历史 BGR。历史 seed 1 已在 BGR 输入契约上完成微调并形成 reference；未来项目颜色谱系仍需以同一 pretrained 权重、同一 seed 和同一训练条件分别进行 BGR/RGB 下游校准后决定。

## 7. paired calibration 为什么现在不能直接开跑

当前训练协议工具支持 protocol v3 的单一 `input_contract`，launcher 也能把 channel order 和 normalization 传给训练入口，但仓库没有现成的 `color-geometry-screening-B0` protocol。

还缺少以下预注册决定：

1. 每臂训练多少 epoch；
2. 使用哪个 seed；
3. 多大 mIoU 差异算“明确超过噪声”；
4. 差异接近时是否补第二 seed；
5. 训练在哪个 GPU 环境执行；
6. 两臂运行顺序；
7. 失败、恢复、证据同步和关机规则；
8. 是否将 `original-full` 先冻结为统一 evaluator geometry。

旧 `tools/mve/run_museg_20epoch_screen.sh` 已明确作废，不能为了省事直接复用。

## 8. paired calibration 的可行实现

### 8.1 最小改动方案

推荐不扩展整个 protocol schema，而是使用：

- 一个 pair registry，记录 calibration 身份、共同预算、seed、运行顺序、容差和两臂 protocol SHA；
- 两个 sibling protocol v3：一个 legacy BGR 下游契约，一个 RGB 下游契约；
- 两个 protocol 共享同一个官方 RGB-D pretrained 权重身份；除 `protocol_id` 和 `input_contract` 外，其余关键字段完全相同；
- 一个最小 paired orchestrator，先做双臂 preflight，再按预登记顺序串行运行，并在任一臂失败时停止。

这样能复用已有 `preflight_train.py`、`run_museg_seed.py`、checkpoint 和运行证据机制，又不会把“颜色臂”硬塞进现有只按 seed 编排的 `run_museg_3seed.py`。

### 8.2 为什么不建议同时并行两臂

并行会占用两个 GPU 或让同一 GPU 发生资源竞争，还会增加环境差异。串行虽然总用时相近，但更容易保证同一设备、同一镜像和稳定显存，也便于失败时立即停止计费。

## 9. 训练预算与资源利用率分析

### 9.1 500 epoch 两臂

历史 seed 1 的 500 epoch 用时约 43,643 秒，即约 12.1 小时。若两臂都跑 500 epoch，粗略需要约 24.2 GPU 小时，另加预检、验证、同步和失败恢复开销。

优点：

- 与历史长程开发预算最接近；
- 颜色排序受短训曲线偶然性的影响较小。

缺点：

- 在模块方向尚未开始筛选前就消耗约两个完整长程 run；
- 如果 RGB 很早就明显不合适，后续大量 epoch 可能浪费；
- 若结果接近还需补第二 seed，成本可能再次翻倍。

### 9.2 50 epoch 两臂

按训练时间近似线性估计，两臂约需 2.4 GPU 小时，实际加上验证和初始化会更高一些。

优点：

- 资源成本约为 500 epoch 两臂的十分之一；
- 足以观察严重不收敛、明显差距和曲线方向；
- 更适合作为颜色协议筛选。

缺点：

- 短训领先不保证 500 epoch 仍领先；
- 必须把它登记为独立 screening protocol，不能与历史 `52.84` 直接比较；
- 如果差异接近噪声，必须两臂一起补第二 seed，不能只补一边。

### 9.3 推荐的资源策略

建议新窗口优先评估“两阶段、成对、逐级升级”的方案：

1. 先跑 `50 epoch × 1 paired seed × 2 arms`；
2. 两臂都使用同一 4090、batch 10、train-dev/val-dev、优化器、学习率、增强和 original-full evaluator；
3. 预先冻结噪声容差，例如 mIoU 绝对差 1.0 个百分点，同时检查最后若干验证点和 per-class 方向；
4. 若差异明显且曲线稳定，冻结颜色谱系或只对领先方案与 legacy BGR 进入更长预算确认；
5. 若差异接近容差，两臂同时补第二 paired seed；
6. 任何升级都使用新 protocol，不看结果后临时改规则。

这里的 1.0 仅是供新窗口讨论的建议值，当前尚未由用户冻结，不能作为已经生效的门槛。

## 10. 云资源与安全策略

当前 CompShare 实例 `cpod-1tyvjsiu6ahe` 已停止，没有新的云实例运行或计费。

paired calibration 若使用 4090，建议固定以下生命周期：

1. 代码、protocol 和 pair registry 先在本地完成关键测试并形成干净提交；
2. 创建实例前查询实时价格和库存，不沿用历史价格猜测成本；
3. 控制面预设最晚停止时间；
4. 先运行双臂共同 preflight 和最小 GPU qualification；
5. 两臂串行运行，任一身份/hash/数据/显存/非有限值问题立即停止；
6. 成功、失败或人工中止都先同步证据；
7. 本地复核必要 checkpoint、日志和 manifest 的 SHA；
8. 无论验收 pass/fail 都停止实例，研究结论不得控制是否继续计费。

本地 RTX 5060 约 8 GB，虽然可以完成 batch 1 后评估，但不能在保持历史 batch 10 和同等训练环境的前提下承担 paired 长训。因此不建议用本地缩 batch 训练后再与 4090 历史结果混比。

## 11. 建议新窗口需要拍板的事项

按优先级建议依次决定：

1. 是否冻结 `original-full` 为 development 默认 validation geometry；
2. paired calibration 使用 50、100 还是 500 epoch；
3. 第一 paired seed 使用哪个预注册 seed；
4. 噪声容差和 per-class/曲线稳定条件；
5. 差异接近时是否两臂补第二 seed；
6. 是否采用“pair registry + 两个 sibling protocol v3 + 串行 orchestrator”的最小实现；
7. 是否授权创建 4090 云实例，以及最大预算和最晚停止时间；
8. pretrained provenance 已通过官方 Hugging Face 资产与官方预训练代码闭合为 RGB；paired calibration 不再核查 pretrained 的上游语义，而只负责决定 MUSeg 下游未来采用 RGB 还是保留 legacy BGR。

## 12. 当前准确恢复点

当前应停在这里：

- 五项后评估：完成；
- 三臂颜色诊断：完成；
- paired calibration 训练：未启动；
- calibration protocol：未创建；
- 新云实例：未创建；
- official test：继续 `sealed_unread`；
- Protocol Gate：未通过；
- development seeds 2/3、A2/B2、模块筛选、正式三 seed：均未启动。

新窗口在完成第 11 节决策后，从“冻结 pair registry、两臂 protocol、共同 pretrained 身份、预算、seed、容差和资源上限”恢复，而不是直接运行旧 20-epoch 脚本、把固定 BGR checkpoint 改用 RGB 推理，或复用现有 `52.84` 作为另一颜色臂的对照。

## 13. 权威证据与索引

### 13.1 实时状态与开放决策

- `doc/main/MUSeg-current-status.md`：当前事实、阻塞项和唯一恢复入口。
- `doc/main/MUSeg-open-decisions.md`：geometry、颜色、三 seed 时机等研究口径。

### 13.2 当前计划与历史计划

- `doc/plans/MUSeg-DFormerv2快速Baseline/00-总方向规划.md`：当前 DFormerv2-S MUSeg baseline 的总方向；细节后续另行规划。
- `doc/plans/archive/README.md`：归档说明。
- `doc/plans/archive/2026-08-MUSeg阶段二长程Baseline与MVE/00-总索引与执行门禁.md`：已封存的阶段结构、门禁定义和历史执行依据。
- `doc/plans/archive/2026-08-MUSeg阶段二长程Baseline与MVE/05-协议校准与Development-B0收口.md`：已封存的 Protocol Gate 计划。
- `doc/plans/archive/2026-08-MUSeg阶段二长程Baseline与MVE/06-MVE问题验证与B2条件分支.md`：已封存的 A2/B2 条件支线。
- `doc/plans/archive/2026-08-MUSeg阶段二长程Baseline与MVE/07-模块配对筛选与GateE冻结.md`：已封存的模块筛选和 Gate E 计划。
- `doc/plans/archive/2026-08-MUSeg阶段二长程Baseline与MVE/08-正式三种子与Official-Test解封.md`：已封存的正式三 seed、Gate F 和 official test 计划。

### 13.3 本轮报告与机器证据

- `doc/reports/2026-08-28-museg-stage05-posteval-protocol-gate.md`：五项结果、身份核验和中间风险说明。
- `cloud/DFormer-stage05-evidence/posteval/`：五份后评估 v2 JSON 和两份新增颜色诊断 JSON；该目录中的旧 `best-original-full.json` 是 v1 失败现场，不计入有效结果。
- `cloud/DFormer-stage05-evidence/checkpoints/`：两个独立 checkpoint，大文件不进入 Git。
- `cloud/DFormer-stage05-evidence/museg-val-dev-local-posteval/`：318 样本本地 val-dev bundle，不含 official test。
- `data/splits/MUSeg/dev-v1/manifest.json` 与 `audit-report.json`：冻结 split 权威。

### 13.4 代码入口

- `tools/evaluate_museg_checkpoint.py`：post-evaluation v2 evaluator。
- `tools/museg_protocol.py`：training protocol v2/v3 解析与身份验证。
- `tools/preflight_train.py`：运行前协议、代码、数据和输入契约检查。
- `tools/run_museg_seed.py`：单 seed 运行入口。
- `tools/train_museg_4090.sh` 与 `tools/run_museg_3seed.py`：现有 seed 编排入口；当前尚不能直接表达颜色 pair。

## 14. 不得跨越的边界

- 不修改原始 `acceptance.json`、`failed.json`、`training_result.json` 或 `liu-test-exp/**`；
- 不把单 seed、短训或固定 checkpoint 诊断写成正式 baseline；
- 不把不同颜色谱系、预算或 evaluator 的结果混进同一 mean±std；
- 不用 official test 做任何开发选择；
- 不把数据、checkpoint、归档、凭据和大运行产物提交到 Git；
- 未冻结 calibration protocol 和资源上限前，不创建实例或启动训练；
- 未经用户确认，不推送远端。
