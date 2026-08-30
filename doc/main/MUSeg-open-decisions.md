# MUSeg 实验口径与处置状态

> 状态时间：2026-08-30 06:06 UTC
> 本文件保留问题缘由，同时明确区分“仍待决定”“本轮已处置”和“仅保留历史解释”。
> 已完成的 seed 1 不回写 protocol 或原始证据；影响后续运行的变更必须使用新 protocol 身份并重新 qualification。

## 1. 新 DFormerv2-MUSeg baseline 方向

**大白话结论：** 新计划使用 DFormerv2-S 和其公开训练/测试方法建立内部 B0，作为后续模块消融的共同起点；目标是结果量级合理、链路可信和比较口径一致，不是三 seed 完整复现论文。

**当前状态：方向、RGB 和 single-seed B0 角色已确认；实现与资源细节尚未冻结。**

- 训练方向：采用官方公开的随机尺度训练增强，尺度候选为 `0.5、0.75、1.0、1.25、1.5、1.75`，之后裁剪到 `480×640`，并保持 RGB/Depth/Label 同步变换。
- 测试方向：采用官方论文公开的 multi-scale flip 推理，尺度为 `0.5、0.75、1.0、1.25、1.5`，暂不把滑动窗口静默混入主基线。
- 输出方向：每个尺度的预测恢复到 MUSeg 原始 Label 网格后融合和计分；`480×640` 是训练或明确命名的模型输入尺寸，不自动等于最终 metric geometry。
- 这是公开 DFormerv2 方法在 MUSeg 上的适配，用于建立后续模块的内部对照；不声称复现 MUSeg 作者未公开的测试代码，也不以论文数值完全相等或三 seed 统计作为当前 B0 门槛。具体 evaluator、预算、checkpoint 规则和运行位置由当前执行方案冻结。
- 旧 `Stage-01` 至 `Stage-05` 计划和其未完成的 Protocol Gate 已封存；历史 seed 1 的单尺度结果只作为 reference，不与新 baseline 混合统计。

## 2. Validation 空间尺寸

**大白话问题：** 文档曾把“输入 640×480”写成统一模型输入，但 seed 1 实际只把训练样本随机裁剪到高 480、宽 640；`sliding=false` 时，validation 使用转换数据的原始高 932、宽 1082 整图前向。不同几何会改变 val mIoU，不能把结果直接混为同一口径。

**当前状态：历史 seed 1 的五项后评估已完成；新 baseline 的测试几何改为优先实现 DFormerv2 论文公开的 multi-scale flip 方向，具体 evaluator 和资源细节待新计划细化。原图计分仍作为输出对齐原则。**

- seed 1 的训练与在线 validation 事实保持为：训练裁剪 480×640，validation 原分辨率整图，`sliding=false`。
- post-evaluator 已改为所有 geometry 保留原始 Label：resize 只改变模型输入，logits 恢复到原图计分；sliding 保持全图覆盖。报告显式记录 input/metric geometry、插值、stride、padding 和输出尺寸。
- production `ValPre`/original-full、resize 原图计分、sliding 覆盖、strict checkpoint load 和 official-test 拒绝的聚焦 CPU 测试已通过；完整验收仍待完成。
- 五项后评估已完成：best 的 original-full/resize/sliding mIoU 为 `52.98`/`56.31`/`51.89`，epoch-500 的 resize/sliding 为 `56.73`/`52.08`；五项都在原始 Label grid 计分，均为 318 样本且 official test 未参与。结果只能用于 geometry 诊断，不改写 seed 1 原始曲线或 best 身份。
- 历史五项后评估的几何排序只用于诊断，不作为新 baseline 的冻结依据。新计划优先实现 DFormerv2 论文的 multi-scale flip；单尺度 original-full、固定 resize 和 sliding 保留为命名清晰的对照或资源备选。

决定前，“480×640”只表示训练裁剪或明确命名的推理输入，不概括为统一 validation 尺寸；后评估只冻结未来 protocol，不改写 seed 1 原始曲线或 best 身份。

## 3. MUSeg 颜色通道顺序

**大白话问题：** 历史 MUSeg loader 使用 OpenCV BGR，但官方预训练模型看到的是 RGB。项目刚起步时，是先做两种颜色的配对训练，还是先选择与预训练一致的输入？

**当前状态：本轮已处置。用户于 2026-08-30 确认 quick B0 直接使用 RGB，取消 RGB/BGR 双臂；这是输入一致性选择，不是颜色性能胜负结论。**

- seed 1 的历史事实保持为 OpenCV BGR 数组，并按位置应用 `[0.485,0.456,0.406]` / `[0.229,0.224,0.225]`；不回写其 protocol 或结果。
- 当前权重已闭合为官方上游资产：Hugging Face `bbynku/DFormerv2` 中 `DFormerv2/pretrained/DFormerv2_Small_pretrained.pth` 的大小为 110,203,103 bytes，LFS SHA-256 为 `19116988fc86dc9f3e879282237941e11b9b1b5c480edb51e92807311dbc11a6`，与本项目权重完全一致。官方 README 将其列为 ImageNet-1K RGB-D pretrained；官方 `VCIP-RGBD/RGBD-Pretrain` 数据代码默认以 PIL `RGB` 读取彩色图并使用 RGB 顺序 ImageNet mean/std，因此 pretrained 上游通道语义判定为 RGB。
- 新 quick B0 明确执行 OpenCV BGR→RGB，再使用 RGB 顺序 ImageNet mean/std。大白话说，这让下游输入保持官方预训练模型已经学习过的通道含义，是当前变量最少、依据最直接的起点。
- 固定历史 best checkpoint 的三臂 original-full 诊断结果仍保留：legacy BGR、RGB+RGB mean/std、BGR+反向 mean/std 的 mIoU 分别为 `52.98`、`33.85`、`49.53`。它只证明旧 checkpoint 对输入契约强敏感，不能用于判断重新训练后的 RGB/BGR 胜负。
- 本轮不做 `color-geometry-screening-B0`、短程颜色训练或第二 seed，也不把“选择 RGB”表述为“RGB 在 MUSeg 上统计显著优于 BGR”。如果未来研究问题明确变成颜色谱系比较，才需要另立 paired calibration protocol 并从相同 pretrained 成对重训。
- 历史 BGR seed 1 继续保留 `development-reference-B0` 身份；新 RGB quick B0 使用独立 protocol identity，两者不混入同一 mean±std 或当作配对实验。

## 4. A2 自然无效深度分层是否为 B2 硬门槛

**大白话问题：** 人工 corruption 可以证明模型对深度破坏敏感，但当前自然缺失深度证据可能不足以证明现实世界中存在同样机制。若把两者都设为硬门槛，会让“能否做 B2”和“能否声称现实机制”混在一起。

**当前状态：已处置。人工 corruption 可作为进入 B2 的工程门槛，自然证据限制结论强度。**

- A2 人工 corruption 达标后可以进入 B2 开发，不要求自然无效深度分层先成为硬前提。
- 若自然缺失/无效深度分层证据不足，只能声明“在人工 corruption 条件下观察到敏感性或改进”，不得扩展为真实缺失机制、现实鲁棒性或部署收益。
- 正式 A2/B2 开发筛查只使用 `val-dev`；official test 等最终模型和协议冻结后再通过独立门禁一次性解封。

## 5. Qualification 与长程训练的 step 计数

**大白话问题：** Stage-04 计划为 3×128=384 次 loop 尝试，报告记录 376 次成功 optimizer update；Stage-05 理论网格为 64,000 次，最终记录 63,973 次有效更新。AMP 可能跳过少量更新，但旧遥测把“尝试”和“成功”混写，导致验收误判。

**当前状态：历史差异不再阻塞，未来遥测已修正，旧 run 不追溯改写。**

- Stage-04 的 8 次差异缺少完整 trace，无法事后证明每次具体原因；该缺口保留为历史限制，不推翻 Gate D 的连续/恢复等价证据。
- Stage-05 的 27 次差异按少量 AMP 跳过更新处理，不作为训练失败条件；500 个 epoch、50 个 validation 点、checkpoint 身份和最终结果已由 v2 裁决独立核验。
- 未来非 probe 运行分别记录实际 loop attempts、completed optimizer updates 和 skipped optimizer steps，并写入遥测 schema 版本。
- 学习率与调度语义必须在新运行中由结构化计数验证，不用修改原始 `acceptance.json` 或 `training_result.json` 来补齐旧证据。

## 6. `run_kind=qualification` 的历史字段名

**大白话问题：** seed 1 明明是 development 长程训练，命令却记录 `run_kind=qualification`。这是旧代码把“所有非 probe 运行”都叫 qualification，不代表研究 phase 真的是 qualification。

**当前状态：已处置。未来使用 `standard`，历史身份保持不变。**

- 启动器和训练入口已允许未来 `run_kind=standard`，并继续兼容旧的 `qualification`。
- seed 1 的原始命令、manifest 和结果仍保留 `run_kind=qualification`，不得改写；其真实研究阶段继续由 `experiment_phase=development` 和 protocol role 决定。
- 新的 development 长程运行应使用 `standard`；`qualification` 只为历史兼容或真正 qualification 保留。

## 7. 云端终态与关机

**大白话问题：** 本次 SwanLab 已显示完成，但自动流程没有及时关机，人工等待约 23 分钟后仍需手动处理，验收失败路径还曾明确记录 `automatic_shutdown=false`。如果让验收结果决定是否关机，失败时会持续计费。

**当前状态：策略已确定；用户于 2026-08-30 要求在正式 RTX 4090 任务前增加无卡自动关机实测，实际云资源操作尚未授权。**

- 生产生命周期由本地控制器处理共同终态：workload 成功、失败或人工中止后，都先取回必要证据并核验哈希，再调用 CompShare 控制面 stop；验收 pass/fail 只决定研究结论，不决定是否停止计费。
- 实例内 `shutdown -h` 不能单独证明平台进入 `Stopped`。自动关机验收必须使用控制面 stop，并等待和复查实例状态为 `Stopped`。
- 正式 RTX 4090 前，用 `run_kind=lifecycle-test`、`simulation=true` 的无卡任务模拟成功 workload、测试报告、证据 manifest 和 SHA-256；测试产物不得进入 B0 指标或被训练裁决器接受。
- 无卡实测通过条件为报告与哈希匹配、自动 stop 成功、实例在 timeout 内进入 `Stopped`，且不需要人工补发普通停止命令；失败则阻塞正式 B0。
- 每次无卡测试和正式训练启动前都使用 `instance schedule set --at` 设置控制面最晚停止兜底，并用 `instance schedule show` 复核；脚本或本地控制器自动 stop 是第一道保障，schedule 是断联兜底。
- 实际创建、启动、停止实例或修改 schedule 前仍需用户对资源、最长时间和预计费用单独授权。

## 8. Single-seed B0 与后续模块消融

**大白话问题：** 当前需要的是模块设计的可信共同起点，而不是先花三倍成本形成论文级随机方差统计。怎样既节省资源，又避免后续比较失去公平性？

**当前状态：已处置。用户于 2026-08-30 确认本轮只训练一个 single-seed RGB B0；它作为后续模块消融的固定内部基线，不以三 seed 完整论文复现为当前目标。**

- B0 的验收重点是训练与评估链可信、指标量级合理、没有明显类别或数值异常，并完整绑定 pretrained、split、seed、config、checkpoint 和 evaluator 身份；不要求与论文数字完全相等。
- 后续模块可以复用这一个 B0 结果作为对照，但模块版本必须从同一 pretrained 独立训练，并保持相同 `train-dev`/`val-dev`、seed、数据顺序、epoch、优化器、增强、checkpoint 规则和主 evaluator。不能从 B0 最终 checkpoint 接着训练模块后再称为公平消融。
- 若后续改变训练预算、优化器、增强、数据或 evaluator，现有 B0 不再是严格配对对照；需要限定结论，或在新协议下重训匹配的 B0。
- 单 seed 足够用于模块探索、淘汰和初步消融，但不能估计随机方差。若模块增益很小、接近训练波动或要支撑重要结论，应对 B0 和该模块增加成对重复或额外 seed；当前不预先要求三 seed，也不因此阻塞模块设计。
- 主 evaluator 可以在本地或云端运行。机器位置不改变实验身份，但必须绑定 checkpoint/split 哈希、冻结代码与配置、输入契约、前向精度、环境和 `official_test_included=false`。本机已核验为 RTX 5060 Laptop 8 GB，历史单尺度 318 样本约 108 秒；本地最多 4 个主评估候选按 2–4 小时规划、硬上限 8 小时，最终运行位置取决于尺度 1.5 显存检查。
- official test 在 B0 和模块开发期间继续 `sealed_unread`；是否以及何时解封由未来独立门禁决定，当前 single-seed 方向不构成解封授权。
