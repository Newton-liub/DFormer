# MUSeg × DFormerv2：从数据适配、可信基线到几何可信 RGB-D 双路径 MVE 组会汇报

- 汇报周期：2026-08-17 至 2026-09-10
- 报告对象：课题组组会
- 报告主线：DFormerv2 + MUSeg 适配 → 可审计 Quick-B0 → 几何可信 RGB-D 双路径 MVE → 当前 DVG-B1 Oracle 门控设计
- 证据边界：项目当前状态、日期化正式报告、冻结 protocol、训练与评价结构化产物、MVE 运行记录及 2026-09-10 文献摘要筛选
- 当前状态：稳定 RGB Quick-B0 已建立；DVC-A1 完整开发评价裁决为 `not-supported`；DVG-B1 已完成项目内实现定位和文献摘要筛选，仍暂停于 A/B/C 三组参考冻结门禁

## 一、结论先行

这段工作的核心成果不是单独得到一个分割分数，而是把 MUSeg 上的 DFormerv2 研究从“数据能否接入、数字是否可信”推进到“可以围绕一个明确机制做可证伪最小实验”。

第一，数据与训练链已经闭合。项目把 MUSeg 的 3171 组 RGB、16-bit Depth 和 Label 确定性转换为 DFormer 可用格式，对约 31.98 亿像素逐像素核验深度量化公式，并明确了标签、无效深度、颜色通道和数据划分语义。

第二，建立了可复核的 single-seed RGB Quick-B0。最终 checkpoint 为 epoch 420，在冻结的 318 张 `val-dev` 上使用五尺度 × 原图/翻转共 10 个 view、原始 Label 网格计分，得到 mIoU `58.79%`、mAcc `69.91%`、mF1 `72.73%`。训练期 selector 第一名 epoch 480 在主评价中只排第三，证明“低成本筛选候选、再用冻结主 evaluator 重排”是必要的。

第三，当前 MVE 已经完成 A 路径的问题验证。`DVC-A1` 检查“人工破坏深度边界是否会特别伤害语义边界”：v1 因操作定义覆盖不足合法停止，v2 完成全量推理后因一个位置组的 Boundary IoU 定义问题停止，v3 修正 background/ignore 标签域后完成 218 张图 × 5 个条件的正式开发评价。主分析 `dose_effect` 为 `+0.0731` 个 Boundary-IoU 百分点，`specificity_effect` 为 `+0.0324` 个百分点，联合裁决为 `not-supported`。这表示没有观察到预注册的边界特异敏感性，因此不能继续把该问题当作首要瓶颈。

第四，B 路径仍值得作为独立方案筛选。`DVG-B1` 问的是：即使 A 没证明“深度边界失效是首要问题”，如果 Oracle 直接告诉模型哪些深度像素被破坏，只降低 Geometry Self-Attention（GSA，几何自注意力）中的 depth geometry contribution，是否仍能得到稳定净收益。项目已经核清唯一 gate 位置、四级 GSA 结构、参数传递链和可复用 evaluator；当前从 73 个唯一 WOS 题录中筛出了全文优先候选，但摘要仍不足以冻结像素 mask 到 token reliability、token reliability 到 pairwise gate，以及正式裁决门槛。

大白话说：共同基线已经可信；第一个“问题定位”实验给出了明确负结果；现在不是继续堆复杂模块，而是先判断一个最小、理想化的门控动作本身有没有价值。

## 二、关键概念与判断边界

- **Quick-B0：** 后续模块比较的固定开发基线。它是一个随机种子的内部对照，不是三 seed 论文复现，也不是 official test 结果。
- **MVE（Minimum Viable Experiment，最小可行实验）：** 用最低必要成本回答一个可证伪问题。它优先判断方向是否值得继续，而不是一次完成全部系统。
- **DVC-A1：** A 路径的问题验证，检查人工深度边界置零是否产生边界特异损害。
- **DVG-B1：** B 路径的方案验证，检查已知坏区条件下抑制 GSA 深度几何项是否有上限收益。
- **Oracle：** 直接使用真实 corruption mask 提供“哪里坏了”的理想信息，只测方案上限，不代表真实系统已经能自动识别坏深度。
- **`protocol-blocked`：** 实验定义或证据链不满足预注册要求，必须停止；它不是模型效果的负结论。
- **`not-supported`：** protocol 和运行有效，但数值没有达到预注册支持条件。它只否定当前窄问题，不等于自然故障机制不存在。
- **official test `sealed_unread`：** 官方测试集在开发期封存未读，没有参与模型选择、阈值冻结、方向筛选或本报告中的结果。

## 三、工作量总览

### 3.1 数据与语义复现

- 完成 3171 组 MUSeg 数据的确定性重建；RGB、Depth、Depth16、Label 主文件名集合一致。
- 核验官方 train/test 为 1595/1576，样本交集和位置组交集均为 0。
- 对 3,197,712,504 个像素验证固定全局深度映射 `round(Depth16 × 255 / 13932)`，匹配率 100%。
- 明确原始 Depth `0` 为无效值，比例 `30.7351%`；明确原始 Label `0` 为 background，`1–15` 为 15 个前景类。
- 识别 11 张全背景图，并将其引出的空有效像素 loss 风险与无效深度问题分开处理。

### 3.2 训练与基线评价

- 累计完成两次 500 epoch 单 seed 长程训练：历史 legacy BGR 训练用于暴露流程问题和沉淀经验，独立 RGB Quick-B0 用于形成当前稳定基线。
- 两次训练墙钟时间约为 12 小时 07 分和 13 小时 38 分，合计约 25 小时 45 分；两者协议不同，指标不能做单变量归因。
- RGB Quick-B0 结构化记录 `64,000 / 63,971 / 29` 个 attempted/completed/skipped step。
- 对 4 个候选 checkpoint 完成五尺度翻转主评价；按 evaluator 合同折算为 `4 × 318 × 10 = 12,720` 个 view 级模型前向。
- 本地主评价用时 `94.044` 分钟，4 个候选均完成且未 OOM。

### 3.3 当前 MVE 实现与运行

- DVC-A1 实现包括独立 protocol、Depth16 corruption、确定性 mask、同面积非边界对照、Boundary IoU、位置组聚合、10,000 次 bootstrap、五尺度翻转运行入口和结构化失败/完成记录。
- v1 扫描完整 318 张 `val-dev`、196 个位置组；发现 58/196 个组无法构造非空 q75，比例 `29.5918%`，在全量模型评价前停止。
- v2 与 v3 均完成 218 张图 × 5 个条件的完整模型推理；每轮按合同折算为 `218 × 5 × 10 = 10,900` 个 view 级模型前向，两轮合计 21,800 个。只有 v3 闭合了有效配对组并形成正式科学裁决；v2 因统计定义问题保持 `protocol-blocked`。
- v2 用时 `5445.565` 秒，v3 用时 `5507.055` 秒；v3 最终完成 138/138 个有效配对组和 10,000 次位置组 bootstrap。
- 从基线 4 候选主评价到 v2/v3 MVE，按 evaluator 合同合计至少完成 34,520 个 view 级模型前向；不含历史后评价、preflight 和训练前向。

### 3.4 文献与实现定位

- 收缩首轮方案，删除缺少数据或直接依据的三维评价、冻结 logits 门控、严格 RGB-only 回退、亮度引导补全和高置信阈值。
- 核对当前 `DFormerv2.py` 与作者保留副本，完整文件 SHA-256 一致，确认 bilinear Depth resize 是作者代码语义，而不是本项目意外修改。
- 核清四级 GSA：前三个 stage 使用 H/W 分解结构，第四个 stage 使用 Full GSA；唯一合规 gate 位于 spatial contribution 与 depth geometry contribution 相加之前。
- 筛选 A-1 44 条和 B-1 30 条 WOS 记录；两文件仅共享 1 条，并集为 73 个唯一题录；形成 A/B 的 P0 全文与官方代码优先级。

## 四、阶段一：DFormerv2 与 MUSeg 的数据适配

### 4.1 起点问题

MUSeg 发布的是 16-bit Depth，而原 DFormer loader 按 8-bit 灰度图读取；原始深度值 `0` 表示缺失，DFormerv2 却会把数值直接用于 patch 深度差。与此同时，原始标签 `0` 是 background，但训练映射后会成为 ignore `255`。如果这些语义不先分开，后续任何“深度有效性”或“边界鲁棒性”实验都可能在错误输入上得出结论。

### 4.2 已完成处理

- 新增确定性转换脚本 `tools/prepare_museg.py`。
- 原样保留 `Depth16/`，按全数据统一最大值 13932 生成 8-bit `Depth/`。
- 写入 `dataset_meta.json`，记录转换公式、invalid policy、样本数和官方 split 哈希。
- 转换先写临时目录，验证通过后原子替换，避免留下半成品。
- 依据官方 `Label_ID.pdf` 固定 `0=background`、`1–15=15` 个前景类别；训练张量使用 `0→255`、`1–15→0–14`。

### 4.3 结果与意义

数据转换已经从一次性人工处理变成可重复、可审计的工程入口。31.98 亿像素的全量核验排除了截低 8 位和逐图 min-max 等错误实现；同时保留 Depth16，为后续在原始深度域注入 corruption 提供了权威入口。

这一步不能证明模型性能，但决定了后续实验是否在同一数据语义上比较。

## 五、阶段二：早期 MVE 和错误路线的收缩

早期路线的价值主要是暴露风险和淘汰不可靠解释，组会中不需要展开所有执行细节。

### 5.1 空有效像素 loss：问题成立并完成修复

11 张全背景图经过标签映射后可形成全 ignore crop 或 batch。旧 masked mean 对空集合求均值会产生非有限值。项目实现 `safe_masked_mean`：空集合返回与计算图连接的有限零值，非空集合保持原均值；该修复进入稳定基线。

该结果证明的是训练数值稳定性，不是分割性能提升。

### 5.2 16 张样本的 Depth block-mask pilot：只够筛方向

历史 epoch-10 checkpoint 上完成 16 张样本 × 3 个条件，共 48 个样本—条件前向。q=0.3 和 q=0.5 相对 q=0 的前景 mIoU 仅下降 `0.1107` 和 `0.3337` 个百分点，没有达到触发完整 validity gating 的预注册门槛。

它只说明“额外深度缺失可能有轻微影响”，不能替代当前 RGB B0 上的正式实验，也不能支持直接实现复杂 B2。

### 5.3 历史 BGR 和 evaluator geometry：数字不能脱离协议

历史固定 checkpoint 在 original-full、resize-480×640、sliding-480×640 下的 mIoU 分别为 `52.98%`、`56.31%`、`51.89%`，跨度 `4.42` 个百分点。固定历史 BGR checkpoint 直接切换到 RGB 输入时 mIoU 大幅变化，也只证明输入契约敏感。

因此后续路线不再根据“哪个临时数字最高”回选协议，而是重新建立 RGB Quick-B0，提前冻结输入、split、checkpoint 选择和主 evaluator。

## 六、阶段三：建立可复核 RGB Quick-B0

### 6.1 固定身份

- 模型：`DFormerv2-S`。
- 初始化：官方 `DFormerv2_Small_pretrained.pth`，SHA-256 与上游资产一致。
- 输入：OpenCV BGR 读取后显式转换为 RGB，并使用 RGB 顺序 ImageNet mean/std。
- 数据：`train-dev` 1277 张，`val-dev` 318 张；同一位置组不跨划分。
- 训练：500 epoch，AdamW，base learning rate `6e-5`，batch size 10，seed `772961337`，六个随机尺度 `0.5–1.75`，训练裁剪 `480×640`。
- 评价：五尺度 `0.5、0.75、1.0、1.25、1.5`，每尺度原图与水平翻转，共 10 个 view；FP32 pre-softmax logits 平均；恢复到原始 Label 网格计分。

### 6.2 四个候选结果

- epoch 420：mIoU `58.79%`，mAcc `69.91%`，mF1 `72.73%`。
- epoch 440：mIoU `58.73%`，mAcc `69.54%`，mF1 `72.67%`。
- epoch 480：mIoU `58.43%`，mAcc `69.34%`，mF1 `72.44%`。
- epoch 500：mIoU `57.68%`，mAcc `68.84%`，mF1 `71.81%`。

训练期 selector 将 epoch 480 排第一、epoch 420 排第二；主 evaluator 将 epoch 420 排第一、epoch 480 排第三。这是当前流程设计最直接的结果之一：最终模型必须由最终 evaluator 决定。

### 6.3 当前稳定基线

最终 checkpoint 为 epoch 420 的 `selector-epoch-420.pth`，SHA-256 为 `f246a3afc50334c81302b7bfebdadf7cf37d00326bf1c3aa54f6a151754e3a1c`。

逐类结果显示 door IoU `86.01%`、rail area `80.10%` 较强；container `24.67%` 最弱，support/mining/rescue equipment 分别为 `40.23%`、`41.92%`、`43.56%`。后续方法不能只看总体 mIoU。

## 七、阶段四：为什么转向几何可信 RGB-D 双路径 MVE

DFormerv2 不使用独立 Depth 编码器，而是把 Depth 差异作为几何先验注入 GSA。MUSeg 中约 30.74% 原始深度像素为无效 0，因此“深度观测是否可靠、模型是否应该无条件相信它”是一个直接连接数据语义与模型机制的问题。

首轮方案被收缩为两条独立问题链：

1. **A 路径——先验问题验证：** 深度边界失效是否会特别损害语义边界？
2. **B 路径——最小方案验证：** 如果已知哪些深度位置被破坏，只抑制这些位置的 GSA 深度几何贡献，是否有稳定净收益？

这种拆分避免了把“问题是否存在”和“某个动作是否有用”混成一个实验。A 不支持时，B 仍可独立测方案上限；B 有效也不能反向证明 A 成立。

## 八、DVC-A1：从两次合法阻塞到完整负结果

### 8.1 v1：覆盖门禁阻塞

v1 在原始 Depth16 上以全局相对深度跳变阈值 `0.05` 构造边界候选，生成 `clean`、`boundary-q25/q50/q75` 和 `nonboundary-q50` 五个条件。

完整 318 张 `val-dev` mask 扫描发现 58/196 个位置组无法构造非空 q75，占 `29.5918%`，高于预注册 `5%` 上限。实验在完整模型评价前停止，状态为 `protocol-blocked`。

这不是“模型不敏感”，而是“操作定义覆盖不足”。

### 8.2 v2：全量推理完成，但统计定义阻塞

v2 不降低 `0.05` 阈值，也不重划数据；预先纳入至少一张图可构造 q75 的全部 138 个位置组、218 张图，并保留组内全部样本。

五个条件的完整本地 GPU 推理全部完成，用时 `5445.565` 秒，每个条件都有 218 张图和 138 个组。但主分析只有 137/138 个有效 Boundary IoU 配对组；无效组 `06-01-01-0346` 在旧标签域下没有可定义的安全计分区域，因此再次以 `protocol-blocked` 收口。

这不是结果不好，而是统计对象的定义还没有闭合。

### 8.3 v3：修正标签域并完成正式裁决

v3 只修正 Boundary IoU 的标签域：训练和普通 mIoU 继续把 raw background `0` 映射为 ignore `255`；Boundary IoU 则把 background 保留为有效几何上下文，只把 true ignore 排除。阈值、allowlist、五个条件、checkpoint、evaluator、bootstrap 和裁决门槛均不改变。

218 张图 CPU 标签域审计通过；覆盖原阻塞组的两样本 GPU preflight 通过；随后完整评价以退出码 0 完成，用时 `5507.055` 秒，约 91 分 47 秒。

五个条件的描述性总体指标为：

- clean：mIoU `54.64`，mAcc `64.23`，mF1 `67.59`；
- boundary-q25：mIoU `54.66`，mAcc `64.27`，mF1 `67.60`；
- boundary-q50：mIoU `54.74`，mAcc `64.33`，mF1 `67.67`；
- boundary-q75：mIoU `54.64`，mAcc `64.32`，mF1 `67.57`；
- nonboundary-q50：mIoU `54.43`，mAcc `64.02`，mF1 `67.39`。

预注册主分析覆盖 138/138 个位置组，执行 10,000 次 bootstrap：

- `dose_effect = boundary-q75 - clean`：`+0.0731348717` 个 Boundary-IoU 百分点，95% 区间 `[-0.0620441424, +0.2226503089]`；
- `specificity_effect = boundary-q50 - nonboundary-q50`：`+0.0323919541` 个百分点，95% 区间 `[-0.3002343847, +0.2899672010]`；
- 联合裁决：`not-supported`。

预定义的 123 组全可构造敏感性分析方向一致；15 个部分可构造组的描述性分析也没有改变主裁决。31 张 q75 实际置零数为 0 的图像被保留并如实报告，没有通过删图制造结果。

### 8.4 该负结果意味着什么

当前证据只支持：在固定的可构造 `val-dev` 位置组中，人工把深度边界像素置零，没有观察到预注册的边界特异敏感性。

它不支持以下扩展：

- 不能说自然无效深度没有问题；
- 不能说低照、粉尘或传感器故障不会影响模型；
- 不能说 Depth 对 DFormerv2 没有作用；
- 不能外推到 official test、真实部署、跨数据集或多 seed 统计。

## 九、当前重点：DVG-B1 Oracle GSA 门控

### 9.1 为什么 A 不支持后仍设计 B

A 检查的是“预设的问题定位是否成立”；B 检查的是“已知坏区时，少信坏深度这个动作有没有用”。两者逻辑独立。

如果 B 支持，只能说明最小门控动作值得继续研究质量信号或可学习可靠性；不能把 A 改写成支持。如果 B 也不支持，则应关闭“深度边界 + GSA Oracle 门控”作为首选路线，转向其他瓶颈。

### 9.2 已关闭的项目内实现问题

- `GeoPriorGen.forward` 在 spatial contribution 与 depth geometry contribution 相加前暴露了可分离位置。
- 前三个 stage 使用 H/W 分解 GSA，第四个 stage 使用 Full GSA。
- 唯一允许改变的是 `self.weight[1] * mask_d*` 对应的 depth geometry contribution；spatial contribution、sin/cos、Q/K/V、Depth 输入、decoder 和最终 logits 结构保持不变。
- 最小参数链已明确：`EncoderDecoder.forward/encode_decode` → `dformerv2.forward` → `BasicLayer.forward` → `RGBD_Block.forward` → `GeoPriorGen.forward`。
- Attention 继续只消费合成后的 geometry prior，不直接接收 Oracle mask。
- `None`、clean、q=0 和全可信 mask 必须走原始 forward 旁路，逐 stage 输出和最终 pre-softmax logits 均以 `torch.equal` 完全相等为验收条件。
- 当前代码与作者保留副本完整一致；论文写 average pooling、作者代码使用 bilinear interpolation 的差异属于上游论文—代码差异，B1 以 checkpoint 对应代码为准。

### 9.3 当前三组阻塞项

- **A：像素 corruption mask → 四级 token reliability。** 仍需冻结每个 view 的 resize、Stage 0–3 聚合、部分受损 patch 的 reliability 语义，以及与 bilinear Depth resize 的对齐。
- **B：token reliability → pairwise depth contribution gate。** 仍需冻结 Full/H/W 三种 shape、query/key 两端组合、对称性，以及 hard gate 或 continuous attenuation。
- **C：正式科学裁决。** 仍需冻结 `oracle-supported` 的最小实际效应量、clean 不劣容忍度，以及 Boundary IoU 与 mIoU 是否足够。

A、B、C 任一未关闭前，不创建 protocol、不修改模型、不运行 preflight 或 GPU。

## 十、文献筛选结果

2026-09-10 对用户提供的 A-1 44 条和 B-1 30 条 WOS 记录完成摘要筛选。两文件各自无重复，仅共享 `BurnDC` 一条，并集为 73 个唯一 WOS ID。

### 10.1 对开放项 A 最优先的全文候选

1. *Confidence Propagation through CNNs for Guided Sparse Depth Regression*：优先核对 normalized convolution 中 confidence 的定义与跨层传播。
2. *Bcap-net*：优先核对 hierarchical multi-scale weighted pooling 和 depth confidence propagation。
3. *LightDepth* 及其官方代码：优先核对 sparse depth resize、validity 同步和插值方式。

### 10.2 对开放项 B 最接近的全文候选

1. *Non-local affinity adaptive acceleration propagation network*：摘要已明确 pixel depth reliability 可与 normalized neighbor affinity 结合并调整传播权重，但未给出两端组合公式。
2. *Deep Sparse Depth Completion Using Multi-Affinity Matrix*：核对 confidence 是否直接进入 pairwise affinity。
3. *NR-MVSNet* 及其代码：核对 reliable attention 是否修改 cost-volume score。
4. *LFDA* 及其代码：核对 pairwise attention score 的方向性和对称性。

当前摘要只能缩小精读范围，不能据此选择 average、valid fraction、乘积、最小值、query-only 或 hard/continuous gate。

## 十一、其他可选方向

当前 MVE 只验证一个最窄机制方向。其他方向保留为候选，但没有进入执行状态：

1. **整体深度有效性 A2/B2：** 检查自然或人工整体深度失效，并在 geometry prior 中显式处理 validity。已延期、未执行、未授权；未来启用必须重新建立 protocol。
2. **后验校准与风险—覆盖：** 在冻结 logits 上研究置信度、错误排序和阈值迁移，不修改融合骨干。该方向与 DVG-B1 逻辑独立，当前同样延期。
3. **RGB 低照与粉尘退化：** 研究彩色观测退化及其与深度可靠性的联合作用。当前只有方向级候选，没有冻结 corruption、数据职责或评价协议。
4. **观测保真、深度补全与标定误差：** 包括深度填补、传感器尺度、RGB-D 错位或标定偏差。首轮因变量过多和依据不足被删除，未来需要独立问题定义。
5. **可学习质量预测与完整联合恢复：** 只有 Oracle 门控先显示实际价值，才值得引入质量估计网络、训练自由度和完整双路径恢复。

## 十二、当前结果、风险和决策

### 12.1 已完成并验证

- MUSeg 数据确定性转换与全量像素核验。
- background/ignore、Depth invalid、RGB/BGR、训练裁剪与评价网格的语义分离。
- `safe_masked_mean` 数值稳定性修复。
- single-seed RGB Quick-B0 的 500 epoch 训练、4 候选主评价和最终 checkpoint 裁决。
- DVC-A1 v1/v2 的合法阻塞记录，以及 v3 的完整开发评价、bootstrap 和 `not-supported` 裁决。
- DVG-B1 的项目内 GSA 实现锚点、唯一 gate 位置、参数链、no-op 验收边界和 73 条摘要筛选。

### 12.2 正在进行

- 获取 DVG-B1 的 P0 全文、补充材料和官方代码。
- 从直接实现中提取 A/B 可复现公式，并准备 C 的预注册选择。

### 12.3 当前主要风险

- 摘要级术语相似不等于公式可复现，过早选 gate 会把项目偏好伪装成文献依据。
- A 的 `not-supported` 使 B 的可观察收益可能很小；必须提前冻结最小效应量和 clean 保持门槛，避免结果后解释。
- Oracle 结果即使为正，也只代表使用真实 corruption mask 的理想上限；真实质量检测仍是独立难题。
- 所有当前结果均来自 development `val-dev` 和 single-seed B0，不能扩展为 official test 或论文级泛化。

## 十三、建议下一步

1. 获取 A-P0 与 B-P0 论文全文、补充材料和官方代码，优先处理 Confidence Propagation、Bcap-net、Non-local affinity propagation、Multi-Affinity Matrix、LightDepth、NR-MVSNet 和 LFDA。
2. 对 A 逐项冻结：view-scale mask resize、四级聚合算子、部分有效与空 patch、全可信恒等。
3. 对 B 逐项冻结：Full/H/W pairwise 公式、query/key 组合、对称性、归一化顺序、hard/continuous 和全 1 恒等。
4. 对 C 明确作出项目预注册选择或补充直接参考：最小实际效应量、clean 不劣界和指标集合。
5. A/B/C 全部关闭后，再申请 `DVG-B1-oracle-gsa-v1` protocol 与代码实现授权。
6. 实现获批后按最小顺序执行：静态/张量检查 → 原始旁路完全等价 → 1–2 样本 preflight → 单独批准的完整 paired development evaluation。
7. 若 DVG-B1 也不支持，停止当前首选路线，按新的问题定义在整体深度失效、RGB 低照/粉尘、后验校准或标定误差中选择一个方向，不并行堆叠多个变量。

## 十四、组会讲述顺序建议

1. 用一句话说明目标：先建立可信基线，再用最小实验判断几何可靠性方向值不值得继续。
2. 用 3171 样本、31.98 亿像素和 30.74% 无效深度说明数据适配工作量与研究动机。
3. 用 epoch 480/420 排名反转和最终 `58.79/69.91/72.73` 说明基线与评价协议的成果。
4. 用三条简短内容交代早期试错：安全 loss、16 张 pilot、BGR/geometry 协议敏感性。
5. 重点讲 DVC-A1 的 v1 → v2 → v3：两次合法阻塞不是失败浪费，而是把操作定义和评价定义修到能产生有效科学裁决。
6. 展示 v3 两个 effect 接近 0 且区间跨 0，明确结论为 `not-supported`。
7. 解释为什么仍做 DVG-B1：A 问“问题是否找准”，B 问“动作是否有效”。
8. 展示唯一 gate 位置和 A/B/C 三个剩余门禁，说明当前进展已从泛泛方案进入可实现公式选择。
9. 最后简述其他方向，并用清晰条件收口：Oracle 有效才进入质量预测；Oracle 无效则换问题，不堆复杂度。

## 十五、证据入口与验证说明

主要事实源：

- `doc/main/MUSeg-current-status.md`；
- `doc/main/MUSeg-open-decisions.md`；
- `doc/reports/2026-08-19-museg-dformer-data-processing-review.md`；
- `doc/reports/2026-08-31-museg-dformerv2-quick-baseline-comprehensive.md`；
- `doc/reports/2026-09-08-museg-dvc-a1-protocol-gate.md`；
- `doc/reports/2026-09-09-museg-dvc-a1-v2-protocol-blocked.md`；
- `doc/plans/2026-09-MUSeg-几何可信RGBD双路径MVE/03-共享协议与DVC-A1问题验证.md`；
- `doc/plans/2026-09-MUSeg-几何可信RGBD双路径MVE/04-DVG-B1条件式Oracle门控.md`；
- `liu-test-exp/方案1/DVG-B1-A1-B1摘要筛选与全文优先级.md`。

本报告是对既有证据的组会整合，没有重新运行训练、GPU 评价、完整测试套件、云资源或 official test。按验证预算，本次只进行内容交叉复核、差异检查、报告索引 JSON 校验和 Canvas TypeScript 检查；未运行项目测试不能写成通过。