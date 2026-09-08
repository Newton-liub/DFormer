# MUSeg `DVC-A1-valdev-boundary-zero-v1`：共享协议与问题验证

> **文档角色：** 第一个详细子计划；保留 v1 的执行记录，并在原方案上追加 A1-v2 的后继执行规划。
> **计划状态：** v1 已执行到合法终点 `protocol-blocked`；v2 已完成研究口径规划，尚未物化 protocol、修改代码或运行评价。
> **形成或核验时点：** 2026-09-08。
> **实时入口：** [`MUSeg-current-status.md`](../../main/MUSeg-current-status.md)。
> **研究选择：** [`MUSeg-open-decisions.md`](../../main/MUSeg-open-decisions.md)。
> **上级方向：** [`00-总方向规划.md`](./00-总方向规划.md)。
> **前序关系：** [`02-文献补充与协议设计门禁.md`](./02-文献补充与协议设计门禁.md) 已完成。
> **当前/后继关系：** 当前规划为 `DVC-A1-valdev-boundary-zero-v2` 的可构造位置组验证；v1 证据见 [`2026-09-08-museg-dvc-a1-protocol-gate.md`](../../reports/2026-09-08-museg-dvc-a1-protocol-gate.md)。[`04-DVG-B1条件式Oracle门控.md`](./04-DVG-B1条件式Oracle门控.md) 仍未解锁。

## 1. 目标、问题与结论边界

目标是回答：在 RGB、Label、checkpoint 和 evaluator 不变时，只将**原始有效 Depth16 中的深度边界像素按剂量置零**，是否比同面积的非边界置零更稳定地降低语义 Boundary IoU。

【文献依据：DFormerv2 以 Depth patch 差形成 GSA 几何先验，见参考资料 P0-1；Boundary IoU 原始定义见 P0-5。】

本实验只能形成“冻结 Quick-B0 在人工边界失效下的 paired development sensitivity”结论。它不能证明自然传感器故障、真实低照/粉尘、绝对三维几何、部署安全或论文级泛化。

## 2. 输入身份与不变量

- checkpoint：epoch 420 `selector-epoch-420.pth`，SHA-256 `f246a3afc50334c81302b7bfebdadf7cf37d00326bf1c3aa54f6a151754e3a1c`。
- split：`val-dev` 318 条、196 个 location group，SHA-256 `1d0719d8f64f016d48995c25ab66d4004d76b7155d9efeef7cbb7454c0dd0e83`。
- RGB contract：`rgb-imagenet-rgb-order-v1`。
- evaluator 基底：`msflip-whole-original-grid-v1`，五尺度各含原图/水平翻转，FP32 平均 pre-softmax logits，恢复到原始 Label 网格。
- Depth contract：原始 `Depth16` 是二维 `uint16`、`0` 无效；注入后按 `round(D16*255/13932)` 量化到 8-bit，再按现有链复制三通道、归一化并双线性 resize。
- Label contract：原始 background `0` 映射为 ignore `255`；只评价 15 个前景类。
- official test：`sealed_unread`，manifest 必须写 `official_test_included=false`。
- 不变量：RGB、Label、checkpoint、类别映射、inference views、logits 融合、metric grid 和指标实现均不得随 condition 改变。

## 3. 唯一主变量与条件

主变量是原始有效 Depth16 边界候选集合中的累计置零比例 $q$。冻结条件为：

- `clean`：$q=0$；
- `boundary-q25`：边界候选中累计 25% 置零；
- `boundary-q50`：边界候选中累计 50% 置零；
- `boundary-q75`：边界候选中累计 75% 置零；
- `nonboundary-q50`：从非边界有效区置零与 `boundary-q50` 完全相同的像素数，作为位置阴性对照。

这些剂量是本项目为首轮早筛预注册的工程选择，不是论文常数。只使用一个固定全局 seed；同一样本的 25%/50%/75% mask 必须由同一确定性排序前缀生成，保证严格嵌套。首轮不增加多 seed，因为主要独立单位是 location group，多 seed 只会制造相关副本和额外评价成本。

## 4. 边界候选定义

1. 仅在原始 `Depth16>0` 的有效像素上计算，不让自然无效值参与阈值估计。
2. 对相邻有效像素计算相对深度跳变：

   $$
   g(p,p')=\frac{|D(p)-D(p')|}{\max(D(p),D(p'),1)}.
   $$

3. 若水平或垂直相邻对满足 $g\ge0.05$，则该对两端像素进入 depth-edge seed；`0.05` 是本项目预注册阈值，不是 PR151 的多视图可见性阈值。
4. 在原始 `1082\times932` 网格上用 `3\times3` 结构元膨胀一次形成边界候选集合。
5. 非边界有效区定义为：有效像素减去边界候选及其再膨胀 5 像素的 guard band。若可选像素少于匹配数量，该样本的 `nonboundary-q50` 标为不可构造并触发 protocol 检查，不允许缩小对照面积。

这是简单、显式、无需额外论文的项目操作定义；不把它称为 PR117 depth integrity prior。【PR117 的原方法依赖 ground truth/pseudo-depth 与训练，见参考资料 P1-1。】

## 5. Boundary IoU 冻结口径

对每个前景类 $c$ 构造 one-vs-rest prediction mask $P_c$ 与 ground-truth mask $G_c$。在原始 Label 网格上，对各自 mask 的**内侧**边界带计算：

$$
\operatorname{BoundaryIoU}_c=
\frac{|(G_{c,d}\cap G_c)\cap(P_{c,d}\cap P_c)|}
{|(G_{c,d}\cap G_c)\cup(P_{c,d}\cap P_c)|}.
$$

【文献依据：Cheng et al., *Boundary IoU*, CVPR 2021；见参考资料 P0-5。】

- 距离带宽 $d=\max(1,\operatorname{round}(0.02\sqrt{H^2+W^2}))$；2% 来自原论文 sensitivity analysis。
- ignore pixels 在边界提取和最终交集/并集时同时排除；为避免 ignore 区域边缘被误当作类别边界，距 ignore 区不超过 $d$ 的像素也不参与该图类 pair 计分。
- 若该图 $G_c$ 与 $P_c$ 都为空，该图类 pair 不进入平均；仅一侧为空时该 pair 计 0。
- 每图先对有定义的前景类做 macro average，得到 image-level Boundary IoU；同一 location group 再对其图像求等权平均。六个 mine 仅作描述性分层，不作六个独立样本。
- 历史 `boundary_band_mIoU` 不符合该定义，禁止复用名称或数值。

## 6. 统计单位与裁决

### 6.1 效应量

每个 location group 先聚合组内图像，随后计算：

- `dose_effect`：`boundary-q75 - clean` 的 Boundary IoU 配对差值；
- `specificity_effect`：`boundary-q50 - nonboundary-q50` 的 Boundary IoU 配对差值；
- 辅助指标：全图 mIoU 的同类差值、各 condition 的有效深度比例、被置零像素数和每类 Boundary IoU。

### 6.2 区间

使用固定 seed `20260908`，对 196 个 location group 做 10,000 次有放回重采样；每次保留抽中 group 的全部图像，报告组均值差的双侧 95% percentile interval。该 bootstrap 是本项目预注册的常规统计流程，不归因于 `RE447`。不做多重阈值搜索、p-value 或结果后扩样。

### 6.3 联合裁决

- `supported`：`dose_effect` 的点估计不高于 `-2.0` Boundary-IoU 百分点且 95% interval 上界小于 0；同时 `specificity_effect` 点估计小于 0，且六个 mine 中至少 4 个方向为负。
- `not-supported`：`dose_effect` 点估计大于 `-1.0` Boundary-IoU 百分点，或 `specificity_effect` 点估计不小于 0。
- `inconclusive`：不满足以上两类，但 protocol 与运行均有效。只允许按预先生成的 per-group 结果检查稀有类/空类贡献，不追加 condition、seed、阈值或样本。
- `protocol-blocked`：身份、对照面积、Boundary IoU、q=0 或证据链任一未闭合。

`-2.0/-1.0` 是首轮资源筛选门槛，不是文献阈值、临床/安全阈值或论文级最小效应。

## 7. 分阶段执行与授权

### 阶段 1：protocol 与最小实现

需要用户批准代码修改和读取 `val-dev` 的元数据/少量样本。只实现：

- 独立 protocol JSON/template 与 schema 校验；
- 从 `Depth16` 生成可重建 corruption manifest/mask；
- 在内存中按 condition 量化并喂给 evaluator，不复制完整退化数据集；
- 标准 Boundary IoU 与 per-image/per-group 聚合；
- official-test 路径/样本拒绝。

### 阶段 2：最小 preflight

阶段 1 获批后只做以下定点检查，不新建大测试套件：

1. 同一样本的 q25/q50/q75 嵌套且 hash 稳定；
2. q=0 生成的 8-bit Depth 与现有 `Depth/` byte-equal；若文件编码差异导致 byte 比较不适用，则 decoded array 必须完全相等；
3. 一个手工小 mask 的 Boundary IoU 与人工期望一致，包括 ignore、双空和单空类；
4. 1–2 个 `val-dev` 样本完成 clean/q50 推理，输出有限、尺寸回到原始 Label 网格、checkpoint strict load；
5. manifest 明确拒绝 official test。

任一失败即 `stop`，不进入完整评价。

### 阶段 3：完整开发评价

只有 preflight 通过且用户另行明确批准 GPU 运行后，才对 318 条 `val-dev` 执行 5 个 condition 的完整五尺度翻转评价。不得把该授权外推为训练、云资源或 official test 授权。

## 8. 交付物与证据位置

获批实现时再冻结具体路径，建议：

- protocol/template：`protocols/dvc-a1-valdev-boundary-zero-v1.template.json`；
- 工具：`tools/mve/` 下语义明确的新文件，不覆盖历史 `a2_*`；
- 小型测试：优先扩展最相关的现有测试文件；只有无法清晰容纳时才新增一个聚焦测试文件；
- 运行证据：仓库外 `cloud/` 对应独立目录；大型 logits、预测、mask 和日志不进 Git；
- Git 只保存 protocol、代码、必要的小 manifest/摘要和日期化报告。

每次正式运行记录命令、环境、退出码、代码 commit/diff、checkpoint/split/protocol/evaluator hash、condition manifest hash、指标 CSV/JSON hash、失败与恢复历史、`official_test_included=false`。

## 9. 失败与恢复

- q=0 不等价：优先检查 PNG decode、16→8 rounding、归一化和 resize；修复前不看科学指标。
- 边界候选为空或过小：记录受影响 group；若超过 5% 的 location group 无法构造 q75，判 `protocol-blocked`，不在结果后降低阈值。
- 非边界对照不足：判 `protocol-blocked`，不缩小面积或删除不利样本。
- OOM/中断：保留已完成 group 和哈希；仅从相同 protocol/代码/环境恢复，输入语义改变则新建 v2 identity。
- 数值结果不支持：按 `not-supported` 或 `inconclusive` 收口，不追加测试寻找正结果。

## 10. 验证预算与状态更新

本节保留 v1 形成时点的预算边界；v1 后续已按该边界完成代码、定点检查、两样本 preflight 和全量 mask 门禁，并在完整模型评价前以 `protocol-blocked` 收口，实际证据以状态头所链报告为准。本次 v2 修订仍只做规划，不运行项目测试、GPU、训练、云端或 official test。后续先按第 11.6 节另行物化 v2 protocol 和执行最小 preflight；完整评价须在其通过后再次确认。protocol 物化、preflight、正式评价或恢复点发生变化时，最终答复前更新实时状态文件。

## 11. A1-v2 后继规划：可构造位置组的全量快速验证

### 11.1 这次修订要解决什么

v1 的问题不是模型结果不支持假设，而是协议在完整模型评价前发现：按相对深度跳变阈值 `0.05` 构造的 `boundary-q75` 在 58/196 个 location group 中没有任何可执行样本，因而触发 `protocol-blocked`。本节不修改 v1 的原始 protocol、manifest 或失败证据，只定义一个新的后继身份：

- **完整实验身份：** `DVC-A1-valdev-boundary-zero-v2`；其中 `DVC` 表示 Depth-validity corruption check（深度有效性破坏检查），`A1` 表示问题/现象验证分支的第一阶段，`valdev-boundary-zero` 表示在开发集上对深度边界进行置零破坏。
- **核心变化：** 保留阈值 `0.05` 和全部五种 condition，仅把“至少一个样本可构造非空 q75”的 138 个 location group 预先定义为本次评价范围。
- **不变事项：** 不重新划分 `train-dev`/`val-dev`，不从原始数据中另抽训练集，不降低 `0.05`，不查看模型结果后选择样本，不读取 official test，不重新训练或重新选择 checkpoint。

大白话说：这次不是换一批数据来让实验通过，而是承认“没有可破坏边界的地点无法回答这个特定问题”，然后在模型结果出现以前，按已经完成的 Depth16 mask 事实固定可验证范围。

### 11.2 数据范围与准确的研究对象

v2 仍然使用冻结的 `val-dev`，但把 v1 mask manifest 中列出的 58 个不可构造组排除在本次主干干预之外；这个排除只依赖原始 Depth16 和已冻结的 v1 mask 规则，不依赖任何模型输出、指标或类别结果。

直接核验并写入 v2 protocol 的覆盖事实为：

- 原 `val-dev`：318 张图、196 个 location group；
- v2 纳入：138 个 location group、218 张图；
- v2 未纳入：58 个 location group；覆盖率为 138/196，即约 `70.4082%`；
- 在纳入的 218 张图中，有 31 张图的 `boundary-q75` 实际置零数量仍为 0；这是因为 v2 的组级纳入规则是“组内至少一张图可构造 q75”，不是“组内每张图都可构造 q75”；
- 纳入的 138 组中，123 组的每张图都能构造非空 q75，另有 15 组包含至少一张 q75 为空的图；
- v1 已直接核验 `nonboundary-q50` 同面积候选不足为 0 条。

因此，v2 的主要估计对象必须写成：**在具有至少一个可执行深度边界干预的 `val-dev` location group 中，固定 Quick-B0 对边界深度置零的开发期敏感性。**

v2 不声称 58 个组没有深度边界，也不声称它们没有问题；只能说在当前“有效相邻深度相对跳变不低于 `0.05`”的操作定义下，无法对它们执行本实验的 q75 干预。

### 11.3 为什么运行 138 组的全部图像

实际运行范围不是只挑每组一张，而是纳入 138 个组中的全部 218 张图。这样做有三个好处：

1. 不再因为图片数量或结果好坏二次挑选样本；
2. 保留每个位置组内原有的图像相关结构；
3. 统计时仍以 location group 为相关性边界，避免把同一地点的多张图误当作完全独立样本。

对其中 31 张 q75 置零数量为 0 的图，程序必须保留并显式记录 `boundary-q75 == clean` 的事实，不能把它们伪装成有实际边界破坏，也不能静默删除。主结果应报告：218 张图中实际发生 q75 置零的图数、各 condition 的置零像素数分布，以及 138 个组中 15 个部分可构造组的单独描述性结果。

同时，v2 protocol 预先定义一个**敏感性分析**：只在 123 个“组内每张图均能构造非空 q75”的组上重复同一聚合和 bootstrap。该分析只用于判断 31 张无实际 q75 干预的图是否稀释主效应，不作为看结果后选择的替代主结论，也不增加新的成功门槛或多重比较裁决。

### 11.4 v2 的唯一主变量和五种 condition

v2 不改变 v1 的边界和剂量定义：

- `clean`：不置零；
- `boundary-q25`：在最终边界候选中按确定性排序置零 `floor(N×0.25)` 个像素；
- `boundary-q50`：置零 `floor(N×0.50)` 个像素；
- `boundary-q75`：置零 `floor(N×0.75)` 个像素；
- `nonboundary-q50`：从远离边界 guard band 的有效非边界区中，置零与该图 `boundary-q50` 完全相同数量的像素。

其中 $N$ 是一张图在 `3×3` 膨胀并扣除无效深度后的最终边界候选像素数。q75 可执行的最低条件是 $N\ge2$；q25 即使在部分可构造图中为 0，也不阻塞 v2，因为本轮核心比较是 q75 相对 clean，以及 boundary-q50 相对 nonboundary-q50。q25 只作为剂量轨迹的辅助记录。

所有 condition 继续复用相同的五尺度原图/水平翻转 evaluator，共 10 个 view；RGB、Label、checkpoint、Depth16 解码、量化、FP32 logits 融合和原始 Label metric grid 全部保持不变。

### 11.5 v2 的主要统计和裁决

主分析只在预先固定的 138 个纳入组上进行：

- `dose_effect`：`boundary-q75 - clean` 的 Boundary IoU 配对差值；
- `specificity_effect`：`boundary-q50 - nonboundary-q50` 的 Boundary IoU 配对差值；
- 先对图像形成记录，再按 location group 聚合；bootstrap 以 138 个组为重采样单位，保留组内全部图像；
- bootstrap seed、重复次数、Boundary IoU 定义、ignore 处理、空类处理和原 v1 相同，除组范围和 protocol identity 外不改动；
- 六个 mine 只作描述性分层，不把六个 mine 当作六个独立样本。

主裁决沿用 v1 的资源筛选门槛，不因为缩小研究范围而降低标准：

- `supported`：`dose_effect` 点估计不高于 `-2.0` Boundary-IoU 百分点，95% 区间上界小于 0；同时 `specificity_effect` 点估计小于 0，且六个 mine 中至少四个方向为负；
- `not-supported`：`dose_effect` 点估计大于 `-1.0` 个百分点，或 `specificity_effect` 点估计不小于 0；
- 其余有效结果为 `inconclusive`；
- 身份、mask、对照面积、数值或证据链不闭合时为 `protocol-blocked`，不得改写成科学上的不支持。

这里的 `supported` 只支持一个窄结论：**在可构造位置组中，固定 RGB-D 模型对人工深度边界失效表现出开发期敏感性。**它不支持“所有 MUSeg 地点都有该问题”、自然传感器故障、真实低照/粉尘机制、三维几何绝对误差或部署安全结论。

### 11.6 v2 执行阶段与门禁

#### 阶段 1：物化 v2 protocol 和评估清单

仅使用已有 v1 `mask-manifest.json` 和冻结 `val-dev` 清单，生成一个新的、只读派生的 v2 evaluation allowlist。它不是新的训练/验证划分，而是本次 MVE 的评价范围清单；建议文件名为 `val-dev-constructable-v2.txt`，内容为 138 个纳入组的全部 218 条样本。

物化时必须记录：

- v1 mask manifest SHA-256；
- v2 allowlist SHA-256；
- 138 个组、218 张图、31 张 q75 无实际置零图和 123/15 组级构造性统计；
- v2 protocol SHA-256、源代码 SHA-256、checkpoint SHA-256、原 `val-dev` split SHA-256；
- `official_test_included=false`。

若物化结果不是 138 组/218 张图，或派生清单不能证明每条样本来自原冻结 `val-dev`，立即 `stop`，不得继续运行。

#### 阶段 2：v2 最小 preflight

对 v2 身份重新执行最小检查，因为 v1 的 preflight 不能替代新 protocol 的精确身份检查：

1. 检查派生清单只包含 138 个既定组，且样本数为 218；
2. 在至少两个不同位置组上检查 mask hash 稳定、q25/q50/q75 嵌套和 q=0 decoded-array 等价；
3. 检查 boundary-q50 与 nonboundary-q50 的置零数相等；
4. 检查至少一个 q75 非空样本和一个 q75 为空但属于部分可构造组的样本，确认两种情况均被明确记录；
5. 用少量样本确认 strict checkpoint load、有限输出和原始 Label 网格恢复；
6. 确认 v2 不读取 official test，也不依赖模型结果生成纳入清单。

任一检查失败即 `stop`；修复若改变 mask 语义，必须建立新的 protocol identity，不能覆盖 v2。

#### 阶段 3：138 组全量五条件评价

preflight 通过后，对 138 个纳入位置组的全部 218 张图运行五个 condition 的完整五尺度翻转评价。运行顺序、batch、FP32 设置、日志字段和内存中生成退化 Depth 的规则与 v1 一致；不保存完整退化数据集，不保存无必要的 logits 或预测图。

正式运行前再次确认：本次只做本地开发评价，不训练、不使用云资源、不读取 official test。运行完成后，先核验每个 condition 的样本覆盖、mask hash 和数值有限性，再进行统计裁决。

### 11.7 结果解释和后续分支

- **A1 问题支持：** 主分析在 138 个组上达到预注册的 `supported` 联合裁决，即可形成“可构造位置组中的条件性开发支持”结论；123 组敏感性分析若方向一致，只增强稳健性说明，不作为额外成功门槛。
- **A1 问题不支持：** 主分析达到预注册的 `not-supported` 裁决，则停止把“深度边界失效”作为首要瓶颈，不直接实现完整联合恢复；敏感性分析不能推翻主裁决。
- **结果不确定：** 主分析有效但区间过宽，先保留 `inconclusive`，不增加 condition、阈值或模型自由度；只有另行批准时才能设计最小扩样或独立验证。
- **主分析与 123 组敏感性分析方向冲突：** 不做结果后择优，也不改写主分析的预注册裁决标签；单独报告组级构造性与样本级构造性不一致，回到边界候选定义诊断，并暂不解锁 `DVG-B1`。
- **A1 失败但 Oracle 仍有兴趣：** 只有在用户重新批准并建立后继 protocol 后，才考虑 `DVG-B1`；当前 `04-DVG-B1条件式Oracle门控.md` 不因 v2 计划自动解锁。

### 11.8 证据、验证预算和禁止事项

v2 正式证据继续保存到仓库外独立目录，例如 `cloud/DVC-A1-valdev-boundary-zero-v2/`；仓库内只保存 protocol/template、必要代码、清单摘要和日期化报告，大型 mask、logits、预测、退化图像和运行日志不进 Git。

计划阶段只做 Markdown 内容、链接和差异检查，不运行项目测试、GPU、训练、云任务或 official test。后续执行只使用本节列出的定点 preflight 和一次 138 组全量本地 GPU 评价；不自动追加阈值搜索、多 seed、额外 condition、重划数据集或完整测试套件。

v1 的 `protocol-blocked`、mask manifest 和失败日志保持原样，不能覆盖、改名或回写为 v2 结果。v2 若运行中断，保留已完成 condition、样本和哈希，只有在完全相同的 protocol/code/environment 下恢复；任何候选定义、剂量规则、纳入组规则或统计单位变化都建立新的 protocol identity。

### 11.9 当前恢复点

本次计划更新完成后，准确恢复点是：**先物化并审核 `DVC-A1-valdev-boundary-zero-v2` 及其 138 组/218 样本评价清单，再执行 v2 preflight；尚未开始代码修改、protocol 物化或 GPU 评价。**

本次用户只确认 v2 的研究设计与全 138 组范围，不构成本对话中的代码修改、protocol 物化或 GPU 运行授权；这些执行不能被描述为已开始或已通过。完成 v2 物化、preflight 和全量评价后，必须更新 `MUSeg-current-status.md`，并在有科学裁决时同步更新 `MUSeg-open-decisions.md`。
