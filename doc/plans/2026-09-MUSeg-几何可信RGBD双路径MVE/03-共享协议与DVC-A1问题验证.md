# MUSeg `DVC-A1-valdev-boundary-zero-v1`：共享协议与问题验证

> **文档角色：** 第一个详细子计划；保留 v1/v2 的执行记录，并追加 A1-v3 的定义修正与最小门禁结果。
> **计划状态：** v1 已执行到合法终点 `protocol-blocked`；v2 已完成独立 allowlist 物化、完整评价并因有效配对组门禁以 `protocol-blocked` 收口；v3 已完成定义实现、218 张图 CPU 标签域审计、新 protocol 物化、两样本本地 GPU preflight 和完整 218 张图 × 5 condition GPU 评价，正式评价状态为 `completed`，预注册主裁决为 `not-supported`。
> **形成或核验时点：** 2026-09-09 04:43 UTC。
> **实时入口：** [`MUSeg-current-status.md`](../../main/MUSeg-current-status.md)。
> **研究选择：** [`MUSeg-open-decisions.md`](../../main/MUSeg-open-decisions.md)。
> **上级方向：** [`00-总方向规划.md`](./00-总方向规划.md)。
> **前序关系：** [`02-文献补充与协议设计门禁.md`](./02-文献补充与协议设计门禁.md) 已完成。
> **当前/后继关系：** `DVC-A1-valdev-boundary-zero-v3-bgcontext` 的定义层、最小模型链门禁和完整 218 张图 × 5 condition GPU 评价均已完成；主分析按预注册规则裁决为 `not-supported`，完整结果见本文件第 12.5 节。v1/v2 历史证据保持原样。[`04-DVG-B1条件式Oracle门控.md`](./04-DVG-B1条件式Oracle门控.md) 已按用户要求完成基础设计，仍未授权代码、GPU、训练或云执行。

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

只有 preflight 通过且用户另行明确批准 GPU 运行后，才对冻结评价范围执行 5 个 condition 的完整五尺度翻转评价。该授权不外推为训练、云资源或 official test 授权。

本阶段已完成：用户已授权 `DVC-A1-valdev-boundary-zero-v3-bgcontext` 的完整本地 GPU 评价，固定范围为 218 张图、138 个 location group 和 5 个 condition。

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

本节保留 v1/v2 形成时点的预算边界；v1 后续已按该边界完成代码、定点检查、两样本 preflight 和全量 mask 门禁，并在完整模型评价前以 `protocol-blocked` 收口。v2 随后已完成独立 protocol/allowlist 物化、聚焦测试、两样本本地 GPU preflight 和完整评价，最终因有效配对组门禁以 `protocol-blocked` 收口。v3 已完成标签契约修正、CPU 标签域审计、protocol 物化、两样本 GPU preflight、完整 GPU 评价、location-group bootstrap 和预注册裁决；未追加阈值搜索、多 seed、额外 condition、重划数据集或完整测试套件。

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

#### 阶段 1：物化 v2 protocol 和评估清单（已完成并验证）

仅使用已有 v1 `mask-manifest.json` 和冻结 `val-dev` 清单，生成一个新的、只读派生的 v2 evaluation allowlist。它不是新的训练/验证划分，而是本次 MVE 的评价范围清单；建议文件名为 `val-dev-constructable-v2.txt`，内容为 138 个纳入组的全部 218 条样本。

物化时必须记录：

- v1 mask manifest SHA-256；
- v2 allowlist SHA-256；
- 138 个组、218 张图、31 张 q75 无实际置零图和 123/15 组级构造性统计；
- v2 protocol SHA-256、源代码 SHA-256、checkpoint SHA-256、原 `val-dev` split SHA-256；
- `official_test_included=false`。

若物化结果不是 138 组/218 张图，或派生清单不能证明每条样本来自原冻结 `val-dev`，立即 `stop`，不得继续运行。

#### 阶段 2：v2 最小 preflight（修正前后均已完成，历史证据）

对 v2 身份重新执行最小检查，因为 v1 的 preflight 不能替代新 protocol 的精确身份检查：

1. 检查派生清单只包含 138 个既定组，且样本数为 218；
2. 在至少两个不同位置组上检查 mask hash 稳定、q25/q50/q75 嵌套和 q=0 decoded-array 等价；
3. 检查 boundary-q50 与 nonboundary-q50 的置零数相等；
4. 检查至少一个 q75 非空样本和一个 q75 为空但属于部分可构造组的样本，确认两种情况均被明确记录；
5. 用少量样本确认 strict checkpoint load、有限输出和原始 Label 网格恢复；
6. 确认 v2 不读取 official test，也不依赖模型结果生成纳入清单。

任一检查失败即 `stop`；修复若改变 mask 语义，必须建立新的 protocol identity，不能覆盖 v2。

实际完成证据：v2 protocol SHA-256 为 `bc71ee97421a97f1ae172ec5f1096bf026750a3e4fae539eccfffc2c53aca535`，allowlist SHA-256 为 `5589eb3378ed2e23180f6205e2d88cea39702ad4bfd5d4e1b739cf2f920a8d89`，preflight SHA-256 为 `9129103d5b3612f21dd7c86291a46a6a3102b4313b30357c4904ced639736e15`。两样本来自不同位置组，分别覆盖 q75 为空和 q75 非空情况；mask 稳定、嵌套、q50 同面积、q=0 decoded-array 等价、有限值、strict checkpoint load 和原始 `932×1082` Label 网格恢复均通过，且 `official_test_included=false`。详细证据见 [`2026-09-08-museg-dvc-a1-v2-materialization-preflight.md`](../../reports/2026-09-08-museg-dvc-a1-v2-materialization-preflight.md)。大白话说，新评价范围和最小模型链已闭合，但还没有运行完整评价或产生科学结论。

#### 阶段 3：138 组全量五条件评价（已完成但协议阻塞）

最终差异审查发现的两项缺口已修正：当前入口把 218 个纳入样本的各 condition mask SHA-256 逐一与冻结 v1 manifest 对照，并在统计阶段前强制主分析 138 组、敏感性分析 123 组及两个 effect 的有效配对组数。旧 protocol、allowlist 和修正前两样本 preflight 证据保持原样，不替代新身份检查。

同一 v2 研究语义下的 protocol 已重新物化，protocol SHA-256 为 `3c6f33562692c8baee85786261de431583e9bdd9b6f8f51cd5ad0f042406f6bb`；allowlist 和构造性摘要哈希未改变。修正后的两样本本地 GPU preflight 已通过，随后按批准完成 218 张图完整评价。五个 condition 产物均已写出，每个包含 218 样本/138 组；统计时发现 `dose_effect` 只有 137 个有效配对组，因此运行以 `protocol-blocked` 结束。只读诊断确认无效组为 `06-01-01-0346`；123 组敏感性范围的两个 effect 均为 123/123 有效。

### 11.7 结果解释和后续分支

- **A1 问题支持：** 主分析在 138 个组上达到预注册的 `supported` 联合裁决，即可形成“可构造位置组中的条件性开发支持”结论；123 组敏感性分析若方向一致，只增强稳健性说明，不作为额外成功门槛。
- **A1 问题不支持：** 主分析达到预注册的 `not-supported` 裁决，则停止把“深度边界失效”作为首要瓶颈，不直接实现完整联合恢复；敏感性分析不能推翻主裁决。
- **结果不确定：** 主分析有效但区间过宽，先保留 `inconclusive`，不增加 condition、阈值或模型自由度；只有另行批准时才能设计最小扩样或独立验证。
- **主分析与 123 组敏感性分析方向冲突：** 不做结果后择优，也不改写主分析的预注册裁决标签；单独报告组级构造性与样本级构造性不一致，回到边界候选定义诊断，并暂不解锁 `DVG-B1`。
- **A1 失败但 Oracle 仍有兴趣：** 只有在用户重新批准并建立后继 protocol 后，才考虑 `DVG-B1`；当前 `04-DVG-B1条件式Oracle门控.md` 不因 v2 计划自动解锁。

### 11.8 证据、验证预算和禁止事项

v2 物化与 preflight 证据已保存到仓库外独立目录 `cloud/DVC-A1-valdev-boundary-zero-v2/`；仓库内保存 protocol template、必要代码和日期化报告，大型 mask、logits、预测、退化图像和运行日志不进 Git。

本轮 v2 执行已结束；v3 已完成标签契约实现、聚焦测试、218 张图 CPU 标签域审计、protocol 物化、两样本本地 GPU preflight，以及用户授权后的完整 138 组/218 样本五条件 GPU 评价。完整评价未新增阈值搜索、多 seed、额外 condition 或数据重划；正式统计和科学裁决按冻结 protocol 收口。

v1 的 `protocol-blocked`、mask manifest 和失败日志保持原样，不能覆盖、改名或回写为 v2 结果。v2 若运行中断，保留已完成 condition、样本和哈希，只有在完全相同的 protocol/code/environment 下恢复；任何候选定义、剂量规则、纳入组规则或统计单位变化都建立新的 protocol identity。

### 11.9 v2 历史恢复点

本节记录 v2 的历史恢复点：**`DVC-A1-valdev-boundary-zero-v2` 的冻结 allowlist、构造性摘要、修正后 preflight、五个 condition 产物和失败证据均保持原样；完整评价已经结束，运行时长为 `5445.565` 秒，状态为 `protocol-blocked`。`dose_effect` 在主分析中只有 137/138 个有效配对组，无效组为 `06-01-01-0346`；123 组敏感性范围的两个 effect 均为 123/123。随后以独立 v3 protocol 处理标签域定义问题。**

当前没有可用于科学裁决的正式 mIoU、Boundary IoU、bootstrap 区间或 `supported/not-supported/inconclusive` 结果；`DVG-B1` 仍未解锁。详细终态见 [`2026-09-09-museg-dvc-a1-v2-protocol-blocked.md`](../../reports/2026-09-09-museg-dvc-a1-v2-protocol-blocked.md)。

## 12. A1-v3 定义修正：有效 background 与 true ignore 分离

### 12.1 修正对象与独立身份

v2 的唯一阻塞组 `06-01-01-0346` 并非没有前景：四张图的原始 Label 都包含 `cable`、`tube`、`rescue equipment`。阻塞原因是旧 Boundary IoU 输入把原始 Label background `0` 先映射为 evaluator ignore `255`，随后又按 ignore 安全距离排除了距离背景不超过 29 像素的区域，导致四张图的计分安全区为空、15 个前景类别全部双空 `None`。

本次只修正 Boundary IoU 的标签域，不回写 v1/v2。新身份为 `DVC-A1-valdev-boundary-zero-v3-bgcontext`，其中 `bgcontext` 表示原始 background 作为有效几何上下文；真正的 void/ignore 才从安全区排除。v2 的阈值、allowlist、Depth corruption、五种 condition、checkpoint、evaluator、bootstrap 和裁决门槛均保持不变。

### 12.2 两套标签契约

- **训练/普通 mIoU 输入：** 继续使用原有 foreground-only 映射，raw Label `0` 映射为 `255`，raw Label `1..15` 映射为前景 `0..14`。
- **v3 Boundary IoU 输入：** 从 raw Label 构造 metric target；raw foreground `1..15` 映射为 `0..14`，raw background `0` 映射为背景上下文 `15`，true ignore 使用 `255`。背景不是待报告类别，但参与 one-vs-rest 的边界几何定义。
- `semantic_boundary_iou` 继续只报告 15 个前景类；`None` 仍表示某类别 prediction 和 target 双空并排除该类 pair，单侧为空仍计 `0`。只有图像/组没有任何定义前景类别时才触发定义层阻塞。

### 12.3 已核验的定义层结果

冻结 v1 manifest 派生的 v3 仍覆盖 218 张图、138 个 location group；其中 215 张图含前景，3 张全背景图保留在范围内，138 个组均至少包含一张前景图。原始标签值为 `0..15`，冻结标签中没有 true ignore，因此 v3 的 Boundary IoU 安全域为全图像素。

对 `06-01-01-0346` 的四张图，v3 metric foreground ids 均为 `[1, 2, 13]`，有效安全域每图为 `1,008,424` 像素，定义层门禁通过。三张全背景图不被静默删除；若某 prediction 与 target 仅一侧有前景，继续按冻结规则计 `0`。

### 12.4 v3 最小门禁结果与边界

- 聚焦测试：`python -m pytest tests/test_dvc_a1.py -q`，`9 passed`；仅出现既有 pytest cache permission warning。
- v3 protocol 已物化，protocol SHA-256 为 `f9960904f51cec11797ada6952c2102da4b2b6832d0bf7b529898bfae9c0f216`；allowlist SHA-256 为 `5589eb3378ed2e23180f6205e2d88cea39702ad4bfd5d4e1b739cf2f920a8d89`，沿用冻结 v1 派生清单；allowlist summary SHA-256 为 `6fa94de96f1b5b4e94c1feecdc4d821e05db1828be05b011f3b48f43ce408dfd`。
- CPU 标签域审计状态为 `passed`，产物 SHA-256 为 `17a4ec36ba231c3be6ea1ed1f4f6e3b9f8380d8d0a619cc3530a6bfd903ad1db`。
- 两样本 CUDA（Compute Unified Device Architecture，图形处理器计算平台）preflight 状态为 `passed`，产物 SHA-256 为 `829b580b6ed4ace977cf578e8391bcc759fc6d135fad72c2bd0dba958712dedd`；样本明确覆盖 `06-01-01-0346` 组，两个样本均通过 q=0 decoded-array 等价、mask hash 与冻结 v1 一致、嵌套、q50 同面积、有限 logits、原始 `932×1082` 网格和 v3 Boundary IoU 有定义检查。

这些结果证明定义、身份和最小模型链门禁已闭合；完整评价和预注册裁决也已完成。当前 A1 的正式结果为 `not-supported`。根据用户明确要求，B 的基础设计已独立完成，但这不改变 A 的裁决，也不自动授权 B 的代码、GPU、训练或云执行。

## 12.5 v3 完整开发评价结果与正式收口

### 12.5.1 运行身份与执行事实

用户已授权并完成 `DVC-A1-valdev-boundary-zero-v3-bgcontext` 的完整本地 GPU 开发评价。运行固定使用冻结的 218 张图、138 个 location group、5 个 condition、五尺度原图/水平翻转 evaluator、epoch 420 checkpoint，且 `official_test_included=false`。

- 运行目录：`cloud/DVC-A1-valdev-boundary-zero-v3-bgcontext-full-20260909/`；
- 执行记录：`executions/20260909T031128418493+0000-full.json`；启动时间为 `2026-09-09T03:11:28.418493+00:00`；
- 执行状态：`completed`，退出码 `0`，运行时长 `5507.055` 秒，约 91 分 47 秒；
- 环境：NVIDIA GeForce RTX 5060 Laptop GPU，PyTorch `2.7.0+cu128`，CUDA `12.8`，Python `3.13.9`，TF32 disabled；
- protocol SHA-256：`d52b3dba2c7a34894b9f4cdf1d8e313e304415d1ea821a191b7753b97be74f5d`；
- mask manifest SHA-256：`3da28ae84806c61b0178c9563e811eed9e6f16e3f0f4040ab5753b26abdc9e79`；
- `summary.json` 状态为 `completed`，SHA-256：`951d8e2005a4c64ac809ffd3de9d7db367df955a46e81c46e89516bd64d9f220`；
- 五个 condition 均生成独立 JSON 产物，每个包含 218 张图和 138 个 location group；具体产物路径及哈希由 `summary.json.condition_files` 固定记录。

### 12.5.2 五个 condition 的总体指标

以下是本次固定开发评价的描述性 mIoU、mAcc 和 mF1；它们不替代 Boundary IoU 的预注册主裁决：

- `clean`：mIoU `54.64`，mAcc `64.23`，mF1 `67.59`；
- `boundary-q25`：mIoU `54.66`，mAcc `64.27`，mF1 `67.60`；
- `boundary-q50`：mIoU `54.74`，mAcc `64.33`，mF1 `67.67`；
- `boundary-q75`：mIoU `54.64`，mAcc `64.32`，mF1 `67.57`；
- `nonboundary-q50`：mIoU `54.43`，mAcc `64.02`，mF1 `67.39`。

### 12.5.3 预注册主分析

主分析覆盖 138 个 location group；`dose_effect` 和 `specificity_effect` 均为 `138/138` 有效配对组。Bootstrap 按冻结 location group 有放回重采样，seed 为 `20260908`，共 `10,000` 次，区间为双侧 95% percentile interval。

- `dose_effect = boundary-q75 - clean`：点估计 `+0.0731348717` 个 Boundary-IoU 百分点；95% 区间为 `[-0.0620441424, +0.2226503089]`；
- `specificity_effect = boundary-q50 - nonboundary-q50`：点估计 `+0.0323919541` 个 Boundary-IoU 百分点；95% 区间为 `[-0.3002343847, +0.2899672010]`；
- 联合裁决：`not-supported`。

该裁决来自预注册规则：`dose_effect` 点估计没有达到支持门槛（不高于 `-2.0` 个百分点且区间上界小于 `0`），并且 `specificity_effect` 点估计为正，满足 `not-supported` 条件。这里的 `not-supported` 只表示在固定可构造位置组的开发期人工 Depth 边界置零实验中，未观察到预注册的边界特异敏感性；它不表示自然故障机制不存在。

### 12.5.4 敏感性与部分可构造组分析

- **全可构造敏感性分析：** 预定义的 123 个“组内每张图均可构造非空 q75”的 location group 均完成配对分析；`dose_effect` 为 `+0.0685432136` 个百分点，95% 区间 `[-0.0798566517, +0.2315669493]`，`123/123` 配对组；`specificity_effect` 为 `+0.0163006238` 个百分点，95% 区间 `[-0.3561380011, +0.2962735383]`，`123/123` 配对组。该分析是预先定义的敏感性分析，不替代 138 组主分析。
- **部分可构造组描述性分析：** 15 个包含至少一张 q75 不可构造图的 location group 保留在范围中；`dose_effect` 为 `+0.1107864683` 个百分点，95% 区间 `[-0.0711737998, +0.3810840876]`，`15/15` 配对组；`specificity_effect` 为 `+0.1643408630` 个百分点，95% 区间 `[-0.0656543536, +0.4910363898]`，`15/15` 配对组。该部分只作描述，不改变主裁决。
- 218 张图中有 31 张图的 `boundary-q75` 实际置零数量为 `0`；这些图仍保留在主范围中，并未被伪装成发生了实际 q75 corruption。
- `06-01-01-0346` 已在 v3 主范围内完成有效 Boundary IoU 计算，不再触发 v2 的标签域阻塞；其进入主分析的 group count 为 `1`。

### 12.5.5 收口与交接边界

v3 完整评价已闭合：定义修正、身份、mask 保护、五条件推理、配对组统计和预注册裁决均有独立证据。当前 A1 的正式结果为 `not-supported`，因此停止把“深度边界失效”作为首要瓶颈，不自动实现完整联合恢复，也不自动把 B 写成已验证。根据用户明确要求，`DVG-B1-oracle-gsa-v1` 已完成基础设计；其代码、GPU、训练和云执行仍需后续细化与单独授权。

本次结果不支持以下扩展结论：

- 不支持自然传感器故障或自然无效深度机制；
- 不支持真实低照/粉尘因果机制或部署安全结论；
- 不支持论文级泛化、跨数据集泛化或多 seed 方差结论；
- 不读取 official test，official test 仍保持 `sealed_unread`。

本轮未进行训练或云资源操作；不追加新的阈值、condition、seed、样本、数据划分或完整联合恢复。下一对话应以本节的 v3 `completed` / `not-supported` 终态为恢复点，优先读取本节记录的运行目录、`summary.json` 和执行记录，不把 v1/v2 的 `protocol-blocked` 历史状态误读为 v3 当前状态。
