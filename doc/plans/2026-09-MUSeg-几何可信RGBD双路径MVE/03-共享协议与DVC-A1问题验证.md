# MUSeg `DVC-A1-valdev-boundary-zero-v1`：共享协议与问题验证

> **文档角色：** 第一个详细子计划；合并共享 protocol、最小实现门禁与 DVC 问题验证，避免拆分同一执行链。
> **计划状态：** 已执行到合法终点 `protocol-blocked`；完整模型评价未开始，后继未解锁。
> **形成或核验时点：** 2026-09-08。
> **实时入口：** [`MUSeg-current-status.md`](../../main/MUSeg-current-status.md)。
> **研究选择：** [`MUSeg-open-decisions.md`](../../main/MUSeg-open-decisions.md)。
> **上级方向：** [`00-总方向规划.md`](./00-总方向规划.md)。
> **前序关系：** [`02-文献补充与协议设计门禁.md`](./02-文献补充与协议设计门禁.md) 已完成。
> **后继关系：** 本计划于 2026-09-08 因 58/196 个 location group 的 q75 不可构造而 `protocol-blocked`；证据见 [`2026-09-08-museg-dvc-a1-protocol-gate.md`](../../reports/2026-09-08-museg-dvc-a1-protocol-gate.md)。[`04-DVG-B1条件式Oracle门控.md`](./04-DVG-B1条件式Oracle门控.md) 未解锁。

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
- `not-supported`：`dose_effect` 点估计大于 `-1.0` 百分点，或 `specificity_effect` 点估计不小于 0。
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

当前仅创建计划，没有执行代码或实验。未来只运行第 7 节列出的最小定点检查；完整测试套件、全仓扫描、多 seed、训练、云端和 official test 均不在本计划默认预算。protocol 物化、preflight、正式评价或恢复点发生变化时，最终答复前更新实时状态文件。
