# DVG-B1 A-1/B-1 检索结果摘要筛选与全文优先级

> **形成时间：** 2026-09-10
>
> **输入材料：** `liu-test-exp/附件/A-1.txt`、`liu-test-exp/附件/B-1.txt`。
>
> **对应问题：** `DVG-B1-必须实现细节靶向检索步骤与WOS检索式.md` 第 6.1 节开放项 A 与第 6.2 节开放项 B。
>
> **证据边界：** 本轮只依据 WOS 导出的题录、关键词和摘要筛选，没有读取论文全文、补充材料或仓库代码，也没有把摘要没有写出的公式视为已经存在。

## 1. 结论

**这批摘要不能直接关闭开放项 A 或 B，但已经把下一步全文获取范围缩小。**

- `A-1.txt` 有 44 条唯一记录，`B-1.txt` 有 30 条唯一记录；两文件只有 `BurnDC` 1 条交集，说明两次检索结果没有发生大面积复制。
- 开放项 A 得到若干直接相关线索：`weighted pooling`、`mask-aware patch-median`、`mean aggregation over valid depths`、`confidence propagation`、稀疏 depth resize 和 coarse-to-fine confidence propagation。
- 开放项 B 得到一个很重要的跨文件线索：`Non-local affinity adaptive acceleration propagation network` 的摘要明确说明把 pixel depth reliability 与 normalized neighbor affinity 结合以调整传播权重；但摘要仍没有给出 query/key 两端组合公式。
- 没有任何摘要同时给出 A 所需的 view-scale resize、四级聚合、部分有效 patch、bilinear 对齐和全可信恒等。
- 没有任何摘要同时给出 B 所需的 Full/H/W gate、两端乘积/最小值/单端选择、对称性、hard/continuous 和全 1 恒等。

大白话说，当前文献列表里已经出现了很接近的问题和方法名，但摘要只能告诉我们“值得看哪几篇”，还不能据此选定 average、valid fraction、乘积或最小值等最终规则。

## 2. 检索结果完整性核对

- `A-1.txt`：44 条记录、44 个唯一 WOS ID、无文件内重复。
- `B-1.txt`：30 条记录、30 个唯一 WOS ID、无文件内重复。
- 两文件唯一交集：
  - `WOS:001772868600015`
  - *BurnDC: A Progressive Propagation Framework for Low Coverage Depth Completion*
  - `A-1.txt:1-28`；`B-1.txt:1-28`。
- 两文件并集为 73 个唯一 WOS ID。

因此，虽然两个平衡检索式都命中了 `BurnDC`，但结果集合总体不同；没有证据表明 A、B 的导出文件再次使用了同一批结果。

## 3. 开放项 A 能回答到什么程度

### 3.1 无效/受损 mask 在层级网络中的下采样规则

**当前判断：不能关闭。**

摘要中最接近的是：

- *Bcap-net* 明确写出 hierarchical multi-scale depth-aware encoder 使用 `weighted pooling` 在不同尺度处理 sparse depth，见 `A-1.txt:149-166`。
- *Confidence Propagation through CNNs for Guided Sparse Depth Regression* 明确写出从 normalized convolution 中确定 confidence 并传播到后续层，见 `A-1.txt:754-780`。
- *LightDepth* 明确把 KITTI sparse depth maps resize 到 31 个 extent，见 `A-1.txt:538-556`。

但摘要没有给出 pooling 权重、有效性 mask 更新公式、stride/downsampling 时 confidence 的递推式或 resize 插值模式，因此不能选择 nearest、area、average、max 或其他规则。

### 3.2 部分有效 patch 的 reliability 语义

**当前判断：只能得到局部支持，不能映射为四级 token reliability。**

- *Integrated RGBD Perception for Clamp-Type Autonomous Forklifts* 使用 `mask-aware patch-median depth`，并用 inward retry 与 vertical-kernel fallback 处理 boundary bleeding 和 missing values，见 `A-1.txt:69-102`。
- *A Unified Evaluation Protocol and Late-Fusion System for Monocular Per-Object Distance Estimation* 使用 `mean aggregation over valid box depths`，见 `A-1.txt:167-184`。

这两条说明“只聚合有效值”以及 median/mean 都是实际使用的区域聚合语义，但它们不是层级 token mask 的直接实现，也没有说明有效比例不足时如何定义 reliability。

### 3.3 与 bilinear Depth resize 的对齐

**当前判断：不能回答。**

`LightDepth` 只说明 sparse depth 被 resize，没有在摘要中写插值方式、mask 是否同步 resize、0/缺失值是否进入插值或像素中心对齐方式。其他候选也没有摘要级 bilinear/nearest/area 对照。

### 3.4 全可信输入在全部尺度保持全可信

**当前判断：不能回答。**

没有摘要声明全 1 validity/confidence 经多级算子后严格保持全 1，也没有与 DVG-B1 原 forward 旁路相同的恒等性验收。

## 4. 开放项 B 能回答到什么程度

### 4.1 局部 reliability 是否可以进入 pairwise affinity 或传播权重

**当前判断：得到直接方向性支持，但不能冻结公式。**

最接近的是 `A-1.txt` 中的：

- `WOS:001026313400003`
- *Non-local affinity adaptive acceleration propagation network for generating dense depth maps from LiDAR*
- `A-1.txt:578-594`

摘要明确写出：网络预测 pixel depth reliability、non-local neighbors、affinities 和 normalization factors，并把 normalized non-local neighbor affinity 与 pixel depth reliability 结合，以自适应调整每个邻居的 propagation weight。

这证明“单点 depth reliability 可以调制成对邻接/affinity 权重”在深度补全文献中确实存在，但摘要没有说明 reliability 属于 query 端、key 端还是两端，也没有给出乘积、最小值或归一化公式。

其他相关线索包括：

- *Deep Sparse Depth Completion Using Multi-Affinity Matrix* 使用邻接像素之间的 multi-affinity matrix，并提到 confidence-map multimodal fusion，见 `A-1.txt:557-577`。
- *Gaussian Splatting Confidence Supervision for SPN-based depth completion* 同时讨论 learned affinities 和 confidence weights，见 `A-1.txt:103-120`。
- *LFDA* 使用 attention score 表示 center/side view 的 similarity 并聚合 cost volume，见 `B-1.txt:373-395`。
- *NR-MVSNet* 使用 reliable attention 结合 attentional reference features 与 cost-volume features，见 `B-1.txt:415-440`。

### 4.2 query/key 两端乘积、最小值、query-only 或 key-only

**当前判断：不能回答。**

没有摘要写出两端 reliability 的组合算子。`DepthRL` 虽然使用模块名 `Symmetric Gated Attention Fusion`，见 `B-1.txt:250-267`，但摘要中的 symmetric 可能表示双向特征融合，不能解释为 query/key gate 对称。

### 4.3 hard gate 或 continuous attenuation

**当前判断：不能冻结。**

多个摘要使用 reliability、confidence、weight、gate 或 reweighting，倾向于连续权重语义，但没有给出取值域、阈值、二值化步骤或明确的 hard/continuous 对照。因此不能仅凭用词选择连续 gate。

### 4.4 全 1 gate 保持原 depth contribution 不变

**当前判断：不能回答。**

`DGQ-YOLO` 使用 zero-initialized residual path，见 `B-1.txt:147-164`，但“残差支路零初始化”不等于“pairwise gate 全 1 时严格恒等”。没有摘要给出 DVG-B1 所需的全 1 gate 恒等性。

### 4.5 是否能直接映射 Full GSA 与 H/W 分解 GSA

**当前判断：不能回答。**

现有摘要没有 DFormerv2 的 `[B,1,L,L]`、`[B,1,W,H,H]`、`[B,1,H,W,W]` 形状，也没有同时覆盖 Full 与轴向分解 attention。即使全文找到一般 pairwise 公式，仍需由项目代码完成 shape-preserving 映射核对。

## 5. 开放项 A：建议获取全文或代码的文献

以下文献按优先级排列。所有条目都需要全文；标注“代码优先”的条目还应直接核对官方实现。

### A-P0：优先获取

#### 1. Confidence Propagation through CNNs for Guided Sparse Depth Regression

- WOS ID：`WOS:000567471300008`
- 来源：`A-1.txt:754-780`
- 用处：直接研究 normalized convolution 中 confidence 的确定和跨层传播，是当前最可能给出确定性 confidence recurrence 的来源。
- 全文必须确认：输入 confidence 的定义、卷积/stride 后的公式、归一化分母、空支持区域、部分有效窗口、全 1 输入行为。
- 代码状态：摘要未提供仓库地址；需要从全文或作者页面继续找官方代码。

#### 2. Bcap-net: a bidirectional cross-fusion and adaptive depth propagation network for depth completion

- WOS ID：`WOS:001778286700014`
- 来源：`A-1.txt:149-166`
- 用处：摘要明确同时出现 hierarchical multi-scale、weighted pooling 和 depth confidence propagation。
- 全文必须确认：weighted pooling 的权重、有效值归一化、每尺度 mask/confidence shape、部分有效 patch、空 patch 和插值对齐。
- 代码状态：摘要未提供仓库地址。

#### 3. LightDepth: A resource efficient depth estimation approach for dealing with ground truth sparsity via curriculum learning

- WOS ID：`WOS:001307644900001`
- 来源：`A-1.txt:538-556`
- 用处：摘要明确对 sparse ground-truth depth maps 执行多 extent resize，最接近 A 中的 view-scale mask/depth resize 问题。
- 全文必须确认：resize 插值模式、validity mask 是否同步变换、零值是否参与插值、像素中心约定、是否改变有效像素比例。
- 代码状态：摘要给出 <https://github.com/fatemehkarimii/lightdepth>，应优先核对代码而不只读方法描述。

#### 4. Image-guided dense depth completion network based on hierarchical feature reconstruction and dynamic weight sampling

- WOS ID：`WOS:001565304100002`
- 来源：`A-1.txt:247-266`
- 用处：层级重建会生成 depth confidence maps，随后使用 dynamic weight sampling 和 confidence gating。
- 全文必须确认：confidence map 的来源、各级 resize/downsample、是否由 validity 统计得到、动态采样是否为归一化加权平均。
- 代码状态：摘要未提供仓库地址。

#### 5. Octagram Propagation Matching for Multi-Scale View Stereopsis

- WOS ID：`WOS:001492121500010`
- 来源：`A-1.txt:316-349`
- 用处：摘要明确描述 multi-scale depth confidence-guided geometric consistency，以及 reliable depth estimates 从 coarse scale 向 finer scale 传播。
- 全文必须确认：confidence 定义、尺度间重采样、聚合算子、有效性阈值和边界处的处理。
- 代码状态：摘要未提供仓库地址。

### A-P1：有直接局部语义，作为补充依据

#### 6. Integrated RGBD Perception for Clamp-Type Autonomous Forklifts

- WOS ID：`WOS:001786002300042`
- 来源：`A-1.txt:69-102`
- 用处：mask-aware patch-median、missing-value fallback 和 boundary bleeding 处理可用于核对“只聚合有效深度”的语义。
- 全文必须确认：valid mask、median 的有效样本集合、样本不足阈值、retry/fallback 的精确顺序。

#### 7. A Unified Evaluation Protocol and Late-Fusion System for Monocular Per-Object Distance Estimation

- WOS ID：`WOS:001860042700001`
- 来源：`A-1.txt:167-184`
- 用处：明确使用 valid-only mean aggregation，可作为有效值平均的实际应用证据。
- 全文必须确认：valid depth 的定义和有效比例不足时的处理。
- 边界：ROI 距离统计不是多级 token reliability，只能作为局部补充。

#### 8. Guided Depth Inpainting in ToF Image Sensing Based on Near Infrared Information

- WOS ID：`WOS:001420327300001`
- 来源：`A-1.txt:447-476`
- 用处：明确处理 invalid pixels、missing depth values 和 valid nearby information。
- 全文必须确认：invalid mask 编码、belief propagation 公式、边缘邻域和空区域处理。

#### 9. Coplane-constrained sparse depth sampling and local depth propagation for depth estimation

- WOS ID：`WOS:001297124400001`
- 来源：`A-1.txt:477-494`
- 用处：明确区分 valid points 和 low-confidence pixels，并进行局部几何传播。
- 全文必须确认：valid/low-confidence 定义、阈值、传播权重及尺度关系。

#### 10. An Efficient Information-Reinforced Lidar Deep Completion Network without RGB Guided

- WOS ID：`WOS:000868001300001`
- 来源：`A-1.txt:638-655`
- 用处：摘要出现 multi-resolution progressive fusion 和 confidence re-aggregation。
- 全文必须确认：re-aggregation 的算子、归一化、空支持和尺度转换。

### A-P2：方向相关，但优先级低于上述文献

#### 11. BurnDC: A Progressive Propagation Framework for Low Coverage Depth Completion

- WOS ID：`WOS:001772868600015`
- 来源：`A-1.txt:1-28`
- 用处：reliable depth anchors、progressive frontier 和 weighted ring attention 可能包含局部可靠性传播规则。
- 全文必须确认：anchor reliability、ring 支持域及其权重更新。

#### 12. Gaussian Splatting Confidence Supervision for SPN-based depth completion

- WOS ID：`WOS:001826397300001`
- 来源：`A-1.txt:103-120`
- 用处：同时涉及 affinity、confidence weights 和 local relation aggregation。
- 全文必须确认：confidence 是输入 validity、预测 uncertainty 还是训练监督量，以及它是否跨尺度传播。

#### 13. AISPO: Enhancing Depth Reliability for Robotic Manipulation of Non-Lambertian Objects

- WOS ID：`WOS:001772868600012`
- 来源：`A-1.txt:121-148`
- 用处：corrupted/missing raw depth 与 multi-scale RGB-D fusion 的场景相符。
- 全文必须确认：是否真的有显式 mask/reliability 传播；若只有端到端补全，则不能关闭 A。

#### 14. Deep Sparse Depth Completion Using Multi-Affinity Matrix

- WOS ID：`WOS:001042003300001`
- 来源：`A-1.txt:557-577`
- 用处：多阶段 confidence-map fusion 和 multi-affinity 更适合作为 A/B 交叉候选。
- 全文必须确认：confidence 是否从 validity 聚合而来，以及各阶段是否有显式 resize/downsample。

#### 15. Non-local affinity adaptive acceleration propagation network

- WOS ID：`WOS:001026313400003`
- 来源：`A-1.txt:578-594`
- 用处：pixel depth reliability 与 neighbor affinity 的结合对 B 更直接，对 A 可核对 reliability 的生成和尺度。
- 全文必须确认：初始 reliability 的来源、是否有多尺度传播以及全可信行为。

## 6. 开放项 B：建议获取全文或代码的文献

B 的优先级综合了 `B-1.txt` 和 `A-1.txt` 中意外命中的 affinity/confidence 文献。

### B-P0：最接近 pairwise reliability/affinity

#### 1. Non-local affinity adaptive acceleration propagation network for generating dense depth maps from LiDAR

- WOS ID：`WOS:001026313400003`
- 来源：`A-1.txt:578-594`
- 用处：本批摘要中唯一明确写出“pixel depth reliability 与 normalized neighbor affinity 结合并调整每个邻居传播权重”的文献。
- 全文必须确认：pairwise weight 的公式；reliability 属于中心点、邻居点还是两端；是否对称；归一化顺序；reliability 取值域；全可信时是否还原原 affinity。

#### 2. Deep Sparse Depth Completion Using Multi-Affinity Matrix

- WOS ID：`WOS:001042003300001`
- 来源：`A-1.txt:557-577`
- 用处：明确以 multi-affinity matrix 表示输出像素与邻居的关系，并使用 confidence maps 做多模态融合。
- 全文必须确认：confidence 是否直接进入 affinity；邻接矩阵是否有方向性；是否按中心/邻居双端组合；是否连续门控。

#### 3. Gaussian Splatting Confidence Supervision for SPN-based depth completion

- WOS ID：`WOS:001826397300001`
- 来源：`A-1.txt:103-120`
- 用处：同一 SPN 框架中同时出现 learned affinities、confidence weights 和 local relation aggregation。
- 全文必须确认：confidence weight 在 propagation 方程中的确切位置；它门控节点更新、边权还是最终融合；是否与 affinity 相乘。

#### 4. NR-MVSNet: Learning Multi-View Stereo Based on Normal Consistency and Depth Refinement

- WOS ID：`WOS:000988473800002`
- 来源：`B-1.txt:415-440`
- 用处：reliable attention 与 cost-volume features 结合，是 B-1 中最接近 pairwise matching/affinity 的候选。
- 全文必须确认：reliability 是否直接修改 cost-volume score/attention logits；是否由 query/key 两端产生；是否对称。
- 代码状态：摘要给出 <https://github.com/wdkyh/NR-MVSNet>，应优先核对实现。

#### 5. LFDA: A Framework for Light Field Depth Estimation With Depth Attention

- WOS ID：`WOS:001219298900001`
- 来源：`B-1.txt:373-395`
- 用处：明确沿 epipolar line 比较 center/side view，并根据 attention-score similarity 聚合 cost volume。
- 全文必须确认：attention score 的公式、方向性、对称性，以及是否存在独立 reliability gate。
- 代码状态：摘要给出 <https://github.com/syt06007/LFDA>，应优先核对实现。

#### 6. Learning Selective Mutual Attention and Contrast for RGB-D Saliency Detection

- WOS ID：`WOS:000880661400035`
- 来源：`B-1.txt:533-555`
- 用处：non-local mutual attention 提供高阶跨模态交互，并使用 selective attention 重加权 added depth cues。
- 全文必须确认：reweight 位于 feature、attention score、non-local affinity 还是输出；是否按 pairwise depth cue 调制。

### B-P1：可靠性/对称性/gate 的补充候选

#### 7. CMNC-Net: A Cross-Modal Dual-Branch Network With Noise Awareness for Depth Completion

- WOS ID：`WOS:001772873400016`
- 来源：`B-1.txt:80-109`
- 用处：reliability-aware noise prior 与 residual attention refinement。
- 全文必须确认：reliability guidance 是否进入 attention score；若只校准 feature response，则不能关闭 B。

#### 8. DGQ-YOLO: Depth-Guided Quality-Aware detection with Pseudo-Depth

- WOS ID：`WOS:001858702300020`
- 来源：`B-1.txt:147-164`
- 用处：depth-conditioned spatial/channel gates、zero-initialized residual path 和 gate-intervention 实验可用于核对可靠性调制与 no-op 设计。
- 全文必须确认：gate 是 feature-level 还是 pairwise score-level；零初始化如何保证基线行为。
- 边界：摘要更像 feature gate，不能直接作为 GSA pairwise gate。

#### 9. DepthRL: a weakly supervised approach for monocular depth estimation using deep reinforcement learning

- WOS ID：`WOS:001510573200007`
- 来源：`B-1.txt:250-267`
- 用处：模块名为 Symmetric Gated Attention Fusion。
- 全文必须确认：symmetric 的数学含义；是否是双向 feature fusion，还是 query/key pairwise symmetry。

#### 10. DGFNet: Depth-Guided Cross-Modality Fusion Network for RGB-D Salient Object Detection

- WOS ID：`WOS:001173299400015`
- 来源：`B-1.txt:269-292`
- 用处：depth attention 逐层引导 RGB-depth fusion。
- 全文必须确认：attention 是空间权重、feature 权重还是 pairwise affinity。

#### 11. Depth-Constrained Network for Multi-Scale Object Detection

- WOS ID：`WOS:001059770900002`
- 来源：`B-1.txt:397-413`
- 用处：depth awareness 调整 attention weight preferences。
- 全文必须确认：attention weight 的定义和 depth contribution 的作用位置。

#### 12. Bcap-net

- WOS ID：`WOS:001778286700014`
- 来源：`A-1.txt:149-166`
- 用处：depth confidence propagation refiner 会根据 hidden features 和 depth differences 调整 depth update weights。
- 全文必须确认：更新权重是否为邻接 pairwise 权重、是否包含中心/邻居 reliability，以及是否归一化。

## 7. 不建议优先获取全文的结果类型

以下类型即使标题包含 depth、attention、gate 或 reliability，也不能直接回答当前问题，除非后续全文候选全部失败：

- 仅做整模态或整分支 reweighting，例如 sample-wise RGB/Depth/IR modality weighting；
- 仅做 channel/spatial feature attention，不修改 pairwise score、bias 或 affinity；
- 仅报告一般 depth completion、depth estimation 或 propagation 性能，没有 validity/confidence 公式；
- causal、padding、window 或可见性 mask，与 depth corruption reliability 无关；
- 医学图像、海洋测深和一般检测中只因关键词偶然出现 depth/downsampling 的记录。

## 8. 推荐全文获取顺序

若只能先获取少量全文，建议按以下顺序：

1. *Confidence Propagation through CNNs for Guided Sparse Depth Regression*——优先关闭 A 的 confidence propagation 公式。
2. *Bcap-net*——优先核对 multi-scale weighted pooling。
3. *Non-local affinity adaptive acceleration propagation network*——优先关闭 B 的 reliability-affinity 组合。
4. *Deep Sparse Depth Completion Using Multi-Affinity Matrix*——核对 confidence 与 pairwise affinity 的关系。
5. *LightDepth* 及其代码——核对 sparse depth resize 与 validity 同步。
6. *NR-MVSNet* 及其代码——核对 reliable attention 与 cost volume。
7. *LFDA* 及其代码——核对 pairwise attention score。
8. *Image-guided dense depth completion network based on hierarchical feature reconstruction and dynamic weight sampling*——核对多级 confidence map。
9. *Gaussian Splatting Confidence Supervision for SPN-based depth completion*——核对 confidence weight 在 SPN 方程中的位置。
10. *Learning Selective Mutual Attention and Contrast for RGB-D Saliency Detection*——核对 depth cue reweight 是否作用于 non-local affinity。

## 9. 准确恢复点

1. 获取上述 P0 文献的全文、补充材料和官方代码；优先利用摘要已经给出的 `LightDepth`、`NR-MVSNet`、`LFDA` 仓库。
2. 对 A 逐篇提取：输入 validity/confidence 定义、resize/downsample 算子、部分有效窗口公式、空窗口规则、全 1 行为和代码位置。
3. 对 B 逐篇提取：中心/邻居或 query/key 两端的 reliability 来源、pairwise 组合式、归一化顺序、方向性/对称性、hard/continuous、全可信恒等和代码位置。
4. 只有得到唯一、可复现并能映射到 DFormerv2 四级 shape 的规则后，才回填原检索文档和 `04-DVG-B1条件式Oracle门控.md`。
5. 当前 A、B 继续保持 `reference-blocked`；本轮不创建 protocol，不修改模型或 evaluator，不运行 preflight、GPU、训练、云资源或 official test。
