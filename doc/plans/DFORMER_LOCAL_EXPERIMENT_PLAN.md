# DFormer 方向 A：第一阶段本地实验落地方案

> 日期：2026-10-13；版本：**v0.2 — 科研计划已通过云端审核，附第13节执行模型实施任务单**。
> 依据：此前完整阅读 [科研总方案 v0.1](DFORMER_RESEARCH_MASTER_PLAN.md)并完成源码/资产核实；用户已确认v0.2通过科研审核。本轮再次定点读取当前几何先验、W/H/full attention及评价/RNG helper，细化第一阶段技术接口，不重新设计方向。源码基线仍为 `3a982fe1a0739f93f735c1afc9e0965a059a7403`。
> 本轮只编制文档任务单，并在项目状态记录审核通过与交接恢复点；不编写模型代码、不运行测试/推理/训练、不启动云端。第13节的接口与命令是执行模型需要实现的目标，当前尚不存在；科学审核通过不等同于本轮启动GPU。

## 1. 结论、科研目标与最短推进路径

**以尽快完成一篇有竞争力的中科院二区论文为目标，优先发现真实、可解释的有效机制，再迅速推进模型适配和论文证据。** 分区依投稿年度与认定口径，目标不构成录用或等级保证。坚持问题明确、数学动机自然、模块简洁；不以繁琐验收或人为百分点门槛替代科研判断。

方向 A 的核心数学自洽，当前 DFormerv2-S 可按原定义接入。SUN R2 Original epoch30 的完整分割权重、固定 dev 划分和评价工程是最经济的起点。方向 B 保留，不同时展开；不新增 Depth 编码器、可靠性预测器、教师或辅助损失。

保留核心：相对于当前块的 spatial-only 参考分布，对完整几何注意力求解带双向密度比限制的 `KL(p || pD)` 最小投影。投影必须发生在 attention 归一化/聚合处，不能用普通 gate 或 `clip(Bd)` 冒充。

最短路线：**小规模算子核验 → 32 图诊断 → 正向候选的一次同32图排伪影复核 → 剩余496图确认 → 必要时三组10epoch短训 → 科研裁决。**

1. 后续获批实施时，完成最小投影算子与配对评价薄入口，不重复已验收工程、不建立新测试体系。
2. 固定32图，比较 clean、75%人工孔洞和16像素局部循环位移；KL投影、有界clip、逐行解析有界gate采用相同候选预算。
3. 循环位移有正向信号时，在大样本或训练前完成一次非循环、非零有效区域平移复核；通过后用剩余496图确认。筛选集与确认集分开报告，必要时合并已有结果，不重复推理。
4. 稳定的小幅正向趋势、部分几何损害缓解及合理独立机制信号均可支持申请一次短训；不要求冻结诊断先达到1pp，也不把冻结结果当成联合训练上限。结果已经清楚时减少冗余探针和重复评价，小结果合理时优先短训，不长期堆积诊断。
5. 模块一旦得到有说服力的正向结果，立即同步深化数学、结构表达和论文叙事，按第10节规划完整证据。第二数据集等后续扩展仍需另行批准。

已核实的 W/H 共享 Q/K 性质和简单方法也可满足倍率界的事实继续保留。A的潜在独立价值是可行域内的KL最小改变及其任务作用，不是独占硬约束。主要工程风险仍是反向梯度与早层开销；本轮仅修订计划，完成后停止。

## 2. 已核实的源码与真实数据流

以下 stage1–4 对应代码 `layers[0]`–`layers[3]`。所有源码行号均为当前文件的 1-based 行号。

### 2.1 模型与几何入口

- `models/encoders/DFormerv2.py`：`DFormerv2_S`（670–678）实际通道 `[64,128,256,512]`、块数 `[3,4,18,4]`、heads `[4,4,8,16]`；构造函数（509–577）设置前三 stage 为轴向，最后为 full。共 29 个 `GeoPriorGen.weight`，每块独立两个 signed 系数。
- `dformerv2.forward`（640–657）：辅助输入只取 `x_e[:,0:1]`，RGB 经 patch embedding 构成唯一主特征流；`original` 不构造 `ObservationGeometry`。Depth 不形成独立语义分支。
- `GeoPriorGen.forward`（173–217）：原始深度先按当前网格双线性插值，`align_corners=False`，再求绝对深度差；分别乘 head 衰减和 `weight[1]`。空间 L1 距离乘同一衰减和 `weight[0]`，最终已合并成 `mask_h/mask_w` 或 `mask`。
- `RGBD_Block.forward`（426–437）：先加卷积位置编码，再生成该块几何先验，对 LayerNorm 后的 RGB 特征进行 attention，最后残差与 FFN。
- `BasicLayer.forward`（491–505）：ODG/mean 的未加权关系可同 stage 共享，但每块权重不同；Original 仍走每块原生成器。本轮 A 固定 `geometry_mode=original`，不叠加 ODG。
- `research/geometry.py::ObservationGeometry`（52–108）是此前观测分布方法，不是 A 的必要依赖；可参考合法输入检查，但不能把其删除 mask 中性化引入 A 后仍称单独验证投影。

实际先验记为：

$$
\ell_h=\log(1-2^{-a-rh/N}),\quad B^s_{ij,h}=w_s\ell_h\|u_i-u_j\|_1,\quad B^d_{ij,h}=w_d\ell_h|\widetilde d_i-\widetilde d_j|.
$$

`w_s/w_d` 均没有正值约束。既有 R1 诊断证明预训练存在负 `w_d`；**不能把 R1 的均值当成 R2 当前值**。R2 的 29 组实际系数与 attention 倍率尚需下一轮加载权重后测量，本轮不造数。

### 2.2 两轴依赖：已经核实，不再保留接口猜测

`Decomposed_GSA.forward`（236–279）：

1. Q、K、V 与 LePE 全部从同一输入 `x` 生成；K 先乘缩放，Q/K 再做 RoPE。
2. W logits 为 `[B,H,heads,W,W]`；几何 `mask_w` 转置后加到 QK，softmax，先聚合 V。
3. H logits 为 `[B,W,heads,H,H]`；H 的 Q/K 是上述同一 Q/K 的另一次维度置换，**没有从 W 输出重新生成**。H 仅接收 W 聚合后的 V。
4. 得到 H 聚合输出，再加原 LePE、经过 `out_proj`。两轴之间没有重新投影 Q/K、残差或 FFN。

关键证据：

```253:266:models/encoders/DFormerv2.py
        qr_w = qr.transpose(1, 2)
        kr_w = kr.transpose(1, 2)
        v = v.reshape(bsz, h, w, self.num_heads, -1).permute(0, 1, 3, 2, 4)

        qk_mat_w = qr_w @ kr_w.transpose(-1, -2)
        bias_w = mask_w.transpose(1, 2)
        if getattr(self, "match_bias_dtype", False):
            bias_w = bias_w.to(qk_mat_w.dtype)
        qk_mat_w = qk_mat_w + bias_w
        qk_mat_w = torch.softmax(qk_mat_w, -1)
        v = torch.matmul(qk_mat_w, v)

        qr_h = qr.permute(0, 3, 1, 2, 4)
        kr_h = kr.permute(0, 3, 1, 2, 4)
```

因此，对**同一块输入、同一 head、固定 V**，可以严格比较 `A_H* A_W*` 与 `A_H0 A_W0`：每条路径倍率及其端点边缘化倍率在 `exp(±(tau_H+tau_W))` 内。LePE 对配对输入相同，不影响聚合差；输出线性层会乘其算子范数。此性质仍不跨越块间残差、FFN、下采样，不是端到端输出/精度保证。H 轴的单次聚合界若要检验，必须使用相同输入 V；不能拿 W 已改变的 V 和参考 V 混比。

### 2.3 Full 与 mask

- `Full_GSA.forward`（307–337）：RoPE 后缩放 QK 的形状 `[B,16,HW,HW]`，加合并先验后 softmax，再与 V 相乘。
- 当前 GSA **没有结构性 key mask**。`depth_support` 不等于 attention 有效键集合；Original 几何路径不使用它来排除孔洞或 padding。
- 首版保持原 key 集合，包括 padding token；GT=255 仅在指标中排除。人为删除区域只用于构造共同输入与诊断区域，不能作为 A 的推理侧可靠性标签。
- Original 模式下 `match_bias_dtype=False`；此前 ODG 会转换偏置 dtype。A 不借切换 ODG 来接入，也不改变原权重符号。

### 2.4 数据与评价

- `research/data.py::ObservationDataset.__getitem__`（229–288）：SUN/v2 读 RGB；深度用 `IMREAD_GRAYSCALE` 读取，复制为三通道。原 uint16 PNG 的该读法涉及高 8 位，不是米制 uint16/255。
- `ObservationValPre`（399–441）：原数组右/下 padding 到 **531×730**，GT 填 255、support 填 0，再归一化；Depth 为 `(D8/255-0.48)/0.28`。天然 raw0 保留为观测；padding/raw0 对应归一化值约 −1.714286，normalized0 又是另一数值。
- `models/builder.py::EncoderDecoder.encode_decode`（255–274）：HAM 使用 stage2–4，decoder 结果双线性恢复输入尺寸。**研究层单尺度评价直接整张 531×730 前向，不是 480×480 滑窗**；`eval_crop_size` 不控制这一入口。
- `research/evaluate_odg.py::main`（85–214）：独立配置和 checkpoint 加载，显式恢复 decoder BN eps/momentum，`model.eval()`，调用研究评价器。用 `ODG_SUNRGBD` 的 dev 合同，但明确覆盖为 `original`；不能直接用默认指向官方 test 的 `DFormerv2_S_SUNRGBD` 配置。
- `research/odg_schedule.py::evaluate_loader`（850–923）：单尺度无 flip 时直接 `forward_logits`，logits 的类别 softmax 在 FP32，按整图累计 confusion matrix。
- `research/train_odg.py::run_validation`（167–205）：eval 前后校验 BN running 状态并恢复训练。后续沿用该保护，不重做原有 smoke。

### 2.5 HAM 的配对随机性

`models/decoders/ham_head.py::_MatrixDecomposition2DBase`（12–101）默认 `RAND_INIT=True`，eval 也重新生成 NMF 基；`NMF2D._build_bases`（110–117）实际先在 **CPU** 调用 `torch.rand`，再移到输入设备。只固定 CUDA seed 不够。

复用 `research/odg_schedule.py::capture_rng/restore_rng`（325–347）：每张图生成/保存一次前向 RNG 状态，各条件、各方法前恢复同一 CPU/CUDA 状态。扰动用独立 numpy RNG，不消耗 HAM 的 torch RNG；方法执行不额外抽随机数。保持作者随机算法，不改成全局固定 basis，不增加论文外模块。

## 3. 可复用资产、事实与仍未知项

**主诊断权重：** `outputs/sun-dev-original-seed12345-r2/20261010-094231/last.pth`。使用完整模型权重严格加载，不仅加载 encoder；R1 和 ODG checkpoint 都不是本轮主初始化。

初版核实已确认该 checkpoint 的文件级存在性，并直接回读该 run 的 `contract.json`、`resolved_config.json`、`validation.csv` 以及 `outputs/odg-r2-eval/original-clean/eval_result.json`，对应：SUN37类、DFormerv2-S/HAM1024、有效 batch16、4×4、seed12345、4757/528 划分、eval 验证、714/714 参数范围。历史 checkpoint 的字段加载核验来自 R2 正式报告；初版设计和本次修订均未重新反序列化权重，本次不重复资产检查。

- 官方 encoder 初始化资产：`checkpoints/pretrained/DFormerv2_Small_pretrained.pth`；本轮不要求重新下载。只有从官方 encoder 重开匹配训练时才用它。
- 数据：`D:\0Project\dataset\SUNRGBD`；dev 清单 `research/splits/sunrgbd_seed12345/dev.txt`、训练清单 `train-dev.txt`。已直接读取 dev/官方 train 清单头部；三模态 10335 和配对闭环依据既有准备报告，不重复全量盘点。
- R2 历史同入口 clean mIoU **45.34**；75% 孔洞为 **42.49/42.64**（两个扰动 seed）。这是已有结果，不是本轮复现，也不是新配对 HAM 协议的硬验收目标。
- 孔洞推理历史每组 528 图约 63–73 秒；clean 原产物 `seconds=116.51`。计时会受进程启动/缓存/协议影响，采用 **1–2 分钟/原始全 dev 前向**作为粗估，不能外推成投影已测速度。
- NYUv2 尚无现成实验条件；DeLiVER 无正式25类训练权重；旧 MUSeg 不迁回。这些不阻塞首轮 SUN 诊断。

下一轮需实测而不能从文档认定：R2 `w_d` 分布、实际 `log(pD/p0)`、坏关系与区外错误量、投影数值/吞吐、当前显存余量；梯度核验留到研究价值获支持后的独立训练任务。天然深度的物理单位和 raw0 语义仍未知，因此本轮只作受控输入扰动研究。

## 4. 方向 A 的数学与可实现求解

### 4.1 保留的定义与保证

每个 stage/block/head/query 行定义：

$$
S=QK/\sqrt{d_h}+B^s,\quad p^0=\operatorname{softmax}(S),\quad p^D=\operatorname{softmax}(S+B^d).
$$

$$
p^*=\arg\min_{p\in\Delta} D_{KL}(p\|p^D),\qquad e^{-\tau}p_j^0\le p_j\le e^{\tau}p_j^0.
$$

`p0` 只去掉当前块新加的 Depth 项，是相同特征的局部参考，不能叫独立 RGB-only 模型。有限 logits 下 `p0/pD` 严格正；约束集包含 `p0`，目标严格凸，分布解唯一：

$$
p_j^*=\operatorname{clip}(c p_j^D,e^{-\tau}p_j^0,e^\tau p_j^0),\quad \sum_jp_j^*=1.
$$

云端 TV 界正确（下式分式取 $\tau>0$；$\tau=0$ 时由端点/连续极限得到0）：

$$
\operatorname{TV}(p^*,p^0)\le\frac{(e^\tau-1)(1-e^{-\tau})}{e^\tau-e^{-\tau}}=\tanh(\tau/2).
$$

它与固定 V 的 diameter 界都是当前算子的直接作用界，不保证预测正确、损坏后更接近 clean，也不保证上游已污染特征被恢复。

### 4.2 首版使用 log-domain 批量一维求根

建议实现在 `research/influence_projection.py`，不是改写 `research/geometry.py`。用 FP32 `log_softmax` 保存 `l0=log p0`。数值实现先令 `b=Bd-max(Bd)`，再计算 `r=b-logsumexp(l0+b)`、`lD=l0+r`，避免两个极负log概率直接相减丢失小倍率；数学上仍为 `r=log(pD/p0)`。令 `z=log c`：

$$
\log p_j^*=l_j^0+\operatorname{clamp}(z+r_j,-\tau,\tau),\qquad f(z)=\sum_j\exp(\log p_j^*)-1.
$$

根有确定括区：

$$
z_{lo}=-\max_j r_j,\quad z_{hi}=-\min_j r_j,\quad f(z_{lo})\le0\le f(z_{hi}).
$$

在 GPU 上批量二分，无逐行 Python 求解或逐迭代 `.item()` 同步。首版固定 24 次；仅小张量对 FP64 参考核验，真实图按行分块。若 24 次未达容差，先定位数值问题/括区大小，不能通过事后再归一化掩盖约束错误。

特殊路径：`tau=0` 直接返回 `p0`；原行已满足界时直接返回 `pD`；禁用模式直接保留作者原路径，不用巨大有限 tau 假装恒等。NaN/Inf 输入拒绝并报告，不偷偷替换深度编码。首版没有结构性 −inf mask；以后若增加 mask，须对共同有效键另立协议。

数值验收目标：FP32 每行概率和误差 ≤ `1e-6`，log-domain 双向倍率超界 ≤ `5e-6`；无 NaN/Inf。检查 `logp* - l0`，**不对 underflow 后的两个零概率做除法**；不先 clamp `p0` 到 epsilon 改变参考。FP32 概率尾部仍可能 underflow，这是有限精度实现与理想数学的差异，记录零概率质量影响，不宣称浮点逐项绝对严格。

手算反例/验收例：`p0=(0.5,0.3,0.2)`、`pD=(0.8,0.1,0.1)`、`tau=ln2`。真正投影为 `(0.75,0.15,0.10)`、`c=0.9375`；先逐项裁成 `(0.8,0.15,0.1)` 再除以 1.05，会重新低于第2/3项下界。该例可直接用于未来算子的少量内置自检，不新建临时测试脚本/全套测试。

### 4.3 反向传播不是“二分代码能 backward”就成立

**第一阶段不实现反向传播、不进行梯度核验。本节仅是后续训练阶段的技术参考。** 冻结诊断用no-grad前向即可；获得研究价值后另行安排反向实现及最小梯度核验。二分中的比较/活跃集选择离散，直接让 autograd 穿过有限二分，或把 `z` detach 后仍将结果当成精确梯度，都不可靠。

建议在 no-grad 下确定自由集合 F、上下界集合 L/U，再对固定活跃集使用可微等价式或隐式反向：

$$
M=1-\sum_{j\in L}e^{-\tau}p_j^0-\sum_{j\in U}e^\tau p_j^0,\quad c=\frac{M}{\sum_{j\in F}p_j^D}.
$$

自由项 `p*=c pD`，其余使用对应边界；保留 `p0/pD` 两条分支梯度，不能 detach 参考以便宜地改变训练机制。分母用 logsumexp 稳定计算；活跃集边界用一致分段导数。全饱和/极小自由质量单独处理，不能用 epsilon 归一化混过去；若首版无法稳定处理，停止短训，先回报工程阻碍。

训练前仅做：远离切换面的几条 FP64 小概率行有限差分/向量积核对，以及一个真实 block 的有限梯度检查；关注 Q/K、`Geo.weight[1]` 和 V，不重跑714参数旧工程测试。完全饱和时 Depth 梯度为零可以是机制结果，不是必须修 warning 的理由。具体反向实现待实施审核，不以本节推导冒称验收通过。

### 4.4 精度与聚合协议

倍率界定义于最终用于聚合的概率。若在 FP32 投影后又由 autocast 将概率转成 FP16，实际聚合权重可能失去该界。因此，首轮各配对方法在 GSA 内统一使用 **FP32 softmax/投影及 FP32 `p @ V`**，再将子模块输出转回原输出 dtype；QK 及其余模型仍复用 AMP。H 的 V 来自对应方法 W 输出。

这属于共同数值协议，不是 A 独占精度优势。复用32图中的前4张，在首次前向比较legacy Original与共同FP32 Original，不另设重复验收阶段；若差异影响判读，明确以共同精度重建基线，不与历史45.34直接归因比较。禁用新机制仍保持legacy分支原样，不要求两套精度bitwise相等。

## 5. 工程改动边界与成本

### 5.1 需要新增/修改的最小职责

以下均为**拟实施**，当前没有这些能力或新 CLI。

- `research/influence_projection.py`：第一阶段仅实现冻结前向投影、clip/gate与少量内置数值核验；活跃集反向留到研究信号支持后的独立训练任务，不提前开发。
- `models/encoders/DFormerv2.py::GeoPriorGen.forward`：增加默认关闭的 components 返回方式，提供已加权 `Bs/Bd`，保持现有 rel_pos 返回及原相加顺序。不能把 `weight` 改名或重新注册参数。
- `RGBD_Block.forward`：只在研究机制启用时传递 components；`Decomposed_GSA.forward/Full_GSA.forward`：在 W/H/full softmax位置调用研究算子，保留 RoPE、空间项、V、LePE、输出投影、残差和 FFN。A是独立 `attention_control`，不是新 `geometry_mode`。
- `research/evaluate_influence.py`：薄配对入口，复用 loader、BN恢复、`SegMetrics`、RNG helper；增加固定对应扰动、区域指标、块探针与一次加载多方法配对。原 `evaluate_odg.py` 不改科研行为。
- `local_configs/research/INFLUENCE_SUNRGBD.py`：拷贝现有 dev 配置且固定 `original`，预算与作用范围显式记录。必要的模型属性在构建后由研究入口安装，避免扩展多个公共 builder 接口。
- 第二步才增加 `research/train_influence.py` 的薄训练适配：共同完整权重初始化、同腐蚀增强、短日程和额外合同字段；优先复用已有训练循环，不能整份复制形成第二套训练系统。旧 ODG resume 合同保持原样。

投影没有新可学习参数，原模型 state_dict key/shape 可保持一致，完整 R2 权重可严格复用。方法、tau、stage/axis范围与初始化身份必须进入新合同，防止改模式后误 resume。

### 5.2 注意力规模与内存预估

训练输入480×480，网格依次120²、60²、30²、15²。单块、batch1的两轴注意力元素量为 `heads×H×W×(H+W)`，full为 `heads×(HW)^2`：

- stage1 约1382万元素，一份FP32概率约52.7 MiB；stage2约6.6 MiB；stage3约1.65 MiB；stage4约3.1 MiB。
- 整图评价531×730时网格133×183、67×92、34×46、17×23；stage1两轴一份约117.4 MiB，**full不是首轮最大概率图**。
- 若训练保存29块的多份概率及24轮二分计算图，额外开销会远大于“参数为零”。此处是张量规模估算，不是CUDA峰值。

首版对query维分块（候选16/32 query，整条key保留）；统计在线累计，不保存全dataset attention。必要时QK也按query算。冻结诊断仅维护当前块工作区；反向用隐式/活跃集实现，避免保存24轮搜索图。投影不能跨图或跨块缓存，`p0`依赖当前QK。

先在所有29块的W/H/full上使用同一tau，避免仅低分辨率插入漏掉早层传播；**固定输入因果探针只取stage1第1块和stage3第1块**。全网成本若不可接受，再向云端提出限定stage1三块的诊断变体；不静默降为late-only，并禁止用该变体阴性结果否定全定义。

## 6. 第一轮冻结权重诊断：完整协议

### 6.1 假设与预期现象

- H1：对应扰动在部分行改变已加权Depth作用，并产生GT有效、损坏区外的新增错误；固定输入仅替换Bd的块探针仍存在明显远端聚合变化。
- H2：有限预算能修复其中一部分错误，且正常输入损失显著小于全网Depth-off，说明保留有益几何有价值。
- H3：在同作用界下，KL最小投影比简单裁剪/缩放具有额外任务收益。**投影满足约束本身只验证算子，不验证H1–H3。**

预期是损害、超限和修复之间存在配对规律，不预设“倍率越大必然越错”。Depth本来帮助纠正RGB关系的行也可能超限；这类反例必须保留。

### 6.2 固定样本、扰动与一次排伪影复核

筛选图：按固定dev清单原顺序等间隔取32图，索引 `floor(k×528/32)`，`k=0,...,31`；不按预测改善挑图。沿用batch1、workers2、eval、531×730 padding、单尺度无flip、37类、GT ignore255。

每张图初筛三种条件：

1. **Clean**：未经新人工损坏的Depth，仍可有天然raw0。
2. **Holes75**：复用 `apply_artificial_holes`（`odg_schedule.py`756–809），ratio0.75、seed12345；按真实支持面积精确取整，输入填raw0归一化值。A与对照均不用删除mask中性化关系。子集使用该图在完整dev中的原始索引作为`start_index`，避免子集重编号改变孔洞。
3. **Correspondence16-Cyclic**：原图支持矩形中心内取最大不超过192×192的操作矩形 $C_{op}$；Depth三通道共同向右循环位移16像素（宽≤32的小图记录跳过）。RGB/GT/padding/support不变，不插值、不新填raw0，矩形内直方图、raw0数量和可观测像素数不变。

循环位移是廉价机制初筛，不是真实传感器平移：回绕接缝、矩形边缘及raw0位置移动可能影响结果。报告实际深度变化率、非零→非零变化率和raw0位置变化，避免把“无新增零值”误写成“没有零值”。

**进入剩余496图或短训前的唯一必要排伪影复核：** 若循环条件出现正向任务/机制信号，同32图、同操作矩形、同16像素位移改用非循环平移。对目标像素 $u$，源为 $u-(0,16)$；仅当源/目标均位于 $C_{op}$、原图support均有效、原raw8深度均>0且有限时，赋值为源的clean深度；其余保留目标clean值。三通道同步，不回绕、不填边、不新增零值，原raw0位置不变。该条件记为 **Correspondence16-Noncyclic**；它不保持直方图，也不是无伪影的真实配准模型，但可排除循环回绕和零值移动这两项主要混杂。

复核只跑O、已选A、同预算C/G（4方法×32=128次图前向），复用已有clean预测；不再增加位移强度、发生器或多轮搜索。若共同预算需从两档中选择，一次性依据32图已有clean代价、净修复及机制趋势确定后再复核。若换掉循环条件后仍有解释价值的正向趋势，进入确认；若正向变化只出现在回绕接缝或零值移动下，停止当前错位结论，不扩大样本或直接训练。

**区域记录与指标使用：** 操作矩形、人工删除mask和真实深度变化区域分别记录，不能混用。

- 实际受扰动区域 $C_\Delta$：原图支持域内，corrupt深度与clean深度数值实际不同的像素；在规范化前raw8或等价规范化值上比较。常量位置置换后没变、孔洞中原raw0填后没变的像素均不属于 $C_\Delta$。
- 循环接缝邻域 $J$：回绕赋值的16像素宽条带，再加其边界及操作矩形外边界的4像素邻域；定义预先固定，只用于区域统计，不输入模型。非循环条件无回绕条带，仍标记局部赋值边缘，不称其绝对无边界伪影。
- 将GT有效像素分为 $C_\Delta\cap J$、$C_\Delta\setminus J$、$C_\Delta^c\cap J$、$C_\Delta^c\setminus J$ 四组；同时汇总实际区内/区外。重点检查非接缝受扰动区和远离接缝的区外结果，不能只展示有利分组。
- Holes75用其自身 $C_\Delta$ 定义内外，另保留完整人工删除区域的计数用于解释严重度；不套用错位矩形。孔洞边界作为辅助邻域记录，不称循环接缝。

区域为空或非零可平移像素很少时报告样本数/面积和“该条件无足够有效干预”，不补造指标，不为了提高损害量反复改发生器。

### 6.3 最小强对照：可比预算下的投影、clip与gate

同一checkpoint、相同完整输入与HAM状态：

- O：Original完整Depth bias。
- OFF：所有块去掉 `Bd`，保留QKV/RoPE/Bs/LePE/FFN/HAM；冻结同权重干预，不是独立训练的RGB-only。
- A1/A2：真正KL投影，全stage/head统一候选 `tau=ln2` 或 `ln4`。W/H各用该tau，路径总预算为2tau。
- C1/C2：同tau的有界signed-bias clip：

$$
m_i=(\max_j B^d_{ij}+\min_j B^d_{ij})/2,\quad \widehat B^d_{ij}=\operatorname{clip}(B^d_{ij}-m_i,-\tau/2,\tau/2),\quad p^C=\operatorname{softmax}(S+\widehat B^d).
$$

- G1/G2：**逐行解析自适应有界gate，替换原固定0.5gate**：

$$
R_i=\max_j B^d_{ij}-\min_j B^d_{ij},\quad g_i=\begin{cases}1,&R_i=0,\\\min(1,\tau/R_i),&R_i>0,\end{cases}\quad p^G=\operatorname{softmax}(S+g_i B_i^d).
$$

clip的bias行跨度≤tau，gate的跨度为 $g_iR_i\le\tau$。对于相应bias $b$，其log归一化常数 $\log\sum_jp_j^0e^{b_j}$ 位于 $\min b$ 与 $\max b$ 之间，所以两者均满足：

$$
-\tau\le\log(p_j^{C,G}/p_j^0)\le\tau.
$$

三类方法每一预算档采用相同的p0定义、signed Bd构造、stage/axis范围与FP32协议。同一固定块输入下p0数值相同；端到端各方法的上游特征可以不同，必须重新计算各自当前块的p0，不能跨方法缓存参考。clip和gate是合法强对照，通常比完整可行域更保守；两者一般不是KL最优解。gate只根据该行bias跨度解析计算，不学习参数、不使用可靠性网络或受损标签。**最大可行gate仅在后续结果有争议时考虑，不属于首轮实现。**

初筛最多8方法×3条件×32=768次图前向，模型加载一次。两档budget不是搜索网格；趋势在首档已清楚时可省略第二档及冗余OFF条件。按相同tau报告A/C/G的任务指标与实际TV、KL、饱和率；如果A只是允许更大的实际变化，不能自动归因于KL最优性。

确认阶段固定一档共同tau，默认从ln2开始；若ln4呈现更合理的clean–修复折中，用32图已有结果一次选择。C/G均在该共同tau下比较并选最强简单组供短训，不给A更多调参机会，不在剩余496图重选预算。冻结阶段不训练，排除了“见过更多腐蚀”解释；短训再统一增强。

### 6.4 两个层级的机制探针

**固定块输入探针，最多8张预定筛选图**（32图索引每隔4张取1张）：分别在stage1第1块、stage3第1块保存clean到达的同一LayerNorm输入x；同Q/K/V与空间项下，用clean/扰动深度重新计算Bd，再比较O/OFF/A。优先复用32图诊断中缓存的少量输入，一次只保留一个块，不复制整网attention；因果作用已清楚时减少样本，不把跑满8图当准入条件。

- W/H分别报告 `log(pD/p0)`、clean与corrupt attention的TV、超预算行率、聚合输出差；固定V后验证单轴作用界。
- 对两轴完整子路径，再比较同输入下的 `A_H A_W V`；H的Q/K共享事实允许解释路径界，H V仍随W方法改变。
- stage深度受损token按 $C_\Delta$ 和**原双线性插值的实际非零权重邻域**标记，不用ODG面积池化替代原插值。对实际区外、非接缝且不接触受损插值邻域的query分别统计。
- 固定input探针只能建立当前块Bd的直接因果作用；端到端另外运行三种完整输入，观察全部历史传播。两类统计不可混作一张“证明”。

少量块探针未观察到明显作用时，结合32图任务变化和其他stage在线摘要判断，不凭8图单一探针直接淘汰方向。若关系损害基本不存在，报告阴性；若已有可解释的损害缓解，尽快进入一次确认/短训，不追加复杂oracle。已知受损区域的oracle不是首轮必跑项，避免引入mask机制。

### 6.5 指标：真实区域、修复与反向损害同时报告

**全局任务指标**：相同37类confusion matrix下的绝对mIoU、各自clean退化、mAcc；三条件平均/最差只作辅助，不能掩盖非零错位阴性。沿用 `SegMetrics` 的空GT类处理，不临时换平均类集合。

令GT为 $y$，GT有效域 $G=\{u:y(u)\ne255\}$，实际受扰动区内 $I=G\cap C_\Delta$、区外 $R=G\cap C_\Delta^c$；另按第6.2节的 $J$ 分组。$O_{cl}/O_{co}$ 为Original clean/corrupt预测，$M_{co}$ 为干预方法corrupt预测。所有方法使用同图同一 $C_\Delta$，不根据预测结果改变区域。

$$
\operatorname{NewErr}_O(R)=\frac{|\{u\in R:O_{cl}(u)=y(u),\ O_{co}(u)\ne y(u)\}|}{|R|},\quad \operatorname{NewErr}_M(R)=\frac{|\{u\in R:O_{cl}(u)=y(u),\ M_{co}(u)\ne y(u)\}|}{|R|}.
$$

方法间固定Original的clean正确集合和区域分母，避免某方法clean变差后虚假降低NewErr；方法自己的clean→corrupt变化另外作为补充。

$$
\operatorname{Repair}_M(R)=\frac{|\{u\in R:O_{cl}=y,\ O_{co}\ne y,\ M_{co}=y\}|}{|\{u\in R:O_{cl}=y,\ O_{co}\ne y\}|}.
$$

**反向损害与净准确率**同时报告，避免只展示修复：

$$
\operatorname{Harm}_M(R)=\frac{|\{u\in R:O_{co}=y,\ M_{co}\ne y\}|}{|\{u\in R:O_{co}=y\}|},\quad \Delta\operatorname{Acc}_M(R)=\frac{|\{u\in R:M_{co}=y\}|-|\{u\in R:O_{co}=y\}|}{|R|}.
$$

区内和接缝分组也报告同类指标。每项同时保留计数和分母；分母为0记“无适用像素/无可修复错误”，不造0%结论。净准确率计入所有GT有效像素的正确/错误变化，不要求每次修复都来自Original clean正确集合。

距离曲线相对于该条件的实际 $C_\Delta$，预定分桶0–16、17–64、>64像素；非循环错位的非接缝区外为传播主观察区。孔洞使用自身实际变化域，另外报告人工删除面积和其中原raw0比例。按图宏平均、按像素合并率和逐图差值方向共同呈现，说明小收益是否广泛而非被少数大图主导；不要求每张图都改善。

**算子指标**：29组ws/wd的符号/量级；stage/head倍率摘要、超预算行率、上下界饱和率、TV、投影 `KL(p*||pD)`，以及归一化/倍率误差。极小p0键与 `p0>=1e-6` 有质量键分开观察。统计服务于解释坏关系与任务修复，不把全部分位图生成完作为继续前置。

**成本指标**：关闭详细探针后同图warmup，取最多8图测端到端时间、求解时间与GPU峰值；一次即可，不开性能扫描。

### 6.6 继续、确认、短训与停止：依据稳定信号，不设百分点硬门槛

**算子正确性是必要条件**：第4节的小张量/真实行归一化、倍率界、特殊路径与共同FP32协议需成立；未成立记实现问题，不能当科研阴性。只做最小定点检查，不重复旧验收。

**32图阶段：** 重点看扰动是否改变Depth关系、A是否产生预期作用、净错误是否减少，以及变化是否在多张图中有一致解释。小幅、跨图一致的任务改善，或真实损害的部分缓解并具有合理独立机制，均可继续；不要求区外NewErr达到0.5pp，不要求半数图严格不恶化。clean代价结合几何修复与OFF表现判断，而非统一0.5pp准入线。关系/任务信号清楚时减少剩余无意义探针。

**一次排伪影后，剩余496图确认：** 固定第6.3节共同预算，保留O/OFF/A/C/G，评价clean、Holes75和 **Correspondence16-Noncyclic**，以非循环条件作为确认和训练后的主要错位评价；32图循环结果只作初筛，不额外要求在496图重跑循环条件。O/A/C/G的完整528图统计由非循环32图复核和496图确认合并，先分别报告两部分；OFF若未补32图非循环预测，只报告同496图比较，不冒称其有完整528图结果。

默认新增5方法×3条件×496=7440次图前向；复核未运行的OFF非循环预测仅在确有诊断价值时补32图。clean/Holes75已有32图结果直接复用。若OFF或非核心条件已有清楚答案，可减少重复条件，保留Original与A/C/G的主要任务比较。

确认重点：正向方向是否保持、是否排除了明显接缝/零值移动解释、clean代价是否合理、A相对有界clip/gate是否存在潜在优势。**即使A仅领先最强简单方法0.2–0.5pp，只要跨图较稳定、净错误减少、clean没有明显恶化，也允许申请一次短训。** 该区间是可接受信号的例子，不是新的最低门槛或统计显著性结论。

**申请一次同预算短训：** 真实几何损害存在、A能部分缓解，并与简单有界方法具有可检验的研究差异即可提出申请；不要求冻结结果先达到1pp或已经完全胜过所有简单方法。冻结权重可能尚未适配投影，它不是联合训练性能上限。较小但合理的结果优先用三组10epoch判断，避免继续堆诊断。若区外传播信号弱但另有稳定任务收益，明确收缩论文主张，不编造区外传播结论。

**主要停止情形：**

- 在实际有效扰动和必要观测下，核心几何损害基本不存在，缺乏研究对象。
- 正向结果主要来自循环接缝、零值移动或其他明显伪影，非循环复核没有支持。
- A既没有任务收益，也没有值得联合适配进一步验证的独立机制信号。
- 简单方法充分解释结果，且一次匹配短训后仍无KL投影优势或独立作用证据。

单一预算近似恒等、少量探针阴性、未达到人为百分点线，不单独作为方向停止理由；不为追到某个分数反复调整预算/扰动。不把“投影满足约束”或孤立TV下降当成充分创新证据。clean代价明显且无法以损害缓解解释时报告取舍，由科研裁决决定是否值得一次适配，不自动增加动态tau或辅助模块。

HAM跨环境历史差异不是新配对协议的固有噪声。仅在收益方向明显随HAM随机基变化、影响裁决时，在同32图用第二HAM seed54321复核O/A/最强简单组一次；稳定信号不例行重复评价。

## 7. 第二步：有条件的最小短程匹配训练

### 7.1 目标与共同起点

本节不是执行授权。第6节发现稳定任务改善，或真实几何损害被A部分缓解且具有合理独立机制信号后，可申请一次三组短训；启动前才进行最小投影梯度核验。**冻结诊断不需达到1pp，也不需先证明完整联合训练优势；正式短训不冻结骨干。**

**建议三组均从同一R2 Original完整模型权重开始，重建optimizer/scaler/RNG，最多各10个新增epoch。** 这叫共同起点适配训练，不是旧ODG resume，不是官方encoder从零建分割头，也不能称公平独立收敛训练。

现有 `train_odg.py::main`（208–625）仅支持官方encoder初始化或同合同完整resume；后者会恢复旧optimizer、计数、日程，不能拿来修改attention与增强合同。因此需新增明确的 `init_from_model` 语义，严格加载模型权重后新建优化器，写新目录和新合同；禁止删旧合同字段绕过resume检查。独立10epoch日程是新实验，不能声称沿用300epoch暂停语义。

三组：

1. Original + 相同腐蚀增强。
2. 第6节同预算比较选出的最强简单有界方法（clip或解析自适应gate）+ 相同增强。
3. A + 相同增强；tau/作用stage范围固定，不增加动态预算。

epoch0的结构插入代价直接引用已有配对诊断，不另跑完整重复评价；胜负比较同新增更新预算的epoch5/10绝对结果，不仅比较各自改善量。全模型包括encoder、HAM与原Geo权重共同优化，免训练诊断的权重保持不变不延续为训练限制。

### 7.2 建议固定合同

- 数据仍为4757 train-dev/528 dev；37类、480×480训练crop、原mirror/random_scale/crop-pad均保留。全部encoder/HAM/Geo参数训练，不新建独立分支或额外损失。
- 三组同seed12345、同有效batch16；4090先用4×4候选。若投影显存不足，全组三组一起改2×8，避免BatchNorm微批差异。默认顺序运行，不能假设历史双模型并行仍可复用。
- AdamW lr **2e-5**、wd0.01、betas0.9/0.999、poly0.9，总10epoch、warmup1epoch；每epoch298次尝试，共2980/组、8940/三组。该lr是保守适配提案，不是已验证最佳值。
- 每次样本增强70% clean、30%局部对应扰动，三组完全相同。使用已完成复核的**非循环、非零有效区域平移**，训练位移8像素、评价16像素，避免训练收益依赖循环接缝；不增设发生器。操作矩形尺寸沿用不超过192×192，训练中心随机、评价中心固定。扰动在同步几何增强后、合法支持域内实施；训练非零判定使用增强后深度的等价raw8>0条件，不用归一化tensor!=0推断。RGB/GT不变，不改padding或raw0位置；记录实际改变像素，避免名义增强没有有效干预。
- 首版不同时加入孔洞/噪声/低光训练、不修改CE、不加蒸馏；评价保留Holes75作为未训练故障。
- 使用独立、按seed/epoch/样本ID决定的腐蚀RNG；三组共享数据顺序与腐蚀参数，投影不消耗随机数。延续现有714/714唯一入组和BN验证保护，不为旧验收再建测试体系。
- 保存epoch5/10及last；只在epoch5/10做clean/Correspondence16-Noncyclic/Holes75配对dev，不用正式test选模型。科研判读以预定epoch10为主，不选三个方法各自最好时点来比较。

epoch5仅因数值/梯度实现错误、资源超额，或clean出现明显且无对应任务收益可解释的恶化而提前暂停并报告；不因未达到指定百分点提前淘汰。epoch10按任务净收益、跨图一致性、clean代价及相对简单方法的差异作科研裁决，不设1pp准入线，不自动续100/300epoch。

**增强替代解释与后续裁决：** 若Original+同增强充分解释收益，或简单有界方法在适配后仍完全解释结果且无投影独立证据，停止当前KL投影贡献主张。若A得到稳定、可解释的小幅增益和净错误减少，即使不足1pp，也可申请下一阶段正式匹配训练并同步启动第10节论文推进；证据尚不足时明确不确定性，不反复诊断或调参追分。共同起点短训不是最终投稿证据，但能快速判断是否值得投入完整验证。

## 8. 时间、资源和验证预算

以下为设计估计，求解器尚未测，不安排本轮资源。

- **实现与定点核验：** 约半天内完成薄接口、FP32冻结算子、扰动/区域指标；反向实现另需约半天，只有诊断值得继续才做。若求根/分块接口比预期复杂，报告阻碍，不先重构全仓。
- **32图诊断与一次排伪影复核：** 本地RTX5060，原始528图约1–2分钟；初筛最多768次图前向，正向时增加非循环复核128次，另有少量块探针，整体预估10–25分钟。首批图顺便测投影开销，趋势清楚时省掉第二档预算/冗余探针，不额外跑一套性能验收。
- **剩余496图确认：** 新增最多7440次图前向；结合已跑32图形成528图报告，不重跑整套528图。假设投影端到端2–5倍原始速度成本，粗估20–45分钟。诊断加确认建议**本地GPU累计上限60分钟**；非必要条件已清楚时减少重复评价，把后续计算优先留给一次短训。到资源上限报告已完成部分，不自动改用长CPU任务。
- 若A端到端>3倍或分块后batch1仍OOM，先报工程可行性问题；其收益须足以覆盖该成本。可批准更快等价活跃集算法，但不能把gate当优化后的同一投影。
- **三组10epoch适配：** 历史R2 4090并行两组30epoch共享窗口约3.08h，不能直接乘比例得到三组投影成本。按旧单组attempt开销量级与投影未知增量，初估**2–4 GPU小时**，建议申请硬上限**5 GPU小时或¥15，先到先停**；参考历史¥1.88/h约¥4–8，仅为历史价格估算，实际规格/报价需启动前确认。
- 有卡与生命周期操作须新授权；若用现有云实例，按[云端指南](../guides/cloud.md)设置并回读关机保险、完成后主动停机。现有历史Stopped记录不是本轮实时查询；不新建资源、不提前上传/推送。

验证只覆盖新增投影的数值/梯度、默认关闭行为、精度对照与配对统计。无需全仓扫描、完整测试、200步×多组旧smoke、全量checkpoint哈希、新临时验证文件。后续实际检查均进入研究入口的必要诊断能力或直接定点调用，不能将测试开发变成本轮主要工作。

产物建议：`outputs/influence-sun-diagnostic/<run-id>/` 保存配置/方法/seed、逐图指标CSV、条件confusion matrix、在线attention摘要、最多8张机制图；原始attention不全量落盘。短训另写独立目录。结果由主代理核验后写正式报告，不把预计值写入当前事实。

## 9. 云端总方案与本地实现的差异及建议修改

### 9.1 核心定义：可以按总方案执行

KL方向、双向相对概率界、signed Bd、参考p0、最优解与TV界均无重大矛盾。无需更换核心创新；新增Depth encoder/可靠性gate/补全均不是首版必要项。

### 9.2 W/H路径界：由待核实升级为限定条件下成立

总方案担心H可能重新形成Q/K；真实源码共享同输入Q/K，W输出只进入H的V。建议总方案v0.2明确：**单块固定输入、固定head的两轴核满足路径/端点倍率界**；继续保留“不能推到全网络/精度”的限制。预算使用每轴tau时路径为2tau，不混用。

### 9.3 简单clip/gate：修正绝对化区分，保留KL最小性区别

“缩放或bias clip不保证指定最终界”对未经预算设计的操作成立，但不适用于所有简单方法。本次按云端审核保留有界clip，并以第6.3节逐行解析有界gate替代固定0.5gate；三者同预算配对。最大可行gate仅保留为后续争议时的补充，不首轮实现。

潜在贡献定位为“**给定相对作用可行域内、对原几何分布的最小KL改变及其任务价值**”，不是“只有投影可以控制影响”。冻结阶段有稳定小幅信号或部分损害缓解即可申请一次适配；若简单方法在匹配短训后仍充分解释结果，才停止当前投影优势主张。

### 9.4 投影求根、精度和梯度：补上具体实现约束

log-domain有显式括区，避免任意扩大c；精度尾部不能用epsilon改参考；二分前向与可微训练不是同一验收。建议补充活跃集/隐式梯度方案及共同FP32聚合协议。这里属于工程实现补充，不更换KL定义。

### 9.5 对应扰动：区分局部置换、零值移动与真实错位

局部循环位移16仍用于初筛；按最新审核，正向后必须在同32图做一次非循环、非零有效区域平移复核，再进入496图或训练。内外区域统一改为实际深度改变区域，循环接缝和操作矩形单独记录，孔洞不套用错位矩形。短训采用已复核的非循环增强，避免学习循环伪影。该改动不新增真实传感器仿真系统。

### 9.6 评价/短训：澄清真实分辨率与复用语义

研究独立评价整张531×730，最大probability开销在stage1；训练480×480。完整R2权重可用，但现有resume不能跨方法/增强合同。建议第一次短训是三组同完整权重、重建优化器的10epoch适配，不是直接给candidate接旧训练或声称正式收敛对照。

### 9.7 其他范围：按总方案保留，不自行扩张

DeLiVER正式匹配训练留到筛选通过后；方向B不实现、不建立候选评分网络；不重新调查近邻论文或承诺新颖性。未发现需要因本地工程把方向A换成普通门控的理由。

## 10. 有效后的论文快速推进路线

**触发条件：** A出现有说服力、可解释的正向信号，并通过必要的排伪影；冻结阶段即可开始方法和论文表达，三组短训进一步支持价值后立即申请正式证据建设。创新整理与后续实验并行推进，不等全部实验结束再构思论文，不长期停留于小实验循环。

- **深化真实数学机制：** 完成KL目标方向、可行性、唯一分布解、KKT形式和最小改变性质的清晰推导；整理双向密度比与局部TV/固定V界，以及同块固定Q/K下W/H复合传播界。说明逐轴预算如何组合、有限精度及活跃集梯度的适用条件。联合训练会改变参考S，跨块残差/非线性也超出该界；不得写成最终分割精度或部署安全保证。
- **形成简洁模块表达：** 从“错误几何干预RGB关系”自然引出spatial-only参考、完整几何分布、预算可行域和KL投影，画一张RGB特征/QKV与Depth bias进入同一attention的框架图，突出W/H共享Q/K与完整路径。以一个核心算子讲清结构，不为美观堆可靠性分支或辅助损失。
- **明确相对简单方法的实质贡献：** 普通有界attention、解析gate和clip都能满足作用界，硬约束不是独占贡献。解释投影如何在可行域内保留原几何关系、何时只改必要关系，以及这种差异是否对应可靠Depth保留、坏关系缓解和任务净收益。以实际数据支持这些命题；KL最优性本身不推出任务最优。
- **同步建立论文故事：** 真实RGB-D几何问题与受控扰动边界 → 现有几何attention/普通强度控制的具体不足 → 关系干预洞察 → 自然数学约束和模块 → 机制及任务证据。没有区外传播证据时收缩为局部几何影响管理，不保留未经支持的传播叙事。
- **尽快形成完整证据计划：** 核心短训有效后，申请正式匹配初始化/训练的Original、最强简单有界方法、A，以及同增强公平性；再选一个与研究主张匹配、数据成本可承受的第二数据集，规划必要预算/作用位置消融、机制图和独立训练seed复核。已有官方test仅在方法与选择协议固定后评价。第二数据集及正式训练是下一阶段申请项，本轮不提前接入或执行DeLiVER/NYUv2。

正向结果明确后即整理问题定义、方法章节和核心图的草稿，并把缺失证据转成最少的正式实验清单。文献竞争性只围绕已形成贡献做必要核查，不重启大规模选题调研；投稿等级与新颖性仍由证据和上级科研审核裁决。若短训后简单方法完全解释结果，诚实收束，不为维持论文题目事后编造理论优势。

## 11. 剩余实质裁决与审核范围

本次审核已经明确有界gate、一次排伪影、放宽继续条件和三组10epoch共同起点路线；不再将这些已确定事项列为待答问题。**目前没有新增核心数学或源码冲突阻碍最小路线。** 默认保留每轴tau=ln2/ln4候选、路径预算合计2tau，所有方法同一解释；不重新询问已确认原则。

仅有两类条件性事项需结果出现后交上级裁决，不要求现在新增流程：

1. **资源受限的作用范围：** 若全29块投影实际成本无法在预算内完成，优先优化等价求解/分块；仍不足时，是否批准限定stage的诊断/适配变体。改变覆盖范围必须明示，不能静默简化后把阴性外推到完整A。
2. **收益与clean代价的研究定位：** 若A稳定缓解几何损害，但有效Depth保护或区外传播证据有限，依据一次短训结果决定继续哪一条有证据的论文主张或停止。以净收益、机制差异和应用意义裁决，不恢复固定1pp门槛，也不自动加入动态预算/新模块。

科研计划v0.2已通过云端审核；第13节据此限定第一阶段执行职责。后续10epoch训练仍不是第一阶段实施内容，文件交接不授权本轮运行GPU或启动云资源。

## 12. 交付、暂停与证据入口

本文件是云端总方案的本地执行设计，原 `DFORMER_RESEARCH_MASTER_PLAN.md` 保留，统一科研总方案修订由上级负责。

此前v0.2修订只改本文，保留已核实数学与源码事实，更新强对照、扰动区域、宽松继续条件和论文路线，当时按授权未更新状态。本轮用户确认科研审核已通过，现补充第13节技术任务单，并更新 `doc/state/current.md` 的阶段、边界与恢复点；其余科研设计和总方案保持原样。只做文档内容/差异复核，不运行项目测试或实验；算子数值、估时和方法收益未在本轮实测，反向传播明确不属于第一阶段。

关键既有证据：

- [当前状态](../state/current.md)：审核通过、任务单交接边界与当前恢复点；文档细化不构成新的实验事实。
- [R2 30epoch报告](../reports/2026-10-10-odg-sunrgbd-r2-30epoch.md)：完整权重、训练合同与耗时。
- [50%/75%孔洞筛选](../reports/2026-10-10-odg-r2-holes5075-screening.md)：严重缺失结果与评价速度。
- [R1几何诊断](../reports/2026-10-10-odg-research-diagnosis.md)：预训练signed系数与输入编码历史证据，不代替R2新测。
- `outputs/sun-dev-original-seed12345-r2/20261010-094231/{contract.json,resolved_config.json,validation.csv}` 与 `outputs/odg-r2-eval/original-clean/eval_result.json`：初版设计直接回读的运行身份/记录；本轮没有重复资产验收。

**恢复点：科研v0.2已审核通过。将第13节连同本文交执行模型作为独立实施任务；接手实施并获得本地GPU运行授权后，按同一任务单完成冻结诊断链。主力模型在第一阶段交付后定点审核代码与验收科研证据。本轮在任务单准备后停止，不启动实施会话、实验、训练、推送或云资源操作。**

## 13. 执行模型实施任务单

### 13.1 接手范围与完成定义

**任务：按已批准科研定义实现并交付方向A第一阶段冻结诊断。** 本节是开发与评价的实施合同，第4/6节提供数学和实验解释；不重新设计方向、不扩展研究范围。当前仅交付文档；接手实施及本地GPU获授权后，执行模型可独立完成以下四步，不需每个函数、一般修复或每次命令另行审批：

1. 核心算子、模型接口、薄评价入口与必要CPU自检。
2. 固定32图三条件诊断，最多O/OFF/A1/A2/C1/C2/G1/G2。
3. 有循环错位正向信号时，固定共同tau，做同32图非循环排伪影复核。
4. 依据第6.6节的宽松原则，有研究信号且无明显伪影解释时，做剩余496图确认；证据明确时可减少冗余条件。若仅孔洞有信号，确认可保留孔洞/clean并如实收缩错位主张，不强行补出循环正结果。

本阶段全程 `model.eval()` + `torch.inference_mode()`，不更新任何权重。**不实现backward、梯度测试、训练入口、10epoch程序或训练运行**，不启动付费云端、正式test或第二数据集。一般维度、设备、dtype、分块、导入、输出和统计错误自行定点修复；必要的等价数值稳定化可自主完成并记录。只在需要改变KL方向/参考/约束/双轴定义、重大资源问题（第8节）、或核心假设与证据矛盾时暂停反馈。研究信号不明确时交付已有证据和建议，不通过反复调参追分；最终科研去留、状态与正式结论由主力模型负责。

### 13.2 文件与最小接口

限定新增3个文件、修改1个模型文件；另交付一份简短执行报告及实际产物。既有ODG代码、HAM、数据预处理和科研总方案不改行为。

- **新增 `research/influence_projection.py`**：`project_log_ratio(log_p0, r, tau, *, iterations=24) -> (log_p, t)`；`controlled_attention(qk, bs, bd, mode, tau, *, observer=None, meta=None) -> p`。最后一维是完整key，前面任意行维；FP32、no-grad，`t=log(p/p0)`；`project_log_ratio` 接收已归一化的l0与满足 `logsumexp(l0+r)=0`（浮点容差内）的r，不对任意未归一化输入另改定义。`mode` 为 `original|depth_off|kl|clip|gate`，`tau` 是有限非负标量；实现13.3节和少量 `run_self_check()`，不建立测试目录/临时验证脚本。
- **修改 `models/encoders/DFormerv2.py`**：`GeoPriorGen.forward(..., return_components=False)` 默认返回原 `geo_prior`；开启时返回 `(geo_prior, parts)`，其中 `parts` 为已乘原signed权重的 `bs_h/bd_h/bs_w/bd_w` 或 `bs/bd`。原合并偏置的顺序、rel_pos格式、参数名/shape不变。`RGBD_Block.forward` 仅研究分支请求parts；`Decomposed_GSA.forward(..., *, bias_parts=None)` / `Full_GSA.forward(..., *, bias_parts=None)` 在softmax位置接入，原调用仍兼容。所有非研究路径不引入新模块依赖或额外浮点计算。
- **新增 `research/evaluate_influence.py`**：`configure_influence(model, mode, tau, query_chunk, observer=None)` 遍历29块，在block/attention安装普通非参数控制属性和算子引用；默认 `legacy`，恢复legacy时移除/禁用探针。入口负责一次完整权重加载、多方法切换、逐图配对、扰动、区域统计、在线摘要和阶段结果复用。不得扩展builder注册表、注册新参数/buffer或改变state_dict。
- **新增 `local_configs/research/INFLUENCE_SUNRGBD.py`**：deepcopy现有 `ODG_SUNRGBD.C`，固定 `geometry_mode='original'`、DFormerv2-S/HAM1024/37类、dev、`C.pad=True`（531×730）、workers2、单尺度无flip，默认关闭在线日志；不修改被导入的配置对象。`DFORMER_DATASET_ROOT` 沿用现有“根目录下再加SUNRGBD”的语义。

`legacy` 指作者现有精度/前向原路径；`original` 指共同FP32 attention协议下完整Bd的配对基线O，两者必须区分。研究控制分支在train模式或启用梯度时明确报错，防止未经验证被误用于训练。执行模型只负责提供实施报告，不自行写入项目实时状态或科研最终结论。

### 13.3 KL求解与强对照：按此实现，不猜核心机制

所有 `sum/max/min/logsumexp` 均沿最后一维key、`keepdim=True`。每个query分块内先关闭autocast，设 `S=qk.float()+bs.float()`、`l0=log_softmax(S)`、`b=bd.float()-amax(bd.float())`、`r=b-logsumexp(l0+b)`、`lD=l0+r`。减行常数仅用于稳定化，不改signed关系或softmax分布。

```python
# project_log_ratio：log_p0=l0，r=log(pD/p0)，有限FP32输入
if tau == 0:
    return l0, zeros_like(r)
feasible = (abs(r) <= tau).all(key_dim, keepdim=True)
lo = -amax(r, key_dim, keepdim=True)
hi = -amin(r, key_dim, keepdim=True)
for _ in range(iterations):                 # 向量化；禁止逐行循环/.item()
    z = (lo + hi) / 2
    t = clamp(z + r, -tau, tau)
    mass = exp(l0 + t).sum(key_dim, keepdim=True)
    lo = where(mass < 1, z, lo)             # 质量不足，增大c
    hi = where(mass >= 1, z, hi)
# 不能取最后一次循环的旧t；使用最终括区中点
z = (lo + hi) / 2
bounded_t = clamp(z + r, -tau, tau)
t = where(feasible, r, bounded_t)           # 已可行时原样保留pD
return l0 + t, t
```

这里 `z=log c`，输出 `p=exp(log_p)` 直接用于聚合；**不再softmax、不再除概率和、不clamp p0到epsilon**。根可不唯一（全饱和平台），但分布仍唯一；单key、常量Bd、tau0及已可行行不应出异常。输入NaN/Inf或非法tau拒绝；本版不处理结构性−inf key mask。24次是默认，不是数学定义：若容差失败先定位原因，可改用32/48次或等价稳定求根并记录开销；不能换成普通裁剪。

同一个 `controlled_attention` 内实现全部配对方法，输出统一FP32：

- `original`：`t=r; p=exp(l0+r)`；`depth_off`：`t=0; p=exp(l0)`。
- `kl`：调用上式，最小化 **KL(p||pD)**，不是KL(pD||p)，也不是投影到bias区间。
- `clip`：用原Bd计算 `m=(max(Bd)+min(Bd))/2`，`bc=clamp(Bd-m,-tau/2,tau/2)`；稳定计算 `t=bc-logsumexp(l0+bc)`、`p=exp(l0+t)`。
- `gate`：用原Bd计算 `R=max(Bd)-min(Bd)`、`g=1 if R==0 else min(1,tau/R)`，可用 `R_safe=where(R>0,R,1)` 避免0/0；取 `bg=g*(Bd-max(Bd))` 后同clip归一化。行常数平移与 `softmax(S+g*Bd)` 等价，不取abs(Bd)、不钳制权重正负。

clip/gate经softmax归一化，是bias行跨度≤tau的更保守控制；两者保证同一双向倍率界，**一般不等于概率可行域内的KL最小投影**。A在归一化条件下共同求c，可按p0质量分配自由项，不是先clip概率再renormalize。W/H各tau，路径预算合计2tau；full预算tau。

内置自检仅包含少量固定小行：手算 `(p0,pD,tau)=((.5,.3,.2),(.8,.1,.1),ln2)` 应得 `(.75,.15,.10)`、`c=.9375`；tau0→p0；常量Bd/单key→p0；已可行pD保持不变；signed正负Bd下A/C/G均满足界。一个很小的FP64求根参考足够检查默认FP32偏差，不做模型级FP64/随机大规模扫描。后续真实行在线累计各方法归一化误差≤1e-6、A/C/G的log倍率超界≤5e-6与有限性，阶段结束汇总；O的超预算率是观测结果，不作为实现失败，OFF只验证p0/归一化。不可把underflow的0/0当倍率。记录尾部零概率数量及由log-domain估计的质量，失败属于实现问题，修好再判科研信号。

### 13.4 注意力接线、维度和精度

设head数N、单head键维dk、值维dv、T=H×W。**作者K已乘 `self.scaling`，因此实际代码的 `qk=qr@kr.T` 不再除sqrt(dk)。** Q/K/RoPE/V/LePE的原生成方式不变。

- **W轴**：qk `[B,H,N,W,W]`，原V `[B,H,N,W,dv]`；Geo的 `bd_w=[B,N,H,W,W]`、`bs_w=[1,N,1,W,W]`。两者各 `transpose(1,2)` 后为 `[B,H,N,W,W]` / `[1,1,N,W,W]`，广播加入S与Bd。
- **H轴**：qk `[B,W,N,H,H]`，V为W聚合结果 `permute(0,3,2,1,4)`，即 `[B,W,N,H,dv]`；`bd_h=[B,N,W,H,H]`、`bs_h=[1,N,1,H,H]` 同样转置1/2。H只用原qr/kr重新排列，**不从W输出生成Q/K、不令H重新使用原V**。
- **Full**：qk `[B,N,T,T]`、V `[B,N,T,dv]`；`bs=[N,T,T]` 加batch维广播，`bd=[B,N,T,T]`；保留原flatten顺序与输出reshape。

query分块16/32只切倒数第二维，每块保留全部key；`Bs/Bd`用对应query切片并广播，不沿key单独softmax，不改变行归一化。可连QK一起分块计算，拼接各query聚合结果；不要为observer复制全attention。分块改变归约数值时记录合理浮点误差，不以缓存别的方法p0提速。

QK生成可沿用AMP；S构造、log_softmax、求根、概率和 `p.float() @ V.float()` 在关闭autocast的区域内完成，A/O/OFF/C/G相同。W聚合结果保持FP32进入H；最终H/full聚合排列后转为 `lepe.dtype`，再走未改动的 `+lepe/out_proj`。严禁W结束即转FP16或让H matmul被autocast降精度。

默认关闭时完整R2 checkpoint必须 `strict=True` 加载、无新增state_dict键，原rel_pos与legacy路径正常工作。复用32图前4张比较legacy与共同FP32 O，差异作为精度基线说明；同一图同RNG在控制开启前后恢复legacy应复现原路径输出（按实际设备数值容差报告，不新增独立整网smoke）。旧45.34不是本轮配对结果的硬目标。小非方形W/H/full固定张量检查分块与整行聚合一致即可，不建立尺寸测试矩阵。

### 13.5 扰动、区域与计数接口

在评价入口就地实现 `make_condition(depth, original_support, condition, original_index, seed)`，返回 `(corrupt_depth, model_support, region_info)`；`region_info` 至少包含 `operation_mask/deleted_mask/changed_mask/seam_mask/assignment_mask` 及面积。用原support定义区域，不能因孔洞support变0而丢掉损坏像素。模型固定original几何，任何support变化都不能变成attention key mask或深度项中性化。

1. 从clean归一化第一深度通道重建 `raw8=round(255*(.28*d+.48)).clamp(0,255)`；本轮ValPre未做深度插值，这是原raw8的等价恢复。用raw8比较clean/corrupt确定 `C_delta`，避免raw0填充值的FP32末位差被算成干预；三深度通道必须同步。GT255与padding仅在指标/支持域排除。
2. holes75调用原 `apply_artificial_holes(..., ratio=.75, seed=12345, start_index=original_index)`，一次生成供所有方法共享；返回的完整删除mask由原support与corrupt support之差得到。`C_delta=deleted_mask & (clean_raw8!=0)`，记录原零比例与实际改变面积，不能用中心矩形定义孔洞区内。
3. cyclic16：原support矩形 `(top,left,h,w)` 内设 `ch=min(h,192), cw=min(w,192)`、中心起点 `top+(h-ch)//2,left+(w-cw)//2`；仅该矩形向右 `roll(..., shifts=16, dims=-1)`。宽≤32按既定规则跳过并记原因，不移动RGB/GT/support/padding。
4. noncyclic16：源列=目标列−16；源/目标都在同操作矩形、原support有效、原raw8>0且有限才赋clean源值。从不可变clean张量一次赋值，不在已修改张量上迭代取源；其他位置保留clean。`assignment_mask` 表示允许赋值域，`C_delta` 仍按实际raw8变化定，不用assignment代替。
5. 为固定第6.2节4像素边界邻域，定义 `band4(M)=dilate_square(M,4) & ~erode_square(M,4)`（9×9邻域，图外按false）。循环 `J=回绕左16列条带 | band4(条带) | band4(Cop)`；非循环 `J=band4(assignment_mask)`；孔洞辅助邻域 `J=band4(deleted_mask)`。这些都是诊断标签，绝不输入模型。

`regional_counts(gt, original_clean_pred, original_corrupt_pred, method_corrupt_pred, region_info)` 按第6.5节输出：区域像素数、共同Original-clean正确数、NewErr分子、可修复错误数、Repair分子、Original-corrupt正确数、Harm分子、方法正确数、净正确数差。覆盖实际区内/区外及四种C_delta/J交集；计数优先整型，率从分母统一派生。分母0输出null及计数，不当0%参与宏平均；同时保留“净收益”中来自Original-clean错误像素的纠正，不能只统计Repair。

距离使用到实际C_delta的欧氏像素距离，向上取整后分桶0–16、17–64、>64；C_delta为空则距离指标null，不能把全图当远端传播。可复用现有OpenCV距离变换，不新增依赖。每图原始计数/比例、逐图方向、有效图数与合并像素率一起给出；mIoU从累计confusion matrix重算，**不平均32图/496图的mIoU获得528图mIoU**。

### 13.6 一次加载、逐图配对与机制摘要

加载固定 `outputs/sun-dev-original-seed12345-r2/20261010-094231/last.pth`，复用 `evaluate_odg.load_state_dict(path, allow_raw=False)`；deepcopy配置，`EncoderDecoder(criterion=None, norm_layer=BatchNorm2d, syncbn=False)`、`restore_decoder_norm_config`、严格加载完整model、eval。checkpoint合同dataset/backbone/classes若与SUN37/v2-S不符立即指出冲突，不能只警告后继续。采用现成loader.dataset + `Subset(original_indices)` 重建batch1 loader，保留原ValPre，不造新split文件，也不为选32图先解码528图。

逐图流程：

```python
for subset_pos, sample in enumerate(subset_loader):
    original_index = original_indices[subset_pos]  # batch1，不能用子集重编号
    conditions = make_shared_conditions(sample, original_index)
    seed_everything((ham_seed + original_index) % (2**31))
    paired_rng = capture_rng()               # 同时覆盖CPU和CUDA
    for condition, method in requested_pairs:
        restore_rng(paired_rng)             # 方法/条件前均恢复
        configure_influence(model, method.mode, method.tau, query_chunk, observer)
        logits = forward_logits(model, conditions[condition], ...)  # inference_mode
        accumulate_confusion_and_regional_counts(logits.float(), ...)
```

所有阶段固定这一逐图HAM seed公式（默认ham_seed12345），不依赖阶段内样本次序，便于复用clean。孔洞seed独立固定12345；不得让方法/observer消耗随机数。统计所需O-clean/O-corrupt先获得；同图所有方法/条件共享输入张量与基状态，不累计污染。用既有 `SegMetrics` 累计37×37矩阵，保持37类平均、GT255、单尺度无flip的现有定义；细小差值从未舍入矩阵值计算，展示可用百分点，保留足够小数。

observer以 `meta={stage, block, head/axis, query_slice, original_index, condition, mode, tau}` 和当前FP32 `l0/r/t` 在线累计：原始超预算行率、实际log倍率范围/质量加权摘要、上下界占比、TV、`sum(p*(t-r))` 即KL(p||pD)、归一化/倍率误差；极小p0与p0≥1e-6键分开。clip的bias截断率另记，不冒充A的概率界活跃率。输出29组signed ws/wd和stage/block/head/axis汇总，不落盘全概率矩阵。

固定输入探针按第6.4节只取stage1/block1和stage3/block1、最多8张预定图：只从共同FP32 O-clean缓存到达的LayerNorm输入及当时Q/K/V，用clean/corrupt Bd比较O/OFF/A。H单轴界使用同一个V；两轴组合使用各自W输出但共享原Q/K，不构造巨大的全路径矩阵。按 `align_corners=False` 的原双线性插值正权重邻域标记stage受干预token，同时记录stage深度实际改变mask；padding/天然零语义不改。块探针与全网自由传播摘要分别命名，不能混成全网络保证。

### 13.7 执行顺序与目标命令

以下CLI由执行模型实现，**当前尚不能据此宣称入口已存在或命令已运行**。入口提供 `--phase self-check|pilot32|noncyclic32|confirm496`、`--methods`、`--tau log2|log4|both`、`--query-chunk`、`--seed`、`--config`、`--checkpoint`、`--device`、`--out`、`--reuse`；不提供训练/官方test/自动云启动功能。`self-check`仅小CPU张量、不加载checkpoint/数据；其余阶段只在获批本地GPU下执行，无CUDA时明确停止，不自动转长CPU评价。

在仓库根目录PowerShell中，完成实现后的目标调用为：

```powershell
$py = 'D:\2Env\anaconda\envs\dformer\python.exe'
$env:DFORMER_DATASET_ROOT = 'D:\0Project\dataset'
$ckpt = 'D:\0Project\DFormer\outputs\sun-dev-original-seed12345-r2\20261010-094231\last.pth'
$runId = Get-Date -Format 'yyyyMMdd-HHmmss'
$out = "D:\0Project\DFormer\outputs\influence-sun-diagnostic\$runId"
$cfg = 'local_configs.research.INFLUENCE_SUNRGBD'
& $py -X utf8 -m research.evaluate_influence --phase self-check --device cpu
& $py -X utf8 -m research.evaluate_influence --phase pilot32 --config $cfg --checkpoint $ckpt --methods original depth_off kl clip gate --tau both --query-chunk 32 --seed 12345 --device cuda:0 --out "$out\pilot32"
# 一次选择共同tau；下面log2仅表示默认，log4更合理时一致替换，不在496图重选
& $py -X utf8 -m research.evaluate_influence --phase noncyclic32 --config $cfg --checkpoint $ckpt --methods original kl clip gate --tau log2 --query-chunk 32 --seed 12345 --device cuda:0 --reuse "$out\pilot32" --out "$out\noncyclic32"
& $py -X utf8 -m research.evaluate_influence --phase confirm496 --config $cfg --checkpoint $ckpt --methods original depth_off kl clip gate --tau log2 --query-chunk 32 --seed 12345 --device cuda:0 --reuse "$out" --out "$out\confirm496"
```

末次pilot交付需记录共同tau选择依据；noncyclic32/confirm496要求单一tau，拒绝both，不能在确认集重选。这些是分阶段命令示例，**不做无条件串行一键全跑**。pilot最多768次、noncyclic复核128次、confirm默认7440次图前向；Original/OFF每条件只跑一次，不因两档tau重复。接口允许显式减少方法/条件并记录原因；仅某预算近似恒等、收益不足1pp、少量图阴性不自动淘汰。固定32图索引 `floor(k*528/32)`，confirm取清单顺序中的严格补集496图，assert互斥及计数，不创建临时清单。

跨阶段复用需要pilot保存32图各实际方法的clean预测（压缩类别图）及统计，尤其共同O-clean；保存必要region元数据。核对checkpoint路径、代码版本/修改说明、HAM种子公式、原索引、tau、精度和配置一致后复用；不搭建hash/复杂缓存框架。O/A/C/G的528非循环结果来自复核32+确认496；OFF没有32非循环则仅报告496。某方法/条件没跑就标未运行，不能补造整套528结果。

首次几图同时做真实行数值记录及legacy精度比较；性能按第6.5节最多8图warmup后测一次，关闭详细探针、CUDA计时同步并报告模型前向/求解时间/峰值显存。OOM先减query chunk并优化等价工作区；仍OOM、不可接受>3倍开销或本地GPU累计60分钟将超限时反馈已完成部分，不暗改stage范围、不转云端、不训练。

### 13.8 最终交付与主力模型验收点

产物在上述outputs目录，简短报告在 `doc/reports/<实际日期>-influence-frozen-diagnostic.md`；报告区分实测、未运行和建议，不写最终科研裁决。至少交付：

- 四处代码/配置改动及接口说明；CPU小行自检、真实行最大归一化/倍率误差、默认关闭和严格加载的实测结果；若未完成说明准确阶段与原因。
- 运行配置、checkpoint身份、代码版本与未提交改动说明、phase原索引、已跑/省略的方法条件及tau、HAM/孔洞seed、FP32与query分块设置；不新增全量权重hash流程。
- 逐图区域计数/分母/率CSV、各方法条件37×37 confusion matrix及任务汇总、32/496分开和可合法合并的528结果；clean代价、NewErr、Repair、Harm、净正确变化及跨图一致性，包含空域/跳过情况。
- signed Geo系数与在线attention摘要、两处固定输入探针结果、实际变化/零值移动/接缝分组；最多8张有输入/GT/O/A/最强简单组的解释图，包含失败或反例，不只挑改善图。
- 实际耗时/峰值显存、已用GPU分钟、局限和“是否值得申请反向实现及一次三组短训”的证据建议。不把0.2–0.5pp例子变成硬门槛，不把局部TV界写成精度保证。

主力模型第一阶段完成后只做定点代码审核：KL方向与共同求c、Bs/Bd加权与维度、W→H和FP32聚合、legacy兼容、配对RNG、实际区域和共同分母、阶段合并是否准确；再按第6.6节验收科研价值。审核已交付证据，不例行重复完整诊断；值得继续时另行安排反向与短训，并同步推进第10节数学/论文工作。
