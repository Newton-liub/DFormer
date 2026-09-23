段落主题: MMFR核心PR文献
================================================================================

## 编号: PR029
发表年份: 2026
作者: Mianzhao Wang; Fan Shi; Xu Cheng; Chen Jia; Shengyong Chen
标题: Uncertainty-Aware Modality Fusion for Unaligned RGB-T Salient Object Detection
期刊/出版物: IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR 2026)
摘要: Unaligned RGB-T salient object detection (SOD) remains challenging due to severe cross-modal spatial discrepancies and unreliable feature fusion. Existing methods often assume perfect alignment or rely on geometric registration, which is computationally demanding and sensitive to cross-modal inconsistencies. To address these limitations, we propose an uncertainty-aware modality fusion network (UMFNet) that reformulates RGB-T SOD as an uncertainty-aware representation learning problem. Specifically, the proposed uncertainty alignment module (UAM) models pixel-wise features as Gaussian latent distributions to estimate local uncertainty and identify cross-modal consistency regions within the feature space, thereby achieving implicit alignment without explicit registration. Furthermore, the confidence-guided global modulation (CGM) mechanism leverages confidence maps derived from uncertainty estimation to adaptively regulate the fusion of RGB and thermal features, enhancing salient cues in reliable regions while suppressing noisy or inconsistent information. Extensive experiments on five unaligned and three aligned RGB-T SOD benchmarks demonstrate that UMFNet achieves state-of-the-art performance across diverse alignment conditions.

提出 UMFNet，面向未对齐 RGB-T 显著目标检测，将跨模态对齐与可靠融合重构为 uncertainty-aware representation learning。核心 UAM 对每个模态、每个空间位置的 feature 建模 Gaussian latent distribution：$z_M(p)\sim\mathcal{N}(\mu_M(p),\sigma_M^2(p))$，其中预测方差 $\sigma_{M,c}^2(p)$ 被解释为 channel-wise pixel uncertainty，并通过 reparameterization $\tilde z_M(p)=\mu_M(p)+\sigma_M(p)\odot\varepsilon$ 获得 latent feature；另利用 KL divergence 将 modality-specific latent distribution 正则到 $\mathcal{N}(0,1)$，但没有 uncertainty/reliability GT，因此 uncertainty 属于 task-supervised latent uncertainty，而非显式 corruption severity。RGB 与 Thermal latent representations 进一步联合产生 cross-modal consistency distribution，并映射得到 aligned Thermal feature $\tilde F_t$。CGM 将 variance 转换成局部可靠性：首先计算 $\mathrm{InvU}_M(p)=1/\left[\frac{1}{C}\sum_c\exp\left(s(\log\sigma_{M,c}^2(p))\right)+\epsilon\right]$，形成 modality-specific pixel-wise reliability map，再将 RGB 与 Thermal 的 $\mathrm{InvU}$ 拼接后通过 pixel-wise convolution、learnable temperature 和 Sigmoid 得到联合 fusion confidence $\mathrm{Conf}(p)\in[0,1]$。该 confidence 同时参与 channel modulation 与 spatial modulation：aligned Thermal feature 产生 channel scale $\gamma_t$、bias $\beta_t$ 和 spatial prior $P_t$，最终以 $\tilde P_t=P_t\cdot Conf$ 控制 residual fusion $F_{fused}=F_v+\tilde P_t(F_m-F_v)$；低 confidence 时输出自然回退至 RGB feature，高 confidence 时允许更强的 Thermal-guided modulation。论文进一步证明 UAM、confidence prior、channel modulation、spatial modulation 均独立有效，并发现仅在浅层 Stage 1–2 插入 UAM+CGM 优于所有四个 stages 全部插入。该工作与 MMFR 在“局部可靠性估计→confidence→动态控制辅助模态贡献”上高度重叠，尤其已经覆盖 Gaussian pixel-wise uncertainty 与 confidence-guided feature modulation，因此 MMFR 不应采用“Gaussian variance→confidence→Depth feature gating”作为核心创新。其主要区别在于 PR029 针对 RGB-T spatial misalignment 和 generic cross-modal fusion，并不显式建模 Depth corruption type/severity、geometry trustworthiness，也不干预 DFormerv2 geometry operator；MMFR 应进一步突出“Depth degradation diagnosis→geometry-specific reliability→DFormerv2 geometry-path intervention”。

--------------------------------------------------------------------------------

## 编号: PR059
发表年份: 2026
作者: Guodong Sun; Junjie Liu; Gaoyang Zhang; Bo Wu; Yang Zhang
标题: Efficient RGB-D Scene Understanding via Multi-task Adaptive Learning and Cross-dimensional Feature Guidance
期刊/出版物: Knowledge-Based Systems, Volume 327, 2025, Article 114107
摘要: Scene understanding plays a critical role in enabling intelligence and autonomy in robotic systems. Traditional approaches often face challenges, including occlusions, ambiguous boundaries, and the inability to adapt attention based on task-specific requirements and sample variations. To address these limitations, this paper presents an efficient RGB-D scene understanding model that performs a range of tasks, including semantic segmentation, instance segmentation, orientation estimation, panoptic segmentation, and scene classification. The proposed model incorporates an enhanced fusion encoder, which effectively leverages redundant information from both RGB and depth inputs. For semantic segmentation, we introduce normalized focus channel layers and a context feature interaction layer, designed to mitigate issues such as shallow feature misguidance and insufficient local-global feature representation. The instance segmentation task benefits from a non-bottleneck 1D structure, which achieves superior contour representation with fewer parameters. Additionally, we propose a multi-task adaptive loss function that dynamically adjusts the learning strategy for different tasks based on scene variations. Extensive experiments on the NYUv2, SUN RGB-D, and Cityscapes datasets demonstrate that our approach outperforms existing methods in both segmentation accuracy and processing speed.

--------------------------------------------------------------------------------

## 编号: PR070
发表年份: 2025
作者: Bo-Wen Yin; Jiao-Long Cao; Ming-Ming Cheng; Qibin Hou
标题: DFormerv2: Geometry Self-Attention for RGBD Semantic Segmentation
期刊/出版物: IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR 2025)
摘要: Recent advances in scene understanding benefit a lot from depth maps because of the 3D geometry information, especially in complex conditions (e.g., low light and overexposed). Existing approaches encode depth maps along with RGB images and perform feature fusion between them to enable more robust predictions. Taking into account that depth can be regarded as a geometry supplement for RGB images, a straightforward question arises: Do we really need to explicitly encode depth information with neural networks as done for RGB images? Based on this insight, in this paper, we investigate a new way to learn RGBD feature representations and present DFormerv2, a strong RGBD encoder that explicitly uses depth maps as geometry priors rather than encoding depth information with neural networks. Our goal is to extract the geometry clues from the depth and spatial distances among all the image patch tokens, which will then be used as geometry priors to allocate attention weights in self-attention. Extensive experiments demonstrate that DFormerv2 exhibits exceptional performance in various RGBD semantic segmentation benchmarks. Code is available at: https://github.com/VCIP-RGBD/DFormer.


DFormerv2 不再使用独立 Depth encoder，而是将 Depth 直接解释为 RGB 自注意力的几何先验。对每个 encoder stage，首先将 Depth map average-pool 到对应 patch resolution，以每个 patch 的平均深度 $z_{ij}$ 构造两两 Depth relationship：

$$
D_{ij,i'j'}=\left|z_{ij}-z_{i'j'}\right|,
$$

得到 $D\in\mathbb{R}^{HW\times HW}$；同时根据 patch 坐标构造 Manhattan spatial prior：

$$
S_{ij,i'j'}=|i-i'|+|j-j'|.
$$

Depth prior $D$ 与 spatial prior $S$ 经 learnable memory weights 融合得到 geometry prior $G\in\mathbb{R}^{HW\times HW}$。其核心 Geometry Self-Attention（GSA）不是将 Depth 编码成额外 feature，而是把 geometry prior 直接乘入 RGB self-attention：

$$
\mathrm{GeoAttn}(Q,K,V,G)
=
\left(
\mathrm{Softmax}(QK^\top)\odot\beta^G
\right)V,
$$

其中 $\beta\in(0,1)$ 为 geometry decay rate，$\beta^G$ 是真正作用于视觉 attention map 的 geometry decay mask。较大的几何距离会产生更强的 attention 衰减。论文进一步采用 horizontal/vertical decomposition 降低前三个 stage 的计算量，而 Stage 4 使用非分解 GSA；Depth 会被分别 pooling 到四个 stage 的尺度，因此四级 encoder 均受到 geometry prior 影响。作者对 $\beta$ 做过消融，并明确指出其用于控制 geometry prior 对 feature 的作用程度；默认不同 attention heads 使用不同 decay rates，并在线性区间 $[0.75,1.0)$ 内采样。

对 MMFR 最重要的是，DFormerv2 默认 Depth-derived geometry prior 始终可信，没有 modality reliability、Depth corruption detection、missing/noisy Depth handling 或 input-dependent geometry-strength control。由于 $G$ 直接调制 RGB attention，受损 Depth 会经“Depth $\rightarrow D\rightarrow G\rightarrow\beta^G$”链路直接错误修改 RGB token-pair attention，而不是仅引入一个可被后续 fusion 忽略的错误 Depth feature。论文 Table 7 进一步表明 Depth 对分类语义的提升较小，但对 foreground segmentation/object shape 明显有益，支持将其作用理解为 geometry/shape cue 而非普通 semantic modality。

从论文结构看，MMFR 的一个自然最小干预对象是 geometry decay mask。对于非分解 GSA，其核心形式为：

$$
\mathrm{GeoAttn}(Q,K,V,G)
=
\left(
\mathrm{Softmax}(QK^\top)\odot\beta^G
\right)V.
$$

论文原设 $\beta\in(0,1)$，因此 $\beta=1$ 并非 DFormerv2 的实际配置；但若将 MMFR 的动态控制扩展到 identity boundary $\beta_{\mathrm{eff}}=1$，则有：

$$
\beta_{\mathrm{eff}}^G=\mathbf{1},
$$

从而非分解 GSA 中 geometry modulation 被完全移除。对于 Stage 4，这时可恢复为不含 geometry prior 的普通 RGB self-attention；对于 Stage 1–3，由于原模型采用 horizontal/vertical decomposed self-attention，$\beta_{\mathrm{eff}}=1$ 后应更准确地描述为“退化为不含 geometry prior 的 decomposed RGB self-attention”，而不是严格意义上的 full vanilla self-attention。

因此 MMFR 更准确的目标是：在 Depth 不可靠时，使 geometry decay mask 逐渐趋近 identity，而不是将整个 attention 输出压向零。前三个 stage 实际需要控制 $\beta^{G_x}$ 与 $\beta^{G_y}$，Stage 4 则对应完整的 $\beta^G$。

**当前核对状态：** 方法级已确认最小干预位置为 $\beta^G$/geometry decay mask；代码级尚未确认，后续仍需检查源码中的 $G$、$G_x/G_y$、multi-head $\beta$ broadcast、Stage 1–3 decomposition、Stage 4 full GSA 以及 Depth invalid-value preprocessing 的具体实现。



----------------------------------

## 编号: PR089
发表年份: 2026
作者: Lekang Wen; Liang Liao; Jing Xiao; Mi Wang
标题: SGMA: Semantic-Guided Modality-Aware Segmentation for Remote Sensing with Incomplete Multimodal Data
期刊/出版物: IEEE Transactions on Geoscience and Remote Sensing, Volume 64, 2026
摘要: Multimodal semantic segmentation integrates complementary information from diverse sensors for remote sensing Earth observation. However, practical systems often encounter missing modalities due to sensor failures or incomplete coverage, termed Incomplete Multimodal Semantic Segmentation (IMSS). IMSS faces three key challenges: (1) multimodal imbalance, where dominant modalities suppress fragile ones; (2) intra-class variation in scale, shape, and orientation across modalities; and (3) cross-modal heterogeneity with conflicting cues producing inconsistent semantic responses. Existing methods rely on contrastive learning or joint optimization, which risk over-alignment, discarding modality-specific cues or imbalanced training, favoring robust modalities, while largely overlooking intra-class variation and cross-modal heterogeneity. To address these limitations, we propose the Semantic-Guided Modality-Aware (SGMA) framework, which ensures balanced multimodal learning while reducing intra-class variation and reconciling cross-modal inconsistencies through semantic guidance. SGMA introduces two complementary plug-and-play modules: (1) Semantic-Guided Fusion (SGF) module extracts multi-scale, class-wise semantic prototypes that capture consistent categorical representations across modalities, estimates per-modality robustness based on prototype-feature alignment, and performs adaptive fusion weighted by robustness scores to mitigate intra-class variation and cross-modal heterogeneity; (2) Modality-Aware Sampling (MAS) module leverages robustness estimations from SGF to dynamically reweight training samples, prioritizing challenging samples from fragile modalities to address modality imbalance. Extensive experiments across multiple datasets and backbones demonstrate that SGMA consistently outperforms state-of-the-art methods, with particularly significant improvements in fragile modalities.

核心方法 SGF 首先通过 modality-specific projector 和 class-aware semantic filter 从多模态多尺度特征构造全局 class prototypes，再以 semantic prototypes 为 query 对多模态局部特征执行 semantic-guided attention。经类别平均后形成 semantic reference feature，并在 Robustness Perceptron 中作为 query，与各模态 feature 进行第二次 MHA；其 attention weights 被直接解释为 modality robustness maps $r_m^i\in\mathbb{R}^{H_i\times W_i}$。因此 reliability 实际为 semantic-conditioned、pixel-wise、scale-wise、跨模态相对可靠性，并无显式 reliability supervision。其 robustness 通过 segmentation CE 端到端隐式学习。MAS 对 robustness 做 reciprocal inversion 并空间平均，获得每尺度 modality sampling probability，优先抽取 fragile modalities 形成额外训练分支；MAS 仅训练使用，推理移除。该方法与 MMFR 在“动态模态可靠性估计+可靠性驱动自适应控制”上高度相关，但 SGMA 作用于 encoder 后的多尺度语义融合，不建模连续 Depth corruption severity，也不进入 backbone 内部 geometry operator。其 reliability 是模态间 SoftMax 归一化的相对 semantic usefulness，而非 Depth 自身的绝对 geometry trustworthiness。故 MMFR 应避免采用“semantic prototype alignment → pixel-wise modality reliability → feature fusion”作为核心创新，而应突出“Depth degradation diagnosis → DFormerv2 geometry-path intervention”的结构与问题定义差异。

--------------------------------------------------------------------------------

## 编号: PR090
发表年份: 2026
作者: Jiaqi Tan; Xu Zheng; Yang Liu
标题: Towards Robust Multi-Modal Semantic Segmentation with Teacher-Student Framework and Hybrid Prototype Distillation
期刊/出版物: IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR 2026)
摘要: Multimodal semantic segmentation (MMSS) faces significant challenges in real-world applications due to incomplete, degraded, or missing sensor data. To address this, we propose RobustSeg, an efficient teacher-student framework that enhances model robustness under missing-modality conditions while maintaining strong performance in full-modality scenarios. RobustSeg adopts a feedback-based self-distillation paradigm consisting of two complementary stages. Firstly, we introduce Hybrid Prototype Distillation (HPD), which enables more reliable knowledge transfer of both cross-modal and modality-specific aspects. Concretely, combined with dominant-modality selection, HPD performs cross-modal semantic distillation with high-level semantic prototypes to reduce modality bias. Meanwhile, HPD conducts intra-class feature variation distillation for modality-specific structural details. Secondly, to enable the teacher model to gradually produce more balanced and robust modality representations, we make the student model provide feedback from the non-dominant modality to the teacher, benefiting the entire distillation process. Experiments on three datasets demonstrate that our method achieves state-of-the-art robustness (e.g., +2.40% missing-modality performance on DeLiVER) while causing almost no degradation in full-modality performance (only -0.1% mIoU). Moreover, evaluations using different backbones (AnySeg and CMNeXt) further validate the generalization ability of RobustSeg.

提出 RobustSeg，一种面向 missing-modality MMSS 的 feedback-based teacher–student self-distillation framework。Teacher 使用完整模态，Student 通过 Anymodal Dropout 接收部分缺失模态。核心 Hybrid Prototype Distillation（HPD）包含两部分：其一为 Cross-modal Prototype Distillation（CPD），首先利用 resize 后的 GT segmentation mask 对四级 modality feature 逐类平均，构造 $p\in\mathbb{R}^{C\times d}$ 的 GT-supervised class prototypes，再随机打乱 modality matching，通过 KL 在 Student 与 Teacher 的不同模态 prototype 间传递高层 semantic knowledge；其二为 dominant-modality IFV distillation，根据 class prototype 构造 Center Feature Map，并计算其与原 feature 的 cosine similarity 得到 IFV map，随后使用 ASM 根据 unimodal feature 与 fused feature 的 cosine similarity 选择 dominant modalities，只从 dominant branches 传递细节结构。Student loss 为 $L_{\mathrm{hp}}=L_{\mathrm{CE}}+\lambda L_{\mathrm{KL}}+\alpha L_{\mathrm{cp}}+\beta L_{\mathrm{ifv}}$。此外作者允许 Student 将 non-dominant modality information 反馈给 Teacher：冻结 Teacher dominant-modality encoders 与 fusion block，仅以较低 learning rate 更新 non-dominant encoders，Teacher feedback loss 为 $L_{\mathrm{feedback}}=L_{\mathrm{CE}}+L_{\mathrm{ifv}}$。实验不仅覆盖 arbitrary missing-modality combinations，还正式采用 sensor-failure benchmark 的 EMM（entire missing）、RMM（random/partial missing）和 NM（noisy modality），因此该工作已经包含 degraded/noisy modality robustness evaluation；但正文核心训练仍基于 Anymodal Dropout，并未进行显式 corruption type/severity estimation，也没有产生 pixel-wise Depth reliability、geometry trustworthiness 或针对 backbone geometry operator 的动态控制。与 MMFR 的主要重叠是 missing/noisy modality robustness 与跨模态 representation robustness；主要差异在于 PR090 属于 robustness-through-distillation，而 MMFR 应突出 Depth failure diagnosis、连续 geometry reliability 以及 reliability-conditioned DFormerv2 geometry-path intervention。

--------------------------------------------------------------------------------

