# MMFR 的 PR 编号文献整理清单

> 用途：根据已补全并核查的 PR 文献，重新整理其对 MMFR 的相关性。分级表示与当前 MMFR 主线的直接程度及防撞车重要性，不代表论文本身质量排名。
>
> 当前 MMFR 主线：**Depth 失效诊断 → geometry reliability → DFormerv2 geometry-path intervention → robust RGB-D semantic segmentation**。

------

## 一、核心文献：必须保留

| 等级 | 编号  | 处理建议                 | 与 MMFR 的关系                                               |
| ---- | ----- | ------------------------ | ------------------------------------------------------------ |
| S    | PR070 | 核心必引                 | DFormerv2 Geometry Self-Attention，是 MMFR 当前 backbone 与 geometry-path 干预的直接依据 |
| S    | PR029 | 核心撞车项               | 已实现 pixel-wise uncertainty → confidence → adaptive multimodal modulation，限制 MMFR 使用通用 uncertainty gating 作为创新 |
| S    | PR089 | 核心撞车项               | 已实现 pixel-wise / multi-scale modality robustness estimation，并据此进行 adaptive fusion |
| S    | PR090 | 核心 robustness baseline | 已系统研究 missing/noisy modality 下的 multimodal semantic segmentation robustness |
| A    | PR038 | 动态路由邻近工作         | RGB-D VSOD 中已有 modality-aware MoE、expert routing 和 gated fusion，但没有显式 Depth reliability |
| A    | PR053 | uncertainty 机制旁证     | RGB-D SLAM 中已有显式 uncertainty → confidence → downstream intervention |
| A    | PR165 | adaptive fusion 邻近工作 | RGB-D semantic segmentation 中已有 region-wise dynamic RGB-only / RGB-D fusion，但 gate 并非 Depth reliability |

其中 PR029、PR070、PR089、PR090 已足以构成当前 MMFR 最重要的防撞车核心文献组。 PR038、PR053、PR165 则分别约束 dynamic routing、uncertainty-conditioned computation 和 dynamic RGB-D fusion 等宽泛创新表述。

------

## 二、评价与模块设计参考

| 编号  | 主要用途                                                     | 不应误用为                           |
| ----- | ------------------------------------------------------------ | ------------------------------------ |
| PR024 | degradation profile estimation、任务驱动增强                 | Depth reliability 直接先例           |
| PR040 | 真实物理 degradation benchmark、condition-wise evaluation    | Depth corruption 方法                |
| PR042 | conditional robustification、conditioning 消融设计           | RGB-D semantic segmentation baseline |
| PR074 | cross-modal representation alignment 与 pretrained representation anchoring | modality reliability estimator       |

PR040 特别适合作为后续 MMFR corruption benchmark 和 condition-wise robustness reporting 的实验设计参考；PR042 适合作为 random/shuffled/fixed conditioning 等消融设计参考。

------

## 三、clean RGB-D / Geometry 背景文献

| 编号  | 保留原因                                                     |
| ----- | ------------------------------------------------------------ |
| PR059 | clean RGB-D scene understanding 与 cross-dimensional feature guidance |
| PR117 | Depth integrity / geometry prior 对 segmentation 的作用      |
| PR138 | Depth prior 与 semantic information 的结合                   |

### PR059 定位修正

PR059 不再放入 A- 核心方法组，调整为 **B：clean RGB-D background**。

其 adaptive mechanism 主要针对 multi-task loss，而不是 RGB/Depth reliability；论文也没有系统研究 noisy/missing Depth，因此主要用于说明 clean RGB-D 场景理解背景。

------

## 四、宽背景：需要时引用，不进入 MMFR 主线

### 退化、域适配与数据生成

```
PR028` `PR036` `PR043` `PR044
```

主要用于 adverse-condition、domain adaptation、degradation-aware learning 等背景。

### 结构、几何与分割方法

```
PR123` `PR140` `PR149
```

主要作为 structure-aware、geometry prior、boundary/shape modeling 的宽背景。

### 任务协同

```
PR158
```

用于 reconstruction 与 downstream segmentation 协同优化的跨任务参考。

------

## 五、建议移出 MMFR 专用主文献池

```
PR030` `PR058` `PR087` `PR092` `PR094` `PR105` `PR107` `PR133` `PR151` `PR168` `PR173` `PR178
```

这些工作与 MMFR 当前的：

**Depth failure → geometry reliability → DFormerv2 geometry-path intervention**

缺乏足够直接的机制联系。

其中：

- `PR087` 可保留为 uncertainty + geometry 的备用旁证；
- `PR151` 可保留为 depth-consistency reliability weighting 的备用旁证；
- `PR178` 虽属于 RGB-D SLAM，但不属于 Depth-failure semantic segmentation 主线。

------

## 六、最新分级总表

### S：核心与高风险撞车

```
PR029` `PR070` `PR089` `PR090
```

### A：重要邻近工作

```
PR038` `PR053` `PR165
```

### B+：评价与设计参考

```
PR024` `PR040` `PR042` `PR074
```

### B：clean RGB-D / geometry 背景

```
PR059` `PR117` `PR138
```

### C：宽背景

```
PR028` `PR036` `PR043` `PR044` `PR123` `PR140` `PR149` `PR158
```

### D+：备用旁证

```
PR087` `PR151
```

### D：移出 MMFR 主文献池

```
PR030` `PR058` `PR092` `PR094` `PR105` `PR107` `PR133` `PR168` `PR173` `PR178
```

------

## 七、原“需要补全文”状态更新

原清单中的 P0/P1 阅读任务已经基本完成，不再需要作为单独的待办部分保留。

### 已完成核心核查

```
PR029` `PR038` `PR040` `PR042` `PR053` `PR059` `PR070` `PR074` `PR089` `PR090` `PR165
```

其中：

- `PR070`：论文方法级已经足够，后续模块冻结前仍需结合源码确认具体 geometry-path 最小干预位置；
- 其余条目当前阶段无需继续展开全文整理。

------

## 八、当前应避免的宽泛创新表述

根据现有 PR 文献，MMFR 后续不宜单独把以下内容描述为核心创新：

- dynamic multimodal fusion；
- RGB-only / RGB-D 动态选择；
- modality-aware MoE / expert routing；
- pixel-wise modality uncertainty；
- uncertainty → confidence；
- confidence-guided feature modulation；
- pixel-wise modality robustness → adaptive fusion；
- missing/noisy modality robustness；
- generic conditional gating。

这些思想已经分别在 PR029、PR038、PR042、PR053、PR089、PR090、PR165 等工作中出现。

MMFR 后续更应集中在：

> **针对 corrupted / missing / misleading Depth，显式判断 Depth-derived geometry 是否可信，并利用该 geometry reliability 直接控制 DFormerv2 内部 geometry computation。**

------

## 九、元数据修正

### PR070

DFormerv2 应为：

**CVPR 2025**

而不是 CVPR 2026。

DOI：

`10.1109/CVPR52734.2025.01802`。

### PR089

已有 IEEE TGRS 正式条目。

DOI：

`10.1109/TGRS.2026.3692798`。

### PR040

作者应为：

```
Hyunseo Koh
```

而不是 `Hyunsuh Koh`。

### PR059

当前补全资料中的出版信息存在冲突：

- 一处记录为 `Knowledge-Based Systems, Volume 327, 2025, Article 114107`；
- 另一处根据所读 PDF 首页记录为 `arXiv:2603.07570v1`，并标注 `Preprint submitted to Elsevier`。

因此 PR059 的最终正式出版信息暂标记为 **待核验**，不影响目前的方法相关性分级。

------

## 十、编号覆盖检查

新版清单继续覆盖原 PR 合集全部 34 个编号：

`PR024` `PR028` `PR029` `PR030` `PR036` `PR038` `PR040` `PR042` `PR043` `PR044` `PR053` `PR058` `PR059` `PR070` `PR074` `PR087` `PR089` `PR090` `PR092` `PR094` `PR105` `PR107` `PR117` `PR123` `PR133` `PR138` `PR140` `PR149` `PR151` `PR158` `PR165` `PR168` `PR173` `PR178`。

------

## 十一、当前阶段用途

新版 PR 清单当前只承担三个作用：

1. 确定哪些论文必须进入 MMFR 核心 related work；
2. 确定哪些已有机制不能再作为宽泛 novelty；
3. 为下一阶段模块设计提供防撞车边界。

当前阶段**不在本清单中继续展开具体模块设计或公式推导**。
