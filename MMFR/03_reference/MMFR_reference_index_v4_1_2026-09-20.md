# MMFR v4.1 文献总索引与实施溯源表
## 135条可识别论文｜保留既有编号｜支持双向追溯

> 版本日期：2026-09-20。与 `../01_research/MMFR_research_blueprint_v4_1_2026-09-20.md` 配套。  
> 本索引恢复文献可检索性，不重新收窄 v4 的方法选择空间。  
> “附件记录”与“本轮原始来源核验”分开保存；缺少字段不依据记忆补造。  
> 已有 P0/P1/P2 可用，无需重新上传；具体忠实复现需读取所选版本的原论文/代码。

主蓝图：[../01_research/MMFR_research_blueprint_v4_1_2026-09-20.md](../01_research/MMFR_research_blueprint_v4_1_2026-09-20.md)。结构化索引：[MMFR_reference_registry_v4_1_2026-09-20.json](MMFR_reference_registry_v4_1_2026-09-20.json)。

## 1. 覆盖范围与编号规则

共有 **135 条带题名的可定位文献**：34 PR、69 RE、AI001–AI030，以及 MoSA、ANGA。PR059 的重复条目在同一编号下保存两条来源，不重复计数。另保留 **124 个只有编号、没有题名的旧归档条目**；RE067 是 S14 中的举例编号，不计作论文。

既有 PR/RE/AI 编号不因优先级变化重排；同一论文新增 arXiv/正式入口时补充到原编号。MoSA、ANGA 暂沿用原代号；本次没有新增 AI031 等编号。

**证据优先级按问题区分，而不是简单按文件新旧：** 项目当前结果以 S6 为直接记录；研发路线以 S5/v4.1 为准；论文机制以指定原版本为准，未重读者沿用已提供审计并标注；实际实现以本地代码/权重身份为准。S1/S2–S4 的旧路线禁令不重新生效。

## 2. 如何使用本索引

先按下面的读文献包选择资料，再按第3节步骤号定位依据；第5–8节可按论文编号反查全部对应步骤。每条记录保留原题名、原作者/年份/出版物字段、DOI/标识、原文件行号、网络核验范围与不可推导的结论。

**定位规则：** `S7:L24–L82` 表示原样 UTF-8 文件物理行，不是 PDF 页码或聊天工具行号；文件名、章节和 SHA-256 同时保留。首次具体复现时，需补入所选 PDF 的实际章节/公式/页码或仓库 commit/函数；当前没有的定位绝不补造。

**原始材料可携带：** 整包 `source_materials/` 中放入14份附件的原样副本；单独下载本索引时，S编号仍能通过原文件名找到来源。

## 3. 子计划读文献包

<a id="PACK-T"></a>
### PACK-T｜T：训练知识迁移
必读：[AI019](#ref-AI019)、[PR090](#ref-PR090)。
按问题追加：[AI024](#ref-AI024)、[PR089](#ref-PR089)、[AI022](#ref-AI022)、[ANGA](#ref-ANGA)、[AI026](#ref-AI026)、[PR074](#ref-PR074)。
已给材料：S2、S3、S7、S8；对应步骤：T.1、T.2、T.3、T.4、E1.1、M.T。
要提取：明确 teacher/student 输入、冻结参数、mask 位置、蒸馏/重建目标、视图数、监督标签与额外算力。
不可默认：不把 MaskMentor 重建和 RobustSeg 原型蒸馏统称为同一个输出 KD；不能混用 PR090 多版本。

<a id="PACK-R"></a>
### PACK-R｜R：输入补偿与任务几何提示
必读：[AI023](#ref-AI023)、[AI017](#ref-AI017)、[PR070](#ref-PR070)。
按问题追加：[AI016](#ref-AI016)、[AI029](#ref-AI029)、[AI030](#ref-AI030)、[RE266](#ref-RE266)、[RE043](#ref-RE043)、[RE049](#ref-RE049)、[PR024](#ref-PR024)、[PR158](#ref-PR158)。
已给材料：S2、S3、S7、S9、S10；对应步骤：R.1、R.2、R.3、M.R、E0.2。
要提取：区分 RGB-only prompt / RGB+Depth recovery；输入量纲、上下界、损失/正则、冻结和普通插值流程。
不可默认：不照搬 0–255、TV/L1 系数或将小残差上限视为整幅缺失补偿能力。

<a id="PACK-F"></a>
### PACK-F｜F：残差适配与特征补偿
必读：[AI024](#ref-AI024)、[MoSA](#ref-MoSA)、[PR070](#ref-PR070)。
按问题追加：[AI009](#ref-AI009)、[PR029](#ref-PR029)、[PR089](#ref-PR089)、[AI025](#ref-AI025)、[AI026](#ref-AI026)、[PR038](#ref-PR038)、[PR165](#ref-PR165)。
已给材料：S3、S4、S7、S8；对应步骤：F.1、F.2、F.3、M.F、E0.2。
要提取：复制编码器与小 adapter 的区别、插入层、输入信号、参数冻结、初始化、梯度通路和部署保留分支。
不可默认：不假设 DFormerv2 存在独立 Depth encoder；不预告零开销；不虚称轻量简化为原文复现。

<a id="PACK-G"></a>
### PACK-G｜G：几何使用与质量信号
必读：[PR070](#ref-PR070)、[AI016](#ref-AI016)、[AI021](#ref-AI021)。
按问题追加：[PR029](#ref-PR029)、[PR089](#ref-PR089)、[AI020](#ref-AI020)、[MoSA](#ref-MoSA)、[PR042](#ref-PR042)、[AI018](#ref-AI018)、[AI027](#ref-AI027)、[AI028](#ref-AI028)、[AI029](#ref-AI029)、[AI030](#ref-AI030)。
已给材料：S1、S2、S3、S4、S7；对应步骤：G.1、G.2、G.3、M.G、E0.2、E0.3。
要提取：softmax 前后位置、Depth/位置项、分解支持域、信号定义、局部/常数对照、无动作边界与可部署输入。
不可默认：不将预测不确定性等同于 Depth 物理可靠性；Oracle 不作为所有学习式方法准入证。

<a id="PACK-EVAL"></a>
### PACK-EVAL｜评价、对照与独立确认
必读：[RE326](#ref-RE326)、[AI010](#ref-AI010)、[AI011](#ref-AI011)、[PR070](#ref-PR070)。
按问题追加：[PR040](#ref-PR040)、[PR042](#ref-PR042)、[AI003](#ref-AI003)、[AI004](#ref-AI004)、[AI005](#ref-AI005)、[AI013](#ref-AI013)、[AI014](#ref-AI014)、[RE266](#ref-RE266)。
已给材料：S6、S1、S5、S9、S10；对应步骤：E0.1、E0.4、V.1、V.2、V.3、V.4、V.5、V.6。
要提取：保留 frozen 条件表与分割混淆矩阵、项目数据划分、选模规则、训练预算、组映射和独立测试条件。
不可默认：M6、196 组/10000次是项目协议；训练随机性与测试重采样分开。

<a id="PACK-SCENE"></a>
### PACK-SCENE｜矿下背景与场景特化
必读：[RE326](#ref-RE326)、[RE240](#ref-RE240)、[RE026](#ref-RE026)。
按问题追加：[RE094](#ref-RE094)、[RE049](#ref-RE049)、[RE266](#ref-RE266)、[RE418](#ref-RE418)、[RE452](#ref-RE452)、[RE360](#ref-RE360)、[RE043](#ref-RE043)。
已给材料：S9、S10、S12；对应步骤：S.1、E3.1、R.3。
要提取：矿下/露天/地表/管廊差异、真实传感器、退化证据、任务和评价单位、方法可迁移部分。
不可默认：矿区地表裂缝不是地下巷道；RGB 检测/SLAM 可靠性不是 MUSeg Depth 失效真值。

## 4. 实验步骤 → 论文 → 原材料 → 参数出处

以下34条把正文的工作分解为可引用的子计划入口。所有对照组合和未标为原文实参的配置均属于项目设计；列出近邻论文不是声称该论文已经做过同样的完整实验。

<a id="step-E0.1"></a>
### E0.1｜冻结数据划分、样本/故障清单与历史结果
论文：[RE326](#ref-RE326)、[AI010](#ref-AI010)、[AI011](#ref-AI011)、[PR040](#ref-PR040)。
资料定位：S6 §1、§5（24–40、180–199）；S1 §S1.1、冻结账本（228–244、284–302）。
网络：W5、W23。
**参数归属：** 318 样本、40/40 评价、旧 10-condition 来自项目文件；哈希/白名单为工程约束。
**不可越界：** 数据集论文不提供本项目 A2 checkpoint 或主验证成绩。

<a id="step-E0.2"></a>
### E0.2｜核对几何入口、Depth/validity 缩放与无动作等价
论文：[PR070](#ref-PR070)、[AI029](#ref-AI029)、[AI030](#ref-AI030)。
资料定位：S1 §S1.3–S1.4（325–382）；S7 PR070（24–82）。
网络：W1、W4、W21、W22。
**参数归属：** 按本地代码确定量纲、张量与缩放；32 样本仅是 v4 排错建议。
**不可越界：** 公开代码核验不能代替本地 fork；不把 PR070 摘要中的 pooling/公式直接当实施代码。

<a id="step-E0.3"></a>
### E0.3｜区分合成状态、任务系数和推理可用信息
论文：[AI018](#ref-AI018)、[AI020](#ref-AI020)、[AI021](#ref-AI021)、[AI027](#ref-AI027)、[AI028](#ref-AI028)。
资料定位：S1 §必须分开的量、S1.2（133–154、304–323）；S4 §四–七。
网络：W13、W14、W18、W19。
**参数归属：** 质量目标和现有头的 14/2 维契约以本地记录核实；不补造未知损失权重。
**不可越界：** 推理不能读取 clean Depth、真实 shift 或注入 severity；任务 gate 不是自动校准的物理概率。

<a id="step-E0.4"></a>
### E0.4｜补齐 matched clean-control 与无动作续训对照
论文：[AI019](#ref-AI019)、[AI024](#ref-AI024)、[PR090](#ref-PR090)。
资料定位：S6 §2–4（50–174）；S1 §S2.1（386–404）；S5 §E0。
网络：W6、W3、W7、W8。
**参数归属：** 原训练预算来自项目；新旧分支/随机性/额外步数必须记录。
**不可越界：** 历史 B0 不是正式配对 control；control 可与候选研发并行，不阻断所有实验。

<a id="step-R.1"></a>
### R.1｜朴素任务驱动 Depth 补偿或 RGB 几何提示
论文：[AI023](#ref-AI023)、[AI017](#ref-AI017)、[PR070](#ref-PR070)。
资料定位：S2 §二（55–76）、§五；S3 §二（38–69）；S5 §5.1（163–187）。
网络：W2、W9、W1。
**参数归属：** 残差接口与分割监督为 v4 候选；输出范围、网络宽度和归一化待本地确定。
**不可越界：** GeomPrompt 与 Recovery 输入不同；本文式子是适配示意，非原论文逐式复现。

<a id="step-R.2"></a>
### R.2｜可靠区域保持、局部缺失与整幅缺失的补偿幅度
论文：[AI023](#ref-AI023)、[AI017](#ref-AI017)、[AI016](#ref-AI016)。
资料定位：S5 §5.1（175–186）；S6 §7（252–280）。
网络：W2、W9。
**参数归属：** 幅度上限、valid 区辅助约束及缺失模式处理是本项目待验证选择。
**不可越界：** 不能照搬其他量纲的 0–255，不能将替代几何记为真实传感观测。

<a id="step-R.3"></a>
### R.3｜用分割结果和关系变化评估补偿而非只看像素误差
论文：[PR070](#ref-PR070)、[AI023](#ref-AI023)、[RE266](#ref-RE266)、[RE043](#ref-RE043)、[RE049](#ref-RE049)。
资料定位：S1 §M1（639–670）；S9/S10 按对应编号；S5 §6、§10。
网络：W1、W4、W2。
**参数归属：** 区域/边界定义、关系抽样和辅助 loss 权重是研发建议，不是以上论文的统一配置。
**不可越界：** 跨任务矿下增强仅支撑任务评价动机；不声称其结果可直接转移。

<a id="step-F.1"></a>
### F.1｜在 RGB 主特征中加入残差适配
论文：[AI024](#ref-AI024)、[MoSA](#ref-MoSA)、[AI009](#ref-AI009)、[PR070](#ref-PR070)。
资料定位：S3 §二（38–69）及 ConD 段落；S4 §九（380 起）；S5 §5.2（188–207）。
网络：W3、W10、W1。
**参数归属：** 小型 adapter 的宽度、插入 stage 与共享方式属于本项目简化方案。
**不可越界：** ConD 原始复制编码器不是小 adapter；简化应标为 inspired/适配，不标忠实复现。

<a id="step-F.2"></a>
### F.2｜身份初始化、梯度启动与冻结策略核验
论文：[AI024](#ref-AI024)、[PR070](#ref-PR070)。
资料定位：S5 §5.2（188–207）、§E1（291–311）。
网络：W3、W1、W4。
**参数归属：** 零输出投影是候选初始化；本地测试 logits、梯度与一步 optimizer update。
**不可越界：** 初始化近似/精确 identity 必须实测；不声称训练后 clean 必然保持。

<a id="step-F.3"></a>
### F.3｜质量条件、抑噪或专家扩展与无条件版本竞争
论文：[MoSA](#ref-MoSA)、[PR029](#ref-PR029)、[PR089](#ref-PR089)、[AI025](#ref-AI025)、[AI026](#ref-AI026)、[PR038](#ref-PR038)、[PR165](#ref-PR165)。
资料定位：S2 §二；S3 §二；S4 §九及 MoSA 段落；S8 PR038/PR165。
网络：W10、W11、W12、W17、W16。
**参数归属：** 只在 F 朴素版不足时单变量扩展；空间 gate、专家数、插入层不预先照搬。
**不可越界：** 先保留无质量头 F；不同任务的 routing/抑噪不等于可直接拼接。

<a id="step-G.1"></a>
### G.1｜保留位置项、单独控制 Depth 几何关系
论文：[PR070](#ref-PR070)、[AI016](#ref-AI016)、[AI029](#ref-AI029)、[AI030](#ref-AI030)。
资料定位：S1 §S2.2、S2.6（406–466、523–554）；S5 §5.3（208–229）。
网络：W1、W4、W21、W22。
**参数归属：** v4 的 $B_S+a\odot B_D$ 是项目动作简写；包含原混合权重的具体实现以本地核验为准。
**不可越界：** 不要关闭整个 attention 或虚构 N² 结构；常数 Depth 的零差关系不能靠乘法补出信息。

<a id="step-G.2"></a>
### G.2｜validity、常数、A2 预测和任务学习信号对照
论文：[AI020](#ref-AI020)、[AI021](#ref-AI021)、[PR089](#ref-PR089)、[PR029](#ref-PR029)、[MoSA](#ref-MoSA)、[AI027](#ref-AI027)。
资料定位：S1 §S2.5、S2.7；S2 §四；S4 §五–七；S5 §5.3。
网络：W13、W14、W12、W11、W10、W18。
**参数归属：** 信号组合、局部/全局对照及是否保留 A2 头是项目选择。
**不可越界：** 引用表示机制相邻；不宣称每篇论文都做过这里完整的对照矩阵。

<a id="step-G.3"></a>
### G.3｜Oracle 与匹配短训共同判断可用性
论文：[AI016](#ref-AI016)、[PR029](#ref-PR029)、[PR070](#ref-PR070)。
资料定位：S1 §S2.2–S2.3（406–489）；S5 §1、§E1、§10。
网络：W11、W1、W4。
**参数归属：** 真实 shift / Q 只用于标明 privileged 的诊断；短训预算为 v4 建议。
**不可越界：** 零训练 Oracle 失败不否定整个方法家族；错位补偿不是仅搬代码位置就形成创新。

<a id="step-T.1"></a>
### T.1｜完整/受损成对视图与教师输出指导
论文：[AI019](#ref-AI019)、[PR090](#ref-PR090)。
资料定位：S2 §二（55–76）；S3 §二（38–69）；S5 §5.4（230–260）。
网络：W6、W7、W8。
**参数归属：** v4 的 clean CE + corrupt CE + KD 是候选组合；各权重未预填，须单独登记。
**不可越界：** MaskMentor 的遮罩重建与 RobustSeg 的原型蒸馏不等同于该简化输出 KD。

<a id="step-T.2"></a>
### T.2｜固定教师来源并防止无条件复制教师错误
论文：[AI019](#ref-AI019)、[PR090](#ref-PR090)、[AI022](#ref-AI022)。
资料定位：S5 §5.4（251–260）；S3 §二。
网络：W6、W7、W8、W15。
**参数归属：** 用 A2 clean 输出作为低成本教师、教师筛选与替换均为本项目研发建议。
**不可越界：** 教师未被证明最优；训练标签继续监督学生；换教师单列变化与成本。

<a id="step-T.3"></a>
### T.3｜以 Cpair 分离多视图暴露和蒸馏收益
论文：[AI019](#ref-AI019)、[PR090](#ref-PR090)、[AI024](#ref-AI024)。
资料定位：S5 §5.4、§E1（230–260、291–311）；S1 §S2.7。
网络：W6、W7、W8、W3。
**参数归属：** 成对视图与学生更新数匹配属于本项目对照；教师计算单列。
**不可越界：** 不能只因 epoch 相同就称算力相同；不把成熟论文当六组设计的直接出处。

<a id="step-T.4"></a>
### T.4｜按需扩展原型、采样或梯度优化
论文：[PR090](#ref-PR090)、[PR089](#ref-PR089)、[ANGA](#ref-ANGA)、[AI026](#ref-AI026)、[PR074](#ref-PR074)。
资料定位：S3 §二及 ANGA 段落；S2 SGMA 段落；S8 PR074。
网络：W7、W8、W12、W20、W16。
**参数归属：** 扩展仅在基础 T 不足时设计；原型数、阈值、课程和 loss 需查固定版本原文。
**不可越界：** 不把全部近邻模块堆成默认网络；同作者多版本不可混合复现。

<a id="step-E1.1"></a>
### E1.1｜运行 C0/R/F/G/Cpair/T 六组首轮筛选
论文：[AI019](#ref-AI019)、[AI023](#ref-AI023)、[AI024](#ref-AI024)、[PR070](#ref-PR070)、[PR090](#ref-PR090)、[MoSA](#ref-MoSA)。
资料定位：S5 §E1（291–311）。
网络：W6、W2、W3、W1、W4、W7、W8、W10。
**参数归属：** 六组构成是 v4 研发设计，不是某篇论文的实验矩阵。
**不可越界：** R/F/G 对 C0；T 对 Cpair；冻结方式或 exposure 变化要列为方法差异。

<a id="step-E1.2"></a>
### E1.2｜匹配初始化、训练步数与选择机会
论文：[AI019](#ref-AI019)、[AI024](#ref-AI024)、[PR090](#ref-PR090)。
资料定位：S1 §训练预算（562–570）；S5 §E1、§8。
网络：W6、W3、W7、W8。
**参数归属：** 额外 50 epochs 是 S1/S5 的提案；新模块 LR 和主干 LR 分开登记。
**不可越界：** 未学会的新分支不能按一次零增益就判方法族无效；也不能无限调到正数才汇报。

<a id="step-E1.3"></a>
### E1.3｜快速排错与全量冻结主验证分离
论文：[AI010](#ref-AI010)、[AI011](#ref-AI011)、[RE326](#ref-RE326)。
资料定位：S5 §E1（303–309）；S6 §1、§5。
网络：W23、W5。
**参数归属：** 32 固定样本仅排错；318 开发样本及 10 条件继承项目。
**不可越界：** 小样本、轻量 evaluator 不作为正式涨点证据。

<a id="step-E2.1"></a>
### E2.1｜选少量互补候选并做共同基础/+A/+B/+A+B
论文：[AI023](#ref-AI023)、[AI024](#ref-AI024)、[PR090](#ref-PR090)、[PR070](#ref-PR070)。
资料定位：S5 §E2（312–321）。
网络：W2、W3、W7、W8、W1、W4。
**参数归属：** 最多两条优先路线、最多两个组合是预算管理建议。
**不可越界：** T+R/T+F 只是工作假说；以收益/成本选，不以相似性或命名选。

<a id="step-E3.1"></a>
### E3.1｜朴素借用与场景/模型特化直接对比
论文：[PR070](#ref-PR070)、[RE326](#ref-RE326)、[AI023](#ref-AI023)、[AI024](#ref-AI024)、[MoSA](#ref-MoSA)。
资料定位：S5 §6、§E3（261–277、322–337）。
网络：W1、W4、W5、W2、W3、W10。
**参数归属：** 特化菜单与改造增量定义为本项目设计。
**不可越界：** 借鉴组件贡献与新增改造贡献分开；移到 backbone 内不自动增加新颖性。

<a id="step-E4.1"></a>
### E4.1｜冻结少量入围模型的完整方法与独立确认
论文：[AI010](#ref-AI010)、[RE326](#ref-RE326)、[PR070](#ref-PR070)。
资料定位：S1 §S2.8（596–635）；S5 §E4、§11.2。
网络：W23、W5、W1、W4。
**参数归属：** 最终三 seed 等来自旧项目规划，非某篇基线的统一要求。
**不可越界：** 所有候选开发不是已完成验证；只对锁定版本做论文确认。

<a id="step-V.1"></a>
### V.1｜预登记 clean 约束下的鲁棒验证 selector
论文：[AI010](#ref-AI010)、[AI011](#ref-AI011)。
资料定位：S5 §8.1（346–369）。
网络：W23。
**参数归属：** 选择目标、候选集合、参考模型及 epsilon=0.50 起点均是项目建议。
**不可越界：** 历史 A2 不追溯重选；论文只是鲁棒评价背景，不是该 selector 的来源。

<a id="step-V.2"></a>
### V.2｜分开系统总收益与模块净收益
论文：[AI019](#ref-AI019)、[AI023](#ref-AI023)、[AI024](#ref-AI024)、[PR090](#ref-PR090)。
资料定位：S5 §8.2、§9.1（370–389）。
网络：W6、W2、W3、W7、W8。
**参数归属：** 同配方同预算的增量控制由项目定义；额外训练/教师成本同时记录。
**不可越界：** 普通强基线和朴素成熟适配都能胜出，不把全部收益包装成自研模块。

<a id="step-V.3"></a>
### V.3｜锁定 M6、clean、逐条件/逐类和成本
论文：[AI010](#ref-AI010)、[AI011](#ref-AI011)、[PR040](#ref-PR040)、[RE266](#ref-RE266)。
资料定位：S6 §5（180–199）；S1 §指标（246–280）；S5 §9。
网络：W23。
**参数归属：** M6 与旧门槛继承项目；约+1.0、进取+1.5–2.0为新研发目标。
**不可越界：** 不是论文给出的预期增益；不换分母/删条件制造涨点。

<a id="step-M.R"></a>
### M.R｜R 入围后检查补偿收益位置与可靠输入保持
论文：[AI023](#ref-AI023)、[AI017](#ref-AI017)、[PR070](#ref-PR070)、[RE266](#ref-RE266)。
资料定位：S5 §10；S1 §M1（639–670）。
网络：W2、W9、W1、W4。
**参数归属：** 区域分层/原图对照是项目机制分析；规模参数待登记。
**不可越界：** 修复看起来更好不等于分割提升；不能只挑成功样例。

<a id="step-M.F"></a>
### M.F｜F 入围后分离额外容量、适应和质量条件
论文：[AI024](#ref-AI024)、[MoSA](#ref-MoSA)、[AI025](#ref-AI025)。
资料定位：S5 §10；S1 §S2.7（572–594）。
网络：W3、W10、W17。
**参数归属：** 同规模 adapter 与有无质量信号由项目设置。
**不可越界：** 来自有效适应的收益可保留，但不强称可靠性诊断必不可少。

<a id="step-M.G"></a>
### M.G｜G 入围后执行状态替换与关系/效用分析
论文：[AI016](#ref-AI016)、[AI021](#ref-AI021)、[PR042](#ref-PR042)、[PR070](#ref-PR070)。
资料定位：S1 §M2（672–748）；S5 §10。
网络：W14、W1、W4。
**参数归属：** 均值、空间置换、反向图、固定采样/分箱是本项目条件替换设计。
**不可越界：** 图/相关性不是现实因果证明；Oracle 未必最高；完整六组并非全由 PR042 原文提供。

<a id="step-M.T"></a>
### M.T｜T 入围后比较无蒸馏/输出/可选特征蒸馏
论文：[AI019](#ref-AI019)、[PR090](#ref-PR090)。
资料定位：S5 §10、§5.4。
网络：W6、W7、W8。
**参数归属：** 比较层级与匹配视图为项目设计；具体原型/特征损失需再读选定原文。
**不可越界：** 教师指导收益须超出成对视图暴露收益。

<a id="step-V.4"></a>
### V.4｜选择普通强基线与相近鲁棒适配
论文：[PR070](#ref-PR070)、[AI003](#ref-AI003)、[AI004](#ref-AI004)、[AI005](#ref-AI005)、[AI019](#ref-AI019)、[AI022](#ref-AI022)、[AI024](#ref-AI024)、[PR090](#ref-PR090)、[MoSA](#ref-MoSA)。
资料定位：S5 §11.1（423–429）；S1 §S2.7。
网络：W1、W4、W24、W25、W6、W15、W3、W7、W8、W10。
**参数归属：** 只选代表性入围方法；同骨干机制比较与跨骨干系统比较分表。
**不可越界：** 预训练/decoder升级允许，但须对照升级后的无MMFR版本。

<a id="step-V.5"></a>
### V.5｜组级配对统计、多 seed 与封存 test
论文：[RE326](#ref-RE326)、[AI010](#ref-AI010)。
资料定位：S1 §S2.8（596–635）；S5 §11.2（431–443）。
网络：W5、W23。
**参数归属：** 196 location groups、10000次、3 seed 来自项目旧规划；测试组数不能照搬196。
**不可越界：** bootstrap不代替训练随机性；full-train并入的318张不再是独立验证集。

<a id="step-V.6"></a>
### V.6｜真正未见故障与外部数据复验
论文：[AI010](#ref-AI010)、[PR070](#ref-PR070)、[AI003](#ref-AI003)、[PR090](#ref-PR090)、[RE326](#ref-RE326)。
资料定位：S1 §S2.8（619–635）；S5 §11.2（439–443）。
网络：W23、W1、W4、W7、W8、W5。
**参数归属：** 未见故障选择、NYUv2适配和超参数冻结策略为项目计划。
**不可越界：** 整条训练链排除才是未见；跨数据集重训不叫零样本迁移。

<a id="step-S.1"></a>
### S.1｜矿下场景动机与特化证据
论文：[RE326](#ref-RE326)、[RE240](#ref-RE240)、[RE026](#ref-RE026)、[RE094](#ref-RE094)、[RE049](#ref-RE049)、[RE266](#ref-RE266)、[RE418](#ref-RE418)、[RE452](#ref-RE452)、[RE360](#ref-RE360)。
资料定位：S9/S10 按编号；S12 §一–二（5–52）；S5 §6。
网络：W5。
**参数归属：** 本项目类权重、区域与传感失效映射须由实际标签/样本验证。
**不可越界：** 露天矿、地表裂缝、SLAM、偏振、PPE检测仅相邻证据；不把它们写成MUSeg真实Depth失效标签。

## 5. PR 编号完整索引
34条；核心与其他文件合并，PR059重复来源保留

<a id="ref-PR024"></a>
### [PR024] ELVIS: Enhance Low-Light for Video Instance Segmentation in the Dark
原始书目：作者：Joanne Lin; Ruirui Lin; Yini Li; David Bull; Nantheera Anantrasirichai；年份（附件）：2026；出版物（附件）：IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR 2026)。
**DOI/标识：** 未提供／未核；按完整题名检索，禁止补造。
**原材料：** S8:L5–L13（编号: PR024）。
**v4.1角色：** 退化估计与下游任务驱动增强。
反查步骤：目前未列为必做步骤；保留检索，使用前补原文；读文献包：PACK-R。
适用边界：低照视频实例分割，不作为 Depth 修复网络的直接复现依据。
**核验状态：** 附件条目（非本轮原论文全文核验）；本轮未另行网络核验。
查找方式：复制本条完整题名检索；没有可确认 DOI 时不补造。
附件自带入口（未自动核验）：<https://joannelin168.github.io/research/ELVIS>。

<a id="ref-PR028"></a>
### [PR028] BiPA: Bilevel Prompt Adaptation for Underwater Instance Segmentation
原始书目：作者：Long Ma; Haoze Zheng; Yuhang Mao; Jinyuan Liu; Chengpei Xu; Xinwei Xue; Yi Wang; Xiangjian He; Weimin Wang；年份（附件）：2026；出版物（附件）：IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR 2026)。
**DOI/标识：** 未提供／未核；按完整题名检索，禁止补造。
**原材料：** S8:L13–L21（编号: PR028）。
**v4.1角色：** 检索保留／背景备用；不因 v4 放宽路线而自动升级为直接依据。
反查步骤：目前未列为必做步骤；保留检索，使用前补原文；读文献包：背景/归档。
**核验状态：** 附件条目（非本轮原论文全文核验）；本轮未另行网络核验。
查找方式：复制本条完整题名检索；没有可确认 DOI 时不补造。
附件自带入口（未自动核验）：<https://github.com/ZeAstra/BiPA>。

<a id="ref-PR029"></a>
### [PR029] Uncertainty-Aware Modality Fusion for Unaligned RGB-T Salient Object Detection
检索别名：UMFNet。
原始书目：作者：Mianzhao Wang; Fan Shi; Xu Cheng; Chen Jia; Shengyong Chen；年份（附件）：2026；出版物（附件）：IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR 2026)。
**DOI/标识：** 未提供／未核；按完整题名检索，禁止补造。
**原材料：** S7:L3–L14（编号: PR029）。
机制审计：S3（出现行：5, 22, 34, 49, 69, 182, 184, 204 等）。
**v4.1角色：** 局部不确定性与空间/通道调制近邻。
反查步骤：F.3、G.2、G.3；读文献包：PACK-F、PACK-G。
适用边界：RGB-T 显著目标检测，不等于 RGB-D 语义分割；confidence 不是已标定的传感质量。
**核验状态：** 附件全文审计的转述；本轮是否另核原文见 W 记录；本轮另见 W11。
原始检索入口：<https://openaccess.thecvf.com/content/CVPR2026/html/Wang_Uncertainty-Aware_Modality_Fusion_for_Unaligned_RGB-T_Salient_Object_Detection_CVPR_2026_paper.html>（官方检索题名/摘要可见；直开失败）。

<a id="ref-PR030"></a>
### [PR030] From 2D Alignment to 3D Plausibility: Unifying Heterogeneous 2D Priors and Penetration-Free Diffusion for Occlusion-Robust Two-Hand Reconstruction
原始书目：作者：Gaoge Han; Yongkang Cheng; Zhe Chen; Shaoli Huang; Tongliang Liu；年份（附件）：2026；出版物（附件）：IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR 2026)。
**DOI/标识：** 未提供／未核；按完整题名检索，禁止补造。
**原材料：** S8:L21–L31（编号: PR030）。
**v4.1角色：** 检索保留／背景备用；不因 v4 放宽路线而自动升级为直接依据。
反查步骤：目前未列为必做步骤；保留检索，使用前补原文；读文献包：背景/归档。
**核验状态：** 附件条目（非本轮原论文全文核验）；本轮未另行网络核验。
查找方式：复制本条完整题名检索；没有可确认 DOI 时不补造。
附件自带入口（未自动核验）：<https://gaogehan.github.io/A2P/>。

<a id="ref-PR036"></a>
### [PR036] Concept-Aware LoRA for Domain-Aligned Segmentation Dataset Generation
原始书目：作者：Minho Park; Sunghyun Park; Jungsoo Lee; Hyojin Park; Kyuwoong Hwang; Fatih Porikli; Jaegul Choo; Sungha Choi；年份（附件）：2026；出版物（附件）：IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR 2026)。
**DOI/标识：** 未提供／未核；按完整题名检索，禁止补造。
**原材料：** S8:L31–L39（编号: PR036）。
**v4.1角色：** 检索保留／背景备用；不因 v4 放宽路线而自动升级为直接依据。
反查步骤：目前未列为必做步骤；保留检索，使用前补原文；读文献包：背景/归档。
**核验状态：** 附件条目（非本轮原论文全文核验）；本轮未另行网络核验。
查找方式：复制本条完整题名检索；没有可确认 DOI 时不补造。

<a id="ref-PR038"></a>
### [PR038] M4-SAM: Multi-Modal Mixture-of-Experts with Memory-Augmented SAM for RGB-D Video Salient Object Detection
检索别名：M4-SAM。
原始书目：作者：Jiyuan Liu; Jia Lin; Xiaofei Zhou; Runmin Cong; Deyang Liu; Zhi Liu；年份（附件）：2026；出版物（附件）：IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR 2026)。
**DOI/标识：** 未提供／未核；按完整题名检索，禁止补造。
**原材料：** S8:L39–L89（编号: PR038）。
**v4.1角色：** 模态专家路由、卷积适配与多层门控。
反查步骤：F.3；读文献包：PACK-F。
适用边界：视频显著目标检测；模态身份路由不等于模态质量路由。
**核验状态：** 附件条目（非本轮原论文全文核验）；本轮未另行网络核验。
查找方式：复制本条完整题名检索；没有可确认 DOI 时不补造。

<a id="ref-PR040"></a>
### [PR040] CLP: A Real-World Dataset of Contaminated Lens Protectors for Robust Semantic Segmentation
检索别名：CLP。
原始书目：作者：Sungyong Park; Sooyoung Choi; Hyunsuh Koh; Youngjae Choi; Heewon Kim；年份（附件）：2026；出版物（附件）：IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR 2026)。
**DOI/标识：** 未提供／未核；按完整题名检索，禁止补造。
**原材料：** S8:L89–L116（编号: PR040）。
**v4.1角色：** 真实污染与条件化评价设计参考。
反查步骤：E0.1、V.3；读文献包：PACK-EVAL。
适用边界：不是 Depth 故障生成器；不能推出 MUSeg 的合成退化等于真实故障。 作者字段原条目写 Hyunsuh Koh，S11:209–217 明确纠正为 Hyunseo Koh；这里保留原值并记录更正，正式引用需核原作者页。
**核验状态：** 附件条目（非本轮原论文全文核验）；本轮未另行网络核验。
查找方式：复制本条完整题名检索；没有可确认 DOI 时不补造。

<a id="ref-PR042"></a>
### [PR042] Robust Promptable Video Object Segmentation
检索别名：MoGA。
原始书目：作者：Sohyun Lee; Yeho Gwon; Lukas Hoyer; Konrad Schindler; Christos Sakaridis; Suha Kwak；年份（附件）：2026；出版物（附件）：IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR 2026)。
**DOI/标识：** 未提供／未核；按完整题名检索，禁止补造。
**原材料：** S8:L116–L172（编号: PR042）。
**v4.1角色：** 条件信号是否有效的消融思路。
反查步骤：M.G；读文献包：PACK-G、PACK-EVAL。
适用边界：视频目标分割跨任务参考；MMFR 的 shuffle、预算与局部统计是本项目设计。
**核验状态：** 附件条目（非本轮原论文全文核验）；本轮未另行网络核验。
查找方式：复制本条完整题名检索；没有可确认 DOI 时不补造。
附件自带入口（未自动核验）：<https://sohyun-l.github.io/RobustPVOS_project_page/>。

<a id="ref-PR043"></a>
### [PR043] 3M-TI: High-Quality Mobile Thermal Imaging via Calibration-free Multi-Camera Cross-Modal Diffusion
原始书目：作者：Minchong Chen; Xiaoyun Yuan; Junzhe Wan; Jianing Zhang; Jun Zhang；年份（附件）：2026；出版物（附件）：IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR 2026)。
**DOI/标识：** 未提供／未核；按完整题名检索，禁止补造。
**原材料：** S8:L172–L180（编号: PR043）。
**v4.1角色：** 检索保留／背景备用；不因 v4 放宽路线而自动升级为直接依据。
反查步骤：目前未列为必做步骤；保留检索，使用前补原文；读文献包：背景/归档。
**核验状态：** 附件条目（非本轮原论文全文核验）；本轮未另行网络核验。
查找方式：复制本条完整题名检索；没有可确认 DOI 时不补造。
附件自带入口（未自动核验）：<https://github.com/work-submit/3MTI>。

<a id="ref-PR044"></a>
### [PR044] Heuristic Self-Paced Learning for Domain Adaptive Semantic Segmentation under Adverse Conditions
原始书目：作者：Shiqin Wang; Haoyang Chen; Huaizhou Huang; Yinkan He; Dongfang Sun; Xiaoqing Chen; Xingyu Liu; Zheng Wang; Kaiyan Zhao；年份（附件）：2026；出版物（附件）：IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR 2026)。
**DOI/标识：** 未提供／未核；按完整题名检索，禁止补造。
**原材料：** S8:L180–L188（编号: PR044）。
**v4.1角色：** 检索保留／背景备用；不因 v4 放宽路线而自动升级为直接依据。
反查步骤：目前未列为必做步骤；保留检索，使用前补原文；读文献包：背景/归档。
**核验状态：** 附件条目（非本轮原论文全文核验）；本轮未另行网络核验。
查找方式：复制本条完整题名检索；没有可确认 DOI 时不补造。

<a id="ref-PR053"></a>
### [PR053] VarSplat: Uncertainty-aware 3D Gaussian Splatting for Robust RGB-D SLAM
检索别名：VarSplat。
原始书目：作者：Anh Thuan Tran; Jana Kosecka；年份（附件）：2026；出版物（附件）：IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR 2026)。
**DOI/标识：** 未提供／未核；按完整题名检索，禁止补造。
**原材料：** S8:L188–L283（编号: PR053）。
**v4.1角色：** 不确定性影响下游处理的跨任务参考。
反查步骤：目前未列为必做步骤；保留检索，使用前补原文；读文献包：背景/归档。
适用边界：RGB-D SLAM，不是语义分割的直接对照。
**核验状态：** 附件条目（非本轮原论文全文核验）；本轮未另行网络核验。
查找方式：复制本条完整题名检索；没有可确认 DOI 时不补造。

<a id="ref-PR058"></a>
### [PR058] Spatial-SAM: Spatially Consistent 3D Electron Microscopy Segmentation with SDF Memory and Semi-Supervised Learning
原始书目：作者：Yikai Huang; Renmin Han; Yuxuan Wang; Youcheng Cai; Ligang Liu；年份（附件）：2026；出版物（附件）：IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR 2026)。
**DOI/标识：** 未提供／未核；按完整题名检索，禁止补造。
**原材料：** S8:L283–L291（编号: PR058）。
**v4.1角色：** 检索保留／背景备用；不因 v4 放宽路线而自动升级为直接依据。
反查步骤：目前未列为必做步骤；保留检索，使用前补原文；读文献包：背景/归档。
**核验状态：** 附件条目（非本轮原论文全文核验）；本轮未另行网络核验。
查找方式：复制本条完整题名检索；没有可确认 DOI 时不补造。
附件自带入口（未自动核验）：<https://github.com/Giluir/Spatial-SAM>。

<a id="ref-PR059"></a>
### [PR059] Efficient RGB-D Scene Understanding via Multi-task Adaptive Learning and Cross-dimensional Feature Guidance
原始书目：作者：Guodong Sun; Junjie Liu; Gaoyang Zhang; Bo Wu; Yang Zhang；年份（附件）：2026；出版物（附件）：Knowledge-Based Systems, Volume 327, 2025, Article 114107。
**DOI/标识：** doi: `10.48550/arXiv.2603.07570`（本轮核对 arXiv 页面列示）。
**原材料：** S7:L14–L23（编号: PR059）；S8:L291–L331（编号: PR059）。
多来源差异：保留两条原记录，不覆盖。字段详情见 JSON；出版信息问题见 K03。
**v4.1角色：** 正常 RGB-D 多任务场景理解。
反查步骤：目前未列为必做步骤；保留检索，使用前补原文；读文献包：背景/归档。
适用边界：多任务 loss 自适应不等于模态可靠性；出版信息冲突见核验台账。
本轮外核补充（W26；不覆盖上面的原记录）：原入口作者：Guodong Sun; Junjie Liu; Gaoyang Zhang; Bo Wu; Yang Zhang；原入口年份：2026；状态：arXiv:2603.07570v1；正式刊物对应仍冲突。
**核验状态：** 附件条目（非本轮原论文全文核验）；本轮另见 W26。
原始检索入口：<https://arxiv.org/abs/2603.07570>（已核题名/作者/提交日期及预印本 DOI）。

<a id="ref-PR070"></a>
### [PR070] DFormerv2: Geometry Self-Attention for RGBD Semantic Segmentation
检索别名：DFormerv2。
原始书目：作者：Bo-Wen Yin; Jiao-Long Cao; Ming-Ming Cheng; Qibin Hou；年份（附件）：2025；出版物（附件）：IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR 2025)。
**DOI/标识：** doi: `10.1109/CVPR52734.2025.01802`（v3 索引继承；本轮未独立核对注册信息）。
**原材料：** S7:L23–L83（编号: PR070）。
**v4.1角色：** 骨干信息使用路径；实现前核对本地几何算子。
反查步骤：E0.2、R.1、R.3、F.1、F.2、G.1、G.3、E1.1、E2.1、E3.1、E4.1、M.R、M.G、V.4、V.6；读文献包：PACK-R、PACK-F、PACK-G、PACK-EVAL。
适用边界：论文表述、公开代码与本地 fork 分开；不假设存在对称 Depth encoder。
**核验状态：** 附件条目（非本轮原论文全文核验）；本轮另见 W1、W4。
原始检索入口：<https://raw.githubusercontent.com/VCIP-RGBD/DFormer/main/models/encoders/DFormerv2.py>（已读取公开代码片段）。
附件自带入口（未自动核验）：<https://github.com/VCIP-RGBD/DFormer>。

<a id="ref-PR074"></a>
### [PR074] A Mixed Diet Makes DINO An Omnivorous Vision Encoder
原始书目：作者：Rishabh Kabra; Maks Ovsjanikov; Drew A. Hudson; Ye Xia; Skanda Koppula; Andre Araujo; Joao Carreira; Niloy J. Mitra；年份（附件）：2026；出版物（附件）：IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR 2026)。
**DOI/标识：** 未提供／未核；按完整题名检索，禁止补造。
**原材料：** S8:L331–L389（编号: PR074）。
**v4.1角色：** 预训练表征锚定与跨模态对齐。
反查步骤：T.4；读文献包：PACK-T。
适用边界：具体训练迁移需重读对应原文；不能据摘要补造配置。
**核验状态：** 附件条目（非本轮原论文全文核验）；本轮未另行网络核验。
查找方式：复制本条完整题名检索；没有可确认 DOI 时不补造。
附件自带入口（未自动核验）：<https://github.com/google-deepmind/representations4d>。

<a id="ref-PR087"></a>
### [PR087] GeoGuide: Hierarchical Geometric Guidance for Open-Vocabulary 3D Semantic Segmentation
原始书目：作者：Xujing Tao; Chuxin Wang; Yubo Ai; Zhixin Cheng; Zhuoyuan Li; Liangsheng Liu; Yujia Chen; Xinjun Li; Qiao Li; Wenfei Yang; Tianzhu Zhang；年份（附件）：2026；出版物（附件）：IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR 2026)。
**DOI/标识：** 未提供／未核；按完整题名检索，禁止补造。
**原材料：** S8:L389–L397（编号: PR087）。
**v4.1角色：** 检索保留／背景备用；不因 v4 放宽路线而自动升级为直接依据。
反查步骤：目前未列为必做步骤；保留检索，使用前补原文；读文献包：背景/归档。
**核验状态：** 附件条目（非本轮原论文全文核验）；本轮未另行网络核验。
查找方式：复制本条完整题名检索；没有可确认 DOI 时不补造。

<a id="ref-PR089"></a>
### [PR089] SGMA: Semantic-Guided Modality-Aware Segmentation for Remote Sensing with Incomplete Multimodal Data
检索别名：SGMA。
原始书目：作者：Lekang Wen; Liang Liao; Jing Xiao; Mi Wang；年份（附件）：2026；出版物（附件）：IEEE Transactions on Geoscience and Remote Sensing, Volume 64, 2026。
**DOI/标识：** doi: `10.1109/TGRS.2026.3692798`（v3 索引继承；本轮未独立核对注册信息）。
**原材料：** S7:L83–L94（编号: PR089）。
机制审计：S2（出现行：4, 62, 72, 141, 177, 233, 252, 293 等）。
**v4.1角色：** 多尺度语义引导可靠性、融合与脆弱模态采样。
反查步骤：F.3、G.2、T.4；读文献包：PACK-T、PACK-F、PACK-G。
适用边界：遥感多模态分割；原型/尺度和数据模态不能照搬。
**核验状态：** 附件全文审计的转述；本轮是否另核原文见 W 记录；本轮另见 W12。
原始检索入口：<https://ieeexplore.ieee.org/abstract/document/11517493>（官方检索条目可见；直开失败）。

<a id="ref-PR090"></a>
### [PR090] Towards Robust Multi-Modal Semantic Segmentation with Teacher-Student Framework and Hybrid Prototype Distillation
检索别名：RobustSeg。
原始书目：作者：Jiaqi Tan; Xu Zheng; Yang Liu；年份（附件）：2026；出版物（附件）：IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR 2026)。
**DOI/标识：** 未提供／未核；按完整题名检索，禁止补造。
**原材料：** S7:L94–L105（编号: PR090）。
机制审计：S3（出现行：5, 47, 65, 83, 158, 229, 283, 391 等）。
**v4.1角色：** 教师—学生与混合原型蒸馏的直接任务近邻。
反查步骤：E0.4、T.1、T.2、T.3、T.4、E1.1、E1.2、E2.1、V.2、M.T、V.4、V.6；读文献包：PACK-T。
适用边界：正式 CVPR 条目与同作者 arXiv 多版本须区分；不混用模块名与损失。
**核验状态：** 附件全文审计的转述；本轮是否另核原文见 W 记录；本轮另见 W7、W8。
原始检索入口：<https://openaccess.thecvf.com/content/CVPR2026/html/Tan_Towards_Robust_Multi-Modal_Semantic_Segmentation_with_Teacher-Student_Framework_and_Hybrid_CVPR_2026_paper.html>（官方检索题名/摘要可见；直开失败）。

<a id="ref-PR092"></a>
### [PR092] DIMOS: Disentangling Instance-level Moving Object Segmentation
原始书目：作者：Hongxiang Huang; Hongwei Ren; Xiaopeng Lin; Yulong Huang; Zeke Xie; Bojun Cheng；年份（附件）：2026；出版物（附件）：IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR 2026)。
**DOI/标识：** 未提供／未核；按完整题名检索，禁止补造。
**原材料：** S8:L397–L406（编号: PR092）。
**v4.1角色：** 检索保留／背景备用；不因 v4 放宽路线而自动升级为直接依据。
反查步骤：目前未列为必做步骤；保留检索，使用前补原文；读文献包：背景/归档。
**核验状态：** 附件条目（非本轮原论文全文核验）；本轮未另行网络核验。
查找方式：复制本条完整题名检索；没有可确认 DOI 时不补造。

<a id="ref-PR094"></a>
### [PR094] MARIS: Marine Open-Vocabulary Instance Segmentation
原始书目：作者：Bingyu Li; Feiyu Wang; Da Zhang; Zhiyuan Zhao; Junyu Gao; Xuelong Li；年份（附件）：2026；出版物（附件）：IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR 2026)。
**DOI/标识：** 未提供／未核；按完整题名检索，禁止补造。
**原材料：** S8:L406–L414（编号: PR094）。
**v4.1角色：** 检索保留／背景备用；不因 v4 放宽路线而自动升级为直接依据。
反查步骤：目前未列为必做步骤；保留检索，使用前补原文；读文献包：背景/归档。
**核验状态：** 附件条目（非本轮原论文全文核验）；本轮未另行网络核验。
查找方式：复制本条完整题名检索；没有可确认 DOI 时不补造。

<a id="ref-PR105"></a>
### [PR105] B³-Seg: Camera-Free, Training-Free 3DGS Segmentation via Analytic EIG and Beta-Bernoulli Bayesian Updates
原始书目：作者：Hiromichi Kamata; Samuel Arthur Munro; Fuminori Homma；年份（附件）：2026；出版物（附件）：IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR 2026)。
**DOI/标识：** 未提供／未核；按完整题名检索，禁止补造。
**原材料：** S8:L414–L422（编号: PR105）。
**v4.1角色：** 检索保留／背景备用；不因 v4 放宽路线而自动升级为直接依据。
反查步骤：目前未列为必做步骤；保留检索，使用前补原文；读文献包：背景/归档。
**核验状态：** 附件条目（非本轮原论文全文核验）；本轮未另行网络核验。
查找方式：复制本条完整题名检索；没有可确认 DOI 时不补造。

<a id="ref-PR107"></a>
### [PR107] VGGT-Segmentor: Geometry-Enhanced Cross-View Segmentation
原始书目：作者：Yulu Gao; Bohao Zhang; Zongheng Tang; Jitong Liao; Wenjun Wu; Si Liu；年份（附件）：2026；出版物（附件）：IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR 2026)。
**DOI/标识：** 未提供／未核；按完整题名检索，禁止补造。
**原材料：** S8:L422–L430（编号: PR107）。
**v4.1角色：** 检索保留／背景备用；不因 v4 放宽路线而自动升级为直接依据。
反查步骤：目前未列为必做步骤；保留检索，使用前补原文；读文献包：背景/归档。
**核验状态：** 附件条目（非本轮原论文全文核验）；本轮未另行网络核验。
查找方式：复制本条完整题名检索；没有可确认 DOI 时不补造。

<a id="ref-PR117"></a>
### [PR117] High-Precision Dichotomous Image Segmentation via Depth Integrity-Prior and Fine-Grained Patch Strategy
原始书目：作者：Xianjie Liu; Keren Fu; Qijun Zhao；年份（附件）：2026；出版物（附件）：IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR 2026)。
**DOI/标识：** 未提供／未核；按完整题名检索，禁止补造。
**原材料：** S8:L430–L438（编号: PR117）。
**v4.1角色：** 深度完整性与几何分割背景。
反查步骤：目前未列为必做步骤；保留检索，使用前补原文；读文献包：背景/归档。
适用边界：仅作背景；具体模态失效机制待原文核实。
**核验状态：** 附件条目（非本轮原论文全文核验）；本轮未另行网络核验。
查找方式：复制本条完整题名检索；没有可确认 DOI 时不补造。

<a id="ref-PR123"></a>
### [PR123] Best Segmentation Buddies for Image-Shape Correspondence
原始书目：作者：Itai Lang; Dongwei Lyu; Dale Decatur; Rana Hanocka；年份（附件）：2026；出版物（附件）：IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR 2026)。
**DOI/标识：** 未提供／未核；按完整题名检索，禁止补造。
**原材料：** S8:L438–L446（编号: PR123）。
**v4.1角色：** 检索保留／背景备用；不因 v4 放宽路线而自动升级为直接依据。
反查步骤：目前未列为必做步骤；保留检索，使用前补原文；读文献包：背景/归档。
**核验状态：** 附件条目（非本轮原论文全文核验）；本轮未另行网络核验。
查找方式：复制本条完整题名检索；没有可确认 DOI 时不补造。
附件自带入口（未自动核验）：<https://threedle.github.io/bsb/>。

<a id="ref-PR133"></a>
### [PR133] Hilbert Curve-Based Attention Enabling Topology-Preserving Image Tensor Representation for Semantic Segmentation Network
原始书目：作者：Linkang Xu; Gang Li; Yue Song; Xiangxin Ji；年份（附件）：2026；出版物（附件）：IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR 2026)。
**DOI/标识：** 未提供／未核；按完整题名检索，禁止补造。
**原材料：** S8:L446–L454（编号: PR133）。
**v4.1角色：** 检索保留／背景备用；不因 v4 放宽路线而自动升级为直接依据。
反查步骤：目前未列为必做步骤；保留检索，使用前补原文；读文献包：背景/归档。
**核验状态：** 附件条目（非本轮原论文全文核验）；本轮未另行网络核验。
查找方式：复制本条完整题名检索；没有可确认 DOI 时不补造。
附件自带入口（未自动核验）：<https://github.com/mumu-k/TPSegformer>。

<a id="ref-PR138"></a>
### [PR138] Beyond Appearance: Camouflaged Object Detection via Geometric Structure
原始书目：作者：Jinyu Han; Changguang Wu; Fuming Sun; Jinhui Tang；年份（附件）：2026；出版物（附件）：IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR 2026)。
**DOI/标识：** 未提供／未核；按完整题名检索，禁止补造。
**原材料：** S8:L454–L462（编号: PR138）。
**v4.1角色：** 几何与语义互补的背景。
反查步骤：目前未列为必做步骤；保留检索，使用前补原文；读文献包：背景/归档。
适用边界：不由题名推定已有失效训练。
**核验状态：** 附件条目（非本轮原论文全文核验）；本轮未另行网络核验。
查找方式：复制本条完整题名检索；没有可确认 DOI 时不补造。

<a id="ref-PR140"></a>
### [PR140] Structure-Aware Representation Distillation for Tiny-Dense Object Segmentation
原始书目：作者：Xuesong Liu; Anke Xu; Wenbo Cao; Emmett Ientilucci；年份（附件）：2026；出版物（附件）：IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR 2026)。
**DOI/标识：** 未提供／未核；按完整题名检索，禁止补造。
**原材料：** S8:L462–L470（编号: PR140）。
**v4.1角色：** 检索保留／背景备用；不因 v4 放宽路线而自动升级为直接依据。
反查步骤：目前未列为必做步骤；保留检索，使用前补原文；读文献包：背景/归档。
**核验状态：** 附件条目（非本轮原论文全文核验）；本轮未另行网络核验。
查找方式：复制本条完整题名检索；没有可确认 DOI 时不补造。
附件自带入口（未自动核验）：<https://github.com/liuuuuuuxuesong/SARD>。

<a id="ref-PR149"></a>
### [PR149] D-Convexity: A Unified Differentiable Convex Shape Prior via Quasi-Concavity for Data-driven Image Segmentation
原始书目：作者：Shengzhe Chen; Hao Yan；年份（附件）：2026；出版物（附件）：IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR 2026)。
**DOI/标识：** 未提供／未核；按完整题名检索，禁止补造。
**原材料：** S8:L470–L478（编号: PR149）。
**v4.1角色：** 检索保留／背景备用；不因 v4 放宽路线而自动升级为直接依据。
反查步骤：目前未列为必做步骤；保留检索，使用前补原文；读文献包：背景/归档。
**核验状态：** 附件条目（非本轮原论文全文核验）；本轮未另行网络核验。
查找方式：复制本条完整题名检索；没有可确认 DOI 时不补造。

<a id="ref-PR151"></a>
### [PR151] MV3DIS: Multi-View Mask Matching via 3D Guides for Zero-Shot 3D Instance Segmentation
原始书目：作者：Yibo Zhao; Yigong Zhang; Jin Xie；年份（附件）：2026；出版物（附件）：IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR 2026)。
**DOI/标识：** 未提供／未核；按完整题名检索，禁止补造。
**原材料：** S8:L478–L486（编号: PR151）。
**v4.1角色：** 检索保留／背景备用；不因 v4 放宽路线而自动升级为直接依据。
反查步骤：目前未列为必做步骤；保留检索，使用前补原文；读文献包：背景/归档。
**核验状态：** 附件条目（非本轮原论文全文核验）；本轮未另行网络核验。
查找方式：复制本条完整题名检索；没有可确认 DOI 时不补造。

<a id="ref-PR158"></a>
### [PR158] Joint Spectral Image Reconstruction and Semantic Segmentation with Cooperative Unfolding
原始书目：作者：Zijun He; Ping Wang; Xiaodong Wang; Chang Chen; Xin Yuan；年份（附件）：2026；出版物（附件）：IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR 2026)。
**DOI/标识：** 未提供／未核；按完整题名检索，禁止补造。
**原材料：** S8:L486–L494（编号: PR158）。
**v4.1角色：** 重建与语义任务联合设计的跨任务参考。
反查步骤：目前未列为必做步骤；保留检索，使用前补原文；读文献包：PACK-R。
适用边界：光谱任务；本项目的深度补偿损失仍需单独设计。
**核验状态：** 附件条目（非本轮原论文全文核验）；本轮未另行网络核验。
查找方式：复制本条完整题名检索；没有可确认 DOI 时不补造。
附件自带入口（未自动核验）：<https://github.com/zjhe02/CRSDUN>。

<a id="ref-PR165"></a>
### [PR165] REL-SF4PASS: Panoramic Semantic Segmentation with REL Depth Representation and Spherical Fusion
检索别名：REL-SF4PASS。
原始书目：作者：Xuewei Li; Xinghan Bao; Zhimin Chen; Xi Li；年份（附件）：2026；出版物（附件）：IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR 2026)。
**DOI/标识：** 未提供／未核；按完整题名检索，禁止补造。
**原材料：** S8:L494–L549（编号: PR165）。
**v4.1角色：** 区域自适应融合与几何表征。
反查步骤：F.3；读文献包：PACK-F。
适用边界：全景 ERP 几何；区域 gate 并非 Depth reliability。
**核验状态：** 附件条目（非本轮原论文全文核验）；本轮未另行网络核验。
查找方式：复制本条完整题名检索；没有可确认 DOI 时不补造。

<a id="ref-PR168"></a>
### [PR168] Unlocking 3D Affordance Segmentation with 2D Semantic Knowledge
原始书目：作者：Yu Huang; Zelin Peng; Changsong Wen; Xiaokang Yang; Wei Shen；年份（附件）：2026；出版物（附件）：IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR 2026)。
**DOI/标识：** 未提供／未核；按完整题名检索，禁止补造。
**原材料：** S8:L549–L557（编号: PR168）。
**v4.1角色：** 检索保留／背景备用；不因 v4 放宽路线而自动升级为直接依据。
反查步骤：目前未列为必做步骤；保留检索，使用前补原文；读文献包：背景/归档。
**核验状态：** 附件条目（非本轮原论文全文核验）；本轮未另行网络核验。
查找方式：复制本条完整题名检索；没有可确认 DOI 时不补造。

<a id="ref-PR173"></a>
### [PR173] Photo-Guided Tooth Segmentation on 3D Oral Scan Model
原始书目：作者：Shaojie Zhuang; Guangshun Wei; Jiangxin He; Yuanfeng Zhou；年份（附件）：2026；出版物（附件）：IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR 2026)。
**DOI/标识：** 未提供／未核；按完整题名检索，禁止补造。
**原材料：** S8:L557–L565（编号: PR173）。
**v4.1角色：** 检索保留／背景备用；不因 v4 放宽路线而自动升级为直接依据。
反查步骤：目前未列为必做步骤；保留检索，使用前补原文；读文献包：背景/归档。
**核验状态：** 附件条目（非本轮原论文全文核验）；本轮未另行网络核验。
查找方式：复制本条完整题名检索；没有可确认 DOI 时不补造。

<a id="ref-PR178"></a>
### [PR178] SGAD-SLAM: Splatting Gaussians at Adjusted Depth for Better Radiance Fields in RGBD SLAM
原始书目：作者：Pengchong Hu; Zhizhong Han；年份（附件）：2026；出版物（附件）：IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR 2026)。
**DOI/标识：** 未提供／未核；按完整题名检索，禁止补造。
**原材料：** S8:L565–L573（编号: PR178）。
**v4.1角色：** 检索保留／背景备用；不因 v4 放宽路线而自动升级为直接依据。
反查步骤：目前未列为必做步骤；保留检索，使用前补原文；读文献包：背景/归档。
**核验状态：** 附件条目（非本轮原论文全文核验）；本轮未另行网络核验。
查找方式：复制本条完整题名检索；没有可确认 DOI 时不补造。
附件自带入口（未自动核验）：<https://machineperceptionlab.github.io/SGAD-SLAM-Project>。

## 6. RE 编号完整索引
69条；保留场景/跨任务边界，不把背景全部升级为直接机制

<a id="ref-RE023"></a>
### [RE023] Enhancing SCSegamba with Fracture Orientation Consistency Loss for Robust Rock Fracture Segmentation in Deep Underground Mining
原始书目：作者：Q. L. Zhang; H. S. Wang; J. L. Pan; Y. B. Tao; Y. Feng；年份（附件）：2026；出版物（附件）：APPLIED SCIENCES-BASEL。
**DOI/标识：** 未提供／未核；按完整题名检索，禁止补造。
**原材料：** S10:L5–L13（编号: RE023）。
**v4.1角色：** 检索保留／背景备用；不因 v4 放宽路线而自动升级为直接依据。
反查步骤：目前未列为必做步骤；保留检索，使用前补原文；读文献包：背景/归档。
**核验状态：** 附件条目（非本轮原论文全文核验）；本轮未另行网络核验。
查找方式：复制本条完整题名检索；没有可确认 DOI 时不补造。

<a id="ref-RE026"></a>
### [RE026] LFR-CMT-3D: Lipschitz-regularized multi-modal fusion for robust object detection in open-pit mines
检索别名：LFR-CMT-3D。
原始书目：作者：H. R. Zhang; Y. Liu; Y. Fang; P. Q. Liu; J. D. Yang; X. X. Zhang；年份（附件）：2026；出版物（附件）：MEASUREMENT SCIENCE AND TECHNOLOGY。
**DOI/标识：** doi: `10.1088/1361-6501/ae58c7`（附件记录；本轮未独立核对注册信息）。
**原材料：** S9:L3–L48（编号: RE026）。
**v4.1角色：** 物理退化先验、局部权重与恢复协同。
反查步骤：S.1；读文献包：PACK-SCENE。
适用边界：露天矿检测；经验粉尘函数不自动适用于地下 RGB-D。
**核验状态：** 附件条目（非本轮原论文全文核验）；本轮未另行网络核验。
DOI检索入口：<https://doi.org/10.1088/1361-6501/ae58c7>（按附件标识生成，未等同成功解析）。

<a id="ref-RE043"></a>
### [RE043] Task-Aware Low-Light Image Enhancement Method for Underground Coal Mine Monitoring
原始书目：作者：Z. R. Yan; Y. R. Li; H. W. Wang; Z. X. Jin; L. Tao; Y. D. Geng；年份（附件）：2026；出版物（附件）：SENSORS。
**DOI/标识：** 未提供／未核；按完整题名检索，禁止补造。
**原材料：** S10:L13–L21（编号: RE043）。
**v4.1角色：** 矿下任务感知低照增强。
反查步骤：R.3；读文献包：PACK-R、PACK-SCENE。
适用边界：RGB 检测任务，借鉴任务驱动思想不照搬 Depth 模块。
**核验状态：** 附件条目（非本轮原论文全文核验）；本轮未另行网络核验。
查找方式：复制本条完整题名检索；没有可确认 DOI 时不补造。

<a id="ref-RE049"></a>
### [RE049] M-SURE: Enhanced and reliable safety monitoring in low-light mines
检索别名：M-SURE。
原始书目：作者：Y. Xin; J. Ding; Z. Zhang；年份（附件）：2026；出版物（附件）：ADVANCED ENGINEERING INFORMATICS。
**DOI/标识：** doi: `10.1016/j.aei.2026.104702`（附件记录；本轮未独立核对注册信息）。
**原材料：** S10:L21–L169（编号: RE049）。
**v4.1角色：** 低照矿下可靠感知与预测过置信。
反查步骤：R.3、S.1；读文献包：PACK-R、PACK-SCENE。
适用边界：RGB PPE 检测；决策概率校准不能替代 Depth 质量验证。
**核验状态：** 附件条目（非本轮原论文全文核验）；本轮未另行网络核验。
DOI检索入口：<https://doi.org/10.1016/j.aei.2026.104702>（按附件标识生成，未等同成功解析）。

<a id="ref-RE081"></a>
### [RE081] Concepts for autonomous navigation of underground mine face haulage equipment using depth cameras
原始书目：作者：J. Sottile; S. Rose; A. Rajvanshi; S. Schafrik; Z. Agioutantis; M. Sizintsev; H. P. Chiu；年份（附件）：2026；出版物（附件）：INTERNATIONAL JOURNAL OF COAL SCIENCE & TECHNOLOGY。
**DOI/标识：** 未提供／未核；按完整题名检索，禁止补造。
**原材料：** S10:L169–L177（编号: RE081）。
**v4.1角色：** 检索保留／背景备用；不因 v4 放宽路线而自动升级为直接依据。
反查步骤：目前未列为必做步骤；保留检索，使用前补原文；读文献包：背景/归档。
**核验状态：** 附件条目（非本轮原论文全文核验）；本轮未另行网络核验。
查找方式：复制本条完整题名检索；没有可确认 DOI 时不补造。

<a id="ref-RE086"></a>
### [RE086] A real-time semantic segmentation algorithm for visual SLAM in coal mine inspection robots on edge devices
原始书目：作者：L. Shi; X. Q. Shen；年份（附件）：2026；出版物（附件）：ALEXANDRIA ENGINEERING JOURNAL。
**DOI/标识：** 未提供／未核；按完整题名检索，禁止补造。
**原材料：** S10:L177–L185（编号: RE086）。
**v4.1角色：** 检索保留／背景备用；不因 v4 放宽路线而自动升级为直接依据。
反查步骤：目前未列为必做步骤；保留检索，使用前补原文；读文献包：背景/归档。
**核验状态：** 附件条目（非本轮原论文全文核验）；本轮未另行网络核验。
查找方式：复制本条完整题名检索；没有可确认 DOI 时不补造。

<a id="ref-RE094"></a>
### [RE094] Zero-Shot Polarization-Intensity Physical Fusion Monocular Depth Estimation for High Dynamic Range Scenes
原始书目：作者：R. H. Rao; Z. Z. Ouyang; S. Chen; L. Chen; G. Q. Huang; C. C. Cui；年份（附件）：2026；出版物（附件）：PHOTONICS。
**DOI/标识：** doi: `10.3390/photonics13030268`（附件记录；本轮未独立核对注册信息）。
**原材料：** S9:L48–L196（编号: RE094）。
**v4.1角色：** 物理有效性与可靠性条件输入处理。
反查步骤：S.1；读文献包：PACK-SCENE。
适用边界：偏振—强度深度估计；不是像素级学习式 Depth reliability。
**核验状态：** 附件条目（非本轮原论文全文核验）；本轮未另行网络核验。
DOI检索入口：<https://doi.org/10.3390/photonics13030268>（按附件标识生成，未等同成功解析）。

<a id="ref-RE117"></a>
### [RE117] Underground construction surveillance image enhancement for hydropower engineering
原始书目：作者：B. Lu; S. Chen; L. N. Niu; B. W. Nie; K. Y. Cao; Y. Chen; L. Z. Luo；年份（附件）：2026；出版物（附件）：ADVANCED ENGINEERING INFORMATICS。
**DOI/标识：** 未提供／未核；按完整题名检索，禁止补造。
**原材料：** S10:L185–L194（编号: RE117）。
**v4.1角色：** 检索保留／背景备用；不因 v4 放宽路线而自动升级为直接依据。
反查步骤：目前未列为必做步骤；保留检索，使用前补原文；读文献包：背景/归档。
**核验状态：** 附件条目（非本轮原论文全文核验）；本轮未另行网络核验。
查找方式：复制本条完整题名检索；没有可确认 DOI 时不补造。

<a id="ref-RE120"></a>
### [RE120] SIRI-YOLO: A Foreign Object Detection Method for Belt Conveyors in High-Entropy Underground Scenes
原始书目：作者：Y. Liu; R. G. Xue; Z. X. Zhao; J. P. Xiao；年份（附件）：2026；出版物（附件）：ENTROPY。
**DOI/标识：** 未提供／未核；按完整题名检索，禁止补造。
**原材料：** S10:L194–L202（编号: RE120）。
**v4.1角色：** 检索保留／背景备用；不因 v4 放宽路线而自动升级为直接依据。
反查步骤：目前未列为必做步骤；保留检索，使用前补原文；读文献包：背景/归档。
**核验状态：** 附件条目（非本轮原论文全文核验）；本轮未另行网络核验。
查找方式：复制本条完整题名检索；没有可确认 DOI 时不补造。

<a id="ref-RE131"></a>
### [RE131] RGB-D semantic mapping for underground robotic inspection using an attention-enhanced and boundary-refined DeepLabv3+network
原始书目：作者：G. L. Liang；年份（附件）：2026；出版物（附件）：INDUSTRIAL ROBOT-THE INTERNATIONAL JOURNAL OF ROBOTICS RESEARCH AND APPLICATION。
**DOI/标识：** 未提供／未核；按完整题名检索，禁止补造。
**原材料：** S10:L202–L210（编号: RE131）。
**v4.1角色：** 检索保留／背景备用；不因 v4 放宽路线而自动升级为直接依据。
反查步骤：目前未列为必做步骤；保留检索，使用前补原文；读文献包：背景/归档。
**核验状态：** 附件条目（非本轮原论文全文核验）；本轮未另行网络核验。
查找方式：复制本条完整题名检索；没有可确认 DOI 时不补造。

<a id="ref-RE135"></a>
### [RE135] MSFCR: MultiScale Fusion Method With Color Retention for Reference-Free Low-Light Enhancement in Underground Environments
原始书目：作者：Y. B. Li; P. Zhou; G. B. Zhou; H. Z. Wang; X. D. Yan; F. Jiang；年份（附件）：2026；出版物（附件）：IEEE TRANSACTIONS ON INDUSTRIAL INFORMATICS。
**DOI/标识：** 未提供／未核；按完整题名检索，禁止补造。
**原材料：** S10:L210–L218（编号: RE135）。
**v4.1角色：** 检索保留／背景备用；不因 v4 放宽路线而自动升级为直接依据。
反查步骤：目前未列为必做步骤；保留检索，使用前补原文；读文献包：背景/归档。
**核验状态：** 附件条目（非本轮原论文全文核验）；本轮未另行网络核验。
查找方式：复制本条完整题名检索；没有可确认 DOI 时不补造。

<a id="ref-RE140"></a>
### [RE140] New insights for enhancing the intelligence of coal mine: A two-stage method for unsupervised low-light image enhancement and lightweight detection
原始书目：作者：H. Li; B. J. Xie; X. X. Li; Z. Luan；年份（附件）：2026；出版物（附件）：ALEXANDRIA ENGINEERING JOURNAL。
**DOI/标识：** 未提供／未核；按完整题名检索，禁止补造。
**原材料：** S10:L218–L226（编号: RE140）。
**v4.1角色：** 检索保留／背景备用；不因 v4 放宽路线而自动升级为直接依据。
反查步骤：目前未列为必做步骤；保留检索，使用前补原文；读文献包：背景/归档。
**核验状态：** 附件条目（非本轮原论文全文核验）；本轮未另行网络核验。
查找方式：复制本条完整题名检索；没有可确认 DOI 时不补造。

<a id="ref-RE143"></a>
### [RE143] Visible light positioning with dynamic covariance and robust optimization fusion in shadowed underground mines
原始书目：作者：X. C. Kou; X. L. Hu; J. Y. Yu; L. Qin; F. Y. Wang; J. Li；年份（附件）：2026；出版物（附件）：OPTICS COMMUNICATIONS。
**DOI/标识：** 未提供／未核；按完整题名检索，禁止补造。
**原材料：** S10:L226–L234（编号: RE143）。
**v4.1角色：** 检索保留／背景备用；不因 v4 放宽路线而自动升级为直接依据。
反查步骤：目前未列为必做步骤；保留检索，使用前补原文；读文献包：背景/归档。
**核验状态：** 附件条目（非本轮原论文全文核验）；本轮未另行网络核验。
查找方式：复制本条完整题名检索；没有可确认 DOI 时不补造。

<a id="ref-RE153"></a>
### [RE153] DURAL: Degradation-Resistant Robust Adaptive Localization by LiDAR-Inertial-UWB-Wheel Fusion for Coal Mine Robots
原始书目：作者：K. Hu; M. G. Li; Z. W. Jin; C. Q. Tang; E. R. Y. Hu; G. B. Zhou；年份（附件）：2026；出版物（附件）：JOURNAL OF FIELD ROBOTICS。
**DOI/标识：** 未提供／未核；按完整题名检索，禁止补造。
**原材料：** S10:L234–L242（编号: RE153）。
**v4.1角色：** 检索保留／背景备用；不因 v4 放宽路线而自动升级为直接依据。
反查步骤：目前未列为必做步骤；保留检索，使用前补原文；读文献包：背景/归档。
**核验状态：** 附件条目（非本轮原论文全文核验）；本轮未另行网络核验。
查找方式：复制本条完整题名检索；没有可确认 DOI 时不补造。

<a id="ref-RE166"></a>
### [RE166] A multi-perspective perception decomposition and fusion framework for mine image enhancement
原始书目：作者：C. C. Fu; G. Y. Zhang; Y. C. Zong；年份（附件）：2026；出版物（附件）：ENGINEERING APPLICATIONS OF ARTIFICIAL INTELLIGENCE。
**DOI/标识：** 未提供／未核；按完整题名检索，禁止补造。
**原材料：** S10:L242–L250（编号: RE166）。
**v4.1角色：** 检索保留／背景备用；不因 v4 放宽路线而自动升级为直接依据。
反查步骤：目前未列为必做步骤；保留检索，使用前补原文；读文献包：背景/归档。
**核验状态：** 附件条目（非本轮原论文全文核验）；本轮未另行网络核验。
查找方式：复制本条完整题名检索；没有可确认 DOI 时不补造。

<a id="ref-RE171"></a>
### [RE171] Review of Essential Generic Technologies for Visual Perception in Underground Coal Mine Robots
原始书目：作者：Y. X. Du; J. H. Zhang; L. Liang; B. Song；年份（附件）：2026；出版物（附件）：JOURNAL OF FIELD ROBOTICS。
**DOI/标识：** 未提供／未核；按完整题名检索，禁止补造。
**原材料：** S10:L250–L258（编号: RE171）。
**v4.1角色：** 检索保留／背景备用；不因 v4 放宽路线而自动升级为直接依据。
反查步骤：目前未列为必做步骤；保留检索，使用前补原文；读文献包：背景/归档。
**核验状态：** 附件条目（非本轮原论文全文核验）；本轮未另行网络核验。
查找方式：复制本条完整题名检索；没有可确认 DOI 时不补造。

<a id="ref-RE183"></a>
### [RE183] LLE-YOLO: Adaptive Low-Light-Enhanced and Degradation-Aware Multi-Scale Attention Network for Miner Detection in Underground Coal Mines
原始书目：作者：Y. Y. Chen; X. R. Meng; C. Y. Yang; Y. J. Wang；年份（附件）：2026；出版物（附件）：APPLIED SCIENCES-BASEL。
**DOI/标识：** 未提供／未核；按完整题名检索，禁止补造。
**原材料：** S10:L258–L266（编号: RE183）。
**v4.1角色：** 检索保留／背景备用；不因 v4 放宽路线而自动升级为直接依据。
反查步骤：目前未列为必做步骤；保留检索，使用前补原文；读文献包：背景/归档。
**核验状态：** 附件条目（非本轮原论文全文核验）；本轮未另行网络核验。
查找方式：复制本条完整题名检索；没有可确认 DOI 时不补造。

<a id="ref-RE191"></a>
### [RE191] Explicit Illumination Modeling for Object Detection in Low-Light Environments
原始书目：作者：W. K. Cao; P. Yang; W. Lyu；年份（附件）：2026；出版物（附件）：ELECTRONICS。
**DOI/标识：** 未提供／未核；按完整题名检索，禁止补造。
**原材料：** S10:L266–L274（编号: RE191）。
**v4.1角色：** 检索保留／背景备用；不因 v4 放宽路线而自动升级为直接依据。
反查步骤：目前未列为必做步骤；保留检索，使用前补原文；读文献包：背景/归档。
**核验状态：** 附件条目（非本轮原论文全文核验）；本轮未另行网络核验。
查找方式：复制本条完整题名检索；没有可确认 DOI 时不补造。

<a id="ref-RE194"></a>
### [RE194] DLR-YOLO: A High-Accuracy Lightweight Object Detector for Complex Underground Coal Mine Environments
原始书目：作者：X. H. Cai; R. M. Wang; J. H. Zhang; J. J. Zeng；年份（附件）：2026；出版物（附件）：SENSORS。
**DOI/标识：** 未提供／未核；按完整题名检索，禁止补造。
**原材料：** S10:L274–L282（编号: RE194）。
**v4.1角色：** 检索保留／背景备用；不因 v4 放宽路线而自动升级为直接依据。
反查步骤：目前未列为必做步骤；保留检索，使用前补原文；读文献包：背景/归档。
**核验状态：** 附件条目（非本轮原论文全文核验）；本轮未另行网络核验。
查找方式：复制本条完整题名检索；没有可确认 DOI 时不补造。

<a id="ref-RE218"></a>
### [RE218] Application of CycleGAN-based low-light image enhancement algorithm in foreign object detection on belt conveyors in underground mines
原始书目：作者：A. X. Zhao; Q. H. Zheng; L. Li；年份（附件）：2025；出版物（附件）：SCIENTIFIC REPORTS。
**DOI/标识：** 未提供／未核；按完整题名检索，禁止补造。
**原材料：** S10:L282–L290（编号: RE218）。
**v4.1角色：** 检索保留／背景备用；不因 v4 放宽路线而自动升级为直接依据。
反查步骤：目前未列为必做步骤；保留检索，使用前补原文；读文献包：背景/归档。
**核验状态：** 附件条目（非本轮原论文全文核验）；本轮未另行网络核验。
查找方式：复制本条完整题名检索；没有可确认 DOI 时不补造。

<a id="ref-RE233"></a>
### [RE233] MSS-YOLO: Multi-Scale Edge-Enhanced Lightweight Network for Personnel Detection and Location in Coal Mines
原始书目：作者：W. J. Yang; Y. Q. Wang; X. H. Zhang; L. Zhu; T. H. Wang; Y. K. Chi; J. Jiang；年份（附件）：2025；出版物（附件）：APPLIED SCIENCES-BASEL。
**DOI/标识：** 未提供／未核；按完整题名检索，禁止补造。
**原材料：** S10:L290–L298（编号: RE233）。
**v4.1角色：** 检索保留／背景备用；不因 v4 放宽路线而自动升级为直接依据。
反查步骤：目前未列为必做步骤；保留检索，使用前补原文；读文献包：背景/归档。
**核验状态：** 附件条目（非本轮原论文全文核验）；本轮未另行网络核验。
查找方式：复制本条完整题名检索；没有可确认 DOI 时不补造。

<a id="ref-RE240"></a>
### [RE240] Mine-DW-Fusion: BEV Multiscale-Enhanced Fusion Object-Detection Model for Underground Coal Mine Based on Dynamic Weight Adjustment
检索别名：Mine-DW-Fusion。
原始书目：作者：W. Z. Yan; Y. D. Zhang; M. T. Xue; Z. C. Zhu; H. Lu; X. Zhang; W. Tang; K. K. Xing；年份（附件）：2025；出版物（附件）：SENSORS。
**DOI/标识：** doi: `10.3390/s25165185`（附件记录；本轮未独立核对注册信息）。
**原材料：** S9:L196–L335（编号: RE240）。
**v4.1角色：** 矿下局部置信图、动态融合与补偿。
反查步骤：S.1；读文献包：PACK-SCENE。
适用边界：Camera+LiDAR 检测，不是 RGB-D 语义分割。
**核验状态：** 附件条目（非本轮原论文全文核验）；本轮未另行网络核验。
DOI检索入口：<https://doi.org/10.3390/s25165185>（按附件标识生成，未等同成功解析）。

<a id="ref-RE241"></a>
### [RE241] A Transformer-Based Low-Light Enhancement Algorithm for Rock Bolt Detection in Low-Light Underground Mine Environments
原始书目：作者：W. Z. Yan; F. M. Qu; Y. Z. Wang; J. J. Xu; J. P. Li; L. Y. Zhao；年份（附件）：2025；出版物（附件）：PROCESSES。
**DOI/标识：** 未提供／未核；按完整题名检索，禁止补造。
**原材料：** S10:L298–L307（编号: RE241）。
**v4.1角色：** 检索保留／背景备用；不因 v4 放宽路线而自动升级为直接依据。
反查步骤：目前未列为必做步骤；保留检索，使用前补原文；读文献包：背景/归档。
**核验状态：** 附件条目（非本轮原论文全文核验）；本轮未另行网络核验。
查找方式：复制本条完整题名检索；没有可确认 DOI 时不补造。

<a id="ref-RE257"></a>
### [RE257] CMCF-DETR: a real-time lightweight DETR model for foreign object detection on coal mine conveyor belts
原始书目：作者：Z. Wang; H. H. Zhou; H. J. Ye; Z. Chu；年份（附件）：2025；出版物（附件）：JOURNAL OF REAL-TIME IMAGE PROCESSING。
**DOI/标识：** 未提供／未核；按完整题名检索，禁止补造。
**原材料：** S10:L307–L315（编号: RE257）。
**v4.1角色：** 检索保留／背景备用；不因 v4 放宽路线而自动升级为直接依据。
反查步骤：目前未列为必做步骤；保留检索，使用前补原文；读文献包：背景/归档。
**核验状态：** 附件条目（非本轮原论文全文核验）；本轮未另行网络核验。
查找方式：复制本条完整题名检索；没有可确认 DOI 时不补造。

<a id="ref-RE259"></a>
### [RE259] Coal mine underground image enhancement method based on efficient multi-scale transformation cycle generative adversarial network
原始书目：作者：Y. B. Wang; Z. X. Yan；年份（附件）：2025；出版物（附件）：SIGNAL IMAGE AND VIDEO PROCESSING。
**DOI/标识：** 未提供／未核；按完整题名检索，禁止补造。
**原材料：** S10:L315–L323（编号: RE259）。
**v4.1角色：** 检索保留／背景备用；不因 v4 放宽路线而自动升级为直接依据。
反查步骤：目前未列为必做步骤；保留检索，使用前补原文；读文献包：背景/归档。
**核验状态：** 附件条目（非本轮原论文全文核验）；本轮未另行网络核验。
查找方式：复制本条完整题名检索；没有可确认 DOI 时不补造。

<a id="ref-RE263"></a>
### [RE263] Multi-sensor information fusion methods for coal mine exploration in GNSS-denied scenarios
原始书目：作者：S. Wang; Y. T. Liang; S. T. Liao; S. Xiao; L. Wang；年份（附件）：2025；出版物（附件）：JOURNAL ON WIRELESS COMMUNICATIONS AND NETWORKING。
**DOI/标识：** 未提供／未核；按完整题名检索，禁止补造。
**原材料：** S10:L323–L331（编号: RE263）。
**v4.1角色：** 检索保留／背景备用；不因 v4 放宽路线而自动升级为直接依据。
反查步骤：目前未列为必做步骤；保留检索，使用前补原文；读文献包：背景/归档。
**核验状态：** 附件条目（非本轮原论文全文核验）；本轮未另行网络核验。
查找方式：复制本条完整题名检索；没有可确认 DOI 时不补造。

<a id="ref-RE266"></a>
### [RE266] Scene Understanding System of Underground Pipeline Corridors Under Characteristic Degradation Conditions
原始书目：作者：J. Wang; R. Y. Xing; M. Zhou; J. B. Xu; X. P. Zhang; S. Ju；年份（附件）：2025；出版物（附件）：SENSORS。
**DOI/标识：** doi: `10.3390/s26010141`（附件记录；本轮未独立核对注册信息）。
**原材料：** S10:L331–L445（编号: RE266）。
**v4.1角色：** 任务驱动处理与均值掩盖类别退化。
反查步骤：R.3、V.3、M.R、S.1；读文献包：PACK-R、PACK-EVAL、PACK-SCENE。
适用边界：地下管廊；以逐类任务结果评价修复，不保证必然涨点。
**核验状态：** 附件条目（非本轮原论文全文核验）；本轮未另行网络核验。
DOI检索入口：<https://doi.org/10.3390/s26010141>（按附件标识生成，未等同成功解析）。

<a id="ref-RE285"></a>
### [RE285] Research on Image Segmentation and Defogging Technique of Coal Gangue Under the Influence of Dust Gradient
原始书目：作者：Z. H. Qin; J. D. Jing; L. B. Li; Y. Yuan; Y. Li; B. Li；年份（附件）：2025；出版物（附件）：APPLIED SCIENCES-BASEL。
**DOI/标识：** 未提供／未核；按完整题名检索，禁止补造。
**原材料：** S10:L445–L453（编号: RE285）。
**v4.1角色：** 检索保留／背景备用；不因 v4 放宽路线而自动升级为直接依据。
反查步骤：目前未列为必做步骤；保留检索，使用前补原文；读文献包：背景/归档。
**核验状态：** 附件条目（非本轮原论文全文核验）；本轮未另行网络核验。
查找方式：复制本条完整题名检索；没有可确认 DOI 时不补造。

<a id="ref-RE304"></a>
### [RE304] Obstacle detection and path planning for intelligent drill rod replacement robotic arm in coal mines
原始书目：作者：J. N. Luo; J. P. Li; D. Y. Zhang; Z. Y. Zu；年份（附件）：2025；出版物（附件）：JOURNAL OF THE BRAZILIAN SOCIETY OF MECHANICAL SCIENCES AND ENGINEERING。
**DOI/标识：** 未提供／未核；按完整题名检索，禁止补造。
**原材料：** S10:L453–L461（编号: RE304）。
**v4.1角色：** 检索保留／背景备用；不因 v4 放宽路线而自动升级为直接依据。
反查步骤：目前未列为必做步骤；保留检索，使用前补原文；读文献包：背景/归档。
**核验状态：** 附件条目（非本轮原论文全文核验）；本轮未另行网络核验。
查找方式：复制本条完整题名检索；没有可确认 DOI 时不补造。

<a id="ref-RE326"></a>
### [RE326] MUSeg: A multimodal semantic segmentation dataset for complex underground mine scenes
检索别名：MUSeg。
原始书目：作者：S. Y. Li; Q. Q. Kong; X. Gao; F. Z. Shi; L. H. Li; Q. Zhang; P. H. Wang; K. H. Yang；年份（附件）：2025；出版物（附件）：SCIENTIFIC DATA。
**DOI/标识：** doi: `10.1038/s41597-025-05493-9`（v3 索引继承；本轮未独立核对注册信息）。
**原材料：** S9:L335–L344（编号: RE326）。
**v4.1角色：** MUSeg 数据集原始来源。
反查步骤：E0.1、E1.3、E3.1、E4.1、V.5、V.6、S.1；读文献包：PACK-EVAL、PACK-SCENE。
适用边界：数据集官方划分/属性与本项目开发划分、合成失效分开。
**核验状态：** 附件条目（非本轮原论文全文核验）；本轮另见 W5。
原始检索入口：<https://www.nature.com/articles/s41597-025-05493-9>（已打开出版商原始页面）。

<a id="ref-RE343"></a>
### [RE343] Downhole Coal-Rock Recognition Based on Joint Migration and Enhanced Multidimensional Full-Scale Visual Features
原始书目：作者：B. Jiao; C. M. Sun; S. C. Qin; W. B. Wang; Y. Wang; Z. B. Wu; Y. Li; D. W. Shen；年份（附件）：2025；出版物（附件）：APPLIED SCIENCES-BASEL。
**DOI/标识：** 未提供／未核；按完整题名检索，禁止补造。
**原材料：** S10:L461–L470（编号: RE343）。
**v4.1角色：** 检索保留／背景备用；不因 v4 放宽路线而自动升级为直接依据。
反查步骤：目前未列为必做步骤；保留检索，使用前补原文；读文献包：背景/归档。
**核验状态：** 附件条目（非本轮原论文全文核验）；本轮未另行网络核验。
查找方式：复制本条完整题名检索；没有可确认 DOI 时不补造。

<a id="ref-RE350"></a>
### [RE350] Enhanced foreign body detection on coal mine conveyor belts using improved DLEA and lightweight SARC-DETR model
原始书目：作者：Y. Hong; L. Wang; J. M. Su; Y. Li; B. Q. Zhu; H. T. Wang；年份（附件）：2025；出版物（附件）：SIGNAL IMAGE AND VIDEO PROCESSING。
**DOI/标识：** 未提供／未核；按完整题名检索，禁止补造。
**原材料：** S10:L470–L478（编号: RE350）。
**v4.1角色：** 检索保留／背景备用；不因 v4 放宽路线而自动升级为直接依据。
反查步骤：目前未列为必做步骤；保留检索，使用前补原文；读文献包：背景/归档。
**核验状态：** 附件条目（非本轮原论文全文核验）；本轮未另行网络核验。
查找方式：复制本条完整题名检索；没有可确认 DOI 时不补造。

<a id="ref-RE354"></a>
### [RE354] A foreign object detection method for coal conveyor belts based on brightness self-balancing and multi-scale feature fusion
原始书目：作者：S. Hao; T. R. Qi; X. Ma; Z. Tian; J. H. Li; S. Fan；年份（附件）：2025；出版物（附件）：MEASUREMENT SCIENCE AND TECHNOLOGY。
**DOI/标识：** 未提供／未核；按完整题名检索，禁止补造。
**原材料：** S10:L478–L486（编号: RE354）。
**v4.1角色：** 检索保留／背景备用；不因 v4 放宽路线而自动升级为直接依据。
反查步骤：目前未列为必做步骤；保留检索，使用前补原文；读文献包：背景/归档。
**核验状态：** 附件条目（非本轮原论文全文核验）；本轮未另行网络核验。
查找方式：复制本条完整题名检索；没有可确认 DOI 时不补造。

<a id="ref-RE360"></a>
### [RE360] DeepFissureNets-Infrared-Visible: Infrared visible image fusion for boosting mining-induced ground fissure semantic segmentation
原始书目：作者：J. H. Guo; Y. X. Zhao; C. W. Ling; K. N. Zhang; S. R. Wang; L. C. Zhao；年份（附件）：2025；出版物（附件）：JOURNAL OF ROCK MECHANICS AND GEOTECHNICAL ENGINEERING。
**DOI/标识：** doi: `10.1016/j.jrmge.2025.03.045`（附件记录；本轮未独立核对注册信息）。
**原材料：** S10:L486–L513（编号: RE360）。
**v4.1角色：** 红外/可见互补与下游分割。
反查步骤：S.1；读文献包：PACK-SCENE。
适用边界：矿区地表裂缝，不当作地下巷道真实故障证据。
**核验状态：** 附件条目（非本轮原论文全文核验）；本轮未另行网络核验。
DOI检索入口：<https://doi.org/10.1016/j.jrmge.2025.03.045>（按附件标识生成，未等同成功解析）。
附件自带入口（未自动核验）：<http://creativecommons.org/licenses/by-nc-nd/4.0/>。

<a id="ref-RE365"></a>
### [RE365] Enhancing Autonomous Truck Navigation in Underground Mines: A Review of 3D Object Detection Systems, Challenges, and Future Trends
原始书目：作者：E. Essien; S. Frimpong；年份（附件）：2025；出版物（附件）：DRONES。
**DOI/标识：** 未提供／未核；按完整题名检索，禁止补造。
**原材料：** S10:L513–L521（编号: RE365）。
**v4.1角色：** 检索保留／背景备用；不因 v4 放宽路线而自动升级为直接依据。
反查步骤：目前未列为必做步骤；保留检索，使用前补原文；读文献包：背景/归档。
**核验状态：** 附件条目（非本轮原论文全文核验）；本轮未另行网络核验。
查找方式：复制本条完整题名检索；没有可确认 DOI 时不补造。

<a id="ref-RE366"></a>
### [RE366] A hybrid zero-reference and dehazing network for joint low-light underground image enhancement
原始书目：作者：Q. Du; S. H. Zhang; Z. P. Wang; J. C. Liang; S. J. Yang；年份（附件）：2025；出版物（附件）：SCIENTIFIC REPORTS。
**DOI/标识：** 未提供／未核；按完整题名检索，禁止补造。
**原材料：** S10:L521–L529（编号: RE366）。
**v4.1角色：** 检索保留／背景备用；不因 v4 放宽路线而自动升级为直接依据。
反查步骤：目前未列为必做步骤；保留检索，使用前补原文；读文献包：背景/归档。
**核验状态：** 附件条目（非本轮原论文全文核验）；本轮未另行网络核验。
查找方式：复制本条完整题名检索；没有可确认 DOI 时不补造。

<a id="ref-RE376"></a>
### [RE376] A multimodal data fusion-based intelligent detection method for lump coal on underground conveyor belts in smart manufacturing
原始书目：作者：L. Chen; L. G. Wu; Q. C. Ren；年份（附件）：2025；出版物（附件）：JOURNAL OF INDUSTRIAL INFORMATION INTEGRATION。
**DOI/标识：** 未提供／未核；按完整题名检索，禁止补造。
**原材料：** S10:L529–L537（编号: RE376）。
**v4.1角色：** 检索保留／背景备用；不因 v4 放宽路线而自动升级为直接依据。
反查步骤：目前未列为必做步骤；保留检索，使用前补原文；读文献包：背景/归档。
**核验状态：** 附件条目（非本轮原论文全文核验）；本轮未另行网络核验。
查找方式：复制本条完整题名检索；没有可确认 DOI 时不补造。

<a id="ref-RE386"></a>
### [RE386] Development of multi-sensors fusion monitoring system for shaft wall deformation
原始书目：作者：X. J. Zhu; Y. X. Chen; P. F. Zhang; H. Liu; G. Yang; J. X. Li; M. J. Qiu；年份（附件）：2024；出版物（附件）：MEASUREMENT SCIENCE AND TECHNOLOGY。
**DOI/标识：** 未提供／未核；按完整题名检索，禁止补造。
**原材料：** S10:L537–L545（编号: RE386）。
**v4.1角色：** 检索保留／背景备用；不因 v4 放宽路线而自动升级为直接依据。
反查步骤：目前未列为必做步骤；保留检索，使用前补原文；读文献包：背景/归档。
**核验状态：** 附件条目（非本轮原论文全文核验）；本轮未另行网络核验。
查找方式：复制本条完整题名检索；没有可确认 DOI 时不补造。

<a id="ref-RE392"></a>
### [RE392] Legged robot-aided 3D tunnel mapping via residual compensation and anomaly detection
原始书目：作者：X. Zhang; Z. P. Huang; Q. Q. Li; R. S. Wang; B. D. Zhou；年份（附件）：2024；出版物（附件）：ISPRS JOURNAL OF PHOTOGRAMMETRY AND REMOTE SENSING。
**DOI/标识：** 未提供／未核；按完整题名检索，禁止补造。
**原材料：** S10:L545–L553（编号: RE392）。
**v4.1角色：** 检索保留／背景备用；不因 v4 放宽路线而自动升级为直接依据。
反查步骤：目前未列为必做步骤；保留检索，使用前补原文；读文献包：背景/归档。
**核验状态：** 附件条目（非本轮原论文全文核验）；本轮未另行网络核验。
查找方式：复制本条完整题名检索；没有可确认 DOI 时不补造。

<a id="ref-RE397"></a>
### [RE397] Robust distance measurement using illumination map estimation and MAHNet in underground coal mines
原始书目：作者：J. J. Zhang; J. C. Li; H. T. Liu; H. L. Wang; D. W. Yi; Q. Li；年份（附件）：2024；出版物（附件）：MEASUREMENT SCIENCE AND TECHNOLOGY。
**DOI/标识：** 未提供／未核；按完整题名检索，禁止补造。
**原材料：** S10:L553–L561（编号: RE397）。
**v4.1角色：** 检索保留／背景备用；不因 v4 放宽路线而自动升级为直接依据。
反查步骤：目前未列为必做步骤；保留检索，使用前补原文；读文献包：背景/归档。
**核验状态：** 附件条目（非本轮原论文全文核验）；本轮未另行网络核验。
查找方式：复制本条完整题名检索；没有可确认 DOI 时不补造。

<a id="ref-RE404"></a>
### [RE404] A low-light image enhancement method for personnel safety monitoring in underground coal mines
原始书目：作者：W. Yang; S. Wang; J. Q. Wu; W. Chen; Z. J. Tian；年份（附件）：2024；出版物（附件）：COMPLEX & INTELLIGENT SYSTEMS。
**DOI/标识：** 未提供／未核；按完整题名检索，禁止补造。
**原材料：** S10:L561–L569（编号: RE404）。
**v4.1角色：** 检索保留／背景备用；不因 v4 放宽路线而自动升级为直接依据。
反查步骤：目前未列为必做步骤；保留检索，使用前补原文；读文献包：背景/归档。
**核验状态：** 附件条目（非本轮原论文全文核验）；本轮未另行网络核验。
查找方式：复制本条完整题名检索；没有可确认 DOI 时不补造。

<a id="ref-RE413"></a>
### [RE413] RLI-SLAM: Fast Robust Ranging-LiDAR-Inertial Tightly-Coupled Localization and Mapping
原始书目：作者：R. Xin; N. Y. Guo; X. Y. Ma; G. Liu; Z. Y. Feng；年份（附件）：2024；出版物（附件）：SENSORS。
**DOI/标识：** 未提供／未核；按完整题名检索，禁止补造。
**原材料：** S10:L569–L577（编号: RE413）。
**v4.1角色：** 检索保留／背景备用；不因 v4 放宽路线而自动升级为直接依据。
反查步骤：目前未列为必做步骤；保留检索，使用前补原文；读文献包：背景/归档。
**核验状态：** 附件条目（非本轮原论文全文核验）；本轮未另行网络核验。
查找方式：复制本条完整题名检索；没有可确认 DOI 时不补造。

<a id="ref-RE417"></a>
### [RE417] LIVER: A Tightly Coupled LiDAR-Inertial-Visual State Estimator With High Robustness for Underground Environments
原始书目：作者：T. C. Wen; Y. C. Fang; B. Lu; X. B. Zhang; C. Q. Tang；年份（附件）：2024；出版物（附件）：IEEE ROBOTICS AND AUTOMATION LETTERS。
**DOI/标识：** 未提供／未核；按完整题名检索，禁止补造。
**原材料：** S10:L577–L585（编号: RE417）。
**v4.1角色：** 检索保留／背景备用；不因 v4 放宽路线而自动升级为直接依据。
反查步骤：目前未列为必做步骤；保留检索，使用前补原文；读文献包：背景/归档。
**核验状态：** 附件条目（非本轮原论文全文核验）；本轮未另行网络核验。
查找方式：复制本条完整题名检索；没有可确认 DOI 时不补造。

<a id="ref-RE418"></a>
### [RE418] Adaptive Image Enhancement Method for Coal-Mine Underground Image Based on No-Reference Quality Evaluation
原始书目：作者：D. Wei; P. Q. Wang; Z. B. Wang; L. Si; X. Y. Zou; J. H. Gu; J. B. Dai; C. Y. Long；年份（附件）：2024；出版物（附件）：IEEE TRANSACTIONS ON INSTRUMENTATION AND MEASUREMENT。
**DOI/标识：** doi: `10.1109/TIM.2024.3470234`（附件记录；本轮未独立核对注册信息）。
**原材料：** S10:L585–L810（编号: RE418）。
**v4.1角色：** 质量代理驱动自适应增强。
反查步骤：S.1；读文献包：PACK-SCENE。
适用边界：无参考图像质量不是几何效用真值。
**核验状态：** 附件条目（非本轮原论文全文核验）；本轮未另行网络核验。
DOI检索入口：<https://doi.org/10.1109/TIM.2024.3470234>（按附件标识生成，未等同成功解析）。

<a id="ref-RE419"></a>
### [RE419] SwinURNet: Hybrid Transformer-CNN Architecture for Real-Time Unstructured Road Segmentation
原始书目：作者：Z. Y. Wang; Z. H. Liao; B. Zhou; G. Z. Yu; W. W. Luo；年份（附件）：2024；出版物（附件）：IEEE TRANSACTIONS ON INSTRUMENTATION AND MEASUREMENT。
**DOI/标识：** 未提供／未核；按完整题名检索，禁止补造。
**原材料：** S10:L810–L818（编号: RE419）。
**v4.1角色：** 检索保留／背景备用；不因 v4 放宽路线而自动升级为直接依据。
反查步骤：目前未列为必做步骤；保留检索，使用前补原文；读文献包：背景/归档。
**核验状态：** 附件条目（非本轮原论文全文核验）；本轮未另行网络核验。
查找方式：复制本条完整题名检索；没有可确认 DOI 时不补造。

<a id="ref-RE426"></a>
### [RE426] Real-time semantic segmentation for underground mine tunnel
原始书目：作者：J. W. Wang; D. W. Li; Q. H. Long; Z. Q. Zhao; X. Gao; J. C. Chen; K. H. Yang；年份（附件）：2024；出版物（附件）：ENGINEERING APPLICATIONS OF ARTIFICIAL INTELLIGENCE。
**DOI/标识：** 未提供／未核；按完整题名检索，禁止补造。
**原材料：** S10:L818–L826（编号: RE426）。
**v4.1角色：** 检索保留／背景备用；不因 v4 放宽路线而自动升级为直接依据。
反查步骤：目前未列为必做步骤；保留检索，使用前补原文；读文献包：背景/归档。
**核验状态：** 附件条目（非本轮原论文全文核验）；本轮未另行网络核验。
查找方式：复制本条完整题名检索；没有可确认 DOI 时不补造。

<a id="ref-RE428"></a>
### [RE428] InLIOM: Tightly-Coupled Intensity LiDAR Inertial Odometry and Mapping
原始书目：作者：H. Q. Wang; H. W. Liang; Z. Y. Li; X. K. Zheng; H. T. Xu; P. F. Zhou; B. Kong；年份（附件）：2024；出版物（附件）：IEEE TRANSACTIONS ON INTELLIGENT TRANSPORTATION SYSTEMS。
**DOI/标识：** 未提供／未核；按完整题名检索，禁止补造。
**原材料：** S10:L826–L834（编号: RE428）。
**v4.1角色：** 检索保留／背景备用；不因 v4 放宽路线而自动升级为直接依据。
反查步骤：目前未列为必做步骤；保留检索，使用前补原文；读文献包：背景/归档。
**核验状态：** 附件条目（非本轮原论文全文核验）；本轮未另行网络核验。
查找方式：复制本条完整题名检索；没有可确认 DOI 时不补造。

<a id="ref-RE452"></a>
### [RE452] Graph-based adaptive weighted fusion SLAM using multimodal data in complex underground spaces
检索别名：AWF-SLAM。
原始书目：作者：X. H. Lin; X. Yang; W. Q. Yao; X. Q. Wang; X. W. Ma; B. L. Ma；年份（附件）：2024；出版物（附件）：ISPRS JOURNAL OF PHOTOGRAMMETRY AND REMOTE SENSING。
**DOI/标识：** doi: `10.1016/j.isprsjprs.2024.08.007`（附件记录；本轮未独立核对注册信息）。
**原材料：** S10:L834–L900（编号: RE452）。
**v4.1角色：** 残差/内点率驱动动态多传感器权重。
反查步骤：S.1；读文献包：PACK-SCENE。
适用边界：SLAM factor 级，不是像素级语义可靠性。
**核验状态：** 附件条目（非本轮原论文全文核验）；本轮未另行网络核验。
DOI检索入口：<https://doi.org/10.1016/j.isprsjprs.2024.08.007>（按附件标识生成，未等同成功解析）。

<a id="ref-RE455"></a>
### [RE455] A CIELAB fusion-based generative adversarial network for reliable sand-dust removal in open-pit mines
原始书目：作者：X. D. Li; C. Liu; Y. Y. Sun; W. J. Li; J. M. Li；年份（附件）：2024；出版物（附件）：JOURNAL OF FIELD ROBOTICS。
**DOI/标识：** 未提供／未核；按完整题名检索，禁止补造。
**原材料：** S10:L900–L908（编号: RE455）。
**v4.1角色：** 检索保留／背景备用；不因 v4 放宽路线而自动升级为直接依据。
反查步骤：目前未列为必做步骤；保留检索，使用前补原文；读文献包：背景/归档。
**核验状态：** 附件条目（非本轮原论文全文核验）；本轮未另行网络核验。
查找方式：复制本条完整题名检索；没有可确认 DOI 时不补造。

<a id="ref-RE460"></a>
### [RE460] A Point Cloud Segmentation Method for Dim and Cluttered Underground Tunnel Scenes Based on the Segment Anything Model
原始书目：作者：J. T. Kang; N. Chen; M. Li; S. J. Mao; H. Y. Zhang; Y. B. Fan; H. Liu；年份（附件）：2024；出版物（附件）：REMOTE SENSING。
**DOI/标识：** 未提供／未核；按完整题名检索，禁止补造。
**原材料：** S10:L908–L916（编号: RE460）。
**v4.1角色：** 检索保留／背景备用；不因 v4 放宽路线而自动升级为直接依据。
反查步骤：目前未列为必做步骤；保留检索，使用前补原文；读文献包：背景/归档。
**核验状态：** 附件条目（非本轮原论文全文核验）；本轮未另行网络核验。
查找方式：复制本条完整题名检索；没有可确认 DOI 时不补造。

<a id="ref-RE465"></a>
### [RE465] MD-TLCF: Miner Distance Detection Based on Trajectory-Based Low-Confidence Filter
原始书目：作者：T. Huang; G. Wang; L. Wu; H. Y. Pu; J. Luo; H. L. Liu; X. Y. Zou; J. Luo；年份（附件）：2024；出版物（附件）：IEEE TRANSACTIONS ON INSTRUMENTATION AND MEASUREMENT。
**DOI/标识：** 未提供／未核；按完整题名检索，禁止补造。
**原材料：** S10:L916–L924（编号: RE465）。
**v4.1角色：** 检索保留／背景备用；不因 v4 放宽路线而自动升级为直接依据。
反查步骤：目前未列为必做步骤；保留检索，使用前补原文；读文献包：背景/归档。
**核验状态：** 附件条目（非本轮原论文全文核验）；本轮未另行网络核验。
查找方式：复制本条完整题名检索；没有可确认 DOI 时不补造。
附件自带入口（未自动核验）：<https://github.com/HT-hlf/MD-TLCF.git>。

<a id="ref-RE474"></a>
### [RE474] A coal and gangue detection method for low light and dusty environments
原始书目：作者：J. H. Gao; B. Li; X. W. Wang; J. Zhang; L. Y. Wang；年份（附件）：2024；出版物（附件）：MEASUREMENT SCIENCE AND TECHNOLOGY。
**DOI/标识：** 未提供／未核；按完整题名检索，禁止补造。
**原材料：** S10:L924–L932（编号: RE474）。
**v4.1角色：** 检索保留／背景备用；不因 v4 放宽路线而自动升级为直接依据。
反查步骤：目前未列为必做步骤；保留检索，使用前补原文；读文献包：背景/归档。
**核验状态：** 附件条目（非本轮原论文全文核验）；本轮未另行网络核验。
查找方式：复制本条完整题名检索；没有可确认 DOI 时不补造。

<a id="ref-RE535"></a>
### [RE535] Comparison of Single-Camera-Based Depth Estimation Technology for Digital Twin Model Synchronization of Underground Utility Tunnels
原始书目：作者：S. Park; C. H. Hong; I. Hwang; J. W. Lee；年份（附件）：2023；出版物（附件）：APPLIED SCIENCES-BASEL。
**DOI/标识：** 未提供／未核；按完整题名检索，禁止补造。
**原材料：** S10:L932–L940（编号: RE535）。
**v4.1角色：** 检索保留／背景备用；不因 v4 放宽路线而自动升级为直接依据。
反查步骤：目前未列为必做步骤；保留检索，使用前补原文；读文献包：背景/归档。
**核验状态：** 附件条目（非本轮原论文全文核验）；本轮未另行网络核验。
查找方式：复制本条完整题名检索；没有可确认 DOI 时不补造。

<a id="ref-RE549"></a>
### [RE549] A Multimodal Robust Simultaneous Localization and Mapping Approach Driven by Geodesic Coordinates for Coal Mine Mobile Robots
原始书目：作者：M. G. Li; K. Hu; Y. W. Liu; E. Hu; C. Q. Tang; H. Zhu; G. B. Zhou；年份（附件）：2023；出版物（附件）：REMOTE SENSING。
**DOI/标识：** 未提供／未核；按完整题名检索，禁止补造。
**原材料：** S10:L940–L948（编号: RE549）。
**v4.1角色：** 检索保留／背景备用；不因 v4 放宽路线而自动升级为直接依据。
反查步骤：目前未列为必做步骤；保留检索，使用前补原文；读文献包：背景/归档。
**核验状态：** 附件条目（非本轮原论文全文核验）；本轮未另行网络核验。
查找方式：复制本条完整题名检索；没有可确认 DOI 时不补造。

<a id="ref-RE551"></a>
### [RE551] Visual perception system design for rock breaking robot based on multi-sensor fusion
原始书目：作者：J. G. Li; Y. Liu; S. Wang; L. W. Wang; Y. M. Sun; X. Li；年份（附件）：2023；出版物（附件）：MULTIMEDIA TOOLS AND APPLICATIONS。
**DOI/标识：** 未提供／未核；按完整题名检索，禁止补造。
**原材料：** S10:L948–L956（编号: RE551）。
**v4.1角色：** 检索保留／背景备用；不因 v4 放宽路线而自动升级为直接依据。
反查步骤：目前未列为必做步骤；保留检索，使用前补原文；读文献包：背景/归档。
**核验状态：** 附件条目（非本轮原论文全文核验）；本轮未另行网络核验。
查找方式：复制本条完整题名检索；没有可确认 DOI 时不补造。

<a id="ref-RE554"></a>
### [RE554] Development of Autonomous Driving Patrol Robot for Improving Underground Mine Safety
原始书目：作者：H. Kim; Y. Choi；年份（附件）：2023；出版物（附件）：APPLIED SCIENCES-BASEL。
**DOI/标识：** 未提供／未核；按完整题名检索，禁止补造。
**原材料：** S10:L956–L964（编号: RE554）。
**v4.1角色：** 检索保留／背景备用；不因 v4 放宽路线而自动升级为直接依据。
反查步骤：目前未列为必做步骤；保留检索，使用前补原文；读文献包：背景/归档。
**核验状态：** 附件条目（非本轮原论文全文核验）；本轮未另行网络核验。
查找方式：复制本条完整题名检索；没有可确认 DOI 时不补造。

<a id="ref-RE561"></a>
### [RE561] VIDO: A Robust and Consistent Monocular Visual-Inertial-Depth Odometry
原始书目：作者：Y. X. Gao; J. Yuan; J. Q. Jiang; Q. X. Sun; X. B. Zhang；年份（附件）：2023；出版物（附件）：IEEE TRANSACTIONS ON INTELLIGENT TRANSPORTATION SYSTEMS。
**DOI/标识：** 未提供／未核；按完整题名检索，禁止补造。
**原材料：** S10:L964–L972（编号: RE561）。
**v4.1角色：** 检索保留／背景备用；不因 v4 放宽路线而自动升级为直接依据。
反查步骤：目前未列为必做步骤；保留检索，使用前补原文；读文献包：背景/归档。
**核验状态：** 附件条目（非本轮原论文全文核验）；本轮未另行网络核验。
查找方式：复制本条完整题名检索；没有可确认 DOI 时不补造。

<a id="ref-RE564"></a>
### [RE564] Applications of Machine Vision in Coal Mine Fully Mechanized Tunneling Faces: A Review
原始书目：作者：Y. X. Du; H. Zhang; L. Liang; J. H. Zhang; B. Song；年份（附件）：2023；出版物（附件）：IEEE ACCESS。
**DOI/标识：** 未提供／未核；按完整题名检索，禁止补造。
**原材料：** S10:L972–L980（编号: RE564）。
**v4.1角色：** 检索保留／背景备用；不因 v4 放宽路线而自动升级为直接依据。
反查步骤：目前未列为必做步骤；保留检索，使用前补原文；读文献包：背景/归档。
**核验状态：** 附件条目（非本轮原论文全文核验）；本轮未另行网络核验。
查找方式：复制本条完整题名检索；没有可确认 DOI 时不补造。

<a id="ref-RE574"></a>
### [RE574] A Tightly Coupled LiDAR-Inertial SLAM for Perceptually Degraded Scenes
原始书目：作者：L. Yang; H. W. Ma; Y. Wang; J. Xia; C. A. W. Wang；年份（附件）：2022；出版物（附件）：SENSORS。
**DOI/标识：** 未提供／未核；按完整题名检索，禁止补造。
**原材料：** S10:L980–L988（编号: RE574）。
**v4.1角色：** 检索保留／背景备用；不因 v4 放宽路线而自动升级为直接依据。
反查步骤：目前未列为必做步骤；保留检索，使用前补原文；读文献包：背景/归档。
**核验状态：** 附件条目（非本轮原论文全文核验）；本轮未另行网络核验。
查找方式：复制本条完整题名检索；没有可确认 DOI 时不补造。

<a id="ref-RE579"></a>
### [RE579] Safety monitoring method of moving target in underground coal mine based on computer vision processing
原始书目：作者：P. F. Xu; Z. Q. Zhou; Z. X. Geng；年份（附件）：2022；出版物（附件）：SCIENTIFIC REPORTS。
**DOI/标识：** 未提供／未核；按完整题名检索，禁止补造。
**原材料：** S10:L988–L996（编号: RE579）。
**v4.1角色：** 检索保留／背景备用；不因 v4 放宽路线而自动升级为直接依据。
反查步骤：目前未列为必做步骤；保留检索，使用前补原文；读文献包：背景/归档。
**核验状态：** 附件条目（非本轮原论文全文核验）；本轮未另行网络核验。
查找方式：复制本条完整题名检索；没有可确认 DOI 时不补造。

<a id="ref-RE586"></a>
### [RE586] Obstacle detection method of unmanned electric locomotive in coal mine based on YOLOv3-4L
原始书目：作者：W. S. Wang; S. Wang; Y. C. Guo; Y. Q. Zhao；年份（附件）：2022；出版物（附件）：JOURNAL OF ELECTRONIC IMAGING。
**DOI/标识：** 未提供／未核；按完整题名检索，禁止补造。
**原材料：** S10:L996–L1004（编号: RE586）。
**v4.1角色：** 检索保留／背景备用；不因 v4 放宽路线而自动升级为直接依据。
反查步骤：目前未列为必做步骤；保留检索，使用前补原文；读文献包：背景/归档。
**核验状态：** 附件条目（非本轮原论文全文核验）；本轮未另行网络核验。
查找方式：复制本条完整题名检索；没有可确认 DOI 时不补造。

<a id="ref-RE603"></a>
### [RE603] Coal Mine Belt Conveyor Foreign Objects Recognition Method of Improved YOLOv5 Algorithm with Defogging and Deblurring
原始书目：作者：Q. H. Mao; S. K. Li; X. Hu; X. S. Xue；年份（附件）：2022；出版物（附件）：ENERGIES。
**DOI/标识：** 未提供／未核；按完整题名检索，禁止补造。
**原材料：** S10:L1004–L1012（编号: RE603）。
**v4.1角色：** 检索保留／背景备用；不因 v4 放宽路线而自动升级为直接依据。
反查步骤：目前未列为必做步骤；保留检索，使用前补原文；读文献包：背景/归档。
**核验状态：** 附件条目（非本轮原论文全文核验）；本轮未另行网络核验。
查找方式：复制本条完整题名检索；没有可确认 DOI 时不补造。

<a id="ref-RE606"></a>
### [RE606] A Detection and Tracking Method Based on Heterogeneous Multi-Sensor Fusion for Unmanned Mining Trucks
原始书目：作者：H. T. Liu; W. B. Pan; Y. Q. Hu; C. Li; X. W. Yuan; T. Long；年份（附件）：2022；出版物（附件）：SENSORS。
**DOI/标识：** 未提供／未核；按完整题名检索，禁止补造。
**原材料：** S10:L1012–L1020（编号: RE606）。
**v4.1角色：** 检索保留／背景备用；不因 v4 放宽路线而自动升级为直接依据。
反查步骤：目前未列为必做步骤；保留检索，使用前补原文；读文献包：背景/归档。
**核验状态：** 附件条目（非本轮原论文全文核验）；本轮未另行网络核验。
查找方式：复制本条完整题名检索；没有可确认 DOI 时不补造。

<a id="ref-RE609"></a>
### [RE609] 3DSG: A 3D LiDAR-Based Object Detection Method for Autonomous Mining Trucks Fusing Semantic and Geometric Features
原始书目：作者：H. Z. Li; Z. Y. Wang; G. Z. Yu; Z. R. Gong; B. Zhou; P. Chen; F. Zhao；年份（附件）：2022；出版物（附件）：APPLIED SCIENCES-BASEL。
**DOI/标识：** 未提供／未核；按完整题名检索，禁止补造。
**原材料：** S10:L1020–L1028（编号: RE609）。
**v4.1角色：** 检索保留／背景备用；不因 v4 放宽路线而自动升级为直接依据。
反查步骤：目前未列为必做步骤；保留检索，使用前补原文；读文献包：背景/归档。
**核验状态：** 附件条目（非本轮原论文全文核验）；本轮未另行网络核验。
查找方式：复制本条完整题名检索；没有可确认 DOI 时不补造。

<a id="ref-RE613"></a>
### [RE613] Ore extraction and analysis from RGB image and 3D Point Cloud
原始书目：作者：F. Jin; K. Zhan; S. J. Chen; S. W. Huang; Y. S. Zhang；年份（附件）：2022；出版物（附件）：GOSPODARKA SUROWCAMI MINERALNYMI-MINERAL RESOURCES MANAGEMENT。
**DOI/标识：** 未提供／未核；按完整题名检索，禁止补造。
**原材料：** S10:L1028–L1036（编号: RE613）。
**v4.1角色：** 检索保留／背景备用；不因 v4 放宽路线而自动升级为直接依据。
反查步骤：目前未列为必做步骤；保留检索，使用前补原文；读文献包：背景/归档。
**核验状态：** 附件条目（非本轮原论文全文核验）；本轮未另行网络核验。
查找方式：复制本条完整题名检索；没有可确认 DOI 时不补造。

<a id="ref-RE646"></a>
### [RE646] LOCUS: A Multi-Sensor Lidar-Centric Solution for High-Precision Odometry and 3D Mapping in Real-Time
原始书目：作者：M. Palieri; B. Morrell; A. Thakur; K. Ebadi; J. Nash; A. Chatterjee; C. Kanellakis; L. Carlone; C. Guaragnella; A. A. Agha-Mohammadi；年份（附件）：2021；出版物（附件）：IEEE ROBOTICS AND AUTOMATION LETTERS。
**DOI/标识：** 未提供／未核；按完整题名检索，禁止补造。
**原材料：** S10:L1036–L1044（编号: RE646）。
**v4.1角色：** 检索保留／背景备用；不因 v4 放宽路线而自动升级为直接依据。
反查步骤：目前未列为必做步骤；保留检索，使用前补原文；读文献包：背景/归档。
**核验状态：** 附件条目（非本轮原论文全文核验）；本轮未另行网络核验。
查找方式：复制本条完整题名检索；没有可确认 DOI 时不补造。

<a id="ref-RE663"></a>
### [RE663] Interpretable deep learning for roof fall hazard detection in underground mines
原始书目：作者：E. Isleyen; S. Duzgun; R. M. Carter；年份（附件）：2021；出版物（附件）：JOURNAL OF ROCK MECHANICS AND GEOTECHNICAL ENGINEERING。
**DOI/标识：** 未提供／未核；按完整题名检索，禁止补造。
**原材料：** S10:L1044–L1052（编号: RE663）。
**v4.1角色：** 检索保留／背景备用；不因 v4 放宽路线而自动升级为直接依据。
反查步骤：目前未列为必做步骤；保留检索，使用前补原文；读文献包：背景/归档。
**核验状态：** 附件条目（非本轮原论文全文核验）；本轮未另行网络核验。
查找方式：复制本条完整题名检索；没有可确认 DOI 时不补造。

<a id="ref-RE667"></a>
### [RE667] DARE-SLAM: Degeneracy-Aware and Resilient Loop Closing in Perceptually-Degraded Environments
原始书目：作者：K. Ebadi; M. Palieri; S. Wood; C. Padgett; A. A. Agha-mohammadi；年份（附件）：2021；出版物（附件）：JOURNAL OF INTELLIGENT & ROBOTIC SYSTEMS。
**DOI/标识：** 未提供／未核；按完整题名检索，禁止补造。
**原材料：** S10:L1052–L1060（编号: RE667）。
**v4.1角色：** 检索保留／背景备用；不因 v4 放宽路线而自动升级为直接依据。
反查步骤：目前未列为必做步骤；保留检索，使用前补原文；读文献包：背景/归档。
**核验状态：** 附件条目（非本轮原论文全文核验）；本轮未另行网络核验。
查找方式：复制本条完整题名检索；没有可确认 DOI 时不补造。

<a id="ref-RE679"></a>
### [RE679] Pedestrian detection in underground mines via parallel feature transfer network
原始书目：作者：X. Wei; H. T. Zhang; S. F. Liu; Y. Lu；年份（附件）：2020；出版物（附件）：PATTERN RECOGNITION。
**DOI/标识：** 未提供／未核；按完整题名检索，禁止补造。
**原材料：** S10:L1060–L1068（编号: RE679）。
**v4.1角色：** 检索保留／背景备用；不因 v4 放宽路线而自动升级为直接依据。
反查步骤：目前未列为必做步骤；保留检索，使用前补原文；读文献包：背景/归档。
**核验状态：** 附件条目（非本轮原论文全文核验）；本轮未另行网络核验。
查找方式：复制本条完整题名检索；没有可确认 DOI 时不补造。

## 7. AI 编号完整索引
AI001–AI030全部恢复；来源浅者不据索引补造方法

<a id="ref-AI001"></a>
### [AI001] Are Multimodal Transformers Robust to Missing Modality?
原始书目：v3/审计索引未完整列出作者、年份和出版物；请由固定原版本补齐。
**DOI/标识：** doi: `10.1109/CVPR52688.2022.01764`（v3 索引继承；本轮未逐项核 DOI 注册）。
**原材料：** S1:L943–L943（B3. AI编号题名与DOI）。
**v4.1角色：** 缺失模态鲁棒性历史参考。
反查步骤：目前未列为必做步骤；保留检索，使用前补原文；读文献包：背景/归档。
适用边界：仅恢复旧索引；本轮不据未重读的正文推导配置。
**核验状态：** v3 文献索引（不足以直接复现损失或网络）；本轮未另行网络核验。
DOI检索入口：<https://doi.org/10.1109/CVPR52688.2022.01764>（按附件标识生成，未等同成功解析）。

<a id="ref-AI002"></a>
### [AI002] Multimodal Token Fusion for Vision Transformers
原始书目：v3/审计索引未完整列出作者、年份和出版物；请由固定原版本补齐。
**DOI/标识：** doi: `10.1109/CVPR52688.2022.01187`（v3 索引继承；本轮未逐项核 DOI 注册）。
**原材料：** S1:L945–L945（B3. AI编号题名与DOI）。
**v4.1角色：** 多模态 token 融合历史参考。
反查步骤：目前未列为必做步骤；保留检索，使用前补原文；读文献包：背景/归档。
适用边界：作为候选背景，不视为本轮已实现方法。
**核验状态：** v3 文献索引（不足以直接复现损失或网络）；本轮未另行网络核验。
DOI检索入口：<https://doi.org/10.1109/CVPR52688.2022.01187>（按附件标识生成，未等同成功解析）。

<a id="ref-AI003"></a>
### [AI003] CMX: Cross-Modal Fusion for RGB-X Semantic Segmentation With Transformers
检索别名：CMX。
原始书目：v3/审计索引未完整列出作者、年份和出版物；请由固定原版本补齐。
**DOI/标识：** doi: `10.1109/TITS.2023.3300537`（v3 索引继承；本轮未逐项核 DOI 注册）。
**原材料：** S1:L947–L947（B3. AI编号题名与DOI）。
**v4.1角色：** 普通 RGB-X 强基线候选。
反查步骤：V.4、V.6；读文献包：PACK-EVAL。
适用边界：与本方法比较需同划分、同退化 manifest；不同骨干不是单变量实验。
**核验状态：** v3 文献索引（不足以直接复现损失或网络）；本轮未另行网络核验。
DOI检索入口：<https://doi.org/10.1109/TITS.2023.3300537>（按附件标识生成，未等同成功解析）。

<a id="ref-AI004"></a>
### [AI004] DFormer: Rethinking RGBD Representation Learning for Semantic Segmentation
检索别名：DFormer。
原始书目：v3/审计索引未完整列出作者、年份和出版物；请由固定原版本补齐。
**DOI/标识：** doi: `10.48550/arXiv.2309.09668`（v3 索引继承；本轮未逐项核 DOI 注册）。
**原材料：** S1:L949–L949（B3. AI编号题名与DOI）。
**v4.1角色：** DFormer 原骨干及 ConD 的结构背景。
反查步骤：V.4；读文献包：PACK-EVAL。
适用边界：DFormer 和 DFormerv2 不得混称。
**核验状态：** v3 文献索引（不足以直接复现损失或网络）；本轮另见 W24。
原始检索入口：<https://arxiv.org/abs/2309.09668>（已打开题名/摘要条目）。

<a id="ref-AI005"></a>
### [AI005] GeminiFusion: Efficient Pixel-wise Multimodal Fusion for Vision Transformer
检索别名：GeminiFusion。
原始书目：v3/审计索引未完整列出作者、年份和出版物；请由固定原版本补齐。
**DOI/标识：** doi: `10.48550/arXiv.2406.01210`（v3 索引继承；本轮未逐项核 DOI 注册）。
**原材料：** S1:L951–L951（B3. AI编号题名与DOI）。
**v4.1角色：** 普通多模态强基线与 GeomPrompt 的原始承载模型背景。
反查步骤：V.4；读文献包：PACK-EVAL。
适用边界：原文在其上有效不证明在 DFormerv2-S 上有效。
**核验状态：** v3 文献索引（不足以直接复现损失或网络）；本轮另见 W25。
原始检索入口：<https://arxiv.org/abs/2406.01210>（已打开题名/摘要条目）。

<a id="ref-AI006"></a>
### [AI006] Missing Modality Robustness in Semi-Supervised Multi-Modal Semantic Segmentation
原始书目：v3/审计索引未完整列出作者、年份和出版物；请由固定原版本补齐。
**DOI/标识：** doi: `10.1109/WACV57701.2024.00106`（v3 索引继承；本轮未逐项核 DOI 注册）。
**原材料：** S1:L953–L953（B3. AI编号题名与DOI）。
**v4.1角色：** 半监督场景下缺失模态鲁棒性历史参考。
反查步骤：目前未列为必做步骤；保留检索，使用前补原文；读文献包：背景/归档。
适用边界：半监督设定与本项目不同。
**核验状态：** v3 文献索引（不足以直接复现损失或网络）；本轮未另行网络核验。
DOI检索入口：<https://doi.org/10.1109/WACV57701.2024.00106>（按附件标识生成，未等同成功解析）。

<a id="ref-AI007"></a>
### [AI007] Multi-Modal Learning with Missing Modality via Shared-Specific Feature Modelling
原始书目：v3/审计索引未完整列出作者、年份和出版物；请由固定原版本补齐。
**DOI/标识：** doi: `10.1109/CVPR52729.2023.01524`（v3 索引继承；本轮未逐项核 DOI 注册）。
**原材料：** S1:L955–L955（B3. AI编号题名与DOI）。
**v4.1角色：** 共享/特有表征历史参考。
反查步骤：目前未列为必做步骤；保留检索，使用前补原文；读文献包：背景/归档。
适用边界：未重读原文时不指定其网络和损失参数。
**核验状态：** v3 文献索引（不足以直接复现损失或网络）；本轮未另行网络核验。
DOI检索入口：<https://doi.org/10.1109/CVPR52729.2023.01524>（按附件标识生成，未等同成功解析）。

<a id="ref-AI008"></a>
### [AI008] Learning Modality-Agnostic Representation for Semantic Segmentation from Any Modalities
原始书目：v3/审计索引未完整列出作者、年份和出版物；请由固定原版本补齐。
**DOI/标识：** doi: `10.1007/978-3-031-72754-2_9`（v3 索引继承；本轮未逐项核 DOI 注册）。
**原材料：** S1:L957–L957（B3. AI编号题名与DOI）。
**v4.1角色：** 任意模态表示学习历史参考。
反查步骤：目前未列为必做步骤；保留检索，使用前补原文；读文献包：背景/归档。
适用边界：按需补原文；不默认与现有 RGB-D 输入结构兼容。
**核验状态：** v3 文献索引（不足以直接复现损失或网络）；本轮未另行网络核验。
DOI检索入口：<https://doi.org/10.1007/978-3-031-72754-2_9>（按附件标识生成，未等同成功解析）。

<a id="ref-AI009"></a>
### [AI009] Robust Multimodal Learning With Missing Modalities via Parameter-Efficient Adaptation
原始书目：v3/审计索引未完整列出作者、年份和出版物；请由固定原版本补齐。
**DOI/标识：** doi: `10.1109/TPAMI.2024.3476487`（v3 索引继承；本轮未逐项核 DOI 注册）。
**原材料：** S1:L959–L959（B3. AI编号题名与DOI）。
**v4.1角色：** 参数高效缺失模态适配历史参考。
反查步骤：F.1；读文献包：PACK-F。
适用边界：与小适配器相邻，不自动成为同一实现。
**核验状态：** v3 文献索引（不足以直接复现损失或网络）；本轮未另行网络核验。
DOI检索入口：<https://doi.org/10.1109/TPAMI.2024.3476487>（按附件标识生成，未等同成功解析）。

<a id="ref-AI010"></a>
### [AI010] Benchmarking Multi-Modal Semantic Segmentation Under Sensor Failures: Missing and Noisy Modality Robustness
原始书目：v3/审计索引未完整列出作者、年份和出版物；请由固定原版本补齐。
**DOI/标识：** doi: `10.1109/CVPRW67362.2025.00146`（v3 索引继承；本轮未逐项核 DOI 注册）；arxiv: `2503.18445`（本轮核对题名入口）。
**原材料：** S1:L961–L961（B3. AI编号题名与DOI）。
**v4.1角色：** 传感器缺失/噪声条件的鲁棒分割评价参考。
反查步骤：E0.1、E1.3、E4.1、V.1、V.3、V.5、V.6；读文献包：PACK-EVAL。
适用边界：M6、196组与10000次不是该文自动授权的统一参数。
**核验状态：** v3 文献索引（不足以直接复现损失或网络）；本轮另见 W23。
原始检索入口：<https://arxiv.org/abs/2503.18445>（已打开题名/摘要条目）。

<a id="ref-AI011"></a>
### [AI011] Benchmarking the Robustness of Semantic Segmentation Models with Respect to Common Corruptions
原始书目：v3/审计索引未完整列出作者、年份和出版物；请由固定原版本补齐。
**DOI/标识：** doi: `10.1007/s11263-020-01383-2`（v3 索引继承；本轮未逐项核 DOI 注册）。
**原材料：** S1:L963–L963（B3. AI编号题名与DOI）。
**v4.1角色：** 分割常见腐蚀评价历史参考。
反查步骤：E0.1、E1.3、V.1、V.3；读文献包：PACK-EVAL。
适用边界：保留源 DOI；不是本项目实际 corruption 参数来源。
**核验状态：** v3 文献索引（不足以直接复现损失或网络）；本轮未另行网络核验。
DOI检索入口：<https://doi.org/10.1007/s11263-020-01383-2>（按附件标识生成，未等同成功解析）。

<a id="ref-AI012"></a>
### [AI012] RoboDepth: Robust Out-of-Distribution Depth Estimation under Corruptions
检索别名：RoboDepth。
原始书目：v3/审计索引未完整列出作者、年份和出版物；请由固定原版本补齐。
**DOI/标识：** doi: `10.52202/075280-0932`（v3 索引继承；本轮未逐项核 DOI 注册）。
**原材料：** S1:L965–L965（B3. AI编号题名与DOI）。
**v4.1角色：** 深度估计的退化鲁棒性参考。
反查步骤：目前未列为必做步骤；保留检索，使用前补原文；读文献包：背景/归档。
适用边界：深度估计任务，不是 MMFR 语义分割的直接结果。
**核验状态：** v3 文献索引（不足以直接复现损失或网络）；本轮未另行网络核验。
DOI检索入口：<https://doi.org/10.52202/075280-0932>（按附件标识生成，未等同成功解析）。

<a id="ref-AI013"></a>
### [AI013] On Calibration of Modern Neural Networks
原始书目：v3/审计索引未完整列出作者、年份和出版物；请由固定原版本补齐。
**DOI/标识：** doi: `10.48550/arXiv.1706.04599`（v3 索引继承；本轮未逐项核 DOI 注册）。
**原材料：** S1:L967–L967（B3. AI编号题名与DOI）。
**v4.1角色：** 预测概率校准。
反查步骤：目前未列为必做步骤；保留检索，使用前补原文；读文献包：PACK-EVAL。
适用边界：任意连续质量分数不能直接当概率套用校准指标。
**核验状态：** v3 文献索引（不足以直接复现损失或网络）；本轮未另行网络核验。
DOI检索入口：<https://doi.org/10.48550/arXiv.1706.04599>（按附件标识生成，未等同成功解析）。

<a id="ref-AI014"></a>
### [AI014] SelectiveNet: A Deep Neural Network with an Integrated Reject Option
原始书目：v3/审计索引未完整列出作者、年份和出版物；请由固定原版本补齐。
**DOI/标识：** doi: `10.48550/arXiv.1901.09192`（v3 索引继承；本轮未逐项核 DOI 注册）。
**原材料：** S1:L969–L969（B3. AI编号题名与DOI）。
**v4.1角色：** 选择性预测与拒绝决策参考。
反查步骤：目前未列为必做步骤；保留检索，使用前补原文；读文献包：PACK-EVAL。
适用边界：不代表本项目已有部署拒绝或安全保证。
**核验状态：** v3 文献索引（不足以直接复现损失或网络）；本轮未另行网络核验。
DOI检索入口：<https://doi.org/10.48550/arXiv.1901.09192>（按附件标识生成，未等同成功解析）。

<a id="ref-AI015"></a>
### [AI015] Incomplete RGB-D Salient Object Detection: Conceal, Correlate and Fuse
检索别名：Conceal, Correlate and Fuse。
原始书目：v3/审计索引未完整列出作者、年份和出版物；请由固定原版本补齐。
**DOI/标识：** doi: `10.1016/j.patcog.2024.110700`（v3 索引继承；本轮未逐项核 DOI 注册）。
**原材料：** S1:L971–L971（B3. AI编号题名与DOI）。
**v4.1角色：** Depth 质量判断与低质量输入处理。
反查步骤：目前未列为必做步骤；保留检索，使用前补原文；读文献包：背景/归档。
适用边界：SOD 跨任务；原方法不得被改名冒充原创。
**核验状态：** v3 文献索引（不足以直接复现损失或网络）；本轮未另行网络核验。
DOI检索入口：<https://doi.org/10.1016/j.patcog.2024.110700>（按附件标识生成，未等同成功解析）。

<a id="ref-AI016"></a>
### [AI016] Rethinking RGB-D Salient Object Detection: Models, Data Sets, and Large-Scale Benchmarks
检索别名：D3Net。
原始书目：v3/审计索引未完整列出作者、年份和出版物；请由固定原版本补齐。
**DOI/标识：** doi: `10.1109/TNNLS.2020.2996406`（v3 索引继承；本轮未逐项核 DOI 注册）。
**原材料：** S1:L973–L973（B3. AI编号题名与DOI）。
机制审计：S3（出现行：5, 46, 61, 83, 142, 233, 278, 367 等）。
**v4.1角色：** 低质量深度筛选、RGB/RGB-D 路径选择。
反查步骤：R.2、G.1、G.3、M.G；读文献包：PACK-R、PACK-G。
适用边界：理想路径选择参照不是所有学习式方法的理论上界。
**核验状态：** 附件全文审计的转述；本轮是否另核原文见 W 记录；本轮未另行网络核验。
DOI检索入口：<https://doi.org/10.1109/TNNLS.2020.2996406>（按附件标识生成，未等同成功解析）。

<a id="ref-AI017"></a>
### [AI017] Calibrated RGB-D Salient Object Detection
检索别名：DCF；Calibrated RGB-D SOD。
原始书目：v3/审计索引未完整列出作者、年份和出版物；请由固定原版本补齐。
**DOI/标识：** doi: `10.1109/CVPR46437.2021.00935`（v3 索引继承；本轮未逐项核 DOI 注册）。
**原材料：** S1:L975–L975（B3. AI编号题名与DOI）。
机制审计：S3（出现行：5, 44, 53, 83, 150, 234, 236, 240 等）。
**v4.1角色：** 原始深度与 RGB 估计深度的质量条件校正。
反查步骤：R.1、R.2、M.R；读文献包：PACK-R。
适用边界：图像级可靠性与 RGB-D SOD；不自动等于本项目空间控制。
**核验状态：** 附件全文审计的转述；本轮是否另核原文见 W 记录；本轮另见 W9。
原始检索入口：<https://openaccess.thecvf.com/content/CVPR2021/html/Ji_Calibrated_RGB-D_Salient_Object_Detection_CVPR_2021_paper.html>（已检索官方题名/作者/摘要）。

<a id="ref-AI018"></a>
### [AI018] Uncertainty Inspired RGB-D Saliency Detection
原始书目：v3/审计索引未完整列出作者、年份和出版物；请由固定原版本补齐。
**DOI/标识：** doi: `10.1109/TPAMI.2021.3073564`（v3 索引继承；本轮未逐项核 DOI 注册）。
**原材料：** S1:L977–L977（B3. AI编号题名与DOI）。
机制审计：S4（出现行：3, 13, 18, 35, 59, 171, 173, 688 等）。
**v4.1角色：** 输出/标注不确定性与输入质量的区分。
反查步骤：E0.3；读文献包：PACK-G。
适用边界：不应仅因含 uncertainty 就列为 Depth 故障方法。
**核验状态：** 附件全文审计的转述；本轮是否另核原文见 W 记录；本轮未另行网络核验。
DOI检索入口：<https://doi.org/10.1109/TPAMI.2021.3073564>（按附件标识生成，未等同成功解析）。

<a id="ref-AI019"></a>
### [AI019] MaskMentor: Unlocking the Potential of Masked Self-Teaching for Missing Modality RGB-D Semantic Segmentation
检索别名：MaskMentor。
原始书目：v3/审计索引未完整列出作者、年份和出版物；请由固定原版本补齐。
**DOI/标识：** doi: `10.1145/3664647.3681698`（v3 索引继承；本轮未逐项核 DOI 注册）。
**原材料：** S1:L979–L979（B3. AI编号题名与DOI）。
机制审计：S2（出现行：4, 59, 66, 97, 293, 453, 492, 649）。
**v4.1角色：** 完整教师到缺失学生、自监督遮罩学习。
反查步骤：E0.4、T.1、T.2、T.3、E1.1、E1.2、V.2、M.T、V.4；读文献包：PACK-T。
适用边界：原 masking/重建机制不等于 v4 的简化输出蒸馏；需核正文和实现。
**核验状态：** 附件全文审计的转述；本轮是否另核原文见 W 记录；本轮另见 W6。
原始检索入口：<https://doi.org/10.1145/3664647.3681698>（DOI 直开失败；原始 PDF 入口已定位，未读取）。

<a id="ref-AI020"></a>
### [AI020] A Conflict-Guided Evidential Multimodal Fusion for Semantic Segmentation
检索别名：ECoLaF。
原始书目：v3/审计索引未完整列出作者、年份和出版物；请由固定原版本补齐。
**DOI/标识：** doi: `10.1109/WACV61041.2025.00141`（v3 索引继承；本轮未逐项核 DOI 注册）。
**原材料：** S1:L981–L981（B3. AI编号题名与DOI）。
机制审计：S2（出现行：4, 60, 68, 131, 177, 232, 252, 494 等）。
**v4.1角色：** 输出证据冲突与可靠性折扣。
反查步骤：E0.3、G.2；读文献包：PACK-G。
适用边界：多模态分割输出层方法；冲突并非深度物理质量真值。
**核验状态：** 附件全文审计的转述；本轮是否另核原文见 W 记录；本轮另见 W13。
原始检索入口：<https://openaccess.thecvf.com/content/WACV2025/html/Deregnaucourt_A_Conflict-Guided_Evidential_Multimodal_Fusion_for_Semantic_Segmentation_WACV_2025_paper.html>（已检索官方题名/摘要）。

<a id="ref-AI021"></a>
### [AI021] Provable Dynamic Fusion for Low-Quality Multimodal Data
检索别名：QMF。
原始书目：v3/审计索引未完整列出作者、年份和出版物；请由固定原版本补齐。
**DOI/标识：** doi: `10.48550/arXiv.2306.02050`（v3 索引继承；本轮未逐项核 DOI 注册）。
**原材料：** S1:L983–L983（B3. AI编号题名与DOI）。
机制审计：S2（出现行：4, 61, 70, 119, 231, 393, 429, 493 等）。
**v4.1角色：** 质量相关动态融合及任务效用关系。
反查步骤：E0.3、G.2、M.G；读文献包：PACK-G。
适用边界：样本/决策层理论不能无条件套用于几何 token 关系。
**核验状态：** 附件全文审计的转述；本轮是否另核原文见 W 记录；本轮另见 W14。
原始检索入口：<https://proceedings.mlr.press/v202/zhang23ar.html>（已打开原始论文条目及 arXiv 元数据）。

<a id="ref-AI022"></a>
### [AI022] Centering the Value of Every Modality: Towards Efficient and Resilient Modality-Agnostic Semantic Segmentation
检索别名：MAGIC。
原始书目：v3/审计索引未完整列出作者、年份和出版物；请由固定原版本补齐。
**DOI/标识：** doi: `10.1007/978-3-031-72890-7_12`（v3 索引继承；本轮未逐项核 DOI 注册）；arxiv: `2407.11344`（本轮核对题名入口）。
**原材料：** S1:L985–L985（B3. AI编号题名与DOI）。
机制审计：S3（出现行：5, 42, 51, 83, 158, 228, 280, 359 等）。
**v4.1角色：** 稳健/脆弱模态与任意模态分割。
反查步骤：T.2、V.4；读文献包：PACK-T。
适用边界：适配须核对模态结构与训练暴露。
**核验状态：** 附件全文审计的转述；本轮是否另核原文见 W 记录；本轮另见 W15。
原始检索入口：<https://arxiv.org/abs/2407.11344>（已打开题名/摘要条目）。

<a id="ref-AI023"></a>
### [AI023] GeomPrompt: Geometric Prompt Learning for RGB-D Semantic Segmentation Under Missing and Degraded Depth
检索别名：GeomPrompt；GeomPrompt-Recovery。
原始书目：v3/审计索引未完整列出作者、年份和出版物；请由固定原版本补齐。
**DOI/标识：** doi: `10.48550/arXiv.2604.11585`（v3 索引继承；本轮未逐项核 DOI 注册）。
**原材料：** S1:L987–L987（B3. AI编号题名与DOI）。
机制审计：S2（出现行：4, 63, 74, 97, 199, 353, 453, 492 等）。
**v4.1角色：** 无深度几何提示及受损深度任务驱动残差修正。
反查步骤：R.1、R.2、R.3、E1.1、E2.1、E3.1、V.2、M.R；读文献包：PACK-R。
适用边界：区分 GeomPrompt 与 Recovery；无深度 GT 不等于只有一个损失项。
本轮外核补充（W2；不覆盖上面的原记录）：原入口作者：Krishna Jaganathan; Patricio Vela；原入口年份：2026；状态：arXiv 条目注明 CVPR 2026 URVIS Workshop 接收；非 CVPR 主会。
**核验状态：** 附件全文审计的转述；本轮是否另核原文见 W 记录；本轮另见 W2。
原始检索入口：<https://arxiv.org/html/2604.11585>（已读取方法/训练段落及元数据）。

<a id="ref-AI024"></a>
### [AI024] Toward Reliable RGB-D Semantic Segmentation: Handling Missing Modalities via Condition Dropout
检索别名：Condition Dropout；ConD。
原始书目：v3/审计索引未完整列出作者、年份和出版物；请由固定原版本补齐。
**DOI/标识：** doi: `10.48550/arXiv.2607.20326`（v3 索引继承；本轮未逐项核 DOI 注册）。
**原材料：** S1:L989–L989（B3. AI编号题名与DOI）。
机制审计：S3（出现行：5, 48, 67, 83, 284, 454, 593, 595 等）。
**v4.1角色：** 冻结原网络、复制编码器和零初始化注入。
反查步骤：E0.4、F.1、F.2、T.3、E1.1、E1.2、E2.1、E3.1、V.2、M.F、V.4；读文献包：PACK-T、PACK-F。
适用边界：小型 adapter 是本项目简化适配；不能预设额外分支没有推理成本。
本轮外核补充（W3；不覆盖上面的原记录）：原入口作者：Xuchen Zhu; Yajuan Wei; Shuang Hao; Jiwei Jiang; Guanxiang Mao; Fang Ren；原入口年份：2026；状态：arXiv v1；本轮未确认正式接收；原摘要称代码将在接收后公开。
**核验状态：** 附件全文审计的转述；本轮是否另核原文见 W 记录；本轮另见 W3。
原始检索入口：<https://arxiv.org/html/2607.20326v1>（已读取方法/训练段落及元数据）。

<a id="ref-AI025"></a>
### [AI025] When Fusion Fails: Corruption-Aware Rebalanced Fusion for Multi-Modal Medical Image Segmentation
检索别名：CoReFuse-Med。
原始书目：v3/审计索引未完整列出作者、年份和出版物；请由固定原版本补齐。
**DOI/标识：** doi: `10.1145/3767308.3836234`（v3 索引继承；本轮未逐项核 DOI 注册）；arxiv: `2609.10261v1`（本轮核对版本入口）；doi: `10.48550/arXiv.2609.10261`（arXiv 页面列示 pending registration，未宣称注册完成）。
**原材料：** S1:L991–L991（B3. AI编号题名与DOI）。
机制审计：S2（出现行：4, 64, 76, 234, 273, 397, 497, 561 等）。
**v4.1角色：** 特征传输抑噪和模态贡献重平衡。
反查步骤：F.3、M.F；读文献包：PACK-F。
适用边界：医学分割、主要为分辨率诱发退化；不是矿下失效实测。
本轮外核补充（W17；不覆盖上面的原记录）：原入口作者：Yuchen Pei; Xiaoyu Hu; Yixiong Zou; Dingwen Hu; Hui Chu; Yutao Ma; Shijun Qiu; Gang Li；原入口年份：2026；状态：arXiv v1 2026-09-09；摘要注明 ACM MM 2026 接收。
**核验状态：** 附件全文审计的转述；本轮是否另核原文见 W 记录；本轮另见 W17。
原始检索入口：<https://arxiv.org/abs/2609.10261>（已读取元数据/摘要，定位原始 HTML）。

<a id="ref-AI026"></a>
### [AI026] SimMLM: A Simple Framework for Multi-Modal Learning with Missing Modality
检索别名：SimMLM。
原始书目：v3/审计索引未完整列出作者、年份和出版物；请由固定原版本补齐。
**DOI/标识：** doi: `10.1109/ICCV51701.2025.02231`（v3 索引继承；本轮未逐项核 DOI 注册）。
**原材料：** S1:L993–L993（B3. AI编号题名与DOI）。
机制审计：S3（出现行：5, 45, 63, 83, 178, 230, 281, 397 等）。
**v4.1角色：** 专家门控和多/少模态约束。
反查步骤：F.3、T.4；读文献包：PACK-T、PACK-F。
适用边界：多模态更多不必更好，受损且存在的 Depth 不能直接套无损更多模态假设。
**核验状态：** 附件全文审计的转述；本轮是否另核原文见 W 记录；本轮另见 W16。
原始检索入口：<https://openaccess.thecvf.com/content/ICCV2025/html/Li_SimMLM_A_Simple_Framework_for_Multi-modal_Learning_with_Missing_Modality_ICCV_2025_paper.html>（已检索官方题名/摘要）。

<a id="ref-AI027"></a>
### [AI027] Addressing Missing and Noisy Modalities in One Solution: Unified Modality-Quality Framework for Low-Quality Multimodal Data
检索别名：UMQ。
原始书目：v3/审计索引未完整列出作者、年份和出版物；请由固定原版本补齐。
**DOI/标识：** doi: `10.48550/arXiv.2603.02695`（v3 索引继承；本轮未逐项核 DOI 注册）。
**原材料：** S1:L995–L995（B3. AI编号题名与DOI）。
机制审计：S4（出现行：3, 14, 18, 32, 69, 127, 242, 244 等）。
**v4.1角色：** 质量排序、增强与质量感知专家路由。
反查步骤：E0.3、G.2；读文献包：PACK-G。
适用边界：情感计算跨任务；不把抽象流程当可直接复制的视觉模块。
**核验状态：** 附件全文审计的转述；本轮是否另核原文见 W 记录；本轮另见 W18。
原始检索入口：<https://arxiv.org/abs/2603.02695>（已打开题名/摘要条目）。

<a id="ref-AI028"></a>
### [AI028] Adaptive Modality Reliability Diagnosis and Restoration for Robust Multimodal Intent Recognition
检索别名：PRIME。
原始书目：v3/审计索引未完整列出作者、年份和出版物；请由固定原版本补齐。
**DOI/标识：** doi: `10.48550/arXiv.2608.03475`（v3 索引继承；本轮未逐项核 DOI 注册）。
**原材料：** S1:L997–L997（B3. AI编号题名与DOI）。
机制审计：S4（出现行：3, 15, 18, 33, 85, 137, 220, 222 等）。
**v4.1角色：** 合成退化监督、恢复后复诊与融合。
反查步骤：E0.3；读文献包：PACK-G。
适用边界：多模态意图识别；复诊闭环不是 MMFR 预先必须具备的模块。
**核验状态：** 附件全文审计的转述；本轮是否另核原文见 W 记录；本轮另见 W19。
原始检索入口：<https://arxiv.org/abs/2608.03475>（已打开题名/摘要条目）。

<a id="ref-AI029"></a>
### [AI029] Confidence Propagation through CNNs for Guided Sparse Depth Regression
原始书目：v3/审计索引未完整列出作者、年份和出版物；请由固定原版本补齐。
**DOI/标识：** doi: `10.48550/arXiv.1811.01791`（v3 索引继承；本轮未逐项核 DOI 注册）。
**原材料：** S1:L999–L999（B3. AI编号题名与DOI）。
**v4.1角色：** 稀疏深度置信传播和归一化聚合。
反查步骤：E0.2、G.1；读文献包：PACK-R、PACK-G。
适用边界：深度回归跨任务；validity 尺度运输方案需在本地测试。
**核验状态：** v3 文献索引（不足以直接复现损失或网络）；本轮另见 W21。
原始检索入口：<https://arxiv.org/abs/1811.01791>（已打开题名/摘要条目）。

<a id="ref-AI030"></a>
### [AI030] Uncertainty-Aware CNNs for Depth Completion: Uncertainty from Beginning to End
原始书目：v3/审计索引未完整列出作者、年份和出版物；请由固定原版本补齐。
**DOI/标识：** doi: `10.48550/arXiv.2006.03349`（v3 索引继承；本轮未逐项核 DOI 注册）。
**原材料：** S1:L1000–L1000（B3. AI编号题名与DOI）。
**v4.1角色：** 深度补全中的输入/传播不确定性。
反查步骤：E0.2、G.1；读文献包：PACK-R、PACK-G。
适用边界：不等同于本项目 segmentation utility。
**核验状态：** v3 文献索引（不足以直接复现损失或网络）；本轮另见 W22。
原始检索入口：<https://arxiv.org/abs/2006.03349>（已打开题名/摘要条目）。

## 8. 未重新编号的既有方法

<a id="ref-MoSA"></a>
### [MoSA] Modality-Aware Spatially-Adaptive Adaptation for RGB-X Semantic Segmentation
检索别名：MoSA。
原始书目：v3/审计索引未完整列出作者、年份和出版物；请由固定原版本补齐。
**DOI/标识：** doi: `10.1109/ACCESS.2026.3694496`（附件记录；出版商入口受限，题名—DOI 对应待进一步核验）。
**原材料：** S13:L314–L314。
机制审计：S4（出现行：3, 16, 18, 34, 96, 109, 155, 197 等）。
**v4.1角色：** 空间可靠性、适配器调制与融合。
反查步骤：F.1、F.3、G.2、E1.1、E3.1、M.F、V.4；读文献包：PACK-F、PACK-G。
适用边界：题名存在版本差异；出版商全文受限，本轮机制依据来自 P2。
本轮外核补充（W10；不覆盖上面的原记录）：原入口题名变体：MoSA: Modality-Aware Spatially-Adaptive Adaptation for Multimodal Semantic Segmentation；状态：仅官方索引题名可见；题名/DOI 对应需继续核。
**核验状态：** 附件全文审计的转述；本轮是否另核原文见 W 记录；本轮另见 W10。
原始检索入口：<https://ieeexplore.ieee.org/document/11523456>（官方索引题名可见；页面正文受限）。

<a id="ref-ANGA"></a>
### [ANGA] Anchor-Guided Gradient Alignment for Incomplete Multimodal Learning
检索别名：ANGA。
原始书目：v3/审计索引未完整列出作者、年份和出版物；请由固定原版本补齐。
**DOI/标识：** 未提供／未核；按完整题名检索，禁止补造。
**原材料：** S13:L321–L321。
机制审计：S3（出现行：5, 43, 51, 83, 174, 231, 282, 403 等）。
**v4.1角色：** 可靠重建样本的选择与梯度对齐。
反查步骤：T.4；读文献包：PACK-T。
适用边界：保持旧代号；本轮已找到正式条目，但 DOI 未确认。
本轮外核补充（W20；不覆盖上面的原记录）：原入口作者：Zhi-Hao Guan; Longfei Huang; Yang Yang；原入口年份：2026；状态：CVPR 2026 官方检索条目；DOI 未核。
**核验状态：** 附件全文审计的转述；本轮是否另核原文见 W 记录；本轮另见 W20。
原始检索入口：<https://openaccess.thecvf.com/content/CVPR2026/html/Guan_Anchor-Guided_Gradient_Alignment_for_Incomplete_Multimodal_Learning_CVPR_2026_paper.html>（已检索官方题名/作者/摘要）。

## 9. 仅有编号的归档条目与提示词示例
下列编号只在 S12 旧清单的移出/归档列表出现，本轮未提供各自题名、作者、DOI或正文。不因为缺少材料删去编号，也不假装已经识别这些论文。

`RE003` `RE017` `RE022` `RE040` `RE042` `RE044` `RE053` `RE063` `RE070` `RE073` `RE076` `RE095` `RE099` `RE111` `RE118` `RE122` `RE124` `RE129` `RE136` `RE138` `RE141` `RE150` `RE168` `RE172` `RE178` `RE188` `RE189` `RE195` `RE209` `RE214` `RE222` `RE239` `RE246` `RE248` `RE251` `RE278` `RE284` `RE289` `RE292` `RE293` `RE297` `RE305` `RE307` `RE309` `RE336` `RE344` `RE356` `RE359` `RE361` `RE377` `RE382` `RE395` `RE398` `RE400` `RE408` `RE409` `RE410` `RE412` `RE423` `RE425` `RE431` `RE433` `RE439` `RE446` `RE449` `RE462` `RE497` `RE501` `RE509` `RE512` `RE519` `RE521` `RE525` `RE530` `RE545` `RE550` `RE553` `RE555` `RE557` `RE568` `RE569` `RE570` `RE575` `RE582` `RE589` `RE591` `RE593` `RE595` `RE596` `RE607` `RE616` `RE618` `RE621` `RE623` `RE624` `RE627` `RE628` `RE629` `RE631` `RE633` `RE635` `RE636` `RE637` `RE641` `RE642` `RE643` `RE648` `RE649` `RE652` `RE653` `RE655` `RE660` `RE661` `RE662` `RE668` `RE690` `RE696` `RE698` `RE700` `RE702` `RE705` `RE707` `RE713` `RE716`

**RE067** 只出现在 S14 的引用格式举例，不是本次提供的论文；独立标为“示例编号”。

## 10. 参数出处与待核验台账

| 项目/参数 | 性质 | 来源 | 边界 |
| --- | --- | --- | --- |
| A2 epoch-420、clean 57.06、M6 55.55 | 项目结果转录 | S6 §1、§5 | 不代表本轮重新评测；B0 未成为 matched control。 |
| 500 epochs、batch 10、主干 LR 6e-5、25% clean/75% corruption | 项目账本继承 | S1 §冻结项目账本（228–244） | 实施须核配置/调度器；不能当作所有文献通用配置。 |
| 318 val-dev、196 location groups、10000 次配对 bootstrap | 项目协议继承 | S1 §S2.8；S6 §1 | test 分组另核；不把像素当独立训练。 |
| 3 个预登记训练 seed | 项目确认阶段计划 | S1 §S2.8 | 不是已完成 3 seed 的事实。 |
| 额外 50 epochs、小型 adapter、最多两路线/两组合 | v4 研发建议 | S5 §5、§E1–E2 | 依据学习曲线与预算登记；不是论文原设。 |
| clean 约束 epsilon=0.50 与鲁棒验证 selector | v4 新实验建议 | S5 §8.1 | 新组同机会；不追溯重选 A2，不用 test。 |
| 约+1.0、进取+1.5–2.0 mIoU 点 | v4 研发目标 | S5 §9.2 | 不是预测/保证/既有结果；不得改指标口径。 |
| GeomPrompt 原 loss 有分割项、TV 与 L1 正则 | 本轮原文核验 | AI023；W2 §3.4 | 无 Depth GT 不等于无正则；v4 简化须标明。 |
| ConD 原三种模态状态等概率、复制编码器 | 本轮原文核验 | AI024；W3 §II-A–II-C | 不是 A2 的 25/75；不是小型 adapter 的现成配置。 |

<a id="K01"></a>
### K01｜附件“全文已核”与本轮核验范围
原记录：P0/P1/P2 记载此前已进行全文审计。
处理：本轮可使用其审计结论作为所提供资料，但不改写为本轮已逐篇重读原 PDF。W 记录单独标明代码/方法段落/摘要/检索入口/访问失败。
依据：S2、S3、S4、S13。

<a id="K02"></a>
### K02｜PR070：论文整理表述与当前公开实现
原记录：S7 PR070 描述 average pooling 与 softmax 后乘几何 mask；S1 已提醒公开实现差异。
处理：W1 当前代码使用双线性插值，并在 softmax 前加 mask/bias。保留原整理文本；实施前核对本地 fork、归一化、位置/深度混合权重。二者不能未经推导直接当同一算子。
依据：S7、S1、W1、W4。

<a id="K03"></a>
### K03｜PR059：两份原始条目与出版时间冲突
原记录：相同题名既被列为 KBS 327 (2025), 114107，又记录 arXiv:2603.07570v1 (2026-03-08)。
处理：一个 PR059 下保留两条 source record；本轮确认预印本题名/作者/日期/DOI，未确认 KBS 对应，正式刊物元数据仍待核。
依据：S7、S8、S11、W26。

<a id="K04"></a>
### K04｜MoSA：RGB-X / Multimodal 题名与 DOI 对应
原记录：附件题名为 Modality-Aware Spatially-Adaptive Adaptation for RGB-X Semantic Segmentation；官方检索显示 MoSA: ... for Multimodal Semantic Segmentation。
处理：两种题名都保留，继续使用同一既有代号 MoSA。DOI 10.1109/ACCESS.2026.3694496 为附件记录；出版商正文受限，未宣称完整匹配已核。
依据：S4、S13、S1、W10。

<a id="K05"></a>
### K05｜PR090：CVPR RobustSeg 与相关 arXiv 多版本
原记录：CVPR 题名为 Towards Robust Multi-Modal ... Teacher-Student ... Hybrid Prototype Distillation；相关 arXiv:2505.12861 存在题名/模块变化。
处理：复现前指定 CVPR 版本或某一 arXiv vN；不能拼接不同版本的 RRM/FSM/HPDM 等模块。W8 只作相关版本入口，不证明完全同构。
依据：S7、S3、W7、W8。

<a id="K06"></a>
### K06｜AI023：两种输入模式、监督与发表身份
原记录：GeomPrompt 用 RGB 生成提示；Recovery 另使用受损 Depth。摘要称无需 Depth 监督，方法段落同时列出 TV 和 L1 正则。
处理：“只依赖下游分割监督”不简写成“原 loss 只有 CE”；v4 先用分割损失是简化候选。arXiv 注明 CVPR 2026 URVIS Workshop 接收，不能标成 CVPR 主会。
依据：S2、S5、W2。

<a id="K07"></a>
### K07｜AI024：复制编码器不等于小型 adapter
原记录：ConD 原方法训练复制编码器并零初始化注入；v4 F 只借鉴残差适应思路。
处理：F 属于适配/启发式简化，除非补齐原结构和协议，否则不称忠实复现。代码是否公开、推理成本以实际仓库和测量为准；摘要仍使用未来公开的措辞。
依据：S3、S5、W3。

<a id="K08"></a>
### K08｜AI025：补充预印本与接收状态
原记录：附件有 ACM DOI；本轮找到 arXiv:2609.10261v1，提交 2026-09-09，摘要注明 ACM MM 2026 接收。
处理：沿用 AI025，不新编 AI；追加预印本入口。arXiv DOI 页面标 pending registration，不能声称注册已完成；接收不等于会议已经举行。
依据：S13、S2、W17。

<a id="K09"></a>
### K09｜ANGA：找到正式检索入口，但未核 DOI
原记录：原清单使用 ANGA 临时代号，并要求后续核对正式标识。
处理：本轮补充 CVPR 2026 官方检索入口；保留 ANGA，不凭篇名或页码猜造 DOI，也不擅自新增 AI 编号。
依据：S13、S3、W20。

<a id="K10"></a>
### K10｜项目结果与研发参数分离
原记录：A2 结果来自 S6；25/75 exposure、196 组等来自项目账本；50 epochs 与新 selector/目标是设计建议。
处理：不能把 50 epochs、epsilon=0.50、+1.0 或 +1.5–2.0 指为论文原设；不能把历史 B0 当 matched control 或把 52.55 当无深度上限。
依据：S6、S1、S5。

<a id="K11"></a>
### K11｜保留文献证据，不恢复旧方法禁令
原记录：旧清单按几何专用主线给出禁区/降级/排除建议。
处理：保留原文件作为历史证据；执行路线以 v4/v4.1 为准。“已有先例”只要求标明来源、适配与比较，不构成不用该方法的理由。背景条目保留索引，按需要重新评估。
依据：S2、S3、S4、S5、S11、S12、S13。

<a id="K12"></a>
### K12｜合成质量、效用及物理真实性
原记录：旧审计部分措辞把 Q_D 称真实质量、把 utility 称真实边际效用；v3/v4 已限制这种解释。
处理：Q_D 仅是已定义合成协议下的状态目标；任务系数与固定模型干预收益另列。真实矿下传感质量/因果安全性须另有证据，不通过引用术语获得。
依据：S2、S4、S1、S5。

<a id="K13"></a>
### K13｜PR040：作者拼写的已有更正
原记录：原 PR040 条目作者含 Hyunsuh Koh；S11 元数据修正（209–217）明确写应为 Hyunseo Koh。
处理：原始字段保留不覆盖；在 PR040 条目与 JSON 追加 supplied_corrections。正式引用使用前核作者页；不把保留原文误作认可其拼写。
依据：S8、S11。

## 11. 原始资料文件索引与身份
| 文件ID | 原文件名 | SHA-256 前12位 | 物理行数 | 包内原样副本 |
| --- | --- | --- | ---: | --- |
| S1 | `MMFR_research_blueprint_v3_2026-09-18.md` | `10fc73aeb654` | 1020 | [S1_blueprint_v3.md](source_materials/S1_blueprint_v3.md) |
| S2 | `防撞车P0.md` | `0be5e394d4c4` | 776 | [S2_P0_audit.md](source_materials/S2_P0_audit.md) |
| S3 | `防撞车P1.md` | `3858ea3525d7` | 947 | [S3_P1_audit.md](source_materials/S3_P1_audit.md) |
| S4 | `防撞车P2.md` | `93bb36120d0f` | 1592 | [S4_P2_audit.md](source_materials/S4_P2_audit.md) |
| S5 | `MMFR_research_blueprint_v4_2026-09-20(1).md` | `dac01bf7e7d6` | 514 | [S5_blueprint_v4_original.md](source_materials/S5_blueprint_v4_original.md) |
| S6 | `A2实验结果报告.md` | `2f9d82de2be2` | 1347 | [S6_A2_report.md](source_materials/S6_A2_report.md) |
| S7 | `PR核心.md` | `5de41a4549b6` | 105 | [S7_PR_core.md](source_materials/S7_PR_core.md) |
| S8 | `PR其他.txt` | `c728325bc524` | 573 | [S8_PR_other.txt](source_materials/S8_PR_other.txt) |
| S9 | `RE核心.md` | `4d603dc6407a` | 344 | [S9_RE_core.md](source_materials/S9_RE_core.md) |
| S10 | `RE全.txt` | `4bc45d751001` | 1068 | [S10_RE_all.txt](source_materials/S10_RE_all.txt) |
| S11 | `MMFR-PR文献整理清单.md` | `84b8cc56c05a` | 246 | [S11_PR_checklist.md](source_materials/S11_PR_checklist.md) |
| S12 | `MMFR-RE文献整理清单.md` | `752acf9c997d` | 94 | [S12_RE_checklist.md](source_materials/S12_RE_checklist.md) |
| S13 | `MMFR-新增文献与Idea撞车审计.md` | `6ef3ea7289b7` | 341 | [S13_literature_audit_checklist.md](source_materials/S13_literature_audit_checklist.md) |
| S14 | `生成研究蓝图使用的提示词.txt` | `5f5ae01f538d` | 65 | [S14_blueprint_instructions.txt](source_materials/S14_blueprint_instructions.txt) |

完整 SHA-256 和 source archive 相对路径见 JSON。文件名的 `(1)` 后缀不用于判断内容是否更新；以实际哈希判断。S5重上传与前版同字节。原v2没有本轮原件，不声明重新核验。

## 12. 后续实现时的引用与版本填写模板

```text
步骤号：
论文ID（沿用PR/RE/AI或MoSA/ANGA）：
采用版本（会议/期刊版或arXiv vN）：
DOI / 原始URL：
已实际读取的位置（章节/公式/图表/代码函数）：
公开仓库与commit / 本地fork与commit：
原文方法是什么：
本项目朴素移植是什么：
本项目针对性改造是什么：
参数：逐个标【原文】【项目继承】【新建议】【待核】
训练/推理可用输入：
匹配对照和不改变的评价协议：
无法核实项（不得用猜测补空）：
输出文件、配置和结果身份：
```

**实施底线：** 能借鉴的成熟方法都允许竞争；引用用来说明来源和差异，而不是再次制造采用禁区。没有材料支持的参数允许作为明确的新建议，但不能写成作者原设。

## 13. 网络信息与原始入口索引（文末长期保留）

访问/检索日期均为 **2026-09-20**。W1–W3 沿用 v4 编号，W4–W26 为本次补充；W 编号标识网页，不是新论文编号。
“打开条目”“检索可见”“方法段落”“公开代码”是不同核验深度。失败/受限的入口也保留，避免后续把“本轮没读到”误作“不存在”。代码/项目链接不表示代码已运行。

<a id="W1"></a>
### W1｜DFormerv2 官方公开代码
- 对应论文：PR070。核验范围：**已读取公开代码片段**。
- 原始入口：<https://raw.githubusercontent.com/VCIP-RGBD/DFormer/main/models/encoders/DFormerv2.py>
- 支持内容：GeoPriorGen 的双线性 resize；Decomposed_GSA / Full_GSA 中 softmax 前加入几何偏置。
- 限制：可变 main 分支；未取得本地 fork / checkpoint；本轮不声称已锁定公开 commit。首次实施时固定 commit 并对照本地函数。

<a id="W2"></a>
### W2｜GeomPrompt / GeomPrompt-Recovery 原始 HTML 与 arXiv 条目
- 对应论文：AI023。核验范围：**已读取方法/训练段落及元数据**。
- 原始入口：<https://arxiv.org/html/2604.11585>
- 支持内容：§3：区分 RGB-only 几何提示与受损 Depth 残差修正；无 Depth GT；训练含分割目标与正则。arXiv 条目标注获 CVPR 2026 URVIS Workshop 接收。
- 限制：不是 CVPR 主会身份；未独立核定 workshop 正式 DOI。原输入量纲、幅度、损失权重不自动适用于 MUSeg。
- 同源补充入口：<https://arxiv.org/abs/2604.11585>。

<a id="W3"></a>
### W3｜Condition Dropout (ConD) 原始 HTML 与 arXiv 条目
- 对应论文：AI024。核验范围：**已读取方法/训练段落及元数据**。
- 原始入口：<https://arxiv.org/html/2607.20326v1>
- 支持内容：§II：冻结原编码器、训练复制编码器、零初始化特征注入；三种模态可用性训练。
- 限制：原方法不等同于轻量 adapter；对 DFormer 的结果不等同于 DFormerv2-S。摘要仍称代码将在接收后公开，不能断言代码已发布或推理零开销。
- 同源补充入口：<https://arxiv.org/abs/2607.20326>。

<a id="W4"></a>
### W4｜DFormerv2 CVPR 2025 官方论文条目
- 对应论文：PR070。核验范围：**官方检索条目可见；直开失败**。
- 原始入口：<https://openaccess.thecvf.com/content/CVPR2025/html/Yin_DFormerv2_Geometry_Self-Attention_for_RGBD_Semantic_Segmentation_CVPR_2025_paper.html>
- 支持内容：题名、作者、CVPR 2025 身份和摘要中的几何先验定位。
- 限制：不能把检索条目当本轮重新读完主文/补充材料；实现细节另看 W1。

<a id="W5"></a>
### W5｜MUSeg 数据集原始论文
- 对应论文：RE326。核验范围：**已打开出版商原始页面**。
- 原始入口：<https://www.nature.com/articles/s41597-025-05493-9>
- 支持内容：MUSeg 数据集与其地下矿山 RGB-D 场景来源；DOI 页面可定位。
- 限制：本项目 train-dev / val-dev 划分、A2 结果与失效 manifest 仍来自 S6/S1，不能归为数据集论文给出的数值。

<a id="W6"></a>
### W6｜MaskMentor DOI 与作者投稿入口
- 对应论文：AI019。核验范围：**DOI 直开失败；原始 PDF 入口已定位，未读取**。
- 原始入口：<https://doi.org/10.1145/3664647.3681698>
- 支持内容：保留 S13 的题名—DOI 索引和 OpenReview 原始检索入口。
- 限制：本轮没有重新核 PDF 方法；MaskMentor 机制仍引 S2。OpenReview forum 出现访问验证，不能据此认定论文不存在。
- 同源补充入口：<https://openreview.net/forum?id=j520wKxnf6>；<https://openreview.net/pdf?id=j520wKxnf6>；<https://openreview.net/pdf/903463b76ae48357e1feb57bbd45e76afc9235c3.pdf>。

<a id="W7"></a>
### W7｜RobustSeg CVPR 2026 官方条目
- 对应论文：PR090。核验范围：**官方检索题名/摘要可见；直开失败**。
- 原始入口：<https://openaccess.thecvf.com/content/CVPR2026/html/Tan_Towards_Robust_Multi-Modal_Semantic_Segmentation_with_Teacher-Student_Framework_and_Hybrid_CVPR_2026_paper.html>
- 支持内容：正式题名、作者 Jiaqi Tan / Xu Zheng / Yang Liu 与 teacher-student / hybrid prototype distillation 定位。
- 限制：具体结构、损失与超参数须锁定 CVPR 版本；不要与 W8 的不同 arXiv 版本拼接。

<a id="W8"></a>
### W8｜与 PR090 相关的同作者 arXiv 多版本记录
- 对应论文：PR090。核验范围：**已检索到相关版本；不自动合并方法**。
- 原始入口：<https://arxiv.org/abs/2505.12861>
- 支持内容：相关预印本题名/模块曾变化：RMMSS；v1 另有 Representation Regularization 表述。
- 限制：本轮未完成版本逐段比对，不能把全部版本的模块都算进 CVPR RobustSeg。只作版本核对入口。
- 同源补充入口：<https://arxiv.org/html/2505.12861v1>。

<a id="W9"></a>
### W9｜Calibrated RGB-D Salient Object Detection 官方条目
- 对应论文：AI017。核验范围：**已检索官方题名/作者/摘要**。
- 原始入口：<https://openaccess.thecvf.com/content/CVPR2021/html/Ji_Calibrated_RGB-D_Salient_Object_Detection_CVPR_2021_paper.html>
- 支持内容：CVPR 2021 深度校正方法身份。
- 限制：具体连续插值和 DCF 机制本轮沿用 S3；未重读原 PDF。

<a id="W10"></a>
### W10｜MoSA 出版商检索入口
- 对应论文：MoSA。核验范围：**官方索引题名可见；页面正文受限**。
- 原始入口：<https://ieeexplore.ieee.org/document/11523456>
- 支持内容：检索题名为 MoSA: Modality-Aware Spatially-Adaptive Adaptation for Multimodal Semantic Segmentation。
- 限制：与附件 RGB-X 题名保留为版本差异；未独立确认该入口与附件 DOI 的完整对应或全部方法细节。
- 同源补充入口：<https://ieeexplore.ieee.org/abstract/document/11523456>。

<a id="W11"></a>
### W11｜UMFNet CVPR 2026 官方条目
- 对应论文：PR029。核验范围：**官方检索题名/摘要可见；直开失败**。
- 原始入口：<https://openaccess.thecvf.com/content/CVPR2026/html/Wang_Uncertainty-Aware_Modality_Fusion_for_Unaligned_RGB-T_Salient_Object_Detection_CVPR_2026_paper.html>
- 支持内容：未对齐 RGB-T SOD 的 uncertainty-aware fusion 方法身份。
- 限制：空间置信与调制细节沿用 S7/S3；不是重新完成 PDF 全文复核。

<a id="W12"></a>
### W12｜SGMA 出版商入口
- 对应论文：PR089。核验范围：**官方检索条目可见；直开失败**。
- 原始入口：<https://ieeexplore.ieee.org/abstract/document/11517493>
- 支持内容：SGMA 题名与不完整遥感多模态分割定位。
- 限制：DOI 仍按 S11/S1 记录；本轮没有独立完成注册元数据与全文核验。
- 同源补充入口：<https://ieeexplore.ieee.org/document/11517493/authors>。

<a id="W13"></a>
### W13｜ECoLaF WACV 2025 官方条目
- 对应论文：AI020。核验范围：**已检索官方题名/摘要**。
- 原始入口：<https://openaccess.thecvf.com/content/WACV2025/html/Deregnaucourt_A_Conflict-Guided_Evidential_Multimodal_Fusion_for_Semantic_Segmentation_WACV_2025_paper.html>
- 支持内容：证据冲突指导的多模态分割身份。
- 限制：本轮不从摘要推定具体 DS 折扣参数；原始机制细节继续引 S2。

<a id="W14"></a>
### W14｜QMF：ICML 2023 / PMLR 原始页面
- 对应论文：AI021。核验范围：**已打开原始论文条目及 arXiv 元数据**。
- 原始入口：<https://proceedings.mlr.press/v202/zhang23ar.html>
- 支持内容：Provable Dynamic Fusion for Low-Quality Multimodal Data；ICML 2023，PMLR 202。
- 限制：理论条件的使用范围按 S2 原审计且需再核定理；不把样本层结果直接当几何关系定理。
- 同源补充入口：<https://arxiv.org/abs/2306.02050>。

<a id="W15"></a>
### W15｜MAGIC arXiv 原始条目
- 对应论文：AI022。核验范围：**已打开题名/摘要条目**。
- 原始入口：<https://arxiv.org/abs/2407.11344>
- 支持内容：为原 ECCV DOI 索引补充原始预印本检索入口。
- 限制：没有在本轮重新核对全部实现与正式版差异。

<a id="W16"></a>
### W16｜SimMLM ICCV 2025 官方条目
- 对应论文：AI026。核验范围：**已检索官方题名/摘要**。
- 原始入口：<https://openaccess.thecvf.com/content/ICCV2025/html/Li_SimMLM_A_Simple_Framework_for_Multi-modal_Learning_with_Missing_Modality_ICCV_2025_paper.html>
- 支持内容：Sijie Li / Chen Chen / Jungong Han；缺失模态学习与动态专家背景。
- 限制：具体 DMoME / MoFe 细节沿用 S3；不能直接推为有噪声模态越多越好。

<a id="W17"></a>
### W17｜CoReFuse-Med arXiv 原始条目与 HTML
- 对应论文：AI025。核验范围：**已读取元数据/摘要，定位原始 HTML**。
- 原始入口：<https://arxiv.org/abs/2609.10261>
- 支持内容：题名、作者、2026-09-09 v1；摘要标注 ACM MM 2026 已接收；主要考虑分辨率退化。
- 限制：ACM DOI 源记录保留；不宣称会议已召开。新增 arXiv DOI 页面标为 pending registration，不能写注册已完成。
- 同源补充入口：<https://arxiv.org/html/2609.10261v1>。

<a id="W18"></a>
### W18｜UMQ arXiv 原始条目
- 对应论文：AI027。核验范围：**已打开题名/摘要条目**。
- 原始入口：<https://arxiv.org/abs/2603.02695>
- 支持内容：补充可定位的原始论文入口。
- 限制：细粒度质量排序/路由机制沿用 P2，不冒称重新逐式核验或确认正式发表。

<a id="W19"></a>
### W19｜PRIME arXiv 原始条目
- 对应论文：AI028。核验范围：**已打开题名/摘要条目**。
- 原始入口：<https://arxiv.org/abs/2608.03475>
- 支持内容：补充可定位的原始论文入口。
- 限制：恢复—复诊细节沿用 P2，不冒称重新逐式核验或确认正式发表。

<a id="W20"></a>
### W20｜ANGA CVPR 2026 官方条目
- 对应论文：ANGA。核验范围：**已检索官方题名/作者/摘要**。
- 原始入口：<https://openaccess.thecvf.com/content/CVPR2026/html/Guan_Anchor-Guided_Gradient_Alignment_for_Incomplete_Multimodal_Learning_CVPR_2026_paper.html>
- 支持内容：Zhi-Hao Guan / Longfei Huang / Yang Yang；正式检索入口已找到。
- 限制：DOI 尚未确认；不猜造 DOI，不另分 AI 号。

<a id="W21"></a>
### W21｜Confidence Propagation 原始条目
- 对应论文：AI029。核验范围：**已打开题名/摘要条目**。
- 原始入口：<https://arxiv.org/abs/1811.01791>
- 支持内容：稀疏深度回归/置信传播历史近邻；恢复 v3 外部文献索引。
- 限制：本轮未重新核网络与公式；不是已移植实验。

<a id="W22"></a>
### W22｜Uncertainty-Aware CNNs for Depth Completion 原始条目
- 对应论文：AI030。核验范围：**已打开题名/摘要条目**。
- 原始入口：<https://arxiv.org/abs/2006.03349>
- 支持内容：深度补全不确定性历史近邻；恢复 v3 外部文献索引。
- 限制：本轮未重新核网络与公式；不是已移植实验。

<a id="W23"></a>
### W23｜多模态语义分割传感器失效 benchmark 原始条目
- 对应论文：AI010。核验范围：**已打开题名/摘要条目**。
- 原始入口：<https://arxiv.org/abs/2503.18445>
- 支持内容：为 AI010 补充原始预印本入口；缺失/噪声模态评价背景。
- 限制：本项目的具体故障表、bootstrap 与阈值不是该文参数。
- 同源补充入口：<https://arxiv.org/html/2503.18445>；<https://ieeexplore.ieee.org/abstract/document/11147584>。

<a id="W24"></a>
### W24｜DFormer 原始条目
- 对应论文：AI004。核验范围：**已打开题名/摘要条目**。
- 原始入口：<https://arxiv.org/abs/2309.09668>
- 支持内容：区分 DFormer 与 DFormerv2；普通强基线来源。
- 限制：未重新核所有训练和预训练版本。

<a id="W25"></a>
### W25｜GeminiFusion 原始条目
- 对应论文：AI005。核验范围：**已打开题名/摘要条目**。
- 原始入口：<https://arxiv.org/abs/2406.01210>
- 支持内容：普通多模态强基线检索入口。
- 限制：未重新核所有训练和预训练版本。

<a id="W26"></a>
### W26｜PR059 原始预印本元数据
- 对应论文：PR059。核验范围：**已核题名/作者/提交日期及预印本 DOI**。
- 原始入口：<https://arxiv.org/abs/2603.07570>
- 支持内容：2026-03-08 v1 与附件标题/作者相符；预印本 DOI 为 10.48550/arXiv.2603.07570。
- 限制：未证实附件 KBS 327 (2025),114107 的正式版对应；两条记录不静默合并。

### 已定位的作者项目/代码入口（不等于代码审计）

| 论文 | 入口 | 来源 | 状态 |
| --- | --- | --- | --- |
| AI023 | <https://geomprompt.github.io/> | W2 | 原始 arXiv 条目链接；未检查代码内容/复现可用性 |
| AI025 | <https://github.com/lrever/CoReFuse> | W17 | 原始论文链接；未检查仓库 commit 或运行 |
| AI026 | <https://github.com/LezJ/SimMLM> | W16 | 原始论文检索结果提供的代码入口；未运行 |
| AI022 | <https://vlislab22.github.io/MAGIC/> | W15 | 原始论文关联入口；未核代码 |
| AI010 | <https://github.com/Chenfei-Liao/Multi-Modal-Semantic-Segmentation-Robustness-Benchmark> | W23 | 原始论文关联入口；未核代码 |
| PR070 | <https://github.com/VCIP-RGBD/DFormer> | W1 | 官方代码仓库入口；已读特定源码，不代表全仓库审计 |

**ConD 代码说明：** W3 摘要仍写接收后公开；本轮没有确认可用的正式代码仓库，不编造下载入口。

### 原附件已经记录的网络入口：完整保留，不自动标为本轮访问

以下入口直接从 S1–S14 提取。与 W 表重复者是同一来源的不同索引；其他条目只表示附件曾记载。版权许可链接仅保留出处，不作方法依据。

| 原附件入口 | 首次及补充记录位置 | 本轮状态 |
| --- | --- | --- |
| <https://raw.githubusercontent.com/VCIP-RGBD/DFormer/main/models/encoders/DFormerv2.py> | S1:L1010；S5:L499 | 见 W1 |
| <https://openaccess.thecvf.com/content/CVPR2025/html/Yin_DFormerv2_Geometry_Self-Attention_for_RGBD_Semantic_Segmentation_CVPR_2025_paper.html> | S1:L1011 | 见 W4 |
| <https://www.nature.com/articles/s41597-025-05493-9> | S1:L1012 | 见 W5 |
| <https://arxiv.org/abs/2604.11585> | S1:L1013 | 见 W2 |
| <https://arxiv.org/abs/2607.20326> | S1:L1014 | 见 W3 |
| <https://arxiv.org/abs/2603.02695> | S1:L1015 | 见 W18 |
| <https://arxiv.org/abs/2608.03475> | S1:L1016 | 见 W19 |
| <https://arxiv.org/abs/1811.01791> | S1:L1017 | 见 W21 |
| <https://arxiv.org/abs/2006.03349> | S1:L1018 | 见 W22 |
| <https://arxiv.org/html/2604.11585> | S5:L500 | 见 W2 |
| <https://arxiv.org/html/2607.20326v1> | S5:L501 | 见 W3 |
| <https://github.com/VCIP-RGBD/DFormer> | S7:L29 | 仅收录附件入口；未重新检查 |
| <https://joannelin168.github.io/research/ELVIS> | S8:L11；S8:L11 | 仅收录附件入口；未重新检查 |
| <https://github.com/ZeAstra/BiPA> | S8:L19 | 仅收录附件入口；未重新检查 |
| <https://gaogehan.github.io/A2P/> | S8:L29 | 仅收录附件入口；未重新检查 |
| <https://sohyun-l.github.io/RobustPVOS_project_page/> | S8:L122 | 仅收录附件入口；未重新检查 |
| <https://github.com/work-submit/3MTI> | S8:L178 | 仅收录附件入口；未重新检查 |
| <https://github.com/Giluir/Spatial-SAM> | S8:L289 | 仅收录附件入口；未重新检查 |
| <https://github.com/google-deepmind/representations4d> | S8:L339 | 仅收录附件入口；未重新检查 |
| <https://threedle.github.io/bsb/> | S8:L444 | 仅收录附件入口；未重新检查 |
| <https://github.com/mumu-k/TPSegformer> | S8:L452 | 仅收录附件入口；未重新检查 |
| <https://github.com/liuuuuuuxuesong/SARD> | S8:L468 | 仅收录附件入口；未重新检查 |
| <https://github.com/zjhe02/CRSDUN> | S8:L492；S8:L492 | 仅收录附件入口；未重新检查 |
| <https://machineperceptionlab.github.io/SGAD-SLAM-Project> | S8:L571 | 仅收录附件入口；未重新检查 |
| <http://creativecommons.org/licenses/by-nc-nd/4.0/> | S10:L492 | 仅收录附件入口；未重新检查 |
| <https://github.com/HT-hlf/MD-TLCF.git> | S10:L922 | 仅收录附件入口；未重新检查 |