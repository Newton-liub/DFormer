# DFormer 项目实时状态

> 事实时间：实验与科研审核事实更新至2026-10-13；论文库与研究材料保存事实更新至2026-10-13；云端旧产物盘点事实更新至2026-10-14。本文件是当前事实、授权边界与恢复点的唯一入口。ODG实现/验收证据见[无卡准备记录](../reports/2026-10-10-odg-no-gpu-preparation.md)，首轮30 epoch结果见[30epoch报告](../reports/2026-10-10-odg-sunrgbd-30epoch.md)，第二轮（修复后）结果见[R2报告](../reports/2026-10-10-odg-sunrgbd-r2-30epoch.md)，严重缺失筛选见[50%/75%孔洞筛选报告](../reports/2026-10-10-odg-r2-holes5075-screening.md)，固定协议与命令见[ODG入口](../guides/odg.md)。已结束论文库阶段摘要见[历史状态](../archive/2026-10-10-odg-start/paper-library-state.md)。2026-10-13 已完成 `临时/` 正式清理，清理时仅保留 `临时/key.txt`；随后科研交接材料重新提供，当前新方案恢复点见下文 OC-RGA v0.3。

**2026-10-13科研深度提取已完成并停止：** 按当前授权，通过`paper-supplement`受支持的串行plan→真实SHA-256→append apply保存10篇旧论文（LIB000002/006/029/030/035/036/037/042/043/049，revision23–32）与全部23篇新论文（LIB000050–072，revision33–55），共33篇不同论文；Cocoon另追加FI名称/跨解码层稳定性/开销措辞更正，最终索引为**schema5/revision56**，指纹`8145a2f2278bce184b24fa3a62170531057a66f874f781462dd98779928e9405`。append保留既有补充，阅读状态、人工编号、核心标记和无关元数据不作为编辑对象；新23篇均保留`abstract_only`。活动67/退库5/总72、next_lib_number73不变；全库有非空补充50篇。受支持`views-refresh`已刷新三份AI导出，直接回读头部确认源revision56、全部67/核心15/已有补充50，时间2026-10-13T09:26:47+08:00；逐篇定位与导出正文定点回读已完成，不宣称全库逐字节审计。保存回执见[该报告](../reports/2026-10-12-research-extraction-and-supplements.md)；本轮草稿与计划、项目证据包均已在 2026-10-13 临时清理中删除。用户交付物为正式论文库Supplement与该独立项目证据包，报告仅内部回执；未创建或编辑`RESEARCH_PREPARATION_PACK.md`或另一综合包。本轮未训练、推理、GPU/云操作、下载、转换、代码/配置修改、项目测试、资料移动/删除或Git提交/推送，未选最终方向。**2026-10-13 已交接到位：** 用户手动导出全部活动67篇AI阅读Markdown并与项目证据包一并交云端高级模型，该模型已完成下一阶段科研方案设计；导出不自动附原始全文。

**2026-10-13 论文库资料完整性核验已完成（只读）：** 按用户指令核验当前论文库与 AI 导出，为云端高级模型深度推导/网络设计准备可追溯资料。确认索引 schema5/**revision56**、指纹 `8145a2f2…e9405`（与保存时一致）、`next_lib_number=73`、活动67/退库5/总72，LIB 编号沿用未重排。三份 AI 导出（all 67 / core 15 / supplemented 50）**均已是源 revision56 最新版，无需重新生成**，逐一计算 SHA-256；导出条目与活动集合逐条一致、0 缺条、0 失效链接。67/67 篇 bundle 的 `resource_counts` 与磁盘文件数**完全相符**，正文入口 md、Figure（516）、Formula（872 图 + 63 篇 LaTeX）、Tables（526）、Word（65）齐备。**明确：导出格式 1.0.3 只含书目+摘要+人工补充，不含正文、不含任何图片、不含公式/表格文件**；仅凭导出可读 67 摘要与 50 补充（42 篇含 LaTeX 公式、48 篇含图表号引用、15 核心全有补充），结构图/结果图/公式文件/xlsx 均需另行单独提供。已记录缺口：无公式资源 4 篇（LIB000029/035/056/070）、无补充 17 篇、图编号空档 20 篇、表编号空档 8 篇、缺 Word 汇总 2 篇（LIB000006/008）；未改索引、未改/移动/删除 bundle、未重提取、未新建格式、未跑测试/云端/Git。报告见[完整性核验报告](../reports/2026-10-13-paper-library-material-integrity.md)。

**2026-10-13 临时目录正式清理已完成：** 经用户授权并按清理预审报告执行，删除 `临时/` 下全部历史过程资料共 100 个文件（废止 Rule/Skill 替换稿与任务 00 备份、已完成任务指令、33 篇论文补充草稿与 35 份 `LIB*.plan.json`、入库计划 JSON、维护请求/计划 JSON、阶段报告、项目证据包、研究准备包、参考指令与书目清单、3 份核验材料），预审报告本身按同一指令删除，目录仅保留 `临时/key.txt`。未改动正式论文库与索引、现行 Rule/Skill、工程代码与科研方案；本文件原指向 `临时/` 的链接已改为正式报告路径或作废。

**2026-10-12书目收尾已由前一维护任务执行，作为本轮阅读基线：** revision20→21（LIB000054正文H1题名修正，目录/文件仍`Eficient`）→22（LIB000068正式卷期年2025并保留接收2024；LIB000070人工论文DOI与auto数据集DOI分离；指定等级写法统一）。前一任务已清理其隔离演练残留；临时/历史资料已于 2026-10-13 单独清理。具体证据见[书目收尾报告](../reports/2026-10-12-paper-library-biblio-fix.md)，下列revision20计数与未处理项属于此前阶段的历史快照，以本段与本轮保存记录覆盖当前事实。

**2026-10-12 论文库阶段一（工程功能，已通过独立验收）与阶段二（任务03正式数据维护，已执行）均已完成。** 阶段一按已审核的维护能力扩展计划（正式记录见[阶段一报告](../reports/2026-10-12-paper-library-maintenance-stage1.md)）完成 `paper_library_store.py`（schema 5、独立 `rating`、可恢复 `status=retired` + `human.retirement`、共享维护 helper、`held_lock` 兼容）、`paper_workflow.py`（`maintenance-plan` / `maintenance-apply` / `views-refresh`）、`paper_library.py`（retired 保留与扫描命中保护）、`paper_library_ui.py`（年份/年份语义/出版物/论文等级独立编辑、差异显示、retired 只读与筛选）四处工具改动，并同步 Rule、`paper-maintenance` 新 Skill 与两个既有 Skill、两份正式指南、`_index/README.md` 与原任务03文件；`.codebuddy` 侧已由仓库自带单向同步脚本补齐。最终代码版本已由独立验收模型在隔离环境实测通过 plan→写入→退库→恢复→导出（见[最终验收报告](../reports/2026-10-12-paper-library-maintenance-stage1-acceptance-final.md)），**阶段一关闭**。

**阶段二（任务03）已按明确授权执行完毕：** 一次 `maintenance-apply`（计划 SHA-256 `ba0ec183…`，保存前指纹 `116f314a6bb137c7`）把索引从 schema4/revision19 升到 **schema5/revision20**，`next_lib_number` 仍为 73；**67 篇写入独立 `rating`**（其中 26 篇补/改 `year`、26 篇补 `venue`，含旧库 4 个缺口与 LIB000069 的 `ACM Multimedia 2025` 出版物纠错），并**按指定 5 篇可恢复退库**（LIB000004/009/011/021/022，同卷移入 `D:\0Project\origin\论文_retired\<LIB>\<原目录名>\`）。现状为**活动 67 / 已退库 5 / 总登记 72**；管理视图与三份 AI 导出已重建（all 67 / core 15 / supplemented 21，源 revision 20）。以保存前备份 `backups\PAPER_LIBRARY_INDEX.json.bak1` 做白名单差分，**0 处越权变化**。详见[任务03验收报告](../reports/2026-10-12-paper-library-task03-acceptance.md)。

## 当前阶段

**2026-10-14 OC-RGA 代码已实现并完成云端准备，等待本地主力模型审核代码结构与训练方案。** 实现提交 `ce24e10`（`feat(oc-rga): research model, data protocol, losses and entries`）已在 `origin/research/observation-relational-geometry` 且云端同步；DeLiVER train+val 三模态各 5988 文件就位，无卡只读检查（配置/入口导入、日程、真实样本读取）通过，**未跑完整 `--check`**。下一步：本地主力模型审核 `research/oc_rga.py`、`oc_rga_model.py`、`oc_rga_data.py`、`oc_rga_losses.py`、`train_oc_rga.py`、`evaluate_oc_rga.py` 与 `local_configs/research/OC_RGA_DeLiVER.py`（清单见[云端准备报告](../reports/2026-10-14-oc-rga-cloud-preparation.md)第 6 节），通过后交上级复核，再上 4090 做 30 epoch 性能试跑（查显存/吞吐/GPU 利用率），最后才推进完整训练。**GPU 尚未授权。**

**2026-10-13 OC-RGA v0.3 实施设计已完成，等待交回云端高级模型审核，尚未开发或训练。** 主力模型结合新方案 v0.2、重新提供的项目证据包、真实 DFormerv2/DeLiVER/训练源码与定点近邻原文，已就地更新[OC-RGA v0.3 实施版](../../临时/OC_RGA_RESEARCH_DESIGN_v0.2.md)（沿用文件名，页内版本为 v0.3，不新建平行计划）。明确 RGB-only 1/4 几何分支、rank20 局部—全局先验、支持量感知精度、一次低秩 Cholesky 后验、精确非重叠尺度传播、相关差值核与有界先后验增量；补齐无泄漏联合几何/观测监督、真实文件/API、匹配 mask-aware baseline、三读出消融及执行任务单。首轮设计为 DeLiVER、120epoch日程/50epoch观察点、480crop/共同640评价；4090首两组预计11–19GPU小时、建议24h申请上限，**均为未实测/未授权的规划值**。数学推导与内容已复核，不代表算子/梯度/GPU验收通过；核心有效性与新颖性仍待实验。本轮仅改新方案文件与本状态，旧总方案/代码/配置/论文库未改，未运行项目测试、推理、训练或云操作，未创建分支、提交/推送；没有云端实际审核回执。

### 方向 A：已完成审核的历史恢复点

**2026-10-13 方向 A 第一阶段已实施并完成主模型科研审核，当前暂停，等待云端高级模型最终裁决。** 方案基线 `74456eb`、实现与[冻结诊断执行报告](../reports/2026-10-13-influence-frozen-diagnostic.md)提交 `b9c094b`；用户确认两者已推送，本地远端缓存与 HEAD 相符，本轮未联网检查远端。产物为 `outputs/influence-sun-diagnostic/20261013-influence-a/`（pilot32、noncyclic32、confirm496）。[正式审核报告](../reports/2026-10-13-influence-research-audit.md)定点核对 KL 共同因子求解、signed bias、W/H、共同 FP32、HAM 配对及 confusion 合并：核心前向与 O/A/C/G 主要比较可信；528 图 Original clean/Holes75 为 44.7845/42.2998，A 为 44.6505/42.1757，独立 496 图同向阴性。**孔洞造成 2.4847pp 任务损害，整个操作域外也有净损害；“错误几何 attention 关系导致传播”尚缺固定块输入探针，不能把 OFF 仍较低解释为没有损害。** OFF 40.8470 实为 496 图，误置于合并表；4.94% 为 block/axis 组等权超预算行率，合并行率约 11.00%，原报告 20.3% 为单组键率而非合并率。原始数据/执行报告保持不变，更正口径见审核报告。**主模型选择 C：暂停当前统一预算全块投影路线，保留数学/实现/局部信号，不直接短训、不追加机制诊断；等待上级决定是否转向更有希望的问题，不自动实现方向 B。** 本轮仅只读审核、既有计数核算和正式文档/状态写入，未重跑自检/完整评价、开发反向、训练、云操作、提交或推送；没有云端最终提交/裁决回执。

### 上一阶段：ODG 实验事实与恢复资产

用户已批准[首轮GPU实验计划](../plans/2026-10-10-odg-gpu-experiment.md)并确认SUN包来自作者入口。**2026-10-10首轮GPU实验已完成：original与odg各30 epoch（300 epoch日程的暂停点）在云端并行跑完，实例已主动停止并回读`Stopped`，产物已在无卡A窗口取回本地。** dev单尺度无翻转mIoU：original 26.01/36.82/**39.39**、odg 24.92/34.83/**36.97**（epoch 10/20/30），ODG落后1.09/1.99/2.42点且差距扩大。**这三个验证点是在 train 模式下取得的（见下一段），不是标准 eval 结果**；上级已裁决该负向趋势不再作为淘汰依据。按计划30epoch停止条件，**不自动续训100epoch**；正式test、mean消融、300epoch均未授权、未运行。

2026-10-10[科研诊断报告](../reports/2026-10-10-odg-research-diagnosis.md)已获上级审核通过。CPU直接核验发现两组29个`Geo.weight`均遗漏于优化器（714个可训练参数张量，仅685个入组），30epoch与预训练逐元素相同；此前“几何权重已更新/所有可训练层更新”的描述已更正。三张dev深度关系抽样显示ODG bias较原关系减弱，天然零值语义仍未知。

**2026-10-10评价配置对齐与孔洞复测已完成，等待上级科研裁决**（[复测报告](../reports/2026-10-10-odg-bn-align-reevaluation.md)；此前未对齐的孔洞结果见[孔洞评价](../reports/2026-10-10-odg-holes25-evaluation.md)）：独立评价入口已恢复训练时的解码器 BatchNorm 数值（`eps=0.001`，CPU 三路径定点核对，792 个浮点张量不变），但四组结果只变化 ≤0.01 点——**`eps` 不是 clean 与训练内不一致的原因**。修复后 clean 为 Original **41.32**、ODG **42.77**，**仍不能复现**训练内 39.39/36.97；定点诊断（同权重、同 528 图、仅令模型保持 train 模式）得 39.09/37.22，并复现训练记录中 `floor_mat`/`shower_curtain`/`night_stand` 三类 IoU 为 0 的特征，证实 `research/train_odg.py` 的周期验证从未调用 `model.eval()`（作者的 `utils/train.py` 在验证前调用了）。25% 孔洞下 ODG 仍 **+2.96** 点、退化改善 1.51 点，正向信号保留；上级已据此裁决 ODG 具有继续研究价值，“ODG 落后 2.42 点”的 train 模式趋势不再作为淘汰依据。

**2026-10-10 训练代码修复与第二轮准备已完成（本地，未启动 GPU）。** 周期验证改为 eval 模式（验证后恢复训练模式，验证前后比对 BatchNorm running 统计量指纹，发生变化即报错）；优化器参数分组改由研究层实现，714 个可训练参数张量恰好各入组一次（decay 302 / no_decay 412，含此前遗漏的 29 个 `GeoPriorGen.weight`），缺失或重复即报错；训练合同新增 `validation_mode=eval` 与 `optimizer_param_scope=all_trainable`，使修复前的 checkpoint 无法被新代码 resume。ODG 设计、超参、数据划分、300 epoch 日程均未改。[第二轮重训计划](../plans/2026-10-10-odg-retrain-30epoch-r2.md)已就绪，等待用户授权 GPU。

**2026-10-10 R2 第二轮 GPU 实验已完成，结果待上级裁决**（[R2报告](../reports/2026-10-10-odg-sunrgbd-r2-30epoch.md)）：修复后的两组从同一官方预训练权重并行（`4×4`）各跑 30 epoch，均 Succeeded（各 8940 次尝试，无失败、无异常）。优化器报告 714/714、`duplicates=0`、`missing=[]`；验证记录 `model_mode=eval` 且带 BN 指纹（两个 epoch-10 指纹的 `batches` 完全相同，证明验证未更新 BN）。训练内 eval 模式 dev：original 33.80/43.60/**45.02**、odg 34.58/43.23/**45.04**（epoch 10/20/30）。本地对齐评价（同入口）：original clean **45.34**→孔洞 **44.70**（降 0.64），odg clean **45.03**→孔洞 **44.45**（降 0.58）；**ODG−original 为 clean −0.31、孔洞 −0.25，R1 的正向优势（+1.45/+2.96）未复现**，且该差异与同协议下本地/云端复现差（约 0.3 点）同量级。费用：有卡 3.303 GPU 小时 ≈¥6.21，无卡取回窗口 4 分 11 秒 ≈¥0.01，合计 ≈¥6.22（上限 ¥15/6 GPU 小时）。实例已主动停止并回读 `Stopped`，计划关机已取消并回读。未自动续训、未做消融、未跑正式 test。

**2026-10-10 R2 严重缺失筛选已完成，建议停止当前方法**（[50%/75%孔洞筛选报告](../reports/2026-10-10-odg-r2-holes5075-screening.md)）：上级审核通过 R2 结果并批准最后一次低成本筛选。本地 4 组授权推理 + 2 组 75% 第二孔洞 seed 复现检查（共 6 次，均正常退出）显示：退化幅度 original −0.64/−1.69/−2.85、ODG −0.58/−1.28/−1.58（25/50/75%）；绝对差 ODG−original 为 clean −0.31、25% −0.25、50% **+0.10**、75% **+0.96**（seed12345）/ **+0.83**（seed54321）。即**只有 75% 严重缺失下存在约 0.8–1.0pp、跨两个孔洞 seed 可复现的小幅优势**，接近但未明显超过“约 1pp”判读线，clean 反而落后 0.31pp。删除像素组成检查：支持域 82.29% 为非零深度；删除像素中非零深度占 85.99/85.94/84.68%，75% 孔洞实际移除 77.18% 的非零深度像素（未把天然零值当作物理缺失）。**建议停止当前方法、不续训；仅当上级要求时才做 75% 条件下的 Mask 中性化消融。**


### 首轮30 epoch结果与执行事实（2026-10-10，详见报告）

- 组合`MICRO_BATCH=4 ACCUM_STEPS=4`（有效batch16），两组同组合、同合同（`contract.json`仅`geometry_mode`不同）；两组**并行**在同一4090上运行，GPU利用87–99%，比串行方案省约1.4–1.6小时。
- 指标与逐类差异、loss趋势、性能诊断短测（`4×4`15.82、`8×2`22.11、`16×1`OOM未测、并行`4×4`合计24.46 samples/s）见[30epoch报告](../reports/2026-10-10-odg-sunrgbd-30epoch.md)。
- 费用：有卡会话2026-10-10 09:04:34–13:23:02，4.31 GPU小时、约¥8.1（上限¥40/20小时未触及）。两组各8940次尝试、8932次应用、8次AMP预热跳步，无失败记录。
- 性能瓶颈定位：数据加载与CPU均未饱和，瓶颈是每次迭代约0.116s加每次optimizer尝试约0.5s的固定开销；未改任何代码。

## 已确认事实

### 研究合同与实现

- 主数据集SUN RGB-D，NYUv2第二验证集；MUSeg仅可选应用扩展，DeLiVER工程接入已完成（25类，尚未正式实验），不替换SUN主线。DFormerv2-S + 当前HAM宽度1024、37类，只实施ODG，不叠加教师/补偿/新损失。
- `research/geometry.py`构造输入观测16区间软分布→实际stage面积池化→归一化核log bias；固定区间覆盖作者归一化深度域。空观测只中性化depth项，原spatial/QKV/RoPE/FFN/可学习权重符号保留。无新增可学习参数；前三stage轴向、最后full，同forward复用stage关系，不跨batch缓存。
- `geometry_mode=original|mean|odg`：original复用作者双线性stage深度差；mean先求同支持域均值再走相同分箱核，是正结果后的关键消融。区间中心点质量可退化，off-grid仅近似，不声称已证明创新或收益。
- 研究层为`research/{data,train_odg,evaluate_odg,odg_schedule,prepare_odg}.py`；独立配置`local_configs.research.ODG_SUNRGBD`及`ODG_NYUv2`。作者`utils/train.py`未改；模型只改必要接口、几何切换、单卡BN与HAM设备分配。
- 新baseline/candidate都只加载`checkpoints/pretrained/DFormerv2_Small_pretrained.pth`，新头、新optimizer/scaler，从头训练；不使用旧MUSeg/MMFR完整分割权重。常规编码器/分割头参数参与优化；**round-1 两组训练时29个`Geo.weight`因作者分组函数遗漏而未进入优化器（2026-10-10诊断直接核验），该问题已在训练入口修复，现要求714个可训练张量恰好各入组一次，缺失或重复即报错**；单卡关闭SyncBN且`norm_eval=False`。
- SUN：AdamW8e-5、wd.01、betas.9/.999、有效batch16、480×480、seed12345、warmup10、poly.9、总300epoch，继承尺度/翻转，不新增损坏增强、不compile。micro-batch/累积须GPU授权后确认并在两模型一致。
- LR在optimizer尝试之前写入真实param-group；AMP成功/跳过分别计数。`stop_after_epoch=30/100`只是暂停，不压缩300epoch。保存last、best-dev、30/100阶段点与optimizer/scaler/RNG/合同；resume限同实验epoch边界，smoke快照不可恢复正式训练。

### 数据、开发验证与CPU验收

- 本地`D:\0Project\dataset\SUNRGBD`：RGB/Depth/labels各10335，官方train5285/test5050。路径配对0缺失。固定图像级seed12345划分4757train-dev/528dev，清单在`research/splits/sunrgbd_seed12345/`，互斥且并集等于官方train；原train/test不改，不声称场景隔离。
- 周期验证只读dev，每10epoch、单尺度无flip。round-1 实际在 train 模式下进行；已修复为 eval 模式（验证后恢复训练模式，验证不得更新 BatchNorm running 统计量）。显式fulltrain用官方全train并关闭周期验证；正式test仅独立入口显式选择，不反复调参。五尺度+flip只在指定节点显式评价，数字不得与单尺度混比。
- 开发训练298次optimizer尝试/epoch，补11张到4768；总89400、warmup2980。全train331次/epoch。原始/ODG从同encoder预训练共同起步；旧MUSeg和论文分数不替代匹配SUN baseline。
- 灰度读取保留作者8位读法及`(D8/255-.48)/.28`；uint16 PNG被读为高8位，不直接除255。天然0保留为观测，支持域排除padding/非有限值/人工删除；不能用normalized tensor==0推断物理缺失。
- `D:\0Project\dataset\SUNRGBD.zip`已存在（2452204576bytes，SUNRGBD单顶层，作者整理格式`RGB/Depth/labels`+`train.txt/test.txt`，三模态各10335）；结构与体积与作者整理版一致，**2026-10-10已由用户确认为作者入口下载**。米制编码仍未知，本轮未发明映射、未改16位读取。
- 三模式各26966591参数，均成功加载官方encoder。missing为extra_norms标准LayerNorm初值，unexpected为不用的预训练头；没有补key、没有重哈希权重。
- 小张量常量/自关系/全空/部分空、轴向/full尺寸与有限性、FP32 autocast隔离、分块一致性均通过。ODG一次CPU batch2、64×64完整前后向loss4.1066256，714个可训练参数张量均有有限梯度。实际没有GPU或正式训练验收。
- 人工25%/50%孔洞会同时填共同depth输入与删mask，按矩形图像支持域精确取整计数；padding不改。非矩形支持域先拒绝并要求单独协议，不以错误严重度评价。五张数据示例在`outputs/odg-preparation/data-preview/`，只是输入对齐检查，不是收益图。
- NYUv2数据本地/云端均缺，已准备作者40类入口，不把test伪作dev；不阻塞SUN。SUN准备阶段未下载NYU；DeLiVER已于后续独立任务完成接入，正式实验未运行。可信整理版入口和目标结构见准备报告。

### Git、环境、监控与云端

- 作者基线仍为`e3273009b759b578945483828ff315d560be94c9`。任务从实际HEAD`08d5940`建立`research/odg-sunrgbd`。核心`71aa4e4`、训练入口`2b2d754`、启动脚本修正`4f3e405`均已推送用户origin；upstream push继续禁用。主体实现提交`4f3e40529399aff91d7af9fc9eb2450a1d752498`已在两端；收尾提交包含状态/报告和数据检查断言/同步临时文件清理入口，其身份用`git rev-parse HEAD`获取，不在文件内自指尚未生成的hash。
- 2026-10-13论文库收尾：`5cdc26d`（论文库 Rule、三个 Skill 与两份指南）与`e439c5a`（六份正式报告）先后提交到`research/odg-sunrgbd`并正常推送`origin`（`a11b835..e439c5a`快进，未 force push）；本文件随后单独更新一次。提交前工作区干净、远端无新增提交；六份报告均被本文件或阶段一报告引用，无删除。upstream、其他分支与 DeLiVER worktree 未动。
- 本机继续用`D:\2Env\anaconda\envs\dformer`：Python3.10.22、torch2.7.0+cu128、mmcv1.7.2、timm1.0.30、SwanLab0.10.1；旧df2冻结，未升级依赖栈。
- 当前会话未发现CompShare MCP工具，已改用既有官方CLI0.3.6与配置，没要求新密钥。实例`cpod-1vbh7faqcauq`、cn-bj2-03、Postpay、50GB系统盘；启动前Stopped且无其它screen/训练任务。
- 2026-10-10有卡会话（用户手动启动）：09:04:34开始、13:23:02主动停止并回读`Stopped`，平台计15508秒=4.31 GPU小时、约¥8.1。保险先设10:48:43、扩容到16:18:40、按实际进度调整为14:45:00并回读确认。随后以无卡A档（2CPU/4GiB/GPU0、`InstancePrice=0.13`，非免费）短窗取回产物，取回校验后再次主动停止并回读`Stopped`。挂起的`stage1`串行链与其screen、`odgpar` screen、诊断脚本进程均已清理，无遗留训练或自动任务。
- 云端工程`/root/rivermind-data/DFormer`，复用`/usr/local/miniconda3/envs/py310`：Python3.10.16、torch2.1.2+cu118、mmcv2.1.0、timm1.0.28、SwanLab0.10.1，官方pretrained已存在110203103bytes。没有开GPU或在4GB机器模拟全尺寸训练。
- 云端GitHub fetch发生TLS握手中止，已按许可用Git bundle fetch/fast-forward同步正常提交关系，不手工覆盖源码、不留云端未记录补丁。手动启动脚本`bash -n`及CPU日程已通过；未创建训练session、cron/start.d或将来自动GPU任务。
- SUN现成zip已上传至`/root/rivermind-data/dataset/SUNRGBD-odg-20261010.zip`并解压到相邻`SUNRGBD/`，原包保留。三模态各10335、train5285/test5050；云端四清单路径配对0缺失、少量train/dev样本CPU可读及padding通过，磁盘空闲约20GiB。8MiB短传5.513秒、保守预计26.86分钟才启动；实际1089.322秒（18分09秒），低于3小时。编码来源仍待确认，不把工程可读性冒称作者来源已闭环。
- 本地在线SwanLab init/log/finish实际上传完成16records：[CPU准备run](https://swanlab.cn/@Newton_liub/dformer-research/runs/bwmgfa13)。GBK错误后只针对性改`-X utf8`重试一次成功。云端登录存储存在，disabled和offline生命周期成功；云端online本轮未新测。key未打印、纳入Git或配置，未直接读取`临时/key.txt`。
- 旧云端工程`/root/rivermind-data/DFormer-archive-20261008`、旧cloud结果/MUSeg/pretrained库，以及本地`D:\0Project\DFormer-archive-20261007\`均保留，不迁回或覆盖。旧实验结论仍以[交接报告](../reports/2026-10-09-research-handoff.md)为准。

### 已结束的论文库工作

- 2026-10-09论文库保存结果为历史基线：schema4/revision18、49篇、核心15、supplement21；当时三范围AI导出和回执就绪，旧revision17/F0审核计划已失效，保存回执与历史授权见[归档摘要](../archive/2026-10-10-odg-start/paper-library-state.md)。
- 2026-10-12任务00/01已完成：论文库规则与Skill简化提交为本地提交`85f228e`（已于 2026-10-13 随本轮收尾推送 origin）；23篇处理后bundle按批准计划成功新增23条、原有匹配0条。外部索引当前schema4/revision19、72条记录、`next_lib_number=73`，`PAPER_LIBRARY_INDEX.md`已重建一次；23个已确认来源目录已送Windows回收站，空的`临时论文`父目录已移除。详细N→LIB映射以正式索引题名为准（当时入库报告为临时文件，已随 2026-10-13 清理删除）。原有资产与既有人工字段未改；AI阅读导出未生成，需手动操作UI。
- 2026-10-12论文库维护能力阶段一已实施并**通过最终独立验收**：四个工具 + Rule + 三个Skill（新增`paper-maintenance`）+ 两份正式指南 + `_index/README.md` + 原任务03同步完成；`.codebuddy` 侧经仓库自带单向同步脚本补齐同一 Rule 与三个论文库 Skill。最终代码版本已在隔离环境实测通过 plan→写入→退库移动→续办幂等→恢复→导出（见[最终验收报告](../reports/2026-10-12-paper-library-maintenance-stage1-acceptance-final.md)）。阶段一第一次演练因父进程控制台编码错误留下了隔离根 `%TEMP%\paper-maintenance-drill-50652`（状态只到 plan 阶段），**经 2026-10-13 复核该目录已不存在**。
- 任务01的LIB写入前置条件现已满足，**任务03已执行完毕并通过独立验收**，**任务04已完成**（交付研究准备包，该包已交云端高级模型消费）。外部索引仍只允许经受支持入口写入；**磁盘索引现为 schema5，旧版本工具不得再写**。该轮未推送Git。

- 2026-10-12任务03（阶段二正式数据维护）已完成：一次 `maintenance-apply` 完成 67 篇元数据（独立 `rating`；26 篇 `year`、26 篇 `venue`）与指定 5 篇可恢复退库，索引 **schema5/revision20**，`next_lib_number=73`；现状 **活动 67 / 已退库 5 / 总登记 72**，退库资产在 `D:\0Project\origin\论文_retired\<LIB>\<原目录名>\`，管理视图与三份 AI 导出（67/15/21）已重建。计划 SHA-256 `ba0ec183…`，保存前指纹 `116f314a6bb137c7`；白名单差分 0 处越权变化。详见[任务03验收报告](../reports/2026-10-12-paper-library-task03-acceptance.md)。LIB000068 年份仍未知、LIB000070 DOI 仍为数据集 DOI、LIB000054 题名拼写与两套等级词汇口径差异列在报告的未解决项。

## 本轮授权与边界

- **OC-RGA 云端准备收尾（2026-10-14，本轮，已完成）：** DeLiVER **img/depth/semantic 各 5988** 文件（train 3983 / val 2005）已在云端 `dataset/DELIVER/` 就位；img 按天气分 5 包（0.84–1.62 GB）上传并逐包解压核对，`sun` 包中断后重传成功，各天气 train/val 样本目录集合三模态 md5 一致。云端已 fetch 并切到 `research/observation-relational-geometry`（HEAD `ce24e10`），旧分支与 `outputs/` 旧产物未动。**轻量只读检查通过**：配置与入口可导入（`dataset_path=/root/rivermind-data/dataset/DELIVER`、预训练软链有效、`checkpoint_dir=outputs/oc-rga-deliver/<run_id>`、SwanLab `dformer-research/online`、`CHECKPOINT_FORMAT=oc-rga-checkpoint-v1`）；`--print-schedule` 输出 3983 样本/epoch、249 updates/epoch、120 epoch/warmup 5、lr 8e-5 poly 0.9 几何 ×2；train/val 各 2 个真实样本为 rgb (1042,1042,3) uint8、depth (1042,1042) uint8 单通道、label (1042,1042,4) uint8，路径经 `deliver_relative_paths` 正确映射。**未运行完整 `--check`（无前向/反向/优化器步）**。已删除本地 `_ocrga_transfer/imgpack/` 5 个 tar（约 6.5 GiB）与云端 `dataset/{depth,semantic}.tar`；云端可用 **14 G**、本地 D 盘可用 86.7 GB。报告见[OC-RGA 云端准备报告](../reports/2026-10-14-oc-rga-cloud-preparation.md)。本轮未启动 GPU、未训练、未扩容。

- **云端旧产物第一轮清理、资产取回与 OC-RGA 数据准备（2026-10-14，本轮，已完成）：** 按已审核盘点报告执行 A 类保守清理并取回 B 类资产。实际删除：`dataset/SUNRGBD-odg-20261010.zip`（2.28G）、`cloud/MMFR_A2_v3_transfer_20260917/checkpoints/` 4 个权重（1.20G）及其在 `DFormer-archive-20261008/experiments/MMFR_A2_v3/checkpoints/selector-epoch-420.pth` 的硬链接副本、`cloud/MMFR_E1_Batch1A_local_transfer_20260922/`（616M）、`DFormer-archive-20261008/MMFR_AV1_cloud_transfer_20261001.zip`（290M）、`DFormer/outputs` 下 10 个 `smoke.pth`（3.02G），合计约 7.4G；磁盘可用 13G→**20G**。保留 A2 transfer 的归档元数据、`checkpoint-candidates.json` 与三份 `.sha256`。取回并字节校验：R2 original/odg `best-dev.pth`（324,567,822／324,567,758）、`MMFR_AV1_local_val_conditional_20261001.zip`（429,993,998）、`mmfr-e1-batch1b-roe-v2/.../update-2560.pth`（360,801,864）、`natural-missing-round1-formal/.../latest.pth`（321,065,050），合计 1.74G，本地位置见报告。经条目级 CRC 比对，R1/R2 四个 run 的 `stage-epoch-30.pth` 与 `last.pth` 仅 `data.pkl` 与序列化 id 不同，属**同一训练状态**，未传输（省约 1.3G）。报告见[第一轮清理报告](../reports/2026-10-14-cloud-cleanup-round1.md)；同时更正盘点报告两处（smoke.pth 实为 10 个；云端并无 `MMFR_E1_Batch1A_local_transfer_20260922.tar.gz`）。**OC-RGA 数据准备（同轮）：** 云端 `dataset/semantic.tar` 原为不完整包（264,929,280 B，tar 解析失败），已改用本地 `_ocrga_transfer/semantic.tar`（274,787,840 B）重传；`depth.tar`+`semantic.tar` 已解压到 `dataset/DELIVER/`（各 5988 文件，共 587M，仅含 train/val 划分）。本轮未启停实例、未启动 GPU、未改研究代码、未扩容、未提交或推送；云端 HEAD 仍为 `a11b835`，本地 OC-RGA 提交 `ce24e10` 尚未同步。待用户决定 `img`（约 8.8G）上传、`test` 划分补传及代码同步方式。

- **云端旧产物盘点与清理预审（2026-10-14，本轮，已完成）：** 用户授权使用当前无卡实例 `cpod-1vbh7faqcauq`（GPU=0，检查期间保持 `Running`）对 `/root/rivermind-data/{cloud,DFormer/outputs,DFormer-archive-20261008,dataset}` 做只读盘点并与本地归档核对。系统盘 50G／已用 38G／可用 13G；四处占用 16G／6.7G／3.8G／6.7G。已核对：MMFR `A2_v3` checkpoints 与 `E1 Batch1A` `update-2560` 权重本地已有；ODG R1/R2 本地缺 6 个权重（约 1.95G）；`mmfr-a2-v3-probe`、natural-missing 权重、E1 Batch1B ROE、AV1 formal 本地无副本；发现一处硬链接（`selector-epoch-420.pth`，nlink=2）与预训练软链接（`DFormer/checkpoints/pretrained/` → `pretrained/`）。建议保守清理约 7.3G、扩大清理累计约 22–25G，需先取回约 3.0–3.5G。报告见[云端旧产物盘点与清理预审](../reports/2026-10-14-cloud-legacy-artifacts-inventory.md)。本轮未删除/移动/覆盖/上传/下载任何资产，未扩容、未启动 GPU、未跑训练或测试、未改 OC-RGA 分支代码与配置、未创建或销毁云资源、未提交/推送；**未执行任何清理，等待上级确认范围**。

- **OC-RGA v0.3 数学与实施设计（2026-10-13，本轮，已完成）：** 用户授权一次性深化新方案、纠正实质数学问题、核对真实源码/训练/DeLiVER接口并就地形成可交执行模型的完整实施版，必要时定点回读原文。实际只读核对、数学/架构/训练设计及新方案/状态写入；未开发网络、建立验证脚本/测试体系、运行项目测试或GPU实验，未重新筛选论文/重做旧A诊断，未修改历史总方案，未启动付费/云资源、改论文索引、创建分支或提交/推送。后续开发分支和GPU预算需执行阶段明确授权，旧实验额度不延续；本轮交回上级审核后停止。

- **论文库资料完整性核验（2026-10-13，前轮，已完成）：** 用户授权核验最新论文库与 `PAPER_LIBRARY_FOR_AI.md` 导出，为云端高级模型深度推导/网络设计准备可靠、完整、可追溯资料，并在必要时用**已有工具**重新生成 AI 导出。实际执行：只读读取索引/导出/资产、计数核对、抽查公式 LaTeX 与图映射、计算 SHA-256、写出完整性报告。判定三份导出已是源 revision56 最新版故**未重新生成**；未改索引（revision 未变）、未改/移动/删除任何 bundle、未重提取全文、未重做 supplement、未新建导出格式或汇总包、未跑测试、未云端、未 Git 提交/推送、未做新方向调研或方案设计。缺口按要求只记录不修复。

- **方向 A 第一阶段科研审核（2026-10-13，本轮，已完成）：** 用户授权主模型定点核对 `74456eb`/`b9c094b` 的核心代码、报告及既有逐图/attention/区域产物，解释科研结果和历史评价差异，保存正式审核并更新状态。主模型建议 C（暂停），最终交云端高级模型裁决；下一阶段实验尚未授权。仅做只读检查和必要计数核算，不重跑528图、不开展文献调研、不创建测试体系、不开发反向/训练、不修改原始数据，不启动云端或付费资源，不提交/推送。最小统计勘误任务已写在报告，未自动执行；本轮报告保存后停止。

- **方向 A 技术任务单（2026-10-13，前轮文档任务，已交接）：** 用户确认科研方案v0.2已审核通过，授权结合真实源码在现有本地方案增加“执行模型实施任务单”，必要时修正接口描述，并记录交接恢复点。本轮仅文档，不编写模型代码、不运行实验；任务单交接后由执行模型独立实施冻结诊断，一般工程问题自主修复，主力模型保留定点代码审核和最终科研验收。第一阶段不含backward/梯度测试/10epoch程序或训练，GPU执行仍需启动实施时明确授权；不含云操作、付费、完整测试、提交/推送。前轮本地方案设计/审核修订已完成，旧ODG实验额度不延续到方向 A。

- **论文库阶段二（任务03 正式数据维护，2026-10-12，已执行）：** 用户批准按 `PAPER_LIBRARY_MAINTENANCE_PLAN.md` 与任务03执行正式数据维护，授权范围：用最新工具与已同步 Rule/Skill 确认无旧版写入者；按现成书目表与 N→LIB 映射批量补 `year` / `venue` / 独立 `rating`（不重新下载、不逐篇联网）；严格执行 5 篇可恢复退库（004/009/011/021/022）并保留 LIB、人工信息与恢复路径；等级值按修订后表格原样写入、不带装饰括号；走现有 maintenance 入口，真冲突即停；完成后核对活动篇数、退库状态与恢复目录、编号与人工笔记、管理视图与三份导出。**明确不含**：重新设计系统、重复阶段一演练、增加复杂测试、修改 DOI/题名等未授权字段、推送 Git。实际执行结果见[任务03验收报告](../reports/2026-10-12-paper-library-task03-acceptance.md)，无冲突、无未执行项；本轮未推送 Git，交由执行模型2独立验收。
- **论文库维护阶段一（2026-10-12，已完成）＋独立验收修正（2026-10-12 晚，已执行）：** 用户批准最新版 `临时/PAPER_LIBRARY_MAINTENANCE_PLAN.md` 并授权执行**阶段一（工程功能实现与最小验收）**：按已审核方案实施独立 `rating` 字段、统一维护入口、可恢复退库、UI与AI导出，并同步 Rule、三个 Skill 与正式指南；完成一次隔离环境真实演练。**明确不含**：修改正式论文索引、移动正式论文资产、执行任务03数据维护、增加复杂测试或额外工作流。LIB000068 的年份语义按要求保持“不得把接收年当成正式发表年”。独立验收后按用户指示做了四项最小修正（均不重跑92项测试）：同步 `.codebuddy` 侧 Rule 与三个论文库 Skill；去掉任务03 `rating` 目标值的装饰性 `【】`；把阶段一报告与本状态文件的验收口径改为“代码与定点检查通过、端到端隔离演练待最终版本复核”并说明临时目录残留；旧版替换稿只保留废止说明、不额外整理。全部改动限于工具、规则、Skill 与文档；未提交或推送 Git。
- **DeLiVER最终验收（2026-10-10，已完成）：** 按批准计划接入RGB+原始单通道Depth、25类/ignore255及官方train3983/val2005/test1897，主力复核已有CPU smoke（29/29几何权重更新）与数据产物后验收通过；`72ec726`已合入最新主线`da0cace`，合并提交`5d61a8c`，保留BN评价配置与eval模式验证。冲突处复用主线完整优化器分组，合并后CPU入口核验714/714唯一入组、含29个Geo.weight，未重复模型前后向。详见[最终验收报告与命令](../reports/2026-10-10-deliver-integration-acceptance.md)。未运行DeLiVER GPU训练/全量val/test、未推送，正式GPU能力和实验预算待另行授权；不改变SUN研究裁决。

- **R2 严重缺失筛选授权（2026-10-10，已执行）：** 上级审核通过 R2 结果并批准最后一次低成本鲁棒性筛选：用现有 R2 两个 epoch30 checkpoint 在本地 RTX 5060 评价 50% 与 75% 人工深度孔洞（授权四组推理），完全沿用现有 eval 模式、数据划分、seed、模型配置与孔洞规则；比较绝对 mIoU 与相对各自 clean 的退化；简单检查被删像素的深度组成但不把天然零值定义为物理缺失；按“优势是否约 1pp 以上且可复现”给出方向建议。实际执行 4 组 + 2 组 75% 第二 seed 复现检查（均为本地推理，未用云 GPU），另做一次删除像素深度组成统计。**未重训、未改模型、未启动云端、未做复杂性能或代码审计。**

- **R2 GPU 实验授权（2026-10-10，已执行完毕）：** 用户批准按 R2 计划推进两组 30 epoch（云端实例由用户手动启动），授权推送代码、同步云端 Git、设置并回读平台关机保险、最小 GPU 检查、并行 `4×4` 训练、每 15–20 分钟监控、训练后主动停卡并用短暂无卡窗口取回产物后再次停机、本地完成 clean 与 25% 孔洞评价与报告。预算安全上限 ¥15 或 6 GPU 小时，超出前先报告。**实际使用 ≈¥6.22 / 3.37 GPU 小时**（内有卡 3.303 h）。未续训 100 epoch、未做 mean 消融、未跑正式 test、未新建或删除云资源。

- **训练代码修复与第二轮准备（2026-10-10，已执行）：** 上级批准只做本地训练代码修复与准备、暂不启动 GPU：周期验证改为 `model.eval()` 并恢复训练模式；修复优化器参数分组遗漏 29 个 `Geo.weight`；保留现有 ODG 设计、超参数、数据划分与 300 epoch 日程；只做最小 CPU 检查。本轮改动限于 `research/{train_odg,odg_schedule}.py` 与 `research/run_odg_gpu.sh`（运行目录后缀默认 `r2`，改动 1 处 experiment 名），未启动云端 GPU、未重训、未新增消融、未改模型结构与作者代码。

- **评价修复与复测授权（2026-10-10，已执行）：** 上级批准修复独立评价入口未恢复训练时 BatchNorm 数值配置的问题，并用现有两个 epoch30 checkpoint 在本地 RTX 5060 重做 Original/ODG 的 clean 与 25% 孔洞四组；优先确认修复后 clean 能否复现训练内 39.39/36.97，仍不一致时只针对实际差异排查。本轮实现限于研究层 `evaluate_odg.py` / `odg_schedule.py`，另做一次 train 模式定点诊断定位差异（同权重、同 528 图，仅改变模型模式）；未修优化器、未重训、未新增消融、未启动云端 GPU。

- **上一轮孔洞评价授权（2026-10-10，已执行）：** 现有Original/ODG epoch30、同528图dev、同seed/config的25%孔洞评价在本地5060完成；因与历史clean不一致，仅补两次同入口clean基准，未添加故障/seed/消融。禁止新训练、修改模型/优化器；付费云GPU未授权且未启动。该轮的“评价BN配置恢复及重复配对仍需批准”已由本轮修复复测替代；没有提交或推送。
- **已结束诊断任务（2026-10-10）：** 只读科研诊断及CPU定点取证、写报告；资料本地齐备，未使用无卡云取证，未修改代码或启动GPU。优化器修正重验仍需另行授权。

- 2026-10-10指令授权本任务代码/配置/简短文档、限定CPU检查、现有实例无卡启动和关停、数据传输/必要下载、一次最小在线日志验证、Git提交及推送用户远端。此授权替代已结束论文库任务的工程限制，但没有扩展论文库写入权。
- **GPU阶段授权范围（2026-10-10用户批准）：** 现有实例有卡运行、最小GPU检查、original与odg各30 epoch正式训练（用户事后明确批准双模型并行4×4方案）、必要的最小代码修复与提交；预算上限¥40且20 GPU小时。**该额度只用掉4.31小时/约¥8.1。** 续至100 epoch、正式test、300 epoch、mean消融与云端GPU孔洞评价仍需另行授权；本轮本地孔洞评价已按最新授权完成，未经批准不启动100 epoch训练。
- 有卡启动、付费规格切换与云资源生命周期由用户手动执行或明确授权；不新建实例、不扩容、不删除磁盘/实例、不覆盖旧实验。GPU检查通过前不开始长训练；训练前必须设置并回读平台关机保险。
- 不跑完整旧测试、全量模型推理、长CPU训练或权重全量重哈希。异常与未运行检查必须如实记录；正式GPU能力不能由CPU验收替代。
- 验证范围曾有偏差：首次CPU检查遇到作者硬编码CUDA基向量分配后在设备一致性检查处终止，已修复并屏蔽CUDA重检；数据整理曾全清单头部尺寸扫描（15620条含重复），超出约5组抽样，已停止并移除全量入口。细节见报告，不隐瞒或继续扩展。

## 下一步、阻塞与恢复点

- **当前优先恢复点：** 将 OC-RGA 实现（提交 `ce24e10`）交本地主力模型审核代码结构与训练方案，重点核对 `research/oc_rga_model.py` 的优化器分组与几何分支 lr×2、`research/oc_rga_data.py` 的支撑标记与退化协议、`research/oc_rga_losses.py` 的 hold-out 无泄漏约束、`research/train_oc_rga.py` 的合同/恢复检查，以及 120 epoch 日程、50 观察点、640 评价的成本合同；通过后交上级复核，之后才申请 4090 的 30 epoch 性能试跑。**当前没有 GPU 侧显存/吞吐/利用率实测，也没有完整 `--check` 的前向/反向证据，不把静态检查写成验收通过。** 云端旧产物 D 类（`cloud/mmfr-a2-v3-probe/` 等）按用户指示暂停进一步清理；`cloud/`、`DFormer/outputs/` 与 `dataset/` 现状见[云端准备报告](../reports/2026-10-14-oc-rga-cloud-preparation.md)。

- **方向 A 历史恢复点保留：** [正式科研审核](../reports/2026-10-13-influence-research-audit.md)和原始产物不变，暂停旧路线，不自动补固定输入探针、重跑32/496/528图、进入三组10epoch或实现旧方向B。OFF样本范围和attention分母的无推理勘误可另轮授权，不作为OC-RGA开发前置条件；没有新增云端最终裁决回执。

- **论文库阶段二（任务03）已执行完毕并通过独立验收。** 验收要点：索引 `schema5/revision20`、活动 67 / 已退库 5 / 总登记 72、`next_lib_number=73`；67 篇 `rating` 与旧库 4 个缺口、LIB000069 出版物纠错；退库集合恰为 004/009/011/021/022 且资产在 `论文_retired\<LIB>\<原目录名>\`、原位置不存在、正文 SHA-256 与索引一致；人工编号（含 LIB000021 `RE188`、LIB000022 `RE447`）与笔记保留；管理视图与三份导出（67/15/21，源 revision 20）不含退库记录；白名单差分 0 处越权变化。证据：[任务03验收报告](../reports/2026-10-12-paper-library-task03-acceptance.md)（维护请求/计划 JSON 与最终维护报告均为临时文件，已随 2026-10-13 清理删除；计划 SHA-256 `ba0ec183…` 保留在验收报告中）；保存前基线为 `backups\PAPER_LIBRARY_INDEX.json.bak1`（revision 19，指纹 `116f314a6bb137c7`）。**未执行**：正式 5 篇的实际恢复（需另行明确授权）、LIB000070 DOI、题名规范化、等级词汇统一。
- 阶段一隔离演练残留根 `%TEMP%\paper-maintenance-drill-50652` 经 2026-10-13 复核已不存在（`Test-Path` 返回 `False`，与[书目收尾报告](../reports/2026-10-12-paper-library-biblio-fix.md)一致）；本文件中“仍未清理”的旧表述作废。
- 2026-10-12任务04已交付研究准备包：整合新23篇定向阅读、重点旧库与任务02项目证据，形成9项问题–机制矩阵、3张机会卡与7种叙事/图表范式。该交付在任务03之前完成（当时索引视图为revision19/72篇），任务03补齐的书目与等级**不改变**该包结论；**该包已交云端高级模型消费，临时原件随 2026-10-13 清理删除**，不构成实验授权。

- 文档归档（2026-10-10）：`临时/`中4份Markdown已留档至`doc/`：交接报告进`reports/`，数据下载指南进`guides/`，ODG无卡指令与GPU计划进`plans/`；相关链接已更新，历史材料已标注时效与授权边界。临时原件已于 2026-10-13 临时清理中删除，本次不改变实验事实或授权。

- 2026-10-10首轮GPU实验计划已执行完毕（[归档计划](../plans/2026-10-10-odg-gpu-experiment.md)）：最小GPU检查、限时性能诊断、两组30 epoch均已结束，实例已停机。结果与产物位置见[30epoch报告](../reports/2026-10-10-odg-sunrgbd-30epoch.md)。

1. 云端工程与数据保留不覆盖；旧云端资产、原zip与本地来源均不动，不继续下载或转换。
2. 本地产物：`outputs/sun-dev-original-seed12345/20261010-013324/`与`outputs/sun-dev-odg-seed12345/20261010-020527/`（CSV、逐类IoU、预测、配置、`last.pth`、`best-dev.pth`、swanlab），诊断与训练日志在`outputs/odg-gpu-20261010/`；4个checkpoint已用本地torch2.7.0 CPU实际加载核对字段。云端保留全部文件含未取回的`stage-epoch-30.pth`。
3. 续训100 epoch、正式test、300 epoch、mean消融仍未授权、未运行；30 epoch差距持续扩大（-1.09→-1.99→-2.42）**为 train 模式验证数值，上级已裁决不再作为淘汰依据**。
4. [评价对齐复测](../reports/2026-10-10-odg-bn-align-reevaluation.md)已完成：解码器BN数值已与训练对齐，但对结果影响≤0.01点；修复后clean为41.32/42.77，孔洞为39.78/42.74，ODG孔洞仍+2.96、退化改善1.51点。产物在`outputs/odg-holes25-bnfix-20261010/{original-clean,original-holes25,odg-clean,odg-holes25}/`，诊断证据在其`diagnostic-trainmode.json`。**已确认原因并修复：`research/train_odg.py`周期验证未切eval模式**，故训练内39.39/36.97为train模式数值，同权重eval模式clean为41.32/42.77（ODG领先1.45）。mask处理与分布贡献仍未分离。
5. **R2 第二轮实验已执行完毕**（[R2报告](../reports/2026-10-10-odg-sunrgbd-r2-30epoch.md)）：两组各30 epoch（17:42:31–20:47，UTC+8）并行完成，均Succeeded；本地对齐评价为 original clean45.34/孔洞44.70（降0.64）、odg clean45.03/孔洞44.45（降0.58），**ODG−original = clean −0.31、孔洞 −0.25，R1 的 +1.45/+2.96 未复现**。产物：`outputs/sun-dev-{original,odg}-seed12345-r2/<run-id>/`（各19文件含`last.pth`）与`outputs/odg-r2-eval/{4组}`；云端保留全部文件含`best-dev.pth`/`stage-epoch-30.pth`/`swanlab`。**实例已 `Stopped`（读回确认），计划关机已取消（`scheduled=false`）；费用≈¥6.22/3.37 GPU小时。**
6. **R2 严重缺失筛选已完成**（[50%/75%筛选报告](../reports/2026-10-10-odg-r2-holes5075-screening.md)）：产物在`outputs/odg-r2-eval/{original-holes50,original-holes75,odg-holes50,odg-holes75,original-holes75-s2,odg-holes75-s2}/`，与既有 clean/25% 四组构成完整对照。**方向建议：停止当前方法，不续训 100/300 epoch**（依据：优势仅出现在 75% 孔洞且约 0.8–1.0pp、25%/50% 无优势、clean 落后 0.31pp）。若上级要求保留最后一次机制判读，唯一高价值项是 75% 条件下的 Mask 中性化消融（判断优势是否只来自显式删除 mask 处理）；未获授权前不执行。
7. 本轮未授权、未运行的项：100/300 epoch 续训、`mean`/Mask 消融、官方 test、NYUv2、DeLiVER 训练、五尺度+翻转、第二 seed 训练。
8. 性能诊断结论（若后续续训仍需提速）：数据加载与CPU未饱和；每次迭代约0.116s加每次optimizer尝试约0.5s固定开销是主要限制；可选优化是经审批后削减每次尝试的逐张量梯度有限性扫描（714张量逐个同步，属监测开销，不改变优化与公平性）；`16×1`仅在整卡独占时可行。
