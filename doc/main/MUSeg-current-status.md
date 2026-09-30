# MUSeg 当前状态与唯一实时入口

> **事实截至：** 2026-09-30 A-v1 正式训练前本地收口。当前为 **`A-v1 Gate-B PASS; formal training contract frozen; local Git handoff pending; waiting for cloud 4090 full-resolution Proposal/Gate 3-update preflight`**。本轮仅本地合同、身份核验与Git准备；没有新模型执行、训练、Quick-Val/Main-Val、official test、云端操作或push。执行选择见 [开放决策](MUSeg-open-decisions.md)。

## 当前阶段与实际意义

**A-v1 固定实现和原 Gate-B 工程资格保持不变；第一轮正式合同、四条件筛选线和4090两阶段各3次成功更新的短预检边界已冻结。** MMFR 是多形式模态失效与可靠性研究；Gate-B 是训练前工程资格检查，Proposal 是补偿残差动作，Gate 是每图动作强度选择器。**大白话：** 下一次运行该做多少、用什么参数、什么时候停已经写清，但当前没有训练结果，也不能说4090一定容纳batch10。

权威合同：[A-v1 protocol（含Cloud handoff）](../../MMFR/01_research/mmfr_a_v1_action_utility_protocol.md)；工程历史与本轮补充：[Gate-B报告§7](../../MMFR/02_evidence/report_mmfr_a_v1_implementation_gateb_20260930.md)；原始证据：[Gate-B JSON](../../MMFR/02_evidence/mmfr_a_v1_gateb.json)；身份：[现行复现信息](../../MMFR/02_evidence/reproducibility_current.json)。[设计报告§9](../reports/2026-09-30-mmfr-next-generation-research-design.md)记录后续冻结合同，原设计历史不替代本协议。

## 固定身份与直接核验

- Identity **`MMFR-A-v1-action-utility-v1`**；run name `MMFR-A-v1-action-utility-v1-formal-v1`；config import `local_configs.MUSeg.DFormerv2_S_MMFR_AV1`，不复用旧E1运行身份。
- C0 fixed-final SHA-256 本轮直接复算为 **`ca618b23d18eabb201a0d11d18da383ac99576d0feae5864e3233bda527d9a1a`**；本地source路径在protocol。checkpoint只读取哈希、不改字节、不纳入Git。
- frozen C0、stage index2、256channels、Proposal256→32→256+DWConv、sample-level连续gate、Proposal→Gate、off/full/learned、paired off/full utility及observed-only gate边界全部不变。
- 收口开始时五个关键文件hash与原Gate-B全部一致；本轮只改正式config，`models/mmfr_av1.py`、`models/builder.py`、`utils/mmfr_av1_training.py`、`tools/mmfr/av1_gateb.py`维持原字节。正式config SHA **`a868bc11e9aa0b950407ae4e0c5fd47fb6b298b81ba7b4a6e54be66f8da56319`**，与原Gate-B config版本明确区分。
- 本地正式config为既有CRLF哈希；Git LF版本SHA为 **`329ac9fd15e2f09102350011bdf2b8b1afcdbad951bf6c96c844ca4b9b641f03`**。5个Git LF代码hash记在既有复现JSON，运行代码内容保持原实现；A-v1证据/hash绑定审核sources/生成包与历史快照定点设 `-text`，保证source/packet hash及原证据字节入Git不变。
- 原Gate-B证据SHA仍为 **`bf87cc2fc46dd413131f7a6778d8df82388cf7fffa08852126d1f62178be01bf`**，必要工程checks全部PASS；没有重跑。原始范围是synthetic B1/64×64、8forward/2one-step update，不能推断全尺寸训练容量或性能。
- 正式config在 `D:\2Env\anaconda\envs\df2\python.exe -B` 可import，冻结参数/身份/禁用Val与official-test字段断言PASS，无model/dataset/GPU调用；定点config静态诊断无问题。旧E1 protocol/evaluator与v3 corruption代码在Git中无内容差异。official test保持 **`sealed_unread`**。

## 第一轮正式合同（冻结，不等于授权执行）

- Proposal **1920**、Gate **640**、总计 **2560 successful optimizer updates**；GradScaler skip不计更新，发生后停止回报。只有工程/数值/身份/基础设施错误或无法解释的明显loss异常可以停止；不新增utility early-stop，不因gate分布不好看停，不通过val-dev延长预算。
- margin **0.01 per-image mean CE difference**；lambda_clean **0.1**两阶段一致；AdamW新branch peak LR **3e-5**、WD **0.01**、现有bias/norm no-decay。是用户预冻结研究/工程起点，不宣称最佳或来自val-dev调参。
- 每阶段独立optimizer/scaler、128次成功更新linear warmup、poly0.9，阶段末LR归零；train/data seed772961337，initialization2026093000；phase corruption seeds **2026093001/2026093002**；p_clean **0.25**；六类既有Depth故障。现有A-v1接口使用phase-local v3 curriculum0..1，数据坐标15×128/5×128，不静默复用E1的0.84..0.88 remap。
- 目标batch **10**、真实crop **480×640**、workers **8**、accumulation **1**、singleGPU/DDP off；AMP fp16 **on**、TF32 matmul/cuDNN **on**（显式沿用E1已核验实际训练行为）。4090容量/速度未验证。
- Proposal结束保存 **`proposal-update-1920.pth`** transition/recovery；Gate必须从同一Proposal权重进入。唯一fixed-final为 **`update-2560.pth`**；每640次成功更新recovery，要求完整optimizer/scheduler/scaler/RNG/data cursor/phase/counters，不通过Val选checkpoint。
- 未来仅一次clean/entire_missing@1.0/spatial_dropout@0.75/misalignment@0.75四条件Quick-Val，同一fixed-final输出off/full/learned。继续须同时满足learned hard mean相对matched off **>=+0.50pp**、clean **>=−0.20pp**、hard mean **严格>full**；否则记录selector-not-supported/stop/inconclusive。pp是百分点；这些是资源筛选线，不是统计显著性或评价授权。

## 本地Git恢复点与审核包

- Branch：`perf/mmfr-a2-v3-pipeline-opt1`；准备前HEAD：`a7c5cc17c5b495f2210efa3604b05794b5fa16c4`；本轮授权**一个本地commit**，精确暂存A-v1文件，当前仍在提交准备，完成后回填完整SHA；不push、不改远端或tag。
- 历史Canvas、论文库整理、旧实验与无关Base config等既存dirty保留，不删除、不清理、不混入commit。混合报告索引仅暂存A-v1条目，保留其他工作区条目。
- SHA只能在commit创建后得知，因此完整SHA及生成审核包的当前Git identity在提交后回填为**元数据回执差异**；不amend、不补第二个commit，不把这些差异冒称已在自身commit内。执行代码/config/冻结合同必须在commit中。
- 既有L1 profile保持 `mmfr-a-v1-action-utility-gateb`，更新到本地收口审核；生成包只能由 `MMFR/98_tools/rebuild_review_packet.ps1` 重建，不能手工编辑。当前已由既有generator重建6文件审核包；canonical、packet及独立复制包坏链均为0，generation identity以既有 `packet_manifest.json`为准。

## 授权边界、阻塞与准确恢复点

- **本轮完成后停止。** 尚未授权正式训练、Quick-Val/Main-Val、official test、云端登录/同步/启动或push；合同冻结不等于训练权限。全文/近邻限制与原研究结论不在本轮改变，也未继续文献搜索。
- 下一步仅允许在上级/用户另授云端权限后同步本轮commit到RTX4090，按正式分辨率/batch/AMP/TF32先Proposal最多**3 successful updates**，正常后Gate最多**3**；每阶段到3立即停、错误立即停，回报peak allocated/reserved、时间、loss、finite/OOM/GradScaler skip；Gate另报utility/positive/negative/ambiguous。**不能自动接1920+640**，预检不计正式预算，不作为正式起点。
- 当前只有两阶段函数接口，**full-resolution preflight/formal runner与A-v1评价入口尚未实现**。恢复时先审核最薄限3-update调用器是否落实合同，不能直接套旧E1 `utils/train.py` 命令。本轮没有增加执行入口。
- 本轮实际验证只做身份hash/config import/字段断言、定点静态与Git差异/文档检查；完整测试、GPU Gate-B重跑、全尺寸预检、训练、Val、benchmark均未运行，因为当前改动只有config/合同且用户禁止模型/云端执行。

## 仍需追溯的独立事项

- R-OE-lite v2已完成正式训练、四条件Quick-Val为inconclusive，路线处置仍独立待决定：[原报告](../../MMFR/02_evidence/report_e1_batch1b_roe_v2_formal_training_quickval_20260924.md)。旧入口修复与Quick-Val文件已在 `1b061cbabaf76575b0812e1513d953b02cdb6254` 中本地提交，已不是“未提交文件处置”阻塞；不等于其评价资格自动通过。
- F-lite十条件Main-Val未复现Quick-Val优势，去留仍待上级裁决：[原Main-Val报告](../reports/2026-09-22-mmfr-e1-batch1a-c0-flite-mainval.md)。Oracle-A geometry suppression维持历史NO-GO。
- 论文实体/人工编号遗留项不影响本合同，见 [论文库整理报告](../reports/2026-09-30-paper-library-normalization-review.md)。旧Gate-B/设计轮完整事实通过原报告、protocol与Git history追溯，不把阶段流水账追加回实时入口。
