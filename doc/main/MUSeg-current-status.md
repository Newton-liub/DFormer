# MUSeg 当前状态与唯一实时入口

> **事实截至：** 2026-10-01 09:30 UTC。A-v1第一轮已按批准的`fp32-full1280-resume`完成：**Proposal1920/Gate640，skip0，唯一四条件Quick-Val完成，筛选结论`stop`**。learned三hard平均50.768491，相对matched off **−0.001873pp**，未达到+0.50pp继续门槛。所有实验进程已退出，GPU空闲，云实例未关闭。当前为 **`formal + single Quick-Val completed → stop/no further experiments`**。本次结果是单seed筛选，不是统计显著性或一般模型结论。合同见 [protocol §8](../../MMFR/01_research/mmfr_a_v1_action_utility_protocol.md)；其他独立未决事项见 [开放决策](MUSeg-open-decisions.md)。

## 当前结论与授权终点

A-v1包含Proposal（stage2候选补偿残差）与Gate（每图连续补偿强度选择器）。`off`禁用补偿，`full`始终全量补偿，`learned`用Gate选择连续强度；matched off是同checkpoint、同输入/随机状态的禁用补偿对照。HAM是原解码头，NMF是其中的非负矩阵分解；FP32为32位浮点计算，AMP为自动混合精度。mIoU是平均交并比，pp是百分点。

**大白话：** 局部精度修订使批准的剩余训练正常完成，但本轮筛选没有显示补偿的困难条件净收益；learned略高于full，二者都略低于off。继续门槛未通过，因此按已有授权停止，不进入Main-Val、official test、新seed、sweep、额外预算或模型重设计。微小差值不作显著性判断。

- 用户批准仅A-v1训练NMF局部FP32，并从原完整1280恢复全部状态；保留原前1280更新，只执行Proposal剩640/Gate640。其他AMP/scaler1024/TF32、结构/loss、batch/种子/逻辑预算不变。这是**混合历史精度的正式续训**，不是全轮FP32重训。
- 原预检3+3、单次受控重放及三组失败现场对照均已结束，本次未重复。额外CPU核验没有model forward/backward/optimizer更新。
- 训练完成后评价首次初始化在任何样本forward之前出现NumPy配置数组比较工程错误；保留原空目录与停止证据，最小精确比较修复/CPU核验/提交后执行了**一轮实际评分**，未重训或重复样本评价。

## 四条件Quick-Val结果与筛选

同fixed-final、318 val-dev/条件、original-full、scale1/no flip、FP32/TF32 off、eval seed2026091401、reset-per-unit、off/full/learned配对HAM随机状态。显示mIoU保留两位，差值和判断由未舍入值计算。

| 条件 | off mIoU% | full mIoU% | learned mIoU% | learned−off pp | Gate均值 / 中位数 |
|---|---:|---:|---:|---:|---:|
| clean | 53.46 | 53.44 | 53.45 | −0.010031 | 0.542778 / 0.543237 |
| entire_missing@1.0 | 48.76 | 48.77 | 48.76 | +0.002500 | 0.543209 / 0.544448 |
| spatial_dropout@0.75 | 51.40 | 51.40 | 51.40 | −0.000125 | 0.540945 / 0.541768 |
| misalignment@0.75 | 52.15 | 52.14 | 52.14 | −0.007993 | 0.542748 / 0.542533 |
| 三hard未加权平均 | 50.770364 | 50.766106 | 50.768491 | −0.001873 | — |

- learned hard−off：−0.0018730026999946858pp，要求≥+0.50pp，**FAIL**。
- learned clean−off：−0.010031370364394832pp，要求≥−0.20pp，PASS。
- learned hard−full：+0.0023852985788366254pp，要求严格>0，PASS。
- full hard−off：−0.004258301278831311pp。full与learned三hard平均均低于off，按冻结描述性规则归类 **`stop`**；不新增数值阈值。
- 权威结果：`cloud/mmfr-av1-formal-fp32-resume1280-v1/quickval-config-equality-fix-v1/summary.json`及四个条件的`metrics.json`。评价实际09:11:42.064156→09:21:50.023665 UTC，607.959805s；Python PID60976于09:21:50.830 UTC exit0。
- 直接复核PASS：四条件各318唯一样本；812个冻结C0张量精确一致；首个真实clean样本strict off与独立C0 logits逐值相等、max_abs_error0、Proposal/Gate调用0。已用保存的3816个逐样本行为confusion重新累计核对聚合、mIoU、hard均值、差值、Gate均值/中位数和筛选规则；无额外GPU forward。

## 正式训练、最终身份与恢复证据

- 训练runtime **`812507385a1b4b805966799bffb9a13f4704e492`**；评价runtime **`c798ed8f27483157ba6b164dafec4995bc47fce1`**，均在执行前提交/push并直接核验远端。评价补丁仅将NumPy数组配置比较改为`np.array_equal`，实际不一致仍拒绝；CPU接受原相同config并拒绝扰动norm_mean。未改评价口径/forward/指标算术。
- 新数值合同SHA **`5f437c5b470077476f28b8e1ecceddf119cc402045e723f5a1aadbc519c6cc2e`**；配置唯一新增数值字段`nmf_training_precision="float32"`，去除后精确匹配原合同SHA`96e93e02bfb2ad8214ab26f826ca66728a80231e16133f81a04709036578f7cd`。仅A-v1显式训练标记且CUDA autocast开启时局部NMF输入/计算FP32；C0/NMF仍eval，算法/迭代/随机初始化不变，其他模型和FP32评价默认路径不变。
- 训练输出`cloud/mmfr-av1-formal-fp32-resume1280-v1`；`identity.json`、`precision-amendment-compatibility.json`、`training-result.json`、两阶段`*-result.json`及步骤日志保留。开始`2026-10-01T08:37:03.420868+00:00`，本次实际640+640=1280成功更新；逻辑Proposal1920/Gate640/global2560、skip0。已逐条核验本次1280记录连续、finite、optimizer_applied、scale1024、skip0及LR坐标。
- Proposal剩640耗时697.9259616411291s，Gate640耗时837.7544067027047s；入口训练总计1539.0116258771159s（约25.65分钟）。**均为恢复段耗时，不含原保留1280，不是完整冷启动耗时。** Proposal/Gate peak allocated2943.554688/3026.777344MiB，reserved3578/3754MiB；仅本次实际运行观测，不作benchmark。
- 唯一性能候选：`formal/checkpoint/update-2560.pth`，SHA **`87ec54d10192d3aedc1d6edb864b7b60b94ce66220a748a12f1ca50ca605d336`**。Transition `proposal-update-1920.pth`，SHA`979b99d1928b42d87ce45d38701791279b56ab78e9e2ef015c9dd5f06da4d26a`。独立CPU核验SHA、完整resume metadata/修订lineage（来源链）、最终model/optimizer全finite、812个C0张量与source完全一致、最终Proposal与transition相同，均PASS。
- 唯一批准父恢复点：原`cloud/mmfr-av1-formal-v1/formal/checkpoint/update-1280.pth`，SHA **`05714d165925c4549aa1111564bbca7d8b8d9896d337c5b68f0b7bb98590564d`**，父runtime **`91c7f96d3768fde0f478e03a4ccda8c6f19368dd`**。恢复model+AdamW+scaler+四类RNG+数据cursor，下一epoch11/iteration0、LR1.185953940035674e-5、scale1024/growth_tracker1280；父文件/protocol未改。普通跨提交恢复仍拒绝，显式兼容性只接受固定父身份。不存在完整1870恢复点，未使用诊断1870权重。
- 训练参数仍为AdamW LR3e-5/WD0.01、margin0.01/lambda_clean0.1，各阶段128-update warmup/poly0.9；batch10/480×640/workers8/accumulation1、AMPfp16/TF32 on、scaler1024/growth2/backoff0.5/interval2000。C0/BN冻结、训练Val关闭、无utility早停；Gate使用同一Proposal及全新Gate optimizer/scaler。

## 输入、历史失败与证据保留

- C0仓库外 `/root/rivermind-data/cloud/MMFR_E1_Batch1A_local_transfer_20260922/C0/checkpoint/update-2560.pth`，SHA`ca618b23d18eabb201a0d11d18da383ac99576d0feae5864e3233bda527d9a1a`，812 source keys。Dataset `/root/rivermind-data/dataset/MUSeg_DFormer`；train-dev1277 SHA`a6b15b63f6d5193e3928ea24ada25be403a48e68d1c1f9372cdbbc3fe5cd8470`，val-dev318 SHA`1d0719d8f64f016d48995c25ab66d4004d76b7155d9efeef7cbb7454c0dd0e83`；计数/唯一性/互斥核验。official test **`sealed_unread`**，未打开该split或图片。上传ZIP仅查看目录，未解压覆盖或使用其中未核验权重。
- 原runtime91c7f96在`cloud/mmfr-av1-formal-v1`完成3+3预检后，从干净C0正式运行；Proposal attempt1871/completed1870/skip0在backward/unscale后、step前停止，Gate/评价未开始。原预检、停止日志与完整1280原样保留；skip0不能证明失败尝试健康。
- 单次受控重放`cloud/mmfr-av1-gradient-diagnosis-v1`，runtime9c257a9：原1281–1870共590条loss/CE/KL/LR/计数/scaler等精确匹配；同1871有限loss在NMF `torch.bmm(x,coef)`的`BmmBackward0`出现NaN。只认证该区间已记录轨迹，非任意逐张量/跨硬件恢复等价。
- 现场`formal/diagnostic-failure-state.pth` SHA`6865e8865bd486fac5b9bf06317af768d5a4634e5e4af4ea839a41c174bb0fb5`是diagnostic-only/nonresumable/non-performance-candidate，model/输入finite、四类pre-forward RNG齐全，但无正式完整optimizer状态；从未用作正式续训。
- 三组同现场对照`cloud/mmfr-av1-failure-controls-v1`，runtime9092bfb：6 forward/3 backward/0 optimizer、全部model张量不变；原scale1024全16992梯度标量NaN，scale1或NMF局部FP32/scale1024均有限。只证明单batch精度/scale依赖，具体中间超界机制未逐算子证明。本次修订轨迹的1871已成功、loss0.07739216089248657；不声称等同原FP16轨迹。
- 本次首次评价初始化工程错误保留在根`stopped.json`、训练console及原空`quickval/`；在`verify_model_identity`配置数组比较处、评价样本循环之前退出，样本forward0/评分0。修复后的完整结果另存上述可追溯目录，旧证据未覆盖。

## Git交付、资源与停止边界

分支`perf/mmfr-a2-v3-pipeline-opt1`，origin `https://github.com/Newton-liub/DFormer.git`。训练8125073/评价c798ed8已普通push并远端核验，未强推或改写历史；两份实时文档与 [既有复现证据](../../MMFR/02_evidence/reproducibility_current.json) 的完整结果提交 **`65d89c1dc03f762fc91316ebd4aef2b3b655acc2`** 已普通push，`ls-remote`直接核验同一SHA、该结果提交后工作区clean。本段仅记录已核验结果提交，不追逐本回执文档自身SHA。大checkpoint/dataset/日志/逐样本评价cache不入Git/MMFR。

09:30:44 UTC直接确认无A-v1训练/评价进程，GPU **1MiB/0%**。仅任务停止，**云实例仍开启、未关闭或销毁**。实验授权已执行完毕；下一恢复点是只读上述最终checkpoint/summary，不再启动训练或评价。未来若有新的研究/资源动作须另获授权。

实际检查仅覆盖本次风险：定点语法/CPU合同和完整恢复/训练eval标记/数组精确比较、checkpoint身份、保存证据算术复核及已批准正式段与一次Quick-Val。未运行完整测试、重复Gate-B、额外GPU诊断、Main-Val、test、新seed或benchmark。

`MMFR/99_review_packet_current/`仍为以前生成的旧快照；环境无`pwsh`，未安装依赖或手工改生成产物，现有`MMFR/98_tools/rebuild_review_packet.ps1`的重建待可用环境。最新事实以实时入口、protocol §8和复现JSON为准；所有历史报告/原始结果不追改。
