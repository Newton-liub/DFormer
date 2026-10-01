# MUSeg 当前状态与唯一实时入口

> **事实截至：** 2026-10-01 08:32 UTC。用户已批准 **`fp32-full1280-resume`**：仅A-v1训练的NMF局部FP32，从原完整1280恢复全部训练状态，完成Proposal剩640/Gate640及唯一四条件Quick-Val。当前为 **`precision amendment implemented → minimal CPU checks passed; awaiting committed compatibility check/start`**。原失败、受控重放和三组现场对照证据保留；Gate/Quick-Val仍未开始，无fixed-final或性能结论。最新合同见 [protocol §8](../../MMFR/01_research/mmfr_a_v1_action_utility_protocol.md)，其他真正未决事项见 [开放决策](MUSeg-open-decisions.md)。

## 当前动作、授权与实际意义

A-v1含Proposal（stage2候选补偿残差）与Gate（每图连续补偿强度选择器）。HAM是原解码头组件，内部NMF是非负矩阵分解。FP32表示32位浮点计算；其余模型仍采用AMP（自动混合精度）。**大白话：** 用户选择保留原前1280步的完整训练状态，仅提高后续训练中NMF的局部精度；这是有明确数值合同修订的续训，不是全轮FP32重训，也未证明整轮稳定。

- 批准范围：先提交修订代码/config并核验固定父身份，再新目录恢复Proposal1280→1920；Gate新optimizer/scaler完成640；仅一次四条件Quick-Val，完成后停止。其他AMP/scaler1024、TF32、结构/loss、batch/种子/逻辑预算不变。
- 原预检3+3、单次受控重放及三组失败现场对照均已执行结束，**不重复、不增加GPU检查**。无Main-Val、official test、新seed、sweep、追加预算或模型重设计授权。数值/身份错误立即停，保留证据，不自动重试或weights-only续训。
- 2026-10-01 08:32 UTC已完成训练专用精度标记和固定1280完整恢复入口；定点语法、零forward CPU合同/原恢复边界/训练标记及eval关闭检查通过，普通跨提交恢复仍拒绝。直接diff确认原phase_loss、Proposal/Gate架构与Quick-Val文件未改。新合同SHA **`5f437c5b470077476f28b8e1ecceddf119cc402045e723f5a1aadbc519c6cc2e`**；仅移除新增精度字段后精确匹配原合同SHA。当前准备提交/push，完整显式兼容性CPU检查在代码提交后执行；剩余GPU运行尚未开始。

## 已核验输入与准确恢复点

- 仓库 `/root/rivermind-data/DFormer`，分支 `perf/mmfr-a2-v3-pipeline-opt1`，origin `https://github.com/Newton-liub/DFormer.git`。开始本次继续时HEAD为`715053d8b657229e60dec548a12134b62ac5d284`且工作区clean；允许普通提交/push同名分支，无强推/历史改写。
- C0（冻结基础模型）仓库外 `/root/rivermind-data/cloud/MMFR_E1_Batch1A_local_transfer_20260922/C0/checkpoint/update-2560.pth`；SHA **`ca618b23d18eabb201a0d11d18da383ac99576d0feae5864e3233bda527d9a1a`**，812个source model keys；通过显式`--source-checkpoint`定位，原权重不改。
- Dataset `/root/rivermind-data/dataset/MUSeg_DFormer`；train-dev1277，SHA `a6b15b63f6d5193e3928ea24ada25be403a48e68d1c1f9372cdbbc3fe5cd8470`；val-dev318，SHA `1d0719d8f64f016d48995c25ab66d4004d76b7155d9efeef7cbb7454c0dd0e83`；计数、唯一性及互斥已核验。official test保持 **`sealed_unread`**。
- 唯一批准父完整恢复点：`cloud/mmfr-av1-formal-v1/formal/checkpoint/update-1280.pth`，SHA **`05714d165925c4549aa1111564bbca7d8b8d9896d337c5b68f0b7bb98590564d`**；父runtime **`91c7f96d3768fde0f478e03a4ccda8c6f19368dd`**、原合同SHA **`96e93e02bfb2ad8214ab26f826ca66728a80231e16133f81a04709036578f7cd`**。
- 原完整1280的CPU检查通过：Proposal attempted/completed1280、skip0，model/optimizer浮点状态finite；scaler1024/growth_tracker1280；下一epoch11/iteration0，next LR1.185953940035674e-5；Python/NumPy/torch CPU/CUDA RNG、scheduler及cursor齐全。正式必须全部恢复，不能只取权重；原文件及protocol字典不修改。**没有完整1870恢复点**。
- 上次直接GPU核验为08:04:48 UTC：RTX4090，torch2.1.2+cu118，used1MiB/utilization0%，实验进程均退出。云实例未关闭/销毁；本次新运行前需重新核验资源。上传ZIP仅查看目录，未解压覆盖代码或采用其未核验C0。

## 原运行与诊断结论（独立历史证据）

1. 原正式目录`cloud/mmfr-av1-formal-v1`，console同名`-console.log`，runtime91c7f96。两阶段真实预检均attempted/completed3、skip0、全部finite、scale1024；Proposal/Gate peak allocated2922.771973/3007.839844MiB，reserved3606MiB；elapsed12.550926/6.593992s。Gate utility正/负/模糊0/0/30是合法路径，不是提前停理由。权威`preflight-result.json`；预检模型/RNG被丢弃后从C0正式初始化。
2. 原正式在Proposal **attempt1871/completed1870/skip0**（epoch15/iteration78）停止；报错`missing/nonfinite active gradient: av1.proposal.0.weight`，位于backward/unscale后、scaler.step/update前，因此skip0不能证明失败尝试健康。启动05:58:33.928477、退出06:27:39.277 UTC；1743.153606s是从预检开始到失败，不是完成训练耗时。Gate与Quick-Val未开始。证据`formal/proposal-stopped.json`、`stopped.json`和原步骤日志保留。
3. 用户授权的单次重放目录`cloud/mmfr-av1-gradient-diagnosis-v1`，runtime`9c257a9557483cd4219399c83e3566ab1965d471`。07:19:04.287965→07:28:16.852 UTC，550.460960s；从完整1280重放，原1281–1870的 **590条loss/CE/KL/LR/计数/scaler等记录精确匹配**，同1871的有限loss0.07739032804965973在HAM/NMF `torch.bmm(x, coef)`的`BmmBackward0`反向出现NaN。只证明该区间已记录轨迹一致，不认证任意逐张量/跨硬件恢复等价；具体中间浮点超界机制未逐算子证明。
4. 保存现场`cloud/mmfr-av1-gradient-diagnosis-v1/formal/diagnostic-failure-state.pth`，SHA`6865e8865bd486fac5b9bf06317af768d5a4634e5e4af4ea839a41c174bb0fb5`；CPU核验model/输入finite、非Proposal张量与完整1280一致、pre-forward四类RNG齐全。它是 **diagnostic-only/nonresumable/non-performance-candidate**，没有完整optimizer/resume状态，不能用于正式1870续训。anomaly提前中止后的grad None不证明原错误是缺失梯度。
5. 三组同现场对照目录`cloud/mmfr-av1-failure-controls-v1`，runtime`9092bfbc49100717afd33aadcabf99a4272d1e5b`。恰好6 forward/3 backward/0 optimizer更新，全部model张量未改。权威`controls-result.json`：原AMP/scale1024全部16992梯度标量NaN、Inf0；scale1全部finite、前向loss/CE/KL完全相同；仅NMF局部FP32/scale1024全部finite，loss0.07738957554101944（变化−7.525086402893066e-7）。证明单失败batch的scale/局部精度依赖，不证明整轮稳定或性能，也不宣称两种有限梯度相同。

完整历史身份与细节保留在 [复现证据](../../MMFR/02_evidence/reproducibility_current.json)、原输出及Git history；不改写原失败/诊断结果。最新授权是后续独立决定，而非追改历史诊断授权。

## 当前实施合同与下一恢复步骤

- Config唯一新增数值字段`nmf_training_precision="float32"`；移除它后的完整原合同必须匹配原SHA。仅A-v1训练、CUDA autocast开启时局部关闭NMF autocast并输入转FP32；C0/NMF保持eval，用显式A-v1训练标记而非NMF.training控制。NMF算法/迭代/RNG不变，其他模型和FP32评价默认路径不变。
- 显式`--precision-amendment-full1280`只接受上述固定父commit/合同/checkpoint SHA、同source/splits/数据、完整optimizer/scaler/RNG/cursor；审定代码差异后原样恢复。新runtime/合同与父身份分开记录，普通恢复仍严格同commit/合同；新checkpoint另存修订lineage（来源链），不伪装父身份。
- 原冻结参数仍为Proposal1920/Gate640，AdamW LR3e-5/WD0.01，margin0.01/lambda_clean0.1；各阶段128-successful-update warmup/poly0.9；batch10、480×640、workers8、accumulation1、AMPfp16/TF32 on、scaler1024/growth2/backoff0.5/interval2000。C0/BN冻结，训练Val禁用，无utility早停。
- Proposal剩640完成保存`proposal-update-1920.pth`，Gate使用同一Proposal和新optimizer/scaler、独立原phase seed；640成功后唯一性能候选`update-2560.pth`。新目录、不覆盖历史；续训段时间明确单列，不冒称完整冷启动时间。
- 唯一Quick-Val：clean、entire_missing@1.0、spatial_dropout@0.75、misalignment@0.75；318 val-dev、同fixed-final off/full/learned、同输入/配对HAM随机状态、original-full/scale1/no flip/FP32/TF32 off/eval seed2026091401/reset-per-unit。strict off与独立C0核验并入首个真实样本；冻结张量/标签网格/累计confusion核验并入本轮。
- 继续线：learned三hard平均相对off≥+0.50pp，clean≥−0.20pp，learned hard严格>full；单seed只筛选，不是统计显著性。评价输出非空时不得另建目录静默重跑；正式完成的resume被拒绝，以防重复Quick-Val。错误立即停止保留现场，恢复兼容性不明或需其他合同改变时交人工。

## Git、证据与交付边界

当前实现入口`tools/mmfr/av1_train.py`、`tools/mmfr/av1_quickval.py`；原概率BCE已在autocast禁用的FP32中采用相同公式，结构/loss不变。A-v1修订已完成定点静态/CPU核验和直接diff复核，待提交/push后的完整兼容性检查通过才启动GPU；结果验收仍由主代理完成。未运行完整测试、重复Gate-B、额外GPU诊断或其他评价。

`MMFR/99_review_packet_current/`仍为以前生成的旧快照；环境无`pwsh`，不能运行既有`MMFR/98_tools/rebuild_review_packet.ps1`，未手工编辑生成产物。最新事实以两份实时文档、protocol §8及既有复现JSON为准。checkpoint/dataset/大日志不入Git/MMFR。下一准确动作是完成数值标记/固定1280恢复入口、最小检查与提交，再启动批准的剩余正式段；本次继续尚未产生训练或评价结果。
