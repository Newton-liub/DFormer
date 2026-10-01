# MUSeg 当前状态与唯一实时入口

> **事实截至：** 2026-10-01 08:04 UTC。A-v1两阶段预检PASS；首次正式Proposal在attempt1871/completed1870触发梯度门禁停止（原runtime **`91c7f96d3768fde0f478e03a4ccda8c6f19368dd`**）。用户授权的单次受控诊断已结束：从完整1280恢复后，**590条原成功记录逐字段精确匹配**，同1871坐标的有限loss在HAM/NMF `torch.bmm(x, coef)`反向出现NaN；07:28:16 UTC自动退出。诊断runtime **`9c257a9557483cd4219399c83e3566ab1965d471`**，现场已保存/CPU核验，GPU当前空闲。当前为 **`formal stopped → replay + three precision controls completed → production precision/recovery decision required`**。三组现场对照恰好6次forward/3次backward/0 optimizer，全部模型张量未改；原AMP+scale1024复现NaN，scale1或NMF局部FP32+scale1024在同失败batch均给出有限梯度。正式精度合同仍未修改，整轮稳定性/性能待正式运行。Gate/Quick-Val未开始，无fixed-final/性能结论。冻结合同见 [A-v1 protocol](../../MMFR/01_research/mmfr_a_v1_action_utility_protocol.md)，阻塞见 [开放决策](MUSeg-open-decisions.md)。

## 当前阶段与实际意义

A-v1 包含 stage2 补偿残差 Proposal（候选补偿动作）和 Gate（每图连续补偿强度选择器）；Quick-Val 是四条件开发集筛选。HAM是原解码头组件，内部NMF（非负矩阵分解）用反复矩阵乘/除计算。**大白话：** 已证实这次恢复重放跟原训练的590步记录一致，并定位到同一步的矩阵乘反向NaN；不是还在训练，也没有最终成绩。输入/权重/前向loss均finite不代表反向稳定。三组同batch对照已证明该处数值异常依赖scale/局部精度，并支持NMF局部FP32作为单batch修复候选；不推断所有batch/整轮训练都稳定。

2026-10-01 05:55 UTC 用户明确“已经开启有卡模式，请继续”，恢复既定连续执行授权：两阶段各3次成功更新预检，通过后从原 C0 正式 Proposal1920/Gate640，再做唯一一次四条件 Quick-Val，完成后停止。06:55 UTC用户再次要求“继续”，在看到停止原因后明确选择受控诊断；本次允许审定兼容性后从完整1280最多重放到原1871坐标，不自动接正式剩余训练/Gate/评价。未改变数值合同。无 Main-Val、official test、新 seed、调参、batch sweep 或追加训练授权。

## 已直接核验的输入与版本

- 分支 `perf/mmfr-a2-v3-pipeline-opt1`，origin `https://github.com/Newton-liub/DFormer.git`。用户交接基线 `77738457eeb2b12221545e7db1a84ec95559ae47` 已 fetch/核验；实际首次正式版本为`91c7f96d3768fde0f478e03a4ccda8c6f19368dd`，启动前工作区clean，首次运行期间只更新文档。之后仅runner诊断补丁及实时证据提交为`9c257a9557483cd4219399c83e3566ab1965d471`，诊断启动前同样clean；两次运行中均未修改代码或HEAD。
- 05:56 UTC直接确认CUDA True、RTX4090 24564MiB、torch `2.1.2+cu118`。原Python PID2014和诊断PID31202均已退出；08:04 UTC直接确认GPU used1MiB/utilization0%，三组对照已结束。仅进程停止，未关闭或销毁云实例；有卡资源仍开启。无卡准备时CUDA False属历史状态。
- C0 使用仓库外 `/root/rivermind-data/cloud/MMFR_E1_Batch1A_local_transfer_20260922/C0/checkpoint/update-2560.pth`；直接 SHA-256 核验为 **`ca618b23d18eabb201a0d11d18da383ac99576d0feae5864e3233bda527d9a1a`**。配置原仓库内位置不存在，运行用 `--source-checkpoint` 显式定位，未改变权重身份。
- 数据根 `/root/rivermind-data/dataset/MUSeg_DFormer`。冻结 train-dev1277/val-dev318 的计数、唯一性、互斥及 SHA 均已直接核验：train `a6b15b63f6d5193e3928ea24ada25be403a48e68d1c1f9372cdbbc3fe5cd8470`；val `1d0719d8f64f016d48995c25ab66d4004d76b7155d9efeef7cbb7454c0dd0e83`。运行 `identity.json` 再次记录同一输入身份；contract SHA `96e93e02bfb2ad8214ab26f826ca66728a80231e16133f81a04709036578f7cd`。
- 用户上传 `MMFR_AV1_cloud_transfer_20261001.zip` 已只查看目录，含 README、source7773845和321150608-byte C0；未解压、未覆盖源码、未计算ZIP内C0内容SHA，未入Git。
- official test 继续 **`sealed_unread`**，没有打开该划分或图片。原 Gate-B JSON 和历史报告不改写、不重跑。

## 两阶段真实预检结果

直接读取 `cloud/mmfr-av1-formal-v1/preflight-result.json` 及六条原始记录，真实 train-dev、batch10、480×640、workers8、AMP fp16、TF32 on：

| 预检阶段 | attempted/completed/skipped | peak allocated | peak reserved | mean update | mean wall含数据 | elapsed |
|---|---:|---:|---:|---:|---:|---:|
| Proposal | 3/3/0 | 2922.771973 MiB | 3606 MiB | 2.347539 s | 3.996947 s | 12.550926 s |
| Gate | 3/3/0 | 3007.839844 MiB | 3606 MiB | 0.851354 s | 2.130085 s | 6.593992 s |

- Proposal 每步 loss：0.0753513500、0.0833466277、0.2348547280；Gate：0.1413684338、0.0573739186、0.1116870567。全部 finite、optimizer_applied，scaler1024不变，无 OOM/NaN/Inf/skip。
- Gate action utility（同图 off CE减full CE）共30个样本：positive0/negative0/ambiguous30，BCE0。全模糊是冻结 margin 下的合法路径，learned CE 仍训练，**不是提前停止理由**。
- 时间只代表这3步（首步含启动），不作 benchmark 或正式耗时预测。通过后入口销毁预检模型和随机状态，从原 C0/冻结初始化/原种子重新开始正式训练；6次预检更新不计正式预算。

## 当前正式运行与准确恢复点

- **正式任务已停止，不自动重试。** 启动时间 `2026-10-01T05:58:33.928477+00:00`，入口 `tools/mmfr/av1_train.py --execute all`；原shell PID2000、Python PID2014，06:27:39 UTC以exit1退出。入口记录从预检开始到停止1743.153606s，不能当作已完成正式训练耗时。
- 输出根 `/root/rivermind-data/DFormer/cloud/mmfr-av1-formal-v1`；console `/root/rivermind-data/DFormer/cloud/mmfr-av1-formal-v1-console.log`；`identity.json`、`preflight-result.json`、`formal/proposal-steps.jsonl`、`formal/proposal-stopped.json`及根`stopped.json`全部保留。
- 直接核验 Proposal **attempted1871/completed1870/skipped0**。最后成功步loss0.1657567471、finite true、scale1024、optimizer_applied true。第1871次尝试在backward/unscale后的梯度门禁抛错，**尚未进入scaler.step/update，因此skip0不能证明失败尝试健康**。现有原报错把“缺失梯度”和“非有限梯度”合并；本次受控复现已确认同坐标反向NaN，定位见下节。具体精度候选已做同batch对照，见下一节；没有据此修改正式合同或宣称整轮修复PASS。Gate0/640，fixed-final/Quick-Val未产生。
- 最近完整恢复点已CPU核验：`formal/checkpoint/update-1280.pth`，SHA-256 **`05714d165925c4549aa1111564bbca7d8b8d9896d337c5b68f0b7bb98590564d`**；现有`validate_resume`严格身份/合同/计数/scheduler/data cursor检查通过，模型和optimizer浮点张量均finite。scaler1024、growth_tracker1280，下一epoch11/iteration0、next_lr1.1859539400e-5，Python/NumPy/torch CPU/CUDA RNG字段齐全。**仅证明文件/状态满足CPU检查，不证明实际中断续训等价；1281–1870没有更近完整恢复文件，不能从1870续训。**
- 恢复门禁绑定同一完整commit；普通`all --resume`门禁保持原样。用户已选择并授权**仅诊断**的跨提交兼容性审定：新入口必须核验原91c7f96/1280checkpoint固定SHA、同source/split/完整合同、Git差异仅runner诊断及三份实时证据；原checkpoint协议字典只读，与实际诊断runtime commit分别记录，不伪装同提交。核心forward/loss/数据/随机流代码未改。诊断重放对原1281–1870的590条成功记录逐步精确比对loss/CE/KL/LR/计数/scaler等，任一不一致立即停；这提供本次恢复轨迹核验，不扩展成普遍恢复等价结论。
- 训练合同不变：Proposal1920/Gate640；margin0.01、lambda_clean0.1、AdamW LR3e-5/WD0.01；各阶段新 optimizer/scaler、128-successful-update warmup/poly0.9；batch10、480×640、workers8、accumulation1、AMP fp16、TF32 on；冻结 C0/BN，训练中 Val 关闭。按 successful update 计预算，不因 utility/gate 分布提前停止。
- 每640次保存完整恢复状态（128-update数据epoch边界），包含 model/optimizer/scaler/scheduler/RNG/cursor/identity，非持久workers在下一epoch从保存的主RNG建立。严格同提交/合同恢复；**普遍/逐张量恢复等价未认证，本次590条已记录轨迹匹配见诊断节**。禁止静默 weights-only 续训；恢复兼容性不确定时交人工。
- 合同约定Proposal1920应保存`proposal-update-1920.pth`；Gate从同一Proposal进入并重置优化器/scaler，Gate640/global2560后唯一性能候选为`update-2560.pth`。**两个产物均尚未生成，本轮未达到这些坐标。**
- OOM、NaN/Inf、scaler skip、身份或基础设施错误立即停止保留原目录/日志，不覆盖、不自动降batch、不无限重试。普通工程修复先提交，只重做受影响检查；改变结构/loss/种子/预算/口径或恢复不确定须人工裁决。
- 正常路径由现有入口自动执行一次四条件 Quick-Val：clean、entire_missing@1.0、spatial_dropout@0.75、misalignment@0.75；同fixed-final的 off/full/learned，同输入及配对HAM随机状态；318 val-dev、original-full、scale1/no flip、FP32/TF32 off、eval seed2026091401/reset-per-unit。首个实际样本嵌入 strict off vs C0、冻结源张量及标签网格核验，逐条件核对 confusion 累积。
- 继续线：learned三hard平均相对off >=+0.50pp，clean >=−0.20pp，learned hard严格 >full；单seed只作筛选，不是统计显著性。评价后停止。formal-final `--resume` 被拒绝以防重复；仅训练完成且评价从未开始时可使用独立 evaluator；评价已开始/输出非空时先核对，不另建目录静默重跑。

## 单次受控诊断结果与下一恢复边界

- 用户选择的单次受控复现已按上限执行结束：`--execute diagnose`从完整1280恢复，实际重做590次原成功更新+第1871次失败尝试；**不追加正式预算，不接Gate/评价，不生成可续训的新checkpoint**。诊断从`2026-10-01T07:19:04.287965+00:00`开始，07:28:16.852 UTC以exit1自动退出，入口计时550.460960s。runtime **`9c257a9557483cd4219399c83e3566ab1965d471`**已push/远端直接核验。
- 输出`cloud/mmfr-av1-gradient-diagnosis-v1`，console同名`-console.log`；`diagnostic-compatibility.json`、`formal/proposal-steps.jsonl`、`formal/proposal-stopped.json`、`formal/diagnostic-failure.json`及根`stopped.json`全部保留。590条1281–1870的loss/CE/KL/LR/计数/scaler及冻结记录字段逐字段精确一致。**只证明本次区间的已记录轨迹匹配，不认证任意恢复逐张量/跨硬件等价。**
- 原1871坐标（epoch15/iteration78）loss **0.0773903280**、CE0.0773900300、KL2.9797172374e-6，均finite；仅此坐标开启autograd anomaly，无额外forward/backward。首个检测到NaN的算子为 **`BmmBackward0`第1输出（0-based索引1，即第二个返回梯度）**，forward堆栈指向 `models/decoders/ham_head.py` 的 `NMF2D.local_step`，`numerator = torch.bmm(x, coef)`（当前line130）。这是**反向非有限**定位；随后的三组对照见下一节，详细哪一个浮点中间量先超界仍未逐算子证明。
- anomaly在backward中提前抛错，尚未unscale/optimizer step；诊断现场的Proposal.grad为None是反向提前中止后的状态，**不能据此把原异常改判为梯度缺失**。scaler skip仍0，失败尝试未被计作成功更新。
- 现场`formal/diagnostic-failure-state.pth`已CPU直接核验，SHA **`6865e8865bd486fac5b9bf06317af768d5a4634e5e4af4ea839a41c174bb0fb5`**：model全finite、非Proposal张量与完整1280完全一致、保存输入浮点张量全finite、batch10/480×640、label网格一致、pre-forward RNG四类齐全。该文件明确 **diagnostic-only/nonresumable/non-performance-candidate**，不含正式resume version/optimizer状态；禁止仅取其1870权重冒充完整续训。
- 单次恢复重放授权已经执行结束，不再重跑590步。用户随后明确批准的三组失败现场精度对照已经完成，具体结果见下一节。`tools/mmfr/av1_failure_controls.py`只在诊断进程临时替换NMF forward，不改正式HAM/phase_loss/config，不做任何optimizer更新；原现场不覆盖，不再重跑590步。
- 普通`all --resume`同提交门禁保持原样；本次仅诊断显式审定parent身份与实际runtime差异，不改写原checkpoint。后续正式跨提交恢复、修复是否需要新正式起点尚未决定，见开放决策。原训练/预检/诊断证据均不覆盖。

## 三组失败现场精度对照与剩余决定

- runtime **`9092bfbc49100717afd33aadcabf99a4272d1e5b`**已提交/push并远端直接核验；实际开始`2026-10-01T08:01:54.824896+00:00`。输出`cloud/mmfr-av1-failure-controls-v1`，console同名`-console.log`，权威结果`controls-result.json`及每组小JSON。恰好**6次模型forward、3次backward、0次optimizer更新**，末尾全部model张量与原现场逐张量相同，无训练更新、Gate或评价。

| 同一失败batch的对照 | loss | Proposal全部梯度 |
|---|---:|---|
| 原AMP/scale1024 | 0.07739032804965973 | 全16992标量NaN，梯度存在，Inf0 |
| 原AMP/scale1 | 0.07739032804965973 | 全finite，NaN/Inf0 |
| 仅NMF局部FP32/scale1024 | 0.07738957554101944 | 全finite，NaN/Inf0 |

- 第一组loss/CE/KL精确复现原失败现场；第二组只改变scale，前向三项完全一致。第三组只临时禁用NMF autocast并把其输入转FP32，其他模型保持AMP/TF32/scaler1024；loss相对原组−7.5250864e-7。**大白话：** 原错误是存在的NaN梯度而非梯度缺失；该批次在较低scale或较高NMF计算精度时均能完成有限反向。局部FP32的梯度与低scale方案不宣称相同。
- 主代理建议只批准**A-v1训练时NMF局部FP32**，其余AMP/scaler1024/结构/loss/种子/batch/预算保持冻结，比全局降低scale的改动更局部；这是数值执行合同修订，必须明确批准，不能当作原合同不变。单batch对照不证明整轮稳定或性能提升。
- 待决定恢复路线：从原完整1280（model+optimizer+scaler+RNG/cursor全部）恢复，完成Proposal剩640/Gate640及唯一Quick-Val；或批准同样精度修订后从原C0干净重启1920+640；或停止本轮。无论哪条都不覆盖原失败证据，不拿1870诊断权重冒充续训，异常立即停。**尚未修改正式精度/恢复门禁或启动剩余训练。**

## 实现、验证与 Git 交付

- 新薄入口 `tools/mmfr/av1_train.py`、`tools/mmfr/av1_quickval.py` 复用现有阶段函数、RGBXDataset/TrainPre/get_train_loader、v3 corruption、RNG/checkpoint及评价工具，不套旧E1循环，不改Proposal/Gate架构。
- 最小 AMP 安全修复：概率形式 BCE 用同一公式在 autocast禁用的FP32区执行；不改标签、margin或权重。GradScaler启动合同精确检查initial1024/growth2/backoff0.5/interval2000。代码初始提交 **`857f1da85a5d5a38a61fe1172a5f3c7d832cc652`**，scaler guard **`aa53eeea152e909e7b7ca2502823d5b10973c965`**，准备状态回执 **`91c7f96d3768fde0f478e03a4ccda8c6f19368dd`**；均已普通push到同名origin分支并核验，无强推或历史改写。
- 已完成必要轻量检查：语法/import、输入/冻结条件身份、LR首步/128步/末步、FP32 BCE等值/有限梯度/全模糊exact-zero、定点代码和差异复核；本次再完成真实3+3 GPU预检。未新增测试文件、未运行完整测试、重复Gate-B、额外seed、benchmark或其他评价。
- 现有 [复现信息](../../MMFR/02_evidence/reproducibility_current.json) 保留准备历史，并补当前执行证据；原 [Gate-B 报告](../../MMFR/02_evidence/report_mmfr_a_v1_implementation_gateb_20260930.md) 只代表 synthetic B1/64×64历史资格。上一准备阶段可由Git91c7f96追溯，不额外创建小动作归档。
- `MMFR/99_review_packet_current/` 仍为此前生成的旧快照；环境无 `pwsh`，未安装依赖或手工修改生成产物。最新事实以本文件/既有复现JSON为准，审核包重建待可用环境执行现有工具。
- 诊断补丁仅修改`tools/mmfr/av1_train.py`的受控入口/兼容性/记录，现场对照新增`tools/mmfr/av1_failure_controls.py`；正式phase_loss、HAM、Proposal/Gate、数据/RNG/数值合同未修改。实际做了定点语法/lint/diff检查、CPU恢复/兼容性/现场核验、单次受控重放及三组已授权对照；没有完整测试、其他GPU对照、额外训练或评价。代码提交9c257a9/9092bfb与重放结果回执fe3bcab均已普通push；三组对照结果只更新实时文档和既有复现JSON后另提交/push，文档不追逐自身提交SHA。
