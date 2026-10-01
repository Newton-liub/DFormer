# MUSeg 当前状态与唯一实时入口

> **事实截至：** 2026-10-01 07:16 UTC。A-v1 两阶段 GPU 预检均 **PASS**，但正式 Proposal 在第1871次尝试、1870次成功更新后触发 `missing/nonfinite active gradient: av1.proposal.0.weight`，已于06:27:39 UTC自动停止。正式 Gate/Quick-Val未开始。运行 HEAD **`91c7f96d3768fde0f478e03a4ccda8c6f19368dd`**；当前为 **`formal stopped → bounded diagnostic authorized/preparing → formal recovery decision after diagnosis`**。完整1280步恢复点已通过CPU合同/有限状态核验，实际重放轨迹仍待核验。冻结合同见 [A-v1 protocol](../../MMFR/01_research/mmfr_a_v1_action_utility_protocol.md)，当前阻塞见 [开放决策](MUSeg-open-decisions.md)。

## 当前阶段与实际意义

A-v1 包含 stage2 补偿残差 Proposal（候选补偿动作）和 Gate（每图连续补偿强度选择器）；Quick-Val 是四条件开发集筛选。**大白话：** 全尺寸预检通过，但正式训练后来出现梯度异常，已按停止规则退出；现在没有最终权重或性能分数。恢复文件可读、状态有限不等于恢复后能稳定完成，更不能当作模型有效。

2026-10-01 05:55 UTC 用户明确“已经开启有卡模式，请继续”，恢复既定连续执行授权：两阶段各3次成功更新预检，通过后从原 C0 正式 Proposal1920/Gate640，再做唯一一次四条件 Quick-Val，完成后停止。06:55 UTC用户再次要求“继续”，在看到停止原因后明确选择受控诊断；本次允许审定兼容性后从完整1280最多重放到原1871坐标，不自动接正式剩余训练/Gate/评价。未改变数值合同。无 Main-Val、official test、新 seed、调参、batch sweep 或追加训练授权。

## 已直接核验的输入与版本

- 分支 `perf/mmfr-a2-v3-pipeline-opt1`，origin `https://github.com/Newton-liub/DFormer.git`。用户交接基线 `77738457eeb2b12221545e7db1a84ec95559ae47` 已 fetch/核验；实际运行版本为 `91c7f96d3768fde0f478e03a4ccda8c6f19368dd`，启动前工作区 clean，运行期间只更新状态/证据，不修改代码或 HEAD。
- 05:56 UTC 直接确认 CUDA True、RTX4090 24564MiB 且启动前空闲，torch `2.1.2+cu118`。06:55 UTC直接确认原Python PID2014已不存在，GPU used1MiB/utilization0%。仅训练进程停止，未关闭或销毁云实例。无卡准备时 CUDA False 属历史状态。
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
- 直接核验 Proposal **attempted1871/completed1870/skipped0**。最后成功步loss0.1657567471、finite true、scale1024、optimizer_applied true。第1871次尝试在backward/unscale后的梯度门禁抛错，**尚未进入scaler.step/update，因此skip0不能证明失败尝试健康**。现有报错把“缺失梯度”和“非有限梯度”合并，未保存失败batch/梯度，具体类别及根因待核验；不擅自归因为AMP溢出、坏数据或HAM。Gate0/640，fixed-final/Quick-Val未产生。
- 最近完整恢复点已CPU核验：`formal/checkpoint/update-1280.pth`，SHA-256 **`05714d165925c4549aa1111564bbca7d8b8d9896d337c5b68f0b7bb98590564d`**；现有`validate_resume`严格身份/合同/计数/scheduler/data cursor检查通过，模型和optimizer浮点张量均finite。scaler1024、growth_tracker1280，下一epoch11/iteration0、next_lr1.1859539400e-5，Python/NumPy/torch CPU/CUDA RNG字段齐全。**仅证明文件/状态满足CPU检查，不证明实际中断续训等价；1281–1870没有更近完整恢复文件，不能从1870续训。**
- 恢复门禁绑定同一完整commit；普通`all --resume`门禁保持原样。用户已选择并授权**仅诊断**的跨提交兼容性审定：新入口必须核验原91c7f96/1280checkpoint固定SHA、同source/split/完整合同、Git差异仅runner诊断及三份实时证据；原checkpoint协议字典只读，与实际诊断runtime commit分别记录，不伪装同提交。核心forward/loss/数据/随机流代码未改。诊断重放对原1281–1870的590条成功记录逐步精确比对loss/CE/KL/LR/计数/scaler等，任一不一致立即停；这提供本次恢复轨迹核验，不扩展成普遍恢复等价结论。
- 训练合同不变：Proposal1920/Gate640；margin0.01、lambda_clean0.1、AdamW LR3e-5/WD0.01；各阶段新 optimizer/scaler、128-successful-update warmup/poly0.9；batch10、480×640、workers8、accumulation1、AMP fp16、TF32 on；冻结 C0/BN，训练中 Val 关闭。按 successful update 计预算，不因 utility/gate 分布提前停止。
- 每640次保存完整恢复状态（128-update数据epoch边界），包含 model/optimizer/scaler/scheduler/RNG/cursor/identity，非持久workers在下一epoch从保存的主RNG建立。严格同提交/合同恢复；**中断续训运行等价未实验验证**。禁止静默 weights-only 续训；恢复兼容性不确定时交人工。
- Proposal1920产物 `proposal-update-1920.pth`；Gate从同一Proposal进入并重置优化器/scaler，唯一性能候选为 Gate640/global2560 的 `update-2560.pth`。
- OOM、NaN/Inf、scaler skip、身份或基础设施错误立即停止保留原目录/日志，不覆盖、不自动降batch、不无限重试。普通工程修复先提交，只重做受影响检查；改变结构/loss/种子/预算/口径或恢复不确定须人工裁决。
- 正常路径由现有入口自动执行一次四条件 Quick-Val：clean、entire_missing@1.0、spatial_dropout@0.75、misalignment@0.75；同fixed-final的 off/full/learned，同输入及配对HAM随机状态；318 val-dev、original-full、scale1/no flip、FP32/TF32 off、eval seed2026091401/reset-per-unit。首个实际样本嵌入 strict off vs C0、冻结源张量及标签网格核验，逐条件核对 confusion 累积。
- 继续线：learned三hard平均相对off >=+0.50pp，clean >=−0.20pp，learned hard严格 >full；单seed只作筛选，不是统计显著性。评价后停止。formal-final `--resume` 被拒绝以防重复；仅训练完成且评价从未开始时可使用独立 evaluator；评价已开始/输出非空时先核对，不另建目录静默重跑。

## 实现、验证与 Git 交付

- 新薄入口 `tools/mmfr/av1_train.py`、`tools/mmfr/av1_quickval.py` 复用现有阶段函数、RGBXDataset/TrainPre/get_train_loader、v3 corruption、RNG/checkpoint及评价工具，不套旧E1循环，不改Proposal/Gate架构。
- 最小 AMP 安全修复：概率形式 BCE 用同一公式在 autocast禁用的FP32区执行；不改标签、margin或权重。GradScaler启动合同精确检查initial1024/growth2/backoff0.5/interval2000。代码初始提交 **`857f1da85a5d5a38a61fe1172a5f3c7d832cc652`**，scaler guard **`aa53eeea152e909e7b7ca2502823d5b10973c965`**，准备状态回执 **`91c7f96d3768fde0f478e03a4ccda8c6f19368dd`**；均已普通push到同名origin分支并核验，无强推或历史改写。
- 已完成必要轻量检查：语法/import、输入/冻结条件身份、LR首步/128步/末步、FP32 BCE等值/有限梯度/全模糊exact-zero、定点代码和差异复核；本次再完成真实3+3 GPU预检。未新增测试文件、未运行完整测试、重复Gate-B、额外seed、benchmark或其他评价。
- 现有 [复现信息](../../MMFR/02_evidence/reproducibility_current.json) 保留准备历史，并补当前执行证据；原 [Gate-B 报告](../../MMFR/02_evidence/report_mmfr_a_v1_implementation_gateb_20260930.md) 只代表 synthetic B1/64×64历史资格。上一准备阶段可由Git91c7f96追溯，不额外创建小动作归档。
- `MMFR/99_review_packet_current/` 仍为此前生成的旧快照；环境无 `pwsh`，未安装依赖或手工修改生成产物。最新事实以本文件/既有复现JSON为准，审核包重建待可用环境执行现有工具。
- 用户已选择受控诊断，新增`--execute diagnose --resume <原1280> --diagnostic-reference-dir <原run>`；最多到attempt1871，不保存可用于续训的新checkpoint，不进入Gate/评价。只在原失败坐标开启autograd anomaly记录，不增加forward/backward；异常保留失败输入、模型和pre-forward RNG到cloud诊断现场（明确nonresumable、非性能候选），源证据不覆盖。诊断输出计划`cloud/mmfr-av1-gradient-diagnosis-v1`；代码先提交并核验后启动，实际身份/结果仍待核验。诊断后正式恢复或任何数值合同变更仍需裁决，见开放决策。
