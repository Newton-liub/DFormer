# MMFR A-v1 第一轮正式训练与 Quick-Val：上级审核报告

- 汇报周期：上一份 2026-09-30 实现/Gate-B 正式报告之后，至 2026-10-01 09:30 UTC 已核验结果；本报告于 2026-10-01 整理。
- 报告对象：上级模型，重点裁决本轮筛选后的研究处置，以及是否例外授权本地验证。
- 整理基线：`4f84b469c4b04de657eaf2c455bd44e991b7f869`；训练、评价与整理提交分别记录，不混为同一运行身份。
- 当前结论：正式训练已完成，唯一四条件 Quick-Val 的预冻结筛选结论为 **`stop`**；本次只报告、打包和 Git 交付，不运行新实验。

## 1. 可直接转交上级的结论

A-v1 的剩余正式训练已经成功完成，最终为 Proposal 1920 次、Gate 640 次成功更新，未发生优化器跳步。训练中出现过混合精度数值失败，经过单次受控重放和同现场对照，用户批准“仅训练中的 NMF 局部 FP32，从原完整 1280 恢复”；该修订完成了剩余 640+640 次更新。它是保留原先 1280 次历史的续训，不是全轮 FP32 重训。

**结果没有通过继续门槛。** 在同一最终权重、相同输入及配对随机状态下，learned 在三个困难条件的平均 mIoU 为 50.768491%，matched off 为 50.770364%，差值 **−0.001873 pp**，而预冻结要求为至少 +0.50 pp。learned 略高于 full，但 full 与 learned 都略低于 off，因此按照原规则停止。本轮只有一个 seed，微小差值不作显著性判断，也不能推导“所有同类方法无效”或“模型创新性成立”。

建议上级接受本轮 `stop`，保留全部证据，结束当前实验授权。如果上级认为仍需本地 Main-Val，应明确说明追加评价的研究问题与理由，并另行批准条件、行为、视图、预算、适配及最小资格核验；不能把已完成的 Quick-Val 授权延续成 Main-Val 权限。本次转移包只是保存材料和降低迁移成本，不是已经可直接启动的 Main-Val 实验包。

## 2. 关键概念与判断边界

| 名称 | 本报告中的含义与实际影响 |
| --- | --- |
| Proposal / Gate | Proposal 在编码器 stage2 生成候选补偿残差；Gate 为每图选择一个连续补偿强度。Gate 控制动作，不解释为深度可靠性或故障概率。 |
| off / full / learned | off 完全禁用补偿；full 使用全部补偿；learned 使用 Gate 的连续强度。matched off 是同一 checkpoint、同输入/随机状态下的禁用补偿对照，而不是引用旧 C0 分数。 |
| C0 | 本轮冻结的基础模型。训练只更新新增分支，原参数、buffer 和归一化状态保持冻结。 |
| HAM / NMF / AMP | HAM 是原分割解码头，NMF 是其中的非负矩阵分解；AMP 是自动混合精度。获批改动只提高 A-v1 训练时 NMF 的局部精度，未重设计该算法。 |
| checkpoint | 模型某时点的参数及训练状态文件。本次保留最终完整文件字节，不另导出成会改变身份的 weights-only 文件。 |
| Quick-Val / Main-Val | 本轮 Quick-Val 为四条件、318 样本/条件、单视图筛选；讨论中的 Main-Val 为十条件、多尺度/翻转评价，尚未执行，也尚未完成 A-v1 适配。 |
| mIoU / pp | mIoU 是类别平均交并比，以百分数报告；pp 是两个百分数的百分点差。三 hard 是三个困难条件 mIoU 的未加权平均。 |

## 3. 主要工作、修复与验证状态

### 3.1 必要入口与全尺寸预检——已完成并验证

实现 A-v1 薄训练入口 `tools/mmfr/av1_train.py` 和四条件评价入口 `tools/mmfr/av1_quickval.py`，复用既有模型、数据、损坏算法、指标及随机状态辅助工具。混合精度下 probability BCE 的局部 FP32 工程修复保持损失公式与权重不变。

原正式运行前，在真实 train-dev、batch10、480×640、AMP/TF32 开启的设置下，Proposal/Gate 各进行了 3 次成功更新预检，全部有限、skip0。预检状态随后丢弃，正式训练从原 C0、原初始化和阶段 seed 干净开始；预检更新不计入正式预算。该历史证据见 `cloud/mmfr-av1-formal-v1/preflight-result.json` 和既有复现记录中的 `cloud_execution`，本次不重复资格运行。

### 3.2 原训练数值失败、定位与批准修订——已完成并验证（限定范围）

1. 原 runtime `91c7f96d3768fde0f478e03a4ccda8c6f19368dd` 在 Proposal 第 1871 次尝试的 backward/unscale 后、optimizer step 前停止；已完成 1870、skip0，Gate 和评价均未开始。skip0 不能证明失败尝试健康。
2. 单次受控重放从原完整 1280 恢复，原 1281–1870 共 590 条 loss/CE/KL/LR/计数/scaler 记录精确匹配，并在第 1871 次定位到 NMF `BmmBackward0` 的 NaN。它只认证所记录区间的轨迹，不是任意硬件或任意张量的恢复等价认证。
3. 三组同现场对照共 6 forward、3 backward、0 optimizer 更新，全部模型张量未变。原 AMP/scale1024 的 16992 个活动梯度标量均为 NaN；scale1 或 NMF 局部 FP32/scale1024 均有限。这支持单 batch 的精度/scale 依赖，具体中间浮点超界机制仍未逐算子证明。
4. 用户批准 `fp32-full1280-resume`，仅 A-v1 训练的 CUDA autocast 路径将 NMF 局部转 FP32，恢复原完整 1280 的 model、AdamW、GradScaler、scheduler、Python/NumPy/torch CPU/CUDA RNG 与数据 cursor；保留原前 1280 次更新，执行剩余 Proposal640 和 Gate640。

代码涉及 `local_configs/MUSeg/DFormerv2_S_MMFR_AV1.py`、`models/builder.py`、`models/decoders/ham_head.py` 和 `tools/mmfr/av1_train.py`。使用显式 A-v1 训练标记，因为冻结 C0/NMF 在训练期间仍保持 eval，不能以 `NMF.training` 判断是否启用修订。算法、迭代和随机初始化不改；其他 AMP、scaler1024、TF32、架构、loss、batch、seed 与逻辑预算不变。普通跨提交恢复仍严格拒绝，显式兼容路径只接受固定父身份。

### 3.3 修订后的正式续训——已完成并验证

| 项目 | 实际结果 |
| --- | --- |
| 训练 runtime | `812507385a1b4b805966799bffb9a13f4704e492`，执行前已提交/push并核验远端 |
| 数值合同 SHA-256 | `5f437c5b470077476f28b8e1ecceddf119cc402045e723f5a1aadbc519c6cc2e` |
| Proposal / Gate | 1920 / 640 attempted 与 completed；各 skip0；本次实际新增成功更新 640+640=1280 |
| 最终全局计数 | 2560；唯一候选 `formal/checkpoint/update-2560.pth` |
| 恢复段耗时 | Proposal剩640：697.925962s；Gate640：837.754407s；入口合计1539.011626s，约25分39秒 |
| peak allocated | Proposal2943.554688MiB；Gate3026.777344MiB |
| peak reserved | Proposal3578MiB；Gate3754MiB |

以上时间只覆盖本次恢复段，不含原先保留的 1280 次更新，也不是冷启动整轮耗时或通用 benchmark。原失败坐标 1871 在修订轨迹中成功、loss0.07739216089248657；不声称它与原 FP16 轨迹相同。

训练仍为 AdamW LR3e-5/WD0.01、margin0.01/lambda_clean0.1、各阶段128-successful-update warmup/poly0.9；batch10、480×640、workers8、accumulation1，AMPfp16/TF32开启，scaler1024/growth2/backoff0.5/interval2000；C0/BN冻结，训练Val关闭，无utility早停或checkpoint选择。Gate保留同一Proposal，使用全新Gate optimizer/scaler。

独立 CPU 核验：最终/transition hash 与完整 metadata/lineage 通过；最终 model/optimizer 的浮点状态全部有限；812 个 C0 张量与源文件精确一致；最终 Proposal 与 transition 精确一致。本次新增1280条训练记录连续、finite、optimizer_applied、scale1024、skip0和LR坐标已核验，核验未新增模型 forward。

### 3.4 评价初始化修复与唯一评分——已完成并验证

第一次自动评价在任何样本 forward 之前，因 NumPy `norm_mean/norm_std` 数组的配置相等比较出现 `ValueError: truth value of NumPy config array is ambiguous`。这是比较器工程异常，不是已经证明的模型身份不一致。原 `stopped.json`、console 与空 `quickval/` 保留，评分0/样本forward0。

最小修复只把 NumPy 数组比较改为 `np.array_equal`，其他比较与评价算术不变，真实不一致继续拒绝。CPU 检查接受相同配置、拒绝扰动后的 `norm_mean`。评价 runtime 为 `c798ed8f27483157ba6b164dafec4995bc47fce1`，执行前已提交/push并核验远端。修复后仅完成一轮实际评分，没有重训或重复样本评价。

## 4. 四条件 Quick-Val 结果

- 数据：冻结 val-dev，每条件318个唯一样本；同一最终checkpoint。
- 输入/评价：original-full、scale1、no flip、FP32、TF32 off，evaluation seed2026091401，reset-per-unit；off/full/learned 共享 observed 输入并回放配对 HAM RNG。
- 时间：2026-10-01 09:11:42.064156 至 09:21:50.023665 UTC，共607.959805s；评价进程退出码0。
- 表内 mIoU 显示六位，判定和差值来自未舍入值。

| 条件 | off mIoU% | full mIoU% | learned mIoU% | learned−off pp | Gate均值 / 中位数 |
| --- | ---: | ---: | ---: | ---: | ---: |
| clean | 53.456270 | 53.436865 | 53.446238 | −0.010031370 | 0.542778 / 0.543237 |
| entire_missing@1.0 | 48.762350 | 48.766590 | 48.764849 | +0.002499572 | 0.543209 / 0.544448 |
| spatial_dropout@0.75 | 51.397338 | 51.396645 | 51.397212 | −0.000125355 | 0.540945 / 0.541768 |
| misalignment@0.75 | 52.151405 | 52.135082 | 52.143411 | −0.007993225 | 0.542748 / 0.542533 |
| 三hard未加权平均 | 50.770364 | 50.766106 | 50.768491 | −0.001873003 | — |

| 原冻结继续条件 | 未舍入实际值 | 结果 |
| --- | ---: | --- |
| learned hard−off ≥ +0.50pp | −0.0018730026999946858pp | **FAIL** |
| learned clean−off ≥ −0.20pp | −0.010031370364394832pp | PASS |
| learned hard严格高于full | +0.0023852985788366254pp | PASS |

full hard−off 为 −0.004258301278831311pp。full 与 learned 的困难条件平均均未超过 matched off，按冻结描述性规则为 `stop`，未增加容差区间或新阈值。

严格身份及指标核验：四条件各318唯一样本；812个冻结C0参数/buffer精确一致；首个真实clean样本 strict off 与独立C0 logits逐值相等、max_abs_error0、Proposal/Gate调用0。保存的3816个逐样本行为confusion已重新求和并核对聚合、mIoU、hard平均、delta、Gate均值/中位数与筛选规则，无额外GPU forward。Gate四条件均值接近，只报告观察，不能仅凭均值推导有效选择能力或其失败机制。

## 5. 权重、代码、数据与证据身份

| 对象 | 身份 |
| --- | --- |
| A-v1最终checkpoint | SHA-256 `87ec54d10192d3aedc1d6edb864b7b60b94ce66220a748a12f1ca50ca605d336` |
| Proposal transition | SHA-256 `979b99d1928b42d87ce45d38701791279b56ab78e9e2ef015c9dd5f06da4d26a` |
| 唯一批准父恢复点 | 原runtime91c7f96的完整1280，SHA-256 `05714d165925c4549aa1111564bbca7d8b8d9896d337c5b68f0b7bb98590564d` |
| 冻结C0 | SHA-256 `ca618b23d18eabb201a0d11d18da383ac99576d0feae5864e3233bda527d9a1a`，812 source keys |
| train-dev | 1277，SHA-256 `a6b15b63f6d5193e3928ea24ada25be403a48e68d1c1f9372cdbbc3fe5cd8470` |
| val-dev | 318，SHA-256 `1d0719d8f64f016d48995c25ab66d4004d76b7155d9efeef7cbb7454c0dd0e83` |
| 原数值合同 | SHA-256 `96e93e02bfb2ad8214ab26f826ca66728a80231e16133f81a04709036578f7cd`；删除唯一新增precision字段后与此精确一致 |
| 运行环境 | RTX4090，torch2.1.2+cu118；跨硬件/软件逐位一致尚未验证 |

仓库内详细入口：`MMFR/02_evidence/reproducibility_current.json` 的 `current_cloud_outcome_pointer` 指向 `cloud_precision_amended_resume`。其他 cloud 对象保留原历史范围，其旧 pending/authorized 字段不代表当前权限。协议 §8 保留批准修订，旧报告和原始结果不追改。

仓库外/运行目录证据，转移包中按原字节保存选定材料：

- `cloud/mmfr-av1-formal-fp32-resume1280-v1/identity.json`、`precision-amendment-compatibility.json`、`training-result.json`、`formal/proposal-result.json`、`formal/gate-result.json`。
- 同根 `quickval-config-equality-fix-v1/summary.json` 及四条件 `metrics.json`，包含逐样本行为记录；保留原数组比较失败的 `stopped.json`。
- 原运行 `cloud/mmfr-av1-formal-v1` 的预检/停止小证据；诊断 `cloud/mmfr-av1-gradient-diagnosis-v1` 与现场对照 `cloud/mmfr-av1-failure-controls-v1` 的小型结论证据。
- 原始console、步骤日志、父/transition/诊断checkpoint在原位置继续保留；转移包不等于全训练恢复归档。诊断1870现场是nonresumable/non-performance-candidate，未用作正式续训，也不打包为验证候选。

official test 保持 **`sealed_unread`**，未读取其split或图片。包不含dataset、test清单/标签/缓存、论文全文、外部clone、旧生成审核包或Git历史。

## 6. 上级需要裁决什么

当前执行边界是停止实验并完成资料交付；对本轮 `stop` 的接受不需要额外运行。若上级例外要求追加本地val，请在回复中明确：

1. 追加验证要回答什么尚未解决的问题，为什么值得在未通过筛选后投入。
2. 是新授权的四条件重评，还是十条件Main-Val；选择off/full/learned三行为还是仅learned。仅learned不能回答同权重补偿净收益与selector相对full问题。
3. 样本、条件、尺度/翻转、FP32/TF32、随机配对、标签网格与指标聚合口径；禁止运行后选择有利条件或再定数值去留线。
4. 是否批准必要的A-v1 Main-Val薄适配与最小资格核验，以及允许的硬件/时间预算。现有入口只有四条件单视图，不能直接把旧十条件入口当成A-v1合格入口。
5. fixed-final唯一候选、C0及val-dev身份不变，official test仍封存；无新seed、调参、续训或结构改动。

估时仅供决策：在同RTX4090、十条件×318样本、五尺度(0.5/0.75/1/1.25/1.5)×原图/水平翻转共十视图、相同FP32评价假设下，按本轮Quick-Val实测607.96s粗略外推，off/full/learned合计约4–5小时，保守留6小时；仅learned约1.5–2小时。**这是外推，不是A-v1 Main-Val实测保证，不含适配开发/审核时间，也不直接适用于本地笔记本GPU。**

## 7. 本地条件性交接与本次检查

转移包计划固定位置：`/root/rivermind-data/cloud/MMFR_AV1_local_val_conditional_20261001.zip`；实际大小、整包SHA和验证结果以交付收据及实时状态为准。包包含最终A-v1和C0原文件、基线提交的限定源码快照、冻结train/val-dev清单、报告/协议/交接与相关小证据。dataset由本地独立提供。源码快照不是Git仓库；需Git身份的入口应从GitHub检出相应提交，不能用`git init`伪造原提交。

条件性操作交接：`MMFR/02_evidence/handoff_mmfr_a_v1_local_val_conditional_20261001.md`。它只允许资料接收、hash/路径准备，任何模型执行仍等待新的明确授权；本次没有生成自动训练/评价启动脚本。

本次只做直接证据/内容复核、改动差异、JSON与链接检查、包内容/CRC/关键字节SHA核验和Git交付检查，不运行完整测试、GPU、训练、重复Quick-Val、Main-Val或official test。既有完整状态/指标核验是上述已完成实验的历史证据，不冒充本次重新运行。

`MMFR/99_review_packet_current/` 是生成产物，当前环境无 `pwsh`；现有PowerShell生成器未运行，旧六文件包仍是旧快照，禁止冒充本轮报告。更新既有profile的源材料并保留生成器恢复点；当前可直接转交本份自包含报告，待具备PowerShell环境后由原 `MMFR/98_tools/rebuild_review_packet.ps1` 在canonical链接核验后重建正式六文件入口，不能手工覆盖它。

云资源最后直接核验为2026-10-01 09:30:44 UTC：实验进程已退出、GPU1MiB/0%，实例仍开启；本次不关闭或销毁实例，也不把打包完毕解释为停止计费。
