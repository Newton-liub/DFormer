# MMFR A-v1 最小实现与 Gate-B 工程审核

- 日期：2026-09-30；独立身份：`MMFR-A-v1-action-utility-v1`。
- 基准 HEAD：`a7c5cc17c5b495f2210efa3604b05794b5fa16c4`；开始时已有未提交修改，未提交或推送。
- 权威执行边界：当前状态（`../../doc/main/MUSeg-current-status.md`）；固定合同：A-v1 protocol（`../01_research/mmfr_a_v1_action_utility_protocol.md`）。

**结论：A-v1 implementation complete; Gate-B PASS; ready to request formal Proposal → Gate training authorization。** 最小实现有资格申请正式两阶段训练；这只证明工程路径、冻结和监督逻辑成立，不证明补偿有效、gate 学会选择、收敛、泛化或论文创新性。真实工程 forward/backward 已停止，没有正式训练、Quick-Val/Main-Val、official test 或云端操作。

## 1. 实现内容与理论尺寸

A-v1 是任务效用控制的单点残差修正：先训练 residual proposal（补偿动作），再固定动作训练 gate（决定本次动作强度的每图标量），不是 Depth reliability/failure probability。

代码文件：

1. `models/mmfr_av1.py`（新增）：`ActionUtilityResidual`、`depth_stats`；严格 off、固定 full、连续 learned。
2. `models/builder.py`（定点修改）：可选 `av1`；stage2 接入、frozen backbone no-grad、原 HAM 保留梯度；A-v1 的 `train()` 固定所有 C0 模块 eval。未启用 A-v1 时保留既有路径。原几何 prior 未修改。
3. `utils/mmfr_av1_training.py`（新增）：`build_phase_batch`、`configure_phase`、`phase_loss`、per-image CE、margin targets、masked BCE、clean KL；复用旧 corruption、RNG helpers 与 optimizer Conv/Linear grouping，没有正式训练循环/自动阶段切换。
4. `local_configs/MUSeg/DFormerv2_S_MMFR_AV1.py`（新增）：deep-copy source C0 config，新运行 identity；旧 `e1_batch1` 关闭；无 evaluator source/selector/正式 schedule。正式 `margin`、`lambda_clean` 为 `None`，未选定超参数。
5. `tools/mmfr/av1_gateb.py`（新增、持久的必要工程入口）：最小 synthetic utility branches 与真实 C0/HAM 资格检查，不是大规模 test suite。

固定结构：stage index 2，256 channels，约输入 1/16；proposal 为带 bias 的 `1×1 256→32 → GELU → DWConv3×3(groups=32,pad=1) → GELU → 1×1 32→256`，最后 weight/bias zero-init。仅替换该 stage，其他三个 feature tensor 不动。无 norm、attention、teacher、prototype、重建、第二 encoder、多 stage adapter 或 residual cap。

Gate 使用 stage2 global-mean pool 的 256 维 + 当前 observed raw Depth 的四统计：zero ratio、nonzero mean、nonzero population std、两端都有效且非零的水平/垂直邻域 absolute difference mean。输入共 **260 维**，MLP `260→16→1→sigmoid`。geometry mask 只排除合法 crop/pad support；统计 detached、FP32，空集合安全。label、hidden cause、severity、corruption metadata/condition name、未损坏 Depth 均不进入 gate。

实测参数：Proposal **16,992**；Gate **4,193**；总新增 **21,185**。原 C0 reliability auxiliary 参数保留只为完整 checkpoint keys 兼容，全部冻结且不参与 A-v1 推理 gate、损失或 optimizer。

理论上，480×640 输入对应 stage2 约 30×40，proposal 中间为 `[B,32,30,40]`，残差为 `[B,256,30,40]`；Conv 部分约 20,006,400 MAC/image，gate Linear 4,176 MAC/image，未计 pool/统计/GELU/加法。MAC 是乘加次数的结构算术，不是实测速度/显存。未做 latency benchmark、profiling 或全尺寸 batch10 容量验证。

## 2. C0 身份、两阶段与随机状态

真实 source 路径：`cloud/MMFR_E1_Batch1A_local_transfer_20260922/MMFR_E1_Batch1A_local_transfer_20260922/C0/checkpoint/update-2560.pth`。

- 实现前直接计算、加载时复核、两次工程更新后再计算的 SHA-256 均为 `ca618b23d18eabb201a0d11d18da383ac99576d0feae5864e3233bda527d9a1a`。
- 原 C0 完整加载 812 keys，无 missing/unexpected；A-v1 只缺失 10 个新增 `av1.*` parameter keys，无 unexpected。source 内 global_optimizer_step=2560；旧 optimizer/RNG 不恢复。
- Phase 1 仅 proposal；Phase 2 仅 gate。每次配置先关闭所有梯度、清空已有梯度，再开放当前 branch。base 参数、buffers、BN/SyncBN statistics 与模式均冻结。
- 训练输入构建复用现有合法框架，阶段 seed 分别 `2026093001` / `2026093002`，每次构建新 corruption tensor，无缓存/target 继承。同 seed words 的其他字段相同，stage seed 不同；确定性 clean/全空仍可能相同，不把它误报为随机流复用。
- `phase_loss` 先拿到一次 observed input，再复用它计算 endpoints 与 active pass。既有 `capture_rng_state`/`restore_rng_state` 回放同一个 snapshot；off/full no-grad、target detached，learned 仍穿过 frozen HAM 保留到 gate 的梯度。

## 3. 文献补丁与创新边界

原设计 §6.1 已只补以下两项，并停止文献搜索：

- **SpotTune，CVPR 2019**：[原论文](https://arxiv.org/html/1811.08737v1) §3.1 式(2)、§3.2 式(4)、§4.1/Table 2 直接复核。已有 per-instance policy、pretrained/fine-tuned block 选择、Standard Fine-tuning 与 `SpotTune (running fine-tuned blocks)` 对 learned policy 的对照；通过任务分类损失与 Gumbel-Softmax 联合训练策略/适配分支。A-v1 的有限区别是固定单个 Proposal 后，显式监督其 paired off/full action utility，而不是联合选择多块路径。
- **DCRM-ViT，CVPR 2026**：[CVF 正式页](https://openaccess.thecvf.com/content/CVPR2026/html/Khan_Keep_It_Frozen_Domain-Routed_Conditional_Residual_Modulation_for_Multi-Domain_Vision_CVPR_2026_paper.html)与官方摘要直接复核：frozen backbone、input-conditioned router、per-sample low-rank conditional residual modulation，结构高度邻近。摘要描述任务监督的双层优化；**PDF 正文未能读取，全部公式/监督细节待核验**，不把摘要未提及 paired utility 当作全文不存在的证明。

统一边界：dynamic routing、sample-level gating、conditional residual、frozen backbone + adapter 均为已有思想。A-v1 当前唯一值得继续验证的潜在新增点是：**对同一 frozen segmenter、同一输入、同一随机状态下，一个固定 compensation action 的 off/full task-loss difference 进行显式监督，用来训练该 residual action 的 selector。** 当前定向检索尚未发现与该 paired off/full action-utility supervision 高度相同的机制，也未发现已核实会直接阻塞最小实现的高度相同方法；覆盖有限，DCRM/SkipNet 全文限制及 MoSA/DCF 重合仍在，不能确认论文创新性。

## 4. Gate-B：逐项结果

直接证据：mmfr_a_v1_gateb.json（`mmfr_a_v1_gateb.json`），SHA-256 `bf87cc2fc46dd413131f7a6778d8df82388cf7fffa08852126d1f62178be01bf`。主代理直接读取 JSON、复核模块/训练接口/builder 差异与证据身份，完成 runtime 与 gate input 的静态联合验收。

运行环境为本地 RTX 5060 Laptop GPU、PyTorch `2.7.0+cu128`、FP32、TF32 off。B=1、64×64 synthetic RGB/Depth/label，真实 DFormerv2-S、真实 C0 source 和未改 HAM；不读取 dataset/split。总计八次真实 forward、两次 backward/optimizer step，每阶段仅一次，未保存新模型 checkpoint。

1. **Source identity — PASS**：SHA-256 完全匹配；工程更新后 checkpoint 字节不变。
2. **Strict off — PASS**：与独立加载的原 C0 输出 bitwise 相等，最大绝对误差 **0.0**；proposal/gate 各 **0 次调用**，不是计算后乘零。
3. **Zero-init — PASS**：初始化 full 与 off bitwise 相等，最大绝对误差 **0.0**；up weight/bias 全零。
4. **Proposal optimizer membership — PASS**：proposal=1、gate=0、base=0；一次 backward/step 后 up weight/bias 合法更新，全部 active gradients finite，gate/base 无变化且无 gradient。首步 down/DW 的初始任务梯度可为零，weight decay 更新不被解释为已学到补偿。
5. **Gate optimizer membership — PASS**：proposal=0、gate=1、base=0；gate 四个 parameter tensors 均更新，proposal/base 无变化，frozen endpoints 无 gradient。
6. **Positive / negative / ambiguous — PASS**：synthetic utility `[+0.2,-0.2,0,+m,-m]`、m=0.125，mask `[1,1,0,0,0]`，正/负 target 为 1/0；BCE 仅取前两项，值 `0.2231435478`，ambiguous 的 BCE gradient=0；utility/target detached，off/full endpoint 无 gradient。
7. **All-ambiguous — PASS**：synthetic BCE exact 0、CE=`0.3132616580`，总 loss finite、CE gradient 存在。真实 Gate probe 同样 all-ambiguous（u≈`0.0019989014`、m=0.01），BCE=0、learned CE 仍产生 gate 更新；这不表示 gate 已经学会 utility 分类。
8. **Frozen base — PASS**：两阶段前后完整 base 参数与全部 buffers 的逐 tensor SHA-256 一致，包括 BN/SyncBN running mean/var 与 counters；所有 base 子模块 eval，base 不进入 optimizer、无 gradient。
9. **RNG / HAM alignment — PASS**：off/full/learned 三次前向的 torch CPU/CUDA 起始状态相同；进入 NMF 的状态及随机初始 bases SHA-256 均相同，bases identity 为 `c9b34093622c33badccf0ced8dceaf0cebec425e1689f5fd520a6e28b972dbbd`。RGB/normalized Depth/observed raw Depth hashes 相同，observed tensor pointer 相同；label hash/pointer 相同。唯一研究差异为 residual action。
10. **Gate input safety — PASS**：静态检查确认 `gate_value(feature, observed_depth, geometry_mask)` 只拼接 256 pooled features 与 4 observed stats；metadata/clean mask/GT 只在构建或 loss 路径。运行时输入维度=260、全零 Depth 统计 `[1,0,0,0]`、空支持 `[0,0,0,0]`，全部 finite。
11. **Finite loss — PASS**：Proposal 一次 loss=`9.1379003525`；Gate 一次 loss=`10.2857255936`；两阶段 active gradients finite。数值来自 synthetic label，仅是工程证据，不能作分割效果比较。
12. **Phase corruption isolation — PASS**：不同 stage seed words，分别调用原生成器生成两份新 batch，raw-depth pointers 不同，无缓存继承；同一个 Gate batch 再被三行为共享。

Gate-B 专用 m=0.01、lambda_clean=0.1、lr=3e-5、weight_decay=0.01 不进入正式配置或最佳参数结论。Proposal probe 为 p_clean=1 以覆盖 clean KL；Gate probe 为 p_clean=0，以覆盖已有 corruption；正式继承值仍 p_clean=0.25。Gate probe 抽中 `entire_missing` 后 `gaussian_noise`，没有新增 failure。

## 5. Git / 文件范围与实际检查

开始时已有 **13 个 tracked 修改、4 个 untracked 根条目**，包括基础配置、旧实验报告、Canvas、论文工具与索引等；本轮保留这些修改，不归为 A-v1 实现。基准 HEAD 未改变，没有 commit/push。

本轮新增上述 4 个 Python 文件、A-v1 protocol、本报告、Gate-B JSON 与简洁实现差异审核；本轮定点修改 builder、当前设计文献补丁、两份实时入口、报告索引和 MMFR 当前导航/变更历史/审核 metadata。当前上级审核材料由既有 `rebuild_review_packet.ps1` 从 canonical sources 生成，不手工编辑生成目录；旧 E1 protocols、evaluator、corruption 算法、C0 checkpoint 和历史实验结论保持不变。

实际检查：source SHA-256、模块限定 CPU check（off 对象/zero-init/空支持统计/参数量）、独立 config/interface import、一次完整上述 bounded GPU Gate-B、主代理直接证据/代码差异复核、builder 静态诊断和定点 diff whitespace 检查。完整测试套件、全仓测试、正式 training、Quick-Val/Main-Val、official test、multi-seed、latency/profile 均未运行，因为超出本轮工程资格预算与授权。

## 6. 最终状态与待审核内容

**A-v1 implementation complete; Gate-B PASS; ready to request formal Proposal → Gate training authorization。**

请上级审核独立身份、严格 off、冻结、utility mask 与 RNG/HAM 证据；再决定是否授权正式两阶段训练。正式 margin、lambda_clean、训练预算/Proposal 收益停止口径、后续筛选继续线仍未决定；本轮未调这些值，也未根据 val-dev 选参数。Gate-B 后已停止一切模型执行；仅维护交付文档和唯一实时入口，等待上级与用户授权。

## 7. 后续本地收口补充：第一轮合同已冻结（2026-09-30）

本节记录 Gate-B 之后的独立本地准备授权；§1–6 的原始工程事实与当时未冻结参数的状态保持历史，不改写 Gate-B JSON。**本地准备：PASS；正式训练仍未授权。** 大白话：训练用量、参数、筛选线和4090短预检的边界已经写死，但没有启动任何新模型运行。

- 独立 identity 不变；正式 run name 增加 `formal-v1`。Proposal/Gate=**1920/640 successful updates**；m=**0.01 per-image mean CE difference**；lambda_clean=**0.1**（两阶段一致）；AdamW new LR=**3e-5**、WD=**0.01**、既有 bias/norm no-decay。数值为用户预冻结起点，不宣称最优、不给 val-dev 调参。
- 每阶段128次成功更新 warmup、poly0.9，到本阶段结尾归零；batch10/workers8/accumulation1、480×640、AMP fp16 on、TF32 matmul/cuDNN on，继承E1实际训练行为；phase corruption seeds=**2026093001/2026093002**，p_clean=0.25。沿用当前 A-v1 phase接口的 v3 phase-local curriculum，不静默复用E1 continuation remap。
- 保存 `proposal-update-1920.pth` transition/recovery；Gate 从同一 Proposal 权重进入，fixed-final 为 `update-2560.pth`；recovery每640次成功更新，包含完整训练/RNG/data cursor身份。无 val selector、追加epoch或训练内 utility early-stop。
- 唯一未来四条件 Quick-Val：clean/entire_missing@1.0/spatial_dropout@0.75/misalignment@0.75，同一fixed-final的off/full/learned。继续线：learned hard mean 相对 matched off >=+0.50pp、clean >=−0.20pp、hard mean 严格>full；否则按selector-not-supported/stop/inconclusive记录，不宣称显著性。
- 4090先 Proposal最多3次成功更新，正常后才 Gate最多3次；全尺寸/正式batch/AMP/TF32，分阶段回报peak allocated/reserved、每步时间/loss/finite/OOM/GradScaler skip，Gate另报utility和positive/negative/ambiguous。每阶段3次立即停，不能自动接正式训练；预检不计正式预算，不作为正式起点。
- 直接核验：开始时五个关键代码hash全部匹配Gate-B；C0 SHA仍为 `ca618b23d18eabb201a0d11d18da383ac99576d0feae5864e3233bda527d9a1a`；原Gate-B JSON SHA仍为 `bf87cc2fc46dd413131f7a6778d8df82388cf7fffa08852126d1f62178be01bf`。仅正式config发生预期差异，新config SHA=`a868bc11e9aa0b950407ae4e0c5fd47fb6b298b81ba7b4a6e54be66f8da56319`；其他四个代码文件未变，旧 E1 protocol/evaluator/corruption无内容差异。
- 为保留证据与审核包身份，`.gitattributes`对A-v1证据JSON、hash绑定的canonical sources/生成审核包与pre-A-v1历史快照定点设 `-text`，防止Git换行转换导致source/packet hash失配；运行代码沿用Git LF策略，既有复现JSON另记5个Git LF blob SHA，明确区分本地CRLF哈希与云端checkout身份。
- `D:\2Env\anaconda\envs\df2\python.exe -B` 的 config import和合同字段断言PASS，定点config静态诊断无问题；没有创建临时test/验证脚本，没有重跑GPU Gate-B、完整测试、训练、Val、official test或云端。当前只有phase函数接口，full-resolution preflight/formal runner及A-v1评价入口仍未实现，不冒称可直接运行。
- 精确Git范围、完整commit SHA和下一步见 protocol Cloud handoff（`../01_research/mmfr_a_v1_action_utility_protocol.md#6-cloud-handoff4090-全尺寸容量测速边界`）与 当前状态（`../../doc/main/MUSeg-current-status.md`）；本轮只允许一个本地commit，不push。SHA回执在提交后回填，保留为元数据差异，不把未提交执行代码带上云。
