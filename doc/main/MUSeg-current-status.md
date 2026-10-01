# MUSeg 当前状态与唯一实时入口

> **事实截至：** 2026-10-01 云端无卡模式准备。当前为 **`A-v1 runners implemented; C0/splits verified; lightweight CPU checks passed; GPU preflight/formal/Quick-Val not started`**。用户在本对话中明确改为无显卡、低配 CPU 模式，要求先完成可在此模式下完成的准备；因此本次不启动模型或实验。冻结合同见 [A-v1 protocol](../../MMFR/01_research/mmfr_a_v1_action_utility_protocol.md)，研究选择见 [开放决策](MUSeg-open-decisions.md)。

## 当前阶段与实际意义

A-v1 是 stage2 补偿残差 Proposal 加每图强度选择器 Gate；Quick-Val 是四条件开发集筛选。**大白话：** 所需 C0 权重已找到，训练和评价命令也已实现，但还没有任何 A-v1 全尺寸容量、训练或分数结果。接下来恢复用户指定现有 RTX4090 后，先完成两阶段各3次成功更新预检；均通过才可从原 C0 干净开始正式1920+640，再做唯一四条件筛选。

2026-10-01 原连续执行授权仍定义研究合同；本次用户最新执行要求是**只完成无卡模式工作**。不因为授权字段现在为 true 就在本会话自动启动 GPU。无卡不是 CPU 模型训练替代方案。

## 已直接核验的版本与输入

- 分支 `perf/mmfr-a2-v3-pipeline-opt1` 已 fetch/ff-only 同步；进入本次工作的 HEAD 为 `77738457eeb2b12221545e7db1a84ec95559ae47`，确实包含用户交接基线。远端 origin 为 `https://github.com/Newton-liub/DFormer.git`。本次新增代码的实际提交在 Git 交付栏记录，不用旧基线冒充新入口版本。
- 当前 `torch.cuda.is_available() == False`，Python 使用 torch `2.1.2+cu118`。会话开始时直接见过空闲 RTX4090，之后用户切换无卡模式；**当前 CUDA 不可用**，不把早先设备查询冒充当前可运行资源。
- 配置原本指向仓库内的 C0 位置，该位置不存在。已在仓库外找到并直接计算核验：`/root/rivermind-data/cloud/MMFR_E1_Batch1A_local_transfer_20260922/C0/checkpoint/update-2560.pth`。
- C0 SHA-256 **`ca618b23d18eabb201a0d11d18da383ac99576d0feae5864e3233bda527d9a1a`**，与冻结 source 完全匹配。运行时用 `--source-checkpoint` 显式传入此路径；配置亦支持 `MMFR_AV1_SOURCE_CHECKPOINT`，只改变文件定位，不改变权重身份。
- 用户上传 `/root/rivermind-data/DFormer/MMFR_AV1_cloud_transfer_20261001.zip`；直接查看 ZIP 目录，含 `README.md`、`source-7773845.zip` 和 `C0/checkpoint/update-2560.pth`（321150608 bytes）。因外部现有 C0 已匹配，**未解压、未覆盖源码、未将上传包或权重入 Git**；ZIP 内 C0 内容哈希没有再次计算，不宣称包内字节已验证。
- `train-dev` 1277 / `val-dev` 318：直接核对计数、唯一性、互斥以及冻结文件 SHA。train SHA `a6b15b63f6d5193e3928ea24ada25be403a48e68d1c1f9372cdbbc3fe5cd8470`；val SHA `1d0719d8f64f016d48995c25ab66d4004d76b7155d9efeef7cbb7454c0dd0e83`。数据根 `/root/rivermind-data/dataset/MUSeg_DFormer`，RGB/Depth/Label 目录存在；未扫描整个数据集、未复核所有图片内容。
- official test 继续 **`sealed_unread`**；本次没有打开该划分或测试图片。原 Gate-B JSON 及历史报告保持原样，没有重跑资格实验。

## 必要实现与核验边界

- 新 `tools/mmfr/av1_train.py`：复用 `utils/mmfr_av1_training.py`、原 `RGBXDataset/TrainPre/get_train_loader`、v3 corruption、RNG 和 checkpoint 工具；没有套用旧 E1 训练循环。提供 `--check-inputs`（不构造模型或加载 workers）、`--execute preflight`、`--execute all` 及严格完整状态 `--resume`。
- 每个预检阶段最多3次成功更新，Gate 直接用同一预检 Proposal 实例；通过后销毁预检模型、从原 C0/冻结初始化/训练种子重新开始正式训练。预检计数不计正式预算。
- 正式配置固定 batch10、480×640、workers8、accumulation1、AMP fp16、TF32 on；Proposal1920/Gate640；margin0.01、lambda_clean0.1、AdamW LR3e-5/WD0.01；每阶段新 optimizer/scaler、128-update warmup/poly0.9；无 Val、checkpoint 选择、utility early-stop 或追加预算。
- 每640次保存完整状态，均为128步数据 epoch 的边界；包含 model/optimizer/scaler/scheduler/RNG/阶段计数/下个 data cursor/身份。非持久 workers 于下一 epoch 从保存的主 RNG 重新建立，未把预取 batch 当成恢复状态。跨提交/身份/边界不匹配拒绝恢复；**此恢复路径目前只有代码复核，尚未经过 GPU 中断续训验证，不宣称运行等价已经 PASS**。
- Proposal1920 transition 为 `proposal-update-1920.pth`；Gate640 fixed-final 为 `update-2560.pth`。新评价入口要求该 formal fixed-final 的阶段/计数/来源身份，拒绝预检或其他时点权重。
- 新 `tools/mmfr/av1_quickval.py`：复用 original-full 输入、冻结 corruption、FP32 评价及指标工具；同一输入/配对 HAM 随机状态输出 off/full/learned。首个实际样本中核对 C0 冻结张量及 strict off、原标签网格和 Proposal/Gate 零调用；逐条件核对累计 confusion 等于逐样本之和。分数显示沿用原指标，严格筛选比较使用同一 confusion 算术的未舍入值，避免显示舍入改变严格大小关系。
- 数值调用的最小修复：概率形式 BCE 在 CUDA autocast 中不允许直接调用，`masked_gate_bce` 将**同一 BCE** 运算置于禁用 autocast 的 FP32 区域；不改公式、标签、margin 或系数。Gate-B 历史结果不改写。
- 收口代码复核补齐 GradScaler（AMP 自动缩放器）初始1024、growth2/backoff0.5/interval2000的精确启动门禁；配置值与原合同一致，未改数值。该启动检查补丁随本次修复提交交付。formal-final 的 `--resume` 被刻意拒绝以防重复训练/评价；若训练结束但评价尚未开始，可显式使用独立 evaluator 对同一 fixed-final 执行本次唯一评价；若评价已开始或输出非空，先停止核对，不能新目录静默重跑。
- 已完成的轻量检查：新入口/config 语法与 import、冻结四条件定义、LR 首步/128步/末步坐标、FP32 BCE 等值/有限梯度/全模糊 exact-zero、输入身份检查、Git 差异检查。没有新增测试文件，未运行全套测试、模型 forward、DataLoader workers、GPU、训练、Val 或 benchmark。

## 准确恢复点与停止边界

恢复用户指定现有 RTX4090 后，先读取本文件；确认运行代码已提交、GPU 可用与来源/划分身份，再显式运行（目录必须不存在）：

```bash
cd /root/rivermind-data/DFormer
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python tools/mmfr/av1_train.py \
  --source-checkpoint /root/rivermind-data/cloud/MMFR_E1_Batch1A_local_transfer_20260922/C0/checkpoint/update-2560.pth \
  --output-dir /root/rivermind-data/DFormer/cloud/mmfr-av1-formal-v1 \
  --execute all
```

`all` 依序完成 Proposal3/Gate3预检、原 C0 干净正式1920+640、同 fixed-final 唯一四条件 Quick-Val，然后停止。所有 GPU 运行资格仍待真实预检，不把 CPU 准备检查当成预检通过。

- 数值/身份/显存/基础设施错误立即停止留日志；没有自动重试或降 batch。普通工程修复只重做受影响检查，必须先提交代码；若涉及结构/loss/种子/预算/口径或不能确认恢复兼容性，交人工决定。
- 四条件仅 clean、entire_missing@1.0、spatial_dropout@0.75、misalignment@0.75；318 val-dev、original-full、scale1/no flip、FP32/TF32 off、eval seed2026091401/reset-per-unit。继续线为 learned hard 相对 matched off >=+0.50pp、clean >=−0.20pp、learned hard 严格超过 full；单 seed 筛选不作统计显著性结论。
- **本次成功更新计数：Proposal0/Gate0；预检0/0。A-v1 fixed-final、训练耗时和四条件分数均未产生。** 不进入 Main-Val、official test、新 seed、调参或追加训练。

## Git 与证据交付

- 代码/config 初始提交为 **`857f1da85a5d5a38a61fe1172a5f3c7d832cc652`**（`feat(mmfr): add bounded A-v1 training and matched quickval runners`，5个文件）；包括新入口、最小 AMP 修复、显式授权 config 和上传 ZIP 排除规则。GradScaler 启动合同检查修复为 **`aa53eeea152e909e7b7ca2502823d5b10973c965`**（1文件、3行）。没有运行 A-v1 GPU 实验。两笔代码提交已普通 push 至 origin 同名分支，直接 `ls-remote` 核验远端为 `aa53eeea152e909e7b7ca2502823d5b10973c965`；状态/证据回执另提交并同步，文档不追逐自身 HEAD。不强推、不改历史、不推其他分支。
- 本次准备事实与检查边界写入既有 [复现信息](../../MMFR/02_evidence/reproducibility_current.json)。原 [Gate-B 报告](../../MMFR/02_evidence/report_mmfr_a_v1_implementation_gateb_20260930.md) 仍只代表 synthetic B1/64×64历史资格。
- `MMFR/99_review_packet_current/` 是前次本地交付的生成快照，不是本次云端准备回执。当前环境无 `pwsh`，未安装依赖或手工编辑生成产物；本次最新事实以本文件和既有复现 JSON 为准，后续有正式实验结果再按已有工具重建审核包。
- 开放决策没有新增研究选择，故 `MUSeg-open-decisions.md` 不更改时间或追加运行状态。独立旧路线/论文问题不阻塞本次 A-v1。
