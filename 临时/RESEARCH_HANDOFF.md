# DFormer 新研究阶段交接报告

> 核实日期：2026-10-09。用途：为高级模型设计、审核两套新研究方案提供工程与历史实验背景；本报告不提出新方法，不包含新的性能实验。
> 路径约定：普通相对路径以当前仓库根目录为基准；`归档/` 指 `../DFormer-archive-20261007/`，`数据/` 指 `../dataset/`。历史实验数字来自已打开的最终报告，本轮未重跑；“未核实”不等于存在工程故障。

## 项目现状摘要

- **当前可用骨干是 DFormerv2-S。** 官方 Small 编码器预训练权重已在本地和云端加载成功；现用研究配置是 SUNRGBD、37 类、HAM（Hamburger）分割解码器。DFormer++ 有编码器代码和 NYUv2 配置，当前本地没有对应权重；本地 README 的权重表仍写 `Coming soon`，本轮没有重新核查远端发布状态。
- **旧 MMFR 是 MUSeg 深度失效鲁棒分割项目，尚未得到可沿用的新方法收益。** F-lite 的单视图筛选提升未在十视图主评价中复现；R-OE-lite v2 的提升只有约 0.01 个百分点；A-v1 完成训练后，配对评价未达到预先冻结的继续门槛，已停止。它们不构成“所有适配、补偿或几何校准都无效”的证据。
- **本地数据状态已变化。** MUSeg 整理版已有 3,171 组；SUNRGBD 已有 10,335 组三模态文件；DeLiVER 已有 RGB、原始 Depth、HHA、标签，按官方路径清点为 train 3,983 / val 2,005 / test 1,897。旧状态和下载指南中的“SUNRGBD、DeLiVER 未下载”已过时；云端是否已有这两套新数据未核实。
- **深度输入语义是首轮实验的重要前置条件。** MUSeg 的原始 uint16→uint8 全局映射与标签映射已有代码和元数据依据，但物理单位未知；SUNRGBD 一个训练 Depth 样本实际是 uint16，而当前 loader 按 8 位灰度读取；DeLiVER 官方 `depth` 模态实际读取 HHA，不能直接当作 DFormerv2 的标量距离输入。
- **基础工程准备成功，完整新 baseline 训练尚未验收。** 本机环境版本本轮重新确认；云端环境、CPU 加载、SwanLab 登录、screen 有历史实测。正式启动仍需固定开发验证/test 边界、确认输入数值协议、核实云端数据和同 commit 同步，并取得 GPU/训练授权；本轮没有启动云实例、训练或评价。
- 论文依据由仓库外最新 AI 阅读版另行提供：`../origin/_index/exports/PAPER_LIBRARY_FOR_AI.md`，以及 `_CORE.md`、`_SUPPLEMENTED.md`。本轮未重新扫描或阅读论文库，不裁定新颖性。

## 一、旧研究实际验证了什么

### 1. 目标、数据与共同起点

旧 MMFR 围绕 MUSeg 地下矿山 RGB-D 语义分割，考察深度整幅缺失、局部丢失、噪声、模糊、量化和错位时，能否改善分割并维持 clean 性能。模型是 DFormerv2-S + HAM，15 个前景类；研究评价使用冻结 `val-dev` 318 张，训练使用 `train-dev` 1,277 张，均来自官方 train，official test 1,576 张不用于方法选择。

旧继续训练筛选阶段（E1）的共同起点是旧损坏训练模型 A2 v3 的 `selector-epoch-420.pth`，它已经接受 MUSeg 损坏训练，不是纯官方预训练模型；历史损失包含分割损失和权重 0.1 的 Depth reliability 辅助损失，该辅助输出不直接进入分割推理。

依据：`归档/MMFR/01_research/MMFR_research_blueprint_v4_1_2026-09-20.md`；`归档/MMFR/01_research/e1_batch1_protocol.md` §5–6。

### 2. C0、F-lite、R-OE-lite、A-v1

- **C0：匹配的继续训练对照。** 从 A2 epoch-420 的模型权重重新初始化训练，不恢复原 optimizer（优化器）或随机数生成器（RNG）状态；进行 2,560 次成功更新，没有新增补偿模块，保留原共同损失和损坏训练合同。它用于排除“只因为多训练了一段”的收益，不能冒称从官方 pretrained 重新训练的 clean baseline。
- **F-lite：无条件特征残差适配。** 在编码器三个后续 stage 加 adapter，和 C0 使用相同预算，新增模块不依赖故障标签或动态 gate。C0/F-lite 均完成 2,560 次成功更新、skip=0；adapter 初始化额外消耗 RNG，导致两个 run 的 shuffle 存在已记录且当时接受的差异。
- **R-OE-lite：可观测全空深度的 RGB 几何替代。** 只有合法图像支持域内当前 raw Depth 全为零才触发，利用 RGB 预测替代几何输入；非空样本严格绕过替代器。它识别的是 observable-empty，不是整幅故障的原因。v1 因全分辨率算子显存不足（OOM）中止；v2 将末端运算移到上采样之前，完成 2,560 次更新，替代器确实获得梯度并更新。
- **A-v1：任务效用监督的残差动作选择。** 冻结 C0，在 stage index 2 训练 residual proposal，再固定 proposal 训练每图标量 gate；gate 输入为特征池化和当前 Depth 四个统计量，不是像素可靠性图或故障概率。Proposal/Gate 最终各完成 1,920/640 次更新。训练中的 HAM 内部非负矩阵分解（NMF）发生混合精度故障，经批准采用局部 32 位浮点（FP32）修复并从完整 1,280 更新恢复，最终训练完成；停止原因是验证收益未过线，而非训练未完成。

依据：`归档/MMFR/01_research/e1_batch1_protocol.md`；`归档/MMFR/02_evidence/report_e1_batch1b_roe_v2_formal_training_quickval_20260924.md`；`归档/MMFR/02_evidence/report_mmfr_a_v1_formal_quickval_upper_review_20261001.md`。

### 3. 只保留关键结果

这里 mIoU 是类别平均交并比，均以 % 报告；pp 表示百分点差。

**F-lite Quick-Val：** 318 张，四条件、原图单尺度、无翻转。C0→F-lite：clean 53.46→54.15（+0.69 pp）；整幅缺失 48.76→50.17（+1.41）；局部 dropout 51.40→52.39（+0.99）；错位 52.15→52.63（+0.48）。三个困难条件均值 50.77→51.73（+0.96 pp），当时按筛选规则进入 Main-Val。

**F-lite Main-Val：** 同一 318 张、十条件、五尺度×翻转共十视图。六个单故障的未加权平均主指标 $M_6$ 为 C0 55.2883、F-lite 54.8917，差 −0.3966 pp；clean 56.69→56.02（−0.67 pp）。十条件仅局部 dropout（+0.40）和 dropout+noise（+0.42）为正。四个共同条件中有三个收益符号翻转，支持“单视图优势未在主评价口径复现”。该 Main-Val 没有预注册去留门槛，不能事后补造正式 stop 判定；单 seed 和 shuffle 差异也不支持统计或因果退化结论。两种评价口径的绝对分数不能混比。

**R-OE-lite v2 Quick-Val：** clean 差 0.00 pp；整幅缺失、局部 dropout、错位各约 +0.01 pp；困难均值约 50.77→50.78，按筛选规则为 `inconclusive`，没有证明实用收益。未完成十条件 Main-Val；未做同权重强制 bypass 对照，因此不能把聚合微差解释为替代器的净因果贡献。

**A-v1 最终配对 Quick-Val：** 同一最终 checkpoint、同输入和配对 HAM 随机状态，比较 off/full/learned 三行为。三个困难条件均值分别 50.770364 / 50.766106 / 50.768491；learned−off 为 **−0.001873 pp**，远低于预冻结 +0.50 pp 继续线。clean 的 learned−off 为 −0.010031 pp，满足 clean 容忍线；learned 比 full 高 +0.002385 pp，但二者均未超过 off，因此正式筛选为 **stop**。A-v1 Main-Val 未执行、对应适配也未完成。

依据：F-lite Quick-Val 见 `归档/MMFR/01_research/e1_batch1_protocol.md` §11；Main-Val 见 `归档/doc/reports/2026-09-22-mmfr-e1-batch1a-c0-flite-mainval.md`；R-OE-lite、A-v1 见上一节两份最终报告。

### 4. 直接结论与可能原因的边界

- **实验直接支持：** 上述具体实现、单 seed、冻结预算/输入/评价下，尚未建立可继续投入的稳健收益；A-v1 的既定筛选失败，F-lite 的主评价未复现筛选优势。
- **观察，不是机制证明：** F-lite 的不同评测排序、shuffle 差异和少数类别的大幅变化；R-OE-lite 与基础网络漂移同量级的微差；A-v1 四条件 gate 均值都约 0.54。这些不能单独证明“数据太少”“动作太弱”“gate 塌缩”或“几何校准没有空间”。
- **与几何干预直接相关的旧证据：** Oracle-A 在冻结 A2 上用已知 validity 抑制 Depth bias，strict 相对 original 的 clean 为 −3.48 pp、局部 dropout 为 −1.77 pp；具体 validity-aware suppression 动作按既定规则 NO-GO。它不等于所有 Depth bias 校准都被否定，也不证明可靠性诊断无用。依据：`归档/doc/reports/2026-09-19-museg-mmfr-oracle-a-validity-aware-pairwise-geometry.md`。
- **后续未完成的自然缺失研究应单独记账：** Natural/Grid/Replay 三种自然缺失训练对照的完整正式运行，以及四组模型×318张×3条件的 S1 验收没有有效完成；自然缺失数值修复只通过 1,664 次成功更新资格运行。同步/数值工程中断不是“方法无收益”的实验结论。依据：`归档/doc/main/MUSeg-current-status.md`、`归档/doc/reports/2026-10-04-natural-missing-numerical-repair-qualification.md`。

## 二、当前代码、Backbone 与启动入口

### 1. Git 与工作区

本轮直接读取：分支 `research/dformerpp-clean-start`，HEAD `79f8e81acd74b9f4ba65be9beaf6e3ee32736eac`；作者基线为 `e3273009b759b578945483828ff315d560be94c9`。任务开始时已有 `doc/state/current.md` 修改、一份未跟踪论文补充报告、9 份浏览器快照删除及 `临时/PAPER_LIBRARY_AUTOMATION_FINAL_PLAN.md` 删除；`models/`、`utils/`、`research/`、`local_configs/` 没有相对 HEAD 的改动。本轮只新增本交接报告并更正现有状态中的过时数据事实，不提交或推送，不修改模型/数据/checkpoint。

### 2. DFormerv2-S 与 DFormer++

- 作者 SUNRGBD 配置：`local_configs/SUNRGBD/DFormerv2_S.py`，数据基础配置：`local_configs/_base_/datasets/SUNRGBD.py`。
- 现用独立配置：`local_configs/research/DFormerv2_S_SUNRGBD.py`；默认路径 `datasets/SUNRGBD` 本地不存在，已有数据在 `D:\0Project\dataset\SUNRGBD`，启动前应将 `DFORMER_DATASET_ROOT` 指向其父目录 `D:\0Project\dataset`。这是已有配置支持的路径选择，本轮没有改配置。
- 模型入口：`models/builder.py::EncoderDecoder` → `models/encoders/DFormerv2.py::DFormerv2_S`。四级宽度 64/128/256/512，block 数 3/4/18/4；SUNRGBD 使用 HAM、decoder embed dim 1024、37 类，历史建模参数量 26,966,591。
- 本地现有官方权重：`checkpoints/pretrained/DFormerv2_Small_pretrained.pth`。历史核实大小 110,203,103 bytes，SHA-256 `19116988fc86dc9f3e879282237941e11b9b1b5c480edb51e92807311dbc11a6`；本轮确认文件仍列于该目录，未重复哈希或下载。
- DFormer++ 编码器在 `models/encoders/DFormerPP.py`，builder 支持 T/S/B；已有配置是 `local_configs/NYUDepthv2/DFormerPP_T.py`、`DFormerPP_S.py`、`DFormerPP_B.py`。**现仓库没有启动示例所提到的 SUNRGBD DFormerPP 配置文件**。本地 `checkpoints/` 仅发现 DFormerv2-S 官方 pretrained，没有 PP 权重；README 的 PP pretrained/NYU/SUN 三行仍为 `Coming soon`。不能把示例文件名或 README 推荐语当成可直接运行证据。

加载证据及官方 missing/unexpected keys 的解释：`doc/guides/environment.md`、`doc/guides/research-setup.md`。`extra_norms` 使用标准 LayerNorm 参数初值，不需要补 key；它不是对输入的恒等算子。

### 3. 训练、验证、推理

- 训练：`utils/train.py`；已记录的一卡启动形式为 `PYTHONPATH=. python utils/train.py --config local_configs.research.DFormerv2_S_SUNRGBD --gpus 1 --no-syncbn --pad_SUNRGBD`，本轮不执行。
- 训练内验证由同一入口调用 `utils/val_mm.py::evaluate/evaluate_msf`；独立验证入口为 `utils/eval.py`，通过 `--continue_fpath` 加载分割 checkpoint，SUNRGBD + v2 必须启用 `--pad_SUNRGBD`。
- 推理/保存预测入口为 `utils/infer.py`，配合 `--continue_fpath`、`--save_path`；它使用数据集验证 loader，并硬设 `config.pad=False`，不是已经验收的任意单图部署 API。SUNRGBD 上不能假定它与训练内验证的 padding/评价设置相同。
- `train.sh`、`eval.sh`、`infer.sh` 是作者多卡示例，当前默认分别指向 PP-NYU、v2-NYU、v1-NYU，不是本研究的一卡 SUNRGBD 启动脚本；实际执行时应采用已核实的 Python 入口和明确 config/CLI。

### 4. Geometry Self-Attention 的实际实现

代码事实均在 `models/encoders/DFormerv2.py`：

- 调用链为 `dformerv2.forward` → `BasicLayer.forward` → `RGBD_Block.forward` → `GeoPriorGen.forward` → `Decomposed_GSA.forward` 或 `Full_GSA.forward`。Depth 只取输入第 0 通道，进入几何先验；RGB 提供主要特征，当前没有独立 Depth 特征编码分支。
- `GeoPriorGen.forward` 先将 Depth 双线性对齐到每个 stage 网格；前三 stage 使用 H/W 分解注意力，最后 stage 使用全二维注意力。
- **Depth bias：** `generate_1d_depth_decay` / `generate_depth_decay` 计算深度绝对差乘各 head 的负 `decay`；**Spatial bias：** `generate_1d_decay` / `generate_pos_decay` 计算网格索引距离，二维为 Manhattan 距离，再乘同一组 `decay`。
- `GeoPriorGen.forward` 用可学习 `weight[0]`、`weight[1]` 分别组合空间项和深度项；两种 GSA 在 softmax **之前**将组合 bias 加入 QK logits。这里记录现有实现，不以论文公式替代代码。
- 若以后校准 Depth bias，首要接口是 `GeoPriorGen.forward` 和两种 depth-decay 函数；必须覆盖分解与全二维路径。若需要额外输入，现有传递链还涉及 `dformerv2.forward`、`BasicLayer.forward`、`RGBD_Block.forward`；bias 形状/相加位置变化时才进一步涉及两个 GSA 的 `forward`。输入读法、归一化和网格插值同样需要纳入协议核对，本报告不指定如何校准。

## 三、数据集与实际接口

### 1. MUSeg

**本地资产与 split。** 原始数据在 `数据/MUSeg/`，模型使用 `数据/MUSeg_DFormer/`；后者 `RGB/`、`Label/`、`Depth/`、`Depth16/` 各 3,171 文件，尺寸 1082×932。官方 train/test 为 1,595/1,576；旧开发训练实际采用 `归档/data/splits/MUSeg/dev-v1/train-dev.txt` 1,277 张、`val-dev.txt` 318 张。冻结 manifest 记录按文件名前四段的 location group 隔离，两组开发集合并为官方 train。官方 split 来源为原数据 `Experiment/DatasetSplit.zip`，来源哈希保存在 `数据/MUSeg_DFormer/dataset_meta.json`。

**15 类与实际 ID。** 以下按“原始 ID / 训练 ID：类别”列出；整理过程直接复制 label，不改 ID：

- 1 / 0：person；2 / 1：cable；3 / 2：tube；4 / 3：indicator；5 / 4：metal fixture。
- 6 / 5：container；7 / 6：tools & materials；8 / 7：door；9 / 8：electrical equipment；10 / 9：electronic equipment。
- 11 / 10：mining equipment；12 / 11：anchoring equipment；13 / 12：support equipment；14 / 13：rescue equipment；15 / 14：rail area。

原始 0 是 **background**，不是原始 ignore；旧基线 `gt_transform=True` 对 uint8 标签减 1，将 0 回绕到 **255=ignore**，1–15→0–14。因此背景不参与 15 类损失/计分；标签 resize 使用最近邻。类别语义来自旧数据报告对官方 `Label_ID.pdf` 的核验，本轮未重读 PDF；当前转换代码、数据元数据和减 1 函数已重新核对。

**Depth 数值链。** 原始为单通道 uint16 PNG，原样保留到 `Depth16/`；0 表示无效。固定全局映射为 $D_8=\operatorname{round}(D_{16}\times255/13932)$，生成单通道 uint8 `Depth/`，不按图或 split 单独 min-max。模型读入后复制三通道，统一归一化 $(D_8/255-0.48)/0.28$，v2 只消费第 0 通道。真实图像 raw zero 因而约为 −1.714286；训练 crop/pad 在归一化之后补 exact 0，两者语义不同，不能用 normalized zero 当 raw missing。

**一致性与边界。** 旧冻结训练与评价共用同一全局深度映射、Depth mean/std、RGB 通道约定及标签映射；训练有同步随机尺度/镜像和 480×640 crop，主评价使用原图、多尺度/翻转、原始标签网格，这些是预定用途差异，不要求几何完全相同。Quick-B0/E1 的 RGB 合同已明确为 RGB；当前干净作者 loader 对非 SUNRGBD 默认走 BGR，且现仓库没有 MUSeg 研究 config，因此直接把旧 config/数据接到新入口，不能默认继承旧 RGB 与 evaluator 合同。

**仍未确认：** 原始 Depth 的物理单位，不能把 13,932 擅自写成毫米或米；本轮不是全量逐像素复验。固定量化公式与 raw-zero policy 已确认，不应继续列为完全未知。

依据：`归档/doc/dataset.md`；`归档/doc/reports/2026-08-19-museg-data-reproduction-update.md`；`归档/tools/prepare_museg.py::quantize_depth/copy_sample`；`归档/utils/dataloader/RGBXDataset.py::_gt_transform`；`归档/local_configs/MUSeg/DFormerv2_S_QuickB0.py`；`归档/data/splits/MUSeg/dev-v1/manifest.json`；`归档/MMFR/02_evidence/audit_depth_input_contract.md`。

### 2. SUN RGB-D

- **已在本地落盘。** `数据/SUNRGBD/{RGB,Depth,labels}` 各 10,335 文件；`train.txt` 5,285 行。训练清单实际形如 `RGB/train_0.jpg labels/train_0.png`，已抽查三文件均存在。当前官方 config 预期 train 5,285 / test 5,050、37 类；本轮没有评价 test，也没有逐张验证整个包的内容和来源。
- **现成官方 v2-S 配置可用，数据入口无需新 loader。** `local_configs/SUNRGBD/DFormerv2_S.py` 及现有独立研究配置使用 RGB `.jpg`、Depth `.png`、小写 `labels/`；标签减 1，0→255，1–37→0–36。其路径解析取 split 行中的图像主名，再拼同名 Depth/label。
- **文件位深与模型读法必须分开。** 抽查 `Depth/train_0.png` 的原生读数为 uint16 单通道、530×730、0–52,456；当前 `RGBXDataset` 用 `cv2.IMREAD_GRAYSCALE` 读取 Depth，未启用 `IMREAD_ANYDEPTH`，随后三通道复制与 `/255`、0.48/0.28 归一化。原生数值不是模型直接接收的数值。本地 README 说作者把中间 `.npy` 用 `plt.imsave(..., cmap='Greys_r')` 转 PNG，但这不足以解释当前 uint16 样本的来源/距离编码；**不能宣称已取得可逆米制深度**。
- 原始官方传感器数据及 toolbox 不能直接等同当前整理目录；本地未核验原始传感器编码/物理单位、作者转换来源及当前包的完整数值一致性。按现有实现跑 baseline 与改用 metric Depth 是两个不同输入协议，需在训练前明确。
- **验证/test 边界尚未冻结。** 当前 `eval_source=test.txt`，且训练入口在多个 epoch 自动评价；原样启动会反复使用正式 test 选方法。研究开发需要从官方 train 内预先固定开发训练/验证，正式 test 留作方案与参数冻结后的最终评价；具体划分规则、是否最终重训及使用次数由高级模型决定，本轮不创建 split。SUNRGBD 的 `--pad_SUNRGBD` 会在验证前补至 531×730；训练与验证 Depth 归一化一致，但 padding 顺序不同，应如实保留并固定。

依据：本轮本地清点与训练样本抽查；`local_configs/_base_/datasets/SUNRGBD.py`；`utils/dataloader/RGBXDataset.py::__getitem__/_open_image`；`utils/dataloader/dataloader.py::TrainPre/ValPre`；`README.md` Datasets；`临时/DATASET_DOWNLOAD_GUIDE.md`（下载状态部分已过时）。

### 3. DeLiVER

- **已在本地落盘。** `数据/DELIVER/{img,depth,hha,semantic}` 每模态按 `*/{train,val,test}/*/*.png` 清点为 3,983/2,005/1,897；数量一致不等于已验证全部文件逐一配对。抽查 `cloud/train/MAP_1_point102/110100` 四模态对应文件存在，均为 1042×1042；原始 `depth` 样本为 uint8 单通道、值域 1–255，物理单位/缩放规则未知。
- **官方 RGB-D loader 实际使用 HHA。** 官方 `semseg/datasets/deliver.py::__getitem__` 将 RGB 的 `/img` 替换为 `/hha`、`_rgb` 替换为 `_depth`；`modals=['img','depth']` 中的 `depth` 名称不表示读取 `depth/`。官方 loader 保留彩色 HHA，不等同 DFormer 当前灰度读取后的三复制通道。
- DFormerv2 的第一通道用于深度差计算。直接喂 HHA 会只利用其中一个编码通道；对 HHA 做灰度转换则是编码通道混合，二者均不能冒称标量物理距离。要保证数值意义，必须先查明 `depth/` 原始编码、距离方向、缩放和无效值，再固定 baseline/候选共同转换及归一化；单位不明时只能按“已固定的深度表示”报告，不能写成 metric geometry。
- **当前模型支持 RGB+Depth 两个张量，当前数据接口尚不支持直接读取该数据布局。** 现 `RGBXDataset.get_path` 没有 DeLiVER 分支，通用路径解析会丢失 weather/split/scene 层级，且官方多模态 loader 返回的列表接口不等于 DFormer 的 `data/label/modal_x`。最小适配范围是保留完整相对路径的双模态 loader/路径解析、25 类 config、标签转换、共同图像变换与 Depth 读法；不必引入 LiDAR/event 或改主干。
- 官方 loader 为 25 类、ignore=255；将原 label 255 置 0 后做 uint8 减 1，前景转 0–24，无效/0 转 255。适配时应复用该标签约定，不能仅把 `num_classes` 改为 25。
- 官方评价默认 **val**，test 构造仅为注释替代项；保留官方 train/val/test，val 用于开发，test 不反复调参。天气子项为 `cloud/fog/night/rain/sun`；failure 子项为 `motionblur/overexposure/underexposure/lidarjitter/eventlowres`，按完整路径包含字符串筛选，all-cases 用 `case=None`。后两项不是 Depth 失效标签；不能把它们当成真实深度故障监督。
- 评价尺度、flip、输入 resize、融合和计分必须固定。官方多尺度评价累加 softmax 概率；旧 MUSeg 主 evaluator 平均 pre-softmax logits，二者不能悄然混用。官方公开包曾只发布 front-view，本地目录存在不证明已获得六视角完整数据。

官方实现本轮直接读取：<https://github.com/InSAI-Lab/DELIVER/blob/main/semseg/datasets/deliver.py>、<https://github.com/InSAI-Lab/DELIVER/blob/main/tools/val_mm.py>（可变 main，读取日期 2026-10-09）；本地入口和既有来源记录：`临时/DATASET_DOWNLOAD_GUIDE.md`。本轮没有下载数据或安装官方模型。

### 4. NYUv2

已有 `local_configs/NYUDepthv2/DFormerv2_S.py`、40 类基础配置和官方预训练加载路径；配置使用 795 train / 654 test、480×640。历史上已用该配置成功前向，但本地约定 `数据/NYUDepthv2` 目录当前不存在。作为第二验证集工程上可行，仍需另行数据准备及验证/test 合同；本轮不下载、不做额外准备。

## 四、基础环境与计算资源

- **本地本轮直接确认：** `D:\2Env\anaconda\envs\dformer`，Python 3.10.22、PyTorch 2.7.0+cu128、torch CUDA 12.8、mmcv 1.7.2、timm 1.0.30、SwanLab 0.10.1；CUDA available=True，设备 RTX 5060 Laptop GPU。版本读取没有启动模型训练；旧 `df2` 保持冻结。
- **本地历史实测：** NYUv2/v2-S 做过一次前向，输出 `(1,40,480,640)` finite；SUNRGBD 研究配置构建成功，774 个匹配编码器权重张量与官方文件逐项相等。该小规模加载/前向证据不代表 batch16 训练容量或完整收敛已验证。
- **云端历史实测：** 现有 RTX 4090 实例复用 `/usr/local/miniconda3/envs/py310`，Python 3.10.16、PyTorch 2.1.2+cu118/CUDA 11.8、mmcv 2.1.0、timm 1.0.28、SwanLab 0.10.1；2026-10-08 仅 GPU=0 准备，CPU 成功构建 SUNRGBD 37 类模型并加载同一 pretrained。旧项目曾在该 4090 上完成 MUSeg 训练；新阶段没有 GPU 正式训练验收，本轮不查询或启动云实例。
- **SwanLab 已实测登录/基础生命周期，在线训练记录尚未新验收：** 两端持久登录可被另一进程复用，disabled init/log/finish 正常；`research/tracking.py` 接入训练的 rank0 可选记录，原 TensorBoard 保留。没有新阶段在线训练 run 的上传完成证据。本轮没有读取 API key 或凭据文件。
- **screen 已实测，代码同步流程已配置：** screen 4.09.00 的短命令会话成功；云端现工程 `/root/rivermind-data/DFormer`，旧工程保存在 `/root/rivermind-data/DFormer-archive-20261008`。已建立 Git/bundle 同步方式，但本轮没有云端实时 HEAD/tracked-clean 回读，不能保证与当前 `79f8e81` 同步；下一次训练前要核实，不能沿用旧工程快照。
- **能否直接训练：** 有配置、依赖和权重，但尚不能把“直接启动”作为已验收能力；开发验证协议、云端数据、同版本同步、全尺寸 batch 的训练数值/显存及 GPU 授权仍是启动前置条件。历史 HAM/NMF 的自动混合精度（AMP）故障应作为数值风险记录，不自动给新代码打补丁，也不宣称新 baseline 一定复现故障。

依据：`doc/guides/environment.md`、`doc/guides/research-setup.md`、`doc/guides/cloud.md`；本轮版本读取。云端事实均标为历史，不冒充当前在线状态。

## 五、可复用资产、Baseline 与成本

### 1. 应保留的旧权重与工具

本轮通过旧仓库 Git 未跟踪/忽略文件清单确认以下本地文件仍在；历史 SHA 用于辨认身份，本轮不重新哈希：

- **Quick-B0 clean 开发基线：** `归档/cloud/DFormer-quick-b0-evidence/museg-dformerv2-s-rgb-quick-b0-v1/active-seed/checkpoint/selector-epoch-420.pth`，历史 SHA 前缀 `f246a3af…`，val-dev 十视图 clean 58.79%。它与同名的 A2 epoch-420 是两个不同模型。
- **A2 损坏训练共同起点：** `归档/experiments/MMFR_A2_v3/checkpoints/selector-epoch-420.pth`，SHA 前缀 `2b72eb7f…`。
- **C0/F-lite 固定最终对照：** `归档/cloud/MMFR_E1_Batch1A_local_transfer_20260922/MMFR_E1_Batch1A_local_transfer_20260922/{C0,FLite}/checkpoint/update-2560.pth`，SHA 前缀分别 `ca618b23…` / `ea9319e5…`；保留以便追溯旧结果，不删除无收益候选。
- **R-OE-lite/A-v1：** 最终权重的身份见各自最终报告，A-v1 final SHA 前缀 `87ec54d1…`。A-v1 云端条件性交接 ZIP 有历史核验收据；本轮没有在本地权重清单找到这两组最终文件，也没有回读云端存在性，不能把“报告记载已打包”当成本地可加载。

可按需审核复用而不整体迁回：`归档/tools/prepare_museg.py`、`归档/tools/splits/create_museg_dev_split.py`、冻结 split/manifest；`归档/utils/dataloader/mmfr_training_v3.py`、`multimodal_failure_v3.py` 的同步变换/损坏/RNG 工具；`归档/tools/evaluate_museg_checkpoint.py`、`evaluate_museg_10condition.py` 的原始网格评价；`归档/models/feature_adapter.py`、`roe_substitute.py`、`mmfr_av1.py` 及对应配置可作为旧实现参考。旧工具绑定协议、checkpoint schema 和自定义接口，不保证能无修改导入当前干净作者工程。

### 2. 新主数据集为什么需要重训 baseline

已有 MUSeg 权重绑定矿井数据、15 类头、旧增强/损坏训练和旧评价协议，不能代表 SUNRGBD 37 类或 DeLiVER 25 类的匹配基线。当前已定工程边界是：新 baseline 与候选从同一官方 DFormerv2-S 编码器 pretrained 起步，共用 seed、训练 schedule、增强、输入和 evaluator；不继承旧 MUSeg optimizer、RNG 或训练资格。官方 NYU/SUN 分割权重虽有下载入口，但不是本机已备齐的资产，也不能替代本研究匹配训练；若其训练集合包含开发验证样本，更不能拿它作无泄漏模型选择的起点。

### 3. 当前参数与可靠耗时

- **现研究 SUNRGBD config：** crop 480×480、batch16（分布式时为总 batch）、300 epochs、workers16、seed12345；AdamW lr8e-5、weight decay0.01、warmup10、poly0.9；尺度增强 0.5/0.75/1/1.25/1.5/1.75。按现官方 train 数，331 iteration/epoch、99,300 iteration 是配置算术，**不是已完成训练计数**；开发划分变化时不能直接保留旧 `num_train_imgs`。训练入口见第二节。
- **评价代价相关设置：** SUN config 有五尺度和 flip，CLI 默认启用 `mst`；训练是否评价由 `utils/train.py::is_eval` 决定，实际为 epoch1、每10 epoch及超过200后的各 epoch，不能把 config 的 `eval_iter=25` 当实际验证频率。这也是 test 合同和耗时必须先固定的原因。
- **历史 RTX4090 MUSeg Quick-B0：** batch10、480×640、500 epochs，完整运行约13小时37分40秒；不是新 SUNRGBD 300 epochs 的成本。依据：`归档/doc/reports/2026-08-31-museg-dformerv2-quick-baseline-comprehensive.md`。
- **历史 R-OE-lite v2：** 2,560 次更新用3,590.197秒，约59分50秒；**A-v1 修订续训段**仅剩余640 Proposal+640 Gate，用1,539.012秒，约25分39秒，不能当完整A-v1训练耗时。依据：对应最终报告。
- **历史评价：** F-lite Main-Val 在 RTX5060 Laptop 上 C0/F-lite 的逐条件耗时合计约4.53/4.41小时；A-v1 四条件×318×三行为 Quick-Val 在4090上约607.960秒。仅是各自固定口径实测，不作跨硬件换算。
- **尚无可靠实测：** 新阶段 SUNRGBD/DeLiVER 完整 baseline 训练时间、batch16 峰值显存、数据吞吐、正式验证占比、候选额外成本。本报告不据旧短训或参数量编造新估时。

## 六、真正影响设计或首轮实验的未决项

1. **主数据集与开发/test 合同未冻结：** 本地两套新数据已存在，但 SUN 配置仍以 test 做周期验证；主次顺序、开发划分/场景隔离、最终 test 次数与最终重训范围由高级模型决定。
2. **SUNRGBD 深度来源与实际编码未闭环：** 已抽查原生 uint16，而现 loader 读取 8 位灰度且 README 只说明另一种保存过程；工程子模型核实包来源/数值链，高级模型决定保留官方表示还是采用有明确数值意义的另一协议。
3. **DeLiVER 深度协议需要区分 HHA 与原始 Depth：** 官方 loader 使用 HHA，本地原始样本 uint8 且单位未知；工程子模型核实编码/无效值并实现获批双模态适配，高级模型决定几何主张和官方 RGB-D 比较口径。
4. **MUSeg 物理单位仍未知：** 标签映射、全局量化和冻结 split 已有明确依据，但 uint16 距离单位未核实；工程子模型在实施需要 metric Depth 时取证，不能靠值域猜测。
5. **新训练容量和精度未验收：** 已成功加载且旧训练存在 NMF/AMP 数值失败，但新 batch16 训练尚未实测；工程子模型在获得首轮运行授权后执行最小启动检查，不能把历史故障直接当作新实现失败。
6. **云端新数据与当前代码身份未确认：** 环境、权重和 screen 有历史成功记录，本轮没有云端数据/HEAD 回读；工程子模型在训练前只读核实数据/代码身份，云资源启动由主代理取得用户授权后负责，旧训练授权不沿用。
7. **DFormer++ 不能作为已就绪首轮骨干：** 当前只有代码、NYUv2配置及权重缺口证据；高级模型决定是否作为后续第二骨干，工程子模型届时核实正式权重和缺失配置，不自行预训练或随机初始化替代。
8. **旧筛选收益不能自动迁移到新协议：** F-lite 评价排序翻转、R-OE近零和A-v1 stop 的适用边界已明确；研究价值与新方法是否避开旧失败动作由高级模型判断，不是让工程子模型重新分析全部旧日志。

## 七、本轮范围与交接终点

本轮完成现有文档/源码定点阅读、Git 身份检查、本地数据目录计数及少量训练样本检查、本地环境版本读取，并读取两份 DeLiVER 官方实现；报告中的历史指标和耗时均未重算。未读取 SwanLab key，未下载新数据，未启动云实例/GPU，未做模型训练、性能评价、新模块、数据转换或 checkpoint 修改。

纯文档交付只做内容与差异复核，不运行项目测试或完整测试套件。现有状态仅更正本轮发现的数据落盘事实、已确认的 MUSeg 接口和交接恢复点；没有写入新的科研裁决。本报告完成后停止，下一步设计、数据适配和实验另行授权。

## 八、最后定点补充（2026-10-09）

仅追加输入数值、pretrained先验、旧干预边界与材料版本；前文历史记录不改。以下新数值来自CPU读取三个SUN训练样本及已有pretrained，未做模型前向或分割评价。

### 1. SUNRGBD：8位数值链已核实，来源/物理编码仍未知

- 本地布局/数量符合作者配置；尚无能绑定本地uint16文件与作者下载包的来源回执或样本对照，**与作者整理版的数值一致性未确认**。定点检查的本地 `SUNRGBD/README.md`、`dataset_meta.json` 不存在；作者README仍只有 `.npy` 经 `plt.imsave(..., cmap='Greys_r')` 保存的声明，没有解释这批uint16文件。单位、是否保留原传感器编码未知，不据位深认定包来源。
- `Depth/train_0.png`、`train_1000.png`、`train_5000.png` 原生均为单通道uint16，范围依次0–52456 / 0–40152 / 0–39720；灰度读取均为uint8，范围0–204 / 0–156 / 0–155，三张逐像素满足 $D_8=\lfloor D_{16}/256\rfloor$。现有 `ValPre(pad=True)` 加loader的float32转换后，三张均为 `(3,531,730)`，范围依次−1.714286–1.142857 / −1.714286–0.470588 / −1.714286–0.456583；train_0中心为 **12744→49→−1.028011**。
- 8位读法符合作者当前loader及0–255灰度接口，**不等于已确认数据编码符合作者实验**。实际 $d=(D_8/255-0.48)/0.28$；stage双线性插值后再计算 $|d_i-d_j|$。仅改为16位并保留 `/255`，差值及depth bias量级会大致放大256倍（非严格比例），spatial项不变；改用 `/65535` 也是另一协议，不能自动获得米制意义。验证padding在归一化前补raw0，训练crop/pad在归一化后补0，仍须区分。

证据：三个现有训练PNG的CPU计算；`RGBXDataset.__getitem__`、`TrainPre/ValPre`、`normalize`、`GeoPriorGen.forward`（路径见第二、三节）；作者 `README.md` Datasets及 `figs/application_new_dataset/README.md`。

### 2. DeLiVER：单通道表示已确认，CARLA转换关系未闭环

- 已知本地 `depth/` 样本为uint8单通道1–255，**不是未处理的CARLA三通道RGB打包深度图**；是否先从该格式解码再量化、线性还是对数、逐图归一化还是全局缩放、无效值及单位均未确认。已核查的官方README、loader及HHA参考代码未提供本地PNG的生成公式，保留未知并停止检索，不称为metric depth，也不擅称已确认线性归一化。
- 官方 `depth` 模态实际读取 `hha/`。README所链接的 `Depth2HHA-python/getHHA.py` 编码逆深度、离地高度及法向角度，并取整/截断；参考函数要求米制输入，**不能反推DeLiVER文件已是米制或使用了相同转换器**。对DFormerv2标量差路径，单通道 `depth/` 比直接HHA更匹配接口，但距离先验含义仍待编码确认；HHA灰度混合或任取通道不等于标量距离差。官方HHA RGB-D对照与标量Depth协议必须分列。
- 最小适配仅涉及完整weather/split/scene路径配对、`data/label/modal_x` 输出、25类/ignore255标签、同步空间变换、冻结Depth数值/无效值处理及官方split/case筛选；不新增模态或改backbone，本轮不实施。

证据：[官方README](https://github.com/InSAI-Lab/DELIVER/blob/main/README.md)、[loader](https://github.com/InSAI-Lab/DELIVER/blob/main/semseg/datasets/deliver.py)、[所链接HHA参考转换](https://github.com/charlesCXK/Depth2HHA-python/blob/master/getHHA.py)，读取日期2026-10-09；本地样本依据见第三节。

### 3. pretrained负权重与Oracle-A的适用边界

CPU直接读取现有 `checkpoints/pretrained/DFormerv2_Small_pretrained.pth` 的29个 `layers.*.blocks.*.Geo.weight`；以下为 **min / mean / max**，括号为负值个数，stage index从0开始：

- stage0，3块：spatial 0.852 / 1.256 / 1.727（0）；depth 0.427 / 3.859 / 10.053（0）。
- stage1，4块：spatial 0.673 / 1.381 / 2.029（0）；depth 0.177 / 0.806 / 1.254（0）。
- stage2，18块：spatial −0.042 / 0.388 / 1.377（3）；depth −0.195 / 1.481 / 4.102（2）。
- stage3，4块：spatial 0.095 / 0.334 / 0.740（0）；depth −0.062 / −0.009 / 0.059（3）。

**确有负权重**：`decay<0` 不保证加权bias非正；$w_d\,\mathrm{decay}\,|d_i-d_j|$ 在 $w_d<0$ 时非负，抑制它会移除正bias，而非一律减轻距离惩罚。这是pretrained权重，不是新SUN/DeLiVER分割训练结果。

旧Oracle-A冻结A2-v3 `归档/experiments/MMFR_A2_v3/checkpoints/selector-epoch-420.pth`（SHA256前缀 `2b72eb7f28e5c4b4`，完整值见原报告），仅将各stage `weight[1]*mask_d` 乘pair gate，spatial/QKV/Depth输入/decoder不动。已知validity包含自然raw-zero及注入缺失；invalidity与Depth同步缩放/flip，再双线性对齐stage，padding记为有效。Strict：插值invalidity精确为0时 $r_i=1$，$\rho_{ij}=r_ir_j$；aggregated：$c_i=1-\mathrm{invalidity}_i$，$\rho_{ij}=c_ic_j$；off全部gate为0。

旧验证为MUSeg val-dev318、seed2026091401、五尺度×flip、原标签网格，official test未读。按clean / dropout@0.75 / entire-missing@1.0顺序，original为57.06 / 54.42 / 52.55，strict为53.58 / 52.65 / 52.61，aggregated为53.80 / 52.72 / 52.61，off为52.61 / 52.61 / 52.61（mIoU%）；strict的clean/dropout差−3.48/−1.77pp，dropout仅比off高0.04pp，历史门槛未通过，NO-GO。来源：`归档/doc/reports/2026-09-19-museg-mmfr-oracle-a-validity-aware-pairwise-geometry.md` §3/6/11/13；本轮未重跑。

共同作用点是attention内部depth-derived pair bias；Oracle-A仅测试“已知缺失有效性→固定乘法抑制”、冻结旧损坏训练模型上的推理动作，并未测试学习reliability、非零但错位深度的校准或联合重训。当前候选尚未定式；若只是预测validity后施加同一抑制，动作仍重合，不能假定已避开旧失败。旧报告的伪影利用等原因解释不作为本轮因果结论。

### 4. 开发集入口与最新材料版本

- 训练读取正式 `test.txt` 的链路是研究配置 `C.eval_source` → `get_val_loader` → `RGBXDataset._get_file_names('val')`；“val”命名没有改变文件身份。`is_eval` 在epoch1、每10及>200的epoch自动评价。最小调整点是**独立研究配置**的 `train_source/eval_source` 改为获批的官方train内部开发清单，同步 `num_train_imgs/num_eval_imgs` 并重算已提前派生的 `niters_per_epoch`；只改验证频率不能隔离test。本轮不创建split或改配置。
- 最新 `D:\0Project\origin\_index\exports\PAPER_LIBRARY_FOR_AI.md` 为 **revision18 / schema4 / format1.0.2 / 49篇，实际header时间2026-10-09T14:51:59+08:00**。11份批准草稿与对应条目仅归一Markdown标题层级后，11/11完整包含：LIB1/2/14/24/37/42/45/46/47/48/49。CORE/SUPPLEMENTED同为revision18、15/21篇，header仍为14:43:23；CORE不含非核心LIB47，不能代替全部版。本轮只核身份和已存笔记包含关系，不重新阅读论文正文、不保存索引。

本次追加及状态更正已做内容/差异复核；没有GPU、训练、新性能验证、数据转换、代码改动、测试框架、commit或push。来源/编码未知项继续保留，任务至此停止。

