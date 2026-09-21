# MMFR v4.1 外部论文代码 provenance ledger

> 日期：2026-09-21
>
> 目的：在任何外部机制移植或主工程代码修改前，固定论文、官方入口、外部仓库、本地路径、版本、用途和不可越界边界。
>
> 当前范围：只读查找、clone、版本固定、许可证核验和静态代码审计。本轮没有把外部代码复制到 DFormer/MMFR，没有运行训练、模型 forward、评估或 official test。

## 1. 统一边界

- 外部仓库统一放在 `D:\\0Project\\origin`。
- 外部仓库只作为实现依据和参考资料；不整体替换 DFormer backbone、MUSeg 数据管线、corruption protocol、evaluator、baseline、checkpoint 逻辑或 v4 Gate 流程。
- 任何后续移植必须先在本 ledger 补齐：原始文件/class/function、许可证、MMFR 对应实现、具体改动、实验身份和实验编号。
- 论文描述、官方代码实际行为和 MMFR 适配设计必须分开记录；发现不一致时不得静默选择。

## 2. 已登记的优先参考

### 2.1 AI024 — Condition Dropout / ConD

- 论文：*Toward Reliable RGB-D Semantic Segmentation: Handling Missing Modalities via Condition Dropout*。
- 论文身份：AI024；`arXiv:2607.20326v1`；2026；本轮按 arXiv v1 记录，正式会议/期刊状态仍未单独核验。
- 官方仓库：`https://github.com/TobuChiri/RGBD-MMCD`
- 本地路径：`D:\\0Project\\origin\\RGBD-MMCD`
- 版本：branch `main`，`HEAD=edb6a2ebf5cb10262cf0f3b4a0968c98d16ac1be`，默认远端分支 `origin/main`。
- 工作区：源码 checkout 在清理 Serena 临时元数据后 clean。原 checkout 曾被 Git LFS smudge 长时间阻塞；已用 `GIT_LFS_SKIP_SMUDGE=1 git reset --hard HEAD` 与 `git clean -fd` 恢复源码。大权重没有通过 LFS checkout，不影响本轮源码静态审计。
- 许可证：仓库根目录没有 `LICENSE`；`DFormer/LICENSE` 为 MIT，Copyright 2024 Zifu Wan；`Sigma_CD/LICENSE` 为 MIT，Copyright 2023 VCIP-RGBD。根目录无许可证意味着不能把整个 RGBD-MMCD 仓库默认为统一 MIT 代码包。
- 计划用途：F 路线，尤其是冻结原 encoder/decoder、复制 encoder、zero-init `1×1` 注入、完整/RGB-missing/Depth-missing condition 组织方式的工程参考。
- 已核验的原始实现：
  - `DFormer/models/builder.py::modality_drop`：按等概率选择 RGB-only、Depth-only、RGB-D，并把未选模态置零；使用 NumPy 抽样和强制 `.cuda()`。
  - `DFormer/models/builder.py::zero_module` 与 `zeroConv`：四个零初始化 `1×1` Conv，通道为 `64/128/256/512`。
  - `DFormer/models/builder.py::EncoderDecoder`：同时构造 `backbone` 与 `backbone_control`；control 特征经过 `zeroConv` 后逐 stage 加回原始特征。
  - `DFormer/utils/train.py`：把 `backbone` 权重复制到 `backbone_control`，冻结原始 backbone 与 decoder，仅让 control 分支和 zeroConv 参与训练。
  - `DFormer/utils/dataloader/RGBXDataset.py::RGBXDataset.__getitem__`：OpenCV BGR 转 RGB；灰度 Depth 复制为三通道。
- 论文—代码边界：官方实现是完整 encoder-copy 加 frozen original encoder/decoder；当前 MMFR F-lite 是独立的 stage-output `1×1 down → GELU → zero-init 1×1 up` adapter，不能写成 ConD faithful reproduction。
- MMFR 对应实现：`models/feature_adapter.py::FeatureLiteResidualAdapter`、`FeatureLiteAdapter` 及 `models/builder.py` 的 F-lite 接线；本轮没有改动这些文件。
- 直接复制：No。
- 当前授权：只读审计；F-lite 已通过当前 v4.1 Batch 1A Gate-B，但本轮不启动正式训练。

### 2.2 PR090 — RobustSeg

- 论文：*Towards Robust Multi-Modal Semantic Segmentation with Teacher-Student Framework and Hybrid Prototype Distillation*。
- 论文身份：PR090；CVPR 2026 正式版；作者记录为 Jiaqi Tan、Xu Zheng、Yang Liu；本轮不补写未核验的 DOI 或预印本身份。
- 官方仓库：`https://github.com/RobustSeg/RMMSS`
- 本地路径：`D:\\0Project\\origin\\RMMSS`
- 版本：branch `master`，`HEAD=fda61fc603e0d110f96db79c52a531aa604f5fc6`，默认远端分支 `origin/master`。
- 工作区：源码 checkout 在清理 Serena 临时元数据后 clean。
- 许可证：根目录 `LICENSE` 为 MIT，Copyright 2025 RobustSeg。
- 计划用途：T 路线，审计 full-modality teacher、受损 student、cross-modal prototype distillation、HPD 训练/评估和部署保留边界。
- 已核验的原始实现：
  - `README.md`：分别提供 full-modality teacher、cross-modal prototype distillation 和 Hybrid Prototype Distillation（HPD）的训练入口；feedback straight training 明确尚未释放。环境主张 PyTorch 2.5.1、CUDA 12.1。
  - `tools/train_teacher.py::main`：full-modality teacher 走普通 segmentation loss/CE 训练，不含 prototype distillation。
  - `tools/train_deliver_mad50_proto_cur.py::main`：student 读取占位字符串 `TEACHER_PATH` 的 checkpoint，切到 `eval()`；teacher 在 `torch.no_grad()` 下生成 logits 与多尺度特征。总损失为 segmentation CE + logits KL 乘 `50` + `PCUMD` prototype loss 乘 `100`。
  - `mmkd/PCUMD.py::PCUMD`：按样本、四级特征和模态 index 计算 student/teacher 的类别 prototype，再对 prototype 的 softmax 分布计算 KL；`PCUMD_random.py`、`PCUMD_zero_padding_random.py` 与 `PCUMD_zero_padding_all.py` 的 feature index、随机重排和 padding 语义并不相同。
  - `mmkd/Prototype.py::PrototypeSegmentation`：`global_prototypes` 是 CUDA tensor，通过 `.data` 以 EMA 方式更新，不在 optimizer 参数中；`calculate_batch_prototypes` 先对类别特征求均值，又除以类别 count，属于需要在后续复用前单独记录和复核的静态实现疑点。
  - `tools/train_deliver_mad50_proto_cur_ifv.py::main` 与 `mmkd/IFV.py::CriterionIFV::forward`：IFV 变体额外加入 IFV loss 乘 `12`。
- 论文—代码边界：本仓库的 prototype distillation 是 teacher/student feature-prototype KL 路线；不得把它写成 MaskMentor 的 masked reconstruction/self-teaching，也不得把 README 的训练入口或预训练权重当成本地 MMFR 实验结果。
- MMFR 对应实现：无；当前只保留 T 路线后续设计的参考锚点。
- 直接复制：No。
- 当前授权：只读审计；未实现、未训练、未评价。

### 2.3 AI019 — MaskMentor

- 论文：*MaskMentor: Unlocking the Potential of Masked Self-Teaching for Missing Modality RGB-D Semantic Segmentation*。
- 论文身份：AI019；DOI `10.1145/3664647.3681698`；ACM Multimedia 2024 论文入口已由 ACM DOI/论文页面核验。
- 候选官方仓库：曾核查 `https://github.com/Zhao-ZD/MaskMentor`。
- 公开代码状态：截至 2026-09-21 未能确认公开官方 GitHub 仓库。`git ls-remote https://github.com/Zhao-ZD/MaskMentor.git` 返回 `Repository not found`；GitHub API 的仓库名搜索 `MaskMentor` 返回 `total_count=0`。作者候选账号 `https://github.com/zzzzzd0825` 的公开身份为 Zhida Zhao，但仅有一个无关公开仓库 `zzzzzd0825/mulit-layer-data`，没有 MaskMentor 代码入口。
- 本地路径：无；未创建 `D:\\0Project\\origin\\MaskMentor`，没有可固定的 commit、tag 或 LICENSE。
- 计划用途：T 路线，审计 masked self-teaching、teacher/student 输入、遮罩/缺失生成、重建与蒸馏位置、teacher 冻结边界。
- 可借鉴：论文页面与公开文本支持“完整模态指导缺失模态”和 masked self-teaching / image modeling 的概念方向；原始代码实现尚无可确认入口。
- 不可照搬：不得把 masking/reconstruction 直接改称当前候选的输出 KD；不得与 PR090 的 hybrid prototype distillation 混写；不得把论文中的数据、loss 权重、teacher 或训练预算直接移到 MMFR。
- MMFR 对应实现：无；当前只保留论文级设计参考。
- 直接复制：No。
- 当前授权：只读窄搜；在确认官方仓库和 commit 前不 clone、不移植、不训练。

### 2.4 AI023 — GeomPrompt / GeomPrompt-Recovery

- 论文：*GeomPrompt: Geometric Prompt Learning for RGB-D Semantic Segmentation Under Missing and Degraded Depth*。
- 论文身份：AI023；`arXiv:2604.11585`；项目页记录为 CVPR 2026 URVIS Workshop 接收，非 CVPR 主会。
- 官方项目页：`https://geomprompt.github.io`
- 官方代码仓库：截至本登记时没有可确认的官方 GitHub 仓库和 commit；项目页可访问不等于官方代码已公开。
- 本地路径：无；不 clone、不把项目页写成官方代码仓库。
- 计划用途：R 路线概念溯源，区分 RGB-only task-driven prompt 与 RGB+degraded-Depth recovery。
- 可借鉴：任务驱动的 geometry-like substitute、bounded residual、zero-init identity 起点以及 clean/degraded 分层评价。
- 不可照搬：不得直接采用论文的 `0–255`/`127.5` 量纲、Recovery depth-condition encoder、loss 权重、ViT/decoder 结构或训练预算；当前 R-OE-lite 只能写成 `GeomPrompt-inspired task-driven observable-empty substitute`，不能写成 faithful reproduction。
- MMFR 对应实现：R-OE-lite 设计记录 `../01_research/r_oe_lite_design.md`；尚未实现。
- 直接复制：No。
- 当前授权：只读设计参考；R-OE-lite implementation-not-authorized。

### 2.5 PR070 — DFormerv2

- 论文：*DFormerv2: Geometry Self-Attention for RGBD Semantic Segmentation*。
- 论文身份：PR070；CVPR 2025；DOI `10.1109/CVPR52734.2025.01802`（沿用项目蓝图登记）。
- 官方仓库：`https://github.com/VCIP-RGBD/DFormer`
- 本地路径：`D:\\0Project\\origin\\DFormer`
- 版本：branch `main`，`HEAD=814799bb1f39eb380f72fdea1cd591f2cc27b6aa`，默认远端分支 `origin/main`。
- 工作区：源码 checkout 在清理 Serena 临时元数据后 clean。
- 许可证：根目录 `LICENSE` 为 MIT，Copyright 2023 VCIP-RGBD。
- 计划用途：R/F/T/G 的共同骨干、输入契约、四级特征、Depth 唯一消费路径、Geometry prior 与可插入点核验。
- 已核验的原始实现：
  - `models/encoders/DFormerv2.py::GeoPriorGen.forward`：Depth 使用 `bilinear, align_corners=False` resize；`generate_depth_decay` 构造成对绝对深度差，并按 head decay 参数缩放。
  - `models/encoders/DFormerv2.py::RGBD_Block.forward`：geometry prior 输入 GSA；前三阶段使用 H/W decomposed GSA，末阶段使用 Full GSA。
  - `models/encoders/DFormerv2.py::dformerv2.forward`：RGB 进入 patch embedding，Depth 只取第一个通道，输出四级 stage features。
  - `models/builder.py::EncoderDecoder.encode_decode`：decoder 输出恢复使用 bilinear。
- 论文—代码差异：论文文字中的 pooling 描述与官方代码的 bilinear 实现不一致；该差异属于作者论文—代码上游差异，不是 MMFR 主工程改写。当前 MMFR 应以与 checkpoint 对应的官方代码/本地实现事实为准，并单独记录论文文字。
- MMFR 对应实现：当前主工程 `models/encoders/DFormerv2.py` 与 `models/builder.py`；本轮只读核验，未修改。
- 直接复制：No。
- 当前授权：只读审计；不得因外部代码核验而替换主工程 backbone、evaluator 或 Gate 流程。

### 2.6 MoSA

- 论文：*Modality-Aware Spatially-Adaptive Adaptation for RGB-X Semantic Segmentation*（题名—DOI 仍待进一步核验）。
- 论文身份：MoSA；附件记录 DOI `10.1109/ACCESS.2026.3694496`；作者、正式版本和会议/期刊元数据未完全核验。
- 官方代码仓库：本轮尚未确认与论文绑定的官方仓库、分支或 commit。
- 本地路径：无；在官方来源和论文对应性确认前不 clone、不移植。
- 计划用途：F/G 的条件空间调制和质量相关适配的邻近参考；当前 F-lite 不采用 spatial modulation/RGCF。
- 当前 MMFR 状态：非当前阻塞项，待后续单独核验。

## 3. 迁移前强制说明模板

在任何外部机制进入 MMFR 主工程前，必须补写以下字段并经主代理复核：

- 原论文解决什么问题：
- 官方实现具体怎么解决：
- 当前 MMFR 为什么需要它：
- 计划保留、删除、修改和插入的位置：
- 路线身份：F / R / T / G：
- 原始文件、class、function、config 或 loss：
- 外部仓库 commit/tag/release：
- LICENSE、copyright、attribution 和 redistribution 条款：
- MMFR 对应文件、class、function：
- 是否直接复制：Yes / No：
- 具体改动和不兼容点：
- 实验身份/编号：
- 当前授权：只读 / 实现 / 训练 / 评价：

## 4. 本轮审计结论与边界

- RGBD-MMCD、DFormer、RMMSS 的 branch、HEAD、默认远端分支、工作区状态、许可证和关键代码锚点已固定。
- RGBD-MMCD 可用于源码静态审计，但 LFS 大权重没有恢复；本轮没有执行依赖权重的 forward 或实验。
- ConD 的官方实现证实“完整 control encoder + frozen original path + zero-init stage injection”，但不改变当前 F-lite 的轻量 stage-output adapter 身份。
- RobustSeg 的官方实现证实“teacher/student + prototype-level KL + 可选 IFV/HPD”路线；其 prototype distillation 与 MaskMentor 的 masked reconstruction/self-teaching 必须分开。
- MaskMentor 的论文入口已核验，但公开官方仓库和 commit 仍未确认，因此没有外部代码可供直接审计或复制。
- 本轮没有修改 DFormer/MMFR 主代码，没有复制外部代码，没有运行训练、模型 forward、Quick-Val、Main-Val、云任务或 official test。当前 v4.1 Gate-B、正式训练授权边界和 official test `sealed_unread` 均保持不变。
