# MMFR 本地外部代码索引（`D:\0Project\origin`）

> **角色：** MMFR 研究包内的“本地 clone 仓库”只读盘点索引。它只回答“本机有哪些外部论文代码、对应什么、能不能用”，不代替 [`external_reference_provenance_2026-09-21.md`](external_reference_provenance_2026-09-21.md) 里的许可证边界、函数锚点与直接复制判定。
> **日期：** 2026-09-23（创建；分支、HEAD、工作区与许可证状态为当日只读实测）。
> **盘点边界：** 本轮只读枚举目录、README、`.git/config`、`.git/HEAD` 与许可证文件。**未**移动、未重新 clone、未修改源码或 Git remote、未联网补仓库、未运行任何训练或 forward。
> **位置边界：** 这些仓库位于仓库外 `D:\0Project\origin\`，**不纳入 Git**。包内不保存任何仓库副本、vendored 源码或外部 checkpoint。

## 1. 索引总表（11 个 clone 仓库）

| 目录名 | 本地路径 | 对应论文 / 方法 | PR / RE / AI 编号 | Git remote | 分支 / HEAD | 许可证 | 用途 | MMFR 当前是否使用 | 备注 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `DFormer` | `D:\0Project\origin\DFormer` | DFormerv2: Geometry Self-Attention for RGBD Semantic Segmentation | PR070 | `https://github.com/VCIP-RGBD/DFormer.git` | `main` / `814799bb1f39eb380f72fdea1cd591f2cc27b6aa` | MIT，Copyright (c) 2023 VCIP-RGBD | baseline | 是 | 作者官方仓库；用于核对四级特征、Depth 消费路径、GSA 拓扑与论文—代码 pooling/bilinear 差异。工作区 clean |
| `RGBD-MMCD` | `D:\0Project\origin\RGBD-MMCD` | Toward Reliable RGB-D Semantic Segmentation: Handling Missing Modalities via Condition Dropout（ConD） | AI024 | `https://github.com/TobuChiri/RGBD-MMCD.git` | `main` / `edb6a2ebf5cb10262cf0f3b4a0968c98d16ac1be` | 根目录**无** LICENSE；子目录 `DFormer\LICENSE` 与 `Sigma_CD\LICENSE` 均为 MIT | 模块参考 | 是（F 路线允许的参考依据） | 官方实现为“完整 control encoder + frozen 原路径 + zero-init stage 注入”，不能整体视为统一 MIT 代码包。历史上曾被 Git LFS smudge 阻塞，源码 checkout 已恢复 clean，LFS 大权重未取回 |
| `RMMSS` | `D:\0Project\origin\RMMSS` | Towards Robust Multi-Modal Semantic Segmentation with Teacher-Student Framework and Hybrid Prototype Distillation（RobustSeg） | PR090 | `https://github.com/RobustSeg/RMMSS.git` | `master` / `fda61fc603e0d110f96db79c52a531aa604f5fc6` | MIT，Copyright (c) 2025 RobustSeg | 模块参考 | 否（只登记为 T 路线后续设计参考） | teacher/student + prototype-level KL；prototype distillation 与 AI019 的 masked reconstruction 必须分开。工作区 clean |
| `nconv` | `D:\0Project\origin\nconv` | Confidence Propagation through CNNs for Guided Sparse Depth Regression（NConv-CNN） | AI029 | `https://github.com/abdo-eldesokey/nconv.git` | `master` / `d85d4b659f2207b397c62d81f27f363baf3397be` | GPL-3.0 | 模块参考 | 否（DVG-B1 已关闭；仅保留历史只读审计） | 官方主仓库。`NConv2d` 说明“连续 confidence 可传播”，但全 1 confidence 下采样后变 `0.25`、零 padding 使边缘低于 1，不满足 strict all-1 no-op |
| `nconv-nyu` | `D:\0Project\origin\nconv-nyu` | Confidence Propagation through CNNs for Guided Sparse Depth Regression（NYU-Depth-v2 版本） | AI029 | `https://github.com/abdo-eldesokey/nconv-nyu.git` | `master` / `a708c2dbeba9679d8474e493ef2c40e07846cf8e` | 未声明（仓库内未找到 LICENSE 文件） | 模块参考 | 否（同 `nconv`） | 作者 NYU 辅助仓库，fork 自 Sparse-to-Dense 实现。使用前必须单独确认许可与再分发条件 |
| `LightDepth` | `D:\0Project\origin\LightDepth` | LightDepth: A resource efficient depth estimation approach for dealing with ground truth sparsity via curriculum learning | 未映射 | `https://github.com/fatemehkarimii/LightDepth.git` | `main` / `785575ad9baa32e476a3f5df84d00f7e764ef915` | 未声明（仓库内未找到 LICENSE 文件） | 暂未使用 | 否 | 训练标签稀疏度课程（`dataloaders.py::dilation` 反复 MaxPool），不是输入 corruption reliability 传播。工作区 clean |
| `NR-MVSNet` | `D:\0Project\origin\NR-MVSNet` | NR-MVSNet: Learning Multi-View Stereo Based on Normal Consistency and Depth Refinement | 未映射 | `https://github.com/wdkyh/NR-MVSNet.git` | `main` / `0c1e3d14c4ee6a54f5fdbabca6f50687067761c9` | 未声明（仓库内未找到 LICENSE 文件） | 暂未使用 | 否 | `models/refinenet.py::DepthUpdate.forward` 是 query/location-only hard gate，不是 pairwise query-key gate |
| `OPM-MVS` | `D:\0Project\origin\OPM-MVS` | Octagram Propagation Matching for Multi-Scale View Stereopsis | 未映射 | `https://github.com/RayKhuboni/OPM-MVS.git` | `main` / `da9635893f81608a997036d8a48e907e03e70d99` | 未声明（仓库内未找到 LICENSE 文件） | 暂未使用 | 否 | README 声明 official implementation 并绑定论文；多视图 bilateral/geometry confidence |
| `LFDA` | `D:\0Project\origin\LFDA` | LFDA: A Framework for Light Field Depth Estimation With Depth Attention | 未映射 | `https://github.com/syt06007/LFDA.git` | `main` / `f5eee3ce1c91fbbb2d0a2e6dc2064e45792670cf` | MIT，Copyright (c) 2024 Kim HyeongSik | 暂未使用 | 否 | README 声明为 `ieeexplore.ieee.org/document/10508345` 的 official implementation |
| `SMAC` | `D:\0Project\origin\SMAC` | Learning Selective Mutual Attention and Contrast for RGB-D Saliency Detection（SMAC） | 未映射 | `https://github.com/nnizhang/SMAC.git` | `main` / `e40f4579fcbd5da2f4ea3426f93a5d6988676dc4` | 未声明（仓库内未找到 LICENSE 文件） | 暂未使用 | 否 | 只含 ReDWeb-S 数据集、统计与结果，**没有模型代码**；不能作为 SMAC 实现证据 |
| `S2MA` | `D:\0Project\origin\S2MA` | Learning Selective Self-Mutual Attention for RGB-D Saliency Detection（S2MA，CVPR 2020） | 未映射 | `https://github.com/nnizhang/S2MA.git` | `master` / `f94fceede09d644f285c271b1d8d41e384e0f8ed` | 未声明（仓库内未找到 LICENSE 文件） | 暂未使用 | 否 | 作者前作，**不是 SMAC 实现**；只能作为 Full-affinity `[B,HW,HW]` 拓扑的邻近参考 |

`D:\0Project\origin\论文` 是论文全文目录，不是 Git 仓库，不计入仓库数。

## 2. 论文编号 → 本地 clone 链路

| 编号 | 论文 | 本地全文（见 paper-index） | 本地 clone |
| --- | --- | --- | --- |
| PR070 | DFormerv2 | 有（`MMFR-附件\` 与 `DFormer-doc-paper\` 两份独立提取） | `D:\0Project\origin\DFormer` |
| AI024 | ConD / Condition Dropout | 有 | `D:\0Project\origin\RGBD-MMCD` |
| PR090 | RobustSeg | 有（两份独立提取） | `D:\0Project\origin\RMMSS` |
| AI029 | Confidence Propagation through CNNs | 有 | `D:\0Project\origin\nconv`、`D:\0Project\origin\nconv-nyu` |
| 其余已映射编号（AI017、AI019、AI023、MoSA、PR029、PR053、PR089、PR117、PR151、RE049、RE131、RE326） | 有本地全文 | 有 | 无本地 clone（其中 AI019、AI023 已确认没有公开官方仓库） |
| RE042、RE053、RE188、RE447 | 有本地全文 | 有 | 无本地 clone；编号—题名对应未经编号总索引确认 |

## 3. 有本地 clone 但没有编号映射

`LightDepth`、`NR-MVSNet`、`OPM-MVS`、`LFDA`、`SMAC`、`S2MA` 都属于 DVG-B1 时期（已关闭路线）的检索资料，编号总索引中没有与它们逐一对应的 PR/RE/AI 条目。按“无法确认编号就标未映射、不猜编号”的规则，本索引不给它们补编号。

## 4. 盘点方式

- 仓库范围：`D:\0Project\origin` 下含 `.git` 目录的直接子目录，共 11 个。
- 分支与 HEAD：读 `.git/HEAD` 与 `.git/refs/`；remote：读 `.git/config`；工作区状态：`git status --porcelain`（当日全部 clean）。
- 许可证：只查仓库根目录与二级子目录中实际存在的 LICENSE/COPYING 文件并读首行，**未联网**核对 SPDX 或官方声明。
- 论文对应关系：优先用 README 自述（含官方实现声明）与论文正文给出的入口；不能确认的写 `未映射`。

## 5. 使用边界

- 外部仓库只作为实现依据和参考，不整体替换 DFormer backbone、MUSeg 数据管线、corruption protocol、evaluator、baseline 或 Gate 流程。
- 任何移植前必须先在本目录与 [`external_reference_provenance_2026-09-21.md`](external_reference_provenance_2026-09-21.md) 补齐：原始文件/class/function、许可证与再分发条款、MMFR 对应实现、具体改动、实验身份和实验编号。
- 未声明许可证的仓库（`nconv-nyu`、`LightDepth`、`NR-MVSNet`、`OPM-MVS`、`SMAC`、`S2MA`）在确认许可前不得复制其代码进入主工程。
