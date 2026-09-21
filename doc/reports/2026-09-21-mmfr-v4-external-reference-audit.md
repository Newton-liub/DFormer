# MMFR v4.1 外部论文代码核验阶段工作汇报

- 汇报周期：2026-09-21
- 报告对象：MMFR v4.1 F/T/R 方向的外部实现参考与 provenance 固定
- 证据边界：`D:\\0Project\\origin` 下的官方源码 checkout、Git 元数据、LICENSE、关键源码符号、MaskMentor 公开仓库窄搜结果
- 当前状态：外部代码静态核验完成；没有复制外部代码，没有修改 MMFR 主代码，没有提前启动 R/T/G 实现或实验。

## 一、本阶段工作概述

本阶段把三类官方代码参考固定到了具体仓库和 commit，并把论文描述、官方代码行为和当前 MMFR 适配边界分开记录。结果可以支持后续 F/T/R 模块的实现核验，但不能把任何一个外部仓库直接当作当前 DFormer/MUSeg 工程模板。

MaskMentor 的论文入口已确认，但截至本阶段没有确认到公开官方代码仓库，因此没有可固定的 MaskMentor commit，也没有进行 clone 或代码复制。

## 二、主要工作进展

### 2.1 AI024 / RGBD-MMCD：ConD 实现核验

- 状态：已完成并验证。
- 官方仓库：`https://github.com/TobuChiri/RGBD-MMCD`
- 本地路径：`D:\\0Project\\origin\\RGBD-MMCD`
- 版本：branch `main`；commit `edb6a2ebf5cb10262cf0f3b4a0968c98d16ac1be`。
- 许可证：仓库根目录无 `LICENSE`；`DFormer/LICENSE` 是 MIT（Copyright 2024 Zifu Wan），`Sigma_CD/LICENSE` 是 MIT（Copyright 2023 VCIP-RGBD）。
- 关键源码：`DFormer/models/builder.py` 的 `modality_drop`、`zero_module`、`zeroConv`、`EncoderDecoder`；`DFormer/utils/train.py` 的 control-backbone 权重复制与冻结逻辑；`DFormer/utils/dataloader/RGBXDataset.py` 的 RGB/Depth 读入。
- 核验结果：训练时等概率选择 RGB-only、Depth-only 或 RGB-D，并把未选模态置零；同时构造原始 backbone 和 control backbone；control 特征通过四级 zero-init `1×1` Conv 后注入原始特征；原始 encoder 与 decoder 冻结，训练 control 分支和 zeroConv。
- 论文—代码边界：这是完整 encoder-copy 方案。当前 F-lite 是独立 stage-output adapter，不能写成 ConD faithful reproduction。
- checkout 限制：原始 LFS smudge 曾阻塞，已用 skip-smudge reset/clean 恢复源码；大权重没有恢复。本阶段只据此做静态审计。

### 2.2 PR090 / RMMSS：teacher-student 与 prototype distillation 核验

- 状态：已完成并验证。
- 官方仓库：`https://github.com/RobustSeg/RMMSS`
- 本地路径：`D:\\0Project\\origin\\RMMSS`
- 版本：branch `master`；commit `fda61fc603e0d110f96db79c52a531aa604f5fc6`。
- 许可证：根目录 `LICENSE` 是 MIT（Copyright 2025 RobustSeg）。
- 关键源码：`tools/train_teacher.py::main`；`tools/train_deliver_mad50_proto_cur.py::main`；`mmkd/Prototype.py::PrototypeSegmentation`；`mmkd/PCUMD.py::PCUMD`；`mmkd/IFV.py::CriterionIFV::forward`；以及 PCUMD 的 random/zero-padding 变体。
- 核验结果：teacher 入口是普通 segmentation loss；student 路径使用 frozen/eval teacher 和 `torch.no_grad()` 生成参考 logits/多尺度特征；基础总损失为 segmentation CE + logits KL 乘 `50` + PCUMD prototype loss 乘 `100`；IFV/HPD 变体额外加入 IFV loss 乘 `12`。
- 静态疑点：`global_prototypes` 通过 `.data` 做 EMA 更新且不在 optimizer 中；batch prototype 先求均值又按 count 除一次。该疑点只登记为复用前必须重新核对的实现事实，不作论文结论。
- 论文—代码边界：prototype-level KL distillation 与 MaskMentor 的 masked reconstruction/self-teaching 不是同一机制，不能混写。

### 2.3 PR070 / DFormer：DFormerv2 几何先验核验

- 状态：已完成并验证。
- 官方仓库：`https://github.com/VCIP-RGBD/DFormer`
- 本地路径：`D:\\0Project\\origin\\DFormer`
- 版本：branch `main`；commit `814799bb1f39eb380f72fdea1cd591f2cc27b6aa`。
- 许可证：根目录 `LICENSE` 是 MIT（Copyright 2023 VCIP-RGBD）。
- 关键源码：`models/encoders/DFormerv2.py::GeoPriorGen.forward`、`generate_depth_decay`、`RGBD_Block.forward`、`dformerv2.forward`；`models/builder.py::EncoderDecoder.encode_decode`。
- 核验结果：Depth 在 `GeoPriorGen` 中按 `bilinear, align_corners=False` resize；Depth decay 使用成对绝对深度差；前三阶段使用 H/W decomposed GSA，末阶段使用 Full GSA；backbone 只取 Depth 的第一个通道并输出四级 stage features；decoder 恢复使用 bilinear。
- 论文—代码差异：论文文字中的 pooling 描述与官方代码 bilinear 实现不一致。当前 MMFR 以 checkpoint 对应的代码事实为准，并把该差异作为上游论文—代码差异保留。

### 2.4 AI019 / MaskMentor：官方代码窄搜

- 状态：已完成但尚未获得官方代码版本。
- 论文：*MaskMentor: Unlocking the Potential of Masked Self-Teaching for Missing Modality RGB-D Semantic Segmentation*，DOI `10.1145/3664647.3681698`，ACM Multimedia 2024。
- 候选仓库：`https://github.com/Zhao-ZD/MaskMentor`。
- 直接核验：`git ls-remote` 返回 `Repository not found`；GitHub API 的仓库名搜索 `MaskMentor` 返回 `total_count=0`；作者候选账号 `https://github.com/zzzzzd0825` 的公开仓库中没有对应项目。
- 结论：尚未确认公开官方 GitHub 仓库、commit、tag 或 LICENSE。没有创建 `D:\\0Project\\origin\\MaskMentor`，没有 clone，也没有复制代码。

## 三、Provenance 与当前工程边界

本阶段已回填：`liu-test-exp/MMFR/MMFR_v4_1_blueprint_and_reference_package_2026-09-20/external_reference_provenance_2026-09-21.md`。记录包括论文编号、论文题目、官方 URL、本地路径、branch/commit、许可证、原始文件/class/function、论文—代码差异、MMFR 对应位置、直接复制状态和当前授权。

当前 F-lite 的对应实现仍是 `models/feature_adapter.py::FeatureLiteResidualAdapter` / `FeatureLiteAdapter`；它是 MMFR 独立实现，不是从 RGBD-MMCD 直接复制。R-OE-lite 仍只有设计冻结记录，R/T/G 没有因本轮审计获得实现或训练授权。

## 四、验证结果与未运行事项

本阶段完成了 Git URL、branch、HEAD、默认远端分支、工作区、LICENSE、关键源码符号和 MaskMentor 公开入口核验。验证方式以静态代码阅读、Git 元数据检查、许可证阅读和公开仓库查询为主。

按当前研究预算，本阶段未运行项目测试、模型 forward、训练、Quick-Val、Main-Val、云任务或 official test。没有创建临时测试文件，也没有生成模型效果结论。official test 继续保持 `sealed_unread`。

## 五、当前问题与风险

1. RGBD-MMCD 根目录没有统一 LICENSE；未来若参考其子目录以外的文件，必须重新确认对应许可，不能默认整个仓库均可复制。
2. RGBD-MMCD 的大权重没有恢复，任何依赖权重的运行行为仍未核验。
3. MaskMentor 的官方代码入口仍缺失；当前只能使用论文级机制描述，不能声称已经完成官方实现复现。
4. DFormerv2 pooling 与官方 bilinear 代码差异必须继续在后续报告中分开写，不能把论文文字自动当作当前 checkpoint 的实际路径。
5. RMMSS 的 prototype 实现包含需要重新设计或重新实现的静态疑点；即使许可证允许，也不应直接复制到 MMFR。

## 六、下一步计划

1. 保持当前 Batch 1A C0/F-lite Gate-B PASS 状态，等待正式训练单独授权，不因外部参考审计提前启动训练。
2. 若后续实现 F/R/T/G 模块，先在 provenance ledger 中补齐“原论文—官方代码—MMFR 改动—实验身份”执行单，再做最小化独立重实现。
3. 若出现 MaskMentor 官方仓库、项目页代码入口或可验证作者 commit，单独追加版本核验；在此之前不把近似实现写成 MaskMentor 官方复现。
