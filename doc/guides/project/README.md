# DFormer 项目指南

> 本文件是面向新成员、协作者和后续代理的稳定项目指南，不承担 MUSeg 实时状态职责。
>
> - 当前事实、正在进行事项、阻塞项和恢复点只看 [`doc/main/MUSeg-current-status.md`](../../main/MUSeg-current-status.md)。
> - 研究口径及其处置状态只看 [`doc/main/MUSeg-open-decisions.md`](../../main/MUSeg-open-decisions.md)。
> - 历史计划、日期化报告、审计和 Canvas 只说明各自形成时点，不能替代上述入口。
> - 全部 Git 跟踪文件的逐文件说明见 [`file-catalog.md`](file-catalog.md)。

## 1. 项目定位

本仓库同时包含两部分：

1. **上游 DFormer/DFormerv2 论文代码**：RGB-D 语义分割模型、NYUv2/SUNRGBD 配置、通用训练/评估/推理代码，以及内嵌的 MMSegmentation 兼容实现。
2. **本地 MUSeg 扩展**：MUSeg 数据转换、冻结 development split、审计 protocol、preflight、单 seed/多 seed 编排、checkpoint/恢复、独立裁决、本地后评估、测试与证据文档。

DFormer 用统一 RGB-D 编码器逐阶段融合彩色图像与深度特征；DFormerv2 由深度和空间位置生成 geometry prior（几何先验），再通过 Geometry Self-Attention 增强视觉特征。模型入口由 `models/builder.py::EncoderDecoder` 根据配置选择编码器和解码器。

根目录 `train.sh`、`eval.sh`、`infer.sh` 是固定 NYUv2、多 GPU 和示例 checkpoint 的**上游示例**，不是 MUSeg 当前审计入口。MUSeg 运行从物化 protocol 开始，经过 `tools/preflight_train.py` 后由 `tools/train_museg_4090.sh`、`tools/run_museg_3seed.py` 和 `tools/run_museg_seed.py` 调用 `utils/train.py`。历史 `tools/mve/run_museg_20epoch_screen.sh` 和 `local_configs/MUSeg/DFormerv2_S_20Epoch.py` 不属于当前 development 协议。

## 2. 推荐阅读顺序

- **第一次接手项目**：本文件 → 当前状态 → 开放决策 → `doc/dataset.md` → `file-catalog.md`。
- **只运行上游模型**：根 `README.md` 的上游说明 → `local_configs/NYUDepthv2/` 或 `local_configs/SUNRGBD/` → 根脚本 → `utils/`。
- **处理 MUSeg 数据**：`doc/dataset.md` → `tools/prepare_museg.py` → `tools/splits/` → `data/splits/MUSeg/dev-v1/`。
- **修改模型**：`models/builder.py` → `models/encoders/` → `models/decoders/` → `models/losses/` → 对应配置和测试。
- **启动 MUSeg 训练**：当前状态与开放决策 → protocol 模板 → `tools/materialize_museg_protocol.py` → `tools/preflight_train.py` → MUSeg 启动器。启动前还要核对当前门禁，不得仅依赖本指南。
- **运行本地后评估**：当前状态 → `tools/evaluate_museg_checkpoint.py` → `tests/test_museg_checkpoint_posteval.py`；validation 几何选择仍以开放决策为准。
- **检查实验报告和证据**：当前状态 → `doc/reports/report-index.json` → 对应日期化报告 → 报告列出的结构化证据位置。
- **恢复中断任务**：只从当前状态文件的“当前正在进行”“下一次更新条件”和权威链路恢复，不从历史计划中的“下一步”恢复。

## 3. 顶层目录和根文件

- `.agents/`：项目级 Agent Skill；当前包含 CompShare 命令行工具说明，不代表云实例当前状态。
- `.cursor/`：强制规则和研究进度报告 Skill。规则决定 MUSeg 状态维护及低风险委派边界。
- `.serena/`：代码符号分析工具的项目配置。
- `data/`：Git 跟踪的冻结 split 与审计产物；不是本地完整数据集。
- `doc/`：当前状态、开放决策、稳定指南、历史计划、报告、审计、论文和 Canvas。
- `figs/`：上游 README 图片和新数据集应用示例资源。
- `liu-test-exp/`：用户思路设计与历史实验草案；不是当前状态来源，本项目维护任务不得改写。
- `local_configs/`：Python 配置模块，定义数据路径、模型、优化器、训练与评估参数。
- `mmseg/`：内嵌的 MMSegmentation 兼容/上游实现和 `.mim` 配置工具集；本地解码器主要直接使用 `mmseg.ops.resize`。
- `models/`：本仓库主要的 DFormer/DFormerv2 编码器、解码器、损失和模型装配。
- `protocols/`：可移植审计协议模板；机器路径物化后的 manifest 放在被忽略的 `protocols/generated/` 或仓库外运行位置。
- `tests/`：MUSeg 门禁、协议、split、checkpoint、裁决、训练操作和损失回归测试。
- `tools/`：数据准备、split、protocol、preflight、训练编排、后评估、裁决、MVE 工具和 Canvas 发布工具。
- `utils/`：通用训练、评估、推理、dataloader、checkpoint、engine、指标、变换、跟踪和性能工具。
- 根 `README.md`：项目总入口和上游说明；`LICENSE` 规定非商业使用；`requirements-monitoring.txt` 固定 SwanLab；`.gitignore` 和 `.gitattributes` 管理本地产物、split 例外和行尾。

## 4. MUSeg 数据流

```text
../dataset/MUSeg（官方原始数据，只读）
  └─ tools/prepare_museg.py
       └─ ../dataset/MUSeg_DFormer
            ├─ RGB/*.jpg
            ├─ Depth/*.png       固定全局公式生成的 8-bit 输入
            ├─ Depth16/*.png     原始 16-bit 深度保留
            ├─ Label/*.png       类别 ID 标签
            ├─ train.txt / test.txt
            └─ dataset_meta.json
                 └─ tools/splits/create_museg_dev_split.py
                      └─ data/splits/MUSeg/dev-v1/
                           ├─ train-dev.txt
                           ├─ val-dev.txt
                           ├─ official-test.txt（封存身份，不供开发选择）
                           ├─ manifest.json
                           └─ audit-report.json
```

`tools/prepare_museg.py` 用 `round(depth16 * 255 / 13932)` 统一量化深度，保留无效深度 `0`，并在临时目录完成全量验证后替换输出。标签原始 `0` 是背景、`1–15` 是前景；`utils/dataloader/RGBXDataset.py::RGBXDataset._gt_transform` 对 uint8 标签减一，使背景回绕为 ignore `255`，前景映射为训练 ID `0–14`。

冻结 split 的 `manifest.json` 记录来源、生成器、样本/组关系、统计、哈希和警告；`audit-report.json` 独立复核 schema、成员关系、隔离、计数和哈希。development 配置消费 `train-dev.txt` 与 `val-dev.txt`。`official-test.txt` 保留官方 test 的冻结字节身份，但 development 训练、checkpoint 选择、validation 几何选择、A2/B2 开发均不得消费其内容。

`utils/dataloader/dataloader.py::get_train_loader` 和 `get_val_loader` 将配置路径传给 `RGBXDataset`。当前 MUSeg loader 通过 OpenCV 保留彩色张量为 **BGR**；“RGB/”只是彩色模态目录名。训练预处理在 `TrainPre` 中执行随机尺度、镜像、裁剪和归一化；validation 预处理与滑窗/整图行为由评估入口控制。训练 crop 480×640 不能概括成统一 validation 输入尺寸。

## 5. 模型架构

### 5.1 装配入口

`models/builder.py::EncoderDecoder` 根据 `C.backbone` 选择 DFormer 或 DFormerv2 变体，并记录各阶段通道；根据 `C.decoder` 选择 MLP、HAM、UPerNet、DeepLabV3+、NL 或 FCN 头。HAM、UPerNet、DeepLabV3+ 和 NL 可配置 auxiliary head；训练时主损失与辅助损失都通过 `models/losses/safe_masked_loss.py::safe_masked_mean` 排除背景/ignore 像素。

`EncoderDecoder.encode_decode` 将 RGB 与 depth 送入 backbone，解码多尺度特征，并将 logits 双线性插值回输入空间。构造器在 criterion 非空时加载 `C.pretrained_model` 并初始化头；完整 checkpoint 后评估以 `criterion=None` 跳过单独预训练初始化，再严格加载完整 state dict。

### 5.2 编码器

- `models/encoders/DFormer.py`：`DFormer` 由 patch embedding、分层 block、RGB-D attention 和 MLP 组成，暴露 Tiny/Small/Base/Large 工厂。
- `models/encoders/DFormerv2.py`：`GeoPriorGen` 根据深度差与位置距离生成几何衰减；`Decomposed_GSA`/`Full_GSA` 使用几何先验；`RGBD_Block` 与 `BasicLayer` 组成 `dformerv2`，暴露 S/B/L 工厂。

### 5.3 解码器与损失

`models/decoders/` 提供通用 decode head、MLP/LMLP、HAM、UPerNet、DeepLabV3+、non-local 和 FCN auxiliary head。部分头通过 `mmseg.ops.resize` 复用兼容实现。`models/losses/` 提供交叉熵、Dice、Focal、Lovász、Tversky、accuracy 与安全 masked reduction；实际使用哪一种由模型构造和配置决定，文件存在不表示当前 MUSeg 已使用全部损失。

## 6. 配置与 protocol

**配置**是可 import 的 Python 模块，决定运行时对象参数：数据根、split 路径、图像格式、模型、loss 相关字段、优化器、epoch、batch、worker、评估和输出目录。`local_configs/MUSeg/DFormerv2_S_MVE.py` 提供 MUSeg 模型/数据基础，`DFormerv2_S_4090.py` 将其绑定到冻结 development split 和可由环境变量覆盖的机器路径。`DFormerv2_S_20Epoch.py` 是旧云路径/official split 入口，不是当前 development 协议。

**protocol**是运行身份与审计合同，不替代配置。模板声明 protocol ID、schedule、phase、模型、配置模块、seed、必需 Git commit、split authority、预训练文件身份、输出根、训练参数和 SwanLab 模式。`tools/materialize_museg_protocol.py` 把模板占位符替换为机器绝对路径和实际哈希；物化 manifest 不提交到 Git。`tools/museg_protocol.py::load_protocol` 严格检查字段、schema、冻结 manifest/audit、split 身份和 phase 允许消费的角色。

`experiment_phase` 表示 research phase；`run_kind` 表示启动器运行种类。历史非 probe 记录可能使用 `qualification`，未来 development 长程运行使用 `standard`；历史证据不回写。protocol 的 required commit、manifest SHA、pretrained SHA 和 seed 共同约束运行身份。

## 7. 训练、恢复与证据

1. `tools/preflight_train.py::audit_protocol` 检查依赖、Git、配置 import、split authority、phase、预训练、输出目录和 SwanLab；可选检查数据样本。
2. `tools/train_museg_4090.sh` 先运行 preflight，再调用 `tools/run_museg_3seed.py`。后者只编排协议声明且被选择的 seeds。
3. `tools/run_museg_seed.py` 验证 seed、phase、override、resume 和空输出目录，构造 `utils/train.py` 命令并捕获 launcher 日志。
4. `utils/train.py` 构造 dataloader、`EncoderDecoder`、AdamW、`WarmUpPolyLR`、AMP scaler、SwanLab/TensorBoard 跟踪和 checkpoint protocol；训练循环分别记录 loop attempts、完成与跳过的 optimizer steps。
5. `utils/training_checkpoint.py` 创建、原子写入、检查和恢复 checkpoint，绑定 phase、run ID、Git、seed、模型、优化器、schedule、split 和进度。恢复时还原模型、优化器、scaler、epoch、全局 step、best mIoU 和 RNG 状态，并拒绝身份不一致。
6. `run_museg_seed.py` 在子进程结束后核验 `training_result.json`、checkpoint 路径和 SHA，再写 `command.json`、`environment.json`、`train.exit_code` 和 `run_manifest.json`。

训练完成判定不能只看进程退出或文件存在；应同时核对结构化结果、epoch/validation 点、checkpoint 身份、日志、裁决和 official-test 标志。云端所有终态都应先同步必要证据并复核哈希，再停止实例；acceptance 通过与否不能决定是否继续计费。具体当前恢复步骤仍以状态文件为准。

## 8. 评估与裁决

- **在线 validation**：`utils/train.py` 按配置的开始 epoch 和间隔调用 `utils.val_mm.evaluate`；best checkpoint 采用严格大于策略，平局保留更早的已评估 epoch。
- **上游独立评估/推理**：`utils/eval.py` 和 `utils/infer.py` 由根示例脚本驱动，默认不是 MUSeg 审计入口。
- **本地 checkpoint 后评估**：`tools/evaluate_museg_checkpoint.py` 只接受非空、唯一且身份不暗示 official test 的 split；支持 `original-full`、`resize-480x640` 和 `sliding-480x640`。它保持 BGR、严格加载完整 checkpoint、累计 confusion matrix，并输出含 checkpoint/split SHA、几何、样本数、指标和 official-test 标志的 JSON。
- **独立裁决**：`tools/adjudicate_museg_seed_acceptance.py` 从原始报告和不可变证据重新核对哈希、训练完成、验证点、best/final 身份与 official-test 边界，生成独立裁决文件；不得改写原始 `acceptance.json`、`failed.json` 或 `training_result.json`。

原分辨率整图、固定 resize 和 sliding 的研究取舍仍见开放决策文件。后评估可以为未来协议提供证据，但不能回写已完成运行的原始曲线或 best 身份。

## 9. 测试体系与证据边界

`tests/` 按职责覆盖：

- split 生成/冻结和 protocol 物化；
- Stage-03/04 preflight、probe、qualification 与训练编排；
- checkpoint schema、保存、恢复和严格后评估加载；
- optimizer/AMP step 与安全 masked loss；
- A2 mask 与敏感性工具；
- seed acceptance 独立裁决。

测试可证明纯函数、schema、CLI 约束、synthetic 输入、mock 调用和回归行为符合预期。测试通过不能单独证明真实数据正确、GPU 显存足够、完整训练成功、SwanLab 可用、checkpoint 指标可信或云端生命周期正确；这些需要运行证据、结构化输出和独立核验。

## 10. 文档治理

- `doc/main/`：唯一当前状态与开放研究决策。
- `doc/guides/`：稳定、可重复使用的项目/云操作说明，不维护实验实时状态。
- `doc/guides/project/`：本指南和全文件目录。
- `doc/guides/cloud/`：CompShare、File Browser/OpenList 等外部服务操作说明；执行前核对版本和验证日期。
- `doc/plans/`：阶段门禁与历史执行设计；“当前”“下一步”按文件形成时点解释。
- `doc/reports/`：日期化正式报告、交接快照和 `report-index.json`；索引是目录元数据，不覆盖当前状态。
- `doc/audits/`：特定时点的一致性或边界审计。
- `doc/canvases/`：可视化汇报源；已发布版本只读，`old/` 为归档。

新稳定知识写指南；实际进度写当前状态；真正开放且会影响口径的选择写开放决策；详细过程和证据写日期化报告；执行设计写计划；可视化从已核实 Markdown 事实生成 Canvas。

## 11. 云端与证据边界

CompShare 只提供 GPU 实例、文件传输和持久存储能力。项目的稳定操作说明位于 `doc/guides/cloud/` 和 `.agents/skills/compshare-cli/`，它们不证明实例当前存在或正在运行。数据、checkpoint、不可变归档、日志和凭据不进入 Git；取回证据后按清单重新计算大小和 SHA-256，再进行实例终态操作。

不得在文档中复制 Cookie、Token、私密 URL 或机器凭据。仓库外路径可以用于说明类别和恢复位置，但只有直接复核后才能写成当前事实。外部平台界面、价格和 CLI 会变化，云指南必须按其验证日期使用。

## 12. 常见误用

- 直接运行根 `train.sh` 并把结果当作 MUSeg 审计训练。
- 用 official test 选择 epoch、checkpoint、seed、validation 几何或模型结构。
- 从历史计划/交接的“下一步”恢复，而不先读当前状态。
- 把单 seed development 结果写成三 seed 正式 baseline。
- 把 loader 的 BGR 张量身份写成 RGB。
- 把 480×640 训练 crop 写成统一 validation 输入。
- 为了让验收通过而改写原始 acceptance、failed、日志或 training result。
- 提交数据、checkpoint、归档、日志、物化 protocol 或凭据。
- 绕过 protocol/preflight，或在非 qualification phase 使用 batch/output override。
- 根据文件存在、单元测试通过或进程退出码推断数据和实验已经完整验证。

## 13. 修改项目时先看哪里

- **模型/编码器**：`models/builder.py`、`models/encoders/`、目标配置、模型/训练测试。
- **解码器或 loss**：`models/decoders/`、`models/losses/`、`models/builder.py`、masked-loss 测试。
- **新增数据集**：`figs/application_new_dataset/README.md`、`utils/dataloader/`、`local_configs/_base_/datasets/` 和数据配置。
- **MUSeg 数据处理**：`doc/dataset.md`、`tools/prepare_museg.py`、`tools/splits/`、split 测试。
- **训练/恢复**：`utils/train.py`、`utils/training_checkpoint.py`、`utils/engine/`、`tools/run_museg_seed.py` 和 checkpoint/ops 测试。
- **评估**：`utils/val_mm.py`、`utils/eval.py`、`tools/evaluate_museg_checkpoint.py` 和后评估测试。
- **protocol/门禁**：模板、`tools/museg_protocol.py`、物化器、preflight、Stage-03/04 测试。
- **当前状态/中断恢复**：`doc/main/MUSeg-current-status.md`。
- **研究口径**：`doc/main/MUSeg-open-decisions.md`。
- **历史证据**：`doc/reports/report-index.json` 和对应报告；必要时继续到报告列出的原始证据。
- **云端操作**：`doc/guides/cloud/` 与 CompShare Skill，执行前复核外部状态。
- **新增测试**：在 `tests/` 找同类边界，并直接引用被测公开符号或 CLI。
- **编写报告**：`.cursor/skills/research-progress-report/`、`PROJECT_EVIDENCE.md` 和 `doc/reports/report-index.json`。
