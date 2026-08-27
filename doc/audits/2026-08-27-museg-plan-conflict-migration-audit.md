# MUSeg 活动计划冲突收口与协议契约迁移审计

> **文档角色：** 日期化迁移审计，不承担实时状态或运行授权。
> **形成或核验时点：** 2026-08-27。
> **实时入口：** `doc/main/MUSeg-current-status.md`；开放选择见 `doc/main/MUSeg-open-decisions.md`。
> **后继关系：** 本文记录本次计划/代码迁移的静态事实；后续实验、门禁和发布状态以实时入口为准。

## 1. 审计范围

本次审计覆盖：

- `doc/plans/MUSeg阶段二长程Baseline与MVE/00` 与旧 05–11；
- MUSeg production loader、训练 protocol/launcher/run metadata；
- `tools/evaluate_museg_checkpoint.py` 的颜色与计分几何；
- `utils/val_mm.py` 的 legacy sliding 接口；
- 相关 CPU 契约测试、导航、开放决策和实时恢复点。

本次没有运行 GPU、训练、五项后评估、云实例或 official test，也没有修改原始实验结果和 checkpoint。

## 2. 已确认冲突及处置

### 2.1 计划状态与阶段职责

旧 Stage-07/08 曾在计划叙述中容易被读成已具备正式 A2 证据，但五项后评估、geometry 冻结和 A2 正式运行尚未完成。旧 05–11 又把通用 screening、A2 工具、B2 规格/实现/短训和正式三 seed 分散在七份活动文件，导致门禁和恢复点重复。

处置：历史 01–04 保留；旧 05–11 未执行部分删除，由新 05–08 接管。`00` 只保留描述性依赖和 Protocol/MVE Evidence/Gate E/Gate F 四类活动门禁，不复制实时进度。

### 2.2 B0 与 seed 身份

旧材料混用了 seed 1 development checkpoint、screening 对照和正式 B0；单 checkpoint A2 路线又与“多 model seeds 硬条件”并存。

处置：固定 `development-reference-B0`、`color-geometry-screening-B0`、`module-screening-B0`、`formal-B0` 四种身份，禁止跨 protocol 复用。A2 至少允许单 checkpoint 形成工程判定，但必须声明 model-seed 不确定性；正式性能结论仍要求 Gate E 后 B0/variant paired 三 seed。

### 2.3 A2/B2 门禁命名

旧 A2-G、A2-N-G、Gate G、G0、G1、G2 混合了研究判断、用户授权、实现资格和效果筛选。

处置：A2 统一为 `A2-pass/stop/inconclusive`，B2 分为 `B2-zero-train`、`B2-short`、`B2-screening`；`A2-pass + 用户批准` 才通过 MVE Evidence Gate。每项在新 Stage-06 中说明聚合单位、方向一致、数值容差、阈值与中间态。

### 2.4 无候选与正式矩阵

旧 Stage-05 允许“无候选”，旧 Stage-06 的语义却隐含至少一个模块。

处置：Gate E 明确允许 `B0-only` 或 `B0 + 最终组合`。前者可形成正式 baseline，但不能形成模块改进结论。

### 2.5 B2 初始化与短训

旧 Stage-11 同时允许 checkpoint 微调和从 pretrained 开始，无法保证 B0/B2 对称；B2 又可能绕过专用 MVE 门禁进入通用 screening。

处置：B2 short 使用独立 protocol，两臂必须同初始化；通过后只获得进入 Stage-07 统一 screening 的资格。任何初始化、预算或数据子集变化都产生新 protocol。

## 3. 颜色契约审计

### 已直接核验

- MUSeg 历史 loader 通过 OpenCV 保留 BGR；配置 mean/std 为 `[0.485,0.456,0.406]` / `[0.229,0.224,0.225]`，按数组位置应用。
- 现有 seed 1 pretrained 文件身份已记录为 110,203,103 bytes、SHA-256 `19116988fc86dc9f3e879282237941e11b9b1b5c480edb51e92807311dbc11a6`。
- 仓库内引用只证明文件名、路径、大小与 SHA，没有发现把该 SHA 绑定到上游发布资产及训练通道语义的元数据。

因此具体 pretrained 的来源通道语义仍为“待核验”；不能由权重数值推断。

### 已实施

- 配置新增显式 `channel_order` 和 `normalization_identity`；MUSeg 为 legacy BGR，SUNRGBD DFormerv2 显式保持历史 RGB。
- `RGBXDataset` 要求调用方提供 BGR/RGB，不再在内部按 dataset/backbone 名称猜测。
- protocol v3 新增 `input_contract`；materializer、launcher、run manifest、run config、preflight 和训练 CLI 记录并核对该契约。
- v2 历史 protocol 保持可读取；launcher 对其记录 `legacy-v2-museg-default`，避免把补录字段伪装成原始 manifest 事实。
- post-evaluator 支持显式 BGR/RGB 与 mean/std/normalization identity，供三臂诊断使用。

这次改动没有改变当前 baseline 的数值行为；颜色谱系仍待 provenance 与 paired calibration 决定。

## 4. Geometry 契约审计

### 已直接核验

- 训练随机 crop 为 480×640。
- production `ValPre` 在 `pad=false` 时不 resize/crop，`sliding=false` 为原图整图前向。
- 旧 post-evaluator 的 resize 分支同时 resize Label，因而与 original/sliding 不在同一原始像素支持计分。
- legacy `utils/val_mm.py::evaluate` 在 sliding 分支把 list 传给期望 tensor 的 `slide_inference`，接口不一致；小图分支还会 resize 输入并改变输出 geometry。

### 已实施

- resize 只改变彩色/深度模型输入；Label 保持原图，logits 以 PyTorch bilinear、`align_corners=false` 恢复到原始 grid。
- 报告 schema 升级为 `museg-checkpoint-post-evaluation-v2`，记录 input/metric geometry、插值、crop、stride、padding和实际输出尺寸。
- sliding 使用完整 count map 检查逐像素覆盖。
- legacy sliding 改为 tensor/tensor 接口，小图保持原 geometry；MUSeg 正式结论仍只允许测试覆盖的 post-evaluator。

## 5. 静态验证状态

聚焦测试覆盖 production/post-evaluator BGR/RGB sentinel、original-full 张量等价、resize 原图计分、sliding 覆盖、official-test 拒绝、strict checkpoint load、protocol v3 input contract 和 launcher/run manifest 透传。

本审计最终验收时，聚焦测试为 `31 passed`，完整 `tests/` 为 `118 passed`，`python -m compileall -q tools utils local_configs tests` 通过；测试只出现既有 GradScaler deprecation 与本机 pytest cache/临时目录权限 warnings。JSON/Markdown/残留引用、whitespace、敏感/大文件与 Git 边界检查见实时状态和本次最终答复；这些静态结果不构成 GPU 或实验协议授权。

## 6. 文件迁移映射

旧 `05-开发长程训练与协议冻结.md` 的 seed 1 收口、颜色/几何冻结迁入新 Stage-05，模块筛选迁入新 Stage-07。

旧 `06-正式三种子Baseline与Test解封.md` 迁入新 Stage-08，并增加 `B0-only` 合法矩阵。

旧 Stage-07/08 的 A2 工具、矩阵和判据，旧 Stage-09/10/11 的 B2 规格、实现、zero-train、short 和 screening 迁入新 Stage-06；通过后的统一模块筛选与 Gate E 回接新 Stage-07。

旧文件从工作树删除；其历史内容仍可由 Git 历史取回，不复制为新的“当前状态”。

## 7. 当前已验证边界与恢复点

已完成代码和计划迁移，且聚焦 CPU 测试通过。尚未执行 GPU、五项后评估、三臂诊断、paired calibration、A2/B2、module screening、正式三 seed 或 official test。

准确恢复顺序是：完成全量静态/CPU 验收与文档引用清理；形成干净、可复核提交并由用户确认后，再单独授权五项 val-dev 后评估。颜色与 geometry 只有在对应证据完成后才能通过 Protocol Gate。
