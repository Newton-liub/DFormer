# 05：协议校准与 Development B0 收口

> **已封存：** 本文件属于 2026-08-29 封存的历史计划，仅保留形成时点的设计、执行记录和证据指针；当前计划见 `doc/plans/MUSeg-A2-B2深度有效性/00-总方向规划.md`。

> **文档角色：** 活动阶段计划，不承担实时状态或运行授权。
> **形成或核验时点：** 2026-08-27。
> **实时入口：** `doc/main/MUSeg-current-status.md`；开放选择见 `doc/main/MUSeg-open-decisions.md`。
> **后继关系：** 取代旧 `05-开发长程训练与协议冻结.md` 中尚未执行的后评估、协议冻结和通用筛选前置内容；筛选职责移至新 Stage-07。

## 1. 目标

1. 把颜色顺序、归一化和 validation 几何变成显式、可测试、可记录的协议字段。
2. 修正 post-evaluator，使所有 geometry 在未经 resize 的原始 Label 像素网格计分。
3. 完成 seed 1 五项 val-dev 后评估和曲线/几何审查，冻结 `development-reference-B0` 的证据边界。
4. 用独立 paired calibration 决定后续颜色谱系和 validation geometry，通过 Protocol Gate。

本阶段不自动运行 GPU、五项后评估、训练、云资源或 official test；任何 workload 需代码/计划提交后另行授权。

## 2. 颜色与 normalization 契约

配置、protocol、launcher、run config/manifest 和 evaluator 必须共同记录：

- `channel_order=BGR|RGB`；
- normalization identity；
- 三通道 mean/std 的数组顺序；
- pretrained 路径、大小、SHA-256 和来源语义状态。

production loader 不再根据 dataset/backbone 名称为 MUSeg 猜颜色顺序。Stage-01 的亮度统计仅是 split 分布统计，不证明模型输入通道语义。

已通过上游公开资产闭合权重 provenance：Hugging Face 官方 `bbynku/DFormerv2` 的 `DFormerv2/pretrained/DFormerv2_Small_pretrained.pth` 为 110,203,103 bytes，LFS SHA-256 与本项目权重的 `19116988fc86dc9f3e879282237941e11b9b1b5c480edb51e92807311dbc11a6` 完全一致。官方 README 将其列为 ImageNet-1K RGB-D pretrained；官方 `VCIP-RGBD/RGBD-Pretrain` 数据代码默认用 PIL `RGB` 并按 RGB 顺序使用 ImageNet mean/std，因此 pretrained 上游通道语义判定为 RGB。该结论只闭合初始化来源，不自动决定 MUSeg 后续应采用 RGB；历史 seed 1 的 BGR 微调事实和 paired calibration 要求保持不变。

### 2.1 三臂颜色诊断

固定同一 development checkpoint，仅在 val-dev 上运行：

1. legacy BGR + `[0.485,0.456,0.406]`；
2. RGB + RGB mean/std；
3. BGR + 反向 mean/std。

该三臂只诊断 checkpoint 对通道/统计的敏感性，不决定新 baseline 胜负。真正颜色选择必须建立独立 `color-geometry-screening-B0` protocol，使 legacy BGR 与候选谱系从相同 pretrained、相同 seed、数据顺序、预算、优化器和 evaluator 成对重训。

预注册升级规则：差异明确超过 calibration protocol 的噪声容差时，对领先方案与 legacy BGR 完整跑一个 paired seed；接近容差时两臂一起补第二 paired seed。若最终离开 legacy BGR，现有 seed 1 只保留历史 reference，新谱系重新 qualification 和 B0。

## 3. Validation geometry 契约

所有 geometry 保留原始 Label 和原始计分支持：

- `original-full`：原图输入、原图 logits、原图计分；
- `resize-480x640`：只 resize 彩色/深度输入，模型 logits 用 bilinear、`align_corners=false` 恢复到原图，Label 不 resize；
- `sliding-480x640`：在原图上用 480×640 crop、`2/3` stride 和覆盖计数平均 logits，保持全图输出；仅小于 crop 时允许右/下 padding，并在汇总前去除 padding。

报告必须记录 `input_geometry`、`metric_geometry=original-label-grid`、输入/恢复插值、crop、stride、padding、输入尺寸和输出尺寸。正式协议只允许通过契约测试的 evaluator；`utils/eval.py` 等 legacy 旁路不得用于 MUSeg 正式结论。

## 4. CPU 验收

至少覆盖：

- production `RGBXDataset + ValPre` 与 post-evaluator 的 `original-full` 张量等价；
- BGR/RGB sentinel 和 normalization 记录一致；
- resize 只改变模型输入，logits 回到原始 Label grid；
- sliding 对原图逐像素覆盖且无零 count；
- official-test 身份拒绝；
- 完整 checkpoint 跳过单独 pretrained 初始化并 `strict=True` 加载；
- protocol v3 强制显式 input contract，launcher 与 run manifest 透传一致。

## 5. 五项 val-dev 后评估

代码、测试和 Protocol Gate 前置经用户授权后，按冻结顺序运行：

1. best checkpoint：original-full；
2. best checkpoint：resize-480x640；
3. best checkpoint：sliding-480x640；
4. epoch-500：resize-480x640；
5. epoch-500：sliding-480x640。

每项核对 checkpoint/split SHA、318 样本、input contract、metric grid、per-class 指标和 `official_test_included=false`。original-full 如受 8 GB 本地显存限制，记为 `environment_limit`，不写成模型失败。

## 6. 几何选择顺序

不简单选择最高 mIoU。按以下预注册优先级判断：

1. 在线 seed 1 original-full 能否在同一契约下复现；
2. 所有候选是否使用相同原始像素支持；
3. 是否避免长宽比扭曲；
4. 显存是否在目标环境可控；
5. 结果是否确定、per-class 是否稳定。

`resize-480x640` 作为诊断；主要候选为 `original-full` 与 `sliding-480x640`。几何决定只影响新 protocol，不改写 seed 1 原始曲线、epoch-460 best 或历史报告。

## 7. Protocol Gate 产物

- 修正后的 evaluator 与 CPU 测试；
- pretrained provenance 记录，无法闭合时明确“待核验”；
- 五项后评估 JSON 及风险处置报告；
- `development-reference-B0` 身份与证据指针；
- 冻结的 channel order、normalization、validation geometry 和 evaluator version；
- 后续 A2 与 module screening 使用的新 protocol 模板。

任一核心契约或后评估未闭合时保持 Protocol Gate 未通过，不进入 A2 正式运行或模块训练。
