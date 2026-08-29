# MUSeg 当前状态与唯一入口

> 状态时间：2026-08-29 20:35 UTC
> 当前阶段：旧“阶段二长程Baseline与MVE”计划已封存；新的“DFormerv2快速Baseline”总方向规划已建立，尚未进入新协议细化或运行授权
> 本文件是 MUSeg 当前状态的唯一入口；其他阶段计划、审计和正式报告按各自日期保留为历史证据。

## 1. 当前结论

- Stage-01 至 Stage-03 已完成，Gate B 已签署。
- 原 `doc/plans/MUSeg阶段二长程Baseline与MVE/` 已于 2026-08-29 移入 `doc/plans/archive/2026-08-MUSeg阶段二长程Baseline与MVE/` 封存；其中 Stage-01 至 Stage-05、MVE、Protocol Gate 设计和历史执行证据保留为历史资料，不再作为当前运行授权依据。新的当前计划总方向见 `doc/plans/MUSeg-DFormerv2快速Baseline/00-总方向规划.md`，后续细节另行规划。
- 颜色/归一化契约已显式进入配置、protocol v3、materializer、launcher、run manifest/config、preflight、production loader 和 post-evaluator；历史 v2 protocol 仍可读取并标记补录来源。当前 seed 1 仍是 legacy BGR 历史事实，未来颜色谱系已重新打开。上游 provenance 已由官方 Hugging Face 资产闭合：`DFormerv2_Small_pretrained.pth` 的 LFS SHA-256 和大小与本项目权重完全一致；官方 RGB-D ImageNet 预训练代码使用 PIL RGB 与 RGB 顺序 ImageNet mean/std，因此 pretrained 通道语义为 RGB。该来源事实不自动裁决 MUSeg 后续谱系，仍需 paired calibration。
- 本轮 CPU 预检已完成：聚焦契约/切分测试 41 passed，`tests/` 全量 CPU 测试 118 passed，`python -m compileall -q tools utils local_configs tests` 通过，7 个 Git 跟踪 JSON 可解析，val-dev 为 318 条唯一记录且与 official-test 无交集；默认无参数 `pytest` 仍会收集历史 `cloud/museg-epoch10-a2-20260821/code` 和 `utils/engine/dist_test.py` 并因缺失历史路径/导入路径失败，该收集问题不涉及当前 `tests/` 契约结果。两个 checkpoint、val-dev split 和完整归档 SHA-256 已独立复核。
- Stage-04 的 batch 选择、3-epoch qualification、checkpoint 连续/恢复等价性和 Gate D 已完成；batch 10 已用于开发长程训练。Stage-04 当前没有未完成执行项；376/384 次有效 update 的旧遥测差异只保留为历史限制，不要求重跑。
- Stage-05 seed 1（seed `772961337`）已完成 500/500 个 epoch，训练进程退出码为 0；云端运行绑定提交 `56a7ed711df2252e6228fc777d7cb92eb2510ef6` 和 protocol `museg-development-long500-v2`。直接核验该提交确认：基线已使用历史 MVE 的 B1 `safe_masked_mean` 数值稳定性修复；B2 depth-validity mask/gating 尚未实现，未进入本次基线。A1 已证实全背景或空有效像素时 masked loss 的空集合归约数值风险；B1 可直接作为新 DFormerv2-MUSeg baseline 的必要稳定性修复，但不是性能提升模块，不应单独宣称带来 mIoU 提升。
- Stage-05 seed 1 五项 val-dev 后评估已完成：best checkpoint 的 original-full/resize/sliding mIoU 分别为 `52.98`/`56.31`/`51.89`，epoch-500 的 resize/sliding 分别为 `56.73`/`52.08`。五项均为 318 样本、原始 Label grid 计分，checkpoint/split SHA 匹配且 `official_test_included=false`。这些是单 seed 几何诊断，不是三 seed 正式 baseline；resize 数值最高但存在长宽比改变，不能据单一最高 mIoU 直接冻结 geometry。详细中间报告见 `doc/reports/2026-08-28-museg-stage05-posteval-protocol-gate.md`。
- 固定 best checkpoint 的三臂 original-full 颜色诊断已完成：legacy BGR、RGB+RGB mean/std、BGR+反向 mean/std 的 mIoU 分别为 `52.98`、`33.85`、`49.53`；三臂均为同一 checkpoint/split SHA、318 样本、原始 Label grid，且 official test 未参与。这证明现有 checkpoint 对输入契约强敏感，但不构成未来颜色谱系的 baseline 选择；paired 重训仍是必要门禁。
- 历史 Stage-05 seed 1 整体已完成其原协议内工作，但不作为新目标的正式 DFormerv2-MUSeg baseline：其训练使用固定 `train_scale_array=[1.0]`，历史验证使用单尺度原图整图，未实现 DFormerv2 论文风格的多尺度翻转评估。其五项后评估、颜色诊断和所有 checkpoint/归档证据继续保留，不回写。
- 50/50 个 val-dev 验证点齐全；最佳结果为 mIoU `52.84`，对应 epoch 460。该结果是单 seed development 结果，不是三 seed 正式 baseline。
- 最终记录 `63,973` 次有效 optimizer update。相对 64,000 次理论 loop 网格的少量 AMP 跳过不阻塞本次训练完成结论；未来遥测已改为分别记录尝试、完成和跳过计数。
- 原始 v1 `acceptance.json` 仍为失败且未改写，唯一失败项是旧版 `milestones_complete` 遥测规则；原始 `failed.json` 同样保持不变。
- 独立 `acceptance-v2.json` 为 `pass=true`：重新哈希原报告列出的全部 18 项证据（含 12 个 checkpoint），核验 500 个 epoch 末核心日志、50 个验证点、best/final 身份和 official-test 封存状态。
- official test 继续保持 `sealed_unread`：原始裁决记录 `official_test_read=false`，训练结果记录 `official_test_included=false`，运行配置记录 `sealed_unread=true`。
- 已在 `main` 的 `c9ad268b5ee3ab685a4f93c945bfcdd843c49ab9` 基线上完成稳定项目指南 `doc/guides/project/README.md` 与逐文件目录 `doc/guides/project/file-catalog.md`：Git 跟踪路径 1,141/1,141 已说明，排除 0、缺失 0、重复 0。该目录是结构与导航证据，不替代本文件的实时事实。

## 2. 证据取回与云实例状态

- 完整不可变归档当前可访问于仓库内暂存路径 `D:\0Project\DFormer\cloud\DFormer-stage05-archive\museg-stage05-seed772961337\museg-stage05-seed772961337-original.tar`；文件大小为 `3,865,057,280` bytes，本地 SHA-256 为 `4f6b079b707266ee358d2522fc6e4e034a5380d09ba8c65696df7aaa3e383c66`，与 sidecar 清单一致。
- 后评估资产已位于 `D:\0Project\DFormer\cloud\DFormer-stage05-evidence`：独立 `best-val-miou.pth` 与 `epoch-500.pth` 的 SHA-256 分别为 `b62ca049e6a647aca109c70e80823cec8e36ae1cc1df27e3bcf2b1d215b160bf` 与 `0b88ab022db5188fd3439ea4e3af2098fe81e7c85757d1d91db33e831df2ff79`；本地 val-dev bundle 为 318 样本、954 个 RGB/Depth/Label 文件，split SHA-256 为 `1d0719d8f64f016d48995c25ab66d4004d76b7155d9efeef7cbb7454c0dd0e83` 且 `official_test_included=false`。五份 v2 结果位于其 `posteval/` 子目录；旧 `best-original-full.json` 是 v1 失败现场，继续保留但不计入五项结果。
- CompShare 实例 `cpod-1tyvjsiu6ahe` 已在本地哈希复核完成后停止，复查状态为 `Stopped`。
- 本次暴露了自动关机风险：训练完成后未按预期及时停止。未来所有终态都必须先同步证据并核验哈希，再停止实例，同时在控制面预设最晚停止兜底；验收结果不得控制是否继续计费。

## 3. 当前正在进行

- `tools/evaluate_museg_checkpoint.py` 使用 `criterion=None` 跳过单独 pretrained 初始化、以 `strict=True` 加载完整 checkpoint，并为 PyTorch 2.6+ 显式使用 `weights_only=False` 读取已核验可信旧 checkpoint；post-evaluation v2 契约显式记录 input contract，所有 geometry 在原始 Label grid 计分。
- 五项后评估与 identity/geometry 核验、固定 checkpoint 三臂诊断已完成；这些结果只属于历史 seed 1 的诊断，不是新目标的 DFormerv2 论文口径 baseline。新计划尚未启动新的 evaluator、protocol、云实例或训练；下一步先细化官方 DFormerv2 训练增强、multi-scale flip 测试和 MUSeg 原始 Label 网格计分契约。
- 唯一实时入口维护规则已写入 `.cursor/rules/museg-current-status.mdc` 和 README；每次 MUSeg 对话必须在最终答复前同步持久状态变化，无事实变化时不更新时间。
- 文档职责已进一步收口：Stage-04 云端交接和 2026-08-19 数据处理评审已归入 `doc/reports/`，File Browser/OpenList 操作说明已归入 `doc/guides/cloud/`，`doc/dataset.md` 继续作为稳定数据入口；有效引用已同步更新。最终一致性检查已通过：旧路径无残留，报告索引 JSON 可解析，归档 Canvas 无诊断，完整暂存 whitespace/rename 检查通过。逐文件复核确认普通差异与忽略 CRLF 后的实质差异一致，没有纯换行污染进入提交；文档治理提交 `c9ad268b5ee3ab685a4f93c945bfcdd843c49ab9` 已直接核验为当前 `origin/main`。
- 2026-08-27 经用户确认删除语义重复且含过时“当前门禁”的 `doc/plans/MUSeg-4090云端训练交接计划.md`；其导航和三份历史报告中的旧引用已改为当前状态入口、Stage-04 计划或日期化报告。`00`/`04` 的形成时点边界、`05` 的训练裁剪与 validation 几何区别、`08` 已处置的自然证据门禁、seed 1 预启动报告的 validation 几何勘误及运行中交接的后继状态均已澄清；`doc/guides/README.md` 与 `doc/plans/README.md` 已规定统一文档状态头。Stage-01 至 Stage-04 的设计证据继续保留，不据此重跑历史 workload。
- 全仓指南任务的历史静态验收和提交 `1fbb0da` 保持不变；本轮新增计划/代码迁移是在其后的未提交差异，未改写该指南基线计数。当前工作区还包含用户此前已有的文档治理和旧交接计划删除/引用更正。
- 本轮计划/代码迁移仍是工作区未提交差异，并叠加在用户此前已有的文档治理修改上；未运行训练、GPU、五项后评估、颜色诊断、云资源或 official test。当前 `main` 相对 `origin/main` ahead 1，远端未推送。
- 新计划的第一步是先确认总方向，再建立细化的 DFormerv2 训练/测试协议；目标是使用官方公开的随机尺度训练和 multi-scale flip 测试，在 MUSeg 原始 Label 网格上计分。颜色/归一化、具体 evaluator、预算、seed 和云资源仍需在后续详细计划中冻结，当前不得据新总规划直接启动 workload。