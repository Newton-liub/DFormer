# DFormer 项目实时状态

> 事实时间：2026-10-07。本文件是新项目的唯一实时状态入口，替代旧项目的 `MMFR/` 与 `doc/main/` 状态体系。
> 维护方式见 `.cursor/skills/project-state/SKILL.md`：有事实变化时就地改写，不追加流水账；超过约 120 行或阶段结束时，把有追溯价值的旧内容归档到 `doc/archive/<日期-事件>/` 后再重写。

## 当前阶段

干净重建已完成，五项裁决已下达并执行完毕，处于“论文方向待确定”阶段。

本轮完成：旧项目整体封存、以作者最新 DFormer++ 代码为基线重建仓库、建立轻量 `doc/` 文档入口与四个 Cursor 项目 Skill、按裁决建立独立运行环境并做最小验证、把外部论文与源码索引迁到资产旁。尚未迁移任何旧研究代码，未运行训练或评价，未操作云资源。

## 已确认事实

- **基线**：`upstream/main` = `e3273009b759b578945483828ff315d560be94c9`，是作者官方 DFormer / DFormerv2 / DFormer++ 代码，含 `models/encoders/DFormerPP.py` 与 `local_configs/NYUDepthv2/DFormerPP_{T,S,B}.py`。2026-10-07 用 `git ls-remote upstream` 复核，作者远端 `main` 仍是该提交，本分支 HEAD 与其一致，未落后。
- **当前分支**：`research/dformerpp-clean-start`，直接从上述提交建立，未合并任何旧 MMFR 分支；工作区相对基线干净，基线 1037 个跟踪文件，加上本轮提交的 11 个文档与 Cursor 文件共 1048 个。
- **提交与推送**：本轮 `doc/` 与 `.cursor/` 共 11 个文件已作为该分支的首次提交 `64bc2bd` 推送到 `origin/research/dformerpp-clean-start`，本地 HEAD 与远端一致，工作区无未提交改动。旧 MMFR 历史仍在归档与新分支之外。
- **远端**：`origin` = `https://github.com/Newton-liub/DFormer.git`（你的 fork，`main` 为 `43012ae`，未改写）；`upstream` = `https://github.com/VCIP-RGBD/DFormer.git`，其 push URL 已设为禁用值，防止误推作者仓库。研究分支现已跟踪 `origin/research/dformerpp-clean-start`，后续 push 默认只到 fork。
- **Cursor 机制**：两个 Rule（`.cursor/rules/project-context.mdc`、`project-safety.mdc`）与四个 Skill（`project-state`、`research-idea`、`subagent-dispatch`、`compshare-cloud`）已随 Cursor 重载被发现，四个描述均正常显示（`compshare-cloud` 原先的冒号缺陷已修复生效）。实际执行验证：`project-state` 已用于读写本文件，`research-idea` 已写入一条真实 idea，`compshare-cloud` 的只读命令已在真实平台跑通，`subagent-dispatch` 已成功启动一次限定范围的只读盘点——但该子代理长时间未返回结果，已按用户决定停止，因此“委派可启动”已验证、“子代理能按时完成”**未验证**。现有子代理模型配置未改动。
- **运行环境**：本地研究环境是新建立的独立 conda 环境 `dformer`（`D:\2Env\anaconda\envs\dformer`），Python 3.10.20、torch 2.7.0+cu128、torchvision 0.22.0+cu128、scipy 1.15.3、mmcv 1.7.2、mmengine 0.10.7、timm 1.0.30、numpy 2.2.6。本机 GPU 为 RTX 5060 Laptop（compute capability 12.0），因此本地按“先满足 GPU”选择 CUDA 12.8 wheel，而非照搬作者的 torch 2.1.2+cu118；mmcv 2.1.0 没有对应 cu128/torch2.7 的预编译包，故本地用 PyPI lite 版 1.7.2。旧 `df2` 环境已**冻结**，不再修改。最小验证已通过：导入 `mmcv.cnn.ConvModule` 与 `LightHamHead` 成功，导入 `DFormerv2_S` 配置成功，用本机 DFormerv2_Small 权重建模型并前向一次成功（26.7M 参数、输出 `(1,40,480,640)`、全 finite、峰值 366 MiB）。详见 [运行环境](../guides/environment.md)。
- **外部资料索引**：索引真源已按裁决迁到外部资产旁 `D:\0Project\origin\_index\`（不放进 Git），内含 `PAPER_LIBRARY_INDEX.json/md`、`paper-index.md`、`code-index.md`、`external_reference_provenance_2026-09-21.md`、`README.md`、`pending-and-duplicates.md` 与 `tools\paper_library.py|.cmd`。7 个迁入文件逐一 sha256 比对与源一致；工具索引路径已改指本目录，`--dry-run` 实跑通过（无待迁移 bundle，报 38 个受管 bundle = 37 篇论文 + `DFormer-doc-paper` 内嵌套的 DFormerv2 副本）。状态分层：`论文\` 37 篇为 canonical；`论文待处理\`（9 项）标 Pending、`论文_duplicates_review\`（1 项）标 Duplicate Review，均不得当已核验论文使用；库外 `origin\DFormer` clone（`814799b`）标 deprecated，当前唯一工作基线是 `D:\0Project\DFormer\`。仓库侧只在 `doc/guides/external-resources.md` 留短入口。人工编号 PR/RE/AI 的历史依据仍留在归档 `MMFR\03_reference\`。
- **数据集**：本机 `D:\0Project\dataset\` 只有 MUSeg 两套产物——原始 `MUSeg`（六个矿区目录加 Experiment/Processing，约 1.44 GB）与已转换的 `MUSeg_DFormer`（RGB / Label / Depth / Depth16 各 3171 个文件，train 1595 + test 1576 = 3171，附 `dataset_meta.json`，约 1.47 GB）；文件后缀 `.jpg` / `.png` 与 RGBXDataset 约定一致。NYU Depth v2 与 SUN RGBD 本地不存在。本轮只做检查，未因此修改任何配置。
- **预训练权重（已裁决 + 已定案）**：第一轮方法验证用 **DFormerv2-Small** 作为可运行 backbone；DFormer++ 作为代码基线与后续目标 backbone，其预训练编码器作者仍未发布（README 三行 “Coming soon”，HuggingFace 只有 `bbynku/DFormerv2`），升级放到“方向被证明值得继续”之后的正式实验阶段。本轮**未下载任何新权重**，也不做完整 RGB-D 预训练、不用随机初始化去和成熟预训练模型比较。权重已按作者默认路径放置：`checkpoints\pretrained\DFormerv2_Small_pretrained.pth`（由原 `D:\0Project\pretrained\` 复制，sha256 一致，`checkpoints/` 被 Git 忽略）。加载方式严格遵循官方代码：`strict=False` 下 `unexpected keys` 是当前 backbone 不使用的预训练头参数（`proj/norm/head/aux_head`），`extra_norms.0/1/2` 不在 checkpoint 中、按作者代码保持标准 LayerNorm 初始化。2026-10-07 实测确认三个 `extra_norms` 的 weight 全为 1、bias 全为 0，即恒等变换。
- **旧项目归档**：`D:\0Project\DFormer-archive-20261007\`。归档前身份 HEAD `8c274c59775427544ab209957c26974f4d40d083`、分支 `perf/mmfr-a2-v3-pipeline-opt1`、1463 个跟踪文件、13 个已修改文件、3 个未跟踪条目；归档后逐项复核一致，未丢失。归档保留完整 `.git`（历史、分支、remote、index）以及被忽略产物的目录结构。
- **旧状态与证据位置**：旧实时状态和授权记录在归档的 `doc/main/MUSeg-current-status.md` 与 `doc/main/MUSeg-open-decisions.md`；旧实验证据在归档的 `MMFR/` 下。
- **外部资料**：论文全文仍在 `D:\0Project\origin\论文\`，外部 clone 代码仍在 `D:\0Project\origin\`，均未复制进仓库；旧论文与代码索引在归档的 `MMFR/03_reference/`。
- **云凭证**：`compshare doctor` 全部通过（凭据 profile 已加载、API 连通并返回 5 个可用区、ssh/scp 可用；SDK 0.11.114、Python 3.13.9）。名为 `default` 的 profile 同时配置了 public key 与 private key，来源为本机 `C:\Users\10094\.config\compshare\config.json`。密钥本身未读取、未输出。
- **云实例**：2026-10-07 直接查询平台，唯一实例 `cpod-1vbh7faqcauq`（名称 `mmfr-a2-4090-probe`，cn-bj2-03，Postpay，1 块 50 GB 系统盘，无数据盘）状态为 `Stopped`、GPU 0，且未设置计划关机。没有实例在运行，因此不产生 GPU 计费；**关机后系统盘是否继续计费未核验**（平台问答服务当时返回 502）。用户 2026-10-07 决定暂不释放该实例，因为没有超过需要收费的存储界限。

## 下一步

1. 确定论文方向与第一轮研究问题；这是迁移旧代码、新增研究代码的唯一前置条件。
2. 方向确定后，在 `doc/plans/` 写实施计划，逐项评审“只迁移确实需要”的旧实现，其余继续留在归档。
3. 需要实验时先确认数据集位置、开发集/评价合同与输出路径（`outputs/<experiment>/<run-id>/`），再单独授权 GPU；云端流程见 `doc/guides/cloud.md`。

## 阻塞与不确定项

- 研究方向未定，因此暂不迁入 MMFR、LER、旧 MUSeg 配置以及 `natural_missing` 等实现。
- 旧 DFormerv2 的 C0 权重、训练授权、指标阈值和数值修复资格**不沿用**到 DFormer++ 新基线；新实验需重新定义并单独授权。
- 新基线未做训练验证：未训练、未做评价、未运行完整测试。作者 NYU 配置的默认评价入口使用 `test.txt`，新方向必须重新决定数据与评价合同。
- DFormerv2-Small 的权重与加载方式已定案：权重复制到作者默认路径 `checkpoints\pretrained\`；`extra_norms.*` 是恒等初始化的 segmentation-side LayerNorm（实测 weight=1、bias=0），预训练头参数由 `strict=False` 忽略。首次 baseline 验收时把这组 missing / unexpected keys 记录一次即可，不每轮重审。
- 旧研究目录（MMFR、LER、protocols、experiments、human、tests）不恢复；`research/`、`local_configs/research/`、`doc/archive/` 按需创建，当前不存在。

## 已裁决（2026-10-07，已执行）

1. **主干与权重**：方向审计照常进行，不被 DFormer++ 权重阻塞；第一轮方法验证用 DFormerv2-Small；DFormer++ 保持代码基线与后续目标 backbone，其升级作为正式实验阶段的一项工作。不下载、不自行做完整 RGB-D 预训练、不随机初始化后直接比较。→ 未下载，已记录。
2. **环境**：旧 `df2` 冻结，为项目新建独立环境（先满足本机 RTX 5060，再最小验证）；明确补 `scipy`，`ftfy` 因无任何 import 而不优先处理；云端 4090 正式训练另按作者推荐栈建环境。→ 已建 `dformer` 环境并通过最小验证。
3. **索引落点**：外部资产目录是索引唯一真源，仓库只保留很短的资源入口文档，不复制第二份索引。→ 已迁至 `D:\0Project\origin\_index\`，仓库入口为 [外部资源入口](../guides/external-resources.md)。
4. **索引范围**：37 篇为 canonical 正式库；`论文待处理` 标 Pending、`论文_duplicates_review` 标 Duplicate Review，索引入口知道它们存在，但方向审计不得把它们当已核验论文使用。→ 已标注。
5. **库外 `origin/DFormer`**：不更新、不删除，先在代码索引里标 deprecated 并禁止研究任务默认使用；当前唯一有效工作基线是 `D:\0Project\DFormer\`，待新方向稳定且确认无独有内容后再删除。→ 已在 `code-index.md` 标注。

## 授权边界

- 当前未授权：GPU/训练/长耗时评价、云实例启停或删除、使用 official test。
- 已获用户确认（2026-10-07）：`doc/` 与 `.cursor/` 改动提交并推送到 `origin` 的 `research/dformerpp-clean-start`；`upstream` 保持 push 禁用，不得推送到作者仓库。
- 已获用户授权执行（2026-10-07）：新建独立环境 `dformer` 并安装依赖；把仍有效的索引与维护工具迁到 `origin\_index\` 继续维护。旧 `df2` 环境与旧项目归档不再改动。
- 需要单独确认：创建或销毁云资源、任何产生费用的操作、大规模重跑、下载新权重、提交与推送。

## 恢复点

新仓库位于 `D:\0Project\DFormer\`，分支 `research/dformerpp-clean-start`（基线 `e3273009`）。本轮裁决执行产生的改动（状态、报告、外部资源与环境入口文档、Rule 微调）已提交到本地该分支，**未推送**；`origin\_index\` 的索引位于 Git 之外。旧项目完整保留在 `D:\0Project\DFormer-archive-20261007\`。中断后从这里继续：确定论文方向 → 评审最小迁移清单 → 需要时才动数据与云端。
