# DFormer 项目实时状态

> 事实时间：2026-10-07。本文件是新项目的唯一实时状态入口，替代旧项目的 `MMFR/` 与 `doc/main/` 状态体系。
> 维护方式见 `.cursor/skills/project-state/SKILL.md`：有事实变化时就地改写，不追加流水账；超过约 120 行或阶段结束时，把有追溯价值的旧内容归档到 `doc/archive/<日期-事件>/` 后再重写。

## 当前阶段

干净重建已完成，处于“论文方向待确定”阶段。

本轮完成：旧项目整体封存、以作者最新 DFormer++ 代码为基线重建仓库、建立轻量 `doc/` 文档入口、`outputs/` 产物约定和四个 Cursor 项目 Skill。尚未迁移任何旧研究代码，未运行训练或评价，未操作云资源。

## 已确认事实

- **基线**：`upstream/main` = `e3273009b759b578945483828ff315d560be94c9`，是作者官方 DFormer / DFormerv2 / DFormer++ 代码，含 `models/encoders/DFormerPP.py` 与 `local_configs/NYUDepthv2/DFormerPP_{T,S,B}.py`。2026-10-07 用 `git ls-remote upstream` 复核，作者远端 `main` 仍是该提交，本分支 HEAD 与其一致，未落后。
- **当前分支**：`research/dformerpp-clean-start`，直接从上述提交建立，未合并任何旧 MMFR 分支；工作区相对基线干净，共 1037 个跟踪文件。
- **远端**：`origin` = `https://github.com/Newton-liub/DFormer.git`（你的 fork，`main` 为 `43012ae`，未改写）；`upstream` = `https://github.com/VCIP-RGBD/DFormer.git`，其 push URL 已设为禁用值，防止误推作者仓库。当前研究分支未设置 upstream 跟踪，推送需显式指定目标。
- **Cursor 机制**：两个 Rule（`.cursor/rules/project-context.mdc`、`project-safety.mdc`）与四个 Skill（`project-state`、`research-idea`、`subagent-dispatch`、`compshare-cloud`）已随 Cursor 重载被发现，四个描述均正常显示（`compshare-cloud` 原先的冒号缺陷已修复生效）。实际执行验证：`project-state` 已用于读写本文件，`research-idea` 已写入一条真实 idea，`compshare-cloud` 的只读命令已在真实平台跑通，`subagent-dispatch` 已实际委派一次限定范围的只读盘点。现有子代理模型配置未改动。
- **运行环境**：`df2` conda 环境（`D:\2Env\anaconda\envs\df2`）为 Python 3.10.20、torch 2.7.0+cu128（CUDA 可用、1 张卡）、mmcv 1.7.2、timm 1.0.28、numpy 2.2.6。与作者 README 建议的 torch 2.1.2+cu118 / mmcv 2.1.0 **不一致**；尚未用该环境 import 项目代码或运行任何流程，兼容性**待核验**。
- **数据集**：本机 `D:\0Project\dataset\` 只有 MUSeg 两套产物——原始 `MUSeg`（六个矿区目录加 Experiment/Processing，约 1.44 GB）与已转换的 `MUSeg_DFormer`（RGB / Label / Depth / Depth16 各 3171 个文件，train 1595 + test 1576 = 3171，附 `dataset_meta.json`，约 1.47 GB）；文件后缀 `.jpg` / `.png` 与 RGBXDataset 约定一致。NYU Depth v2 与 SUN RGBD 本地不存在。本轮只做检查，未因此修改任何配置。
- **预训练权重**：本机 `D:\0Project\pretrained\DFormerv2_Small_pretrained.pth`（约 105 MB）确认存在，对应作者 DFormerv2_Small 配置。作者 DFormer++ 配置需要的 `checkpoints/pretrained/DFormerPP_{Tiny,Small,Base}.pth.tar` 在 `D:\0Project\pretrained` 与仓库根目录的直接检查中均未发现，仓库内也没有 `checkpoints/` 目录；更大范围的只读盘点在本次收尾时仍未返回，故 DFormer++ 权重是否存在于其他位置**待核验**。
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
- 新基线未做环境验证：未安装或升级依赖、未 import 配置、未做 forward、未运行测试。作者 NYU 配置的默认评价入口使用 `test.txt`，新方向必须重新决定数据与评价合同。
- 旧研究目录（MMFR、LER、protocols、experiments、human、tests）不恢复；`research/`、`local_configs/research/`、`doc/reports/`、`doc/archive/` 按需创建，当前不存在。

## 授权边界

- 当前未授权：GPU/训练/长耗时评价、云实例启停或删除、依赖升级、使用 official test。
- 已获用户确认（2026-10-07）：把本轮 `doc/` 与 `.cursor/` 改动提交并推送到 `origin` 的 `research/dformerpp-clean-start`；`upstream` 保持 push 禁用，不得推送到作者仓库。
- 需要单独确认：创建或销毁云资源、任何产生费用的操作、大规模重跑、其他提交与推送。

## 恢复点

新仓库位于 `D:\0Project\DFormer\`，分支 `research/dformerpp-clean-start`（HEAD `e3273009b759b578945483828ff315d560be94c9`），工作区只有本轮新增的文档与 Cursor 文件未跟踪，没有源码改动。旧项目完整保留在 `D:\0Project\DFormer-archive-20261007\`。中断后从这里继续：确定论文方向 → 评审最小迁移清单 → 需要时才动数据与云端。
