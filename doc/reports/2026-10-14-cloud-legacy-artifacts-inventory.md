# 云端旧产物盘点与清理预审（2026-10-14）

> 范围：只读盘点。本轮未删除、移动、覆盖、上传或下载任何云端资产，未扩容，未启动 GPU，未改动 OC-RGA 分支代码与配置。
> 实例：`cpod-1vbh7faqcauq`（无卡，GPU=0，2C/4G）。检查开始时即为 `Running`，本轮未做启动/停止等生命周期变更。
> 本地核对基线：`D:\0Project\DFormer-archive-20261007\`（含 `cloud\`、`outputs\`、`experiments\`、`.git` 的旧工程快照）、`D:\0Project\DFormer\outputs\`、`D:\0Project\dataset\`、`D:\0Project\dformer历史\`。
> 判定口径：**已验证**＝有字节数/链接关系等直接证据；**推测**＝仅凭目录名与来源判断，未做进一步校验。

## 1. 磁盘现状

| 项目 | 数值 |
|---|---|
| `/`（overlay 系统盘） | 50G，已用 38G，可用 13G，使用率 76% |
| `/root/rivermind-data/cloud` | 16G |
| `/root/rivermind-data/DFormer` | 6.7G（其中 `outputs/` 6.7G，代码仅约 20M） |
| `/root/rivermind-data/dataset` | 6.7G |
| `/root/rivermind-data/DFormer-archive-20261008` | 3.8G |
| `/root/rivermind-data/pretrained` | 106M |

## 2. 旧产物清单与本地备份情况

### 2.1 `/root/rivermind-data/cloud/`（16G，MMFR 旧实验）

| 子目录/文件 | 占用 | 内容 | 本地副本 |
|---|---:|---|---|
| `mmfr-a2-v3-probe/` | 5.4G | 20260915 云端容量/checkpoint 闸门探针：`depth-corruption-v3-checkpoint-gate/{parent,child,reference}/checkpoint` 与 `attempts/checkpoint-gate-*` 各含 3–4 个 ~321M 权重；`attempts/` 另含 missing-mmcv、vram-gate-abort 失败尝试；`clean-control-v3`(272K)、`depth-corruption-v3`(680K)、`manifests`(12K) | **无**（本地只有 `mmfr-a2-v3-*` 的 CPU 级 KB–MB 记录） |
| `mmfr-e1-batch1a-v1/` | 3.7G | `C0/` 与 `FLite/` 各含 `checkpoint/{epoch-5,10,15,20,latest,update-2560}.pth`（各约 321M，共 12 个） | **部分**：本地 `cloud\MMFR_E1_Batch1A_local_transfer_20260922\`（618M）含两边 `update-2560.pth` 与全部 mainval/quickval metrics；5 个中间点无副本 |
| `mmfr-a2-v3-one-arm-screening-v1/` | 1.6G | `development/seed-772961337/checkpoint/{best-val-miou,latest,selector-epoch-370,390,420}.pth` | **4/5 已验证**：本地 `experiments\MMFR_A2_v3\checkpoints\{latest,selector-370,390,420}.pth` 字节数逐一相同；`best-val-miou.pth`(321,088,863) 未见副本 |
| `MMFR_A2_v3_transfer_20260917/` | 1.3G | `checkpoints/{latest,selector-370,390,420}.pth` + `MANIFEST.txt` + `training_records`(3.5M) + `cloud_evidence`(8.5M) + `ARCHIVE_SHA256SUMS.txt` + `RESTORE_INSTRUCTIONS.txt` | **checkpoints 已验证**：本地 `experiments\MMFR_A2_v3\checkpoints\` 4 个文件字节数逐一相同；元数据类未逐项核对 |
| `MMFR_A2_v3_transfer_20260917.7z` | 1.1G | 同上的打包版 | 未解压核对（**推测**与解压目录重复） |
| `MMFR_E1_Batch1A_local_transfer_20260922/` | 616M | `C0/checkpoint/update-2560.pth` + `FLite/checkpoint/update-2560.pth` + quickval/metrics | **有**（本地同名目录 618M） |
| `MMFR_E1_Batch1A_local_transfer_20260922.tar.gz` | 566M | 同上的包装 | **有**（本地同名 tar.gz 566M） |
| `MMFR_AV1_local_val_conditional_20261001.zip` | 411M | 996 条目：`checkpoints/C0/update-2560.pth`(321,150,608)、`checkpoints/A-v1/update-2560.pth`(107,210,042)、`evidence/quickval/*`、`evidence/amended-run/*`、`source/*` | **无**（本地只有另一个 290M 的 C0 版 transfer 包，不含 A-v1 权重与 quickval evidence） |
| `natural-missing-round1-formal-20261003-edb660a/` | 929M | `NaturalMissing-Natural-formal/.../checkpoint/{latest,update-640,update-1280}.pth` + `monitor/`（含 10M stdout） | **无权重**（本地仅 0.0MB 记录目录） |
| `natural-missing-preflight-20261003-18271ad/` | 919M | Grid/Natural/Replay 三个 `preflight-update-000003.pth` + `monitor/` | **无权重**（本地 `natural-missing-cloud-preflight-20261003` 仅 0.2MB） |
| `natural-missing-qualification-20261004-7d6246a/` | 21M | monitor 日志 | **有**（本地 20.4M） |
| `natural-missing-numerical-20261004-1cf9a8a/` | 16M | monitor 日志 | **有**（本地 15.4M） |
| `natural-missing-round1-formal-20261003-8d2490e/` | 16K | 失败尝试残件（stdout 为 0 字节） | 无（无保留价值） |
| `*.sha256`（3 个） | 12K | 对应 `.7z` / `.zip` / `.tar.gz` 的校验文件 | — |

### 2.2 `/root/rivermind-data/DFormer/outputs/`（6.7G，ODG R1/R2）

| 子目录 | 占用 | 云端文件 | 本地副本 |
|---|---:|---|---|
| `sun-dev-original-seed12345/20261010-013324/`（R1） | 931M | `last.pth`(324,396,242)、`best-dev.pth`(324,542,542)、`stage-epoch-30.pth`(324,559,720) + csv/json/log | **2/3**：本地有 `last.pth`、`best-dev.pth`（字节数相同）；缺 `stage-epoch-30.pth` |
| `sun-dev-odg-seed12345/20261010-020527/`（R1） | 931M | 同上（324,396,242 / 324,542,478 / 324,559,656） | **2/3**：同上，缺 `stage-epoch-30.pth` |
| `sun-dev-original-seed12345-r2/20261010-094231/`（R2） | 931M | `last.pth`(324,421,238)、`best-dev.pth`(324,567,822)、`stage-epoch-30.pth`(324,585,522) | **1/3**：本地仅有 `last.pth` |
| `sun-dev-odg-seed12345-r2/20261010-094236/`（R2） | 931M | 同上（324,421,174 / 324,567,758 / 324,585,522） | **1/3**：本地仅有 `last.pth` |
| `sun-odg-smoke-original/`（3 个 run） | 929M | 每个 `smoke.pth`(324,481,281 / 324,481,281 / 324,508,284) + json/csv/log | 无 `.pth`（本地有 json/log/csv/swanlab） |
| `sun-odg-smoke-odg/`（1 个 run） | 929M | `smoke.pth`(324,508,220) + json/csv/log | 无 `.pth` |
| `probe-mb4/`、`probe-mb8/`、`probe-par-odg/`、`probe-par-original/` | 各 310M | 各含 1 个 `smoke.pth`（约 324M） + json/csv/log | 无 `.pth` |
| `probe-mb16/` | 40K | 仅 json/log（OOM 未产出权重） | 有 |
| `odg-gpu-20261010/` | 144K | 9 个 screen/probe 日志 | 有 |

本地 `D:\0Project\DFormer\outputs\` 另有今日（2026-10-14）`oc-rga-deliver/20261014-090517/smoke.pth`(110,123,647) 与 `20261014-090606/smoke.pth`(330,342,489)，属当前 OC-RGA 分支，**不在本次清理范围**。

### 2.3 `/root/rivermind-data/DFormer-archive-20261008/`（3.8G，旧工程快照）

- 顶层是完整旧工程（`models/`、`utils/`、`mmseg/`、`local_configs/`、`MMFR/`、`doc/`、`.git` 等，代码约 20M）；`.git` HEAD = `7d6246a7db77f77e26e20d84074dff6e191e9fa5`（2026-10-04）。本地 `D:\0Project\DFormer-archive-20261007\` 为同源快照（含 `.git` 与几乎相同目录），**推测**高度重叠，未逐文件核对。
- `cloud/` 3.1G：

| 子目录 | 占用 | 大文件 | 本地副本 |
|---|---:|---|---|
| `mmfr-e1-batch1b-roe-v2/` | 2.1G | `R-OE-lite/development/seed-772961337/checkpoint/{epoch-5,10,15,20,latest,update-2560}.pth`（各约 360M） | **无** |
| `mmfr-e1-batch1b-roe-v1/` | 345M | `.../checkpoint/latest.pth`(360,786,246) | **无**（本地只有 `mmfr-e1-batch1b-roe-3update-local-...` 0.1MB 记录） |
| `mmfr-av1-gradient-diagnosis-v1/` | 288M | `formal/diagnostic-failure-state.pth`(300,675,333) | **无** |
| `mmfr-av1-formal-fp32-resume1280-v1/` | 223M | `formal/checkpoint/{proposal-update-1920,update-2560}.pth`（各约 107M） | **无**（`update-2560.pth` 与 2.1 节 411M 条件包内 `A-v1/update-2560.pth` 字节数同为 107,210,042） |
| `mmfr-av1-formal-v1/` | 206M | `formal/checkpoint/{update-640,update-1280}.pth`（各约 107M） | **无** |
| `*-console.log`（6 个） | 约 2.5M | 训练控制台日志 | 未核对 |

- `experiments/MMFR_A2_v3/checkpoints/selector-epoch-420.pth` 314M：与 `cloud/MMFR_A2_v3_transfer_20260917/checkpoints/selector-epoch-420.pth` 为**同一硬链接**（nlink=2，同一 inode），且本地已有同名同字节文件。
- `experiments/MMFR_OracleA/` 22M（eval metrics json）——本地 `experiments\MMFR_OracleA\` 有同构内容。
- `MMFR_AV1_cloud_transfer_20261001.zip` 290M（303,621,580 bytes）——本地 `D:\0Project\MMFR_AV1_cloud_transfer_20261001.zip` 大小一致（3 条目：`C0/checkpoint/update-2560.pth`、`source-7773845.zip`、`README.md`）。
- `outputs/` 16M（`find -size +5M` 为空）。

### 2.4 `/root/rivermind-data/dataset/`（6.7G）

| 条目 | 占用 | 本地副本/说明 |
|---|---:|---|
| `SUNRGBD/` | 2.4G | 本地 `D:\0Project\dataset\SUNRGBD`（2,338M）已有；ODG R1/R2 训练数据 |
| `SUNRGBD-odg-20261010.zip` | 2.3G | 本地 `SUNRGBD.zip` 同为 2,452,204,576 bytes（**已验证**）；与同目录解压版重复 |
| `MUSeg_DFormer/` | 1.5G | 本地 `D:\0Project\dataset\MUSeg_DFormer`（1,501M）+ `.7z`（1,468M）已有 |
| `semantic.tar` | 253M | 修改时间 2026-10-14 01:18，**疑似当前 OC-RGA 工作产物** |
| `depth.tar` | 306M | 修改时间 2026-10-14 01:15，同上 |
| `DELIVER/` | 20M | `img/`、`depth/`、`semantic/` × `{cloud,fog,night,rain,sun}`，仅为部分副本（本地 `D:\0Project\dataset\DELIVER` 为 12,993M 完整版） |

### 2.5 链接关系（清理时必须注意）

- **硬链接**：`cloud/MMFR_A2_v3_transfer_20260917/checkpoints/selector-epoch-420.pth` ↔ `DFormer-archive-20261008/experiments/MMFR_A2_v3/checkpoints/selector-epoch-420.pth`（nlink=2）。只删其一不释放空间。
- **软链接**：`DFormer/checkpoints/pretrained/DFormerv2_Small_pretrained.pth` → `/root/rivermind-data/pretrained/DFormerv2_Small_pretrained.pth`。`pretrained/` 是当前工程的预训练入口，**必须保留**；不要因“工程内只有软链接”而把 `pretrained/` 判为无用。
- 各实验目录下 `tb/tb` 为指向同目录 TensorBoard 子目录的软链接（无独立体积）。

## 3. 建议清理清单

### A. 建议删除（已验证重复或无保留必要，无使用依赖；合计约 **7.3G**）

| 路径 | 释放 | 依据 | 风险 |
|---|---:|---|---|
| `dataset/SUNRGBD-odg-20261010.zip` | 2.3G | 本地同名 zip 字节数一致（**已验证**），云端解压版 `SUNRGBD/` 保持可用 | 低 |
| `cloud/MMFR_A2_v3_transfer_20260917/checkpoints/`（4 个 pth） | 1.28G | 本地 `experiments\MMFR_A2_v3\checkpoints\` 4 文件字节数逐一相同（**已验证**） | 低；删除时须连同下方硬链接项 |
| `DFormer-archive-20261008/experiments/MMFR_A2_v3/checkpoints/selector-epoch-420.pth` | 314M | 硬链接副本且本地已有（**已验证**） | 低；须与上项同时删 |
| `cloud/MMFR_E1_Batch1A_local_transfer_20260922/` | 616M | 本地同名目录 618M（含全部权重与 metrics） | 低 |
| `cloud/MMFR_E1_Batch1A_local_transfer_20260922.tar.gz` | 566M | 本地同名 tar.gz 566M | 低 |
| `DFormer-archive-20261008/MMFR_AV1_cloud_transfer_20261001.zip` | 290M | 本地同名 zip 同字节数 | 低 |
| `DFormer/outputs/{sun-odg-smoke-original,sun-odg-smoke-odg,probe-mb4,probe-mb8,probe-par-odg,probe-par-original}/**/smoke.pth`（6 个） | 约 1.95G | 冒烟/性能诊断临时快照，本地已留同 run 的 json/log/csv；非任何报告结论依据 | **中低（推测性保留判定）**：若上级认为 smoke 权重需留档，应降为 B 类 |
| `cloud/natural-missing-{numerical,qualification}-2026*/monitor` | 37M | 本地已有同内容记录 | 低 |
| `cloud/*.sha256`（3 个） | 12K | 校验文件，对应包删除后失去意义 | 低 |

> 说明：`cloud/MMFR_A2_v3_transfer_20260917/` 下的 `MANIFEST.txt`、`training_records`、`cloud_evidence`（约 12M）本地是否已有同内容**未逐项核对**，故该目录的元数据部分不列入 A 类。

### B. 建议先取回本地再删除（仍有科研价值；合计约 **3.0G**）

| 路径 | 体积 | 理由 |
|---|---:|---|
| `DFormer/outputs/sun-dev-{original,odg}-seed12345/**/stage-epoch-30.pth`、`sun-dev-{original,odg}-seed12345-r2/**/stage-epoch-30.pth`（4 个） | 约 1.3G | R1/R2 的 30 epoch 阶段点，本地缺失；状态记录明确“云端保留未取回” |
| `DFormer/outputs/sun-dev-{original,odg}-seed12345-r2/**/best-dev.pth`（2 个） | 约 0.65G | R2 的 best-dev 权重，本地 R2 仅有 `last.pth` |
| `cloud/MMFR_AV1_local_val_conditional_20261001.zip` | 411M | 含本地完全缺失的 `checkpoints/A-v1/update-2560.pth`(107M) 与 quickval evidence；本地只有仅含 C0 的 290M transfer 包 |
| `DFormer-archive-20261008/cloud/mmfr-e1-batch1b-roe-v2/**/update-2560.pth` | 360M | E1 Batch1B 的最终点，本地 v1/v2 均无 |
| `cloud/natural-missing-round1-formal-20261003-edb660a/**/latest.pth` | 321M | 自然缺失正式轮的代表权重，本地无 |

### C. 建议保留

| 路径 | 体积 | 理由 |
|---|---:|---|
| `pretrained/DFormerv2_Small_pretrained.pth` | 106M | 当前 DFormer 工程预训练入口，被 `DFormer/checkpoints/pretrained/` 软链接引用 |
| `DFormer/`（`research/`、`local_configs/`、`models/` 等代码） | 约 20M | 当前研究分支云端工作副本 |
| `dataset/semantic.tar`、`dataset/depth.tar` | 559M | 2026-10-14 01:15–01:18 生成，疑似当前 OC-RGA 工作产物，本轮不动 |
| `dataset/DELIVER/` | 20M | 当前工程已接入 DeLiVER（部分副本，20M 成本极低） |
| `DFormer/outputs/{odg-gpu-20261010,probe-mb16}/`、各 run 的 json/csv/log | < 10M | 结论与配置记录 |
| `cloud/MMFR_A2_v3_transfer_20260917/{MANIFEST.txt,training_records,cloud_evidence}` | 约 12M | 归档元数据，成本极低，保留以避免失去恢复线索 |

### D. 待确认（不能证明可安全清理；合计约 **17G+**）

| 路径 | 体积 | 待确认点 |
|---|---:|---|
| `cloud/mmfr-a2-v3-probe/` | 5.4G | 本地无任何副本；用途为 20260915 云端容量/checkpoint 闸门机制诊断（含失败尝试），是否仍需回读未定 |
| `cloud/mmfr-e1-batch1a-v1/{C0,FLite}/**/{epoch-5,10,15,20,latest}.pth`（10 个） | 约 3.2G | 中间 checkpoint；本地已有 `update-2560`，中间点是否需留待定 |
| `cloud/mmfr-a2-v3-one-arm-screening-v1/**/best-val-miou.pth` | 321M | 同目录其余 4 个 checkpoint 本地已有，仅此项无副本 |
| `cloud/natural-missing-preflight-20261003-18271ad/checkpoints/` | 919M | 3 个 preflight 权重，本地无；preflight 是否属可弃中间态未定 |
| `cloud/MMFR_A2_v3_transfer_20260917.7z` | 1.1G | **推测**与同目录解压版重复，未解压核对 |
| `DFormer-archive-20261008/cloud/{mmfr-av1-formal-v1,mmfr-av1-formal-fp32-resume1280-v1,mmfr-av1-gradient-diagnosis-v1,mmfr-e1-batch1b-roe-v1}/*` | 约 1.06G | 本地未见对应副本；AV1 formal 的 `update-2560.pth` 可能已被 B 类 411M 条件包覆盖（同为 107,210,042 bytes），其余无替代 |
| `DFormer-archive-20261008/` 代码目录与 `.git`（约 22M） | 22M | 与本地 `DFormer-archive-20261007\`（同源快照）重叠程度未逐文件核对 |
| `DFormer-archive-20261008/cloud/*-console.log` | 约 2.5M | 日志是否已并入本地证据未核对 |
| `dataset/MUSeg_DFormer/` | 1.5G | 本地已有同数据（+`.7z`），云端是否仍需保留以便重跑旧 MUSeg 实验待定 |
| `dataset/SUNRGBD/`（解压版） | 2.4G | 若 OC-RGA 后续在云端跑 SUN 则需保留；是否继续云端训练未定 |
| `DFormer/outputs/sun-dev-*/last.pth`、`best-dev.pth`（本地已有） | 约 2.6G | 云端是否留档以备复核由上级决定 |

## 4. 需要先取回的文件（仅存云端、有保留价值）

1. ODG R1/R2 缺失权重共 6 个：R1 × `stage-epoch-30.pth`（2）、R2 × `best-dev.pth`（2）与 `stage-epoch-30.pth`（2），约 **1.95G**。
2. `cloud/MMFR_AV1_local_val_conditional_20261001.zip`，**411M**（含 `A-v1/update-2560.pth` 与 quickval evidence）。
3. `DFormer-archive-20261008/cloud/mmfr-e1-batch1b-roe-v2/**/update-2560.pth`，**360M**。
4. `cloud/natural-missing-round1-formal-20261003-edb660a/**/latest.pth`，**321M**。
5. （可选）`cloud/mmfr-e1-batch1b-roe-v1/**/latest.pth`，360M；`cloud/mmfr-av1-formal-v1/formal/checkpoint/update-1280.pth`，107M。

合计建议取回体积约 **3.0–3.5G**（无卡窗口传输，不启动 GPU）。

## 5. 最终建议

- **保守清理（仅 A 类）**：预计释放 **约 7.3G**；若上级认为 `smoke.pth` 与云端 `SUNRGBD.zip` 需保留，则保守值降为 **约 3.0G**（仅重复归档、硬链接副本与已重复的压缩包）。可用空间从 13G 增至约 **16–20G**。
- **扩大清理（A 类 + B 类取回后删除 + D 类中确认可弃的 probe / 中间 checkpoint / 旧快照 cloud）**：再释放 **约 15–18G**，累计 **约 22–25G**，可用空间可回到 **35G 以上**。
- 关键依赖提醒：删除前必须先处理 `selector-epoch-420.pth` 硬链接对（单独删一项不释放空间），并确保 `pretrained/DFormerv2_Small_pretrained.pth` 不被误删。
- `dataset/semantic.tar`、`dataset/depth.tar` 及本地 `D:\0Project\DFormer\outputs\oc-rga-deliver\` 属当前 OC-RGA 工作，本轮与下一轮清理范围均应排除。
- 已核对的“仅存云端”项集中在 `mmfr-a2-v3-probe`、`natural-missing-*` 权重、`MMFR E1 Batch1B ROE`、`MMFR AV1 formal` 四类；其中 MMFR 系列属已结束的旧实验，是否保留完整权重请上级按研究需要裁决。

## 6. 本轮边界

未删除/移动/覆盖任何云端或本地资产；未上传或下载大文件；未扩容；未启动 GPU；未运行训练、评价或测试；未改 OC-RGA 分支代码与配置；未创建或销毁云资源；实例 `cpod-1vbh7faqcauq` 保持检查开始时的 `Running`（无卡）状态。等待上级确认具体清理范围后再执行删除。
