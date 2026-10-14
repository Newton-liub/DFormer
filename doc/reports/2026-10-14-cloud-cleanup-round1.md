# 云端旧产物第一轮清理、资产取回与 OC-RGA DeLiVER 上传（2026-10-14）

> 依据：已审核的 [云端旧产物盘点与清理预审](2026-10-14-cloud-legacy-artifacts-inventory.md)（A 类保守清理 + B 类取回）。
> 实例 `cpod-1vbh7faqcauq`（无卡，GPU=0）：本轮未启动/停止实例，未启动 GPU 训练，未修改研究代码，未扩容。

## 1. 实际删除与释放

| 路径 | 类型 | 大小 |
|---|---|---:|
| `dataset/SUNRGBD-odg-20261010.zip` | 重复 zip（本地同字节副本） | 2,452,204,576 B ≈ 2.28G |
| `cloud/MMFR_A2_v3_transfer_20260917/checkpoints/{latest,selector-epoch-370,390,420}.pth` | 本地已验证重复 | 1,284,378,452 B ≈ 1.20G |
| `DFormer-archive-20261008/experiments/MMFR_A2_v3/checkpoints/selector-epoch-420.pth` | 与上一项同一 inode 的硬链接副本 | 321,103,318 B（与应用上项一并释放） |
| `cloud/MMFR_E1_Batch1A_local_transfer_20260922/` | 本地同名目录已覆盖（618M） | 约 616M |
| `DFormer-archive-20261008/MMFR_AV1_cloud_transfer_20261001.zip` | 本地顶层同字节同名 zip | 303,621,580 B ≈ 290M |
| `DFormer/outputs/**/smoke.pth`（10 个） | 冒烟/性能诊断临时快照 | 3,244,928,022 B ≈ 3.02G |

**合计释放约 7.4G。** 磁盘：已用 38G→31G，可用 13G→**20G**（使用率 76%→61%）。

保留（按用户要求）：`MMFR_A2_v3_transfer_20260917/` 的 `MANIFEST.txt`、`training_records`、`cloud_evidence`、`ARCHIVE_SHA256SUMS.txt`、`RESTORE_INSTRUCTIONS.txt`、`checkpoints/checkpoint-candidates.json`；三份 `.sha256` 未删（对应包仍有保留项）。smoke 各 run 的 json/csv/log/config 全部保留。

## 2. 与盘点报告的偏差更正

1. **smoke.pth 实为 10 个**（报告误记 6 个）：`sun-odg-smoke-original` 3 个、`sun-odg-smoke-odg` 3 个、`probe-mb4/mb8/par-odg/par-original` 各 1 个；均为 nlink=1，无硬链接。
2. **云端 `cloud/MMFR_E1_Batch1A_local_transfer_20260922.tar.gz` 不存在**（报告误列为云端项）。云端只有其 `.sha256`；同名 566M 文件仅存在于本地。
3. 硬链接判断经 `ls` 复核：`selector-epoch-420.pth` 两处 nlink=2，其余待删文件均 nlink=1。

## 3. 取回的重要资产（均已校验字节一致）

| 云端来源 | 本地位置 | 字节数 |
|---|---|---:|
| `DFormer/outputs/sun-dev-original-seed12345-r2/20261010-094231/best-dev.pth` | `D:\0Project\DFormer\outputs\sun-dev-original-seed12345-r2\20261010-094231\` | 324,567,822 |
| `DFormer/outputs/sun-dev-odg-seed12345-r2/20261010-094236/best-dev.pth` | `D:\0Project\DFormer\outputs\sun-dev-odg-seed12345-r2\20261010-094236\` | 324,567,758 |
| `cloud/MMFR_AV1_local_val_conditional_20261001.zip`（含 `A-v1/update-2560.pth` 与 quickval evidence） | `D:\0Project\DFormer-archive-20261007\cloud\` | 429,993,998 |
| `DFormer-archive-20261008/cloud/mmfr-e1-batch1b-roe-v2/R-OE-lite/development/seed-772961337/checkpoint/update-2560.pth` | `D:\0Project\DFormer-archive-20261007\cloud\mmfr-e1-batch1b-roe-v2\R-OE-lite\development\seed-772961337\checkpoint\` | 360,801,864 |
| `cloud/natural-missing-round1-formal-20261003-edb660a/checkpoints/NaturalMissing-Natural-formal/development/seed-772961337/checkpoint/latest.pth` | `D:\0Project\DFormer-archive-20261007\outputs\natural-missing-round1-formal-20261003\checkpoints\NaturalMissing-Natural-formal\development\seed-772961337\checkpoint\` | 321,065,050 |

合计约 1.74G，五个文件字节数均与云端一致。

## 4. `stage-epoch-30.pth` 一致性检查（未传输依据）

对 R1/R2 四个 run 分别比较 `last.pth` 与 `stage-epoch-30.pth` 的 PyTorch-zip 条目（去掉保存点前缀后按 size+CRC 比对）：

| run | 条目数 | CRC 相同 | 差异条目 |
|---|---:|---:|---|
| `sun-dev-original-seed12345/20261010-013324` | 2863 / 2863 | 2861 | `data.pkl`、`.data/serialization_id` |
| `sun-dev-odg-seed12345/20261010-020527` | 2863 / 2863 | 2861 | 同上 |
| `sun-dev-original-seed12345-r2/20261010-094231` | 2950 / 2950 | 2948 | 同上 |
| `sun-dev-odg-seed12345-r2/20261010-094236` | 2950 / 2950 | 2948 | 同上 |

即全部模型/优化器张量逐字节相同，仅序列化元数据不同 → **`stage-epoch-30.pth` 与 `last.pth` 为同一训练状态**，按用户要求未取回，节省约 1.3G 传输。云端 4 个 `stage-epoch-30.pth` 本轮未删（B 类原件保留）。

## 5. OC-RGA DeLiVER 数据上传与无卡环境

### 5.1 已执行

- **修复中断的 `semantic.tar`**：云端原为 264,929,280 B（`tarfile` 报 `unexpected end of data`，属不完整包），已用本地 `D:\0Project\dataset\_ocrga_transfer\semantic.tar`（274,787,840 B）重传覆盖；复核 6598 条目可正常解析。
- **解压**：`depth.tar` 与 `semantic.tar` 已解压到 `/root/rivermind-data/dataset/DELIVER/`，`depth/` 与 `semantic/` 各 5988 个文件，`DELIVER/` 合计 587M。结构与包内一致：`DELIVER/{depth,semantic}/{cloud,fog,night,rain,sun}/{train,val}/...`。

### 5.2 现状与待确认

| 项目 | 现状 |
|---|---|
| `DELIVER/img/` | 空（仅 5 个天气子目录）；本地 `D:\0Project\dataset\DELIVER\img` 为 8,962 MB / 7,885 文件，尚未上传 |
| `DELIVER/{depth,semantic}` 划分 | 包内仅含 `train`+`val`；本地另有 `test`（如 `depth/cloud/test` 379 文件），未包含在包中 |
| 云端代码 | HEAD `a11b835`（2026-10-10，无 OC-RGA 代码）；本地 HEAD `ce24e10`（2026-10-14 `feat(oc-rga)`），**尚未同步** |
| 无卡环境 | Python 3.10.16 / torch 2.1.2+cu118 / mmcv 2.1.0 / timm 1.0.28 / swanlab 0.10.1，与项目记录一致 |
| 配置 | 云端 `local_configs/research/` 仅有 `DFormerv2_S_{DeLiVER,SUNRGBD}.py`、`ODG_{NYUv2,SUNRGBD}.py`；`research/splits/` 有 `deliver_official`、`sunrgbd_seed12345` |

需要用户决定：① 是否上传 `img`（约 8.8G，需约 9G 磁盘与较长传输）；② 是否补传含 `test` 划分的包；③ 代码同步方式（推送 origin 后云端 pull，或 Git bundle）。

## 6. 云端剩余空间与未处理风险

- 可用 **20G**（61%），本轮后仍足以容纳 img（约 9G）与解压开销，但余量不宽裕。
- 未处理（D 类，等下一轮）：`cloud/mmfr-a2-v3-probe/` 5.4G、`DFormer-archive-20261008/cloud/` 3.1G、`natural-missing-preflight` 0.9G、`mmfr-e1-batch1a-v1` 中间 checkpoint 3.2G 等。
- **本地单点副本风险**：`cloud/MMFR_E1_Batch1A_local_transfer_20260922/` 云端已删，本地 `D:\0Project\DFormer-archive-20261007\cloud\MMFR_E1_Batch1A_local_transfer_20260922\`（618M）现为该资产的唯一副本，后续本地整理不得再删。
- 数据一致性提示：`DELIVER/{depth,semantic}` 云端为 5988 文件/模态，与本地完整集合（7,885 文件/模态）不同，差异在 `test` 划分，非传输缺陷。
