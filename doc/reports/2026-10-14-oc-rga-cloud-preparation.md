# OC-RGA 云端准备报告（2026-10-14）

> 实例 `cpod-1vbh7faqcauq`（无卡，GPU=0，2C/4G）。本轮**未启动 GPU、未执行完整 CPU `--check`（前向/反向/优化器）、未训练、未扩容**；仅做数据补齐、Git 同步、只读轻量检查与临时文件清理。
> 前序：[云端旧产物盘点与清理预审](2026-10-14-cloud-legacy-artifacts-inventory.md)、[第一轮清理与取回](2026-10-14-cloud-cleanup-round1.md)。

## 1. DeLiVER 数据补齐（完成）

| 模态 | 云端文件数 | 目标 |
|---|---:|---:|
| `DELIVER/img` | 5988 | 5988 |
| `DELIVER/depth` | 5988 | 5988 |
| `DELIVER/semantic` | 5988 | 5988 |

- 划分（与官方一致）：train 3983、val 2005；天气分布 train 794/795/797/799/798，val 398/400/410/398/399（cloud/fog/night/rain/sun）。
- splits：`research/splits/deliver_official/{train,val,test}.txt` = 3983 / 2005 / 1897，manifest 为 `img/` 相对路径（如 `cloud/train/MAP_1_point102/110100_rgb_front.png`）。
- 传输方式：`img` 按天气分 5 包（单包 0.84–1.62 GB），逐包"上传 → 解压 → 计数核对 → 删除云端临时包"；`img_sun` 曾在上传 49.1% 处中断，残留半包清除后重传成功。
- 样本配对核对：5 个天气 × {train,val} 的样本目录集合在 img/depth/semantic 三者间 md5 完全一致。

## 2. Git 同步（完成）

- 云端 `git fetch origin research/observation-relational-geometry` 成功（GitHub 可达，未再出现此前 TLS 中止）。
- `git checkout -B research/observation-relational-geometry FETCH_HEAD` → 云端 HEAD = `ce24e104a6b09b5a6f6e21b84664fc520a8fdc33`，工作区干净。
- 旧分支 `research/odg-sunrgbd`、`research/dformerpp-clean-start` 与 `outputs/` 等旧实验产物未受影响；未使用 Git bundle。

## 3. 轻量运行检查（完成，只读，无前向/反向/训练）

### 3.1 配置与入口导入

- `local_configs.research.OC_RGA_DeLiVER` 导入成功；
  `dataset_path=/root/rivermind-data/dataset/DELIVER`（由 `DFORMER_DATASET_ROOT` 控制），
  `rgb/x/gt` 根目录 = `.../img`、`.../depth`、`.../semantic`。
- 预训练入口 `checkpoints/pretrained/DFormerv2_Small_pretrained.pth` → `/root/rivermind-data/pretrained/DFormerv2_Small_pretrained.pth`（软链接有效，106M）。
- 输出：`checkpoint_dir = outputs/oc-rga-deliver/<run_id>`；SwanLab `enabled=True, project=dformer-research, mode=online, log_dir=<run>/swanlab`。
- `research.train_oc_rga` 导入成功：`CHECKPOINT_FORMAT=oc-rga-checkpoint-v1`，
  `ALL_MODES=(masked_baseline, oc_rga, mean_relation, diag_relation, direct_post)`。

### 3.2 训练命令与日程（`--print-schedule`，CPU 只读）

```
python -m research.train_oc_rga --config local_configs.research.OC_RGA_DeLiVER --print-schedule
```

输出要点：3983 样本/epoch；micro 4 × accum 4 = 16；996 micro-batches/epoch；**249 updates/epoch**；120 epochs；warmup 5；total_updates 29880（warmup 1245）；lr 8e-5 / poly 0.9 / 几何分支 ×2；退化混合 clean .40 / local_holes .25 / pixel_missing .15 / nonzero .15 / entire_missing .05；loss ramp λ_cond .01→.05、λ_obs .002→.010。

注意：入口须以**模块方式**运行（`python -m research.train_oc_rga`）；直接用脚本路径会 `ModuleNotFoundError: No module named 'research'`。

### 3.3 真实样本读取（train/val 各 2 个，共 4 个样本 × 3 模态）

| split | 样本 | rgb | depth | label |
|---|---|---|---|---|
| train | `cloud/train/MAP_1_point102/110100` | (1042,1042,3) uint8, 0–246 | (1042,1042) uint8, 唯一值 178 | (1042,1042,4) uint8 |
| train | `sun/train/MAP_6_point91/070700` | (1042,1042,3) uint8, 0–240 | (1042,1042) uint8, 唯一值 249 | (1042,1042,4) uint8 |
| val | `cloud/val/MAP_4_point12/157550` | (1042,1042,3) uint8, 0–255 | (1042,1042) uint8, 唯一值 221 | (1042,1042,4) uint8 |
| val | `sun/val/MAP_5_point91/138450` | (1042,1042,3) uint8, 0–237 | (1042,1042) uint8, 唯一值 203 | (1042,1042,4) uint8 |

- 路径映射：label/depth 路径由 `research.deliver.deliver_relative_paths()` 把 `_rgb` 替换为 `_semantic` / `_depth` 得到（`_read_rgb`、`_read_depth`、`map_deliver_label` 均可用并已实测）。
- 与配置声明一致：RGB 为原生 8-bit、depth 为**单通道 uint8**；label 为 4 通道 PNG，由 loader 取红通道并映射（`1..25 → 0..24`，`0`/`255 → ignore`）。

### 3.4 未执行

- 完整 `--check`（真实 micro-batch 的模型前向/反向/优化器步）按本轮边界**未运行**，避免云端 4 GB RAM 的 OOM 风险；因此**没有**模型构建/全链路运行的实测证据。

## 4. 临时文件清理（完成）

- 本地 `D:\0Project\dataset\_ocrga_transfer\imgpack\` 的 5 个 tar（约 6.5 GiB）已删除；`_ocrga_transfer` 下仅保留本地源包 `depth.tar`、`semantic.tar`。
- 云端 `dataset/depth.tar`、`dataset/semantic.tar` 已删除。
- 云端上传临时包 `_tmp_img_*.tar` 无残留；检查用临时脚本 `/tmp/oc_smoke.py` 已删除。
- 未删除：原始数据集（`DELIVER/`、`SUNRGBD/`、`MUSeg_DFormer/`）、正式权重、`pretrained/`、旧产物。

## 5. 磁盘、Git 与产物位置

- 云端磁盘：50 G，已用 37 G，**可用 14 G**（74%）。
- 云端 `dataset/` 合计 11 G：`DELIVER/`（三模态各 5988）、`MUSeg_DFormer/` 1.5 G、`SUNRGBD/` 2.4 G。
- 本地 D 盘可用约 86.7 GB。
- Git：本轮文档改动提交后推送 `origin`，云端 `research/observation-relational-geometry` 同步到同一 commit（提交清单见 `doc/state/current.md`）。

## 6. 下一阶段代码审核所需文件

| 用途 | 路径 |
|---|---|
| 概率几何核心 | `research/oc_rga.py` |
| 模型包装与优化器分组 | `research/oc_rga_model.py` |
| 数据协议（支撑标记/退化） | `research/oc_rga_data.py` |
| 联合目标与 hold-out 泄漏防护 | `research/oc_rga_losses.py` |
| 训练入口 | `research/train_oc_rga.py` |
| 评价入口 | `research/evaluate_oc_rga.py` |
| 配置 | `local_configs/research/OC_RGA_DeLiVER.py` |
| DeLiVER 数据层与路径映射 | `research/deliver.py` |
| 日程/恢复合同 | `research/odg_schedule.py` |
| 本轮环境与数据事实 | 本报告 |

云端与之同 commit（`ce24e10` 之后的本轮文档提交），路径 `root@cpod-1vbh7faqcauq:/root/rivermind-data/DFormer`。

## 7. 边界与未决

- 未启动有卡实例、未跑训练、未做完整 CPU 模型检查；GPU 侧的显存/吞吐/利用率仍未知。
- 旧产物 D 类（`cloud/mmfr-a2-v3-probe/` 等）按用户指示停止进一步清理。
- 已知实现细节：label 为 4 通道 PNG（loader 取红通道）；depth 为单通道 uint8 且物理单位未知（不在此推断）。
