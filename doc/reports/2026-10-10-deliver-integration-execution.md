# DeLiVER RGB-D 接入执行报告

日期：2026-10-10。执行模型交付，待主力模型验收。工作区：独立 Git worktree
`D:\0Project\DFormer-deliver-worktree`，分支 `research/deliver-integration`，基线提交
`5c082ba`（主线工作目录未改动，未合并、未推送）。

权威依据为 [DeLiVER 接入计划](../plans/2026-10-10-deliver-integration.md) 与上级模型裁决（RGB + 原始单通道
`depth/`，保留官方 train/val/test 与 25 类协议，复用现有研究入口，独立 worktree）。本报告只记录实际执行与产物，
不做科研判断。

执行期间主线继续推进：本分支从稳定提交 `5c082ba` 建立，主线随后提交了
`7d06f05 fix(odg): restore training decoder BN config in the evaluation entry`（`research/odg_schedule.py`
+36、`research/evaluate_odg.py` +18）。`git merge-tree --write-tree research/deliver-integration 7d06f05`
返回空冲突集（退出码 0），说明本分支与当前主线合并无冲突：主线新增的是解码器 BN 恢复辅助函数，本分支只改
`import_data_module` 与两个 loader factory，区域不重叠。按裁决，本分支保持基线不变，合并与最终接线由主力模型决定。

## 1. 实际修改文件

| 文件 | 类型 | 说明 |
| --- | --- | --- |
| `research/deliver.py` | 新增 | `DeLiVERDataset`（RGB + `depth/` + 红通道 ID 标签）、路径/标签映射辅助、`complete_optimizer_groups`、`prepare`/`preview`/`smoke` 三个子命令 |
| `local_configs/research/DFormerv2_S_DeLiVER.py` | 新增 | 独立配置：25 类、`data_module="research.deliver"`、`pad=False`、官方三清单 |
| `research/splits/deliver_official/{train,val,test}.txt` | 新增 | 相对 `img/` 的官方划分清单，3983/2005/1897 行 |
| `research/splits/deliver_official/README.md` | 新增 | 清单来源、模态派生规则与逐 case 计数 |
| `research/odg_schedule.py` | 小改 | `import_data_module(config)` 读取可选 `config.data_module`（缺省仍 `research.data`）；两个 loader factory 显式传入 config |
| `research/train_odg.py` | 小改 | 仅当 `dataset_name=="DeLiVER"` 时调用补齐函数；`--print-schedule` 增打 train/eval 来源 |

共享文件合计 29 行新增、6 行删除（`git diff --stat`）。未改动作者 `utils/`、模型主干、`research/data.py`、
`research/evaluate_odg.py` 与任何 SUN/NYU 配置。

## 2. 输入与标签合同（已实现）

- 清单每行是相对 `img/` 的路径，如 `cloud/train/MAP_1_point102/110100_rgb_front.png`；Depth/semantic 由同一
  路径替换模态根目录与文件名中的 `_rgb`（`_depth` / `_semantic`），用 `Path` 拼接。
- RGB 显式按 `COLOR_BGR2RGB` 解码为三通道，记为 `rgb_order=RGB`。
- Depth 用 `IMREAD_UNCHANGED` 保留原生位深；抽样为单通道 uint8。非单通道或非 uint8 直接报错，不做静默截位或
  取任意通道。按作者规则归一化 $d=(D_8/255-0.48)/0.28$（域约 $[-1.714286, 1.857143]$），再复制为三通道。
- 标签读原图红通道（OpenCV 三/四通道的索引 2），按官方规则显式映射：$1..25 \to 0..24$，$0$ 与 $255 \to 255$
  （ignore）。未知 ID 抛错并报出样本，不静默吞成 ignore。绝不 `IMREAD_GRAYSCALE`，也不套调色板匹配。
- 类别顺序与 25 色调色板取自官方 loader `semseg/datasets/deliver.py` 的 `CLASSES` / `PALETTE`。
- 支持域为有限像素域（原生全 1），经训练 crop/pad 后 padding 处为 0；`pad=False`，不使用 SUN 的 531×730。
- 输出沿用 `data/label/modal_x/depth_support/fn/n` 键集与 dtypes（float32 / int64），因此
  `research/odg_schedule.py` 的 loader factory、`ObservationTrainPre/ObservationValPre`、`evaluate_loader` 与
  `SegMetrics` 直接复用；`research/deliver.py` 对外暴露 `ObservationDataset` 别名。

## 3. 关键执行结果

全部在本地 CPU（`CUDA_VISIBLE_DEVICES=-1`）运行，未启动 GPU、未开始训练。

### 3.1 清单生成（`prepare`，对 `img/` 单遍枚举）

```
[prepare] train total=3983  -> research\splits\deliver_official\train.txt
[prepare] val   total=2005  -> research\splits\deliver_official\val.txt
[prepare] test  total=1897  -> research\splits\deliver_official\test.txt
```

三个总数与交接记录 3983/2005/1897 完全一致；含 `result: unchanged` 的重复运行，说明写入口是幂等的。
逐 case 计数（train/val/test）：`clean` 2585/1298/1198，`motionblur` 600/299/300，
`underexposure` 199/99/100，`overexposure` 200/100/100，`lidarjitter` 199/100/99，
`eventlowres` 200/109/100。

### 3.2 样本读取与可视化（`preview --limit 6`）

6 个样本覆盖 train/val/test、cloud/fog/night/rain/sun、以及一个非空相机故障子集：

| split | weather | case | 尺寸 | Depth | 原生标签 ID | ignore 占比 |
| --- | --- | --- | --- | --- | --- | --- |
| train | cloud | clean | 1042×1042 | uint8, 1..255 | 1..25 子集 | 0.000 |
| train | fog | clean | 1042×1042 | uint8, 1..255 | 1..25 子集 | 0.000 |
| train | night | motionblur | 1042×1042 | uint8, 1..255 | 1..25 子集 | 0.000 |
| train | rain | underexposure | 1042×1042 | uint8, 1..255 | 1..25 子集 | 0.000 |
| val | sun | clean | 1042×1042 | uint8, 1..255 | 1..25 子集 | 0.000 |
| test | cloud | clean | 1042×1042 | uint8, 1..255 | 1..25 子集 | 0.000 |

每个样本都断言了三模态存在、尺寸一致、原生 dtype/通道、标签取值域。映射小数组断言：
`[1, 25, 0, 255] -> [0, 24, 255, 255]`，且未知 ID（26）被拒绝。

预处理断言：训练 crop 输出 480×480（RGB/Depth/标签/support 形状与 dtype 全部符合），
val 保持原生 1042×1042、support 全为 1（`pad=False` 无 padding 带）。

3 张可视化（`outputs/deliver-preparation/preview-*.png`）为 4 联图：RGB、`depth/` 灰度（归一化域）、按官方
调色板着色的 GT、RGB+GT 叠图。抽查 train/cloud 与 train/night+motionblur：道路、车辆、天空、植被、路侧边界
在 RGB、深度与 GT 三视图上位置对应；night+motionblur 样本 RGB 明显变暗且带拖影，而 `depth/` 仍清晰，符合
“该故障作用于 RGB 相机、不作用于本轮不使用的 HHA”的预期。

### 3.3 CPU 日程（`--print-schedule`，只运行一次）

```
[schedule] dataset=DeLiVER geometry=original
[schedule] train_source=...\deliver_official\train.txt
[schedule] eval_source=...\deliver_official\val.txt (periodic validation target)
[schedule] samples/epoch=3983 micro_batch=1 accum_steps=16 effective_batch=16
[schedule] micro_batches/epoch=3984 updates/epoch=249 repeated_samples=1 file_length=3984
[schedule] schedule_epochs=300 warmup_epochs=10 total_updates=74700 warmup_updates=2490
```

train 计数来自实际清单，周期评价目标是官方 val（不是 test）。此命令不创建训练会话。

### 3.4 一次模型 smoke（CPU，128×128，`original` 模式）

```
[smoke] logits=[1, 25, 128, 128] loss=4.005939 finite grads=True trainable=714 with_grad=714
[smoke] optimizer groups 2(685 params) -> 3(714 params); added 29 ungrouped tensors, 29 of them Geo.weight
[smoke] after one AdamW step 29/29 previously-ungrouped tensors changed; metrics miou=12.0 (expected 12.0)
```

- 真实 train 样本、同步缩放到 128×128，仅 `original` 模式；加载官方 encoder（`missing` 为 `extra_norms` 初值、
  `unexpected` 为不使用的预训练头，与既有 SUN 记录一致）。
- 输出 25 通道；loss 有限；714 个可训练参数张量全部拿到有限梯度（`no_grad` 为 0）。
- **优化器漏参修正生效**：作者 `group_weight` 只得到 685 个参数，补齐后 714 个；新增的 29 个恰好全部是
  `Geo.weight`，一次 AdamW 更新后 29/29 全部发生变化。这正是首轮 SUN 实验中遗漏的同一批参数。
- `SegMetrics`：混淆矩阵 25×25，注入一个 ignore 像素后 hist 合计为 3（ignore 未计分），mIoU 等于预期的
  12.0（3/25）。仅验证流程，不是性能。
- 产物 `outputs/deliver-preparation/smoke-report.json`，`resumable_checkpoint_written=false`，未写任何
  可恢复为正式训练的 checkpoint。

### 3.5 入口接通与隔离复核

- `import_data_module(cfg)` 返回 `research.deliver`；无参调用仍返回 `research.data`。
- `build_train_loader` / `build_eval_loader` 端到端通过：train 批次
  `(1,3,480,480)/(1,480,480)/(1,3,480,480)/(1,1,480,480)`，val 批次原生 1042×1042、2005 张、support 全 1。
  `evaluate_odg` 使用的正是这两个 factory，且 `--split dev` 读取 `dev_eval_source`（= `val.txt`）、
  `--split test` 读取 `full_test_source`（= `test.txt`）。
- 配置隔离：DeLiVER 25 类 / `pad=False` / `dataset=DeLiVER` / 路径指向 `DELIVER`；SUN 配置仍 37 类、无
  `data_module` 键、`gt_transform=True`。
- SUN 回归：同一 worktree 内 `ODG_SUNRGBD --print-schedule` 输出 4757 样本、298 更新/epoch、总 89400、
  warmup 2980、`repeated_samples=11`、`file_length=4768`，与主线已记录数值完全一致，说明共享文件改动未改变
  SUN/NYU 行为。
- `prepare` 的安全路径：`--case` 过滤实测（motionblur 600/299/300，overexposure 200/100/100），空子集报错退出；
  对内容不同的同名清单拒绝覆盖并保留原文件。这两项在 `outputs/` 下的临时目录中验证，验证后已删除。

## 4. 可运行命令

以下命令在 worktree 根目录、用现有 `dformer` 环境执行；本轮**未**运行训练与 test。

```powershell
$env:DFORMER_DATASET_ROOT = 'D:\0Project\dataset'          # 或指向 DELIVER 的父目录
# checkpoints/ 被 .gitignore 忽略，全新 worktree/检出没有它；用绝对路径覆盖即可
$env:DFORMER_PRETRAINED   = 'D:\0Project\DFormer\checkpoints\pretrained\DFormerv2_Small_pretrained.pth'
$py = 'D:\2Env\anaconda\envs\dformer\python.exe'
# 清单（幂等；数据不移动、不重划分）
& $py -X utf8 -m research.deliver prepare --root D:\0Project\dataset\DELIVER --split-dir research/splits/deliver_official
# 周期性评价只读官方 val；--split dev 即 val
& $py -X utf8 -m research.train_odg --config local_configs.research.DFormerv2_S_DeLiVER --geometry-mode original --micro-batch <M> --accum-steps <A>
# 全量 train 时加 --fulltrain（关闭训练内周期验证，避免 test 参与选点）
& $py -X utf8 -m research.train_odg --config local_configs.research.DFormerv2_S_DeLiVER --geometry-mode original --fulltrain --micro-batch <M> --accum-steps <A>
# dev（=官方 val）评价，单尺度无翻转
& $py -X utf8 -m research.evaluate_odg --config local_configs.research.DFormerv2_S_DeLiVER --checkpoint <run>/last.pth --split dev --no-pad_SUNRGBD
# 最终 test（仅在模型与协议冻结后）
& $py -X utf8 -m research.evaluate_odg --config local_configs.research.DFormerv2_S_DeLiVER --checkpoint <run>/best-dev.pth --split test --no-pad_SUNRGBD
```

约束：`--micro-batch` × `--accum-steps` 必须等于 `effective_batch=16`；训练入口仅支持单卡 CUDA，CPU 上只允许
`--help` / `--print-schedule`。正式训练前需要在获批 GPU 上用所选 crop/batch 做一次显存与吞吐定点检查，本轮
没有做。

## 5. 尚未解决的实际障碍 / 未运行项

1. **无 GPU 验证**：没有运行任何真实训练、没有测量 1042×1042 原生整图评价的显存与吞吐。原图分辨率（1042）比
   480 crop 大，正式评价若 OOM，按计划先降 eval batch，仍不行再单独提出固定推理 resize 协议，不静默更改。
2. **`evaluate_odg` 未端到端跑通**：本地没有训练好的 DeLiVER checkpoint，因此只验证了它依赖的 loader 路径、
   划分映射与指标组件；checkpoint 载入之后的流程与 SUN 主线共用同一段代码。
3. **test 未计分**：本轮只读 1 张 test 作格式检查，不产生任何 test 数值，也不作为调参信号。
4. **Depth 物理语义未知**：`depth/` 的单通道 uint8 图像表示可用，但其米制单位与天然无效值约定仍未知；读到的
   raw 0 保留为观测，不用 `tensor==0` 推断缺失，不声称米制几何。
5. **不等同官方协议**：官方 `depth` 模态实际读取 `hha/`，本接入读 `depth/`。因此本表示下的数值不与官方 HHA
   RGB-D 分数可比，配置与产物中均已显式标记。
6. **抽样而非全量**：只读了 6 个样本，未做全量数据审计；读到的样本 ignore 占比为 0，`0/255` 的映射用合成小数组
   断言覆盖。
7. **本分支基线较主线落后一个提交**：本分支基于 `5c082ba`，主线在其后提交了解码器 BN 恢复修复 `7d06f05`。
   `research/evaluate_odg.py` 与 `research/odg_schedule.py` 在主线已有该改动，本分支不含。合并无冲突（见开头），
   但合并后应确认评价入口使用主线版本，避免退回无 BN 恢复的旧行为。
8. **未推送远端**：只在 `research/deliver-integration` 分支做本地提交，未合并主线、未推送。
9. **清单换行符与跨平台指纹**：`prepare` 以 LF 写清单，Git 索引中也存为 LF，但本仓 `core.autocrlf=true` 且
   无 `.gitattributes`，因此在 Windows 上重新 checkout 后工作区副本会变成 CRLF。`file_fingerprint` 哈希原始
   字节，所以跨 Windows/Linux 续训时清单指纹可能不一致而拒绝 resume。SUN 清单目前就是这种状态（索引 LF、
   工作区 CRLF 21950 字节 vs 索引 21422 字节），本轮未改动该全局行为，只在此记录。
10. **编码器预训练权重不在版本控制内**：`checkpoints/` 被 `.gitignore` 忽略，独立 worktree 默认没有该文件。
    配置支持用 `$DFORMER_PRETRAINED` 指向绝对路径；上面 3.4 的 smoke 就是在删除本地 `checkpoints` 目录、
    仅设置 `$DFORMER_PRETRAINED` 的条件下跑通的（loss 与结果一致）。未在 worktree 内保留软链接或复制权重。

## 6. 与计划的一致性

- 只接入 RGB + `depth/`；未实现 HHA、LiDAR、Event，也未实现故障合成或人工孔洞训练。
- 官方 train/val/test 保持原样，未另切 dev、未合并 val/test。
- 复用现有 Dataset/训练/评价入口，未复制新 train/eval 脚本，未改模型主干与作者 `utils/`。
- 优化器漏参按裁决只对 DeLiVER 条件补齐，且断言全部可训练参数恰好入组一次；未改作者 `utils/init_func.py`。
- 未重新下载、移动或复制数据，未做来源验证或哈希审计，未启动云实例。
