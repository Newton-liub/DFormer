# MUSeg 数据复现与标签语义阶段工作汇报

> **历史阶段报告：** 本文冻结 2026-08-19 数据复现完成时的状态；“训练配置与无效深度处理尚未完成”只代表当时边界。后续状态与当前恢复点见 `doc/main/MUSeg-current-status.md`。

- 汇报周期：2026-08-17 至 2026-08-19
- 报告对象：研发复核、组会与云端交接
- 证据边界：旧版 MUSeg 数据处理审计之后的转换脚本实现、本地全量重建、标签语义确认与文档修订
- 当前状态：数据转换复现闭环已经完成并验证，MUSeg 训练配置与无效深度处理尚未完成

## 一、本阶段工作概述

本阶段解决了旧审计中两个关键不确定项：一是将 MUSeg 原始六矿区目录到 DFormer 输入目录的转换过程代码化；二是确认原始标签 `0` 与 `1–15` 的准确语义。当前本地已使用仓库脚本从原始数据完整重建 `MUSeg_DFormer`，3171 组输出通过四模态、位深、尺寸、官方划分、标签 ID 和逐像素深度映射验证。

标签语义已明确为：原始 `0` 是 background，原始 `1–15` 按官方顺序对应 15 个前景类别。当前基线不训练 background，因此由 `gt_transform=True` 将 `0→255`（ignore）、`1–15→0–14`（训练类别）。这一区分修正了旧报告中“background/ignore 混写”和“类别顺序仍待核验”的表述。

## 二、主要工作进展

### 2.1 确定性数据转换

- 状态：已完成并验证
- 问题：旧数据虽然已经生成过 8-bit Depth，但仓库没有可重复执行的转换脚本，本地和云端无法保证使用同一映射和划分。
- 措施：新增 `tools/prepare_museg.py`，从 `../dataset/MUSeg` 读取官方六矿区原始数据，输出到 `../dataset/MUSeg_DFormer`。
- 原理与取舍：保留原始 16-bit Depth，并固定使用全数据集映射 `round(depth16 × 255 / 13932)` 生成 8-bit Depth；禁止逐图缩放、按 split 分别缩放和直接截断位深。
- 结果与证据：本地观测原始全局最大值为 13932；脚本完成 3171 组转换，输出 train/test 为 1595/1576；转换结果通过脚本内置全量验证。

### 2.2 原子输出与复现元数据

- 状态：已完成并验证
- 问题：直接写入目标目录会在异常中断时留下半成品，也无法追溯官方划分来源。
- 措施：转换先写入临时目录，验证通过后再替换目标目录；输出 `dataset_meta.json`，记录映射公式、固定最大值、观测最大值、invalid policy、样本数量和 `DatasetSplit.zip` SHA-256。
- 结果与证据：默认拒绝覆盖非空输出，只有显式传入 `--overwrite` 才重建；覆盖保护、Python 语法和最终全量重建均已通过。

### 2.3 标签语义确认

- 状态：已完成并验证
- 问题：旧审计只能确认 Label 数值范围为 0–15，仍把类别名称顺序和 background 处理写成暂定。
- 措施：依据官方 `Label_ID.pdf` 确认原始 ID 语义，并结合 `RGBXDataset._gt_transform()` 与 `background=255` 明确训练映射。
- 结果与证据：原始 `0=background`；原始 `1–15` 对应 15 个前景类别；进入当前 DFormer 基线后 `0→255`、`1–15→0–14`。

前景类别顺序为：

1. person
2. cable
3. tube
4. indicator
5. metal fixture
6. container
7. tools & materials
8. door
9. electrical equipment
10. electronic equipment
11. mining equipment
12. anchoring equipment
13. support equipment
14. rescue equipment
15. rail area

### 2.4 文档与云端交接

- 状态：已完成并验证
- 措施：修订 `doc/dataset.md` 和 `doc/reports/2026-08-19-museg-dformer-data-processing-review.md`；当时另建云端重建临时指南，后续在数据验收完成并建立 4090 正式训练交接计划后已清理。
- 结果与证据：稳定数据规则保留在 `doc/dataset.md`，文档已明确区分原始 background 与训练 ignore，并将转换脚本状态更新为已落地。

## 三、代码与配置改动

- `tools/prepare_museg.py`
  - 收集并校验六矿区 RGB、Depth 和 Label；
  - 读取官方 `DatasetSplit.zip`；
  - 保留 `Depth16/`，生成 `Depth/`；
  - 写入 `dataset_meta.json`；
  - 提供 `verify_output()` 做全量输出验证；
  - 采用临时目录和原子替换，避免半成品。
- `doc/dataset.md`
  - 区分 `MUSeg` 原始目录和 `MUSeg_DFormer` 转换目录；
  - 固化本地/云端转换命令和深度公式；
  - 写入确定的原始标签语义与训练映射。
- `doc/reports/2026-08-19-museg-dformer-data-processing-review.md`
  - 增加 2026-08-19 状态修订；
  - 将标签结论从暂定改为已确认；
  - 将全零标签图改称全背景图；
  - 将转换脚本状态更新为已实现并验证。
- 历史云端重建临时指南（后续已清理）
  - 当时提供云端同步、重建、验收和 Git 数据保护流程；稳定规则现由 `doc/dataset.md` 承接。

## 四、验证结果

本阶段没有训练或模型性能实验，以下结果只证明数据转换与语义映射正确，不代表模型精度：

- 原始/输出样本：3171 组；
- 图像尺寸：1082×932；
- 输出 RGB、Depth、Depth16、Label：各 3171 个；
- 官方 train/test：1595/1576；
- 原始深度类型：uint16；
- 模型深度类型：uint8；
- 原始深度观测最大值：13932；
- 深度映射：`round(depth16 × 255 / 13932)`；
- 标签唯一值：0–15；
- 全量逐像素深度映射验证：通过；
- 转换脚本语法检查和覆盖保护测试：通过。

## 五、标签与深度语义结论

### 5.1 标签

原始 `0` 是 background，而不是原始 ignore 标签。当前 DFormer 基线通过 uint8 减 1 将其转换为 `255`，随后损失函数以 `ignore_index=255` 排除该像素。因此：

- 数据语义层：`0=background`；
- 训练张量层：原始 `0` 经转换后为 `255=ignore`；
- 原始 `1–15`：经转换后为训练 ID `0–14`。

11 张全部为原始 ID 0 的标签图应称为“全背景图”。其中 train 5 张、test 6 张；经过 `gt_transform` 后才成为全 ignore 图。

### 5.2 深度

原始 Depth=0 表示无效/缺失深度，当前转换保留为 0。DFormerv2 尚无 validity mask，会把该值当作普通数值参与 patch 深度差和 geometry prior。因此，标签 ID 0 的语义确认不会消除深度 0 风险；两种“0”属于不同模态，必须分别处理。

## 六、当前问题与风险

1. **MUSeg 训练配置缺失**：尚未新增 `local_configs/_base_/datasets/MUSeg.py` 和模型组合配置，当前还不能按正式配置启动 MUSeg 训练。
2. **全背景 batch 风险**：若一个本地 batch 经标签转换后没有有效像素，当前 `models/builder.py` 的空均值可能产生 NaN。
3. **无效深度风险**：约 30.74% 原始 Depth 像素为 0，当前几何先验无法区分无效值和真实数值。
4. **验证策略未定**：MUSeg 没有独立 validation；必须从官方 train 按采集组划分，不能逐图随机划分。
5. **预处理未显式化**：RGB/BGR、验证 resize/sliding 和 invalid depth policy 仍需写入配置与实验日志。

## 七、下一步计划

1. 新增 MUSeg 数据集配置，写入确定的 15 类 `class_names`、`gt_transform=True`、`background=255` 和 `MUSeg_DFormer` 路径。
2. 为全背景 batch 增加安全 masked mean，并验证单样本与分布式本地 batch 不产生 NaN。
3. 从官方 train 按采集组生成 train/validation，保持官方 test 封存。
4. 建立保留 Depth=0 的 B0 基线，再依次验证深度填补和 validity mask。
5. 显式配置 RGB/BGR 与验证尺寸，记录 checkpoint、seed、GPU、split hash 和 depth policy。

## 八、复现与交接说明

本地或云端都必须从仓库根目录调用同一脚本：

```bash
python tools/prepare_museg.py --overwrite
```

默认目录：

- 原始输入：`../dataset/MUSeg`；
- 转换输出：`../dataset/MUSeg_DFormer`。

禁止删除原始 `MUSeg`，禁止分别为 train/test 计算量化范围，禁止把数据集、转换结果或 checkpoint 加入 Git。稳定的数据重建规则见 `doc/dataset.md`。本文形成时曾指向现已删除的早期 4090 交接计划；当前事实、执行边界和恢复点只看 `doc/main/MUSeg-current-status.md`，Stage-04 历史证据见 `doc/reports/2026-08-25-museg-stage04-cloud-qualification-handoff.md`。