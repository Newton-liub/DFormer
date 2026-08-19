# MUSeg / DFormerv2 MVE 执行总计划

> 计划类型：阶段执行清单  
> 适用范围：A1、B1、A2 筛查，以及条件触发后的 B2 方案验证  
> 状态基准：2026-08-19 现有仓库与项目文档  
> 目标：用最低工程成本确认“全背景样本数值风险”和“Depth=0 是否构成实际性能瓶颈”，再决定是否进入正式训练与模型改造。

---

## 一、状态标记与执行原则

- `~~[x] 已完成~~`：已有代码、数据、文档或验证证据支持，不重复执行。
- `[ ] 未执行`：尚未完成，后续需要执行。
- `[>] 当前阶段`：当前应优先处理的事项。
- `阻断`：前置条件未满足时，不进入下一阶段。
- 所有实验必须记录：代码 commit、配置、checkpoint、seed、样本 manifest、数据版本和输出位置。
- 数据集、checkpoint、预测结果和训练日志不上传 GitHub；GitHub 只保存代码、配置、manifest 和结果摘要。

---

## 二、总览结论

### 已完成的项目基础

- ~~[x] MUSeg 原始数据到 DFormer 目录的确定性转换~~
- ~~[x] 3171 组数据的四模态、尺寸、文件配对和深度映射验证~~
- ~~[x] 官方 train/test 划分验证：1595/1576~~
- ~~[x] 原始标签语义确认：0=background，1–15 为 15 个前景类别~~
- ~~[x] 训练标签映射确认：0→255(ignore)，1–15→0–14~~
- ~~[x] 原始 Depth=0 的无效/缺失深度语义确认~~
- ~~[x] 数据转换脚本、原子输出、dataset_meta.json 和覆盖保护完成~~
- ~~[x] 云端数据重建和 Git 数据保护指南完成~~
- ~~[x] A1/A2 的最小验证设计、成功判据和停止判据完成~~

### 尚未完成的核心事项

- `[ ] MUSeg 专属正式训练配置`
- `[ ] A1：全 ignore loss 风险复现`
- `[ ] B1：safe masked loss 实现与等价性验证`
- `[ ] A2：Depth=0 受控注入筛查`
- `[ ] B2：validity mask / geometry prior gating`
- `[ ] MUSeg 基线训练与正式性能报告`

---

# 三、阶段 0：已完成基础工作，不重复执行

## 0.1 数据转换与数据证据

- ~~[x] 确认原始数据位于 MUSeg 六矿区目录~~
- ~~[x] 新增并执行 `tools/prepare_museg.py`~~
- ~~[x] 保留原始 16-bit 深度到 `Depth16/`~~
- ~~[x] 使用固定全局公式生成 8-bit `Depth/`~~
- ~~[x] 固定全局最大值 13932，禁止逐图归一化~~
- ~~[x] 完成 3171 组样本转换~~
- ~~[x] 完成 RGB、Depth、Depth16、Label 数量和尺寸验证~~
- ~~[x] 完成逐像素深度映射验证~~
- ~~[x] 完成官方 train/test 划分和 SHA-256 元数据记录~~
- ~~[x] 确认 11 张全背景图，其中 train 5 张、test 6 张~~

证据文件：

- `tools/prepare_museg.py`
- `doc/reports/2026-08-19-museg-data-reproduction-update.md`
- `doc/museg-dformer-data-processing-review.md`
- `doc/云/MUSeg云端重建指南.md`

## 0.2 当前状态边界

- ~~[x] 明确当前没有完成模型训练和性能实验~~
- ~~[x] 明确当前没有完成 MUSeg 专属训练配置~~
- ~~[x] 明确当前 Depth=0 仍直接进入 DFormerv2 geometry prior~~
- ~~[x] 明确当前 loss 对空有效像素集合直接求 mean 存在风险~~

---

# 四、阶段 1：Windows 本地完成 A1 + B1

> 资源：CPU 即可。  
> 目的：先关闭确定性的 loss 数值风险，不等待云端 GPU。

## 1.1 A1：复现全 ignore 风险

- `[>] 在随机 logits 上复用项目同款 CrossEntropyLoss`
- `[ ] 覆盖有效像素比例：100%、1%、1 pixel、0%`
- `[ ] 记录 loss 是否 finite`
- `[ ] 调用 backward，记录 gradient 是否全部 finite`
- `[ ] 分别验证主头和 aux head 的归约路径`
- `[ ] 验证 0% 条件下旧实现是否出现 NaN 或非有限值`
- `[ ] 验证 1 pixel、1% 和 100% 条件仍然 finite`
- `[ ] 用 rank 0 全 ignore、rank 1 1 pixel 的最小模拟确认本地 rank 语义`

成功门槛：

- 0% 条件触发非有限 loss 或 gradient；
- 非空条件保持 finite；
- 若 0% 也 finite，则核对当前代码版本并停止针对该风险的修改。

## 1.2 B1：safe masked loss

- `[ ] 抽取单一 safe masked mean 函数`
- `[ ] 有效像素存在时返回原始 masked mean`
- `[ ] 无有效像素时返回与计算图相连的零值 loss`
- `[ ] 比较旧实现和新实现的非空 loss`
- `[ ] 比较旧实现和新实现的非空 logits gradient`
- `[ ] 验证主头和 aux head`
- `[ ] 验证全 ignore 条件下新 loss 为有限 0`
- `[ ] 验证全 ignore 条件下新 gradient finite`
- `[ ] 验证“1 张全背景图 + 1 张普通图”的混合 batch`
- `[ ] 将 A1/B1 固化为可重复测试，避免只保留临时 REPL 结果`

成功门槛：

- 非空条件新旧 loss 与 gradient 最大绝对差 `<1e-6`；
- 全 ignore 条件新 loss 为有限 0；
- 主头和 aux head 均通过。

建议代码位置：

```text
tests/test_masked_loss.py
models/losses/ 或 models/builder.py
```

## 1.3 本地验证与提交

- `[ ] 在 Windows 本地运行 A1/B1`
- `[ ] 运行 Python 语法检查`
- `[ ] 检查最近修改文件的 linter`
- `[ ] 检查 Git diff，确认没有数据、checkpoint、日志进入提交`
- `[ ] 提交 A1/B1 代码和测试`
- `[ ] 记录 commit SHA，作为后续云端实验代码版本`

阻断条件：B1 未通过前，不启动正式训练，也不把 NaN 风险带到云端训练。

---

# 五、阶段 2：Windows 本地准备 A2

> 资源：CPU 可完成数据处理；不要求本地 Windows 跑完整 DFormerv2 推理。  
> 目的：把云端 GPU 实验所需输入固定下来，减少 GPU 空转时间。

## 2.1 固定筛查样本

- `[ ] 从官方 test 固定筛选 16 张含前景且原始有效深度比例较高的图`
- `[ ] 保存样本 ID、原始有效深度比例和筛选规则`
- `[ ] 固定 corruption seed`
- `[ ] 固定 q={0, 0.3, 0.5}`
- `[ ] 先只生成 block mask，不立即扩展 random mask`

建议 manifest：

```text
experiments/museg_mve/manifests/a2_screening_16.json
```

## 2.2 生成和验证 mask

- `[ ] 只在原始有效深度像素上注入额外 0`
- `[ ] 生成每张图的 block mask`
- `[ ] 保存 mask 的 seed、目标缺失率和实际缺失率`
- `[ ] 验证 mask 使用 nearest 几何缩放或保持原始尺寸`
- `[ ] 验证 RGB、Depth、Depth16、Label、mask 尺寸一致`
- `[ ] 生成 q=0、q=0.3、q=0.5 的输入目录或可复现索引`
- `[ ] 用中位数深度填充生成负对照的规则，但暂不运行完整负对照`

建议脚本位置：

```text
tools/mve/a2_prepare_masks.py
tools/mve/a2_evaluate_results.py
```

## 2.3 A2 运行脚本准备

- `[ ] 将数据根目录、checkpoint 路径、输出目录改为命令行参数`
- `[ ] 将 seed、q、mask 类型和样本 manifest 改为命令行参数`
- `[ ] 输出每图预测结果和汇总 CSV`
- `[ ] 记录代码 commit、checkpoint、数据 meta、输入尺寸和 depth policy`
- `[ ] 确认 Windows 只负责脚本和数据准备，不强求完整模型 CPU 推理`
- `[ ] 完成 Python 语法检查和小尺寸输入 smoke test`

建议脚本位置：

```text
tools/mve/a2_run_sensitivity.py
tools/mve/a2_evaluate_results.py
```

---

# 六、阶段 3：GitHub 与云端 4090 准备

> 原则：GitHub 传代码，持久化磁盘或对象存储传数据和 checkpoint。

## 3.1 本地上传前检查

- `[ ] 确认 A1/B1 已通过`
- `[ ] 确认 A2 manifest 和 mask 生成逻辑固定`
- `[ ] 确认数据集、checkpoint、预测结果和日志被 Git 忽略`
- `[ ] 确认项目中没有密钥、私有地址或本机绝对路径`
- `[ ] 提交并推送代码`
- `[ ] 记录目标分支和 commit SHA`

## 3.2 4090 实例环境

- `[ ] 创建或选择 CUDA 与 PyTorch 兼容的 Linux 镜像`
- `[ ] 确认 `nvidia-smi` 正常`
- `[ ] 确认 `torch.cuda.is_available()` 为 True`
- `[ ] 确认 PyTorch、CUDA、mmcv/mmengine、timm 版本`
- `[ ] 从 GitHub clone 或 checkout 固定 commit`
- `[ ] 不直接迁移 Windows venv/conda 环境`
- `[ ] 在云端重新安装与镜像兼容的依赖`
- `[ ] 记录 Python、PyTorch、CUDA、GPU 和依赖版本`

## 3.3 云端数据与 checkpoint

- ~~[x] 已有云端 MUSeg 数据重建操作指南~~
- `[ ] 挂载或下载原始 MUSeg 数据`
- `[ ] 保留原始数据，不删除备份`
- `[ ] 使用同一个 tools/prepare_museg.py 重建或确认 MUSeg_DFormer`
- `[ ] 校验 dataset_meta.json`
- `[ ] 准备 DFormerv2-S checkpoint`
- `[ ] 校验 checkpoint 文件大小和 SHA-256`
- `[ ] 确认输出目录位于持久化磁盘`

---

# 七、阶段 4：A2 最小筛查版

> 第一轮只做 16 张图、3 个 q、block mask，共 48 次前向。  
> 不先做 64 张图、random mask 或完整微调。

## 4.1 基线检查

- `[ ] 使用固定 checkpoint 运行 q=0`
- `[ ] 确认模型可以正常加载`
- `[ ] 确认 RGB、Depth、Label 预处理正确`
- `[ ] 确认预测输出尺寸正确`
- `[ ] 确认 q=0 结果可重复`
- `[ ] 保存 q=0 的 per-image 预测和 mIoU`

## 4.2 受控缺失实验

- `[ ] 运行 q=0.3 block`
- `[ ] 运行 q=0.5 block`
- `[ ] 保存每图 mIoU`
- `[ ] 保存前景 mIoU`
- `[ ] 计算同图配对差值 ΔmIoU(q)=mIoU(q)-mIoU(0)`
- `[ ] 计算缺失区域外扩 5 像素边界带 mIoU`
- `[ ] 记录 wall-clock、显存、异常和 NaN 状态`
- `[ ] 生成一页筛查摘要`

## 4.3 A2 决策门槛

### 进入扩大实验（A2-G）

同时满足以下条件才扩大：

- `[ ] q=0.3 block 相对 q=0 有明显前景 mIoU 下降`
- `[ ] 至少大多数图像的配对差值为负`
- `[ ] q=0 → 0.3 → 0.5 总体没有反向上升趋势`
- `[ ] 退化主要出现在 mask 区域或边界附近`

### 停止或降级（A2-N-G）

满足以下任一情况则不进入 B2：

- `[ ] q=0.5 仍几乎没有退化`
- `[ ] 不同 q 没有方向一致趋势`
- `[ ] 结果只在极少数异常样本上退化`
- `[ ] 退化无法与 mask 区域或边界建立关系`

此时保留 A2 结果，暂不投入复杂 validity mask；优先回到基线配置、预处理或其他瓶颈。

## 4.4 A2-G 后的扩大实验

- `[ ] 扩大到 64 张固定 test 图`
- `[ ] 增加 q={0, 0.1, 0.3, 0.5}`
- `[ ] 加入 random mask`
- `[ ] 加入中位数填充负对照`
- `[ ] 完成配对曲线、边界带指标和结果可视化`
- `[ ] 形成 A2 正式结论`

---

# 八、阶段 5：B2 validity mask 方案验证

> 只有 A2 证明 Depth=0 存在稳定退化后才执行。

## 5.1 B2 代码方案

- `[ ] 从 Depth16 > 0 生成原始 validity mask`
- `[ ] RGB、Depth、Label、mask 使用同步几何增强`
- `[ ] mask resize 使用 nearest`
- `[ ] 每个 stage 将 mask 对齐到 depth patch 网格`
- `[ ] 构造 pair_valid = valid_i & valid_j`
- `[ ] 有效 pair 保留 depth decay`
- `[ ] 无效 pair 仅保留 positional decay`
- `[ ] 不增加可学习参数`
- `[ ] 不修改 decoder`
- `[ ] 不把无效深度对应 Label 改为 ignore`
- `[ ] 不用深度补全伪造观测值`

## 5.2 B2 零训练 A/B

- `[ ] 使用与 A2 完全相同的 16/64 张图和 mask`
- `[ ] 固定同一个 checkpoint`
- `[ ] 比较 B0 原始实现与 B2 gating`
- `[ ] 比较 q=0 的干净样本性能`
- `[ ] 比较 q=0.3 block 性能`
- `[ ] 比较自然高无效率与低无效率样本`
- `[ ] 检查预测变化、mIoU 和边界带 mIoU`

进入微调门槛：

- `[ ] q=0.3 block 至少恢复 1.0 个 mIoU 百分点`
- `[ ] q=0 干净样本下降不超过 0.5 个百分点`

## 5.3 B2 最小微调

只有零训练 A/B 有正向信号时执行：

- `[ ] 固定同一初始化 checkpoint`
- `[ ] 固定同一 20% group-disjoint train 子集`
- `[ ] B0/B2 使用同一 seed`
- `[ ] B0/B2 使用同一 5 epoch 配置`
- `[ ] 比较自然高无效率组前景 mIoU`
- `[ ] 比较 A2 corruption sweep`
- `[ ] 比较完整官方 test`
- `[ ] 保存训练曲线、最佳 checkpoint 和环境记录`

最终成功门槛：

- `[ ] 自然高无效率组前景 mIoU 提升至少 1.0 个百分点，或 q=0.3 block 提升至少 2.0 个百分点`
- `[ ] 完整 test 前景 mIoU 不下降超过 0.5 个百分点`

失败则：

- `[ ] 停止复杂 mask 开发`
- `[ ] 只保留 B0 基线和失败证据`
- `[ ] 必要时仅做简单深度填补对照`

---

# 九、阶段 6：MVE 之后的正式研究工作

以下工作不属于当前最小验证，必须等 A1/A2 决策后再排期：

- `[ ] 新增完整 MUSeg dataset config`
- `[ ] 按官方采集组从 train 生成 group-disjoint validation`
- `[ ] 训练并记录 B0 MUSeg baseline`
- `[ ] 明确 RGB/BGR、resize、sliding、归一化和 depth policy`
- `[ ] 完整 test 评估`
- `[ ] 至少 3-seed 正式实验`
- `[ ] 自然无效率分层分析`
- `[ ] 深度填补、validity mask 和其他方案消融`
- `[ ] 形成正式实验报告或论文结果表`

---

# 十、建议的仓库结构

```text
DFormer/
├── models/
│   └── losses/
├── tests/
│   └── test_masked_loss.py              # A1/B1 单元与回归测试
├── tools/
│   ├── prepare_museg.py                  # 已完成的数据转换
│   └── mve/
│       ├── a2_prepare_masks.py           # Windows 本地准备
│       ├── a2_run_sensitivity.py         # 4090 推理
│       └── a2_evaluate_results.py        # 结果汇总
├── local_configs/
│   └── MUSeg/
│       └── DFormerv2_S_MVE.py            # 尚未创建
└── experiments/
    └── museg_mve/
        └── manifests/
            └── a2_screening_16.json      # 尚未创建

../dataset/
├── MUSeg/                                # 原始数据，不入 Git
└── MUSeg_DFormer/                        # 转换数据，不入 Git
```

---

# 十一、当前下一步：只执行这 5 件事

1. `[>] 在 Windows 本地实现并运行 A1/B1 测试`
2. `[ ] 记录 A1/B1 的 loss、gradient 和 finite 结果`
3. `[ ] 提交 A1/B1 代码与测试，记录 commit SHA`
4. `[ ] 在 Windows 本地完成 A2 的 16 张样本 manifest 和 block mask 生成`
5. `[ ] 准备云端 4090 的固定 commit、数据、checkpoint 和环境检查清单`

**不要在 A1/B1 和 A2 筛查版完成前做以下事情：**

- 不进行完整 MUSeg 训练；
- 不运行 64 张图的完整 sweep；
- 不实现复杂 B2 validity mask；
- 不进行 5 epoch B0/B2 微调；
- 不把数据集和 checkpoint 上传 GitHub；
- 不把 Windows 虚拟环境直接迁移到 Linux 云端。

---

## 关联文件

- [最小验证路径设计](../reports/2026-08-19-museg-minimum-validation-paths.md)
- [数据复现阶段汇报](../reports/2026-08-19-museg-data-reproduction-update.md)
- [数据处理评审](../museg-dformer-data-processing-review.md)
- [云端重建指南](../云/MUSeg云端重建指南.md)