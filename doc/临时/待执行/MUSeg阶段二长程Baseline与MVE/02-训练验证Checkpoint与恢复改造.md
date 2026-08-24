# 02：训练、验证、checkpoint 与恢复链路改造计划

> 任务类型：训练核心语义改造。
>
> 模型要求：**Sol 必须设计接口、checkpoint schema、验证与恢复语义，并逐行复核；Terra 只能在冻结规格下机械实现。**
>
> 原因：本任务同时影响学习率、模型选择和中断恢复，错误可能静默污染 500-epoch 结果。

## 1. 前置条件

- 01 已冻结 train-dev/val-dev/official-test 的路径、数量和哈希格式。
- 当前代码和测试基线已记录。
- 执行者先读：
  - `utils/train.py`
  - `utils/engine/engine.py`
  - `utils/dataloader/RGBXDataset.py`
  - `utils/dataloader/dataloader.py`
  - `local_configs/MUSeg/DFormerv2_S_4090.py`
  - `tests/`

## 2. 要解决的现状

- 训练期 `eval_source` 指向官方 test，造成开发期 test 泄漏。
- `is_eval()` 将验证调度与 `checkpoint_start_epoch` 耦合，不能明确配置。
- best checkpoint 名称不稳定，best metric 仅在内存中，resume 后重置。
- resume 主要恢复 model/optimizer/epoch；AMP scaler、best、global step、RNG 和协议信息不完整。
- `--epochs` 虽会重算迭代数，但缺少 poly LR 与恢复一致性测试。
- 正式 shell 尚未暴露受控 resume。

## 3. 冻结的训练角色

实现后至少支持两个明确模式：

1. `development`：train-dev 训练、val-dev 周期验证；禁止读取 official test。
2. `official`：完整 official train 训练；训练期仍不得读取 official test。其训练周期和 checkpoint 规则来自开发阶段冻结协议。

独立 test 评估由 06 的只读 evaluator 完成，不嵌入训练循环。

## 4. 配置与 CLI 规格

建议由 Sol 冻结并让 Terra 实现以下显式字段：

- `train_source`、`val_source`、`test_source`，不再用含糊的训练期 `eval_source` 表示 test。
- `experiment_phase=development|official|qualification`。
- `--epochs`、`--eval-interval`、`--eval-start-epoch`、`--save-interval`。
- `--seed`，不得再依赖 `--no-use_seed`。
- `--resume` 或兼容现有 `--continue_fpath` 的唯一规范入口。
- `--run-id`、`--output-dir`；已存在且非空时默认拒绝覆盖。
- `--expected-split-sha256` 或等效 config 字段。

兼容原则：已有 NYU/SUNRGBD 配置不能因字段缺失直接崩溃；兼容默认值必须有测试，并在 MUSeg 正式配置中全部显式设置。

## 5. 验证调度规格

- 抽取纯函数，例如 `should_evaluate(epoch, total_epochs, start, interval)`，不依赖全局 config。
- epoch 号、首个验证 epoch、周期和末 epoch语义必须在测试中枚举。
- development 推荐每 5 或 10 epochs 验证；最终间隔由 05 冻结。
- qualification 可每 epoch 验证。
- official 训练若不设内部 val，禁止回退到 test；可以只保存固定 epoch checkpoint。
- 验证失败、指标非有限或数据清单哈希变化时训练失败，不更新 best。

## 6. checkpoint schema

统一写入 `checkpoint/`：

- `latest.pth`：最近一次完整 epoch 的原子快照。
- `epoch-N.pth`：按 save interval 保存。
- `best-val-miou.pth`：development 中 val mIoU 严格提升时保存。

checkpoint 至少包含：

- schema version；
- model、optimizer、AMP scaler；
- 已完成 epoch、下一 epoch、global optimizer step；
- best val mIoU、对应 epoch、tie-break 规则；
- seed 与 Python/NumPy/PyTorch CPU/CUDA RNG state；
- 目标总 epochs、每 epoch iterations、warmup、poly power、基础 LR；
- split 路径、数量和 SHA-256；
- 完整 Git commit、run ID、phase；
- 关键配置摘要与哈希。

写入采用临时文件 + 原子替换。加载前支持 CPU `map_location` 验证；损坏或 schema 不兼容必须失败。

## 7. resume 语义

- 只允许 epoch 边界恢复；若要支持 step 中恢复需另立计划。
- 恢复后不得重复已完成 epoch，不得重置 poly LR、best metric、AMP scaler或 RNG。
- 目标 total epochs、split hash、模型、optimizer 和关键 LR 设置不兼容时拒绝恢复。
- output directory 不覆盖旧日志；SwanLab 新 run 明确记录 `resume_from` 和 parent run。
- `latest.pth` 与最高可读 `epoch-N.pth` 不一致时停止并报告，不自动猜测。

## 8. Terra 实现边界

建议修改：

- `utils/train.py`
- `utils/engine/engine.py`
- `local_configs/MUSeg/DFormerv2_S_4090.py`，或新增明确的 development/official 配置
- 对应 `tests/` 文件

Terra 不得自行：

- 改变 optimizer、LR、warmup、模型结构或数据增强；
- 决定 validation 间隔和 best tie-break；
- 删除旧 checkpoint 兼容路径；
- 将 test 接回训练循环；
- 为通过测试而放宽关键配置校验。

## 9. 测试矩阵

### 纯函数测试

- 20/100/500 epochs 下验证 epoch 列表。
- save interval 和末 epoch 强制保存。
- best 首次有效值、严格提升、相等和下降时的行为。
- epoch override 后 total iterations、warmup iterations、poly LR 首/中/末值。

### checkpoint 测试

- CPU 保存/读取，schema 和必需键完整。
- 连续 3 epochs 与 1 epoch + resume + 2 epochs 的 epoch、global step、LR、best 和 scaler 状态一致。
- RNG 恢复后下一批随机数一致。
- split hash、total epochs、optimizer 不兼容时拒绝。
- 截断/损坏 checkpoint 快速失败，旧输出不被覆盖。

### 集成测试

- development 使用 val-dev，测试夹具证明 official test 文件未被打开。
- official 模式不存在 val 时也不读取 test。
- 非有限 validation metric 不保存 best。
- 单 rank 和现有分布式路径至少完成静态/最小测试；本任务目标仍为单卡 4090。

## 10. 验收与提交

1. Terra 实现并运行受影响测试、`python -m pytest -q tests`、compile 和 diff check。
2. Sol 逐行审查 `utils/train.py`、`engine.py` 和 checkpoint schema。
3. 在无 GPU 条件下完成纯函数/CPU checkpoint 测试；GPU 恢复演练留给 04。
4. 单独提交，推荐标题：`完善 MUSeg 长程训练验证与恢复语义`。
5. 输出接口说明、测试结果和迁移风险；不得生成正式训练结果。

## 11. 完成标准

- 训练期 official test 不可被读取。
- 验证/保存计划可由纯函数精确枚举。
- best、latest、periodic checkpoint 路径和内容稳定。
- epoch 边界 resume 恢复所有关键状态且拒绝不兼容协议。
- Sol 完成门禁 B 的核心代码复核。
