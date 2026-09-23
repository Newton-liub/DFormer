# MMFR E1 Batch 1A Implementation Diff Report

> **范围：** C0/F-lite 实现、共同 optimizer coverage 修复与 Gate-B 工具。  
> **日期：** 2026-09-21  
> **结论：** 实现已完成，Gate-B 已通过；正式训练未授权。

## 1. 变更规模

以下 `+/-` 以当前工作区相对 Git 基线的实际行数统计；新文件全部计为新增：

- `models/builder.py`：`+61 / -0`；
- `utils/init_func.py`：`+122 / -0`；
- `utils/train.py`：`+118 / -16`；
- `utils/training_checkpoint.py`：`+67 / -0`；
- `models/feature_adapter.py`：`+73 / -0`；
- `tools/mmfr/e1_batch1a_gateb.py`：`+964 / -0`；
- `local_configs/MUSeg/DFormerv2_S_MMFR_E1_Batch1A_Common.py`：`+108 / -0`；
- `local_configs/MUSeg/DFormerv2_S_MMFR_E1_Batch1A_C0.py`：`+5 / -0`；
- `local_configs/MUSeg/DFormerv2_S_MMFR_E1_Batch1A_FLite.py`：`+5 / -0`。

实现代码合计：`+1523 / -16`。

本表只统计 E1 Batch 1A 实现范围，不把本轮协议、设计与报告 Markdown 计入代码规模，也不把工作区中其它既存修改归入 E1。

## 2. 新增类与函数

### 2.1 F-lite

`models/feature_adapter.py` 新增：

- `FeatureLiteResidualAdapter`：单级 `1×1 down → GELU → zero-init 1×1 up` residual adapter；
- `FeatureLiteAdapter`：只把独立 adapter 应用于 backbone stage 1/2/3，保持 stage 0 不变。

`models/builder.py` 新增：

- `_e1_feature_adapter_config`：fail-closed 核验 F-lite 冻结字段与禁止输入；
- `EncoderDecoder` 中可选 `feature_adapter` 构造；
- adapter trainable parameter count 必须等于配置中的 `173152`；
- backbone tuple 返回后、decoder 前应用 adapter。

### 2.2 Optimizer 与 checkpoint

`utils/init_func.py` 新增：

- `build_e1_optimizer_param_groups`：构造 `base_decay / base_no_decay / new_decay / new_no_decay` 四组；
- 显式纳入 29 个 `*.Geo.weight`；
- 拒绝未知未分类参数、重复分类或非 29 个 Geo；
- 对所有 trainable parameters 强制 membership=1。

`utils/training_checkpoint.py` 新增：

- `load_weights_only_model_state`：核验 checkpoint SHA-256、schema、model key count，仅加载 model state；
- 只允许指定新模块 prefix 的 missing keys，拒绝 unexpected keys；
- `optimizer_step_was_applied`：根据 GradScaler scale 变化判断 optimizer step 是否实际执行。

### 2.3 Config 与训练入口

`local_configs/MUSeg/DFormerv2_S_MMFR_E1_Batch1A_Common.py` 新增：

- `configure_e1_batch1a_candidate`；
- C0/F-lite 共同 source、weights-only restart、四 optimizer groups；
- 20 epoch、128 attempts/epoch、2560 successful updates；
- base/new LR、warmup、poly、GradScaler、curriculum continuation、fixed final checkpoint；
- inherited A2 `mmfr_a2.frozen` 明确标注为 `source-a2-v3-history-only`，E1 活跃 schedule authority 为 `e1_batch1`。

C0 与 F-lite 分别由两个薄 config 选择候选，避免候选字段静默混用。

`utils/train.py` 增加 E1 fail-closed 路径：

- 只接受 `C0` 或 `F-lite`；
- 拒绝恢复 optimizer/scheduler/GradScaler/RNG；
- 强制 20×128 与 2560 successful updates；
- 强制 AMP on、SyncBN on、DDP off、`--gpus 1`；
- 使用 weights-only source load；
- 使用四组 optimizer；
- F new groups 保持 3× base LR；
- A2 curriculum 从 epoch 420 对应进度继续，而非重置为 0；
- 任一 GradScaler skip 立即 fail；
- 末尾只在 successful updates=2560 且 skips=0 时写 fixed final checkpoint。

### 2.4 Gate-B 工具

`tools/mmfr/e1_batch1a_gateb.py` 新增独立资格入口，覆盖：

- source/config/split/checkpoint identity；
- C0/source shared state、logits、prediction、loss identity；
- F shared-state identity与 stage residual、decoder input、logits、prediction、loss no-op；
- 四 optimizer group 名称、顺序、LR、WD、参数集合与 name digest；
- 29 个 Geo、14 个 SyncBN、6 个 reliability 参数、12 个 F adapter 参数的精确归组；
- C0 new groups 为空；
- C0 一步与 F 两步 AMP startup；
- shared/reliability/Geo/F up/down gradient 与 parameter update；
- 参数量、peak allocated/reserved memory、forward-loss latency 与 inference latency；
- canonical JSON 与 `failed_checks`。

## 3. Optimizer 修复

历史 A2+SyncBN 的实际 coverage bug 是 29 个 `backbone.layers.*.blocks.*.Geo.weight` 未进入 optimizer，不是 43 个：

- 29 个 Geo 现在进入 `base_decay`；
- 8 个 patch-embed SyncBN 与 6 个 downsample SyncBN 参数继续在 `base_no_decay`；
- reliability head 保持 weight decay / bias no-decay；
- F adapter Conv weights 进入 `new_decay`；
- F adapter biases 进入 `new_no_decay`；
- C0 的 new groups 显式保留为空；
- 不迁移 epoch-420 optimizer state。

Gate-B 实测确认所有 trainable parameter membership 都为 1，且 C0/F 的 29 个 Geo 都出现合法梯度/更新行为。

## 4. F adapter 实现

结构保持冻结合同：

$$
F_l'
=
F_l+
W_{up}^{(l)}
\operatorname{GELU}
\left(
W_{down}^{(l)}F_l+b_{down}^{(l)}
\right)
+b_{up}^{(l)}.
$$

- stages：1/2/3；
- stage 独立；
- ratio：1/4；
- normalization：none；
- down weight：trunc-normal std 0.02；
- down bias：0；
- up weight/bias：0；
- 新增 trainable parameters：`173152`；
- 不读取 reliability、condition、severity 或 oracle。

## 5. 明确未修改项

以下路径和语义未因 Batch 1A 实现而改变：

- evaluator：未改；
- Quick-Val/Main-Val：未新增逻辑、未运行；
- corruption generator 与六类 A2 v3 corruption：未改；
- `mmfr_training_v3.py` 数据生成语义：未改；
- RGB/Depth preprocessing、crop、pad、normalization：未改；
- DFormerv2 encoder 与 `GeoPriorGen` 实现：未改；
- HAM decoder 结构与参数：未改；
- reliability auxiliary 的输入、target、loss 权重与隔离边界：未改；
- R-EM：未实现；
- R-OE substitute：未实现；
- T/Cpair、G、Oracle-B：未实现；
- official test：未读，继续 `sealed_unread`。

F-lite 只在 backbone tuple 与既有 decoder 之间增加可选 adapter；C0 不创建该模块。

## 6. 验证

实际完成：

- E1 相关 Python 文件定点编译；
- 强化版 Gate-B：`C0 PASS`、`F-lite PASS`；
- canonical JSON SHA-256：`5d5f0526e95ed81b6da8fbcd3cc395911f2266132a9148826b5ad261042e4e89`。

按验证预算未运行完整测试套件、全仓扫描、20 epoch 训练、2560-update run、Quick-Val、Main-Val、云任务或 official test。

## 7. 恢复点

实现已停在 `Batch 1A C0/F Gate-B PASS`。正式训练必须等待上级审计与用户单独授权；不得从本报告推导为已批准训练。