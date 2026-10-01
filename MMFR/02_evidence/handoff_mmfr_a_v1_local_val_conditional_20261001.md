# MMFR A-v1 本地 val 条件性交接说明

- 日期：2026-10-01
- 用途：说明如何接收 A-v1 转移包，并在上级另行明确同意后准备本地验证环境。
- 性质：**普通操作交接，仅代表资料准备；不是运行授权、研究结论或 Main-Val 入口。**
- 当前实时状态见 `doc/main/MUSeg-current-status.md`；A-v1 固定合同见 `review/mmfr_a_v1_action_utility_protocol.md`（Git内原路径为 `MMFR/01_research/mmfr_a_v1_action_utility_protocol.md`）。正式 Quick-Val 审核材料为同目录的 [`report_mmfr_a_v1_formal_quickval_upper_review_20261001.md`](report_mmfr_a_v1_formal_quickval_upper_review_20261001.md)。本文件和报告在包内同置 `review/`，其相对链接仍可使用。

## 1. 现在的授权边界

A-v1 已完成 Proposal 1920 次、Gate 640 次成功更新，skip=0，并完成唯一一轮四条件 Quick-Val，筛选记录为 `stop`。该轮授权已经执行完毕；本地 val、Main-Val、official test、重训、额外 seed 或其他评价都没有因此获得新授权。现有结果是单 seed 筛选证据，不等于统计显著性或一般模型结论。official test 仍为 `sealed_unread`。

**大白话：** 现在只能准备和核对转移资料。任何新的 val 都必须先取得新的、范围明确的上级同意；收到 checkpoint 或配置里有 `authorized: true` 都不会改变这一点。

配置中的 `formal_training_authorized`、`quickval_contract.authorized` 和 `cloud_preflight.execution_authorized_this_round` 是已经执行那一轮时冻结在配置里的历史合同字段，不是当前权限开关。后续操作以实时状态文档和针对新操作的明确授权为准。

## 2. 预期转移包内容与身份

转移包固定交付位置为 `/root/rivermind-data/cloud/MMFR_AV1_local_val_conditional_20261001.zip`。以下为打包采用的布局；云端生成后的整包大小、SHA-256与内容核验结果见Git内交付收据 `MMFR/02_evidence/delivery_mmfr_a_v1_local_val_20261001.md`。本地收到后仍须独立核对目录和哈希。

| 位置 | 预期内容 | 核对身份 / 边界 |
|---|---|---|
| `source/` | Git `HEAD 4f84b469c4b04de657eaf2c455bd44e991b7f869` 的限定源码快照 | 不含数据或 test；这是源码快照身份，不代表整个仓库或可独立运行的 Git 工作树。 |
| `checkpoints/A-v1/update-2560.pth` | A-v1 唯一 fixed-final checkpoint | SHA-256：`87ec54d10192d3aedc1d6edb864b7b60b94ce66220a748a12f1ca50ca605d336`。 |
| `checkpoints/C0/update-2560.pth` | C0 源 checkpoint | SHA-256：`ca618b23d18eabb201a0d11d18da383ac99576d0feae5864e3233bda527d9a1a`。 |
| `review/` | 正式 Quick-Val 审核报告、A-v1 protocol 和本交接说明 | 报告正文用于追溯本轮结果；本交接说明不替代实时状态或 protocol。 |

Dataset 不在包内。两份 checkpoint 的原始完整字节须保留，不得覆盖、重存或以转换后的文件代替。包内排除 official-test 样本、清单和缓存；本文件也不提供自动 GPU 启动脚本。整包 SHA-256 由同名 `.zip.sha256` 配套校验文件和 Git 内交付收据提供，不能只凭 ZIP 文件名确认归档身份。`source/` 包含冻结的 train-dev/val-dev 清单而非数据图片；`evidence/` 包含按原字节保存的训练/Quick-Val及诊断小证据，内含原云端路径，不代表本地新运行。

## 3. 收包、解压与 checkpoint 哈希

1. 保留收到的 ZIP 原件，并先列出归档条目。解压目标使用一个新建、权限收紧、位于仓库之外的空目录；不要解到当前 Git 工作树、数据目录或已有 checkpoint 目录，也不要使用覆盖已有文件的选项。
2. 解压前拒绝绝对路径、`..` 路径段、反斜线歧义路径、重复条目、符号链接和其他特殊文件。只解压普通文件与目录，不运行归档中的程序。检查归档内容符合上表后，再继续核验。
3. 对两个 checkpoint 计算 SHA-256，并与上表逐字比较。例如在解压目录执行：

   ```bash
   sha256sum checkpoints/A-v1/update-2560.pth checkpoints/C0/update-2560.pth
   ```

   任一值不匹配时，隔离该副本并停止；不要尝试加载、修复或重新保存它。哈希匹配证明文件字节相同，不证明本地数据、源码工作树或评价口径也正确。
4. 将 `source/` 与完整本地仓库中指定提交的对应文件逐项比对。现有 A-v1 Quick-Val 代码会读取 Git 提交身份，并要求其列出的运行文件在工作区没有未提交差异（该检查针对指定文件，不等同于要求整个工作树干净）；单独解压的源码快照没有被证明是完整仓库或可直接运行的 Git 工作树。未经授权不要拉取、切换分支、创建 worktree 或修改当前工作树。
5. 冻结的 val-dev split 为 318 条，SHA-256：`1d0719d8f64f016d48995c25ab66d4004d76b7155d9efeef7cbb7454c0dd0e83`。在获准准备评价时，另核对本地 `data/splits/MUSeg/dev-v1/val-dev.txt` 的字节哈希与条目数；不要把其他 split 或官方 test 列表替代它。

## 4. 本地环境与路径

- 当前数据位置记录为 `/root/rivermind-data/dataset/MUSeg_DFormer`，目录结构应包含 `RGB/`、`Depth/` 和 `Label/`。配置以环境变量 `DFORMER_DATA_ROOT` 指定 dataset 的父目录；按当前路径应指向 `/root/rivermind-data/dataset`。仓库迁移后不要依赖配置推导出的默认绝对路径，应显式核对实际目录。
- A-v1 配置支持 `MMFR_AV1_SOURCE_CHECKPOINT` 覆盖 C0 checkpoint 路径，默认路径位于项目树下的历史 `cloud/` 目录。转移包位置与默认值不同。该变量供配置中的 source checkpoint 身份检查使用；现有 Quick-Val 命令行另有必填 `--source-c0-checkpoint` 参数，不能只设置环境变量就认为评价输入已指向包内 C0。
- Quick-Val 入口还要求显式给出 A-v1 checkpoint、其预期 SHA、C0 checkpoint、C0 预期 SHA、dataset root、split 路径、split 预期 SHA 和一个尚不存在的输出目录。输出目录使用独立新路径，不要覆盖历史评价结果。
- `DFORMER_OUTPUT_ROOT` 是继承配置的训练输出根路径，不是 Quick-Val 的输出目录设置。不要据此推断评价输出路径。
- 转移参考环境为 PyTorch `2.1.2+cu118` 与 RTX 4090。它只是原环境兼容参考，不保证其他 GPU、驱动、PyTorch/CUDA 组合逐位一致；本次没有运行导入或环境检查。基线 `source/README.md` 的安装章节包含历史依赖说明，`requirements-monitoring.txt` 只覆盖监控依赖，均不是原运行环境的完整锁定清单；本交接不执行安装或升级。

### 路径迁移与身份比较限制

A-v1 训练入口会把 Git commit、解析后的 source checkpoint 路径、split 路径和哈希写入运行身份，并要求 checkpoint 与冻结 SHA 相符；普通完整恢复还要求保存的 protocol identity 与当前 identity 完全相同。精确的 Proposal1280 特例也绑定批准父 checkpoint、父 commit 和合同 SHA。**因此，文件内容相同不代表路径或 Git 身份迁移后仍可用于恢复；本交接包不授权训练或恢复。**

Quick-Val 对传入的 checkpoint 和 split 做 SHA 校验，并把解析后的路径和 Git commit 写入结果身份。相同字节换到新路径时，内容哈希可以仍匹配，但记录的路径会改变；换 Git 提交或直接在非 Git 解压目录运行也不能视为原有评价身份。应分别核对源码提交、checkpoint 字节、split 字节和本地数据可用性，不要用其中一项替代其他项。Dataset 不随包交付，其完整数据身份或全量内容哈希本次未核验。

## 5. 已实现的评价入口与 Main-Val 缺口

当前 `tools/mmfr/av1_quickval.py` 只实现 A-v1 四条件、单视图评价：`clean`、`entire_missing@1.0`、`spatial_dropout@0.75`、`misalignment@0.75`；每条件比较 `off/full/learned`。视图为 original-full、scale 1.0、无翻转，单视图，使用冻结的 318 条 val-dev。它不是十条件、多尺度/翻转 Main-Val 实现。

十条件、多尺度/翻转的 Main-Val 适配**尚未实现**。因此本交接不提供也不暗示任何可运行 Main-Val 命令；不得把 Quick-Val 入口改参数、拼接旧 evaluator 或把其四条件输出称为 Main-Val。已经完成的单次 Quick-Val 也不得因收到转移包而重跑。

## 6. 另行批准后的最小后续步骤

只有上级先书面批准新的评价范围后，才进入相应准备。若批准目标是 Main-Val，至少需明确：

1. 评价目标与授权边界：确认仅使用哪一开发集 split、明确排除 official test，指定 checkpoint 和评价轮数，并说明是否允许任何代码改动、GPU 计算和结果写入。
2. 冻结 Main-Val 口径：十个条件的精确定义、严重度、尺度/翻转视图集合、视图汇总方法、随机状态配对方式、原始标签网格和指标口径、输出与判定要求。
3. 在单独批准的代码修改任务中实现最小适配，并单独审核条件构造、原图/损坏输入对齐、尺度与翻转回映、off/full/learned 配对、混淆矩阵累计、身份记录和拒绝覆盖行为。不得改变既有四条件 Quick-Val 合同或将新入口接到旧 E1 路径。
4. 完成经批准的资格核验后，核对源码 Git 身份、两个 checkpoint SHA、val-dev split SHA/计数、数据目录可读性、环境版本与空输出目录；把核验结果提交上级复核。
5. 只有在评价运行本身也获得明确授权后，才按批准预算运行并留存原始结果。资格核验通过本身不等于运行授权。

如果上级只批准四条件单视图验证，也必须明确这是新的运行授权；应先评估是否有必要重复已完成的唯一 Quick-Val，不得因已有入口而自动运行。

## 7. 本次交接完成与待核验项

本次只准备文档和转移资料，不新增模型运行。云端包生成与原文件/归档字节核验以交付收据为准；本地收包、目标源码工作树、数据可读性、软件环境与跨硬件行为尚未验证。收包后按上述步骤做资料核对，并先取得评价/适配所需授权。打包不解码或改写 checkpoint，不运行模型、项目测试、GPU、训练或评价；普通文件与哈希核验不能被解释为 Main-Val 已完成。
