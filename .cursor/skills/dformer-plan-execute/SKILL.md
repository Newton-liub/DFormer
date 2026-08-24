---
name: dformer-plan-execute
description: 使用此 Skill 时，先用复杂模型规划，再自动用简单模型执行并验证 DFormer 任务。用户只需说“使用 dformer-plan-execute：任务”。
---

# DFormer 自动规划执行

## 明确调用方式

在 Cursor 对话框中输入 `/dformer-plan-execute`，选择该 Skill 后紧跟任务描述；或发送：

`使用 dformer-plan-execute：<任务>`

本 Skill 只有在当前会话的 Task 工具明确暴露以下三个自定义代理时，才允许自动编排：

- `dformer-planner`
- `dformer-executor`
- `dformer-verifier`

## 运行前硬性检查

1. 在开始调查或修改前，检查当前 Task 工具的可用自定义代理名称；不得仅依据 `.cursor/agents/*.md` 文件存在就认定代理可调用。
2. 三个名称全部可用后，严格按 `dformer-planner` → `dformer-executor` → `dformer-verifier` 串行调用；每次 Task 调用必须使用对应的精确自定义代理名称。
3. 禁止把 `generalPurpose`、`explore`、`shell` 或其他内置/通用代理包装成这三个 DFormer 代理；禁止在提示词中自称“规划代理/执行代理/验证代理”来替代命名代理。
4. 任一名称不可用时立即停止，输出 `DFORMER_AGENT_UNAVAILABLE`、缺失名称和当前可用的代理类型。不得回退到父代理或通用子代理，不得声称已经使用 Terra。
5. 每阶段只把 Task 启动元数据中的自定义代理名称作为“代理已正确加载”的证据。代理自行报告的名称或模型不是可靠证据。

预检查通过后自动完成以下流程，不要求用户分别调用代理：

1. 调用 `dformer-planner`，使用 `.cursor/agents/dformer-planner.md` 的配置只读调查并生成计划。
2. 将计划直接交给 `dformer-executor` 完成代码、配置、测试和必要实验。
3. 调用 `dformer-verifier` 独立验证实现和证据。
4. 汇总 Task 启动名称、修改文件、验证结果、未完成事项和风险。

只有以下情况需要暂停并询问用户：任务目标无法判断、涉及破坏性资源操作、或计划需要超出用户授权范围的变更。普通代码任务不要在规划完成后再次询问确认。

## 代理限制

- 直接子代理最多 4 个；默认依次使用规划、执行、验证三个代理。
- 子代理不得创建子代理，不得递归委派。
- 多个代理修改同一文件时串行执行；并行编辑必须使用隔离 worktree。
- 不自动提交 Git。

## 任务规则

- 规划代理只读，输出目标、文件、步骤、验证命令和风险。
- 执行代理先核对计划，再分阶段实施并验证。
- 验证代理只读，独立检查实现、测试和实验依据。
- 不修改数据集、checkpoint 或原始实验结果来配合结论。
- 实验结论必须有配置、日志、命令或结果证据。

## 现有 Skill

涉及 CompShare GPU 云资源时继续遵守 `compshare-cli` 的 JSON、dry-run、授权和敏感信息规则。涉及阶段报告、Markdown 或 Canvas 时继续遵守 `research-progress-report` 的边界、证据和版本规则；更具体的现有 Skill 优先。

## 云端工作目录与环境

涉及云端执行、训练、验证、数据处理或实验命令时，统一使用以下配置：

- 云端项目目录：`~/rivermind-data/DFormer#`；其中末尾的 `#` 是目录名的一部分。
- 云端 Python 环境：`py310`。
- 禁止使用云端根目录下的 `DFormer`（例如 `~/DFormer`）作为项目目录；不得把它与 `~/rivermind-data/DFormer#` 混用。
- 执行云端命令前，先切换到 `~/rivermind-data/DFormer#`，并使用 `py310` 环境；命令、日志和结果路径应以该项目目录为基准。
- 由于目录名包含 `#`，在 Shell 命令中应将路径整体加引号（例如 `cd '~/rivermind-data/DFormer#'`），避免 `#` 被解释为注释起始符。

## 模型配置与证据

Cursor 实际读取三个 `.cursor/agents/*.md` 文件中的 `model` 字段；`.dformer/agent-workflow.yaml` 仅用于项目内集中对照，不会创建代理，也不会控制 Task 的模型路由。更换模型时同步修改两处：

- 复杂模型：`dformer-planner.md` 的 `model`；
- 简单模型：`dformer-executor.md` 和 `dformer-verifier.md` 的 `model`。

当前配置为规划 `gpt-5.6-sol`，执行和验证 `gpt-5.6-terra`。Cursor 可能因团队管理员限制、套餐限制或旧版请求制套餐未开启 Max Mode 而回退到兼容模型。最终模型应以 Cursor 的 Task 运行详情或用量记录为准；代理文本中的自报模型不得作为证据。

若当前 Task 工具没有暴露命名自定义代理，修改 Skill 提示词或 `.dformer/agent-workflow.yaml` 无法强制切换模型。此时应重新加载 Cursor 窗口并新建会话后重试；仍不可用时，可用 `/dformer-executor` 或 `/dformer-verifier` 直接测试对应代理是否被 Cursor 注册。

## MUSeg 单卡 4090 固定知识

- 本地已核实的数据集路径是项目上一级的 `../dataset/MUSeg_DFormer`；训练划分 `train.txt` 含 1595 项，验证划分 `test.txt` 含 1576 项，样本由 `RGB/*.jpg`、`Depth/*.png` 和 `Label/*.png` 三部分组成。
- 训练前先执行 `python tools/preflight_train.py`。仅做无 GPU 开发机检查时可加 `--allow-no-gpu`，仅在明确不加载预训练权重的专项检查中可加 `--skip-pretrained`；正式训练不得跳过预训练权重检查。
- 4090 显存与吞吐探测使用 `bash tools/probe_museg_4090.sh`，依次测试 batch 4、8、12、16，每档约 60 步；根据稳定吞吐、峰值显存、损失与 AMP scale 稳定性及至少 10% 显存余量人工选档，不自动选择。
- 正式启动使用 `bash tools/train_museg_4090.sh`。脚本固定进入 `~/rivermind-data/DFormer#` 并使用 `py310` 的 Python，可通过 `DFORMER_DATA_ROOT`、`DFORMER_PRETRAINED`、`DFORMER_OUTPUT_ROOT`、`DFORMER_BATCH_SIZE`、`DFORMER_VAL_BATCH_SIZE`、`DFORMER_WORKERS` 和 `DFORMER_EPOCHS` 覆盖运行参数。正式训练默认单尺度验证以匹配显存探测；仅在单独复核多尺度验证显存后设置 `DFORMER_MST=1`。
- SwanLab 默认项目为 `DFormer-liu`、工作区为 `Newton_liub`。凭据只允许通过云端已有登录状态或 `SWANLAB_API_KEY` 环境变量提供，禁止写入仓库、Skill、配置、脚本或日志；online 初始化失败必须终止训练，offline/disabled 才允许不连接远端运行。
