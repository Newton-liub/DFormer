# MUSeg Stage-05 seed 1 本地收口进度交接

- 记录时间：2026-08-26 09:33 UTC
- 任务范围：Stage-05 seed 1 训练完成后的证据取回、独立裁决、本地后评估、文档收口与发布
- 当前状态：云端证据已经安全取回且实例已停止；后评估工具正在修正本地模型构造路径，最终文档、Canvas、全量验证和 Git 发布尚未完成
- 性质：工作中交接快照，不替代待生成的正式完成与风险处置报告
- 排除范围：`liu-test-exp/**` 是用户思路设计区，本任务不分析、不修改

## 1. 已完成并验证

### 1.1 训练结果与独立裁决

- seed `772961337` 已完成 500/500 个 epoch，进程退出码为 0。
- val-dev 共完成 50/50 个验证点；最佳结果为 mIoU `52.84`，对应 epoch 460。
- 最终记录的有效优化器更新数为 `63,973`。
- 独立 v2 裁决 `acceptance-v2.json` 为 `pass=true`，重新核验了原报告列出的 18 项证据，其中包括 12 个 checkpoint。
- 原始 `acceptance.json` 保持失败且未改写；唯一失败项为旧版 `milestones_complete` 遥测规则。原始 `failed.json` 也保持不变。
- official test 仍为封存未读取：原始裁决记录 `official_test_read=false`，训练结果记录 `official_test_included=false`，运行配置记录 `sealed_unread=true`。

### 1.2 证据取回与本地哈希复核

完整不可变归档已保存到仓库外：

- 路径：`D:\0Project\DFormer-stage05-archive\museg-stage05-seed772961337\museg-stage05-seed772961337-original.tar`
- 大小：`3,865,057,280` bytes
- SHA-256：`4f6b079b707266ee358d2522fc6e4e034a5380d09ba8c65696df7aaa3e383c66`
- 本地重新计算结果与云端清单一致。

本地后评估所需 checkpoint 已保存到仓库外并复核：

- `best-val-miou.pth`：`b62ca049e6a647aca109c70e80823cec8e36ae1cc1df27e3bcf2b1d215b160bf`
- `epoch-500.pth`：`0b88ab022db5188fd3439ea4e3af2098fe81e7c85757d1d91db33e831df2ff79`

val-dev 本地评估包、日志、报告、运行 JSON 和 `acceptance-v2.json` 已保存到 `D:\0Project\DFormer-stage05-evidence`。val-dev split SHA-256 为 `1d0719d8f64f016d48995c25ab66d4004d76b7155d9efeef7cbb7454c0dd0e83`，包中不含 official test。

### 1.3 云实例停止

- CompShare 实例：`cpod-1tyvjsiu6ahe`。
- 停止前状态：`Running`，观察到实例价格为 `0.13` 单位/小时。
- 在归档和两个 checkpoint 的本地 SHA-256 全部匹配后执行停止。
- 停止命令成功，随后查询状态为 `Stopped`。
- 本次出现了训练完成后未按预期自动关机的风险。未来运行必须把“同步证据并验证哈希后关机”用于成功和失败两类终态，并在控制面预设最晚停止兜底；验收结果不得决定是否继续计费。

## 2. 已完成但尚需最终验证

### 2.1 裁决与后评估工具

已新增：

- `tools/adjudicate_museg_seed_acceptance.py`
- `tests/test_museg_seed_acceptance_adjudication.py`
- `tools/evaluate_museg_checkpoint.py`
- `tests/test_museg_checkpoint_posteval.py`

现有定向测试已经通过：裁决测试 3 项；裁决与后评估合计测试 7 项。最终提交前仍需重跑定向测试、全量测试、编译和格式核验。

### 2.2 训练遥测与启动器

`utils/train.py` 和 `tools/run_museg_seed.py` 已加入未来 `run_kind=standard`、尝试/完成/跳过更新计数和稳定 epoch 末遥测。`tests/test_museg_stage03.py` 已覆盖 `standard` 转发，当前该测试文件结果为 18 项通过。最终提交前需要检查 diff，并补充或确认优化器计数语义的直接测试。

## 3. 当前阻塞：本地后评估模型构造

首次后评估尚未产生有效指标，失败均发生在数据推理前：

1. 直接执行脚本时无法从 `tools` 目录解析仓库根目录下的 `models`，已改用 `python -m tools.evaluate_museg_checkpoint`。
2. 本地 `df2` 环境缺少云端模型依赖，现已安装 `timm 1.0.28`、`mmengine 0.10.7` 和兼容旧代码的 `mmcv 1.7.2`；这些环境改动不进入 Git。
3. 当前 `load_model()` 使用 `EncoderDecoder` 的默认训练 criterion 构造模型，触发预训练权重初始化，因而在完整 checkpoint 严格加载之前错误要求本地存在 `DFormerv2_Small_pretrained.pth`。

正确修复是以 `criterion=None` 构造评估模型，跳过不必要的预训练初始化，然后使用 `strict=True` 加载完整 checkpoint。该问题属于后评估工具实现问题，不表示 checkpoint 损坏或训练失败。修复后必须增加回归测试。

## 4. 后续任务与完成条件

按以下顺序继续，不跳过证据门禁：

1. 修正后评估模型构造并增加测试，确认评估不依赖单独的预训练文件且 checkpoint 仍严格加载。
2. 运行最佳 checkpoint 的 `original-full`；若本地 8 GB 显存 OOM，将结构化记录为 `environment_limit`。
3. 运行最佳 checkpoint 的 `resize-480x640` 与 `sliding-480x640`。
4. 运行 epoch-500 checkpoint 的 `resize-480x640` 与 `sliding-480x640`。
5. 核对五份 JSON 的 checkpoint SHA、split SHA、BGR、val-dev、`official_test_included=false`、样本数和指标。
6. 更新 `doc/main/MUSeg-current-status.md` 与 `doc/main/MUSeg-open-decisions.md`，将运行中描述收口为已完成、保留原始 v1 失败事实并登记 v2 裁决。
7. 为运行中交接、预启动报告和 Markdown 审计添加后继文档指针，不重写历史时点事实。
8. 新增正式的 Stage-05 seed 1 完成与风险处置报告，并登记训练结果、后评估边界和自动关机风险。
9. 将未来云生命周期策略写入当前模板或指南：所有终态先同步证据、复核哈希、再关机，并在启动前设置控制面最晚停止兜底。
10. 等事实报告定稿后生成 Canvas `0.0.10`，发布到 Cursor 受管目录，并把报告索引的下一版本更新为 `0.0.11`。
11. 删除临时帮助文件 `.tmp-export-stage05.py`；确认 `liu-test-exp/**` 无 diff，且没有数据、checkpoint、归档或凭据进入仓库。
12. 执行定向与全量测试、冻结 manifest 测试、Python 编译、JSON 校验、Bash/PowerShell 语法检查、Canvas 发布与类型检查、`git diff --check`。
13. 检查 `origin/main` 未前移，审阅全部既有文档一致性改动，然后创建一个 seed 1 收口提交并推送 GitHub。

## 5. 不得改变的边界

- 不修改作为原始证据的 `acceptance.json`、`failed.json` 和 `liu-test-exp/**`。
- 不把仓库外归档、数据、checkpoint 或凭据提交到 Git。
- 不把单 seed 的 52.84 mIoU 表述为三 seed 正式 baseline。
- 不因后评估几何结果选择或读取 official test。
- 不编辑用户指定的原计划文件。
