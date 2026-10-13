# DFormer 论文库快速操作指南

索引唯一真源在 `D:\0Project\origin\_index\PAPER_LIBRARY_INDEX.json`，正式论文资产在 `D:\0Project\origin\论文\`。日常流程与权限按以下约定执行；详细字段归属见外部索引目录的 `README.md`。

## 常规新论文入库

用户用外部转换器完成的 Markdown/图表/公式论文 bundle，放在 `D:\0Project\DFormer\临时\临时论文\`。交**执行模型**，一句话即可：

> `/paper-intake` 请对 `临时\临时论文` 中已经处理好的论文批量查重并入库。无争议新条目本次授权按现有 CLI 流程完成 plan 和 apply，完整重复跳过；只在确认成功入库或完整重复后，将对应来源送 Windows 回收站。异常和未入库的留下并报告。入库只读既有正式论文，不再次下载或转换。

普通入库不必拆成两次 Cursor 消息审批；程序仍要求真实计划 SHA-256 与索引并发校验。**只有要求“先预检”时才在计划后停止。**来源清理必须明确授权，且仅限确认成功入库或完整重复的对应 bundle。

## 读论文和补书目

- **书目字段与论文等级：** 给执行模型精确 LIB 编号、目标字段（`year` / `venue` / `rating` / `doi`）与来源依据，走 `/paper-maintenance`。其中 `doi` 只用于点名 LIB 的定点纠错（例如把数据集 DOI 与论文 DOI 分开），写入的是人工确认值，`auto.doi` 扫描值保留不动：

  ```text
  maintenance-plan  --changes <维护请求.json> --plan <审核计划.json> [--recovery-root <库外恢复根>]
  maintenance-apply --plan <审核计划.json> --approved-plan-sha256 <真实计划 SHA-256> [--export-dir <导出目录>]
  views-refresh     [--export-dir <导出目录>]
  ```

  请求是一份短 JSON：根为 `items`，`metadata` 项给准确 `lib_id` 与 `changes`（值为 override 载荷，或 `null` 撤销该字段的人工覆盖）；计划会自动带出标题、原有效值与目标值供核对。也可在索引 UI 的「人工确认书目」里直接编辑这三个字段，走同样的差异确认与冲突检查。`rating` 是独立字段，未记录显示「未记录」；预印本年份必须写明 `year_kind` 口径，不得当成正式发表年。
- **精确退库与恢复：** 必须点名 LIB 与完整题名。`retire` 首次使用 `论文_retired\<LIB>\<原目录名>\`（可用 `--recovery-root` 换到同卷的库外恢复根）作同卷整目录移动，记录保留在索引里、人工编号仍占用；`restore` 只按已有恢复映射移回，需**另行明确授权**。中断可续办，纯目录续办不保存、不增加 revision。临时来源清理与退库是两种不同授权，不得混用。
- **科研提取：** 给主力模型，在正式库读取处理后的 Markdown；关键数据看 xlsx、关键图看 jpg，默认输出独立科研卡片。只有明确要求把证据写回索引时才用 `/paper-supplement`。
- **AI 阅读导出：** `views-refresh` 重建管理视图与三份固定导出；导出只含活动记录，已退库记录不出现。

## 并行注意

**索引只有一个写入者**，入库、改书目/等级、退库与恢复串行；主力模型可以同时只读论文或项目报告。异常版本、正文相同但附件不同的来源先保留，再由主力模型裁决。不要让高速模型重复转换、重复做无关全库哈希审计或逐篇生成长报告。
