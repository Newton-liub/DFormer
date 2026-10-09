---
name: paper-intake
description: Intake processed paper bundles into the external library under an explicit approval. Use when the user asks to 入库/接入/新增论文 bundle, or to pre-check a 处理后论文 source before anything is copied. Produces a review plan only by default; copying and index writes need a separately approved plan.
disable-model-invocation: true
---

# 论文入库（paper-intake）

把用户指定的「处理后 bundle」接入外部论文库 `D:\0Project\origin\论文\`。默认只产出审核计划并停止；复制与索引写入必须另有一次明确批准。

## 先读

1. 本 Skill 所在仓库的 `doc/state/current.md`（当前授权边界与恢复点）。
2. 共享 Rule `.cursor/rules/paper-library.mdc`（原资产只读、唯一 store 写入、人工字段保护、审批与冲突停止）。
3. 需要目录与字段约定时读 `D:\0Project\origin\_index\README.md`。

## 边界（默认行为）

- 只接受用户明确给出的来源目录；不做全盘搜索，不自动解压压缩包。
- 只复制全新论文；完全重复、正文同附件异、疑似版本、异常 bundle 一律留原处并报告。
- 来源不移动、不删除；不清理来源目录；不手工分配 LIB 编号。
- 索引只经 `_index\tools\paper_workflow.py` → `paper_library_store.py` 保存一次；不直接改 JSON。

## 命令

工具：`D:\2Env\anaconda\python.exe D:\0Project\origin\_index\tools\paper_workflow.py`

```text
# 1) 预检：只读索引与来源，写一份审核计划，不改索引、不复制
intake-plan --source <bundle 或一层父目录> [--source <...>] --plan <审核计划 JSON>

# 2) 执行：仅在用户/上级明确批准后
intake-apply --plan <审核计划 JSON> --approved-plan-sha256 <该文件的 SHA-256>
```

隔离或非默认索引时追加 `--index <JSON>`；需要时追加 `--root <资产根>`（必须与索引的 `library_root` 一致，否则停止）。

计算批准指纹：

```powershell
(Get-FileHash -Algorithm SHA256 -LiteralPath "<审核计划 JSON>").Hash.ToLower()
```

## 流程

1. **预检**：运行 `intake-plan`。它枚举来源、检查正文、递归清点全部普通文件的相对路径/大小/SHA-256，并完成索引内 canonical/alternate 与同批互比。
2. **分类**：每项标为 `new`、`full_duplicate`、`body_same_attachments_differ`、`suspected_version`、`anomaly` 之一并给出理由。只有 `new` 且 `approved=true` 会被复制。
3. **送审并停止**：把计划路径、来源、分类计数、每项理由与目标目录名提交给用户/上级。不要自行把疑似版本变成新 LIB、alternate 或 duplicate-review。
4. **获批后执行**：确认批准，运行 `intake-apply`。审核子集时先编辑计划文件（把不批准项的 `approved` 改为 `false` 或删除该项）再计算新的 SHA-256。
5. **报告**：来源→目标→实际 LIB 对照、提交指纹、dry-run 结论；若出现残留目录或失败，报告已复制目标与实际索引提交状态。

再次审批、再次保存都需要在新消息里重新显式调用本 Skill；Skill 不会跨消息自动继续。

## 计划的判定要点（供审核阅读）

- 全资产比较只看「正文角色 + 其余按原相对路径」；文件数相同但某个附件 hash 不同即属「正文同、附件异」。
- 候选的实际目录缺失、不可读或正文 hash 与索引不符时，不做等同判断，归入疑似版本。
- 目标目录撞名不自动加后缀选版本；目标已被占用即停止。

## 停止条件

审批缺失或指纹不符、来源在审批后变化、索引并发修改、候选不明确、目标冲突、复制/资产校验失败、dry-run 超出批准范围、store 拒绝或验收失败。停止时保留来源，报告残留文件与已发生的复制/提交；不自动回滚或重试提交。若索引 JSON 已保存而派生 Markdown 重建失败，按「JSON 已提交、视图失败」报告：不重跑 `intake-apply`、不二次保存，管理视图另行单独重建。

## 不做

不调用模型 API、不生成科研结论、不清理来源、不合并版本、不执行 Git 提交或推送、不新建测试框架或常驻服务。
