---
name: paper-intake
description: Intake processed paper bundles into the external library. When the user explicitly authorizes intake, complete the supported plan/apply flow in the same session; preflight-only requests stop after planning. Source cleanup requires separate explicit authorization.
disable-model-invocation: true
---

# paper-intake：批量入库

工具：`D:\2Env\anaconda\python.exe D:\0Project\origin\_index\tools\paper_workflow.py`。遵守 `.cursor/rules/paper-library.mdc`；命令或参数变化时先看对应子命令 `--help`。

## 必要上下文

- 本轮尚未读取 `doc/state/current.md` 时先读取一次，确认授权边界与恢复点；同一会话、同一任务范围已读取则复用，不重复通读。
- 索引位置、字段所有权或冲突边界不清楚时，再查 `doc/guides/external-resources.md` 或 `D:\0Project\origin\_index\README.md` 的相关段落。

## 流程

1. **输入为用户指定的处理后 bundle 目录**：论文正文 Markdown 及相关图、表、公式等资产。按题名、DOI、arXiv、别名及必要指纹与正式索引查重；已有论文不重新下载、转换或入库。可同批指定多个来源。
2. 执行 `intake-plan --source <目录> --plan <plan.json>`；`--source` 可重复。依据真实计划分类 `new`、`full_duplicate`、`body_same_attachments_differ`、`suspected_version`、`anomaly`。完整重复记录对应 LIB；争议项不自动合并。
3. 用户明确要求入库时，只对授权范围内无争议的 `new` 项按真实计划执行 `intake-apply --plan <plan.json> --approved-plan-sha256 <sha>`。用户只要求“预检”时，plan 后停止。审批子集只能用工具支持的计划字段调整；计划有任何修改都重新计算 SHA-256。
4. apply 成功后核对来源→目标 bundle、实际 LIB 与索引结果；需要时统一重建视图。若 JSON 已提交而派生视图失败，单独报告该阶段，不重复 apply 或二次保存。
5. **临时来源清理另需明确授权**：仅对确认成功入库且目标存在的来源，或已核实为活动库完整重复的来源，才可送 Windows 回收站；无法安全回收时放受控恢复目录并报告。疑似版本、附件差异、异常、未入库及其他文件留在原处；全部清空后才可移除来源目录本身。
6. **需要补书目或等级时转 `/paper-maintenance`**：新论文取得真实 LIB 后，可在同一已授权会话用 `maintenance-plan` / `maintenance-apply` 补 `year` / `venue` / `rating`，但字段依据与 LIB 必须明确；只授权导入不等于授权维护所有字段。扫描命中已退库记录的路径或身份时会被阻止，改走明确授权的恢复流程。

## 停止条件与简报

来源在 plan 后变化、索引并发修改、身份警告、目标冲突、校验/store 失败或结果超出批准范围时立即停止。报告准确阶段、已复制目标、索引实际提交状态、残留文件与恢复前置条件；不自动回滚、不自动重试提交。

简报新 LIB、完整重复对应 LIB、少数冲突、来源清理去向及最终库数量。不改已有论文内容或人工编号；不下载或转换论文，不做科研精读，不推送 Git。
