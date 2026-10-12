---
name: paper-supplement
description: Add evidence-based notes to a specified paper in the external library. If the user explicitly asks to save the note, run the supported plan/apply workflow in the same session; otherwise produce a research note without writing the index.
description_zh: 对指定 LIB 论文做证据补充，默认只读阅读
---

# paper-supplement：按需提取，不强制每篇写库

入口：`D:\2Env\anaconda\python.exe D:\0Project\origin\_index\tools\paper_workflow.py`；规则见 `.codebuddy/rules/paper-library.mdc`。

## 必要上下文

- 本轮尚未读取 `doc/state/current.md` 时先读取一次，确认授权边界与恢复点；同一会话、同一任务范围已读取则复用，不重复通读。
- 字段约定不清楚时查共享 Rule 或 `D:\0Project\origin\_index\README.md` 的相关段落。

**默认阅读模式：** 从指定 LIB 定位正式 bundle，复用已有笔记，按研究问题只读方法、1—3 个关键公式、关键图表与近邻对照。主要输出是科研 Markdown 卡片：具体问题｜最小机制/公式｜关键实证｜最近工作差异｜可迁移性/创新边界。只标真正看过的章节与图表；论文事实、官方实现事实、项目推断分开。阅读本身不授权写索引，不为每篇论文机械生成长报告或重复 plan/apply。

## 用户明确要求保存时

1. `supplement-plan --id <LIB>` 定位唯一论文并取得当前 `F0`；确认已有补充和阅读状态。
2. 写简短追加草稿；运行 `supplement-plan --id <LIB> --draft <草稿.md> --expected-fingerprint <F0> --plan <plan.json> [--scope <范围>]`。只对实际打开并引用的本地证据按 CLI 要求传 `--evidence`；原笔记与人工编号不覆写。
3. 当前指令已明确授权保存时，同一会话计算真实计划文件 SHA-256 并运行 `supplement-apply --plan <plan.json> --approved-plan-sha256 <sha>`。遵循对应 `--help`；多条写入必须串行，每次使用最新 `F0`。遇到索引、草稿或证据冲突即停止，不强制覆盖。
4. 报告实际保存状态、更新后的指纹、阅读状态与验收摘要。若重复追加为 no-op，不保存、不增加 revision。

阅读状态默认保持不变；局部阅读不自动标为全文完成。只有用户明确指定且全文范围记录符合 CLI 要求时才调整为 `full_text_completed`（需使用 `--full-text-scope`）；未核实内容写“待核”，不虚构公式、未读资产清单或新目录。只在用户明确要求保存论文库笔记时使用 plan/apply。
